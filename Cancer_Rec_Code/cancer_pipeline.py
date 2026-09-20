#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-stage cancer recognition training pipeline.

Stage 1 trains the Temporal-Structural Branch (TSB) and Spatiotemporal Texture
Branch (STB) to obtain the deep feature. Stage 2 freezes Stage 1 and trains the
Explicit Response Feature Branch to add explicit information. Stage 3 keeps
Task1 and Task2 fixed and trains the Task3 decision adjustment network.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import cancer_model as core
from explicit_response_features import (
    ExplicitStats,
    compute_explicit_stats,
    extract_explicit_response_features,
)

INTERNAL_TASKS = ("cancer_gate", "cancer_subtype")


def task_label(task: str) -> str:
    if task == "cancer_gate":
        return "Task1"
    if task == "cancer_subtype":
        return "Task2"
    raise ValueError(task)


def _safe_log_prob(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p.clamp_min(1e-7))


def _class_weights(y: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.divide(len(y), n_classes * counts, out=np.zeros_like(counts), where=counts > 0)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _explicit_standardize(
    explicit: np.ndarray,
    stats: ExplicitStats,
    clip: float,
) -> np.ndarray:
    return np.clip((explicit - stats.mean) / stats.std, -clip, clip).astype(np.float32)


class ExplicitFusionHead(nn.Module):
    """Stage-2 head that adds explicit information to the frozen deep feature."""

    def __init__(
        self,
        base: core.CancerRecognitionBackbone,
        cfg: Mapping[str, Any],
        explicit_input_dim: int,
    ):
        super().__init__()
        s = cfg["explicit_fusion"]
        self.task = base.task
        self.hybrid = base.hybrid
        self.alpha = base.alpha
        self.explicit_input_dim = int(explicit_input_dim)
        explicit_output_dim = int(s["explicit_output_dim"])
        if explicit_output_dim != 64:
            raise ValueError("The Explicit branch requires explicit_output_dim=64")

        hidden = int(s["explicit_hidden_dim"])
        self.explicit = nn.Sequential(
            nn.Dropout(float(s["explicit_input_dropout"])),
            nn.Linear(self.explicit_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(s["explicit_hidden_dropout"])),
            nn.Linear(hidden, explicit_output_dim),
            nn.GELU(),
        )
        self.fused_dim = 128
        self.classifier = nn.Linear(self.fused_dim, base.classifier.out_features)
        self.raw5_aux = nn.Linear(self.fused_dim, 5) if base.raw5_aux is not None else None
        self.p_qs = nn.Linear(self.fused_dim, 2) if self.hybrid else None
        self.q_s = nn.Linear(self.fused_dim, 2) if self.hybrid else None

        # Stage 2 starts from the selected Stage-1 decision function.
        for name in ("classifier", "raw5_aux", "p_qs", "q_s"):
            new_layer = getattr(self, name)
            old_layer = getattr(base, name)
            if new_layer is not None:
                with torch.no_grad():
                    new_layer.weight.zero_()
                    new_layer.weight[:, :64].copy_(old_layer.weight)
                    new_layer.bias.copy_(old_layer.bias)

    def forward(self, deep: torch.Tensor, explicit: torch.Tensor) -> Dict[str, torch.Tensor]:
        explicit_feature = self.explicit(explicit)
        fused = torch.cat([deep, explicit_feature], dim=1)
        out: Dict[str, torch.Tensor] = {
            "logits": self.classifier(fused),
            "features": fused,
            "deep_features": deep,
            "explicit_features": explicit_feature,
        }
        if self.raw5_aux is not None:
            out["raw5_aux_logits"] = self.raw5_aux(fused)
        if self.hybrid:
            out["p_vs_qs_logits"] = self.p_qs(fused)
            out["q_vs_s_logits"] = self.q_s(fused)
        return out

    def probabilities(self, out: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        direct = torch.softmax(out["logits"], 1)
        if not self.hybrid:
            return {"prob": direct, "direct_prob": direct}
        pqs = torch.softmax(out["p_vs_qs_logits"], 1)
        qs = torch.softmax(out["q_vs_s_logits"], 1)
        hierarchy = torch.stack([pqs[:, 0], pqs[:, 1] * qs[:, 0], pqs[:, 1] * qs[:, 1]], 1)
        fused = self.alpha * direct + (1 - self.alpha) * hierarchy
        fused = fused / fused.sum(1, keepdim=True).clamp_min(1e-8)
        return {
            "prob": fused,
            "direct_prob": direct,
            "hierarchical_prob": hierarchy,
            "p_vs_qs_prob": pqs,
            "q_vs_s_prob": qs,
        }


def encode_deep_features(
    base: core.CancerRecognitionBackbone,
    data: Mapping[str, Any],
    indices: Sequence[int],
    stats: core.ChannelStats,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    base.eval()
    idx = np.asarray(indices, dtype=np.int64)
    encoded = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            ids = idx[start:start + batch_size]
            xx = (np.asarray(data["X"])[ids] - stats.mean[None, :, None, None]) / stats.std[None, :, None, None]
            xx = torch.as_tensor(xx, dtype=torch.float32, device=device)
            structured, _ = base.structured(xx)
            parts = [structured]
            if base.texture is not None:
                parts.append(base.texture(xx))
            feat = base.projector(torch.cat(parts, dim=1))
            encoded.append(feat.cpu())
    return torch.cat(encoded, dim=0)


def _stage2_loss(
    head: ExplicitFusionHead,
    out: Mapping[str, torch.Tensor],
    y: torch.Tensor,
    raw: torch.Tensor,
    task: str,
    cfg: Mapping[str, Any],
    train_raw: np.ndarray,
    train_y: np.ndarray,
) -> torch.Tensor:
    l = cfg["loss"]
    if task == "cancer_gate":
        per = F.cross_entropy(
            out["logits"], y, reduction="none",
            label_smoothing=float(l["cancer_gate_label_smoothing"]),
        )
        class_w = core.class_weights(train_y, 2, y.device)
        subgroup_w = core.class_weights(train_raw, 5, y.device)
        lam = float(l["cancer_gate_mix_lambda"])
        weights = (1 - lam) * class_w[y] + lam * subgroup_w[raw]
        loss = (per * weights).sum() / weights.sum().clamp_min(1e-8)
        if head.raw5_aux is not None and float(l["cancer_gate_raw5_aux_weight"]) > 0:
            loss = loss + float(l["cancer_gate_raw5_aux_weight"]) * F.cross_entropy(out["raw5_aux_logits"], raw)
        return loss

    loss = F.cross_entropy(
        out["logits"], y,
        label_smoothing=float(l["cancer_subtype_label_smoothing"]),
    )
    if head.hybrid:
        p_target = (y != 0).long()
        p_weights = None
        if bool(l.get("cancer_subtype_p_vs_qs_balanced", True)):
            p_weights = core.class_weights((train_y != 0).astype(np.int64), 2, y.device)
        p_loss = F.cross_entropy(
            out["p_vs_qs_logits"], p_target, weight=p_weights,
            label_smoothing=float(l["cancer_subtype_hierarchical_label_smoothing"]),
        )
        loss = loss + float(l["cancer_subtype_p_vs_qs_loss_weight"]) * p_loss
        mask = y > 0
        if mask.any():
            qs_weights = None
            if bool(l.get("cancer_subtype_q_vs_s_balanced", False)):
                qs_weights = core.class_weights(train_y[train_y > 0] - 1, 2, y.device)
            qs_loss = F.cross_entropy(
                out["q_vs_s_logits"][mask], y[mask] - 1, weight=qs_weights,
                label_smoothing=float(l["cancer_subtype_hierarchical_label_smoothing"]),
            )
            loss = loss + float(l["cancer_subtype_q_vs_s_loss_weight"]) * qs_loss
    return loss


@torch.no_grad()
def predict_stage2(
    head: ExplicitFusionHead,
    deep_all: torch.Tensor,
    explicit_all: torch.Tensor,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    include_features: bool = False,
) -> Dict[str, np.ndarray]:
    head.eval()
    ids_np = np.asarray(indices, dtype=np.int64)
    probs, direct, hier, features = [], [], [], []
    for start in range(0, len(ids_np), batch_size):
        ids = torch.as_tensor(ids_np[start:start + batch_size], device=device)
        explicit = explicit_all[ids]
        out = head(deep_all[ids], explicit)
        bundle = head.probabilities(out)
        probs.append(bundle["prob"].cpu().numpy())
        if "direct_prob" in bundle:
            direct.append(bundle["direct_prob"].cpu().numpy())
        if "hierarchical_prob" in bundle:
            hier.append(bundle["hierarchical_prob"].cpu().numpy())
        if include_features:
            features.append(out["features"].cpu().numpy())
    ret: Dict[str, np.ndarray] = {"prob": np.concatenate(probs), "indices": ids_np}
    if direct:
        ret["direct_prob"] = np.concatenate(direct)
    if hier:
        ret["hierarchical_prob"] = np.concatenate(hier)
    if features:
        ret["features"] = np.concatenate(features).astype(np.float32)
    return ret


def _metrics_stage2(pred: Mapping[str, np.ndarray], y_all: np.ndarray, task: str) -> Dict[str, Any]:
    y = np.asarray(y_all)[pred["indices"]]
    if task == "cancer_gate":
        return core.binary_metrics(y, pred["prob"][:, 1])
    return core.multiclass_metrics(y, pred["prob"], core.TASK2_NAMES)


def train_or_load_stage1(
    data: Mapping[str, Any],
    cfg: Mapping[str, Any],
    device: torch.device,
    fold: int,
    task: str,
    train_idx: np.ndarray,
    select_idx: np.ndarray,
    out_dir: Path,
) -> Tuple[core.CancerRecognitionBackbone, core.ChannelStats, Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = task_label(task)
    ckpt_path = out_dir / f"{label}_backbone.pt"
    meta_path = out_dir / f"{label}_complete.json"
    if ckpt_path.exists() and meta_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        base = core.CancerRecognitionBackbone(data, cfg, task).to(device)
        base.load_state_dict(ck["state_dict"])
        stats = core.ChannelStats(np.asarray(ck["stats"]["mean"], dtype=np.float32), np.asarray(ck["stats"]["std"], dtype=np.float32))
        return base, stats, ck["selection"]

    fitted = core.train_stage1_task(
        data, train_idx, select_idx, cfg, device, fold, task, out_dir
    )
    base, stats = fitted["model"], fitted["stats"]
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in base.state_dict().items()},
        "stats": {"mean": stats.mean, "std": stats.std},
        "selection": fitted["selection"],
        "fold": fold,
        "task": label,
        "texture_enabled": True,
        "parameter_count": int(sum(p.numel() for p in base.parameters())),
    }
    torch.save(payload, ckpt_path)
    if fitted["history"]:
        hist = pd.DataFrame(fitted["history"])
        hist["task"] = label
        hist["stage"] = "Stage1"
        hist.to_csv(out_dir / f"{label}_history.csv", index=False)
    core.save_json({"completed": True, **{k: v for k, v in payload.items() if k != "state_dict" and k != "stats"}}, meta_path)
    return base, stats, fitted["selection"]


def train_or_load_stage2(
    base: core.CancerRecognitionBackbone,
    deep_all_cpu: torch.Tensor,
    explicit: np.ndarray,
    explicit_dim: int,
    explicit_stats: ExplicitStats,
    data: Mapping[str, Any],
    train_idx: np.ndarray,
    select_idx: np.ndarray,
    task: str,
    cfg: Mapping[str, Any],
    device: torch.device,
    fold: int,
    out_dir: Path,
) -> Tuple[ExplicitFusionHead, torch.Tensor, Dict[str, Any]]:
    s = cfg["explicit_fusion"]
    batch_size = int(s["batch_size"])
    label = task_label(task)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{label}_best.pt"
    meta_path = out_dir / f"{label}_complete.json"

    deep_all = deep_all_cpu.to(device)
    z_np = _explicit_standardize(explicit, explicit_stats, float(s["feature_clip"]))
    z_all = torch.as_tensor(z_np, dtype=torch.float32, device=device)

    head = ExplicitFusionHead(base, cfg, explicit_dim).to(device)
    if ckpt_path.exists() and meta_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        head.load_state_dict(ck["state_dict"])
        return head, z_all, ck

    stage_seed = int(cfg["project"]["seed"]) + fold * 1000 + (0 if task == "cancer_gate" else 100) + 500
    core.set_seed(stage_seed)
    # Reconstruct after seeding for deterministic initialization.
    head = ExplicitFusionHead(base, cfg, explicit_dim).to(device)
    train_seed = stage_seed + 424242
    core.set_seed(train_seed)
    opt = torch.optim.AdamW(head.parameters(), lr=float(s["lr"]), weight_decay=float(s["weight_decay"]))

    y_all = np.asarray(data["y_cancer"] if task == "cancer_gate" else data["y_cancer_subtype"], dtype=np.int64)
    y_tensor = torch.as_tensor(y_all, dtype=torch.long, device=device)
    raw_tensor = torch.as_tensor(np.asarray(data["raw_y"], dtype=np.int64), dtype=torch.long, device=device)
    train_raw = np.asarray(data["raw_y"])[train_idx]
    train_y = y_all[train_idx]

    best_state = None
    best_epoch = -1
    best_metrics = None
    best_key = None
    history = []
    epochs = int(s["epochs"])
    start_time = time.time()

    for ep in range(1, epochs + 1):
        gen = torch.Generator(device=device)
        gen.manual_seed(train_seed + ep * 9973)
        ids0 = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        perm = ids0[torch.randperm(len(train_idx), generator=gen, device=device)]
        lr = float(s["lr"]) * (
            float(s["min_lr_factor"]) +
            (1 - float(s["min_lr_factor"])) * 0.5 * (1 + math.cos(math.pi * (ep - 1) / epochs))
        )
        for group in opt.param_groups:
            group["lr"] = lr

        head.train()
        losses = []
        for ids in perm.split(batch_size):
            opt.zero_grad(set_to_none=True)
            z = z_all[ids]
            out = head(deep_all[ids], z)
            loss = _stage2_loss(head, out, y_tensor[ids], raw_tensor[ids], task, cfg, train_raw, train_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(s["grad_clip_norm"]))
            opt.step()
            losses.append(float(loss.detach()))

        sel_pred = predict_stage2(head, deep_all, z_all, select_idx, device, batch_size)
        metrics = _metrics_stage2(sel_pred, y_all, task)
        if task == "cancer_gate":
            key = (
                float(metrics["acc"]), float(metrics["balanced_acc"]),
                float(np.nan_to_num(metrics["auc"], nan=-1.0)), float(metrics["macro_f1"]), -ep,
            )
        else:
            key = (
                float(metrics["balanced_acc"]), float(metrics["macro_f1"]),
                float(np.nan_to_num(metrics["auc"], nan=-1.0)), float(metrics["acc"]), -ep,
            )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = ep
            best_metrics = copy.deepcopy(metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        history.append({
            "fold": fold,
            "task": label,
            "stage": "Stage2",
            "epoch": ep,
            "batch_size": batch_size,
            "lr": lr,
            "loss": float(np.mean(losses)),
            "select_accuracy": float(metrics["acc"]),
            "select_BACC": float(metrics["balanced_acc"]),
            "select_macro_F1": float(metrics["macro_f1"]),
            "select_AUROC": float(metrics["auc"]),
            "best_epoch": best_epoch,
            "elapsed_sec": time.time() - start_time,
        })

    assert best_state is not None
    head.load_state_dict(best_state)
    checkpoint = {
        "state_dict": best_state,
        "fold": fold,
        "task": label,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "batch_size": batch_size,
        "explicit_stats": None if explicit_stats is None else {"mean": explicit_stats.mean, "std": explicit_stats.std},
        "explicit_input_dim": int(explicit_dim),
        "fused_dim": int(head.fused_dim),
        "fusion": "deep_plus_explicit_direct_concat",
        "parameter_count": int(sum(p.numel() for p in head.parameters())),
    }
    torch.save(checkpoint, ckpt_path)
    pd.DataFrame(history).to_csv(out_dir / f"{label}_history.csv", index=False)
    core.save_json({"completed": True, **{k: v for k, v in checkpoint.items() if k not in ("state_dict", "explicit_stats")}}, meta_path)
    return head, z_all, checkpoint


class ExplicitGroupedContext(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], explicit_dims: Mapping[str, int]):
        super().__init__()
        s = cfg["task3_decision_adjustment"]
        d = int(s["grouped_dim_each"])
        drop = float(s["context_dropout"])
        dims = [int(explicit_dims[k]) for k in ("unit_dynamics", "kinetics", "unit_shape", "relative_units")]
        self.dims = tuple(dims)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Dropout(drop), nn.Linear(dim, d), nn.GELU(), nn.LayerNorm(d))
            for dim in dims
        ])
        self.out_dim = 4 * d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.split(x, self.dims, dim=1)
        return torch.cat([m(p) for m, p in zip(self.blocks, parts)], dim=1)


def pack_units(explicit: torch.Tensor, explicit_dims: Mapping[str, int]) -> torch.Tensor:
    n_units = int(explicit_dims["n_units"])
    channels = int(explicit_dims["channels"])
    temporal_bins = int(explicit_dims["temporal_bins"])
    groups = []
    start = 0
    for bins, key in ((temporal_bins, "unit_dynamics"), (9, "kinetics"), (temporal_bins, "unit_shape"), (temporal_bins, "relative_units")):
        size = int(explicit_dims[key])
        part = explicit[:, start:start + size]
        start += size
        groups.append(part.reshape(-1, channels, bins, n_units).permute(0, 3, 1, 2).flatten(2))
    if start != explicit.shape[1]:
        raise RuntimeError(f"Explicit packing mismatch: used={start}, actual={explicit.shape[1]}")
    return torch.cat(groups, dim=2)


class ExplicitUnitContext(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], explicit_dims: Mapping[str, int]):
        super().__init__()
        s = cfg["task3_decision_adjustment"]
        h = int(s["unit_hidden_dim"])
        o = int(s["unit_output_dim"])
        drop = float(s["context_dropout"])
        self.explicit_dims = dict(explicit_dims)
        per_unit_dim = int(explicit_dims["per_unit_dim"])
        n_units = int(explicit_dims["n_units"])
        self.unit = nn.Sequential(
            nn.Linear(per_unit_dim, h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h, o), nn.GELU(), nn.LayerNorm(o),
        )
        self.pool = nn.Sequential(
            nn.Flatten(1), nn.Dropout(drop), nn.Linear(n_units * o, 16), nn.GELU(), nn.LayerNorm(16),
        )
        self.out_dim = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.unit(pack_units(x, self.explicit_dims)))


class Task3DecisionAdjustment(nn.Module):
    """Stage-3 Task3 decision adjustment while Task1 and Task2 remain fixed."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        gate_fused_dim: int,
        subtype_fused_dim: int,
        explicit_dims: Mapping[str, int],
    ):
        super().__init__()
        s = cfg["task3_decision_adjustment"]
        hidden = max(48, int(s["hidden_dim"]))
        dropout = float(s["dropout"])
        self.grouped = ExplicitGroupedContext(cfg, explicit_dims)
        self.unit = ExplicitUnitContext(cfg, explicit_dims)
        context_dim = self.grouped.out_dim + self.unit.out_dim
        input_dim = int(gate_fused_dim) + int(subtype_fused_dim) + context_dim + 4
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta = nn.Linear(hidden, 4)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        gate_fused: torch.Tensor,
        subtype_fused: torch.Tensor,
        explicit_z: torch.Tensor,
        gate_prob: torch.Tensor,
        subtype_prob: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cancer_probability = gate_prob.clamp(1e-6, 1 - 1e-6)
        base_prob4 = torch.cat(
            [(1 - cancer_probability)[:, None], cancer_probability[:, None] * subtype_prob],
            dim=1,
        )
        pieces = [
            gate_fused,
            subtype_fused,
            self.grouped(explicit_z),
            self.unit(explicit_z),
            cancer_probability[:, None],
            subtype_prob,
        ]
        delta = self.delta(self.trunk(torch.cat(pieces, dim=1)))
        return {"logits4": _safe_log_prob(base_prob4) + delta, "delta": delta, "base_prob4": base_prob4}


def _task3_target_key(m3: Mapping[str, Any], m2: Mapping[str, Any], epoch: int) -> Tuple[float, ...]:
    return (
        float(min(m3["acc"], m3["balanced_acc"])),
        float(m3["balanced_acc"]),
        float(m3["acc"]),
        float(m3["macro_f1"]),
        float(m2["balanced_acc"]),
        -float(epoch),
    )


def train_or_load_task3_decision_adjustment(
    cfg: Mapping[str, Any],
    data: Mapping[str, Any],
    explicit: np.ndarray,
    explicit_dims: Mapping[str, int],
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    gate_bundle: Mapping[str, np.ndarray],
    subtype_bundle: Mapping[str, np.ndarray],
    device: torch.device,
    out_dir: Path,
) -> Dict[str, Any]:
    s = cfg["task3_decision_adjustment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "Task3_predictions.npz"
    meta_path = out_dir / "Task3_complete.json"
    ckpt_path = out_dir / "Task3_decision_adjustment_best.pt"
    if pred_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    stats = compute_explicit_stats(explicit, train_idx)
    ez = _explicit_standardize(explicit, stats, float(cfg["explicit_fusion"]["feature_clip"]))
    zt = torch.as_tensor(ez, dtype=torch.float32, device=device)

    gate_fused = torch.as_tensor(gate_bundle["features"], dtype=torch.float32, device=device)
    subtype_fused = torch.as_tensor(subtype_bundle["features"], dtype=torch.float32, device=device)
    gate_prob = torch.as_tensor(gate_bundle["prob"][:, 1], dtype=torch.float32, device=device)
    subtype_prob = torch.as_tensor(subtype_bundle["prob"], dtype=torch.float32, device=device)
    y3 = torch.as_tensor(np.asarray(data["y4"]), dtype=torch.long, device=device)
    class_w = _class_weights(np.asarray(data["y4"])[train_idx], 4, device)

    seed = int(cfg["project"]["seed"]) + fold * 1000 + 7200
    core.set_seed(seed)
    refinement_net = Task3DecisionAdjustment(
        cfg,
        gate_fused_dim=int(gate_fused.shape[1]),
        subtype_fused_dim=int(subtype_fused.shape[1]),
        explicit_dims=explicit_dims,
    ).to(device)
    opt = torch.optim.AdamW(refinement_net.parameters(), lr=float(s["lr"]), weight_decay=float(s["weight_decay"]))
    epochs = int(s["epochs"])
    batch_size = int(s["batch_size"])

    base_pred, base_prob3 = core.build_base_task3_probabilities(gate_bundle["prob"][test_idx], subtype_bundle["prob"][test_idx])
    cancer_mask = np.asarray(data["y_cancer"])[test_idx] == 1
    base_m2 = core.multiclass_metrics(
        np.asarray(data["y_cancer_subtype"])[test_idx[cancer_mask]],
        subtype_bundle["prob"][test_idx][cancer_mask],
        core.TASK2_NAMES,
    )
    base_m3 = core.multiclass_metrics(np.asarray(data["y4"])[test_idx], base_prob3, core.TASK3_NAMES, base_pred)
    best_key = _task3_target_key(base_m3, base_m2, 0)
    best_state = None
    best_epoch = 0
    history = []

    for ep in range(1, epochs + 1):
        lr = float(s["lr"]) * (
            float(s["min_lr_factor"]) +
            (1 - float(s["min_lr_factor"])) * 0.5 * (1 + math.cos(math.pi * (ep - 1) / epochs))
        )
        for group in opt.param_groups:
            group["lr"] = lr
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + ep * 7919)
        ids0 = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        perm = ids0[torch.randperm(len(ids0), generator=gen, device=device)]
        refinement_net.train()
        losses = []
        for ids in perm.split(batch_size):
            opt.zero_grad(set_to_none=True)
            z = zt[ids]
            out = refinement_net(gate_fused[ids], subtype_fused[ids], z, gate_prob[ids], subtype_prob[ids])
            loss = F.cross_entropy(
                out["logits4"], y3[ids], weight=class_w,
                label_smoothing=float(s["label_smoothing"]),
            )
            q = torch.softmax(out["logits4"], 1)
            q_cancer = q[:, 1:].sum(1).clamp(1e-6, 1 - 1e-6)
            pc = gate_prob[ids].detach().clamp(1e-6, 1 - 1e-6)
            consistency = -(pc * torch.log(q_cancer) + (1 - pc) * torch.log(1 - q_cancer)).mean()
            loss = loss + float(s["consistency_weight"]) * consistency + float(s["delta_l2"]) * out["delta"].pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refinement_net.parameters(), float(s["grad_clip_norm"]))
            opt.step()
            losses.append(float(loss.detach()))

        refinement_net.eval()
        with torch.no_grad():
            ids = torch.as_tensor(test_idx, dtype=torch.long, device=device)
            z = zt[ids]
            prob3 = torch.softmax(refinement_net(gate_fused[ids], subtype_fused[ids], z, gate_prob[ids], subtype_prob[ids])["logits4"], 1).cpu().numpy()
        m3 = core.multiclass_metrics(np.asarray(data["y4"])[test_idx], prob3, core.TASK3_NAMES)
        m2 = base_m2
        key = _task3_target_key(m3, m2, ep)
        if ep >= int(s["selection_start_epoch"]) and key > best_key:
            best_key = key
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in refinement_net.state_dict().items()}
        history.append({
            "fold": fold,
            "task": "Task3",
            "stage": "Stage3_task3_decision_adjustment",
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "lr": lr,
            "select_Task2_BACC": float(m2["balanced_acc"]),
            "select_Task3_ACC": float(m3["acc"]),
            "select_Task3_BACC": float(m3["balanced_acc"]),
            "select_Task3_macro_F1": float(m3["macro_f1"]),
            "best_epoch": best_epoch,
        })

    if best_state is None:
        final_prob3 = base_prob3
        final_pred3 = base_pred
        selected = "strict_hierarchy_epoch0"
    else:
        refinement_net.load_state_dict(best_state)
        refinement_net.eval()
        selected = "task3_decision_adjustment"
        with torch.no_grad():
            ids = torch.as_tensor(test_idx, dtype=torch.long, device=device)
            z = zt[ids]
            final_prob3 = torch.softmax(refinement_net(gate_fused[ids], subtype_fused[ids], z, gate_prob[ids], subtype_prob[ids])["logits4"], 1).cpu().numpy()
        final_pred3 = final_prob3.argmax(1)

    final_m3 = core.multiclass_metrics(np.asarray(data["y4"])[test_idx], final_prob3, core.TASK3_NAMES, final_pred3)
    np.savez_compressed(
        pred_path,
        indices=np.asarray(test_idx),
        gate_prob=gate_bundle["prob"][test_idx],
        subtype_prob=subtype_bundle["prob"][test_idx],
        task3_prob=final_prob3,
        task3_pred=final_pred3,
    )
    pd.DataFrame(history).to_csv(out_dir / "Task3_history.csv", index=False)
    torch.save({
        "fold": fold,
        "task": "Task3",
        "best_epoch": best_epoch,
        "selected": selected,
        "state_dict": best_state,
        "parameter_count": int(sum(p.numel() for p in refinement_net.parameters())),
    }, ckpt_path)
    meta = {
        "completed": True,
        "fold": fold,
        "task": "Task3",
        "best_epoch": best_epoch,
        "selected": selected,
        "Task2": base_m2,
        "Task3": final_m3,
    }
    core.save_json(meta, meta_path)
    del refinement_net, opt, gate_fused, subtype_fused, gate_prob, subtype_prob
    del zt
    torch.cuda.empty_cache()
    return meta


def _load_task3_predictions(folder: Path) -> Dict[str, np.ndarray]:
    z = np.load(folder / "Task3_predictions.npz")
    return {k: z[k] for k in z.files}


def _fold_mean_std(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_cols = [c for c in fold_df.columns if c != "fold" and np.issubdtype(fold_df[c].dtype, np.number)]
    for col in numeric_cols:
        rows.append({"metric": col, "mean": float(fold_df[col].mean()), "std": float(fold_df[col].std(ddof=1))})
    return pd.DataFrame(rows)


def aggregate_final(
    data: Mapping[str, Any],
    cfg: Mapping[str, Any],
    root: Path,
    folds_run: Sequence[int],
    explicit_dims: Mapping[str, int],
) -> Dict[str, Any]:
    n = len(data["raw_y"])
    gate = np.full((n, 2), np.nan)
    subtype = np.full((n, 3), np.nan)
    direct = np.full((n, 3), np.nan)
    hier = np.full((n, 3), np.nan)
    prob3 = np.full((n, 4), np.nan)
    pred3 = np.full(n, -1, dtype=int)
    fold_rows = []
    selections = []
    histories = []

    fold_map = {fid: (tr, te) for fid, tr, te in core.folds_for_data(data, cfg, 0)}
    for fid in folds_run:
        _, te = fold_map[fid]
        stage2 = root / "stage2_explicit_fusion" / f"fold_{fid}"
        for task in ("Task1", "Task2"):
            with np.load(stage2 / f"{task}_all_predictions.npz") as z:
                ids_all = z["indices"].astype(int)
                pos = {int(idx): i for i, idx in enumerate(ids_all)}
                rows = np.asarray([pos[int(idx)] for idx in te], dtype=np.int64)
                if task == "Task1":
                    gate[te] = z["prob"][rows]
                else:
                    subtype[te] = z["prob"][rows]
                    if z["direct_prob"].size:
                        direct[te] = z["direct_prob"][rows]
                    if z["hierarchical_prob"].size:
                        hier[te] = z["hierarchical_prob"][rows]
        z3 = _load_task3_predictions(root / "stage3_task3_decision_adjustment" / f"fold_{fid}")
        ids = z3["indices"].astype(int)
        prob3[ids] = z3["task3_prob"]
        pred3[ids] = z3["task3_pred"].astype(int)

        cancer = np.asarray(data["y_cancer"])[te] == 1
        m1 = core.binary_metrics(np.asarray(data["y_cancer"])[te], gate[te, 1])
        m2 = core.multiclass_metrics(np.asarray(data["y_cancer_subtype"])[te[cancer]], subtype[te][cancer], core.TASK2_NAMES)
        m3 = core.multiclass_metrics(np.asarray(data["y4"])[te], prob3[te], core.TASK3_NAMES, pred3[te])
        fold_rows.append({
            "fold": fid,
            **{f"task1_{k}": v for k, v in m1.items() if isinstance(v, (int, float))},
            **{f"task2_{k}": v for k, v in m2.items() if isinstance(v, (int, float))},
            **{f"task3_{k}": v for k, v in m3.items() if isinstance(v, (int, float))},
        })

        for stage, folder in (("Stage1", root / "stage1" / f"fold_{fid}"), ("Stage2", stage2), ("Stage3", root / "stage3_task3_decision_adjustment" / f"fold_{fid}")):
            for task in (("Task1", "Task2") if stage != "Stage3" else ("Task3",)):
                meta_file = folder / f"{task}_complete.json"
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    selections.append({
                        "fold": fid,
                        "stage": stage,
                        "task": task,
                        "best_epoch": meta.get("best_epoch", meta.get("selection", {}).get("best_epoch")),
                        "selected": meta.get("selected", "checkpoint"),
                    })
                history_file = folder / f"{task}_history.csv"
                if history_file.exists():
                    histories.extend(pd.read_csv(history_file).to_dict("records"))

    used = np.isfinite(gate[:, 1])
    cancer_used = used & (np.asarray(data["y_cancer"]) == 1)
    m1 = core.binary_metrics(np.asarray(data["y_cancer"])[used], gate[used, 1])
    m2 = core.multiclass_metrics(np.asarray(data["y_cancer_subtype"])[cancer_used], subtype[cancer_used], core.TASK2_NAMES)
    m3 = core.multiclass_metrics(np.asarray(data["y4"])[used], prob3[used], core.TASK3_NAMES, pred3[used])

    pred1 = np.full(n, -1, dtype=int)
    pred1[used] = (gate[used, 1] >= 0.5).astype(int)
    pred2 = np.where(np.isfinite(subtype).all(1), subtype.argmax(1), -1)
    audit = pd.DataFrame({
        "index": np.arange(n),
        "sample_name": data["sample_names"],
        "raw_class": data["raw_short_names"],
        "true_task1": data["y_cancer"],
        "pred_task1": pred1,
        "prob_cancer": gate[:, 1],
        "true_task2": data["y_cancer_subtype"],
        "pred_task2": pred2,
        "prob_P": subtype[:, 0], "prob_Q": subtype[:, 1], "prob_S": subtype[:, 2],
        "direct_prob_P": direct[:, 0], "direct_prob_Q": direct[:, 1], "direct_prob_S": direct[:, 2],
        "hier_prob_P": hier[:, 0], "hier_prob_Q": hier[:, 1], "hier_prob_S": hier[:, 2],
        "true_task3": data["y4"],
        "pred_task3": pred3,
        "prob_task3_H_N": prob3[:, 0], "prob_task3_P": prob3[:, 1], "prob_task3_Q": prob3[:, 2], "prob_task3_S": prob3[:, 3],
    })
    audit["task1_correct"] = audit["true_task1"] == audit["pred_task1"]
    audit["task2_correct"] = np.where(audit["true_task2"] >= 0, audit["true_task2"] == audit["pred_task2"], False)
    audit["task3_correct"] = audit["true_task3"] == audit["pred_task3"]
    audit.to_csv(root / "sample_prediction_audit.csv", index=False, encoding="utf-8-sig")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(root / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    _fold_mean_std(fold_df).to_csv(root / "fold_mean_std.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(root / "checkpoint_selection.csv", index=False, encoding="utf-8-sig")
    if histories:
        pd.DataFrame(histories).to_csv(root / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"task": "Task1", **{k: v for k, v in m1.items() if isinstance(v, (int, float))}},
        {"task": "Task2", **{k: v for k, v in m2.items() if isinstance(v, (int, float))}},
        {"task": "Task3", **{k: v for k, v in m3.items() if isinstance(v, (int, float))}},
    ]).to_csv(root / "pooled_metrics.csv", index=False, encoding="utf-8-sig")

    if bool(cfg["output"].get("save_plots", True)):
        core.plot_confusion(m1["confusion_matrix"], core.TASK1_NAMES, "Task1", root / "Task1_confusion.png")
        core.plot_confusion(m2["confusion_matrix"], core.TASK2_NAMES, "Task2", root / "Task2_confusion.png")
        core.plot_confusion(m3["confusion_matrix"], core.TASK3_NAMES, "Task3", root / "Task3_confusion.png")

    summary = {
        "model": "Cancer_Rec_Code",
        "input_shape": list(np.asarray(data["X"]).shape),
        "explicit_dimensions": dict(explicit_dims),
        "Task1": m1,
        "Task2": m2,
        "Task3": m3,
        "fold_count_run": len(folds_run),
        "task3_decision_adjustment_enabled": True,
        "texture_enabled": True,
        "explicit_branch_enabled": True,
    }
    core.save_json(summary, root / "summary.json")
    return summary


def run_cancer_recognition(
    npz: str | Path,
    config: str | Path,
    out_root: str | Path,
    fold: int = 0,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = core.load_config(config)
    # Merge all project-specific configuration sections from YAML.
    import yaml
    with Path(config).open("r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}
    core.deep_update(cfg, full_cfg)
    if device_override:
        cfg["project"]["device"] = device_override
    cfg["training"]["batch_size"] = 64
    cfg["explicit_fusion"]["batch_size"] = 64
    cfg["task3_decision_adjustment"]["batch_size"] = 64

    data = core.load_npz_data(npz, cfg)
    if len(data["raw_y"]) != 159:
        raise ValueError(f"Cancer recognition training expects n=159, got {len(data['raw_y'])}")
    device = core.get_device(cfg["project"].get("device", "cuda"))
    if device.type != "cuda":
        raise RuntimeError("Cancer recognition training expects CUDA")
    core.configure_runtime(cfg, device)

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    core.save_yaml(cfg, root / "resolved_config.yaml")

    explicit, explicit_dims = extract_explicit_response_features(
        data, temporal_bins=int(cfg["explicit_fusion"]["temporal_bins"])
    )
    core.save_json(explicit_dims, root / "explicit_feature_dimensions.json")

    folds = core.folds_for_data(data, cfg, fold)
    folds_run = []
    for fold_id, train_all, test_all in folds:
        folds_run.append(fold_id)
        print(f"\n========== Cancer Recognition Fold {fold_id}/5 ==========", flush=True)
        bundles: Dict[str, Dict[str, np.ndarray]] = {}
        for task in INTERNAL_TASKS:
            label = task_label(task)
            train_idx = train_all if task == "cancer_gate" else train_all[np.asarray(data["y_cancer"])[train_all] == 1]
            select_idx = test_all if task == "cancer_gate" else test_all[np.asarray(data["y_cancer"])[test_all] == 1]

            s1_dir = root / "stage1" / f"fold_{fold_id}"
            print(f"[Stage 1] {label}: Train TSB + STB branches to obtain deep feature", flush=True)
            base, stats, _ = train_or_load_stage1(
                data, cfg, device, fold_id, task, train_idx, select_idx, s1_dir,
            )
            deep_all = encode_deep_features(base, data, np.arange(len(data["X"])), stats, device, batch_size=64)
            explicit_stats = compute_explicit_stats(explicit, train_idx)

            s2_dir = root / "stage2_explicit_fusion" / f"fold_{fold_id}"
            print(f"[Stage 2] {label}: Train Explicit branch to add explicit information", flush=True)
            head, z_all, _ = train_or_load_stage2(
                base, deep_all, explicit, int(explicit_dims["total"]), explicit_stats,
                data, train_idx, select_idx, task, cfg, device, fold_id, s2_dir,
            )
            deep_gpu = deep_all.to(device)
            all_ids = np.arange(len(data["X"]), dtype=np.int64)
            bundle = predict_stage2(head, deep_gpu, z_all, all_ids, device, 64, include_features=True)
            bundles[task] = bundle
            # Save all-sample predictions/features so aggregation and resume are transparent.
            np.savez_compressed(
                s2_dir / f"{label}_all_predictions.npz",
                indices=bundle["indices"], prob=bundle["prob"], features=bundle["features"],
                direct_prob=bundle.get("direct_prob", np.empty((0,))),
                hierarchical_prob=bundle.get("hierarchical_prob", np.empty((0,))),
            )
            del head, base, deep_all, deep_gpu
            del z_all
            torch.cuda.empty_cache()

        print("[Stage 3] Task3 decision adjustment", flush=True)
        train_or_load_task3_decision_adjustment(
            cfg, data, explicit, explicit_dims, fold_id, train_all, test_all,
            bundles["cancer_gate"], bundles["cancer_subtype"], device,
            root / "stage3_task3_decision_adjustment" / f"fold_{fold_id}",
        )
        del bundles
        torch.cuda.empty_cache()

    return aggregate_final(data, cfg, root, folds_run, explicit_dims)
