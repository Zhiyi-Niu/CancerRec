# -*- coding: utf-8 -*-
"""Core model, training, metrics, and cross-validation utilities.

Stage 1 trains the Temporal-Structural Branch (TSB) and the Spatiotemporal
Texture Branch (STB) to obtain a 64-dimensional deep feature.
"""
from __future__ import annotations

import copy
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

RAW5_NAMES = ["P", "Q", "S", "N", "H"]
TASK1_NAMES = ["H_N", "P_Q_S"]
TASK2_NAMES = ["P", "Q", "S"]
TASK3_NAMES = ["H_N", "P", "Q", "S"]
VALID_UNITS: Tuple[int, ...] = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15)
DEFAULT_COUNTS = {"P": 32, "Q": 25, "S": 24, "N": 30, "H": 48}

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"name": "Cancer_Rec_Code", "seed": 2026, "device": "cuda"},
    "data": {
        "strict_expected_counts": True,
        "expected_counts": DEFAULT_COUNTS,
        "expected_time": 512,
        "allowed_channels": [6, 12],
    },
    "training": {
        "folds": 5,
        "batch_size": 64,
        "num_workers": 0,
        "pin_memory": True,
        "use_amp": True,
        "cpu_num_threads": 4,
        "grad_clip_norm": 2.0,
        "cancer_gate": {
            "epochs": 180, "selection_start_epoch": 10, "lr": 3e-4,
            "min_lr": 1e-6, "warmup_epochs": 5, "warmup_start_factor": 0.25,
            "weight_decay": 0.01,
        },
        "cancer_subtype": {
            "epochs": 170, "selection_start_epoch": 10, "lr": 3e-4,
            "min_lr": 1e-6, "warmup_epochs": 5, "warmup_start_factor": 0.25,
            "weight_decay": 0.01,
        },
        "augment": {"enable": False, "gaussian_noise_std": 0.008, "amplitude_scale_std": 0.025, "time_shift_max": 2},
    },
    "early_stopping": {
        "cancer_gate": {"enabled": True, "min_epochs": 30, "patience": 25, "min_delta": 0.002},
        "cancer_subtype": {"enabled": True, "min_epochs": 30, "patience": 30, "min_delta": 0.002},
    },
    "selection": {"smoothing_window": 5, "prediction_mode": "best_single"},
    "model": {
        "embed_dim": 48, "unit_depth": 1, "unit_heads": 2, "mlp_ratio": 1.5,
        "dropout": 0.10, "max_unit_id": 16, "temporal_kernel": 5,
        "temporal_dilations": [1, 2, 4, 8], "temporal_stem_kernels": [5, 9, 15],
        "texture_dim": 48, "texture_dropout": 0.10, "head_hidden_dim": 64,
        "cancer_gate_head_dropout": 0.10, "cancer_subtype_head_dropout": 0.25,
        "cancer_gate_unit_attention_mode": "learned_prior",
        "cancer_gate_unit_attention_temperature": 0.75,
        "cancer_subtype_unit_attention_mode": "original",
        "cancer_subtype_unit_attention_temperature": 1.0,
    },
    "loss": {
        "cancer_gate_strategy": "soft_subgroup_ce",
        "cancer_gate_mix_lambda": 0.25,
        "cancer_gate_label_smoothing": 0.05,
        "cancer_gate_raw5_aux_weight": 0.10,
        "cancer_subtype_strategy": "hybrid_hierarchical_ce",
        "cancer_subtype_label_smoothing": 0.05,
        "cancer_subtype_direct_fusion_weight": 0.60,
        "cancer_subtype_p_vs_qs_loss_weight": 0.30,
        "cancer_subtype_q_vs_s_loss_weight": 0.20,
        "cancer_subtype_hierarchical_label_smoothing": 0.02,
        "cancer_subtype_p_vs_qs_balanced": True,
        "cancer_subtype_q_vs_s_balanced": False,
    },
    "output": {"save_checkpoints": True, "save_plots": True},
    "progress": {"show_batch_progress": False, "print_every_epoch": True},
}

def deep_update(base: MutableMapping[str, Any], extra: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: Optional[str | Path]) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        deep_update(cfg, user_cfg)
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x: Any) -> Any:
        if isinstance(x, Mapping): return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [convert(v) for v in x]
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
        if isinstance(x, Path): return str(x)
        return x
    with path.open("w", encoding="utf-8") as f:
        json.dump(convert(obj), f, ensure_ascii=False, indent=2, allow_nan=True)


def save_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def configure_runtime(cfg: Mapping[str, Any], device: torch.device) -> None:
    torch.set_num_threads(int(cfg["training"].get("cpu_num_threads", 4)))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _normalize_raw_name(value: object) -> str:
    s = str(value).strip()
    aliases = {
        "P": "P", "P_BladderCancer": "P", "P_Bladder": "P",
        "Q": "Q", "Q_ProstateCancer": "Q", "Q_Prostate": "Q",
        "S": "S", "S_KidneyTumor": "S", "S_Kidney": "S",
        "N": "N", "N_NonNeoplastic": "N", "N_NonNeoplasticDisease": "N",
        "H": "H", "H_Healthy": "H", "H_HealthyControl": "H",
    }
    if s in aliases: return aliases[s]
    compact = re.sub(r"[ _]", "", s).lower()
    for key, val in aliases.items():
        if re.sub(r"[ _]", "", key).lower() == compact: return val
    raise KeyError(f"Cannot identify raw class name: {value!r}")


def load_npz_data(npz_path: str | Path, cfg: Mapping[str, Any], allow_count_mismatch: bool = False) -> Dict[str, Any]:
    z = np.load(npz_path, allow_pickle=True)
    data: Dict[str, Any] = {k: z[k] for k in z.files}
    if "X" not in data or "unit_ids" not in data:
        raise KeyError("NPZ must contain X and unit_ids")
    X = np.asarray(data["X"], dtype=np.float32)
    if X.ndim != 4:
        raise ValueError(f"X must be [N,C,T,H], got {X.shape}")
    if X.shape[1] not in set(int(v) for v in cfg["data"].get("allowed_channels", [6, 12])):
        raise ValueError(f"Unsupported channel count: {X.shape[1]}; expected x_6 or x_dx_12")
    if X.shape[2] != int(cfg["data"].get("expected_time", 512)):
        raise ValueError(f"Expected T=512, got {X.shape[2]}")

    n = X.shape[0]
    raw_names: Optional[np.ndarray] = None
    for key in ("raw_short_names", "source_dirnames"):
        if key in data and len(data[key]) == n:
            raw_names = np.asarray([_normalize_raw_name(x) for x in data[key]], dtype=object)
            break
    if raw_names is None:
        raw_y_old = np.asarray(data.get("raw_y", data.get("y")), dtype=np.int64)
        class_names = np.asarray(data.get("raw_class_names", data.get("class_names", RAW5_NAMES))).astype(str)
        if len(class_names) > int(raw_y_old.max()):
            raw_names = np.asarray([_normalize_raw_name(class_names[int(v)]) for v in raw_y_old], dtype=object)
        else:
            raw_names = np.asarray(RAW5_NAMES, dtype=object)[raw_y_old]

    raw_map = {name: i for i, name in enumerate(RAW5_NAMES)}
    raw_y = np.asarray([raw_map[str(x)] for x in raw_names], dtype=np.int64)
    y_cancer = (raw_y <= 2).astype(np.int64)
    y_sub = np.full(n, -1, dtype=np.int64)
    y_sub[raw_y <= 2] = raw_y[raw_y <= 2]
    y4 = np.zeros(n, dtype=np.int64)
    y4[raw_y == 0] = 1
    y4[raw_y == 1] = 2
    y4[raw_y == 2] = 3
    names = np.asarray(data.get("sample_names", [f"sample_{i:04d}" for i in range(n)])).astype(str)
    unit_ids = np.asarray(data["unit_ids"], dtype=np.int64)
    sub_ids = np.asarray(data.get("sub_ids", np.zeros_like(unit_ids)), dtype=np.int64)

    counts = {name: int((raw_names == name).sum()) for name in RAW5_NAMES}
    expected = {k: int(v) for k, v in cfg["data"].get("expected_counts", DEFAULT_COUNTS).items()}
    if bool(cfg["data"].get("strict_expected_counts", True)) and not allow_count_mismatch and counts != expected:
        raise ValueError(f"Sample counts mismatch: expected={expected}, actual={counts}")

    return {
        **data,
        "X": X, "raw_y": raw_y, "raw_short_names": raw_names,
        "y_cancer": y_cancer, "y_cancer_subtype": y_sub, "y4": y4,
        "sample_names": names, "unit_ids": unit_ids, "sub_ids": sub_ids,
        "class_counts": counts,
        "task3_counts": {"H_N": int((y4 == 0).sum()), "P": int((y4 == 1).sum()), "Q": int((y4 == 2).sum()), "S": int((y4 == 3).sum())},
    }


def slice_layout(data: Mapping[str, Any], unit_id: Optional[int]) -> Dict[str, Any]:
    if unit_id is None: return dict(data)
    unit_id = int(unit_id)
    if unit_id not in VALID_UNITS: raise ValueError(f"Invalid Unit {unit_id}; valid={VALID_UNITS}")
    mask = np.asarray(data["unit_ids"], dtype=np.int64) == unit_id
    if not np.any(mask): raise ValueError(f"Unit {unit_id} absent from NPZ")
    out = dict(data)
    out["X"] = np.asarray(data["X"])[:, :, :, mask]
    out["unit_ids"] = np.asarray(data["unit_ids"])[mask]
    out["sub_ids"] = np.asarray(data["sub_ids"])[mask]
    return out


@dataclass
class ChannelStats:
    mean: np.ndarray
    std: np.ndarray


def compute_channel_stats(X: np.ndarray, indices: Sequence[int]) -> ChannelStats:
    indices = np.asarray(indices, dtype=np.int64)
    c = X.shape[1]
    sums = np.zeros(c, dtype=np.float64)
    sqs = np.zeros(c, dtype=np.float64)
    count = 0
    for idx in indices:
        xi = np.asarray(X[int(idx)], dtype=np.float32)
        sums += xi.sum(axis=(1, 2), dtype=np.float64)
        sqs += np.square(xi, dtype=np.float32).sum(axis=(1, 2), dtype=np.float64)
        count += xi.shape[1] * xi.shape[2]
    mean = sums / max(count, 1)
    var = np.maximum(sqs / max(count, 1) - mean * mean, 1e-12)
    return ChannelStats(mean.astype(np.float32), np.sqrt(var).astype(np.float32))


class STMapDataset(Dataset):
    def __init__(self, data: Mapping[str, Any], indices: Sequence[int], stats: ChannelStats, cfg: Mapping[str, Any], training: bool):
        self.X = np.asarray(data["X"], dtype=np.float32)
        self.raw_y = np.asarray(data["raw_y"], dtype=np.int64)
        self.y_gate = np.asarray(data["y_cancer"], dtype=np.int64)
        self.y_sub = np.asarray(data["y_cancer_subtype"], dtype=np.int64)
        self.y4 = np.asarray(data["y4"], dtype=np.int64)
        self.names = np.asarray(data["sample_names"]).astype(str)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.stats = stats
        self.training = training
        self.aug = cfg["training"].get("augment", {})

    def __len__(self) -> int: return len(self.indices)

    def _shift(self, x: np.ndarray, shift: int) -> np.ndarray:
        if shift == 0: return x
        out = np.empty_like(x)
        if shift > 0:
            out[:, :shift] = x[:, :1]
            out[:, shift:] = x[:, :-shift]
        else:
            s = -shift
            out[:, -s:] = x[:, -1:]
            out[:, :-s] = x[:, s:]
        return out

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = int(self.indices[i])
        x = (self.X[idx] - self.stats.mean[:, None, None]) / self.stats.std[:, None, None]
        if self.training and self.aug.get("enable", False):
            x = x.copy()
            max_shift = int(self.aug.get("time_shift_max", 0))
            if max_shift: x = self._shift(x, int(np.random.randint(-max_shift, max_shift + 1)))
            scale_std = float(self.aug.get("amplitude_scale_std", 0.0))
            if scale_std: x *= np.random.normal(1.0, scale_std, (x.shape[0], 1, 1)).astype(np.float32)
            noise_std = float(self.aug.get("gaussian_noise_std", 0.0))
            if noise_std: x += np.random.normal(0.0, noise_std, x.shape).astype(np.float32)
        return {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "raw_y": torch.tensor(int(self.raw_y[idx])),
            "y_gate": torch.tensor(int(self.y_gate[idx])),
            "y_sub": torch.tensor(int(self.y_sub[idx])),
            "y4": torch.tensor(int(self.y4[idx])),
            "index": torch.tensor(idx),
            "sample_name": self.names[idx],
        }


def make_loader(data: Mapping[str, Any], indices: Sequence[int], stats: ChannelStats, cfg: Mapping[str, Any], training: bool) -> DataLoader:
    workers = int(cfg["training"].get("num_workers", 0))
    kwargs: Dict[str, Any] = {
        "dataset": STMapDataset(data, indices, stats, cfg, training),
        "batch_size": int(cfg["training"].get("batch_size", 64)),
        "shuffle": training, "num_workers": workers,
        "pin_memory": bool(cfg["training"].get("pin_memory", True)),
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


# ------------------------------- metrics ----------------------------------

def safe_div(a: float, b: float) -> float: return float(a / b) if b else 0.0


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1); out = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if np.any(m): out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def binary_metrics(y: Sequence[int], p: Sequence[float]) -> Dict[str, Any]:
    """Detailed Task1 metrics using cancer as the positive class."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rn = safe_div(tn, tn + fp)
    rp = safe_div(tp, tp + fn)
    pn = safe_div(tn, tn + fn)
    pp = safe_div(tp, tp + fp)
    f1n = safe_div(2 * pn * rn, pn + rn)
    f1p = safe_div(2 * pp * rp, pp + rp)
    try:
        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = float("nan")
    try:
        auprc = float(average_precision_score(y, p))
    except Exception:
        auprc = float("nan")
    pc = np.clip(p, 1e-8, 1 - 1e-8)
    return {
        "n": len(y),
        "acc": float((pred == y).mean()),
        "balanced_acc": (rn + rp) / 2,
        "macro_precision": (pn + pp) / 2,
        "macro_recall": (rn + rp) / 2,
        "macro_f1": (f1n + f1p) / 2,
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "auc": auc,
        "auprc": auprc,
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "negative_recall": rn,
        "positive_recall": rp,
        "negative_precision": pn,
        "positive_precision": pp,
        "negative_f1": f1n,
        "positive_f1": f1p,
        "specificity": rn,
        "sensitivity": rp,
        "log_loss": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
        "brier_score": float(np.square(p - y).mean()),
        "ece_10bin": ece(np.maximum(p, 1 - p), (pred == y).astype(float)),
        "confusion_matrix": cm.tolist(),
    }


def multiclass_metrics(
    y: Sequence[int],
    prob: np.ndarray,
    names: Sequence[str],
    pred: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Detailed Task2/Task3 metrics, including per-class metrics."""
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = prob.argmax(1) if pred is None else np.asarray(pred, dtype=np.int64)
    labels = np.arange(prob.shape[1])
    cm = confusion_matrix(y, pred, labels=labels)
    recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    precisions = precision_score(y, pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    try:
        auc = float(roc_auc_score(y, prob, labels=labels, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    try:
        onehot_y = np.eye(prob.shape[1], dtype=np.float64)[y]
        auprc = float(average_precision_score(onehot_y, prob, average="macro"))
    except Exception:
        auprc = float("nan")
    picked = np.clip(prob[np.arange(len(y)), y], 1e-8, 1.0)
    onehot = np.eye(prob.shape[1], dtype=np.float64)[y]
    out: Dict[str, Any] = {
        "n": len(y),
        "acc": float((pred == y).mean()),
        "balanced_acc": float(recalls.mean()),
        "macro_precision": float(precisions.mean()),
        "macro_recall": float(recalls.mean()),
        "macro_f1": float(f1s.mean()),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "auc": auc,
        "auprc": auprc,
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "log_loss": float(-np.log(picked).mean()),
        "brier_score": float(np.square(prob - onehot).sum(1).mean()),
        "ece_10bin": ece(prob.max(1), (pred == y).astype(float)),
        "confusion_matrix": cm.tolist(),
    }
    for i, name in enumerate(names):
        key = re.sub(r"[^A-Za-z0-9]+", "_", name)
        out[f"recall_{key}"] = float(recalls[i])
        out[f"precision_{key}"] = float(precisions[i])
        out[f"f1_{key}"] = float(f1s[i])
        out[f"support_{key}"] = int((y == i).sum())
    return out

def plot_confusion(cm: Sequence[Sequence[int]], labels: Sequence[str], title: str, path: Path) -> None:
    arr = np.asarray(cm); fig, ax = plt.subplots(figsize=(5.5, 5.0)); im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right"); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]): ax.text(j, i, str(arr[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


# --------------------------- proposed architecture -------------------------

class MultiScaleTemporalStem(nn.Module):
    def __init__(self, in_channels: int, dim: int, kernels: Sequence[int], dropout: float):
        super().__init__(); kernels = [int(k) | 1 for k in kernels]; branch = max(8, dim // len(kernels))
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels, branch, k, padding=k // 2, bias=False), nn.GroupNorm(1, branch), nn.GELU()) for k in kernels])
        self.proj = nn.Sequential(nn.Conv1d(branch * len(kernels), dim, 1, bias=False), nn.GroupNorm(1, dim), nn.GELU(), nn.Dropout(dropout))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.proj(torch.cat([b(x) for b in self.branches], 1))


class TemporalResidualBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float, ratio: float):
        super().__init__(); hidden = int(dim * ratio); pad = kernel // 2 * dilation
        self.dw = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation, groups=dim, bias=False)
        self.norm = nn.GroupNorm(1, dim)
        self.pw = nn.Sequential(nn.Conv1d(dim, hidden, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(hidden, dim, 1), nn.Dropout(dropout))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x + self.pw(self.norm(self.dw(x)))


class StageTemporalPooling(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__(); h = max(8, dim // 2)
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 1))
        self.proj = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = feat.transpose(1, 2); w = torch.softmax(self.score(tok).squeeze(-1), 1); attn = (tok * w.unsqueeze(-1)).sum(1)
        t = tok.shape[1]; seg = max(t // 3, 1); usable = min(seg * 3, t)
        if usable < 3: early = mid = late = tok.mean(1)
        else:
            st = tok[:, :usable].reshape(tok.shape[0], 3, seg, tok.shape[2]); early, mid, late = [st[:, i].mean(1) for i in range(3)]
        return self.proj(torch.cat([early, mid, late, attn], 1)), w


class TemporalCurveEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int, cfg: Mapping[str, Any], kernels: Sequence[int]):
        super().__init__(); m = cfg["model"]
        self.stem = MultiScaleTemporalStem(in_channels, dim, kernels, float(m["dropout"]))
        self.blocks = nn.Sequential(*[TemporalResidualBlock(dim, int(m["temporal_kernel"]), int(d), float(m["dropout"]), float(m["mlp_ratio"])) for d in m["temporal_dilations"]])
        self.pool = StageTemporalPooling(dim, float(m["dropout"]))
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, t, h = x.shape; z = x.permute(0, 3, 1, 2).reshape(b * h, c, t); token, tw = self.pool(self.blocks(self.stem(z)))
        return token.reshape(b, h, -1), tw.reshape(b, h, t)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, ratio: float):
        super().__init__(); hidden = int(dim * ratio); self.n1 = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True); self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.n1(x); y, _ = self.attn(y, y, y, need_weights=False); x = x + y; return x + self.mlp(self.n2(x))


class TemporalStructuralBranch(nn.Module):
    """Temporal-Structural Branch (TSB)."""

    def __init__(self, in_channels: int, unit_ids: np.ndarray, cfg: Mapping[str, Any], task: str):
        super().__init__()
        m = cfg["model"]
        dim = int(m["embed_dim"])
        self.unit_ids_np = np.asarray(unit_ids, dtype=np.int64)
        self.register_buffer("unit_ids", torch.as_tensor(unit_ids, dtype=torch.long), persistent=True)
        self.temporal = TemporalCurveEncoder(in_channels, dim, cfg, m["temporal_stem_kernels"])
        hidden = max(8, dim // 2)
        self.subroi_score = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(float(m["dropout"])), nn.Linear(hidden, 1)
        )
        self.unit_embedding = nn.Embedding(int(m["max_unit_id"]) + 1, dim)
        nn.init.normal_(self.unit_embedding.weight, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, int(m["unit_heads"]), float(m["dropout"]), float(m["mlp_ratio"]))
            for _ in range(int(m["unit_depth"]))
        ])
        self.unit_score = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(float(m["dropout"])), nn.Linear(hidden, 1)
        )
        self.prior = nn.Parameter(torch.zeros(int(m["max_unit_id"]) + 1))
        self.use_prior = task == "cancer_gate"
        self.temperature = float(
            m["cancer_gate_unit_attention_temperature"]
            if task == "cancer_gate" else m["cancer_subtype_unit_attention_temperature"]
        )
        self.output_dim = dim * 3

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, _, _, h = x.shape
        if h != self.unit_ids.numel():
            raise ValueError(f"Input H={h}, layout H={self.unit_ids.numel()}")
        htok, _ = self.temporal(x)
        unique = torch.unique(self.unit_ids, sorted=True)
        units = []
        h_to_u = torch.zeros(h, dtype=torch.long, device=x.device)
        for ui, unit in enumerate(unique):
            idx = torch.where(self.unit_ids == unit)[0]
            toks = htok[:, idx]
            if toks.shape[1] == 1:
                w = torch.ones(b, 1, device=x.device, dtype=x.dtype)
            else:
                w = torch.softmax(self.subroi_score(toks).squeeze(-1), 1)
            units.append((toks * w.unsqueeze(-1)).sum(1))
            h_to_u[idx] = ui
        utok = torch.stack(units, 1) + self.unit_embedding(unique).unsqueeze(0)
        tokens = torch.cat([self.cls.expand(b, -1, -1), utok], 1)
        tokens = self.transformer(tokens)
        cls, uout = tokens[:, 0], tokens[:, 1:]
        logits = self.unit_score(uout).squeeze(-1)
        if self.use_prior:
            logits = logits + self.prior[unique].unsqueeze(0)
        weights = torch.softmax(logits / max(self.temperature, 1e-3), 1)
        context = (uout * weights.unsqueeze(-1)).sum(1)
        mean = uout.mean(1)
        pooled = torch.cat([cls, context, mean], 1)
        return pooled, {
            "unit_weights": weights,
            "unique_units": unique,
            "h_to_unit": h_to_u,
            "unit_entropy": -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1),
        }


class SpatiotemporalTextureBranch(nn.Module):
    def __init__(self, in_channels: int, dim: int, dropout: float):
        super().__init__(); self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (7, 3), padding=(3, 1), groups=in_channels, bias=False), nn.Conv2d(in_channels, dim, 1, bias=False), nn.GroupNorm(1, dim), nn.GELU(), nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(dim, dim, (5, 3), padding=(2, 1), groups=dim, bias=False), nn.Conv2d(dim, dim, 1, bias=False), nn.GroupNorm(1, dim), nn.GELU(), nn.Dropout2d(dropout * 0.5), nn.AdaptiveAvgPool2d(1))
        self.norm = nn.LayerNorm(dim); self.output_dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.norm(self.net(x).flatten(1))


class CancerRecognitionBackbone(nn.Module):
    """Stage-1 network that combines TSB and STB into a 64-D deep feature."""

    def __init__(self, data: Mapping[str, Any], cfg: Mapping[str, Any], task: str):
        super().__init__()
        self.task = task
        m = cfg["model"]
        l = cfg["loss"]
        channels = data["X"].shape[1]
        self.structured = TemporalStructuralBranch(channels, np.asarray(data["unit_ids"]), cfg, task)
        self.texture = SpatiotemporalTextureBranch(channels, int(m["texture_dim"]), float(m["texture_dropout"]))
        fused_dim = self.structured.output_dim + self.texture.output_dim
        hidden = int(m["head_hidden_dim"])
        dropout = float(m["cancer_gate_head_dropout"] if task == "cancer_gate" else m["cancer_subtype_head_dropout"])
        self.projector = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(hidden)
        )
        self.classifier = nn.Linear(hidden, 2 if task == "cancer_gate" else 3)
        self.raw5_aux_weight = float(l["cancer_gate_raw5_aux_weight"]) if task == "cancer_gate" else 0.0
        self.raw5_aux = nn.Linear(hidden, 5) if self.raw5_aux_weight > 0 else None
        self.hybrid = task == "cancer_subtype"
        self.alpha = float(l["cancer_subtype_direct_fusion_weight"])
        self.p_qs = nn.Linear(hidden, 2) if self.hybrid else None
        self.q_s = nn.Linear(hidden, 2) if self.hybrid else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        structured_feature, attention = self.structured(x)
        texture_feature = self.texture(x)
        feature = self.projector(torch.cat([structured_feature, texture_feature], 1))
        out = {"logits": self.classifier(feature), "features": feature, **attention}
        if self.raw5_aux is not None:
            out["raw5_aux_logits"] = self.raw5_aux(feature)
        if self.hybrid:
            out["p_vs_qs_logits"] = self.p_qs(feature)
            out["q_vs_s_logits"] = self.q_s(feature)
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


def class_weights(y: np.ndarray, classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=classes).astype(float)
    if np.any(counts == 0): raise ValueError(f"Training fold missing class: {counts.tolist()}")
    return torch.tensor(len(y) / (classes * counts), dtype=torch.float32, device=device)


def cosine_lr(epoch: int, total: int, base: float, minimum: float, warmup: int, factor: float) -> float:
    if epoch <= warmup: return base * (factor + (1 - factor) * epoch / max(warmup, 1))
    p = (epoch - warmup) / max(total - warmup, 1); return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * p))


def compute_stage1_loss(
    model: CancerRecognitionBackbone,
    out: Mapping[str, torch.Tensor],
    y: torch.Tensor,
    raw_y: torch.Tensor,
    task: str,
    cfg: Mapping[str, Any],
    train_raw: np.ndarray,
    train_y: np.ndarray,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    l = cfg["loss"]
    if task == "cancer_gate":
        per = F.cross_entropy(
            out["logits"], y, reduction="none",
            label_smoothing=float(l["cancer_gate_label_smoothing"]),
        )
        class_w = class_weights(train_y, 2, y.device)
        group_w = class_weights(train_raw, 5, y.device)
        mix = float(l["cancer_gate_mix_lambda"])
        sample_w = (1 - mix) * class_w[y] + mix * group_w[raw_y]
        ce = (per * sample_w).sum() / sample_w.sum().clamp_min(1e-8)
        total = ce
        raw_aux = out["logits"].new_zeros(())
        if model.raw5_aux_weight > 0 and "raw5_aux_logits" in out:
            raw_aux = F.cross_entropy(out["raw5_aux_logits"], raw_y)
            total = total + model.raw5_aux_weight * raw_aux
        return total, {
            "ce": float(ce.detach()),
            "raw5_aux": float(raw_aux.detach()),
            "p_vs_qs": 0.0,
            "q_vs_s": 0.0,
        }

    ce = F.cross_entropy(
        out["logits"], y,
        label_smoothing=float(l["cancer_subtype_label_smoothing"]),
    )
    p_target = (y != 0).long()
    p_weights = class_weights((train_y != 0).astype(np.int64), 2, y.device) if l.get("cancer_subtype_p_vs_qs_balanced", True) else None
    p_loss = F.cross_entropy(
        out["p_vs_qs_logits"], p_target, weight=p_weights,
        label_smoothing=float(l["cancer_subtype_hierarchical_label_smoothing"]),
    )
    q_s_loss = out["logits"].new_zeros(())
    mask = y > 0
    if mask.any():
        q_s_weights = class_weights(train_y[train_y > 0] - 1, 2, y.device) if l.get("cancer_subtype_q_vs_s_balanced", False) else None
        q_s_loss = F.cross_entropy(
            out["q_vs_s_logits"][mask], y[mask] - 1, weight=q_s_weights,
            label_smoothing=float(l["cancer_subtype_hierarchical_label_smoothing"]),
        )
    total = ce + float(l["cancer_subtype_p_vs_qs_loss_weight"]) * p_loss + float(l["cancer_subtype_q_vs_s_loss_weight"]) * q_s_loss
    return total, {
        "ce": float(ce.detach()),
        "raw5_aux": 0.0,
        "p_vs_qs": float(p_loss.detach()),
        "q_vs_s": float(q_s_loss.detach()),
    }


def evaluate_stage1(model: nn.Module, loader: DataLoader, task: str, device: torch.device) -> Dict[str, Any]:
    model.eval(); probs = []; ys = []; idxs = []; extras: Dict[str, List[np.ndarray]] = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True); out = model(x); bundle = model.probabilities(out)
            probs.append(bundle["prob"].cpu().numpy()); ys.append((batch["y_gate"] if task == "cancer_gate" else batch["y_sub"]).numpy()); idxs.append(batch["index"].numpy())
            for k, v in bundle.items():
                if k != "prob": extras.setdefault(k, []).append(v.cpu().numpy())
    prob = np.concatenate(probs); y = np.concatenate(ys); indices = np.concatenate(idxs)
    metrics = binary_metrics(y, prob[:, 1]) if task == "cancer_gate" else multiclass_metrics(y, prob, TASK2_NAMES)
    return {"prob": prob, "y": y, "indices": indices, "metrics": metrics, **{k: np.concatenate(v) for k, v in extras.items()}}


def train_stage1_task(data: Mapping[str, Any], train_idx: np.ndarray, val_idx: np.ndarray, cfg: Mapping[str, Any], device: torch.device, fold_id: int, task: str, out_dir: Path) -> Dict[str, Any]:
    task_cfg = cfg["training"][task]; es = cfg["early_stopping"][task]; sel = cfg["selection"]
    seed = int(cfg["project"]["seed"]) + fold_id * 1000 + (0 if task == "cancer_gate" else 100)
    set_seed(seed); stats = compute_channel_stats(np.asarray(data["X"]), train_idx)
    tr_loader = make_loader(data, train_idx, stats, cfg, True); va_loader = make_loader(data, val_idx, stats, cfg, False)
    model = CancerRecognitionBackbone(data, cfg, task).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(task_cfg["lr"]), weight_decay=float(task_cfg["weight_decay"]))
    amp = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"; scaler = torch.cuda.amp.GradScaler(enabled=amp)
    train_y_all = np.asarray(data["y_cancer"] if task == "cancer_gate" else data["y_cancer_subtype"])[train_idx]
    train_raw_all = np.asarray(data["raw_y"])[train_idx]
    best_state = None; best_epoch = -1; best_smooth = -np.inf; best_key = None; early_best = -np.inf; scores: List[float] = []; wait = 0; history = []
    max_epochs = int(task_cfg["epochs"]); selection_start = int(task_cfg["selection_start_epoch"]); window = int(sel["smoothing_window"])
    pbar = tqdm(range(1, max_epochs + 1), disable=not bool(cfg["progress"].get("print_every_epoch", True)), desc=f"fold{fold_id}-stage1-{task}")
    for epoch in pbar:
        lr = cosine_lr(epoch, max_epochs, float(task_cfg["lr"]), float(task_cfg["min_lr"]), int(task_cfg["warmup_epochs"]), float(task_cfg["warmup_start_factor"])); [g.update(lr=lr) for g in opt.param_groups]
        model.train(); losses = []; train_probs = []; train_labels = []; comp = {"ce": [], "raw5_aux": [], "p_vs_qs": [], "q_vs_s": []}
        for batch in tr_loader:
            x = batch["x"].to(device, non_blocking=True); y = (batch["y_gate"] if task == "cancer_gate" else batch["y_sub"]).to(device); raw = batch["raw_y"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(x); loss, parts = compute_stage1_loss(model, out, y, raw, task, cfg, train_raw_all, train_y_all)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip_norm", 2.0))); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach())); [comp[k].append(v) for k, v in parts.items()]
            with torch.no_grad(): train_probs.append(model.probabilities(out)["prob"].detach().cpu().numpy()); train_labels.append(y.detach().cpu().numpy())
        tp = np.concatenate(train_probs); ty = np.concatenate(train_labels); trm = binary_metrics(ty, tp[:, 1]) if task == "cancer_gate" else multiclass_metrics(ty, tp, TASK2_NAMES)
        val = evaluate_stage1(model, va_loader, task, device); vm = val["metrics"]
        smooth = np.nan
        if epoch >= selection_start:
            scores.append(float(vm["balanced_acc"])); smooth = float(np.mean(scores[-window:]))
            if task == "cancer_gate":
                candidate_key = (smooth, float(vm["balanced_acc"]), float(np.nan_to_num(vm.get("auc", np.nan), nan=-1.0)), float(vm.get("macro_precision", 0.0)), -float(vm.get("log_loss", 1e9)), -epoch)
            else:
                candidate_key = (smooth, float(vm["balanced_acc"]), float(vm.get("macro_f1", 0.0)), float(vm.get("macro_precision", 0.0)), float(np.nan_to_num(vm.get("auc", np.nan), nan=-1.0)), -float(vm.get("log_loss", 1e9)), -epoch)
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key; best_smooth = smooth; best_epoch = epoch; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if smooth > early_best + float(es["min_delta"]):
                early_best = smooth; wait = 0
            else:
                wait += 1
        row = {"fold": fold_id, "task": task, "epoch": epoch, "lr": lr, "train_loss": np.mean(losses), "train_bacc": trm["balanced_acc"], "val_bacc": vm["balanced_acc"], "val_acc": vm["acc"], "val_auc": vm["auc"], "selection_smooth_bacc": smooth, "best_epoch": best_epoch, "wait": wait, **{f"train_{k}": float(np.mean(v)) if v else 0.0 for k, v in comp.items()}}
        history.append(row); pbar.set_postfix(va=f"{vm['balanced_acc']:.3f}", best=best_epoch, wait=wait)
        if bool(es["enabled"]) and epoch >= int(es["min_epochs"]) and wait >= int(es["patience"]): break
    if best_state is None: best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_epoch = history[-1]["epoch"]
    model.load_state_dict(best_state); final = evaluate_stage1(model, va_loader, task, device)
    if cfg["output"].get("save_checkpoints", True):
        torch.save({"state_dict": best_state, "epoch": best_epoch, "stats": {"mean": stats.mean, "std": stats.std}, "task": task}, out_dir / f"fold_{fold_id}_{task}_best.pt")
    return {"model": model, "evaluation": final, "history": history, "stats": stats, "selection": {"fold": fold_id, "task": task, "best_epoch": best_epoch, "best_smooth_bacc": best_smooth, "parameter_count": int(sum(p.numel() for p in model.parameters()))}}


def build_base_task3_probabilities(gate_prob: np.ndarray, subtype_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p_cancer = gate_prob[:, 1]; pred = np.zeros(len(p_cancer), dtype=np.int64); cancer = p_cancer >= 0.5; pred[cancer] = subtype_prob[cancer].argmax(1) + 1
    prob4 = np.column_stack([1 - p_cancer, p_cancer[:, None] * subtype_prob]); prob4 /= prob4.sum(1, keepdims=True).clip(1e-8)
    return pred, prob4


def folds_for_data(data: Mapping[str, Any], cfg: Mapping[str, Any], fold: int) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=int(cfg["training"]["folds"]), shuffle=True, random_state=int(cfg["project"]["seed"])); pairs = list(skf.split(np.zeros(len(data["raw_y"])), data["raw_y"]))
    if fold == 0: return [(i + 1, tr, va) for i, (tr, va) in enumerate(pairs)]
    if not 1 <= fold <= len(pairs): raise ValueError(f"fold must be 0 or 1..{len(pairs)}")
    tr, va = pairs[fold - 1]; return [(fold, tr, va)]
