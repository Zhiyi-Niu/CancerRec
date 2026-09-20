#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast structural verification for the cancer recognition project."""
from __future__ import annotations

import numpy as np
import torch
import yaml

import cancer_model as core
from cancer_pipeline import ExplicitFusionHead, Task3DecisionAdjustment
from explicit_response_features import extract_explicit_response_features


def build_synthetic_data() -> dict:
    units = np.array(core.VALID_UNITS, dtype=np.int64)
    unit_ids = np.repeat(units, 9)
    sub_ids = np.tile(np.arange(1, 10, dtype=np.int64), len(units))
    x = np.random.default_rng(2026).normal(size=(4, 12, 512, len(unit_ids))).astype(np.float32)
    return {
        "X": x,
        "unit_ids": unit_ids,
        "sub_ids": sub_ids,
        "raw_y": np.array([0, 1, 2, 3], dtype=np.int64),
        "y_cancer": np.array([1, 1, 1, 0], dtype=np.int64),
        "y_cancer_subtype": np.array([0, 1, 2, -1], dtype=np.int64),
        "y4": np.array([1, 2, 3, 0], dtype=np.int64),
        "sample_names": np.array(["P_demo", "Q_demo", "S_demo", "N_demo"], dtype=object),
        "raw_short_names": np.array(["P", "Q", "S", "N"], dtype=object),
    }


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = build_synthetic_data()
    explicit, dims = extract_explicit_response_features(data, temporal_bins=int(cfg["explicit_fusion"]["temporal_bins"]))
    if explicit.shape[1] != 8892:
        raise AssertionError(f"Expected 8892 explicit features, got {explicit.shape[1]}")

    for task in ("cancer_gate", "cancer_subtype"):
        backbone = core.CancerRecognitionBackbone(data, cfg, task).eval()
        with torch.no_grad():
            out = backbone(torch.as_tensor(data["X"][:2]))
        if out["features"].shape != (2, 64):
            raise AssertionError(f"Stage-1 deep feature has wrong shape: {out['features'].shape}")
        head = ExplicitFusionHead(backbone, cfg, dims["total"]).eval()
        with torch.no_grad():
            fused = head(out["features"], torch.as_tensor(explicit[:2]))
        if fused["features"].shape != (2, 128):
            raise AssertionError(f"Stage-2 joint feature has wrong shape: {fused['features'].shape}")

    refinement = Task3DecisionAdjustment(cfg, 128, 128, dims).eval()
    cancer_prob = torch.tensor([0.4, 0.7], dtype=torch.float32)
    subtype_prob = torch.softmax(torch.randn(2, 3), dim=1)
    with torch.no_grad():
        refined = refinement(
            torch.randn(2, 128),
            torch.randn(2, 128),
            torch.as_tensor(explicit[:2]),
            cancer_prob,
            subtype_prob,
        )
    if refined["logits4"].shape != (2, 4):
        raise AssertionError("Stage-3 Task3 decision adjustment output has wrong shape")

    print("[OK] Stage 1: TSB + STB -> 64-D deep feature")
    print("[OK] Stage 2: Explicit branch -> 128-D deep + explicit feature")
    print("[OK] Stage 3: Task3 decision adjustment -> 4-class logits")
    print("[OK] Structural verification passed")


if __name__ == "__main__":
    main()
