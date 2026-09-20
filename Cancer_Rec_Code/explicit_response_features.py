#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit response feature extraction for Stage 2.

The four feature families are:
1) unit_dynamics
2) kinetics
3) unit_shape
4) relative_units
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


FEATURE_FAMILIES = ("unit_dynamics", "kinetics", "unit_shape", "relative_units")


def _unique_in_order(x: np.ndarray) -> List[int]:
    out: List[int] = []
    for value in np.asarray(x).reshape(-1).tolist():
        value = int(value)
        if value not in out:
            out.append(value)
    return out


def extract_explicit_features(
    data: Mapping[str, Any],
    temporal_bins: int = 16,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, np.ndarray]]:
    """Return concatenated explicit features, dimensions, and family blocks."""
    x = np.asarray(data["X"], dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected X=[N,C,T,H], got {x.shape}")
    n, c, t, h = x.shape
    if t % temporal_bins != 0:
        raise ValueError(f"T={t} must be divisible by temporal_bins={temporal_bins}")

    unit_ids = np.asarray(data["unit_ids"]).reshape(-1)
    if len(unit_ids) != h:
        raise ValueError(f"unit_ids length={len(unit_ids)} does not match H={h}")
    units = _unique_in_order(unit_ids)
    if not units:
        raise ValueError("No sensing units found")

    counts = [int(np.sum(unit_ids == u)) for u in units]
    if len(set(counts)) != 1:
        raise ValueError(f"Each sensing unit must have the same number of subROIs, got {counts}")
    rois_per_unit = counts[0]
    # STMap positions are stored as contiguous blocks for each sensing Unit.
    expected = np.repeat(np.asarray(units, dtype=unit_ids.dtype), rois_per_unit)
    if not np.array_equal(unit_ids, expected):
        raise ValueError("Expected contiguous equal-sized subROI blocks for each sensing unit")

    bin_size = t // temporal_bins
    roi_bins = x.reshape(n, c, temporal_bins, bin_size, h).mean(axis=3)
    unit_bins = roi_bins.reshape(n, c, temporal_bins, len(units), rois_per_unit).mean(axis=4)
    unit_x = x.reshape(n, c, t, len(units), rois_per_unit).mean(axis=4)

    # Kinetic descriptors summarize global level, spread, extrema, early/late response, and temporal variation.
    early = unit_x[:, :, :64].mean(axis=2)
    late = unit_x[:, :, -64:].mean(axis=2)
    stats = [
        unit_x.mean(axis=2),
        unit_x.std(axis=2),
        unit_x.min(axis=2),
        unit_x.max(axis=2),
        early,
        late,
        late - early,
        np.abs(unit_x).mean(axis=2),
        np.abs(np.diff(unit_x, axis=2)).mean(axis=2),
    ]
    kinetics = np.stack(stats, axis=2)

    # Successful ordering: temporal binning first, then shape normalization.
    centered = unit_bins - unit_bins.mean(axis=2, keepdims=True)
    shape_scale = np.maximum(np.sqrt((centered ** 2).mean(axis=2, keepdims=True)), eps)
    unit_shape = centered / shape_scale

    # Relative Units: center across units but normalize by RMS of ORIGINAL unit response.
    relative = unit_bins - unit_bins.mean(axis=3, keepdims=True)
    relative_scale = np.maximum(np.sqrt((unit_bins ** 2).mean(axis=(2, 3), keepdims=True)), eps)
    relative_units = relative / relative_scale

    groups: Dict[str, np.ndarray] = {
        "unit_dynamics": unit_bins.reshape(n, -1).astype(np.float32, copy=False),
        "kinetics": kinetics.reshape(n, -1).astype(np.float32, copy=False),
        "unit_shape": unit_shape.reshape(n, -1).astype(np.float32, copy=False),
        "relative_units": relative_units.reshape(n, -1).astype(np.float32, copy=False),
    }
    groups = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in groups.items()}
    feat = np.concatenate([groups[k] for k in FEATURE_FAMILIES], axis=1).astype(np.float32, copy=False)
    dims = {k: int(groups[k].shape[1]) for k in FEATURE_FAMILIES}
    dims.update(
        total=int(feat.shape[1]),
        n_units=int(len(units)),
        rois_per_unit=int(rois_per_unit),
        temporal_bins=int(temporal_bins),
        channels=int(c),
        per_unit_dim=int(c * (temporal_bins + 9 + temporal_bins + temporal_bins)),
    )
    return feat, dims, groups


def extract_explicit_response_features(
    data: Mapping[str, Any], temporal_bins: int = 16, eps: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, int]]:
    feat, dims, _ = extract_explicit_features(data, temporal_bins=temporal_bins, eps=eps)
    return feat, dims


@dataclass
class ExplicitStats:
    mean: np.ndarray
    std: np.ndarray


def compute_explicit_stats(features: np.ndarray, indices: Sequence[int]) -> ExplicitStats:
    idx = np.asarray(indices, dtype=np.int64)
    mean = features[idx].mean(axis=0).astype(np.float32)
    std = np.maximum(features[idx].std(axis=0).astype(np.float32), 1e-4)
    return ExplicitStats(mean=mean, std=std)
