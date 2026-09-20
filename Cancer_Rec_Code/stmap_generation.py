# -*- coding: utf-8 -*-
"""Generate the Full STMap NPZ used by the cancer recognition model.

Supported channel modes:
  x_6      : R/G/B/Y/U/V six base channels.
  x_dx_12  : six base channels plus their first temporal derivatives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

RAW5_NAMES = ["P", "Q", "S", "N", "H"]
DEFAULT_COUNTS = {"P": 32, "Q": 25, "S": 24, "N": 30, "H": 48}
VALID_UNITS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15)

def set_seed(seed: int) -> None:
    np.random.seed(seed)

def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

BASE_CHANNEL_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("R", "RGB", "R"), ("G", "RGB", "G"), ("B", "RGB", "B"),
    ("Y", "YCrCb", "Y"), ("U", "YCrCb", "Cb"), ("V", "YCrCb", "Cr"),
)
DATA_SUFFIX = {
    "raw": "_01_raw_pixel_values.xlsx",
    "pixel_delta": "_02_pixel_delta_values.xlsx",
    "drift_corrected": "_03_drift_corrected_delta_values.xlsx",
}


def subroi_key(unit: int, sub: int) -> str:
    row = (sub - 1) // 3 + 1; col = (sub - 1) % 3 + 1
    return f"U{unit:02d}_r{row}c{col}"


def discover_samples(root: Path) -> List[Dict]:
    items: List[Dict] = []
    for label, cls in enumerate(RAW5_NAMES):
        class_dir = root / cls
        if not class_dir.exists():
            print(f"[WARN] Missing class directory: {class_dir}")
            continue
        for sample_dir in sorted(class_dir.iterdir()):
            if sample_dir.is_dir():
                items.append({"sample_name": sample_dir.name, "sample_dir": sample_dir, "raw_label": label, "raw_class": cls})
    return items


def find_workbook(sample_dir: Path, sample_name: str, kind: str) -> Path:
    suffix = DATA_SUFFIX[kind]; exact = sample_dir / f"{sample_name}{suffix}"
    if exact.exists(): return exact
    matches = sorted(sample_dir.glob(f"*{suffix}"))
    if matches: return matches[0]
    raise FileNotFoundError(f"No *{suffix} under {sample_dir}")


def smooth_time(x: np.ndarray, kernel: Sequence[float] = (1, 4, 6, 4, 1)) -> np.ndarray:
    k = np.asarray(kernel, dtype=np.float32); k /= k.sum(); r = len(k) // 2
    xp = np.pad(x, ((0, 0), (r, r), (0, 0)), mode="edge"); out = np.zeros_like(x, dtype=np.float32)
    for i, w in enumerate(k): out += float(w) * xp[:, i:i + x.shape[1], :]
    return out


def fit_time(x: np.ndarray, target: int = 512) -> np.ndarray:
    if x.shape[1] >= target: return x[:, :target].astype(np.float32, copy=False)
    pad = np.repeat(x[:, -1:, :], target - x.shape[1], axis=1)
    return np.concatenate([x, pad], axis=1).astype(np.float32, copy=False)


def fill_nan(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    for c in range(x.shape[0]):
        finite = np.isfinite(x[c]); fill = float(np.nanmedian(x[c][finite])) if finite.any() else 0.0; x[c][~finite] = fill
    return x


def add_channels(x6: np.ndarray, mode: str) -> Tuple[np.ndarray, List[str]]:
    names = [x[0] for x in BASE_CHANNEL_SPECS]
    if mode == "x_6": return x6.astype(np.float32, copy=False), names
    xs = smooth_time(x6); dx = np.zeros_like(xs)
    dx[:, 1:-1] = 0.5 * (xs[:, 2:] - xs[:, :-2]); dx[:, 0] = xs[:, 1] - xs[:, 0]; dx[:, -1] = xs[:, -1] - xs[:, -2]
    return np.concatenate([x6, dx], axis=0).astype(np.float32), names + [f"d{name}_dt" for name in names]


def build_sample(sample_dir: Path, sample_name: str, kind: str, channel_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    book = find_workbook(sample_dir, sample_name, kind); df = pd.read_excel(book, sheet_name="STMap_wide", engine="openpyxl")
    per_h: List[np.ndarray] = []; unit_ids: List[int] = []; sub_ids: List[int] = []; missing: List[str] = []
    for unit in VALID_UNITS:
        for sub in range(1, 10):
            key = subroi_key(unit, sub); cols = [f"{key}_{space}_{channel}" for _, space, channel in BASE_CHANNEL_SPECS]
            absent = [c for c in cols if c not in df.columns]
            if absent: missing.extend(absent); continue
            per_h.append(np.stack([df[c].to_numpy(dtype=np.float32) for c in cols], axis=0)); unit_ids.append(unit); sub_ids.append(sub)
    if missing: raise KeyError(f"Missing {len(missing)} columns; examples: {missing[:10]}")
    x6 = np.stack(per_h, axis=-1); x6 = fit_time(fill_nan(x6), 512); x, channel_names = add_channels(x6, channel_mode)
    meta = {"sample_name": sample_name, "source_excel": str(book), "original_t": len(df), "target_t": 512, "channel_mode": channel_mode, "input_data_kind": kind}
    return x, np.asarray(unit_ids), np.asarray(sub_ids), channel_names, meta


def save_npz(path: Path, X: np.ndarray, raw_y: np.ndarray, names: Sequence[str], raw_names: Sequence[str], unit_ids: np.ndarray, sub_ids: np.ndarray, channel_names: Sequence[str], channel_mode: str, kind: str, layout: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X=X.astype(np.float32), y=raw_y.astype(np.int64), raw_y=raw_y.astype(np.int64),
        sample_names=np.asarray(names, dtype=object), raw_short_names=np.asarray(raw_names, dtype=object), source_dirnames=np.asarray(raw_names, dtype=object),
        unit_ids=unit_ids.astype(np.int64), sub_ids=sub_ids.astype(np.int64), channel_names=np.asarray(channel_names, dtype=object),
        class_names=np.asarray(RAW5_NAMES, dtype=object), raw_class_names=np.asarray(RAW5_NAMES, dtype=object),
        channel_mode=np.asarray(channel_mode, dtype=object), input_data_kind=np.asarray(kind, dtype=object), target_t=np.asarray(512), layout_name=np.asarray(layout, dtype=object),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the Full STMap dataset")
    p.add_argument("--processed-root", default="./Cancer_Data/Ori-Data"); p.add_argument("--npz-root", default="./Cancer_Data/Pro-STMap")
    p.add_argument("--channel-mode", choices=["x_6", "x_dx_12"], default="x_dx_12")
    p.add_argument("--input-data-kind", choices=list(DATA_SUFFIX), default="drift_corrected")
    p.add_argument("--expected-counts", default="P=32,Q=25,S=24,N=30,H=48")
    p.add_argument("--allow-count-mismatch", action="store_true"); p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(); set_seed(args.seed)
    expected = {k: int(v) for k, v in (part.split("=") for part in args.expected_counts.split(","))}
    items = discover_samples(Path(args.processed_root)); actual = {c: sum(i["raw_class"] == c for i in items) for c in RAW5_NAMES}
    if actual != expected and not args.allow_count_mismatch: raise ValueError(f"Sample counts mismatch: expected={expected}, actual={actual}")
    if actual != expected: print(f"[WARN] Sample counts mismatch: expected={expected}, actual={actual}")
    xs = []; ys = []; names = []; raw_names = []; metadata = []; unit_ids = sub_ids = channel_names = None; failed = []
    for item in tqdm(items, desc="Generate Full STMap"):
        try:
            x, uid, sid, ch, meta = build_sample(item["sample_dir"], item["sample_name"], args.input_data_kind, args.channel_mode)
            xs.append(x); ys.append(item["raw_label"]); names.append(item["sample_name"]); raw_names.append(item["raw_class"]); metadata.append(meta); unit_ids, sub_ids, channel_names = uid, sid, ch
        except Exception as exc:
            failed.append({"sample_name": item["sample_name"], "raw_class": item["raw_class"], "error": repr(exc)}); print(f"[WARN] {item['sample_name']}: {exc}")
    if not xs: raise RuntimeError("No sample was generated")
    X = np.stack(xs); y = np.asarray(ys); root = Path(args.npz_root); full_path = root / "full" / "stmaps_full.npz"
    save_npz(full_path, X, y, names, raw_names, unit_ids, sub_ids, channel_names, args.channel_mode, args.input_data_kind, "full")
    pd.DataFrame(metadata).to_csv(root / "sample_generation_metadata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"layout": "full", "path": str(full_path), "shape": list(X.shape)}]).to_csv(
        root / "layout_manifest.csv", index=False, encoding="utf-8-sig"
    )
    if failed: pd.DataFrame(failed).to_csv(root / "failed_samples.csv", index=False, encoding="utf-8-sig")
    save_json({"channel_mode": args.channel_mode, "input_data_kind": args.input_data_kind, "counts": actual, "full_shape": list(X.shape), "valid_units": VALID_UNITS, "failed_count": len(failed)}, root / "generation_manifest.json")
    print(f"[DONE] {full_path}, shape={X.shape}")


if __name__ == "__main__": main()
