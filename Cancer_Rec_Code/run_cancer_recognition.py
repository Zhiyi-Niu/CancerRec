#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

from cancer_pipeline import run_cancer_recognition


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete cancer recognition experiment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--npz",
        default="./Cancer_Data/Pro-STMap/full/stmaps_full.npz",
    )
    parser.add_argument("--out-dir", default="./results_cancer_recognition")
    parser.add_argument("--fold", type=int, default=0, help="0 runs all five folds; 1-5 runs one fold")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_cancer_recognition(
        args.npz,
        args.config,
        args.out_dir,
        fold=args.fold,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
