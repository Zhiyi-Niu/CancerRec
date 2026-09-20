#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
FOLD="${FOLD:-0}"
CONFIG="${CONFIG:-config.yaml}"
SOURCE_ROOT="${SOURCE_ROOT:-./Cancer_Data/Ori-Data/}"
STMAP_ROOT="${STMAP_ROOT:-./Cancer_Data/Pro-STMap/}"
FULL_NPZ="${FULL_NPZ:-${STMAP_ROOT}/full/stmaps_full.npz}"
RESULT_ROOT="${RESULT_ROOT:-./results_cancer_recognition}"
LOG_ROOT="${LOG_ROOT:-./logs_cancer_recognition}"
SKIP_STMAP="${SKIP_STMAP:-0}"
EXPECTED_COUNTS="${EXPECTED_COUNTS:-P=32,Q=25,S=24,N=30,H=48}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

python verify_cancer_recognition.py 2>&1 | tee "${LOG_ROOT}/00_verify.log"

if [[ "${SKIP_STMAP}" != "1" ]]; then
  echo "[Data] Generate the Full STMap dataset"
  python -u stmap_generation.py \
    --processed-root "${SOURCE_ROOT}" \
    --npz-root "${STMAP_ROOT}" \
    --channel-mode x_dx_12 \
    --input-data-kind drift_corrected \
    --expected-counts "${EXPECTED_COUNTS}" \
    --seed 2026 \
    2>&1 | tee "${LOG_ROOT}/01_stmap_generation.log"
fi

echo "[Stage 1] Train TSB + STB branches to obtain deep feature"
echo "[Stage 2] Train Explicit branch to add explicit information"
echo "[Stage 3] Task3 decision adjustment"
python -u run_cancer_recognition.py \
  --config "${CONFIG}" \
  --npz "${FULL_NPZ}" \
  --out-dir "${RESULT_ROOT}" \
  --fold "${FOLD}" \
  --device cuda \
  2>&1 | tee "${LOG_ROOT}/02_cancer_recognition.log"

echo
printf 'Cancer recognition experiment completed.\nResults: %s\nLogs: %s\n' "${RESULT_ROOT}" "${LOG_ROOT}"
