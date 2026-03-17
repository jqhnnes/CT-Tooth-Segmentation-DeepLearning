#!/usr/bin/env bash
# Run nnUNetv2_train and log GPU usage via nvidia-smi.
# Usage: bash scripts/02_training.sh DATASET_NAME CONFIG FOLD [TRIAL_NAME]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_NAME="${1:-Dataset001_GroundTruth}"
CONFIG="${2:-3d_fullres}"
FOLD="${3:-0}"

if [[ "${DATASET_NAME}" =~ Dataset([0-9]+)_.* ]]; then
  DATASET_ID="${BASH_REMATCH[1]}"
else
  DATASET_ID="${DATASET_NAME}"
fi

echo "=== nnU-Net Training ==="
echo "Dataset name : ${DATASET_NAME}"
echo "Dataset ID   : ${DATASET_ID}"
echo "Configuration: ${CONFIG}"
echo "Fold         : ${FOLD}"
echo

if [[ ! -f "${SCRIPT_DIR}/nnunet_env.sh" ]]; then
  echo "[ERROR] scripts/nnunet_env.sh not found."
  exit 1
fi

source "${SCRIPT_DIR}/nnunet_env.sh"
cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs/training_run"
mkdir -p "${LOG_DIR}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOG_DIR}/train_${DATE_TAG}.log"
SMI_LOG="${LOG_DIR}/gpu_${DATE_TAG}.csv"

echo "[INFO] Training log: ${TRAIN_LOG}"
echo "[INFO] GPU log     : ${SMI_LOG}"

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
           --format=csv -l 30 > "${SMI_LOG}" &
SMI_PID=$!

{
  nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" -tr nnUNetTrainer -p nnUNetPlans --npz
} |& tee "${TRAIN_LOG}"

kill "${SMI_PID}" >/dev/null 2>&1 || true

echo
echo "[DONE] Training finished. Logs stored in ${LOG_DIR}"

