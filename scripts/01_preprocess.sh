#!/usr/bin/env bash
# Preprocessing only, expects existing plans.
# Usage: bash scripts/01_preprocess.sh [DATASET_NAME] [CONFIG] [NUM_PROCESSES]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_NAME="${1:-Dataset001_GroundTruth}"
CONFIG="${2:-3d_fullres}"
NUM_PROCESSES="${3:-8}"

if [[ "${DATASET_NAME}" =~ Dataset([0-9]+)_.* ]]; then
  DATASET_ID="${BASH_REMATCH[1]}"
else
  DATASET_ID="${DATASET_NAME}"
fi

echo "=== nnU-Net Preprocessing (using existing plans) ==="
echo "Dataset name : ${DATASET_NAME}"
echo "Dataset ID   : ${DATASET_ID}"
echo "Configuration: ${CONFIG}"
echo "Processes    : ${NUM_PROCESSES}"
echo

if [[ ! -f "${SCRIPT_DIR}/nnunet_env.sh" ]]; then
  echo "[ERROR] scripts/nnunet_env.sh not found."
  exit 1
fi

source "${SCRIPT_DIR}/nnunet_env.sh"
cd "${PROJECT_ROOT}"

# copy best-model plans if present
BEST_MODEL_PLANS="${PROJECT_ROOT}/hpo/best_model/nnUNetPlans.json"
TARGET_PLANS="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans.json"
if [[ -f "${BEST_MODEL_PLANS}" ]]; then
  echo "[INFO] Copying best-model plans from ${BEST_MODEL_PLANS} to ${TARGET_PLANS}"
  mkdir -p "$(dirname "${TARGET_PLANS}")"
  cp "${BEST_MODEL_PLANS}" "${TARGET_PLANS}"
else
  echo "[WARN] Best-model plans not found at ${BEST_MODEL_PLANS}; using existing plans."
fi

CMD=(nnUNetv2_preprocess
  -d "${DATASET_ID}"
  -c "${CONFIG}"
  -np "${NUM_PROCESSES}"
)
if [[ -n "${PLANS_NAME:-}" ]]; then
  CMD+=(-plans_name "${PLANS_NAME}")
fi

"${CMD[@]}"

echo
echo "Preprocessing finished. Outputs stored under \${nnUNet_preprocessed}."

