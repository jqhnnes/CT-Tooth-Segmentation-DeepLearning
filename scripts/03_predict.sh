#!/usr/bin/env bash
# Run nnUNetv2_predict for all trained folds on a common input dataset.
# Usage: bash scripts/03_predict.sh DATASET_NAME CONFIG INPUT_DIR OUTPUT_ROOT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_NAME="${1:-Dataset001_GroundTruth}"
CONFIG="${2:-3d_fullres}"
INPUT_DIR="${3:-${nnUNet_raw:-}/Dataset002_Karies/imagesTr}"
OUTPUT_ROOT="${4:-${PROJECT_ROOT}/predictions}"

if [[ "${DATASET_NAME}" =~ Dataset([0-9]+)_.* ]]; then
  DATASET_ID="${BASH_REMATCH[1]}"
else
  DATASET_ID="${DATASET_NAME}"
fi

echo "=== nnU-Net Multi-Fold Prediction ==="
echo "Dataset name : ${DATASET_NAME}"
echo "Dataset ID   : ${DATASET_ID}"
echo "Configuration: ${CONFIG}"
echo "Input dir    : ${INPUT_DIR}"
echo "Output root  : ${OUTPUT_ROOT}"
echo

if [[ ! -f "${SCRIPT_DIR}/nnunet_env.sh" ]]; then
  echo "[ERROR] scripts/nnunet_env.sh not found."
  exit 1
fi

source "${SCRIPT_DIR}/nnunet_env.sh"
cd "${PROJECT_ROOT}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[ERROR] Input directory ${INPUT_DIR} not found."
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

FOLDS=(0 1 2 3 4)
for FOLD in "${FOLDS[@]}"; do
  OUT_DIR="${OUTPUT_ROOT}/fold_${FOLD}"
  mkdir -p "${OUT_DIR}"
  echo "[INFO] Predicting fold ${FOLD}..."
  nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${OUT_DIR}" \
    -d "${DATASET_ID}" \
    -c "${CONFIG}" \
    -f "${FOLD}" \
    -tr nnUNetTrainer \
    -p nnUNetPlans \
    --disable_tta \
    --save_probabilities
done

echo
echo "[DONE] Predictions written to ${OUTPUT_ROOT}/fold_<n>/"

