#!/usr/bin/env bash
# Run nnUNetv2_ensemble using per-fold prediction folders.
# Usage: bash scripts/04_ensemble.sh DATASET_NAME CONFIG [PREDICTIONS_ROOT]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLEAN_TMP_PARENT=""

clean_up() {
  if [[ -n "${CLEAN_TMP_PARENT}" && -d "${CLEAN_TMP_PARENT}" ]]; then
    rm -rf "${CLEAN_TMP_PARENT}"
  fi
}

trap clean_up EXIT

DATASET_NAME="${1:-Dataset001_GroundTruth}"
CONFIG="${2:-3d_fullres}"
PRED_ROOT_ARG="${3:-}"

if [[ "${DATASET_NAME}" =~ Dataset([0-9]+)_.* ]]; then
  DATASET_ID="${BASH_REMATCH[1]}"
else
  DATASET_ID="${DATASET_NAME}"
fi

echo "=== nnU-Net Ensemble ==="
echo "Dataset name : ${DATASET_NAME}"
echo "Dataset ID   : ${DATASET_ID}"
echo "Configuration: ${CONFIG}"
echo

if [[ ! -f "${SCRIPT_DIR}/nnunet_env.sh" ]]; then
  echo "[ERROR] scripts/nnunet_env.sh not found."
  exit 1
fi

source "${SCRIPT_DIR}/nnunet_env.sh"
cd "${PROJECT_ROOT}"

DEFAULT_PRED_DIR="${PROJECT_ROOT}/predictions/${DATASET_NAME}_${CONFIG}"
if [[ ! -d "${DEFAULT_PRED_DIR}" ]]; then
  DEFAULT_PRED_DIR="${PROJECT_ROOT}/data/nnUNet_results/Dataset001_GroundTruth/nnUNetTrainer__nnUNetPlans__3d_fullres/predictions"
fi
PRED_ROOT="${PRED_ROOT_ARG:-${DEFAULT_PRED_DIR}}"

echo "Pred root    : ${PRED_ROOT}"

if [[ ! -d "${PRED_ROOT}" ]]; then
  echo "[ERROR] Prediction root ${PRED_ROOT} not found."
  exit 1
fi

INPUT_FOLDS=()
for FOLD_DIR in "${PRED_ROOT}"/fold_*; do
  if [[ -d "${FOLD_DIR}" ]]; then
    INPUT_FOLDS+=("${FOLD_DIR}")
  fi
done

if [[ ${#INPUT_FOLDS[@]} -eq 0 ]]; then
  echo "[ERROR] No fold_* directories found under ${PRED_ROOT}"
  exit 1
fi

echo "[INFO] Folds to ensemble:"
printf '  %s\n' "${INPUT_FOLDS[@]}"

OUTPUT_DIR="${PROJECT_ROOT}/ensemble_predictions/${DATASET_NAME}_${CONFIG}"
mkdir -p "${OUTPUT_DIR}"

nnUNetv2_ensemble \
  -i "${INPUT_FOLDS[@]}" \
  -o "${OUTPUT_DIR}"

echo
echo "[DONE] Ensemble predictions written to ${OUTPUT_DIR}"