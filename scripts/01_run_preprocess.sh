#!/usr/bin/env bash
# Convenience wrapper around nnUNetv2_plan_and_preprocess.
# Usage: bash scripts/01_run_preprocess.sh [DATASET_NAME] [CONFIG] [NUM_PROCESSES]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_NAME="${1:-Dataset001_GroundTruth}"
CONFIG="${2:-3d_fullres}"
NUM_PROCESSES="${3:-8}"
PLANS_NAME="${PLANS_NAME:-nnUNetPlans}"

if [[ "${DATASET_NAME}" =~ Dataset([0-9]+)_.* ]]; then
  DATASET_ID="${BASH_REMATCH[1]}"
else
  DATASET_ID="${DATASET_NAME}"
fi

echo "=== nnU-Net Preprocessing ==="
echo "Dataset name : ${DATASET_NAME}"
echo "Dataset ID   : ${DATASET_ID}"
echo "Configuration: ${CONFIG}"
echo "Processes    : ${NUM_PROCESSES}"
echo

if [[ ! -f "${SCRIPT_DIR}/00_nnunet_env.sh" ]]; then
  echo "[ERROR] scripts/00_nnunet_env.sh not found."
  exit 1
fi

source "${SCRIPT_DIR}/00_nnunet_env.sh"

cd "${PROJECT_ROOT}"

nnUNetv2_plan_and_preprocess \
  -d "${DATASET_ID}" \
  -c "${CONFIG}" \
  -pl "${PLANS_NAME}" \
  -np "${NUM_PROCESSES}"

echo
echo "Preprocessing finished. Outputs stored under \${nnUNet_preprocessed}."

