#!/usr/bin/env bash
# Fingerprint + planning only (no preprocessing).
# Usage: bash scripts/00_plan.sh [DATASET_NAME] [NUM_PROCESSES]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_NAME="${1:-Dataset001_GroundTruth}"
NUM_PROCESSES="${2:-8}"

if [[ "${DATASET_NAME}" =~ Dataset([0-9]+)_.* ]]; then
  DATASET_ID="${BASH_REMATCH[1]}"
else
  DATASET_ID="${DATASET_NAME}"
fi

echo "=== nnU-Net Planning (no preprocessing) ==="
echo "Dataset name : ${DATASET_NAME}"
echo "Dataset ID   : ${DATASET_ID}"
echo "Processes    : ${NUM_PROCESSES}"
echo

if [[ ! -f "${SCRIPT_DIR}/nnunet_env.sh" ]]; then
  echo "[ERROR] scripts/nnunet_env.sh not found."
  exit 1
fi

source "${SCRIPT_DIR}/nnunet_env.sh"
cd "${PROJECT_ROOT}"

CMD=(nnUNetv2_plan_and_preprocess
  -d "${DATASET_ID}"
  -npfp "${NUM_PROCESSES}"
  --no_pp
  -clean
)
if [[ -n "${PLANS_NAME:-}" ]]; then
  CMD+=(-overwrite_plans_name "${PLANS_NAME}")
fi

"${CMD[@]}"

echo
echo "[INFO] Planner finished. You can now edit \${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans.json."

