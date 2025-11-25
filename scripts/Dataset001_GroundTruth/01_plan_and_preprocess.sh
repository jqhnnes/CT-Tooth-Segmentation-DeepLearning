#!/bin/bash
# Script: 01_plan_and_preprocess.sh
# Purpose: Run nnUNetv2 planning and preprocessing for dataset 001
# Note: Uses relative paths and direct Python environment

# --- Python Environment ---
# Set the path to the Python executable in your environment
PYTHON=/ssd/geiger/myenv/bin/python

# --- Base Directory ---
# Get the absolute path to the folder two levels above this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$SCRIPT_DIR/../../"

# --- nnUNet Environment Variables ---
# These point to the raw data, preprocessed data, and results folders
export nnUNet_raw="$BASE_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$BASE_DIR/data/nnUNet_results"

# --- Dataset ID ---
DATASET_ID=1

# --- Run nnUNetv2 Planning & Preprocessing ---
echo "=== nnUNetv2: Planning & Preprocessing for Dataset ${DATASET_ID} ==="

# This command internally runs both planning and preprocessing
nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 3d_fullres

echo "=== nnUNetv2: Done for Dataset ${DATASET_ID} ==="
