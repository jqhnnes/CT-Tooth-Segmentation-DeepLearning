#!/bin/bash
export nnUNet_raw="/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/data/nnUNet_raw"
export nnUNet_preprocessed="/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/data/nnUNet_preprocessed"
export nnUNet_results="/ssd/geiger/CT-Tooth-Segmentation-DeepLearning/data/nnUNet_results"

# Make sure nnU-Net CLI tools from the conda env are available.
NNUNET_BIN_DIR="/ssd/geiger/myenv/bin"
case ":$PATH:" in
    *":$NNUNET_BIN_DIR:"*) ;;
    *) export PATH="$NNUNET_BIN_DIR:$PATH" ;;
esac

echo "nnUNet Environment geladen."
