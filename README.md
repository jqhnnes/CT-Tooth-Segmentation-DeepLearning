# 3D Tooth Segmentation using nnU-Net

A deep learning project for 3D segmentation of tooth structures from micro-CT (µCT) scans using the nnU-Net framework. This project implements a complete pipeline for automated segmentation of dental tissues including enamel, dentin, and pulp.

## 🎯 Project Overview

This project focuses on semantic segmentation of tooth structures in 3D CT volumes. The system uses nnU-Net, a self-configuring framework for biomedical image segmentation, to automatically segment three main dental tissue classes:

- **Enamel**: The outer protective layer of the tooth
- **Dentin**: The main hard tissue component
- **Pulp**: The inner soft tissue containing nerves and blood vessels

The project includes a comprehensive hyperparameter optimization (HPO) pipeline using Optuna to systematically explore parameter spaces and identify optimal configurations.

## ✨ Key Features

- 🔬 **nnU-Net Framework**: Leverages the state-of-the-art nnU-Net architecture for biomedical segmentation
- 🎛️ **Hyperparameter Optimization**: Automated HPO pipeline with Optuna for systematic parameter search
- 📊 **Comprehensive Evaluation**: Multiple metrics including Dice Score, IoU, and Hausdorff Distance
- 🏆 **Best Model Selection**: Automated identification and preparation of top-performing models
- 🔄 **Reproducible Workflow**: Complete pipeline from preprocessing to evaluation with full traceability
- 📈 **Performance Tracking**: Detailed logging and analysis of all trials and experiments

## 📁 Project Structure

```
CT-Tooth-Segmentation-DeepLearning/
├── hpo/                          # Hyperparameter Optimization
│   ├── scripts/                  # HPO pipeline scripts
│   │   ├── preprocessing/       # Preprocessing and trial generation
│   │   ├── training/            # Training and evaluation pipeline
│   │   ├── analysis/            # Model comparison and analysis
│   │   └── utils/               # Utility scripts
│   ├── config/                  # Configuration templates
│   ├── docs/                    # Documentation
│   ├── best_model/              # Best performing model (trial_8, Dice: 0.9725)
│   ├── preprocessing_output/    # Preprocessed datasets per trial
│   ├── training_output/         # Trained models and checkpoints
│   └── results/                 # Evaluation results
├── scripts/                      # Main project scripts
│   └── nnunet_env.sh            # Environment setup
├── data/                        # Data directory (not in repo)
├── notebooks/                   # Jupyter notebooks for exploration
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- nnU-Net v2 installed
- Conda environment (see `environment.yml`)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CT-Tooth-Segmentation-DeepLearning
   ```

2. **Set up environment**
   ```bash
   conda env create -f environment.yml
   conda activate <env-name>
   source scripts/nnunet_env.sh
   ```

3. **Prepare your data**
   - Organize your CT scans following nnU-Net conventions
   - Place data in the directory specified by `nnUNet_raw` environment variable

## 📖 Usage

### Hyperparameter Optimization

The project includes a comprehensive HPO pipeline to systematically explore parameter spaces:

```bash
# Generate HPO trials
python hpo/scripts/preprocessing/nnunet_hpo_preprocess.py --n_trials 50

# Train and evaluate trials
python hpo/scripts/training/nnunet_train_eval_pipeline.py --folds 0

# Prepare best model for deployment
python hpo/scripts/analysis/prepare_best_model.py
```

For detailed HPO documentation, see [`hpo/README.md`](hpo/README.md).

### Training a Single Model

```bash
# Standard nnU-Net training
nnUNetv2_train <DATASET_ID> <CONFIG> <FOLD> -tr nnUNetTrainer -p nnUNetPlans
```

### Evaluation

```bash
# Evaluate trained model
nnUNetv2_evaluate -d <DATASET_ID> -c <CONFIG> -tr nnUNetTrainer -p nnUNetPlans -f <FOLD>
```

## 📊 Results

### Best Model Performance

After running 10 HPO trials, the best performing model achieved:

- **Trial**: `trial_8`
- **Dice Score**: **0.9725**
- **Configuration**:
  - Patch Size: [160, 160, 64]
  - Batch Size: 4
  - Features Base: 24
  - Batch Dice: False
  - Use Mask for Norm: False

### Top 3 Trials

| Rank | Trial | Dice Score | Key Parameters |
|------|-------|------------|----------------|
| 1 | trial_8 | 0.9725 | Patch: [160,160,64], Batch: 4 |
| 2 | trial_1 | 0.9723 | Patch: [128,64,128], Batch: 2 |
| 3 | trial_3 | 0.9721 | Patch: [64,128,64], Batch: 2 |

For detailed parameter analysis, see [`hpo/docs/BEST_PARAMETERS_SUMMARY.md`](hpo/docs/BEST_PARAMETERS_SUMMARY.md).

## 🔧 Configuration

### Environment Variables

Set the following environment variables (via `scripts/nnunet_env.sh`):

- `nnUNet_raw`: Path to raw dataset directory
- `nnUNet_preprocessed`: Path to preprocessed data
- `nnUNet_results`: Path to training results

### Dataset Format

The project follows nnU-Net conventions:

```
nnUNet_raw/
└── Dataset001_GroundTruth/
    ├── imagesTr/          # Training images
    ├── labelsTr/          # Training labels
    ├── imagesTs/          # Test images (optional)
    └── dataset.json       # Dataset metadata
```

## 📈 Evaluation Metrics

The system computes the following metrics:

- **Dice Score**: Overlap measure between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index
- **Hausdorff Distance**: Maximum surface distance
- **Volume Similarity**: Volume overlap ratio
- **Precision & Recall**: Classification metrics

## 🛠️ Development

### Project Components

- **HPO Pipeline**: Automated hyperparameter search using Optuna
- **Training Pipeline**: Automated training and evaluation workflow
- **Analysis Tools**: Model comparison and best model selection
- **Utility Scripts**: Data validation, label remapping, and preprocessing fixes

### Key Scripts

- `hpo/scripts/preprocessing/nnunet_hpo_preprocess.py`: Generate HPO trials
- `hpo/scripts/training/nnunet_train_eval_pipeline.py`: Train and evaluate models
- `hpo/scripts/analysis/prepare_best_model.py`: Prepare best model for deployment
- `hpo/scripts/utils/`: Various utility scripts for data management

## 📚 Documentation

- **[HPO Runbook](hpo/README.md)**: Complete guide for hyperparameter optimization
- **[Best Parameters Summary](hpo/docs/BEST_PARAMETERS_SUMMARY.md)**: Analysis of top-performing configurations
- **[Best Model Documentation](hpo/best_model/README.md)**: Instructions for using the best model

## 🎓 Hinweise für Bachelorarbeiten

Wenn du dieses Projekt als Grundlage deiner Bachelorarbeit verwendest, empfehlen wir:

- **Zitiere das Projekt** mit dem untenstehenden BibTeX-Eintrag und verlinke das Repository.
- **Dokumentiere die Reproduzierbarkeit** (Commit-Hash, `environment.yml`, verwendete Datensätze und Befehle).
- **Speichere Experimente** unter `hpo/results/` und referenziere die Auswertung in deiner Arbeit.
- **Beachte die Lizenz** und führe die ursprünglichen Autor:innen in den Danksagungen auf.

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@software{tooth_segmentation_2025,
  author = {Johannes},
  title = {3D Tooth Segmentation using nnU-Net},
  year = {2025},
  url = {https://github.com/jqhnnes/Segmentierung-von-CT-Aufnahmen-extrahierter-Z-hne-mittels-Deep-Learning}
}
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project uses nnU-Net v2. For more information about nnU-Net, visit the [official repository](https://github.com/MIC-DKFZ/nnUNet).
