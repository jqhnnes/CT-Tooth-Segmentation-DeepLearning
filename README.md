# 3D Tooth Segmentation using Deep Learning

Deep-Learning-Projekt zur 3D-Segmentierung von Zähnen in µCT-Aufnahmen (Micro-CT Scans).

## Projektübersicht

Dieses Projekt implementiert eine vollständige Pipeline für die 3D-Segmentierung von Zahnstrukturen aus µCT-Aufnahmen mittels Deep Learning. Das System segmentiert drei Hauptklassen:
- **Zahnschmelz (Enamel)**: Äußere Schutzschicht
- **Dentin**: Hauptbestandteil der Zahnsubstanz
- **Pulpa**: Inneres Zahnmark

## Features

✅ **3D U-Net Architektur** für volumetrische Segmentierung  
✅ **Umfassende Datenvorverarbeitung** (Resampling, Normalisierung, Augmentation)  
✅ **Mehrere Loss-Funktionen** (Dice, Focal, Combined, Tversky)  
✅ **Evaluation Metriken** (Dice Score, IoU, Hausdorff Distance)  
✅ **Training & Evaluation** auf internem und externem Testset  
✅ **Optional: Active Learning** für iterative Modellverbesserung  
✅ **TensorBoard Integration** für Trainingsvisualisierung  

## Projektstruktur

```
.
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet3d.py              # 3D U-Net Implementierung
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # Datenvorverarbeitung
│   │   └── dataset.py             # PyTorch Dataset Klassen
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Evaluation Metriken
│   │   └── losses.py              # Loss-Funktionen
│   ├── training/
│   │   ├── train.py               # Training Script
│   │   └── active_learning.py     # Active Learning Modul
│   └── evaluation/
│       └── evaluate.py            # Evaluation Script
├── configs/
│   └── train_config.yaml          # Konfigurationsdatei
├── notebooks/
│   ├── 01_training_example.ipynb
│   └── 02_data_exploration.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Repository klonen
```bash
git clone https://github.com/jqhnnes/Segmentierung-von-CT-Aufnahmen-extrahierter-Z-hne-mittels-Deep-Learning.git
cd Segmentierung-von-CT-Aufnahmen-extrahierter-Z-hne-mittels-Deep-Learning
```

### 2. Virtual Environment erstellen (empfohlen)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

## Datenformat

### Erwartete Verzeichnisstruktur
```
data/
├── train/
│   ├── images/
│   │   ├── tooth_001.npy
│   │   ├── tooth_002.npy
│   │   └── ...
│   ├── masks/
│   │   ├── tooth_001.npy
│   │   ├── tooth_002.npy
│   │   └── ...
│   └── metadata.json (optional)
├── val/
│   ├── images/
│   └── masks/
├── test_internal/
│   ├── images/
│   └── masks/
└── test_external/
    ├── images/
    └── masks/
```

### Datenformat
- **Images**: NumPy arrays (.npy) mit Form (D, H, W) - 3D-Volumen
- **Masks**: NumPy arrays (.npy) mit Integer-Labels:
  - 0: Background
  - 1: Enamel (Zahnschmelz)
  - 2: Dentin
  - 3: Pulpa

### Metadata (optional)
```json
{
  "tooth_001": {
    "spacing": [0.15, 0.15, 0.15],
    "patient_id": "P001"
  }
}
```

## Verwendung

### 1. Training

#### Command Line
```bash
python src/training/train.py \
    --train_data data/train \
    --val_data data/val \
    --batch_size 2 \
    --num_epochs 100 \
    --lr 0.0001 \
    --checkpoint_dir checkpoints \
    --log_dir logs
```

#### Mit Konfigurationsdatei
```bash
python src/training/train.py --config configs/train_config.yaml
```

#### Python Script
```python
from src.models import UNet3D
from src.data import CTPreprocessor, VolumeAugmenter, create_data_loaders
from src.utils import get_loss_function
from src.training.train import Trainer

# Model erstellen
model = UNet3D(n_channels=1, n_classes=4, base_channels=32)

# Data Loaders erstellen
preprocessor = CTPreprocessor(target_size=(128, 128, 128))
augmenter = VolumeAugmenter()
train_loader, val_loader = create_data_loaders(
    train_root='data/train',
    val_root='data/val',
    preprocessor=preprocessor,
    augmenter=augmenter
)

# Training konfigurieren
criterion = get_loss_function('combined', num_classes=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
trainer.train(num_epochs=100)
```

### 2. Evaluation

```bash
python src/evaluation/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --internal_test data/test_internal \
    --external_test data/test_external \
    --output_dir evaluation_results \
    --save_predictions
```

### 3. TensorBoard Visualisierung

```bash
tensorboard --logdir logs
```

Öffne dann `http://localhost:6006` im Browser.

## Evaluation Metriken

Das System berechnet folgende Metriken:

- **Dice Score**: Überlappungsmaß zwischen Vorhersage und Ground Truth
- **IoU (Intersection over Union)**: Jaccard Index
- **Hausdorff Distance**: Maximaler Abstand zwischen Oberflächen
- **Volume Similarity**: Ähnlichkeit der segmentierten Volumina
- **Precision & Recall**: Klassifikationsmetriken

## Active Learning (Optional)

Für iterative Modellverbesserung:

```python
from src.training.active_learning import ActiveLearner, UncertaintyEstimator

# Uncertainty-basierte Sampleauswahl
learner = ActiveLearner(
    model=trained_model,
    unlabeled_pool=unlabeled_files,
    uncertainty_method='entropy'
)

# Uncertainties berechnen
uncertainties = learner.compute_uncertainties(preprocessor)

# Samples für Annotation auswählen
selected_samples = learner.select_samples(n_samples=10, strategy='uncertainty')
```

## Modellarchitektur

### 3D U-Net
- **Encoder**: 5 Stufen mit Double Convolution Blocks
- **Decoder**: 4 Upsampling-Stufen mit Skip Connections
- **Output**: 4 Klassen (Background, Enamel, Dentin, Pulpa)
- **Aktivierung**: Softmax für Klassenprobabilitäten

### Loss-Funktionen
- **Dice Loss**: Direkte Optimierung des Dice Scores
- **Focal Loss**: Fokus auf schwierige Samples
- **Combined Loss**: Gewichtete Kombination mehrerer Losses
- **Tversky Loss**: Kontrolle über False Positives/Negatives

## Konfiguration

Alle Hyperparameter können in `configs/train_config.yaml` angepasst werden:

```yaml
model:
  n_classes: 4
  base_channels: 32

training:
  batch_size: 2
  num_epochs: 100
  learning_rate: 0.0001

data:
  target_size: [128, 128, 128]
  target_spacing: [0.1, 0.1, 0.1]
```

## Beispiel Notebooks

- `notebooks/01_training_example.ipynb`: Vollständiges Trainingsbeispiel
- `notebooks/02_data_exploration.ipynb`: Datenexploration und Visualisierung

## Hardware-Anforderungen

- **Minimum**: 8 GB RAM, GPU mit 6 GB VRAM
- **Empfohlen**: 16 GB RAM, GPU mit 11+ GB VRAM
- **CPU-only**: Möglich, aber deutlich langsamer

## Tipps für bessere Ergebnisse

1. **Datenaugmentation**: Nutze Rotation, Flipping und Elastic Deformation
2. **Class Weighting**: Bei unbalancierten Klassen Gewichte anpassen
3. **Learning Rate**: Starte mit 1e-4 und nutze Scheduler
4. **Batch Size**: Größer = stabiler, aber mehr VRAM nötig
5. **Preprocessing**: Normalisierung und Clipping sind wichtig

## Troubleshooting

### CUDA Out of Memory
- Reduziere `batch_size`
- Reduziere `target_size`
- Reduziere `base_channels` im Modell
- Nutze `gradient_checkpointing`

### Schlechte Segmentierung
- Überprüfe Datenqualität und Labels
- Erhöhe Trainingsdauer
- Passe Loss-Funktion an (z.B. höheres `dice_weight`)
- Nutze Class Weights bei unbalancierten Klassen

## Zitation

Wenn du dieses Projekt verwendest, bitte zitiere:

```bibtex
@software{tooth_segmentation_2025,
  author = {Johannes},
  title = {3D Tooth Segmentation using Deep Learning},
  year = {2025},
  url = {https://github.com/jqhnnes/Segmentierung-von-CT-Aufnahmen-extrahierter-Z-hne-mittels-Deep-Learning}
}
```

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.

## Kontakt

Bei Fragen oder Problemen öffne bitte ein Issue auf GitHub.
