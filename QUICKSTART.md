# Quick Start Guide

Get started with 3D tooth segmentation in 5 minutes!

## Installation

```bash
# Clone repository
git clone https://github.com/jqhnnes/Segmentierung-von-CT-Aufnahmen-extrahierter-Z-hne-mittels-Deep-Learning.git
cd Segmentierung-von-CT-Aufnahmen-extrahierter-Z-hne-mittels-Deep-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Prepare Your Data

Organize your data like this:

```
data/
├── train/
│   ├── images/
│   │   ├── tooth_001.npy
│   │   └── tooth_002.npy
│   └── masks/
│       ├── tooth_001.npy
│       └── tooth_002.npy
└── val/
    ├── images/
    └── masks/
```

**Data format:**
- Images: `.npy` files with shape (D, H, W) - 3D volumes
- Masks: `.npy` files with integer labels (0=Background, 1=Enamel, 2=Dentin, 3=Pulpa)

### Generate Test Data (Optional)

```bash
python tests/generate_dummy_data.py --output_dir data/train --num_samples 10
python tests/generate_dummy_data.py --output_dir data/val --num_samples 5
```

## Train Your First Model

```bash
python src/training/train.py \
    --train_data data/train \
    --val_data data/val \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 0.0001
```

**Monitor training:**
```bash
tensorboard --logdir logs
```
Open http://localhost:6006 in your browser.

## Evaluate Your Model

```bash
python src/evaluation/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --internal_test data/test_internal \
    --save_predictions
```

## Using in Python

```python
import torch
import numpy as np
from src.models import UNet3D
from src.data import CTPreprocessor

# Load model
model = UNet3D(n_channels=1, n_classes=4)
checkpoint = torch.load('checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess input
preprocessor = CTPreprocessor(target_size=(128, 128, 128))
volume = np.load('my_tooth_scan.npy')
volume_prep = preprocessor.preprocess(volume)

# Predict
input_tensor = torch.from_numpy(volume_prep[None, None, ...])
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).cpu().numpy()[0]

# Save result
np.save('prediction.npy', prediction)
```

## Configuration

Edit `configs/train_config.yaml` to customize:
- Model architecture (channels, depth)
- Training hyperparameters (batch size, learning rate)
- Data preprocessing (spacing, size, normalization)
- Augmentation settings
- Loss function weights

## Common Settings

### For Limited GPU Memory (< 8 GB)
```yaml
model:
  base_channels: 16  # Reduce from 32

training:
  batch_size: 1      # Reduce from 2

data:
  preprocessing:
    target_size: [96, 96, 96]  # Reduce from [128, 128, 128]
```

### For Better Performance
```yaml
training:
  batch_size: 4      # Increase if you have memory
  num_epochs: 200    # More training

data:
  augmentation:
    rotation_range: 20.0    # More augmentation
    flip_prob: 0.7
```

## Next Steps

1. **Read full documentation**: See README.md for comprehensive guide
2. **Check examples**: Explore notebooks/ for detailed examples
3. **Advanced usage**: See USAGE.md for active learning and more
4. **Troubleshooting**: Check README.md troubleshooting section

## Need Help?

- Check README.md for comprehensive documentation
- See USAGE.md for advanced features
- Open an issue on GitHub
- Review example notebooks in notebooks/

## Tips

✅ Start with dummy data to verify setup works
✅ Use smaller batch sizes if you run out of memory
✅ Monitor training with TensorBoard
✅ Save checkpoints regularly
✅ Validate on held-out test set

## System Requirements

**Minimum:**
- Python 3.8+
- 8 GB RAM
- GPU with 6 GB VRAM (or CPU, slower)

**Recommended:**
- Python 3.9+
- 16 GB RAM
- GPU with 11+ GB VRAM
- SSD for faster data loading

That's it! You're ready to segment teeth in 3D! 🦷
