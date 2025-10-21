# Usage Guide

## Quick Start

### 1. Prepare Your Data

Organize your µCT scan data following this structure:

```
data/
├── train/
│   ├── images/    # .npy files with 3D volumes
│   └── masks/     # .npy files with segmentation labels
├── val/
│   ├── images/
│   └── masks/
└── test_internal/
    ├── images/
    └── masks/
```

### 2. Train a Model

Basic training:
```bash
python src/training/train.py \
    --train_data data/train \
    --val_data data/val \
    --num_epochs 100
```

With custom settings:
```bash
python src/training/train.py \
    --train_data data/train \
    --val_data data/val \
    --batch_size 4 \
    --num_epochs 200 \
    --lr 0.0001 \
    --checkpoint_dir my_checkpoints
```

### 3. Monitor Training

Start TensorBoard to visualize training progress:
```bash
tensorboard --logdir logs
```

Open your browser at `http://localhost:6006`

### 4. Evaluate Model

Evaluate on test sets:
```bash
python src/evaluation/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --internal_test data/test_internal \
    --external_test data/test_external \
    --save_predictions
```

## Advanced Usage

### Custom Preprocessing

```python
from src.data import CTPreprocessor

preprocessor = CTPreprocessor(
    target_spacing=(0.08, 0.08, 0.08),  # Finer resolution
    target_size=(160, 160, 160),         # Larger volume
    normalize=True,
    clip_range=(-500, 2000)              # Custom HU range
)
```

### Data Augmentation

```python
from src.data import VolumeAugmenter

augmenter = VolumeAugmenter(
    rotation_range=20.0,      # More rotation
    flip_prob=0.7,            # Higher flip probability
    noise_std=0.1,            # More noise
    brightness_range=0.3,     # Wider brightness range
    elastic_alpha=15.0,       # Stronger elastic deformation
    elastic_sigma=5.0
)
```

### Custom Loss Function

```python
from src.utils import get_loss_function

# Weighted combined loss
criterion = get_loss_function(
    loss_type='combined',
    num_classes=4,
    dice_weight=0.6,
    ce_weight=0.2,
    focal_weight=0.2,
    class_weights=torch.tensor([1.0, 2.0, 2.0, 3.0])  # Weight rare classes more
)
```

### Learning Rate Scheduling

```python
import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# Or step decay
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)

# Or ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)
```

## Active Learning Workflow

### 1. Initial Training

Train model on initial labeled dataset:
```bash
python src/training/train.py --train_data data/train_initial
```

### 2. Select Uncertain Samples

```python
from src.training.active_learning import ActiveLearner
from src.data import CTPreprocessor

# Load model
model = torch.load('checkpoints/best.pth')

# Create active learner
learner = ActiveLearner(
    model=model,
    unlabeled_pool=unlabeled_files,
    uncertainty_method='entropy'
)

# Compute uncertainties
preprocessor = CTPreprocessor(...)
uncertainties = learner.compute_uncertainties(preprocessor)

# Select most uncertain samples
selected = learner.select_samples(n_samples=20, strategy='uncertainty')
```

### 3. Label Selected Samples

Manually annotate the selected samples and add them to training data.

### 4. Retrain

Retrain model with expanded dataset:
```bash
python src/training/train.py \
    --train_data data/train_expanded \
    --resume
```

### 5. Iterate

Repeat steps 2-4 until desired performance is achieved.

## Inference on New Data

```python
from src.models import UNet3D
from src.data import CTPreprocessor
import torch
import numpy as np

# Load model
model = UNet3D(n_channels=1, n_classes=4)
checkpoint = torch.load('checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess input
preprocessor = CTPreprocessor(target_size=(128, 128, 128))
volume = np.load('new_scan.npy')
volume_prep = preprocessor.preprocess(volume)

# Add batch and channel dimensions
input_tensor = torch.from_numpy(volume_prep[None, None, ...])

# Predict
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).cpu().numpy()[0]

# Save result
np.save('prediction.npy', prediction)
```

## Tips and Best Practices

### Memory Optimization

1. **Reduce batch size**: Use smaller batches if running out of memory
2. **Reduce volume size**: Use smaller `target_size` in preprocessing
3. **Mixed precision training**: Use `torch.cuda.amp` for faster training
4. **Gradient checkpointing**: Trade compute for memory

```python
# Mixed precision example
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, masks)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Improving Segmentation Quality

1. **Ensemble models**: Train multiple models and average predictions
2. **Post-processing**: Apply morphological operations to clean predictions
3. **Test-time augmentation**: Average predictions over augmented versions
4. **Fine-tune on domain**: If evaluating on different scanner, fine-tune

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check data quality:
```python
# Verify data loading
batch = next(iter(train_loader))
print(f"Image shape: {batch['image'].shape}")
print(f"Mask shape: {batch['mask'].shape}")
print(f"Unique mask values: {torch.unique(batch['mask'])}")
```

Visualize predictions during training:
```python
# In training loop
if epoch % 10 == 0:
    with torch.no_grad():
        sample = next(iter(val_loader))
        pred = model(sample['image'].to(device))
        # Save visualization
        visualize_prediction(sample['image'], sample['mask'], pred)
```

## Common Issues

### Issue: Training is very slow
**Solution**: 
- Reduce `num_workers` in DataLoader (try 2 or 4)
- Check if GPU is being used: `torch.cuda.is_available()`
- Reduce `target_size` or `batch_size`

### Issue: Loss is NaN
**Solution**:
- Reduce learning rate (try 1e-5)
- Check data for NaN/Inf values
- Use gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

### Issue: Model predicts only background
**Solution**:
- Use class weights to handle imbalance
- Increase weight of Dice loss in combined loss
- Check mask labels are correct (0, 1, 2, 3)

### Issue: Poor generalization to external test set
**Solution**:
- Increase data augmentation
- Use domain adaptation techniques
- Fine-tune on small subset of external data
- Normalize intensities more carefully
