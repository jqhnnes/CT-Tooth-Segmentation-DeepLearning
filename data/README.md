# Data Directory

This directory should contain your µCT tooth scan data organized in the following structure:

## Directory Structure

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
│
├── val/
│   ├── images/
│   ├── masks/
│   └── metadata.json (optional)
│
├── test_internal/
│   ├── images/
│   ├── masks/
│   └── metadata.json (optional)
│
└── test_external/
    ├── images/
    ├── masks/
    └── metadata.json (optional)
```

## File Formats

### Image Files (`.npy`)
- NumPy arrays with shape `(D, H, W)` where:
  - D = depth (number of slices)
  - H = height (pixels)
  - W = width (pixels)
- Data type: `float32` or `float64`
- Values: Hounsfield Units (HU) for CT scans, typically ranging from -1000 to 3000
- Example dimensions: (100, 120, 120) or (150, 150, 150)

### Mask Files (`.npy`)
- NumPy arrays with same shape as corresponding image `(D, H, W)`
- Data type: `int64` or `int32`
- Class labels:
  - `0`: Background
  - `1`: Enamel (Zahnschmelz)
  - `2`: Dentin
  - `3`: Pulpa
- Each voxel should have exactly one class label

### Metadata File (optional `metadata.json`)
```json
{
  "tooth_001": {
    "spacing": [0.15, 0.15, 0.15],
    "patient_id": "P001",
    "scan_date": "2024-01-15",
    "notes": "Premolar tooth"
  },
  "tooth_002": {
    "spacing": [0.12, 0.12, 0.12],
    "patient_id": "P002"
  }
}
```

## Data Requirements

### Minimum Requirements
- **Training set**: At least 20-30 annotated samples
- **Validation set**: At least 5-10 samples
- **Test set**: At least 5-10 samples for evaluation

### Recommended
- **Training set**: 50-100+ samples for good performance
- **Validation set**: 10-20 samples
- **Internal test set**: 10-20 samples (same scanner/protocol)
- **External test set**: 10-20 samples (different scanner/protocol)

## Creating Your Data

### From DICOM Files

```python
import SimpleITK as sitk
import numpy as np

# Read DICOM series
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames('path/to/dicom/folder')
reader.SetFileNames(dicom_names)
image = reader.Execute()

# Convert to numpy array
volume = sitk.GetArrayFromImage(image)

# Save as .npy
np.save('data/train/images/tooth_001.npy', volume)
```

### From NIfTI Files

```python
import nibabel as nib
import numpy as np

# Load NIfTI
img = nib.load('path/to/file.nii.gz')
volume = img.get_fdata()

# Save as .npy
np.save('data/train/images/tooth_001.npy', volume)
```

### Generate Test Data

For testing the pipeline without real data:

```bash
python tests/generate_dummy_data.py --output_dir data/train --num_samples 20
python tests/generate_dummy_data.py --output_dir data/val --num_samples 5
python tests/generate_dummy_data.py --output_dir data/test_internal --num_samples 5
```

## Data Quality Checks

Before training, verify your data:

```python
import numpy as np

# Load sample
image = np.load('data/train/images/tooth_001.npy')
mask = np.load('data/train/masks/tooth_001.npy')

# Check shapes match
assert image.shape == mask.shape, "Image and mask shapes must match"

# Check mask labels
unique_labels = np.unique(mask)
assert all(label in [0, 1, 2, 3] for label in unique_labels), \
    "Mask should only contain labels 0, 1, 2, 3"

# Check for NaN/Inf
assert not np.any(np.isnan(image)), "Image contains NaN values"
assert not np.any(np.isinf(image)), "Image contains Inf values"

# Check intensity range
print(f"Image range: [{image.min():.2f}, {image.max():.2f}]")
print(f"Mask labels: {unique_labels}")
print(f"Class distribution:")
for label in unique_labels:
    count = np.sum(mask == label)
    percentage = 100 * count / mask.size
    print(f"  Class {label}: {count} voxels ({percentage:.2f}%)")
```

## Class Imbalance

If your data has class imbalance (e.g., much more background than teeth structures), consider:

1. **Using class weights in loss function**:
   ```python
   # In training script
   class_weights = torch.tensor([1.0, 2.0, 2.0, 3.0])
   criterion = get_loss_function('combined', class_weights=class_weights)
   ```

2. **Adjusting loss function weights**:
   ```yaml
   # In config file
   loss:
     dice_weight: 0.6  # Increase Dice weight
     ce_weight: 0.2
     focal_weight: 0.2
   ```

## Privacy and Ethics

⚠️ **Important**: When working with medical data:
- Ensure you have proper authorization to use the data
- Remove or anonymize patient identifiers
- Follow HIPAA/GDPR guidelines if applicable
- Do not share patient data publicly

## Need Help?

- Check `QUICKSTART.md` for quick setup guide
- See `README.md` for detailed documentation
- Review example notebooks in `notebooks/`
- Use the dummy data generator for testing
