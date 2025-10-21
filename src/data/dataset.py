"""
PyTorch Dataset classes for µCT tooth scan data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict
import json

from .preprocessing import CTPreprocessor, VolumeAugmenter


class ToothDataset(Dataset):
    """
    Dataset for 3D tooth segmentation from µCT scans.
    
    Expected directory structure:
        data_root/
            images/
                tooth_001.npy
                tooth_002.npy
                ...
            masks/
                tooth_001.npy
                tooth_002.npy
                ...
            metadata.json  # Optional: contains spacing info
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 preprocessor: Optional[CTPreprocessor] = None,
                 augmenter: Optional[VolumeAugmenter] = None,
                 transform: Optional[Callable] = None,
                 file_list: Optional[List[str]] = None):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing images and masks
            split: Dataset split ('train', 'val', 'test')
            preprocessor: Preprocessing pipeline
            augmenter: Data augmentation pipeline (applied only for training)
            transform: Additional transforms
            file_list: Optional list of specific files to use
        """
        self.data_root = Path(data_root)
        self.split = split
        self.preprocessor = preprocessor
        self.augmenter = augmenter if split == 'train' else None
        self.transform = transform
        
        # Get file paths
        self.image_dir = self.data_root / 'images'
        self.mask_dir = self.data_root / 'masks'
        
        if file_list is not None:
            self.files = file_list
        else:
            self.files = sorted([f.stem for f in self.image_dir.glob('*.npy')])
        
        # Load metadata if available
        self.metadata = {}
        metadata_path = self.data_root / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"Loaded {len(self.files)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dictionary containing:
                - 'image': 4D tensor (C, D, H, W)
                - 'mask': 3D tensor (D, H, W) with class indices
                - 'filename': str
        """
        filename = self.files[idx]
        
        # Load image and mask
        image_path = self.image_dir / f"{filename}.npy"
        mask_path = self.mask_dir / f"{filename}.npy"
        
        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)
        
        # Get spacing if available
        spacing = None
        if filename in self.metadata:
            spacing = self.metadata[filename].get('spacing')
        
        # Preprocessing
        if self.preprocessor is not None:
            if spacing is not None:
                image, mask = self.preprocessor.preprocess(image, spacing, mask)
            else:
                image, mask = self.preprocessor.preprocess(image, mask=mask)
        
        # Augmentation (only for training)
        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)
        
        # Add channel dimension to image
        if image.ndim == 3:
            image = image[np.newaxis, ...]  # Add channel dimension
        
        # Convert to tensors
        image = torch.from_numpy(image.copy())
        mask = torch.from_numpy(mask.copy())
        
        # Additional transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            'image': image,
            'mask': mask,
            'filename': filename
        }


class InferenceDataset(Dataset):
    """Dataset for inference (no masks required)."""
    
    def __init__(self,
                 data_root: str,
                 preprocessor: Optional[CTPreprocessor] = None,
                 file_list: Optional[List[str]] = None):
        """
        Initialize inference dataset.
        
        Args:
            data_root: Root directory containing images
            preprocessor: Preprocessing pipeline
            file_list: Optional list of specific files to use
        """
        self.data_root = Path(data_root)
        self.preprocessor = preprocessor
        
        self.image_dir = self.data_root / 'images'
        
        if file_list is not None:
            self.files = file_list
        else:
            self.files = sorted([f.stem for f in self.image_dir.glob('*.npy')])
        
        # Load metadata if available
        self.metadata = {}
        metadata_path = self.data_root / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"Loaded {len(self.files)} samples for inference")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename = self.files[idx]
        
        # Load image
        image_path = self.image_dir / f"{filename}.npy"
        image = np.load(image_path).astype(np.float32)
        
        # Get spacing if available
        spacing = None
        if filename in self.metadata:
            spacing = self.metadata[filename].get('spacing')
        
        # Preprocessing
        if self.preprocessor is not None:
            if spacing is not None:
                image = self.preprocessor.preprocess(image, spacing)
            else:
                image = self.preprocessor.preprocess(image)
        
        # Add channel dimension
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        
        # Convert to tensor
        image = torch.from_numpy(image.copy())
        
        return {
            'image': image,
            'filename': filename,
            'original_shape': torch.tensor(np.load(image_path).shape)
        }


def create_data_loaders(
    train_root: str,
    val_root: str,
    batch_size: int = 2,
    num_workers: int = 4,
    preprocessor: Optional[CTPreprocessor] = None,
    augmenter: Optional[VolumeAugmenter] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_root: Root directory for training data
        val_root: Root directory for validation data
        batch_size: Batch size
        num_workers: Number of data loading workers
        preprocessor: Preprocessing pipeline
        augmenter: Augmentation pipeline
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = ToothDataset(
        data_root=train_root,
        split='train',
        preprocessor=preprocessor,
        augmenter=augmenter
    )
    
    val_dataset = ToothDataset(
        data_root=val_root,
        split='val',
        preprocessor=preprocessor,
        augmenter=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset creation
    from preprocessing import CTPreprocessor, VolumeAugmenter
    
    preprocessor = CTPreprocessor(
        target_spacing=(0.1, 0.1, 0.1),
        target_size=(128, 128, 128),
        normalize=True
    )
    
    augmenter = VolumeAugmenter(
        rotation_range=15.0,
        flip_prob=0.5
    )
    
    # This would require actual data to test
    print("Dataset classes defined successfully")
