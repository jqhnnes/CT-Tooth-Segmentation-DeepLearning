"""
Basic tests for the 3D tooth segmentation project.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import UNet3D
from data import CTPreprocessor, VolumeAugmenter
from utils import (
    DiceScore, IoUScore, SegmentationMetrics,
    DiceLoss, FocalLoss, CombinedLoss, get_loss_function
)


class TestModel:
    """Test model functionality."""
    
    def test_unet3d_forward(self):
        """Test 3D U-Net forward pass."""
        model = UNet3D(n_channels=1, n_classes=4, base_channels=16)
        x = torch.randn(1, 1, 32, 32, 32)
        output = model(x)
        
        assert output.shape == (1, 4, 32, 32, 32), f"Expected shape (1, 4, 32, 32, 32), got {output.shape}"
    
    def test_unet3d_different_sizes(self):
        """Test U-Net with different input sizes."""
        model = UNet3D(n_channels=1, n_classes=4, base_channels=16)
        
        sizes = [(64, 64, 64), (128, 128, 128)]
        for size in sizes:
            x = torch.randn(1, 1, *size)
            output = model(x)
            assert output.shape == (1, 4, *size)


class TestPreprocessing:
    """Test preprocessing functionality."""
    
    def test_preprocessor_normalize(self):
        """Test intensity normalization."""
        preprocessor = CTPreprocessor(normalize=True)
        volume = np.random.randn(50, 50, 50)
        
        normalized = preprocessor.normalize_intensity(volume, method='zscore')
        
        assert abs(normalized.mean()) < 0.1, "Mean should be close to 0"
        assert abs(normalized.std() - 1.0) < 0.1, "Std should be close to 1"
    
    def test_preprocessor_crop_pad(self):
        """Test crop/pad functionality."""
        preprocessor = CTPreprocessor(target_size=(64, 64, 64))
        
        # Test padding
        small_volume = np.random.randn(32, 32, 32)
        padded = preprocessor.crop_or_pad(small_volume)
        assert padded.shape == (64, 64, 64)
        
        # Test cropping
        large_volume = np.random.randn(96, 96, 96)
        cropped = preprocessor.crop_or_pad(large_volume)
        assert cropped.shape == (64, 64, 64)
    
    def test_augmenter(self):
        """Test data augmentation."""
        augmenter = VolumeAugmenter(rotation_range=10.0, flip_prob=1.0)
        volume = np.random.randn(32, 32, 32)
        mask = np.random.randint(0, 4, (32, 32, 32))
        
        aug_volume, aug_mask = augmenter(volume, mask)
        
        assert aug_volume.shape == volume.shape
        assert aug_mask.shape == mask.shape
        assert set(np.unique(aug_mask)).issubset({0, 1, 2, 3})


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_dice_score(self):
        """Test Dice score calculation."""
        dice_calculator = DiceScore(num_classes=4)
        
        # Perfect prediction
        pred = torch.randint(0, 4, (2, 32, 32, 32))
        target = pred.clone()
        
        dice = dice_calculator(pred, target)
        assert dice > 0.99, f"Perfect prediction should have Dice > 0.99, got {dice}"
    
    def test_iou_score(self):
        """Test IoU score calculation."""
        iou_calculator = IoUScore(num_classes=4)
        
        # Perfect prediction
        pred = torch.randint(0, 4, (2, 32, 32, 32))
        target = pred.clone()
        
        iou = iou_calculator(pred, target)
        assert iou > 0.99, f"Perfect prediction should have IoU > 0.99, got {iou}"
    
    def test_metrics_with_logits(self):
        """Test metrics with logit inputs."""
        dice_calculator = DiceScore(num_classes=4)
        
        # Create logits
        logits = torch.randn(2, 4, 32, 32, 32)
        pred = torch.argmax(logits, dim=1)
        target = pred.clone()
        
        dice = dice_calculator(logits, target)
        assert dice > 0.99


class TestLosses:
    """Test loss functions."""
    
    def test_dice_loss(self):
        """Test Dice loss."""
        criterion = DiceLoss(num_classes=4)
        
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.randint(0, 4, (2, 16, 16, 16))
        
        loss = criterion(pred, target)
        
        assert loss.item() >= 0.0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_focal_loss(self):
        """Test Focal loss."""
        criterion = FocalLoss()
        
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.randint(0, 4, (2, 16, 16, 16))
        
        loss = criterion(pred, target)
        
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
    
    def test_combined_loss(self):
        """Test combined loss."""
        criterion = CombinedLoss(num_classes=4)
        
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.randint(0, 4, (2, 16, 16, 16))
        
        loss = criterion(pred, target)
        
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
    
    def test_get_loss_function(self):
        """Test loss function factory."""
        loss_types = ['dice', 'ce', 'focal', 'combined', 'tversky']
        
        for loss_type in loss_types:
            criterion = get_loss_function(loss_type, num_classes=4)
            assert criterion is not None


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end(self):
        """Test end-to-end forward pass through model with loss."""
        # Create model
        model = UNet3D(n_channels=1, n_classes=4, base_channels=16)
        criterion = DiceLoss(num_classes=4)
        
        # Create dummy data
        images = torch.randn(2, 1, 32, 32, 32)
        masks = torch.randint(0, 4, (2, 32, 32, 32))
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
