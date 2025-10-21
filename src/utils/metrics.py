"""
Evaluation metrics for 3D segmentation.
Implements Dice Score, IoU, and other relevant metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class DiceScore:
    """Dice Score (F1 Score) for segmentation."""
    
    def __init__(self, num_classes: int = 4, smooth: float = 1e-6):
        """
        Initialize Dice Score calculator.
        
        Args:
            num_classes: Number of classes
            smooth: Smoothing factor to avoid division by zero
        """
        self.num_classes = num_classes
        self.smooth = smooth
    
    def __call__(self, 
                 pred: torch.Tensor, 
                 target: torch.Tensor,
                 per_class: bool = False) -> torch.Tensor:
        """
        Calculate Dice score.
        
        Args:
            pred: Predicted logits (B, C, D, H, W) or predicted classes (B, D, H, W)
            target: Ground truth classes (B, D, H, W)
            per_class: Return per-class scores
            
        Returns:
            Dice score (scalar or per-class tensor)
        """
        # Convert logits to predictions if needed
        if pred.ndim == 5:  # (B, C, D, H, W)
            pred = torch.argmax(pred, dim=1)  # (B, D, H, W)
        
        # One-hot encode
        pred_one_hot = F.one_hot(pred, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        
        # Calculate Dice per class
        dice_scores = []
        for c in range(self.num_classes):
            pred_c = pred_one_hot[:, c].float()
            target_c = target_one_hot[:, c].float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores)
        
        if per_class:
            return dice_scores
        else:
            return dice_scores.mean()


class IoUScore:
    """Intersection over Union (Jaccard Index) for segmentation."""
    
    def __init__(self, num_classes: int = 4, smooth: float = 1e-6):
        """
        Initialize IoU calculator.
        
        Args:
            num_classes: Number of classes
            smooth: Smoothing factor
        """
        self.num_classes = num_classes
        self.smooth = smooth
    
    def __call__(self,
                 pred: torch.Tensor,
                 target: torch.Tensor,
                 per_class: bool = False) -> torch.Tensor:
        """
        Calculate IoU score.
        
        Args:
            pred: Predicted logits (B, C, D, H, W) or predicted classes (B, D, H, W)
            target: Ground truth classes (B, D, H, W)
            per_class: Return per-class scores
            
        Returns:
            IoU score (scalar or per-class tensor)
        """
        # Convert logits to predictions if needed
        if pred.ndim == 5:  # (B, C, D, H, W)
            pred = torch.argmax(pred, dim=1)
        
        # One-hot encode
        pred_one_hot = F.one_hot(pred, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        
        # Calculate IoU per class
        iou_scores = []
        for c in range(self.num_classes):
            pred_c = pred_one_hot[:, c].float()
            target_c = target_one_hot[:, c].float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_scores.append(iou)
        
        iou_scores = torch.stack(iou_scores)
        
        if per_class:
            return iou_scores
        else:
            return iou_scores.mean()


class SegmentationMetrics:
    """Collection of segmentation metrics."""
    
    def __init__(self, 
                 num_classes: int = 4,
                 class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        self.dice_score = DiceScore(num_classes)
        self.iou_score = IoUScore(num_classes)
        
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.dice_sum = torch.zeros(self.num_classes)
        self.iou_sum = torch.zeros(self.num_classes)
        self.count = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch.
        
        Args:
            pred: Predictions (B, C, D, H, W) or (B, D, H, W)
            target: Ground truth (B, D, H, W)
        """
        dice = self.dice_score(pred, target, per_class=True)
        iou = self.iou_score(pred, target, per_class=True)
        
        self.dice_sum += dice.cpu()
        self.iou_sum += iou.cpu()
        self.count += 1
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metric values
        """
        if self.count == 0:
            return {}
        
        dice_mean = self.dice_sum / self.count
        iou_mean = self.iou_sum / self.count
        
        metrics = {
            'dice_mean': dice_mean.mean().item(),
            'iou_mean': iou_mean.mean().item()
        }
        
        # Per-class metrics
        for i, name in enumerate(self.class_names):
            metrics[f'dice_{name}'] = dice_mean[i].item()
            metrics[f'iou_{name}'] = iou_mean[i].item()
        
        return metrics
    
    def compute_and_reset(self) -> Dict[str, float]:
        """Compute metrics and reset."""
        metrics = self.compute()
        self.reset()
        return metrics


class HausdorffDistance:
    """Hausdorff Distance for segmentation evaluation."""
    
    def __init__(self, percentile: int = 95):
        """
        Initialize Hausdorff Distance calculator.
        
        Args:
            percentile: Percentile for robust Hausdorff (default: 95)
        """
        self.percentile = percentile
    
    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate Hausdorff distance.
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            Hausdorff distance
        """
        from scipy.spatial.distance import directed_hausdorff
        
        # Get coordinates of foreground pixels
        pred_points = np.argwhere(pred)
        target_points = np.argwhere(target)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        d1 = directed_hausdorff(pred_points, target_points)[0]
        d2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(d1, d2)


class VolumeMetrics:
    """Volume-based metrics for segmentation."""
    
    @staticmethod
    def volume_similarity(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate volume similarity.
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            Volume similarity score
        """
        pred_vol = np.sum(pred)
        target_vol = np.sum(target)
        
        if target_vol == 0:
            return 0.0
        
        return 1.0 - abs(pred_vol - target_vol) / target_vol
    
    @staticmethod
    def precision(pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate precision."""
        tp = np.sum(pred & target)
        fp = np.sum(pred & ~target)
        
        if tp + fp == 0:
            return 0.0
        
        return tp / (tp + fp)
    
    @staticmethod
    def recall(pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate recall (sensitivity)."""
        tp = np.sum(pred & target)
        fn = np.sum(~pred & target)
        
        if tp + fn == 0:
            return 0.0
        
        return tp / (tp + fn)
    
    @staticmethod
    def specificity(pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate specificity."""
        tn = np.sum(~pred & ~target)
        fp = np.sum(pred & ~target)
        
        if tn + fp == 0:
            return 0.0
        
        return tn / (tn + fp)


def compute_all_metrics(pred: torch.Tensor, 
                       target: torch.Tensor,
                       num_classes: int = 4,
                       class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute all metrics for a prediction.
    
    Args:
        pred: Predictions (B, C, D, H, W) or (B, D, H, W)
        target: Ground truth (B, D, H, W)
        num_classes: Number of classes
        class_names: Names of classes
        
    Returns:
        Dictionary of all metrics
    """
    metrics = SegmentationMetrics(num_classes, class_names)
    metrics.update(pred, target)
    return metrics.compute()


if __name__ == "__main__":
    # Test metrics
    num_classes = 4
    batch_size = 2
    
    # Create dummy predictions and targets
    pred = torch.randn(batch_size, num_classes, 32, 32, 32)
    target = torch.randint(0, num_classes, (batch_size, 32, 32, 32))
    
    # Calculate metrics
    metrics = compute_all_metrics(
        pred, target,
        num_classes=num_classes,
        class_names=['Background', 'Enamel', 'Dentin', 'Pulpa']
    )
    
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
