"""
Loss functions for 3D segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, num_classes: int = 4, smooth: float = 1e-6, weight: torch.Tensor = None):
        """
        Initialize Dice Loss.
        
        Args:
            num_classes: Number of classes
            smooth: Smoothing factor
            weight: Class weights
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            pred: Predicted logits (B, C, D, H, W)
            target: Ground truth classes (B, D, H, W)
            
        Returns:
            Dice loss
        """
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Calculate Dice per class
        dice_losses = []
        for c in range(self.num_classes):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1. - dice
            
            if self.weight is not None:
                dice_loss = dice_loss * self.weight[c]
            
            dice_losses.append(dice_loss)
        
        return torch.stack(dice_losses).mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss.
        
        Args:
            pred: Predicted logits (B, C, D, H, W)
            target: Ground truth classes (B, D, H, W)
            
        Returns:
            Focal loss
        """
        # Calculate cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Calculate pt
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss: Weighted sum of multiple losses."""
    
    def __init__(self, 
                 num_classes: int = 4,
                 dice_weight: float = 0.5,
                 ce_weight: float = 0.3,
                 focal_weight: float = 0.2,
                 class_weights: torch.Tensor = None):
        """
        Initialize combined loss.
        
        Args:
            num_classes: Number of classes
            dice_weight: Weight for Dice loss
            ce_weight: Weight for Cross Entropy loss
            focal_weight: Weight for Focal loss
            class_weights: Weights for each class
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(num_classes=num_classes, weight=class_weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted logits (B, C, D, H, W)
            target: Ground truth classes (B, D, H, W)
            
        Returns:
            Combined loss
        """
        loss = 0.0
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(pred, target)
        
        if self.ce_weight > 0:
            loss += self.ce_weight * self.ce_loss(pred, target)
        
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(pred, target)
        
        return loss


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice loss."""
    
    def __init__(self, 
                 num_classes: int = 4,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 smooth: float = 1e-6):
        """
        Initialize Tversky Loss.
        
        Args:
            num_classes: Number of classes
            alpha: Weight of false positives
            beta: Weight of false negatives
            smooth: Smoothing factor
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Tversky loss.
        
        Args:
            pred: Predicted logits (B, C, D, H, W)
            target: Ground truth classes (B, D, H, W)
            
        Returns:
            Tversky loss
        """
        pred = F.softmax(pred, dim=1)
        
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        tversky_losses = []
        for c in range(self.num_classes):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]
            
            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1 - target_c)).sum()
            fn = ((1 - pred_c) * target_c).sum()
            
            tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky_loss = 1. - tversky_index
            
            tversky_losses.append(tversky_loss)
        
        return torch.stack(tversky_losses).mean()


class BoundaryLoss(nn.Module):
    """Boundary Loss for better boundary segmentation."""
    
    def __init__(self, num_classes: int = 4):
        """
        Initialize Boundary Loss.
        
        Args:
            num_classes: Number of classes
        """
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate boundary loss.
        
        Args:
            pred: Predicted logits (B, C, D, H, W)
            target: Ground truth classes (B, D, H, W)
            
        Returns:
            Boundary loss
        """
        pred = F.softmax(pred, dim=1)
        
        # Calculate gradients to find boundaries
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                               dtype=pred.dtype, device=pred.device)
        sobel_y = sobel_x.transpose(1, 2)
        sobel_z = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                               dtype=pred.dtype, device=pred.device)
        
        loss = 0.0
        for c in range(self.num_classes):
            pred_c = pred[:, c:c+1]
            target_c = (target == c).float().unsqueeze(1)
            
            # Compute gradients
            grad_pred_x = F.conv3d(pred_c, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
            grad_pred_y = F.conv3d(pred_c, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
            grad_pred_z = F.conv3d(pred_c, sobel_z.unsqueeze(0).unsqueeze(0), padding=1)
            
            grad_target_x = F.conv3d(target_c, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
            grad_target_y = F.conv3d(target_c, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
            grad_target_z = F.conv3d(target_c, sobel_z.unsqueeze(0).unsqueeze(0), padding=1)
            
            # Calculate boundary loss
            loss += F.mse_loss(grad_pred_x, grad_target_x)
            loss += F.mse_loss(grad_pred_y, grad_target_y)
            loss += F.mse_loss(grad_pred_z, grad_target_z)
        
        return loss / self.num_classes


def get_loss_function(loss_type: str = 'combined', **kwargs) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: Type of loss ('dice', 'ce', 'focal', 'combined', 'tversky', 'boundary')
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function module
    """
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'boundary':
        return BoundaryLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test losses
    batch_size = 2
    num_classes = 4
    
    pred = torch.randn(batch_size, num_classes, 32, 32, 32)
    target = torch.randint(0, num_classes, (batch_size, 32, 32, 32))
    
    # Test different losses
    dice_loss = DiceLoss(num_classes=num_classes)
    focal_loss = FocalLoss()
    combined_loss = CombinedLoss(num_classes=num_classes)
    tversky_loss = TverskyLoss(num_classes=num_classes)
    
    print(f"Dice Loss: {dice_loss(pred, target).item():.4f}")
    print(f"Focal Loss: {focal_loss(pred, target).item():.4f}")
    print(f"Combined Loss: {combined_loss(pred, target).item():.4f}")
    print(f"Tversky Loss: {tversky_loss(pred, target).item():.4f}")
