"""
Training script for 3D U-Net tooth segmentation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import json
from typing import Dict, Optional
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import UNet3D
from data import CTPreprocessor, VolumeAugmenter, create_data_loaders
from utils import get_loss_function, SegmentationMetrics


class Trainer:
    """Trainer for 3D segmentation model."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 num_classes: int = 4,
                 class_names: list = None,
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs'):
        """
        Initialize trainer.
        
        Args:
            model: Segmentation model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            num_classes: Number of classes
            class_names: Names of classes
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        self.metrics = SegmentationMetrics(num_classes, self.class_names)
        
        self.best_val_dice = 0.0
        self.current_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.metrics.update(outputs.detach(), masks)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        metrics = self.metrics.compute_and_reset()
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                self.metrics.update(outputs, masks)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        metrics = self.metrics.compute_and_reset()
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_dice': self.best_val_dice
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, resume: bool = False):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume: Whether to resume from checkpoint
        """
        if resume:
            checkpoint_path = self.checkpoint_dir / 'last.pth'
            if checkpoint_path.exists():
                self.load_checkpoint('last.pth')
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            self.writer.add_scalar('Learning_rate', 
                                  self.optimizer.param_groups[0]['lr'], 
                                  epoch)
            
            # Print metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Dice: {train_metrics['dice_mean']:.4f}, "
                  f"IoU: {train_metrics['iou_mean']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Dice: {val_metrics['dice_mean']:.4f}, "
                  f"IoU: {val_metrics['iou_mean']:.4f}")
            
            # Save checkpoints
            self.save_checkpoint('last.pth', val_metrics)
            
            if val_metrics['dice_mean'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice_mean']
                self.save_checkpoint('best.pth', val_metrics)
                print(f"  New best model saved! Dice: {self.best_val_dice:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth', val_metrics)
        
        print(f"\nTraining completed! Best validation Dice: {self.best_val_dice:.4f}")
        self.writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train 3D U-Net for tooth segmentation')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--train_data', type=str, default='data/train',
                       help='Path to training data')
    parser.add_argument('--val_data', type=str, default='data/val',
                       help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of classes')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create preprocessor and augmenter
    preprocessor = CTPreprocessor(
        target_spacing=(0.1, 0.1, 0.1),
        target_size=(128, 128, 128),
        normalize=True,
        clip_range=(-1000, 3000)
    )
    
    augmenter = VolumeAugmenter(
        rotation_range=15.0,
        flip_prob=0.5,
        noise_std=0.05,
        brightness_range=0.2
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_root=args.train_data,
        val_root=args.val_data,
        batch_size=args.batch_size,
        num_workers=4,
        preprocessor=preprocessor,
        augmenter=augmenter
    )
    
    # Create model
    model = UNet3D(
        n_channels=1,
        n_classes=args.num_classes,
        base_channels=32,
        trilinear=False
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = get_loss_function(
        loss_type='combined',
        num_classes=args.num_classes,
        dice_weight=0.5,
        ce_weight=0.3,
        focal_weight=0.2
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # Class names
    class_names = ['Background', 'Enamel', 'Dentin', 'Pulpa']
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=args.num_classes,
        class_names=class_names,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs, resume=args.resume)


if __name__ == '__main__':
    main()
