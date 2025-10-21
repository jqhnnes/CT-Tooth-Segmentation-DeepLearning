"""
Evaluation script for 3D tooth segmentation.
Evaluates model on internal and external test sets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import UNet3D
from data import ToothDataset, InferenceDataset, CTPreprocessor
from utils import SegmentationMetrics, HausdorffDistance, VolumeMetrics


class Evaluator:
    """Evaluator for 3D segmentation model."""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 num_classes: int = 4,
                 class_names: list = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained segmentation model
            device: Device to use
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        self.metrics = SegmentationMetrics(num_classes, self.class_names)
        self.hausdorff = HausdorffDistance()
        self.volume_metrics = VolumeMetrics()
    
    def evaluate_batch(self, batch: Dict) -> Dict[str, float]:
        """Evaluate a single batch."""
        images = batch['image'].to(self.device)
        masks = batch['mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
        
        # Update metrics
        self.metrics.update(outputs, masks)
        
        return {}
    
    def evaluate_dataset(self, data_loader: DataLoader, dataset_name: str = 'test') -> Dict[str, float]:
        """
        Evaluate on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating on {dataset_name} dataset...")
        
        self.metrics.reset()
        all_results = []
        
        pbar = tqdm(data_loader, desc=f"Evaluating {dataset_name}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            filenames = batch['filename']
            
            with torch.no_grad():
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            self.metrics.update(outputs, masks)
            
            # Calculate per-sample metrics
            for i in range(len(filenames)):
                pred_i = predictions[i].cpu().numpy()
                mask_i = masks[i].cpu().numpy()
                
                sample_metrics = {
                    'filename': filenames[i],
                    'dataset': dataset_name
                }
                
                # Per-class metrics
                for c, class_name in enumerate(self.class_names):
                    pred_c = (pred_i == c)
                    mask_c = (mask_i == c)
                    
                    if mask_c.sum() > 0:  # Only calculate if class exists
                        # Volume similarity
                        vol_sim = self.volume_metrics.volume_similarity(pred_c, mask_c)
                        sample_metrics[f'{class_name}_vol_similarity'] = vol_sim
                        
                        # Precision/Recall
                        precision = self.volume_metrics.precision(pred_c, mask_c)
                        recall = self.volume_metrics.recall(pred_c, mask_c)
                        sample_metrics[f'{class_name}_precision'] = precision
                        sample_metrics[f'{class_name}_recall'] = recall
                
                all_results.append(sample_metrics)
        
        # Compute overall metrics
        metrics = self.metrics.compute_and_reset()
        
        return metrics, all_results
    
    def save_results(self, 
                    results: Dict[str, float],
                    detailed_results: List[Dict],
                    output_dir: Path):
        """
        Save evaluation results.
        
        Args:
            results: Overall metrics
            detailed_results: Per-sample results
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save overall metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed results
        df = pd.DataFrame(detailed_results)
        df.to_csv(output_dir / 'detailed_results.csv', index=False)
        
        print(f"\nResults saved to {output_dir}")
        
        # Print summary
        print("\n=== Evaluation Results ===")
        print(f"Mean Dice Score: {results['dice_mean']:.4f}")
        print(f"Mean IoU Score: {results['iou_mean']:.4f}")
        print("\nPer-class Dice Scores:")
        for class_name in self.class_names:
            dice_key = f'dice_{class_name}'
            if dice_key in results:
                print(f"  {class_name}: {results[dice_key]:.4f}")
        
        print("\nPer-class IoU Scores:")
        for class_name in self.class_names:
            iou_key = f'iou_{class_name}'
            if iou_key in results:
                print(f"  {class_name}: {results[iou_key]:.4f}")
    
    def plot_results(self, results: Dict[str, float], output_dir: Path):
        """Plot evaluation results."""
        output_dir = Path(output_dir)
        
        # Plot per-class metrics
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Dice scores
        dice_scores = [results[f'dice_{name}'] for name in self.class_names 
                      if f'dice_{name}' in results]
        axes[0].bar(self.class_names, dice_scores)
        axes[0].set_ylabel('Dice Score')
        axes[0].set_title('Per-Class Dice Scores')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # IoU scores
        iou_scores = [results[f'iou_{name}'] for name in self.class_names
                     if f'iou_{name}' in results]
        axes[1].bar(self.class_names, iou_scores)
        axes[1].set_ylabel('IoU Score')
        axes[1].set_title('Per-Class IoU Scores')
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_plot.png', dpi=150)
        plt.close()
        
        print(f"Plots saved to {output_dir}")


def predict_and_save(model: nn.Module,
                    data_loader: DataLoader,
                    output_dir: Path,
                    device: str = 'cuda'):
    """
    Generate predictions and save them.
    
    Args:
        model: Trained model
        data_loader: Data loader
        output_dir: Output directory
        device: Device to use
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating predictions...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch['image'].to(device)
            filenames = batch['filename']
            
            # Predict
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Save predictions
            for i, filename in enumerate(filenames):
                pred = predictions[i]
                np.save(output_dir / f'{filename}_pred.npy', pred)
    
    print(f"Predictions saved to {output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate 3D U-Net for tooth segmentation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--internal_test', type=str, default='data/test_internal',
                       help='Path to internal test data')
    parser.add_argument('--external_test', type=str, default='data/test_external',
                       help='Path to external test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of classes')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction masks')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create preprocessor
    preprocessor = CTPreprocessor(
        target_spacing=(0.1, 0.1, 0.1),
        target_size=(128, 128, 128),
        normalize=True,
        clip_range=(-1000, 3000)
    )
    
    # Create model
    model = UNet3D(
        n_channels=1,
        n_classes=args.num_classes,
        base_channels=32,
        trilinear=False
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Class names
    class_names = ['Background', 'Enamel', 'Dentin', 'Pulpa']
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=args.num_classes,
        class_names=class_names
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on internal test set
    if Path(args.internal_test).exists():
        internal_dataset = ToothDataset(
            data_root=args.internal_test,
            split='test',
            preprocessor=preprocessor
        )
        
        internal_loader = DataLoader(
            internal_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        internal_metrics, internal_detailed = evaluator.evaluate_dataset(
            internal_loader, 
            'internal_test'
        )
        
        # Save results
        evaluator.save_results(
            internal_metrics,
            internal_detailed,
            output_dir / 'internal_test'
        )
        
        evaluator.plot_results(internal_metrics, output_dir / 'internal_test')
        
        # Save predictions if requested
        if args.save_predictions:
            predict_and_save(
                model,
                internal_loader,
                output_dir / 'internal_test' / 'predictions',
                device
            )
    
    # Evaluate on external test set
    if Path(args.external_test).exists():
        external_dataset = ToothDataset(
            data_root=args.external_test,
            split='test',
            preprocessor=preprocessor
        )
        
        external_loader = DataLoader(
            external_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        external_metrics, external_detailed = evaluator.evaluate_dataset(
            external_loader,
            'external_test'
        )
        
        # Save results
        evaluator.save_results(
            external_metrics,
            external_detailed,
            output_dir / 'external_test'
        )
        
        evaluator.plot_results(external_metrics, output_dir / 'external_test')
        
        # Save predictions if requested
        if args.save_predictions:
            predict_and_save(
                model,
                external_loader,
                output_dir / 'external_test' / 'predictions',
                device
            )
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
