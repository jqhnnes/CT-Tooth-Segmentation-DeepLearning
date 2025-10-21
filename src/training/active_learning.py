"""
Active Learning module for iterative model improvement.
Implements uncertainty-based sampling strategies.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import UNet3D


class UncertaintyEstimator:
    """Estimate prediction uncertainty using different methods."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize uncertainty estimator.
        
        Args:
            model: Trained segmentation model
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
    
    def mc_dropout_uncertainty(self, 
                              image: torch.Tensor,
                              n_iterations: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty using Monte Carlo Dropout.
        
        Args:
            image: Input image (1, C, D, H, W)
            n_iterations: Number of forward passes
            
        Returns:
            Tuple of (mean prediction, uncertainty map)
        """
        self.model.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_iterations):
                output = self.model(image)
                pred = torch.softmax(output, dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Uncertainty as variance
        uncertainty = predictions.var(dim=0).sum(dim=1)  # Sum over classes
        
        self.model.eval()
        
        return mean_pred, uncertainty
    
    def entropy_uncertainty(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty using prediction entropy.
        
        Args:
            image: Input image (1, C, D, H, W)
            
        Returns:
            Tuple of (prediction, entropy map)
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(image)
            pred = torch.softmax(output, dim=1)
            
            # Calculate entropy
            entropy = -torch.sum(pred * torch.log(pred + 1e-10), dim=1)
        
        return pred, entropy
    
    def margin_uncertainty(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty using margin sampling.
        
        Args:
            image: Input image (1, C, D, H, W)
            
        Returns:
            Tuple of (prediction, margin map)
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(image)
            pred = torch.softmax(output, dim=1)
            
            # Get top 2 probabilities
            top2_probs, _ = torch.topk(pred, k=2, dim=1)
            
            # Margin is difference between top 2
            margin = top2_probs[:, 0] - top2_probs[:, 1]
            
            # Uncertainty is inverse of margin
            uncertainty = 1.0 - margin
        
        return pred, uncertainty


class ActiveLearner:
    """Active Learning framework for iterative data selection."""
    
    def __init__(self,
                 model: nn.Module,
                 unlabeled_pool: List[str],
                 device: str = 'cuda',
                 uncertainty_method: str = 'entropy'):
        """
        Initialize active learner.
        
        Args:
            model: Trained segmentation model
            unlabeled_pool: List of unlabeled data file paths
            device: Device to use
            uncertainty_method: Method for uncertainty estimation
                ('entropy', 'mc_dropout', 'margin')
        """
        self.model = model
        self.unlabeled_pool = unlabeled_pool
        self.device = device
        self.uncertainty_method = uncertainty_method
        
        self.estimator = UncertaintyEstimator(model, device)
        self.uncertainty_scores = {}
    
    def compute_uncertainties(self, 
                            preprocessor,
                            batch_size: int = 1) -> Dict[str, float]:
        """
        Compute uncertainty scores for unlabeled pool.
        
        Args:
            preprocessor: Data preprocessor
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping file paths to uncertainty scores
        """
        print(f"Computing uncertainties for {len(self.unlabeled_pool)} samples...")
        
        self.uncertainty_scores = {}
        
        for file_path in tqdm(self.unlabeled_pool):
            # Load and preprocess image
            image = np.load(file_path).astype(np.float32)
            image = preprocessor.preprocess(image)
            
            # Add batch and channel dimensions
            if image.ndim == 3:
                image = image[np.newaxis, np.newaxis, ...]
            
            image = torch.from_numpy(image).to(self.device)
            
            # Compute uncertainty
            if self.uncertainty_method == 'entropy':
                _, uncertainty = self.estimator.entropy_uncertainty(image)
            elif self.uncertainty_method == 'mc_dropout':
                _, uncertainty = self.estimator.mc_dropout_uncertainty(image)
            elif self.uncertainty_method == 'margin':
                _, uncertainty = self.estimator.margin_uncertainty(image)
            else:
                raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
            
            # Average uncertainty over volume
            mean_uncertainty = uncertainty.mean().item()
            self.uncertainty_scores[file_path] = mean_uncertainty
        
        return self.uncertainty_scores
    
    def select_samples(self, 
                      n_samples: int,
                      strategy: str = 'uncertainty') -> List[str]:
        """
        Select samples for labeling.
        
        Args:
            n_samples: Number of samples to select
            strategy: Selection strategy ('uncertainty', 'random')
            
        Returns:
            List of selected file paths
        """
        if strategy == 'uncertainty':
            # Sort by uncertainty (highest first)
            sorted_samples = sorted(
                self.uncertainty_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            selected = [path for path, _ in sorted_samples[:n_samples]]
        
        elif strategy == 'random':
            # Random sampling
            indices = np.random.choice(
                len(self.unlabeled_pool),
                size=min(n_samples, len(self.unlabeled_pool)),
                replace=False
            )
            selected = [self.unlabeled_pool[i] for i in indices]
        
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        return selected
    
    def update_pool(self, labeled_samples: List[str]):
        """
        Remove labeled samples from unlabeled pool.
        
        Args:
            labeled_samples: List of newly labeled sample paths
        """
        self.unlabeled_pool = [
            path for path in self.unlabeled_pool 
            if path not in labeled_samples
        ]
        
        # Remove from uncertainty scores
        for path in labeled_samples:
            if path in self.uncertainty_scores:
                del self.uncertainty_scores[path]
    
    def save_state(self, output_path: str):
        """
        Save active learning state.
        
        Args:
            output_path: Path to save state
        """
        state = {
            'unlabeled_pool': self.unlabeled_pool,
            'uncertainty_scores': self.uncertainty_scores,
            'uncertainty_method': self.uncertainty_method
        }
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Active learning state saved to {output_path}")
    
    def load_state(self, input_path: str):
        """
        Load active learning state.
        
        Args:
            input_path: Path to load state from
        """
        with open(input_path, 'r') as f:
            state = json.load(f)
        
        self.unlabeled_pool = state['unlabeled_pool']
        self.uncertainty_scores = state['uncertainty_scores']
        self.uncertainty_method = state['uncertainty_method']
        
        print(f"Active learning state loaded from {input_path}")


class ActiveLearningLoop:
    """Complete active learning loop."""
    
    def __init__(self,
                 initial_model: nn.Module,
                 train_function,
                 unlabeled_pool: List[str],
                 device: str = 'cuda'):
        """
        Initialize active learning loop.
        
        Args:
            initial_model: Initial trained model
            train_function: Function to train model on new data
            unlabeled_pool: Initial unlabeled data pool
            device: Device to use
        """
        self.model = initial_model
        self.train_function = train_function
        self.device = device
        
        self.learner = ActiveLearner(
            model=self.model,
            unlabeled_pool=unlabeled_pool,
            device=device,
            uncertainty_method='entropy'
        )
        
        self.iteration = 0
        self.labeled_samples = []
    
    def run_iteration(self,
                     preprocessor,
                     n_samples: int = 10,
                     selection_strategy: str = 'uncertainty') -> List[str]:
        """
        Run one iteration of active learning.
        
        Args:
            preprocessor: Data preprocessor
            n_samples: Number of samples to select
            selection_strategy: Selection strategy
            
        Returns:
            Selected samples for labeling
        """
        print(f"\n=== Active Learning Iteration {self.iteration} ===")
        print(f"Unlabeled pool size: {len(self.learner.unlabeled_pool)}")
        
        # Compute uncertainties
        self.learner.compute_uncertainties(preprocessor)
        
        # Select samples
        selected = self.learner.select_samples(n_samples, selection_strategy)
        
        print(f"Selected {len(selected)} samples for labeling")
        print("Top 5 samples with highest uncertainty:")
        sorted_uncertain = sorted(
            self.learner.uncertainty_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for path, score in sorted_uncertain:
            print(f"  {Path(path).name}: {score:.4f}")
        
        self.iteration += 1
        
        return selected
    
    def update_with_labels(self, labeled_samples: List[str]):
        """
        Update after samples are labeled.
        
        Args:
            labeled_samples: Newly labeled sample paths
        """
        self.labeled_samples.extend(labeled_samples)
        self.learner.update_pool(labeled_samples)
        
        print(f"Total labeled samples: {len(self.labeled_samples)}")
        print(f"Remaining unlabeled: {len(self.learner.unlabeled_pool)}")
    
    def retrain_model(self, new_training_data: List[str]):
        """
        Retrain model with new labeled data.
        
        Args:
            new_training_data: Paths to new training data
        """
        print("\nRetraining model with new labeled data...")
        
        # Use provided train function
        self.model = self.train_function(new_training_data)
        
        # Update learner with new model
        self.learner.model = self.model
        self.learner.estimator = UncertaintyEstimator(self.model, self.device)
        
        print("Model retraining completed")


if __name__ == "__main__":
    # Test active learning components
    print("Active learning module initialized")
    
    # Create dummy model
    model = UNet3D(n_channels=1, n_classes=4, base_channels=32)
    
    # Create uncertainty estimator
    estimator = UncertaintyEstimator(model, device='cpu')
    
    # Test uncertainty estimation
    dummy_image = torch.randn(1, 1, 32, 32, 32)
    pred, uncertainty = estimator.entropy_uncertainty(dummy_image)
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Mean uncertainty: {uncertainty.mean().item():.4f}")
