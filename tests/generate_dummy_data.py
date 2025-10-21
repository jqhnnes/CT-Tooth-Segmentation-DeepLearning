"""
Script to generate dummy data for testing the pipeline.
"""

import numpy as np
from pathlib import Path
import json


def generate_dummy_tooth_data(output_dir: str, num_samples: int = 5):
    """
    Generate dummy µCT tooth data for testing.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
    """
    output_dir = Path(output_dir)
    
    # Create directories
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'masks').mkdir(parents=True, exist_ok=True)
    
    metadata = {}
    
    for i in range(num_samples):
        sample_id = f'tooth_{i:03d}'
        
        # Generate synthetic volume (simulating µCT scan)
        # Size varies to test preprocessing
        size = np.random.randint(80, 120, 3)
        
        # Create base volume with gaussian noise
        volume = np.random.randn(*size) * 500 + 1000
        
        # Add tooth-like structures
        center = size // 2
        radius = min(size) // 3
        
        # Create synthetic tooth structure
        z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
        dist_from_center = np.sqrt((z - center[0])**2 + 
                                   (y - center[1])**2 + 
                                   (x - center[2])**2)
        
        # Enamel (outer layer)
        enamel_mask = (dist_from_center > radius * 0.7) & (dist_from_center < radius * 0.9)
        volume[enamel_mask] += 1500
        
        # Dentin (middle layer)
        dentin_mask = (dist_from_center > radius * 0.3) & (dist_from_center <= radius * 0.7)
        volume[dentin_mask] += 800
        
        # Pulpa (core)
        pulpa_mask = dist_from_center <= radius * 0.3
        volume[pulpa_mask] += 200
        
        # Create segmentation mask
        mask = np.zeros(size, dtype=np.int64)
        mask[enamel_mask] = 1  # Enamel
        mask[dentin_mask] = 2  # Dentin
        mask[pulpa_mask] = 3   # Pulpa
        
        # Save
        np.save(output_dir / 'images' / f'{sample_id}.npy', volume.astype(np.float32))
        np.save(output_dir / 'masks' / f'{sample_id}.npy', mask)
        
        # Add metadata
        metadata[sample_id] = {
            'spacing': [0.15, 0.15, 0.15],
            'original_size': size.tolist()
        }
        
        print(f"Generated {sample_id}: shape={size}, "
              f"classes={np.unique(mask).tolist()}")
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGenerated {num_samples} dummy samples in {output_dir}")
    
    # Print statistics
    print("\nClass distribution across all samples:")
    all_masks = [np.load(output_dir / 'masks' / f'tooth_{i:03d}.npy') 
                 for i in range(num_samples)]
    
    class_names = ['Background', 'Enamel', 'Dentin', 'Pulpa']
    for cls in range(4):
        total_voxels = sum(np.sum(mask == cls) for mask in all_masks)
        print(f"  {class_names[cls]}: {total_voxels} voxels")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dummy tooth data')
    parser.add_argument('--output_dir', type=str, default='data/dummy',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    generate_dummy_tooth_data(args.output_dir, args.num_samples)
