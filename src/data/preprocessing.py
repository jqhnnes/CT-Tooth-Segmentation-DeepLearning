"""
Data preprocessing utilities for µCT tooth scans.
Handles loading, normalization, and augmentation of 3D volumetric data.
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, Union
import torch
from scipy import ndimage


class CTPreprocessor:
    """Preprocessor for µCT scan data."""
    
    def __init__(self, 
                 target_spacing: Tuple[float, float, float] = (0.1, 0.1, 0.1),
                 target_size: Optional[Tuple[int, int, int]] = None,
                 normalize: bool = True,
                 clip_range: Optional[Tuple[float, float]] = None):
        """
        Initialize preprocessor.
        
        Args:
            target_spacing: Target voxel spacing in mm (z, y, x)
            target_size: Target volume size in voxels (D, H, W)
            normalize: Whether to normalize intensity values
            clip_range: Intensity clipping range (min, max)
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.normalize = normalize
        self.clip_range = clip_range
    
    def load_nifti(self, path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load NIfTI file.
        
        Args:
            path: Path to .nii or .nii.gz file
            
        Returns:
            Tuple of (image data, affine matrix)
        """
        img = nib.load(str(path))
        data = img.get_fdata()
        affine = img.affine
        return data, affine
    
    def load_dicom_series(self, directory: Union[str, Path]) -> np.ndarray:
        """
        Load DICOM series from directory.
        
        Args:
            directory: Directory containing DICOM files
            
        Returns:
            3D volume as numpy array
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        volume = sitk.GetArrayFromImage(image)
        return volume
    
    def resample_volume(self, 
                       volume: np.ndarray, 
                       original_spacing: Tuple[float, float, float],
                       target_spacing: Optional[Tuple[float, float, float]] = None,
                       order: int = 3) -> np.ndarray:
        """
        Resample volume to target spacing.
        
        Args:
            volume: Input 3D volume
            original_spacing: Original voxel spacing
            target_spacing: Target voxel spacing
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
            
        Returns:
            Resampled volume
        """
        if target_spacing is None:
            target_spacing = self.target_spacing
        
        # Calculate zoom factors
        zoom_factors = np.array(original_spacing) / np.array(target_spacing)
        
        # Resample
        resampled = ndimage.zoom(volume, zoom_factors, order=order)
        return resampled
    
    def normalize_intensity(self, 
                           volume: np.ndarray,
                           method: str = 'zscore',
                           clip_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Normalize intensity values.
        
        Args:
            volume: Input volume
            method: Normalization method ('zscore', 'minmax', 'percentile')
            clip_range: Optional intensity clipping range
            
        Returns:
            Normalized volume
        """
        volume = volume.astype(np.float32)
        
        if clip_range is not None:
            volume = np.clip(volume, clip_range[0], clip_range[1])
        elif self.clip_range is not None:
            volume = np.clip(volume, self.clip_range[0], self.clip_range[1])
        
        if method == 'zscore':
            mean = np.mean(volume)
            std = np.std(volume)
            volume = (volume - mean) / (std + 1e-8)
        elif method == 'minmax':
            vmin, vmax = np.min(volume), np.max(volume)
            volume = (volume - vmin) / (vmax - vmin + 1e-8)
        elif method == 'percentile':
            p1, p99 = np.percentile(volume, [1, 99])
            volume = np.clip(volume, p1, p99)
            volume = (volume - p1) / (p99 - p1 + 1e-8)
        
        return volume
    
    def crop_or_pad(self, 
                    volume: np.ndarray,
                    target_size: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Crop or pad volume to target size.
        
        Args:
            volume: Input volume
            target_size: Target size (D, H, W)
            
        Returns:
            Cropped/padded volume
        """
        if target_size is None:
            target_size = self.target_size
        
        if target_size is None:
            return volume
        
        current_size = volume.shape
        result = np.zeros(target_size, dtype=volume.dtype)
        
        # Calculate crop/pad for each dimension
        slices = []
        result_slices = []
        
        for current, target in zip(current_size, target_size):
            if current > target:
                # Crop
                start = (current - target) // 2
                slices.append(slice(start, start + target))
                result_slices.append(slice(0, target))
            else:
                # Pad
                start = (target - current) // 2
                slices.append(slice(0, current))
                result_slices.append(slice(start, start + current))
        
        result[result_slices[0], result_slices[1], result_slices[2]] = \
            volume[slices[0], slices[1], slices[2]]
        
        return result
    
    def preprocess(self, 
                   volume: np.ndarray,
                   original_spacing: Optional[Tuple[float, float, float]] = None,
                   mask: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Full preprocessing pipeline.
        
        Args:
            volume: Input volume
            original_spacing: Original voxel spacing
            mask: Optional segmentation mask
            
        Returns:
            Preprocessed volume (and mask if provided)
        """
        # Resample if spacing provided
        if original_spacing is not None:
            volume = self.resample_volume(volume, original_spacing, order=3)
            if mask is not None:
                mask = self.resample_volume(mask, original_spacing, order=0)
        
        # Normalize intensity
        if self.normalize:
            volume = self.normalize_intensity(volume)
        
        # Crop or pad to target size
        if self.target_size is not None:
            volume = self.crop_or_pad(volume)
            if mask is not None:
                mask = self.crop_or_pad(mask)
        
        if mask is not None:
            return volume, mask
        return volume


class VolumeAugmenter:
    """Data augmentation for 3D volumes."""
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 flip_prob: float = 0.5,
                 noise_std: float = 0.1,
                 brightness_range: float = 0.2,
                 elastic_alpha: float = 10.0,
                 elastic_sigma: float = 4.0):
        """
        Initialize augmenter.
        
        Args:
            rotation_range: Max rotation angle in degrees
            flip_prob: Probability of flipping
            noise_std: Standard deviation of Gaussian noise
            brightness_range: Brightness adjustment range
            elastic_alpha: Elastic deformation alpha parameter
            elastic_sigma: Elastic deformation sigma parameter
        """
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    def random_rotation(self, 
                       volume: np.ndarray, 
                       mask: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply random 3D rotation."""
        angles = np.random.uniform(-self.rotation_range, self.rotation_range, 3)
        
        # Rotate around each axis
        for axis, angle in enumerate(angles):
            axes = (axis, (axis + 1) % 3)
            volume = ndimage.rotate(volume, angle, axes=axes, reshape=False, order=3)
            if mask is not None:
                mask = ndimage.rotate(mask, angle, axes=axes, reshape=False, order=0)
        
        if mask is not None:
            return volume, mask
        return volume
    
    def random_flip(self, 
                   volume: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply random flips along axes."""
        for axis in range(3):
            if np.random.rand() < self.flip_prob:
                volume = np.flip(volume, axis=axis).copy()
                if mask is not None:
                    mask = np.flip(mask, axis=axis).copy()
        
        if mask is not None:
            return volume, mask
        return volume
    
    def add_gaussian_noise(self, volume: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.noise_std, volume.shape)
        return volume + noise
    
    def adjust_brightness(self, volume: np.ndarray) -> np.ndarray:
        """Adjust brightness."""
        factor = 1.0 + np.random.uniform(-self.brightness_range, self.brightness_range)
        return volume * factor
    
    def elastic_deformation(self,
                          volume: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply elastic deformation."""
        shape = volume.shape
        
        # Generate random displacement fields
        dx = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma, mode="constant", cval=0
        ) * self.elastic_alpha
        dy = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma, mode="constant", cval=0
        ) * self.elastic_alpha
        dz = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma, mode="constant", cval=0
        ) * self.elastic_alpha
        
        # Create meshgrid
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        # Apply displacement
        indices = [
            np.clip(z + dz, 0, shape[0] - 1).ravel(),
            np.clip(y + dy, 0, shape[1] - 1).ravel(),
            np.clip(x + dx, 0, shape[2] - 1).ravel()
        ]
        
        volume_deformed = ndimage.map_coordinates(volume, indices, order=1).reshape(shape)
        
        if mask is not None:
            mask_deformed = ndimage.map_coordinates(mask, indices, order=0).reshape(shape)
            return volume_deformed, mask_deformed
        
        return volume_deformed
    
    def __call__(self, 
                volume: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply random augmentations."""
        # Geometric augmentations (applied to both volume and mask)
        if np.random.rand() < 0.5:
            if mask is not None:
                volume, mask = self.random_rotation(volume, mask)
            else:
                volume = self.random_rotation(volume)
        
        if np.random.rand() < 0.5:
            if mask is not None:
                volume, mask = self.random_flip(volume, mask)
            else:
                volume = self.random_flip(volume)
        
        if np.random.rand() < 0.3:
            if mask is not None:
                volume, mask = self.elastic_deformation(volume, mask)
            else:
                volume = self.elastic_deformation(volume)
        
        # Intensity augmentations (only for volume)
        if np.random.rand() < 0.5:
            volume = self.add_gaussian_noise(volume)
        
        if np.random.rand() < 0.5:
            volume = self.adjust_brightness(volume)
        
        if mask is not None:
            return volume, mask
        return volume


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = CTPreprocessor(
        target_spacing=(0.1, 0.1, 0.1),
        target_size=(128, 128, 128)
    )
    
    # Create dummy data
    volume = np.random.randn(100, 120, 120)
    mask = np.random.randint(0, 4, (100, 120, 120))
    
    # Preprocess
    volume_prep, mask_prep = preprocessor.preprocess(
        volume, 
        original_spacing=(0.15, 0.15, 0.15),
        mask=mask
    )
    
    print(f"Original volume shape: {volume.shape}")
    print(f"Preprocessed volume shape: {volume_prep.shape}")
    print(f"Preprocessed mask shape: {mask_prep.shape}")
    
    # Test augmentation
    augmenter = VolumeAugmenter()
    aug_volume, aug_mask = augmenter(volume_prep, mask_prep)
    print(f"Augmented volume shape: {aug_volume.shape}")
