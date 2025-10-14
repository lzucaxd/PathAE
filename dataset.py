#!/usr/bin/env python3
"""
PyTorch dataset for PathAE with complete preprocessing pipeline.

Features:
- Stain normalization (Macenko/Reinhard)
- RGB normalization (mean/std from PCam)
- Quality filtering
- Data augmentation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from stain_utils import StainNormalizer


class TissueDataset(Dataset):
    """
    Dataset for tissue patches with complete preprocessing.
    
    Preprocessing pipeline:
    1. Load RGB image
    2. Stain normalization (Macenko → Reinhard fallback)
    3. Quality filtering (optional)
    4. Scale to [0, 1]
    5. Normalize with mean/std
    6. Augmentation (train only)
    7. To tensor [C, H, W]
    """
    
    def __init__(
        self,
        csv_path,
        split='train',
        stain_norm=True,
        reference_tile='reference_tile.npy',
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        augment=True,
        tissue_threshold=0.65,
        blur_threshold=30,
    ):
        """
        Args:
            csv_path: Path to dataset CSV
            split: 'train' or 'test'
            stain_norm: Apply stain normalization
            reference_tile: Path to reference tile for stain norm
            normalize: Apply mean/std normalization
            mean: RGB mean for normalization
            std: RGB std for normalization
            augment: Apply data augmentation (train only)
            tissue_threshold: Minimum tissue fraction (HSV-based)
            blur_threshold: Minimum blur variance
        """
        self.csv_path = Path(csv_path)
        self.split = split
        self.base_dir = self.csv_path.parent
        
        # Load dataset
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        
        # Stain normalization
        self.stain_norm = stain_norm
        if stain_norm:
            self.stain_normalizer = StainNormalizer(
                reference_path=reference_tile,
                method='macenko'
            )
        
        # RGB normalization
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
        # Quality filtering
        self.tissue_threshold = tissue_threshold
        self.blur_threshold = blur_threshold
        
        # Augmentation (train only)
        self.augment = augment and (split == 'train')
        
        if self.augment:
            self.transform = A.Compose([
                # Geometric
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # Color (light, on uint8 [0,255])
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,  # ±10%
                    contrast_limit=0.1,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=2,      # ±2°
                    sat_shift_limit=5,      # ±5% (in uint8: ±12.75)
                    val_shift_limit=0,
                    p=0.3
                ),
            ])
        else:
            self.transform = None
        
        print(f"Dataset initialized:")
        print(f"  Split: {split}")
        print(f"  Samples: {len(self.df):,}")
        print(f"  Stain norm: {stain_norm}")
        print(f"  RGB norm: {normalize}")
        print(f"  Augment: {self.augment}")
    
    def __len__(self):
        return len(self.df)
    
    def _check_quality(self, img_rgb):
        """
        Check tissue quality using HSV and blur variance.
        
        Args:
            img_rgb: RGB image (uint8)
            
        Returns:
            True if passes quality checks
        """
        # Tissue fraction (HSV-based)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] /= 255.0  # Normalize saturation
        
        # Tissue mask: saturation > 0.07 (from PCam paper)
        tissue_mask = hsv[:, :, 1] > 0.07
        tissue_frac = tissue_mask.sum() / tissue_mask.size
        
        if tissue_frac < self.tissue_threshold:
            return False
        
        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_var < self.blur_threshold:
            return False
        
        return True
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single patch.
        
        Returns:
            image: Preprocessed tensor [C, H, W]
            image: Same (for autoencoder/VAE reconstruction)
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.base_dir / row['path']
        img_bgr = cv2.imread(str(img_path))
        
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Quality check (optional - can skip if data is pre-filtered)
        # if not self._check_quality(img_rgb):
        #     # Return a random other sample if quality fails
        #     return self.__getitem__(np.random.randint(len(self)))
        
        # Stain normalization
        if self.stain_norm:
            img_rgb = self.stain_normalizer.normalize(img_rgb)
        
        # Ensure we have exactly 3 channels (RGB)
        if img_rgb.shape[2] != 3:
            img_rgb = img_rgb[:, :, :3]
        
        # Augmentation (on uint8 image, before normalization)
        if self.augment and self.transform is not None:
            # Albumentations expects uint8 [0,255] for color transforms
            augmented = self.transform(image=img_rgb)
            img_augmented = augmented['image']
        else:
            img_augmented = img_rgb
        
        # Scale to [0, 1] and convert to tensor
        img_float = img_augmented.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)
        
        # RGB normalization (mean/std)
        if self.normalize:
            for c in range(3):
                img_tensor[c] = (img_tensor[c] - self.mean[c]) / (self.std[c] + 1e-6)
        
        return img_tensor, img_tensor  # Return twice for VAE (input, target)


class TestDataset(Dataset):
    """
    Test dataset (applies same preprocessing but no augmentation).
    """
    
    def __init__(
        self,
        csv_path,
        stain_norm=True,
        reference_tile='reference_tile.npy',
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ):
        self.csv_path = Path(csv_path)
        self.base_dir = self.csv_path.parent
        
        # Load dataset
        self.df = pd.read_csv(csv_path)
        
        # Stain normalization
        self.stain_norm = stain_norm
        if stain_norm:
            self.stain_normalizer = StainNormalizer(
                reference_path=reference_tile,
                method='macenko'
            )
        
        # RGB normalization
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
        # No augmentation for test
        self.transform = A.Compose([ToTensorV2()])
        
        print(f"Test dataset initialized:")
        print(f"  Samples: {len(self.df):,}")
        print(f"  Stain norm: {stain_norm}")
        print(f"  RGB norm: {normalize}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.base_dir / row['path']
        img_bgr = cv2.imread(str(img_path))
        
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Stain normalization
        if self.stain_norm:
            img_rgb = self.stain_normalizer.normalize(img_rgb)
        
        # Scale to [0, 1]
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # To tensor
        augmented = self.transform(image=img_float)
        img_tensor = augmented['image']
        
        # RGB normalization
        if self.normalize:
            for c in range(3):
                img_tensor[c] = (img_tensor[c] - self.mean[c]) / (self.std[c] + 1e-6)
        
        return img_tensor, row['tile_id']


if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset...")
    
    dataset = TissueDataset(
        csv_path='final_dataset/dataset.csv',
        split='train',
        stain_norm=False,  # Skip for speed test
        normalize=False,
    )
    
    print(f"\nLoading first sample...")
    img, target = dataset[0]
    print(f"  Shape: {img.shape}")
    print(f"  Min: {img.min():.4f}")
    print(f"  Max: {img.max():.4f}")
    print(f"  Mean: {img.mean():.4f}")
    
    print("\n✓ Dataset working correctly!")

