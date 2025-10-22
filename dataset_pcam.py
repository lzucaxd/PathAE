#!/usr/bin/env python3
"""
PatchCamelyon (PCam) Dataset Loader.

Loads directly from HDF5 files without any filtering.
Supports both supervised and unsupervised training.

File structure:
    PCam/
        camelyonpatch_level_2_split_train_x.h5  (262,144 patches, 96x96x3)
        camelyonpatch_level_2_split_train_y.h5  (262,144 labels)
        camelyonpatch_level_2_split_valid_x.h5  (32,768 patches)
        camelyonpatch_level_2_split_valid_y.h5  (32,768 labels)
        camelyonpatch_level_2_split_test_x.h5   (32,768 patches)
        camelyonpatch_level_2_split_test_y.h5   (32,768 labels)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from pathlib import Path
import albumentations as A

from stain_utils import StainNormalizer


class PatchCamelyonDataset(Dataset):
    """
    PatchCamelyon dataset for both supervised and unsupervised training.
    
    Args:
        data_dir: Path to PCam directory with HDF5 files
        split: 'train', 'valid', or 'test'
        filter_normal_only: If True, only load normal patches (for autoencoder training)
        stain_norm: Apply stain normalization
        reference_tile: Path to reference tile for stain norm
        normalize: Apply mean/std normalization
        mean: RGB mean for normalization
        std: RGB std for normalization
        augment: Apply data augmentation (train only)
    """
    
    def __init__(
        self,
        data_dir='PCam',
        split='train',
        filter_normal_only=False,
        stain_norm=True,
        reference_tile='reference_tile.npy',
        normalize=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        augment=False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.filter_normal_only = filter_normal_only
        self.stain_norm = stain_norm
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.augment = augment and (split == 'train')
        
        # Map split names to file names
        split_mapping = {
            'train': 'training',
            'valid': 'validation',
            'test': 'test'
        }
        file_split = split_mapping.get(split, split)
        
        # Load HDF5 files (different paths for images and labels)
        x_path = self.data_dir / 'pcam' / f'{file_split}_split.h5'
        y_path = self.data_dir / 'Labels' / 'Labels' / f'camelyonpatch_level_2_split_{split}_y.h5'
        
        if not x_path.exists():
            raise FileNotFoundError(f"Data file not found: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Label file not found: {y_path}")
        
        print(f"Loading PCam {split} set...")
        print(f"  Data: {x_path}")
        print(f"  Labels: {y_path}")
        
        # Load data into memory (PCam is small enough)
        with h5py.File(x_path, 'r') as f:
            self.images = np.array(f['x'])  # [N, 96, 96, 3], uint8
        
        with h5py.File(y_path, 'r') as f:
            self.labels = np.array(f['y'])  # [N, 1, 1, 1]
            self.labels = self.labels.reshape(-1)  # [N]
        
        print(f"  Loaded: {len(self.images):,} patches")
        print(f"    Normal: {(self.labels == 0).sum():,}")
        print(f"    Tumor: {(self.labels == 1).sum():,}")
        
        # Filter to normal only if requested
        if filter_normal_only:
            normal_mask = self.labels == 0
            self.images = self.images[normal_mask]
            self.labels = self.labels[normal_mask]
            print(f"  Filtered to normal only: {len(self.images):,} patches")
        
        # Stain normalization
        if stain_norm:
            self.stain_normalizer = StainNormalizer(
                reference_path=reference_tile,
                method='macenko'
            )
        else:
            self.stain_normalizer = None
        
        # Augmentation (train only)
        if self.augment:
            self.transform = A.Compose([
                # Geometric
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # Color (light)
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=2,
                    sat_shift_limit=5,
                    val_shift_limit=0,
                    p=0.3
                ),
            ])
        else:
            self.transform = None
        
        print(f"Dataset initialized:")
        print(f"  Split: {split}")
        print(f"  Samples: {len(self.images):,}")
        print(f"  Stain norm: {stain_norm}")
        print(f"  RGB norm: {normalize}")
        print(f"  Augment: {self.augment}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single patch.
        
        Returns:
            image: Preprocessed tensor [C, H, W]
            label: Label (0=normal, 1=tumor)
        """
        # Load image (already RGB uint8 [H, W, 3])
        img_rgb = self.images[idx]
        label = int(self.labels[idx])
        
        # Stain normalization
        if self.stain_norm and self.stain_normalizer:
            img_rgb = self.stain_normalizer.normalize(img_rgb)
        
        # Augmentation (on uint8 image, before normalization)
        if self.augment and self.transform is not None:
            augmented = self.transform(image=img_rgb)
            img_rgb = augmented['image']
        
        # Scale to [0, 1] and convert to tensor
        img_float = img_rgb.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)
        
        # RGB normalization (mean/std)
        if self.normalize:
            for c in range(3):
                img_tensor[c] = (img_tensor[c] - self.mean[c]) / (self.std[c] + 1e-6)
        
        return img_tensor, label


class PatchCamelyonDatasetUnsupervised(PatchCamelyonDataset):
    """
    PCam dataset for unsupervised training (autoencoders).
    Returns (image, image) instead of (image, label).
    """
    
    def __getitem__(self, idx):
        img_tensor, label = super().__getitem__(idx)
        # Return image twice for autoencoder training
        return img_tensor, img_tensor


if __name__ == '__main__':
    # Test loading
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("Testing PatchCamelyonDataset")
    print("="*70)
    
    # Test supervised loading
    dataset_test = PatchCamelyonDataset(
        data_dir='PCam',
        split='test',
        filter_normal_only=False,
        stain_norm=False,
        normalize=False,
        augment=False
    )
    
    print(f"\nTest sample:")
    img, label = dataset_test[0]
    print(f"  Image shape: {img.shape}")
    print(f"  Label: {label}")
    print(f"  Min/Max: {img.min():.3f} / {img.max():.3f}")
    
    # Visualize a few samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        img, label = dataset_test[i * 3000]  # Sample across dataset
        img_vis = img.permute(1, 2, 0).numpy()
        
        row = i // 5
        col = i % 5
        axes[row, col].imshow(img_vis)
        axes[row, col].set_title(f'{"Tumor" if label == 1 else "Normal"}', 
                                  color='red' if label == 1 else 'blue')
        axes[row, col].axis('off')
    
    plt.suptitle('PatchCamelyon Test Set Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pcam_samples.png', dpi=150)
    print(f"\n✓ Saved sample visualization: pcam_samples.png")
    
    print("\n" + "="*70)
    print("✓ PatchCamelyonDataset working correctly!")
    print("="*70)

