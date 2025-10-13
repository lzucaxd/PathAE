#!/usr/bin/env python3
"""
Stain normalization utilities for histopathology images.

Implements:
- Macenko normalization (primary)
- Reinhard normalization (fallback)
- Reference tile creation
"""

import numpy as np
import cv2
from pathlib import Path

# Try to import torchstain for Macenko
try:
    from torchstain.base.normalizers import MacenkoNormalizer
    TORCHSTAIN_AVAILABLE = True
except ImportError:
    try:
        from torchstain.normalizers import MacenkoNormalizer
        TORCHSTAIN_AVAILABLE = True
    except ImportError:
        TORCHSTAIN_AVAILABLE = False
        print("Warning: torchstain not available, using Reinhard only")


class StainNormalizer:
    """
    Stain normalization with Macenko (primary) or Reinhard (fallback).
    """
    
    def __init__(self, reference_path='reference_tile.npy', method='macenko'):
        """
        Args:
            reference_path: Path to reference tile (.npy file with RGB image)
            method: 'macenko' or 'reinhard'
        """
        self.method = method
        self.reference_path = Path(reference_path)
        
        # Load reference tile
        if self.reference_path.exists():
            self.reference = np.load(self.reference_path)
            print(f"Loaded reference tile: {self.reference_path}")
        else:
            print(f"Warning: Reference tile not found at {self.reference_path}")
            print("Creating default reference from first image...")
            self.reference = None
        
        # Initialize normalizer
        if method == 'macenko' and TORCHSTAIN_AVAILABLE:
            self.normalizer = MacenkoNormalizer()
            if self.reference is not None:
                self.normalizer.fit(self.reference)
                self.fitted = True
            else:
                self.fitted = False
        elif method == 'reinhard' or not TORCHSTAIN_AVAILABLE:
            self.method = 'reinhard'
            if self.reference is not None:
                self._compute_reinhard_stats(self.reference)
            else:
                self.ref_means = None
                self.ref_stds = None
    
    def _compute_reinhard_stats(self, img):
        """Compute LAB statistics for Reinhard normalization."""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        self.ref_means = lab.reshape(-1, 3).mean(axis=0)
        self.ref_stds = lab.reshape(-1, 3).std(axis=0)
    
    def normalize(self, img_rgb):
        """
        Normalize stain of input image.
        
        Args:
            img_rgb: RGB image (uint8, 0-255) [H, W, 3]
            
        Returns:
            Normalized RGB image (uint8, 0-255) [H, W, 3]
        """
        if img_rgb is None or img_rgb.size == 0:
            return img_rgb
        
        try:
            if self.method == 'macenko' and TORCHSTAIN_AVAILABLE:
                if not self.fitted and self.reference is None:
                    # Use first image as reference
                    self.normalizer.fit(img_rgb)
                    self.fitted = True
                    return img_rgb
                
                # Normalize
                normalized = self.normalizer.normalize(img_rgb)
                return normalized
            
            else:  # Reinhard
                if self.ref_means is None:
                    # Use first image as reference
                    self._compute_reinhard_stats(img_rgb)
                    return img_rgb
                
                # Convert to LAB
                lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
                
                # Normalize
                means = lab.reshape(-1, 3).mean(axis=0)
                stds = lab.reshape(-1, 3).std(axis=0)
                
                lab_norm = (lab - means) / (stds + 1e-6) * self.ref_stds + self.ref_means
                lab_norm = np.clip(lab_norm, 0, 255).astype(np.uint8)
                
                # Convert back to RGB
                rgb_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)
                return rgb_norm
        
        except Exception as e:
            print(f"Warning: Stain normalization failed: {e}")
            return img_rgb


def create_reference_tile(csv_path, output_path='reference_tile.npy', n_samples=10):
    """
    Create reference tile by averaging multiple normal tissue patches.
    
    Args:
        csv_path: Path to dataset.csv
        output_path: Where to save reference tile
        n_samples: Number of patches to average
    """
    import pandas as pd
    
    print(f"Creating reference tile from {n_samples} normal patches...")
    
    df = pd.read_csv(csv_path)
    df = df[df['split'] == 'train'].sample(n=n_samples, random_state=42)
    base_dir = Path(csv_path).parent
    
    tiles = []
    for idx, row in df.iterrows():
        img_path = base_dir / row['path']
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tiles.append(img_rgb)
    
    # Average
    reference = np.mean(tiles, axis=0).astype(np.uint8)
    
    # Save
    np.save(output_path, reference)
    print(f"✓ Reference tile saved to: {output_path}")
    print(f"  Shape: {reference.shape}")
    print(f"  Mean: {reference.mean():.1f}")
    
    return reference


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create reference tile for stain normalization')
    parser.add_argument('--csv', type=str, default='final_dataset/dataset.csv')
    parser.add_argument('--output', type=str, default='reference_tile.npy')
    parser.add_argument('--n-samples', type=int, default=10)
    
    args = parser.parse_args()
    
    create_reference_tile(args.csv, args.output, args.n_samples)

