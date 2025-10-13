#!/usr/bin/env python3
"""
Compute normalization statistics from PCam training normals.

This calculates mean and std for RGB normalization after stain normalization.
"""

import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm


def compute_stats(csv_path, split='train', max_samples=10000):
    """
    Compute mean and std from training normals.
    
    Args:
        csv_path: Path to dataset.csv
        split: Which split to use ('train')
        max_samples: Max number of samples to use (for speed)
    """
    print(f"Computing normalization statistics from {split} split...")
    print(f"Using up to {max_samples:,} samples for efficiency")
    print()
    
    # Load dataset
    df = pd.read_csv(csv_path)
    df = df[df['split'] == split]
    base_dir = Path(csv_path).parent
    
    # Sample if too large
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    print(f"Processing {len(df):,} patches...")
    
    # Accumulate pixel values
    all_pixels = []
    
    for idx in tqdm(range(len(df)), desc="Loading patches"):
        row = df.iloc[idx]
        img_path = base_dir / row['path']
        
        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Flatten and collect
        all_pixels.append(img_float.reshape(-1, 3))
    
    # Concatenate all pixels
    all_pixels = np.concatenate(all_pixels, axis=0)
    
    # Compute statistics
    mean = all_pixels.mean(axis=0)
    std = all_pixels.std(axis=0)
    
    print()
    print("="*70)
    print("NORMALIZATION STATISTICS")
    print("="*70)
    print(f"Samples processed: {len(df):,}")
    print(f"Total pixels: {len(all_pixels):,}")
    print()
    print(f"Mean (R, G, B): [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"Std  (R, G, B): [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print()
    
    # Save to file
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'n_samples': len(df),
        'n_pixels': len(all_pixels),
    }
    
    np.save('normalization_stats.npy', stats)
    print(f"✓ Saved to: normalization_stats.npy")
    print()
    print("Use in code:")
    print(f"  mean = {mean.tolist()}")
    print(f"  std = {std.tolist()}")
    
    return mean, std


def main():
    parser = argparse.ArgumentParser(description='Compute normalization statistics')
    parser.add_argument('--csv', type=str, default='final_dataset/dataset.csv')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max-samples', type=int, default=10000)
    
    args = parser.parse_args()
    
    compute_stats(args.csv, args.split, args.max_samples)


if __name__ == '__main__':
    main()

