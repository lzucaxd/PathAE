#!/usr/bin/env python3
"""Quick debug script to check data pipeline."""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path

# Load dataset
df = pd.read_csv('final_dataset/dataset.csv')
df_train = df[df['split'] == 'train'].head(10)

base_dir = Path('final_dataset')

print("="*70)
print("DATA PIPELINE DEBUG")
print("="*70)

for idx, row in df_train.iterrows():
    img_path = base_dir / row['path']
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"⚠️  Failed to load: {img_path}")
        continue
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    print(f"\nImage {idx}: {img_path.name}")
    print(f"  Shape: {img_rgb.shape}")
    print(f"  Min/Max: {img_float.min():.3f} / {img_float.max():.3f}")
    print(f"  Mean per channel: R={img_float[:,:,0].mean():.4f}, G={img_float[:,:,1].mean():.4f}, B={img_float[:,:,2].mean():.4f}")
    print(f"  Std per channel:  R={img_float[:,:,0].std():.4f}, G={img_float[:,:,1].std():.4f}, B={img_float[:,:,2].std():.4f}")
    
    # Check if all channels are identical
    if np.allclose(img_rgb[:,:,0], img_rgb[:,:,1], atol=1) and np.allclose(img_rgb[:,:,1], img_rgb[:,:,2], atol=1):
        print(f"  ⚠️  WARNING: All channels are NEARLY IDENTICAL (grayscale!)")

print("\n" + "="*70)
print("CHECKING NORMALIZATION STATS FILE")
print("="*70)

# Check normalization stats
stats = np.load('normalization_stats.npy', allow_pickle=True).item()
print(f"\nStored stats:")
print(f"  Mean: {stats['mean']}")
print(f"  Std: {stats['std']}")
print(f"  Samples: {stats['n_samples']:,}")

# Check if grayscale
if np.allclose(stats['mean'][0], stats['mean'][1], atol=0.01) and np.allclose(stats['mean'][1], stats['mean'][2], atol=0.01):
    print(f"\n⚠️  PROBLEM DETECTED: All channel statistics are IDENTICAL!")
    print(f"     This suggests the images are grayscale or improperly processed.")
    print(f"     PCam should have RGB color images, not grayscale!")


