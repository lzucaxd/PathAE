#!/usr/bin/env python3
"""Debug: Why are RGB stats identical?"""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Load dataset
df = pd.read_csv('final_dataset/dataset.csv')
df_train = df[df['split'] == 'train']
base_dir = Path('final_dataset')

# Sample 100 images
df_sample = df_train.sample(n=100, random_state=42)

all_pixels = []

for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    img_path = base_dir / row['path']
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Reshape to (N_pixels, 3)
    pixels = img_float.reshape(-1, 3)
    all_pixels.append(pixels)

# Concatenate
all_pixels = np.concatenate(all_pixels, axis=0)

print(f"\nCollected pixels shape: {all_pixels.shape}")
print(f"First few pixels:")
print(all_pixels[:5])

# Compute stats
mean = all_pixels.mean(axis=0)
std = all_pixels.std(axis=0)

print(f"\nMean per channel: {mean}")
print(f"Std per channel: {std}")

# Check if identical
if np.allclose(mean[0], mean[1], atol=0.01) and np.allclose(mean[1], mean[2], atol=0.01):
    print("\n⚠️  BUG: Stats are identical across channels!")
    print("    Checking column-wise vs row-wise...")
    
    # Try alternative calculation
    mean_alt = np.mean(all_pixels, axis=0)
    std_alt = np.std(all_pixels, axis=0)
    print(f"\n  Alternative mean: {mean_alt}")
    print(f"  Alternative std: {std_alt}")
else:
    print("\n✓ Stats are DIFFERENT across channels (correct!)")


