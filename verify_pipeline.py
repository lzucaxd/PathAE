#!/usr/bin/env python3
"""
Verify the full data pipeline: loading → stain norm → RGB norm → denorm.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from stain_utils import StainNormalizer

# Load one sample
df = pd.read_csv('final_dataset/dataset.csv')
sample = df[df['split'] == 'train'].iloc[0]
img_path = Path('final_dataset') / sample['path']

# Load image
img_bgr = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Stain normalization
stain_normalizer = StainNormalizer(method='macenko', reference_path='reference_tile.npy')
img_stain_norm = stain_normalizer.normalize(img_rgb)

# RGB normalization (with CORRECT stats)
stats = np.load('normalization_stats.npy', allow_pickle=True).item()
mean = np.array(stats['mean'])
std = np.array(stats['std'])

img_float = img_stain_norm.astype(np.float32) / 255.0
img_normalized = (img_float - mean) / std

# Denormalize (for visualization)
img_denorm = img_normalized * std + mean
img_denorm = np.clip(img_denorm, 0, 1)

# Plot pipeline
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img_rgb)
axes[0].set_title('1. Raw Image')
axes[0].axis('off')

axes[1].imshow(img_stain_norm)
axes[1].set_title('2. Stain Normalized')
axes[1].axis('off')

# Show normalized (will look weird - it's z-scored)
img_norm_vis = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())
axes[2].imshow(img_norm_vis)
axes[2].set_title('3. RGB Normalized\n(z-scored, for model)')
axes[2].axis('off')

axes[3].imshow(img_denorm)
axes[3].set_title('4. Denormalized\n(reconstructed)')
axes[3].axis('off')

plt.suptitle(f'Data Pipeline Verification\nStats: mean={mean.round(3)}, std={std.round(3)}', fontsize=12)
plt.tight_layout()
plt.savefig('pipeline_verification.png', dpi=150, bbox_inches='tight')
print("✓ Saved pipeline verification to: pipeline_verification.png")
print()
print("Check that:")
print("  1. Raw image looks normal")
print("  2. Stain norm preserves color")
print("  3. Denormalized image looks similar to raw")
print()
print(f"Stats used:")
print(f"  Mean: {mean}")
print(f"  Std: {std}")


