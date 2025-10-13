#!/usr/bin/env python3
"""
Dataset inspection and monitoring tool.

Visualize samples, check quality, verify preprocessing.
"""

import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import TissueDataset
from stain_utils import StainNormalizer


def visualize_samples(csv_path, n_samples=16, split='train', stain_norm=False):
    """Visualize random samples from dataset."""
    
    df = pd.read_csv(csv_path)
    df = df[df['split'] == split].sample(n=n_samples, random_state=42)
    base_dir = Path(csv_path).parent
    
    # Setup stain normalizer if requested
    normalizer = None
    if stain_norm:
        normalizer = StainNormalizer('reference_tile.npy', method='macenko')
    
    # Load images
    images = []
    images_norm = []
    
    for idx, row in df.iterrows():
        img_path = base_dir / row['path']
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        images.append(img_rgb)
        
        if normalizer:
            img_norm = normalizer.normalize(img_rgb)
            images_norm.append(img_norm)
    
    # Plot
    rows = 4
    cols = 4 if not stain_norm else 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    
    for i in range(n_samples):
        row = i // 4
        col = i % 4
        
        # Original
        axes[row, col].imshow(images[i])
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Sample {i+1}', fontsize=8)
        
        # Normalized (if requested)
        if stain_norm:
            axes[row, col+4].imshow(images_norm[i])
            axes[row, col+4].axis('off')
            axes[row, col+4].set_title(f'Stain Norm {i+1}', fontsize=8)
    
    title = f'{split.capitalize()} Set Samples'
    if stain_norm:
        title += ' (Left: Original, Right: Stain Normalized)'
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = f'dataset_samples_{split}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.close()


def analyze_statistics(csv_path):
    """Analyze dataset statistics."""
    
    df = pd.read_csv(csv_path)
    
    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print()
    
    print("Overall:")
    print(f"  Total samples: {len(df):,}")
    print()
    
    print("By split:")
    for split in df['split'].unique():
        split_df = df[df['split'] == split]
        print(f"  {split:6}: {len(split_df):8,} samples", end="")
        
        if 'label' in df.columns:
            n_tumor = (split_df['label'] == 1).sum()
            n_normal = (split_df['label'] == 0).sum()
            print(f" ({n_tumor:,} tumor, {n_normal:,} normal)")
        else:
            print()
    
    print()
    
    if 'wsi_id' in df.columns:
        print("By slide (top 10):")
        slide_counts = df['wsi_id'].value_counts().head(10)
        for wsi_id, count in slide_counts.items():
            slide_df = df[df['wsi_id'] == wsi_id]
            if 'label' in df.columns:
                n_tumor = (slide_df['label'] == 1).sum()
                n_normal = (slide_df['label'] == 0).sum()
                tumor_pct = 100 * n_tumor / len(slide_df) if len(slide_df) > 0 else 0
                print(f"  {wsi_id:15}: {count:8,} tiles ({tumor_pct:5.1f}% tumor)")
            else:
                print(f"  {wsi_id:15}: {count:8,} tiles")


def check_preprocessing(csv_path, n_samples=5):
    """Test preprocessing pipeline."""
    
    print("="*70)
    print("TESTING PREPROCESSING PIPELINE")
    print("="*70)
    print()
    
    # Load normalization stats
    if Path('normalization_stats.npy').exists():
        stats = np.load('normalization_stats.npy', allow_pickle=True).item()
        mean = stats['mean']
        std = stats['std']
        print(f"Normalization stats:")
        print(f"  Mean: {mean}")
        print(f"  Std: {std}")
    else:
        print("Warning: normalization_stats.npy not found")
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    
    print()
    
    # Create dataset with full preprocessing
    try:
        dataset = TissueDataset(
            csv_path=csv_path,
            split='train',
            stain_norm=True,
            reference_tile='reference_tile.npy',
            normalize=True,
            mean=mean,
            std=std,
            augment=True,
        )
        
        print(f"Loading {n_samples} samples...")
        for i in range(n_samples):
            img, target = dataset[i]
            print(f"  Sample {i+1}: shape={img.shape}, min={img.min():.3f}, max={img.max():.3f}, mean={img.mean():.3f}")
        
        print()
        print("✓ Preprocessing pipeline working correctly!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you've run:")
        print("  python stain_utils.py --csv final_dataset/dataset.csv")
        print("  python compute_normalization_stats.py --csv final_dataset/dataset.csv")


def main():
    parser = argparse.ArgumentParser(description='Inspect dataset')
    parser.add_argument('--csv', type=str, default='final_dataset/dataset.csv')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--test-pipeline', action='store_true', help='Test preprocessing')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--stain-norm', action='store_true', help='Show stain normalization')
    parser.add_argument('--all', action='store_true', help='Run all checks')
    
    args = parser.parse_args()
    
    if args.all:
        args.stats = True
        args.visualize = True
        args.test_pipeline = True
    
    if args.stats:
        analyze_statistics(args.csv)
        print()
    
    if args.visualize:
        visualize_samples(args.csv, n_samples=16, split=args.split, stain_norm=args.stain_norm)
        print()
    
    if args.test_pipeline:
        check_preprocessing(args.csv, n_samples=5)


if __name__ == '__main__':
    main()

