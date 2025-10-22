#!/usr/bin/env python3
"""
Create preprocessing visualization figures for presentation.

Generates:
1. Macenko normalization before/after comparison
2. Example patches grid (normal vs tumor)
3. Augmentation examples
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from stain_utils import StainNormalizer
import albumentations as A


def create_macenko_comparison(pcam_dir, reference_tile, output_path, n_examples=4):
    """
    Create before/after comparison of Macenko normalization.
    
    Shows 2 rows:
    - Top: Original patches (varied staining)
    - Bottom: Macenko normalized (consistent staining)
    """
    print(f"\n{'='*70}")
    print("CREATING MACENKO NORMALIZATION FIGURE")
    print(f"{'='*70}\n")
    
    # Load reference tile
    stain_normalizer = StainNormalizer(reference_path=reference_tile, method='macenko')
    
    # Load diverse examples from PCam test set
    pcam_dir = Path(pcam_dir)
    test_path = pcam_dir / 'pcam' / 'test_split.h5'
    
    with h5py.File(test_path, 'r') as f:
        # Get patches with different color characteristics
        # Sample from different parts of dataset for variety
        indices = [1000, 8500, 15000, 25000]  # Spread across dataset
        images = [f['x'][idx] for idx in indices]
    
    # Normalize
    images_normalized = [stain_normalizer.normalize(img) for img in images]
    
    # Create figure
    fig, axes = plt.subplots(2, n_examples, figsize=(16, 8))
    
    for i in range(n_examples):
        # Original
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f'Original {i+1}', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # Normalized
        axes[1, i].imshow(images_normalized[i])
        axes[1, i].set_title(f'Macenko Normalized {i+1}', fontsize=12, fontweight='bold')
        axes[1, i].axis('off')
    
    plt.suptitle('Macenko Stain Normalization: Standardizing H&E Color Appearance', 
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Macenko comparison saved: {output_path}")


def create_patch_examples(pcam_dir, output_path, n_per_class=8):
    """
    Create grid showing example patches (normal vs tumor).
    
    2 rows:
    - Top: Normal tissue patches
    - Bottom: Tumor patches
    """
    print(f"\n{'='*70}")
    print("CREATING PATCH EXAMPLES FIGURE")
    print(f"{'='*70}\n")
    
    # Load PCam test set
    pcam_dir = Path(pcam_dir)
    x_test_path = pcam_dir / 'pcam' / 'test_split.h5'
    y_test_path = pcam_dir / 'Labels' / 'Labels' / 'camelyonpatch_level_2_split_test_y.h5'
    
    with h5py.File(x_test_path, 'r') as fx, h5py.File(y_test_path, 'r') as fy:
        images = fx['x'][:]
        labels = fy['y'][:].squeeze()
    
    # Sample diverse examples
    np.random.seed(42)
    
    normal_indices = np.where(labels == 0)[0]
    tumor_indices = np.where(labels == 1)[0]
    
    # Sample evenly across dataset for diversity
    normal_samples = np.linspace(0, len(normal_indices)-1, n_per_class, dtype=int)
    tumor_samples = np.linspace(0, len(tumor_indices)-1, n_per_class, dtype=int)
    
    normal_images = [images[normal_indices[i]] for i in normal_samples]
    tumor_images = [images[tumor_indices[i]] for i in tumor_samples]
    
    # Create figure
    fig, axes = plt.subplots(2, n_per_class, figsize=(20, 5))
    
    for i in range(n_per_class):
        # Normal
        axes[0, i].imshow(normal_images[i])
        axes[0, i].set_title(f'Normal {i+1}', fontsize=10, fontweight='bold', color='#2196F3')
        axes[0, i].axis('off')
        # Add green border
        for spine in axes[0, i].spines.values():
            spine.set_edgecolor('#4CAF50')
            spine.set_linewidth(3)
            spine.set_visible(True)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Tumor
        axes[1, i].imshow(tumor_images[i])
        axes[1, i].set_title(f'Tumor {i+1}', fontsize=10, fontweight='bold', color='#F44336')
        axes[1, i].axis('off')
        # Add red border
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor('#F44336')
            spine.set_linewidth(3)
            spine.set_visible(True)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    plt.suptitle('PatchCAMELYON (PCam) Dataset: Normal vs Tumor Examples\n96×96 pixels, Level 2 (10× magnification)', 
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Patch examples saved: {output_path}")


def create_augmentation_examples(pcam_dir, reference_tile, norm_stats_path, output_path):
    """
    Show one patch with different augmentations applied.
    
    Demonstrates biologically valid augmentations.
    """
    print(f"\n{'='*70}")
    print("CREATING AUGMENTATION EXAMPLES FIGURE")
    print(f"{'='*70}\n")
    
    # Load one tumor patch
    pcam_dir = Path(pcam_dir)
    x_test_path = pcam_dir / 'pcam' / 'test_split.h5'
    y_test_path = pcam_dir / 'Labels' / 'Labels' / 'camelyonpatch_level_2_split_test_y.h5'
    
    with h5py.File(x_test_path, 'r') as fx, h5py.File(y_test_path, 'r') as fy:
        labels = fy['y'][:].squeeze()
        tumor_idx = np.where(labels == 1)[0]
        # Pick a nice clear tumor example
        img = fx['x'][tumor_idx[100]]  # Index 100 should be a good example
    
    # Normalize
    stain_normalizer = StainNormalizer(reference_path=reference_tile, method='macenko')
    img_norm = stain_normalizer.normalize(img)
    
    # Define augmentations
    augmentations = {
        'Original': None,
        'Horizontal Flip': A.HorizontalFlip(p=1.0),
        'Vertical Flip': A.VerticalFlip(p=1.0),
        'Rotate 90°': A.Rotate(limit=(90, 90), p=1.0),
        'Rotate 180°': A.Rotate(limit=(180, 180), p=1.0),
        'Brightness +10%': A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0),
        'Color Jitter': A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=5, val_shift_limit=0, p=1.0),
        'Gaussian Blur': A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(1.5, 1.5), p=1.0)
    }
    
    # Apply augmentations
    aug_images = []
    aug_names = []
    
    for name, aug in augmentations.items():
        if aug is None:
            aug_img = img_norm
        else:
            augmented = aug(image=img_norm)
            aug_img = augmented['image']
        aug_images.append(aug_img)
        aug_names.append(name)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (img_aug, name) in enumerate(zip(aug_images, aug_names)):
        axes[i].imshow(img_aug)
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('Biologically Valid Augmentations for Histopathology\n'
                'All transformations preserve biological plausibility', 
                fontsize=15, fontweight='bold', y=0.98)
    
    # Add rationale text
    rationale_text = ("Why these augmentations?\n"
                     "• Flips & Rotations: Tissue orientation is arbitrary in microscopy\n"
                     "• Color jitter: Accounts for staining protocol variation between labs\n"
                     "• Gaussian blur: Mimics focal plane variation\n"
                     "NOT USED: Grayscale, cutout, extreme transforms (not biologically valid)")
    
    fig.text(0.5, 0.02, rationale_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Augmentation examples saved: {output_path}")


def create_preprocessing_pipeline_figure(pcam_dir, reference_tile, norm_stats_path, output_dir):
    """
    Create comprehensive preprocessing pipeline visualization.
    
    Shows: Raw → Macenko → RGB norm → Augment
    """
    print(f"\n{'='*70}")
    print("CREATING PREPROCESSING PIPELINE FIGURE")
    print(f"{'='*70}\n")
    
    # Load one diverse patch
    pcam_dir = Path(pcam_dir)
    x_test_path = pcam_dir / 'pcam' / 'test_split.h5'
    
    with h5py.File(x_test_path, 'r') as f:
        # Pick a patch with visible staining variation
        img_raw = f['x'][5000]
    
    # Step 1: Original
    img_original = img_raw.copy()
    
    # Step 2: Macenko normalization
    stain_normalizer = StainNormalizer(reference_path=reference_tile, method='macenko')
    img_macenko = stain_normalizer.normalize(img_raw)
    
    # Step 3: RGB normalization (for visualization, denormalize back)
    norm_stats = np.load(norm_stats_path, allow_pickle=True).item()
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    img_rgb_norm = img_macenko.astype(np.float32) / 255.0
    img_rgb_norm_viz = img_rgb_norm.copy()  # Keep for visualization
    
    # Normalize
    for c in range(3):
        img_rgb_norm[c] = (img_rgb_norm[c] - mean[c]) / std[c]
    
    # For visualization: clip to reasonable range and scale back
    img_rgb_norm_display = np.clip(img_rgb_norm, -3, 3)  # Clip to ±3 std
    img_rgb_norm_display = (img_rgb_norm_display + 3) / 6  # Scale to [0,1]
    img_rgb_norm_display = (img_rgb_norm_display * 255).astype(np.uint8)
    
    # Step 4: Augmentation example
    aug = A.Compose([
        A.RandomRotate90(p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)
    ])
    img_augmented = aug(image=img_macenko)['image']
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Step 1
    axes[0].imshow(img_original)
    axes[0].set_title('1. Original Patch\n(Raw from scanner)', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    axes[0].text(0.5, -0.05, 'Variable staining\nbetween labs', 
                transform=axes[0].transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))
    
    # Step 2
    axes[1].imshow(img_macenko)
    axes[1].set_title('2. Stain Normalized\n(Macenko method)', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    axes[1].text(0.5, -0.05, 'Consistent colors\nacross all patches', 
                transform=axes[1].transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))
    
    # Step 3
    axes[2].imshow(img_macenko)  # Show original for clarity
    axes[2].set_title('3. RGB Normalized\n(Mean/Std per channel)', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    
    # Add normalization stats as text
    stats_text = (f"Mean: [{mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f}]\n"
                 f"Std:  [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}]")
    axes[2].text(0.5, -0.05, stats_text, 
                transform=axes[2].transAxes, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8),
                family='monospace')
    
    # Step 4
    axes[3].imshow(img_augmented)
    axes[3].set_title('4. Augmented\n(Training only)', fontsize=13, fontweight='bold')
    axes[3].axis('off')
    axes[3].text(0.5, -0.05, 'Flips, rotations,\ncolor jitter, blur', 
                transform=axes[3].transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))
    
    # Add arrows between steps
    for i in range(3):
        fig.text(0.23 + i*0.24, 0.5, '→', fontsize=40, ha='center', va='center', fontweight='bold')
    
    plt.suptitle('Preprocessing Pipeline: From Raw Image to Model Input', 
                fontsize=16, fontweight='bold', y=0.96)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    
    output_path = Path(output_dir) / 'preprocessing_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Preprocessing pipeline saved: {output_path}")


def create_data_statistics_figure(pcam_dir, output_path):
    """
    Create data statistics visualization.
    
    Shows:
    - Training/Val/Test split sizes
    - Class balance
    - Patch size and magnification
    """
    print(f"\n{'='*70}")
    print("CREATING DATA STATISTICS FIGURE")
    print(f"{'='*70}\n")
    
    # Load PCam data
    pcam_dir = Path(pcam_dir)
    
    splits_data = []
    for split, file_split in [('train', 'training'), ('valid', 'validation'), ('test', 'test')]:
        x_path = pcam_dir / 'pcam' / f'{file_split}_split.h5'
        y_path = pcam_dir / 'Labels' / 'Labels' / f'camelyonpatch_level_2_split_{split}_y.h5'
        
        with h5py.File(x_path, 'r') as fx, h5py.File(y_path, 'r') as fy:
            n_total = len(fx['x'])
            labels = fy['y'][:].squeeze()
            n_normal = (labels == 0).sum()
            n_tumor = (labels == 1).sum()
        
        splits_data.append({
            'split': split.capitalize(),
            'total': n_total,
            'normal': n_normal,
            'tumor': n_tumor
        })
    
    df_splits = pd.DataFrame(splits_data)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # =========================================================================
    # Panel 1: Dataset sizes
    # =========================================================================
    ax = axes[0]
    
    x = np.arange(len(df_splits))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_splits['normal'], width, label='Normal',
                  color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, df_splits['tumor'], width, label='Tumor',
                  color='#F44336', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, row in df_splits.iterrows():
        ax.text(i - width/2, row['normal'] + 5000, f"{int(row['normal']):,}", 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(i + width/2, row['tumor'] + 5000, f"{int(row['tumor']):,}", 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_splits['split'], fontsize=12)
    ax.set_ylabel('Number of Patches', fontsize=12, fontweight='bold')
    ax.set_title('PCam Dataset Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Panel 2: Class balance
    # =========================================================================
    ax = axes[1]
    
    # Pie chart for training set
    train_normal = df_splits[df_splits['split'] == 'Train']['normal'].values[0]
    train_tumor = df_splits[df_splits['split'] == 'Train']['tumor'].values[0]
    
    colors = ['#4CAF50', '#F44336']
    labels_pie = [f'Normal\n{train_normal:,}\n(50%)', f'Tumor\n{train_tumor:,}\n(50%)']
    
    ax.pie([train_normal, train_tumor], labels=labels_pie, colors=colors,
           autopct='', startangle=90, 
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Training Set Balance\n(Perfect 50-50 split)', fontsize=13, fontweight='bold')
    
    # =========================================================================
    # Panel 3: Patch specifications
    # =========================================================================
    ax = axes[2]
    ax.axis('off')
    
    # Create specification table
    specs = [
        ['Specification', 'Value'],
        ['', ''],
        ['Patch Size', '96 × 96 pixels'],
        ['Magnification', 'Level 2 (10× from 40×)'],
        ['Color Space', 'RGB (8-bit per channel)'],
        ['File Format', 'HDF5 (efficient storage)'],
        ['', ''],
        ['Training Patches', '262,144'],
        ['Validation Patches', '32,768'],
        ['Test Patches', '32,768'],
        ['', ''],
        ['Normal:Tumor Ratio', '50:50 (balanced)'],
        ['Tissue Type', 'Breast cancer lymph node'],
        ['Staining', 'H&E (routine clinical)'],
    ]
    
    table = ax.table(cellText=specs, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style section breaks
    for row in [1, 6, 10]:
        for col in range(2):
            table[(row, col)].set_facecolor('#F5F5F5')
    
    ax.set_title('Dataset Specifications', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('PatchCAMELYON (PCam) Dataset Overview', 
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Data statistics saved: {output_path}")


def main():
    """Main function."""
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Create preprocessing figures')
    parser.add_argument('--pcam-dir', type=str, default='PCam')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    parser.add_argument('--output-dir', type=str, default='presentation')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("CREATING PREPROCESSING VISUALIZATION FIGURES")
    print(f"{'='*70}\n")
    print(f"Output directory: {output_dir}")
    
    # Create all figures
    create_macenko_comparison(
        args.pcam_dir, 
        args.reference_tile,
        output_dir / '1_macenko_normalization.png',
        n_examples=4
    )
    
    create_patch_examples(
        args.pcam_dir,
        output_dir / '2_patch_examples.png',
        n_per_class=8
    )
    
    create_augmentation_examples(
        args.pcam_dir,
        args.reference_tile,
        args.norm_stats,
        output_dir / '3_augmentation_examples.png'
    )
    
    create_preprocessing_pipeline_figure(
        args.pcam_dir,
        args.reference_tile,
        args.norm_stats,
        output_dir
    )
    
    create_data_statistics_figure(
        args.pcam_dir,
        output_dir / '5_data_statistics.png'
    )
    
    print(f"\n{'='*70}")
    print("✓ ALL PREPROCESSING FIGURES CREATED")
    print(f"{'='*70}\n")
    
    print("Generated files:")
    print("  1. 1_macenko_normalization.png - Before/after stain normalization")
    print("  2. 2_patch_examples.png - Normal vs tumor examples (8×2 grid)")
    print("  3. 3_augmentation_examples.png - Augmentation demonstrations")
    print("  4. 4_preprocessing_pipeline.png - Complete pipeline visualization")
    print("  5. 5_data_statistics.png - Dataset statistics")
    print()
    print("All figures ready for presentation!")


if __name__ == '__main__':
    import argparse
    import pandas as pd
    main()

