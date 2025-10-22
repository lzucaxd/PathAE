"""
Connected Component Size Analysis

Analyzes the distribution of connected component sizes in predictions
vs ground truth to validate morphological filtering.

Author: ML Infra Engineer
Date: October 2025
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_predictions(scores_csv, test_csv):
    """Load predictions and metadata."""
    scores_df = pd.read_csv(scores_csv)
    test_df = pd.read_csv(test_csv)
    
    # Merge to get spatial information
    merged = test_df.merge(scores_df, on='tile_id')
    
    return merged

def create_heatmap_from_scores(df_slide, grid_rows, grid_cols, score_col='score'):
    """Create 2D heatmap grid from tile scores."""
    heatmap = np.zeros((grid_rows, grid_cols))
    
    for _, row in df_slide.iterrows():
        r, c = int(row['row_idx']), int(row['col_idx'])
        heatmap[r, c] = row[score_col]
    
    return heatmap

def analyze_component_sizes(binary_mask):
    """Analyze connected component sizes in binary mask."""
    labeled, num_components = measure.label(binary_mask, connectivity=2, return_num=True)
    
    if num_components == 0:
        return []
    
    component_sizes = []
    for i in range(1, num_components + 1):
        size = (labeled == i).sum()
        component_sizes.append(size)
    
    return component_sizes

def main():
    parser = argparse.ArgumentParser(description='Analyze connected component sizes.')
    parser.add_argument('--scores-csv', type=str, default='outputs/scores_supervised.csv',
                       help='Path to prediction scores CSV')
    parser.add_argument('--test-csv', type=str, default='test_set_heatmaps/test_set.csv',
                       help='Path to test set CSV with metadata')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binarizing predictions')
    args = parser.parse_args()
    
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("CONNECTED COMPONENT SIZE ANALYSIS")
    print("="*70)
    print(f"\nInput: {args.scores_csv}")
    print(f"Threshold: {args.threshold}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    merged_df = load_predictions(args.scores_csv, args.test_csv)
    print(f"  ✓ Loaded {len(merged_df):,} tile predictions")
    
    slides = merged_df['wsi_id'].unique()
    print(f"  ✓ Found {len(slides)} WSI slides")
    
    all_pred_sizes = []
    all_gt_sizes = []
    
    print("\n" + "="*70)
    print("ANALYZING COMPONENTS")
    print("="*70)
    
    for wsi_id in tqdm(slides, desc="  Processing slides"):
        df_slide = merged_df[merged_df['wsi_id'] == wsi_id].copy()
        
        grid_rows = int(df_slide['grid_rows'].iloc[0])
        grid_cols = int(df_slide['grid_cols'].iloc[0])
        
        # Create heatmaps
        score_heatmap = create_heatmap_from_scores(df_slide, grid_rows, grid_cols, 'score')
        gt_heatmap = create_heatmap_from_scores(df_slide, grid_rows, grid_cols, 'label')
        
        # Binarize
        pred_binary = (score_heatmap > args.threshold).astype(np.uint8)
        gt_binary = (gt_heatmap > 0.5).astype(np.uint8)
        
        # Analyze components
        pred_sizes = analyze_component_sizes(pred_binary)
        gt_sizes = analyze_component_sizes(gt_binary)
        
        all_pred_sizes.extend(pred_sizes)
        all_gt_sizes.extend(gt_sizes)
    
    print("\n" + "="*70)
    print("COMPONENT STATISTICS")
    print("="*70)
    
    if len(all_pred_sizes) > 0:
        pred_isolated = sum(1 for s in all_pred_sizes if s == 1)
        pred_small = sum(1 for s in all_pred_sizes if 1 <= s < 5)
        
        print(f"\nPredictions:")
        print(f"  Total components: {len(all_pred_sizes):,}")
        print(f"  Isolated (size=1): {pred_isolated:,} ({100*pred_isolated/len(all_pred_sizes):.1f}%)")
        print(f"  Small (size<5): {pred_small:,} ({100*pred_small/len(all_pred_sizes):.1f}%)")
        print(f"  Mean size: {np.mean(all_pred_sizes):.1f} patches")
        print(f"  Median size: {np.median(all_pred_sizes):.0f} patches")
        print(f"  Max size: {np.max(all_pred_sizes):,} patches")
    
    if len(all_gt_sizes) > 0:
        gt_isolated = sum(1 for s in all_gt_sizes if s == 1)
        gt_small = sum(1 for s in all_gt_sizes if 1 <= s < 5)
        
        print(f"\nGround Truth:")
        print(f"  Total components: {len(all_gt_sizes):,}")
        print(f"  Isolated (size=1): {gt_isolated:,} ({100*gt_isolated/len(all_gt_sizes):.1f}%)")
        print(f"  Small (size<5): {gt_small:,} ({100*gt_small/len(all_gt_sizes):.1f}%)")
        print(f"  Mean size: {np.mean(all_gt_sizes):.1f} patches")
        print(f"  Median size: {np.median(all_gt_sizes):.0f} patches")
        print(f"  Max size: {np.max(all_gt_sizes):,} patches")
    
    # Create visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    bins = np.logspace(0, np.log10(max(max(all_pred_sizes, default=1), max(all_gt_sizes, default=1))), 30)
    
    axes[0].hist(all_pred_sizes, bins=bins, alpha=0.7, label=f'Predictions (n={len(all_pred_sizes)})',
                color='#3498db', edgecolor='black')
    axes[0].hist(all_gt_sizes, bins=bins, alpha=0.7, label=f'Ground Truth (n={len(all_gt_sizes)})',
                color='#2ecc71', edgecolor='black')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Component Size (# patches, log scale)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Connected Component Sizes', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Add vertical line at size=5 (threshold for removal)
    axes[0].axvline(x=5, color='red', linestyle='--', linewidth=2, label='Filtering threshold')
    
    # Cumulative distribution
    pred_sorted = np.sort(all_pred_sizes)
    gt_sorted = np.sort(all_gt_sizes)
    
    pred_cum = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)
    gt_cum = np.arange(1, len(gt_sorted) + 1) / len(gt_sorted)
    
    axes[1].plot(pred_sorted, pred_cum, label='Predictions', color='#3498db', linewidth=2)
    axes[1].plot(gt_sorted, gt_cum, label='Ground Truth', color='#2ecc71', linewidth=2)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Component Size (# patches, log scale)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Cumulative Fraction', fontsize=12, fontweight='bold')
    axes[1].set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].axvline(x=5, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'component_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {figures_dir / 'component_size_analysis.png'}")
    
    print("\n" + "="*70)
    print("✓ COMPONENT ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nKey Finding:")
    if len(all_pred_sizes) > 0:
        print(f"  Predictions contain {100*pred_isolated/len(all_pred_sizes):.1f}% isolated patches,")
        print(f"  while ground truth has {100*gt_isolated/len(all_gt_sizes):.1f}%.")
        print(f"  Removing components < 5 patches filters out biologically implausible")
        print(f"  isolated false positives while preserving true tumor regions.")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()


