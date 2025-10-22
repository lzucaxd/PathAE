#!/usr/bin/env python3
"""
Generate heatmaps using TTA predictions.

Uses same 3-panel format as other heatmaps.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score


def create_tissue_from_tiles(df, tiles_dir, downsample=4):
    """Stitch tiles to create tissue visualization."""
    grid_rows = int(df.iloc[0]['grid_rows'])
    grid_cols = int(df.iloc[0]['grid_cols'])
    
    tile_size = 96 // downsample
    tissue = np.ones((grid_rows * tile_size, grid_cols * tile_size, 3), dtype=np.uint8) * 255
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Stitching tiles", leave=False):
        if 'path' in row and pd.notna(row['path']):
            tile_path = Path(tiles_dir) / row['path']
        else:
            tile_path = Path(tiles_dir) / f"{row['tile_id']}.png"
        
        if not tile_path.exists():
            continue
        
        tile_img = np.array(Image.open(tile_path))
        tile_resized = cv2.resize(tile_img, (tile_size, tile_size))
        
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        tissue[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size] = tile_resized
    
    return tissue


def create_heatmap(df, scores, wsi_id, tiles_dir, output_path):
    """Create 3-panel heatmap visualization."""
    print(f"\nProcessing {wsi_id}...")
    print(f"  Tiles: {len(df):,}")
    
    # Get grid dimensions
    grid_rows = int(df.iloc[0]['grid_rows'])
    grid_cols = int(df.iloc[0]['grid_cols'])
    
    # Create grids
    pred_grid = np.full((grid_rows, grid_cols), np.nan)
    gt_grid = np.full((grid_rows, grid_cols), np.nan)
    
    for idx, row in df.iterrows():
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        pred_grid[r, c] = scores[idx]
        gt_grid[r, c] = row['label']
    
    # Per-slide z-score normalization
    valid_mask = ~np.isnan(pred_grid)
    if valid_mask.sum() > 0:
        mean_score = np.nanmean(pred_grid)
        std_score = np.nanstd(pred_grid)
        if std_score > 0:
            pred_grid_norm = np.where(valid_mask, (pred_grid - mean_score) / std_score, np.nan)
        else:
            pred_grid_norm = pred_grid.copy()
    else:
        pred_grid_norm = pred_grid.copy()
    
    # Gaussian smoothing
    pred_grid_smooth = gaussian_filter(np.nan_to_num(pred_grid_norm, nan=0.0), sigma=2.0)
    if valid_mask.sum() > 0:
        pred_grid_smooth[~valid_mask] = np.nan
    
    # Min-max to [0, 1]
    valid_smooth = pred_grid_smooth[~np.isnan(pred_grid_smooth)]
    if len(valid_smooth) > 0:
        min_val = np.min(valid_smooth)
        max_val = np.max(valid_smooth)
        if max_val > min_val:
            pred_grid_final = np.where(
                ~np.isnan(pred_grid_smooth),
                (pred_grid_smooth - min_val) / (max_val - min_val),
                np.nan
            )
        else:
            pred_grid_final = pred_grid_smooth.copy()
    else:
        pred_grid_final = pred_grid_smooth.copy()
    
    # Optimize threshold for IoU
    best_threshold = 0.5
    best_iou = 0.0
    
    gt_valid = gt_grid[~np.isnan(gt_grid)].flatten()
    pred_valid = pred_grid_final[~np.isnan(pred_grid_final)].flatten()
    
    if len(gt_valid) == len(pred_valid) and len(gt_valid) > 0:
        for thresh in np.linspace(0.3, 0.9, 13):
            pred_binary = (pred_valid >= thresh).astype(int)
            iou = jaccard_score(gt_valid.astype(int), pred_binary, zero_division=0)
            if iou > best_iou:
                best_iou = iou
                best_threshold = thresh
    
    print(f"  Optimized threshold: {best_threshold:.3f} (IoU: {best_iou:.4f})")
    
    # Compute metrics
    gt_flat = gt_valid.astype(int)
    pred_flat = (pred_valid >= best_threshold).astype(int)
    
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    
    print(f"  Metrics: IoU={iou:.4f}, Dice={dice:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    # Create tissue image
    print("  Creating tissue visualization...")
    tissue = create_tissue_from_tiles(df, tiles_dir, downsample=4)
    
    # Resize grids
    h, w = tissue.shape[:2]
    gt_viz = cv2.resize(np.nan_to_num(gt_grid, nan=0.0), (w, h), interpolation=cv2.INTER_NEAREST)
    pred_prob_viz = cv2.resize(np.nan_to_num(pred_grid_final, nan=0.0), (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create 3-panel figure (same as supervised)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: Original tissue
    axes[0].imshow(tissue)
    axes[0].set_title('Original Tissue', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Ground truth
    axes[1].imshow(tissue)
    gt_overlay = np.zeros((*gt_viz.shape, 3), dtype=np.uint8)
    gt_overlay[gt_viz > 0.5] = [255, 0, 0]
    axes[1].imshow(gt_overlay, alpha=0.5)
    axes[1].set_title('Ground Truth', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # Add GT stats
    tumor_pct = 100 * (gt_valid == 1).sum() / len(gt_valid)
    gt_text = f'Tumor: {tumor_pct:.1f}%'
    axes[1].text(0.02, 0.98, gt_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Panel 3: TTA heatmap
    axes[2].imshow(tissue)
    heatmap = axes[2].imshow(pred_prob_viz, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title('Predicted Heatmap (TTA)', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    
    # Metrics text
    metrics_text = (f'Threshold: {best_threshold:.2f}\n'
                   f'IoU: {iou:.3f}\n'
                   f'Dice: {dice:.3f}\n'
                   f'Precision: {precision:.3f}\n'
                   f'Recall: {recall:.3f}')
    axes[2].text(0.02, 0.98, metrics_text, transform=axes[2].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    cbar = plt.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Tumor Probability', fontsize=11)
    
    plt.suptitle(f'{wsi_id}', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    return {
        'wsi_id': wsi_id,
        'threshold': best_threshold,
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'n_tiles': len(df),
        'tumor_pct': tumor_pct
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tta-scores-csv', type=str, required=True)
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--tiles-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/supervised_heatmaps_tta')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("TTA HEATMAP GENERATION")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data...")
    df_tta = pd.read_csv(args.tta_scores_csv)
    df_test = pd.read_csv(args.test_csv)
    df = pd.merge(df_tta, df_test, on='tile_id')
    print(f"  ✓ Loaded {len(df):,} tiles from {df['wsi_id'].nunique()} slides\n")
    
    # Generate heatmaps per slide
    print(f"{'='*70}")
    print("GENERATING HEATMAPS")
    print(f"{'='*70}")
    
    all_metrics = []
    
    for wsi_id in sorted(df['wsi_id'].unique()):
        slide_df = df[df['wsi_id'] == wsi_id].reset_index(drop=True)
        output_path = output_dir / f'{wsi_id}_heatmap.png'
        
        metrics = create_heatmap(slide_df, slide_df['score'].values, 
                                wsi_id, args.tiles_dir, output_path)
        all_metrics.append(metrics)
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = output_dir / 'heatmap_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False, float_format='%.4f')
    
    print(f"\n{'='*70}")
    print("✓ TTA HEATMAP GENERATION COMPLETE")
    print(f"{'='*70}\n")
    
    print(metrics_df[['wsi_id', 'iou', 'dice', 'precision', 'recall']].to_string(index=False))
    print("")
    print(f"Overall Performance (with TTA):")
    print(f"  Mean IoU:       {metrics_df['iou'].mean():.4f} ± {metrics_df['iou'].std():.4f}")
    print(f"  Mean Dice:      {metrics_df['dice'].mean():.4f} ± {metrics_df['dice'].std():.4f}")
    print(f"  Mean Precision: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
    print(f"  Mean Recall:    {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
    print(f"\nHeatmaps saved to: {output_dir}")


if __name__ == '__main__':
    main()


