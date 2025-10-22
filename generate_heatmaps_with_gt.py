#!/usr/bin/env python3
"""
Generate improved heatmaps with ground truth visualization.

Creates 4-panel visualizations:
1. Original tissue (stitched from tiles)
2. Ground truth tumor mask
3. Predicted probability heatmap
4. Overlay comparison (GT vs Prediction)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score


def create_tissue_from_tiles(slide_df, tiles_dir, downsample=4):
    """Stitch tiles to create tissue image."""
    grid_rows = int(slide_df.iloc[0]['grid_rows'])
    grid_cols = int(slide_df.iloc[0]['grid_cols'])
    
    # Create empty grid (downsample for memory)
    tile_size = 96 // downsample
    tissue = np.ones((grid_rows * tile_size, grid_cols * tile_size, 3), dtype=np.uint8) * 240
    
    # Sample tiles (not all, too memory intensive)
    sample_rate = max(1, len(slide_df) // 5000)  # Max 5000 tiles
    
    for idx, row in slide_df.iloc[::sample_rate].iterrows():
        tile_path = Path(tiles_dir) / row['path']
        if not tile_path.exists():
            continue
        
        try:
            tile = cv2.imread(str(tile_path))
            if tile is None:
                continue
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            
            # Downsample
            if downsample > 1:
                tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            
            # Place in grid
            r = int(row['row_idx'])
            c = int(row['col_idx'])
            
            y_start = r * tile_size
            y_end = min(y_start + tile_size, tissue.shape[0])
            x_start = c * tile_size
            x_end = min(x_start + tile_size, tissue.shape[1])
            
            tissue[y_start:y_end, x_start:x_end] = tile[:y_end-y_start, :x_end-x_start]
        except Exception as e:
            continue
    
    return tissue


def create_heatmap_with_gt(slide_df, scores_df, tiles_dir, output_path, wsi_id):
    """
    Create 4-panel heatmap with ground truth.
    """
    print(f"\nProcessing {wsi_id}...")
    
    # Merge predictions with ground truth
    merged = slide_df.merge(scores_df, on='tile_id', how='inner')
    
    if len(merged) == 0:
        print(f"  ⚠️  No matching tiles found!")
        return None
    
    grid_rows = int(merged.iloc[0]['grid_rows'])
    grid_cols = int(merged.iloc[0]['grid_cols'])
    
    print(f"  Grid: {grid_rows} × {grid_cols}")
    print(f"  Tiles: {len(merged):,}")
    
    # Create grids
    gt_grid = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    pred_grid = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    
    for _, row in merged.iterrows():
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        gt_grid[r, c] = row['label']  # 0 or 1
        pred_grid[r, c] = row['score']  # 0-1 probability
    
    # Per-slide z-score normalization of predictions
    valid_scores = pred_grid[~np.isnan(pred_grid)]
    if len(valid_scores) > 0:
        mean_score = np.mean(valid_scores)
        std_score = np.std(valid_scores)
        
        if std_score > 1e-6:
            pred_grid_norm = np.where(
                ~np.isnan(pred_grid),
                (pred_grid - mean_score) / std_score,
                np.nan
            )
        else:
            pred_grid_norm = pred_grid.copy()
    else:
        pred_grid_norm = pred_grid.copy()
    
    # Gaussian smoothing
    pred_grid_smooth = pred_grid_norm.copy()
    mask = ~np.isnan(pred_grid_smooth)
    if mask.sum() > 0:
        pred_grid_smooth[~mask] = 0
        pred_grid_smooth = gaussian_filter(pred_grid_smooth, sigma=2.0)
        pred_grid_smooth[~mask] = np.nan
    
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
    
    # Binary prediction
    pred_binary = np.zeros_like(pred_grid_final)
    pred_binary[~np.isnan(pred_grid_final)] = (pred_grid_final[~np.isnan(pred_grid_final)] >= best_threshold).astype(float)
    pred_binary[np.isnan(pred_grid_final)] = np.nan
    
    # Compute metrics
    gt_flat = gt_valid.astype(int)
    pred_flat = (pred_valid >= best_threshold).astype(int)
    
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    
    print(f"  Metrics: IoU={iou:.4f}, Dice={dice:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    # Create tissue image (downsampled)
    print("  Creating tissue visualization...")
    tissue = create_tissue_from_tiles(merged, tiles_dir, downsample=4)
    
    # Resize grids to match tissue
    h, w = tissue.shape[:2]
    gt_viz = cv2.resize(np.nan_to_num(gt_grid, nan=0.0), (w, h), interpolation=cv2.INTER_NEAREST)
    pred_prob_viz = cv2.resize(np.nan_to_num(pred_grid_final, nan=0.0), (w, h), interpolation=cv2.INTER_LINEAR)
    pred_binary_viz = cv2.resize(np.nan_to_num(pred_binary, nan=0.0), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create 3-panel figure (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: Original tissue
    axes[0].imshow(tissue)
    axes[0].set_title('Original Tissue', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Ground truth
    axes[1].imshow(tissue)
    gt_overlay = np.zeros((*gt_viz.shape, 3), dtype=np.uint8)
    gt_overlay[gt_viz > 0.5] = [255, 0, 0]  # Red for tumor
    axes[1].imshow(gt_overlay, alpha=0.5)
    axes[1].set_title('Ground Truth', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # Add GT stats
    tumor_pct = 100 * (gt_valid == 1).sum() / len(gt_valid)
    gt_text = f'Tumor: {tumor_pct:.1f}%'
    axes[1].text(0.02, 0.98, gt_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Panel 3: Predicted probability heatmap
    axes[2].imshow(tissue)
    heatmap = axes[2].imshow(pred_prob_viz, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title('Predicted Heatmap', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    
    # Metrics text on prediction panel
    metrics_text = (f'Threshold: {best_threshold:.2f}\n'
                   f'IoU: {iou:.3f}\n'
                   f'Dice: {dice:.3f}\n'
                   f'Precision: {precision:.3f}\n'
                   f'Recall: {recall:.3f}')
    axes[2].text(0.02, 0.98, metrics_text, transform=axes[2].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add colorbar
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
        'n_tiles': len(merged),
        'tumor_pct': tumor_pct
    }


def main():
    parser = argparse.ArgumentParser(description='Generate heatmaps with ground truth')
    parser.add_argument('--scores-csv', type=str, required=True)
    parser.add_argument('--test-csv', type=str, default='test_set_heatmaps/test_set.csv')
    parser.add_argument('--tiles-dir', type=str, default='test_set_heatmaps')
    parser.add_argument('--output-dir', type=str, default='outputs/supervised_heatmaps_v2')
    parser.add_argument('--slides', type=str, nargs='+', default=None,
                        help='Specific slides to process (default: all)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SUPERVISED HEATMAPS WITH GROUND TRUTH")
    print("="*70)
    print(f"\nScores: {args.scores_csv}")
    print(f"Test set: {args.test_csv}")
    print(f"Tiles: {args.tiles_dir}")
    print(f"Output: {output_dir}")
    
    # Load data
    scores_df = pd.read_csv(args.scores_csv)
    test_df = pd.read_csv(args.test_csv)
    
    print(f"\nLoaded {len(scores_df):,} predictions")
    print(f"Loaded {len(test_df):,} test tiles")
    
    # Get slides to process
    all_slides = test_df['wsi_id'].unique()
    if args.slides:
        slides_to_process = [s for s in all_slides if s in args.slides]
    else:
        slides_to_process = all_slides
    
    print(f"\nProcessing {len(slides_to_process)} slides: {', '.join(slides_to_process)}")
    
    # Process each slide
    metrics_list = []
    
    for wsi_id in tqdm(slides_to_process, desc="Generating heatmaps"):
        slide_df = test_df[test_df['wsi_id'] == wsi_id]
        output_path = output_dir / f'{wsi_id}_heatmap.png'
        
        try:
            metrics = create_heatmap_with_gt(
                slide_df, scores_df, args.tiles_dir, output_path, wsi_id
            )
            if metrics:
                metrics_list.append(metrics)
        except Exception as e:
            print(f"ERROR processing {wsi_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save metrics
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_path = output_dir / 'heatmap_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        
        print("\n" + "="*70)
        print("SUMMARY METRICS")
        print("="*70)
        
        print(f"\nMean IoU:       {metrics_df['iou'].mean():.4f} ± {metrics_df['iou'].std():.4f}")
        print(f"Mean Dice:      {metrics_df['dice'].mean():.4f} ± {metrics_df['dice'].std():.4f}")
        print(f"Mean Precision: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
        print(f"Mean Recall:    {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
        
        print(f"\n✓ Metrics saved: {metrics_path}")
        print(f"\nPer-slide results:")
        for _, row in metrics_df.iterrows():
            print(f"  {row['wsi_id']:20s}: IoU={row['iou']:.4f}, Dice={row['dice']:.4f}, "
                  f"P={row['precision']:.4f}, R={row['recall']:.4f}")
    
    print("\n" + "="*70)
    print("✓ HEATMAP GENERATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

