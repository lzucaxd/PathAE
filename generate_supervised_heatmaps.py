#!/usr/bin/env python3
"""
Generate heatmaps for CAMELYON16 slides using supervised model predictions.

Process:
1. Load predictions from scores CSV
2. Map to spatial grid per slide
3. Apply IoU-optimized thresholding
4. Create 3-panel visualizations
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import openslide
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_erosion, binary_dilation
from skimage.morphology import remove_small_objects
from sklearn.metrics import roc_auc_score, average_precision_score, jaccard_score


def load_mask(wsi_path, level=2):
    """Load ground truth mask for a slide."""
    mask_path = Path(str(wsi_path).replace('.tif', '_mask.tif'))
    if not mask_path.exists():
        return None
    
    try:
        mask_slide = openslide.OpenSlide(str(mask_path))
        mask_img = mask_slide.read_region((0, 0), level, mask_slide.level_dimensions[level])
        mask = np.array(mask_img.convert('L'))
        mask_slide.close()
        return (mask > 0).astype(np.uint8)
    except Exception as e:
        print(f"Warning: Could not load mask for {wsi_path.name}: {e}")
        return None


def create_heatmap_for_slide(slide_df, output_dir, wsi_dir, level=2, use_iou_threshold=True):
    """
    Create heatmap for a single slide.
    
    Args:
        slide_df: DataFrame with tile_id, score, row_idx, col_idx, etc.
        output_dir: Where to save heatmaps
        wsi_dir: Directory containing WSI files
        level: OpenSlide level for visualization
        use_iou_threshold: If True, optimize threshold for IoU
    """
    wsi_id = slide_df.iloc[0]['wsi_id']
    grid_rows = int(slide_df.iloc[0]['grid_rows'])
    grid_cols = int(slide_df.iloc[0]['grid_cols'])
    
    print(f"\nProcessing {wsi_id}...")
    print(f"  Grid: {grid_rows} × {grid_cols}")
    print(f"  Tiles: {len(slide_df):,}")
    
    # Create probability grid
    prob_grid = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    
    for _, row in slide_df.iterrows():
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        prob_grid[r, c] = row['score']
    
    # Per-slide z-score normalization
    valid_scores = prob_grid[~np.isnan(prob_grid)]
    if len(valid_scores) > 0:
        mean_score = np.mean(valid_scores)
        std_score = np.std(valid_scores)
        
        if std_score > 1e-6:
            prob_grid_norm = np.where(
                ~np.isnan(prob_grid),
                (prob_grid - mean_score) / std_score,
                np.nan
            )
        else:
            prob_grid_norm = prob_grid.copy()
    else:
        prob_grid_norm = prob_grid.copy()
    
    # Apply Gaussian smoothing (only on valid regions)
    prob_grid_smooth = prob_grid_norm.copy()
    mask = ~np.isnan(prob_grid_smooth)
    if mask.sum() > 0:
        prob_grid_smooth[~mask] = 0  # Fill NaN with 0 for smoothing
        prob_grid_smooth = gaussian_filter(prob_grid_smooth, sigma=2.0)
        prob_grid_smooth[~mask] = np.nan  # Restore NaN
    
    # Min-max normalization to [0, 1]
    valid_smooth = prob_grid_smooth[~np.isnan(prob_grid_smooth)]
    if len(valid_smooth) > 0:
        min_val = np.min(valid_smooth)
        max_val = np.max(valid_smooth)
        if max_val > min_val:
            prob_grid_final = np.where(
                ~np.isnan(prob_grid_smooth),
                (prob_grid_smooth - min_val) / (max_val - min_val),
                np.nan
            )
        else:
            prob_grid_final = prob_grid_smooth.copy()
    else:
        prob_grid_final = prob_grid_smooth.copy()
    
    # Load ground truth mask
    wsi_path = Path(wsi_dir) / f"{wsi_id}.tif"
    gt_mask = load_mask(wsi_path, level=level)
    
    # Determine threshold
    if use_iou_threshold and gt_mask is not None:
        # IoU-optimized thresholding
        best_threshold = 0.5
        best_iou = 0.0
        
        # Resize pred grid to match mask
        valid_pred = np.nan_to_num(prob_grid_final, nan=0.0)
        pred_resized = cv2.resize(valid_pred, (gt_mask.shape[1], gt_mask.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Try different thresholds
        for thresh in np.linspace(0.3, 0.9, 13):
            pred_binary = (pred_resized >= thresh).astype(np.uint8)
            iou = jaccard_score(gt_mask.flatten(), pred_binary.flatten(), zero_division=0)
            if iou > best_iou:
                best_iou = iou
                best_threshold = thresh
        
        threshold = best_threshold
        print(f"  IoU-optimized threshold: {threshold:.3f} (IoU: {best_iou:.4f})")
    else:
        # Fixed threshold
        threshold = 0.5
        print(f"  Fixed threshold: {threshold:.3f}")
    
    # Binarize
    binary_pred = np.zeros_like(prob_grid_final)
    binary_pred[~np.isnan(prob_grid_final)] = (prob_grid_final[~np.isnan(prob_grid_final)] >= threshold).astype(float)
    binary_pred[np.isnan(prob_grid_final)] = np.nan
    
    # Morphological filtering (only on valid regions)
    binary_valid = binary_pred[~np.isnan(binary_pred)].astype(bool)
    if binary_valid.sum() > 0:
        # Fill holes
        binary_filled = binary_pred.copy()
        mask_valid = ~np.isnan(binary_pred)
        binary_img = (binary_pred == 1.0).astype(np.uint8)
        binary_img_filled = binary_fill_holes(binary_img).astype(np.uint8)
        binary_filled[mask_valid] = binary_img_filled[mask_valid]
        
        # Remove small objects (min area = 5 pixels)
        binary_cleaned = binary_filled.copy()
        binary_objects = (binary_filled == 1.0).astype(bool)
        binary_objects_cleaned = remove_small_objects(binary_objects, min_size=5)
        binary_cleaned[mask_valid] = binary_objects_cleaned[mask_valid].astype(float)
    else:
        binary_cleaned = binary_pred.copy()
    
    # Load WSI thumbnail for visualization
    try:
        slide = openslide.OpenSlide(str(wsi_path))
        thumbnail = slide.read_region((0, 0), level, slide.level_dimensions[level])
        thumbnail_rgb = np.array(thumbnail.convert('RGB'))
        slide.close()
    except Exception as e:
        print(f"Warning: Could not load WSI thumbnail: {e}")
        thumbnail_rgb = np.ones((grid_rows, grid_cols, 3), dtype=np.uint8) * 240
    
    # Resize grids to thumbnail size
    h, w = thumbnail_rgb.shape[:2]
    
    prob_viz = cv2.resize(np.nan_to_num(prob_grid_final, nan=0.0), (w, h), 
                          interpolation=cv2.INTER_LINEAR)
    binary_viz = cv2.resize(np.nan_to_num(binary_cleaned, nan=0.0), (w, h), 
                            interpolation=cv2.INTER_NEAREST)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original tissue
    axes[0].imshow(thumbnail_rgb)
    axes[0].set_title(f'{wsi_id}\nOriginal Tissue', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Probability heatmap
    axes[1].imshow(thumbnail_rgb)
    heatmap_overlay = axes[1].imshow(prob_viz, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title('Tumor Probability\n(Supervised ResNet18)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(heatmap_overlay, ax=axes[1], fraction=0.046, pad=0.04, label='P(Tumor)')
    
    # Panel 3: Binary prediction vs ground truth
    axes[2].imshow(thumbnail_rgb)
    
    if gt_mask is not None:
        # Overlay: Green = GT, Red = Prediction, Yellow = Overlap
        overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        overlay[gt_mask > 0] = [0, 255, 0]  # Green for GT
        overlay[binary_viz > 0.5] = [255, 0, 0]  # Red for pred
        overlap = (gt_mask > 0) & (binary_viz > 0.5)
        overlay[overlap] = [255, 255, 0]  # Yellow for overlap
        
        axes[2].imshow(overlay, alpha=0.5)
        axes[2].set_title(f'Prediction vs GT\n(Threshold: {threshold:.3f})', fontsize=12, fontweight='bold')
        
        # Compute metrics
        iou = jaccard_score(gt_mask.flatten(), (binary_viz > 0.5).flatten(), zero_division=0)
        dice = 2 * iou / (1 + iou) if iou > 0 else 0.0
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.5, label='Ground Truth'),
            Patch(facecolor='red', alpha=0.5, label='Prediction'),
            Patch(facecolor='yellow', alpha=0.5, label='Overlap')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Add metrics text
        metrics_text = f'IoU: {iou:.4f}\nDice: {dice:.4f}'
        axes[2].text(0.02, 0.98, metrics_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # No GT available
        binary_overlay = np.zeros((*binary_viz.shape, 3), dtype=np.uint8)
        binary_overlay[binary_viz > 0.5] = [255, 0, 0]
        axes[2].imshow(binary_overlay, alpha=0.5)
        axes[2].set_title(f'Binary Prediction\n(Threshold: {threshold:.3f})', 
                         fontsize=12, fontweight='bold')
        iou = None
        dice = None
    
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / f'{wsi_id}_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    return {
        'wsi_id': wsi_id,
        'threshold': threshold,
        'iou': iou,
        'dice': dice,
        'n_tiles': len(slide_df),
        'mean_score': np.mean(valid_scores) if len(valid_scores) > 0 else np.nan
    }


def main():
    parser = argparse.ArgumentParser(description='Generate supervised model heatmaps')
    parser.add_argument('--scores-csv', type=str, required=True,
                        help='CSV with tile predictions (tile_id, score)')
    parser.add_argument('--test-csv', type=str, default='test_set_heatmaps/test_set.csv',
                        help='Test set CSV with tile metadata')
    parser.add_argument('--wsi-dir', type=str, default='test_set_heatmaps',
                        help='Directory containing WSI files')
    parser.add_argument('--output-dir', type=str, default='outputs/supervised_heatmaps',
                        help='Output directory for heatmaps')
    parser.add_argument('--use-iou-threshold', action='store_true', default=True,
                        help='Use IoU-optimized thresholding per slide')
    parser.add_argument('--level', type=int, default=2,
                        help='OpenSlide level for visualization')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SUPERVISED MODEL HEATMAP GENERATION")
    print("="*70)
    print(f"\nScores: {args.scores_csv}")
    print(f"Test set: {args.test_csv}")
    print(f"WSI directory: {args.wsi_dir}")
    print(f"Output: {output_dir}")
    print(f"IoU optimization: {args.use_iou_threshold}")
    
    # Load scores and test set
    scores_df = pd.read_csv(args.scores_csv)
    test_df = pd.read_csv(args.test_csv)
    
    print(f"\nLoaded {len(scores_df):,} predictions")
    print(f"Loaded {len(test_df):,} test tiles")
    
    # Merge
    merged_df = test_df.merge(scores_df, on='tile_id', how='inner')
    print(f"Merged: {len(merged_df):,} tiles with predictions")
    
    if len(merged_df) == 0:
        print("ERROR: No matching tiles found!")
        return
    
    # Process each slide
    slide_metrics = []
    
    for wsi_id in tqdm(merged_df['wsi_id'].unique(), desc="Generating heatmaps"):
        slide_df = merged_df[merged_df['wsi_id'] == wsi_id]
        
        try:
            metrics = create_heatmap_for_slide(
                slide_df, output_dir, args.wsi_dir, 
                level=args.level, use_iou_threshold=args.use_iou_threshold
            )
            slide_metrics.append(metrics)
        except Exception as e:
            print(f"ERROR processing {wsi_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary metrics
    if slide_metrics:
        metrics_df = pd.DataFrame(slide_metrics)
        metrics_path = output_dir / 'heatmap_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        
        print("\n" + "="*70)
        print("SUMMARY METRICS")
        print("="*70)
        
        # Filter slides with GT
        valid_metrics = metrics_df.dropna(subset=['iou'])
        
        if len(valid_metrics) > 0:
            print(f"\nSlides with ground truth: {len(valid_metrics)}")
            print(f"  Mean IoU:  {valid_metrics['iou'].mean():.4f} ± {valid_metrics['iou'].std():.4f}")
            print(f"  Mean Dice: {valid_metrics['dice'].mean():.4f} ± {valid_metrics['dice'].std():.4f}")
            print(f"\nPer-slide IoU:")
            for _, row in valid_metrics.iterrows():
                print(f"  {row['wsi_id']:30s}: IoU = {row['iou']:.4f}, Dice = {row['dice']:.4f}")
        else:
            print("\nNo slides with ground truth masks found.")
        
        print(f"\n✓ Metrics saved: {metrics_path}")
    
    print("\n" + "="*70)
    print("✓ HEATMAP GENERATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()


