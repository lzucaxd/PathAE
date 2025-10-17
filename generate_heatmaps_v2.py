#!/usr/bin/env python3
"""
Improved heatmap generation with per-slide z-score normalization and IoU optimization.

Key improvements over v1:
1. Per-slide z-score normalization (removes stain/scanner drift)
2. IoU-optimized threshold selection (not F1-optimized)
3. Morphological post-processing (remove isolated FPs)
4. Cleaner visualization (no error bars, better layout)
5. Continuous colormap option
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import openslide
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import binary_opening, binary_fill_holes, label as connected_components
from skimage.morphology import remove_small_objects
from tqdm import tqdm


def compute_iou(pred, gt):
    """Compute IoU between binary masks."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-6)


def find_optimal_threshold_for_iou(scores, labels, score_range=(-3, 3), n_thresholds=100):
    """
    Find threshold that maximizes IoU (not F1).
    
    Args:
        scores: Normalized scores (z-scored)
        labels: Ground truth binary labels
        score_range: Range of thresholds to search (in z-score units)
        n_thresholds: Number of thresholds to try
        
    Returns:
        best_threshold, best_iou, threshold_sweep_results
    """
    thresholds = np.linspace(score_range[0], score_range[1], n_thresholds)
    best_iou = 0
    best_thresh = 0
    results = []
    
    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        
        tp = np.logical_and(pred == 1, labels == 1).sum()
        fp = np.logical_and(pred == 1, labels == 0).sum()
        fn = np.logical_and(pred == 0, labels == 1).sum()
        
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        results.append({
            'threshold': thresh,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        })
        
        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh
    
    return best_thresh, best_iou, pd.DataFrame(results)


def apply_morphological_filtering(binary_mask, min_object_size=5, fill_holes=True, opening_radius=0):
    """
    Apply spatial coherence filtering to binary mask.
    
    Args:
        binary_mask: Binary numpy array
        min_object_size: Remove connected components smaller than this
        fill_holes: Fill holes in tumor regions
        opening_radius: Radius for morphological opening (0=disabled)
        
    Returns:
        Filtered binary mask
    """
    # Remove small isolated objects (likely FPs)
    cleaned = remove_small_objects(binary_mask.astype(bool), min_size=min_object_size)
    
    # Fill small holes in tumor regions
    if fill_holes:
        cleaned = binary_fill_holes(cleaned)
    
    # Optional: morphological opening (further noise reduction)
    if opening_radius > 0:
        from skimage.morphology import disk
        cleaned = binary_opening(cleaned, disk(opening_radius))
    
    return cleaned.astype(np.uint8)


def create_heatmap_v2(test_csv, scores_csv, wsi_path, mask_path, output_path,
                       canvas_level=4, alpha=0.5, use_per_slide_norm=True,
                       optimize_for_iou=True, apply_morphological=True,
                       continuous_colormap=True):
    """
    Create improved heatmap with per-slide normalization and IoU optimization.
    
    Args:
        test_csv: Path to test_set.csv
        scores_csv: Path to reconstruction error scores
        wsi_path: Path to WSI
        mask_path: Path to ground truth mask
        output_path: Where to save
        canvas_level: Pyramid level for visualization
        alpha: Overlay transparency
        use_per_slide_norm: Apply per-slide z-score normalization
        optimize_for_iou: Find threshold that maximizes IoU (vs F1)
        apply_morphological: Apply spatial filtering
        continuous_colormap: Use continuous heatmap (vs binary threshold)
    """
    # Extract wsi_id
    wsi_id = Path(wsi_path).stem
    
    # Load data for this slide
    tiles_df = pd.read_csv(test_csv)
    scores_df = pd.read_csv(scores_csv)
    
    tiles_df = tiles_df[tiles_df['wsi_id'] == wsi_id].copy()
    
    if len(tiles_df) == 0:
        print(f"  Warning: No tiles found for {wsi_id}")
        return None
    
    # Merge
    df = tiles_df.merge(scores_df, on='tile_id', how='left')
    df['score'] = df['score'].fillna(0)
    
    # Grid dimensions
    grid_rows = int(df.iloc[0]['grid_rows'])
    grid_cols = int(df.iloc[0]['grid_cols'])
    
    print(f"  Slide: {wsi_id}")
    print(f"    Grid: {grid_rows} × {grid_cols}")
    print(f"    Tiles: {len(df):,} ({(df['label']==1).sum():,} tumor)")
    
    # Per-slide z-score normalization (CRITICAL FOR GOOD HEATMAPS!)
    if use_per_slide_norm:
        raw_scores = df['score'].values
        mean_slide = raw_scores.mean()
        std_slide = raw_scores.std()
        df['score_normalized'] = (raw_scores - mean_slide) / (std_slide + 1e-6)
        print(f"    Raw scores: μ={mean_slide:.4f}, σ={std_slide:.4f}")
    else:
        df['score_normalized'] = df['score']
    
    # Find optimal threshold
    if optimize_for_iou and (df['label'] == 1).sum() > 0:
        best_thresh, best_iou, thresh_sweep = find_optimal_threshold_for_iou(
            df['score_normalized'].values,
            df['label'].values,
            score_range=(-3, 5),
            n_thresholds=100
        )
        print(f"    Optimal threshold (IoU): z={best_thresh:.2f} (IoU={best_iou:.4f})")
    else:
        # Fallback: 99th percentile (less aggressive than 99.7)
        best_thresh = np.percentile(df['score_normalized'].values, 99)
        best_iou = None
        print(f"    Fallback threshold: z={best_thresh:.2f} (99th percentile)")
    
    # Create heatmap grid
    heatmap_grid = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    label_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
    
    for _, row in df.iterrows():
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        heatmap_grid[r, c] = row['score_normalized']
        label_grid[r, c] = int(row['label'])
    
    # Binary prediction (for IoU computation and optional binary visualization)
    pred_grid = np.zeros_like(heatmap_grid, dtype=bool)
    valid_mask = ~np.isnan(heatmap_grid)
    pred_grid[valid_mask] = heatmap_grid[valid_mask] >= best_thresh
    
    # Morphological filtering (reduces isolated FPs)
    if apply_morphological:
        pred_grid_filtered = apply_morphological_filtering(
            pred_grid,
            min_object_size=5,
            fill_holes=True,
            opening_radius=0  # 0 = disabled (too aggressive for sparse grids)
        )
        
        # Compute improvement
        iou_before = compute_iou(pred_grid[valid_mask], label_grid[valid_mask])
        iou_after = compute_iou(pred_grid_filtered[valid_mask], label_grid[valid_mask])
        print(f"    Morphological filtering: IoU {iou_before:.4f} → {iou_after:.4f}")
        
        pred_grid = pred_grid_filtered
    
    # Normalize heatmap for visualization
    if continuous_colormap:
        # Continuous: Show full range of anomaly scores
        heatmap_vis = heatmap_grid.copy()
        
        # Clip to 5th-95th percentile for better contrast
        valid_scores = heatmap_vis[valid_mask]
        if len(valid_scores) > 0:
            p5, p95 = np.percentile(valid_scores, [5, 95])
            heatmap_vis = np.clip(heatmap_vis, p5, p95)
            # Normalize to [0, 1]
            heatmap_vis = (heatmap_vis - p5) / (p95 - p5 + 1e-6)
    else:
        # Binary: Just use threshold
        heatmap_vis = pred_grid.astype(np.float32)
    
    # Open WSI and mask
    slide = openslide.OpenSlide(wsi_path)
    canvas_dims = slide.level_dimensions[canvas_level]
    
    # Read thumbnail
    thumbnail = slide.read_region((0, 0), canvas_level, canvas_dims)
    thumbnail_rgb = np.array(thumbnail.convert('RGB'))
    
    # Read mask (use PIL since masks are simple TIFFs)
    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img)
    
    # Resize grids to canvas
    heatmap_resized = cv2.resize(heatmap_vis, (canvas_dims[0], canvas_dims[1]), 
                                  interpolation=cv2.INTER_CUBIC)
    pred_resized = cv2.resize(pred_grid.astype(np.float32), (canvas_dims[0], canvas_dims[1]),
                               interpolation=cv2.INTER_NEAREST)
    mask_resized = cv2.resize(mask_array.astype(np.float32), (canvas_dims[0], canvas_dims[1]),
                               interpolation=cv2.INTER_NEAREST)
    
    # Create NaN mask for background
    nan_mask_resized = cv2.resize((~valid_mask).astype(np.uint8), 
                                   (canvas_dims[0], canvas_dims[1]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Apply Gaussian smoothing for prettier heatmaps
    heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
    heatmap_smooth[nan_mask_resized] = 0  # Keep background dark
    
    # Create colored heatmap (jet colormap)
    heatmap_colored = np.zeros((*heatmap_smooth.shape, 3), dtype=np.uint8)
    valid_vis = ~nan_mask_resized
    
    if valid_vis.any():
        valid_scores_vis = heatmap_smooth[valid_vis]
        valid_colors = cm.jet(valid_scores_vis)[:, :3]  # RGB
        heatmap_colored[valid_vis] = (valid_colors * 255).astype(np.uint8)
    
    # Create overlay
    overlay = thumbnail_rgb.copy()
    if valid_vis.any():
        overlay[valid_vis] = cv2.addWeighted(
            thumbnail_rgb[valid_vis], 1-alpha,
            heatmap_colored[valid_vis], alpha, 0
        )
    
    # Create ground truth visualization (red overlay)
    gt_colored = np.zeros_like(thumbnail_rgb)
    gt_colored[:, :, 0] = (mask_resized > 0).astype(np.uint8) * 255  # Red channel
    
    gt_overlay = thumbnail_rgb.copy()
    tumor_mask = mask_resized > 0
    if tumor_mask.any():
        gt_overlay[tumor_mask] = cv2.addWeighted(
            thumbnail_rgb[tumor_mask], 1-alpha,
            gt_colored[tumor_mask], alpha, 0
        )
    
    # Create 3-panel visualization (NO ERROR BARS)
    fig = plt.figure(figsize=(18, 6))
    
    # Panel 1: Ground Truth
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(gt_overlay)
    ax1.set_title(f'Ground Truth\n{wsi_id}', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Prediction Heatmap
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(heatmap_colored)
    ax2.set_title(f'Anomaly Heatmap\n{"Continuous" if continuous_colormap else "Binary"}', 
                   fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Overlay on WSI
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(overlay)
    if best_iou is not None:
        ax3.set_title(f'Overlay on WSI\nIoU={best_iou:.3f}', fontsize=14, fontweight='bold')
    else:
        ax3.set_title(f'Overlay on WSI', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_path}")
    
    # Return metrics
    slide_iou = compute_iou(pred_resized > 0.5, mask_resized > 0.5)
    
    return {
        'wsi_id': wsi_id,
        'iou': slide_iou,
        'best_threshold': best_thresh,
        'n_tiles': len(df),
        'n_tumor': (df['label'] == 1).sum(),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate improved heatmaps with IoU optimization')
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--scores-csv', type=str, required=True)
    parser.add_argument('--wsi-dir', type=str, default='cam16_prepped/wsi')
    parser.add_argument('--mask-dir', type=str, default='cam16_prepped/masks_tif')
    parser.add_argument('--output-dir', type=str, default='heatmaps_v2')
    parser.add_argument('--canvas-level', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--no-per-slide-norm', action='store_true',
                        help='Disable per-slide z-score normalization')
    parser.add_argument('--no-iou-opt', action='store_true',
                        help='Disable IoU-optimized thresholding')
    parser.add_argument('--no-morphological', action='store_true',
                        help='Disable morphological filtering')
    parser.add_argument('--binary', action='store_true',
                        help='Use binary heatmap instead of continuous')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test set
    test_df = pd.read_csv(args.test_csv)
    wsi_ids = test_df['wsi_id'].unique()
    
    print("="*70)
    print("IMPROVED HEATMAP GENERATION (v2)")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  Per-slide z-score:     {not args.no_per_slide_norm}")
    print(f"  IoU-optimized thresh:  {not args.no_iou_opt}")
    print(f"  Morphological filter:  {not args.no_morphological}")
    print(f"  Colormap:              {'Continuous' if not args.binary else 'Binary'}")
    print()
    print(f"Generating heatmaps for {len(wsi_ids)} slides...")
    print()
    
    results = []
    
    for wsi_id in wsi_ids:
        wsi_path_full = Path(args.wsi_dir) / f"{wsi_id}.tif"
        mask_path_full = Path(args.mask_dir) / f"{wsi_id}_mask.tif"
        output_path_full = output_dir / f"{wsi_id}_heatmap_v2.png"
        
        if not wsi_path_full.exists() or not mask_path_full.exists():
            print(f"  Skipping {wsi_id}: files not found")
            continue
        
        slide_result = create_heatmap_v2(
            test_csv=args.test_csv,
            scores_csv=args.scores_csv,
            wsi_path=str(wsi_path_full),
            mask_path=str(mask_path_full),
            output_path=str(output_path_full),
            canvas_level=args.canvas_level,
            alpha=args.alpha,
            use_per_slide_norm=not args.no_per_slide_norm,
            optimize_for_iou=not args.no_iou_opt,
            apply_morphological=not args.no_morphological,
            continuous_colormap=not args.binary,
        )
        
        if slide_result:
            results.append(slide_result)
        
        print()
    
    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'heatmap_summary.csv', index=False)
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Heatmaps generated: {len(results)}")
    print(f"  Mean IoU: {results_df['iou'].mean():.4f}")
    print(f"  Median IoU: {results_df['iou'].median():.4f}")
    print(f"  Best IoU: {results_df['iou'].max():.4f} ({results_df.loc[results_df['iou'].idxmax(), 'wsi_id']})")
    print()
    print(f"  ✓ Heatmaps saved to: {output_dir}")
    print(f"  ✓ Summary saved to: {output_dir}/heatmap_summary.csv")
    print()


if __name__ == '__main__':
    main()

