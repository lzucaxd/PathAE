#!/usr/bin/env python3
"""
Proper WSI heatmap stitching from per-tile scores.

Algorithm:
1. Compute score per tile (MSE + SSIM)
2. Normalize per-slide (z-score)
3. Stitch into grid
4. Gaussian smoothing
5. Overlay on thumbnail
"""

import argparse
import numpy as np
import pandas as pd
import cv2
import openslide
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import cm


def stitch_heatmap(test_csv, scores_csv, wsi_id, wsi_path, mask_path, output_dir, 
                    threshold=None, smooth_sigma=2.0, canvas_level=4, alpha=0.5):
    """
    Create heatmap for a single WSI using proper grid stitching.
    
    Args:
        test_csv: Path to test_set.csv
        scores_csv: Path to reconstruction scores CSV
        wsi_id: Slide ID to process
        wsi_path: Path to WSI file
        mask_path: Path to ground truth mask
        output_dir: Output directory
        threshold: Anomaly threshold (or None for auto)
        smooth_sigma: Gaussian smoothing sigma
        canvas_level: Pyramid level for thumbnail
        alpha: Heatmap transparency
    """
    print(f"\n{'='*70}")
    print(f"Stitching heatmap: {wsi_id}")
    print(f"{'='*70}")
    
    # Load data
    tiles_df = pd.read_csv(test_csv)
    scores_df = pd.read_csv(scores_csv)
    
    # Filter for this slide
    slide_tiles = tiles_df[tiles_df['wsi_id'] == wsi_id].copy()
    
    if len(slide_tiles) == 0:
        print(f"No tiles found for {wsi_id}")
        return
    
    # Merge with scores
    slide_tiles = slide_tiles.merge(scores_df, on='tile_id', how='inner')
    
    print(f"  Tiles: {len(slide_tiles):,}")
    print(f"  Grid: {slide_tiles.iloc[0]['grid_cols']}×{slide_tiles.iloc[0]['grid_rows']}")
    
    # Step 2: Per-slide z-score normalization
    scores = slide_tiles['score'].values
    mean_score = scores.mean()
    std_score = scores.std()
    
    slide_tiles['score_zscore'] = (slide_tiles['score'] - mean_score) / (std_score + 1e-6)
    
    print(f"  Score stats: mean={mean_score:.6f}, std={std_score:.6f}")
    print(f"  Z-score range: [{slide_tiles['score_zscore'].min():.2f}, {slide_tiles['score_zscore'].max():.2f}]")
    
    # Step 3: Stitch into grid using max aggregation
    grid_rows = int(slide_tiles.iloc[0]['grid_rows'])
    grid_cols = int(slide_tiles.iloc[0]['grid_cols'])
    
    # Initialize grid with NaN (for background)
    heatmap_grid = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    
    for _, row in slide_tiles.iterrows():
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        
        # Use max aggregation (in case of overlaps)
        current = heatmap_grid[r, c]
        new_val = row['score_zscore']
        
        if np.isnan(current):
            heatmap_grid[r, c] = new_val
        else:
            heatmap_grid[r, c] = max(current, new_val)
    
    print(f"  Grid coverage: {(~np.isnan(heatmap_grid)).sum()} / {grid_rows*grid_cols} ({100*(~np.isnan(heatmap_grid)).sum()/(grid_rows*grid_cols):.1f}%)")
    
    # Step 4: Smooth seams with Gaussian blur (only on valid values)
    heatmap_smooth = heatmap_grid.copy()
    valid_mask = ~np.isnan(heatmap_grid)
    
    if smooth_sigma > 0 and valid_mask.any():
        # Fill NaN with mean for smoothing, then restore NaN
        temp = heatmap_grid.copy()
        temp[np.isnan(temp)] = np.nanmean(heatmap_grid)
        temp_smooth = gaussian_filter(temp, sigma=smooth_sigma)
        heatmap_smooth = temp_smooth
        heatmap_smooth[~valid_mask] = np.nan
    
    # Step 5: Min-max normalization for visualization [0, 1]
    heatmap_vis = heatmap_smooth.copy()
    valid_values = heatmap_vis[~np.isnan(heatmap_vis)]
    
    if len(valid_values) > 0:
        vmin = valid_values.min()
        vmax = valid_values.max()
        if vmax > vmin:
            heatmap_vis = (heatmap_vis - vmin) / (vmax - vmin)
        else:
            heatmap_vis = np.zeros_like(heatmap_vis)
    
    # Step 6: Create thumbnail
    slide = openslide.OpenSlide(wsi_path)
    canvas_dims = slide.level_dimensions[canvas_level]
    
    thumbnail = slide.read_region((0, 0), canvas_level, canvas_dims)
    thumbnail_rgb = np.array(thumbnail.convert('RGB'))
    
    # Resize heatmap to match thumbnail
    heatmap_resized = cv2.resize(
        np.nan_to_num(heatmap_vis, nan=0),
        (canvas_dims[0], canvas_dims[1]),
        interpolation=cv2.INTER_CUBIC
    )
    
    valid_mask_resized = cv2.resize(
        valid_mask.astype(np.uint8),
        (canvas_dims[0], canvas_dims[1]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    
    # Apply colormap (jet: blue=low, red=high)
    heatmap_colored = np.zeros((*heatmap_resized.shape, 3), dtype=np.uint8)
    
    if valid_mask_resized.any():
        valid_scores = heatmap_resized[valid_mask_resized]
        valid_colors = cm.jet(valid_scores)[:, :3]
        heatmap_colored[valid_mask_resized] = (valid_colors * 255).astype(np.uint8)
    
    # Create overlay
    overlay = thumbnail_rgb.copy()
    if valid_mask_resized.any():
        overlay[valid_mask_resized] = cv2.addWeighted(
            thumbnail_rgb[valid_mask_resized], 1-alpha,
            heatmap_colored[valid_mask_resized], alpha, 0
        )
    
    # Load ground truth mask
    mask_pil = Image.open(mask_path)
    mask_array = np.array(mask_pil)
    mask_resized = cv2.resize(
        mask_array,
        (canvas_dims[0], canvas_dims[1]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Ground truth overlay
    mask_colored = np.zeros_like(thumbnail_rgb)
    mask_colored[mask_resized > 127] = [255, 0, 0]
    gt_overlay = cv2.addWeighted(thumbnail_rgb, 0.7, mask_colored, 0.3, 0)
    
    # Binary map (if threshold provided)
    binary_map = None
    if threshold is not None:
        # Use original (non-normalized) scores for thresholding
        binary_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
        for _, row in slide_tiles.iterrows():
            r = int(row['row_idx'])
            c = int(row['col_idx'])
            if row['score'] >= threshold:
                binary_grid[r, c] = 255
        
        binary_map = cv2.resize(
            binary_grid,
            (canvas_dims[0], canvas_dims[1]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Create comparison figure
    n_panels = 4 if threshold is None else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels*5, 5))
    
    # Panel 1: Original
    axes[0].imshow(thumbnail_rgb)
    axes[0].set_title('Original WSI', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Ground truth
    axes[1].imshow(gt_overlay)
    axes[1].set_title('Ground Truth\n(Red = Tumor)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Heatmap
    axes[2].imshow(heatmap_colored)
    axes[2].set_title('Reconstruction Error\n(Z-score, Smoothed)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Panel 4: Overlay
    axes[3].imshow(overlay)
    axes[3].set_title(f'Overlay (α={alpha})', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # Panel 5: Binary (if threshold provided)
    if threshold is not None:
        binary_overlay = thumbnail_rgb.copy()
        binary_overlay[binary_map > 127] = [255, 255, 0]  # Yellow for detections
        axes[4].imshow(binary_overlay)
        axes[4].set_title(f'Binary Map\n(threshold={threshold:.4f})', fontsize=12, fontweight='bold')
        axes[4].axis('off')
    
    plt.suptitle(f'Tumor Detection Heatmap: {wsi_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'{wsi_id}_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    # Save individual components
    cv2.imwrite(str(output_dir / f'{wsi_id}_heatmap_only.png'),
                cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f'{wsi_id}_overlay.png'),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Save raw heatmap grid for metrics
    np.save(output_dir / f'{wsi_id}_heatmap_grid.npy', {
        'heatmap_raw': heatmap_grid,  # Z-score values
        'heatmap_smooth': heatmap_smooth,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
    })
    
    slide.close()
    
    return heatmap_smooth


def main():
    parser = argparse.ArgumentParser(description='Stitch WSI heatmap from tile scores')
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--scores-csv', type=str, required=True)
    parser.add_argument('--wsi-id', type=str, help='Specific slide to process (or all if not specified)')
    parser.add_argument('--wsi-dir', type=str, default='cam16_prepped/wsi')
    parser.add_argument('--mask-dir', type=str, default='cam16_prepped/masks_tif')
    parser.add_argument('--output-dir', type=str, default='heatmaps_v2')
    parser.add_argument('--threshold', type=float, help='Anomaly threshold (or load from threshold.npy)')
    parser.add_argument('--smooth-sigma', type=float, default=2.0)
    parser.add_argument('--canvas-level', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Load threshold if not provided
    threshold = args.threshold
    if threshold is None and Path('threshold.npy').exists():
        threshold_data = np.load('threshold.npy', allow_pickle=True).item()
        threshold = threshold_data['threshold']
        print(f"Loaded threshold: {threshold:.6f}")
    
    # Load test set
    test_df = pd.read_csv(args.test_csv)
    
    # Get list of slides to process
    if args.wsi_id:
        wsi_ids = [args.wsi_id]
    else:
        wsi_ids = test_df['wsi_id'].unique()
    
    print(f"Processing {len(wsi_ids)} slides...")
    print()
    
    for wsi_id in wsi_ids:
        wsi_path = Path(args.wsi_dir) / f"{wsi_id}.tif"
        mask_path = Path(args.mask_dir) / f"{wsi_id}_mask.tif"
        
        if not wsi_path.exists():
            print(f"Warning: {wsi_id} WSI not found, skipping")
            continue
        if not mask_path.exists():
            print(f"Warning: {wsi_id} mask not found, skipping")
            continue
        
        stitch_heatmap(
            test_csv=args.test_csv,
            scores_csv=args.scores_csv,
            wsi_id=wsi_id,
            wsi_path=str(wsi_path),
            mask_path=str(mask_path),
            output_dir=args.output_dir,
            threshold=threshold,
            smooth_sigma=args.smooth_sigma,
            canvas_level=args.canvas_level,
            alpha=args.alpha,
        )
    
    print()
    print(f"✓ All heatmaps saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

