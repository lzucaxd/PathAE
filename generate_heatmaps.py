#!/usr/bin/env python3
"""
Generate presentation-quality heatmaps overlaid on WSIs.

Takes reconstruction errors from autoencoder and creates:
1. Full-slide heatmaps
2. Overlay visualizations
3. Side-by-side comparisons with ground truth
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
from tqdm import tqdm


def create_heatmap_from_scores(test_csv, scores_csv, wsi_path, mask_path, output_path, 
                                 canvas_level=4, alpha=0.5):
    """
    Create heatmap overlay from reconstruction errors.
    
    Args:
        test_csv: Path to test_set.csv with tile metadata
        scores_csv: Path to CSV with reconstruction errors (tile_id, score)
        wsi_path: Path to WSI file
        mask_path: Path to ground truth mask
        output_path: Where to save the heatmap image
        canvas_level: Pyramid level for canvas (higher = smaller image)
        alpha: Transparency for overlay
    """
    # Load data
    tiles_df = pd.read_csv(test_csv)
    scores_df = pd.read_csv(scores_csv)
    
    # Merge scores with tiles
    df = tiles_df.merge(scores_df, on='tile_id', how='left')
    df['score'] = df['score'].fillna(0)
    
    # Get grid dimensions from first tile
    grid_rows = int(df.iloc[0]['grid_rows'])
    grid_cols = int(df.iloc[0]['grid_cols'])
    
    print(f"  Reconstructing {grid_cols}×{grid_rows} heatmap...")
    
    # Create heatmap grid (NaN for background/missing tiles)
    heatmap_grid = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    
    for _, row in df.iterrows():
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        heatmap_grid[r, c] = row['score']
    
    # Normalize scores to 0-1 (only for tissue tiles with valid scores)
    valid_scores = heatmap_grid[~np.isnan(heatmap_grid)]
    if len(valid_scores) > 0:
        score_min = valid_scores.min()
        score_max = valid_scores.max()
        if score_max > score_min:
            heatmap_grid = (heatmap_grid - score_min) / (score_max - score_min)
    
    # Open WSI for thumbnail
    slide = openslide.OpenSlide(wsi_path)
    canvas_dims = slide.level_dimensions[canvas_level]
    canvas_downsample = slide.level_downsamples[canvas_level]
    
    # Read thumbnail
    thumbnail = slide.read_region((0, 0), canvas_level, canvas_dims)
    thumbnail_rgb = np.array(thumbnail.convert('RGB'))
    
    # Resize heatmap to match thumbnail (handle NaN properly)
    # Replace NaN with 0 for resize, then restore NaN mask
    heatmap_for_resize = heatmap_grid.copy()
    nan_mask = np.isnan(heatmap_for_resize)
    heatmap_for_resize[nan_mask] = 0
    
    heatmap_resized = cv2.resize(heatmap_for_resize, (canvas_dims[0], canvas_dims[1]), 
                                  interpolation=cv2.INTER_CUBIC)
    nan_mask_resized = cv2.resize(nan_mask.astype(np.uint8), (canvas_dims[0], canvas_dims[1]), 
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
    heatmap_resized[nan_mask_resized] = np.nan
    
    # Apply colormap (hot = high reconstruction error = likely tumor)
    # Create colored heatmap, but keep background transparent
    heatmap_colored = np.zeros((*heatmap_resized.shape, 3), dtype=np.uint8)
    valid_mask = ~nan_mask_resized
    
    if valid_mask.any():
        # Apply jet colormap only to valid (tissue) regions
        valid_scores = heatmap_resized[valid_mask]
        valid_colors = cm.jet(valid_scores)[:, :3]  # RGB
        heatmap_colored[valid_mask] = (valid_colors * 255).astype(np.uint8)
    
    # Create overlay (only blend where we have tissue)
    overlay = thumbnail_rgb.copy()
    if valid_mask.any():
        overlay[valid_mask] = cv2.addWeighted(
            thumbnail_rgb[valid_mask], 1-alpha,
            heatmap_colored[valid_mask], alpha, 0
        )
    
    # Load ground truth mask
    mask_pil = Image.open(mask_path)
    mask_array = np.array(mask_pil)
    mask_resized = cv2.resize(mask_array, (canvas_dims[0], canvas_dims[1]), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Create ground truth overlay
    mask_colored = np.zeros_like(thumbnail_rgb)
    mask_colored[mask_resized > 127] = [255, 0, 0]  # Red for tumor
    gt_overlay = cv2.addWeighted(thumbnail_rgb, 0.7, mask_colored, 0.3, 0)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Original thumbnail
    axes[0, 0].imshow(thumbnail_rgb)
    axes[0, 0].set_title('Original WSI', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title('Ground Truth (Red = Tumor)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Heatmap
    axes[1, 0].imshow(heatmap_colored)
    axes[1, 0].set_title('Reconstruction Error Heatmap', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Overlay
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Heatmap Overlay on WSI', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add colorbar
    fig.colorbar(cm.ScalarMappable(cmap='jet'), ax=axes, 
                 orientation='horizontal', fraction=0.05, pad=0.02,
                 label='Reconstruction Error (Low → High)')
    
    plt.suptitle(f'Tumor Detection Heatmap: {Path(wsi_path).stem}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    slide.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    # Also save individual components
    output_dir = Path(output_path).parent
    wsi_name = Path(wsi_path).stem
    
    cv2.imwrite(str(output_dir / f"{wsi_name}_heatmap_only.png"), 
                cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{wsi_name}_overlay.png"), 
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description='Generate heatmaps from reconstruction errors')
    parser.add_argument('--test-csv', type=str, required=True,
                        help='Path to test_set.csv')
    parser.add_argument('--scores-csv', type=str, required=True,
                        help='Path to CSV with reconstruction errors (tile_id, score)')
    parser.add_argument('--wsi-dir', type=str, default='cam16_prepped/wsi')
    parser.add_argument('--mask-dir', type=str, default='cam16_prepped/masks_tif')
    parser.add_argument('--output-dir', type=str, default='heatmaps')
    parser.add_argument('--canvas-level', type=int, default=4,
                        help='Pyramid level for visualization')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Heatmap transparency (0-1)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test set
    test_df = pd.read_csv(args.test_csv)
    wsi_ids = test_df['wsi_id'].unique()
    
    print(f"Generating heatmaps for {len(wsi_ids)} slides...")
    print()
    
    for wsi_id in wsi_ids:
        wsi_path = Path(args.wsi_dir) / f"{wsi_id}.tif"
        mask_path = Path(args.mask_dir) / f"{wsi_id}_mask.tif"
        output_path = output_dir / f"{wsi_id}_heatmap_comparison.png"
        
        if not wsi_path.exists() or not mask_path.exists():
            print(f"Skipping {wsi_id}: files not found")
            continue
        
        create_heatmap_from_scores(
            test_csv=args.test_csv,
            scores_csv=args.scores_csv,
            wsi_path=str(wsi_path),
            mask_path=str(mask_path),
            output_path=str(output_path),
            canvas_level=args.canvas_level,
            alpha=args.alpha,
        )
    
    print()
    print(f"✓ All heatmaps saved to: {output_dir}")


if __name__ == '__main__':
    main()

