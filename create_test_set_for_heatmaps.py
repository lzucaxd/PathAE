#!/usr/bin/env python3
"""
Create COMPLETE test set for heatmap evaluation.

Extracts ALL tiles from tumor slides in a grid pattern for:
1. Dense heatmap reconstruction
2. FROC computation
3. Pixel-level metrics
4. Presentation visualizations
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import openslide
from tqdm import tqdm


def extract_slide_grid(wsi_path, mask_path, wsi_id, level, patch, stride, output_dir):
    """
    Extract complete grid of tiles from a slide for heatmap evaluation.
    
    Uses MINIMAL filtering - we want complete coverage for heatmaps.
    
    Args:
        wsi_path: Path to WSI
        mask_path: Path to mask TIF
        wsi_id: Slide identifier
        level: Pyramid level
        patch: Patch size
        stride: Stride (should equal patch for non-overlapping grid)
        output_dir: Output directory
    
    Returns:
        List of tile metadata dicts
    """
    print(f"\n{'='*70}")
    print(f"Processing: {wsi_id}")
    print(f"{'='*70}")
    
    # Open slide
    slide = openslide.OpenSlide(wsi_path)
    level_dims = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    
    # Load mask
    mask_pil = Image.open(mask_path)
    mask_array = np.array(mask_pil)
    mask_downsample = slide.level_dimensions[0][0] / mask_array.shape[1]
    
    print(f"  WSI: {level_dims[0]} x {level_dims[1]} at level {level}")
    print(f"  Mask: {mask_array.shape[1]} x {mask_array.shape[0]}")
    print(f"  Downsample: WSI={downsample:.2f}x, Mask={mask_downsample:.2f}x")
    
    # Generate non-overlapping grid
    tiles = []
    patch_0 = int(patch * downsample)
    
    # Create grid coordinates
    rows = list(range(0, level_dims[1] - patch + 1, stride))
    cols = list(range(0, level_dims[0] - patch + 1, stride))
    
    total_positions = len(rows) * len(cols)
    print(f"  Grid: {len(cols)} x {len(rows)} = {total_positions:,} positions")
    
    # Create output directory
    tile_dir = output_dir / 'tiles' / wsi_id
    tile_dir.mkdir(parents=True, exist_ok=True)
    
    kept = 0
    rejected_bg = 0
    
    for row_idx, y in enumerate(tqdm(rows, desc=f"  Extracting rows", leave=False)):
        for col_idx, x in enumerate(cols):
            # Level 0 coordinates
            x0 = int(x * downsample)
            y0 = int(y * downsample)
            
            # Read tile
            try:
                tile_pil = slide.read_region((x0, y0), level, (patch, patch))
                tile_rgb = np.array(tile_pil.convert('RGB'))
            except Exception:
                continue
            
            # PCam-style HSV filtering:
            # 1. Convert to HSV
            # 2. Apply Gaussian blur to reduce noise
            # 3. Reject if max saturation < 0.07 (background/non-tissue)
            # 4. Reject if mean value < 0.1 (too dark) or > 0.9 (too bright/overexposed)
            
            # Convert to HSV (OpenCV HSV: H in [0,180], S in [0,255], V in [0,255])
            hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Normalize S and V to [0, 1]
            hsv[:, :, 1] /= 255.0  # Saturation
            hsv[:, :, 2] /= 255.0  # Value
            
            # Apply Gaussian blur to saturation channel (reduces noise)
            sat_blurred = cv2.GaussianBlur(hsv[:, :, 1], (7, 7), 0)
            
            # Check max saturation (tissue should have color)
            max_sat = sat_blurred.max()
            if max_sat < 0.07:
                rejected_bg += 1
                continue
            
            # Check mean value (avoid too dark or too bright patches)
            mean_val = hsv[:, :, 2].mean()
            if mean_val < 0.1 or mean_val > 0.9:
                rejected_bg += 1
                continue
            
            # Get mask value at this location
            mask_x = int(x0 / mask_downsample)
            mask_y = int(y0 / mask_downsample)
            mask_size = int(patch_0 / mask_downsample)
            
            mask_x1 = min(mask_x + mask_size, mask_array.shape[1])
            mask_y1 = min(mask_y + mask_size, mask_array.shape[0])
            
            if mask_x < mask_array.shape[1] and mask_y < mask_array.shape[0]:
                mask_patch = mask_array[mask_y:mask_y1, mask_x:mask_x1]
                mask_frac = (mask_patch > 127).sum() / mask_patch.size if mask_patch.size > 0 else 0.0
            else:
                mask_frac = 0.0
            
            # Save tile
            filename = f"{wsi_id}_L{level}_r{row_idx:04d}_c{col_idx:04d}.png"
            tile_path = tile_dir / filename
            
            tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(tile_path), tile_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            
            # Metadata
            tiles.append({
                'tile_id': filename[:-4],
                'wsi_id': wsi_id,
                'x0': x0,
                'y0': y0,
                'level': level,
                'patch': patch,
                'row_idx': row_idx,
                'col_idx': col_idx,
                'grid_rows': len(rows),
                'grid_cols': len(cols),
                'mask_frac': f"{mask_frac:.4f}",
                'label': 1 if mask_frac >= 0.05 else 0,
                'path': str(tile_path.relative_to(output_dir)),
            })
            
            kept += 1
    
    slide.close()
    
    print(f"  Kept: {kept:,} tiles ({100*kept/total_positions:.1f}%)")
    print(f"  Rejected: {rejected_bg:,} background")
    
    # Summary statistics
    tiles_df = pd.DataFrame(tiles)
    n_tumor = (tiles_df['label'] == 1).sum()
    n_normal = (tiles_df['label'] == 0).sum()
    print(f"  Labels: {n_tumor:,} tumor, {n_normal:,} normal")
    
    return tiles


def main():
    parser = argparse.ArgumentParser(
        description='Create complete test set for heatmap evaluation'
    )
    parser.add_argument('--wsi-dir', type=str, default='cam16_prepped/wsi')
    parser.add_argument('--mask-dir', type=str, default='cam16_prepped/masks_tif')
    parser.add_argument('--output', type=str, default='test_set_heatmaps')
    parser.add_argument('--level', type=int, default=2,
                        help='Pyramid level (2=5x mag, good for heatmaps)')
    parser.add_argument('--patch', type=int, default=96,
                        help='Patch size')
    parser.add_argument('--stride', type=int, default=96,
                        help='Stride (use patch size for non-overlapping grid)')
    
    args = parser.parse_args()
    
    # All 8 tumor slides with masks
    tumor_slides = [
        'tumor_008',
        'tumor_020',
        'tumor_023',
        'tumor_028',
        'tumor_036',
        'tumor_056',
        'tumor_086',
        'test_002',
    ]
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMPLETE TEST SET FOR HEATMAP EVALUATION")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  Slides: {len(tumor_slides)} tumor slides with masks")
    print(f"  Level: {args.level} (5× magnification)")
    print(f"  Patch: {args.patch}×{args.patch} pixels")
    print(f"  Stride: {args.stride} (non-overlapping grid)")
    print(f"  Filtering: MINIMAL (just reject blank background)")
    print()
    print(f"Purpose:")
    print(f"  ✓ Dense coverage for heatmap reconstruction")
    print(f"  ✓ Exact grid coordinates for visualization")
    print(f"  ✓ Complete tumor + normal distribution")
    print(f"  ✓ FROC and pixel-level metrics")
    print()
    
    all_tiles = []
    
    for wsi_id in tumor_slides:
        wsi_path = Path(args.wsi_dir) / f"{wsi_id}.tif"
        mask_path = Path(args.mask_dir) / f"{wsi_id}_mask.tif"
        
        if not wsi_path.exists():
            print(f"Warning: {wsi_id} WSI not found, skipping")
            continue
        if not mask_path.exists():
            print(f"Warning: {wsi_id} mask not found, skipping")
            continue
        
        slide_tiles = extract_slide_grid(
            wsi_path=str(wsi_path),
            mask_path=str(mask_path),
            wsi_id=wsi_id,
            level=args.level,
            patch=args.patch,
            stride=args.stride,
            output_dir=output_dir,
        )
        
        all_tiles.extend(slide_tiles)
    
    # Save CSV
    if all_tiles:
        df = pd.DataFrame(all_tiles)
        csv_path = output_dir / 'test_set.csv'
        df.to_csv(csv_path, index=False)
        
        print()
        print("="*70)
        print("TEST SET SUMMARY")
        print("="*70)
        print(f"Total tiles: {len(df):,}")
        print()
        
        # Per-slide summary
        print("Per-slide breakdown:")
        for wsi_id in df['wsi_id'].unique():
            slide_df = df[df['wsi_id'] == wsi_id]
            n_tumor = (slide_df['label'] == 1).sum()
            n_normal = (slide_df['label'] == 0).sum()
            grid_info = slide_df.iloc[0]
            print(f"  {wsi_id:15}: {len(slide_df):6,} tiles "
                  f"({grid_info['grid_cols']}×{grid_info['grid_rows']} grid, "
                  f"{n_tumor:,} tumor, {n_normal:,} normal)")
        
        print()
        print(f"✓ Test set saved to: {output_dir}")
        print(f"  Metadata: {csv_path}")
        print(f"  Tiles: {output_dir / 'tiles'}")
        print()
        print("NEXT STEPS:")
        print("  1. Train autoencoder on final_dataset/train + val")
        print("  2. Run inference on this test set")
        print("  3. Use generate_heatmaps.py to create visualizations")
        print("  4. Use compute_metrics.py for FROC, AUC, etc.")
        
    else:
        print("Error: No tiles extracted", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

