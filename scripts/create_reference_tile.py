#!/usr/bin/env python3
"""
Create a fixed reference tile for Macenko stain normalization.

This ensures consistent normalization across all tiles and runs.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import openslide
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from prep_cam16 import stain


def create_reference_tile(
    links_csv: str,
    output_path: str,
    level: int = 0,
    tile_size: int = 256,
    num_samples: int = 50,
):
    """
    Create a reference tile by sampling from multiple normal slides.
    
    Args:
        links_csv: Path to links.csv with slide information.
        output_path: Where to save the reference tile (.npy).
        level: Pyramid level to sample from.
        tile_size: Size of tile to sample.
        num_samples: Number of tiles to sample per slide.
    """
    # Load links
    df = pd.read_csv(links_csv)
    
    # Get normal training slides
    normal_train = df[(df['kind'] == 'normal') & (df['split'] == 'train')]
    
    if len(normal_train) == 0:
        print("Error: No normal training slides found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Sampling from {len(normal_train)} normal training slides...")
    
    all_candidates = []
    
    for idx, row in normal_train.iterrows():
        wsi_path = row['wsi_source']
        wsi_id = row['wsi_id']
        
        print(f"  Processing {wsi_id}...")
        
        try:
            slide = openslide.OpenSlide(wsi_path)
            
            # Sample multiple tiles from this slide
            for _ in range(num_samples // len(normal_train) + 1):
                ref_tile = stain.pick_ref_tile(slide, level, num_samples=1, tile_size=tile_size)
                if ref_tile is not None:
                    all_candidates.append(ref_tile)
            
            slide.close()
        except Exception as e:
            print(f"    Warning: Failed to process {wsi_id}: {e}")
            continue
    
    if len(all_candidates) == 0:
        print("Error: No valid reference tiles found", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nCollected {len(all_candidates)} candidate tiles")
    
    # Select the most "average" tile (closest to median statistics)
    # Compute mean saturation and value for each candidate
    scores = []
    for tile in all_candidates:
        import cv2
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        mean_sat = hsv[:, :, 1].mean()
        mean_val = hsv[:, :, 2].mean()
        # Score by distance from median
        score = abs(mean_sat - 100) + abs(mean_val - 180)
        scores.append(score)
    
    # Pick the most average one
    best_idx = np.argmin(scores)
    reference_tile = all_candidates[best_idx]
    
    # Save as numpy array
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, reference_tile)
    
    print(f"\n✓ Reference tile saved to: {output_path}")
    print(f"  Shape: {reference_tile.shape}")
    print(f"  Dtype: {reference_tile.dtype}")
    print(f"  Mean RGB: ({reference_tile[:,:,0].mean():.1f}, "
          f"{reference_tile[:,:,1].mean():.1f}, "
          f"{reference_tile[:,:,2].mean():.1f})")


def main():
    parser = argparse.ArgumentParser(
        description='Create reference tile for stain normalization'
    )
    parser.add_argument('--links', type=str, required=True,
                        help='Path to links.csv')
    parser.add_argument('--output', type=str, default='ref_tile_camelyon.npy',
                        help='Output path for reference tile')
    parser.add_argument('--level', type=int, default=0,
                        help='Pyramid level to sample from')
    parser.add_argument('--size', type=int, default=256,
                        help='Tile size')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples per slide')
    
    args = parser.parse_args()
    
    create_reference_tile(
        args.links,
        args.output,
        level=args.level,
        tile_size=args.size,
        num_samples=args.samples,
    )


if __name__ == '__main__':
    main()

