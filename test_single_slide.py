#!/usr/bin/env python3
"""
Quick test on a single tumor-heavy slide for demo.

Automatically selects slide with most tumor tiles and generates heatmap.
"""

import argparse
import pandas as pd
from pathlib import Path
import subprocess


def select_best_slide(test_csv):
    """Select slide with most tumor tiles for demo."""
    
    df = pd.read_csv(test_csv)
    
    # Count tumor tiles per slide
    tumor_counts = []
    for wsi_id in df['wsi_id'].unique():
        slide_df = df[df['wsi_id'] == wsi_id]
        n_tumor = (slide_df['label'] == 1).sum()
        n_total = len(slide_df)
        tumor_pct = 100 * n_tumor / n_total if n_total > 0 else 0
        
        tumor_counts.append({
            'wsi_id': wsi_id,
            'n_tumor': n_tumor,
            'n_total': n_total,
            'tumor_pct': tumor_pct
        })
    
    # Sort by tumor count
    tumor_counts = sorted(tumor_counts, key=lambda x: x['n_tumor'], reverse=True)
    
    print("="*70)
    print("SLIDES RANKED BY TUMOR CONTENT")
    print("="*70)
    print()
    
    for i, info in enumerate(tumor_counts[:5], 1):
        print(f"{i}. {info['wsi_id']:15}: {info['n_tumor']:4,} tumor tiles ({info['tumor_pct']:5.1f}%), {info['n_total']:6,} total")
    
    print()
    
    # Select best (most tumor)
    best = tumor_counts[0]
    print(f"✓ Selected: {best['wsi_id']} ({best['n_tumor']:,} tumor tiles, {best['tumor_pct']:.1f}%)")
    
    return best['wsi_id']


def main():
    parser = argparse.ArgumentParser(description='Test on single tumor-heavy slide')
    parser.add_argument('--test-csv', type=str, default='test_set_heatmaps/test_set.csv')
    parser.add_argument('--scores-csv', type=str, default='reconstruction_scores.csv')
    parser.add_argument('--wsi-id', type=str, help='Specific slide (or auto-select if not provided)')
    parser.add_argument('--output-dir', type=str, default='demo_heatmap')
    
    args = parser.parse_args()
    
    # Select slide
    if args.wsi_id:
        wsi_id = args.wsi_id
        print(f"Using specified slide: {wsi_id}")
    else:
        wsi_id = select_best_slide(args.test_csv)
    
    print()
    print("="*70)
    print(f"GENERATING DEMO HEATMAP FOR: {wsi_id}")
    print("="*70)
    print()
    
    # Check if scores exist
    if not Path(args.scores_csv).exists():
        print(f"Error: {args.scores_csv} not found!")
        print()
        print("You need to run inference first:")
        print(f"  python run_inference_vae.py --model vae_best.pth --test-csv {args.test_csv}")
        return
    
    # Generate heatmap
    print("Generating heatmap with proper stitching...")
    
    cmd = [
        'python', 'stitch_heatmap.py',
        '--test-csv', args.test_csv,
        '--scores-csv', args.scores_csv,
        '--wsi-id', wsi_id,
        '--output-dir', args.output_dir,
        '--smooth-sigma', '2.0',
        '--canvas-level', '4',
        '--alpha', '0.5',
    ]
    
    # Add threshold if exists
    if Path('threshold.npy').exists():
        import numpy as np
        threshold_data = np.load('threshold.npy', allow_pickle=True).item()
        threshold = threshold_data['threshold']
        cmd.extend(['--threshold', str(threshold)])
        print(f"Using threshold: {threshold:.6f}")
    
    # Run
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print()
        print("="*70)
        print("✓ DEMO HEATMAP READY!")
        print("="*70)
        print(f"Location: {args.output_dir}/{wsi_id}_heatmap.png")
        print()
        print("This slide has the most tumor content - perfect for presentation!")
    else:
        print("Error generating heatmap")


if __name__ == '__main__':
    main()

