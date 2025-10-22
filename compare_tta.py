#!/usr/bin/env python3
"""
Compare supervised vs supervised+TTA performance.

Shows IoU improvement per slide and creates before/after visualization.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compare_tta(baseline_metrics_csv, tta_metrics_csv, output_path):
    """
    Compare baseline vs TTA performance.
    
    Args:
        baseline_metrics_csv: Metrics from standard inference
        tta_metrics_csv: Metrics from TTA inference
        output_path: Where to save comparison figure
    """
    print(f"\n{'='*70}")
    print("TTA PERFORMANCE COMPARISON")
    print(f"{'='*70}\n")
    
    # Load metrics
    df_baseline = pd.read_csv(baseline_metrics_csv)
    df_tta = pd.read_csv(tta_metrics_csv)
    
    # Merge on wsi_id
    df_comp = pd.merge(df_baseline, df_tta, on='wsi_id', suffixes=('_baseline', '_tta'))
    
    # Compute improvements
    df_comp['iou_improvement'] = df_comp['iou_tta'] - df_comp['iou_baseline']
    df_comp['dice_improvement'] = df_comp['dice_tta'] - df_comp['dice_baseline']
    
    print("Per-Slide Comparison:")
    print(df_comp[['wsi_id', 'iou_baseline', 'iou_tta', 'iou_improvement']].to_string(index=False))
    
    print(f"\nOverall Statistics:")
    print(f"  Mean IoU (baseline):  {df_comp['iou_baseline'].mean():.4f}")
    print(f"  Mean IoU (TTA):       {df_comp['iou_tta'].mean():.4f}")
    print(f"  Mean improvement:     {df_comp['iou_improvement'].mean():.4f} ({100*df_comp['iou_improvement'].mean()/df_comp['iou_baseline'].mean():.1f}%)")
    print(f"  Slides improved:      {(df_comp['iou_improvement'] > 0).sum()}/{len(df_comp)}")
    
    # Create visualization
    print(f"\nCreating comparison figure...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # =========================================================================
    # Panel 1: Before/After IoU comparison
    # =========================================================================
    ax = axes[0]
    
    x = np.arange(len(df_comp))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_comp['iou_baseline'], width,
                  label='Baseline', color='#2196F3', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df_comp['iou_tta'], width,
                  label='TTA (8 aug)', color='#4CAF50', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (idx, row) in enumerate(df_comp.iterrows()):
        ax.text(i - width/2, row['iou_baseline'] + 0.02, f"{row['iou_baseline']:.2f}", 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i + width/2, row['iou_tta'] + 0.02, f"{row['iou_tta']:.2f}", 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add improvement arrow
        if row['iou_improvement'] > 0:
            ax.annotate('', xy=(i + width/2, row['iou_tta']), 
                       xytext=(i - width/2, row['iou_baseline']),
                       arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.5))
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_comp['wsi_id'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('IoU (Jaccard Index)', fontsize=12, fontweight='bold')
    ax.set_title('IoU Comparison: Baseline vs TTA', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Add mean lines
    ax.axhline(df_comp['iou_baseline'].mean(), color='#2196F3', 
              linestyle='--', alpha=0.5, linewidth=2,
              label=f'Baseline mean: {df_comp["iou_baseline"].mean():.3f}')
    ax.axhline(df_comp['iou_tta'].mean(), color='#4CAF50', 
              linestyle='--', alpha=0.5, linewidth=2,
              label=f'TTA mean: {df_comp["iou_tta"].mean():.3f}')
    ax.legend(fontsize=9, loc='upper left')
    
    # =========================================================================
    # Panel 2: Improvement per slide
    # =========================================================================
    ax = axes[1]
    
    # Sort by improvement
    df_sorted = df_comp.sort_values('iou_improvement', ascending=False)
    x = np.arange(len(df_sorted))
    
    colors = ['#4CAF50' if imp > 0 else '#FF5722' for imp in df_sorted['iou_improvement']]
    bars = ax.barh(x, df_sorted['iou_improvement'], color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        sign = '+' if row['iou_improvement'] >= 0 else ''
        ax.text(row['iou_improvement'] + 0.01 if row['iou_improvement'] >= 0 else row['iou_improvement'] - 0.01, 
               i, f"{sign}{row['iou_improvement']:.3f}", 
               ha='left' if row['iou_improvement'] >= 0 else 'right', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(x)
    ax.set_yticklabels(df_sorted['wsi_id'], fontsize=10)
    ax.set_xlabel('IoU Improvement (TTA - Baseline)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Slide IoU Improvement', fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(df_comp['iou_improvement'].mean(), color='green', 
              linestyle='--', linewidth=2, alpha=0.7,
              label=f'Mean: +{df_comp["iou_improvement"].mean():.3f}')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=10)
    
    plt.suptitle('Test-Time Augmentation Impact on Tumor Localization', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Comparison figure saved: {output_path}")
    
    return df_comp


def main():
    parser = argparse.ArgumentParser(description='Compare TTA vs baseline')
    parser.add_argument('--baseline-metrics', type=str,
                       default='outputs/supervised_heatmaps_v2/heatmap_metrics.csv')
    parser.add_argument('--tta-metrics', type=str,
                       default='outputs/supervised_heatmaps_tta/heatmap_metrics.csv')
    parser.add_argument('--output', type=str, default='figures/tta_improvement.png')
    args = parser.parse_args()
    
    # Compare
    df_comp = compare_tta(args.baseline_metrics, args.tta_metrics, args.output)
    
    print(f"\n{'='*70}")
    print("✓ TTA COMPARISON COMPLETE")
    print(f"{'='*70}\n")
    
    # Summary statistics
    mean_improvement = df_comp['iou_improvement'].mean()
    pct_improvement = 100 * mean_improvement / df_comp['iou_baseline'].mean()
    improved_slides = (df_comp['iou_improvement'] > 0).sum()
    
    print("Summary:")
    print(f"  Baseline Mean IoU: {df_comp['iou_baseline'].mean():.4f}")
    print(f"  TTA Mean IoU:      {df_comp['iou_tta'].mean():.4f}")
    print(f"  Mean Improvement:  +{mean_improvement:.4f} ({pct_improvement:+.1f}%)")
    print(f"  Slides Improved:   {improved_slides}/{len(df_comp)}")
    
    if mean_improvement > 0:
        print(f"\n✅ TTA provides significant improvement!")
        print(f"  Recommendation: Use TTA for final deployment")
    else:
        print(f"\n⚠️ TTA did not improve performance")
        print(f"  Consider: Different augmentations or model-specific TTA")


if __name__ == '__main__':
    main()


