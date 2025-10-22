#!/usr/bin/env python3
"""
Create 3-way heatmap comparisons (Supervised vs Contrastive vs Ensemble).

For each slide, creates a 1×3 comparison showing all three models' heatmaps.
Also creates a summary figure with IoU comparison across all slides.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def create_slide_comparison(wsi_id, supervised_dir, contrastive_dir, ensemble_dir, 
                           sup_metrics, con_metrics, ens_metrics, output_path):
    """
    Create 1×3 comparison for a single slide.
    
    Shows supervised, contrastive, and ensemble heatmaps side-by-side.
    """
    # Load heatmap images
    sup_path = Path(supervised_dir) / f'{wsi_id}_heatmap.png'
    con_path = Path(contrastive_dir) / f'{wsi_id}_heatmap.png'
    ens_path = Path(ensemble_dir) / f'{wsi_id}_heatmap.png'
    
    if not all([sup_path.exists(), con_path.exists(), ens_path.exists()]):
        print(f"  ⚠ Missing heatmaps for {wsi_id}, skipping...")
        return None
    
    sup_img = Image.open(sup_path)
    con_img = Image.open(con_path)
    ens_img = Image.open(ens_path)
    
    # Get metrics
    sup_iou = sup_metrics[sup_metrics['wsi_id'] == wsi_id]['iou'].values[0]
    con_iou = con_metrics[con_metrics['wsi_id'] == wsi_id]['iou'].values[0]
    ens_iou = ens_metrics[ens_metrics['wsi_id'] == wsi_id]['iou'].values[0]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Supervised
    axes[0].imshow(sup_img)
    axes[0].set_title(f'Supervised\nIoU: {sup_iou:.3f}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Contrastive
    axes[1].imshow(con_img)
    axes[1].set_title(f'Contrastive\nIoU: {con_iou:.3f}', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Ensemble
    axes[2].imshow(ens_img)
    axes[2].set_title(f'Ensemble\nIoU: {ens_iou:.3f}', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Overall title
    best_model = 'Supervised' if sup_iou >= max(con_iou, ens_iou) else \
                 'Ensemble' if ens_iou >= con_iou else 'Contrastive'
    
    plt.suptitle(f'{wsi_id} - Model Comparison (Best: {best_model})', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return {
        'wsi_id': wsi_id,
        'supervised_iou': sup_iou,
        'contrastive_iou': con_iou,
        'ensemble_iou': ens_iou,
        'best_model': best_model
    }


def create_summary_figure(comparison_stats, output_path):
    """
    Create summary bar chart comparing IoU across all slides and models.
    """
    df = pd.DataFrame(comparison_stats)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax.bar(x - width, df['supervised_iou'], width, 
                  label='Supervised', color='#2196F3', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, df['contrastive_iou'], width, 
                  label='Contrastive', color='#FF9800', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, df['ensemble_iou'], width, 
                  label='Ensemble', color='#4CAF50', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, row in df.iterrows():
        ax.text(i - width, row['supervised_iou'] + 0.02, f"{row['supervised_iou']:.2f}", 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i, row['contrastive_iou'] + 0.02, f"{row['contrastive_iou']:.2f}", 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i + width, row['ensemble_iou'] + 0.02, f"{row['ensemble_iou']:.2f}", 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['wsi_id'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('IoU (Jaccard Index)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Slide ID', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison Across All Test Slides', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Add average lines
    avg_sup = df['supervised_iou'].mean()
    avg_con = df['contrastive_iou'].mean()
    avg_ens = df['ensemble_iou'].mean()
    
    ax.axhline(avg_sup, color='#2196F3', linestyle='--', alpha=0.5, linewidth=2,
              label=f'Avg Supervised: {avg_sup:.3f}')
    ax.axhline(avg_con, color='#FF9800', linestyle='--', alpha=0.5, linewidth=2,
              label=f'Avg Contrastive: {avg_con:.3f}')
    ax.axhline(avg_ens, color='#4CAF50', linestyle='--', alpha=0.5, linewidth=2,
              label=f'Avg Ensemble: {avg_ens:.3f}')
    
    # Update legend to include averages
    ax.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Summary figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervised-dir', type=str, default='outputs/supervised_heatmaps_v2')
    parser.add_argument('--contrastive-dir', type=str, default='outputs/contrastive_heatmaps_v2')
    parser.add_argument('--ensemble-dir', type=str, default='outputs/ensemble_heatmaps')
    parser.add_argument('--output-dir', type=str, default='figures')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("3-WAY HEATMAP COMPARISON")
    print(f"{'='*70}\n")
    
    # Load metrics
    print("Loading metrics...")
    sup_metrics = pd.read_csv(Path(args.supervised_dir) / 'heatmap_metrics.csv')
    con_metrics = pd.read_csv(Path(args.contrastive_dir) / 'heatmap_metrics.csv')
    ens_metrics = pd.read_csv(Path(args.ensemble_dir) / 'heatmap_metrics.csv')
    
    print(f"  Supervised: {len(sup_metrics)} slides")
    print(f"  Contrastive: {len(con_metrics)} slides")
    print(f"  Ensemble: {len(ens_metrics)} slides")
    
    # Get common slides
    common_slides = set(sup_metrics['wsi_id']) & set(con_metrics['wsi_id']) & set(ens_metrics['wsi_id'])
    print(f"  Common: {len(common_slides)} slides")
    
    # Create comparisons
    print(f"\n{'='*70}")
    print("GENERATING COMPARISONS")
    print(f"{'='*70}\n")
    
    comparison_stats = []
    
    for wsi_id in sorted(common_slides):
        print(f"Processing {wsi_id}...")
        output_path = output_dir / f'heatmap_comparison_{wsi_id}.png'
        
        stats = create_slide_comparison(
            wsi_id,
            args.supervised_dir,
            args.contrastive_dir,
            args.ensemble_dir,
            sup_metrics,
            con_metrics,
            ens_metrics,
            output_path
        )
        
        if stats:
            comparison_stats.append(stats)
    
    # Create summary figure
    print(f"\nCreating summary figure...")
    summary_path = output_dir / 'heatmap_comparison_summary.png'
    create_summary_figure(comparison_stats, summary_path)
    
    # Save comparison stats
    comparison_df = pd.DataFrame(comparison_stats)
    stats_path = output_dir / 'model_comparison_stats.csv'
    comparison_df.to_csv(stats_path, index=False, float_format='%.4f')
    print(f"  ✓ Comparison stats saved: {stats_path}")
    
    print(f"\n{'='*70}")
    print("✓ 3-WAY COMPARISON COMPLETE")
    print(f"{'='*70}\n")
    
    print("Per-Slide Best Model:")
    print(comparison_df[['wsi_id', 'best_model', 'supervised_iou', 
                        'contrastive_iou', 'ensemble_iou']].to_string(index=False))
    
    print(f"\nOverall Averages:")
    print(f"  Supervised:  {comparison_df['supervised_iou'].mean():.4f}")
    print(f"  Contrastive: {comparison_df['contrastive_iou'].mean():.4f}")
    print(f"  Ensemble:    {comparison_df['ensemble_iou'].mean():.4f}")
    
    # Count best models
    best_counts = comparison_df['best_model'].value_counts()
    print(f"\nBest Model Count:")
    for model, count in best_counts.items():
        print(f"  {model}: {count}/{len(comparison_df)} slides")


if __name__ == '__main__':
    main()


