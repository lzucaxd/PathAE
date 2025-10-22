#!/usr/bin/env python3
"""
Create final summary figure with all key results.

Creates a comprehensive visualization showing:
1. PCam test performance (all models)
2. WSI heatmap performance (all models)  
3. Uncertainty validation
4. Clinical triage breakdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_summary_figure():
    """Create comprehensive summary figure."""
    
    # Load data
    pcam_results = pd.read_csv('outputs/ensemble_pcam/pcam_test_comparison.csv')
    triage_stats = pd.read_csv('figures/triage_stats.csv')
    
    # Heatmap results (manually entered from outputs)
    wsi_results = pd.DataFrame([
        {'Model': 'Supervised', 'Mean IoU': 0.5174, 'Std': 0.21},
        {'Model': 'Contrastive', 'Mean IoU': 0.2542, 'Std': 0.29},
        {'Model': 'Ensemble', 'Mean IoU': 0.3765, 'Std': 0.31}
    ])
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    colors = {'Supervised': '#2196F3', 'Contrastive': '#FF9800', 'Ensemble': '#4CAF50'}
    
    # =========================================================================
    # Row 1: PCam Test Performance
    # =========================================================================
    
    # Panel 1: PR-AUC and ROC-AUC
    ax1 = fig.add_subplot(gs[0, 0])
    
    metrics = ['pr_auc', 'roc_auc']
    metric_names = ['PR-AUC', 'ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (idx, row) in enumerate(pcam_results.iterrows()):
        model = row['Model']
        values = [row[m] for m in metrics]
        offset = (i - 1) * width
        ax1.bar(x + offset, values, width, label=model, 
               color=colors[model], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for j, val in enumerate(values):
            ax1.text(j + offset, val + 0.01, f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('PCam Test: Classification Metrics', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.9, 1.0)
    
    # Panel 2: Accuracy and F1
    ax2 = fig.add_subplot(gs[0, 1])
    
    metrics = ['accuracy', 'f1']
    metric_names = ['Accuracy', 'F1-Score']
    x = np.arange(len(metrics))
    
    for i, (idx, row) in enumerate(pcam_results.iterrows()):
        model = row['Model']
        values = [row[m] for m in metrics]
        offset = (i - 1) * width
        ax2.bar(x + offset, values, width, label=model, 
               color=colors[model], alpha=0.8, edgecolor='black')
        
        for j, val in enumerate(values):
            ax2.text(j + offset, val + 0.01, f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names)
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('PCam Test: Performance Metrics', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.7, 1.0)
    
    # Panel 3: Precision vs Recall
    ax3 = fig.add_subplot(gs[0, 2])
    
    for idx, row in pcam_results.iterrows():
        model = row['Model']
        ax3.scatter(row['recall'], row['precision'], s=300, 
                   color=colors[model], alpha=0.7, edgecolor='black', linewidth=2,
                   label=model, marker='o')
        ax3.text(row['recall'], row['precision'], model, 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax3.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax3.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.6, 1.0)
    ax3.set_ylim(0.8, 1.0)
    
    # Add diagonal (F1 contours)
    for f1_val in [0.8, 0.85, 0.9]:
        recall_range = np.linspace(0.6, 1.0, 100)
        precision_vals = f1_val * recall_range / (2 * recall_range - f1_val)
        precision_vals = np.clip(precision_vals, 0.8, 1.0)
        ax3.plot(recall_range, precision_vals, '--', alpha=0.3, color='gray', linewidth=1)
        ax3.text(0.95, f1_val * 0.95 / (2 * 0.95 - f1_val), f'F1={f1_val}', 
                fontsize=8, alpha=0.5, rotation=45)
    
    # =========================================================================
    # Row 2: WSI Heatmap Performance
    # =========================================================================
    
    # Panel 4: WSI Mean IoU
    ax4 = fig.add_subplot(gs[1, 0])
    
    x = np.arange(len(wsi_results))
    bars = ax4.bar(x, wsi_results['Mean IoU'], 
                  color=[colors[m] for m in wsi_results['Model']],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax4.errorbar(x, wsi_results['Mean IoU'], yerr=wsi_results['Std'], 
                fmt='none', ecolor='black', capsize=5, linewidth=2)
    
    # Add value labels
    for i, (idx, row) in enumerate(wsi_results.iterrows()):
        ax4.text(i, row['Mean IoU'] + row['Std'] + 0.03, 
                f"{row['Mean IoU']:.3f}±{row['Std']:.2f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(wsi_results['Model'])
    ax4.set_ylabel('Mean IoU (Jaccard Index)', fontsize=11, fontweight='bold')
    ax4.set_title('WSI Heatmap Performance (8 slides)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 0.9)
    
    # Panel 5: Uncertainty validation
    ax5 = fig.add_subplot(gs[1, 1])
    
    unc_stats = pd.read_csv('figures/uncertainty_stats.csv')
    
    x = np.arange(len(unc_stats))
    colors_unc = ['#4CAF50', '#FFC107', '#FF9800', '#FF5722']
    bars = ax5.bar(x, unc_stats['Error Rate'], color=colors_unc, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, (idx, row) in enumerate(unc_stats.iterrows()):
        ax5.text(i, row['Error Rate'] + 0.005, f"{row['Error Rate']:.4f}", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax5.text(i, -0.01, f"{int(row['Count']):,}\n({row['Percentage']:.0f}%)", 
                ha='center', va='top', fontsize=8)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(unc_stats['Uncertainty Bin'])
    ax5.set_ylabel('Error Rate', fontsize=11, fontweight='bold')
    ax5.set_title('Uncertainty Validation\n(Higher uncertainty → Higher error)', 
                 fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 0.2)
    
    # Panel 6: Clinical triage
    ax6 = fig.add_subplot(gs[1, 2])
    
    colors_tier = ['#4CAF50', '#FFC107', '#FF5722']
    sizes = triage_stats['Percentage'].values
    labels = [f"{tier}\n{pct:.1f}%\n(err: {err:.2%})" 
             for tier, pct, err in zip(triage_stats['Tier'], 
                                       triage_stats['Percentage'],
                                       triage_stats['Error Rate'])]
    
    ax6.pie(sizes, labels=labels, colors=colors_tier, autopct='',
           startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
    ax6.set_title('Clinical Triage Distribution', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Row 3: Summary Statistics Table
    # =========================================================================
    
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create summary table
    table_data = [
        ['', '', 'Supervised', 'Contrastive', 'Ensemble', 'Best'],
        ['', '', '', '', '', ''],
        ['PCam Test', 'PR-AUC', '0.961', '0.948', '0.960', '✓ Supervised'],
        ['', 'Accuracy', '82.8%', '86.9%', '87.8%', '✓ Ensemble'],
        ['', 'F1-Score', '0.797', '0.870', '0.868', '✓ Contrastive'],
        ['', '', '', '', '', ''],
        ['WSI Heatmaps', 'Mean IoU', '0.517', '0.254', '0.377', '✓ Supervised'],
        ['(8 slides)', 'Best on N slides', '7/8', '0/8', '1/8', '✓ Supervised'],
        ['', '', '', '', '', ''],
        ['Feature Space', 'Separation Ratio', '1.52', '1.70', '-', '✓ Contrastive'],
        ['', 'Silhouette (UMAP)', '0.372', '0.407', '-', '✓ Contrastive'],
        ['', '', '', '', '', ''],
        ['Clinical', 'Auto-process %', '-', '-', '74.5%', '✓ Ensemble'],
        ['Deployment', 'Auto-tier Accuracy', '-', '-', '97.4%', '✓ Ensemble'],
        ['', 'Error Capture', '-', '-', '63.6%', '✓ Ensemble'],
    ]
    
    table = ax7.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style section headers
    for row in [2, 6, 9, 12]:
        for col in range(2):
            table[(row, col)].set_facecolor('#E3F2FD')
            table[(row, col)].set_text_props(weight='bold')
    
    # Highlight best values
    for row in [2, 3, 4, 6, 7, 9, 10, 12, 13, 14]:
        table[(row, 5)].set_facecolor('#C8E6C9')
        table[(row, 5)].set_text_props(weight='bold')
    
    ax7.set_title('Complete Performance Summary: Supervised vs Contrastive vs Ensemble', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add main title
    plt.suptitle('PathAE: Tumor Detection in Histopathology - Final Results',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('figures/final_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Summary figure saved: figures/final_summary.png")


def main():
    """Main function."""
    print(f"\n{'='*70}")
    print("CREATING FINAL SUMMARY FIGURE")
    print(f"{'='*70}\n")
    
    create_summary_figure()
    
    print(f"\n{'='*70}")
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print(f"{'='*70}\n")
    
    print("Generated Files:")
    print("\n📊 Main Summary:")
    print("  • figures/final_summary.png - Comprehensive results table")
    print("  • FINAL_SUMMARY.md - Detailed written summary")
    
    print("\n🔬 Model Comparisons:")
    print("  • figures/heatmap_comparison_summary.png - 3-way IoU comparison")
    print("  • figures/distance_distributions_test.png - Feature space analysis")
    
    print("\n🎯 Interpretability:")
    print("  • figures/gradcam_comparison.png - Attention visualization")
    print("  • figures/uncertainty_analysis.png - Error vs uncertainty")
    
    print("\n🏥 Clinical Deployment:")
    print("  • figures/clinical_triage.png - 3-tier triage system")
    
    print("\n📁 Heatmaps (8 slides × 3 models):")
    print("  • outputs/supervised_heatmaps_v2/")
    print("  • outputs/contrastive_heatmaps_v2/")
    print("  • outputs/ensemble_heatmaps/")
    
    print("\n✅ Project Complete!")


if __name__ == '__main__':
    main()


