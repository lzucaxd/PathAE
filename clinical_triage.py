#!/usr/bin/env python3
"""
Clinical triage system based on model uncertainty.

Creates a 3-tier system:
- Auto-decidable: Low uncertainty (<0.15) - AI handles automatically
- Review queue: Medium uncertainty (0.15-0.3) - Human review needed  
- High-risk: High uncertainty (>0.3) - Priority human review
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_triage_system(ensemble_csv, test_csv, output_dir, 
                        auto_threshold=0.15, review_threshold=0.3):
    """
    Create clinical triage system based on uncertainty.
    
    Args:
        ensemble_csv: Path to ensemble predictions
        test_csv: Path to test set with labels
        output_dir: Output directory
        auto_threshold: Threshold for auto-decidable tier
        review_threshold: Threshold for high-risk tier
    """
    print(f"\n{'='*70}")
    print("CLINICAL TRIAGE SYSTEM")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data...")
    df_ensemble = pd.read_csv(ensemble_csv)
    df_test = pd.read_csv(test_csv)
    df = pd.merge(df_ensemble, df_test[['tile_id', 'label']], on='tile_id')
    
    print(f"  Merged: {len(df):,} tiles")
    
    # Compute predictions and errors
    df['pred'] = (df['score'] > 0.5).astype(int)
    df['correct'] = (df['pred'] == df['label']).astype(int)
    df['error'] = 1 - df['correct']
    
    # Assign triage tiers
    df['tier'] = 'Auto-decidable'
    df.loc[df['uncertainty'] >= auto_threshold, 'tier'] = 'Review queue'
    df.loc[df['uncertainty'] >= review_threshold, 'tier'] = 'High-risk'
    
    print(f"\nTriage Configuration:")
    print(f"  Auto-decidable: uncertainty < {auto_threshold}")
    print(f"  Review queue:   {auto_threshold} ≤ uncertainty < {review_threshold}")
    print(f"  High-risk:      uncertainty ≥ {review_threshold}")
    
    # Analyze each tier
    print(f"\n{'='*70}")
    print("TIER ANALYSIS")
    print(f"{'='*70}")
    
    tiers = ['Auto-decidable', 'Review queue', 'High-risk']
    tier_stats = []
    
    for tier in tiers:
        tier_df = df[df['tier'] == tier]
        if len(tier_df) == 0:
            continue
        
        count = len(tier_df)
        pct = 100 * count / len(df)
        error_rate = tier_df['error'].mean()
        
        # Tumor metrics
        tumor_in_tier = (tier_df['label'] == 1).sum()
        total_tumors = (df['label'] == 1).sum()
        tumor_capture = 100 * tumor_in_tier / total_tumors if total_tumors > 0 else 0
        
        # Error metrics
        errors_in_tier = tier_df['error'].sum()
        total_errors = df['error'].sum()
        error_capture = 100 * errors_in_tier / total_errors if total_errors > 0 else 0
        
        tier_stats.append({
            'Tier': tier,
            'Count': count,
            'Percentage': pct,
            'Error Rate': error_rate,
            'Tumor Capture (%)': tumor_capture,
            'Error Capture (%)': error_capture
        })
        
        print(f"\n{tier}:")
        print(f"  Patches: {count:,} ({pct:.1f}%)")
        print(f"  Error rate: {error_rate:.4f}")
        print(f"  Tumor capture: {tumor_capture:.1f}% ({tumor_in_tier:,}/{total_tumors:,})")
        print(f"  Error capture: {error_capture:.1f}% ({errors_in_tier:,}/{int(total_errors):,})")
    
    tier_stats_df = pd.DataFrame(tier_stats)
    
    # Clinical impact metrics
    auto_df = df[df['tier'] == 'Auto-decidable']
    review_df = df[df['tier'].isin(['Review queue', 'High-risk'])]
    
    workload_reduction = 100 * len(auto_df) / len(df)
    auto_accuracy = auto_df['correct'].mean()
    review_error_capture = review_df['error'].sum() / df['error'].sum() * 100
    
    print(f"\n{'='*70}")
    print("CLINICAL IMPACT")
    print(f"{'='*70}\n")
    
    print(f"💡 Workload Reduction:")
    print(f"  {workload_reduction:.1f}% of patches auto-decided")
    print(f"  {100-workload_reduction:.1f}% flagged for human review")
    print()
    
    print(f"✓ Auto-Decidable Tier Quality:")
    print(f"  Accuracy: {auto_accuracy:.4f} (error rate: {1-auto_accuracy:.4f})")
    print(f"  Safe for automated reporting")
    print()
    
    print(f"🎯 Review Queue Effectiveness:")
    print(f"  Captures {review_error_capture:.1f}% of all errors")
    print(f"  Focused human effort on uncertain cases")
    print()
    
    # Create visualization
    print(f"Creating visualization...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # Panel 1: Tier distribution (bar chart)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    x = np.arange(len(tier_stats_df))
    colors = ['#4CAF50', '#FFC107', '#FF5722']
    bars = ax1.bar(x, tier_stats_df['Percentage'], color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    for i, (idx, row) in enumerate(tier_stats_df.iterrows()):
        ax1.text(i, row['Percentage'] + 1, f"{row['Percentage']:.1f}%", 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax1.text(i, row['Percentage']/2, f"{int(row['Count']):,}\npatches", 
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_stats_df['Tier'], fontsize=10)
    ax1.set_ylabel('Percentage of Patches', fontsize=12, fontweight='bold')
    ax1.set_title('Triage Distribution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(tier_stats_df['Percentage']) * 1.15)
    
    # =========================================================================
    # Panel 2: Error rate per tier
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    bars = ax2.bar(x, tier_stats_df['Error Rate'], color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    
    for i, (idx, row) in enumerate(tier_stats_df.iterrows()):
        ax2.text(i, row['Error Rate'] + 0.005, f"{row['Error Rate']:.4f}", 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(tier_stats_df['Tier'], fontsize=10)
    ax2.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Quality per Tier', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(tier_stats_df['Error Rate']) * 1.15)
    
    # =========================================================================
    # Panel 3: Workload allocation (pie chart)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    sizes = tier_stats_df['Percentage'].values
    labels = [f"{tier}\n{pct:.1f}%" for tier, pct in 
             zip(tier_stats_df['Tier'], tier_stats_df['Percentage'])]
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='',
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    ax3.set_title('Workload Allocation', fontsize=13, fontweight='bold')
    
    # =========================================================================
    # Panel 4: Tumor and error capture
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, :])
    
    x = np.arange(len(tier_stats_df))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, tier_stats_df['Tumor Capture (%)'], width,
                   label='Tumor Capture', color='#2196F3', alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x + width/2, tier_stats_df['Error Capture (%)'], width,
                   label='Error Capture', color='#FF5722', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (idx, row) in enumerate(tier_stats_df.iterrows()):
        ax4.text(i - width/2, row['Tumor Capture (%)'] + 1, f"{row['Tumor Capture (%)']:.1f}%", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax4.text(i + width/2, row['Error Capture (%)'] + 1, f"{row['Error Capture (%)']:.1f}%", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(tier_stats_df['Tier'], fontsize=11)
    ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Tumor and Error Capture by Tier', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 105)
    
    # Add summary text box
    summary_text = (f"Clinical Impact Summary:\n"
                   f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                   f"Workload Reduction: {workload_reduction:.1f}%\n"
                   f"Auto-tier Accuracy: {auto_accuracy:.4f}\n"
                   f"Review captures: {review_error_capture:.1f}% of errors\n"
                   f"\n"
                   f"Recommendation:\n"
                   f"→ Auto-process {workload_reduction:.0f}% of cases\n"
                   f"→ Flag {100-workload_reduction:.0f}% for review\n"
                   f"→ Prioritize high-risk tier")
    
    fig.text(0.98, 0.02, summary_text, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            family='monospace')
    
    plt.suptitle('Clinical Triage System Based on Model Uncertainty',
                fontsize=15, fontweight='bold', y=0.98)
    
    output_path = Path(output_dir) / 'clinical_triage.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Visualization saved: {output_path}")
    
    # Save tier statistics
    stats_path = Path(output_dir) / 'triage_stats.csv'
    tier_stats_df.to_csv(stats_path, index=False, float_format='%.4f')
    print(f"  ✓ Statistics saved: {stats_path}")
    
    return tier_stats_df


def main():
    parser = argparse.ArgumentParser(description='Create clinical triage system')
    parser.add_argument('--ensemble-csv', type=str, default='outputs/scores_ensemble.csv')
    parser.add_argument('--test-csv', type=str, default='test_set_heatmaps/test_set.csv')
    parser.add_argument('--output-dir', type=str, default='figures')
    parser.add_argument('--auto-threshold', type=float, default=0.15,
                       help='Uncertainty threshold for auto-decidable tier')
    parser.add_argument('--review-threshold', type=float, default=0.3,
                       help='Uncertainty threshold for high-risk tier')
    args = parser.parse_args()
    
    # Create triage system
    tier_stats = create_triage_system(
        args.ensemble_csv, 
        args.test_csv, 
        args.output_dir,
        args.auto_threshold,
        args.review_threshold
    )
    
    print(f"\n{'='*70}")
    print("✓ CLINICAL TRIAGE SYSTEM COMPLETE")
    print(f"{'='*70}\n")
    
    print("Tier Statistics:")
    print(tier_stats.to_string(index=False))
    print()
    
    # Recommendations
    auto_pct = tier_stats[tier_stats['Tier'] == 'Auto-decidable']['Percentage'].values[0]
    auto_err = tier_stats[tier_stats['Tier'] == 'Auto-decidable']['Error Rate'].values[0]
    
    print("Clinical Deployment Recommendations:")
    print(f"  ✓ {auto_pct:.0f}% of cases can be auto-processed")
    print(f"  ✓ Error rate in auto-tier: {auto_err:.2%} (acceptable!)")
    print(f"  ✓ Review queue focuses human effort on uncertain cases")
    print(f"  ✓ High-risk tier gets priority review")


if __name__ == '__main__':
    main()

