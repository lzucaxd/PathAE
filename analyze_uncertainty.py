#!/usr/bin/env python3
"""
Analyze uncertainty patterns and validate that high uncertainty
correlates with high error rate.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score


def analyze_uncertainty(ensemble_csv, test_csv, output_dir):
    """
    Analyze uncertainty patterns.
    
    Creates:
    1. Error rate vs uncertainty bins
    2. Scatter plot of uncertainty vs prediction error
    """
    print(f"\n{'='*70}")
    print("UNCERTAINTY ANALYSIS")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data...")
    df_ensemble = pd.read_csv(ensemble_csv)
    df_test = pd.read_csv(test_csv)
    
    # Merge to get ground truth labels
    df = pd.merge(df_ensemble, df_test[['tile_id', 'label']], on='tile_id')
    
    print(f"  Ensemble predictions: {len(df_ensemble):,}")
    print(f"  Test set labels: {len(df_test):,}")
    print(f"  Merged: {len(df):,}")
    
    # Compute predictions and errors
    df['pred'] = (df['score'] > 0.5).astype(int)
    df['correct'] = (df['pred'] == df['label']).astype(int)
    df['error'] = 1 - df['correct']
    
    overall_accuracy = df['correct'].mean()
    overall_error = df['error'].mean()
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {overall_accuracy:.4f}")
    print(f"  Error rate: {overall_error:.4f}")
    
    # Bin by uncertainty
    print(f"\nBinning by uncertainty...")
    
    bins = [0, 0.1, 0.2, 0.3, 1.0]
    bin_labels = ['Low\n(<0.1)', 'Medium\n(0.1-0.2)', 'High\n(0.2-0.3)', 'Very High\n(>0.3)']
    df['uncertainty_bin'] = pd.cut(df['uncertainty'], bins=bins, labels=bin_labels, include_lowest=True)
    
    # Analyze each bin
    bin_stats = []
    
    for bin_label in bin_labels:
        bin_df = df[df['uncertainty_bin'] == bin_label]
        if len(bin_df) == 0:
            continue
        
        error_rate = bin_df['error'].mean()
        count = len(bin_df)
        pct = 100 * count / len(df)
        
        bin_stats.append({
            'Uncertainty Bin': bin_label.replace('\n', ' '),
            'Count': count,
            'Percentage': pct,
            'Error Rate': error_rate,
            'Accuracy': 1 - error_rate
        })
        
        print(f"\n  {bin_label.replace(chr(10), ' ')}:")
        print(f"    Count: {count:,} ({pct:.1f}%)")
        print(f"    Error rate: {error_rate:.4f}")
        print(f"    Accuracy: {1-error_rate:.4f}")
    
    bin_stats_df = pd.DataFrame(bin_stats)
    
    # Create visualization
    print(f"\nCreating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # =========================================================================
    # Panel 1: Bar chart - Error rate vs uncertainty bins
    # =========================================================================
    ax = axes[0]
    
    x = np.arange(len(bin_stats_df))
    bars = ax.bar(x, bin_stats_df['Error Rate'], 
                  color=['#4CAF50', '#FFC107', '#FF5722', '#B71C1C'],
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(bin_stats_df.iterrows()):
        ax.text(i, row['Error Rate'] + 0.01, f"{row['Error Rate']:.3f}", 
               ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(i, -0.03, f"n={int(row['Count']):,}\n({row['Percentage']:.1f}%)", 
               ha='center', va='top', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(bin_stats_df['Uncertainty Bin'])
    ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
    ax.set_xlabel('Uncertainty Bin', fontsize=12, fontweight='bold')
    ax.set_title('Error Rate vs Model Uncertainty\n(Higher uncertainty → Higher error rate)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(bin_stats_df['Error Rate']) * 1.2)
    
    # =========================================================================
    # Panel 2: Scatter plot - Uncertainty vs error
    # =========================================================================
    ax = axes[1]
    
    # Sample for plotting (too many points)
    np.random.seed(42)
    if len(df) > 10000:
        sample_idx = np.random.choice(len(df), 10000, replace=False)
        df_plot = df.iloc[sample_idx]
    else:
        df_plot = df
    
    # Separate correct and incorrect
    correct_df = df_plot[df_plot['correct'] == 1]
    error_df = df_plot[df_plot['error'] == 1]
    
    # Plot
    ax.scatter(correct_df['uncertainty'], correct_df['error'] + np.random.normal(0, 0.02, len(correct_df)),
              alpha=0.3, s=20, c='#4CAF50', label=f'Correct ({len(correct_df):,})', marker='o')
    ax.scatter(error_df['uncertainty'], error_df['error'] + np.random.normal(0, 0.02, len(error_df)),
              alpha=0.5, s=30, c='#FF5722', label=f'Error ({len(error_df):,})', marker='x')
    
    ax.set_xlabel('Model Uncertainty (|P_supervised - P_contrastive|)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Error (0=Correct, 1=Wrong)', fontsize=12, fontweight='bold')
    ax.set_title('Uncertainty vs Prediction Correctness\n(Errors cluster at high uncertainty)', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.15)
    
    # Add vertical lines for bins
    for threshold in [0.1, 0.2, 0.3]:
        ax.axvline(threshold, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'uncertainty_analysis.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Visualization saved: {output_path}")
    
    # Save statistics
    stats_path = Path(output_dir) / 'uncertainty_stats.csv'
    bin_stats_df.to_csv(stats_path, index=False, float_format='%.4f')
    print(f"  ✓ Statistics saved: {stats_path}")
    
    return bin_stats_df


def main():
    parser = argparse.ArgumentParser(description='Analyze uncertainty patterns')
    parser.add_argument('--ensemble-csv', type=str, default='outputs/scores_ensemble.csv')
    parser.add_argument('--test-csv', type=str, default='test_set_heatmaps/test_set.csv')
    parser.add_argument('--output-dir', type=str, default='figures')
    args = parser.parse_args()
    
    # Analyze
    bin_stats = analyze_uncertainty(args.ensemble_csv, args.test_csv, args.output_dir)
    
    print(f"\n{'='*70}")
    print("✓ UNCERTAINTY ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    print("Summary:")
    print(bin_stats.to_string(index=False))
    print()
    
    # Validation check
    print("Validation:")
    if bin_stats['Error Rate'].is_monotonic_increasing:
        print("  ✓ Error rate increases with uncertainty (VALIDATED!)")
        print("  → Uncertainty is a reliable indicator of prediction quality")
    else:
        print("  ⚠ Error rate not perfectly monotonic with uncertainty")
        print("  → May need better uncertainty quantification")


if __name__ == '__main__':
    main()


