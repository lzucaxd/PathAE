"""
Comprehensive Comparison of All Improvements

Compares: Baseline, TTA, Morphological, TTA+Morphological, Ensemble, Optimized Ensemble

Author: ML Infra Engineer
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE IMPROVEMENT COMPARISON")
    print("="*70)
    
    # Load all results
    results = []
    
    # 1. Baseline (Supervised)
    sup_df = pd.read_csv('outputs/supervised_scores.csv')
    print(f"\n✓ Loaded baseline supervised predictions: {len(sup_df):,} tiles")
    
    # 2. TTA
    tta_df = pd.read_csv('outputs/scores_supervised_tta.csv')
    print(f"✓ Loaded TTA predictions: {len(tta_df):,} tiles")
    
    # 3. Morphological results
    morph_results = pd.read_csv('outputs/morphological_filtered_gentle/morphological_results.csv')
    print(f"✓ Loaded morphological results: {len(morph_results)} slides")
    
    # 4. TTA + Morphological results
    tta_morph_results = pd.read_csv('outputs/tta_morphological_filtered/morphological_results.csv')
    print(f"✓ Loaded TTA+Morphological results: {len(tta_morph_results)} slides")
    
    # 5. Ensemble results
    try:
        ensemble_results = pd.read_csv('outputs/ensemble_optimization/weight_search_results.csv')
        best_ensemble_iou = ensemble_results['mean_iou'].max()
        equal_ensemble_iou = ensemble_results.loc[ensemble_results['w_supervised'].sub(0.5).abs().idxmin(), 'mean_iou']
        print(f"✓ Loaded ensemble results")
    except:
        best_ensemble_iou = None
        equal_ensemble_iou = None
        print("! Ensemble results not found")
    
    # Create comprehensive comparison table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    comparison_data = []
    
    # Baseline
    baseline_iou = morph_results['original_iou'].mean()
    comparison_data.append({
        'Method': 'Baseline (Supervised)',
        'Mean IoU': baseline_iou,
        'Std IoU': morph_results['original_iou'].std(),
        'vs Baseline': 0.0,
        'Description': 'ResNet18 from scratch'
    })
    
    # Morphological
    morph_iou = morph_results['filtered_iou'].mean()
    comparison_data.append({
        'Method': 'Baseline + Morphological',
        'Mean IoU': morph_iou,
        'Std IoU': morph_results['filtered_iou'].std(),
        'vs Baseline': morph_iou - baseline_iou,
        'Description': 'Remove isolated patches (min_size=2)'
    })
    
    # TTA
    tta_iou = tta_morph_results['original_iou'].mean()
    comparison_data.append({
        'Method': 'TTA (8 augmentations)',
        'Mean IoU': tta_iou,
        'Std IoU': tta_morph_results['original_iou'].std(),
        'vs Baseline': tta_iou - baseline_iou,
        'Description': 'Test-time augmentation'
    })
    
    # TTA + Morphological (BEST)
    tta_morph_iou = tta_morph_results['filtered_iou'].mean()
    comparison_data.append({
        'Method': 'TTA + Morphological ⭐',
        'Mean IoU': tta_morph_iou,
        'Std IoU': tta_morph_results['filtered_iou'].std(),
        'vs Baseline': tta_morph_iou - baseline_iou,
        'Description': 'Best single-model approach'
    })
    
    # Ensemble (if available)
    if equal_ensemble_iou is not None:
        comparison_data.append({
            'Method': 'Ensemble (Equal 0.5/0.5)',
            'Mean IoU': equal_ensemble_iou,
            'Std IoU': np.nan,
            'vs Baseline': equal_ensemble_iou - baseline_iou,
            'Description': 'Sup + Contrastive (equal weight)'
        })
    
    if best_ensemble_iou is not None:
        comparison_data.append({
            'Method': 'Ensemble (Optimized 0.9/0.1)',
            'Mean IoU': best_ensemble_iou,
            'Std IoU': np.nan,
            'vs Baseline': best_ensemble_iou - baseline_iou,
            'Description': 'Optimized ensemble weights'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print table
    print("\n" + "-"*110)
    print(f"{'Method':<35} {'Mean IoU':>10} {'Std IoU':>10} {'Improvement':>12} {'Description':<35}")
    print("-"*110)
    for _, row in comparison_df.iterrows():
        improvement_str = f"{row['vs Baseline']:+.4f}" if row['vs Baseline'] != 0 else "baseline"
        std_str = f"{row['Std IoU']:.4f}" if not pd.isna(row['Std IoU']) else "N/A"
        print(f"{row['Method']:<35} {row['Mean IoU']:>10.4f} {std_str:>10} {improvement_str:>12} {row['Description']:<35}")
    print("-"*110)
    
    # Save to CSV
    comparison_df.to_csv('outputs/comprehensive_comparison.csv', index=False)
    print(f"\n✓ Saved: outputs/comprehensive_comparison.csv")
    
    # Create visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart comparison
    methods = comparison_df['Method'].values
    ious = comparison_df['Mean IoU'].values
    improvements = comparison_df['vs Baseline'].values
    
    # Color code: baseline=gray, improvements=gradient from blue to green
    colors = ['#95a5a6']  # Baseline
    for imp in improvements[1:]:
        if imp > 0.08:
            colors.append('#27ae60')  # Dark green for best
        elif imp > 0.04:
            colors.append('#2ecc71')  # Green
        elif imp > 0:
            colors.append('#3498db')  # Blue
        else:
            colors.append('#e74c3c')  # Red for regression
    
    bars = axes[0].barh(range(len(methods)), ious, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels(methods, fontsize=10)
    axes[0].set_xlabel('Mean IoU', fontsize=12, fontweight='bold')
    axes[0].set_title('IoU Comparison Across All Methods', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, iou) in enumerate(zip(bars, ious)):
        width = bar.get_width()
        axes[0].text(width + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{iou:.4f}',
                    ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Improvement waterfall
    baseline_val = comparison_df.iloc[0]['Mean IoU']
    x_pos = 0
    cumulative = baseline_val
    
    axes[1].bar(0, baseline_val, width=0.8, color='#95a5a6', alpha=0.8, 
               edgecolor='black', linewidth=1.5, label='Baseline')
    axes[1].text(0, baseline_val/2, f'{baseline_val:.4f}', 
                ha='center', va='center', fontweight='bold', fontsize=10)
    
    improvement_methods = comparison_df.iloc[1:].copy()
    for i, (_, row) in enumerate(improvement_methods.iterrows(), 1):
        improvement = row['vs Baseline']
        if improvement > 0:
            axes[1].bar(i, improvement, bottom=cumulative, width=0.8, 
                       color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
            axes[1].text(i, cumulative + improvement/2, f'+{improvement:.3f}',
                        ha='center', va='center', fontweight='bold', fontsize=9)
        else:
            axes[1].bar(i, -improvement, bottom=cumulative+improvement, width=0.8,
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
            axes[1].text(i, cumulative + improvement/2, f'{improvement:.3f}',
                        ha='center', va='center', fontweight='bold', fontsize=9)
        cumulative += improvement
    
    axes[1].set_xticks(range(len(comparison_df)))
    axes[1].set_xticklabels([m.replace(' + ', '+\n').replace(' (', '\n(') for m in comparison_df['Method']], 
                            rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('IoU', fontsize=12, fontweight='bold')
    axes[1].set_title('Cumulative Improvement Waterfall', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, max(ious) * 1.1)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'comprehensive_improvement_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {figures_dir / 'comprehensive_improvement_comparison.png'}")
    
    # Print key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    best_method = comparison_df.loc[comparison_df['Mean IoU'].idxmax()]
    total_improvement = best_method['vs Baseline']
    percent_improvement = 100 * total_improvement / baseline_iou
    
    print(f"\n✨ BEST APPROACH: {best_method['Method']}")
    print(f"   Mean IoU: {best_method['Mean IoU']:.4f}")
    print(f"   Total Improvement: +{total_improvement:.4f} ({percent_improvement:+.1f}%)")
    print(f"   {best_method['Description']}")
    
    print(f"\n📊 Breakdown of improvements:")
    for _, row in comparison_df.iloc[1:].iterrows():
        if row['vs Baseline'] > 0:
            print(f"   • {row['Method']}: +{row['vs Baseline']:.4f} ({100*row['vs Baseline']/baseline_iou:+.1f}%)")
    
    print("\n" + "="*70)
    print("✓ COMPREHENSIVE COMPARISON COMPLETE")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()


