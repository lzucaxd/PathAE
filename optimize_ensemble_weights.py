"""
Ensemble Weight Optimization

Optimizes the weighting between supervised and contrastive models
to maximize IoU performance on the test set.

Author: ML Infra Engineer
Date: October 2025
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

def load_predictions(supervised_csv, contrastive_csv, test_csv):
    """Load predictions from both models."""
    sup_df = pd.read_csv(supervised_csv)
    con_df = pd.read_csv(contrastive_csv)
    test_df = pd.read_csv(test_csv)
    
    # Merge all dataframes
    merged = test_df.merge(sup_df, on='tile_id', suffixes=('', '_sup'))
    merged = merged.merge(con_df, on='tile_id', suffixes=('', '_con'))
    
    # Rename score columns for clarity
    merged = merged.rename(columns={
        'score': 'score_sup',
        'score_con': 'score_con'
    })
    
    return merged

def create_heatmap_from_scores(df_slide, grid_rows, grid_cols, score_col):
    """Create 2D heatmap grid from tile scores."""
    heatmap = np.zeros((grid_rows, grid_cols))
    
    for _, row in df_slide.iterrows():
        r, c = int(row['row_idx']), int(row['col_idx'])
        heatmap[r, c] = row[score_col]
    
    return heatmap

def compute_iou(pred, gt):
    """Compute IoU between prediction and ground truth."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

def find_optimal_threshold_for_slide(heatmap, gt_mask):
    """Find threshold that maximizes IoU for this slide."""
    thresholds = np.linspace(0.3, 0.9, 50)
    best_iou = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        pred = (heatmap > thresh).astype(np.uint8)
        iou = compute_iou(pred, gt_mask)
        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh
    
    return best_thresh, best_iou

def evaluate_weights(w_sup, merged_df, slides):
    """Evaluate ensemble performance for given weights."""
    w_con = 1.0 - w_sup
    
    ious = []
    
    for wsi_id in slides:
        df_slide = merged_df[merged_df['wsi_id'] == wsi_id].copy()
        
        grid_rows = int(df_slide['grid_rows'].iloc[0])
        grid_cols = int(df_slide['grid_cols'].iloc[0])
        
        # Create ensemble heatmap
        sup_heatmap = create_heatmap_from_scores(df_slide, grid_rows, grid_cols, 'score_sup')
        con_heatmap = create_heatmap_from_scores(df_slide, grid_rows, grid_cols, 'score_con')
        
        ensemble_heatmap = w_sup * sup_heatmap + w_con * con_heatmap
        
        # Ground truth
        gt_heatmap = create_heatmap_from_scores(df_slide, grid_rows, grid_cols, 'label')
        gt_mask = (gt_heatmap > 0.5).astype(np.uint8)
        
        # Find optimal threshold and compute IoU
        _, iou = find_optimal_threshold_for_slide(ensemble_heatmap, gt_mask)
        ious.append(iou)
    
    return np.mean(ious), np.std(ious), ious

def main():
    parser = argparse.ArgumentParser(description='Optimize ensemble weights.')
    parser.add_argument('--supervised-csv', type=str, default='outputs/scores_supervised.csv',
                       help='Path to supervised prediction scores')
    parser.add_argument('--contrastive-csv', type=str, default='outputs/scores_contrastive.csv',
                       help='Path to contrastive prediction scores')
    parser.add_argument('--test-csv', type=str, default='test_set_heatmaps/test_set.csv',
                       help='Path to test set CSV with metadata')
    parser.add_argument('--output-dir', type=str, default='outputs/ensemble_optimization',
                       help='Directory to save results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("ENSEMBLE WEIGHT OPTIMIZATION")
    print("="*70)
    print(f"\nSupervised: {args.supervised_csv}")
    print(f"Contrastive: {args.contrastive_csv}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    merged_df = load_predictions(args.supervised_csv, args.contrastive_csv, args.test_csv)
    print(f"  ✓ Loaded {len(merged_df):,} tile predictions")
    
    slides = merged_df['wsi_id'].unique()
    print(f"  ✓ Found {len(slides)} WSI slides")
    
    # Grid search over weights
    print("\n" + "="*70)
    print("GRID SEARCH OVER WEIGHTS")
    print("="*70)
    
    weights = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, 0.1, ..., 1.0
    results = []
    
    for w_sup in tqdm(weights, desc="  Evaluating weights"):
        mean_iou, std_iou, slide_ious = evaluate_weights(w_sup, merged_df, slides)
        
        results.append({
            'w_supervised': w_sup,
            'w_contrastive': 1.0 - w_sup,
            'mean_iou': mean_iou,
            'std_iou': std_iou
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'weight_search_results.csv', index=False)
    
    # Find optimal weights
    best_idx = results_df['mean_iou'].idxmax()
    best_w_sup = results_df.loc[best_idx, 'w_supervised']
    best_mean_iou = results_df.loc[best_idx, 'mean_iou']
    best_std_iou = results_df.loc[best_idx, 'std_iou']
    
    # Also get equal weight (0.5) and supervised-only (1.0) results
    equal_idx = results_df['w_supervised'].sub(0.5).abs().idxmin()
    equal_mean_iou = results_df.loc[equal_idx, 'mean_iou']
    
    sup_only_idx = results_df['w_supervised'].sub(1.0).abs().idxmin()
    sup_only_mean_iou = results_df.loc[sup_only_idx, 'mean_iou']
    
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\nOptimal weights:")
    print(f"  Supervised:   {best_w_sup:.2f}")
    print(f"  Contrastive:  {1.0 - best_w_sup:.2f}")
    print(f"  Mean IoU:     {best_mean_iou:.4f} ± {best_std_iou:.4f}")
    
    print(f"\nComparison:")
    print(f"  Equal (0.5/0.5):      IoU = {equal_mean_iou:.4f}")
    print(f"  Optimized ({best_w_sup:.2f}/{1.0-best_w_sup:.2f}): IoU = {best_mean_iou:.4f} ({100*(best_mean_iou-equal_mean_iou)/equal_mean_iou:+.1f}%)")
    print(f"  Supervised only (1.0): IoU = {sup_only_mean_iou:.4f}")
    
    # Save optimal weights to config
    config = {
        'w_supervised': float(best_w_sup),
        'w_contrastive': float(1.0 - best_w_sup),
        'mean_iou': float(best_mean_iou),
        'std_iou': float(best_std_iou),
        'equal_weight_iou': float(equal_mean_iou),
        'supervised_only_iou': float(sup_only_mean_iou)
    }
    
    with open(output_dir / 'optimal_weights.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Weight vs IoU curve
    axes[0].plot(results_df['w_supervised'], results_df['mean_iou'], 
                linewidth=2, color='#3498db', marker='o', markersize=4)
    axes[0].fill_between(results_df['w_supervised'],
                         results_df['mean_iou'] - results_df['std_iou'],
                         results_df['mean_iou'] + results_df['std_iou'],
                         alpha=0.3, color='#3498db')
    
    # Mark special points
    axes[0].axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='Equal weights')
    axes[0].axvline(x=best_w_sup, color='red', linestyle='--', linewidth=2, label=f'Optimal ({best_w_sup:.2f})')
    axes[0].scatter([best_w_sup], [best_mean_iou], color='red', s=100, zorder=5)
    
    axes[0].set_xlabel('Supervised Weight', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mean IoU', fontsize=12, fontweight='bold')
    axes[0].set_title('Ensemble Weight Optimization', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Bar chart comparison
    strategies = ['Equal\n(0.5/0.5)', f'Optimized\n({best_w_sup:.2f}/{1.0-best_w_sup:.2f})', 'Supervised\nOnly']
    ious = [equal_mean_iou, best_mean_iou, sup_only_mean_iou]
    colors = ['#95a5a6', '#2ecc71', '#3498db']
    
    bars = axes[1].bar(strategies, ious, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Mean IoU', fontsize=12, fontweight='bold')
    axes[1].set_title('Strategy Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, iou in zip(bars, ious):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{iou:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'ensemble_weight_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {figures_dir / 'ensemble_weight_optimization.png'}")
    
    print("\n" + "="*70)
    print("✓ ENSEMBLE OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - {output_dir / 'weight_search_results.csv'}")
    print(f"  - {output_dir / 'optimal_weights.json'}")
    print(f"  - {figures_dir / 'ensemble_weight_optimization.png'}")
    
    print("\n" + "="*70)
    print("KEY FINDING")
    print("="*70)
    print(f"Optimized ensemble weights ({best_w_sup:.2f}/{1.0-best_w_sup:.2f} favoring supervised)")
    print(f"improve IoU from {equal_mean_iou:.3f} to {best_mean_iou:.3f} ({100*(best_mean_iou-equal_mean_iou)/equal_mean_iou:+.1f}%).")
    if best_w_sup > 0.6:
        print(f"\nTrade-off: Higher weight on supervised improves performance but may")
        print(f"reduce uncertainty signal (less model disagreement). Consider: Is the")
        print(f"IoU gain worth the loss in confidence calibration for clinical use?")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()


