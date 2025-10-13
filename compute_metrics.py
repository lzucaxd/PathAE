#!/usr/bin/env python3
"""
Compute evaluation metrics for tumor detection.

Metrics:
1. Patch-level: AUC-ROC, precision, recall
2. Pixel-level: Heatmap AUC
3. Slide-level: FROC curve
4. Localization: Centroid distance, Hausdorff
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, jaccard_score
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import label as connected_components
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_patch_level_metrics(test_csv, scores_csv):
    """Compute patch-level classification metrics."""
    print("\n" + "="*70)
    print("PATCH-LEVEL METRICS")
    print("="*70)
    
    # Load data
    tiles_df = pd.read_csv(test_csv)
    scores_df = pd.read_csv(scores_csv)
    
    # Merge scores with tiles (inner join ensures only tissue tiles with scores)
    df = tiles_df.merge(scores_df, on='tile_id', how='inner')
    
    print(f"\n  Total tiles: {len(df):,}")
    print(f"  Tumor tiles: {(df['label'] == 1).sum():,}")
    print(f"  Normal tiles: {(df['label'] == 0).sum():,}")
    
    y_true = df['label'].values
    y_scores = df['score'].values
    
    # AUC-ROC
    auc = roc_auc_score(y_true, y_scores)
    print(f"\n  AUC-ROC: {auc:.4f}")
    
    # PR-AUC (Average Precision)
    pr_auc = average_precision_score(y_true, y_scores)
    print(f"  PR-AUC (Precision-Recall): {pr_auc:.4f}")
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find optimal threshold (Youden's index)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_thresh = thresholds[optimal_idx]
    
    # Metrics at optimal threshold
    y_pred = (y_scores >= optimal_thresh).astype(int)
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # F1 using sklearn (same result, but more robust)
    f1_sklearn = f1_score(y_true, y_pred, zero_division=0)
    
    # Dice Score (equivalent to F1 for binary)
    dice = f1_sklearn
    
    # IoU (Jaccard Index)
    iou = jaccard_score(y_true, y_pred, zero_division=0)
    
    print(f"\n  Optimal threshold: {optimal_thresh:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Dice Score: {dice:.4f}")
    print(f"  IoU (Jaccard): {iou:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {tp:6,}  FP: {fp:6,}")
    print(f"    FN: {fn:6,}  TN: {tn:6,}")
    
    return {
        'auc_roc': auc,
        'pr_auc': pr_auc,
        'optimal_threshold': optimal_thresh,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'dice': dice,
        'iou': iou,
    }


def compute_pixel_level_auc(test_csv, scores_csv, mask_dir, wsi_dir):
    """Compute pixel-level AUC from heatmaps."""
    print("\n" + "="*70)
    print("PIXEL-LEVEL METRICS")
    print("="*70)
    
    tiles_df = pd.read_csv(test_csv)
    scores_df = pd.read_csv(scores_csv)
    
    df = tiles_df.merge(scores_df, on='tile_id', how='inner')
    
    all_pixel_labels = []
    all_pixel_scores = []
    
    for wsi_id in tqdm(df['wsi_id'].unique(), desc="  Computing pixel-level AUC"):
        slide_df = df[df['wsi_id'] == wsi_id]
        
        # Get grid dimensions
        grid_rows = int(slide_df.iloc[0]['grid_rows'])
        grid_cols = int(slide_df.iloc[0]['grid_cols'])
        
        # Create score grid
        score_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        for _, row in slide_df.iterrows():
            score_grid[int(row['row_idx']), int(row['col_idx'])] = row['score']
        
        # Load mask
        mask_path = Path(mask_dir) / f"{wsi_id}_mask.tif"
        if not mask_path.exists():
            continue
        
        mask = np.array(Image.open(mask_path))
        
        # Resize mask to grid size
        mask_resized = cv2.resize(mask, (grid_cols, grid_rows), 
                                   interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 127).astype(int)
        
        # Collect pixel-level predictions
        all_pixel_labels.extend(mask_binary.flatten())
        all_pixel_scores.extend(score_grid.flatten())
    
    # Compute pixel-level AUC
    pixel_auc = roc_auc_score(all_pixel_labels, all_pixel_scores)
    print(f"\n  Pixel-level AUC: {pixel_auc:.4f}")
    
    return pixel_auc


def compute_froc(test_csv, scores_csv, mask_dir, wsi_dir):
    """
    Compute FROC curve (Free-Response ROC).
    
    Sensitivity vs. average false positives per slide.
    """
    print("\n" + "="*70)
    print("FROC (Free-Response ROC)")
    print("="*70)
    
    tiles_df = pd.read_csv(test_csv)
    scores_df = pd.read_csv(scores_csv)
    
    df = tiles_df.merge(scores_df, on='tile_id', how='inner')
    
    # Group by slide
    wsi_ids = df['wsi_id'].unique()
    n_slides = len(wsi_ids)
    
    thresholds = np.linspace(df['score'].min(), df['score'].max(), 100)
    
    sensitivities = []
    fps_per_slide = []
    
    for thresh in tqdm(thresholds, desc="  Computing FROC"):
        total_tp = 0
        total_tumors = 0
        total_fp = 0
        
        for wsi_id in wsi_ids:
            slide_df = df[df['wsi_id'] == wsi_id]
            
            # Predictions at this threshold
            preds = (slide_df['score'] >= thresh).astype(int)
            labels = slide_df['label'].values
            
            # Count metrics
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            
            total_tp += tp
            total_tumors += (labels == 1).sum()
            total_fp += fp
        
        sensitivity = total_tp / total_tumors if total_tumors > 0 else 0
        avg_fp = total_fp / n_slides
        
        sensitivities.append(sensitivity)
        fps_per_slide.append(avg_fp)
    
    # Compute partial FROC (mean sensitivity at specific FP rates)
    fp_targets = [0.25, 0.5, 1, 2, 4, 8]
    partial_froc = []
    
    for fp_target in fp_targets:
        # Find closest FP rate
        idx = np.argmin(np.abs(np.array(fps_per_slide) - fp_target))
        sens = sensitivities[idx]
        partial_froc.append(sens)
    
    avg_partial_froc = np.mean(partial_froc)
    
    print(f"\n  Partial FROC (mean sensitivity at fixed FP rates):")
    for fp_target, sens in zip(fp_targets, partial_froc):
        print(f"    {fp_target:4.2f} FP/slide: sensitivity = {sens:.4f}")
    
    print(f"\n  Average Partial FROC: {avg_partial_froc:.4f}")
    
    # Plot FROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fps_per_slide, sensitivities, linewidth=2)
    plt.scatter([fps_per_slide[np.argmin(np.abs(np.array(fps_per_slide) - fp))] 
                 for fp in fp_targets],
                partial_froc, c='red', s=100, zorder=5, label='Evaluation points')
    plt.xlabel('Average False Positives per Slide', fontsize=12)
    plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
    plt.title('FROC Curve: Tumor Localization Performance', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('froc_curve.png', dpi=150)
    plt.close()
    
    print(f"\n  ✓ FROC curve saved to: froc_curve.png")
    
    return {
        'partial_froc': avg_partial_froc,
        'fps_per_slide': fps_per_slide,
        'sensitivities': sensitivities,
    }


def main():
    parser = argparse.ArgumentParser(description='Compute evaluation metrics')
    parser.add_argument('--test-csv', type=str, required=True,
                        help='Path to test_set.csv')
    parser.add_argument('--scores-csv', type=str, required=True,
                        help='Path to reconstruction scores CSV (tile_id, score)')
    parser.add_argument('--wsi-dir', type=str, default='cam16_prepped/wsi')
    parser.add_argument('--mask-dir', type=str, default='cam16_prepped/masks_tif')
    
    args = parser.parse_args()
    
    print("="*70)
    print("EVALUATION METRICS FOR TUMOR DETECTION")
    print("="*70)
    
    # 1. Patch-level metrics
    patch_metrics = compute_patch_level_metrics(args.test_csv, args.scores_csv)
    
    # 2. Pixel-level AUC
    pixel_auc = compute_pixel_level_auc(args.test_csv, args.scores_csv, 
                                         args.mask_dir, args.wsi_dir)
    
    # 3. FROC
    froc_metrics = compute_froc(args.test_csv, args.scores_csv,
                                  args.mask_dir, args.wsi_dir)
    
    # Save summary
    summary = {
        'Patch-level AUC-ROC': patch_metrics['auc_roc'],
        'Patch-level PR-AUC': patch_metrics['pr_auc'],
        'Pixel-level AUC': pixel_auc,
        'Partial FROC': froc_metrics['partial_froc'],
        'Optimal Threshold': patch_metrics['optimal_threshold'],
        'Precision': patch_metrics['precision'],
        'Recall': patch_metrics['recall'],
        'F1-Score': patch_metrics['f1'],
        'Dice Score': patch_metrics['dice'],
        'IoU (Jaccard)': patch_metrics['iou'],
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('evaluation_summary.csv', index=False)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for key, val in summary.items():
        print(f"  {key:25}: {val:.4f}")
    
    print("\n✓ Evaluation complete!")
    print(f"  Summary: evaluation_summary.csv")
    print(f"  FROC curve: froc_curve.png")


if __name__ == '__main__':
    main()

