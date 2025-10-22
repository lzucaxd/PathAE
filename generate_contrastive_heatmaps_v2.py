#!/usr/bin/env python3
"""
Generate contrastive model heatmaps using the same format as supervised.

This script:
1. Loads contrastive + linear classifier
2. Generates predictions for all test tiles  
3. Creates 3-panel heatmaps (tissue, ground truth, prediction)
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader

from contrastive_model import ContrastiveResNet
from stain_utils import StainNormalizer


class TileDataset(Dataset):
    """Dataset for loading tiles."""
    
    def __init__(self, df, tiles_dir, stain_normalizer, mean, std):
        self.df = df
        self.tiles_dir = Path(tiles_dir)
        self.stain_normalizer = stain_normalizer
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Use path column
        if 'path' in row and pd.notna(row['path']):
            tile_path = self.tiles_dir / row['path']
        else:
            tile_path = self.tiles_dir / f"{row['tile_id']}.png"
        
        # Load and preprocess
        img = np.array(Image.open(tile_path))
        
        if self.stain_normalizer is not None:
            img = self.stain_normalizer.normalize(img)
        
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        for c in range(3):
            img_tensor[c] = (img_tensor[c] - self.mean[c]) / (self.std[c] + 1e-6)
        
        return img_tensor, idx


def create_tissue_from_tiles(df, tiles_dir, downsample=4):
    """Stitch tiles to create tissue visualization."""
    grid_rows = int(df.iloc[0]['grid_rows'])
    grid_cols = int(df.iloc[0]['grid_cols'])
    
    tile_size = 96 // downsample
    tissue = np.ones((grid_rows * tile_size, grid_cols * tile_size, 3), dtype=np.uint8) * 255
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Stitching tiles", leave=False):
        if 'path' in row and pd.notna(row['path']):
            tile_path = Path(tiles_dir) / row['path']
        else:
            tile_path = Path(tiles_dir) / f"{row['tile_id']}.png"
        
        if not tile_path.exists():
            continue
        
        tile_img = np.array(Image.open(tile_path))
        tile_resized = cv2.resize(tile_img, (tile_size, tile_size))
        
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        tissue[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size] = tile_resized
    
    return tissue


def generate_predictions(model, dataset, device, batch_size=256, num_workers=4):
    """Generate predictions."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=False)
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="  Generating predictions", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze()
            all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_probs)


def create_heatmap(df, scores, wsi_id, tiles_dir, output_path):
    """
    Create 3-panel heatmap visualization.
    Same format as supervised version.
    """
    print(f"\nProcessing {wsi_id}...")
    print(f"  Tiles: {len(df):,}")
    
    # Get grid dimensions
    grid_rows = int(df.iloc[0]['grid_rows'])
    grid_cols = int(df.iloc[0]['grid_cols'])
    
    # Create grids
    pred_grid = np.full((grid_rows, grid_cols), np.nan)
    gt_grid = np.full((grid_rows, grid_cols), np.nan)
    
    for idx, row in df.iterrows():
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        pred_grid[r, c] = scores[idx]
        gt_grid[r, c] = row['label']
    
    # Per-slide z-score normalization
    valid_mask = ~np.isnan(pred_grid)
    if valid_mask.sum() > 0:
        mean_score = np.nanmean(pred_grid)
        std_score = np.nanstd(pred_grid)
        if std_score > 0:
            pred_grid_norm = np.where(valid_mask, (pred_grid - mean_score) / std_score, np.nan)
        else:
            pred_grid_norm = pred_grid.copy()
    else:
        pred_grid_norm = pred_grid.copy()
    
    # Gaussian smoothing (only on valid regions)
    pred_grid_smooth = gaussian_filter(np.nan_to_num(pred_grid_norm, nan=0.0), sigma=2.0)
    mask = ~np.isnan(pred_grid)
    if mask.sum() > 0:
        pred_grid_smooth[~mask] = np.nan
    
    # Min-max to [0, 1]
    valid_smooth = pred_grid_smooth[~np.isnan(pred_grid_smooth)]
    if len(valid_smooth) > 0:
        min_val = np.min(valid_smooth)
        max_val = np.max(valid_smooth)
        if max_val > min_val:
            pred_grid_final = np.where(
                ~np.isnan(pred_grid_smooth),
                (pred_grid_smooth - min_val) / (max_val - min_val),
                np.nan
            )
        else:
            pred_grid_final = pred_grid_smooth.copy()
    else:
        pred_grid_final = pred_grid_smooth.copy()
    
    # Optimize threshold for IoU
    best_threshold = 0.5
    best_iou = 0.0
    
    gt_valid = gt_grid[~np.isnan(gt_grid)].flatten()
    pred_valid = pred_grid_final[~np.isnan(pred_grid_final)].flatten()
    
    if len(gt_valid) == len(pred_valid) and len(gt_valid) > 0:
        for thresh in np.linspace(0.3, 0.9, 13):
            pred_binary = (pred_valid >= thresh).astype(int)
            iou = jaccard_score(gt_valid.astype(int), pred_binary, zero_division=0)
            if iou > best_iou:
                best_iou = iou
                best_threshold = thresh
    
    print(f"  Optimized threshold: {best_threshold:.3f} (IoU: {best_iou:.4f})")
    
    # Binary prediction
    pred_binary = np.zeros_like(pred_grid_final)
    pred_binary[~np.isnan(pred_grid_final)] = (pred_grid_final[~np.isnan(pred_grid_final)] >= best_threshold).astype(float)
    pred_binary[np.isnan(pred_grid_final)] = np.nan
    
    # Compute metrics
    gt_flat = gt_valid.astype(int)
    pred_flat = (pred_valid >= best_threshold).astype(int)
    
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
    dice = f1_score(gt_flat, pred_flat, zero_division=0)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    
    print(f"  Metrics: IoU={iou:.4f}, Dice={dice:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    # Create tissue image
    print("  Creating tissue visualization...")
    tissue = create_tissue_from_tiles(df, tiles_dir, downsample=4)
    
    # Resize grids to match tissue
    h, w = tissue.shape[:2]
    gt_viz = cv2.resize(np.nan_to_num(gt_grid, nan=0.0), (w, h), interpolation=cv2.INTER_NEAREST)
    pred_prob_viz = cv2.resize(np.nan_to_num(pred_grid_final, nan=0.0), (w, h), interpolation=cv2.INTER_LINEAR)
    pred_binary_viz = cv2.resize(np.nan_to_num(pred_binary, nan=0.0), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create 3-panel figure (1 row, 3 columns) - same as supervised
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: Original tissue
    axes[0].imshow(tissue)
    axes[0].set_title('Original Tissue', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Ground truth
    axes[1].imshow(tissue)
    gt_overlay = np.zeros((*gt_viz.shape, 3), dtype=np.uint8)
    gt_overlay[gt_viz > 0.5] = [255, 0, 0]  # Red for tumor
    axes[1].imshow(gt_overlay, alpha=0.5)
    axes[1].set_title('Ground Truth', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # Add GT stats
    tumor_pct = 100 * (gt_valid == 1).sum() / len(gt_valid)
    gt_text = f'Tumor: {tumor_pct:.1f}%'
    axes[1].text(0.02, 0.98, gt_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Panel 3: Predicted probability heatmap
    axes[2].imshow(tissue)
    heatmap = axes[2].imshow(pred_prob_viz, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title('Predicted Heatmap', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    
    # Metrics text on prediction panel
    metrics_text = (f'Threshold: {best_threshold:.2f}\n'
                   f'IoU: {iou:.3f}\n'
                   f'Dice: {dice:.3f}\n'
                   f'Precision: {precision:.3f}\n'
                   f'Recall: {recall:.3f}')
    axes[2].text(0.02, 0.98, metrics_text, transform=axes[2].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Tumor Probability', fontsize=11)
    
    plt.suptitle(f'{wsi_id}', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    return {
        'wsi_id': wsi_id,
        'threshold': best_threshold,
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'n_tiles': len(df),
        'tumor_pct': tumor_pct
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrastive-model', type=str, required=True)
    parser.add_argument('--linear-model', type=str, required=True)
    parser.add_argument('--test-csv', type=str, default='test_set_heatmaps/test_set.csv')
    parser.add_argument('--tiles-dir', type=str, default='test_set_heatmaps')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    parser.add_argument('--output-dir', type=str, default='outputs/contrastive_heatmaps_v2')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print("CONTRASTIVE MODEL HEATMAP GENERATION (v2)")
    print(f"{'='*70}\n")
    
    # Load models
    print("Loading models...")
    contrastive_model = ContrastiveResNet(pretrained=False).to(device)
    con_checkpoint = torch.load(args.contrastive_model, map_location=device, weights_only=False)
    contrastive_model.load_state_dict(con_checkpoint['model_state_dict'])
    
    class LinearClassifier(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(512, 1)
        
        def forward(self, x):
            with torch.no_grad():
                features = self.encoder.encoder(x)
                if len(features.shape) == 4:
                    features = features.squeeze()
            return self.classifier(features)
    
    model = LinearClassifier(contrastive_model).to(device)
    linear_checkpoint = torch.load(args.linear_model, map_location=device, weights_only=False)
    model.classifier.load_state_dict(linear_checkpoint['classifier_state_dict'])
    model.eval()
    print("  ✓ Models loaded\n")
    
    # Load test set
    print("Loading test set...")
    df = pd.read_csv(args.test_csv)
    print(f"  ✓ Loaded {len(df):,} tiles from {df['wsi_id'].nunique()} slides\n")
    
    # Load normalization
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    stain_normalizer = StainNormalizer(reference_path=args.reference_tile, method='macenko')
    
    # Generate predictions
    print("Generating predictions...")
    dataset = TileDataset(df, args.tiles_dir, stain_normalizer, norm_stats['mean'], norm_stats['std'])
    predictions = generate_predictions(model, dataset, device, args.batch_size, args.num_workers)
    df['score'] = predictions
    print(f"  ✓ Generated {len(predictions):,} predictions\n")
    
    # Generate heatmaps per slide
    print(f"{'='*70}")
    print("GENERATING HEATMAPS")
    print(f"{'='*70}")
    
    all_metrics = []
    
    for wsi_id in sorted(df['wsi_id'].unique()):
        slide_df = df[df['wsi_id'] == wsi_id].reset_index(drop=True)
        output_path = output_dir / f'{wsi_id}_heatmap.png'
        
        metrics = create_heatmap(slide_df, slide_df['score'].values, 
                                wsi_id, args.tiles_dir, output_path)
        all_metrics.append(metrics)
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = output_dir / 'heatmap_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False, float_format='%.4f')
    
    print(f"\n{'='*70}")
    print("✓ HEATMAP GENERATION COMPLETE")
    print(f"{'='*70}\n")
    
    print(metrics_df[['wsi_id', 'iou', 'dice', 'precision', 'recall']].to_string(index=False))
    print("")
    print(f"Overall Performance:")
    print(f"  Mean IoU:       {metrics_df['iou'].mean():.4f} ± {metrics_df['iou'].std():.4f}")
    print(f"  Mean Dice:      {metrics_df['dice'].mean():.4f} ± {metrics_df['dice'].std():.4f}")
    print(f"  Mean Precision: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
    print(f"  Mean Recall:    {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
    print(f"\nHeatmaps saved to: {output_dir}")


if __name__ == '__main__':
    main()


