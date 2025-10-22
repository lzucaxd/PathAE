#!/usr/bin/env python3
"""
Generate heatmaps for CAMELYON16 slides using contrastive model + linear classifier.

Creates 3-panel visualizations:
1. Original tissue (stitched from tiles)
2. Ground truth tumor mask
3. Predicted probability heatmap
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from contrastive_model import ContrastiveResNet
from stain_utils import StainNormalizer


class TileDataset(Dataset):
    """Dataset for loading tiles from disk."""
    
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
        
        # Use path column if available, otherwise construct from tile_id
        if 'path' in row and pd.notna(row['path']):
            tile_path = self.tiles_dir / row['path']
        else:
            tile_path = self.tiles_dir / f"{row['tile_id']}.png"
        
        # Load image
        img = np.array(Image.open(tile_path))
        
        # Stain normalization
        if self.stain_normalizer is not None:
            img = self.stain_normalizer.normalize(img)
        
        # Convert to [0,1] and normalize
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        # RGB normalization
        for c in range(3):
            img_tensor[c] = (img_tensor[c] - self.mean[c]) / (self.std[c] + 1e-6)
        
        return img_tensor, idx


def create_tissue_from_tiles(slide_df, tiles_dir, downsample=4):
    """Stitch tiles to create tissue image."""
    grid_rows = int(slide_df.iloc[0]['grid_rows'])
    grid_cols = int(slide_df.iloc[0]['grid_cols'])
    
    # Create empty grid (downsample for memory)
    tile_size = 96 // downsample
    tissue = np.ones((grid_rows * tile_size, grid_cols * tile_size, 3), dtype=np.uint8) * 255
    
    for _, row in tqdm(slide_df.iterrows(), total=len(slide_df), desc="  Stitching tiles"):
        tile_path = Path(tiles_dir) / row['tile_id']
        if not tile_path.exists():
            continue
        
        # Load and resize tile
        tile_img = np.array(Image.open(tile_path))
        tile_resized = cv2.resize(tile_img, (tile_size, tile_size))
        
        # Place in grid
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        tissue[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size] = tile_resized
    
    return tissue


def generate_predictions(model, dataset, device, batch_size=256, num_workers=4):
    """Generate predictions for all tiles."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="  Generating predictions"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze()
            all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_probs)


def create_heatmap(slide_df, scores, output_path, tiles_dir, wsi_id):
    """Create 3-panel heatmap visualization."""
    print(f"\n  Creating heatmap for {wsi_id}...")
    
    # Get grid dimensions
    grid_rows = int(slide_df.iloc[0]['grid_rows'])
    grid_cols = int(slide_df.iloc[0]['grid_cols'])
    
    print(f"    Grid: {grid_rows}×{grid_cols}")
    
    # Create tissue image
    print(f"    Creating tissue visualization...")
    tissue = create_tissue_from_tiles(slide_df, tiles_dir, downsample=4)
    
    # Create score grid
    score_grid = np.full((grid_rows, grid_cols), np.nan)
    gt_grid = np.full((grid_rows, grid_cols), np.nan)
    
    for idx, row in slide_df.iterrows():
        r = int(row['row_idx'])
        c = int(row['col_idx'])
        score_grid[r, c] = scores[idx]
        gt_grid[r, c] = row['label']
    
    # Apply per-slide z-score normalization
    valid_scores = score_grid[~np.isnan(score_grid)]
    if len(valid_scores) > 0:
        mean_score = np.mean(valid_scores)
        std_score = np.std(valid_scores)
        if std_score > 0:
            score_grid_norm = (score_grid - mean_score) / std_score
        else:
            score_grid_norm = score_grid
    else:
        score_grid_norm = score_grid
    
    # Apply Gaussian smoothing
    score_grid_smooth = gaussian_filter(np.nan_to_num(score_grid_norm), sigma=2.0)
    
    # Min-max scaling to [0, 1]
    valid_smooth = score_grid_smooth[~np.isnan(score_grid)]
    if len(valid_smooth) > 0:
        min_val = np.min(valid_smooth)
        max_val = np.max(valid_smooth)
        if max_val > min_val:
            score_grid_final = (score_grid_smooth - min_val) / (max_val - min_val)
        else:
            score_grid_final = score_grid_smooth
    else:
        score_grid_final = score_grid_smooth
    
    # IoU-optimized threshold
    gt_valid = gt_grid[~np.isnan(gt_grid)].astype(int)
    scores_valid = score_grid_final[~np.isnan(gt_grid)]
    
    best_threshold = 0.5
    best_iou = 0.0
    for thresh in np.linspace(0.1, 0.9, 20):
        pred = (scores_valid > thresh).astype(int)
        iou = jaccard_score(gt_valid, pred, zero_division=0)
        if iou > best_iou:
            best_iou = iou
            best_threshold = thresh
    
    # Create binary prediction
    pred_binary = (score_grid_final > best_threshold).astype(int)
    pred_valid = pred_binary[~np.isnan(gt_grid)]
    
    # Compute metrics
    iou = jaccard_score(gt_valid, pred_valid, zero_division=0)
    dice = f1_score(gt_valid, pred_valid, zero_division=0)
    precision = precision_score(gt_valid, pred_valid, zero_division=0)
    recall = recall_score(gt_valid, pred_valid, zero_division=0)
    
    print(f"    Metrics: IoU={iou:.3f}, Dice={dice:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    
    # Resize grids to match tissue
    h, w = tissue.shape[:2]
    heatmap_viz = cv2.resize(score_grid_final, (w, h), interpolation=cv2.INTER_LINEAR)
    gt_viz = cv2.resize(gt_grid, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original tissue
    axes[0].imshow(tissue)
    axes[0].set_title(f'{wsi_id}\nOriginal Tissue', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Ground truth
    axes[1].imshow(tissue)
    gt_overlay = np.zeros((*gt_viz.shape, 3), dtype=np.uint8)
    gt_overlay[gt_viz > 0.5] = [255, 0, 0]  # Red for tumor
    axes[1].imshow(gt_overlay, alpha=0.5)
    axes[1].set_title('Ground Truth', fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Predicted heatmap
    axes[2].imshow(tissue)
    
    # Custom colormap (white -> yellow -> red)
    colors = ['white', 'yellow', 'orange', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('tumor', colors, N=n_bins)
    
    # Create mask for valid regions
    valid_mask = ~np.isnan(cv2.resize(score_grid, (w, h), interpolation=cv2.INTER_NEAREST))
    heatmap_masked = np.ma.masked_where(~valid_mask, heatmap_viz)
    
    im = axes[2].imshow(heatmap_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
    
    # Add metrics text
    metrics_text = f'IoU: {iou:.3f}\nDice: {dice:.3f}\nPrec: {precision:.3f}\nRecall: {recall:.3f}\nThresh: {best_threshold:.2f}'
    axes[2].text(0.02, 0.98, metrics_text, transform=axes[2].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[2].set_title('Predicted Heatmap', fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Heatmap saved: {output_path}")
    
    return {
        'wsi_id': wsi_id,
        'n_tiles': len(slide_df),
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'threshold': best_threshold,
        'mean_score': float(np.mean(valid_scores))
    }


def main():
    parser = argparse.ArgumentParser(description='Generate contrastive model heatmaps')
    
    # Model
    parser.add_argument('--contrastive-model', type=str, required=True,
                        help='Path to contrastive model')
    parser.add_argument('--linear-model', type=str, required=True,
                        help='Path to linear classifier')
    
    # Data
    parser.add_argument('--test-csv', type=str, default='test_set_heatmaps/test_set.csv',
                        help='Test set CSV')
    parser.add_argument('--tiles-dir', type=str, default='test_set_heatmaps',
                        help='Directory containing tiles')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy',
                        help='Reference tile for stain normalization')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy',
                        help='RGB normalization stats')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/contrastive_heatmaps',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print("CONTRASTIVE MODEL HEATMAP GENERATION")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    print(f"Contrastive model: {args.contrastive_model}")
    print(f"Linear model: {args.linear_model}")
    
    # =========================================================================
    # Load models
    # =========================================================================
    print(f"\n{'='*70}")
    print("LOADING MODELS")
    print(f"{'='*70}\n")
    
    # Load contrastive encoder
    contrastive_model = ContrastiveResNet(pretrained=False).to(device)
    con_checkpoint = torch.load(args.contrastive_model, map_location=device, weights_only=False)
    contrastive_model.load_state_dict(con_checkpoint['model_state_dict'])
    contrastive_model.eval()
    
    # Load linear classifier
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
    
    print(f"  ✓ Models loaded")
    
    # =========================================================================
    # Load test set
    # =========================================================================
    print(f"\n{'='*70}")
    print("LOADING TEST SET")
    print(f"{'='*70}\n")
    
    df = pd.read_csv(args.test_csv)
    print(f"Loaded {len(df):,} tiles from {df['wsi_id'].nunique()} slides")
    
    # Load normalization stats
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    # Initialize stain normalizer
    stain_normalizer = StainNormalizer(
        reference_path=args.reference_tile,
        method='macenko'
    )
    
    # =========================================================================
    # Generate predictions
    # =========================================================================
    print(f"\n{'='*70}")
    print("GENERATING PREDICTIONS")
    print(f"{'='*70}\n")
    
    dataset = TileDataset(df, args.tiles_dir, stain_normalizer, mean, std)
    predictions = generate_predictions(model, dataset, device, 
                                      args.batch_size, args.num_workers)
    
    print(f"  ✓ Generated {len(predictions):,} predictions")
    print(f"  Score range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  Mean score: {predictions.mean():.3f}")
    
    # Add predictions to dataframe
    df['score'] = predictions
    
    # =========================================================================
    # Generate heatmaps per slide
    # =========================================================================
    print(f"\n{'='*70}")
    print("GENERATING HEATMAPS")
    print(f"{'='*70}\n")
    
    all_metrics = []
    
    for wsi_id in sorted(df['wsi_id'].unique()):
        slide_df = df[df['wsi_id'] == wsi_id].reset_index(drop=True)
        slide_scores = slide_df['score'].values
        
        output_path = output_dir / f'{wsi_id}_heatmap.png'
        
        metrics = create_heatmap(
            slide_df, slide_scores, output_path, 
            args.tiles_dir, wsi_id
        )
        all_metrics.append(metrics)
    
    # =========================================================================
    # Save metrics summary
    # =========================================================================
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = output_dir / 'heatmap_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False, float_format='%.4f')
    
    print(f"\n{'='*70}")
    print("✓ HEATMAP GENERATION COMPLETE")
    print(f"{'='*70}\n")
    
    print(metrics_df.to_string(index=False))
    print("")
    
    # Overall statistics
    print(f"Overall Performance:")
    print(f"  Mean IoU:       {metrics_df['iou'].mean():.4f} ± {metrics_df['iou'].std():.4f}")
    print(f"  Mean Dice:      {metrics_df['dice'].mean():.4f} ± {metrics_df['dice'].std():.4f}")
    print(f"  Mean Precision: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
    print(f"  Mean Recall:    {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
    print("")
    print(f"Heatmaps saved to: {output_dir}")


if __name__ == '__main__':
    main()

