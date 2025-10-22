#!/usr/bin/env python3
"""
Run inference with trained Autoencoder to compute reconstruction errors for heatmap generation.

Usage:
    python run_inference_ae.py --model experiments/B2_AE-z128-L1SSIM/model_best.pth --test-csv test_set_heatmaps/test_set.csv
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from dataset import TestDataset
from model_ae import AE96
from pytorch_msssim import ssim
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Run AE inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--output', type=str, default='test_scores.csv')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--stain-norm', action='store_true', default=True)
    
    args = parser.parse_args()
    
    print("="*70)
    print("AUTOENCODER INFERENCE: Computing Reconstruction Errors")
    print("="*70)
    print()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    config = checkpoint['config']
    z_dim = config['z_dim']
    mean = config['mean']
    std = config['std']
    
    # Build model
    model = AE96(z_dim=z_dim, num_groups=config.get('num_groups', 8)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Model: Standard AE")
    print(f"  z_dim: {z_dim}")
    print(f"  Bottleneck: {z_dim * 9} dims")
    print(f"  Train loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print()
    
    # Load test set
    print(f"Loading test set...")
    test_dataset = TestDataset(
        csv_path=args.test_csv,
        stain_norm=args.stain_norm,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"  Total tiles: {len(test_dataset):,}")
    print()
    
    # Load metadata for label analysis
    test_df = pd.read_csv(args.test_csv)
    
    # Compute reconstruction errors
    print("Computing reconstruction errors (0.6*MSE + 0.4*(1-SSIM))...")
    results = []
    
    # Convert mean/std to tensors for denormalization
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for images, tile_ids in tqdm(test_loader, desc="Processing"):
            images = images.to(device)
            
            # Reconstruction (AE is deterministic, no sampling)
            recon, _, _ = model(images)
            
            # Denormalize to [0,1] for proper MSE and SSIM computation
            recon_01 = torch.clamp(recon * std_t + mean_t, 0, 1)
            images_01 = torch.clamp(images * std_t + mean_t, 0, 1)
            
            # MSE per tile (in [0,1] space)
            mse = ((images_01 - recon_01) ** 2).mean(dim=[1, 2, 3])
            
            # SSIM per tile (in [0,1] space)
            ssim_vals = ssim(images_01, recon_01, data_range=1.0, size_average=False)
            
            # Combined score: 0.6*MSE + 0.4*(1-SSIM)
            combined_score = 0.6 * mse + 0.4 * (1.0 - ssim_vals)
            
            # Save results
            for tile_id, score in zip(tile_ids, combined_score.cpu().numpy()):
                results.append({'tile_id': tile_id, 'score': float(score)})
    
    # Save scores
    scores_df = pd.DataFrame(results)
    scores_df.to_csv(args.output, index=False)
    
    # Statistics
    print()
    print("="*70)
    print("RECONSTRUCTION ERROR STATISTICS")
    print("="*70)
    print(f"Total tiles: {len(scores_df):,}")
    print(f"Mean error: {scores_df['score'].mean():.6f}")
    print(f"Std error: {scores_df['score'].std():.6f}")
    print(f"Min error: {scores_df['score'].min():.6f}")
    print(f"Max error: {scores_df['score'].max():.6f}")
    print()
    
    # Compare tumor vs normal
    merged = test_df.merge(scores_df, on='tile_id', how='inner')
    
    if 'label' in merged.columns:
        tumor_scores = merged[merged['label'] == 1]['score']
        normal_scores = merged[merged['label'] == 0]['score']
        
        print("By label:")
        print(f"  Tumor  (n={len(tumor_scores):,}): {tumor_scores.mean():.6f} ± {tumor_scores.std():.6f}")
        print(f"  Normal (n={len(normal_scores):,}): {normal_scores.mean():.6f} ± {normal_scores.std():.6f}")
        
        if len(tumor_scores) > 0 and len(normal_scores) > 0:
            ratio = tumor_scores.mean() / normal_scores.mean()
            print(f"  Ratio (tumor/normal): {ratio:.2f}x")
            
            if ratio > 1.5:
                print(f"\n✓ Good separation! Tumors have {ratio:.2f}x higher error than normals")
            else:
                print(f"\n⚠ Warning: Low separation ({ratio:.2f}x). May need more training.")
    
    print()
    print(f"✓ Scores saved to: {args.output}")
    print()
    print("NEXT STEPS:")
    print(f"  1. Generate heatmaps:")
    print(f"     python generate_heatmaps_v2.py --test-csv {args.test_csv} --scores-csv {args.output}")
    print(f"  2. Compute metrics:")
    print(f"     python compute_metrics.py --test-csv {args.test_csv} --scores-csv {args.output}")


if __name__ == '__main__':
    main()


