#!/usr/bin/env python3
"""
Train β-VAE for unsupervised tumor detection.

Complete pipeline with:
- Stain normalization (Macenko)
- RGB normalization (PCam mean/std)
- β-VAE with L1 + SSIM + KL loss
- KL warm-up schedule
- Data augmentation
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from dataset import TissueDataset
from model_vae import BetaVAE, vae_loss, KLWarmup


def train_epoch(model, loader, optimizer, kl_scheduler, lambda_l1, lambda_ssim, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    
    beta = kl_scheduler.get_beta()
    
    for batch_idx, (images, _) in enumerate(tqdm(loader, desc=f"Training (β={beta:.3f})", leave=False)):
        images = images.to(device)
        
        # Forward
        recon, mu, logvar = model(images)
        
        # Loss
        loss, loss_dict = vae_loss(
            recon, images, mu, logvar,
            lambda_l1=lambda_l1,
            lambda_ssim=lambda_ssim,
            beta=beta
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss_dict['total']
        total_recon += loss_dict['recon']
        total_kl += loss_dict['kl']
    
    return {
        'loss': total_loss / len(loader),
        'recon': total_recon / len(loader),
        'kl': total_kl / len(loader),
        'beta': beta,
    }


def main():
    parser = argparse.ArgumentParser(description='Train β-VAE for tumor detection')
    
    # Data
    parser.add_argument('--data-csv', type=str, default='final_dataset/dataset.csv')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    
    # Model
    parser.add_argument('--z-dim', type=int, default=128, choices=[64, 128])
    parser.add_argument('--num-groups', type=int, default=8)
    
    # Loss
    parser.add_argument('--lambda-l1', type=float, default=0.6)
    parser.add_argument('--lambda-ssim', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=1.0, choices=[1.0, 3.0])
    parser.add_argument('--kl-warmup', type=int, default=10)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Preprocessing
    parser.add_argument('--stain-norm', action='store_true', default=True)
    parser.add_argument('--no-stain-norm', action='store_false', dest='stain_norm')
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', action='store_false', dest='augment')
    
    # Output
    parser.add_argument('--output', type=str, default='vae_best.pth')
    
    args = parser.parse_args()
    
    print("="*70)
    print("β-VAE TRAINING FOR TUMOR DETECTION")
    print("="*70)
    print()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  z_dim: {args.z_dim}")
    print(f"  β: {args.beta} (warmup over {args.kl_warmup} epochs)")
    print(f"  λ_L1: {args.lambda_l1}, λ_SSIM: {args.lambda_ssim}")
    print(f"  Stain norm: {args.stain_norm}")
    print(f"  Augment: {args.augment}")
    print()
    
    # Load normalization statistics
    if Path(args.norm_stats).exists():
        stats = np.load(args.norm_stats, allow_pickle=True).item()
        mean = stats['mean']
        std = stats['std']
        print(f"Loaded normalization stats:")
        print(f"  Mean: {mean}")
        print(f"  Std: {std}")
    else:
        print(f"Warning: {args.norm_stats} not found, using default [0.5, 0.5, 0.5]")
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    print()
    
    # Create dataset
    print("Loading dataset...")
    train_dataset = TissueDataset(
        csv_path=args.data_csv,
        split='train',
        stain_norm=args.stain_norm,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
        augment=args.augment,
        tissue_threshold=0.65,
        blur_threshold=30,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # For stable batch norm
    )
    
    print(f"  Batches per epoch: {len(train_loader):,}")
    print()
    
    # Create model
    print("Building β-VAE...")
    model = BetaVAE(z_dim=args.z_dim, num_groups=args.num_groups).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # KL warm-up
    kl_warmup = KLWarmup(beta_max=args.beta, warmup_epochs=args.kl_warmup)
    
    print("="*70)
    print("TRAINING")
    print("="*70)
    print()
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        metrics = train_epoch(
            model, train_loader, optimizer, kl_warmup,
            args.lambda_l1, args.lambda_ssim, device
        )
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {metrics['loss']:.6f} | "
              f"Recon: {metrics['recon']:.6f} | "
              f"KL: {metrics['kl']:.6f} | "
              f"β: {metrics['beta']:.3f}", end="")
        
        # Learning rate scheduler
        scheduler.step(metrics['loss'])
        
        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['loss'],
                'config': {
                    'z_dim': args.z_dim,
                    'beta': args.beta,
                    'lambda_l1': args.lambda_l1,
                    'lambda_ssim': args.lambda_ssim,
                    'num_groups': args.num_groups,
                    'mean': mean,
                    'std': std,
                }
            }, args.output)
            print(" ← Best!")
        else:
            print()
        
        # Step KL warmup
        kl_warmup.step()
    
    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Model saved to: {args.output}")
    print()
    print("NEXT STEPS:")
    print(f"  1. Run inference:")
    print(f"     python run_inference_vae.py --model {args.output} --test-csv test_set_heatmaps/test_set.csv")
    print(f"  2. Generate heatmaps:")
    print(f"     python generate_heatmaps.py --test-csv test_set_heatmaps/test_set.csv --scores-csv reconstruction_scores.csv")
    print(f"  3. Compute metrics:")
    print(f"     python compute_metrics.py --test-csv test_set_heatmaps/test_set.csv --scores-csv reconstruction_scores.csv")


if __name__ == '__main__':
    main()

