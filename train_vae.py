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
import matplotlib.pyplot as plt

from dataset import TissueDataset
from model_vae import BetaVAE, vae_loss, KLWarmup


def save_reconstructions(model, dataset, epoch, output_dir, mean, std, device, n_samples=8):
    """Save reconstruction visualizations."""
    model.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*2, 4))
    
    with torch.no_grad():
        for i in range(n_samples):
            # Get sample
            img, _ = dataset[i]
            img_batch = img.unsqueeze(0).to(device)
            
            # Reconstruct
            mu, _ = model.encode(img_batch)
            recon = model.decode(mu)
            
            # Denormalize for visualization
            img_vis = img.cpu().numpy().transpose(1, 2, 0)
            recon_vis = recon[0].cpu().numpy().transpose(1, 2, 0)
            
            # Denormalize both (decoder outputs normalized space, not [0,1])
            img_vis = img_vis * std + mean
            recon_vis = recon_vis * std + mean
            
            # Clip to [0, 1]
            img_vis = np.clip(img_vis, 0, 1)
            recon_vis = np.clip(recon_vis, 0, 1)
            
            # Plot
            axes[0, i].imshow(img_vis)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Original', fontsize=10, fontweight='bold')
            
            axes[1, i].imshow(recon_vis)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Reconstructed', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Reconstructions - Epoch {epoch}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'recon_epoch_{epoch:03d}.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    model.train()


def plot_loss_curves(history, output_path):
    """Plot and save loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['total']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['total'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, history['recon'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss (L1 + SSIM)')
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[2].plot(epochs, history['kl'], 'r-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Loss curves saved to: {output_path}")


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
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--checkpoint-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--recon-dir', type=str, default='reconstructions',
                        help='Directory to save reconstruction visualizations')
    parser.add_argument('--save-every', type=int, default=1,
                        help='Save reconstructions every N epochs')
    
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
    
    # Loss history tracking
    loss_history = {
        'total': [],
        'recon': [],
        'kl': [],
    }
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create validation dataset for reconstruction monitoring (no augmentation)
    val_dataset = TissueDataset(
        csv_path=args.data_csv,
        split='train',
        stain_norm=args.stain_norm,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
        augment=False,  # No augmentation for visualization
    )
    
    for epoch in range(args.epochs):
        # Train
        metrics = train_epoch(
            model, train_loader, optimizer, kl_warmup,
            args.lambda_l1, args.lambda_ssim, device
        )
        
        # Track loss history
        loss_history['total'].append(metrics['loss'])
        loss_history['recon'].append(metrics['recon'])
        loss_history['kl'].append(metrics['kl'])
        
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
                },
                'loss_history': loss_history,
            }, args.output)
            print(" ← Best!")
        else:
            print()
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
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
                },
                'loss_history': loss_history,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved to: {checkpoint_path}")
        
        # Save reconstructions
        if (epoch + 1) % args.save_every == 0:
            save_reconstructions(
                model, val_dataset, epoch+1, args.recon_dir,
                np.array(mean), np.array(std), device
            )
            print(f"  ✓ Reconstructions saved to: {args.recon_dir}/recon_epoch_{epoch+1:03d}.png")
        
        # Plot loss curves
        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            plot_loss_curves(loss_history, f'{args.recon_dir}/loss_curves.png')
        
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

