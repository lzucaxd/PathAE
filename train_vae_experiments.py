#!/usr/bin/env python3
"""
Unified training script for all VAE experiments.

Supports:
- VAE-Skip96 (U-Net style with skip connections)
- VAE-ResNet18 (pretrained encoder, TODO)
- VAE-P4M (rotation-equivariant, TODO)
- Optional denoising (Gaussian noise σ=0.03)
- β-VAE loss with KL warm-up
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from dataset import TissueDataset
from model_vae_skip import VAESkip96, KLCapacity
from model_vae_pure import VAEPure96
from model_vae_freebits import VAEFreeBits96, free_bits_vae_loss, KLWarmup
from model_ae import AE96, ae_loss
from model_vae_resnet import VAEResNet96


def add_noise(x, sigma=0.03):
    """Add Gaussian noise for denoising VAE."""
    noise = torch.randn_like(x) * sigma
    return x + noise


def save_reconstructions(model, dataset, epoch, output_dir, mean, std, device, n_samples=8):
    """Save reconstruction visualizations."""
    model.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*2, 4))
    
    # Convert mean/std to numpy arrays for broadcasting
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)
    
    with torch.no_grad():
        for i in range(n_samples):
            # Get sample
            img, _ = dataset[i]
            img_batch = img.unsqueeze(0).to(device)
            
            # Reconstruct (no use_mean parameter for FreeBits VAE)
            recon, mu, logvar = model(img_batch)
            
            # Move to CPU and transpose to HWC format
            img_vis = img.cpu().numpy().transpose(1, 2, 0)  # Already normalized
            recon_vis = recon[0].cpu().numpy().transpose(1, 2, 0)  # Model outputs normalized
            
            # Denormalize both: (x * std) + mean
            img_vis = img_vis * std_np + mean_np
            recon_vis = recon_vis * std_np + mean_np
            
            # Clip to [0, 1] for display
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    model.train()


def plot_loss_curves(history, output_path):
    """Plot and save train/val loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_total']) + 1)
    
    # Total loss (train vs val)
    axes[0].plot(epochs, history['train_total'], 'b-', linewidth=2, label='Train')
    axes[0].plot(epochs, history['val_total'], 'r-', linewidth=2, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss (train vs val)
    axes[1].plot(epochs, history['train_recon'], 'b-', linewidth=2, label='Train')
    axes[1].plot(epochs, history['val_recon'], 'r-', linewidth=2, label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss (L1 + SSIM)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence (train vs val)
    axes[2].plot(epochs, history['train_kl'], 'b-', linewidth=2, label='Train')
    axes[2].plot(epochs, history['val_kl'], 'r-', linewidth=2, label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence (nats)')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_epoch(model, loader, optimizer, kl_scheduler, capacity_scheduler, mean, std, 
                lambda_l1, lambda_ssim, device, denoise_sigma=0.0, use_freebits=False, free_bits=0.5, use_ae=False):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    
    beta = kl_scheduler.get_beta() if not use_ae else 0.0
    capacity = capacity_scheduler.get_capacity() if not use_ae else 0.0
    
    # Simplify progress bar
    if use_ae:
        desc = "Training (AE - no KL)"
    elif capacity > 0:
        desc = f"Training (β={beta:.2f}, C={capacity:.1f})"
    else:
        desc = f"Training (β={beta:.2f})"
    
    for batch_idx, (images, _) in enumerate(tqdm(loader, desc=desc, leave=False)):
        images = images.to(device)
        
        # Optional denoising (add noise to input, reconstruct clean)
        if denoise_sigma > 0:
            images_noisy = add_noise(images, sigma=denoise_sigma)
            recon, mu, logvar = model(images_noisy)
            # Loss compares recon to CLEAN images
        else:
            recon, mu, logvar = model(images)
        
        # Loss (AE, free-bits, or regular VAE)
        if use_ae:
            loss, loss_dict = ae_loss(
                recon, images,
                mean=mean,
                std=std,
                lambda_l1=lambda_l1,
                lambda_ssim=lambda_ssim
            )
        elif use_freebits:
            loss, loss_dict = free_bits_vae_loss(
                recon, images, mu, logvar,
                mean=mean,
                std=std,
                lambda_l1=lambda_l1,
                lambda_ssim=lambda_ssim,
                beta=beta,
                free_bits=free_bits
            )
        else:
            from model_vae_pure import vae_loss
            loss, loss_dict = vae_loss(
                recon, images, mu, logvar,
                mean=mean,
                std=std,
                lambda_l1=lambda_l1,
                lambda_ssim=lambda_ssim,
                beta=beta,
                capacity=capacity
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


def validate_epoch(model, loader, capacity_scheduler, mean, std, lambda_l1, lambda_ssim, beta, device, 
                   use_freebits=False, free_bits=0.5, use_ae=False):
    """Validate for one epoch (no gradient, no noise)."""
    model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    
    capacity = capacity_scheduler.get_capacity() if not use_ae else 0.0
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(loader, desc=f"Validation", leave=False)):
            images = images.to(device)
            
            # Forward (no noise)
            if use_ae:
                recon, mu, logvar = model(images)  # AE is deterministic
            else:
                recon, mu, logvar = model(images, use_mean=False)  # VAE still samples for proper KL
            
            # Loss (AE, free-bits, or regular VAE)
            if use_ae:
                loss, loss_dict = ae_loss(
                    recon, images,
                    mean=mean,
                    std=std,
                    lambda_l1=lambda_l1,
                    lambda_ssim=lambda_ssim
                )
            elif use_freebits:
                loss, loss_dict = free_bits_vae_loss(
                    recon, images, mu, logvar,
                    mean=mean,
                    std=std,
                    lambda_l1=lambda_l1,
                    lambda_ssim=lambda_ssim,
                    beta=beta,
                    free_bits=free_bits
                )
            else:
                from model_vae_pure import vae_loss
                loss, loss_dict = vae_loss(
                    recon, images, mu, logvar,
                    mean=mean,
                    std=std,
                    lambda_l1=lambda_l1,
                    lambda_ssim=lambda_ssim,
                    beta=beta,
                    capacity=capacity
                )
            
            total_loss += loss_dict['total']
            total_recon += loss_dict['recon']
            total_kl += loss_dict['kl']
    
    return {
        'loss': total_loss / len(loader),
        'recon': total_recon / len(loader),
        'kl': total_kl / len(loader),
    }


def main():
    parser = argparse.ArgumentParser(description='Train VAE for experiments')
    
    # Experiment
    parser.add_argument('--exp-id', type=str, required=True,
                        help='Experiment ID (B1, B2, A1, A2, P1, P2)')
    parser.add_argument(        '--model', type=str, default='skip96',
                        choices=['skip96', 'pure', 'freebits', 'ae', 'resnet', 'p4m'])
    
    # Data
    parser.add_argument('--data-csv', type=str, default='final_dataset/dataset.csv')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    
    # Model
    parser.add_argument('--z-dim', type=int, default=128)
    parser.add_argument('--num-groups', type=int, default=8)
    parser.add_argument('--skip-dropout', type=float, default=0.25,
                        help='Dropout rate for skip connections (higher = more regularization)')
    
    # Loss
    parser.add_argument('--lambda-l1', type=float, default=0.6)
    parser.add_argument('--lambda-ssim', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=1.0, help='KL weight (use 1.0 with capacity)')
    parser.add_argument('--beta-min', type=float, default=0.1, help='Starting β value (0.1-0.3 prevents latent bypass)')
    parser.add_argument('--kl-warmup', type=int, default=5, help='β warm-up epochs (beta_min→beta_max)')
    parser.add_argument('--capacity-max', type=float, default=120.0,
                        help='Max KL capacity in nats (prevents collapse)')
    parser.add_argument('--free-bits', type=float, default=0.5,
                        help='Minimum KL per dimension (nats) for free-bits VAE')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Preprocessing
    parser.add_argument('--stain-norm', action='store_true', default=True)
    parser.add_argument('--no-stain-norm', action='store_false', dest='stain_norm')
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', action='store_false', dest='augment')
    
    # Denoising VAE
    parser.add_argument('--denoise', action='store_true', default=False)
    parser.add_argument('--noise-sigma', type=float, default=0.03)
    
    # Output
    parser.add_argument('--output', type=str, default='vae_best.pth')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint-every', type=int, default=5)
    parser.add_argument('--recon-dir', type=str, default='reconstructions')
    parser.add_argument('--save-every', type=int, default=1)
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print(f"β-VAE TRAINING: {args.exp_id}")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Model: {args.model}")
    print(f"  z_dim: {args.z_dim}")
    print(f"  skip_dropout: {args.skip_dropout}")
    print(f"  β: {args.beta} (warmup over {args.kl_warmup} epochs)")
    print(f"  λ_L1: {args.lambda_l1}, λ_SSIM: {args.lambda_ssim}")
    print(f"  Denoise: {args.denoise} (σ={args.noise_sigma if args.denoise else 0})")
    print(f"  Stain norm: {args.stain_norm}")
    print(f"  Augment: {args.augment}")
    print()
    
    # Load normalization stats
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    print("Loaded normalization stats:")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")
    print()
    
    # Dataset and loader
    print("Loading dataset...")
    
    # Load full training dataset (no augmentation yet)
    full_dataset = TissueDataset(
        csv_path=args.data_csv,
        split='train',
        stain_norm=args.stain_norm,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
        augment=False,  # Will enable for train split only
    )
    
    # Split into train (85%) and validation (15%)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Enable augmentation for training set only (modifying underlying dataset)
    if args.augment:
        import albumentations as A
        full_dataset.augment = True
        full_dataset.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=5, val_shift_limit=0, p=0.3),
        ])
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # For stable GroupNorm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"Dataset initialized:")
    print(f"  Total samples: {len(full_dataset):,}")
    print(f"  Train samples: {len(train_dataset):,} (85%)")
    print(f"  Val samples: {len(val_dataset):,} (15%)")
    print(f"  Stain norm: {args.stain_norm}")
    print(f"  RGB norm: True")
    print(f"  Augment (train only): {args.augment}")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")
    print()
    
    # Model
    print(f"Building {args.model}...")
    if args.model == 'skip96':
        model = VAESkip96(z_ch=args.z_dim, num_groups=args.num_groups, skip_dropout=args.skip_dropout).to(device)
    elif args.model == 'pure':
        model = VAEPure96(z_ch=args.z_dim, num_groups=args.num_groups).to(device)
    elif args.model == 'freebits':
        model = VAEFreeBits96(z_ch=args.z_dim, num_groups=args.num_groups).to(device)
        print(f"  Free-bits constraint: {args.free_bits} nats/dim")
    elif args.model == 'ae':
        model = AE96(z_dim=args.z_dim, num_groups=args.num_groups).to(device)
        print(f"  Standard AE (no KL, no collapse)")
    elif args.model == 'resnet':
        model = VAEResNet96(z_ch=args.z_dim, num_groups=args.num_groups).to(device)
        print(f"  ResNet-based VAE (trained from scratch)")
    elif args.model == 'p4m':
        raise NotImplementedError("P4M VAE not implemented yet")
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print()
    
    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    
    # KL warm-up (β: beta_min→beta_max over warmup epochs)
    kl_warmup = KLWarmup(beta_max=args.beta, warmup_epochs=args.kl_warmup, beta_min=args.beta_min)
    
    # KL capacity scheduling (C: 0→120 nats over 20 epochs)
    capacity_scheduler = KLCapacity(capacity_max=args.capacity_max, warmup_epochs=args.epochs)
    
    print("="*70)
    print("TRAINING")
    print("="*70)
    if args.model != 'ae' and args.kl_warmup > 0:
        print(f"\nNote: Warm-up active for first {args.kl_warmup} epochs (β: 0→{args.beta})")
        print(f"      Best model tracking starts at epoch {args.kl_warmup+1}\n")
    elif args.model == 'ae':
        print(f"\nNote: Standard AE (no KL divergence, no collapse)\n")
    print()
    
    best_loss = float('inf')
    
    # Loss history tracking
    loss_history = {
        'train_total': [],
        'train_recon': [],
        'train_kl': [],
        'val_total': [],
        'val_recon': [],
        'val_kl': [],
        'capacity': [],
    }
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Validation dataset (no augmentation, for reconstruction monitoring)
    val_dataset = TissueDataset(
        csv_path=args.data_csv,
        split='train',
        stain_norm=args.stain_norm,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
        augment=False,
    )
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        # Determine model type
        use_freebits = (args.model == 'freebits')
        use_ae = (args.model == 'ae')
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, kl_warmup, capacity_scheduler,
            mean, std, args.lambda_l1, args.lambda_ssim, device,
            denoise_sigma=args.noise_sigma if args.denoise else 0.0,
            use_freebits=use_freebits,
            free_bits=args.free_bits,
            use_ae=use_ae
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, capacity_scheduler,
            mean, std, args.lambda_l1, args.lambda_ssim,
            kl_warmup.get_beta(), device,
            use_freebits=use_freebits,
            free_bits=args.free_bits,
            use_ae=use_ae
        )
        
        # Track history
        loss_history['train_total'].append(train_metrics['loss'])
        loss_history['train_recon'].append(train_metrics['recon'])
        loss_history['train_kl'].append(train_metrics['kl'])
        loss_history['val_total'].append(val_metrics['loss'])
        loss_history['val_recon'].append(val_metrics['recon'])
        loss_history['val_kl'].append(val_metrics['kl'])
        loss_history['capacity'].append(capacity_scheduler.get_capacity())
        
        # Print progress
        # Print epoch progress (different format for AE vs VAE)
        if use_ae:
            # AE: No KL or β
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}", end="")
        else:
            # VAE: Show KL and β (hide C when not used)
            capacity_current = capacity_scheduler.get_capacity()
            if capacity_current > 0:
                print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"KL: {train_metrics['kl']:.2f}/{val_metrics['kl']:.2f} | "
                      f"β: {train_metrics['beta']:.2f} | "
                      f"C: {capacity_current:.1f}", end="")
            else:
                print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"KL: {train_metrics['kl']:.2f}/{val_metrics['kl']:.2f} | "
                      f"β: {train_metrics['beta']:.2f}", end="")
        
        # Learning rate scheduler
        scheduler.step()
        
        # Save best model (based on validation loss)
        # For VAE: Only consider epochs AFTER warm-up completes (when β reaches target)
        # For AE: Track from epoch 1 (no warm-up needed)
        if use_ae:
            track_best = True
        else:
            track_best = (epoch >= args.kl_warmup)
        
        if val_metrics['loss'] < best_loss and track_best:
            best_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'config': {
                    'exp_id': args.exp_id,
                    'model': args.model,
                    'z_dim': args.z_dim,
                    'beta': args.beta,
                    'lambda_l1': args.lambda_l1,
                    'lambda_ssim': args.lambda_ssim,
                    'num_groups': args.num_groups,
                    'denoise': args.denoise,
                    'noise_sigma': args.noise_sigma if args.denoise else 0.0,
                    'mean': mean,
                    'std': std,
                },
                'loss_history': loss_history,
            }, args.output)
            if use_ae or epoch >= args.kl_warmup:
                print(" ← Best!")
            else:
                print(" (warm-up, not tracked)")
        else:
            print()
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'config': {
                    'exp_id': args.exp_id,
                    'model': args.model,
                    'z_dim': args.z_dim,
                    'beta': args.beta,
                    'lambda_l1': args.lambda_l1,
                    'lambda_ssim': args.lambda_ssim,
                    'num_groups': args.num_groups,
                    'denoise': args.denoise,
                    'noise_sigma': args.noise_sigma if args.denoise else 0.0,
                    'mean': mean,
                    'std': std,
                },
                'loss_history': loss_history,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint: {checkpoint_path}")
        
        # Save reconstructions
        if (epoch + 1) % args.save_every == 0:
            save_reconstructions(
                model, val_dataset, epoch+1, args.recon_dir,
                np.array(mean), np.array(std), device
            )
            print(f"  ✓ Reconstructions: {args.recon_dir}/recon_epoch_{epoch+1:03d}.png")
        
        # Plot loss curves
        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            plot_loss_curves(loss_history, f'{args.recon_dir}/loss_curves.png')
            print(f"  ✓ Loss curves: {args.recon_dir}/loss_curves.png")
        
        # Step schedulers EVERY epoch (not just when saving checkpoints!)
        if not use_ae:
            kl_warmup.step()
            capacity_scheduler.step()
    
    print()
    print("="*70)
    print(f"TRAINING COMPLETE: {args.exp_id}")
    print("="*70)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Model saved to: {args.output}")
    print()


if __name__ == '__main__':
    main()

