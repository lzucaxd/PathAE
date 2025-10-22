#!/usr/bin/env python3
"""
Train ContrastiveResNet using supervised contrastive learning.

Key configuration:
- Model: ResNet18 + projection head (from scratch)
- Loss: Supervised Contrastive Loss
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Epochs: 20
- Batch size: 256 (large batches critical for contrastive)
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_pcam import PatchCamelyonDataset
from contrastive_model import ContrastiveResNet
from supcon_loss import SupConLoss


def compute_class_distances(features, labels):
    """
    Compute intra-class and inter-class distances.
    
    Args:
        features: [B, D] feature vectors
        labels: [B] class labels
    
    Returns:
        intra_dist: Mean distance within same class
        inter_dist: Mean distance between different classes
        separation_ratio: inter_dist / intra_dist
    """
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(features, features, metric='euclidean')
    
    # Intra-class distances (same label, excluding diagonal)
    intra_dists = []
    for i in range(len(labels)):
        same_class = (labels == labels[i]) & (np.arange(len(labels)) != i)
        if same_class.sum() > 0:
            intra_dists.extend(dist_matrix[i, same_class])
    
    # Inter-class distances (different label)
    inter_dists = []
    for i in range(len(labels)):
        diff_class = labels != labels[i]
        if diff_class.sum() > 0:
            inter_dists.extend(dist_matrix[i, diff_class])
    
    intra_dist = np.mean(intra_dists) if len(intra_dists) > 0 else 0.0
    inter_dist = np.mean(inter_dists) if len(inter_dists) > 0 else 0.0
    separation_ratio = inter_dist / (intra_dist + 1e-12)
    
    return intra_dist, inter_dist, separation_ratio


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_features = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward (get projections)
        projections = model(images, return_features=False)
        
        # Compute contrastive loss
        loss = criterion(projections, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Collect features for distance computation (every 10 batches to save memory)
        if len(all_features) < 50:  # Limit to ~12,800 samples
            with torch.no_grad():
                features = model(images, return_features=True)
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
    
    avg_loss = total_loss / len(loader)
    
    # Compute class separation
    if len(all_features) > 0:
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        intra_dist, inter_dist, sep_ratio = compute_class_distances(all_features, all_labels)
    else:
        intra_dist, inter_dist, sep_ratio = 0.0, 0.0, 0.0
    
    return {
        'loss': avg_loss,
        'intra_dist': intra_dist,
        'inter_dist': inter_dist,
        'separation_ratio': sep_ratio
    }


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            projections = model(images, return_features=False)
            loss = criterion(projections, labels)
            
            # Track metrics
            total_loss += loss.item()
            
            # Collect features
            features = model(images, return_features=True)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    
    avg_loss = total_loss / len(loader)
    
    # Compute class separation
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    intra_dist, inter_dist, sep_ratio = compute_class_distances(all_features, all_labels)
    
    return {
        'loss': avg_loss,
        'intra_dist': intra_dist,
        'inter_dist': inter_dist,
        'separation_ratio': sep_ratio
    }


def plot_training_curves(history, output_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Contrastive Loss', fontsize=12)
    axes[0].set_title('Contrastive Loss', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Separation ratio
    axes[1].plot(epochs, history['train_sep_ratio'], 'b-', linewidth=2, label='Train')
    axes[1].plot(epochs, history['val_sep_ratio'], 'r-', linewidth=2, label='Val')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Separation Ratio', fontsize=12)
    axes[1].set_title('Separation Ratio (Inter/Intra Distance)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Intra vs Inter distance
    axes[2].plot(epochs, history['train_intra'], 'b--', linewidth=2, label='Train Intra', alpha=0.7)
    axes[2].plot(epochs, history['val_intra'], 'r--', linewidth=2, label='Val Intra', alpha=0.7)
    axes[2].plot(epochs, history['train_inter'], 'b-', linewidth=2, label='Train Inter')
    axes[2].plot(epochs, history['val_inter'], 'r-', linewidth=2, label='Val Inter')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Distance', fontsize=12)
    axes[2].set_title('Intra-class vs Inter-class Distance', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train contrastive ResNet18')
    
    # Data
    parser.add_argument('--pcam-dir', type=str, default='PCam')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    
    # Model
    parser.add_argument('--pretrained', action='store_true', default=False)
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints/contrastive_scratch')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load normalization stats
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    print("="*70)
    print("SUPERVISED CONTRASTIVE LEARNING (ResNet18 from scratch)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temperature: {args.temperature}")
    
    # Load datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    train_dataset = PatchCamelyonDataset(
        data_dir=args.pcam_dir,
        split='train',
        filter_normal_only=False,
        stain_norm=True,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
        augment=True  # Use biologically valid augmentations
    )
    
    val_dataset = PatchCamelyonDataset(
        data_dir=args.pcam_dir,
        split='valid',
        filter_normal_only=False,
        stain_norm=True,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Important for contrastive learning
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset summary:")
    print(f"  Train: {len(train_dataset):,} patches")
    print(f"  Val: {len(val_dataset):,} patches")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")
    
    # Model
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    model = ContrastiveResNet(pretrained=args.pretrained).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    projection_params = sum(p.numel() for p in model.projection_head.parameters())
    
    print(f"  Total parameters: {total_params:,}")
    print(f"    Encoder: {encoder_params:,}")
    print(f"    Projection head: {projection_params:,}")
    
    # Loss and optimizer
    criterion = SupConLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    history = {
        'train_loss': [],
        'train_intra': [],
        'train_inter': [],
        'train_sep_ratio': [],
        'val_loss': [],
        'val_intra': [],
        'val_inter': [],
        'val_sep_ratio': [],
    }
    
    best_sep_ratio = -1
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_intra'].append(train_metrics['intra_dist'])
        history['train_inter'].append(train_metrics['inter_dist'])
        history['train_sep_ratio'].append(train_metrics['separation_ratio'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_intra'].append(val_metrics['intra_dist'])
        history['val_inter'].append(val_metrics['inter_dist'])
        history['val_sep_ratio'].append(val_metrics['separation_ratio'])
        
        # Print metrics
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Sep: {train_metrics['separation_ratio']:.4f} | "
              f"Intra: {train_metrics['intra_dist']:.4f} | "
              f"Inter: {train_metrics['inter_dist']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Sep: {val_metrics['separation_ratio']:.4f} | "
              f"Intra: {val_metrics['intra_dist']:.4f} | "
              f"Inter: {val_metrics['inter_dist']:.4f}")
        
        # Save best model (by validation separation ratio)
        if val_metrics['separation_ratio'] > best_sep_ratio:
            best_sep_ratio = val_metrics['separation_ratio']
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_sep_ratio': val_metrics['separation_ratio'],
                'val_loss': val_metrics['loss'],
            }
            
            best_path = output_dir / 'contrastive_scratch_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved (Val Sep Ratio: {best_sep_ratio:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    # Plot training curves
    print("\n" + "="*70)
    print("SAVING TRAINING CURVES")
    print("="*70)
    
    plot_path = output_dir / 'training_curves.png'
    plot_training_curves(history, plot_path)
    print(f"  ✓ Training curves saved: {plot_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("✓ CONTRASTIVE TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest model:")
    print(f"  Val Separation Ratio: {best_sep_ratio:.4f}")
    print(f"  Saved to: {output_dir}/contrastive_scratch_best.pt")
    print(f"\nNext step: Run linear evaluation to test embedding quality")


if __name__ == '__main__':
    main()


