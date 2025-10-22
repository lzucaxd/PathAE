#!/usr/bin/env python3
"""
Train ResNet18 for supervised tumor detection (from scratch).

Key configuration:
- Model: ResNet18 (random initialization, no pretrained weights)
- Loss: BCEWithLogitsLoss
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Epochs: 20 (fast iteration)
- Early stopping: patience=5
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

from dataset_pcam import PatchCamelyonDataset
from models import SimpleResNet18


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)
        
        # Forward
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    
    # Compute PR-AUC and accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    pr_auc = average_precision_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))
    
    return {
        'loss': avg_loss,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'acc': acc
    }


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.float().to(device)
            
            # Forward
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    pr_auc = average_precision_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))
    
    return {
        'loss': avg_loss,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'acc': acc
    }


def plot_training_curves(history, output_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PR-AUC
    axes[1].plot(epochs, history['train_pr_auc'], 'b-', linewidth=2, label='Train')
    axes[1].plot(epochs, history['val_pr_auc'], 'r-', linewidth=2, label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PR-AUC')
    axes[1].set_title('Precision-Recall AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Accuracy
    axes[2].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train')
    axes[2].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 for tumor detection')
    
    # Data
    parser.add_argument('--pcam-dir', type=str, default='PCam')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    
    # Model
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Use ImageNet pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints/supervised_scratch')
    
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
    print("SUPERVISED RESNET18 TRAINING (from scratch)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Early stopping patience: {args.patience}")
    
    # Load datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    train_dataset = PatchCamelyonDataset(
        data_dir=args.pcam_dir,
        split='train',
        filter_normal_only=False,  # Include tumor!
        stain_norm=True,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
        augment=True  # Augmentations for training
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
        augment=False  # No augmentation for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
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
    
    model = SimpleResNet18(pretrained=args.pretrained).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
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
        'train_pr_auc': [],
        'train_roc_auc': [],
        'train_acc': [],
        'val_loss': [],
        'val_pr_auc': [],
        'val_roc_auc': [],
        'val_acc': [],
    }
    
    best_val_pr_auc = -1
    patience_counter = 0
    
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
        history['train_pr_auc'].append(train_metrics['pr_auc'])
        history['train_roc_auc'].append(train_metrics['roc_auc'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_pr_auc'].append(val_metrics['pr_auc'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['val_acc'].append(val_metrics['acc'])
        
        # Print metrics
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | PR-AUC: {train_metrics['pr_auc']:.4f} | Acc: {train_metrics['acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f} | Acc: {val_metrics['acc']:.4f}")
        
        # Save best model (by val PR-AUC)
        if val_metrics['pr_auc'] > best_val_pr_auc:
            best_val_pr_auc = val_metrics['pr_auc']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_pr_auc': val_metrics['pr_auc'],
                'val_roc_auc': val_metrics['roc_auc'],
                'val_acc': val_metrics['acc'],
            }
            
            best_path = output_dir / 'supervised_scratch_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved (Val PR-AUC: {best_val_pr_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{args.patience})")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break
        
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
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest model:")
    print(f"  Val PR-AUC: {best_val_pr_auc:.4f}")
    print(f"  Saved to: {output_dir}/supervised_scratch_best.pt")
    print(f"\nOutputs:")
    print(f"  • Best model: {output_dir}/supervised_scratch_best.pt")
    print(f"  • Training curves: {plot_path}")
    print(f"  • Checkpoints: {output_dir}/checkpoint_epoch_*.pt")


if __name__ == '__main__':
    main()


