#!/usr/bin/env python3
"""
Linear evaluation of contrastive embeddings.

Test embedding quality by training a single linear layer on frozen features.
If embeddings are good, a simple linear classifier should achieve high accuracy.
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
from contrastive_model import ContrastiveResNet


class LinearClassifier(nn.Module):
    """Simple linear classifier on top of frozen encoder."""
    
    def __init__(self, encoder, feature_dim=512):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, 1)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Initialize classifier
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
            features = features.flatten(1)
        logits = self.classifier(features)
        return logits.squeeze(1)


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
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    
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
    """Plot linear evaluation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('BCE Loss', fontsize=12)
    axes[0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PR-AUC
    axes[1].plot(epochs, history['train_pr_auc'], 'b-', linewidth=2, label='Train')
    axes[1].plot(epochs, history['val_pr_auc'], 'r-', linewidth=2, label='Val')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('PR-AUC', fontsize=12)
    axes[1].set_title('Linear Evaluation: PR-AUC', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Linear evaluation of contrastive embeddings')
    
    # Data
    parser.add_argument('--pcam-dir', type=str, default='PCam')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    
    # Model
    parser.add_argument('--contrastive-model', type=str, required=True,
                        help='Path to trained contrastive model')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints/contrastive_linear')
    
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
    print("LINEAR EVALUATION OF CONTRASTIVE EMBEDDINGS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Contrastive model: {args.contrastive_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    # Load datasets (NO augmentation for linear eval)
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
        augment=False  # NO augmentation for linear eval
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
        pin_memory=True
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
    
    # Load contrastive model
    print("\n" + "="*70)
    print("LOADING CONTRASTIVE MODEL")
    print("="*70)
    
    contrastive_model = ContrastiveResNet(pretrained=False).to(device)
    checkpoint = torch.load(args.contrastive_model, map_location=device, weights_only=False)
    contrastive_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  ✓ Contrastive model loaded")
    print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val separation ratio: {checkpoint.get('val_sep_ratio', 'N/A'):.4f}")
    
    # Create linear classifier
    model = LinearClassifier(contrastive_model.encoder, feature_dim=512).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"\n  Trainable parameters: {trainable_params:,} (linear layer only)")
    print(f"  Frozen parameters: {frozen_params:,} (encoder)")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),  # Only optimize classifier
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING LINEAR CLASSIFIER")
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
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
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
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | PR-AUC: {train_metrics['pr_auc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}")
        
        # Save best model
        if val_metrics['pr_auc'] > best_val_pr_auc:
            best_val_pr_auc = val_metrics['pr_auc']
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': model.classifier.state_dict(),
                'val_pr_auc': val_metrics['pr_auc'],
                'val_roc_auc': val_metrics['roc_auc'],
                'val_acc': val_metrics['acc'],
            }
            
            best_path = output_dir / 'contrastive_linear_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved (Val PR-AUC: {best_val_pr_auc:.4f})")
    
    # Plot curves
    print("\n" + "="*70)
    print("SAVING TRAINING CURVES")
    print("="*70)
    
    plot_path = output_dir / 'linear_eval_curves.png'
    plot_training_curves(history, plot_path)
    print(f"  ✓ Curves saved: {plot_path}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ LINEAR EVALUATION COMPLETE")
    print("="*70)
    print(f"\nBest linear classifier:")
    print(f"  Val PR-AUC: {best_val_pr_auc:.4f}")
    print(f"  Saved to: {output_dir}/contrastive_linear_best.pt")


if __name__ == '__main__':
    main()

