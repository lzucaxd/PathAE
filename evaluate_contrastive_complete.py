#!/usr/bin/env python3
"""
Complete contrastive learning evaluation:
1. Visualize latent space (train + test sets)
2. Train KNN on contrastive features
3. Compare KNN vs Linear classifier on test set

Usage:
    python evaluate_contrastive_complete.py \
        --contrastive-model checkpoints/contrastive_scratch/contrastive_scratch_best.pt \
        --linear-model checkpoints/linear_eval/contrastive_linear_best.pt
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (average_precision_score, roc_auc_score, 
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, silhouette_score,
                              davies_bouldin_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

from dataset_pcam import PatchCamelyonDataset
from contrastive_model import ContrastiveResNet
from linear_eval import LinearClassifier


def extract_features(model, dataset, device, batch_size=256, num_workers=4):
    """Extract 512-dim encoder features."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            
            # Extract encoder features (512-dim, before projection)
            features = model.encoder(images)  # [B, 512]
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Reshape features from [N, 512, 1, 1] to [N, 512]
    if len(features.shape) == 4:
        features = features.squeeze()
    
    return features, labels


def visualize_latent_space(train_features, train_labels, test_features, test_labels, 
                           output_dir, n_samples=5000):
    """
    Create SEPARATE t-SNE and UMAP visualizations for train and test sets.
    
    Args:
        train_features: [N_train, 512]
        train_labels: [N_train]
        test_features: [N_test, 512]
        test_labels: [N_test]
        output_dir: Directory to save visualizations
        n_samples: Max samples per set to visualize (for speed)
    """
    print(f"\n{'='*70}")
    print("VISUALIZING CONTRASTIVE LATENT SPACE")
    print(f"{'='*70}\n")
    
    # Sample for visualization
    np.random.seed(42)
    
    # Sample train
    n_train = min(n_samples, len(train_features))
    train_idx = np.random.choice(len(train_features), n_train, replace=False)
    train_feat_sample = train_features[train_idx]
    train_lab_sample = train_labels[train_idx]
    
    # Sample test
    n_test = min(n_samples, len(test_features))
    test_idx = np.random.choice(len(test_features), n_test, replace=False)
    test_feat_sample = test_features[test_idx]
    test_lab_sample = test_labels[test_idx]
    
    print(f"  Train samples: {n_train:,}")
    print(f"  Test samples: {n_test:,}")
    
    # Color palette
    label_colors = {0: '#1f77b4', 1: '#ff7f0e'}  # Blue=Normal, Orange=Tumor
    
    # =========================================================================
    # TRAIN SET VISUALIZATION
    # =========================================================================
    print("\n  Computing train t-SNE...")
    train_tsne = TSNE(n_components=2, random_state=42, max_iter=1000, perplexity=30)
    train_tsne_embed = train_tsne.fit_transform(train_feat_sample)
    
    print("  Computing train UMAP...")
    train_umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    train_umap_embed = train_umap.fit_transform(train_feat_sample)
    
    # Compute cluster metrics (train)
    sil_tsne_train = silhouette_score(train_tsne_embed, train_lab_sample)
    sil_umap_train = silhouette_score(train_umap_embed, train_lab_sample)
    db_tsne_train = davies_bouldin_score(train_tsne_embed, train_lab_sample)
    db_umap_train = davies_bouldin_score(train_umap_embed, train_lab_sample)
    
    print(f"\n  Train cluster quality:")
    print(f"    t-SNE: Silhouette={sil_tsne_train:.4f}, Davies-Bouldin={db_tsne_train:.4f}")
    print(f"    UMAP:  Silhouette={sil_umap_train:.4f}, Davies-Bouldin={db_umap_train:.4f}")
    
    # Create train visualization
    fig_train, axes_train = plt.subplots(1, 2, figsize=(16, 7))
    
    # Train t-SNE
    for label in [0, 1]:
        mask = train_lab_sample == label
        axes_train[0].scatter(
            train_tsne_embed[mask, 0], train_tsne_embed[mask, 1],
            c=label_colors[label],
            label=['Normal', 'Tumor'][label],
            s=10, alpha=0.6, edgecolors='none'
        )
    axes_train[0].set_title(f't-SNE\nSilhouette={sil_tsne_train:.3f}, DB={db_tsne_train:.3f}', 
                           fontsize=12, fontweight='bold')
    axes_train[0].legend(markerscale=2)
    axes_train[0].set_xlabel('t-SNE 1')
    axes_train[0].set_ylabel('t-SNE 2')
    axes_train[0].grid(True, alpha=0.3)
    
    # Train UMAP
    for label in [0, 1]:
        mask = train_lab_sample == label
        axes_train[1].scatter(
            train_umap_embed[mask, 0], train_umap_embed[mask, 1],
            c=label_colors[label],
            label=['Normal', 'Tumor'][label],
            s=10, alpha=0.6, edgecolors='none'
        )
    axes_train[1].set_title(f'UMAP\nSilhouette={sil_umap_train:.3f}, DB={db_umap_train:.3f}', 
                           fontsize=12, fontweight='bold')
    axes_train[1].legend(markerscale=2)
    axes_train[1].set_xlabel('UMAP 1')
    axes_train[1].set_ylabel('UMAP 2')
    axes_train[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Contrastive Feature Space - Training Set ({n_train:,} samples)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    train_path = Path(output_dir) / 'contrastive_latent_train.png'
    plt.savefig(train_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Train visualization saved: {train_path}")
    plt.close()
    
    # =========================================================================
    # TEST SET VISUALIZATION
    # =========================================================================
    print("\n  Computing test t-SNE...")
    test_tsne = TSNE(n_components=2, random_state=42, max_iter=1000, perplexity=30)
    test_tsne_embed = test_tsne.fit_transform(test_feat_sample)
    
    print("  Computing test UMAP...")
    test_umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    test_umap_embed = test_umap.fit_transform(test_feat_sample)
    
    # Compute cluster metrics (test)
    sil_tsne_test = silhouette_score(test_tsne_embed, test_lab_sample)
    sil_umap_test = silhouette_score(test_umap_embed, test_lab_sample)
    db_tsne_test = davies_bouldin_score(test_tsne_embed, test_lab_sample)
    db_umap_test = davies_bouldin_score(test_umap_embed, test_lab_sample)
    
    print(f"\n  Test cluster quality:")
    print(f"    t-SNE: Silhouette={sil_tsne_test:.4f}, Davies-Bouldin={db_tsne_test:.4f}")
    print(f"    UMAP:  Silhouette={sil_umap_test:.4f}, Davies-Bouldin={db_umap_test:.4f}")
    
    # Create test visualization
    fig_test, axes_test = plt.subplots(1, 2, figsize=(16, 7))
    
    # Test t-SNE
    for label in [0, 1]:
        mask = test_lab_sample == label
        axes_test[0].scatter(
            test_tsne_embed[mask, 0], test_tsne_embed[mask, 1],
            c=label_colors[label],
            label=['Normal', 'Tumor'][label],
            s=10, alpha=0.6, edgecolors='none'
        )
    axes_test[0].set_title(f't-SNE\nSilhouette={sil_tsne_test:.3f}, DB={db_tsne_test:.3f}', 
                          fontsize=12, fontweight='bold')
    axes_test[0].legend(markerscale=2)
    axes_test[0].set_xlabel('t-SNE 1')
    axes_test[0].set_ylabel('t-SNE 2')
    axes_test[0].grid(True, alpha=0.3)
    
    # Test UMAP
    for label in [0, 1]:
        mask = test_lab_sample == label
        axes_test[1].scatter(
            test_umap_embed[mask, 0], test_umap_embed[mask, 1],
            c=label_colors[label],
            label=['Normal', 'Tumor'][label],
            s=10, alpha=0.6, edgecolors='none'
        )
    axes_test[1].set_title(f'UMAP\nSilhouette={sil_umap_test:.3f}, DB={db_umap_test:.3f}', 
                          fontsize=12, fontweight='bold')
    axes_test[1].legend(markerscale=2)
    axes_test[1].set_xlabel('UMAP 1')
    axes_test[1].set_ylabel('UMAP 2')
    axes_test[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Contrastive Feature Space - Test Set ({n_test:,} samples)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    test_path = Path(output_dir) / 'contrastive_latent_test.png'
    plt.savefig(test_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Test visualization saved: {test_path}")
    plt.close()
    
    return {
        'sil_tsne_train': sil_tsne_train,
        'sil_umap_train': sil_umap_train,
        'db_tsne_train': db_tsne_train,
        'db_umap_train': db_umap_train,
        'sil_tsne_test': sil_tsne_test,
        'sil_umap_test': sil_umap_test,
        'db_tsne_test': db_tsne_test,
        'db_umap_test': db_umap_test
    }


def train_knn(train_features, train_labels, n_neighbors=5):
    """Train KNN classifier on contrastive features."""
    print(f"\n{'='*70}")
    print(f"TRAINING KNN CLASSIFIER (k={n_neighbors})")
    print(f"{'='*70}\n")
    
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights='distance',  # Weight by inverse distance
        metric='cosine',     # Cosine distance (good for embeddings)
        n_jobs=-1            # Use all CPU cores
    )
    
    print(f"  Training on {len(train_features):,} samples...")
    knn.fit(train_features, train_labels)
    print(f"  ✓ KNN trained (k={n_neighbors}, metric=cosine, weights=distance)")
    
    return knn


def evaluate_knn(knn, features, labels):
    """Evaluate KNN classifier."""
    # Predict probabilities
    probs = knn.predict_proba(features)[:, 1]  # P(tumor)
    preds = (probs >= 0.5).astype(int)
    
    # Compute metrics
    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_linear(linear_model, dataset, device, batch_size=128):
    """Evaluate linear classifier."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    linear_model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Linear eval"):
            images = images.to(device)
            logits = linear_model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)
    
    # Compute metrics
    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def create_comparison_table(knn_metrics, linear_metrics, output_path):
    """Create comparison table."""
    data = {
        'Metric': ['PR-AUC', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'KNN (k=5)': [
            knn_metrics['pr_auc'],
            knn_metrics['roc_auc'],
            knn_metrics['accuracy'],
            knn_metrics['precision'],
            knn_metrics['recall'],
            knn_metrics['f1']
        ],
        'Linear': [
            linear_metrics['pr_auc'],
            linear_metrics['roc_auc'],
            linear_metrics['accuracy'],
            linear_metrics['precision'],
            linear_metrics['recall'],
            linear_metrics['f1']
        ]
    }
    
    df = pd.DataFrame(data)
    df['Δ (Linear - KNN)'] = df['Linear'] - df['KNN (k=5)']
    
    # Save CSV
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n  ✓ Comparison table saved: {output_path}")
    
    # Print to console
    print(f"\n{'='*70}")
    print("COMPARISON: KNN vs LINEAR CLASSIFIER")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))
    print("")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Complete contrastive evaluation')
    
    # Models
    parser.add_argument('--contrastive-model', type=str, required=True,
                        help='Path to trained contrastive model')
    parser.add_argument('--linear-model', type=str, required=True,
                        help='Path to trained linear classifier')
    
    # Data
    parser.add_argument('--pcam-dir', type=str, default='PCam',
                        help='Path to PCam directory')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy',
                        help='Reference tile for stain normalization')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy',
                        help='RGB normalization stats')
    
    # KNN
    parser.add_argument('--knn-neighbors', type=int, default=5,
                        help='Number of neighbors for KNN')
    
    # Visualization
    parser.add_argument('--viz-samples', type=int, default=5000,
                        help='Max samples to visualize per set')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/contrastive_evaluation',
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
    print("CONTRASTIVE LEARNING COMPLETE EVALUATION")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    print(f"Contrastive model: {args.contrastive_model}")
    print(f"Linear model: {args.linear_model}")
    print(f"KNN neighbors: {args.knn_neighbors}")
    
    # =========================================================================
    # Load datasets
    # =========================================================================
    print(f"\n{'='*70}")
    print("LOADING DATASETS")
    print(f"{'='*70}\n")
    
    # Load normalization stats
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    # Train set (no augmentation for feature extraction)
    train_dataset = PatchCamelyonDataset(
        data_dir=args.pcam_dir,
        split='train',
        stain_norm=True,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
        augment=False,  # No augmentation for feature extraction
        filter_normal_only=False
    )
    
    # Test set
    test_dataset = PatchCamelyonDataset(
        data_dir=args.pcam_dir,
        split='test',
        stain_norm=True,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
        augment=False,
        filter_normal_only=False
    )
    
    print(f"\nDataset summary:")
    print(f"  Train: {len(train_dataset):,} patches")
    print(f"  Test: {len(test_dataset):,} patches")
    
    # =========================================================================
    # Load contrastive model
    # =========================================================================
    print(f"\n{'='*70}")
    print("LOADING CONTRASTIVE MODEL")
    print(f"{'='*70}\n")
    
    contrastive_model = ContrastiveResNet(pretrained=False).to(device)
    checkpoint = torch.load(args.contrastive_model, map_location=device, weights_only=False)
    contrastive_model.load_state_dict(checkpoint['model_state_dict'])
    contrastive_model.eval()
    
    print(f"  ✓ Contrastive model loaded")
    print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val separation ratio: {checkpoint.get('val_sep_ratio', 'N/A'):.4f}")
    
    # =========================================================================
    # Extract features
    # =========================================================================
    print(f"\n{'='*70}")
    print("EXTRACTING CONTRASTIVE FEATURES")
    print(f"{'='*70}\n")
    
    # Train features
    print("Extracting train features...")
    train_features, train_labels = extract_features(
        contrastive_model, train_dataset, device, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"  ✓ Train features: {train_features.shape}")
    
    # Test features
    print("\nExtracting test features...")
    test_features, test_labels = extract_features(
        contrastive_model, test_dataset, device, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"  ✓ Test features: {test_features.shape}")
    
    # =========================================================================
    # Visualize latent space
    # =========================================================================
    cluster_metrics = visualize_latent_space(
        train_features, train_labels,
        test_features, test_labels,
        output_dir,
        n_samples=args.viz_samples
    )
    
    # =========================================================================
    # Train and evaluate KNN
    # =========================================================================
    knn = train_knn(train_features, train_labels, n_neighbors=args.knn_neighbors)
    
    print(f"\n{'='*70}")
    print("EVALUATING KNN ON TEST SET")
    print(f"{'='*70}\n")
    
    knn_metrics = evaluate_knn(knn, test_features, test_labels)
    
    print(f"  KNN Test Metrics:")
    print(f"    PR-AUC:    {knn_metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC:   {knn_metrics['roc_auc']:.4f}")
    print(f"    Accuracy:  {knn_metrics['accuracy']:.4f}")
    print(f"    Precision: {knn_metrics['precision']:.4f}")
    print(f"    Recall:    {knn_metrics['recall']:.4f}")
    print(f"    F1-Score:  {knn_metrics['f1']:.4f}")
    
    # =========================================================================
    # Evaluate linear classifier
    # =========================================================================
    print(f"\n{'='*70}")
    print("EVALUATING LINEAR CLASSIFIER ON TEST SET")
    print(f"{'='*70}\n")
    
    # Load linear classifier
    linear_checkpoint = torch.load(args.linear_model, map_location=device, weights_only=False)
    linear_model = LinearClassifier(contrastive_model.encoder, feature_dim=512).to(device)
    linear_model.load_state_dict(linear_checkpoint['model_state_dict'])
    
    linear_metrics = evaluate_linear(linear_model, test_dataset, device, batch_size=args.batch_size)
    
    print(f"  Linear Test Metrics:")
    print(f"    PR-AUC:    {linear_metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC:   {linear_metrics['roc_auc']:.4f}")
    print(f"    Accuracy:  {linear_metrics['accuracy']:.4f}")
    print(f"    Precision: {linear_metrics['precision']:.4f}")
    print(f"    Recall:    {linear_metrics['recall']:.4f}")
    print(f"    F1-Score:  {linear_metrics['f1']:.4f}")
    
    # =========================================================================
    # Create comparison table
    # =========================================================================
    comparison_path = output_dir / 'knn_vs_linear_comparison.csv'
    comparison_df = create_comparison_table(knn_metrics, linear_metrics, comparison_path)
    
    # =========================================================================
    # Save cluster metrics
    # =========================================================================
    cluster_df = pd.DataFrame([{
        'Split': 'Train',
        'Method': 't-SNE',
        'Silhouette': cluster_metrics['sil_tsne_train'],
        'Davies-Bouldin': cluster_metrics['db_tsne_train']
    }, {
        'Split': 'Train',
        'Method': 'UMAP',
        'Silhouette': cluster_metrics['sil_umap_train'],
        'Davies-Bouldin': cluster_metrics['db_umap_train']
    }, {
        'Split': 'Test',
        'Method': 't-SNE',
        'Silhouette': cluster_metrics['sil_tsne_test'],
        'Davies-Bouldin': cluster_metrics['db_tsne_test']
    }, {
        'Split': 'Test',
        'Method': 'UMAP',
        'Silhouette': cluster_metrics['sil_umap_test'],
        'Davies-Bouldin': cluster_metrics['db_umap_test']
    }])
    
    cluster_path = output_dir / 'cluster_metrics.csv'
    cluster_df.to_csv(cluster_path, index=False, float_format='%.4f')
    print(f"  ✓ Cluster metrics saved: {cluster_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("✓ CONTRASTIVE EVALUATION COMPLETE")
    print(f"{'='*70}\n")
    
    print(f"Results summary:")
    print(f"  Train latent space viz:     {output_dir / 'contrastive_latent_train.png'}")
    print(f"  Test latent space viz:      {output_dir / 'contrastive_latent_test.png'}")
    print(f"  KNN vs Linear comparison:   {comparison_path}")
    print(f"  Cluster quality metrics:    {cluster_path}")
    print("")
    
    # Determine winner
    if knn_metrics['pr_auc'] > linear_metrics['pr_auc']:
        winner = "KNN"
        improvement = knn_metrics['pr_auc'] - linear_metrics['pr_auc']
    else:
        winner = "Linear"
        improvement = linear_metrics['pr_auc'] - knn_metrics['pr_auc']
    
    print(f"🏆 Winner: {winner} (+{improvement:.4f} PR-AUC)")


if __name__ == '__main__':
    main()

