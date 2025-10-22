#!/usr/bin/env python3
"""
Complete supervised model evaluation:
1. PCam test set metrics
2. Latent space visualization (features from penultimate layer)
3. CAMELYON16 heatmap generation
4. IoU and other metrics
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
                              f1_score, confusion_matrix)
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

from dataset_pcam import PatchCamelyonDataset
from dataset import TestDataset
from models import SimpleResNet18


class FeatureExtractor(nn.Module):
    """Extract features from penultimate layer of ResNet18."""
    
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.backbone.children())[:-1])  # Remove final FC
        self.classifier = model.backbone.fc
    
    def forward(self, x):
        features = self.features(x)
        features = features.flatten(1)  # [B, 512]
        logits = self.classifier(features)
        return features, logits


def evaluate_on_pcam(model, device, pcam_dir, reference_tile, norm_stats):
    """Evaluate on PCam test set and extract features."""
    print("\n" + "="*70)
    print("EVALUATION ON PCAM TEST SET")
    print("="*70)
    
    # Load dataset
    test_dataset = PatchCamelyonDataset(
        data_dir=pcam_dir,
        split='test',
        filter_normal_only=False,
        stain_norm=True,
        reference_tile=reference_tile,
        normalize=True,
        mean=norm_stats['mean'],
        std=norm_stats['std'],
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()
    
    # Run inference
    all_features = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="PCam test inference"):
            images = images.to(device)
            features, logits = feature_extractor(images)
            probs = torch.sigmoid(logits)
            
            all_features.append(features.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    pr_auc = average_precision_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    
    # Threshold at 0.5
    binary_preds = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nPCam Test Results:")
    print(f"  Total patches: {len(all_labels):,}")
    print(f"  Normal: {(all_labels == 0).sum():,}")
    print(f"  Tumor: {(all_labels == 1).sum():,}")
    print(f"\nMetrics:")
    print(f"  PR-AUC:     {pr_auc:.4f}")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:6,}  FP: {fp:6,}")
    print(f"  FN: {fn:6,}  TN: {tn:6,}")
    
    metrics = {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
    
    return all_features, all_labels, all_preds, metrics


def visualize_latent_space(features, labels, predictions, output_dir):
    """Visualize feature space using t-SNE and UMAP."""
    print("\n" + "="*70)
    print("LATENT SPACE VISUALIZATION")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample for visualization (too many points)
    np.random.seed(42)
    n_samples = min(5000, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)
    
    features_sample = features[indices]
    labels_sample = labels[indices]
    preds_sample = predictions[indices]
    
    print(f"\nSampling {n_samples:,} points for visualization...")
    
    # t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, max_iter=1000, perplexity=30)
    tsne_features = tsne.fit_transform(features_sample)
    
    # UMAP
    print("  Running UMAP...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_features = umap_reducer.fit_transform(features_sample)
    
    # Create 2x2 visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # t-SNE colored by true label
    scatter1 = axes[0, 0].scatter(
        tsne_features[:, 0], tsne_features[:, 1],
        c=labels_sample, cmap='coolwarm', s=10, alpha=0.6
    )
    axes[0, 0].set_title('t-SNE: True Labels', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Label (0=Normal, 1=Tumor)')
    
    # t-SNE colored by prediction
    scatter2 = axes[0, 1].scatter(
        tsne_features[:, 0], tsne_features[:, 1],
        c=preds_sample, cmap='viridis', s=10, alpha=0.6
    )
    axes[0, 1].set_title('t-SNE: Predicted Probability', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[0, 1], label='P(Tumor)')
    
    # UMAP colored by true label
    scatter3 = axes[1, 0].scatter(
        umap_features[:, 0], umap_features[:, 1],
        c=labels_sample, cmap='coolwarm', s=10, alpha=0.6
    )
    axes[1, 0].set_title('UMAP: True Labels', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    plt.colorbar(scatter3, ax=axes[1, 0], label='Label (0=Normal, 1=Tumor)')
    
    # UMAP colored by prediction
    scatter4 = axes[1, 1].scatter(
        umap_features[:, 0], umap_features[:, 1],
        c=preds_sample, cmap='viridis', s=10, alpha=0.6
    )
    axes[1, 1].set_title('UMAP: Predicted Probability', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')
    plt.colorbar(scatter4, ax=axes[1, 1], label='P(Tumor)')
    
    plt.suptitle('Supervised Model Feature Space (PCam Test Set)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    viz_path = output_dir / 'supervised_latent_space.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Visualization saved: {viz_path}")
    
    # Compute separation metrics
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    silhouette = silhouette_score(features_sample, labels_sample)
    davies_bouldin = davies_bouldin_score(features_sample, labels_sample)
    calinski = calinski_harabasz_score(features_sample, labels_sample)
    
    print(f"\nSeparation Metrics:")
    print(f"  Silhouette Score:        {silhouette:.4f}  (higher is better, range [-1, 1])")
    print(f"  Davies-Bouldin Index:    {davies_bouldin:.4f}  (lower is better)")
    print(f"  Calinski-Harabasz Score: {calinski:.2f}  (higher is better)")
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski
    }


def run_inference_on_camelyon(model, device, test_csv, reference_tile, 
                                norm_stats, output_csv):
    """Run inference on CAMELYON16 test slides."""
    print("\n" + "="*70)
    print("INFERENCE ON CAMELYON16 TEST SLIDES")
    print("="*70)
    
    # Load test dataset
    test_dataset = TestDataset(
        csv_path=test_csv,
        stain_norm=True,
        reference_tile=reference_tile,
        normalize=True,
        mean=norm_stats['mean'],
        std=norm_stats['std']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nTest slides: {len(test_dataset):,} patches")
    
    # Run inference
    model.eval()
    all_tile_ids = []
    all_probs = []
    
    with torch.no_grad():
        for images, tile_ids in tqdm(test_loader, desc="CAMELYON16 inference"):
            images = images.to(device)
            probs = model.predict_proba(images)
            
            all_tile_ids.extend(tile_ids)
            all_probs.extend(probs.cpu().numpy())
    
    # Save scores
    scores_df = pd.DataFrame({
        'tile_id': all_tile_ids,
        'score': all_probs
    })
    
    scores_df.to_csv(output_csv, index=False)
    print(f"\n  ✓ Scores saved: {output_csv}")
    print(f"  Score range: [{scores_df['score'].min():.4f}, {scores_df['score'].max():.4f}]")
    print(f"  Mean score: {scores_df['score'].mean():.4f}")
    
    return scores_df


def main():
    parser = argparse.ArgumentParser(description='Complete supervised model evaluation')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--pcam-dir', type=str, default='PCam')
    parser.add_argument('--test-csv', type=str, default='test_set_heatmaps/test_set.csv',
                        help='CAMELYON16 test set CSV')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    parser.add_argument('--output-dir', type=str, default='outputs/supervised_evaluation',
                        help='Output directory')
    parser.add_argument('--scores-csv', type=str, default='outputs/supervised_scores.csv',
                        help='Output scores for CAMELYON16')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load normalization stats
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    
    print("="*70)
    print("SUPERVISED MODEL COMPLETE EVALUATION")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    model = SimpleResNet18(pretrained=False).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ Model loaded")
    print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val PR-AUC: {checkpoint.get('val_pr_auc', 'N/A'):.4f}")
    
    # 1. Evaluate on PCam test set and extract features
    features, labels, predictions, pcam_metrics = evaluate_on_pcam(
        model, device, args.pcam_dir, args.reference_tile, norm_stats
    )
    
    # Save PCam metrics
    pcam_metrics_df = pd.DataFrame([pcam_metrics])
    pcam_metrics_path = output_dir / 'pcam_test_metrics.csv'
    pcam_metrics_df.to_csv(pcam_metrics_path, index=False)
    print(f"\n  ✓ PCam metrics saved: {pcam_metrics_path}")
    
    # 2. Visualize latent space
    separation_metrics = visualize_latent_space(features, labels, predictions, output_dir)
    
    # Save separation metrics
    sep_metrics_df = pd.DataFrame([separation_metrics])
    sep_metrics_path = output_dir / 'separation_metrics.csv'
    sep_metrics_df.to_csv(sep_metrics_path, index=False)
    print(f"  ✓ Separation metrics saved: {sep_metrics_path}")
    
    # 3. Run inference on CAMELYON16 slides
    camelyon_scores = run_inference_on_camelyon(
        model, device, args.test_csv, args.reference_tile, norm_stats, args.scores_csv
    )
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    print(f"\nPCam Test Results:")
    print(f"  PR-AUC:    {pcam_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:   {pcam_metrics['roc_auc']:.4f}")
    print(f"  Accuracy:  {pcam_metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {pcam_metrics['f1']:.4f}")
    print(f"\nLatent Space Separation:")
    print(f"  Silhouette: {separation_metrics['silhouette']:.4f}")
    print(f"\nNext step: Generate heatmaps using generate_supervised_heatmaps.py")


if __name__ == '__main__':
    main()

