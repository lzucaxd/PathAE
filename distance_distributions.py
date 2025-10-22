#!/usr/bin/env python3
"""
Distance distribution analysis for supervised and contrastive embeddings.

Computes pairwise distances and separates into:
- Intra-class: distances within same class (normal-normal, tumor-tumor)
- Inter-class: distances between classes (normal-tumor)

Visualizes distributions for both embedding types to compare separation quality.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

from dataset_pcam import PatchCamelyonDataset
from models import SimpleResNet18
from contrastive_model import ContrastiveResNet


def extract_features_supervised(model, dataset, device, batch_size=256, num_workers=4):
    """Extract features from supervised model (penultimate layer)."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get penultimate layer features
    model.eval()
    all_features = []
    all_labels = []
    
    # Hook to extract features
    features_hook = []
    
    def hook_fn(module, input, output):
        features_hook.append(output)
    
    # Register hook on avgpool layer (before final FC)
    handle = model.backbone.avgpool.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting supervised features"):
            images = images.to(device)
            _ = model(images)
            
            # Get features from hook
            feats = features_hook[-1].squeeze()  # [B, 512]
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            features_hook.clear()
    
    handle.remove()
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Ensure 2D
    if len(features.shape) > 2:
        features = features.reshape(len(features), -1)
    
    return features, labels


def extract_features_contrastive(model, dataset, device, batch_size=256, num_workers=4):
    """Extract features from contrastive model (encoder output)."""
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
        for images, labels in tqdm(loader, desc="Extracting contrastive features"):
            images = images.to(device)
            
            # Extract encoder features (512-dim)
            features = model.encoder(images)  # [B, 512, 1, 1] or [B, 512]
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Ensure 2D: [N, 512]
    if len(features.shape) == 4:
        features = features.squeeze()
    elif len(features.shape) > 2:
        features = features.reshape(len(features), -1)
    
    return features, labels


def compute_distance_distributions(features, labels, n_samples=5000):
    """
    Compute pairwise distance distributions.
    
    Args:
        features: [N, D] feature vectors
        labels: [N] binary labels (0=normal, 1=tumor)
        n_samples: Max samples to use (for computational efficiency)
    
    Returns:
        dict with intra_normal, intra_tumor, inter_class distances
    """
    print(f"\n  Computing distance distributions...")
    
    # Sample for efficiency
    np.random.seed(42)
    if len(features) > n_samples:
        idx = np.random.choice(len(features), n_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
    
    print(f"    Using {len(features):,} samples")
    
    # Get class masks
    normal_mask = labels == 0
    tumor_mask = labels == 1
    
    normal_features = features[normal_mask]
    tumor_features = features[tumor_mask]
    
    print(f"    Normal: {len(normal_features):,}, Tumor: {len(tumor_features):,}")
    
    # Compute pairwise distances (Euclidean)
    print("    Computing intra-class distances (normal)...")
    if len(normal_features) > 1:
        intra_normal = pdist(normal_features, metric='euclidean')
    else:
        intra_normal = np.array([])
    
    print("    Computing intra-class distances (tumor)...")
    if len(tumor_features) > 1:
        intra_tumor = pdist(tumor_features, metric='euclidean')
    else:
        intra_tumor = np.array([])
    
    # Inter-class distances (normal to tumor)
    print("    Computing inter-class distances...")
    inter_distances = []
    # Sample pairs for efficiency
    n_pairs = min(10000, len(normal_features) * len(tumor_features))
    for _ in range(n_pairs):
        i = np.random.randint(0, len(normal_features))
        j = np.random.randint(0, len(tumor_features))
        dist = np.linalg.norm(normal_features[i] - tumor_features[j])
        inter_distances.append(dist)
    inter_class = np.array(inter_distances)
    
    print(f"    ✓ Intra-normal: {len(intra_normal):,} distances")
    print(f"    ✓ Intra-tumor: {len(intra_tumor):,} distances")
    print(f"    ✓ Inter-class: {len(inter_class):,} distances")
    
    return {
        'intra_normal': intra_normal,
        'intra_tumor': intra_tumor,
        'inter_class': inter_class
    }


def plot_distance_distributions(sup_dists, con_dists, output_path):
    """
    Create comparison plots of distance distributions.
    
    Args:
        sup_dists: Supervised distances (dict)
        con_dists: Contrastive distances (dict)
        output_path: Where to save plot
    """
    print(f"\n  Creating distance distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color palette
    colors = {
        'intra_normal': '#1f77b4',  # Blue
        'intra_tumor': '#ff7f0e',   # Orange
        'inter_class': '#2ca02c'    # Green
    }
    
    # =========================================================================
    # Supervised embeddings
    # =========================================================================
    ax_sup = axes[0, 0]
    
    # Plot distributions
    ax_sup.hist(sup_dists['intra_normal'], bins=50, alpha=0.6, 
               color=colors['intra_normal'], label='Intra-class (Normal)', density=True)
    ax_sup.hist(sup_dists['intra_tumor'], bins=50, alpha=0.6, 
               color=colors['intra_tumor'], label='Intra-class (Tumor)', density=True)
    ax_sup.hist(sup_dists['inter_class'], bins=50, alpha=0.6, 
               color=colors['inter_class'], label='Inter-class', density=True)
    
    # Add mean lines
    ax_sup.axvline(np.mean(sup_dists['intra_normal']), 
                  color=colors['intra_normal'], linestyle='--', linewidth=2, 
                  label=f'Normal μ={np.mean(sup_dists["intra_normal"]):.1f}')
    ax_sup.axvline(np.mean(sup_dists['intra_tumor']), 
                  color=colors['intra_tumor'], linestyle='--', linewidth=2, 
                  label=f'Tumor μ={np.mean(sup_dists["intra_tumor"]):.1f}')
    ax_sup.axvline(np.mean(sup_dists['inter_class']), 
                  color=colors['inter_class'], linestyle='--', linewidth=2, 
                  label=f'Inter μ={np.mean(sup_dists["inter_class"]):.1f}')
    
    ax_sup.set_title('Supervised Model - Distance Distributions', 
                    fontsize=12, fontweight='bold')
    ax_sup.set_xlabel('Euclidean Distance')
    ax_sup.set_ylabel('Density')
    ax_sup.legend(loc='upper right', fontsize=9)
    ax_sup.grid(True, alpha=0.3)
    
    # =========================================================================
    # Contrastive embeddings
    # =========================================================================
    ax_con = axes[0, 1]
    
    # Plot distributions
    ax_con.hist(con_dists['intra_normal'], bins=50, alpha=0.6, 
               color=colors['intra_normal'], label='Intra-class (Normal)', density=True)
    ax_con.hist(con_dists['intra_tumor'], bins=50, alpha=0.6, 
               color=colors['intra_tumor'], label='Intra-class (Tumor)', density=True)
    ax_con.hist(con_dists['inter_class'], bins=50, alpha=0.6, 
               color=colors['inter_class'], label='Inter-class', density=True)
    
    # Add mean lines
    ax_con.axvline(np.mean(con_dists['intra_normal']), 
                  color=colors['intra_normal'], linestyle='--', linewidth=2, 
                  label=f'Normal μ={np.mean(con_dists["intra_normal"]):.1f}')
    ax_con.axvline(np.mean(con_dists['intra_tumor']), 
                  color=colors['intra_tumor'], linestyle='--', linewidth=2, 
                  label=f'Tumor μ={np.mean(con_dists["intra_tumor"]):.1f}')
    ax_con.axvline(np.mean(con_dists['inter_class']), 
                  color=colors['inter_class'], linestyle='--', linewidth=2, 
                  label=f'Inter μ={np.mean(con_dists["inter_class"]):.1f}')
    
    ax_con.set_title('Contrastive Model - Distance Distributions', 
                    fontsize=12, fontweight='bold')
    ax_con.set_xlabel('Euclidean Distance')
    ax_con.set_ylabel('Density')
    ax_con.legend(loc='upper right', fontsize=9)
    ax_con.grid(True, alpha=0.3)
    
    # =========================================================================
    # Separation ratio comparison
    # =========================================================================
    ax_ratio = axes[1, 0]
    
    # Compute separation ratios
    sup_intra_mean = (np.mean(sup_dists['intra_normal']) + np.mean(sup_dists['intra_tumor'])) / 2
    sup_inter_mean = np.mean(sup_dists['inter_class'])
    sup_ratio = sup_inter_mean / sup_intra_mean
    
    con_intra_mean = (np.mean(con_dists['intra_normal']) + np.mean(con_dists['intra_tumor'])) / 2
    con_inter_mean = np.mean(con_dists['inter_class'])
    con_ratio = con_inter_mean / con_intra_mean
    
    # Bar plot
    models = ['Supervised', 'Contrastive']
    intra_means = [sup_intra_mean, con_intra_mean]
    inter_means = [sup_inter_mean, con_inter_mean]
    ratios = [sup_ratio, con_ratio]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax_ratio.bar(x - width/2, intra_means, width, label='Intra-class (avg)', 
                        color='#ff7f0e', alpha=0.7)
    bars2 = ax_ratio.bar(x + width/2, inter_means, width, label='Inter-class', 
                        color='#2ca02c', alpha=0.7)
    
    # Add value labels on bars
    for i, (intra, inter, ratio) in enumerate(zip(intra_means, inter_means, ratios)):
        ax_ratio.text(i, max(intra, inter) + 0.5, f'Ratio: {ratio:.2f}', 
                     ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax_ratio.set_title('Mean Distance Comparison', fontsize=12, fontweight='bold')
    ax_ratio.set_ylabel('Mean Euclidean Distance')
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(models)
    ax_ratio.legend()
    ax_ratio.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Statistical summary table
    # =========================================================================
    ax_table = axes[1, 1]
    ax_table.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric', 'Supervised', 'Contrastive'],
        ['', '', ''],
        ['Intra-class (Normal)', '', ''],
        [f'  Mean ± Std', 
         f'{np.mean(sup_dists["intra_normal"]):.2f} ± {np.std(sup_dists["intra_normal"]):.2f}',
         f'{np.mean(con_dists["intra_normal"]):.2f} ± {np.std(con_dists["intra_normal"]):.2f}'],
        ['', '', ''],
        ['Intra-class (Tumor)', '', ''],
        [f'  Mean ± Std', 
         f'{np.mean(sup_dists["intra_tumor"]):.2f} ± {np.std(sup_dists["intra_tumor"]):.2f}',
         f'{np.mean(con_dists["intra_tumor"]):.2f} ± {np.std(con_dists["intra_tumor"]):.2f}'],
        ['', '', ''],
        ['Inter-class', '', ''],
        [f'  Mean ± Std', 
         f'{np.mean(sup_dists["inter_class"]):.2f} ± {np.std(sup_dists["inter_class"]):.2f}',
         f'{np.mean(con_dists["inter_class"]):.2f} ± {np.std(con_dists["inter_class"]):.2f}'],
        ['', '', ''],
        ['Separation Metrics', '', ''],
        [f'  Inter/Intra Ratio', f'{sup_ratio:.3f}', f'{con_ratio:.3f}'],
        [f'  Separation Gap', 
         f'{sup_inter_mean - sup_intra_mean:.2f}',
         f'{con_inter_mean - con_intra_mean:.2f}'],
    ]
    
    table = ax_table.table(cellText=table_data, cellLoc='left', loc='center',
                          colWidths=[0.35, 0.32, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style section headers
    for row in [2, 5, 8, 11]:
        for col in range(3):
            table[(row, col)].set_facecolor('#f0f0f0')
            table[(row, col)].set_text_props(weight='bold')
    
    ax_table.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Distance Distribution Analysis: Supervised vs Contrastive', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Distance distributions saved: {output_path}")
    plt.close()
    
    return {
        'supervised': {
            'intra_mean': sup_intra_mean,
            'inter_mean': sup_inter_mean,
            'ratio': sup_ratio,
            'gap': sup_inter_mean - sup_intra_mean
        },
        'contrastive': {
            'intra_mean': con_intra_mean,
            'inter_mean': con_inter_mean,
            'ratio': con_ratio,
            'gap': con_inter_mean - con_intra_mean
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Distance distribution analysis')
    
    # Models
    parser.add_argument('--supervised-model', type=str, required=True,
                        help='Path to supervised model')
    parser.add_argument('--contrastive-model', type=str, required=True,
                        help='Path to contrastive model')
    
    # Data
    parser.add_argument('--pcam-dir', type=str, default='PCam',
                        help='Path to PCam directory')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy',
                        help='Reference tile for stain normalization')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy',
                        help='RGB normalization stats')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Dataset split to analyze')
    
    # Analysis
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Max samples for distance computation')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/distance_analysis',
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
    print("DISTANCE DISTRIBUTION ANALYSIS")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    print(f"Supervised model: {args.supervised_model}")
    print(f"Contrastive model: {args.contrastive_model}")
    print(f"Dataset split: {args.split}")
    print(f"Max samples: {args.n_samples:,}")
    
    # =========================================================================
    # Load dataset
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"LOADING {args.split.upper()} DATASET")
    print(f"{'='*70}\n")
    
    # Load normalization stats
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    dataset = PatchCamelyonDataset(
        data_dir=args.pcam_dir,
        split=args.split,
        stain_norm=True,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=mean,
        std=std,
        augment=False,
        filter_normal_only=False
    )
    
    print(f"\nDataset: {len(dataset):,} patches")
    
    # =========================================================================
    # Load models
    # =========================================================================
    print(f"\n{'='*70}")
    print("LOADING MODELS")
    print(f"{'='*70}\n")
    
    # Supervised model
    supervised_model = SimpleResNet18(pretrained=False).to(device)
    sup_checkpoint = torch.load(args.supervised_model, map_location=device, weights_only=False)
    supervised_model.load_state_dict(sup_checkpoint['model_state_dict'])
    supervised_model.eval()
    print(f"  ✓ Supervised model loaded")
    
    # Contrastive model
    contrastive_model = ContrastiveResNet(pretrained=False).to(device)
    con_checkpoint = torch.load(args.contrastive_model, map_location=device, weights_only=False)
    contrastive_model.load_state_dict(con_checkpoint['model_state_dict'])
    contrastive_model.eval()
    print(f"  ✓ Contrastive model loaded")
    
    # =========================================================================
    # Extract features
    # =========================================================================
    print(f"\n{'='*70}")
    print("EXTRACTING FEATURES")
    print(f"{'='*70}\n")
    
    print("Extracting supervised features...")
    sup_features, sup_labels = extract_features_supervised(
        supervised_model, dataset, device, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"  ✓ Supervised features: {sup_features.shape}")
    
    print("\nExtracting contrastive features...")
    con_features, con_labels = extract_features_contrastive(
        contrastive_model, dataset, device, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"  ✓ Contrastive features: {con_features.shape}")
    
    # =========================================================================
    # Compute distance distributions
    # =========================================================================
    print(f"\n{'='*70}")
    print("COMPUTING DISTANCE DISTRIBUTIONS")
    print(f"{'='*70}\n")
    
    print("Supervised embeddings:")
    sup_dists = compute_distance_distributions(sup_features, sup_labels, n_samples=args.n_samples)
    
    print("\nContrastive embeddings:")
    con_dists = compute_distance_distributions(con_features, con_labels, n_samples=args.n_samples)
    
    # =========================================================================
    # Create visualizations
    # =========================================================================
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    plot_path = output_dir / f'distance_distributions_{args.split}.png'
    summary_stats = plot_distance_distributions(sup_dists, con_dists, plot_path)
    
    # =========================================================================
    # Save summary statistics
    # =========================================================================
    summary_df = pd.DataFrame([
        {
            'Model': 'Supervised',
            'Split': args.split,
            'Intra-class Mean': summary_stats['supervised']['intra_mean'],
            'Inter-class Mean': summary_stats['supervised']['inter_mean'],
            'Separation Ratio': summary_stats['supervised']['ratio'],
            'Separation Gap': summary_stats['supervised']['gap']
        },
        {
            'Model': 'Contrastive',
            'Split': args.split,
            'Intra-class Mean': summary_stats['contrastive']['intra_mean'],
            'Inter-class Mean': summary_stats['contrastive']['inter_mean'],
            'Separation Ratio': summary_stats['contrastive']['ratio'],
            'Separation Gap': summary_stats['contrastive']['gap']
        }
    ])
    
    summary_path = output_dir / f'distance_summary_{args.split}.csv'
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"  ✓ Summary statistics saved: {summary_path}")
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("✓ DISTANCE ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    print(summary_df.to_string(index=False))
    print("")
    
    # Determine which has better separation
    sup_ratio = summary_stats['supervised']['ratio']
    con_ratio = summary_stats['contrastive']['ratio']
    
    if sup_ratio > con_ratio:
        print(f"🏆 Better separation: Supervised (ratio={sup_ratio:.3f} vs {con_ratio:.3f})")
    else:
        print(f"🏆 Better separation: Contrastive (ratio={con_ratio:.3f} vs {sup_ratio:.3f})")


if __name__ == '__main__':
    main()

