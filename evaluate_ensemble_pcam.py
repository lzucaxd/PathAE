#!/usr/bin/env python3
"""
Evaluate ensemble model on PCam test set.

Generates predictions from both supervised and contrastive models,
creates ensemble, and computes comprehensive metrics.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

from dataset_pcam import PatchCamelyonDataset
from models import SimpleResNet18
from contrastive_model import ContrastiveResNet


def generate_predictions(model, dataset, device, batch_size=256, num_workers=4, model_type='supervised'):
    """Generate predictions for entire dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=num_workers, pin_memory=False)
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {model_type}"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze()
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    
    # Ensure correct shape
    if len(probs.shape) == 0:
        probs = np.array([probs])
    
    return probs, labels


def compute_metrics(probs, labels, threshold=0.5):
    """Compute comprehensive metrics."""
    preds = (probs > threshold).astype(int)
    
    # Classification metrics
    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate ensemble on PCam test set')
    
    # Models
    parser.add_argument('--supervised-model', type=str,
                       default='checkpoints/supervised_scratch/supervised_scratch_best.pt')
    parser.add_argument('--contrastive-model', type=str,
                       default='checkpoints/contrastive_scratch/contrastive_scratch_best.pt')
    parser.add_argument('--linear-model', type=str,
                       default='checkpoints/linear_eval/contrastive_linear_best.pt')
    
    # Data
    parser.add_argument('--pcam-dir', type=str, default='PCam')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/ensemble_pcam')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print("ENSEMBLE EVALUATION ON PCAM TEST SET")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    
    # =========================================================================
    # Load PCam test set
    # =========================================================================
    print(f"\n{'='*70}")
    print("LOADING PCAM TEST SET")
    print(f"{'='*70}\n")
    
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    
    test_dataset = PatchCamelyonDataset(
        data_dir=args.pcam_dir,
        split='test',
        stain_norm=True,
        reference_tile=args.reference_tile,
        normalize=True,
        mean=norm_stats['mean'],
        std=norm_stats['std'],
        augment=False,
        filter_normal_only=False
    )
    
    print(f"\nDataset: {len(test_dataset):,} test patches")
    
    # =========================================================================
    # Load models
    # =========================================================================
    print(f"\n{'='*70}")
    print("LOADING MODELS")
    print(f"{'='*70}\n")
    
    # Supervised
    supervised_model = SimpleResNet18(pretrained=False).to(device)
    sup_checkpoint = torch.load(args.supervised_model, map_location=device, weights_only=False)
    supervised_model.load_state_dict(sup_checkpoint['model_state_dict'])
    supervised_model.eval()
    print("  ✓ Supervised model loaded")
    
    # Contrastive + Linear
    contrastive_model = ContrastiveResNet(pretrained=False).to(device)
    con_checkpoint = torch.load(args.contrastive_model, map_location=device, weights_only=False)
    contrastive_model.load_state_dict(con_checkpoint['model_state_dict'])
    
    class LinearClassifier(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(512, 1)
        
        def forward(self, x):
            with torch.no_grad():
                features = self.encoder.encoder(x)
                if len(features.shape) == 4:
                    features = features.squeeze()
            return self.classifier(features)
    
    contrastive_linear = LinearClassifier(contrastive_model).to(device)
    linear_checkpoint = torch.load(args.linear_model, map_location=device, weights_only=False)
    contrastive_linear.classifier.load_state_dict(linear_checkpoint['classifier_state_dict'])
    contrastive_linear.eval()
    print("  ✓ Contrastive + Linear model loaded")
    
    # =========================================================================
    # Generate predictions
    # =========================================================================
    print(f"\n{'='*70}")
    print("GENERATING PREDICTIONS")
    print(f"{'='*70}\n")
    
    print("Supervised model:")
    sup_probs, labels = generate_predictions(supervised_model, test_dataset, device, 
                                            args.batch_size, args.num_workers, 'Supervised')
    
    print("\nContrastive model:")
    con_probs, _ = generate_predictions(contrastive_linear, test_dataset, device, 
                                       args.batch_size, args.num_workers, 'Contrastive')
    
    # Create ensemble
    print("\nCreating ensemble...")
    ensemble_probs = (sup_probs + con_probs) / 2
    uncertainty = np.abs(sup_probs - con_probs)
    
    print(f"  ✓ Ensemble created")
    print(f"  Mean ensemble score: {ensemble_probs.mean():.4f}")
    print(f"  Mean uncertainty: {uncertainty.mean():.4f}")
    
    # =========================================================================
    # Evaluate all three models
    # =========================================================================
    print(f"\n{'='*70}")
    print("COMPUTING METRICS")
    print(f"{'='*70}\n")
    
    print("Supervised:")
    sup_metrics = compute_metrics(sup_probs, labels)
    for key, val in sup_metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    print("\nContrastive:")
    con_metrics = compute_metrics(con_probs, labels)
    for key, val in con_metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    print("\nEnsemble:")
    ens_metrics = compute_metrics(ensemble_probs, labels)
    for key, val in ens_metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    # =========================================================================
    # Create comparison table
    # =========================================================================
    comparison_df = pd.DataFrame([
        {'Model': 'Supervised', **sup_metrics},
        {'Model': 'Contrastive', **con_metrics},
        {'Model': 'Ensemble', **ens_metrics}
    ])
    
    comparison_path = output_dir / 'pcam_test_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False, float_format='%.4f')
    
    print(f"\n{'='*70}")
    print("✓ EVALUATION COMPLETE")
    print(f"{'='*70}\n")
    
    print("Comparison Table:")
    print(comparison_df[['Model', 'pr_auc', 'roc_auc', 'accuracy', 'f1']].to_string(index=False))
    
    print(f"\n  ✓ Results saved: {comparison_path}")
    
    # Determine winner
    print(f"\n🏆 Best Model by Metric:")
    print(f"  PR-AUC:   {comparison_df.loc[comparison_df['pr_auc'].idxmax(), 'Model']}")
    print(f"  ROC-AUC:  {comparison_df.loc[comparison_df['roc_auc'].idxmax(), 'Model']}")
    print(f"  Accuracy: {comparison_df.loc[comparison_df['accuracy'].idxmax(), 'Model']}")
    print(f"  F1:       {comparison_df.loc[comparison_df['f1'].idxmax(), 'Model']}")


if __name__ == '__main__':
    main()


