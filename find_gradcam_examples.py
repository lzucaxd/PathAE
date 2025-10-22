#!/usr/bin/env python3
"""
Find representative examples from PCam test set for Grad-CAM visualization.

Finds one example each of:
- True Positive (TP): Correctly detected tumor
- True Negative (TN): Correctly identified normal
- False Positive (FP): Normal misclassified as tumor
- False Negative (FN): Missed tumor
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_pcam import PatchCamelyonDataset
from models import SimpleResNet18
from contrastive_model import ContrastiveResNet
import torch.nn as nn


def generate_predictions(model, dataset, device, batch_size=256):
    """Generate predictions for entire dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=4, pin_memory=False)
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Generating predictions"):
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


def find_examples(probs, labels, n_per_category=1):
    """
    Find representative examples for each category.
    
    Returns:
        dict with indices for TP, TN, FP, FN
    """
    # Convert to binary predictions (threshold = 0.5)
    preds = (probs > 0.5).astype(int)
    
    # Find indices for each category
    tp_idx = np.where((labels == 1) & (preds == 1) & (probs > 0.7))[0]  # High confidence TP
    tn_idx = np.where((labels == 0) & (preds == 0) & (probs < 0.3))[0]  # High confidence TN
    fp_idx = np.where((labels == 0) & (preds == 1) & (probs > 0.7))[0]  # High confidence FP
    fn_idx = np.where((labels == 1) & (preds == 0) & (probs < 0.3))[0]  # High confidence FN
    
    print(f"\nFound candidates:")
    print(f"  True Positives (TP):  {len(tp_idx):,} (label=1, pred=1, prob>0.7)")
    print(f"  True Negatives (TN):  {len(tn_idx):,} (label=0, pred=0, prob<0.3)")
    print(f"  False Positives (FP): {len(fp_idx):,} (label=0, pred=1, prob>0.7)")
    print(f"  False Negatives (FN): {len(fn_idx):,} (label=1, pred=0, prob<0.3)")
    
    # Sample one from each category (use median probability)
    examples = {}
    
    if len(tp_idx) > 0:
        # Pick TP with probability closest to 0.85
        tp_probs = probs[tp_idx]
        best_tp = tp_idx[np.argmin(np.abs(tp_probs - 0.85))]
        examples['TP'] = {
            'index': int(best_tp),
            'label': int(labels[best_tp]),
            'prob': float(probs[best_tp])
        }
    
    if len(tn_idx) > 0:
        # Pick TN with probability closest to 0.15
        tn_probs = probs[tn_idx]
        best_tn = tn_idx[np.argmin(np.abs(tn_probs - 0.15))]
        examples['TN'] = {
            'index': int(best_tn),
            'label': int(labels[best_tn]),
            'prob': float(probs[best_tn])
        }
    
    if len(fp_idx) > 0:
        # Pick FP with highest probability (most confident mistake)
        best_fp = fp_idx[np.argmax(probs[fp_idx])]
        examples['FP'] = {
            'index': int(best_fp),
            'label': int(labels[best_fp]),
            'prob': float(probs[best_fp])
        }
    
    if len(fn_idx) > 0:
        # Pick FN with lowest probability (most confident mistake)
        best_fn = fn_idx[np.argmin(probs[fn_idx])]
        examples['FN'] = {
            'index': int(best_fn),
            'label': int(labels[best_fn]),
            'prob': float(probs[best_fn])
        }
    
    return examples


def main():
    parser = argparse.ArgumentParser(description='Find Grad-CAM examples from PCam test set')
    parser.add_argument('--supervised-model', type=str, 
                       default='checkpoints/supervised_scratch/supervised_scratch_best.pt',
                       help='Path to supervised model')
    parser.add_argument('--pcam-dir', type=str, default='PCam',
                       help='Path to PCam directory')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    parser.add_argument('--output', type=str, default='gradcam_examples.json',
                       help='Output JSON file with example indices')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print("FINDING GRAD-CAM EXAMPLES FROM PCAM TEST SET")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    
    # Load normalization stats
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    
    # Load PCam test set (no augmentation)
    print("\nLoading PCam test set...")
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
    print(f"  ✓ Loaded {len(test_dataset):,} test patches")
    
    # Load supervised model
    print("\nLoading supervised model...")
    model = SimpleResNet18(pretrained=False).to(device)
    checkpoint = torch.load(args.supervised_model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model loaded")
    
    # Generate predictions
    print("\nGenerating predictions...")
    probs, labels = generate_predictions(model, test_dataset, device, args.batch_size)
    print(f"  ✓ Generated {len(probs):,} predictions")
    print(f"  Mean probability: {probs.mean():.3f}")
    print(f"  Tumor prevalence: {labels.mean():.3f}")
    
    # Find examples
    print("\nFinding representative examples...")
    examples = find_examples(probs, labels)
    
    # Save examples
    import json
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✓ EXAMPLES FOUND")
    print(f"{'='*70}\n")
    
    print("Selected examples:")
    for case_type, info in examples.items():
        print(f"\n  {case_type}:")
        print(f"    Index:       {info['index']}")
        print(f"    True label:  {info['label']} ({'Tumor' if info['label'] == 1 else 'Normal'})")
        print(f"    Probability: {info['prob']:.4f}")
        print(f"    Prediction:  {'Tumor' if info['prob'] > 0.5 else 'Normal'}")
    
    print(f"\n  ✓ Examples saved to: {output_path}")
    print("\nNext step: Run create_gradcam_comparison.py to generate visualizations")


if __name__ == '__main__':
    main()


