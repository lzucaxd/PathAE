#!/usr/bin/env python3
"""
Create Grad-CAM comparison figure showing supervised vs contrastive attention.

Generates a 3×4 grid:
- Row 1: Original patches (TP, TN, FP, FN)
- Row 2: Supervised Grad-CAM overlays
- Row 3: Contrastive Grad-CAM overlays
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import h5py

from gradcam import GradCAM
from dataset_pcam import PatchCamelyonDataset
from models import SimpleResNet18
from contrastive_model import ContrastiveResNet


def load_patch_from_pcam(pcam_dir, split, index):
    """
    Load a specific patch from PCam HDF5 file.
    
    Returns:
        image: [96, 96, 3] RGB image (uint8)
        label: Binary label
    """
    pcam_dir = Path(pcam_dir)
    
    # Map split names
    split_mapping = {'train': 'training', 'valid': 'validation', 'test': 'test'}
    file_split = split_mapping.get(split, split)
    
    # Load from HDF5
    x_path = pcam_dir / 'pcam' / f'{file_split}_split.h5'
    y_path = pcam_dir / 'Labels' / 'Labels' / f'camelyonpatch_level_2_split_{split}_y.h5'
    
    with h5py.File(x_path, 'r') as f:
        image = f['x'][index]
    
    with h5py.File(y_path, 'r') as f:
        label = f['y'][index].squeeze()
    
    return image, label


def preprocess_image(image, stain_normalizer, mean, std):
    """
    Preprocess image for model input.
    
    Args:
        image: [96, 96, 3] RGB uint8
        stain_normalizer: StainNormalizer instance
        mean: RGB mean
        std: RGB std
    
    Returns:
        tensor: [1, 3, 96, 96] preprocessed tensor
    """
    # Stain normalization
    if stain_normalizer is not None:
        image_norm = stain_normalizer.normalize(image)
    else:
        image_norm = image
    
    # Convert to [0, 1] and normalize
    image_float = image_norm.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)
    
    # RGB normalization
    for c in range(3):
        image_tensor[c] = (image_tensor[c] - mean[c]) / (std[c] + 1e-6)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--examples-json', type=str, default='gradcam_examples.json')
    parser.add_argument('--supervised-model', type=str,
                       default='checkpoints/supervised_scratch/supervised_scratch_best.pt')
    parser.add_argument('--contrastive-model', type=str,
                       default='checkpoints/contrastive_scratch/contrastive_scratch_best.pt')
    parser.add_argument('--linear-model', type=str,
                       default='checkpoints/linear_eval/contrastive_linear_best.pt')
    parser.add_argument('--pcam-dir', type=str, default='PCam')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    parser.add_argument('--output', type=str, default='figures/gradcam_comparison.png')
    parser.add_argument('--alpha', type=float, default=0.4, help='Overlay transparency')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print("GENERATING GRAD-CAM COMPARISON")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load examples
    print("\nLoading examples...")
    with open(args.examples_json, 'r') as f:
        examples = json.load(f)
    print(f"  ✓ Loaded {len(examples)} examples")
    
    # Load normalization
    from stain_utils import StainNormalizer
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    stain_normalizer = StainNormalizer(reference_path=args.reference_tile, method='macenko')
    
    # Load models
    print("\nLoading models...")
    
    # Supervised
    supervised_model = SimpleResNet18(pretrained=False).to(device)
    sup_checkpoint = torch.load(args.supervised_model, map_location=device, weights_only=False)
    supervised_model.load_state_dict(sup_checkpoint['model_state_dict'])
    supervised_model.eval()
    print(f"  ✓ Supervised model loaded")
    
    # Contrastive + Linear
    contrastive_model = ContrastiveResNet(pretrained=False).to(device)
    con_checkpoint = torch.load(args.contrastive_model, map_location=device, weights_only=False)
    contrastive_model.load_state_dict(con_checkpoint['model_state_dict'])
    
    class LinearClassifier(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(512, 1)
            # ContrastiveResNet.encoder is Sequential(*list(resnet.children())[:-1])
            # This Sequential contains: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
            # So layer4 is at index -2 (before avgpool)
            # Access it from the Sequential
            encoder_children = list(encoder.encoder.children())
            # Find layer4 (should be second to last before avgpool)
            self.layer4 = encoder_children[-2]
        
        def forward(self, x):
            # Don't use no_grad here - we need gradients for Grad-CAM
            features = self.encoder.encoder(x)
            if len(features.shape) == 4:
                features = features.squeeze()
            return self.classifier(features)
    
    contrastive_linear = LinearClassifier(contrastive_model).to(device)
    linear_checkpoint = torch.load(args.linear_model, map_location=device, weights_only=False)
    contrastive_linear.classifier.load_state_dict(linear_checkpoint['classifier_state_dict'])
    contrastive_linear.eval()
    print(f"  ✓ Contrastive + Linear model loaded")
    
    # Create Grad-CAM instances
    print("\nInitializing Grad-CAM...")
    supervised_gradcam = GradCAM(supervised_model, supervised_model.backbone.layer4[-1])
    contrastive_gradcam = GradCAM(contrastive_linear, contrastive_linear.layer4[-1])
    print(f"  ✓ Grad-CAM initialized")
    
    # Process each example
    print("\nGenerating Grad-CAMs...")
    
    case_order = ['TP', 'TN', 'FP', 'FN']
    case_names = {
        'TP': 'True Positive',
        'TN': 'True Negative',
        'FP': 'False Positive',
        'FN': 'False Negative'
    }
    
    images_original = []
    images_sup_gradcam = []
    images_con_gradcam = []
    
    for case_type in case_order:
        if case_type not in examples:
            print(f"  ⚠ {case_type} not found, skipping...")
            continue
        
        info = examples[case_type]
        index = info['index']
        label = info['label']
        prob = info['prob']
        
        print(f"\n  Processing {case_type} (index={index}, label={label}, prob={prob:.4f})...")
        
        # Load patch
        image, _ = load_patch_from_pcam(args.pcam_dir, 'test', index)
        images_original.append(image)
        
        # Preprocess
        image_tensor = preprocess_image(image, stain_normalizer, 
                                       norm_stats['mean'], norm_stats['std'])
        image_tensor = image_tensor.to(device)
        
        # Generate Grad-CAM for supervised
        print(f"    Generating Supervised Grad-CAM...")
        sup_cam = supervised_gradcam.generate_cam(image_tensor)
        sup_overlay = supervised_gradcam.overlay_cam(image, sup_cam, alpha=args.alpha)
        images_sup_gradcam.append(sup_overlay)
        
        # Generate Grad-CAM for contrastive
        print(f"    Generating Contrastive Grad-CAM...")
        con_cam = contrastive_gradcam.generate_cam(image_tensor)
        con_overlay = contrastive_gradcam.overlay_cam(image, con_cam, alpha=args.alpha)
        images_con_gradcam.append(con_overlay)
        
        print(f"    ✓ Grad-CAMs generated")
    
    # Create figure
    print("\nCreating comparison figure...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for col_idx, case_type in enumerate(case_order):
        if case_type not in examples:
            continue
        
        info = examples[case_type]
        prob = info['prob']
        label_str = 'Tumor' if info['label'] == 1 else 'Normal'
        pred_str = 'Tumor' if prob > 0.5 else 'Normal'
        
        # Row 1: Original
        axes[0, col_idx].imshow(images_original[col_idx])
        axes[0, col_idx].set_title(f'{case_names[case_type]}\n'
                                   f'Label: {label_str}\n'
                                   f'Pred: {pred_str} ({prob:.3f})',
                                   fontsize=11, fontweight='bold')
        axes[0, col_idx].axis('off')
        
        # Row 2: Supervised Grad-CAM
        axes[1, col_idx].imshow(images_sup_gradcam[col_idx])
        if col_idx == 0:
            axes[1, col_idx].set_ylabel('Supervised', fontsize=12, fontweight='bold')
        axes[1, col_idx].axis('off')
        
        # Row 3: Contrastive Grad-CAM
        axes[2, col_idx].imshow(images_con_gradcam[col_idx])
        if col_idx == 0:
            axes[2, col_idx].set_ylabel('Contrastive', fontsize=12, fontweight='bold')
        axes[2, col_idx].axis('off')
    
    plt.suptitle('Grad-CAM Analysis: Supervised vs Contrastive Models',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print("✓ GRAD-CAM COMPARISON COMPLETE")
    print(f"{'='*70}\n")
    print(f"  Figure saved: {args.output}")
    print("\n  Interpretation:")
    print("  - Red regions: High importance for prediction")
    print("  - Blue regions: Low importance")
    print("  - Compare supervised vs contrastive attention patterns")


if __name__ == '__main__':
    main()

