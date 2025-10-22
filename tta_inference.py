#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) for improved predictions.

Applies 8 augmentations to each patch and averages predictions:
1. Original
2. Horizontal flip
3. Vertical flip
4. 90° rotation
5. 180° rotation
6. 270° rotation
7. H-flip + 90° rotation
8. V-flip + 90° rotation

This provides free performance improvement without retraining!
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from models import SimpleResNet18
from stain_utils import StainNormalizer


class TTADataset(Dataset):
    """Dataset that applies Test-Time Augmentation."""
    
    def __init__(self, df, tiles_dir, stain_normalizer, mean, std):
        self.df = df
        self.tiles_dir = Path(tiles_dir)
        self.stain_normalizer = stain_normalizer
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        if 'path' in row and pd.notna(row['path']):
            tile_path = self.tiles_dir / row['path']
        else:
            tile_path = self.tiles_dir / f"{row['tile_id']}.png"
        
        img = np.array(Image.open(tile_path))
        
        # Stain normalization
        if self.stain_normalizer is not None:
            img = self.stain_normalizer.normalize(img)
        
        # Convert to [0,1] and normalize
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        for c in range(3):
            img_tensor[c] = (img_tensor[c] - self.mean[c]) / (self.std[c] + 1e-6)
        
        return img_tensor, idx


def apply_augmentation(img_tensor, aug_id):
    """
    Apply specific augmentation to image tensor.
    
    Args:
        img_tensor: [C, H, W] tensor
        aug_id: Augmentation ID (0-7)
    
    Returns:
        aug_tensor: Augmented [C, H, W] tensor
    """
    if aug_id == 0:
        # Original
        return img_tensor
    
    elif aug_id == 1:
        # Horizontal flip
        return torch.flip(img_tensor, dims=[2])
    
    elif aug_id == 2:
        # Vertical flip
        return torch.flip(img_tensor, dims=[1])
    
    elif aug_id == 3:
        # 90° rotation (clockwise)
        return torch.rot90(img_tensor, k=1, dims=[1, 2])
    
    elif aug_id == 4:
        # 180° rotation
        return torch.rot90(img_tensor, k=2, dims=[1, 2])
    
    elif aug_id == 5:
        # 270° rotation (= -90°)
        return torch.rot90(img_tensor, k=3, dims=[1, 2])
    
    elif aug_id == 6:
        # H-flip + 90° rotation
        flipped = torch.flip(img_tensor, dims=[2])
        return torch.rot90(flipped, k=1, dims=[1, 2])
    
    elif aug_id == 7:
        # V-flip + 90° rotation
        flipped = torch.flip(img_tensor, dims=[1])
        return torch.rot90(flipped, k=1, dims=[1, 2])
    
    else:
        raise ValueError(f"Invalid augmentation ID: {aug_id}")


def tta_predict(model, image_tensor, device, n_augmentations=8):
    """
    Perform TTA prediction on a single image.
    
    Args:
        model: Trained model
        image_tensor: [C, H, W] tensor
        device: torch device
        n_augmentations: Number of augmentations (default: 8)
    
    Returns:
        avg_prob: Averaged probability across all augmentations
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for aug_id in range(n_augmentations):
            # Apply augmentation
            aug_img = apply_augmentation(image_tensor, aug_id)
            aug_img = aug_img.unsqueeze(0).to(device)  # Add batch dim
            
            # Get prediction
            logit = model(aug_img)
            prob = torch.sigmoid(logit).squeeze()
            
            predictions.append(prob.cpu().item())
    
    # Average predictions
    avg_prob = np.mean(predictions)
    
    return avg_prob


def generate_tta_predictions(model, dataset, device, batch_size=32, num_workers=0):
    """
    Generate TTA predictions for entire dataset.
    
    Note: Smaller batch size and no multiprocessing due to TTA overhead.
    
    Args:
        model: Trained model
        dataset: Dataset
        device: torch device
        batch_size: Batch size (smaller for TTA)
        num_workers: Number of workers (0 for TTA to avoid issues)
    
    Returns:
        predictions: Array of averaged predictions
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=False)
    
    all_predictions = []
    
    for images, indices in tqdm(loader, desc="  TTA inference"):
        batch_preds = []
        
        for img in images:
            # TTA for each image in batch
            avg_prob = tta_predict(model, img, device)
            batch_preds.append(avg_prob)
        
        all_predictions.extend(batch_preds)
    
    return np.array(all_predictions)


def main():
    parser = argparse.ArgumentParser(description='Test-Time Augmentation inference')
    
    # Model
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    
    # Data
    parser.add_argument('--test-csv', type=str, required=True,
                       help='Test set CSV')
    parser.add_argument('--tiles-dir', type=str, required=True,
                       help='Directory containing tiles')
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    
    # TTA config
    parser.add_argument('--n-augmentations', type=int, default=8,
                       help='Number of augmentations (default: 8)')
    
    # Output
    parser.add_argument('--output-csv', type=str, required=True,
                       help='Output predictions CSV')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (smaller for TTA)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='DataLoader workers (0 recommended for TTA)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print("TEST-TIME AUGMENTATION INFERENCE")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    print(f"Augmentations: {args.n_augmentations}")
    print(f"Batch size: {args.batch_size} (smaller for TTA overhead)")
    
    # Load model
    print(f"\nLoading model...")
    model = SimpleResNet18(pretrained=False).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model loaded: {args.model}")
    
    # Load test set
    print(f"\nLoading test set...")
    df = pd.read_csv(args.test_csv)
    print(f"  ✓ Loaded {len(df):,} tiles from {df['wsi_id'].nunique()} slides")
    
    # Load normalization
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    stain_normalizer = StainNormalizer(reference_path=args.reference_tile, method='macenko')
    
    # Create dataset
    dataset = TTADataset(df, args.tiles_dir, stain_normalizer, 
                        norm_stats['mean'], norm_stats['std'])
    
    # Generate TTA predictions
    print(f"\n{'='*70}")
    print("GENERATING TTA PREDICTIONS")
    print(f"{'='*70}\n")
    print(f"Processing {len(dataset):,} tiles with {args.n_augmentations} augmentations each...")
    print(f"Total predictions: {len(dataset) * args.n_augmentations:,}")
    print(f"\nEstimated time: ~{len(dataset) * args.n_augmentations / 1000:.0f} minutes")
    print(f"(This will take longer than standard inference due to TTA overhead)\n")
    
    predictions = generate_tta_predictions(model, dataset, device, 
                                          args.batch_size, args.num_workers)
    
    print(f"\n  ✓ Generated {len(predictions):,} TTA predictions")
    print(f"  Mean score: {predictions.mean():.4f}")
    print(f"  Score range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Create output dataframe
    df_output = pd.DataFrame({
        'tile_id': df['tile_id'],
        'score': predictions
    })
    
    # Save
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"\n{'='*70}")
    print("✓ TTA PREDICTIONS SAVED")
    print(f"{'='*70}\n")
    print(f"  Output: {output_path}")
    print(f"\n  Next step: Generate heatmaps using these TTA predictions")


if __name__ == '__main__':
    main()


