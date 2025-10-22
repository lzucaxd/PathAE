#!/usr/bin/env python3
"""
Generate contrastive model predictions and save to CSV.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image

from contrastive_model import ContrastiveResNet
from stain_utils import StainNormalizer


class TileDataset(torch.utils.data.Dataset):
    """Dataset for loading tiles."""
    
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
        
        # Use path column
        if 'path' in row and pd.notna(row['path']):
            tile_path = self.tiles_dir / row['path']
        else:
            tile_path = self.tiles_dir / f"{row['tile_id']}.png"
        
        # Load and preprocess
        img = np.array(Image.open(tile_path))
        
        if self.stain_normalizer is not None:
            img = self.stain_normalizer.normalize(img)
        
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        for c in range(3):
            img_tensor[c] = (img_tensor[c] - self.mean[c]) / (self.std[c] + 1e-6)
        
        return img_tensor, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrastive-model', type=str, required=True)
    parser.add_argument('--linear-model', type=str, required=True)
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--tiles-dir', type=str, required=True)
    parser.add_argument('--reference-tile', type=str, default='reference_tile.npy')
    parser.add_argument('--norm-stats', type=str, default='normalization_stats.npy')
    parser.add_argument('--output-csv', type=str, default='outputs/contrastive_scores.csv')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print("GENERATING CONTRASTIVE PREDICTIONS")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    
    # Load models
    print("Loading models...")
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
    
    model = LinearClassifier(contrastive_model).to(device)
    linear_checkpoint = torch.load(args.linear_model, map_location=device, weights_only=False)
    model.classifier.load_state_dict(linear_checkpoint['classifier_state_dict'])
    model.eval()
    print("  ✓ Models loaded\n")
    
    # Load test set
    print("Loading test set...")
    df = pd.read_csv(args.test_csv)
    print(f"  ✓ Loaded {len(df):,} tiles\n")
    
    # Load normalization
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    stain_normalizer = StainNormalizer(reference_path=args.reference_tile, method='macenko')
    
    # Generate predictions
    print("Generating predictions...")
    dataset = TileDataset(df, args.tiles_dir, stain_normalizer, norm_stats['mean'], norm_stats['std'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=False)
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="  Predicting"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze()
            all_probs.append(probs.cpu().numpy())
    
    predictions = np.concatenate(all_probs)
    
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
    print("✓ PREDICTIONS SAVED")
    print(f"{'='*70}\n")
    print(f"  Output: {output_path}")
    print(f"  Predictions: {len(predictions):,}")
    print(f"  Mean score: {predictions.mean():.4f}")
    print(f"  Score range: [{predictions.min():.4f}, {predictions.max():.4f}]")


if __name__ == '__main__':
    main()


