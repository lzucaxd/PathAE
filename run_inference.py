#!/usr/bin/env python3
"""
Run inference on test set to compute reconstruction errors.

Usage:
    python run_inference.py --test-csv test_set_heatmaps/test_set.csv
"""

import argparse
import torch
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class ConvAutoencoder(torch.nn.Module):
    """Convolutional Autoencoder for 96x96 RGB tissue patches."""
    
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 6 * 6, latent_dim),
        )
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256 * 6 * 6),
            torch.nn.ReLU(inplace=True),
            torch.nn.Unflatten(1, (256, 6, 6)),
            
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


def main():
    parser = argparse.ArgumentParser(description='Run inference to compute reconstruction errors')
    parser.add_argument('--test-csv', type=str, required=True,
                        help='Path to test_set.csv')
    parser.add_argument('--model-path', type=str, default='autoencoder_best.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='reconstruction_scores.csv',
                        help='Output CSV path')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    
    print("="*70)
    print("INFERENCE: Computing Reconstruction Errors")
    print("="*70)
    print()
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    latent_dim = checkpoint['config']['latent_dim'] if 'config' in checkpoint else 256
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Latent dim: {latent_dim}")
    print(f"  Training loss: {checkpoint['loss']:.6f}")
    print()
    
    # Load test set (ONLY tissue tiles - background already filtered out!)
    test_df = pd.read_csv(args.test_csv)
    test_base = Path(args.test_csv).parent
    
    print(f"Test set: {len(test_df):,} tissue tiles")
    print(f"  Tumor tiles: {(test_df['label'] == 1).sum():,}")
    print(f"  Normal tiles: {(test_df['label'] == 0).sum():,}")
    print()
    print("Note: Background tiles already filtered out during test set creation!")
    print("      Only processing tissue regions.")
    print()
    
    # Compute reconstruction errors
    results = []
    
    with torch.no_grad():
        for idx in tqdm(range(0, len(test_df), args.batch_size), desc="Processing batches"):
            batch_df = test_df.iloc[idx:idx+args.batch_size]
            
            # Load batch
            batch_images = []
            batch_ids = []
            
            for _, row in batch_df.iterrows():
                tile_path = test_base / row['path']
                img_bgr = cv2.imread(str(tile_path))
                if img_bgr is None:
                    continue
                
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
                
                batch_images.append(img)
                batch_ids.append(row['tile_id'])
            
            if not batch_images:
                continue
            
            # Stack into batch
            images = torch.stack(batch_images).to(device)
            
            # Reconstruction
            recons = model(images)
            
            # MSE per tile
            mse = ((images - recons) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
            
            # Save results
            for tile_id, score in zip(batch_ids, mse):
                results.append({'tile_id': tile_id, 'score': float(score)})
    
    # Save scores
    scores_df = pd.DataFrame(results)
    scores_df.to_csv(args.output, index=False)
    
    # Statistics
    print()
    print("="*70)
    print("RECONSTRUCTION ERROR STATISTICS")
    print("="*70)
    print(f"Total tiles processed: {len(scores_df):,}")
    print(f"Mean error: {scores_df['score'].mean():.6f}")
    print(f"Std error: {scores_df['score'].std():.6f}")
    print(f"Min error: {scores_df['score'].min():.6f}")
    print(f"Max error: {scores_df['score'].max():.6f}")
    print()
    
    # Compare tumor vs normal
    merged = test_df.merge(scores_df, on='tile_id')
    tumor_scores = merged[merged['label'] == 1]['score']
    normal_scores = merged[merged['label'] == 0]['score']
    
    print("By label:")
    print(f"  Tumor mean: {tumor_scores.mean():.6f} ± {tumor_scores.std():.6f}")
    print(f"  Normal mean: {normal_scores.mean():.6f} ± {normal_scores.std():.6f}")
    print(f"  Ratio: {tumor_scores.mean() / normal_scores.mean():.2f}x")
    print()
    print(f"✓ Scores saved to: {args.output}")
    print()
    print("NEXT STEPS:")
    print(f"  1. Generate heatmaps:")
    print(f"     python generate_heatmaps.py --test-csv {args.test_csv} --scores-csv {args.output}")
    print(f"  2. Compute metrics:")
    print(f"     python compute_metrics.py --test-csv {args.test_csv} --scores-csv {args.output}")


if __name__ == '__main__':
    main()

