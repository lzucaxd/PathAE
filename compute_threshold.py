#!/usr/bin/env python3
"""
Compute optimal threshold from training normal reconstruction errors.

Uses 99.7th percentile (3σ in normal distribution) as threshold for anomaly detection.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

#from model_vae import BetaVAE
from model_vae_skip import VAESkip96

def compute_threshold_from_normals(model_path, data_csv, n_samples=5000, percentile=99.7):
    """
    Compute threshold from reconstruction errors on training normals.
    
    Args:
        model_path: Path to trained model
        data_csv: Path to dataset CSV
        n_samples: Number of normal samples to use
        percentile: Percentile for threshold (99.7 = 3σ)
        
    Returns:
        threshold: Float threshold value
    """
    print("="*70)
    print("COMPUTING ANOMALY THRESHOLD FROM TRAINING NORMALS")
    print("="*70)
    print()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']
    z_dim = config['z_dim']
    # mean = config['mean']
    # std = config['std']
    
    # Fallback for mean/std if not in config
    if 'mean' in config:
        mean = config['mean']
        std = config['std']
    else:
        mean = np.array([0.182, 0.182, 0.182])
        std = np.array([0.427, 0.427, 0.427])
    
    model = VAESkip96(z_ch=z_dim, num_groups=config.get('num_groups', 8)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  z_dim: {z_dim}")
    print()
    
    # Load training normals
    df = pd.read_csv(data_csv)
    df = df[df['split'] == 'train'].sample(n=min(n_samples, len(df)), random_state=42)
    base_dir = Path(data_csv).parent
    
    print(f"Computing errors on {len(df):,} training normals...")
    
    # Compute reconstruction errors
    errors = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(df)), desc="Processing"):
            row = df.iloc[idx]
            img_path = base_dir / row['path']
            
            # Load and preprocess
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Scale to [0,1]
            img_float = img_rgb.astype(np.float32) / 255.0
            
            # To tensor and normalize
            img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)
            for c in range(3):
                img_tensor[c] = (img_tensor[c] - mean[c]) / (std[c] + 1e-6)
            
            img_batch = img_tensor.unsqueeze(0).to(device)
            
            # Reconstruct
            mu, logvar, skips = model.encode(img_batch)
            recon = model.decode(mu, skips)
            
            # Compute error (MSE in normalized space)
            mse = ((img_batch - recon) ** 2).mean().item()
            errors.append(mse)
    
    errors = np.array(errors)
    
    # Compute threshold
    threshold = np.percentile(errors, percentile)
    
    print()
    print("="*70)
    print("THRESHOLD STATISTICS")
    print("="*70)
    print(f"Training normal errors (n={len(errors):,}):")
    print(f"  Mean: {errors.mean():.6f}")
    print(f"  Std:  {errors.std():.6f}")
    print(f"  Min:  {errors.min():.6f}")
    print(f"  Max:  {errors.max():.6f}")
    print()
    print(f"Threshold ({percentile}th percentile): {threshold:.6f}")
    print()
    print(f"Interpretation:")
    print(f"  Scores > {threshold:.6f} are anomalies (top {100-percentile:.1f}%)")
    print(f"  Approximately {100-percentile:.1f}% false positive rate on normals")
    print()
    
    # Save threshold
    np.save('threshold.npy', {'threshold': threshold, 'percentile': percentile, 'errors': errors})
    print(f"✓ Threshold saved to: threshold.npy")
    
    return threshold


def main():
    parser = argparse.ArgumentParser(description='Compute anomaly threshold')
    parser.add_argument('--model', type=str, default='vae_best.pth')
    parser.add_argument('--csv', type=str, default='final_dataset/dataset.csv')
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--percentile', type=float, default=99.7)
    
    args = parser.parse_args()
    
    threshold = compute_threshold_from_normals(
        args.model, args.csv, args.n_samples, args.percentile
    )
    
    print()
    print("USAGE:")
    print(f"  Use threshold={threshold:.6f} for anomaly detection")
    print(f"  Or load from threshold.npy:")
    print(f"    threshold = np.load('threshold.npy', allow_pickle=True).item()['threshold']")


if __name__ == '__main__':
    main()

