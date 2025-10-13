#!/usr/bin/env python3
"""
Train autoencoder on tissue patches for tumor detection.

Usage:
    python train_autoencoder.py --epochs 50 --batch-size 128 --latent-dim 256
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm


class TissueDataset(Dataset):
    """Dataset for tissue patches (PCam normals)."""
    
    def __init__(self, csv_path, split='train'):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split]
        self.base_dir = Path(csv_path).parent
        print(f"  Loaded {len(self.df):,} {split} patches")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.base_dir / row['path']
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img_rgb.astype(np.float32) / 255.0
        
        # To tensor [C, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        return img_tensor, img_tensor  # Input and target same for AE


class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for 96x96 RGB tissue patches."""
    
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # Encoder: 96x96x3 → latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, latent_dim),
        )
        
        # Decoder: latent_dim → 96x96x3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 6 * 6),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 6, 6)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward
        recon = model(images)
        loss = criterion(recon, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description='Train autoencoder for tumor detection')
    parser.add_argument('--data-csv', type=str, default='final_dataset/dataset.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    
    args = parser.parse_args()
    
    print("="*70)
    print("AUTOENCODER TRAINING FOR TUMOR DETECTION")
    print("="*70)
    print()
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Data loader
    print("Loading dataset...")
    train_dataset = TissueDataset(args.data_csv, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"  Batches per epoch: {len(train_loader):,}")
    print()
    
    # Model
    print("Building model...")
    model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Latent dim: {args.latent_dim}")
    print()
    
    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_loss = float('inf')
    
    print("="*70)
    print("TRAINING")
    print("="*70)
    print()
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {train_loss:.6f}", end="")
        
        # Scheduler
        scheduler.step(train_loss)
        
        # Save best
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'config': {
                    'latent_dim': args.latent_dim,
                    'batch_size': args.batch_size,
                }
            }, 'autoencoder_best.pth')
            print(" ← Best model saved!")
        else:
            print()
    
    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Model saved to: autoencoder_best.pth")
    print()
    print("NEXT STEPS:")
    print("  1. Run inference: python run_inference.py")
    print("  2. Generate heatmaps: python generate_heatmaps.py ...")
    print("  3. Compute metrics: python compute_metrics.py ...")


if __name__ == '__main__':
    main()

