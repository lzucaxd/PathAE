#!/usr/bin/env python3
"""
β-VAE model for tissue anomaly detection.

Architecture:
- 4 conv blocks encoder: 64→128→256→256 (stride 2)
- Mirror decoder with ConvTranspose
- GroupNorm for stability
- Latent: z_dim ∈ {64, 128}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class BetaVAE(nn.Module):
    """
    β-VAE for 96x96 RGB tissue patches.
    
    Encoder: 96→48→24→12→6 (4 stride-2 convs)
    Latent: z_dim (64 or 128)
    Decoder: 6→12→24→48→96 (mirror)
    """
    
    def __init__(self, z_dim=128, num_groups=8):
        super().__init__()
        
        self.z_dim = z_dim
        
        # Encoder: 96x96x3 → 6x6x256
        self.encoder = nn.Sequential(
            # Block 1: 96→48, channels 3→64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2: 48→24, channels 64→128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3: 24→12, channels 128→256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4: 12→6, channels 256→256
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 6 * 6, z_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, z_dim)
        
        # From latent to decoder
        self.fc_decode = nn.Linear(z_dim, 256 * 6 * 6)
        
        # Decoder: 6x6x256 → 96x96x3 (mirror of encoder)
        self.decoder = nn.Sequential(
            # Block 4: 6→12, channels 256→256
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3: 12→24, channels 256→128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2: 24→48, channels 128→64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 1: 48→96, channels 64→3
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] (after denormalization)
        )
    
    def encode(self, x):
        """
        Encode input to latent distribution.
        
        Args:
            x: Input tensor [B, 3, 96, 96]
            
        Returns:
            mu: Mean [B, z_dim]
            logvar: Log variance [B, z_dim]
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε
        
        Args:
            mu: Mean [B, z_dim]
            logvar: Log variance [B, z_dim]
            
        Returns:
            z: Sampled latent [B, z_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent to reconstruction.
        
        Args:
            z: Latent tensor [B, z_dim]
            
        Returns:
            recon: Reconstructed image [B, 3, 96, 96]
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 6, 6)
        recon = self.decoder(h)
        return recon
    
    def forward(self, x):
        """
        Full forward pass.
        
        Args:
            x: Input [B, 3, 96, 96]
            
        Returns:
            recon: Reconstruction [B, 3, 96, 96]
            mu: Latent mean [B, z_dim]
            logvar: Latent log variance [B, z_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, lambda_l1=0.6, lambda_ssim=0.4, beta=1.0):
    """
    β-VAE loss: L = λ1*L1 + λs*(1-SSIM) + β*KL
    
    Args:
        recon: Reconstruction [B, 3, H, W]
        x: Input [B, 3, H, W]
        mu: Latent mean [B, z_dim]
        logvar: Latent log variance [B, z_dim]
        lambda_l1: Weight for L1 loss (0.6)
        lambda_ssim: Weight for SSIM loss (0.4)
        beta: Weight for KL divergence (1 or 3)
        
    Returns:
        loss: Total loss (scalar)
        loss_dict: Dictionary with individual losses
    """
    # L1 reconstruction loss
    l1_loss = F.l1_loss(recon, x, reduction='mean')
    
    # SSIM loss (higher SSIM = better, so use 1-SSIM as loss)
    ssim_val = ssim(recon, x, data_range=1.0, size_average=True)
    ssim_loss = 1.0 - ssim_val
    
    # KL divergence: KL(q(z|x) || N(0,1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Average over batch
    
    # Total loss
    recon_loss = lambda_l1 * l1_loss + lambda_ssim * ssim_loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'l1': l1_loss.item(),
        'ssim': ssim_loss.item(),
        'kl': kl_loss.item(),
    }


class KLWarmup:
    """
    Linear KL warm-up scheduler over first N epochs.
    
    β increases from 0 → β_max linearly.
    """
    
    def __init__(self, beta_max=1.0, warmup_epochs=10):
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step(self):
        """Increment epoch counter."""
        self.current_epoch += 1
    
    def get_beta(self):
        """Get current beta value."""
        if self.current_epoch >= self.warmup_epochs:
            return self.beta_max
        else:
            return self.beta_max * (self.current_epoch / self.warmup_epochs)


if __name__ == '__main__':
    # Test model
    print("Testing β-VAE model...")
    
    model = BetaVAE(z_dim=128)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 96, 96)
    recon, mu, logvar = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Recon shape: {recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test loss
    loss, loss_dict = vae_loss(recon, x, mu, logvar, beta=1.0)
    print(f"\nLoss: {loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    # Test KL warmup
    print("\nKL Warmup schedule:")
    scheduler = KLWarmup(beta_max=3.0, warmup_epochs=10)
    for epoch in range(15):
        print(f"  Epoch {epoch:2d}: β = {scheduler.get_beta():.3f}")
        scheduler.step()
    
    print("\n✓ Model working correctly!")

