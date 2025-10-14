#!/usr/bin/env python3
"""
VAE-Skip96: β-VAE with U-Net style skip connections.

Architecture:
- 5× down / 5× up encoder-decoder
- Skip connections between encoder and decoder at each level
- Spatial latent z_ch × 3 × 3
- GroupNorm for stability with small batches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class EncoderBlock(nn.Module):
    """Encoder block with conv, GroupNorm, LeakyReLU."""
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    Decoder block with upsample+conv.
    
    Args:
        in_ch_upsampled: Channels from upsampled features (before skip)
        skip_ch: Channels from skip connection (0 if no skip)
        out_ch: Output channels
        skip_dropout: Dropout probability for skip connections (prevents bypassing latent)
    """
    def __init__(self, in_ch_upsampled, skip_ch, out_ch, num_groups=8, skip_dropout=0.25):
        super().__init__()
        self.use_skip = (skip_ch > 0)
        self.skip_dropout = skip_dropout
        
        # Total input channels = upsampled + skip
        total_in_ch = in_ch_upsampled + skip_ch
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(total_in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x, skip=None):
        x = self.upsample(x)
        if self.use_skip and skip is not None:
            # Apply dropout to skip connection to force latent usage
            skip = F.dropout(skip, p=self.skip_dropout, training=self.training)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class VAESkip96(nn.Module):
    """
    VAE-Skip96: β-VAE with skip connections for 96×96 patches.
    
    Encoder: 96→48→24→12→6→3 (5 levels)
    Channels: 3→64→128→256→256→256
    Latent: z_ch × 3 × 3 (spatial)
    Decoder: Mirror with skip connections
    """
    
    def __init__(self, z_ch=128, num_groups=8):
        super().__init__()
        self.z_ch = z_ch
        
        # Encoder: 5 downsampling blocks
        self.enc1 = EncoderBlock(3, 64, num_groups)      # 96→48
        self.enc2 = EncoderBlock(64, 128, num_groups)    # 48→24
        self.enc3 = EncoderBlock(128, 256, num_groups)   # 24→12
        self.enc4 = EncoderBlock(256, 256, num_groups)   # 12→6
        self.enc5 = EncoderBlock(256, 256, num_groups)   # 6→3
        
        # Latent space: μ and logσ² heads
        self.fc_mu = nn.Conv2d(256, z_ch, kernel_size=1)
        self.fc_logvar = nn.Conv2d(256, z_ch, kernel_size=1)
        
        # Latent to decoder
        self.fc_decode = nn.Conv2d(z_ch, 256, kernel_size=1)
        
        # Decoder: 5 upsampling blocks with skip connections
        # (in_ch_upsampled, skip_ch, out_ch)
        # dec5: upsample 256, skip e4 (256), out 256
        # dec4: upsample 256, skip e3 (256), out 256
        # dec3: upsample 256, skip e2 (128), out 128
        # dec2: upsample 128, skip e1 (64), out 64
        # dec1: upsample 64, no skip (0), out 32
        self.dec5 = DecoderBlock(256, 256, 256, num_groups)   # 3→6, +e4 (256)
        self.dec4 = DecoderBlock(256, 256, 256, num_groups)   # 6→12, +e3 (256)
        self.dec3 = DecoderBlock(256, 128, 128, num_groups)   # 12→24, +e2 (128)
        self.dec2 = DecoderBlock(128, 64, 64, num_groups)     # 24→48, +e1 (64)
        self.dec1 = DecoderBlock(64, 0, 32, num_groups)       # 48→96, no skip
        
        # Final conv to RGB (no activation - output in normalized space)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
    
    def encode(self, x):
        """
        Encode with skip connections stored.
        
        Returns:
            mu, logvar, skips: Latent params and skip connections [e1, e2, e3, e4]
                               (e5 is bottleneck, not a skip)
        """
        # Encoder with skip storage
        e1 = self.enc1(x)      # 48x48
        e2 = self.enc2(e1)     # 24x24
        e3 = self.enc3(e2)     # 12x12
        e4 = self.enc4(e3)     # 6x6
        e5 = self.enc5(e4)     # 3x3 (bottleneck)
        
        # Latent
        mu = self.fc_mu(e5)
        logvar = self.fc_logvar(e5)
        
        # Store skips (from encoder to decoder, NOT including bottleneck e5)
        skips = [e1, e2, e3, e4]
        
        return mu, logvar, skips
    
    def decode(self, z, skips):
        """
        Decode with skip connections.
        
        Skip connections match spatial resolution AFTER upsampling:
        - 3→6: skip e4 (6x6)
        - 6→12: skip e3 (12x12)
        - 12→24: skip e2 (24x24)
        - 24→48: skip e1 (48x48)
        - 48→96: no skip
        
        Args:
            z: Latent [B, z_ch, 3, 3]
            skips: List [e1@48, e2@24, e3@12, e4@6]
        """
        e1, e2, e3, e4 = skips
        
        # Latent to decoder features
        d = self.fc_decode(z)  # 3x3, 256 channels
        
        # Decoder with skip connections (match resolution AFTER upsample)
        d = self.dec5(d, e4)    # 3→6, skip e4 (6x6)
        d = self.dec4(d, e3)    # 6→12, skip e3 (12x12)
        d = self.dec3(d, e2)    # 12→24, skip e2 (24x24)
        d = self.dec2(d, e1)    # 24→48, skip e1 (48x48)
        d = self.dec1(d, None)  # 48→96, no skip
        
        # Final conv (no activation)
        recon = self.final_conv(d)
        
        return recon
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, use_mean=False):
        """
        Forward pass.
        
        Args:
            x: Input [B, 3, 96, 96]
            use_mean: If True, use μ instead of sampling (for inference)
        
        Returns:
            recon, mu, logvar
        """
        mu, logvar, skips = self.encode(x)
        
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        
        recon = self.decode(z, skips)
        
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, mean, std, lambda_l1=0.6, lambda_ssim=0.4, beta=1.0, capacity=0.0):
    """
    β-VAE loss with capacity scheduling: L = λ1*L1 + λs*(1-SSIM) + β*max(KL - C, 0)
    
    Args:
        recon: Reconstruction [B, 3, H, W] in normalized space
        x: Input [B, 3, H, W] in normalized space
        mu: Latent mean [B, z_ch, H_latent, W_latent]
        logvar: Latent log variance [B, z_ch, H_latent, W_latent]
        mean: RGB mean for denormalization [3]
        std: RGB std for denormalization [3]
        lambda_l1: Weight for L1 loss (0.6)
        lambda_ssim: Weight for SSIM loss (0.4)
        beta: Weight for KL divergence (1.0 recommended with capacity)
        capacity: KL capacity target in nats (0→120 over training)
    
    Returns:
        loss: Total loss (scalar)
        loss_dict: Dictionary with individual losses
    """
    # Denormalize to [0,1] for proper SSIM computation
    # x_01 = x * std + mean (but std/mean are per-channel)
    mean_t = torch.tensor(mean, device=recon.device, dtype=recon.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=recon.device, dtype=recon.dtype).view(1, 3, 1, 1)
    
    recon_01 = recon * std_t + mean_t
    x_01 = x * std_t + mean_t
    
    # Clamp to [0, 1] for stability
    recon_01 = torch.clamp(recon_01, 0, 1)
    x_01 = torch.clamp(x_01, 0, 1)
    
    # L1 loss (compute in [0,1] space for consistency)
    l1_loss = F.l1_loss(recon_01, x_01, reduction='mean')
    
    # SSIM loss (now properly in [0,1] with data_range=1.0)
    ssim_val = ssim(recon_01, x_01, data_range=1.0, size_average=True)
    ssim_loss = 1.0 - ssim_val
    
    # KL divergence: KL(q(z|x) || N(0,1))
    # Total KL in nats (sum over latents, average over batch)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
    kl_loss = torch.mean(kl_per_sample)
    
    # Capacity scheduling (free-bits style): max(KL - C, 0)
    kl_constrained = torch.clamp(kl_loss - capacity, min=0.0)
    
    # Total loss
    recon_loss = lambda_l1 * l1_loss + lambda_ssim * ssim_loss
    total_loss = recon_loss + beta * kl_constrained
    
    return total_loss, {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'l1': l1_loss.item(),
        'ssim': ssim_loss.item(),
        'kl': kl_loss.item(),
        'kl_constrained': kl_constrained.item(),
    }


class KLWarmup:
    """
    Linear KL warm-up scheduler over first N epochs.
    
    β increases from 0 → β_max linearly.
    """
    
    def __init__(self, beta_max=1.0, warmup_epochs=5):
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
    
    def get_beta(self):
        if self.current_epoch >= self.warmup_epochs:
            return self.beta_max
        return self.beta_max * (self.current_epoch / self.warmup_epochs)


class KLCapacity:
    """
    KL capacity scheduler (free-bits style).
    
    Linearly increases capacity C from 0 → C_max over N epochs.
    Loss uses max(KL - C, 0) to prevent posterior collapse.
    
    Typical values: C_max=100-200 nats for 128×3×3 latent (1152 dims).
    """
    
    def __init__(self, capacity_max=120.0, warmup_epochs=20):
        self.capacity_max = capacity_max
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
    
    def get_capacity(self):
        if self.current_epoch >= self.warmup_epochs:
            return self.capacity_max
        return self.capacity_max * (self.current_epoch / self.warmup_epochs)


if __name__ == '__main__':
    # Test
    import numpy as np
    
    print("Testing VAE-Skip96...")
    model = VAESkip96(z_ch=128, num_groups=8)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Forward pass
    x = torch.randn(2, 3, 96, 96)
    recon, mu, logvar = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Recon shape: {recon.shape}")
    print(f"Latent μ shape: {mu.shape}")
    print(f"Latent logσ² shape: {logvar.shape}")
    
    # Test loss (with dummy mean/std)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.2, 0.2, 0.2])
    loss, loss_dict = vae_loss(recon, x, mu, logvar, mean, std, beta=1.0, capacity=50.0)
    print(f"\nLoss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✓ VAE-Skip96 working correctly!")

