#!/usr/bin/env python3
"""
Standard Autoencoder (AE) - Collapse-proof baseline for tumor detection.

No probabilistic component = No KL divergence = No collapse possible.
Simple reconstruction loss: L = λ1*L1 + λs*(1-SSIM)
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
    """Decoder block with upsample+conv."""
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class AE96(nn.Module):
    """
    Standard Autoencoder for 96×96 patches.
    
    Encoder: 96→48→24→12→6→3 (5 levels)
    Channels: 3→64→128→256→256→256
    Bottleneck: z_dim × 3 × 3 (deterministic, no sampling)
    Decoder: Mirror (no skips for simplicity)
    """
    
    def __init__(self, z_dim=64, num_groups=8):
        super().__init__()
        self.z_dim = z_dim
        
        # Encoder: 5 downsampling blocks
        self.enc1 = EncoderBlock(3, 64, num_groups)      # 96→48
        self.enc2 = EncoderBlock(64, 128, num_groups)    # 48→24
        self.enc3 = EncoderBlock(128, 256, num_groups)   # 24→12
        self.enc4 = EncoderBlock(256, 256, num_groups)   # 12→6
        self.enc5 = EncoderBlock(256, 256, num_groups)   # 6→3
        
        # Bottleneck (deterministic encoding)
        self.fc_encode = nn.Conv2d(256, z_dim, kernel_size=1)
        
        # Bottleneck to decoder
        self.fc_decode = nn.Conv2d(z_dim, 256, kernel_size=1)
        
        # Decoder: 5 upsampling blocks (no skips)
        self.dec5 = DecoderBlock(256, 256, num_groups)   # 3→6
        self.dec4 = DecoderBlock(256, 256, num_groups)   # 6→12
        self.dec3 = DecoderBlock(256, 128, num_groups)   # 12→24
        self.dec2 = DecoderBlock(128, 64, num_groups)    # 24→48
        self.dec1 = DecoderBlock(64, 32, num_groups)     # 48→96
        
        # Final conv to RGB
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
    
    def encode(self, x):
        """Encode input to bottleneck (deterministic)."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        z = self.fc_encode(e5)
        return z
    
    def decode(self, z):
        """Decode bottleneck to reconstruction."""
        d = self.fc_decode(z)
        d = self.dec5(d)
        d = self.dec4(d)
        d = self.dec3(d)
        d = self.dec2(d)
        d = self.dec1(d)
        recon = self.final_conv(d)
        return recon
    
    def forward(self, x):
        """Forward pass."""
        z = self.encode(x)
        recon = self.decode(z)
        # Return dummy mu/logvar for compatibility with training script
        return recon, z, torch.zeros_like(z)


def ae_loss(recon, x, mean, std, lambda_l1=0.6, lambda_ssim=0.4):
    """
    Standard Autoencoder loss: L = λ1*L1 + λs*(1-SSIM)
    
    No KL divergence = no collapse possible!
    
    Args:
        recon: Reconstruction [B, 3, H, W]
        x: Input [B, 3, H, W]
        mean: RGB mean for denormalization
        std: RGB std for denormalization
        lambda_l1: L1 loss weight
        lambda_ssim: SSIM loss weight (set to 0 for L1-only)
    
    Returns:
        loss, loss_dict
    """
    # Denormalize to [0,1]
    mean_t = torch.tensor(mean, device=recon.device, dtype=recon.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=recon.device, dtype=recon.dtype).view(1, 3, 1, 1)
    
    recon_01 = torch.clamp(recon * std_t + mean_t, 0, 1)
    x_01 = torch.clamp(x * std_t + mean_t, 0, 1)
    
    # L1 loss
    l1_loss = F.l1_loss(recon_01, x_01, reduction='mean')
    
    # SSIM loss (skip computation if lambda_ssim=0 for speed)
    if lambda_ssim > 0:
        ssim_val = ssim(recon_01, x_01, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val
    else:
        ssim_loss = 0.0
    
    total_loss = lambda_l1 * l1_loss + lambda_ssim * ssim_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'recon': total_loss.item(),  # Same as total for AE
        'l1': l1_loss.item(),
        'ssim': ssim_loss if isinstance(ssim_loss, float) else ssim_loss.item(),
        'kl': 0.0,  # No KL for standard AE
    }


if __name__ == '__main__':
    # Test
    print("Testing Standard AE...")
    model = AE96(z_dim=64, num_groups=8)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Forward pass
    x = torch.randn(2, 3, 96, 96)
    recon, z, _ = model(x)
    
    print(f"\nInput: {x.shape}")
    print(f"Recon: {recon.shape}")
    print(f"Bottleneck z: {z.shape} ({z.numel() // 2} dims per sample)")
    
    # Test loss
    import numpy as np
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.2, 0.2, 0.2])
    loss, loss_dict = ae_loss(recon, x, mean, std)
    print(f"\nLoss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✓ Standard AE working!")

