#!/usr/bin/env python3
"""
Contrastive ResNet18 for supervised contrastive learning.

Architecture:
- Base: ResNet18 (from scratch)
- Projection head: 512 → 256 → 128 (L2 normalized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ContrastiveResNet(nn.Module):
    """
    ResNet18 with projection head for supervised contrastive learning.
    
    Architecture:
    - Encoder: ResNet18 (512-dim features)
    - Projection head: 512 → 256 → 128 (L2 normalized)
    
    During training: Use projection head output for contrastive loss
    During inference: Discard projection head, use 512-dim encoder features
    """
    
    def __init__(self, pretrained=False):
        super().__init__()
        
        # Load ResNet18 backbone (no pretrained weights)
        resnet = models.resnet18(pretrained=pretrained)
        
        # Extract encoder (remove final FC layer)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Output: [B, 512, 1, 1]
        
        # Projection head (for contrastive learning)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        # Initialize projection head
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            return_features: If True, return encoder features instead of projection
        
        Returns:
            If return_features=False: L2-normalized projection [B, 128]
            If return_features=True: Encoder features [B, 512]
        """
        # Extract features
        features = self.encoder(x)  # [B, 512, 1, 1]
        features = features.flatten(1)  # [B, 512]
        
        if return_features:
            return features
        
        # Project and normalize
        projection = self.projection_head(features)  # [B, 128]
        projection = F.normalize(projection, dim=1)  # L2 normalize
        
        return projection
    
    def get_encoder(self):
        """Get encoder for downstream tasks (without projection head)."""
        return self.encoder


if __name__ == '__main__':
    # Test model
    print("="*70)
    print("Testing ContrastiveResNet")
    print("="*70)
    
    model = ContrastiveResNet(pretrained=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    projection_params = sum(p.numel() for p in model.projection_head.parameters())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Projection head: {projection_params:,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 96, 96)
    
    # Contrastive mode
    projection = model(x, return_features=False)
    print(f"\nContrastive mode:")
    print(f"  Input: {x.shape}")
    print(f"  Projection: {projection.shape}")
    print(f"  Norm check: {torch.norm(projection, dim=1)}")  # Should be ~1.0
    
    # Feature extraction mode
    features = model(x, return_features=True)
    print(f"\nFeature extraction mode:")
    print(f"  Features: {features.shape}")
    
    print("\n" + "="*70)
    print("✓ ContrastiveResNet working correctly!")
    print("="*70)


