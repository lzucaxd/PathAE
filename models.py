#!/usr/bin/env python3
"""
Supervised models for tumor detection.

SimpleResNet18: Standard ResNet18 for binary classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleResNet18(nn.Module):
    """
    ResNet18 for binary classification (tumor vs normal).
    
    Architecture:
    - ResNet18 backbone (from scratch, no pretrained weights)
    - Global average pooling
    - Final FC: 512 → 1 (binary classification)
    - Total parameters: ~11M
    
    Args:
        pretrained: If True, use ImageNet pretrained weights (default: False)
    """
    
    def __init__(self, pretrained=False):
        super().__init__()
        
        # Load ResNet18 (no pretrained weights)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final fully connected layer
        # Original: nn.Linear(512, 1000) for ImageNet
        # New: nn.Linear(512, 1) for binary classification
        self.backbone.fc = nn.Linear(512, 1)
        
        # Initialize new FC layer with Kaiming initialization
        nn.init.kaiming_normal_(self.backbone.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.backbone.fc.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            logits: Raw logits [B, 1] (use with BCEWithLogitsLoss)
        """
        logits = self.backbone(x)
        return logits.squeeze(1)  # [B, 1] → [B]
    
    def predict_proba(self, x):
        """
        Get probability predictions.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            probs: Probabilities [B] (0-1)
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs


if __name__ == '__main__':
    # Test model
    print("="*70)
    print("Testing SimpleResNet18")
    print("="*70)
    
    model = SimpleResNet18(pretrained=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 96, 96)
    logits = model(x)
    probs = model.predict_proba(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"Probs shape: {probs.shape}")
    print(f"Probs range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
    
    print("\n" + "="*70)
    print("✓ SimpleResNet18 working correctly!")
    print("="*70)


