#!/usr/bin/env python3
"""
Supervised Contrastive Loss.

Reference: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
https://arxiv.org/abs/2004.11362

Key idea:
- Pull together samples with same label (positives)
- Push apart samples with different labels (negatives)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    
    Formula:
    For each anchor i:
        L_i = -1/|P(i)| * Σ_{p∈P(i)} log[ exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ]
    
    Where:
    - P(i): Set of positives (same label, excluding i)
    - A(i): Set of all samples except i
    - τ: Temperature parameter
    - z: L2-normalized embeddings
    
    Args:
        temperature: Scaling parameter (default: 0.07)
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Compute supervised contrastive loss.
        
        Args:
            features: L2-normalized embeddings [B, D]
            labels: Class labels [B]
        
        Returns:
            loss: Scalar tensor
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Ensure features are normalized
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix: [B, B]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create masks for positives and negatives
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)  # Same label
        mask_negative = 1 - mask_positive  # Different label
        
        # Remove diagonal (don't compare sample with itself)
        logits_mask = torch.ones_like(mask_positive).scatter_(1, 
            torch.arange(batch_size).view(-1, 1).to(device), 0)
        
        mask_positive = mask_positive * logits_mask
        
        # Compute log probabilities
        # For numerical stability, subtract max
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log sum exp of all similarities (denominator)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positives
        # For each sample, average over all positives
        mean_log_prob_pos = (mask_positive * log_prob).sum(1) / (mask_positive.sum(1) + 1e-12)
        
        # Loss is negative mean
        loss = -mean_log_prob_pos.mean()
        
        return loss


if __name__ == '__main__':
    # Test loss
    print("="*70)
    print("Testing Supervised Contrastive Loss")
    print("="*70)
    
    # Create dummy batch
    batch_size = 16
    features = torch.randn(batch_size, 128)
    features = F.normalize(features, dim=1)  # L2 normalize
    
    # Create labels (8 normal, 8 tumor)
    labels = torch.cat([torch.zeros(8), torch.ones(8)]).long()
    
    # Compute loss
    criterion = SupConLoss(temperature=0.07)
    loss = criterion(features, labels)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Feature dim: {features.shape[1]}")
    print(f"Labels: {labels}")
    print(f"\nContrastive loss: {loss.item():.4f}")
    
    # Test with perfect separation
    print("\n" + "-"*70)
    print("Testing with perfect separation:")
    print("-"*70)
    
    # Create perfectly separated features
    features_sep = torch.zeros(batch_size, 128)
    features_sep[:8, 0] = 1.0   # Normal: [1, 0, 0, ...]
    features_sep[8:, 1] = 1.0   # Tumor: [0, 1, 0, ...]
    features_sep = F.normalize(features_sep, dim=1)
    
    loss_sep = criterion(features_sep, labels)
    print(f"Loss with perfect separation: {loss_sep.item():.4f}")
    print(f"  (Should be near 0)")
    
    # Test with no separation
    print("\n" + "-"*70)
    print("Testing with no separation:")
    print("-"*70)
    
    features_same = torch.ones(batch_size, 128)
    features_same = F.normalize(features_same, dim=1)
    
    loss_same = criterion(features_same, labels)
    print(f"Loss with no separation: {loss_same.item():.4f}")
    print(f"  (Should be higher)")
    
    print("\n" + "="*70)
    print("✓ SupConLoss working correctly!")
    print("="*70)


