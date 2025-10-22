#!/usr/bin/env python3
"""
Biologically valid augmentations for H&E histopathology.

Augmentations account for:
- Arbitrary tissue orientation (flips, rotations)
- Staining protocol variation (color jitter)
- Scanner differences (brightness/contrast)
- Focal plane variation (blur)
"""

import torchvision.transforms as transforms


def get_train_transforms(mean, std):
    """
    Get training augmentations for H&E pathology images.
    
    All augmentations are biologically plausible:
    - Geometric: Tissue orientation is arbitrary
    - Color: Staining and scanning variations exist
    - Blur: Focal plane and resolution variations
    
    Args:
        mean: RGB mean for normalization [3]
        std: RGB std for normalization [3]
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        # Geometric augmentations (arbitrary tissue orientation)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),  # Full rotation (tissue can be oriented any way)
        
        # Color augmentations (staining and scanner variation)
        transforms.ColorJitter(
            brightness=0.1,   # ±10% for scanner/illumination variation
            contrast=0.1,     # ±10% for scanner differences
            saturation=0.05,  # ±5% for H&E balance variation
            hue=0.02          # ±2% (~7 degrees) for slight stain shift
        ),
        
        # Blur (focal plane variation)
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_val_transforms(mean, std):
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        mean: RGB mean for normalization [3]
        std: RGB std for normalization [3]
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


if __name__ == '__main__':
    # Test transforms
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("Testing Biologically Valid Augmentations")
    print("="*70)
    
    # Load a sample image
    import cv2
    import pandas as pd
    
    df = pd.read_csv('final_dataset/dataset.csv')
    sample_path = 'final_dataset/' + df.iloc[0]['path']
    img_bgr = cv2.imread(sample_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Define dummy normalization
    mean = [0.5, 0.5, 0.5]
    std = [0.2, 0.2, 0.2]
    
    # Get transforms
    train_transform = get_train_transforms(mean, std)
    val_transform = get_val_transforms(mean, std)
    
    print("\nTrain transforms:")
    print(train_transform)
    
    print("\nVal transforms:")
    print(val_transform)
    
    # Apply transforms and visualize
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Augmented examples
    for i in range(1, 4):
        aug_tensor = train_transform(img_pil)
        # Denormalize for visualization
        aug_img = aug_tensor.permute(1, 2, 0).numpy()
        aug_img = aug_img * np.array(std) + np.array(mean)
        aug_img = np.clip(aug_img, 0, 1)
        
        axes[0, i].imshow(aug_img)
        axes[0, i].set_title(f'Augmented {i}', fontweight='bold')
        axes[0, i].axis('off')
    
    # More augmented examples
    for i in range(4):
        aug_tensor = train_transform(img_pil)
        aug_img = aug_tensor.permute(1, 2, 0).numpy()
        aug_img = aug_img * np.array(std) + np.array(mean)
        aug_img = np.clip(aug_img, 0, 1)
        
        axes[1, i].imshow(aug_img)
        axes[1, i].set_title(f'Augmented {i+4}', fontweight='bold')
        axes[1, i].axis('off')
    
    plt.suptitle('Biologically Valid H&E Augmentations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150)
    print("\n✓ Saved augmentation examples: augmentation_examples.png")
    
    print("\n" + "="*70)
    print("✓ Augmentations working correctly!")
    print("="*70)


