#!/usr/bin/env python3
"""
Grad-CAM implementation for visualizing model attention.

Reference: "Grad-CAM: Visual Explanations from Deep Networks via 
Gradient-based Localization" (Selvaraju et al., ICCV 2017)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Grad-CAM for visualizing which regions of an image are important
    for a model's prediction.
    
    Args:
        model: Neural network model
        target_layer: Layer to extract activations from (e.g., ResNet18's layer4[-1])
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to capture forward activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to capture backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: [1, 3, H, W] input image tensor
            target_class: Target class index (None for max prediction)
        
        Returns:
            cam: [H, W] normalized heatmap in [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Use sigmoid output for binary classification
        if output.shape[-1] == 1:
            pred = torch.sigmoid(output)
        else:
            pred = output
        
        # If target_class not specified, use the predicted class
        if target_class is None:
            target_class = (pred > 0.5).float()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        # Handle different output shapes
        if len(output.shape) == 2 and output.shape[-1] == 1:
            # Binary classification [1, 1] - use the raw logit
            output[0, 0].backward()
        elif len(output.shape) == 1:
            # Squeezed output [1] - use the scalar
            output[0].backward()
        else:
            # Multi-class
            output[0, target_class].backward()
        
        # Get activations and gradients
        activations = self.activations[0]  # [C, H, W]
        gradients = self.gradients[0]      # [C, H, W]
        
        # Global average pooling on gradients to get weights
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive influence)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def overlay_cam(self, image, cam, alpha=0.4):
        """
        Overlay CAM heatmap on original image.
        
        Args:
            image: [H, W, 3] RGB image (uint8)
            cam: [h, w] CAM heatmap in [0, 1]
            alpha: Transparency for overlay
        
        Returns:
            overlay: [H, W, 3] RGB image with CAM overlay
        """
        # Resize CAM to match image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap (jet)
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
        
        return overlay
    
    def __call__(self, input_tensor, target_class=None):
        """Convenience method."""
        return self.generate_cam(input_tensor, target_class)


def visualize_gradcam(model, target_layer, image_tensor, original_image, 
                      target_class=None, alpha=0.4):
    """
    End-to-end Grad-CAM visualization.
    
    Args:
        model: Neural network model
        target_layer: Layer to extract from
        image_tensor: [1, 3, H, W] preprocessed tensor
        original_image: [H, W, 3] original RGB image (uint8)
        target_class: Target class (None for predicted class)
        alpha: Overlay transparency
    
    Returns:
        cam: [H, W] heatmap
        overlay: [H, W, 3] overlay image
    """
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(image_tensor, target_class)
    overlay = gradcam.overlay_cam(original_image, cam, alpha)
    
    return cam, overlay


if __name__ == '__main__':
    """Test Grad-CAM implementation."""
    import torchvision.models as models
    
    print("Testing Grad-CAM implementation...")
    
    # Create a dummy ResNet18
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 1)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 96, 96)
    dummy_image = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
    
    # Generate Grad-CAM
    cam, overlay = visualize_gradcam(
        model, 
        model.layer4[-1],  # Target last conv layer
        dummy_input,
        dummy_image
    )
    
    print(f"✓ Grad-CAM generation successful!")
    print(f"  CAM shape: {cam.shape}")
    print(f"  CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    print(f"  Overlay shape: {overlay.shape}")
    print(f"  Overlay dtype: {overlay.dtype}")

