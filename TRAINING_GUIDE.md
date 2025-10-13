# 🏋️ Training Guide: Autoencoder for Tumor Detection

## 🎯 **How the Pipeline Handles Background**

### **✅ Already Handled For You!**

**Test Set** (`test_set_heatmaps/`):
- ✅ Only **166,030 tissue tiles** saved (not 1M grid positions)
- ✅ Background tiles **filtered out** during extraction (HSV: max_sat ≥ 0.07, value ∈ [0.1, 0.9])
- ✅ Heatmap generation **only processes tissue regions** (background shown as original slide)
- ✅ Metrics **only computed on tissue tiles** (background excluded)

**Training Set** (`final_dataset/`):
- ✅ **147,471 PCam normals** - all high-quality tissue
- ✅ No background tiles included
- ✅ Pre-validated by experts

---

## 🔄 **Complete Workflow**

### **Step 1: Prepare Data Loaders**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from pathlib import Path
import numpy as np

class TissueDataset(Dataset):
    """Dataset for tissue patches (PCam normals)."""
    
    def __init__(self, csv_path, split='train', transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split]
        self.base_dir = Path(csv_path).parent
        self.transform = transform
    
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
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, img_tensor  # Input and target are the same for AE


# Create data loaders
train_dataset = TissueDataset('final_dataset/dataset.csv', split='train')
train_loader = DataLoader(
    train_dataset, 
    batch_size=128,  # Adjust based on your GPU memory
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)

print(f"Training set: {len(train_dataset):,} tissue patches")
print(f"Batches per epoch: {len(train_loader):,}")
```

---

### **Step 2: Define Autoencoder**

```python
class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for 96x96 RGB tissue patches."""
    
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # Encoder: 96x96x3 → latent_dim
        self.encoder = nn.Sequential(
            # 96x96x3 → 48x48x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 48x48x32 → 24x24x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 24x24x64 → 12x12x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 12x12x128 → 6x6x256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 6x6x256 → latent_dim
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, latent_dim),
        )
        
        # Decoder: latent_dim → 96x96x3
        self.decoder = nn.Sequential(
            # latent_dim → 6x6x256
            nn.Linear(latent_dim, 256 * 6 * 6),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 6, 6)),
            
            # 6x6x256 → 12x12x128
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 12x12x128 → 24x24x64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 24x24x64 → 48x48x32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 48x48x32 → 96x96x3
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


# Initialize model
device = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ConvAutoencoder(latent_dim=256).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

### **Step 3: Train the Model**

```python
# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

num_epochs = 50
best_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        reconstructions = model(images)
        loss = criterion(reconstructions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.6f}")
    
    # Epoch summary
    avg_loss = train_loss / len(train_loader)
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{num_epochs} Complete")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"{'='*70}\n")
    
    # Learning rate scheduling
    scheduler.step(avg_loss)
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, 'autoencoder_best.pth')
        print(f"✓ Saved best model (loss: {avg_loss:.6f})\n")

print("Training complete!")
```

---

### **Step 4: Run Inference on Test Set**

```python
# Load best model
checkpoint = torch.load('autoencoder_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test set
test_df = pd.read_csv('test_set_heatmaps/test_set.csv')
print(f"Test set: {len(test_df):,} tissue tiles")

# Compute reconstruction errors
results = []

with torch.no_grad():
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Computing errors"):
        # Load tile
        tile_path = Path('test_set_heatmaps') / row['path']
        img_bgr = cv2.imread(str(tile_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        img = img.unsqueeze(0).to(device)  # Add batch dimension
        
        # Reconstruction
        recon = model(img)
        
        # MSE per tile (reconstruction error)
        mse = ((img - recon) ** 2).mean().item()
        
        results.append({
            'tile_id': row['tile_id'],
            'score': mse
        })

# Save scores
scores_df = pd.DataFrame(results)
scores_df.to_csv('reconstruction_scores.csv', index=False)

print(f"\n✓ Reconstruction scores saved!")
print(f"  Mean error: {scores_df['score'].mean():.6f}")
print(f"  Std error: {scores_df['score'].std():.6f}")
print(f"  Min error: {scores_df['score'].min():.6f}")
print(f"  Max error: {scores_df['score'].max():.6f}")
```

---

### **Step 5: Generate Heatmaps**

```bash
python generate_heatmaps.py \
  --test-csv test_set_heatmaps/test_set.csv \
  --scores-csv reconstruction_scores.csv \
  --output-dir heatmaps
```

**What happens:**
- ✅ Loads only tissue tiles (166k)
- ✅ Creates sparse grid (tissue positions only)
- ✅ Background regions show original slide (not colored)
- ✅ Tumor regions show high error (red/yellow in jet colormap)

---

### **Step 6: Compute Metrics**

```bash
python compute_metrics.py \
  --test-csv test_set_heatmaps/test_set.csv \
  --scores-csv reconstruction_scores.csv
```

**What happens:**
- ✅ Only evaluates on tissue tiles
- ✅ Background excluded from metrics
- ✅ Reports AUC-ROC, PR-AUC, F1, Dice, IoU, FROC

---

## 💡 **Key Points**

### **Why Background Doesn't Confuse the Model**

1. **Training**: Only trained on tissue (PCam) - never sees background
2. **Inference**: Only run on tissue tiles - background not processed
3. **Heatmaps**: Background regions left as original slide image
4. **Metrics**: Only computed on tissue tiles - background excluded

### **What to Expect**

**Normal Tissue** (training distribution):
- Low reconstruction error (~0.001-0.01 MSE)
- Model reconstructs well (trained on this)

**Tumor Tissue** (anomaly):
- High reconstruction error (~0.05-0.2 MSE)
- Model fails to reconstruct (never saw this)

**Background** (never processed):
- No score computed
- Shown as original slide in heatmaps
- Excluded from metrics

---

## 🎯 **Expected Performance**

### Training (147k normals)
- **Convergence**: 20-30 epochs typically
- **Loss**: Should reach ~0.001-0.005 MSE
- **Time**: 2-4 hours on M1/M2 MPS, 1-2 hours on GPU

### Evaluation (166k tissue tiles)
- **Good Model**: AUC > 0.75, PR-AUC > 0.70
- **Excellent Model**: AUC > 0.85, PR-AUC > 0.80
- **FROC**: Sensitivity > 0.60 at 1 FP/slide

---

## 🚀 **Quick Start Script**

Save as `train_autoencoder.py`:

```python
#!/usr/bin/env python3
"""
Train autoencoder on tissue patches for tumor detection.
Usage: python train_autoencoder.py
"""

# Paste the complete code from Steps 1-4 above
# Then run: python train_autoencoder.py

if __name__ == '__main__':
    main()
```

---

## ✅ **Summary**

Your pipeline is **ready** and properly handles background:

1. ✅ Training on **tissue only** (PCam)
2. ✅ Inference on **tissue only** (filtered test set)
3. ✅ Heatmaps show **tissue regions only**
4. ✅ Metrics computed on **tissue only**

**No manual background detection needed!** The HSV filtering during test set creation already did this for you.

Train your model and enjoy beautiful heatmaps! 🎨

