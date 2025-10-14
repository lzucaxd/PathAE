# β-VAE Experiment Suite for Tumor Detection

## Overview

This repository implements a systematic evaluation of β-VAE architectures for unsupervised tumor detection in histopathology (CAMELYON16 + PCam).

**Goal**: Train on PCam normal patches, detect tumor anomalies in CAMELYON16 test slides using reconstruction error.

---

## Experiment Matrix

| ID  | Model         | β   | Denoise (σ) | Mahal | Description                           |
|-----|---------------|-----|-------------|-------|---------------------------------------|
| B1  | VAE-Skip96    | 3.0 | No          | No    | **Baseline**: U-Net skip connections  |
| B2  | VAE-Skip96    | 1.0 | No          | No    | Lower β (sharper reconstructions)     |
| A1  | VAE-ResNet18  | 3.0 | No          | No    | Transfer learning (ImageNet encoder)  |
| A2  | VAE-ResNet18  | 3.0 | No          | Yes   | Transfer + Mahalanobis scoring        |
| P1  | VAE-P4M       | 3.0 | No          | No    | Rotation-equivariant (group conv)     |
| P2  | VAE-P4M       | 3.0 | Yes (0.03)  | No    | Equivariant + denoising VAE           |

---

## Architecture Details

### VAE-Skip96 (Baseline)

**Encoder**:
```
96 → 48 → 24 → 12 → 6 → 3 (spatial latent)
3 → 64 → 128 → 256 → 256 → 256 (channels)
```

**Latent**: `z_ch × 3 × 3` (default z_ch=128)

**Decoder**: Mirror encoder with U-Net skip connections
- Skip e4 (6×6, 256ch) → dec5 output
- Skip e3 (12×12, 256ch) → dec4 output
- Skip e2 (24×24, 128ch) → dec3 output
- Skip e1 (48×48, 64ch) → dec2 output

**Normalization**: GroupNorm (8 groups, stable for small batches)

**Parameters**: ~5.8M

---

## Loss Function

```
L = λ₁·L1 + λₛ·(1 − SSIM) + β·KL
```

- **Reconstruction**: `λ₁=0.6` (L1), `λₛ=0.4` (1-SSIM)
- **KL Divergence**: `β ∈ {1, 3}` with linear warm-up over 10 epochs
- **Rationale**: Higher β → more compressed latent → better anomaly detection

---

## Training Setup

### Data
- **Train**: PCam normal patches (train + val splits combined, ~147k patches)
- **Test**: CAMELYON16 tumor tiles (20k patches from 10 tumor slides)

### Preprocessing
1. **Stain normalization**: Macenko (with Reinhard fallback for edge cases)
2. **RGB normalization**: Z-score using PCam-normal mean/std
3. **HSV filtering**: Remove background (max_sat < 0.07, mean_val ∈ [0.1, 0.9])

### Augmentation (Training Only)
- Horizontal/vertical flips (p=0.5)
- 90° rotations (p=0.5)
- Color jitter: brightness ±10%, contrast ±10%, saturation ±5%, hue ±2°

### Hyperparameters
- **Batch size**: 256 (96×96 patches)
- **Optimizer**: Adam, lr=1e-3 → 1e-5 (cosine annealing)
- **Epochs**: 50
- **Device**: MPS (Mac) or CUDA (GPU)
- **KL warm-up**: 10 epochs (0 → β linearly)

---

## Running Experiments

###1. Baseline (B1, B2)

```bash
# B1: β=3 (recommended baseline)
python run_experiments.py --exp B1

# B2: β=1 (sharper recon, weaker anomaly)
python run_experiments.py --exp B2
```

### 2. All Baselines
```bash
python run_experiments.py --exp all  # Runs B1, B2 sequentially
```

### 3. Monitor Training
```bash
# View latest reconstruction
python monitor_training.py --show-latest

# Follow training log
python monitor_training.py --follow

# Check specific experiment
tail -f experiments/B1_VAE-Skip96-β3/training.log
```

---

## Evaluation Pipeline

### 1. Compute Threshold (on training normals)
```bash
python compute_threshold.py \
  --model experiments/B1_VAE-Skip96-β3/model_best.pth \
  --data-csv final_dataset/dataset.csv \
  --output experiments/B1_VAE-Skip96-β3/threshold.txt
```

### 2. Run Inference (test set)
```bash
python run_inference_vae.py \
  --model experiments/B1_VAE-Skip96-β3/model_best.pth \
  --test-csv test_heatmap_tiles/test_tiles.csv \
  --output experiments/B1_VAE-Skip96-β3/test_scores.csv
```

### 3. Compute Metrics
```bash
python compute_metrics.py \
  --test-csv test_heatmap_tiles/test_tiles.csv \
  --scores-csv experiments/B1_VAE-Skip96-β3/test_scores.csv
```

**Metrics**:
- AUC-ROC (patch-level classification)
- PR-AUC (precision-recall, robust to imbalance)
- F1/Dice Score
- IoU (Jaccard Index)

### 4. Generate Heatmaps
```bash
python stitch_heatmap.py \
  --test-csv test_heatmap_tiles/test_tiles.csv \
  --scores-csv experiments/B1_VAE-Skip96-β3/test_scores.csv \
  --output-dir experiments/B1_VAE-Skip96-β3/heatmaps \
  --threshold experiments/B1_VAE-Skip96-β3/threshold.txt
```

---

##Fixed Issues

### ✅ Macenko Normalization
- **Problem**: Failed on low-saturation patches (background, fat, necrosis)
- **Fix**: Auto-fallback to Reinhard when Macenko fails
- **Result**: Robust stain normalization for all patches

### ✅ Stain Normalization is Biologically Relevant
- Macenko separates H&E stains (biologically meaningful)
- Reinhard is simpler but still effective
- Training uses Macenko with graceful Reinhard fallback

### ✅ Augmentations Enabled
- All training uses augmentations by default (`--augment`)
- Validation/reconstruction monitoring uses clean images (`augment=False`)

### ✅ Architecture Improvements (VAE-Skip96)
- U-Net style skip connections for better reconstructions
- GroupNorm for stability with small batches
- Spatial latent (128×3×3) preserves locality for heatmaps

---

## Next Steps

1. **Run B1, B2**: Establish baseline performance
2. **Compare β=1 vs β=3**: Evaluate reconstruction quality vs. anomaly detection
3. **Implement A1, A2** (ResNet18 transfer learning - optional)
4. **Implement P1, P2** (P4M equivariance - optional, advanced)
5. **Full evaluation**: Metrics + heatmaps for best model
6. **Demo**: Single tumor-rich slide (e.g., tumor_036) for compelling visualization

---

## File Structure

```
PathAE/
├── model_vae_skip.py          # VAE-Skip96 architecture
├── train_vae_experiments.py   # Unified training script
├── run_experiments.py         # Experiment runner
├── monitor_training.py        # Training monitoring tool
├── stain_utils.py             # Macenko/Reinhard stain norm
├── dataset.py                 # PyTorch datasets
├── compute_threshold.py       # Threshold from training normals
├── run_inference_vae.py       # Inference on test set
├── compute_metrics.py         # AUC-ROC, PR-AUC, F1, IoU
├── stitch_heatmap.py          # Heatmap generation
├── experiments/               # Experiment outputs
│   ├── B1_VAE-Skip96-β3/
│   │   ├── model_best.pth
│   │   ├── config.json
│   │   ├── training.log
│   │   ├── reconstructions/
│   │   ├── checkpoints/
│   │   └── heatmaps/
│   └── B2_VAE-Skip96-β1/
│       └── ...
├── final_dataset/             # PCam normals (train)
│   └── dataset.csv
├── test_heatmap_tiles/        # CAMELYON16 tumors (test)
│   └── test_tiles.csv
├── reference_tile.npy         # Macenko reference
└── normalization_stats.npy    # PCam mean/std
```

---

## Key Insights

1. **β=3 vs β=1**: Higher β compresses latent more → better anomaly detection at the cost of reconstruction fidelity
2. **Skip connections**: Critical for high-quality reconstructions (U-Net style)
3. **Spatial latent**: 128×3×3 preserves spatial information for accurate heatmaps
4. **Stain normalization**: Macenko is biologically relevant, Reinhard is fallback
5. **Augmentations**: Essential for generalization to new slides/scanners

---

**Status**: B1, B2 experiments ready to run. A1, A2, P1, P2 require additional model implementations (ResNet18 encoder, P4M group convolutions).

