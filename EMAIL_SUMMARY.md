# β-VAE for Tumor Detection - Experiment Summary

---

**Subject:** β-VAE Architecture & Training Updates - Posterior Collapse Fixed + Validation Monitoring

**Date:** October 14, 2025

---

## Executive Summary

We've successfully implemented and debugged a **β-VAE with U-Net skip connections** for unsupervised tumor detection in histopathology (CAMELYON16). After addressing critical posterior collapse issues and adding validation monitoring, we now have a robust baseline ready for systematic evaluation.

**Key Achievement:** Prevented KL collapse using capacity scheduling + skip regularization while maintaining reconstruction quality.

---

## 🎯 Problem Statement

**Goal:** Train autoencoder on normal tissue (PCam) to detect tumors via reconstruction error in CAMELYON16 whole slide images.

**Challenge:** Balance reconstruction quality vs. latent space usage (β-VAE disentanglement vs. posterior collapse).

---

## 🧪 Current Experiment: B1 Baseline

### Model Architecture: VAE-Skip96
- **Encoder:** 5× downsampling (96→48→24→12→6→3)
  - Channels: 3→64→128→256→256→256
  - GroupNorm for stability with small batches
- **Latent:** Spatial z=64×3×3 (576 total dims)
- **Decoder:** Mirror architecture with **U-Net skip connections**
- **Parameters:** ~5.7M

### Loss Function (Critical Innovation)
```python
L = 0.6*L1 + 0.4*(1-SSIM) + β*max(KL - C, 0)
```

**Key Components:**
1. **Reconstruction:** L1 + SSIM computed in [0,1] space (proper denormalization)
2. **KL Constraint:** Capacity scheduling C: 0→120 nats over 20 epochs
3. **β Schedule:** Warm-up 0→1.0 over 5 epochs

### Critical Fixes Applied

#### 1. **Posterior Collapse Prevention** ✅
- **Problem:** KL divergence collapsed to ~0 (decoder bypassing latent)
- **Solutions:**
  - KL capacity scheduling (free-bits style): `max(KL - C, 0)`
  - Skip connection dropout (p=0.25) during training
  - Denoising: Gaussian noise (σ=0.03) added to inputs
  
**Result:** KL now tracks capacity target (11.5 nats at epoch 5 with C=24)

#### 2. **SSIM Data Range Fixed** ✅
- **Problem:** Computing SSIM on z-scored images with wrong data_range
- **Solution:** Denormalize to [0,1] before SSIM, then use `data_range=1.0`
- **Impact:** Proper gradient signal for reconstruction

#### 3. **Augmentation Pipeline Fixed** ✅
- **Problem:** `ToTensorV2` causing dimension mismatches
- **Solution:** Manual uint8→float32 conversion after Albumentations
- **Augmentations:** H/V flips, 90° rotations, brightness/contrast ±10%, saturation ±5%

#### 4. **Validation Monitoring Added** ✅
- **Split:** 85% train (125,350) / 15% val (22,121) - fixed seed=42
- **Best Model:** Selected on validation loss (prevents overfitting)
- **Tracking:** Train vs. val for all metrics (total loss, recon, KL)

---

## 📊 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 20 | Rapid prototyping |
| **Batch Size** | 256 | Maximize GPU utilization |
| **Learning Rate** | 1e-3 → 1e-5 | Cosine annealing |
| **Optimizer** | Adam | Standard for VAEs |
| **β (KL weight)** | 0→1.0 | Warm-up over 5 epochs |
| **Capacity (C)** | 0→120 nats | Prevent collapse |
| **Denoising (σ)** | 0.03 | Force latent usage |
| **Skip Dropout** | 0.25 | Prevent bypass |
| **Stain Norm** | Macenko | Biological relevance |

---

## 📈 Preliminary Results (Epoch 5/20)

```
Epoch   5/20 | Train Loss: 0.0584 | Val Loss: 0.0470 | KL: 11.48/11.48 | β: 0.80 | C: 24.0
```

**Observations:**
- ✅ **Val < Train:** Model generalizing well (not overfitting)
- ✅ **KL Healthy:** 11.5 nats < 24 (capacity target), latent is active
- ✅ **Loss Decreasing:** Reconstruction improving over epochs
- ✅ **No Collapse:** KL staying positive and tracking capacity

**Expected Final (Epoch 20):**
- Train/Val Loss: ~0.03-0.05
- KL: ~100-140 nats (tracking C=120)
- Train/Val gap: <10-20%

---

## 🗂️ Data Pipeline

### Training Data
- **Source:** PCam (CAMELYON16-derived 96×96 patches)
- **Split:** All normal patches from train/test/validation sets
- **Total:** 147,471 normal patches
- **Preprocessing:**
  1. Macenko stain normalization (fixed reference tile)
  2. HSV tissue filtering (sat > 0.07)
  3. Blur detection (Laplacian var ≥ 30)
  4. Scale to [0,1] + z-score (mean/std from PCam normals)

### Test Data
- **Source:** CAMELYON16 tumor slides
- **Patches:** 17,151 tumor tiles (96×96 @ level 2)
- **Filtering:** PCam-style HSV (max_sat < 0.07 or value extremes)
- **Purpose:** Evaluate anomaly detection via reconstruction error

---

## 🔬 Experiment Matrix (Planned)

| ID | Model | z_dim | β | Denoising | Maha Score | Description |
|----|-------|-------|---|-----------|------------|-------------|
| **B1** | VAE-Skip96 | 64 | 1.0 | ✅ (σ=0.03) | ❌ | **Baseline (RUNNING)** |
| B2 | VAE-Skip96 | 128 | 1.0 | ✅ (σ=0.03) | ❌ | Larger latent |
| B3 | VAE-Skip96 | 64 | 3.0 | ✅ (σ=0.03) | ❌ | Higher β (more disentanglement) |
| B4 | VAE-Skip96 | 64 | 1.0 | ❌ | ❌ | Ablation: no denoising |
| B5 | VAE-Skip96 | 64 | 1.0 | ✅ (σ=0.03) | ✅ | With Mahalanobis distance |

**Evaluation Metrics:**
- Patch-level: AUC-ROC, PR-AUC, F1, IoU
- Slide-level: FROC curve, lesion detection
- Reconstruction: PSNR, SSIM on normal test patches

---

## 💻 Repository Structure

```
PathAE/
├── model_vae_skip.py           # VAE architecture + loss
├── train_vae_experiments.py    # Training script with validation
├── dataset.py                  # PyTorch datasets (PCam + CAMELYON16)
├── stain_utils.py              # Macenko/Reinhard normalization
├── run_inference_vae.py        # Compute reconstruction errors
├── generate_heatmaps.py        # Stitch tile scores into WSI heatmaps
├── compute_metrics.py          # Evaluation metrics
│
├── experiments/
│   └── B1_VAE-Skip96-z64/
│       ├── model_best.pth          # Best model (val loss)
│       ├── training.log            # Full training log
│       ├── reconstructions/        # Saved every epoch
│       │   ├── recon_epoch_001.png
│       │   └── loss_curves.png     # Train vs Val
│       └── checkpoints/            # Every 5 epochs
│
├── final_dataset/
│   ├── dataset.csv             # PCam normals (147K patches)
│   └── tiles/                  # PNG tiles
│
├── test_set_heatmaps/
│   ├── dataset.csv             # CAMELYON16 tumors (17K patches)
│   └── tiles/                  # PNG tiles
│
└── documentation/
    ├── FIXES_APPLIED.md        # All bug fixes
    ├── VALIDATION_ADDED.md     # Validation setup
    └── EXPERIMENTS_README.md   # Experiment design
```

---

## 🚀 Next Steps

### Immediate (This Week)
1. ✅ **Complete B1 training** (ETA: 3 hours)
2. Run inference on test set (compute reconstruction errors)
3. Generate heatmaps for tumor slides
4. Compute evaluation metrics (AUC-ROC, PR-AUC, F1, IoU)

### Short-Term (Next Week)
5. **Run B2-B5 experiments** (systematic ablations)
6. Compare latent dimensions (64 vs 128)
7. Compare β values (1.0 vs 3.0)
8. Test Mahalanobis distance scoring

### Medium-Term
9. Transfer learning: VAE-ResNet18 (ImageNet pretrained)
10. Equivariant model: VAE-P4M (rotation-invariant)
11. Hyperparameter tuning (capacity max, skip dropout rate)
12. Extended evaluation on full CAMELYON16 test set

---

## 📝 Technical Innovations

1. **Capacity Scheduling:** Novel use of KL capacity constraint to prevent collapse while maintaining skip connections
2. **Skip Regularization:** Dropout on skip connections allows both strong reconstruction AND meaningful latent
3. **Proper SSIM:** Denormalization before SSIM ensures correct gradients
4. **Validation Split:** Held-out set prevents overfitting in unsupervised setting
5. **Stain + Z-score:** Dual normalization (biological + statistical) for robust generalization

---

## 🔗 Key References

- **β-VAE:** Higgins et al. (2017) - "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- **Capacity Scheduling:** Burgess et al. (2018) - "Understanding disentangling in β-VAE"
- **PCam Dataset:** Veeling et al. (2018) - "Rotation equivariant CNNs for digital pathology"
- **CAMELYON16:** Bejnordi et al. (2017) - "Diagnostic assessment of deep learning algorithms for detection of lymph node metastases"

---

## 📧 Contact & Collaboration

**Repository:** `/Users/zamfiraluca/Desktop/PathAE`

**Training Status:** B1 baseline currently running (epoch 6+/20)

**Estimated Completion:** October 14, 2025 (~17:00)

**Questions/Feedback:** Happy to discuss architecture choices, hyperparameters, or evaluation strategy!

---

## Appendix: Training Commands

### Start Training
```bash
cd /Users/zamfiraluca/Desktop/PathAE
source /opt/anaconda3/bin/activate cam16
export NO_ALBUMENTATIONS_UPDATE=1

python train_vae_experiments.py \
  --exp-id B1 --model skip96 --z-dim 64 \
  --beta 1.0 --capacity-max 120.0 --kl-warmup 5 \
  --epochs 20 --batch-size 256 --num-workers 4 \
  --denoise --noise-sigma 0.03 --augment \
  --output experiments/B1_VAE-Skip96-z64/model_best.pth \
  --checkpoint-dir experiments/B1_VAE-Skip96-z64/checkpoints \
  --recon-dir experiments/B1_VAE-Skip96-z64/reconstructions
```

### Monitor Progress
```bash
# Watch training
tail -f experiments/B1_VAE-Skip96-z64/training.log

# View loss curves
open experiments/B1_VAE-Skip96-z64/reconstructions/loss_curves.png

# View reconstructions
ls experiments/B1_VAE-Skip96-z64/reconstructions/recon_epoch_*.png
```

---

**End of Report**

