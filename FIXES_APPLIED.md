# β-VAE Training Fixes Applied

## Critical Fixes for Posterior Collapse

### 1. **KL Capacity Scheduling** ✅
- **Problem**: KL divergence was collapsing to ~0 (posterior collapse)
- **Solution**: Added `KLCapacity` scheduler with free-bits style constraint
- **Implementation**: `L = recon + β*max(KL - C, 0)`
- **Settings**: C linearly increases from 0→120 nats over 20 epochs
- **Result**: Forces model to use latent space

### 2. **Skip Connection Regularization** ✅
- **Problem**: U-Net skips allowing decoder to bypass latent entirely
- **Solution**: Added dropout (p=0.25) to skip connections during training
- **Implementation**: `F.dropout(skip, p=0.25, training=self.training)` in `DecoderBlock`
- **Result**: Prevents decoder from memorizing through skips

### 3. **Denoising** ✅
- **Problem**: Model could memorize pixel patterns without using latent
- **Solution**: Added Gaussian noise (σ=0.03) to inputs during training
- **Implementation**: `add_noise(x, sigma=0.03)` before forward pass
- **Result**: Forces latent to model signal, not memorize pixels

### 4. **SSIM Data Range Fixed** ✅
- **Problem**: Computing SSIM on z-scored images with wrong `data_range=3.0`
- **Solution**: Denormalize to [0,1] before computing SSIM
- **Implementation**: 
  ```python
  recon_01 = torch.clamp(recon * std + mean, 0, 1)
  x_01 = torch.clamp(x * std + mean, 0, 1)
  ssim_val = ssim(recon_01, x_01, data_range=1.0)
  ```
- **Result**: Proper SSIM loss computation

### 5. **Augmentations Fixed** ✅
- **Problem**: `ToTensorV2` causing double normalization; `HueSaturationValue` failing
- **Solution**: Removed `ToTensorV2`, handle conversion manually
- **Implementation**: Apply augmentations on uint8 [0,255], then convert to [0,1]
- **Result**: Proper color/geometric augmentations

### 6. **Macenko Normalization with Fallback** ✅
- **Problem**: Macenko failing on edge cases
- **Solution**: Robust try-except with automatic fallback to Reinhard
- **Implementation**: Catch all torchstain errors and fallback gracefully
- **Result**: No stain norm failures during training

---

## Training Configuration (B1 Baseline)

### Model Architecture
- **Name**: VAE-Skip96
- **Latent**: z_ch=64 (spatial 64×3×3 = 576 dims)
- **Encoder**: 5× downsampling (96→48→24→12→6→3)
- **Channels**: 3→64→128→256→256→256
- **Decoder**: Mirror with skip connections + dropout
- **Parameters**: ~5.7M

### Loss Function
```python
L = 0.6*L1 + 0.4*(1-SSIM) + β*max(KL - C, 0)
```
- **L1 & SSIM**: Computed in [0,1] space (after denormalization)
- **β**: 0→1.0 linearly over 5 epochs (KL warm-up)
- **C (capacity)**: 0→120 nats linearly over 20 epochs

### Training Settings
- **Epochs**: 20 (rapid prototyping)
- **Batch size**: 256
- **Learning rate**: 1e-3 (Adam) with cosine annealing →1e-5
- **Denoising**: σ=0.03 Gaussian noise
- **Stain norm**: Macenko (with Reinhard fallback)
- **RGB norm**: PCam-normal mean/std

### Data Augmentation
- **Geometric**: H/V flips (p=0.5), 90° rotations (p=0.5)
- **Color**: 
  - Brightness/contrast ±10% (p=0.5)
  - Saturation ±5% (p=0.3)
  - Hue ±2° (p=0.3)

### Environment
- **Device**: MPS (Apple Silicon) or CUDA
- **Workers**: 4
- **Environment**: `export NO_ALBUMENTATIONS_UPDATE=1`

---

## Expected Behavior (No Collapse)

### Healthy KL Trajectory
```
Epoch 1:  β=0.20, C=6.0,  KL ≈ 10-20   (low β, small C)
Epoch 5:  β=1.00, C=30.0, KL ≈ 40-60   (β stable, C rising)
Epoch 10: β=1.00, C=60.0, KL ≈ 70-90   (KL tracking capacity)
Epoch 20: β=1.00, C=120.0, KL ≈ 100-140 (converged near capacity)
```

### What to Monitor
1. **KL should track capacity**: KL ≈ C ± 20 nats
2. **Reconstruction improving**: Recon loss decreasing over time
3. **Active latent units**: Var(μ) > 1e-3 for most channels
4. **Reconstructions**: Should show tissue structure, not blur

---

## Next Steps (After B1 Complete)

### B2: β=1.0 → β=3.0
- Same as B1 but sweep β ∈ {1, 3}
- Expected: Higher β → better anomaly separation

### Sweep z_ch
- Test: {32, 64, 96, 128}
- Expected: 64-96 sweet spot for 96×96 patches

### Ablations
- No skip dropout
- No denoising
- No capacity (pure β-VAE)

---

## Files Modified

### Core Models
- `model_vae_skip.py`: Added skip dropout, capacity scheduling, fixed loss
- `train_vae_experiments.py`: Updated training loop with all fixes
- `dataset.py`: Fixed augmentations (removed ToTensorV2)
- `stain_utils.py`: Robust Macenko with fallback

### Experiment Structure
- `experiments/B1_VAE-Skip96-z64/`: B1 baseline run
- `EXPERIMENTS_README.md`: Full experiment matrix
- `FIXES_APPLIED.md`: This document

---

## Quick Test Command

```bash
# Test model forward pass
python model_vae_skip.py

# Start B1 training (20 epochs)
cd /Users/zamfiraluca/Desktop/PathAE && \
source /opt/anaconda3/bin/activate cam16 && \
export NO_ALBUMENTATIONS_UPDATE=1 && \
python train_vae_experiments.py \
  --exp-id B1 --model skip96 --z-dim 64 \
  --beta 1.0 --capacity-max 120.0 --kl-warmup 5 \
  --epochs 20 --batch-size 256 --num-workers 4 \
  --denoise --noise-sigma 0.03 --augment \
  --output experiments/B1_VAE-Skip96-z64/model_best.pth \
  --checkpoint-dir experiments/B1_VAE-Skip96-z64/checkpoints \
  --recon-dir experiments/B1_VAE-Skip96-z64/reconstructions \
  2>&1 | tee experiments/B1_VAE-Skip96-z64/training.log &

# Monitor progress
tail -f experiments/B1_VAE-Skip96-z64/training.log
```

---

## Status

- ✅ All critical fixes applied
- ✅ Model tested successfully
- ✅ B1 training running (ETA: ~3-4 hours for 20 epochs)
- ⏳ Waiting for first epoch completion...

