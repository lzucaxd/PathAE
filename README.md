# PathAE: Unsupervised Tumor Detection with β-VAE

**Autoencoder-based anomaly detection for histopathology whole slide images (CAMELYON16)**

Train on normal tissue (PCam), detect tumors via reconstruction error.

---

## 🎯 Quick Start

### 1. Setup
```bash
# Create conda environment
conda create -n cam16 python=3.11 -y
conda activate cam16

# Install dependencies
pip install -r requirements.txt

# OR use setup script
bash setup_vae.sh
```

### 2. Data Preprocessing (Already Done)
```bash
# Normalization stats computed from PCam normals
python compute_normalization_stats.py

# Test set created from CAMELYON16 tumor slides
python create_test_set_for_heatmaps.py
```

### 3. Train Baseline Model
```bash
# B1: VAE-Skip96, β=3 (recommended baseline)
python run_experiments.py --exp B1

# B2: VAE-Skip96, β=1 (sharper reconstructions)
python run_experiments.py --exp B2

# Monitor training
tail -f experiments/B1_VAE-Skip96-β3/training.log
```

### 4. Evaluate
```bash
# Compute threshold from training normals
python compute_threshold.py \
  --model experiments/B1_VAE-Skip96-β3/model_best.pth \
  --output experiments/B1_VAE-Skip96-β3/threshold.txt

# Run inference on test set
python run_inference_vae.py \
  --model experiments/B1_VAE-Skip96-β3/model_best.pth \
  --test-csv test_set_heatmaps/test_set.csv \
  --output experiments/B1_VAE-Skip96-β3/test_scores.csv

# Compute metrics (AUC-ROC, PR-AUC, F1, IoU)
python compute_metrics.py \
  --test-csv test_set_heatmaps/test_set.csv \
  --scores-csv experiments/B1_VAE-Skip96-β3/test_scores.csv

# Generate heatmaps
python stitch_heatmap.py \
  --test-csv test_set_heatmaps/test_set.csv \
  --scores-csv experiments/B1_VAE-Skip96-β3/test_scores.csv \
  --output-dir experiments/B1_VAE-Skip96-β3/heatmaps
```

---

## 📊 Dataset

### Training: PCam Normal Patches
- **Source**: PatchCamelyon (derived from CAMELYON16)
- **Samples**: 147,471 normal tissue patches (96×96 @ 10× magnification)
- **Splits**: Combined train + validation (unsupervised learning)

### Test: CAMELYON16 Tumor Tiles
- **Source**: 8 tumor WSIs from CAMELYON16
- **Samples**: ~20k tumor patches (96×96 @ level 2)
- **Quality Filtering**: HSV-based tissue detection, blur filtering

---

## 🏗️ Architecture: VAE-Skip96

**U-Net style β-VAE with skip connections**

```
Encoder:  96 → 48 → 24 → 12 → 6 → 3  (5× downsampling)
Channels:  3 → 64 → 128 → 256 → 256 → 256

Latent: z_ch × 3 × 3 (spatial, default z_ch=128)

Decoder: Mirror with skip connections
  - Skip e4 (6×6, 256ch) → dec5 output
  - Skip e3 (12×12, 256ch) → dec4 output
  - Skip e2 (24×24, 128ch) → dec3 output
  - Skip e1 (48×48, 64ch) → dec2 output

Norm: GroupNorm (8 groups)
Parameters: ~5.8M
```

**Key Features**:
- ✅ Skip connections → high-quality reconstructions
- ✅ Spatial latent → preserves locality for heatmaps
- ✅ GroupNorm → stable with small batches

---

## 🔬 Experiments

| ID  | β   | Description                       | Status |
|-----|-----|-----------------------------------|--------|
| B1  | 3.0 | **Baseline** (recommended)        | 🏃 Running |
| B2  | 1.0 | Lower β (sharper recon)           | ⏳ Pending |
| A1  | 3.0 | ResNet18 encoder (transfer)       | 📝 TODO |
| A2  | 3.0 | ResNet18 + Mahalanobis score      | 📝 TODO |
| P1  | 3.0 | P4M equivariant (rotation-inv)    | 📝 TODO |
| P2  | 3.0 | P4M + denoising (σ=0.03)          | 📝 TODO |

See [EXPERIMENTS_README.md](EXPERIMENTS_README.md) for full details.

---

## 📐 Loss Function

```
L = λ₁·L1 + λₛ·(1 − SSIM) + β·KL
```

- **Reconstruction**: `λ₁=0.6` (L1), `λₛ=0.4` (1-SSIM)
- **KL Divergence**: `β ∈ {1, 3}` with 10-epoch linear warm-up
- **Rationale**: Higher β → more compressed latent → better anomaly detection

---

## 🧪 Preprocessing Pipeline

### Stain Normalization
- **Primary**: Macenko (biologically relevant, separates H&E stains)
- **Fallback**: Reinhard (on failure for edge cases like fat/necrosis)
- **Reference**: Fixed tile from PCam (`reference_tile.npy`)

### RGB Normalization
- Z-score using PCam-normal mean/std (`normalization_stats.npy`)
- Applied after stain normalization

### Quality Filtering (Test Set)
```python
# HSV-based tissue detection
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
sat_blurred = cv2.GaussianBlur(hsv[:,:,1], (7,7), 0)

# Reject if:
max_sat < 0.07           # Background
mean_val < 0.1           # Too dark
mean_val > 0.9           # Overexposed
blur_variance < 30       # Out of focus
```

### Augmentation (Training Only)
- Flips: horizontal, vertical (p=0.5)
- Rotations: 90° (p=0.5)
- Color jitter: brightness ±10%, contrast ±10%, saturation ±5%, hue ±2°

---

## 📈 Evaluation Metrics

### Patch-Level Classification
- **AUC-ROC**: Discriminative ability
- **PR-AUC**: Robust to class imbalance
- **F1/Dice Score**: Harmonic mean of precision/recall
- **IoU (Jaccard)**: Spatial overlap

### Heatmap Quality (TODO)
- **Pixel-level AUC**: Using ground truth masks
- **FROC**: Free-response ROC (lesion detection)

---

## 🗂️ Repository Structure

```
PathAE/
├── model_vae_skip.py          # VAE-Skip96 architecture
├── train_vae_experiments.py   # Unified training script
├── run_experiments.py         # Experiment runner
├── dataset.py                 # PyTorch datasets
├── stain_utils.py             # Macenko/Reinhard stain norm
│
├── compute_normalization_stats.py
├── compute_threshold.py
├── run_inference_vae.py
├── compute_metrics.py
├── stitch_heatmap.py
│
├── scripts/
│   ├── convert_xml_to_mask.py
│   └── create_reference_tile.py
│
├── experiments/               # Experiment outputs
│   └── B1_VAE-Skip96-β3/
│       ├── model_best.pth
│       ├── config.json
│       ├── training.log
│       ├── reconstructions/
│       └── checkpoints/
│
├── final_dataset/             # PCam normals (train)
│   ├── dataset.csv
│   └── tiles/...
│
├── test_set_heatmaps/         # CAMELYON16 tumors (test)
│   ├── test_set.csv
│   └── tiles/...
│
├── reference_tile.npy         # Macenko reference
├── normalization_stats.npy    # PCam mean/std
│
├── EXPERIMENTS_README.md      # Detailed experiment guide
└── requirements.txt
```

---

## 🔑 Key Insights

1. **β=3 vs β=1**: Higher β compresses latent more → better anomaly detection, lower reconstruction fidelity
2. **Skip connections**: Critical for high-quality reconstructions (U-Net style)
3. **Spatial latent**: 128×3×3 preserves spatial structure for accurate heatmaps
4. **Stain normalization**: Macenko is biologically relevant; Reinhard fallback ensures robustness
5. **Augmentations**: Essential for generalization across slides/scanners

---

## 📋 Requirements

```
opencv-python
numpy
pandas
tqdm
openslide-python
torchstain>=1.3.1
scikit-image
requests

# For β-VAE training
torch
torchvision
albumentations
pytorch-msssim
scikit-learn
scipy
matplotlib
Pillow
```

---

## 🚀 Current Status

- ✅ Data preprocessing complete (PCam + CAMELYON16)
- ✅ Stain normalization fixed (Macenko with Reinhard fallback)
- ✅ VAE-Skip96 architecture implemented
- 🏃 **B1 baseline training in progress** (epoch 1/50)
- ⏳ B2 experiment pending
- 📝 A1, A2, P1, P2 require additional implementations

---

## 📚 References

- **CAMELYON16**: [https://camelyon16.grand-challenge.org/](https://camelyon16.grand-challenge.org/)
- **PatchCamelyon (PCam)**: [https://github.com/basveeling/pcam](https://github.com/basveeling/pcam)
- **β-VAE**: Higgins et al., "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"

---

## 📝 License

Research use only.
