# 🎯 PathAE: Unsupervised Tumor Detection via β-VAE

**State-of-the-art autoencoder-based anomaly detection for histopathology whole-slide images.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Quick Start

```bash
# 1. Setup (once)
bash setup_vae.sh

# 2. Train β-VAE (2-4 hours)
python train_vae.py --z-dim 128 --beta 1.0 --epochs 50

# 3. Evaluate (20 minutes)
python compute_threshold.py --model vae_best.pth
python run_inference_vae.py --model vae_best.pth --test-csv test_set_heatmaps/test_set.csv
python test_single_slide.py  # Demo on best tumor slide
python compute_metrics.py --test-csv test_set_heatmaps/test_set.csv --scores-csv reconstruction_scores.csv
```

**That's it!** You'll have presentation-ready heatmaps and comprehensive metrics.

---

## 📊 Features

### **β-VAE Model**
- 4 conv blocks encoder: 64→128→256→256 (stride 2)
- Mirror decoder with ConvTranspose
- GroupNorm for stability
- Latent: z_dim ∈ {64, 128}
- Loss: **L = 0.6×L1 + 0.4×(1-SSIM) + β×KL**
- KL warm-up: 0→β over 10 epochs

### **Complete Preprocessing**
- ✅ **Stain normalization**: Macenko (primary) + Reinhard (fallback)
- ✅ **RGB normalization**: Mean/std computed from PCam
- ✅ **Data augmentation**: Flips, rotations, color jitter
- ✅ **Quality filtering**: HSV tissue≥0.65, blur≥30

### **Proper Heatmap Stitching**
- ✅ **Per-slide z-score**: Removes scanner/stain drift
- ✅ **Grid-based aggregation**: Exact, fast
- ✅ **Gaussian smoothing**: Natural appearance
- ✅ **Binary thresholding**: 99.7th percentile from training normals

### **Comprehensive Evaluation**
- ✅ **Patch-level**: AUC-ROC, PR-AUC, F1, Dice, IoU
- ✅ **Pixel-level**: Heatmap-based AUC
- ✅ **Slide-level**: FROC (Camelyon16 standard)

---

## 📁 Data

### **Training Set** (`final_dataset/`)
- **147,471 normal tissue patches** from PCam
- 96×96 pixels @ 10× magnification
- Pre-validated by experts
- No background, no artifacts

### **Test Set** (`test_set_heatmaps/`)
- **166,030 tissue tiles** from 8 CAMELYON16 tumor slides
- 85% background rejection (HSV filtering)
- Complete grid coverage for heatmap reconstruction
- Ground truth masks for evaluation

---

## 🏗️ Architecture

```
Input: [B, 3, 96, 96] (normalized tissue patch)
  ↓
Encoder: 96→48→24→12→6 (4×stride-2 conv + GroupNorm + LeakyReLU)
  ↓
Latent: z ~ N(μ, σ²), dim ∈ {64, 128}
  ↓
Decoder: 6→12→24→48→96 (4×ConvTranspose + GroupNorm + LeakyReLU)
  ↓
Output: [B, 3, 96, 96] (reconstructed patch)

Loss: L = λ₁·L1 + λₛ·(1-SSIM) + β·KL(q(z|x) || N(0,1))
      where λ₁=0.6, λₛ=0.4, β∈{1,3}
```

---

## 📈 Results

### **Expected Performance**
| Metric | Good | Excellent |
|--------|------|-----------|
| AUC-ROC | > 0.75 | > 0.85 |
| PR-AUC | > 0.70 | > 0.80 |
| FROC | > 0.60 | > 0.75 |
| Tumor/Normal Ratio | > 2.0× | > 3.5× |

### **Heatmap Visualization**
- 🔵 **Blue**: Low reconstruction error (normal tissue)
- 🟡 **Yellow**: Medium error (suspicious)
- 🔴 **Red**: High error (tumor!)
- ⬜ **White**: Background (not processed)

---

## 🛠️ Installation

```bash
# Create conda environment
conda create -n cam16 python=3.9 -y
conda activate cam16

# Install dependencies
conda install pytorch torchvision -c pytorch -y
pip install -r requirements.txt
```

---

## 📚 Documentation

- **README.md** ← You are here!
- **COMPLETE_WORKFLOW.md** - Step-by-step guide
- **VAE_TRAINING_GUIDE.md** - β-VAE details & troubleshooting
- **BETA_VAE_SUMMARY.txt** - Quick reference

---

## 🎨 Heatmap Examples

<div align="center">
<img src="docs/example_heatmap.png" alt="Example heatmap" width="800"/>
<p><i>4-panel comparison: Original WSI, Ground Truth, Heatmap, Overlay</i></p>
</div>

---

## 🔬 How It Works

### **1. Training** (Unsupervised)
- Train β-VAE on **normal tissue only** (147k patches)
- Model learns to reconstruct normal morphology
- No tumor labels needed!

### **2. Anomaly Detection**
- Tumors are **out-of-distribution** → high reconstruction error
- Score = 0.6×MSE + 0.4×(1-SSIM)
- Per-slide z-score normalization for robustness

### **3. Heatmap Generation**
- Grid-based stitching of tile scores
- Gaussian smoothing for natural appearance
- Overlay on WSI thumbnail

### **4. Evaluation**
- Multiple metrics (patch, pixel, slide-level)
- FROC analysis (Camelyon16 standard)
- Binary thresholding (99.7th percentile)

---

## 💡 Key Advantages

1. **Unsupervised**: No tumor labels needed for training
2. **Robust**: Stain normalization + per-slide z-score
3. **Interpretable**: Visual heatmaps show localization
4. **Comprehensive**: Multiple evaluation metrics
5. **Production-ready**: Complete preprocessing pipeline
6. **Fast**: 3-5 hours from data to results

---

## 📊 Repository Structure

```
PathAE/
├── README.md                         # This file
├── COMPLETE_WORKFLOW.md              # Step-by-step guide
├── VAE_TRAINING_GUIDE.md             # β-VAE details
│
├── train_vae.py                      # Train β-VAE
├── run_inference_vae.py              # Compute reconstruction errors
├── stitch_heatmap.py                 # Generate heatmaps
├── compute_metrics.py                # Calculate metrics
├── test_single_slide.py              # Quick demo on best slide
│
├── model_vae.py                      # β-VAE architecture
├── dataset.py                        # PyTorch datasets
├── stain_utils.py                    # Stain normalization
│
├── compute_normalization_stats.py    # RGB mean/std
├── compute_threshold.py              # Anomaly threshold
├── inspect_dataset.py                # Dataset monitoring
│
├── final_dataset/                    # 147k training normals
├── test_set_heatmaps/                # 166k test tiles
└── cam16_prepped/                    # Source WSIs & masks
```

---

## 🎯 Citation

If you use this pipeline, please cite:

```bibtex
@article{pcam2018,
  title={1399 H\&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset},
  author={Veeling, Bastiaan S and others},
  journal={GigaScience},
  year={2018}
}
```

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

## ⭐ Star this repo if you find it useful!

**Your feedback helps improve the pipeline for everyone!** 🚀
