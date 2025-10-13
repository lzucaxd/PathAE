# 🎯 Complete β-VAE Workflow: End-to-End

## ✅ **Current Status**

**Data Ready:**
- ✅ Training: 147,471 PCam normals
- ✅ Test: 166,030 tissue tiles from 8 tumor slides
- ✅ Reference tile: `reference_tile.npy`
- ✅ Normalization stats: `normalization_stats.npy`

**Code Ready:**
- ✅ β-VAE model with proper architecture
- ✅ Complete preprocessing pipeline
- ✅ Proper heatmap stitching (grid-based, per-slide z-score)
- ✅ All evaluation metrics

---

## 🚀 **Complete Workflow** (5 Steps)

### **Step 1: Train β-VAE** (2-4 hours)

```bash
# Recommended configuration:
python train_vae.py \
  --z-dim 128 \
  --beta 1.0 \
  --epochs 50 \
  --batch-size 128 \
  --output vae_best.pth
```

**What happens:**
- Loads 147k PCam normals
- Applies stain normalization (Macenko)
- RGB normalization (mean=[0.182], std=[0.427])
- Data augmentation (flips, rotations, color jitter)
- Trains β-VAE with L = 0.6*L1 + 0.4*(1-SSIM) + β*KL
- KL warm-up over 10 epochs (0→1.0)

**Output:** `vae_best.pth`

---

### **Step 2: Compute Threshold** (5 min)

```bash
python compute_threshold.py \
  --model vae_best.pth \
  --csv final_dataset/dataset.csv \
  --n-samples 5000 \
  --percentile 99.7
```

**What happens:**
- Runs inference on 5000 training normals
- Computes reconstruction errors
- Sets threshold at 99.7th percentile (3σ)

**Output:** `threshold.npy`

**Why?** Defines what "anomaly" means (top 0.3% of normal errors)

---

### **Step 3: Run Inference on Test Set** (10-15 min)

```bash
python run_inference_vae.py \
  --model vae_best.pth \
  --test-csv test_set_heatmaps/test_set.csv \
  --output reconstruction_scores.csv
```

**What happens:**
- Applies same preprocessing (stain norm + RGB norm)
- Computes score = 0.6*MSE + 0.4*(1-SSIM) per tile
- Saves to CSV

**Output:** `reconstruction_scores.csv`

**Expected:**
- Normal tissue: low scores (~0.01-0.05)
- Tumor tissue: high scores (~0.10-0.30)
- Ratio: 2-5× higher for tumors

---

### **Step 4: Generate Heatmaps** (5-10 min)

#### **Option A: Single Tumor-Heavy Slide (For Demo)**

```bash
python test_single_slide.py \
  --test-csv test_set_heatmaps/test_set.csv \
  --scores-csv reconstruction_scores.csv \
  --output-dir demo_heatmap
```

**This:**
- Automatically selects slide with most tumor (tumor_036: 881 tumor tiles)
- Applies per-slide z-score normalization
- Creates grid-based heatmap with Gaussian smoothing
- Generates 5-panel comparison (original, GT, heatmap, overlay, binary)

**Perfect for presentations!**

#### **Option B: All 8 Slides**

```bash
python stitch_heatmap.py \
  --test-csv test_set_heatmaps/test_set.csv \
  --scores-csv reconstruction_scores.csv \
  --output-dir heatmaps_v2
```

---

### **Step 5: Compute Metrics** (2-3 min)

```bash
python compute_metrics.py \
  --test-csv test_set_heatmaps/test_set.csv \
  --scores-csv reconstruction_scores.csv
```

**Metrics:**
- AUC-ROC, PR-AUC (classification)
- F1, Dice, IoU (segmentation)
- FROC (Camelyon16 standard)
- Pixel-level AUC

**Output:**
- `evaluation_summary.csv`
- `froc_curve.png`

---

## 📊 **Heatmap Stitching Algorithm**

### **Step-by-Step Process**

```python
# 1. Compute score per tile
score = 0.6 * MSE(x, x̂) + 0.4 * (1 - SSIM(x, x̂))

# 2. Per-slide z-score normalization
s' = (s - mean_wsi) / (std_wsi + 1e-6)
# This suppresses slide-to-slide stain/scanner drift

# 3. Stitch into grid
heatmap_grid[row_idx, col_idx] = s'  # or max if overlapping

# 4. Gaussian smoothing (smooth seams)
heatmap_smooth = gaussian_filter(heatmap_grid, sigma=2.0)

# 5. Min-max to [0,1] for visualization
heatmap_vis = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

# 6. Apply colormap (jet)
heatmap_colored = jet_colormap(heatmap_vis)

# 7. Overlay on thumbnail
overlay = alpha*heatmap_colored + (1-alpha)*thumbnail

# 8. Binary map (using threshold)
binary_map = (scores >= threshold)
```

**Key features:**
- ✅ Per-slide normalization (removes scanner differences)
- ✅ Grid-based stitching (exact, fast)
- ✅ Gaussian smoothing (natural appearance)
- ✅ Background handling (NaN for non-tissue)

---

## 🎨 **Demo Workflow** (For Presentation)

### **Quick Test on Best Slide**

```bash
# After training and inference:
python test_single_slide.py

# This automatically:
# 1. Selects slide with most tumor (tumor_036: 881 tumor tiles)
# 2. Generates beautiful heatmap
# 3. Saves to demo_heatmap/
```

**Perfect for showing in your presentation!**

---

## 📈 **Expected Timeline**

```
Setup (once):         5 min   (bash setup_vae.sh)
Training:             2-4h    (python train_vae.py)
Compute threshold:    5 min   (python compute_threshold.py)
Inference (test):     10-15min (python run_inference_vae.py)
Demo heatmap:         2-3min  (python test_single_slide.py)
All heatmaps:         5-10min (python stitch_heatmap.py)
Metrics:              2-3min  (python compute_metrics.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:                3-5h    (mostly training!)
```

---

## 🔧 **Dataset Monitoring**

### **Inspect Dataset Quality**

```bash
# Quick statistics
python inspect_dataset.py --stats

# Visualize samples
python inspect_dataset.py --visualize --split train

# Visualize with stain normalization
python inspect_dataset.py --visualize --stain-norm

# Test preprocessing pipeline
python inspect_dataset.py --test-pipeline

# Run all checks
python inspect_dataset.py --all
```

**This creates:**
- `dataset_samples_train.png` - Grid of 16 random patches
- Statistics printout (counts by split, slide, label)
- Preprocessing verification

---

## 💡 **Preprocessing Details**

### **Complete Pipeline Per Image:**

```
1. Load RGB (96×96, uint8)
2. Stain norm (Macenko → reference_tile.npy)
3. Scale [0, 1]
4. Augment (train only):
   - H/V flips (50%)
   - 90° rotation (50%)
   - Brightness/contrast ±10% (50%)
   - Saturation ±5% (50%)
   - Hue ±2° (50%)
5. To tensor [3, 96, 96]
6. RGB normalize: (x - mean) / std
   mean = [0.182, 0.182, 0.182]
   std  = [0.427, 0.427, 0.427]
7. Feed to β-VAE
```

### **Quality Filters (Optional)**

Already applied during test set creation, but can verify:
- Tissue fraction ≥ 0.65 (HSV saturation > 0.07)
- Blur variance ≥ 30 (Laplacian)

---

## 🎯 **Best Slides for Demo**

Based on tumor content:

| Rank | Slide | Tumor Tiles | Total Tiles | % Tumor | Quality |
|------|-------|-------------|-------------|---------|---------|
| 1 | **tumor_036** | **881** | 22,930 | **3.8%** | ⭐ **Best for demo!** |
| 2 | **tumor_020** | **706** | 13,334 | **5.3%** | ⭐ **Good** |
| 3 | tumor_056 | 320 | 11,360 | 2.8% | Good |
| 4 | test_002 | 304 | 14,098 | 2.2% | Good |

**Recommendation:** Use `tumor_036` or `tumor_020` for presentation - they have the most visible tumors!

---

## 📊 **Heatmap Interpretation**

### **Colors (Jet Colormap)**
- 🔵 **Blue**: Low z-score (normal tissue)
- 🟢 **Green**: Medium z-score
- 🟡 **Yellow**: High z-score (suspicious)
- 🔴 **Red**: Very high z-score (tumor!)

### **Per-Slide Z-Score**
```
z = (score - slide_mean) / slide_std

Why? Removes scanner/stain variability
- Each slide gets own mean/std
- Anomalies are relative to that slide's baseline
- More robust than global normalization
```

### **Binary Map (Thresholded)**
- Yellow overlay: Detected tumors (score > threshold)
- Threshold = 99.7th percentile of training normals
- ~0.3% false positive rate on normals

---

## ✅ **Complete Checklist**

**Setup** (once):
- [x] Reference tile created
- [x] Normalization stats computed
- [x] Dependencies installed

**Training**:
- [ ] Train β-VAE (~2-4h)
- [ ] Compute threshold from normals (~5min)

**Evaluation**:
- [ ] Run inference on test set (~10-15min)
- [ ] Generate demo heatmap (~2-3min)
- [ ] Generate all heatmaps (~5-10min)
- [ ] Compute metrics (~2-3min)

---

## 🚀 **Quick Start Commands**

```bash
# 1. Train
python train_vae.py --z-dim 128 --beta 1.0 --epochs 50

# 2. Threshold
python compute_threshold.py --model vae_best.pth

# 3. Inference
python run_inference_vae.py --model vae_best.pth --test-csv test_set_heatmaps/test_set.csv

# 4. Demo (best tumor slide)
python test_single_slide.py

# 5. All heatmaps
python stitch_heatmap.py --test-csv test_set_heatmaps/test_set.csv --scores-csv reconstruction_scores.csv

# 6. Metrics
python compute_metrics.py --test-csv test_set_heatmaps/test_set.csv --scores-csv reconstruction_scores.csv
```

---

## 🎤 **For Your Presentation**

### **Show:**
1. Training curve (loss over epochs)
2. Demo heatmap (tumor_036 - most tumor content)
3. FROC curve
4. Metrics table

### **Key Points:**
- "Unsupervised β-VAE trained on 147k normal tissue patches"
- "Per-slide z-score normalization for robustness"
- "Grid-based heatmap stitching with Gaussian smoothing"
- "Achieves [your AUC] tumor detection with FROC evaluation"

---

## 💡 **Pro Tips**

### **Improving Heatmap Quality**
- Increase smoothing: `--smooth-sigma 3.0`
- Higher resolution: `--canvas-level 3` (larger image)
- Adjust transparency: `--alpha 0.6`

### **If Performance is Low**
- Try β=3.0 (more regularization)
- Train longer (100 epochs)
- Increase z_dim to 256
- Check stain normalization is working

### **Debugging**
```bash
# Inspect dataset
python inspect_dataset.py --all

# Test single slide
python test_single_slide.py --wsi-id tumor_036

# Check threshold
python -c "import numpy as np; t=np.load('threshold.npy', allow_pickle=True).item(); print(f'Threshold: {t[\"threshold\"]:.6f}')"
```

---

## 🎯 **Your Pipeline is Production-Ready!**

**All preprocessing automated:**
- ✅ Stain normalization (Macenko)
- ✅ RGB normalization (PCam stats)
- ✅ Quality filtering (HSV + blur)
- ✅ Data augmentation

**Proper heatmap stitching:**
- ✅ Per-slide z-score (removes scanner drift)
- ✅ Grid-based aggregation (exact)
- ✅ Gaussian smoothing (natural)
- ✅ Binary thresholding (99.7th percentile)

**Comprehensive evaluation:**
- ✅ 7 metrics (AUC-ROC, PR-AUC, F1, Dice, IoU, FROC, Pixel-AUC)
- ✅ Publication-quality figures
- ✅ Ready for demo!

**Just start training!** 🚀

