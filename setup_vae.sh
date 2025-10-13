#!/bin/bash
# Setup script for β-VAE training environment

echo "======================================================================"
echo "PathAE β-VAE Setup"
echo "======================================================================"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /opt/anaconda3/bin/activate cam16

# Install PyTorch (if not already installed)
echo ""
echo "Checking PyTorch installation..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')" 2>/dev/null || {
    echo "Installing PyTorch..."
    # For Mac with MPS
    pip install torch torchvision torchaudio
}

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install -q albumentations pytorch-msssim

echo ""
echo "======================================================================"
echo "Preprocessing"
echo "======================================================================"
echo ""

# Create reference tile
if [ ! -f "reference_tile.npy" ]; then
    echo "Creating reference tile for stain normalization..."
    python stain_utils.py --csv final_dataset/dataset.csv --output reference_tile.npy --n-samples 10
else
    echo "✓ Reference tile already exists"
fi

# Compute normalization stats
if [ ! -f "normalization_stats.npy" ]; then
    echo ""
    echo "Computing RGB normalization statistics..."
    python compute_normalization_stats.py --csv final_dataset/dataset.csv --split train --max-samples 10000
else
    echo "✓ Normalization stats already exist"
fi

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "✓ Reference tile: reference_tile.npy"
echo "✓ Normalization stats: normalization_stats.npy"
echo ""
echo "Ready to train! Run:"
echo "  python train_vae.py --z-dim 128 --beta 1.0 --epochs 50"
echo ""
echo "Or with grid search:"
echo "  python train_vae.py --z-dim 64 --beta 1.0 --epochs 50"
echo "  python train_vae.py --z-dim 128 --beta 1.0 --epochs 50"
echo "  python train_vae.py --z-dim 128 --beta 3.0 --epochs 50"
echo ""

