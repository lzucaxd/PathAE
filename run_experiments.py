#!/usr/bin/env python3
"""
Run full experiment suite for β-VAE tumor detection.

Experiment Matrix (6 runs):
  B1: VAE-Skip96, β=3, no denoise, no Maha
  B2: VAE-Skip96, β=1, no denoise, no Maha  
  A1: VAE-ResNet18, β=3, no denoise, no Maha
  A2: VAE-ResNet18, β=3, no denoise, with Maha
  P1: VAE-P4M, β=3, no denoise, no Maha
  P2: VAE-P4M, β=3, with denoise (σ=0.03), no Maha
"""

import argparse
import subprocess
import json
from pathlib import Path
import time

# Experiment configurations
EXPERIMENTS = {
    'B1': {
        'name': 'VAE-Skip96-β3',
        'model': 'skip96',
        'z_dim': 128,
        'beta': 3.0,
        'denoise': False,
        'use_maha': False,
        'desc': 'Baseline: Skip96, β=3, standard setup'
    },
    'B2': {
        'name': 'VAE-Skip96-β1',
        'model': 'skip96',
        'z_dim': 128,
        'beta': 1.0,
        'denoise': False,
        'use_maha': False,
        'desc': 'Baseline: Skip96, β=1, sharper reconstructions'
    },
    'A1': {
        'name': 'VAE-ResNet18-β3',
        'model': 'resnet18',
        'z_dim': 128,
        'beta': 3.0,
        'denoise': False,
        'use_maha': False,
        'desc': 'Transfer: ResNet18 encoder, β=3'
    },
    'A2': {
        'name': 'VAE-ResNet18-β3-Maha',
        'model': 'resnet18',
        'z_dim': 128,
        'beta': 3.0,
        'denoise': False,
        'use_maha': True,
        'desc': 'Transfer: ResNet18, β=3, + Mahalanobis score'
    },
    'P1': {
        'name': 'VAE-P4M-β3',
        'model': 'p4m',
        'z_dim': 128,
        'beta': 3.0,
        'denoise': False,
        'use_maha': False,
        'desc': 'Equivariant: P4M group conv, β=3'
    },
    'P2': {
        'name': 'VAE-P4M-β3-Denoise',
        'model': 'p4m',
        'z_dim': 128,
        'beta': 3.0,
        'denoise': True,
        'use_maha': False,
        'desc': 'Equivariant: P4M, β=3, + denoising (σ=0.03)'
    },
}


def run_experiment(exp_id, config, dry_run=False):
    """Run a single experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_id}: {config['name']}")
    print(f"{'='*70}")
    print(f"Description: {config['desc']}")
    print(f"Model: {config['model']}, z_dim={config['z_dim']}, β={config['beta']}")
    print(f"Denoise: {config['denoise']}, Mahalanobis: {config['use_maha']}")
    print(f"{'='*70}\n")
    
    # Create experiment directory
    exp_dir = Path(f"experiments/{exp_id}_{config['name']}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Build training command
    cmd = [
        'python', 'train_vae_experiments.py',
        '--exp-id', exp_id,
        '--model', config['model'],
        '--z-dim', str(config['z_dim']),
        '--beta', str(config['beta']),
        '--epochs', '50',
        '--batch-size', '256',  # Larger batch for 96×96
        '--lr', '1e-3',
        '--num-workers', '4',
        '--augment',
        '--output', str(exp_dir / 'model_best.pth'),
        '--checkpoint-dir', str(exp_dir / 'checkpoints'),
        '--recon-dir', str(exp_dir / 'reconstructions'),
    ]
    
    if config['denoise']:
        cmd.append('--denoise')
        cmd.extend(['--noise-sigma', '0.03'])
    
    # Log file
    log_file = exp_dir / 'training.log'
    
    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        print(f"Log: {log_file}\n")
        return
    
    print(f"Starting training...")
    print(f"Log: {log_file}\n")
    
    with open(log_file, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    print(f"✓ Experiment {exp_id} complete!\n")


def main():
    parser = argparse.ArgumentParser(description='Run VAE experiment suite')
    parser.add_argument('--exp', type=str, default='all',
                        help='Experiment ID to run (B1, B2, A1, A2, P1, P2, or "all")')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without running')
    parser.add_argument('--sequential', action='store_true', default=True,
                        help='Run experiments sequentially (default)')
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if args.exp == 'all':
        exp_list = ['B1', 'B2']  # Start with baselines only
        print("\n🚀 Running BASELINE experiments (B1, B2)")
        print("   (A1, A2, P1, P2 require additional model implementations)\n")
    else:
        exp_list = [args.exp]
    
    # Run experiments
    for exp_id in exp_list:
        if exp_id not in EXPERIMENTS:
            print(f"Error: Unknown experiment '{exp_id}'")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            return
        
        config = EXPERIMENTS[exp_id]
        run_experiment(exp_id, config, dry_run=args.dry_run)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Compare reconstructions in experiments/*/reconstructions/")
    print("  2. Run inference: python run_inference_experiments.py")
    print("  3. Generate heatmaps and compute metrics")
    print()


if __name__ == '__main__':
    main()

