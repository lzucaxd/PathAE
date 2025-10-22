#!/usr/bin/env python3
"""
Display all supervised model heatmaps in a grid.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from PIL import Image

# Load heatmaps
heatmap_dir = Path('outputs/supervised_heatmaps')
heatmap_files = sorted(heatmap_dir.glob('*_heatmap.png'))

print(f"Found {len(heatmap_files)} heatmaps:")
for f in heatmap_files:
    print(f"  • {f.name}")

# Load metrics
metrics_df = pd.read_csv(heatmap_dir / 'heatmap_metrics.csv')
print("\nHeatmap Metrics:")
print(metrics_df.to_string(index=False))

# Create grid visualization
n_heatmaps = len(heatmap_files)
n_cols = 2
n_rows = (n_heatmaps + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten() if n_heatmaps > 1 else [axes]

for idx, heatmap_file in enumerate(heatmap_files):
    img = Image.open(heatmap_file)
    axes[idx].imshow(img)
    axes[idx].set_title(heatmap_file.stem.replace('_heatmap', ''), 
                        fontsize=14, fontweight='bold')
    axes[idx].axis('off')
    
    # Add metrics text
    wsi_id = heatmap_file.stem.replace('_heatmap', '')
    row = metrics_df[metrics_df['wsi_id'] == wsi_id]
    if not row.empty:
        mean_score = row['mean_score'].values[0]
        n_tiles = row['n_tiles'].values[0]
        text = f'Mean score: {mean_score:.4f}\nTiles: {n_tiles:,}'
        axes[idx].text(0.02, 0.98, text, transform=axes[idx].transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Hide empty subplots
for idx in range(len(heatmap_files), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Supervised ResNet18 Heatmaps (CAMELYON16 Test Set)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_path = 'outputs/supervised_heatmaps_summary.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Summary saved: {output_path}")

plt.show()


