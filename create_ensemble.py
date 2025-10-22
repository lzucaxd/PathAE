#!/usr/bin/env python3
"""
Create ensemble predictions from supervised and contrastive models.

Combines predictions and quantifies uncertainty as model disagreement.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def create_ensemble(supervised_csv, contrastive_csv, output_csv):
    """
    Create ensemble predictions with uncertainty quantification.
    
    Args:
        supervised_csv: Path to supervised predictions
        contrastive_csv: Path to contrastive predictions
        output_csv: Path to save ensemble predictions
    
    Returns:
        DataFrame with ensemble scores and uncertainty
    """
    print(f"\n{'='*70}")
    print("CREATING ENSEMBLE PREDICTIONS")
    print(f"{'='*70}\n")
    
    # Load predictions
    print(f"Loading predictions...")
    df_sup = pd.read_csv(supervised_csv)
    df_con = pd.read_csv(contrastive_csv)
    
    print(f"  Supervised: {len(df_sup):,} predictions")
    print(f"  Contrastive: {len(df_con):,} predictions")
    
    # Merge on tile_id
    print(f"\nMerging predictions...")
    df_merged = pd.merge(df_sup, df_con, on='tile_id', suffixes=('_sup', '_con'))
    
    print(f"  Merged: {len(df_merged):,} tiles")
    
    if len(df_merged) != len(df_sup):
        print(f"  ⚠ Warning: {len(df_sup) - len(df_merged)} tiles not found in both sets!")
    
    # Compute ensemble prediction (average)
    print(f"\nComputing ensemble...")
    df_merged['supervised_score'] = df_merged['score_sup']
    df_merged['contrastive_score'] = df_merged['score_con']
    df_merged['score'] = (df_merged['supervised_score'] + df_merged['contrastive_score']) / 2
    
    # Compute uncertainty (absolute disagreement)
    df_merged['uncertainty'] = np.abs(df_merged['supervised_score'] - df_merged['contrastive_score'])
    
    print(f"  ✓ Ensemble scores computed")
    print(f"  ✓ Uncertainty quantified")
    
    # Statistics
    print(f"\nEnsemble Statistics:")
    print(f"  Mean ensemble score: {df_merged['score'].mean():.4f}")
    print(f"  Mean uncertainty:    {df_merged['uncertainty'].mean():.4f}")
    print(f"  Max uncertainty:     {df_merged['uncertainty'].max():.4f}")
    
    print(f"\nUncertainty Distribution:")
    low_unc = (df_merged['uncertainty'] < 0.1).sum()
    med_unc = ((df_merged['uncertainty'] >= 0.1) & (df_merged['uncertainty'] < 0.2)).sum()
    high_unc = (df_merged['uncertainty'] >= 0.2).sum()
    
    print(f"  Low (<0.1):      {low_unc:,} ({100*low_unc/len(df_merged):.1f}%)")
    print(f"  Medium (0.1-0.2): {med_unc:,} ({100*med_unc/len(df_merged):.1f}%)")
    print(f"  High (>0.2):     {high_unc:,} ({100*high_unc/len(df_merged):.1f}%)")
    
    # Select columns for output
    output_cols = ['tile_id', 'score', 'uncertainty', 'supervised_score', 'contrastive_score']
    df_output = df_merged[output_cols]
    
    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"\n{'='*70}")
    print("✓ ENSEMBLE CREATED")
    print(f"{'='*70}\n")
    print(f"  Saved: {output_path}")
    print(f"  Columns: {', '.join(output_cols)}")
    
    return df_output


def main():
    parser = argparse.ArgumentParser(description='Create ensemble predictions')
    parser.add_argument('--supervised-csv', type=str, 
                       default='outputs/supervised_scores.csv',
                       help='Supervised predictions CSV')
    parser.add_argument('--contrastive-csv', type=str,
                       default='outputs/contrastive_scores.csv',
                       help='Contrastive predictions CSV')
    parser.add_argument('--output-csv', type=str,
                       default='outputs/scores_ensemble.csv',
                       help='Output ensemble CSV')
    args = parser.parse_args()
    
    # Create ensemble
    df_ensemble = create_ensemble(
        args.supervised_csv,
        args.contrastive_csv,
        args.output_csv
    )
    
    print(f"\n📊 Next Steps:")
    print(f"  1. Run analyze_uncertainty.py to analyze uncertainty patterns")
    print(f"  2. Run clinical_triage.py to design triage system")
    print(f"  3. Run generate_ensemble_heatmaps.py to create heatmaps")


if __name__ == '__main__':
    main()


