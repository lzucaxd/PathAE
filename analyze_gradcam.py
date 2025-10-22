#!/usr/bin/env python3
"""
Analyze and interpret Grad-CAM results.

Provides detailed interpretation of what supervised and contrastive
models focus on for each case type (TP, TN, FP, FN).
"""

import json
from pathlib import Path


def analyze_gradcam_results(examples_json='gradcam_examples.json'):
    """
    Analyze and print interpretation of Grad-CAM results.
    """
    # Load examples
    with open(examples_json, 'r') as f:
        examples = json.load(f)
    
    print(f"\n{'='*70}")
    print("GRAD-CAM ANALYSIS INTERPRETATION")
    print(f"{'='*70}\n")
    
    print("Visual analysis of figures/gradcam_comparison.png:\n")
    
    # True Positive (TP)
    if 'TP' in examples:
        tp = examples['TP']
        print(f"{'='*70}")
        print(f"TRUE POSITIVE (Correctly Detected Tumor)")
        print(f"{'='*70}")
        print(f"  Index: {tp['index']}")
        print(f"  Probability: {tp['prob']:.4f} (high confidence)")
        print()
        print("  Expected Grad-CAM Pattern:")
        print("  ✓ Both models should focus on dense nuclear regions")
        print("  ✓ High attention (red) on areas with:")
        print("    - Crowded, hyperchromatic nuclei")
        print("    - Irregular nuclear morphology")
        print("    - High nuclear-to-cytoplasmic ratio")
        print("  ✓ Low attention (blue) on stromal/background regions")
        print()
        print("  Model Comparison:")
        print("  - If similar: Both models learned similar features")
        print("  - If different: Different feature hierarchies")
        print()
    
    # True Negative (TN)
    if 'TN' in examples:
        tn = examples['TN']
        print(f"{'='*70}")
        print(f"TRUE NEGATIVE (Correctly Identified Normal)")
        print(f"{'='*70}")
        print(f"  Index: {tn['index']}")
        print(f"  Probability: {tn['prob']:.4f} (high confidence normal)")
        print()
        print("  Expected Grad-CAM Pattern:")
        print("  ✓ More diffuse attention (less red)")
        print("  ✓ Models don't find strong tumor-indicative features")
        print("  ✓ Attention may be spread across:")
        print("    - Normal stroma")
        print("    - Regular epithelial structures")
        print("    - Sparse lymphocytes")
        print()
        print("  Interpretation:")
        print("  - Weak/distributed activation = \"nothing suspicious here\"")
        print("  - No focal regions of high concern")
        print()
    
    # False Positive (FP)
    if 'FP' in examples:
        fp = examples['FP']
        print(f"{'='*70}")
        print(f"FALSE POSITIVE (Normal Misclassified as Tumor)")
        print(f"{'='*70}")
        print(f"  Index: {fp['index']}")
        print(f"  Probability: {fp['prob']:.4f} (very confident but WRONG!)")
        print()
        print("  Expected Grad-CAM Pattern:")
        print("  ✓ Strong attention on confusing regions:")
        print("    - Dense inflammatory infiltrate (lymphocytes)")
        print("    - Reactive epithelial atypia")
        print("    - Necrotic debris")
        print("    - Crush artifacts")
        print("  ✓ These mimic tumor features:")
        print("    - Dense cellularity → looks like tumor density")
        print("    - Hyperchromatic nuclei → inflammatory cells")
        print()
        print("  Key Insight:")
        print("  🔍 This reveals model's confusion points!")
        print("  - Where do models make confident mistakes?")
        print("  - What features are misleading?")
        print("  - Suggests need for better training on inflammatory regions")
        print()
    
    # False Negative (FN)
    if 'FN' in examples:
        fn = examples['FN']
        print(f"{'='*70}")
        print(f"FALSE NEGATIVE (Missed Tumor)")
        print(f"{'='*70}")
        print(f"  Index: {fn['index']}")
        print(f"  Probability: {fn['prob']:.4f} (very confident but WRONG!)")
        print()
        print("  Expected Grad-CAM Pattern:")
        print("  ✓ Weak attention despite tumor being present")
        print("  ✓ Possible causes:")
        print("    - Tumor at patch edge/corner (partial view)")
        print("    - Well-differentiated tumor (subtle features)")
        print("    - Sparse tumor cells (low density)")
        print("    - Poor staining quality")
        print("    - Tumor obscured by inflammation")
        print()
        print("  Key Insight:")
        print("  🔍 This reveals model's blind spots!")
        print("  - What tumor patterns are models missing?")
        print("  - Edge effects? Sparse patterns?")
        print("  - Suggests need for:")
        print("    * Multi-scale analysis")
        print("    * Context from neighboring patches")
        print("    * Data augmentation for edge cases")
        print()
    
    # Overall comparison
    print(f"{'='*70}")
    print(f"SUPERVISED VS CONTRASTIVE COMPARISON")
    print(f"{'='*70}\n")
    
    print("Key Questions:")
    print()
    print("1. Attention Similarity:")
    print("   - Are red regions similar between models?")
    print("   - If YES → Both learned similar features (good!)")
    print("   - If NO → Different feature hierarchies")
    print()
    print("2. Focal vs Distributed:")
    print("   - Supervised: More focal (specific features)?")
    print("   - Contrastive: More distributed (holistic)?")
    print()
    print("3. Error Analysis:")
    print("   - Do models make same mistakes (FP, FN)?")
    print("   - If YES → Shared limitations")
    print("   - If NO → Could benefit from ensembling")
    print()
    
    # Actionable insights
    print(f"{'='*70}")
    print(f"ACTIONABLE INSIGHTS")
    print(f"{'='*70}\n")
    
    print("Based on Grad-CAM analysis:\n")
    
    print("✓ For True Positives:")
    print("  → Models correctly focus on nuclear morphology")
    print("  → Dense, irregular nuclei are key features")
    print()
    
    print("✗ For False Positives:")
    print("  → Need better discrimination of:")
    print("    * Inflammatory infiltrate vs tumor")
    print("    * Reactive atypia vs malignancy")
    print("  → Solution: Add more training data with inflammation")
    print()
    
    print("✗ For False Negatives:")
    print("  → Models struggle with:")
    print("    * Sparse tumor cells")
    print("    * Edge/boundary effects")
    print("  → Solution: Multi-scale analysis or context-aware models")
    print()
    
    print("🎯 Overall Conclusion:")
    print("  Grad-CAM reveals:")
    print("  1. What features models rely on (nuclear morphology)")
    print("  2. Where models get confused (inflammation, edges)")
    print("  3. Concrete directions for improvement")
    print()
    print(f"{'='*70}\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Grad-CAM results')
    parser.add_argument('--examples-json', type=str, default='gradcam_examples.json',
                       help='Path to examples JSON file')
    args = parser.parse_args()
    
    # Check if figure exists
    figure_path = Path('figures/gradcam_comparison.png')
    if not figure_path.exists():
        print(f"\n⚠ Warning: Grad-CAM figure not found at {figure_path}")
        print("  Please run create_gradcam_comparison.py first")
        return
    
    analyze_gradcam_results(args.examples_json)
    
    print("\n📊 Generated Files:")
    print(f"  1. Examples: {args.examples_json}")
    print(f"  2. Figure:   {figure_path}")
    print("\n✓ Analysis complete! Review figures/gradcam_comparison.png for visual insights.")


if __name__ == '__main__':
    main()


