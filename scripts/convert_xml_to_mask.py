#!/usr/bin/env python3
"""
Convert CAMELYON16 XML annotations to binary TIF masks.

The XML files contain polygon annotations that need to be rendered as binary masks
matching the WSI dimensions.
"""

import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import openslide
from PIL import Image
import cv2
from tqdm import tqdm


def parse_asap_xml(xml_path):
    """
    Parse ASAP XML annotation file and extract polygon coordinates.
    
    Args:
        xml_path: Path to XML annotation file.
    
    Returns:
        List of polygons, where each polygon is a list of (x, y) tuples.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    polygons = []
    annotations = root.find('Annotations')
    
    if annotations is None:
        return polygons
    
    for annotation in annotations.findall('Annotation'):
        annotation_type = annotation.get('Type')
        
        # Only process polygon annotations
        if annotation_type != 'Polygon':
            continue
        
        coordinates = annotation.find('Coordinates')
        if coordinates is None:
            continue
        
        polygon = []
        for coord in coordinates.findall('Coordinate'):
            x = float(coord.get('X'))
            y = float(coord.get('Y'))
            polygon.append((x, y))
        
        if len(polygon) >= 3:  # Valid polygon needs at least 3 points
            polygons.append(polygon)
    
    return polygons


def create_mask_from_polygons(wsi_path, polygons, output_path, level=4):
    """
    Create a binary mask from polygon annotations.
    
    Args:
        wsi_path: Path to WSI file.
        polygons: List of polygons (each is a list of (x, y) tuples).
        output_path: Path to save the output mask TIF.
        level: Pyramid level to create mask at (higher = smaller/faster).
    """
    # Open WSI to get dimensions
    slide = openslide.OpenSlide(wsi_path)
    
    # Get dimensions at chosen level
    level = min(level, slide.level_count - 1)
    level_dims = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    
    print(f"  Creating mask at level {level}: {level_dims[0]}x{level_dims[1]} (downsample: {downsample:.2f}x)")
    
    # Create blank mask
    mask = np.zeros((level_dims[1], level_dims[0]), dtype=np.uint8)
    
    # Draw each polygon
    for polygon in polygons:
        # Scale coordinates to chosen level
        scaled_polygon = [(int(x / downsample), int(y / downsample)) for x, y in polygon]
        
        # Convert to numpy array
        pts = np.array(scaled_polygon, dtype=np.int32)
        
        # Fill polygon
        cv2.fillPoly(mask, [pts], 255)
    
    # Save as TIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with proper metadata for OpenSlide
    mask_pil = Image.fromarray(mask)
    mask_pil.save(output_path, compression='jpeg', quality=90)
    
    slide.close()
    
    return mask


def main():
    parser = argparse.ArgumentParser(description='Convert ASAP XML annotations to binary TIF masks')
    parser.add_argument('--xml-dir', type=str, required=True,
                        help='Directory containing XML annotation files')
    parser.add_argument('--wsi-dir', type=str, required=True,
                        help='Directory containing WSI files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for mask TIF files')
    parser.add_argument('--level', type=int, default=4,
                        help='Pyramid level for mask (higher=smaller, default=4)')
    parser.add_argument('--pattern', type=str, default='*.xml',
                        help='Pattern to match XML files (default: *.xml)')
    
    args = parser.parse_args()
    
    xml_dir = Path(args.xml_dir)
    wsi_dir = Path(args.wsi_dir)
    output_dir = Path(args.output_dir)
    
    if not xml_dir.exists():
        print(f"Error: XML directory not found: {xml_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not wsi_dir.exists():
        print(f"Error: WSI directory not found: {wsi_dir}", file=sys.stderr)
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all XML files
    xml_files = sorted(xml_dir.glob(args.pattern))
    
    if not xml_files:
        print(f"No XML files found in {xml_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(xml_files)} XML annotation files")
    print(f"Output directory: {output_dir}\n")
    
    # Process each XML file
    for xml_path in tqdm(xml_files, desc="Converting masks"):
        # Get corresponding WSI file
        wsi_id = xml_path.stem  # e.g., "tumor_008"
        wsi_path = wsi_dir / f"{wsi_id}.tif"
        
        if not wsi_path.exists():
            print(f"Warning: WSI not found for {wsi_id}, skipping")
            continue
        
        output_path = output_dir / f"{wsi_id}_mask.tif"
        
        # Skip if already exists
        if output_path.exists():
            print(f"  Mask already exists: {output_path.name}, skipping")
            continue
        
        print(f"\n{wsi_id}:")
        
        # Parse XML
        try:
            polygons = parse_asap_xml(xml_path)
            print(f"  Found {len(polygons)} annotation polygons")
            
            if len(polygons) == 0:
                print(f"  Warning: No valid polygons found in {xml_path.name}")
                continue
            
            # Create mask
            create_mask_from_polygons(wsi_path, polygons, output_path, level=args.level)
            print(f"  ✓ Saved: {output_path}")
            
        except Exception as e:
            print(f"  Error processing {xml_path.name}: {e}")
            continue
    
    print(f"\n✓ Done! Masks saved to: {output_dir}")


if __name__ == '__main__':
    main()

