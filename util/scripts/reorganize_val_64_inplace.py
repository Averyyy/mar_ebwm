#!/usr/bin/env python3
"""
In-place reorganization of ImageNet-64 val dataset using val_annotations.txt
Creates class folders and moves images to appropriate directories
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

def parse_annotations(annotations_file):
    """Parse val_annotations.txt to get filename -> class_id mapping"""
    filename_to_class = {}
    
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                filename = parts[0]
                class_id = parts[1]
                filename_to_class[filename] = class_id
    
    return filename_to_class

def reorganize_val_dataset(val_root):
    """Reorganize val dataset in-place using annotations"""
    val_root = Path(val_root)
    images_dir = val_root / "images"
    annotations_file = val_root / "val_annotations.txt"
    
    # Check if directories exist
    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} does not exist")
        return False
    
    if not annotations_file.exists():
        print(f"Error: Annotations file {annotations_file} does not exist")
        return False
    
    print(f"Processing val dataset at: {val_root}")
    print(f"Images directory: {images_dir}")
    print(f"Annotations file: {annotations_file}")
    
    # Parse annotations
    print("Parsing annotations...")
    filename_to_class = parse_annotations(annotations_file)
    print(f"Found {len(filename_to_class)} image-class mappings")
    
    # Get list of image files
    image_files = list(images_dir.glob("*.JPEG"))
    print(f"Found {len(image_files)} image files")
    
    # Create class directories and move files
    print("Reorganizing images by class...")
    moved_count = 0
    skipped_count = 0
    
    for image_file in tqdm(image_files, desc="Moving images"):
        filename = image_file.name
        
        if filename in filename_to_class:
            class_id = filename_to_class[filename]
            
            # Create class directory if it doesn't exist
            class_dir = val_root / class_id
            class_dir.mkdir(exist_ok=True)
            
            # Move file to class directory
            dst_path = class_dir / filename
            try:
                shutil.move(str(image_file), str(dst_path))
                moved_count += 1
            except Exception as e:
                print(f"Error moving {filename}: {e}")
                skipped_count += 1
        else:
            print(f"Warning: No class mapping found for {filename}")
            skipped_count += 1
    
    print(f"\nReorganization complete!")
    print(f"Moved: {moved_count} files")
    print(f"Skipped: {skipped_count} files")
    
    # Remove empty images directory if all files were moved
    if moved_count > 0 and len(list(images_dir.glob("*"))) == 0:
        images_dir.rmdir()
        print(f"Removed empty images directory")
    
    # Show final structure
    class_dirs = [d for d in val_root.iterdir() if d.is_dir() and d.name.startswith('n')]
    print(f"Created {len(class_dirs)} class directories")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Reorganize ImageNet-64 val dataset in-place')
    parser.add_argument('val_root', help='Path to val directory containing images/ and val_annotations.txt')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ImageNet-64 Val Dataset In-Place Reorganization")
    print("=" * 80)
    print(f"Val root: {args.val_root}")
    print("WARNING: This will modify your dataset structure!")
    print("=" * 80)
    
    # Confirm operation
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    success = reorganize_val_dataset(args.val_root)
    
    if success:
        print("\n" + "=" * 80)
        print("REORGANIZATION COMPLETE!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("REORGANIZATION FAILED!")
        print("=" * 80)

if __name__ == "__main__":
    main()