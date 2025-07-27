#!/usr/bin/env python3
"""
In-place reorganization of ImageNet val and test datasets
Creates temporary reconstructed folders, verifies completion, then replaces originals
"""
import os
import shutil
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)

def get_class_from_filename(filename, dataset_type):
    """Extract class ID from ImageNet filename"""
    if dataset_type == 'val':
        pattern = r'ILSVRC2012_val_\d+_(n\d+)\.JPEG'
        match = re.match(pattern, filename)
        if match:
            return match.group(1)
    elif dataset_type == 'test':
        pattern = r'ILSVRC2012_test_\d+\.JPEG'
        if re.match(pattern, filename):
            return 'test_unknown'  # Test set has no class labels
    
    return None

def copy_file_batch(args):
    """Copy a batch of files to new structure - used for parallel processing"""
    files_batch, src_dir, dst_dir, dataset_type = args
    copied_count = 0
    
    for filename in files_batch:
        class_id = get_class_from_filename(filename, dataset_type)
        if class_id is None:
            continue
            
        # Create class directory
        class_dir = os.path.join(dst_dir, class_id)
        ensure_dir(class_dir)
        
        # Copy file (not move, to be safe)
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(class_dir, filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
            copied_count += 1
    
    return copied_count

def count_images_in_dir(directory):
    """Count all JPEG images in directory and subdirectories"""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len([f for f in files if f.endswith('.JPEG')])
    return count

def reorganize_dataset_inplace(imagenet_root, dataset_type, num_workers=32):
    """
    In-place reorganize ImageNet dataset
    
    Args:
        imagenet_root: Root ImageNet directory
        dataset_type: 'val' or 'test'
        num_workers: Number of parallel workers
    """
    print(f"\n{'='*60}")
    print(f"REORGANIZING {dataset_type.upper()} DATASET IN-PLACE")
    print(f"{'='*60}")
    
    # Define paths
    if dataset_type == 'val':
        src_dir = os.path.join(imagenet_root, 'val', 'images')
        if not os.path.exists(src_dir):
            src_dir = os.path.join(imagenet_root, 'val')
        construct_dir = os.path.join(imagenet_root, 'val_construct')
        final_dir = os.path.join(imagenet_root, 'val')
        
    elif dataset_type == 'test':
        src_dir = os.path.join(imagenet_root, 'test')
        construct_dir = os.path.join(imagenet_root, 'test_construct')
        final_dir = os.path.join(imagenet_root, 'test')
    
    print(f"Source: {src_dir}")
    print(f"Construct: {construct_dir}")
    print(f"Final: {final_dir}")
    
    # Check if source exists
    if not os.path.exists(src_dir):
        print(f"Warning: Source directory {src_dir} does not exist, skipping {dataset_type}")
        return False
    
    # Count original images
    if dataset_type == 'val':
        jpeg_files = [f for f in os.listdir(src_dir) 
                      if f.endswith('.JPEG') and 'ILSVRC2012_val_' in f]
    elif dataset_type == 'test':
        jpeg_files = [f for f in os.listdir(src_dir) 
                      if f.endswith('.JPEG') and 'ILSVRC2012_test_' in f]
    
    original_count = len(jpeg_files)
    print(f"Found {original_count} {dataset_type} images to reorganize")
    
    if original_count == 0:
        print(f"No {dataset_type} images found, skipping")
        return False
    
    # Clean up any existing construct directory
    if os.path.exists(construct_dir):
        print(f"Removing existing construct directory: {construct_dir}")
        shutil.rmtree(construct_dir)
    
    # Create construct directory
    ensure_dir(construct_dir)
    
    # Split files into batches for parallel processing
    batch_size = max(1, len(jpeg_files) // num_workers)
    file_batches = [jpeg_files[i:i + batch_size] 
                   for i in range(0, len(jpeg_files), batch_size)]
    
    # Prepare arguments for parallel processing
    process_args = [(batch, src_dir, construct_dir, dataset_type) for batch in file_batches]
    
    # Process in parallel
    total_copied = 0
    print(f"Copying files to construct directory with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_batch = {executor.submit(copy_file_batch, args): i 
                          for i, args in enumerate(process_args)}
        
        # Collect results with progress bar
        with tqdm(total=len(future_to_batch), desc=f"Processing {dataset_type} batches") as pbar:
            for future in as_completed(future_to_batch):
                copied_count = future.result()
                total_copied += copied_count
                pbar.update(1)
    
    print(f"Copied {total_copied} {dataset_type} images to construct directory")
    
    # Verify the reorganization
    print("Verifying reorganization...")
    construct_count = count_images_in_dir(construct_dir)
    
    print(f"Original images: {original_count}")
    print(f"Construct images: {construct_count}")
    
    if construct_count != original_count:
        print(f"ERROR: Image count mismatch! Expected {original_count}, got {construct_count}")
        print("Not proceeding with replacement. Check the construct directory manually.")
        return False
    
    # Count classes
    class_dirs = [d for d in os.listdir(construct_dir) 
                 if os.path.isdir(os.path.join(construct_dir, d))]
    print(f"Created {len(class_dirs)} class directories")
    
    # All good, proceed with replacement
    print(f"\n✅ Verification passed! Proceeding with replacement...")
    
    # Create backup name for original
    backup_dir = f"{final_dir}_original_backup"
    
    # Step 1: Backup original directory
    print(f"Step 1: Backing up original {dataset_type} directory...")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.move(final_dir, backup_dir)
    
    # Step 2: Move construct to final location
    print(f"Step 2: Moving construct directory to final location...")
    shutil.move(construct_dir, final_dir)
    
    # Step 3: Verify final result
    print(f"Step 3: Final verification...")
    final_count = count_images_in_dir(final_dir)
    
    if final_count == original_count:
        print(f"✅ SUCCESS! Final verification passed ({final_count} images)")
        
        # Step 4: Remove backup
        print(f"Step 4: Removing backup directory...")
        shutil.rmtree(backup_dir)
        
        print(f"🎉 {dataset_type.upper()} dataset successfully reorganized in-place!")
        return True
    else:
        print(f"❌ FAILED! Final count mismatch: expected {original_count}, got {final_count}")
        print(f"Restoring from backup...")
        
        # Restore from backup
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.move(backup_dir, final_dir)
        
        print(f"Original {dataset_type} directory restored.")
        return False

def main():
    parser = argparse.ArgumentParser(description='In-place reorganize ImageNet datasets')
    parser.add_argument('--imagenet_root', 
                       default='/work/nvme/belh/aqian1/imagenet-1k',
                       help='Root ImageNet directory')
    parser.add_argument('--num_workers', type=int, default=32,
                       help='Number of parallel workers')
    parser.add_argument('--datasets', nargs='+', default=['val', 'test'],
                       choices=['val', 'test'], 
                       help='Which datasets to reorganize')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ImageNet Dataset In-Place Reorganization")
    print("=" * 80)
    print(f"ImageNet root: {args.imagenet_root}")
    print(f"Workers: {args.num_workers}")
    print(f"Datasets: {args.datasets}")
    print(f"WARNING: This will modify your original dataset structure!")
    print("=" * 80)
    
    success_count = 0
    total_datasets = len(args.datasets)
    
    for dataset in args.datasets:
        success = reorganize_dataset_inplace(args.imagenet_root, dataset, args.num_workers)
        if success:
            success_count += 1
    
    print("\n" + "=" * 80)
    print("IN-PLACE REORGANIZATION COMPLETE!")
    print("=" * 80)
    print(f"Successfully reorganized: {success_count}/{total_datasets} datasets")
    
    if success_count == total_datasets:
        print("🎉 ALL DATASETS SUCCESSFULLY REORGANIZED!")
        print("\nYou can now update your val_data_path to:")
        print(f"{args.imagenet_root}/val")
    else:
        print("⚠️  Some datasets failed reorganization. Check the output above.")
    
    return 0 if success_count == total_datasets else 1

if __name__ == "__main__":
    exit(main())