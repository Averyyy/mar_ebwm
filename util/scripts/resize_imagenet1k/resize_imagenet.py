#!/usr/bin/env python3
"""
Resize ImageNet dataset from 256x256 to 64x64 with multiprocessing support.
Preserves directory structure and handles errors gracefully.
"""

import os
import sys
import shutil
import time
from multiprocessing import Pool, cpu_count, Manager, Value
from pathlib import Path
from PIL import Image
import argparse
from datetime import datetime
import threading
import signal

# Global variables for progress tracking
processed_files = None
total_files = None
error_count = None
start_time = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n[INFO] Received interrupt signal. Cleaning up...")
    sys.exit(0)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def resize_and_copy_image(args):
    """
    Resize a single image from 256x256 to 64x64 and copy to destination.
    
    Args:
        args: tuple of (src_path, dst_path)
    
    Returns:
        tuple: (success: bool, src_path: str, error_msg: str or None)
    """
    src_path, dst_path = args
    
    try:
        # Create destination directory
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Skip if destination already exists and has reasonable size
        if os.path.exists(dst_path):
            if os.path.getsize(dst_path) > 100:  # Skip if file exists and > 100 bytes
                return True, src_path, None
        
        # Open and resize image
        with Image.open(src_path) as img:
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Only resize if current size is not 64x64
            if img.size != (64, 64):
                # Use LANCZOS for high-quality downsampling
                img_resized = img.resize((64, 64), Image.Resampling.LANCZOS)
            else:
                img_resized = img
            
            # Save as PNG (consistent format)
            if dst_path.endswith('.JPEG') or dst_path.endswith('.jpg'):
                dst_path = dst_path.rsplit('.', 1)[0] + '.png'
            elif not dst_path.endswith('.png'):
                dst_path += '.png' if not dst_path.endswith('.png') else ''
            
            img_resized.save(dst_path, 'PNG', optimize=True)
        
        return True, src_path, None
        
    except Exception as e:
        return False, src_path, str(e)

def find_all_images(source_dir):
    """
    Find all image files in the source directory.
    
    Args:
        source_dir (str): Source directory path
        
    Returns:
        list: List of (src_path, dst_path) tuples
    """
    source_path = Path(source_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPEG', '.JPG', '.PNG'}
    
    image_pairs = []
    
    print("[INFO] Scanning for images...")
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                src_path = os.path.join(root, file)
                # Create corresponding destination path
                rel_path = os.path.relpath(src_path, source_dir)
                dst_path = os.path.join(args.dst_dir, rel_path)
                
                # Change extension to .png for consistency
                if dst_path.endswith('.JPEG') or dst_path.endswith('.jpg') or dst_path.endswith('.JPG'):
                    dst_path = dst_path.rsplit('.', 1)[0] + '.png'
                
                image_pairs.append((src_path, dst_path))
    
    return image_pairs

def progress_reporter(processed_counter, total_files, error_counter, start_time):
    """Background thread to report progress"""
    while True:
        time.sleep(30)  # Report every 30 seconds
        
        current_processed = processed_counter.value
        current_errors = error_counter.value
            
        elapsed = time.time() - start_time
        if current_processed > 0:
            rate = current_processed / elapsed
            eta = (total_files - current_processed) / rate if rate > 0 else 0
            
            print(f"[PROGRESS] {current_processed}/{total_files} processed "
                  f"({current_processed/total_files*100:.1f}%) | "
                  f"Errors: {current_errors} | "
                  f"Rate: {rate:.1f} files/sec | "
                  f"ETA: {eta/3600:.1f}h")

def main():
    global processed_files, total_files, error_count, start_time
    
    parser = argparse.ArgumentParser(description='Resize ImageNet dataset from 256x256 to 64x64')
    parser.add_argument('--src_dir', default='/work/nvme/belh/aqian1/imagenet-1k/',
                        help='Source ImageNet directory')
    parser.add_argument('--dst_dir', default='/work/hdd/bdta/aqian1/data/imagenet-1k-64/',
                        help='Destination directory')
    parser.add_argument('--num_workers', type=int, default=min(32, cpu_count()),
                        help='Number of worker processes')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for processing')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be done without actually doing it')
    
    global args
    args = parser.parse_args()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    print(f"[INFO] Starting ImageNet resize job at {datetime.now()}")
    print(f"[INFO] Source: {args.src_dir}")
    print(f"[INFO] Destination: {args.dst_dir}")
    print(f"[INFO] Workers: {args.num_workers}")
    print(f"[INFO] Batch size: {args.batch_size}")
    
    # Verify source directory exists
    if not os.path.exists(args.src_dir):
        print(f"[ERROR] Source directory does not exist: {args.src_dir}")
        sys.exit(1)
    
    # Create destination directory
    os.makedirs(args.dst_dir, exist_ok=True)
    
    # Find all images
    image_pairs = find_all_images(args.src_dir)
    total_files = len(image_pairs)
    
    if total_files == 0:
        print("[ERROR] No images found in source directory")
        sys.exit(1)
    
    print(f"[INFO] Found {total_files} images to process")
    
    if args.dry_run:
        print("[DRY RUN] Would process the following structure:")
        for src, dst in image_pairs[:10]:  # Show first 10
            print(f"  {src} -> {dst}")
        if total_files > 10:
            print(f"  ... and {total_files - 10} more files")
        return
    
    # Setup multiprocessing counters
    manager = Manager()
    processed_counter = manager.Value('i', 0)
    error_counter = manager.Value('i', 0)
    
    start_time = time.time()
    
    # Start progress reporter thread
    progress_thread = threading.Thread(
        target=progress_reporter,
        args=(processed_counter, total_files, error_counter, start_time),
        daemon=True
    )
    progress_thread.start()
    
    # Process images in batches
    print(f"[INFO] Starting processing with {args.num_workers} workers...")
    
    with Pool(processes=args.num_workers) as pool:
        # Prepare arguments for workers (just src and dst paths)
        worker_args = [(src_path, dst_path) for src_path, dst_path in image_pairs]
        
        processed_count = 0
        error_count = 0
        
        try:
            # Process in batches to avoid memory issues
            for i in range(0, len(worker_args), args.batch_size):
                batch = worker_args[i:i + args.batch_size]
                batch_num = i//args.batch_size + 1
                total_batches = (len(worker_args) + args.batch_size - 1)//args.batch_size
                print(f"[INFO] Processing batch {batch_num}/{total_batches}")
                
                results = pool.map(resize_and_copy_image, batch)
                
                # Update counters and log errors from this batch
                batch_processed = 0
                batch_errors = 0
                for success, src_path, error_msg in results:
                    if success:
                        batch_processed += 1
                    else:
                        batch_errors += 1
                        print(f"[ERROR] Failed to process {src_path}: {error_msg}")
                
                # Update global counters
                processed_count += batch_processed
                error_count += batch_errors
                
                # Update shared counters for progress reporter
                processed_counter.value = processed_count
                error_counter.value = error_count
                
                # Show batch progress
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"[BATCH] Processed {batch_processed}/{len(batch)} files | "
                      f"Total: {processed_count}/{total_files} ({processed_count/total_files*100:.1f}%) | "
                      f"Rate: {rate:.1f} files/sec")
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
            pool.terminate()
            pool.join()
            sys.exit(1)
    
    # Final statistics
    elapsed = time.time() - start_time
    final_processed = processed_count
    final_errors = error_count
    
    print(f"\n[COMPLETED] Processing finished at {datetime.now()}")
    print(f"[STATS] Processed: {final_processed}/{total_files}")
    print(f"[STATS] Errors: {final_errors}")
    print(f"[STATS] Success rate: {(final_processed-final_errors)/final_processed*100:.1f}%")
    print(f"[STATS] Total time: {elapsed/3600:.1f} hours")
    print(f"[STATS] Average rate: {final_processed/elapsed:.1f} files/sec")
    
    if final_errors > 0:
        print(f"\n[WARNING] {final_errors} files failed to process. Check the logs above.")
    
    print(f"[INFO] Results saved to: {args.dst_dir}")

if __name__ == "__main__":
    main()