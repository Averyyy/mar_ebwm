#!/usr/bin/env python3
"""
Script to compute and cache FID statistics for 64x64 ImageNet validation dataset.
This allows using precomputed stats instead of loading 50K validation images during training.
Uses torch-fidelity's API to ensure compatibility.
"""

import os
import argparse
import torch_fidelity
import time

def main():
    parser = argparse.ArgumentParser(description='Compute FID statistics for 64x64 ImageNet')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet validation dataset')
    parser.add_argument('--output_path', type=str, default='fid_stats/imagenet_64_stats.npz',
                        help='Output path for FID statistics')
    parser.add_argument('--img_size', type=int, default=64,
                        help='Image size to resize to')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"Computing FID statistics for {args.img_size}x{args.img_size} images")
    print(f"Dataset path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    
    # Use torch-fidelity's calculate_metrics to compute statistics
    # This is the same API used in engine_mar.py, ensuring compatibility
    start_time = time.time()
    
    try:
        # Use direct approach: manually extract features and compute statistics
        print("Computing FID statistics using direct feature extraction...")
        
        import torch
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder
        import torchvision.transforms as transforms
        import numpy as np
        
        # Custom dataset class to handle transforms properly
        class ImageDatasetForFID(torch.utils.data.Dataset):
            def __init__(self, root, img_size):
                self.dataset = ImageFolder(root)
                self.img_size = img_size
                
                # Transform pipeline: resize to target size first, then to InceptionV3 size
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.Resize(299),  # InceptionV3 input size
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                img = self.transform(img)
                return img, label
        
        print(f"Loading dataset from {args.data_path}")
        dataset = ImageDatasetForFID(args.data_path, args.img_size)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Found {len(dataset)} images")
        
        # Create InceptionV3 feature extractor (matching torch-fidelity behavior)
        print("Creating InceptionV3 feature extractor...")
        import torchvision.models as models
        from torch import nn
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained InceptionV3
        inception = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        inception.fc = nn.Identity()  # Remove classification layer to get 2048-dim features
        inception = inception.to(device)
        inception.eval()
        
        # Extract features
        print("Extracting features...")
        all_features = []
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i % 100 == 0:
                    print(f"Processing batch {i}/{len(dataloader)}")
                
                images = images.to(device)
                
                # Extract features using InceptionV3
                # During training, InceptionV3 returns InceptionOutputs with .logits
                # During eval, it returns just the tensor
                features = inception(images)
                
                # Handle InceptionOutputs vs direct tensor
                if hasattr(features, 'logits'):
                    features = features.logits
                
                # Ensure we have 2048-dimensional features
                if features.shape[1] != 2048:
                    print(f"Warning: Expected 2048-dim features, got {features.shape[1]}")
                
                all_features.append(features.cpu())
        
        # Concatenate all features
        print("Computing statistics...")
        all_features = torch.cat(all_features, dim=0)
        print(f"Extracted features shape: {all_features.shape}")
        
        # Compute statistics exactly like torch-fidelity (from metric_fid.py)
        assert all_features.dim() == 2, f"Expected 2D features, got {all_features.dim()}D"
        features_np = all_features.numpy()
        mu = np.mean(features_np, axis=0)
        sigma = np.cov(features_np, rowvar=False)
        
        stats = {'mu': mu, 'sigma': sigma}
        
        # Save statistics
        print(f"Saving statistics to {args.output_path}")
        np.savez(args.output_path, **stats)
        
        end_time = time.time()
        print(f"Computation completed in {end_time - start_time:.2f} seconds")
        
        # Verify the generated file
        if os.path.exists(args.output_path):
            import numpy as np
            loaded = np.load(args.output_path)
            print("Verification:")
            print(f"Keys: {list(loaded.keys())}")
            print(f"mu shape: {loaded['mu'].shape}")
            print(f"sigma shape: {loaded['sigma'].shape}")
            print(f"File size: {os.path.getsize(args.output_path) / 1024 / 1024:.2f} MB")
            print("FID statistics computation completed successfully!")
        else:
            print(f"ERROR: Statistics file not generated at {args.output_path}")
            
    except Exception as e:
        print(f"ERROR during computation: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure torch-fidelity is properly installed and CUDA is available.")

if __name__ == '__main__':
    main()