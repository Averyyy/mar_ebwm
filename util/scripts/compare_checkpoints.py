import torch
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
import copy

# Add the project root to path
import sys
sys.path.append('/work/hdd/bdta/aqian1/mar_ebwm')

from models.vae import AutoencoderKL
from models import mar
import util.misc as misc


def load_checkpoint(checkpoint_path, model, device):
    """Load checkpoint and return model state dict and ema params"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model_state = checkpoint['model']
    
    # Load EMA params if available
    ema_params = None
    if 'model_ema' in checkpoint:
        ema_state_dict = checkpoint['model_ema']
        if ema_state_dict is not None:
            ema_params = [ema_state_dict[name] for name, _ in model.named_parameters()]
    
    return model_state, ema_params, checkpoint.get('epoch', 0)


def generate_images(model, vae, labels, num_iter=64, cfg=1.0, temperature=1.0, seed=42):
    """Generate images for given labels"""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model.eval()
    with torch.no_grad():
        # Generate tokens
        sampled_tokens = model.sample_tokens(
            bsz=len(labels),
            num_iter=num_iter,
            cfg=cfg,
            cfg_schedule="linear",
            labels=labels,
            temperature=temperature
        )
        
        # Decode to images
        sampled_images = vae.decode(sampled_tokens / 0.2325)
        
    return sampled_tokens, sampled_images


def save_comparison_results(tokens1, tokens2, images1, images2, output_dir, prefix=""):
    """Save tokens and images for comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokens as numpy arrays
    np.save(os.path.join(output_dir, f"{prefix}tokens_model1.npy"), tokens1.cpu().numpy())
    np.save(os.path.join(output_dir, f"{prefix}tokens_model2.npy"), tokens2.cpu().numpy())
    
    # Calculate token difference
    token_diff = torch.abs(tokens1 - tokens2)
    token_diff_stats = {
        'mean': token_diff.mean().item(),
        'max': token_diff.max().item(),
        'min': token_diff.min().item(),
        'std': token_diff.std().item()
    }
    
    # Save images
    images1_np = (images1 + 1) / 2  # [-1, 1] -> [0, 1]
    images2_np = (images2 + 1) / 2
    
    for i in range(images1.shape[0]):
        # Convert to numpy and transpose
        img1 = images1_np[i].cpu().numpy().transpose(1, 2, 0)
        img2 = images2_np[i].cpu().numpy().transpose(1, 2, 0)
        
        # Convert to uint8
        img1 = (img1 * 255).round().clip(0, 255).astype(np.uint8)
        img2 = (img2 * 255).round().clip(0, 255).astype(np.uint8)
        
        # Save images
        cv2.imwrite(os.path.join(output_dir, f"{prefix}class1_model1_img{i}.png"), img1[:, :, ::-1])
        cv2.imwrite(os.path.join(output_dir, f"{prefix}class1_model2_img{i}.png"), img2[:, :, ::-1])
        
        # Create difference image
        diff_img = np.abs(img1.astype(float) - img2.astype(float))
        diff_img = (diff_img / diff_img.max() * 255).astype(np.uint8) if diff_img.max() > 0 else diff_img.astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{prefix}class1_diff_img{i}.png"), diff_img[:, :, ::-1])
    
    return token_diff_stats


def main():
    parser = argparse.ArgumentParser(description='Compare two model checkpoints')
    parser.add_argument('--checkpoint1', type=str, 
                        default='/work/hdd/bdta/aqian1/mar_ebwm/output/mar-base-energy-a-0.01-m-1/checkpoint-last.pth',
                        help='Path to first checkpoint')
    parser.add_argument('--checkpoint2', type=str,
                        default='/work/hdd/bdta/aqian1/mar_ebwm/output/mar-base-energy-lr_1e-4-alpha_3-mult_9/checkpoint-last.pth',
                        help='Path to second checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoint_comparison',
                        help='Output directory for comparison results')
    parser.add_argument('--num_images', type=int, default=4,
                        help='Number of images to generate per class')
    parser.add_argument('--class_id', type=int, default=1,
                        help='Class ID to generate')
    parser.add_argument('--num_iter', type=int, default=64,
                        help='Number of iterations for generation')
    parser.add_argument('--cfg', type=float, default=1.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA parameters if available')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize VAE
    print("Loading VAE...")
    vae = AutoencoderKL(
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path='/work/hdd/bdta/aqian1/mar_ebwm/pretrained_models/vae/kl16.ckpt'
    ).to(device).eval()
    
    # Initialize model
    print("Initializing MAR model...")
    model = mar.mar_base(
        img_size=64,
        vae_stride=16,
        patch_size=1,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.1,
        proj_dropout=0.1,
        buffer_size=64,
        grad_checkpointing=False,
        mcmc_step_size=0.01,
    ).to(device)
    
    # Create labels
    labels = torch.full((args.num_images,), args.class_id, dtype=torch.long, device=device)
    
    # Load first checkpoint
    print(f"\nLoading checkpoint 1: {args.checkpoint1}")
    model_state1, ema_params1, epoch1 = load_checkpoint(args.checkpoint1, model, device)
    
    # Generate with first model
    if args.use_ema and ema_params1 is not None:
        print("Using EMA parameters for model 1")
        original_state = copy.deepcopy(model.state_dict())
        ema_state_dict = copy.deepcopy(model.state_dict())
        for i, (name, _) in enumerate(model.named_parameters()):
            ema_state_dict[name] = ema_params1[i]
        model.load_state_dict(ema_state_dict)
    else:
        print("Using regular parameters for model 1")
        model.load_state_dict(model_state1)
    
    print(f"Generating images with model 1 (epoch {epoch1})...")
    tokens1, images1 = generate_images(model, vae, labels, args.num_iter, args.cfg, args.temperature, args.seed)
    
    # Load second checkpoint
    print(f"\nLoading checkpoint 2: {args.checkpoint2}")
    model_state2, ema_params2, epoch2 = load_checkpoint(args.checkpoint2, model, device)
    
    # Generate with second model
    if args.use_ema and ema_params2 is not None:
        print("Using EMA parameters for model 2")
        ema_state_dict2 = copy.deepcopy(model.state_dict())
        for i, (name, _) in enumerate(model.named_parameters()):
            ema_state_dict2[name] = ema_params2[i]
        model.load_state_dict(ema_state_dict2)
    else:
        print("Using regular parameters for model 2")
        model.load_state_dict(model_state2)
    
    print(f"Generating images with model 2 (epoch {epoch2})...")
    tokens2, images2 = generate_images(model, vae, labels, args.num_iter, args.cfg, args.temperature, args.seed)
    
    # Compare and save results
    print("\nComparing results...")
    token_diff_stats = save_comparison_results(tokens1, tokens2, images1, images2, args.output_dir)
    
    # Print comparison statistics
    print("\n=== Token Comparison Statistics ===")
    print(f"Token shape: {tokens1.shape}")
    print(f"Mean absolute difference: {token_diff_stats['mean']:.6f}")
    print(f"Max absolute difference: {token_diff_stats['max']:.6f}")
    print(f"Min absolute difference: {token_diff_stats['min']:.6f}")
    print(f"Std of difference: {token_diff_stats['std']:.6f}")
    
    # Check if tokens are identical
    are_identical = torch.allclose(tokens1, tokens2, rtol=1e-5, atol=1e-8)
    print(f"\nTokens are identical: {are_identical}")
    
    # Calculate image-level statistics
    img_diff = torch.abs(images1 - images2)
    print("\n=== Image Comparison Statistics ===")
    print(f"Image shape: {images1.shape}")
    print(f"Mean absolute difference: {img_diff.mean().item():.6f}")
    print(f"Max absolute difference: {img_diff.max().item():.6f}")
    print(f"Min absolute difference: {img_diff.min().item():.6f}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 