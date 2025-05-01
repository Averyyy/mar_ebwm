"""
Quick VAE reconstruction quality check.

For each sampled image:
 ┌─────────────┬──────────────┐
 │  original   │ reconstruction│
 └─────────────┴──────────────┘
Saved to <output_dir>/sample_XXX.png
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from models.vae import AutoencoderKL
from util.crop import center_crop_arr

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_path', type=str, default='pretrained_models/vae/kl16.ckpt')
    p.add_argument('--data_path', type=str, required=True,
                   help='root folder of ImageNet (subfolders train/ or class dirs)')
    p.add_argument('--output_dir', type=str, default='./vae_recon_output')
    p.add_argument('--num_samples', type=int, default=8)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

# -------------------- utils ---------------------------------------
def load_random_images(root, num, img_size, seed=0):
    """
    Randomly walk through class sub-folders (one level) and pick <num> images.
    Returns list[PIL.Image]
    """
    rng = random.Random(seed)
    all_imgs = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_imgs.append(os.path.join(dirpath, f))
    chosen = rng.sample(all_imgs, num)

    pil_imgs = []
    for path in chosen:
        img = Image.open(path).convert('RGB')
        img = img.resize((img_size, img_size), resample=Image.BICUBIC)
        pil_imgs.append(img)
    return pil_imgs

def pil_to_tensor(pil_list):
    """
    [PIL] -> (N,3,H,W) tensor in [-1,1]
    """
    tf = T.Compose([
        T.ToTensor(),                      # [0,1]
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  # -> [-1,1]
    ])
    return torch.stack([tf(im) for im in pil_list])

def tensor_to_uint8(t):
    """
    (-1,1) tensor → uint8 numpy (H,W,C) in RGB
    """
    t = t.clamp(-1,1).add(1).mul(127.5).round().byte()
    return t.permute(1,2,0).cpu().numpy()

# -------------------- main ----------------------------------------
@torch.no_grad()
def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. sample images
    pil_imgs = load_random_images(args.data_path, args.num_samples,
                                  args.img_size, seed=args.seed)
    x = pil_to_tensor(pil_imgs).to(args.device)

    # 2. load VAE
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1,1,2,2,4),
                        ckpt_path=args.vae_path).to(args.device).eval()

    # 3. encode -> decode  (latent std normalised *0.2325 per training code)
    posterior = vae.encode(x)
    latents   = posterior.sample().mul_(0.2325)
    x_rec     = vae.decode(latents)

    # 4. save comparisons
    for idx in range(args.num_samples):
        print(f"Processing sample {idx+1}/{args.num_samples}...")
        orig_np = tensor_to_uint8(x[idx])
        rec_np  = tensor_to_uint8(x_rec[idx])

        # side-by-side concat
        concat = np.concatenate([orig_np, rec_np], axis=1)  # (H, 2W, 3)
        out_path = os.path.join(args.output_dir,
                                f'{args.img_size}_sample_{idx:03d}.png')
        Image.fromarray(concat).save(out_path)
        print(f"Saved {out_path}")

    grid = vutils.make_grid(torch.cat([x, x_rec]), nrow=args.num_samples,
                            normalize=True, scale_each=True)
    vutils.save_image(grid, os.path.join(args.output_dir, 'grid_all.png'))
    print(f"Saved summary grid -> grid_all.png")

if __name__ == '__main__':
    main()
