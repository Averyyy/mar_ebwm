#!/usr/bin/env python
"""
Quick VAE reconstruction sanity-check.

For each sampled image, save side-by-side:
 ┌─────────────┬──────────────┐
 │  original   │ reconstruction│
 └─────────────┴──────────────┘

Usage example
-------------
python vae_recon_quality.py \
  --data_path /path/to/imagenet/val \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --output_dir recon_vis \
  --num_samples 12 \
  --img_size 256
"""
import argparse, random, os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from models.vae import AutoencoderKL
from util.crop import center_crop_arr


# ----------------------------- CLI -----------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_path', type=str,
                   default='pretrained_models/vae/kl16.ckpt')
    p.add_argument('--data_path', type=str, required=True,
                   help='root folder holding class-subdirs or flat images')
    p.add_argument('--output_dir', type=str, default='./vae_recon_output')
    p.add_argument('--num_samples', type=int, default=8)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)

    # optional flags
    p.add_argument('--stochastic', action='store_true',
                   help='sample from posterior instead of using the mode')
    p.add_argument('--latent_scale', type=float, default=1.0,
                   help='multiply latents by this factor before decoding '
                        '(keep 1.0 for correct colours)')
    return p.parse_args()


# ----------------------- helpers -------------------------------
def find_images(root):
    exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(exts):
                yield os.path.join(dp, f)


def load_random_images(root, num, img_size, seed=0):
    rng = random.Random(seed)
    all_imgs = list(find_images(root))
    assert len(all_imgs) >= num, "Not enough images in data_path"
    chosen = rng.sample(all_imgs, num)

    pil_images = []
    for p in chosen:
        img = Image.open(p).convert('RGB')
        img = center_crop_arr(img, img_size)
        img = img.resize((img_size, img_size), Image.BICUBIC)
        pil_images.append(img)
    return pil_images


def pil_list_to_tensor(pil_list):
    tf = T.Compose([
        T.ToTensor(),                             # [0,1]
        T.Normalize([0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5])              # [-1,1]
    ])
    return torch.stack([tf(im) for im in pil_list])


def to_uint8_rgb(t):
    """[-1,1] CHW → HWC uint8 RGB"""
    t = t.clamp(-1, 1).add(1).mul(127.5).round().byte()
    return t.permute(1, 2, 0).cpu().numpy()


# ----------------------------- main -----------------------------
@torch.no_grad()
def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. sample & preprocess images
    pil_imgs = load_random_images(args.data_path, args.num_samples,
                                  args.img_size, seed=args.seed)
    x = pil_list_to_tensor(pil_imgs).to(args.device)

    # 2. init VAE
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                        ckpt_path=args.vae_path).to(args.device).eval()

    # 3. encode → latent → decode
    posterior = vae.encode(x)
    latents = (posterior.sample() if args.stochastic else posterior.mode())
    latents = latents * args.latent_scale          # default 1.0  (no colour shift)
    x_rec = vae.decode(latents)

    # 4. save each pair
    for i in range(args.num_samples):
        orig = to_uint8_rgb(x[i])
        rec  = to_uint8_rgb(x_rec[i])
        concat = np.concatenate([orig, rec], axis=1)
        out_p = os.path.join(args.output_dir,
                             f'sample_{i:03d}.png')
        Image.fromarray(concat).save(out_p)
        print(f"✓ saved {out_p}")

    # 5. nice looking grid (orig top row | rec bottom row)
    grid = vutils.make_grid(torch.cat([x, x_rec]),
                            nrow=args.num_samples,
                            normalize=True, value_range=(-1, 1))
    vutils.save_image(grid,
                      os.path.join(args.output_dir, 'grid_all.png'))
    print("✓ saved grid_all.png")


if __name__ == '__main__':
    main()
