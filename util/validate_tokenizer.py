import argparse
import os
from io import BytesIO
from pathlib import Path

import requests
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from models.vae import AutoencoderKL


def download_images(urls, size=64):
    """Download and resize images."""
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    imgs = []
    for url in urls:
        resp = requests.get(url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        imgs.append(tf(img))
    return torch.stack(imgs)


def tensor_to_uint8(t):
    t = t.clamp(-1, 1).add(1).mul(127.5).round().byte()
    return t.permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    urls = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
        "https://picsum.photos/id/237/256/256",
        "https://picsum.photos/id/238/256/256",
    ]
    x = download_images(urls, size=64).to(args.device)

    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.ckpt)
    vae = vae.to(args.device).eval()

    posterior = vae.encode(x)
    latents = posterior.sample().mul_(0.2325)
    x_rec = vae.decode(latents)

    mse = torch.nn.functional.mse_loss(x_rec, x).item()
    l1 = torch.nn.functional.l1_loss(x_rec, x).item()
    print(f"MSE: {mse:.6f}, L1: {l1:.6f}")

    for idx in range(x.size(0)):
        orig = tensor_to_uint8(x[idx])
        rec = tensor_to_uint8(x_rec[idx])
        concat = np.concatenate([orig, rec], axis=1)
        out_path = os.path.join(args.out_dir, f"sample_{idx:02d}.png")
        Image.fromarray(concat).save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="./pretrained_models/vae/kl16.ckpt")
    p.add_argument("--out_dir", type=str, default="./output/tokenizer_check")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    main(args)