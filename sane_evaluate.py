#!/usr/bin/env python3
import argparse
import os
import shutil
import numpy as np
import wandb
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch_fidelity
import sys
import torch

class ImageOnlyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.ds = datasets.ImageFolder(root=root, transform=transform)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        image, _ = self.ds[idx]  # discard label
        return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True,
                        help="Model architecture name (e.g., DiT-XL).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint used during sampling.")
    parser.add_argument("--sample-dir", type=str, required=True,
                        help="Base samples dir (the parent folder where sample_ddp created model-specific folder).")
    parser.add_argument("--imagenet-dir", type=str, required=True,
                        help="Path to ImageNet validation folder (ImageFolder layout).")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--vae_stride", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fid-statistics-file", type=str,
                        default="util/fid_stats/adm_in256_stats.npz",
                        help="Optional precomputed FID stats file (.npz).")
    parser.add_argument("--keep-npz", action="store_true",
                        help="If set, do not delete the generated .npz file after evaluation.")
    parser.add_argument("--keep-pngs", action="store_true",
                        help="If set, do not delete the generated PNG folder after evaluation.")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="If set, log results to this wandb project.")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="If set, log results to this wandb project.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Optional name for the wandb run.")
    return parser.parse_args()


class NPZDataset(Dataset):
    """Loads samples from a .npz file produced by sample_ddp.create_npz_from_sample_folder"""
    def __init__(self, path, transform=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f".npz file not found: {path}")
        data = np.load(path)
        if "arr_0" in data:
            self.data = data["arr_0"]
        else:
            self.data = data[data.files[0]]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]  # HWC uint8
        img_t = torch.from_numpy(img).permute(2, 0, 1).to(dtype=torch.uint8)

        return img_t

def build_folder_name(args):
    model_string_name = args.model_type.replace("/", "-")
    ckpt_string_name = os.path.basename(args.resume).replace(".pth", "") if args.resume else "pretrained"
    folder_name = (
        f"{model_string_name}-{ckpt_string_name}-"
        f"size-{args.img_size}-vae-{args.vae_stride}-"
        f"cfg-{args.cfg}-seed-{args.seed}"
    )
    return folder_name


def main():
    args = parse_args()
    args.cfg += 1


    folder_name = build_folder_name(args)
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    npz_path = f"{sample_folder_dir}.npz"

    print(f"[INFO] Looking for PNG folder: {sample_folder_dir}")
    print(f"[INFO] Looking for NPZ file: {npz_path}")

    # Reference dataset
    ref_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.PILToTensor()
    ])
    ref_dataset = ImageOnlyDataset(root=args.imagenet_dir, transform=ref_transform)

    # Generated dataset input
    if os.path.exists(npz_path):
        print(f"[INFO] Found .npz file: {npz_path}")
        gen_dataset = NPZDataset(npz_path)
        gen_input = gen_dataset
    elif os.path.isdir(sample_folder_dir):
        print(f"[INFO] Found PNG folder: {sample_folder_dir}")
        gen_input = sample_folder_dir
    else:
        print(f"[ERROR] Neither PNG folder ({sample_folder_dir}) nor NPZ ({npz_path}) found.", file=sys.stderr)
        sys.exit(2)

    # TODO set prec rec = 0 and False, try swapping input1 and 2

    # Run metrics
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=gen_input,
        input2=ref_dataset,
        fid_statistics_file=args.fid_statistics_file,
        cuda=True,
        isc=True,
        fid=True,
        prc=True,
        verbose=False,
        samples_find_deep=True,
    )

    fid = metrics_dict.get("frechet_inception_distance")
    inception_score = metrics_dict.get("inception_score_mean")
    precision = metrics_dict.get("precision")
    recall = metrics_dict.get("recall")

    print("[RESULTS]")
    if fid is not None:
        print(f"FID: {fid:.4f}")
    if inception_score is not None:
        print(f"Inception Score: {inception_score:.4f}")
    if precision is not None and recall is not None:
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    # ---- wandb logging ----
    if args.wandb_project:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        log_dict = {}
        if fid is not None:
            log_dict["FID"] = fid
        if inception_score is not None:
            log_dict["InceptionScore"] = inception_score
        if precision is not None:
            log_dict["Precision"] = precision
        if recall is not None:
            log_dict["Recall"] = recall
        wandb.log(log_dict)
        wandb.finish()

    # Cleanup
    if os.path.isdir(sample_folder_dir) and not args.keep_pngs:
        print(f"[CLEANUP] Removing PNG folder {sample_folder_dir}")
        shutil.rmtree(sample_folder_dir, ignore_errors=True)
    if os.path.exists(npz_path) and not args.keep_npz:
        print(f"[CLEANUP] Removing NPZ file {npz_path}")
        os.remove(npz_path)


if __name__ == "__main__":
    main()
