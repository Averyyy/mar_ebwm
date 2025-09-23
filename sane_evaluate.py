import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from pathlib import Path
import datetime
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torch.distributed import is_initialized, destroy_process_group
from argparse import ArgumentParser

class ImageOnlyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.ds = datasets.ImageFolder(root=root, transform=transform)
        print(len(self.ds))

    def __len__(self):
        return min(len(self.ds), 50000)
    
    def __getitem__(self, idx):
        image, _ = self.ds[idx]  # discard label
        return image


def main(args):
    accelerator_config = ProjectConfiguration(
        project_dir=args.output_dir,
        automatic_checkpoint_naming=False,
        total_limit=None,
    )
    accelerator = Accelerator(
        log_with='wandb',
        project_config=accelerator_config,
        gradient_accumulation_steps=1,
        mixed_precision='fp16',
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.project_name,
            config=vars(args),
            init_kwargs={
                'wandb': {'name': args.run_name, "entity": args.wandb_entity}
            }
        )

    device = accelerator.device

    fid_metric = FrechetInceptionDistance(normalize=True, dist_sync_on_step=True).to(device)
    is_metric = InceptionScore(normalize=True, dist_sync_on_step=True).to(device)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    
    val_dataset = ImageOnlyDataset(args.val_dataset, transform=transform)
    gen_dataset = ImageOnlyDataset(args.gen_dataset, transform=transform)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=False
    )

    gen_loader = DataLoader(
        gen_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=False
    )

    val_loader, gen_loader = accelerator.prepare(val_loader, gen_loader)

    for samples in tqdm(val_loader, leave=False, desc="Real", disable=not accelerator.is_main_process):
        samples= samples.to(device)

        fid_metric.update(samples, real=True)
    
    for samples in tqdm(gen_loader, leave=False, desc="Gen", disable=not accelerator.is_main_process):
        samples = samples.to(device)

        fid_metric.update(samples, real=False)
        is_metric.update(samples)

    fids = accelerator.gather(torch.tensor(fid_metric.compute().item(), device=device))
    is_mean, is_std = is_metric.compute()
    is_mean = accelerator.gather(torch.tensor(is_mean.item(), device=device))
    is_std = accelerator.gather(torch.tensor(is_std.item(), device=device))

    if accelerator.is_main_process:
        final_fid = fids.mean().item()
        final_is_mean, final_is_std = is_mean.mean().item(), is_std.mean().item()

        accelerator.log({
            'FID': final_fid,
            'IS_MEAN': final_is_mean,
            'IS_STD': final_is_std
        })

        print("FID:", final_fid)
        print(f"IS: {final_is_mean:.4f}±{final_is_std:.4f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, help='', default=512
    )

    parser.add_argument(
        '--num_workers', type=int, help='', default=64
    )

    parser.add_argument(
        '--val_dataset', type=str, help='', default='/work/nvme/bdjz/shared/image_datasets/imagenet1k/validation'
    )

    parser.add_argument(
        '--gen_dataset', type=str, help='', required=True
    )

    parser.add_argument(
        '--img_size', type=int, help='', default=256
    )

    parser.add_argument(
        '--output_dir', type=str, help='', default='logs'
    )

    parser.add_argument(
        '--project_name', help='', type=str, default='metric-stats'
    )

    parser.add_argument(
        '--run_name', help='', type=str, default='test'
    )

    parser.add_argument(
        '--wandb_entity', help='', type=str, default='ebwm_nlp'
    )

    args = parser.parse_args()
    main(args)
    if is_initialized():
        destroy_process_group()
