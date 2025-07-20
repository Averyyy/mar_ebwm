import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import util.misc as misc
from util.loader import ImageFolderWithFilename

from models.vae import AutoencoderKL
from engine_mar import cache_latents

from util.crop import center_crop_arr


def get_args_parser():
    parser = argparse.ArgumentParser('Cache VAE latents', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--effective_img_size', default=256, type=int,
                        help='effective image size caching')

    # caching latents
    parser.add_argument('--cached_path', default='', help='path to cached latents')
    parser.add_argument('--cache_format', default='npz', choices=['npz', 'pt', 'ptshard'],
                        help='Format to save cached latents (npz for compressed, pt for uncompressed, ptshard for sharded pt)')
    parser.add_argument('--cache_shard_size', default=20000, type=int,
                        help='Number of samples per shard when --cache_format ptshard')

    # Selective caching options
    parser.add_argument('--cache_classes', default='', type=str,
                        help='Comma-separated list of ImageNet class folder names (e.g. n01440764,n01443537). If provided, only images whose directory name is in this list will be cached.')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        # resize to effective image size
        transforms.Resize((args.effective_img_size, args.effective_img_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset_train = ImageFolderWithFilename(os.path.join(args.data_path, 'train'), transform=transform_train)

    # If selective caching, filter dataset to only requested class folders
    if args.cache_classes:
        selected_set = set([cls.strip() for cls in args.cache_classes.split(',') if cls.strip()])
        filtered_samples = []
        filtered_targets = []
        for path, target in dataset_train.samples:
            class_dir = os.path.basename(os.path.dirname(path))
            if class_dir in selected_set:
                filtered_samples.append((path, target))
                filtered_targets.append(target)

        dataset_train.samples = filtered_samples
        dataset_train.targets = filtered_targets
        # Attribute `imgs` is an alias used internally by ImageFolder
        dataset_train.imgs = filtered_samples

        print(f"Filtered dataset to {len(dataset_train)} samples from classes {selected_set}")
    else:
        print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # Don't drop in cache
    )
    batch = next(iter(data_loader_train))
    print(f"Input image shape: {batch[0].shape}")


    # define the vae
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()

    # training
    print(f"Start caching VAE latents")
    start_time = time.time()
    cache_latents(
        vae,
        data_loader_train,
        device,
        args=args
    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
