# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import ebm
from models.vae import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def safe_load_ckpt(resume_dir):
    return torch.load(resume_dir, map_location='cpu', weights_only=False)


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling with DDP requires GPUs (use sample.py for CPU)."
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Model setup
    latent_size = args.img_size // args.vae_stride
    ebm_model = ebm.__dict__[args.model_size](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        class_num=args.class_num,
        class_dropout_prob=args.label_drop_prob,
        num_diffusion_timesteps=args.diffusion_timesteps,
        num_sampling_steps=int(args.num_sampling_steps),
        use_energy=args.use_energy,
        use_innerloop_opt=args.use_innerloop_opt,
        always_accept_opt_steps=args.always_accept_opt_steps,
        supervise_energy_landscape=args.supervise_energy_landscape,
        mcmc_step_size=args.mcmc_step_size,
        beta_schedule=args.beta_schedule,
        energy_gradient_multiplier=args.energy_grad_multiplier,
    )
    ebm_model.to(device)

    # Load checkpoint
    ckpt_path = args.resume or f"DiT-XL-2-{args.img_size}x{args.img_size}.pt"
    state_dict = safe_load_ckpt(ckpt_path)
    ebm_model.load_state_dict(state_dict['model'])
    ema_state_dict = state_dict['model_ema']
    ema_params = [ema_state_dict[name].cuda() for name, _ in ebm_model.named_parameters()]
    current_state = ebm_model.state_dict()
    for i, (name, _) in enumerate(ebm_model.named_parameters()):
        assert name in current_state
        current_state[name] = ema_params[i]
    ebm_model.load_state_dict(current_state)

    model = ebm_model.dit
    model.eval()
    diffusion = ebm_model.gen_diffusion

    # Load VAE
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim,
                        ch_mult=(1, 1, 2, 2, 4),
                        ckpt_path=args.vae_path).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    using_cfg = args.cfg > 1.0

    # Sample folder
    ckpt_string_name = os.path.basename(ckpt_path).replace(".pth", "")
    folder_name = f"{args.model_size}-{ckpt_string_name}-size-{args.img_size}-vae-{args.vae_stride}-cfg-{args.cfg}-seed-{args.seed}"
    sample_folder_dir = f"{args.output_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Workload division
    n = args.eval_bsz
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_images / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    samples_needed_this_gpu = total_samples // dist.get_world_size()
    iterations = samples_needed_this_gpu // n

    total = 0
    pbar = tqdm(range(iterations)) if rank == 0 and not args.disable_progress_bar else range(iterations)
    for _ in pbar:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.class_num, (n,), device=device)

        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([args.class_num] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z,
            clip_denoised=False, model_kwargs=model_kwargs,
            progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples / 0.2325)
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_images)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
    # parser.add_argument('--model', default='ebm_small', type=str, metavar='MODEL', help='Name of model to train') TODO maybe remove this
    parser.add_argument('--model_size', default='base', type=str, choices=["small", "base", "large", "xlarge"], help='Model Sizes for training')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='VAE checkpoint path')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=0.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=2, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')
    parser.add_argument('--eval_real_dataset', type=str, default=None, help='path to real dataset for KID and PRC metrics')
    parser.add_argument('--kid_subset_size', type=int, default=None, help='KID subset size (default: auto-select based on dataset size)')
    parser.add_argument('--use_fid_stats', action='store_true', help='use precomputed FID statistics file instead of real dataset for FID calculation')
    parser.add_argument('--fid_stats_file', type=str, default='util/fid_stats/adm_in256_stats.npz', help='path to precomputed FID statistics file')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)

    # Training params
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')

    parser.add_argument('--num_sampling_steps', type=str, default="250")
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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

    # caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')
    parser.add_argument('--cached_format', default='npz', choices=['npz', 'pt', 'ptshard'],
                        help='Format of cached latents (npz or pt or ptshard)')
    
    # model selection
    parser.add_argument('--model_type', default='ebm', choices=['ebm'],
                         help="Type of model to run ('ebm' for pure diffusion, use --use_energy for energy-based diffusion)")
    # ---------------- Energy Diffusion args (re-added) ----------------
    parser.add_argument('--dit_model', type=str, default=None, help='[EnergyDiffusion] DiT model size, e.g. DiT-B/4. Overrides embed_dim, depth, num_heads')
    parser.add_argument('--diffusion_timesteps', default=1000, type=int, help='[EnergyDiffusion] Number of diffusion timesteps')
    parser.add_argument('--contrasive_loss_scale', default=0.05, type=float, help='[EnergyDiffusion] Contrastive loss scale for energy supervision')
    parser.add_argument('--mcmc_refinement_loss_scale', default=0.1, type=float, help='[EnergyDiffusion] MCMC refinement loss scale for alpha learning')
    parser.add_argument('--linear_then_mean', action='store_true', help='[EnergyDiffusion] If set, EnergyLayer applies linear layers first then mean pooling')

    # Model architecture parameters
    parser.add_argument('--run_name', default=None, help='name of the run for logging. If not specified, wandb logging is disabled')
    
    # Energy Diffusion parameters  
    parser.add_argument('--mcmc_num_steps', default=None, type=int, help='[EnergyDiffusion] Number of MCMC/energy optimization steps. If None, uses adaptive steps during inference')
    parser.add_argument('--mcmc_step_size', default=0.01, type=float, help='[EnergyDiffusion] MCMC step size')
    parser.add_argument('--use_energy', action='store_true', help='[PureDiffusion] Use IRED-style energy diffusion mode')
    parser.add_argument('--use_innerloop_opt', action='store_true',
                        help='[PureDiffusion] Use inner loop optimization during energy diffusion sampling')
    parser.add_argument('--always_accept_opt_steps', action='store_true',
                        help='[PureDiffusion] When use_innerloop_opt=True, always accept optimization steps regardless of energy evaluation')
    parser.add_argument('--supervise_energy_landscape', action='store_true',
                        help='[PureDiffusion] Use IRED-style energy landscape supervision during training')
    
    parser.add_argument('--learnable_mcmc_step_size', action='store_true',
                        help='[PureDiffusion] Make MCMC step size (alpha) a learnable parameter instead of fixed')
    parser.add_argument('--energy_grad_multiplier', default=1.0, type=float,
                        help='[PureDiffusion] Multiplier for energy gradients used as diffusion score')
    parser.add_argument('--langevin_noise_std', default=0.01, type=float, help='[EnergyMLP] Langevin dynamics noise standard deviation')
    parser.add_argument('--enable_amp_eval', action='store_true',
                        help='[Evaluation] Enable mixed precision (AMP) during evaluation for speedup')
    parser.add_argument('--disable_progress_bar', action='store_true',
                        help='[Evaluation] Disable progress bar during sampling for speedup')
    parser.add_argument(
        '--beta_schedule', default='linear', type=str ,help=''
    )
    parser.add_argument('--grad_accu', default=1, type=int,
                    help='Number of gradient accumulation steps')
    
    parser.add_argument('--mcmc_step_size_lr_multiplier', default=None, type=float,
                    help='Learning rate multiplier for MCMC step size of energymlp (defaults to 3*mcmc_step_size)')
    
    # preview sampling parameters
    parser.add_argument('--preview', action='store_true',
                        help='turn on epoch-wise preview sampling')
    parser.add_argument('--preview_interval', type=int, default=10,
                        help='log preview every N epochs (ignored if --preview_epochs given)')
    parser.add_argument('--preview_epochs', type=str, default='',
                        help='comma-separated epoch numbers to preview, e.g. "0,5,10"')
    parser.add_argument('--preview_labels', type=str, default='0,1,2',
                        help='comma-separated ImageNet class ids to preview')
    parser.add_argument('--preview_seed', type=int, default=42,
                        help='global torch seed so that the SAME noise is reused each epoch')
    parser.add_argument('--preview_only', action='store_true',
                        help='only do preview generation, skip training and wandb initialization')
    
    parser.add_argument('--val_data_path',
                        default='./data/imagenet-1k/val',
                        type=str, help='path to ImageNet val')
    parser.add_argument('--val_batch_size', default=64, type=int)
    parser.add_argument('--val_freq',        default=1,  type=int,
                        help='validate every N epochs (1 = every epoch)')
    parser.add_argument('--val', action='store_true',
                        help='')

    # Debug: half sampling (preview first half tokens with gt, and next half with inferenced tokens)
    parser.add_argument('--test_half_sampling', action='store_true',
                        help='Debug mode: feed half ground-truth tokens then generate the rest')

    # Logging arguments
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--log_energy_accept_rate', action='store_true', help='[PureDiffusion] Log accept rate during opt_step in energy diffusion sampling for each picture')

    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project')
    parser.add_argument('--wandb_log_mse_only', action='store_true', help='[PureDiffusion] When using ebm with supervise_energy_landscape, only log MSE loss to wandb (not total loss)')
    
    # Dtype selection
    parser.add_argument('--train_dtype', default='bf16', type=str, 
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Data type for training (default: bf16)')
    parser.add_argument('--eval_dtype', default='bf16', type=str,
                        choices=['fp16', 'bf16', 'fp32'], 
                        help='Data type for evaluation (default: bf16)')
    parser.add_argument('--auxiliary_eval_dtypes', type=str, default='',
                        help='Comma-separated list of additional eval dtypes to run with separate wandb runs (e.g., "fp16,fp32")')

    parser.add_argument('--syn_dataloader', action='store_true',
                        help='Use synthetic dataloader (random data) instead of loading from disk')
    parser.add_argument('--syn_dataset_len', default=1281167, type=int,
                        help='Number of synthetic samples to generate when using --syn_dataloader')

    # Streaming processing arguments
    parser.add_argument('--use_streaming', action='store_true',
                        help='Enable streaming processing to overlap computation and data transfer for better GPU utilization')
    parser.add_argument('--stream_buffer_size', default=2, type=int,
                        help='Number of CUDA streams to use for streaming processing (default: 2)')
    
    parser.add_argument(
        '--use_flow', action='store_true', help='Flag to start flow matching instead of diffusion'
    )

    # repa params
    parser.add_argument('--cknna_k', default=10, type=int, help='')
    parser.add_argument('--linear_epochs', default=1, type=int, help='')
    parser.add_argument('--n_layers', default=4,type=int, help='')
    parser.add_argument('--layers_start_idx', default=2, type=int,help='')
    parser.add_argument('--cache_latents', action='store_true', help='')
    parser.add_argument('--cache_shard_size', default=2000, type=int,help='')

    #eval_ckpt
    parser.add_argument(
        '--eval_ckpt', default='', type=str, help=""
    )
    
    args = parser.parse_args()
    args.cfg += 1.0
    main(args)