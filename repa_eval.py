import os
import argparse
import random
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets

from models.vae import AutoencoderKL
from engine_repa import evaluate_model_representation, cache_layers, CachedLatentDataset, train_probe_cached, eval_probe_cached
import wandb


def safe_load_ckpt(resume_dir):
    return torch.load(resume_dir, map_location='cpu', weights_only=False)


def pick_dit_layers(model, n_layers=4, start_idx=2):
    total_blocks = len(model.blocks)
    valid_idxs = list(range(start_idx, total_blocks))
    if len(valid_idxs) < n_layers:
        raise ValueError(f"Not enough blocks ({total_blocks}) to pick {n_layers} layers starting from {start_idx}")
    chosen = np.linspace(start_idx, total_blocks - 1, n_layers, dtype=int)
    return [f"blocks.{i}" for i in chosen]


def load_diffusion_model(args, device):
    from models import ebm
    model = ebm.__dict__[args.model_size](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        class_num=args.class_num,
        class_dropout_prob=args.label_drop_prob,
        num_diffusion_timesteps=getattr(args, 'diffusion_timesteps', 1000),
        num_sampling_steps=int(args.num_sampling_steps),
        use_energy=args.use_energy,
        use_innerloop_opt=args.use_innerloop_opt,
        always_accept_opt_steps=args.always_accept_opt_steps,
        supervise_energy_landscape=args.supervise_energy_landscape,
        mcmc_step_size=args.mcmc_step_size,
        beta_schedule=args.beta_schedule,
        mcmc_num_steps=args.mcmc_num_steps,
        linear_then_mean=args.linear_then_mean,
        log_energy_accept_rate=args.log_energy_accept_rate,
        learnable_mcmc_step_size=args.learnable_mcmc_step_size,
        contrasive_loss_scale=args.contrasive_loss_scale,
        mcmc_refinement_loss_scale=args.mcmc_refinement_loss_scale,
        energy_gradient_multiplier=args.energy_grad_multiplier,
        use_flow=args.use_flow
    )

    if args.resume:
        ckpt = safe_load_ckpt(args.resume)
        model.load_state_dict(ckpt['model'])
        if dist.get_rank() == 0:
            print(f"✅ Loaded diffusion model from {args.resume}")
    
    model.to(device)
    return model


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not running in distributed mode")
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend="nccl", init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def get_args_parser():
    parser = argparse.ArgumentParser('EBM training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
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
    parser.add_argument("--rank", default=0, type=int, help="global rank of the process")
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
    return parser

def main(args):
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{args.gpu}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # dataloaders with DistributedSampler
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    dataset_val = datasets.ImageFolder(args.val_data_path, transform=transform_val)
    dataset_train = datasets.ImageFolder(args.data_path, transform=transform_val)    

    train_sampler = DistributedSampler(dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    val_sampler = DistributedSampler(dataset_val, num_replicas=args.world_size, rank=args.rank, shuffle=False)

    train_loader = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset_val,
                            batch_size=args.batch_size,
                            sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    # vae
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim,
                        ch_mult=(1,1,2,2,4),
                        ckpt_path=args.vae_path).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    # diffusion model
    model = load_diffusion_model(args, device)

    # pick layers
    layer_names = pick_dit_layers(model.dit, args.n_layers, args.layers_start_idx)
    if args.rank == 0:
        print("Layers Used:", layer_names)

    # ----------- caching step (optional) -----------
    if args.cache_latents and args.cached_path:
        if args.rank == 0:
            print(f"📦 Caching features to {args.cached_path}")
        cache_layers(model.dit, vae, train_loader, layer_names, device, args=args, pool_mode="global_mean")
        cache_layers(model.dit, vae, val_loader, layer_names, device, args=args, pool_mode="global_mean", typ='val')
        dist.barrier()  # make sure caching finishes everywhere
        return 

    if rank ==0:
        train_latent_dataset = CachedLatentDataset(args.cached_path, "train", layer_names)
        val_latent_dataset   = CachedLatentDataset(args.cached_path, "val", layer_names)
        
        train_latent_loader = DataLoader(train_latent_dataset, batch_size=128, shuffle=True)
        val_latent_loader   = DataLoader(val_latent_dataset, batch_size=128, shuffle=False)
        num_classes = 1000

        for layer in layer_names:
            print(f"Training linear probe for layer {layer}...")
            probe = train_probe_cached(train_latent_loader, layer, num_classes,
                                                epochs=args.linear_epochs, device=device)

            acc = eval_probe_cached(val_latent_loader, probe, layer, device=device)
            print(f"[REPA] Layer {layer} validation accuracy: {acc:.4f}")
            wandb.log({f"linear_probe/{layer}": acc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser('REPA eval', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.rank == 0 or not dist.is_initialized():
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.run_name)    
    args.data_path = os.path.join(args.data_path, 'train')
    main(args)
