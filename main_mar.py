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
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder

from models.vae import AutoencoderKL
from engine_mar import train_one_epoch, evaluate, log_preview, validate_one_epoch, log_preview_half
import copy
import wandb

def safe_load_ckpt(resume_dir):
    last  = Path(resume_dir) / 'checkpoint-last.pth'
    prev  = Path(resume_dir) / 'checkpoint-last-prev.pth'
    try:
        return torch.load(last, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"⚠️  {last.name} damaged {e}")
        if prev.exists():
            print("↪️  rollback checkpoint-last-prev.pth")
            return torch.load(prev, map_location='cpu', weights_only=False)
        raise

def get_args_parser():
    parser = argparse.ArgumentParser('MAR training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
    parser.add_argument('--model', default='mar_large', type=str, metavar='MODEL',
                        help='Name of model to train')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
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
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=2, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')

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
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)

    # MAR params
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)

    # Diffusion Loss params
    parser.add_argument('--diffloss_d', type=int, default=12)
    parser.add_argument('--diffloss_w', type=int, default=1536)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
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
    
    # ddit selection
    parser.add_argument('--model_type', default='mar', choices=['mar', 'ddit', 'debt'],
                        help="Type of model to run ('mar' for MAR or 'ddit' for DDiT or debt)")

    # DDiT-specific parameters (only used if --model_type ddit is specified)
    parser.add_argument('--run_name', default='ddit', help='name of the run for logging')
    parser.add_argument('--embed_dim', default=1024, type=int,
                        help='[DDiT] Embedding dimension')
    parser.add_argument('--depth', default=16, type=int,
                        help='[DDiT] Number of transformer layers')
    parser.add_argument('--num_heads', default=16, type=int,
                        help='[DDiT] Number of attention heads')
    parser.add_argument('--mlp_ratio', default=4.0, type=float,
                        help='[DDiT] MLP hidden dim expansion ratio')
    parser.add_argument('--learn_sigma', action='store_true',
                        help='[DDiT] Whether to learn the noise prediction sigma')
    
    # DEBT-specific parameters
    parser.add_argument('--mcmc_num_steps', default=10, type=int, help='[DEBT] Number of MCMC steps')
    parser.add_argument('--mcmc_step_size', default=0.01, type=float, help='[DEBT] MCMC step size')
    parser.add_argument('--langevin_dynamics_noise', default=0.01, type=float, help='[DEBT] Langevin dynamics noise std')
    parser.add_argument('--denoising_initial_condition', default='random_noise', type=str, 
                        choices=['random_noise', 'most_recent_embedding', 'zeros'], 
                        help='[DEBT] Initial condition for denoising')
    
    # Energy MLP specific parameters
    parser.add_argument('--use_energy_loss', action='store_true',
                        help='Use energy-based loss instead of diffusion loss')
    parser.add_argument('--langevin_noise_std', default=0.01, type=float, help='[EnergyMLP] Langevin dynamics noise standard deviation')
    
    parser.add_argument('--grad_accu', default=1, type=int,
                    help='Number of gradient accumulation steps')
    
    parser.add_argument('--mcmc_step_size_lr_multiplier', default=1, type=float,
                    help='Learning rate multiplier for MCMC step size of energymlp')
    
    # preview sampling parameters
    parser.add_argument('--preview', action='store_true',
                        help='turn on epoch-wise preview sampling')
    parser.add_argument('--preview_interval', type=int, default=1,
                        help='log preview every N epochs (ignored if --preview_epochs given)')
    parser.add_argument('--preview_epochs', type=str, default='',
                        help='comma-separated epoch numbers to preview, e.g. "0,5,10"')
    parser.add_argument('--preview_labels', type=str, default='0,1,2,3,4,5,6',
                        help='comma-separated ImageNet class ids to preview')
    parser.add_argument('--preview_seed', type=int, default=42,
                        help='global torch seed so that the SAME noise is reused each epoch')
    parser.add_argument('--preview_iter', type=int, default=64,
                        help='num_iter fed to model.sample_tokens() when previewing')
    
    parser.add_argument('--val_data_path',
                        default='/work/nvme/belh/aqian1/imagenet-1k/val/images',
                        type=str, help='path to ImageNet val/images')
    parser.add_argument('--val_batch_size', default=64, type=int)
    parser.add_argument('--val_freq',        default=1,  type=int,
                        help='validate every N epochs (1 = every epoch)')
    parser.add_argument('--val', action='store_true',
                        help='disable validation completely')

    # Debug: half sampling
    parser.add_argument('--test_half_sampling', action='store_true',
                        help='Debug mode: feed half ground-truth tokens then generate the rest')

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

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.use_cached:
        dataset_train = CachedFolder(args.cached_path)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    if args.val:
        transform_val = transforms.Compose([
            transforms.Lambda(lambda im: center_crop_arr(im, args.img_size)),
            transforms.Resize((args.effective_img_size, args.effective_img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        dataset_val = datasets.ImageFolder(args.val_data_path, transform=transform_val)

        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    # define the vae and mar model
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False


    if args.model_type == "ddit":
        from models.ddit import DDiT  # import the DDiT model
        model = DDiT(
            img_size=args.img_size,
            vae_stride=args.vae_stride,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,            # new argument for DDiT
            depth=args.depth,                    # new argument for DDiT
            num_heads=args.num_heads,            # new argument for DDiT
            mlp_ratio=args.mlp_ratio,            # new argument for DDiT
            class_num=args.class_num,
            dropout_prob=args.attn_dropout,      # you can reuse (--dropout_prob) if appropriate
            learn_sigma=args.learn_sigma,        # new argument for DDiT
            num_sampling_steps=args.num_sampling_steps,
            diffusion_batch_mul=args.diffusion_batch_mul,
        )
    elif args.model_type == "debt":
        from models import debt
        # Check if args.model specifies a DEBT variant (debt_base, debt_large, etc.)
        if hasattr(debt, args.model):
            model = debt.__dict__[args.model](
                img_size=args.img_size,
                vae_stride=args.vae_stride,
                patch_size=args.patch_size,
                class_num=args.class_num,
                dropout_prob=args.attn_dropout,
                mcmc_num_steps=args.mcmc_num_steps,
                mcmc_step_size=args.mcmc_step_size,
                langevin_dynamics_noise=args.langevin_dynamics_noise,
                denoising_initial_condition=args.denoising_initial_condition,
            )
        else:
            # Fallback to manual specification
            from models.debt import DEBT
            model = DEBT(
                img_size=args.img_size,
                vae_stride=args.vae_stride,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                depth=args.depth,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                class_num=args.class_num,
                dropout_prob=args.attn_dropout,
                mcmc_num_steps=args.mcmc_num_steps,
                mcmc_step_size=args.mcmc_step_size,
                langevin_dynamics_noise=args.langevin_dynamics_noise,
                denoising_initial_condition=args.denoising_initial_condition,
            )
    else:
        from models import mar
        model = mar.__dict__[args.model](
            img_size=args.img_size,
            vae_stride=args.vae_stride,
            patch_size=args.patch_size,
            vae_embed_dim=args.vae_embed_dim,
            mask_ratio_min=args.mask_ratio_min,
            label_drop_prob=args.label_drop_prob,
            class_num=args.class_num,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            buffer_size=args.buffer_size,
            grad_checkpointing=args.grad_checkpointing,
            # Loss type selection
            use_energy_loss=args.use_energy_loss,
            # DiffLoss parameters
            diffloss_d=args.diffloss_d,
            diffloss_w=args.diffloss_w,
            num_sampling_steps=args.num_sampling_steps,
            diffusion_batch_mul=args.diffusion_batch_mul,
            # Energy loss parameters
            mcmc_step_size=args.mcmc_step_size,
            langevin_noise_std=args.langevin_noise_std,
        )

    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size() * args.grad_accu
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # no weight decay on bias, norm layers, and diffloss MLP    (legacy for mar)
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay, (), args)
    # alpha_params = [param for name, param in model_without_ddp.named_parameters() if 'alpha' in name]
    # other_params = [param for name, param in model_without_ddp.named_parameters() if 'alpha' not in name]
    # param_groups = [
    #     {'params': alpha_params, 'weight_decay': 0.0, 'lr': mcmc_step_size_lr_multiplier * args.lr},
    #     {'params': other_params, 'weight_decay': args.weight_decay, 'lr': args.lr}
    # ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # log grads/params to wandb once
    if misc.is_main_process():
        wandb.watch(model_without_ddp, log="all", log_freq=256)

    # resume training
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        # checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu', weights_only=False)
        checkpoint = safe_load_ckpt(args.resume)
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    # evaluate FID and IS
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer,
                 cfg=args.cfg, use_ema=True)
        return
    
    if os.path.exists('util/imagenet_id_to_name.txt'):
        cls_map = {}
        try:
            with open('util/imagenet_id_to_name.txt') as f:
                for line in f:
                    k, v = line.strip().split(None, 1)
                    cls_map[int(k)] = v
            print(f"☑️ Loaded {len(cls_map)} class id to name mappings")
        except Exception as e:
            print(f"❌ Error loading class id to name mapping: {e}")
            cls_map = {}
    else:
        cls_map = {}
    class_id_to_name = cls_map

    # ------------------------------------------------------------
    # Debug: ground-truth half sampling preview (no training)
    # ------------------------------------------------------------
    if args.test_half_sampling:
        log_preview_half(model_without_ddp, vae, data_loader_train, args, epoch=args.start_epoch, class_id_to_name=class_id_to_name)
        return

    # training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, vae,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="last")

        # preview sampling
        if args.preview:
            epoch_list = [int(x) for x in args.preview_epochs.split(',') if x != '']
            do_preview = (
                (epoch in epoch_list) if epoch_list else
                (epoch % args.preview_interval == 0)
            )
            if do_preview:
                print(f"Preview sampling at epoch {epoch}")
                log_preview(model_without_ddp, vae, args, epoch, class_id_to_name)
                
        # validation
        if (args.val) and (epoch % args.val_freq == 0):
            val_loss = validate_one_epoch(
                model_without_ddp, vae, data_loader_val, device
            )
            if misc.is_main_process():
                print(f"[epoch {epoch}]  validation loss: {val_loss:.6f}")
                if log_writer is not None:
                    log_writer.add_scalar('val_loss', val_loss, epoch)
                wandb.log({"val_loss": val_loss}, step=epoch)


        # online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                     cfg=1.0, use_ema=True)
            if not (args.cfg == 1.0 or args.cfg == 0.0):
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz // 2,
                         log_writer=log_writer, cfg=args.cfg, use_ema=True)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
