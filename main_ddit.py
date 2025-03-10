# main_ddit.py
import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import shutil

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder
from util.lr_sched import adjust_learning_rate

from models.vae import AutoencoderKL, DiagonalGaussianDistribution
from models.ddit import DDiT
import copy

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def get_args_parser():
    parser = argparse.ArgumentParser('DDiT training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
    parser.add_argument('--embed_dim', default=1024, type=int,
                        help='Embedding dimension')
    parser.add_argument('--depth', default=16, type=int,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', default=16, type=int,
                        help='Number of attention heads')
    parser.add_argument('--mlp_ratio', default=4.0, type=float,
                        help='MLP hidden dim expansion ratio')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='path to pretrained VAE')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')

    # Generation parameters
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--dropout_prob', default=0.1, type=float,
                        help='Probability of label dropout for classifier-free guidance')
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')
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

    # Diffusion parameters
    parser.add_argument('--learn_sigma', action='store_true',
                        help='Learn the noise prediction sigma')
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float, help='sampling temperature')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')

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
    
    parser.add_argument('--run_name', default='ddit', help='name of the run for logging')

    return parser


def train_one_epoch(
        model, vae, model_params, ema_params, data_loader, optimizer, device, epoch, loss_scaler, log_writer=None,
        args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Adjust learning rate using the original mar scheduler
        adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Get latent representation
        with torch.no_grad():
            if args.use_cached:
                moments = samples
                posterior = DiagonalGaussianDistribution(moments)
            else:
                posterior = vae.encode(samples)

            # Normalize latents
            latents = posterior.sample().mul_(0.2325)
            
        with torch.cuda.amp.autocast():
            if hasattr(model, 'module'):
                loss = model.module(latents,labels)
            else:
                loss = model(latents, labels)
        # Log loss value
        loss_value = loss.item()
        if not np.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if hasattr(model, 'module') and model.module.output_proj.weight.grad is not None:
            print(f"Gradient norm: {torch.norm(model.module.output_proj.weight.grad)}")
        # Backpropagation
        loss_scaler(loss.mean(), optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        
        # Make sure processes are synced
        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        # Update metrics
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Log to tensorboard
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0, use_ema=True):
    model.eval()

    # Setup evaluation steps
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1

    # Create output directory
    save_folder = os.path.join(
        args.output_dir, f"diffsteps{args.num_sampling_steps}-cfg{cfg}-temp{args.temperature}-img{args.num_images}")
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"

    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    # Use EMA parameters if requested
    if use_ema:
        model_state_dict = copy.deepcopy(model.state_dict())
        ema_state_dict = copy.deepcopy(model.state_dict())
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to EMA")
        model.load_state_dict(ema_state_dict)

    # Setup for distributed generation
    class_num = args.class_num
    assert args.num_images % class_num == 0  # Images per class must be the same
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    # Generate images
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        # Get class labels for this batch
        labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                           world_size * batch_size * i + (local_rank + 1) * batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        # Measure generation time
        torch.cuda.synchronize()
        start_time = time.time()

        # Generate samples
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Sample shape: [batch_size, channels, height, width]
                sample_shape = (batch_size, args.vae_embed_dim,
                                args.img_size // args.vae_stride,
                                args.img_size // args.vae_stride)

                # Generate samples
                sampled_latents = model.sample(
                    shape=sample_shape,
                    labels=labels_gen,
                    cfg_scale=cfg,
                    device='cuda'
                )

                # Decode latents to images
                sampled_images = vae.decode(sampled_latents / 0.18215)

        # Measure generation time
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(
                gen_img_cnt, used_time, used_time / gen_img_cnt))

        # Synchronize processes
        torch.distributed.barrier()

        # Process and save images
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2  # Normalize to [0, 1]

        # Save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break

            # Convert to numpy and save
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]  # RGB to BGR for OpenCV
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    # Synchronize processes
    torch.distributed.barrier()
    time.sleep(10)

    # Restore original model
    if use_ema:
        print("Switch back from EMA")
        model.load_state_dict(model_state_dict)

    # Calculate FID and IS if needed
    if log_writer is not None and misc.get_rank() == 0:
        import torch_fidelity

        if args.img_size == 256:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        else:
            raise NotImplementedError

        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )

        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']

        postfix = ""
        if use_ema:
            postfix = postfix + "_ema"
        if not cfg == 1.0:
            postfix = postfix + "_cfg{}".format(cfg)

        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

        # Remove temp folder
        shutil.rmtree(save_folder)

    # Final synchronization
    torch.distributed.barrier()
    time.sleep(10)


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix random seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Setup distributed training
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Setup tensorboard logging
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Setup data transformations
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load dataset
    if args.use_cached:
        dataset_train = CachedFolder(args.cached_path)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    # Setup data sampler
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    # Create data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Load VAE
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Create DDiT model
    model = DDiT(
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        class_num=args.class_num,
        dropout_prob=args.dropout_prob,
        learn_sigma=args.learn_sigma,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul
    )

    print("Model = %s" % str(model))

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    # Move model to device
    model.to(device)
    model_without_ddp = model

    # Calculate effective batch size
    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    # Setup distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Setup optimizer with weight decay
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Setup loss scaler for mixed precision
    loss_scaler = NativeScaler()

    # Resume from checkpoint if available
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu')
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
            print("Loaded optimizer & scheduler!")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    # Evaluation only mode
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer,
                 cfg=args.cfg, use_ema=True)
        return
        
        # 使用处理后的潜变量调用模型
        with torch.cuda.amp.autocast():
            if hasattr(model, 'module'):
                initial_loss = model.module(x, labels)
            else:
                initial_loss = model(x, labels)
        # initial_loss_value = initial_loss.item()
        print(f"Initial loss: {initial_loss}")

    # Main training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # Train for one epoch
        train_one_epoch(
            model, vae,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # Save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="last")

        # Run evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz, log_writer=log_writer,
                     cfg=1.0, use_ema=True)
            if not (args.cfg == 1.0 or args.cfg == 0.0):
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz // 2,
                         log_writer=log_writer, cfg=args.cfg, use_ema=True)
            torch.cuda.empty_cache()

        # Flush tensorboard
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    # Log total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
