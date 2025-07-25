import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution
import torch_fidelity
import shutil
import cv2
import numpy as np
import os
import copy
import time

import wandb


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


def train_one_epoch(model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mcmc_step_size', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.update(mcmc_step_size=0.0)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
        
    accum_steps = args.grad_accu 
    loss_sum = 0.0
    batch_count = 0
    
    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            if args.use_cached:
                moments = samples
                posterior = DiagonalGaussianDistribution(moments)
            else:
                posterior = vae.encode(samples)

            # normalize the std of latent to be 1. Change it if you use a different tokenizer
            x = posterior.sample().mul_(0.2325)

        # forward
        with torch.cuda.amp.autocast():
            loss = model(x, labels)
            loss = loss / accum_steps

        loss_value = loss.item()
        loss_sum += loss_value * accum_steps #for logging
        batch_count += 1

        # disable loss nan check for now
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=False, do_backward=True)
        if (data_iter_step + 1) % accum_steps == 0 or (data_iter_step + 1) == len(data_loader):
            loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True, do_backward=False)
            optimizer.zero_grad()

            avg_loss = loss_sum / batch_count
            metric_logger.update(loss=avg_loss)
            loss_sum = 0.0
            batch_count = 0

        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value * accum_steps)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if args.model_type == "debt":
            metric_logger.update(mcmc_step_size=model.module.alpha.item())
        elif hasattr(model.module, "use_energy_loss") and model.module.use_energy_loss:
            metric_logger.update(mcmc_step_size=model.module.energy_mlp.alpha.item())
        elif args.model_type == "pure_diffusion" and hasattr(model.module, "alpha"):
            metric_logger.update(mcmc_step_size=model.module.alpha.item())

        loss_value_reduce = misc.all_reduce_mean(avg_loss if batch_count == 0 else loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True):
    model_without_ddp.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "ariter{}-diffsteps{}-temp{}-{}cfg{}-image{}".format(args.num_iter,
                                                                                                     args.num_sampling_steps,
                                                                                                     args.temperature,
                                                                                                     args.cfg_schedule,
                                                                                                     cfg,
                                                                                                     args.num_images))
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    class_num = args.class_num
    assert args.num_images % class_num == 0  # number of images per class must be the same
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                                world_size * batch_size * i + (local_rank + 1) * batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()


        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            with torch.cuda.amp.autocast(enabled=False):
                sampled_tokens = model_without_ddp.sample_tokens(bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
                                                                 cfg_schedule=args.cfg_schedule, labels=labels_gen,
                                                                 temperature=args.temperature)
                sampled_images = vae.decode(sampled_tokens / 0.2325)

        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2
        # print("sampled_images shape:", sampled_images.shape)
        # print("sampled_images", sampled_images)
        # print("min:", sampled_images.min().item(), "max:", sampled_images.max().item(), "mean:", sampled_images.mean().item())
        if torch.isnan(sampled_images).any() or torch.isinf(sampled_images).any():
            print("nan detacted!")
        
        # if misc.is_main_process() and i == 0:
        #     images_to_log = []
        #     for b_id in range(min(5, sampled_images.size(0))):
        #         gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
        #         label = labels_gen[b_id].item()
        #         images_to_log.append(wandb.Image(gen_img, caption=f"Class {label}"))
        #     wandb.log({"eval_images": images_to_log}, step=epoch)

        # distributed save
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()
    time.sleep(10)

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        else:
            # raise NotImplementedError
            pass
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
        if misc.is_main_process() and hasattr(args, 'run_name') and args.run_name is not None:
            wandb.log({"fid": fid, "inception_score": inception_score}, step=epoch)
        postfix = ""
        if use_ema:
           postfix = postfix + "_ema"
        if not cfg == 1.0:
           postfix = postfix + "_cfg{}".format(cfg)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        # remove temporal saving folder
        shutil.rmtree(save_folder)

    torch.distributed.barrier()
    time.sleep(10)


def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args=None):
    import os
    import numpy as np
    import torch
    import util.misc as misc

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    # Ensure cache directory exists
    os.makedirs(args.cached_path, exist_ok=True)

    last_idx_file = os.path.join(args.cached_path, 'last_idx.txt')
    start_iter = 0
    if os.path.exists(last_idx_file):
        try:
            start_iter = int(open(last_idx_file, 'r').read().strip())
        except Exception:
            start_iter = 0

    shard_buffer_m = []
    shard_buffer_f = []
    shard_buffer_lbl = []
    shard_id = 0

    # Parse selected classes for caching
    selected_class_set = None
    if getattr(args, 'cache_classes', ''):
        selected_class_set = set([cls.strip() for cls in args.cache_classes.split(',') if cls.strip()])

    for data_iter_step, (samples, labels, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step < start_iter:
            continue

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        try:
            with torch.no_grad():
                posterior = vae.encode(samples)
                moments = posterior.parameters
                posterior_flip = vae.encode(samples.flip(dims=[3]))
                moments_flip = posterior_flip.parameters

            for i, path in enumerate(paths):
                if selected_class_set is not None:
                    class_dir = path.split(os.sep)[0]
                    if class_dir not in selected_class_set:
                        # skip caching this sample
                        continue

                cache_fmt = getattr(args, 'cache_format', 'npz')
                if cache_fmt == 'ptshard':
                    # Accumulate into shard buffer
                    shard_buffer_m.append(moments[i].half().cpu())
                    shard_buffer_f.append(moments_flip[i].half().cpu())
                    shard_buffer_lbl.append(labels[i].cpu())

                    if len(shard_buffer_m) >= getattr(args, 'cache_shard_size', 20000):
                        save_path = os.path.join(args.cached_path, f'shard_{shard_id:05d}.pt')
                        torch.save({
                            'moments': torch.stack(shard_buffer_m),
                            'moments_flip': torch.stack(shard_buffer_f),
                            'labels': torch.stack(shard_buffer_lbl),
                        }, save_path)
                        shard_id += 1
                        shard_buffer_m, shard_buffer_f, shard_buffer_lbl = [], [], []

                elif cache_fmt == 'pt':
                    # Save as uncompressed torch tensor dict for faster loading
                    save_path = os.path.join(args.cached_path, path + '.pt')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'moments': moments[i].half().cpu(),
                        'moments_flip': moments_flip[i].half().cpu(),
                    }, save_path)
                else:
                    # Fallback to original compressed npz behaviour
                    save_path = os.path.join(args.cached_path, path + '.npz')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.savez(save_path,
                             moments=moments[i].cpu().numpy(),
                             moments_flip=moments_flip[i].cpu().numpy())

            if misc.is_dist_avail_and_initialized():
                torch.cuda.synchronize()

        except Exception as e:
            print(f"[Warning] batch {data_iter_step} warning，paths={paths}，skip：{e}")
            with open(last_idx_file, 'w') as f:
                f.write(str(data_iter_step + 1))
            continue

        with open(last_idx_file, 'w') as f:
            f.write(str(data_iter_step + 1))

    # Flush remaining shard buffer
    if getattr(args, 'cache_format', 'npz') == 'ptshard' and len(shard_buffer_m) > 0:
        save_path = os.path.join(args.cached_path, f'shard_{shard_id:05d}.pt')
        torch.save({
            'moments': torch.stack(shard_buffer_m),
            'moments_flip': torch.stack(shard_buffer_f),
            'labels': torch.stack(shard_buffer_lbl),
        }, save_path)

    return


def log_preview(model, vae, args, epoch, class_id_to_name=None):
    if not misc.is_main_process() or not args.preview:
        return

    device = next(model.parameters()).device     # cuda / cpu

    torch.manual_seed(args.preview_seed)
    np.random.seed(args.preview_seed)

    print("Logging preview images at epoch {} with seed {}".format(epoch, args.preview_seed))
    label_list = [int(x) for x in args.preview_labels.split(',') if x != '']
    labels = torch.tensor(label_list, device=device, dtype=torch.long)

    with torch.no_grad():
        token_latents = model.sample_tokens(
            bsz=len(labels),
            num_iter=args.num_iter,
            cfg=args.cfg,
            cfg_schedule=args.cfg_schedule,
            labels=labels,
            temperature=args.temperature
        )
        imgs = vae.decode(token_latents / 0.2325).clamp(-1, 1)

    out_dir = os.path.join(args.output_dir, 'preview')
    os.makedirs(out_dir, exist_ok=True)

    log_images = []
    for i, lbl in enumerate(label_list):
        # (C,H,W) -> (H,W,C) & [0,255]
        img_np = ((imgs[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"epoch{epoch:04d}_{lbl}.png"), img_np[:, :, ::-1])

        cls_name = class_id_to_name.get(lbl, '') if class_id_to_name else ''
        caption  = f"class {lbl}: {cls_name}" if cls_name else f"class {lbl}"
        log_images.append(wandb.Image(img_np, caption=caption))
    if hasattr(args, 'run_name') and args.run_name is not None:
        wandb.log({"epoch": epoch, "preview": log_images})
    
    
@torch.no_grad()
def validate_one_epoch(model, vae, data_loader, device):
    model.eval()
    loss_sum, n_samples = 0.0, 0
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        posterior = vae.encode(imgs)
        latents   = posterior.sample().mul_(0.2325)

        loss = model(latents, labels)
        bs   = imgs.size(0)
        loss_sum += loss.item() * bs
        n_samples += bs

    avg_loss = misc.all_reduce_mean(loss_sum) / misc.all_reduce_mean(n_samples)
    return avg_loss

# --------------------------- DEBUG FUNCTION (FOR DEBT)---------------------------
def log_preview_half(model, vae, data_loader, args, epoch, class_id_to_name=None):
    """log preview images with half ground-truth tokens"""
    if not misc.is_main_process():
        return

    device = next(model.parameters()).device

    torch.manual_seed(args.preview_seed)
    np.random.seed(args.preview_seed)

    label_list = [int(x) for x in args.preview_labels.split(',') if x != '']
    needed = set(label_list)

    latents_per_label = {}
    with torch.no_grad():
        for imgs_batch, lbls_batch in data_loader:
            imgs_batch = imgs_batch.to(device, non_blocking=True)
            lbls_batch = lbls_batch.to(device, non_blocking=True)

            if args.use_cached:
                posterior = DiagonalGaussianDistribution(imgs_batch)
            else:
                posterior = vae.encode(imgs_batch)
            latents_batch = posterior.sample().mul_(0.2325)

            for j in range(lbls_batch.size(0)):
                lbl_int = int(lbls_batch[j].item())
                if lbl_int in needed and lbl_int not in latents_per_label:
                    latents_per_label[lbl_int] = latents_batch[j:j+1]
                if len(latents_per_label) == len(needed):
                    break
            if len(latents_per_label) == len(needed):
                break

    latents = torch.cat([latents_per_label[lbl] for lbl in label_list], dim=0)

    prefix_len = model.seq_len // 2
    gt_tokens = model.patchify(latents)
    gt_prefix_tokens = gt_tokens[:, :prefix_len, :].contiguous()

    preview_labels_tensor = torch.tensor(label_list, device=device, dtype=torch.long)

    with torch.no_grad():
        token_latents = model.sample_tokens(
            bsz=len(label_list),
            num_iter=args.num_iter,
            cfg=args.cfg,
            cfg_schedule=args.cfg_schedule,
            labels=preview_labels_tensor,
            temperature=args.temperature,
            gt_prefix_tokens=gt_prefix_tokens,
            progress=False,
        )
        imgs = vae.decode(token_latents / 0.2325).clamp(-1, 1)

    run_dir = args.run_name if hasattr(args, "run_name") and args.run_name else "preview_half"
    out_dir = os.path.join(args.output_dir, run_dir)
    os.makedirs(out_dir, exist_ok=True)

    log_images = []
    for i, lbl in enumerate(label_list):
        img_np = ((imgs[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"epoch{epoch:04d}_{lbl}.png"), img_np[:, :, ::-1])

        cls_name = class_id_to_name.get(lbl, "") if class_id_to_name else ""
        caption = f"class {lbl}: {cls_name}" if cls_name else f"class {lbl}"
        log_images.append(wandb.Image(img_np, caption=caption))

    if hasattr(args, 'run_name') and args.run_name is not None:
        wandb.log({"epoch": epoch, "preview_half": log_images})