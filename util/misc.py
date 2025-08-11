import builtins
import datetime
import os
import sys
import time
import json
import glob
from collections import defaultdict, deque
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import inf
else:
    from torch import inf
import copy


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", args=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.args = args

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, global_step=0):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            if is_main_process() and "loss" in self.meters:
                # Determine which loss to use for wandb logging based on wandb_log_mse_only flag
                use_mse_loss = (self.args and 
                               hasattr(self.args, 'wandb_log_mse_only') and self.args.wandb_log_mse_only and
                               "wandb_loss" in self.meters)
                
                if use_mse_loss:
                    # Log MSE loss as main "loss" when wandb_log_mse_only is set
                    loss_value = float(self.meters["wandb_loss"].value)
                else:
                    # Standard logging - use total loss
                    loss_value = float(self.meters["loss"].value)
                
                log_dict = {
                    "eta": float(eta_seconds),                    
                    "loss": loss_value,
                    "lr": float(self.meters["lr"].value),       
                    "time": float(iter_time.value),  
                    "data": float(data_time.value),
                    "max_mem": float(torch.cuda.max_memory_allocated() / MB),
                    "mcmc step size": float(self.meters["mcmc_step_size"].value),
                }
                
                if "langevin_noise_std" in self.meters:
                    log_dict["langevin_noise_std"] = float(self.meters["langevin_noise_std"].value)
                if wandb.run is not None:
                    wandb.log(log_dict, step=global_step + i)

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def find_wandb_run_id_for_resume(resume_path, run_name):
    """
    Find wandb run ID from existing wandb logs that matches the output directory and run_name.
    
    Args:
        resume_path: Path to the checkpoint directory being resumed from
        run_name: The run name to match
        
    Returns:
        str: wandb run ID if found, None otherwise
    """
    # Get the absolute path for comparison
    resume_abs_path = os.path.abspath(resume_path)
    
    # Look for wandb directory in current working directory and parent directories
    current_dir = os.getcwd()
    wandb_dirs_to_check = [
        os.path.join(current_dir, 'wandb'),
        os.path.join(os.path.dirname(current_dir), 'wandb'),
        os.path.join(resume_abs_path, 'wandb'),
        os.path.join(os.path.dirname(resume_abs_path), 'wandb')
    ]
    
    for wandb_base_dir in wandb_dirs_to_check:
        if not os.path.exists(wandb_base_dir):
            continue
            
        # Find all run directories
        run_pattern = os.path.join(wandb_base_dir, "run-*-*")
        run_dirs = glob.glob(run_pattern)
        
        for run_dir in run_dirs:
            try:
                metadata_file = os.path.join(run_dir, "files", "wandb-metadata.json")
                if not os.path.exists(metadata_file):
                    continue
                    
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check if this run matches our criteria
                args_list = metadata.get('args', [])
                
                # Extract run_name from args
                saved_run_name = None
                output_dir = None
                
                for i, arg in enumerate(args_list):
                    if arg == '--run_name' and i + 1 < len(args_list):
                        saved_run_name = args_list[i + 1]
                    elif arg == '--output_dir' and i + 1 < len(args_list):
                        output_dir = args_list[i + 1]
                
                # Match by run_name and optionally by output_dir
                run_name_match = (saved_run_name == run_name)
                output_dir_match = True  # Default to True if no output_dir found
                
                if output_dir:
                    output_abs_path = os.path.abspath(output_dir)
                    output_dir_match = (output_abs_path == resume_abs_path)
                
                if run_name_match and output_dir_match:
                    # Extract run ID from directory name (format: run-YYYYMMDD_HHMMSS-RUNID)
                    dir_name = os.path.basename(run_dir)
                    if dir_name.startswith('run-') and '-' in dir_name:
                        parts = dir_name.split('-')
                        if len(parts) >= 3:
                            run_id = parts[-1]  # Last part is the run ID
                            return run_id
                            
            except (json.JSONDecodeError, IOError, KeyError):
                # Skip this run directory if we can't read it
                continue
    
    return None


def init_wandb(args, is_resuming_checkpoint=False, resume_path=None):
    """
    Initialize wandb with proper resume logic
    
    Args:
        args: Arguments object
        is_resuming_checkpoint: True if successfully loaded a checkpoint
        resume_path: Path to the checkpoint being resumed from
    """
    if not is_main_process() or not hasattr(args, 'run_name') or args.run_name is None:
        return None
    
    wandb_run_id = None
    resume_mode = "allow"  # Default wandb resume mode
    
    if is_resuming_checkpoint and resume_path:
        wandb_run_id = find_wandb_run_id_for_resume(resume_path, args.run_name)
        if wandb_run_id:
            print(f"🔄 Resuming wandb run ID: {wandb_run_id} from checkpoint: {resume_path}")
            resume_mode = "allow"
        else:
            print(f"⚠️  No matching wandb run found for checkpoint resume from {resume_path}, creating new run")
    else:
        print(f"🆕 Creating new wandb run: {args.run_name}")
    
    run = wandb.init(
        project="energy-diffusion", # Jul.24: changed to a new project | prev: ebwm-mar
        config=vars(args),
        name=args.run_name,
        id=wandb_run_id,
        resume=resume_mode
    )
    wandb.define_metric("preview", step_metric="epoch", summary="last")
    
    # Store the wandb run object for step synchronization
    args._wandb_run = run
    return run


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    
    if is_main_process():
        if hasattr(args, 'run_name') and args.run_name is not None:
            # Wandb initialization will be handled after checkpoint loading in main
            pass


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, do_backward=True):
        if do_backward:
            self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), args=None):
    decay = []
    no_decay = []
    mcmc_step_size = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'alpha' in name:
            mcmc_step_size.append(param)  # MCMC step size parameters (separate group)
        elif len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}, 
        {'params': mcmc_step_size, 'weight_decay': 0., 'lr': args.mcmc_step_size_lr_multiplier * args.lr},]
    


def save_model(args, epoch, model, model_without_ddp, optimizer,
               loss_scaler, ema_params=None, epoch_name=None):

    if epoch_name is None:
        epoch_name = str(epoch)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt   = output_dir / "checkpoint-last.pth"
    prev_ckpt   = output_dir / "checkpoint-last-prev.pth"
    tmp_ckpt    = output_dir / "checkpoint-last.tmp"
    epoch_ckpt  = output_dir / f"checkpoint-{epoch_name}.pth"

    if is_main_process() and last_ckpt.exists():
        last_ckpt.replace(prev_ckpt)

    if ema_params is not None:
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            ema_state_dict[name] = ema_params[i]
    else:
        ema_state_dict = None

    to_save = {
        'model'    : model_without_ddp.state_dict(),
        'model_ema': ema_state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch'    : epoch,
        'scaler'   : loss_scaler.state_dict(),
        'args'     : args,
    }

    save_on_master(to_save, tmp_ckpt)
    if is_main_process() and tmp_ckpt.exists():
        os.replace(tmp_ckpt, last_ckpt)

    save_on_master(to_save, epoch_ckpt)



def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x