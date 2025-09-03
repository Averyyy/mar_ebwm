# Energy-Diffusion Outscales Diffusion <br><sub>Official PyTorch Implementation</sub>

[![arXiv](#TODO)&nbsp;

This is a PyTorch/GPU implementation of the paper [Energy-Diffusion Outscales Diffusion](#TODO) (#TODO ICLRbalabala):

This repo contains:

* 🪐 A simple PyTorch implementation of [Standard & Energy diffusion](models/ebm.py)
* ⚡️ Pre-trained class-conditional energy diffusion models trained on ImageNet 64x64 & 256x256

* 🛸 An Energy diffusion [training and evaluation script](main_ebm.py) using PyTorch DDP
* 👏 [Credit] A lot of training pipeline codes are borrowed from [MAR](https://github.com/LTH14/mar), huge thanks to the authors!


## Preparation

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

Download the code:
```
git clone git@github.com:Averyyy/mar_ebwm.git
cd mar_ebwm
```

Setup the environment (make sure you have [conda](https://conda.io/) installed):
If you are on a gh200 gpu, you can use the following command to create an environment called `ebm_gh200`: 
```
chmod +x env_setup/setup_gh200.sh
./env_setup/setup_gh200.sh
```
Warning: running the script would remove & reinstall your current environment called `ebm_gh200`.

Download pre-trained VAE and energy diffusion models:

```
python util/download.py
```

For convenience, our pre-trained EBM models can be downloaded directly here as well:

| EBM Model                                                              | FID-50K | Inception Score | #params | 
|------------------------------------------------------------------------|---------|-----------------|---------|
| [EBM-Base](#TODO) | #TODO    | #TODO           | 130M    |
| [EBM-Large](#TODO) | #TODO    | #TODO           | 458M    |
| [EBM-XLarge](#TODO) | #TODO    | #TODO           | 675M    |

### (Optional) Caching VAE Latents

Given that our data augmentation consists of simple center cropping and random flipping, 
the VAE latents can be pre-computed and saved to `CACHED_PATH` to save computations during EBM training:

```
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
--batch_size 128 \
--data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}
--cache_format ptshard --cache_shard_size 64
```

Cache format:
1. `npz`: default cache format. However, it might influence performance during dataloading on gh200 gpus.
2. `ptshard`: *Recommended* a shard format that is more efficient for dataloading on gh200 gpus.

check `slurm/job_configs/cache_latents.sh`

### Note
If you are using slurm files, remember to change your environment variables on the top of every slurm file you are using.
Check all of path before you run! Remember to cache correctly in the correct path if you are using `--use_cached`.

### Training

## Tips & explanations:
A. Training pipeline:
  1. Wandb logging: 
    a. Install and login to wandb in your terminal, then in `util/misc.py`, in function init_wandb, set `project` param in `wandb.init` to `energy-diffusion`. 
    b. If no run_name is specified, then the run will not be uploaded to wandb web. 
    c. There is a resume logic implemented by default. Just resume from the directory the wandb id and ckpt would be loaded and continue the same wandb logging (if exist).
    d. Preview: set preview labels/freq, There would be preview images shown on the wandb run.
    e. Other features: wandb watch: watch param/gradients to check errors; system section to check gpu memory and utilization;
  2. Caching:
    a. Cache format: 
      - `npz`: default cache format. However, it might influence performance during dataloading on gh200 gpus.
      - `ptshard`: *Recommended* a shard format that is more efficient for dataloading on gh200 gpus. Improves GPU utilization. Also, when using this format, please specify `--cache_format` in the training script.
    b. Resizing: Given a 256 image dataset, just set effective image size to n (eg. 64) to cache a n by n dataset. 
    c. Check `slurm/job_configs/cache_latents.sh` for the script to cache the VAE latents.
    d. Recommended cache location: on nvme/. Recommended cache size for ptshard: 64.
    e. For caching scripts, please check `slurm/job_configs/cache_latents.sh` for types of script to cache the VAE latents.
  3. Evaluation:
  A. Metrics
    All of the metrics are calculated using torch_fidelity. Specify `--eval_real_dataset` to evaluate on selected dataset.
      a. FID: To save a cached fid_stats on selected dataset (that saves time), please check out `util/scripts/compute_fid_stats_64.py` for the script to compute the fid_stats. (It should work for every dataset, just check args in the script for image size, etc.). To specify existing fid_stat, please use `--use_fid_stats` and `--fid_stats_file` to specify the path to the fid_stats.
      b. Other metrics includes: recall, prevision, IS, KID (std/mean), inception score, PRC, etc. They rely solely on eval_real_dataset.
      c. Based on experiments, eval_bsz should be set to 1/4 of your training batchsize.
      d. You should set num_images the integer that is divisible by dataset classes. eg. for imagenet-1k, it should be 1000, 2000, etc.
      e. `--evaluate` is a flag to enable evaluation for a trained model. If you would like to see the metrics during training, please set `--online_eval` to enable online evaluation. Taking 1000 images for example, for 64 by 64 images the eval time is roughly 8 minutes, for 256 by 256 images the eval time is roughly 2 hours (num_sampling_steps: 250)
  B. Validation:
    a. To enable validation, please add `--val` to the training script.
    b. Validation dataset can be specified by `--val_data_path` (for most cases, it is the same as eval_real_dataset)
    c. Validation frequency can be specified by `--val_freq` (default is 25).
    d. Validation batch size can be specified by `--val_batch_size` (You should set it the same as your training batchsize).
  All of the evaluation data are logged to wandb by the global step (which is also important for resuming wandb).

B. Model (energy diffusion, EDM):
  1. All of the diffusion models are located in `models/ebm.py`. By default, it is standard diffusion using vanilla DIT. 
  2. To train any diffusion model, please set `model_type` to `ebm` and `model` to `ebm_base` (or `ebm_large` or `ebm_xlarge`). The default training setting is standard diffusion.
  3. To train energy diffusion, please set `--use_energy`. Other args explanation:
    1. `--use_innerloop_opt`: To enable mcmc during sampling process.
    2. `--mcmc_step_size`: To set the MCMC step size. If `--learnable_mcmc_step_size` is not set, then this would not influence training, just inference.
    3. `--energy_grad_multiplier`: To set the energy gradient multiplier. (returned gradient * multiplier) default set to 1.
    4. `--supervise_energy_landscape`: To supervise the energy landscape during training by adding a contrastive loss. This would increase memory/gpu resources usage, and according to our experiments, it would not improve the performance. 
    5. `--learnable_mcmc_step_size`: To enable learnable mcmc step size by adding a refinement loss that punishes on energy acceptance during mcmc steps. This could improve performance by a little, but it is also very computationally expensive.
    6. `--log_energy_accept_rate`: To log the energy acceptance rate during training to wandb.
    7. `--wandb_log_mse_only`: To log only the mse loss to wandb so that we could compare with other model variants.
    8. `--mcmc_num_steps`: To set the number of mcmc steps during sampling process. Otherwise it is adaptive steps during inference. (which may be slower)
 C. Training:
    1. When mentioning learning rate, the default meaning the base lr (blr), Real LR is calculated by a blr * eff_batch_size / 256
    2. Effective batch size is calculated by batch_size * grad_accu * num_gpus. By enabling gradient accumulation, just specify `--grad_accu` to the number of gradient accumulation steps in your args.
    3. If you are training on a very large batch size (not effective batch size), and getting bumps in GPU utilization between 0% to 100%, please try increasing the `--num_workers`. Eg. batchsize 1024 + `--num_workers 16` would be a good choice. Total number of workers are calculated by num_workers * num_gpus. However, if you are using a small batchsize like 128, then 8 workers per gpu is enough.




Script for the default setting (EDM-Base, 500 diffusion steps, 80 epochs, 128 batchsize, 9e-6 blr):
```
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_ebm.py \
  --run_name ${RUN_NAME} \
  --img_size 256 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ebm \
  --model ebm_base \
  --epochs 20 \
  --warmup_epochs 1 \
  --use_energy \
  --use_innerloop_opt \
  --mcmc_step_size 0.001 \
  --diffusion_timesteps 500 \
  --batch_size 128 \
  --num_workers 32 \
  --blr 9e-6 \
  --use_cached \
  --cached_path ${CACHED_PATH} \
  --cached_format ptshard \
  --output_dir ${OUTPUT_DIR} \
  --online_eval \
  --eval_bsz 32 \
  --eval_real_dataset ${EVAL_PATH} \
  --num_sampling_steps 250 \
  --num_images 1000
```


Args explanations: 
- `model_type`: To train energy diffusion, set to `ebm`.
- (Optional) To train with cached VAE latents, add `--use_cached --cached_path ${CACHED_PATH}` to the arguments. 
Training time with cached latents is ~1d11h on 16 H100 GPUs with `--batch_size 128` (nearly 2x faster than without caching).
Note that this may slightly reduce training speed.

### Evaluation (ImageNet 256x256)

Evaluate MAR-B (DiffLoss MLP with 6 blocks and a width of 1024 channels, 800 epochs) with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_ebm.py \
--model mar_base --diffloss_d 6 --diffloss_w 1024 \
--eval_bsz 256 --num_images 50000 \
--num_iter 256 --num_sampling_steps 100 --cfg 2.9 --cfg_schedule linear --temperature 1.0 \
--output_dir pretrained_models/mar/mar_base \
--resume pretrained_models/mar/mar_base \
--data_path ${IMAGENET_PATH} --evaluate
```

Evaluate MAR-L (DiffLoss MLP with 8 blocks and a width of 1280 channels, 800 epochs) with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_ebm.py \
--model mar_large --diffloss_d 8 --diffloss_w 1280 \
--eval_bsz 256 --num_images 50000 \
--num_iter 256 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
--output_dir pretrained_models/mar/mar_large \
--resume pretrained_models/mar/mar_large \
--data_path ${IMAGENET_PATH} --evaluate
```

Evaluate MAR-H (DiffLoss MLP with 12 blocks and a width of 1536 channels, 800 epochs) with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_ebm.py \
--model mar_huge --diffloss_d 12 --diffloss_w 1536 \
--eval_bsz 128 --num_images 50000 \
--num_iter 256 --num_sampling_steps 100 --cfg 3.2 --cfg_schedule linear --temperature 1.0 \
--output_dir pretrained_models/mar/mar_huge \
--resume pretrained_models/mar/mar_huge \
--data_path ${IMAGENET_PATH} --evaluate
```

- Set `--cfg 1.0 --temperature 0.95` to evaluate without classifier-free guidance.
- Generation speed can be significantly increased by reducing the number of autoregressive iterations (e.g., `--num_iter 64`).

## Directory explanation:
./diffusion: all the diffusion util functions
./env_setup: scripts for setting up environments (on different systems)
./models: the models used in the paper, including DiT, EBM, and vae tokenizer
./output: [ignored] default output folder when you run experiments
./slurm/job_configs: all the slurm scripts
./src: torch-fidelity files
./util/*.py: all the util functions needed in training/inference
./util/scripts: all the scripts needed for preparing dataset, computing metrics, etc.
./util/fid_stats: all the fid stats files when evaluating fids. including imagenet1k-256 and imagenet1k-64 stats.
./main_ebm.py: the main training script for energy diffusion
./main_cache.py: the main script for caching the vae latents
./engine.py: the engine for training/inference


A large portion of codes in this repo is based on [MAR](https://github.com/LTH14/mar) and [DiT](https://github.com/facebookresearch/DiT).

## Contact

If you have any questions, feel free to contact me through email (hangkai2@illinois.edu). Enjoy!
