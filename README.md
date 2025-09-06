# Energy-Diffusion Outscales Diffusion
_Official PyTorch Implementation_

<!-- arxiv field -->

This is a PyTorch/GPU implementation of the paper [Energy-Diffusion Outscales Diffusion](#TODO) (#TODO ICLRbalabala).

This repo contains:

- 🪐 A simple PyTorch implementation of [Standard & Energy diffusion](models/ebm.py)
- ⚡️ Pre-trained class-conditional energy diffusion models trained on ImageNet 64x64 & 256x256

## Table of Contents
- [Preparation](#preparation)
- [Caching VAE Latents](#optional-caching-vae-latents)
- [Training](#training)
- [Evaluation (ImageNet 256x256)](#evaluation-imagenet-256x256)
- [Directory explanation](#directory-explanation)
- [Contact](#contact)

## Preparation

### Dataset
The repo is using Imagenet-1k which is available for download at [ImageNet](http://image-net.org/download). You could also download the dataset via [huggingface](#TODO:linktohuggingface). After downloading and unzipping, you could use `util/scripts/reorganize_imagenet_inplace.py` to reorganize the dataset from the original structure into a structure like this: 

```
./
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10022.JPEG
│   │   ├── n01440764_10023.JPEG
│   │   └── ...
│   └── ...
├── val/
│   ├── n01440764/
│   │   ├── n01440764_10022.JPEG
│   │   ├── n01440764_10023.JPEG
│   │   └── ...
│   └── ...
├── test/
│   ├── n01440764/
│   │   ├── n01440764_10022.JPEG
│   │   ├── n01440764_10023.JPEG
│   │   └── ...
│   └── ...
```

There is also a file calld `util/imagenet_id_to_name.txt` to map the class id to the class name.

### Installation

Download the code:
```bash
git clone git@github.com:Averyyy/mar_ebwm.git
cd mar_ebwm
```

Set up the environment (make sure you have [conda](https://conda.io/) installed).
If you are on a GH200 GPU, you can use the following command to create an environment called `ebm_gh200`:
```bash
chmod +x env_setup/setup_gh200.sh
./env_setup/setup_gh200.sh
```
> Warning: running the script will remove & reinstall your current environment named `ebm_gh200`.
> Theoretically, the environment should work on CUDA versions <= 12.6.1. (Tested on A100 GPUs.)

Download pre-trained VAE and energy diffusion models:
```bash
python util/download.py
```

For convenience, our pre-trained EBM models can be downloaded directly here as well:

| EBM Model | FID-50K | Inception Score | #params |
|---|---|---|---|
| [EBM-Base](#TODO) | #TODO | #TODO | 130M |
| [EBM-Large](#TODO) | #TODO | #TODO | 458M |
| [EBM-XLarge](#TODO) | #TODO | #TODO | 675M |

### (Optional) Caching VAE Latents

Given that our data augmentation consists of simple center cropping and random flipping, the VAE latents can be pre-computed and saved to `CACHED_ROOT` to save computations during EBM training:

```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
--batch_size 128 \
--data_path ${IMAGENET1K_ROOT} --cached_path ${CACHED_ROOT} \
--cache_format ptshard --cache_shard_size 64
```

Cache format:
1. `npz`: default cache format. However, it might influence performance during dataloading on GH200 GPUs.
2. `ptshard`: recommended shard format that is more efficient for dataloading on GH200 GPUs.

See `slurm/job_configs/cache_latents.sh`.

### Note
If you are using Slurm files, remember to change your environment variables at the top of every Slurm file you use.
Check all paths before you run! Remember to cache in the correct path if you are using `--use_cached`.

## Training

Default training command (EDM-Base, 500 diffusion steps, 80 epochs, batch size 128, base LR 9e-6):
```bash
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
  --cached_path ${CACHED_ROOT} \
  --cached_format ptshard \
  --output_dir ${OUTPUT_DIR} \
  --online_eval \
  --eval_bsz 32 \
  --eval_real_dataset ${EVAL_PATH} \
  --num_sampling_steps 250 \
  --num_images 1000
```

Arguments:
- `model_type`: to train energy diffusion, set to `ebm`.
- (Optional) To train with cached VAE latents, add `--use_cached --cached_path ${CACHED_ROOT}`.

## Guides

### Training pipeline
1. Wandb logging:
   - Install and login to wandb in your terminal, then in `util/misc.py`, in function `init_wandb`, set `project` in `wandb.init` to `energy-diffusion`.
   - If no `run_name` is specified, the run will not be uploaded to wandb web.
   - Resume logic is implemented by default. Resuming from a directory will load the wandb id and checkpoint and continue the same logging (if it exists).
   - Preview: set preview labels/frequency to show preview images on the wandb run.
   - Other features: wandb watch (parameters/gradients), system section (GPU memory and utilization).
2. Caching:
   - Cache format: `npz` (default); `ptshard` (recommended for GH200 GPUs; improves GPU utilization). When using `ptshard`, specify `--cache_format`.
   - Resizing: given a 256 image dataset, set effective image size to n (e.g., 64) to cache an n×n dataset.
   - See `slurm/job_configs/cache_latents.sh` for scripts to cache VAE latents.
   - Recommended cache location: NVMe. Recommended `ptshard` shard size: 64.
3. Evaluation
   - Metrics: computed using torch_fidelity. Specify `--eval_real_dataset`.
   - FID: to precompute `fid_stats` on a selected dataset, see `util/scripts/compute_fid_stats_64.py`. To use existing stats, set `--use_fid_stats` and `--fid_stats_file`.
   - Other metrics: recall, precision, IS, KID (std/mean), PRC, etc. They rely on `--eval_real_dataset`.
   - Set `eval_bsz` to 1/4 of your training batch size.
   - Set `num_images` to a value divisible by the number of classes (e.g., for ImageNet-1k: 1000, 2000, ...).
   - `--evaluate` enables evaluation for a trained model. For metrics during training, set `--online_eval`.
   - Rough eval time for 1000 images: 64×64 ≈ 8 min; 256×256 ≈ 2 h (num_sampling_steps: 250).
4. Validation
   - Enable with `--val`.
   - Dataset via `--val_data_path` (often same as `--eval_real_dataset`).
   - Frequency via `--val_freq` (default 25).
   - Batch size via `--val_batch_size` (match training batch size).
   - All evaluation data are logged to wandb by the global step (important for resuming).

### Model (energy diffusion, EDM)
1. All diffusion models are in `models/ebm.py`. By default, standard diffusion uses vanilla DiT.
2. To train any diffusion model, set `model_type=ebm` and `model` to one of `ebm_base`, `ebm_large`, `ebm_xlarge`.
3. To train energy diffusion, add `--use_energy`. Other args:
   - `--use_innerloop_opt`: enable MCMC during sampling.
   - `--mcmc_step_size`: MCMC step size. If `--learnable_mcmc_step_size` is not set, this affects inference only.
   - `--energy_grad_multiplier`: multiplies returned energy gradient (default 1).
   - `--supervise_energy_landscape`: adds a contrastive loss to supervise the landscape. Increases memory/GPU usage; not observed to improve performance.
   - `--learnable_mcmc_step_size`: learnable step size via a refinement loss penalizing energy acceptance; slightly improves performance but is computationally expensive.
   - `--log_energy_accept_rate`: logs energy acceptance rate to wandb.
   - `--wandb_log_mse_only`: logs only MSE loss to wandb for comparison across variants.
   - `--mcmc_num_steps`: fixed number of MCMC steps during sampling; otherwise adaptive (may be slower).

### Training tips
1. Learning rate: `blr` denotes base LR; real LR = `blr * eff_batch_size / 256`.
2. Effective batch size = `batch_size * grad_accu * num_gpus`. Enable gradient accumulation via `--grad_accu`.
3. If GPU utilization oscillates with very large batch sizes, try increasing `--num_workers` (e.g., batch size 1024 with `--num_workers 16`). Total workers = `num_workers * num_gpus`. For small batch sizes (e.g., 128), 8 workers per GPU are often enough.

## Evaluation (Energy Diffusion)

We provide a ready-to-run evaluation script that compares two acceptance strategies during inner-loop MCMC optimization:
1) always accept optimization steps; 2) vanilla accept/reject based on energy.

### Slurm job (recommended)
Submit the following job (edit paths at the top of the script as needed):

```bash
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/done/eval_EDM_compare_accept.sh
```

Key flags used in the script:
- `--evaluate`: enable evaluation mode
- `--use_fid_stats --fid_stats_file`: reuse precomputed FID stats
- `--eval_real_dataset`: path to the real dataset used for metrics
- `--num_images`: number of generated images for evaluation (should be divisible by classes)
- `--eval_bsz`: evaluation batch size
- `--always_accept_opt_steps`: if set, forces always-accept behavior for inner-loop optimization

### Direct torchrun examples

Always-accept inner-loop optimization:
```bash
torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=6748 \
  main_ebm.py \
  --run_name EDM-eval-s64-step_0.001-diffusion_step-500-c1k-always_accept \
  --img_size 64 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ebm \
  --model ebm_small \
  --use_energy \
  --use_innerloop_opt \
  --always_accept_opt_steps \
  --mcmc_step_size 0.001 \
  --use_cached \
  --cached_path ${CACHE_ROOT}/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --data_path ${IMAGENET1K_ROOT}/imagenet-1k-64 \
  --diffusion_timesteps 500 \
  --num_sampling_steps 250 \
  --evaluate \
  --use_fid_stats \
  --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
  --eval_real_dataset ${IMAGENET1K_ROOT}/imagenet-1k-64/val \
  --eval_bsz 256 \
  --num_images 1000 \
  --output_dir ${REPO_ROOT}/output/EDM-eval-s64-step_0.001-diffusion_step-500-c1k-always_accept \
  --resume ${REPO_ROOT}/output/EDM-step_0.001-cl0-rl0-closs0.05-rloss0.1-lr_9e-6-small-64-bz1024-epo2000-c1k
```

Vanilla accept/reject:
```bash
torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=7748 \
  main_ebm.py \
  --run_name EDM-eval-s64-step_0.001-diffusion_step-500-c1k-vanilla \
  --img_size 64 \
  --use_cached \
  --cached_path ${CACHE_ROOT}/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --data_path ${IMAGENET1K_ROOT}/imagenet-1k-64 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ebm \
  --model ebm_small \
  --use_energy \
  --use_innerloop_opt \
  --mcmc_step_size 0.001 \
  --diffusion_timesteps 500 \
  --num_sampling_steps 250 \
  --use_fid_stats \
  --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
  --eval_real_dataset ${IMAGENET1K_ROOT}/imagenet-1k-64/val \
  --eval_bsz 256 \
  --num_images 1000 \
  --output_dir ${REPO_ROOT}/output/EDM-eval-s64-step_0.001-diffusion_step-500-c1k-vanilla \
  --resume ${REPO_ROOT}/output/EDM-step_0.001-cl0-rl0-closs0.05-rloss0.1-lr_9e-6-small-64-bz1024-epo2000-c1k \
  --evaluate
```

You could decrease inference time by reducing the number of mcmc steps during sampling, but it may introduce some performance degradation.

## Directory explanation
- `diffusion`: all diffusion utility functions
- `env_setup`: scripts for setting up environments (on different systems)
- `models`: models used in the paper, including DiT, EBM, and VAE tokenizer
- `output`: [ignored] default output folder for experiments
- `slurm/job_configs`: all Slurm scripts
- `src`: torch-fidelity files
- `util/*.py`: utility functions for training/inference
- `util/scripts`: scripts for preparing datasets, computing metrics, etc.
- `util/fid_stats`: FID stats files for evaluation (e.g., ImageNet-1k 256 and 64)
- `main_ebm.py`: main training script for energy diffusion
- `main_cache.py`: main script for caching VAE latents
- `engine.py`: training/inference engine

A large portion of codes in this repo is based on [MAR](https://github.com/LTH14/mar) and [DiT](https://github.com/facebookresearch/DiT).

## Contact

If you have any questions, feel free to contact me through email (hangkai2@illinois.edu). Enjoy!
