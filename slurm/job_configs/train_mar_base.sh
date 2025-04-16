#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --job-name=mar-base-1k
#SBATCH --time=24:00:00
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm_coding/logs/slurm_outputs/mar-base-%j.out

source activate mar
cd /work/hdd/bdta/aqian1/mar_ebwm_coding

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=5748 main_mar.py \
  --run_name mar-base-1k \
  --img_size 256 \
  --vae_path /work/hdd/bdta/aqian1/mar_ebwm/pretrained_models/vae/kl16.ckpt \
  --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
  --model mar_base  --model_type mar \
  --diffloss_d 3 --diffloss_w 1024 \
  --epochs 400 --warmup_epochs 100 \
  --batch_size 64 --blr 8.0e-4 \
  --diffusion_batch_mul 4 \
  --output_dir ./output/mar-base-1k \
  --resume ./output/mar-base-1k \
  --use_cached --cached_path /work/hdd/bdta/aqian1/mar_ebwm/data/cached-imagenet-1k
