#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=mar-huge-1k
#SBATCH --time=24:00:00
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm_coding/logs/slurm_outputs/mar-huge-%j.out

cd /work/hdd/bdta/aqian1/mar_ebwm_coding

export MASTER_PORT=${MASTER_PORT:-$(( ( RANDOM % 10000 ) ))}

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=$MASTER_PORT main_mar.py \
  --run_name mar-huge-1k-bz2048 \
  --img_size 256 \
  --vae_path /work/hdd/bdta/aqian1/mar_ebwm/pretrained_models/vae/kl16.ckpt \
  --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
  --model mar_huge  --model_type mar \
  --diffloss_d 6 --diffloss_w 1536 \
  --epochs 800 --warmup_epochs 100 \
  --batch_size 64 --grad_accu 32 --blr 8.0e-4 \
  --diffusion_batch_mul 4 \
  --output_dir ./output/mar-huge-1k-bz2048 \
  --resume ./output/mar-huge-1k-bz2048 \
  --use_cached --cached_path /work/hdd/bdta/aqian1/mar_ebwm/data/cached-imagenet-1k