#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH --job-name=mar-2xs-1k
#SBATCH --time=06:00:00
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm_coding/logs/slurm_outputs/mar-2xs-%j.out

cd /work/hdd/bdta/aqian1/mar_ebwm

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py \
--run_name mar-2xs-1k \
--img_size 256 \
--vae_path pretrained_models/vae/kl16.ckpt \
--vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model mar_2xs --model_type mar \
--diffloss_d 3 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 \
--batch_size 64 --blr 5e-5\
 --diffusion_batch_mul 4 \
--output_dir ./output/mar-2xs-1k \
--resume ./output/mar-2xs-1k \
--use_cached --cached_path ./data/cached-imagenet-1k
