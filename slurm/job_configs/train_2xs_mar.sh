#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --job-name=mar-2xs-1k
#SBATCH --time=06:00:00
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm_outputs/mar-2xs-%j.out

source activate mar
cd /work/hdd/bdta/aqian1/mar_ebwm

# on 1k
# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py \
# --run_name mar-2xs-1k \
# --img_size 256 \
# --vae_path pretrained_models/vae/kl16.ckpt \
# --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model mar_2xs --model_type mar \
# --diffloss_d 3 --diffloss_w 1024 \
# --epochs 50 --warmup_epochs 5 \
# --batch_size 64 --blr 5e-5\
#  --diffusion_batch_mul 4 \
# --output_dir ./output/mar-2xs-1k \
# --resume ./output/mar-2xs-1k \
# --use_cached --cached_path ./data/cached-imagenet-1k

# small experiment
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py \
--run_name mar-2xs-energy \
--img_size 256 \
--vae_path pretrained_models/vae/kl16.ckpt \
--vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model mar_2xs --model_type mar \
--diffloss_d 3 --diffloss_w 1024 \
--epochs 50 --warmup_epochs 5 \
--batch_size 64 --blr 5e-5\
 --diffusion_batch_mul 4 \
--output_dir ./output/mar-2xs-energy \
--resume ./output/mar-2xs-energy \
--use_cached --cached_path ./data/cached-tiny-imagenet
