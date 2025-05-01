#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --job-name=mar-2xs-tiny-energy
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm_outputs/mar-2xs-energy-%j.out

cd /work/hdd/bdta/aqian1/mar_ebwm

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py \
--run_name mar-2xs-energy-tiny \
--img_size 256 \
--vae_path pretrained_models/vae/kl16.ckpt \
--vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model mar_2xs --model_type mar \
--diffloss_d 3 --diffloss_w 1024 \
--epochs 100 --warmup_epochs 20 \
--batch_size 24 --blr 8e-4 \
 --diffusion_batch_mul 4 \
--output_dir ./output/mar-2xs-energy-tiny \
--resume ./output/mar-2xs-energy-tiny \
--use_cached --cached_path ./data/cached-tiny-imagenet


# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py \
# --run_name mar-2xs-energy \
# --img_size 256 \
# --vae_path pretrained_models/vae/kl16.ckpt \
# --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model mar_2xs --model_type mar \
# --epochs 50 --warmup_epochs 5 \
# --batch_size 12 --blr 5e-5 \
# --output_dir ./output/mar-2xs-energy \
# --use_cached --cached_path ./data/cached-tiny-imagenet \
# --seed 42 \

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py \
--run_name mar-2xs-energy-tiny \
--img_size 256 \
--vae_path pretrained_models/vae/kl16.ckpt \
--vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model mar_2xs --model_type mar \
--batch_size 24 --blr 8e-4 \
 --diffusion_batch_mul 4 \
--output_dir ./output/mar-2xs-energy-tiny \
--resume ./output/mar-2xs-energy-tiny \
--use_cached --cached_path ./data/cached-tiny-imagenet \
--evaluate
