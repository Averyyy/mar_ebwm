#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --job-name=ddit-2xs-latest
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm_outputs/ddit-2xs-%j.out

cd /work/hdd/bdta/aqian1/mar_ebwm

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4739 main_mar.py --model_type ddit \
--run_name ddit-2xs-sp \
--epochs 50 --warmup_epochs 5 \
--batch_size 64 \
--blr 5e-5 \
--img_size 256 \
--vae_path pretrained_models/vae/kl16.ckpt \
--vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--embed_dim 384 --depth 6 --num_heads 6 --diffusion_batch_mul 4 \
--output_dir ./output/ddit/ddit-2xs-sp  --data_path ./data/tiny-imagenet-200 \
--use_cached --cached_path ./data/cached-tiny-imagenet \
--seed 42 \
# --online_eval \
# --eval_freq 10 \
# --evaluate