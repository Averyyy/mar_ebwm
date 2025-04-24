#!/bin/bash
#SBATCH --job-name=mar-base-1k-energy
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm_outputs/mar-base-energy-%j.out

cd /work/hdd/bdta/aqian1/mar_ebwm

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_port=4733 \
    main_mar.py \
    --run_name mar-base-energy-1k-64-a.01m9000 \
    --img_size 64 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --model mar_base \
    --model_type mar \
    --epochs 100 \
    --warmup_epochs 10 \
    --batch_size 128 \
    --grad_accu 4 \
    --blr 3e-4 \
    --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/mar-base-energy-tiny-alpha-0.01-mult-9000 \
    --use_cached \
    --cached_path /work/hdd/bdta/aqian1/mar_ebwm/data/cached-imagenet1k-64 \
    --online_eval \
    --eval_freq 20 \
    --num_images 100 \
    --seed 42 \
    --mcmc_step_size_lr_multiplier 9000 \
    --mcmc_step_size 0.01
# --resume ./output/mar-base-energy-tiny \



# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py \
# --run_name mar-base-energy \
# --img_size 256 \
# --vae_path pretrained_models/vae/kl16.ckpt \
# --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model mar_base --model_type mar \
# --epochs 50 --warmup_epochs 5 \
# --batch_size 12 --blr 5e-5 \
# --output_dir ./output/mar-base-energy \
# --use_cached --cached_path ./data/cached-tiny-imagenet \
# --seed 42 \

# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py \
# --run_name mar-base-energy-tiny \
# --img_size 256 \
# --vae_path pretrained_models/vae/kl16.ckpt \
# --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model mar_base --model_type mar \
# --batch_size 24 --blr 8e-4 \
#  --diffusion_batch_mul 4 \
# --output_dir ./output/mar-base-energy-tiny \
# --resume ./output/mar-base-energy-tiny \
# --use_cached --cached_path ./data/cached-tiny-imagenet \
# --evaluate
