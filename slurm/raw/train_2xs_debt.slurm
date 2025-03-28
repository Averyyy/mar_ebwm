#!/bin/bash
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdta-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH --job-name=debt-2xs
#SBATCH --output=/work/hdd/bdta/aqian1/distillEBWM/logs/experiment.out
#SBATCH --error=/work/hdd/bdta/aqian1/distillEBWM/logs/experiment.err

source activate mar
cd /work/hdd/bdta/aqian1/mar_ebwm

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4739 main_mar.py --model_type debt \
--run_name debt-2xs \
--epochs 50 --warmup_epochs 5 \
--batch_size 32 \
--blr 5e-5 \
--img_size 256 \
--vae_path pretrained_models/vae/kl16.ckpt \
--vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--embed_dim 384 --depth 6 --num_heads 6 --diffusion_batch_mul 4 \
--mcmc_num_steps 10 --mcmc_step_size 0.1 --langevin_dynamics_noise 0.01 --denoising_initial_condition random_noise \
--output_dir ./output/debt/debt-2xs --data_path ./data/tiny-imagenet-200 \
--use_cached --cached_path ./data/cached-tiny-imagenet \
--seed 42 \
--resume ./output/debt/debt-2xs