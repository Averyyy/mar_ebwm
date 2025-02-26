#!/bin/bash
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdta-delta-gpu
#SBATCH --time=02:00:00
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1
#SBATCH --job-name=mar_ebwm_exp-11
#SBATCH --output=/work/hdd/bdta/aqian1/distillEBWM/logs/experiment.out
#SBATCH --error=/work/hdd/bdta/aqian1/distillEBWM/logs/experiment.err

source activate mar
cd /work/hdd/bdta/aqian1/mar_ebwm
# for mar
# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --model mar_large --diffloss_d 3 --diffloss_w 1024 --epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 --output_dir ./output/ --resume ./output/ --data_path ./data/tiny-imagenet-200 --use_cached --cached_path ./data/cached-tiny-imagenet

# for ddit
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_ddit.py --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --embed_dim 1024 --depth 16 --num_heads 16 --diffusion_batch_mul 4 --output_dir ./output/ddit --resume ./output/ddit  --data_path ./data/tiny-imagenet-200 --use_cached --cached_path ./data/cached-tiny-imagenet