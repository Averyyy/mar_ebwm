#!/bin/bash
#SBATCH --job-name=mar-base-1k-2048-continue


cd /work/hdd/bdta/aqian1/mar_ebwm_coding

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=5748 main_mar.py \
  --run_name mar-base-1k-bz2048-1e-4-continue \
  --img_size 256 \
  --vae_path /work/hdd/bdta/aqian1/mar_ebwm/pretrained_models/vae/kl16.ckpt \
  --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
  --model mar_base  --model_type mar \
  --diffloss_d 3 --diffloss_w 1024 \
  --epochs 400 --warmup_epochs 100 \
  --batch_size 64 --grad_accu 1 --blr 1.0e-4 \
  --diffusion_batch_mul 4 \
  --use_cached --cached_path /work/hdd/bdta/aqian1/mar_ebwm/data/cached-imagenet-1k \
  --num_images 1000 \
  --online_eval \
  --eval_freq 50 \
  --output_dir ./output/mar-base-1k-bz2048-1e-4 \
  --resume ./output/mar-base-1k-bz2048-1e-4

# evaluate
# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=5748 main_mar.py \
#   --run_name mar-base-1k-bz2048 \
#   --img_size 256 \
#   --vae_path /work/hdd/bdta/aqian1/mar_ebwm/pretrained_models/vae/kl16.ckpt \
#   --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
#   --model mar_base  --model_type mar \
#   --diffloss_d 6 --diffloss_w 1024 \
#   --epochs 400 --warmup_epochs 100 \
#   --batch_size 128 --grad_accu 8 --blr 8.0e-4 \
#   --diffusion_batch_mul 4 \
#   --use_cached --cached_path /work/hdd/bdta/aqian1/mar_ebwm/data/cached-imagenet-1k \
#   --num_images 1000 \
#   --online_eval \
#   --eval_freq 10 \
#   --output_dir /work/hdd/bdta/aqian1/mar_ebwm_coding/output/mar-base-1k-bz2048 \
#   --resume /work/hdd/bdta/aqian1/mar_ebwm_coding/output/mar-base-1k-bz2048 \
#   --evaluate

  
