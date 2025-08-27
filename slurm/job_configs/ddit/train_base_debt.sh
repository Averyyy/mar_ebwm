#!/bin/bash
#SBATCH --job-name=debt-base-1k-64
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm_outputs/debt-base-%j.out

cd /work/hdd/bdta/aqian1/mar_ebwm

torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_port=4744 \
  main_mar.py \
  --run_name debt-base-1k-64-a.01m9000 \
  --img_size 64 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --vae_embed_dim 16 \
  --vae_stride 16 \
  --patch_size 1 \
  --model_type debt \
  --embed_dim 768 \
  --depth 24 \
  --num_heads 24 \
  --mlp_ratio 4 \
  --epochs 100 \
  --warmup_epochs 10 \
  --batch_size 512 \
  --grad_accu 1 \
  --blr 3e-4 \
  --mcmc_num_steps 1 \
  --mcmc_step_size 0.01 \
  --mcmc_step_size_lr_multiplier 9000 \
  --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/debt-base-1k-64 \
  --resume     /work/hdd/bdta/aqian1/mar_ebwm/output/debt-base-1k-64 \
  --use_cached \
  --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64 \
  --seed 42


#   --online_eval \
#   --eval_freq 20 \
#   --num_images 1000 \