#!/bin/bash
#SBATCH --job-name=mar-cache-1k-512


cd /work/hdd/bdta/aqian1/mar_ebwm

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --effective_img_size 64 \
--batch_size 4096 \
--data_path /work/nvme/belh/aqian1/imagenet-1k --cached_path /work/hdd/bdta/aqian1/mar_ebwm/data/cached-imagenet1k-64
