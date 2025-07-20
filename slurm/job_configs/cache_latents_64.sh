#!/bin/bash
#SBATCH --job-name=mar-cache-1k-64
#SBATCH --time=8:00:00
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm_outputs/cache_latents-%j.out




cd /work/hdd/bdta/aqian1/mar_ebwm

# torchrun --nproc_per_node=4 \
# main_cache.py \
# --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --effective_img_size 256 \
# --batch_size 4096 \
# --data_path /work/nvme/belh/aqian1/imagenet-1k --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64


# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
# main_cache.py \
# --img_size 256 --vae_path /work/hdd/bdta/aqian1/mar_ebwm/pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
# --batch_size 128 \
# --data_path /work/nvme/belh/aqian1/imagenet-1k --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-256

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=5740 \
main_cache.py \
--img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --effective_img_size 64 \
--batch_size 4096 \
--data_path /work/nvme/belh/aqian1/imagenet-1k --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-pt \
--cache_format pt

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=5740 \
main_cache.py \
--img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --effective_img_size 64 \
--batch_size 4096 \
--data_path /work/nvme/belh/aqian1/imagenet-1k --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard \
--cache_format ptshard --cache_shard_size 20000 \
--cache_classes n01440764,n01443537,n01484850,n01491361,n01494475,n01496331,n01498041