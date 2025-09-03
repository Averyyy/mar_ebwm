#!/bin/bash
#SBATCH --job-name=cache-1k-64
#SBATCH --time=24:00:00
#SBATCH --output=${REPO_ROOT}/logs/slurm/cache_latents/cache_latents-%j.out
#SBATCH --gpus-per-node=4

# --- Environment Setup ---
# Set these variables for your system:
export REPO_ROOT="/work/hdd/bdta/aqian1/mar_ebwm"  # Change this to your repo path
export DATA_ROOT="/work/hdd/bdta/aqian1/data"     # Change this to your data path
export CACHE_ROOT="/work/nvme/bdta/aqian1/data"   # Change this to your cache path




cd ${REPO_ROOT}

# torchrun --nproc_per_node=4  --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=5712 \
# main_cache.py \
# --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --effective_img_size 64 \
# --batch_size 256 \
# --num_workers 12 \
# --data_path ${CACHE_ROOT}/imagenet-1k \
# --cached_path ${DATA_ROOT}/test-cached-imagenet1k-64


torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_addr=localhost --master_port=19374 \
  main_cache.py \
  --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
  --effective_img_size 256 \
  --batch_size 128 \
  --num_workers 8 \
  --data_path ${CACHE_ROOT}/imagenet-1k \
  --cached_path ${CACHE_ROOT}/cached-imagenet1k-256-ptshard-16 \
  --cache_format ptshard --cache_shard_size 16


# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
# main_cache.py \
# --img_size 256 --vae_path ${REPO_ROOT}/pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
# --batch_size 128 \
# --data_path ${CACHE_ROOT}/imagenet-1k --cached_path ${DATA_ROOT}/cached-imagenet1k-256

# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=5740 \
# main_cache.py \
# --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --effective_img_size 64 \
# --batch_size 4096 \
# --data_path ${CACHE_ROOT}/imagenet-1k --cached_path ${DATA_ROOT}/cached-imagenet1k-64-pt \
# --cache_format pt

# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=5740 \
# main_cache.py \
# --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --effective_img_size 64 \
# --batch_size 4096 \
# --data_path ${CACHE_ROOT}/imagenet-1k --cached_path ${DATA_ROOT}/cached-imagenet1k-64-ptshard \
# --cache_format ptshard --cache_shard_size 20000 \
# --cache_classes n01440764,n01443537,n01484850,n01491361,n01494475,n01496331,n01498041