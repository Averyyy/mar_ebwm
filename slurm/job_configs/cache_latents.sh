#!/bin/bash
#SBATCH --job-name=cache-1k-64
#SBATCH --time=24:00:00
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm/cache_latents/cache_latents-%j.out
#SBATCH --gpus-per-node=4




cd /work/hdd/bdta/aqian1/mar_ebwm

# torchrun --nproc_per_node=4  --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=5712 \
# main_cache.py \
# --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --effective_img_size 64 \
# --batch_size 256 \
# --num_workers 12 \
# --data_path /work/nvme/belh/aqian1/imagenet-1k \
# --cached_path /work/hdd/bdta/aqian1/data/test-cached-imagenet1k-64


torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_addr=localhost --master_port=19374 \
  main_cache.py \
  --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
  --effective_img_size 256 \
  --batch_size 128 \
  --num_workers 8 \
  --data_path /work/nvme/belh/aqian1/imagenet-1k \
  --cached_path /work/nvme/bdta/aqian1/data/cached-imagenet1k-256-ptshard-16 \
  --cache_format ptshard --cache_shard_size 16


# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
# main_cache.py \
# --img_size 256 --vae_path /work/hdd/bdta/aqian1/mar_ebwm/pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
# --batch_size 128 \
# --data_path /work/nvme/belh/aqian1/imagenet-1k --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-256

# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=5740 \
# main_cache.py \
# --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --effective_img_size 64 \
# --batch_size 4096 \
# --data_path /work/nvme/belh/aqian1/imagenet-1k --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-pt \
# --cache_format pt

# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=5740 \
# main_cache.py \
# --img_size 64 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --effective_img_size 64 \
# --batch_size 4096 \
# --data_path /work/nvme/belh/aqian1/imagenet-1k --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard \
# --cache_format ptshard --cache_shard_size 20000 \
# --cache_classes n01440764,n01443537,n01484850,n01491361,n01494475,n01496331,n01498041