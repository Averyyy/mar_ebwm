#!/bin/bash

export WANDB_MODE=offline

# ---- Minimal hyperparams for testing ----
RUN_NAME="test"
MODEL_SIZE="base"                 # match your model naming convention
BATCH_SIZE_PER_DEVICE=64           # smaller batch for local GPU
LR=0.0001
GRAD_ACCU=1
IMAGENET1K_ROOT="/work/nvme/bdjz/shared/image_datasets/imagenet1k"  # <-- update

torchrun --nproc_per_node=1 main_ebm.py \
  --run_name ${RUN_NAME} \
  --model_type "ebm" \
  --model_size ${MODEL_SIZE} \
  \
  --use_energy \
  --use_innerloop_opt \
  --mcmc_step_size 0.0001 \
  --energy_grad_multiplier 1 \
  \
  --diffusion_timesteps 1000 \
  \
  --epochs 1 \
  --warmup_epochs 0 \
  --batch_size ${BATCH_SIZE_PER_DEVICE} \
  --lr ${LR} \
  --grad_accu ${GRAD_ACCU} \
  --weight_decay 0.02 \
  --data_path ${IMAGENET1K_ROOT}
  \
  --img_size 256 \
  --vae_path "pretrained_models/vae/kl16.ckpt" \
  --num_workers 2 \
  \
  --output_dir "./logs/output/${RUN_NAME}" \
  --wandb_entity "ebwm_nlp" \
  --wandb_project "energy_diffusion_final" \
  \
  --preview \
  --preview_interval 20 \
  --preview_labels 0,1,2,3,430,485,605,726,850 \
  \
  --val \
  --val_batch_size ${BATCH_SIZE_PER_DEVICE} \
  --val_freq 20 \
  --val_data_path ${IMAGENET1K_ROOT}/val \
  --use_flow