#!/bin/bash
#SBATCH --job-name=mar-comparison
#SBATCH --array=0-0
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm_outputs/%A/mar-comparison-grid-%a.out
#SBATCH --time=24:00:00

# --- Setup ---
module load cuda/12.6.1
source activate mar_gh200
cd /work/hdd/bdta/aqian1/mar_ebwm

# --- Parameters (matching your diffusion experiments) ---
BLR=9e-6
BATCH_SIZE=256
EPOCHES=500
WARMUP_EPOCHS=5
MODEL=mar_small  # MAR small model for similar size comparison
NUM_EVAL_IMAGES=1000
NUM_EVAL_STEPS=64  # MAR uses autoregressive steps, not diffusion steps
IMG_SIZE=256

# --- Run Name and Output Dir ---
RUN_NAME="MAR-small-256-bz${BATCH_SIZE}-lr_${BLR}-epo${EPOCHES}-comparison"
OUTPUT_DIR="/work/hdd/bdta/aqian1/mar_ebwm/output/${RUN_NAME}"

# --- Log Parameters ---
echo "--- Starting MAR Comparison job ${SLURM_ARRAY_TASK_ID} ---"
echo "Model: ${MODEL}"
echo "Base Learning Rate: ${BLR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHES}"
echo "Run Name: ${RUN_NAME}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "--------------------"

# --- Training Command for MAR ---
torchrun \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=$((5748 + SLURM_ARRAY_TASK_ID)) \
  main_mar.py \
  --run_name ${RUN_NAME} \
  --img_size ${IMG_SIZE} \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type mar \
  --model ${MODEL} \
  --epochs ${EPOCHES} \
  --warmup_epochs ${WARMUP_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --blr ${BLR} \
  --lr_schedule cosine \
  --use_cached \
  --cached_format pt \
  --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-train-256-pt \
  --output_dir ${OUTPUT_DIR} \
  --seed 42 \
  --preview \
  --preview_interval 10 \
  --online_eval \
  --eval_freq 50 \
  --use_fid_stats \
  --eval_real_dataset /work/nvme/belh/aqian1/imagenet-1k/val \
  --num_iter ${NUM_EVAL_STEPS} \
  --eval_bsz 128 \
  --num_images ${NUM_EVAL_IMAGES} \
  --val \
  --val_batch_size 256 \
  --val_data_path /work/nvme/belh/aqian1/imagenet-1k/val \
  --mask_ratio_min 0.7 \
  --label_drop_prob 0.1 \
  --attn_dropout 0.1 \
  --proj_dropout 0.1 \
  --buffer_size 64 \
  --grad_clip 3.0

echo "--- MAR Comparison job ${SLURM_ARRAY_TASK_ID} completed ---"