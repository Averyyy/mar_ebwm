#!/bin/bash
#SBATCH --job-name=standard-diffusion
#SBATCH --array=0-0
#SBATCH --output=${REPO_ROOT}/logs/slurm/energy-diffusion/%A/standard-diffusion-grid-%a.out
#SBATCH --time=48:00:00

# --- Environment Setup ---
# Set these variables for your system:
export REPO_ROOT="/work/hdd/bdta/aqian1/mar_ebwm"  # Change this to your repo path
export DATA_ROOT="/work/hdd/bdta/aqian1/data"     # Change this to your data path
export CACHE_ROOT="/work/nvme/bdta/aqian1/data"   # Change this to your cache path


# --- Grid Search Parameters ---
step_sizes=(0.1)

# --- Calculate parameters for this job ---
step_size=${step_sizes[$SLURM_ARRAY_TASK_ID]}
multiplier=$(echo "${step_size} * 3" | bc -l)

# --- Setup ---
module load cuda/12.6.1
source activate ebm_gh200
cd ${REPO_ROOT}

# --- Parameters ---
BLR=9e-6
BATCH_SIZE=256
EPOCHES=500
WARMUP_EPOCHS=5
MODEL=ebm_small
NUM_EVAL_IMAGES=1000
NUM_EVAL_STEPS=250
IMG_SIZE=256

# --- Run Name and Output Dir ---
RUN_NAME="standard-diffusion-small-256-bz${BATCH_SIZE}-lr_${BLR}-epo${EPOCHES}"
OUTPUT_DIR="${REPO_ROOT}/output/${RUN_NAME}"

# --- Log Parameters ---
echo "--- Starting Standard Diffusion job ${SLURM_ARRAY_TASK_ID} ---"
echo "Base Learning Rate: ${BLR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHES}"
echo "Warmup Epochs: ${WARMUP_EPOCHS}"
echo "Model: ${MODEL}"
echo "Image Size: ${IMG_SIZE}"
echo "Num Evaluation Images: ${NUM_EVAL_IMAGES}"
echo "Run Name: ${RUN_NAME}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "--------------------"

# --- Choose a random master port (per run) ---
if [ -z "${MASTER_PORT}" ]; then
  if command -v shuf >/dev/null 2>&1; then
    MASTER_PORT=$(shuf -i 20000-65000 -n 1)
  else
    MASTER_PORT=$(( (RANDOM % 45000) + 20000 ))
  fi
fi
echo "Using MASTER_PORT=${MASTER_PORT}"

# --- Training Command for Standard Diffusion ---
torchrun \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=${MASTER_PORT} \
  main_ebm.py \
  --run_name ${RUN_NAME} \
  --img_size ${IMG_SIZE} \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ebm \
  --model ${MODEL} \
  --epochs ${EPOCHES} \
  --warmup_epochs ${WARMUP_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --blr ${BLR} \
  --lr_schedule cosine \
  --use_cached \
  --cached_format pt \
  --cached_path ${DATA_ROOT}/cached-imagenet1k-train-256-pt \
  --output_dir ${OUTPUT_DIR} \
  --seed 42 \
  --preview \
  --preview_interval 10 \
  --online_eval \
  --eval_freq 50 \
  --use_fid_stats \
  --eval_real_dataset ${CACHE_ROOT}/imagenet-1k/val \
  --num_sampling_steps ${NUM_EVAL_STEPS} \
  --eval_bsz 128 \
  --num_images ${NUM_EVAL_IMAGES} \
  --val \
  --val_batch_size 256 \
  --val_data_path ${CACHE_ROOT}/imagenet-1k/val


echo "--- Standard Diffusion job ${SLURM_ARRAY_TASK_ID} completed ---"
