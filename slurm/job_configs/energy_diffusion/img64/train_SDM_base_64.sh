#!/bin/bash
# ===== PURPOSE: Standard Diffusion Model training on ImageNet-64 with base model =====
# ===== USAGE: bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/img64/train_SDM_base_64.sh =====
# ===== NOTE: Remeber to change --array to the number of jobs you want to run =====

#SBATCH --job-name=SDM-base-64
#SBATCH --array=0-0
#SBATCH --output=${REPO_ROOT}/logs/slurm/SDM-base-64/%A/SDM-base-64-%a.out
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1

# --- Environment Setup ---
# Set these variables for your system:
export REPO_ROOT="/work/hdd/bdta/aqian1/mar_ebwm"  # Change this to your repo path
export CACHE_ROOT="/work/nvme/bdta/aqian1/data"   # Change this to your cache path
export IMAGENET1K_ROOT="/work/nvme/belh/aqian1/imagenet-1k"


# --- Grid Search Parameters ---

# --- Setup ---
module load cuda/12.6.1
source activate ebm_gh200
cd ${REPO_ROOT}

# --- Parameters ---
NUM_GPUS=1
GRAD_ACCU=1
BLR=9e-6
BATCH_SIZE=1024
EPOCHES=2000
WARMUP_EPOCHS=100
MODEL_TYPE=ebm
MODEL=ebm_base
NUM_EVAL_IMAGES=1000
NUM_EVAL_STEPS=250
IMG_SIZE=64

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCU * NUM_GPUS))

# --- Run Name and Output Dir ---
RUN_NAME="SDM-base-64-bz${EFFECTIVE_BATCH_SIZE}-lr_${BLR}-epo${EPOCHES}-c1k"
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

# --- Training Command for Standard Diffusion ---
torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=$((8489 + SLURM_ARRAY_TASK_ID)) \
  main_ebm.py \
  \
  --run_name ${RUN_NAME} \
  --output_dir ${OUTPUT_DIR} \
  --resume ${OUTPUT_DIR} \
  \
  --img_size ${IMG_SIZE} \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ${MODEL_TYPE} \
  --model ${MODEL} \
  \
  --epochs ${EPOCHES} \
  --warmup_epochs ${WARMUP_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --grad_accu ${GRAD_ACCU} \
  --blr ${BLR} \
  \
  --use_cached \
  --cached_path ${CACHE_ROOT}/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --num_workers 32 \
  \
  --preview \
  --preview_interval 25 \
  --preview_labels 0,1,2,3,430,485,605,726,850 \
  \
  --val \
  --val_batch_size ${BATCH_SIZE} \
  --val_freq 25 \
  --val_data_path ${IMAGENET1K_ROOT}/val


echo "--- Standard Diffusion job ${SLURM_ARRAY_TASK_ID} completed ---"


# ===== if you want to add online evaluation, uncomment the following lines and paste it back to the training command =====

  # --online_eval \
  # --eval_freq 50 \
  # --use_fid_stats \
  # --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
  # --eval_real_dataset ${IMAGENET1K_ROOT}/val \
  # --num_sampling_steps ${NUM_EVAL_STEPS} \
  # --eval_bsz 256 \
  # --num_images ${NUM_EVAL_IMAGES} \


# ===== hardcoded scratch for your convenience to run in a srun interactive shell =====

