#!/bin/bash
# ===== PURPOSE: Energy Diffusion Model training on ImageNet-256 with base model =====
# ===== USAGE: bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/train_EDM_base_256_bz256.sh =====
# ===== NOTE: Remeber to change --array to the number of jobs you want to run =====

#SBATCH --job-name=EDM_LR_GS_BS_1024
#SBATCH --array=0
#SBATCH --output=logs/slurm/256/EDM_LR_GS_BS_1024-%A-%a.log
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=4
mkdir -p logs/slurm/256/

# --- Environment Setup ---
# Set these variables for your system:
export REPO_ROOT="/work/hdd/bcsi/agladstone/mar_ebwm"  # Change this to your repo path
export CACHE_ROOT="/work/nvme/bdta/aqian1/data"   # Change this to your cache path
export IMAGENET1K_ROOT="/work/nvme/belh/aqian1/imagenet-1k"

# --- Grid Search Parameters ---
# mcmc_steps=(0.0001)
learning_rates=(0.0001)
# --- Calculate parameters for this job (generic for any list length) ---
len_lr=${#learning_rates[@]}
# len_ms=${#mcmc_steps[@]}

total=$((len_lr))

task_id=${SLURM_ARRAY_TASK_ID}

# If request too many tasks, exit redundant tasks
if [ -z "${task_id}" ]; then
  echo "[INFO] SLURM_ARRAY_TASK_ID is not set. Manually setting it to 0 to allow for running as a bash script." >&2
  SLURM_ARRAY_TASK_ID=0
fi

# lr_idx=$(( task_id % len_ms ))

# step_size=${mcmc_steps[$mcmc_idx]}
learning_rate=${learning_rates[$SLURM_ARRAY_TASK_ID]}
echo "learning_rate: ${learning_rate}"

# --- Setup ---
module load cuda/12.6.1
source activate ebm_gh200
cd ${REPO_ROOT}

# --- Parameters ---
NUM_GPUS=1
NUM_NODES=1
GRAD_ACCU=2
lr=${learning_rate}
BATCH_SIZE=128
EPOCHES=1000
WARMUP_EPOCHS=10
MODEL_TYPE=ebm
MODEL=ebm_base
NUM_EVAL_IMAGES=1000
NUM_EVAL_STEPS=250
IMG_SIZE=256
ENERGY_GRAD_MULTIPLIER=1
DIFFUSION_TIMESTEPS=1000
EVAL_BATCH_SIZE=$((BATCH_SIZE / 4))
step_size=0.0001

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCU * NUM_GPUS * NUM_NODES))


# --- Run Name and Output Dir ---
RUN_NAME="EDM-256-base-lr${lr}-no_wd-timesteps${DIFFUSION_TIMESTEPS}-bz${EFFECTIVE_BATCH_SIZE}-epo${EPOCHES}-c1k"
OUTPUT_DIR="${REPO_ROOT}/output/${RUN_NAME}"

# --- Log Parameters ---
echo "--- Starting Energy Diffusion Grid Search job ${SLURM_ARRAY_TASK_ID} ---"
echo "Grid Parameters:"
echo "  MCMC Step Size: ${step_size}"
echo "  Learning Rate: ${lr}"
echo "  Diffusion Time Steps: ${DIFFUSION_TIMESTEPS}"
echo "Training Parameters:"
echo "  Batch Size: ${EFFECTIVE_BATCH_SIZE}"
echo "  Epochs: ${EPOCHES}"
echo "  Warmup Epochs: ${WARMUP_EPOCHS}"
echo "  Model: ${MODEL}"
echo "  Image Size: ${IMG_SIZE}"
echo "  Num Evaluation Images: ${NUM_EVAL_IMAGES}"
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
# TODO check this code make sure works for multinode and will work for bash as well

# --- Training Command for Energy Diffusion ---
torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=localhost \
  --master_port=${MASTER_PORT} \
  --nnodes=${NUM_NODES} \
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
  --lr ${lr} \
  --grad_accu ${GRAD_ACCU} \
  --weight_decay 0.0 \
  \
  --use_energy \
  --use_innerloop_opt \
  --mcmc_step_size ${step_size} \
  --energy_grad_multiplier ${ENERGY_GRAD_MULTIPLIER} \
  \
  --diffusion_timesteps ${DIFFUSION_TIMESTEPS} \
  \
  --use_cached \
  --cached_path ${CACHE_ROOT}/cached-imagenet1k-256-ptshard-16 \
  --cached_format ptshard \
  --num_workers 8 \
  \
  --preview \
  --preview_interval 20 \
  --preview_labels 0,1,2,3,430,485,605,726,850 \
  \
  --val \
  --val_batch_size ${BATCH_SIZE} \
  --val_freq 20 \
  --val_data_path ${IMAGENET1K_ROOT}/val


echo "--- Energy Diffusion Grid Search job ${SLURM_ARRAY_TASK_ID} completed ---"


# ===== if you want to add online evaluation, uncomment the following lines and paste it back to the training command =====

  # --online_eval \
  # --eval_freq 20 \
  # --use_fid_stats \
  # --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
  # --eval_real_dataset ${IMAGENET1K_ROOT}/val \
  # --num_sampling_steps ${NUM_EVAL_STEPS} \
  # --eval_bsz ${EVAL_BATCH_SIZE} \
  # --num_images ${NUM_EVAL_IMAGES} \