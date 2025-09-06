#!/bin/bash
#SBATCH --job-name=EDM-base-bz-test
#SBATCH --array=0-0
#SBATCH --output=logs/slurm/EDM-base-bztest/1024/%A/EDM-base-%a.out
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=4

# ===== PURPOSE: Grid search training for Energy Diffusion Model (Base) on ImageNet-256 with batch size 1024 =====
# ===== USAGE: bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/train_EDM_base_256_bz1024.sh =====
# ===== NOTE: Remeber to change --array to the number of jobs you want to run =====

# --- Environment Setup ---
# Set these variables for your system:
export REPO_ROOT="/work/hdd/bdta/aqian1/mar_ebwm"  # Change this to your repo path
export CACHE_ROOT="/work/nvme/bdta/aqian1/data"   # Change this to your cache path
export IMAGENET1K_ROOT="/work/nvme/belh/aqian1/imagenet-1k"



# --- Grid Search Parameters ---
mcmc_steps=(0.0001 )
learning_rates=(3e-6 )
# --- Calculate parameters for this job (generic for any list length) ---
len_lr=${#learning_rates[@]}
len_ms=${#mcmc_steps[@]}

total=$((len_lr * len_ms))

task_id=${SLURM_ARRAY_TASK_ID}

# If request too many tasks, exit redundant tasks
if [ -z "${task_id}" ]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID is not set. Are you running this as an array job?" >&2
  exit 1
fi
if [ "${task_id}" -ge "${total}" ]; then
  echo "[INFO] SLURM_ARRAY_TASK_ID ${task_id} >= total combinations ${total}. Exiting redundant array task." >&2
  exit 0
fi

lr_idx=$(( task_id / len_ms ))
mcmc_idx=$(( task_id % len_ms ))

step_size=${mcmc_steps[$mcmc_idx]}
learning_rate=${learning_rates[$lr_idx]}

# --- Setup ---
module load cuda/12.6.1
source activate mar_gh200
cd ${REPO_ROOT}

# --- Parameters ---
NUM_GPUS=4
GRAD_ACCU=2
BLR=${learning_rate}
BATCH_SIZE=128
EPOCHES=80
WARMUP_EPOCHS=4
MODEL_TYPE=ebm
MODEL=ebm_base
NUM_EVAL_IMAGES=1000
NUM_EVAL_STEPS=250
IMG_SIZE=256
ENERGY_GRAD_MULTIPLIER=1
DIFFUSION_TIMESTEPS=500
EVAL_BATCH_SIZE=$((BATCH_SIZE / 4))

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCU * NUM_GPUS))


# --- Run Name and Output Dir ---
RUN_NAME="EDM-256-base-lr${BLR}-timesteps${DIFFUSION_TIMESTEPS}-bz${EFFECTIVE_BATCH_SIZE}-epo${EPOCHES}-c1k"
OUTPUT_DIR="${REPO_ROOT}/output/${RUN_NAME}"

# --- Log Parameters ---
echo "--- Starting Energy Diffusion Grid Search job ${SLURM_ARRAY_TASK_ID} ---"
echo "Grid Parameters:"
echo "  MCMC Step Size: ${step_size}"
echo "  Learning Rate: ${BLR}"
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

# --- Training Command for Energy Diffusion ---
torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=localhost \
  --master_port=${MASTER_PORT} \
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
  --grad_accu 2 \
  --blr ${BLR} \
  \
  --use_energy \
  --use_innerloop_opt \
  --mcmc_step_size ${step_size} \
  --energy_grad_multiplier ${ENERGY_GRAD_MULTIPLIER} \
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


# ===== hardcoded scratch for your convenience to run in a srun interactive shell for testing =====

# torchrun \
#   --nproc_per_node=1 \
#   --master_addr=localhost \
#   --master_port=6748 \
#   main_ebm.py \
#   \
#   --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/test-loggging \
#   \
#   --img_size 64 \
#   --vae_path pretrained_models/vae/kl16.ckpt \
#   --model_type ebm \
#   --model ebm_base \
#   \
#   --epochs 80 \
#   --warmup_epochs 4 \
#   --batch_size 128 \
#   --grad_accu 1 \
#   --blr 3e-6 \
#   \
#   --use_energy \
#   --use_innerloop_opt \
#   --diffusion_timesteps 500 \
#   \
#   --use_cached \
#   --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard-c7 \
#   --cached_format ptshard \
#   --num_workers 32 \
#   \
#   --online_eval \
#   --eval_freq 20 \
#   --use_fid_stats \
#   --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
#   --eval_real_dataset /work/nvme/belh/aqian1/imagenet-1k/val \
#   --num_sampling_steps 5 \
#   --eval_bsz 512 \
#   --num_images 7 \
#   --class_num 7
