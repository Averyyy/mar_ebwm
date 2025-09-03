#!/bin/bash
#SBATCH --job-name=EDM-preview-mcmc
#SBATCH --array=0-9
#SBATCH --output=logs/slurm/EDM-preview-mcmc/%A/EDM-preview-%a.out
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1

# --- Environment Setup ---
# Set these variables for your system:
export REPO_ROOT="/work/hdd/bdta/aqian1/mar_ebwm"  # Change this to your repo path
export DATA_ROOT="/work/hdd/bdta/aqian1/data"     # Change this to your data path
export CACHE_ROOT="/work/nvme/bdta/aqian1/data"   # Change this to your cache path

# --- Grid Search Parameters (MCMC step sizes to test) ---
mcmc_steps=(1e-7 1e-6 1e-5 1e-3 1e-2 1e-1 1.0 100 1000 10000)

# --- Calculate parameters for this job ---
len_ms=${#mcmc_steps[@]}

  task_id=${SLURM_ARRAY_TASK_ID}

# If request too many tasks, exit redundant tasks
if [ -z "${task_id}" ]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID is not set. Are you running this as an array job?" >&2
  exit 1
fi
if [ "${task_id}" -ge "${len_ms}" ]; then
  echo "[INFO] SLURM_ARRAY_TASK_ID ${task_id} >= total mcmc steps ${len_ms}. Exiting redundant array task." >&2
  exit 0
fi

mcmc_step_size=${mcmc_steps[$task_id]}

# --- Setup ---
module load cuda/12.6.1
source activate ebm_gh200
cd ${REPO_ROOT}

# --- Parameters ---
MODEL_TYPE=ebm
MODEL=ebm_base
IMG_SIZE=256
DIFFUSION_TIMESTEPS=500

# --- Checkpoint to test (CHANGE THIS PATH) ---
CHECKPOINT_PATH="${REPO_ROOT}/output/EDM-256-base-lr3e-6-timesteps500-bz256-epo320-c1k"
TEST_OUTPUT_DIR="${REPO_ROOT}/output/preview-test-mcmc${mcmc_step_size}"

# --- Log Parameters ---
echo "--- Starting MCMC Step Size Preview Test ${SLURM_ARRAY_TASK_ID} ---"
echo "MCMC Step Size: ${mcmc_step_size}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Test Output: ${TEST_OUTPUT_DIR}"
echo "Model: ${MODEL}"
echo "Image Size: ${IMG_SIZE}"
echo "--------------------"

# --- Preview Command (No Training) ---
torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=$((7638 + SLURM_ARRAY_TASK_ID)) \
  main_ebm.py \
  --run_name preview-mcmc-${mcmc_step_size} \
  --img_size ${IMG_SIZE} \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ${MODEL_TYPE} \
  --model ${MODEL} \
  --use_energy \
  --use_innerloop_opt \
  --mcmc_step_size ${mcmc_step_size} \
  --energy_grad_multiplier 1 \
  --diffusion_timesteps ${DIFFUSION_TIMESTEPS} \
  --batch_size 16 \
  --num_workers 8 \
  --syn_dataloader \
  --resume ${CHECKPOINT_PATH} \
  --output_dir ${TEST_OUTPUT_DIR} \
  --preview_only \
  --preview_labels 0,1,2,3,430,485,605,726,850

echo "--- MCMC Step Size Preview Test ${SLURM_ARRAY_TASK_ID} completed ---"


# torchrun \
#   --nproc_per_node=1 \
#   --master_addr=localhost \
#   --master_port=7638 \
#   main_ebm.py \
#   --img_size 256 \
#   --vae_path pretrained_models/vae/kl16.ckpt \
#   --model_type ebm \
#   --model ebm_base \
#   --use_energy \
#   --use_innerloop_opt \
#   --mcmc_step_size 1e-10 \
#   --energy_grad_multiplier 1 \
#   --diffusion_timesteps 500 \
#   --batch_size 16 \
#   --num_workers 16 \
#   --syn_dataloader \
#   --resume /work/hdd/bdta/aqian1/mar_ebwm/output/EDM-256-base-lr3e-6-timesteps500-bz256-epo320-c1k \
#   --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/preview-test-mcmc \
#   --preview_only \
#   --preview_labels 0,1,2,3,430,485,605,726,850