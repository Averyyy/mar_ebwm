#SBATCH --job-name=EDM-base-dtype-grid
#SBATCH --array=0-0
#SBATCH --output=${REPO_ROOT}/logs/slurm/EDM-base-dtype-grid/%A/EDM-base-dtype-%a.out
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1

# --- Environment Setup ---
# Set these variables for your system:
export REPO_ROOT="/work/hdd/bdta/aqian1/mar_ebwm"  # Change this to your repo path
export DATA_ROOT="/work/hdd/bdta/aqian1/data"     # Change this to your data path
export CACHE_ROOT="/work/nvme/bdta/aqian1/data"   # Change this to your cache path



# --- Grid Search Parameters (training dtypes and their auxiliary evaluations) ---
# train_dtypes=(fp32 bf16)
# For fp32 training: auxiliary eval with bf16,fp16
# For bf16 training: auxiliary eval with fp32,fp16
# auxiliary_evals=("bf16,fp16" "fp32,fp16")

train_dtypes=(fp16 )
auxiliary_evals=("fp32,bf16" )


# --- Calculate parameters for this job ---
total=${#train_dtypes[@]}

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

train_dtype=${train_dtypes[$task_id]}
auxiliary_eval_dtypes=${auxiliary_evals[$task_id]}

# --- Setup ---
module load cuda/12.6.1
source activate ebm_gh200
cd ${REPO_ROOT}

# --- Parameters (same as original) ---
NUM_GPUS=1
GRAD_ACCU=1
BLR=9e-6
BATCH_SIZE=1024
EPOCHES=1000
WARMUP_EPOCHS=50
MODEL_TYPE=ebm
MODEL=ebm_base
NUM_EVAL_IMAGES=1000
NUM_EVAL_STEPS=250
IMG_SIZE=64
ENERGY_GRAD_MULTIPLIER=1
DIFFUSION_TIMESTEPS=500
MCMC_STEP_SIZE=0.001

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCU * NUM_GPUS))

# --- Base Run Name and Output Dir ---
BASE_RUN_NAME="EDM-base-step_${MCMC_STEP_SIZE}-lr${BLR}-timesteps${DIFFUSION_TIMESTEPS}-bz${EFFECTIVE_BATCH_SIZE}-epo${EPOCHES}-c1k-train_${train_dtype}"
BASE_OUTPUT_DIR="${REPO_ROOT}/output/${BASE_RUN_NAME}"

# --- Log Parameters ---
echo "--- Starting Energy Diffusion Training with Dtype Grid Evaluation job ${SLURM_ARRAY_TASK_ID} ---"
echo "Training Dtype: ${train_dtype}"
echo "Auxiliary Eval Dtypes: ${auxiliary_eval_dtypes}"
echo "Grid Parameters:"
echo "  MCMC Step Size: ${MCMC_STEP_SIZE}"
echo "  Learning Rate: ${BLR}"
echo "  Diffusion Time Steps: ${DIFFUSION_TIMESTEPS}"
echo "Training Parameters:"
echo "  Batch Size: ${EFFECTIVE_BATCH_SIZE}"
echo "  Epochs: ${EPOCHES}"
echo "  Warmup Epochs: ${WARMUP_EPOCHS}"
echo "  Model: ${MODEL}"
echo "  Image Size: ${IMG_SIZE}"
echo "  Num Evaluation Images: ${NUM_EVAL_IMAGES}"
echo "Base Run Name: ${BASE_RUN_NAME}"
echo "Base Output Dir: ${BASE_OUTPUT_DIR}"
echo "--------------------"

# --- Training Phase ---
# --- Training Phase with Auxiliary Evaluations ---
echo "=== TRAINING WITH AUXILIARY EVALUATIONS ==="
torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=localhost \
  --master_port=$((6758 + SLURM_ARRAY_TASK_ID)) \
  main_ebm.py \
  --run_name ${BASE_RUN_NAME} \
  --img_size ${IMG_SIZE} \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ${MODEL_TYPE} \
  --model ${MODEL} \
  --epochs ${EPOCHES} \
  --warmup_epochs ${WARMUP_EPOCHS} \
  --use_energy \
  --use_innerloop_opt \
  --mcmc_step_size ${MCMC_STEP_SIZE} \
  --energy_grad_multiplier ${ENERGY_GRAD_MULTIPLIER} \
  --diffusion_timesteps ${DIFFUSION_TIMESTEPS} \
  --train_dtype ${train_dtype} \
  --eval_dtype ${train_dtype} \
  --auxiliary_eval_dtypes ${auxiliary_eval_dtypes} \
  --batch_size ${BATCH_SIZE} \
  --num_workers 32 \
  --blr ${BLR} \
  --use_cached \
  --cached_path ${CACHE_ROOT}/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --output_dir ${BASE_OUTPUT_DIR} \
  --preview \
  --preview_interval 25 \
  --preview_labels 0,1,2,3,430,485,605,726,850 \
  --online_eval \
  --eval_freq 50 \
  --use_fid_stats \
  --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
  --eval_real_dataset ${DATA_ROOT}/imagenet-1k-64/val \
  --num_sampling_steps ${NUM_EVAL_STEPS} \
  --eval_bsz 256 \
  --num_images ${NUM_EVAL_IMAGES} \
  --disable_progress_bar \
  --val \
  --val_batch_size ${BATCH_SIZE} \
  --val_freq 25 \
  --val_data_path ${DATA_ROOT}/imagenet-1k-64/val

echo "--- Energy Diffusion Dtype Grid job ${SLURM_ARRAY_TASK_ID} completed ---"