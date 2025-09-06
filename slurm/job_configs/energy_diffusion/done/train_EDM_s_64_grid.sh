#SBATCH --job-name=energy-diffusion-grid
#SBATCH --array=0-3
#SBATCH --output=${REPO_ROOT}/logs/slurm/energy-diffusion/%A/energy-diffusion-%a.out
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1

# --- Environment Setup ---
# Set these variables for your system:
export REPO_ROOT="/work/hdd/bdta/aqian1/mar_ebwm"  # Change this to your repo path
export DATA_ROOT="/work/hdd/bdta/aqian1/data"     # Change this to your data path
export CACHE_ROOT="/work/nvme/bdta/aqian1/data"   # Change this to your cache path



# --- Grid Search Parameters (keep lists for easy future expansion) ---
contrastive_scales=(0.05 )
mcmc_steps=(0.001 )
learning_rates=(9e-6 )

# --- Calculate parameters for this job (generic for any list length) ---
len_lr=${#learning_rates[@]}
len_cl=${#contrastive_scales[@]}
len_ms=${#mcmc_steps[@]}

total_param_combos=$((len_lr * len_cl * len_ms))
if [ ${total_param_combos} -lt 1 ]; then
  echo "[ERR] No parameter combinations (total_param_combos=${total_param_combos})." >&2
  exit 1
fi

raw_task_id=${SLURM_ARRAY_TASK_ID}

# Map raw_task_id into parameter-combo index (for indexing lists) and mode index (0..3)
param_id=$(( raw_task_id % total_param_combos ))
mode_idx=$(( raw_task_id % 4 ))

lr_idx=$(( param_id / (len_cl * len_ms) ))
cl_idx=$(( (param_id / len_ms) % len_cl ))
mcmc_idx=$(( param_id % len_ms ))

contrastive_loss_scale=${contrastive_scales[$cl_idx]}
step_size=${mcmc_steps[$mcmc_idx]}
learning_rate=${learning_rates[$lr_idx]}
multiplier=$(echo "${step_size} * 3" | bc -l)

# --- Setup ---
module load cuda/12.6.1
source activate mar_gh200
cd ${REPO_ROOT}

# --- Parameters ---
NUM_GPUS=1
GRAD_ACCU=1
BLR=${learning_rate}
BATCH_SIZE=1024
EPOCHES=2000
WARMUP_EPOCHS=100
MODEL_TYPE=ebm
MODEL=ebm_small
NUM_EVAL_IMAGES=1000
NUM_EVAL_STEPS=250
IMG_SIZE=64
CONTRASTIVE_LOSS_SCALE=${contrastive_loss_scale}
MCMC_REFINEMENT_LOSS_SCALE=0.1

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCU * NUM_GPUS))

# --- Mode-specific flags (4-job array) ---
# mode 0: with closs & rloss -> full
# mode 1: without closs & with rloss
# mode 2: with closs & without rloss
# mode 3: without closs & rloss (neither)
CLOSS_ARGS=""
RLOSS_ARGS=""
CFLAG=0
RFLAG=0
MODE_DESC=""
case ${mode_idx} in
  0)
    CLOSS_ARGS="--supervise_energy_landscape --contrasive_loss_scale ${CONTRASTIVE_LOSS_SCALE}"
    RLOSS_ARGS="--learnable_mcmc_step_size --mcmc_refinement_loss_scale ${MCMC_REFINEMENT_LOSS_SCALE}"
    CFLAG=1; RFLAG=1; MODE_DESC="with closs & rloss";;
  1)
    CLOSS_ARGS=""
    RLOSS_ARGS="--learnable_mcmc_step_size --mcmc_refinement_loss_scale ${MCMC_REFINEMENT_LOSS_SCALE}"
    CFLAG=0; RFLAG=1; MODE_DESC="without closs & with rloss";;
  2)
    CLOSS_ARGS="--supervise_energy_landscape --contrasive_loss_scale ${CONTRASTIVE_LOSS_SCALE}"
    RLOSS_ARGS=""
    CFLAG=1; RFLAG=0; MODE_DESC="with closs & without rloss";;
  3)
    CLOSS_ARGS=""
    RLOSS_ARGS=""
    CFLAG=0; RFLAG=0; MODE_DESC="without closs & rloss";;
  *)
    echo "[ERR] Invalid mode_idx=${mode_idx}" >&2; exit 2;;
esac

EXTRA_ARGS="${CLOSS_ARGS} ${RLOSS_ARGS}"

# --- Run Name and Output Dir ---
RUN_NAME="EDM-step_${step_size}-cl${CFLAG}-rl${RFLAG}-closs${CONTRASTIVE_LOSS_SCALE}-rloss${MCMC_REFINEMENT_LOSS_SCALE}-lr_${BLR}-small-64-bz${EFFECTIVE_BATCH_SIZE}-epo${EPOCHES}-c1k"
OUTPUT_DIR="${REPO_ROOT}/output/${RUN_NAME}"

# --- Log Parameters ---
echo "--- Starting Energy Diffusion Grid job ${SLURM_ARRAY_TASK_ID} (${MODE_DESC}) ---"
echo "Grid Parameters:"
echo "  Contrastive Loss Scale: ${CONTRASTIVE_LOSS_SCALE}"
echo "  MCMC Step Size: ${step_size}"
echo "  MCMC Multiplier: ${multiplier}"
echo "  Learning Rate: ${BLR}"
echo "Training Parameters:"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHES}"
echo "  Warmup Epochs: ${WARMUP_EPOCHS}"
echo "  Model: ${MODEL}"
echo "  Image Size: ${IMG_SIZE}"
echo "  Num Evaluation Images: ${NUM_EVAL_IMAGES}"
echo "Run Name: ${RUN_NAME}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Mode: ${MODE_DESC} (CFLAG=${CFLAG}, RFLAG=${RFLAG})"
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
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=${MASTER_PORT} \
  main_ebm.py \
  --run_name ${RUN_NAME} \
  --img_size ${IMG_SIZE} \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ${MODEL_TYPE} \
  --model ${MODEL} \
  --epochs ${EPOCHES} \
  --warmup_epochs ${WARMUP_EPOCHS} \
  --use_energy \
  --use_innerloop_opt \
  ${EXTRA_ARGS} \
  --wandb_log_mse_only \
  --mcmc_step_size ${step_size} \
  --mcmc_step_size_lr_multiplier ${multiplier} \
  --batch_size ${BATCH_SIZE} \
  --num_workers 8 \
  --blr ${BLR} \
  --use_cached \
  --cached_path ${CACHE_ROOT}/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --output_dir ${OUTPUT_DIR} \
  --preview \
  --preview_interval 10 \
  --preview_labels 0,1,2,3,4,5,6,7,8,9,10,11,113,130,282,283,284,309,430,485,605,726,850 \
  --online_eval \
  --eval_freq 50 \
  --use_fid_stats \
  --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
  --eval_real_dataset ${DATA_ROOT}/imagenet-1k-64/val \
  --num_sampling_steps ${NUM_EVAL_STEPS} \
  --eval_bsz 256 \
  --num_images ${NUM_EVAL_IMAGES} \
  --val \
  --val_batch_size ${BATCH_SIZE} \
  --val_freq 50 \
  --val_data_path ${DATA_ROOT}/imagenet-1k-64/val



echo "--- Energy Diffusion Grid job ${SLURM_ARRAY_TASK_ID} (${MODE_DESC}) completed ---"


