#SBATCH --job-name=EDM-base-bz256-400k
#SBATCH --array=0-0
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm/EDM-base-bztest/256/%A/EDM-base-%a.out
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=2



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
cd /work/hdd/bdta/aqian1/mar_ebwm

# --- Parameters ---
NUM_GPUS=2
GRAD_ACCU=1
BLR=${learning_rate}
BATCH_SIZE=128
EPOCHES=320
WARMUP_EPOCHS=16
MODEL_TYPE=pure_diffusion
MODEL=pure_diffusion_base
NUM_EVAL_IMAGES=1000
NUM_EVAL_STEPS=250
IMG_SIZE=256
ENERGY_GRAD_MULTIPLIER=1
DIFFUSION_TIMESTEPS=500
EVAL_BATCH_SIZE=$((BATCH_SIZE / 4))

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCU * NUM_GPUS))


# --- Run Name and Output Dir ---
RUN_NAME="EDM-256-base-lr${BLR}-timesteps${DIFFUSION_TIMESTEPS}-bz${EFFECTIVE_BATCH_SIZE}-epo${EPOCHES}-c1k"
OUTPUT_DIR="/work/hdd/bdta/aqian1/mar_ebwm/output/${RUN_NAME}"

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

# --- Training Command for Energy Diffusion ---
torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=localhost \
  --master_port=$((6284 + SLURM_ARRAY_TASK_ID)) \
  main_mar.py \
  --run_name ${RUN_NAME} \
  --img_size ${IMG_SIZE} \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ${MODEL_TYPE} \
  --model ${MODEL} \
  --epochs ${EPOCHES} \
  --warmup_epochs ${WARMUP_EPOCHS} \
  --use_energy \
  --use_innerloop_opt \
  --mcmc_step_size ${step_size} \
  --energy_grad_multiplier ${ENERGY_GRAD_MULTIPLIER} \
  --diffusion_timesteps ${DIFFUSION_TIMESTEPS} \
  --batch_size ${BATCH_SIZE} \
  --num_workers 8 \
  --blr ${BLR} \
  --use_cached \
  --cached_path /work/nvme/bdta/aqian1/data/cached-imagenet1k-256-ptshard-16 \
  --cached_format ptshard \
  --output_dir ${OUTPUT_DIR} \
  --resume ${OUTPUT_DIR} \
  --preview \
  --preview_interval 20 \
  --preview_labels 0,1,2,3,430,485,605,726,850 \
  --val \
  --val_batch_size ${BATCH_SIZE} \
  --val_freq 20 \
  --val_data_path /work/nvme/belh/aqian1/imagenet-1k/val


echo "--- Energy Diffusion Grid Search job ${SLURM_ARRAY_TASK_ID} completed ---"