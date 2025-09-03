#SBATCH --job-name=energy-diffusion
#SBATCH --array=0-0
#SBATCH --output=logs/slurm/energy-diffusion/fewer-diffusion-steps-%A/energy-diffusion-%a.out
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1

# --- Environment Setup ---
# Set these variables for your system:
export REPO_ROOT="/work/hdd/bdta/aqian1/mar_ebwm"  # Change this to your repo path
export DATA_ROOT="/work/hdd/bdta/aqian1/data"     # Change this to your data path  
export CACHE_ROOT="/work/nvme/bdta/aqian1/data"   # Change this to your cache path

# --- Grid Search Parameters ---
contrastive_scales=(0.05 )
mcmc_steps=(0.001 )
learning_rates=(9e-6 )
# --- Calculate parameters for this job (generic for any list length) ---
len_lr=${#learning_rates[@]}
len_cl=${#contrastive_scales[@]}
len_ms=${#mcmc_steps[@]}

total=$((len_lr * len_cl * len_ms))

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

lr_idx=$(( task_id / (len_cl * len_ms) ))
cl_idx=$(( (task_id / len_ms) % len_cl ))
mcmc_idx=$(( task_id % len_ms ))

contrastive_loss_scale=${contrastive_scales[$cl_idx]}
step_size=${mcmc_steps[$mcmc_idx]}
learning_rate=${learning_rates[$lr_idx]}
multiplier=$(echo "${step_size} * 3" | bc -l)

# --- Setup ---
module load cuda/12.6.1
source activate ebm_gh200
cd ${REPO_ROOT}

# --- Parameters ---
NUM_GPUS=1
GRAD_ACCU=1
BLR=${learning_rate}
BATCH_SIZE=1024
EPOCHES=2000
WARMUP_EPOCHS=100
MODEL_TYPE=ebm
MODEL=ebm_base
NUM_EVAL_IMAGES=1000
NUM_EVAL_STEPS=250
IMG_SIZE=64
CONTRASTIVE_LOSS_SCALE=${contrastive_loss_scale}
MCMC_REFINEMENT_LOSS_SCALE=0.5
ENERGY_GRAD_MULTIPLIER=1

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCU * NUM_GPUS))


# --- Run Name and Output Dir ---
RUN_NAME="EDM-small-step_${step_size}-diffusion_step-500-epo${EPOCHES}-c1k"
OUTPUT_DIR="${REPO_ROOT}/output/${RUN_NAME}"

# --- Log Parameters ---
echo "--- Starting Energy Diffusion Grid Search job ${SLURM_ARRAY_TASK_ID} ---"
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
echo "--------------------"

# --- Training Command for Energy Diffusion ---
torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=$((5748 + SLURM_ARRAY_TASK_ID)) \
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
  --mcmc_step_size ${step_size} \
  --diffusion_timesteps 500 \
  --energy_grad_multiplier ${ENERGY_GRAD_MULTIPLIER} \
  --batch_size ${BATCH_SIZE} \
  --num_workers 16 \
  --blr ${BLR} \
  --use_cached \
  --cached_path ${CACHE_ROOT}/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --output_dir ${OUTPUT_DIR} \
  --preview \
  --preview_interval 50 \
  --preview_labels 0,1,2,3,430,485,605,726,850 \
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
  --val_freq 20 \
  --val_data_path ${DATA_ROOT}/imagenet-1k-64/val


echo "--- Energy Diffusion Grid Search job ${SLURM_ARRAY_TASK_ID} completed ---"

  # --supervise_energy_landscape \
  # --contrasive_loss_scale ${CONTRASTIVE_LOSS_SCALE} \
  #   --learnable_mcmc_step_size \
  # --mcmc_refinement_loss_scale ${MCMC_REFINEMENT_LOSS_SCALE} \
  # --wandb_log_mse_only \

# torchrun \
#   --nproc_per_node=4 \
#   --master_addr=localhost \
#   --master_port=15149 \
#   main_ebm.py \
#   --run_name test-EDM \
#   --img_size 64 \
#   --vae_path pretrained_models/vae/kl16.ckpt \
#   --model_type ebm \
#   --model ebm_small \
#   --use_energy \
#   --use_innerloop_opt \
#   --supervise_energy_landscape \
#   --wandb_log_mse_only \
#   --mcmc_step_size 0.1 \
#   --learnable_mcmc_step_size \
#   --log_energy_accept_rate \
#   --epochs 500 \
#   --warmup_epochs 5 \
#   --batch_size 2048 \
#   --num_workers 8 \
#   --blr 9e-6 \
#   --use_cached \
#   --cached_path ${CACHE_ROOT}/cached-imagenet1k-64-ptshard-32 \
#   --cached_format ptshard \
#   --output_dir ${REPO_ROOT}/output/test-edm \
#   --preview \
#   --preview_interval 1 \
#   --preview_labels 0,1,2,3,4,5,6,7,8,9,10,11,113,130,282,283,284,309,430,485,605,726,850 \
#   --online_eval \
#   --eval_freq 50 \
#   --eval_real_dataset ${DATA_ROOT}/imagenet-1k-64/val \
#   --use_fid_stats \
#   --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
#   --num_sampling_steps 250 \
#   --eval_bsz 256 \
#   --num_images 1000 \
#   --val \
#   --val_batch_size 2048 \
#   --val_data_path ${DATA_ROOT}/imagenet-1k-64/val \
#   --val_freq 5 


    # --run_name test-EDM-small-64-step_0.1-bz2048-lr_9e-6-epo500-c1k \
