#!/bin/bash
#SBATCH --job-name=SDM-base-64
#SBATCH --array=0-0
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm/SDM-base-64/%A/SDM-base-64-%a.out
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1


# --- Grid Search Parameters ---

# --- Setup ---
module load cuda/12.6.1
source activate mar_gh200
cd /work/hdd/bdta/aqian1/mar_ebwm

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
OUTPUT_DIR="/work/hdd/bdta/aqian1/mar_ebwm/output/${RUN_NAME}"

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
  --run_name ${RUN_NAME} \
  --img_size ${IMG_SIZE} \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ${MODEL_TYPE} \
  --model ${MODEL} \
  --epochs ${EPOCHES} \
  --warmup_epochs ${WARMUP_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --num_workers 32 \
  --blr ${BLR} \
  --use_cached \
  --cached_path /work/nvme/bdta/aqian1/data/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --output_dir ${OUTPUT_DIR} \
  --resume ${OUTPUT_DIR} \
  --preview \
  --preview_interval 25 \
  --preview_labels 0,1,2,3,430,485,605,726,850 \
  --online_eval \
  --eval_freq 50 \
  --use_fid_stats \
  --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
  --eval_real_dataset /work/hdd/bdta/aqian1/data/imagenet-1k-64/val \
  --num_sampling_steps ${NUM_EVAL_STEPS} \
  --eval_bsz 256 \
  --num_images ${NUM_EVAL_IMAGES} \
  --val \
  --val_batch_size ${BATCH_SIZE} \
  --val_freq 25 \
  --val_data_path /work/hdd/bdta/aqian1/data/imagenet-1k-64/val


echo "--- Standard Diffusion job ${SLURM_ARRAY_TASK_ID} completed ---"

