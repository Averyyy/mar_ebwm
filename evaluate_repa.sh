#!/bin/bash
export WANDB_MODE=${WANDB_MODE:-offline}
export OMP_NUM_THREADS=8

# ------------------ GPU Setup ------------------
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  # Get all GPU indices from nvidia-smi
  ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
  export CUDA_VISIBLE_DEVICES=$ALL_GPUS
  echo "CUDA_VISIBLE_DEVICES not set, using all GPUs: $CUDA_VISIBLE_DEVICES"
else
  echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Count number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

HYPERPARAMS_FILE=$1
RESUME_PATH=$2

CHECKPOINT_NAME=$(basename "$RESUME_PATH" .pth)
# Source hyperparameters (pass $1=resume_path, $2=use_energy_flag)
source "$HYPERPARAMS_FILE" "$RESUME_PATH" "$CFG_VALUE"
ARGS+=" --evaluate"
RUN_NAME="${RUN_NAME}_${CHECKPOINT_NAME}"
ARGS+=" --run_name ${RUN_NAME}"

# ------------------ Debug Info ------------------
echo "[INFO] Running evaluation with the following settings:"
echo "       GPUs       : $CUDA_VISIBLE_DEVICES"
echo "       Num GPUs   : $NUM_GPUS"
echo "       ARGS       : $ARGS"
echo "       WANDB       : $WANDB_MODE"

python repa_eval.py ${ARGS}


