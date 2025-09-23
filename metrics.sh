#!/bin/bash
export WANDB_MODE=${WANDB_MODE:-offline}
export OMP_NUM_THREADS=8

# ------------------ GPU Setup ------------------
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
    export CUDA_VISIBLE_DEVICES=$ALL_GPUS
    echo "[INFO] CUDA_VISIBLE_DEVICES not set, using all GPUs: $CUDA_VISIBLE_DEVICES"
else
    echo "[INFO] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
    export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)}
    export MASTER_PORT=${MASTER_PORT:-29500}
    export NODE_RANK=${NODE_RANK:-$SLURM_NODEID}
    export WORLD_SIZE=${WORLD_SIZE:-$(( SLURM_NNODES * gpus_per_node ))}
fi


NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# ------------------ Input Arguments ------------------
PYTHON_FILE=sane_evaluate.py
HYPERPARAMS_FILE=$1
CHECKPOINT_NUMBER=$2
CHECKPOINT_NAME="checkpoint-${CHECKPOINT_NUMBER}"
CFG_VALUE=${3:-0.0} 

source "$HYPERPARAMS_FILE" "$CHECKPOINT_NAME" "$CFG_VALUE"

RUN_NAME="${RUN_NAME}_${CHECKPOINT_NAME}_${CFG_VALUE}"
OUTPUT_DIR="./samples/${RUN_NAME}"

ARGS+=" --run_name ${RUN_NAME}"

# Append evaluation-related args
ARGS+=" --resume $RESUME_PATH"


ACCELERATE_CMD="accelerate launch \
    --multi_gpu \
    --gpu_ids $CUDA_VISIBLE_DEVICES \
    --num_processes $NUM_GPUS \
    --num_machines $SLURM_NNODES \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --mixed_precision "no" \
    --dynamo_backend no"

    
NEW_ARGS=" --run_name $RUN_NAME"
NEW_ARGS+=" --project_name $WANDB_PROJECT"
NEW_ARGS+=" --wandb_entity $WANDB_ENTITY"
NEW_ARGS+=" --gen_dataset $OUTPUT_DIR"
NEW_ARGS+=" --val_dataset $IMAGENET1K_ROOT/validation"

# ------------------ Debug Info ------------------
echo "[INFO] Starting sampling + evaluation with the following settings:"
echo "       Model       : $MODEL_SIZE"
echo "       Resume Path : $RESUME_PATH"
echo "       Num Samples : $NUM_SAMPLES"
echo "       GPUs        : $CUDA_VISIBLE_DEVICES"
echo "       Num GPUs    : $NUM_GPUS"
echo "       Run Name    : $RUN_NAME"
echo "       WANDB       : $WANDB_MODE"
echo "       ARGS        : $NEW_ARGS"
echo "  ACCELERATE_CMD   : $ACCELERATE_CMD"
echo "   Python file     : $PYTHON_FILE"

$ACCELERATE_CMD $PYTHON_FILE $NEW_ARGS


