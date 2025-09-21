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

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# ------------------ Input Arguments ------------------
MODE=$1             # "repa" or "fid"
HYPERPARAMS_FILE=$2
RESUME_PATH=$3
CFG_VALUE=${4:-0.0}  # optional
TIMESTEP_VALUE=${5:0}

CHECKPOINT_NAME=$(basename "$RESUME_PATH" .pth)
source "$HYPERPARAMS_FILE" "$RESUME_PATH" "$CFG_VALUE"

ARGS+=" --evaluate"

RUN_NAME="${RUN_NAME}_${CHECKPOINT_NAME}"
OUTPUT_DIR="./logs/output/${RUN_NAME}"

ARGS+=" --output_dir ${OUTPUT_DIR}"
ARGS+=" --run_name ${RUN_NAME}"

# ------------------ Mode-specific handling ------------------
if [ "$MODE" == "repa" ]; then
    CACHE_DIR=/projects/bdjz/sshekhar/cache/${RUN_NAME}_${TIMESTEP_VALUE}
    ARGS+=" --cached_path $CACHE_DIR"
    ARGS+=" --timestep_to_eval $TIMESTEP_VALUE"

    # Conditional cache creation
    if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A $CACHE_DIR)" ]; then
        echo "[INFO] Cache directory not found or empty. Creating cache with $NUM_GPUS GPUs..."
        ARGS_CACHE="${ARGS} --cache_latents"
        torchrun --nproc_per_node=$NUM_GPUS repa_eval.py ${ARGS_CACHE}
    else
        echo "[INFO] Cache directory exists. Skipping cache creation."
    fi

    echo "[INFO] Running REPA evaluation on 1 GPU using cached features..."
    torchrun --nproc_per_node=1 repa_eval.py ${ARGS}

elif [ "$MODE" == "fid" ]; then
    echo "[INFO] Running FID evaluation with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS main_ebm.py ${ARGS}

else
    echo "[ERROR] Unknown mode: $MODE. Must be 'repa' or 'fid'."
    exit 1
fi

# ------------------ Debug Info ------------------
echo "[INFO] Evaluation finished."
