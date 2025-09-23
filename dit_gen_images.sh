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
HYPERPARAMS_FILE=$1
RESUME_PATH=$2
CFG_VALUE=${3:-0.0}  # optional
TIMESTEP_VALUE=${4:0}

echo "HYPERPARAMS_FILE: $HYPERPARAMS_FILE"
echo "RESUME_PATH: $RESUME_PATH"
echo "CFG_VALUE: $CFG_VALUE"

CHECKPOINT_NAME=$(basename "$RESUME_PATH" .pth)
source "$HYPERPARAMS_FILE" "$RESUME_PATH" "$CFG_VALUE"

RUN_NAME="${RUN_NAME}_${CHECKPOINT_NAME}_${CFG_VALUE}"
OUTPUT_DIR="./samples/${RUN_NAME}"

ARGS+=" --output_dir ${OUTPUT_DIR}"
ARGS+=" --run_name ${RUN_NAME}"

# Append evaluation-related args
ARGS+=" --resume $RESUME_PATH"

# ------------------ Debug Info ------------------
echo "[INFO] Starting sampling + evaluation with the following settings:"
echo "       Model       : $MODEL_SIZE"
echo "       Resume Path : $RESUME_PATH"
echo "       Num Samples : $NUM_SAMPLES"
echo "       GPUs        : $CUDA_VISIBLE_DEVICES"
echo "       Num GPUs    : $NUM_GPUS"
echo "       Run Name    : $RUN_NAME"
echo "       WANDB       : $WANDB_MODE"
echo "       ARGS        : $ARGS"

# ------------------ Step 1: Sampling ------------------
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS sample_ddp.py $ARGS


echo "Running evaluation"
# ------------------ Step 2: Evaluation ------------------
torchrun --nnodes=1 --nproc_per_node=1 sane_evaluate.py \
    --model_type $MODEL_SIZE \
    --resume $RESUME_PATH \
    --sample-dir $OUTPUT_DIR \
    --imagenet-dir $IMAGENET1K_ROOT/val \
    --wandb-entity $WANDB_ENTITY \
    --wandb-project $WANDB_PROJECT \
    --wandb-run-name $RUN_NAME \
    --cfg ${CFG_VALUE} \
    --seed ${SEED} \
    --keep-pngs \

echo "[INFO] Evaluation finished."
