### RESOURCE CONFIG ###

#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

### LOG CONFIG ###

#SBATCH --job-name=SDM-base-effbs_@BZ-lr_@LR
#SBATCH --output=logs/slurm/img_256/SDM-base-effbs_@BZ-lr_@LR%A-%a.log
RUN_NAME="SDM-base-effbs_@BZ-lr_@LR"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
# MODEL_TYPE="${RUN_NAME%%-*}" # unused for now
MODEL_SIZE="${RUN_NAME#*-}"; MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/img_256/
module purge
# these scripts are formatted similarly to the EBT codebase https://github.com/alexiglad/ebwm/tree/alexi_inference

# --- Set the MASTER_PORT, MASTER_ADDR, NUM_GPUS, and NUM_NODES ---
NUM_GPUS=${SLURM_GPUS_ON_NODE:-$([ -n "${CUDA_VISIBLE_DEVICES:-}" ] && awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES//[[:space:]]/}" || (nvidia-smi -L 2>/dev/null | wc -l || echo 1))}
NUM_GPUS=${NUM_GPUS:-1}; [ "${NUM_GPUS}" -gt 0 ] || NUM_GPUS=1
echo "NUM_GPUS: ${NUM_GPUS}" # NOTE this defaults to all available Nvidia GPUs, adjust as needed

NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
echo "NUM_NODES: ${NUM_NODES}"

source job_scripts/job_utils.sh # sets the MASTER_PORT, MASTER_ADDR

# --- Slurm Arrays for Possible Grid Search ---

learning_rates=(0.0001 0.0003 0.00003)


# --- Set Key Hyperparameters ---
LR=${learning_rates[$SLURM_ARRAY_TASK_ID]}
BATCH_SIZE_PER_DEVICE=128
GRAD_ACCU=2
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE_PER_DEVICE * GRAD_ACCU * NUM_GPUS * NUM_NODES))
RUN_NAME="${RUN_NAME//@BZ/${EFFECTIVE_BATCH_SIZE}}"; RUN_NAME="${RUN_NAME//@LR/${LR}}"
echo "RUN_NAME: ${RUN_NAME}"

# --- Launch Run With Specified Hparams ---

${SLURM_ARRAY_TASK_ID:+srun} torchrun --nproc_per_node=${NUM_GPUS} --nnodes=${NUM_NODES} --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT main_ebm.py \
--run_name ${RUN_NAME} \
--model_type "ebm" \
--model_size ${MODEL_SIZE} \
\
--diffusion_timesteps 1000 \
\
--epochs 1000 \
--warmup_epochs 10 \
--batch_size ${BATCH_SIZE_PER_DEVICE} \
--lr ${LR} \
--grad_accu ${GRAD_ACCU} \
--weight_decay 0.02 \
\
--img_size 256 \
--vae_path "pretrained_models/vae/kl16.ckpt" \
--use_cached \
--cached_path ${IMAGENET1K_CACHE} \
--cached_format ptshard \
--num_workers 8 \
\
--output_dir "./logs/output/${RUN_NAME}" \
--save_last_freq 20 \
--wandb_entity "ebwm_nlp" \
--wandb_project "energy_diffusion_final" \
\
--preview \
--preview_interval 20 \
--preview_labels 0,1,2,3,430,485,605,726,850 \
\
--val \
--val_batch_size ${BATCH_SIZE_PER_DEVICE} \
--val_freq 20 \
--val_data_path ${IMAGENET1K_ROOT}/val


echo "--- Job Completed ---"

# ===== if you want to resume a run with the same RUN_NAME, uncomment the following line and paste it back to the commands above =====

# --resume "./logs/output/${RUN_NAME}" \

# ===== if you want to add online evaluation, uncomment the following lines and paste it back to the commands above =====

# --online_eval \
# --eval_freq 20 \
# --use_fid_stats \
# --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
# --eval_real_dataset ${IMAGENET1K_ROOT}/val \
# --num_sampling_steps 250 \
# --eval_bsz ${BATCH_SIZE_PER_DEVICE // 4} \
# --num_images 1000 \