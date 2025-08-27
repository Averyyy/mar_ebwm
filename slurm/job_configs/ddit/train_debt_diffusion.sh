#SBATCH --job-name=debt-diffusion-base-grid
#SBATCH --array=0-0
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm_outputs/%A/debt-diffusion-grid-%a.out
#SBATCH --time=24:00:00


# --- Grid Search Parameters ---
# lrs=(6e-6 )
# alphas=(10 )

# # --- Calculate parameters for this job ---
# num_lrs=${#lrs[@]}
# num_alphas=${#alphas[@]}
# total_combinations=$((num_lrs * num_alphas))

# # Calculate indices for this job
# lr_idx=$((SLURM_ARRAY_TASK_ID / num_alphas))
# alpha_idx=$((SLURM_ARRAY_TASK_ID % num_alphas))

# lr=${lrs[$lr_idx]}
# alpha=${alphas[$alpha_idx]}
# multiplier=$(echo "${alpha} * 3" | bc)

# # --- Setup ---
# module load cuda/12.6.1
# source activate mar_gh200
# cd /work/hdd/bdta/aqian1/mar_ebwm

# # --- Run Name and Output Dir ---
# RUN_NAME="debt-diffusion-b-lr_${lr}-alpha_${alpha}"
# OUTPUT_DIR="/work/hdd/bdta/aqian1/mar_ebwm/output/${RUN_NAME}"

# # --- Log Parameters ---
# echo "--- Starting DEBT Diffusion (IRED) Training job ${SLURM_ARRAY_TASK_ID} ---"
# echo "Learning Rate (lr): ${lr}"
# echo "Alpha (mcmc_step_size): ${alpha}"
# echo "Multiplier (mcmc_step_size_lr_multiplier): ${multiplier}"
# echo "Run Name: ${RUN_NAME}"
# echo "Output Dir: ${OUTPUT_DIR}"
# echo "--------------------"

# # --- Training Command ---
# torchrun \
#     --nproc_per_node=4 \
#     main_mar.py \
#     --run_name ${RUN_NAME} \
#     --img_size 64 \
#     --vae_path pretrained_models/vae/kl16.ckpt \
#     --model_type debt_diffusion \
#     --model debt_diffusion_base \
#     --epochs 200 \
#     --warmup_epochs 20 \
#     --batch_size 256 \
#     --grad_accu 2 \
#     --blr ${lr} \
#     --output_dir ${OUTPUT_DIR} \
#     --use_cached \
#     --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64 \
#     --preview \
#     --seed 42



torchrun \
    --nproc_per_node=1 \
    --master_port 4938 \
    main_mar.py \
    --run_name debt-diffusion-test \
    --img_size 64 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --model_type debt_diffusion \
    --model debt_diffusion_2xs \
    --epochs 500 \
    --warmup_epochs 50 \
    --batch_size 2048 \
    --grad_accu 1 \
    --blr 1e-5 \
    --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/debt-diffusion-test \
    --use_cached \
    --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-c10 \
    --preview \
    --preview_interval 10 \
    --seed 42