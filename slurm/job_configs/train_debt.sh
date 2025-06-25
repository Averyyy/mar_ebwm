#SBATCH --job-name=debt-base-grid
#SBATCH --array=0-0
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm_outputs/%A/debt-grid-%a.out
#SBATCH --time=24:00:00


# --- Grid Search Parameters ---
Define the grid of hyperparameters to search over for DEBT
lrs=(1e-4 1e-5)
alphas=(0.5 1 5 10)

# --- Calculate parameters for this job ---
num_lrs=${#lrs[@]}
num_alphas=${#alphas[@]}
total_combinations=$((num_lrs * num_alphas))

# Calculate indices for this job
lr_idx=$((SLURM_ARRAY_TASK_ID / num_alphas))
alpha_idx=$((SLURM_ARRAY_TASK_ID % num_alphas))

lr=${lrs[$lr_idx]}
alpha=${alphas[$alpha_idx]}
multiplier=$(echo "${alpha} * 3" | bc)

# --- Setup ---
module load cuda/12.6.1
source activate mar_gh200
cd /work/hdd/bdta/aqian1/mar_ebwm

# --- Run Name and Output Dir ---
RUN_NAME="debt-b-lr_${lr}-alpha_${alpha}"
OUTPUT_DIR="/work/hdd/bdta/aqian1/mar_ebwm/output/${RUN_NAME}"

# --- Log Parameters ---
echo "--- Starting DEBT Energy Training job ${SLURM_ARRAY_TASK_ID} ---"
echo "Learning Rate (lr): ${lr}"
echo "Alpha (mcmc_step_size): ${alpha}"
echo "Multiplier (mcmc_step_size_lr_multiplier): ${multiplier}"
echo "Run Name: ${RUN_NAME}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "--------------------"

# --- Training Command ---
torchrun \
    --nproc_per_node=4 \
    main_mar.py \
    --run_name ${RUN_NAME} \
    --img_size 64 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --model_type debt \
    --embed_dim 768 \
    --depth 24 \
    --num_heads 12 \
    --mlp_ratio 4.0 \
    --mcmc_num_steps 2 \
    --mcmc_step_size ${alpha} \
    --mcmc_step_size_lr_multiplier ${multiplier} \
    --epochs 200 \
    --warmup_epochs 20 \
    --batch_size 512 \
    --grad_accu 1 \
    --blr ${lr} \
    --output_dir ${OUTPUT_DIR} \
    --use_cached \
    --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64 \
    --preview \
    --seed 42

echo "--- DEBT Energy Training job ${SLURM_ARRAY_TASK_ID} completed ---" 


# torchrun \
#     --nproc_per_node=4 \
#     main_mar.py \
#     --run_name debt-test \
#     --img_size 64 \
#     --vae_path pretrained_models/vae/kl16.ckpt \
#     --model_type debt \
#     --embed_dim 768 \
#     --depth 24 \
#     --num_heads 12 \
#     --mlp_ratio 4.0 \
#     --mcmc_num_steps 2 \
#     --mcmc_step_size 3 \
#     --mcmc_step_size_lr_multiplier 9 \
#     --epochs 200 \
#     --warmup_epochs 20 \
#     --batch_size 512 \
#     --grad_accu 1 \
#     --blr 5e-5 \
#     --lr_schedule cosine \
#     --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/debt-test \
#     --use_cached \
#     --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64 \
#     --preview \
#     --seed 42