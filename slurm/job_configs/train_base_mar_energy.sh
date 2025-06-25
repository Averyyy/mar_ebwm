#SBATCH --job-name=mar-energy-grid
#SBATCH --array=0-0
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm_outputs/%A/mar-energy-grid-%a.out
#SBATCH --time=24:00:00


# --- Grid Search Parameters ---
# Define the grid of hyperparameters to search over.
# lr only 1e-6, alpha only 5 without changing the rest
lrs=(1e-6 )
alphas=(5 )

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
RUN_NAME="emar-base-lr_${lr}-alpha_${alpha}-mult_${multiplier}"
OUTPUT_DIR="/work/hdd/bdta/aqian1/mar_ebwm/output/${RUN_NAME}"


# --- Log Parameters ---
echo "--- Starting job ${SLURM_ARRAY_TASK_ID} ---"
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
    --model mar_base \
    --model_type mar \
    --use_energy_loss \
    --epochs 200 \
    --warmup_epochs 20 \
    --batch_size 512 \
    --grad_accu 1 \
    --grad_clip 1.0 \
    --blr ${lr} \
    --output_dir ${OUTPUT_DIR} \
    --use_cached \
    --cached_path /work/hdd/bdta/aqian1/mar_ebwm/data/cached-imagenet1k-64 \
    --use_energy_loss \
    --mcmc_step_size_lr_multiplier ${multiplier} \
    --mcmc_step_size ${alpha} \
    --preview \
    --seed 42
    # --resume ${OUTPUT_DIR} \



# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py \
# --run_name mar-base-energy \
# --img_size 256 \
# --vae_path pretrained_models/vae/kl16.ckpt \
# --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model mar_base --model_type mar \
# --epochs 50 --warmup_epochs 5 \
# --batch_size 12 --blr 5e-5 \
# --output_dir ./output/mar-base-energy \
# --use_cached --cached_path ./data/cached-tiny-imagenet \
# --seed 42 \

# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=4738 main_mar.py \
# --run_name mar-base-energy-tiny \
# --img_size 256 \
# --vae_path pretrained_models/vae/kl16.ckpt \
# --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
# --model mar_base --model_type mar \
# --batch_size 24 --blr 8e-4 \
#  --diffusion_batch_mul 4 \
# --output_dir ./output/mar-base-energy-tiny \
# --resume ./output/mar-base-energy-tiny \
# --use_cached --cached_path ./data/cached-tiny-imagenet \
# --evaluate

# small test run
# torchrun \
#     --nproc_per_node=1 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_port=4733 \
#     main_mar.py \
#     --run_name mar-base-energy-1k-64-test \
#     --img_size 64 \
#     --vae_path pretrained_models/vae/kl16.ckpt \
#     --vae_embed_dim 16 \
#     --vae_stride 16 \
#     --patch_size 1 \
#     --model mar_base \
#     --model_type mar \
#     --epochs 50 \
#     --warmup_epochs 5 \
#     --batch_size 128 \
#     --grad_accu 4 \
#     --blr 1e-5 \
#     --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/mar-base-energy-tiny-alpha-0.01-mult-50-test \
#     --use_cached \
#     --cached_path /work/hdd/bdta/aqian1/mar_ebwm/data/cached-tiny \
#     --seed 42 \
#     --mcmc_step_size_lr_multiplier 1 \
#     --mcmc_step_size 0.01 \
#     --preview

        # --resume /work/hdd/bdta/aqian1/mar_ebwm/output/mar-base-energy-tiny-alpha-0.01-mult-100-test \


# torchrun \
#     --nproc_per_node=4 \
#     main_mar.py \
#     --run_name emar-base-test \
#     --img_size 64 \
#     --vae_path pretrained_models/vae/kl16.ckpt \
#     --model mar_base \
#     --model_type mar \
#     --epochs 100 \
#     --warmup_epochs 10 \
#     --batch_size 512 \
#     --grad_accu 1 \
#     --grad_clip 1.0 \
#     --blr 1e-6 \
#     --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/mar-base-energy-test \
#     --use_cached \
#     --cached_path /work/hdd/bdta/aqian1/mar_ebwm/data/cached-imagenet1k-64 \
#     --use_energy_loss \
#     --mcmc_step_size_lr_multiplier 1 \
#     --mcmc_step_size 5 \
#     --preview \
#     --seed 42



torchrun \
    --nproc_per_node=4 \
    main_mar.py \
    --run_name mar-base-test \
    --img_size 64 \
    --vae_path pretrained_models/vae/kl16.ckpt \
    --model mar_base \
    --model_type mar \
    --epochs 100 \
    --warmup_epochs 10 \
    --batch_size 512 \
    --grad_accu 1 \
    --grad_clip 1.0 \
    --blr 1e-4 \
    --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/mar-base-energy-test \
    --use_cached \
    --cached_path /work/hdd/bdta/aqian1/data/cached-tiny \
    --preview \
    --seed 42