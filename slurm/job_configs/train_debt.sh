#SBATCH --job-name=debt-base-grid
#SBATCH --array=0-0
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm/debt/%A/debt-grid-%a.out
#SBATCH --time=24:00:00


# --- Grid Search Parameters ---
lrs=(9e-6 )
alphas=(0.1 )

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
RUN_NAME="debt-b-lr_${lr}-alpha_${alpha}-tti-bz4096-epoch80k"
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
    --nproc_per_node=1 \
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
    --epochs 80000 \
    --warmup_epochs 2000 \
    --batch_size 4096 \
    --blr ${lr} \
    --output_dir ${OUTPUT_DIR} \
    --use_cached \
    --cached_format ptshard \
    --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard \
    --preview \
    --preview_interval 1000 \
    --seed 42

echo "--- DEBT Energy Training job ${SLURM_ARRAY_TASK_ID} completed ---" 


# torchrun \
#     --nproc_per_node=1 \
#     main_mar.py \
#     --run_name debt-test-gpu \
#     --img_size 64 \
#     --vae_path pretrained_models/vae/kl16.ckpt \
#     --model_type debt \
#     --model debt_2xs \
#     --mcmc_num_steps 2 \
#     --mcmc_step_size 10 \
#     --mcmc_step_size_lr_multiplier 30 \
#     --epochs 200 \
#     --warmup_epochs 20 \
#     --batch_size 2048 \
#     --grad_accu 1 \
#     --blr 1e-5 \
#     --lr_schedule cosine \
#     --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/debt-test-1 \
#     --use_cached \
#     --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard \
#     --cached_format ptshard \
#     --preview \
#     --seed 42



# torchrun \
#   --nproc_per_node=1 \
#   main_mar.py \
#   --run_name diffusion-small-64 \
#   --img_size 64 \
#   --vae_path pretrained_models/vae/kl16.ckpt \
#   --model_type pure_diffusion \
#   --model pure_diffusion_small \
#   --epochs 1600 \
#   --warmup_epochs 160 \
#   --batch_size 1024 \
#   --blr 1e-5 \
#   --lr_schedule cosine \
#   --use_cached \
#   --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard \
#   --cached_format ptshard \
#   --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/diffusion-small \
#   --seed 42 \
#   --preview \
#   --preview_interval 10


# torchrun \
#   --nproc_per_node=1 \
#   main_mar.py \
#   --run_name diffusion-small-64-6e-5-scratch-continue \
#   --img_size 64 \
#   --vae_path pretrained_models/vae/kl16.ckpt \
#   --model_type pure_diffusion \
#   --model pure_diffusion_small \
#   --epochs 200000 \
#   --warmup_epochs 1000 \
#   --batch_size 8192 \
#   --blr 6e-5 \
#   --lr_schedule cosine \
#   --use_cached \
#   --cached_format ptshard \
#   --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard \
#   --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/diffusion-small-6e-5-scratch \
#   --resume /work/hdd/bdta/aqian1/mar_ebwm/output/diffusion-small-6e-5-scratch \
#   --seed 42 \
#   --preview \
#   --preview_interval 100


# torchrun \
#   --nproc_per_node=1 \
#   --master_addr=localhost \
#   --master_port=4837 \
#   main_mar.py \
#   --img_size 64 \
#   --run_name diffusion-small-64-energy-full-alpha-test \
#   --use_energy \
#   --use_innerloop_opt \
#   --supervise_energy_landscape \
#   --wandb_log_mse_only \
#   --mcmc_step_size 5 \
#   --mcmc_step_size_lr_multiplier 15 \
#   --vae_path pretrained_models/vae/kl16.ckpt \
#   --model_type pure_diffusion \
#   --model pure_diffusion_small \
#   --epochs 20000 \
#   --warmup_epochs 2000 \
#   --batch_size 4096 \
#   --blr 9e-6 \
#   --lr_schedule cosine \
#   --use_cached \
#   --cached_format ptshard \
#   --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard \
#   --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/diffusion-small-9e-6-energy-full-alpha-test \
#   --seed 42 \
#   --preview \
#   --preview_interval 100


# ————  Trainning command for standard diffusion ---
torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=9382 \
  main_mar.py \
  --run_name standard-diffusion-9e-6-test-online-eval \
  --img_size 256 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type pure_diffusion \
  --model pure_diffusion_small \
  --epochs 80000 \
  --warmup_epochs 2000 \
  --batch_size 4096 \
  --blr 9e-6 \
  --lr_schedule cosine \
  --use_cached \
  --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet-1k-part \
  --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/standard-diffusion-9e-6-test-online-eval \
  --seed 42 \
  --online_eval \
  --class_num 6 \
  --num_images 4092 \
  --eval_bsz 4096 \
  --eval_freq 2


