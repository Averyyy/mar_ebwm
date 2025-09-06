#SBATCH --job-name=energy-diffusion-eval-compare
#SBATCH --output=logs/slurm/energy-diffusion/eval-compare-%A/energy-diffusion-%a.out
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1

# --- Environment Setup ---
# Set these variables for your system:
export REPO_ROOT="/work/hdd/bdta/aqian1/mar_ebwm"  # Change this to your repo path
export DATA_ROOT="/work/hdd/bdta/aqian1/data"     # Change this to your data path
export CACHE_ROOT="/work/nvme/bdta/aqian1/data"   # Change this to your cache path



# --- Setup ---
module load cuda/12.6.1
source activate mar_gh200
cd ${REPO_ROOT}

# --- Choose a random master port (per run) ---
if [ -z "${MASTER_PORT}" ]; then
  if command -v shuf >/dev/null 2>&1; then
    MASTER_PORT=$(shuf -i 20000-65000 -n 1)
  else
    MASTER_PORT=$(( (RANDOM % 45000) + 20000 ))
  fi
fi
echo "Using MASTER_PORT=${MASTER_PORT}"

# --- Branch 1: always accept optimization steps ---
echo "[Branch 1] always_accept_opt_steps = True"
mkdir -p \
  ${REPO_ROOT}/output/EDM-eval-s64-step_0.001-diffusion_step-500-c1k-always_accept

torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=${MASTER_PORT} \
  main_ebm.py \
  --run_name EDM-eval-s64-step_0.001-diffusion_step-500-c1k-always_accept \
  --img_size 64 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ebm \
  --model ebm_small \
  --use_energy \
  --use_innerloop_opt \
  --always_accept_opt_steps \
  --mcmc_step_size 0.001 \
  --use_cached \
  --cached_path ${CACHE_ROOT}/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --data_path ${DATA_ROOT}/imagenet-1k-64 \
  --diffusion_timesteps 500 \
  --num_sampling_steps 250 \
  --evaluate \
  --use_fid_stats \
  --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
  --eval_real_dataset ${DATA_ROOT}/imagenet-1k-64/val \
  --eval_bsz 256 \
  --num_images 1000 \
  --output_dir ${REPO_ROOT}/output/EDM-eval-s64-step_0.001-diffusion_step-500-c1k-always_accept \
  --resume ${REPO_ROOT}/output/EDM-step_0.001-cl0-rl0-closs0.05-rloss0.1-lr_9e-6-small-64-bz1024-epo2000-c1k

# --- Branch 2: vanilla accept/reject energy logic ---
echo "[Branch 2] always_accept_opt_steps = False (vanilla accept/reject)"
mkdir -p \
  ${REPO_ROOT}/output/EDM-eval-s64-step_0.001-diffusion_step-500-c1k-vanilla

torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=${MASTER_PORT} \
  main_ebm.py \
  --run_name EDM-eval-s64-step_0.001-diffusion_step-500-c1k-vanilla \
  --img_size 64 \
  --use_cached \
  --cached_path ${CACHE_ROOT}/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --data_path ${DATA_ROOT}/imagenet-1k-64 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type ebm \
  --model ebm_small \
  --use_energy \
  --use_innerloop_opt \
  --mcmc_step_size 0.001 \
  --diffusion_timesteps 500 \
  --num_sampling_steps 250 \
  --use_fid_stats \
  --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
  --eval_real_dataset ${DATA_ROOT}/imagenet-1k-64/val \
  --eval_bsz 256 \
  --num_images 1000 \
  --output_dir ${REPO_ROOT}/output/EDM-eval-s64-step_0.001-diffusion_step-500-c1k-vanilla \
  --resume ${REPO_ROOT}/output/EDM-step_0.001-cl0-rl0-closs0.05-rloss0.1-lr_9e-6-small-64-bz1024-epo2000-c1k \
  --evaluate

echo "--- Energy Diffusion EVAL Compare job completed ---"


