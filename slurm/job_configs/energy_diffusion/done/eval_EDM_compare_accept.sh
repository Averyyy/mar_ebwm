#SBATCH --job-name=energy-diffusion-eval-compare
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm/energy-diffusion/eval-compare-%A/energy-diffusion-%a.out
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1



# --- Setup ---
module load cuda/12.6.1
source activate mar_gh200
cd /work/hdd/bdta/aqian1/mar_ebwm

# --- Branch 1: always accept optimization steps ---
echo "[Branch 1] always_accept_opt_steps = True"
mkdir -p \
  /work/hdd/bdta/aqian1/mar_ebwm/output/EDM-eval-s64-step_0.001-diffusion_step-500-c1k-always_accept

torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=6748 \
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
  --cached_path /work/nvme/bdta/aqian1/data/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --data_path /work/hdd/bdta/aqian1/data/imagenet-1k-64 \
  --diffusion_timesteps 500 \
  --num_sampling_steps 250 \
  --evaluate \
  --use_fid_stats \
  --fid_stats_file util/fid_stats/imagenet_64_stats.npz \
  --eval_real_dataset /work/hdd/bdta/aqian1/data/imagenet-1k-64/val \
  --eval_bsz 256 \
  --num_images 1000 \
  --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/EDM-eval-s64-step_0.001-diffusion_step-500-c1k-always_accept \
  --resume /work/hdd/bdta/aqian1/mar_ebwm/output/EDM-step_0.001-cl0-rl0-closs0.05-rloss0.1-lr_9e-6-small-64-bz1024-epo2000-c1k

# --- Branch 2: vanilla accept/reject energy logic ---
echo "[Branch 2] always_accept_opt_steps = False (vanilla accept/reject)"
mkdir -p \
  /work/hdd/bdta/aqian1/mar_ebwm/output/EDM-eval-s64-step_0.001-diffusion_step-500-c1k-vanilla

torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=7748 \
  main_ebm.py \
  --run_name EDM-eval-s64-step_0.001-diffusion_step-500-c1k-vanilla \
  --img_size 64 \
  --use_cached \
  --cached_path /work/nvme/bdta/aqian1/data/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \
  --data_path /work/hdd/bdta/aqian1/data/imagenet-1k-64 \
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
  --eval_real_dataset /work/hdd/bdta/aqian1/data/imagenet-1k-64/val \
  --eval_bsz 256 \
  --num_images 1000 \
  --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/EDM-eval-s64-step_0.001-diffusion_step-500-c1k-vanilla \
  --resume /work/hdd/bdta/aqian1/mar_ebwm/output/EDM-step_0.001-cl0-rl0-closs0.05-rloss0.1-lr_9e-6-small-64-bz1024-epo2000-c1k \
  --evaluate

echo "--- Energy Diffusion EVAL Compare job completed ---"


