#SBATCH --job-name=unlearnable
#SBATCH --array=0-0
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm/logs/slurm/energy_diffusion/%A/energy-diffusion-grid-%a.out
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00


# # --- Grid Search Parameters ---
# step_sizes=(0.1)

# # --- Calculate parameters for this job ---
# step_size=${step_sizes[$SLURM_ARRAY_TASK_ID]}
# multiplier=$(echo "${step_size} * 3" | bc -l)

# # --- Setup ---
# module load cuda/12.6.1
# source activate mar_gh200
# cd /work/hdd/bdta/aqian1/mar_ebwm

# # --- Run Name and Output Dir ---
# RUN_NAME="energy-diffusion-step_${step_size}-mult_${multiplier}-without_cl_loss"
# OUTPUT_DIR="/work/hdd/bdta/aqian1/mar_ebwm/output/${RUN_NAME}"

# # --- Log Parameters ---
# echo "--- Starting Energy Diffusion Grid Search job ${SLURM_ARRAY_TASK_ID} ---"
# echo "MCMC Step Size: ${step_size}"
# echo "MCMC Step Size LR Multiplier: ${multiplier}"
# echo "Base Learning Rate: 9e-6"
# echo "Run Name: ${RUN_NAME}"
# echo "Output Dir: ${OUTPUT_DIR}"
# echo "--------------------"

# # --- Training Command for Energy Diffusion ---
# torchrun \
#   --nproc_per_node=1 \
#   --master_addr=localhost \
#   --master_port=$((5748 + SLURM_ARRAY_TASK_ID)) \
#   main_mar.py \
#   --run_name ${RUN_NAME} \
#   --img_size 64 \
#   --vae_path pretrained_models/vae/kl16.ckpt \
#   --model_type pure_diffusion \
#   --use_energy \
#   --use_innerloop_opt \
#   --mcmc_step_size ${step_size} \
#   --mcmc_step_size_lr_multiplier ${multiplier} \
#   --model pure_diffusion_small \
#   --epochs 80000 \
#   --warmup_epochs 2000 \
#   --batch_size 4096 \
#   --blr 9e-6 \
#   --lr_schedule cosine \
#   --use_cached \
#   --cached_format ptshard \
#   --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard \
#   --output_dir ${OUTPUT_DIR} \
#   --seed 42 \
#   --preview \
#   --preview_interval 100


# echo "--- Energy Diffusion Grid Search job ${SLURM_ARRAY_TASK_ID} completed ---"


#   # --supervise_energy_landscape \
#   # --wandb_log_mse_only \
#   # --always_accept_opt_steps \





# torchrun \
#   --nproc_per_node=4 \
#   --master_addr=localhost \
#   --master_port=4837 \
#   main_mar.py \
#   --run_name test_edm_256 \
#   --img_size 256 \
#   --vae_path pretrained_models/vae/kl16.ckpt \
#   --model_type pure_diffusion \
#   --use_energy \
#   --use_innerloop_opt \
#   --mcmc_step_size 0.1 \
#   --model pure_diffusion_small \
#   --epochs 500 \
#   --warmup_epochs 5 \
#   --batch_size 256 \
#   --blr 9e-6 \
#   --lr_schedule cosine \
#   --use_cached \
#   --cached_format pt \
#   --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-256-pt-c7  \
#   --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/test_edm_256 \
#   --seed 42 \
#   --online_eval \
#   --eval_freq 1 \
#   --eval_real_dataset /work/nvme/belh/aqian1/imagenet-1k/val \
#   --num_sampling_steps 5 \
#   --eval_bsz 128 \
#   --num_images 1000 \
#   --class_num 1000 \
#   --val \
#   --val_batch_size 256 \
#   --val_data_path /work/nvme/belh/aqian1/imagenet-1k/val



# torchrun \
#   --nproc_per_node=1 \
#   --master_addr=localhost \
#   --master_port=5750 \
#   main_mar.py \
#   --img_size 64 \
#   --vae_path pretrained_models/vae/kl16.ckpt \
#   --model_type pure_diffusion \
#   --model pure_diffusion_small \
#   --use_energy \
#   --use_innerloop_opt \
#   --supervise_energy_landscape \
#   --wandb_log_mse_only \
#   --mcmc_step_size 0.1 \
#   --log_energy_accept_rate \
#   --num_workers 16 \
#   --epochs 80000 \
#   --warmup_epochs 2000 \
#   --batch_size 2048 \
#   --blr 9e-6 \
#   --lr_schedule cosine \
#   --use_cached \
#   --cached_format ptshard \
#   --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard-c7 \
#   --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/EDM-64-c7-TEST \
#   --val \
#   --val_batch_size 1024 \
#   --val_data_path /work/hdd/bdta/aqian1/data/imagenet-1k-64/val \
#   --val_freq 20 \
#   --num_images 700 \
#   --class_num 7 \
#   --preview \
#   --preview_interval 20 \
#   --linear_then_mean \
#   --online_eval \
#   --eval_freq 1000 \
#   --eval_real_dataset /work/hdd/bdta/aqian1/data/val-64-c7 \
#   --kid_subset_size 400 \
#   --eval_bsz 1024 \


# torchrun \
#   --nproc_per_node=4 \
#   --master_addr=localhost \
#   --master_port=15149 \
#   main_mar.py \
#   --img_size 64 \
#   --vae_path pretrained_models/vae/kl16.ckpt \
#   --model_type pure_diffusion \
#   --model pure_diffusion_small \
#   --use_energy \
#   --use_innerloop_opt \
#   --supervise_energy_landscape \
#   --wandb_log_mse_only \
#   --mcmc_step_size 0.1 \
#   --learnable_mcmc_step_size \
#   --log_energy_accept_rate \
#   --epochs 500 \
#   --warmup_epochs 5 \
#   --batch_size 1024 \
#   --num_workers 8 \
#   --blr 9e-6 \
#   --use_cached \
#   --cached_path /work/nvme/bdta/aqian1/data/cached-imagenet1k-64-ptshard-32 \
#   --cached_format ptshard \
#   --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/test-edm \
#   --preview \
#   --preview_interval 1 \
#   --preview_labels 0,1,2,3,4,5,6,7,8,9,10,11,113,130,282,283,284,309,430,485,605,726,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,990,991,992,993,994,995,996,997,998,999

  # --run_name EDM-syn-dataloader \

torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=1491 \
  main_mar.py \
  --run_name EDM-8192-32 \
  --img_size 64 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type pure_diffusion \
  --model pure_diffusion_small \
  --use_energy \
  --use_innerloop_opt \
  --epochs 500 \
  --warmup_epochs 500 \
  --batch_size 8192 \
  --num_workers 32 \
  --blr 9e-6 \
  --output_dir /work/nvme/bdta/aqian1/output \
  --use_cached \
  --cached_path /work/nvme/bdta/aqian1/data/cached-imagenet1k-64-ptshard-32 \
  --cached_format ptshard \


  # --use_streaming \
  # --stream_buffer_size 16




  # --output_dir /work/hdd/bdta/aqian1/mar_ebwm/output/test-edm-streaming-4096-16 \


  # --run_name SDM-syn-dataloader \
  # --use_energy \
  # --contrasive_loss_scale 0 \
  # --mcmc_refinement_loss_scale 0 \




# quick test fp16 experiments
torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=4831 \
  main_mar.py \
  --run_name EDM-test-dtype \
  --img_size 64 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type pure_diffusion \
  --model pure_diffusion_base \
  --epochs 500 \
  --warmup_epochs 10 \
  --use_energy \
  --use_innerloop_opt \
  --mcmc_step_size 0.001 \
  --diffusion_timesteps 500 \
  --batch_size 1024 \
  --num_workers 64 \
  --train_dtype fp32 \
  --eval_dtype fp32 \
  --auxiliary_eval_dtypes "bf16,fp16" \
  --blr 9e-6 \
  --use_cached \
  --cached_path /work/hdd/bdta/aqian1/data/cached-imagenet1k-64-ptshard-c7 \
  --cached_format ptshard \
  --output_dir /work/nvme/bdta/aqian1/output/EDM-test \
  --online_eval \
  --eval_freq 1 \
  --use_fid_stats \
  --fid_stats_file fid_stats/imagenet_64_stats.npz \
  --eval_real_dataset /work/hdd/bdta/aqian1/data/val-64-c7 \
  --num_sampling_steps 20 \
  --eval_bsz 256 \
  --num_images 7 \
  --class_num 7 \
  --disable_progress_bar

torchrun \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=6751 \
  main_mar.py \
  --img_size 256 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type pure_diffusion \
  --model pure_diffusion_base \
  --epochs 20 \
  --warmup_epochs 1 \
  --use_energy \
  --use_innerloop_opt \
  --diffusion_timesteps 500 \
  --batch_size 256 \
  --num_workers 16 \
  --blr 9e-6 \
  --use_cached \
  --cached_path /work/nvme/bdta/aqian1/data/cached-imagenet1k-256-ptshard-64 \
  --cached_format ptshard \
  --output_dir output/test_edm_256_bz256_9e-6-mcmc2



torchrun \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=9842 \
  main_mar.py \
  --run_name test-step \
  --img_size 256 \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --model_type pure_diffusion \
  --model pure_diffusion_base \
  --epochs 20 \
  --warmup_epochs 1 \
  --use_energy \
  --use_innerloop_opt \
  --diffusion_timesteps 500 \
  --batch_size 128 \
  --grad_accu 2 \
  --num_workers 8 \
  --blr 9e-6 \
  --use_cached \
  --cached_path /work/nvme/bdta/aqian1/data/cached-imagenet1k-256-ptshard-64 \
  --cached_format ptshard \
  --output_dir output/test_edm_256_bz1024