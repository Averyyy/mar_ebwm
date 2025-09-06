# bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/train_debt.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/test_energy_diffusion_train.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/train_EDM_large_64.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/train_SDM_base_64.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/train_SDM_large_64.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/train_EDM_xlarge_64.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/eval_EDM_compare_accept.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/train_EDM_base_64_dtype_grid.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/train_EDM_base_256_dtype_grid.sh

bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/train_EDM_base_256_bz256.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/train_EDM_base_256_bz1024.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/img64/train_EDM_xlarge_64.sh
bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/energy_diffusion/done/EDM_base_preview_test.sh

bash slurm/slurm_exec.sh ncsa_gh200 slurm/job_configs/cache_latents.sh

