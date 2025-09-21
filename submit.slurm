#!/bin/bash
#SBATCH --job-name=ebm_eval
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --partition=ghx4
#SBATCH --account=bdjz-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4       # number of processes for torchrun
#SBATCH --gpus-per-node=4         # request 4 GPUs
#SBATCH --cpus-per-task=32        # number of CPU cores per process
#SBATCH --mem=256G
#SBATCH --time=20:59:00

# ------------------ Setup ------------------
# Always good to load modules if needed, e.g.:
# module load cuda/12.1
# module load anaconda

# Arguments: $1=resume_path
bash evaluate.sh "$@"
