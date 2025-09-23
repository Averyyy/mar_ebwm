#!/bin/bash

# #SBATCH --account=ar-ai-hipri
# #SBATCH --qos=ar-ai-hipri
#SBATCH --qos=midpri
#SBATCH --reservation=research

### NODE/CPU/MEM/GPU  ###
### #SBATCH --ntasks=1
#SBATCH --wait-all-nodes=1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=0

########################

#SBATCH --array=0
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
# #SBATCH --exclusive

########################
#SBATCH --job-name=ebffn_base
#SBATCH --output=logs/slurm/ebffn_base%A-%a.log

# ------------------ Setup ------------------
# Always good to load modules if needed, e.g.:
# module load cuda/12.1
# module load anaconda

# Arguments: $1=resume_path
# bash evaluate.sh "$@"

# TODO set task here properly

# bash dit_gen_images.sh "$@"

bash metrics.sh "$@"
