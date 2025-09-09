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
