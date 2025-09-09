#!/bin/bash

#SBATCH --qos=midpri

### NODE/CPU/MEM/GPU  ###
### #SBATCH --ntasks=1
#SBATCH --wait-all-nodes=1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=0
