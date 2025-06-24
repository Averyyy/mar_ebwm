#!/bin/bash

#SBATCH --account=bdta-delta-gpu
#SBATCH --partition=gpuA100x4
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=8

source activate mar
