#!/bin/bash

#SBATCH --account=belh-dtai-gh # fill out xxxx with your allocation
#SBATCH --partition=ghx4
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=72

source activate mar_gh200
