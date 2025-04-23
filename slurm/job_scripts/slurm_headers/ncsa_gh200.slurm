#!/bin/bash

#SBATCH --account=belh-dtai-gh # fill out xxxx with your allocation
#SBATCH --partition=ghx4
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-bind=verbose,local
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=72
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/hdd/bdta/aqian1/mar_ebwm_coding/logs/slurm_outputs/mar-base-2048-%j.out

module load cuda/12.6.1
source activate mar_gh200
