#!/bin/bash

#SBATCH --account=bcsi-dtai-gh # fill out with your allocation/account
#SBATCH --partition=ghx4
### NODE/CPU/MEM/GPU  ###
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=72
#SBATCH --nodes=1