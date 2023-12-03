#!/bin/bash

#SBATCH --job-name=graphvae_torch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./Data/log.txt

conda activate torch-gpu
python3 hpc_run.py
