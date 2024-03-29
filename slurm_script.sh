#!/bin/bash

#SBATCH --job-name=graphvae_torch
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -N 1
#SBATCH -J graph_vae_torch
#SBATCH -p gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem=50G 
#SBATCH --output=./Data/log.txt

cd /home/dpr/Projects/GraphVAE_NPP/Quark_Gluon_data/
source /share/apps/modulefiles/conda_init.sh
conda activate torch-geo

python3 hpc_run.py
