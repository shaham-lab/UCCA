#!/bin/bash
#SBATCH --job-name=create_cfg
#SBATCH --partition=generic
#SBATCH --output=outs/%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/dsi/nirbenari/miniconda3/envs/sca_env

python run.py
