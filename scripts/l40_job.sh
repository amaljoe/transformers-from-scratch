#!/bin/bash
#SBATCH --job-name=l40-job
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00

# Print node info
hostname
nvidia-smi

source scripts/setup-compute.sh

jupyter lab --ip=0.0.0.0 --port=8888

sleep infinity

