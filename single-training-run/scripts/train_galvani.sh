#!/bin/bash
#SBATCH --time=3-00:00:00  # Runtime in D-HH:MM:SS    
#SBATCH --output=/mnt/lustre/work/luxburg/sbordt10/logs/single-training-run/%x_%A_%a.out  
#SBATCH --error=/mnt/lustre/work/luxburg/sbordt10/logs/single-training-run/%x_%A_%a.err   
#SBATCH --open-mode=append
#SBATCH --job-name=olmo-mod  
#SBATCH --partition=a100-galvani 
#SBATCH --nodes=1  
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:2           


scontrol show job ${SLURM_JOB_ID}
nvidia-smi
export NCCL_TIMEOUT=1800000
export WANDB__SERVICE_WAIT=6000

cd /mnt/lustre/work/luxburg/sbordt10/OLMo
source activate olmo-2

torchrun --nproc_per_node=2 --master_port=29501 scripts/train.py single-training-run/scripts/OLMo2-1B-Mod.yaml 