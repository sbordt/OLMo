#!/bin/bash
#SBATCH --time=0-00:30:00  # Runtime in D-HH:MM:SS    
#SBATCH --output=/weka/luxburg/sbordt10/logs/single-training-run/%j.out  
#SBATCH --error=/weka/luxburg/sbordt10/logs/single-training-run/%j.err   
#SBATCH --open-mode=append
#SBATCH --job-name=olmo  
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    
#SBATCH --mem=500G   
#SBATCH --gres=gpu:2              

scontrol show job ${SLURM_JOB_ID}
nvidia-smi
export NCCL_TIMEOUT=1800000
export WANDB__SERVICE_WAIT=6000

cd /weka/luxburg/sbordt10/OLMo
source activate olmo-2

torchrun --nproc_per_node=2 --master_port=29501 scripts/train.py single-training-run/scripts/OLMo2-1B-Mod.yaml 