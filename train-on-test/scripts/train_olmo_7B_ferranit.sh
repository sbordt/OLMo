#!/bin/bash
#SBATCH --time=3-00:00:00  # Runtime in D-HH:MM:SS    
#SBATCH --output=/weka/luxburg/sbordt10/logs/olmo/%j.out  
#SBATCH --error=/weka/luxburg/sbordt10/logs/olmo/%j.err   
#SBATCH --open-mode=append
#SBATCH --job-name=olmo  
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64    
#SBATCH --mem=500G   
#SBATCH --gres=gpu:4              

scontrol show job ${SLURM_JOB_ID}
nvidia-smi
export NCCL_TIMEOUT=1800000
export WANDB__SERVICE_WAIT=6000

cd /weka/luxburg/sbordt10/OLMo
source activate olmo-3.11

torchrun --nproc_per_node=4 --master_port=29501 scripts/train.py configs/official/OLMo-7B-step300080-unsharded.yaml --load_path="/weka/luxburg/sbordt10/v1_5-mix-mitch-ish/latest-unsharded"