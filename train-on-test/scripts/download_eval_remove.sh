#!/bin/bash
#SBATCH --time=0-05:00:00  # Runtime in D-HH:MM:SS    
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

python train-on-test/download_checkpoint.py --checkpoint_url "https://olmo-checkpoints.org/ai2-llm/olmo-medium/obde4w9j/step310000-unsharded" --output_dir "/mnt/qb/luxburg/sbordt10/OLMo-7B-checkpoints/step310000-unsharded-eval-only"

torchrun --nproc_per_node=4 scripts/train.py configs/official/OLMo-7B-step300080-eval_only.yaml --load_path="/mnt/qb/luxburg/sbordt10/OLMo-7B-checkpoints/step310000-unsharded-eval-only"

#rm -r "/mnt/qb/luxburg/sbordt10/OLMo-7B-checkpoints/should_not_be_here"
#rm -r "/mnt/qb/luxburg/sbordt10/OLMo-7B-checkpoints/step310000-unsharded-eval-only"