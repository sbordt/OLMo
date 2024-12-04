#!/bin/bash
#SBATCH --time=0-02:00:00  # Runtime in D-HH:MM:SS    
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
export WANDB__SERVICE_WAIT=9000

cd /weka/luxburg/sbordt10/OLMo
source activate olmo-3.11

# Check if the checkpoint URL is provided
if [ -z "$1" ]; then
  echo "Error: No checkpoint URL provided."
  echo "Usage: $0 <checkpoint_url>"
  exit 1
fi

CHECKPOINT_URL=$1

# generate a random prefix for the output directory name
PREFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

# Print the checkpoint URL to the command line
echo "Checkpoint URL: $CHECKPOINT_URL"

python train-on-test/download_checkpoint.py --checkpoint_url "$CHECKPOINT_URL" --output_dir "/weka/luxburg/sbordt10/OLMo-7B-checkpoints/$PREFIX-eval-unsharded"

torchrun --nproc_per_node=4 scripts/train.py configs/official/OLMo-7B-step300080-eval_only.yaml --load_path="/weka/luxburg/sbordt10/OLMo-7B-checkpoints/$PREFIX-eval-unsharded"

rm -r "/weka/luxburg/sbordt10/OLMo-7B-checkpoints/$PREFIX-eval-unsharded"