#!/bin/bash
#SBATCH --job-name=EEGConformer
#SBATCH --output=logs/eeg_conformer.out
#SBATCH --error=logs/eeg_conformer.err
#SBATCH --gres=gpu:rtx3090:1               
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --cpus-per-gpu=4            

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "/Brain/private" ]; then
    export HOME="/Brain/private/${USER}"
    export XDG_CACHE_HOME="$HOME/.cache"
    export HF_HOME="$HOME/.cache/huggingface"
    export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
    export TORCH_HOME="$HOME/.cache/torch"
fi

echo "starting job on $(hostname) at $(date)"

# Go to the repository directory
cd "$SCRIPT_DIR"

# Activate the virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Running with system python."
fi

python train.py --n-seeds 3

