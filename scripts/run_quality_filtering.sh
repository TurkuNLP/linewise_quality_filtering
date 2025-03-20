#!/bin/bash
#SBATCH --job-name=vllm_inference
#SBATCH --account=project_462000353
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --gpus-per-node=8
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

# Load pytorch module
module purge
module use /appl/local/csc/modulefiles
module load pytorch

# Activate venv
source ../.venv/bin/activate

# Check if you are logged in to HuggingFace
echo "HuggingFace username:"
huggingface-cli whoami

# Memory management
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

gpu-energy --save

run_id="hplt_en_classification"

srun python3 label_lines.py --run-id=$run_id \
                            --temperature=0.1 \
                            --batch-size=10 \
                            --max-vocab=50 \
                            --synonym-threshold=0.3 \
                            --start-index=10002 \
                            --stop-index=40000 \
                            --language="english" \
                            --use-fixed-labels

gpu-energy --diff

