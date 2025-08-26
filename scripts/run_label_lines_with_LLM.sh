#!/bin/bash
#SBATCH --job-name=label_lines_with_LLM
#SBATCH --account=project_462000353
#SBATCH --partition=standard-g
#SBATCH --time=23:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --gpus-per-node=8
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
#SBATCH --array=1-4

# Load pytorch module
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5

# Activate venv
source ../.venv_pt2.5/bin/activate

# Check if you are logged in to HuggingFace
echo "HuggingFace username:"
huggingface-cli whoami

# Memory management
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

gpu-energy --save

LANG_ID="$1"

run_id="hplt_${LANG_ID}_${SLURM_ARRAY_TASK_ID}"

start_index=$(( ($SLURM_ARRAY_TASK_ID - 1) * 10000))
stop_index=$(( $SLURM_ARRAY_TASK_ID * 10000))

srun python3 ../src/label_lines_with_LLM.py --run-id=$run_id \
                                            --temperature=0.1 \
                                            --batch-size=10 \
                                            --max-vocab=50 \
                                            --synonym-threshold=0.3 \
                                            --start-index=$start_index \
                                            --stop-index=$stop_index \
                                            --language=${LANG_ID} \
                                            --use-fixed-labels

gpu-energy --diff

