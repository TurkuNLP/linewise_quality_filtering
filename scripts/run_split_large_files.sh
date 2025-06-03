#!/bin/bash
#SBATCH --job-name=split_files
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

# === Load Required Modules ===
module purge
module use /appl/local/csc/modulefiles
module load pytorch

# === Activate Python Virtual Environment ===
source ../.venv/bin/activate

LANG_ID="$1"

srun python3 ../src/split_large_files.py \
    --input-dir "/scratch/project_462000353/data/hplt/all_languages/${LANG_ID}" \
    --output-dir "${SLURM_SUBMIT_DIR}/../data/hplt/${LANG_ID}" \
    --split-count 10