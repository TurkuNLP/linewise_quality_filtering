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

# Give input dir for directory of files you want to split. Can be in .JSONL or .JSONL.ZST format
# Give split-count for how many files you want to split each file in input-dir
# Resulting files will be named <original_file_name>_<split_number>.jsonl

LANG_ID="$1"

srun python3 ../src/split_large_files.py \
    --input-dir "${SLURM_SUBMIT_DIR}/../data/hplt_dedup/${LANG_ID}" \
    --output-dir "${SLURM_SUBMIT_DIR}/../data/hplt_dedup/${LANG_ID}/splits" \
    --split-count 5