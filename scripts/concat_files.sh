#!/bin/bash
#SBATCH --job-name=concat_files
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

LANG_ID="deu_Latn"

srun python3 ../src/concat_jsonl.py \
    --input-dir="../data/hplt/line_quality_labelled/${LANG_ID}/just_clean_trimmed" \
    --output="../data/hplt/line_quality_labelled/${LANG_ID}/just_clean_trimmed/concatenated/${LANG_ID}_just_clean_trimmed.jsonl" \
