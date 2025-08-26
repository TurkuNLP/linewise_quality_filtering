#!/bin/bash
#SBATCH --job-name=data_salvage
#SBATCH --account=project_462000353
#SBATCH --partition=dev-g
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

# Load pytorch module
module purge
module use /appl/local/csc/modulefiles
module load pytorch

source ../.venv/bin/activate

srun python3 ../src/salvage_low_quality_data.py --model-path="../models/line_quality_classifier_fin_Latn" \
                                                --data-path="../data/hplt_dedup/fin_Latn/1.jsonl.zst" \
                                                --save-path="../data/hplt_dedup_salvaged/fin_Latn/fin_Latn_dedup_line_quality_labelled.jsonl" \
                                                --filter