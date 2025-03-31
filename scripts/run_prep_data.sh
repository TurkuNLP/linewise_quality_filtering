#!/bin/bash
#SBATCH --job-name=data_preprocessing
#SBATCH --account=project_462000353
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

srun python3 prep_data.py --data-path="../results/hplt_fr_classification/results_hplt_fr_classification.jsonl" \
                          --save-path="../data/hplt_fr_linequality"

