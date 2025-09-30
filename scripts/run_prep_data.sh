#!/bin/bash
#SBATCH --job-name=data_preprocessing
#SBATCH --account=project_462000353
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

LANG_CODE=$1

DATA_PATH="../results/LLM_labelled_data/hplt_${LANG_CODE}/results_hplt_${LANG_CODE}.jsonl"
SAVE_PATH="../data/train_dev_test/hplt_${LANG_CODE}_linequality"

srun python3 ../src/prep_data_for_training.py --data-path=$DATA_PATH --save-path=$SAVE_PATH

