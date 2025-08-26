#!/bin/bash
#SBATCH --job-name=train_classifier
#SBATCH --account=project_462000353
#SBATCH --partition=small-g
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40GB
#SBATCH --gpus-per-node=1
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err


module purge
module use /appl/local/csc/modulefiles
module load pytorch

source ../.venv/bin/activate

LANG_ID="eus_Latn"

run_id="${LANG_ID}_classifier_smoothing"

srun python3 ../src/train_classifier.py --run-id=$run_id \
                                        --data-path="../data/train_dev_test/hplt_${LANG_ID}_linequality" \
                                        --base-model="FacebookAI/xlm-roberta-large" \
                                        --learning-rate=0.00002 \
                                        --use-label-smoothing \
                                        --train \
