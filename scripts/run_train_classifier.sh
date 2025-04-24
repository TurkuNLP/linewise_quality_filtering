#!/bin/bash
#SBATCH --job-name=train_classifier
#SBATCH --account=project_462000353
#SBATCH --partition=small-g
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40GB
#SBATCH --gpus-per-node=1
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err


module purge
module use /appl/local/csc/modulefiles
module load pytorch

source ../.venv/bin/activate

run_id="tur_Latn_classifier"

srun python3 ../src/train_classifier.py --run-id=$run_id \
                                        --data-path="../data/train_dev_test/hplt_tur_Latn_linequality" \
                                        --base-model="FacebookAI/xlm-roberta-large" \
                                        --learning-rate=0.00002\
                                        --train \
