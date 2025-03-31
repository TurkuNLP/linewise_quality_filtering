#!/bin/bash
#SBATCH --job-name=train_classifier
#SBATCH --account=project_2011109
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40GB
#SBATCH --gres=gpu:a100:1
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err


# NB! This script is for Mahti.
# LUMI's AMD GPUs gave me a hard time :(

module load pytorch
source ../.venv/bin/activate

export HF_HOME="../../hf_cache"

run_id="XLMR6_mahti"

srun python3 train_classifier.py --run-id=$run_id \
                                 --base-model="FacebookAI/xlm-roberta-large" \
                                 --learning-rate=0.00004\
                                 --train \
