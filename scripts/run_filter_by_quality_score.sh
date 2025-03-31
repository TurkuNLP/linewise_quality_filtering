#!/bin/bash
#SBATCH --job-name=data_preprocessing
#SBATCH --account=project_462000353
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

srun python3 filter_by_quality_score.py

