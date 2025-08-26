#!/bin/bash
#SBATCH --job-name=nemotron_quality_pipeline
#SBATCH --account=project_462000353
#SBATCH --partition=small-g
#SBATCH --time=1:55:00
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

srun python3 ../src/nemotron_quality_pipeline.py