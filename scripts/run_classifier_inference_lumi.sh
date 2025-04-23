#!/bin/bash
#SBATCH --job-name=classifier_inference
#SBATCH --account=project_462000353
#SBATCH --partition=standard-g
#SBATCH --time=23:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:mi250:8
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

# activate venv to use sentence_transformers, since it's not part of the pytorch module.
# If you don't use sentence_transformers, all you need is in the pytorch module.
source ../.venv/bin/activate

gpu-energy --save

DATA_DIR="$2"
OUT_DIR="../data/fineweb2_fra_line_quality_labelled_split"

files="$1"
# files = "path/to/file1,path/to/file2,path/to/file3"

# Convert comma-separated list to array
IFS=',' read -ra file_array <<< "$files"

# Process each file
for i in "${!file_array[@]}"; do
    filename="${file_array[i]}"
    filename=$(basename "$filename") #remove full path
    input_path="$DATA_DIR/$filename"
    output_path="$OUT_DIR/$filename"
    
    echo "Loading data from $input_path"
    echo "Saving results to $output_path"

    srun \
        --ntasks=1 \
        --gres=gpu:mi250:1 \
        --mem=20G \
        accelerate launch ../src/classifier_inference.py \
        --data-path="$input_path" \
        --save-path="$output_path" \
        &
    done

# Wait for all background processes to complete
wait

gpu-energy --diff
