#!/bin/bash
#SBATCH --job-name=filter_data
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

DATA_DIR="$2"
OUT_DIR="../data/finerweb-350BT-filtered-threshold-09"

files="$1"
# files = "path/to/file1,path/to/file2,path/to/file3"

# Convert comma-separated list to array
IFS=',' read -ra file_array <<< "$files"

# Process each file
for i in "${!file_array[@]}"; do
    filename="${file_array[i]}"
    filename=$(basename "$filename") #remove full path
    input_path="$DATA_DIR/$filename"
    output_path="$OUT_DIR/$filename.jsonl"
        
    srun \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=8G \
        srun python3 ../src/fix_and_validate_json.py \
        --path=$input_path
        &
    done

# Wait for all background processes to complete
wait

echo "All tasks completed."

# Exit with success
exit 0




