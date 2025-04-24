#!/bin/bash
#SBATCH --job-name=filter_data
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

# This script is used to remove low-quality labelled lines from data.
# First, give the correct DATA_DIR (where the data comes from)
# and OUT_DIR (where results will be saved).
# Give ORIGINAL_FILE_TYPE that matches the file extion of your data, e.g. .jsonl, .parquet, .csv, etc.

# Below, in the srun command, choose cleaning strategy: either trim or filter.
# Give names of labels to be removed. See valid labels from the main python script.

# Run this script by calling sbatch run_filter_by_quality_labels.sh <file_prefix>
# File prefix is the start of a bunch of files that will be processed at once, e.g. 00000_
# At most, 30 files can be processed at once with this script.


DATA_DIR="../data/fineweb2_tur_Latn_line_quality_labelled/full"
OUT_DIR="../data/fineweb2_tur_Latn_line_quality_labelled/just_clean_trimmed"

mkdir -p $OUT_DIR

FILE_PREFIX="$1"
ORIGINAL_FILE_TYPE=".jsonl"

# Create an array of files starting with FILE_PREFIX (filenames only)
file_array=()
while IFS= read -r -d '' file; do
    filename="${file##*/}"        # strips path, keeps only filename
    file_array+=("$filename")
done < <(find "$DATA_DIR" -maxdepth 1 -type f -name "${FILE_PREFIX}*" -print0)

echo "Found ${#file_array[@]} files starting with $FILE_PREFIX"

# Process each file
for i in "${!file_array[@]}"; do
    filename="${file_array[i]}"
    input_path="$DATA_DIR/$filename"
    filename="${filename%${ORIGINAL_FILE_TYPE}}"
    output_path="$OUT_DIR/$filename.jsonl"
        
    srun \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=8G \
        python3 ../src/filter_by_quality_label.py \
        --quality-labels="all" \
        --trim \
        --data-path="$input_path" \
        --save-path="$output_path" \
        &
    done

# Wait for all background processes to complete
wait

echo "All filtering tasks completed."

# Exit with success
exit 0

