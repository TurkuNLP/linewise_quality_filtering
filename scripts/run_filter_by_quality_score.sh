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

DATA_DIR="/scratch/project_462000615/ehenriks/FINEWEB/quality_predictions/sample/350BT"
OUT_DIR="../data/finerweb-350BT-trimmed-threshold-05"

FILE_PREFIX="$1"

# Create an array of files starting with FILE_PREFIX (filenames only)
file_array=()
while IFS= read -r -d '' file; do
    filename="${file##*/}"        # strips path, keeps only filename
    file_array+=("$filename")
done < <(find "$DATA_DIR" -maxdepth 1 -type f -name "$FILE_PREFIX" -print0)

# Process each file
for i in "${!file_array[@]}"; do
    filename="${file_array[i]}"
    input_path="$DATA_DIR/$filename"
    filename="${filename%.parquet}"
    output_path="$OUT_DIR/$filename.jsonl"
        
    srun \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=8G \
        python3 ../src/filter_by_quality_score.py \
        --quality-threshold=0.5 \
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

