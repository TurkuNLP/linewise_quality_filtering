#!/bin/bash
#SBATCH --job-name=concat_files
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

# Directory containing the files
INPUT_DIR="$1"
# Output file
OUTPUT_FILE="$2"

# Create or empty the output file
> "$OUTPUT_FILE"

# Loop through all files in the directory
for file in "$INPUT_DIR"/*; do
    if [[ $file = "0"* ]]; then
        # Check if it's a regular file
        if [[ -f "$file" ]]; then
            echo "Adding: $file"
            cat "$file" >> "$OUTPUT_FILE"
        fi
    fi
done

echo "All files concatenated into $OUTPUT_FILE"
