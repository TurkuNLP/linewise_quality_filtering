#!/bin/bash
#SBATCH --job-name=split_files
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err

# This script will list all .jsonl files in $FILE_DIR
# and split each into $SPLIT_COUNT smaller files.
# The resulting files are saved in $OUTPUT_DIR
# Run this script by calling sbatch split_large_files.sh

FILE_DIR="/scratch/project_462000615/vitiugin/data/fineweb_tur"
OUTPUT_DIR="/scratch/project_462000353/tarkkaot/linewise_quality_filtering/data/fineweb2/fineweb2-tur_Latn"
SPLIT_COUNT=30

mkdir -p "$OUTPUT_DIR"

for INPUT_FILE in "$FILE_DIR"/*.jsonl; do
    BASENAME=$(basename "$INPUT_FILE" .jsonl)

    # Count total lines in the file
    TOTAL_LINES=$(wc -l < "$INPUT_FILE")
    LINES_PER_FILE=$(( (TOTAL_LINES + SPLIT_COUNT - 1) / SPLIT_COUNT ))  # Ceiling division

    # Use split to split file
    split -l "$LINES_PER_FILE" -d --additional-suffix=.jsonl "$INPUT_FILE" "$OUTPUT_DIR/${BASENAME}_"

    # Rename files to have unique names with 2-digit padding
    cd "$OUTPUT_DIR" || exit
    a=0
    for f in ${BASENAME}_*; do
        mv "$f" "${BASENAME}_$(printf "%02d" $a).jsonl"
        ((a++))
    done

    echo "Split $INPUT_FILE into $SPLIT_COUNT parts of $LINES_PER_FILE lines in $OUTPUT_DIR/"
done
