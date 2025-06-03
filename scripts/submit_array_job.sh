#!/bin/bash

LANG_ID="$1"
DATA_DIR="../data/hplt/${LANG_ID}"

# Count number of .jsonl files
NUM_FILES=$(find "$DATA_DIR" -maxdepth 1 -name '*.jsonl' | wc -l)

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No .jsonl files found in $DATA_DIR"
    exit 1
fi

# Submit job with dynamic array size
echo "Submitting SLURM job with --array=0-$((NUM_FILES - 1))"
sbatch --array=0-$((NUM_FILES - 1)) run_classifier_inference.sh "$LANG_ID"

