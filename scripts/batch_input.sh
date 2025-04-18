#!/bin/bash

PROCESS_BATCH="./run_classifier_inference_lumi.sh"

FILE_DIR="$1"
FILES=("$FILE_DIR"/*)

echo "Found ${#FILES[@]} files in $FILE_DIR"

total_files=${#FILES[@]}
batch_size=8  # Number of GPUs/files to process in parallel


# Calculate number of batches needed
num_batches=$(( (total_files + batch_size - 1) / batch_size ))

echo "Splitting data into $num_batches batches"

for ((batch=0; batch<num_batches; batch++)); do
    # Calculate start and end indices for this batch
    start=$((batch * batch_size))
    end=$((start + batch_size - 1))

    # Ensure we don't exceed the array bounds
    if [ $end -ge $total_files ]; then
        end=$((total_files - 1))
    fi

    # Create a comma-separated list of files for this batch
    file_list=""
    echo "Batch $batch would process:"
    for ((i=start; i<=end; i++)); do
        echo "  - ${files[i]} (would use GPU $((i-start)))"
        if [ -n "$file_list" ]; then
            file_list="${file_list},"
        fi
        file_list="${file_list}${files[i]}"
    done

    sbatch "$PROCESS_BATCH" "$FILE_DIR" "$file_list"
    echo "Submitted batch $batch (files $start to $end)"
    
done