#!/bin/bash

# Use this script to batch a bunch of files in FILE_DIR
# and the pass the batch onto another script.

# Usage: bash batch_input path/to/file/dir

# Where to pass the batch to
PROCESS_BATCH="./run_classifier_inference_new.sh"

# Size of batch. Make sure the receiving script can handle the batch size
batch_size=8

FILE_DIR="$1"
FILES=("$FILE_DIR"/*)  # Full paths
BASENAMES=($(printf "%s\n" "${FILES[@]}" | xargs -n1 basename))  # Just file names

# Convert to comma-separated string for quick display
FILE_LIST=$(printf "%s\n" "${BASENAMES[@]}" | paste -sd, -)
echo "Found ${#BASENAMES[@]} files in $FILE_DIR"

total_files=${#BASENAMES[@]}

# Calculate number of batches needed
num_batches=$(( (total_files + batch_size - 1) / batch_size ))

echo "Splitting data into $num_batches batches"

for ((batch=0; batch<num_batches; batch++)); do
    # Calculate start and end indices for this batch
    start=$((batch * batch_size))
    end=$((start + batch_size - 1))
    if [ $end -ge $total_files ]; then
        end=$((total_files - 1))
    fi

    # Collect filenames for this batch
    file_list=""
    echo "Batch $batch would process:"
    for ((i=start; i<=end; i++)); do
        echo "  - ${BASENAMES[i]}"
        if [ -n "$file_list" ]; then
            file_list="${file_list},"
        fi
        file_list="${file_list}${BASENAMES[i]}"
    done

    # Submit batch
    sbatch "$PROCESS_BATCH" --input "$file_list" --data-dir "$FILE_DIR"
    echo "Submitted batch $batch (files $start to $end)"
done
