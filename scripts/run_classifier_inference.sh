#!/bin/bash
#SBATCH --job-name=classifier_inference
#SBATCH --account=project_462000353
#SBATCH --partition=small-g
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=20G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
#SBATCH --array=1-5

####gpu-energy --save

# === Load Required Modules ===
module purge
module use /appl/local/csc/modulefiles
module load pytorch

# === Activate Python Virtual Environment ===
source ../.venv/bin/activate

# Print the task index.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "ROCR visible devices: " $ROCR_VISIBLE_DEVICES

LANG_ID="$1"
BASEDIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="${BASEDIR}/../data/hplt_dedup/${LANG_ID}/splits"
OUT_DIR="${BASEDIR}/../data/hplt_dedup_salvaged/${LANG_ID}/full"
MODEL="${BASEDIR}/../models/line_quality_classifier_${LANG_ID}"

echo "================================"
echo "Reading data from $DATA_DIR"
echo "Results will be saved in $OUT_DIR"
echo "Using model stored at $MODEL"
echo "================================"

mkdir -p "$OUT_DIR"

# Get the list of both .jsonl and .jsonl.zst files into an array
mapfile -t FILES < <(find "$DATA_DIR" -maxdepth 1 \( -name '*.jsonl' -o -name '*.jsonl.zst' \) | sort)

# Debug: show files found
echo "Found ${#FILES[@]} files:"
for f in "${FILES[@]}"; do
    echo "  - $f"
done

# Adjust index for 0-based arrays (SLURM is 1-based, bash arrays are 0-based)
ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Verify we have a valid index
if [ "$ARRAY_INDEX" -lt 0 ] || [ "$ARRAY_INDEX" -ge "${#FILES[@]}" ]; then
    echo "Error: ARRAY_INDEX ($ARRAY_INDEX) is out of range. Only ${#FILES[@]} files found."
    exit 1
fi

# Select files using the array task ID (adjusted for 0-based indexing)
INPUT_FILE="${FILES[$ARRAY_INDEX]}"
BASENAME=$(basename "$INPUT_FILE")

echo "Processing file: $INPUT_FILE"
echo "Saving to: ${OUT_DIR}/${BASENAME}"

python3 ../src/classifier_inference_pipeline.py \
    --data-path "$INPUT_FILE" \
    --save-path "${OUT_DIR}/${BASENAME}" \
    --model-path "${MODEL}"

###gpu-energy --diff