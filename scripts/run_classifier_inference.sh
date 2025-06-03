#!/bin/bash
#SBATCH --job-name=classifier_inference
#SBATCH --account=project_462000353
#SBATCH --partition=small-g
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=20G
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err
#SBATCH --array=1-20

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
DATA_DIR="${BASEDIR}/../data/hplt/${LANG_ID}"
OUT_DIR="${BASEDIR}/../data/hplt/line_quality_labelled/${LANG_ID}/full"
MODEL="${BASEDIR}/../models/line_quality_classifier_${LANG_ID}"

mkdir -p "$OUT_DIR"

# Get the list of .jsonl files into an array
mapfile -t FILES < <(find "$DATA_DIR" -maxdepth 1 -name '*.jsonl' | sort)

# Select files using the array task ID
INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
BASENAME=$(basename "$INPUT_FILE")

echo "Processing file: $INPUT_FILE"
echo "Saving to: ${OUT_DIR}/${BASENAME}"

python3 ../src/classifier_inference_pipeline.py \
    --data-path "$INPUT_FILE" \
    --save-path "${OUT_DIR}/${BASENAME}" \
    --model-path "${MODEL}"

###gpu-energy --diff