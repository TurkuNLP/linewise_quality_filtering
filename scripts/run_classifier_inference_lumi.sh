#!/bin/bash
#SBATCH --job-name=classifier_inference
#SBATCH --account=project_462000353
#SBATCH --partition=standard-g
#SBATCH --time=23:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:mi250:8
#SBATCH -o ../logs/%j.out
#SBATCH -e ../logs/%j.err


################################################################################
# Line Quality Classifier - Batch Inference Script
# Used to run inference on a finetuned line-quality classifier.

# Takes at most 8 files at a time!

# Supports: comma-separated filenames or shell wildcard (e.g. *.jsonl)
################################################################################

# === Load Required Modules ===
module purge
module use /appl/local/csc/modulefiles
module load pytorch

# === Activate Python Virtual Environment ===
source ../.venv/bin/activate

# === Energy Logging (Optional) ===
gpu-energy --save

# === Argument Parsing and Validation ===
# Either change the default values to suit your use case
# or give appropriate values as command-line arguments
# If you just want to change language but all other paths are okay,
# you can just modify the LANG_ID variable
LANG_ID="fra_Latn"
DEFAULT_DATA_DIR="../data/fineweb2/fineweb2-${LANG_ID}"
DEFAULT_OUT_DIR="../data/fineweb2_${LANG_ID}_line_quality_labelled/full"
DEFAULT_MODEL="../results/finetuned_models/line_quality_classifier_${LANG_ID}"

# === Parse Keyword-style Arguments ===
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input) FILES_INPUT="$2"; shift ;;
        --data-dir) DATA_DIR="$2"; shift ;;
        --output-dir) OUT_DIR="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --help)
            echo "Usage: $0 --input '*.jsonl' --data-dir DIR [--output-dir DIR] [--model PATH]"
            exit 0 ;;
        *) echo "❌ Unknown argument: $1"; exit 1 ;;
    esac
    shift
done


DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
MODEL="${MODEL:-$DEFAULT_MODEL}"

# === Normalize model path to absolute ===
MODEL=$(realpath "$MODEL")
if [[ ! -d "$MODEL" ]]; then
    echo "❌ ERROR: Model path does not exist or is not a directory: $MODEL"
    exit 1
fi


if [[ -z "$FILES_INPUT" || -z "$DATA_DIR" ]]; then
    echo "Usage: $0 --input '*.jsonl' --data-dir DIR [--output-dir DIR] [--model PATH]"
    exit 1
fi

# === File List Expansion ===
if [[ "$FILES_INPUT" == *'*'* ]]; then
    echo "Expanding wildcard pattern: $FILES_INPUT"
    mapfile -t FILES < <(find "$DATA_DIR" -type f -name "$FILES_INPUT" -exec basename {} \;)
else
    IFS=',' read -ra FILES <<< "$FILES_INPUT"
fi

# === Summary ===
echo "Starting inference on ${#FILES[@]} files..."
echo "Input directory: $DATA_DIR"
echo "Output directory: $OUT_DIR"
mkdir -p "$OUT_DIR"

# === Inference Loop ===
for FILE_NAME in "${FILES[@]}"; do
    BASENAME=$(basename "$FILE_NAME")
    INPUT_PATH="$DATA_DIR/$BASENAME"
    OUTPUT_PATH="$OUT_DIR/$BASENAME"
    srun \
        --ntasks=1 \
        --gres=gpu:mi250:1 \
        --mem=20G \
        accelerate launch ../src/classifier_inference.py \
        --data-path "$INPUT_PATH" \
        --save-path "$OUTPUT_PATH" \
        --model-path "$MODEL" \
        &
done

# === Wait for All Inference Jobs ===
wait

# === Show GPU Energy Usage ===
gpu-energy --diff
