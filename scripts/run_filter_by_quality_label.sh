#!/bin/bash
#SBATCH --job-name=filter_data
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-29
#SBATCH -o ../logs/%A_%a.out
#SBATCH -e ../logs/%A_%a.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

# --------- CONFIGURATION ---------
LANG_ID="deu_Latn"
DATA_DIR="../data/hplt/line_quality_labelled/${LANG_ID}/full"
OUT_DIR="../data/hplt/line_quality_labelled/${LANG_ID}/just_clean_trimmed"
ORIGINAL_FILE_TYPE=".jsonl"
FILE_PREFIX="$1"

mkdir -p "$OUT_DIR"

# --------- FILE DISCOVERY ---------
# Build an array of matching files
file_array=()
while IFS= read -r -d '' file; do
    file_array+=("$file")
done < <(find "$DATA_DIR" -maxdepth 1 -type f -name "${FILE_PREFIX}*" -print0 | sort -z)

# --------- ARRAY TASK INDEX ---------
input_path="${file_array[$SLURM_ARRAY_TASK_ID]}"
filename="$(basename "$input_path" "$ORIGINAL_FILE_TYPE")"
output_path="$OUT_DIR/${filename}.jsonl"

echo "Task $SLURM_ARRAY_TASK_ID processing $input_path -> $output_path"

# --------- PROCESSING ---------
python3 ../src/filter_by_quality_label.py \
    --quality-labels="all" \
    --trim \
    --data-path="$input_path" \
    --save-path="$output_path"
