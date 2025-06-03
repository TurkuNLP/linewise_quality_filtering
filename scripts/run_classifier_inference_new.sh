#!/bin/bash -l
###############################################################################
#  Line-quality classifier – 8-GPU batch inference
###############################################################################
#SBATCH --job-name=classifier_inference
#SBATCH --account=project_462000353
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00

# one node, eight tasks, one GPU and four CPU cores per task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
###############################################################################

### 0.  helper that isolates the right GPU for every task #####################
cat << 'EOF' > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}
export HIP_VISIBLE_DEVICES=${SLURM_LOCALID}     # <- makes PyTorch/Accelerate happy
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID} 
exec "$@"
EOF
chmod +x select_gpu                                            # :contentReference[oaicite:0]{index=0}

### 1.  optional CPU affinity (one CCD per GPU) ###############################
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"   # LUMI-G default; change if needed

export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500 

### 2.  environment & modules #################################################
module purge
module use /appl/local/csc/modulefiles
module load pytorch
source ../.venv/bin/activate

### 3.  CLI parsing (unchanged) ###############################################
LANG_ID="spa_Latn"
DEFAULT_OUT_DIR="../data/hplt/line_quality_labelled/${LANG_ID}/full"
DEFAULT_MODEL="../models/line_quality_classifier_${LANG_ID}"

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

OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
MODEL="$(realpath "${MODEL:-$DEFAULT_MODEL}")"
[[ -d $MODEL ]] || { echo "❌ Model path not found: $MODEL"; exit 1; }
[[ -z $FILES_INPUT || -z $DATA_DIR ]] && {
    echo "Usage: $0 --input '*.jsonl' --data-dir DIR …"; exit 1; }

# expand file list
if [[ $FILES_INPUT == *'*'* ]]; then
    mapfile -t FILES < <(find "$DATA_DIR" -type f -name "$FILES_INPUT" -exec basename {} \;)
else
    IFS=',' read -ra FILES <<< "$FILES_INPUT"
fi
[[ ${#FILES[@]} -le 8 ]] || { echo "❌ More than 8 files supplied"; exit 1; }

mkdir -p "$OUT_DIR"
echo "▶ Launching ${#FILES[@]} parallel tasks on 8 GPUs…"

### 4.  pack everything into a single srun ####################################
# every rank gets its own filename via an array export
export FILES_STR="${FILES[*]}"

srun --cpu-bind=${CPU_BIND} ./select_gpu bash -c '
    IFS=" " read -ra ALL <<< "$FILES_STR"
    FNAME=${ALL[$SLURM_LOCALID]}                   # pick file that matches rank 0-7
    [[ -z $FNAME ]] && exit 0                      # idle task if <8 input files

    accelerate launch ../src/classifier_inference_new.py \
        --data-path  "'"$DATA_DIR"'"/"$FNAME"      \
        --save-path  "'"$OUT_DIR"'"/"$FNAME"       \
        --model-path "'"$MODEL"'" 
'

### 5.  energy log (optional) ##################################################
gpu-energy --diff
rm -f select_gpu
