#!/usr/bin/env bash
# Shared settings for every cluster script. Source it, don't run it:
#   source scripts/slurm/env.sh
# Override any of these by exporting them before sbatch / bash.

CONDA_ENV="${CONDA_ENV:-diablo}"
source /shared/anaconda/anaconda/bin/activate
conda activate "$CONDA_ENV"

export MODEL="${MODEL:-Qwen2.5-1.5B}"            # short name in diablo/common.py MODEL_MAP
export HF_ID="${HF_ID:-Qwen/Qwen2.5-1.5B}"

# Everything persistent lives here (same layout as MyDrive/diablo in the Colab notebooks).
export WORK="${DIABLO_WORK:-$HOME/diablo}"
export DATA="$WORK/datasets/$MODEL"              # tokenized data is tokenizer-specific
export RESULTS="$WORK/results"
export EVALS="$WORK/eval"
export HF_HOME="$WORK/hf_cache"

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation, no effect on results
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$DATA" "$RESULTS" "$EVALS"

# Jobs must be submitted from the repo root (sbatch copies the script, so its own path is useless).
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f train.py ]; then
    echo "ERROR: run this from the DiaBlo repo root (train.py not found in $PWD)" >&2
    return 1 2>/dev/null || exit 1
fi
