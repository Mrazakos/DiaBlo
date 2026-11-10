#!/usr/bin/env bash
# One-time environment setup. Run on the login node (deep21) from the repo root:
#   bash scripts/slurm/setup_env.sh
# Package installs are light enough for the login node (see the cluster's guides/environment.md).
set -eo pipefail

CONDA_ENV="${CONDA_ENV:-diablo}"
WORK="${DIABLO_WORK:-$HOME/diablo}"

if [ ! -f requirements.txt ]; then
    echo "ERROR: run this from the DiaBlo repo root" >&2
    exit 1
fi

source /shared/anaconda/anaconda/bin/activate

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "conda env '$CONDA_ENV' already exists, reusing it"
else
    # conda-forge only: avoids the Terms-of-Service prompt of the default channels.
    conda create -n "$CONDA_ENV" -c conda-forge --override-channels python=3.12 -y
fi
conda activate "$CONDA_ENV"

# cu126 is the build that works on every GPU generation in the cluster.
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt pytest

mkdir -p "$WORK" logs
pip freeze > "$WORK/env_freeze.txt"

python - <<'EOF'
import torch, transformers, datasets, peft
print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda}) | transformers {transformers.__version__} "
      f"| datasets {datasets.__version__} | peft {peft.__version__}")
assert torch.version.cuda is not None, "CPU-only torch was installed"
EOF
echo "Environment ready. Versions saved to $WORK/env_freeze.txt"
