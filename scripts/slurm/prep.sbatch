#!/usr/bin/env bash
# CPU-only preparation: no GPU requested, so it costs 0 GPU-hours.
#   sbatch scripts/slurm/prep.sbatch          (from the repo root)
# If compute nodes turn out to have no internet, run it on deep21 instead:
#   bash scripts/slurm/prep.sbatch
#
#SBATCH --job-name=prep
#SBATCH --account=dl_2026_fall
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

echo "prep started on $(hostname) at $(date)"
source scripts/slurm/env.sh || exit 1
set -eo pipefail

# 1. Internet check (model + dataset come from the HF Hub)
if ! python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=20)"; then
    echo "ERROR: no internet on $(hostname). Run 'bash scripts/slurm/prep.sbatch' on deep21 instead." >&2
    exit 1
fi

# 2. Base model into the shared HF cache (GPU jobs then run offline)
python - <<EOF
from huggingface_hub import snapshot_download
print("model cached at", snapshot_download("$HF_ID"))
EOF

# 3. Tokenize GSM8k for this model (once)
if [ ! -d "$DATA/GSM8k_processed/train" ] || [ ! -d "$DATA/GSM8k_raw" ]; then
    python data_processing/process_gsm8k.py --model_name "$HF_ID" --output_dir "$DATA" --max_length 512
fi
python - <<EOF
from datasets import load_from_disk
tr = load_from_disk("$DATA/GSM8k_processed/train")
lens = [len(x) for x in tr["input_ids"]]
print(f"{len(tr)} train examples | max len {max(lens)} | mean {sum(lens)/len(lens):.1f}")
print(len(load_from_disk("$DATA/GSM8k_raw")["test"]), "test questions")
EOF

# 4. Unit tests + tiny end-to-end pipeline (train -> resume -> merge -> eval, DiaBlo and LoRA) on CPU.
#    Catches library-version problems before any GPU budget is spent.
python -m pytest -q tests
python tests/e2e_tiny.py --workdir "$WORK/e2e" --precision fp32

echo "PREP OK at $(date)"
