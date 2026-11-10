#!/usr/bin/env bash
# GPU smoke test: run this ONCE before the real runs (costs ~10 GPU-minutes).
#   sbatch scripts/slurm/smoke.sbatch          (from the repo root, after prep)
# It answers two questions and prints them as "SMOKE ..." lines at the end of the log:
#   1. does batch 4 fit in this GPU's memory (else batch 2 x accumulation 8)?
#   2. how long does a full run take (-> the --time to request, and the total budget)?
#
#SBATCH --job-name=smoke
#SBATCH --account=dl_2026_fall
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out

echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"
source scripts/slurm/env.sh || exit 1
set -eo pipefail
export HF_HUB_OFFLINE=1          # model and data were cached by prep.sbatch

SMOKE="${WORK:?}/smoke"
rm -rf "$SMOKE" && mkdir -p "$SMOKE"

nvidia-smi

# ── 1. Worst-case memory ─────────────────────────────────────────────────────
python scripts/slurm/mem_probe.py --model "$HF_ID" --data_dir "$DATA" | tee "$SMOKE/probe.txt"
if [ -z "${BS:-}" ]; then
    if grep -q "^PROBE batch_size=4: .*OK$" "$SMOKE/probe.txt"; then BS=4; GA=4; else BS=2; GA=8; fi
fi
GA="${GA:-$((16 / BS))}"
echo "Using per_device_batch_size=$BS gradient_accumulation_steps=$GA"

# ── 2. Real-model fp16 training: 20 steps, then resume to 30 ─────────────────
common=(--model_name "$MODEL" --task GSM8k --data_dir "$DATA" --output_dir "$SMOKE" --precision fp16
        --per_device_batch_size "$BS" --gradient_accumulation_steps "$GA" --save_steps 10 --warmup_steps 5
        --method diablo --num_blocks 64 --lr 5e-4 --run_name smoke_diablo)
python train.py "${common[@]}" --max_steps 20
python train.py "${common[@]}" --max_steps 30 --resume | tee "$SMOKE/resume.txt"
grep -q "Resumed adapter weights" "$SMOKE/resume.txt" || { echo "ERROR: resume did not load the adapter" >&2; exit 1; }

# ── 3. Eval timing: 1 batch, then 2 more batches (difference = seconds per batch) ──
ev=(--model_name "$MODEL" --method diablo --num_blocks 64 --target_modules qkvud
    --adapter_path "$SMOKE/smoke_diablo/adapter.chkpt" --data_dir "$DATA" --precision fp16
    --batch_size 32 --output_file "$SMOKE/eval_smoke.jsonl")
t0=$(date +%s)
python evaluate_gsm8k.py "${ev[@]}" --limit 32
t1=$(date +%s)
python evaluate_gsm8k.py "${ev[@]}" --limit 96
t2=$(date +%s)

# ── 4. Summary ───────────────────────────────────────────────────────────────
python - <<EOF
import json, math
smoke = "$SMOKE/smoke_diablo"
lh = json.load(open(f"{smoke}/log_history.json"))
st = json.load(open(f"{smoke}/checkpoint-30/trainer_state.json"))
losses = [h["loss"] for h in st["log_history"] if "loss" in h]
assert losses and all(l == l for l in losses), f"NaN loss in smoke run: {losses}"

sps = lh[-1]["train_runtime"] / 10            # the resumed run covers exactly steps 21-30
steps = 1401                                  # 7473 examples / 16 per step * 3 epochs
train_h = steps * sps / 3600

T1, T2 = $t1 - $t0, $t2 - $t1                 # load + 1 batch, load + 2 batches
per_batch = max(T2 - T1, 1)
load = max(T1 - per_batch, 0)
eval_h = (load + math.ceil(1319 / 32) * per_batch) / 3600

run_h = train_h + eval_h
req_min = math.ceil(run_h * 1.25 * 60 / 10) * 10   # +25%, rounded up to 10 minutes
total = 12 * run_h + eval_h

print()
print(f"SMOKE losses (every 10 steps): {losses}")
print(f"SMOKE batch: per_device_batch_size=$BS gradient_accumulation_steps=$GA")
print(f"SMOKE train: {sps:.2f} s/step -> {train_h:.2f} h per run ({steps} steps)")
print(f"SMOKE eval : model load {load} s, {per_batch} s per 32-question batch -> {eval_h * 60:.0f} min per run (upper bound)")
print(f"SMOKE per run: {run_h:.2f} h -> request TIME={req_min // 60:02d}:{req_min % 60:02d}:00")
print(f"SMOKE whole matrix (12 runs + base eval): about {total:.1f} of 20 GPU-hours")
EOF
echo "SMOKE OK at $(date)"
