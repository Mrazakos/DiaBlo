#!/usr/bin/env bash
# One reproduction run = train (resumable) + GSM8k eval (resumable), as a single Slurm job.
#   sbatch -J R1 --time=02:00:00 scripts/slurm/run.sbatch R1      (from the repo root)
# or submit the whole matrix with scripts/slurm/submit_all.sh.
# If the job hits its time limit or fails, submit the same command again: it continues
# from the last checkpoint (saved every 100 steps) or from the last evaluated batch.
#
# Environment overrides: BS / GA (batch size x accumulation, default 2 x GA from runs.sh), EVAL_BS (default 32).
#
#SBATCH --job-name=diablo-run
#SBATCH --account=dl_2026_fall
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

RUN_ID="${1:?usage: sbatch -J <ID> scripts/slurm/run.sbatch <ID>   (ids: see scripts/slurm/runs.sh)}"

# Run table lives in runs.sh (shared with submit_all.sh). It decides MODEL, so load it before env.sh.
source scripts/slurm/runs.sh
load_run "$RUN_ID" || exit 2
export MODEL
BS="$RUN_BS"; GA="$RUN_GA"
EVAL_BS="${EVAL_BS:-32}"
QUANT=(); [ "$LOAD4" = 1 ] && QUANT=(--load_in_4bit)

echo "Job ${SLURM_JOB_ID:-local} ($RUN_ID) started on $(hostname) at $(date)"
echo "GPUs visible to this job: ${CUDA_VISIBLE_DEVICES:-none}"
source scripts/slurm/env.sh || exit 1
set -eo pipefail
export HF_HUB_OFFLINE=1          # model and data were cached by prep.sbatch
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RUN_DIR="$RESULTS/$RUN_ID"
EVAL_OUT="$EVALS/${RUN_ID}_hf.jsonl"
SIZE_FLAG=()
[ "$METHOD" = diablo ] && SIZE_FLAG=(--num_blocks "$SIZE")
[ "$METHOD" = lora ]   && SIZE_FLAG=(--lora_r "$SIZE")

# ── Train ────────────────────────────────────────────────────────────────────
if [ "$METHOD" = none ]; then
    echo "$RUN_ID: base model, nothing to train"
elif [ -f "$RUN_DIR/log_history.json" ] && [ -f "$RUN_DIR/adapter.chkpt" ]; then
    echo "$RUN_ID: training already finished, skipping to eval"
else
    mkdir -p "$RUN_DIR"
    if [ -f "$RUN_DIR/run_config.json" ]; then
        # Resuming: refuse to continue a run with different settings than it was started with.
        python - <<EOF
import json, sys
have = json.load(open("$RUN_DIR/run_config.json"))
want = dict(method="$METHOD", lr=float("$LR"), seed=$SEED, max_epochs=$EPOCHS, warmup_steps=$WARMUP,
            target_modules="$MODULES", per_device_batch_size=$BS, gradient_accumulation_steps=$GA)
bad = {k: f"started with {have.get(k)!r}, now {v!r}" for k, v in want.items() if have.get(k) != v}
if bad:
    sys.exit(f"ERROR: $RUN_ID was started with different settings: {bad}. "
             f"Use the same BS/GA, or delete $RUN_DIR to restart it.")
EOF
    fi
    python train.py --model_name "$MODEL" --task GSM8k --data_dir "$DATA" \
        --output_dir "$RESULTS" --run_name "$RUN_ID" --resume \
        --method "$METHOD" "${SIZE_FLAG[@]}" --lr "$LR" --seed "$SEED" \
        --per_device_batch_size "$BS" --gradient_accumulation_steps "$GA" \
        --max_epochs 3 --warmup_steps 100 --dropout 0.05 --target_modules qkvud \
        --precision fp16 --save_steps 100 2>&1 | tee -a "$RUN_DIR/train.log"
fi

# ── Evaluate (1319 GSM8k test questions, greedy, 512 new tokens) ─────────────
if [ -f "${EVAL_OUT%.jsonl}_summary.json" ]; then
    echo "$RUN_ID: eval already finished"
elif [ "$METHOD" = none ]; then
    python evaluate_gsm8k.py --model_name "$MODEL" --method none --data_dir "$DATA" \
        --precision fp16 --batch_size "$EVAL_BS" --output_file "$EVAL_OUT"
else
    python evaluate_gsm8k.py --model_name "$MODEL" --method "$METHOD" "${SIZE_FLAG[@]}" \
        --target_modules qkvud --adapter_path "$RUN_DIR/adapter.chkpt" --data_dir "$DATA" \
        --precision fp16 --batch_size "$EVAL_BS" --output_file "$EVAL_OUT"
fi

echo "RUN $RUN_ID DONE at $(date)"
cat "${EVAL_OUT%.jsonl}_summary.json"
