#!/usr/bin/env bash
# Submit the run matrix in priority order. Run on deep21 from the repo root:
#   TIME=02:00:00 bash scripts/slurm/submit_all.sh              # everything
#   TIME=02:00:00 bash scripts/slurm/submit_all.sh R1 R2 R3     # only these
#   TIME=02:00:00 bash scripts/slurm/submit_all.sh paper        # P1 P2 S0 S1 S2 (paper recipe + weaker model)
# Safe to run again at any time: finished runs and runs already in the queue are skipped,
# unfinished ones resume from their last checkpoint.
#
# Settings (environment variables):
#   TIME   --time per job, HH:MM:SS (take it from the smoke test; default is run.sbatch's 03:00:00)
#   BS GA  batch size x accumulation for ALL runs (default 2 x per-run GA from runs.sh)
#   GPU    GPU type (default rtx2080ti). Keep it the same for every run.
set -eo pipefail

source scripts/slurm/runs.sh   # DEFAULT_ORDER, PAPER_ORDER, LLAMA_ORDER, load_run

if [ ! -f train.py ]; then
    echo "ERROR: run this from the DiaBlo repo root" >&2
    exit 1
fi
mkdir -p logs

IDS=("$@")
# Group names work as arguments: first | paper | llama (default: first)
expand=()
for a in "${IDS[@]}"; do
    case "$a" in
        first) expand+=("${DEFAULT_ORDER[@]}") ;;
        paper) expand+=("${PAPER_ORDER[@]}") ;;
        llama) expand+=("${LLAMA_ORDER[@]}") ;;
        *)     expand+=("$a") ;;
    esac
done
IDS=("${expand[@]}")
[ ${#IDS[@]} -eq 0 ] && IDS=("${DEFAULT_ORDER[@]}")

[ -n "${BS:-}" ] && export BS
[ -n "${GA:-}" ] && export GA
queued="$(squeue --me -h -o %j)"

for id in "${IDS[@]}"; do
    if [ -f "$(eval_summary_path "$id")" ]; then
        echo "$id: finished, skipping"
        continue
    fi
    if grep -qx "$id" <<<"$queued"; then
        echo "$id: already in the queue, skipping"
        continue
    fi
    opts=(--parsable --export=ALL -J "$id")
    [ -n "${TIME:-}" ] && opts+=(--time="$TIME")
    [ -n "${GPU:-}" ] && opts+=(--gres="gpu:${GPU}:1")
    jid="$(sbatch "${opts[@]}" scripts/slurm/run.sbatch "$id")"
    echo "$id: submitted as job $jid  (log: logs/${id}-${jid}.out)"
done

echo
squeue --me
