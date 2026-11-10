#!/usr/bin/env bash
# Submit the run matrix in priority order. Run on deep21 from the repo root:
#   TIME=02:00:00 bash scripts/slurm/submit_all.sh              # everything
#   TIME=02:00:00 bash scripts/slurm/submit_all.sh R1 R2 R3     # only these
# Safe to run again at any time: finished runs and runs already in the queue are skipped,
# unfinished ones resume from their last checkpoint.
#
# Settings (environment variables):
#   TIME   --time per job, HH:MM:SS (take it from the smoke test; default is run.sbatch's 03:00:00)
#   BS GA  batch size x accumulation for ALL runs (default 4 x 4; use BS=2 GA=8 if the smoke test says so)
#   GPU    GPU type (default rtx2080ti). Keep it the same for every run.
set -eo pipefail

ORDER=(R0 R1 R2 R3 R8 R4 R9 R10 R7 R11 R12 R5a R5b)

if [ ! -f train.py ]; then
    echo "ERROR: run this from the DiaBlo repo root" >&2
    exit 1
fi
EVALS="${DIABLO_WORK:-$HOME/diablo}/eval"
mkdir -p logs

IDS=("$@")
[ ${#IDS[@]} -eq 0 ] && IDS=("${ORDER[@]}")

[ -n "${BS:-}" ] && export BS
[ -n "${GA:-}" ] && export GA
queued="$(squeue --me -h -o %j)"

for id in "${IDS[@]}"; do
    if [ -f "$EVALS/${id}_hf_summary.json" ]; then
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
