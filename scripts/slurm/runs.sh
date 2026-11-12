#!/usr/bin/env bash
# Run table, sourced by run.sbatch and submit_all.sh (single source of truth).
#   source scripts/slurm/runs.sh; load_run R1     -> sets MODEL METHOD SIZE LR SEED EPOCHS WARMUP MODULES RUN_BS RUN_GA LOAD4
#
# Groups:
#   R*  first wave: Qwen2.5-1.5B with the repo's train_gsm8k.sh recipe (3 epochs, batch 16, warmup 100, qkvud)
#   P*  paper recipe (Table 20: 2 epochs, batch 32, lr 4e-4 DiaBlo, warmup 20, all 7 modules) on Qwen2.5-1.5B
#   S*  weaker model Qwen2.5-0.5B, paper recipe
#   L*/H*  Llama-family in 4-bit (paper's model family). NOT in the default order: they need more GPU hours,
#          gated HF access, and probably GPU=a5000 and a larger --mem. Run scripts/slurm/smoke.sbatch's memory probe first.
# Effective batch = BS x GA. BS=2 because batch 4 does not fit the 10.6 GB RTX 2080 Ti. BS/GA from the environment win.

DEFAULT_ORDER=(R0 R1 R2 R3 R8 R4 R9 R10 R7 R11 R12 R5a R5b)
PAPER_ORDER=(P1 P2 S0 S1 S2)
LLAMA_ORDER=(L0 L1 L2 H1 H2)

load_run() {
    local id="$1"
    MODEL=Qwen2.5-1.5B; METHOD=; SIZE=; LR=; SEED=42; EPOCHS=3; WARMUP=100; MODULES=qkvud; GA_DEF=8; LOAD4=0
    case "$id" in
        R0)  METHOD=none ;;                                   # base model, eval only
        R1)  METHOD=diablo; SIZE=64;  LR=5e-4 ;;
        R2)  METHOD=lora;   SIZE=16;  LR=2e-4 ;;
        R3)  METHOD=diablo; SIZE=32;  LR=5e-4 ;;
        R4)  METHOD=diablo; SIZE=128; LR=5e-4 ;;
        R5a) METHOD=diablo; SIZE=64;  LR=5e-4; SEED=1 ;;
        R5b) METHOD=lora;   SIZE=16;  LR=2e-4; SEED=1 ;;
        R7)  METHOD=diablo; SIZE=64;  LR=2e-4 ;;
        R8)  METHOD=lora;   SIZE=32;  LR=2e-4 ;;
        R9)  METHOD=diablo; SIZE=512; LR=5e-4 ;;
        R10) METHOD=diablo; SIZE=16;  LR=5e-4 ;;
        R11) METHOD=diablo; SIZE=64;  LR=1e-4 ;;
        R12) METHOD=diablo; SIZE=64;  LR=1e-3 ;;
        # paper recipe on the same model (LoRA keeps the first-wave lr 2e-4: the paper gives no LoRA lr for this setup)
        P1)  METHOD=diablo; SIZE=64;  LR=4e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; GA_DEF=16 ;;
        P2)  METHOD=lora;   SIZE=16;  LR=2e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; GA_DEF=16 ;;
        # weaker model
        S0)  MODEL=Qwen2.5-0.5B; METHOD=none ;;
        S1)  MODEL=Qwen2.5-0.5B; METHOD=diablo; SIZE=64; LR=4e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; GA_DEF=16 ;;
        S2)  MODEL=Qwen2.5-0.5B; METHOD=lora;   SIZE=16; LR=2e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; GA_DEF=16 ;;
        # Llama family, 4-bit NF4 base (paper's models; batch size is a guess until probed: override BS/GA)
        L0)  MODEL=Llama3-3B; METHOD=none; LOAD4=1 ;;
        L1)  MODEL=Llama3-3B; METHOD=diablo; SIZE=64; LR=4e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; GA_DEF=16; LOAD4=1 ;;
        L2)  MODEL=Llama3-3B; METHOD=lora;   SIZE=16; LR=2e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; GA_DEF=16; LOAD4=1 ;;
        H1)  MODEL=Llama3-8B; METHOD=diablo; SIZE=64; LR=4e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; GA_DEF=16; LOAD4=1 ;;
        H2)  MODEL=Llama3-8B; METHOD=lora;   SIZE=16; LR=2e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; GA_DEF=16; LOAD4=1 ;;
        *)   echo "ERROR: unknown run id '$id'" >&2; return 2 ;;
    esac
    RUN_BS="${BS:-2}"          # not written back to BS/GA, so one shell can load several runs
    RUN_GA="${GA:-$GA_DEF}"
}

# Where a run's eval summary lives (same rule as env.sh), without sourcing conda.
eval_summary_path() {
    local id="$1" work="${DIABLO_WORK:-$HOME/diablo}"
    load_run "$id" || return 2
    if [ "$MODEL" = "Qwen2.5-1.5B" ]; then echo "$work/eval/${id}_hf_summary.json"
    else echo "$work/eval-$MODEL/${id}_hf_summary.json"; fi
}
