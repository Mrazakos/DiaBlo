# DiaBlo reproduction report

> **Status: team draft (2026-11-24).** Produced with AI assistance (Claude Code) from the run outputs listed in
> [Provenance](#provenance). Each number below was taken from those files; the team should spot-check them before presenting.

**Paper:** *DiaBlo: Diagonal Blocks Are Sufficient For Finetuning*, arXiv:2506.03230 (ICLR 2026).
**Code:** authors' repo (upstream `83ffb6c`), our fork, branch `paper-repro`. How to rerun: [REPRODUCTION.md](REPRODUCTION.md).

## TL;DR (one slide)

| What we checked | Paper | Ours | Verdict |
|---|---|---|---|
| Llama3-8B, Commonsense-170k, DiaBlo N=64 (Table 1, headline) | 87.3 | **87.15** | Reproduced (−0.15; SE ≈ 0.31) |
| Llama3.2-3B, Commonsense-170k, DiaBlo N=64 (Table 10) | 84.44 | **84.45** | Reproduced (+0.01; SE 0.34) |
| Llama3-8B, GSM8K fine-tuning, DiaBlo 1/64 (Table 5) | 67.68 | **65.96** | Close, slightly lower (−1.72; SE 1.30) |
| Accuracy falls as N grows: N=512 / N=1024 (Table 10) | 82.54 / 81.60 | **80.10 / 77.50** | Trend reproduced, drop ~2× steeper |
| DiaBlo vs LoRA, same pipeline (3B, ~29M vs ~33M params) | no 3B number | **84.45 vs 83.90** (best of 3 LoRA LRs) | DiaBlo ahead by +0.55 (95% CI +0.01 to +1.20): small, borderline |
| Memory and speed vs LoRA (same GPU model) | "LoRA-like" | peak 9.88 vs 9.96 GB; 1.73 vs 1.43–1.73 s/step | Reproduced |

**Bottom line:** DiaBlo's own accuracy numbers reproduce almost exactly on the paper's models and data, and it costs the same
memory as LoRA, at about the same speed. Its advantage over LoRA is real but **much smaller** in our same-pipeline test (+0.55 points) than the gap
in the paper's Table 1 (DiaBlo 87.3 vs LoRA 80.8 / DoRA 85.2 on Llama3-8B). Those baseline numbers come from a different
pipeline, and LoRA turned out to be very sensitive to its learning rate (61.54 → 83.90 across the three rates we tried).

## 1. Goal and the paper's claims

The paper's central claims, and which runs test them:

| # | Claim (paper) | Runs |
|---|---|---|
| C1 | Fine-tuning only diagonal blocks reaches the accuracy the paper reports (Table 1: Llama3-8B 87.3) | C3 |
| C2 | DiaBlo matches or beats LoRA and DoRA (Table 1: LoRA 80.8, DoRA 85.2 on Llama3-8B) | A64 vs AL32 / AL32a / AL32b |
| C3 | Memory and speed similar to LoRA | train_stats of A64 vs the LoRA runs |
| C4 | Too many (too small) blocks hurt: Table 10 drops from N=64 to 512 and 1024 | A64, A512, A1024 |
| C5 | Works for math fine-tuning too (Table 5: 67.68 on GSM8K) | G1 |

The scope was a **minimum viable reproduction**: the smallest set of runs that each have a published number to match, within
1000 GPU-hours on a student cluster and about two weeks of wall clock.

## 2. Setup

**Hardware.** SZTE Deep Cluster (Slurm). 8B runs on the one NVIDIA RTX A5000 (24 GB), the only card that fits an 8B model in
16-bit. 3B runs on RTX 2080 Ti (11 GB, no bf16). The paper used one A100-80GB. Limits: 2 running and 10 queued jobs per user,
24 h per job, so long runs are chained 24 h segments that resume from checkpoints.

**Runs** (single seed 42 each; paper hyper-parameters from its appendix unless noted under Deviations):

| Run | Model | Data | Method | Trainable params | LR | Epochs | Eff. batch | Max len | Precision | GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| C3 | Llama3-8B | Commonsense-170k | DiaBlo N=64, q k v up down | – | 5e-5 | 3 | 16 | 256 | fp16 base + fp32 adapter | A5000 |
| G1 | Llama3-8B | GSM8K train | DiaBlo N=64, q k v o gate up down | – | 4e-4 | 2 | 32 | 512 | bf16 | A5000 |
| A64 | Llama3.2-3B | Commonsense-170k | DiaBlo N=64 | 28,901,376 | 1e-4 | 3 | 16 | 256 | fp16 + fp32 adapter | 2080 Ti |
| A512 | Llama3.2-3B | Commonsense-170k | DiaBlo N=512 | 3,612,672 | 1e-4 | 3 | 16 | 256 | fp16 + fp32 adapter | 2080 Ti |
| A1024 | Llama3.2-3B | Commonsense-170k | DiaBlo N=1024 | 1,806,336 | 1e-4 | 3 | 16 | 256 | fp16 + fp32 adapter | 2080 Ti |
| AL32 | Llama3.2-3B | Commonsense-170k | LoRA r=32, α=64 | 33,030,144 (1.02%) | 3e-4 | 3 | 16 | 256 | fp16 + fp32 adapter | 2080 Ti |
| AL32a | Llama3.2-3B | Commonsense-170k | LoRA r=32, α=64 | 33,030,144 | 1e-4 | 3 | 16 | 256 | fp16 + fp32 adapter | 2080 Ti |
| AL32b | Llama3.2-3B | Commonsense-170k | LoRA r=32, α=64 | 33,030,144 | 2e-4 | 3 | 16 | 256 | fp16 + fp32 adapter | 2080 Ti |

DiaBlo parameter counts are N=64's Table 10 count (checked in `tests/test_diablo.py`), which scales as 1/N. The LoRA count is
from the training log. All 3B runs share everything except the adapter and LR; 31,932 optimizer steps each.

**Evaluation.** Commonsense: the 8 LLM-Adapters test sets (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA;
22,419 questions in total), beam search 4, accuracy averaged over the 8 tasks. GSM8K: greedy decoding on all 1319 test questions.

**Statistics.** SE = binomial standard error of the 8-task average. Run-vs-run comparisons are **paired** (same questions):
a bootstrap over questions within each task (1000 resamples) and a pooled McNemar test.

## 3. Results

### 3.1 Llama3-8B (paper Tables 1 and 5)

| Run | BoolQ | PIQA | SIQA | HellaS. | WinoG. | ARC-e | ARC-c | OBQA | **Avg** | Paper |
|---|---|---|---|---|---|---|---|---|---|---|
| C3 DiaBlo N=64 | 75.84 | 89.93 | 81.42 | 95.62 | 89.50 | 93.35 | 83.53 | 88.00 | **87.15** | 87.3 |

Paper's Table 1 baselines on Llama3-8B (not rerun by us): LoRA 80.8, DoRA 85.2.

| Run | GSM8K (1319 q.) | Original protocol (first 1312 q.) | Paper |
|---|---|---|---|
| G1 DiaBlo 1/64 | **65.96** | 66.16 | 67.68 |

### 3.2 Llama3.2-3B: N ablation and LoRA baseline (paper Table 10)

| Run | Method | LR | BoolQ | PIQA | SIQA | HellaS. | WinoG. | ARC-e | ARC-c | OBQA | **Avg** (SE) | Paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A64 | DiaBlo N=64 | 1e-4 | 74.50 | 87.54 | 80.76 | 92.77 | 86.03 | 89.56 | 79.44 | 85.00 | **84.45** (0.34) | 84.44 |
| A512 | DiaBlo N=512 | 1e-4 | 69.48 | 84.39 | 78.97 | 88.75 | 80.03 | 86.95 | 73.63 | 78.60 | **80.10** (0.38) | 82.54 |
| A1024 | DiaBlo N=1024 | 1e-4 | 66.18 | 82.70 | 76.77 | 84.67 | 76.72 | 85.48 | 72.27 | 75.20 | **77.50** (0.39) | 81.60 |
| AL32 | LoRA r=32 | 3e-4 | 64.50 | 68.99 | 68.83 | 45.59 | 68.11 | 63.97 | 49.74 | 62.60 | **61.54** (0.45) | – |
| AL32a | LoRA r=32 | 1e-4 | 73.39 | 87.11 | 80.19 | 92.54 | 86.50 | 89.18 | 77.65 | 84.60 | **83.90** (0.34) | – |
| AL32b | LoRA r=32 | 2e-4 | 71.28 | 82.75 | 78.15 | 85.39 | 82.48 | 84.13 | 70.56 | 79.40 | **79.27** (0.38) | – |

**N ablation trend:**

| Step | Paper Table 10 | Ours | Ours, paired 95% CI |
|---|---|---|---|
| N=64 → N=512 | −1.90 | −4.35 | −3.68 to −5.05 |
| N=512 → N=1024 | −0.94 | −2.60 | −2.17 to −3.09 |

**DiaBlo vs LoRA (paired, same questions):**

| Comparison | Difference | 95% paired bootstrap CI | McNemar z (DiaBlo-only right / LoRA-only right) |
|---|---|---|---|
| A64 − AL32a (LoRA lr 1e-4, best) | +0.55 | +0.01 to +1.20 | 2.33 (1052 / 948) |
| A64 − AL32b (LoRA lr 2e-4) | +5.18 | +4.43 to +5.95 | 21.98 (2371 / 1080) |
| A64 − AL32 (LoRA lr 3e-4) | +22.91 | +21.95 to +23.88 | 72.88 (7993 / 1059) |

### 3.3 Cost: memory, speed, GPU-hours

| Run | GPU (node) | s/step | Peak memory (GB) | Training time | GPU-hours (train + eval) |
|---|---|---|---|---|---|
| C3 | A5000 (deep1) | 2.149 | 21.20 | 19.1 h | 20.51 |
| G1 | A5000 (deep1) | 7.520 | 19.75 | 0.98 h (468 steps) | 1.26 |
| A64 | 2080 Ti (deep1) | 1.733 | 9.88 | 15.4 h | 16.53 |
| AL32 | 2080 Ti (deep23) | 1.730 | 9.96 | 15.3 h | 16.35 |
| AL32a | 2080 Ti (deep23) | 1.697 | 9.96 | 15.1 h | 16.03 |
| AL32b | 2080 Ti (deep29) | 1.430 | 9.96 | 12.7 h | 13.64 |
| A512 | 2080 Ti (deep1, then deep23) | 1.568* | 9.39* | * | 16.01 |
| A1024 | 2080 Ti (deep1, then deep29) | 1.469* | 9.36* | * | 15.86 |

\* A512 and A1024 were resumed after an interruption (Section 5); `train_stats.json` covers only the final segment.

Peak memory doesn't depend on the node (same GPU model): DiaBlo N=64 used 9.88 GB and every LoRA r=32 run 9.96 GB. Speed does
depend on the node. The three identical-except-LR LoRA runs took 1.43–1.73 s/step on different nodes, and DiaBlo N=64 (1.733,
deep1) falls inside that range. No DiaBlo N=64 and LoRA pair ran on the same node, so we can only say their speed is the same
to within node-to-node variation, not measure a precise ratio.

**Total GPU-hours for the paper-scale work:** 116.5 of the 1000 available (sum of Slurm elapsed time per job, including 0.31 h
of benchmarking). The account shows 129.2 h used in total, which includes the earlier small-scale Qwen study.

## 4. Conclusions per claim

- **C1, DiaBlo accuracy: supported.** C3 is 87.15 vs 87.3 (about 0.5 SE away), and A64 is 84.45 vs 84.44.
- **C2, DiaBlo ≥ LoRA: partly supported.** DiaBlo beat every LoRA run we trained, but against the best LoRA LR the margin is
  only +0.55 points, with a CI that nearly touches zero. Our C3 (87.15) is above the paper's Llama3-8B LoRA (80.8) and DoRA (85.2)
  numbers, but those come from another pipeline, and our same-pipeline 3B test suggests a well-tuned LoRA closes most of that gap.
  A practical point in DiaBlo's favour: its LR of 1e-4 was not tuned, while LoRA ranged from 61.54 to 83.90 across three LRs.
- **C3, LoRA-like cost: supported.** Peak memory is essentially equal (9.88 vs 9.96 GB). Speed is within the node-to-node
  spread of the LoRA runs (DiaBlo 1.733 s/step; LoRA 1.430–1.730). DiaBlo also has about 12% fewer trainable parameters
  (28.9M vs 33.0M).
- **C4, too many blocks hurt: supported in direction, not in size.** Accuracy falls with N as in Table 10, but our drop is
  about twice as large. One possible explanation (untested): Table 10's LR is not published, so we used 1e-4 for every N. Larger N
  means far fewer parameters (3.6M at N=512, 1.8M at N=1024), which may need a larger LR.
- **C5, math fine-tuning: approximately supported.** G1 is 65.96 vs 67.68 (−1.72, about 1.3 SE from one seed), so it is
  consistent with the paper but slightly lower.

## 5. Deviations from the paper, and incidents

- **Hardware and batching:** an A5000 / 2080 Ti instead of an A100-80GB. We kept the paper's effective batch size, using a
  smaller per-device batch with gradient accumulation (G1 also used gradient checkpointing). This changes speed, not the method.
- **Evaluation prompt:** the released eval scripts don't use the training template (commonsense sends only the bare
  instruction; math indents with 16 spaces where training uses 12). We evaluate with the exact training prompt
  (`diablo/prompts.py`). Decoding and answer extraction are unchanged.
- **Pad token:** no new `<PAD>` token is added (Llama-3 pads with EOS). This avoids a randomly initialised extra output row.
- **Unpublished settings:** Table 10's LR (we used 1e-4 for all N) and adapter dropout (0.05, from the released scripts).
- **LoRA baseline:** not in the authors' repo; added via `peft` (r=32, α=2r, same modules). The first LR (3e-4, LLM-Adapters'
  default) underfit badly: its training loss stayed around 0.05 while DiaBlo's fell to about 0.002, and HellaSwag reached only
  45.59. We confirmed this was not an evaluation bug (all 280 adapter tensors load strictly, eval settings match training) and
  added two lower LRs. The LoRA LR was thus tuned on test results (3 values); DiaBlo was not tuned at all.
- **HellaSwag truncation (faithful to the paper's setup):** with max length 256, 24,015 of the 39,905 HellaSwag training
  examples (60.2%) are longer than 256 tokens and lose their answer, against 0.2% of the other tasks. This is the standard
  LLM-Adapters setting the paper uses, so we kept it.
- **Incident, 2026-11-22 17:39 UTC:** the cluster's shared disk (used by all users) filled up and crashed A512, A1024 and
  AL32. A512 resumed from step 28,400 and A1024 from step 15,600, restoring the optimizer, LR scheduler, grad-scaler and RNG
  state. A half-written checkpoint was discarded automatically. AL32 had no checkpoint yet and restarted from scratch. About
  6 minutes of training was recomputed per run.

## 6. Limitations

- **One seed per run.** Run-to-run variance is not measured. The paired CIs cover test-set sampling only, not training noise.
- **The LoRA comparison is on 3B only**, with r=32 and three LRs. Other ranks, α values, DoRA, and an 8B LoRA rerun were not tried.
- **Not reproduced:** Llama2-7B commonsense (Table 1), MetaMathQA (Table 2), the LR and N sweeps on Llama3-8B (Fig. 5),
  N=16 / N=256 (Table 10), 13B models (they don't fit 24 GB), quantized and code/safety experiments. See REPRODUCTION.md for reasons.
- **Speed numbers are node-dependent**, and A512/A1024's stats cover only their last segment.
- Our eval-prompt fix means our numbers are not strictly the released evaluation protocol. We didn't measure the effect
  (`--prompt_style upstream` exists for that).

## 7. Suggested slide outline

1. What DiaBlo is: train only diagonal blocks of each weight matrix (figure: `Figures/DiaBlo.png`).
2. What we set out to check: claims C1–C5, minimum viable reproduction, 1000 GPU-hours, student cluster.
3. Setup: models, data, hardware constraints (one 24 GB card for 8B), chained 24 h jobs.
4. Headline: 87.15 vs 87.3 and 84.45 vs 84.44 (TL;DR table).
5. N ablation: our curve vs Table 10 (same direction, steeper).
6. DiaBlo vs LoRA: the three LoRA LRs vs DiaBlo; same memory and speed.
7. GSM8K: 65.96 vs 67.68.
8. What we had to change, and why (Deviations); the disk-full incident and resume.
9. Limitations and takeaways.

## Provenance

All raw outputs live on the cluster under `~/diablo/`. Copies of the summary files used here are in
[`report/data/`](report/data/), and the auto-generated tables are in [`report/paper_results.md`](report/paper_results.md).

| Numbers | Source |
|---|---|
| Per-task and average accuracy | `eval-<model>/<RUN>/summary.json` (commonsense), `eval-Llama3-8B/G1_hf_summary.json` (GSM8K) |
| s/step, peak memory, training time, step counts | `results-<model>/<RUN>/train_stats.json` |
| Hyper-parameters | `results-<model>/<RUN>/run_config.json`, `scripts/slurm/runs.sh` |
| Paper numbers | arXiv:2506.03230, Tables 1, 5, 10 (as recorded in `scripts/slurm/runs.sh`) |
| SEs, paired CIs, McNemar | computed from the per-question `eval-<model>/<RUN>/<task>.jsonl` files (bootstrap seed 0, 1000 resamples) |
| GPU-hours | `sacct` elapsed time per job (2026-11-19 to 2026-11-24); account total from `my_gpu_usage.sh` |
| HellaSwag truncation | Llama-3.2-3B tokenizer over `commonsense_170k.json` with the training template, plus EOS |
| Training losses | `results-<model>/<RUN>/log_history.json` |
