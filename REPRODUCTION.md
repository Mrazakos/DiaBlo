# Reproducing DiaBlo on a free Colab T4

Paper: *DiaBlo: Diagonal Blocks Are Sufficient For Finetuning* (arXiv:2506.03230, ICLR 2026).
Upstream code: https://github.com/ziyangjoy/DiaBlo @ `83ffb6c`. This branch: `colab-repro`.

## What is being reproduced
The paper's full experiments (Llama2/3 7–8B, MetaMathQA 395k × 2048 tokens, Commonsense-170k) are far beyond a free T4.
This study is a **method-level reproduction at 1.5B scale**: GSM8k-train → GSM8k-test on **Qwen2.5-1.5B**. It tests
the paper's relative claims with matched baselines:

| Claim | Paper evidence | Runs |
|---|---|---|
| **C1** DiaBlo ≥ LoRA at a comparable trainable-parameter budget | Table 2: Llama2-7B/MetaMathQA, GSM8k 66.5 (DiaBlo N=64, 1.04%) vs 60.6 (LoRA r=64, 1.67%) | R1 vs R2, R3 vs R8 |
| **C2** Accuracy improves as N decreases but plateaus around N=64 | App. A.3.5 (Table 10, Fig. 5) | R3 / R1 / R4 |
| **C3** Robust across learning rates | App. A.3.5, Fig. 5 (GSM8k, Llama3-8B) | R1 vs R7 |

It does **not** reproduce the paper's absolute numbers.

Trainable parameters on Qwen2.5-1.5B (q,k,v,up,down × 28 layers):

| DiaBlo | params | LoRA | params |
|---|---|---|---|
| N=32 | 26.84M | r=32 | 24.77M |
| N=64 | 13.42M | r=16 | 12.39M |
| N=128 | 6.71M | r=8 | 6.19M |

The optional stretch (`notebooks/04_stretch_llama2_4bit.ipynb`) compares NF4 DiaBlo with QLoRA on Llama2-7B. The paper's quantized table
uses GPTQ/MagR and Math10K, which aren't in the repo, so this is a related experiment rather than a reproduction of Table 4.

## How to run
1. Push this branch to your GitHub fork. Set `REPO_URL` in the first cell of each notebook. Alternatively, copy the repo folder to
   `MyDrive/diablo/DiaBlo`.
2. In Colab, open `notebooks/00_setup_smoketest.ipynb` on a T4 and run all cells. This runs the tests, tokenizes GSM8k, checks zero-init and
   runs the fp16 smoke test with resume, and prints s/step → hours per run.
3. Open `notebooks/01_train.ipynb`, set `RUN_ID`, and run all cells. Re-run after a disconnect and it resumes from Drive.
   Priority: R1, R2, R3, R4, then R6/R7, then R8/R5.
4. Run `notebooks/02_eval.ipynb`: the parity check once, then the full eval of R0 (base model) and the finished runs.
5. Run `notebooks/03_results.ipynb` for the table, loss curves, accuracy-vs-params plot, McNemar tests and write-up template.

Everything persistent lives in `MyDrive/diablo/` (datasets, results/<RUN_ID>, eval). The HF model cache is kept on local disk (`/content/hf_cache`), because Drive corrupts multi-GB safetensors files.

## Running on the SZTE Deep Cluster (Slurm)
Branch `deepcluster` adds `scripts/slurm/`, which wraps the same `train.py` / `evaluate_gsm8k.py` calls as the notebooks.
Cluster rules that shape it: 20 GPU-hours per course account, 2 running jobs, 24 h per job, mostly 11–12 GB GPUs
(see https://github.com/szegedai/deepcluster_onboarding).

```bash
# on deep21, once
git clone -b deepcluster https://github.com/Mrazakos/DiaBlo.git ~/projects/DiaBlo && cd ~/projects/DiaBlo
bash scripts/slurm/setup_env.sh           # conda env "diablo" (python 3.12, torch cu126)
sbatch scripts/slurm/prep.sbatch          # CPU-only: model + data download, unit tests, tiny e2e
sbatch scripts/slurm/smoke.sbatch         # ~10 GPU-min: memory probe, s/step, eval timing -> "SMOKE ..." lines
# then the whole matrix, 2 jobs at a time (TIME / BS / GA from the smoke output)
TIME=02:00:00 bash scripts/slurm/submit_all.sh
```

Each run is one job: train with `--resume`, then HF-generate eval. Re-running `submit_all.sh` skips finished runs and resumes the
rest. All jobs are pinned to one GPU type (`rtx2080ti`). Everything persistent lives in `~/diablo/` with the same layout as on
Drive (`datasets/`, `results/<RUN_ID>/`, `eval/<RUN_ID>_hf.jsonl`), so `notebooks/03_results.ipynb` works on a copy with `BACKEND = "hf"`.

Run matrix on the cluster (shared settings as in the notebooks; R0 is the base model):

| ID | Method | Setting | Trainable | LR | Seed | Claim tested |
|---|---|---|---|---|---|---|
| R1 | DiaBlo | N=64 | 13.42M | 5e-4 | 42 | C1, C2, C3 |
| R2 | LoRA | r=16 | 12.39M | 2e-4 | 42 | C1 |
| R3 | DiaBlo | N=32 | 26.84M | 5e-4 | 42 | C1, C2 |
| R8 | LoRA | r=32 | 24.77M | 2e-4 | 42 | C1 |
| R4 | DiaBlo | N=128 | 6.71M | 5e-4 | 42 | C2 |
| R9 | DiaBlo | N=512 | 1.76M | 5e-4 | 42 | C2 |
| R10 | DiaBlo | N=16 | 53.67M | 5e-4 | 42 | C2 |
| R7 | DiaBlo | N=64 | 13.42M | 2e-4 | 42 | C3 |
| R11 | DiaBlo | N=64 | 13.42M | 1e-4 | 42 | C3 |
| R12 | DiaBlo | N=64 | 13.42M | 1e-3 | 42 | C3 |
| R5a | DiaBlo | N=64 | 13.42M | 5e-4 | 1 | variance |
| R5b | LoRA | r=16 | 12.39M | 2e-4 | 1 | variance |

Compared with the Colab matrix, R9/R10 widen the N sweep and R11/R12 widen the LR sweep. The paper's N ablation (LLaMA3.2-3B,
commonsense) differs by only 0.3 points between N=32, 64 and 128 and drops visibly only from N=512, while one GSM8k accuracy has
about ±2.6 points of sampling noise (1319 questions), so three adjacent N values cannot test C2. R6 (LoRA at lr 5e-4) is dropped.

At N=512, Qwen's `k_proj`/`v_proj` (256 outputs) get 1-wide output blocks and only the first 256 blocks reach the output. This is
upstream `BlockLinear` behaviour when N exceeds the layer width, and it affects only those two modules.

### Second wave: closer to the paper (run table in `scripts/slurm/runs.sh`)
The first wave's fine-tuned runs all scored below the base model, probably because Qwen2.5-1.5B is already strong at GSM8k
(see `RESULTS.md`). Two follow-ups:

| Group | Runs | What changes |
|---|---|---|
| `paper` | P1 (DiaBlo N=64), P2 (LoRA r=16) | Paper Table 20 recipe on Qwen2.5-1.5B: 2 epochs, effective batch 32, DiaBlo lr 4e-4, warmup 20, all seven modules. LoRA keeps lr 2e-4 (no LoRA lr is given for this setup) |
| `paper` | S0 (base), S1, S2 | Same recipe on Qwen2.5-0.5B, a weaker base. Results go to `~/diablo/{results,eval}-Qwen2.5-0.5B/` |
| `llama` | L0-L2 (Llama-3.2-3B), H1/H2 (Llama3-8B) | Paper's model family, 4-bit NF4 base. Needs more GPU hours than the course budget, gated HF access (`HF_TOKEN`), probably `GPU=a5000` and a larger `--mem`, and a memory probe for BS/GA first |

```bash
TIME=02:00:00 bash scripts/slurm/submit_all.sh paper     # P1 P2 S0 S1 S2, 2 jobs at a time
MODEL=Qwen2.5-0.5B sbatch scripts/slurm/prep.sbatch      # once, before the S runs: model + tokenized data
```

Budget order if hours are short: finish R12/R5b, then P1/P2, then S0 (base check, ~0.2 GPU-h) and S1/S2 only if S0 is well below the 1.5B base.

## Changes vs upstream (all backwards compatible for the original Llama scripts)
- `diablo/common.py`: one shared `MODEL_MAP` (adds Qwen2.5-0.5B/1.5B, accepts raw HF ids) and `setup_pad_token`.
  Upstream always added `<PAD>` and resized embeddings to `len(tokenizer)`, which **shrinks** Qwen's padded embedding matrix.
  Now a pad token is added only if missing (Llama-2 behaviour is unchanged).
- `diablo/adapters.py`: `apply_adapter` (DiaBlo / LoRA via peft / none), `mark_trainable`, `load_adapter` (strict key check),
  `merge_adapter`, and `BlockAdapter`, a DiaBlo wrapper for 4-bit bitsandbytes layers that reuses the upstream einsum.
- `diablo/trainer.py`: adapter-only checkpoints also cover LoRA. Adds `_load_from_checkpoint`, so `--resume` works.
  Upstream's adapter-only saving broke the HF Trainer's resume.
- `train.py`: `--method {diablo,lora}`, `--lora_r`, `--resume`, `--save_steps` (default steps/100), `--max_steps`, `--report_to`
  (wandb is now opt-in), `--load_in_4bit`, `--run_name`, and a bf16-on-T4 guard. Writes `run_config.json` and `log_history.json`.
- `evaluate_gsm8k.py`: same prompt and scoring, now via `gsm8k_utils.py`. Adds LoRA/base/merged/4-bit support, resumable JSONL output, and
  reports all 1319 questions plus the original first-1312 protocol.
- `eval_gsm8k_vllm.py` (new): fast eval of merged models with identical scoring (the answer is extracted from prompt+completion, as upstream does).
- `merge.py`: also merges LoRA, merges in fp32, and can read `run_config.json` via `--run_dir`.
- `tests/`: unit tests (zero-init identity, merge == adapter forward incl. non-divisible shapes, block-diagonal structure,
  `BlockAdapter` == `BlockLinear`) and `tests/e2e_tiny.py`, which runs train → resume → merge → eval on a tiny random Qwen2 for both methods
  and asserts that the merged and adapter generations are identical.

Not touched: commonsense/MetaMath/MATH scripts, which are out of scope here.

## Known deviations from the paper
Smaller model (Qwen2.5-1.5B), GSM8k-train instead of MetaMathQA, fp16 on T4 (the paper uses bf16), LoRA at a matched parameter count rather than
r=64, a single seed by default (R5 adds a second), and repo-default hyperparameters (`scripts/train_gsm8k.sh`: lr 5e-4, effective batch 16,
3 epochs, warmup 100, dropout 0.05, qkvud) with only a small LR check for both methods (R6, R7).

The paper's only stated settings for fine-tuning on GSM8k-train (Table 20, Llama3-8B) are 2 epochs, batch 32, lr 4e-4, 20 warmup
steps and all seven modules (q,k,v,o,gate,up,down). This study uses the released script's settings instead. The paper's LR-robustness
figure (Fig. 5) is on Llama3-8B and has no LoRA comparison, and its N ablation is on commonsense reasoning, not GSM8k.
