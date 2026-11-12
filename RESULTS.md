# DiaBlo reproduction: results vs. the paper

Paper: *DiaBlo: Diagonal Blocks Are Sufficient For Finetuning* (arXiv:2506.03230, ICLR 2026).
Study: method-level reproduction on **Qwen2.5-1.5B**, GSM8k-train → GSM8k-test, run on the SZTE Deep Cluster (RTX 2080 Ti, fp16).
Run settings are in `REPRODUCTION.md`; this file compares outcomes.

**Status:** R12 (DiaBlo N=64, lr 1e-3) and R5b (LoRA r=16, seed 1) were not finished when this was written. Their rows are marked pending.

## 1. What the paper states (claims we test)

DiaBlo trains only the diagonal blocks of each weight matrix (N blocks per matrix). Claims, as recorded in `REPRODUCTION.md`:

| ID | Claim | Paper evidence |
|---|---|---|
| C1 | DiaBlo ≥ LoRA at a comparable trainable-parameter budget | Table 2, Llama2-7B on MetaMathQA: GSM8k 66.5 (DiaBlo N=64, 1.04% params) vs 60.6 (LoRA r=64, 1.67% params) |
| C2 | Accuracy improves as N decreases (more parameters) but plateaus around N=64 | App. A.3.5 (Table 10, Fig. 5). Differences between N=32/64/128 are about 0.3 points (LLaMA3.2-3B, commonsense); a visible drop appears only from N=512 |
| C3 | Robust across learning rates | App. A.3.5, Fig. 5 (GSM8k, Llama3-8B; no LoRA comparison) |

Differences from the paper that limit comparability:

- Model is 1.5B (Qwen2.5), not 7–8B Llama.
- Training data is GSM8k-train (7473 examples), not MetaMathQA (395k).
- Hyperparameters follow the repo's `scripts/train_gsm8k.sh` (lr 5e-4, effective batch 16, 3 epochs, warmup 100, dropout 0.05, modules q,k,v,up,down). The paper's Table 20 (Llama3-8B, GSM8k) uses 2 epochs, batch 32, lr 4e-4, warmup 20 and all seven modules.
- fp16 on RTX 2080 Ti, since the paper uses bf16. Single seed 42 except for the variance runs.
- LoRA is matched on parameter count (r=16, 32), not r=64.
- Our absolute numbers are **not** comparable to the paper's. Only relative claims are tested.

## 2. Our results

1319 GSM8k test questions, greedy decoding, 512 new tokens. One accuracy has a standard error of about ±1.4 points, so differences under about 2–3 points are noise.

| Run | Method | Size | Trainable (M) | LR | Seed | GSM8k acc (%) | Final train loss* | Train time (h) |
|---|---|---|---|---|---|---|---|---|
| R0 | base model | – | 0 | – | – | **61.79** | – | – |
| R1 | DiaBlo | N=64 | 13.42 | 5e-4 | 42 | 54.36 | 0.125 | 0.64 |
| R2 | LoRA | r=16 | 12.39 | 2e-4 | 42 | 56.03 | 0.203 | 0.86 |
| R3 | DiaBlo | N=32 | 26.84 | 5e-4 | 42 | 53.15 | 0.092 | 0.64 |
| R8 | LoRA | r=32 | 24.77 | 2e-4 | 42 | 52.92 | 0.157 | 0.89 |
| R4 | DiaBlo | N=128 | 6.71 | 5e-4 | 42 | 53.53 | 0.184 | 0.62 |
| R9 | DiaBlo | N=512 | 1.76 | 5e-4 | 42 | 58.68 | 0.287 | 0.82 |
| R10 | DiaBlo | N=16 | 53.67 | 5e-4 | 42 | 55.95 | 0.079 | 0.65 |
| R7 | DiaBlo | N=64 | 13.42 | 2e-4 | 42 | 54.81 | 0.196 | 0.81 |
| R11 | DiaBlo | N=64 | 13.42 | 1e-4 | 42 | 58.45 | 0.253 | 0.63 |
| R5a | DiaBlo | N=64 | 13.42 | 5e-4 | 1 | 55.50 | 0.125 | 0.80 |
| R5b | LoRA | r=16 | 12.39 | 2e-4 | 1 | pending | – | – |
| R12 | DiaBlo | N=64 | 13.42 | 1e-3 | 42 | pending (training finished, eval incomplete) | – | – |

\*Mean of the last 10 logged training losses.

### Paired significance (McNemar exact test, same 1319 questions)

| Comparison | Diff (pts) | p | Reading |
|---|---|---|---|
| R1 (DiaBlo N=64) vs R2 (LoRA r=16) | −1.67 | 0.236 | no significant difference |
| R3 (DiaBlo N=32) vs R8 (LoRA r=32) | +0.23 | 0.913 | no significant difference |
| R1 vs R3 (N=64 vs 32) | +1.21 | 0.409 | no significant difference |
| R1 vs R4 (N=64 vs 128) | +0.83 | 0.581 | no significant difference |
| R1 vs R10 (N=64 vs 16) | −1.59 | 0.292 | no significant difference |
| R1 vs R9 (N=64 vs 512) | −4.32 | 0.002 | N=512 is **better** |
| R1 vs R7 (lr 5e-4 vs 2e-4) | −0.45 | 0.781 | no significant difference |
| R1 vs R11 (lr 5e-4 vs 1e-4) | −4.09 | 0.004 | lr 1e-4 is **better** |
| R1 vs R5a (seed 42 vs 1) | −1.14 | 0.417 | seed-to-seed spread of about 1 point |
| R1 vs R0 (DiaBlo vs base) | −7.43 | <0.001 | fine-tuning **hurt** |
| R2 vs R0 (LoRA vs base) | −5.76 | <0.001 | fine-tuning **hurt** |

## 3. Claim-by-claim comparison

### C1: DiaBlo ≥ LoRA at matched parameters: not reproduced, not refuted
- Paper: DiaBlo beats LoRA by 5.9 points (66.5 vs 60.6).
- Ours: at about 13M parameters DiaBlo is 1.67 points *below* LoRA (p=0.24). At about 26M it is +0.23 points above (p=0.91). Neither is significant.
- Both methods are statistically tied, so we neither confirm nor contradict the paper's advantage. The paper's setting (7B, 395k examples) differs a lot from ours.

### C2: Accuracy improves with more parameters and plateaus near N=64: not reproduced
- Paper: accuracy rises as N falls and plateaus around N=64. The drop appears at N=512.
- Ours, by trainable parameters: N=512 (1.76M) 58.68, N=128 (6.71M) 53.53, N=64 (13.42M) 54.36, N=32 (26.84M) 53.15, N=16 (53.67M) 55.95.
- N=32 through N=128 are within noise, which matches the paper's observation of about 0.3 points of spread, but the N=512 result goes the *opposite* way from the paper: the smallest-parameter run is significantly the best (p=0.002).
- Training loss falls monotonically with parameters (0.287 → 0.079 from N=512 to N=16), so more parameters fit the training set better but do not generalise better here.
- N=512 caveat: Qwen's k_proj/v_proj have 256 outputs, so upstream `BlockLinear` gives them 1-wide output blocks and only the first 256 blocks reach the output. This affects only those two modules.

### C3: Robustness across learning rates: partly reproduced
- Paper: DiaBlo is stable across learning rates.
- Ours: 5e-4 → 54.36, 2e-4 → 54.81 (tied, p=0.78). 1e-4 → 58.45, which is significantly better (p=0.004). 1e-3 (R12) is pending.
- No divergence at any tested rate so far, which fits "robust". But accuracy still moves by about 4 points across the range, and the best rate is the lowest tested.

## 4. Main finding that the paper does not predict

Every fine-tuned run scores **below the base model** (61.79). The less a run moves the model, the closer it stays to base:

| Run | Final train loss | Acc (%) |
|---|---|---|
| R9 (N=512) | 0.287 | 58.68 |
| R11 (lr 1e-4) | 0.253 | 58.45 |
| R10 (N=16) | 0.079 | 55.95 |
| R1 (N=64) | 0.125 | 54.36 |

R10 does not fit the pattern, so "less change is better" is a loose trend, not a rule.

Likely explanation (untested): Qwen2.5-1.5B is already strong at GSM8k because its pretraining included a lot of math data. Three epochs on 7473 GSM8k solutions then cause forgetting or style overfitting rather than a gain. The paper's setting, a weaker base with much more data, benefits from fine-tuning. Fine-tuning on GSM8k-train alone therefore cannot show what DiaBlo is worth here.

## 5. Conclusions

1. The method runs and trains stably on Qwen2.5-1.5B, with no divergence and no NaNs in the runs finished so far. Block-diagonal training is cheaper in time than LoRA (0.62–0.65 h vs 0.86–0.89 h per run at similar parameter counts) in this implementation.
2. C1 and C2 as stated in the paper are not reproduced at this scale. All DiaBlo-vs-LoRA and N=32/64/128 comparisons are statistically ties.
3. The comparison is confounded by the base model already being strong. A weaker base model (for example Qwen2.5-0.5B) would be the next test, because there fine-tuning should help and differences between methods should show.
4. Statistical power is limited: one seed per configuration, and R1 vs R5a already differ by 1.14 points from seed alone.

## 6. Limitations

- Single seed except R5a/R5b; no confidence intervals across seeds.
- Hyperparameters were not tuned per method (LoRA lr 2e-4, DiaBlo lr 5e-4).
- Paper's Table 20 settings were not used.
- R12 and R5b are incomplete.
- The paper's numbers quoted above come from `REPRODUCTION.md` and were not re-checked against the PDF in this write-up.
