# DiaBlo paper-scale reproduction: results so far

(source: `~/diablo`)

### Minimum viable reproduction: our number vs the paper's

| Run | Experiment | Ours | Paper | Ours - paper | Status |
|---|---|---|---|---|---|
| G1 | Table 5 DiaBlo 1/64: 67.68 | 65.96 | 67.68 | -1.72 | done |
| C3 | Table 1 LLaMA3-8B N=64: avg 87.3 | 87.15 | 87.30 | -0.15 | done |
| A64 | Table 10 N=64: avg 84.44 | 84.45 | 84.44 | +0.01 | done |
| A512 | Table 10 N=512: avg 82.54 | 80.10 | 82.54 | -2.44 | done |
| A1024 | Table 10 N=1024: avg 81.60 | 77.50 | 81.60 | -4.10 | done |
| A16 | Table 10 N=16: avg 84.38 | - | 84.38 | - | not started |
| AL32 | no paper number (LoRA baseline for DiaBlo N=64) | 61.54 | - | - | done |
| AL32a | no paper number (LoRA r=32 baseline, lr 1e-4) | 83.90 | - | - | done |
| AL32b | no paper number (LoRA r=32 baseline, lr 2e-4) | 79.27 | - | - | done |

### Table 5 / Fig. 5: Llama3-8B fine-tuned on GSM8K (greedy, 1319 questions)

| Run | Model | Method | LR | GSM8K acc | Paper |
|---|---|---|---|---|---|
| G0 | Llama3-8B | base | - | _not started_ | base model (not in paper) |
| G1 | Llama3-8B | DiaBlo N=64 | 4e-4 | **65.96** | Table 5 DiaBlo 1/64: 67.68 |
| G2 | Llama3-8B | DiaBlo N=64 | 1e-4 | _not started_ | Fig. 5 (LR sweep) |
| G3 | Llama3-8B | DiaBlo N=64 | 2e-4 | _not started_ | Fig. 5 (LR sweep) |
| G4 | Llama3-8B | DiaBlo N=64 | 8e-4 | _not started_ | Fig. 5 (LR sweep) |
| G5 | Llama3-8B | DiaBlo N=64 | 1.6e-3 | _not started_ | Fig. 5 (LR sweep) |
| G6 | Llama3-8B | DiaBlo N=32 | 4e-4 | _not started_ | Fig. 5 (N sweep); Table 5 1/32 retained-update: 67.30 |
| G7 | Llama3-8B | DiaBlo N=128 | 4e-4 | _not started_ | Fig. 5 (N sweep); Table 5 1/128 retained-update: 63.11 |

### Table 1 (FP16): Commonsense-170k, 8 benchmarks (beam 4)

| Run | Model | Method | LR | BoolQ | PIQA | SIQA | HellaS. | WinoG. | ARC-e | ARC-c | OBQA | Avg | Paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | Llama2-7B | DiaBlo N=64 | 2e-4 | _not started_ | | | | | | | | | Table 1 LLaMA2-7B N=64: avg 83.4 |
| C2 | Llama2-7B | DiaBlo N=128 | 3e-4 | _not started_ | | | | | | | | | Table 1 LLaMA2-7B N=128: avg 83.5 |
| C5 | Llama2-7B | LoRA r=32 | 3e-4 | _not started_ | | | | | | | | | Table 1 LLaMA2-7B LoRA r=32: avg 77.6 (from DoRA) |
| C3 | Llama3-8B | DiaBlo N=64 | 5e-5 | 75.84 | 89.93 | 81.42 | 95.62 | 89.50 | 93.35 | 83.53 | 88.00 | **87.15** | Table 1 LLaMA3-8B N=64: avg 87.3 |
| C4 | Llama3-8B | DiaBlo N=128 | 1e-4 | _not started_ | | | | | | | | | Table 1 LLaMA3-8B N=128: avg 87.2 |

### Table 10: N ablation, Llama3.2-3B, commonsense

| Run | Model | Method | LR | BoolQ | PIQA | SIQA | HellaS. | WinoG. | ARC-e | ARC-c | OBQA | Avg | Paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A16 | Llama3.2-3B | DiaBlo N=16 | 1e-4 | _not started_ | | | | | | | | | Table 10 N=16: avg 84.38 |
| A64 | Llama3.2-3B | DiaBlo N=64 | 1e-4 | 74.50 | 87.54 | 80.76 | 92.77 | 86.03 | 89.56 | 79.44 | 85.00 | **84.45** | Table 10 N=64: avg 84.44 |
| A256 | Llama3.2-3B | DiaBlo N=256 | 1e-4 | _not started_ | | | | | | | | | Table 10 N=256: avg 83.92 |
| A512 | Llama3.2-3B | DiaBlo N=512 | 1e-4 | 69.48 | 84.39 | 78.97 | 88.75 | 80.03 | 86.95 | 73.63 | 78.60 | **80.10** | Table 10 N=512: avg 82.54 |
| A1024 | Llama3.2-3B | DiaBlo N=1024 | 1e-4 | 66.18 | 82.70 | 76.77 | 84.67 | 76.72 | 85.48 | 72.27 | 75.20 | **77.50** | Table 10 N=1024: avg 81.60 |
| AL32 | Llama3.2-3B | LoRA r=32 | 3e-4 | 64.50 | 68.99 | 68.83 | 45.59 | 68.11 | 63.97 | 49.74 | 62.60 | **61.54** | no paper number (LoRA baseline for DiaBlo N=64) |
| AL32a | Llama3.2-3B | LoRA r=32 | 1e-4 | 73.39 | 87.11 | 80.19 | 92.54 | 86.50 | 89.18 | 77.65 | 84.60 | **83.90** | no paper number (LoRA r=32 baseline, lr 1e-4) |
| AL32b | Llama3.2-3B | LoRA r=32 | 2e-4 | 71.28 | 82.75 | 78.15 | 85.39 | 82.48 | 84.13 | 70.56 | 79.40 | **79.27** | no paper number (LoRA r=32 baseline, lr 2e-4) |

### Table 2: Llama2-7B on MetaMathQA (beam 4)

| Run | Model | Method | LR | GSM8K | MATH | Avg | Paper |
|---|---|---|---|---|---|---|---|
| M1 | Llama2-7B | DiaBlo N=64 | 2e-4 | _not started_ | | | Table 2 N=64: GSM8K 66.5, MATH 17.6 |
| M2 | Llama2-7B | DiaBlo N=32 | 2e-4 | _not started_ | | | Table 2 N=32: GSM8K 66.3, MATH 20.4 |
| M1r | Llama2-7B | DiaBlo N=64 | 1.2e-3 | _not started_ | | | Table 2 N=64: GSM8K 66.5, MATH 17.6 (released-script recipe) |

### Cost projection from the bench job (GPU-hours, before the 30% margin)

| Run | GPU | batch x accum | grad. ckpt | s/step | train h | eval h | total h |
|---|---|---|---|---|---|---|---|
| A1024 | NVIDIA GeForce RTX 2080 Ti | 2 x 8 | no | 3.19 | 28.28 | 0.92 | 29.21 |
| A256 | NVIDIA GeForce RTX 2080 Ti | 2 x 8 | no | 3.19 | 28.28 | 0.92 | 29.21 |
| A512 | NVIDIA GeForce RTX 2080 Ti | 2 x 8 | no | 3.19 | 28.28 | 0.92 | 29.21 |
| A64 | NVIDIA GeForce RTX 2080 Ti | 2 x 8 | no | 3.19 | 28.28 | 0.92 | 29.21 |
| AL32 | NVIDIA GeForce RTX 2080 Ti | 2 x 8 | no | 3.19 | 28.28 | 0.92 | 29.21 |
| AL32a | NVIDIA GeForce RTX 2080 Ti | 2 x 8 | no | 3.19 | 28.28 | 0.92 | 29.21 |
| AL32b | NVIDIA GeForce RTX 2080 Ti | 2 x 8 | no | 3.19 | 28.28 | 0.92 | 29.21 |
| C3 | NVIDIA RTX A5000 | 2 x 8 | no | 3.61 | 32.05 | 1.23 | 33.28 |
| C4 | NVIDIA RTX A5000 | 2 x 8 | no | 3.61 | 32.05 | 1.23 | 33.28 |
| G0 | NVIDIA RTX A5000 | 4 x 8 | yes | 8.18 | 0.0 | 0.54 | 0.54 |
| G1 | NVIDIA RTX A5000 | 4 x 8 | yes | 8.18 | 1.06 | 0.54 | 1.59 |

Sum: 273 GPU-hours.
