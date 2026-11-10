"""Worst-case GPU memory of one training step, measured before spending budget on real runs.

Builds the model exactly like train.py (fp16 base, fp32 adapters, fp16 autocast + GradScaler),
then runs one forward/backward/optimizer step on the LONGEST training examples. A real batch
can never be larger than that, so the peak printed here is an upper bound for the whole run.

    python scripts/slurm/mem_probe.py --model Qwen/Qwen2.5-1.5B --data_dir $DATA
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from diablo import apply_adapter, mark_trainable, parse_target_modules, setup_pad_token

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--data_dir", required=True)
ap.add_argument("--target_modules", default="qkvud")
ap.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4], help="ascending, so an OOM comes last")
args = ap.parse_args()

# Largest configuration of each method in the run matrix, plus the N=512 edge case.
CONFIGS = [("diablo", dict(num_blocks=16)), ("lora", dict(lora_r=32)), ("diablo", dict(num_blocks=512))]

assert torch.cuda.is_available(), "no GPU visible: was the job submitted with --gres=gpu:...?"
dev = torch.device("cuda")
total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)} | {total_gb:.1f} GB | torch {torch.__version__}")

train = load_from_disk(os.path.join(args.data_dir, "GSM8k_processed/train"))
order = sorted(range(len(train)), key=lambda i: len(train[i]["input_ids"]), reverse=True)
tok = AutoTokenizer.from_pretrained(args.model)


def longest_batch(bs, pad_id):
    rows = [train[i] for i in order[:bs]]
    width = max(len(r["input_ids"]) for r in rows)

    def left_pad(key, value):
        return torch.tensor([[value] * (width - len(r[key])) + list(r[key]) for r in rows], device=dev)

    return width, dict(input_ids=left_pad("input_ids", pad_id), attention_mask=left_pad("attention_mask", 0),
                       labels=left_pad("labels", -100))


worst = {}
for method, size in CONFIGS:
    tag = f"{method} {size}"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    setup_pad_token(tok, model)
    model = apply_adapter(model, method, parse_target_modules(args.target_modules), dropout=0.05, **size)
    model.to(dev)
    _, trainable = mark_trainable(model, torch.float32)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    for bs in args.batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        try:
            width, batch = longest_batch(bs, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_reserved() / 2**30
            alloc = torch.cuda.max_memory_allocated() / 2**30
            worst[bs] = max(worst.get(bs, 0.0), peak)
            print(f"[{tag}] {trainable / 1e6:.2f}M trainable | batch {bs} x {width} tokens | loss {loss.item():.3f} "
                  f"| peak reserved {peak:.2f} GB (allocated {alloc:.2f} GB)")
            del batch, loss
        except torch.OutOfMemoryError:
            worst[bs] = float("inf")
            print(f"[{tag}] batch {bs}: OUT OF MEMORY")
            opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    del model, opt, scaler
    gc.collect()
    torch.cuda.empty_cache()

print()
for bs in sorted(worst):
    fits = worst[bs] <= total_gb - 1.0   # keep ~1 GB headroom
    print(f"PROBE batch_size={bs}: worst-case peak {worst[bs]:.2f} GB of {total_gb:.1f} GB -> {'OK' if fits else 'TOO BIG'}")
