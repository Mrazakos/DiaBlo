"""End-to-end pipeline test on a tiny random Qwen2 model (CPU or GPU, ~1-2 min).

train -> resume -> merge -> eval(adapter) vs eval(merged) for both DiaBlo and LoRA.

    python tests/e2e_tiny.py [--workdir /tmp/diablo_e2e] [--precision fp32|fp16]
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.join(REPO, "_e2e_tmp"))
ap.add_argument("--precision", default="fp32")
ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
a = ap.parse_args()

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

W = a.workdir
os.makedirs(W, exist_ok=True)
model_dir, data_dir, out_dir = (os.path.join(W, d) for d in ("tiny_model", "datasets", "results"))


def run(*cmd):
    print("\n$", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-6000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout + r.stderr


# 1. tiny model + real tokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer)
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                  tie_word_embeddings=True)
torch.manual_seed(0)
Qwen2ForCausalLM(cfg).save_pretrained(model_dir)
tok.save_pretrained(model_dir)

# 2. tiny datasets in the layout process_gsm8k.py produces
qs = [f"Tom has {i} apples and buys {i + 2} more. How many apples?" for i in range(40)]
ans = [f"He has {i} + {i + 2} = {2 * i + 2}.\n#### {2 * i + 2}" for i in range(40)]
rows = {"input_ids": [], "attention_mask": [], "labels": []}
for q, s in zip(qs, ans):
    p = tok(q + "\nAnswer:\n", add_special_tokens=False)["input_ids"]
    r = tok(s.replace("####", "The final answer is: "), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    rows["input_ids"].append(p + r)
    rows["attention_mask"].append([1] * (len(p) + len(r)))
    rows["labels"].append([-100] * len(p) + r)
Dataset.from_dict(rows).save_to_disk(os.path.join(data_dir, "GSM8k_processed/train"))
DatasetDict({"train": Dataset.from_dict({"question": qs[:4], "answer": ans[:4]}),
             "test": Dataset.from_dict({"question": qs[:6], "answer": ans[:6]})}
            ).save_to_disk(os.path.join(data_dir, "GSM8k_raw"))

common = ["--model_name", model_dir, "--data_dir", data_dir, "--output_dir", out_dir,
          "--precision", a.precision, "--per_device_batch_size", "2", "--gradient_accumulation_steps", "2",
          "--save_steps", "3", "--warmup_steps", "1", "--lr", "1e-2"]

for method, extra in [("diablo", ["--num_blocks", "4"]), ("lora", ["--lora_r", "2"])]:
    name = f"e2e_{method}"
    run_dir = os.path.join(out_dir, name)
    run("train.py", *common, "--method", method, *extra, "--run_name", name, "--max_steps", "3")
    log = run("train.py", *common, "--method", method, *extra, "--run_name", name, "--max_steps", "6", "--resume")
    assert "Resumed adapter weights" in log, "resume did not load adapter"
    state = json.load(open(os.path.join(run_dir, "log_history.json")))
    cfg_run = json.load(open(os.path.join(run_dir, "run_config.json")))
    print(f"[{method}] trainable params: {cfg_run['trainable_params']:,}; steps logged: {len(state)}")

    run("merge.py", "--run_dir", run_dir, "--output_dir", os.path.join(W, f"merged_{method}"), "--precision", "fp32")

    ev = ["evaluate_gsm8k.py", "--data_dir", data_dir, "--precision", "fp32", "--batch_size", "3",
          "--max_new_tokens", "12"]
    f_ad, f_mg = (os.path.join(W, f"eval_{method}_{k}.jsonl") for k in ("adapter", "merged"))
    for f in (f_ad, f_mg):
        if os.path.exists(f):
            os.remove(f)
    run(*ev, "--model_name", model_dir, "--method", method, *extra, "--adapter_path",
        os.path.join(run_dir, "adapter.chkpt"), "--output_file", f_ad)
    run(*ev, "--model_path", os.path.join(W, f"merged_{method}"), "--method", "none", "--output_file", f_mg)
    o_ad = [json.loads(l)["output"] for l in open(f_ad)]
    o_mg = [json.loads(l)["output"] for l in open(f_mg)]
    assert len(o_ad) == 6, len(o_ad)
    same = sum(x == y for x, y in zip(o_ad, o_mg))
    print(f"[{method}] adapter vs merged generations identical: {same}/6")
    assert same == 6, "merged model diverges from adapter model"

print("\nE2E OK")
