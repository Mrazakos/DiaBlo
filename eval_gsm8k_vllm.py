"""
Fast GSM8k evaluation of a *merged* model (see merge.py) or a base model with vLLM.

Same prompt, greedy decoding, 512 new tokens and scoring as evaluate_gsm8k.py
(answer extracted from prompt + completion, as in the original script).

Usage:
    python eval_gsm8k_vllm.py --model_path merged/<run> --output_file eval/<run>.jsonl
    python eval_gsm8k_vllm.py --model_path Qwen/Qwen2.5-1.5B --output_file eval/base.jsonl
"""

import os
import json
import argparse

from datasets import load_from_disk

from gsm8k_utils import build_prompt, is_correct, summarize

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, help="Merged model dir or HF id")
parser.add_argument("--data_dir", type=str, default="./datasets")
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--dtype", type=str, default="float16", help="float16 on T4")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
parser.add_argument("--max_model_len", type=int, default=1024)
args = parser.parse_args()

from vllm import LLM, SamplingParams  # imported late so --help works without vllm

test_data = list(load_from_disk(os.path.join(args.data_dir, "GSM8k_raw"))["test"])
n_test = len(test_data)
if args.limit:
    test_data = test_data[: args.limit]
prompts = [build_prompt(ex["question"]) for ex in test_data]

llm = LLM(model=args.model_path, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization,
          max_model_len=args.max_model_len, seed=0)
tok = llm.get_tokenizer()
# Original script tokenizes with add_special_tokens=False -> pass token ids explicitly.
prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, skip_special_tokens=True)

try:
    from vllm.inputs import TokensPrompt
    inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]
except ImportError:
    inputs = [{"prompt_token_ids": ids} for ids in prompt_ids]
outputs = llm.generate(inputs, params)

os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
records = []
with open(args.output_file, "w") as f:
    for i, (ex, prompt, out) in enumerate(zip(test_data, prompts, outputs)):
        text = prompt + out.outputs[0].text
        rec = {"idx": i, "output": text, "correct": is_correct(ex, text)}
        records.append(rec)
        f.write(json.dumps(rec) + "\n")

summary = summarize(records, n_test if not args.limit else len(test_data))
summary.update({"model": args.model_path, "backend": "vllm"})
print(json.dumps(summary, indent=2))
with open(args.output_file.replace(".jsonl", "") + "_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
