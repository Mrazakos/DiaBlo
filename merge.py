"""
Merge DiaBlo (or LoRA) adapter weights back into the base model.

Usage:
    python merge.py --model_name Qwen2.5-1.5B --method diablo --num_blocks 64 \
        --adapter_path results/.../adapter.chkpt --output_dir merged_model/ --precision fp16
    # or read method/num_blocks/lora_r/target_modules from the run's run_config.json:
    python merge.py --run_dir results/<run_name> --output_dir merged_model/ --precision fp16
"""

import os
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from diablo import resolve_model, parse_target_modules, setup_pad_token, apply_adapter, load_adapter, merge_adapter

parser = argparse.ArgumentParser(description="Merge adapter into base model")
parser.add_argument("--run_dir", type=str, default=None, help="Training output dir (reads run_config.json)")
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--method", type=str, default="diablo", choices=["diablo", "lora"])
parser.add_argument("--adapter_path", type=str, default=None, help="Path to adapter.chkpt file")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged model")
parser.add_argument("--num_blocks", type=int, default=64)
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=None)
parser.add_argument("--target_modules", type=str, default="qkvud")
parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
parser.add_argument("--model_cache_dir", type=str, default=None)
parser.add_argument("--hf_token", type=str, default=None)
args = parser.parse_args()

if args.run_dir:
    with open(os.path.join(args.run_dir, "run_config.json")) as f:
        cfg = json.load(f)
    for k in ["model_name", "method", "num_blocks", "lora_r", "lora_alpha", "target_modules", "layer_min", "layer_max"]:
        if k in cfg:
            setattr(args, k, cfg[k])
    args.adapter_path = args.adapter_path or os.path.join(args.run_dir, "adapter.chkpt")
if not args.model_name or not args.adapter_path:
    parser.error("Need --run_dir, or both --model_name and --adapter_path")

hf_token = args.hf_token or os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(hf_token)

model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", None)
compute_precision = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
load_name = resolve_model(args.model_name)

# Load in fp32 so the merge W + D^T is done at full precision, then cast once.
model = AutoModelForCausalLM.from_pretrained(load_name, cache_dir=model_dir, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir)
setup_pad_token(tokenizer, model)

model = apply_adapter(model, args.method, parse_target_modules(args.target_modules),
                      num_blocks=args.num_blocks, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                      dropout=0.0, layer_min=getattr(args, "layer_min", 0) or 0,
                      layer_max=getattr(args, "layer_max", None))
load_adapter(model, args.adapter_path)
model = merge_adapter(model, args.method)

model.to(compute_precision)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"Merged model saved to {args.output_dir}")
