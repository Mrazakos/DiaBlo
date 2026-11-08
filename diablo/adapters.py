"""Build base models with DiaBlo or LoRA adapters, load adapter checkpoints, merge.

Used by train.py, merge.py and the evaluation scripts so the model is
constructed identically everywhere.
"""

import math

import torch
import torch.nn as nn

from .block_linear import BlockLinear, replace_linear_with_blocklinear, replace_blocklinear_with_linear


# ── DiaBlo on top of an arbitrary (e.g. bitsandbytes 4-bit) base layer ────────

class BlockAdapter(nn.Module):
    """DiaBlo adapter that *wraps* an existing layer instead of copying its weight.

    Needed for quantized bases (bnb Linear4bit), whose weights can't be copied
    into a plain nn.Linear. Output = base_layer(x) + blockdiag(dropout(x)).
    """

    block_forward = BlockLinear.block_forward  # reuse the reference einsum

    def __init__(self, base_layer: nn.Module, num_blocks: int, drop_out: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.block_size_in = math.ceil(self.in_features / num_blocks)
        self.block_size_out = math.ceil(self.out_features / num_blocks)
        n1 = math.ceil(self.in_features / self.block_size_in)
        n2 = math.ceil(self.out_features / self.block_size_out)
        self.num_blocks = max(n1, n2)
        self.block_A = nn.Parameter(torch.zeros(self.num_blocks, self.block_size_in, self.block_size_out))
        self.in_diff = self.block_size_in * self.num_blocks - self.in_features
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        result = self.base_layer(x)
        delta = self.block_forward(self.dropout(x).to(self.block_A.dtype))
        return result + delta.to(result.dtype)


def wrap_with_block_adapter(module: nn.Module, num_blocks: int, drop_out: float, target_modules: list):
    for name, child in module.named_children():
        if name in target_modules and hasattr(child, "in_features"):
            adapter = BlockAdapter(child, num_blocks, drop_out)
            # Move only the new parameter; don't call .to() on the quantized base.
            adapter.block_A.data = adapter.block_A.data.to(child.weight.device)
            setattr(module, name, adapter)
        else:
            wrap_with_block_adapter(child, num_blocks, drop_out, target_modules)


# ── Unified entry points ──────────────────────────────────────────────────────

def apply_adapter(model, method: str, target_modules: list, num_blocks: int = 64,
                  lora_r: int = 16, lora_alpha: int = None, dropout: float = 0.05,
                  layer_min: int = 0, layer_max: int = None, quantized: bool = False):
    """Attach adapters to `model`; returns the (possibly wrapped) model.

    method: "diablo" | "lora" | "none"
    """
    if method == "none":
        return model
    layers = model.model.layers
    layer_max = len(layers) - 1 if layer_max is None else min(layer_max, len(layers) - 1)
    selected = layers[layer_min: layer_max + 1]

    if method == "diablo":
        if quantized:
            for layer in selected:
                wrap_with_block_adapter(layer, num_blocks, dropout, target_modules)
        else:
            replace_linear_with_blocklinear(selected, num_blocks=num_blocks, drop_out=dropout,
                                            target_modules=target_modules)
        return model

    if method == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha if lora_alpha is not None else 2 * lora_r,
            lora_dropout=dropout,
            target_modules=target_modules,
            layers_to_transform=list(range(layer_min, layer_max + 1)),
            bias="none",
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, cfg)

    raise ValueError(f"Unknown method: {method}")


def mark_trainable(model, adapter_dtype=torch.float32):
    """Freeze everything except adapter params; keep adapters in `adapter_dtype`."""
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "block_A" in name or "lora_" in name:
            p.data = p.data.to(adapter_dtype)
            p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_adapter(model, path: str):
    sd = torch.load(path, map_location="cpu")
    sd = {k: v for k, v in sd.items() if "scores" not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Adapter keys not found in model (wrong method/num_blocks?): {unexpected[:5]}")
    n_adapter = sum(1 for k in model.state_dict() if "block_A" in k or "lora_" in k)
    if n_adapter != len(sd):
        raise RuntimeError(f"Adapter has {len(sd)} tensors but model expects {n_adapter}")
    return model


def merge_adapter(model, method: str):
    """Fold adapters into plain nn.Linear weights; returns a plain HF model."""
    if method == "diablo":
        replace_blocklinear_with_linear(model.model)
        return model
    if method == "lora":
        return model.merge_and_unload()
    return model
