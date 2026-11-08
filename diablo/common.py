"""Shared helpers: model name mapping and tokenizer/pad handling."""

MODEL_MAP = {
    "Llama2-7B": "meta-llama/Llama-2-7b-hf",
    "Llama2-13B": "meta-llama/Llama-2-13b-hf",
    "Llama3-8B": "meta-llama/Meta-Llama-3-8B",
    "Llama3-3B": "meta-llama/Llama-3.2-3B",
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
}

MODULE_MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "u": "up_proj",
              "d": "down_proj", "o": "o_proj", "g": "gate_proj"}


def resolve_model(name: str) -> str:
    """Map a short name to an HF id; pass raw HF ids through unchanged."""
    return MODEL_MAP.get(name, name)


def parse_target_modules(spec: str) -> list:
    return [MODULE_MAP[c] for c in spec if c in MODULE_MAP]


def setup_pad_token(tokenizer, model=None):
    """Add a '<PAD>' token only if the tokenizer has none (Llama-2).

    The original code always added '<PAD>' and resized embeddings to len(tokenizer),
    which *shrinks* Qwen's padded embedding matrix. Here we only resize when a
    token was actually added and the embedding matrix is too small.
    """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        if model is not None and model.get_input_embeddings().weight.shape[0] < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
    if model is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer
