"""GSM8k prompt + answer extraction, copied verbatim from the original evaluate_gsm8k.py
so the HF and vLLM evaluators score identically."""

import json
import os
import re

QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
ANSWER_PROMPT = "The final answer is: "


def build_prompt(question: str) -> str:
    return question + QUESTION_PROMPT


def extract_answer_gsm8k(sentence):
    """Extract answer from generated text. Try 'The final answer is:' first, then last number."""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = str(pred[-1])
    else:
        pred_answer = str(pred[-1])
    try:
        return float(pred_answer)
    except ValueError:
        return float('inf')


def target_answer(example) -> float:
    return float(example["answer"].split("####")[-1].replace(",", ""))


def is_correct(example, output: str) -> bool:
    return target_answer(example) == extract_answer_gsm8k(output)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(records, n_test: int, batch_size_for_parity: int = 16) -> dict:
    """Accuracy on all examples, plus the original script's floor-batch subset."""
    records = sorted(records, key=lambda r: r["idx"])
    n = len(records)
    acc = sum(r["correct"] for r in records) / max(n, 1)
    n_parity = (n_test // batch_size_for_parity) * batch_size_for_parity
    parity = [r for r in records if r["idx"] < n_parity]
    acc_parity = sum(r["correct"] for r in parity) / max(len(parity), 1)
    return {"n": n, "accuracy": acc, f"n_first_{n_parity}": len(parity), "accuracy_orig_protocol": acc_parity}
