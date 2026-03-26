#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a LoRA adapter on the frozen test set.

Computes avg_loss and perplexity over all examples in the test JSONL,
then writes a metrics JSON file.

Usage:
  python eval_adapter.py \
    --adapter qwen25-1.5b-tokipona-unsloth-A01 \
    --test-file pedagogy_dataset_test.jsonl

The output file is named after the adapter by default:
  eval_<adapter_name>_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA adapter on a frozen test JSONL (avg_loss + perplexity)."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the adapter directory (e.g. qwen25-1.5b-tokipona-unsloth-A01)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model HF id (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--test-file",
        default="pedagogy_dataset_test.jsonl",
        help="Frozen test JSONL file (default: pedagogy_dataset_test.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output metrics JSON file. "
            "Default: eval_<adapter_name>_metrics.json"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Evaluation batch size (default: 2)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length for tokenization (default: 1024)",
    )
    return parser.parse_args(argv)


def load_texts(test_file: Path, tokenizer) -> List[str]:
    texts: List[str] = []
    with test_file.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ligne {i}: JSON invalide ({exc})") from exc
            txt = tokenizer.apply_chat_template(
                obj["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(txt)
    return texts


def evaluate(model, tokenizer, texts: List[str], batch_size: int, max_length: int):
    import torch

    loss_sum = 0.0
    n_batches = 0
    n_tokens = 0

    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)
        labels = input_ids.clone()
        labels[attn == 0] = -100

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)

        loss_sum += out.loss.detach().float().item()
        n_batches += 1
        n_tokens += int((attn == 1).sum().item())

        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(texts))
            print(f"  {done}/{len(texts)} exemples traites...", flush=True)

    avg_loss = loss_sum / max(1, n_batches)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, n_tokens


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    adapter_path = Path(args.adapter)
    test_file = Path(args.test_file)

    if not adapter_path.exists():
        print(f"ERROR: adapter not found: {adapter_path}")
        return 1
    if not test_file.exists():
        print(f"ERROR: test file not found: {test_file}")
        return 1

    output_file = Path(args.output) if args.output else Path(f"eval_{adapter_path.name}_metrics.json")

    print(f"Adapter : {adapter_path}")
    print(f"Test    : {test_file}")
    print(f"Output  : {output_file}")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}")
        return 1

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, evaluation will be slow.")

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("[1/3] Chargement tokenizer + modele de base...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
    )

    print("[2/3] Chargement adapter LoRA...")
    model = PeftModel.from_pretrained(base, str(adapter_path))

    print("[3/3] Evaluation sur le test set fige...")
    texts = load_texts(test_file, tokenizer)
    print(f"  {len(texts)} exemples charges depuis {test_file.name}")

    avg_loss, ppl, n_tokens = evaluate(
        model, tokenizer, texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    result = {
        "adapter": str(adapter_path),
        "base_model": args.base_model,
        "test_file": str(test_file),
        "num_examples": len(texts),
        "avg_loss": avg_loss,
        "perplexity": ppl,
        "evaluated_tokens": n_tokens,
    }

    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
