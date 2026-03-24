#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning LoRA (QLoRA) for Qwen2.5-1.5B-Instruct on conversational JSONL data.

Expected input format per line (as in pedagogy_dataset_*):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

Usage example:
  python train_qwen25_lora.py \
    --train-file pedagogy_dataset_train.jsonl \
    --val-file pedagogy_dataset_val.jsonl \
    --output-dir qwen25-1.5b-tokipona-lora
"""

from __future__ import annotations

import argparse
import os
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen2.5-1.5B-Instruct on chat JSONL data"
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model on Hugging Face",
    )
    parser.add_argument(
        "--train-file",
        default="pedagogy_dataset_train.jsonl",
        help="Path to train JSONL",
    )
    parser.add_argument(
        "--val-file",
        default="pedagogy_dataset_val.jsonl",
        help="Path to validation JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="qwen25-1.5b-tokipona-lora",
        help="Output directory for adapter + tokenizer",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Save/eval interval in steps (default: 50)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a checkpoint directory to resume from",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantization (requires bitsandbytes)",
    )
    return parser.parse_args()


def _format_messages(example: dict[str, Any], tokenizer) -> dict[str, str]:
    messages = example.get("messages", [])
    if not messages:
        return {"text": ""}

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": rendered}


def main() -> int:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
    )
    from trl import SFTConfig, SFTTrainer

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if not os.path.exists(args.val_file):
        raise FileNotFoundError(f"Validation file not found: {args.val_file}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment. "
            "Likely cause: PyTorch CUDA build is incompatible with installed NVIDIA driver. "
            "Install a matching PyTorch build (e.g. cu128) and retry."
        )

    bnb_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        # 4-bit quantization keeps VRAM usage low on 12GB GPUs.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    raw_ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    train_ds = raw_ds["train"].map(
        lambda e: _format_messages(e, tokenizer),
        remove_columns=raw_ds["train"].column_names,
    )
    val_ds = raw_ds["validation"].map(
        lambda e: _format_messages(e, tokenizer),
        remove_columns=raw_ds["validation"].column_names,
    )

    train_ds = train_ds.filter(lambda x: len(x["text"].strip()) > 0)
    val_ds = val_ds.filter(lambda x: len(x["text"].strip()) > 0)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=100,
        logging_steps=max(10, args.save_steps // 2),
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        max_length=args.max_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training finished.")
    print(f"Adapter saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
