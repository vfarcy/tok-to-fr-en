#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning LoRA avec Unsloth pour Qwen2.5-1.5B-Instruct sur JSONL conversationnel.

Format attendu par ligne:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

Usage:
  python train_qwen25_unsloth.py \
    --train-file pedagogy_dataset_train.jsonl \
    --val-file pedagogy_dataset_val.jsonl \
    --output-dir qwen25-1.5b-tokipona-unsloth
"""

from __future__ import annotations

import argparse
import os
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsloth LoRA fine-tuning for Qwen2.5-1.5B-Instruct on chat JSONL",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model HF")
    parser.add_argument("--train-file", default="pedagogy_dataset_train.jsonl", help="Path to train JSONL")
    parser.add_argument("--val-file", default="pedagogy_dataset_val.jsonl", help="Path to validation JSONL")
    parser.add_argument("--output-dir", default="qwen25-1.5b-tokipona-unsloth", help="Output dir")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-steps", type=int, default=50, help="Save/eval interval in steps")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a checkpoint directory to resume from",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop if eval does not improve for N evals (0 to disable)",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum eval improvement to reset early stopping patience",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading (enabled by default with Unsloth)",
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

    # Import Unsloth first so it can patch TRL/Transformers consistently.
    # Otherwise, mixed class instances can trigger a to_dict round-trip that
    # rewrites eos_token to the '<EOS_TOKEN>' placeholder.
    try:
        from unsloth import FastLanguageModel
    except Exception as exc:
        raise RuntimeError(
            "Unsloth is not installed in this environment. "
            "Install with: pip install unsloth"
        ) from exc
    import torch
    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if not os.path.exists(args.val_file):
        raise FileNotFoundError(f"Validation file not found: {args.val_file}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    load_in_4bit = not args.no_4bit

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    # TRL can fail if tokenizer.eos_token is a placeholder not present in vocab.
    # Force EOS/PAD from eos_token_id, which is reliable for Qwen tokenizers.
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id; cannot configure SFT EOS token.")
    eos_token = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
    if eos_token is None:
        raise RuntimeError("Unable to resolve eos_token from eos_token_id.")
    tokenizer.eos_token = eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
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
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit" if load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
        max_length=args.max_length,
        packing=False,
    )

    callbacks = None
    if args.early_stopping_patience > 0:
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
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
