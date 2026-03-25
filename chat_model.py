#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat interactif avec le modèle fine-tuné (LoRA checkpoint).

Usage:
  python chat_model.py
  python chat_model.py --adapter qwen25-1.5b-tokipona-lora/checkpoint-1800
  python chat_model.py --system "Tu es un tuteur de toki pona."
"""
import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Chat interactif avec le modèle LoRA fine-tuné")
    p.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Modèle de base HuggingFace",
    )
    p.add_argument(
        "--adapter",
        default="qwen25-1.5b-tokipona-lora/checkpoint-1800",
        help="Chemin vers le checkpoint LoRA",
    )
    p.add_argument(
        "--system",
        default=(
            "Tu es un professeur de toki pona pour debutants francophones. "
            "Tu enseignes de facon orale guidee: etapes courtes, pratique active, "
            "correction bienveillante, recapitulatif frequent, sans jargon inutile."
        ),
        help="Prompt système",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=200,
        help="Longueur max de la réponse générée",
    )
    p.add_argument(
        "--temperature", type=float, default=0.7,
        help="Température de sampling (0 = déterministe)",
    )
    p.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p nucleus sampling",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Chargement du tokenizer ({args.base_model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Chargement du modèle de base...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )

    print(f"Chargement de l'adaptateur LoRA ({args.adapter})...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    device = next(model.parameters()).device
    print(f"Modèle prêt sur {device}.")
    print("Tape 'quit' ou 'exit' pour quitter, 'reset' pour recommencer la conversation.\n")

    history = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = input("Toi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Au revoir.")
            break
        if user_input.lower() == "reset":
            history = [{"role": "system", "content": args.system}]
            print("[Conversation réinitialisée]")
            continue

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        in_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
        print(f"Modèle: {answer}\n")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    sys.exit(main())
