#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat interactif avec le modèle fine-tuné (LoRA checkpoint).

Usage:
  python chat_model.py
    python chat_model.py --adapter qwen25-1.5b-tokipona-lora-v2
  python chat_model.py --system "Tu es un tuteur de toki pona."
"""
import argparse
import sys
from threading import Thread

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


DRILL_SEQUENCE = [
    ("Je parle.", "mi toki"),
    ("Je mange.", "mi moku"),
    ("Je dors.", "mi lape"),
    ("Je viens.", "mi kama"),
    ("Je veux manger.", "mi wile moku"),
]

ACK_WORDS = {
    "oui",
    "oui.",
    "ok",
    "d'accord",
    "daccord",
    "laquelle",
    "laquelle?",
    "laquelle ?",
    "alors",
    "alors?",
    "alors ?",
    "ensuite",
    "la suite",
}


def parse_args():
    p = argparse.ArgumentParser(description="Chat interactif avec le modèle LoRA fine-tuné")
    p.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Modèle de base HuggingFace",
    )
    p.add_argument(
        "--adapter",
        default="qwen25-1.5b-tokipona-lora-v2",
        help="Chemin vers le checkpoint LoRA",
    )
    p.add_argument(
        "--system",
        default=(
            "Tu es un professeur de toki pona pour debutants francophones. "
            "Tu enseignes de facon orale guidee: etapes courtes, pratique active, "
            "correction bienveillante, recapitulatif frequent, sans jargon inutile. "
            "Regles strictes: n'invente jamais une traduction ni une regle. "
            "Si tu es incertain, dis-le explicitement et demande une clarification "
            "au lieu d'affirmer. N'annonce pas \"correct\" si tu n'es pas sur. "
            "Utilise le mot 'phrase' ou 'exemple', jamais 'meme'. "
            "Ne repete pas la meme phrase deux tours de suite. "
            "Si l'utilisateur dit 'oui' ou 'd'accord', propose une nouvelle phrase differente."
        ),
        help="Prompt système",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=200,
        help="Longueur max de la réponse générée",
    )
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="Température de sampling (0 = déterministe)",
    )
    p.add_argument(
        "--top-p", type=float, default=1.0,
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

    # Avoid noisy warnings from stale generation defaults when running greedy decoding.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.top_k = None
        model.generation_config.top_p = None

    device = next(model.parameters()).device
    print(f"Modèle prêt sur {device}.")
    print("Tape 'quit' ou 'exit' pour quitter, 'reset' pour recommencer la conversation.\n")

    history = [{"role": "system", "content": args.system}]
    last_assistant = ""
    expected_phrase = ""
    drill_index = 0

    def next_drill_message() -> str:
        nonlocal drill_index, expected_phrase
        fr, tok = DRILL_SEQUENCE[drill_index % len(DRILL_SEQUENCE)]
        drill_index += 1
        expected_phrase = tok.lower().strip()
        return f'Nouvelle phrase: "{fr}" se dit: {tok}. Repete a voix haute: {tok}.'

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
            last_assistant = ""
            expected_phrase = ""
            drill_index = 0
            print("[Conversation réinitialisée]")
            continue

        user_norm = user_input.lower().strip()

        # Guardrail anti-boucle: si l'utilisateur acquiesce sans contenu,
        # proposer explicitement une nouvelle phrase valide au lieu de répéter.
        if user_norm in ACK_WORDS:
            forced = next_drill_message()
            print(f"Modèle: {forced}\n")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": forced})
            last_assistant = forced
            continue

        # Si l'utilisateur répète la phrase attendue, valider proprement.
        if expected_phrase and user_norm == expected_phrase:
            forced = f"Parfait. Phrase valide: {expected_phrase}. {next_drill_message()}"
            print(f"Modèle: {forced}\n")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": forced})
            last_assistant = forced
            continue

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        in_len = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        do_sample = args.temperature > 0
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
        if do_sample:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print("Modèle: ", end="", flush=True)
        chunks = []
        for chunk in streamer:
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        thread.join()

        answer = "".join(chunks).strip()

        # Filet de securite: si le modele recopie literalement une relance vide,
        # on remplace par une relance pedagogique valide.
        if "phrase tres proche est" in answer.lower() and user_norm in ACK_WORDS:
            answer = next_drill_message()

        print("\n")
        history.append({"role": "assistant", "content": answer})
        last_assistant = answer


if __name__ == "__main__":
    sys.exit(main())
