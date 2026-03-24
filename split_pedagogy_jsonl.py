#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a pedagogical JSONL dataset into train/validation/test without data leakage.

This script reconstructs the underlying French/Toki Pona pair from the
conversation templates and keeps all records from the same pair in the same
split. That prevents near-duplicate dialogues from being spread across
train/validation/test.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TERMINAL_PUNCT_RE = re.compile(r"[\s.!?]+$")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def clean_terminal_punctuation(text: str) -> str:
    text = normalize_text(text)
    return TERMINAL_PUNCT_RE.sub("", text)


def load_jsonl(input_file: Path) -> List[dict]:
    records: List[dict] = []
    with input_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ligne {line_number}: JSON invalide ({exc})") from exc
    return records


def get_message_content(record: dict, role: str, occurrence: int = 1) -> str:
    count = 0
    for message in record.get("messages", []):
        if message.get("role") == role:
            count += 1
            if count == occurrence:
                return str(message.get("content", ""))
    return ""


def extract_pair_from_record(record: dict) -> Tuple[str, str]:
    lesson = record.get("lesson", {})
    lesson_type = lesson.get("lesson_type", "")

    user_1 = get_message_content(record, "user", 1)
    user_2 = get_message_content(record, "user", 2)
    assistant_1 = get_message_content(record, "assistant", 1)

    fr_text = ""
    tok_text = ""

    if lesson_type == "guided_dialogue":
        marker = "Je debute. Comment dire en toki pona:"
        if marker in user_1:
            fr_text = user_1.split(marker, 1)[1].strip()
        tok_marker = "tu peux dire:"
        if tok_marker in assistant_1:
            tok_text = assistant_1.split(tok_marker, 1)[1].split("Repete la phrase", 1)[0].strip()
        elif user_2:
            tok_text = user_2

    elif lesson_type == "pattern_drill":
        match = re.search(r'idee en toki pona: "(.*)"\.? Ecris seulement la phrase\.?$', assistant_1)
        if match:
            fr_text = match.group(1)
        tok_text = user_2 or get_message_content(record, "user", 1)

    elif lesson_type == "error_correction":
        match = re.search(r'Je veux dire "(.*)"\. J\'ai essaye:', user_1)
        if match:
            fr_text = match.group(1)
        tok_marker = "Forme correcte:"
        if tok_marker in assistant_1:
            tok_text = assistant_1.split(tok_marker, 1)[1].split("Repete la version correcte", 1)[0].strip()
        elif user_2:
            tok_text = user_2

    elif lesson_type == "review_recap":
        match = re.search(r'phrase toki pona pour "(.*)"\. Puis relis-la', assistant_1)
        if match:
            fr_text = match.group(1)
        tok_text = user_2 or get_message_content(record, "user", 1)

    elif lesson_type == "translation_with_explanation":
        marker = "Traduis en toki pona et explique tres simplement:"
        if marker in user_1:
            fr_text = user_1.split(marker, 1)[1].strip()
        tok_marker = "Traduction:"
        if tok_marker in assistant_1:
            tok_text = assistant_1.split(tok_marker, 1)[1].split("Explication courte", 1)[0].strip()

    fr_text = clean_terminal_punctuation(fr_text)
    tok_text = clean_terminal_punctuation(tok_text)

    if fr_text and tok_text:
        return fr_text.lower(), tok_text.lower()

    # Fallback for future template variants: keep the entire dialogue together.
    fallback = json.dumps(record.get("messages", []), ensure_ascii=False, sort_keys=True)
    return "__fallback__", normalize_text(fallback).lower()


def build_groups(records: Sequence[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for record in records:
        groups[extract_pair_from_record(record)].append(record)
    return groups


def split_grouped_records(
    records: Sequence[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, int]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Les ratios doivent totaliser 1.0, recu: {total}")

    groups = build_groups(records)
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)

    train_group_count = int(len(group_items) * train_ratio)
    val_group_count = int(len(group_items) * val_ratio)

    train_items = group_items[:train_group_count]
    val_items = group_items[train_group_count:train_group_count + val_group_count]
    test_items = group_items[train_group_count + val_group_count:]

    def flatten(items: Iterable[Tuple[Tuple[str, str], List[dict]]]) -> List[dict]:
        output: List[dict] = []
        for _, group_records in items:
            output.extend(group_records)
        return output

    train_records = flatten(train_items)
    val_records = flatten(val_items)
    test_records = flatten(test_items)

    stats = {
        "records_total": len(records),
        "groups_total": len(group_items),
        "groups_train": len(train_items),
        "groups_val": len(val_items),
        "groups_test": len(test_items),
        "records_train": len(train_records),
        "records_val": len(val_records),
        "records_test": len(test_records),
    }
    return train_records, val_records, test_records, stats


def write_jsonl(records: Sequence[dict], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_report(input_file: Path, train_file: Path, val_file: Path, test_file: Path, stats: Dict[str, int]) -> None:
    total_records = stats["records_total"]
    print("=" * 72)
    print(f"Split groupe sans fuite: {input_file.name}")
    print("=" * 72)
    print(f"Groupes extraits: {stats['groups_total']:,}")
    print(f"Train: {stats['records_train']:,} enregistrements sur {stats['groups_train']:,} groupes")
    print(f"Val:   {stats['records_val']:,} enregistrements sur {stats['groups_val']:,} groupes")
    print(f"Test:  {stats['records_test']:,} enregistrements sur {stats['groups_test']:,} groupes")
    if total_records:
        print()
        print(f"Train: {stats['records_train'] / total_records * 100:.1f}%")
        print(f"Val:   {stats['records_val'] / total_records * 100:.1f}%")
        print(f"Test:  {stats['records_test'] / total_records * 100:.1f}%")
    print()
    print(f"Fichier train: {train_file.name}")
    print(f"Fichier val:   {val_file.name}")
    print(f"Fichier test:  {test_file.name}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split un dataset pedagogique JSONL en train/val/test sans fuite entre paires fr/tok.",
    )
    parser.add_argument("input_file", help="Chemin du fichier JSONL pedagogique source")
    parser.add_argument("--train", type=float, default=0.8, help="Ratio train (defaut: 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Ratio validation (defaut: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Ratio test (defaut: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Graine aleatoire (defaut: 42)")
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefixe optionnel pour les fichiers de sortie, ex: splits/",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Fichier non trouve: {input_path}")
        return 1

    try:
        records = load_jsonl(input_path)
        train_records, val_records, test_records, stats = split_grouped_records(
            records,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Erreur: {exc}")
        return 1

    prefix = Path(args.prefix) if args.prefix else Path("")
    if str(prefix) not in ("", "."):
        prefix.mkdir(parents=True, exist_ok=True)

    train_file = prefix / f"{input_path.stem}_train{input_path.suffix}"
    val_file = prefix / f"{input_path.stem}_val{input_path.suffix}"
    test_file = prefix / f"{input_path.stem}_test{input_path.suffix}"

    write_jsonl(train_records, train_file)
    write_jsonl(val_records, val_file)
    write_jsonl(test_records, test_file)
    print_report(input_path, train_file, val_file, test_file, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())