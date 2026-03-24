#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a pedagogical conversational JSONL dataset from Tatoeba CSV files.

Goal:
- Build French <-> Toki Pona sentence pairs from sentences.csv + links.csv
- Convert them into short teaching dialogues for conversational fine-tuning
- Emit records that match schema.json (strict pedagogical schema)

Usage example:
  python generate_pedagogical_dataset.py \
    --sentences sentences.csv \
    --links links.csv \
    --output pedagogy_dataset.jsonl \
    --depth 3 \
    --max-samples 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

SentenceMap = Dict[int, Dict[str, str]]
Pair = Tuple[str, str]  # (fr, tok)


SYSTEM_PROMPT = (
    "Tu es un professeur de toki pona pour debutants francophones. "
    "Tu enseignes de facon orale guidee: etapes courtes, pratique active, "
    "correction bienveillante, recapitulatif frequent, sans jargon inutile."
)

# Light profanity / low-pedagogical-value filter for French examples.
BLOCKLIST_FR = {
    "merde",
    "putain",
    "connard",
    "connasse",
    "salope",
    "nique",
    "ta gueule",
    "casse-toi",
    "degage",
    "foutre",
}

# Keep allowed toki pona letters plus punctuation/spaces.
TOKI_PONA_CHARS_RE = re.compile(r"^[a-zA-Z\s.,!?':;\-]+$")
MULTISPACE_RE = re.compile(r"\s+")
TERMINAL_PUNCT_RE = re.compile(r"[\s.!?]+$")


def normalize_text(text: str) -> str:
    text = text.strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def contains_blocked_french(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in BLOCKLIST_FR)


def words_count(text: str) -> int:
    return len([w for w in text.split(" ") if w])


def clean_terminal_punctuation(text: str) -> str:
    """Normalize sentence-ending punctuation to avoid duplicated marks in templates."""
    text = normalize_text(text)
    text = TERMINAL_PUNCT_RE.sub("", text)
    return text


def is_beginner_friendly_french(text: str) -> bool:
    """Reject sentences that are likely too advanced for beginner oral lessons."""
    lower = text.lower()

    if any(mark in text for mark in [";", "(", ")", "\""]):
        return False

    # Typical complexity indicators for early-stage learning dialogues.
    advanced_markers = [
        "si ",
        "quoique",
        "cependant",
        "neanmoins",
        "pourtant",
        "j'",
        "aurais",
        "serais",
        "subjonctif",
    ]
    if any(marker in lower for marker in advanced_markers):
        return False

    return True


def is_reasonable_toki(text: str) -> bool:
    if not TOKI_PONA_CHARS_RE.match(text):
        return False
    if len(text) < 2 or len(text) > 120:
        return False
    return True


def load_sentences_selective(sentences_file: Path, target_langs: Set[str]) -> Tuple[SentenceMap, Dict[str, int]]:
    sentences: SentenceMap = {}
    lang_count: Dict[str, int] = defaultdict(int)

    with sentences_file.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            try:
                sent_id = int(row[0])
            except ValueError:
                continue

            lang = row[1].strip()
            text = normalize_text(row[2])
            if not text or lang not in target_langs:
                continue

            sentences[sent_id] = {"lang": lang, "text": text}
            lang_count[lang] += 1

    return sentences, lang_count


def load_links_selective(links_file: Path, sentence_ids: Iterable[int]) -> Dict[int, Set[int]]:
    sentence_ids_set = set(sentence_ids)
    links: Dict[int, Set[int]] = defaultdict(set)

    with links_file.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            try:
                id1, id2 = int(row[0]), int(row[1])
            except ValueError:
                continue

            # Keep edges where at least one endpoint belongs to our loaded sentence set.
            if id1 in sentence_ids_set or id2 in sentence_ids_set:
                links[id1].add(id2)
                links[id2].add(id1)

    return links


def find_target_ids(start_id: int, target_lang: str, sentences: SentenceMap, links: Dict[int, Set[int]], max_depth: int) -> Set[int]:
    if start_id not in sentences:
        return set()

    results: Set[int] = set()
    visited: Set[Tuple[int, str]] = {(start_id, sentences[start_id]["lang"])}
    queue = deque([(start_id, 0)])

    while queue:
        current_id, depth = queue.popleft()
        if depth > max_depth:
            continue

        for neighbor_id in links.get(current_id, set()):
            if neighbor_id not in sentences:
                continue

            neighbor_lang = sentences[neighbor_id]["lang"]
            key = (neighbor_id, neighbor_lang)
            if key in visited:
                continue

            visited.add(key)

            if neighbor_lang == target_lang:
                results.add(neighbor_id)

            if depth < max_depth:
                queue.append((neighbor_id, depth + 1))

    return results


def build_french_toki_pairs(
    sentences: SentenceMap,
    links: Dict[int, Set[int]],
    depth: int,
    max_source_sentences: int,
) -> Set[Pair]:
    pairs: Set[Pair] = set()

    toki_ids = [sid for sid, s in sentences.items() if s["lang"] == "tok"]
    if max_source_sentences > 0:
        random.shuffle(toki_ids)
        toki_ids = toki_ids[:max_source_sentences]

    total = len(toki_ids)

    for i, tok_id in enumerate(toki_ids):
        if total > 0 and i % max(1, total // 20) == 0:
            print(f"  progression paires: {int((i / total) * 100)}% ({i:,}/{total:,})")

        tok_text = sentences[tok_id]["text"]
        fra_ids = find_target_ids(tok_id, "fra", sentences, links, max_depth=depth)

        for fra_id in fra_ids:
            fra_text = sentences[fra_id]["text"]
            if fra_text != tok_text:
                pairs.add((fra_text, tok_text))

    return pairs


def filter_pairs(
    pairs: Iterable[Pair],
    min_words_fr: int,
    max_words_fr: int,
    min_words_tok: int,
    max_words_tok: int,
) -> List[Pair]:
    filtered: List[Pair] = []
    seen: Set[Tuple[str, str]] = set()

    for fr_text, tok_text in pairs:
        fr_text = normalize_text(fr_text)
        tok_text = normalize_text(tok_text)

        if contains_blocked_french(fr_text):
            continue
        if not is_beginner_friendly_french(fr_text):
            continue
        if not is_reasonable_toki(tok_text):
            continue

        fr_wc = words_count(fr_text)
        tok_wc = words_count(tok_text)
        if not (min_words_fr <= fr_wc <= max_words_fr):
            continue
        if not (min_words_tok <= tok_wc <= max_words_tok):
            continue

        key = (fr_text.lower(), tok_text.lower())
        if key in seen:
            continue
        seen.add(key)
        filtered.append((fr_text, tok_text))

    return filtered


VALID_LEVELS = ("A0", "A1", "A2", "B1")


def infer_level(fr_text: str, tok_text: str) -> str:
    complexity = words_count(fr_text) + words_count(tok_text)
    if complexity <= 6:
        return "A0"
    if complexity <= 12:
        return "A1"
    if complexity <= 18:
        return "A2"
    return "B1"


def filter_pairs_by_level(pairs: List[Pair], target_levels: Set[str]) -> List[Pair]:
    """Keep only pairs whose inferred CECRL level is in target_levels."""
    if not target_levels or target_levels == set(VALID_LEVELS):
        return pairs
    return [(fr, tok) for fr, tok in pairs if infer_level(fr, tok) in target_levels]


def make_messages(lesson_type: str, fr_text: str, tok_text: str) -> List[Dict[str, str]]:
    fr_clean = clean_terminal_punctuation(fr_text)
    tok_clean = clean_terminal_punctuation(tok_text)

    if lesson_type == "guided_dialogue":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Je debute. Comment dire en toki pona: {fr_clean} ?"},
            {
                "role": "assistant",
                "content": (
                    f"On procede pas a pas. Pour dire \"{fr_clean}\", tu peux dire: {tok_clean}. "
                    "Repete la phrase a voix haute."
                ),
            },
            {"role": "user", "content": f"{tok_clean}."},
            {
                "role": "assistant",
                "content": (
                    "Bien joue. Mini recap: garde une prononciation claire et des phrases courtes. "
                    "Pret pour une nouvelle phrase du meme type ?"
                ),
            },
        ]

    if lesson_type == "pattern_drill":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Fais-moi pratiquer avec un petit exercice."},
            {
                "role": "assistant",
                "content": (
                    f"Exercice rapide. Transforme cette idee en toki pona: \"{fr_clean}\". "
                    "Ecris seulement la phrase."
                ),
            },
            {"role": "user", "content": f"{tok_clean}."},
            {
                "role": "assistant",
                "content": (
                    f"Correct. Reponse attendue: {tok_clean}. "
                    "Refais-la une fois, lentement, pour fixer la structure."
                ),
            },
        ]

    if lesson_type == "error_correction":
        wrong_attempt = tok_clean.replace(" li ", " ", 1).strip() if " li " in f" {tok_clean} " else tok_clean
        if wrong_attempt == tok_clean:
            wrong_attempt = tok_clean + "?"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Je veux dire \"{fr_clean}\". J'ai essaye: {wrong_attempt} "
                    "C'est correct ?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"Presque. Forme correcte: {tok_clean}. "
                    "Repete la version correcte et garde ce modele."
                ),
            },
            {"role": "user", "content": f"{tok_clean}."},
            {"role": "assistant", "content": "Parfait. La phrase est correcte maintenant."},
        ]

    if lesson_type == "review_recap":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "On fait une revision rapide ?"},
            {
                "role": "assistant",
                "content": (
                    f"Revision: donne la phrase toki pona pour \"{fr_clean}\". "
                    "Puis relis-la une deuxieme fois."
                ),
            },
            {"role": "user", "content": f"{tok_clean}."},
            {
                "role": "assistant",
                "content": (
                    f"Excellent. Reponse valide: {tok_clean}. "
                    "On continue avec une phrase tres proche au prochain tour."
                ),
            },
        ]

    # translation_with_explanation
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Traduis en toki pona et explique tres simplement: {fr_clean}",
        },
        {
            "role": "assistant",
            "content": (
                f"Traduction: {tok_clean}. "
                "Explication courte: on garde une phrase simple et directe, sans surcharge. "
                "Dis-la une fois pour t'entrainer."
            ),
        },
    ]


def lesson_plan_for_type(lesson_type: str) -> Tuple[str, List[str], bool, bool, bool, int, str]:
    if lesson_type == "guided_dialogue":
        return (
            "dialogue_guide",
            ["vocabulaire_de_base", "reformulation"],
            True,
            False,
            True,
            1,
            "Apprendre une phrase utile avec repetition active.",
        )
    if lesson_type == "pattern_drill":
        return (
            "transformation",
            ["ordre_des_mots", "vocabulaire_de_base"],
            True,
            True,
            True,
            2,
            "Renforcer un patron de phrase par production guidee.",
        )
    if lesson_type == "error_correction":
        return (
            "correction",
            ["correction", "reformulation"],
            True,
            True,
            False,
            2,
            "Corriger une tentative et stabiliser la forme correcte.",
        )
    if lesson_type == "review_recap":
        return (
            "revision",
            ["reformulation", "vocabulaire_de_base"],
            True,
            False,
            True,
            1,
            "Reviser une structure deja vue avec rappel actif.",
        )
    return (
        "traduction_expliquee",
        ["vocabulaire_de_base", "ordre_des_mots"],
        True,
        False,
        False,
        2,
        "Traduire une phrase et donner une explication pedagogique courte.",
    )


def build_sample(index: int, fr_text: str, tok_text: str, lesson_type: str) -> dict:
    lesson_number = ((index - 1) // 50) + 1
    lesson_id = f"L{lesson_number:03d}"
    sample_id = f"{lesson_id}_{index:04d}"
    level = infer_level(fr_text, tok_text)

    topic, skills, expects_prod, includes_corr, includes_recap, difficulty, objective = lesson_plan_for_type(lesson_type)

    sample = {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "lesson": {
            "lesson_id": lesson_id,
            "lesson_type": lesson_type,
            "level": level,
            "topic": topic,
            "objective": objective,
        },
        "language": {
            "source_language": "fr",
            "target_language": "tok",
        },
        "messages": make_messages(lesson_type, fr_text, tok_text),
        "pedagogy": {
            "skills": skills,
            "difficulty": difficulty,
            "expects_student_production": expects_prod,
            "includes_correction": includes_corr,
            "includes_recap": includes_recap,
        },
        "quality": {
            "safe_for_beginners": True,
            "no_vulgarity": True,
            "no_hate": True,
            "max_new_concepts": 1 if difficulty <= 2 else 2,
        },
    }

    return sample


def level_distribution(pairs: Sequence[Pair]) -> str:
    """Return a one-line summary of level counts."""
    from collections import Counter
    counts: Counter = Counter(infer_level(fr, tok) for fr, tok in pairs)
    total = sum(counts.values())
    parts = []
    for lvl in VALID_LEVELS:
        n = counts.get(lvl, 0)
        parts.append(f"{lvl}={n} ({n / total * 100:.0f}%)" if total else f"{lvl}=0")
    return "  " + ", ".join(parts)


def generate_samples(pairs: Sequence[Pair], max_samples: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    lesson_types = [
        "guided_dialogue",
        "pattern_drill",
        "error_correction",
        "review_recap",
        "translation_with_explanation",
    ]

    output: List[dict] = []
    for i, (fr_text, tok_text) in enumerate(pairs[:max_samples], start=1):
        lesson_type = lesson_types[(i - 1) % len(lesson_types)]
        output.append(build_sample(i, fr_text, tok_text, lesson_type))

    return output


def write_jsonl(records: Sequence[dict], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a pedagogical conversational JSONL dataset for French speakers learning Toki Pona."
    )
    parser.add_argument("--sentences", default="sentences.csv", help="Path to sentences.csv")
    parser.add_argument("--links", default="links.csv", help="Path to links.csv")
    parser.add_argument("--output", default="pedagogy_dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--depth", type=int, default=3, help="BFS depth for indirect links (default: 3)")
    parser.add_argument(
        "--max-source-sentences",
        type=int,
        default=0,
        help="Limit number of toki source sentences scanned (0 = all)",
    )
    parser.add_argument("--max-samples", type=int, default=5000, help="Max output samples (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--min-words-fr", type=int, default=1, help="Min words in French sentence")
    parser.add_argument("--max-words-fr", type=int, default=8, help="Max words in French sentence")
    parser.add_argument("--min-words-tok", type=int, default=1, help="Min words in toki pona sentence")
    parser.add_argument("--max-words-tok", type=int, default=12, help="Max words in toki pona sentence")
    parser.add_argument(
        "--level",
        default="all",
        help=(
            "Target CECRL level(s) to include. "
            "Accepts a single level (A0, A1, A2, B1), "
            "a comma-separated list (A0,A1), or 'all' (default). "
            "Example: --level A0,A1"
        ),
    )
    return parser.parse_args()


def parse_level_arg(level_arg: str) -> Set[str]:
    """Parse and validate the --level argument. Returns a set of level strings."""
    if level_arg.strip().lower() == "all":
        return set(VALID_LEVELS)
    requested = {v.strip().upper() for v in level_arg.split(",") if v.strip()}
    unknown = requested - set(VALID_LEVELS)
    if unknown:
        raise ValueError(
            f"Unknown level(s): {', '.join(sorted(unknown))}. "
            f"Valid values: {', '.join(VALID_LEVELS)}, all"
        )
    return requested


def main() -> int:
    args = parse_args()

    start = time.time()
    sentences_file = Path(args.sentences)
    links_file = Path(args.links)
    output_file = Path(args.output)

    if args.depth < 1:
        print("ERROR: --depth must be >= 1")
        return 1
    if args.max_samples < 1:
        print("ERROR: --max-samples must be >= 1")
        return 1

    try:
        target_levels = parse_level_arg(args.level)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    level_label = "all" if target_levels == set(VALID_LEVELS) else ", ".join(sorted(target_levels))
    print(f"Target level(s): {level_label}")

    if not sentences_file.exists():
        print(f"ERROR: file not found: {sentences_file}")
        return 1
    if not links_file.exists():
        print(f"ERROR: file not found: {links_file}")
        return 1

    print("[1/5] Loading selective sentences...")
    sentences, lang_count = load_sentences_selective(sentences_file, {"tok", "fra", "eng"})
    print(f"  Loaded sentences: {len(sentences):,}")
    print(f"  by language: tok={lang_count.get('tok', 0):,}, fra={lang_count.get('fra', 0):,}, eng={lang_count.get('eng', 0):,}")

    if lang_count.get("tok", 0) == 0 or lang_count.get("fra", 0) == 0:
        print("ERROR: Missing required languages (tok and fra)")
        return 1

    print("[2/5] Loading links...")
    links = load_links_selective(links_file, sentences.keys())
    print(f"  Loaded link nodes: {len(links):,}")

    print("[3/5] Building French-Toki pairs...")
    raw_pairs = build_french_toki_pairs(
        sentences,
        links,
        depth=args.depth,
        max_source_sentences=args.max_source_sentences,
    )
    print(f"  Raw pairs: {len(raw_pairs):,}")

    print("[4/5] Filtering pairs...")
    filtered_pairs = filter_pairs(
        raw_pairs,
        min_words_fr=args.min_words_fr,
        max_words_fr=args.max_words_fr,
        min_words_tok=args.min_words_tok,
        max_words_tok=args.max_words_tok,
    )
    print(f"  Filtered pairs: {len(filtered_pairs):,}")

    if target_levels != set(VALID_LEVELS):
        print(f"[4b] Filtering pairs by level ({level_label})...")
        filtered_pairs = filter_pairs_by_level(filtered_pairs, target_levels)
        print(f"  Pairs after level filter: {len(filtered_pairs):,}")

    if not filtered_pairs:
        print(
            "ERROR: No pairs left after filtering. "
            "Try a different --level, relax word-count limits, or increase --depth."
        )
        return 1

    print("  Level distribution:")
    print(level_distribution(filtered_pairs))

    print("[5/5] Generating pedagogical dialogues...")
    records = generate_samples(filtered_pairs, max_samples=args.max_samples, seed=args.seed)
    write_jsonl(records, output_file)

    elapsed = time.time() - start
    print("Done")
    print(f"  Output: {output_file}")
    print(f"  Records: {len(records):,}")
    print(f"  Time: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
