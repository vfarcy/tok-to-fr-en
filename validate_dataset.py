#!/usr/bin/env python3
"""
Validate a JSONL dataset against a JSON Schema.

Usage:
  python validate_dataset.py --jsonl pedagogy_train.jsonl --schema schema.json

Exit codes:
  0 -> all records are valid
  1 -> at least one record is invalid or a runtime error occurred
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def _load_schema(schema_path: Path) -> dict[str, Any]:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Schema is not valid JSON: {schema_path} (line {exc.lineno}, col {exc.colno})"
        ) from exc


def _build_validator(schema: dict[str, Any]):
    try:
        jsonschema = importlib.import_module("jsonschema")
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'jsonschema'. Install it with: pip install jsonschema"
        ) from exc

    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc

    return jsonschema.Draft202012Validator(schema)


def _format_path(error_path: Any) -> str:
    parts = list(error_path)
    if not parts:
        return "$"

    path = "$"
    for item in parts:
        if isinstance(item, int):
            path += f"[{item}]"
        else:
            path += f".{item}"
    return path


def validate_jsonl(
    jsonl_path: Path,
    validator,
    max_errors: int,
    skip_empty_lines: bool,
) -> int:
    if not jsonl_path.exists():
        print(f"ERROR: JSONL file not found: {jsonl_path}")
        return 1

    total_lines = 0
    checked_records = 0
    valid_records = 0
    invalid_records = 0
    parse_errors = 0
    schema_errors_shown = 0

    print(f"Validating: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            total_lines += 1
            line = raw_line.strip()

            if not line:
                if skip_empty_lines:
                    continue
                invalid_records += 1
                parse_errors += 1
                if schema_errors_shown < max_errors:
                    print(f"[line {line_number}] ERROR: Empty line is not allowed")
                    schema_errors_shown += 1
                continue

            checked_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_records += 1
                parse_errors += 1
                if schema_errors_shown < max_errors:
                    print(
                        f"[line {line_number}] JSON parse error: "
                        f"{exc.msg} (line {exc.lineno}, col {exc.colno})"
                    )
                    schema_errors_shown += 1
                continue

            errors = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
            if not errors:
                valid_records += 1
                continue

            invalid_records += 1
            if schema_errors_shown < max_errors:
                print(f"[line {line_number}] Schema validation failed:")
                for err in errors:
                    if schema_errors_shown >= max_errors:
                        break
                    location = _format_path(err.path)
                    print(f"  - {location}: {err.message}")
                    schema_errors_shown += 1

    print("\nSummary")
    print("-------")
    print(f"Total lines:          {total_lines}")
    print(f"Checked records:      {checked_records}")
    print(f"Valid records:        {valid_records}")
    print(f"Invalid records:      {invalid_records}")
    print(f"JSON parse errors:    {parse_errors}")

    if schema_errors_shown >= max_errors and invalid_records > 0:
        print(f"Displayed errors:     {max_errors} (max reached)")

    if invalid_records == 0:
        print("\nResult: VALID (all checked records match the schema)")
        return 0

    print("\nResult: INVALID (some records do not match the schema)")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a JSONL dataset against a JSON Schema (line by line)."
    )
    parser.add_argument(
        "--jsonl",
        "-j",
        required=True,
        help="Path to the JSONL dataset to validate",
    )
    parser.add_argument(
        "--schema",
        "-s",
        default="schema.json",
        help="Path to JSON Schema file (default: schema.json)",
    )
    parser.add_argument(
        "--max-errors",
        "-m",
        type=int,
        default=20,
        help="Maximum number of validation errors to print (default: 20)",
    )
    parser.add_argument(
        "--no-skip-empty-lines",
        action="store_true",
        help="Treat empty lines as errors (default behavior is to skip them)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_errors < 1:
        print("ERROR: --max-errors must be >= 1")
        return 1

    schema_path = Path(args.schema)
    jsonl_path = Path(args.jsonl)

    try:
        schema = _load_schema(schema_path)
        validator = _build_validator(schema)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    return validate_jsonl(
        jsonl_path=jsonl_path,
        validator=validator,
        max_errors=args.max_errors,
        skip_empty_lines=not args.no_skip_empty_lines,
    )


if __name__ == "__main__":
    sys.exit(main())
