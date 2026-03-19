#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABEL_MAP = {
    "-1": "NEG",
    "0": "NEU",
    "1": "POS",
    "NEG": "NEG",
    "NEU": "NEU",
    "POS": "POS",
}


def sanitize_text(text: str) -> str:
    # Keep TSV shape stable by removing embedded tabs/newlines from sentences.
    return " ".join(text.replace("\t", " ").split())


def extract_fields(record: dict, line_number: int) -> tuple[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Line {line_number}: missing or invalid 'messages' list")

    text = None
    raw_label = None

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if role == "user" and text is None:
            text = content
        elif role == "assistant" and raw_label is None:
            raw_label = content

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Line {line_number}: missing user sentence text")
    if not isinstance(raw_label, str):
        raise ValueError(f"Line {line_number}: missing assistant label")

    normalized = raw_label.strip().upper()
    if normalized not in LABEL_MAP:
        raise ValueError(
            f"Line {line_number}: unsupported label {raw_label!r}. "
            "Expected -1/0/1 or NEG/NEU/POS."
        )

    return sanitize_text(text), LABEL_MAP[normalized]


def convert_jsonl_to_tsv(input_path: Path, output_path: Path) -> int:
    rows_written = 0

    with input_path.open("r", encoding="utf-8") as in_f, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        writer = csv.writer(out_f, delimiter="\t")

        for line_number, line in enumerate(in_f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                text, label = extract_fields(record, line_number)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON ({exc})") from exc

            writer.writerow([text, label])
            rows_written += 1

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert chat-style sentiment JSONL to TSV with two columns: "
            "sentence<TAB>label."
        )
    )
    parser.add_argument("input_jsonl", type=Path, help="Path to input .jsonl file")
    parser.add_argument(
        "output_tsv",
        type=Path,
        nargs="?",
        help="Path to output .tsv file (default: same name as input)",
    )
    args = parser.parse_args()

    input_path = args.input_jsonl
    output_path = args.output_tsv or input_path.with_suffix(".tsv")

    count = convert_jsonl_to_tsv(input_path, output_path)
    print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
