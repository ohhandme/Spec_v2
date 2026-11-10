"""Freeze raw chat prompts into the token-ID JSONL used by the pilot.

Input rows: {"id":"...", "split":"train|test", "text":"..."} or an
explicit Hugging Face chat-template ``messages`` list. Tokenizer is local only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".meta.json").exists():
        raise FileExistsError("output and metadata paths must be new")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise ValueError("tokenizer has no chat template")
    lines: list[str] = []
    identifiers: set[str] = set()
    token_sequences: set[tuple[int, ...]] = set()
    counts = {"train": 0, "test": 0}
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identifier, split = row.get("id"), row.get("split")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError(f"line {line_number}: invalid or repeated id")
        if split not in counts:
            raise ValueError(f"line {line_number}: split must be train or test")
        if "messages" in row:
            messages = row["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"line {line_number}: messages must be a nonempty list")
        else:
            text = row.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"line {line_number}: text must be nonempty")
            messages = [{"role": "user", "content": text}]
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if not isinstance(tokens, list) or not tokens or any(type(x) is not int for x in tokens):
            raise RuntimeError(f"line {line_number}: tokenizer returned invalid IDs")
        token_tuple = tuple(tokens)
        if token_tuple in token_sequences:
            raise ValueError(f"line {line_number}: repeated tokenized prompt")
        identifiers.add(identifier)
        token_sequences.add(token_tuple)
        counts[split] += 1
        lines.append(
            json.dumps(
                {"id": identifier, "split": split, "input_ids": tokens},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    if not counts["train"] or not counts["test"]:
        raise ValueError("both train and test splits need prompts")
    output_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    metadata = {
        "tokenizer_path": str(args.tokenizer.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        "counts": counts,
        "tokenizer_default_chat_template_options": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(output_bytes)
    temporary.replace(args.output)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
