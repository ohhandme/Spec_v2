"""Freeze explicit train/val/test prompt families for the P1 fixed probe.

Input JSONL rows require id, family_id, split, and either text or messages.
All tokenization uses one local tokenizer and an explicit Qwen thinking mode.
No automatic split assignment, download, or replacement of existing output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thinking-mode", choices=("enabled", "disabled"), required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.tokenizer.exists() or not args.input.exists():
        raise FileNotFoundError("tokenizer and input must exist locally")
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if args.output.exists() or meta_path.exists():
        raise FileExistsError("tokenized output and metadata paths must be new")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("tokenizer has no fixed chat template")
    seen_ids: set[str] = set()
    seen_tokens: set[tuple[int, ...]] = set()
    family_splits: dict[str, str] = {}
    counts = {"train": 0, "val": 0, "test": 0, "smoke": 0}
    lines: list[str] = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identifier, family, split = row.get("id"), row.get("family_id"), row.get("split")
        task_type = row.get("task_type", "unknown")
        if any(not isinstance(value, str) or not value for value in
               (identifier, family, task_type)) or identifier in seen_ids:
            raise ValueError(f"line {line_number}: invalid or duplicate ID/family/task")
        if split not in counts:
            raise ValueError(f"line {line_number}: invalid split")
        if family in family_splits and family_splits[family] != split:
            raise ValueError(f"line {line_number}: one family crosses splits")
        if "messages" in row:
            messages = row["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"line {line_number}: messages must be nonempty")
        else:
            content = row.get("text")
            if not isinstance(content, str) or not content:
                raise ValueError(f"line {line_number}: text must be nonempty")
            messages = [{"role": "user", "content": content}]
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=args.thinking_mode == "enabled",
        )
        if not isinstance(tokens, list) or not tokens or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise RuntimeError(f"line {line_number}: tokenizer returned invalid IDs")
        token_tuple = tuple(tokens)
        if token_tuple in seen_tokens:
            raise ValueError(f"line {line_number}: duplicate tokenized prompt")
        seen_ids.add(identifier)
        seen_tokens.add(token_tuple)
        family_splits[family] = split
        counts[split] += 1
        lines.append(json.dumps({
            "id": identifier, "family_id": family, "split": split,
            "task_type": task_type, "input_ids": tokens,
        }, ensure_ascii=False, separators=(",", ":")))
    if not counts["train"] or not counts["val"] or not counts["test"]:
        raise ValueError("locked train, val, and test splits must each be nonempty")
    output_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    metadata = {
        "tokenizer_path": str(args.tokenizer.resolve()),
        "input_sha256": _sha256(args.input.read_bytes()),
        "output_sha256": _sha256(output_bytes),
        "chat_template_sha256": _sha256(tokenizer.chat_template.encode("utf-8")),
        "thinking_mode": args.thinking_mode,
        "counts": counts,
        "family_count": len(family_splits),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(output_bytes)
    temporary.replace(args.output)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
