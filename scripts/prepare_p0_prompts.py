"""Freeze disjoint user-only UltraChat prompts for the real Qwen P0/P1 pilot.

The source rows may contain reference assistant answers. This script never
passes them into the model prompt or stores them in the output dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

from transformers import AutoTokenizer


SPLITS = {"train": "train", "validation": "val", "evaluation": "test"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_family(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\b\d+(?:[.,]\d+)*\b", "<number>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")
    source_manifest = json.loads((args.source_dir / "manifest.json").read_text(encoding="utf-8"))
    source_dataset = source_manifest["dataset"]
    source_revision = source_manifest["revision"]
    if source_dataset != "HuggingFaceH4/ultrachat_200k" or source_revision != "8049631c405ae6576f93f445c6b8166f76f5505a":
        raise ValueError("source dataset or revision differs from the locked pilot")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template")
    rows: list[dict] = []
    seen_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    seen_token_ids: set[tuple[int, ...]] = set()
    seen_family: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for source_split, output_split in SPLITS.items():
        path = args.source_dir / f"{source_split}.jsonl"
        source_hashes[path.name] = sha256_file(path)
        if source_manifest["files"][source_split]["sha256"] != source_hashes[path.name]:
            raise ValueError(f"source file differs from its prior audited manifest: {path.name}")
        counts[output_split] = 0
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                source_row = json.loads(line)
                identifier = source_row.get("id")
                source_id = source_row.get("source_id")
                messages = source_row.get("messages")
                if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
                    raise ValueError(f"{path.name}:{line_number}: duplicate or invalid id")
                if not isinstance(source_id, str) or not source_id:
                    raise ValueError(f"{path.name}:{line_number}: missing source_id")
                if source_id in seen_source_ids:
                    raise ValueError(f"{path.name}:{line_number}: duplicate source_id")
                if (source_row.get("source_dataset") != source_dataset
                        or source_row.get("source_revision") != source_revision
                        or source_row.get("pilot_split") != source_split):
                    raise ValueError(f"{path.name}:{line_number}: source provenance mismatch")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"{path.name}:{line_number}: missing messages")
                if messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str):
                    raise ValueError(f"{path.name}:{line_number}: first message is not a user prompt")
                user_text = messages[0]["content"]
                if not user_text.strip():
                    raise ValueError(f"{path.name}:{line_number}: empty user prompt")
                family_sha = sha256_bytes(normalized_family(user_text).encode("utf-8"))
                prior_split = seen_family.get(family_sha)
                if prior_split is not None:
                    raise ValueError(
                        f"{path.name}:{line_number}: duplicate normalized prompt family "
                        f"across {prior_split} and {output_split}"
                    )
                tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if not isinstance(tokens, list) or not tokens or any(type(x) is not int for x in tokens):
                    raise ValueError(f"{path.name}:{line_number}: invalid tokenizer output")
                if len(tokens) > args.max_prompt_tokens:
                    raise ValueError(f"{path.name}:{line_number}: prompt exceeds locked length limit")
                token_tuple = tuple(tokens)
                if token_tuple in seen_token_ids:
                    raise ValueError(f"{path.name}:{line_number}: duplicate token sequence")
                seen_ids.add(identifier)
                seen_source_ids.add(source_id)
                seen_token_ids.add(token_tuple)
                seen_family[family_sha] = output_split
                counts[output_split] += 1
                rows.append({
                    "id": identifier,
                    "split": output_split,
                    "input_ids": tokens,
                    "source_id": source_id,
                    "prompt_family_sha256": family_sha,
                })
        if counts[output_split] != source_manifest["files"][source_split]["rows"]:
            raise ValueError(f"source row count differs from prior audited manifest: {path.name}")
    if any(count == 0 for count in counts.values()):
        raise ValueError("Every train/val/test split must be nonempty")
    payload = ("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                         for row in rows) + "\n").encode("utf-8")
    metadata = {
        "scope": "p0_locked_user_prompts_only",
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_sha256": source_hashes,
        "tokenizer_path": str(args.tokenizer.resolve()),
        "tokenizer_json_sha256": sha256_file(args.tokenizer / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(args.tokenizer / "tokenizer_config.json"),
        "chat_template_sha256": sha256_bytes(tokenizer.chat_template.encode("utf-8")),
        "enable_thinking": False,
        "add_generation_prompt": True,
        "max_prompt_tokens": args.max_prompt_tokens,
        "counts": counts,
        "prompts_jsonl_sha256": sha256_bytes(payload),
        "assistant_reference_excluded": True,
        "normalized_family_rule": "NFKC casefold, number masking, whitespace collapse",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write(args.output_dir / "prompts.jsonl", payload)
    atomic_write(args.output_dir / "manifest.json",
                 (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"counts": counts, "sha256": metadata["prompts_jsonl_sha256"]}))


if __name__ == "__main__":
    main()
