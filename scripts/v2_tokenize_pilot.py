"""Freeze P0 pilot train/val prompt IDs with the pinned Qwen3 chat template.

The script accepts only the already selected v2 raw pilot JSONLs. It never
loads a test split or reference answer. Target and Draft tokenizers must be
byte-identical; thinking mode is explicitly disabled in both splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tokenizer", type=Path, required=True)
    parser.add_argument("--draft-tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-json-sha256", required=True)
    parser.add_argument("--expected-tokenizer-config-sha256", required=True)
    parser.add_argument("--expected-chat-template-sha256", required=True)
    args = parser.parse_args()
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if args.output.exists() or meta_path.exists():
        raise FileExistsError("tokenized output or metadata already exists")
    selected = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    input_hashes = {split: sha256_file(path) for split, path in
                    (("train", args.train), ("val", args.val))}
    if selected.get("prompt_jsonl_sha256") != input_hashes:
        raise ValueError("raw pilot JSONL differs from frozen selection manifest")
    file_hashes = {}
    tokenizers = {}
    for role, path in (("target", args.target_tokenizer),
                       ("draft", args.draft_tokenizer)):
        for name, expected in (("tokenizer.json", args.expected_tokenizer_json_sha256),
                               ("tokenizer_config.json", args.expected_tokenizer_config_sha256)):
            actual = sha256_file(path / name)
            if actual != expected:
                raise ValueError(f"{role} {name} differs from frozen tokenizer hash")
            file_hashes[f"{role}/{name}"] = actual
        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False)
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise ValueError(f"{role} chat template is missing")
        template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if template_sha != args.expected_chat_template_sha256:
            raise ValueError(f"{role} chat template differs from frozen hash")
        file_hashes[f"{role}/chat_template"] = template_sha
        tokenizers[role] = tokenizer
    if tokenizers["target"].eos_token_id != tokenizers["draft"].eos_token_id:
        raise ValueError("Target and Draft EOS token ID differs")

    rows = []
    seen_ids = set()
    seen_tokens = set()
    seen_family_split = {}
    counts = {"train": 0, "val": 0}
    token_lengths = {"train": [], "val": []}
    for split, path in (("train", args.train), ("val", args.val)):
        with path.open("r", encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"blank raw row {split}:{lineno}")
                row = json.loads(line)
                row_id, prompt, family = (row.get("id"), row.get("prompt"),
                                          row.get("prompt_family_sha256"))
                if (row.get("split") != split or not isinstance(row_id, str)
                        or row_id in seen_ids or not isinstance(prompt, str)
                        or not prompt.strip() or not isinstance(family, str)):
                    raise ValueError(f"invalid raw prompt/split/id at {split}:{lineno}")
                if seen_family_split.get(family, split) != split:
                    raise ValueError("one prompt family crosses train and val")
                if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != row.get("prompt_sha256"):
                    raise ValueError(f"raw prompt SHA mismatch at {split}:{lineno}")
                messages = [{"role": "user", "content": prompt}]
                ids = tokenizers["target"].apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False)
                if (not isinstance(ids, list) or not ids
                        or any(type(token) is not int or token < 0 for token in ids)):
                    raise RuntimeError("Qwen3 tokenizer returned invalid token IDs")
                draft_ids = tokenizers["draft"].apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False)
                if ids != draft_ids:
                    raise ValueError("Target and Draft tokenization differs")
                if tuple(ids) in seen_tokens:
                    raise ValueError("duplicate tokenized prompt across pilot")
                seen_ids.add(row_id)
                seen_tokens.add(tuple(ids))
                seen_family_split[family] = split
                counts[split] += 1
                token_lengths[split].append(len(ids))
                rows.append({"id": row_id, "split": split, "family_id": family,
                             "source": row["source"], "input_ids": ids})
    if counts != {"train": 1000, "val": 200}:
        raise ValueError("pilot row counts differ from frozen 1000/200 plan")
    output_bytes = ("\n".join(json.dumps(row, ensure_ascii=False,
                                         separators=(",", ":")) for row in rows) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(output_bytes)
    os.replace(temporary, args.output)
    meta = {"schema": "feedback64_v2_pilot_tokenized_v1",
            "state": "tokenized_train_val_only",
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
            "source_jsonl_sha256": input_hashes,
            "tokenizer_sha256": file_hashes,
            "target_eos_token_id": tokenizers["target"].eos_token_id,
            "enable_thinking": False, "add_generation_prompt": True,
            "counts": counts,
            "token_length_min_max": {key: [min(values), max(values)]
                                     for key, values in token_lengths.items()},
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "sealed_test_loaded": False}
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
    os.replace(meta_tmp, meta_path)
    print(json.dumps({"state": meta["state"], "counts": counts,
                      "output_sha256": meta["output_sha256"]}))


if __name__ == "__main__":
    main()
