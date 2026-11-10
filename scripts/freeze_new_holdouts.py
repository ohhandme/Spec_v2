"""Freeze independent UltraChat train_sft validation/test before train sampling.

This script selects 20 validation and 10 sealed-test prompts per topic from
official train_sft. It only performs dataset hygiene, classification and
tokenization; it never loads models or evaluates model outputs. Old test first
user text is reconstructed only from previously locked prompt token IDs for
split dedup, checked against its frozen family hash, and never written out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from select_balanced_pilot import (
    DATASET, REVISION, TOPICS, classify_topic, family_hash,
    load_train_rows, sha256_file,
)


OLD_COUNTS = {"train": 1024, "val": 64, "test": 60}
HOLDOUT_SCHEMA = "ultrachat_six_topic_holdout_lock_v2"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
SPACE = re.compile(r"\s+")


def normalized(text: str) -> str:
    return SPACE.sub(" ", NUMBER.sub("<number>",
        unicodedata.normalize("NFKC", text).casefold())).strip()


def shingles(text: str) -> frozenset[str]:
    text = normalized(text)
    return frozenset(text[i:i + 5] for i in range(max(1, len(text) - 4)))


def near_duplicate(a: frozenset[str], b: frozenset[str], threshold: float = 0.85) -> bool:
    if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < threshold:
        return False
    overlap = len(a & b)
    return overlap / (len(a) + len(b) - overlap) >= threshold


def old_split_metadata(path: Path, train_raw: dict[str, str | None],
                       decode_test_user: Callable[[list[int]], str],
                       expected_counts: dict[str, int] = OLD_COUNTS
                       ) -> dict[str, list[tuple[str, str, str]]]:
    """Read old IDs/family hashes; verify against pinned source first turns."""
    old: dict[str, list[tuple[str, str, str]]] = {key: [] for key in OLD_COUNTS}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            split, source_id, family = (row.get("split"), row.get("source_id"),
                                        row.get("prompt_family_sha256"))
            if split not in old or not isinstance(source_id, str) or not isinstance(family, str):
                raise ValueError(f"invalid old prompt provenance at line {line_number}")
            if split == "test":
                ids = row.get("input_ids")
                if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
                    raise ValueError(f"old test prompt tokens invalid at line {line_number}")
                text = decode_test_user(ids)
            else:
                text = train_raw.get(source_id)
            if not isinstance(text, str) or family_hash(text) != family:
                raise ValueError(f"old prompt family/source mismatch at line {line_number}")
            old[split].append((source_id, family, text))
    counts = {name: len(rows) for name, rows in old.items()}
    if counts != expected_counts:
        raise ValueError(f"old split counts {counts} != {expected_counts}")
    return old


def decode_locked_qwen_user(ids: list[int], tokenizer: object) -> str:
    """Recover only the old first-user prompt, then verify its frozen hash."""
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    start = "<|im_start|>user\n"
    end = "<|im_end|>"
    if not decoded.startswith(start) or end not in decoded[len(start):]:
        raise ValueError("old sealed prompt has unexpected Qwen chat template")
    text = decoded[len(start):].split(end, 1)[0]
    if not text.strip():
        raise ValueError("old sealed first-user prompt is empty")
    return text


def holdout_rank(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"{HOLDOUT_SCHEMA}\0{seed}\0{source_id}".encode()).hexdigest()


def select_holdouts(
    candidate_raw: dict[str, str | None],
    old: dict[str, list[tuple[str, str, str]]],
    tokenize: Callable[[str], list[int]],
    *, seed: int, val_per_topic: int = 20, test_per_topic: int = 10,
    max_prompt_tokens: int = 256,
    classify: Callable[[str], object] = classify_topic,
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    if min(val_per_topic, test_per_topic, max_prompt_tokens) < 1:
        raise ValueError("holdout quotas and length must be positive")
    old_ids = {source_id for rows in old.values() for source_id, _, _ in rows}
    old_families = {family for rows in old.values() for _, family, _ in rows}
    heldout_grams = [shingles(text) for rows in old.values() for _, _, text in rows]
    val = {topic: [] for topic in TOPICS}
    sealed_test = {topic: [] for topic in TOPICS}
    seen_families = set(old_families)
    seen_tokens: set[tuple[int, ...]] = {
        tuple(tokenize(text)) for rows in old.values() for _, _, text in rows
    }
    counts: Counter[str] = Counter()
    val_sample: list[dict] = []
    ranked = sorted(candidate_raw, key=lambda sid: (holdout_rank(seed, sid), sid))
    for source_id in ranked:
        if all(len(val[t]) == val_per_topic and len(sealed_test[t]) == test_per_topic
               for t in TOPICS):
            break
        text = candidate_raw[source_id]
        if source_id in old_ids or text is None:
            counts["old_or_invalid_source"] += 1
            continue
        family = family_hash(text)
        if family in seen_families:
            counts["duplicate_family"] += 1
            continue
        decision = classify(text)
        if decision.label not in TOPICS:
            counts[decision.label] += 1
            continue
        topic = decision.label
        if len(val[topic]) >= val_per_topic and len(sealed_test[topic]) >= test_per_topic:
            counts[f"surplus_{topic}"] += 1
            continue
        grams = shingles(text)
        if any(near_duplicate(grams, old_grams) for old_grams in heldout_grams):
            counts["near_duplicate_holdout"] += 1
            continue
        tokens = tokenize(text)
        if (not isinstance(tokens, list) or not tokens
                or any(type(x) is not int or x < 0 for x in tokens)):
            raise ValueError(f"invalid tokenizer output: {source_id}")
        if len(tokens) > max_prompt_tokens:
            counts["prompt_over_length_limit"] += 1
            continue
        if tuple(tokens) in seen_tokens:
            counts["duplicate_token_sequence"] += 1
            continue
        split = "val" if len(val[topic]) < val_per_topic else "test"
        row = {
            "id": f"six-topic-{split}-{source_id}", "source_id": source_id,
            "split": split, "task_type": topic, "input_ids": tokens,
            "prompt_family_sha256": family, "classifier_hits": decision.hits,
        }
        (val if split == "val" else sealed_test)[topic].append(row)
        heldout_grams.append(grams)
        seen_families.add(family)
        seen_tokens.add(tuple(tokens))
        counts[f"selected_{split}_{topic}"] += 1
        if split == "val" and len(val_sample) < 120:
            val_sample.append({"source_id": source_id, "topic": topic,
                               "text": text, "classifier_hits": decision.hits})
    val_rows = [row for topic in TOPICS for row in val[topic]]
    test_rows = [row for topic in TOPICS for row in sealed_test[topic]]
    missing = {
        f"{split}_{topic}": target - len(group[topic])
        for split, group, target in (("val", val, val_per_topic),
                                     ("test", sealed_test, test_per_topic))
        for topic in TOPICS if len(group[topic]) < target
    }
    audit = {
        "official_train_sft_rows": len(candidate_raw),
        "raw_source_rows_examined": sum(counts.values()),
        "val_per_topic": val_per_topic, "test_per_topic": test_per_topic,
        "val_counts": {topic: len(val[topic]) for topic in TOPICS},
        "test_counts": {topic: len(sealed_test[topic]) for topic in TOPICS},
        "counts": dict(sorted(counts.items())), "missing_quota": missing,
        "gate_passed": not missing,
        "test_contents_used_for_model_selection": False,
    }
    return val_rows, test_rows, audit, val_sample


def blocked_near_duplicate_train(
    train_raw: dict[str, str | None],
    old: dict[str, list[tuple[str, str, str]]],
    val_rows: list[dict], test_rows: list[dict],
    *, threshold: float = 0.85,
) -> list[str]:
    """Dataset-only 5-gram screen; report IDs, never heldout text."""
    # The old training set is also excluded under the frozen new protocol.
    # Screening its prompt text here catches different IDs and family hashes
    # whose first-user requests are nevertheless near duplicates.
    heldout_text = [text for split in ("train", "val", "test")
                    for _, _, text in old[split]]
    for row in val_rows + test_rows:
        text = train_raw[row["source_id"]]
        if text is None:
            raise ValueError("selected heldout source text absent")
        heldout_text.append(text)
    signatures = [shingles(text) for text in heldout_text]
    index: dict[str, set[int]] = defaultdict(set)
    for index_number, grams in enumerate(signatures):
        for gram in grams:
            index[gram].add(index_number)
    blocked = []
    for source_id, text in train_raw.items():
        if text is None:
            continue
        grams = shingles(text)
        shared: Counter[int] = Counter()
        for gram in grams:
            shared.update(index.get(gram, ()))
        for index_number, overlap in shared.items():
            other = signatures[index_number]
            if overlap / (len(grams) + len(other) - overlap) >= threshold:
                blocked.append(source_id)
                break
    return sorted(blocked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sft-jsonl", type=Path, required=True)
    parser.add_argument("--train-sft-sha256", required=True)
    parser.add_argument("--old-prompts-jsonl", type=Path, required=True)
    parser.add_argument("--old-prompts-sha256", required=True)
    parser.add_argument("--source-conversion-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path, expected in ((args.train_sft_jsonl, args.train_sft_sha256),
                           (args.old_prompts_jsonl, args.old_prompts_sha256)):
        if not HEX64.fullmatch(expected) or sha256_file(path) != expected:
            raise ValueError(f"input SHA256 differs: {path}")
    source_manifest = json.loads(args.source_conversion_manifest.read_text(encoding="utf-8"))
    if (source_manifest.get("schema") != "ultrachat_first_user_jsonl_v1"
            or source_manifest.get("dataset") != DATASET
            or source_manifest.get("revision") != REVISION
            or source_manifest.get("train_sft_jsonl_sha256") != args.train_sft_sha256
            or len(source_manifest.get("parquet_sha256", {})) != 4):
        raise ValueError("official source conversion manifest differs")
    train_raw = load_train_rows(args.train_sft_jsonl, 207_865)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise ValueError("tokenizer has no chat template")

    old = old_split_metadata(
        args.old_prompts_jsonl, train_raw,
        lambda ids: decode_locked_qwen_user(ids, tokenizer),
    )

    def tokenize(text: str) -> list[int]:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )

    val, sealed_test, audit, val_sample = select_holdouts(
        train_raw, old, tokenize, seed=args.seed
    )
    args.output_dir.mkdir(parents=True)
    selection_audit_path = args.output_dir / "selection_audit.json"
    selection_audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not audit["gate_passed"]:
        raise SystemExit(2)
    blocked = blocked_near_duplicate_train(train_raw, old, val, sealed_test)
    near_audit = {
        "schema": "ultrachat_six_topic_near_duplicate_audit_v2",
        "threshold": 0.85, "n_gram": 5,
        "reference_counts": {split: len(old[split]) for split in ("train", "val", "test")}
            | {"new_val": len(val), "new_test": len(sealed_test)},
        "train_sft_sha256": args.train_sft_sha256,
        "blocked_train_source_ids": blocked,
        "blocked_count": len(blocked),
    }
    near_path = args.output_dir / "near_duplicate_audit.json"
    near_path.write_text(json.dumps(near_audit, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    for name, rows in (("new_validation.jsonl", val), ("new_test_sealed.jsonl", sealed_test),
                       ("validation_purity_sample.jsonl", val_sample)):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    lock = {
        "schema": HOLDOUT_SCHEMA, "dataset": DATASET, "revision": REVISION,
        "frozen_before_selection": True,
        "selection_seed": args.seed,
        "train_sft_sha256": args.train_sft_sha256,
        "holdout_source_split": "train_sft",
        "source_conversion_manifest_sha256": sha256_file(args.source_conversion_manifest),
        "official_parquet_sha256": source_manifest["parquet_sha256"],
        "old_prompts_sha256": args.old_prompts_sha256,
        "selection_audit_sha256": sha256_file(selection_audit_path),
        "new_validation_manifest_sha256": sha256_file(args.output_dir / "new_validation.jsonl"),
        "new_test_manifest_sha256": sha256_file(args.output_dir / "new_test_sealed.jsonl"),
        "near_duplicate_audit_sha256": sha256_file(near_path),
        "near_duplicate_threshold": 0.85,
        "old_train_source_ids": [source_id for source_id, _, _ in old["train"]],
        "old_train_family_sha256_excluded": sorted({family for _, family, _ in old["train"]}),
        "old_val_family_sha256": [family for _, family, _ in old["val"]],
        "old_test_family_sha256": [family for _, family, _ in old["test"]],
        "new_val_family_sha256": [row["prompt_family_sha256"] for row in val],
        "new_test_family_sha256": [row["prompt_family_sha256"] for row in sealed_test],
        "new_val_source_ids": [row["source_id"] for row in val],
        "new_test_source_ids": [row["source_id"] for row in sealed_test],
        "blocked_train_source_ids_near_duplicate": blocked,
        "reserved_train_source_ids": sorted(
            {source_id for split in ("train", "val", "test")
             for source_id, _, _ in old[split]}
            | {row["source_id"] for row in val + sealed_test}
        ),
        "classifier_source_sha256": sha256_file(Path(__file__).with_name("topic_classifier.py")),
    }
    (args.output_dir / "holdout_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "val": len(val),
                      "sealed_test": len(sealed_test), "near_duplicate_blocked": len(blocked)}))


if __name__ == "__main__":
    main()
