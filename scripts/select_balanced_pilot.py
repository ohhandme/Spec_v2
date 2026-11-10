"""Pre-register a six-topic, 3,000-prompt UltraChat train_sft pilot.

The official dataset has no labels for these six research categories. This
selector uses a frozen, deliberately conservative rule classifier and marks
multi-hit prompts ambiguous. It reads only train_sft and frozen holdout hashes;
it never opens old or new test prompt content. Assistant replies are ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable

from topic_classifier import TOPICS, classify_topic


DATASET = "HuggingFaceH4/ultrachat_200k"
REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
TRAIN_SFT_ROWS = 207_865
QUOTA = 500
SCHEMA = "ultrachat_six_topic_3000_v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
SPACE = re.compile(r"\s+")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family_hash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = NUMBER.sub("<number>", text)
    text = SPACE.sub(" ", text).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rank_key(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"{SCHEMA}\0{seed}\0{source_id}".encode()).hexdigest()


def validate_lock(lock: dict) -> tuple[set[str], set[str]]:
    if (lock.get("schema") != "ultrachat_six_topic_holdout_lock_v2"
            or lock.get("dataset") != DATASET or lock.get("revision") != REVISION
            or lock.get("frozen_before_selection") is not True):
        raise ValueError("holdout lock schema, provenance or freeze flag differs")
    for name in ("new_validation_manifest_sha256", "new_test_manifest_sha256",
                 "near_duplicate_audit_sha256"):
        if not isinstance(lock.get(name), str) or not HEX64.fullmatch(lock[name]):
            raise ValueError(f"missing valid {name}")
    if lock.get("near_duplicate_threshold") != 0.85:
        raise ValueError("near duplicate threshold must remain 0.85")
    groups = ("old_train_family_sha256_excluded",
              "old_val_family_sha256", "old_test_family_sha256",
              "new_val_family_sha256", "new_test_family_sha256")
    for name in groups:
        hashes = lock.get(name)
        if (not isinstance(hashes, list) or not hashes
                or len(hashes) != len(set(hashes))
                or any(not isinstance(item, str) or not HEX64.fullmatch(item)
                       for item in hashes)):
            raise ValueError(f"missing or invalid {name}")
    near = lock.get("blocked_train_source_ids_near_duplicate")
    reserved = lock.get("reserved_train_source_ids")
    old_train_ids = lock.get("old_train_source_ids")
    if any(not isinstance(items, list) or any(not isinstance(x, str) or not x
                                             for x in items)
           for items in (near, reserved, old_train_ids)):
        raise ValueError("invalid reserved source ID lists")
    if (not old_train_ids or len(old_train_ids) != len(set(old_train_ids))
            or not set(old_train_ids).issubset(reserved)):
        raise ValueError("old train source IDs must be explicitly reserved")
    return set().union(*(lock[name] for name in groups)), set(near + reserved)


def verify_holdout_files(folder: Path, lock: dict) -> None:
    """Hash sealed test bytes without parsing or exposing its prompt contents."""
    expected = {
        "new_validation.jsonl": "new_validation_manifest_sha256",
        "new_test_sealed.jsonl": "new_test_manifest_sha256",
        "near_duplicate_audit.json": "near_duplicate_audit_sha256",
        "selection_audit.json": "selection_audit_sha256",
    }
    for filename, key in expected.items():
        if sha256_file(folder / filename) != lock[key]:
            raise ValueError(f"frozen holdout file SHA256 differs: {filename}")
    if (len(lock["old_train_source_ids"]) != 1024
            or len(lock["old_val_family_sha256"]) != 64
            or len(lock["old_test_family_sha256"]) != 60
            or len(lock["new_val_family_sha256"]) != 120
            or len(lock["new_test_family_sha256"]) != 60):
        raise ValueError("frozen old/new holdout family counts differ")
    audit = json.loads((folder / "near_duplicate_audit.json").read_text(encoding="utf-8"))
    if (audit.get("schema") != "ultrachat_six_topic_near_duplicate_audit_v2"
            or audit.get("threshold") != 0.85
            or audit.get("reference_counts") != {
                "train": 1024, "val": 64, "test": 60,
                "new_val": 120, "new_test": 60,
            }
            or audit.get("train_sft_sha256") != lock.get("train_sft_sha256")
            or audit.get("blocked_train_source_ids")
               != lock["blocked_train_source_ids_near_duplicate"]):
        raise ValueError("near duplicate audit does not match holdout lock")
    if not set(lock["old_train_source_ids"]).issubset(audit["blocked_train_source_ids"]):
        raise ValueError("old train source IDs missing from near duplicate block")
    if lock.get("holdout_source_split") != "train_sft":
        raise ValueError("new holdouts must originate from train_sft")
    if lock.get("classifier_source_sha256") != sha256_file(
        Path(__file__).with_name("topic_classifier.py")
    ):
        raise ValueError("topic classifier changed since holdout freeze")


def load_train_rows(path: Path, expected_rows: int) -> dict[str, str | None]:
    """Read only first user content; reject duplicate official prompt IDs."""
    rows: dict[str, str | None] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank train_sft row {line_number}")
            record = json.loads(line)
            source_id = record.get("prompt_id")
            if not isinstance(source_id, str) or not source_id or source_id in rows:
                raise ValueError(f"invalid/duplicate prompt_id at row {line_number}")
            messages = record.get("messages")
            first = messages[0] if isinstance(messages, list) and messages else None
            text = first.get("content") if isinstance(first, dict) and first.get("role") == "user" else None
            rows[source_id] = text if isinstance(text, str) and text.strip() else None
    if len(rows) != expected_rows:
        raise ValueError(f"train_sft has {len(rows)} rows, expected {expected_rows}")
    return rows


def select_balanced(
    raw: dict[str, str | None], lock: dict, tokenize: Callable[[str], list[int]],
    *, seed: int, quota: int = QUOTA, max_prompt_tokens: int = 256,
) -> tuple[list[dict], dict, list[dict]]:
    if quota < 1 or max_prompt_tokens < 1:
        raise ValueError("quota and max prompt length must be positive")
    reserved_families, reserved_ids = validate_lock(lock)
    ranked = sorted(raw, key=lambda source_id: (rank_key(seed, source_id), source_id))
    pools: dict[str, list[dict]] = {topic: [] for topic in TOPICS}
    counts: Counter[str] = Counter()
    seen_family: set[str] = set()
    seen_tokens: set[tuple[int, ...]] = set()
    ambiguous: list[dict] = []
    unclassified: list[dict] = []
    for source_id in ranked:
        text = raw[source_id]
        if source_id in reserved_ids:
            counts["reserved_source_id"] += 1
            continue
        if text is None:
            counts["invalid_first_user"] += 1
            continue
        family = family_hash(text)
        if family in reserved_families:
            counts["reserved_family"] += 1
            continue
        decision = classify_topic(text)
        if decision.label == "ambiguous":
            counts["ambiguous"] += 1
            ambiguous.append({"source_id": source_id, "text": text, "hits": decision.hits})
            continue
        if decision.label == "unclassified":
            counts["unclassified"] += 1
            unclassified.append({"source_id": source_id, "text": text, "hits": {}})
            continue
        if len(pools[decision.label]) >= quota:
            counts[f"surplus_{decision.label}"] += 1
            continue
        if family in seen_family:
            counts["duplicate_family"] += 1
            continue
        tokens = tokenize(text)
        if (not isinstance(tokens, list) or not tokens
                or any(type(x) is not int or x < 0 for x in tokens)):
            raise ValueError(f"invalid tokenizer output for {source_id}")
        if len(tokens) > max_prompt_tokens:
            counts["prompt_over_length_limit"] += 1
            continue
        token_tuple = tuple(tokens)
        if token_tuple in seen_tokens:
            counts["duplicate_token_sequence"] += 1
            continue
        seen_family.add(family)
        seen_tokens.add(token_tuple)
        pools[decision.label].append({
            "id": f"six-topic-{source_id}", "source_id": source_id,
            "split": "train", "task_type": decision.label,
            "input_ids": tokens, "prompt_family_sha256": family,
            "classifier_hits": decision.hits,
        })
        counts[f"selected_{decision.label}"] += 1
    selected = [row for topic in TOPICS for row in pools[topic]]
    audit_sample = []
    for topic in TOPICS:
        for row in pools[topic][:20]:
            source_id = row["source_id"]
            audit_sample.append({
                "source_id": source_id, "topic": topic, "text": raw[source_id],
                "classifier_hits": row["classifier_hits"],
            })
    for label, group in (("ambiguous", ambiguous), ("unclassified", unclassified)):
        for row in group[:20]:
            audit_sample.append({"source_id": row["source_id"],
                                 "topic": label, "text": row["text"],
                                 "classifier_hits": row["hits"]})
    missing = {topic: quota - len(pools[topic]) for topic in TOPICS if len(pools[topic]) < quota}
    old_train_families = set(lock["old_train_family_sha256_excluded"])
    old_train_overlap = sum(
        row["prompt_family_sha256"] in old_train_families for row in selected
    )
    if old_train_overlap:
        raise AssertionError("old training family entered new training prompts")
    audit = {
        "official_train_sft_rows": len(raw),
        "quota_per_topic": quota,
        "selected_per_topic": {topic: len(pools[topic]) for topic in TOPICS},
        "selected_total": len(selected),
        "excluded_and_classification_counts": dict(sorted(counts.items())),
        "missing_quota": missing,
        "gate_passed": not missing,
        "sampling_caveat": "independent rule-based topic strata; not uniform train_sft sample",
        "sealed_test_tokens_parsed_by_selector": False,
        "overlap_with_old_train_prompts": old_train_overlap,
    }
    return selected, audit, audit_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sft-jsonl", type=Path, required=True)
    parser.add_argument("--train-sft-sha256", required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not HEX64.fullmatch(args.train_sft_sha256):
        raise ValueError("train_sft SHA256 must be lowercase hex")
    source_sha = sha256_file(args.train_sft_jsonl)
    if source_sha != args.train_sft_sha256:
        raise ValueError("train_sft source SHA256 differs")
    lock_path = args.holdout_dir / "holdout_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    validate_lock(lock)
    verify_holdout_files(args.holdout_dir, lock)
    if lock.get("train_sft_sha256") != source_sha:
        raise ValueError("holdout lock and train_sft source differ")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise ValueError("tokenizer has no chat template")

    def tokenize(text: str) -> list[int]:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )

    raw = load_train_rows(args.train_sft_jsonl, TRAIN_SFT_ROWS)
    selected, audit, sample = select_balanced(raw, lock, tokenize, seed=args.seed)
    args.output_dir.mkdir(parents=True)
    sample_path = args.output_dir / "topic_purity_audit_sample.jsonl"
    with sample_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in sample:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    prompts_path = args.output_dir / "train_prompts.jsonl"
    if audit["gate_passed"]:
        with prompts_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        active_path = args.output_dir / "train_val_prompts.jsonl"
        with active_path.open("wb") as output:
            for source_path in (
                prompts_path, args.holdout_dir / "new_validation.jsonl",
            ):
                with source_path.open("rb") as source_stream:
                    shutil.copyfileobj(source_stream, output)
    manifest = {
        "schema": SCHEMA, "dataset": DATASET, "revision": REVISION,
        "official_split": "train_sft", "selection_seed": args.seed,
        "source_jsonl_sha256": source_sha,
        "holdout_lock_sha256": sha256_file(lock_path),
        "tokenizer_json_sha256": sha256_file(args.tokenizer_dir / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(args.tokenizer_dir / "tokenizer_config.json"),
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        "enable_thinking": False,
        "add_generation_prompt": True,
        "max_prompt_tokens": 256, "assistant_reference_excluded": True,
        "classifier_source_sha256": sha256_file(Path(__file__).with_name("topic_classifier.py")),
        "topic_purity_audit_sample_sha256": sha256_file(sample_path),
        "train_prompts_sha256": sha256_file(prompts_path) if audit["gate_passed"] else None,
        "train_val_prompts_sha256": sha256_file(active_path) if audit["gate_passed"] else None,
        "prompts_jsonl_sha256": sha256_file(active_path) if audit["gate_passed"] else None,
        "sealed_test_sha256": lock["new_test_manifest_sha256"],
        "sealed_test_available_to_gpu": False,
        "planned_sealed_test_count": 60,
        "counts": {"train": len(selected), "val": 120, "test": 0}
        if audit["gate_passed"] else None,
        **audit,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), **audit}, ensure_ascii=False))
    if not audit["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
