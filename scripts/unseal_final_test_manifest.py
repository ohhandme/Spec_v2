"""Append a locked, locally sealed test split after the validation gate.

This is the only normal path that reads the new test prompt bytes. It does not
run a model or evaluate anything. The output manifest requires a fresh real P0
because its aggregate SHA differs from the train+val manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path, expected_split: str | None = None) -> list[dict]:
    result = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank row in {path.name}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or (
                not isinstance(row.get("prompt_family_sha256"), str)
                or not HEX64.fullmatch(row["prompt_family_sha256"])
                or not isinstance(row.get("task_type"), str)
                or not row["task_type"]
                or row.get("split") not in ("train", "val", "test")
                or not isinstance(row.get("input_ids"), list)
                or not row["input_ids"]
                or any(type(token) is not int or token < 0 for token in row["input_ids"])
            ):
                raise ValueError(f"invalid prompt row in {path.name}:{number}")
            if expected_split is not None and row.get("split") != expected_split:
                raise ValueError(f"{path.name}:{number} is not {expected_split}")
            if any(key in row for key in ("messages", "assistant", "answer", "reference")):
                raise ValueError("prompt manifest must not contain assistant reference content")
            result.append(row)
    return result


def _check_selection_lock(lock: dict, active_sha: str, sealed_sha: str) -> None:
    """Validate the pre-existing val decision before opening the test file."""
    if lock.get("schema") != "balanced3000_validation_selection_v1" or (
        lock.get("state") != "selected_for_one_time_test"
        or lock.get("active_prompts_sha256") != active_sha
        or lock.get("sealed_test_sha256") != sealed_sha
    ):
        raise ValueError("validation selection lock is absent or mismatched")
    gate = lock.get("gate")
    if not isinstance(gate, dict) or any(gate.get(name) is not True for name in (
        "all_three_seeds_D_greater_C", "D_separates_from_E",
        "full_token_eos_truncation_correctness",
    )):
        raise ValueError("preregistered validation gate did not pass")
    reports = lock.get("validation_report_sha256")
    if not isinstance(reports, dict) or set(reports) != {"42", "43", "44"} or (
        any(not isinstance(value, str) or not HEX64.fullmatch(value)
            for value in reports.values())
    ):
        raise ValueError("validation selection requires three immutable report hashes")
    if not isinstance(lock.get("selected_candidate"), str) or not lock["selected_candidate"]:
        raise ValueError("selected candidate is missing")


def create_final_manifest(
    *, active_path: Path, active_meta_path: Path, sealed_test_path: Path,
    sealed_test_sha: str, selection_lock_path: Path, output_dir: Path,
    expected_train: int = 3000, expected_val: int = 120,
    expected_test: int = 60,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not isinstance(sealed_test_sha, str) or not HEX64.fullmatch(sealed_test_sha):
        raise ValueError("sealed test SHA256 must be lowercase hex")
    active_sha = sha256_file(active_path)
    meta = json.loads(active_meta_path.read_text(encoding="utf-8"))
    if (meta.get("prompts_jsonl_sha256") != active_sha
            or meta.get("output_sha256", active_sha) != active_sha
            or meta.get("counts") != {
                "train": expected_train, "val": expected_val, "test": 0,
            }
            or meta.get("sealed_test_sha256") != sealed_test_sha):
        raise ValueError("active train+val manifest is not the locked input")
    lock = json.loads(selection_lock_path.read_text(encoding="utf-8"))
    _check_selection_lock(lock, active_sha, sealed_test_sha)
    # The test file is not opened before the positive validation gate above.
    if sha256_file(sealed_test_path) != sealed_test_sha:
        raise ValueError("sealed test bytes differ from the frozen SHA256")
    active_rows = _rows(active_path)
    test_rows = _rows(sealed_test_path, "test")
    if not active_path.read_bytes().endswith(b"\n") or (
        not sealed_test_path.read_bytes().endswith(b"\n")
    ):
        raise ValueError("source prompt JSONL must end with a newline")
    split_counts = Counter(row.get("split") for row in active_rows)
    if split_counts != {"train": expected_train, "val": expected_val}:
        raise ValueError("active prompt split counts differ")
    if len(test_rows) != expected_test:
        raise ValueError("sealed test count differs")
    train_topics = Counter(row.get("task_type") for row in active_rows
                           if row.get("split") == "train")
    topics = tuple(sorted(train_topics))
    if len(topics) != 6 or set(train_topics.values()) != {expected_train // 6}:
        raise ValueError("active training topics are not six equal strata")
    expected_per_topic = expected_test // len(topics)
    if expected_per_topic * len(topics) != expected_test or (
        Counter(row.get("task_type") for row in test_rows)
        != {topic: expected_per_topic for topic in topics}
    ):
        raise ValueError("sealed test topic quotas differ")
    ids = [row["id"] for row in active_rows + test_rows]
    families = [row["prompt_family_sha256"] for row in active_rows + test_rows]
    tokens = [tuple(row["input_ids"]) for row in active_rows + test_rows]
    if len(set(ids)) != len(ids) or len(set(families)) != len(families) or (
        len(set(tokens)) != len(tokens)
    ):
        raise ValueError("train/val/test has duplicate ID, family or token sequence")
    tmp_dir = output_dir.with_name(output_dir.name + ".tmp-" + str(os.getpid()))
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        final_path = tmp_dir / "train_val_test_prompts.jsonl"
        with final_path.open("wb") as output:
            for source in (active_path, sealed_test_path):
                with source.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        output.write(block)
        final_sha = sha256_file(final_path)
        final_meta = meta | {
            "prompts_jsonl_sha256": final_sha,
            "output_sha256": final_sha,
            "counts": {"train": expected_train, "val": expected_val,
                       "test": expected_test},
            "active_train_val_prompts_sha256": active_sha,
            "sealed_test_sha256": sealed_test_sha,
            "validation_selection_lock_sha256": sha256_file(selection_lock_path),
            "late_unseal_utc": datetime.now(timezone.utc).isoformat(),
            "test_model_outcomes_opened_by_this_script": False,
        }
        (tmp_dir / "manifest.json").write_text(
            json.dumps(final_meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        tmp_dir.rename(output_dir)
    except BaseException:
        # Keep a failed attempt for inspection; never delete sealed evidence.
        raise
    return {"output": str(output_dir), "active_sha256": active_sha,
            "sealed_test_sha256": sealed_test_sha,
            "final_prompts_sha256": final_sha,
            "manifest_sha256": sha256_file(output_dir / "manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-prompts", type=Path, required=True)
    parser.add_argument("--active-meta", type=Path, required=True)
    parser.add_argument("--sealed-test", type=Path, required=True)
    parser.add_argument("--sealed-test-sha256", required=True)
    parser.add_argument("--validation-selection-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = create_final_manifest(
        active_path=args.active_prompts, active_meta_path=args.active_meta,
        sealed_test_path=args.sealed_test,
        sealed_test_sha=args.sealed_test_sha256,
        selection_lock_path=args.validation_selection_lock,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
