"""Versioned P0 erratum: reserve formal train/val before any pilot val outcome.

The original prelock ordered formal reservation before pilot selection. The
pilot 1000/200 has already been selected (without looking at quality), so it
is preserved byte-for-byte. This script explicitly records the order
deviation and reserves disjoint 6000/600 formal prompts from the remaining
pinned source pool. It reads only prompt fields of the Spec-Bench source to
exclude overlaps; it never loads reference answers or model outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import v2_build_dataset_manifest as pilot


SCHEMA = "feedback64_v2_formal_reservation_after_pilot_v1"
FORMAL_QUOTAS = {
    "ultrachat": {"formal_confirmation_val": 360, "formal_train": 3600},
    "dolly": {"formal_confirmation_val": 120, "formal_train": 1200},
    "codealpaca": {"formal_confirmation_val": 60, "formal_train": 600},
    "gsm8k": {"formal_confirmation_val": 60, "formal_train": 600},
}
SPLITS = ("formal_confirmation_val", "formal_train")


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row {path}:{index}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row {path}:{index}")
            result.append(row)
    return result


def add_prompt(index: pilot.SimilarityIndex, text: str) -> None:
    index.add(pilot.exact_hash(text), pilot.family_hash(text),
              pilot.char5gram_hashes(text))


def seed_exclusions(index: pilot.SimilarityIndex, *, pilot_train: Path,
                    pilot_val: Path, pilot_manifest: Path,
                    specbench_source: Path, specbench_prelock: Path,
                    prelock: dict) -> dict:
    manifest = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    actual = {"train": pilot.sha256_file(pilot_train),
              "val": pilot.sha256_file(pilot_val)}
    if (manifest.get("schema") != pilot.SCHEMA
            or manifest.get("counts") != {"train": 1000, "val": 200}
            or manifest.get("prompt_jsonl_sha256") != actual):
        raise ValueError("pilot prompt/manifest byte lock differs")
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    for split, path in (("train", pilot_train), ("val", pilot_val)):
        for row in read_jsonl(path):
            row_id, text = row.get("id"), row.get("prompt")
            if (row.get("split") != split or not isinstance(row_id, str)
                    or row_id in ids or not isinstance(text, str) or not text.strip()
                    or pilot.exact_hash(text) != row.get("prompt_sha256")
                    or pilot.family_hash(text) != row.get("prompt_family_sha256")):
                raise ValueError("pilot row changed or has duplicate ID")
            ids.add(row_id)
            counts[split] += 1
            add_prompt(index, text)
    if dict(counts) != {"train": 1000, "val": 200}:
        raise ValueError("pilot split row count changed")

    bench_lock = json.loads(specbench_prelock.read_text(encoding="utf-8"))
    source_sha = pilot.sha256_file(specbench_source)
    if source_sha != bench_lock["official_source"]["file_sha256"]:
        raise ValueError("Spec-Bench prompt source SHA differs from prelock")
    bench_rows = read_jsonl(specbench_source)
    dev_ids = set(bench_lock["split"]["dev"]["question_ids"])
    formal_ids = set(bench_lock["split"]["formal_test"]["question_ids"])
    if (len(bench_rows) != 480 or len(dev_ids) != 96 or len(formal_ids) != 384
            or dev_ids & formal_ids):
        raise ValueError("Spec-Bench prelocked split schema changed")
    bench_ids = set()
    turns = 0
    for row in bench_rows:
        question_id, prompt_turns = row.get("question_id"), row.get("turns")
        if (type(question_id) is not int or question_id in bench_ids
                or question_id not in dev_ids | formal_ids
                or not isinstance(prompt_turns, list) or not prompt_turns):
            raise ValueError("Spec-Bench prompt row differs from prelock")
        bench_ids.add(question_id)
        for text in prompt_turns:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Spec-Bench prompt turn is invalid")
            add_prompt(index, text)
            turns += 1
    if bench_ids != dev_ids | formal_ids:
        raise ValueError("Spec-Bench prelocked IDs changed")
    if prelock["dataset_plan"]["final_test"]["status"] != "SEALED_NOT_ACCESSED":
        raise ValueError("test prelock no longer sealed")
    return {"pilot_jsonl_sha256": actual,
            "pilot_manifest_sha256": pilot.sha256_file(pilot_manifest),
            "specbench_source_sha256": source_sha,
            "specbench_prelock_sha256": pilot.sha256_file(specbench_prelock),
            "specbench_prompt_rows": len(bench_rows),
            "specbench_user_turns_screened": turns,
            "reference_answers_read": False,
            "model_test_outcomes_read": False}


def select_formal(rows: dict[str, list[dict]], index: pilot.SimilarityIndex,
                  blocked_ultrachat_source_ids: set[str], *, seed: int,
                  quotas: dict[str, dict[str, int]] = FORMAL_QUOTAS
                  ) -> tuple[dict[str, list[dict]], dict]:
    selected: dict[str, list[dict]] = {split: [] for split in SPLITS}
    rejections = {split: {source: Counter() for source in quotas}
                  for split in SPLITS}
    for split in SPLITS:
        for source in (plan.key for plan in pilot.SOURCE_PLANS):
            quota = quotas[source][split]
            candidates = sorted(rows[source], key=lambda row: (
                hashlib.sha256(f"{SCHEMA}\0{seed}\0{split}\0{source}\0"
                                f"{row['source_row_id']}".encode()).hexdigest(),
                row["source_row_id"]))
            accepted = 0
            for row in candidates:
                if accepted == quota:
                    break
                if (source == "ultrachat"
                        and row["source_row_id"] in blocked_ultrachat_source_ids):
                    rejections[split][source]["prior_ultrachat_near_duplicate_source_id"] += 1
                    continue
                text = row["prompt"]
                if not isinstance(text, str) or not text.strip():
                    rejections[split][source]["invalid_prompt"] += 1
                    continue
                exact, family, grams = (pilot.exact_hash(text),
                                        pilot.family_hash(text),
                                        pilot.char5gram_hashes(text))
                reason = index.rejection_reason(exact, family, grams)
                if reason is not None:
                    rejections[split][source][reason] += 1
                    continue
                selected[split].append({
                    "id": f"v2-{split}-{source}-{row['source_row_id']}",
                    "split": split, "source": source,
                    "source_row_id": row["source_row_id"],
                    "source_file_sha256": row["source_file_sha256"],
                    "prompt_sha256": exact, "prompt_family_sha256": family,
                    "prompt": text,
                })
                index.add(exact, family, grams)
                accepted += 1
            if accepted != quota:
                raise ValueError(f"{source} {split} quota shortfall: {accepted}/{quota}")
    return selected, {split: {src: dict(c) for src, c in per.items()}
                      for split, per in rejections.items()}


def audit_formal(selected: dict[str, list[dict]], index: pilot.SimilarityIndex) -> dict:
    overlap_count = 0
    ids = set()
    counts = {}
    for split in SPLITS:
        counts[split] = Counter()
        for row in selected[split]:
            text = row["prompt"]
            reason = index.rejection_reason(pilot.exact_hash(text),
                                            pilot.family_hash(text),
                                            pilot.char5gram_hashes(text))
            if reason is not None or row["id"] in ids:
                overlap_count += 1
            ids.add(row["id"])
            counts[split][row["source"]] += 1
            add_prompt(index, text)
    return {"status": "passed" if overlap_count == 0 else "overlap_failure",
            "cross_prior_pilot_specbench_and_internal_overlap_count": overlap_count,
            "counts": {split: dict(counts[split]) for split in SPLITS},
            "threshold": pilot.NEAR_DUPLICATE_THRESHOLD,
            "test_reference_or_model_outcomes_accessed": False}


def build(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError("formal reservation output already exists")
    prelock = json.loads(args.prelock.read_text(encoding="utf-8"))
    if (prelock.get("schema") != "feedback64_v2_dataset_prelock_v1"
            or prelock.get("dataset_plan", {}).get("selection_seed") != pilot.SEED
            or prelock.get("dataset_plan", {}).get("split_order") != [
                "verify source snapshots and licenses",
                "screen old study families and near duplicates",
                "reserve independent final test and formal confirmation val",
                "reserve pilot val", "sample pilot and later formal train"]):
        raise ValueError("dataset prelock differs from original registered order")
    planned = prelock["dataset_plan"]["source_quotas"]
    for row in planned:
        source = next((x.key for x in pilot.SOURCE_PLANS if x.repo == row["source"]), None)
        if (source is None or row["formal_train"] != FORMAL_QUOTAS[source]["formal_train"]
                or row["formal_confirmation_val"]
                != FORMAL_QUOTAS[source]["formal_confirmation_val"]):
            raise ValueError("formal quotas differ from registered prelock")
    if len(planned) != len(FORMAL_QUOTAS):
        raise ValueError("formal source count differs from prelock")

    source_rows, source_provenance = pilot.load_sources(
        args.source_lock, pilot.SOURCE_PLANS)
    exclusions, blocked_ids, old_provenance = pilot.load_balanced3000_exclusions(
        source_rows["ultrachat"], args.old_active,
        args.old_selection_manifest, args.old_holdout_lock)
    seed_evidence = seed_exclusions(
        exclusions, pilot_train=args.pilot_train, pilot_val=args.pilot_val,
        pilot_manifest=args.pilot_manifest,
        specbench_source=args.specbench_source,
        specbench_prelock=args.specbench_prelock, prelock=prelock)
    selected, rejections = select_formal(source_rows, exclusions, blocked_ids,
                                         seed=pilot.SEED)
    # Independently rebuild the exclusion index to audit serialized selection.
    audit_index, _, _ = pilot.load_balanced3000_exclusions(
        source_rows["ultrachat"], args.old_active,
        args.old_selection_manifest, args.old_holdout_lock)
    seed_exclusions(audit_index, pilot_train=args.pilot_train,
                    pilot_val=args.pilot_val,
                    pilot_manifest=args.pilot_manifest,
                    specbench_source=args.specbench_source,
                    specbench_prelock=args.specbench_prelock, prelock=prelock)
    audit = audit_formal(selected, audit_index)
    if audit["status"] != "passed":
        raise RuntimeError("formal reservation independent overlap audit failed")
    expected = {split: sum(FORMAL_QUOTAS[s][split] for s in FORMAL_QUOTAS)
                for split in SPLITS}
    if {split: len(selected[split]) for split in SPLITS} != expected:
        raise AssertionError("formal reservation counts differ from prelock")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".formal-reservation-",
                                     dir=args.output_dir.parent) as temporary:
        temp = Path(temporary)
        files = {split: temp / f"{split}.jsonl" for split in SPLITS}
        for split in SPLITS:
            pilot._write_jsonl(files[split], selected[split])
        report = {
            "schema": SCHEMA,
            "state": "reserved_before_any_new_pilot_val_outcome",
            "original_prelock_order_honored": False,
            "order_deviation": "pilot val/train were selected first; this versioned reservation leaves them unchanged and reserves disjoint formal confirmation val then formal train before any pilot quality result",
            "original_prelock_order": prelock["dataset_plan"]["split_order"],
            "actual_order": ["pilot val", "pilot train", *SPLITS],
            "selection_seed": pilot.SEED,
            "source_lock_sha256": pilot.sha256_file(args.source_lock),
            "prelock_sha256": pilot.sha256_file(args.prelock),
            "source_provenance": source_provenance,
            "old_study_exclusions": old_provenance,
            "seed_exclusions": seed_evidence,
            "formal_quotas": FORMAL_QUOTAS,
            "formal_counts": expected,
            "rejection_counts": rejections,
            "selected": {split: [{key: row[key] for key in (
                "source", "source_row_id", "source_file_sha256",
                "prompt_sha256", "prompt_family_sha256")}
                for row in selected[split]] for split in SPLITS},
            "formal_jsonl_sha256": {split: pilot.sha256_file(files[split])
                                    for split in SPLITS},
            "independent_overlap_audit": audit,
            "pilot_val_outcomes_read": False,
            "specbench_reference_or_model_outcomes_read": False,
            "retired_balanced3000_test_content_read": False,
            "old_balanced3000_test_cross_source_near_duplicate_independence_proved": False,
            "formal_train_or_val_model_run_started": False,
        }
        pilot._write_json(temp / "formal_reservation_manifest.json", report)
        pilot._write_json(temp / "formal_overlap_audit.json", audit)
        temp.rename(args.output_dir)
    return {"state": report["state"], "counts": expected,
            "formal_jsonl_sha256": report["formal_jsonl_sha256"],
            "order_deviation": report["order_deviation"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-lock", "prelock", "pilot-train", "pilot-val",
                 "pilot-manifest", "old-active", "old-selection-manifest",
                 "old-holdout-lock", "specbench-source", "specbench-prelock",
                 "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
