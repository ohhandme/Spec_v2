"""Audit a frozen pilot split against a prelocked Spec-Bench source.

This is a CPU-only data gate. It reads prompt text solely to compute overlap;
the output contains IDs and hashes, never prompt or reference content. It does
not tokenize, run a model, choose checkpoints, or inspect evaluation outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


THRESHOLD = 0.85
NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
SPACE = re.compile(r"\s+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return SPACE.sub(" ", NUMBER.sub("<number>",
                     unicodedata.normalize("NFKC", text).casefold())).strip()


def fingerprints(text: str) -> tuple[str, str, frozenset[str]]:
    norm = normalize(text)
    exact = hashlib.sha256(text.encode("utf-8")).hexdigest()
    family = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    grams = frozenset(norm[i:i + 5] for i in range(max(1, len(norm) - 4)))
    return exact, family, grams


class Index:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, frozenset[str]]] = []
        self.exact: dict[str, set[int]] = defaultdict(set)
        self.family: dict[str, set[int]] = defaultdict(set)
        self.postings: dict[str, set[int]] = defaultdict(set)

    def add(self, row_id: str, split: str, text: str) -> None:
        exact, family, grams = fingerprints(text)
        at = len(self.rows)
        self.rows.append((row_id, split, family, grams))
        self.exact[exact].add(at)
        self.family[family].add(at)
        for gram in grams:
            self.postings[gram].add(at)

    def overlaps(self, text: str, *, other_split: str | None = None) -> list[dict]:
        exact, family, grams = fingerprints(text)
        candidates = set(self.exact.get(exact, ())) | set(self.family.get(family, ()))
        shared: Counter[int] = Counter()
        for gram in grams:
            shared.update(self.postings.get(gram, ()))
        for at, intersection in shared.items():
            other = self.rows[at][3]
            if (other and min(len(grams), len(other)) / max(len(grams), len(other))
                    >= THRESHOLD and intersection / (len(grams) + len(other) - intersection)
                    >= THRESHOLD):
                candidates.add(at)
        found = []
        for at in sorted(candidates):
            row_id, split, other_family, other = self.rows[at]
            if other_split is not None and split == other_split:
                continue
            if at in self.exact.get(exact, ()):
                reason, similarity = "exact", 1.0
            elif at in self.family.get(family, ()):
                reason, similarity = "normalized_family", 1.0
            else:
                intersection = len(grams & other)
                similarity = intersection / (len(grams) + len(other) - intersection)
                if similarity < THRESHOLD:
                    continue
                reason = "char5_jaccard"
            found.append({"other_id": row_id, "other_split": split,
                          "reason": reason, "similarity": round(similarity, 6),
                          "other_family_sha256": other_family})
        return found


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank row at {path.name}:{lineno}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path.name}:{lineno}")
            rows.append(row)
    return rows


def audit(train: Path, val: Path, manifest: Path, source: Path,
          prelock: Path) -> dict:
    manifest_obj = json.loads(manifest.read_text(encoding="utf-8"))
    lock = json.loads(prelock.read_text(encoding="utf-8"))
    actual = {"train": sha256_file(train), "val": sha256_file(val)}
    if manifest_obj.get("prompt_jsonl_sha256") != actual:
        raise ValueError("pilot prompt JSONL differs from the frozen manifest")
    source_sha = sha256_file(source)
    if source_sha != lock["official_source"]["file_sha256"]:
        raise ValueError("Spec-Bench source SHA differs from the frozen prelock")
    train_rows, val_rows, test_rows = read_jsonl(train), read_jsonl(val), read_jsonl(source)
    if len(train_rows) != 1000 or len(val_rows) != 200 or len(test_rows) != 480:
        raise ValueError("pilot or Spec-Bench row count differs from frozen plan")
    pilot = Index()
    seen_ids: set[str] = set()
    within_pilot = []
    for split, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            row_id, text = row.get("id"), row.get("prompt")
            if (row.get("split") != split or not isinstance(row_id, str)
                    or not isinstance(text, str) or not text.strip()
                    or row_id in seen_ids):
                raise ValueError("pilot split, ID, or prompt contract failed")
            seen_ids.add(row_id)
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != row.get("prompt_sha256"):
                raise ValueError("pilot prompt SHA differs from row metadata")
            if fingerprints(text)[1] != row.get("prompt_family_sha256"):
                raise ValueError("pilot family SHA differs from row metadata")
            matches = pilot.overlaps(text)
            if matches:
                within_pilot.append({"id": row_id, "split": split, "matches": matches})
            pilot.add(row_id, split, text)
    if within_pilot:
        raise ValueError("frozen pilot contains internal exact/near duplicates")
    test = Index()
    cross_pilot = []
    cross_test_split = []
    category_count: Counter[str] = Counter()
    test_ids = set()
    for row in test_rows:
        question_id, category, turns = (row.get("question_id"), row.get("category"),
                                        row.get("turns"))
        if (type(question_id) is not int or question_id in test_ids
                or not isinstance(category, str) or not isinstance(turns, list)
                or not turns or any(not isinstance(x, str) or not x.strip() for x in turns)):
            raise ValueError("Spec-Bench ID/category/turn schema differs from prelock")
        test_ids.add(question_id)
        category_count[category] += 1
        split = "dev" if question_id % 5 == 1 else "formal_test"
        for turn_index, text in enumerate(turns):
            row_id = f"{question_id}:turn{turn_index + 1}"
            matches = pilot.overlaps(text)
            if matches:
                cross_pilot.append({"question_id": question_id, "turn": turn_index + 1,
                                    "test_split": split, "category": category,
                                    "matches": matches})
            prior = test.overlaps(text, other_split=split)
            if prior:
                cross_test_split.append({"question_id": question_id,
                                         "turn": turn_index + 1,
                                         "test_split": split, "category": category,
                                         "matches": prior})
            test.add(row_id, split, text)
    expected_split = lock["split"]
    expected_dev = set(expected_split["dev"]["question_ids"])
    expected_formal = set(expected_split["formal_test"]["question_ids"])
    if (test_ids != expected_dev | expected_formal or expected_dev & expected_formal
            or any(q % 5 != 1 for q in expected_dev)
            or any(q % 5 == 1 for q in expected_formal)):
        raise ValueError("Spec-Bench ID split differs from frozen prelock")
    if dict(sorted(category_count.items())) != lock["schema"]["category_counts"]:
        raise ValueError("Spec-Bench category counts differ from frozen prelock")
    return {"schema": "feedback64_v2_specbench_overlap_audit_v1",
            "status": "passed" if not cross_pilot and not cross_test_split else "overlap_requires_erratum",
            "rule": "NFKC casefold numbers-to-<number> whitespace normalized, char5 Jaccard >= 0.85",
            "input_sha256": {"pilot_train": actual["train"], "pilot_val": actual["val"],
                             "pilot_manifest": sha256_file(manifest),
                             "specbench_source": source_sha,
                             "specbench_prelock": sha256_file(prelock)},
            "pilot_counts": {"train": len(train_rows), "val": len(val_rows)},
            "specbench_counts": {"dev": len(expected_dev),
                                 "formal_test": len(expected_formal)},
            "pilot_internal_overlap_count": 0,
            "test_pilot_overlap_count": len(cross_pilot),
            "test_dev_formal_overlap_count": len(cross_test_split),
            "test_pilot_overlaps": cross_pilot,
            "test_dev_formal_overlaps": cross_test_split,
            "no_model_evaluation": True,
            "sealed_test_model_outcomes_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-train", type=Path, required=True)
    parser.add_argument("--pilot-val", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--specbench-source", type=Path, required=True)
    parser.add_argument("--specbench-prelock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("data audit output must be a new path")
    result = audit(args.pilot_train, args.pilot_val, args.pilot_manifest,
                   args.specbench_source, args.specbench_prelock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({key: result[key] for key in (
        "status", "pilot_counts", "specbench_counts",
        "test_pilot_overlap_count", "test_dev_formal_overlap_count")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
