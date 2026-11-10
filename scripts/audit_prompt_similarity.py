"""Audit cross-split near duplicates without exposing held-out answers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\b\d+(?:[.,]\d+)*\b", "<number>", text)
    return re.sub(r"\s+", " ", text).strip()


def shingles(text: str) -> set[str]:
    return {text[i:i + 5] for i in range(max(1, len(text) - 4))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    by_split: dict[str, list[tuple[str, set[str]]]] = {}
    source_hashes = {}
    for split in ("train", "validation", "evaluation"):
        path = args.source_dir / f"{split}.jsonl"
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        by_split[split] = [
            (row["id"], shingles(normalized(row["messages"][0]["content"])))
            for row in rows
        ]
    index: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for split, rows in by_split.items():
        for i, (_identifier, grams) in enumerate(rows):
            for gram in grams:
                index[gram].add((split, i))
    near = []
    for split, rows in by_split.items():
        for identifier, grams in rows:
            shared: dict[tuple[str, int], int] = defaultdict(int)
            for gram in grams:
                for candidate in index[gram]:
                    if candidate[0] != split:
                        shared[candidate] += 1
            for (other_split, other_index), overlap in shared.items():
                other_id, other_grams = by_split[other_split][other_index]
                score = overlap / (len(grams) + len(other_grams) - overlap)
                if score >= args.threshold and (split, identifier) < (other_split, other_id):
                    near.append({"left_split": split, "left_id": identifier,
                                 "right_split": other_split, "right_id": other_id,
                                 "char5_jaccard": round(score, 6)})
    report = {
        "scope": "cross_split_prompt_near_duplicate_audit",
        "source_sha256": source_hashes,
        "counts": {split: len(rows) for split, rows in by_split.items()},
        "threshold": args.threshold,
        "near_duplicates": sorted(near, key=lambda item: -item["char5_jaccard"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"near_duplicates": len(near), "threshold": args.threshold}))


if __name__ == "__main__":
    main()
