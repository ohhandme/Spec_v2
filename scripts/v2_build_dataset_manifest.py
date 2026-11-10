"""Build the preregistered v2 pilot prompt manifest from pinned *train* files.

This script does not download data, tokenize prompts, inspect model outcomes, or
read the balanced3000 sealed test.  It verifies the old *active train/validation*
manifest and reconstructs their prompt fingerprints from the same pinned
UltraChat train source.  The old sealed test is retired and never used for v2;
only its previously locked IDs/family hashes are inspected.  Cross-source
near-duplicate independence from that retired test remains unproved and is
recorded as a limitation, not silently asserted.

Source-lock format (all paths are relative to the lock file unless absolute)::

  {"schema":"feedback64_v2_source_snapshot_v1","sources":{
    "ultrachat":{"repo":"...","revision":"...","config":null,
      "split":"train_sft","license":"MIT","upstream_data_license":"MIT",
      "format":"parquet","expected_rows":207865,
      "files":[{"path":"train_sft-...parquet","sha256":"..."}, ...]}, ...}}

The old study's selection and holdout locks are checked without opening its
sealed test JSONL.  SHA-256 of each source file and input lock is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SCHEMA = "feedback64_v2_pilot_dataset_manifest_v1"
SOURCE_LOCK_SCHEMA = "feedback64_v2_source_snapshot_v1"
SEED = 20261101
NEAR_DUPLICATE_THRESHOLD = 0.85
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class SourcePlan:
    key: str
    repo: str
    revision: str
    config: str | None
    split: str
    license: str
    upstream_data_license: str
    file_format: str
    filenames: tuple[str, ...]
    expected_rows: int
    train_quota: int
    val_quota: int


SOURCE_PLANS = (
    SourcePlan("ultrachat", "HuggingFaceH4/ultrachat_200k",
               "8049631c405ae6576f93f445c6b8166f76f5505a", None,
               "train_sft", "MIT", "MIT", "parquet", (
                   "train_sft-00000-of-00003-a3ecf92756993583.parquet",
                   "train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
                   "train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
               ), 207_865, 600, 120),
    SourcePlan("dolly", "databricks/databricks-dolly-15k",
               "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", None,
               "train", "CC-BY-SA-3.0", "CC-BY-SA-3.0", "jsonl",
               ("databricks-dolly-15k.jsonl",), 15_011, 200, 40),
    SourcePlan("codealpaca", "sahil2801/CodeAlpaca-20k",
               "152bb5e9a29651266b018106053980070a0521a1", None,
               "train", "CC-BY-4.0", "CC-BY-NC-4.0", "json_array",
               ("code_alpaca_20k.json",), 20_022, 100, 20),
    SourcePlan("gsm8k", "openai/gsm8k",
               "740312add88f781978c0658806c59bc2815b9866", "main",
               "train", "MIT", "MIT", "parquet",
               ("train-00000-of-00001.parquet",), 7_473, 100, 20),
)

# Official SHA-256 values published in the immutable HF file views (and the
# previous UltraChat source lock).  CodeAlpaca's Git-stored JSON has no Xet/LFS
# SHA in its file view; its bytes must still match the independently frozen
# source lock before sampling.
OFFICIAL_FILE_SHA256 = {
    "train_sft-00000-of-00003-a3ecf92756993583.parquet":
        "afa8fa7426081b2a0e732fb50dbb5cd402a28ad5f0dbe66c0d996d63e7220727",
    "train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet":
        "c8230190bc8b29084c974ff32f05589b458fad1908fc4b2fb2b9e1e9f7921f03",
    "train_sft-00002-of-00003-ee46ed25cfae92c6.parquet":
        "6fe7d2a5e95cf660f972ecaf304aa5632d7f53384e4b0e0d0f44d9c96733c03e",
    "databricks-dolly-15k.jsonl":
        "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
    "train-00000-of-00001.parquet":
        "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_hash(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def normalize_family(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return SPACE.sub(" ", NUMBER.sub("<number>", text)).strip()


def family_hash(text: str) -> str:
    return exact_hash(normalize_family(text))


def char5gram_hashes(text: str) -> frozenset[str]:
    normalized = normalize_family(text)
    if not normalized:
        return frozenset()
    return frozenset(exact_hash(normalized[i:i + 5])
                     for i in range(max(1, len(normalized) - 4)))


def _hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _source_prompt(plan: SourcePlan, row: dict) -> tuple[str | None, str | None]:
    """Return stable row ID and user-only prompt; never copy reference answers."""
    if plan.key == "ultrachat":
        source_id, text = row.get("prompt_id"), row.get("prompt")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("UltraChat row lacks prompt_id")
        return source_id, text if isinstance(text, str) and text.strip() else None
    if plan.key in ("dolly", "codealpaca"):
        instruction = row.get("instruction")
        context_key = "context" if plan.key == "dolly" else "input"
        answer_key = "response" if plan.key == "dolly" else "output"
        if context_key not in row or answer_key not in row:
            raise ValueError(f"{plan.key} source fields differ from pinned schema")
        if not isinstance(instruction, str) or not instruction.strip():
            return None, None
        context = row[context_key]
        if not isinstance(context, str):
            raise ValueError(f"{plan.key} {context_key} must be text")
        if context.strip():
            label = "Context" if plan.key == "dolly" else "Input"
            return None, f"{instruction.strip()}\n\n{label}:\n{context.strip()}"
        return None, instruction.strip()
    if plan.key == "gsm8k":
        if "answer" not in row:
            raise ValueError("GSM8K source fields differ from pinned schema")
        text = row.get("question")
        return None, text if isinstance(text, str) and text.strip() else None
    raise ValueError(f"unrecognized source adapter: {plan.key}")


def _file_rows(path: Path, file_format: str, columns: tuple[str, ...]) -> Iterator[dict]:
    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read official Parquet") from exc
        parquet = pq.ParquetFile(path)
        if not set(columns).issubset(parquet.schema_arrow.names):
            raise ValueError(f"missing columns in {path.name}")
        for batch in parquet.iter_batches(batch_size=1024, columns=list(columns)):
            yield from batch.to_pylist()
    elif file_format == "jsonl":
        with path.open("r", encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"blank JSONL row: {path.name}:{lineno}")
                row = json.loads(line)
                if not isinstance(row, dict) or not set(columns).issubset(row):
                    raise ValueError(f"invalid schema: {path.name}:{lineno}")
                yield row
    elif file_format == "json_array":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"JSON array required: {path.name}")
        for lineno, row in enumerate(rows, 1):
            if not isinstance(row, dict) or not set(columns).issubset(row):
                raise ValueError(f"invalid schema: {path.name}:{lineno}")
            yield row
    else:
        raise ValueError(f"unsupported source format: {file_format}")


def _columns(plan: SourcePlan) -> tuple[str, ...]:
    return {
        "ultrachat": ("prompt_id", "prompt"),
        "dolly": ("instruction", "context", "response", "category"),
        "codealpaca": ("instruction", "input", "output"),
        "gsm8k": ("question", "answer"),
    }[plan.key]


def load_sources(source_lock_path: Path, plans: tuple[SourcePlan, ...]
                 ) -> tuple[dict[str, list[dict]], dict]:
    lock = _read_json(source_lock_path)
    if lock.get("schema") != SOURCE_LOCK_SCHEMA:
        raise ValueError("source snapshot lock schema differs")
    source_locks = lock.get("sources")
    if not isinstance(source_locks, dict) or set(source_locks) != {p.key for p in plans}:
        raise ValueError("source snapshot must contain exactly the four pinned sources")
    all_rows: dict[str, list[dict]] = {}
    provenance: dict[str, dict] = {}
    for plan in plans:
        spec = source_locks[plan.key]
        expected = {
            "repo": plan.repo, "revision": plan.revision, "config": plan.config,
            "split": plan.split, "license": plan.license,
            "upstream_data_license": plan.upstream_data_license,
            "format": plan.file_format, "expected_rows": plan.expected_rows,
        }
        if not isinstance(spec, dict) or any(spec.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{plan.key} source identity/license/row lock differs")
        files = spec.get("files")
        if (not isinstance(files, list) or len(files) != len(plan.filenames)
                or [Path(f.get("path", "")).name for f in files if isinstance(f, dict)]
                   != list(plan.filenames)):
            raise ValueError(f"{plan.key} source filenames differ from official train split")
        rows: list[dict] = []
        seen_source_ids: set[str] = set()
        file_provenance = []
        for file in files:
            expected_sha = _hex64(file.get("sha256"), f"{plan.key} file SHA")
            path = Path(file["path"])
            official_sha = OFFICIAL_FILE_SHA256.get(path.name)
            if official_sha is not None and expected_sha != official_sha:
                raise ValueError(f"{plan.key} source lock differs from official file SHA")
            if not path.is_absolute():
                path = source_lock_path.parent / path
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{plan.key} source file missing or symlink: {path}")
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise ValueError(f"{plan.key} source file SHA differs: {path.name}")
            relative_repo_path = (
                f"data/{path.name}" if plan.key == "ultrachat" else
                f"main/{path.name}" if plan.key == "gsm8k" else path.name
            )
            file_provenance.append({
                "name": path.name, "sha256": actual_sha,
                "source_url": (f"https://huggingface.co/datasets/{plan.repo}/resolve/"
                               f"{plan.revision}/{relative_repo_path}"),
            })
            for index, raw in enumerate(_file_rows(path, plan.file_format, _columns(plan))):
                official_id, prompt = _source_prompt(plan, raw)
                row_id = official_id or f"{path.name}:{index}"
                if row_id in seen_source_ids:
                    raise ValueError(f"duplicate source row ID in {plan.key}: {row_id}")
                seen_source_ids.add(row_id)
                rows.append({"source_row_id": row_id, "source_file_sha256": actual_sha,
                             "prompt": prompt})
        if len(rows) != plan.expected_rows:
            raise ValueError(f"{plan.key}: {len(rows)} rows, expected {plan.expected_rows}")
        all_rows[plan.key] = rows
        provenance[plan.key] = {**expected,
                                "dataset_url": f"https://huggingface.co/datasets/{plan.repo}",
                                "files": file_provenance,
                                "observed_rows": len(rows)}
    return all_rows, provenance


class SimilarityIndex:
    def __init__(self) -> None:
        self.exact: set[str] = set()
        self.families: set[str] = set()
        self.shingles: list[frozenset[str]] = []
        self.postings: dict[str, set[int]] = defaultdict(set)

    def add(self, prompt_sha: str, family_sha: str, shingles: frozenset[str]) -> None:
        index = len(self.shingles)
        self.exact.add(prompt_sha)
        self.families.add(family_sha)
        self.shingles.append(shingles)
        for gram in shingles:
            self.postings[gram].add(index)

    def rejection_reason(self, prompt_sha: str, family_sha: str,
                         shingles: frozenset[str]) -> str | None:
        if prompt_sha in self.exact:
            return "exact_prompt"
        if family_sha in self.families:
            return "family"
        shared: Counter[int] = Counter()
        for gram in shingles:
            shared.update(self.postings.get(gram, ()))
        for index, intersection in shared.items():
            other = self.shingles[index]
            if (other and min(len(shingles), len(other)) / max(len(shingles), len(other))
                    >= NEAR_DUPLICATE_THRESHOLD
                    and intersection / (len(shingles) + len(other) - intersection)
                    >= NEAR_DUPLICATE_THRESHOLD):
                return "char5gram_jaccard"
        return None


def load_balanced3000_exclusions(
    ultrachat_rows: list[dict], active_path: Path, selection_manifest_path: Path,
    holdout_lock_path: Path, *, expected_counts: dict[str, int] | None = None,
    expected_retired_test_count: int = 60,
) -> tuple[SimilarityIndex, set[str], dict]:
    """Fingerprint only old train/val; use previously frozen test metadata.

    The old test JSONL is intentionally neither accepted as an argument nor
    opened.  Its pre-existing family hashes exclude obvious cross-source
    relatives, while prior blocked train_sft source IDs cover near duplicates
    within UltraChat.  No cross-source near-duplicate claim is made for it.
    """
    selection = _read_json(selection_manifest_path)
    holdout = _read_json(holdout_lock_path)
    counts = expected_counts or {"train": 3000, "val": 120}
    active_sha = sha256_file(active_path)
    holdout_sha = sha256_file(holdout_lock_path)
    if (selection.get("schema") != "ultrachat_six_topic_3000_v1"
            or selection.get("train_val_prompts_sha256") != active_sha
            or selection.get("holdout_lock_sha256") != holdout_sha
            or selection.get("counts") != {**counts, "test": 0}
            or holdout.get("schema") != "ultrachat_six_topic_holdout_lock_v2"
            or holdout.get("near_duplicate_threshold") != NEAR_DUPLICATE_THRESHOLD):
        raise ValueError("balanced3000 active prompts / frozen locks differ")
    old_test_file_sha = _hex64(holdout.get("new_test_manifest_sha256"),
                               "retired sealed test file SHA")
    old_val_ids = holdout.get("new_val_source_ids")
    old_val_families = holdout.get("new_val_family_sha256")
    old_test_ids = holdout.get("new_test_source_ids")
    old_test_families = holdout.get("new_test_family_sha256")
    blocked_ids = holdout.get("blocked_train_source_ids_near_duplicate")
    if (any(not isinstance(value, list) for value in (
            old_val_ids, old_val_families, old_test_ids, old_test_families, blocked_ids))
            or len(old_val_ids) != counts["val"]
            or len(old_val_families) != counts["val"]
            or len(old_test_ids) != expected_retired_test_count
            or len(old_test_families) != expected_retired_test_count
            or any(not isinstance(value, str) or not value for value in
                   old_val_ids + old_test_ids + blocked_ids)
            or any(not HEX64.fullmatch(value) for value in
                   old_val_families + old_test_families)):
        raise ValueError("balanced3000 frozen holdout metadata incomplete")
    for family in old_test_families:
        _hex64(family, "retired test family")
    by_id = {row["source_row_id"]: row["prompt"] for row in ultrachat_rows}
    similarity = SimilarityIndex()
    seen_ids: set[str] = set()
    seen_counts: Counter[str] = Counter()
    val_ids: set[str] = set()
    val_families: set[str] = set()
    with active_path.open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank balanced3000 active row {lineno}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"invalid balanced3000 active row {lineno}")
            split, source_id, family = (row.get("split"), row.get("source_id"),
                                        row.get("prompt_family_sha256"))
            if (split not in counts or not isinstance(source_id, str)
                    or not source_id or source_id in seen_ids):
                raise ValueError(f"invalid/duplicate balanced3000 source at {lineno}")
            text = by_id.get(source_id)
            if (not isinstance(text, str) or not text.strip()
                    or family_hash(text) != family):
                raise ValueError(f"balanced3000 active family/source mismatch at {lineno}")
            seen_ids.add(source_id)
            seen_counts[split] += 1
            similarity.add(exact_hash(text), family, char5gram_hashes(text))
            if split == "val":
                val_ids.add(source_id)
                val_families.add(family)
    if dict(seen_counts) != counts:
        raise ValueError("balanced3000 active train/val counts differ")
    if val_ids != set(old_val_ids) or val_families != set(old_val_families):
        raise ValueError("balanced3000 active validation differs from frozen holdout")
    if not val_ids.issubset(blocked_ids):
        raise ValueError("balanced3000 validation IDs absent from prior near-dup audit")
    # These are opaque pre-existing hashes only; no retired test text is read.
    similarity.families.update(old_test_families)
    for name in ("old_train_family_sha256_excluded", "old_val_family_sha256",
                 "old_test_family_sha256"):
        values = holdout.get(name)
        if not isinstance(values, list) or any(not isinstance(v, str) or
                                               not HEX64.fullmatch(v) for v in values):
            raise ValueError(f"missing old-study frozen family hashes: {name}")
        similarity.families.update(values)
    provenance = {
        "study": "balanced3000", "active_prompt_sha256": active_sha,
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "holdout_lock_sha256": holdout_sha,
        "active_counts": counts,
        "retired_test_count": expected_retired_test_count,
        "retired_test_file_sha256": old_test_file_sha,
        "retired_test_policy": "retired_never_used_for_v2",
        "retired_test_content_read": False,
        "retired_test_cross_source_near_duplicate_independence_proved": False,
        "retired_test_exact_family_hashes_excluded": True,
        "prior_ultrachat_near_duplicate_blocked_source_ids": len(set(blocked_ids)),
    }
    return similarity, set(blocked_ids), provenance


def rank_key(seed: int, split: str, source: str, row_id: str) -> str:
    return exact_hash(f"{SCHEMA}\0{seed}\0{split}\0{source}\0{row_id}")


def select_pilot(rows: dict[str, list[dict]], plans: tuple[SourcePlan, ...],
                 exclusions: SimilarityIndex, *,
                 blocked_ultrachat_source_ids: set[str] | None = None,
                 seed: int = SEED
                 ) -> tuple[list[dict], list[dict], dict]:
    blocked_ultrachat_source_ids = blocked_ultrachat_source_ids or set()
    selected: dict[str, list[dict]] = {"val": [], "train": []}
    rejection_counts: dict[str, dict[str, Counter[str]]] = {
        split: {p.key: Counter() for p in plans} for split in ("val", "train")
    }
    # Reserve validation before training; both are ranked without reading quality.
    for split in ("val", "train"):
        for plan in plans:
            quota = plan.val_quota if split == "val" else plan.train_quota
            candidates = sorted(rows[plan.key], key=lambda row: (
                rank_key(seed, split, plan.key, row["source_row_id"]),
                row["source_row_id"]))
            accepted = 0
            for row in candidates:
                if accepted == quota:
                    break
                if (plan.key == "ultrachat"
                        and row["source_row_id"] in blocked_ultrachat_source_ids):
                    rejection_counts[split][plan.key]["prior_ultrachat_near_duplicate_source_id"] += 1
                    continue
                text = row["prompt"]
                if not isinstance(text, str) or not text.strip():
                    rejection_counts[split][plan.key]["invalid_prompt"] += 1
                    continue
                exact, family, grams = exact_hash(text), family_hash(text), char5gram_hashes(text)
                reason = exclusions.rejection_reason(exact, family, grams)
                if reason is not None:
                    rejection_counts[split][plan.key][reason] += 1
                    continue
                selected[split].append({
                    "id": f"v2-{split}-{plan.key}-{row['source_row_id']}",
                    "split": split, "source": plan.key,
                    "source_row_id": row["source_row_id"],
                    "source_file_sha256": row["source_file_sha256"],
                    "prompt_sha256": exact, "prompt_family_sha256": family,
                    "prompt": text,
                })
                exclusions.add(exact, family, grams)
                accepted += 1
            if accepted != quota:
                raise ValueError(f"{plan.key} {split} quota shortfall: {accepted}/{quota}")
    audits = {split: {key: dict(counts) for key, counts in per_source.items()}
              for split, per_source in rejection_counts.items()}
    return selected["train"], selected["val"], audits


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")


def build(source_lock_path: Path, old_active_path: Path,
          old_selection_manifest_path: Path, holdout_lock_path: Path,
          output_dir: Path,
          *, plans: tuple[SourcePlan, ...] = SOURCE_PLANS, seed: int = SEED,
          expected_old_active_counts: dict[str, int] | None = None,
          expected_retired_test_count: int = 60) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_rows, provenance = load_sources(source_lock_path, plans)
    exclusions, blocked_ids, old = load_balanced3000_exclusions(
        source_rows["ultrachat"], old_active_path, old_selection_manifest_path,
        holdout_lock_path, expected_counts=expected_old_active_counts,
        expected_retired_test_count=expected_retired_test_count)
    train, val, rejection_counts = select_pilot(
        source_rows, plans, exclusions,
        blocked_ultrachat_source_ids=blocked_ids, seed=seed)
    if len(train) != sum(p.train_quota for p in plans) or len(val) != sum(p.val_quota for p in plans):
        raise AssertionError("pilot split counts differ from preregistered quotas")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v2-dataset-", dir=output_dir.parent) as tmp:
        temporary = Path(tmp)
        train_path, val_path = temporary / "pilot_train.jsonl", temporary / "pilot_val.jsonl"
        _write_jsonl(train_path, train)
        _write_jsonl(val_path, val)
        manifest = {
            "schema": SCHEMA, "selection_seed": seed,
            "selection_order": ["pilot_val", "pilot_train"],
            "near_duplicate_rule": "NFKC/casefold/number/space normalized char5 Jaccard >= 0.85",
            "source_lock_sha256": sha256_file(source_lock_path),
            "sources": provenance, "legacy_exclusions": old,
            "quotas": {p.key: {"train": p.train_quota, "val": p.val_quota} for p in plans},
            "counts": {"train": len(train), "val": len(val)},
            "rejection_counts": rejection_counts,
            "selected": {
                split: [{key: row[key] for key in (
                    "source", "source_row_id", "source_file_sha256",
                    "prompt_sha256", "prompt_family_sha256")}
                    for row in group]
                for split, group in (("train", train), ("val", val))
            },
            "prompt_jsonl_sha256": {"train": sha256_file(train_path),
                                    "val": sha256_file(val_path)},
            "assistant_and_reference_answers_excluded": True,
            "source_licenses_mixed": True,
            "codealpaca_upstream_noncommercial_license_recorded": True,
            "new_validation_quality_unseen_before_selection": True,
            "retired_legacy_test_content_not_read": True,
            "retired_legacy_test_cross_source_near_duplicate_independence_proved": False,
        }
        _write_json(temporary / "datasets_manifest.json", manifest)
        temporary.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--balanced3000-active-prompts", type=Path, required=True)
    parser.add_argument("--balanced3000-selection-manifest", type=Path, required=True)
    parser.add_argument("--balanced3000-holdout-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source_lock, args.balanced3000_active_prompts,
                     args.balanced3000_selection_manifest,
                     args.balanced3000_holdout_lock, args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
