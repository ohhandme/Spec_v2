"""Fail-closed reader for the bounded v10, train-only v3 wiring smoke.

The wrapper protocol is a separate contract from the historical P1 corpus.
The independent AR suffix is returned as labels, never as part of an arm view.
Each merged row is also checked against its original per-prompt v3 chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .fixed_probe import ProbeExample, ProbeObservation, ProbePrompt, _validate_prompts
from .packed import NUMERICAL_RECHECK_BACKEND_V2
from .packed_probe_numeric_v3 import (
    NUMERICAL_RECHECK_TOP2_ULPS,
    PACKED_PROBE_PROTOCOL_VERSION,
    canonical_bytes,
)


V10_TRAIN_SMOKE_SCHEMA = "v10_train_smoke_v1"
V10_TRAIN_SMOKE_SCOPE = "v10_train_smoke"
_SHA_HEX_LENGTH = 64


@dataclass(frozen=True)
class V10TrainWindow:
    """Causal observation and future labels remain separate typed fields."""

    observation: ProbeObservation
    labels: tuple[int, ...]
    trace_sha256: str
    chunk_sha256: str


@dataclass(frozen=True)
class V10TrainSmokeCorpus:
    windows: tuple[V10TrainWindow, ...]
    prompt_ids: tuple[str, ...]
    corpus_sha256: str
    protocol_sha256: str
    protocol: Mapping[str, Any]

    def training_windows(self) -> tuple[V10TrainWindow, ...]:
        """Exactly the first two complete L8 windows per frozen prompt."""
        selected: list[V10TrainWindow] = []
        for prompt_id in self.prompt_ids:
            complete = [w for w in self.windows
                        if w.observation.prompt_id == prompt_id and len(w.labels) == 8]
            if len(complete) < 2:
                raise ValueError(f"fewer than two full L8 windows: {prompt_id}")
            selected.extend(complete[:2])
        if len(selected) != 8 or len({w.observation.key for w in selected}) != 8:
            raise ValueError("train smoke requires eight distinct full windows")
        return tuple(selected)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_value(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != _SHA_HEX_LENGTH
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty map")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{name} has an invalid key")
    return value


def _confined_path(base: Path, relative: str, name: str) -> Path:
    rel = Path(relative)
    if (not relative or rel.is_absolute() or ".." in rel.parts
            or rel.drive or "\\" in relative or ":" in relative):
        raise ValueError(f"{name} must be a safe relative path")
    resolved = (base / rel).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"{name} escapes its root")
    return resolved


def _strict_prompt(record: Any) -> ProbePrompt:
    if not isinstance(record, dict):
        raise ValueError("prompt record must be an object")
    required = {"id", "family_id", "split", "task_type", "input_ids"}
    if not required.issubset(record):
        raise ValueError("prompt record lacks a required field")
    if record["split"] != "train" or not isinstance(record["input_ids"], list):
        raise ValueError("train smoke prompt has an invalid split or input_ids")
    if any(type(token) is not int or token < 0 for token in record["input_ids"]):
        raise ValueError("train smoke prompt has invalid token IDs")
    if any(not isinstance(record[key], str) or not record[key]
           for key in ("id", "family_id", "task_type")):
        raise ValueError("train smoke prompt identity is invalid")
    return ProbePrompt(record["id"], record["family_id"], record["split"],
                       tuple(record["input_ids"]), record["task_type"])


def load_v10_train_smoke(
    corpus_path: str | Path,
    protocol_path: str | Path,
    *,
    expected_parent_sha: str,
    expected_plan_sha: str,
    expected_source_sha: str,
    expected_model_hashes: Mapping[str, Any],
    expected_tokenizer_sha: Mapping[str, str],
    expected_p0_sha: str,
) -> V10TrainSmokeCorpus:
    """Validate a complete, frozen four-request v3 smoke before arm preparation.

    ``source_sha256`` in the wrapper is the per-file map relative to this source
    tree. ``source_zip_sha256`` binds the byte-exact frozen v10 archive. Model
    hashes are compared with the separately frozen parent manifest, without
    loading or rewriting any model file.
    """
    corpus_path = Path(corpus_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("train smoke protocol must be an object")
    expected = {
        "parent_manifest_sha256": expected_parent_sha,
        "collection_plan_sha256": expected_plan_sha,
        "source_zip_sha256": expected_source_sha,
        "packed_p0_report_sha256": expected_p0_sha,
    }
    if any(protocol.get(key) != _sha_value(value, key)
           for key, value in expected.items()):
        raise ValueError("train smoke parent/plan/source/P0 binding differs")
    if (protocol.get("schema_version") != V10_TRAIN_SMOKE_SCHEMA
            or protocol.get("scope") != V10_TRAIN_SMOKE_SCOPE
            or protocol.get("train_smoke") is not True
            or protocol.get("split") != "train"
            or protocol.get("full_pilot_complete") is not False
            or protocol.get("collection_protocol_version") != PACKED_PROBE_PROTOCOL_VERSION
            or protocol.get("numeric_decision_backend") != NUMERICAL_RECHECK_BACKEND_V2
            or protocol.get("numerical_recheck_top2_ulps") != NUMERICAL_RECHECK_TOP2_ULPS
            or protocol.get("window") != 8
            or protocol.get("max_new_tokens") != 128
            or protocol.get("max_probe_windows_per_prompt") != 8
            or type(protocol.get("eos_token_id")) is not int
            or protocol["eos_token_id"] < 0
            or type(protocol.get("shard_count")) is not int
            or protocol["shard_count"] < 1):
        raise ValueError("train smoke protocol is not the frozen v3 contract")
    if protocol.get("model_file_sha256") != dict(expected_model_hashes):
        raise ValueError("model hashes differ from parent manifest")
    if protocol.get("tokenizer_sha256") != dict(expected_tokenizer_sha):
        raise ValueError("tokenizer hashes differ from parent manifest")
    source_map = _mapping(protocol.get("source_sha256"), "source_sha256")
    source_root = Path(__file__).resolve().parent.parent
    for relative, digest in source_map.items():
        _sha_value(digest, f"source_sha256[{relative}]")
        path = _confined_path(source_root, relative, "source file")
        if not path.is_file() or _sha_file(path) != digest:
            raise ValueError(f"source file differs from frozen v10 archive: {relative}")
    for required in ("feedback_codec/packed.py", "feedback_codec/packed_probe_numeric_v3.py",
                     "feedback_codec/v10_train_smoke_reader.py"):
        if required not in source_map:
            raise ValueError(f"source map omits {required}")
    records = protocol.get("prompt_records")
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("train smoke requires exactly four fixed train prompts")
    prompts = [_strict_prompt(record) for record in records]
    _validate_prompts(prompts)
    prompt_ids = tuple(prompt.id for prompt in prompts)
    input_hashes = _mapping(protocol.get("prompt_input_sha256"), "prompt_input_sha256")
    chunk_paths = _mapping(protocol.get("chunk_paths"), "chunk_paths")
    chunk_hashes = _mapping(protocol.get("chunk_sha256"), "chunk_sha256")
    if any(set(mapping) != set(prompt_ids)
           for mapping in (input_hashes, chunk_paths, chunk_hashes)):
        raise ValueError("prompt/chunk inventory differs from four fixed prompts")
    for prompt in prompts:
        actual = hashlib.sha256(canonical_bytes(list(prompt.input_ids))).hexdigest()
        if input_hashes[prompt.id] != actual:
            raise ValueError(f"train prompt input hash differs: {prompt.id}")
    corpus_sha = _sha_value(protocol.get("fixed_prefix_sha256"),
                            "fixed_prefix_sha256")
    if not corpus_path.is_file() or _sha_file(corpus_path) != corpus_sha:
        raise ValueError("merged fixed-prefix JSONL differs from frozen SHA")
    # Reuse the v3 collector's complete row and request-level correctness gate.
    from scripts.build_packed_probe_numeric_v3 import _checked_chunk

    expected_rows: list[dict[str, Any]] = []
    row_chunk_hash: dict[tuple[str, int], str] = {}
    for prompt in prompts:
        relpath = chunk_paths[prompt.id]
        if not isinstance(relpath, str):
            raise ValueError("chunk path must be a string")
        path = _confined_path(protocol_path.parent, relpath, "chunk path")
        digest = _sha_value(chunk_hashes[prompt.id], "chunk_sha256")
        if not path.is_file() or _sha_file(path) != digest:
            raise ValueError(f"atomic v3 chunk SHA differs: {prompt.id}")
        chunk = _checked_chunk(path, prompt, expected_plan_sha,
                               protocol["shard_count"], protocol["eos_token_id"])
        for row in chunk["examples"]:
            key = (row["prompt_id"], row["round_index"])
            if key in row_chunk_hash:
                raise ValueError("duplicate prompt/round in v3 chunks")
            row_chunk_hash[key] = digest
            expected_rows.append(row)
    rows = []
    for line_number, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"fixed-prefix row {line_number} is not an object")
        rows.append(row)
    if rows != expected_rows or len(rows) > 32:
        raise ValueError("merged train rows differ from validated atomic v3 chunks")
    windows: list[V10TrainWindow] = []
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("nontrain row in train smoke")
        example = ProbeExample.from_json(row)
        observation = example.observation
        if (observation.remaining_budget <= 0
                or observation.target_hidden_prefix != observation.valid_prefix
                or not 1 <= len(example.target_future_ids) <= 8):
            raise ValueError("terminal, future, or invalid-boundary row in train smoke")
        key = observation.key
        windows.append(V10TrainWindow(
            observation, example.target_future_ids,
            _sha_value(row["packed_trace_sha256"], "packed_trace_sha256"),
            row_chunk_hash[key],
        ))
    return V10TrainSmokeCorpus(tuple(windows), prompt_ids, corpus_sha,
                               _sha_file(protocol_path), protocol)
