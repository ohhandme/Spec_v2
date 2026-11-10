"""Focused CPU gates for the versioned four-prompt v10 train-smoke reader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from feedback_codec.fixed_probe import ProbePrompt
from feedback_codec.packed import NUMERICAL_RECHECK_BACKEND_V2
from feedback_codec.packed_probe_numeric_v3 import (
    NUMERICAL_RECHECK_TOP2_ULPS,
    PACKED_PROBE_PROTOCOL_VERSION,
    canonical_bytes,
    collect_packed_fixed_probe_numeric_v3,
)
from feedback_codec.v10_train_smoke_reader import load_v10_train_smoke
from scripts.build_packed_probe_numeric_v3 import _prompt_digest, _shard_of
from test_packed import FakeTarget, ScriptedDraft


PARENT = "1" * 64
PLAN = "2" * 64
SOURCE = "3" * 64
P0 = "4" * 64
MODELS = {"target": {"model.safetensors": "5" * 64},
          "draft": {"model.safetensors": "6" * 64}}
TOKENIZER = {"tokenizer.json": "7" * 64}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_map() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    names = ("feedback_codec/packed.py", "feedback_codec/packed_probe_numeric_v3.py",
             "feedback_codec/v10_train_smoke_reader.py")
    return {name: _sha(root / name) for name in names}


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    prompts = []
    rows = []
    chunk_paths = {}
    chunk_hashes = {}
    for index in range(4):
        input_id = index + 1
        prompt = ProbePrompt(f"toy-{index}", f"family-{index}", "train",
                             (input_id,), "chat")
        prompts.append(prompt)
        target = FakeTarget([input_id], [10, 12, 13, 0])
        draft = ScriptedDraft([[10, 11, 0], [13, 0]])
        corpus = collect_packed_fixed_probe_numeric_v3(
            target=target, raw_draft=draft, prompts=[prompt], eos_token_id=0)
        assert len(corpus.rows) == 1
        chunk = {
            "prompt_id": prompt.id, "prompt_sha256": _prompt_digest(prompt),
            "plan_sha256": PLAN, "shard_index": _shard_of(prompt, 1),
            "shard_count": 1, "eos_token_id": 0, "window": 8,
            "max_probe_windows_per_prompt": 8, "capture_target_logits": True,
            "feature_provenance_gate": True,
            "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
            "numerical_recheck_top2_ulps": NUMERICAL_RECHECK_TOP2_ULPS,
            "request_audit": corpus.request_audit[0], "audit": corpus.audit,
            "examples": [row.to_json() for row in corpus.rows],
        }
        chunk["checksum_sha256"] = hashlib.sha256(canonical_bytes(chunk)).hexdigest()
        path = tmp_path / f"chunk-{index}.json"
        path.write_text(json.dumps(chunk), encoding="utf-8")
        chunk_paths[prompt.id] = path.name
        chunk_hashes[prompt.id] = _sha(path)
        rows.extend(chunk["examples"])
    corpus_path = tmp_path / "train_smoke.jsonl"
    corpus_path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                           encoding="utf-8")
    protocol = {
        "schema_version": "v10_train_smoke_v1", "scope": "v10_train_smoke",
        "train_smoke": True, "split": "train", "full_pilot_complete": False,
        "collection_protocol_version": PACKED_PROBE_PROTOCOL_VERSION,
        "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
        "numerical_recheck_top2_ulps": NUMERICAL_RECHECK_TOP2_ULPS,
        "window": 8, "max_new_tokens": 128, "max_probe_windows_per_prompt": 8,
        "eos_token_id": 0, "shard_count": 1,
        "parent_manifest_sha256": PARENT, "collection_plan_sha256": PLAN,
        "source_zip_sha256": SOURCE, "source_sha256": _source_map(),
        "packed_p0_report_sha256": P0,
        "model_file_sha256": MODELS, "tokenizer_sha256": TOKENIZER,
        "fixed_prefix_sha256": _sha(corpus_path),
        "prompt_records": [{"id": p.id, "family_id": p.family_id,
                            "split": p.split, "task_type": p.task_type,
                            "input_ids": list(p.input_ids)} for p in prompts],
        "prompt_input_sha256": {
            p.id: hashlib.sha256(canonical_bytes(list(p.input_ids))).hexdigest()
            for p in prompts},
        "chunk_paths": chunk_paths, "chunk_sha256": chunk_hashes,
    }
    protocol_path = tmp_path / "train_smoke_protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return corpus_path, protocol_path, protocol


def _load(corpus_path: Path, protocol_path: Path):
    return load_v10_train_smoke(
        corpus_path, protocol_path, expected_parent_sha=PARENT,
        expected_plan_sha=PLAN, expected_source_sha=SOURCE,
        expected_model_hashes=MODELS, expected_tokenizer_sha=TOKENIZER,
        expected_p0_sha=P0)


def test_valid_v3_chunks_are_read_as_separate_observation_and_labels(tmp_path):
    corpus_path, protocol_path, _ = _fixture(tmp_path)
    corpus = _load(corpus_path, protocol_path)
    assert len(corpus.windows) == 4
    assert corpus.prompt_ids == ("toy-0", "toy-1", "toy-2", "toy-3")
    assert all(window.observation.split == "train" for window in corpus.windows)
    assert all(window.labels == (13, 0) for window in corpus.windows)
    assert all(not hasattr(window.observation, "target_future_ids")
               for window in corpus.windows)
    with pytest.raises(ValueError, match="two full L8"):
        corpus.training_windows()


@pytest.mark.parametrize("change, error", [
    (lambda p: p.update(split="val"), "contract"),
    (lambda p: p.update(full_pilot_complete=True), "contract"),
    (lambda p: p.update(source_zip_sha256="9" * 64), "binding"),
    (lambda p: p.update(packed_p0_report_sha256="9" * 64), "binding"),
    (lambda p: p.update(numerical_recheck_top2_ulps=0), "contract"),
    (lambda p: p.update(model_file_sha256={"target": {}}), "model hashes"),
])
def test_protocol_mismatch_is_rejected(tmp_path, change, error):
    corpus_path, protocol_path, protocol = _fixture(tmp_path)
    change(protocol)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        _load(corpus_path, protocol_path)


def test_rehashed_merged_future_label_tamper_is_rejected(tmp_path):
    corpus_path, protocol_path, protocol = _fixture(tmp_path)
    rows = [json.loads(line) for line in corpus_path.read_text().splitlines()]
    rows[0]["target_future_ids"] = [99, 0]
    corpus_path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                           encoding="utf-8")
    protocol["fixed_prefix_sha256"] = _sha(corpus_path)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from validated atomic"):
        _load(corpus_path, protocol_path)


def test_rehashed_chunk_future_label_tamper_is_rejected(tmp_path):
    corpus_path, protocol_path, protocol = _fixture(tmp_path)
    path = tmp_path / protocol["chunk_paths"]["toy-0"]
    chunk = json.loads(path.read_text())
    chunk["examples"][0]["target_future_ids"] = [99, 0]
    chunk.pop("checksum_sha256")
    chunk["checksum_sha256"] = hashlib.sha256(canonical_bytes(chunk)).hexdigest()
    path.write_text(json.dumps(chunk), encoding="utf-8")
    protocol["chunk_sha256"]["toy-0"] = _sha(path)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="future label"):
        _load(corpus_path, protocol_path)


def test_missing_source_and_cross_split_are_rejected(tmp_path):
    corpus_path, protocol_path, protocol = _fixture(tmp_path)
    protocol["source_sha256"].pop("feedback_codec/packed.py")
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="source map omits"):
        _load(corpus_path, protocol_path)
    corpus_path, protocol_path, protocol = _fixture(tmp_path)
    protocol["prompt_records"][0]["split"] = "test"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid split"):
        _load(corpus_path, protocol_path)
