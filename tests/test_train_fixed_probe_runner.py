"""The training runner must reject corpus/model drift before optimization."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
from torch import nn

from feedback_codec.codec import CodecConfig, MatchedP1Codec
from feedback_codec.fixed_probe import EVAL_PROTOCOL_VERSION, ProbeExample, ProbeObservation
from scripts.train_fixed_probe import (
    ROOT, _edge_initial_hash, _load_locked_train, _model_hashes,
    _histogram_summary, _seeds, _verify_p1_gpu_gate,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_train_runner_locks_split_corpus_and_draft_hash(tmp_path) -> None:
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (draft / "model.safetensors").write_bytes(b"test fixture, not a real weight file")
    observation = ProbeObservation(
        prompt_id="p", family_id="f", split="train", task_type="toy",
        round_index=0, source_prefix=(1,), proposals=(3,), valid_prefix=(1,),
        accepted_count=0, rejected_token=3, correction_token=2,
        corrected_prefix=(1, 2), target_hidden=torch.randn(12),
        target_hidden_prefix=(1,), remaining_budget=8, window=8,
    )
    corpus = tmp_path / "fixed_prefix.jsonl"
    corpus.write_text(json.dumps(ProbeExample(observation, (4, 5)).to_json()) + "\n",
                      encoding="utf-8")
    protocol = tmp_path / "collection_protocol.json"
    protocol.write_text(json.dumps({
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        "split": "train", "window": 8, "dtype": "bfloat16",
        "fixed_prefix_sha256": _sha256(corpus),
        "draft_file_sha256": _model_hashes(draft),
        "target_file_sha256": {"model.safetensors": "frozen-target-hash"},
    }), encoding="utf-8")
    rows, _ = _load_locked_train(corpus, protocol, draft)
    assert len(rows) == 1 and rows[0].observation.prompt_id == "p"
    with pytest.raises(ValueError, match="locked L=8 smoke"):
        _load_locked_train(corpus, protocol, draft, smoke=True)
    corpus.write_text(corpus.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after collection"):
        _load_locked_train(corpus, protocol, draft)


def test_same_seed_gives_identical_edge_initialization_for_b_c_d() -> None:
    config = CodecConfig(12, 10, 17, feature_dim=9, hidden_width=20,
                         rank=4, code_dim=3)
    head = nn.Linear(10, 17, bias=False)
    hashes = []
    for mode in ("edge_only", "protocol_only", "target_8B"):
        torch.manual_seed(29)
        hashes.append(_edge_initial_hash(MatchedP1Codec(config, head, mode=mode)))
    assert len(set(hashes)) == 1
    assert _seeds("1,2,3") == (1, 2, 3)
    with pytest.raises(Exception):
        _seeds("1,1")


def test_formal_training_requires_real_gpu_gate_and_unchanged_source(tmp_path) -> None:
    model_hashes = {"model.safetensors": "model-sha"}
    protocol = {"target_file_sha256": model_hashes,
                "draft_file_sha256": model_hashes,
                "prompt_manifest_sha256": "c" * 64}
    modes = ("edge_only", "protocol_only", "target_8B")
    report = {
        "state": "passed", "scope": "matched_p1_real_gpu_p0",
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        "dtype": "bfloat16", "window": 8,
        "device": "cuda:0", "cuda": True, "gpu_name": "RTX 4090",
        "target_file_sha256": model_hashes,
        "draft_file_sha256": model_hashes,
        "dataset_sha256": "c" * 64,
        "target_frozen": True, "draft_frozen": True,
        "real_p0_report_sha256": "a" * 64,
        "smoke_probe_count": 2, "smoke_request_ids": ["p1", "p2"],
        "checks": {mode: {"passed": True, "exact_target_ar": True,
                          "finite_logits": True, "same_prefix": True,
                          "exact_requests": 1,
                          "cloud_message_bytes": 8 if mode == "target_8B" else 0}
                   for mode in modes},
        "gradient_checks": {mode: {"passed": True, "local_heads_nonzero": True,
                                   "repair_nonzero": True,
                                   "cloud_nonzero": mode == "target_8B"}
                            for mode in modes},
        "source_sha256": {name: _sha256(ROOT / name) for name in (
            "feedback_codec/codec.py", "feedback_codec/fixed_probe.py",
            "feedback_codec/hf_backend.py", "feedback_codec/decoding.py",
        )},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _verify_p1_gpu_gate(path, protocol)["state"] == "passed"
    report["checks"]["target_8B"]["passed"] = False
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="exact B/C/D check"):
        _verify_p1_gpu_gate(path, protocol)
    report["checks"]["target_8B"]["passed"] = True
    report["source_sha256"]["feedback_codec/codec.py"] = "tampered"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        _verify_p1_gpu_gate(path, protocol)


def test_code_histogram_reports_each_group_and_effective_words() -> None:
    stats = _histogram_summary(
        [[[0, 1], [1, 1]], [[0, 1], [1, 2]]],
        heads=2, groups=2, choices=4,
    )
    assert stats["rows"] == 2
    assert stats["histogram"][0][0] == [2, 0, 0, 0]
    assert stats["effective_codewords"] == [[1, 1], [1, 2]]
    assert stats["entropy_bits"][1][1] == 1.0
