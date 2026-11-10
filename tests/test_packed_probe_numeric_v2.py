"""CPU gates for the isolated costed numeric-recheck probe collector."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from feedback_codec.fixed_probe import ProbePrompt
from feedback_codec.packed import NUMERICAL_RECHECK_BACKEND
from feedback_codec.packed_probe_numeric_v2 import (
    PACKED_PROBE_PROTOCOL_VERSION, NUMERICAL_RECHECK_TOP2_MARGIN,
    canonical_bytes, collect_packed_fixed_probe_numeric_v2,
)
from scripts.build_packed_probe_numeric_v2 import (
    _checked_chunk, _prompt_digest, _shard_of, _validate_real_p0,
)
from test_packed import FakeTarget, ScriptedDraft


def toy(*, tie: bool = False):
    prompt = ProbePrompt("toy-numeric", "toy-family", "train", (1,), "chat")
    target = FakeTarget([1], [10, 12, 13, 0])
    draft = ScriptedDraft([[10, 11, 0], [13, 0]])
    if tie:
        original = target.model.forward

        def shape_sensitive(**kwargs):
            output = original(**kwargs)
            chunk = tuple(int(x) for x in kwargs["input_ids"][0].tolist())
            if kwargs["output_hidden_states"] and chunk == (1, 10, 11, 0):
                output.logits[0, 1, 11] = 100.0
                output.logits[0, 1, 12] = 100.0
            return output

        target.model.forward = shape_sensitive
    return prompt, target, draft


def test_numeric_collector_keeps_exact_request_and_costed_valid_row():
    prompt, target, draft = toy(tie=True)
    corpus = collect_packed_fixed_probe_numeric_v2(
        target=target, raw_draft=draft, prompts=[prompt], eos_token_id=0)
    assert corpus.audit["exact_requests"] == 1
    assert corpus.audit["probe_windows"] == 1
    assert corpus.audit["runtime_extra_target_forwards"] >= 1
    assert corpus.request_audit[0]["output_ids"] == [10, 12, 13, 0]
    assert corpus.request_audit[0]["numeric_decision_backend"] == NUMERICAL_RECHECK_BACKEND
    row = corpus.rows[0].to_json()
    assert row["collection_protocol_version"] == PACKED_PROBE_PROTOCOL_VERSION
    assert row["numerical_recheck_top2_margin"] == NUMERICAL_RECHECK_TOP2_MARGIN
    assert row["packed_feature_source"] == "numeric_recheck_query_forward"
    assert row["target_logit_summary"]["greedy_token_id"] == 12
    trace = row["packed_trace"]
    assert trace["cache_commit_source"] == "online_short_recheck"
    assert trace["numerical_recheck_rows"] == [1]
    assert trace["numerical_recheck_extra_target_forwards"] == 1
    assert trace["feature_provenance"]["feature_replay_logits_bitwise_equal"] is True
    assert trace["feature_provenance"]["cache_source"] == "same_online_preverify_KV_snapshot"
    assert "argmax_ids" not in trace  # Off-path rows are never training features.
    assert trace["used_argmax_ids"] == [10, 12]
    assert row["packed_trace_sha256"] == hashlib.sha256(
        canonical_bytes(trace)).hexdigest()
    assert all(codes is None for _, codes, _ in draft.calls)


def test_numeric_collector_fails_closed_on_independent_ar_mismatch(monkeypatch):
    import feedback_codec.packed_probe_numeric_v2 as module

    prompt, target, draft = toy(tie=True)
    monkeypatch.setattr(module, "autoregressive_generate", lambda *_: (10, 11, 13, 0))
    with pytest.raises(RuntimeError, match="independent AR"):
        collect_packed_fixed_probe_numeric_v2(
            target=target, raw_draft=draft, prompts=[prompt], eos_token_id=0)


def test_numeric_collector_fails_closed_on_bad_provenance(monkeypatch):
    import feedback_codec.packed_probe_numeric_v2 as module

    original = module.PackedSpeculativeDecoder.generate

    def changed(self, prompt, max_new_tokens):
        result = original(self, prompt, max_new_tokens)
        first = replace(result.traces[0], feature_provenance={
            **result.traces[0].feature_provenance, "passed": False})
        return replace(result, traces=(first, *result.traces[1:]))

    monkeypatch.setattr(module.PackedSpeculativeDecoder, "generate", changed)
    prompt, target, draft = toy(tie=True)
    with pytest.raises(RuntimeError, match="provenance"):
        collect_packed_fixed_probe_numeric_v2(
            target=target, raw_draft=draft, prompts=[prompt], eos_token_id=0)


def test_numeric_chunk_integrity_and_future_label(tmp_path):
    prompt, target, draft = toy(tie=True)
    corpus = collect_packed_fixed_probe_numeric_v2(
        target=target, raw_draft=draft, prompts=[prompt], eos_token_id=0)
    chunk = {
        "prompt_id": prompt.id, "prompt_sha256": _prompt_digest(prompt),
        "plan_sha256": "plan-v2", "shard_index": _shard_of(prompt, 1),
        "shard_count": 1, "eos_token_id": 0, "window": 8,
        "max_probe_windows_per_prompt": 8, "capture_target_logits": True,
        "feature_provenance_gate": True,
        "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND,
        "numerical_recheck_top2_margin": NUMERICAL_RECHECK_TOP2_MARGIN,
        "request_audit": corpus.request_audit[0], "audit": corpus.audit,
        "examples": [row.to_json() for row in corpus.rows],
    }
    path = tmp_path / "numeric-chunk.json"

    def save() -> None:
        chunk.pop("checksum_sha256", None)
        chunk["checksum_sha256"] = hashlib.sha256(canonical_bytes(chunk)).hexdigest()
        path.write_text(json.dumps(chunk), encoding="utf-8")

    save()
    assert _checked_chunk(path, prompt, "plan-v2", 1, 0)["audit"]["probe_windows"] == 1
    chunk["examples"][0]["target_future_ids"] = [99, 0]
    save()
    with pytest.raises(ValueError, match="future label"):
        _checked_chunk(path, prompt, "plan-v2", 1, 0)
    chunk["examples"][0]["target_future_ids"] = [13, 0]
    chunk["examples"][0]["packed_trace"]["numerical_recheck_extra_target_forwards"] = 0
    chunk["examples"][0]["packed_trace_sha256"] = hashlib.sha256(
        canonical_bytes(chunk["examples"][0]["packed_trace"])).hexdigest()
    save()
    with pytest.raises(ValueError, match="provenance|recheck"):
        _checked_chunk(path, prompt, "plan-v2", 1, 0)


def test_old_p0_scope_is_rejected_before_loading_models(tmp_path):
    report = tmp_path / "old-p0.json"
    report.write_text(json.dumps({"state": "passed",
                                  "scope": "real_qwen3_bf16_packed_target_committed_path_p0",
                                  "dtype": "bfloat16", "window": 8}), encoding="utf-8")
    with pytest.raises(ValueError, match="P0 gate"):
        _validate_real_p0(report, target=tmp_path, draft=tmp_path)


def test_sealed_test_split_is_refused():
    prompt, target, draft = toy()
    with pytest.raises(ValueError, match="train/val"):
        collect_packed_fixed_probe_numeric_v2(
            target=target, raw_draft=draft, prompts=[replace(prompt, split="test")],
            eos_token_id=0)
