"""CPU causal and integrity checks for the versioned packed P1 collector."""

from dataclasses import replace
import hashlib
import json

import pytest

from feedback_codec.fixed_probe import ProbeExample, ProbePrompt
from feedback_codec.packed_probe import (
    PACKED_PROBE_PROTOCOL_VERSION, canonical_bytes, collect_packed_fixed_probe,
)
from test_packed import FakeTarget, ScriptedDraft


def _toy():
    prompt = ProbePrompt("toy-1", "family-1", "train", (1,), "chat")
    target = FakeTarget([1], [10, 11, 12, 0])
    # First round accepts 10, rejects 99 and commits Target 11.  The next
    # round accepts 12 and EOS.  The rejected packed suffix must not leak.
    draft = ScriptedDraft([[10, 99, 0], [12, 0]])
    return prompt, target, draft


def test_packed_probe_uses_online_valid_row_and_independent_future():
    prompt, target, draft = _toy()
    corpus = collect_packed_fixed_probe(
        target=target, raw_draft=draft, prompts=[prompt], eos_token_id=0)
    assert corpus.audit["exact_requests"] == 1
    assert corpus.audit["probe_windows"] == 1
    assert corpus.audit["independent_ar_prefix_exact_windows"] == 1
    assert corpus.request_audit[0]["output_ids"] == [10, 11, 12, 0]
    row = corpus.rows[0].to_json()
    assert row["collection_protocol_version"] == PACKED_PROBE_PROTOCOL_VERSION
    assert row["online_feature_provenance_passed"] is True
    assert row["independent_ar_prefix_exact"] is True
    assert row["source_prefix"] == [1]
    assert row["valid_prefix"] == [1, 10]
    assert row["corrected_prefix"] == [1, 10, 11]
    assert row["target_future_ids"] == [12, 0]
    assert row["target_hidden_prefix"] == [1, 10]
    assert row["target_logit_summary"]["greedy_token_id"] == 11
    assert row["packed_trace"]["used_argmax_ids"] == [10, 11]
    assert "used_top2_margins" not in row["packed_trace"]
    assert row["packed_trace_sha256"] == hashlib.sha256(
        canonical_bytes(row["packed_trace"])).hexdigest()
    restored = ProbeExample.from_json(row)
    assert restored.target_future_ids == (12, 0)
    assert all(codes is None for _, codes, _ in draft.calls)


def test_packed_probe_fails_closed_on_independent_ar_mismatch(monkeypatch):
    import feedback_codec.packed_probe as module

    prompt, target, draft = _toy()
    monkeypatch.setattr(module, "autoregressive_generate", lambda *_: (10, 99, 0))
    with pytest.raises(RuntimeError, match="independent AR"):
        collect_packed_fixed_probe(
            target=target, raw_draft=draft, prompts=[prompt], eos_token_id=0)


def test_packed_probe_fails_closed_on_bad_online_feature_provenance(monkeypatch):
    import feedback_codec.packed_probe as module

    base = module.PackedSpeculativeDecoder
    original = base.generate

    def altered(self, prompt, max_new_tokens):
        result = original(self, prompt, max_new_tokens)
        first = replace(result.traces[0], feature_provenance={
            **result.traces[0].feature_provenance, "passed": False})
        return replace(result, traces=(first, *result.traces[1:]))

    monkeypatch.setattr(base, "generate", altered)
    prompt, target, draft = _toy()
    with pytest.raises(RuntimeError, match="provenance"):
        collect_packed_fixed_probe(
            target=target, raw_draft=draft, prompts=[prompt], eos_token_id=0)


def test_packed_probe_rejects_test_split_even_with_toy_model():
    prompt, target, draft = _toy()
    sealed = replace(prompt, split="test")
    with pytest.raises(ValueError, match="train/val"):
        collect_packed_fixed_probe(
            target=target, raw_draft=draft, prompts=[sealed], eos_token_id=0)


def test_canonical_trace_digest_has_no_post_rejection_target_ids():
    prompt, target, draft = _toy()
    row = collect_packed_fixed_probe(
        target=target, raw_draft=draft, prompts=[prompt], eos_token_id=0
    ).rows[0].to_json()
    trace = row["packed_trace"]
    assert "argmax_ids" not in trace
    assert "hypothetical_argmax_ids" not in trace
    assert "hidden_max_abs_diff_fp32" not in trace["feature_provenance"]
    assert "logits_max_abs_diff_fp32" not in trace["feature_provenance"]
    assert "target_future_ids" not in json.dumps(trace)


def test_chunk_resume_rechecks_trace_and_independent_ar_suffix(tmp_path):
    from scripts.build_packed_probe import _checked_chunk, _prompt_digest, _shard_of

    prompt, target, draft = _toy()
    corpus = collect_packed_fixed_probe(
        target=target, raw_draft=draft, prompts=[prompt], eos_token_id=0)
    chunk = {
        "prompt_id": prompt.id, "prompt_sha256": _prompt_digest(prompt),
        "plan_sha256": "plan", "shard_index": _shard_of(prompt, 1),
        "shard_count": 1, "eos_token_id": 0, "window": 8,
        "max_probe_windows_per_prompt": 8, "capture_target_logits": True,
        "feature_provenance_gate": True, "request_audit": corpus.request_audit[0],
        "audit": corpus.audit, "examples": [row.to_json() for row in corpus.rows],
    }
    chunk["checksum_sha256"] = hashlib.sha256(canonical_bytes(chunk)).hexdigest()
    path = tmp_path / "chunk.json"
    path.write_text(json.dumps(chunk), encoding="utf-8")
    assert _checked_chunk(path, prompt, "plan", 1, 0)["audit"]["probe_windows"] == 1
    chunk["examples"][0]["target_future_ids"] = [99, 0]
    chunk.pop("checksum_sha256")
    chunk["checksum_sha256"] = hashlib.sha256(canonical_bytes(chunk)).hexdigest()
    path.write_text(json.dumps(chunk), encoding="utf-8")
    with pytest.raises(ValueError, match="future label"):
        _checked_chunk(path, prompt, "plan", 1, 0)
