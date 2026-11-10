"""CPU-only one-pass stream preservation and v3 row equivalence."""

from __future__ import annotations

import pytest

from feedback_codec.fixed_probe import ProbePrompt
from feedback_codec.packed_probe_numeric_v3 import collect_packed_fixed_probe_numeric_v3
from feedback_codec.v10_collect_one import collect_one
from test_packed import FakeTarget, ScriptedDraft


def _toy():
    prompt = ProbePrompt("v10-toy", "v10-family", "train", (1,), "chat")
    target = FakeTarget([1], [10, 12, 13, 0])
    draft = ScriptedDraft([[10, 11, 0], [13, 0]])
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


def test_one_pass_rows_match_v3_and_keep_both_full_streams():
    prompt, target, draft = _toy()
    corpus, receipt = collect_one(target, draft, prompt, 0)
    prompt2, target2, draft2 = _toy()
    reference = collect_packed_fixed_probe_numeric_v3(
        target=target2, raw_draft=draft2, prompts=[prompt2], eos_token_id=0)
    def without_timing(value):
        if isinstance(value, dict):
            return {key: without_timing(item) for key, item in value.items()
                    if not key.endswith("wall_ms")
                    and key != "packed_trace_sha256"}
        if isinstance(value, list):
            return [without_timing(item) for item in value]
        return value

    assert [without_timing(r.to_json()) for r in corpus.rows] == [
        without_timing(r.to_json()) for r in reference.rows]
    assert corpus.audit["probe_windows"] == reference.audit["probe_windows"] == 1
    assert receipt["exact"] is True
    assert receipt["packed_ids"] == receipt["oracle_ids"] == [10, 12, 13, 0]
    assert receipt["packed_eos"] and receipt["oracle_eos"]
    assert not receipt["packed_truncated"] and not receipt["oracle_truncated"]
    assert receipt["first_difference_index"] is None
    assert receipt["rounds"][0]["feature_source"] == (
        "canonical_full_prefix_numeric_recheck")


def test_one_pass_mismatch_returns_full_streams_without_training_rows(monkeypatch):
    import feedback_codec.v10_collect_one as module

    prompt, target, draft = _toy()
    monkeypatch.setattr(module, "autoregressive_generate",
                        lambda *_: (10, 11, 13, 0))
    corpus, receipt = collect_one(target, draft, prompt, 0)
    assert corpus.rows == ()
    assert corpus.audit["exact_requests"] == 0
    assert receipt["exact"] is False
    assert receipt["first_difference_index"] == 1
    assert receipt["packed_ids"] == [10, 12, 13, 0]
    assert receipt["oracle_ids"] == [10, 11, 13, 0]
    assert receipt["rounds"]


def test_one_pass_refuses_sealed_split():
    prompt, target, draft = _toy()
    sealed = ProbePrompt(prompt.id, prompt.family_id, "test",
                         prompt.input_ids, prompt.task_type)
    with pytest.raises(ValueError, match="train"):
        collect_one(target, draft, sealed, 0)
