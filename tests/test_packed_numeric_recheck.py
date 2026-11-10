"""Candidate-only CPU checks for a costed packed BF16 numerical recheck."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from feedback_codec.decoding import autoregressive_generate
from feedback_codec.packed import (
    HFPackedTargetSession, NUMERICAL_RECHECK_BACKEND,
    PackedSpeculativeDecoder,
)
from test_packed import Encoder, FakeTarget, ScriptedDraft


def shape_sensitive_target() -> FakeTarget:
    """Emulate a packed tie that wrongly accepts an off-oracle proposal."""
    target = FakeTarget([1], [10, 12, 13])
    original = target.model.forward

    def shaped_forward(**kwargs):
        result = original(**kwargs)
        if not kwargs["output_hidden_states"]:
            return result
        chunk = tuple(int(token) for token in kwargs["input_ids"][0].tolist())
        if chunk == (1, 10, 11):
            # The normal packed L2 query has a tie at row1. The smaller ID
            # equals the draft proposal, so blindly accepting is wrong.
            result.logits[0, 1, 11] = 100.0
            result.logits[0, 1, 12] = 100.0
            result.hidden_states[-1][0, 1, 0] = 202.0
        elif chunk == (1, 10):
            # The same-online-KV short query ends at row1 and agrees with
            # independent full-prefix greedy Target. Mark its hidden source.
            result.hidden_states[-1][0, 1, 0] = 102.0
        return result

    target.model.forward = shaped_forward
    return target


def test_tie_recheck_changes_acceptance_before_commit_and_provenance():
    target = shape_sensitive_target()
    unguarded = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 11], [13]]), None, window=2,
    ).generate([1], 3)
    assert unguarded.generation.tokens != autoregressive_generate(
        target, [1], 3)
    encoder = Encoder()
    draft = ScriptedDraft([[10, 11], [13]])
    result = PackedSpeculativeDecoder(
        target, draft, encoder, window=2,
        capture_target_logits=True, feature_provenance_gate=True,
        numerical_recheck_top2_margin=0.0,
    ).generate([1], 3)
    assert result.generation.tokens == autoregressive_generate(
        target, [1], 3)
    first = result.generation.rounds[0]
    assert first.proposals == (10, 11)
    assert first.accepted_count == 1
    assert first.rejected_token == 11
    assert first.correction_token == 12
    assert first.committed_tokens == (10, 12)
    assert first.observation_hidden.item() == 102.0
    assert first.observation_logit_summary.greedy_token_id == 12
    assert encoder.calls[0] == (102.0, 11.0, 12.0, 1)
    trace = result.traces[0]
    assert trace.decision_backend == NUMERICAL_RECHECK_BACKEND
    assert trace.numerical_recheck_rows == (1,)
    assert trace.numerical_recheck_extra_target_forwards == 1
    assert trace.numerical_recheck_snapshot_wall_ms > 0
    assert trace.numerical_recheck_forward_wall_ms > 0
    assert trace.numerical_recheck_total_wall_ms >= (
        trace.numerical_recheck_snapshot_wall_ms
        + trace.numerical_recheck_forward_wall_ms)
    assert trace.cache_length_raw == 3
    assert trace.cache_length_after == 2
    assert trace.cache_commit_source == "online_short_recheck"
    assert result.generation.rounds[1].prefix_before == (1, 10, 12)
    assert result.traces[1].cache_length_before == 2
    causal = trace.feature_provenance
    assert causal["source"] == "numeric_recheck_query_forward"
    assert causal["valid_row_query_length"] == 2
    assert causal["reason"] == "rechecked_query_ends_at_valid_row"
    assert causal["snapshot_replay_hidden_bitwise_equal"]
    assert causal["snapshot_replay_logits_bitwise_equal"]
    assert causal["passed"]
    assert causal["numerical_recheck_extra_target_forwards"] == 1

    # In particular, a changed decision must not reuse the numerically
    # different long packed KV, even though cropping it would have right size.
    direct = HFPackedTargetSession(
        shape_sensitive_target(), [1],
        numerical_recheck_top2_margin=0.0)
    verification = direct.verify([10, 11])
    assert verification.cache.tokens == (1, 10, 11)
    assert verification.decision_cache.tokens == (1, 10)
    assert direct.commit(verification, 1, 12) == 2
    assert direct._cache is verification.decision_cache
    assert direct._cache.tokens == (1, 10)


def test_low_margin_after_first_rejection_is_not_rechecked():
    target = shape_sensitive_target()
    previous = target.model.forward

    def tied_off_path(**kwargs):
        result = previous(**kwargs)
        if (kwargs["output_hidden_states"]
                and tuple(kwargs["input_ids"][0].tolist()) == (1, 99, 11)):
            result.logits[0, 1, 11] = 100.0
            result.logits[0, 1, 12] = 100.0
        return result

    target.model.forward = tied_off_path
    # Directly verify both proposed positions. A one-token generation budget
    # would shorten the actual proposal to one token before this check.
    session = HFPackedTargetSession(
        target, [1], numerical_recheck_top2_margin=0.0)
    verification = session.verify([99, 11])
    assert verification.argmax_ids[0] == 10
    assert verification.top2_margins[1] == 0.0
    assert verification.numerical_recheck_rows == ()
    assert verification.numerical_recheck_extra_target_forwards == 0
    assert verification.numerical_recheck_snapshot_wall_ms > 0
    assert verification.numerical_recheck_total_wall_ms >= (
        verification.numerical_recheck_snapshot_wall_ms)
    assert session.commit(verification, 0, 10) == 1


def test_tied_bonus_gets_costed_full_prefix_recheck():
    target = FakeTarget([1], [10, 12])
    original = target.model.forward

    def cached_bonus_tie(**kwargs):
        result = original(**kwargs)
        chunk = tuple(int(token) for token in kwargs["input_ids"][0].tolist())
        if chunk == (1, 10) and kwargs["output_hidden_states"]:
            if kwargs["use_cache"]:
                result.logits[0, 1, 11] = 100.0
                result.logits[0, 1, 12] = 100.0
                result.hidden_states[-1][0, 1, 0] = 202.0
            else:
                result.hidden_states[-1][0, 1, 0] = 102.0
        return result

    target.model.forward = cached_bonus_tie
    raw = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10]]), None, window=1,
    ).generate([1], 2)
    assert raw.generation.tokens == (10, 11)
    corrected = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10]]), None, window=1,
        capture_target_logits=True, feature_provenance_gate=True,
        numerical_recheck_top2_margin=0.0,
    ).generate([1], 2)
    assert corrected.generation.tokens == autoregressive_generate(
        target, [1], 2) == (10, 12)
    trace = corrected.traces[0]
    assert trace.numerical_recheck_rows == (1,)
    assert trace.numerical_recheck_extra_target_forwards == 2
    assert trace.cache_length_after == 2
    assert trace.cache_commit_source == "canonical_full_prefix_refill"
    feature = trace.feature_provenance
    assert feature["source"] == "canonical_full_prefix_numeric_recheck"
    assert feature["cache_source"] == "fresh_full_prefix_no_KV"
    assert feature["reason"] == "canonical_full_prefix_ends_at_valid_row"
    assert feature["feature_replay_hidden_bitwise_equal"]
    assert feature["feature_replay_logits_bitwise_equal"]
    assert feature["passed"]
    assert corrected.generation.rounds[0].observation_hidden.item() == 102.0
    assert corrected.generation.rounds[0].observation_logit_summary.greedy_token_id == 12

    # With no output budget for a bonus, no canonical prefill is needed.
    terminal = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10]]), None, window=1,
        numerical_recheck_top2_margin=0.0,
    ).generate([1], 1)
    assert terminal.generation.tokens == (10,)
    assert terminal.traces[0].unrechecked_low_margin_bonus_rows == (1,)
    assert terminal.traces[0].numerical_recheck_extra_target_forwards == 0


def test_recheck_parameter_validation_and_default_identity():
    target = FakeTarget([1], [10, 11])
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="margin"):
            HFPackedTargetSession(
                target, [1], numerical_recheck_top2_margin=bad)
        with pytest.raises(ValueError, match="margin"):
            PackedSpeculativeDecoder(
                target, ScriptedDraft([[10]]), None,
                numerical_recheck_top2_margin=bad)
    raw = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10]]), None, window=1,
    ).generate([1], 1)
    assert raw.traces[0].decision_backend == "one_forward_packed"
    assert raw.traces[0].numerical_recheck_extra_target_forwards == 0


def test_short_recheck_rejects_hidden_that_is_not_final_norm():
    target = shape_sensitive_target()
    previous = target.model.forward

    def broken_final_hidden(**kwargs):
        result = previous(**kwargs)
        if (kwargs["output_hidden_states"]
                and tuple(kwargs["input_ids"][0].tolist()) == (1, 10)):
            result.hidden_states = (result.hidden_states[-1].clone() + 1,)
        return result

    target.model.forward = broken_final_hidden
    session = HFPackedTargetSession(
        target, [1], numerical_recheck_top2_margin=0.0)
    with pytest.raises(RuntimeError, match="not final norm"):
        session.verify([10, 11])
