"""CPU checks for the explicitly versioned canonical BF16 ULP candidate.

The constructed row matches the real long-prompt failure pattern: packed and
same-online-KV short queries choose the larger token by one BF16 ULP, while
the independent full-prefix query ties and chooses the smaller token.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from feedback_codec.decoding import autoregressive_generate
from feedback_codec.packed import (
    HFPackedTargetSession, NUMERICAL_RECHECK_BACKEND,
    NUMERICAL_RECHECK_BACKEND_V2, STRICT_CANONICAL_BACKEND,
    PackedSpeculativeDecoder, _floating_ulp,
    _top2_max_ulp,
)
from test_packed import Encoder, FakeTarget, ScriptedDraft


def _wide_margin_shape_sensitive_target() -> FakeTarget:
    """The packed row wins by five BF16 ULPs, but the AR query ties."""
    target = FakeTarget([1], [10, 11, 13, 14])
    original = target.model.forward

    def shape_sensitive(**kwargs):
        result = original(**kwargs)
        query = tuple(int(token) for token in kwargs["input_ids"][0].tolist())
        if kwargs["output_hidden_states"]:
            if query == (1, 10, 11, 13) and kwargs["use_cache"]:
                result.logits[0, 2, :] = -100.0
                result.logits[0, 2, 14] = 30.875
                result.logits[0, 2, 13] = 30.25
            elif query == (1, 10, 11) and not kwargs["use_cache"]:
                result.logits[0, 2, :] = -100.0
                result.logits[0, 2, 13] = 29.875
                result.logits[0, 2, 14] = 29.875
        result.logits = result.logits.to(torch.bfloat16)
        return result

    target.model.forward = shape_sensitive
    return target


def test_strict_used_rows_fix_wide_margin_query_shape_mismatch():
    old = PackedSpeculativeDecoder(
        _wide_margin_shape_sensitive_target(), ScriptedDraft([[10, 11, 13], [14]]),
        None, window=3, numerical_recheck_top2_ulps=2,
    ).generate([1], 4)
    assert old.generation.tokens[:3] == (10, 11, 14)
    assert old.traces[0].original_packed_top2_margins[2] == 0.625
    assert old.traces[0].numerical_recheck_rows == ()

    repaired = PackedSpeculativeDecoder(
        _wide_margin_shape_sensitive_target(), ScriptedDraft([[10, 11, 13]]),
        None, window=3, strict_canonical_used_rows=True,
        capture_target_logits=True, feature_provenance_gate=True,
    ).generate([1], 4)
    assert repaired.generation.tokens == (10, 11, 13, 14)
    trace = repaired.traces[0]
    assert trace.decision_backend == STRICT_CANONICAL_BACKEND
    assert trace.numerical_recheck_rows == (0, 1, 2, 3)
    assert trace.numerical_recheck_trigger_reasons == ("strict",) * 4
    assert trace.numerical_recheck_extra_target_forwards == 8
    assert trace.row_source_modes == ("canonical_full_prefix",) * 4
    assert trace.cache_commit_source == "canonical_full_prefix_refill"
    assert repaired.generation.rounds[0].observation_logit_summary.greedy_token_id == 14
    assert trace.feature_provenance["passed"] is True
    assert trace.feature_provenance["decision_feature_available"] is True


def test_strict_policy_is_explicit_and_exclusive():
    target = FakeTarget([1], [10])
    with pytest.raises(ValueError, match="exclusive"):
        HFPackedTargetSession(target, [1], numerical_recheck_top2_ulps=2,
                              strict_canonical_used_rows=True)
    with pytest.raises(TypeError, match="bool"):
        PackedSpeculativeDecoder(target, ScriptedDraft([]), None,
                                 strict_canonical_used_rows=1)


def _one_ulp_wrong_target() -> FakeTarget:
    target = FakeTarget([1], [10, 11, 13])
    original = target.model.forward

    def shape_sensitive(**kwargs):
        result = original(**kwargs)
        if kwargs["output_hidden_states"]:
            query = tuple(int(token) for token in kwargs["input_ids"][0].tolist())
            if query == (1, 10, 12) and kwargs["use_cache"]:
                result.logits[0, 1, :] = -100.0
                result.logits[0, 1, 12] = 29.875
                result.logits[0, 1, 11] = 29.75
            elif query == (1, 10):
                result.logits[0, 1, :] = -100.0
                result.logits[0, 1, 12] = 29.875
                result.logits[0, 1, 11] = (
                    29.75 if kwargs["use_cache"] else 29.875)
        result.logits = result.logits.to(torch.bfloat16)
        return result

    target.model.forward = shape_sensitive
    return target


def test_two_bf16_ulps_trigger_full_prefix_and_charge_cache_refill():
    target = _one_ulp_wrong_target()
    assert _floating_ulp(29.875, torch.bfloat16) == 0.125
    assert autoregressive_generate(target, [1], 3) == (10, 11, 13)

    # The old tie-only policy is retained verbatim and does not catch the
    # observed nonzero 0.125 margin.
    old = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 12], [13]]), None, window=2,
        numerical_recheck_top2_margin=0.0,
    ).generate([1], 3)
    assert old.traces[0].decision_backend == NUMERICAL_RECHECK_BACKEND
    assert old.traces[0].numerical_recheck_rows == ()
    assert old.generation.tokens[1] == 12

    new = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 12], [13]]), None, window=2,
        capture_target_logits=True, feature_provenance_gate=True,
        numerical_recheck_top2_ulps=2,
    ).generate([1], 3)
    assert new.generation.tokens == (10, 11, 13)
    first = new.generation.rounds[0]
    assert first.accepted_count == 1
    assert first.rejected_token == 12
    assert first.correction_token == 11
    assert first.observation_logit_summary.greedy_token_id == 11
    trace = new.traces[0]
    assert trace.decision_backend == NUMERICAL_RECHECK_BACKEND_V2
    assert trace.original_packed_top2_margins[1] == 0.125
    assert trace.numerical_recheck_ulp_thresholds[1] == 0.25
    assert trace.numerical_recheck_rows == (1,)
    assert trace.numerical_recheck_trigger_reasons[1] == "margin"
    assert trace.original_packed_argmax_ids[1] == 12
    assert trace.original_packed_top2_logits_fp32[1] == (29.875, 29.75)
    assert trace.numerical_recheck_extra_target_forwards == 2
    assert trace.cache_commit_source == "canonical_full_prefix_refill"
    assert trace.cache_length_after == 2
    assert new.generation.rounds[1].prefix_before == (1, 10, 11)
    assert new.traces[1].cache_length_before == 2
    proof = trace.feature_provenance
    assert proof["source"] == "canonical_full_prefix_numeric_recheck"
    assert proof["valid_row_query_length"] == 2
    assert proof["reason"] == "canonical_full_prefix_ends_at_valid_row"
    assert proof["passed"]

    direct = HFPackedTargetSession(
        _one_ulp_wrong_target(), [1], numerical_recheck_top2_ulps=2)
    verification = direct.verify([10, 12])
    assert verification.row_source_modes[1] == "canonical_full_prefix"
    assert verification.row_source_query_lengths[1] == 2
    assert verification.decision_cache.tokens == (1, 10)
    assert direct.commit(verification, 1, 11) == 2
    assert direct._cache is verification.decision_cache


def test_after_first_ambiguous_acceptance_remaining_used_rows_are_canonical():
    target = FakeTarget([1], [10, 11, 12, 13])
    original = target.model.forward

    def first_ambiguous(**kwargs):
        result = original(**kwargs)
        if kwargs["output_hidden_states"]:
            query = tuple(int(token) for token in kwargs["input_ids"][0].tolist())
            if query == (1, 10, 11, 99) and kwargs["use_cache"]:
                result.logits[0, 1, :] = -100.0
                result.logits[0, 1, 11] = 29.875
                result.logits[0, 1, 12] = 29.75
            elif query == (1, 10) and not kwargs["use_cache"]:
                result.logits[0, 1, :] = -100.0
                result.logits[0, 1, 11] = 29.875
                result.logits[0, 1, 12] = 29.875
        result.logits = result.logits.to(torch.bfloat16)
        return result

    target.model.forward = first_ambiguous
    session = HFPackedTargetSession(target, [1], numerical_recheck_top2_ulps=2)
    verification = session.verify([10, 11, 99])
    assert verification.argmax_ids[:3] == (10, 11, 12)
    assert verification.numerical_recheck_rows == (1, 2)
    assert verification.row_source_modes[1:3] == (
        "canonical_full_prefix", "canonical_full_prefix")
    assert verification.row_source_query_lengths[1:3] == (2, 3)
    assert verification.numerical_recheck_extra_target_forwards == 4
    assert verification.numerical_recheck_trigger_reasons[1:3] == (
        "margin", "chain")
    assert session.commit(verification, 2, 12) == 3
    assert session._cache.tokens == (1, 10, 11)


@pytest.mark.parametrize("top1,top2,unit,trigger", [
    (29.875, 29.875, 0.125, True),  # zero ULP
    (29.875, 29.75, 0.125, True),   # one ULP
    (29.875, 29.625, 0.125, True),  # two ULPs, inclusive boundary
    (29.875, 29.5, 0.125, False),  # above the frozen threshold
    (-31.75, -32.25, 0.25, True),  # top2 crosses negative power boundary
    (32.0, 31.875, 0.25, True),   # positive power boundary
    (-31.875, -32.0, 0.25, True), # negative power boundary
])
def test_top2_ulp_max_and_exact_trigger_boundary(top1, top2, unit, trigger):
    assert _top2_max_ulp(top1, top2, torch.bfloat16) == unit
    assert ((top1 - top2) <= 2 * unit) is trigger


def test_bf16_zero_subnormal_and_nonfinite_ulp_contract():
    step = math.ldexp(1.0, -133)
    for value in (0.0, -0.0, step, -step, math.ldexp(1.0, -126)):
        assert _floating_ulp(value, torch.bfloat16) == step
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(FloatingPointError, match="nonfinite"):
            _top2_max_ulp(0.0, bad, torch.bfloat16)


def test_negative_cross_power_top2_triggers_canonical_path():
    target = FakeTarget([1], [10, 11, 13])
    original = target.model.forward

    def shape_sensitive(**kwargs):
        result = original(**kwargs)
        if kwargs["output_hidden_states"]:
            query = tuple(int(token) for token in kwargs["input_ids"][0].tolist())
            if query == (1, 10, 12) and kwargs["use_cache"]:
                result.logits[0, 1, :] = -100.0
                result.logits[0, 1, 12] = -31.75
                result.logits[0, 1, 11] = -32.25
            elif query == (1, 10) and not kwargs["use_cache"]:
                result.logits[0, 1, :] = -100.0
                result.logits[0, 1, 12] = -31.75
                result.logits[0, 1, 11] = -31.75
        result.logits = result.logits.to(torch.bfloat16)
        return result

    target.model.forward = shape_sensitive
    verification = HFPackedTargetSession(
        target, [1], numerical_recheck_top2_ulps=2).verify([10, 12])
    assert verification.original_packed_top2_margins[1] == 0.5
    assert verification.numerical_recheck_ulp_units[1] == 0.25
    assert verification.numerical_recheck_ulp_thresholds[1] == 0.5
    assert verification.original_packed_argmax_ids[1] == 12
    assert verification.argmax_ids[1] == 11
    assert verification.numerical_recheck_rows == (1,)
    assert verification.numerical_recheck_trigger_reasons[1] == "margin"


@pytest.mark.parametrize("role,prompt,continuation,draft,window,expected_used", [
    ("accepted", [3], [4, 0], [[4, 0]], 3, 2),
    ("correction", [3], [4, 0], [[4, 5, 6]], 3, 2),
    ("bonus", [1], [10, 11, 0], [[10, 11]], 2, 3),
])
def test_canonical_ulp_three_eos_roles_stop_without_feedback(
        role, prompt, continuation, draft, window, expected_used):
    target = FakeTarget(prompt, continuation)
    original = target.model.forward

    def ambiguous_used_rows(**kwargs):
        result = original(**kwargs)
        if kwargs["output_hidden_states"]:
            for row in range(result.logits.shape[1]):
                winner = int(result.logits[0, row].argmax().item())
                runner_up = 1 if winner == 0 else 0
                result.logits[0, row, :] = -100.0
                result.logits[0, row, winner] = 29.875
                result.logits[0, row, runner_up] = 29.75
        result.logits = result.logits.to(torch.bfloat16)
        return result

    target.model.forward = ambiguous_used_rows
    encoder = Encoder()
    packed = PackedSpeculativeDecoder(
        target, ScriptedDraft(draft), encoder, window=window,
        eos_token_id=0, capture_target_logits=True,
        numerical_recheck_top2_ulps=2,
    ).generate(prompt, 8)
    assert packed.generation.tokens == autoregressive_generate(
        target, prompt, 8, eos_token_id=0)
    assert packed.generation.tokens[-1] == 0
    assert len(packed.generation.rounds) == len(packed.traces) == 1
    record, trace = packed.generation.rounds[0], packed.traces[0]
    assert record.feedback_codes is None and encoder.calls == []
    assert trace.numerical_recheck_rows == tuple(range(expected_used))
    assert trace.numerical_recheck_trigger_reasons[:expected_used] == (
        "margin", *("chain",) * (expected_used - 1))
    assert trace.numerical_recheck_extra_target_forwards == 2 * expected_used
    assert trace.cache_length_after == len(prompt) + len(packed.generation.tokens) - 1
    if role == "accepted":
        assert record.accepted_count == 2
        assert record.correction_token is None and record.bonus_token is None
        assert record.observation_logit_summary is None
        assert trace.unrechecked_low_margin_bonus_rows == (2,)
    elif role == "correction":
        assert record.accepted_count == 1 and record.correction_token == 0
        assert record.bonus_token is None
        assert record.observation_logit_summary.greedy_token_id == 0
        assert trace.cache_commit_source == "canonical_full_prefix_refill"
    else:
        assert record.accepted_count == 2 and record.bonus_token == 0
        assert record.correction_token is None
        assert record.observation_logit_summary.greedy_token_id == 0
        assert trace.cache_commit_source == "canonical_full_prefix_refill"


def test_ulp_policy_rejects_invalid_and_preserves_old_default():
    target = FakeTarget([1], [10])
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="ULP count"):
            HFPackedTargetSession(target, [1], numerical_recheck_top2_ulps=bad)
        with pytest.raises(ValueError, match="ULP count"):
            PackedSpeculativeDecoder(target, ScriptedDraft([[10]]), None,
                                     numerical_recheck_top2_ulps=bad)
    with pytest.raises(ValueError, match="exclusive"):
        HFPackedTargetSession(target, [1], numerical_recheck_top2_margin=0.0,
                              numerical_recheck_top2_ulps=2)
    with pytest.raises(ValueError, match="exclusive"):
        PackedSpeculativeDecoder(target, ScriptedDraft([[10]]), None,
                                 numerical_recheck_top2_margin=0.0,
                                 numerical_recheck_top2_ulps=2)
    default = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10]]), None, window=1).generate([1], 1)
    assert default.traces[0].decision_backend == "one_forward_packed"
    assert default.traces[0].numerical_recheck_extra_target_forwards == 0
