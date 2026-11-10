"""The canonical 2-ULP P0 runner gates used rows against independent AR."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from feedback_codec.decoding import autoregressive_generate
from feedback_codec.packed import PackedSpeculativeDecoder
from scripts.run_packed_p0_numeric_v3 import (
    GateFailure, check_rounds, controlled_coverage_probes, coverage_summary,
)
from test_packed import FakeTarget, ScriptedDraft
from test_packed_numeric_recheck import shape_sensitive_target


def test_original_tied_commit_fails_and_canonical_rechecked_commit_passes():
    target = shape_sensitive_target()
    oracle = autoregressive_generate(target, [1], 2, eos_token_id=0)
    assert oracle == (10, 12)
    raw = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 11]]), None, window=2,
        capture_target_logits=True, feature_provenance_gate=True,
    ).generate([1], 2)
    with pytest.raises(GateFailure, match="output/EOS/truncation") as failure:
        check_rounds(target, raw, oracle, [1], 2, 0, "raw_draft", "cpu",
                     feature_provenance=True)
    receipt = failure.value.evidence
    assert receipt["first_difference_output_index"] == 1
    assert receipt["first_difference_round"] == {
        "round_index": 0, "row_index": 1, "local_committed_index": 1}
    decision = receipt["rounds_before_raise"][0]
    assert decision["original_top2_ids"]
    assert decision["effective_top2_ids"]
    assert decision["original_top2_logits_fp32"]
    assert decision["effective_top2_logits_fp32"]
    assert decision["cache_commit_source"]
    assert decision["row_source_modes"]

    corrected = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 11]]), None, window=2,
        capture_target_logits=True, feature_provenance_gate=True,
        numerical_recheck_top2_ulps=2,
    ).generate([1], 2)
    rows = check_rounds(target, corrected, oracle, [1], 2, 0,
                        "raw_draft", "cpu", feature_provenance=True)
    assert len(rows) == 1
    assert rows[0]["used_positions"][-1]["argmax_equal"]
    assert rows[0]["cache_commit_source"] == "canonical_full_prefix_refill"
    assert rows[0]["numerical_recheck_rows"] == [1]
    assert rows[0]["numerical_recheck_extra_target_forwards"] == 2
    assert rows[0]["numerical_recheck_snapshot_wall_ms"] >= 0
    assert rows[0]["feature_provenance"]["hidden_source"] == (
        "canonical_full_prefix_numeric_recheck")


def test_frozen_natural_eos_prefix_covers_accept_correction_and_bonus():
    target = FakeTarget([1], [10, 11, 0])
    prompt = {"id": "train-1", "input_ids": [1]}
    oracle = {"train-1": autoregressive_generate(target, [1], 3, 0)}
    lock = {"id": "train-1", "split": "train", "prefix_ids": [1, 10, 11],
            "eos_token_id": 0}
    rows = controlled_coverage_probes(target, [prompt], oracle, 0, lock)
    by_arm = {row["arm"]: row for row in rows}
    assert by_arm["controlled_accepted_eos"]["output_ids"] == [0]
    assert by_arm["controlled_correction_eos"]["output_ids"] == [0]
    assert by_arm["controlled_bonus_eos"]["output_ids"] == [11, 0]
    assert all(by_arm[arm]["eos"] for arm in (
        "controlled_accepted_eos", "controlled_correction_eos",
        "controlled_bonus_eos"))
    coverage = coverage_summary([], rows)
    assert coverage["missing_eos_roles"] == []
    assert all(coverage["eos_roles"][role] for role in (
        "accepted", "correction", "bonus"))


def test_frozen_eos_prefix_rejects_unrelated_prompt():
    target = FakeTarget([1], [10, 11, 0])
    prompt = {"id": "train-1", "input_ids": [1]}
    oracle = {"train-1": autoregressive_generate(target, [1], 3, 0)}
    bad_lock = {"id": "train-1", "split": "train", "prefix_ids": [2, 10, 11],
                "eos_token_id": 0}
    with pytest.raises(GateFailure, match="frozen train EOS prefix"):
        controlled_coverage_probes(target, [prompt], oracle, 0, bad_lock)
