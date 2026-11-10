"""The versioned P0 runner gates used rows against an independent AR oracle."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from feedback_codec.decoding import autoregressive_generate
from feedback_codec.packed import PackedSpeculativeDecoder
from scripts.run_packed_p0_numeric_v2 import GateFailure, check_rounds
from test_packed import ScriptedDraft
from test_packed_numeric_recheck import shape_sensitive_target


def test_original_tied_commit_fails_numeric_gate_and_rechecked_commit_passes():
    target = shape_sensitive_target()
    oracle = autoregressive_generate(target, [1], 2, eos_token_id=0)
    assert oracle == (10, 12)
    raw = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 11]]), None, window=2,
        capture_target_logits=True, feature_provenance_gate=True,
    ).generate([1], 2)
    with pytest.raises(GateFailure, match="output/EOS/truncation"):
        check_rounds(target, raw, oracle, [1], 2, 0, "raw_draft", "cpu",
                     feature_provenance=True)

    corrected = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 11]]), None, window=2,
        capture_target_logits=True, feature_provenance_gate=True,
        numerical_recheck_top2_margin=0.0,
    ).generate([1], 2)
    rows = check_rounds(target, corrected, oracle, [1], 2, 0,
                        "raw_draft", "cpu", feature_provenance=True)
    assert len(rows) == 1
    assert rows[0]["used_positions"][-1]["argmax_equal"]
    assert rows[0]["cache_commit_source"] == "online_short_recheck"
    assert rows[0]["numerical_recheck_rows"] == [1]
    assert rows[0]["numerical_recheck_extra_target_forwards"] == 1
    assert rows[0]["feature_provenance"]["hidden_source"] == (
        "numeric_recheck_query_forward")
