"""CPU-only gate checks for the numeric verifier's versioned cost run."""

from __future__ import annotations

import pytest

from scripts.v2_numeric_cost_calibrate import (
    natural_numeric_guard_trace, validate_packed_gate_header,
)


def _numeric_gate() -> dict:
    return {
        "state": "passed",
        "scope": "real_qwen3_bf16_packed_numeric_recheck_committed_path_p0",
        "dtype": "bfloat16", "window": 8,
        "train_prompts_sha256": "train-sha",
        "token_gate_version": "packed_numeric_recheck_committed_path_v1",
        "numeric_feature_gate_version": "packed_valid_feature_numeric_recheck_v1",
        "numeric_backend": "packed_with_numerical_recheck_candidate_v1",
        "numerical_recheck_top2_margin": 0.0,
        "numeric_recheck_applied": True,
        "feature_protocol_version": "packed_valid_feature_v2",
        "packed_feature_provenance_gate_passed": True,
        "packed_feature_provenance_rounds": 72,
        "coverage": {"missing": []},
    }


def test_accepts_only_passed_versioned_numeric_p0():
    gate = _numeric_gate()
    validate_packed_gate_header(gate, "train-sha")
    for changed in (
        {"scope": "real_qwen3_bf16_packed_target_committed_path_p0"},
        {"token_gate_version": "packed_committed_path_v2"},
        {"numeric_recheck_applied": False},
        {"numerical_recheck_top2_margin": 0.5},
        {"packed_feature_provenance_gate_passed": False},
        {"coverage": {"missing": ["eos"]}},
    ):
        with pytest.raises(RuntimeError, match="numeric packed/KV P0 gate"):
            validate_packed_gate_header({**gate, **changed}, "train-sha")


def test_numeric_cost_gate_rejects_a_different_prompt_file():
    with pytest.raises(RuntimeError, match="numeric packed/KV P0 gate"):
        validate_packed_gate_header(_numeric_gate(), "different-sha")


def test_cost_trace_excludes_coverage_and_feature_replay_forwards():
    natural = [
        {"numerical_recheck_extra_target_forwards": count,
         "numerical_recheck_total_wall_ms": float(4 + count * 45)}
        for count in ([0] * 62 + [1] * 4 + [2] * 2)
    ]
    gate = {"checks": [{"round_checks": natural}],
            "numeric_recheck_extra_target_forwards": 8,
            "normal_packed_verify_forwards": 72,
            "feature_gate_diagnostic_extra_target_forwards": 120}
    result = natural_numeric_guard_trace(gate)
    assert result["natural_rounds"] == 68
    assert result["controlled_coverage_rounds_excluded"] == 4
    assert result["feature_gate_diagnostic_target_forwards_excluded"] == 120
    assert result["extra_target_forward_distribution"] == {"0": 62, "1": 4, "2": 2}
    assert result["extra_target_forwards_total"] == 8
    assert result["guard_total_wall_ms"]["p95_ms"] > 4
