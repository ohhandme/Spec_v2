"""CPU-only checks for v2 preregistered cost math and calibration hygiene."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.v2_cost_calibrate import (
    ROOT, PRELOCK_SHA256, frame_size_bytes, load_train_prompt, measure_cuda, quantile_stats,
    scenario_thresholds, sha256_file, validate_packed_gate_header,
    validate_prelock, verified_inference_rows, wire_sizes,
)


def _prelock() -> dict:
    return json.loads((ROOT / "cost_lock.prelock.json").read_text(encoding="utf-8"))


def test_prelock_is_five_scenarios_with_original_sensitivity_values():
    value = _prelock()
    validate_prelock(value)
    assert sha256_file(ROOT / "cost_lock.prelock.json") == PRELOCK_SHA256
    assert [row["name"] for row in value[
        "preregistered_scenarios_not_observed_network"]] == [
            "near_edge", "wan_high_rtt", "uplink_limited",
            "downlink_limited", "both_constrained"]
    assert value["synchronous_break_even"][
        "A_C_sensitivity_values_from_prior_only"] == [2.0, 3.1228, 4.5]
    value["preregistered_scenarios_not_observed_network"][0]["rtt_ms"] = 0
    with pytest.raises(ValueError, match="scenario changed"):
        validate_prelock(value)


def test_quantiles_keep_raw_count_and_interpolate():
    result = quantile_stats([4.0, 1.0, 3.0, 2.0])
    assert result["count"] == 4
    assert result["p10_ms"] == pytest.approx(1.3)
    assert result["p50_ms"] == pytest.approx(2.5)
    assert result["p95_ms"] == pytest.approx(3.85)
    with pytest.raises(ValueError):
        quantile_stats([float("nan")])


def test_wire_sizes_match_local_socket_reference():
    torch = pytest.importorskip("torch")
    from feedback_codec.cloud_edge import send_frame

    class Capture:
        def __init__(self) -> None:
            self.data = b""

        def sendall(self, payload: bytes) -> None:
            self.data += payload

    metadata = {"type": "result", "a": 0, "b": 99,
                "kind": "correction", "done": False, "feedback_bytes": 8}
    channel = Capture()
    send_frame(channel, metadata, bytes(range(8)))
    assert len(channel.data) == frame_size_bytes(metadata, bytes(range(8)))
    sizes = wire_sizes([1, 2], list(range(8)), 0, 99)
    assert sizes["raw_feedback_payload_bytes"] == 8
    assert sizes["incremental_feedback_wire_bytes"] == 8
    assert sizes["reply_with_feedback_bytes"] == sizes[
        "ordinary_reply_bytes"] + 8
    assert torch.__version__


def test_prelocked_break_even_uses_conservative_quantiles_and_never_calls_it_measured_network():
    stats = {
        "draft_incremental_token": {"p10_ms": 1.0, "p95_ms": 1.5},
        "packed_target_verify": {"p10_ms": 2.0, "p95_ms": 3.0},
        "cloud_encode_pack": {"p10_ms": 0.2, "p95_ms": 0.5},
        "edge_decode": {"p10_ms": 0.1, "p95_ms": 0.25},
        "rank32_repair": {"p10_ms": 0.05, "p95_ms": 0.1},
    }
    wire = {"upload_bytes": 100, "ordinary_reply_bytes": 20,
            "incremental_feedback_wire_bytes": 8}
    rows = scenario_thresholds(stats, wire, _prelock())
    assert len(rows) == 5
    near = rows[0]
    baseline = 8 + 2 + 5 + 100 / 6250 + 20 / 6250
    extra = 0.5 + 0.25 + 8 * 0.1 + 8 / 6250
    assert near["baseline_round_p10_ms"] == pytest.approx(baseline)
    assert near["extra_feedback_p95_ms"] == pytest.approx(extra)
    assert near["delta_A_min_by_A_C"][0]["delta_A_min_tokens_per_round"] == pytest.approx(
        3 * extra / baseline)
    assert near["assumption_based"] is True
    assert near["network_observed"] is False
    with pytest.raises(ValueError, match="feedback"):
        scenario_thresholds(stats, {**wire, "incremental_feedback_wire_bytes": 7},
                            _prelock())


def test_cost_calibration_rejects_any_val_or_test_prompt(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"id": "train-1", "split": "train", "input_ids": [1, 2]})
        + "\n"
        + json.dumps({"id": "val-1", "split": "val", "input_ids": [3, 4]})
        + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="val/test"):
        load_train_prompt(path, sha256_file(path), ["train-1"])
    path.write_text(
        json.dumps({"id": "train-1", "split": "train", "input_ids": [1, 2]})
        + "\n", encoding="utf-8")
    assert load_train_prompt(path, sha256_file(path), ["train-1"]) == (
        "train-1", [1, 2])
    with pytest.raises(ValueError, match="chosen prompt"):
        load_train_prompt(path, sha256_file(path), ["train-1"], "not-in-p0")


def test_synchronized_sampling_enforces_warmup_and_preserves_twenty_raw_samples():
    calls = {"setup": 0, "operation": 0, "synchronize": 0}

    def setup() -> int:
        calls["setup"] += 1
        return calls["setup"]

    def operation(value: int) -> int:
        calls["operation"] += 1
        return value + 1

    def sync() -> None:
        calls["synchronize"] += 1

    values = measure_cuda("cpu_fake", warmup=5, repeats=20,
                          synchronize=sync, setup=setup,
                          operation=operation)
    assert len(values) == 20
    assert calls == {"setup": 25, "operation": 25, "synchronize": 50}
    with pytest.raises(ValueError, match="at least"):
        measure_cuda("cpu_fake", warmup=0, repeats=20,
                     synchronize=sync, setup=setup, operation=operation)


def test_cost_gate_requires_full_committed_token_and_online_feature_provenance():
    gate = {
        "state": "passed", "scope": "real_qwen3_bf16_packed_target_committed_path_p0",
        "dtype": "bfloat16", "window": 8,
        "train_prompts_sha256": "train-only-sha",
        "token_gate_version": "packed_committed_path_v2",
        "feature_protocol_version": "packed_valid_feature_v2",
        "packed_feature_provenance_gate_passed": True,
        "packed_feature_provenance_rounds": 72,
        "packed_full_hidden_feature_transfer_validated": False,
        "packed_full_logit_feature_transfer_validated": False,
        "coverage": {"missing": []},
    }
    validate_packed_gate_header(gate, "train-only-sha")
    for change in (
        {"packed_feature_provenance_gate_passed": False},
        {"packed_feature_provenance_rounds": 0},
        {"scope": "real_qwen3_bf16_packed_target_kv_exactness_p0"},
        {"coverage": {"missing": ["eos"]}},
    ):
        with pytest.raises(RuntimeError, match="not matched and passed"):
            validate_packed_gate_header({**gate, **change}, "train-only-sha")


def test_cost_target_asset_list_permits_only_documentation_omissions():
    expected = json.loads((ROOT / "qwen3-8b-expected.json").read_text(encoding="utf-8"))
    inference = [row for row in expected["files"]
                 if row["path"] not in {"LICENSE", "README.md"}]
    download = {
        "verified_files": [{"file": row["path"], "sha256": row["sha256"]}
                           for row in inference],
        "omitted_non_inference_files": ["LICENSE", "README.md"],
    }
    assert verified_inference_rows(expected, download) == inference
    with pytest.raises(RuntimeError, match="exactly the inference assets"):
        verified_inference_rows(expected, {
            **download, "verified_files": download["verified_files"][:-1]})
    with pytest.raises(RuntimeError, match="unexpected asset"):
        verified_inference_rows(expected, {
            **download, "omitted_non_inference_files": ["LICENSE", "README.md", "config.json"]})
