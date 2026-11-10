"""P1-B constant-input control, with no changes to the frozen evaluator."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from feedback_codec.codec import CodecConfig, MatchedP1Codec
from feedback_codec.fixed_probe import ProbeExample, ProbeObservation
from scripts.no_hidden_control import (
    assert_no_target_hidden_information, constant_hidden_examples,
    fold_constant_hidden,
)
from scripts.evaluate_no_hidden_p1 import _reference_rows
from scripts.train_no_hidden_p1 import _verify_no_hidden_gpu_gate


def _codec() -> MatchedP1Codec:
    config = CodecConfig(12, 10, 17, feature_dim=9, hidden_width=20,
                         rank=4, code_dim=3)
    return MatchedP1Codec(config, nn.Linear(10, 17, bias=False), mode="target_8B")


def test_constant_hidden_replaces_only_target_feature_in_memory() -> None:
    observation = ProbeObservation(
        prompt_id="p", family_id="f", split="train", task_type="unknown",
        round_index=1, source_prefix=(1,), proposals=(2,), valid_prefix=(1,),
        accepted_count=0, rejected_token=2, correction_token=3,
        corrected_prefix=(1, 3), target_hidden=torch.arange(12, dtype=torch.bfloat16),
        target_hidden_prefix=(1,), remaining_budget=8, window=8,
    )
    original = ProbeExample(observation, (4, 5, 6))
    result = constant_hidden_examples([original])[0]
    assert result is not original
    assert result.target_future_ids == original.target_future_ids
    assert result.observation.corrected_prefix == original.observation.corrected_prefix
    assert result.observation.target_hidden_prefix == original.observation.target_hidden_prefix
    assert result.observation.target_hidden.dtype == original.observation.target_hidden.dtype
    assert torch.all(result.observation.target_hidden == 1)
    assert torch.equal(original.observation.target_hidden,
                       torch.arange(12, dtype=torch.bfloat16))


def test_fold_keeps_constant_response_but_removes_all_target_information() -> None:
    torch.manual_seed(7)
    codec = _codec()
    original_budget = codec.parameter_budget()
    ones = torch.ones(1, 12)
    before = codec.cloud_scores(ones).detach().clone()
    codec.cloud_scores(ones).sum().backward()
    assert codec.cloud_projection is not None
    assert codec.cloud_projection.weight.grad is not None
    assert codec.cloud_projection.weight.grad.abs().sum().item() > 0
    fold_constant_hidden(codec)
    after = codec.cloud_scores(ones)
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
    assert codec.parameter_budget() == original_budget
    first, second = torch.randn(1, 12), torch.randn(1, 12)
    assert_no_target_hidden_information(codec, first, second)
    assert codec.pack_codes(codec.cloud_encode(first)[0]).__len__() == 8
    with pytest.raises(ValueError, match="distinct"):
        assert_no_target_hidden_information(codec, first, first)


def test_unfolded_model_fails_no_hidden_gate() -> None:
    codec = _codec()
    with pytest.raises(RuntimeError, match="still reads"):
        assert_no_target_hidden_information(codec, torch.zeros(1, 12),
                                             torch.ones(1, 12))


def test_reference_rows_require_identical_val_tokens_and_coverage() -> None:
    observation = ProbeObservation(
        prompt_id="p", family_id="f", split="val", task_type="unknown",
        round_index=0, source_prefix=(1,), proposals=(2,), valid_prefix=(1,),
        accepted_count=0, rejected_token=2, correction_token=3,
        corrected_prefix=(1, 3), target_hidden=torch.ones(12),
        target_hidden_prefix=(1,), remaining_budget=8, window=8,
    )
    item = ProbeExample(observation, tuple(range(4, 12)))
    names = ("raw_draft", "edge_only", "protocol_only", "target_8B",
             "shuffled_global_8B", "shuffled_stratified_8B")
    rows = [{"arm": name, "prompt_id": "p", "round_index": 0,
             "target_future_ids": list(item.target_future_ids),
             "draft_ids": list(item.target_future_ids), "accepted_count": 8}
            for name in names]
    report = {"scope": "fixed_prefix_probe", "eval_protocol_version": "fixed_prefix_v1",
              "split": "val", "probe_windows": 1, "primary_windows": 1, "rows": rows}
    assert set(_reference_rows(report, [item], 99)) == set(names)
    rows[0]["target_future_ids"][0] = 99
    with pytest.raises(ValueError, match="continuation differs"):
        _reference_rows(report, [item], 99)


def test_gpu_gate_rejects_missing_gradient_groups(tmp_path, monkeypatch) -> None:
    import json
    import scripts.train_no_hidden_p1 as train_module

    monkeypatch.setattr(train_module, "_sha256", lambda _: "known")
    report = {
        "state": "passed", "scope": "P1-B_no_target_hidden_real_gpu_gate",
        "cuda": True, "dtype": "bfloat16", "target_file_sha256": {},
        "draft_file_sha256": {}, "train_corpus_sha256": "known",
        "reference_training_protocol_sha256": "known",
        "reference_d_checkpoint_sha256": "known", "train_smoke_steps": 2,
        "finite_losses": True, "cloud_hidden_invariance": True,
        "eight_byte_message": True, "target_frozen": True,
        "draft_frozen": True, "gradient_l1": {},
        "independent_target_ar_round_checks": [
            {"exact_target_ar": True}, {"exact_target_ar": True},
        ],
        "source_sha256": {
            name: "known" for name in (
                "scripts/no_hidden_control.py", "scripts/check_no_hidden_gpu_p0.py",
                "scripts/train_no_hidden_p1.py", "feedback_codec/codec.py",
                "feedback_codec/fixed_probe.py", "feedback_codec/training.py",
                "feedback_codec/hf_backend.py", "feedback_codec/decoding.py",
            )
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="GPU gate did not pass"):
        _verify_no_hidden_gpu_gate(report_path, {
            "target_file_sha256": {}, "draft_file_sha256": {},
        }, tmp_path / "corpus", tmp_path / "reference", "known")
