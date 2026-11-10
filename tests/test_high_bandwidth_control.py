"""Mechanism and provenance checks for the isolated P1-B F upper bound."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from feedback_codec.codec import CodecConfig, MatchedP1Codec
from feedback_codec.fixed_probe import TargetView, _generate_probe
from scripts.high_bandwidth_control import (
    HighBandwidthP1Codec, MESSAGE_BYTES, high_bandwidth_arm,
    pack_distribution, unpack_distribution,
)
from scripts.train_high_bandwidth_p1 import _verify_f_gate
from scripts.evaluate_high_bandwidth_p1 import _verify_f_source_chain


def _codec() -> HighBandwidthP1Codec:
    return HighBandwidthP1Codec(
        CodecConfig(12, 10, 17, feature_dim=9, hidden_width=20,
                    rank=4, code_dim=3),
        nn.Linear(10, 17, bias=False),
    )


def test_wire_roundtrip_is_exact_fp32_and_never_eight_bytes() -> None:
    torch.manual_seed(1)
    codec = _codec()
    distribution = codec.cloud_distribution(torch.randn(1, 12))
    wire = pack_distribution(distribution)
    assert len(wire) == MESSAGE_BYTES == 8192
    assert torch.equal(distribution, unpack_distribution(wire, device="cpu"))
    assert codec.parameter_budget()["message_bytes"] == 8192
    with pytest.raises(ValueError, match="8192"):
        unpack_distribution(wire[:8], device="cpu")
    with pytest.raises(ValueError, match="finite probabilities"):
        pack_distribution(torch.full((1, 8, 256), float("nan")))
    with pytest.raises(RuntimeError, match="not D byte codes"):
        codec.cloud_encode(torch.randn(1, 12))


def test_continuous_path_preserves_original_parameters_and_gradients() -> None:
    config = CodecConfig(12, 10, 17, feature_dim=9, hidden_width=20,
                         rank=4, code_dim=3)
    torch.manual_seed(9)
    original = MatchedP1Codec(config, nn.Linear(10, 17, bias=False),
                              mode="target_8B")
    torch.manual_seed(9)
    upper = HighBandwidthP1Codec(config, nn.Linear(10, 17, bias=False))
    assert [name for name, p in original.named_parameters() if p.requires_grad] == [
        name for name, p in upper.named_parameters() if p.requires_grad]
    assert all(torch.equal(a, b) for a, b in zip(
        original.trainable_state_dict().values(),
        upper.trainable_state_dict().values()))
    first, second = torch.randn(1, 12), torch.randn(1, 12)
    cloud = upper.cloud_sample_st(first)
    assert not torch.allclose(cloud, upper.cloud_distribution(second))
    local = F.one_hot(torch.randint(0, 256, (1, 2, 8)), 256).float()
    logits = upper.repair_logits(torch.randn(1, 10), local, 1, cloud)
    logits.square().mean().backward()
    assert upper.cloud_projection.weight.grad is not None
    assert upper.cloud_projection.weight.grad.abs().sum().item() > 0
    assert upper.cloud_encoder[0].weight.grad is not None
    assert upper.cloud_encoder[0].weight.grad.abs().sum().item() > 0
    assert all(p.grad is None for p in upper.repair.output_head.parameters())


def test_one_hot_continuous_cloud_reduces_to_frozen_hard_lookup() -> None:
    torch.manual_seed(2)
    codec = _codec()
    local = torch.randint(0, 256, (1, 2, 8))
    cloud = torch.randint(0, 256, (1, 8))
    hidden = torch.randn(1, 10)
    hard = codec.repair_logits(hidden, local, 1, cloud)
    soft = codec.repair_logits(
        hidden, F.one_hot(local, 256).float(), 1,
        F.one_hot(cloud, 256).float())
    assert torch.allclose(hard, soft, atol=1e-6, rtol=1e-6)


class _TinyDraft:
    def __init__(self) -> None:
        self.model = nn.Module()
        self.model.set_input_embeddings = lambda _: None
        self.model.get_input_embeddings = lambda: self.embedding
        self.embedding = nn.Embedding(17, 10)
        self.seen_prefixes: list[tuple[int, ...]] = []

    def teacher_hidden(self, prefix: list[int], _sentinel: list[int]) -> torch.Tensor:
        self.seen_prefixes.append(tuple(prefix))
        return self.embedding(torch.tensor(prefix)).mean(0).reshape(1, -1)


def test_arm_serializes_only_target_view_then_autoregresses() -> None:
    torch.manual_seed(4)
    codec = _codec()
    draft = _TinyDraft()
    arm = high_bandwidth_arm(draft, codec)
    view = TargetView(corrected_prefix=(1, 2), window=8,
                      accepted_count=0, rejected_token=3,
                      correction_token=2, target_hidden=torch.randn(12))
    prepared = arm.prepare(view)
    assert len(prepared.message) == MESSAGE_BYTES
    result = arm.next([1, 2], prepared, 1)
    assert isinstance(result, int) and 0 <= result < 17
    assert draft.seen_prefixes == [(1, 2), (1, 2)]
    with pytest.raises(TypeError, match="Target view"):
        arm.prepare(object())


def test_f_gate_rejects_malformed_wire_budget(tmp_path, monkeypatch) -> None:
    import scripts.train_high_bandwidth_p1 as train_module

    monkeypatch.setattr(train_module, "_sha256", lambda _: "known")
    gradients = {name: 1.0 for name in (
        "local_head_0", "local_head_1", "codebooks", "repair_U",
        "cloud_projection", "cloud_encoder")}
    sources = {name: "known" for name in (
        "scripts/high_bandwidth_control.py",
        "scripts/check_high_bandwidth_gpu_p0.py",
        "scripts/train_high_bandwidth_p1.py",
        "scripts/train_no_hidden_p1.py",
        "feedback_codec/codec.py", "feedback_codec/fixed_probe.py",
        "feedback_codec/training.py", "feedback_codec/hf_backend.py",
        "feedback_codec/decoding.py")}
    gate = {
        "state": "passed", "scope": "P1-B_F_continuous_real_gpu_gate",
        "cuda": True, "dtype": "bfloat16", "target_file_sha256": {},
        "draft_file_sha256": {}, "train_corpus_sha256": "known",
        "reference_training_protocol_sha256": "known",
        "reference_d_checkpoint_sha256": "known",
        "message_bytes": 8, "train_smoke_steps": 2,
        "finite_losses": True, "target_frozen": True, "draft_frozen": True,
        "gradient_l1": gradients,
        "independent_target_ar_round_checks": [
            {"exact_target_ar": True}, {"exact_target_ar": True}],
        "source_sha256": sources,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="F real GPU gate"):
        _verify_f_gate(path, {"target_file_sha256": {},
                              "draft_file_sha256": {}},
                       tmp_path / "corpus", tmp_path / "reference", "known")


@pytest.mark.parametrize("field,value", [
    ("wire_roundtrip_exact", False),
    ("wire_roundtrip_exact", None),
    ("target_dependence", False),
    ("target_dependence", None),
])
def test_f_gate_requires_wire_and_target_evidence(
        tmp_path, monkeypatch, field: str, value: bool | None) -> None:
    import scripts.train_high_bandwidth_p1 as train_module

    monkeypatch.setattr(train_module, "_sha256", lambda _: "known")
    gradients = {name: 1.0 for name in (
        "local_head_0", "local_head_1", "codebooks", "repair_U",
        "cloud_projection", "cloud_encoder")}
    sources = {name: "known" for name in (
        "scripts/high_bandwidth_control.py",
        "scripts/check_high_bandwidth_gpu_p0.py",
        "scripts/train_high_bandwidth_p1.py",
        "scripts/train_no_hidden_p1.py",
        "feedback_codec/codec.py", "feedback_codec/fixed_probe.py",
        "feedback_codec/training.py", "feedback_codec/hf_backend.py",
        "feedback_codec/decoding.py")}
    gate = {
        "state": "passed", "scope": "P1-B_F_continuous_real_gpu_gate",
        "cuda": True, "dtype": "bfloat16", "target_file_sha256": {},
        "draft_file_sha256": {}, "train_corpus_sha256": "known",
        "reference_training_protocol_sha256": "known",
        "reference_d_checkpoint_sha256": "known",
        "message_bytes": MESSAGE_BYTES, "train_smoke_steps": 2,
        "wire_roundtrip_exact": True, "target_dependence": True,
        "finite_losses": True, "target_frozen": True, "draft_frozen": True,
        "gradient_l1": gradients,
        "independent_target_ar_round_checks": [
            {"exact_target_ar": True}, {"exact_target_ar": True}],
        "source_sha256": sources,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    _verify_f_gate(path, {"target_file_sha256": {},
                          "draft_file_sha256": {}},
                   tmp_path / "corpus", tmp_path / "reference", "known")
    if value is None:
        del gate[field]
    else:
        gate[field] = value
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="F real GPU gate"):
        _verify_f_gate(path, {"target_file_sha256": {},
                              "draft_file_sha256": {}},
                       tmp_path / "corpus", tmp_path / "reference", "known")


def test_f_evaluation_requires_gate_script_provenance(monkeypatch) -> None:
    import scripts.evaluate_high_bandwidth_p1 as eval_module

    monkeypatch.setattr(eval_module, "_sha256", lambda _: "known")
    names = (
        "scripts/high_bandwidth_control.py",
        "scripts/check_high_bandwidth_gpu_p0.py",
        "scripts/train_high_bandwidth_p1.py",
        "scripts/train_no_hidden_p1.py",
        "feedback_codec/codec.py", "feedback_codec/training.py",
        "feedback_codec/fixed_probe.py",
    )
    training = {"source_sha256": {name: "known" for name in names}}
    gate = {"source_sha256": {
        "scripts/check_high_bandwidth_gpu_p0.py": "known"}}
    _verify_f_source_chain(training, gate)
    gate["source_sha256"]["scripts/check_high_bandwidth_gpu_p0.py"] = "wrong"
    with pytest.raises(ValueError, match="GPU gate script differs"):
        _verify_f_source_chain(training, gate)
    gate["source_sha256"]["scripts/check_high_bandwidth_gpu_p0.py"] = "known"
    training["source_sha256"]["scripts/check_high_bandwidth_gpu_p0.py"] = "wrong"
    with pytest.raises(ValueError, match="source differs from training"):
        _verify_f_source_chain(training, gate)
