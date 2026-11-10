"""CPU contracts for the new packed P1 corpus, CE schedule and causal rollout."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from feedback_codec.codec import CodecConfig
from feedback_codec.decoding import TargetLogitSummary
from feedback_codec.fixed_probe import ProbeExample, ProbeObservation
from feedback_codec.v2_p1_packed import (
    PACKED_P1_VERSION, audit_evaluation_row, canonical_sha256,
    evaluate_one_window, exposure_order,
    load_packed_p1_corpus, paired_prompt_ci, sha256_file, train_one_window,
)
from feedback_codec.v2_upper_bound import MatchedUpperBound, UPPER_MODES, assert_matched_capacity
from scripts.evaluate_packed_upper_p1 import _ablation_examples
from scripts.train_packed_upper_p1 import make_protocol


class DraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 10)

    def get_input_embeddings(self):
        return self.embedding


class FrozenDraft:
    def __init__(self):
        self.model = DraftModel().requires_grad_(False)
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def teacher_hidden(self, prefix: list[int], continuation: list[int]):
        self.calls.append((tuple(prefix), tuple(continuation)))
        ids = [prefix[-1]] + continuation[:-1]
        return self.model.embedding(torch.tensor(ids)).detach()


def _example(split: str = "train", prompt: str = "p1") -> ProbeExample:
    ids = (4,) + tuple(x for x in range(17) if x != 4)
    summary = TargetLogitSummary(
        greedy_token_id=4, top_token_ids=ids,
        top_logits_fp16=tuple(float(17 - x) for x in range(17)),
        logsumexp_fp32=18.0, top2_margin_fp32=1.0, vocab_size=17)
    observation = ProbeObservation(
        prompt_id=prompt, family_id=prompt, split=split, task_type="chat",
        round_index=0, source_prefix=(3,), proposals=(9, 10, 11, 12, 13, 14, 15, 16),
        valid_prefix=(3,), accepted_count=0, rejected_token=9,
        correction_token=4, corrected_prefix=(3, 4),
        target_hidden=torch.arange(12, dtype=torch.bfloat16),
        target_hidden_prefix=(3,), remaining_budget=8, window=8,
        target_logit_summary=summary)
    return ProbeExample(observation, (7, 8, 9, 10, 11, 12, 13, 14))


def _row(example: ProbeExample) -> dict:
    trace = {"accepted_count": 0, "verified": True}
    return {**example.to_json(),
            "collection_protocol_version": PACKED_P1_VERSION,
            "packed_feature_protocol_version": "packed_valid_feature_v2",
            "packed_feature_source": "original_packed_verify_forward",
            "online_feature_provenance_passed": True,
            "independent_ar_prefix_exact": True,
            "packed_trace": trace,
            "packed_trace_sha256": canonical_sha256(trace)}


def _protocol(path, *, split: str = "train", windows: int = 1) -> dict:
    return {"scope": "packed_fixed_prefix_collection",
            "eval_protocol_version": PACKED_P1_VERSION,
            "split": split, "window": 8, "max_future_tokens": 8,
            "eos_token_id": 16,
            "fixed_prefix_sha256": sha256_file(path),
            "target_file_sha256": {"model.safetensors": "a" * 64},
            "draft_file_sha256": {"model.safetensors": "b" * 64},
            "packed_p0_source_sha256": {"feedback_codec/packed.py": "c" * 64},
            "prompt_manifest_sha256": "d" * 64,
            "prompt_manifest_meta_sha256": "e" * 64,
            "packed_p0_report_sha256": "f" * 64,
            "collection_plan_sha256": "0" * 64,
            "audit": {"probe_windows": windows,
                      "independent_ar_prefix_exact_windows": windows,
                      "prompts": windows, "exact_requests": windows}}


def test_loader_requires_packed_provenance_and_independent_ar(tmp_path):
    corpus = tmp_path / "fixed_prefix.jsonl"
    protocol = tmp_path / "collection_protocol.json"
    row = _row(_example())
    corpus.write_text(json.dumps(row) + "\n", encoding="utf-8")
    protocol.write_text(json.dumps(_protocol(corpus)), encoding="utf-8")
    loaded = load_packed_p1_corpus(corpus, protocol, split="train")
    assert len(loaded.examples) == 1
    assert loaded.examples[0].observation.target_hidden.dtype is torch.bfloat16
    assert loaded.protocol_sha256 == sha256_file(protocol)
    wrong = dict(row, packed_feature_source="sequential_reference")
    corpus.write_text(json.dumps(wrong) + "\n", encoding="utf-8")
    p = _protocol(corpus)
    protocol.write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(ValueError, match="packed provenance"):
        load_packed_p1_corpus(corpus, protocol, split="train")
    wrong = dict(row, independent_ar_prefix_exact=False)
    corpus.write_text(json.dumps(wrong) + "\n", encoding="utf-8")
    protocol.write_text(json.dumps(_protocol(corpus)), encoding="utf-8")
    with pytest.raises(ValueError, match="packed provenance"):
        load_packed_p1_corpus(corpus, protocol, split="train")
    with pytest.raises(ValueError, match="test is sealed"):
        load_packed_p1_corpus(corpus, protocol, split="test")


def test_no_replacement_matched_schedule():
    order = exposure_order(20, seed=42, steps=20)
    assert len(order) == len(set(order)) == 20
    assert order == exposure_order(20, seed=42, steps=20)
    assert order != exposure_order(20, seed=43, steps=20)
    with pytest.raises(ValueError, match="no-replacement"):
        exposure_order(20, seed=42, steps=21)


def test_four_arm_ce_and_greedy_rollout_use_legal_observation_only():
    draft = FrozenDraft()
    frozen = draft.model.embedding.weight.detach().clone()
    cfg = CodecConfig(target_hidden_size=12, draft_hidden_size=10, vocab_size=17,
                      feature_dim=9, hidden_width=20, rank=4, code_dim=3)
    output_head = nn.Linear(10, 17, bias=False)
    arms = {}
    for mode in UPPER_MODES:
        torch.manual_seed(123)
        arms[mode] = MatchedUpperBound(cfg, output_head, mode=mode)
    assert_matched_capacity(arms)
    example = _example()
    for mode in UPPER_MODES:
        arm = arms[mode]
        optimizer = torch.optim.AdamW((p for p in arm.parameters() if p.requires_grad),
                                      lr=1e-4)
        loss = train_one_window(arm, draft, example, optimizer,
                                temperature=1.0, random_seed=42)
        assert loss > 0
        assert draft.calls[-2:] == [((3, 4), tuple(example.target_future_ids)),
                                    ((3, 4), (0,))]
        row = evaluate_one_window(arm, draft, example, eos_token_id=16)
        assert row["mode"] == mode and row["primary_eligible"]
        assert len(row["draft_ids"]) <= 8
        assert 0 <= row["accepted_count"] <= 8
        audit_evaluation_row(row, example, eos_token_id=16, expected_mode=mode)
        with pytest.raises(ValueError, match="accepted_count"):
            audit_evaluation_row(dict(row, accepted_count=9), example,
                                 eos_token_id=16, expected_mode=mode)
        with pytest.raises(ValueError, match="primary_eligible"):
            audit_evaluation_row(dict(row, primary_eligible=False), example,
                                 eos_token_id=16, expected_mode=mode)
        with pytest.raises(ValueError, match="truncation_match"):
            audit_evaluation_row(dict(row, truncation_match=not row["truncation_match"]),
                                 example, eos_token_id=16, expected_mode=mode)
    assert torch.equal(draft.model.embedding.weight, frozen)
    assert draft.model.embedding.weight.grad is None


def test_prompt_bootstrap_keeps_correlated_windows_together():
    left = [{"prompt_id": "p1", "primary_eligible": True, "accepted_count": 2},
            {"prompt_id": "p1", "primary_eligible": True, "accepted_count": 4},
            {"prompt_id": "p2", "primary_eligible": True, "accepted_count": 5}]
    right = [{"prompt_id": "p1", "primary_eligible": True, "accepted_count": 1},
             {"prompt_id": "p1", "primary_eligible": True, "accepted_count": 1},
             {"prompt_id": "p2", "primary_eligible": True, "accepted_count": 4}]
    result = paired_prompt_ci(left, right, repetitions=1000, seed=42)
    assert result["prompt_count"] == 2
    assert result["mean_delta"] == 1.5  # (prompt p1=2, prompt p2=1) / 2
    assert result["ci95"][0] <= result["mean_delta"] <= result["ci95"][1]


def test_frozen_hidden_zero_and_stratified_cross_request_donors():
    first = _example(prompt="p1")
    second = _example(prompt="p2")
    second = ProbeExample(replace(second.observation,
                                  target_hidden=second.observation.target_hidden + 5),
                          second.target_future_ids)
    zero, zero_audit = _ablation_examples((first, second), ablation="zero",
                                          seed=42, length_bin_size=128)
    assert zero_audit["windows"] == 2
    assert all(not example.observation.target_hidden.any() for example, _ in zero)
    shuffled, audit = _ablation_examples((first, second),
                                          ablation="shuffle_stratified",
                                          seed=42, length_bin_size=128)
    assert audit["coverage"] == 1.0
    assert [donor["prompt_id"] for _, donor in shuffled] == ["p2", "p1"]
    assert torch.equal(shuffled[0][0].observation.target_hidden,
                       second.observation.target_hidden)
    assert all(donor["stratum"] == ["chat", 0, 0] for _, donor in shuffled)
    isolated = ProbeExample(replace(first.observation, task_type="math"),
                            first.target_future_ids)
    with pytest.raises(ValueError, match="no cross-request donors"):
        _ablation_examples((isolated, second), ablation="shuffle_stratified",
                           seed=42, length_bin_size=128)


def test_initialize_only_protocol_binds_budget_and_cost_sha(tmp_path):
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({"hidden_size": 10,
                                                    "vocab_size": 17}), encoding="utf-8")
    corpus = SimpleNamespace(
        examples=(_example(),), corpus_sha256="1" * 64,
        protocol_sha256="2" * 64,
        protocol={"target_file_sha256": {"model.safetensors": "3" * 64},
                  "packed_p0_report_sha256": "4" * 64})
    budget_path = tmp_path / "budget.json"
    cost_path = tmp_path / "cost.json"
    freeze_path = tmp_path / "cost_freeze.json"
    args = SimpleNamespace(draft=draft, budget_lock=budget_path,
                           cost_lock=cost_path, cost_freeze_manifest=freeze_path,
                           steps=1, seeds=[42, 43, 44], learning_rate=1e-4,
                           weight_decay=.01, temperature=1.0,
                           checkpoint_every=256)
    budget_path.write_text(json.dumps({
        "schema": "P1_packed_budget_lock_v1", "frozen_before_validation": True,
        "train_corpus_sha256": corpus.corpus_sha256,
        "train_collection_protocol_sha256": corpus.protocol_sha256,
        "steps_per_arm": 1, "training_seeds": [42, 43, 44],
        "learning_rate": 1e-4, "weight_decay": .01,
        "temperature": 1.0, "checkpoint_every": 256}), encoding="utf-8")
    cost_path.write_text(json.dumps({"packed_p0_report_sha256": "4" * 64}),
                         encoding="utf-8")
    freeze_path.write_text(json.dumps({
        "schema": "feedback64_v2_p0_cost_proxy_freeze_v1",
        "cost_lock_sha256": sha256_file(cost_path),
        "packed_p0_report_sha256": "4" * 64,
        "pilot_val_acceptance_results_read_before_freeze": False}),
        encoding="utf-8")
    protocol = make_protocol(args, corpus, {"model.safetensors": "5" * 64})
    assert protocol["pre_val_budget_lock_sha256"] == sha256_file(budget_path)
    assert protocol["p0_cost_lock_sha256"] == sha256_file(cost_path)
    assert protocol["codec_config"]["target_hidden_size"] == 12
    freeze_path.write_text(freeze_path.read_text(encoding="utf-8").replace(
        '"pilot_val_acceptance_results_read_before_freeze": false',
        '"pilot_val_acceptance_results_read_before_freeze": true'), encoding="utf-8")
    with pytest.raises(ValueError, match="byte-frozen"):
        make_protocol(args, corpus, {"model.safetensors": "5" * 64})
