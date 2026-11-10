"""CPU gates for P1 B/C/F causal views and capacity-matched repair."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from feedback_codec.codec import CodecConfig
from feedback_codec.decoding import TargetLogitSummary
from feedback_codec.fixed_probe import ProbeExample, ProbeObservation
from feedback_codec.v2_upper_bound import (
    MatchedUpperBound, UPPER_MODES, assert_matched_capacity,
)


class DraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 10)

    def get_input_embeddings(self):
        return self.embedding


class FrozenDraft:
    def __init__(self):
        self.model = DraftModel().requires_grad_(False)
        self.calls = []

    def teacher_hidden(self, prefix: list[int], continuation: list[int]):
        self.calls.append((tuple(prefix), tuple(continuation)))
        assert continuation == [0], "upper bound must not ask for Target future"
        return self.model.embedding(torch.tensor([prefix[-1]])).detach()


def config():
    return CodecConfig(
        target_hidden_size=12, draft_hidden_size=10, vocab_size=17,
        feature_dim=9, hidden_width=20, rank=4, code_dim=3,
    )


def summary(delta=0.0):
    ids = (4,) + tuple(x for x in range(17) if x != 4)
    values = tuple(float(17 - j) + (delta if j == 0 else 0.0)
                   for j in range(17))
    return TargetLogitSummary(
        greedy_token_id=4, top_token_ids=ids,
        top_logits_fp16=values, logsumexp_fp32=18.0,
        top2_margin_fp32=1.0 + delta, vocab_size=17,
    )


def observation():
    return ProbeObservation(
        prompt_id="train-1", family_id="family-1", split="train",
        task_type="chat", round_index=0, source_prefix=(3,),
        proposals=(9, 10), valid_prefix=(3,), accepted_count=0,
        rejected_token=9, correction_token=4, corrected_prefix=(3, 4),
        target_hidden=torch.arange(12, dtype=torch.bfloat16),
        target_hidden_prefix=(3,), remaining_budget=8, window=8,
        target_logit_summary=summary(),
    )


def models():
    result = {}
    for mode in UPPER_MODES:
        torch.manual_seed(441)
        result[mode] = MatchedUpperBound(
            config(), nn.Linear(10, 17, bias=False), mode=mode)
    return result


def test_four_arms_have_identical_active_parameter_names_count_and_initialization():
    arms = models()
    evidence = assert_matched_capacity(arms)
    assert evidence["arms"] == 4
    assert evidence["edge_trainable_each"] > 0
    for arm in arms.values():
        budget = arm.parameter_budget()
        assert budget["edge_trainable"] == evidence["edge_trainable_each"]
        assert budget["cloud_trainable"] == 0
        assert budget["wire_bytes"] is None
        assert sum(p.numel() for p in arm.parameters() if p.requires_grad) == evidence[
            "edge_trainable_each"]
    assert arms["full_hidden"].parameter_budget()["diagnostic_teacher_elements"] == 12
    assert arms["full_logit"].parameter_budget()["diagnostic_teacher_elements"] == 67
    with torch.no_grad():
        next(p for p in arms["full_hidden"].parameters() if p.requires_grad).add_(1)
    with pytest.raises(ValueError, match="initialization mismatch"):
        assert_matched_capacity(arms)
    assert_matched_capacity(arms, equal_initial_weights=False)


def test_exact_view_types_and_no_future_label_access():
    draft = FrozenDraft()
    obs = observation()
    arms = models()
    expected = {
        "edge_only": obs.edge_view(),
        "protocol_only": obs.protocol_view(),
        "full_hidden": obs.target_view(),
        "full_logit": obs.target_logit_view(),
    }
    for mode, model in arms.items():
        prepared = model.prepare_view(expected[mode], draft)
        assert prepared.mode == mode and prepared.corrected_prefix == (3, 4)
        assert draft.calls[-1] == ((3, 4), (0,))
        other = expected["protocol_only" if mode == "edge_only" else "edge_only"]
        with pytest.raises(TypeError, match="permitted view"):
            model.prepare_view(other, draft)
    item = ProbeExample(obs, (7, 8))
    with pytest.raises(TypeError, match="no ProbeExample/future labels"):
        arms["full_hidden"].prepare_observation(item, draft)
    assert not hasattr(arms["full_hidden"].prepare_observation(obs, draft),
                       "target_future_ids")
    assert all(continuation == (0,) for _, continuation in draft.calls)


def test_target_feature_provenance_and_type_isolation():
    draft = FrozenDraft()
    obs = observation()
    arms = models()
    hidden_state = arms["full_hidden"].prepare_observation(obs, draft)
    logit_state = arms["full_logit"].prepare_observation(obs, draft)
    assert torch.equal(hidden_state.target_hidden[0], obs.target_hidden)
    assert hidden_state.target_logit_features is None
    assert logit_state.target_hidden is None
    assert logit_state.target_logit_features.shape == (1, 67)
    with pytest.raises(ValueError, match="causal verification boundary"):
        arms["full_hidden"].prepare_observation(
            replace(obs, target_hidden_prefix=(99,)), draft)
    with pytest.raises(ValueError, match="recorded correction"):
        arms["full_logit"].prepare_observation(
            replace(obs, target_logit_summary=replace(summary(), greedy_token_id=5)), draft)
    with pytest.raises(ValueError, match="vocabulary"):
        arms["full_logit"].prepare_observation(
            replace(obs, target_logit_summary=replace(summary(), vocab_size=18,
                                                      top_token_ids=tuple(range(18)),
                                                      top_logits_fp16=tuple(float(x) for x in range(18)))), draft)

    changed_hidden = replace(obs, target_hidden=obs.target_hidden + 10)
    changed_logit = replace(obs, target_logit_summary=summary(delta=5))
    for mode in ("edge_only", "protocol_only"):
        base = arms[mode].prepare_observation(obs, draft)
        assert torch.equal(base.local_hidden,
                           arms[mode].prepare_observation(changed_hidden, draft).local_hidden)
        assert torch.equal(base.local_hidden,
                           arms[mode].prepare_observation(changed_logit, draft).local_hidden)
        assert base.target_hidden is None and base.target_logit_features is None
        with pytest.raises(TypeError, match="forbidden or missing Target feature"):
            arms[mode].forward_logits(torch.randn(1, 10),
                                      replace(base, target_hidden=torch.ones(1, 12)), 1)
    assert not torch.equal(hidden_state.target_hidden,
                           arms["full_hidden"].prepare_observation(changed_hidden, draft).target_hidden)
    assert not torch.equal(logit_state.target_logit_features,
                           arms["full_logit"].prepare_observation(changed_logit, draft).target_logit_features)


def test_ce_only_matched_exposure_loss_gradients_and_frozen_draft():
    draft = FrozenDraft()
    obs = observation()
    arms = models()
    assert_matched_capacity(arms)
    frozen = draft.model.embedding.weight.detach().clone()
    hidden = torch.randn(1, 2, 10)
    labels = torch.tensor([7, 8])  # caller-owned future; never passed to prepare
    traces = []
    for mode, model in arms.items():
        state = model.prepare_observation(obs, draft)
        torch.manual_seed(91)  # same stochastic local-code schedule in each arm
        logits = model.forward_logits(hidden, state, torch.tensor([1, 2]),
                                      temperature=1.0)
        assert logits.shape == (1, 2, 17)
        loss = F.cross_entropy(logits.reshape(-1, 17), labels)
        assert torch.isfinite(loss)
        loss.backward()
        grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads.values())
        assert model.core.repair.output_head.weight.grad is None
        assert model.local_hidden_lift[0].weight.grad.abs().sum() > 0
        assert model.hidden_projection.weight.grad.abs().sum() > 0
        assert model.local_logit_surrogate.weight.grad.abs().sum() > 0
        assert model.logit_projection.weight.grad.abs().sum() > 0
        traces.append((mode, len(labels), tuple(labels.tolist()), float(loss.detach())))
        with pytest.raises(TypeError):
            model.forward_logits(hidden, state, 1, future_labels=labels)
    assert all(length == 2 and targets == (7, 8) for _, length, targets, _ in traces)
    assert torch.equal(draft.model.embedding.weight, frozen)
    assert draft.model.embedding.weight.grad is None


def test_frozen_f_weights_respond_to_current_target_feature_ablation():
    draft = FrozenDraft()
    obs = observation()
    arms = models()
    hidden = torch.randn(1, 10)
    h_model = arms["full_hidden"]
    h_original = h_model.prepare_observation(obs, draft)
    h_zero = h_model.prepare_observation(
        replace(obs, target_hidden=torch.zeros_like(obs.target_hidden)), draft)
    assert not torch.equal(h_model.forward_logits(hidden, h_original, 1),
                           h_model.forward_logits(hidden, h_zero, 1))
    z_model = arms["full_logit"]
    z_original = z_model.prepare_observation(obs, draft)
    z_changed = z_model.prepare_observation(
        replace(obs, target_logit_summary=summary(delta=5)), draft)
    assert not torch.equal(z_model.forward_logits(hidden, z_original, 1),
                           z_model.forward_logits(hidden, z_changed, 1))


def test_inference_uses_only_prepared_corrected_branch_and_valid_positions():
    draft = FrozenDraft()
    model = models()["full_hidden"]
    state = model.prepare_observation(observation(), draft)
    token = model.next_token([3, 4], state, 1, draft)
    assert type(token) is int and 0 <= token < 17
    assert draft.calls[-1] == ((3, 4), (0,))
    with pytest.raises(ValueError, match="corrected-prefix branch"):
        model.next_token([3, 5], state, 1, draft)
    with pytest.raises(ValueError, match="position"):
        model.next_token([3, 4], state, 9, draft)
