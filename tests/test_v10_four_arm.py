"""Focused CPU contract tests for the v10 four-arm wiring heads."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from feedback_codec.codec import CodecConfig
from feedback_codec.decoding import TargetLogitSummary
from feedback_codec.fixed_probe import ProbeExample, ProbeObservation
from feedback_codec.v10_four_arm import (
    V10_MODES, assert_matched_v10_arms, build_matched_v10_arms,
    semantic_current_top32, top32_semantic_dim,
)


class TinyDraftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 10, dtype=torch.bfloat16)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class TinyDraft:
    def __init__(self) -> None:
        self.model = TinyDraftModel().requires_grad_(False).eval()
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def teacher_hidden(self, prefix: list[int], continuation: list[int]) -> torch.Tensor:
        self.calls.append((tuple(prefix), tuple(continuation)))
        assert continuation == [0]
        return self.model.embedding(torch.tensor([prefix[-1]])).detach()


def _config() -> CodecConfig:
    return CodecConfig(
        target_hidden_size=12, draft_hidden_size=10, vocab_size=17,
        feature_dim=9, hidden_width=20, rank=4, code_dim=3,
    )


def _summary(delta: float = 0.0) -> TargetLogitSummary:
    ids = (4,) + tuple(i for i in range(17) if i != 4)
    values = tuple(float(17 - i) + (delta if i == 0 else 0.0)
                   for i in range(17))
    lse = float(torch.logsumexp(torch.tensor(values), 0))
    return TargetLogitSummary(4, ids, values, lse, 1.0 + delta, 17)


def _observation() -> ProbeObservation:
    return ProbeObservation(
        prompt_id="train-1", family_id="family-1", split="train",
        task_type="chat", round_index=0, source_prefix=(3,),
        proposals=(9, 10), valid_prefix=(3,), accepted_count=0,
        rejected_token=9, correction_token=4, corrected_prefix=(3, 4),
        target_hidden=torch.arange(12, dtype=torch.bfloat16),
        target_hidden_prefix=(3,), remaining_budget=8, window=8,
        target_logit_summary=_summary(),
    )


def _arms(draft: TinyDraft):
    head = nn.Linear(10, 17, bias=False, dtype=torch.bfloat16)
    return build_matched_v10_arms(
        _config(), head, seed=42,
        draft_embedding=draft.model.get_input_embeddings(),
    )


def test_semantic_top32_layout_mask_and_truncation_accounting():
    draft = TinyDraft()
    table = draft.model.get_input_embeddings()
    summary = _summary()
    encoding = semantic_current_top32(summary, table)
    d = 10
    x = encoding.features[0]
    assert x.shape == (top32_semantic_dim(d),)
    selected = table.weight.detach().float()[list(summary.top_token_ids)]
    log_p = torch.tensor(summary.top_logits_fp16) - summary.logsumexp_fp32
    p = log_p.exp()
    assert torch.allclose(x[:d], selected[0])
    assert torch.allclose(x[d:2 * d], selected[1])
    assert torch.allclose(x[2 * d:3 * d], (p[:, None] * selected).sum(0) / p.sum())
    assert torch.allclose(x[3 * d:3 * d + 17], log_p)
    assert torch.count_nonzero(x[3 * d + 17:3 * d + 32]) == 0
    assert torch.all(x[3 * d + 32:3 * d + 32 + 17] == 1)
    assert torch.count_nonzero(x[3 * d + 32 + 17:3 * d + 64]) == 0
    assert x[-3] == summary.top2_margin_fp32
    assert torch.allclose(x[-2], p.sum())
    assert x[-1] == 17 / 32
    assert encoding.valid_k == 17
    assert encoding.saved_top32_mass_fp32 == pytest.approx(float(p.sum()))
    assert encoding.estimated_omitted_mass_fp32 >= 0
    assert encoding.mass_excess_due_saved_rounding_fp32 >= 0

    with pytest.raises(ValueError, match="ordered"):
        semantic_current_top32(
            replace(summary, top_logits_fp16=tuple(reversed(summary.top_logits_fp16))),
            table,
        )
    table.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        semantic_current_top32(summary, table)


def test_semantic_top32_records_actual_omitted_vocabulary_mass():
    table = nn.Embedding(40, 10).requires_grad_(False)
    logits = torch.linspace(1.0, 0.0, 40, dtype=torch.float32)
    summary = TargetLogitSummary(
        greedy_token_id=0, top_token_ids=tuple(range(32)),
        top_logits_fp16=tuple(float(v) for v in logits[:32]),
        logsumexp_fp32=float(torch.logsumexp(logits, 0)),
        top2_margin_fp32=float(logits[0] - logits[1]), vocab_size=40,
    )
    encoded = semantic_current_top32(summary, table)
    expected_mass = torch.exp(logits[:32] - torch.logsumexp(logits, 0)).sum()
    assert encoded.valid_k == 32
    assert encoded.saved_top32_mass_fp32 == pytest.approx(float(expected_mass))
    assert encoded.estimated_omitted_mass_fp32 == pytest.approx(
        1 - float(expected_mass))
    assert 0 < encoded.estimated_omitted_mass_fp32 < 1
    assert torch.count_nonzero(encoded.features[0, 3 * 10 + 32:3 * 10 + 64]) == 32


def test_four_arms_match_and_prepare_only_legal_current_boundary():
    draft = TinyDraft()
    arms = _arms(draft)
    assert tuple(arms) == V10_MODES
    evidence = assert_matched_v10_arms(arms)
    assert evidence["arms"] == 4 and evidence["edge_trainable_each"] > 0
    obs = _observation()
    states = {mode: model.prepare_observation(obs, draft)
              for mode, model in arms.items()}
    assert all(state.corrected_prefix == (3, 4) for state in states.values())
    assert states["B"].accepted_count is None
    assert states["C"].accepted_count == 0
    assert states["F-H"].target_hidden.shape == (1, 12)
    top32 = states["F-CURRENT-TOP32"].target_top32_features
    assert top32.shape == (1, 97) and top32.dtype == torch.float32
    assert states["B"].target_hidden is None
    assert states["C"].target_top32_features is None
    assert all(cont == (0,) for _, cont in draft.calls)

    changed = replace(obs, target_hidden=obs.target_hidden + 10,
                      target_logit_summary=_summary(delta=5))
    for mode in ("B", "C"):
        other = arms[mode].prepare_observation(changed, draft)
        assert torch.equal(states[mode].local_hidden, other.local_hidden)
        assert other.target_hidden is None and other.target_top32_features is None
    assert not torch.equal(states["F-H"].target_hidden,
                           arms["F-H"].prepare_observation(changed, draft).target_hidden)
    assert not torch.equal(top32,
                           arms["F-CURRENT-TOP32"].prepare_observation(
                               changed, draft).target_top32_features)
    with pytest.raises(TypeError, match="future labels"):
        arms["F-H"].prepare_observation(ProbeExample(obs, (7, 8)), draft)
    with pytest.raises(ValueError, match="causal"):
        arms["F-H"].prepare_observation(
            replace(obs, target_hidden_prefix=(99,)), draft)
    with pytest.raises(ValueError, match="correction"):
        arms["F-CURRENT-TOP32"].prepare_observation(
            replace(obs, target_logit_summary=replace(_summary(), greedy_token_id=5)),
            draft,
        )
    with torch.no_grad():
        next(p for p in arms["F-H"].parameters() if p.requires_grad).add_(1)
    with pytest.raises(ValueError, match="initialization"):
        assert_matched_v10_arms(arms)


def test_bf16_ce_all_active_paths_gradients_checkpoint_and_next_token():
    draft = TinyDraft()
    arms = _arms(draft)
    obs = _observation()
    frozen = draft.model.embedding.weight.detach().clone()
    hidden = torch.randn(1, 2, 10).to(torch.bfloat16)
    labels = torch.tensor([7, 8])
    before = {name: model.trainable_state_dict() for name, model in arms.items()}
    for mode, model in arms.items():
        state = model.prepare_observation(obs, draft)
        torch.manual_seed(9)
        logits = model.forward_logits(hidden, state, torch.tensor([1, 2]),
                                      temperature=1.0)
        assert logits.shape == (1, 2, 17) and logits.dtype == torch.bfloat16
        loss = F.cross_entropy(logits.float().reshape(-1, 17), labels)
        loss.backward()
        grads = {name: p.grad for name, p in model.named_parameters()
                 if p.requires_grad}
        assert grads and all(g is not None and torch.isfinite(g).all()
                             for g in grads.values())
        assert model.local_hidden_lift[0].weight.grad.abs().sum() > 0
        assert model.hidden_projection.weight.grad.abs().sum() > 0
        assert model.local_top32_surrogate.weight.grad.abs().sum() > 0
        assert model.top32_projection.weight.grad.abs().sum() > 0
        assert model.core.repair.output_head.weight.grad is None
        assert not hasattr(state, "target_future_ids")
        with pytest.raises(TypeError):
            model.forward_logits(hidden, state, 1, future_labels=labels)
        model.load_trainable_state_dict(before[mode])
        assert all(torch.equal(value, model.trainable_state_dict()[name])
                   for name, value in before[mode].items())
        token = model.next_token([3, 4], state, 1, draft)
        assert type(token) is int and 0 <= token < 17
        with pytest.raises(ValueError, match="corrected-prefix branch"):
            model.next_token([3, 5], state, 1, draft)
    assert torch.equal(draft.model.embedding.weight, frozen)
    assert draft.model.embedding.weight.grad is None
