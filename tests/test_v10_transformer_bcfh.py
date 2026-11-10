"""CPU contracts for the isolated B/C/F-H Transformer repair candidate."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from feedback_codec.codec import CodecConfig
from feedback_codec.fixed_probe import ProbeObservation
from feedback_codec.decoding import TargetStep
from feedback_codec.v11_full_contract import PilotWindow
from feedback_codec.v10_transformer_bcfh import (
    CausalTransformerResidual, TransformerBCFHHead,
    TransformerRepairConfig, build_matched_transformer_arms,
)
from scripts.transformer_bcfh_experiment import _train_step
from scripts.load_transformer_v13_windows import OriginalRawRequest
from scripts.transformer_bcfh_closed_loop import generate_closed_loop


def _config() -> CodecConfig:
    return CodecConfig(target_hidden_size=12, draft_hidden_size=10,
                       vocab_size=17, feature_dim=9, hidden_width=20,
                       rank=4, code_dim=3)


def _observation() -> ProbeObservation:
    return ProbeObservation(
        prompt_id="p1", family_id="f1", split="train", task_type="chat",
        round_index=0, source_prefix=(3,), proposals=(9, 10),
        valid_prefix=(3,), accepted_count=0, rejected_token=9,
        correction_token=4, corrected_prefix=(3, 4),
        target_hidden=torch.arange(12, dtype=torch.float32),
        target_hidden_prefix=(3,), remaining_budget=8, window=8)


class _Draft:
    def __init__(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(17, 10)

            def get_input_embeddings(self) -> nn.Embedding:
                return self.embedding

        self.model = Model().requires_grad_(False)
        self.embedding = self.model.embedding
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def teacher_hidden(self, prefix: list[int], continuation: list[int]) -> torch.Tensor:
        self.calls.append((tuple(prefix), tuple(continuation)))
        ids = [prefix[-1]] + continuation[:-1]
        return self.embedding(torch.tensor(ids)).detach()


def _arms():
    draft = _Draft()
    output = nn.Linear(10, 17, bias=False).requires_grad_(False)
    choice = TransformerRepairConfig(depth=2, width=64)
    arms = build_matched_transformer_arms(
        _config(), output, seed=42, choice=choice,
        draft_embedding=draft.embedding)
    return arms, draft, output


def test_matched_initialization_raw_identity_and_forbidden_inputs():
    arms, draft, output = _arms()
    observation = _observation()
    history = draft.teacher_hidden([3, 4], [2, 5, 0]).unsqueeze(0)
    expected = output(history)
    budgets = {arm.parameter_budget()["edge_trainable"] for arm in arms.values()}
    assert len(budgets) == 1
    for mode, arm in arms.items():
        state = arm.prepare_observation(observation, draft)
        assert state.mode == mode
        assert torch.equal(arm.core.repair.output_head(history), expected)
        assert torch.equal(arm.core.repair.transformer(history,
                           torch.zeros(1, 24)), history)
        if mode != "F-H":
            assert state.target_hidden is None
    with pytest.raises(ValueError):
        TransformerBCFHHead(_config(), output, mode="F-CURRENT-TOP32")
    with pytest.raises(ValueError):
        arms["C"].core.repair.repair_from_control(
            history, torch.zeros(1, 24), torch.tensor([1, 3, 2]))


def test_future_hidden_cannot_change_past_logits_and_incremental_path():
    arms, draft, _ = _arms()
    arm = arms["F-H"]
    state = arm.prepare_observation(_observation(), draft)
    history = draft.teacher_hidden([3, 4], [2, 5, 6, 0]).unsqueeze(0)
    control = torch.randn(1, 24)
    adapter: CausalTransformerResidual = arm.core.repair.transformer
    with torch.no_grad():
        adapter.up.weight.normal_(std=0.02)
        full = adapter(history, control)
        changed = history.clone()
        changed[:, -1] += 20
        assert torch.allclose(full[:, :-1], adapter(changed, control)[:, :-1],
                              atol=1e-5, rtol=1e-5)
        for length in range(1, 5):
            prefix_logits = arm.forward_logits(
                history[:, :length], state, torch.arange(1, length + 1))
            all_logits = arm.forward_logits(history, state, torch.arange(1, 5))
            assert torch.allclose(prefix_logits, all_logits[:, :length],
                                  atol=1e-5, rtol=1e-5)
        token = arm.next_token([3, 4, 2, 5], state, 3, draft)
        direct = arm.forward_logits(history[:, :3], state, torch.arange(1, 4))
        assert token == int(direct[0, -1].argmax())
    assert draft.calls[-1] == ((3, 4), (2, 5, 0))
    with pytest.raises(ValueError):
        arm.next_token([3, 4, 2], state, 3, draft)


def test_transformer_receives_gradient_after_zero_initialized_residual_update():
    arms, draft, _ = _arms()
    arm = arms["C"]
    state = arm.prepare_observation(_observation(), draft)
    history = draft.teacher_hidden([3, 4], [2, 5, 0]).unsqueeze(0)
    optimizer = torch.optim.AdamW((p for p in arm.parameters() if p.requires_grad),
                                  lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = arm.forward_logits(history, state, torch.arange(1, 4),
                                    temperature=1.0)
        loss = F.cross_entropy(logits.reshape(3, -1), torch.tensor([2, 5, 6]))
        loss.backward()
        assert arm.core.repair.transformer.up.weight.grad is not None
        if step:
            assert arm.core.repair.transformer.blocks[0].attn.in_proj_weight.grad is not None
        optimizer.step()
    assert not torch.equal(logits.detach(), arm.core.repair.output_head(history))
    saved = arm.trainable_state_dict()
    with pytest.raises(ValueError):
        arm.load_trainable_state_dict({k: v for k, v in saved.items() if k != "core.repair.transformer.up.weight"})


def test_fh_hidden_perturbation_uses_same_weights_and_changes_output():
    arms, draft, _ = _arms()
    arm = arms["F-H"]
    observation = _observation()
    state = arm.prepare_observation(observation, draft)
    altered = arm.prepare_observation(
        replace(observation, target_hidden=observation.target_hidden + 30), draft)
    history = draft.teacher_hidden([3, 4], [2, 5, 0]).unsqueeze(0)
    with torch.no_grad():
        arm.core.repair.transformer.up.weight.normal_(std=0.02)
        first = arm.forward_logits(history, state, torch.arange(1, 4))
        second = arm.forward_logits(history, altered, torch.arange(1, 4))
    assert not torch.equal(first, second)


def test_weighted_training_step_updates_causal_transformer_on_cpu_fixture():
    arms, draft, _ = _arms()
    arm = arms["C"]
    optimizer = torch.optim.AdamW((p for p in arm.parameters() if p.requires_grad),
                                  lr=1e-3)
    window = PilotWindow(_observation(), (2, 5, 6), "a" * 64, False)
    before = arm.core.repair.transformer.up.weight.detach().clone()
    loss = _train_step(arm, draft, window, optimizer, step_seed=42)
    assert loss > 0
    assert not torch.equal(before, arm.core.repair.transformer.up.weight)


def test_full_request_compares_actual_original_sd_and_delayed_repair():
    oracle = (1, 2, 3, 0)

    class Target:
        def next(self, prefix):
            return TargetStep(oracle[len(prefix) - 1], torch.ones(4))

    class RawDraft:
        hidden_size = 4

        def __init__(self):
            self.model = self

        def get_output_embeddings(self):
            return nn.Identity()

        def teacher_hidden(self, prefix, continuation):
            return F.one_hot(torch.tensor([1] * len(continuation)), 4).float()

    class BetterHead:
        def __init__(self):
            self.boundaries = []

        def prepare_observation(self, observation, draft):
            self.boundaries.append(observation)
            return observation

        def next_token(self, prefix, state, position, draft):
            assert tuple(prefix[:len(state.corrected_prefix)]) == state.corrected_prefix
            return oracle[len(prefix) - 1]

    request = OriginalRawRequest(
        "p", "f", "chat", (4,), oracle,
        ((1,) * 8, (1,) * 8, (1,) * 8), (1, 0, 0), "a" * 64)
    raw = generate_closed_loop(request, Target(), RawDraft(), None, eos=0)
    assert raw.tokens == oracle
    assert raw.proposals == request.round_proposals
    assert raw.accepted == request.round_accepted
    head = BetterHead()
    repaired = generate_closed_loop(request, Target(), RawDraft(), head, eos=0)
    assert repaired.tokens == oracle
    assert repaired.accepted_tokens_per_round > raw.accepted_tokens_per_round
    assert len(head.boundaries) == 1
    assert head.boundaries[0].valid_prefix == (4, 1)
    assert head.boundaries[0].corrected_prefix == (4, 1, 2)
