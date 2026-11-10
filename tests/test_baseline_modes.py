"""P1 controls: information isolation and matched edge-side capacity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from feedback_codec.codec import CodecConfig, MatchedP1Codec
from feedback_codec.fixed_probe import ProbeExample, ProbeObservation
from feedback_codec.training import TrajectoryExample, train_matched_p1_examples


def _config() -> CodecConfig:
    return CodecConfig(
        target_hidden_size=12, draft_hidden_size=10, vocab_size=17,
        feature_dim=9, hidden_width=20, rank=4, code_dim=3,
    )


def _codec(mode: str) -> MatchedP1Codec:
    return MatchedP1Codec(_config(), nn.Linear(10, 17, bias=False), mode=mode)


def test_matched_edge_budget_and_only_d_has_cloud_encoder() -> None:
    arms = {mode: _codec(mode) for mode in
            ("edge_only", "protocol_only", "target_8B")}
    budgets = {mode: codec.parameter_budget() for mode, codec in arms.items()}
    assert len({budget["edge_trainable"] for budget in budgets.values()}) == 1
    assert budgets["edge_only"]["message_bytes"] == 0
    assert budgets["protocol_only"]["message_bytes"] == 0
    assert budgets["target_8B"]["message_bytes"] == 8
    assert budgets["edge_only"]["cloud_trainable"] == 0
    assert budgets["protocol_only"]["cloud_trainable"] == 0
    assert budgets["target_8B"]["cloud_trainable"] > 0
    for mode, codec in arms.items():
        assert sum(p.numel() for p in codec.parameters() if p.requires_grad) == (
            budgets[mode]["edge_trainable"] + budgets[mode]["cloud_trainable"]
        )


def test_edge_only_rejects_protocol_and_cloud_inputs() -> None:
    codec = _codec("edge_only")
    hidden = torch.randn(1, 10)
    local = codec.local_encode(hidden)
    assert local.shape == (1, 2, 8)
    assert codec.repair_logits(hidden, local, 1).shape == (1, 17)
    with pytest.raises(ValueError, match="protocol metadata"):
        codec.local_encode(hidden, accepted_count=0)
    with pytest.raises(ValueError, match="only in target_8B"):
        codec.cloud_encode(torch.randn(1, 12))
    with pytest.raises(ValueError, match="cannot receive cloud_codes"):
        codec.repair_logits(hidden, local, 1, local)


def test_protocol_only_uses_correction_and_rejects_target_message() -> None:
    torch.manual_seed(4)
    codec = _codec("protocol_only")
    local = torch.randn(2, 10)
    rejected = torch.randn(2, 10)
    correction = torch.randn(2, 10)
    counts = torch.tensor([0, 8], dtype=torch.long)
    first = codec._local_scores(local, rejected, correction, counts)
    different_correction = correction + 10
    second = codec._local_scores(local, rejected, different_correction, counts)
    assert not torch.equal(first, second)
    changed_rejected = rejected.clone()
    changed_rejected[1] = 1000
    assert torch.equal(
        first[1], codec._local_scores(local, changed_rejected, correction, counts)[1]
    )
    with pytest.raises(ValueError, match="require correction"):
        codec.local_encode(local, accepted_count=counts)
    with pytest.raises(ValueError, match="required for rejected"):
        codec.local_encode(local, correction_embedding=correction,
                           accepted_count=counts)
    with pytest.raises(ValueError, match="only in target_8B"):
        codec.cloud_encode(torch.randn(2, 12))


def test_d_message_is_eight_bytes_and_gradients_reach_both_sides() -> None:
    torch.manual_seed(8)
    codec = _codec("target_8B")
    local = torch.randn(1, 10)
    target = torch.randn(1, 12)
    correction = torch.randn(1, 10)
    rejected = torch.randn(1, 10)
    message = codec.cloud_encode(target)
    payload = codec.pack_codes(message[0])
    assert isinstance(payload, bytes) and len(payload) == 8
    assert torch.equal(codec.unpack_codes(payload), message[0])
    local_st = codec.local_sample_st(local, rejected, correction, 0)
    cloud_st = codec.cloud_sample_st(target)
    logits = codec.repair_logits(local, local_st, 1, cloud_st)
    logits.square().mean().backward()
    assert codec.cloud_projection.weight.grad is not None
    assert codec.cloud_projection.weight.grad.abs().sum() > 0
    assert codec.local_encoder[-1].weight.grad is not None
    local_grad = codec.local_encoder[-1].weight.grad
    assert local_grad[: 8 * 256].abs().sum() > 0
    assert local_grad[8 * 256 :].abs().sum() > 0
    assert codec.repair.codebooks.grad is not None
    assert codec.repair.codebooks.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="requires cloud_codes"):
        codec.repair_logits(local, codec.local_encode(local, rejected, correction, 0), 1)


def test_modular_message_shift_matches_hard_and_st_paths() -> None:
    codec = _codec("target_8B")
    hidden = torch.randn(1, 10)
    local = torch.tensor([[[1] * 8, [250] * 8]], dtype=torch.long)
    cloud = torch.tensor([[10] * 8], dtype=torch.long)
    hard = codec.repair_logits(hidden, local, 1, cloud)
    local_st = torch.nn.functional.one_hot(local, 256).float()
    cloud_st = torch.nn.functional.one_hot(cloud, 256).float()
    soft = codec.repair_logits(hidden, local_st, 1, cloud_st)
    assert torch.allclose(hard, soft, atol=1e-6, rtol=0)
    shifted = (local[:, 1] + cloud) % 256
    expected_control = codec.repair._lookup(local[:, 0], 1) + codec.repair._lookup(shifted, 1)
    assert torch.allclose(
        hard, codec.repair.repair_from_control(hidden, expected_control, 1),
        atol=1e-6, rtol=0,
    )


def test_trainable_checkpoint_omits_frozen_base_and_checks_exact_keys() -> None:
    codec = _codec("target_8B")
    state = codec.trainable_state_dict()
    assert state and not any("output_head" in name for name in state)
    assert sum(t.numel() for t in state.values()) == sum(
        codec.parameter_budget()[key] for key in ("edge_trainable", "cloud_trainable")
    )
    fresh = _codec("target_8B")
    fresh.load_trainable_state_dict(state)
    assert all(torch.equal(value, fresh.trainable_state_dict()[name])
               for name, value in state.items())
    with pytest.raises(ValueError, match="key mismatch"):
        fresh.load_trainable_state_dict({**state, "extra": torch.zeros(1)})
    bad = dict(state)
    key = next(iter(bad))
    bad[key] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        fresh.load_trainable_state_dict(bad)


class _Target:
    def __init__(self) -> None:
        self.model = nn.Embedding(17, 12).requires_grad_(False)

    def next(self, prefix: list[int]):
        return SimpleNamespace(
            token_id=(prefix[-1] + 1) % 17,
            hidden=self.model(torch.tensor(prefix[-1])).detach(),
        )


class _DraftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 10)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding


class _Draft:
    def __init__(self) -> None:
        self.model = _DraftModel().requires_grad_(False)

    def teacher_hidden(self, prefix: list[int], continuation: list[int]) -> torch.Tensor:
        return self.model.embedding(torch.tensor(
            [prefix[-1], *continuation[:-1]], dtype=torch.long
        )).detach()


def test_matched_training_keeps_exposure_and_frozen_bases() -> None:
    target = _Target()
    draft = _Draft()
    # valid prefix (3,) predicts correction 4; the future starts at 5.
    item = TrajectoryExample(
        prompt_id="p", round_index=0, phase="train",
        valid_prefix=(3,), rejected_token=8, correction_token=4,
        accepted_count=0, corrected_prefix=(3, 4), continuation=(5, 6),
    )
    frozen_target = target.model.weight.detach().clone()
    frozen_draft = draft.model.embedding.weight.detach().clone()
    for mode in ("edge_only", "protocol_only", "target_8B"):
        codec = _codec(mode)
        optimizer = torch.optim.AdamW(
            (p for p in codec.parameters() if p.requires_grad), lr=1e-3
        )
        history = train_matched_p1_examples(
            examples=[item], target=target, draft=draft, codec=codec,
            optimizer=optimizer, steps=2, prefix_weight=0.5, seed=17,
        )
        assert [row["prompt_id"] for row in history] == ["p", "p"]
        assert all(torch.isfinite(torch.tensor(row["loss"])) for row in history)
    assert torch.equal(target.model.weight, frozen_target)
    assert torch.equal(draft.model.embedding.weight, frozen_draft)


def test_locked_probe_training_reuses_valid_target_state_without_new_target_call() -> None:
    class NoRequeryTarget(_Target):
        def next(self, prefix: list[int]):
            raise AssertionError("locked target state must be reused")

    target = NoRequeryTarget()
    draft = _Draft()
    observation = ProbeObservation(
        prompt_id="locked", family_id="family", split="train", task_type="toy",
        round_index=0, source_prefix=(3,), proposals=(8,), valid_prefix=(3,),
        accepted_count=0, rejected_token=8, correction_token=4,
        corrected_prefix=(3, 4), target_hidden=torch.randn(12),
        target_hidden_prefix=(3,), remaining_budget=8, window=8,
    )
    item = ProbeExample(observation, (5, 6))
    codec = _codec("target_8B")
    optimizer = torch.optim.AdamW(
        (p for p in codec.parameters() if p.requires_grad), lr=1e-3
    )
    history = train_matched_p1_examples(
        examples=[item], target=target, draft=draft, codec=codec,
        optimizer=optimizer, steps=1, prefix_weight=0.5, seed=19,
    )
    assert len(history) == 1 and history[0]["phase"] == "train"
    bad = ProbeExample(
        ProbeObservation(**{**observation.__dict__, "target_hidden_prefix": (9,)}),
        item.target_future_ids,
    )
    with pytest.raises(ValueError, match="not from the valid prefix"):
        train_matched_p1_examples(
            examples=[bad], target=target, draft=draft, codec=codec,
            optimizer=optimizer, steps=1, prefix_weight=0.5,
        )
    sealed = ProbeExample(
        ProbeObservation(**{**observation.__dict__, "split": "test"}),
        item.target_future_ids,
    )
    with pytest.raises(ValueError, match="sealed test"):
        train_matched_p1_examples(
            examples=[sealed], target=target, draft=draft, codec=codec,
            optimizer=optimizer, steps=1, prefix_weight=0.5,
        )


def test_training_resume_replays_exact_same_sampler_and_weights(tmp_path) -> None:
    target = _Target()
    draft = _Draft()
    rows = [TrajectoryExample(
        prompt_id=f"p{i}", round_index=0, phase="train",
        valid_prefix=(i,), rejected_token=8, correction_token=i + 1,
        accepted_count=0, corrected_prefix=(i, i + 1),
        continuation=(i + 2, i + 3),
    ) for i in (2, 3, 4)]
    torch.manual_seed(44)
    full = _codec("target_8B")
    full_opt = torch.optim.AdamW((p for p in full.parameters() if p.requires_grad), lr=1e-3)
    full_history = train_matched_p1_examples(
        examples=rows, target=target, draft=draft, codec=full,
        optimizer=full_opt, steps=4, prefix_weight=0.5, seed=31,
    )

    torch.manual_seed(44)
    resumed = _codec("target_8B")
    resumed_opt = torch.optim.AdamW((p for p in resumed.parameters() if p.requires_grad), lr=1e-3)
    save_path = tmp_path / "last.pt"

    def interrupt_at_two(state: dict) -> None:
        if state["completed_steps"] == 2:
            torch.save({
                "sampler": state,
                "codec": resumed.trainable_state_dict(),
                "optimizer": resumed_opt.state_dict(),
            }, save_path)
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_matched_p1_examples(
            examples=rows, target=target, draft=draft, codec=resumed,
            optimizer=resumed_opt, steps=4, prefix_weight=0.5, seed=31,
            checkpoint_every=2, checkpoint_callback=interrupt_at_two,
        )
    saved = torch.load(save_path, map_location="cpu", weights_only=True)
    torch.manual_seed(44)
    restored = _codec("target_8B")
    restored.load_trainable_state_dict(saved["codec"])
    restored_opt = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=1e-3)
    restored_opt.load_state_dict(saved["optimizer"])
    continued_history = train_matched_p1_examples(
        examples=rows, target=target, draft=draft, codec=restored,
        optimizer=restored_opt, steps=4, prefix_weight=0.5, seed=31,
        resume_state=saved["sampler"],
    )
    assert continued_history == full_history
    assert all(torch.equal(tensor, restored.trainable_state_dict()[name])
               for name, tensor in full.trainable_state_dict().items())
