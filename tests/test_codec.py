"""Focused invariants for the feedback byte channel and repair objective."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from feedback_codec.codec import CodecConfig, FeedbackCodec, pack_codes, unpack_codes
from feedback_codec.losses import feedback_loss


def small_codec() -> FeedbackCodec:
    config = CodecConfig(
        target_hidden_size=12, draft_hidden_size=10, vocab_size=17,
        feature_dim=9, hidden_width=20, rank=4, code_dim=3,
    )
    return FeedbackCodec(config, nn.Linear(10, 17, bias=False))


def test_config_requires_exact_eight_byte_channel() -> None:
    with pytest.raises(ValueError, match="8 groups"):
        CodecConfig(12, 10, 17, groups=7)
    with pytest.raises(ValueError, match="8 groups"):
        CodecConfig(12, 10, 17, choices=128)
    with pytest.raises(ValueError, match="positive integer"):
        CodecConfig(12, 10, 0)


def test_codes_are_exactly_eight_bytes_and_round_trip() -> None:
    codes = torch.tensor([0, 1, 2, 127, 128, 253, 254, 255], dtype=torch.long)
    message = pack_codes(codes)
    assert isinstance(message, bytes)
    assert len(message) == 8
    assert torch.equal(unpack_codes(message), codes)
    batched = torch.stack((codes, 255 - codes))
    messages = pack_codes(batched)
    assert all(len(m) == 8 for m in messages)
    assert torch.equal(unpack_codes(messages), batched)
    with pytest.raises(ValueError):
        unpack_codes(b"short")
    with pytest.raises(ValueError):
        pack_codes(torch.zeros(7, dtype=torch.long))
    with pytest.raises(ValueError):
        pack_codes(torch.tensor([0, 1, 2, 3, 4, 5, 6, 256]))
    with pytest.raises(TypeError):
        pack_codes(torch.zeros(8, dtype=torch.float32))


def test_mixed_batch_null_marker_and_hard_shape() -> None:
    torch.manual_seed(2)
    codec = small_codec()
    target = torch.randn(2, 12)
    rejected = torch.randn(2, 12)
    correction = torch.randn(2, 12)
    counts = torch.tensor([2, 8], dtype=torch.long)
    original = codec.scores(target, rejected, correction, counts)
    changed = rejected.clone()
    changed[1] = 1000.0
    # The all-accepted example cannot leak a nonexistent rejected token.
    assert torch.equal(
        original[1], codec.scores(target, changed, correction, counts)[1]
    )
    assert original.shape == (2, 8, 256)
    assert codec.encode(target, rejected, correction, counts).shape == (2, 8)
    with pytest.raises(ValueError, match="required"):
        codec.scores(target, None, correction, counts)
    assert codec.scores(target[1:], None, correction[1:], 8).shape == (1, 8, 256)
    with pytest.raises(ValueError, match="disagrees"):
        codec.scores(
            target, rejected, correction, counts,
            rejected_mask=torch.tensor([False, False]),
        )


def test_st_training_reaches_encoder_codebooks_and_repair_but_not_head() -> None:
    torch.manual_seed(3)
    codec = small_codec()
    target = torch.randn(2, 12)
    rejected = torch.randn(2, 12)
    correction = torch.randn(2, 12)
    counts = torch.tensor([1, 3], dtype=torch.long)
    st = codec.sample_st(target, rejected, correction, counts, temperature=0.7)
    assert st.shape == (2, 8, 256)
    assert torch.equal(st.sum(-1), torch.ones(2, 8))
    assert torch.all((st == 0) | (st == 1))
    hidden = torch.randn(2, 4, 10)
    logits = codec.repair_logits(hidden, st, torch.tensor([1, 2, 3, 4]))
    assert logits.shape == (2, 4, 17)
    loss = feedback_loss(logits, torch.randint(0, 17, (2, 4)))["total"]
    loss.backward()
    assert codec.encoder[-1].weight.grad is not None
    assert codec.encoder[-1].weight.grad.abs().sum() > 0
    assert codec.codebooks.grad is not None and codec.codebooks.grad.abs().sum() > 0
    assert codec.U.weight.grad is not None and codec.U.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in codec.output_head.parameters())


def test_no_feedback_is_exact_frozen_head_baseline() -> None:
    codec = small_codec()
    hidden = torch.randn(2, 3, 10)
    expected = codec.output_head(hidden)
    actual = codec.repair_logits(hidden, None, 1, no_feedback=True)
    assert torch.equal(actual, expected)
    hard = torch.zeros((2, 8), dtype=torch.long)
    assert codec.repair_logits(hidden, hard, [1, 2, 3]).shape == expected.shape
    with pytest.raises(ValueError, match="required"):
        codec.repair_logits(hidden, None, 1)


def test_prefix_objective_matches_manual_formula_and_mask() -> None:
    # Two 2-token cases: target probabilities (0.8, 0.5) and (0.5, 0.5).
    probabilities = torch.tensor([
        [[0.8, 0.2], [0.5, 0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
    ], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    targets = torch.zeros((2, 2), dtype=torch.long)
    terms = feedback_loss(logits, targets, prefix_weight=2.0)
    expected_prefix = 1 - ((0.8 + 0.8 * 0.5) / 2 + (0.5 + 0.5 * 0.5) / 2) / 2
    expected_ce = -(math.log(0.8) + 3 * math.log(0.5)) / 4
    assert terms["prefix_surrogate"].item() == pytest.approx(expected_prefix)
    assert terms["cross_entropy"].item() == pytest.approx(expected_ce)
    assert terms["total"].item() == pytest.approx(expected_ce + 2 * expected_prefix)
    terms["total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    masked = feedback_loss(
        probabilities.log(), targets, torch.tensor([[1, 1], [1, 0]])
    )
    # Second example now has only one valid position, with survival 0.5.
    assert masked["prefix_surrogate"].item() == pytest.approx(
        (1 - (0.8 + 0.8 * 0.5) / 2 + 1 - 0.5) / 2
    )
    with pytest.raises(ValueError, match="contiguous prefix"):
        feedback_loss(torch.zeros(1, 3, 2), torch.zeros(1, 3, dtype=torch.long),
                      torch.tensor([[1, 0, 1]]))
