"""Teacher-forced next-window objective for feedback-coded drafting."""

from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F


def feedback_loss(
    logits: Tensor,
    target_ids: Tensor,
    mask: Tensor | None = None,
    *,
    prefix_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Return CE + lambda times a differentiable prefix-survival surrogate.

    For each example, survival at k is the product of teacher-forced target
    probabilities at positions 1..k. Its mean over valid positions is an
    approximation, *not* the measured greedy acceptance rate. `mask` must be
    a contiguous valid prefix for every example. CE averages valid tokens;
    the prefix term averages per-example normalized prefix lengths.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, length, vocabulary]")
    batch, length, vocabulary = logits.shape
    if target_ids.shape != (batch, length) or target_ids.dtype != torch.long:
        raise ValueError("target_ids must be LongTensor[batch, length]")
    if mask is None:
        valid = torch.ones_like(target_ids, dtype=torch.bool)
    else:
        if mask.shape != (batch, length):
            raise ValueError("mask must have shape [batch, length]")
        if mask.dtype == torch.bool:
            valid = mask
        else:
            if not bool(torch.all((mask == 0) | (mask == 1)).item()):
                raise ValueError("mask must contain only 0 and 1")
            valid = mask.bool()
    if not bool(torch.all(valid.any(dim=1)).item()):
        raise ValueError("each example needs at least one valid position")
    if length > 1 and bool(torch.any(valid[:, 1:] & ~valid[:, :-1]).item()):
        raise ValueError("mask must mark a contiguous prefix")
    if bool(torch.any((target_ids[valid] < 0) | (target_ids[valid] >= vocabulary)).item()):
        raise ValueError("valid target IDs must lie in the vocabulary")
    if (not isinstance(prefix_weight, (int, float)) or
            not math.isfinite(prefix_weight) or prefix_weight < 0):
        raise ValueError("prefix_weight must be finite and nonnegative")

    safe_ids = target_ids.masked_fill(~valid, 0)
    # Float32 avoids half-precision underflow when multiplying a full window.
    log_prob = F.log_softmax(logits.float(), dim=-1).gather(
        -1, safe_ids.unsqueeze(-1)
    ).squeeze(-1)
    cross_entropy = -log_prob.masked_fill(~valid, 0.0).sum() / valid.sum()
    target_probability = log_prob.exp().masked_fill(~valid, 1.0)
    survival = target_probability.cumprod(dim=1) * valid
    expected_prefix_length = survival.sum(dim=1).mean()
    prefix_surrogate = (1.0 - survival.sum(dim=1) / valid.sum(dim=1)).mean()
    total = cross_entropy + float(prefix_weight) * prefix_surrogate
    return {
        "total": total,
        "cross_entropy": cross_entropy,
        "prefix_surrogate": prefix_surrogate,
        "expected_prefix_length": expected_prefix_length,
    }
