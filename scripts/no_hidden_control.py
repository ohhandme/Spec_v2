"""P1-B control that removes Target-hidden information without changing D.

The normal D codec and frozen evaluator are left untouched. During training,
every cached Target-hidden vector is replaced in memory by the same nonzero
constant. After training, the cloud projection's response to that constant
is folded into its bias and its weight is set to zero. The saved D checkpoint
then emits the same eight-byte message for every possible Target hidden.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from feedback_codec.codec import MatchedP1Codec
from feedback_codec.fixed_probe import ProbeExample


def constant_hidden_examples(examples: list[ProbeExample]) -> list[ProbeExample]:
    """Keep prefixes, labels and exposure intact; remove only h_T values."""
    return [
        replace(item, observation=replace(
            item.observation,
            target_hidden=torch.ones_like(item.observation.target_hidden),
        ))
        for item in examples
    ]


@torch.no_grad()
def fold_constant_hidden(codec: MatchedP1Codec) -> None:
    """Make the trained constant-input encoder independent of its input.

    For projection z = W*1 + b, replace W by zero and b by z. All cloud and
    edge modules, message bytes and downstream computations retain their
    original shapes. This transformation is applied only after optimization.
    """
    if codec.mode != "target_8B" or codec.cloud_projection is None:
        raise ValueError("folding requires the unchanged target_8B architecture")
    projection = codec.cloud_projection
    if not isinstance(projection, torch.nn.Linear) or projection.bias is None:
        raise TypeError("cloud projection must be a biased linear layer")
    constant = torch.ones(
        (1, projection.in_features), device=projection.weight.device,
        dtype=projection.weight.dtype,
    )
    projected = projection(constant)[0].clone()
    projection.weight.zero_()
    projection.bias.copy_(projected)
    if torch.count_nonzero(projection.weight).item() != 0:
        raise RuntimeError("Target-hidden path was not removed")


@torch.no_grad()
def assert_no_target_hidden_information(
    codec: MatchedP1Codec, first: torch.Tensor, second: torch.Tensor,
) -> None:
    """Check the saved D codec is invariant to two distinct valid h_T values."""
    if codec.mode != "target_8B" or codec.cloud_projection is None:
        raise ValueError("no-hidden check requires target_8B codec")
    if first.shape != second.shape or first.ndim != 2 or torch.equal(first, second):
        raise ValueError("checks require two distinct same-shaped Target states")
    if torch.count_nonzero(codec.cloud_projection.weight).item() != 0:
        raise RuntimeError("cloud projection still reads Target-hidden values")
    first_scores = codec.cloud_scores(first)
    second_scores = codec.cloud_scores(second)
    if not torch.equal(first_scores, second_scores):
        raise RuntimeError("cloud message depends on Target-hidden values")
    if not torch.equal(codec.cloud_encode(first), codec.cloud_encode(second)):
        raise RuntimeError("cloud bytes depend on Target-hidden values")
