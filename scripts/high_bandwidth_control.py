"""P1-B F: continuous Target-message upper bound for the frozen fixed probe.

F changes exactly the cloud message of matched D: eight hard byte indices are
replaced by the corresponding eight complete 256-way soft distributions.  The
same cloud encoder, local encoder, codebooks, modular composition, and repair
parameters remain active.  The only Target input is the causally valid h_T
that predicted this round's correction or bonus token.  This 8192-byte
diagnostic is never an 8-byte deployable method or a latency result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from feedback_codec.codec import CodecConfig, MatchedP1Codec
from feedback_codec.fixed_probe import ProbeArm, TargetView


MESSAGE_FLOATS = 8 * 256
MESSAGE_BYTES = MESSAGE_FLOATS * 4


class HighBandwidthP1Codec(MatchedP1Codec):
    """D with a differentiable, unquantized cloud distribution at train time."""

    def __init__(self, config: CodecConfig, output_head: nn.Module) -> None:
        super().__init__(config, output_head, mode="target_8B")

    def cloud_distribution(self, target_hidden: Tensor, *,
                           temperature: float = 1.0) -> Tensor:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        # The encoder's scores are FP32 trainable values.  No future target
        # continuation or post-correction target state enters this function.
        return torch.softmax(self.cloud_scores(target_hidden).float() /
                             temperature, dim=-1)

    def cloud_sample_st(self, target_hidden: Tensor, *,
                        temperature: float = 1.0) -> Tensor:
        # The frozen P1 trainer calls this hook.  F removes only the hard
        # cloud selection; local Gumbel/ST sampling is still unchanged.
        return self.cloud_distribution(target_hidden, temperature=temperature)

    def cloud_encode(self, target_hidden: Tensor) -> Tensor:
        raise RuntimeError("F must serialize continuous distributions, not D byte codes")

    def parameter_budget(self) -> dict[str, int]:
        budget = super().parameter_budget()
        budget["message_bytes"] = MESSAGE_BYTES
        return budget


def _validate_distribution(values: Tensor) -> Tensor:
    if values.shape == (8, 256):
        values = values.unsqueeze(0)
    if values.shape != (1, 8, 256):
        raise ValueError("F message must have one [8,256] distribution")
    values = values.float()
    if (not torch.isfinite(values).all().item()
            or (values < 0).any().item()
            or not torch.allclose(values.sum(-1),
                                  torch.ones_like(values.sum(-1)),
                                  atol=1e-5, rtol=1e-5)):
        raise ValueError("F message must contain eight finite probabilities")
    return values


def pack_distribution(values: Tensor) -> bytes:
    """Serialize the complete cloud distribution as little-endian FP32."""
    checked = _validate_distribution(values).detach().cpu().contiguous()
    payload = struct.pack("<" + "f" * MESSAGE_FLOATS,
                          *checked.reshape(-1).tolist())
    if len(payload) != MESSAGE_BYTES:
        raise RuntimeError("F wire size differs from 8192 bytes")
    return payload


def unpack_distribution(payload: bytes, *, device: torch.device | str) -> Tensor:
    if type(payload) is not bytes or len(payload) != MESSAGE_BYTES:
        raise ValueError("F message must be exactly 8192 bytes")
    values = torch.tensor(struct.unpack("<" + "f" * MESSAGE_FLOATS, payload),
                          dtype=torch.float32, device=device).reshape(1, 8, 256)
    return _validate_distribution(values)


@dataclass(frozen=True)
class HighBandwidthState:
    local: tuple[tuple[int, ...], tuple[int, ...]]
    message: bytes

    def __post_init__(self) -> None:
        if (len(self.local) != 2 or any(len(group) != 8 for group in self.local)
                or any(type(x) is not int or not 0 <= x < 256
                       for group in self.local for x in group)
                or type(self.message) is not bytes
                or len(self.message) != MESSAGE_BYTES):
            raise ValueError("invalid F prepared state")


def high_bandwidth_arm(draft: Any, codec: HighBandwidthP1Codec) -> ProbeArm:
    """Return a genuine autoregressive fixed-prefix F arm."""
    if not isinstance(codec, HighBandwidthP1Codec):
        raise TypeError("F arm requires HighBandwidthP1Codec")
    if not hasattr(draft, "teacher_hidden") or not hasattr(draft, "model"):
        raise TypeError("draft must provide frozen teacher_hidden and model")
    embedding_table = draft.model.get_input_embeddings().weight

    def embedding(token_id: int | None) -> Tensor | None:
        if token_id is None:
            return None
        if type(token_id) is not int or not 0 <= token_id < embedding_table.shape[0]:
            raise ValueError("protocol token ID is outside draft vocabulary")
        return embedding_table[token_id].detach().reshape(1, -1)

    @torch.inference_mode()
    def prepare(view: TargetView) -> HighBandwidthState:
        if type(view) is not TargetView:
            raise TypeError("F arm requires exactly the causally valid Target view")
        local_hidden = draft.teacher_hidden(list(view.corrected_prefix), [0])[0]
        local = codec.local_encode(
            local_hidden.reshape(1, -1), embedding(view.rejected_token),
            embedding(view.correction_token), view.accepted_count,
        )
        local_codes = tuple(tuple(int(x) for x in group)
                            for group in local[0].tolist())
        cloud = codec.cloud_distribution(view.target_hidden.reshape(1, -1))
        return HighBandwidthState(local=local_codes,
                                  message=pack_distribution(cloud))

    @torch.inference_mode()
    def next_token(prefix: list[int], state: HighBandwidthState,
                   position: int) -> int:
        if type(state) is not HighBandwidthState:
            raise TypeError("F next token requires its prepared state")
        hidden = draft.teacher_hidden(prefix, [0])[0].reshape(1, -1)
        device = codec.repair.codebooks.device
        local = torch.tensor([state.local], dtype=torch.long, device=device)
        local_st = F.one_hot(local, num_classes=256).float()
        cloud = unpack_distribution(state.message, device=device)
        logits = codec.repair_logits(hidden, local_st, position, cloud)
        if logits.shape != (1, codec.config.vocab_size) or not torch.isfinite(logits).all():
            raise RuntimeError("F fixed probe returned invalid logits")
        return int(logits[0].argmax(-1).item())

    return ProbeArm(prepare=prepare, next=next_token)
