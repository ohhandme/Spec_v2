"""Eight-byte verification feedback and an output-only draft repair.

The caller supplies *causally valid* target features from the ordinary
verification pass.  This module never runs the target model or reads future
continuation labels at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CodecConfig:
    target_hidden_size: int
    draft_hidden_size: int
    vocab_size: int
    groups: int = 8
    choices: int = 256
    code_dim: int = 16
    rank: int = 32
    window: int = 8
    feature_dim: int = 256
    hidden_width: int = 256

    def __post_init__(self) -> None:
        for name in (
            "target_hidden_size", "draft_hidden_size", "vocab_size", "groups",
            "choices", "code_dim", "rank", "window", "feature_dim",
            "hidden_width",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (self.groups, self.choices) != (8, 256):
            raise ValueError("the eight-byte codec requires 8 groups and 256 choices")


_INTEGER_DTYPES = {
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
}


def _checked_codes(codes: Tensor | Sequence[int]) -> Tensor:
    values = torch.as_tensor(codes)
    if values.dtype not in _INTEGER_DTYPES:
        raise TypeError("codes must have an integer dtype")
    if values.ndim not in (1, 2) or values.shape[-1] != 8:
        raise ValueError("codes must have shape [8] or [batch, 8]")
    if torch.any((values < 0) | (values > 255)).item():
        raise ValueError("each group index must lie in [0, 255]")
    return values.to(torch.long)


def pack_codes(codes: Tensor | Sequence[int]) -> bytes | list[bytes]:
    """Serialize eight group indices into exactly eight bytes per example."""
    checked = _checked_codes(codes).detach().cpu()
    if checked.ndim == 1:
        return bytes(checked.tolist())
    return [bytes(row.tolist()) for row in checked]


def unpack_codes(payload: bytes | bytearray | Sequence[bytes]) -> Tensor:
    """Deserialize one eight-byte message or a batch of messages on CPU."""
    if isinstance(payload, (bytes, bytearray)):
        if len(payload) != 8:
            raise ValueError("one feedback message must contain exactly 8 bytes")
        return torch.tensor(list(payload), dtype=torch.long)
    if not isinstance(payload, Sequence):
        raise TypeError("payload must be bytes or a sequence of byte strings")
    rows: list[list[int]] = []
    for item in payload:
        if not isinstance(item, (bytes, bytearray)) or len(item) != 8:
            raise ValueError("every feedback message must contain exactly 8 bytes")
        rows.append(list(item))
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 8)


class FeedbackCodec(nn.Module):
    """Cloud code encoder plus edge-side conditional low-rank logit repair.

    `target_hidden` is the verification state for the accepted prefix that
    predicts the correction/bonus token.  Embeddings are supplied by the
    frozen target embedding table; no target weights are copied here.
    `draft_hidden` must be the draft model's final-normalized hidden state.
    """

    def __init__(self, config: CodecConfig, output_head: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.output_head = output_head
        for parameter in self.output_head.parameters():
            parameter.requires_grad_(False)

        f = config.feature_dim
        self.target_projection = nn.Linear(config.target_hidden_size, f)
        self.embedding_projection = nn.Linear(config.target_hidden_size, f)
        self.null_rejected = nn.Parameter(torch.zeros(f))
        self.encoder = nn.Sequential(
            nn.Linear(3 * f + 1, config.hidden_width),
            nn.SiLU(),
            nn.Linear(config.hidden_width, config.groups * config.choices),
        )

        self.codebooks = nn.Parameter(
            torch.empty(config.groups, config.choices, config.code_dim)
        )
        nn.init.normal_(self.codebooks, std=0.02)
        self.V = nn.Linear(config.draft_hidden_size, config.rank, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(config.groups * config.code_dim + config.rank + 1,
                      config.hidden_width),
            nn.SiLU(),
            nn.Linear(config.hidden_width, config.rank),
        )
        self.U = nn.Linear(config.rank, config.draft_hidden_size, bias=False)
        nn.init.normal_(self.U.weight, std=1e-3)

    def scores(
        self,
        target_hidden: Tensor,
        rejected_embedding: Tensor | None,
        correction_embedding: Tensor,
        accepted_count: Tensor | int,
        *,
        rejected_mask: Tensor | None = None,
    ) -> Tensor:
        """Return [batch, 8, 256] scores using only current-round evidence.

        `rejected_mask` is true where a first rejected draft token exists.
        It defaults to `accepted_count < window`.  All-accepted rows use a
        learned null marker even if a placeholder embedding is supplied.
        """
        d = self.config.target_hidden_size
        if target_hidden.ndim != 2 or target_hidden.shape[1] != d:
            raise ValueError("target_hidden must have shape [batch, target_hidden_size]")
        batch = target_hidden.shape[0]
        if correction_embedding.shape != (batch, d):
            raise ValueError("correction_embedding has the wrong shape")
        counts = torch.as_tensor(accepted_count, device=target_hidden.device)
        if counts.ndim == 0:
            counts = counts.expand(batch)
        if counts.shape != (batch,) or counts.dtype not in _INTEGER_DTYPES:
            raise ValueError("accepted_count must be an integer scalar or [batch]")
        if torch.any((counts < 0) | (counts > self.config.window)).item():
            raise ValueError("accepted_count must lie in [0, window]")
        inferred_mask = counts < self.config.window
        if rejected_mask is None:
            rejected_mask = inferred_mask
        else:
            if rejected_mask.shape != (batch,) or rejected_mask.dtype != torch.bool:
                raise ValueError("rejected_mask must be boolean [batch]")
            if not torch.equal(rejected_mask.to(inferred_mask.device), inferred_mask):
                raise ValueError("rejected_mask disagrees with accepted_count")
        if rejected_embedding is None:
            if bool(torch.any(rejected_mask).item()):
                raise ValueError("rejected_embedding is required for rejected rows")
            rejected_feature = self.null_rejected.expand(batch, -1)
        else:
            if rejected_embedding.shape != (batch, d):
                raise ValueError("rejected_embedding has the wrong shape")
            projected = self.embedding_projection(
                rejected_embedding.to(self.embedding_projection.weight.dtype)
            )
            rejected_feature = torch.where(
                rejected_mask.to(projected.device).unsqueeze(-1),
                projected,
                self.null_rejected.expand(batch, -1),
            )
        target_feature = self.target_projection(
            target_hidden.to(self.target_projection.weight.dtype)
        )
        correction_feature = self.embedding_projection(
            correction_embedding.to(self.embedding_projection.weight.dtype)
        )
        fraction = (counts.to(target_feature.dtype) / self.config.window).unsqueeze(-1)
        features = torch.cat(
            (target_feature, rejected_feature, correction_feature, fraction), dim=-1
        )
        return self.encoder(features).reshape(batch, self.config.groups,
                                              self.config.choices)

    def encode(
        self,
        target_hidden: Tensor,
        rejected_embedding: Tensor | None,
        correction_embedding: Tensor,
        accepted_count: Tensor | int,
        *,
        rejected_mask: Tensor | None = None,
    ) -> Tensor:
        """Choose the hard eight-byte inference message as LongTensor[B,8]."""
        return self.scores(
            target_hidden, rejected_embedding, correction_embedding,
            accepted_count, rejected_mask=rejected_mask,
        ).argmax(dim=-1)

    def sample_st(
        self,
        target_hidden: Tensor,
        rejected_embedding: Tensor | None,
        correction_embedding: Tensor,
        accepted_count: Tensor | int,
        *,
        temperature: float = 1.0,
        rejected_mask: Tensor | None = None,
    ) -> Tensor:
        """Hard Gumbel selections with a soft straight-through gradient."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        scores = self.scores(
            target_hidden, rejected_embedding, correction_embedding,
            accepted_count, rejected_mask=rejected_mask,
        )
        return F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1)

    def _lookup(self, codes_or_st: Tensor, batch: int) -> Tensor:
        c = self.config
        if codes_or_st.ndim == 2:
            if codes_or_st.shape != (batch, c.groups):
                raise ValueError("hard codes must have shape [batch, 8]")
            codes = _checked_codes(codes_or_st)
            group = torch.arange(c.groups, device=codes.device)[None, :]
            vectors = self.codebooks[group, codes]
        elif codes_or_st.ndim == 3:
            if codes_or_st.shape != (batch, c.groups, c.choices):
                raise ValueError("ST codes must have shape [batch, 8, 256]")
            vectors = torch.einsum(
                "bgk,gkd->bgd", codes_or_st.to(self.codebooks.dtype), self.codebooks
            )
        else:
            raise ValueError("codes must be hard indices or ST one-hot selections")
        return vectors.reshape(batch, c.groups * c.code_dim)

    def repair_logits(
        self,
        draft_hidden: Tensor,
        codes_or_st: Tensor | None,
        position: Tensor | int,
        *,
        no_feedback: bool = False,
    ) -> Tensor:
        """Apply output-side repair; no-feedback returns the exact baseline head.

        The draft transformer/KV cache is untouched. `position` is the
        one-based position within the next draft window, divided by `window`.
        """
        d = self.config.draft_hidden_size
        if draft_hidden.ndim not in (2, 3) or draft_hidden.shape[-1] != d:
            raise ValueError("draft_hidden must have shape [B,D] or [B,L,D]")
        if no_feedback:
            return self.output_head(draft_hidden)
        if codes_or_st is None:
            raise ValueError("codes_or_st is required when feedback is enabled")
        control = self._lookup(codes_or_st, draft_hidden.shape[0])
        return self.repair_from_control(draft_hidden, control, position)

    def repair_from_control(
        self, draft_hidden: Tensor, control: Tensor, position: Tensor | int
    ) -> Tensor:
        """Apply the unchanged edge repair to an explicit control vector."""
        d = self.config.draft_hidden_size
        if draft_hidden.ndim not in (2, 3) or draft_hidden.shape[-1] != d:
            raise ValueError("draft_hidden must have shape [B,D] or [B,L,D]")
        single_position = draft_hidden.ndim == 2
        hidden = draft_hidden.unsqueeze(1) if single_position else draft_hidden
        batch, length, _ = hidden.shape
        if control.shape != (batch, self.config.groups * self.config.code_dim):
            raise ValueError("control must have shape [batch, groups * code_dim]")
        control = control.to(hidden.device)
        pos = torch.as_tensor(position, device=hidden.device, dtype=self.V.weight.dtype)
        if pos.ndim == 0:
            pos = pos.expand(batch, length)
        elif pos.shape == (length,):
            pos = pos[None, :].expand(batch, length)
        elif pos.shape == (batch,) and length == 1:
            pos = pos[:, None]
        elif pos.shape != (batch, length):
            raise ValueError("position must be scalar, [length], or [batch,length]")
        if torch.any((pos < 1) | (pos > self.config.window)).item():
            raise ValueError("position must lie in [1, window]")
        # Keep the small trainable repair in its parameter dtype; the frozen
        # output head receives the draft model's original activation dtype.
        u = self.V(hidden.to(self.V.weight.dtype))
        control = control.to(u.dtype)[:, None, :].expand(batch, length, -1)
        gate_input = torch.cat((control, u, (pos / self.config.window)[..., None]), -1)
        gate = self.gate_mlp(gate_input)
        delta = self.U(gate * u).to(hidden.dtype)
        repaired = hidden + delta
        if single_position:
            repaired = repaired[:, 0, :]
        return self.output_head(repaired)

    pack_codes = staticmethod(pack_codes)
    unpack_codes = staticmethod(unpack_codes)


P1Mode = Literal["edge_only", "protocol_only", "target_8B"]


class MatchedP1Codec(nn.Module):
    """P1 B/C/D controls with an identical trainable edge-side budget.

    All three arms execute the same local encoder, codebook, and output-only
    repair. B's local encoder sees only the corrected-prefix draft hidden; C
    also sees ordinary protocol token embeddings and accepted count. D sees
    precisely C's inputs, plus eight bytes encoded from the valid target
    verification hidden. D's cloud encoder is counted separately.
    """

    def __init__(
        self, config: CodecConfig, output_head: nn.Module, *, mode: P1Mode
    ) -> None:
        super().__init__()
        if mode not in ("edge_only", "protocol_only", "target_8B"):
            raise ValueError("mode must be edge_only, protocol_only, or target_8B")
        self.config = config
        self.mode: P1Mode = mode
        self.repair = FeedbackCodec(config, output_head)
        # Only the pre-existing repair part is reused. The original cloud
        # encoder is frozen and unused by all P1 arms.
        for module in (self.repair.target_projection,
                       self.repair.embedding_projection, self.repair.encoder):
            module.requires_grad_(False)
        self.repair.null_rejected.requires_grad_(False)

        f = config.feature_dim
        d = config.draft_hidden_size
        self.local_projection = nn.ModuleList([nn.Linear(d, f) for _ in range(3)])
        self.local_encoder = nn.Sequential(
            nn.Linear(3 * f + 1, config.hidden_width),
            nn.SiLU(),
            nn.Linear(config.hidden_width, 2 * config.groups * config.choices),
        )
        if mode == "target_8B":
            self.cloud_projection: nn.Module | None = nn.Linear(config.target_hidden_size, f)
            self.cloud_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(f, config.hidden_width), nn.SiLU(),
                nn.Linear(config.hidden_width, config.groups * config.choices),
            )
        else:
            self.cloud_projection = None
            self.cloud_encoder = None

    def _local_scores(
        self,
        local_hidden: Tensor,
        rejected_embedding: Tensor | None,
        correction_embedding: Tensor | None,
        accepted_count: Tensor | int | None,
    ) -> Tensor:
        d = self.config.draft_hidden_size
        if local_hidden.ndim != 2 or local_hidden.shape[1] != d:
            raise ValueError("local_hidden must have shape [batch, draft_hidden_size]")
        batch = local_hidden.shape[0]
        sources = [local_hidden, local_hidden, local_hidden]
        if self.mode == "edge_only":
            if any(x is not None for x in (rejected_embedding,
                                           correction_embedding, accepted_count)):
                raise ValueError("edge_only must not receive protocol metadata")
            fraction = torch.zeros((batch, 1), device=local_hidden.device)
        else:
            if correction_embedding is None or accepted_count is None:
                raise ValueError("protocol_only and target_8B require correction and count")
            counts = torch.as_tensor(accepted_count, device=local_hidden.device)
            if counts.ndim == 0:
                counts = counts.expand(batch)
            if counts.shape != (batch,) or counts.dtype not in _INTEGER_DTYPES:
                raise ValueError("accepted_count must be integer scalar or [batch]")
            if bool(torch.any((counts < 0) | (counts > self.config.window)).item()):
                raise ValueError("accepted_count must lie in [0, window]")
            if correction_embedding.shape != (batch, d):
                raise ValueError("correction_embedding has the wrong shape")
            rejected_mask = counts < self.config.window
            if rejected_embedding is None:
                if bool(rejected_mask.any().item()):
                    raise ValueError("rejected_embedding required for rejected rows")
            elif rejected_embedding.shape != (batch, d):
                raise ValueError("rejected_embedding has the wrong shape")
            if rejected_embedding is not None:
                # A full-acceptance row has no rejected token. Its fallback
                # comes only from local hidden, regardless of the placeholder.
                sources[1] = torch.where(
                    rejected_mask[:, None],
                    rejected_embedding.to(local_hidden.device), local_hidden,
                )
            sources[2] = correction_embedding
            fraction = counts.to(dtype=torch.float32)[:, None] / self.config.window
        projected = [
            layer(source.to(device=layer.weight.device, dtype=layer.weight.dtype))
            for layer, source in zip(self.local_projection, sources)
        ]
        fraction = fraction.to(device=projected[0].device, dtype=projected[0].dtype)
        features = torch.cat((*projected, fraction), dim=-1)
        return self.local_encoder(features).reshape(
            batch, 2, self.config.groups, self.config.choices
        )

    def local_encode(
        self,
        local_hidden: Tensor,
        rejected_embedding: Tensor | None = None,
        correction_embedding: Tensor | None = None,
        accepted_count: Tensor | int | None = None,
    ) -> Tensor:
        """Two independent eight-index local controls; none cross the network."""
        return self._local_scores(local_hidden, rejected_embedding,
                                  correction_embedding, accepted_count).argmax(-1)

    def local_sample_st(
        self,
        local_hidden: Tensor,
        rejected_embedding: Tensor | None = None,
        correction_embedding: Tensor | None = None,
        accepted_count: Tensor | int | None = None,
        *, temperature: float = 1.0,
    ) -> Tensor:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        scores = self._local_scores(local_hidden, rejected_embedding,
                                    correction_embedding, accepted_count)
        return F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1)

    def cloud_scores(self, target_hidden: Tensor) -> Tensor:
        """Encode only the causally valid target verification hidden."""
        if self.mode != "target_8B" or self.cloud_projection is None or self.cloud_encoder is None:
            raise ValueError("cloud encoder is available only in target_8B mode")
        if target_hidden.ndim != 2 or target_hidden.shape[1] != self.config.target_hidden_size:
            raise ValueError("target_hidden must have shape [batch, target_hidden_size]")
        projected = self.cloud_projection(target_hidden.to(
            device=self.cloud_projection.weight.device,
            dtype=self.cloud_projection.weight.dtype,
        ))
        return self.cloud_encoder(projected).reshape(
            target_hidden.shape[0], self.config.groups, self.config.choices
        )

    def cloud_encode(self, target_hidden: Tensor) -> Tensor:
        """Return D's exact eight byte-valued indices."""
        return self.cloud_scores(target_hidden).argmax(dim=-1)

    def cloud_sample_st(self, target_hidden: Tensor, *, temperature: float = 1.0) -> Tensor:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        return F.gumbel_softmax(
            self.cloud_scores(target_hidden), tau=temperature, hard=True, dim=-1,
        )

    def repair_logits(
        self,
        draft_hidden: Tensor,
        local_codes: Tensor,
        position: Tensor | int,
        cloud_codes: Tensor | None = None,
    ) -> Tensor:
        """Use two local controls; D's byte shifts the second code index.

        For each group, the effective second index is `(local_2 + cloud) % 256`.
        B/C use a zero shift. This keeps the two local encoder heads, two
        codebook lookups, and the downstream repair active in every arm. The
        training path computes the same operation on hard one-hot ST codes so
        gradients reach both local heads and D's cloud encoder.
        """
        batch = draft_hidden.shape[0]
        if local_codes.ndim == 3:
            if local_codes.shape != (batch, 2, self.config.groups):
                raise ValueError("hard local codes must have shape [batch, 2, 8]")
            first = _checked_codes(local_codes[:, 0])
            second = _checked_codes(local_codes[:, 1])
            soft = False
        elif local_codes.ndim == 4:
            if local_codes.shape != (batch, 2, self.config.groups, self.config.choices):
                raise ValueError("ST local codes must have shape [batch, 2, 8, 256]")
            first, second = local_codes[:, 0], local_codes[:, 1]
            soft = True
        else:
            raise ValueError("local codes must be hard indices or ST one-hot pairs")
        if self.mode == "target_8B":
            if cloud_codes is None:
                raise ValueError("target_8B requires cloud_codes")
        else:
            if cloud_codes is not None:
                raise ValueError("B/C cannot receive cloud_codes")
        if soft:
            if cloud_codes is None:
                cloud = torch.zeros_like(second)
                cloud[..., 0] = 1.0
            else:
                if cloud_codes.shape != (batch, self.config.groups, self.config.choices):
                    raise ValueError("ST cloud codes must have shape [batch, 8, 256]")
                cloud = cloud_codes
            k = self.config.choices
            positions = torch.arange(k, device=second.device)
            shifts = (positions[:, None] - positions[None, :]) % k
            # effective[k] = sum_i local_2[i] * cloud[(k-i) mod K].
            # Hard one-hot ST inputs yield exactly the hard modular sum in
            # the forward pass, while the backward pass stays differentiable.
            effective_second = torch.einsum(
                "bgi,bgki->bgk", second, cloud[:, :, shifts]
            )
        else:
            if cloud_codes is None:
                cloud = torch.zeros_like(second)
            else:
                if cloud_codes.shape != (batch, self.config.groups):
                    raise ValueError("hard cloud codes must have shape [batch, 8]")
                cloud = _checked_codes(cloud_codes)
            effective_second = (second + cloud) % self.config.choices
        control = self.repair._lookup(first, batch) + self.repair._lookup(
            effective_second, batch
        )
        return self.repair.repair_from_control(draft_hidden, control, position)

    def parameter_budget(self) -> dict[str, int]:
        """Report active edge versus additional D cloud trainable parameters."""
        edge_modules: tuple[nn.Module, ...] = (
            self.local_projection, self.local_encoder,
            self.repair.V, self.repair.gate_mlp, self.repair.U,
        )
        edge = sum(p.numel() for module in edge_modules for p in module.parameters())
        edge += self.repair.codebooks.numel()
        cloud = 0 if self.cloud_projection is None else sum(
            p.numel() for module in (self.cloud_projection, self.cloud_encoder)
            for p in module.parameters()
        )
        return {"edge_trainable": edge, "cloud_trainable": cloud,
                "message_bytes": 8 if self.mode == "target_8B" else 0}

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """Checkpoint only active parameters; frozen draft weights stay external."""
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        """Reject missing, extra, malformed, or nonfinite trainable tensors."""
        expected = {
            name: parameter for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            extra = sorted(set(state) - set(expected))
            raise ValueError(f"trainable checkpoint key mismatch: missing={missing}, extra={extra}")
        for name, parameter in expected.items():
            value = state[name]
            if not isinstance(value, Tensor) or value.shape != parameter.shape:
                raise ValueError(f"trainable checkpoint shape mismatch: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"trainable checkpoint is nonfinite: {name}")
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name].to(device=parameter.device,
                                                dtype=parameter.dtype))

    pack_codes = staticmethod(pack_codes)
    unpack_codes = staticmethod(unpack_codes)
