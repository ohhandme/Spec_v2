"""Isolated causal Transformer repair for the existing B/C/F-H information arms.

Only the output-side residual is replaced.  The frozen Draft, local encoder,
codebooks, ordinary protocol features, and legal F-H Target view come from the
audited v10 implementation.  Training and free generation both present the
whole *current-window* Draft hidden history to the same causal stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from .codec import CodecConfig, FeedbackCodec
from .v10_four_arm import V10FourArmHead, V10PreparedState


ARMS = ("B", "C", "F-H")


@dataclass(frozen=True)
class TransformerRepairConfig:
    depth: int = 2
    width: int = 128
    heads: int = 4

    def __post_init__(self) -> None:
        if self.depth not in (1, 2) or self.width not in (64, 128):
            raise ValueError("repair depth/width must be a prespecified bounded choice")
        if self.heads != 4 or self.width % self.heads:
            raise ValueError("repair attention must use four dividing heads")


class _CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.0,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(),
                                 nn.Linear(4 * width, width))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        normalized = self.norm1(values)
        update, _ = self.attn(normalized, normalized, normalized,
                              attn_mask=mask, need_weights=False)
        values = values + update
        return values + self.ffn(self.norm2(values))


class CausalTransformerResidual(nn.Module):
    """Current-window causal attention; no state crosses a corrected boundary."""

    def __init__(self, hidden_size: int, control_size: int, window: int,
                 choice: TransformerRepairConfig) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.control_size = control_size
        self.window = window
        self.choice = choice
        self.down = nn.Linear(hidden_size, choice.width, bias=False)
        self.control = nn.Linear(control_size, choice.width, bias=False)
        self.position = nn.Linear(1, choice.width, bias=False)
        self.blocks = nn.ModuleList(_CausalBlock(choice.width, choice.heads)
                                    for _ in range(choice.depth))
        self.up = nn.Linear(choice.width, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, history: Tensor, control: Tensor) -> Tensor:
        if (history.ndim != 3 or not 1 <= history.shape[1] <= self.window
                or history.shape[2] != self.hidden_size
                or control.shape != (history.shape[0], self.control_size)):
            raise ValueError("repair expects [B,L,D] history and [B,C] control")
        if not torch.isfinite(history).all() or not torch.isfinite(control).all():
            raise ValueError("nonfinite repair input")
        x = history.to(dtype=self.down.weight.dtype)
        length = x.shape[1]
        positions = (torch.arange(1, length + 1, device=x.device, dtype=x.dtype)
                     .view(1, length, 1) / self.window)
        z = (self.down(x) + self.control(control.to(device=x.device, dtype=x.dtype))[:, None]
             + self.position(positions))
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            z = block(z, mask)
        return history + self.up(z).to(history.dtype)


class _TransformerFeedbackCodec(FeedbackCodec):
    """Keep both v10 codebooks and replace only V/gate/U output repair."""

    def __init__(self, config: CodecConfig, output_head: nn.Module,
                 choice: TransformerRepairConfig) -> None:
        super().__init__(config, output_head)
        for module in (self.target_projection, self.embedding_projection,
                       self.encoder):
            module.requires_grad_(False)
        self.null_rejected.requires_grad_(False)
        for module in (self.V, self.gate_mlp, self.U):
            module.requires_grad_(False)
        self.transformer = CausalTransformerResidual(
            config.draft_hidden_size, config.groups * config.code_dim,
            config.window, choice)

    def repair_from_control(self, draft_hidden: Tensor, control: Tensor,
                            position: Tensor | int) -> Tensor:
        if draft_hidden.ndim != 3:
            raise ValueError("Transformer repair needs the entire causal window history")
        length = draft_hidden.shape[1]
        positions = torch.as_tensor(position, device=draft_hidden.device)
        expected = torch.arange(1, length + 1, device=draft_hidden.device)
        if positions.shape != expected.shape or not torch.equal(positions, expected):
            raise ValueError("Transformer positions must be the full causal prefix 1..L")
        return self.output_head(self.transformer(draft_hidden, control))


class FrozenDraftHistory(Protocol):
    def teacher_hidden(self, prefix: list[int], continuation: list[int]) -> Tensor: ...


class TransformerBCFHHead(V10FourArmHead):
    """A matched B, C, or F-H head with a causal Transformer residual."""

    def __init__(self, config: CodecConfig, output_head: nn.Module, *, mode: str,
                 choice: TransformerRepairConfig = TransformerRepairConfig(),
                 draft_embedding: nn.Module | None = None) -> None:
        if mode not in ARMS:
            raise ValueError("this isolated experiment supports only B/C/F-H")
        super().__init__(config, output_head, mode=mode,
                         draft_embedding=draft_embedding)
        self.core.repair = _TransformerFeedbackCodec(config, output_head, choice)
        self.transformer_choice = choice

    @torch.inference_mode()
    def next_token(self, prefix: list[int], state: V10PreparedState,
                   position: int, draft: FrozenDraftHistory) -> int:
        corrected = state.corrected_prefix
        if state.mode != self.mode or tuple(prefix[:len(corrected)]) != corrected:
            raise ValueError("head/state mismatch or prefix left corrected branch")
        generated = prefix[len(corrected):]
        if type(position) is not int or not 1 <= position <= self.config.window or (
                len(generated) != position - 1):
            raise ValueError("current-window history and position differ")
        # teacher_hidden(prefix, x+[0]) returns one hidden for every already
        # generated token and the current next-token position.  No future
        # Target label or later Draft token enters this call.
        history = draft.teacher_hidden(list(corrected), list(generated) + [0])
        if history.shape != (position, self.config.draft_hidden_size):
            raise ValueError("frozen Draft returned the wrong causal history")
        positions = torch.arange(1, position + 1, device=history.device)
        logits = self.forward_logits(history.unsqueeze(0), state, positions)
        if (logits.shape != (1, position, self.config.vocab_size)
                or not torch.isfinite(logits).all()):
            raise ValueError("Transformer head returned invalid logits")
        return int(logits[0, -1].argmax(-1).item())

    def parameter_budget(self) -> dict[str, int]:
        active = {name: p for name, p in self.named_parameters() if p.requires_grad}
        repair = sum(p.numel() for p in self.core.repair.transformer.parameters())
        return {"edge_trainable": sum(p.numel() for p in active.values()),
                "repair_trainable": repair,
                "shared_encoder_trainable": sum(p.numel() for p in active.values()) - repair,
                "diagnostic_teacher_elements": (
                    self.config.target_hidden_size if self.mode == "F-H" else 0)}


def build_matched_transformer_arms(config: CodecConfig, output_head: nn.Module, *,
                                   seed: int,
                                   choice: TransformerRepairConfig = TransformerRepairConfig(),
                                   draft_embedding: nn.Module | None = None,
                                   ) -> dict[str, TransformerBCFHHead]:
    """Equal initialization and active parameters across all information arms."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        arms = {}
        for mode in ARMS:
            torch.manual_seed(seed)
            arms[mode] = TransformerBCFHHead(
                config, output_head, mode=mode, choice=choice,
                draft_embedding=draft_embedding)
    reference = arms["B"].trainable_state_dict()
    for mode, head in arms.items():
        current = head.trainable_state_dict()
        if (set(current) != set(reference) or any(
                not torch.equal(current[name], reference[name])
                for name in reference)):
            raise ValueError(f"{mode} Transformer head differs at initialization")
        if any(p.requires_grad for p in head.core.repair.output_head.parameters()):
            raise ValueError("frozen Draft output head became trainable")
    return arms
