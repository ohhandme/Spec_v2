"""P1 B/C/F diagnostic adapters with matched edge capacity and causal inputs.

Every arm trains the same local encoder, two codebook lookups, output repair,
local Hidden surrogate and local logit-summary surrogate.  F-H adds the *raw*
valid-prefix Target hidden to the shared Hidden projection input.  F-LOGIT adds
only the same-forward TargetLogitSummary to the shared summary input.  These
uncompressed Target features are diagnostic upper bounds, not 64-bit payloads
or claims about packed/WAN latency.  Future Target tokens never enter a view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor, nn

from .codec import CodecConfig, MatchedP1Codec
from .decoding import TargetLogitSummary
from .fixed_probe import (
    EdgeView, ProbeObservation, ProtocolView, TargetLogitView, TargetView,
)


UpperMode = Literal["edge_only", "protocol_only", "full_hidden", "full_logit"]
UPPER_MODES = ("edge_only", "protocol_only", "full_hidden", "full_logit")
_SUMMARY_TOP_K = 32
_SUMMARY_DIM = 2 * _SUMMARY_TOP_K + 3


class FrozenDraftView(Protocol):
    model: nn.Module

    def teacher_hidden(self, prefix: list[int], continuation: list[int]) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class PreparedUpperState:
    """One legal boundary; deliberately has no continuation or future label."""

    mode: UpperMode
    corrected_prefix: tuple[int, ...]
    local_hidden: Tensor
    rejected_embedding: Tensor | None
    correction_embedding: Tensor | None
    accepted_count: int | None
    target_hidden: Tensor | None
    target_logit_features: Tensor | None


def _summary_features(summary: TargetLogitSummary, vocab_size: int) -> Tensor:
    if type(summary) is not TargetLogitSummary:
        raise TypeError("F-LOGIT requires a TargetLogitSummary")
    # Validate even when a caller constructs the dataclass directly instead of
    # using its JSON loader.  No Target forward or future oracle is called.
    checked = TargetLogitSummary.from_json(summary.to_json())
    if checked.vocab_size != vocab_size:
        raise ValueError("Target logit summary vocabulary differs from the draft")
    k = len(checked.top_token_ids)
    ids = [token / vocab_size for token in checked.top_token_ids]
    relative_logits = [value - checked.logsumexp_fp32 for value in checked.top_logits_fp16]
    values = (
        ids + [0.0] * (_SUMMARY_TOP_K - k)
        + relative_logits + [0.0] * (_SUMMARY_TOP_K - k)
        + [checked.logsumexp_fp32, checked.top2_margin_fp32,
           checked.greedy_token_id / vocab_size]
    )
    result = torch.tensor([values], dtype=torch.float32)
    if result.shape != (1, _SUMMARY_DIM) or not torch.isfinite(result).all():
        raise ValueError("invalid current-boundary logit summary features")
    return result


class MatchedUpperBound(nn.Module):
    """One shared repair graph; the four arms differ only in allowed inputs.

    The local surrogate paths are active for *all* arms.  The F modes add a
    legal Target feature to the corresponding surrogate before its projection;
    thus B/C do not lose trainable parameters or a repair path.  The high-D
    projection and its cost are explicitly counted in every arm.
    """

    def __init__(self, config: CodecConfig, output_head: nn.Module,
                 *, mode: UpperMode) -> None:
        super().__init__()
        if mode not in UPPER_MODES:
            raise ValueError(f"unknown P1 upper-bound mode: {mode}")
        self.mode: UpperMode = mode
        self.config = config
        core_mode = "edge_only" if mode == "edge_only" else "protocol_only"
        self.core = MatchedP1Codec(config, output_head, mode=core_mode)
        control_dim = config.groups * config.code_dim
        self.local_hidden_lift = nn.Sequential(
            nn.Linear(config.draft_hidden_size, config.feature_dim), nn.SiLU(),
            nn.Linear(config.feature_dim, config.target_hidden_size),
        )
        self.hidden_projection = nn.Linear(config.target_hidden_size, control_dim)
        self.local_logit_surrogate = nn.Linear(config.draft_hidden_size, _SUMMARY_DIM)
        self.logit_projection = nn.Linear(_SUMMARY_DIM, control_dim)

    @staticmethod
    def _protocol_embedding(table: nn.Module, token: int | None) -> Tensor | None:
        if token is None:
            return None
        weight = table.weight
        if type(token) is not int or not 0 <= token < weight.shape[0]:
            raise ValueError("protocol token ID is outside the frozen draft vocabulary")
        return weight[token].detach().clone().reshape(1, -1)

    @torch.no_grad()
    def prepare_view(self, view: EdgeView | ProtocolView | TargetView | TargetLogitView,
                     draft: FrozenDraftView) -> PreparedUpperState:
        required_type = {
            "edge_only": EdgeView, "protocol_only": ProtocolView,
            "full_hidden": TargetView, "full_logit": TargetLogitView,
        }[self.mode]
        if type(view) is not required_type:
            raise TypeError(f"{self.mode} received information outside its permitted view")
        if not view.corrected_prefix or view.window != self.config.window:
            raise ValueError("upper-bound view has a wrong prefix or window")
        if any(parameter.requires_grad for parameter in draft.model.parameters()):
            raise ValueError("Draft base must be frozen")
        hidden = draft.teacher_hidden(list(view.corrected_prefix), [0])
        if hidden.shape != (1, self.config.draft_hidden_size) or not torch.isfinite(hidden).all():
            raise ValueError("draft local hidden has the wrong shape or nonfinite values")
        local = hidden.detach().clone()
        rejected = correction = None
        accepted_count = None
        target_hidden = logit_features = None
        if self.mode != "edge_only":
            assert isinstance(view, ProtocolView)
            if (view.corrected_prefix[-1] != view.correction_token
                    or type(view.accepted_count) is not int
                    or not 0 <= view.accepted_count <= self.config.window
                    or (view.rejected_token is None) !=
                       (view.accepted_count == self.config.window)):
                raise ValueError("invalid ordinary protocol boundary")
            table = draft.model.get_input_embeddings()
            if table is None or any(p.requires_grad for p in table.parameters()):
                raise ValueError("Draft input table must be frozen")
            correction = self._protocol_embedding(table, view.correction_token)
            rejected = self._protocol_embedding(table, view.rejected_token)
            accepted_count = view.accepted_count
        if self.mode == "full_hidden":
            assert isinstance(view, TargetView)
            state = view.target_hidden
            if (not isinstance(state, Tensor)
                    or state.shape != (self.config.target_hidden_size,)
                    or not state.is_floating_point() or not torch.isfinite(state).all()):
                raise ValueError("F-H requires the raw finite valid Target hidden")
            target_hidden = state.detach().clone().reshape(1, -1)
        elif self.mode == "full_logit":
            assert isinstance(view, TargetLogitView)
            if view.target_logit_summary.greedy_token_id != view.correction_token:
                raise ValueError("F-LOGIT summary is not from the correction boundary")
            logit_features = _summary_features(
                view.target_logit_summary, self.config.vocab_size)
        return PreparedUpperState(
            self.mode, view.corrected_prefix, local, rejected, correction,
            accepted_count, target_hidden, logit_features,
        )

    def prepare_observation(self, observation: ProbeObservation,
                            draft: FrozenDraftView) -> PreparedUpperState:
        """Validate cached provenance, then expose exactly one permitted view."""
        if type(observation) is not ProbeObservation:
            raise TypeError("prepare_observation accepts no ProbeExample/future labels")
        if (observation.target_hidden_prefix != observation.valid_prefix
                or observation.valid_prefix != observation.source_prefix
                   + observation.proposals[:observation.accepted_count]
                or observation.corrected_prefix != observation.valid_prefix
                   + (observation.correction_token,)
                or observation.remaining_budget <= 0):
            raise ValueError("observation has an invalid causal verification boundary")
        view = {
            "edge_only": observation.edge_view,
            "protocol_only": observation.protocol_view,
            "full_hidden": observation.target_view,
            "full_logit": observation.target_logit_view,
        }[self.mode]()
        return self.prepare_view(view, draft)

    def forward_logits(self, draft_hidden: Tensor, state: PreparedUpperState,
                       position: int | Tensor, *, temperature: float | None = None) -> Tensor:
        """Return repaired logits; caller owns CE labels and matched schedule.

        `temperature=None` uses deterministic hard local indices for inference.
        A positive temperature enables straight-through local codes for CE
        training.  No Target model or Target future is accepted by this API.
        """
        if type(state) is not PreparedUpperState or state.mode != self.mode:
            raise TypeError("prepared state is for a different information arm")
        if ((self.mode == "full_hidden") != (state.target_hidden is not None)
                or (self.mode == "full_logit") !=
                   (state.target_logit_features is not None)):
            raise TypeError("prepared state contains a forbidden or missing Target feature")
        if self.mode == "edge_only" and any(x is not None for x in (
                state.rejected_embedding, state.correction_embedding,
                state.accepted_count)):
            raise TypeError("B prepared state contains protocol metadata")
        if self.mode != "edge_only" and (state.correction_embedding is None
                                          or state.accepted_count is None):
            raise TypeError("protocol prepared state is incomplete")
        batch = state.local_hidden.shape[0]
        if batch != 1 or draft_hidden.shape[0] != batch:
            raise ValueError("upper-bound pilot expects one matched window at a time")
        if (draft_hidden.shape[-1] != self.config.draft_hidden_size
                or draft_hidden.ndim not in (2, 3)):
            raise ValueError("draft hidden has the wrong shape")
        local = state.local_hidden
        rejected = state.rejected_embedding
        correction = state.correction_embedding
        count = state.accepted_count
        if temperature is None:
            codes = self.core.local_encode(local, rejected, correction, count)
        else:
            codes = self.core.local_sample_st(local, rejected, correction, count,
                                              temperature=temperature)
        first, second = codes[:, 0], codes[:, 1]
        control = (self.core.repair._lookup(first, batch)
                   + self.core.repair._lookup(second, batch))
        surrogate = self.local_hidden_lift(local.to(
            device=self.local_hidden_lift[0].weight.device,
            dtype=self.local_hidden_lift[0].weight.dtype))
        if state.target_hidden is not None:
            surrogate = surrogate + state.target_hidden.to(
                device=surrogate.device, dtype=surrogate.dtype)
        hidden_control = self.hidden_projection(surrogate)
        summary = self.local_logit_surrogate(local.to(
            device=self.local_logit_surrogate.weight.device,
            dtype=self.local_logit_surrogate.weight.dtype))
        if state.target_logit_features is not None:
            summary = summary + state.target_logit_features.to(
                device=summary.device, dtype=summary.dtype)
        logit_control = self.logit_projection(summary)
        return self.core.repair.repair_from_control(
            draft_hidden, control + hidden_control + logit_control, position)

    @torch.inference_mode()
    def next_token(self, prefix: list[int], state: PreparedUpperState,
                   position: int, draft: FrozenDraftView) -> int:
        if tuple(prefix[:len(state.corrected_prefix)]) != state.corrected_prefix:
            raise ValueError("draft prefix left the prepared corrected-prefix branch")
        hidden = draft.teacher_hidden(prefix, [0])
        if hidden.shape != (1, self.config.draft_hidden_size):
            raise ValueError("draft next-token hidden has the wrong shape")
        logits = self.forward_logits(hidden, state, position)
        if logits.shape != (1, self.config.vocab_size) or not torch.isfinite(logits).all():
            raise RuntimeError("upper-bound arm returned invalid draft logits")
        return int(logits[0].argmax(-1).item())

    def parameter_budget(self) -> dict[str, int | None]:
        core = self.core.parameter_budget()
        added = sum(p.numel() for module in (
            self.local_hidden_lift, self.hidden_projection,
            self.local_logit_surrogate, self.logit_projection,
        ) for p in module.parameters())
        teacher_elements = (self.config.target_hidden_size if self.mode == "full_hidden"
                            else _SUMMARY_DIM if self.mode == "full_logit" else 0)
        return {"edge_trainable": core["edge_trainable"] + added,
                "cloud_trainable": 0, "diagnostic_teacher_elements": teacher_elements,
                "wire_bytes": None}


def assert_matched_capacity(models: dict[str, MatchedUpperBound],
                            *, equal_initial_weights: bool = True) -> dict[str, int]:
    """Fail before P1 if active architecture or initialization is mismatched."""
    if set(models) != set(UPPER_MODES) or any(model.mode != name for name, model in models.items()):
        raise ValueError("P1 requires exactly B/C/F-H/F-LOGIT mapped to their modes")
    reference = models[UPPER_MODES[0]]
    parameters = {name: parameter for name, parameter in reference.named_parameters()
                  if parameter.requires_grad}
    reference_head = reference.core.repair.output_head.state_dict()
    for mode in UPPER_MODES:
        model = models[mode]
        current = {name: parameter for name, parameter in model.named_parameters()
                   if parameter.requires_grad}
        if (set(current) != set(parameters)
                or any(current[name].shape != parameter.shape
                       for name, parameter in parameters.items())
                or model.parameter_budget()["edge_trainable"]
                   != reference.parameter_budget()["edge_trainable"]
                or any(p.requires_grad for p in model.core.repair.output_head.parameters())
                or set(model.core.repair.output_head.state_dict()) != set(reference_head)
                or any(not torch.equal(model.core.repair.output_head.state_dict()[key], value)
                       for key, value in reference_head.items())):
            raise ValueError(f"P1 edge architecture/parameter mismatch: {mode}")
        if equal_initial_weights and any(not torch.equal(
                current[name].detach(), parameter.detach())
                for name, parameter in parameters.items()):
            raise ValueError(f"P1 trainable initialization mismatch: {mode}")
    return {"edge_trainable_each": sum(p.numel() for p in parameters.values()),
            "arms": len(models)}
