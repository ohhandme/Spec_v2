"""Matched, high-bandwidth P1 wiring heads for the bounded v10 train smoke.

The four arms have the same active repair graph and initial trainable weights.
They differ only in the legal information added at a corrected-prefix boundary:
B uses local Draft state, C also uses ordinary verification protocol data,
F-H adds the valid Target hidden, and F-CURRENT-TOP32 adds a semantic summary
of the current Target top-32 logits.  Neither F feature is an eight-byte code.

This module deliberately does not collect data, read future labels, or call the
Target.  The caller owns provenance validation and the CE labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol
import weakref

import torch
from torch import Tensor, nn

from .codec import CodecConfig, MatchedP1Codec
from .decoding import TargetLogitSummary
from .fixed_probe import (
    EdgeView, ProbeObservation, ProtocolView, TargetLogitView, TargetView,
)


V10Mode = Literal["B", "C", "F-H", "F-CURRENT-TOP32"]
V10_MODES: tuple[V10Mode, ...] = ("B", "C", "F-H", "F-CURRENT-TOP32")
TOP_K = 32


class FrozenDraftView(Protocol):
    model: nn.Module

    def teacher_hidden(self, prefix: list[int], continuation: list[int]) -> Tensor: ...


def top32_semantic_dim(draft_hidden_size: int) -> int:
    if type(draft_hidden_size) is not int or draft_hidden_size <= 0:
        raise ValueError("draft hidden size must be a positive integer")
    return 3 * draft_hidden_size + 67


@dataclass(frozen=True, slots=True)
class Top32SemanticEncoding:
    features: Tensor
    valid_k: int
    saved_top32_mass_fp32: float
    estimated_omitted_mass_fp32: float
    mass_excess_due_saved_rounding_fp32: float
    logit_storage: str = "saved_top_logits_fp16_promoted_to_fp32"


def semantic_current_top32(
    summary: TargetLogitSummary, draft_embedding: nn.Module | Tensor,
) -> Top32SemanticEncoding:
    """Encode saved same-boundary top-k without token-ID scalar shortcuts.

    Layout: E(top1), E(top2), top-k probability-weighted E mean, 32 log-p,
    32 masks, top2 margin, saved top-k probability mass, k/32.  A saved FP32
    full-vocabulary logsumexp normalizes the saved rounded top logits.  The
    omitted mass is an estimate, not a recovered full distribution.
    """
    if type(summary) is not TargetLogitSummary:
        raise TypeError("F-CURRENT-TOP32 requires TargetLogitSummary")
    checked = TargetLogitSummary.from_json(summary.to_json())
    weight = (draft_embedding if isinstance(draft_embedding, Tensor)
              else getattr(draft_embedding, "weight", None))
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise TypeError("Draft input embedding must expose a 2D weight")
    if weight.requires_grad:
        raise ValueError("Draft input embedding must be frozen")
    if weight.shape[0] != checked.vocab_size:
        raise ValueError("Target summary and Draft embedding vocabularies differ")
    if not weight.is_floating_point():
        raise TypeError("Draft embedding must be floating point")
    if any(a < b for a, b in zip(checked.top_logits_fp16,
                                 checked.top_logits_fp16[1:])):
        raise ValueError("Target top logits must be ordered from high to low")

    k = len(checked.top_token_ids)
    ids = torch.tensor(checked.top_token_ids, dtype=torch.long,
                       device=weight.device)
    with torch.no_grad():
        selected = weight.index_select(0, ids).detach().to(torch.float32)
        if not torch.isfinite(selected).all():
            raise ValueError("selected frozen Draft embeddings are nonfinite")
        top_logits = torch.tensor(checked.top_logits_fp16, dtype=torch.float32,
                                  device=weight.device)
        log_p = top_logits - torch.tensor(checked.logsumexp_fp32,
                                          dtype=torch.float32,
                                          device=weight.device)
        probabilities = log_p.exp()
        mass = probabilities.sum()
        if not torch.isfinite(log_p).all() or not torch.isfinite(mass) or mass <= 0:
            raise ValueError("saved top-k probabilities have invalid mass")
        weighted = (probabilities[:, None] * selected).sum(0) / mass
        log_p_padded = torch.zeros(TOP_K, dtype=torch.float32,
                                   device=weight.device)
        mask = torch.zeros(TOP_K, dtype=torch.float32, device=weight.device)
        log_p_padded[:k] = log_p
        mask[:k] = 1.0
        scalars = torch.tensor(
            [checked.top2_margin_fp32, float(mass.item()), k / TOP_K],
            dtype=torch.float32, device=weight.device,
        )
        features = torch.cat((selected[0], selected[1], weighted,
                              log_p_padded, mask, scalars)).reshape(1, -1)
        if features.shape != (1, top32_semantic_dim(weight.shape[1])):
            raise RuntimeError("semantic TOP32 feature dimension changed")
        if not torch.isfinite(features).all():
            raise ValueError("semantic TOP32 features are nonfinite")
        mass_float = float(mass.item())
    return Top32SemanticEncoding(
        features=features, valid_k=k, saved_top32_mass_fp32=mass_float,
        estimated_omitted_mass_fp32=max(0.0, 1.0 - mass_float),
        mass_excess_due_saved_rounding_fp32=max(0.0, mass_float - 1.0),
    )


@dataclass(frozen=True, slots=True)
class V10PreparedState:
    """A causal boundary with no future token labels or Target model handle."""

    mode: V10Mode
    corrected_prefix: tuple[int, ...]
    local_hidden: Tensor
    rejected_embedding: Tensor | None
    correction_embedding: Tensor | None
    accepted_count: int | None
    target_hidden: Tensor | None
    target_top32_features: Tensor | None
    top32_accounting: Top32SemanticEncoding | None


class V10FourArmHead(nn.Module):
    """Matched local-code and repair graph for four legal input views."""

    def __init__(
        self, config: CodecConfig, output_head: nn.Module, *, mode: V10Mode,
        draft_embedding: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if mode not in V10_MODES:
            raise ValueError(f"unknown v10 arm: {mode}")
        self.config = config
        self.mode: V10Mode = mode
        # Keep the optionally pinned table outside nn.Module registration.  It
        # is a frozen Draft base parameter, never a trainable head parameter.
        self._draft_embedding_ref = (weakref.ref(draft_embedding)
                                     if draft_embedding is not None else None)
        self.core = MatchedP1Codec(
            config, output_head,
            mode="edge_only" if mode == "B" else "protocol_only",
        )
        control_dim = config.groups * config.code_dim
        semantic_dim = top32_semantic_dim(config.draft_hidden_size)
        self.local_hidden_lift = nn.Sequential(
            nn.Linear(config.draft_hidden_size, config.feature_dim), nn.SiLU(),
            nn.Linear(config.feature_dim, config.target_hidden_size),
        )
        self.hidden_projection = nn.Linear(config.target_hidden_size, control_dim)
        self.local_top32_surrogate = nn.Linear(config.draft_hidden_size,
                                                semantic_dim)
        self.top32_projection = nn.Linear(semantic_dim, control_dim)

    @staticmethod
    def _protocol_embedding(table: nn.Module, token: int | None) -> Tensor | None:
        if token is None:
            return None
        weight = table.weight
        if type(token) is not int or not 0 <= token < weight.shape[0]:
            raise ValueError("protocol token is outside the frozen Draft vocabulary")
        return weight[token].detach().clone().reshape(1, -1)

    @torch.no_grad()
    def prepare_view(
        self, view: EdgeView | ProtocolView | TargetView | TargetLogitView,
        draft: FrozenDraftView,
    ) -> V10PreparedState:
        required = {"B": EdgeView, "C": ProtocolView,
                    "F-H": TargetView, "F-CURRENT-TOP32": TargetLogitView}[self.mode]
        if type(view) is not required:
            raise TypeError(f"{self.mode} received information outside its legal view")
        if not view.corrected_prefix or view.window != self.config.window:
            raise ValueError("wrong corrected prefix or window")
        if any(p.requires_grad for p in draft.model.parameters()):
            raise ValueError("Draft base must be frozen")
        local = draft.teacher_hidden(list(view.corrected_prefix), [0])
        if (local.shape != (1, self.config.draft_hidden_size)
                or not local.is_floating_point() or not torch.isfinite(local).all()):
            raise ValueError("Draft local hidden has wrong shape or nonfinite values")
        local = local.detach().clone()
        rejected = correction = None
        accepted_count = None
        hidden = top32 = accounting = None
        if self.mode != "B":
            assert isinstance(view, ProtocolView)
            if (view.corrected_prefix[-1] != view.correction_token
                    or type(view.accepted_count) is not int
                    or not 0 <= view.accepted_count <= self.config.window
                    or (view.rejected_token is None)
                       != (view.accepted_count == self.config.window)):
                raise ValueError("invalid ordinary protocol boundary")
            table = draft.model.get_input_embeddings()
            if table is None or any(p.requires_grad for p in table.parameters()):
                raise ValueError("Draft input table must be frozen")
            if (self._draft_embedding_ref is not None
                    and self._draft_embedding_ref() is not table):
                raise ValueError("Draft embedding differs from frozen head binding")
            correction = self._protocol_embedding(table, view.correction_token)
            rejected = self._protocol_embedding(table, view.rejected_token)
            accepted_count = view.accepted_count
            if self.mode == "F-H":
                assert isinstance(view, TargetView)
                if (not isinstance(view.target_hidden, Tensor)
                        or view.target_hidden.shape
                           != (self.config.target_hidden_size,)
                        or not view.target_hidden.is_floating_point()
                        or not torch.isfinite(view.target_hidden).all()):
                    raise ValueError("F-H requires finite valid Target hidden")
                hidden = view.target_hidden.detach().clone().reshape(1, -1)
            elif self.mode == "F-CURRENT-TOP32":
                assert isinstance(view, TargetLogitView)
                if view.target_logit_summary.greedy_token_id != view.correction_token:
                    raise ValueError("TOP32 summary is not from the correction boundary")
                accounting = semantic_current_top32(
                    view.target_logit_summary, table,
                )
                top32 = accounting.features
                if top32.shape != (1, top32_semantic_dim(
                        self.config.draft_hidden_size)):
                    raise ValueError("Draft embedding and configured hidden sizes differ")
        return V10PreparedState(
            self.mode, view.corrected_prefix, local, rejected, correction,
            accepted_count, hidden, top32, accounting,
        )

    def prepare_observation(
        self, observation: ProbeObservation, draft: FrozenDraftView,
    ) -> V10PreparedState:
        """Accept observations only; the separate future labels stay outside."""
        if type(observation) is not ProbeObservation:
            raise TypeError("prepare_observation accepts no future labels")
        if (observation.target_hidden_prefix != observation.valid_prefix
                or observation.valid_prefix
                   != observation.source_prefix
                      + observation.proposals[:observation.accepted_count]
                or observation.corrected_prefix
                   != observation.valid_prefix + (observation.correction_token,)
                or observation.remaining_budget <= 0):
            raise ValueError("invalid causal verification boundary")
        view = {"B": observation.edge_view,
                "C": observation.protocol_view,
                "F-H": observation.target_view,
                "F-CURRENT-TOP32": observation.target_logit_view}[self.mode]()
        return self.prepare_view(view, draft)

    def forward_logits(
        self, draft_hidden: Tensor, state: V10PreparedState,
        position: int | Tensor, *, temperature: float | None = None,
    ) -> Tensor:
        """Return repaired logits; CE labels never enter this method."""
        if type(state) is not V10PreparedState or state.mode != self.mode:
            raise TypeError("prepared state belongs to another information arm")
        if ((self.mode == "F-H") != (state.target_hidden is not None)
                or (self.mode == "F-CURRENT-TOP32")
                   != (state.target_top32_features is not None)):
            raise TypeError("missing or forbidden Target feature")
        if self.mode == "B" and any(x is not None for x in (
                state.rejected_embedding, state.correction_embedding,
                state.accepted_count)):
            raise TypeError("B state contains protocol metadata")
        if self.mode != "B" and (state.correction_embedding is None
                                  or state.accepted_count is None):
            raise TypeError("protocol state is incomplete")
        if (state.local_hidden.shape != (1, self.config.draft_hidden_size)
                or draft_hidden.ndim not in (2, 3)
                or draft_hidden.shape[0] != 1
                or draft_hidden.shape[-1] != self.config.draft_hidden_size):
            raise ValueError("Draft hidden has wrong shape")
        local = state.local_hidden
        if temperature is None:
            codes = self.core.local_encode(
                local, state.rejected_embedding,
                state.correction_embedding, state.accepted_count,
            )
        else:
            codes = self.core.local_sample_st(
                local, state.rejected_embedding,
                state.correction_embedding, state.accepted_count,
                temperature=temperature,
            )
        control = (self.core.repair._lookup(codes[:, 0], 1)
                   + self.core.repair._lookup(codes[:, 1], 1))
        surrogate = self.local_hidden_lift(local.to(
            device=self.local_hidden_lift[0].weight.device,
            dtype=self.local_hidden_lift[0].weight.dtype,
        ))
        if state.target_hidden is not None:
            surrogate = surrogate + state.target_hidden.to(
                device=surrogate.device, dtype=surrogate.dtype)
        hidden_control = self.hidden_projection(surrogate)
        semantic = self.local_top32_surrogate(local.to(
            device=self.local_top32_surrogate.weight.device,
            dtype=self.local_top32_surrogate.weight.dtype,
        ))
        if state.target_top32_features is not None:
            semantic = semantic + state.target_top32_features.to(
                device=semantic.device, dtype=semantic.dtype)
        top32_control = self.top32_projection(semantic)
        return self.core.repair.repair_from_control(
            draft_hidden, control + hidden_control + top32_control, position,
        )

    @torch.inference_mode()
    def next_token(
        self, prefix: list[int], state: V10PreparedState,
        position: int, draft: FrozenDraftView,
    ) -> int:
        if tuple(prefix[:len(state.corrected_prefix)]) != state.corrected_prefix:
            raise ValueError("Draft prefix left the prepared corrected-prefix branch")
        hidden = draft.teacher_hidden(prefix, [0])
        if hidden.shape != (1, self.config.draft_hidden_size):
            raise ValueError("Draft next-token hidden has wrong shape")
        logits = self.forward_logits(hidden, state, position)
        if logits.shape != (1, self.config.vocab_size) or not torch.isfinite(logits).all():
            raise RuntimeError("four-arm head returned invalid logits")
        return int(logits[0].argmax(-1).item())

    def parameter_budget(self) -> dict[str, int | None]:
        core = self.core.parameter_budget()
        added = sum(p.numel() for module in (
            self.local_hidden_lift, self.hidden_projection,
            self.local_top32_surrogate, self.top32_projection,
        ) for p in module.parameters())
        teacher_elements = (self.config.target_hidden_size if self.mode == "F-H"
                            else top32_semantic_dim(self.config.draft_hidden_size)
                            if self.mode == "F-CURRENT-TOP32" else 0)
        return {"edge_trainable": core["edge_trainable"] + added,
                "cloud_trainable": 0,
                "diagnostic_teacher_elements": teacher_elements,
                "wire_bytes": None}

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return {name: p.detach().cpu().clone()
                for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        expected = {name: p for name, p in self.named_parameters()
                    if p.requires_grad}
        if set(state) != set(expected):
            raise ValueError("trainable checkpoint keys differ")
        for name, parameter in expected.items():
            value = state[name]
            if (not isinstance(value, Tensor) or value.shape != parameter.shape
                    or not torch.isfinite(value).all()):
                raise ValueError(f"bad trainable checkpoint tensor: {name}")
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name].to(
                    device=parameter.device, dtype=parameter.dtype))


def assert_matched_v10_arms(
    arms: dict[str, V10FourArmHead], *, equal_initial_weights: bool = True,
) -> dict[str, int]:
    if set(arms) != set(V10_MODES) or any(arms[name].mode != name
                                          for name in V10_MODES):
        raise ValueError("v10 requires exactly B/C/F-H/F-CURRENT-TOP32")
    reference = arms[V10_MODES[0]]
    ref = {name: p for name, p in reference.named_parameters()
           if p.requires_grad}
    ref_head = reference.core.repair.output_head.state_dict()
    for mode in V10_MODES:
        model = arms[mode]
        current = {name: p for name, p in model.named_parameters()
                   if p.requires_grad}
        if (set(current) != set(ref)
                or any(current[name].shape != ref[name].shape for name in ref)
                or model.parameter_budget()["edge_trainable"]
                   != reference.parameter_budget()["edge_trainable"]
                or any(p.requires_grad for p in model.core.repair.output_head.parameters())
                or set(model.core.repair.output_head.state_dict()) != set(ref_head)
                or any(not torch.equal(
                    model.core.repair.output_head.state_dict()[key], value)
                    for key, value in ref_head.items())):
            raise ValueError(f"v10 active architecture mismatch: {mode}")
        if equal_initial_weights and any(not torch.equal(
                current[name].detach(), ref[name].detach()) for name in ref):
            raise ValueError(f"v10 trainable initialization mismatch: {mode}")
    return {"arms": 4, "edge_trainable_each": sum(p.numel() for p in ref.values())}


def build_matched_v10_arms(
    config: CodecConfig, output_head: nn.Module, *, seed: int,
    draft_embedding: nn.Module | None = None,
) -> dict[str, V10FourArmHead]:
    """Build identical initial trainable weights without changing caller RNG."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        result = {}
        for mode in V10_MODES:
            torch.manual_seed(seed)
            result[mode] = V10FourArmHead(
                config, output_head, mode=mode,
                draft_embedding=draft_embedding,
            )
    assert_matched_v10_arms(result)
    return result
