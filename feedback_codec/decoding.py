"""Greedy speculative decoding with a causal, round-delayed feedback message.

This module deliberately leaves model execution to the backends.  In particular,
``TargetBackend.next`` must return the target's next-token decision and the
target hidden state for *the supplied confirmed prefix*.  A backend may cache
that prefix, but it must not return a hidden state from a rejected branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class TargetLogitSummary:
    """Current-boundary Target logits from the same forward as the decision.

    This is a diagnostic high-bandwidth feature, never an eight-byte payload.
    ``greedy_token_id`` retains the model's argmax tie rule; top-k ordering is
    allowed to differ when logits tie.
    """

    greedy_token_id: int
    top_token_ids: tuple[int, ...]
    top_logits_fp16: tuple[float, ...]
    logsumexp_fp32: float
    top2_margin_fp32: float
    vocab_size: int

    def to_json(self) -> dict:
        return {
            "greedy_token_id": self.greedy_token_id,
            "top_token_ids": list(self.top_token_ids),
            "top_logits_fp16": list(self.top_logits_fp16),
            "logsumexp_fp32": self.logsumexp_fp32,
            "top2_margin_fp32": self.top2_margin_fp32,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_json(cls, value: dict) -> "TargetLogitSummary":
        import math

        if not isinstance(value, dict):
            raise ValueError("target logit summary must be an object")
        token_ids = value.get("top_token_ids")
        logits = value.get("top_logits_fp16")
        vocab_size = value.get("vocab_size")
        greedy = value.get("greedy_token_id")
        lse = value.get("logsumexp_fp32")
        margin = value.get("top2_margin_fp32")
        if (
            type(vocab_size) is not int or vocab_size < 2
            or type(greedy) is not int or not 0 <= greedy < vocab_size
            or not isinstance(token_ids, list) or not isinstance(logits, list)
            or len(token_ids) != min(32, vocab_size) or len(logits) != len(token_ids)
            or len(set(token_ids)) != len(token_ids)
            or any(type(t) is not int or not 0 <= t < vocab_size for t in token_ids)
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)
            or type(lse) not in (int, float) or not math.isfinite(lse)
            or type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0
        ):
            raise ValueError("invalid current-boundary target logit summary")
        return cls(greedy, tuple(token_ids), tuple(float(x) for x in logits),
                   float(lse), float(margin), vocab_size)


@dataclass(frozen=True)
class TargetStep:
    token_id: int
    hidden: torch.Tensor
    logit_summary: TargetLogitSummary | None = None


class TargetBackend(Protocol):
    def next(self, prefix: list[int]) -> TargetStep:
        """Choose the greedy next token from a confirmed prefix."""

    def embedding(self, token_id: int) -> torch.Tensor:
        """Return the input embedding for one token ID."""


class DraftBackend(Protocol):
    def next(
        self, prefix: list[int], codes: tuple[int, ...] | None, position: int
    ) -> int:
        """Choose a draft token; ``position`` is one-based within the round."""


class FeedbackEncoderProtocol(Protocol):
    def encode(
        self,
        target_hidden: torch.Tensor,
        rejected_embedding: torch.Tensor | None,
        correction_embedding: torch.Tensor,
        accepted_count: int,
    ) -> tuple[int, ...]:
        """Encode only information available after this round's verification."""


@dataclass(frozen=True)
class RoundRecord:
    """One proposal window; feedback can only be used by a later round."""

    prefix_before: tuple[int, ...]
    codes_in: tuple[int, ...] | None
    proposals: tuple[int, ...]
    accepted_count: int
    committed_tokens: tuple[int, ...]
    rejected_token: int | None
    correction_token: int | None
    bonus_token: int | None
    observation_hidden: torch.Tensor | None
    feedback_codes: tuple[int, ...] | None
    message_bytes: int
    observation_logit_summary: TargetLogitSummary | None = None

    @property
    def draft_tokens(self) -> tuple[int, ...]:
        return self.proposals

    @property
    def codes_out(self) -> tuple[int, ...] | None:
        return self.feedback_codes


@dataclass(frozen=True)
class GenerationResult:
    """Generated tokens exclude the prompt; rounds preserve proposal details."""

    tokens: tuple[int, ...]
    rounds: tuple[RoundRecord, ...]

    @property
    def accepted_draft_tokens(self) -> int:
        return sum(round_.accepted_count for round_ in self.rounds)

    @property
    def verification_rounds(self) -> int:
        return len(self.rounds)

    @property
    def total_message_bytes(self) -> int:
        return sum(round_.message_bytes for round_ in self.rounds)


class SpeculativeDecoder:
    def __init__(
        self,
        target: TargetBackend,
        draft: DraftBackend,
        encoder: FeedbackEncoderProtocol | None,
        window: int = 8,
        eos_token_id: int | None = None,
        capture_target_logits: bool = False,
    ) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self.target = target
        self.draft = draft
        self.encoder = encoder
        self.window = window
        self.eos_token_id = eos_token_id
        self.capture_target_logits = capture_target_logits
        if capture_target_logits and not callable(
            getattr(target, "next_with_logit_summary", None)
        ):
            raise ValueError("capturing logits requires a same-forward Target backend")

    def generate(self, prompt: list[int], max_new_tokens: int) -> GenerationResult:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")

        confirmed = list(prompt)
        generated: list[int] = []
        records: list[RoundRecord] = []
        codes: tuple[int, ...] | None = None

        while len(generated) < max_new_tokens:
            prefix_before = tuple(confirmed)
            remaining = max_new_tokens - len(generated)
            proposal: list[int] = []
            for position in range(1, min(self.window, remaining) + 1):
                token = int(self.draft.next(confirmed + proposal, codes, position))
                proposal.append(token)
                if token == self.eos_token_id:
                    break

            codes_in = codes
            accepted = 0
            correction: int | None = None
            bonus: int | None = None
            rejected: int | None = None
            observation: torch.Tensor | None = None
            observation_logit_summary: TargetLogitSummary | None = None
            committed: list[int] = []

            for token in proposal:
                # The backend sees only tokens already confirmed by the target.
                step = (self.target.next_with_logit_summary(confirmed.copy())
                        if self.capture_target_logits else self.target.next(confirmed.copy()))
                if token != step.token_id:
                    rejected = token
                    correction = int(step.token_id)
                    observation = step.hidden
                    observation_logit_summary = step.logit_summary
                    committed.append(correction)
                    confirmed.append(correction)
                    generated.append(correction)
                    break

                accepted += 1
                committed.append(token)
                confirmed.append(token)
                generated.append(token)
                if token == self.eos_token_id:
                    break

            stopped_on_eos = bool(generated) and generated[-1] == self.eos_token_id
            if (
                rejected is None
                and not stopped_on_eos
                and len(generated) < max_new_tokens
            ):
                # All proposed tokens matched.  The target's decision on the
                # fully accepted prefix is the standard speculative bonus.
                step = (self.target.next_with_logit_summary(confirmed.copy())
                        if self.capture_target_logits else self.target.next(confirmed.copy()))
                bonus = int(step.token_id)
                observation = step.hidden
                observation_logit_summary = step.logit_summary
                committed.append(bonus)
                confirmed.append(bonus)
                generated.append(bonus)
                stopped_on_eos = bonus == self.eos_token_id

            codes_out: tuple[int, ...] | None = None
            if self.encoder is not None and not stopped_on_eos and len(generated) < max_new_tokens:
                # A mismatch supplies a rejected token and correction; full
                # acceptance supplies no rejected token and the bonus.  Both
                # hidden states were obtained at a confirmed prefix.
                assert observation is not None
                answer = correction if rejected is not None else bonus
                assert answer is not None
                codes_out = tuple(
                    self.encoder.encode(
                        observation.detach(),
                        None if rejected is None else self.target.embedding(rejected).detach(),
                        self.target.embedding(answer).detach(),
                        accepted,
                    )
                )
                if len(codes_out) != 8 or any(
                    not isinstance(code, int) or not 0 <= code <= 255
                    for code in codes_out
                ):
                    raise ValueError("feedback encoder must return exactly eight byte indices")

            records.append(
                RoundRecord(
                    prefix_before=prefix_before,
                    codes_in=codes_in,
                    proposals=tuple(proposal),
                    accepted_count=accepted,
                    committed_tokens=tuple(committed),
                    rejected_token=rejected,
                    correction_token=correction,
                    bonus_token=bonus,
                    observation_hidden=None if observation is None else observation.detach(),
                    feedback_codes=codes_out,
                    message_bytes=8 if codes_out is not None else 0,
                    observation_logit_summary=observation_logit_summary,
                )
            )
            codes = codes_out
            if stopped_on_eos:
                break

        return GenerationResult(tokens=tuple(generated), rounds=tuple(records))


def autoregressive_generate(
    target: TargetBackend,
    prompt: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> tuple[int, ...]:
    """Independent greedy target reference, excluding the prompt."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    prefix = list(prompt)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token = int(target.next(prefix.copy()).token_id)
        generated.append(token)
        prefix.append(token)
        if token == eos_token_id:
            break
    return tuple(generated)
