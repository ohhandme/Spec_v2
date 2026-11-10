"""Versioned P1 high-bandwidth closed loop using the frozen v9c verifier.

This is a wiring diagnostic.  The head consumes a legal preceding Target
decision feature through an explicit prepared state, never eight-byte codes.
The target verifier, BF16 2-ULP policy, and independent AR remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable

import torch

from .decoding import autoregressive_generate
from .fixed_probe import ProbeObservation, ProbePrompt
from .packed import HFPackedTargetSession, NUMERICAL_RECHECK_BACKEND_V2, PackedRoundTrace
from .packed_probe_numeric_v3 import trace_payload, validate_numeric_provenance


@dataclass(frozen=True)
class V10ClosedLoopResult:
    prompt_id: str
    arm: str
    prompt_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    oracle_ids: tuple[int, ...]
    eos: bool
    truncated: bool
    oracle_eos: bool
    oracle_truncated: bool
    exact: bool
    first_difference: int | None
    stopped_on_first_difference: bool
    rounds: tuple[dict[str, Any], ...]
    runtime_extra_target_forwards: int
    diagnostic_feature_gate_target_forwards: int
    runtime_snapshot_ms: float
    runtime_recheck_forward_ms: float
    runtime_recheck_total_ms: float

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id, "arm": self.arm,
            "prompt_ids": list(self.prompt_ids),
            "output_ids": list(self.output_ids), "oracle_ids": list(self.oracle_ids),
            "eos": self.eos, "truncated": self.truncated,
            "oracle_eos": self.oracle_eos,
            "oracle_truncated": self.oracle_truncated,
            "exact": self.exact, "first_difference": self.first_difference,
            "stopped_on_first_difference": self.stopped_on_first_difference,
            "rounds": list(self.rounds),
            "normal_packed_verify_forwards": len(self.rounds),
            "runtime_extra_target_forwards": self.runtime_extra_target_forwards,
            "diagnostic_feature_gate_target_forwards":
                self.diagnostic_feature_gate_target_forwards,
            "runtime_snapshot_ms": self.runtime_snapshot_ms,
            "runtime_recheck_forward_ms": self.runtime_recheck_forward_ms,
            "runtime_recheck_total_ms": self.runtime_recheck_total_ms,
            "guard_cost_subtotals_overlap_total": True,
        }


def _first_difference(left: tuple[int, ...], right: tuple[int, ...]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _full_trace(verification: Any, accepted: int, cache_after: int,
                feature_provenance: dict[str, Any] | None) -> PackedRoundTrace:
    """Preserve the same row accounting as the frozen packed decoder."""
    return PackedRoundTrace(
        prefix_before=verification.prefix_before,
        proposals=verification.proposals,
        argmax_ids=verification.argmax_ids,
        top2_margins=verification.top2_margins,
        accepted_count=accepted,
        cache_length_before=verification.cache_length_before,
        cache_length_raw=verification.cache_length_raw,
        cache_length_after=cache_after,
        valid_prefix=verification.prefix_before + verification.proposals[:accepted],
        row_source_query_lengths=verification.row_source_query_lengths,
        row_source_modes=verification.row_source_modes,
        feature_provenance=feature_provenance,
        decision_backend=NUMERICAL_RECHECK_BACKEND_V2,
        numerical_recheck_rows=verification.numerical_recheck_rows,
        numerical_recheck_extra_target_forwards=(
            verification.numerical_recheck_extra_target_forwards),
        numerical_recheck_snapshot_wall_ms=(
            verification.numerical_recheck_snapshot_wall_ms),
        numerical_recheck_forward_wall_ms=(
            verification.numerical_recheck_forward_wall_ms),
        numerical_recheck_total_wall_ms=(
            verification.numerical_recheck_total_wall_ms),
        unrechecked_low_margin_bonus_rows=(
            verification.unrechecked_low_margin_bonus_rows),
        cache_commit_source=(
            verification.decision_cache_source
            if accepted == verification.decision_cache_row_index
            else "original_packed"),
        original_packed_top2_margins=(
            verification.original_packed_top2_margins),
        original_packed_argmax_ids=(verification.original_packed_argmax_ids),
        original_packed_top2_ids=(verification.original_packed_top2_ids),
        original_packed_top2_logits_fp32=(
            verification.original_packed_top2_logits_fp32),
        effective_top2_ids=verification.effective_top2_ids,
        effective_top2_logits_fp32=verification.effective_top2_logits_fp32,
        numerical_recheck_ulp_units=(
            verification.numerical_recheck_ulp_units),
        numerical_recheck_ulp_thresholds=(
            verification.numerical_recheck_ulp_thresholds),
        numerical_recheck_trigger_reasons=(
            verification.numerical_recheck_trigger_reasons),
    )


@torch.inference_mode()
def run_v10_closed_loop(*, target: Any, draft: Any, head: Any,
                        prompt: ProbePrompt, eos_token_id: int,
                        max_new_tokens: int = 32,
                        progress_hook: Callable[[dict[str, Any]], None] | None = None,
                        ) -> V10ClosedLoopResult:
    """Run one fixed train request, returning both complete streams and traces.

    The caller must persist the returned receipt before enforcing ``exact``.
    ``progress_hook`` can atomically save partial evidence on a raised runtime
    error.  No future AR token enters ``prepare_observation`` or the head.
    """
    if prompt.split != "train" or max_new_tokens != 32 or type(eos_token_id) is not int:
        raise ValueError("v10 closed loop requires one train prompt and 32-token cap")
    if not prompt.input_ids:
        raise ValueError("empty train prompt")
    oracle = tuple(autoregressive_generate(
        target, list(prompt.input_ids), max_new_tokens, eos_token_id))
    if progress_hook is not None:
        progress_hook({"stage": "oracle_ready", "prompt_id": prompt.id,
                       "oracle_ids": list(oracle)})
    session = HFPackedTargetSession(
        target, list(prompt.input_ids), capture_online_cache_snapshot=True,
        numerical_recheck_top2_ulps=2)
    generated: list[int] = []
    rounds: list[dict[str, Any]] = []
    prepared_state: Any | None = None
    extra = diagnostic = 0
    snapshot_ms = forward_ms = total_ms = 0.0
    diverged = False
    while len(generated) < max_new_tokens:
        prefix_before = tuple(session.confirmed)
        remaining = max_new_tokens - len(generated)
        proposal: list[int] = []
        for position in range(1, min(8, remaining) + 1):
            current = session.confirmed + proposal
            if prepared_state is None:
                token = int(draft.next(current, None, position))
            else:
                token = int(head.next_token(current, prepared_state, position, draft))
            proposal.append(token)
            if token == eos_token_id:
                break
        bonus_needed = proposal[-1] != eos_token_id and len(proposal) < remaining
        verification = session.verify(proposal, bonus_needed=bonus_needed)
        accepted = 0
        while (accepted < len(proposal)
               and proposal[accepted] == verification.argmax_ids[accepted]):
            accepted += 1
        feature_provenance = session.check_online_suffix_noninterference(
            verification, accepted)
        rejected = proposal[accepted] if accepted < len(proposal) else None
        correction = verification.argmax_ids[accepted] if rejected is not None else None
        draft_eos = accepted == len(proposal) and proposal[-1] == eos_token_id
        has_budget_for_bonus = len(generated) + accepted < max_new_tokens
        bonus = (verification.argmax_ids[-1]
                 if rejected is None and not draft_eos and has_budget_for_bonus else None)
        answer = correction if correction is not None else bonus
        committed = proposal[:accepted]
        if answer is not None:
            committed = committed + [answer]
        if not committed:
            raise RuntimeError("packed v10 round made no progress")
        generated.extend(committed)
        cache_after = session.commit(verification, accepted, answer)
        stopped_on_eos = generated[-1] == eos_token_id
        hidden = (verification.hidden_at_valid_prefix(accepted)
                  if answer is not None else None)
        summary = (verification.logit_summary_at_valid_prefix(accepted)
                   if answer is not None else None)
        if summary is not None and summary.greedy_token_id != answer:
            raise RuntimeError("valid-prefix Target summary predicts wrong answer")
        feature_provenance = {
            **feature_provenance,
            "decision_feature_available": answer is not None,
            "same_forward_hidden": (
                torch.equal(hidden, verification.hidden_at_valid_prefix(accepted))
                if hidden is not None else None),
            "same_forward_logit_summary": (
                summary == verification.logit_summary_at_valid_prefix(accepted)
                if summary is not None else None),
        }
        trace = _full_trace(verification, accepted, cache_after, feature_provenance)
        if answer is not None:
            validate_numeric_provenance(trace)
        trace_json = trace_payload(trace) if answer is not None else None
        used = len(generated)
        record = {
            "round_index": len(rounds), "prefix_before": list(prefix_before),
            "proposals": list(proposal), "argmax_ids": list(verification.argmax_ids),
            "accepted_count": accepted, "rejected_token": rejected,
            "correction_token": correction, "bonus_token": bonus,
            "committed_tokens": list(committed),
            "used_tokens": used, "cache_length_after": cache_after,
            "row_source_modes": list(verification.row_source_modes),
            "row_source_query_lengths": list(verification.row_source_query_lengths),
            "valid_prefix": list(trace.valid_prefix),
            "feature_source": feature_provenance.get("source"),
            "feature_provenance": feature_provenance,
            "valid_hidden_sha256": _tensor_sha(hidden) if hidden is not None else None,
            "valid_logit_sha256": _tensor_sha(
                verification.logits_at_valid_prefix(accepted))
                if answer is not None else None,
            "trace": trace_json,
        }
        rounds.append(record)
        extra += verification.numerical_recheck_extra_target_forwards
        diagnostic += feature_provenance.get(
            "diagnostic_feature_gate_target_forwards", 0)
        snapshot_ms += verification.numerical_recheck_snapshot_wall_ms
        forward_ms += verification.numerical_recheck_forward_wall_ms
        total_ms += verification.numerical_recheck_total_wall_ms
        if progress_hook is not None:
            progress_hook({"stage": "round_complete", "prompt_id": prompt.id,
                           "oracle_ids": list(oracle), "output_ids": list(generated),
                           "rounds": rounds})
        if tuple(generated) != oracle[:len(generated)]:
            diverged = True
            break
        if stopped_on_eos:
            break
        if len(generated) < max_new_tokens:
            if answer is None or hidden is None or summary is None:
                raise RuntimeError("nonterminal round has no legal Target feature")
            valid = prefix_before + tuple(proposal[:accepted])
            corrected = valid + (answer,)
            observation = ProbeObservation(
                prompt_id=prompt.id, family_id=prompt.family_id,
                split="train", task_type=prompt.task_type,
                round_index=len(rounds) - 1,
                source_prefix=prefix_before, proposals=tuple(proposal),
                valid_prefix=valid, accepted_count=accepted,
                rejected_token=rejected, correction_token=answer,
                corrected_prefix=corrected,
                target_hidden=hidden.detach().cpu().clone(),
                target_hidden_prefix=valid,
                remaining_budget=max_new_tokens - len(generated),
                window=8, target_logit_summary=summary)
            prepared_state = head.prepare_observation(observation, draft)
    output = tuple(generated)
    eos = bool(output and output[-1] == eos_token_id)
    oracle_eos = bool(oracle and oracle[-1] == eos_token_id)
    return V10ClosedLoopResult(
        prompt_id=prompt.id, arm=getattr(head, "mode", type(head).__name__),
        prompt_ids=prompt.input_ids, output_ids=output, oracle_ids=oracle,
        eos=eos, truncated=len(output) == max_new_tokens and not eos,
        oracle_eos=oracle_eos,
        oracle_truncated=len(oracle) == max_new_tokens and not oracle_eos,
        exact=(output == oracle and eos == oracle_eos and
               (len(output) == max_new_tokens and not eos) ==
               (len(oracle) == max_new_tokens and not oracle_eos)),
        first_difference=_first_difference(output, oracle),
        stopped_on_first_difference=diverged,
        rounds=tuple(rounds), runtime_extra_target_forwards=extra,
        diagnostic_feature_gate_target_forwards=diagnostic,
        runtime_snapshot_ms=snapshot_ms,
        runtime_recheck_forward_ms=forward_ms,
        runtime_recheck_total_ms=total_ms,
    )
