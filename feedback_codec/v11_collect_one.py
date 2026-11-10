"""One-pass v11 train or val prompt collection with independently saved output streams.

This isolates the v10 driver from the frozen v3 collector.  It uses the same
numeric verifier, legal row contract, and provenance gate.  The full packed and
independent AR streams are returned *before* any strict mismatch decision, so
the caller can atomically save the failure without retrying the Target.
"""

from __future__ import annotations

from typing import Any

import torch

from .decoding import autoregressive_generate
from .fixed_probe import ProbeExample, ProbeObservation, ProbePrompt
from .packed import NUMERICAL_RECHECK_BACKEND_V2, PackedSpeculativeDecoder
from .packed_probe_numeric_v3 import (
    NUMERICAL_RECHECK_TOP2_ULPS, PackedProbeCorpus, PackedProbeRow,
    trace_payload, validate_numeric_provenance,
)


def _stream_trace(record: Any, trace: Any, round_index: int) -> dict[str, Any]:
    """Keep real and hypothetical verifier decisions in an audit-only receipt."""
    source = trace.feature_provenance or {}
    return {
        "round_index": round_index,
        "prefix_before": list(record.prefix_before),
        "proposals": list(record.proposals),
        "accepted_count": record.accepted_count,
        "committed_tokens": list(record.committed_tokens),
        "rejected_token": record.rejected_token,
        "correction_token": record.correction_token,
        "bonus_token": record.bonus_token,
        "argmax_ids_full_audit_only": list(trace.argmax_ids),
        "original_packed_argmax_ids_full_audit_only": list(
            trace.original_packed_argmax_ids),
        "original_packed_top2_ids_full_audit_only": [list(pair) for pair in
            trace.original_packed_top2_ids],
        "original_packed_top2_logits_fp32_full_audit_only": [list(pair)
            for pair in trace.original_packed_top2_logits_fp32],
        "original_packed_top2_margins_full_audit_only": list(
            trace.original_packed_top2_margins),
        "numerical_recheck_rows": list(trace.numerical_recheck_rows),
        "numerical_recheck_extra_target_forwards": (
            trace.numerical_recheck_extra_target_forwards),
        "numerical_recheck_snapshot_wall_ms": (
            trace.numerical_recheck_snapshot_wall_ms),
        "numerical_recheck_forward_wall_ms": (
            trace.numerical_recheck_forward_wall_ms),
        "numerical_recheck_total_wall_ms": (
            trace.numerical_recheck_total_wall_ms),
        "feature_source": source.get("source"),
        "feature_provenance_passed": source.get("passed"),
        "decision_backend": trace.decision_backend,
        "cache_commit_source": trace.cache_commit_source,
        "cache_length_before": trace.cache_length_before,
        "cache_length_raw": trace.cache_length_raw,
        "cache_length_after": trace.cache_length_after,
        "valid_prefix": list(trace.valid_prefix),
    }


def collect_one(
    target: Any, draft: Any, prompt: ProbePrompt, eos_token_id: int,
    max_new_tokens: int = 128,
) -> tuple[PackedProbeCorpus, dict[str, Any]]:
    """Collect one frozen train prompt, L8, cap8, preserving both full streams.

    A nonexact request yields an empty corpus and ``receipt['exact']=False``.
    The caller must save the receipt and stop; it must never train on that
    corpus.  Structural/provenance failure after exact stream agreement raises.
    """
    if type(prompt) is not ProbePrompt or prompt.split not in ("train", "val"):
        raise ValueError("v11 collect_one accepts one train or val ProbePrompt")
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("EOS must be a pinned nonnegative integer token ID")
    if type(max_new_tokens) is not int or max_new_tokens != 128:
        raise ValueError("v10 frozen request generation budget is 128 tokens")
    window = 8
    cap = 8
    decoder = PackedSpeculativeDecoder(
        target, draft, None, window, eos_token_id,
        capture_target_logits=True, feature_provenance_gate=True,
        numerical_recheck_top2_ulps=NUMERICAL_RECHECK_TOP2_ULPS,
    )
    online = decoder.generate(list(prompt.input_ids), max_new_tokens)
    oracle = autoregressive_generate(
        target, list(prompt.input_ids), max_new_tokens, eos_token_id)
    tokens = tuple(online.generation.tokens)
    oracle = tuple(oracle)
    if len(online.generation.rounds) != len(online.traces):
        raise RuntimeError("packed records/traces have different lengths")
    packed_eos = bool(tokens and tokens[-1] == eos_token_id)
    oracle_eos = bool(oracle and oracle[-1] == eos_token_id)
    packed_truncated = len(tokens) == max_new_tokens and not packed_eos
    oracle_truncated = len(oracle) == max_new_tokens and not oracle_eos
    first_difference = next(
        (i for i, (a, b) in enumerate(zip(tokens, oracle)) if a != b),
        min(len(tokens), len(oracle)) if len(tokens) != len(oracle) else None,
    )
    exact = (tokens == oracle and packed_eos == oracle_eos
             and packed_truncated == oracle_truncated)
    receipt: dict[str, Any] = {
        "prompt_id": prompt.id, "split": prompt.split,
        "packed_ids": list(tokens), "oracle_ids": list(oracle),
        "packed_eos": packed_eos, "oracle_eos": oracle_eos,
        "packed_truncated": packed_truncated,
        "oracle_truncated": oracle_truncated,
        "first_difference_index": first_difference,
        "exact": exact, "window": window,
        "max_new_tokens": max_new_tokens,
        "max_probe_windows_per_prompt": cap,
        "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
        "numerical_recheck_top2_ulps": NUMERICAL_RECHECK_TOP2_ULPS,
        "rounds": [_stream_trace(record, trace, index)
                   for index, (record, trace) in enumerate(zip(
                       online.generation.rounds, online.traces, strict=True))],
    }
    audit: dict[str, int | float] = {
        "prompts": 1, "exact_requests": int(exact),
        "raw_rounds": len(online.traces), "probe_windows": 0,
        "full_nonterminal_windows": 0,
        "independent_ar_prefix_exact_windows": 0,
        "terminal_or_budget_rounds": 0, "skipped_by_cap": 0,
        "runtime_extra_target_forwards": sum(
            trace.numerical_recheck_extra_target_forwards for trace in online.traces),
        "runtime_numerical_recheck_snapshot_wall_ms": sum(
            trace.numerical_recheck_snapshot_wall_ms for trace in online.traces),
        "runtime_numerical_recheck_forward_wall_ms": sum(
            trace.numerical_recheck_forward_wall_ms for trace in online.traces),
        "runtime_numerical_recheck_total_wall_ms": sum(
            trace.numerical_recheck_total_wall_ms for trace in online.traces),
        "diagnostic_feature_gate_target_forwards": sum(
            (trace.feature_provenance or {}).get(
                "diagnostic_feature_gate_target_forwards", 0)
            for trace in online.traces),
    }
    request_audit = ({
        "prompt_id": prompt.id, "output_ids": list(tokens),
        "eos": packed_eos, "truncated": packed_truncated,
        "verification_rounds": len(online.traces),
        "independent_ar_exact": exact,
        "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
        "numerical_recheck_top2_ulps": NUMERICAL_RECHECK_TOP2_ULPS,
        "runtime_extra_target_forwards": audit["runtime_extra_target_forwards"],
        "runtime_numerical_recheck_snapshot_wall_ms": audit[
            "runtime_numerical_recheck_snapshot_wall_ms"],
        "runtime_numerical_recheck_forward_wall_ms": audit[
            "runtime_numerical_recheck_forward_wall_ms"],
        "runtime_numerical_recheck_total_wall_ms": audit[
            "runtime_numerical_recheck_total_wall_ms"],
        "diagnostic_feature_gate_target_forwards": audit[
            "diagnostic_feature_gate_target_forwards"],
    },)
    if not exact:
        return PackedProbeCorpus((), audit, request_audit), receipt

    rows: list[PackedProbeRow] = []
    collected = 0
    for round_index, (record, trace) in enumerate(zip(
            online.generation.rounds, online.traces, strict=True)):
        if record.codes_in is not None or record.feedback_codes is not None:
            raise RuntimeError("raw Draft collection used feedback codes")
        if (record.prefix_before != trace.prefix_before
                or record.proposals != trace.proposals
                or record.accepted_count != trace.accepted_count):
            raise RuntimeError("packed round/trace alignment failed")
        if collected >= cap:
            audit["skipped_by_cap"] += 1
            continue
        answer = (record.correction_token if record.rejected_token is not None
                  else record.bonus_token)
        if answer is None:
            audit["terminal_or_budget_rounds"] += 1
            continue
        accepted = record.accepted_count
        valid = record.prefix_before + record.proposals[:accepted]
        corrected = valid + (answer,)
        if (trace.valid_prefix != valid
                or record.committed_tokens
                   != record.proposals[:accepted] + (answer,)
                or corrected != record.prefix_before + record.committed_tokens
                or trace.argmax_ids[accepted] != answer
                or trace.cache_length_after != len(corrected) - 1):
            raise RuntimeError("packed valid boundary or KV was misaligned")
        if record.rejected_token is not None and (
                accepted >= len(record.proposals)
                or record.proposals[accepted] != record.rejected_token
                or record.proposals[accepted] == answer):
            raise RuntimeError("first rejection was misrecorded")
        if record.rejected_token is None and (
                len(record.proposals) != window or accepted != window
                or record.bonus_token != answer):
            raise RuntimeError("full-accept window was not full L8")
        validate_numeric_provenance(trace)
        summary = record.observation_logit_summary
        hidden = record.observation_hidden
        if (hidden is None or hidden.ndim != 1 or not torch.isfinite(hidden).all()
                or summary is None or summary.greedy_token_id != answer):
            raise RuntimeError("same-forward valid Target feature absent")
        used = len(corrected) - len(prompt.input_ids)
        if corrected != prompt.input_ids + oracle[:used]:
            raise RuntimeError("corrected prefix is not on independent AR path")
        remaining = max_new_tokens - used
        if answer == eos_token_id or remaining <= 0:
            audit["terminal_or_budget_rounds"] += 1
            continue
        future = oracle[used:used + min(window, remaining)]
        if not future:
            audit["terminal_or_budget_rounds"] += 1
            continue
        observation = ProbeObservation(
            prompt_id=prompt.id, family_id=prompt.family_id,
            split=prompt.split, task_type=prompt.task_type,
            round_index=round_index, source_prefix=record.prefix_before,
            proposals=record.proposals, valid_prefix=valid,
            accepted_count=accepted, rejected_token=record.rejected_token,
            correction_token=answer, corrected_prefix=corrected,
            target_hidden=hidden.detach().cpu().clone(),
            target_hidden_prefix=valid, remaining_budget=remaining,
            window=window, target_logit_summary=summary,
        )
        rows.append(PackedProbeRow(
            ProbeExample(observation, future), trace_payload(trace)))
        collected += 1
        audit["probe_windows"] += 1
        audit["independent_ar_prefix_exact_windows"] += 1
        if len(future) == window and eos_token_id not in future:
            audit["full_nonterminal_windows"] += 1
    return PackedProbeCorpus(tuple(rows), audit, request_audit), receipt
