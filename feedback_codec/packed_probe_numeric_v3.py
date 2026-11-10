"""Collect legal probes from the costed canonical BF16 2-ULP recheck Target.

This module is separate from the frozen v2 numeric collector and its plan.
Every prompt still requires the whole closed-loop request to match independent
full-prefix Target AR, including EOS and truncation. Features are from the
effective *online* valid decision row; if a BF16 row was numerically rechecked,
the extra Target work and changed cache source are explicit. Independent AR
supplies labels and a correctness oracle, never an encoder input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable

import torch

from .decoding import autoregressive_generate
from .fixed_probe import ProbeExample, ProbeObservation, ProbePrompt, _validate_prompts
from .packed import (PACKED_FEATURE_PROTOCOL_VERSION, NUMERICAL_RECHECK_BACKEND_V2,
                     PackedSpeculativeDecoder)


PACKED_PROBE_PROTOCOL_VERSION = "packed_fixed_prefix_p1_canonical_ulp_recheck_v3"
NUMERICAL_RECHECK_TOP2_ULPS = 2
FEATURE_SOURCES = {
    "original_packed_verify_forward": "same_online_preverify_KV_snapshot",
    "numeric_recheck_query_forward": "same_online_preverify_KV_snapshot",
    "canonical_full_prefix_numeric_recheck": "fresh_full_prefix_no_KV",
}


def validate_numeric_provenance(trace: Any) -> dict[str, Any]:
    """Fail closed on a valid row with missing, free, or future information."""
    accepted = trace.accepted_count
    source = getattr(trace, "feature_provenance", None)
    decisions = getattr(trace, "argmax_ids", None)
    margins = getattr(trace, "original_packed_top2_margins", None)
    thresholds = getattr(trace, "numerical_recheck_ulp_thresholds", None)
    if decisions is None:
        decisions = getattr(trace, "used_argmax_ids", None)
        margins = getattr(trace, "used_original_packed_top2_margins", None)
        thresholds = getattr(trace, "used_numerical_recheck_ulp_thresholds", None)
    if (getattr(trace, "decision_backend", None) != NUMERICAL_RECHECK_BACKEND_V2
            or not isinstance(source, dict)
            or source.get("feature_protocol_version") != PACKED_FEATURE_PROTOCOL_VERSION
            or source.get("numeric_backend") != NUMERICAL_RECHECK_BACKEND_V2
            or source.get("source") not in FEATURE_SOURCES
            or source.get("cache_source") != FEATURE_SOURCES.get(source.get("source"))
            or source.get("accepted_count") != accepted
            or source.get("proposal_count") != len(trace.proposals)
            or source.get("passed") is not True
            or source.get("decision_feature_available") is not True
            or source.get("same_forward_hidden") is not True
            or source.get("same_forward_logit_summary") is not True
            or source.get("numerical_recheck_extra_target_forwards")
               != trace.numerical_recheck_extra_target_forwards
            or source.get("numerical_recheck_rows") != list(trace.numerical_recheck_rows)
            or source.get("diagnostic_feature_gate_target_forwards", -1) < 1
            or trace.unrechecked_low_margin_bonus_rows
            or not isinstance(decisions, (tuple, list))
            or not isinstance(margins, (tuple, list))
            or not isinstance(thresholds, (tuple, list))
            or len(margins) != len(decisions)
            or len(thresholds) != len(decisions)
            or any(not isinstance(value, (int, float))
                   or not math.isfinite(float(value))
                   for value in (*margins, *thresholds))
            or any(value < 0 for value in (*margins, *thresholds))
            or trace.numerical_recheck_extra_target_forwards
               != 2 * len(trace.numerical_recheck_rows)):
        raise RuntimeError("numeric valid feature provenance is incomplete")
    name = source["source"]
    if name == "original_packed_verify_forward":
        if (accepted in trace.numerical_recheck_rows
                or trace.cache_commit_source != "original_packed"
                or source.get("snapshot_replay_hidden_bitwise_equal") is not True
                or source.get("snapshot_replay_logits_bitwise_equal") is not True
                or source.get("snapshot_replay_greedy_token_equal") is not True):
            raise RuntimeError("original packed valid row lacks online-KV proof")
    else:
        if (accepted not in trace.numerical_recheck_rows
                or source.get("feature_replay_hidden_bitwise_equal") is not True
                or source.get("feature_replay_logits_bitwise_equal") is not True
                or source.get("feature_replay_greedy_token_equal") is not True
                or trace.numerical_recheck_extra_target_forwards < 1):
            raise RuntimeError("costed numeric valid row was not replayed")
        if name == "numeric_recheck_query_forward":
            raise RuntimeError("v3 canonical ULP policy cannot use online short recheck")
        elif (source.get("reason") != "canonical_full_prefix_ends_at_valid_row"
              or source.get("valid_row_query_length")
                 != len(trace.prefix_before) + accepted
              or trace.cache_commit_source != "canonical_full_prefix_refill"
              or trace.numerical_recheck_extra_target_forwards < 2):
            raise RuntimeError("canonical valid row lacks charged full-prefix refill")
    return source


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def trace_payload(trace: Any) -> dict[str, Any]:
    """Keep decisions through the valid row, never hypothetical suffix logits.

    A packed forward computes rows beyond the first rejection.  Those rows do
    not belong to the confirmed trajectory and must not enter a training row.
    The full trace remains available in the separate P0 correctness report.
    """
    used_rows = trace.accepted_count + 1
    validate_numeric_provenance(trace)
    provenance_keys = (
        "feature_protocol_version", "source", "cache_source", "passed",
        "decision_feature_available", "same_forward_hidden",
        "same_forward_logit_summary",
        "snapshot_replay_hidden_bitwise_equal",
        "snapshot_replay_logits_bitwise_equal", "snapshot_replay_greedy_token_equal",
        "feature_replay_hidden_bitwise_equal", "feature_replay_logits_bitwise_equal",
        "feature_replay_greedy_token_equal", "applicable", "reason",
        "valid_row_query_length", "numeric_backend",
        "accepted_count", "proposal_count",
        "numerical_recheck_rows", "numerical_recheck_extra_target_forwards",
        "numerical_recheck_snapshot_wall_ms",
        "numerical_recheck_forward_wall_ms", "numerical_recheck_total_wall_ms",
        "diagnostic_feature_gate_target_forwards",
    )
    # Intervention diagnostics can involve extra counterfactual forwards.
    # Retain only the boolean gate, not their outputs or numerical distances.
    provenance = {key: trace.feature_provenance.get(key)
                  for key in provenance_keys}
    return {
        "prefix_before": list(trace.prefix_before),
        "proposals": list(trace.proposals),
        "used_argmax_ids": list(trace.argmax_ids[:used_rows]),
        "used_original_packed_argmax_ids": list(
            trace.original_packed_argmax_ids[:used_rows]),
        "used_original_packed_top2_ids": [list(pair) for pair in
            trace.original_packed_top2_ids[:used_rows]],
        "used_original_packed_top2_logits_fp32": [list(pair) for pair in
            trace.original_packed_top2_logits_fp32[:used_rows]],
        "accepted_count": trace.accepted_count,
        "cache_length_before": trace.cache_length_before,
        "cache_length_raw": trace.cache_length_raw,
        "cache_length_after": trace.cache_length_after,
        "valid_prefix": list(trace.valid_prefix),
        "feature_provenance": provenance,
        "decision_backend": trace.decision_backend,
        "cache_commit_source": trace.cache_commit_source,
        "numerical_recheck_rows": list(trace.numerical_recheck_rows),
        "used_original_packed_top2_margins": list(
            trace.original_packed_top2_margins[:used_rows]),
        "used_numerical_recheck_ulp_units": list(
            trace.numerical_recheck_ulp_units[:used_rows]),
        "used_numerical_recheck_ulp_thresholds": list(
            trace.numerical_recheck_ulp_thresholds[:used_rows]),
        "used_numerical_recheck_trigger_reasons": list(
            trace.numerical_recheck_trigger_reasons[:used_rows]),
        "numerical_recheck_extra_target_forwards": (
            trace.numerical_recheck_extra_target_forwards),
        "numerical_recheck_snapshot_wall_ms": (
            trace.numerical_recheck_snapshot_wall_ms),
        "numerical_recheck_forward_wall_ms": (
            trace.numerical_recheck_forward_wall_ms),
        "numerical_recheck_total_wall_ms": (
            trace.numerical_recheck_total_wall_ms),
        "unrechecked_low_margin_bonus_rows": (
            list(trace.unrechecked_low_margin_bonus_rows)),
    }


@dataclass(frozen=True)
class PackedProbeRow:
    example: ProbeExample
    trace: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            **self.example.to_json(),
            "collection_protocol_version": PACKED_PROBE_PROTOCOL_VERSION,
            "packed_feature_protocol_version": PACKED_FEATURE_PROTOCOL_VERSION,
            "packed_feature_source": self.trace["feature_provenance"]["source"],
            "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
            "numerical_recheck_top2_ulps": NUMERICAL_RECHECK_TOP2_ULPS,
            "online_feature_provenance_passed": True,
            "independent_ar_prefix_exact": True,
            "packed_trace_sha256": hashlib.sha256(canonical_bytes(self.trace)).hexdigest(),
            "packed_trace": self.trace,
        }


@dataclass(frozen=True)
class PackedProbeCorpus:
    rows: tuple[PackedProbeRow, ...]
    audit: dict[str, int | float]
    request_audit: tuple[dict[str, Any], ...]


def collect_packed_fixed_probe_numeric_v3(
    *, target: Any, raw_draft: Any, prompts: Iterable[ProbePrompt],
    window: int = 8, max_new_tokens: int = 128,
    eos_token_id: int | None = None, max_probe_windows_per_prompt: int = 8,
) -> PackedProbeCorpus:
    """Collect at most eight legal raw-Draft windows per original prompt.

    All arms will later consume the same corrected prefixes, so no trained arm
    can influence which rows are collected.  This function never constructs an
    encoder or exposes independent AR future tokens as features.
    """
    selected = list(prompts)
    _validate_prompts(selected)
    if any(prompt.split not in ("train", "val") for prompt in selected):
        raise ValueError("packed P1 collection accepts train/val only")
    if window != 8 or max_new_tokens < window or max_probe_windows_per_prompt != 8:
        raise ValueError("frozen pilot requires L8, budget >=8 and eight-window cap")
    if eos_token_id is None or type(eos_token_id) is not int:
        raise ValueError("real packed collection requires a pinned integer EOS")
    decoder = PackedSpeculativeDecoder(
        target, raw_draft, None, window, eos_token_id,
        capture_target_logits=True, feature_provenance_gate=True,
        numerical_recheck_top2_ulps=NUMERICAL_RECHECK_TOP2_ULPS,
    )
    rows: list[PackedProbeRow] = []
    request_audit: list[dict[str, Any]] = []
    audit = {"prompts": len(selected), "exact_requests": 0, "raw_rounds": 0,
             "probe_windows": 0, "full_nonterminal_windows": 0,
             "independent_ar_prefix_exact_windows": 0,
             "terminal_or_budget_rounds": 0, "skipped_by_cap": 0,
             "runtime_extra_target_forwards": 0,
             "runtime_numerical_recheck_snapshot_wall_ms": 0.0,
             "runtime_numerical_recheck_forward_wall_ms": 0.0,
             "runtime_numerical_recheck_total_wall_ms": 0.0,
             "diagnostic_feature_gate_target_forwards": 0}
    for prompt in selected:
        online = decoder.generate(list(prompt.input_ids), max_new_tokens)
        oracle = autoregressive_generate(
            target, list(prompt.input_ids), max_new_tokens, eos_token_id)
        tokens = online.generation.tokens
        online_eos = bool(tokens and tokens[-1] == eos_token_id)
        oracle_eos = bool(oracle and oracle[-1] == eos_token_id)
        online_truncated = len(tokens) == max_new_tokens and not online_eos
        oracle_truncated = len(oracle) == max_new_tokens and not oracle_eos
        if (tokens != oracle or online_eos != oracle_eos
                or online_truncated != oracle_truncated):
            raise RuntimeError(f"packed raw Draft differs from independent AR: {prompt.id}")
        if len(online.generation.rounds) != len(online.traces):
            raise RuntimeError("packed records/traces have different lengths")
        audit["exact_requests"] += 1
        audit["raw_rounds"] += len(online.traces)
        runtime_forwards = sum(trace.numerical_recheck_extra_target_forwards
                               for trace in online.traces)
        runtime_forward_ms = sum(trace.numerical_recheck_forward_wall_ms
                                 for trace in online.traces)
        runtime_snapshot_ms = sum(trace.numerical_recheck_snapshot_wall_ms
                                  for trace in online.traces)
        runtime_total_ms = sum(trace.numerical_recheck_total_wall_ms
                               for trace in online.traces)
        diagnostic_forwards = sum(
            (trace.feature_provenance or {}).get(
                "diagnostic_feature_gate_target_forwards", 0)
            for trace in online.traces)
        audit["runtime_extra_target_forwards"] += runtime_forwards
        audit["runtime_numerical_recheck_snapshot_wall_ms"] += runtime_snapshot_ms
        audit["runtime_numerical_recheck_forward_wall_ms"] += runtime_forward_ms
        audit["runtime_numerical_recheck_total_wall_ms"] += runtime_total_ms
        audit["diagnostic_feature_gate_target_forwards"] += diagnostic_forwards
        request_audit.append({
            "prompt_id": prompt.id, "output_ids": list(tokens),
            "eos": online_eos, "truncated": online_truncated,
            "verification_rounds": len(online.traces),
            "independent_ar_exact": True,
            "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
            "numerical_recheck_top2_ulps": NUMERICAL_RECHECK_TOP2_ULPS,
            "runtime_extra_target_forwards": runtime_forwards,
            "runtime_numerical_recheck_snapshot_wall_ms": runtime_snapshot_ms,
            "runtime_numerical_recheck_forward_wall_ms": runtime_forward_ms,
            "runtime_numerical_recheck_total_wall_ms": runtime_total_ms,
            "diagnostic_feature_gate_target_forwards": diagnostic_forwards,
        })
        collected = 0
        for round_index, (record, trace) in enumerate(
                zip(online.generation.rounds, online.traces, strict=True)):
            if record.codes_in is not None or record.feedback_codes is not None:
                raise RuntimeError("raw Draft collection unexpectedly used a feedback code")
            if record.prefix_before != trace.prefix_before or (
                    record.proposals != trace.proposals
                    or record.accepted_count != trace.accepted_count):
                raise RuntimeError("packed round/trace alignment failed")
            if collected >= max_probe_windows_per_prompt:
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
                    or record.committed_tokens != record.proposals[:accepted] + (answer,)
                    or corrected != record.prefix_before + record.committed_tokens
                    or trace.argmax_ids[accepted] != answer
                    or trace.cache_length_after != len(corrected) - 1):
                raise RuntimeError("packed valid boundary, answer, or KV was misaligned")
            if record.rejected_token is not None and (
                    accepted >= len(record.proposals)
                    or record.proposals[accepted] != record.rejected_token
                    or record.proposals[accepted] == answer):
                raise RuntimeError("first rejection was not recorded correctly")
            if record.rejected_token is None and (
                    len(record.proposals) != window
                    or accepted != window or record.bonus_token != answer):
                raise RuntimeError("nonterminal full-accept probe is not a full L8 window")
            provenance = validate_numeric_provenance(trace)
            summary = record.observation_logit_summary
            hidden = record.observation_hidden
            if (hidden is None or hidden.ndim != 1 or not torch.isfinite(hidden).all()
                    or summary is None or summary.greedy_token_id != answer):
                raise RuntimeError("same-forward packed valid feature is absent")
            used = len(corrected) - len(prompt.input_ids)
            if corrected != prompt.input_ids + oracle[:used]:
                raise RuntimeError("corrected prefix is not on the independent AR path")
            remaining = max_new_tokens - used
            if answer == eos_token_id or remaining <= 0:
                audit["terminal_or_budget_rounds"] += 1
                continue
            future = oracle[used:used + min(window, remaining)]
            if not future:
                audit["terminal_or_budget_rounds"] += 1
                continue
            observation = ProbeObservation(
                prompt_id=prompt.id, family_id=prompt.family_id, split=prompt.split,
                task_type=prompt.task_type, round_index=round_index,
                source_prefix=record.prefix_before, proposals=record.proposals,
                valid_prefix=valid, accepted_count=accepted,
                rejected_token=record.rejected_token, correction_token=answer,
                corrected_prefix=corrected,
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
    return PackedProbeCorpus(tuple(rows), audit, tuple(request_audit))
