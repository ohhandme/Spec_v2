"""One real v12 raw-policy request, fail closed before accepted manifest."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .decoding import autoregressive_generate
from .fixed_probe import ProbePrompt
from .packed import (NUMERICAL_RECHECK_BACKEND_V2,
                     PackedSpeculativeDecoder)
from .packed_probe_numeric_v3 import (NUMERICAL_RECHECK_TOP2_ULPS,
                                      validate_numeric_provenance)
from .v11_collect_one import _stream_trace
from .v12_draft_observer import RawDraftForwardObserver
from .v12_oracle_observer import TargetARSummaryObserver
from .v12_round_assembly import request_id
from .v12_storage import _atomic_bytes, _json_bytes, persist_request


@torch.inference_mode()
def collect_v12_request(
    *, target: Any, draft: Any, prompt: ProbePrompt, root: Path,
    eos_token_id: int, dataset_lock_sha256: str,
    source_sha256: str, schema_lock_sha256: str,
    provenance: dict[str, Any],
    policy_id: str = "raw_draft_v9c_ulp2",
    execution_id: str = "v12_full_pilot_1",
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    if (type(prompt) is not ProbePrompt or prompt.split not in ("train", "val")
            or root.name != prompt.split or max_new_tokens != 128):
        raise ValueError("v12 request split or fixed generation budget differs")
    rid = request_id(dataset_lock_sha256, prompt.split, prompt.id,
                     policy_id, execution_id)
    if (root / "manifests" / f"{rid}.json").exists():
        raise FileExistsError("complete v12 request already exists; do not rerun")
    observed_draft = RawDraftForwardObserver(draft)
    observed_target = TargetARSummaryObserver(target)
    begin = time.perf_counter()
    online = PackedSpeculativeDecoder(
        observed_target, observed_draft, None, 8, eos_token_id,
        capture_target_logits=True, feature_provenance_gate=True,
        numerical_recheck_top2_ulps=NUMERICAL_RECHECK_TOP2_ULPS,
    ).generate(list(prompt.input_ids), max_new_tokens)
    packed_elapsed = time.perf_counter() - begin
    begin = time.perf_counter()
    oracle = tuple(autoregressive_generate(
        observed_target, list(prompt.input_ids), max_new_tokens,
        eos_token_id))
    oracle_elapsed = time.perf_counter() - begin
    tokens = tuple(online.generation.tokens)
    packed_eos = bool(tokens and tokens[-1] == eos_token_id)
    oracle_eos = bool(oracle and oracle[-1] == eos_token_id)
    packed_truncated = len(tokens) == max_new_tokens and not packed_eos
    oracle_truncated = len(oracle) == max_new_tokens and not oracle_eos
    exact = (tokens == oracle and packed_eos == oracle_eos
             and packed_truncated == oracle_truncated)
    if len(online.generation.rounds) != len(online.traces):
        raise RuntimeError("real packed record/trace count differs")
    rounds = [_stream_trace(record, trace, i)
              for i, (record, trace) in enumerate(zip(
                  online.generation.rounds, online.traces, strict=True))]
    audit = {
        "request_id": rid, "prompt_id": prompt.id,
        "split": prompt.split, "policy_id": policy_id,
        "execution_id": execution_id, "exact": exact,
        "packed_ids": list(tokens), "oracle_ids": list(oracle),
        "packed_eos": packed_eos, "oracle_eos": oracle_eos,
        "packed_truncated": packed_truncated,
        "oracle_truncated": oracle_truncated,
        "first_difference_index": next(
            (i for i, (a, b) in enumerate(zip(tokens, oracle)) if a != b),
            min(len(tokens), len(oracle)) if len(tokens) != len(oracle) else None),
        "rounds": rounds,
        "draft_actual_forward_count": observed_draft.actual_forwards_observed,
        "draft_proposals_count": sum(len(r.proposals)
                                      for r in online.generation.rounds),
        "oracle_actual_forward_count": observed_target.actual_forwards_observed,
        "packed_wall_seconds": packed_elapsed,
        "oracle_wall_seconds": oracle_elapsed,
        "numerical_backend": NUMERICAL_RECHECK_BACKEND_V2,
        "numerical_recheck_top2_ulps": NUMERICAL_RECHECK_TOP2_ULPS,
    }
    if not exact:
        failure = root / "audit" / "failures" / f"{rid}.json"
        _atomic_bytes(failure, _json_bytes(audit))
        raise RuntimeError(f"v12 packed/independent AR mismatch: {prompt.id}")
    if (observed_draft.actual_forwards_observed
            != audit["draft_proposals_count"]
            or observed_target.actual_forwards_observed != len(oracle)):
        raise RuntimeError("observer changed or missed actual proposal/AR calls")
    for round_index, (record, trace) in enumerate(zip(
            online.generation.rounds, online.traces, strict=True)):
        has_answer = record.correction_token is not None or record.bonus_token is not None
        if has_answer:
            try:
                validate_numeric_provenance(trace)
            except Exception:
                failure = root / "audit" / "failures" / f"{rid}.json"
                _atomic_bytes(failure, _json_bytes({
                    **audit, "failed_round_index": round_index,
                    "failed_round_has_answer": True,
                    "failed_round_feature_provenance": trace.feature_provenance,
                    "failed_round_numerical_recheck_rows": list(trace.numerical_recheck_rows),
                    "failed_round_numerical_recheck_extra_target_forwards":
                        trace.numerical_recheck_extra_target_forwards,
                }))
                raise
        else:
            source = trace.feature_provenance or {}
            if (record.rejected_token is not None
                    or source.get("decision_feature_available") is not False
                    or source.get("same_forward_hidden") is not None
                    or source.get("same_forward_logit_summary") is not None):
                raise RuntimeError("terminal no-answer row has invalid feature status")
        if (record.prefix_before != trace.prefix_before
                or record.proposals != trace.proposals
                or record.accepted_count != trace.accepted_count
                or record.codes_in is not None
                or record.feedback_codes is not None):
            raise RuntimeError("raw-policy round/trace provenance differs")
        b = (record.correction_token if record.rejected_token is not None
             else record.bonus_token)
        if b is not None:
            if (record.observation_hidden is None
                    or not torch.isfinite(record.observation_hidden).all()
                    or record.observation_logit_summary is None
                    or record.observation_logit_summary.greedy_token_id != b):
                raise RuntimeError("actual same-boundary Target feature absent")
    begin = time.perf_counter()
    result = persist_request(
        root=root, prompt=prompt, records=online.generation.rounds,
        draft_steps=observed_draft.steps,
        oracle_tokens=oracle, oracle_steps=observed_target.steps,
        eos_token_id=eos_token_id, dataset_lock_sha256=dataset_lock_sha256,
        policy_id=policy_id, execution_id=execution_id,
        source_sha256=source_sha256, schema_lock_sha256=schema_lock_sha256,
        provenance=provenance,
        audit=audit, max_new_tokens=max_new_tokens)
    result["serialization_wall_seconds"] = time.perf_counter() - begin
    return result
