"""Assemble post-verify observations and physically separate future labels.

This module performs no model forward.  Its inputs are the actual frozen
packed-decoder rounds, observed proposal forwards, and independent AR tape.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence

from .packed_probe_numeric_v3 import canonical_bytes


SCHEMA = "feedback64-rounds-v12"


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def request_id(dataset_lock_sha256: str, split: str, prompt_id: str,
               policy_id: str, execution_id: str) -> str:
    return digest([dataset_lock_sha256, split, prompt_id, policy_id, execution_id])


def _status(obs: dict[str, Any]) -> tuple[int | None, str, int | None]:
    m, accepted = obs["draft_length"], obs["accepted_count"]
    if accepted < m:
        return accepted + 1, "observed_rejection", None
    if m == 8:
        return 9, "full_window_accepted", None
    if obs["terminal_reason"] == "eos":
        return None, "terminal_eos_without_rejection", None
    if obs["terminal_reason"] == "length":
        return None, "right_censored_length", m + 1
    raise ValueError("short full acceptance must end naturally")


def assemble_request(
    *, prompt: Any, records: Sequence[Any], draft_steps: Sequence[dict[str, Any]],
    oracle_tokens: Sequence[int], oracle_steps: Sequence[dict[str, Any]],
    eos_token_id: int, dataset_lock_sha256: str, policy_id: str,
    execution_id: str,
    hidden_ref: Callable[[int, Any, list[int]], dict[str, Any]],
    oracle_ref: Callable[[int, int], dict[str, Any]],
    max_new_tokens: int = 128,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build all real rounds; labels are never inserted into observations."""
    if (max_new_tokens != 128 or not records or
            len(oracle_tokens) != len(oracle_steps)):
        raise ValueError("incomplete full request or independent AR tape")
    rid = request_id(dataset_lock_sha256, prompt.split, prompt.id,
                     policy_id, execution_id)
    committed_all = tuple(t for record in records for t in record.committed_tokens)
    if committed_all != tuple(oracle_tokens):
        raise ValueError("packed stream differs from independent AR")
    if any(step["greedy_token_id"] != token for step, token in
           zip(oracle_steps, oracle_tokens, strict=True)):
        raise ValueError("independent AR Top-K tape differs from greedy IDs")
    if len(draft_steps) != sum(len(r.proposals) for r in records):
        raise ValueError("actual Draft step count differs from all proposals")
    observations: list[dict[str, Any]] = []
    used = 0
    step_cursor = 0
    for index, record in enumerate(records):
        prefix = list(record.prefix_before)
        proposals = list(record.proposals)
        m = len(proposals)
        accepted = record.accepted_count
        if (not 1 <= m <= 8 or not 0 <= accepted <= m
                or prefix != (list(prompt.input_ids) if index == 0 else
                              observations[-1]["prefix_after_commit_token_ids"])):
            raise ValueError("bad actual round length, acceptance, or prefix link")
        steps = [dict(s) for s in draft_steps[step_cursor:step_cursor + m]]
        step_cursor += m
        for position, (step, token) in enumerate(zip(steps, proposals), 1):
            if (step.get("position") != position or step.get("token_id") != token
                    or step.get("proposal_prefix_sha256")
                       != digest(prefix + proposals[:position - 1])):
                raise ValueError("observed Draft step is off the actual prefix")
        b = (record.correction_token if record.rejected_token is not None
             else record.bonus_token)
        valid = prefix + proposals[:accepted]
        after = prefix + list(record.committed_tokens)
        if (after != valid + ([] if b is None else [b])
                or record.rejected_token
                   != (proposals[accepted] if accepted < m else None)):
            raise ValueError("actual correction or commitment differs")
        before_budget = max_new_tokens - used
        used += len(record.committed_tokens)
        after_budget = max_new_tokens - used
        if before_budget < m or after_budget < 0:
            raise ValueError("actual proposal exceeded remaining budget")
        terminal = ("eos" if after[-1] == eos_token_id else
                    "length" if after_budget == 0 else "continue")
        if b is None:
            if terminal == "continue" or record.observation_hidden is not None:
                raise ValueError("nonterminal round lacks correction/bonus")
            teacher_ref, teacher_topk = None, None
        else:
            summary = record.observation_logit_summary
            if (record.observation_hidden is None or summary is None
                    or summary.greedy_token_id != b):
                raise ValueError("actual same-boundary teacher feature absent")
            teacher_ref = hidden_ref(index, record, valid)
            teacher_topk = {
                "greedy_token_id": int(summary.greedy_token_id),
                "token_ids": list(summary.top_token_ids),
                "logits": list(summary.top_logits_fp16),
                "logsumexp": float(summary.logsumexp_fp32),
                "vocab_size": int(summary.vocab_size),
                "topk_storage_dtype": "torch.float16",
            }
        obs = {
            "schema_version": SCHEMA,
            "request_id": rid,
            "round_id": index,
            "split": prompt.split,
            "family_id": prompt.family_id,
            "source": prompt.task_type,
            "policy_id": policy_id,
            "execution_id": execution_id,
            "window_size": 8,
            "eos_token_id": eos_token_id,
            "prefix_token_ids": prefix,
            "draft_length": m,
            "draft_token_ids": proposals,
            "accepted_count": accepted,
            "first_reject_position": None,
            "rejected_token_id": record.rejected_token,
            "corrected_token_id": b,
            "correction_kind": ("correction" if record.rejected_token is not None
                                else "bonus" if b is not None else "none"),
            "valid_prefix_token_ids": valid,
            "prefix_after_commit_token_ids": after,
            "target_valid_hidden_ref": teacher_ref,
            "target_topk": teacher_topk,
            "draft_steps": steps,
            "terminal_reason": terminal,
            "remaining_budget_before": before_budget,
            "remaining_budget_after": after_budget,
            "feature_status": "available" if b is not None else "not_applicable_no_b",
        }
        obs["first_reject_position"] = _status(obs)[0]
        observations.append(obs)
    if observations[-1]["terminal_reason"] == "continue":
        raise ValueError("request stopped without natural EOS or length")
    labels: list[dict[str, Any]] = []
    for index, obs in enumerate(observations):
        successor = observations[index + 1] if index + 1 < len(observations) else None
        if (successor is None) != (obs["terminal_reason"] != "continue"):
            raise ValueError("missing or extraneous actual successor")
        if successor is None:
            next_id = next_m = next_accepted = next_reject = censor = None
            next_status = "no_next_round"
        else:
            if (successor["prefix_token_ids"] != obs["prefix_after_commit_token_ids"]
                    or successor["remaining_budget_before"]
                       != obs["remaining_budget_after"]):
                raise ValueError("actual next-round boundary differs")
            next_id = successor["round_id"]
            next_m = successor["draft_length"]
            next_accepted = successor["accepted_count"]
            next_reject, next_status, censor = _status(successor)
        offset = len(obs["prefix_after_commit_token_ids"]) - len(prompt.input_ids)
        future = list(oracle_tokens[offset:offset + 8])
        if obs["terminal_reason"] != "continue" and future:
            raise ValueError("future labels cross natural request termination")
        labels.append({
            "request_id": rid,
            "round_id": index,
            "next_round_id": next_id,
            "next_draft_length": next_m,
            "next_accepted_count": next_accepted,
            "next_first_reject_position": next_reject,
            "next_label_status": next_status,
            "next_censor_lower_bound": censor,
            "target_future": {
                "token_ids": future,
                "horizon": len(future),
                "start_prefix_sha256": digest(obs["prefix_after_commit_token_ids"]),
                "topk_ref": oracle_ref(offset, len(future)) if future else None,
                "status": "available" if future else "terminal_no_future",
            },
        })
    return observations, labels
