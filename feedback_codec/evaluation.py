"""Paired, correctness-gated acceptance diagnostics for the feedback codec.

These metrics count proposal/verification rounds.  The full-prefix reference
backends deliberately cannot support any throughput or WAN latency claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import mean
from typing import Iterable

from .decoding import GenerationResult, SpeculativeDecoder, autoregressive_generate
from .training import CodecEncoder


@dataclass(frozen=True)
class Prompt:
    id: str
    input_ids: tuple[int, ...]


class _ReplayEncoder:
    def __init__(self, messages: list[tuple[int, ...]]) -> None:
        self.messages = messages
        self.index = 0

    def encode(self, target_hidden, rejected_embedding, correction_embedding, accepted_count):
        if not self.messages:
            raise RuntimeError("shuffled control has no donor messages")
        message = self.messages[self.index % len(self.messages)]
        self.index += 1
        return message


def _summarize_result(
    prompt: Prompt,
    strategy: str,
    result: GenerationResult,
    oracle: tuple[int, ...],
    eos_token_id: int | None,
    max_new_tokens: int,
) -> dict:
    if result.tokens != oracle:
        raise RuntimeError(f"{strategy} differs from independent target AR on {prompt.id}")
    rounds = result.verification_rounds
    accepted = result.accepted_draft_tokens
    proposed = sum(len(row.proposals) for row in result.rounds)
    repair_positions = sum(len(row.proposals) for row in result.rounds if row.codes_in is not None)
    rows = [
        {
            "accepted_draft_tokens": row.accepted_count,
            "proposal_tokens": len(row.proposals),
            "confirmed_tokens": len(row.committed_tokens),
            "first_rejection_position": (
                row.accepted_count + 1 if row.rejected_token is not None else None
            ),
            "message_bytes": row.message_bytes,
        }
        for row in result.rounds
    ]
    return {
        "id": prompt.id,
        "strategy": strategy,
        "tokens": list(result.tokens),
        "exact_target_ar": True,
        "stopped_on_eos": bool(result.tokens) and result.tokens[-1] == eos_token_id,
        "hit_max_new_tokens": len(result.tokens) == max_new_tokens,
        "verification_rounds": rounds,
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "repair_positions": repair_positions,
        "draft_token_acceptance_fraction": accepted / proposed if proposed else 0.0,
        "accepted_draft_tokens_per_round": accepted / rounds if rounds else 0.0,
        "confirmed_tokens_per_round": len(result.tokens) / rounds if rounds else 0.0,
        "feedback_bytes": result.total_message_bytes,
        "rounds": rows,
    }


def _bootstrap_mean_ci(differences: list[float], *, repetitions: int, seed: int) -> list[float]:
    if not differences:
        raise ValueError("paired differences cannot be empty")
    if repetitions < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    rng = random.Random(seed)
    size = len(differences)
    draws = sorted(
        mean(differences[rng.randrange(size)] for _ in range(size))
        for _ in range(repetitions)
    )
    return [draws[int(0.025 * (repetitions - 1))], draws[int(0.975 * (repetitions - 1))]]


def evaluate_paired(
    *,
    target,
    draft,
    codec,
    prompts: Iterable[Prompt],
    window: int = 8,
    max_new_tokens: int = 128,
    eos_token_id: int | None = None,
    shuffled_control: bool = True,
    bootstrap_repetitions: int = 2000,
    seed: int = 0,
) -> dict:
    """Run identical prompts under no feedback, learned, and shuffled codes.

    Shuffled messages are drawn from another request's *measured* learned-code
    stream, and are used only as a control.  They are never training labels.
    """
    prompts = list(prompts)
    if not prompts or len({p.id for p in prompts}) != len(prompts):
        raise ValueError("prompts must be nonempty with unique IDs")
    if shuffled_control and len(prompts) < 2:
        raise ValueError("shuffled control requires at least two prompts")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    baseline_decoder = SpeculativeDecoder(target, draft, None, window, eos_token_id)
    learned_decoder = SpeculativeDecoder(target, draft, CodecEncoder(codec), window, eos_token_id)
    rows: list[dict] = []
    message_streams: list[list[tuple[int, ...]]] = []
    oracles: list[tuple[int, ...]] = []
    for prompt in prompts:
        oracle = autoregressive_generate(target, list(prompt.input_ids), max_new_tokens, eos_token_id)
        oracles.append(oracle)
        for strategy, decoder in (("no_feedback", baseline_decoder), ("learned_feedback", learned_decoder)):
            result = decoder.generate(list(prompt.input_ids), max_new_tokens)
            rows.append(_summarize_result(prompt, strategy, result, oracle, eos_token_id, max_new_tokens))
            if strategy == "learned_feedback":
                message_streams.append([r.feedback_codes for r in result.rounds if r.feedback_codes is not None])
    usable_donors = [index for index, stream in enumerate(message_streams) if stream]
    shuffled_available = shuffled_control and len(usable_donors) >= 2
    if shuffled_available:
        for index, prompt in enumerate(prompts):
            donor_index = next(
                candidate for candidate in usable_donors if candidate != index
            )
            donor = message_streams[donor_index]
            decoder = SpeculativeDecoder(target, draft, _ReplayEncoder(donor), window, eos_token_id)
            result = decoder.generate(list(prompt.input_ids), max_new_tokens)
            rows.append(
                _summarize_result(
                    prompt, "shuffled_feedback", result, oracles[index], eos_token_id, max_new_tokens
                )
            )

    by_strategy = {name: [r for r in rows if r["strategy"] == name] for name in (
        "no_feedback", "learned_feedback", "shuffled_feedback"
    )}
    summary = {
        name: {
            "prompt_count": len(group),
            "accepted_draft_tokens_per_round": mean(r["accepted_draft_tokens_per_round"] for r in group),
            "draft_token_acceptance_fraction": mean(r["draft_token_acceptance_fraction"] for r in group),
            "verification_rounds_per_request": mean(r["verification_rounds"] for r in group),
            "feedback_bytes_per_request": mean(r["feedback_bytes"] for r in group),
        }
        for name, group in by_strategy.items()
        if group
    }
    left = {r["id"]: r for r in by_strategy["no_feedback"]}
    right = {r["id"]: r for r in by_strategy["learned_feedback"]}
    differences = [
        right[p.id]["accepted_draft_tokens_per_round"]
        - left[p.id]["accepted_draft_tokens_per_round"]
        for p in prompts
    ]
    acceptance_fraction_differences = [
        right[p.id]["draft_token_acceptance_fraction"]
        - left[p.id]["draft_token_acceptance_fraction"]
        for p in prompts
    ]
    return {
        "scope": "canonical_full_prefix_acceptance_only",
        "latency_claim_valid": False,
        "shuffled_control_status": (
            "completed"
            if shuffled_available
            else "unavailable_fewer_than_two_donor_message_streams"
            if shuffled_control
            else "disabled"
        ),
        "window": window,
        "max_new_tokens": max_new_tokens,
        "rows": rows,
        "summary": summary,
        "paired_learned_minus_no_feedback": {
            "mean_accepted_draft_tokens_per_round": mean(differences),
            "bootstrap_95_percent_ci": _bootstrap_mean_ci(
                differences, repetitions=bootstrap_repetitions, seed=seed
            ),
            "bootstrap_repetitions": bootstrap_repetitions,
            "mean_draft_token_acceptance_fraction": mean(acceptance_fraction_differences),
            "acceptance_fraction_bootstrap_95_percent_ci": _bootstrap_mean_ci(
                acceptance_fraction_differences,
                repetitions=bootstrap_repetitions,
                seed=seed + 1,
            ),
        },
    }
