"""Full-request B/C/F-H acceptance versus recorded original packed Vanilla SD.

This is a semantic, full-prefix verifier. The original packed trace is replayed
exactly before scoring any repaired arm. It makes no throughput or WAN claim.
The caller owns model-asset checks, GPU admission and the cross-root flock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from feedback_codec.codec import CodecConfig
from feedback_codec.fixed_probe import ProbeObservation
from feedback_codec.v11_stats import paired_family_ci
from feedback_codec.v10_transformer_bcfh import ARMS, TransformerBCFHHead
from scripts.load_transformer_v13_windows import OriginalRawRequest
from scripts.transformer_bcfh_experiment import (
    ExperimentLock, _digest_json, _draft_fingerprint, _head, _source_hashes,
    _atomic_json,
)


@dataclass(frozen=True)
class ClosedLoopResult:
    tokens: tuple[int, ...]
    proposals: tuple[tuple[int, ...], ...]
    accepted: tuple[int, ...]

    @property
    def accepted_tokens_per_round(self) -> float:
        return sum(self.accepted) / len(self.accepted)


def _raw_token(draft: Any, prefix: list[int], hidden_size: int) -> int:
    hidden = draft.teacher_hidden(prefix, [0])
    if hidden.shape != (1, hidden_size):
        raise ValueError("frozen Draft next-token hidden has wrong shape")
    logits = draft.model.get_output_embeddings()(hidden)
    if logits.ndim != 2 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise ValueError("frozen Draft next-token logits are invalid")
    return int(logits[0].argmax(-1).item())


@torch.inference_mode()
def generate_closed_loop(request: OriginalRawRequest, target: Any, draft: Any,
                         head: TransformerBCFHHead | None, *, eos: int,
                         max_new_tokens: int = 128) -> ClosedLoopResult:
    """Greedy speculative request; feedback starts only after round one."""
    if max_new_tokens != 128 or not request.prompt_input_ids:
        raise ValueError("closed-loop prompt or output limit differs")
    confirmed = list(request.prompt_input_ids)
    generated: list[int] = []
    proposals_all: list[tuple[int, ...]] = []
    accepted_all: list[int] = []
    state = None
    while len(generated) < max_new_tokens:
        before = tuple(confirmed)
        proposal: list[int] = []
        for position in range(1, min(8, max_new_tokens - len(generated)) + 1):
            token = (_raw_token(draft, confirmed + proposal,
                                draft.hidden_size)
                     if head is None or state is None else
                     head.next_token(confirmed + proposal, state, position, draft))
            proposal.append(token)
            if token == eos:
                break
        accepted = 0
        rejected: int | None = None
        correction: int | None = None
        boundary_hidden = None
        valid = None
        for token in proposal:
            step = target.next(confirmed.copy())
            if token != int(step.token_id):
                rejected = token
                correction = int(step.token_id)
                boundary_hidden = step.hidden
                valid = tuple(confirmed)
                confirmed.append(correction)
                generated.append(correction)
                break
            accepted += 1
            confirmed.append(token)
            generated.append(token)
            if token == eos:
                break
        terminal = (bool(generated) and generated[-1] == eos
                    or len(generated) == max_new_tokens)
        if rejected is None and not terminal:
            step = target.next(confirmed.copy())
            correction = int(step.token_id)  # standard speculative bonus
            boundary_hidden = step.hidden
            valid = tuple(confirmed)
            confirmed.append(correction)
            generated.append(correction)
            terminal = (correction == eos or len(generated) == max_new_tokens)
        proposals_all.append(tuple(proposal))
        accepted_all.append(accepted)
        if terminal:
            break
        if correction is None or valid is None or boundary_hidden is None:
            raise RuntimeError("nonterminal verification has no correction or bonus")
        if head is not None:
            observation = ProbeObservation(
                prompt_id=request.prompt_id, family_id=request.family_id,
                split="val", task_type=request.source,
                round_index=len(proposals_all) - 1, source_prefix=before,
                proposals=tuple(proposal), valid_prefix=valid,
                accepted_count=accepted, rejected_token=rejected,
                correction_token=correction, corrected_prefix=tuple(confirmed),
                target_hidden=boundary_hidden.detach().clone(),
                target_hidden_prefix=valid,
                remaining_budget=max_new_tokens - len(generated), window=8)
            state = head.prepare_observation(observation, draft)
    return ClosedLoopResult(tuple(generated), tuple(proposals_all),
                            tuple(accepted_all))


@torch.inference_mode()
def evaluate_closed_loop(requests: Mapping[str, OriginalRawRequest],
                         target: Any, draft: Any, draft_model: torch.nn.Module,
                         config: CodecConfig, output: Path, *, eos: int,
                         val_acceptance_receipt: Path,
                         lock: ExperimentLock = ExperimentLock(),
                         ) -> dict[str, Any]:
    """Formal acceptance gate: full-request tokens accepted per round > original SD."""
    if (len(requests) != lock.val_prompts or draft.model is not draft_model
            or any(p.requires_grad for p in draft_model.parameters())):
        raise ValueError("independent 200-prompt frozen Draft validation is required")
    val_receipt = json.loads(val_acceptance_receipt.read_text(encoding="utf-8"))
    receipt_members = val_receipt.get("members", [])
    expected_members = {prompt_id: request.manifest_sha256
                        for prompt_id, request in requests.items()}
    observed_members = {member["prompt_id"]: member["manifest_sha256"]
                        for member in receipt_members}
    if (val_receipt.get("exact_requests") != lock.val_prompts
            or val_receipt.get("fully_valid_requests") != lock.val_prompts
            or len(receipt_members) != lock.val_prompts
            or observed_members != expected_members):
        raise ValueError("independent full-field val acceptance does not bind raw requests")
    val_receipt_sha = hashlib.sha256(val_acceptance_receipt.read_bytes()).hexdigest()
    terminal = json.loads((output / "training_terminal.json").read_text(encoding="utf-8"))
    record = json.loads((output / "transformer_experiment_lock.json")
                        .read_text(encoding="utf-8"))
    if (terminal.get("state") != "trained"
            or terminal.get("lock_sha256") != _digest_json(record)
            or record.get("lock") != asdict(lock)
            or record.get("codec_config") != asdict(config)
            or record.get("source_sha256") != _source_hashes()
            or record.get("draft_runtime_fingerprint")
               != _draft_fingerprint(draft_model)):
        raise ValueError("closed-loop model or training lock differs")
    rows_path = output / "closed_loop_rows.jsonl"
    summary_path = output / "closed_loop_summary.json"
    if rows_path.exists() or summary_path.exists():
        raise FileExistsError("preserve existing closed-loop evaluation")
    heads: dict[str, TransformerBCFHHead] = {}
    for mode in ARMS:
        payload = torch.load(output / f"{mode}.final.pt", map_location="cpu",
                             weights_only=False)
        if (payload.get("mode") != mode
                or payload.get("lock_sha256") != terminal["lock_sha256"]
                or payload.get("updates_this_arm")
                   != lock.epochs * lock.train_windows):
            raise ValueError(f"{mode} final Transformer checkpoint differs")
        head = _head(config, draft_model, lock, mode, draft.device)
        head.load_trainable_state_dict(payload["trainable_state"])
        heads[mode] = head.eval()
    rows: list[dict[str, Any]] = []
    scores: dict[tuple[int, str, str, str, str], float] = {}
    for prompt_id in sorted(requests):
        request = requests[prompt_id]
        if request.prompt_id != prompt_id:
            raise ValueError("original SD request key differs")
        raw = generate_closed_loop(request, target, draft, None, eos=eos)
        if (raw.tokens != request.oracle_tokens
                or raw.proposals != request.round_proposals
                or raw.accepted != request.round_accepted):
            raise ValueError(f"original packed SD cannot be replayed exactly: {prompt_id}")
        outcomes = {"ORIGINAL_SD": raw}
        for mode, head in heads.items():
            result = generate_closed_loop(request, target, draft, head, eos=eos)
            if result.tokens != request.oracle_tokens:
                raise ValueError(f"{mode} changes Target greedy output: {prompt_id}")
            outcomes[mode] = result
        row = {
            "prompt_id": prompt_id, "family_id": request.family_id,
            "source": request.source, "original_manifest_sha256": request.manifest_sha256,
            "oracle_sha256": hashlib.sha256(
                json.dumps(request.oracle_tokens).encode("ascii")).hexdigest(),
            "arms": {mode: {"accepted_total": sum(result.accepted),
                            "verification_rounds": len(result.accepted),
                            "accepted_tokens_per_round": result.accepted_tokens_per_round,
                            "round_accepted": list(result.accepted)}
                     for mode, result in outcomes.items()},
        }
        rows.append(row)
        for mode, result in outcomes.items():
            scores[(lock.seed, request.source, request.family_id, prompt_id, mode)] = (
                result.accepted_tokens_per_round)
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    comparisons = {}
    for mode in ARMS:
        result = paired_family_ci(scores, seed=lock.seed, left=mode,
                                  right="ORIGINAL_SD", draws=lock.bootstrap_draws,
                                  bootstrap_seed=lock.bootstrap_seed)
        comparisons[mode] = {**result, "confirmed_positive": result["ci95"][0] > 0}
    summary = {
        "scope": "transformer_bcfh_full_request_vs_original_packed_vanilla_sd",
        "train_lock_sha256": terminal["lock_sha256"],
        "val_acceptance_receipt_sha256": val_receipt_sha,
        "original_request_count": len(requests),
        "original_raw_replay_exact": True,
        "target_ar_output_exact_all_arms": True,
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "mean_accepted_tokens_per_round": {
            mode: sum(row["arms"][mode]["accepted_tokens_per_round"] for row in rows)
                  / len(rows) for mode in ("ORIGINAL_SD",) + ARMS},
        "paired_vs_original_sd": comparisons,
        "interpretation": "Full-request acceptance only; no packed repair throughput, "
                          "speed or WAN claim. F-H is an uncompressed teacher diagnostic.",
    }
    _atomic_json(summary_path, summary)
    return summary
