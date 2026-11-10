"""Locked packed-verifier P1 corpus, training and acceptance helpers.

This module deliberately has no Target forward.  The collector establishes
same-forward Target feature provenance and independent AR labels.  A P1 arm
receives only ``ProbeObservation``; future labels stay in the trainer/evaluator.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .fixed_probe import ProbeExample
from .v2_upper_bound import MatchedUpperBound, UPPER_MODES


PACKED_P1_VERSION = "packed_fixed_prefix_p1_v1"
PACKED_FEATURE_VERSION = "packed_valid_feature_v2"
PACKED_FEATURE_SOURCE = "original_packed_verify_forward"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False)
                           .encode("utf-8")).hexdigest()


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class PackedP1Corpus:
    examples: tuple[ProbeExample, ...]
    protocol: dict[str, Any]
    corpus_sha256: str
    protocol_sha256: str


def load_packed_p1_corpus(corpus_path: Path, protocol_path: Path, *, split: str,
                          allow_test: bool = False) -> PackedP1Corpus:
    """Reject legacy sequential labels, incomplete provenance and cross-split rows."""
    if split not in ("train", "val", "test", "smoke") or (
            split == "test" and not allow_test):
        raise ValueError("test is sealed; an explicit final-evaluation opt-in is required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("scope") != "packed_fixed_prefix_collection"
            or protocol.get("eval_protocol_version") != PACKED_P1_VERSION
            or protocol.get("split") != split
            or protocol.get("window") != 8
            or protocol.get("max_future_tokens") != 8):
        raise ValueError("P1 requires the new locked packed L8 collection protocol")
    corpus_sha = sha256_file(corpus_path)
    if protocol.get("fixed_prefix_sha256") != corpus_sha:
        raise ValueError("packed fixed-prefix corpus SHA differs from collection")
    for key in ("target_file_sha256", "draft_file_sha256", "packed_p0_source_sha256"):
        hashes = protocol.get(key)
        if not isinstance(hashes, dict) or not hashes or not all(
                isinstance(path, str) and _is_sha(value)
                for path, value in hashes.items()):
            raise ValueError(f"missing or malformed {key}")
    for key in ("prompt_manifest_sha256", "prompt_manifest_meta_sha256",
                "packed_p0_report_sha256", "collection_plan_sha256"):
        if not _is_sha(protocol.get(key)):
            raise ValueError(f"missing or malformed {key}")
    if "eos_token_id" not in protocol or (
            protocol["eos_token_id"] is not None and
            (type(protocol["eos_token_id"]) is not int or protocol["eos_token_id"] < 0)):
        raise ValueError("collector must pin its EOS token ID")
    audit = protocol.get("audit")
    if not isinstance(audit, dict) or any(
            type(audit.get(key)) is not int or audit[key] <= 0
            for key in ("probe_windows", "independent_ar_prefix_exact_windows",
                        "prompts", "exact_requests")) or (
            audit["probe_windows"] != audit["independent_ar_prefix_exact_windows"]
            or audit["prompts"] != audit["exact_requests"]):
        raise ValueError("collector lacks complete independent AR correctness audit")
    examples: list[ProbeExample] = []
    keys: set[tuple[str, int]] = set()
    hidden_width: int | None = None
    with corpus_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("collection_protocol_version") != PACKED_P1_VERSION
                    or row.get("packed_feature_protocol_version") != PACKED_FEATURE_VERSION
                    or row.get("packed_feature_source") != PACKED_FEATURE_SOURCE
                    or row.get("online_feature_provenance_passed") is not True
                    or row.get("independent_ar_prefix_exact") is not True
                    or not isinstance(row.get("packed_trace"), dict)
                    or row.get("packed_trace_sha256") != canonical_sha256(row["packed_trace"])):
                raise ValueError(f"row {line_number} lacks authentic packed provenance")
            example = ProbeExample.from_json(row)
            o = example.observation
            if o.split != split or o.key in keys or o.window != 8 or (
                    o.round_index < 0 or o.remaining_budget <= 0
                    or not 0 < len(o.proposals) <= o.window
                    or not 0 <= o.accepted_count <= len(o.proposals)
                    or o.valid_prefix != o.source_prefix + o.proposals[:o.accepted_count]
                    or o.target_hidden_prefix != o.valid_prefix
                    or o.corrected_prefix != o.valid_prefix + (o.correction_token,)
                    or (o.rejected_token is None) != (o.accepted_count == len(o.proposals))
                    or (o.rejected_token is not None and
                        o.rejected_token != o.proposals[o.accepted_count])
                    or o.target_logit_summary is None
                    or o.target_logit_summary.greedy_token_id != o.correction_token):
                raise ValueError(f"row {line_number} has an invalid causal boundary")
            if (not example.target_future_ids
                    or len(example.target_future_ids) > min(o.window, o.remaining_budget)
                    or any(type(t) is not int or t < 0 for t in example.target_future_ids)
                    or (len(example.target_future_ids) < min(o.window, o.remaining_budget)
                        and (protocol["eos_token_id"] is None or
                             example.target_future_ids[-1] != protocol["eos_token_id"]))):
                raise ValueError(f"row {line_number} has incomplete future labels")
            if hidden_width is None:
                hidden_width = o.target_hidden.numel()
            if o.target_hidden.numel() != hidden_width:
                raise ValueError("target Hidden width varies within one corpus")
            keys.add(o.key)
            examples.append(example)
    if not examples:
        raise ValueError("packed collection contains no P1 windows")
    if len(examples) != audit["probe_windows"]:
        raise ValueError("packed corpus window count differs from independent AR audit")
    return PackedP1Corpus(tuple(examples), protocol, corpus_sha,
                          sha256_file(protocol_path))


def exposure_order(count: int, *, seed: int, steps: int) -> tuple[int, ...]:
    """One no-replacement epoch; every arm of a seed gets the same order."""
    if count <= 0 or steps <= 0 or steps > count or seed < 0:
        raise ValueError("P1 CE-only budget must fit within one no-replacement epoch")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(torch.randperm(count, generator=rng)[:steps].tolist())


def train_one_window(model: MatchedUpperBound, draft: Any,
                     example: ProbeExample, optimizer: torch.optim.Optimizer,
                     *, temperature: float, random_seed: int) -> float:
    """Teacher-forced CE; labels are never passed into prepare_observation."""
    if model.mode not in UPPER_MODES or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid matched P1 model or Gumbel temperature")
    labels = list(example.target_future_ids)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        hidden = draft.teacher_hidden(list(example.observation.corrected_prefix), labels)
    if (hidden.shape != (len(labels), model.config.draft_hidden_size)
            or not torch.isfinite(hidden).all()):
        raise ValueError("frozen Draft teacher hidden is invalid")
    prepared = model.prepare_observation(example.observation, draft)
    device = next(p for p in model.parameters() if p.requires_grad).device
    hidden = hidden.detach().unsqueeze(0).to(device)
    positions = torch.arange(1, len(labels) + 1, device=device)
    generators = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=generators):
        torch.manual_seed(random_seed)
        logits = model.forward_logits(hidden, prepared, positions,
                                      temperature=temperature)
    if logits.shape != (1, len(labels), model.config.vocab_size):
        raise ValueError("matched P1 logits have an unexpected shape")
    target = torch.tensor(labels, dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits.float().reshape(len(labels), -1), target)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("nonfinite P1 CE loss")
    loss.backward()
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None and (
                not torch.isfinite(parameter.grad).all().item()):
            raise FloatingPointError("nonfinite P1 adapter gradient")
    optimizer.step()
    return float(loss.detach().item())


def accepted_prefix(draft_ids: Iterable[int], target_ids: Iterable[int]) -> int:
    accepted = 0
    for draft_token, target_token in zip(draft_ids, target_ids):
        if draft_token != target_token:
            break
        accepted += 1
    return accepted


def audit_evaluation_row(row: dict[str, Any], example: ProbeExample, *,
                         eos_token_id: int | None, expected_mode: str) -> None:
    """Independently recompute every acceptance/EOS/length field from token IDs.

    This runs on a newly generated row, on every saved row during resume, and
    again when the cross-arm comparison loads the sealed row files.
    """
    o = example.observation
    expected_identity = (o.prompt_id, o.family_id, o.task_type, o.round_index)
    actual_identity = (row.get("prompt_id"), row.get("family_id"),
                       row.get("task_type"), row.get("round_index"))
    if row.get("mode") != expected_mode or actual_identity != expected_identity:
        raise ValueError("evaluation row identity/mode differs from frozen probe")
    draft_ids = row.get("draft_ids")
    target_ids = list(example.target_future_ids)
    limit = min(o.window, o.remaining_budget)
    if (not isinstance(draft_ids, list) or not 1 <= len(draft_ids) <= limit
            or any(type(token) is not int or token < 0 for token in draft_ids)
            or row.get("target_future_ids") != target_ids):
        raise ValueError("evaluation row has malformed or changed token IDs")
    if eos_token_id is not None:
        if eos_token_id in draft_ids[:-1] or eos_token_id in target_ids[:-1]:
            raise ValueError("evaluation row continued after EOS")
        if len(draft_ids) < limit and draft_ids[-1] != eos_token_id:
            raise ValueError("evaluation row stopped early without EOS")
    elif len(draft_ids) < limit:
        raise ValueError("evaluation row stopped early without an EOS definition")
    accepted = accepted_prefix(draft_ids, target_ids)
    first_rejection = (accepted + 1 if accepted < min(len(draft_ids), len(target_ids))
                       else None)
    primary = len(target_ids) == o.window and (
        eos_token_id is None or eos_token_id not in target_ids)
    eos_match = ((eos_token_id in draft_ids) == (eos_token_id in target_ids)
                 if eos_token_id is not None else True)
    expected = {
        "accepted_count": accepted,
        "first_rejection_position": first_rejection,
        "primary_eligible": primary,
        "eos_match": eos_match,
        "truncation_match": len(draft_ids) == len(target_ids),
        "next_token_top1": int(draft_ids[0] == target_ids[0]),
    }
    for key, value in expected.items():
        if type(row.get(key)) is not type(value) or row[key] != value:
            raise ValueError(f"evaluation row {key} is inconsistent with token IDs")
    ce = row.get("teacher_forced_ce")
    if type(ce) not in (float, int) or not math.isfinite(ce) or ce < 0:
        raise ValueError("evaluation row has invalid teacher-forced CE")


@torch.inference_mode()
def evaluate_one_window(model: MatchedUpperBound, draft: Any,
                        example: ProbeExample, *, eos_token_id: int | None) -> dict[str, Any]:
    """Greedy autoregressive Draft continuation on the same corrected prefix."""
    o = example.observation
    model.eval()
    state = model.prepare_observation(o, draft)
    prefix = list(o.corrected_prefix)
    target_ids = tuple(example.target_future_ids)
    draft_ids: list[int] = []
    for position in range(1, min(o.window, o.remaining_budget) + 1):
        token = model.next_token(prefix, state, position, draft)
        draft_ids.append(token)
        prefix.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
    count = accepted_prefix(draft_ids, target_ids)
    teacher_hidden = draft.teacher_hidden(list(o.corrected_prefix), list(target_ids))
    device = next(p for p in model.parameters() if p.requires_grad).device
    positions = torch.arange(1, len(target_ids) + 1, device=device)
    teacher_logits = model.forward_logits(teacher_hidden.unsqueeze(0).to(device),
                                          state, positions)
    teacher_labels = torch.tensor(target_ids, dtype=torch.long,
                                  device=teacher_logits.device)
    teacher_ce = F.cross_entropy(teacher_logits.float().reshape(len(target_ids), -1),
                                 teacher_labels)
    if not torch.isfinite(teacher_ce).item():
        raise FloatingPointError("nonfinite validation CE")
    primary = len(target_ids) == o.window and (
        eos_token_id is None or eos_token_id not in target_ids)
    return {"mode": model.mode, "prompt_id": o.prompt_id,
            "family_id": o.family_id, "task_type": o.task_type,
            "round_index": o.round_index, "accepted_count": count,
            "first_rejection_position": (count + 1 if count < min(
                len(draft_ids), len(target_ids)) else None),
            "draft_ids": draft_ids, "target_future_ids": list(target_ids),
            "teacher_forced_ce": float(teacher_ce.item()),
            "next_token_top1": int(bool(draft_ids) and draft_ids[0] == target_ids[0]),
            "primary_eligible": primary, "eos_match": (
                (eos_token_id in draft_ids) == (eos_token_id in target_ids)
                if eos_token_id is not None else True),
            "truncation_match": len(draft_ids) == len(target_ids)}


def prompt_means(rows: Iterable[dict[str, Any]], *, primary_only: bool) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if not primary_only or row["primary_eligible"]:
            groups[row["prompt_id"]].append(row["accepted_count"])
    return {key: mean(values) for key, values in groups.items() if values}


def paired_prompt_ci(left_rows: Iterable[dict[str, Any]],
                     right_rows: Iterable[dict[str, Any]], *,
                     primary_only: bool = True, repetitions: int = 10_000,
                     seed: int = 42) -> dict[str, Any]:
    """Paired bootstrap resamples prompts, never individual correlated windows."""
    if repetitions < 100 or seed < 0:
        raise ValueError("invalid bootstrap plan")
    left = prompt_means(left_rows, primary_only=primary_only)
    right = prompt_means(right_rows, primary_only=primary_only)
    if set(left) != set(right) or not left:
        raise ValueError("paired prompt sets differ or primary subset is empty")
    prompt_ids = sorted(left)
    values = [left[key] - right[key] for key in prompt_ids]
    random_state = random.Random(seed)
    draws = sorted(mean(random_state.choices(values, k=len(values)))
                   for _ in range(repetitions))
    return {"prompt_count": len(values), "mean_delta": mean(values),
            "ci95": [draws[int(0.025 * repetitions)],
                     draws[min(repetitions - 1, int(0.975 * repetitions))]],
            "bootstrap_repetitions": repetitions, "seed": seed}
