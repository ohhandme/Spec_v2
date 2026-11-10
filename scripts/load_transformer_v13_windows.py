"""Bind B/C/F-H Transformer input to the repaired v13 1000/200 artifacts.

The index is built from the original frozen prompt roster plus exact request
manifests.  The reader verifies every selected artifact SHA before joining
causal boundary observations with separately stored future labels.  It does
not read a sealed test split or create replacement data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import torch
from safetensors.torch import load_file

from feedback_codec.fixed_probe import ProbeObservation
from feedback_codec.v11_full_contract import PilotWindow, locked_prompts, sha256_file
from feedback_codec.v12_acceptance import _artifact, _read_jsonl
from feedback_codec.v12_boundary_reader import load_boundary_inputs
from feedback_codec.v12_round_assembly import digest
from feedback_codec.v11_stats import prefix_acceptance


class OriginalRawNext(NamedTuple):
    """Actual next round of the original packed Vanilla SD collection."""

    request_id: str
    manifest_sha256: str
    proposals: tuple[int, ...]
    accepted_count: int


class OriginalRawRequest(NamedTuple):
    """Full actual packed Vanilla SD request, including the first round."""

    prompt_id: str
    family_id: str
    source: str
    prompt_input_ids: tuple[int, ...]
    oracle_tokens: tuple[int, ...]
    round_proposals: tuple[tuple[int, ...], ...]
    round_accepted: tuple[int, ...]
    manifest_sha256: str

    @property
    def accepted_tokens_per_round(self) -> float:
        return sum(self.round_accepted) / len(self.round_accepted)


def build_member_index(parent: Path, parent_meta: Path,
                       train_roots: Sequence[Path],
                       val_roots: Sequence[Path]) -> dict[str, Any]:
    """Discover exact terminal manifests for every pinned train/val prompt."""
    prompts = locked_prompts(parent, parent_meta)
    result: dict[str, Any] = {
        "scope": "transformer_bcfh_repaired_member_index_v1",
        "parent_sha256": sha256_file(parent),
        "parent_meta_sha256": sha256_file(parent_meta),
        "train": [], "val": [],
    }
    for split, roots in (("train", train_roots), ("val", val_roots)):
        roster = {prompt.id: prompt for prompt in prompts[split]}
        chosen: dict[str, dict[str, str]] = {}
        for root in roots:
            root = root.resolve()
            if root.name != split or not (root / "manifests").is_dir():
                raise ValueError(f"{split} root is not an artifact split directory")
            for path in sorted((root / "manifests").glob("*.json")):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                prompt_id = manifest.get("prompt_id")
                if prompt_id not in roster:
                    continue
                prompt = roster[prompt_id]
                if (prompt_id in chosen or manifest.get("exact") is not True
                        or manifest.get("split") != split
                        or manifest.get("family_id") != prompt.family_id
                        or manifest.get("source") != prompt.task_type
                        or manifest.get("request_id") != path.stem
                        or manifest.get("dataset_lock_sha256") != result["parent_sha256"]):
                    raise ValueError(f"duplicate or non-exact {split} prompt: {prompt_id}")
                chosen[prompt_id] = {
                    "root": str(root), "request_id": path.stem,
                    "prompt_id": prompt_id, "family_id": prompt.family_id,
                    "task_type": prompt.task_type,
                    "prompt_input_sha256": digest(list(prompt.input_ids)),
                    "manifest_sha256": sha256_file(path),
                }
        if set(chosen) != set(roster):
            raise ValueError(f"{split} is missing {len(set(roster)-set(chosen))} locked prompts")
        result[split] = [chosen[prompt.id] for prompt in prompts[split]]
    if {x["prompt_id"] for x in result["train"]} & {
            x["prompt_id"] for x in result["val"]}:
        raise ValueError("train/validation prompt overlap")
    return result


def verify_train_receipt(index: dict[str, Any], receipt_path: Path) -> str:
    """Require the complete v12 acceptance receipt, including exact members."""
    receipt_sha = sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    members = {row["request_id"]: row["manifest_sha256"]
               for row in receipt.get("members", [])}
    indexed = {row["request_id"]: row["manifest_sha256"]
               for row in index["train"]}
    counts = receipt.get("counts", {})
    if (receipt.get("train_ready") is not True
            or receipt.get("training_updates_before_gate") != 0
            or receipt.get("physical_future_label_input_isolation") is not True
            or receipt.get("dataset_lock_sha256") != index["parent_sha256"]
            or counts.get("fully_valid_requests") != 1000
            or counts.get("exact_requests") != 1000
            or len(receipt.get("members", [])) != 1000
            or members != indexed):
        raise ValueError("complete audited train receipt does not bind this 1000-member index")
    return receipt_sha


def load_windows(index: dict[str, Any], split: str, *, eos: int,
                 expected_windows: int) -> tuple[PilotWindow, ...]:
    """Read the first eight eligible legal windows of every locked request."""
    if (split not in ("train", "val")
            or index.get("scope") != "transformer_bcfh_repaired_member_index_v1"
            or len(index.get(split, [])) != (1000 if split == "train" else 200)):
        raise ValueError("repaired member index has the wrong scope or size")
    windows: list[PilotWindow] = []
    for member in index[split]:
        root = Path(member["root"]).resolve()
        request_id = member["request_id"]
        path = root / "manifests" / f"{request_id}.json"
        if (root.name != split or sha256_file(path) != member["manifest_sha256"]):
            raise ValueError("request manifest changed since member index")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("exact") is not True
                or manifest.get("prompt_id") != member["prompt_id"]
                or manifest.get("family_id") != member["family_id"]
                or manifest.get("source") != member["task_type"]
                or manifest.get("request_id") != request_id
                or manifest.get("split") != split
                or manifest.get("dataset_lock_sha256") != index["parent_sha256"]):
            raise ValueError("indexed request identity, split, or exact flag differs")
        boundary = load_boundary_inputs(root, path)
        labels = _read_jsonl(_artifact(root, manifest["files"]["labels"]))
        oracle = json.loads(_artifact(root, manifest["files"]["oracle_tape"])
                            .read_text(encoding="utf-8"))
        hidden_entry = manifest["files"].get("hidden")
        hidden = (load_file(str(_artifact(root, hidden_entry)), device="cpu")
                  if hidden_entry else {})
        if len(boundary) != len(labels):
            raise ValueError("boundary and future-label row counts differ")
        kept = 0
        for round_id, (observation, label) in enumerate(zip(boundary, labels, strict=True)):
            corrected = observation["corrected_token_id"]
            future = tuple(label["target_future"]["token_ids"])
            if (corrected is None or corrected == eos or not future):
                continue
            hidden_ref = observation["target_valid_hidden_ref"]
            future_ref = label["target_future"].get("topk_ref")
            if (hidden_ref is None or hidden_ref.get("key") not in hidden
                    or label.get("request_id") != request_id
                    or label.get("round_id") != round_id
                    or hidden_ref.get("sha256") != hidden_entry["sha256"]
                    or hidden_ref.get("prefix_sha256")
                       != digest(observation["valid_prefix_token_ids"])
                    or label["target_future"].get("start_prefix_sha256")
                       != digest(observation["prefix_after_commit_token_ids"])
                    or label["target_future"].get("horizon") != len(future)
                    or not isinstance(future_ref, dict)
                    or future_ref.get("path")
                       != manifest["files"]["oracle_tape"]["path"]
                    or future_ref.get("sha256")
                       != manifest["files"]["oracle_tape"]["sha256"]
                    or future_ref.get("length") != len(future)
                    or type(future_ref.get("offset")) is not int
                    or future_ref["offset"] < 0
                    or tuple(oracle["token_ids"][future_ref["offset"]:
                                                   future_ref["offset"] + len(future)])
                       != future
                    or not 1 <= len(future) <= 8):
                raise ValueError("eligible window lacks legal Target boundary or future")
            target_hidden = hidden[hidden_ref["key"]]
            if (target_hidden.ndim != 1
                    or list(target_hidden.shape) != hidden_ref.get("shape")
                    or str(target_hidden.dtype) != hidden_ref.get("dtype")
                    or not torch.isfinite(target_hidden).all()):
                raise ValueError("Target boundary hidden tensor is invalid")
            item = ProbeObservation(
                prompt_id=member["prompt_id"], family_id=member["family_id"],
                split=split, task_type=member["task_type"], round_index=round_id,
                source_prefix=tuple(observation["prefix_token_ids"]),
                proposals=tuple(observation["draft_token_ids"]),
                valid_prefix=tuple(observation["valid_prefix_token_ids"]),
                accepted_count=observation["accepted_count"],
                rejected_token=observation["rejected_token_id"],
                correction_token=corrected,
                corrected_prefix=tuple(observation["prefix_after_commit_token_ids"]),
                target_hidden=target_hidden.detach().clone(),
                target_hidden_prefix=tuple(observation["valid_prefix_token_ids"]),
                remaining_budget=observation["remaining_budget_after"],
                window=8)
            if (item.remaining_budget <= 0
                    or item.corrected_prefix != item.valid_prefix + (corrected,)
                    or item.valid_prefix != item.source_prefix
                       + item.proposals[:item.accepted_count]):
                raise ValueError("window contains a noncausal verification boundary")
            windows.append(PilotWindow(item, future,
                                       member["manifest_sha256"], False))
            kept += 1
            if kept == 8:
                break
    if (len(windows) != expected_windows
            or len({window.key for window in windows}) != len(windows)):
        raise ValueError(f"{split} must contain exactly {expected_windows} unique windows")
    return tuple(windows)


def index_sha256(index: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(index, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def load_original_raw_next(index: dict[str, Any], windows: Sequence[PilotWindow],
                           *, split: str) -> dict[tuple[str, int], OriginalRawNext]:
    """Join selected windows to the *actual* next Vanilla SD packed round.

    A RAW head replay is not allowed to substitute for this observed baseline.
    The caller can additionally replay Draft tokens to detect model drift.
    """
    if split not in ("train", "val") or len(index.get(split, [])) != (
            1000 if split == "train" else 200):
        raise ValueError("original raw trace requires the full repaired split")
    by_request: dict[str, list[PilotWindow]] = {}
    for window in windows:
        if window.observation.split != split:
            raise ValueError("selected window crosses original trace split")
        by_request.setdefault(window.observation.prompt_id, []).append(window)
    expected = {window.key for window in windows}
    if len(expected) != len(windows):
        raise ValueError("selected window keys repeat")
    result: dict[tuple[str, int], OriginalRawNext] = {}
    for member in index[split]:
        root = Path(member["root"]).resolve()
        request_id = member["request_id"]
        manifest_path = root / "manifests" / f"{request_id}.json"
        if (root.name != split or sha256_file(manifest_path)
                != member["manifest_sha256"]):
            raise ValueError("original packed manifest changed")
        boundary = load_boundary_inputs(root, manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        labels = _read_jsonl(_artifact(root, manifest["files"]["labels"]))
        if len(boundary) != len(labels):
            raise ValueError("original packed round/label counts differ")
        for window in by_request.get(member["prompt_id"], []):
            round_id = window.observation.round_index
            if not 0 <= round_id < len(boundary):
                raise ValueError("selected original packed round is absent")
            label = labels[round_id]
            next_id = label.get("next_round_id")
            if (type(next_id) is not int or next_id != round_id + 1
                    or next_id >= len(boundary)):
                raise ValueError("selected window lacks its actual Vanilla SD next round")
            next_round = boundary[next_id]
            proposals = tuple(next_round["draft_token_ids"])
            accepted = next_round["accepted_count"]
            if (tuple(next_round["prefix_token_ids"])
                    != window.observation.corrected_prefix
                    or label.get("next_draft_length") != len(proposals)
                    or label.get("next_accepted_count") != accepted
                    or not 1 <= len(proposals) <= 8
                    or not 0 <= accepted <= len(proposals)
                    or accepted != prefix_acceptance(proposals, window.labels)):
                raise ValueError("original packed acceptance differs from paired future")
            for position, (step, token) in enumerate(zip(
                    next_round["draft_steps"], proposals, strict=True), 1):
                if (step.get("position") != position
                        or step.get("token_id") != token
                        or step.get("proposal_prefix_sha256")
                           != digest(list(next_round["prefix_token_ids"])
                                     + list(proposals[:position - 1]))):
                    raise ValueError("original packed proposal forward is not exact")
            result[window.key] = OriginalRawNext(
                request_id, member["manifest_sha256"], proposals, accepted)
    if set(result) != expected:
        raise ValueError("original packed baseline is missing selected windows")
    return result


def load_original_raw_requests(index: dict[str, Any], *, split: str,
                               eos: int) -> dict[str, OriginalRawRequest]:
    """Load all rounds of each original packed request for closed-loop pairing."""
    if (split not in ("train", "val")
            or len(index.get(split, [])) != (1000 if split == "train" else 200)):
        raise ValueError("original SD requires the complete repaired split")
    result: dict[str, OriginalRawRequest] = {}
    for member in index[split]:
        root = Path(member["root"]).resolve()
        request_id = member["request_id"]
        manifest_path = root / "manifests" / f"{request_id}.json"
        if (root.name != split or sha256_file(manifest_path)
                != member["manifest_sha256"]):
            raise ValueError("original packed request manifest changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("prompt_id") != member["prompt_id"]
                or manifest.get("family_id") != member["family_id"]
                or manifest.get("source") != member["task_type"]
                or manifest.get("exact") is not True):
            raise ValueError("original packed request identity differs")
        boundary = load_boundary_inputs(root, manifest_path)
        oracle = json.loads(_artifact(root, manifest["files"]["oracle_tape"])
                            .read_text(encoding="utf-8"))
        future = tuple(oracle["token_ids"])
        first = tuple(boundary[0]["prefix_token_ids"])
        if (not first or digest(list(first)) != member["prompt_input_sha256"]
                or len(boundary) != manifest["rounds"]):
            raise ValueError("original packed prompt or rounds differ")
        proposals: list[tuple[int, ...]] = []
        accepted: list[int] = []
        cursor = first
        for round_id, row in enumerate(boundary):
            candidate = tuple(row["draft_token_ids"])
            count = row["accepted_count"]
            after = tuple(row["prefix_after_commit_token_ids"])
            offset = len(cursor) - len(first)
            committed = after[len(cursor):]
            if (tuple(row["prefix_token_ids"]) != cursor
                    or not 1 <= len(candidate) <= 8
                    or not 0 <= count <= len(candidate)
                    or count != prefix_acceptance(candidate, future[offset:])
                    or after[:len(cursor)] != cursor
                    or committed[:count] != candidate[:count]
                    or committed != future[offset:offset + len(committed)]
                    or len(committed) not in (count, count + 1)
                    or tuple(row["valid_prefix_token_ids"])
                       != cursor + candidate[:count]
                    or row["corrected_token_id"]
                       != (committed[-1] if len(committed) == count + 1 else None)):
                raise ValueError(f"original packed round {round_id} is not causal")
            for position, (step, token) in enumerate(zip(
                    row["draft_steps"], candidate, strict=True), 1):
                if (step.get("position") != position
                        or step.get("token_id") != token
                        or step.get("proposal_prefix_sha256")
                           != digest(list(cursor + candidate[:position - 1]))):
                    raise ValueError(f"original packed round {round_id} Draft step differs")
            proposals.append(candidate)
            accepted.append(count)
            cursor = after
        if (cursor[len(first):] != future
                or any(step.get("greedy_token_id") != token for step, token
                       in zip(oracle["steps"], future, strict=True))
                or not future or len(future) > 128
                or (future[-1] != eos and len(future) != 128)):
            raise ValueError("original packed output differs from independent Target AR")
        result[member["prompt_id"]] = OriginalRawRequest(
            member["prompt_id"], member["family_id"], member["task_type"],
            first, future, tuple(proposals), tuple(accepted),
            member["manifest_sha256"])
    if len(result) != len(index[split]):
        raise ValueError("original packed prompt IDs repeat")
    return result
