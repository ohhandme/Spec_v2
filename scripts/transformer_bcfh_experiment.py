"""Train/evaluate B, C, F-H Transformer heads on *already audited* v13 windows.

This is a local library, not a job dispatcher.  The caller supplies the frozen
Draft backend and the exact repaired 1000/200 PilotWindow lists.  No Target
model is loaded here.  Validation stays unread until all three training arms
have completed; the caller owns GPU admission and cross-root flock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from feedback_codec import codec as codec_source
from feedback_codec import v10_four_arm as arm_source
from feedback_codec import v10_transformer_bcfh as transformer_source
from feedback_codec.codec import CodecConfig
from feedback_codec.v10_transformer_bcfh import (
    ARMS, TransformerBCFHHead, TransformerRepairConfig,
    build_matched_transformer_arms,
)
from feedback_codec.v11_full_contract import PilotWindow
from feedback_codec.v11_stats import (
    full_window_eligible, paired_family_ci, prefix_acceptance,
)
from scripts.load_transformer_v13_windows import OriginalRawNext


@dataclass(frozen=True)
class ExperimentLock:
    seed: int = 42
    epochs: int = 5
    train_prompts: int = 1000
    train_windows: int = 7982
    val_prompts: int = 200
    val_windows: int = 1598
    depth: int = 2
    width: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    checkpoint_every: int = 256
    bootstrap_draws: int = 10000
    bootstrap_seed: int = 20261110

    def choice(self) -> TransformerRepairConfig:
        return TransformerRepairConfig(depth=self.depth, width=self.width)


def _digest_json(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _state_digest(state: dict[str, Tensor]) -> str:
    sha = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        sha.update(name.encode("utf-8"))
        sha.update(str(tuple(value.shape)).encode("ascii"))
        sha.update(str(value.dtype).encode("ascii"))
        sha.update(value.numpy().tobytes())
    return sha.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = {"runner": Path(__file__),
             "window_reader": Path(__file__).with_name("load_transformer_v13_windows.py"),
             "closed_loop": Path(__file__).with_name("transformer_bcfh_closed_loop.py"),
             "transformer": Path(transformer_source.__file__),
             "arms": Path(arm_source.__file__),
             "codec": Path(codec_source.__file__)}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()}


def _draft_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    """Cheap runtime identity check, in addition to the caller's asset hash."""
    named = list(model.named_parameters())
    if not named or any(parameter.requires_grad for _, parameter in named):
        raise ValueError("Draft parameters are missing or trainable")
    sample: dict[str, str] = {}
    for name, parameter in (named[0], named[len(named) // 2], named[-1]):
        flat = parameter.detach().flatten()
        data = torch.cat((flat[:64], flat[-64:])).float().cpu().numpy().tobytes()
        sample[name] = hashlib.sha256(data).hexdigest()
    return {"parameter_count": len(named),
            "parameter_elements": sum(p.numel() for _, p in named),
            "sample_sha256": sample}


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def audit_windows(windows: Sequence[PilotWindow], *, split: str,
                  lock: ExperimentLock) -> dict[str, Any]:
    """Enforce the new complete 1000/200 corpus, never the old 984 subset."""
    expected_windows = lock.train_windows if split == "train" else lock.val_windows
    expected_prompts = lock.train_prompts if split == "train" else lock.val_prompts
    if split not in ("train", "val") or len(windows) != expected_windows:
        raise ValueError(f"{split} must contain exactly {expected_windows} accepted windows")
    keys: set[tuple[str, int]] = set()
    prompts: set[str] = set()
    manifest_hashes: set[str] = set()
    for window in windows:
        if not isinstance(window, PilotWindow) or window.observation.split != split:
            raise ValueError("window split/type differs from locked corpus")
        observation = window.observation
        if (window.key in keys or not 1 <= len(window.labels) <= 8
                or observation.window != 8 or not observation.corrected_prefix
                or observation.target_hidden_prefix != observation.valid_prefix
                or observation.target_hidden.ndim != 1
                or not torch.isfinite(observation.target_hidden).all()
                or not isinstance(window.chunk_sha256, str)
                or len(window.chunk_sha256) != 64):
            raise ValueError("duplicate, malformed or causally invalid window")
        keys.add(window.key)
        prompts.add(observation.prompt_id)
        manifest_hashes.add(window.chunk_sha256)
    if len(prompts) != expected_prompts:
        raise ValueError(f"{split} must contain exactly {expected_prompts} prompts")
    # The source inventory itself is kept in the separately audited input
    # receipt.  This digest binds the exposure to prompt/round/labels and each
    # immutable per-request manifest, without embedding hidden tensors.
    digest = _digest_json(sorted((w.observation.prompt_id,
                                  w.observation.round_index, w.chunk_sha256,
                                  list(w.labels)) for w in windows))
    return {"split": split, "prompts": len(prompts), "windows": len(windows),
            "manifest_count": len(manifest_hashes), "exposure_sha256": digest,
            "prompt_ids": sorted(prompts)}


def _head(config: CodecConfig, draft_model: torch.nn.Module,
          lock: ExperimentLock, mode: str, device: torch.device,
          ) -> TransformerBCFHHead:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(lock.seed)
        head = TransformerBCFHHead(
            config, draft_model.get_output_embeddings(), mode=mode,
            choice=lock.choice(),
            draft_embedding=draft_model.get_input_embeddings())
    return head.to(device)


def _train_step(head: TransformerBCFHHead, draft: Any, window: PilotWindow,
                optimizer: torch.optim.Optimizer, step_seed: int) -> float:
    observation, labels = window.observation, window.labels
    with torch.no_grad():
        history = draft.teacher_hidden(list(observation.corrected_prefix),
                                       list(labels))
    if history.shape != (len(labels), head.config.draft_hidden_size):
        raise ValueError("frozen Draft teacher history has wrong shape")
    state = head.prepare_observation(observation, draft)
    positions = torch.arange(1, len(labels) + 1, device=history.device)
    torch.manual_seed(step_seed)
    optimizer.zero_grad(set_to_none=True)
    logits = head.forward_logits(history.detach().unsqueeze(0), state,
                                 positions, temperature=1.0)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=logits.device)
    per_token = F.cross_entropy(logits.float().reshape(len(labels), -1),
                                labels_tensor, reduction="none")
    weights = torch.arange(8, 8 - len(labels), -1,
                           dtype=per_token.dtype, device=per_token.device)
    loss = (weights * per_token).sum() / weights.sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite weighted CE")
    loss.backward()
    gradients = [p.grad for p in head.parameters() if p.requires_grad]
    if not gradients or any(g is None or not torch.isfinite(g).all()
                            for g in gradients):
        raise FloatingPointError("missing or nonfinite trainable gradients")
    optimizer.step()
    return float(loss.detach())


def train_three_arms(train_windows: Sequence[PilotWindow], draft: Any,
                     draft_model: torch.nn.Module, config: CodecConfig,
                     output: Path, *, source_receipt_sha256: str,
                     draft_checkpoint_sha256: str,
                     lock: ExperimentLock = ExperimentLock(),
                     max_updates_this_call: int | None = None,
                     progress: Callable[[dict[str, Any]], None] | None = None,
                     ) -> dict[str, Any]:
    """Five equal-exposure epochs per arm; resume only this isolated output.

    Caller must attest the 1000-request collection audit before calling this
    function.  The receipt hash is stored in every checkpoint.  No validation
    examples are opened here, and no old MLP checkpoint is reused.
    """
    if not torch.cuda.is_available() or not str(draft.device).startswith("cuda"):
        raise RuntimeError("formal training requires an admitted CUDA Draft")
    if draft.model is not draft_model or any(p.requires_grad for p in draft_model.parameters()):
        raise ValueError("supplied Draft model differs or is not frozen")
    if (any(len(value) != 64 or any(char not in "0123456789abcdef"
                                   for char in value)
            for value in (source_receipt_sha256, draft_checkpoint_sha256))
            or max_updates_this_call is not None
            and max_updates_this_call <= 0):
        raise ValueError("source receipt or update budget differs")
    exposure = audit_windows(train_windows, split="train", lock=lock)
    if output.exists() and any(output.iterdir()) and not (
            output / "transformer_experiment_lock.json").exists():
        raise FileExistsError("new Transformer output must be empty and isolated")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "transformer_experiment_lock.json"
    lock_record = {"protocol": "transformer_bcfh_v1",
                   "lock": asdict(lock), "source_receipt_sha256": source_receipt_sha256,
                   "draft_checkpoint_sha256": draft_checkpoint_sha256,
                   "draft_runtime_fingerprint": _draft_fingerprint(draft_model),
                   "codec_config": asdict(config),
                   "train_exposure_sha256": exposure["exposure_sha256"],
                   "source_sha256": _source_hashes()}
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock_record:
            raise ValueError("isolated Transformer lock changed on resume")
    else:
        _atomic_json(lock_path, lock_record)
    lock_sha = _digest_json(lock_record)
    if (output / "training_terminal.json").exists():
        terminal = json.loads((output / "training_terminal.json").read_text(encoding="utf-8"))
        if terminal.get("lock_sha256") != lock_sha:
            raise ValueError("terminal training lock differs")
        return terminal
    device = draft.device
    initial = build_matched_transformer_arms(
        config, draft_model.get_output_embeddings(), seed=lock.seed,
        choice=lock.choice(), draft_embedding=draft_model.get_input_embeddings())
    initial_sha = _state_digest(initial["B"].trainable_state_dict())
    budget_each = initial["B"].parameter_budget()
    del initial
    progress_path = output / "progress.pt"
    saved = (torch.load(progress_path, map_location="cpu", weights_only=False)
             if progress_path.exists() else None)
    if saved is not None and (saved.get("lock_sha256") != lock_sha
                              or saved.get("initial_sha256") != initial_sha):
        raise ValueError("Transformer resume source or initialization differs")
    arm_index = saved["arm_index"] if saved else 0
    completed = {mode for mode in ARMS if (output / f"{mode}.final.pt").exists()}
    if completed != set(ARMS[:arm_index]):
        raise RuntimeError("completed-arm checkpoints and progress need an audit")
    epoch = saved["epoch"] if saved else 0
    cursor = saved["cursor"] if saved else 0
    updates = saved["updates"] if saved else 0
    total_loss = saved["total_loss"] if saved else 0.0
    this_call = 0
    for index in range(arm_index, len(ARMS)):
        mode = ARMS[index]
        head = _head(config, draft_model, lock, mode, device)
        if _state_digest(head.trainable_state_dict()) != initial_sha:
            raise ValueError("cross-arm initialization differs")
        optimizer = torch.optim.AdamW(
            (p for p in head.parameters() if p.requires_grad),
            lr=lock.learning_rate, weight_decay=lock.weight_decay)
        if saved is not None and index == arm_index:
            head.load_trainable_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
        else:
            epoch = cursor = 0
            total_loss = 0.0
        for e in range(epoch, lock.epochs):
            order = list(range(len(train_windows)))
            random.Random(lock.seed * 1000 + e).shuffle(order)
            start = cursor if e == epoch else 0
            for position in range(start, len(order)):
                step_seed = (lock.seed * 1_000_000 + e * len(order)
                             + position + 1)
                total_loss += _train_step(head, draft,
                                          train_windows[order[position]],
                                          optimizer, step_seed)
                updates += 1
                this_call += 1
                next_epoch, next_cursor = (e + 1, 0) if position + 1 == len(order) else (e, position + 1)
                if (updates % lock.checkpoint_every == 0
                        or max_updates_this_call is not None
                        and this_call >= max_updates_this_call):
                    _atomic_torch(progress_path, {
                        "lock_sha256": lock_sha, "initial_sha256": initial_sha,
                        "arm_index": index, "epoch": next_epoch,
                        "cursor": next_cursor, "updates": updates,
                        "total_loss": total_loss,
                        "model": head.trainable_state_dict(),
                        "optimizer": optimizer.state_dict()})
                    if progress is not None:
                        progress({"arm": mode, "epoch": next_epoch,
                                  "cursor": next_cursor, "updates": updates})
                    if max_updates_this_call is not None and this_call >= max_updates_this_call:
                        return {"state": "paused", "arm": mode,
                                "updates": updates, "lock_sha256": lock_sha}
            cursor = 0
        _atomic_torch(output / f"{mode}.final.pt", {
            "protocol": "transformer_bcfh_v1", "mode": mode,
            "lock_sha256": lock_sha, "initial_sha256": initial_sha,
            "trainable_state": head.trainable_state_dict(),
            "updates_this_arm": lock.epochs * len(train_windows),
            "mean_weighted_ce": total_loss / (lock.epochs * len(train_windows))})
        saved = None
        epoch = cursor = 0
        total_loss = 0.0
        if index + 1 < len(ARMS):
            next_head = _head(config, draft_model, lock, ARMS[index + 1], device)
            next_optimizer = torch.optim.AdamW(
                (p for p in next_head.parameters() if p.requires_grad),
                lr=lock.learning_rate, weight_decay=lock.weight_decay)
            _atomic_torch(progress_path, {
                "lock_sha256": lock_sha, "initial_sha256": initial_sha,
                "arm_index": index + 1, "epoch": 0, "cursor": 0,
                "updates": updates, "total_loss": 0.0,
                "model": next_head.trainable_state_dict(),
                "optimizer": next_optimizer.state_dict()})
    terminal = {"state": "trained", "lock_sha256": lock_sha,
                "initial_sha256": initial_sha,
                "updates_per_arm": lock.epochs * len(train_windows),
                "updates_total": updates, "parameter_budget": budget_each,
                "arms": list(ARMS), "validation_read": False}
    _atomic_json(output / "training_terminal.json", terminal)
    return terminal


def _free(head: TransformerBCFHHead | None, draft: Any,
          window: PilotWindow, eos: int) -> list[int]:
    observation = window.observation
    state = head.prepare_observation(observation, draft) if head else None
    prefix = list(observation.corrected_prefix)
    proposed: list[int] = []
    for position in range(1, 9):
        if head is None:
            hidden = draft.teacher_hidden(prefix, [0])
            token = int(draft.model.get_output_embeddings()(hidden)[0].argmax(-1))
        else:
            token = head.next_token(prefix, state, position, draft)
        proposed.append(token)
        prefix.append(token)
        if token == eos:
            break
    return proposed


def _family_paired_delta(rows: Sequence[dict[str, Any]], arm: str,
                         lock: ExperimentLock) -> dict[str, Any]:
    """Equal prompt means, paired family bootstrap within each source."""
    from collections import defaultdict

    grouped: dict[tuple[int, str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["eligible"]:
            for mode in ("ORIGINAL_SD", arm):
                key = (lock.seed, row["source"], row["family_id"],
                       row["prompt_id"], mode)
                grouped[key].append(float(row["acceptance"][mode]))
    if not grouped:
        raise ValueError("no eligible full-window validation prompts")
    scores = {key: sum(values) / len(values) for key, values in grouped.items()}
    result = paired_family_ci(scores, seed=lock.seed, left=arm,
                              right="ORIGINAL_SD",
                              draws=lock.bootstrap_draws,
                              bootstrap_seed=lock.bootstrap_seed)
    return {**result, "positive_point_estimate": result["point"] > 0,
            "fixed_prefix_confirmed_positive": result["ci95"][0] > 0}


def _acceptance_profile(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Prompt-equal absolute lengths and L1..L8 survival, eligible rows only."""
    from collections import defaultdict

    grouped: dict[str, list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        if row["eligible"]:
            grouped[row["prompt_id"]].append(row["acceptance"])
    if not grouped:
        raise ValueError("no full-window validation rows")
    modes = ("ORIGINAL_SD",) + ARMS
    result: dict[str, Any] = {}
    for mode in modes:
        prompt_means = [sum(r[mode] for r in values) / len(values)
                        for values in grouped.values()]
        survival = [sum(sum(r[mode] >= k for r in values) / len(values)
                        for values in grouped.values()) / len(grouped)
                    for k in range(1, 9)]
        result[mode] = {"mean_accepted_tokens": sum(prompt_means) / len(prompt_means),
                        "pr_accept_at_least_1_to_8": survival}
    return result


@torch.inference_mode()
def evaluate_three_arms(val_windows: Sequence[PilotWindow], draft: Any,
                        draft_model: torch.nn.Module, config: CodecConfig,
                        output: Path, *, eos: int,
                        original_raw_next: Mapping[tuple[str, int], OriginalRawNext],
                        lock: ExperimentLock = ExperimentLock(),
                        ) -> dict[str, Any]:
    """Compare to the recorded original packed SD next round on 200 prompts."""
    terminal_path = output / "training_terminal.json"
    if not terminal_path.is_file():
        raise RuntimeError("all three Transformer arms must finish before validation")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("state") != "trained":
        raise ValueError("training terminal receipt is not successful")
    if draft.model is not draft_model or any(p.requires_grad for p in draft_model.parameters()):
        raise ValueError("supplied Draft model differs or is not frozen")
    receipt = audit_windows(val_windows, split="val", lock=lock)
    if set(original_raw_next) != {window.key for window in val_windows}:
        raise ValueError("recorded original SD baseline must cover every selected window")
    lock_record = json.loads((output / "transformer_experiment_lock.json").read_text(encoding="utf-8"))
    if (_digest_json(lock_record) != terminal["lock_sha256"]
            or lock_record["lock"] != asdict(lock)
            or lock_record["source_sha256"] != _source_hashes()
            or lock_record["codec_config"] != asdict(config)
            or lock_record["draft_runtime_fingerprint"]
               != _draft_fingerprint(draft_model)):
        raise ValueError("evaluation source/architecture differs from training")
    if (output / "validation_summary.json").exists():
        raise FileExistsError("preserve the existing validation result")
    heads: dict[str, TransformerBCFHHead] = {}
    for mode in ARMS:
        checkpoint = torch.load(output / f"{mode}.final.pt", map_location="cpu",
                                weights_only=False)
        if (checkpoint.get("lock_sha256") != terminal["lock_sha256"]
                or checkpoint.get("initial_sha256") != terminal["initial_sha256"]
                or checkpoint.get("updates_this_arm") != lock.epochs * lock.train_windows):
            raise ValueError(f"{mode} checkpoint mismatches the frozen training lock")
        head = _head(config, draft_model, lock, mode, draft.device)
        head.load_trainable_state_dict(checkpoint["trainable_state"])
        heads[mode] = head.eval()
    rows: list[dict[str, Any]] = []
    for window in val_windows:
        original = original_raw_next[window.key]
        if (not isinstance(original, OriginalRawNext)
                or original.manifest_sha256 != window.chunk_sha256
                or original.accepted_count
                   != prefix_acceptance(original.proposals, window.labels)):
            raise ValueError("recorded original SD round differs from selected window")
        raw_replay = _free(None, draft, window, eos)
        if tuple(raw_replay[:len(original.proposals)]) != original.proposals:
            raise ValueError("frozen Draft replay differs from original packed proposal")
        proposed = {"ORIGINAL_SD": list(original.proposals)}
        proposed.update({mode: _free(head, draft, window, eos)
                         for mode, head in heads.items()})
        rows.append({
            "prompt_id": window.observation.prompt_id,
            "family_id": window.observation.family_id,
            "source": window.observation.task_type,
            "round_index": window.observation.round_index,
            "eligible": full_window_eligible(window.labels, eos),
            "target_future_ids": list(window.labels),
            "proposed_ids": proposed,
            "original_sd_request_id": original.request_id,
            "original_sd_manifest_sha256": original.manifest_sha256,
            "original_sd_actual_accepted_count": original.accepted_count,
            "acceptance": {mode: prefix_acceptance(ids, window.labels)
                           for mode, ids in proposed.items()},
        })
    score_path = output / "validation_scores.jsonl"
    if score_path.exists():
        raise FileExistsError("preserve the existing validation score rows")
    with score_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    comparisons = {mode: _family_paired_delta(rows, mode, lock) for mode in ARMS}
    summary = {"scope": "transformer_bcfh_v1_repaired_1000_200_original_packed_next_round",
               "train_lock_sha256": terminal["lock_sha256"],
               "val_exposure_sha256": receipt["exposure_sha256"],
               "val_score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
               "eligible_windows": sum(row["eligible"] for row in rows),
               "acceptance_profile": _acceptance_profile(rows),
               "comparisons": comparisons,
               "formal_goal_requires_closed_loop": True,
               "interpretation": "Paired fixed-prefix next-round acceptance versus recorded "
                                 "original packed Vanilla SD, conditional on one training seed; "
                                 "not an end-to-end request, speed or WAN claim. F-H is an "
                                 "uncompressed Target diagnostic upper bound."}
    _atomic_json(output / "validation_summary.json", summary)
    return summary
