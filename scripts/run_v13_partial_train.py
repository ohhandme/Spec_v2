"""One seed42 B/C/F-H run on the audited qualified v12 train subset.

The original 1000-valid v13 train_ready gate remains false.  This is a
separate, user-authorized qualified-subset gate after all 1000 locked requests
have a terminal collection outcome.  The frozen collector is not modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file

from feedback_codec.fixed_probe import ProbeObservation
from feedback_codec.v10_four_arm import build_matched_v10_arms
from feedback_codec.v11_full_contract import PilotWindow, window_id, sha256_file
from feedback_codec.v12_acceptance import _artifact, _read_jsonl
from feedback_codec.v12_boundary_reader import load_boundary_inputs
from feedback_codec.v12_round_assembly import digest
from scripts import run_v11_full_pilot as training
from scripts import run_v12_collect_train as base
from scripts import run_v13_collect_continue as continuation
from scripts.run_v10_p1_wiring import atomic_json, state_digest

SEED = 42
ARMS = ("B", "C", "F-H")
EPOCHS = 3
MAX_GPU_SECONDS = 129600
MAX_SEGMENTS = 6
OVERRIDE_SHA256 = "9b5012af65eb38bbf3eba3ab389760eaff06254da7c57fa1e2a5d25771960492"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project_root", "v10_source_manifest", "v10_source_zip",
                 "v11_source_manifest", "v11_source_zip", "v12_source_manifest",
                 "v12_source_zip", "program", "contract", "backfill_manifest",
                 "schema_lock", "reference_validator", "parent", "parent_meta",
                 "v9_p0_report", "v9_inputs", "prior_plan", "target", "draft",
                 "prior_output", "output", "v13_program", "training_protocol",
                 "continuation_manifest", "override_receipt", "train_output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    for name in ("v10_source_sha256", "v11_source_sha256", "v12_source_sha256"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--deadline-seconds", type=int, default=21000)
    return parser.parse_args()


def _freeze(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"frozen receipt changed: {path}")
    else:
        atomic_json(path, value)


def _provenance(models: dict, meta: dict, files: dict, args, schema_sha: str) -> dict:
    return {
        "dataset_lock_sha256": base.PARENT_SHA,
        "execution_id": base.EXECUTION, "policy_id": base.POLICY,
        "draft_checkpoint_sha256": models["draft"]["model.safetensors"],
        "feedback_policy_sha256": base.digest({"policy": base.POLICY,
            "window": 8, "numeric_guard_top2_ulps": 2}),
        "seed": SEED, "model_sha256": models,
        "tokenizer_sha256": meta["tokenizer_sha256"],
        "chat_template_sha256": meta["tokenizer_sha256"]["target/chat_template"],
        "verifier_source_sha256": files["feedback_codec/packed.py"],
        "extension_source_sha256": args.v12_source_sha256,
        "observer_source_sha256": files["feedback_codec/v12_draft_observer.py"],
        "schema_lock_sha256": schema_sha,
        "collection_phase": "train_collection", "max_new_tokens": 128,
    }


def audit_qualified_subset(args, prompts, source_files, schema_sha,
                           provenance, train_output: Path) -> tuple[dict, tuple]:
    old, new = args.prior_output, args.output
    if (old / "active_worker.json").exists() or (new / "active_worker.json").exists():
        raise RuntimeError("collection worker still active")
    ledger_path = new / "collection_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if (ledger.get("state") != "collected_with_anomalies"
            or ledger.get("processed_train_requests") != 1000
            or ledger.get("training_updates") != 0
            or ledger.get("segments_used") != 5
            or ledger.get("gpu_seconds_charged", 0) >= MAX_GPU_SECONDS):
        raise ValueError("1000 terminal requests or remaining budget not established")
    index_path = new / "quarantine_index.json"
    quarantine = json.loads(index_path.read_text(encoding="utf-8"))
    reported_errors = {row["request_id"] for row in quarantine["mismatches"]}
    if len(reported_errors) != ledger["quarantined_mismatch_requests"]:
        raise ValueError("quarantine index does not match ledger")
    qualified, errors = [], []
    source_counts, family_counts = Counter(), Counter()
    seen_ids = set()
    for prompt in prompts:
        rid = base.request_id(base.PARENT_SHA, "train", prompt.id,
                              base.POLICY, base.EXECUTION)
        if rid in seen_ids:
            raise ValueError("duplicate locked request")
        seen_ids.add(rid)
        old_root, new_root = old / "train", new / "train"
        old_m = old_root / "manifests" / f"{rid}.json"
        old_f = old_root / "audit" / "failures" / f"{rid}.json"
        new_m = new_root / "manifests" / f"{rid}.json"
        new_f = new_root / "audit" / "failures" / f"{rid}.json"
        new_e = new_root / "audit" / "request_failures" / f"{rid}.json"
        if old_m.exists():
            if old_f.exists() or new_m.exists() or new_f.exists() or new_e.exists():
                raise ValueError(f"duplicate old request outcome: {rid}")
            root = old_root
        elif old_f.exists():
            if new_m.exists() or new_f.exists() or new_e.exists():
                raise ValueError(f"duplicate first failure: {rid}")
            errors.append(continuation.checked_failure(old_f, prompt, rid))
            errors[-1]["category"] = "numeric_mismatch"
            continue
        elif new_e.exists():
            errors.append(continuation.checked_request_error(
                new_e, new_root, prompt, rid, args.v12_source_sha256, schema_sha))
            raw = json.loads(new_e.read_text(encoding="utf-8"))
            message = (raw.get("error_type", "") + " " + raw.get("error", "")).lower()
            errors[-1]["category"] = (
                "numeric_mismatch" if type(raw.get("first_difference_index")) is int
                else "oom" if "out of memory" in message or "outofmemory" in message
                else "other")
            continue
        elif new_m.exists():
            if new_f.exists():
                raise ValueError(f"manifest plus exact failure without receipt: {rid}")
            root = new_root
        else:
            raise ValueError(f"locked request lacks terminal outcome: {rid}")
        row = base.audit_one(root, prompt, args, source_files, schema_sha, provenance)
        row.update({"source": prompt.task_type, "family_id": prompt.family_id,
                    "root": str(root)})
        qualified.append(row)
        source_counts[prompt.task_type] += 1
        family_counts[prompt.family_id] += 1
    if (len(seen_ids) != 1000 or len(qualified) + len(errors) != 1000
            or len(qualified) != ledger["fully_valid_train_requests"]
            or {e["request_id"] for e in errors} != reported_errors):
        raise ValueError("qualified/error partition or counts differ")
    prompt_by_id = {p.id: p for p in prompts}
    for error in errors:
        prompt = prompt_by_id[error["prompt_id"]]
        error["source"] = prompt.task_type
        error["family_id"] = prompt.family_id
    category_counts = Counter(e["category"] for e in errors)
    qualified_by_prompt = {q["prompt_id"]: q for q in qualified}
    receipt = {
        "scope": "user_authorized_v13_qualified_subset_after_locked1000",
        "original_v13_train_ready": False,
        "qualified_subset_ready": True,
        "v13_program_sha256": sha256_file(args.v13_program),
        "training_protocol_sha256": sha256_file(args.training_protocol),
        "user_override_receipt_sha256": sha256_file(args.override_receipt),
        "dataset_lock_sha256": base.PARENT_SHA,
        "schema_lock_sha256": schema_sha,
        "v12c_source_zip_sha256": args.v12_source_sha256,
        "continuation_driver_sha256": sha256_file(Path(continuation.__file__)),
        "collection_ledger_sha256": sha256_file(ledger_path),
        "quarantine_index_sha256": sha256_file(index_path),
        "processed": 1000, "qualified": len(qualified), "errors": len(errors),
        "error_category_counts": dict(sorted(category_counts.items())),
        "optimizer_updates_before_gate": 0,
        "qualified_source_counts": dict(sorted(source_counts.items())),
        "qualified_family_count": len(family_counts),
        "qualified_members": qualified, "errors_inventory": errors,
        "qualified_members_digest_sha256": digest(qualified),
        "excluded_error_ids": sorted(reported_errors),
        "training_seeds": [SEED], "training_arms": list(ARMS),
        "epochs_per_arm": EPOCHS,
    }
    if receipt["user_override_receipt_sha256"] != OVERRIDE_SHA256:
        raise ValueError("user partial-training override receipt differs")
    path = train_output / "qualified_subset_acceptance.json"
    _freeze(path, receipt)
    if json.loads(path.read_text(encoding="utf-8")) != receipt:
        raise ValueError("qualified-subset acceptance changed after freeze")
    return receipt, tuple((p, qualified_by_prompt[p.id]) for p in prompts
                          if p.id in qualified_by_prompt)


def load_windows(members: tuple, eos: int) -> tuple[PilotWindow, ...]:
    windows = []
    for prompt, record in members:
        root = Path(record["root"])
        mpath = root / "manifests" / f'{record["request_id"]}.json'
        if sha256_file(mpath) != record["manifest_sha256"]:
            raise ValueError("qualified manifest changed after acceptance")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        boundary = load_boundary_inputs(root, mpath)
        labels_path = _artifact(root, manifest["files"]["labels"])
        labels = _read_jsonl(labels_path)
        hidden_entry = manifest["files"].get("hidden")
        hidden = load_file(str(_artifact(root, hidden_entry)), device="cpu") if hidden_entry else {}
        if len(boundary) != len(labels):
            raise ValueError("observation/label row count differs")
        kept = 0
        for round_id, (o, label) in enumerate(zip(boundary, labels, strict=True)):
            future = tuple(label["target_future"]["token_ids"])
            corrected = o["corrected_token_id"]
            if corrected is None or corrected == eos or not future:
                continue
            ref = o["target_valid_hidden_ref"]
            if ref is None or ref["key"] not in hidden:
                raise ValueError("trainable boundary lacks legal teacher hidden")
            observation = ProbeObservation(
                prompt_id=prompt.id, family_id=prompt.family_id,
                split="train", task_type=prompt.task_type, round_index=round_id,
                source_prefix=tuple(o["prefix_token_ids"]),
                proposals=tuple(o["draft_token_ids"]),
                valid_prefix=tuple(o["valid_prefix_token_ids"]),
                accepted_count=o["accepted_count"],
                rejected_token=o["rejected_token_id"],
                correction_token=corrected,
                corrected_prefix=tuple(o["prefix_after_commit_token_ids"]),
                target_hidden=hidden[ref["key"]].detach().clone(),
                target_hidden_prefix=tuple(o["valid_prefix_token_ids"]),
                remaining_budget=o["remaining_budget_after"], window=8,
            )
            if (not 1 <= len(future) <= 8 or observation.remaining_budget <= 0
                    or label["round_id"] != round_id):
                raise ValueError("future label or boundary eligibility differs")
            windows.append(PilotWindow(observation, future,
                                       record["manifest_sha256"], False))
            kept += 1
            if kept == 8:
                break
    if not 1 <= len(windows) <= 8000 or len({w.key for w in windows}) != len(windows):
        raise ValueError("qualified H1 window count violates contract")
    return tuple(windows)


def lock_training(output: Path, windows, config, draft_model,
                  acceptance_sha: str) -> dict:
    arms = build_matched_v10_arms(
        config, draft_model.get_output_embeddings(), seed=SEED,
        draft_embedding=draft_model.get_input_embeddings())
    hashes = {mode: state_digest(arms[mode].trainable_state_dict()) for mode in ARMS}
    if len(set(hashes.values())) != 1 or any(
            arms[mode].parameter_budget()["edge_trainable"] != 7643875 for mode in ARMS):
        raise ValueError("matched arm initialization or capacity differs")
    del arms
    n = len(windows)
    order = []
    for epoch in range(EPOCHS):
        indices = list(range(n))
        random.Random(SEED * 1000 + epoch).shuffle(indices)
        order.append(indices)
    lock = {
        "scope": "v13_qualified_subset_three_arm_single_seed",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "qualified_subset_acceptance_sha256": acceptance_sha,
        "train_N": n, "window_ids": [window_id(w) for w in windows],
        "label_lengths": [len(w.labels) for w in windows],
        "seeds": [SEED], "arms": list(ARMS), "epochs": EPOCHS,
        "epoch_order_indices": {str(SEED): order},
        "gumbel_seed_formula": "training_seed*1000000 + epoch*N + position + 1",
        "initial_trainable_sha256": {str(SEED): hashes[ARMS[0]]},
        "parameter_count_each": 7643875, "optimizer": "AdamW",
        "learning_rate": 1e-4, "weight_decay": .01,
        "betas": [.9, .999], "epsilon": 1e-8,
        "temperature": 1.0, "batch_windows": 1,
        "train_curve_window_indices": list(range(min(64, n))),
        "final_checkpoint": "epoch3", "select_epoch_on_val": False,
        "validation_read_before_lock": False,
    }
    _freeze(output / "train_lock.json", lock)
    return lock


def main() -> None:
    args = parse_args()
    if not 0 < args.deadline_seconds <= 21000:
        raise ValueError("training segment deadline exceeds six-hour bound")
    train_output = args.train_output.resolve()
    if not train_output.is_relative_to(args.project_root.resolve()):
        raise ValueError("training output outside isolated project")
    train_output.mkdir(parents=True, exist_ok=True)
    meta, models, files, schema_sha = base.check_source(args)
    prompts = base.locked_prompts(args.parent, args.parent_meta)["train"]
    args.legacy_ids = base.check_backfill(args, prompts)
    continuation.check_continuation(args, prompts)
    provenance = _provenance(models, meta, files, args, schema_sha)
    acceptance, members = audit_qualified_subset(
        args, prompts, files, schema_sha, provenance, train_output)
    acceptance_path = train_output / "qualified_subset_acceptance.json"
    acceptance_sha = sha256_file(acceptance_path)
    if acceptance["qualified"] != len(members):
        raise ValueError("qualified subset member lookup differs")
    windows = load_windows(members, meta["target_eos_token_id"])
    if sha256_file(acceptance_path) != acceptance_sha:
        raise ValueError("acceptance changed while loading labels")
    previous = json.loads((args.output / "collection_ledger.json").read_text())
    if previous["segments_used"] >= MAX_SEGMENTS:
        raise ValueError("no GPU segment remains")
    remaining = MAX_GPU_SECONDS - previous["gpu_seconds_charged"]
    deadline = min(args.deadline_seconds, remaining)
    if deadline <= 900:
        raise ValueError("insufficient authorized GPU time to start training")
    active = train_output / "active_worker.json"
    if active.exists() or (args.output / "active_worker.json").exists():
        raise RuntimeError("a collection or training worker is active")
    status = {"state": "starting", "phase": "gpu_admission",
              "segment": previous["segments_used"] + 1,
              "qualified_subset_acceptance_sha256": acceptance_sha,
              "training_updates": 0, "seeds": [SEED], "arms": list(ARMS)}
    atomic_json(train_output / "status.json", status)
    admission = base._admit_idle_gpu("cuda:0", 22000)
    status["gpu_admission"] = admission
    with active.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "segment": status["segment"],
                   "acceptance_sha256": acceptance_sha}, f)
        f.flush(); os.fsync(f.fileno())
    begin = time.monotonic()
    training_ledger = {
        "scope": "v13_qualified_subset_three_arm_training",
        "prior_collection_segments": previous["segments_used"],
        "prior_collection_gpu_seconds": previous["gpu_seconds_charged"],
        "qualified_subset_acceptance_sha256": acceptance_sha,
        "training_updates": 0, "state": "running",
    }
    path = train_output / "pilot_ledger.json"
    if path.exists():
        raise ValueError("existing training ledger requires separate resume audit")
    atomic_json(path, training_ledger)
    try:
        status["phase"] = "load_frozen_models"; atomic_json(train_output / "status.json", status)
        target, draft, draft_model, config = base._load_models(args, meta["target_eos_token_id"])
        del target
        lock = lock_training(train_output, windows, config, draft_model, acceptance_sha)
        if sha256_file(acceptance_path) != acceptance_sha:
            raise ValueError("acceptance changed before optimizer")
        status["phase"] = "train"; atomic_json(train_output / "status.json", status)
        training.SEEDS = (SEED,)
        training.ARM_ORDER = ARMS
        training.PROGRAM_SHA256 = acceptance_sha
        finished = training._train_all(
            args, train_output, windows, config, draft, draft_model, lock,
            meta["target_eos_token_id"],
            lambda: deadline - (time.monotonic() - begin), training_ledger)
        status["state"] = "trained_three_arms" if finished else "paused_budget"
        training_ledger["state"] = status["state"]
    except Exception as exc:
        status.update({"state": "failed", "error_type": type(exc).__name__,
                       "error": str(exc)})
        training_ledger["state"] = "failed"
        raise
    finally:
        elapsed = time.monotonic() - begin
        training_ledger["gpu_seconds_charged_this_segment"] = elapsed
        training_ledger["cumulative_gpu_seconds"] = previous["gpu_seconds_charged"] + elapsed
        training_ledger["segments_used_total"] = previous["segments_used"] + 1
        atomic_json(path, training_ledger)
        status.update({"elapsed_seconds": elapsed,
                       "training_updates": training_ledger["training_updates"],
                       "cumulative_gpu_seconds": training_ledger["cumulative_gpu_seconds"]})
        atomic_json(train_output / "status.json", status)
        active.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
