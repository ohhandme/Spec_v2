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
from feedback_codec.v12_acceptance import (
    _artifact, _read_jsonl, _schema_validator, PROVENANCE_KEYS,
)
from feedback_codec.v12_boundary_reader import load_boundary_inputs
from feedback_codec.v12_round_assembly import digest
from feedback_codec.v11_stats import paired_family_ci, prompt_scores, full_window_eligible
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


def load_windows(members: tuple, eos: int, *, split: str = "train") -> tuple[PilotWindow, ...]:
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
                split=split, task_type=prompt.task_type, round_index=round_id,
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


def audit_val_one(root: Path, prompt, args, schema_sha: str,
                  provenance: dict) -> dict:
    """Audit the independent AR stream, every val row and referenced byte file."""
    if root.name != "val" or prompt.split != "val":
        raise ValueError("val split boundary differs")
    rid = base.request_id(base.PARENT_SHA, "val", prompt.id,
                          base.POLICY, base.EXECUTION)
    mpath = root / "manifests" / f"{rid}.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    if (m.get("request_id") != rid or m.get("prompt_id") != prompt.id
            or m.get("family_id") != prompt.family_id
            or m.get("source") != prompt.task_type
            or m.get("split") != "val" or m.get("exact") is not True
            or m.get("schema_lock_sha256") != schema_sha
            or m.get("extension_source_sha256") != args.v12_source_sha256
            or m.get("dataset_lock_sha256") != base.PARENT_SHA):
        raise ValueError("val manifest identity/source/exact gate failed")
    files = m["files"]
    pv = m.get("provenance")
    expected = {**provenance, "prompt_id": prompt.id,
                "family_id": prompt.family_id, "source": prompt.task_type,
                "split": "val", "collection_phase": "val_collection"}
    if (not isinstance(pv, dict) or not set(PROVENANCE_KEYS).issubset(pv)
            or any(pv.get(k) != value for k, value in expected.items())
            or pv.get("exact_stream_receipt_sha256") != files["audit"]["sha256"]):
        raise ValueError("val run provenance differs")
    paths = {k: _artifact(root, files[k]) for k in
             ("observations", "labels", "oracle_tape", "audit")}
    hidden_path = _artifact(root, files["hidden"]) if files.get("hidden") else None
    hidden = load_file(str(hidden_path), device="cpu") if hidden_path else {}
    obs, labs = _read_jsonl(paths["observations"]), _read_jsonl(paths["labels"])
    tape = json.loads(paths["oracle_tape"].read_text(encoding="utf-8"))
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    if (len(obs) != len(labs) or len(obs) != m.get("rounds")
            or sum(len(x["draft_steps"]) for x in obs) != m.get("draft_positions")
            or audit.get("exact") is not True
            or audit.get("packed_ids") != audit.get("oracle_ids")
            or audit.get("packed_eos") != audit.get("oracle_eos")
            or audit.get("packed_truncated") != audit.get("oracle_truncated")
            or audit.get("first_difference_index") is not None
            or tape.get("token_ids") != audit.get("oracle_ids")
            or tape.get("request_id") != rid
            or tape.get("schema_lock_sha256") != schema_sha
            or tape.get("source_sha256") != args.v12_source_sha256
            or audit.get("draft_actual_forward_count")
               != audit.get("draft_proposals_count")
            or audit.get("oracle_actual_forward_count") != len(tape.get("steps", []))
            or len(tape.get("steps", [])) != len(tape.get("token_ids", []))):
        raise ValueError("val independent AR/forward receipt failed")
    valid = _schema_validator(args.reference_validator)([
        {"observation": o, "labels": l} for o, l in zip(obs, labs, strict=True)])
    if valid != {"requests": 1, "rounds": len(obs)}:
        raise ValueError("val full-field reference validator failed")
    cursor = list(prompt.input_ids)
    used_hidden = set()
    for i, (o, lab) in enumerate(zip(obs, labs, strict=True)):
        if (o["request_id"] != rid or o["round_id"] != i
                or o["prefix_token_ids"] != cursor
                or any(len(step.get("actual_forward_logits_sha256", "")) != 64
                       for step in o["draft_steps"])):
            raise ValueError("val round/prefix chain differs")
        cursor = o["prefix_after_commit_token_ids"]
        ref = o["target_valid_hidden_ref"]
        if ref is not None:
            key = ref["key"]
            if (hidden_path is None or key not in hidden or key in used_hidden
                    or ref.get("path") != files["hidden"]["path"]
                    or ref.get("sha256") != files["hidden"]["sha256"]
                    or list(hidden[key].shape) != ref.get("shape")
                    or str(hidden[key].dtype) != ref.get("dtype")
                    or not torch.isfinite(hidden[key]).all().item()
                    or ref.get("prefix_sha256")
                       != digest(o["valid_prefix_token_ids"])):
                raise ValueError("val teacher ref differs")
            used_hidden.add(key)
        future = lab["target_future"]
        fref = future.get("topk_ref")
        if fref is not None:
            start, length = fref.get("offset"), fref.get("length")
            if (fref.get("path") != files["oracle_tape"]["path"]
                    or fref.get("sha256") != files["oracle_tape"]["sha256"]
                    or type(start) is not int or type(length) is not int
                    or future["token_ids"] != tape["token_ids"][start:start+length]
                    or len(future["token_ids"]) != length):
                raise ValueError("val future oracle ref differs")
    if (set(hidden) != used_hidden or cursor[len(prompt.input_ids):]
            != tape["token_ids"]):
        raise ValueError("val hidden/committed stream differs")
    prefix = list(prompt.input_ids)
    for token, step in zip(tape["token_ids"], tape["steps"], strict=True):
        if (step.get("query_prefix_sha256") != digest(prefix)
                or step.get("greedy_token_id") != token
                or len(step.get("topk_ids", [])) != 32
                or len(step.get("topk_logits", [])) != 32
                or len(step.get("actual_forward_logits_sha256", "")) != 64):
            raise ValueError("val independent Target Top32/prefix differs")
        prefix.append(token)
    return {"request_id": rid, "prompt_id": prompt.id,
            "family_id": prompt.family_id, "source": prompt.task_type,
            "root": str(root), "manifest_sha256": sha256_file(mpath),
            "rounds": len(obs)}


def lock_val(output: Path, prompts, train_lock_sha: str) -> dict:
    lock = {"scope": "v13_post_training_independent_200_val",
            "train_lock_sha256": train_lock_sha,
            "parent_dataset_sha256": base.PARENT_SHA,
            "prompt_ids": [p.id for p in prompts],
            "seeds": [SEED], "arms": list(ARMS),
            "checkpoint_epoch": 3, "max_windows_per_prompt": 8,
            "primary_eligibility": "future_length_is_8_and_future_excludes_eos",
            "primary_metric": "first_prefix_matching_tokens_0_to_8",
            "aggregation": "mean_windows_per_prompt_then_equal_prompt_weight",
            "bootstrap": "family_clusters_within_source_10000_draws",
            "teacher_controls": ["correct", "zero", "stratified_shuffle"],
            "val_labels_read_before_lock": False,
            "not_a_WAN_or_speed_result": True}
    _freeze(output / "val_analysis_lock.json", lock)
    return lock


def collect_and_evaluate_val(args, output: Path, prompts, target, draft,
                             draft_model, config, train_lock, schema_sha: str,
                             provenance: dict, eos: int, time_left,
                             ledger: dict) -> bool:
    lock_val(output, prompts, sha256_file(output / "train_lock.json"))
    root = output / "val"
    root.mkdir(exist_ok=True)
    inventory = []
    for prompt in prompts:
        rid = base.request_id(base.PARENT_SHA, "val", prompt.id,
                              base.POLICY, base.EXECUTION)
        mpath = root / "manifests" / f"{rid}.json"
        if not mpath.exists():
            if time_left() < 600:
                return False
            base.collect_v12_request(
                target=target, draft=draft, prompt=prompt, root=root,
                eos_token_id=eos, dataset_lock_sha256=base.PARENT_SHA,
                source_sha256=args.v12_source_sha256,
                schema_lock_sha256=schema_sha,
                provenance={**provenance, "prompt_id": prompt.id,
                            "family_id": prompt.family_id, "source": prompt.task_type,
                            "split": "val", "collection_phase": "val_collection"},
                policy_id=base.POLICY, execution_id=base.EXECUTION)
        record = audit_val_one(root, prompt, args, schema_sha, provenance)
        inventory.append(record)
        ledger["val_collected"] = len(inventory)
        atomic_json(output / "pilot_ledger.json", ledger)
    if len(inventory) != 200:
        raise ValueError("independent val membership incomplete")
    _freeze(output / "val_data_acceptance.json", {
        "scope": "v13_independent_200_val_full_field_acceptance",
        "exact_requests": 200, "fully_valid_requests": 200,
        "train_lock_sha256": sha256_file(output / "train_lock.json"),
        "val_analysis_lock_sha256": sha256_file(output / "val_analysis_lock.json"),
        "members": inventory, "member_digest_sha256": digest(inventory),
    })
    members = tuple((p, next(x for x in inventory if x["prompt_id"] == p.id))
                    for p in prompts)
    windows = load_windows(members, eos, split="val")
    if not training._evaluate_all(args, output, windows, prompts, config, draft,
                                  draft_model, train_lock, eos, time_left, ledger):
        return False
    rows = []
    for prompt in prompts:
        item = json.loads((output / "val_scores" / f"seed{SEED}" /
                           training._chunk_name(prompt)).read_text(encoding="utf-8"))
        rows.extend(item["rows"])
    scores = prompt_scores(rows, eligible_only=True)
    arm_mean = {arm: (sum(v for (seed, _, _, _, a), v in scores.items()
                          if seed == SEED and a == arm) /
                      sum(seed == SEED and a == arm
                          for (seed, _, _, _, a) in scores))
                for arm in ("raw_draft", *ARMS, "F-H-zero")
                if any(seed == SEED and a == arm
                       for (seed, _, _, _, a) in scores)}
    intervals = {f"{left}-{right}": paired_family_ci(
        scores, seed=SEED, left=left, right=right)
        for left, right in (("F-H", "C"), ("F-H", "B"), ("C", "B"))}
    stats = {"scope": "v13_single_seed42_three_arm_independent_val",
             "seed": SEED, "arms": list(ARMS),
             "val_data_acceptance_sha256": sha256_file(output / "val_data_acceptance.json"),
             "prompt_count": 200, "window_count": len(windows),
             "primary_window_count": sum(full_window_eligible(w.labels, eos)
                                         for w in windows),
             "seed42_prompt_mean_A_probe": arm_mean,
             "paired_family_bootstrap_95pct": intervals,
             "not_a_WAN_or_speed_result": True}
    _freeze(output / "val_statistics.json", stats)
    return True


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
    prompt_splits = base.locked_prompts(args.parent, args.parent_meta)
    prompts, val_prompts = prompt_splits["train"], prompt_splits["val"]
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
        if finished:
            status["phase"] = "val_collection"
            atomic_json(train_output / "status.json", status)
            finished = collect_and_evaluate_val(
                args, train_output, val_prompts, target, draft, draft_model,
                config, lock, schema_sha, provenance, meta["target_eos_token_id"],
                lambda: deadline - (time.monotonic() - begin), training_ledger)
        status["state"] = "complete_three_arms_val200" if finished else "paused_budget"
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
