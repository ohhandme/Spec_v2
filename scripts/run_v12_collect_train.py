"""Collect the locked 1000 train requests under the v12 data contract.

This worker never constructs an optimizer or reads validation/test data.  A
complete request is checked before its manifest can count toward the 1000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file

from feedback_codec.v11_full_contract import PARENT_SHA256, locked_prompts, sha256_file
from feedback_codec.v12_acceptance import _artifact, _read_json, _read_jsonl, _schema_validator, PROVENANCE_KEYS
from feedback_codec.v12_collect_one import collect_v12_request
from feedback_codec.v12_round_assembly import digest, request_id
from scripts.build_packed_probe_numeric_v3 import _admit_idle_gpu, _prompt_digest
from scripts.run_v11_full_pilot import _check_source, _load_models
from scripts.run_v10_p1_wiring import atomic_json

PARENT_SHA = PARENT_SHA256
PROGRAM_SHA = "7279c447deb183262d560bde717a5a356d8f6ca88f42338315004f6ef95104d9"
CONTRACT_SHA = "a6a7030c2b01aa87056a620d1dbbeb06414567bce52c6250e59b58d54d275405"
BACKFILL_SHA = "d5d6bd87b49355eee8978d07881891483dc99aad46276f8559a8a86203705024"
POLICY = "raw_draft_v9c_ulp2"
EXECUTION = "v12_full_pilot_1"
MAX_SEGMENTS = 6
MAX_SEGMENT_SECONDS = 21600
PREVIOUS_SEGMENTS = 2
PREVIOUS_SECONDS = 896.2508818041533


def check_source(args):
    if (sha256_file(args.program) != PROGRAM_SHA
            or sha256_file(args.contract) != CONTRACT_SHA
            or sha256_file(args.backfill_manifest) != BACKFILL_SHA
            or sha256_file(args.parent) != PARENT_SHA
            or sha256_file(args.v12_source_zip) != args.v12_source_sha256):
        raise ValueError("v12 frozen program/source/data/backfill bytes differ")
    source = _read_json(args.v12_source_manifest)
    files = source.get("file_sha256")
    if (source.get("program_sha256") != PROGRAM_SHA
            or source.get("archive_sha256") != args.v12_source_sha256
            or not isinstance(files, dict)
            or "scripts/run_v12_collect_train.py" not in files):
        raise ValueError("v12 source manifest is incomplete")
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha256_file(path) != expected:
            raise ValueError(f"runtime source byte mismatch: {name}")
    meta, models, previous = _check_source(args)
    if any(previous.get(k) != files.get(k) for k in previous):
        raise ValueError("v9c/v10b/v11 frozen source bytes changed")
    lock = _read_json(args.schema_lock)
    if (lock.get("program_sha256") != PROGRAM_SHA
            or lock.get("data_contract_sha256") != CONTRACT_SHA
            or lock.get("backfill_manifest_sha256") != BACKFILL_SHA
            or lock.get("parent_dataset_sha256") != PARENT_SHA
            or lock.get("source_zip_sha256") != args.v12_source_sha256
            or lock.get("source_manifest_sha256")
                != sha256_file(args.v12_source_manifest)
            or lock.get("policy_id") != POLICY
            or lock.get("execution_id") != EXECUTION):
        raise ValueError("v12 source/schema lock differs")
    return meta, models, files, sha256_file(args.schema_lock)


def check_backfill(args, prompts):
    manifest = _read_json(args.backfill_manifest)
    entries = manifest.get("entries")
    if (manifest.get("parent_dataset_sha256") != PARENT_SHA
            or manifest.get("v12_policy_id") != POLICY
            or manifest.get("v12_execution_id") != EXECUTION
            or not isinstance(entries, list) or len(entries) != 31):
        raise ValueError("31-member backfill lock differs")
    inventory = {p.id: p for p in prompts}
    if len(inventory) != 1000:
        raise ValueError("locked train membership differs")
    seen = set()
    for entry in entries:
        pid = entry["prompt_id"]
        if pid in seen or pid not in inventory:
            raise ValueError("duplicate/nonmember backfill prompt")
        seen.add(pid)
        p = inventory[pid]
        if (entry.get("split") != "train"
                or entry.get("family_id") != p.family_id
                or entry.get("source") != p.task_type
                or entry.get("prompt_token_sha256") != digest(list(p.input_ids))
                or entry.get("v12_request_id")
                    != request_id(PARENT_SHA, "train", pid, POLICY, EXECUTION)):
            raise ValueError("backfill prompt/token/request identity differs")
        for kind in ("chunk", "stream"):
            path = (args.project_root / entry[f"legacy_{kind}_path"]).resolve()
            if (not path.is_relative_to(args.project_root.resolve())
                    or sha256_file(path) != entry[f"legacy_{kind}_sha256"]):
                raise ValueError(f"legacy {kind} original changed: {pid}")
    return seen


def audit_one(root, prompt, args, source_files, schema_sha, provenance):
    """Verify one real request, its separate bytes and every row/ref."""
    rid = request_id(PARENT_SHA, "train", prompt.id, POLICY, EXECUTION)
    manifest_path = root / "manifests" / f"{rid}.json"
    m = _read_json(manifest_path)
    if (m.get("request_id") != rid or m.get("prompt_id") != prompt.id
            or m.get("family_id") != prompt.family_id
            or m.get("source") != prompt.task_type
            or m.get("split") != "train" or m.get("exact") is not True
            or m.get("schema_lock_sha256") != schema_sha
            or m.get("extension_source_sha256") != args.v12_source_sha256
            or m.get("dataset_lock_sha256") != PARENT_SHA):
        raise ValueError("one-request identity/source/exact gate failed")
    files = m.get("files")
    if not isinstance(files, dict):
        raise ValueError("one-request file map missing")
    paths = {k: _artifact(root, files[k]) for k in
             ("observations", "labels", "oracle_tape", "audit")}
    hidden_path = _artifact(root, files["hidden"]) if files.get("hidden") else None
    hidden = load_file(str(hidden_path), device="cpu") if hidden_path else {}
    pv = m.get("provenance")
    if (not isinstance(pv, dict) or not set(PROVENANCE_KEYS).issubset(pv)
            or any(pv.get(k) != v for k, v in provenance.items())
            or pv.get("prompt_id") != prompt.id
            or pv.get("family_id") != prompt.family_id
            or pv.get("source") != prompt.task_type
            or pv.get("exact_stream_receipt_sha256") != files["audit"]["sha256"]):
        raise ValueError("one-request provenance gate failed")
    obs, labs = _read_jsonl(paths["observations"]), _read_jsonl(paths["labels"])
    tape, audit = _read_json(paths["oracle_tape"]), _read_json(paths["audit"])
    if (len(obs) != len(labs) or len(obs) != m.get("rounds")
            or sum(len(x["draft_steps"]) for x in obs) != m.get("draft_positions")
            or tape.get("token_ids") != audit.get("oracle_ids")
            or audit.get("packed_ids") != audit.get("oracle_ids")
            or audit.get("packed_eos") != audit.get("oracle_eos")
            or audit.get("packed_truncated") != audit.get("oracle_truncated")
            or audit.get("first_difference_index") is not None
            or audit.get("draft_actual_forward_count")
                != audit.get("draft_proposals_count")
            or audit.get("oracle_actual_forward_count") != len(tape.get("steps", []))
            or len(tape.get("steps", [])) != len(tape.get("token_ids", []))):
        raise ValueError("one-request independent AR/forward receipt failed")
    result = _schema_validator(args.reference_validator)([
        {"observation": o, "labels": l} for o, l in zip(obs, labs, strict=True)])
    if result != {"requests": 1, "rounds": len(obs)}:
        raise ValueError("one-request reference schema validation failed")
    cursor = list(prompt.input_ids)
    used_hidden = set()
    for i, (o, l) in enumerate(zip(obs, labs, strict=True)):
        if (o["request_id"] != rid or o["round_id"] != i
                or o["prefix_token_ids"] != cursor
                or any(not isinstance(s.get("actual_forward_logits_sha256"), str)
                       or len(s["actual_forward_logits_sha256"]) != 64
                       for s in o["draft_steps"])):
            raise ValueError("one-request actual Draft prefix/source differs")
        cursor = o["prefix_after_commit_token_ids"]
        ref = o["target_valid_hidden_ref"]
        if ref is not None:
            key = ref.get("key")
            if (hidden_path is None or key not in hidden or key in used_hidden
                    or ref.get("path") != files["hidden"]["path"]
                    or ref.get("sha256") != files["hidden"]["sha256"]
                    or list(hidden[key].shape) != ref.get("shape")
                    or str(hidden[key].dtype) != ref.get("dtype")
                    or not torch.isfinite(hidden[key]).all().item()
                    or ref.get("prefix_sha256") != digest(o["valid_prefix_token_ids"])):
                raise ValueError("one-request teacher tensor ref differs")
            used_hidden.add(key)
        future = l["target_future"]
        fref = future.get("topk_ref")
        if fref is not None:
            start, length = fref.get("offset"), fref.get("length")
            if (fref.get("path") != files["oracle_tape"]["path"]
                    or fref.get("sha256") != files["oracle_tape"]["sha256"]
                    or type(start) is not int or type(length) is not int
                    or future["token_ids"] != tape["token_ids"][start:start+length]
                    or len(future["token_ids"]) != length):
                raise ValueError("one-request future Target ref differs")
    if (set(hidden) != used_hidden or cursor[len(prompt.input_ids):]
            != tape["token_ids"]):
        raise ValueError("one-request teacher/committed stream differs")
    prefix = list(prompt.input_ids)
    for token, step in zip(tape["token_ids"], tape["steps"], strict=True):
        if (step.get("query_prefix_sha256") != digest(prefix)
                or step.get("greedy_token_id") != token
                or len(step.get("topk_ids", [])) != 32
                or len(step.get("topk_logits", [])) != 32
                or len(step.get("actual_forward_logits_sha256", "")) != 64):
            raise ValueError("one-request real Target Top32/prefix differs")
        prefix.append(token)
    return {"request_id": rid, "prompt_id": prompt.id,
            "manifest_sha256": sha256_file(manifest_path),
            "rounds": len(obs), "draft_positions": m["draft_positions"],
            "legacy_backfill": prompt.id in args.legacy_ids}


def main():
    parser = argparse.ArgumentParser()
    for name in ("project_root", "v10_source_manifest", "v10_source_zip",
                 "v11_source_manifest", "v11_source_zip", "v12_source_manifest",
                 "v12_source_zip", "program", "contract", "backfill_manifest",
                 "schema_lock", "reference_validator", "parent", "parent_meta",
                 "v9_p0_report", "v9_inputs", "prior_plan", "target", "draft",
                 "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    for name in ("v10_source_sha256", "v11_source_sha256", "v12_source_sha256"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--deadline-seconds", type=int, default=21000)
    args = parser.parse_args()
    if not 0 < args.deadline_seconds <= 21000:
        raise ValueError("v12 worker segment deadline exceeds fixed bound")
    begin = time.monotonic()
    meta, models, files, schema_sha = check_source(args)
    prompts = locked_prompts(args.parent, args.parent_meta)["train"]
    args.legacy_ids = check_backfill(args, prompts)
    if sha256_file(args.reference_validator) != _read_json(args.schema_lock)["reference_validator_sha256"]:
        raise ValueError("reference schema validator bytes differ")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "segments").mkdir(exist_ok=True)
    train_root = output / "train"
    state_path, active = output / "collection_ledger.json", output / "active_worker.json"
    if active.exists():
        raise RuntimeError("v12 worker lock exists; inspect PID before resume")
    state = _read_json(state_path) if state_path.exists() else {
        "program_sha256": PROGRAM_SHA, "source_zip_sha256": args.v12_source_sha256,
        "schema_lock_sha256": schema_sha, "dataset_lock_sha256": PARENT_SHA,
        "segments_used": PREVIOUS_SEGMENTS, "gpu_seconds_charged": PREVIOUS_SECONDS,
        "fully_valid_train_requests": 0, "training_updates": 0,
        "state": "ready", "phase": "train_collection"}
    if (state.get("program_sha256") != PROGRAM_SHA
            or state.get("source_zip_sha256") != args.v12_source_sha256
            or state.get("schema_lock_sha256") != schema_sha
            or state.get("training_updates") != 0
            or state.get("segments_used", 0) >= MAX_SEGMENTS
            or state.get("gpu_seconds_charged", 0) >= 129600
            or state.get("state") in ("failed", "complete")):
        raise ValueError("v12 resume/budget/zero-training gate failed")
    if not state_path.exists():
        atomic_json(state_path, state)
    segment = state["segments_used"] + 1
    deadline = min(args.deadline_seconds, 129600 - state["gpu_seconds_charged"])
    with active.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "segment": segment,
                   "source_zip_sha256": args.v12_source_sha256,
                   "schema_lock_sha256": schema_sha,
                   "started_unix_s": time.time()}, f)
        f.flush()
        os.fsync(f.fileno())
    status = {"state": "running", "phase": "gpu_admission", "segment": segment,
              "pid": os.getpid(), "training_updates": 0,
              "source_zip_sha256": args.v12_source_sha256}
    atomic_json(output / "status.json", status)
    try:
        status["gpu_admission"] = _admit_idle_gpu("cuda:0", 22000)
        atomic_json(output / "status.json", status)
        status["phase"] = "load_frozen_models"
        atomic_json(output / "status.json", status)
        eos = meta["target_eos_token_id"]
        if eos != 151645:
            raise ValueError("Target EOS differs")
        target, draft, _, _ = _load_models(args, eos)
        provenance = {
            "dataset_lock_sha256": PARENT_SHA, "execution_id": EXECUTION,
            "policy_id": POLICY,
            "draft_checkpoint_sha256": models["draft"]["model.safetensors"],
            "feedback_policy_sha256": digest({"policy": POLICY, "window": 8,
                                               "numeric_guard_top2_ulps": 2}),
            "seed": 42, "model_sha256": models,
            "tokenizer_sha256": meta["tokenizer_sha256"],
            "chat_template_sha256": meta["tokenizer_sha256"]["target/chat_template"],
            "verifier_source_sha256": files["feedback_codec/packed.py"],
            "extension_source_sha256": args.v12_source_sha256,
            "observer_source_sha256": files["feedback_codec/v12_draft_observer.py"],
            "schema_lock_sha256": schema_sha,
            "collection_phase": "train_collection", "max_new_tokens": 128,
        }
        train_root.mkdir(exist_ok=True)
        accepted = []
        for prompt in prompts:
            rid = request_id(PARENT_SHA, "train", prompt.id, POLICY, EXECUTION)
            manifest = train_root / "manifests" / f"{rid}.json"
            if not manifest.exists():
                if deadline - (time.monotonic() - begin) < 600:
                    status["state"] = "paused_exact_time_slice"
                    break
                status.update({"phase": "train_collection", "prompt_id": prompt.id,
                               "fully_valid_train_requests": len(accepted)})
                atomic_json(output / "status.json", status)
                collect_v12_request(
                    target=target, draft=draft, prompt=prompt, root=train_root,
                    eos_token_id=eos, dataset_lock_sha256=PARENT_SHA,
                    source_sha256=args.v12_source_sha256,
                    schema_lock_sha256=schema_sha,
                    provenance={**provenance, "prompt_id": prompt.id,
                                "family_id": prompt.family_id,
                                "source": prompt.task_type, "split": "train"},
                    policy_id=POLICY, execution_id=EXECUTION)
            row = audit_one(train_root, prompt, args, files, schema_sha,
                            provenance)
            accepted.append(row)
            state.update({"state": "running", "phase": "train_collection",
                          "fully_valid_train_requests": len(accepted),
                          "last_accepted_prompt_id": prompt.id,
                          "last_manifest_sha256": row["manifest_sha256"]})
            atomic_json(state_path, state)
            status["fully_valid_train_requests"] = len(accepted)
            status["last_accepted_prompt_id"] = prompt.id
            atomic_json(output / "status.json", status)
        else:
            status["state"] = "collected_1000_pending_full_audit"
            state["state"] = status["state"]
    except Exception as exc:
        status.update({"state": "failed", "error_type": type(exc).__name__,
                       "error": str(exc), "traceback": traceback.format_exc()})
        state.update({"state": "failed", "error_type": type(exc).__name__,
                      "error": str(exc)})
        raise
    finally:
        elapsed = time.monotonic() - begin
        state["segments_used"] += 1
        state["gpu_seconds_charged"] += elapsed
        atomic_json(state_path, state)
        status.update({"elapsed_seconds": elapsed, "ended_unix_s": time.time(),
                       "fully_valid_train_requests": state["fully_valid_train_requests"]})
        atomic_json(output / "status.json", status)
        atomic_json(output / "segments" / f"segment{segment}.json", {
            "segment": segment, "state": status["state"], "pid": os.getpid(),
            "elapsed_seconds": elapsed,
            "cumulative_gpu_worker_wall_seconds": state["gpu_seconds_charged"],
            "fully_valid_train_requests": state["fully_valid_train_requests"],
            "training_updates": 0, "program_sha256": PROGRAM_SHA,
            "source_zip_sha256": args.v12_source_sha256,
            "schema_lock_sha256": schema_sha})
        active.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
