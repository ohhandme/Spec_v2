"""Continue locked train collection after quarantining exactness mismatches.

The frozen v12c collector still owns every model decision and every accepted
request.  This driver only changes what the outer loop does after that collector
has atomically written an exactness failure audit.  It never trains a model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from feedback_codec.v12_acceptance import _read_json
from scripts import run_v12_collect_train as base

V13_PROGRAM_SHA = "1b838da0aa6943c021af1cd382b4a1deb1d53a47fcad412afcffc7bc369043eb"
TRAINING_PROTOCOL_SHA = "0217c5ae4a3b3cf8c0d741140aade4dcaa70960326607b8639015ed64333bc63"
PRIOR_LEDGER_SHA = "b5307739ae31d0207bcb4c16874e455a2d4055b9abf2034bb564ad0e5857b889"
PRIOR_FAILURE_SHA = "412e287fda966cf3707993f223d9968b4d7acb19bd51ab60bfffb481b0a594b7"
PRIOR_REQUEST_ID = "8ea73a7e761ed22c9273b0815d4fead397feb6ea11a62135806c22f5beb03de2"
PRIOR_SEGMENTS = 3
PRIOR_SECONDS = 4033.931892745197


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("project_root", "v10_source_manifest", "v10_source_zip",
                 "v11_source_manifest", "v11_source_zip", "v12_source_manifest",
                 "v12_source_zip", "program", "contract", "backfill_manifest",
                 "schema_lock", "reference_validator", "parent", "parent_meta",
                 "v9_p0_report", "v9_inputs", "prior_plan", "target", "draft",
                 "prior_output", "output", "v13_program", "training_protocol",
                 "continuation_manifest"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    for name in ("v10_source_sha256", "v11_source_sha256", "v12_source_sha256"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--deadline-seconds", type=int, default=21000)
    return parser.parse_args()


def checked_failure(path: Path, prompt, rid: str) -> dict:
    audit = _read_json(path)
    if (audit.get("request_id") != rid or audit.get("prompt_id") != prompt.id
            or audit.get("split") != "train" or audit.get("policy_id") != base.POLICY
            or audit.get("execution_id") != base.EXECUTION
            or audit.get("exact") is not False
            or type(audit.get("first_difference_index")) is not int
            or not isinstance(audit.get("packed_ids"), list)
            or not isinstance(audit.get("oracle_ids"), list)
            or not isinstance(audit.get("rounds"), list)):
        raise ValueError(f"non-exact audit identity/content differs: {prompt.id}")
    return {"request_id": rid, "prompt_id": prompt.id,
            "audit_sha256": base.sha256_file(path),
            "audit_path": str(path),
            "first_difference_index": audit["first_difference_index"]}


def checked_request_error(path: Path, root: Path, prompt, rid: str,
                          source_sha: str, schema_sha: str) -> dict:
    receipt = _read_json(path)
    if (receipt.get("request_id") != rid or receipt.get("prompt_id") != prompt.id
            or receipt.get("source_zip_sha256") != source_sha
            or receipt.get("schema_lock_sha256") != schema_sha
            or not isinstance(receipt.get("error_type"), str)
            or not isinstance(receipt.get("error"), str)
            or not isinstance(receipt.get("traceback"), str)
            or not isinstance(receipt.get("files"), list)):
        raise ValueError(f"request-error receipt differs: {prompt.id}")
    for item in receipt["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("request-error file reference is malformed")
        artifact = (root / item["path"]).resolve()
        if (not artifact.is_relative_to(root.resolve())
                or base.sha256_file(artifact) != item.get("sha256")):
            raise ValueError("request-error original file changed")
    return {"request_id": rid, "prompt_id": prompt.id,
            "error_receipt_sha256": base.sha256_file(path),
            "error_receipt_path": str(path),
            "error_type": receipt["error_type"]}


def save_request_error(root: Path, prompt, rid: str, exc: Exception,
                       source_sha: str, schema_sha: str) -> dict:
    receipt_path = root / "audit" / "request_failures" / f"{rid}.json"
    if receipt_path.exists():
        raise FileExistsError("request-error receipt already exists")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    related = []
    for artifact in root.rglob(f"*{rid}*"):
        if artifact.is_file():
            related.append({"path": str(artifact.relative_to(root)),
                            "sha256": base.sha256_file(artifact)})
    mismatch_path = root / "audit" / "failures" / f"{rid}.json"
    mismatch = _read_json(mismatch_path) if mismatch_path.exists() else {}
    first = mismatch.get("first_difference_index")
    packed = mismatch.get("packed_ids")
    oracle = mismatch.get("oracle_ids")
    base.atomic_json(receipt_path, {
        "request_id": rid, "prompt_id": prompt.id, "split": "train",
        "policy_id": base.POLICY, "execution_id": base.EXECUTION,
        "source_zip_sha256": source_sha,
        "schema_lock_sha256": schema_sha,
        "error_type": type(exc).__name__, "error": str(exc),
        "traceback": traceback.format_exc(),
        "first_difference_index": first if type(first) is int and first >= 0 else None,
        "packed_token_at_first_difference":
            packed[first] if type(first) is int and first >= 0 and isinstance(packed, list)
            and first < len(packed) else None,
        "oracle_token_at_first_difference":
            oracle[first] if type(first) is int and first >= 0 and isinstance(oracle, list)
            and first < len(oracle) else None,
        "files": sorted(related, key=lambda row: row["path"]),
        "valid_training_request": False,
    })
    return checked_request_error(receipt_path, root, prompt, rid,
                                 source_sha, schema_sha)


def check_continuation(args, prompts) -> tuple[int, float]:
    if (base.sha256_file(args.v13_program) != V13_PROGRAM_SHA
            or base.sha256_file(args.training_protocol) != TRAINING_PROTOCOL_SHA):
        raise ValueError("v13 program/training protocol bytes differ")
    protocol = _read_json(args.training_protocol)
    if (protocol.get("training_seeds") != [42]
            or protocol.get("forbidden_training_seeds") != [43, 44]):
        raise ValueError("v13 single-seed contract differs")
    manifest = _read_json(args.continuation_manifest)
    if (manifest.get("driver_sha256") != base.sha256_file(Path(__file__))
            or manifest.get("v13_program_sha256") != V13_PROGRAM_SHA
            or manifest.get("v12c_source_zip_sha256") != args.v12_source_sha256
            or manifest.get("prior_ledger_sha256") != PRIOR_LEDGER_SHA
            or manifest.get("prior_failure_sha256") != PRIOR_FAILURE_SHA):
        raise ValueError("continuation driver/source receipt differs")
    prior = args.prior_output
    if (prior.resolve() == args.output.resolve()
            or prior.parent != args.output.parent
            or (prior / "active_worker.json").exists()
            or base.sha256_file(prior / "collection_ledger.json") != PRIOR_LEDGER_SHA):
        raise ValueError("prior failed worker is mutable or output is not isolated")
    ledger = _read_json(prior / "collection_ledger.json")
    if (ledger.get("state") != "failed"
            or ledger.get("segments_used") != PRIOR_SEGMENTS
            or ledger.get("gpu_seconds_charged") != PRIOR_SECONDS
            or ledger.get("fully_valid_train_requests") != 107
            or ledger.get("training_updates") != 0
            or ledger.get("source_zip_sha256") != args.v12_source_sha256):
        raise ValueError("prior GPU budget or accepted count differs")
    failure = prior / "train" / "audit" / "failures" / f"{PRIOR_REQUEST_ID}.json"
    if base.sha256_file(failure) != PRIOR_FAILURE_SHA:
        raise ValueError("original first mismatch evidence changed")
    prior_manifests = list((prior / "train" / "manifests").glob("*.json"))
    prior_failures = list((prior / "train" / "audit" / "failures").glob("*.json"))
    if len(prior_manifests) != 107 or prior_failures != [failure]:
        raise ValueError("prior complete/quarantine inventory differs")
    inventory = {base.request_id(base.PARENT_SHA, "train", p.id,
                                 base.POLICY, base.EXECUTION): p for p in prompts}
    if len(inventory) != 1000 or PRIOR_REQUEST_ID not in inventory:
        raise ValueError("locked dataset/failure membership differs")
    checked_failure(failure, inventory[PRIOR_REQUEST_ID], PRIOR_REQUEST_ID)
    return ledger["segments_used"], ledger["gpu_seconds_charged"]


def main() -> None:
    args = parse_args()
    if not 0 < args.deadline_seconds <= 21000:
        raise ValueError("segment deadline exceeds frozen six-hour ceiling")
    begin = time.monotonic()
    meta, models, files, schema_sha = base.check_source(args)
    prompts = base.locked_prompts(args.parent, args.parent_meta)["train"]
    args.legacy_ids = base.check_backfill(args, prompts)
    previous_segments, previous_seconds = check_continuation(args, prompts)
    if base.sha256_file(args.reference_validator) != _read_json(args.schema_lock)["reference_validator_sha256"]:
        raise ValueError("reference validator bytes differ")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "segments").mkdir(exist_ok=True)
    active = output / "active_worker.json"
    ledger_path = output / "collection_ledger.json"
    if active.exists():
        raise RuntimeError("continuation worker lock exists; inspect real PID")
    state = _read_json(ledger_path) if ledger_path.exists() else {
        "program_sha256": V13_PROGRAM_SHA,
        "continuation_driver_sha256": base.sha256_file(Path(__file__)),
        "v12c_source_zip_sha256": args.v12_source_sha256,
        "schema_lock_sha256": schema_sha,
        "dataset_lock_sha256": base.PARENT_SHA,
        "segments_used": previous_segments,
        "gpu_seconds_charged": previous_seconds,
        "fully_valid_train_requests": 107,
        "quarantined_mismatch_requests": 1,
        "processed_train_requests": 108,
        "training_updates": 0,
        "state": "ready",
    }
    if (state.get("program_sha256") != V13_PROGRAM_SHA
            or state.get("continuation_driver_sha256") != base.sha256_file(Path(__file__))
            or state.get("v12c_source_zip_sha256") != args.v12_source_sha256
            or state.get("schema_lock_sha256") != schema_sha
            or state.get("training_updates") != 0
            or state.get("segments_used", 0) >= base.MAX_SEGMENTS
            or state.get("gpu_seconds_charged", 0) >= 129600
            or state.get("state") in ("failed", "complete", "collected_with_anomalies")):
        raise ValueError("continuation source/budget/zero-training resume gate failed")
    if not ledger_path.exists():
        base.atomic_json(ledger_path, state)
    segment = state["segments_used"] + 1
    deadline = min(args.deadline_seconds, 129600 - state["gpu_seconds_charged"])
    with active.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "segment": segment,
                   "continuation_driver_sha256": state["continuation_driver_sha256"],
                   "v12c_source_zip_sha256": args.v12_source_sha256,
                   "schema_lock_sha256": schema_sha,
                   "started_unix_s": time.time()}, f)
        f.flush()
        os.fsync(f.fileno())
    status = {"state": "running", "phase": "gpu_admission", "segment": segment,
              "pid": os.getpid(), "training_updates": 0,
              "v12c_source_zip_sha256": args.v12_source_sha256,
              "continuation_driver_sha256": state["continuation_driver_sha256"]}
    base.atomic_json(output / "status.json", status)
    try:
        status["gpu_admission"] = base._admit_idle_gpu("cuda:0", 22000)
        status["phase"] = "load_frozen_models"
        base.atomic_json(output / "status.json", status)
        eos = meta["target_eos_token_id"]
        if eos != 151645:
            raise ValueError("Target EOS differs")
        target, draft, _, _ = base._load_models(args, eos)
        provenance = {
            "dataset_lock_sha256": base.PARENT_SHA, "execution_id": base.EXECUTION,
            "policy_id": base.POLICY,
            "draft_checkpoint_sha256": models["draft"]["model.safetensors"],
            "feedback_policy_sha256": base.digest({"policy": base.POLICY, "window": 8,
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
        train_root = output / "train"
        prior_train = args.prior_output / "train"
        train_root.mkdir(exist_ok=True)
        accepted, anomalies = [], []
        for prompt in prompts:
            rid = base.request_id(base.PARENT_SHA, "train", prompt.id,
                                  base.POLICY, base.EXECUTION)
            prior_manifest = prior_train / "manifests" / f"{rid}.json"
            new_manifest = train_root / "manifests" / f"{rid}.json"
            prior_failure = prior_train / "audit" / "failures" / f"{rid}.json"
            new_failure = train_root / "audit" / "failures" / f"{rid}.json"
            new_error = train_root / "audit" / "request_failures" / f"{rid}.json"
            if prior_manifest.exists() and (new_manifest.exists() or new_failure.exists()
                                            or new_error.exists()):
                raise ValueError("duplicate complete/failed request across source roots")
            if prior_failure.exists() and (prior_manifest.exists()
                                           or new_manifest.exists()
                                           or new_failure.exists() or new_error.exists()):
                raise ValueError("quarantined request duplicated across source roots")
            if prior_manifest.exists():
                accepted.append(base.audit_one(prior_train, prompt, args, files,
                                               schema_sha, provenance))
            elif prior_failure.exists():
                anomalies.append(checked_failure(prior_failure, prompt, rid))
            elif new_error.exists():
                anomalies.append(checked_request_error(new_error, train_root, prompt,
                                                      rid, args.v12_source_sha256,
                                                      schema_sha))
            else:
                if not new_manifest.exists() and not new_failure.exists():
                    if deadline - (time.monotonic() - begin) < 600:
                        status["state"] = "paused_exact_time_slice"
                        break
                    status.update({"phase": "train_collection", "prompt_id": prompt.id,
                                   "fully_valid_train_requests": len(accepted),
                                   "quarantined_mismatch_requests": len(anomalies),
                                   "processed_train_requests": len(accepted) + len(anomalies)})
                    base.atomic_json(output / "status.json", status)
                    try:
                        base.collect_v12_request(
                            target=target, draft=draft, prompt=prompt,
                            root=train_root, eos_token_id=eos,
                            dataset_lock_sha256=base.PARENT_SHA,
                            source_sha256=args.v12_source_sha256,
                            schema_lock_sha256=schema_sha,
                            provenance={**provenance, "prompt_id": prompt.id,
                                        "family_id": prompt.family_id,
                                        "source": prompt.task_type, "split": "train"},
                            policy_id=base.POLICY, execution_id=base.EXECUTION)
                    except Exception as exc:
                        anomalies.append(save_request_error(
                            train_root, prompt, rid, exc,
                            args.v12_source_sha256, schema_sha))
                    else:
                        try:
                            accepted.append(base.audit_one(
                                train_root, prompt, args, files,
                                schema_sha, provenance))
                        except Exception as exc:
                            anomalies.append(save_request_error(
                                train_root, prompt, rid, exc,
                                args.v12_source_sha256, schema_sha))
                elif new_manifest.exists():
                    try:
                        accepted.append(base.audit_one(train_root, prompt, args,
                                                       files, schema_sha, provenance))
                    except Exception as exc:
                        anomalies.append(save_request_error(
                            train_root, prompt, rid, exc,
                            args.v12_source_sha256, schema_sha))
                else:
                    anomalies.append(checked_failure(new_failure, prompt, rid))
            if len(accepted) + len(anomalies) < 108:
                # Recheck historical manifests without publishing a lower count.
                continue
            state.update({"state": "running", "phase": "train_collection",
                          "fully_valid_train_requests": len(accepted),
                          "quarantined_mismatch_requests": len(anomalies),
                          "processed_train_requests": len(accepted) + len(anomalies),
                          "last_examined_prompt_id": prompt.id})
            base.atomic_json(ledger_path, state)
            status.update({"fully_valid_train_requests": len(accepted),
                           "quarantined_mismatch_requests": len(anomalies),
                           "processed_train_requests": len(accepted) + len(anomalies),
                           "last_examined_prompt_id": prompt.id})
            base.atomic_json(output / "status.json", status)
            if anomalies:
                base.atomic_json(output / "quarantine_index.json", {
                    "dataset_lock_sha256": base.PARENT_SHA,
                    "policy_id": base.POLICY, "execution_id": base.EXECUTION,
                    "mismatches": anomalies,
                    "training_updates": 0,
                })
        else:
            status["state"] = ("collected_with_anomalies" if anomalies
                               else "collected_1000_pending_full_audit")
            state["state"] = status["state"]
        status["train_ready"] = False
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
        base.atomic_json(ledger_path, state)
        status.update({"elapsed_seconds": elapsed, "ended_unix_s": time.time(),
                       "fully_valid_train_requests": state["fully_valid_train_requests"],
                       "quarantined_mismatch_requests":
                           state["quarantined_mismatch_requests"],
                       "processed_train_requests": state["processed_train_requests"]})
        base.atomic_json(output / "status.json", status)
        base.atomic_json(output / "segments" / f"segment{segment}.json", {
            "segment": segment, "state": status["state"], "pid": os.getpid(),
            "elapsed_seconds": elapsed,
            "cumulative_gpu_worker_wall_seconds": state["gpu_seconds_charged"],
            "fully_valid_train_requests": state["fully_valid_train_requests"],
            "quarantined_mismatch_requests":
                state["quarantined_mismatch_requests"],
            "processed_train_requests": state["processed_train_requests"],
            "training_updates": 0, "v13_program_sha256": V13_PROGRAM_SHA,
            "v12c_source_zip_sha256": args.v12_source_sha256,
            "continuation_driver_sha256": state["continuation_driver_sha256"],
            "schema_lock_sha256": schema_sha})
        active.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
