"""Frozen, single-candidate v11 1000/200 pilot with exact time-slice resume.

Only this driver owns v11 collection, training and fixed-prefix evaluation.
The v9c verifier and v10b arm/reader implementations remain byte-identical.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, FeedbackCodec
from feedback_codec.fixed_probe import ProbePrompt
from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
from feedback_codec.packed import NUMERICAL_RECHECK_BACKEND_V2
from feedback_codec.packed_probe_numeric_v3 import (
    NUMERICAL_RECHECK_TOP2_ULPS, PACKED_PROBE_PROTOCOL_VERSION, canonical_bytes,
)
from feedback_codec.v10_four_arm import V10FourArmHead, V10_MODES, build_matched_v10_arms
from feedback_codec.v10_train_smoke_reader import load_v10_train_smoke
from feedback_codec.v10_closed_loop import run_v10_closed_loop
from feedback_codec.v11_collect_one import collect_one
from feedback_codec.v11_full_contract import (
    PARENT_SHA256, PROGRAM_SHA256, checked_new_chunk, checked_v10_import,
    causal_window_id, locked_prompts, sha256_file, window_id,
)
from feedback_codec.v11_stats import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, full_window_eligible,
    paired_family_ci, prompt_scores, prefix_acceptance,
)
from feedback_codec.v11_teacher_diagnostic import run_teacher_diagnostic
from scripts.build_packed_probe_numeric_v3 import _admit_idle_gpu, _prompt_digest
from scripts.run_v10_p1_wiring import (
    V9_P0_SHA, atomic_json, atomic_torch, fixed_prefix_free,
    frozen_sample, source_and_inputs, state_digest, train_one,
)

MAX_TOTAL_GPU_SECONDS = 36 * 3600
MAX_SEGMENT_SECONDS = 6 * 3600
SEEDS = (42, 43, 44)
ARM_ORDER = tuple(V10_MODES)
CHECKPOINT_EVERY_STEPS = 512
TRAIN_CURVE_WINDOWS = 64


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_source(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    v10_args = argparse.Namespace(
        source_manifest=args.v10_source_manifest,
        source_zip=args.v10_source_zip,
        source_zip_sha256=args.v10_source_sha256,
        parent=args.parent, parent_meta=args.parent_meta,
        v9_p0_report=args.v9_p0_report, v9_inputs=args.v9_inputs,
        prior_plan=args.prior_plan, target=args.target, draft=args.draft,
    )
    _, meta, model_hashes, _ = source_and_inputs(v10_args)
    source = _read_json(args.v11_source_manifest)
    if (source.get("program_sha256") != PROGRAM_SHA256
            or sha256_file(args.v11_source_zip) != args.v11_source_sha256):
        raise ValueError("v11 frozen source/program ZIP differs")
    files = source.get("file_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("v11 source inventory empty")
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256_file(path) != expected:
            raise ValueError(f"v11 runtime source mismatch: {name}")
    required = {"feedback_codec/v11_collect_one.py",
                "feedback_codec/v11_full_contract.py",
                "feedback_codec/v11_stats.py",
                "feedback_codec/v11_teacher_diagnostic.py",
                "scripts/run_v11_full_pilot.py"}
    if not required.issubset(files):
        raise ValueError("v11 extension source manifest is incomplete")
    return meta, model_hashes, files


def _load_models(args: argparse.Namespace, eos: int):
    device = torch.device("cuda:0")
    target = HFTargetBackend(args.target, None, device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(str(args.draft), local_files_only=True,
                                               trust_remote_code=False)
    draft_model = AutoModelForCausalLM.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", torch_dtype=torch.bfloat16).to(device)
    codec = FeedbackCodec(CodecConfig(target_hidden_size=target.hidden_size,
                                     draft_hidden_size=int(draft_model.config.hidden_size),
                                     vocab_size=target.vocab_size),
                          draft_model.get_output_embeddings()).to(device)
    draft = HFDraftBackend(draft_model, tokenizer, codec, target=target,
                           device=device, dtype=torch.bfloat16)
    if (target.tokenizer.eos_token_id != eos or
            any(p.requires_grad for p in target.model.parameters()) or
            any(p.requires_grad for p in draft_model.parameters())):
        raise ValueError("frozen BF16 base/EOS gate failed")
    config = CodecConfig(target_hidden_size=target.hidden_size,
                         draft_hidden_size=draft.hidden_size,
                         vocab_size=draft.vocab_size, window=8)
    return target, draft, draft_model, config


def _init_ledger(args, output: Path, meta: dict, model_hashes: dict,
                 source_files: dict, prompts: dict) -> dict:
    path = output / "pilot_ledger.json"
    if path.exists():
        ledger = _read_json(path)
        if (ledger.get("program_sha256") != PROGRAM_SHA256
                or ledger.get("v11_source_zip_sha256") != args.v11_source_sha256
                or ledger.get("parent_sha256") != PARENT_SHA256):
            raise ValueError("pilot resume source/data/protocol differs")
        return ledger
    if any(output.iterdir()):
        raise FileExistsError("nonempty output without v11 pilot ledger")
    ledger = {
        "state": "running", "phase": "teacher_diagnostic",
        "program_sha256": PROGRAM_SHA256,
        "v10_source_zip_sha256": args.v10_source_sha256,
        "v11_source_zip_sha256": args.v11_source_sha256,
        "v11_source_file_sha256": source_files,
        "parent_sha256": PARENT_SHA256, "model_file_sha256": model_hashes,
        "tokenizer_sha256": meta["tokenizer_sha256"],
        "v9_p0_sha256": V9_P0_SHA,
        "prompts_locked": {key: len(value) for key, value in prompts.items()},
        "old_v10_evidence_root": str(args.v10_evidence),
        "gpu_wall_seconds_finished_segments": 0.0,
        "segments_finished": 0,
        "train_collected_new": 0, "val_collected": 0,
        "training_updates": 0,
    }
    atomic_json(path, ledger)
    return ledger


def _plans(args, output, prompts, meta, model_hashes, source_files):
    paths = {}
    for split in ("train", "val"):
        path = output / f"{split}_collection_plan.json"
        plan = {
            "scope": "v11_full_pilot_locked_split", "split": split,
            "program_sha256": PROGRAM_SHA256,
            "source_zip_sha256": args.v11_source_sha256,
            "source_sha256": source_files,
            "parent_manifest_sha256": PARENT_SHA256,
            "model_file_sha256": model_hashes,
            "tokenizer_sha256": meta["tokenizer_sha256"],
            "packed_p0_report_sha256": V9_P0_SHA,
            "prompt_records": [asdict(p) for p in prompts[split]],
            "prompt_sha256": [_prompt_digest(p) for p in prompts[split]],
            "window": 8, "max_new_tokens": 128,
            "max_probe_windows_per_prompt": 8,
            "eos_token_id": meta["target_eos_token_id"],
            "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
            "numerical_recheck_top2_ulps": NUMERICAL_RECHECK_TOP2_ULPS,
            "shard_count": 1,
            "old_v10_chunks_are_immutable_imports": split == "train",
        }
        if path.exists():
            if _read_json(path) != plan:
                raise ValueError(f"{split} collection plan changed")
        else:
            atomic_json(path, plan)
        paths[split] = (path, sha256_file(path))
    return paths


def _check_v10(args, output, prompts, meta, model_hashes):
    old, receipt = checked_v10_import(
        args.v10_evidence, parent_prompts=prompts["train"],
        source_zip_sha256=args.v10_source_sha256,
        model_hashes=model_hashes,
        tokenizer_sha256=meta["tokenizer_sha256"],
        p0_sha256=V9_P0_SHA,
        eos_token_id=meta["target_eos_token_id"],
    )
    path = output / "immutable_v10_import_receipt.json"
    if path.exists():
        if _read_json(path) != receipt:
            raise ValueError("old four-chunk receipt changed")
    else:
        atomic_json(path, receipt)
    return old, receipt


def _diagnostic(args, output, draft, draft_model, config, old, eos):
    path = output / "teacher_diagnostic.json"
    if path.exists():
        result = _read_json(path)
        if result.get("status") != "passed":
            raise RuntimeError("stored teacher diagnostic failed")
        return result
    protocol = _read_json(args.v10_evidence / "collection_protocol.json")
    loaded = load_v10_train_smoke(
        args.v10_evidence / "fixed_prefix.jsonl",
        args.v10_evidence / "collection_protocol.json",
        expected_parent_sha=PARENT_SHA256,
        expected_plan_sha=protocol["collection_plan_sha256"],
        expected_source_sha=args.v10_source_sha256,
        expected_model_hashes=protocol["model_file_sha256"],
        expected_tokenizer_sha=protocol["tokenizer_sha256"],
        expected_p0_sha=V9_P0_SHA,
    )
    windows = loaded.training_windows()
    if len(windows) != 8:
        raise ValueError("v10 first eight diagnostic windows absent")
    heads = build_matched_v10_arms(
        config, draft_model.get_output_embeddings(), seed=42,
        draft_embedding=draft_model.get_input_embeddings())
    for mode, head in heads.items():
        head.to(torch.device("cuda:0"))
        checkpoint = torch.load(args.v10_evidence / f"{mode}.trainable.pt",
                                map_location="cpu", weights_only=True)
        if checkpoint.get("mode") != mode or checkpoint.get("steps") != 8:
            raise ValueError("v10 accepted checkpoint identity differs")
        head.load_trainable_state_dict(checkpoint["state"])
    result = run_teacher_diagnostic(heads=heads, windows=windows, draft=draft)
    result.update({"v10_protocol_sha256": sha256_file(
        args.v10_evidence / "collection_protocol.json"),
        "v10_checkpoint_sha256": {mode: sha256_file(
            args.v10_evidence / f"{mode}.trainable.pt") for mode in V10_MODES},
        "v11_source_zip_sha256": args.v11_source_sha256})
    atomic_json(path, result)
    del heads
    torch.cuda.empty_cache()
    if result.get("status") != "passed":
        raise RuntimeError("teacher continuous path disconnected")
    return result


def _chunk_name(prompt: ProbePrompt) -> str:
    return hashlib.sha256(prompt.id.encode("utf-8")).hexdigest() + ".json"


def _collect_split(args, output, split, prompts, plan_sha, target, draft, eos,
                   *, old_ids, time_left, ledger):
    directory = output / split
    chunks = directory / "chunks"
    streams = directory / "streams"
    chunks.mkdir(parents=True, exist_ok=True)
    streams.mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        if split == "train" and prompt.id in old_ids:
            continue
        name = _chunk_name(prompt)
        chunk_path = chunks / name
        stream_path = streams / name
        if chunk_path.exists():
            if not stream_path.exists() or _read_json(stream_path).get("exact") is not True:
                raise ValueError("completed chunk lacks exact independent stream")
            checked_new_chunk(chunk_path, prompt=prompt,
                              plan_sha256=plan_sha, eos_token_id=eos)
            continue
        if stream_path.exists():
            raise RuntimeError("stream exists without atomic chunk; stop for audit")
        if time_left() < 180:
            return False
        corpus, stream = collect_one(target, draft, prompt, eos_token_id=eos,
                                     max_new_tokens=128)
        atomic_json(stream_path, stream)
        if stream.get("exact") is not True:
            raise RuntimeError(f"independent Target AR mismatch: {prompt.id}")
        chunk = {
            "prompt_id": prompt.id, "prompt_sha256": _prompt_digest(prompt),
            "plan_sha256": plan_sha, "shard_index": 0, "shard_count": 1,
            "eos_token_id": eos, "window": 8,
            "max_probe_windows_per_prompt": 8,
            "capture_target_logits": True, "feature_provenance_gate": True,
            "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
            "numerical_recheck_top2_ulps": NUMERICAL_RECHECK_TOP2_ULPS,
            "gpu_name": torch.cuda.get_device_name(torch.device("cuda:0")),
            "audit": corpus.audit, "request_audit": corpus.request_audit[0],
            "examples": [row.to_json() for row in corpus.rows],
        }
        chunk["checksum_sha256"] = hashlib.sha256(canonical_bytes(chunk)).hexdigest()
        atomic_json(chunk_path, chunk)
        checked_new_chunk(chunk_path, prompt=prompt,
                          plan_sha256=plan_sha, eos_token_id=eos)
        key = "train_collected_new" if split == "train" else "val_collected"
        ledger[key] += 1
        ledger["last_collected_prompt_id"] = prompt.id
        atomic_json(output / "pilot_ledger.json", ledger)
    return True


def _load_split(output, split, prompts, plan_sha, eos, *, old_windows=()):
    windows = list(old_windows if split == "train" else ())
    old_ids = {w.observation.prompt_id for w in windows}
    inventory = []
    for prompt in prompts:
        if prompt.id in old_ids:
            members = [w for w in windows if w.observation.prompt_id == prompt.id]
            if not members:
                raise ValueError("empty legacy prompt import")
            inventory.append({"prompt_id": prompt.id, "imported_v10": True,
                              "chunk_sha256": members[0].chunk_sha256,
                              "windows": len(members)})
            continue
        path = output / split / "chunks" / _chunk_name(prompt)
        if not path.exists():
            raise RuntimeError(f"INCOMPLETE_DATA: {split} chunk absent: {prompt.id}")
        members = checked_new_chunk(path, prompt=prompt,
                                    plan_sha256=plan_sha, eos_token_id=eos)
        windows.extend(members)
        inventory.append({"prompt_id": prompt.id, "imported_v10": False,
                          "chunk_sha256": sha256_file(path),
                          "windows": len(members)})
    if len(inventory) != len(prompts) or len(windows) > 8 * len(prompts):
        raise ValueError("full pilot count or window cap differs")
    keys = [w.key for w in windows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate pilot window")
    receipt = {
        "split": split, "prompts": len(prompts),
        "exact_requests": len(prompts),
        "valid_prompts": sum(item["windows"] > 0 for item in inventory),
        "zero_window_prompts": [item["prompt_id"] for item in inventory
                                if item["windows"] == 0],
        "windows": len(windows), "label_tokens": sum(len(w.labels) for w in windows),
        "imported_v10_prompts": len(old_ids),
        "inventory": inventory,
    }
    return tuple(windows), receipt


def _training_lock(args, output, windows, config, draft_model):
    if not 1 <= len(windows) <= 8000:
        raise ValueError("v11 train window N violates cap")
    ids = [window_id(w) for w in windows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate train window identity")
    initial = {}
    for seed in SEEDS:
        arms = build_matched_v10_arms(
            config, draft_model.get_output_embeddings(), seed=seed,
            draft_embedding=draft_model.get_input_embeddings())
        hashes = {mode: state_digest(head.trainable_state_dict())
                  for mode, head in arms.items()}
        if len(set(hashes.values())) != 1:
            raise ValueError("initial trainable weights differ within seed")
        if any(head.parameter_budget()["edge_trainable"] != 7643875
               for head in arms.values()):
            raise ValueError("v10b active capacity changed")
        initial[str(seed)] = hashes["B"]
        del arms
    n = len(windows)
    order = {}
    for seed in SEEDS:
        order[str(seed)] = []
        for epoch in range(3):
            values = list(range(n))
            random.Random(seed * 1000 + epoch).shuffle(values)
            order[str(seed)].append(values)
    # Fixed before looking at any validation outcome.  These 64 train windows
    # are diagnostic learning curves only and cannot select an epoch.
    curve_indices = list(range(min(TRAIN_CURVE_WINDOWS, n)))
    lock = {
        "scope": "v11_complete_train_exposure", "program_sha256": PROGRAM_SHA256,
        "v11_source_zip_sha256": args.v11_source_sha256,
        "parent_sha256": PARENT_SHA256, "train_N": n,
        "window_ids": ids, "label_lengths": [len(w.labels) for w in windows],
        "seeds": list(SEEDS), "arms": list(ARM_ORDER), "epochs": 3,
        "epoch_order_indices": order,
        "gumbel_seed_formula": "training_seed*1000000 + epoch*N + position + 1",
        "initial_trainable_sha256": initial,
        "parameter_count_each": 7643875, "optimizer": "AdamW",
        "learning_rate": 1e-4, "weight_decay": .01,
        "betas": [.9, .999], "epsilon": 1e-8,
        "temperature": 1.0, "batch_windows": 1,
        "final_checkpoint": "epoch3", "select_epoch_on_val": False,
        "train_curve_window_indices": curve_indices,
        "val_or_test_used_for_lock": False,
    }
    path = output / "train_lock.json"
    if path.exists():
        if _read_json(path) != lock:
            raise ValueError("train exposure lock changed on resume")
    else:
        atomic_json(path, lock)
    return lock


def _eval_lock(args, output, train_lock, prompts):
    lock = {
        "scope": "v11_independent_val_frozen_before_collection",
        "program_sha256": PROGRAM_SHA256,
        "v11_source_zip_sha256": args.v11_source_sha256,
        "train_lock_sha256": sha256_file(output / "train_lock.json"),
        "seeds": list(SEEDS), "arms": list(ARM_ORDER),
        "checkpoint_epoch": 3, "max_windows_per_prompt": 8,
        "primary_eligibility": "future_length_is_8_and_future_excludes_eos",
        "primary_metric": "first_prefix_matching_tokens_0_to_8",
        "aggregation": "mean_windows_per_prompt_then_equal_prompt_weight",
        "bootstrap": "family_clusters_within_source",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary": ["F-H-C", "F-H-B"],
        "secondary": ["F-CURRENT-TOP32-C", "F-CURRENT-TOP32-B"],
        "teacher_controls": ["correct", "zero", "stratified_shuffle"],
        "closed_loop_train_prompt_ids": [p.id for p in prompts["train"][:2]],
        "closed_loop_val_prompt_ids": [p.id for p in prompts["val"][:2]],
        "cost_scope_lock": {
            "question": "H1_high_bandwidth_teacher_increment",
            "old_margin0_or_8B_cost_gate_applies": False,
            "deployment_cost_measured": False,
            "latency_or_WAN_claim_allowed": False,
        },
    }
    path = output / "val_analysis_lock.json"
    if path.exists():
        if _read_json(path) != lock:
            raise ValueError("val/stat/cost scope lock changed")
    else:
        atomic_json(path, lock)
    return lock


def _parameter_norms_delta(before: dict, after: dict) -> dict[str, float]:
    return {name: float((after[name].float() - value.float()).norm().item())
            for name, value in before.items()}


def _curve(head, draft, windows, indices, eos):
    return [fixed_prefix_free(head, draft, windows[i].observation,
                              windows[i].labels, eos)
            for i in indices]


def _save_training_state(path, *, head, optimizer, seed, mode, epoch,
                         position, aggregate, epoch_start_state,
                         initial_sha, train_lock_sha):
    payload = {
        "program_sha256": PROGRAM_SHA256,
        "train_lock_sha256": train_lock_sha,
        "seed": seed, "mode": mode, "epoch": epoch,
        "position": position,
        "model": head.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "aggregate": aggregate,
        "epoch_start_state": epoch_start_state,
        "initial_sha256": initial_sha,
    }
    atomic_torch(path, payload)


def _train_all(args, output, windows, config, draft, draft_model,
               train_lock, eos, time_left, ledger):
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    current = checkpoints / "last.pt"
    lock_sha = sha256_file(output / "train_lock.json")
    for seed in SEEDS:
        for mode in ARM_ORDER:
            final_path = checkpoints / f"seed{seed}_{mode}_epoch3.pt"
            receipt_path = checkpoints / f"seed{seed}_{mode}.training.json"
            if receipt_path.exists():
                receipt = _read_json(receipt_path)
                if (receipt.get("completed_steps") != 3 * len(windows)
                        or not final_path.exists()
                        or receipt.get("final_sha256") != sha256_file(final_path)):
                    raise ValueError("completed arm receipt/checkpoint differs")
                continue
            if time_left() < 900:
                return False
            torch.manual_seed(seed)
            head = V10FourArmHead(
                config, draft_model.get_output_embeddings(), mode=mode,
                draft_embedding=draft_model.get_input_embeddings()).to("cuda:0")
            initial_sha = state_digest(head.trainable_state_dict())
            if initial_sha != train_lock["initial_trainable_sha256"][str(seed)]:
                raise ValueError("seed/arm initial trainable state differs")
            params = {name: p for name, p in head.named_parameters() if p.requires_grad}
            if len(params) != 27 or sum(p.numel() for p in params.values()) != 7643875:
                raise ValueError("trainable optimizer membership differs")
            optimizer = torch.optim.AdamW(
                params.values(), lr=1e-4, weight_decay=.01,
                betas=(.9, .999), eps=1e-8)
            epoch = 0; position = 0
            aggregate = {"epoch_history": [], "window_loss_sum": 0.0,
                         "token_loss_sum": 0.0, "label_tokens": 0,
                         "gradient_norm_sum": {name: 0.0 for name in params},
                         "gradient_norm_max": {name: 0.0 for name in params},
                         "curve": []}
            epoch_start = head.trainable_state_dict()
            if current.exists():
                saved = torch.load(current, map_location="cpu", weights_only=False)
                if saved.get("seed") == seed and saved.get("mode") == mode:
                    if (saved.get("program_sha256") != PROGRAM_SHA256
                            or saved.get("train_lock_sha256") != lock_sha
                            or saved.get("initial_sha256") != initial_sha):
                        raise ValueError("training resume contract differs")
                    head.load_trainable_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                    epoch = saved["epoch"]; position = saved["position"]
                    aggregate = saved["aggregate"]
                    epoch_start = saved["epoch_start_state"]
                    torch.set_rng_state(saved["torch_cpu_rng"])
                    torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
                    random.setstate(saved["python_rng"])
                else:
                    previous_receipt = checkpoints / (
                        f"seed{saved.get('seed')}_{saved.get('mode')}.training.json")
                    if not previous_receipt.exists():
                        raise ValueError("last checkpoint belongs to another incomplete arm")
            if epoch == 0 and position == 0 and not aggregate["curve"]:
                aggregate["curve"].append({
                    "epoch": 0,
                    "results": _curve(head, draft, windows,
                                      train_lock["train_curve_window_indices"], eos)})
                _save_training_state(current, head=head, optimizer=optimizer,
                                     seed=seed, mode=mode, epoch=epoch,
                                     position=position, aggregate=aggregate,
                                     epoch_start_state=epoch_start,
                                     initial_sha=initial_sha, train_lock_sha=lock_sha)
            elif epoch > 0 and len(aggregate["curve"]) <= epoch:
                if time_left() < 600:
                    return False
                aggregate["curve"].append({
                    "epoch": epoch,
                    "results": _curve(head, draft, windows,
                                      train_lock["train_curve_window_indices"], eos)})
                _save_training_state(current, head=head, optimizer=optimizer,
                                     seed=seed, mode=mode, epoch=epoch,
                                     position=position, aggregate=aggregate,
                                     epoch_start_state=epoch_start,
                                     initial_sha=initial_sha, train_lock_sha=lock_sha)
            while epoch < 3:
                order = train_lock["epoch_order_indices"][str(seed)][epoch]
                while position < len(order):
                    if time_left() < 180:
                        _save_training_state(current, head=head, optimizer=optimizer,
                                             seed=seed, mode=mode, epoch=epoch,
                                             position=position, aggregate=aggregate,
                                             epoch_start_state=epoch_start,
                                             initial_sha=initial_sha, train_lock_sha=lock_sha)
                        return False
                    index = order[position]
                    item = windows[index]
                    gumbel_seed = seed * 1_000_000 + epoch * len(windows) + position + 1
                    result = train_one(head, draft, item.observation,
                                       item.labels, optimizer, gumbel_seed)
                    aggregate["window_loss_sum"] += result["loss"]
                    aggregate["token_loss_sum"] += result["loss"] * len(item.labels)
                    aggregate["label_tokens"] += len(item.labels)
                    for name, value in result["grad_norms"].items():
                        aggregate["gradient_norm_sum"][name] += value
                        aggregate["gradient_norm_max"][name] = max(
                            aggregate["gradient_norm_max"][name], value)
                    position += 1
                    ledger["training_updates"] += 1
                    if position % CHECKPOINT_EVERY_STEPS == 0:
                        _save_training_state(current, head=head, optimizer=optimizer,
                                             seed=seed, mode=mode, epoch=epoch,
                                             position=position, aggregate=aggregate,
                                             epoch_start_state=epoch_start,
                                             initial_sha=initial_sha, train_lock_sha=lock_sha)
                        atomic_json(output / "pilot_ledger.json", ledger)
                trained = head.trainable_state_dict()
                summary = {
                    "epoch": epoch + 1, "steps": len(order),
                    "window_mean_CE": aggregate["window_loss_sum"] / len(order),
                    "token_mean_CE": aggregate["token_loss_sum"] / aggregate["label_tokens"],
                    "label_tokens": aggregate["label_tokens"],
                    "gradient_mean_norm": {name: value / len(order) for name, value
                                           in aggregate["gradient_norm_sum"].items()},
                    "gradient_max_norm": aggregate["gradient_norm_max"],
                    "net_parameter_update_norm": _parameter_norms_delta(epoch_start, trained),
                }
                aggregate["epoch_history"].append(summary)
                epoch += 1; position = 0
                aggregate.update({"window_loss_sum": 0.0,
                                  "token_loss_sum": 0.0, "label_tokens": 0,
                                  "gradient_norm_sum": {name: 0.0 for name in params},
                                  "gradient_norm_max": {name: 0.0 for name in params}})
                epoch_start = trained
                epoch_path = checkpoints / f"seed{seed}_{mode}_epoch{epoch}.pt"
                _save_training_state(epoch_path, head=head, optimizer=optimizer,
                                     seed=seed, mode=mode, epoch=epoch,
                                     position=position, aggregate=aggregate,
                                     epoch_start_state=epoch_start,
                                     initial_sha=initial_sha, train_lock_sha=lock_sha)
                _save_training_state(current, head=head, optimizer=optimizer,
                                     seed=seed, mode=mode, epoch=epoch,
                                     position=position, aggregate=aggregate,
                                     epoch_start_state=epoch_start,
                                     initial_sha=initial_sha, train_lock_sha=lock_sha)
                if time_left() < 600:
                    return False
                aggregate["curve"].append({
                    "epoch": epoch,
                    "results": _curve(head, draft, windows,
                                      train_lock["train_curve_window_indices"], eos)})
                _save_training_state(epoch_path, head=head, optimizer=optimizer,
                                     seed=seed, mode=mode, epoch=epoch,
                                     position=position, aggregate=aggregate,
                                     epoch_start_state=epoch_start,
                                     initial_sha=initial_sha, train_lock_sha=lock_sha)
                _save_training_state(current, head=head, optimizer=optimizer,
                                     seed=seed, mode=mode, epoch=epoch,
                                     position=position, aggregate=aggregate,
                                     epoch_start_state=epoch_start,
                                     initial_sha=initial_sha, train_lock_sha=lock_sha)
            receipt = {
                "seed": seed, "mode": mode,
                "completed_steps": 3 * len(windows),
                "initial_sha256": initial_sha,
                "final_trainable_sha256": state_digest(head.trainable_state_dict()),
                "final_sha256": sha256_file(final_path),
                "epoch_history": aggregate["epoch_history"],
                "curve": aggregate["curve"],
                "optimizer_member_names": list(params),
                "train_lock_sha256": lock_sha,
            }
            atomic_json(receipt_path, receipt)
            atomic_json(output / "pilot_ledger.json", ledger)
            del head, optimizer
            torch.cuda.empty_cache()
    return True


def _donor_bucket(observation):
    count = observation.accepted_count
    accepted = "0" if count == 0 else "1-3" if count <= 3 else "4-8"
    bonus = observation.rejected_token is None
    return observation.task_type, accepted, bonus


def _donor_lock(output, windows):
    path = output / "val_donor_lock.json"
    by_key = {causal_window_id(w): w for w in windows}
    rng = random.Random(20261102)
    table = {}
    coverage = defaultdict(int)
    for w in windows:
        bucket = _donor_bucket(w.observation)
        choices = [other for other in windows
                   if other.observation.prompt_id != w.observation.prompt_id
                   and other.observation.family_id != w.observation.family_id
                   and _donor_bucket(other.observation) == bucket]
        key = causal_window_id(w)
        donor = causal_window_id(choices[rng.randrange(len(choices))]) if choices else None
        table[key] = donor
        coverage["matched" if donor is not None else "NA"] += 1
    lock = {"scope": "v11_teacher_shuffle_pre_quality",
            "seed": 20261102,
            "donor_must_differ_in_prompt_and_family": True,
            "stratum": "same_source_accepted_count_bucket_bonus_type",
            "table": table, "coverage": dict(coverage)}
    if path.exists():
        if _read_json(path) != lock:
            raise ValueError("teacher shuffle donor table changed")
    else:
        atomic_json(path, lock)
    return lock, by_key


@torch.inference_mode()
def _free_with_state(head, draft, state, eos):
    prefix = list(state.corrected_prefix)
    proposed = []
    for position in range(1, 9):
        token = head.next_token(prefix, state, position, draft)
        proposed.append(token)
        prefix.append(token)
        if token == eos:
            break
    return proposed


@torch.inference_mode()
def _raw_free(draft, observation, eos):
    prefix = list(observation.corrected_prefix)
    proposed = []
    for position in range(1, 9):
        token = draft.next(prefix, None, position)
        proposed.append(token)
        prefix.append(token)
        if token == eos:
            break
    return proposed


def _score_row(seed, arm, window, proposed, eos):
    o = window.observation
    return {
        "seed": seed, "arm": arm, "source": o.task_type,
        "family_id": o.family_id, "prompt_id": o.prompt_id,
        "round_index": o.round_index,
        "window_id": causal_window_id(window),
        "accepted_count_at_boundary": o.accepted_count,
        "bonus_type": o.rejected_token is None,
        "proposed_ids": proposed, "future_length": len(window.labels),
        "target_future_ids": list(window.labels),
        "full_window_eligible": full_window_eligible(window.labels, eos),
        "acceptance": prefix_acceptance(proposed, window.labels),
        "first_rejection_position": next(
            (i for i, (a, b) in enumerate(zip(proposed, window.labels)) if a != b),
            len(proposed) if len(proposed) < len(window.labels) else None),
        "next_token_top1_exact": bool(proposed and proposed[0] == window.labels[0]),
    }


@torch.inference_mode()
def _teacher_forced_ce(head, draft, state, labels):
    """Secondary CE only; the held-out continuation never enters free generation."""
    hidden = draft.teacher_hidden(list(state.corrected_prefix), list(labels))
    logits = head.forward_logits(
        hidden.unsqueeze(0), state,
        torch.arange(1, len(labels) + 1, device=hidden.device))
    target = torch.tensor(labels, dtype=torch.long, device=logits.device)
    return float(F.cross_entropy(logits.float().reshape(len(labels), -1),
                                 target).item())


def _load_final_head(output, seed, mode, config, draft_model, train_lock):
    checkpoint = output / "checkpoints" / f"seed{seed}_{mode}_epoch3.pt"
    receipt = _read_json(output / "checkpoints" / f"seed{seed}_{mode}.training.json")
    if receipt.get("final_sha256") != sha256_file(checkpoint):
        raise ValueError("final epoch3 checkpoint SHA differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (payload.get("train_lock_sha256") != sha256_file(output / "train_lock.json")
            or payload.get("epoch") != 3 or payload.get("position") != 0
            or payload.get("seed") != seed or payload.get("mode") != mode):
        raise ValueError("final epoch3 checkpoint contract differs")
    torch.manual_seed(seed)
    head = V10FourArmHead(config, draft_model.get_output_embeddings(), mode=mode,
                          draft_embedding=draft_model.get_input_embeddings()).to("cuda:0")
    if state_digest(head.trainable_state_dict()) != train_lock[
            "initial_trainable_sha256"][str(seed)]:
        raise ValueError("final reload initialization differs")
    head.load_trainable_state_dict(payload["model"])
    head.eval()
    return head


def _evaluate_all(args, output, windows, prompts, config, draft, draft_model,
                  train_lock, eos, time_left, ledger):
    donor, by_key = _donor_lock(output, windows)
    by_prompt = defaultdict(list)
    for window in windows:
        by_prompt[window.observation.prompt_id].append(window)
    directory = output / "val_scores"
    directory.mkdir(exist_ok=True)
    for seed in SEEDS:
        heads = {mode: _load_final_head(output, seed, mode, config,
                                        draft_model, train_lock)
                 for mode in ARM_ORDER}
        seed_dir = directory / f"seed{seed}"
        seed_dir.mkdir(exist_ok=True)
        for prompt in prompts:
            path = seed_dir / _chunk_name(prompt)
            if path.exists():
                existing = _read_json(path)
                if (existing.get("seed") != seed
                        or existing.get("prompt_id") != prompt.id
                        or existing.get("val_lock_sha256") != sha256_file(
                            output / "val_analysis_lock.json")):
                    raise ValueError("val score prompt receipt differs")
                continue
            if time_left() < 180:
                return False
            rows = []
            for window in by_prompt[prompt.id]:
                observation = window.observation
                raw = _raw_free(draft, observation, eos)
                rows.append(_score_row(seed, "raw_draft", window, raw, eos))
                for mode, head in heads.items():
                    state = head.prepare_observation(observation, draft)
                    proposed = _free_with_state(head, draft, state, eos)
                    scored = _score_row(seed, mode, window, proposed, eos)
                    scored["teacher_forced_CE_secondary"] = _teacher_forced_ce(
                        head, draft, state, window.labels)
                    rows.append(scored)
                    if mode == "F-H":
                        zero = replace(state, target_hidden=torch.zeros_like(
                            state.target_hidden))
                        rows.append(_score_row(seed, "F-H-zero", window,
                                               _free_with_state(head, draft, zero, eos), eos))
                        donor_key = donor["table"][causal_window_id(window)]
                        if donor_key is not None:
                            teacher = by_key[donor_key].observation.target_hidden
                            wrong = replace(state, target_hidden=teacher.reshape(
                                1, -1).to(state.target_hidden.device,
                                           state.target_hidden.dtype))
                            rows.append(_score_row(
                                seed, "F-H-stratified_shuffle", window,
                                _free_with_state(head, draft, wrong, eos), eos))
            payload = {"seed": seed, "prompt_id": prompt.id,
                       "source": prompt.task_type, "family_id": prompt.family_id,
                       "val_lock_sha256": sha256_file(output / "val_analysis_lock.json"),
                       "donor_lock_sha256": sha256_file(output / "val_donor_lock.json"),
                       "rows": rows}
            atomic_json(path, payload)
            ledger["last_val_scored_prompt_id"] = prompt.id
            ledger["val_scored_prompt_seed_pairs"] = ledger.get(
                "val_scored_prompt_seed_pairs", 0) + 1
            atomic_json(output / "pilot_ledger.json", ledger)
        del heads
        torch.cuda.empty_cache()
    return True


def _statistics(output, prompts, val_windows, eos):
    path = output / "val_statistics.json"
    if path.exists():
        return _read_json(path)
    score_dir = output / "val_scores"
    rows = []
    for seed in SEEDS:
        for prompt in prompts:
            item = _read_json(score_dir / f"seed{seed}" / _chunk_name(prompt))
            if item.get("prompt_id") != prompt.id or item.get("seed") != seed:
                raise ValueError("val score prompt/seed mismatch")
            expected_windows = sum(w.observation.prompt_id == prompt.id
                                   for w in val_windows)
            for arm in ("raw_draft", *ARM_ORDER, "F-H-zero"):
                if sum(row["arm"] == arm for row in item["rows"]) != expected_windows:
                    raise ValueError("val score arm/window coverage mismatch")
            rows.extend(item["rows"])
    scores = prompt_scores(rows, eligible_only=True)
    comparisons = [("F-H", "C"), ("F-H", "B"),
                   ("F-CURRENT-TOP32", "C"),
                   ("F-CURRENT-TOP32", "B")]
    intervals = {f"seed{seed}": {
        f"{left}-{right}": paired_family_ci(
            scores, seed=seed, left=left, right=right)
        for left, right in comparisons} for seed in SEEDS}
    # A seed-mean interval is conditional on the three fixed training runs.
    # Average matched prompt scores across seeds, then resample whole families.
    collapsed = defaultdict(list)
    for (seed, source, family, prompt, arm), value in scores.items():
        collapsed[(source, family, prompt, arm)].append(value)
    if any(len(values) != 3 for values in collapsed.values()):
        raise ValueError("seed mean lacks three matched prompt scores")
    mean_scores = {(0, *key): sum(values) / 3
                   for key, values in collapsed.items()}
    seed_mean = {f"{left}-{right}": paired_family_ci(
        mean_scores, seed=0, left=left, right=right)
        for left, right in comparisons}
    matched = {row["window_id"] for row in rows
               if row["arm"] == "F-H-stratified_shuffle"}
    matched_rows = [row for row in rows if row["window_id"] in matched]
    matched_scores = prompt_scores(matched_rows, eligible_only=True)
    negative_intervals = {}
    if matched_scores:
        for seed in SEEDS:
            negative_intervals[str(seed)] = paired_family_ci(
                matched_scores, seed=seed, left="F-H",
                right="F-H-stratified_shuffle")
    zero_intervals = {str(seed): paired_family_ci(
        scores, seed=seed, left="F-H", right="F-H-zero")
        for seed in SEEDS}
    primary_prompts = {w.observation.prompt_id for w in val_windows
                       if full_window_eligible(w.labels, eos)}
    by_source = defaultdict(lambda: {"prompts": set(), "primary_prompts": set(),
                                     "windows": 0, "primary_windows": 0})
    for prompt in prompts:
        by_source[prompt.task_type]["prompts"].add(prompt.id)
    for w in val_windows:
        bucket = by_source[w.observation.task_type]
        bucket["windows"] += 1
        if full_window_eligible(w.labels, eos):
            bucket["primary_windows"] += 1
            bucket["primary_prompts"].add(w.observation.prompt_id)
    coverage = {source: {
        "prompts": len(x["prompts"]),
        "primary_prompts": len(x["primary_prompts"]),
        "windows": x["windows"], "primary_windows": x["primary_windows"]}
        for source, x in by_source.items()}
    all_scores = prompt_scores(rows, eligible_only=False)
    arm_mean = {}
    arm_source_mean = {}
    all_window_arm_mean = {}
    next_token_top1 = {}
    teacher_forced_ce = {}
    first_rejection_histogram = {}
    for seed in SEEDS:
        arm_mean[str(seed)] = {}
        arm_source_mean[str(seed)] = {}
        all_window_arm_mean[str(seed)] = {}
        next_token_top1[str(seed)] = {}
        teacher_forced_ce[str(seed)] = {}
        first_rejection_histogram[str(seed)] = {}
        for arm in ("raw_draft", *ARM_ORDER, "F-H-zero"):
            values = [v for (s, _, _, _, a), v in scores.items()
                      if s == seed and a == arm]
            arm_mean[str(seed)][arm] = sum(values) / len(values) if values else None
            all_values = [v for (s, _, _, _, a), v in all_scores.items()
                          if s == seed and a == arm]
            all_window_arm_mean[str(seed)][arm] = (
                sum(all_values) / len(all_values) if all_values else None)
            arm_source_mean[str(seed)][arm] = {
                source: (sum(v for (s, src, _, _, a), v in scores.items()
                             if s == seed and src == source and a == arm) /
                         sum(s == seed and src == source and a == arm
                             for (s, src, _, _, a) in scores))
                for source in by_source
                if any(s == seed and src == source and a == arm
                       for (s, src, _, _, a) in scores)}
            selected = [row for row in rows if row["seed"] == seed
                        and row["arm"] == arm and row["full_window_eligible"]]
            next_token_top1[str(seed)][arm] = (
                sum(row["next_token_top1_exact"] for row in selected) / len(selected)
                if selected else None)
            values_ce = [row["teacher_forced_CE_secondary"] for row in selected
                         if "teacher_forced_CE_secondary" in row]
            teacher_forced_ce[str(seed)][arm] = (
                sum(values_ce) / len(values_ce) if values_ce else None)
            histogram = defaultdict(int)
            for row in selected:
                key = str(row["first_rejection_position"])
                histogram[key] += 1
            first_rejection_histogram[str(seed)][arm] = dict(histogram)
    horizon = defaultdict(int)
    for window in val_windows:
        horizon[str(len(window.labels))] += 1
    result = {
        "scope": "v11_complete_200_prompt_fixed_prefix_probe",
        "primary_eligibility": "true_future_len8_without_eos",
        "seed_arm_prompt_mean_A_probe": arm_mean,
        "seed_arm_source_prompt_mean_A_probe": arm_source_mean,
        "seed_arm_all_legal_window_prompt_mean_acceptance": all_window_arm_mean,
        "seed_arm_next_token_top1_window_rate": next_token_top1,
        "seed_arm_teacher_forced_CE_window_mean": teacher_forced_ce,
        "seed_arm_first_rejection_histogram_primary_windows": first_rejection_histogram,
        "paired_seed_intervals": intervals,
        "paired_seed_mean_intervals": seed_mean,
        "teacher_correct_minus_zero": zero_intervals,
        "teacher_correct_minus_stratified_shuffle_on_matched": negative_intervals,
        "matched_teacher_window_ids": len(matched),
        "coverage_by_source": coverage,
        "prompts_total": len(prompts),
        "prompts_with_primary_window": len(primary_prompts),
        "valid_windows": len(val_windows),
        "future_label_horizon_distribution": dict(horizon),
        "primary_windows": sum(full_window_eligible(w.labels, eos)
                               for w in val_windows),
        "row_count_all_seeds_arms_controls": len(rows),
        "seed_mean_CI_conditions_on_three_fixed_seeds": True,
        "not_a_WAN_or_speed_result": True,
    }
    atomic_json(path, result)
    return result


def _closed_loop(args, output, prompts, target, draft, draft_model,
                 config, train_lock, eos, time_left):
    directory = output / "closed_loop"
    directory.mkdir(exist_ok=True)
    train_ids = _read_json(output / "val_analysis_lock.json")[
        "closed_loop_train_prompt_ids"]
    val_ids = _read_json(output / "val_analysis_lock.json")[
        "closed_loop_val_prompt_ids"]
    inventory = {p.id: p for split in prompts.values() for p in split}
    chosen = [inventory[key] for key in train_ids + val_ids]
    if len(chosen) != 4:
        raise ValueError("closed-loop prompt lock differs")
    for mode in ARM_ORDER:
        head = _load_final_head(output, 42, mode, config,
                                draft_model, train_lock)
        for prompt in chosen:
            path = directory / f"{mode}.{_chunk_name(prompt)}"
            if path.exists():
                if _read_json(path).get("exact") is not True:
                    raise RuntimeError("stored closed-loop request is not exact")
                continue
            if time_left() < 180:
                return False
            run = run_v10_closed_loop(
                target=target, draft=draft, head=head,
                prompt=prompt, max_new_tokens=32, eos_token_id=eos).to_json()
            atomic_json(path, run)
            if run.get("exact") is not True:
                raise RuntimeError(f"closed-loop differs from independent AR: {prompt.id}")
        del head
        torch.cuda.empty_cache()
    return True


def main():
    parser = argparse.ArgumentParser()
    for name in ("v10_source_manifest", "v10_source_zip", "v11_source_manifest",
                 "v11_source_zip", "parent", "parent_meta", "v9_p0_report",
                 "v9_inputs", "prior_plan", "target", "draft", "v10_evidence",
                 "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--v10-source-sha256", required=True)
    parser.add_argument("--v11-source-sha256", required=True)
    parser.add_argument("--deadline-seconds", type=int, default=21600)
    args = parser.parse_args()
    if not 0 < args.deadline_seconds <= MAX_SEGMENT_SECONDS:
        raise ValueError("one GPU worker segment may not exceed six hours")
    begin = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    # Never overwrite a crashed worker's lock.  A human must reconcile the
    # real PID and state before any later exact time-slice resume.
    active = args.output / "active_worker.json"
    if active.exists():
        raise RuntimeError("v11 worker lock exists; inspect live PID before resume")
    meta, model_hashes, source_files = _check_source(args)
    prompts = locked_prompts(args.parent, args.parent_meta)
    ledger = _init_ledger(args, args.output, meta, model_hashes,
                          source_files, prompts)
    if ledger.get("state") in ("failed", "completed"):
        raise RuntimeError("terminal v11 worker ledger cannot be restarted")
    if ledger["segments_finished"] >= 6:
        raise RuntimeError("v11 six-segment budget exhausted")
    if ledger["gpu_wall_seconds_finished_segments"] >= MAX_TOTAL_GPU_SECONDS:
        raise RuntimeError("v11 cumulative GPU wall budget exhausted")
    segment = ledger["segments_finished"] + 1
    budget_left = MAX_TOTAL_GPU_SECONDS - ledger["gpu_wall_seconds_finished_segments"]
    deadline = min(float(args.deadline_seconds), budget_left)
    def time_left():
        return deadline - (time.monotonic() - begin)
    # Exclusive creation closes the race between two launchers that both saw
    # the prelaunch check pass.  A crash leaves this file for manual PID audit.
    with active.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "segment": segment,
                   "program_sha256": PROGRAM_SHA256,
                   "v11_source_zip_sha256": args.v11_source_sha256,
                   "started_unix_s": time.time()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    segment_path = args.output / "segments" / f"segment{segment}.json"
    segment_path.parent.mkdir(exist_ok=True)
    status = {"state": "running", "segment": segment,
              "pid": os.getpid(), "started_unix_s": time.time(),
              "program_sha256": PROGRAM_SHA256,
              "v11_source_zip_sha256": args.v11_source_sha256}
    atomic_json(args.output / "status.json", status)
    def phase(name):
        ledger["phase"] = name
        atomic_json(args.output / "pilot_ledger.json", ledger)
        status["phase"] = name
        status["elapsed_seconds"] = time.monotonic() - begin
        atomic_json(args.output / "status.json", status)
    try:
        phase("real_gpu0_admission")
        status["gpu_admission"] = _admit_idle_gpu("cuda:0", 22000)
        atomic_json(args.output / "status.json", status)
        phase("load_frozen_models")
        eos = meta["target_eos_token_id"]
        if eos != 151645:
            raise ValueError("Target EOS differs")
        target, draft, draft_model, config = _load_models(args, eos)
        base_before = frozen_sample(target.model), frozen_sample(draft_model)
        old, _ = _check_v10(args, args.output, prompts, meta, model_hashes)
        plans = _plans(args, args.output, prompts, meta, model_hashes,
                       source_files)
        phase("same_worker_teacher_consumption")
        _diagnostic(args, args.output, draft, draft_model, config, old, eos)
        old_ids = {w.observation.prompt_id for w in old}
        phase("collect_all_train1000_exact")
        if not _collect_split(args, args.output, "train", prompts["train"],
                              plans["train"][1], target, draft, eos,
                              old_ids=old_ids, time_left=time_left, ledger=ledger):
            status["state"] = "paused_exact_time_slice"
            return
        train_windows, train_inventory = _load_split(
            args.output, "train", prompts["train"], plans["train"][1], eos,
            old_windows=old)
        inventory_path = args.output / "train_inventory.json"
        if inventory_path.exists() and _read_json(inventory_path) != train_inventory:
            raise ValueError("train inventory differs on resume")
        atomic_json(inventory_path, train_inventory)
        phase("freeze_full_train_exposure_and_val_scope")
        train_lock = _training_lock(args, args.output, train_windows,
                                    config, draft_model)
        _eval_lock(args, args.output, train_lock, prompts)
        phase("train_three_seeds_four_matched_arms_three_epochs")
        if not _train_all(args, args.output, train_windows, config, draft,
                          draft_model, train_lock, eos, time_left, ledger):
            status["state"] = "paused_exact_time_slice"
            return
        phase("collect_all_val200_exact_after_lock")
        if not _collect_split(args, args.output, "val", prompts["val"],
                              plans["val"][1], target, draft, eos,
                              old_ids=set(), time_left=time_left, ledger=ledger):
            status["state"] = "paused_exact_time_slice"
            return
        val_windows, val_inventory = _load_split(
            args.output, "val", prompts["val"], plans["val"][1], eos)
        val_path = args.output / "val_inventory.json"
        if val_path.exists() and _read_json(val_path) != val_inventory:
            raise ValueError("val inventory differs on resume")
        atomic_json(val_path, val_inventory)
        phase("independent_val_fixed_prefix_three_seeds")
        if not _evaluate_all(args, args.output, val_windows, prompts["val"],
                             config, draft, draft_model, train_lock, eos,
                             time_left, ledger):
            status["state"] = "paused_exact_time_slice"
            return
        if time_left() < 1200:
            status["state"] = "paused_exact_time_slice"
            return
        phase("paired_prompt_family_bootstrap_10000")
        _statistics(args.output, prompts["val"], val_windows, eos)
        phase("fixed_16_closed_loop_exact")
        if not _closed_loop(args, args.output, prompts, target, draft,
                            draft_model, config, train_lock, eos, time_left):
            status["state"] = "paused_exact_time_slice"
            return
        phase("final_frozen_model_integrity")
        if base_before != (frozen_sample(target.model), frozen_sample(draft_model)):
            raise RuntimeError("Target/Draft frozen base sample changed")
        ledger["state"] = "completed"
        ledger["phase"] = "complete"
        status["state"] = "completed"
    except Exception as exc:
        ledger["state"] = "failed"
        ledger["error_type"] = type(exc).__name__
        ledger["error"] = str(exc)
        status["state"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        elapsed = time.monotonic() - begin
        ledger["gpu_wall_seconds_finished_segments"] += elapsed
        ledger["segments_finished"] += 1
        atomic_json(args.output / "pilot_ledger.json", ledger)
        status["elapsed_seconds"] = elapsed
        status["ended_unix_s"] = time.time()
        atomic_json(args.output / "status.json", status)
        atomic_json(segment_path, {
            "segment": segment, "state": status["state"],
            "pid": os.getpid(), "elapsed_seconds": elapsed,
            "cumulative_gpu_worker_wall_seconds": ledger[
                "gpu_wall_seconds_finished_segments"],
            "phase": ledger["phase"],
            "train_collected_new": ledger["train_collected_new"],
            "val_collected": ledger["val_collected"],
            "training_updates": ledger["training_updates"],
            "program_sha256": PROGRAM_SHA256,
            "v11_source_zip_sha256": args.v11_source_sha256,
        })
        active.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
