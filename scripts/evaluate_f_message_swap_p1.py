"""Frozen-checkpoint P1-B diagnostic: swap only F's valid cloud message.

This is a causal message-use probe on the locked validation windows.  It
replays the archived F arm exactly before evaluating the swapped arm.  It
does not train, inspect test, measure latency, or modify the Target oracle.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig
from feedback_codec.fixed_probe import (
    EVAL_PROTOCOL_VERSION, ProbeArm, ProbeExample, _accepted_prefix,
    _generate_probe, _paired_prompt_ci, _prompt_grouped_summary,
)
from scripts.evaluate_fixed_probe import _model_hashes, _validate_collection_sources
from scripts.evaluate_high_bandwidth_p1 import (
    _FrozenDraftFeatures, _verify_f_source_chain,
)
from scripts.evaluate_no_hidden_p1 import _reference_rows
from scripts.high_bandwidth_control import (
    HighBandwidthP1Codec, HighBandwidthState, MESSAGE_BYTES,
    high_bandwidth_arm, unpack_distribution,
)
from scripts.train_fixed_probe import _atomic_json, _sha256


Key = tuple[str, int]
LOCK_SHA256 = "c3ae2fba6aeb33c17134146f313af4fc0647153619c6ae9c963a67b223f251e2"
RESULT_SOURCE_FILES = (
    "scripts/evaluate_f_message_swap_p1.py",
    "scripts/evaluate_high_bandwidth_p1.py",
    "scripts/high_bandwidth_control.py",
    "scripts/evaluate_no_hidden_p1.py",
    "scripts/evaluate_fixed_probe.py",
    "scripts/train_fixed_probe.py",
    "feedback_codec/codec.py",
    "feedback_codec/fixed_probe.py",
    "feedback_codec/hf_backend.py",
    "feedback_codec/decoding.py",
    "feedback_codec/training.py",
)


def _source_snapshot() -> dict[str, str]:
    return {name: _sha256(ROOT / name) for name in RESULT_SOURCE_FILES}


def _verify_lock_file(path: Path) -> None:
    if _sha256(path) != LOCK_SHA256:
        raise ValueError("message-swap protocol lock differs from preregistered SHA")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "draft", "corpus", "collection-protocol",
        "reference-training-protocol", "reference-evaluation-report",
        "reference-evaluation-protocol", "f-training-protocol",
        "f-checkpoint", "f-gpu-p0-report", "f-evaluation-report",
        "f-evaluation-protocol", "lock-file", "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--smoke-windows", type=int,
        help="Run 1-8 frozen windows as a correctness gate; omit for all 567.",
    )
    return parser.parse_args()


def _select_donors(observations: Sequence[Any]) -> dict[Key, Key]:
    """Cycle within the locked stratum to the next *different prompt*.

    The input order is irrelevant; prompt_id and round_index provide a stable
    total order.  Every probe is retained, including terminal windows.
    """
    groups: dict[tuple[str, int], list[Key]] = defaultdict(list)
    seen: set[Key] = set()
    for o in observations:
        if (type(o.prompt_id) is not str or type(o.round_index) is not int
                or type(o.task_type) is not str
                or type(o.accepted_count) is not int):
            raise TypeError("invalid donor observation")
        key = (o.prompt_id, o.round_index)
        if key in seen:
            raise ValueError("duplicate probe key")
        seen.add(key)
        groups[(o.task_type, o.accepted_count)].append(key)
    donors: dict[Key, Key] = {}
    for group in groups.values():
        keys = sorted(group)
        if len({key[0] for key in keys}) < 2:
            raise ValueError("no different-prompt donor within locked stratum")
        for index, key in enumerate(keys):
            for offset in range(1, len(keys)):
                donor = keys[(index + offset) % len(keys)]
                if donor[0] != key[0]:
                    donors[key] = donor
                    break
            else:
                raise ValueError("no different-prompt donor within locked stratum")
    if len(donors) != len(observations):
        raise RuntimeError("donor map does not cover every fixed probe")
    return donors


def _validate_archived_f(
    report: dict, protocol: dict, examples: Sequence[ProbeExample],
    eos_token_id: int | None,
) -> dict[Key, dict]:
    if (report.get("scope") != "fixed_prefix_continuous_target_upper_bound"
            or report.get("split") != "val"
            or report.get("eval_protocol_version") != EVAL_PROTOCOL_VERSION
            or report.get("wire_bytes_per_window") != MESSAGE_BYTES
            or report.get("probe_windows") != len(examples)
            or report.get("primary_windows") != sum(
                len(x.target_future_ids) == x.observation.window
                and (eos_token_id is None
                     or eos_token_id not in x.target_future_ids)
                for x in examples)):
        raise ValueError("archived F report differs from locked validation")
    if (protocol.get("scope") != report["scope"]
            or protocol.get("source_corpus_sha256") is None
            or protocol.get("wire_bytes_per_window") != MESSAGE_BYTES):
        raise ValueError("archived F evaluation protocol is invalid")
    if any(x.observation.split != "val" for x in examples):
        raise ValueError("message swap can only read the locked val split")
    expected = {x.observation.key: x for x in examples}
    if len(expected) != len(examples):
        raise ValueError("fixed validation contains duplicate keys")
    rows: dict[Key, dict] = {}
    for row in report["rows"]:
        key = (row["prompt_id"], row["round_index"])
        if key not in expected or key in rows or row.get("arm") != "target_continuous_F":
            raise ValueError("archived F has missing/duplicate/wrong-arm row")
        item = expected[key]
        draft_ids = tuple(row["draft_ids"])
        target_ids = item.target_future_ids
        accepted = _accepted_prefix(draft_ids, target_ids)
        first_reject = (
            accepted + 1 if accepted < min(len(draft_ids), len(target_ids))
            else None
        )
        primary = (
            len(target_ids) == item.observation.window
            and (eos_token_id is None or eos_token_id not in target_ids)
        )
        if (row["target_future_ids"] != list(target_ids)
                or row["accepted_count"] != accepted
                or row["first_rejection_position"] != first_reject
                or row["primary_eligible"] != primary
                or row["message_bytes"] != MESSAGE_BYTES
                or row["prior_accepted_count"] != item.observation.accepted_count
                or row["task_type"] != item.observation.task_type):
            raise ValueError("archived F row fails independent token audit")
        rows[key] = row
    if set(rows) != set(expected):
        raise ValueError("archived F lacks fixed validation windows")
    return rows


def _swap_state(local: HighBandwidthState,
                donor: HighBandwidthState) -> HighBandwidthState:
    if type(local) is not HighBandwidthState or type(donor) is not HighBandwidthState:
        raise TypeError("F swap requires two valid prepared states")
    return HighBandwidthState(local=local.local, message=donor.message)


def _validate_inputs(args: argparse.Namespace) -> tuple[
    dict, dict, dict, dict, dict, dict, list[ProbeExample], dict,
    dict[Key, dict], dict[str, str],
]:
    for name in (
        "draft", "corpus", "collection_protocol",
        "reference_training_protocol", "reference_evaluation_report",
        "reference_evaluation_protocol", "f_training_protocol",
        "f_checkpoint", "f_gpu_p0_report", "f_evaluation_report",
        "f_evaluation_protocol", "lock_file",
    ):
        if not getattr(args, name).exists():
            raise FileNotFoundError(getattr(args, name))
    if args.output.exists():
        raise FileExistsError(args.output)
    _verify_lock_file(args.lock_file)
    source_start = _source_snapshot()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("F message swap requires real CUDA")
    collection = json.loads(args.collection_protocol.read_text(encoding="utf-8"))
    if (collection.get("eval_protocol_version") != EVAL_PROTOCOL_VERSION
            or collection.get("split") != "val"
            or collection.get("window") != 8
            or collection.get("fixed_prefix_sha256") != _sha256(args.corpus)
            or collection.get("draft_file_sha256") != _model_hashes(args.draft)):
        raise ValueError("F message swap collection differs from frozen P1-A")
    _validate_collection_sources(collection)
    original_training = json.loads(args.reference_training_protocol.read_text(
        encoding="utf-8"))
    f_training = json.loads(args.f_training_protocol.read_text(encoding="utf-8"))
    gate = json.loads(args.f_gpu_p0_report.read_text(encoding="utf-8"))
    if (f_training.get("scope") != "P1-B_F_continuous_training"
            or f_training.get("wire_bytes") != MESSAGE_BYTES
            or f_training.get("wire_dtype") != "little_endian_float32"
            or f_training.get("f_gpu_p0_report_sha256")
               != _sha256(args.f_gpu_p0_report)
            or f_training.get("reference_training_protocol_sha256")
               != _sha256(args.reference_training_protocol)
            or f_training.get("collection_corpus_sha256")
               != original_training.get("collection_corpus_sha256")
            or f_training.get("draft_model_file_sha256")
               != collection["draft_file_sha256"]
            or f_training.get("target_model_file_sha256_from_collection")
               != collection["target_file_sha256"]
            or f_training.get("config") != original_training.get("config")
            or f_training.get("steps_per_arm")
               != original_training.get("steps_per_arm")
            or f_training.get("seeds") != original_training.get("seeds")
            or f_training.get("learning_rate")
               != original_training.get("learning_rate")
            or f_training.get("weight_decay")
               != original_training.get("weight_decay")
            or f_training.get("prefix_weight")
               != original_training.get("prefix_weight")
            or gate.get("state") != "passed"
            or gate.get("scope") != "P1-B_F_continuous_real_gpu_gate"):
        raise ValueError("F training or real GPU gate differs from locked run")
    _verify_f_source_chain(f_training, gate)
    ref_protocol = json.loads(args.reference_evaluation_protocol.read_text(
        encoding="utf-8"))
    if (ref_protocol.get("result_sha256")
            != _sha256(args.reference_evaluation_report)
            or ref_protocol.get("source_corpus_sha256") != _sha256(args.corpus)
            or ref_protocol.get("source_training_protocol_sha256")
               != _sha256(args.reference_training_protocol)):
        raise ValueError("original C/D reference provenance differs")
    f_protocol = json.loads(args.f_evaluation_protocol.read_text(encoding="utf-8"))
    expected_f_sources = {
        "scripts/high_bandwidth_control.py",
        "scripts/evaluate_high_bandwidth_p1.py",
        "scripts/evaluate_no_hidden_p1.py",
        "feedback_codec/codec.py",
        "feedback_codec/fixed_probe.py",
        "feedback_codec/hf_backend.py",
    }
    if (f_protocol.get("result_sha256") != _sha256(args.f_evaluation_report)
            or f_protocol.get("source_corpus_sha256") != _sha256(args.corpus)
            or f_protocol.get("collection_protocol_sha256")
               != _sha256(args.collection_protocol)
            or f_protocol.get("reference_training_protocol_sha256")
               != _sha256(args.reference_training_protocol)
            or f_protocol.get("reference_evaluation_report_sha256")
               != _sha256(args.reference_evaluation_report)
            or f_protocol.get("f_training_protocol_sha256")
               != _sha256(args.f_training_protocol)
            or f_protocol.get("f_gpu_p0_report_sha256")
               != _sha256(args.f_gpu_p0_report)
            or set(f_protocol.get("source_sha256", {})) != expected_f_sources
            or any(source_start[name] != digest for name, digest in
                   f_protocol["source_sha256"].items())):
        raise ValueError("archived F evaluation provenance differs")
    baseline = json.loads(args.reference_evaluation_report.read_text(
        encoding="utf-8"))
    f_report = json.loads(args.f_evaluation_report.read_text(encoding="utf-8"))
    examples = [
        ProbeExample.from_json(json.loads(line))
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (len(examples) != 567
            or len({x.observation.prompt_id for x in examples}) != 64
            or sum(
                len(x.target_future_ids) == x.observation.window
                and (collection["eos_token_id"] is None
                     or collection["eos_token_id"] not in x.target_future_ids)
                for x in examples
            ) != 427):
        raise ValueError("locked val must contain 64 prompts, 567 windows, 427 primary")
    by_arm = _reference_rows(baseline, examples, collection["eos_token_id"])
    archived_f = _validate_archived_f(
        f_report, f_protocol, examples, collection["eos_token_id"],
    )
    payload = torch.load(args.f_checkpoint, map_location="cpu", weights_only=True)
    seed = payload.get("seed")
    original_d = f_training["reference_d"].get(str(seed))
    if (payload.get("format") != "matched_p1_F_continuous_v1"
            or payload.get("mode") != "target_8B"
            or type(seed) is not int or seed not in original_training["seeds"]
            or f_report.get("training_seed") != seed
            or payload.get("training_protocol_sha256")
               != _sha256(args.f_training_protocol)
            or payload.get("reference_training_protocol_sha256")
               != _sha256(args.reference_training_protocol)
            or payload.get("source_corpus_sha256")
               != original_training["collection_corpus_sha256"]
            or payload.get("draft_model_file_sha256")
               != collection["draft_file_sha256"]
            or payload.get("steps") != original_training["steps_per_arm"]
            or payload.get("wire_bytes") != MESSAGE_BYTES
            or payload.get("wire_dtype") != "little_endian_float32"
            or not isinstance(original_d, dict)
            or payload.get("reference_d_checkpoint_sha256")
               != ref_protocol["checkpoint_sha256"]["target_8B"]
            or payload.get("reference_d_checkpoint_sha256")
               != original_d["checkpoint_sha256"]
            or payload.get("exposure_sha256") != original_d["exposure_sha256"]
            or payload.get("edge_initial_sha256")
               != original_d["edge_initial_sha256"]
            or f_protocol.get("f_checkpoint_sha256")
               != _sha256(args.f_checkpoint)):
        raise ValueError("F checkpoint differs from archived fair training")
    return (
        collection, original_training, f_training, ref_protocol, f_protocol,
        payload, examples, by_arm, archived_f, source_start,
    )


def main() -> None:
    args = _args()
    if args.smoke_windows is not None and not 1 <= args.smoke_windows <= 8:
        raise ValueError("smoke-windows must be in [1,8]")
    (collection, _, f_training, ref_protocol, f_protocol, payload,
     examples, by_arm, archived_f, source_start) = _validate_inputs(args)
    donors = _select_donors([x.observation for x in examples])
    eval_examples = examples[:args.smoke_windows] if args.smoke_windows else examples
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", torch_dtype=torch.bfloat16,
    ).to(device)
    draft = _FrozenDraftFeatures(model, tokenizer, device=device,
                                 dtype=torch.bfloat16)
    config = CodecConfig(**f_training["config"])
    if (config.draft_hidden_size != draft.hidden_size
            or config.vocab_size != draft.vocab_size):
        raise ValueError("F checkpoint architecture differs from Draft")
    codec = HighBandwidthP1Codec(config, model.get_output_embeddings()).to(device)
    codec.load_trainable_state_dict(payload["state"])
    codec.eval()
    arm = high_bandwidth_arm(draft, codec)
    prepared: dict[Key, HighBandwidthState] = {}
    for item in examples:
        prepared[item.observation.key] = arm.prepare(item.observation.target_view())
    if set(prepared) != set(donors):
        raise RuntimeError("prepared F states do not cover donor map")
    for state in prepared.values():
        if len(state.message) != MESSAGE_BYTES:
            raise RuntimeError("F cloud message has wrong wire length")
        unpack_distribution(state.message, device="cpu")
    eos = collection["eos_token_id"]
    # Replay the complete archived F arm before running any intervention.
    for item in eval_examples:
        o = item.observation
        replay = _generate_probe(arm, o, prepared[o.key], eos)
        archived = archived_f[o.key]
        if (list(replay) != archived["draft_ids"]
                or _accepted_prefix(replay, item.target_future_ids)
                   != archived["accepted_count"]
                or archived["target_future_ids"] != list(item.target_future_ids)
                or by_arm["target_8B"][o.key]["target_future_ids"]
                   != list(item.target_future_ids)):
            raise RuntimeError(f"archived F replay differs at {o.key}")
    rows: list[dict] = []
    changed_messages = 0
    changed_drafts = 0
    changed_acceptance = 0
    for item in eval_examples:
        o = item.observation
        donor_key = donors[o.key]
        own = prepared[o.key]
        donor = prepared[donor_key]
        state = _swap_state(own, donor)
        changed_messages += state.message != own.message
        draft_ids = _generate_probe(arm, o, state, eos)
        accepted = _accepted_prefix(draft_ids, item.target_future_ids)
        original = archived_f[o.key]
        changed_drafts += list(draft_ids) != original["draft_ids"]
        changed_acceptance += accepted != original["accepted_count"]
        rows.append({
            "arm": "target_continuous_F_swap",
            "prompt_id": o.prompt_id, "round_index": o.round_index,
            "task_type": o.task_type, "prior_accepted_count": o.accepted_count,
            "accepted_count": accepted,
            "first_rejection_position": (
                accepted + 1 if accepted < min(len(draft_ids),
                                               len(item.target_future_ids))
                else None
            ),
            "draft_ids": list(draft_ids),
            "target_future_ids": list(item.target_future_ids),
            "primary_eligible": original["primary_eligible"],
            "message_bytes": MESSAGE_BYTES,
            "original_message_sha256": hashlib.sha256(own.message).hexdigest(),
            "donor_message_sha256": hashlib.sha256(donor.message).hexdigest(),
            "donor": {"prompt_id": donor_key[0], "round_index": donor_key[1]},
        })
    diagnostic_state = (
        "valid_swap" if changed_messages else "degenerate_no_wire_change"
    )
    paired = rows + list(archived_f.values()) + list(by_arm["protocol_only"].values())
    comparison_pairs = (
        ("target_continuous_F", "target_continuous_F_swap"),
        ("target_continuous_F_swap", "protocol_only"),
    )
    repetitions = ref_protocol["bootstrap_repetitions"]
    bootstrap_seed = ref_protocol["bootstrap_seed"]
    comparisons = {} if args.smoke_windows else {
        f"{left}-{right}": _paired_prompt_ci(
            paired, left, right, primary_only=True,
            bootstrap_repetitions=repetitions, seed=bootstrap_seed,
        ) for left, right in comparison_pairs
    }
    all_windows_comparisons = {} if args.smoke_windows else {
        f"{left}-{right}": _paired_prompt_ci(
            paired, left, right, primary_only=False,
            bootstrap_repetitions=repetitions, seed=bootstrap_seed,
        ) for left, right in comparison_pairs
    }
    report = {
        "scope": (
            "fixed_prefix_F_causal_message_swap_smoke"
            if args.smoke_windows else "fixed_prefix_F_causal_message_swap"
        ),
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        "split": "val", "latency_claim_valid": False,
        "smoke_only": bool(args.smoke_windows),
        "training_seed": payload["seed"],
        "wire_bytes_per_window": MESSAGE_BYTES,
        "donor_policy": "next_different_prompt_sorted_within_task_type_and_prior_accepted_count",
        "probe_windows": len(eval_examples),
        "reference_probe_windows": len(examples),
        "primary_windows": sum(r["primary_eligible"] for r in rows),
        "archived_F_replay_exact": True,
        "replayed_F_windows": len(eval_examples),
        "diagnostic_state": diagnostic_state,
        "swap_gate_passed": bool(changed_messages),
        "changed_messages": changed_messages,
        "changed_draft_windows": changed_drafts,
        "changed_acceptance_windows": changed_acceptance,
        "rows": rows,
        "primary_summary": _prompt_grouped_summary(rows, primary_only=True),
        "all_windows_sensitivity_summary": _prompt_grouped_summary(
            rows, primary_only=False,
        ),
        "primary_prompt_paired_comparisons": comparisons,
        "all_windows_prompt_paired_comparisons": all_windows_comparisons,
    }
    if _source_snapshot() != source_start:
        raise RuntimeError("result-critical source changed during F swap run")
    _verify_lock_file(args.lock_file)
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "message_swap_report.json"
    _atomic_json(result_path, report)
    _atomic_json(args.output / "evaluation_protocol.json", {
        "scope": report["scope"],
        "latency_claim_valid": False,
        "source_corpus_sha256": _sha256(args.corpus),
        "collection_protocol_sha256": _sha256(args.collection_protocol),
        "reference_evaluation_report_sha256":
            _sha256(args.reference_evaluation_report),
        "reference_evaluation_protocol_sha256":
            _sha256(args.reference_evaluation_protocol),
        "f_training_protocol_sha256": _sha256(args.f_training_protocol),
        "f_gpu_p0_report_sha256": _sha256(args.f_gpu_p0_report),
        "f_checkpoint_sha256": _sha256(args.f_checkpoint),
        "f_evaluation_report_sha256": _sha256(args.f_evaluation_report),
        "f_evaluation_protocol_sha256": _sha256(args.f_evaluation_protocol),
        "lock_file_sha256": LOCK_SHA256,
        "source_sha256": source_start,
        "script_sha256": source_start["scripts/evaluate_f_message_swap_p1.py"],
        "f_runtime_source_sha256": f_protocol["source_sha256"],
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_repetitions": repetitions,
        "result_sha256": _sha256(result_path),
    })
    print(json.dumps({
        "output": str(args.output), "seed": payload["seed"],
        "changed_messages": changed_messages,
        "changed_draft_windows": changed_drafts,
        "changed_acceptance_windows": changed_acceptance,
        "comparisons": comparisons,
    }))


if __name__ == "__main__":
    main()
