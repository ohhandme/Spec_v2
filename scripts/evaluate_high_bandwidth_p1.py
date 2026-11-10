"""Evaluate P1-B F's genuine AR fixed-prefix acceptance on frozen val only.

The protected generator, acceptance rule and prompt bootstrap are imported
unchanged.  Original C/D/E rows are hash-verified and reused.  F's complete
continuous cloud distribution is serialized into 8192 bytes each window.
This is a mechanism upper bound, not a latency or 8-byte deployment claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig
from feedback_codec.fixed_probe import (
    EVAL_PROTOCOL_VERSION, ProbeExample, _accepted_prefix, _generate_probe,
    _paired_prompt_ci, _prompt_grouped_summary,
)
from feedback_codec.hf_backend import _FrozenHFBackend, HFDraftBackend
from scripts.evaluate_fixed_probe import _model_hashes, _validate_collection_sources
from scripts.evaluate_no_hidden_p1 import _reference_rows
from scripts.high_bandwidth_control import (
    HighBandwidthP1Codec, MESSAGE_BYTES, high_bandwidth_arm,
)
from scripts.train_fixed_probe import _atomic_json, _sha256


class _FrozenDraftFeatures(_FrozenHFBackend):
    teacher_hidden = HFDraftBackend.teacher_hidden


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("draft", "corpus", "collection-protocol",
                 "reference-training-protocol", "reference-evaluation-report",
                 "reference-evaluation-protocol", "f-training-protocol",
                 "f-checkpoint", "f-gpu-p0-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _verify_f_source_chain(f_training: dict, gate: dict) -> None:
    gate_script = "scripts/check_high_bandwidth_gpu_p0.py"
    for name in ("scripts/high_bandwidth_control.py", gate_script,
                 "scripts/train_high_bandwidth_p1.py",
                 "scripts/train_no_hidden_p1.py",
                 "feedback_codec/codec.py", "feedback_codec/training.py",
                 "feedback_codec/fixed_probe.py"):
        if f_training.get("source_sha256", {}).get(name) != _sha256(ROOT / name):
            raise ValueError(f"F source differs from training: {name}")
    if (gate.get("source_sha256", {}).get(gate_script)
            != f_training["source_sha256"][gate_script]):
        raise ValueError("F GPU gate script differs from training provenance")


def main() -> None:
    args = _args()
    for path in (args.draft, args.corpus, args.collection_protocol,
                 args.reference_training_protocol, args.reference_evaluation_report,
                 args.reference_evaluation_protocol, args.f_training_protocol,
                 args.f_checkpoint, args.f_gpu_p0_report):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("F fixed probe requires real CUDA")
    collection = json.loads(args.collection_protocol.read_text(encoding="utf-8"))
    if (collection.get("eval_protocol_version") != EVAL_PROTOCOL_VERSION
            or collection.get("split") != "val" or collection.get("window") != 8
            or collection.get("fixed_prefix_sha256") != _sha256(args.corpus)
            or collection.get("draft_file_sha256") != _model_hashes(args.draft)):
        raise ValueError("F val collection or Draft model differs from frozen P1-A")
    _validate_collection_sources(collection)
    reference_training = json.loads(args.reference_training_protocol.read_text(
        encoding="utf-8"))
    f_training = json.loads(args.f_training_protocol.read_text(encoding="utf-8"))
    gate = json.loads(args.f_gpu_p0_report.read_text(encoding="utf-8"))
    if (f_training.get("scope") != "P1-B_F_continuous_training"
            or f_training.get("wire_bytes") != MESSAGE_BYTES
            or f_training.get("wire_dtype") != "little_endian_float32"
            or gate.get("state") != "passed"
            or gate.get("scope") != "P1-B_F_continuous_real_gpu_gate"
            or f_training.get("f_gpu_p0_report_sha256")
               != _sha256(args.f_gpu_p0_report)
            or f_training.get("reference_training_protocol_sha256")
               != _sha256(args.reference_training_protocol)
            or f_training.get("collection_corpus_sha256")
               != reference_training.get("collection_corpus_sha256")
            or f_training.get("draft_model_file_sha256")
               != collection["draft_file_sha256"]
            or f_training.get("target_model_file_sha256_from_collection")
               != collection["target_file_sha256"]
            or f_training.get("config") != reference_training.get("config")
            or f_training.get("steps_per_arm")
               != reference_training.get("steps_per_arm")
            or f_training.get("seeds") != reference_training.get("seeds")
            or f_training.get("learning_rate")
               != reference_training.get("learning_rate")
            or f_training.get("weight_decay")
               != reference_training.get("weight_decay")
            or f_training.get("prefix_weight")
               != reference_training.get("prefix_weight")):
        raise ValueError("F and original D training budgets or gate differ")
    _verify_f_source_chain(f_training, gate)
    ref_eval_protocol = json.loads(args.reference_evaluation_protocol.read_text(
        encoding="utf-8"))
    if (ref_eval_protocol.get("result_sha256")
            != _sha256(args.reference_evaluation_report)
            or ref_eval_protocol.get("source_corpus_sha256") != _sha256(args.corpus)
            or ref_eval_protocol.get("source_training_protocol_sha256")
               != _sha256(args.reference_training_protocol)):
        raise ValueError("original D fixed val evaluation provenance differs")
    baseline = json.loads(args.reference_evaluation_report.read_text(encoding="utf-8"))
    examples = [ProbeExample.from_json(json.loads(line)) for line in
                args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples:
        raise ValueError("F val corpus is empty")
    by_arm = _reference_rows(baseline, examples, collection["eos_token_id"])
    payload = torch.load(args.f_checkpoint, map_location="cpu", weights_only=True)
    seed = payload.get("seed")
    reference_d = f_training["reference_d"].get(str(seed))
    if (payload.get("format") != "matched_p1_F_continuous_v1"
            or payload.get("mode") != "target_8B"
            or type(seed) is not int or seed not in reference_training["seeds"]
            or payload.get("training_protocol_sha256")
               != _sha256(args.f_training_protocol)
            or payload.get("reference_training_protocol_sha256")
               != _sha256(args.reference_training_protocol)
            or payload.get("source_corpus_sha256")
               != reference_training["collection_corpus_sha256"]
            or payload.get("draft_model_file_sha256")
               != collection["draft_file_sha256"]
            or payload.get("steps") != reference_training["steps_per_arm"]
            or payload.get("wire_bytes") != MESSAGE_BYTES
            or payload.get("wire_dtype") != "little_endian_float32"
            or not isinstance(reference_d, dict)
            or payload.get("reference_d_checkpoint_sha256")
               != ref_eval_protocol["checkpoint_sha256"]["target_8B"]
            or payload.get("reference_d_checkpoint_sha256")
               != reference_d["checkpoint_sha256"]
            or payload.get("exposure_sha256") != reference_d["exposure_sha256"]
            or payload.get("edge_initial_sha256")
               != reference_d["edge_initial_sha256"]):
        raise ValueError("F checkpoint is not fair/traceable to original D")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", torch_dtype=torch.bfloat16,
    ).to(device)
    draft = _FrozenDraftFeatures(model, tokenizer, device=device, dtype=torch.bfloat16)
    config = CodecConfig(**f_training["config"])
    if (config.draft_hidden_size != draft.hidden_size
            or config.vocab_size != draft.vocab_size):
        raise ValueError("F checkpoint architecture differs from Draft")
    codec = HighBandwidthP1Codec(config, model.get_output_embeddings()).to(device)
    codec.load_trainable_state_dict(payload["state"])
    codec.eval()
    arm = high_bandwidth_arm(draft, codec)
    rows = []
    eos = collection["eos_token_id"]
    for item in examples:
        o = item.observation
        prepared = arm.prepare(o.target_view())
        if len(prepared.message) != MESSAGE_BYTES:
            raise RuntimeError("F failed to emit 8192 wire bytes")
        draft_ids = _generate_probe(arm, o, prepared, eos)
        accepted = _accepted_prefix(draft_ids, item.target_future_ids)
        reference = by_arm["target_8B"][o.key]
        if (reference["target_future_ids"] != list(item.target_future_ids)
                or reference["primary_eligible"] != (
                    len(item.target_future_ids) == o.window and
                    (eos is None or eos not in item.target_future_ids))):
            raise RuntimeError("F and original D fixed windows differ")
        rows.append({
            "arm": "target_continuous_F", "prompt_id": o.prompt_id,
            "round_index": o.round_index, "task_type": o.task_type,
            "prior_accepted_count": o.accepted_count,
            "accepted_count": accepted,
            "first_rejection_position": (
                accepted + 1 if accepted < min(len(draft_ids),
                                               len(item.target_future_ids)) else None
            ),
            "draft_ids": list(draft_ids),
            "target_future_ids": list(item.target_future_ids),
            "primary_eligible": reference["primary_eligible"],
            "message_bytes": MESSAGE_BYTES, "donor": None,
        })
    paired = rows + [row for name in (
        "target_8B", "protocol_only", "shuffled_global_8B",
        "shuffled_stratified_8B",
    ) for row in by_arm[name].values()]
    repetitions = ref_eval_protocol["bootstrap_repetitions"]
    shuffle_seed = ref_eval_protocol["bootstrap_seed"]
    comparisons = {
        f"{left}-{right}": _paired_prompt_ci(
            paired, left, right, primary_only=True,
            bootstrap_repetitions=repetitions, seed=shuffle_seed,
        ) for left, right in (
            ("target_continuous_F", "protocol_only"),
            ("target_continuous_F", "target_8B"),
            ("target_8B", "protocol_only"),
            ("target_8B", "shuffled_global_8B"),
            ("target_8B", "shuffled_stratified_8B"),
        )
    }
    report = {
        "scope": "fixed_prefix_continuous_target_upper_bound",
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        "latency_claim_valid": False, "split": "val", "training_seed": seed,
        "wire_bytes_per_window": MESSAGE_BYTES,
        "probe_windows": len(examples),
        "primary_windows": sum(row["primary_eligible"] for row in rows),
        "rows": rows,
        "primary_summary": _prompt_grouped_summary(rows, primary_only=True),
        "all_windows_sensitivity_summary": _prompt_grouped_summary(
            rows, primary_only=False),
        "primary_prompt_paired_comparisons": comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "high_bandwidth_report.json"
    _atomic_json(result_path, report)
    sources = [ROOT / "scripts" / name for name in
               ("high_bandwidth_control.py", "evaluate_high_bandwidth_p1.py",
                "evaluate_no_hidden_p1.py")]
    sources += [ROOT / "feedback_codec" / name for name in
                ("codec.py", "fixed_probe.py", "hf_backend.py")]
    _atomic_json(args.output / "evaluation_protocol.json", {
        "scope": report["scope"], "latency_claim_valid": False,
        "wire_bytes_per_window": MESSAGE_BYTES,
        "source_corpus_sha256": _sha256(args.corpus),
        "collection_protocol_sha256": _sha256(args.collection_protocol),
        "reference_training_protocol_sha256":
            _sha256(args.reference_training_protocol),
        "reference_evaluation_report_sha256":
            _sha256(args.reference_evaluation_report),
        "f_training_protocol_sha256": _sha256(args.f_training_protocol),
        "f_gpu_p0_report_sha256": _sha256(args.f_gpu_p0_report),
        "f_checkpoint_sha256": _sha256(args.f_checkpoint),
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in sources
        },
        "bootstrap_seed": shuffle_seed,
        "bootstrap_repetitions": repetitions,
        "result_sha256": _sha256(result_path),
    })
    print(json.dumps({"output": str(args.output), "comparisons": comparisons}))


if __name__ == "__main__":
    main()
