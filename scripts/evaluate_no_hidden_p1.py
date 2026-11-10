"""Evaluate a P1-B no-h_T D control against the frozen P1-A val report.

The protected fixed-probe generator, acceptance rule and prompt-bootstrap
functions are imported unchanged. Only the new control is run on the same
frozen corrected prefixes; original C/D rows are verified and reused. This
runner cannot open the sealed test split and makes no latency claim.
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

from feedback_codec.codec import CodecConfig, MatchedP1Codec
from feedback_codec.fixed_probe import (
    EVAL_PROTOCOL_VERSION, ProbeExample, _accepted_prefix, _generate_probe,
    _paired_prompt_ci, _prompt_grouped_summary, matched_p1_arm,
)
from feedback_codec.hf_backend import _FrozenHFBackend, HFDraftBackend
from scripts.evaluate_fixed_probe import _model_hashes, _validate_collection_sources
from scripts.no_hidden_control import assert_no_target_hidden_information
from scripts.train_fixed_probe import _atomic_json, _sha256


class _FrozenDraftFeatures(_FrozenHFBackend):
    teacher_hidden = HFDraftBackend.teacher_hidden


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("draft", "corpus", "collection-protocol",
                 "reference-training-protocol", "reference-evaluation-report",
                 "reference-evaluation-protocol", "control-training-protocol",
                 "control-checkpoint", "no-hidden-gpu-p0-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _reference_rows(report: dict, examples: list[ProbeExample],
                    eos_token_id: int | None) -> dict:
    if (report.get("scope") != "fixed_prefix_probe"
            or report.get("eval_protocol_version") != EVAL_PROTOCOL_VERSION
            or report.get("split") != "val"
            or report.get("probe_windows") != len(examples)
            or report.get("primary_windows") != sum(
                len(item.target_future_ids) == item.observation.window and
                (eos_token_id is None or eos_token_id not in item.target_future_ids)
                for item in examples)):
        raise ValueError("reference evaluation is not the locked val fixed probe")
    expected = {item.observation.key: item for item in examples}
    by_arm: dict[str, dict[tuple[str, int], dict]] = {}
    for row in report["rows"]:
        key = (row["prompt_id"], row["round_index"])
        arm = row["arm"]
        if key not in expected or key in by_arm.setdefault(arm, {}):
            raise ValueError("reference result has missing/duplicate probe keys")
        item = expected[key]
        if row["target_future_ids"] != list(item.target_future_ids):
            raise ValueError("reference result Target continuation differs")
        if row["accepted_count"] != _accepted_prefix(
                tuple(row["draft_ids"]), item.target_future_ids):
            raise ValueError("reference result acceptance does not match tokens")
        by_arm[arm][key] = row
    for arm in ("raw_draft", "edge_only", "protocol_only", "target_8B",
                "shuffled_global_8B", "shuffled_stratified_8B"):
        if set(by_arm.get(arm, {})) != set(expected):
            raise ValueError(f"reference {arm} does not cover all val probes")
    return by_arm


def main() -> None:
    args = _args()
    for path in (args.draft, args.corpus, args.collection_protocol,
                 args.reference_training_protocol, args.reference_evaluation_report,
                 args.reference_evaluation_protocol, args.control_training_protocol,
                 args.control_checkpoint, args.no_hidden_gpu_p0_report):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("real-model no-h_T probe requires CUDA")
    collection = json.loads(args.collection_protocol.read_text(encoding="utf-8"))
    if (collection.get("eval_protocol_version") != EVAL_PROTOCOL_VERSION
            or collection.get("split") != "val" or collection.get("window") != 8
            or collection.get("fixed_prefix_sha256") != _sha256(args.corpus)
            or collection.get("draft_file_sha256") != _model_hashes(args.draft)):
        raise ValueError("val collection or draft model differs from frozen P1-A")
    _validate_collection_sources(collection)
    reference_training = json.loads(args.reference_training_protocol.read_text(
        encoding="utf-8"))
    control_training = json.loads(args.control_training_protocol.read_text(
        encoding="utf-8"))
    no_hidden_gate = json.loads(args.no_hidden_gpu_p0_report.read_text(
        encoding="utf-8"))
    if (control_training.get("scope") != "P1-B_no_target_hidden_training"
            or no_hidden_gate.get("state") != "passed"
            or no_hidden_gate.get("scope") != "P1-B_no_target_hidden_real_gpu_gate"
            or control_training.get("no_hidden_gpu_p0_report_sha256")
               != _sha256(args.no_hidden_gpu_p0_report)
            or control_training.get("reference_training_protocol_sha256")
               != _sha256(args.reference_training_protocol)
            or control_training.get("collection_corpus_sha256")
               != reference_training.get("collection_corpus_sha256")
            or control_training.get("draft_model_file_sha256")
               != collection["draft_file_sha256"]
            or control_training.get("target_model_file_sha256_from_collection")
               != collection["target_file_sha256"]
            or control_training.get("config") != reference_training.get("config")
            or control_training.get("steps_per_arm")
               != reference_training.get("steps_per_arm")
            or control_training.get("seeds") != reference_training.get("seeds")
            or control_training.get("learning_rate")
               != reference_training.get("learning_rate")
            or control_training.get("weight_decay")
               != reference_training.get("weight_decay")
            or control_training.get("prefix_weight")
               != reference_training.get("prefix_weight")):
        raise ValueError("no-h_T and original D training budgets differ")
    ref_eval_protocol = json.loads(args.reference_evaluation_protocol.read_text(
        encoding="utf-8"))
    if (ref_eval_protocol.get("result_sha256")
            != _sha256(args.reference_evaluation_report)
            or ref_eval_protocol.get("source_corpus_sha256") != _sha256(args.corpus)
            or ref_eval_protocol.get("source_training_protocol_sha256")
               != _sha256(args.reference_training_protocol)):
        raise ValueError("original D evaluation provenance differs")
    baseline = json.loads(args.reference_evaluation_report.read_text(encoding="utf-8"))
    examples = [ProbeExample.from_json(json.loads(line)) for line in
                args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples:
        raise ValueError("val corpus is empty")
    by_arm = _reference_rows(baseline, examples, collection["eos_token_id"])
    payload = torch.load(args.control_checkpoint, map_location="cpu", weights_only=True)
    seed = payload.get("seed")
    reference_d = control_training["reference_d"].get(str(seed))
    if (payload.get("format") != "matched_p1_no_hidden_v1"
            or payload.get("mode") != "target_8B"
            or type(seed) is not int or seed not in reference_training["seeds"]
            or payload.get("training_protocol_sha256")
               != _sha256(args.control_training_protocol)
            or payload.get("reference_training_protocol_sha256")
               != _sha256(args.reference_training_protocol)
            or payload.get("source_corpus_sha256")
               != reference_training["collection_corpus_sha256"]
            or payload.get("draft_model_file_sha256")
               != collection["draft_file_sha256"]
            or payload.get("steps") != reference_training["steps_per_arm"]
            or payload.get("constant_cloud_input") != "ones"
            or payload.get("folded_projection_weight_zero") is not True
            or not isinstance(reference_d, dict)
            or payload.get("reference_d_checkpoint_sha256")
               != ref_eval_protocol["checkpoint_sha256"]["target_8B"]
            or payload.get("reference_d_checkpoint_sha256")
               != reference_d["checkpoint_sha256"]
            or payload.get("exposure_sha256") != reference_d["exposure_sha256"]
            or payload.get("edge_initial_sha256")
               != reference_d["edge_initial_sha256"]):
        raise ValueError("no-h_T checkpoint is not fair/traceable to original D")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", torch_dtype=torch.bfloat16,
    ).to(device)
    draft = _FrozenDraftFeatures(model, tokenizer, device=device, dtype=torch.bfloat16)
    config = CodecConfig(**control_training["config"])
    if (config.draft_hidden_size != draft.hidden_size
            or config.vocab_size != draft.vocab_size):
        raise ValueError("no-h_T checkpoint architecture differs from frozen draft")
    codec = MatchedP1Codec(
        config, model.get_output_embeddings(), mode="target_8B",
    ).to(device)
    codec.load_trainable_state_dict(payload["state"])
    codec.eval()
    first = examples[0].observation.target_hidden.reshape(1, -1).to(device)
    second = first + torch.ones_like(first)
    assert_no_target_hidden_information(codec, first, second)
    arm = matched_p1_arm(draft, codec)
    rows = []
    eos = collection["eos_token_id"]
    for item in examples:
        o = item.observation
        prepared = arm.prepare(o.target_view())
        if prepared.message is None or len(prepared.message) != 8:
            raise RuntimeError("no-h_T control failed to emit eight bytes")
        draft_ids = _generate_probe(arm, o, prepared, eos)
        accepted = _accepted_prefix(draft_ids, item.target_future_ids)
        reference = by_arm["target_8B"][o.key]
        if (reference["target_future_ids"] != list(item.target_future_ids)
                or reference["primary_eligible"] != (
                    len(item.target_future_ids) == o.window and
                    (eos is None or eos not in item.target_future_ids))):
            raise RuntimeError("no-h_T and original D windows differ")
        rows.append({
            "arm": "target_8B_no_hidden", "prompt_id": o.prompt_id,
            "round_index": o.round_index, "task_type": o.task_type,
            "prior_accepted_count": o.accepted_count,
            "accepted_count": accepted,
            "first_rejection_position": (
                accepted + 1 if accepted < min(len(draft_ids),
                                               len(item.target_future_ids)) else None
            ),
            "draft_ids": list(draft_ids),
            "target_future_ids": list(item.target_future_ids),
            "primary_eligible": reference["primary_eligible"], "donor": None,
        })
    paired = rows + [row for arm in (
        "target_8B", "protocol_only", "shuffled_global_8B",
        "shuffled_stratified_8B",
    ) for row in by_arm[arm].values()]
    repetitions = ref_eval_protocol["bootstrap_repetitions"]
    shuffle_seed = ref_eval_protocol["bootstrap_seed"]
    comparison_pairs = (
        ("target_8B", "target_8B_no_hidden"),
        ("target_8B_no_hidden", "protocol_only"),
        ("target_8B", "protocol_only"),
        ("target_8B", "shuffled_global_8B"),
        ("target_8B", "shuffled_stratified_8B"),
    )
    comparisons = {
        f"{left}-{right}": _paired_prompt_ci(
            paired, left, right, primary_only=True,
            bootstrap_repetitions=repetitions, seed=shuffle_seed,
        )
        for left, right in comparison_pairs
    }
    report = {
        "scope": "fixed_prefix_no_target_hidden_diagnostic",
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        "latency_claim_valid": False, "split": "val", "training_seed": seed,
        "probe_windows": len(examples),
        "primary_windows": sum(row["primary_eligible"] for row in rows),
        "rows": rows,
        "primary_summary": _prompt_grouped_summary(rows, primary_only=True),
        "all_windows_sensitivity_summary": _prompt_grouped_summary(
            rows, primary_only=False),
        "primary_prompt_paired_comparisons": comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "no_hidden_report.json"
    _atomic_json(result_path, report)
    sources = [ROOT / "scripts" / name for name in
               ("no_hidden_control.py", "evaluate_no_hidden_p1.py")]
    sources += [ROOT / "feedback_codec" / name for name in
                ("codec.py", "fixed_probe.py", "hf_backend.py")]
    _atomic_json(args.output / "evaluation_protocol.json", {
        "scope": report["scope"], "latency_claim_valid": False,
        "source_corpus_sha256": _sha256(args.corpus),
        "collection_protocol_sha256": _sha256(args.collection_protocol),
        "reference_training_protocol_sha256":
            _sha256(args.reference_training_protocol),
        "reference_evaluation_report_sha256":
            _sha256(args.reference_evaluation_report),
        "control_training_protocol_sha256":
            _sha256(args.control_training_protocol),
        "no_hidden_gpu_p0_report_sha256": _sha256(args.no_hidden_gpu_p0_report),
        "control_checkpoint_sha256": _sha256(args.control_checkpoint),
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                          for path in sources},
        "bootstrap_seed": shuffle_seed,
        "bootstrap_repetitions": repetitions,
        "result_sha256": _sha256(result_path),
        "cloud_hidden_invariance": True,
    })
    print(json.dumps({"output": str(args.output),
                      "comparisons": comparisons}, ensure_ascii=False))


if __name__ == "__main__":
    main()
