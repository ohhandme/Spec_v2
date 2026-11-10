"""Evaluate frozen B/C/D P1 checkpoints on one locked fixed-prefix corpus.

This runs real autoregressive draft calls on the same corrected prefixes.  It
never trains, downloads models, changes probe labels, or reports speed.
`test` requires an explicit final-evaluation opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, FeedbackCodec, MatchedP1Codec
from feedback_codec.fixed_probe import (
    EVAL_PROTOCOL_VERSION, ProbeExample, evaluate_fixed_probe,
    matched_p1_arm, raw_draft_arm,
)
from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_hashes(root: Path) -> dict[str, str]:
    names = {"config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "model.safetensors.index.json"}
    files = sorted(path for path in root.rglob("*") if path.is_file() and (
        path.name in names or path.suffix in (".safetensors", ".bin")
    ))
    if not any(path.suffix in (".safetensors", ".bin") for path in files):
        raise ValueError(f"model has no local weights: {root}")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in files}


def _validate_collection_sources(protocol: dict) -> None:
    """Reject inference or metric changes after fixed-prefix collection."""
    immutable_sources = (
        "feedback_codec/codec.py",
        "feedback_codec/fixed_probe.py",
        "feedback_codec/decoding.py",
        "feedback_codec/hf_backend.py",
    )
    recorded = protocol.get("source_sha256")
    if not isinstance(recorded, dict) or any(
        recorded.get(name) != _sha256(ROOT / name)
        for name in immutable_sources
    ):
        raise ValueError("locked fixed-probe codec/metric/backend source changed after collection")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="fixed_prefix.jsonl from build_fixed_probe.py")
    parser.add_argument("--collection-protocol", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--edge-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "smoke"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _args()
    for path in (args.target, args.draft, args.corpus, args.collection_protocol,
                 args.training_protocol,
                 args.edge_checkpoint, args.protocol_checkpoint, args.target_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.split == "test" and not args.allow_test:
        raise ValueError("test remains sealed unless --allow-test is specified")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("real-model fixed probe requires CUDA")
    protocol = json.loads(args.collection_protocol.read_text(encoding="utf-8"))
    if protocol.get("eval_protocol_version") != EVAL_PROTOCOL_VERSION:
        raise ValueError("collection and evaluation protocol versions differ")
    _validate_collection_sources(protocol)
    if protocol.get("fixed_prefix_sha256") != _sha256(args.corpus):
        raise ValueError("fixed-prefix corpus changed after collection")
    if protocol.get("split") != args.split or protocol.get("window") != 8:
        raise ValueError("split/window differs from collection protocol")
    if protocol.get("target_file_sha256") != _model_hashes(args.target):
        raise ValueError("target model files differ from collection")
    if protocol.get("draft_file_sha256") != _model_hashes(args.draft):
        raise ValueError("draft model files differ from collection")
    training_protocol = json.loads(args.training_protocol.read_text(encoding="utf-8"))
    if training_protocol.get("scope") != "P1_locked_fixed_prefix_training" or (
        training_protocol.get("draft_model_file_sha256") != protocol["draft_file_sha256"]
        or training_protocol.get("target_model_file_sha256_from_collection")
        != protocol["target_file_sha256"]
    ):
        raise ValueError("training and evaluation used different frozen models")
    if args.split in ("val", "test") and (
        training_protocol.get("collection_corpus_sha256") == _sha256(args.corpus)
    ):
        raise ValueError("evaluation corpus was also used for training")
    examples = [ProbeExample.from_json(json.loads(line)) for line in
                args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples:
        raise ValueError("fixed-prefix corpus is empty")
    device = torch.device(args.device)
    target = HFTargetBackend(args.target, None, device=device, dtype=torch.bfloat16)
    draft_tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", torch_dtype=torch.bfloat16,
    ).to(device)
    config = CodecConfig(
        target_hidden_size=target.hidden_size,
        draft_hidden_size=int(draft_model.config.hidden_size),
        vocab_size=target.vocab_size, window=8,
    )
    # The raw arm uses no feedback.  Its codec object is only the existing HF
    # backend's required constructor argument; no codec weights are applied.
    raw_codec = FeedbackCodec(config, draft_model.get_output_embeddings()).to(device)
    draft = HFDraftBackend(draft_model, draft_tokenizer, raw_codec,
                           target=target, device=device, dtype=torch.bfloat16)
    frozen_head = draft_model.get_output_embeddings().weight.detach().cpu().clone()
    ckpts = {
        "edge_only": args.edge_checkpoint,
        "protocol_only": args.protocol_checkpoint,
        "target_8B": args.target_checkpoint,
    }
    codecs = {}
    checkpoint_seeds = set()
    common_training_fields = (
        "training_protocol_sha256", "source_corpus_sha256",
        "draft_model_file_sha256", "steps", "exposure_sha256",
        "edge_initial_sha256",
    )
    first_training_fields = None
    for mode, path in ckpts.items():
        codec = MatchedP1Codec(config, draft_model.get_output_embeddings(), mode=mode).to(device)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != "matched_p1_trainable_v1" or (
            payload.get("mode") != mode
        ) or (
            type(payload.get("seed")) is not int or not isinstance(payload.get("state"), dict)
        ):
            raise ValueError(f"{mode} checkpoint lacks matching mode/seed/state provenance")
        checked_training_fields = tuple(payload.get(field) for field in common_training_fields)
        if first_training_fields is None:
            first_training_fields = checked_training_fields
        elif checked_training_fields != first_training_fields:
            raise ValueError("B/C/D checkpoints differ in training corpus/budget/exposure")
        if payload["training_protocol_sha256"] != _sha256(args.training_protocol) or (
            payload["source_corpus_sha256"] != training_protocol["collection_corpus_sha256"]
            or payload["draft_model_file_sha256"] != protocol["draft_file_sha256"]
            or payload["steps"] != training_protocol["steps_per_arm"]
        ):
            raise ValueError("checkpoint does not match locked training protocol")
        checkpoint_seeds.add(payload["seed"])
        codec.load_trainable_state_dict(
            payload["state"]
        )
        if not torch.equal(draft_model.get_output_embeddings().weight.detach().cpu(), frozen_head):
            raise RuntimeError("checkpoint changed the supposedly frozen draft output head")
        codec.eval()
        codecs[mode] = codec
    if len(checkpoint_seeds) != 1:
        raise ValueError("B/C/D checkpoints must use the same training seed")
    if next(iter(checkpoint_seeds)) not in training_protocol["seeds"]:
        raise ValueError("checkpoint seed is absent from training protocol")
    budgets = {mode: codec.parameter_budget() for mode, codec in codecs.items()}
    edge_counts = [budget["edge_trainable"] for budget in budgets.values()]
    if max(edge_counts) / min(edge_counts) > 1.05:
        raise RuntimeError("edge parameter budgets differ by more than five percent")
    arms = {"raw_draft": raw_draft_arm(draft)}
    arms.update({mode: matched_p1_arm(draft, codec) for mode, codec in codecs.items()})
    report = evaluate_fixed_probe(
        examples=examples, arms=arms, eos_token_id=protocol["eos_token_id"],
        split=args.split, allow_test=args.allow_test, shuffle_seed=args.seed,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    result_file = args.output / "fixed_probe_report.json"
    result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    metadata = {
        "scope": "fixed_prefix_probe",
        "latency_claim_valid": False,
        "source_collection_protocol_sha256": _sha256(args.collection_protocol),
        "source_training_protocol_sha256": _sha256(args.training_protocol),
        "source_corpus_sha256": _sha256(args.corpus),
        "checkpoint_sha256": {mode: _sha256(path) for mode, path in ckpts.items()},
        "training_seed": next(iter(checkpoint_seeds)),
        "parameter_budgets": budgets,
        "result_sha256": _sha256(result_file),
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in sorted((ROOT / "feedback_codec").glob("*.py")) + [Path(__file__)]
        },
        "bootstrap_seed": args.seed,
        "bootstrap_repetitions": args.bootstrap_repetitions,
    }
    (args.output / "evaluation_protocol.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output),
                      "primary_prompt_paired_comparisons": report["primary_prompt_paired_comparisons"]},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
