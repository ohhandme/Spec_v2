"""Train one matched P1 upper-bound arm from a locked packed-verifier corpus.

Run ``--initialize-only`` once to freeze a common protocol, then run one
``--mode``/``--seed`` worker at a time (or on independently admitted GPUs).
This command never reads val/test, loads the Target, or collects new labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig
from feedback_codec.hf_backend import HFDraftBackend, _FrozenHFBackend
from feedback_codec.v2_upper_bound import (
    MatchedUpperBound, UPPER_MODES, assert_matched_capacity,
)
from feedback_codec.v2_p1_packed import (
    exposure_order, load_packed_p1_corpus, sha256_file, canonical_sha256,
    train_one_window,
)


class FrozenDraftFeatures(_FrozenHFBackend):
    teacher_hidden = HFDraftBackend.teacher_hidden


def model_hashes(root: Path, expected: dict[str, str]) -> dict[str, str]:
    """Hash precisely P0's asset inventory and reject extra weight shards."""
    if not root.is_dir() or not expected or not any(
            name.endswith(".safetensors") for name in expected):
        raise ValueError("P0 Draft asset inventory is absent")
    observed = {}
    for name, digest in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"P0 Draft asset is absent or leaves model root: {name}")
        observed[name] = sha256_file(path)
        if observed[name] != digest:
            raise ValueError(f"P0 Draft asset changed: {name}")
    weights = {path.relative_to(root).as_posix() for path in root.rglob("*.safetensors")}
    if weights != {name for name in expected if name.endswith(".safetensors")}:
        raise ValueError("Draft weight shard inventory differs from P0")
    return observed


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def trainable_state(model: MatchedUpperBound) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()
            if param.requires_grad}


def load_trainable_state(model: MatchedUpperBound,
                         saved: dict[str, torch.Tensor]) -> None:
    current = {name: param for name, param in model.named_parameters()
               if param.requires_grad}
    if set(current) != set(saved):
        raise ValueError("checkpoint trainable parameter names changed")
    with torch.no_grad():
        for name, param in current.items():
            state = saved[name]
            if state.shape != param.shape or state.dtype != param.dtype or (
                    not torch.isfinite(state).all()):
                raise ValueError(f"invalid checkpoint parameter {name}")
            param.copy_(state.to(param.device))


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    names = ("feedback_codec/codec.py", "feedback_codec/fixed_probe.py",
             "feedback_codec/decoding.py", "feedback_codec/hf_backend.py",
             "feedback_codec/v2_upper_bound.py", "feedback_codec/v2_p1_packed.py",
             "scripts/train_packed_upper_p1.py")
    return {name: sha256_file(ROOT / name) for name in names}


def make_protocol(args: argparse.Namespace, corpus: object,
                  asset_hashes: dict[str, str]) -> dict:
    if args.steps <= 0 or args.steps > len(corpus.examples) or (
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or not math.isfinite(args.temperature) or args.temperature <= 0):
        raise ValueError("P1 budget or optimizer settings are invalid")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(
            type(seed) is not int or seed < 0 for seed in args.seeds):
        raise ValueError("P1 seeds must be distinct nonnegative integers")
    draft_config = json.loads((args.draft / "config.json").read_text(encoding="utf-8"))
    if (type(draft_config.get("hidden_size")) is not int or
            type(draft_config.get("vocab_size")) is not int):
        raise ValueError("Draft config lacks pinned hidden/vocabulary sizes")
    codec_config = CodecConfig(
        target_hidden_size=corpus.examples[0].observation.target_hidden.numel(),
        draft_hidden_size=draft_config["hidden_size"],
        vocab_size=draft_config["vocab_size"], window=8)
    budget_lock = json.loads(args.budget_lock.read_text(encoding="utf-8"))
    expected_budget = {
        "schema": "P1_packed_budget_lock_v1",
        "frozen_before_validation": True,
        "train_corpus_sha256": corpus.corpus_sha256,
        "train_collection_protocol_sha256": corpus.protocol_sha256,
        "steps_per_arm": args.steps,
        "training_seeds": args.seeds,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "checkpoint_every": args.checkpoint_every,
    }
    if any(budget_lock.get(key) != value for key, value in expected_budget.items()):
        raise ValueError("pre-validation P1 budget lock differs from requested run")
    cost_lock = json.loads(args.cost_lock.read_text(encoding="utf-8"))
    cost_freeze = json.loads(args.cost_freeze_manifest.read_text(encoding="utf-8"))
    if (cost_freeze.get("schema") != "feedback64_v2_p0_cost_proxy_freeze_v1"
            or cost_freeze.get("cost_lock_sha256") != sha256_file(args.cost_lock)
            or cost_freeze.get("packed_p0_report_sha256") !=
               corpus.protocol["packed_p0_report_sha256"]
            or cost_freeze.get("pilot_val_acceptance_results_read_before_freeze") is not False
            or cost_lock.get("packed_p0_report_sha256") !=
               corpus.protocol["packed_p0_report_sha256"]):
        raise ValueError("P0 cost proxy is not byte-frozen before validation")
    return {
        "scope": "P1_packed_matched_upper_training",
        "claim": "CE_only_fixed_prefix_diagnostic_no_speed_or_WAN_claim",
        "collection_protocol_sha256": corpus.protocol_sha256,
        "collection_corpus_sha256": corpus.corpus_sha256,
        "target_file_sha256_from_collection": corpus.protocol["target_file_sha256"],
        "draft_file_sha256": asset_hashes,
        "packed_p0_report_sha256": corpus.protocol["packed_p0_report_sha256"],
        "pre_val_budget_lock_sha256": sha256_file(args.budget_lock),
        "p0_cost_lock_sha256": sha256_file(args.cost_lock),
        "p0_cost_freeze_manifest_sha256": sha256_file(args.cost_freeze_manifest),
        "p0_cost_is_proxy_not_deployment_evidence": True,
        "window": 8,
        "codec_config": asdict(codec_config),
        "steps_per_arm": args.steps,
        "training_seeds": args.seeds,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "checkpoint_every": args.checkpoint_every,
        "source_sha256": source_hashes(),
        "exposure_order_sha256": {str(seed): canonical_sha256(exposure_order(
            len(corpus.examples), seed=seed, steps=args.steps)) for seed in args.seeds},
        "same_initialization_required": True,
        "target_and_draft_frozen": True,
        "no_replacement_one_epoch": True,
        "future_labels_excluded_from_prepare": True,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--collection-protocol", type=Path, required=True)
    parser.add_argument("--budget-lock", type=Path, required=True)
    parser.add_argument("--cost-lock", type=Path, required=True)
    parser.add_argument("--cost-freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seeds", type=lambda value: [int(x) for x in value.split(",")],
                        required=True)
    parser.add_argument("--mode", choices=UPPER_MODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--minimum-free-mib", type=int, default=22000)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint interval must be positive")
    corpus = load_packed_p1_corpus(args.corpus, args.collection_protocol, split="train")
    actual_draft_hashes = model_hashes(args.draft, corpus.protocol["draft_file_sha256"])
    if actual_draft_hashes != corpus.protocol["draft_file_sha256"]:
        raise ValueError("local Draft asset SHA differs from packed collection")
    run_protocol = make_protocol(args, corpus, actual_draft_hashes)
    protocol_file = args.output / "training_protocol.json"
    if args.initialize_only:
        if args.resume or args.mode is not None or args.seed is not None or args.output.exists():
            raise ValueError("initialize-only requires a new output and no selected worker")
        args.output.mkdir(parents=True)
        atomic_json(protocol_file, run_protocol)
        print(json.dumps({"training_protocol": str(protocol_file),
                          "sha256": sha256_file(protocol_file),
                          "windows": len(corpus.examples)}, ensure_ascii=False))
        return
    if args.mode is None or args.seed not in args.seeds or not protocol_file.exists():
        raise ValueError("mode/seed must belong to an existing initialized protocol")
    if json.loads(protocol_file.read_text(encoding="utf-8")) != run_protocol:
        raise ValueError("training protocol, source, inputs or budget changed")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal packed P1 requires a real CUDA Draft")
    device = torch.device(args.device)
    index = 0 if device.index is None else device.index
    free_mib = torch.cuda.mem_get_info(index)[0] // (1024 * 1024)
    if free_mib < args.minimum_free_mib:
        raise RuntimeError(f"insufficient free GPU memory: {free_mib} MiB")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", torch_dtype=torch.bfloat16).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False)
    draft = FrozenDraftFeatures(model, tokenizer, device=device, dtype=torch.bfloat16)
    config = CodecConfig(
        target_hidden_size=corpus.examples[0].observation.target_hidden.numel(),
        draft_hidden_size=draft.hidden_size, vocab_size=draft.vocab_size, window=8)
    if asdict(config) != run_protocol["codec_config"]:
        raise ValueError("loaded Draft architecture differs from locked P1 geometry")
    arms: dict[str, MatchedUpperBound] = {}
    for mode in UPPER_MODES:
        torch.manual_seed(args.seed)
        arms[mode] = MatchedUpperBound(config, model.get_output_embeddings(), mode=mode)
    budget = assert_matched_capacity(arms)
    initial_sha = state_sha256(trainable_state(arms[args.mode]))
    selected = arms[args.mode]
    del arms
    selected.to(device)
    optimizer = torch.optim.AdamW((p for p in selected.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    order = exposure_order(len(corpus.examples), seed=args.seed, steps=args.steps)
    worker_dir = args.output / f"seed-{args.seed}" / args.mode
    checkpoint = worker_dir / "last.pt"
    if args.resume and (worker_dir / "summary.json").exists():
        raise FileExistsError("this P1 arm already completed; preserve its final evidence")
    step = 0
    history: list[dict] = []
    if args.resume:
        if not checkpoint.is_file():
            raise FileNotFoundError("resume requires an atomic matching last.pt")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (payload.get("format") != "packed_upper_p1_resume_v1"
                or payload.get("training_protocol_sha256") != sha256_file(protocol_file)
                or payload.get("seed") != args.seed or payload.get("mode") != args.mode
                or payload.get("initial_sha256") != initial_sha
                or payload.get("exposure_order_sha256") !=
                   run_protocol["exposure_order_sha256"][str(args.seed)]):
            raise ValueError("resume checkpoint provenance mismatch")
        step = payload["step"]
        history = payload["history"]
        if (type(step) is not int or not 0 <= step <= args.steps
                or len(history) != step
                or any(row["example_index"] != order[i] or row["step"] != i + 1
                       for i, row in enumerate(history))):
            raise ValueError("resume exposure history is incomplete")
        load_trainable_state(selected, payload["state"])
        optimizer.load_state_dict(payload["optimizer"])
    elif worker_dir.exists():
        raise FileExistsError("worker output exists; resume an exact checkpoint")
    else:
        worker_dir.mkdir(parents=True)
    for i in range(step, args.steps):
        example = corpus.examples[order[i]]
        loss = train_one_window(selected, draft, example, optimizer,
                                temperature=args.temperature, random_seed=args.seed + i + 1)
        history.append({"step": i + 1, "example_index": order[i],
                        "prompt_id": example.observation.prompt_id,
                        "round_index": example.observation.round_index,
                        "loss": loss})
        if (i + 1) % args.checkpoint_every == 0 or i + 1 == args.steps:
            atomic_torch(checkpoint, {
                "format": "packed_upper_p1_resume_v1", "mode": args.mode,
                "seed": args.seed, "step": i + 1,
                "training_protocol_sha256": sha256_file(protocol_file),
                "exposure_order_sha256": run_protocol["exposure_order_sha256"][str(args.seed)],
                "initial_sha256": initial_sha,
                "state": trainable_state(selected),
                "optimizer": optimizer.state_dict(), "history": history,
            })
            atomic_json(worker_dir / "progress.json", {
                "mode": args.mode, "seed": args.seed, "step": i + 1,
                "steps": args.steps, "finite_losses": all(math.isfinite(
                    row["loss"]) for row in history),
                "last_loss": loss,
                "exposure_order_sha256": run_protocol["exposure_order_sha256"][str(args.seed)],
            })
    atomic_torch(worker_dir / "trainable.pt", {
        "format": "packed_upper_p1_trainable_v1", "mode": args.mode,
        "seed": args.seed, "steps": args.steps,
        "training_protocol_sha256": sha256_file(protocol_file),
        "source_corpus_sha256": corpus.corpus_sha256,
        "draft_model_file_sha256": actual_draft_hashes,
        "exposure_order_sha256": run_protocol["exposure_order_sha256"][str(args.seed)],
        "initial_sha256": initial_sha, "capacity": budget,
        "state": trainable_state(selected),
    })
    atomic_json(worker_dir / "summary.json", {
        "mode": args.mode, "seed": args.seed, "steps": args.steps,
        "initial_sha256": initial_sha,
        "trained_sha256": sha256_file(worker_dir / "trainable.pt"),
        "exposure_order_sha256": run_protocol["exposure_order_sha256"][str(args.seed)],
        "mean_ce_loss": sum(row["loss"] for row in history) / len(history),
        "parameter_budget": budget, "claim": "training_loss_only_no_acceptance_result",
    })
    print(json.dumps({"mode": args.mode, "seed": args.seed,
                      "steps": args.steps, "output": str(worker_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
