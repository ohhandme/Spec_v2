"""Train the P1-B strict no-h_T D control on the frozen train probes.

Only this D control is trained. It keeps the original D architecture, optimizer,
seed, exposure and eight-byte route. The Target-hidden values are replaced in
memory with a constant before training; the final checkpoint is folded so its
message is provably independent of h_T at inference. No val/test rows are read.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, MatchedP1Codec
from feedback_codec.hf_backend import _FrozenHFBackend, HFDraftBackend
from feedback_codec.training import train_matched_p1_examples
from scripts.no_hidden_control import (
    assert_no_target_hidden_information, constant_hidden_examples,
    fold_constant_hidden,
)
from scripts.train_fixed_probe import (
    _atomic_json, _atomic_torch, _code_diagnostics, _edge_initial_hash,
    _exposure_hash, _load_locked_train, _sha256, _verify_p1_gpu_gate,
)


class _FrozenDraftFeatures(_FrozenHFBackend):
    teacher_hidden = HFDraftBackend.teacher_hidden


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("draft", "corpus", "collection-protocol",
                 "reference-training-protocol", "reference-run",
                 "p1-gpu-p0-report", "no-hidden-gpu-p0-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-mib", type=int, default=22000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _reference_arm(root: Path, seed: int, protocol_sha: str,
                   steps: int, corpus_sha: str, draft_hashes: dict) -> dict:
    folder = root / f"seed-{seed}" / "target_8B"
    summary_path, checkpoint_path = folder / "summary.json", folder / "trainable.pt"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("seed") != seed or summary.get("mode") != "target_8B"
            or summary.get("checkpoint_sha256") != _sha256(checkpoint_path)):
        raise ValueError(f"reference D summary/checkpoint differs: {folder}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format") != "matched_p1_trainable_v1"
            or checkpoint.get("mode") != "target_8B"
            or checkpoint.get("seed") != seed
            or checkpoint.get("training_protocol_sha256") != protocol_sha
            or checkpoint.get("steps") != steps
            or checkpoint.get("source_corpus_sha256") != corpus_sha
            or checkpoint.get("draft_model_file_sha256") != draft_hashes
            or checkpoint.get("exposure_sha256") != summary.get("exposure_sha256")
            or checkpoint.get("edge_initial_sha256") != summary.get("edge_initial_sha256")):
        raise ValueError(f"reference D checkpoint provenance differs: {folder}")
    return {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "edge_initial_sha256": summary["edge_initial_sha256"],
        "exposure_sha256": summary["exposure_sha256"],
    }


def _validate_reference(reference: dict, collection: dict, corpus: Path,
                        collection_protocol: Path) -> None:
    if (reference.get("scope") != "P1_locked_fixed_prefix_training"
            or reference.get("phase") != "P1-A_CE_only"
            or reference.get("smoke") is not False
            or reference.get("steps_per_arm") != 9366
            or reference.get("seeds") != [42, 43, 44]
            or reference.get("prefix_weight") != 0.0
            or reference.get("collection_protocol_sha256") != _sha256(collection_protocol)
            or reference.get("collection_corpus_sha256") != _sha256(corpus)
            or reference.get("draft_model_file_sha256") != collection["draft_file_sha256"]
            or reference.get("target_model_file_sha256_from_collection")
               != collection["target_file_sha256"]):
        raise ValueError("P1-B reference must be the frozen 9366-step CE-only P1-A run")
    for name in ("feedback_codec/codec.py", "feedback_codec/training.py",
                 "feedback_codec/losses.py", "feedback_codec/fixed_probe.py"):
        if reference.get("source_sha256", {}).get(name) != _sha256(ROOT / name):
            raise ValueError(f"frozen source changed since P1-A: {name}")
    for name in ("learning_rate", "temperature"):
        if not math.isfinite(reference[name]) or reference[name] <= 0:
            raise ValueError(f"invalid reference {name}")


def _verify_no_hidden_gpu_gate(path: Path, collection: dict,
                               corpus: Path, reference_protocol: Path,
                               reference_checkpoint_sha: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    gradients = report.get("gradient_l1")
    required_gradients = {
        "local_head_0", "local_head_1", "codebooks", "repair_U",
        "cloud_projection", "cloud_encoder",
    }
    if (report.get("state") != "passed"
            or report.get("scope") != "P1-B_no_target_hidden_real_gpu_gate"
            or report.get("cuda") is not True
            or report.get("dtype") != "bfloat16"
            or report.get("target_file_sha256") != collection["target_file_sha256"]
            or report.get("draft_file_sha256") != collection["draft_file_sha256"]
            or report.get("train_corpus_sha256") != _sha256(corpus)
            or report.get("reference_training_protocol_sha256")
               != _sha256(reference_protocol)
            or report.get("reference_d_checkpoint_sha256")
               != reference_checkpoint_sha
            or report.get("train_smoke_steps") != 2
            or report.get("finite_losses") is not True
            or report.get("cloud_hidden_invariance") is not True
            or report.get("eight_byte_message") is not True
            or report.get("target_frozen") is not True
            or report.get("draft_frozen") is not True
            or len(report.get("independent_target_ar_round_checks", [])) != 2
            or not all(row.get("exact_target_ar") is True for row in
                       report["independent_target_ar_round_checks"])
            or not isinstance(gradients, dict)
            or set(gradients) != required_gradients
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or value <= 0 for value in gradients.values())):
        raise ValueError("P1-B no-hidden real GPU gate did not pass")
    source = report.get("source_sha256")
    if not isinstance(source, dict) or any(
        source.get(name) != _sha256(ROOT / name)
        for name in ("scripts/no_hidden_control.py",
                     "scripts/check_no_hidden_gpu_p0.py",
                     "scripts/train_no_hidden_p1.py",
                     "feedback_codec/codec.py", "feedback_codec/fixed_probe.py",
                     "feedback_codec/training.py", "feedback_codec/hf_backend.py",
                     "feedback_codec/decoding.py")
    ):
        raise ValueError("P1-B no-hidden GPU gate source changed")
    return report


def main() -> None:
    args = _args()
    for path in (args.draft, args.corpus, args.collection_protocol,
                 args.reference_training_protocol, args.reference_run,
                 args.p1_gpu_p0_report, args.no_hidden_gpu_p0_report):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("P1-B real-model training requires CUDA")
    examples, collection = _load_locked_train(
        args.corpus, args.collection_protocol, args.draft,
    )
    reference = json.loads(args.reference_training_protocol.read_text(encoding="utf-8"))
    _validate_reference(reference, collection, args.corpus, args.collection_protocol)
    _verify_p1_gpu_gate(args.p1_gpu_p0_report, collection)
    reference_sha = _sha256(args.reference_training_protocol)
    references = {
        seed: _reference_arm(args.reference_run, seed, reference_sha,
                             reference["steps_per_arm"],
                             reference["collection_corpus_sha256"],
                             reference["draft_model_file_sha256"])
        for seed in reference["seeds"]
    }
    _verify_no_hidden_gpu_gate(
        args.no_hidden_gpu_p0_report, collection, args.corpus,
        args.reference_training_protocol, references[reference["seeds"][0]][
            "checkpoint_sha256"],
    )
    device = torch.device(args.device)
    with torch.cuda.device(device):
        free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    if free_mib < args.minimum_free_mib:
        raise RuntimeError(f"insufficient free GPU memory: {free_mib} MiB")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", torch_dtype=torch.bfloat16,
    ).to(device)
    draft = _FrozenDraftFeatures(model, tokenizer, device=device, dtype=torch.bfloat16)
    config = CodecConfig(
        target_hidden_size=examples[0].observation.target_hidden.numel(),
        draft_hidden_size=draft.hidden_size, vocab_size=draft.vocab_size, window=8,
    )
    if config.__dict__ != reference["config"]:
        raise ValueError("P1-B architecture differs from P1-A")
    ablated = constant_hidden_examples(examples)
    source_files = [ROOT / "scripts" / name for name in
                    ("no_hidden_control.py", "train_no_hidden_p1.py")]
    source_files += [ROOT / "feedback_codec" / name for name in
                     ("codec.py", "training.py", "losses.py", "fixed_probe.py")]
    protocol = {
        "scope": "P1-B_no_target_hidden_training",
        "claim": "strict constant-input ablation; no acceptance result until val",
        "reference_training_protocol_sha256": reference_sha,
        "reference_d": references,
        "collection_protocol_sha256": _sha256(args.collection_protocol),
        "collection_corpus_sha256": _sha256(args.corpus),
        "draft_model_file_sha256": reference["draft_model_file_sha256"],
        "target_model_file_sha256_from_collection":
            reference["target_model_file_sha256_from_collection"],
        "p1_gpu_p0_report_sha256": _sha256(args.p1_gpu_p0_report),
        "no_hidden_gpu_p0_report_sha256": _sha256(args.no_hidden_gpu_p0_report),
        "config": config.__dict__, "device": str(device), "dtype": "bfloat16",
        "steps_per_arm": reference["steps_per_arm"],
        "seeds": reference["seeds"],
        "learning_rate": reference["learning_rate"],
        "weight_decay": reference["weight_decay"],
        "prefix_weight": reference["prefix_weight"],
        "temperature": reference["temperature"],
        "checkpoint_every": reference["checkpoint_every"],
        "diagnostic_probes": reference["diagnostic_probes"],
        "ablation": "replace only cached h_T in memory with ones; fold W@ones+b into b and zero W after training",
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                          for path in source_files},
    }
    protocol_path = args.output / "training_protocol.json"
    if args.resume:
        if not protocol_path.is_file() or json.loads(protocol_path.read_text(
                encoding="utf-8")) != json.loads(json.dumps(protocol)):
            raise ValueError("P1-B resume protocol differs from locked run")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        _atomic_json(protocol_path, protocol)
    protocol_sha = _sha256(protocol_path)
    results = []
    for seed in reference["seeds"]:
        folder = args.output / f"seed-{seed}" / "target_8B_no_hidden"
        completed = folder / "summary.json"
        checkpoint = folder / "trainable.pt"
        if completed.exists():
            if not args.resume:
                raise FileExistsError(completed)
            row = json.loads(completed.read_text(encoding="utf-8"))
            if (row.get("checkpoint_sha256") != _sha256(checkpoint)
                    or row.get("seed") != seed
                    or row.get("exposure_sha256") != references[seed]["exposure_sha256"]
                    or row.get("edge_initial_sha256") != references[seed]["edge_initial_sha256"]):
                raise ValueError(f"completed P1-B arm differs: {folder}")
            results.append(row)
            continue
        torch.manual_seed(seed)
        codec = MatchedP1Codec(
            config, model.get_output_embeddings(), mode="target_8B",
        ).to(device)
        edge_initial = _edge_initial_hash(codec)
        if edge_initial != references[seed]["edge_initial_sha256"]:
            raise RuntimeError(f"P1-B seed{seed} edge initialization differs")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in codec.parameters() if parameter.requires_grad),
            lr=reference["learning_rate"], weight_decay=reference["weight_decay"],
        )
        folder.mkdir(parents=True, exist_ok=True)
        last = folder / "last.pt"
        resume_sampler = None
        prior_wall_s = 0.0
        prior_peak_mib = 0.0
        if last.exists():
            if not args.resume:
                raise FileExistsError(last)
            previous = torch.load(last, map_location="cpu", weights_only=True)
            if (previous.get("format") != "matched_p1_no_hidden_resume_v1"
                    or previous.get("seed") != seed
                    or previous.get("training_protocol_sha256") != protocol_sha
                    or previous.get("edge_initial_sha256") != edge_initial):
                raise ValueError("P1-B resume checkpoint provenance differs")
            codec.load_trainable_state_dict(previous["codec_state"])
            optimizer.load_state_dict(previous["optimizer_state"])
            resume_sampler = previous["sampler_state"]
            prior_wall_s = float(previous["train_wall_s"])
            prior_peak_mib = float(previous["peak_vram_mib"])
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()

        def save_stage(sampler_state: dict) -> None:
            _atomic_torch(last, {
                "format": "matched_p1_no_hidden_resume_v1",
                "seed": seed, "training_protocol_sha256": protocol_sha,
                "edge_initial_sha256": edge_initial,
                "train_wall_s": prior_wall_s + time.perf_counter() - start,
                "peak_vram_mib": max(prior_peak_mib,
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
                "codec_state": codec.trainable_state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "sampler_state": sampler_state,
            })

        history = train_matched_p1_examples(
            examples=ablated, target=None, draft=draft, codec=codec,
            optimizer=optimizer, steps=reference["steps_per_arm"],
            prefix_weight=reference["prefix_weight"],
            temperature=reference["temperature"], seed=seed,
            resume_state=resume_sampler,
            checkpoint_every=reference["checkpoint_every"],
            checkpoint_callback=save_stage,
        )
        train_wall_s = prior_wall_s + time.perf_counter() - start
        peak_mib = max(prior_peak_mib,
                       torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        exposure = _exposure_hash(history)
        if exposure != references[seed]["exposure_sha256"]:
            raise RuntimeError(f"P1-B seed{seed} exposure differs")
        constant = torch.ones((1, config.target_hidden_size), device=device,
                              dtype=examples[0].observation.target_hidden.dtype)
        with torch.no_grad():
            before = codec.cloud_scores(constant).detach().clone()
            fold_constant_hidden(codec)
            after = codec.cloud_scores(constant)
            if not torch.allclose(before, after, atol=1e-5, rtol=1e-5):
                raise RuntimeError("folding changed the learned constant message")
            first = examples[0].observation.target_hidden.reshape(1, -1).to(device)
            second = first + torch.ones_like(first)
            assert_no_target_hidden_information(codec, first, second)
        diagnostics = _code_diagnostics(
            codec=codec, draft=draft, examples=examples, history=history,
            seed=seed, limit=reference["diagnostic_probes"],
        )
        _atomic_json(folder / "code_diagnostics.json", diagnostics)
        _atomic_torch(checkpoint, {
            "format": "matched_p1_no_hidden_v1",
            "mode": "target_8B", "seed": seed,
            "training_protocol_sha256": protocol_sha,
            "reference_training_protocol_sha256": reference_sha,
            "reference_d_checkpoint_sha256": references[seed]["checkpoint_sha256"],
            "source_corpus_sha256": _sha256(args.corpus),
            "draft_model_file_sha256": reference["draft_model_file_sha256"],
            "steps": reference["steps_per_arm"],
            "exposure_sha256": exposure,
            "edge_initial_sha256": edge_initial,
            "constant_cloud_input": "ones",
            "folded_projection_weight_zero": True,
            "state": codec.trainable_state_dict(),
        })
        history_path = folder / "history.jsonl"
        pending = history_path.with_suffix(".jsonl.tmp")
        with pending.open("w", encoding="utf-8", newline="\n") as out:
            for row in history:
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        pending.replace(history_path)
        row = {
            "seed": seed, "mode": "target_8B_no_hidden",
            "steps": reference["steps_per_arm"],
            "edge_initial_sha256": edge_initial,
            "exposure_sha256": exposure,
            "checkpoint_sha256": _sha256(checkpoint),
            "history_sha256": _sha256(history_path),
            "code_diagnostics_sha256": _sha256(folder / "code_diagnostics.json"),
            "train_wall_s": train_wall_s, "peak_vram_mib": peak_mib,
            "last_loss": history[-1]["loss"],
            "parameter_budget": codec.parameter_budget(),
            "cloud_hidden_invariance": True,
        }
        _atomic_json(completed, row)
        results.append(row)
        del optimizer, codec
        gc.collect()
        torch.cuda.empty_cache()
    _atomic_json(args.output / "training_summary.json", results)
    print(json.dumps({"trained_arms": len(results), "output": str(args.output),
                      "claim": "no acceptance result until fixed-prefix evaluation"}))


if __name__ == "__main__":
    main()
