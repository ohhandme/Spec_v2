"""Train P1-B F on the same frozen train windows and CE-only P1-A budget.

Only D's hard eight-byte cloud selection becomes a continuous 8192-byte
distribution.  The protected trainer, data, loss, and target oracle remain
unchanged.  This is a mechanism upper bound, not a deployable speed result.
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

from feedback_codec.codec import CodecConfig
from feedback_codec.hf_backend import _FrozenHFBackend, HFDraftBackend
from feedback_codec.training import train_matched_p1_examples
from scripts.high_bandwidth_control import HighBandwidthP1Codec, MESSAGE_BYTES
from scripts.train_fixed_probe import (
    _atomic_json, _atomic_torch, _edge_initial_hash, _exposure_hash,
    _load_locked_train, _sha256, _verify_p1_gpu_gate,
)
from scripts.train_no_hidden_p1 import _reference_arm, _validate_reference


class _FrozenDraftFeatures(_FrozenHFBackend):
    teacher_hidden = HFDraftBackend.teacher_hidden


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("draft", "corpus", "collection-protocol",
                 "reference-training-protocol", "reference-run",
                 "p1-gpu-p0-report", "f-gpu-p0-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-mib", type=int, default=22000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _verify_f_gate(path: Path, collection: dict, corpus: Path,
                   reference_protocol: Path, reference_d_sha: str) -> None:
    gate = json.loads(path.read_text(encoding="utf-8"))
    gradients = gate.get("gradient_l1")
    required = {"local_head_0", "local_head_1", "codebooks", "repair_U",
                "cloud_projection", "cloud_encoder"}
    if (gate.get("state") != "passed"
            or gate.get("scope") != "P1-B_F_continuous_real_gpu_gate"
            or gate.get("cuda") is not True or gate.get("dtype") != "bfloat16"
            or gate.get("target_file_sha256") != collection["target_file_sha256"]
            or gate.get("draft_file_sha256") != collection["draft_file_sha256"]
            or gate.get("train_corpus_sha256") != _sha256(corpus)
            or gate.get("reference_training_protocol_sha256")
               != _sha256(reference_protocol)
            or gate.get("reference_d_checkpoint_sha256") != reference_d_sha
            or gate.get("message_bytes") != MESSAGE_BYTES
            or gate.get("wire_roundtrip_exact") is not True
            or gate.get("target_dependence") is not True
            or gate.get("train_smoke_steps") != 2
            or gate.get("finite_losses") is not True
            or gate.get("target_frozen") is not True
            or gate.get("draft_frozen") is not True
            or len(gate.get("independent_target_ar_round_checks", [])) != 2
            or not all(row.get("exact_target_ar") is True for row in
                       gate["independent_target_ar_round_checks"])
            or not isinstance(gradients, dict) or set(gradients) != required
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or value <= 0 for value in gradients.values())):
        raise ValueError("F real GPU gate did not pass")
    for name in ("scripts/high_bandwidth_control.py",
                 "scripts/check_high_bandwidth_gpu_p0.py",
                 "scripts/train_high_bandwidth_p1.py",
                 "scripts/train_no_hidden_p1.py",
                 "feedback_codec/codec.py", "feedback_codec/fixed_probe.py",
                 "feedback_codec/training.py", "feedback_codec/hf_backend.py",
                 "feedback_codec/decoding.py"):
        if gate.get("source_sha256", {}).get(name) != _sha256(ROOT / name):
            raise ValueError(f"F GPU gate source changed: {name}")


def _distribution_diagnostics(codec: HighBandwidthP1Codec, examples: list,
                              limit: int) -> dict:
    selected = sorted(examples, key=lambda x: x.observation.key)[:limit]
    hist = [[0] * codec.config.choices for _ in range(codec.config.groups)]
    entropies: list[list[float]] = []
    with torch.no_grad():
        for item in selected:
            h = item.observation.target_hidden.reshape(1, -1).to(
                codec.cloud_projection.weight.device)
            probabilities = codec.cloud_distribution(h)[0]
            entropy = -(probabilities * probabilities.clamp_min(1e-30).log2()).sum(-1)
            entropies.append(entropy.cpu().tolist())
            for group, index in enumerate(probabilities.argmax(-1).tolist()):
                hist[group][int(index)] += 1
    return {
        "train_rows": len(selected),
        "mean_distribution_entropy_bits": [
            sum(row[g] for row in entropies) / len(entropies)
            for g in range(codec.config.groups)
        ],
        "argmax_histogram": hist,
        "argmax_effective_codewords": [sum(n > 0 for n in group) for group in hist],
        "message_bytes": MESSAGE_BYTES,
    }


def main() -> None:
    args = _args()
    for path in (args.draft, args.corpus, args.collection_protocol,
                 args.reference_training_protocol, args.reference_run,
                 args.p1_gpu_p0_report, args.f_gpu_p0_report):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("F training requires real CUDA")
    examples, collection = _load_locked_train(
        args.corpus, args.collection_protocol, args.draft)
    reference = json.loads(args.reference_training_protocol.read_text(encoding="utf-8"))
    _validate_reference(reference, collection, args.corpus, args.collection_protocol)
    _verify_p1_gpu_gate(args.p1_gpu_p0_report, collection)
    reference_sha = _sha256(args.reference_training_protocol)
    references = {
        seed: _reference_arm(
            args.reference_run, seed, reference_sha, reference["steps_per_arm"],
            reference["collection_corpus_sha256"],
            reference["draft_model_file_sha256"],
        ) for seed in reference["seeds"]
    }
    _verify_f_gate(args.f_gpu_p0_report, collection, args.corpus,
                   args.reference_training_protocol,
                   references[reference["seeds"][0]]["checkpoint_sha256"])
    device = torch.device(args.device)
    with torch.cuda.device(device):
        free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    if free_mib < args.minimum_free_mib:
        raise RuntimeError(f"insufficient free GPU memory: {free_mib} MiB")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False)
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
        raise ValueError("F architecture differs from P1-A D")
    source_files = [ROOT / "scripts" / name for name in
                    ("high_bandwidth_control.py", "check_high_bandwidth_gpu_p0.py",
                     "train_high_bandwidth_p1.py", "train_no_hidden_p1.py")]
    source_files += [ROOT / "feedback_codec" / name for name in
                     ("codec.py", "training.py", "losses.py", "fixed_probe.py")]
    protocol = {
        "scope": "P1-B_F_continuous_training", "claim": "mechanism upper bound only",
        "reference_training_protocol_sha256": reference_sha,
        "reference_d": references,
        "collection_protocol_sha256": _sha256(args.collection_protocol),
        "collection_corpus_sha256": _sha256(args.corpus),
        "draft_model_file_sha256": reference["draft_model_file_sha256"],
        "target_model_file_sha256_from_collection":
            reference["target_model_file_sha256_from_collection"],
        "p1_gpu_p0_report_sha256": _sha256(args.p1_gpu_p0_report),
        "f_gpu_p0_report_sha256": _sha256(args.f_gpu_p0_report),
        "config": config.__dict__, "device": str(device), "dtype": "bfloat16",
        "steps_per_arm": reference["steps_per_arm"], "seeds": reference["seeds"],
        "learning_rate": reference["learning_rate"],
        "weight_decay": reference["weight_decay"],
        "prefix_weight": reference["prefix_weight"],
        "temperature": reference["temperature"],
        "checkpoint_every": reference["checkpoint_every"],
        "diagnostic_probes": reference["diagnostic_probes"],
        "single_change": "replace hard cloud message with continuous softmax distribution; this also removes cloud Gumbel noise and uses expected codebook embeddings",
        "wire_dtype": "little_endian_float32", "wire_floats": 2048,
        "wire_bytes": MESSAGE_BYTES, "latency_claim_valid": False,
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in source_files
        },
    }
    protocol_path = args.output / "training_protocol.json"
    if args.resume:
        if (not protocol_path.is_file()
                or json.loads(protocol_path.read_text(encoding="utf-8"))
                   != json.loads(json.dumps(protocol))):
            raise ValueError("F resume protocol differs from frozen run")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        _atomic_json(protocol_path, protocol)
    protocol_sha = _sha256(protocol_path)
    results = []
    for seed in reference["seeds"]:
        folder = args.output / f"seed-{seed}" / "target_continuous_F"
        completed, checkpoint = folder / "summary.json", folder / "trainable.pt"
        if completed.exists():
            if not args.resume:
                raise FileExistsError(completed)
            row = json.loads(completed.read_text(encoding="utf-8"))
            if (row.get("seed") != seed
                    or row.get("checkpoint_sha256") != _sha256(checkpoint)
                    or row.get("exposure_sha256") != references[seed]["exposure_sha256"]
                    or row.get("edge_initial_sha256") != references[seed]["edge_initial_sha256"]):
                raise ValueError(f"completed F arm differs: {folder}")
            results.append(row)
            continue
        torch.manual_seed(seed)
        codec = HighBandwidthP1Codec(config, model.get_output_embeddings()).to(device)
        edge_initial = _edge_initial_hash(codec)
        if edge_initial != references[seed]["edge_initial_sha256"]:
            raise RuntimeError(f"F seed{seed} edge initialization differs")
        optimizer = torch.optim.AdamW(
            (p for p in codec.parameters() if p.requires_grad),
            lr=reference["learning_rate"], weight_decay=reference["weight_decay"],
        )
        folder.mkdir(parents=True, exist_ok=True)
        last = folder / "last.pt"
        resume_sampler = None
        prior_wall_s = prior_peak_mib = 0.0
        if last.exists():
            if not args.resume:
                raise FileExistsError(last)
            previous = torch.load(last, map_location="cpu", weights_only=True)
            if (previous.get("format") != "matched_p1_F_resume_v1"
                    or previous.get("seed") != seed
                    or previous.get("training_protocol_sha256") != protocol_sha
                    or previous.get("edge_initial_sha256") != edge_initial):
                raise ValueError("F resume checkpoint provenance differs")
            codec.load_trainable_state_dict(previous["codec_state"])
            optimizer.load_state_dict(previous["optimizer_state"])
            resume_sampler = previous["sampler_state"]
            prior_wall_s = float(previous["train_wall_s"])
            prior_peak_mib = float(previous["peak_vram_mib"])
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()

        def save_stage(sampler_state: dict) -> None:
            _atomic_torch(last, {
                "format": "matched_p1_F_resume_v1", "seed": seed,
                "training_protocol_sha256": protocol_sha,
                "edge_initial_sha256": edge_initial,
                "train_wall_s": prior_wall_s + time.perf_counter() - start,
                "peak_vram_mib": max(prior_peak_mib,
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
                "codec_state": codec.trainable_state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "sampler_state": sampler_state,
            })

        history = train_matched_p1_examples(
            examples=examples, target=None, draft=draft, codec=codec,
            optimizer=optimizer, steps=reference["steps_per_arm"],
            prefix_weight=reference["prefix_weight"],
            temperature=reference["temperature"], seed=seed,
            resume_state=resume_sampler,
            checkpoint_every=reference["checkpoint_every"],
            checkpoint_callback=save_stage,
        )
        wall = prior_wall_s + time.perf_counter() - start
        peak = max(prior_peak_mib,
                   torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        exposure = _exposure_hash(history)
        if exposure != references[seed]["exposure_sha256"]:
            raise RuntimeError(f"F seed{seed} exposure differs")
        diagnostics = _distribution_diagnostics(
            codec, examples, reference["diagnostic_probes"])
        _atomic_json(folder / "distribution_diagnostics.json", diagnostics)
        _atomic_torch(checkpoint, {
            "format": "matched_p1_F_continuous_v1", "mode": "target_8B",
            "seed": seed, "training_protocol_sha256": protocol_sha,
            "reference_training_protocol_sha256": reference_sha,
            "reference_d_checkpoint_sha256": references[seed]["checkpoint_sha256"],
            "source_corpus_sha256": _sha256(args.corpus),
            "draft_model_file_sha256": reference["draft_model_file_sha256"],
            "steps": reference["steps_per_arm"],
            "exposure_sha256": exposure, "edge_initial_sha256": edge_initial,
            "wire_bytes": MESSAGE_BYTES, "wire_dtype": "little_endian_float32",
            "state": codec.trainable_state_dict(),
        })
        history_path = folder / "history.jsonl"
        pending = history_path.with_suffix(".jsonl.tmp")
        with pending.open("w", encoding="utf-8", newline="\n") as out:
            for row in history:
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        pending.replace(history_path)
        row = {
            "seed": seed, "mode": "target_continuous_F",
            "steps": reference["steps_per_arm"],
            "edge_initial_sha256": edge_initial,
            "exposure_sha256": exposure,
            "checkpoint_sha256": _sha256(checkpoint),
            "history_sha256": _sha256(history_path),
            "distribution_diagnostics_sha256": _sha256(
                folder / "distribution_diagnostics.json"),
            "train_wall_s": wall, "peak_vram_mib": peak,
            "last_loss": history[-1]["loss"],
            "parameter_budget": codec.parameter_budget(),
        }
        _atomic_json(completed, row)
        results.append(row)
        del optimizer, codec
        gc.collect()
        torch.cuda.empty_cache()
    _atomic_json(args.output / "training_summary.json", results)
    print(json.dumps({"trained_arms": len(results), "output": str(args.output),
                      "claim": "F acceptance and mechanism pending frozen val"}))


if __name__ == "__main__":
    main()
