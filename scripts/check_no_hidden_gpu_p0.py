"""Real BF16 P1-B gate for constant-h_T D before full no-hidden training.

This uses only frozen train probes. It tests the new in-memory masking/folding
path while preserving the original codec, Target oracle and acceptance rule.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, FeedbackCodec, MatchedP1Codec
from feedback_codec.fixed_probe import matched_p1_arm
from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
from feedback_codec.training import train_matched_p1_examples
from scripts.build_fixed_probe import _model_hashes
from scripts.no_hidden_control import (
    assert_no_target_hidden_information, constant_hidden_examples,
    fold_constant_hidden,
)
from scripts.run_fixed_probe_gpu_p0 import _check_one_round
from scripts.train_fixed_probe import (
    _atomic_json, _edge_initial_hash, _load_locked_train, _sha256,
    _verify_p1_gpu_gate,
)
from scripts.train_no_hidden_p1 import _reference_arm, _validate_reference


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "draft", "corpus", "collection-protocol",
                 "reference-training-protocol", "reference-run",
                 "p1-gpu-p0-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-mib", type=int, default=22000)
    return parser.parse_args()


def _gradient_evidence(codec: MatchedP1Codec) -> dict[str, float]:
    groups = {name: 0.0 for name in
              ("local_head_0", "local_head_1", "codebooks", "repair_U",
               "cloud_projection", "cloud_encoder")}
    for name, parameter in codec.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError(f"nonfinite no-h_T gradient: {name}")
        amount = float(grad.abs().sum().item())
        if name == "local_encoder.2.weight":
            split = codec.config.groups * codec.config.choices
            groups["local_head_0"] += float(grad[:split].abs().sum().item())
            groups["local_head_1"] += float(grad[split:].abs().sum().item())
        elif name == "repair.codebooks":
            groups["codebooks"] += amount
        elif name == "repair.U.weight":
            groups["repair_U"] += amount
        elif name.startswith("cloud_projection."):
            groups["cloud_projection"] += amount
        elif name.startswith("cloud_encoder."):
            groups["cloud_encoder"] += amount
    if any(value <= 0 for value in groups.values()):
        raise RuntimeError(f"no-h_T gate lacks required gradients: {groups}")
    return groups


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        for path in (args.target, args.draft, args.corpus, args.collection_protocol,
                     args.reference_training_protocol, args.reference_run,
                     args.p1_gpu_p0_report):
            if not path.exists():
                raise FileNotFoundError(path)
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("no-h_T gate requires real CUDA")
        device = torch.device(args.device)
        with torch.cuda.device(device):
            free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        if free_mib < args.minimum_free_mib:
            raise RuntimeError(f"insufficient free GPU memory: {free_mib} MiB")
        examples, collection = _load_locked_train(
            args.corpus, args.collection_protocol, args.draft,
        )
        _verify_p1_gpu_gate(args.p1_gpu_p0_report, collection)
        reference = json.loads(args.reference_training_protocol.read_text(
            encoding="utf-8"))
        _validate_reference(reference, collection, args.corpus,
                            args.collection_protocol)
        ref_seed = reference["seeds"][0]
        ref_arm = _reference_arm(
            args.reference_run, ref_seed,
            _sha256(args.reference_training_protocol),
            reference["steps_per_arm"],
            reference["collection_corpus_sha256"],
            reference["draft_model_file_sha256"],
        )
        if collection["target_file_sha256"] != _model_hashes(args.target):
            raise ValueError("Target model differs from frozen collection")
        target = HFTargetBackend(args.target, None, device=device,
                                 dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False,
            attn_implementation="eager", torch_dtype=torch.bfloat16,
        ).to(device)
        config = CodecConfig(
            target_hidden_size=target.hidden_size,
            draft_hidden_size=int(model.config.hidden_size),
            vocab_size=target.vocab_size, window=8,
        )
        if config.__dict__ != reference["config"]:
            raise ValueError("P1-B real GPU gate architecture differs")
        raw_codec = FeedbackCodec(config, model.get_output_embeddings()).to(device)
        draft = HFDraftBackend(model, tokenizer, raw_codec, target=target,
                               device=device, dtype=torch.bfloat16)
        torch.manual_seed(ref_seed)
        codec = MatchedP1Codec(
            config, model.get_output_embeddings(), mode="target_8B",
        ).to(device)
        if _edge_initial_hash(codec) != ref_arm["edge_initial_sha256"]:
            raise RuntimeError("no-h_T gate edge initialization differs from D")
        fixed_head = model.get_output_embeddings().weight.detach().clone()
        constant_examples = constant_hidden_examples(examples[:2])
        optimizer = torch.optim.AdamW(
            (parameter for parameter in codec.parameters() if parameter.requires_grad),
            lr=reference["learning_rate"], weight_decay=reference["weight_decay"],
        )
        history = train_matched_p1_examples(
            examples=constant_examples, target=None, draft=draft, codec=codec,
            optimizer=optimizer, steps=2, prefix_weight=0.0,
            temperature=reference["temperature"], seed=ref_seed,
        )
        gradients = _gradient_evidence(codec)
        constant = torch.ones((1, config.target_hidden_size), device=device,
                              dtype=examples[0].observation.target_hidden.dtype)
        with torch.no_grad():
            before_codes = codec.cloud_encode(constant)
            fold_constant_hidden(codec)
            if not torch.equal(before_codes, codec.cloud_encode(constant)):
                raise RuntimeError("constant-input fold changed the emitted message")
            first = examples[0].observation.target_hidden.reshape(1, -1).to(device)
            second = next((row.observation.target_hidden.reshape(1, -1).to(device)
                           for row in examples[1:]
                           if not torch.equal(row.observation.target_hidden,
                                              examples[0].observation.target_hidden)), None)
            if second is None:
                raise RuntimeError("P1-B gate lacks two distinct real Target states")
            assert_no_target_hidden_information(codec, first, second)
        arm = matched_p1_arm(draft, codec)
        eos = target.tokenizer.eos_token_id
        if not isinstance(eos, int):
            raise ValueError("Target lacks one integer EOS ID")
        round_checks = []
        for item in examples[:2]:
            state = arm.prepare(item.observation.target_view())
            if state.message is None or len(state.message) != 8:
                raise RuntimeError("no-h_T D did not emit exactly eight bytes")
            round_checks.append(_check_one_round(target, arm, item, state, eos))
        if not torch.equal(model.get_output_embeddings().weight, fixed_head):
            raise RuntimeError("frozen Draft output head changed")
        if (any(p.grad is not None for p in target.model.parameters())
                or any(p.grad is not None for p in model.parameters())):
            raise RuntimeError("frozen base accumulated gradients")
        source_files = [ROOT / "scripts" / name for name in
                        ("no_hidden_control.py", "check_no_hidden_gpu_p0.py",
                         "train_no_hidden_p1.py")]
        source_files += [ROOT / "feedback_codec" / name for name in
                         ("codec.py", "fixed_probe.py", "training.py",
                          "hf_backend.py", "decoding.py")]
        _atomic_json(args.output / "report.json", {
            "state": "passed", "scope": "P1-B_no_target_hidden_real_gpu_gate",
            "latency_claim_valid": False,
            "device": str(device), "cuda": True, "dtype": "bfloat16",
            "gpu_name": torch.cuda.get_device_name(device),
            "target_file_sha256": collection["target_file_sha256"],
            "draft_file_sha256": collection["draft_file_sha256"],
            "train_corpus_sha256": _sha256(args.corpus),
            "collection_protocol_sha256": _sha256(args.collection_protocol),
            "reference_training_protocol_sha256":
                _sha256(args.reference_training_protocol),
            "matched_p1_gpu_p0_report_sha256": _sha256(args.p1_gpu_p0_report),
            "reference_d_checkpoint_sha256": ref_arm["checkpoint_sha256"],
            "edge_initial_sha256": ref_arm["edge_initial_sha256"],
            "train_smoke_steps": 2,
            "finite_losses": all(math.isfinite(row["loss"]) for row in history),
            "gradient_l1": gradients,
            "cloud_hidden_invariance": True,
            "eight_byte_message": True,
            "independent_target_ar_round_checks": round_checks,
            "target_frozen": True, "draft_frozen": True,
            "peak_gpu_allocated_mib":
                torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            "seconds": time.time() - started,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                for path in source_files
            },
        })
    except BaseException as exc:
        _atomic_json(args.output / "failure.json", {
            "state": "failed", "error": repr(exc),
            "traceback": traceback.format_exc(), "seconds": time.time() - started,
        })
        raise


if __name__ == "__main__":
    main()
