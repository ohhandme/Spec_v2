"""Independent real-GPU P0 gate for the matched P1 B/C/D inference paths.

It reuses frozen train prompts only for a small correctness/calibration smoke.
The gate establishes causal state use, finite GPU computation, nonzero small
module gradients, and one-round target-AR correctness.  It is not an
acceptance-rate comparison or a speed result, and it never reads sealed test.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, FeedbackCodec, MatchedP1Codec
from feedback_codec.decoding import autoregressive_generate
from feedback_codec.fixed_probe import (
    EVAL_PROTOCOL_VERSION, ProbePrompt, _generate_probe, _view_for,
    collect_fixed_probe, matched_p1_arm,
)
from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
from feedback_codec.training import train_matched_p1_examples
from scripts.build_fixed_probe import (
    _atomic_json, _load_prompts, _model_hashes, _sha256, _validate_real_p0,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-meta", type=Path, required=True)
    parser.add_argument("--real-p0-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--minimum-free-mib", type=int, default=22000)
    return parser.parse_args()


def _check_one_round(target, arm, example, state, eos_token_id: int) -> dict:
    o = example.observation
    proposed = _generate_probe(arm, o, state, eos_token_id)
    confirmed = list(o.corrected_prefix)
    committed = []
    for candidate in proposed:
        correct = int(target.next(confirmed.copy()).token_id)
        if candidate != correct:
            committed.append(correct)
            break
        committed.append(candidate)
        confirmed.append(candidate)
        if candidate == eos_token_id:
            break
    oracle = autoregressive_generate(
        target, list(o.corrected_prefix), len(committed), eos_token_id
    )
    if tuple(committed) != oracle:
        raise RuntimeError("matched P1 one-round verification differs from independent target AR")
    return {
        "prompt_id": o.prompt_id,
        "round_index": o.round_index,
        "proposal_tokens": len(proposed),
        "committed_tokens": len(committed),
        "exact_target_ar": True,
    }


def _gradient_check(codec, draft, example, mode: str) -> dict:
    optimizer = torch.optim.AdamW(
        (parameter for parameter in codec.parameters() if parameter.requires_grad),
        lr=1e-4,
    )
    history = train_matched_p1_examples(
        examples=[example], target=None, draft=draft, codec=codec,
        optimizer=optimizer, steps=1, prefix_weight=0.0,
        temperature=1.0, seed=20261030,
    )
    groups = {
        "local_encoder": 0.0,
        "local_head_0": 0.0,
        "local_head_1": 0.0,
        "local_projection": 0.0,
        "codebooks": 0.0,
        "repair_U": 0.0,
        "cloud_projection": 0.0,
        "cloud_encoder": 0.0,
    }
    for name, parameter in codec.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError(f"nonfinite matched GPU gradient: {mode}/{name}")
        amount = float(grad.abs().sum().item())
        if name.startswith("local_encoder."):
            groups["local_encoder"] += amount
            if name == "local_encoder.2.weight":
                split = codec.config.groups * codec.config.choices
                groups["local_head_0"] = float(grad[:split].abs().sum().item())
                groups["local_head_1"] = float(grad[split:].abs().sum().item())
        elif name.startswith("local_projection."):
            groups["local_projection"] += amount
        elif name == "repair.codebooks":
            groups["codebooks"] += amount
        elif name == "repair.U.weight":
            groups["repair_U"] += amount
        elif name.startswith("cloud_projection."):
            groups["cloud_projection"] += amount
        elif name.startswith("cloud_encoder."):
            groups["cloud_encoder"] += amount
    required = ("local_encoder", "local_head_0", "local_head_1",
                "local_projection", "codebooks", "repair_U")
    if mode == "target_8B":
        required += ("cloud_projection", "cloud_encoder")
    if any(groups[field] <= 0 for field in required):
        raise RuntimeError(f"missing required matched GPU gradient: {mode}: {groups}")
    return {
        "passed": True,
        "loss": history[0]["loss"],
        "gradient_l1": groups,
        "local_heads_nonzero": groups["local_head_0"] > 0 and
            groups["local_head_1"] > 0 and groups["local_projection"] > 0,
        "repair_nonzero": groups["codebooks"] > 0 and groups["repair_U"] > 0,
        "cloud_nonzero": groups["cloud_encoder"] > 0 and
            groups["cloud_projection"] > 0 if mode == "target_8B" else None,
    }


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        for path in (args.target, args.draft, args.dataset, args.dataset_meta,
                     args.real_p0_report):
            if not path.exists():
                raise FileNotFoundError(path)
        if args.request_count < 2 or args.max_new_tokens < 9:
            raise ValueError("matched GPU P0 needs at least two prompts and one next window")
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("matched P1 gate requires real CUDA")
        device = torch.device(args.device)
        with torch.cuda.device(device):
            free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        if free_mib < args.minimum_free_mib:
            raise RuntimeError(f"insufficient free GPU memory: {free_mib} MiB")
        dataset_sha = _sha256(args.dataset)
        dataset_meta = json.loads(args.dataset_meta.read_text(encoding="utf-8"))
        if dataset_meta.get("prompts_jsonl_sha256", dataset_meta.get("output_sha256")) != dataset_sha:
            raise ValueError("prompt manifest does not match its metadata")
        _validate_real_p0(args.real_p0_report, target=args.target, draft=args.draft,
                          dataset_sha=dataset_sha)
        prompts = sorted((prompt for prompt in _load_prompts(args.dataset)
                          if prompt.split == "train"), key=lambda prompt: prompt.id)
        if len(prompts) < args.request_count:
            raise ValueError("not enough locked train prompts for the GPU P0 smoke")
        smoke = [replace(prompt, split="smoke") for prompt in prompts[:args.request_count]]
        target_hashes = _model_hashes(args.target)
        draft_hashes = _model_hashes(args.draft)
        target = HFTargetBackend(args.target, None, device=device, dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(
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
        raw_codec = FeedbackCodec(config, draft_model.get_output_embeddings()).to(device)
        draft = HFDraftBackend(draft_model, tokenizer, raw_codec,
                               target=target, device=device, dtype=torch.bfloat16)
        eos = target.tokenizer.eos_token_id
        if not isinstance(eos, int):
            raise ValueError("target tokenizer lacks one integer EOS ID")
        corpus = collect_fixed_probe(
            target=target, raw_draft=draft, prompts=smoke,
            window=8, max_new_tokens=args.max_new_tokens, eos_token_id=eos,
        )
        if not corpus.examples:
            raise RuntimeError("P0 smoke produced no corrected next-window prefix")
        checks = {}
        gradient_checks = {}
        budgets = {}
        draft_head = draft_model.get_output_embeddings().weight.detach().cpu().clone()
        for mode in ("edge_only", "protocol_only", "target_8B"):
            torch.manual_seed(20261030)
            codec = MatchedP1Codec(config, draft_model.get_output_embeddings(),
                                   mode=mode).to(device)
            arm = matched_p1_arm(draft, codec)
            mode_rows = []
            for example in corpus.examples[:2]:
                view = _view_for(mode, example.observation)
                state = arm.prepare(view)
                if len(state.local) != 2 or any(len(group) != 8 for group in state.local):
                    raise RuntimeError("matched local control is not two eight-index groups")
                if (state.message is not None) != (mode == "target_8B"):
                    raise RuntimeError("matched cloud message crossed the wrong information boundary")
                mode_rows.append(_check_one_round(target, arm, example, state, eos))
            checks[mode] = {"passed": True, "exact_target_ar": True,
                            "exact_requests": len(mode_rows),
                            "finite_logits": True,
                            "same_prefix": True,
                            "probe_examples": len(mode_rows), "rows": mode_rows,
                            "cloud_message_bytes": 8 if mode == "target_8B" else 0}
            gradient_checks[mode] = _gradient_check(codec, draft, corpus.examples[0], mode)
            budgets[mode] = codec.parameter_budget()
            if not torch.equal(draft_model.get_output_embeddings().weight.detach().cpu(), draft_head):
                raise RuntimeError("frozen draft output head changed during P1 GPU gate")
        edge_counts = [budget["edge_trainable"] for budget in budgets.values()]
        if max(edge_counts) != min(edge_counts):
            raise RuntimeError("B/C/D edge trainable parameter counts are unequal")
        if any(parameter.grad is not None for parameter in target.model.parameters()) or (
            any(parameter.grad is not None for parameter in draft.model.parameters())
        ):
            raise RuntimeError("a frozen base accumulated a gradient")
        torch.cuda.synchronize(device)
        source = sorted((ROOT / "feedback_codec").glob("*.py")) + [Path(__file__)]
        report = {
            "state": "passed", "scope": "matched_p1_real_gpu_p0",
            "eval_protocol_version": EVAL_PROTOCOL_VERSION,
            "latency_claim_valid": False,
            "dtype": "bfloat16", "window": 8,
            "device": str(device), "cuda": True,
            "gpu_name": torch.cuda.get_device_name(device),
            "target_path": str(args.target.resolve()),
            "draft_path": str(args.draft.resolve()),
            "target_file_sha256": target_hashes,
            "draft_file_sha256": draft_hashes,
            "dataset_sha256": dataset_sha,
            "real_p0_report_sha256": _sha256(args.real_p0_report),
            "smoke_request_ids": [prompt.id for prompt in smoke],
            "smoke_probe_count": len(corpus.examples),
            "checks": checks,
            "gradient_checks": gradient_checks,
            "parameter_budgets": budgets,
            "target_frozen": True, "draft_frozen": True,
            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            "seconds": time.time() - started,
            "source_sha256": {path.relative_to(ROOT).as_posix(): _sha256(path)
                              for path in source},
        }
        _atomic_json(args.output / "report.json", report)
        print(json.dumps({"state": "passed", "output": str(args.output),
                          "checks": {mode: value["probe_examples"]
                                     for mode, value in checks.items()}}, ensure_ascii=False))
    except BaseException as exc:
        _atomic_json(args.output / "failure.json", {
            "state": "failed", "error": repr(exc),
            "traceback": traceback.format_exc(),
            "seconds": time.time() - started,
        })
        raise


if __name__ == "__main__":
    main()
