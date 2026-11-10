"""Collect a frozen P1 fixed-prefix corpus from real local Qwen model files.

The input is a pretokenized, prompt-family-disjoint JSONL manifest.  This
command downloads nothing and refuses to replace an existing output folder.
Example:

  python scripts/build_fixed_probe.py --target /models/Qwen3-8B \
    --draft /models/Qwen3-0.6B --dataset prompts.jsonl \
    --dataset-meta prompts.meta.json --split train --thinking-mode disabled \
    --output runs/fixed-probe-train

The output is a data/feature collection artifact, not an acceptance result.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, FeedbackCodec
from feedback_codec.fixed_probe import (
    EVAL_PROTOCOL_VERSION, ProbeExample, ProbePrompt,
    _validate_prompts, collect_fixed_probe,
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


def _load_prompts(path: Path) -> list[ProbePrompt]:
    prompts = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            family_id = row.get("family_id", row.get("prompt_family_sha256"))
            if any(not isinstance(value, str) for value in
                   (row["id"], family_id, row["split"])):
                raise ValueError("id, family_id, and split must be strings")
            prompts.append(ProbePrompt(
                id=row["id"], family_id=family_id, split=row["split"],
                input_ids=tuple(row["input_ids"]),
                task_type=row.get("task_type", "unknown"),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prompt manifest line {line_number}") from exc
    _validate_prompts(prompts)
    return prompts


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def _prompt_digest(prompt: ProbePrompt) -> str:
    return hashlib.sha256(_canonical_bytes({
        "id": prompt.id, "family_id": prompt.family_id,
        "split": prompt.split, "task_type": prompt.task_type,
        "input_ids": prompt.input_ids,
    })).hexdigest()


def _chunk_path(output: Path, prompt: ProbePrompt) -> Path:
    name = hashlib.sha256(prompt.id.encode("utf-8")).hexdigest() + ".json"
    return output / "chunks" / name


def _shard_of(prompt: ProbePrompt, count: int) -> int:
    return int(hashlib.sha256(prompt.id.encode("utf-8")).hexdigest(), 16) % count


def _checked_chunk(path: Path, prompt: ProbePrompt, plan_sha: str,
                   shard_count: int, *, max_probe_windows_per_prompt: int = 0,
                   capture_target_logits: bool = False) -> dict:
    chunk = json.loads(path.read_text(encoding="utf-8"))
    checksum = chunk.pop("checksum_sha256", None)
    if checksum != hashlib.sha256(_canonical_bytes(chunk)).hexdigest():
        raise ValueError(f"chunk checksum differs: {path}")
    if (chunk.get("prompt_id") != prompt.id or
            chunk.get("prompt_sha256") != _prompt_digest(prompt) or
            chunk.get("plan_sha256") != plan_sha or
            chunk.get("shard_count") != shard_count or
            chunk.get("shard_index") != _shard_of(prompt, shard_count) or
            not isinstance(chunk.get("examples"), list) or
            not isinstance(chunk.get("audit"), dict)):
        raise ValueError(f"chunk does not match the locked prompt/protocol: {path}")
    if (chunk.get("max_probe_windows_per_prompt", 0) != max_probe_windows_per_prompt
            or chunk.get("capture_target_logits", False) != capture_target_logits
            or (max_probe_windows_per_prompt
                and len(chunk["examples"]) > max_probe_windows_per_prompt)):
        raise ValueError(f"chunk violates the locked feature/window budget: {path}")
    for row in chunk["examples"]:
        if row.get("prompt_id") != prompt.id or row.get("split") != prompt.split:
            raise ValueError(f"chunk contains a different prompt: {path}")
        restored = ProbeExample.from_json(row)
        o = restored.observation
        if (o.window != chunk.get("window") or o.target_hidden_prefix != o.valid_prefix
                or o.corrected_prefix != o.valid_prefix + (o.correction_token,)):
            raise ValueError(f"chunk has invalid causal provenance: {path}")
        if capture_target_logits and (
            o.target_logit_summary is None
            or o.target_logit_summary.greedy_token_id != o.correction_token
        ):
            raise ValueError(f"chunk lacks the normal-forward Target logit feature: {path}")
    chunk["checksum_sha256"] = checksum
    return chunk


def _validate_real_p0(report_path: Path, *, target: Path, draft: Path,
                      dataset_sha: str) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("state") != "passed" or (
        report.get("scope") != "real_qwen3_bf16_full_prefix_p0_correctness_only"
        or report.get("dtype") != "bfloat16" or report.get("window") != 8
        or report.get("dataset_sha256") != dataset_sha
        or Path(report.get("target_path", "")).resolve() != target.resolve()
        or Path(report.get("draft_path", "")).resolve() != draft.resolve()
    ):
        raise ValueError("real Qwen pair P0 gate is absent or mismatched")
    rows = report.get("checks")
    if not isinstance(rows, list) or not rows or not all(
        row.get("exact_target_ar") is True
        for row in rows if row.get("arm") in ("raw_draft", "random_untrained_8B")
    ):
        raise ValueError("P0 report does not establish exact AR correctness")
    if not any(row.get("arm") == "raw_draft" for row in rows) or (
        not any(row.get("arm") == "random_untrained_8B" for row in rows)
    ):
        raise ValueError("P0 report is missing one speculative arm")
    if report.get("target_file_sha256") != _model_hashes(target) or (
        report.get("draft_file_sha256") != _model_hashes(draft)
    ):
        raise ValueError("P0 report does not match the current model files")
    current_source = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in sorted((ROOT / "feedback_codec").glob("*.py"))
    }
    recorded_source = report.get("source_sha256")
    if not isinstance(recorded_source, dict) or any(
        recorded_source.get(name) != digest
        for name, digest in current_source.items()
    ):
        raise ValueError("P0 report does not match the current feedback_codec source")
    return report


def _release_owned_lock(path: Path, pid: int, plan_sha: str) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return
    if current.get("pid") == pid and current.get("plan_sha256") == plan_sha:
        path.unlink(missing_ok=True)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-meta", type=Path, required=True)
    parser.add_argument("--p0-report", type=Path, required=True,
                        help="passed real Qwen BF16 gate report.json")
    parser.add_argument("--split", choices=("train", "val", "test", "smoke"), required=True)
    parser.add_argument("--allow-test", action="store_true",
                        help="explicit final-test access after val selection")
    parser.add_argument("--thinking-mode", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="continue only an identical existing collection plan")
    parser.add_argument("--initialize-only", action="store_true",
                        help="create a frozen plan before starting parallel shards")
    parser.add_argument("--merge-only", action="store_true",
                        help="verify all chunks and finalize without loading models")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-prompts-this-call", type=int, default=0,
                        help="0 means all; positive limits work per invocation without changing the corpus")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-probe-windows-per-prompt", type=int, default=0,
                        help="0 keeps legacy unlimited collection; v2 P0/P1 passes 8")
    parser.add_argument("--capture-target-logits", action="store_true",
                        help="capture F-LOGIT from the normal verification forward")
    parser.add_argument("--minimum-free-mib", type=int, default=22000)
    return parser.parse_args()


def main() -> None:
    args = _args()
    for path in (args.target, args.draft, args.dataset, args.dataset_meta,
                 args.p0_report):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError("--resume requires the existing output directory")
    if not args.resume and args.output.exists():
        raise FileExistsError(args.output)
    if args.max_prompts_this_call < 0 or args.shard_count <= 0 or (
        not 0 <= args.shard_index < args.shard_count
    ):
        raise ValueError("max-prompts-this-call cannot be negative")
    if args.initialize_only and (args.resume or args.merge_only):
        raise ValueError("--initialize-only starts a new plan without --resume/--merge-only")
    if args.merge_only and not args.resume:
        raise ValueError("--merge-only requires --resume")
    if args.split == "test" and not args.allow_test:
        raise ValueError("test remains sealed unless --allow-test is specified")
    if args.window != 8 or args.max_new_tokens < args.window:
        raise ValueError("P1 requires L=8 and at least one full window of budget")
    if args.max_probe_windows_per_prompt < 0:
        raise ValueError("probe window cap cannot be negative")
    dataset_sha = _sha256(args.dataset)
    meta = json.loads(args.dataset_meta.read_text(encoding="utf-8"))
    if meta.get("output_sha256", meta.get("prompts_jsonl_sha256")) != dataset_sha:
        raise ValueError("dataset metadata does not match the prompt manifest hash")
    recorded_thinking = meta.get("thinking_mode")
    if recorded_thinking is None and isinstance(meta.get("enable_thinking"), bool):
        recorded_thinking = "enabled" if meta["enable_thinking"] else "disabled"
    if recorded_thinking != args.thinking_mode:
        raise ValueError("dataset metadata does not pin the requested thinking mode")
    token_mapping_sha = meta.get("tokenizer_json_sha256")
    if token_mapping_sha is not None:
        for name, model_path in (("target", args.target), ("draft", args.draft)):
            tokenizer_json = model_path / "tokenizer.json"
            if not tokenizer_json.is_file() or _sha256(tokenizer_json) != token_mapping_sha:
                raise ValueError(f"{name} tokenizer.json differs from frozen prompt tokenizer")
    prompts = _load_prompts(args.dataset)
    selected = [prompt for prompt in prompts if prompt.split == args.split]
    if not selected:
        raise ValueError(f"no {args.split} prompts in manifest")
    p0_report = _validate_real_p0(
        args.p0_report, target=args.target, draft=args.draft, dataset_sha=dataset_sha
    )
    # Hash actual model files before any computation or resume.  No hub
    # fallback and no model substitution are allowed.
    target_hashes = _model_hashes(args.target)
    draft_hashes = _model_hashes(args.draft)
    source = sorted((ROOT / "feedback_codec").glob("*.py")) + [Path(__file__)]
    source_sha = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in source}
    plan = {
        "scope": "fixed_prefix_collection_plan",
        "eval_protocol_version": (
            "fixed_prefix_v2_same_forward_logit"
            if args.capture_target_logits else EVAL_PROTOCOL_VERSION
        ),
        "target_path": str(args.target.resolve()),
        "draft_path": str(args.draft.resolve()),
        "target_file_sha256": target_hashes,
        "draft_file_sha256": draft_hashes,
        "source_sha256": source_sha,
        "real_p0_report_sha256": _sha256(args.p0_report),
        "real_p0_target_revision": p0_report["target_revision"],
        "prompt_manifest_sha256": dataset_sha,
        "prompt_manifest_meta_sha256": _sha256(args.dataset_meta),
        "split": args.split,
        "thinking_mode": args.thinking_mode,
        "dtype": "bfloat16",
        "window": args.window,
        "max_new_tokens": args.max_new_tokens,
        "max_probe_windows_per_prompt": args.max_probe_windows_per_prompt,
        "capture_target_logits": args.capture_target_logits,
        "shard_count": args.shard_count,
        "selected_prompt_sha256": [_prompt_digest(prompt) for prompt in selected],
    }
    plan_path = args.output / "collection_plan.json"
    if args.resume:
        if not plan_path.is_file() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("resume inputs/source differ from the frozen collection plan")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "chunks").mkdir()
        _atomic_json(plan_path, plan)
    plan_sha = _sha256(plan_path)
    if args.initialize_only:
        print(json.dumps({"status": "initialized", "output": str(args.output),
                          "planned_prompts": len(selected),
                          "shard_count": args.shard_count}, ensure_ascii=False))
        return
    completed: dict[str, dict] = {}
    expected_paths = {_chunk_path(args.output, prompt).name for prompt in selected}
    if len(expected_paths) != len(selected):
        raise RuntimeError("prompt IDs collide in the content-addressed chunk namespace")
    observed_paths = {path.name for path in (args.output / "chunks").glob("*.json")}
    if observed_paths - expected_paths:
        raise ValueError("chunks directory contains foreign or duplicate prompt files")
    for prompt in selected:
        path = _chunk_path(args.output, prompt)
        if path.is_file():
            completed[prompt.id] = _checked_chunk(
                path, prompt, plan_sha, args.shard_count,
                max_probe_windows_per_prompt=args.max_probe_windows_per_prompt,
                capture_target_logits=args.capture_target_logits,
            )
    final_path = args.output / "collection_protocol.json"
    rows_path = args.output / "fixed_prefix.jsonl"
    if final_path.exists():
        report = json.loads(final_path.read_text(encoding="utf-8"))
        if (len(completed) != len(selected) or not rows_path.is_file()
                or report.get("fixed_prefix_sha256") != _sha256(rows_path)
                or report.get("collection_plan_sha256") != plan_sha):
            raise ValueError("existing final collection report is incomplete or changed")
        print(json.dumps({"status": "complete", "output": str(args.output),
                          "audit": report["audit"]}, ensure_ascii=False))
        return

    remaining_prompts = [prompt for prompt in selected if prompt.id not in completed
                         and _shard_of(prompt, args.shard_count) == args.shard_index]
    if args.max_prompts_this_call:
        remaining_prompts = remaining_prompts[:args.max_prompts_this_call]
    if args.merge_only:
        remaining_prompts = []
    if remaining_prompts:
        shard_lock = args.output / f"shard-{args.shard_index}.lock"
        try:
            with shard_lock.open("x", encoding="utf-8") as owner:
                owner.write(json.dumps({
                    "pid": os.getpid(), "plan_sha256": plan_sha,
                    "shard_index": args.shard_index,
                    "created_unix_s": time.time(),
                }) + "\n")
        except FileExistsError as exc:
            raise RuntimeError(
                f"shard {args.shard_index} already has a worker lock; verify its process before retrying"
            ) from exc
        atexit.register(_release_owned_lock, shard_lock, os.getpid(), plan_sha)
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("real-model collection requires CUDA")
        device = torch.device(args.device)
        with torch.cuda.device(device):
            free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        if free_mib < args.minimum_free_mib:
            raise RuntimeError(f"insufficient free GPU memory before model load: {free_mib} MiB")
        target = HFTargetBackend(args.target, None, device=device, dtype=torch.bfloat16)
        draft_tokenizer = AutoTokenizer.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False
        )
        template_sha = hashlib.sha256(target.tokenizer.chat_template.encode("utf-8")).hexdigest()
        if meta.get("chat_template_sha256") != template_sha:
            raise ValueError("tokenized prompt metadata uses a different chat template")
        draft_model = AutoModelForCausalLM.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False,
            attn_implementation="eager", torch_dtype=torch.bfloat16,
        ).to(device)
        codec = FeedbackCodec(
            CodecConfig(target_hidden_size=target.hidden_size,
                        draft_hidden_size=int(draft_model.config.hidden_size),
                        vocab_size=target.vocab_size, window=args.window),
            draft_model.get_output_embeddings(),
        ).to(device)
        draft = HFDraftBackend(
            draft_model, draft_tokenizer, codec, target=target,
            device=device, dtype=torch.bfloat16,
        )
        eos = target.tokenizer.eos_token_id
        if not isinstance(eos, int):
            raise ValueError("target tokenizer must have one integer EOS token ID")
        for prompt in remaining_prompts:
            started = time.time()
            corpus = collect_fixed_probe(
                target=target, raw_draft=draft, prompts=[prompt],
                window=args.window, max_new_tokens=args.max_new_tokens,
                eos_token_id=eos,
                max_probe_windows_per_prompt=(
                    args.max_probe_windows_per_prompt or None
                ),
                capture_target_logits=args.capture_target_logits,
            )
            chunk = {
                "prompt_id": prompt.id,
                "prompt_sha256": _prompt_digest(prompt),
                "plan_sha256": plan_sha,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "eos_token_id": eos,
                "window": args.window,
                "max_probe_windows_per_prompt": args.max_probe_windows_per_prompt,
                "capture_target_logits": args.capture_target_logits,
                "gpu_name": torch.cuda.get_device_name(device),
                "seconds": time.time() - started,
                "audit": corpus.audit,
                "examples": [example.to_json() for example in corpus.examples],
            }
            chunk["checksum_sha256"] = hashlib.sha256(_canonical_bytes(chunk)).hexdigest()
            path = _chunk_path(args.output, prompt)
            _atomic_json(path, chunk)
            completed[prompt.id] = _checked_chunk(
                path, prompt, plan_sha, args.shard_count
            )
            _atomic_json(args.output / f"progress-shard-{args.shard_index}.json", {
                "status": "collecting",
                "completed_global_prompts_snapshot": len(completed),
                "planned_global_prompts": len(selected),
                "completed_shard_prompts_snapshot": sum(
                    _shard_of(item, args.shard_count) == args.shard_index
                    for item in selected if item.id in completed
                ),
                "last_prompt_id": prompt.id,
            })
    if len(completed) != len(selected):
        if args.merge_only:
            missing = [prompt.id for prompt in selected if prompt.id not in completed]
            raise RuntimeError(f"cannot merge: {len(missing)} prompt chunks missing")
        print(json.dumps({"status": "shard_complete_or_partial", "output": str(args.output),
                          "completed_prompts": len(completed),
                          "planned_prompts": len(selected)}, ensure_ascii=False))
        return
    if args.shard_count > 1 and not args.merge_only:
        print(json.dumps({"status": "awaiting_explicit_merge", "output": str(args.output),
                          "completed_prompts": len(completed)}, ensure_ascii=False))
        return

    audit = {"prompts": 0, "raw_rounds": 0, "terminal_or_budget_rounds": 0,
             "probe_windows": 0, "full_nonterminal_windows": 0}
    eos_values = set()
    gpu_names = set()
    pending = rows_path.with_suffix(".jsonl.tmp")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        for prompt in selected:
            chunk = completed[prompt.id]
            eos_values.add(chunk["eos_token_id"])
            gpu_names.add(chunk["gpu_name"])
            for key in audit:
                audit[key] += int(chunk["audit"][key])
            for row in chunk["examples"]:
                stream.write(json.dumps(row, ensure_ascii=False,
                                        separators=(",", ":"), allow_nan=False) + "\n")
    if len(eos_values) != 1 or not audit["probe_windows"]:
        raise RuntimeError("completed shards have inconsistent EOS or zero next-window probes")
    pending.replace(rows_path)
    report = plan | {
        "scope": "fixed_prefix_data_collection_only",
        "latency_claim_valid": False,
        "collection_plan_sha256": plan_sha,
        "fixed_prefix_sha256": _sha256(rows_path),
        "target_future_source": "independent_target_request_AR_suffix_at_verified_corrected_prefix",
        "eos_token_id": eos_values.pop(),
        "gpu_names": sorted(gpu_names),
        "chunk_sha256": {prompt.id: _sha256(_chunk_path(args.output, prompt))
                         for prompt in selected},
        "audit": audit,
    }
    _atomic_json(final_path, report)
    _atomic_json(args.output / "progress.json", {
        "status": "complete", "completed_prompts": len(completed),
        "planned_prompts": len(selected), "audit": audit,
    })
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "audit": audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
