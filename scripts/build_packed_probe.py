"""Versioned P1 fixed-prefix collection from the original packed Target pass.

The historical sequential collector is not a valid BF16 packed feature source.
This command refuses test data, changed P0 source or model assets, and changed
resumption plans. Each atomic prompt chunk is checked before final merge.
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
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PILOT_TOKENIZED_SHA256 = "1bfaeeb2f8699933a94a180633b5c361601630cb37409502d670158e3a55fdc2"
PILOT_META_SHA256 = "aeca32aced7f35488eeb863a974314e872e7375a2372d63575d6204b66e0fd52"
PACKED_V4_P0_REPORT_SHA256 = "9245cd96aa5c99b7e0dd8e97c1d0a7568ba53b80c7f3ab93a5b53b82e1bec74a"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, FeedbackCodec
from feedback_codec.fixed_probe import ProbeExample, ProbePrompt, _validate_prompts
from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
from feedback_codec.packed_probe import (
    PACKED_PROBE_PROTOCOL_VERSION, canonical_bytes, collect_packed_fixed_probe,
)
from feedback_codec.packed import PACKED_FEATURE_PROTOCOL_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_hashes(root: Path, expected: dict[str, str]) -> dict[str, str]:
    if not root.is_dir() or not expected or not any(
            name.endswith(".safetensors") for name in expected):
        raise ValueError("P0 model hash list is absent or model root is invalid")
    observed = {}
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid P0 model relative path: {name}")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"P0 model file is absent: {name}")
        observed[name] = _sha256(path)
        if observed[name] != digest:
            raise ValueError(f"P0 model file changed: {name}")
    weight_names = {p.relative_to(root).as_posix() for p in root.rglob("*.safetensors")}
    if weight_names != {n for n in expected if n.endswith(".safetensors")}:
        raise ValueError("model weight inventory differs from P0")
    return observed


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
            if row["split"] not in ("train", "val"):
                raise ValueError("sealed test or unsupported split is present")
            prompts.append(ProbePrompt(
                id=row["id"], family_id=family_id, split=row["split"],
                input_ids=tuple(row["input_ids"]),
                task_type=row.get("task_type", row.get("source", "unknown")),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prompt manifest line {line_number}") from exc
    _validate_prompts(prompts)
    return prompts


_canonical_bytes = canonical_bytes


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
                   shard_count: int, eos_token_id: int) -> dict:
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
    if (chunk.get("max_probe_windows_per_prompt") != 8
            or chunk.get("capture_target_logits") is not True
            or chunk.get("feature_provenance_gate") is not True
            or len(chunk["examples"]) > 8
            or chunk.get("eos_token_id") != eos_token_id
            or chunk.get("audit", {}).get("exact_requests") != 1
            or chunk.get("request_audit", {}).get("independent_ar_exact") is not True):
        raise ValueError(f"chunk violates the locked feature/window budget: {path}")
    if (chunk.get("audit", {}).get("probe_windows") != len(chunk["examples"])
            or chunk.get("audit", {}).get("independent_ar_prefix_exact_windows")
               != len(chunk["examples"])):
        raise ValueError(f"chunk probe count differs from examples: {path}")
    request = chunk["request_audit"]
    output_ids = request.get("output_ids")
    if (request.get("prompt_id") != prompt.id or not isinstance(output_ids, list)
            or any(type(token) is not int or token < 0 for token in output_ids)
            or len(output_ids) > 128 or not output_ids
            or request.get("eos") != (output_ids[-1] == eos_token_id)
            or request.get("truncated") != (
                len(output_ids) == 128 and output_ids[-1] != eos_token_id)
            or request.get("verification_rounds") != chunk["audit"].get("raw_rounds")):
        raise ValueError(f"chunk request audit is inconsistent: {path}")
    for row in chunk["examples"]:
        if (row.get("prompt_id") != prompt.id or row.get("split") != prompt.split
                or row.get("collection_protocol_version") != PACKED_PROBE_PROTOCOL_VERSION
                or row.get("packed_feature_protocol_version") != PACKED_FEATURE_PROTOCOL_VERSION
                or row.get("packed_feature_source") != "original_packed_verify_forward"
                or row.get("online_feature_provenance_passed") is not True
                or row.get("independent_ar_prefix_exact") is not True):
            raise ValueError(f"chunk contains a different prompt: {path}")
        trace = row.get("packed_trace")
        if (not isinstance(trace, dict)
                or row.get("packed_trace_sha256") != hashlib.sha256(
                    _canonical_bytes(trace)).hexdigest()
                or trace.get("valid_prefix") != row.get("valid_prefix")
                or trace.get("used_argmax_ids", [])[-1:] != [row.get("correction_token")]
                or len(trace.get("used_argmax_ids", [])) != row.get("accepted_count", -1) + 1):
            raise ValueError(f"chunk packed trace is invalid: {path}")
        provenance = trace.get("feature_provenance")
        if (not isinstance(provenance, dict)
                or provenance.get("source") != "original_packed_verify_forward"
                or provenance.get("cache_source") != "same_online_preverify_KV_snapshot"
                or provenance.get("passed") is not True
                or provenance.get("same_forward_hidden") is not True
                or provenance.get("same_forward_logit_summary") is not True
                or provenance.get("snapshot_replay_hidden_bitwise_equal") is not True
                or provenance.get("snapshot_replay_logits_bitwise_equal") is not True):
            raise ValueError(f"chunk packed feature provenance is invalid: {path}")
        restored = ProbeExample.from_json(row)
        o = restored.observation
        if (o.window != chunk.get("window") or o.target_hidden_prefix != o.valid_prefix
                or o.corrected_prefix != o.valid_prefix + (o.correction_token,)
                or (o.rejected_token is None and
                    (len(o.proposals) != 8 or o.accepted_count != 8))):
            raise ValueError(f"chunk has invalid causal provenance: {path}")
        if (o.target_logit_summary is None
                or o.target_logit_summary.greedy_token_id != o.correction_token
                or not 1 <= len(restored.target_future_ids) <= 8):
            raise ValueError(f"chunk lacks the normal-forward Target logit feature: {path}")
        used = len(o.corrected_prefix) - len(prompt.input_ids)
        if (used <= 0 or o.corrected_prefix != prompt.input_ids + tuple(output_ids[:used])
                or restored.target_future_ids != tuple(output_ids[used:used + 8])):
            raise ValueError(f"chunk future label is not the independent AR suffix: {path}")
    chunk["checksum_sha256"] = checksum
    return chunk


def _validate_real_p0(report_path: Path, *, target: Path, draft: Path) -> tuple[dict, dict, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("state") != "passed" or (
        report.get("scope") != "real_qwen3_bf16_packed_target_committed_path_p0"
        or report.get("dtype") != "bfloat16" or report.get("window") != 8
        or report.get("token_gate_version") != "packed_committed_path_v2"
        or report.get("feature_protocol_version") != PACKED_FEATURE_PROTOCOL_VERSION
        or report.get("packed_feature_provenance_gate_passed") is not True
        or report.get("request_count", 0) < 6
    ):
        raise ValueError("real packed BF16 P0 gate is absent or mismatched")
    rows = report.get("checks")
    oracle = report.get("oracle_results")
    if not isinstance(rows, list) or len(rows) < 12 or not isinstance(oracle, list):
        raise ValueError("P0 report lacks paired Target AR checks")
    reference = {row["id"]: row for row in oracle}
    for arm in ("raw_draft", "random_untrained_8B"):
        arm_rows = [row for row in rows if row.get("arm") == arm]
        if len(arm_rows) != len(reference) or any(
                row.get("id") not in reference
                or row.get("output_ids") != reference[row["id"]].get("output_ids")
                or row.get("eos") != reference[row["id"]].get("eos")
                or row.get("truncated") != reference[row["id"]].get("truncated")
                for row in arm_rows):
            raise ValueError(f"P0 {arm} outputs differ from independent Target AR")
    recorded_model = report.get("model_file_sha256")
    if not isinstance(recorded_model, dict):
        raise ValueError("P0 report lacks model file digests")
    expected_target = {name.removeprefix("target/"): digest
                       for name, digest in recorded_model.items() if name.startswith("target/")}
    expected_draft = {name.removeprefix("draft/"): digest
                      for name, digest in recorded_model.items() if name.startswith("draft/")}
    target_hashes = _model_hashes(target, expected_target)
    draft_hashes = _model_hashes(draft, expected_draft)
    recorded_source = report.get("source_sha256")
    if not isinstance(recorded_source, dict) or len(recorded_source) < 14:
        raise ValueError("P0 report lacks full runtime source digests")
    for name, digest in recorded_source.items():
        path = (ROOT / name).resolve()
        if (not path.is_relative_to(ROOT.resolve()) or not path.is_file()
                or _sha256(path) != digest):
            raise ValueError(f"P0 runtime source differs: {name}")
    return report, target_hashes, draft_hashes


def _release_owned_lock(path: Path, pid: int, plan_sha: str) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return
    if current.get("pid") == pid and current.get("plan_sha256") == plan_sha:
        path.unlink(missing_ok=True)


def _admit_idle_gpu(device_name: str, minimum_free_mib: int) -> dict:
    """Three observed physical-card snapshots before creating a CUDA context."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        if not visible.isdigit() or device_name != "cuda:0":
            raise ValueError("use one physical CUDA_VISIBLE_DEVICES index and cuda:0")
        physical = int(visible)
    elif device_name.startswith("cuda:") and device_name[5:].isdigit():
        physical = int(device_name[5:])
    else:
        raise ValueError("a single explicit CUDA device is required")
    readings = []
    for index in range(3):
        gpu = subprocess.run([
            "nvidia-smi", "-i", str(physical),
            "--query-gpu=uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, check=True)
        fields = [value.strip() for value in gpu.stdout.strip().split(",")]
        if len(fields) != 3:
            raise RuntimeError("unexpected GPU admission reading")
        uuid, free_mib, utilization = fields[0], int(fields[1]), int(fields[2])
        apps = subprocess.run([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, check=True)
        other_pids = [int(parts[1].strip()) for line in apps.stdout.splitlines()
                      if len(parts := line.split(",")) == 2
                      and parts[0].strip() == uuid]
        readings.append({"gpu_index": physical, "uuid": uuid,
                         "free_mib": free_mib, "utilization_percent": utilization,
                         "compute_pids": other_pids})
        if free_mib < minimum_free_mib or utilization >= 10 or other_pids:
            raise RuntimeError(f"GPU is not exclusive and idle: {readings[-1]}")
        if index < 2:
            time.sleep(1)
    return {"readings": readings, "minimum_free_mib": minimum_free_mib}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-meta", type=Path, required=True)
    parser.add_argument("--p0-report", type=Path, required=True,
                        help="passed real BF16 packed v4 gate report.json")
    parser.add_argument("--split", choices=("train", "val"), required=True)
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
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
    if args.window != 8 or args.max_new_tokens != 128:
        raise ValueError("frozen P1 pilot requires L=8 and max_new_tokens=128")
    dataset_sha = _sha256(args.dataset)
    meta = json.loads(args.dataset_meta.read_text(encoding="utf-8"))
    if (dataset_sha != PILOT_TOKENIZED_SHA256
            or _sha256(args.dataset_meta) != PILOT_META_SHA256
            or _sha256(args.p0_report) != PACKED_V4_P0_REPORT_SHA256
            or meta.get("schema") != "feedback64_v2_pilot_tokenized_v1"
            or meta.get("state") != "tokenized_train_val_only"
            or meta.get("output_sha256") != dataset_sha
            or meta.get("enable_thinking") is not False
            or meta.get("sealed_test_loaded") is not False
            or meta.get("counts") != {"train": 1000, "val": 200}
            or type(meta.get("target_eos_token_id")) is not int):
        raise ValueError("pilot train/val tokenized manifest metadata is invalid")
    prompts = _load_prompts(args.dataset)
    if (len(prompts) != 1200 or
            sum(p.split == "train" for p in prompts) != 1000 or
            sum(p.split == "val" for p in prompts) != 200):
        raise ValueError("pilot train/val manifest counts differ from lock")
    selected = [prompt for prompt in prompts if prompt.split == args.split]
    if not selected:
        raise ValueError(f"no {args.split} prompts in manifest")
    p0_report, target_hashes, draft_hashes = _validate_real_p0(
        args.p0_report, target=args.target, draft=args.draft)
    tokenizer_hashes = meta["tokenizer_sha256"]
    for side, model_path in (("target", args.target), ("draft", args.draft)):
        for name in ("tokenizer.json", "tokenizer_config.json"):
            if _sha256(model_path / name) != tokenizer_hashes[f"{side}/{name}"]:
                raise ValueError(f"{side} tokenizer differs from pilot lock")
    source_sha = dict(p0_report["source_sha256"])
    source_sha["feedback_codec/packed_probe.py"] = _sha256(
        ROOT / "feedback_codec/packed_probe.py")
    source_sha["scripts/build_packed_probe.py"] = _sha256(Path(__file__))
    plan = {
        "scope": "packed_fixed_prefix_collection_plan",
        "eval_protocol_version": PACKED_PROBE_PROTOCOL_VERSION,
        "target_path": str(args.target.resolve()),
        "draft_path": str(args.draft.resolve()),
        "target_file_sha256": target_hashes,
        "draft_file_sha256": draft_hashes,
        "source_sha256": source_sha,
        "packed_p0_report_sha256": _sha256(args.p0_report),
        "packed_p0_source_sha256": p0_report["source_sha256"],
        "prompt_manifest_sha256": dataset_sha,
        "prompt_manifest_meta_sha256": _sha256(args.dataset_meta),
        "split": args.split,
        "thinking_mode": "disabled",
        "dtype": "bfloat16",
        "window": args.window,
        "max_new_tokens": args.max_new_tokens,
        "max_future_tokens": args.window,
        "max_probe_windows_per_prompt": 8,
        "capture_target_logits": True,
        "feature_provenance_gate": True,
        "eos_token_id": meta["target_eos_token_id"],
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
                path, prompt, plan_sha, args.shard_count, meta["target_eos_token_id"])
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
        gpu_admission = _admit_idle_gpu(args.device, args.minimum_free_mib)
        if not torch.cuda.is_available():
            raise RuntimeError("real-model collection requires CUDA")
        device = torch.device(args.device)
        with torch.cuda.device(device):
            free_after_context = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        if free_after_context < args.minimum_free_mib:
            raise RuntimeError("GPU free memory fell below admission threshold")
        target = HFTargetBackend(args.target, None, device=device, dtype=torch.bfloat16)
        draft_tokenizer = AutoTokenizer.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False
        )
        for side, tokenizer in (("target", target.tokenizer),
                                ("draft", draft_tokenizer)):
            template_sha = hashlib.sha256(
                tokenizer.chat_template.encode("utf-8")).hexdigest()
            if template_sha != tokenizer_hashes[f"{side}/chat_template"]:
                raise ValueError(f"{side} chat template differs from pilot lock")
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
        if type(eos) is not int or eos != meta["target_eos_token_id"]:
            raise ValueError("target EOS differs from pilot lock")
        for prompt in remaining_prompts:
            started = time.time()
            try:
                corpus = collect_packed_fixed_probe(
                    target=target, raw_draft=draft, prompts=[prompt],
                    window=args.window, max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos, max_probe_windows_per_prompt=8,
                )
            except Exception as exc:
                failures = args.output / "failures"
                failures.mkdir(exist_ok=True)
                failure = failures / (
                    hashlib.sha256(prompt.id.encode("utf-8")).hexdigest()
                    + f"-{time.time_ns()}.json")
                _atomic_json(failure, {
                    "state": "failed_preserved", "prompt_id": prompt.id,
                    "prompt_sha256": _prompt_digest(prompt), "plan_sha256": plan_sha,
                    "gpu_admission": gpu_admission, "exception_type": type(exc).__name__,
                    "exception": str(exc), "traceback": traceback.format_exc(),
                    "started_unix_s": started, "failed_unix_s": time.time(),
                })
                raise
            chunk = {
                "prompt_id": prompt.id,
                "prompt_sha256": _prompt_digest(prompt),
                "plan_sha256": plan_sha,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "eos_token_id": eos,
                "window": args.window,
                "max_probe_windows_per_prompt": 8,
                "capture_target_logits": True,
                "feature_provenance_gate": True,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_admission": gpu_admission,
                "seconds": time.time() - started,
                "audit": corpus.audit,
                "request_audit": corpus.request_audit[0],
                "examples": [example.to_json() for example in corpus.rows],
            }
            chunk["checksum_sha256"] = hashlib.sha256(_canonical_bytes(chunk)).hexdigest()
            path = _chunk_path(args.output, prompt)
            _atomic_json(path, chunk)
            completed[prompt.id] = _checked_chunk(
                path, prompt, plan_sha, args.shard_count, eos
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
             "probe_windows": 0, "full_nonterminal_windows": 0,
             "exact_requests": 0, "independent_ar_prefix_exact_windows": 0,
             "skipped_by_cap": 0}
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
    if (eos_values != {meta["target_eos_token_id"]}
            or audit["exact_requests"] != len(selected)
            or audit["independent_ar_prefix_exact_windows"] != audit["probe_windows"]
            or not audit["probe_windows"]):
        raise RuntimeError("completed shards have inconsistent EOS or zero next-window probes")
    pending.replace(rows_path)
    report = plan | {
        "scope": "packed_fixed_prefix_collection",
        "latency_claim_valid": False,
        "acceptance_only": True,
        "online_feature_provenance_passed": True,
        "collection_plan_sha256": plan_sha,
        "fixed_prefix_sha256": _sha256(rows_path),
        "target_future_source": "independent_target_request_AR_suffix_at_verified_corrected_prefix",
        "eos_token_id": eos_values.pop(),
        "gpu_names": sorted(gpu_names),
        "chunk_sha256": {prompt.id: _sha256(_chunk_path(args.output, prompt))
                         for prompt in selected},
        "audit": audit,
        "sealed_test_loaded": False,
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
