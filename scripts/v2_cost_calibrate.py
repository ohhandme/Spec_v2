"""P0 GPU cost calibration for the frozen v2 feedback study.

Only train-only prompts are accepted. The measured kernels are synchronized
local GPU operations. Network rows are preregistered scenarios, not observed
WAN timings; this CLI cannot establish end-to-end or asynchronous speedup.
Run only after the exact BF16 packed/KV correctness gate has passed and an
external queue has admitted an idle GPU under the study's three-sample rule.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import sys
import time
import traceback
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_SHA256 = "798a15439ffcbca43d15809bb1d492ee0e5b66daa09ed722288abfae72c418d0"
PRELOCK_SHA256 = "bc9b452e3ac31d4671358acc836d67e49936eabeeea56e137f20e53ae9e90aa1"
DRAFT_WEIGHT_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
REQUIRED_SOURCE = (
    "feedback_codec/codec.py",
    "feedback_codec/decoding.py",
    "feedback_codec/hf_backend.py",
    "feedback_codec/packed.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def quantile_stats(samples: list[float]) -> dict[str, float | int]:
    if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError("timing samples must be nonempty, finite and nonnegative")
    ordered = sorted(samples)

    def q(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(samples),
        "min_ms": ordered[0],
        "p10_ms": q(0.10),
        "p50_ms": q(0.50),
        "p95_ms": q(0.95),
        "max_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
    }


def frame_size_bytes(metadata: dict[str, Any], payload: bytes = b"") -> int:
    """Exact size of cloud_edge.send_frame's length + JSON + payload framing."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be raw bytes")
    header = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body = struct.pack("!I", len(header)) + header + payload
    return 4 + len(body)


def wire_sizes(prefix: list[int], proposals: list[int], accepted_count: int,
               answer: int, *, kind: str = "correction") -> dict[str, int]:
    if not prefix or not proposals or kind not in ("correction", "bonus"):
        raise ValueError("wire calibration requires one nonterminal verification round")
    if type(accepted_count) is not int or not 0 <= accepted_count <= len(proposals):
        raise ValueError("invalid accepted count")
    if type(answer) is not int or answer < 0:
        raise ValueError("invalid answer")
    if (kind == "correction" and accepted_count == len(proposals)) or (
            kind == "bonus" and accepted_count != len(proposals)):
        raise ValueError("answer kind disagrees with accepted count")
    upload = frame_size_bytes(
        {"type": "verify", "prefix": prefix, "proposals": proposals})
    ordinary = {"type": "result", "a": accepted_count, "b": answer,
                "kind": kind, "done": False, "feedback_bytes": 0}
    with_feedback = {**ordinary, "feedback_bytes": 8}
    down_ordinary = frame_size_bytes(ordinary)
    down_feedback = frame_size_bytes(with_feedback, bytes(8))
    extra = down_feedback - down_ordinary
    if extra < 8:
        raise RuntimeError("frame accounting lost the eight raw feedback bytes")
    return {"upload_bytes": upload, "ordinary_reply_bytes": down_ordinary,
            "reply_with_feedback_bytes": down_feedback,
            "incremental_feedback_wire_bytes": extra,
            "raw_feedback_payload_bytes": 8}


def validate_prelock(prelock: dict[str, Any]) -> None:
    if (prelock.get("schema") != "feedback64_v2_cost_prelock_v1"
            or prelock.get("state") != "P0_BLOCKED_PENDING_REAL_GPU_CALIBRATION"
            or prelock.get("program_sha256") != PROGRAM_SHA256
            or prelock.get("message_payload_bytes") != 8):
        raise ValueError("unexpected or already altered P0 cost prelock")
    expected = [
        ("near_edge", 5, 50, 50),
        ("wan_high_rtt", 80, 10, 10),
        ("uplink_limited", 40, 0.5, 10),
        ("downlink_limited", 40, 10, 0.1),
        ("both_constrained", 120, 0.1, 0.1),
    ]
    scenarios = prelock.get("preregistered_scenarios_not_observed_network")
    if not isinstance(scenarios, list) or len(scenarios) != len(expected):
        raise ValueError("cost scenario count differs from the frozen matrix")
    for scenario, (name, rtt, up, down) in zip(scenarios, expected):
        if (scenario.get("name") != name
                or scenario.get("rtt_ms") != rtt
                or scenario.get("uplink_mbps") != up
                or scenario.get("downlink_mbps") != down):
            raise ValueError(f"cost scenario changed: {name}")
    values = prelock.get("synchronous_break_even", {}).get(
        "A_C_sensitivity_values_from_prior_only")
    if values != [2.0, 3.1228, 4.5]:
        raise ValueError("baseline acceptance sensitivity values changed")


def scenario_thresholds(
    stats: dict[str, dict[str, float | int]],
    wire: dict[str, int],
    prelock: dict[str, Any],
    *,
    window: int = 8,
    extra_target_forward_count: int = 0,
    extra_target_forward_p95_ms: float = 0.0,
) -> list[dict[str, Any]]:
    """Conservative synchronous break-even model, not a speed measurement."""
    validate_prelock(prelock)
    if window != 8 or extra_target_forward_count < 0:
        raise ValueError("prelocked L8 or forward count changed")
    for key in ("upload_bytes", "ordinary_reply_bytes",
                "incremental_feedback_wire_bytes"):
        if type(wire.get(key)) is not int or wire[key] < 0:
            raise ValueError(f"invalid wire byte count: {key}")
    if (wire["incremental_feedback_wire_bytes"] < 8
            or not math.isfinite(extra_target_forward_p95_ms)
            or extra_target_forward_p95_ms < 0):
        raise ValueError("invalid feedback or extra-forward cost")
    if extra_target_forward_count and extra_target_forward_p95_ms == 0:
        raise ValueError("an extra Target forward needs measured positive cost")
    for key in ("draft_incremental_token", "packed_target_verify",
                "cloud_encode_pack", "edge_decode", "rank32_repair"):
        if key not in stats:
            raise ValueError(f"missing measured kernel: {key}")
        for quantile in ("p10_ms", "p95_ms"):
            value = stats[key].get(quantile)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid measured {key}.{quantile}")
    rows = []
    for scenario in prelock["preregistered_scenarios_not_observed_network"]:
        up_bytes_per_ms = scenario["uplink_mbps"] * 1_000_000 / 8 / 1000
        down_bytes_per_ms = scenario["downlink_mbps"] * 1_000_000 / 8 / 1000
        baseline = (
            window * stats["draft_incremental_token"]["p10_ms"]
            + stats["packed_target_verify"]["p10_ms"]
            + scenario["rtt_ms"]
            + wire["upload_bytes"] / up_bytes_per_ms
            + wire["ordinary_reply_bytes"] / down_bytes_per_ms
        )
        extra = (
            stats["cloud_encode_pack"]["p95_ms"]
            + stats["edge_decode"]["p95_ms"]
            + window * stats["rank32_repair"]["p95_ms"]
            + extra_target_forward_count * extra_target_forward_p95_ms
            + wire["incremental_feedback_wire_bytes"] / down_bytes_per_ms
        )
        if baseline <= 0:
            raise ValueError("baseline round time must be positive")
        rows.append({
            "scenario": scenario["name"],
            "assumption_based": True,
            "network_observed": False,
            "baseline_round_p10_ms": baseline,
            "extra_feedback_p95_ms": extra,
            "delta_A_min_by_A_C": [
                {"A_C": a, "delta_A_min_tokens_per_round": (1.0 + a) * extra / baseline}
                for a in prelock["synchronous_break_even"][
                    "A_C_sensitivity_values_from_prior_only"]
            ],
        })
    return rows


def measure_cuda(
    name: str, *,
    warmup: int, repeats: int, synchronize: Callable[[], None],
    setup: Callable[[], Any], operation: Callable[[Any], Any],
) -> list[float]:
    """Wall-clock timings with CUDA synchronized immediately before and after."""
    if warmup < 5 or repeats < 20:
        raise ValueError("calibration needs at least five warmups and twenty samples")
    samples: list[float] = []
    for index in range(warmup + repeats):
        state = setup()
        synchronize()
        start_ns = time.perf_counter_ns()
        result = operation(state)
        synchronize()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise RuntimeError(f"invalid synchronized timing for {name}")
        if index >= warmup:
            samples.append(elapsed_ms)
        del result, state
    return samples


def load_train_prompt(path: Path, expected_sha: str, allowed_ids: list[str],
                      preferred_id: str | None = None) -> tuple[str, list[int]]:
    if sha256_file(path) != expected_sha:
        raise ValueError("train-only prompt SHA differs from the packed P0 gate")
    if not allowed_ids:
        raise ValueError("packed P0 report has no train prompt IDs")
    allowed = set(allowed_ids)
    found: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("cost calibration may not read a val/test prompt")
            identifier, tokens = row.get("id"), row.get("input_ids")
            if (not isinstance(identifier, str) or identifier in found
                    or not isinstance(tokens, list) or not tokens
                    or any(type(token) is not int or token < 0 for token in tokens)):
                raise ValueError("invalid or duplicate train prompt")
            found[identifier] = tokens
    if preferred_id is not None:
        if preferred_id not in allowed or preferred_id not in found:
            raise ValueError("chosen prompt is not in the passed train-only packed P0 gate")
        return preferred_id, found[preferred_id]
    for identifier in allowed_ids:
        if identifier in found:
            return identifier, found[identifier]
    raise ValueError("no packed P0 train prompt appears in the supplied file")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target-download-status", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--train-prompts-sha256", required=True)
    parser.add_argument("--prompt-id", help="optional nonterminal L8 prompt from the passed packed P0")
    parser.add_argument("--packed-p0-report", type=Path, required=True)
    parser.add_argument("--packed-p0-report-sha256", required=True)
    parser.add_argument("--runtime-source-root", type=Path, required=True,
                        help="the exact extracted source tree from the passed packed P0 gate")
    parser.add_argument("--cost-prelock", type=Path, default=ROOT / "cost_lock.prelock.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    return parser.parse_args(argv)


def validate_packed_gate_header(gate: dict[str, Any], train_prompts_sha256: str) -> None:
    """Reject a token-only or old-cache P0 before any cost claim is produced."""
    if (gate.get("state") != "passed"
            or gate.get("scope") != "real_qwen3_bf16_packed_target_committed_path_p0"
            or gate.get("dtype") != "bfloat16"
            or gate.get("window") != 8
            or gate.get("train_prompts_sha256") != train_prompts_sha256
            or gate.get("token_gate_version") != "packed_committed_path_v2"
            or gate.get("feature_protocol_version") != "packed_valid_feature_v2"
            or gate.get("packed_feature_provenance_gate_passed") is not True
            or not isinstance(gate.get("packed_feature_provenance_rounds"), int)
            or gate["packed_feature_provenance_rounds"] < 1
            or gate.get("coverage", {}).get("missing") != []):
        raise RuntimeError("real BF16 train-only packed/KV P0 gate is not matched and passed")


def verified_inference_rows(expected: dict[str, Any],
                            download: dict[str, Any]) -> list[dict[str, Any]]:
    """Permit the pinned mirror's two documentation omissions, nothing else."""
    omitted = {"LICENSE", "README.md"}
    if set(download.get("omitted_non_inference_files", [])) != omitted:
        raise RuntimeError("Target downloader omitted an unexpected asset")
    inference_rows = [row for row in expected["files"]
                      if row["path"] not in omitted]
    status_files = {row["file"]: row["sha256"] for row in download["verified_files"]}
    if len(status_files) != len(download["verified_files"]) or set(status_files) != {
            row["path"] for row in inference_rows}:
        raise RuntimeError("Target download status does not list exactly the inference assets")
    for row in inference_rows:
        if status_files.get(row["path"]) != row["sha256"]:
            raise RuntimeError(
                f"official Target status differs from pinned LFS manifest: {row['path']}")
    return inference_rows


def _verify_gate_and_assets(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    if sha256_file(ROOT / "program.md") != PROGRAM_SHA256:
        raise RuntimeError("v2 program differs from its frozen 683-line SHA")
    if sha256_file(args.packed_p0_report) != args.packed_p0_report_sha256:
        raise RuntimeError("packed P0 report SHA changed")
    gate = read_json(args.packed_p0_report)
    validate_packed_gate_header(gate, args.train_prompts_sha256)
    gate_source = gate.get("source_sha256", {})
    if not isinstance(gate_source, dict) or len(gate_source) < 14:
        raise RuntimeError("packed P0 gate omitted its frozen runtime source hashes")
    runtime_root = args.runtime_source_root.resolve()
    if not runtime_root.is_relative_to(ROOT.resolve()):
        raise RuntimeError("runtime source must stay inside the isolated v2 root")
    for relative, expected_sha in gate_source.items():
        if (not isinstance(relative, str) or not isinstance(expected_sha, str)
                or Path(relative).is_absolute() or ".." in Path(relative).parts):
            raise RuntimeError("packed P0 source manifest has an unsafe entry")
        if expected_sha != sha256_file(runtime_root / relative):
            raise RuntimeError(f"packed P0 source changed after the gate: {relative}")
    if any(relative not in gate_source for relative in REQUIRED_SOURCE):
        raise RuntimeError("packed P0 gate omitted required cost runtime code")
    expected = read_json(ROOT / "qwen3-8b-expected.json")
    download = read_json(args.target_download_status)
    if (expected.get("repo_id") != "Qwen/Qwen3-8B"
            or expected.get("revision") != "b968826d9c46dd6066d109eabc6255188de91218"
            or download.get("state") != "complete"
            or download.get("repo_id") != expected["repo_id"]
            or download.get("hf_revision") != expected["revision"]):
        raise RuntimeError("official Target revision/download verification is missing")
    inference_rows = verified_inference_rows(expected, download)
    hashes: dict[str, str] = {}
    for row in inference_rows:
        relative = row["path"]
        if (args.target / relative).stat().st_size != row["bytes"]:
            raise RuntimeError(f"Target asset size changed: {relative}")
        actual = sha256_file(args.target / relative)
        if actual != row["sha256"]:
            raise RuntimeError(f"Target asset changed: {relative}")
        hashes[f"target/{relative}"] = actual
    draft_weight = sha256_file(args.draft / "model.safetensors")
    if draft_weight != DRAFT_WEIGHT_SHA256:
        raise RuntimeError("Draft weight differs from pinned official Qwen3-0.6B")
    hashes["draft/model.safetensors"] = draft_weight
    for relative in ("config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json"):
        hashes[f"draft/{relative}"] = sha256_file(args.draft / relative)
    gated_assets = gate.get("model_file_sha256", {})
    for name, digest in gated_assets.items():
        if hashes.get(name) != digest:
            raise RuntimeError(f"model asset differs from the passed packed P0: {name}")
    if not gated_assets:
        raise RuntimeError("packed P0 gate omitted model asset hashes")
    return gate, hashes


def _verify_tokenizers(args: argparse.Namespace, target_tokenizer: Any,
                       draft_tokenizer: Any) -> dict[str, str]:
    token_manifest = read_json(args.tokenizer_manifest)
    if token_manifest.get("enable_thinking") is not False:
        raise RuntimeError("tokenizer manifest does not lock non-thinking mode")
    hashes = {}
    for role, path, tokenizer in (
        ("target", args.target, target_tokenizer),
        ("draft", args.draft, draft_tokenizer),
    ):
        for name, field in (
            ("tokenizer.json", "tokenizer_json_sha256"),
            ("tokenizer_config.json", "tokenizer_config_sha256"),
        ):
            actual = sha256_file(path / name)
            if actual != token_manifest.get(field):
                raise RuntimeError(f"{role} tokenizer differs from the train prompt manifest")
            hashes[f"{role}/{name}"] = actual
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            raise RuntimeError(f"{role} has no locked chat template")
        actual_template = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if actual_template != token_manifest.get("chat_template_sha256"):
            raise RuntimeError(f"{role} chat template differs from the train prompt manifest")
        hashes[f"{role}/chat_template"] = actual_template
    return hashes


def _prepare_calibration_inputs(target: Any, draft: Any, codec: Any,
                                prompt: list[int], eos: int) -> dict[str, Any]:
    from feedback_codec.packed import HFPackedTargetSession

    proposals: list[int] = []
    for position in range(1, 9):
        token = int(draft.next(prompt + proposals, None, position))
        if token == eos:
            raise RuntimeError("calibration prompt ended draft proposal before L8; choose another train-only P0 prompt")
        proposals.append(token)
    session = HFPackedTargetSession(target, prompt)
    verification = session.verify(proposals)
    accepted = 0
    while accepted < len(proposals) and proposals[accepted] == verification.argmax_ids[accepted]:
        accepted += 1
    answer = int(verification.argmax_ids[accepted])
    if answer == eos:
        raise RuntimeError("calibration boundary is terminal; choose another train-only P0 prompt")
    kind = "correction" if accepted < len(proposals) else "bonus"
    rejected = proposals[accepted] if accepted < len(proposals) else None
    corrected = prompt + proposals[:accepted] + [answer]
    hidden = verification.hidden_at_valid_prefix(accepted).reshape(1, -1)
    local_hidden = draft.teacher_hidden(corrected, [answer])
    if local_hidden.shape != (1, draft.hidden_size):
        raise RuntimeError("draft hidden for repaired next-window state has the wrong shape")
    draft_embedding = draft.model.get_input_embeddings().weight
    rejected_embedding = None if rejected is None else draft_embedding[rejected].reshape(1, -1)
    correction_embedding = draft_embedding[answer].reshape(1, -1)
    local_codes = codec.local_encode(
        local_hidden, rejected_embedding, correction_embedding, accepted)
    cloud_codes = codec.cloud_encode(hidden)
    payload = codec.pack_codes(cloud_codes.reshape(-1))
    if not isinstance(payload, bytes) or len(payload) != 8:
        raise RuntimeError("cloud code did not serialize as eight original bytes")
    first = local_codes[:, 0]
    second = (local_codes[:, 1] + cloud_codes) % 256
    control = codec.repair._lookup(first, 1) + codec.repair._lookup(second, 1)
    if control.shape != (1, 8 * codec.config.code_dim):
        raise RuntimeError("edge decoding produced the wrong control shape")

    # The packed verify already computed h_T(valid_prefix), even for a=0.
    # Count real Target forwards rather than assuming the lookup is free.
    target_forwards = [0]

    def count_forward(_module: Any, _inputs: Any, _output: Any) -> None:
        target_forwards[0] += 1

    hook = target.model.register_forward_hook(count_forward)
    try:
        zero_session = HFPackedTargetSession(target, prompt)
        target_forwards[0] = 0  # prompt prefill is outside a verification round
        zero_verification = zero_session.verify(proposals)
        verification_forwards = target_forwards[0]
        zero_hidden = zero_verification.hidden_at_valid_prefix(0)
        extra_forwards = target_forwards[0] - verification_forwards
    finally:
        hook.remove()
    if (verification_forwards != 1 or extra_forwards != 0
            or zero_hidden.shape != (target.hidden_size,)):
        raise RuntimeError("a=0 valid hidden unexpectedly required another Target forward")
    return {
        "proposals": proposals, "accepted": accepted, "answer": answer,
        "kind": kind, "hidden": hidden, "local_hidden": local_hidden,
        "rejected_embedding": rejected_embedding,
        "correction_embedding": correction_embedding,
        "local_codes": local_codes, "cloud_codes": cloud_codes,
        "control": control, "a0_extra_target_forwards": extra_forwards,
        "wire": wire_sizes(prompt, proposals, accepted, answer, kind=kind),
    }


def run_gpu(args: argparse.Namespace) -> None:
    # Heavy imports are intentionally delayed so pure cost math can be CPU-tested.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prelock = read_json(args.cost_prelock)
    validate_prelock(prelock)
    if sha256_file(args.cost_prelock) != PRELOCK_SHA256:
        raise RuntimeError("cost prelock bytes changed after preregistration")
    if args.warmup < 5 or args.repeats < 20:
        raise ValueError("calibration needs at least five warmups and twenty samples")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("real cost calibration requires an admitted CUDA GPU")
    gate, model_hashes = _verify_gate_and_assets(args)
    runtime_root = args.runtime_source_root.resolve()
    sys.path.insert(0, str(runtime_root))
    import feedback_codec
    if Path(feedback_codec.__file__).resolve().parent != runtime_root / "feedback_codec":
        raise RuntimeError("calibration imported feedback_codec from outside frozen P0 source")
    from feedback_codec.codec import CodecConfig, MatchedP1Codec
    from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
    from feedback_codec.packed import HFPackedTargetSession

    prompt_id, prompt = load_train_prompt(
        args.train_prompts, args.train_prompts_sha256, gate["prompt_ids"], args.prompt_id)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(20261101)
    target_tokenizer = AutoTokenizer.from_pretrained(
        str(args.target), local_files_only=True, trust_remote_code=False)
    draft_tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False)
    tokenizer_hashes = _verify_tokenizers(args, target_tokenizer, draft_tokenizer)
    target = HFTargetBackend(args.target, target_tokenizer, device=device,
                             dtype=torch.bfloat16)
    draft_model = AutoModelForCausalLM.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", torch_dtype=torch.bfloat16).to(device)
    config = CodecConfig(target_hidden_size=target.hidden_size,
                         draft_hidden_size=int(draft_model.config.hidden_size),
                         vocab_size=target.vocab_size, window=8, rank=32)
    codec = MatchedP1Codec(config, draft_model.get_output_embeddings(),
                           mode="target_8B").to(device).eval()
    draft = HFDraftBackend(draft_model, draft_tokenizer, codec.repair,
                           target=target, device=device, dtype=torch.bfloat16)
    if any(parameter.requires_grad for parameter in target.model.parameters()):
        raise RuntimeError("Target is not frozen")
    if any(parameter.requires_grad for parameter in draft.model.parameters()):
        raise RuntimeError("Draft is not frozen")
    eos = target_tokenizer.eos_token_id
    if type(eos) is not int:
        raise RuntimeError("Target EOS is not one token ID")
    for role, model in (("target", target.model), ("draft", draft.model)):
        if next(model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError(f"{role} is not BF16")
    props = torch.cuda.get_device_properties(device)
    source_hashes = {
        relative: sha256_file(runtime_root / relative)
        for relative in (*REQUIRED_SOURCE,
                         "feedback_codec/cloud_edge.py",
                         "feedback_codec/training.py",
                         )
    }
    source_hashes["scripts/v2_cost_calibrate.py"] = sha256_file(Path(__file__))
    source_hashes["cost_lock.prelock.json"] = sha256_file(args.cost_prelock)
    source_hashes["program.md"] = sha256_file(ROOT / "program.md")
    metadata = {
        "state": "calibrating",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_level": "real_local_gpu_kernels_plus_simulated_network",
        "train_prompt_id": prompt_id,
        "train_prompts_sha256": args.train_prompts_sha256,
        "packed_p0_report_sha256": args.packed_p0_report_sha256,
        "cost_prelock_sha256": sha256_file(args.cost_prelock),
        "source_sha256": source_hashes,
        "model_file_sha256": model_hashes,
        "tokenizer_sha256": tokenizer_hashes,
        "device": {
            "requested": args.device, "name": props.name,
            "total_memory_bytes": props.total_memory,
            "cuda_device_index": device.index,
            "uuid_if_exposed_by_torch": (
                None if getattr(props, "uuid", None) is None
                else str(getattr(props, "uuid"))),
        },
        "models": {"target_path": str(args.target.resolve()),
                   "target_revision": "b968826d9c46dd6066d109eabc6255188de91218",
                   "draft_path": str(args.draft.resolve()),
                   "draft_revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "timing": {"warmup": args.warmup, "repeats": args.repeats,
                   "torch_seed_for_untrained_shape_matched_codec": 20261101,
                   "method": "perf_counter_ns with torch.cuda.synchronize before/after each call",
                   "prompt_prefill_excluded_from_per_round_kernels": True,
                   "gpu_idle_admission": "required externally; this CLI does not select or evict GPU processes"},
    }
    atomic_json(args.output / "provenance.json", metadata)
    with torch.inference_mode():
        torch.cuda.reset_peak_memory_stats(device)
        inputs = _prepare_calibration_inputs(target, draft, codec, prompt, eos)
        torch.cuda.synchronize(device)
        allocated_after_load = torch.cuda.memory_allocated(device)
        synchronize = lambda: torch.cuda.synchronize(device)

        def draft_setup() -> Any:
            if len(prompt) == 1:
                return None
            prefill = draft.model(
                input_ids=draft._ids(prompt[:-1]), use_cache=True,
                return_dict=True)
            return prefill.past_key_values

        def draft_incremental(cache: Any) -> Any:
            output = draft.model(
                input_ids=draft._ids([prompt[-1]]),
                past_key_values=cache, use_cache=True,
                output_hidden_states=True, return_dict=True)
            return output.logits[0, -1].argmax()

        direct_draft_id = int(draft_incremental(draft_setup()).item())
        reference_draft_id = int(draft.next(prompt, None, 1))
        if direct_draft_id != reference_draft_id:
            raise RuntimeError("incremental Draft argmax differs from full-prefix reference")

        def edge_decode(_unused: Any) -> Any:
            local_codes = codec.local_encode(
                inputs["local_hidden"], inputs["rejected_embedding"],
                inputs["correction_embedding"], inputs["accepted"])
            first = local_codes[:, 0]
            second = (local_codes[:, 1] + inputs["cloud_codes"]) % 256
            return (codec.repair._lookup(first, 1)
                    + codec.repair._lookup(second, 1))

        samples = {
            "draft_incremental_token": measure_cuda(
                "draft_incremental_token", warmup=args.warmup,
                repeats=args.repeats, synchronize=synchronize,
                setup=draft_setup, operation=draft_incremental),
            "packed_target_verify": measure_cuda(
                "packed_target_verify", warmup=args.warmup,
                repeats=args.repeats, synchronize=synchronize,
                setup=lambda: HFPackedTargetSession(target, prompt),
                operation=lambda session: session.verify(inputs["proposals"])),
            "cloud_encode_pack": measure_cuda(
                "cloud_encode_pack", warmup=args.warmup,
                repeats=args.repeats, synchronize=synchronize,
                setup=lambda: None,
                operation=lambda _unused: codec.pack_codes(
                    codec.cloud_encode(inputs["hidden"]).reshape(-1))),
            "edge_decode": measure_cuda(
                "edge_decode", warmup=args.warmup,
                repeats=args.repeats, synchronize=synchronize,
                setup=lambda: None, operation=edge_decode),
            "rank32_repair": measure_cuda(
                "rank32_repair", warmup=args.warmup,
                repeats=args.repeats, synchronize=synchronize,
                setup=lambda: None,
                operation=lambda _unused: codec.repair.repair_from_control(
                    inputs["local_hidden"], inputs["control"], 1)),
        }
        stats = {key: quantile_stats(value) for key, value in samples.items()}
        thresholds = scenario_thresholds(
            stats, inputs["wire"], prelock, window=8,
            extra_target_forward_count=inputs["a0_extra_target_forwards"],
            extra_target_forward_p95_ms=0.0)
        raw = {"samples_ms": samples, "stats_ms": stats,
               "a0_extra_target_forward_count": inputs["a0_extra_target_forwards"],
               "a0_extra_target_forward_samples_ms": [],
               "a0_extra_target_forward_p95_ms": 0.0,
               "wire_bytes": inputs["wire"],
               "allocated_after_load_bytes": allocated_after_load,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    atomic_json(args.output / "raw_samples.json", raw)
    lock = {
        **metadata, "state": "calibrated_pending_ledger_freeze",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "latency_claim_valid": False,
        "codec_identity": "existing MatchedP1Codec target_8B shape proxy, not an implemented v2 H-VQ64-TASK candidate",
        "final_v2_64bit_candidate_cost_lock_valid": False,
        "draft_token_scope": "one-token KV incremental microbenchmark, not an integrated packed Draft backend",
        "packed_verify_scope": "one L8 target verification forward, prompt prefill excluded",
        "edge_decode_scope": "local protocol-conditioned encoder, hard code modular shift and two codebook lookups on this server GPU; no actual edge device measured",
        "edge_local_hidden_acquisition_scope": "provided precomputed corrected-prefix Draft hidden; its online incremental acquisition and KV timing are not separately measured",
        "rank32_repair_scope": "full repair_from_control including frozen lm_head, conservatively charged as additional cost",
        "cloud_encode_scope": "hard cloud_encode plus exact eight-byte CPU pack",
        "deployment_cost_claim_status": "blocked_until_actual_edge_device_and_integrated_draft_and_network_or_shaping_measurements",
        "a0_extra_target_forward_count": inputs["a0_extra_target_forwards"],
        "a0_extra_target_forward_p95_ms": 0.0,
        "wire_bytes": inputs["wire"],
        "timing_stats_ms": stats,
        "scenario_thresholds": thresholds,
        "network_observed": False,
        "speedup_measured": False,
        "asynchronous_critical_path_measured": False,
        "raw_samples_sha256": sha256_file(args.output / "raw_samples.json"),
        "allocated_after_load_bytes": allocated_after_load,
        "peak_allocated_bytes": raw["peak_allocated_bytes"],
    }
    atomic_json(args.output / "cost_lock.json", lock)
    atomic_json(args.output / "lock_sha256.json", {
        "state": "calibrated_pending_ledger_freeze",
        "cost_lock_sha256": sha256_file(args.output / "cost_lock.json"),
        "raw_samples_sha256": lock["raw_samples_sha256"],
        "must_precede_new_val_acceptance_read": True,
    })


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        run_gpu(args)
    except BaseException as exc:
        atomic_json(args.output / "failure.json", {
            "state": "failed", "error": repr(exc),
            "traceback": traceback.format_exc(),
            "seconds_until_failure": time.time() - started,
            "args_without_credentials": {
                "device": args.device, "warmup": args.warmup,
                "repeats": args.repeats,
                "train_prompts_sha256": args.train_prompts_sha256,
                "packed_p0_report_sha256": args.packed_p0_report_sha256,
            },
        })
        raise


if __name__ == "__main__":
    main()
