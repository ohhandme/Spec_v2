"""Train matched B/C/D controls on a sealed P1 train fixed-prefix corpus.

The target's valid verification states are already in the collection artifact.
Only the frozen draft backbone is loaded here. This command does not collect
new boundaries, evaluate acceptance, download models, or access the test set.
Supply an explicit step budget and at least three seeds for formal P1 work.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from feedback_codec.fixed_probe import EVAL_PROTOCOL_VERSION, ProbeExample
from feedback_codec.hf_backend import HFDraftBackend, _FrozenHFBackend
from feedback_codec.training import train_matched_p1_examples


class _FrozenDraftFeatures(_FrozenHFBackend):
    # Reuse the exact final-norm and token-alignment implementation used by
    # HFDraftBackend, without loading the 8B target for every training step.
    teacher_hidden = HFDraftBackend.teacher_hidden


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
        raise ValueError(f"draft model has no local weights: {root}")
    return {str(path.relative_to(root)): _sha256(path) for path in files}


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct nonnegative integers")
    return seeds


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--collection-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seeds", type=_seeds, required=True)
    parser.add_argument("--single-seed-pilot", action="store_true",
                        help="explicitly mark a one-seed mechanism pilot, not a three-seed P1 claim")
    parser.add_argument("--only-seed", type=int,
                        help="run one seed from the frozen --seeds protocol; B/C/D stay paired")
    parser.add_argument("--initialize-only", action="store_true",
                        help="write one shared protocol before seed-parallel workers")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--prefix-weight", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=4000)
    parser.add_argument("--smoke", action="store_true",
                        help="permit a smoke-only corpus and fewer than three seeds")
    parser.add_argument("--resume", action="store_true",
                        help="resume only matching atomic checkpoints in an existing output")
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--diagnostic-probes", type=int, default=128,
                        help="fixed number of locked train rows for post-training code diagnostics")
    parser.add_argument("--p1-gpu-p0-report", type=Path,
                        help="passed real-model B/C/D GPU correctness gate")
    return parser.parse_args()


def _load_locked_train(corpus_path: Path, protocol_path: Path, draft_path: Path,
                       *, smoke: bool = False
                       ) -> tuple[list[ProbeExample], dict]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("eval_protocol_version") != EVAL_PROTOCOL_VERSION:
        raise ValueError("collection/evaluation protocol versions differ")
    required_split = "smoke" if smoke else "train"
    if protocol.get("split") != required_split or protocol.get("window") != 8:
        raise ValueError(f"training requires the locked L=8 {required_split} split")
    if protocol.get("dtype") != "bfloat16":
        raise ValueError("training requires the sealed BF16 target-state corpus")
    if protocol.get("fixed_prefix_sha256") != _sha256(corpus_path):
        raise ValueError("fixed-prefix corpus changed after collection")
    if protocol.get("draft_file_sha256") != _model_hashes(draft_path):
        raise ValueError("draft model files differ from collection")
    if not protocol.get("target_file_sha256"):
        raise ValueError("collection did not pin the target model files")
    examples = [ProbeExample.from_json(json.loads(line)) for line in
                corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples:
        raise ValueError("locked corpus has no fixed-prefix probes")
    width = examples[0].observation.target_hidden.numel()
    for example in examples:
        o = example.observation
        if (o.split != required_split or o.window != 8
                or o.target_hidden_prefix != o.valid_prefix
                or o.valid_prefix + (o.correction_token,) != o.corrected_prefix
                or o.target_hidden.numel() != width or not example.target_future_ids):
            raise ValueError("locked corpus has an invalid or cross-split row")
    return examples, protocol


def _edge_initial_hash(codec: MatchedP1Codec) -> str:
    digest = hashlib.sha256()
    for name, parameter in codec.named_parameters():
        if parameter.requires_grad and not name.startswith("cloud_"):
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(parameter.shape)).encode("ascii"))
            digest.update(str(parameter.dtype).encode("ascii"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _exposure_hash(history: list[dict]) -> str:
    rows = [(row["step"], row["prompt_id"], row["round_index"], row["phase"])
            for row in history]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def _atomic_torch(path: Path, value: object) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, pending)
    pending.replace(path)


def _verify_p1_gpu_gate(report_path: Path, train_protocol: dict) -> dict:
    """Refuse formal training unless the real-model B/C/D CUDA gate passed."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("state") != "passed"
            or report.get("scope") != "matched_p1_real_gpu_p0"
            or report.get("eval_protocol_version") != EVAL_PROTOCOL_VERSION
            or report.get("dtype") != "bfloat16"
            or report.get("window") != 8
            or not isinstance(report.get("device"), str)
            or not report["device"].startswith("cuda")
            or report.get("cuda") is not True
            or not isinstance(report.get("gpu_name"), str)
            or not report["gpu_name"]
            or report.get("target_file_sha256") != train_protocol["target_file_sha256"]
            or report.get("draft_file_sha256") != train_protocol["draft_file_sha256"]
            or report.get("dataset_sha256") != train_protocol["prompt_manifest_sha256"]
            or report.get("target_frozen") is not True
            or report.get("draft_frozen") is not True):
        raise ValueError("real-model B/C/D GPU P0 gate did not pass the locked protocol")
    for field in ("real_p0_report_sha256",):
        value = report.get(field)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"real-model GPU P0 report lacks valid {field}")
    sources = report.get("source_sha256")
    if not isinstance(sources, dict):
        raise ValueError("real-model GPU P0 lacks source hashes")
    for name in ("feedback_codec/codec.py", "feedback_codec/fixed_probe.py",
                 "feedback_codec/hf_backend.py", "feedback_codec/decoding.py"):
        if sources.get(name) != _sha256(ROOT / name):
            raise ValueError(f"real-model GPU P0 source changed: {name}")
    modes = ("edge_only", "protocol_only", "target_8B")
    checks = report.get("checks")
    gradients = report.get("gradient_checks")
    if not isinstance(checks, dict) or not isinstance(gradients, dict):
        raise ValueError("real-model GPU P0 lacks B/C/D checks and gradients")
    for mode in modes:
        check = checks.get(mode)
        grad = gradients.get(mode)
        if (not isinstance(check, dict) or check.get("passed") is not True
                or check.get("exact_target_ar") is not True
                or check.get("finite_logits") is not True
                or check.get("same_prefix") is not True
                or type(check.get("exact_requests")) is not int
                or check["exact_requests"] < 1
                or check.get("cloud_message_bytes") != (8 if mode == "target_8B" else 0)):
            raise ValueError(f"real-model GPU P0 lacks exact B/C/D check: {mode}")
        if (not isinstance(grad, dict) or grad.get("passed") is not True
                or grad.get("local_heads_nonzero") is not True
                or grad.get("repair_nonzero") is not True
                or (mode == "target_8B" and grad.get("cloud_nonzero") is not True)):
            raise ValueError(f"real-model GPU P0 lacks B/C/D gradient: {mode}")
    if (type(report.get("smoke_probe_count")) is not int
            or report["smoke_probe_count"] < 1
            or not isinstance(report.get("smoke_request_ids"), list)
            or len(report["smoke_request_ids"]) < 2):
        raise ValueError("real-model GPU P0 lacks smoke request evidence")
    return report


def _histogram_summary(code_rows: list, *, heads: int, groups: int,
                       choices: int) -> dict:
    """Per-head/per-group usage, entropy in bits, and effective vocabulary."""
    counts = [[[0 for _ in range(choices)] for _ in range(groups)]
              for _ in range(heads)]
    for row in code_rows:
        if heads == 1:
            row = [row]
        if (len(row) != heads or any(len(head) != groups for head in row)):
            raise ValueError("code diagnostics have the wrong head/group shape")
        for head_index, head in enumerate(row):
            for group_index, value in enumerate(head):
                if type(value) is not int or not 0 <= value < choices:
                    raise ValueError("code diagnostic index is outside the codebook")
                counts[head_index][group_index][value] += 1
    entropy = []
    effective = []
    for head in counts:
        entropy_row = []
        effective_row = []
        for group in head:
            total = sum(group)
            entropy_row.append(0.0 if total == 0 else -sum(
                (count / total) * math.log2(count / total)
                for count in group if count
            ))
            effective_row.append(sum(count > 0 for count in group))
        entropy.append(entropy_row)
        effective.append(effective_row)
    return {"histogram": counts, "entropy_bits": entropy,
            "effective_codewords": effective, "rows": len(code_rows)}


def _code_diagnostics(
    *, codec: MatchedP1Codec, draft: _FrozenDraftFeatures,
    examples: list[ProbeExample], history: list[dict], seed: int,
    limit: int,
) -> dict:
    """Use only locked train rows; never inspect validation/test for tuning."""
    selected = sorted(examples, key=lambda item: (
        item.observation.prompt_id, item.observation.round_index
    ))[:limit]
    table = draft.model.get_input_embeddings().weight
    hard_local: list = []
    gumbel_local: list = []
    hard_cloud: list = []
    gumbel_cloud: list = []
    codec.eval()
    with torch.inference_mode():
        for index, item in enumerate(selected):
            o = item.observation
            local_hidden = draft.teacher_hidden(list(o.corrected_prefix), [0])[0].reshape(1, -1)
            if codec.mode == "edge_only":
                rejected = correction = None
                accepted = None
            else:
                rejected = None if o.rejected_token is None else table[
                    o.rejected_token].detach().reshape(1, -1)
                correction = table[o.correction_token].detach().reshape(1, -1)
                accepted = o.accepted_count
            hard = codec.local_encode(local_hidden, rejected, correction, accepted)
            cuda_devices = [local_hidden.device.index if local_hidden.device.index is not None
                            else torch.cuda.current_device()] if local_hidden.device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed + 1_000_000 + index)
                sampled = codec.local_sample_st(
                    local_hidden, rejected, correction, accepted,
                ).argmax(dim=-1)
                if codec.mode == "target_8B":
                    target_hidden = o.target_hidden.reshape(1, -1)
                    cloud_hard = codec.cloud_encode(target_hidden)
                    cloud_sample = codec.cloud_sample_st(target_hidden).argmax(dim=-1)
                else:
                    cloud_hard = cloud_sample = None
            hard_local.append(hard[0].detach().cpu().tolist())
            gumbel_local.append(sampled[0].detach().cpu().tolist())
            if cloud_hard is not None and cloud_sample is not None:
                hard_cloud.append(cloud_hard[0].detach().cpu().tolist())
                gumbel_cloud.append(cloud_sample[0].detach().cpu().tolist())
    config = codec.config
    local_agree = sum(
        a == b for hard, sampled in zip(hard_local, gumbel_local)
        for head_h, head_s in zip(hard, sampled)
        for a, b in zip(head_h, head_s)
    ) / max(1, len(selected) * 2 * config.groups)
    cloud_agree = None if not hard_cloud else sum(
        a == b for hard, sampled in zip(hard_cloud, gumbel_cloud)
        for a, b in zip(hard, sampled)
    ) / (len(selected) * config.groups)
    result = {
        "scope": "locked_train_code_usage_diagnostic_not_acceptance",
        "selected_train_probes": len(selected),
        "selection": "first_N_sorted_by_prompt_id_and_round_index",
        "train_hard_gumbel_local": _histogram_summary(
            [row["train_hard_gumbel_local_codes"] for row in history],
            heads=2, groups=config.groups, choices=config.choices,
        ),
        "post_training_argmax_local": _histogram_summary(
            hard_local, heads=2, groups=config.groups, choices=config.choices,
        ),
        "post_training_gumbel_local": _histogram_summary(
            gumbel_local, heads=2, groups=config.groups, choices=config.choices,
        ),
        "post_training_gumbel_vs_argmax_local_agreement": local_agree,
        "post_training_gumbel_vs_argmax_cloud_agreement": cloud_agree,
    }
    if codec.mode == "target_8B":
        result["train_hard_gumbel_cloud"] = _histogram_summary(
            [row["train_hard_gumbel_cloud_codes"] for row in history],
            heads=1, groups=config.groups, choices=config.choices,
        )
        result["post_training_argmax_cloud"] = _histogram_summary(
            hard_cloud, heads=1, groups=config.groups, choices=config.choices,
        )
        result["post_training_gumbel_cloud"] = _histogram_summary(
            gumbel_cloud, heads=1, groups=config.groups, choices=config.choices,
        )
    return result


def _validate_seed_plan(args: argparse.Namespace) -> None:
    """Distinguish an explicitly limited pilot from a three-seed P1 claim."""
    if args.single_seed_pilot:
        if args.smoke or len(args.seeds) != 1 or args.only_seed is not None:
            raise ValueError("single-seed pilot requires exactly one seed and the full B/C/D sequence")
    elif not args.smoke and len(args.seeds) < 3:
        raise ValueError("formal three-seed P1 training requires at least three distinct seeds")
    if args.only_seed is not None and args.only_seed not in args.seeds:
        raise ValueError("--only-seed must be present in the frozen --seeds list")
    if args.initialize_only and (args.resume or args.only_seed is not None):
        raise ValueError("--initialize-only creates the protocol before seed workers")


def main() -> None:
    args = _args()
    for path in (args.draft, args.corpus, args.collection_protocol):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.smoke and args.p1_gpu_p0_report is None:
        raise ValueError("formal P1 training requires --p1-gpu-p0-report")
    if args.p1_gpu_p0_report is not None and not args.p1_gpu_p0_report.exists():
        raise FileNotFoundError(args.p1_gpu_p0_report)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal P1 training requires a CUDA device")
    if (args.steps <= 0 or args.minimum_free_mib <= 0
            or args.checkpoint_every <= 0 or args.diagnostic_probes <= 0):
        raise ValueError("steps, checkpoint/diagnostic budget, and free memory must be positive")
    for name in ("learning_rate", "temperature"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("prefix_weight", "weight_decay"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not args.smoke and args.prefix_weight != 0.0:
        raise ValueError("P1-A formal comparison must train CE-only (prefix-weight=0)")
    _validate_seed_plan(args)
    examples, protocol = _load_locked_train(
        args.corpus, args.collection_protocol, args.draft, smoke=args.smoke
    )
    if args.p1_gpu_p0_report is not None:
        _verify_p1_gpu_gate(args.p1_gpu_p0_report, protocol)
    device = torch.device(args.device)
    with torch.cuda.device(device):
        free_mib = torch.cuda.mem_get_info()[0] // (1024 * 1024)
    if free_mib < args.minimum_free_mib:
        raise RuntimeError(f"insufficient free GPU memory: {free_mib} MiB")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False
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
    source_files = [ROOT / "feedback_codec" / name for name in
                    ("codec.py", "training.py", "losses.py", "fixed_probe.py")]
    source_files.append(Path(__file__))
    run_protocol = {
        "scope": "P1_locked_fixed_prefix_training",
        "phase": "P1_single_seed_pilot_CE_only" if args.single_seed_pilot
                 else "P1-A_CE_only",
        "claim": "smoke_only_no_research_conclusion" if args.smoke else
                 ("one_seed_pilot_training_loss_only_no_acceptance_result"
                  if args.single_seed_pilot else
                  "teacher_forcing_objective_only_no_acceptance_result"),
        "collection_protocol_sha256": _sha256(args.collection_protocol),
        "collection_corpus_sha256": _sha256(args.corpus),
        "draft_model_file_sha256": protocol["draft_file_sha256"],
        "target_model_file_sha256_from_collection": protocol["target_file_sha256"],
        "config": config.__dict__,
        "device": str(device), "dtype": "bfloat16",
        "steps_per_arm": args.steps, "seeds": args.seeds,
        "smoke": args.smoke,
        "checkpoint_every": args.checkpoint_every,
        "diagnostic_probes": args.diagnostic_probes,
        "p1_gpu_p0_report_sha256": None if args.p1_gpu_p0_report is None
                                 else _sha256(args.p1_gpu_p0_report),
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "prefix_weight": args.prefix_weight, "temperature": args.temperature,
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path)
                          for path in source_files},
    }
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError("resume output directory does not exist")
        prior_protocol = args.output / "training_protocol.json"
        if not prior_protocol.exists() or json.loads(prior_protocol.read_text(
                encoding="utf-8")) != json.loads(json.dumps(run_protocol)):
            raise ValueError("resume protocol differs from frozen source/data/budget")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        _atomic_json(args.output / "training_protocol.json", run_protocol)
    if args.initialize_only:
        print(json.dumps({"output": str(args.output),
                          "status": "shared_training_protocol_initialized",
                          "seeds": args.seeds}, ensure_ascii=False))
        return
    modes = ("edge_only", "protocol_only", "target_8B")
    summary: list[dict] = []
    protocol_sha = _sha256(args.output / "training_protocol.json")
    for seed in ((args.only_seed,) if args.only_seed is not None else args.seeds):
        expected_init: str | None = None
        expected_exposure: str | None = None
        for mode in modes:
            folder = args.output / f"seed-{seed}" / mode
            checkpoint = folder / "trainable.pt"
            completed = folder / "summary.json"
            if completed.exists():
                if not args.resume:
                    raise FileExistsError(completed)
                row = json.loads(completed.read_text(encoding="utf-8"))
                history_file = folder / "history.jsonl"
                diagnostic_file = folder / "code_diagnostics.json"
                if (row.get("seed") != seed or row.get("mode") != mode
                        or not checkpoint.exists() or not history_file.exists()
                        or not diagnostic_file.exists()
                        or row.get("checkpoint_sha256") != _sha256(checkpoint)
                        or row.get("history_sha256") != _sha256(history_file)
                        or row.get("code_diagnostics_sha256") != _sha256(diagnostic_file)):
                    raise ValueError(f"completed arm files failed integrity check: {folder}")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if (not isinstance(payload, dict)
                        or payload.get("format") != "matched_p1_trainable_v1"
                        or payload.get("mode") != mode or payload.get("seed") != seed
                        or payload.get("training_protocol_sha256") != protocol_sha
                        or payload.get("steps") != args.steps
                        or payload.get("source_corpus_sha256") != run_protocol["collection_corpus_sha256"]
                        or payload.get("draft_model_file_sha256") != run_protocol["draft_model_file_sha256"]
                        or payload.get("exposure_sha256") != row.get("exposure_sha256")
                        or payload.get("edge_initial_sha256") != row.get("edge_initial_sha256")):
                    raise ValueError(f"completed checkpoint provenance mismatch: {folder}")
                if expected_init is None:
                    expected_init = row["edge_initial_sha256"]
                    expected_exposure = row["exposure_sha256"]
                elif (row["edge_initial_sha256"] != expected_init
                      or row["exposure_sha256"] != expected_exposure):
                    raise ValueError("completed B/C/D arms differ in initialization/exposure")
                summary.append(row)
                continue

            torch.manual_seed(seed)
            codec = MatchedP1Codec(
                config, model.get_output_embeddings(), mode=mode
            ).to(device)
            edge_init = _edge_initial_hash(codec)
            if expected_init is None:
                expected_init = edge_init
            elif edge_init != expected_init:
                raise RuntimeError("B/C/D edge initial parameters differ")
            optimizer = torch.optim.AdamW(
                (p for p in codec.parameters() if p.requires_grad),
                lr=args.learning_rate, weight_decay=args.weight_decay,
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
                if (not isinstance(previous, dict)
                        or previous.get("format") != "matched_p1_resume_v1"
                        or previous.get("mode") != mode or previous.get("seed") != seed
                        or previous.get("training_protocol_sha256") != protocol_sha
                        or previous.get("edge_initial_sha256") != edge_init
                        or previous.get("steps") != args.steps):
                    raise ValueError(f"resume checkpoint provenance mismatch: {last}")
                codec.load_trainable_state_dict(previous["codec_state"])
                optimizer.load_state_dict(previous["optimizer_state"])
                resume_sampler = previous["sampler_state"]
                prior_wall_s = float(previous["train_wall_s"])
                prior_peak_mib = float(previous["peak_vram_mib"])
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()

            def save_stage(sampler_state: dict) -> None:
                stage = {
                    "format": "matched_p1_resume_v1",
                    "mode": mode, "seed": seed, "steps": args.steps,
                    "training_protocol_sha256": protocol_sha,
                    "edge_initial_sha256": edge_init,
                    "train_wall_s": prior_wall_s + time.perf_counter() - start,
                    "peak_vram_mib": max(prior_peak_mib,
                        torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
                    "codec_state": codec.trainable_state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "sampler_state": sampler_state,
                }
                _atomic_torch(last, stage)

            history = train_matched_p1_examples(
                examples=examples, target=None, draft=draft, codec=codec,
                optimizer=optimizer, steps=args.steps,
                prefix_weight=args.prefix_weight, temperature=args.temperature,
                seed=seed, resume_state=resume_sampler,
                checkpoint_every=args.checkpoint_every,
                checkpoint_callback=save_stage,
            )
            train_wall_s = prior_wall_s + time.perf_counter() - start
            peak_vram_mib = max(prior_peak_mib,
                torch.cuda.max_memory_allocated(device) / (1024 * 1024))
            exposure = _exposure_hash(history)
            if expected_exposure is None:
                expected_exposure = exposure
            elif exposure != expected_exposure:
                raise RuntimeError("B/C/D saw different prompt-window exposure")
            diagnostics = _code_diagnostics(
                codec=codec, draft=draft, examples=examples, history=history,
                seed=seed, limit=args.diagnostic_probes,
            )
            diagnostic_file = folder / "code_diagnostics.json"
            _atomic_json(diagnostic_file, diagnostics)
            _atomic_torch(checkpoint, {
                "format": "matched_p1_trainable_v1",
                "mode": mode,
                "seed": seed,
                "training_protocol_sha256": protocol_sha,
                "source_corpus_sha256": run_protocol["collection_corpus_sha256"],
                "draft_model_file_sha256": run_protocol["draft_model_file_sha256"],
                "steps": args.steps,
                "exposure_sha256": exposure,
                "edge_initial_sha256": edge_init,
                "state": codec.trainable_state_dict(),
            })
            history_file = folder / "history.jsonl"
            pending_history = folder / "history.jsonl.tmp"
            with pending_history.open("w", encoding="utf-8", newline="\n") as stream:
                for row in history:
                    stream.write(json.dumps(row, ensure_ascii=False,
                                            allow_nan=False) + "\n")
            pending_history.replace(history_file)
            row = {
                "seed": seed, "mode": mode,
                "edge_initial_sha256": edge_init,
                "exposure_sha256": exposure,
                "checkpoint_sha256": _sha256(checkpoint),
                "history_sha256": _sha256(history_file),
                "code_diagnostics_sha256": _sha256(diagnostic_file),
                "last_loss": history[-1]["loss"],
                "train_wall_s": train_wall_s,
                "peak_vram_mib": peak_vram_mib,
                "parameter_budget": codec.parameter_budget(),
            }
            summary.append(row)
            _atomic_json(folder / "summary.json", row)
            del optimizer, codec
            gc.collect()
            torch.cuda.empty_cache()
    if args.only_seed is None:
        _atomic_json(args.output / "training_summary.json", summary)
    else:
        # Seed workers share one training_protocol.json but write disjoint
        # output subtrees. A separate merger validates all nine arms before
        # publishing the global summary.
        _atomic_json(args.output / f"seed-{args.only_seed}" / "seed_summary.json",
                     summary)
    print(json.dumps({"output": str(args.output), "trained_arms": len(summary),
                      "claim": "no acceptance result until fixed-prefix evaluation"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
