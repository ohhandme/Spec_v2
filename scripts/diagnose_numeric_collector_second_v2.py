"""Read-only replay of the second train request rejected by numeric P1 collection.

This diagnostic loads the *frozen* numeric P0/collector source and model assets,
replays one failed train prompt, and records complete packed versus independent
BF16 AR outputs.  It does not change a verifier decision, save training rows,
read validation/test prompts, or establish a new correctness gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


SOURCE_ZIP_SHA = "15f13434baf943e6de02d3a4472508fa06d5da35263b014e383405f0e7c769f5"
SOURCE_MANIFEST_SHA = "594fc0724101f9d140e4ae3a40f29138879527cf09c57592d43ac43325fd608b"
PLAN_SHA = "1530415e2bdb04fdac7547859ca7388c6d73b279728aeca9860f987f306eb51f"
P0_SHA = "6a626705b9b7971aad5dc778888ee438e407dc78bb4ec1b2f5f23de1a13e7701"
DATA_SHA = "1bfaeeb2f8699933a94a180633b5c361601630cb37409502d670158e3a55fdc2"
META_SHA = "aeca32aced7f35488eeb863a974314e872e7375a2372d63575d6204b66e0fd52"
FAILURE_SHA = "99f6a993b37a1b8ba0a03b7ca1b8aabe934c9c78ead45b31fee8742a492539ec"
PROMPT_ID = "v2-train-ultrachat-7096255a802cf44eb7481c0d6dccdf479e4323c0ce44e38c7e217583b56f3328"
SOURCE_ROOT = Path("/mnt/files2/dream/feedback-coded-specdec-v2/runs/p0-numeric-revise1-20261101/source_v1")
TARGET_ROOT = Path("/mnt/files2/dream/feedback-coded-specdec/models/Qwen3-8B")
DRAFT_ROOT = Path("/mnt/files2/dream/specbranch-reproduction/models/Qwen3-0.6B")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def first_difference(left: tuple[int, ...], right: tuple[int, ...]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def round_for_difference(rounds: tuple[Any, ...], difference: int | None
                         ) -> tuple[int | None, int | None, int | None]:
    if difference is None:
        return None, None, None
    offset = 0
    for index, record in enumerate(rounds):
        end = offset + len(record.committed_tokens)
        if offset <= difference < end:
            return index, difference - offset, offset
        offset = end
    return None, None, offset


def logit_summary(logits: Any, candidates: tuple[int, ...]) -> dict[str, Any]:
    values = logits.detach().float()
    top_values, top_ids = values.topk(min(5, values.numel()))
    return {
        "argmax_id": int(logits.argmax().item()), "dtype": str(logits.dtype),
        "top5": [{"token_id": int(token), "logit_fp32_view": float(value)}
                 for token, value in zip(top_ids.cpu().tolist(),
                                         top_values.cpu().tolist())],
        "candidate_logits_fp32_view": {
            str(token): float(values[token].item()) for token in candidates},
        "top2_margin_fp32_view": float((top_values[0] - top_values[1]).item())
        if top_values.numel() >= 2 else None,
    }


def check_frozen_inputs(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    expected = ((args.source_zip, SOURCE_ZIP_SHA),
                (args.source_manifest, SOURCE_MANIFEST_SHA),
                (args.collection_plan, PLAN_SHA),
                (args.failure, FAILURE_SHA),
                (args.dataset, DATA_SHA), (args.dataset_meta, META_SHA),
                (args.p0_report, P0_SHA))
    for path, digest in expected:
        if sha256(path) != digest:
            raise RuntimeError(f"frozen diagnostic input SHA changed: {path}")
    source = args.source_root.resolve(strict=True)
    if source != SOURCE_ROOT:
        raise RuntimeError("diagnostic source is not the frozen numeric source_v1")
    if (args.target.resolve(strict=True) != TARGET_ROOT.resolve(strict=True)
            or args.draft.resolve(strict=True) != DRAFT_ROOT.resolve(strict=True)):
        raise RuntimeError("diagnostic models differ from the locked assets")
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    file_hashes = manifest.get("file_sha256")
    if (manifest.get("archive_sha256") != SOURCE_ZIP_SHA
            or not isinstance(file_hashes, dict) or len(file_hashes) != 24):
        raise RuntimeError("frozen numeric source manifest is invalid")
    for name, digest in file_hashes.items():
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(source) or sha256(path) != digest:
            raise RuntimeError(f"frozen numeric source file changed: {name}")
    sys.path.insert(0, str(source))
    collector = importlib.import_module("scripts.build_packed_probe_numeric_v2")
    if Path(collector.__file__).resolve() != source / "scripts/build_packed_probe_numeric_v2.py":
        raise RuntimeError("imported collector is not frozen numeric source")
    failure = json.loads(args.failure.read_text(encoding="utf-8"))
    plan = json.loads(args.collection_plan.read_text(encoding="utf-8"))
    if (failure.get("state") != "failed_preserved"
            or failure.get("prompt_id") != PROMPT_ID
            or failure.get("plan_sha256") != PLAN_SHA
            or plan.get("scope") != "packed_numeric_recheck_fixed_prefix_collection_plan_v2"
            or plan.get("split") != "train"
            or plan.get("prompt_manifest_sha256") != DATA_SHA
            or plan.get("prompt_manifest_meta_sha256") != META_SHA
            or plan.get("packed_p0_report_sha256") != P0_SHA
            or plan.get("numeric_decision_backend")
               != "packed_with_numerical_recheck_candidate_v1"
            or plan.get("numerical_recheck_top2_margin") != 0.0):
        raise RuntimeError("preserved failure or numeric collector plan mismatched")
    if (plan.get("target_path") != str(args.target.resolve(strict=True))
            or plan.get("draft_path") != str(args.draft.resolve(strict=True))):
        raise RuntimeError("collector model paths changed")
    prompts = collector._load_prompts(args.dataset)
    prompt = next((row for row in prompts
                   if row.id == PROMPT_ID and row.split == "train"), None)
    if prompt is None or collector._prompt_digest(prompt) != failure["prompt_sha256"]:
        raise RuntimeError("failed prompt is not the locked train row")
    p0, target_hashes, draft_hashes = collector._validate_real_p0(
        args.p0_report, target=args.target, draft=args.draft)
    if (target_hashes != plan["target_file_sha256"]
            or draft_hashes != plan["draft_file_sha256"]
            or p0["source_sha256"] != plan["packed_p0_source_sha256"]):
        raise RuntimeError("model/P0 source changed since collector failure")
    return collector, prompt, failure, {"target_file_sha256": target_hashes,
                                        "draft_file_sha256": draft_hashes}


def replay_decision(target: Any, prompt: Any, run: Any, oracle: tuple[int, ...],
                    difference: int, round_index: int, position: int,
                    output_offset: int, eos: int) -> dict[str, Any]:
    import torch
    from feedback_codec.packed import HFPackedTargetSession, _clone_cache_for_probe

    session = HFPackedTargetSession(target, list(prompt.input_ids),
                                   capture_online_cache_snapshot=True,
                                   numerical_recheck_top2_margin=0.0)
    verification = None
    prior_decisions_equal = True
    for index, (record, trace) in enumerate(zip(run.generation.rounds,
                                                 run.traces, strict=True)):
        offset = sum(len(row.committed_tokens)
                     for row in run.generation.rounds[:index])
        remaining = 128 - offset
        bonus_needed = (record.proposals[-1] != eos
                        and len(record.proposals) < remaining)
        verification = session.verify(list(record.proposals), bonus_needed=bonus_needed)
        if (verification.argmax_ids != trace.argmax_ids
                or verification.numerical_recheck_rows != trace.numerical_recheck_rows):
            prior_decisions_equal = False
        if index == round_index:
            break
        answer = (record.correction_token if record.rejected_token is not None
                  else record.bonus_token)
        session.commit(verification, record.accepted_count, answer)
    assert verification is not None
    if position > run.generation.rounds[round_index].accepted_count:
        raise RuntimeError("first difference is not a used Target decision")
    packed_token = int(run.generation.tokens[difference])
    ar_token = int(oracle[difference])
    candidates = tuple(dict.fromkeys((packed_token, ar_token)))
    effective_logits = verification.logits[position].detach().clone()
    canonical_prefix = list(prompt.input_ids) + list(oracle[:difference])
    expected_prefix = session.confirmed + list(
        run.generation.rounds[round_index].proposals[:position])
    if canonical_prefix != expected_prefix:
        raise RuntimeError("packed path differed before the first reported row")
    with torch.inference_mode(), target._lock:
        full_output, _ = target._forward_with_final_hidden(target._ids(canonical_prefix))
    full_logits = full_output.logits[0, -1].detach().clone()
    query = [session.confirmed[-1],
             *run.generation.rounds[round_index].proposals[:position]]
    snapshot = session._preverify_cache_snapshot
    if not session._preverify_snapshot_valid:
        raise RuntimeError("online preverify KV snapshot was not captured")
    with torch.inference_mode(), target._lock:
        short = target.model(input_ids=target._ids(query),
                             past_key_values=_clone_cache_for_probe(snapshot),
                             use_cache=True, output_hidden_states=False,
                             return_dict=True)
    short_logits = short.logits[0, -1].detach().clone()
    return {
        "state": "first_used_decision_replayed",
        "round_index": round_index,
        "position_in_round": position,
        "output_offset_before_round": output_offset,
        "prior_trace_decisions_reproduced": prior_decisions_equal,
        "packed_token": packed_token,
        "independent_ar_token": ar_token,
        "original_packed_top2_margin_fp32_view": verification.top2_margins[position],
        "effective_row_source_mode": verification.row_source_modes[position],
        "effective_row_query_length": verification.row_source_query_lengths[position],
        "numerical_recheck_rows": list(verification.numerical_recheck_rows),
        "numerical_recheck_extra_target_forwards":
            verification.numerical_recheck_extra_target_forwards,
        "cache_length_before": verification.cache_length_before,
        "cache_length_raw": verification.cache_length_raw,
        "cache_commit_source": run.traces[round_index].cache_commit_source,
        "original_packed_effective_row": logit_summary(effective_logits, candidates),
        "independent_full_prefix_row": logit_summary(full_logits, candidates),
        "same_online_kv_short_query_row": logit_summary(short_logits, candidates),
        "effective_vs_full_logits_bitwise_equal": bool(torch.equal(effective_logits,
                                                                 full_logits)),
        "short_vs_full_logits_bitwise_equal": bool(torch.equal(short_logits,
                                                             full_logits)),
        "query_length_short": len(query),
        "query_length_full_prefix": len(canonical_prefix),
        "canonical_prefix_tail": canonical_prefix[-16:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "source-zip", "source-manifest", "collection-plan",
                 "failure", "dataset", "dataset-meta", "p0-report", "target", "draft",
                 "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    provenance = {
        "scope": "P1_numeric_second_train_prompt_divergence_diagnostic_v2",
        "state": "running_no_gate_claim", "p1_gate_claimed": False,
        "latency_claim_valid": False, "device": args.device,
        "diagnostic_script_sha256": sha256(Path(__file__)),
        "source_zip_sha256": sha256(args.source_zip),
        "source_manifest_sha256": sha256(args.source_manifest),
        "collection_plan_sha256": sha256(args.collection_plan),
        "collector_failure_sha256": sha256(args.failure),
        "dataset_sha256": sha256(args.dataset),
        "dataset_meta_sha256": sha256(args.dataset_meta),
        "p0_report_sha256": sha256(args.p0_report),
        "utc_epoch": started,
    }
    try:
        collector, prompt, failure, assets = check_frozen_inputs(args)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from feedback_codec.codec import CodecConfig, FeedbackCodec
        from feedback_codec.decoding import autoregressive_generate
        from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
        from feedback_codec.packed import PackedSpeculativeDecoder

        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("real BF16 replay requires CUDA")
        gpu_admission = collector._admit_idle_gpu(args.device, 22000)
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        if torch.cuda.mem_get_info(device)[0] < 22000 * 1024 * 1024:
            raise RuntimeError("GPU free memory fell below the admission threshold")
        target = HFTargetBackend(args.target, None, device=device, dtype=torch.bfloat16)
        draft_tokenizer = AutoTokenizer.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False)
        draft_model = AutoModelForCausalLM.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False,
            attn_implementation="eager", torch_dtype=torch.bfloat16).to(device)
        codec = FeedbackCodec(
            CodecConfig(target_hidden_size=target.hidden_size,
                        draft_hidden_size=int(draft_model.config.hidden_size),
                        vocab_size=target.vocab_size, window=8),
            draft_model.get_output_embeddings()).to(device)
        draft = HFDraftBackend(draft_model, draft_tokenizer, codec, target=target,
                               device=device, dtype=torch.bfloat16)
        eos = target.tokenizer.eos_token_id
        meta = json.loads(args.dataset_meta.read_text(encoding="utf-8"))
        if type(eos) is not int or eos != meta["target_eos_token_id"]:
            raise RuntimeError("Target EOS changed from locked pilot")
        oracle = autoregressive_generate(target, list(prompt.input_ids), 128, eos)
        decoder = PackedSpeculativeDecoder(
            target, draft, None, 8, eos, capture_target_logits=True,
            feature_provenance_gate=True, numerical_recheck_top2_margin=0.0)
        run = decoder.generate(list(prompt.input_ids), 128)
        packed_tokens = run.generation.tokens
        difference = first_difference(packed_tokens, oracle)
        round_index, position, offset = round_for_difference(
            run.generation.rounds, difference)
        nearby = []
        for index, (record, trace) in enumerate(zip(run.generation.rounds,
                                                     run.traces, strict=True)):
            if round_index is None or abs(index - round_index) > 1:
                continue
            nearby.append({
                "round_index": index,
                "prefix_length": len(record.prefix_before),
                "proposals": list(record.proposals),
                "effective_argmax_ids": list(trace.argmax_ids),
                "original_packed_top2_margins_fp32_view": list(trace.top2_margins),
                "numerical_recheck_rows": list(trace.numerical_recheck_rows),
                "numerical_recheck_extra_target_forwards":
                    trace.numerical_recheck_extra_target_forwards,
                "accepted_count": record.accepted_count,
                "committed_tokens": list(record.committed_tokens),
                "cache_length_before": trace.cache_length_before,
                "cache_length_raw": trace.cache_length_raw,
                "cache_length_after": trace.cache_length_after,
                "cache_commit_source": trace.cache_commit_source,
                "valid_feature_source": (trace.feature_provenance or {}).get("source"),
            })
        replay = (replay_decision(target, prompt, run, oracle,
                                  difference, round_index, position, offset, eos)
                  if (difference is not None and round_index is not None
                      and position is not None and difference < len(oracle)
                      and difference < len(packed_tokens)) else None)
        torch.cuda.synchronize(device)
        atomic_json(args.output / "report.json", {
            **provenance, **assets, "state": "diagnostic_complete_no_gate_claim",
            "prompt_id": prompt.id, "prompt_sha256": failure["prompt_sha256"],
            "prompt_length": len(prompt.input_ids),
            "gpu_admission": gpu_admission,
            "first_difference_zero_based": difference,
            "mismatch_reproduced": difference is not None,
            "packed_output_ids": list(packed_tokens),
            "independent_ar_output_ids": list(oracle),
            "packed_eos": bool(packed_tokens and packed_tokens[-1] == eos),
            "independent_ar_eos": bool(oracle and oracle[-1] == eos),
            "packed_truncated": len(packed_tokens) == 128 and packed_tokens[-1] != eos,
            "independent_ar_truncated": len(oracle) == 128 and oracle[-1] != eos,
            "nearby_rounds": nearby, "first_decision_replay": replay,
            "runtime_extra_target_forwards": sum(
                trace.numerical_recheck_extra_target_forwards for trace in run.traces),
            "seconds": time.time() - started,
        })
    except BaseException as exc:
        atomic_json(args.output / "failure.json", {
            **provenance, "state": "failed_preserved", "error": repr(exc),
            "traceback": traceback.format_exc(), "seconds": time.time() - started,
        })
        raise


if __name__ == "__main__":
    main()
