"""Replay one preserved packed P1 collector divergence without changing its gate.

This is a diagnostic only. It loads the exact frozen collector source and
assets, reproduces the original raw-Draft packed and independent Target AR
requests, then inspects the first differing committed position. No AR answer
is injected into the packed path, and no training or acceptance result is
produced.
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


SOURCE_ZIP_SHA = "fd09b6d49094550a58c42b5af9258f8003d305fac28c963a9b4a931c343f20ec"
SOURCE_MANIFEST_SHA = "79d8f779eeac7168004a69e152ea5e790d60a52123efd13e80403e93c54eb3e3"
PLAN_SHA = "fe7f35d9ed7f044611850c6fceba9b4df267da62285186168e17138e505a0f09"
P0_SHA = "9245cd96aa5c99b7e0dd8e97c1d0a7568ba53b80c7f3ab93a5b53b82e1bec74a"
DATA_SHA = "1bfaeeb2f8699933a94a180633b5c361601630cb37409502d670158e3a55fdc2"
META_SHA = "aeca32aced7f35488eeb863a974314e872e7375a2372d63575d6204b66e0fd52"


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


def nearby_rounds(run: Any, differing_index: int | None) -> tuple[int | None, list[dict]]:
    start = 0
    selected = None
    for index, record in enumerate(run.generation.rounds):
        end = start + len(record.committed_tokens)
        if differing_index is not None and start <= differing_index < end:
            selected = index
            break
        start = end
    if selected is None and differing_index is not None and run.generation.rounds:
        selected = len(run.generation.rounds) - 1
    rows = []
    for index, (record, trace) in enumerate(zip(
            run.generation.rounds, run.traces, strict=True)):
        if selected is None or abs(index - selected) > 1:
            continue
        begin = sum(len(item.committed_tokens)
                    for item in run.generation.rounds[:index])
        rows.append({
            "round_index": index,
            "output_offset": begin,
            "prefix_length": len(record.prefix_before),
            "prefix_before_tail": list(record.prefix_before[-16:]),
            "proposals": list(record.proposals),
            "packed_all_argmax_ids": list(trace.argmax_ids),
            "packed_used_argmax_ids": list(trace.argmax_ids[:record.accepted_count + 1]),
            "packed_hypothetical_argmax_ids": list(
                trace.argmax_ids[record.accepted_count + 1:]),
            "top2_margins_fp32_view": list(trace.top2_margins),
            "accepted_count": record.accepted_count,
            "rejected_token": record.rejected_token,
            "correction_token": record.correction_token,
            "bonus_token": record.bonus_token,
            "committed_tokens": list(record.committed_tokens),
            "valid_prefix_length": len(trace.valid_prefix),
            "cache_length_before": trace.cache_length_before,
            "cache_length_raw": trace.cache_length_raw,
            "cache_length_after": trace.cache_length_after,
            "feature_provenance": trace.feature_provenance,
        })
    return selected, rows


def logits_summary(logits: Any, candidate_ids: tuple[int, ...]) -> dict[str, Any]:
    scores = logits.detach().float()
    top_values, top_ids = scores.topk(min(5, scores.numel()))
    return {
        "argmax_id": int(logits.argmax().item()),
        "dtype": str(logits.dtype),
        "top5": [{"token_id": int(token), "logit_fp32_view": float(value)}
                 for token, value in zip(top_ids.cpu().tolist(),
                                         top_values.cpu().tolist())],
        "candidate_logits_fp32_view": {
            str(token): float(scores[token].item()) for token in candidate_ids},
        "top2_margin_fp32_view": (
            float((top_values[0] - top_values[1]).item())
            if top_values.numel() >= 2 else None),
    }


def inspect_first_decision(target: Any, prompt: Any, run: Any,
                           oracle: tuple[int, ...], differing_index: int) -> dict:
    import torch
    from feedback_codec.packed import HFPackedTargetSession

    round_index, _ = nearby_rounds(run, differing_index)
    if round_index is None:
        return {"state": "difference_after_final_packed_round"}
    record = run.generation.rounds[round_index]
    begin = sum(len(item.committed_tokens)
                for item in run.generation.rounds[:round_index])
    position = differing_index - begin
    if position < 0 or position > record.accepted_count:
        return {"state": "difference_without_used_packed_row",
                "round_index": round_index, "position_in_round": position}

    online_replay = HFPackedTargetSession(
        target, list(prompt.input_ids), capture_online_cache_snapshot=True)
    original_argmax_reproduced = True
    verification = None
    for index, (expected_record, expected_trace) in enumerate(zip(
            run.generation.rounds, run.traces, strict=True)):
        verification = online_replay.verify(list(expected_record.proposals))
        if verification.argmax_ids != expected_trace.argmax_ids:
            original_argmax_reproduced = False
        if index == round_index:
            break
        answer = (expected_record.correction_token
                  if expected_record.rejected_token is not None
                  else expected_record.bonus_token)
        online_replay.commit(
            verification, expected_record.accepted_count, answer)
    assert verification is not None
    packed_token = (int(run.generation.tokens[differing_index])
                    if differing_index < len(run.generation.tokens) else None)
    ar_token = int(oracle[differing_index]) if differing_index < len(oracle) else None
    candidates = tuple(token for token in (packed_token, ar_token)
                       if token is not None)
    packed_logits = verification.logits[position].detach().clone()
    packed_hidden = verification.hidden[position].detach().clone()
    ar_prefix = list(prompt.input_ids) + list(oracle[:differing_index])
    with torch.inference_mode(), target._lock:
        ar_output, ar_final = target._forward_with_final_hidden(
            target._ids(ar_prefix))
    ar_logits = ar_output.logits[0, -1].detach().clone()
    ar_hidden = ar_final[0, -1].detach().clone()
    if packed_logits.shape != ar_logits.shape:
        raise RuntimeError("diagnostic logits have different vocabulary shape")
    snapshot = online_replay.check_online_suffix_noninterference(
        verification, expected_record.accepted_count)
    result = {
        "state": "first_used_decision_replayed",
        "round_index": round_index,
        "position_in_committed_round": position,
        "packed_row_index": position,
        "packed_live_replay_trace_argmax_reproduced": original_argmax_reproduced,
        "packed_replay_argmax_ids": list(verification.argmax_ids),
        "packed_row": logits_summary(packed_logits, candidates),
        "independent_full_prefix_row": logits_summary(ar_logits, candidates),
        "logits_bitwise_equal": bool(torch.equal(packed_logits, ar_logits)),
        "hidden_bitwise_equal": bool(torch.equal(packed_hidden, ar_hidden)),
        "max_abs_logit_diff_fp32_view": float(
            (packed_logits.float() - ar_logits.float()).abs().max().item()),
        "max_abs_hidden_diff_fp32_view": float(
            (packed_hidden.float() - ar_hidden.float()).abs().max().item()),
        "same_online_KV_suffix_intervention": snapshot,
        "independent_AR_prefix_length": len(ar_prefix),
        "independent_AR_prefix_tail": ar_prefix[-16:],
        "packed_prefix_before_tail": list(record.prefix_before[-16:]),
    }
    # A zero BF16 margin identifies a tie at the observed packed row; it is
    # evidence about numerical sensitivity, not a reason to waive exactness.
    return result


def generate_with_live_capture(target: Any, draft: Any, prompt: Any,
                               oracle: tuple[int, ...], eos: int) -> tuple[Any, dict | None]:
    """Capture the original packed forward at its first *used* AR mismatch.

    The hook changes neither the verifier nor its acceptance rule. Its extra
    Target forwards use independent clones of the exact online preverify KV.
    """
    import torch
    from feedback_codec import packed as packed_module

    session_type = packed_module.HFPackedTargetSession
    original_verify = session_type.verify
    capture: dict[str, Any] = {}

    def wrapped_verify(self: Any, proposals: list[int]) -> Any:
        verification = original_verify(self, proposals)
        # P0's same-query forks set this flag false. Observe only the original
        # online session, never the controlled fork queries themselves.
        if not self._capture_online_cache_snapshot or capture:
            return verification
        accepted = 0
        while (accepted < len(proposals)
               and proposals[accepted] == verification.argmax_ids[accepted]):
            accepted += 1
        output_offset = len(self.confirmed) - len(prompt.input_ids)
        used_rows = accepted
        if accepted < len(proposals):
            used_rows += 1  # correction token
        elif proposals[-1] != eos and output_offset + accepted < 128:
            used_rows += 1  # full-accept bonus token
        for row_index in range(used_rows):
            oracle_index = output_offset + row_index
            packed_token = verification.argmax_ids[row_index]
            target_token = (oracle[oracle_index]
                            if oracle_index < len(oracle) else None)
            if target_token is not None and packed_token == target_token:
                continue
            candidates = tuple(token for token in (packed_token, target_token)
                               if token is not None)
            capture.update({
                "global_output_index_zero_based": oracle_index,
                "packed_row_index": row_index,
                "packed_token": packed_token,
                "independent_AR_token": target_token,
                "prefix_length_before_round": len(self.confirmed),
                "prefix_before_tail": self.confirmed[-16:],
                "proposals": list(proposals),
                "accepted_count": accepted,
                "cache_length_before": verification.cache_length_before,
                "cache_length_raw": verification.cache_length_raw,
                "original_packed_same_forward_row": logits_summary(
                    verification.logits[row_index], candidates),
            })
            try:
                snapshot = self._preverify_cache_snapshot
                if not self._preverify_snapshot_valid:
                    raise RuntimeError("online preverify KV snapshot was not captured")

                def replay(chunk: list[int]) -> tuple[Any, Any]:
                    cache = packed_module._clone_cache_for_probe(snapshot)
                    with torch.inference_mode(), target._lock:
                        result = target.model(
                            input_ids=target._ids(chunk), past_key_values=cache,
                            use_cache=True, output_hidden_states=True,
                            return_dict=True)
                    return result.logits[0, -1].detach().clone(), (
                        result.hidden_states[-1][0, -1].detach().clone())

                same_logits, same_hidden = replay(
                    [self.confirmed[-1], *proposals])
                short_logits, short_hidden = replay(
                    [self.confirmed[-1], *proposals[:row_index]])
                canonical_prefix = self.confirmed + proposals[:row_index]
                if (tuple(canonical_prefix) != tuple(prompt.input_ids)
                        + oracle[:oracle_index]):
                    raise RuntimeError(
                        "packed path left independent AR before captured row")
                with torch.inference_mode(), target._lock:
                    full, final = target._forward_with_final_hidden(
                        target._ids(canonical_prefix))
                ar_logits = full.logits[0, -1].detach().clone()
                ar_hidden = final[0, -1].detach().clone()
                original_hidden = verification.hidden[row_index].detach()
                original_logits = verification.logits[row_index].detach()
                capture["controlled_queries"] = {
                    "same_online_KV_same_query": {
                        **logits_summary(same_logits, candidates),
                        "original_logit_bitwise_equal":
                            bool(torch.equal(original_logits, same_logits)),
                        "original_hidden_bitwise_equal":
                            bool(torch.equal(original_hidden, same_hidden)),
                        "query_length": len(proposals) + 1,
                    },
                    "same_online_KV_shortened_query": {
                        **logits_summary(short_logits, candidates),
                        "original_logit_bitwise_equal":
                            bool(torch.equal(original_logits, short_logits)),
                        "original_hidden_bitwise_equal":
                            bool(torch.equal(original_hidden, short_hidden)),
                        "query_length": row_index + 1,
                    },
                    "independent_full_prefix": {
                        **logits_summary(ar_logits, candidates),
                        "original_logit_bitwise_equal":
                            bool(torch.equal(original_logits, ar_logits)),
                        "original_hidden_bitwise_equal":
                            bool(torch.equal(original_hidden, ar_hidden)),
                        "query_length": len(canonical_prefix),
                    },
                }
                if row_index < len(proposals):
                    altered = list(proposals)
                    altered[row_index] = (
                        altered[row_index] + 1) % target.vocab_size
                    altered_logits, altered_hidden = replay(
                        [self.confirmed[-1], *altered])
                    capture["controlled_queries"]["same_online_KV_altered_suffix"] = {
                        "original_logit_bitwise_equal":
                            bool(torch.equal(original_logits, altered_logits)),
                        "original_hidden_bitwise_equal":
                            bool(torch.equal(original_hidden, altered_hidden)),
                        "query_length": len(altered) + 1,
                        "changed_suffix_from": row_index,
                    }
            except Exception as exc:
                capture["controlled_query_error"] = repr(exc)
            break
        return verification

    session_type.verify = wrapped_verify
    try:
        decoder = packed_module.PackedSpeculativeDecoder(
            target, draft, None, 8, eos, capture_target_logits=True,
            feature_provenance_gate=True)
        packed = decoder.generate(list(prompt.input_ids), 128)
    finally:
        session_type.verify = original_verify
    return packed, capture or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--failure-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-meta", type=Path, required=True)
    parser.add_argument("--p0-report", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    provenance: dict[str, Any] = {
        "scope": "P1_first_packed_collector_divergence_diagnostic",
        "state": "running_no_gate_claim", "p1_gate_claimed": False,
        "online_speed_claim_valid": False, "diagnostic_script_sha256":
            sha256(Path(__file__)),
        "source_zip_sha256": sha256(args.source_zip),
        "source_manifest_sha256": sha256(args.source_manifest),
        "collection_plan_sha256": sha256(args.collection_plan),
        "collector_failure_sha256": sha256(args.failure),
        "dataset_sha256": sha256(args.dataset),
        "dataset_meta_sha256": sha256(args.dataset_meta),
        "p0_report_sha256": sha256(args.p0_report),
        "device": args.device, "utc_epoch": started,
    }
    try:
        expected = {
            "source_zip_sha256": SOURCE_ZIP_SHA,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA,
            "collection_plan_sha256": PLAN_SHA,
            "collector_failure_sha256": args.failure_sha256,
            "dataset_sha256": DATA_SHA,
            "dataset_meta_sha256": META_SHA,
            "p0_report_sha256": P0_SHA,
        }
        if any(provenance[key] != value for key, value in expected.items()):
            raise RuntimeError("frozen diagnostic input SHA mismatch")
        source = args.source_root.resolve(strict=True)
        manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        source_files = manifest.get("file_sha256")
        if (manifest.get("archive_sha256") != SOURCE_ZIP_SHA
                or not isinstance(source_files, dict) or len(source_files) != 18
                or any(not (source / name).resolve().is_relative_to(source)
                       or sha256(source / name) != digest
                       for name, digest in source_files.items())):
            raise RuntimeError("frozen collector source differs from manifest")
        sys.path.insert(0, str(source))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from feedback_codec.codec import CodecConfig, FeedbackCodec
        from feedback_codec.decoding import autoregressive_generate
        from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
        collector = importlib.import_module("scripts.build_packed_probe")
        if collector.ROOT.resolve() != source:
            raise RuntimeError("imported collector is not the frozen source")

        failure = json.loads(args.failure.read_text(encoding="utf-8"))
        plan = json.loads(args.collection_plan.read_text(encoding="utf-8"))
        if (failure.get("state") != "failed_preserved"
                or failure.get("plan_sha256") != PLAN_SHA
                or failure.get("prompt_sha256") is None
                or plan.get("scope") != "packed_fixed_prefix_collection_plan"
                or plan.get("split") != "train"
                or plan.get("prompt_manifest_sha256") != DATA_SHA
                or plan.get("prompt_manifest_meta_sha256") != META_SHA
                or plan.get("packed_p0_report_sha256") != P0_SHA
                or any(source_files.get(name) != digest
                       for name, digest in plan.get("source_sha256", {}).items())
                or len(plan.get("source_sha256", {})) != 16):
            raise RuntimeError("preserved failure or collector plan provenance mismatched")
        prompt_rows = collector._load_prompts(args.dataset)
        prompt = next((row for row in prompt_rows
                       if row.id == failure["prompt_id"] and row.split == "train"), None)
        if prompt is None or collector._prompt_digest(prompt) != failure["prompt_sha256"]:
            raise RuntimeError("failed prompt is not the locked train row")
        p0, target_hashes, draft_hashes = collector._validate_real_p0(
            args.p0_report, target=args.target, draft=args.draft)
        if (target_hashes != plan["target_file_sha256"]
                or draft_hashes != plan["draft_file_sha256"]
                or p0["source_sha256"] != plan["packed_p0_source_sha256"]):
            raise RuntimeError("model or P0 source changed since collector failure")
        provenance.update({
            "prompt_id": prompt.id, "prompt_sha256": failure["prompt_sha256"],
            "prompt_length": len(prompt.input_ids),
            "target_file_sha256": target_hashes,
            "draft_file_sha256": draft_hashes,
            "p0_source_sha256": p0["source_sha256"],
            "torch_version": torch.__version__,
        })
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("real BF16 diagnostic requires CUDA")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        if torch.cuda.mem_get_info(device)[0] < 22000 * 1024 * 1024:
            raise RuntimeError("diagnostic CUDA card lacks 22000 MiB free before load")
        torch.cuda.reset_peak_memory_stats(device)
        target = HFTargetBackend(
            args.target, None, device=device, dtype=torch.bfloat16)
        if (next(target.model.parameters()).dtype != torch.bfloat16
                or target.model.config._attn_implementation != "eager"):
            raise RuntimeError("diagnostic requires frozen BF16 eager Target")
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
        draft = HFDraftBackend(
            draft_model, draft_tokenizer, codec, target=target,
            device=device, dtype=torch.bfloat16)
        eos = target.tokenizer.eos_token_id
        if eos != plan["eos_token_id"]:
            raise RuntimeError("Target EOS changed since collector failure")
        oracle = autoregressive_generate(
            target, list(prompt.input_ids), 128, eos)
        packed, live_capture = generate_with_live_capture(
            target, draft, prompt, oracle, eos)
        packed_tokens = packed.generation.tokens
        difference = first_difference(packed_tokens, oracle)
        selected_round, traces = nearby_rounds(packed, difference)
        replay = (inspect_first_decision(
            target, prompt, packed, oracle, difference)
            if difference is not None and difference < len(packed_tokens)
            and difference < len(oracle) else None)
        torch.cuda.synchronize(device)
        atomic_json(args.output / "report.json", {
            **provenance, "state": "diagnostic_complete_no_gate_claim",
            "first_difference_zero_based": difference,
            "mismatch_reproduced": difference is not None,
            "packed_output_ids": list(packed_tokens),
            "independent_AR_output_ids": list(oracle),
            "packed_eos": bool(packed_tokens and packed_tokens[-1] == eos),
            "independent_AR_eos": bool(oracle and oracle[-1] == eos),
            "packed_truncated": len(packed_tokens) == 128
                                 and (not packed_tokens or packed_tokens[-1] != eos),
            "independent_AR_truncated": len(oracle) == 128
                                        and (not oracle or oracle[-1] != eos),
            "packed_rounds": len(packed.traces),
            "diagnostic_order": "independent_AR_then_original_packed_with_read_only_verify_wrapper",
            "collector_failure_order": "original_packed_then_independent_AR",
            "first_difference_round_index": selected_round,
            "adjacent_rounds": traces,
            "first_live_used_decision": live_capture,
            "first_decision_online_history_replay": replay,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "seconds": time.time() - started,
        })
    except BaseException as exc:
        atomic_json(args.output / "failure.json", {
            **provenance, "state": "diagnostic_failed_preserved_no_gate_claim",
            "error": repr(exc), "traceback": traceback.format_exc(),
            "seconds": time.time() - started,
        })
        raise


if __name__ == "__main__":
    main()
