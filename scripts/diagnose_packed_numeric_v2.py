"""Bounded, read-only BF16 diagnosis of one frozen packed P0 argmax failure.

This is a diagnostic, never a P0 pass or a verifier fallback. It replays the
failed prefix/proposals under different *legal* query shapes, preserving the
official Target weights, BF16 arithmetic, greedy rule, and source archive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def row_summary(logits, hidden, candidate_ids: tuple[int, int]) -> tuple[dict, object, object]:
    """Summarize one already-computed Target row without a second forward."""
    import torch

    scores = logits.detach().float().cpu().clone()
    state = hidden.detach().float().cpu().clone()
    top_values, top_ids = scores.topk(min(5, scores.numel()))
    summary = {
        "argmax_id": int(logits.argmax().item()),
        "logits_dtype": str(logits.dtype),
        "hidden_dtype": str(hidden.dtype),
        "top5": [{"id": int(token), "value_fp32_view": float(value)}
                 for token, value in zip(top_ids.tolist(), top_values.tolist())],
        "candidate_logits_fp32_view": {
            str(token): float(scores[token]) for token in candidate_ids},
        "top2_margin_fp32_view": float(top_values[0] - top_values[1]),
        "hidden_abs_max_fp32_view": float(state.abs().max()),
        "hidden_l2_fp32_view": float(torch.linalg.vector_norm(state)),
    }
    return summary, scores, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--source-zip-sha256", required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--failure-sha256", required=True)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target-download-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(source))
    import torch
    import transformers
    from transformers import AutoTokenizer
    from feedback_codec.hf_backend import HFTargetBackend
    from feedback_codec.packed import HFPackedTargetSession
    frozen_path = source / "scripts/run_packed_p0.py"
    spec = importlib.util.spec_from_file_location("_frozen_run_packed_p0", frozen_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen packed P0 gate")
    frozen_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen_gate)

    provenance = {"state": "running", "scope": "one_frozen_bf16_packed_numeric_failure",
                  "failure_sha256": sha256(args.failure),
                  "source_zip_sha256": sha256(args.source_zip),
                  "diagnostic_script_sha256": sha256(Path(__file__)),
                  "torch_version": torch.__version__,
                  "transformers_version": transformers.__version__,
                  "cuda_version": torch.version.cuda,
                  "device": args.device, "dtype": "bfloat16",
                  "latency_claim_valid": False,
                  "p0_gate_claimed": False}
    try:
        if provenance["failure_sha256"] != args.failure_sha256:
            raise RuntimeError("frozen failure report hash changed")
        if provenance["source_zip_sha256"] != args.source_zip_sha256:
            raise RuntimeError("frozen source ZIP hash changed")
        failed = json.loads(args.failure.read_text(encoding="utf-8"))
        if failed.get("state") != "failed" or failed.get("coverage_mode") != "full":
            raise RuntimeError("input is not the expected full P0 failure")
        if frozen_gate.source_hashes() != failed["source_sha256"]:
            raise RuntimeError("diagnostic imports differ from the failed source archive")
        if sha256(args.train_prompts) != failed["train_prompts_sha256"]:
            raise RuntimeError("six-prompt train-only input changed")
        prompt_rows = [json.loads(line) for line in args.train_prompts.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if any(row.get("split") != "train" for row in prompt_rows):
            raise RuntimeError("diagnostic input is not train-only")
        evidence = failed["evidence"]
        prompt = next((row for row in prompt_rows if row.get("id") == evidence["id"]), None)
        prefix = evidence["prefix_before"]
        proposals = evidence["proposals"]
        position = evidence["position"]
        if (prompt is None or prefix[:len(prompt["input_ids"])] != prompt["input_ids"]
                or not 0 <= position < len(proposals)
                or len(evidence["packed_argmax_ids"]) != len(proposals) + 1):
            raise RuntimeError("failure trace does not match frozen train prompt")
        provenance["source_file_sha256"] = frozen_gate.source_hashes()
        model_hashes = frozen_gate.verify_models(
            args.target, args.draft, args.target_download_status)
        if model_hashes != failed["model_file_sha256"]:
            raise RuntimeError("Target/Draft model assets changed since failure")
        provenance["model_file_sha256"] = model_hashes
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.target), local_files_only=True, trust_remote_code=False)
        draft_tokenizer = AutoTokenizer.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False)
        token_manifest = json.loads(args.tokenizer_manifest.read_text(encoding="utf-8"))
        provenance["tokenizer_manifest_sha256"] = sha256(args.tokenizer_manifest)
        tokenizer_hashes = frozen_gate.verify_tokenizers(
            args.target, args.draft, tokenizer, draft_tokenizer, token_manifest)
        if tokenizer_hashes != failed["tokenizer_sha256"]:
            raise RuntimeError("tokenizer changed since failure")
        provenance["gpu_admission"] = frozen_gate.admit_gpu(args.device)

        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        target = HFTargetBackend(args.target, tokenizer, device=device,
                                 dtype=torch.bfloat16)
        if (next(target.model.parameters()).dtype != torch.bfloat16
                or target.model.config._attn_implementation != "eager"):
            raise RuntimeError("diagnostic requires the original BF16 eager model")
        candidate_ids = (int(evidence["independent_target_argmax"]),
                         int(evidence["packed_argmax"]))
        reference_prefix = prefix + proposals[:position]
        records: dict[str, dict] = {}
        raw_scores: dict[str, torch.Tensor] = {}
        raw_hidden: dict[str, torch.Tensor] = {}

        def record(name: str, output, row_index: int, *, prefill_length: int,
                   query_length: int, explicit_positions: bool = False) -> None:
            logits = output.logits[0, row_index]
            hidden = output.hidden_states[-1][0, row_index]
            summary, scores, state = row_summary(logits, hidden, candidate_ids)
            summary.update({"prefill_length": prefill_length,
                            "query_length": query_length,
                            "observed_row_index": row_index,
                            "global_token_position": prefill_length + row_index,
                            "explicit_positions_and_mask": explicit_positions,
                            "cache_length_after": (None if output.past_key_values is None
                                                   else int(output.past_key_values.get_seq_length()))})
            records[name] = summary
            raw_scores[name] = scores
            raw_hidden[name] = state

        def forward_full(name: str, ids: list[int], row_index: int) -> None:
            with torch.inference_mode(), target._lock:
                output = target.model(input_ids=target._ids(ids), use_cache=False,
                                      output_hidden_states=True, return_dict=True)
            record(name, output, row_index, prefill_length=0, query_length=len(ids))

        def forward_cached(name: str, prefill: list[int], chunk: list[int],
                           row_index: int, *, explicit: bool = False) -> None:
            with torch.inference_mode(), target._lock:
                pre = target.model(input_ids=target._ids(prefill), use_cache=True,
                                   return_dict=True)
                cache = pre.past_key_values
                if int(cache.get_seq_length()) != len(prefill):
                    raise RuntimeError("prefill cache length mismatch")
                kwargs = {}
                if explicit:
                    positions = torch.arange(len(prefill), len(prefill) + len(chunk),
                                             dtype=torch.long, device=device)
                    kwargs = {"attention_mask": torch.ones(
                        (1, len(prefill) + len(chunk)), dtype=torch.long, device=device),
                        "position_ids": positions.unsqueeze(0),
                        "cache_position": positions}
                output = target.model(input_ids=target._ids(chunk),
                                      past_key_values=cache, use_cache=True,
                                      output_hidden_states=True, return_dict=True,
                                      **kwargs)
            record(name, output, row_index, prefill_length=len(prefill),
                   query_length=len(chunk), explicit_positions=explicit)

        # Exact failure path and alternative query shapes, all in BF16.
        forward_full("full_prefix_exact", reference_prefix, len(reference_prefix) - 1)
        forward_full("full_prefix_plus_suffix", prefix + proposals,
                     len(reference_prefix) - 1)
        forward_cached("canonical_cached_one_token", reference_prefix[:-1],
                       reference_prefix[-1:], 0)
        forward_cached("cached_chunk_len3", prefix[:-1],
                       [prefix[-1], *proposals[:position]], position)
        forward_cached("cached_chunk_len9", prefix[:-1],
                       [prefix[-1], *proposals], position)
        forward_cached("cached_chunk_len9_repeat", prefix[:-1],
                       [prefix[-1], *proposals], position)
        forward_cached("cached_chunk_len9_explicit_positions", prefix[:-1],
                       [prefix[-1], *proposals], position, explicit=True)

        # Same-length suffix perturbations distinguish content leakage from
        # arithmetic differences caused solely by query length/shape.
        altered = proposals.copy()
        for idx in range(position, len(altered)):
            altered[idx] = (altered[idx] + 1 + idx) % target.vocab_size
        forward_cached("cached_chunk_len9_altered_future", prefix[:-1],
                       [prefix[-1], *altered], position)
        changed_all = [(value + 1 + idx) % target.vocab_size
                       for idx, value in enumerate(proposals)]
        forward_cached("cached_chunk_len9_altered_after_valid_row0", prefix[:-1],
                       [prefix[-1], *changed_all], 0)
        forward_cached("cached_chunk_len9_valid_row0", prefix[:-1],
                       [prefix[-1], *proposals], 0)

        fresh = HFPackedTargetSession(target, prefix).verify(proposals)
        fresh_hidden = fresh.hidden[position].detach().float().cpu().clone()
        records["fresh_session_packed_replay"] = {
            "argmax_id": fresh.argmax_ids[position],
            "all_argmax_ids": list(fresh.argmax_ids),
            "matches_recorded_failure_trace":
                list(fresh.argmax_ids) == evidence["packed_argmax_ids"],
            "cache_length_before": fresh.cache_length_before,
            "cache_length_raw": fresh.cache_length_raw,
        }
        raw_hidden["fresh_session_packed_replay"] = fresh_hidden

        comparisons = {}
        for left, right in (
            ("full_prefix_exact", "full_prefix_plus_suffix"),
            ("full_prefix_exact", "canonical_cached_one_token"),
            ("full_prefix_exact", "cached_chunk_len3"),
            ("cached_chunk_len3", "cached_chunk_len9"),
            ("cached_chunk_len9", "cached_chunk_len9_repeat"),
            ("cached_chunk_len9", "cached_chunk_len9_explicit_positions"),
            ("cached_chunk_len9", "cached_chunk_len9_altered_future"),
            ("cached_chunk_len9_valid_row0", "cached_chunk_len9_altered_after_valid_row0"),
        ):
            scores_left, scores_right = raw_scores[left], raw_scores[right]
            hidden_left, hidden_right = raw_hidden[left], raw_hidden[right]
            comparisons[f"{left}__vs__{right}"] = {
                "logits_bitwise_equal": bool(torch.equal(scores_left, scores_right)),
                "hidden_bitwise_equal": bool(torch.equal(hidden_left, hidden_right)),
                "max_abs_logit_diff_fp32_view": float((scores_left - scores_right).abs().max()),
                "max_abs_hidden_diff_fp32_view": float((hidden_left - hidden_right).abs().max()),
            }
        comparisons["cached_chunk_len9__vs__fresh_session_packed_replay"] = {
            "argmax_equal": (records["cached_chunk_len9"]["argmax_id"]
                             == records["fresh_session_packed_replay"]["argmax_id"]),
            "hidden_bitwise_equal": bool(torch.equal(
                raw_hidden["cached_chunk_len9"], raw_hidden["fresh_session_packed_replay"])),
            "max_abs_hidden_diff_fp32_view": float((
                raw_hidden["cached_chunk_len9"]
                - raw_hidden["fresh_session_packed_replay"]).abs().max()),
        }

        torch.cuda.synchronize(device)
        atomic_json(args.output / "report.json", {
            **provenance, "state": "diagnostic_complete_no_gate_claim",
            "failure_id": evidence["id"], "arm": evidence["arm"],
            "round_index": evidence["round_index"],
            "position": position, "prefix_length": len(prefix),
            "proposals": proposals, "recorded_packed_argmax_ids": evidence["packed_argmax_ids"],
            "reference_argmax": evidence["independent_target_argmax"],
            "recorded_packed_argmax": evidence["packed_argmax"],
            "first_rejection_position": next((idx for idx, token in enumerate(proposals)
                                              if token != evidence["packed_argmax_ids"][idx]),
                                             len(proposals)),
            "variants": records, "comparisons": comparisons,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "seconds": time.time() - started,
        })
    except BaseException as exc:
        atomic_json(args.output / "failure.json", {
            **provenance, "state": "diagnostic_failed_no_gate_claim",
            "error": repr(exc), "traceback": traceback.format_exc(),
            "seconds": time.time() - started,
        })
        raise


if __name__ == "__main__":
    main()
