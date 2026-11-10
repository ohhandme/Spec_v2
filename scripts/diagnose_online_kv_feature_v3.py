"""Diagnose one frozen online-KV vs fresh-prefill packed feature failure.

This only replays the failed train prompt under the frozen BF16 v3 source. It
never declares a P0 pass, changes a Target decision, or substitutes an oracle
feature for the original packed feature.
"""

from __future__ import annotations

import argparse
import copy
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
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from feedback_codec.codec import CodecConfig, FeedbackCodec
    from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
    from feedback_codec.packed import HFPackedTargetSession

    frozen_script = source / "scripts/run_packed_p0.py"
    spec = importlib.util.spec_from_file_location("_frozen_packed_v3", frozen_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen v3 gate")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    provenance = {
        "scope": "one_train_prompt_online_kv_vs_fresh_feature_diagnostic",
        "state": "running", "p0_gate_claimed": False,
        "failure_sha256": sha256(args.failure),
        "source_zip_sha256": sha256(args.source_zip),
        "diagnostic_script_sha256": sha256(Path(__file__)),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": "bfloat16", "device": args.device,
        "latency_claim_valid": False,
    }
    try:
        if (provenance["failure_sha256"] != args.failure_sha256
                or provenance["source_zip_sha256"] != args.source_zip_sha256):
            raise RuntimeError("frozen v3 failure or source ZIP hash changed")
        failed = json.loads(args.failure.read_text(encoding="utf-8"))
        evidence = failed["evidence"]
        if (failed.get("state") != "failed"
                or "packed feature differs from same-query replay" not in failed["error"]
                or evidence.get("arm") != "raw_draft"
                or evidence.get("round_index") != 1):
            raise RuntimeError("this script only diagnoses the frozen raw-draft round-1 failure")
        if gate.source_hashes() != failed["source_sha256"]:
            raise RuntimeError("imported code does not match v3 failure source")
        if sha256(args.train_prompts) != failed["train_prompts_sha256"]:
            raise RuntimeError("frozen train-only prompt file changed")
        rows = [json.loads(line) for line in args.train_prompts.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if any(row.get("split") != "train" for row in rows):
            raise RuntimeError("a prompt is not from the train split")
        chosen = next((row for row in rows if row.get("id") == evidence["id"]), None)
        if chosen is None:
            raise RuntimeError("failed prompt ID absent from train-only input")
        prompt = chosen["input_ids"]
        prefix1 = evidence["prefix_before"]
        if (prefix1[:len(prompt)] != prompt or len(prefix1) <= len(prompt)
                or not 0 < len(evidence["proposals"]) <= 8):
            raise RuntimeError("failure prefix/proposals do not match prompt")
        model_hashes = gate.verify_models(args.target, args.draft,
                                          args.target_download_status)
        if model_hashes != failed["model_file_sha256"]:
            raise RuntimeError("model assets differ from v3 failure")
        provenance["model_file_sha256"] = model_hashes
        target_tokenizer = AutoTokenizer.from_pretrained(
            str(args.target), local_files_only=True, trust_remote_code=False)
        draft_tokenizer = AutoTokenizer.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False)
        manifest = json.loads(args.tokenizer_manifest.read_text(encoding="utf-8"))
        provenance["tokenizer_manifest_sha256"] = sha256(args.tokenizer_manifest)
        token_hashes = gate.verify_tokenizers(args.target, args.draft,
                                              target_tokenizer, draft_tokenizer, manifest)
        if token_hashes != failed["tokenizer_sha256"]:
            raise RuntimeError("tokenizer assets differ from v3 failure")
        provenance["gpu_admission"] = gate.admit_gpu(args.device)

        device = torch.device(args.device)
        torch.manual_seed(int(failed["seed"]))
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        target = HFTargetBackend(args.target, target_tokenizer,
                                 device=device, dtype=torch.bfloat16)
        draft_model = AutoModelForCausalLM.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False,
            attn_implementation="eager", torch_dtype=torch.bfloat16).to(device)
        gate.verify_draft_metadata(args.draft, draft_model, draft_tokenizer,
                                   model_hashes)
        codec = FeedbackCodec(CodecConfig(
            target_hidden_size=target.hidden_size,
            draft_hidden_size=int(draft_model.config.hidden_size),
            vocab_size=target.vocab_size), draft_model.get_output_embeddings()).to(device)
        draft = HFDraftBackend(draft_model, draft_tokenizer, codec,
                               target=target, device=device, dtype=torch.bfloat16)
        eos = target.tokenizer.eos_token_id
        session = HFPackedTargetSession(target, prompt)

        def proposal_for(prefix: list[int], budget: int) -> list[int]:
            proposals = []
            for position in range(1, min(8, budget) + 1):
                token = int(draft.next(prefix + proposals, None, position))
                proposals.append(token)
                if token == eos:
                    break
            return proposals

        first_proposals = proposal_for(prompt, 24)
        first = session.verify(first_proposals)
        first_accepted = 0
        while (first_accepted < len(first_proposals)
               and first_proposals[first_accepted] == first.argmax_ids[first_accepted]):
            first_accepted += 1
        first_rejected = first_accepted < len(first_proposals)
        first_answer = (first.argmax_ids[first_accepted] if first_rejected
                        else first.argmax_ids[-1]
                        if first_proposals[-1] != eos and first_accepted < 24 else None)
        first_committed = first_proposals[:first_accepted]
        if first_answer is not None:
            first_committed.append(first_answer)
        session.commit(first, first_accepted, first_answer)
        if (session.confirmed != prefix1
                or session.confirmed != prompt + first_committed):
            raise RuntimeError("reconstructed round0 did not reach the failed online prefix")

        proposals = proposal_for(prefix1, 24 - len(first_committed))
        if proposals != evidence["proposals"]:
            raise RuntimeError("reconstructed round1 draft proposals differ from v3 failure")
        online_cache_before = copy.deepcopy(session._cache)
        online_cache_length = session.cache_length
        online = session.verify(proposals)
        if list(online.argmax_ids) != evidence["packed_argmax_ids"]:
            raise RuntimeError("online round1 argmax trace is not reproducible")
        accepted = 0
        while accepted < len(proposals) and proposals[accepted] == online.argmax_ids[accepted]:
            accepted += 1
        if accepted != 5:
            raise RuntimeError("frozen v3 accepted count was not reproduced")

        fresh_session = HFPackedTargetSession(target, prefix1)
        fresh_cache_before = copy.deepcopy(fresh_session._cache)
        fresh = fresh_session.verify(proposals)

        def from_same_online_cache(values: list[int]):
            replay = object.__new__(HFPackedTargetSession)
            replay.target = target
            replay.confirmed = list(prefix1)
            replay._cache = copy.deepcopy(online_cache_before)
            replay._pending = None
            return replay.verify(values)

        clone = from_same_online_cache(proposals)
        altered = proposals.copy()
        altered[accepted] = (altered[accepted] + 1) % target.vocab_size
        perturbed = from_same_online_cache(altered)

        def summary(verification, label: str) -> dict:
            logits = verification.logits_at_valid_prefix(accepted).detach().float().cpu()
            hidden = verification.hidden_at_valid_prefix(accepted).detach().float().cpu()
            values, ids = logits.topk(5)
            return {
                "label": label,
                "argmax_ids": list(verification.argmax_ids),
                "valid_argmax": verification.argmax_ids[accepted],
                "valid_top5": [{"id": int(token), "value_fp32_view": float(value)}
                               for token, value in zip(ids.tolist(), values.tolist())],
                "valid_hidden_l2_fp32_view": float(torch.linalg.vector_norm(hidden)),
                "cache_length_before": verification.cache_length_before,
                "cache_length_raw": verification.cache_length_raw,
            }

        def compare(left, right) -> dict:
            left_logits = left.logits_at_valid_prefix(accepted).detach().float()
            right_logits = right.logits_at_valid_prefix(accepted).detach().float()
            left_hidden = left.hidden_at_valid_prefix(accepted).detach().float()
            right_hidden = right.hidden_at_valid_prefix(accepted).detach().float()
            return {
                "valid_argmax_equal": left.argmax_ids[accepted] == right.argmax_ids[accepted],
                "valid_logits_bitwise_equal": bool(torch.equal(left_logits, right_logits)),
                "valid_hidden_bitwise_equal": bool(torch.equal(left_hidden, right_hidden)),
                "valid_logits_max_abs_diff_fp32_view": float((left_logits-right_logits).abs().max()),
                "valid_hidden_max_abs_diff_fp32_view": float((left_hidden-right_hidden).abs().max()),
            }

        cache_compare = {"available": False}
        online_layers = getattr(online_cache_before, "layers", None)
        fresh_layers = getattr(fresh_cache_before, "layers", None)
        if (isinstance(online_layers, (tuple, list))
                and isinstance(fresh_layers, (tuple, list))
                and len(online_layers) == len(fresh_layers)):
            per_layer = []
            for idx, (left, right) in enumerate(zip(online_layers, fresh_layers)):
                left_k, right_k = getattr(left, "keys", None), getattr(right, "keys", None)
                left_v, right_v = getattr(left, "values", None), getattr(right, "values", None)
                if not all(isinstance(t, torch.Tensor)
                           for t in (left_k, right_k, left_v, right_v)):
                    break
                per_layer.append({
                    "layer": idx,
                    "keys_bitwise_equal": bool(torch.equal(left_k, right_k)),
                    "values_bitwise_equal": bool(torch.equal(left_v, right_v)),
                    "keys_max_abs_diff_fp32_view": float((left_k.float()-right_k.float()).abs().max()),
                    "values_max_abs_diff_fp32_view": float((left_v.float()-right_v.float()).abs().max()),
                })
            if len(per_layer) == len(online_layers):
                cache_compare = {"available": True, "per_layer": per_layer,
                                 "all_keys_bitwise_equal": all(x["keys_bitwise_equal"] for x in per_layer),
                                 "all_values_bitwise_equal": all(x["values_bitwise_equal"] for x in per_layer)}

        torch.cuda.synchronize(device)
        atomic_json(args.output / "report.json", {
            **provenance, "state": "diagnostic_complete_no_gate_claim",
            "id": evidence["id"], "round_index": 1,
            "prompt_length": len(prompt), "prefix_length": len(prefix1),
            "round0": {"proposals": first_proposals, "accepted": first_accepted,
                       "committed": first_committed,
                       "cache_length_after": online_cache_length},
            "round1": {"proposals": proposals, "accepted": accepted,
                       "valid_row": accepted, "altered_proposals": altered},
            "variants": {
                "original_online": summary(online, "original_online"),
                "cloned_same_online_cache": summary(clone, "cloned_same_online_cache"),
                "fresh_prefix_prefill": summary(fresh, "fresh_prefix_prefill"),
                "same_online_cache_altered_unconfirmed_suffix": summary(
                    perturbed, "same_online_cache_altered_unconfirmed_suffix"),
            },
            "comparisons": {
                "original_online_vs_clone_same_state": compare(online, clone),
                "original_online_vs_fresh_prefill": compare(online, fresh),
                "original_online_vs_altered_suffix_same_state": compare(online, perturbed),
            },
            "online_vs_fresh_preverification_cache": cache_compare,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "seconds": time.time()-started,
        })
    except BaseException as exc:
        atomic_json(args.output / "failure.json", {
            **provenance, "state": "diagnostic_failed_no_gate_claim",
            "error": repr(exc), "traceback": traceback.format_exc(),
            "seconds": time.time()-started,
        })
        raise


if __name__ == "__main__":
    main()
