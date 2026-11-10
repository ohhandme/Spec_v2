"""Real Qwen3-8B/0.6B BF16 correctness gate before any P1 training.

This full-prefix sequential reference is an acceptance/correctness check only.
It does not measure packed verification, WAN latency, or speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, FeedbackCodec, pack_codes, unpack_codes
from feedback_codec.decoding import SpeculativeDecoder, autoregressive_generate
from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
from feedback_codec.training import CodecEncoder


DRAFT_WEIGHT_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_file_hashes(root: Path) -> dict[str, str]:
    """Use the same pinned model-file set as fixed-probe collection."""
    names = {"config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "model.safetensors.index.json"}
    files = sorted(path for path in root.rglob("*") if path.is_file() and (
        path.name in names or path.suffix in (".safetensors", ".bin")
    ))
    if not any(path.suffix in (".safetensors", ".bin") for path in files):
        raise ValueError(f"model has no local weights: {root}")
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in files}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def load_smoke(dataset: Path, manifest: Path, count: int) -> tuple[list[dict], dict]:
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if sha256_file(dataset) != metadata["prompts_jsonl_sha256"]:
        raise ValueError("Locked prompt file SHA256 mismatch")
    if metadata["enable_thinking"] is not False or metadata["assistant_reference_excluded"] is not True:
        raise ValueError("Unexpected prompt formatting provenance")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    train = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["id"])
    if len(train) != metadata["counts"]["train"] or len(train) < count:
        raise ValueError("Locked train split count mismatch")
    return train[:count], metadata


def verify_tokenizer_provenance(
    target_path: Path, draft_path: Path, target_tokenizer, draft_tokenizer,
    dataset_meta: dict,
) -> dict:
    """Require both live tokenizers to match the tokenizer that encoded the prompts."""
    hashes = {}
    for role, path, tokenizer in (
        ("target", target_path, target_tokenizer),
        ("draft", draft_path, draft_tokenizer),
    ):
        for name, field in (
            ("tokenizer.json", "tokenizer_json_sha256"),
            ("tokenizer_config.json", "tokenizer_config_sha256"),
        ):
            actual = sha256_file(path / name)
            if actual != dataset_meta[field]:
                raise RuntimeError(f"{role} {name} differs from locked prompt tokenizer")
            hashes[f"{role}_{name}_sha256"] = actual
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise RuntimeError(f"{role} tokenizer has no single chat template")
        actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if actual != dataset_meta["chat_template_sha256"]:
            raise RuntimeError(f"{role} chat template differs from locked prompt template")
        hashes[f"{role}_chat_template_sha256"] = actual
    return hashes


def generation_signature(tokens: tuple[int, ...], eos_token_id: int,
                         max_new_tokens: int) -> tuple[tuple[int, ...], bool, bool]:
    ended_on_eos = bool(tokens) and tokens[-1] == eos_token_id
    return tokens, ended_on_eos, len(tokens) == max_new_tokens and not ended_on_eos


def verify_greedy_reproducibility(target, prompt: list[int], baseline: tuple[int, ...],
                                  max_new_tokens: int, eos_token_id: int) -> None:
    repeated = autoregressive_generate(target, prompt, max_new_tokens, eos_token_id)
    if generation_signature(repeated, eos_token_id, max_new_tokens) != generation_signature(
        baseline, eos_token_id, max_new_tokens
    ):
        raise RuntimeError("Independent target AR is not reproducible on the fixed request")


def gradient_calibration(target, draft, codec, record, eos_token_id) -> dict:
    if record.feedback_codes is None:
        raise ValueError("Calibration round has no next-window feedback")
    valid_prefix = list(record.prefix_before + record.proposals[:record.accepted_count])
    answer = record.correction_token if record.rejected_token is not None else record.bonus_token
    if answer is None or record.observation_hidden is None:
        raise ValueError("Calibration round has no valid verification state")
    corrected = valid_prefix + [answer]
    target_next = target.next(valid_prefix)
    if target_next.token_id != answer or not torch.equal(
        target_next.hidden, record.observation_hidden
    ):
        raise RuntimeError("Stored message state differs from effective-prefix target state")
    future = autoregressive_generate(target, corrected, 2, eos_token_id)
    if not future:
        raise RuntimeError("Calibration corrected prefix has no future target token")
    hidden = draft.teacher_hidden(corrected, list(future))[:1].detach()
    target_hidden = target_next.hidden.detach().clone().reshape(1, -1).to(torch.float32)
    rejected = (None if record.rejected_token is None else
                target.embedding(record.rejected_token).detach().clone().reshape(1, -1).to(torch.float32))
    correction = target.embedding(answer).detach().clone().reshape(1, -1).to(torch.float32)
    codec.zero_grad(set_to_none=True)
    codes = codec.sample_st(target_hidden, rejected, correction, record.accepted_count)
    logits = codec.repair_logits(hidden, codes, 1)
    loss = F.cross_entropy(logits.float(), torch.tensor([future[0]], device=logits.device))
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite real-model calibration loss")
    loss.backward()
    gradients = {}
    for name, parameter in codec.named_parameters():
        if name.startswith("output_head.") or not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is not None and not torch.isfinite(grad).all():
            raise FloatingPointError(f"Non-finite gradient: {name}")
        gradients[name] = float(grad.abs().sum().item()) if grad is not None else 0.0
    for key in ("target_projection.weight", "codebooks", "V.weight", "U.weight"):
        if gradients.get(key, 0.0) <= 0:
            raise RuntimeError(f"Missing required codec gradient: {key}")
    if any(parameter.grad is not None for parameter in target.model.parameters()):
        raise RuntimeError("Target base accumulated gradients")
    if any(parameter.grad is not None for parameter in draft.model.parameters()):
        raise RuntimeError("Draft base accumulated gradients")
    return {"loss": float(loss.item()), "gradient_l1": gradients,
            "target_frozen": True, "draft_frozen": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target-download-status", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        download = json.loads(args.target_download_status.read_text(encoding="utf-8"))
        if download.get("state") != "complete" or download.get("repo_id") != "Qwen/Qwen3-8B":
            raise RuntimeError("Pinned Qwen3-8B download has not passed SHA256 verification")
        if download.get("hf_revision") != "b968826d9c46dd6066d109eabc6255188de91218":
            raise RuntimeError("Qwen3-8B revision differs from pinned protocol")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        smoke, dataset_meta = load_smoke(args.dataset, args.dataset_manifest, args.request_count)
        target_files = {row["file"]: row["sha256"] for row in download["verified_files"]}
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "model.safetensors.index.json",
                     "model-00001-of-00005.safetensors",
                     "model-00002-of-00005.safetensors",
                     "model-00003-of-00005.safetensors",
                     "model-00004-of-00005.safetensors",
                     "model-00005-of-00005.safetensors"):
            if sha256_file(args.target / name) != target_files[name]:
                raise RuntimeError(f"Target model asset changed after verification: {name}")
        if sha256_file(args.draft / "model.safetensors") != DRAFT_WEIGHT_SHA256:
            raise RuntimeError("Draft model weight does not match pinned Qwen snapshot")
        target_model_hashes = model_file_hashes(args.target)
        draft_model_hashes = model_file_hashes(args.draft)
        dtype = torch.bfloat16
        device = torch.device(args.device)
        torch.manual_seed(20261030)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        target_tokenizer = AutoTokenizer.from_pretrained(
            str(args.target), local_files_only=True, trust_remote_code=False
        )
        draft_tokenizer = AutoTokenizer.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False
        )
        tokenizer_hashes = verify_tokenizer_provenance(
            args.target, args.draft, target_tokenizer, draft_tokenizer, dataset_meta
        )
        target = HFTargetBackend(args.target, target_tokenizer, device=device, dtype=dtype)
        draft_model = AutoModelForCausalLM.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False,
            attn_implementation="eager", torch_dtype=dtype,
        ).to(device)
        codec = FeedbackCodec(CodecConfig(
            target_hidden_size=target.hidden_size,
            draft_hidden_size=int(draft_model.config.hidden_size),
            vocab_size=target.vocab_size,
        ), draft_model.get_output_embeddings()).to(device)
        draft = HFDraftBackend(draft_model, draft_tokenizer, codec,
                               target=target, device=device, dtype=dtype)
        eos = target.tokenizer.eos_token_id
        if not isinstance(eos, int):
            raise RuntimeError("Target EOS must be a single token ID")
        if any(parameter.requires_grad for parameter in target.model.parameters()):
            raise RuntimeError("Target is not frozen")
        if any(parameter.requires_grad for parameter in draft.model.parameters()):
            raise RuntimeError("Draft is not frozen")
        no_feedback = SpeculativeDecoder(target, draft, None, 8, eos)
        with_feedback = SpeculativeDecoder(target, draft, CodecEncoder(codec), 8, eos)
        rows = []
        calibration_record = None
        for row in smoke:
            prompt = row["input_ids"]
            oracle = autoregressive_generate(target, list(prompt), args.max_new_tokens, eos)
            if not rows:
                verify_greedy_reproducibility(
                    target, list(prompt), oracle, args.max_new_tokens, eos
                )
            rows.append({
                "id": row["id"], "arm": "independent_target_ar",
                "output_ids": list(oracle), "eos": bool(oracle and oracle[-1] == eos),
                "truncated": len(oracle) == args.max_new_tokens and not (oracle and oracle[-1] == eos),
            })
            for arm, decoder in (("raw_draft", no_feedback), ("random_untrained_8B", with_feedback)):
                result = decoder.generate(list(prompt), args.max_new_tokens)
                if result.tokens != oracle:
                    raise RuntimeError(f"{arm} token mismatch on {row['id']}")
                if result.rounds and result.rounds[0].codes_in is not None:
                    raise RuntimeError("First round received a feedback message")
                for current, next_ in zip(result.rounds, result.rounds[1:]):
                    if next_.codes_in != current.feedback_codes:
                        raise RuntimeError("Feedback was not used exactly one round later")
                if arm == "random_untrained_8B" and calibration_record is None:
                    calibration_record = next((record for record in result.rounds
                                               if record.feedback_codes is not None), None)
                rows.append({
                    "id": row["id"], "arm": arm, "output_ids": list(result.tokens),
                    "exact_target_ar": True,
                    "eos": bool(result.tokens and result.tokens[-1] == eos),
                    "truncated": len(result.tokens) == args.max_new_tokens and not (result.tokens and result.tokens[-1] == eos),
                    "accepted_draft_tokens": result.accepted_draft_tokens,
                    "verification_rounds": result.verification_rounds,
                    "feedback_bytes": result.total_message_bytes,
                })
            atomic_json(args.output / "progress.json", {"rows": rows})
        if calibration_record is None:
            raise RuntimeError("No real-model feedback calibration boundary was produced")
        payload = pack_codes(calibration_record.feedback_codes)
        if len(payload) != 8 or tuple(unpack_codes(payload).tolist()) != calibration_record.feedback_codes:
            raise RuntimeError("Eight-byte feedback round-trip failed")
        gradients = gradient_calibration(target, draft, codec, calibration_record, eos)
        torch.cuda.synchronize(device)
        atomic_json(args.output / "report.json", {
            "state": "passed", "scope": "real_qwen3_bf16_full_prefix_p0_correctness_only",
            "latency_claim_valid": False,
            "target_path": str(args.target.resolve()),
            "draft_path": str(args.draft.resolve()),
            "target_revision": download["hf_revision"],
            "draft_weight_sha256": DRAFT_WEIGHT_SHA256,
            "target_file_sha256": target_model_hashes,
            "draft_file_sha256": draft_model_hashes,
            "dataset_sha256": dataset_meta["prompts_jsonl_sha256"],
            "tokenizer_provenance": tokenizer_hashes,
            "independent_target_ar_repeated_on": smoke[0]["id"],
            "device": str(device), "dtype": "bfloat16", "window": 8,
            "max_new_tokens": args.max_new_tokens,
            "request_count": args.request_count,
            "request_ids": [row["id"] for row in smoke],
            "checks": rows, "gradient_calibration": gradients,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "seconds": time.time() - started,
            "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path)
                              for path in sorted((ROOT / "feedback_codec").glob("*.py"))}
                            | {"scripts/run_real_p0.py": sha256_file(Path(__file__))},
        })
    except BaseException as exc:
        atomic_json(args.output / "failure.json", {
            "state": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
            "seconds": time.time() - started,
        })
        raise


if __name__ == "__main__":
    main()
