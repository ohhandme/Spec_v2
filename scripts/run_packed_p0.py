"""Real BF16 packed/KV committed-path gate against independent Target AR.

The input JSONL must contain *train* prompts only.  Version 2 strictly checks
every Target decision that can enter the output: accepted candidates, the first
rejection's correction, and an actually used all-accepted bonus.  Target rows
after the first rejection, or an unused terminal bonus row, are recorded as
diagnostics without gating.  Final tokens, EOS, truncation and KV crop remain
strict gates.  Passing this gate does not establish a latency gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, FeedbackCodec
from feedback_codec.decoding import (
    SpeculativeDecoder, TargetLogitSummary, autoregressive_generate,
)
from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
from feedback_codec.packed import (
    PACKED_FEATURE_PROTOCOL_VERSION, HFPackedTargetSession, PackedRun,
    PackedSpeculativeDecoder,
)
from feedback_codec.training import CodecEncoder


TARGET_REPO = "Qwen/Qwen3-8B"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_WEIGHT_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
REQUIRED_COVERAGE = (
    "zero_accept", "partial_accept", "full_accept", "bonus",
    "eos", "budget_truncation",
)
PACKED_TOKEN_GATE_VERSION = "packed_committed_path_v2"


class GateFailure(RuntimeError):
    def __init__(self, message: str, evidence: dict) -> None:
        super().__init__(message)
        self.evidence = evidence


def _nvidia_smi(*arguments: str) -> list[str]:
    result = subprocess.run(["nvidia-smi", *arguments], capture_output=True,
                            text=True, timeout=20, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()[:500]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def admit_gpu(device: str, *, pause_seconds: float = 2.0) -> dict:
    """Map logical CUDA ordinal to a physical card and inspect it three times.

    `CUDA_VISIBLE_DEVICES` must list physical numeric indexes explicitly. UUID,
    MIG, and implicit mappings are rejected until separately audited. No CUDA
    API is invoked here; callers run this after CPU provenance and immediately
    before their first CUDA allocation.
    """
    match = re.fullmatch(r"cuda:(\d+)", device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    parts = [part.strip() for part in visible.split(",")]
    if (match is None or not visible or any(not re.fullmatch(r"\d+", part) for part in parts)
            or len(set(parts)) != len(parts) or int(match.group(1)) >= len(parts)):
        raise GateFailure("ambiguous CUDA-to-physical-GPU mapping", {
            "device": device, "CUDA_VISIBLE_DEVICES": visible,
            "required": "numeric physical indexes and explicit cuda:<logical ordinal>"})
    physical = int(parts[int(match.group(1))])
    evidence = {"device": device, "CUDA_VISIBLE_DEVICES": visible,
                "physical_gpu_index": physical, "samples": []}
    for sample_index in range(3):
        gpu_lines = _nvidia_smi(
            "--query-gpu=index,uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits")
        gpus = {}
        for line in gpu_lines:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                raise GateFailure("unparseable nvidia-smi GPU row", {**evidence, "row": line})
            try:
                index = int(fields[0])
                gpus[index] = {"uuid": fields[1], "free_mib": int(fields[2]),
                               "util_percent": int(fields[3])}
            except ValueError as exc:
                raise GateFailure("unparseable nvidia-smi GPU metric", {
                    **evidence, "row": line}) from exc
        if physical not in gpus:
            raise GateFailure("visible physical GPU absent from nvidia-smi", evidence)
        compute_lines = _nvidia_smi(
            "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits")
        pids = []
        for line in compute_lines:
            if line == "No running processes found":
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or not fields[1].isdigit():
                raise GateFailure("unparseable nvidia-smi compute-process row", {
                    **evidence, "row": line})
            if fields[0] == gpus[physical]["uuid"]:
                pids.append(int(fields[1]))
        sample = {"sample": sample_index + 1, **gpus[physical],
                  "compute_pids": sorted(set(pids)), "utc_epoch": time.time()}
        evidence["samples"].append(sample)
        if sample["compute_pids"] or sample["free_mib"] < 22000 or sample["util_percent"] >= 10:
            raise GateFailure("selected physical GPU is not exclusively idle", evidence)
        if sample_index < 2:
            time.sleep(pause_seconds)
    return evidence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "feedback_codec").glob("*.py")) + [Path(__file__)]
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def load_train_only(path: Path, expected_sha: str, count: int) -> list[dict]:
    if count < 1:
        raise ValueError("request count must be positive")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError("train-only prompt file SHA256 differs from the frozen manifest")
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("packed P0 accepts a train-only JSONL, never val/test")
            prompt_id, ids = row.get("id"), row.get("input_ids")
            if not isinstance(prompt_id, str) or prompt_id in seen:
                raise ValueError("train-only P0 has a missing or duplicate prompt ID")
            if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
                raise ValueError(f"invalid tokenized train prompt: {prompt_id}")
            seen.add(prompt_id)
            rows.append({"id": prompt_id, "input_ids": ids})
    if len(rows) < count:
        raise ValueError("too few frozen train-only prompts for the requested P0")
    return sorted(rows, key=lambda row: row["id"])[:count]


def verify_tokenizers(target_path: Path, draft_path: Path,
                      target_tokenizer, draft_tokenizer, manifest: dict) -> dict:
    hashes = {}
    for role, path, tokenizer in (("target", target_path, target_tokenizer),
                                   ("draft", draft_path, draft_tokenizer)):
        for name, field in (("tokenizer.json", "tokenizer_json_sha256"),
                            ("tokenizer_config.json", "tokenizer_config_sha256")):
            actual = sha256_file(path / name)
            if actual != manifest[field]:
                raise RuntimeError(f"{role} {name} differs from frozen prompt tokenizer")
            hashes[f"{role}_{name}_sha256"] = actual
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise RuntimeError(f"{role} tokenizer lacks a fixed chat template")
        actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if actual != manifest["chat_template_sha256"]:
            raise RuntimeError(f"{role} chat template differs from frozen prompt formatting")
        hashes[f"{role}_chat_template_sha256"] = actual
    return hashes


def verify_models(target_path: Path, draft_path: Path, download_status: Path) -> dict:
    download = json.loads(download_status.read_text(encoding="utf-8"))
    if (download.get("state") != "complete" or download.get("repo_id") != TARGET_REPO
            or download.get("hf_revision") != TARGET_REVISION):
        raise RuntimeError("official Qwen3-8B download/revision is not verified")
    hashes = {}
    for row in download["verified_files"]:
        name = row["file"]
        actual = sha256_file(target_path / name)
        if actual != row["sha256"]:
            raise RuntimeError(f"Target asset changed after official SHA verification: {name}")
        hashes[f"target/{name}"] = actual
    draft_weight = sha256_file(draft_path / "model.safetensors")
    if draft_weight != DRAFT_WEIGHT_SHA256:
        raise RuntimeError("Draft asset differs from pinned Qwen3-0.6B weight")
    hashes["draft/model.safetensors"] = draft_weight
    hashes["draft/config.json"] = sha256_file(draft_path / "config.json")
    generation_config = draft_path / "generation_config.json"
    if generation_config.is_file():
        hashes["draft/generation_config.json"] = sha256_file(generation_config)
    return hashes


def verify_draft_metadata(draft_path: Path, draft_model, draft_tokenizer,
                          recorded_hashes: dict) -> dict:
    """Tie the hashed disk config to the loaded Qwen3 draft and tokenizer.

    The tokenizer can contain fewer IDs than a padded model vocabulary; the
    actual invariant is that every tokenizer ID and special ID is representable
    by the loaded embedding/output tables, not equality of the two lengths.
    """
    config_path = draft_path / "config.json"
    if sha256_file(config_path) != recorded_hashes["draft/config.json"]:
        raise RuntimeError("Draft config changed between SHA verification and model load")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Draft config.json must be an object")
    fields = ("model_type", "vocab_size", "hidden_size")
    for field in fields:
        disk = config.get(field)
        loaded = getattr(draft_model.config, field, None)
        if disk != loaded:
            raise RuntimeError(f"Draft loaded config differs from hashed config.json: {field}")
    if config["model_type"] != "qwen3":
        raise RuntimeError("Draft model_type is not the pinned Qwen3 architecture")
    vocab, hidden = config["vocab_size"], config["hidden_size"]
    if (type(vocab) is not int or vocab < 2
            or type(hidden) is not int or hidden < 1):
        raise ValueError("Draft config has invalid vocab_size or hidden_size")
    input_weight = draft_model.get_input_embeddings().weight
    output_weight = draft_model.get_output_embeddings().weight
    if (tuple(input_weight.shape) != (vocab, hidden)
            or tuple(output_weight.shape) != (vocab, hidden)):
        raise RuntimeError("Draft loaded embeddings/head differ from hashed config dimensions")
    token_map = draft_tokenizer.get_vocab()
    if (not token_map or len(draft_tokenizer) > vocab
            or any(type(token) is not int or not 0 <= token < vocab
                   for token in token_map.values())):
        raise RuntimeError("Draft tokenizer IDs exceed the loaded model vocabulary")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token = getattr(draft_tokenizer, name, None)
        if token is not None and (type(token) is not int or not 0 <= token < vocab):
            raise RuntimeError(f"Draft tokenizer {name} exceeds the loaded vocabulary")
    generation_path = draft_path / "generation_config.json"
    if generation_path.is_file():
        if sha256_file(generation_path) != recorded_hashes.get("draft/generation_config.json"):
            raise RuntimeError("Draft generation_config changed after SHA verification")
    elif "draft/generation_config.json" in recorded_hashes:
        raise RuntimeError("Draft generation_config disappeared after SHA verification")
    return {"model_type": config["model_type"], "vocab_size": vocab,
            "hidden_size": hidden, "tokenizer_length": len(draft_tokenizer),
            "tokenizer_max_id": max(token_map.values()),
            "generation_config_present": generation_path.is_file()}


def generation_signature(tokens: tuple[int, ...], eos: int, budget: int) -> dict:
    ended = bool(tokens) and tokens[-1] == eos
    return {"output_ids": list(tokens), "eos": ended,
            "truncated": len(tokens) == budget and not ended}


def _reference_top2(target: HFTargetBackend, prefix: list[int]) -> list[dict]:
    """Failure evidence only; the independent Target decision remains next()."""
    with torch.inference_mode(), target._lock:
        output = target.model(input_ids=target._ids(prefix), use_cache=False,
                              return_dict=True)
        values, ids = torch.topk(output.logits[0, -1].float(), 2)
    return [{"token_id": int(token), "logit_fp32_view": float(value)}
            for token, value in zip(ids.tolist(), values.tolist())]


def check_rounds(target: HFTargetBackend, packed: PackedRun, oracle: tuple[int, ...],
                 prompt: list[int], budget: int, eos: int, arm: str, prompt_id: str,
                 feature_provenance: bool = False) -> list[dict]:
    generation = packed.generation
    if generation_signature(generation.tokens, eos, budget) != generation_signature(oracle, eos, budget):
        raise GateFailure("packed output/EOS/truncation differs from independent Target AR", {
            "id": prompt_id, "arm": arm, "oracle": generation_signature(oracle, eos, budget),
            "packed": generation_signature(generation.tokens, eos, budget)})
    if len(generation.rounds) != len(packed.traces):
        raise GateFailure("round/trace count mismatch", {"id": prompt_id, "arm": arm})
    expected_prefix = tuple(prompt)
    previous_codes = None
    checks = []
    for round_index, (record, trace) in enumerate(zip(generation.rounds, packed.traces)):
        evidence = {"id": prompt_id, "arm": arm, "round_index": round_index,
                    "prefix_before": list(expected_prefix), "proposals": list(trace.proposals),
                    "packed_argmax_ids": list(trace.argmax_ids),
                    "packed_top2_margins": list(trace.top2_margins)}
        if record.prefix_before != expected_prefix or trace.prefix_before != expected_prefix:
            raise GateFailure("round starts at the wrong confirmed prefix", evidence)
        if record.proposals != trace.proposals or record.codes_in != previous_codes:
            raise GateFailure("proposal or delayed-message alignment differs", evidence)
        if (len(trace.argmax_ids) != len(trace.proposals) + 1
                or len(trace.top2_margins) != len(trace.argmax_ids)):
            raise GateFailure("packed forward did not return proposal+bonus positions", evidence)
        if (trace.cache_length_before != len(expected_prefix) - 1
                or trace.cache_length_raw != len(expected_prefix) + len(trace.proposals)):
            raise GateFailure("KV prefill/raw length does not equal token prefix", evidence)
        accepted = 0
        while accepted < len(trace.proposals) and trace.proposals[accepted] == trace.argmax_ids[accepted]:
            accepted += 1
        rejected = trace.proposals[accepted] if accepted < len(trace.proposals) else None
        correction = trace.argmax_ids[accepted] if rejected is not None else None
        remaining = budget - (len(expected_prefix) - len(prompt))
        draft_eos = accepted == len(trace.proposals) and trace.proposals[-1] == eos
        bonus = (trace.argmax_ids[-1] if rejected is None and not draft_eos
                 and accepted < remaining else None)
        answer = correction if correction is not None else bonus
        # A rejected suffix is hypothetical. Likewise the final row is unused
        # when the entire draft ends on EOS or exhausts the output budget.
        last_used_position = accepted if answer is not None else accepted - 1
        reference_valid_hidden = None
        used_positions = []
        hypothetical_positions = []
        for position, packed_id in enumerate(trace.argmax_ids):
            reference_prefix = list(expected_prefix + trace.proposals[:position])
            independent = target.next(reference_prefix)
            if position == accepted:
                reference_valid_hidden = independent.hidden.detach().clone()
            position_check = {
                "position": position, "packed_argmax": packed_id,
                "independent_target_argmax": independent.token_id,
                "argmax_equal": packed_id == independent.token_id,
                "packed_top2_margin": trace.top2_margins[position],
            }
            if position <= last_used_position:
                used_positions.append(position_check)
            else:
                if packed_id != independent.token_id:
                    position_check["reference_top2"] = _reference_top2(
                        target, reference_prefix)
                hypothetical_positions.append(position_check)
            if position <= last_used_position and packed_id != independent.token_id:
                evidence.update({"position": position, "reference_prefix": reference_prefix,
                                 "independent_target_argmax": independent.token_id,
                                 "packed_argmax": packed_id,
                                 "last_used_position": last_used_position,
                                 "reference_top2": _reference_top2(target, reference_prefix)})
                raise GateFailure("packed Target argmax differs at a used verification position", evidence)
        if reference_valid_hidden is None:
            raise GateFailure("valid-prefix Target hidden was not checked", evidence)
        packed_valid_hidden = record.observation_hidden
        hidden_source = ("original_packed_round" if packed_valid_hidden is not None
                         else "terminal_no_outgoing_feature")
        hidden_bitwise_equal = None
        hidden_maxdiff = None
        if packed_valid_hidden is not None:
            if (packed_valid_hidden.shape != reference_valid_hidden.shape
                    or not torch.isfinite(packed_valid_hidden).all()
                    or not torch.isfinite(reference_valid_hidden).all()):
                raise GateFailure("valid-prefix hidden shape/nonfinite mismatch", evidence)
            hidden_bitwise_equal = bool(torch.equal(packed_valid_hidden,
                                                    reference_valid_hidden))
            hidden_maxdiff = float((packed_valid_hidden.float()
                                    - reference_valid_hidden.float()).abs().max().item())
        committed = trace.proposals[:accepted] + (() if correction is None and bonus is None
                                                    else (correction if correction is not None else bonus,))
        expected_after = expected_prefix + committed
        if (record.accepted_count != accepted or trace.accepted_count != accepted
                or record.rejected_token != rejected or record.correction_token != correction
                or record.bonus_token != bonus or record.committed_tokens != committed
                or trace.valid_prefix != expected_prefix + trace.proposals[:accepted]
                or trace.cache_length_after != len(expected_after) - 1):
            evidence.update({"expected_accept": accepted, "expected_committed": list(committed),
                             "record_committed": list(record.committed_tokens),
                             "cache_length_after": trace.cache_length_after,
                             "expected_cache_length_after": len(expected_after) - 1})
            raise GateFailure("accept/bonus/EOS/KV commit differs from independent round rules", evidence)
        ended = bool(committed) and committed[-1] == eos
        if ended or len(expected_after) - len(prompt) == budget:
            if record.feedback_codes is not None:
                raise GateFailure("terminal round emitted a next-window message", evidence)
        elif arm == "raw_draft" or arm.startswith("controlled_"):
            if record.feedback_codes is not None:
                raise GateFailure("raw arm emitted a feedback message", evidence)
        elif record.feedback_codes is None or len(record.feedback_codes) != 8:
            raise GateFailure("nonterminal feedback arm did not emit eight bytes", evidence)
        feature_checks = None
        if feature_provenance:
            online = trace.feature_provenance
            if (not isinstance(online, dict)
                    or online.get("feature_protocol_version") != PACKED_FEATURE_PROTOCOL_VERSION
                    or online.get("cache_source") != "same_online_preverify_KV_snapshot"
                    or online.get("accepted_count") != accepted
                    or online.get("cache_length_before") != trace.cache_length_before
                    or online.get("decision_feature_available") != (answer is not None)):
                raise GateFailure("packed online feature provenance is missing or misaligned",
                                  {**evidence, "online": online})
            if (online.get("snapshot_replay_hidden_bitwise_equal") is not True
                    or online.get("snapshot_replay_logits_bitwise_equal") is not True
                    or online.get("snapshot_replay_greedy_token_equal") is not True):
                raise GateFailure("packed online KV snapshot did not reproduce the original forward",
                                  {**evidence, "online": online})
            summary = record.observation_logit_summary
            if answer is not None:
                if (packed_valid_hidden is None or not isinstance(summary, TargetLogitSummary)
                        or TargetLogitSummary.from_json(summary.to_json()) != summary
                        or summary.greedy_token_id != answer
                        or online.get("same_forward_hidden") is not True
                        or online.get("same_forward_logit_summary") is not True):
                    raise GateFailure("packed valid feature lacks original-forward provenance",
                                      {**evidence, "online": online})
            elif (summary is not None or online.get("same_forward_hidden") is not None
                  or online.get("same_forward_logit_summary") is not None):
                raise GateFailure("terminal packed round exported an unused logit summary", evidence)
            if online.get("applicable"):
                if (accepted >= len(trace.proposals) or online.get("passed") is not True
                        or online.get("same_query_length") is not True
                        or online.get("hidden_bitwise_equal") is not True
                        or online.get("logits_bitwise_equal") is not True
                        or online.get("greedy_token_equal") is not True):
                    raise GateFailure("packed valid feature depends on unconfirmed candidates",
                                      {**evidence, "online": online})
            elif (accepted != len(trace.proposals) or online.get("passed") is not True
                  or online.get("reason") != "no_unconfirmed_suffix_in_original_query"):
                raise GateFailure("packed feature causality was not checked at a rejection",
                                  {**evidence, "online": online})
            # A fresh full-prefix prefill has different BF16 KV history from
            # the live session. Preserve comparison evidence but never gate on it.
            fresh_prefill = {}
            try:
                replay = HFPackedTargetSession(target, list(expected_prefix)).verify(
                    list(trace.proposals))
                fresh_prefill["used_argmax_equal"] = (
                    replay.argmax_ids[:last_used_position + 1]
                    == trace.argmax_ids[:last_used_position + 1])
                if packed_valid_hidden is not None:
                    replay_hidden = replay.hidden_at_valid_prefix(accepted)
                    fresh_prefill["valid_hidden_bitwise_equal_to_online"] = torch.equal(
                        replay_hidden, packed_valid_hidden)
                    fresh_prefill["valid_hidden_max_abs_diff_fp32"] = float((
                        replay_hidden.float() - packed_valid_hidden.float()).abs().max().item())
                if summary is not None:
                    fresh_prefill["valid_logit_summary_equal_to_online"] = (
                        replay.logit_summary_at_valid_prefix(accepted) == summary)
            except Exception as exc:
                fresh_prefill["diagnostic_error"] = repr(exc)
            feature_checks = {
                "feature_protocol_version": PACKED_FEATURE_PROTOCOL_VERSION,
                "hidden_source": hidden_source,
                "logit_source": "original_packed_verify_forward" if summary is not None else None,
                "online_cache_causality": online,
                "fresh_prefill_diagnostic_only": fresh_prefill,
            }
        checks.append({**evidence, "accepted_count": accepted,
                       "token_gate_version": PACKED_TOKEN_GATE_VERSION,
                       "last_used_position": last_used_position,
                       "used_positions": used_positions,
                       "hypothetical_positions": hypothetical_positions,
                       "committed_tokens": list(committed),
                       "rejected_token": rejected, "correction_token": correction,
                       "bonus_token": bonus, "proposal_count": len(trace.proposals),
                       "cache_length_before": trace.cache_length_before,
                       "cache_length_raw": trace.cache_length_raw,
                       "cache_length_after": trace.cache_length_after,
                       "valid_hidden_source": hidden_source,
                       "valid_hidden_bitwise_equal_to_full_prefix": hidden_bitwise_equal,
                       "valid_hidden_max_abs_diff_fp32": hidden_maxdiff,
                       "eos": ended,
                       **({"feature_provenance": feature_checks}
                          if feature_checks is not None else {})})
        expected_prefix = expected_after
        previous_codes = record.feedback_codes
    if expected_prefix != tuple(prompt) + generation.tokens:
        raise GateFailure("round commits do not reconstruct generated sequence",
                          {"id": prompt_id, "arm": arm})
    return checks


class _ControlledProposalDraft:
    """P0-only proposal fixture; it never decides which Target token commits."""

    def __init__(self, proposals: tuple[int, ...]) -> None:
        self.proposals = proposals

    def next(self, _prefix: list[int], _codes: tuple[int, ...] | None,
             position: int) -> int:
        return self.proposals[position - 1]


def controlled_coverage_probes(target: HFTargetBackend, prompts: list[dict],
                               natural_oracles: dict[str, tuple[int, ...]], eos: int,
                               eos_search_limit: int,
                               feature_provenance: bool = False) -> list[dict]:
    """Exercise real BF16 verifier branches with explicitly oracle-built *drafts*.

    This is correctness coverage, never acceptance-rate data. The Target's
    packed answer is still computed independently and must match full-prefix
    AR at every packed query position.
    """
    cases: list[tuple[str, str, list[int], tuple[int, ...], int]] = []
    for row in prompts:
        oracle = natural_oracles[row["id"]]
        if oracle and oracle[0] != eos:
            wrong = (oracle[0] + 1) % target.vocab_size
            cases.append(("controlled_zero", row["id"], list(row["input_ids"]),
                          (wrong,), 1))
            break
    for row in prompts:
        oracle = natural_oracles[row["id"]]
        if len(oracle) >= 2 and oracle[0] != eos:
            wrong = (oracle[1] + 1) % target.vocab_size
            cases.append(("controlled_partial", row["id"], list(row["input_ids"]),
                          (oracle[0], wrong), 2))
            break
    for row in prompts:
        oracle = natural_oracles[row["id"]]
        if len(oracle) >= 3 and eos not in oracle[:2]:
            cases.append(("controlled_full_bonus", row["id"],
                          list(row["input_ids"]), tuple(oracle[:2]), 3))
            break
    eos_case = None
    for row in prompts:
        oracle = natural_oracles[row["id"]]
        if eos not in oracle and eos_search_limit > len(oracle):
            oracle = autoregressive_generate(target, list(row["input_ids"]),
                                             eos_search_limit, eos)
        if eos in oracle:
            position = oracle.index(eos)
            eos_case = ("controlled_accepted_eos", row["id"],
                        list(row["input_ids"]) + list(oracle[:position]), (eos,), 1)
            break
    if eos_case is not None:
        cases.append(eos_case)

    rows = []
    for name, prompt_id, prefix, proposals, budget in cases:
        oracle = autoregressive_generate(target, prefix, budget, eos)
        sequential = SpeculativeDecoder(
            target, _ControlledProposalDraft(proposals), None,
            len(proposals), eos).generate(prefix, budget)
        packed = PackedSpeculativeDecoder(
            target, _ControlledProposalDraft(proposals), None,
            len(proposals), eos,
            capture_target_logits=feature_provenance,
            feature_provenance_gate=feature_provenance).generate(prefix, budget)
        if sequential.tokens != oracle:
            raise GateFailure("controlled sequential path differs from Target AR", {
                "id": prompt_id, "case": name, "prefix": prefix})
        checks = check_rounds(target, packed, oracle, prefix, budget, eos, name, prompt_id,
                              feature_provenance=feature_provenance)
        if len(checks) != 1 or packed.generation.rounds[0].proposals != proposals:
            raise GateFailure("controlled probe did not execute its one packed round", {
                "id": prompt_id, "case": name, "prefix": prefix})
        rows.append({"id": prompt_id, "arm": name, "coverage_probe_only": True,
                     "draft_proposal_source": "independent_target_ar_for_branch_coverage",
                     "prefix_ids": prefix, "budget": budget,
                     **generation_signature(packed.generation.tokens, eos, budget),
                     "round_checks": checks})
    return rows


def coverage_summary(natural_rows: list[dict], controlled_rows: list[dict]) -> dict:
    evidence = {name: [] for name in REQUIRED_COVERAGE}
    for row in [*natural_rows, *controlled_rows]:
        location = f"{row['id']}:{row['arm']}"
        if row["eos"]:
            evidence["eos"].append(location)
        if row["truncated"]:
            evidence["budget_truncation"].append(location)
        for index, check in enumerate(row["round_checks"]):
            at = f"{location}:round{index}"
            accepted, proposed = check["accepted_count"], check["proposal_count"]
            if accepted == 0:
                evidence["zero_accept"].append(at)
            elif accepted == proposed:
                evidence["full_accept"].append(at)
            else:
                evidence["partial_accept"].append(at)
            if check["bonus_token"] is not None:
                evidence["bonus"].append(at)
    return {"required": list(REQUIRED_COVERAGE),
            "observed": evidence,
            "missing": [name for name in REQUIRED_COVERAGE if not evidence[name]],
            "controlled_probes_are_not_acceptance_metrics": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target-download-status", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--train-prompts-sha256", required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--coverage-mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--feature-provenance", action="store_true",
                        help="gate packed same-forward features and rejected-suffix causality")
    parser.add_argument("--eos-search-max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    provenance = {"source_sha256": source_hashes(),
                  "token_gate_version": PACKED_TOKEN_GATE_VERSION,
                  "train_prompts_sha256": args.train_prompts_sha256,
                  "seed": 20261030, "device": args.device,
                  "dtype": "bfloat16", "window": 8,
                  "request_count": args.request_count,
                  "max_new_tokens": args.max_new_tokens,
                  "coverage_mode": args.coverage_mode,
                  "feature_protocol_version": (PACKED_FEATURE_PROTOCOL_VERSION
                                               if args.feature_provenance else None),
                  "eos_search_max_new_tokens": args.eos_search_max_new_tokens}
    current = {}
    try:
        if args.max_new_tokens < 9:
            raise ValueError("max_new_tokens must exceed the eight-token draft window")
        if args.eos_search_max_new_tokens < args.max_new_tokens:
            raise ValueError("EOS search cap must be at least max_new_tokens")
        prompts = load_train_only(args.train_prompts, args.train_prompts_sha256,
                                  args.request_count)
        token_manifest = json.loads(args.tokenizer_manifest.read_text(encoding="utf-8"))
        provenance["tokenizer_manifest_sha256"] = sha256_file(args.tokenizer_manifest)
        provenance["model_file_sha256"] = verify_models(
            args.target, args.draft, args.target_download_status)
        target_tokenizer = AutoTokenizer.from_pretrained(
            str(args.target), local_files_only=True, trust_remote_code=False)
        draft_tokenizer = AutoTokenizer.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False)
        provenance["tokenizer_sha256"] = verify_tokenizers(
            args.target, args.draft, target_tokenizer, draft_tokenizer, token_manifest)
        # All file hashing/tokenizer loading above is CPU-only. The three
        # physical-card checks immediately precede first CUDA allocation.
        provenance["gpu_admission"] = admit_gpu(args.device)
        if not torch.cuda.is_available():
            raise RuntimeError("real packed P0 requires a CUDA GPU")
        device = torch.device(args.device)
        torch.manual_seed(20261030)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        target = HFTargetBackend(args.target, target_tokenizer, device=device, dtype=torch.bfloat16)
        draft_model = AutoModelForCausalLM.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False,
            attn_implementation="eager", torch_dtype=torch.bfloat16).to(device)
        provenance["draft_loaded_metadata"] = verify_draft_metadata(
            args.draft, draft_model, draft_tokenizer,
            provenance["model_file_sha256"])
        codec = FeedbackCodec(CodecConfig(
            target_hidden_size=target.hidden_size,
            draft_hidden_size=int(draft_model.config.hidden_size),
            vocab_size=target.vocab_size), draft_model.get_output_embeddings()).to(device)
        draft = HFDraftBackend(draft_model, draft_tokenizer, codec,
                               target=target, device=device, dtype=torch.bfloat16)
        for role, model in (("Target", target.model), ("Draft", draft.model)):
            if next(model.parameters()).dtype != torch.bfloat16:
                raise RuntimeError(f"{role} model did not load in BF16")
        eos = target.tokenizer.eos_token_id
        if type(eos) is not int:
            raise RuntimeError("Target EOS must be a single token ID")
        if any(parameter.requires_grad for parameter in target.model.parameters()):
            raise RuntimeError("Target has an unfrozen parameter")
        if any(parameter.requires_grad for parameter in draft.model.parameters()):
            raise RuntimeError("Draft has an unfrozen parameter")
        arms = (("raw_draft", None), ("random_untrained_8B", CodecEncoder(codec)))
        rows = []
        oracle_rows = []
        natural_oracles = {}
        for prompt_row in prompts:
            prompt = list(prompt_row["input_ids"])
            current = {"id": prompt_row["id"], "phase": "independent_target_ar"}
            oracle = autoregressive_generate(target, prompt, args.max_new_tokens, eos)
            natural_oracles[prompt_row["id"]] = oracle
            if not rows and autoregressive_generate(target, prompt, args.max_new_tokens, eos) != oracle:
                raise GateFailure("independent Target AR is not repeatable", current)
            oracle_rows.append({"id": prompt_row["id"],
                                **generation_signature(oracle, eos, args.max_new_tokens)})
            for arm, encoder in arms:
                current = {"id": prompt_row["id"], "arm": arm,
                           "phase": "sequential_and_packed"}
                sequential = SpeculativeDecoder(target, draft, encoder, 8, eos).generate(
                    prompt, args.max_new_tokens)
                packed = PackedSpeculativeDecoder(
                    target, draft, encoder, 8, eos,
                    capture_target_logits=args.feature_provenance,
                    feature_provenance_gate=args.feature_provenance).generate(
                    prompt, args.max_new_tokens)
                if generation_signature(sequential.tokens, eos, args.max_new_tokens) != generation_signature(
                    oracle, eos, args.max_new_tokens):
                    raise GateFailure("sequential reference differs from independent Target AR", current)
                checks = check_rounds(target, packed, oracle, prompt,
                                      args.max_new_tokens, eos, arm, prompt_row["id"],
                                      feature_provenance=args.feature_provenance)
                if arm == "raw_draft":
                    sequential_shape = [(r.prefix_before, r.proposals, r.accepted_count,
                                         r.committed_tokens, r.bonus_token) for r in sequential.rounds]
                    packed_shape = [(r.prefix_before, r.proposals, r.accepted_count,
                                     r.committed_tokens, r.bonus_token) for r in packed.generation.rounds]
                    if sequential_shape != packed_shape:
                        raise GateFailure("raw packed rounds differ from sequential reference", current)
                rows.append({"id": prompt_row["id"], "arm": arm,
                             **generation_signature(packed.generation.tokens, eos,
                                                    args.max_new_tokens),
                             "verification_rounds": len(checks), "round_checks": checks})
                atomic_json(args.output / "progress.json", {"state": "running",
                            "oracle_results": oracle_rows, "rows": rows})
        controlled_rows = (controlled_coverage_probes(
            target, prompts, natural_oracles, eos, args.eos_search_max_new_tokens,
            feature_provenance=args.feature_provenance)
            if args.coverage_mode == "full" else [])
        coverage = coverage_summary(rows, controlled_rows)
        atomic_json(args.output / "coverage.json", coverage)
        torch.cuda.synchronize(device)
        provenance["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        if args.coverage_mode == "full" and coverage["missing"]:
            raise GateFailure("real BF16 packed branch coverage is incomplete", {
                "state": "incomplete_coverage", "coverage": coverage})
        hidden_checks = [check for row in [*rows, *controlled_rows]
                         for check in row["round_checks"]]
        exported_hidden_checks = [check for check in hidden_checks
                                  if check["valid_hidden_max_abs_diff_fp32"] is not None]
        hypothetical_mismatches = sum(
            not position["argmax_equal"]
            for check in hidden_checks
            for position in check["hypothetical_positions"])
        if args.feature_provenance and any("feature_provenance" not in check
                                           for check in hidden_checks):
            raise GateFailure("packed feature provenance is missing a round", current)
        atomic_json(args.output / "report.json", {
                    "state": ("passed" if args.coverage_mode == "full"
                              else ("smoke_feature_provenance_only" if args.feature_provenance
                                    else "smoke_token_equivalence_only")),
                    "scope": "real_qwen3_bf16_packed_target_committed_path_p0",
                    "latency_claim_valid": False, "oracle": "independent_full_prefix_target_ar",
                    "packed_full_hidden_feature_transfer_validated": False,
                    "packed_full_logit_feature_transfer_validated": False,
                    "packed_feature_provenance_gate_passed": args.feature_provenance,
                    "packed_feature_provenance_rounds": (
                        len(hidden_checks) if args.feature_provenance else 0),
                    "hypothetical_argmax_mismatches_diagnostic_only":
                        hypothetical_mismatches,
                    "feature_transfer_note": (
                        "F-H/F-LOGIT packed feature transfer remains unverified; "
                        "per-round hidden numerical differences are diagnostic only"),
                    "device": str(device), "dtype": "bfloat16", "window": 8,
                    "request_count": len(prompts), "max_new_tokens": args.max_new_tokens,
                    "prompt_ids": [r["id"] for r in prompts],
                    "oracle_results": oracle_rows, "checks": rows,
                    "controlled_coverage_probes": controlled_rows,
                    "coverage": coverage,
                    "packed_hidden_all_rounds_bitwise_equal_to_full_prefix":
                        all(check["valid_hidden_bitwise_equal_to_full_prefix"]
                            for check in exported_hidden_checks),
                    "packed_hidden_max_abs_diff_fp32": max((
                        check["valid_hidden_max_abs_diff_fp32"]
                        for check in exported_hidden_checks), default=None),
                    "seconds": time.time() - started, **provenance})
    except BaseException as exc:
        evidence = exc.evidence if isinstance(exc, GateFailure) else current
        state = evidence.get("state", "failed") if isinstance(evidence, dict) else "failed"
        atomic_json(args.output / "failure.json", {"state": state,
                    "error": repr(exc), "evidence": evidence,
                    "traceback": traceback.format_exc(),
                    "seconds": time.time() - started, **provenance})
        raise


if __name__ == "__main__":
    main()
