"""Fail-closed audit of the complete v12 train collection before training.

The reference schema validator checks JSON semantics.  This audit additionally
opens every referenced byte artifact and verifies the locked prompt membership,
independent AR receipt, original-forward counts, and physical data separation.
It never reads validation data or starts an optimizer.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file

from .fixed_probe import ProbePrompt
from .v12_round_assembly import digest, request_id
from .v12_storage import _atomic_bytes, _json_bytes, file_sha256


PROVENANCE_KEYS = (
    "dataset_lock_sha256", "prompt_id", "family_id", "split", "source",
    "execution_id", "policy_id", "draft_checkpoint_sha256",
    "feedback_policy_sha256", "seed", "model_sha256", "tokenizer_sha256",
    "chat_template_sha256", "verifier_source_sha256",
    "extension_source_sha256", "observer_source_sha256",
    "schema_lock_sha256", "collection_phase", "exact_stream_receipt_sha256",
    "max_new_tokens",
)


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(value, dict), f"expected object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    _need(bool(rows) and all(isinstance(r, dict) for r in rows),
          f"empty or malformed rows: {path}")
    return rows


def _artifact(root: Path, entry: dict[str, Any]) -> Path:
    _need(isinstance(entry, dict), "missing artifact descriptor")
    name = entry.get("path")
    _need(isinstance(name, str) and name and "\\" not in name,
          "invalid artifact relative path")
    path = (root / name).resolve()
    _need(path.is_relative_to(root.resolve()) and path.is_file(),
          f"artifact outside split or missing: {name}")
    _need(file_sha256(path) == entry.get("sha256")
          and path.stat().st_size == entry.get("bytes"),
          f"artifact hash or length mismatch: {name}")
    return path


def _schema_validator(path: Path):
    _need(path.is_file(), "reference schema validator missing")
    spec = importlib.util.spec_from_file_location("v12_locked_validator", path)
    _need(spec is not None and spec.loader is not None,
          "reference validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate


def audit_train_collection(
    *, root: Path, prompts: Sequence[ProbePrompt],
    dataset_lock_sha256: str, schema_lock_sha256: str,
    extension_source_sha256: str, verifier_source_sha256: str,
    policy_id: str, execution_id: str,
    expected_provenance: dict[str, Any],
    reference_validator: Path, reference_validator_sha256: str,
    input_isolation_receipt: Path, input_isolation_receipt_sha256: str,
    output: Path | None = None,
) -> dict[str, Any]:
    """Return a receipt only for 1000 fully audited actual train requests.

    ``output`` is written once, after every byte and row passes.  A caller
    must separately verify this receipt and its SHA immediately before its
    first optimizer update.  No partial collection receives train_ready=true.
    """
    _need(root.name == "train" and len(prompts) == 1000
          and all(type(p) is ProbePrompt and p.split == "train" for p in prompts),
          "locked train membership must contain exactly 1000 prompts")
    _need(len({p.id for p in prompts}) == 1000, "duplicate locked prompt ID")
    _need(file_sha256(reference_validator) == reference_validator_sha256,
          "reference validator source differs")
    _need(file_sha256(input_isolation_receipt)
          == input_isolation_receipt_sha256,
          "future-label input isolation receipt differs")
    isolation = _read_json(input_isolation_receipt)
    _need(isolation.get("passed") is True
          and isolation.get("extension_source_sha256")
              == extension_source_sha256
          and isolation.get("schema_lock_sha256") == schema_lock_sha256
          and isolation.get("future_labels_visible_to_input_reader") is False,
          "future-label input reader has no real locked isolation proof")
    validate = _schema_validator(reference_validator)
    _need(all(k in expected_provenance for k in PROVENANCE_KEYS
              if k not in ("prompt_id", "family_id", "source", "split",
                           "exact_stream_receipt_sha256")),
          "locked run provenance incomplete")
    manifest_dir = root / "manifests"
    expected = {request_id(dataset_lock_sha256, "train", p.id,
                           policy_id, execution_id): p for p in prompts}
    actual = {p.stem for p in manifest_dir.glob("*.json")}
    _need(actual == set(expected),
          f"train manifests incomplete or extra: expected1000 got{len(actual)}")
    _need(not list((root / "audit" / "failures").glob("*.json")),
          "unresolved exact-stream failures exist")
    all_rows: list[dict[str, Any]] = []
    counts = {"locked_requests": 1000, "exact_requests": 0,
              "fully_valid_requests": 0, "rounds": 0,
              "draft_positions": 0, "future_tokens": 0,
              "teacher_available_rounds": 0, "next_rounds": 0,
              "missing_required": 0, "bad_links": 0, "bad_refs": 0,
              "failed_requests": 0, "wrong_split_or_policy": 0}
    member_hashes: list[dict[str, str]] = []
    for rid in sorted(expected):
        prompt = expected[rid]
        manifest_path = manifest_dir / f"{rid}.json"
        manifest = _read_json(manifest_path)
        _need(manifest.get("request_id") == rid
              and manifest.get("prompt_id") == prompt.id
              and manifest.get("family_id") == prompt.family_id
              and manifest.get("source") == prompt.task_type
              and manifest.get("split") == "train"
              and manifest.get("policy_id") == policy_id
              and manifest.get("execution_id") == execution_id
              and manifest.get("dataset_lock_sha256") == dataset_lock_sha256
              and manifest.get("schema_lock_sha256") == schema_lock_sha256
              and manifest.get("extension_source_sha256")
                  == extension_source_sha256
              and manifest.get("exact") is True,
              f"manifest membership/source differs: {rid}")
        files = manifest.get("files")
        _need(isinstance(files, dict), f"missing files map: {rid}")
        provenance = manifest.get("provenance")
        _need(isinstance(provenance, dict)
              and all(k in provenance for k in PROVENANCE_KEYS)
              and all(provenance.get(k) == value for k, value in
                      expected_provenance.items())
              and provenance.get("prompt_id") == prompt.id
              and provenance.get("family_id") == prompt.family_id
              and provenance.get("source") == prompt.task_type
              and provenance.get("split") == "train"
              and provenance.get("exact_stream_receipt_sha256")
                  == files.get("audit", {}).get("sha256")
              and provenance.get("verifier_source_sha256")
                  == verifier_source_sha256
              and provenance.get("max_new_tokens") == 128
              and provenance.get("collection_phase") == "train_collection"
              and provenance.get("policy_id") == policy_id,
              f"immutable run provenance absent or wrong: {rid}")
        resolved = {name: _artifact(root, files[name]) for name in (
            "observations", "labels", "oracle_tape", "audit")}
        hidden_entry = files.get("hidden")
        hidden_path = _artifact(root, hidden_entry) if hidden_entry else None
        hidden = load_file(str(hidden_path), device="cpu") if hidden_path else {}
        observations = _read_jsonl(resolved["observations"])
        labels = _read_jsonl(resolved["labels"])
        oracle = _read_json(resolved["oracle_tape"])
        audit = _read_json(resolved["audit"])
        _need(len(observations) == len(labels) == manifest.get("rounds")
              and sum(len(o.get("draft_steps", [])) for o in observations)
                  == manifest.get("draft_positions"),
              f"manifest rows/steps count differs: {rid}")
        _need(audit.get("exact") is True
              and audit.get("packed_ids") == audit.get("oracle_ids")
              and audit.get("packed_eos") == audit.get("oracle_eos")
              and audit.get("packed_truncated") == audit.get("oracle_truncated")
              and audit.get("first_difference_index") is None
              and audit.get("draft_actual_forward_count")
                  == audit.get("draft_proposals_count")
                  == manifest.get("draft_positions")
              and audit.get("oracle_actual_forward_count")
                  == len(oracle.get("steps", []))
                  == len(oracle.get("token_ids", [])),
              f"real forward/exact receipt differs: {rid}")
        _need(oracle.get("request_id") == rid
              and oracle.get("token_ids") == audit.get("oracle_ids")
              and oracle.get("schema_lock_sha256") == schema_lock_sha256
              and oracle.get("source_sha256") == extension_source_sha256,
              f"independent AR tape differs: {rid}")
        oracle_prefix = list(prompt.input_ids)
        for j, (token, step) in enumerate(zip(
                oracle["token_ids"], oracle["steps"], strict=True)):
            _need(step.get("query_prefix_sha256") == digest(oracle_prefix)
                  and step.get("greedy_token_id") == token
                  and step.get("distribution_stage")
                      == "independent_AR_full_prefix_before_argmax"
                  and isinstance(step.get("actual_forward_logits_sha256"), str)
                  and len(step["actual_forward_logits_sha256"]) == 64
                  and step.get("topk_storage_dtype") == "torch.float16"
                  and step.get("temperature") == 1.0
                  and step.get("vocab_size", 0) >= 32
                  and len(step.get("topk_ids", [])) == 32
                  and len(step.get("topk_logits", [])) == 32
                  and all(type(t) is int and 0 <= t < step["vocab_size"]
                          for t in step["topk_ids"])
                  and len(set(step["topk_ids"])) == 32
                  and all(type(v) in (int, float) and math.isfinite(v)
                          for v in step["topk_logits"])
                  and math.isfinite(step.get("logsumexp", math.nan)),
                  f"independent AR step Top32/prefix missing: {rid}/{j}")
            oracle_prefix.append(token)
        cursor = list(prompt.input_ids)
        referenced_hidden: set[str] = set()
        for i, (obs, lab) in enumerate(zip(observations, labels, strict=True)):
            _need(obs.get("round_id") == i and obs.get("request_id") == rid
                  and obs.get("prefix_token_ids") == cursor
                  and lab.get("request_id") == rid,
                  f"actual prefix/round key differs: {rid}/{i}")
            _need(all(isinstance(s.get("actual_forward_logits_sha256"), str)
                      and len(s["actual_forward_logits_sha256"]) == 64
                      for s in obs.get("draft_steps", [])),
                  f"actual Draft logits hash absent: {rid}/{i}")
            cursor = obs["prefix_after_commit_token_ids"]
            ref = obs.get("target_valid_hidden_ref")
            if ref is None:
                _need(obs.get("corrected_token_id") is None,
                      f"teacher absent with b: {rid}/{i}")
            else:
                _need(hidden_path is not None
                      and ref.get("path") == hidden_entry["path"]
                      and ref.get("sha256") == hidden_entry["sha256"]
                      and ref.get("key") in hidden,
                      f"teacher file/key differs: {rid}/{i}")
                tensor = hidden[ref["key"]]
                _need(ref["key"] not in referenced_hidden,
                      f"teacher tensor reused across rounds: {rid}/{i}")
                referenced_hidden.add(ref["key"])
                _need(list(tensor.shape) == ref.get("shape")
                      and str(tensor.dtype) == ref.get("dtype")
                      and torch.isfinite(tensor).all().item()
                      and ref.get("prefix_sha256")
                          == digest(obs["valid_prefix_token_ids"]),
                      f"teacher tensor/boundary differs: {rid}/{i}")
                counts["teacher_available_rounds"] += 1
            future = lab.get("target_future", {})
            future_ref = future.get("topk_ref")
            if future_ref is not None:
                offset, length = future_ref.get("offset"), future_ref.get("length")
                _need(future_ref.get("path") == files["oracle_tape"]["path"]
                      and future_ref.get("sha256")
                          == files["oracle_tape"]["sha256"]
                      and type(offset) is int and type(length) is int
                      and 0 <= offset <= len(oracle["token_ids"]) - length
                      and future.get("token_ids")
                          == oracle["token_ids"][offset:offset + length]
                      and all(oracle["steps"][j].get("greedy_token_id")
                              == oracle["token_ids"][j]
                              for j in range(offset, offset + length)),
                      f"future oracle ref differs: {rid}/{i}")
            counts["future_tokens"] += len(future.get("token_ids", []))
            counts["next_rounds"] += int(lab.get("next_round_id") is not None)
        _need(set(hidden) == referenced_hidden,
              f"orphan or missing teacher tensor: {rid}")
        _need(cursor[len(prompt.input_ids):] == oracle["token_ids"],
              f"committed full stream differs from independent AR: {rid}")
        all_rows.extend({"observation": o, "labels": l}
                        for o, l in zip(observations, labels, strict=True))
        counts["exact_requests"] += 1
        counts["fully_valid_requests"] += 1
        counts["rounds"] += len(observations)
        counts["draft_positions"] += manifest["draft_positions"]
        member_hashes.append({"request_id": rid,
                              "manifest_sha256": file_sha256(manifest_path)})
    reference = validate(all_rows)
    _need(reference == {"requests": 1000, "rounds": counts["rounds"]},
          "reference validator rejected full collection")
    receipt = {
        "schema_version": "feedback64-rounds-v12",
        "train_ready": True, "training_updates_before_gate": 0,
        "dataset_lock_sha256": dataset_lock_sha256,
        "schema_lock_sha256": schema_lock_sha256,
        "extension_source_sha256": extension_source_sha256,
        "verifier_source_sha256": verifier_source_sha256,
        "reference_validator_sha256": reference_validator_sha256,
        "policy_id": policy_id, "execution_id": execution_id,
        "counts": counts, "members": member_hashes,
        "member_digest_sha256": digest(member_hashes),
        "physical_future_label_input_isolation": True,
        "input_isolation_receipt_sha256": input_isolation_receipt_sha256,
    }
    if output is not None:
        _atomic_bytes(output, _json_bytes(receipt))
    return receipt
