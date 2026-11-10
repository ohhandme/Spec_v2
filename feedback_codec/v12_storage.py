"""Atomic, split-specific storage for one complete v12 raw-policy request."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file, save_file

from .v12_round_assembly import assemble_request, digest, request_id


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    if path.exists() or tmp.exists():
        raise FileExistsError(f"request artifact already exists: {path}")
    try:
        with tmp.open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def persist_request(
    *, root: Path, prompt: Any, records: Sequence[Any],
    draft_steps: Sequence[dict[str, Any]], oracle_tokens: Sequence[int],
    oracle_steps: Sequence[dict[str, Any]], eos_token_id: int,
    dataset_lock_sha256: str, policy_id: str, execution_id: str,
    source_sha256: str, schema_lock_sha256: str,
    provenance: dict[str, Any],
    audit: dict[str, Any], max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Persist exact complete request; write the accepted manifest last.

    A crash before the manifest leaves partial artifacts that require explicit
    audit.  Repeated requests never overwrite an existing manifest or artifact.
    """
    if audit.get("exact") is not True:
        raise ValueError("nonexact request cannot be accepted as v12 data")
    from .v12_acceptance import PROVENANCE_KEYS
    required_before_audit = set(PROVENANCE_KEYS) - {"exact_stream_receipt_sha256"}
    if (not isinstance(provenance, dict)
            or not required_before_audit.issubset(provenance)
            or provenance.get("dataset_lock_sha256") != dataset_lock_sha256
            or provenance.get("prompt_id") != prompt.id
            or provenance.get("family_id") != prompt.family_id
            or provenance.get("split") != prompt.split
            or provenance.get("source") != prompt.task_type
            or provenance.get("policy_id") != policy_id
            or provenance.get("execution_id") != execution_id
            or provenance.get("extension_source_sha256") != source_sha256
            or provenance.get("schema_lock_sha256") != schema_lock_sha256
            or provenance.get("max_new_tokens") != max_new_tokens):
        raise ValueError("v12 immutable source/model/data provenance absent")
    if root.name != prompt.split or prompt.split not in ("train", "val"):
        raise ValueError("request storage split differs from prompt")
    rid = request_id(dataset_lock_sha256, prompt.split, prompt.id,
                     policy_id, execution_id)
    manifest = root / "manifests" / f"{rid}.json"
    if manifest.exists():
        raise FileExistsError("complete request manifest already exists")
    oracle_path = root / "oracle_tapes" / f"{rid}.json"
    hidden_path = root / "hidden" / f"{rid}.safetensors"
    observations_path = root / "observations" / f"{rid}.jsonl"
    labels_path = root / "labels" / f"{rid}.jsonl"
    audit_path = root / "audit" / f"{rid}.json"
    paths = [oracle_path, observations_path, labels_path, audit_path]
    if any(p.exists() or p.with_name(p.name + ".partial").exists()
           for p in paths + [hidden_path]):
        raise FileExistsError("incomplete prior request needs explicit audit")
    oracle = {
        "request_id": rid,
        "prompt_id": prompt.id,
        "split": prompt.split,
        "token_ids": list(oracle_tokens),
        "steps": list(oracle_steps),
        "source_sha256": source_sha256,
        "schema_lock_sha256": schema_lock_sha256,
    }
    _atomic_bytes(oracle_path, _json_bytes(oracle))
    oracle_sha = file_sha256(oracle_path)
    tensors = {
        f"round_{i}": record.observation_hidden.detach().cpu().contiguous()
        for i, record in enumerate(records)
        if record.observation_hidden is not None
    }
    hidden_sha: str | None = None
    if tensors:
        hidden_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = hidden_path.with_name(hidden_path.name + ".partial")
        save_file(tensors, str(tmp))
        os.replace(tmp, hidden_path)
        hidden_sha = file_sha256(hidden_path)
        loaded = load_file(str(hidden_path), device="cpu")
        if (set(loaded) != set(tensors)
                or any(not torch.equal(loaded[k], value)
                       for k, value in tensors.items())):
            raise RuntimeError("saved teacher tensor bytes differ")

    def hidden_ref(index: int, record: Any, valid: list[int]) -> dict[str, Any]:
        key = f"round_{index}"
        if hidden_sha is None or key not in tensors:
            raise RuntimeError("legal teacher hidden tensor is not persisted")
        value = tensors[key]
        return {
            "path": hidden_path.relative_to(root).as_posix(), "key": key,
            "shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hidden_sha, "prefix_sha256": digest(valid),
        }

    def oracle_ref(offset: int, length: int) -> dict[str, Any]:
        if not 0 <= offset < len(oracle_tokens) or not 0 < length <= 8:
            raise ValueError("future Target ref crosses the real AR tape")
        return {"path": oracle_path.relative_to(root).as_posix(),
                "sha256": oracle_sha, "offset": offset, "length": length}

    observations, labels = assemble_request(
        prompt=prompt, records=records, draft_steps=draft_steps,
        oracle_tokens=oracle_tokens, oracle_steps=oracle_steps,
        eos_token_id=eos_token_id, dataset_lock_sha256=dataset_lock_sha256,
        policy_id=policy_id, execution_id=execution_id,
        hidden_ref=hidden_ref, oracle_ref=oracle_ref,
        max_new_tokens=max_new_tokens)
    _atomic_bytes(observations_path, _jsonl_bytes(observations))
    _atomic_bytes(labels_path, _jsonl_bytes(labels))
    _atomic_bytes(audit_path, _json_bytes(audit))
    provenance = dict(provenance)
    provenance["exact_stream_receipt_sha256"] = file_sha256(audit_path)
    files = {name: {
        "path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    } for name, path in (
        ("observations", observations_path), ("labels", labels_path),
        ("oracle_tape", oracle_path), ("audit", audit_path))}
    files["hidden"] = (None if hidden_sha is None else {
        "path": hidden_path.relative_to(root).as_posix(), "sha256": hidden_sha,
        "bytes": hidden_path.stat().st_size,
        "tensor_keys": sorted(tensors),
    })
    result = {
        "request_id": rid, "prompt_id": prompt.id,
        "family_id": prompt.family_id, "split": prompt.split,
        "source": prompt.task_type, "policy_id": policy_id,
        "execution_id": execution_id, "schema_lock_sha256": schema_lock_sha256,
        "extension_source_sha256": source_sha256,
        "dataset_lock_sha256": dataset_lock_sha256,
        "provenance": provenance,
        "exact": True, "rounds": len(observations),
        "draft_positions": len(draft_steps), "oracle_steps": len(oracle_steps),
        "files": files,
    }
    _atomic_bytes(manifest, _json_bytes(result))
    return result
