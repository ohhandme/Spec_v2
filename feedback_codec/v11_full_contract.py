"""Fail-closed inventory and data reader for the frozen v11 1000/200 pilot.

The old v10 smoke chunks retain their own immutable plan and source. New v11
chunks live in split-specific directories. Future labels are returned only as
separate values and never entered into a prepared observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .fixed_probe import ProbeExample, ProbeObservation, ProbePrompt, _validate_prompts
from .packed_probe_numeric_v3 import canonical_bytes


PARENT_SHA256 = "1bfaeeb2f8699933a94a180633b5c361601630cb37409502d670158e3a55fdc2"
PROGRAM_SHA256 = "498abcadf1a639a0cac96e4e5e29da53e9d2ede357f21c6cdf6b0d629b55cd10"
SOURCE_COUNTS = {
    "train": {"ultrachat": 600, "dolly": 200, "codealpaca": 100, "gsm8k": 100},
    "val": {"ultrachat": 120, "dolly": 40, "codealpaca": 20, "gsm8k": 20},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class PilotWindow:
    observation: ProbeObservation
    labels: tuple[int, ...]
    chunk_sha256: str
    imported_v10: bool

    @property
    def key(self) -> tuple[str, int]:
        return self.observation.key


def locked_prompts(parent: Path, meta_path: Path) -> dict[str, tuple[ProbePrompt, ...]]:
    if sha256_file(parent) != PARENT_SHA256:
        raise ValueError("v11 parent tokenized bytes differ")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("output_sha256") != PARENT_SHA256
            or meta.get("counts") != {"train": 1000, "val": 200}
            or meta.get("sealed_test_loaded") is not False):
        raise ValueError("v11 parent metadata differs")
    result: dict[str, list[ProbePrompt]] = {"train": [], "val": []}
    for line in parent.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("split")
        if split not in result:
            raise ValueError("formal, test, or unsupported split in v11 parent")
        result[split].append(ProbePrompt(
            row["id"], row.get("family_id", row.get("prompt_family_sha256")),
            split, tuple(row["input_ids"]), row.get("task_type", row.get("source")),
        ))
    all_prompts = result["train"] + result["val"]
    _validate_prompts(all_prompts)
    for split, prompts in result.items():
        if len(prompts) != sum(SOURCE_COUNTS[split].values()):
            raise ValueError(f"{split} prompt count differs")
        counts: dict[str, int] = {}
        for prompt in prompts:
            source = prompt.task_type.lower()
            if source == "gsm8k_train":
                source = "gsm8k"
            counts[source] = counts.get(source, 0) + 1
        if counts != SOURCE_COUNTS[split]:
            raise ValueError(f"{split} source allocation differs: {counts}")
    return {key: tuple(value) for key, value in result.items()}


def checked_window_rows(rows: Iterable[dict], *, prompt: ProbePrompt,
                        chunk_sha256: str, imported_v10: bool,
                        eos_token_id: int) -> tuple[PilotWindow, ...]:
    result: list[PilotWindow] = []
    seen: set[int] = set()
    for row in rows:
        if (row.get("prompt_id") != prompt.id or row.get("split") != prompt.split
                or row.get("family_id") != prompt.family_id):
            raise ValueError("pilot row crosses prompt, family, or split")
        ex = ProbeExample.from_json(row)
        o = ex.observation
        if (o.round_index in seen or o.window != 8 or o.remaining_budget <= 0
                or o.target_hidden_prefix != o.valid_prefix
                or o.corrected_prefix != o.valid_prefix + (o.correction_token,)
                or o.correction_token == eos_token_id
                or not 1 <= len(ex.target_future_ids) <= 8
                or row.get("independent_ar_prefix_exact") is not True):
            raise ValueError("terminal, duplicate, future, or invalid pilot row")
        seen.add(o.round_index)
        result.append(PilotWindow(o, ex.target_future_ids, chunk_sha256,
                                  imported_v10))
    if len(result) > 8:
        raise ValueError("prompt has more than eight causal windows")
    return tuple(result)


def prompt_input_sha256(prompt: ProbePrompt) -> str:
    return canonical_sha256(list(prompt.input_ids))


def window_id(window: PilotWindow) -> str:
    return canonical_sha256({
        "prompt_id": window.observation.prompt_id,
        "round_index": window.observation.round_index,
        "corrected_prefix": window.observation.corrected_prefix,
        "future_ids": window.labels,
        "chunk_sha256": window.chunk_sha256,
    })


def causal_window_id(window: PilotWindow) -> str:
    """ID for interventions that must not inspect future Target labels."""
    return canonical_sha256({
        "prompt_id": window.observation.prompt_id,
        "round_index": window.observation.round_index,
        "corrected_prefix": window.observation.corrected_prefix,
    })


def checked_new_chunk(path: Path, *, prompt: ProbePrompt, plan_sha256: str,
                      eos_token_id: int) -> tuple[PilotWindow, ...]:
    """Verify the original v3 checksum, provenance, AR and causal row gates."""
    from scripts.build_packed_probe_numeric_v3 import _checked_chunk

    raw_sha = sha256_file(path)
    chunk = _checked_chunk(path, prompt, plan_sha256, 1, eos_token_id)
    return checked_window_rows(chunk["examples"], prompt=prompt,
                               chunk_sha256=raw_sha, imported_v10=False,
                               eos_token_id=eos_token_id)


def checked_v10_import(v10_root: Path, *, parent_prompts: tuple[ProbePrompt, ...],
                       source_zip_sha256: str, model_hashes: dict,
                       tokenizer_sha256: dict, p0_sha256: str,
                       eos_token_id: int) -> tuple[tuple[PilotWindow, ...], dict]:
    """Import the four immutable v10 train chunks under their original plan.

    The v10 reader verifies original source, plan, merged corpus and every
    chunk.  This adapter never copies or relabels a legacy row.
    """
    from .v10_train_smoke_reader import load_v10_train_smoke

    protocol_path = v10_root / "collection_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    corpus = load_v10_train_smoke(
        v10_root / "fixed_prefix.jsonl", protocol_path,
        expected_parent_sha=PARENT_SHA256,
        expected_plan_sha=protocol["collection_plan_sha256"],
        expected_source_sha=source_zip_sha256,
        expected_model_hashes=model_hashes,
        expected_tokenizer_sha=tokenizer_sha256,
        expected_p0_sha=p0_sha256,
    )
    inventory = {p.id: p for p in parent_prompts}
    old_ids = set(corpus.prompt_ids)
    if len(old_ids) != 4 or not old_ids.issubset(inventory):
        raise ValueError("legacy four prompts absent from locked train")
    grouped: dict[str, list[dict]] = {key: [] for key in old_ids}
    for line in (v10_root / "fixed_prefix.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[row["prompt_id"]].append(row)
    windows = tuple(w for key in corpus.prompt_ids for w in checked_window_rows(
        grouped[key], prompt=inventory[key],
        chunk_sha256=protocol["chunk_sha256"][key], imported_v10=True,
        eos_token_id=eos_token_id))
    if len(windows) != len(corpus.windows):
        raise ValueError("legacy window inventory changed")
    receipt = {
        "scope": "immutable_v10_chunk_import", "prompt_ids": list(corpus.prompt_ids),
        "old_plan_sha256": protocol["collection_plan_sha256"],
        "old_source_zip_sha256": protocol["source_zip_sha256"],
        "old_protocol_sha256": sha256_file(protocol_path),
        "old_corpus_sha256": corpus.corpus_sha256,
        "old_chunk_sha256": protocol["chunk_sha256"],
        "windows": len(windows),
    }
    return windows, receipt
