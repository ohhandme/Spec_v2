"""Read only post-verify online observations; future labels stay inaccessible.

Training labels are joined by a separate, gate-protected reader after the
complete 1000-request acceptance.  This module has no label/oracle path input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .v12_storage import file_sha256


BOUNDARY_FIELDS = frozenset({
    "prefix_token_ids", "draft_length", "draft_token_ids",
    "accepted_count", "first_reject_position", "rejected_token_id",
    "corrected_token_id", "correction_kind", "valid_prefix_token_ids",
    "prefix_after_commit_token_ids", "target_valid_hidden_ref",
    "target_topk", "draft_steps", "terminal_reason",
    "remaining_budget_before", "remaining_budget_after",
})


def load_boundary_inputs(root: Path, manifest_path: Path) -> tuple[dict[str, Any], ...]:
    """Return only causal boundary fields from the observations artifact.

    The manifest identifies exactly one observations file.  No labels or
    oracle_tapes are opened by this reader, including when those directories
    are absent or unreadable.  The caller may separately load legal current
    teacher tensors by the returned same-boundary hidden refs.
    """
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    if (root.name not in ("train", "val") or
            not manifest_path.is_relative_to(root / "manifests") or
            not manifest_path.is_file()):
        raise ValueError("observation manifest outside locked split")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("files", {}).get("observations", {})
    name = entry.get("path")
    if not isinstance(name, str) or not name.startswith("observations/"):
        raise ValueError("observation artifact missing or wrong namespace")
    path = (root / name).resolve()
    if (not path.is_relative_to(root / "observations") or not path.is_file()
            or file_sha256(path) != entry.get("sha256")):
        raise ValueError("observation file does not match manifest")
    rows = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != manifest.get("rounds"):
        raise ValueError("incomplete observation artifact")
    result: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if (not BOUNDARY_FIELDS.issubset(row)
                or row.get("request_id") != manifest.get("request_id")
                or row.get("round_id") != i
                or row.get("split") != root.name
                or row.get("policy_id") != manifest.get("policy_id")):
            raise ValueError("observation boundary identity or fields differ")
        result.append({key: row[key] for key in BOUNDARY_FIELDS})
    return tuple(result)
