"""Observe the *actual* raw Draft proposal forward without changing its token.

The v10 backend and numerical verifier remain frozen.  This adapter installs a
temporary hook only while ``HFDraftBackend.next`` is executing.  It cannot be
used for repaired proposals: their final distribution is downstream of the
model hook and needs a separate observer at the repair output.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from .packed_probe_numeric_v3 import canonical_bytes


def original_logits_sha256(logits: torch.Tensor) -> str:
    """Hash the actual native-dtype row captured by the original forward."""
    row = logits.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(row).hexdigest()


def draft_step_stats(logits: torch.Tensor, token_id: int, *, prefix: list[int],
                     position: int, topk: int = 32) -> dict[str, Any]:
    """Reduce a single final proposal distribution over the whole vocabulary."""
    if (logits.ndim != 1 or logits.numel() < 2 or type(token_id) is not int
            or not 0 <= token_id < logits.numel() or not 1 <= position <= 8
            or topk != 32 or not torch.isfinite(logits).all()):
        raise ValueError("invalid actual Draft distribution or proposal context")
    native_dtype = str(logits.dtype)
    z = logits.float()
    values, ids = torch.topk(z, min(topk, z.numel()))
    logz = torch.logsumexp(z, dim=0)
    logp = z - logz
    p = torch.exp(logp)
    entropy = -torch.sum(p * logp)
    chosen = float(z[token_id].item())
    result = {
        "position": position,
        "token_id": token_id,
        "selected_prob": float(math.exp(chosen - float(logz.item()))),
        "entropy": float(entropy.item()),
        "top2_margin": float((values[0] - values[1]).item()),
        "logit_std": float(torch.std(z, correction=0).item()),
        "topk_ids": [int(x) for x in ids.cpu().tolist()],
        "topk_logits": [float(x) for x in values.cpu().tolist()],
        "selected_logit": chosen,
        "logsumexp": float(logz.item()),
        "vocab_size": int(z.numel()),
        "logits_dtype": native_dtype,
        "topk_storage_dtype": "torch.float32",
        "temperature": 1.0,
        "distribution_stage": "final_proposal_logits",
        "actual_forward_logits_sha256": original_logits_sha256(logits),
        "proposal_prefix_sha256": hashlib.sha256(
            canonical_bytes(list(prefix))).hexdigest(),
    }
    if not all(math.isfinite(result[k]) for k in (
            "selected_prob", "entropy", "top2_margin", "logit_std",
            "selected_logit", "logsumexp")):
        raise FloatingPointError("nonfinite Draft distribution statistic")
    return result


class RawDraftForwardObserver:
    """Delegate unchanged token selection while retaining one real forward row."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.steps: list[dict[str, Any]] = []
        self.actual_forwards_observed = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def next(self, prefix: list[int], codes: tuple[int, ...] | None,
             position: int) -> int:
        if codes is not None:
            raise ValueError("raw observer cannot label repaired logits as raw")
        captured: list[torch.Tensor] = []

        def observe(_module: Any, _args: Any, output: Any) -> None:
            row = getattr(output, "logits", None)
            if not isinstance(row, torch.Tensor) or row.ndim != 3:
                raise RuntimeError("Draft forward hook did not receive logits")
            captured.append(row[0, -1])

        handle = self.backend.model.register_forward_hook(observe)
        try:
            token = int(self.backend.next(prefix, codes, position))
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("expected exactly one original Draft forward")
        if token != int(captured[0].argmax(-1).item()):
            raise RuntimeError("Draft observer token differs from original decision")
        self.steps.append(draft_step_stats(
            captured[0], token, prefix=prefix, position=position))
        self.actual_forwards_observed += 1
        return token
