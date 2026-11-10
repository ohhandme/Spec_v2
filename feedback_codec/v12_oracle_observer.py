"""Collect independent Target AR labels and Top-32 from each original step.

``HFTargetBackend.next_with_logit_summary`` uses the same full-prefix BF16
forward as ``next`` and adds reductions only.  The adapter delegates to that
method once per AR token, preserving the original token and EOS rule.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .packed_probe_numeric_v3 import canonical_bytes
from .v12_draft_observer import original_logits_sha256


class TargetARSummaryObserver:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.steps: list[dict[str, Any]] = []
        self.actual_forwards_observed = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def next(self, prefix: list[int]) -> Any:
        native_dtypes: list[str] = []
        original_hashes: list[str] = []

        def count_forward(_module: Any, _args: Any, output: Any) -> None:
            logits = getattr(output, "logits", None)
            if logits is None:
                raise RuntimeError("Target oracle forward has no original logits")
            native_dtypes.append(str(logits.dtype))
            if logits.ndim != 3:
                raise RuntimeError("Target oracle forward shape differs")
            original_hashes.append(original_logits_sha256(logits[0, -1]))

        handle = self.backend.model.register_forward_hook(count_forward)
        try:
            step = self.backend.next_with_logit_summary(prefix)
        finally:
            handle.remove()
        if len(native_dtypes) != 1:
            raise RuntimeError("Target oracle observer expected one original forward")
        summary = step.logit_summary
        if summary is None or summary.greedy_token_id != step.token_id:
            raise RuntimeError("Target oracle summary differs from greedy token")
        if (len(summary.top_token_ids) != min(32, summary.vocab_size)
                or len(summary.top_logits_fp16) != len(summary.top_token_ids)
                or len(set(summary.top_token_ids)) != len(summary.top_token_ids)
                or not math.isfinite(summary.logsumexp_fp32)):
            raise RuntimeError("Target oracle Top-K summary incomplete")
        self.steps.append({
            "query_prefix_sha256": hashlib.sha256(
                canonical_bytes(list(prefix))).hexdigest(),
            "greedy_token_id": int(step.token_id),
            "topk_ids": list(summary.top_token_ids),
            "topk_logits": list(summary.top_logits_fp16),
            "logsumexp": float(summary.logsumexp_fp32),
            "vocab_size": int(summary.vocab_size),
            "temperature": 1.0,
            "logits_dtype": native_dtypes[0],
            "topk_storage_dtype": "torch.float16",
            "distribution_stage": "independent_AR_full_prefix_before_argmax",
            "actual_forward_logits_sha256": original_hashes[0],
        })
        self.actual_forwards_observed += 1
        return step
