"""KV-backed greedy verification with an explicit correctness and cost trace.

The default backend uses one packed Target forward.  An opt-in numerical
recheck backend adds explicitly counted Target forwards when BF16 logits are
ambiguous; it must pass the versioned real-model correctness gate before use.
A full-prefix reference and cached ``L+1`` query can disagree at numerical
ties.  No reference answer is silently substituted into the packed path.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
import time
from typing import Any

import torch

from .decoding import (
    DraftBackend, FeedbackEncoderProtocol, GenerationResult, RoundRecord,
    TargetBackend, TargetLogitSummary,
)


PACKED_FEATURE_PROTOCOL_VERSION = "packed_valid_feature_v2"
NUMERICAL_RECHECK_BACKEND = "packed_with_numerical_recheck_candidate_v1"
NUMERICAL_RECHECK_BACKEND_V2 = "packed_with_canonical_ulp_recheck_candidate_v2"
STRICT_CANONICAL_BACKEND = "packed_with_strict_canonical_used_rows_v1"


def _floating_ulp(value: float, dtype: torch.dtype) -> float:
    """Maximum upward/downward neighbor spacing in the original dtype.

    A BF16 top logit of 29.875 has one ULP of 0.125.  This is a
    conservative trigger heuristic, not a bound on packed/AR drift.
    """
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite packed top logit")
    info = torch.finfo(dtype)
    magnitude = abs(value)
    if magnitude <= info.tiny:
        return float(info.tiny * info.eps)
    _, exponent = math.frexp(magnitude)
    return math.ldexp(float(info.eps), exponent - 1)


def _top2_max_ulp(top1: float, top2: float, dtype: torch.dtype) -> float:
    """Largest adjacent spacing around either original packed top-2 logit."""
    return max(_floating_ulp(top1, dtype), _floating_ulp(top2, dtype))


def _cache_length(cache: Any | None) -> int:
    if cache is None:
        return 0
    get_length = getattr(cache, "get_seq_length", None)
    if callable(get_length):
        return int(get_length())
    if isinstance(cache, (tuple, list)) and cache:
        return int(cache[0][0].shape[-2])
    raise TypeError("packed verification needs a crop-capable KV cache")


def _crop_cache(cache: Any, length: int) -> Any:
    if length < 0 or length > _cache_length(cache):
        raise ValueError("invalid KV crop length")
    crop = getattr(cache, "crop", None)
    if callable(crop):
        result = crop(length)
        cropped = cache if result is None else result
    elif isinstance(cache, (tuple, list)) and cache:
        cropped = tuple((key[..., :length, :], value[..., :length, :])
                        for key, value in cache)
    else:
        raise TypeError("target returned a KV cache without a safe crop operation")
    if _cache_length(cropped) != length:
        raise RuntimeError("KV crop did not preserve the confirmed-prefix length")
    return cropped


def _clone_cache_for_probe(cache: Any | None) -> Any | None:
    """Copy the *online* pre-verification KV, never recreate it by prefill."""
    if cache is None:
        return None
    cloned = copy.deepcopy(cache)
    if cloned is cache or _cache_length(cloned) != _cache_length(cache):
        raise RuntimeError("could not independently snapshot the online KV cache")
    def storage_pointers(root: Any) -> list[int]:
        seen: set[int] = set()
        pointers: list[int] = []

        def visit(value: Any) -> None:
            if isinstance(value, torch.Tensor):
                if value.numel():
                    pointers.append(value.untyped_storage().data_ptr())
                return
            if value is None or isinstance(value, (str, bytes, int, float, bool)):
                return
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, dict):
                children = value.values()
            elif isinstance(value, (list, tuple, set, frozenset)):
                children = value
            else:
                children = list(vars(value).values()) if hasattr(value, "__dict__") else []
                for cls in type(value).__mro__:
                    slots = cls.__dict__.get("__slots__", ())
                    for slot in (slots,) if isinstance(slots, str) else slots:
                        if slot not in ("__dict__", "__weakref__") and hasattr(value, slot):
                            children.append(getattr(value, slot))
            for child in children:
                visit(child)

        visit(root)
        return pointers

    original_storage = storage_pointers(cache)
    cloned_storage = storage_pointers(cloned)
    if (len(original_storage) != len(cloned_storage)
            or set(original_storage) & set(cloned_storage)):
        raise RuntimeError("online KV snapshot shares tensor storage with the live cache")
    return cloned


@dataclass(frozen=True)
class PackedVerification:
    """All packed decisions, including hypothetical positions after a reject.

    ``argmax_ids[i]`` predicts candidate ``proposals[i]`` from the prefix
    ending immediately before it.  The last ID is the possible bonus.  Only
    the prefix up to the first rejection may be committed to the KV cache.
    """

    prefix_before: tuple[int, ...]
    proposals: tuple[int, ...]
    argmax_ids: tuple[int, ...]
    top2_margins: tuple[float, ...]
    hidden: torch.Tensor = field(repr=False)
    logits: torch.Tensor = field(repr=False, compare=False)
    cache_length_before: int = 0
    cache_length_raw: int = 0
    cache: Any = field(default=None, repr=False, compare=False)
    row_source_query_lengths: tuple[int, ...] = ()
    row_source_modes: tuple[str, ...] = ()
    original_packed_argmax_ids: tuple[int, ...] = ()
    original_packed_top2_ids: tuple[tuple[int, int], ...] = ()
    original_packed_top2_logits_fp32: tuple[tuple[float, float], ...] = ()
    effective_top2_ids: tuple[tuple[int, int], ...] = ()
    effective_top2_logits_fp32: tuple[tuple[float, float], ...] = ()
    original_packed_top2_margins: tuple[float, ...] = ()
    numerical_recheck_ulp_units: tuple[float, ...] = ()
    numerical_recheck_ulp_thresholds: tuple[float, ...] = ()
    numerical_recheck_trigger_reasons: tuple[str, ...] = ()
    numerical_recheck_rows: tuple[int, ...] = ()
    numerical_recheck_extra_target_forwards: int = 0
    numerical_recheck_snapshot_wall_ms: float = 0.0
    numerical_recheck_forward_wall_ms: float = 0.0
    numerical_recheck_total_wall_ms: float = 0.0
    unrechecked_low_margin_bonus_rows: tuple[int, ...] = ()
    decision_cache_row_index: int | None = None
    decision_cache_source: str = "original_packed"
    decision_cache: Any = field(default=None, repr=False, compare=False)

    def hidden_at_valid_prefix(self, accepted_count: int) -> torch.Tensor:
        if not 0 <= accepted_count <= len(self.proposals):
            raise ValueError("invalid accepted count")
        return self.hidden[accepted_count].detach().clone()

    def logits_at_valid_prefix(self, accepted_count: int) -> torch.Tensor:
        """Return only the row that predicted this round's correction/bonus."""
        if not 0 <= accepted_count <= len(self.proposals):
            raise ValueError("invalid accepted count")
        return self.logits[accepted_count].detach().clone()

    def logit_summary_at_valid_prefix(self, accepted_count: int) -> TargetLogitSummary:
        """Summarize the legal row from this verification forward, without replay."""
        row = self.logits_at_valid_prefix(accepted_count).float()
        count = min(32, row.numel())
        if count < 2 or not torch.isfinite(row).all():
            raise FloatingPointError("nonfinite or undersized packed valid-prefix logits")
        values, ids = row.topk(count)
        values16 = values.to(torch.float16)
        if not torch.isfinite(values16).all():
            raise FloatingPointError("packed top logits overflow float16 summary")
        return TargetLogitSummary(
            greedy_token_id=self.argmax_ids[accepted_count],
            top_token_ids=tuple(int(x) for x in ids.cpu().tolist()),
            top_logits_fp16=tuple(float(x) for x in values16.cpu().tolist()),
            logsumexp_fp32=float(row.logsumexp(-1).item()),
            top2_margin_fp32=float((values[0] - values[1]).item()),
            vocab_size=row.numel(),
        )


def check_packed_feature_noninterference(
    target: Any, prefix: list[int], proposals: list[int], altered: list[int],
    accepted_count: int,
) -> dict[str, Any]:
    """Fresh-prefill diagnostic only; online P0 uses the session KV snapshot.

    Both queries have the same length and identical tokens up to the valid
    boundary.  For a fully accepted window, append one extra probe token to
    each query and compare the preceding (formerly bonus) row.  Separate
    sessions prevent a cropped or mutated cache from hiding a dependency.
    """
    if (len(proposals) != len(altered) or not proposals
            or not 0 <= accepted_count < len(proposals)
            or proposals[:accepted_count] != altered[:accepted_count]
            or proposals[accepted_count:] == altered[accepted_count:]):
        raise ValueError("noninterference probe must change only the unconfirmed suffix")
    first = HFPackedTargetSession(target, prefix).verify(proposals)
    second = HFPackedTargetSession(target, prefix).verify(altered)
    hidden_equal = torch.equal(first.hidden_at_valid_prefix(accepted_count),
                               second.hidden_at_valid_prefix(accepted_count))
    logits_equal = torch.equal(first.logits_at_valid_prefix(accepted_count),
                               second.logits_at_valid_prefix(accepted_count))
    evidence = {
        "feature_protocol_version": "fresh_prefill_suffix_diagnostic_v1",
        "source": "fresh_prefill_not_online_KV",
        "valid_prefix": prefix + proposals[:accepted_count],
        "accepted_count": accepted_count,
        "proposal_count": len(proposals),
        "changed_suffix_from": accepted_count,
        "hidden_bitwise_equal": hidden_equal,
        "logits_bitwise_equal": logits_equal,
        "greedy_token_equal": first.argmax_ids[accepted_count] == second.argmax_ids[accepted_count],
    }
    if not (hidden_equal and logits_equal and evidence["greedy_token_equal"]):
        raise RuntimeError(f"packed feature depends on an unconfirmed suffix: {evidence}")
    return evidence


@dataclass(frozen=True)
class PackedRoundTrace:
    prefix_before: tuple[int, ...]
    proposals: tuple[int, ...]
    argmax_ids: tuple[int, ...]
    top2_margins: tuple[float, ...]
    accepted_count: int
    cache_length_before: int
    cache_length_raw: int
    cache_length_after: int
    valid_prefix: tuple[int, ...]
    row_source_query_lengths: tuple[int, ...] = ()
    row_source_modes: tuple[str, ...] = ()
    feature_provenance: dict[str, Any] | None = field(default=None, repr=False,
                                                       compare=False)
    decision_backend: str = "one_forward_packed"
    numerical_recheck_rows: tuple[int, ...] = ()
    numerical_recheck_extra_target_forwards: int = 0
    numerical_recheck_snapshot_wall_ms: float = 0.0
    numerical_recheck_forward_wall_ms: float = 0.0
    numerical_recheck_total_wall_ms: float = 0.0
    unrechecked_low_margin_bonus_rows: tuple[int, ...] = ()
    cache_commit_source: str = "original_packed"
    original_packed_top2_margins: tuple[float, ...] = ()
    original_packed_argmax_ids: tuple[int, ...] = ()
    original_packed_top2_ids: tuple[tuple[int, int], ...] = ()
    original_packed_top2_logits_fp32: tuple[tuple[float, float], ...] = ()
    effective_top2_ids: tuple[tuple[int, int], ...] = ()
    effective_top2_logits_fp32: tuple[tuple[float, float], ...] = ()
    numerical_recheck_ulp_units: tuple[float, ...] = ()
    numerical_recheck_ulp_thresholds: tuple[float, ...] = ()
    numerical_recheck_trigger_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackedRun:
    generation: GenerationResult
    traces: tuple[PackedRoundTrace, ...]


class HFPackedTargetSession:
    """Retain KV for confirmed tokens except their final, unprocessed token.

    At each round the target processes ``[last_confirmed] + proposals`` in
    one forward.  The resulting cache includes rejected branches and is then
    cropped back to ``confirmed_after[:-1]``.  The correction or bonus remains
    the unprocessed final token for the next round; this avoids an extra
    Target forward merely to obtain a next-round cache.
    """

    def __init__(self, target: Any, prompt: list[int], *,
                 capture_online_cache_snapshot: bool = False,
                 numerical_recheck_top2_margin: float | None = None,
                 numerical_recheck_top2_ulps: int | None = None,
                 strict_canonical_used_rows: bool = False) -> None:
        if not prompt:
            raise ValueError("packed target requires a nonempty prompt")
        self.target = target
        self.confirmed = list(prompt)
        self._cache: Any | None = None
        self._pending: PackedVerification | None = None
        self._capture_online_cache_snapshot = capture_online_cache_snapshot
        if (numerical_recheck_top2_margin is not None
                and (not math.isfinite(numerical_recheck_top2_margin)
                     or numerical_recheck_top2_margin < 0)):
            raise ValueError("numerical recheck margin must be finite and nonnegative")
        if (numerical_recheck_top2_ulps is not None
                and (type(numerical_recheck_top2_ulps) is not int
                     or numerical_recheck_top2_ulps <= 0)):
            raise ValueError("numerical recheck ULP count must be a positive integer")
        if (numerical_recheck_top2_margin is not None
                and numerical_recheck_top2_ulps is not None):
            raise ValueError("numerical recheck margin and ULP policies are exclusive")
        if type(strict_canonical_used_rows) is not bool:
            raise TypeError("strict canonical policy must be a bool")
        if strict_canonical_used_rows and (numerical_recheck_top2_margin is not None
                                           or numerical_recheck_top2_ulps is not None):
            raise ValueError("strict canonical and margin/ULP policies are exclusive")
        self.numerical_recheck_top2_margin = numerical_recheck_top2_margin
        self.numerical_recheck_top2_ulps = numerical_recheck_top2_ulps
        self.strict_canonical_used_rows = strict_canonical_used_rows
        self._preverify_cache_snapshot: Any | None = None
        self._preverify_snapshot_valid = False
        self.target._ids(prompt)
        if len(prompt) > 1:
            with torch.inference_mode(), self.target._lock:
                output = self.target.model(
                    input_ids=self.target._ids(prompt[:-1]),
                    use_cache=True, output_hidden_states=False, return_dict=True,
                )
            self._cache = getattr(output, "past_key_values", None)
            if _cache_length(self._cache) != len(prompt) - 1:
                raise RuntimeError("prompt prefill returned an unexpected KV length")

    @property
    def cache_length(self) -> int:
        return _cache_length(self._cache)

    @torch.inference_mode()
    def verify(self, proposals: list[int], *, bonus_needed: bool = True) -> PackedVerification:
        if self._pending is not None:
            raise RuntimeError("previous packed verification has not been committed")
        if not proposals:
            raise ValueError("packed verification requires at least one proposal")
        self.target._ids(self.confirmed + proposals)  # IDs and total context limit
        chunk = [self.confirmed[-1], *proposals]
        chunk_ids = self.target._ids(chunk)
        before = self.cache_length
        if before != len(self.confirmed) - 1:
            raise RuntimeError("KV does not equal the confirmed prefix except its last token")
        recheck_enabled = (self.numerical_recheck_top2_margin is not None
                           or self.numerical_recheck_top2_ulps is not None
                           or self.strict_canonical_used_rows)
        snapshot_wall_ms = 0.0
        if self._capture_online_cache_snapshot or recheck_enabled:
            if recheck_enabled:
                if chunk_ids.device.type == "cuda":
                    torch.cuda.synchronize(chunk_ids.device)
                snapshot_started = time.perf_counter_ns()
            self._preverify_cache_snapshot = _clone_cache_for_probe(self._cache)
            self._preverify_snapshot_valid = True
            if recheck_enabled:
                if chunk_ids.device.type == "cuda":
                    torch.cuda.synchronize(chunk_ids.device)
                snapshot_wall_ms = (
                    time.perf_counter_ns() - snapshot_started) / 1_000_000

        norm_outputs: list[torch.Tensor] = []

        def capture_norm(_module: torch.nn.Module, _inputs: tuple[Any, ...],
                         output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("final norm returned a non-tensor")
            norm_outputs.append(output)

        with self.target._lock:
            hook = self.target.model.model.norm.register_forward_hook(capture_norm)
            try:
                output = self.target.model(
                    input_ids=chunk_ids,
                    past_key_values=self._cache,
                    use_cache=True, output_hidden_states=True, return_dict=True,
                )
            finally:
                hook.remove()
        logits = getattr(output, "logits", None)
        states = getattr(output, "hidden_states", None)
        cache = getattr(output, "past_key_values", None)
        if (not isinstance(logits, torch.Tensor) or logits.shape[:2] != (1, len(chunk))
                or logits.shape[-1] != self.target.vocab_size):
            raise RuntimeError("packed logits have an unexpected shape")
        if (not states or len(norm_outputs) != 1
                or not isinstance(states[-1], torch.Tensor)
                or not torch.equal(states[-1], norm_outputs[0])
                or states[-1].shape != (1, len(chunk), self.target.hidden_size)):
            raise RuntimeError("packed final hidden does not match the final decoder norm")
        if not torch.isfinite(logits).all() or not torch.isfinite(states[-1]).all():
            raise FloatingPointError("nonfinite packed logits or hidden")
        raw_length = _cache_length(cache)
        expected = len(self.confirmed) + len(proposals)
        if raw_length != expected:
            raise RuntimeError(f"packed KV has {raw_length} tokens, expected {expected}")
        original_top2 = torch.topk(logits[0].float(), 2, dim=-1)
        top2 = original_top2.values
        argmax_ids = [int(x) for x in logits[0].argmax(dim=-1).tolist()]
        original_argmax_ids = tuple(argmax_ids)
        original_top2_ids = tuple(
            (int(row[0]), int(row[1])) for row in original_top2.indices.tolist())
        original_top2_logits = tuple(
            (float(row[0]), float(row[1])) for row in top2.tolist())
        margins = [float(x) for x in (top2[:, 0] - top2[:, 1]).tolist()]
        original_margins = tuple(margins)
        ulp_units = (
            [_top2_max_ulp(first, second, logits.dtype)
             for first, second in original_top2_logits]
            if self.numerical_recheck_top2_ulps is not None else [])
        ulp_thresholds = (
            [self.numerical_recheck_top2_ulps * unit for unit in ulp_units]
            if self.numerical_recheck_top2_ulps is not None else [])
        selected_hidden = states[-1][0].detach().clone()
        selected_logits = logits[0].detach().clone()
        source_lengths = [len(chunk)] * len(chunk)
        source_modes = ["original_packed"] * len(chunk)
        rechecked: list[int] = []
        trigger_reasons = ["none"] * len(chunk)
        unrechecked_bonus: list[int] = []
        extra_forward_count = 0
        extra_forward_wall_ms = 0.0
        total_recheck_wall_ms = snapshot_wall_ms
        decision_cache_row_index: int | None = None
        decision_cache: Any | None = None
        decision_cache_source = "original_packed"
        if recheck_enabled:
            if not self._preverify_snapshot_valid:
                raise RuntimeError("numeric recheck requires the online KV snapshot")
            # Only rows that can enter the committed trajectory are visited.
            # A wrong accepted row must be reconsidered before the next row.
            canonical_chain_active = False
            for row_index in range(len(chunk)):
                threshold = (ulp_thresholds[row_index]
                             if self.numerical_recheck_top2_ulps is not None
                             else self.numerical_recheck_top2_margin)
                ambiguous = (not self.strict_canonical_used_rows
                             and margins[row_index] <= threshold)
                if row_index == len(proposals) and not bonus_needed:
                    if ambiguous:
                        unrechecked_bonus.append(row_index)
                    break
                if (self.strict_canonical_used_rows or ambiguous
                        or (self.numerical_recheck_top2_ulps is not None
                            and canonical_chain_active)):
                    trigger_reasons[row_index] = (
                        "strict" if self.strict_canonical_used_rows else
                        "chain" if canonical_chain_active else "margin")
                    recheck_started = time.perf_counter_ns()
                    if (row_index < len(proposals)
                            and self.numerical_recheck_top2_ulps is None
                            and not self.strict_canonical_used_rows):
                        query = chunk[:row_index + 1]
                        query_ids = self.target._ids(query)
                        fork_cache = _clone_cache_for_probe(
                            self._preverify_cache_snapshot)
                        mode = "online_kv_short_query"
                    else:
                        # V2 uses the independent AR query shape for every
                        # ambiguous *used* row.  A short online-KV query can
                        # retain the BF16 discrepancy.  Once V2 enters this
                        # path, all later used rows in the same round are
                        # checked canonically so accepted decisions cannot
                        # splice a packed numerical history.  Every forward
                        # and cache refill is charged; this is not free.
                        query = self.confirmed + proposals[:row_index]
                        query_ids = self.target._ids(query)
                        fork_cache = None
                        mode = "canonical_full_prefix"
                        if self.numerical_recheck_top2_ulps is not None:
                            canonical_chain_active = True
                    if query_ids.device.type == "cuda":
                        torch.cuda.synchronize(query_ids.device)
                    started = time.perf_counter_ns()
                    with self.target._lock:
                        if mode == "canonical_full_prefix" and callable(
                                getattr(self.target, "_forward_with_final_hidden", None)):
                            replay, final = self.target._forward_with_final_hidden(
                                query_ids)
                            replay_states = (final,)
                        else:
                            norm_outputs: list[torch.Tensor] = []
                            def capture_norm(_module: torch.nn.Module, _inputs: tuple[Any, ...],
                                             output: Any) -> None:
                                norm_outputs.append(output)
                            hook = self.target.model.model.norm.register_forward_hook(
                                capture_norm)
                            try:
                                replay = self.target.model(
                                    input_ids=query_ids, past_key_values=fork_cache,
                                    use_cache=(mode == "online_kv_short_query"),
                                    output_hidden_states=True, return_dict=True)
                            finally:
                                hook.remove()
                            replay_states = getattr(replay, "hidden_states", None)
                            if (not replay_states or len(norm_outputs) != 1
                                    or not isinstance(norm_outputs[0], torch.Tensor)
                                    or not torch.equal(replay_states[-1], norm_outputs[0])):
                                raise RuntimeError("numeric recheck hidden is not final norm")
                    if query_ids.device.type == "cuda":
                        torch.cuda.synchronize(query_ids.device)
                    extra_forward_wall_ms += (
                        time.perf_counter_ns() - started) / 1_000_000
                    extra_forward_count += 1
                    if mode == "canonical_full_prefix":
                        # The no-cache full-prefix decision has no KV output.
                        # Refill the *same* confirmed prefix for the next
                        # online round, charged as a second Target forward.
                        if query_ids.device.type == "cuda":
                            torch.cuda.synchronize(query_ids.device)
                        refill_started = time.perf_counter_ns()
                        with self.target._lock:
                            refill = self.target.model(
                                input_ids=query_ids, use_cache=True,
                                output_hidden_states=False, return_dict=True)
                        if query_ids.device.type == "cuda":
                            torch.cuda.synchronize(query_ids.device)
                        extra_forward_wall_ms += (
                            time.perf_counter_ns() - refill_started) / 1_000_000
                        extra_forward_count += 1
                        cache_candidate = getattr(refill, "past_key_values", None)
                        if _cache_length(cache_candidate) != len(query):
                            raise RuntimeError("canonical bonus refill returned wrong KV length")
                    else:
                        cache_candidate = getattr(replay, "past_key_values", None)
                    replay_logits = getattr(replay, "logits", None)
                    replay_row = (row_index if mode == "online_kv_short_query"
                                  else len(query) - 1)
                    if (not isinstance(replay_logits, torch.Tensor)
                            or replay_logits.shape != (1, len(query), self.target.vocab_size)
                            or not replay_states
                            or replay_states[-1].shape != (
                                1, len(query), self.target.hidden_size)
                            or (mode == "online_kv_short_query"
                                and _cache_length(replay.past_key_values) != before + len(query))):
                        raise RuntimeError("numeric recheck returned invalid shapes or KV")
                    row_logits = replay_logits[0, replay_row].detach()
                    row_hidden = replay_states[-1][0, replay_row].detach()
                    if not (torch.isfinite(row_logits).all()
                            and torch.isfinite(row_hidden).all()):
                        raise FloatingPointError("nonfinite numeric recheck row")
                    selected_logits[row_index] = row_logits
                    selected_hidden[row_index] = row_hidden
                    argmax_ids[row_index] = int(row_logits.argmax().item())
                    values = torch.topk(row_logits.float(), 2).values
                    margins[row_index] = float((values[0] - values[1]).item())
                    source_lengths[row_index] = len(query)
                    source_modes[row_index] = mode
                    rechecked.append(row_index)
                    decision_cache_row_index = row_index
                    decision_cache = cache_candidate
                    decision_cache_source = (
                        "canonical_full_prefix_refill"
                        if mode == "canonical_full_prefix" else "online_short_recheck")
                    total_recheck_wall_ms += (
                        time.perf_counter_ns() - recheck_started) / 1_000_000
                if row_index < len(proposals) and argmax_ids[row_index] != proposals[row_index]:
                    break
        effective_top2 = torch.topk(selected_logits.float(), 2, dim=-1)
        effective_top2_ids = tuple(
            (int(row[0]), int(row[1])) for row in effective_top2.indices.tolist())
        effective_top2_logits = tuple(
            (float(row[0]), float(row[1])) for row in effective_top2.values.tolist())
        verification = PackedVerification(
            prefix_before=tuple(self.confirmed),
            proposals=tuple(proposals),
            argmax_ids=tuple(argmax_ids),
            top2_margins=tuple(margins),
            hidden=selected_hidden,
            logits=selected_logits,
            cache_length_before=before,
            cache_length_raw=raw_length,
            cache=cache,
            row_source_query_lengths=tuple(source_lengths),
            row_source_modes=tuple(source_modes),
            original_packed_argmax_ids=original_argmax_ids,
            original_packed_top2_ids=original_top2_ids,
            original_packed_top2_logits_fp32=original_top2_logits,
            effective_top2_ids=effective_top2_ids,
            effective_top2_logits_fp32=effective_top2_logits,
            original_packed_top2_margins=original_margins,
            numerical_recheck_ulp_units=tuple(ulp_units),
            numerical_recheck_ulp_thresholds=tuple(ulp_thresholds),
            numerical_recheck_trigger_reasons=tuple(trigger_reasons),
            numerical_recheck_rows=tuple(rechecked),
            numerical_recheck_extra_target_forwards=extra_forward_count,
            numerical_recheck_snapshot_wall_ms=snapshot_wall_ms,
            numerical_recheck_forward_wall_ms=extra_forward_wall_ms,
            numerical_recheck_total_wall_ms=total_recheck_wall_ms,
            unrechecked_low_margin_bonus_rows=tuple(unrechecked_bonus),
            decision_cache_row_index=decision_cache_row_index,
            decision_cache_source=decision_cache_source,
            decision_cache=decision_cache,
        )
        self._pending = verification
        return verification

    def check_online_suffix_noninterference(
        self, verification: PackedVerification, accepted_count: int,
    ) -> dict[str, Any]:
        """Compare a changed suffix against the original online KV history.

        The snapshot is taken immediately before the original packed forward.
        An independent Target prefill from token IDs is never used as the
        causal comparison because BF16 KV values may depend on that history.
        """
        if (verification is not self._pending or not self._preverify_snapshot_valid
                or not 0 <= accepted_count <= len(verification.proposals)):
            raise RuntimeError("online feature intervention lacks the pending KV snapshot")
        selected_source_mode = (
            verification.row_source_modes[accepted_count]
            if verification.row_source_modes else "original_packed")
        evidence: dict[str, Any] = {
            "feature_protocol_version": PACKED_FEATURE_PROTOCOL_VERSION,
            "source": {
                "original_packed": "original_packed_verify_forward",
                "online_kv_short_query": "numeric_recheck_query_forward",
                "canonical_full_prefix": "canonical_full_prefix_numeric_recheck",
            }[selected_source_mode],
            "cache_source": (
                "fresh_full_prefix_no_KV" if selected_source_mode == "canonical_full_prefix"
                else "same_online_preverify_KV_snapshot"),
            "accepted_count": accepted_count,
            "proposal_count": len(verification.proposals),
            "cache_length_before": verification.cache_length_before,
            "numeric_backend": (NUMERICAL_RECHECK_BACKEND
                                if self.numerical_recheck_top2_margin is not None
                                else (NUMERICAL_RECHECK_BACKEND_V2
                                      if self.numerical_recheck_top2_ulps is not None
                                      else (STRICT_CANONICAL_BACKEND
                                            if self.strict_canonical_used_rows
                                            else "one_forward_packed"))),
            "numerical_recheck_rows": list(verification.numerical_recheck_rows),
            "numerical_recheck_extra_target_forwards": (
                verification.numerical_recheck_extra_target_forwards),
            "numerical_recheck_snapshot_wall_ms": (
                verification.numerical_recheck_snapshot_wall_ms),
            "numerical_recheck_forward_wall_ms": (
                verification.numerical_recheck_forward_wall_ms),
            "numerical_recheck_total_wall_ms": (
                verification.numerical_recheck_total_wall_ms),
            "diagnostic_feature_gate_target_forwards": (
                1 if accepted_count in verification.numerical_recheck_rows
                or accepted_count == len(verification.proposals) else 2),
        }
        if accepted_count in verification.numerical_recheck_rows:
            # The selected feature comes from a costed extra forward. Replay
            # exactly that source, not the original L+1 packed row.
            if selected_source_mode == "online_kv_short_query":
                query = [self.confirmed[-1],
                         *verification.proposals[:accepted_count]]
                cache = _clone_cache_for_probe(self._preverify_cache_snapshot)
                row_index = accepted_count
            elif selected_source_mode == "canonical_full_prefix":
                query = self.confirmed + list(
                    verification.proposals[:accepted_count])
                cache = None
                row_index = len(query) - 1
            else:
                raise RuntimeError("rechecked row lacks a recorded source mode")
            with self.target._lock, torch.inference_mode():
                if (selected_source_mode == "canonical_full_prefix"
                        and callable(getattr(self.target, "_forward_with_final_hidden", None))):
                    replay, final = self.target._forward_with_final_hidden(
                        self.target._ids(query))
                    replay_hidden = final[0, row_index]
                else:
                    replay = self.target.model(
                        input_ids=self.target._ids(query),
                        past_key_values=cache,
                        use_cache=(selected_source_mode == "online_kv_short_query"),
                        output_hidden_states=True, return_dict=True)
                    replay_hidden = replay.hidden_states[-1][0, row_index]
            replay_logits = replay.logits[0, row_index]
            original_hidden = verification.hidden_at_valid_prefix(accepted_count)
            original_logits = verification.logits_at_valid_prefix(accepted_count)
            hidden_equal = torch.equal(original_hidden, replay_hidden)
            logits_equal = torch.equal(original_logits, replay_logits)
            greedy_equal = (verification.argmax_ids[accepted_count]
                            == int(replay_logits.argmax().item()))
            proof = {**evidence, "applicable": False,
                    "reason": ("canonical_full_prefix_ends_at_valid_row"
                               if selected_source_mode == "canonical_full_prefix"
                               else "rechecked_query_ends_at_valid_row"),
                    "valid_row_query_length": len(query),
                    "feature_replay_hidden_bitwise_equal": hidden_equal,
                    "feature_replay_logits_bitwise_equal": logits_equal,
                    "feature_replay_greedy_token_equal": greedy_equal,
                    "passed": hidden_equal and logits_equal and greedy_equal}
            if selected_source_mode == "online_kv_short_query":
                proof.update({
                    "snapshot_replay_hidden_bitwise_equal": hidden_equal,
                    "snapshot_replay_logits_bitwise_equal": logits_equal,
                    "snapshot_replay_greedy_token_equal": greedy_equal,
                })
            return proof
        def fork_verify(proposals: list[int]) -> PackedVerification:
            fork = object.__new__(type(self))
            fork.target = self.target
            fork.confirmed = self.confirmed.copy()
            fork._cache = _clone_cache_for_probe(self._preverify_cache_snapshot)
            fork._pending = None
            fork._capture_online_cache_snapshot = False
            fork._preverify_cache_snapshot = None
            fork._preverify_snapshot_valid = False
            fork.numerical_recheck_top2_margin = None
            fork.numerical_recheck_top2_ulps = None
            fork.strict_canonical_used_rows = False
            return fork.verify(proposals)

        # First prove that a fork of the online KV reproduces the original
        # valid row. A fresh prefill can differ numerically even with the
        # same token IDs; it is never used for this baseline.
        baseline = fork_verify(list(verification.proposals))
        original_hidden = verification.hidden_at_valid_prefix(accepted_count)
        original_logits = verification.logits_at_valid_prefix(accepted_count)
        snapshot_hidden_equal = torch.equal(
            original_hidden, baseline.hidden_at_valid_prefix(accepted_count))
        snapshot_logits_equal = torch.equal(
            original_logits, baseline.logits_at_valid_prefix(accepted_count))
        snapshot_greedy_equal = (verification.argmax_ids[accepted_count]
                                 == baseline.argmax_ids[accepted_count])
        snapshot_equal = (snapshot_hidden_equal and snapshot_logits_equal
                          and snapshot_greedy_equal)
        evidence.update({
            "snapshot_replay_hidden_bitwise_equal": snapshot_hidden_equal,
            "snapshot_replay_logits_bitwise_equal": snapshot_logits_equal,
            "snapshot_replay_greedy_token_equal": snapshot_greedy_equal,
        })
        if accepted_count == len(verification.proposals):
            # The original packed query ends at the bonus row. There is no
            # later unconfirmed input in that query to intervene on.
            return {**evidence, "applicable": False, "passed": snapshot_equal,
                    "reason": "no_unconfirmed_suffix_in_original_query"}
        altered = list(verification.proposals)
        altered[accepted_count] = (altered[accepted_count] + 1) % self.target.vocab_size
        alternate = fork_verify(altered)
        alternate_hidden = alternate.hidden_at_valid_prefix(accepted_count)
        alternate_logits = alternate.logits_at_valid_prefix(accepted_count)
        hidden_equal = torch.equal(baseline.hidden_at_valid_prefix(accepted_count),
                                   alternate_hidden)
        logits_equal = torch.equal(baseline.logits_at_valid_prefix(accepted_count),
                                   alternate_logits)
        greedy_equal = (baseline.argmax_ids[accepted_count]
                        == alternate.argmax_ids[accepted_count])
        return {**evidence, "applicable": True,
                "changed_suffix_from": accepted_count,
                "same_query_length": len(verification.proposals) == len(altered),
                "hidden_bitwise_equal": hidden_equal,
                "logits_bitwise_equal": logits_equal,
                "greedy_token_equal": greedy_equal,
                "hidden_max_abs_diff_fp32": float((original_hidden.float()
                    - alternate_hidden.float()).abs().max().item()),
                "logits_max_abs_diff_fp32": float((original_logits.float()
                    - alternate_logits.float()).abs().max().item()),
                "passed": snapshot_equal and hidden_equal and logits_equal and greedy_equal}

    def commit(self, verification: PackedVerification, accepted_count: int,
               answer: int | None) -> int:
        if verification is not self._pending:
            raise RuntimeError("can only commit the current packed verification")
        if not 0 <= accepted_count <= len(verification.proposals):
            raise ValueError("invalid accepted count")
        if any(verification.proposals[i] != verification.argmax_ids[i]
               for i in range(accepted_count)):
            raise ValueError("an unverified draft token cannot enter the KV cache")
        if accepted_count < len(verification.proposals):
            if verification.proposals[accepted_count] == verification.argmax_ids[accepted_count]:
                raise ValueError("stopped before the first rejection")
            if answer != verification.argmax_ids[accepted_count]:
                raise ValueError("correction is not the packed Target decision")
        elif answer is not None and answer != verification.argmax_ids[-1]:
            raise ValueError("bonus is not the packed Target decision")
        if answer is not None and not 0 <= answer < self.target.vocab_size:
            raise ValueError("invalid correction or bonus token")
        confirmed_after = self.confirmed + list(verification.proposals[:accepted_count])
        if answer is not None:
            confirmed_after.append(answer)
        desired = len(confirmed_after) - 1
        # A numerically rechecked decision is paired with the KV produced by
        # that same effective query. Cropping the original long packed cache
        # here would silently splice two BF16 execution histories.
        if accepted_count == verification.decision_cache_row_index:
            if verification.decision_cache is None:
                raise RuntimeError("rechecked valid row has no matching KV")
            commit_cache = verification.decision_cache
        else:
            commit_cache = verification.cache
        self._cache = _crop_cache(commit_cache, desired)
        self.confirmed = confirmed_after
        self._pending = None
        self._preverify_cache_snapshot = None
        self._preverify_snapshot_valid = False
        return desired


class PackedSpeculativeDecoder:
    """Closed-loop greedy SD using the candidate packed Target backend."""

    def __init__(self, target: TargetBackend, draft: DraftBackend,
                 encoder: FeedbackEncoderProtocol | None, window: int = 8,
                 eos_token_id: int | None = None,
                 capture_target_logits: bool = False,
                 feature_provenance_gate: bool = False,
                 numerical_recheck_top2_margin: float | None = None,
                 numerical_recheck_top2_ulps: int | None = None,
                 strict_canonical_used_rows: bool = False) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        if (numerical_recheck_top2_margin is not None
                and (not math.isfinite(numerical_recheck_top2_margin)
                     or numerical_recheck_top2_margin < 0)):
            raise ValueError("numerical recheck margin must be finite and nonnegative")
        if (numerical_recheck_top2_ulps is not None
                and (type(numerical_recheck_top2_ulps) is not int
                     or numerical_recheck_top2_ulps <= 0)):
            raise ValueError("numerical recheck ULP count must be a positive integer")
        if (numerical_recheck_top2_margin is not None
                and numerical_recheck_top2_ulps is not None):
            raise ValueError("numerical recheck margin and ULP policies are exclusive")
        if type(strict_canonical_used_rows) is not bool:
            raise TypeError("strict canonical policy must be a bool")
        if strict_canonical_used_rows and (numerical_recheck_top2_margin is not None
                                           or numerical_recheck_top2_ulps is not None):
            raise ValueError("strict canonical and margin/ULP policies are exclusive")
        self.target = target
        self.draft = draft
        self.encoder = encoder
        self.window = window
        self.eos_token_id = eos_token_id
        self.capture_target_logits = capture_target_logits or feature_provenance_gate
        self.feature_provenance_gate = feature_provenance_gate
        self.numerical_recheck_top2_margin = numerical_recheck_top2_margin
        self.numerical_recheck_top2_ulps = numerical_recheck_top2_ulps
        self.strict_canonical_used_rows = strict_canonical_used_rows

    def generate(self, prompt: list[int], max_new_tokens: int) -> PackedRun:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if max_new_tokens == 0:
            return PackedRun(GenerationResult((), ()), ())
        session = HFPackedTargetSession(
            self.target, prompt,
            capture_online_cache_snapshot=self.feature_provenance_gate,
            numerical_recheck_top2_margin=self.numerical_recheck_top2_margin,
            numerical_recheck_top2_ulps=self.numerical_recheck_top2_ulps,
            strict_canonical_used_rows=self.strict_canonical_used_rows)
        generated: list[int] = []
        rounds: list[RoundRecord] = []
        traces: list[PackedRoundTrace] = []
        codes: tuple[int, ...] | None = None
        while len(generated) < max_new_tokens:
            prefix_before = tuple(session.confirmed)
            remaining = max_new_tokens - len(generated)
            proposal: list[int] = []
            for position in range(1, min(self.window, remaining) + 1):
                token = int(self.draft.next(session.confirmed + proposal, codes, position))
                proposal.append(token)
                if token == self.eos_token_id:
                    break
            bonus_needed = (proposal[-1] != self.eos_token_id
                            and len(proposal) < remaining)
            verification = session.verify(proposal, bonus_needed=bonus_needed)
            accepted = 0
            while (accepted < len(proposal)
                   and proposal[accepted] == verification.argmax_ids[accepted]):
                accepted += 1
            feature_provenance = (session.check_online_suffix_noninterference(
                verification, accepted) if self.feature_provenance_gate else None)
            rejected = proposal[accepted] if accepted < len(proposal) else None
            correction = verification.argmax_ids[accepted] if rejected is not None else None
            draft_eos = accepted == len(proposal) and proposal[-1] == self.eos_token_id
            has_budget_for_bonus = len(generated) + accepted < max_new_tokens
            bonus = (verification.argmax_ids[-1]
                     if rejected is None and not draft_eos and has_budget_for_bonus else None)
            answer = correction if correction is not None else bonus
            committed = list(proposal[:accepted])
            if answer is not None:
                committed.append(answer)
            if not committed:
                raise RuntimeError("packed round made no progress")
            generated.extend(committed)
            cache_after = session.commit(verification, accepted, answer)
            stopped_on_eos = generated[-1] == self.eos_token_id
            observation = (verification.hidden_at_valid_prefix(accepted)
                           if answer is not None else None)
            logit_summary = (verification.logit_summary_at_valid_prefix(accepted)
                             if self.capture_target_logits and answer is not None else None)
            if logit_summary is not None and logit_summary.greedy_token_id != answer:
                raise RuntimeError("packed valid-prefix logits do not predict the committed answer")
            if feature_provenance is not None:
                feature_provenance = {**feature_provenance,
                    "decision_feature_available": answer is not None,
                    "same_forward_hidden": (
                        torch.equal(observation,
                                    verification.hidden_at_valid_prefix(accepted))
                        if observation is not None else None),
                    "same_forward_logit_summary": (
                        logit_summary == verification.logit_summary_at_valid_prefix(accepted)
                        if logit_summary is not None else None),
                }
            codes_out: tuple[int, ...] | None = None
            if self.encoder is not None and not stopped_on_eos and len(generated) < max_new_tokens:
                if observation is None or answer is None:
                    raise RuntimeError("nonterminal packed round has no legal feedback state")
                codes_out = tuple(self.encoder.encode(
                    observation.detach(),
                    None if rejected is None else self.target.embedding(rejected).detach(),
                    self.target.embedding(answer).detach(), accepted,
                ))
                if len(codes_out) != 8 or any(type(code) is not int or not 0 <= code <= 255
                                               for code in codes_out):
                    raise ValueError("feedback encoder must return exactly eight byte indices")
            rounds.append(RoundRecord(
                prefix_before=prefix_before, codes_in=codes,
                proposals=tuple(proposal), accepted_count=accepted,
                committed_tokens=tuple(committed), rejected_token=rejected,
                correction_token=correction, bonus_token=bonus,
                observation_hidden=observation, feedback_codes=codes_out,
                message_bytes=8 if codes_out is not None else 0,
                observation_logit_summary=logit_summary,
            ))
            traces.append(PackedRoundTrace(
                prefix_before=prefix_before, proposals=tuple(proposal),
                argmax_ids=verification.argmax_ids,
                top2_margins=verification.top2_margins,
                accepted_count=accepted,
                cache_length_before=verification.cache_length_before,
                cache_length_raw=verification.cache_length_raw,
                cache_length_after=cache_after,
                valid_prefix=prefix_before + tuple(proposal[:accepted]),
                row_source_query_lengths=verification.row_source_query_lengths,
                row_source_modes=verification.row_source_modes,
                feature_provenance=feature_provenance,
                decision_backend=(NUMERICAL_RECHECK_BACKEND
                                  if self.numerical_recheck_top2_margin is not None
                                  else (NUMERICAL_RECHECK_BACKEND_V2
                                        if self.numerical_recheck_top2_ulps is not None
                                        else (STRICT_CANONICAL_BACKEND
                                              if self.strict_canonical_used_rows
                                              else "one_forward_packed"))),
                numerical_recheck_rows=verification.numerical_recheck_rows,
                numerical_recheck_extra_target_forwards=(
                    verification.numerical_recheck_extra_target_forwards),
                numerical_recheck_snapshot_wall_ms=(
                    verification.numerical_recheck_snapshot_wall_ms),
                numerical_recheck_forward_wall_ms=(
                    verification.numerical_recheck_forward_wall_ms),
                numerical_recheck_total_wall_ms=(
                    verification.numerical_recheck_total_wall_ms),
                unrechecked_low_margin_bonus_rows=(
                    verification.unrechecked_low_margin_bonus_rows),
                cache_commit_source=(
                    verification.decision_cache_source
                    if accepted == verification.decision_cache_row_index
                    else "original_packed"),
                original_packed_top2_margins=(
                    verification.original_packed_top2_margins),
                original_packed_argmax_ids=(
                    verification.original_packed_argmax_ids),
                original_packed_top2_ids=(
                    verification.original_packed_top2_ids),
                original_packed_top2_logits_fp32=(
                    verification.original_packed_top2_logits_fp32),
                effective_top2_ids=verification.effective_top2_ids,
                effective_top2_logits_fp32=(
                    verification.effective_top2_logits_fp32),
                numerical_recheck_ulp_units=(
                    verification.numerical_recheck_ulp_units),
                numerical_recheck_ulp_thresholds=(
                    verification.numerical_recheck_ulp_thresholds),
                numerical_recheck_trigger_reasons=(
                    verification.numerical_recheck_trigger_reasons),
            ))
            codes = codes_out
            if stopped_on_eos:
                break
        return PackedRun(GenerationResult(tuple(generated), tuple(rounds)), tuple(traces))
