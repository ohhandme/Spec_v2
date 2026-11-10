"""Fixed corrected-prefix probe for the P1 feedback-information experiment.

All examples are collected with the *raw* draft policy.  Every arm is then
evaluated on those same prefixes and independent target greedy continuations.
Only the view allowed for an arm reaches its preparation function.  In
particular, future labels never reach an arm, and a shuffled control replaces
only D's eight-byte cloud message, retaining the recipient's local state.

This is an acceptance-length diagnostic.  The full-prefix backends execute
sequential target decisions and establish neither throughput nor WAN latency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import defaultdict
import random
from statistics import mean, median
from typing import Any, Callable, Iterable, Protocol

import torch

from .decoding import TargetLogitSummary, SpeculativeDecoder, autoregressive_generate


EVAL_PROTOCOL_VERSION = "fixed_prefix_v1"
_SPLITS = frozenset({"train", "val", "test", "smoke"})
_ARMS = frozenset({"raw_draft", "edge_only", "protocol_only", "target_8B"})


@dataclass(frozen=True)
class ProbePrompt:
    id: str
    family_id: str
    split: str
    input_ids: tuple[int, ...]
    task_type: str = "unknown"

    def __post_init__(self) -> None:
        if not self.id or not self.family_id or self.split not in _SPLITS:
            raise ValueError("prompt needs a nonempty ID/family and a known split")
        if not self.input_ids or any(type(x) is not int or x < 0 for x in self.input_ids):
            raise ValueError("input_ids must be nonempty nonnegative integer IDs")
        if not self.task_type:
            raise ValueError("task_type must be nonempty")


@dataclass(frozen=True)
class EdgeView:
    """Information available at the edge even without a protocol feature."""

    corrected_prefix: tuple[int, ...]
    window: int


@dataclass(frozen=True)
class ProtocolView(EdgeView):
    """Ordinary verification facts already known to the edge."""

    accepted_count: int
    rejected_token: int | None
    correction_token: int


@dataclass(frozen=True)
class TargetView(ProtocolView):
    """The one valid target state that predicted the correction/bonus token."""

    target_hidden: torch.Tensor


@dataclass(frozen=True)
class TargetLogitView(ProtocolView):
    """Only same-boundary logit summary, never future Target logits."""

    target_logit_summary: TargetLogitSummary


@dataclass(frozen=True)
class ProbeObservation:
    prompt_id: str
    family_id: str
    split: str
    task_type: str
    round_index: int
    source_prefix: tuple[int, ...]
    proposals: tuple[int, ...]
    valid_prefix: tuple[int, ...]
    accepted_count: int
    rejected_token: int | None
    correction_token: int
    corrected_prefix: tuple[int, ...]
    target_hidden: torch.Tensor
    target_hidden_prefix: tuple[int, ...]
    remaining_budget: int
    window: int
    target_logit_summary: TargetLogitSummary | None = None

    @property
    def key(self) -> tuple[str, int]:
        return self.prompt_id, self.round_index

    def edge_view(self) -> EdgeView:
        return EdgeView(self.corrected_prefix, self.window)

    def protocol_view(self) -> ProtocolView:
        return ProtocolView(
            self.corrected_prefix, self.window, self.accepted_count,
            self.rejected_token, self.correction_token,
        )

    def target_view(self) -> TargetView:
        return TargetView(
            self.corrected_prefix, self.window, self.accepted_count,
            self.rejected_token, self.correction_token,
            self.target_hidden.detach().clone(),
        )

    def target_logit_view(self) -> TargetLogitView:
        if self.target_logit_summary is None:
            raise ValueError("this probe lacks a same-boundary Target logit summary")
        if self.target_logit_summary.greedy_token_id != self.correction_token:
            raise ValueError("Target logit summary does not predict the recorded correction")
        return TargetLogitView(
            self.corrected_prefix, self.window, self.accepted_count,
            self.rejected_token, self.correction_token, self.target_logit_summary,
        )


@dataclass(frozen=True)
class ProbeExample:
    observation: ProbeObservation
    target_future_ids: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        o = self.observation
        row = {
            "prompt_id": o.prompt_id,
            "family_id": o.family_id,
            "split": o.split,
            "task_type": o.task_type,
            "round_index": o.round_index,
            "source_prefix": list(o.source_prefix),
            "proposals": list(o.proposals),
            "valid_prefix": list(o.valid_prefix),
            "accepted_count": o.accepted_count,
            "rejected_token": o.rejected_token,
            "correction_token": o.correction_token,
            "corrected_prefix": list(o.corrected_prefix),
            "target_hidden_prefix": list(o.target_hidden_prefix),
            "target_hidden_dtype": str(o.target_hidden.dtype),
            "target_hidden": o.target_hidden.detach().cpu().tolist(),
            "remaining_budget": o.remaining_budget,
            "window": o.window,
            "target_future_ids": list(self.target_future_ids),
        }
        if o.target_logit_summary is not None:
            row["target_logit_summary"] = o.target_logit_summary.to_json()
        return row

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "ProbeExample":
        dtype_by_name = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
        }
        dtype = dtype_by_name.get(row.get("target_hidden_dtype"))
        if dtype is None:
            raise ValueError("unknown recorded target hidden dtype")
        hidden = torch.tensor(row["target_hidden"], dtype=dtype)
        if hidden.ndim != 1 or not hidden.numel() or not torch.isfinite(hidden).all():
            raise ValueError("recorded target hidden must be a finite vector")
        observation = ProbeObservation(
            prompt_id=str(row["prompt_id"]), family_id=str(row["family_id"]),
            split=str(row["split"]), task_type=str(row["task_type"]),
            round_index=int(row["round_index"]),
            source_prefix=tuple(row["source_prefix"]),
            proposals=tuple(row["proposals"]),
            valid_prefix=tuple(row["valid_prefix"]),
            accepted_count=int(row["accepted_count"]),
            rejected_token=row["rejected_token"],
            correction_token=int(row["correction_token"]),
            corrected_prefix=tuple(row["corrected_prefix"]),
            target_hidden=hidden,
            target_hidden_prefix=tuple(row["target_hidden_prefix"]),
            remaining_budget=int(row["remaining_budget"]),
            window=int(row["window"]),
            target_logit_summary=(
                TargetLogitSummary.from_json(row["target_logit_summary"])
                if "target_logit_summary" in row else None
            ),
        )
        return cls(observation, tuple(row["target_future_ids"]))


@dataclass(frozen=True)
class ProbeCorpus:
    examples: tuple[ProbeExample, ...]
    audit: dict[str, int]


def _validate_prompts(prompts: list[ProbePrompt]) -> None:
    if not prompts:
        raise ValueError("no prompts supplied")
    ids: set[str] = set()
    tokens: set[tuple[int, ...]] = set()
    family_split: dict[str, str] = {}
    for prompt in prompts:
        if prompt.id in ids or prompt.input_ids in tokens:
            raise ValueError("duplicate prompt ID or token sequence")
        if prompt.family_id in family_split and family_split[prompt.family_id] != prompt.split:
            raise ValueError("one prompt family crosses train/val/test splits")
        ids.add(prompt.id)
        tokens.add(prompt.input_ids)
        family_split[prompt.family_id] = prompt.split


def collect_fixed_probe(
    *,
    target: Any,
    raw_draft: Any,
    prompts: Iterable[ProbePrompt],
    window: int = 8,
    max_new_tokens: int = 128,
    eos_token_id: int | None = None,
    max_probe_windows_per_prompt: int | None = None,
    capture_target_logits: bool = False,
) -> ProbeCorpus:
    """Collect raw-draft boundaries and independent target AR labels once.

    The target state is rechecked at the valid prefix *before* the correction
    token.  We never store a state from a rejected suffix or after the
    correction.  Terminal/budget-exhausted rounds have no next-window probe.
    """
    prompts = list(prompts)
    _validate_prompts(prompts)
    if window <= 0 or max_new_tokens <= 0:
        raise ValueError("window and max_new_tokens must be positive")
    if max_probe_windows_per_prompt is not None and max_probe_windows_per_prompt <= 0:
        raise ValueError("max_probe_windows_per_prompt must be positive")
    decoder = SpeculativeDecoder(
        target, raw_draft, None, window, eos_token_id,
        capture_target_logits=capture_target_logits,
    )
    examples: list[ProbeExample] = []
    audit = {"prompts": len(prompts), "raw_rounds": 0, "terminal_or_budget_rounds": 0,
             "probe_windows": 0, "full_nonterminal_windows": 0}
    for prompt in prompts:
        prompt_windows = 0
        result = decoder.generate(list(prompt.input_ids), max_new_tokens)
        target_ar = autoregressive_generate(
            target, list(prompt.input_ids), max_new_tokens, eos_token_id
        )
        if result.tokens != target_ar:
            raise RuntimeError(f"raw speculative output differs from target AR: {prompt.id}")
        audit["raw_rounds"] += len(result.rounds)
        for round_index, row in enumerate(result.rounds):
            if (max_probe_windows_per_prompt is not None
                    and prompt_windows >= max_probe_windows_per_prompt):
                break
            correction = row.correction_token if row.rejected_token is not None else row.bonus_token
            if correction is None:
                audit["terminal_or_budget_rounds"] += 1
                continue
            valid = row.prefix_before + row.proposals[: row.accepted_count]
            corrected = valid + (correction,)
            if corrected != row.prefix_before + row.committed_tokens:
                raise RuntimeError("the corrected prefix does not match the committed tokens")
            if row.rejected_token is not None and (
                row.accepted_count >= len(row.proposals)
                or row.rejected_token != row.proposals[row.accepted_count]
            ):
                raise RuntimeError("the recorded first rejection is inconsistent")
            if row.observation_hidden is None:
                raise RuntimeError("a nonterminal correction lacks its target observation")
            # This extra forward is deliberately a data-integrity gate, not a
            # throughput measurement. It proves the feature predicts b.
            # This is an offline integrity re-forward, never the source of
            # F-LOGIT's online feature. That feature must be captured by the
            # original verification forward and recorded in RoundRecord.
            checked = target.next(list(valid))
            if checked.token_id != correction or not torch.equal(
                checked.hidden.detach().cpu(), row.observation_hidden.detach().cpu()
            ):
                raise RuntimeError("target feature was not observed at the valid prefix")
            summary = row.observation_logit_summary
            if capture_target_logits and (
                summary is None or summary.greedy_token_id != correction
            ):
                raise RuntimeError("normal verification lacks a legal current-boundary logit summary")
            used = len(corrected) - len(prompt.input_ids)
            remaining = max_new_tokens - used
            if correction == eos_token_id or remaining <= 0:
                audit["terminal_or_budget_rounds"] += 1
                continue
            # The independent request-level target AR was generated before
            # examining any arm.  Since this corrected prefix must equal its
            # exact leading tokens, its suffix is the independent greedy AR
            # future from this prefix.  Slicing avoids repeating up to L
            # expensive 8B full-prefix forwards at every baseline round.
            if corrected != prompt.input_ids + target_ar[:used]:
                raise RuntimeError("corrected prefix is not on the independent AR path")
            future = target_ar[used : used + min(window, remaining)]
            if not future:
                audit["terminal_or_budget_rounds"] += 1
                continue
            observation = ProbeObservation(
                prompt_id=prompt.id, family_id=prompt.family_id, split=prompt.split,
                task_type=prompt.task_type, round_index=round_index,
                source_prefix=row.prefix_before, proposals=row.proposals,
                valid_prefix=valid, accepted_count=row.accepted_count,
                rejected_token=row.rejected_token, correction_token=correction,
                corrected_prefix=corrected,
                target_hidden=row.observation_hidden.detach().cpu().clone(),
                target_hidden_prefix=valid,
                remaining_budget=remaining, window=window,
                target_logit_summary=summary if capture_target_logits else None,
            )
            examples.append(ProbeExample(observation, future))
            prompt_windows += 1
            audit["probe_windows"] += 1
            if len(future) == window and (eos_token_id is None or eos_token_id not in future):
                audit["full_nonterminal_windows"] += 1
    return ProbeCorpus(tuple(examples), audit)


@dataclass(frozen=True)
class PreparedState:
    """Local state is recipient-specific; message is D's raw eight bytes."""

    local: Any = None
    message: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.message is not None and (
            not isinstance(self.message, tuple) or len(self.message) != 8
            or any(type(code) is not int or not 0 <= code < 256 for code in self.message)
        ):
            raise ValueError("the cloud message must contain exactly eight byte indices")


class ProbeNext(Protocol):
    def __call__(self, prefix: list[int], state: PreparedState, position: int) -> int: ...


@dataclass(frozen=True)
class ProbeArm:
    """`prepare` sees only the view allowed for this arm.

    The same frozen draft backbone must be supplied to all arms by the runner.
    P1's model/hash manifest provides the auditable proof of that constraint.
    """

    prepare: Callable[[EdgeView | ProtocolView | TargetView], PreparedState]
    next: ProbeNext


def raw_draft_arm(draft: Any) -> ProbeArm:
    return ProbeArm(
        prepare=lambda _view: PreparedState(),
        next=lambda prefix, _state, position: int(draft.next(prefix, None, position)),
    )


def matched_p1_arm(draft: Any, codec: Any) -> ProbeArm:
    """Adapt a trained MatchedP1Codec to genuine AR fixed-prefix inference.

    `teacher_hidden(prefix, [0])` computes the final-normalized draft state
    for the already generated prefix only.  The dummy continuation token is
    never included in that model input.  C/D protocol embeddings come from
    the frozen *draft* input table, because those IDs are ordinary edge-known
    information.  The target state is passed only to D's cloud encoder.
    """
    mode = getattr(codec, "mode", None)
    if mode not in ("edge_only", "protocol_only", "target_8B"):
        raise ValueError("matched_p1_arm requires a B/C/D codec mode")
    if not hasattr(draft, "teacher_hidden") or not hasattr(draft, "model"):
        raise TypeError("draft must provide teacher_hidden and a frozen model")
    embedding_table = draft.model.get_input_embeddings().weight

    def embedding(token_id: int | None) -> torch.Tensor | None:
        if token_id is None:
            return None
        if type(token_id) is not int or token_id < 0 or token_id >= embedding_table.shape[0]:
            raise ValueError("protocol token ID is outside the draft vocabulary")
        return embedding_table[token_id].detach().reshape(1, -1)

    @torch.inference_mode()
    def prepare(view: EdgeView | ProtocolView | TargetView) -> PreparedState:
        if (mode == "edge_only" and type(view) is not EdgeView) or (
            mode == "protocol_only" and type(view) is not ProtocolView
        ) or (mode == "target_8B" and type(view) is not TargetView):
            raise TypeError("arm received information outside its permitted view")
        local_hidden = draft.teacher_hidden(list(view.corrected_prefix), [0])[0]
        if mode == "edge_only":
            local = codec.local_encode(local_hidden.reshape(1, -1))
            cloud_message = None
        else:
            assert isinstance(view, ProtocolView)
            local = codec.local_encode(
                local_hidden.reshape(1, -1),
                embedding(view.rejected_token),
                embedding(view.correction_token),
                view.accepted_count,
            )
            if mode == "target_8B":
                assert isinstance(view, TargetView)
                cloud_codes = codec.cloud_encode(view.target_hidden.reshape(1, -1))
                cloud_message = tuple(int(code) for code in cloud_codes[0].tolist())
            else:
                cloud_message = None
        local_codes = tuple(
            tuple(int(code) for code in group) for group in local[0].tolist()
        )
        if (len(local_codes) != 2 or any(len(group) != 8 for group in local_codes)
                or any(not 0 <= code < 256 for group in local_codes for code in group)):
            raise RuntimeError("local encoder did not produce two eight-index controls")
        return PreparedState(local=local_codes, message=cloud_message)

    @torch.inference_mode()
    def next_token(prefix: list[int], state: PreparedState, position: int) -> int:
        hidden = draft.teacher_hidden(prefix, [0])[0].reshape(1, -1)
        code_device = codec.repair.codebooks.device
        local_codes = torch.tensor([state.local], dtype=torch.long, device=code_device)
        cloud_codes = (
            torch.tensor([state.message], dtype=torch.long, device=code_device)
            if state.message is not None else None
        )
        logits = codec.repair_logits(hidden, local_codes, position, cloud_codes)
        if logits.shape != (1, codec.config.vocab_size) or not torch.isfinite(logits).all():
            raise RuntimeError("matched probe returned invalid logits")
        return int(logits[0].argmax(-1).item())

    return ProbeArm(prepare=prepare, next=next_token)


def _view_for(name: str, observation: ProbeObservation) -> EdgeView | ProtocolView | TargetView:
    if name in ("raw_draft", "edge_only"):
        return observation.edge_view()
    if name == "protocol_only":
        return observation.protocol_view()
    if name == "target_8B":
        return observation.target_view()
    raise ValueError(f"unknown fixed-probe arm: {name}")


def _accepted_prefix(draft_ids: tuple[int, ...], target_ids: tuple[int, ...]) -> int:
    accepted = 0
    for left, right in zip(draft_ids, target_ids):
        if left != right:
            break
        accepted += 1
    return accepted


def _generate_probe(
    arm: ProbeArm, observation: ProbeObservation, state: PreparedState,
    eos_token_id: int | None,
) -> tuple[int, ...]:
    generated: list[int] = []
    limit = min(observation.window, observation.remaining_budget)
    for position in range(1, limit + 1):
        # The next call sees the *actual generated* draft prefix.  There is no
        # teacher forcing, no target future label, and no target model call.
        token = arm.next(list(observation.corrected_prefix + tuple(generated)), state, position)
        if type(token) is not int or token < 0:
            raise ValueError("probe draft returned an invalid token ID")
        generated.append(token)
        if token == eos_token_id:
            break
    return tuple(generated)


def _deranged_prompt_donors(examples: list[ProbeExample], seed: int) -> dict[str, str]:
    prompts = sorted({x.observation.prompt_id for x in examples})
    if len(prompts) < 2:
        raise ValueError("cross-request shuffle needs at least two prompt IDs")
    rng = random.Random(seed)
    donors = prompts.copy()
    rng.shuffle(donors)
    if any(left == right for left, right in zip(prompts, donors)):
        offset = rng.randrange(1, len(prompts))
        donors = prompts[offset:] + prompts[:offset]
    return dict(zip(prompts, donors))


def _prompt_grouped_summary(rows: list[dict], *, primary_only: bool) -> dict[str, Any] | None:
    selected = [row for row in rows if row["primary_eligible"] or not primary_only]
    if not selected:
        return None
    by_arm_prompt: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        by_arm_prompt[row["arm"]][row["prompt_id"]].append(row["accepted_count"])
    arms = sorted(by_arm_prompt)
    prompt_sets = {arm: set(by_arm_prompt[arm]) for arm in arms}
    if any(ids != prompt_sets[arms[0]] for ids in prompt_sets.values()):
        raise RuntimeError("probe arms were evaluated on different prompt sets")
    return {
        arm: {
            "prompts": len(by_arm_prompt[arm]),
            "windows": sum(map(len, by_arm_prompt[arm].values())),
            "mean_prompt_accept": mean(mean(values) for values in by_arm_prompt[arm].values()),
            "median_prompt_accept": median(mean(values) for values in by_arm_prompt[arm].values()),
        }
        for arm in arms
    }


def _paired_prompt_ci(
    rows: list[dict], left: str, right: str, *, primary_only: bool,
    bootstrap_repetitions: int, seed: int,
    allowed_keys: set[tuple[str, int]] | None = None,
) -> dict[str, Any] | None:
    selected = [
        row for row in rows
        if (row["primary_eligible"] or not primary_only)
        and (allowed_keys is None or (row["prompt_id"], row["round_index"]) in allowed_keys)
    ]
    by_arm_prompt: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        by_arm_prompt[row["arm"]][row["prompt_id"]].append(row["accepted_count"])
    ids = sorted(set(by_arm_prompt[left]) & set(by_arm_prompt[right]))
    if not ids:
        return None
    if set(by_arm_prompt[left]) != set(by_arm_prompt[right]):
        raise RuntimeError("paired arms have different prompt IDs")
    differences = [mean(by_arm_prompt[left][id_]) - mean(by_arm_prompt[right][id_]) for id_ in ids]
    rng = random.Random(seed)
    draws = sorted(
        mean(differences[rng.randrange(len(ids))] for _ in ids)
        for _ in range(bootstrap_repetitions)
    )
    return {
        "left_minus_right": f"{left}-{right}",
        "prompt_count": len(ids),
        "mean_prompt_delta": mean(differences),
        "ci95": [draws[int(0.025 * (bootstrap_repetitions - 1))],
                 draws[int(0.975 * (bootstrap_repetitions - 1))]],
        "bootstrap_repetitions": bootstrap_repetitions,
    }


def evaluate_fixed_probe(
    *,
    examples: Iterable[ProbeExample],
    arms: dict[str, ProbeArm],
    eos_token_id: int | None,
    split: str = "val",
    allow_test: bool = False,
    shuffle_seed: int = 0,
    bootstrap_repetitions: int = 2000,
) -> dict[str, Any]:
    """Evaluate A/B/C/D at identical frozen prefixes; add D-message shuffles.

    The caller is responsible for identical training budgets and a shared
    frozen draft model. `test` requires an explicit opt-in after val selection.
    """
    examples = list(examples)
    if not examples or set(arms) != _ARMS:
        raise ValueError("nonempty examples and exactly A/B/C/D arms are required")
    if split not in _SPLITS or (split == "test" and not allow_test):
        raise ValueError("test is sealed unless allow_test=True")
    if bootstrap_repetitions < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    keys: set[tuple[str, int]] = set()
    for example in examples:
        o = example.observation
        if o.split != split or o.key in keys:
            raise ValueError("examples have a wrong split or duplicate probe key")
        keys.add(o.key)
        if o.window <= 0 or o.remaining_budget <= 0 or (
            o.accepted_count < 0 or o.accepted_count > len(o.proposals)
        ):
            raise ValueError("probe has an invalid window, budget, or accepted_count")
        if o.valid_prefix != o.source_prefix + o.proposals[:o.accepted_count]:
            raise ValueError("probe valid prefix does not match the accepted proposal")
        if o.rejected_token is None:
            if o.accepted_count != len(o.proposals):
                raise ValueError("full acceptance record has a missing accepted token")
        elif (o.accepted_count >= len(o.proposals)
              or o.rejected_token != o.proposals[o.accepted_count]):
            raise ValueError("probe first rejection is inconsistent")
        if o.target_hidden_prefix != o.valid_prefix or (
            o.corrected_prefix != o.valid_prefix + (o.correction_token,)
        ):
            raise ValueError("target feature/prefix integrity check failed")
        if not example.target_future_ids or len(example.target_future_ids) > min(
            o.window, o.remaining_budget
        ) or any(type(token) is not int or token < 0 for token in example.target_future_ids):
            raise ValueError("a probe needs a bounded target AR continuation")
        if (len(example.target_future_ids) < min(o.window, o.remaining_budget)
                and (eos_token_id is None or example.target_future_ids[-1] != eos_token_id)):
            raise ValueError("short target continuation must end at EOS")
    prepared: dict[tuple[str, tuple[str, int]], PreparedState] = {}
    for example in examples:
        for name, arm in arms.items():
            state = arm.prepare(_view_for(name, example.observation))
            if not isinstance(state, PreparedState):
                raise TypeError("arm.prepare must return PreparedState")
            if name == "target_8B" and state.message is None:
                raise ValueError("target_8B must produce an eight-byte cloud message")
            if name != "target_8B" and state.message is not None:
                raise ValueError("non-target arm cannot send a cloud message")
            prepared[name, example.observation.key] = state
    donor_prompts = _deranged_prompt_donors(examples, shuffle_seed)
    by_prompt: dict[str, list[ProbeExample]] = defaultdict(list)
    for example in examples:
        by_prompt[example.observation.prompt_id].append(example)
    for stream in by_prompt.values():
        stream.sort(key=lambda x: x.observation.round_index)
    rows: list[dict] = []

    def record(name: str, example: ProbeExample, state: PreparedState,
               donor: tuple[str, int] | None = None) -> None:
        o = example.observation
        arm = arms["target_8B"] if name.startswith("shuffled_") else arms[name]
        draft_ids = _generate_probe(arm, o, state, eos_token_id)
        eligible = len(example.target_future_ids) == o.window and (
            eos_token_id is None or eos_token_id not in example.target_future_ids
        )
        accepted = _accepted_prefix(draft_ids, example.target_future_ids)
        rows.append({
            "arm": name, "prompt_id": o.prompt_id, "round_index": o.round_index,
            "task_type": o.task_type, "prior_accepted_count": o.accepted_count,
            "accepted_count": accepted,
            "first_rejection_position": (
                accepted + 1
                if accepted < min(len(draft_ids), len(example.target_future_ids)) else None
            ),
            "draft_ids": list(draft_ids),
            "target_future_ids": list(example.target_future_ids),
            "primary_eligible": eligible,
            "donor": list(donor) if donor is not None else None,
        })

    for example in examples:
        for name in ("raw_draft", "edge_only", "protocol_only", "target_8B"):
            record(name, example, prepared[name, example.observation.key])
        recipient = example.observation
        donor_id = donor_prompts[recipient.prompt_id]
        donor_stream = by_prompt[donor_id]
        donor_example = donor_stream[recipient.round_index % len(donor_stream)]
        donor_key = donor_example.observation.key
        own = prepared["target_8B", recipient.key]
        borrowed = prepared["target_8B", donor_key]
        record("shuffled_global_8B", example, replace(own, message=borrowed.message), donor_key)

    # Stratification is a required negative-control audit.  If the frozen
    # sample does not contain another prompt in every stratum, report its
    # unavailability instead of silently using same-request or wrong strata.
    strata: dict[tuple[str, int], list[ProbeExample]] = defaultdict(list)
    for example in examples:
        o = example.observation
        strata[o.task_type, o.accepted_count].append(example)
    stratified_eligible = [
        example for example in examples
        if any(candidate.observation.prompt_id != example.observation.prompt_id
               for candidate in strata[example.observation.task_type,
                                       example.observation.accepted_count])
    ]
    stratified_keys = {example.observation.key for example in stratified_eligible}
    if stratified_eligible:
        rng = random.Random(shuffle_seed + 1)
        for example in stratified_eligible:
            o = example.observation
            candidates = [candidate for candidate in strata[o.task_type, o.accepted_count]
                          if candidate.observation.prompt_id != o.prompt_id]
            donor = rng.choice(candidates).observation.key
            own = prepared["target_8B", o.key]
            borrowed = prepared["target_8B", donor]
            record("shuffled_stratified_8B", example,
                   replace(own, message=borrowed.message), donor)

    full_rows = [row for row in rows if row["arm"] != "shuffled_stratified_8B"]
    primary_summary = _prompt_grouped_summary(full_rows, primary_only=True)
    sensitivity_summary = _prompt_grouped_summary(full_rows, primary_only=False)
    stratified_pair_rows = [
        row for row in rows
        if row["arm"] in ("target_8B", "shuffled_stratified_8B")
        and (row["prompt_id"], row["round_index"]) in stratified_keys
    ]
    stratified_primary_summary = (
        _prompt_grouped_summary(stratified_pair_rows, primary_only=True)
        if stratified_keys else None
    )
    comparisons = {}
    for left, right in (("target_8B", "protocol_only"),
                        ("target_8B", "edge_only"),
                        ("target_8B", "raw_draft"),
                        ("target_8B", "shuffled_global_8B")):
        comparisons[f"{left}-{right}"] = _paired_prompt_ci(
            rows, left, right, primary_only=True,
            bootstrap_repetitions=bootstrap_repetitions, seed=shuffle_seed,
        )
    if stratified_keys:
        comparisons["target_8B-shuffled_stratified_8B"] = _paired_prompt_ci(
            rows, "target_8B", "shuffled_stratified_8B", primary_only=True,
            bootstrap_repetitions=bootstrap_repetitions, seed=shuffle_seed,
            allowed_keys=stratified_keys,
        )
    eligible_keys = {(row["prompt_id"], row["round_index"])
                     for row in rows if row["arm"] == "raw_draft" and row["primary_eligible"]}
    rejection_distribution: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    task_summary: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["primary_eligible"]:
            rejection_distribution[row["arm"]][
                str(row["first_rejection_position"])
                if row["first_rejection_position"] is not None else "none"
            ] += 1
            task_summary[row["arm"]][row["task_type"]].append(row["accepted_count"])
    return {
        "scope": "fixed_prefix_probe",
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        "latency_claim_valid": False,
        "split": split,
        "probe_windows": len(examples),
        "primary_windows": len(eligible_keys),
        "excluded_fraction": 1 - len(eligible_keys) / len(examples),
        "stratified_shuffle_status": (
            "complete_coverage" if len(stratified_keys) == len(examples)
            else "eligible_subset_only" if stratified_keys
            else "unavailable_insufficient_cross_prompt_donors"
        ),
        "stratified_shuffle_coverage": {
            "windows": len(stratified_keys),
            "prompts": len({prompt_id for prompt_id, _ in stratified_keys}),
            "excluded_windows": len(examples) - len(stratified_keys),
            "fraction": len(stratified_keys) / len(examples),
            "primary_windows": len(stratified_keys & eligible_keys),
            "task_type_source": "prompt_manifest_or_unknown",
        },
        "rows": rows,
        "primary_summary": primary_summary,
        "all_windows_sensitivity_summary": sensitivity_summary,
        "stratified_eligible_primary_summary": stratified_primary_summary,
        "primary_first_rejection_distribution": {
            arm: dict(counts) for arm, counts in rejection_distribution.items()
        },
        "primary_task_type_mean_accept": {
            arm: {task: mean(values) for task, values in groups.items()}
            for arm, groups in task_summary.items()
        },
        "primary_prompt_paired_comparisons": comparisons,
    }
