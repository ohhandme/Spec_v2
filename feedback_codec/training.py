"""Offline, causally separated training for the eight-byte feedback channel.

The target continuation is collected only after a verification round has ended.
It is passed to the edge-side teacher-forcing loss, never to the cloud encoder.
This is deliberately a reference implementation: the frozen model backends may
recompute full prefixes and do not establish a cloud/edge latency result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Protocol

import torch

from .codec import MatchedP1Codec
from .decoding import SpeculativeDecoder, autoregressive_generate
from .fixed_probe import ProbeExample
from .losses import feedback_loss


@dataclass(frozen=True)
class TrajectoryExample:
    prompt_id: str
    round_index: int
    phase: str
    valid_prefix: tuple[int, ...]
    rejected_token: int | None
    correction_token: int
    accepted_count: int
    corrected_prefix: tuple[int, ...]
    continuation: tuple[int, ...]

    def to_json(self) -> dict:
        return asdict(self)


class TeacherForcedDraft(Protocol):
    def teacher_hidden(
        self, prefix: list[int], continuation: list[int]
    ) -> torch.Tensor:
        """Final-normalized hidden states predicting each continuation token."""


class CodecEncoder:
    """Inference adapter; only the four permitted verification fields cross it."""

    def __init__(self, codec: torch.nn.Module) -> None:
        self.codec = codec

    @torch.no_grad()
    def encode(
        self,
        target_hidden: torch.Tensor,
        rejected_embedding: torch.Tensor | None,
        correction_embedding: torch.Tensor,
        accepted_count: int,
    ) -> tuple[int, ...]:
        parameter = self.codec.target_projection.weight
        hidden = target_hidden.reshape(1, -1).to(
            device=parameter.device, dtype=parameter.dtype
        )
        rejected = (
            None
            if rejected_embedding is None
            else rejected_embedding.reshape(1, -1).to(
                device=parameter.device, dtype=parameter.dtype
            )
        )
        correction = correction_embedding.reshape(1, -1).to(
            device=parameter.device, dtype=parameter.dtype
        )
        codes = self.codec.encode(hidden, rejected, correction, accepted_count)
        payload = self.codec.pack_codes(codes.reshape(-1))
        if not isinstance(payload, bytes) or len(payload) != 8:
            raise RuntimeError("feedback did not serialize to exactly eight bytes")
        return tuple(int(x) for x in self.codec.unpack_codes(payload).tolist())


def collect_trajectories(
    decoder: SpeculativeDecoder,
    prompts: Iterable[tuple[str, list[int]]],
    *,
    max_new_tokens: int,
    phase: str,
) -> list[TrajectoryExample]:
    """Collect actual accepted boundaries and independent future AR labels.

    A round ending the request has no subsequent window to repair, so it does
    not form a training example.  The target future is generated from the
    *corrected* prefix, never from rejected draft tokens.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if not phase:
        raise ValueError("phase must be named")
    examples: list[TrajectoryExample] = []
    seen_ids: set[str] = set()
    for prompt_id, prompt in prompts:
        if prompt_id in seen_ids:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        seen_ids.add(prompt_id)
        result = decoder.generate(prompt, max_new_tokens)
        oracle = autoregressive_generate(
            decoder.target, prompt, max_new_tokens, decoder.eos_token_id
        )
        if result.tokens != oracle:
            raise RuntimeError(f"target-token mismatch on prompt {prompt_id}")
        for round_index, record in enumerate(result.rounds):
            b = (
                record.correction_token
                if record.correction_token is not None
                else record.bonus_token
            )
            # The no-feedback baseline has no outgoing code even when another
            # round follows.  Next-round existence, not code presence, decides
            # whether a useful future training window exists.
            if b is None or round_index + 1 >= len(result.rounds):
                continue
            valid = record.prefix_before + record.proposals[: record.accepted_count]
            corrected = valid + (b,)
            if tuple(record.prefix_before + record.committed_tokens) != corrected:
                raise RuntimeError("verification boundary was not preserved")
            continuation = autoregressive_generate(
                decoder.target,
                list(corrected),
                decoder.window,
                decoder.eos_token_id,
            )
            if not continuation:
                continue
            examples.append(
                TrajectoryExample(
                    prompt_id=prompt_id,
                    round_index=round_index,
                    phase=phase,
                    valid_prefix=valid,
                    rejected_token=record.rejected_token,
                    correction_token=b,
                    accepted_count=record.accepted_count,
                    corrected_prefix=corrected,
                    continuation=continuation,
                )
            )
    return examples


def train_examples(
    *,
    examples: list[TrajectoryExample],
    target,
    draft: TeacherForcedDraft,
    codec: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: int,
    prefix_weight: float,
    temperature: float = 1.0,
    seed: int = 0,
) -> list[dict[str, float | int | str]]:
    """Train only codec parameters, with one teacher-forced window per step."""
    if steps < 0 or prefix_weight < 0 or temperature <= 0:
        raise ValueError("invalid training schedule")
    if steps and not examples:
        raise ValueError("no trajectories to train on")
    for parameter in getattr(target, "model", torch.nn.Module()).parameters():
        if parameter.requires_grad:
            raise ValueError("target model must be frozen")
    for parameter in getattr(draft, "model", torch.nn.Module()).parameters():
        if parameter.requires_grad:
            raise ValueError("draft model must be frozen")
    trainable = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not trainable or any(not p.requires_grad for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("optimizer must contain only trainable codec parameters")
    if any(id(p) not in {id(q) for q in codec.parameters() if q.requires_grad} for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("optimizer contains parameters outside the codec")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(examples), generator=generator).tolist()
    cursor = 0
    history: list[dict[str, float | int | str]] = []
    codec.train()
    for step in range(1, steps + 1):
        if cursor == len(order):
            order = torch.randperm(len(examples), generator=generator).tolist()
            cursor = 0
        item = examples[order[cursor]]
        cursor += 1
        with torch.no_grad():
            target_step = target.next(list(item.valid_prefix))
            if int(target_step.token_id) != item.correction_token:
                raise RuntimeError("stored correction no longer matches frozen target")
            # HF target.next uses inference_mode. Clone into an ordinary
            # no-grad tensor before a trainable Linear saves it for backward.
            target_hidden = target_step.hidden.detach().clone().reshape(1, -1)
            rejected = (
                None
                if item.rejected_token is None
                else target.embedding(item.rejected_token).detach().clone().reshape(1, -1)
            )
            correction = target.embedding(item.correction_token).detach().clone().reshape(1, -1)
            draft_hidden = draft.teacher_hidden(
                list(item.corrected_prefix), list(item.continuation)
            ).detach()
        if draft_hidden.ndim != 2 or draft_hidden.shape[0] != len(item.continuation):
            raise RuntimeError("teacher hidden states do not align with labels")
        device = draft_hidden.device
        target_hidden = target_hidden.to(device=device, dtype=torch.float32)
        rejected = None if rejected is None else rejected.to(device=device, dtype=torch.float32)
        correction = correction.to(device=device, dtype=torch.float32)
        st = codec.sample_st(
            target_hidden,
            rejected,
            correction,
            item.accepted_count,
            temperature=temperature,
        )
        logits = torch.stack(
            [
                codec.repair_logits(draft_hidden[j : j + 1], st, j + 1).squeeze(0)
                for j in range(len(item.continuation))
            ],
            dim=0,
        ).unsqueeze(0)
        labels = torch.tensor(item.continuation, device=logits.device).unsqueeze(0)
        losses = feedback_loss(logits, labels, prefix_weight=prefix_weight)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        for parameter in codec.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite codec gradient")
        optimizer.step()
        history.append(
            {
                "step": step,
                "prompt_id": item.prompt_id,
                "round_index": item.round_index,
                "phase": item.phase,
                "loss": float(losses["total"].detach().cpu()),
                "cross_entropy": float(losses["cross_entropy"].detach().cpu()),
                "prefix_surrogate": float(losses["prefix_surrogate"].detach().cpu()),
            }
        )
    return history


def train_matched_p1_examples(
    *,
    examples: list[TrajectoryExample | ProbeExample],
    target,
    draft: TeacherForcedDraft,
    codec: MatchedP1Codec,
    optimizer: torch.optim.Optimizer,
    steps: int,
    prefix_weight: float,
    temperature: float = 1.0,
    seed: int = 0,
    resume_state: dict | None = None,
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Train one P1 arm on the same locked baseline trajectories.

    The local control comes from the *corrected* prefix through the frozen
    draft. For C/D, ordinary protocol token IDs are embedded with the draft's
    own frozen input table. Only D reads the valid-prefix target hidden. When
    given locked ProbeExample rows, that state is taken from their validated
    raw-baseline cache and the 8B target is not rerun every optimization step.
    The target continuation is used solely by the teacher-forcing loss.

    This auxiliary loss is not an acceptance-rate measurement; the locked
    fixed-prefix autoregressive probe must decide which arm is retained.
    """
    if steps < 0 or prefix_weight < 0 or temperature <= 0:
        raise ValueError("invalid training schedule")
    if checkpoint_every < 0 or (checkpoint_callback is not None and checkpoint_every == 0):
        raise ValueError("checkpoint callback requires positive checkpoint_every")
    if steps and not examples:
        raise ValueError("no trajectories to train on")
    if target is not None and any(p.requires_grad for p in target.model.parameters()):
        raise ValueError("target model must be frozen")
    if any(p.requires_grad for p in draft.model.parameters()):
        raise ValueError("draft model must be frozen")
    active = {id(p) for p in codec.parameters() if p.requires_grad}
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if not active or len(optimized) != len({id(p) for p in optimized}) or (
        {id(p) for p in optimized} != active
    ):
        raise ValueError("optimizer must contain exactly the active codec parameters")
    if codec.mode != "edge_only":
        table = draft.model.get_input_embeddings()
        if table is None or any(p.requires_grad for p in table.parameters()):
            raise ValueError("draft input embeddings must exist and be frozen")
    else:
        table = None

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if resume_state is None:
        order = torch.randperm(len(examples), generator=generator).tolist()
        cursor = 0
        completed_steps = 0
        history: list[dict] = []
    else:
        completed_steps = resume_state.get("completed_steps")
        order = resume_state.get("order")
        cursor = resume_state.get("cursor")
        saved_generator = resume_state.get("generator_state")
        history = resume_state.get("history")
        if (type(completed_steps) is not int or not 0 <= completed_steps <= steps
                or not isinstance(order, list)
                or sorted(order) != list(range(len(examples)))
                or type(cursor) is not int or not 0 <= cursor <= len(order)
                or not isinstance(saved_generator, torch.Tensor)
                or saved_generator.dtype != torch.uint8
                or not isinstance(history, list) or len(history) != completed_steps
                or any(not isinstance(row, dict) or row.get("step") != index
                       for index, row in enumerate(history, 1))):
            raise ValueError("invalid P1 training resume state")
        generator.set_state(saved_generator.cpu())
    codec.train()
    for step in range(completed_steps + 1, steps + 1):
        if cursor == len(order):
            order = torch.randperm(len(examples), generator=generator).tolist()
            cursor = 0
        item = examples[order[cursor]]
        cursor += 1
        if isinstance(item, ProbeExample):
            observation = item.observation
            prompt_id = observation.prompt_id
            round_index = observation.round_index
            phase = observation.split
            valid_prefix = observation.valid_prefix
            corrected_prefix = observation.corrected_prefix
            rejected_token = observation.rejected_token
            correction_token = observation.correction_token
            accepted_count = observation.accepted_count
            continuation = item.target_future_ids
            cached_target_hidden = observation.target_hidden
            if observation.target_hidden_prefix != valid_prefix:
                raise ValueError("cached target state is not from the valid prefix")
        else:
            prompt_id = item.prompt_id
            round_index = item.round_index
            phase = item.phase
            valid_prefix = item.valid_prefix
            corrected_prefix = item.corrected_prefix
            rejected_token = item.rejected_token
            correction_token = item.correction_token
            accepted_count = item.accepted_count
            continuation = item.continuation
            cached_target_hidden = None
        if phase not in ("train", "smoke"):
            raise ValueError("P1 training cannot use validation or sealed test probes")
        if (valid_prefix + (correction_token,) != corrected_prefix
                or not continuation):
            raise ValueError("trajectory has an invalid corrected prefix or no continuation")
        with torch.no_grad():
            if cached_target_hidden is None:
                if target is None:
                    raise ValueError("uncached trajectories require a target backend")
                verification = target.next(list(valid_prefix))
                if int(verification.token_id) != correction_token:
                    raise RuntimeError("stored correction no longer matches frozen target")
            # A one-token teacher call evaluates the corrected prefix only:
            # the fixed sentinel is never input to the transformer.
            local_hidden = draft.teacher_hidden(
                list(corrected_prefix), [0]
            ).detach().clone()
            draft_hidden = draft.teacher_hidden(
                list(corrected_prefix), list(continuation)
            ).detach()
            if codec.mode == "target_8B":
                source_hidden = (verification.hidden if cached_target_hidden is None
                                 else cached_target_hidden)
                target_hidden = source_hidden.detach().clone().reshape(1, -1)
            else:
                target_hidden = None
            if table is not None:
                embedding_device = next(table.parameters()).device
                correction = table(torch.tensor(
                    [correction_token], dtype=torch.long, device=embedding_device
                )).detach().clone()
                rejected = None if rejected_token is None else table(torch.tensor(
                    [rejected_token], dtype=torch.long, device=embedding_device
                )).detach().clone()
            else:
                correction = rejected = None
        if (local_hidden.shape != (1, codec.config.draft_hidden_size)
                or draft_hidden.shape != (len(continuation), codec.config.draft_hidden_size)):
            raise RuntimeError("draft hidden states do not align with the fixed window")
        device = draft_hidden.device
        local_hidden = local_hidden.to(device=device)
        correction = None if correction is None else correction.to(device=device)
        rejected = None if rejected is None else rejected.to(device=device)
        target_hidden = None if target_hidden is None else target_hidden.to(device=device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed + step)
            local_st = codec.local_sample_st(
                local_hidden, rejected, correction,
                None if codec.mode == "edge_only" else accepted_count,
                temperature=temperature,
            )
            cloud_st = None if target_hidden is None else codec.cloud_sample_st(
                target_hidden, temperature=temperature,
            )
        logits = torch.stack([
            codec.repair_logits(draft_hidden[j : j + 1], local_st,
                                j + 1, cloud_st).squeeze(0)
            for j in range(len(continuation))
        ], dim=0).unsqueeze(0)
        labels = torch.tensor(continuation, device=logits.device).unsqueeze(0)
        losses = feedback_loss(logits, labels, prefix_weight=prefix_weight)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in codec.parameters()):
            raise FloatingPointError("non-finite P1 codec gradient")
        optimizer.step()
        history.append({
            "step": step,
            "prompt_id": prompt_id,
            "round_index": round_index,
            "phase": phase,
            "mode": codec.mode,
            "loss": float(losses["total"].detach().cpu()),
            "cross_entropy": float(losses["cross_entropy"].detach().cpu()),
            "prefix_surrogate": float(losses["prefix_surrogate"].detach().cpu()),
            "train_hard_gumbel_local_codes": local_st.argmax(dim=-1)[0].detach().cpu().tolist(),
            "train_hard_gumbel_cloud_codes": None if cloud_st is None else
                cloud_st.argmax(dim=-1)[0].detach().cpu().tolist(),
        })
        if (checkpoint_callback is not None
                and (step % checkpoint_every == 0 or step == steps)):
            checkpoint_callback({
                "completed_steps": step,
                "order": order.copy(),
                "cursor": cursor,
                "generator_state": generator.get_state().clone(),
                "history": history.copy(),
            })
    return history
