"""Local socket reference for causally delayed cloud-to-edge feedback.

The cloud owns the target and encoder; the edge owns the draft.  This is a
correctness and wire-format reference, not a performance implementation: the
cloud uses sequential full-prefix target calls and the loopback timings cannot
establish acceleration or wide-area latency.  The eight-byte *feedback payload*
is separate from the length frame and JSON control metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import struct
from typing import Any

from .decoding import (
    DraftBackend,
    FeedbackEncoderProtocol,
    GenerationResult,
    RoundRecord,
    TargetBackend,
)


_MAX_FRAME_BYTES = 16 * 1024 * 1024


def _read_exact(channel: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise EOFError("socket closed before a complete frame arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(channel: socket.socket, metadata: dict[str, Any], payload: bytes = b"") -> None:
    """Send a 4-byte length frame containing JSON control and raw payload."""
    header = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body = struct.pack("!I", len(header)) + header + payload
    if len(body) > _MAX_FRAME_BYTES:
        raise ValueError("frame exceeds the local reference size limit")
    channel.sendall(struct.pack("!I", len(body)) + body)


def recv_frame(channel: socket.socket) -> tuple[dict[str, Any], bytes]:
    size = struct.unpack("!I", _read_exact(channel, 4))[0]
    if size < 4 or size > _MAX_FRAME_BYTES:
        raise ValueError("invalid frame size")
    body = _read_exact(channel, size)
    header_size = struct.unpack("!I", body[:4])[0]
    if header_size > len(body) - 4:
        raise ValueError("invalid JSON header size")
    metadata = json.loads(body[4 : 4 + header_size].decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("frame metadata must be an object")
    return metadata, body[4 + header_size :]


def _byte_codes(codes: tuple[int, ...]) -> bytes:
    if len(codes) != 8 or any(type(code) is not int or not 0 <= code <= 255 for code in codes):
        raise ValueError("feedback must contain exactly eight byte-valued codes")
    return bytes(codes)


@dataclass(frozen=True)
class CloudEdgeResult(GenerationResult):
    """Acceptance-only result; network timings have no acceleration validity."""

    acceptance_only: bool = True
    latency_claim_valid: bool = False


class CloudVerifier:
    """Authoritative cloud state for one request and one socket connection."""

    def __init__(
        self,
        target: TargetBackend,
        encoder: FeedbackEncoderProtocol | None,
        prompt: list[int],
        max_new_tokens: int,
        *,
        window: int = 8,
        eos_token_id: int | None = None,
    ) -> None:
        if window < 1 or max_new_tokens < 0:
            raise ValueError("window must be positive and max_new_tokens nonnegative")
        self.target = target
        self.encoder = encoder
        self.confirmed = list(prompt)
        self.generated: list[int] = []
        self.max_new_tokens = max_new_tokens
        self.window = window
        self.eos_token_id = eos_token_id
        self.done = max_new_tokens == 0
        self.rounds: list[RoundRecord] = []

    def _verify(self, metadata: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
        if payload or metadata.get("type") != "verify":
            raise ValueError("cloud expects a verify request without a payload")
        prefix = metadata.get("prefix")
        proposals = metadata.get("proposals")
        if not isinstance(prefix, list) or any(type(t) is not int for t in prefix):
            raise ValueError("prefix must be a list of token IDs")
        if prefix != self.confirmed:
            raise ValueError("edge prefix differs from the cloud-confirmed prefix")
        if not isinstance(proposals, list) or any(type(t) is not int for t in proposals):
            raise ValueError("proposals must be a list of token IDs")
        remaining = self.max_new_tokens - len(self.generated)
        expected = min(self.window, remaining)
        if not proposals or len(proposals) > expected:
            raise ValueError("proposal window has an invalid length")
        if len(proposals) != expected and proposals[-1] != self.eos_token_id:
            raise ValueError("a short proposal window must end at draft EOS")
        if self.eos_token_id in proposals[:-1]:
            raise ValueError("draft EOS must terminate its proposal window")

        prefix_before = tuple(self.confirmed)
        accepted = 0
        rejected: int | None = None
        correction: int | None = None
        bonus: int | None = None
        observation = None
        committed: list[int] = []
        for proposal in proposals:
            # The target is queried only on cloud-confirmed history; no
            # rejected or speculative suffix is ever added to this prefix.
            step = self.target.next(self.confirmed.copy())
            if proposal != step.token_id:
                rejected = proposal
                correction = int(step.token_id)
                observation = step.hidden.detach()
                committed.append(correction)
                self.confirmed.append(correction)
                self.generated.append(correction)
                break
            accepted += 1
            committed.append(proposal)
            self.confirmed.append(proposal)
            self.generated.append(proposal)
            if proposal == self.eos_token_id:
                break

        stopped_on_eos = bool(self.generated) and self.generated[-1] == self.eos_token_id
        if rejected is None and not stopped_on_eos and len(self.generated) < self.max_new_tokens:
            step = self.target.next(self.confirmed.copy())
            bonus = int(step.token_id)
            observation = step.hidden.detach()
            committed.append(bonus)
            self.confirmed.append(bonus)
            self.generated.append(bonus)
            stopped_on_eos = bonus == self.eos_token_id

        self.done = stopped_on_eos or len(self.generated) == self.max_new_tokens
        kind = "correction" if rejected is not None else "bonus" if bonus is not None else "none"
        answer = correction if rejected is not None else bonus
        feedback = b""
        if not self.done and self.encoder is not None:
            assert observation is not None and answer is not None
            codes = self.encoder.encode(
                observation,
                None if rejected is None else self.target.embedding(rejected).detach(),
                self.target.embedding(answer).detach(),
                accepted,
            )
            feedback = _byte_codes(tuple(codes))

        self.rounds.append(
            RoundRecord(
                prefix_before=prefix_before,
                codes_in=None,  # Codes stay at the edge and are not sent to the cloud.
                proposals=tuple(proposals),
                accepted_count=accepted,
                committed_tokens=tuple(committed),
                rejected_token=rejected,
                correction_token=correction,
                bonus_token=bonus,
                observation_hidden=observation,
                feedback_codes=tuple(feedback) if feedback else None,
                message_bytes=len(feedback),
            )
        )
        return (
            {
                "type": "result",
                "a": accepted,
                "b": answer,
                "kind": kind,
                "done": self.done,
                "feedback_bytes": len(feedback),
            },
            feedback,
        )

    def serve(self, channel: socket.socket) -> None:
        """Process one request until EOS or the token budget is reached."""
        while not self.done:
            metadata, payload = recv_frame(channel)
            response, feedback = self._verify(metadata, payload)
            send_frame(channel, response, feedback)


class EdgeDrafter:
    """Edge state: only draft proposals and returned (a, b, feedback) are used."""

    def __init__(
        self,
        draft: DraftBackend,
        prompt: list[int],
        max_new_tokens: int,
        *,
        window: int = 8,
        eos_token_id: int | None = None,
    ) -> None:
        if window < 1 or max_new_tokens < 0:
            raise ValueError("window must be positive and max_new_tokens nonnegative")
        self.draft = draft
        self.prompt = list(prompt)
        self.max_new_tokens = max_new_tokens
        self.window = window
        self.eos_token_id = eos_token_id

    def generate(self, channel: socket.socket) -> CloudEdgeResult:
        confirmed = self.prompt.copy()
        generated: list[int] = []
        records: list[RoundRecord] = []
        codes: tuple[int, ...] | None = None
        while len(generated) < self.max_new_tokens:
            remaining = self.max_new_tokens - len(generated)
            prefix_before = tuple(confirmed)
            proposal: list[int] = []
            for position in range(1, min(self.window, remaining) + 1):
                token = int(self.draft.next(confirmed + proposal, codes, position))
                proposal.append(token)
                if token == self.eos_token_id:
                    break
            send_frame(channel, {"type": "verify", "prefix": confirmed, "proposals": proposal})
            reply, feedback = recv_frame(channel)
            if reply.get("type") != "result":
                raise ValueError("cloud returned an unexpected response")
            a = reply.get("a")
            b = reply.get("b")
            kind = reply.get("kind")
            done = reply.get("done")
            declared_bytes = reply.get("feedback_bytes")
            if type(a) is not int or not 0 <= a <= len(proposal):
                raise ValueError("cloud returned an invalid acceptance count")
            if type(done) is not bool or type(declared_bytes) is not int:
                raise ValueError("cloud returned an invalid completion or payload size")
            if declared_bytes != len(feedback) or len(feedback) not in (0, 8):
                raise ValueError("cloud returned an invalid eight-byte feedback payload")
            if kind == "correction" and a < len(proposal) and type(b) is int:
                rejected = proposal[a]
                correction = b
                bonus = None
                committed = proposal[:a] + [b]
            elif kind == "bonus" and a == len(proposal) and type(b) is int:
                rejected = None
                correction = None
                bonus = b
                committed = proposal + [b]
            elif kind == "none" and a == len(proposal) and b is None and done:
                rejected = None
                correction = None
                bonus = None
                committed = proposal.copy()
            else:
                raise ValueError("cloud returned an inconsistent verification result")
            if not committed or len(committed) > remaining:
                raise ValueError("cloud result exceeded the generation budget")
            generated.extend(committed)
            confirmed.extend(committed)
            locally_done = (
                len(generated) == self.max_new_tokens
                or (self.eos_token_id is not None and generated[-1] == self.eos_token_id)
            )
            if done != locally_done or (done and feedback):
                raise ValueError("cloud completion and edge state disagree")
            next_codes = tuple(feedback) if feedback else None
            records.append(
                RoundRecord(
                    prefix_before=prefix_before,
                    codes_in=codes,
                    proposals=tuple(proposal),
                    accepted_count=a,
                    committed_tokens=tuple(committed),
                    rejected_token=rejected,
                    correction_token=correction,
                    bonus_token=bonus,
                    observation_hidden=None,  # Target hidden never crosses the wire.
                    feedback_codes=next_codes,
                    message_bytes=len(feedback),
                )
            )
            codes = next_codes
            if done:
                break
        return CloudEdgeResult(tokens=tuple(generated), rounds=tuple(records))
