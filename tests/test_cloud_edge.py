"""Loopback wire checks; none of these timings establish acceleration."""

from __future__ import annotations

import json
import socket
import struct
import threading

import pytest
import torch

from feedback_codec.cloud_edge import CloudVerifier, EdgeDrafter
from feedback_codec.decoding import TargetStep, autoregressive_generate


class Target:
    def __init__(self, prompt: list[int], continuation: list[int]) -> None:
        self.prompt = tuple(prompt)
        self.continuation = tuple(continuation)
        self.prefixes: list[tuple[int, ...]] = []
        self.embeddings: list[int] = []

    def next(self, prefix: list[int]) -> TargetStep:
        seen = tuple(prefix)
        index = len(prefix) - len(self.prompt)
        assert seen[: len(self.prompt)] == self.prompt
        assert seen[len(self.prompt) :] == self.continuation[:index], (
            "cloud target saw an unconfirmed draft branch"
        )
        self.prefixes.append(seen)
        return TargetStep(
            self.continuation[index],
            torch.tensor([float(len(prefix))], requires_grad=True),
        )

    def embedding(self, token_id: int) -> torch.Tensor:
        self.embeddings.append(token_id)
        return torch.tensor([float(token_id)], requires_grad=True)


class Draft:
    def __init__(self, windows: list[list[int]]) -> None:
        self.windows = windows
        self.round_index = -1
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...] | None, int]] = []

    def next(self, prefix: list[int], codes: tuple[int, ...] | None, position: int) -> int:
        if position == 1:
            self.round_index += 1
        self.calls.append((tuple(prefix), codes, position))
        return self.windows[self.round_index][position - 1]


class Encoder:
    codes = (0, 1, 2, 3, 4, 5, 6, 255)

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]] = []

    def encode(
        self,
        target_hidden: torch.Tensor,
        rejected_embedding: torch.Tensor | None,
        correction_embedding: torch.Tensor,
        accepted_count: int,
    ) -> tuple[int, ...]:
        self.calls.append(
            (target_hidden, rejected_embedding, correction_embedding, accepted_count)
        )
        return self.codes


class TapSocket:
    """Record complete sendall buffers while forwarding real socket bytes."""

    def __init__(self, raw: socket.socket) -> None:
        self.raw = raw
        self.sent: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        self.sent.append(bytes(data))
        self.raw.sendall(data)

    def recv(self, size: int) -> bytes:
        return self.raw.recv(size)


def _unpack_frame(raw: bytes) -> tuple[dict, bytes]:
    size = struct.unpack("!I", raw[:4])[0]
    assert size == len(raw) - 4
    body = raw[4:]
    header_size = struct.unpack("!I", body[:4])[0]
    header = json.loads(body[4 : 4 + header_size])
    return header, body[4 + header_size :]


def _run_pair(server: CloudVerifier, edge: EdgeDrafter):
    cloud_raw, edge_raw = socket.socketpair()
    cloud_wire = TapSocket(cloud_raw)
    edge_wire = TapSocket(edge_raw)
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            server.serve(cloud_wire)
        except BaseException as exc:  # propagate thread errors to the test
            errors.append(exc)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        result = edge.generate(edge_wire)
    finally:
        worker.join(timeout=3)
        cloud_raw.close()
        edge_raw.close()
    assert not worker.is_alive(), "cloud did not terminate at EOS or token limit"
    assert not errors, errors
    return result, cloud_wire.sent, edge_wire.sent


def _assert_ar(result, prompt: list[int], continuation: list[int], budget: int, eos=None):
    assert result.tokens == autoregressive_generate(
        Target(prompt, continuation), prompt, budget, eos_token_id=eos
    )
    assert result.acceptance_only is True
    assert result.latency_claim_valid is False


def test_first_rejection_transfers_only_next_round_eight_bytes() -> None:
    prompt = [1]
    continuation = [10, 11, 12, 13, 14]
    target = Target(prompt, continuation)
    draft = Draft([[10, 99], [12, 13]])
    encoder = Encoder()
    server = CloudVerifier(target, encoder, prompt, 5, window=2)
    result, cloud_frames, edge_frames = _run_pair(
        server, EdgeDrafter(draft, prompt, 5, window=2)
    )

    _assert_ar(result, prompt, continuation, 5)
    assert result.tokens == tuple(continuation)
    assert result.accepted_draft_tokens == 3
    assert result.verification_rounds == 2
    assert result.total_message_bytes == 8
    assert len(encoder.calls) == 1
    hidden, rejected, correction, accepted = encoder.calls[0]
    assert not hidden.requires_grad
    assert hidden.item() == 2.0  # confirmed prefix [1, 10]
    assert rejected is not None and rejected.item() == 99.0
    assert correction.item() == 11.0
    assert accepted == 1
    assert len(target.prefixes) == len(continuation)  # no encoding-only target forward
    assert 99 not in [token for prefix in target.prefixes for token in prefix]
    assert [call[1] for call in draft.calls] == [None, None, encoder.codes, encoder.codes]
    assert result.rounds[0].observation_hidden is None  # never transmitted
    assert result.rounds[0].feedback_codes == encoder.codes
    assert result.rounds[1].feedback_codes is None

    first_request, request_payload = _unpack_frame(edge_frames[0])
    assert first_request == {"type": "verify", "prefix": prompt, "proposals": [10, 99]}
    assert request_payload == b""
    first_reply, feedback = _unpack_frame(cloud_frames[0])
    assert first_reply["a"] == 1 and first_reply["b"] == 11
    assert first_reply["kind"] == "correction" and first_reply["done"] is False
    assert first_reply["feedback_bytes"] == 8
    assert feedback == bytes(encoder.codes) and len(feedback) == 8
    final_reply, final_payload = _unpack_frame(cloud_frames[1])
    assert final_reply["feedback_bytes"] == 0 and final_payload == b""


def test_full_acceptance_bonus_uses_null_rejection_then_final_truncation() -> None:
    prompt = [2]
    continuation = [5, 6, 7, 8, 9]
    target = Target(prompt, continuation)
    draft = Draft([[5, 6], [8, 9]])
    encoder = Encoder()
    result, cloud_frames, _ = _run_pair(
        CloudVerifier(target, encoder, prompt, 5, window=2),
        EdgeDrafter(draft, prompt, 5, window=2),
    )
    _assert_ar(result, prompt, continuation, 5)
    assert result.rounds[0].accepted_count == 2
    assert result.rounds[0].bonus_token == 7
    assert result.rounds[1].accepted_count == 2
    assert result.rounds[1].bonus_token is None  # budget exactly exhausted
    assert len(target.prefixes) == 5
    assert len(encoder.calls) == 1
    assert encoder.calls[0][1] is None
    assert encoder.calls[0][2].item() == 7.0
    assert encoder.calls[0][3] == 2
    assert len(_unpack_frame(cloud_frames[0])[1]) == 8
    assert len(_unpack_frame(cloud_frames[1])[1]) == 0


@pytest.mark.parametrize(
    ("continuation", "windows", "expected", "expected_messages"),
    [
        ([4, 0], [[4, 0]], (4, 0), 0),  # accepted EOS: no bonus
        ([4, 0], [[4, 5, 6]], (4, 0), 0),  # target EOS correction
        ([4, 5, 0], [[4, 0], [0]], (4, 5, 0), 1),  # rejected draft EOS
    ],
)
def test_eos_cases_are_greedy_exact(continuation, windows, expected, expected_messages) -> None:
    prompt = [3]
    target = Target(prompt, continuation)
    draft = Draft(windows)
    encoder = Encoder()
    result, cloud_frames, _ = _run_pair(
        CloudVerifier(target, encoder, prompt, 10, window=3, eos_token_id=0),
        EdgeDrafter(draft, prompt, 10, window=3, eos_token_id=0),
    )
    _assert_ar(result, prompt, continuation, 10, eos=0)
    assert result.tokens == expected
    assert result.total_message_bytes == 8 * expected_messages
    assert [len(_unpack_frame(frame)[1]) for frame in cloud_frames].count(8) == expected_messages
    assert result.rounds[-1].feedback_codes is None


def test_no_feedback_baseline_uses_no_codes_or_embeddings() -> None:
    prompt = [1]
    continuation = [10, 11, 12, 13, 14]
    target = Target(prompt, continuation)
    draft = Draft([[10, 99], [12, 13]])
    result, cloud_frames, _ = _run_pair(
        CloudVerifier(target, None, prompt, 5, window=2),
        EdgeDrafter(draft, prompt, 5, window=2),
    )
    _assert_ar(result, prompt, continuation, 5)
    assert all(call[1] is None for call in draft.calls)
    assert all(len(_unpack_frame(frame)[1]) == 0 for frame in cloud_frames)
    assert result.total_message_bytes == 0
    assert target.embeddings == []


def test_zero_budget_sends_no_frames_or_target_queries() -> None:
    prompt = [1]
    target = Target(prompt, [2])
    draft = Draft([])
    result, cloud_frames, edge_frames = _run_pair(
        CloudVerifier(target, Encoder(), prompt, 0), EdgeDrafter(draft, prompt, 0)
    )
    _assert_ar(result, prompt, [2], 0)
    assert result.tokens == () and result.rounds == ()
    assert cloud_frames == [] and edge_frames == []
    assert target.prefixes == [] and draft.calls == []


def test_cloud_rejects_unconfirmed_prefix_and_wrong_message_size() -> None:
    server = CloudVerifier(Target([1], [2, 3]), Encoder(), [1], 2, window=1)
    with pytest.raises(ValueError, match="cloud-confirmed"):
        server._verify({"type": "verify", "prefix": [1, 99], "proposals": [2]}, b"")

    class BadEncoder(Encoder):
        def encode(self, *args):
            return (0, 1)

    server = CloudVerifier(Target([1], [2, 3, 4]), BadEncoder(), [1], 3, window=1)
    with pytest.raises(ValueError, match="eight byte"):
        server._verify({"type": "verify", "prefix": [1], "proposals": [2]}, b"")
