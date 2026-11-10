"""Deterministic causal and greedy-exactness checks for round-delayed feedback."""

from __future__ import annotations

import torch

from feedback_codec.decoding import SpeculativeDecoder, TargetStep, autoregressive_generate


class ScriptedTarget:
    def __init__(self, prompt: list[int], continuation: list[int]) -> None:
        self.prompt = tuple(prompt)
        self.continuation = tuple(continuation)
        self.prefixes: list[tuple[int, ...]] = []
        self.embeddings: list[int] = []

    def next(self, prefix: list[int]) -> TargetStep:
        key = tuple(prefix)
        assert key[: len(self.prompt)] == self.prompt
        index = len(key) - len(self.prompt)
        assert key[len(self.prompt) :] == self.continuation[:index], (
            "target was queried on a rejected or otherwise unconfirmed branch"
        )
        self.prefixes.append(key)
        return TargetStep(
            self.continuation[index],
            torch.tensor([float(len(key))], requires_grad=True),
        )

    def embedding(self, token_id: int) -> torch.Tensor:
        self.embeddings.append(token_id)
        return torch.tensor([float(token_id)], requires_grad=True)


class ScriptedDraft:
    def __init__(self, rounds: list[list[int]]) -> None:
        self.rounds = rounds
        self.round_index = -1
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...] | None, int]] = []

    def next(
        self, prefix: list[int], codes: tuple[int, ...] | None, position: int
    ) -> int:
        if position == 1:
            self.round_index += 1
        self.calls.append((tuple(prefix), codes, position))
        return self.rounds[self.round_index][position - 1]


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]] = []
        self.message = tuple(range(8))

    def encode(
        self,
        target_hidden: torch.Tensor,
        rejected_embedding: torch.Tensor | None,
        correction_embedding: torch.Tensor,
        accepted_count: int,
    ) -> tuple[int, ...]:
        self.calls.append(
            (
                target_hidden,
                rejected_embedding,
                correction_embedding,
                accepted_count,
            )
        )
        return self.message


def _assert_ar_exact(
    result: tuple[int, ...], prompt: list[int], continuation: list[int],
    max_new_tokens: int, eos_token_id: int | None = None,
) -> None:
    fresh_target = ScriptedTarget(prompt, continuation)
    assert result == autoregressive_generate(
        fresh_target, prompt, max_new_tokens, eos_token_id
    )


def test_first_rejection_uses_valid_prefix_hidden_and_delays_feedback() -> None:
    prompt = [1]
    target = ScriptedTarget(prompt, [10, 11, 12, 13, 14])
    draft = ScriptedDraft([[10, 99], [12, 13]])
    encoder = RecordingEncoder()
    result = SpeculativeDecoder(target, draft, encoder, window=2).generate(prompt, 5)

    assert result.tokens == (10, 11, 12, 13, 14)
    _assert_ar_exact(result.tokens, prompt, [10, 11, 12, 13, 14], 5)
    assert [call[2] for call in draft.calls] == [1, 2, 1, 2]
    assert [call[1] for call in draft.calls] == [None, None, encoder.message, encoder.message]
    assert draft.calls[1][0] == (1, 10)  # draft's still unverified branch
    assert 99 not in [token for prefix in target.prefixes for token in prefix]
    assert target.prefixes == [(1,), (1, 10), (1, 10, 11), (1, 10, 11, 12), (1, 10, 11, 12, 13)]
    assert len(encoder.calls) == 1  # no extra target call for the message
    hidden, rejected, correction, count = encoder.calls[0]
    assert hidden.item() == 2.0  # prefix [1, 10], before rejected 99
    assert rejected is not None and rejected.item() == 99.0
    assert correction.item() == 11.0
    assert count == 1
    first, second = result.rounds
    assert first.prefix_before == (1,)
    assert first.proposals == (10, 99)
    assert first.rejected_token == 99
    assert first.correction_token == 11
    assert first.accepted_count == 1
    assert first.committed_tokens == (10, 11)
    assert first.feedback_codes == encoder.message
    assert first.message_bytes == 8
    assert first.observation_hidden is not None
    assert first.observation_hidden.item() == 2.0
    assert not first.observation_hidden.requires_grad
    assert second.prefix_before == (1, 10, 11)
    assert second.codes_in == encoder.message
    assert second.feedback_codes is None
    assert result.accepted_draft_tokens == 3
    assert result.verification_rounds == 2
    assert result.total_message_bytes == 8


def test_rejection_at_position_zero_preserves_causal_boundary() -> None:
    prompt = [1]
    continuation = [10, 11, 12]
    target = ScriptedTarget(prompt, continuation)
    draft = ScriptedDraft([[99, 88], [11, 12]])
    encoder = RecordingEncoder()
    result = SpeculativeDecoder(target, draft, encoder, window=2).generate(prompt, 3)

    _assert_ar_exact(result.tokens, prompt, continuation, 3)
    assert result.rounds[0].accepted_count == 0
    assert result.rounds[0].committed_tokens == (10,)
    assert result.rounds[0].rejected_token == 99
    assert result.rounds[0].observation_hidden is not None
    assert result.rounds[0].observation_hidden.item() == 1.0
    assert target.prefixes == [(1,), (1, 10), (1, 10, 11)]
    assert [call[1] for call in draft.calls] == [None, None, encoder.message, encoder.message]
    assert encoder.calls[0][3] == 0
    assert encoder.calls[0][2].item() == 10.0


def test_full_acceptance_uses_bonus_and_null_rejection() -> None:
    prompt = [2]
    continuation = [5, 6, 7, 8, 9]
    target = ScriptedTarget(prompt, continuation)
    draft = ScriptedDraft([[5, 6], [8, 9]])
    encoder = RecordingEncoder()
    result = SpeculativeDecoder(target, draft, encoder, window=2).generate(prompt, 5)

    _assert_ar_exact(result.tokens, prompt, continuation, 5)
    assert result.tokens == tuple(continuation)
    assert len(encoder.calls) == 1
    hidden, rejected, correction, count = encoder.calls[0]
    assert hidden.item() == 3.0  # prefix [2, 5, 6] before bonus
    assert not hidden.requires_grad
    assert rejected is None
    assert correction.item() == 7.0
    assert count == 2
    assert result.rounds[0].bonus_token == 7
    assert result.rounds[0].rejected_token is None
    assert result.rounds[0].committed_tokens == (5, 6, 7)
    assert result.rounds[1].bonus_token is None  # max_new reached
    assert result.rounds[1].feedback_codes is None
    assert len(target.prefixes) == 5


def test_accepted_draft_eos_stops_without_bonus_or_feedback() -> None:
    target = ScriptedTarget([3], [4, 0])
    draft = ScriptedDraft([[4, 0]])
    encoder = RecordingEncoder()
    result = SpeculativeDecoder(target, draft, encoder, window=3, eos_token_id=0).generate([3], 10)
    assert result.tokens == (4, 0)
    _assert_ar_exact(result.tokens, [3], [4, 0], 10, 0)
    assert len(target.prefixes) == 2
    assert len(draft.calls) == 2
    assert encoder.calls == []
    assert result.rounds[0].accepted_count == 2
    assert result.rounds[0].bonus_token is None
    assert result.rounds[0].feedback_codes is None


def test_rejected_draft_eos_can_continue_with_next_round() -> None:
    target = ScriptedTarget([3], [4, 5, 0])
    draft = ScriptedDraft([[4, 0], [0]])
    encoder = RecordingEncoder()
    result = SpeculativeDecoder(target, draft, encoder, window=3, eos_token_id=0).generate([3], 10)
    assert result.tokens == (4, 5, 0)
    _assert_ar_exact(result.tokens, [3], [4, 5, 0], 10, 0)
    assert result.rounds[0].rejected_token == 0
    assert result.rounds[0].correction_token == 5
    assert result.rounds[0].feedback_codes == encoder.message
    assert draft.calls[-1][1] == encoder.message


def test_target_eos_as_correction_stops_without_feedback() -> None:
    target = ScriptedTarget([3], [4, 0])
    draft = ScriptedDraft([[4, 5, 6]])
    encoder = RecordingEncoder()
    result = SpeculativeDecoder(target, draft, encoder, window=3, eos_token_id=0).generate([3], 10)
    assert result.tokens == (4, 0)
    _assert_ar_exact(result.tokens, [3], [4, 0], 10, 0)
    assert result.rounds[0].rejected_token == 5
    assert result.rounds[0].correction_token == 0
    assert result.rounds[0].feedback_codes is None
    assert encoder.calls == []


def test_zero_and_negative_budgets() -> None:
    target = ScriptedTarget([1], [2])
    draft = ScriptedDraft([])
    encoder = RecordingEncoder()
    decoder = SpeculativeDecoder(target, draft, encoder)
    result = decoder.generate([1], 0)
    assert result.tokens == () and result.rounds == ()
    assert target.prefixes == [] and draft.calls == [] and encoder.calls == []
    assert autoregressive_generate(target, [1], 0) == ()

    try:
        decoder.generate([1], -1)
    except ValueError as exc:
        assert "max_new_tokens" in str(exc)
    else:
        raise AssertionError("negative budget was accepted")


def test_partial_final_window_has_no_bonus_or_feedback() -> None:
    prompt = [1]
    target = ScriptedTarget(prompt, [2, 3, 4])
    draft = ScriptedDraft([[2, 3, 4]])
    encoder = RecordingEncoder()
    result = SpeculativeDecoder(target, draft, encoder, window=8).generate(prompt, 3)
    assert result.tokens == (2, 3, 4)
    assert result.rounds[0].accepted_count == 3
    assert result.rounds[0].bonus_token is None
    assert result.rounds[0].feedback_codes is None
    assert len(target.prefixes) == 3
    assert len(draft.calls) == 3
    assert encoder.calls == []


def test_none_encoder_keeps_all_rounds_unconditioned() -> None:
    prompt = [1]
    continuation = [10, 11, 12, 13, 14]
    target = ScriptedTarget(prompt, continuation)
    draft = ScriptedDraft([[10, 99], [12, 13]])
    result = SpeculativeDecoder(target, draft, None, window=2).generate(prompt, 5)
    assert result.tokens == tuple(continuation)
    assert all(call[1] is None for call in draft.calls)
    assert all(round_.feedback_codes is None and round_.message_bytes == 0 for round_ in result.rounds)
    assert result.total_message_bytes == 0
    assert target.embeddings == []


def test_malformed_feedback_message_is_rejected() -> None:
    class BadEncoder(RecordingEncoder):
        def encode(self, *args: object) -> tuple[int, ...]:
            return (1, 2)

    target = ScriptedTarget([1], [2, 3, 4])
    draft = ScriptedDraft([[2]])
    decoder = SpeculativeDecoder(target, draft, BadEncoder(), window=1)
    try:
        decoder.generate([1], 3)
    except ValueError as exc:
        assert "eight byte" in str(exc)
    else:
        raise AssertionError("malformed feedback message was accepted")
