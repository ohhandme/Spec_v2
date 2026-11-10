"""Indexing checks for the preserved train-failure replay diagnostic."""

from types import SimpleNamespace

from scripts.diagnose_numeric_collector_second_v2 import (
    FAILURE_SHA, first_difference, round_for_difference,
)


def test_preserved_failure_sha_is_actual_frozen_evidence() -> None:
    assert FAILURE_SHA == "99f6a993b37a1b8ba0a03b7ca1b8aabe934c9c78ead45b31fee8742a492539ec"


def test_first_difference_includes_length_mismatch() -> None:
    assert first_difference((1, 2, 3), (1, 4, 3)) == 1
    assert first_difference((1, 2), (1, 2, 3)) == 2
    assert first_difference((1, 2), (1, 2)) is None


def test_round_position_maps_global_first_divergence() -> None:
    rounds = (
        SimpleNamespace(committed_tokens=(11, 12)),
        SimpleNamespace(committed_tokens=(13,)),
        SimpleNamespace(committed_tokens=(14, 15, 16)),
    )
    assert round_for_difference(rounds, 0) == (0, 0, 0)
    assert round_for_difference(rounds, 2) == (1, 0, 2)
    assert round_for_difference(rounds, 4) == (2, 1, 3)
    assert round_for_difference(rounds, 6) == (None, None, 6)
