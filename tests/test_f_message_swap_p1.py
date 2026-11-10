"""The F intervention preserves local state and uses other-prompt donors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.evaluate_f_message_swap_p1 import (
    LOCK_SHA256, RESULT_SOURCE_FILES, ROOT, _select_donors, _source_snapshot,
    _swap_state, _verify_lock_file,
)
from scripts.high_bandwidth_control import HighBandwidthState, MESSAGE_BYTES


def _row(prompt: str, round_: int, task: str, accepted: int):
    return SimpleNamespace(
        prompt_id=prompt, round_index=round_, task_type=task,
        accepted_count=accepted,
    )


def test_donor_map_is_stable_stratified_and_cross_prompt() -> None:
    rows = [
        _row("b", 1, "x", 2), _row("a", 0, "x", 2),
        _row("b", 0, "x", 2), _row("c", 0, "x", 2),
        _row("d", 0, "y", 1), _row("e", 0, "y", 1),
    ]
    donors = _select_donors(rows)
    assert donors == _select_donors(list(reversed(rows)))
    strata = {(x.prompt_id, x.round_index): (x.task_type, x.accepted_count)
              for x in rows}
    assert set(donors) == set(strata)
    for source, donor in donors.items():
        assert source[0] != donor[0]
        assert strata[source] == strata[donor]


def test_donor_map_rejects_single_prompt_or_duplicate_key() -> None:
    with pytest.raises(ValueError, match="different-prompt"):
        _select_donors([_row("a", 0, "x", 2), _row("a", 1, "x", 2)])
    with pytest.raises(ValueError, match="duplicate"):
        _select_donors([_row("a", 0, "x", 2), _row("a", 0, "x", 2)])


def test_swap_changes_only_cloud_message() -> None:
    local = ((1,) * 8, (2,) * 8)
    donor_local = ((3,) * 8, (4,) * 8)
    own = HighBandwidthState(local=local, message=b"a" * MESSAGE_BYTES)
    donor = HighBandwidthState(local=donor_local,
                               message=b"b" * MESSAGE_BYTES)
    swapped = _swap_state(own, donor)
    assert swapped.local == own.local
    assert swapped.message == donor.message
    assert swapped.local != donor.local
    assert len(swapped.message) == MESSAGE_BYTES


def test_swap_rejects_untyped_prepared_state() -> None:
    state = HighBandwidthState(
        local=((1,) * 8, (2,) * 8), message=b"a" * MESSAGE_BYTES,
    )
    with pytest.raises(TypeError, match="valid prepared"):
        _swap_state(state, object())


def test_pre_run_lock_is_pinned_and_source_snapshot_covers_oracle(
    tmp_path,
) -> None:
    lock = ROOT / "docs" / "P1B_F_MESSAGE_SWAP_LOCK.md"
    _verify_lock_file(lock)
    assert len(LOCK_SHA256) == 64
    assert {
        "scripts/evaluate_fixed_probe.py",
        "scripts/train_fixed_probe.py",
        "feedback_codec/decoding.py",
        "feedback_codec/fixed_probe.py",
    }.issubset(RESULT_SOURCE_FILES)
    assert set(_source_snapshot()) == set(RESULT_SOURCE_FILES)
    changed = tmp_path / "changed-lock.md"
    changed.write_bytes(lock.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError, match="preregistered SHA"):
        _verify_lock_file(changed)
