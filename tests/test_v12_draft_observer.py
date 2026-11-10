import types

import pytest
import torch

from feedback_codec.v12_draft_observer import (
    RawDraftForwardObserver, draft_step_stats,
)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, prefix):
        self.calls += 1
        return types.SimpleNamespace(logits=torch.tensor(
            [[[1.0, 2.0, -1.0, 0.0]]], dtype=torch.float32))


class _Backend:
    def __init__(self):
        self.model = _Model()

    def next(self, prefix, codes, position):
        return int(self.model(prefix).logits[0, -1].argmax().item())


def test_raw_observer_records_actual_forward_without_second_call():
    backend = _Backend()
    observed = RawDraftForwardObserver(backend)
    assert observed.next([7, 8], None, 1) == 1
    assert backend.model.calls == 1
    assert observed.actual_forwards_observed == 1
    row = observed.steps[0]
    assert row["token_id"] == 1 and row["position"] == 1
    assert row["topk_ids"][0] == 1
    assert row["top2_margin"] == 1.0
    assert row["vocab_size"] == 4
    assert row["distribution_stage"] == "final_proposal_logits"
    assert len(row["actual_forward_logits_sha256"]) == 64


def test_raw_observer_rejects_repair_without_forward():
    backend = _Backend()
    observed = RawDraftForwardObserver(backend)
    with pytest.raises(ValueError, match="repaired"):
        observed.next([7], (0,) * 8, 1)
    assert backend.model.calls == 0


def test_full_vocabulary_stats_and_nonfinite_gate():
    row = draft_step_stats(torch.tensor([0.0, 2.0, -2.0]), 1,
                           prefix=[3], position=1)
    assert len(row["topk_ids"]) == 3
    assert 0 < row["selected_prob"] < 1
    assert row["logit_std"] == pytest.approx(1.632993, rel=1e-5)
    with pytest.raises(ValueError):
        draft_step_stats(torch.tensor([0.0, float("inf")]), 0,
                         prefix=[3], position=1)
