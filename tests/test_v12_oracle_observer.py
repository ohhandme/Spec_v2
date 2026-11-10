import types

import pytest
import torch

from feedback_codec.v12_oracle_observer import TargetARSummaryObserver


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, prefix):
        self.calls += 1
        return types.SimpleNamespace(logits=torch.tensor([[[1.0, 2.0, 0.0]]]))


class _Target:
    def __init__(self):
        self.model = _Model()
        self.dtype = torch.bfloat16

    def next_with_logit_summary(self, prefix):
        z = self.model(prefix).logits[0, -1]
        values, ids = z.topk(3)
        summary = types.SimpleNamespace(
            greedy_token_id=int(z.argmax().item()),
            top_token_ids=tuple(int(x) for x in ids.tolist()),
            top_logits_fp16=tuple(float(x) for x in values.tolist()),
            logsumexp_fp32=float(z.logsumexp(0).item()),
            vocab_size=3,
        )
        return types.SimpleNamespace(token_id=1, logit_summary=summary)


def test_oracle_observer_delegates_single_real_forward():
    base = _Target()
    observer = TargetARSummaryObserver(base)
    assert observer.next([5, 6]).token_id == 1
    assert base.model.calls == observer.actual_forwards_observed == 1
    assert observer.steps[0]["topk_ids"] == [1, 0, 2]
    assert observer.steps[0]["greedy_token_id"] == 1


def test_oracle_observer_rejects_missing_summary():
    base = _Target()
    base.next_with_logit_summary = lambda prefix: types.SimpleNamespace(
        token_id=1, logit_summary=None)
    observer = TargetARSummaryObserver(base)
    with pytest.raises(RuntimeError, match="one original forward"):
        observer.next([5])
