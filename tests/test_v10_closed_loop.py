"""CPU contract checks for the isolated v10 high-bandwidth closed loop."""

from types import SimpleNamespace

import torch

from feedback_codec.fixed_probe import ProbePrompt
from feedback_codec import v10_closed_loop as module


def test_closed_loop_uses_raw_warm_window_then_previous_legal_observation(monkeypatch):
    class Draft:
        def __init__(self):
            self.calls = []

        def next(self, prefix, codes, position):
            self.calls.append((tuple(prefix), codes, position))
            return 9

    class Head:
        mode = "F-H"

        def __init__(self):
            self.observations = []
            self.calls = []

        def prepare_observation(self, observation, draft):
            self.observations.append(observation)
            assert observation.corrected_prefix == (10, 1)
            assert observation.target_hidden_prefix == observation.valid_prefix == (10,)
            assert observation.correction_token == 1
            assert observation.remaining_budget == 31
            return "prepared"

        def next_token(self, prefix, state, position, draft):
            assert state == "prepared"
            self.calls.append((tuple(prefix), position))
            return 2

    class Verification:
        def __init__(self, proposal, token):
            self.prefix_before = tuple(session.confirmed)
            self.proposals = tuple(proposal)
            self.argmax_ids = (token,) + tuple(1 for _ in proposal)
            self.numerical_recheck_extra_target_forwards = 0
            self.numerical_recheck_snapshot_wall_ms = 0.0
            self.numerical_recheck_forward_wall_ms = 0.0
            self.numerical_recheck_total_wall_ms = 0.0
            self.row_source_modes = ("original_packed",)
            self.row_source_query_lengths = (len(session.confirmed),)

        def hidden_at_valid_prefix(self, accepted):
            return torch.tensor([3.0, 4.0], dtype=torch.bfloat16)

        def logits_at_valid_prefix(self, accepted):
            return torch.tensor([0.0, 1.0, 2.0], dtype=torch.bfloat16)

        def logit_summary_at_valid_prefix(self, accepted):
            return SimpleNamespace(greedy_token_id=1)

    class Session:
        def __init__(self, target, prompt, **kwargs):
            assert kwargs == {"capture_online_cache_snapshot": True,
                              "numerical_recheck_top2_ulps": 2}
            self.confirmed = list(prompt)

        def verify(self, proposal, *, bonus_needed):
            token = 1 if len(self.confirmed) == 1 else 2
            return Verification(proposal, token)

        def check_online_suffix_noninterference(self, verification, accepted):
            return {"source": "original_packed_verify_forward",
                    "diagnostic_feature_gate_target_forwards": 1}

        def commit(self, verification, accepted, answer):
            self.confirmed.extend(verification.proposals[:accepted])
            if answer is not None:
                self.confirmed.append(answer)
            return len(self.confirmed) - 1

    session = None
    original_init = Session.__init__

    def session_init(self, target, prompt, **kwargs):
        nonlocal session
        original_init(self, target, prompt, **kwargs)
        session = self

    Session.__init__ = session_init
    monkeypatch.setattr(module, "HFPackedTargetSession", Session)
    monkeypatch.setattr(module, "autoregressive_generate", lambda *args: [1, 2])
    monkeypatch.setattr(module, "_full_trace", lambda verification, accepted,
                        cache_after, provenance: SimpleNamespace(
                            valid_prefix=verification.prefix_before
                            + verification.proposals[:accepted]))
    monkeypatch.setattr(module, "validate_numeric_provenance", lambda trace: {})
    monkeypatch.setattr(module, "trace_payload", lambda trace: {"valid_prefix":
                        list(trace.valid_prefix)})
    draft, head = Draft(), Head()
    progress = []
    result = module.run_v10_closed_loop(
        target=object(), draft=draft, head=head,
        prompt=ProbePrompt("p", "f", "train", (10,), "chat"),
        eos_token_id=2, progress_hook=progress.append)
    assert result.exact and result.output_ids == result.oracle_ids == (1, 2)
    assert result.eos and not result.truncated
    assert result.first_difference is None
    assert len(head.observations) == 1 and len(head.calls) == 1
    assert len(draft.calls) == 8
    assert len(progress) == 3 and progress[-1]["output_ids"] == [1, 2]


def test_first_difference_includes_length_mismatch():
    assert module._first_difference((1, 2), (1, 3)) == 1
    assert module._first_difference((1,), (1, 2)) == 1
    assert module._first_difference((1, 2), (1, 2)) is None


def test_bfloat16_tensor_digest_is_supported():
    value = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    assert len(module._tensor_sha(value)) == 64
