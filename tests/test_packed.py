"""CPU correctness tests for the candidate packed Target and KV discipline."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import threading
import json

import pytest

torch = pytest.importorskip("torch")

from feedback_codec.decoding import TargetStep, autoregressive_generate
from feedback_codec.packed import (
    PACKED_FEATURE_PROTOCOL_VERSION, HFPackedTargetSession, PackedRun,
    PackedSpeculativeDecoder, _clone_cache_for_probe,
    check_packed_feature_noninterference,
)


class FakeCache:
    def __init__(self, tokens: tuple[int, ...]) -> None:
        self.tokens = tokens

    def get_seq_length(self) -> int:
        return len(self.tokens)

    def crop(self, length: int) -> None:
        self.tokens = self.tokens[:length]


def test_online_kv_snapshot_requires_independent_tensor_storage():
    class TensorCache:
        def __init__(self, keys):
            self.layers = [SimpleNamespace(keys=keys)]

        def get_seq_length(self):
            return int(self.layers[0].keys.shape[-2])

    class SharingCache(TensorCache):
        def __deepcopy__(self, _memo):
            return SharingCache(self.layers[0].keys)

    original = TensorCache(torch.arange(3.0).reshape(1, 1, 3, 1))
    snapshot = _clone_cache_for_probe(original)
    assert snapshot is not original
    assert torch.equal(snapshot.layers[0].keys, original.layers[0].keys)
    assert (snapshot.layers[0].keys.untyped_storage().data_ptr()
            != original.layers[0].keys.untyped_storage().data_ptr())

    sharing = SharingCache(torch.arange(3.0).reshape(1, 1, 3, 1))
    with pytest.raises(RuntimeError, match="shares tensor storage"):
        _clone_cache_for_probe(sharing)


class FakeModel(torch.nn.Module):
    def __init__(self, target: "FakeTarget") -> None:
        super().__init__()
        self.target = target
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.Identity()

    def forward(self, *, input_ids, past_key_values=None, use_cache,
                output_hidden_states=False, return_dict):
        assert return_dict
        past = () if past_key_values is None else past_key_values.tokens
        chunk = tuple(int(x) for x in input_ids[0].tolist())
        full = past + chunk
        logits = torch.full((1, len(chunk), self.target.vocab_size), -100.0)
        hidden = torch.zeros((1, len(chunk), self.target.hidden_size))
        for j in range(len(chunk)):
            prefix = full[:len(past) + j + 1]
            logits[0, j, self.target.token_at(prefix)] = 100.0
            hidden[0, j, 0] = float(len(prefix))
        final = self.model.norm(hidden)
        return SimpleNamespace(logits=logits,
                               hidden_states=(final,) if output_hidden_states else None,
                               past_key_values=FakeCache(full))


class FakeTarget:
    """Deterministic Target defined on both the true path and rejected branches."""

    def __init__(self, prompt: list[int], continuation: list[int], vocab_size=101) -> None:
        self.prompt = tuple(prompt)
        self.continuation = tuple(continuation)
        self.vocab_size = vocab_size
        self.hidden_size = 1
        self._lock = threading.RLock()
        self.model = FakeModel(self)

    def _ids(self, tokens: list[int]):
        if not tokens or any(type(t) is not int or not 0 <= t < self.vocab_size
                             for t in tokens):
            raise ValueError("invalid prefix")
        return torch.tensor([tokens], dtype=torch.long)

    def token_at(self, prefix: tuple[int, ...]) -> int:
        assert prefix[:len(self.prompt)] == self.prompt
        index = len(prefix) - len(self.prompt)
        if prefix[len(self.prompt):] == self.continuation[:index] and index < len(self.continuation):
            return self.continuation[index]
        return (sum(prefix) * 7 + 3) % self.vocab_size

    def next(self, prefix: list[int]) -> TargetStep:
        return TargetStep(self.token_at(tuple(prefix)), torch.tensor([float(len(prefix))]))

    def embedding(self, token_id: int):
        return torch.tensor([float(token_id)])


class ScriptedDraft:
    def __init__(self, rounds: list[list[int]]) -> None:
        self.rounds = rounds
        self.index = -1
        self.calls = []

    def next(self, prefix, codes, position):
        if position == 1:
            self.index += 1
        self.calls.append((tuple(prefix), codes, position))
        return self.rounds[self.index][position - 1]


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, hidden, rejected, correction, accepted_count):
        self.calls.append((hidden.item(), None if rejected is None else rejected.item(),
                           correction.item(), accepted_count))
        return tuple(range(8))


def _run(prompt, continuation, draft_rounds, *, window, budget, eos=None):
    target = FakeTarget(prompt, continuation)
    draft = ScriptedDraft(draft_rounds)
    encoder = Encoder()
    packed = PackedSpeculativeDecoder(target, draft, encoder, window, eos).generate(prompt, budget)
    oracle = autoregressive_generate(FakeTarget(prompt, continuation), prompt, budget, eos)
    assert packed.generation.tokens == oracle
    assert [r.prefix_before for r in packed.generation.rounds] == [t.prefix_before for t in packed.traces]
    for trace in packed.traces:
        assert trace.cache_length_before == len(trace.prefix_before) - 1
        assert trace.cache_length_raw == len(trace.prefix_before) + len(trace.proposals)
        for i, argmax in enumerate(trace.argmax_ids):
            ref = target.next(list(trace.prefix_before + trace.proposals[:i]))
            assert argmax == ref.token_id
    return packed, draft, encoder


def test_zero_acceptance_discards_both_wrong_branches_and_delays_message():
    packed, draft, encoder = _run([1], [10, 11, 12, 13],
                                  [[99, 88], [11, 12]], window=2, budget=4)
    first, second = packed.generation.rounds
    assert first.accepted_count == 0 and first.rejected_token == 99
    assert first.correction_token == 10 and first.committed_tokens == (10,)
    assert first.observation_hidden.item() == 1.0
    assert first.feedback_codes == tuple(range(8))
    assert packed.traces[0].cache_length_after == 1
    assert second.prefix_before == (1, 10)
    assert second.accepted_count == 2 and second.bonus_token == 13
    assert packed.traces[1].cache_length_after == 4
    assert [call[1] for call in draft.calls] == [None, None, tuple(range(8)), tuple(range(8))]
    assert encoder.calls == [(1.0, 99.0, 10.0, 0)]


def test_partial_acceptance_crops_rejected_suffix_and_uses_valid_hidden():
    packed, _, encoder = _run([1], [10, 11, 12],
                              [[10, 99], [12]], window=2, budget=3)
    first = packed.generation.rounds[0]
    assert first.accepted_count == 1 and first.committed_tokens == (10, 11)
    assert first.observation_hidden.item() == 2.0
    assert packed.traces[0].cache_length_after == 2
    assert packed.generation.rounds[1].prefix_before == (1, 10, 11)
    assert packed.generation.rounds[1].bonus_token is None
    assert encoder.calls == [(2.0, 99.0, 11.0, 1)]


def test_full_acceptance_bonus_and_budget_boundary():
    packed, _, encoder = _run([1], [10, 11, 12],
                              [[10, 11]], window=2, budget=3)
    round_ = packed.generation.rounds[0]
    assert round_.accepted_count == 2 and round_.bonus_token == 12
    assert round_.committed_tokens == (10, 11, 12)
    assert packed.traces[0].cache_length_after == 3
    assert round_.feedback_codes is None and encoder.calls == []
    final, _, _ = _run([1], [10, 11, 12], [[10, 11, 12]], window=8, budget=3)
    assert final.generation.rounds[0].bonus_token is None
    assert final.traces[0].cache_length_after == 3
    assert len(final.generation.tokens) == 3 and final.generation.tokens[-1] != 0


def test_full_acceptance_bonus_can_be_eos():
    packed, _, encoder = _run([1], [10, 11, 0], [[10, 11]],
                              window=2, budget=8, eos=0)
    round_ = packed.generation.rounds[0]
    assert packed.generation.tokens == (10, 11, 0)
    assert round_.accepted_count == 2 and round_.bonus_token == 0
    assert round_.feedback_codes is None and encoder.calls == []
    assert packed.traces[0].cache_length_after == 3


def test_accepted_eos_and_target_eos_correction_stop_without_message():
    accepted, _, encoder = _run([3], [4, 0], [[4, 0]], window=3, budget=10, eos=0)
    assert accepted.generation.tokens == (4, 0)
    assert accepted.generation.rounds[0].bonus_token is None
    assert accepted.generation.rounds[0].feedback_codes is None
    assert encoder.calls == []
    corrected, _, encoder2 = _run([3], [4, 0], [[4, 5, 6]], window=3, budget=10, eos=0)
    assert corrected.generation.rounds[0].rejected_token == 5
    assert corrected.generation.rounds[0].correction_token == 0
    assert corrected.generation.rounds[0].feedback_codes is None
    assert encoder2.calls == []


def test_rejected_draft_eos_can_continue():
    packed, draft, encoder = _run([3], [4, 5, 0], [[4, 0], [0]],
                                  window=3, budget=10, eos=0)
    assert packed.generation.tokens == (4, 5, 0)
    assert packed.generation.rounds[0].rejected_token == 0
    assert packed.generation.rounds[0].correction_token == 5
    assert packed.generation.rounds[1].accepted_count == 1
    assert draft.calls[-1][1] == tuple(range(8))
    assert encoder.calls == [(2.0, 0.0, 5.0, 1)]


def test_session_commits_only_verified_prefix_and_cannot_double_verify():
    target = FakeTarget([1], [10, 11, 12, 13])
    session = HFPackedTargetSession(target, [1])
    v = session.verify([10, 99])
    assert tuple(target.next([1] + [10, 99][:i]).token_id for i in range(3)) == v.argmax_ids
    with pytest.raises(RuntimeError, match="not been committed"):
        session.verify([12])
    with pytest.raises(ValueError, match="correction"):
        session.commit(v, 1, 42)
    assert session.commit(v, 1, 11) == 2
    assert session._cache.tokens == (1, 10)
    assert session.confirmed == [1, 10, 11]
    v2 = session.verify([12])
    assert v2.cache_length_before == 2
    assert v2.argmax_ids[0] == 12 and v2.argmax_ids[1] == 13
    assert session.commit(v2, 1, 13) == 4
    assert session._cache.tokens == (1, 10, 11, 12)


def test_valid_feature_hidden_and_top32_are_from_one_packed_forward():
    target = FakeTarget([1], [10, 11, 12], vocab_size=101)
    calls = 0
    original = target.model.forward

    def counted_forward(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    target.model.forward = counted_forward
    verification = HFPackedTargetSession(target, [1]).verify([10, 99])
    assert calls == 1
    assert verification.hidden_at_valid_prefix(1).item() == 2.0
    row = verification.logits_at_valid_prefix(1).float()
    summary = verification.logit_summary_at_valid_prefix(1)
    assert calls == 1  # Neither feature makes another Target call.
    assert summary.greedy_token_id == verification.argmax_ids[1] == 11
    assert len(summary.top_token_ids) == len(summary.top_logits_fp16) == 32
    assert summary.top_token_ids[0] == 11
    assert summary.logsumexp_fp32 == float(row.logsumexp(-1).item())
    assert summary.top2_margin_fp32 == verification.top2_margins[1]
    assert summary.vocab_size == target.vocab_size
    with pytest.raises(ValueError, match="invalid accepted count"):
        verification.logit_summary_at_valid_prefix(3)


def test_packed_decoder_can_record_legal_same_forward_logit_summary():
    target = FakeTarget([1], [10, 11, 12])
    packed = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 99], [12]]), None, 2,
        capture_target_logits=True,
    ).generate([1], 3)
    first, last = packed.generation.rounds
    assert first.accepted_count == 1
    assert first.observation_logit_summary is not None
    assert first.observation_logit_summary.greedy_token_id == first.correction_token == 11
    assert first.observation_logit_summary.vocab_size == target.vocab_size
    assert last.observation_logit_summary is None  # Budget-terminated round.
    bonus = PackedSpeculativeDecoder(
        FakeTarget([1], [10, 11, 12, 13]), ScriptedDraft([[10, 11], [13]]),
        None, 2, capture_target_logits=True,
    ).generate([1], 4).generation.rounds[0]
    assert bonus.accepted_count == 2 and bonus.bonus_token == 12
    assert bonus.observation_logit_summary.greedy_token_id == 12


def test_rejected_suffix_noninterference_for_zero_partial_and_full_boundaries():
    target = FakeTarget([1], [10, 11, 12, 13])
    cases = (
        ([99, 88], [98, 87], 0),
        ([10, 99], [10, 98], 1),
        ([10, 11, 77], [10, 11, 78], 2),  # Formerly full accept; extra probe token.
    )
    for proposals, altered, accepted in cases:
        evidence = check_packed_feature_noninterference(
            target, [1], proposals, altered, accepted)
        assert evidence["feature_protocol_version"] == "fresh_prefill_suffix_diagnostic_v1"
        assert evidence["source"] == "fresh_prefill_not_online_KV"
        assert evidence["hidden_bitwise_equal"] and evidence["logits_bitwise_equal"]
        assert evidence["valid_prefix"] == [1] + proposals[:accepted]


def test_rejected_suffix_leak_is_a_hard_feature_gate_failure():
    target = FakeTarget([1], [10, 11])
    original = target.model.forward

    def leaky_forward(**kwargs):
        output = original(**kwargs)
        # Deliberate violation: a later candidate changes the valid row.
        output.hidden_states[-1][0, 0, 0] += float(kwargs["input_ids"][0, -1])
        return output

    target.model.forward = leaky_forward
    with pytest.raises(RuntimeError, match="unconfirmed suffix"):
        check_packed_feature_noninterference(target, [1], [99, 88], [98, 87], 0)


def test_empty_budget_is_inert_and_negative_budget_rejected():
    target = FakeTarget([1], [2])
    draft = ScriptedDraft([])
    decoder = PackedSpeculativeDecoder(target, draft, None)
    assert decoder.generate([1], 0).generation.tokens == ()
    assert draft.calls == []
    with pytest.raises(ValueError, match="max_new_tokens"):
        decoder.generate([1], -1)


def test_gpu_gate_records_hypothetical_position_but_rejects_used_position_and_bad_kv():
    pytest.importorskip("transformers")
    from scripts.run_packed_p0 import PACKED_TOKEN_GATE_VERSION, GateFailure, check_rounds

    target = FakeTarget([1], [10, 11, 12, 13])
    result = PackedSpeculativeDecoder(
        target, ScriptedDraft([[99, 88], [11, 12]]), None, 2
    ).generate([1], 4)
    oracle = autoregressive_generate(target, [1], 4)
    assert len(check_rounds(target, result, oracle, [1], 4, 100, "raw_draft", "train-1")) == 2

    first = result.traces[0]
    bad_ids = first.argmax_ids[:1] + ((first.argmax_ids[1] + 1) % target.vocab_size,) + first.argmax_ids[2:]
    bad_position = PackedRun(result.generation,
                             (replace(first, argmax_ids=bad_ids), *result.traces[1:]))
    checks = check_rounds(target, bad_position, oracle, [1], 4, 100, "raw_draft", "train-1")
    assert checks[0]["token_gate_version"] == PACKED_TOKEN_GATE_VERSION
    assert checks[0]["last_used_position"] == 0
    assert checks[0]["hypothetical_positions"][0]["position"] == 1
    assert checks[0]["hypothetical_positions"][0]["argmax_equal"] is False
    bad_later_ids = first.argmax_ids[:2] + (
        (first.argmax_ids[2] + 1) % target.vocab_size,)
    bad_later = PackedRun(result.generation,
                          (replace(first, argmax_ids=bad_later_ids), *result.traces[1:]))
    later_check = check_rounds(target, bad_later, oracle, [1], 4, 100,
                               "raw_draft", "train-1")[0]
    assert later_check["hypothetical_positions"][1]["position"] == 2
    assert later_check["hypothetical_positions"][1]["argmax_equal"] is False

    bad_used_ids = ((first.argmax_ids[0] + 1) % target.vocab_size,) + first.argmax_ids[1:]
    bad_used = PackedRun(result.generation,
                         (replace(first, argmax_ids=bad_used_ids), *result.traces[1:]))
    with pytest.raises(GateFailure, match="verification position") as failed:
        check_rounds(target, bad_used, oracle, [1], 4, 100, "raw_draft", "train-1")
    assert failed.value.evidence["position"] == 0
    assert failed.value.evidence["reference_prefix"] == [1]
    with pytest.raises(GateFailure, match="verification position"):
        check_rounds(target, bad_used, oracle, [1], 4, 100,
                     "raw_draft", "train-1", feature_provenance=True)

    bad_cache = PackedRun(result.generation,
                          (replace(first, cache_length_after=first.cache_length_after + 1),
                           *result.traces[1:]))
    with pytest.raises(GateFailure, match="KV commit"):
        check_rounds(target, bad_cache, oracle, [1], 4, 100, "raw_draft", "train-1")


def test_all_accepted_bonus_is_used_but_terminal_bonus_is_hypothetical():
    pytest.importorskip("transformers")
    from scripts.run_packed_p0 import GateFailure, check_rounds

    target = FakeTarget([1], [10, 11, 12])
    used_bonus = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 11]]), None, 2).generate([1], 3)
    trace = used_bonus.traces[0]
    assert check_rounds(target, used_bonus, (10, 11, 12), [1], 3, 100,
                        "raw_draft", "train-1")[0]["last_used_position"] == 2
    altered = PackedRun(used_bonus.generation, (replace(
        trace, argmax_ids=trace.argmax_ids[:-1] + (13,)),))
    with pytest.raises(GateFailure, match="used verification position") as failed:
        check_rounds(target, altered, (10, 11, 12), [1], 3, 100,
                     "raw_draft", "train-1")
    assert failed.value.evidence["position"] == 2

    budget_terminal = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 11]]), None, 2).generate([1], 2)
    trace = budget_terminal.traces[0]
    altered = PackedRun(budget_terminal.generation, (replace(
        trace, argmax_ids=trace.argmax_ids[:-1] + (13,)),))
    check = check_rounds(target, altered, (10, 11), [1], 2, 100,
                         "raw_draft", "train-1")[0]
    assert check["last_used_position"] == 1
    assert check["hypothetical_positions"][0]["position"] == 2
    assert check["hypothetical_positions"][0]["argmax_equal"] is False

    eos_target = FakeTarget([1], [10, 0])
    eos_terminal = PackedSpeculativeDecoder(
        eos_target, ScriptedDraft([[10, 0]]), None, 2, eos_token_id=0
    ).generate([1], 8)
    eos_trace = eos_terminal.traces[0]
    eos_altered = PackedRun(eos_terminal.generation, (replace(
        eos_trace, argmax_ids=eos_trace.argmax_ids[:-1] + (13,)),))
    eos_check = check_rounds(eos_target, eos_altered, (10, 0), [1], 8, 0,
                             "raw_draft", "train-1")[0]
    assert eos_check["eos"] and eos_check["last_used_position"] == 1
    assert eos_check["hypothetical_positions"][0]["argmax_equal"] is False


def test_partial_acceptance_checks_correction_but_not_later_branch():
    pytest.importorskip("transformers")
    from scripts.run_packed_p0 import GateFailure, check_rounds

    target = FakeTarget([1], [10, 11, 12])
    packed = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 99]]), None, 2).generate([1], 2)
    trace = packed.traces[0]
    assert trace.accepted_count == 1
    good = check_rounds(target, packed, (10, 11), [1], 2, 100,
                        "raw_draft", "train-1")[0]
    assert good["last_used_position"] == 1
    hypothetical_ids = trace.argmax_ids[:-1] + ((trace.argmax_ids[-1] + 1) % 101,)
    hypothetical = PackedRun(packed.generation, (replace(
        trace, argmax_ids=hypothetical_ids),))
    check = check_rounds(target, hypothetical, (10, 11), [1], 2, 100,
                         "raw_draft", "train-1")[0]
    assert check["hypothetical_positions"][0]["position"] == 2
    assert check["hypothetical_positions"][0]["argmax_equal"] is False
    wrong_correction_ids = trace.argmax_ids[:1] + (
        (trace.argmax_ids[1] + 1) % 101,) + trace.argmax_ids[2:]
    wrong_correction = PackedRun(packed.generation, (replace(
        trace, argmax_ids=wrong_correction_ids),))
    with pytest.raises(GateFailure, match="used verification position"):
        check_rounds(target, wrong_correction, (10, 11), [1], 2, 100,
                     "raw_draft", "train-1")


def test_gate_rejects_final_token_eos_and_truncation_mismatches():
    pytest.importorskip("transformers")
    from scripts.run_packed_p0 import GateFailure, check_rounds

    target = FakeTarget([1], [10, 0])
    packed = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 0]]), None, 2, eos_token_id=0
    ).generate([1], 8)
    for bad_tokens in ((10, 1), (10,), (10, 0, 4)):
        bad = PackedRun(replace(packed.generation, tokens=bad_tokens), packed.traces)
        with pytest.raises(GateFailure, match="output/EOS/truncation"):
            check_rounds(target, bad, (10, 0), [1], 8, 0,
                         "raw_draft", "train-1")


def test_gpu_admission_maps_logical_to_physical_and_requires_three_idle_samples(monkeypatch):
    pytest.importorskip("transformers")
    from scripts import run_packed_p0 as gate

    calls = []
    def fake_smi(*args):
        calls.append(args)
        if args[0].startswith("--query-gpu"):
            return ["0, GPU-A, 1000, 99", "2, GPU-C, 23000, 0"]
        return ["GPU-A, 1234"]  # Another GPU's process must not block GPU 2.

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(gate, "_nvidia_smi", fake_smi)
    monkeypatch.setattr(gate.time, "sleep", lambda _seconds: None)
    admitted = gate.admit_gpu("cuda:0")
    assert admitted["physical_gpu_index"] == 2
    assert len(admitted["samples"]) == 3 and len(calls) == 6
    assert all(s["compute_pids"] == [] and s["free_mib"] >= 22000
               and s["util_percent"] < 10 for s in admitted["samples"])

    monkeypatch.setattr(gate, "_nvidia_smi", lambda *args: (
        ["2, GPU-C, 23000, 0"] if args[0].startswith("--query-gpu")
        else ["GPU-C, 5678"]))
    with pytest.raises(gate.GateFailure, match="not exclusively idle") as busy:
        gate.admit_gpu("cuda:0", pause_seconds=0)
    assert busy.value.evidence["samples"][0]["compute_pids"] == [5678]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-C")
    with pytest.raises(gate.GateFailure, match="ambiguous"):
        gate.admit_gpu("cuda:0", pause_seconds=0)


def test_full_coverage_probes_use_real_target_decisions_not_substitution():
    pytest.importorskip("transformers")
    from scripts.run_packed_p0 import controlled_coverage_probes, coverage_summary

    target = FakeTarget([1], [10, 11, 0])
    prompt = {"id": "train-1", "input_ids": [1]}
    natural = {"train-1": autoregressive_generate(target, [1], 8, 0)}
    rows = controlled_coverage_probes(target, [prompt], natural, 0, 8)
    assert {row["arm"] for row in rows} == {
        "controlled_zero", "controlled_partial", "controlled_full_bonus",
        "controlled_accepted_eos",
    }
    assert all(row["coverage_probe_only"] for row in rows)
    coverage = coverage_summary([], rows)
    assert coverage["missing"] == []
    terminal = next(row for row in rows if row["arm"] == "controlled_accepted_eos")
    assert terminal["round_checks"][0]["valid_hidden_source"] == (
        "terminal_no_outgoing_feature")
    assert terminal["round_checks"][0]["valid_hidden_bitwise_equal_to_full_prefix"] is None


def test_gate_records_packed_hidden_difference_without_replacing_decision():
    pytest.importorskip("transformers")
    from scripts.run_packed_p0 import check_rounds

    target = FakeTarget([1], [10, 11])
    packed = PackedSpeculativeDecoder(target, ScriptedDraft([[99]]), None, 1).generate([1], 1)
    record = packed.generation.rounds[0]
    altered = PackedRun(
        replace(packed.generation, rounds=(replace(
            record, observation_hidden=record.observation_hidden + 1),)),
        packed.traces,
    )
    checks = check_rounds(target, altered, (10,), [1], 1, 100, "raw_draft", "train-1")
    assert checks[0]["valid_hidden_max_abs_diff_fp32"] == 1.0
    assert checks[0]["valid_hidden_bitwise_equal_to_full_prefix"] is False
    assert altered.generation.tokens == (10,)


def test_versioned_feature_gate_requires_same_forward_summary_and_causal_suffix():
    pytest.importorskip("transformers")
    from scripts.run_packed_p0 import GateFailure, check_rounds

    target = FakeTarget([1], [10, 11, 12, 13])
    packed = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 99], [12]]), None, 2,
        feature_provenance_gate=True,
    ).generate([1], 3)
    checks = check_rounds(target, packed, (10, 11, 12), [1], 3, 100,
                          "raw_draft", "train-1", feature_provenance=True)
    assert len(checks) == 2
    assert all(c["feature_provenance"]["feature_protocol_version"]
               == PACKED_FEATURE_PROTOCOL_VERSION for c in checks)
    assert checks[0]["feature_provenance"]["online_cache_causality"][
        "cache_source"] == "same_online_preverify_KV_snapshot"
    assert checks[0]["feature_provenance"]["fresh_prefill_diagnostic_only"][
        "used_argmax_equal"]
    assert checks[0]["feature_provenance"]["online_cache_causality"][
        "logits_bitwise_equal"]

    first_trace = packed.traces[0]
    hypothetical_ids = first_trace.argmax_ids[:-1] + (
        (first_trace.argmax_ids[-1] + 1) % target.vocab_size,)
    hypothetical = PackedRun(packed.generation, (replace(
        first_trace, argmax_ids=hypothetical_ids), *packed.traces[1:]))
    revised = check_rounds(target, hypothetical, (10, 11, 12), [1], 3, 100,
                           "raw_draft", "train-1", feature_provenance=True)
    assert revised[0]["hypothetical_positions"][0]["argmax_equal"] is False

    first = packed.generation.rounds[0]
    missing_summary = PackedRun(
        replace(packed.generation, rounds=(
            replace(first, observation_logit_summary=None),
            *packed.generation.rounds[1:])), packed.traces)
    with pytest.raises(GateFailure, match="original-forward provenance"):
        check_rounds(target, missing_summary, (10, 11, 12), [1], 3, 100,
                     "raw_draft", "train-1", feature_provenance=True)


def test_online_feature_gate_allows_fresh_prefill_history_difference():
    """A fresh prefill is diagnostic, while causal comparison keeps online KV."""
    pytest.importorskip("transformers")
    from scripts.run_packed_p0 import check_rounds

    target = FakeTarget([1], [10, 11, 12, 13])
    original = target.model.forward

    def history_sensitive_forward(**kwargs):
        previous = getattr(kwargs.get("past_key_values"), "history_bias", 0)
        output = original(**kwargs)
        output.past_key_values.history_bias = previous + int(kwargs["input_ids"].shape[1])
        if output.hidden_states is not None:
            # The hook and returned hidden refer to the same tensor. This
            # emulates BF16 cache-history dependence without changing tokens.
            output.hidden_states[-1][..., 0] += float(previous)
        return output

    target.model.forward = history_sensitive_forward
    packed = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 99], [12, 98]]), None, 2,
        feature_provenance_gate=True,
    ).generate([1], 4)
    checks = check_rounds(target, packed, (10, 11, 12, 13), [1], 4,
                          100, "raw_draft", "train-1", feature_provenance=True)
    assert len(checks) == 2
    second = checks[1]["feature_provenance"]
    assert second["online_cache_causality"]["passed"] is True
    assert second["online_cache_causality"]["hidden_bitwise_equal"] is True
    assert second["fresh_prefill_diagnostic_only"][
        "valid_hidden_bitwise_equal_to_online"] is False
    assert checks[1]["valid_hidden_bitwise_equal_to_full_prefix"] is False


def test_full_accept_bonus_emits_original_forward_feature_for_next_round():
    pytest.importorskip("transformers")
    from scripts.run_packed_p0 import check_rounds

    target = FakeTarget([1], [10, 11, 12, 13, 14])
    packed = PackedSpeculativeDecoder(
        target, ScriptedDraft([[10, 11], [13, 99]]), None, 2,
        feature_provenance_gate=True,
    ).generate([1], 5)
    first = packed.generation.rounds[0]
    assert first.accepted_count == 2 and first.bonus_token == 12
    assert first.observation_hidden is not None
    assert first.observation_logit_summary.greedy_token_id == 12
    assert packed.generation.rounds[1].prefix_before == (1, 10, 11, 12)
    checks = check_rounds(target, packed, (10, 11, 12, 13, 14), [1], 5,
                          100, "raw_draft", "train-1", feature_provenance=True)
    first_feature = checks[0]["feature_provenance"]["online_cache_causality"]
    assert first_feature["applicable"] is False
    assert first_feature["reason"] == "no_unconfirmed_suffix_in_original_query"
    assert first_feature["snapshot_replay_hidden_bitwise_equal"] is True
    assert first_feature["snapshot_replay_logits_bitwise_equal"] is True
    assert first_feature["decision_feature_available"] is True


def test_online_feature_gate_rejects_candidate_suffix_leak():
    pytest.importorskip("transformers")
    from scripts.run_packed_p0 import GateFailure, check_rounds

    target = FakeTarget([1], [10])
    original = target.model.forward

    def leaky_forward(**kwargs):
        output = original(**kwargs)
        if output.hidden_states is not None:
            # For the zero-acceptance case the valid row is the last
            # confirmed token; it must not read either candidate token.
            output.hidden_states[-1][0, 0, 0] += float(
                kwargs["input_ids"][0, 1:].sum())
        return output

    target.model.forward = leaky_forward
    packed = PackedSpeculativeDecoder(
        target, ScriptedDraft([[99, 88]]), None, 2,
        feature_provenance_gate=True,
    ).generate([1], 1)
    assert packed.traces[0].feature_provenance["passed"] is False
    with pytest.raises(GateFailure, match="depends on unconfirmed candidates"):
        check_rounds(target, packed, (10,), [1], 1,
                     100, "raw_draft", "train-1", feature_provenance=True)


def test_draft_config_and_optional_generation_sha_are_recorded_and_validated(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from scripts import run_packed_p0 as gate

    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text("target", encoding="utf-8")
    (draft / "model.safetensors").write_bytes(b"pinned draft weight fixture")
    (draft / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "vocab_size": 17, "hidden_size": 10,
    }), encoding="utf-8")
    (draft / "generation_config.json").write_text('{"eos_token_id": 2}', encoding="utf-8")
    status = tmp_path / "download.json"
    status.write_text(json.dumps({
        "state": "complete", "repo_id": gate.TARGET_REPO,
        "hf_revision": gate.TARGET_REVISION,
        "verified_files": [{"file": "config.json",
                            "sha256": gate.sha256_file(target / "config.json")}],
    }), encoding="utf-8")
    monkeypatch.setattr(gate, "DRAFT_WEIGHT_SHA256",
                        gate.sha256_file(draft / "model.safetensors"))
    hashes = gate.verify_models(target, draft, status)
    assert hashes["draft/config.json"] == gate.sha256_file(draft / "config.json")
    assert hashes["draft/generation_config.json"] == gate.sha256_file(
        draft / "generation_config.json")

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        def get_vocab(self):
            return {str(i): i for i in range(16)}
        def __len__(self):
            return 16

    input_table = torch.nn.Embedding(17, 10)
    output_head = torch.nn.Linear(10, 17, bias=False)
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3", vocab_size=17, hidden_size=10),
        get_input_embeddings=lambda: input_table,
        get_output_embeddings=lambda: output_head,
    )
    metadata = gate.verify_draft_metadata(draft, model, Tokenizer(), hashes)
    assert metadata == {"model_type": "qwen3", "vocab_size": 17,
                        "hidden_size": 10, "tokenizer_length": 16,
                        "tokenizer_max_id": 15, "generation_config_present": True}
    with pytest.raises(RuntimeError, match="hidden_size"):
        gate.verify_draft_metadata(
            draft, SimpleNamespace(**{**model.__dict__, "config": SimpleNamespace(
                model_type="qwen3", vocab_size=17, hidden_size=11)}), Tokenizer(), hashes)
    (draft / "generation_config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="generation_config changed"):
        gate.verify_draft_metadata(draft, model, Tokenizer(), hashes)


def test_draft_tokenizer_id_cannot_exceed_loaded_embedding_table(tmp_path):
    pytest.importorskip("transformers")
    from scripts import run_packed_p0 as gate

    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "vocab_size": 17, "hidden_size": 10,
    }), encoding="utf-8")
    hashes = {"draft/config.json": gate.sha256_file(draft / "config.json")}
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3", vocab_size=17, hidden_size=10),
        get_input_embeddings=lambda: torch.nn.Embedding(17, 10),
        get_output_embeddings=lambda: torch.nn.Linear(10, 17, bias=False),
    )
    class BadTokenizer:
        bos_token_id = None
        eos_token_id = None
        pad_token_id = None
        def get_vocab(self):
            return {"bad": 17}
        def __len__(self):
            return 1
    with pytest.raises(RuntimeError, match="tokenizer IDs exceed"):
        gate.verify_draft_metadata(draft, model, BadTokenizer(), hashes)


def test_tiny_hf_dynamic_cache_matches_full_prefix_cpu():
    transformers = pytest.importorskip("transformers")
    from feedback_codec.hf_backend import HFTargetBackend

    class TinyTokenizer:
        pass

    config = transformers.Qwen3Config(
        vocab_size=41, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2,
        pad_token_id=0, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(31)
    target = HFTargetBackend(transformers.Qwen3ForCausalLM(config), TinyTokenizer())
    prompt = [1, 8, 11]
    first = target.next(prompt).token_id
    second = target.next(prompt + [first]).token_id
    session = HFPackedTargetSession(target, prompt)
    v = session.verify([first, second])
    assert v.argmax_ids == tuple(target.next(prompt + [first, second][:i]).token_id
                                  for i in range(3))
    assert session.commit(v, 2, v.argmax_ids[-1]) == len(prompt) + 2
    assert session.cache_length == len(session.confirmed) - 1

    # A rejected cached branch must not contaminate the next Target decision.
    rejected = (first + 1) % target.vocab_size
    reset = HFPackedTargetSession(target, prompt)
    bad = reset.verify([rejected, 7])
    assert bad.argmax_ids[0] == first
    assert reset.commit(bad, 0, first) == len(prompt)
    assert reset.confirmed == prompt + [first]
    subsequent = reset.verify([second])
    assert subsequent.argmax_ids[0] == target.next(prompt + [first]).token_id
