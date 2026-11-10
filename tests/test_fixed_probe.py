from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest
import torch

from feedback_codec.decoding import TargetStep
from feedback_codec.fixed_probe import (
    EdgeView, PreparedState, ProbeArm, ProbePrompt, ProtocolView,
    TargetView, collect_fixed_probe, evaluate_fixed_probe,
    matched_p1_arm, raw_draft_arm,
)


class ToyTarget:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def next(self, prefix: list[int]) -> TargetStep:
        self.calls.append(tuple(prefix))
        return TargetStep((prefix[-1] + 1) % 20, torch.tensor([float(prefix[-1])]))

    def embedding(self, token_id: int) -> torch.Tensor:
        return torch.tensor([float(token_id)])


class WrongDraft:
    def next(self, prefix: list[int], codes, position: int) -> int:
        return 19


def _corpus():
    target = ToyTarget()
    draft = WrongDraft()
    corpus = collect_fixed_probe(
        target=target,
        raw_draft=draft,
        prompts=[
            ProbePrompt("p1", "family-1", "val", (1,), "math"),
            ProbePrompt("p2", "family-2", "val", (3,), "math"),
        ],
        window=2,
        max_new_tokens=6,
    )
    return target, draft, corpus


def test_baseline_collection_locks_valid_prefix_and_independent_future():
    target, _draft, corpus = _corpus()
    assert corpus.audit["prompts"] == 2
    assert corpus.audit["probe_windows"] == len(corpus.examples) > 0
    assert 0 < corpus.audit["full_nonterminal_windows"] < len(corpus.examples)
    assert all(19 not in call for call in target.calls)
    for example in corpus.examples:
        o = example.observation
        assert o.accepted_count == 0
        assert o.source_prefix == o.valid_prefix == o.target_hidden_prefix
        assert o.corrected_prefix == o.valid_prefix + (o.correction_token,)
        assert float(o.target_hidden.item()) == o.valid_prefix[-1]
        assert example.target_future_ids[0] == (o.corrected_prefix[-1] + 1) % 20
        assert "target_future_ids" not in o.target_view().__dict__
        restored = type(example).from_json(example.to_json())
        assert restored.target_future_ids == example.target_future_ids
        assert torch.equal(restored.observation.target_hidden, o.target_hidden)


def test_probe_is_autoregressive_and_donor_shuffles_only_cloud_message():
    _target, raw, corpus = _corpus()
    seen_views: list[type] = []
    draft_prefixes: list[tuple[str, tuple[int, ...], PreparedState, int]] = []

    def arm(name: str, expected_view: type) -> ProbeArm:
        def prepare(view):
            assert type(view) is expected_view
            seen_views.append(type(view))
            message = (int(view.target_hidden.item()),) * 8 if name == "target_8B" else None
            return PreparedState(local=view.corrected_prefix[-1], message=message)

        def next_token(prefix: list[int], state: PreparedState, position: int) -> int:
            draft_prefixes.append((name, tuple(prefix), state, position))
            # Deliberately depend on the previous emitted token, proving that
            # the second call cannot be a teacher-forced top-1 at a target ID.
            return (prefix[-1] + 2) % 20

        return ProbeArm(prepare=prepare, next=next_token)

    arms = {
        "raw_draft": raw_draft_arm(raw),
        "edge_only": arm("edge_only", EdgeView),
        "protocol_only": arm("protocol_only", ProtocolView),
        "target_8B": arm("target_8B", TargetView),
    }
    report = evaluate_fixed_probe(
        examples=corpus.examples, arms=arms, eos_token_id=None,
        split="val", bootstrap_repetitions=100,
    )
    assert report["scope"] == "fixed_prefix_probe"
    assert report["primary_windows"] == corpus.audit["full_nonterminal_windows"]
    assert report["stratified_shuffle_status"] == "complete_coverage"
    assert {EdgeView, ProtocolView, TargetView} <= set(seen_views)
    by_key = {(r["arm"], r["prompt_id"], r["round_index"]): r for r in report["rows"]}
    for example in corpus.examples:
        o = example.observation
        row = by_key["edge_only", o.prompt_id, o.round_index]
        expected = [(o.corrected_prefix[-1] + 2) % 20,
                    (o.corrected_prefix[-1] + 4) % 20]
        assert row["draft_ids"] == expected[:min(o.window, o.remaining_budget)]
        assert row["accepted_count"] == 0
        global_row = by_key["shuffled_global_8B", o.prompt_id, o.round_index]
        assert global_row["donor"][0] != o.prompt_id
        stratified_row = by_key["shuffled_stratified_8B", o.prompt_id, o.round_index]
        assert stratified_row["donor"][0] != o.prompt_id
    for name, prefix, state, position in draft_prefixes:
        if name == "target_8B":
            # The shuffled call keeps recipient local state.  It never
            # substitutes the donor's corrected prefix or public protocol.
            assert state.local == prefix[-position]


def test_cannot_read_test_by_default_or_cross_prompt_family_splits():
    target, draft, corpus = _corpus()
    with pytest.raises(ValueError, match="sealed"):
        evaluate_fixed_probe(
            examples=[replace(x, observation=replace(x.observation, split="test"))
                      for x in corpus.examples],
            arms={
                "raw_draft": raw_draft_arm(draft),
                "edge_only": ProbeArm(lambda _v: PreparedState(), lambda p, s, pos: 19),
                "protocol_only": ProbeArm(lambda _v: PreparedState(), lambda p, s, pos: 19),
                "target_8B": ProbeArm(lambda _v: PreparedState(message=(0,) * 8),
                                       lambda p, s, pos: 19),
            },
            eos_token_id=None, split="test", bootstrap_repetitions=100,
        )
    with pytest.raises(ValueError, match="family crosses"):
        collect_fixed_probe(
            target=target, raw_draft=draft,
            prompts=[ProbePrompt("a", "same", "train", (1,)),
                     ProbePrompt("b", "same", "val", (2,))],
            window=2, max_new_tokens=3,
        )


def test_bad_message_or_invalid_observation_is_rejected():
    _target, draft, corpus = _corpus()
    with pytest.raises(ValueError, match="eight byte"):
        PreparedState(message=(1,) * 7)
    bad = replace(corpus.examples[0], observation=replace(
        corpus.examples[0].observation, target_hidden_prefix=(999,)
    ))
    arms = {
        "raw_draft": raw_draft_arm(draft),
        "edge_only": ProbeArm(lambda _v: PreparedState(), lambda p, s, pos: 19),
        "protocol_only": ProbeArm(lambda _v: PreparedState(), lambda p, s, pos: 19),
        "target_8B": ProbeArm(lambda _v: PreparedState(message=(0,) * 8),
                               lambda p, s, pos: 19),
    }
    with pytest.raises(ValueError, match="integrity"):
        evaluate_fixed_probe(
            examples=[bad], arms=arms, eos_token_id=None,
            split="val", bootstrap_repetitions=100,
        )


def test_matched_codec_adapter_uses_only_available_prefix_states():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from feedback_codec.codec import CodecConfig, FeedbackCodec, MatchedP1Codec
    from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend

    class Tokenizer:
        special_tokens_map = {"bos_token": "t1", "eos_token": "t2"}
        all_special_ids = [1, 2]
        added_tokens_decoder = {}
        chat_template = "{{ messages }}"

        def get_vocab(self):
            return {f"t{i}": i for i in range(31)}

    config = Qwen3Config(
        vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(9)
    target = HFTargetBackend(Qwen3ForCausalLM(config), Tokenizer())
    draft_model = Qwen3ForCausalLM(config)
    codec_config = CodecConfig(
        target_hidden_size=16, draft_hidden_size=16, vocab_size=31,
        window=8, feature_dim=8, hidden_width=12, rank=4,
    )
    raw_codec = FeedbackCodec(codec_config, draft_model.get_output_embeddings())
    draft = HFDraftBackend(draft_model, Tokenizer(), raw_codec, target=target)
    prefix = (1, 4, 7)
    valid_state = target.next(list(prefix)).hidden
    protocol = ProtocolView(prefix + (8,), 8, 0, 9, 8)
    for mode in ("edge_only", "protocol_only", "target_8B"):
        codec = MatchedP1Codec(codec_config, draft_model.get_output_embeddings(), mode=mode)
        arm = matched_p1_arm(draft, codec)
        view = (
            EdgeView(prefix + (8,), 8) if mode == "edge_only"
            else protocol if mode == "protocol_only"
            else TargetView(**protocol.__dict__, target_hidden=valid_state)
        )
        state = arm.prepare(view)
        assert len(state.local) == 2 and all(len(group) == 8 for group in state.local)
        assert (state.message is not None) == (mode == "target_8B")
        first = arm.next(list(view.corrected_prefix), state, 1)
        second = arm.next(list(view.corrected_prefix) + [first], state, 2)
        assert 0 <= first < 31 and 0 <= second < 31


def test_resumable_chunk_rejects_changed_prompt_or_corrupt_evidence(tmp_path):
    from scripts.build_fixed_probe import (
        _canonical_bytes, _checked_chunk, _chunk_path, _prompt_digest,
        _shard_of,
    )

    _target, _draft, corpus = _corpus()
    prompt = ProbePrompt("p1", "family-1", "val", (1,), "math")
    (tmp_path / "chunks").mkdir()
    path = _chunk_path(tmp_path, prompt)
    chunk = {
        "prompt_id": prompt.id,
        "prompt_sha256": _prompt_digest(prompt),
        "plan_sha256": "fixed-plan",
        "shard_index": _shard_of(prompt, 4),
        "shard_count": 4,
        "window": 2,
        "audit": corpus.audit,
        "examples": [example.to_json() for example in corpus.examples
                     if example.observation.prompt_id == prompt.id],
    }
    chunk["checksum_sha256"] = hashlib.sha256(_canonical_bytes(chunk)).hexdigest()
    path.write_text(json.dumps(chunk), encoding="utf-8")
    assert _checked_chunk(path, prompt, "fixed-plan", 4)["prompt_id"] == prompt.id
    with pytest.raises(ValueError, match="locked prompt/protocol"):
        _checked_chunk(path, replace(prompt, input_ids=(4,)), "fixed-plan", 4)
    path.write_text(path.read_text(encoding="utf-8").replace('"prompt_id": "p1"',
                                                       '"prompt_id": "p2"', 1),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        _checked_chunk(path, prompt, "fixed-plan", 4)


def test_stable_prompt_shards_partition_without_overlap():
    from scripts.build_fixed_probe import _shard_of

    prompts = [ProbePrompt(f"p-{index}", f"family-{index}", "train", (index + 1,))
               for index in range(101)]
    assignments = {prompt.id: _shard_of(prompt, 4) for prompt in prompts}
    assert set(assignments.values()) == {0, 1, 2, 3}
    assert sum(sum(_shard_of(prompt, 4) == shard for prompt in prompts)
               for shard in range(4)) == len(prompts)
    assert assignments == {prompt.id: _shard_of(prompt, 4) for prompt in reversed(prompts)}


def test_real_p0_authorization_rejects_stale_core_source_and_model(tmp_path):
    from scripts.build_fixed_probe import ROOT, _model_hashes, _sha256, _validate_real_p0

    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "model.safetensors").write_bytes(b"target weights")
    (draft / "model.safetensors").write_bytes(b"draft weights")
    sources = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in sorted((ROOT / "feedback_codec").glob("*.py"))
    }
    report = {
        "state": "passed", "scope": "real_qwen3_bf16_full_prefix_p0_correctness_only",
        "dtype": "bfloat16", "window": 8, "dataset_sha256": "locked",
        "target_path": str(target.resolve()), "draft_path": str(draft.resolve()),
        "target_file_sha256": _model_hashes(target),
        "draft_file_sha256": _model_hashes(draft),
        "checks": [{"arm": "raw_draft", "exact_target_ar": True},
                   {"arm": "random_untrained_8B", "exact_target_ar": True}],
        "source_sha256": sources.copy(),
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _validate_real_p0(path, target=target, draft=draft,
                             dataset_sha="locked")["state"] == "passed"
    report["source_sha256"]["feedback_codec/fixed_probe.py"] = "stale"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="current feedback_codec source"):
        _validate_real_p0(path, target=target, draft=draft, dataset_sha="locked")
    report["source_sha256"] = sources
    path.write_text(json.dumps(report), encoding="utf-8")
    (draft / "model.safetensors").write_bytes(b"changed draft weights")
    with pytest.raises(ValueError, match="current model files"):
        _validate_real_p0(path, target=target, draft=draft, dataset_sha="locked")


def test_evaluator_rejects_codec_drift_after_collection():
    from scripts.evaluate_fixed_probe import ROOT, _sha256, _validate_collection_sources

    source_names = (
        "feedback_codec/codec.py",
        "feedback_codec/fixed_probe.py",
        "feedback_codec/decoding.py",
        "feedback_codec/hf_backend.py",
    )
    source = {name: _sha256(ROOT / name) for name in source_names}
    _validate_collection_sources({"source_sha256": source})
    changed = {**source, "feedback_codec/codec.py": "stale"}
    with pytest.raises(ValueError, match="codec/metric/backend source changed"):
        _validate_collection_sources({"source_sha256": changed})


def test_stratified_shuffle_reports_sparse_stratum_coverage():
    _target, draft, corpus = _corpus()
    examples = list(corpus.examples)
    examples[0] = replace(examples[0], observation=replace(
        examples[0].observation, task_type="rare-only"
    ))
    arms = {
        "raw_draft": raw_draft_arm(draft),
        "edge_only": ProbeArm(lambda _v: PreparedState(), lambda p, s, pos: 19),
        "protocol_only": ProbeArm(lambda _v: PreparedState(), lambda p, s, pos: 19),
        "target_8B": ProbeArm(
            lambda v: PreparedState(message=(int(v.target_hidden.item()),) * 8),
            lambda p, s, pos: 19,
        ),
    }
    report = evaluate_fixed_probe(
        examples=examples, arms=arms, eos_token_id=None,
        split="val", bootstrap_repetitions=100,
    )
    assert report["stratified_shuffle_status"] == "eligible_subset_only"
    assert report["stratified_shuffle_coverage"]["excluded_windows"] == 1
    assert report["stratified_shuffle_coverage"]["windows"] == len(examples) - 1
    assert report["stratified_eligible_primary_summary"] is not None
