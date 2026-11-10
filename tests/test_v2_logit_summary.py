"""CPU checks for v2's legal current-boundary FULL-LOGIT feature."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from feedback_codec.decoding import TargetLogitSummary, TargetStep
from feedback_codec.fixed_probe import (
    ProbeExample, ProbeObservation, ProbePrompt, collect_fixed_probe,
)
from feedback_codec.hf_backend import HFTargetBackend


def test_target_logit_summary_is_from_one_current_prefix_forward():
    config = transformers.LlamaConfig(
        vocab_size=41, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(79)
    target = HFTargetBackend(transformers.LlamaForCausalLM(config), object())
    count = []
    hook = target.model.register_forward_hook(lambda *_: count.append(1))
    try:
        step = target.next_with_logit_summary([1, 5, 7])
    finally:
        hook.remove()
    assert len(count) == 1
    assert step.logit_summary is not None
    summary = step.logit_summary
    assert summary.greedy_token_id == step.token_id
    assert len(summary.top_token_ids) == len(summary.top_logits_fp16) == 32
    with torch.inference_mode():
        direct = target.model(input_ids=torch.tensor([[1, 5, 7]]), use_cache=False,
                              output_hidden_states=True, return_dict=True)
    logits = direct.logits[0, -1].float()
    assert summary.greedy_token_id == int(logits.argmax().item())
    assert summary.logsumexp_fp32 == pytest.approx(float(logits.logsumexp(-1).item()))
    assert set(summary.top_token_ids) == set(logits.topk(32).indices.tolist())
    torch.testing.assert_close(step.hidden, direct.hidden_states[-1][0, -1], rtol=0, atol=0)
    assert target.next([1, 5, 7]).logit_summary is None


def test_probe_logit_view_and_roundtrip_never_reveal_future_labels():
    summary = TargetLogitSummary(
        greedy_token_id=9, top_token_ids=tuple(range(32)),
        top_logits_fp16=tuple(float(32 - x) for x in range(32)),
        logsumexp_fp32=33.0, top2_margin_fp32=1.0, vocab_size=41,
    )
    observation = ProbeObservation(
        prompt_id="p", family_id="f", split="train", task_type="chat",
        round_index=0, source_prefix=(1, 4), proposals=(7,),
        valid_prefix=(1, 4), accepted_count=0, rejected_token=7,
        correction_token=9, corrected_prefix=(1, 4, 9),
        target_hidden=torch.arange(4, dtype=torch.float32),
        target_hidden_prefix=(1, 4), remaining_budget=8, window=8,
        target_logit_summary=summary,
    )
    example = ProbeExample(observation, target_future_ids=(3, 8))
    view = observation.target_logit_view()
    assert view.target_logit_summary == summary
    assert not hasattr(view, "target_future_ids")
    assert not hasattr(observation.target_view(), "target_logit_summary")
    restored = ProbeExample.from_json(example.to_json())
    assert restored.observation.target_logit_summary == summary
    assert restored.target_future_ids == (3, 8)
    with pytest.raises(ValueError, match="greedy_token_id|summary"):
        TargetLogitSummary.from_json({**summary.to_json(), "greedy_token_id": 41})


def test_collector_uses_normal_forward_summary_and_caps_first_eight_windows():
    class Target:
        def __init__(self):
            self.summary_calls = 0

        def next(self, prefix):
            return TargetStep((prefix[-1] + 1) % 41,
                              torch.tensor([float(len(prefix))]))

        def next_with_logit_summary(self, prefix):
            self.summary_calls += 1
            plain = self.next(prefix)
            summary = TargetLogitSummary(
                greedy_token_id=plain.token_id,
                top_token_ids=tuple(range(32)),
                top_logits_fp16=tuple(float(32 - x) for x in range(32)),
                logsumexp_fp32=float(self.summary_calls),
                top2_margin_fp32=1.0, vocab_size=41,
            )
            return TargetStep(plain.token_id, plain.hidden, summary)

    class WrongDraft:
        def next(self, _prefix, _codes, _position):
            return 40

    target = Target()
    corpus = collect_fixed_probe(
        target=target, raw_draft=WrongDraft(),
        prompts=[ProbePrompt("p", "family", "train", (1,))],
        window=8, max_new_tokens=16, eos_token_id=None,
        max_probe_windows_per_prompt=8, capture_target_logits=True,
    )
    assert corpus.audit["raw_rounds"] == 16
    assert corpus.audit["probe_windows"] == 8
    assert [example.observation.round_index for example in corpus.examples] == list(range(8))
    assert [example.observation.target_logit_summary.logsumexp_fp32
            for example in corpus.examples] == list(map(float, range(1, 9)))
    assert target.summary_calls == 16
    with pytest.raises(ValueError, match="same-forward"):
        collect_fixed_probe(
            target=object(), raw_draft=WrongDraft(),
            prompts=[ProbePrompt("p", "family", "train", (1,))],
            capture_target_logits=True,
        )
