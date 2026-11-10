"""Tiny CPU checks for the HF correctness adapters, not model quality evidence."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
from feedback_codec.codec import CodecConfig, FeedbackCodec


class TinyTokenizer:
    def __init__(self, *, eos_token="tok2", shift=0):
        self._vocab = {f"tok{i}": i + shift for i in range(41)}
        self.special_tokens_map = {"bos_token": "tok1", "eos_token": eos_token}
        self.all_special_ids = [1 + shift, 2 + shift]
        self.added_tokens_decoder = {}
        self.chat_template = "{{ messages }}"

    def get_vocab(self):
        return dict(self._vocab)


class ProbeCodec(torch.nn.Module):
    def __init__(self, vocab_size=41):
        super().__init__()
        self.vocab_size = vocab_size
        self.calls = []

    def repair_logits(self, draft_hidden, codes, position, no_feedback=False):
        self.calls.append((draft_hidden.detach().clone(), codes.clone(), position))
        logits = torch.full(
            (draft_hidden.shape[0], self.vocab_size),
            -100.0,
            device=draft_hidden.device,
        )
        logits[:, 7] = 100.0
        return logits


@pytest.fixture(params=("llama", "qwen3"))
def tiny_models(request):
    config_class = (
        transformers.LlamaConfig if request.param == "llama" else transformers.Qwen3Config
    )
    model_class = (
        transformers.LlamaForCausalLM
        if request.param == "llama"
        else transformers.Qwen3ForCausalLM
    )
    cfg = config_class(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(31)
    target = model_class(cfg)
    torch.manual_seed(32)
    draft = model_class(cfg)
    return target, draft


def _pair(tiny_models):
    target_model, draft_model = tiny_models
    target = HFTargetBackend(target_model, TinyTokenizer())
    codec = ProbeCodec()
    draft = HFDraftBackend(draft_model, TinyTokenizer(), codec, target=target)
    return target, draft, codec


def test_target_greedy_and_final_norm_hidden_match_direct_hf(tiny_models):
    target, _, _ = _pair(tiny_models)
    prefix = [1, 8, 11]
    result = target.next(prefix)
    with torch.inference_mode():
        direct = target.model(
            input_ids=torch.tensor([prefix]),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    assert result.token_id == int(direct.logits[0, -1].argmax().item())
    torch.testing.assert_close(result.hidden, direct.hidden_states[-1][0, -1], rtol=0, atol=0)
    torch.testing.assert_close(
        target.embedding(prefix[-1]),
        target.model.get_input_embeddings().weight[prefix[-1]],
        rtol=0,
        atol=0,
    )
    assert not any(parameter.requires_grad for parameter in target.model.parameters())


def test_draft_baseline_and_eight_byte_repair(tiny_models):
    _, draft, codec = _pair(tiny_models)
    prefix = [1, 3, 4]
    with torch.inference_mode():
        direct = draft.model(
            input_ids=torch.tensor([prefix]),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    assert draft.next(prefix, None, 1) == int(direct.logits[0, -1].argmax().item())
    assert codec.calls == []
    codes = (1, 2, 3, 4, 5, 6, 7, 255)
    assert draft.next(prefix, codes, 3) == 7
    assert len(codec.calls) == 1
    hidden, received_codes, position = codec.calls[0]
    torch.testing.assert_close(hidden[0], direct.hidden_states[-1][0, -1], rtol=0, atol=0)
    assert received_codes.tolist() == [list(codes)]
    assert position == 3
    assert not any(parameter.requires_grad for parameter in draft.model.parameters())


def test_real_codec_uses_the_drafts_frozen_output_head(tiny_models):
    target_model, draft_model = tiny_models
    target = HFTargetBackend(target_model, TinyTokenizer())
    codec = FeedbackCodec(
        CodecConfig(target_hidden_size=32, draft_hidden_size=32, vocab_size=41),
        draft_model.get_output_embeddings(),
    )
    draft = HFDraftBackend(draft_model, TinyTokenizer(), codec, target=target)
    prefix = [1, 3, 4]
    codes = (0,) * 8
    with torch.inference_mode():
        direct = draft_model(
            input_ids=torch.tensor([prefix]),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        repaired = codec.repair_logits(
            direct.hidden_states[-1][:, -1],
            torch.tensor([codes], dtype=torch.long),
            1,
        )
    assert draft.next(prefix, codes, 1) == int(repaired[0].argmax().item())
    assert codec.output_head is draft.model.get_output_embeddings()


def test_teacher_hidden_is_aligned_with_each_next_continuation_token(tiny_models):
    _, draft, _ = _pair(tiny_models)
    prefix = [1, 4, 9]
    continuation = [6, 7, 8, 2]
    actual = draft.teacher_hidden(prefix, continuation)
    teacher_input = prefix + continuation[:-1]
    with torch.inference_mode():
        direct = draft.model.model(
            input_ids=torch.tensor([teacher_input]),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    start = len(prefix) - 1
    expected = direct.hidden_states[-1][0, start : start + len(continuation)]
    assert actual.shape == (len(continuation), draft.hidden_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert draft.teacher_hidden(prefix, []).shape == (0, draft.hidden_size)


@pytest.mark.parametrize(
    "draft_tokenizer",
    [TinyTokenizer(eos_token="tok3"), TinyTokenizer(shift=1)],
)
def test_pair_rejects_tokenizer_or_special_mismatch(tiny_models, draft_tokenizer):
    target_model, draft_model = tiny_models
    target = HFTargetBackend(target_model, TinyTokenizer())
    with pytest.raises(ValueError, match="token-ID mappings|special tokens"):
        HFDraftBackend(draft_model, draft_tokenizer, ProbeCodec(), target=target)


def test_feedback_must_be_exactly_eight_bytes(tiny_models):
    _, draft, _ = _pair(tiny_models)
    with pytest.raises(ValueError, match="one-based"):
        draft.next([1, 3], None, 0)
    with pytest.raises(ValueError, match="eight byte"):
        draft.next([1, 3], (1, 2), 1)
    with pytest.raises(ValueError, match="eight byte"):
        draft.next([1, 3], (0, 1, 2, 3, 4, 5, 6, 256), 1)
