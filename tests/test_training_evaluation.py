from __future__ import annotations

import torch

from feedback_codec.codec import CodecConfig, FeedbackCodec
from feedback_codec.decoding import SpeculativeDecoder, TargetStep
from feedback_codec.evaluation import Prompt, evaluate_paired
from feedback_codec.pipeline import PilotConfig, load_prompt_jsonl, run_reference_pilot
from feedback_codec.training import collect_trajectories, train_examples


class ToyTarget:
    def __init__(self, vocab: int = 11, hidden: int = 6):
        self.vocab = vocab
        self.model = torch.nn.Embedding(vocab, hidden)
        self.model.requires_grad_(False)
        self.calls: list[tuple[int, ...]] = []

    def next(self, prefix: list[int]) -> TargetStep:
        self.calls.append(tuple(prefix))
        return TargetStep(
            token_id=(prefix[-1] + 1) % self.vocab,
            hidden=self.model(torch.tensor(prefix[-1])).detach(),
        )

    def embedding(self, token_id: int) -> torch.Tensor:
        return self.model(torch.tensor(token_id)).detach()


class ToyDraft:
    def __init__(self, vocab: int = 11, hidden: int = 5):
        self.vocab = vocab
        self.model = torch.nn.Embedding(vocab, hidden)
        self.model.requires_grad_(False)

    def next(self, prefix: list[int], codes, position: int) -> int:
        return (prefix[-1] + 2) % self.vocab

    def teacher_hidden(self, prefix: list[int], continuation: list[int]) -> torch.Tensor:
        tokens = [prefix[-1], *continuation[:-1]]
        return self.model(torch.tensor(tokens)).detach()


def _system():
    target, draft = ToyTarget(), ToyDraft()
    head = torch.nn.Linear(5, 11, bias=False)
    codec = FeedbackCodec(
        CodecConfig(target_hidden_size=6, draft_hidden_size=5, vocab_size=11, window=2),
        head,
    )
    return target, draft, codec


def test_collect_then_train_without_future_leakage_or_base_updates():
    target, draft, codec = _system()
    decoder = SpeculativeDecoder(target, draft, None, window=2)
    examples = collect_trajectories(
        decoder, [("a", [1]), ("b", [3])], max_new_tokens=5, phase="baseline"
    )
    assert examples
    assert all(e.phase == "baseline" and e.continuation for e in examples)
    assert all(e.valid_prefix + (e.correction_token,) == e.corrected_prefix for e in examples)
    frozen_target = target.model.weight.detach().clone()
    frozen_draft = draft.model.weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in codec.parameters() if p.requires_grad), lr=1e-3)
    history = train_examples(
        examples=examples,
        target=target,
        draft=draft,
        codec=codec,
        optimizer=optimizer,
        steps=2,
        prefix_weight=0.5,
        seed=7,
    )
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in history)
    assert torch.equal(target.model.weight, frozen_target)
    assert torch.equal(draft.model.weight, frozen_draft)


def test_paired_acceptance_reports_exactness_and_no_latency_claim():
    target, draft, codec = _system()
    result = evaluate_paired(
        target=target,
        draft=draft,
        codec=codec,
        prompts=[Prompt("a", (1,)), Prompt("b", (3,))],
        window=2,
        max_new_tokens=5,
        bootstrap_repetitions=100,
    )
    assert len(result["rows"]) == 6
    assert all(row["exact_target_ar"] for row in result["rows"])
    assert result["latency_claim_valid"] is False
    assert result["paired_learned_minus_no_feedback"]["mean_accepted_draft_tokens_per_round"] == 0
    assert result["summary"]["no_feedback"]["feedback_bytes_per_request"] == 0
    assert result["summary"]["learned_feedback"]["feedback_bytes_per_request"] > 0


def test_pipeline_recollects_trajectories_after_warmup(tmp_path):
    target, draft, codec = _system()
    output = tmp_path / "run"
    result = run_reference_pilot(
        target=target,
        draft=draft,
        codec=codec,
        train_prompts=[Prompt("train_a", (1,)), Prompt("train_b", (2,))],
        test_prompts=[Prompt("test_a", (3,)), Prompt("test_b", (4,))],
        output_dir=output,
        config=PilotConfig(
            warmup_steps=1,
            refresh_steps=1,
            train_max_new_tokens=5,
            test_max_new_tokens=5,
            bootstrap_repetitions=100,
        ),
        eos_token_id=None,
    )
    assert (output / "baseline_trajectories.json").exists()
    assert (output / "refreshed_trajectories.json").exists()
    assert (output / "after_warmup.pt").exists()
    assert (output / "final_codec.pt").exists()
    assert len(result["rows"]) == 6


def test_tiny_hf_models_complete_training_and_exact_evaluation(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend

    class Tokenizer:
        special_tokens_map = {"bos_token": "t1", "eos_token": "t2"}
        all_special_ids = [1, 2]
        added_tokens_decoder = {}
        chat_template = "{{ messages }}"

        def get_vocab(self):
            return {f"t{i}": i for i in range(31)}

    config = Qwen3Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(11)
    target = HFTargetBackend(Qwen3ForCausalLM(config), Tokenizer())
    torch.manual_seed(12)
    draft_model = Qwen3ForCausalLM(config)
    codec = FeedbackCodec(
        CodecConfig(target_hidden_size=16, draft_hidden_size=16, vocab_size=31, window=2),
        draft_model.get_output_embeddings(),
    )
    draft = HFDraftBackend(draft_model, Tokenizer(), codec, target=target)
    result = run_reference_pilot(
        target=target,
        draft=draft,
        codec=codec,
        train_prompts=[Prompt("train", (1, 4, 7))],
        test_prompts=[Prompt("test", (1, 9, 13))],
        output_dir=tmp_path / "tiny_hf",
        config=PilotConfig(
            warmup_steps=1,
            refresh_steps=1,
            train_max_new_tokens=6,
            test_max_new_tokens=6,
            bootstrap_repetitions=100,
        ),
        eos_token_id=None,
    )
    assert len(result["rows"]) == 2
    assert all(row["exact_target_ar"] for row in result["rows"])


def test_token_manifest_rejects_duplicate_prompt_across_splits(tmp_path):
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(
        '{"id":"a","split":"train","input_ids":[1,2]}\n'
        '{"id":"b","split":"test","input_ids":[1,2]}\n',
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="duplicate prompt token sequence"):
        load_prompt_jsonl(manifest, vocab_size=10)
