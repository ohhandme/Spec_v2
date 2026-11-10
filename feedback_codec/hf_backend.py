"""Frozen Hugging Face correctness backends for feedback coded decoding.

These backends intentionally run a full, eager prefix forward. They provide an
independent greedy reference and aligned hidden states, not a latency result.
Only dense Llama/Qwen2/Qwen3 causal models with a final decoder ``norm`` are
supported; a hook checks that the returned final hidden state is post norm.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .codec import FeedbackCodec
    from .decoding import TargetStep


_SUPPORTED_MODEL_TYPES = frozenset({"llama", "qwen2", "qwen3"})


def _load_model_and_tokenizer(
    model: Any,
    tokenizer: Any | None,
    *,
    device: str | torch.device | None,
    dtype: torch.dtype | None,
) -> tuple[Any, Any]:
    if isinstance(model, (str, Path)):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = str(model)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
        options: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "attn_implementation": "eager",
        }
        if dtype is not None:
            options["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(path, **options)
    if tokenizer is None:
        raise ValueError("a tokenizer is required with an already loaded model")
    if isinstance(tokenizer, (str, Path)):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer), local_files_only=True, trust_remote_code=False
        )
    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    if getattr(model.config, "model_type", None) not in _SUPPORTED_MODEL_TYPES:
        raise ValueError("only dense Llama/Qwen2/Qwen3 causal models are supported")
    decoder = getattr(model, "model", None)
    if decoder is None or not isinstance(getattr(decoder, "norm", None), torch.nn.Module):
        raise ValueError("the model must expose its final decoder norm as model.norm")
    if model.get_input_embeddings() is None or model.get_output_embeddings() is None:
        raise ValueError("the model must expose input embeddings and an output head")
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def _specials(tokenizer: Any) -> tuple[Any, ...]:
    # get_vocab includes added-token IDs. The remaining fields cover special
    # token roles and formatting that get_vocab alone cannot distinguish.
    added = getattr(tokenizer, "added_tokens_decoder", {})
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = backend.to_str() if backend is not None else None
    return (
        type(tokenizer),
        backend_json,
        getattr(tokenizer, "special_tokens_map", None),
        tuple(getattr(tokenizer, "all_special_ids", ())),
        tuple(sorted((int(k), repr(v)) for k, v in added.items())),
        getattr(tokenizer, "chat_template", None),
    )


def assert_compatible_pair(target: "HFTargetBackend", draft: "HFDraftBackend") -> None:
    """Reject any target/draft token-ID or special-token mismatch."""
    target_vocab = int(target.model.config.vocab_size)
    draft_vocab = int(draft.model.config.vocab_size)
    if target_vocab != draft_vocab:
        raise ValueError("target and draft model vocabulary sizes differ")
    for model in (target.model, draft.model):
        if int(model.get_output_embeddings().weight.shape[0]) != target_vocab:
            raise ValueError("an output head does not cover the model vocabulary")
    if target.tokenizer.get_vocab() != draft.tokenizer.get_vocab():
        raise ValueError("target and draft token-ID mappings differ")
    if _specials(target.tokenizer) != _specials(draft.tokenizer):
        raise ValueError("target and draft special tokens or templates differ")
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(target.model.config, field, None) != getattr(draft.model.config, field, None):
            raise ValueError(f"target and draft model {field} differ")


class _FrozenHFBackend:
    def __init__(
        self,
        model: Any,
        tokenizer: Any | None,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model, self.tokenizer = _load_model_and_tokenizer(
            model, tokenizer, device=device, dtype=dtype
        )
        self.vocab_size = int(self.model.config.vocab_size)
        self.hidden_size = int(self.model.config.hidden_size)
        self._lock = threading.RLock()

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def _ids(self, tokens: list[int]) -> torch.Tensor:
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("prefix must be a nonempty list of token IDs")
        if any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens):
            raise ValueError("token IDs must be integers within the model vocabulary")
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if maximum is not None and len(tokens) > int(maximum):
            raise ValueError("prefix exceeds the model context limit")
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def _forward_with_final_hidden(
        self, ids: torch.Tensor, *, decoder_only: bool = False
    ) -> tuple[Any, torch.Tensor]:
        """Verify hidden_states[-1] is exactly the final norm's output."""
        norm_output: list[torch.Tensor] = []

        def save_norm(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("final norm returned a non-tensor output")
            norm_output.append(output)

        hook = self.model.model.norm.register_forward_hook(save_norm)
        try:
            module = self.model.model if decoder_only else self.model
            output = module(
                input_ids=ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            hook.remove()
        states = getattr(output, "hidden_states", None)
        if states is None or not states or len(norm_output) != 1:
            raise RuntimeError("could not capture exactly one final-norm hidden state")
        final = states[-1]
        if not isinstance(final, torch.Tensor) or not torch.equal(final, norm_output[0]):
            raise RuntimeError("hidden_states[-1] is not the final-norm decoder output")
        if final.shape[:2] != ids.shape or final.shape[-1] != self.hidden_size:
            raise RuntimeError("unexpected final hidden-state shape")
        return output, final


class HFTargetBackend(_FrozenHFBackend):
    """A frozen greedy target with an effective-prefix hidden-state interface."""

    @torch.inference_mode()
    def next(self, prefix: list[int]) -> "TargetStep":
        return self._step(prefix, include_logit_summary=False)

    @torch.inference_mode()
    def next_with_logit_summary(self, prefix: list[int]) -> "TargetStep":
        """Observe current-boundary logits and hidden in one legal forward.

        The returned summary is diagnostic high-bandwidth information. It is
        computed before the correction/bonus token and carries no future
        Target decisions. Ordinary ``next`` avoids this extra GPU reduction.
        """
        return self._step(prefix, include_logit_summary=True)

    def _step(self, prefix: list[int], *, include_logit_summary: bool) -> "TargetStep":
        from .decoding import TargetLogitSummary, TargetStep

        with self._lock:
            output, final = self._forward_with_final_hidden(self._ids(prefix))
            logits = output.logits[0, -1]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite target logits")
            token_id = int(logits.argmax(dim=-1).item())
            summary = None
            if include_logit_summary:
                # Reductions use float32 over the same BF16/FP16 model logits.
                # In particular, no second Target forward is performed.
                logits32 = logits.float()
                count = min(32, self.vocab_size)
                values, ids = logits32.topk(count, dim=-1)
                values16 = values.to(torch.float16)
                if not torch.isfinite(values16).all():
                    raise FloatingPointError("target top logits overflow float16 summary")
                summary = TargetLogitSummary(
                    greedy_token_id=token_id,
                    top_token_ids=tuple(int(x) for x in ids.cpu().tolist()),
                    top_logits_fp16=tuple(float(x) for x in values16.cpu().tolist()),
                    logsumexp_fp32=float(logits32.logsumexp(-1).item()),
                    top2_margin_fp32=(
                        float((values[0] - values[1]).item()) if count >= 2 else 0.0
                    ),
                    vocab_size=self.vocab_size,
                )
            return TargetStep(
                token_id=token_id,
                hidden=final[0, -1].detach().clone(),
                logit_summary=summary,
            )

    @torch.inference_mode()
    def embedding(self, token_id: int) -> torch.Tensor:
        if type(token_id) is not int or not 0 <= token_id < self.vocab_size:
            raise ValueError("embedding token ID is outside the model vocabulary")
        return self.model.get_input_embeddings().weight[token_id].detach().clone()


class HFDraftBackend(_FrozenHFBackend):
    """A frozen draft base with optional output-side feedback repair."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any | None,
        codec: "FeedbackCodec",
        *,
        target: HFTargetBackend,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(model, tokenizer, device=device, dtype=dtype)
        self.codec = codec
        assert_compatible_pair(target, self)

    @torch.inference_mode()
    def next(
        self, prefix: list[int], codes: tuple[int, ...] | None, position: int
    ) -> int:
        if type(position) is not int or not 1 <= position <= 8:
            raise ValueError("position must be one-based within the eight-token window")
        if codes is not None and (
            not isinstance(codes, tuple)
            or len(codes) != 8
            or any(type(c) is not int or not 0 <= c < 256 for c in codes)
        ):
            raise ValueError("feedback must be exactly eight byte-valued codes")
        with self._lock:
            output, final = self._forward_with_final_hidden(self._ids(prefix))
            if codes is None:
                logits = output.logits[0, -1]
            else:
                code_tensor = torch.tensor([codes], dtype=torch.long, device=final.device)
                repaired = self.codec.repair_logits(final[:, -1], code_tensor, position)
                if repaired.shape != (1, self.vocab_size):
                    raise RuntimeError("feedback repair returned the wrong vocabulary shape")
                logits = repaired[0]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite draft logits")
            return int(logits.argmax(dim=-1).item())

    @torch.no_grad()
    def teacher_hidden(
        self, prefix: list[int], continuation: list[int]
    ) -> torch.Tensor:
        """Return hidden states predicting each continuation token, in order.

        For continuation x[0:L], run the frozen draft on
        ``prefix + x[:L-1]`` and select positions ``len(prefix)-1:...``.
        Position zero is therefore the final prefix token, which predicts x0.
        """
        if not isinstance(continuation, list):
            raise ValueError("continuation must be a list of token IDs")
        if any(type(t) is not int or not 0 <= t < self.vocab_size for t in continuation):
            raise ValueError("continuation contains invalid token IDs")
        self._ids(prefix)
        if not continuation:
            return torch.empty(
                (0, self.hidden_size),
                dtype=self.model.get_input_embeddings().weight.dtype,
                device=self.device,
            )
        teacher_input = prefix + continuation[:-1]
        with self._lock:
            _, final = self._forward_with_final_hidden(
                self._ids(teacher_input), decoder_only=True
            )
            start = len(prefix) - 1
            return final[0, start : start + len(continuation)].detach().clone()
