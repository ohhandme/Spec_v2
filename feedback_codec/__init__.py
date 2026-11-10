"""Reference implementation of eight-byte verification-feedback coding."""

from .codec import CodecConfig, FeedbackCodec, pack_codes, unpack_codes
from .decoding import SpeculativeDecoder, autoregressive_generate
from .evaluation import Prompt, evaluate_paired
from .pipeline import PilotConfig, load_prompt_jsonl, run_reference_pilot

__all__ = [
    "CodecConfig",
    "FeedbackCodec",
    "pack_codes",
    "unpack_codes",
    "SpeculativeDecoder",
    "autoregressive_generate",
    "Prompt",
    "evaluate_paired",
    "PilotConfig",
    "load_prompt_jsonl",
    "run_reference_pilot",
]
