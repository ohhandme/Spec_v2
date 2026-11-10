"""Bounded three-stage local reference pilot for feedback-coded drafting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch

from .decoding import SpeculativeDecoder
from .evaluation import Prompt, evaluate_paired
from .training import CodecEncoder, TrajectoryExample, collect_trajectories, train_examples


@dataclass(frozen=True)
class PilotConfig:
    warmup_steps: int = 100
    refresh_steps: int = 100
    train_max_new_tokens: int = 32
    test_max_new_tokens: int = 128
    learning_rate: float = 1e-4
    prefix_weight: float = 1.0
    gumbel_temperature: float = 1.0
    seed: int = 42
    bootstrap_repetitions: int = 2000

    def __post_init__(self) -> None:
        for field in ("warmup_steps", "refresh_steps", "train_max_new_tokens", "test_max_new_tokens"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} cannot be negative")
        if self.learning_rate <= 0 or self.gumbel_temperature <= 0 or self.prefix_weight < 0:
            raise ValueError("invalid optimizer or objective configuration")
        if self.bootstrap_repetitions < 100:
            raise ValueError("bootstrap_repetitions must be at least 100")


def load_prompt_jsonl(path: str | Path, *, vocab_size: int) -> dict[str, list[Prompt]]:
    """Read locked token-ID prompts with train/test isolation.

    Each JSONL row contains ``id``, ``split`` (train or test), and
    ``input_ids``. Text tokenization is intentionally external so chat-template
    differences cannot silently change a recorded experiment.
    """
    splits: dict[str, list[Prompt]] = {"train": [], "test": []}
    ids: set[str] = set()
    sequences: set[tuple[int, ...]] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            split, identifier, tokens = row.get("split"), row.get("id"), row.get("input_ids")
            if split not in splits:
                raise ValueError(f"line {line_number}: split must be train or test")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError(f"line {line_number}: invalid or duplicate id")
            if not isinstance(tokens, list) or not tokens or any(
                type(token) is not int or token < 0 or token >= vocab_size for token in tokens
            ):
                raise ValueError(f"line {line_number}: invalid input_ids")
            token_tuple = tuple(tokens)
            if token_tuple in sequences:
                raise ValueError(f"line {line_number}: duplicate prompt token sequence")
            ids.add(identifier)
            sequences.add(token_tuple)
            splits[split].append(Prompt(identifier, token_tuple))
    if not splits["train"] or not splits["test"]:
        raise ValueError("both train and test splits need at least one prompt")
    return splits


def _atomic_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _phase_rows(examples: Iterable[TrajectoryExample]) -> list[dict]:
    return [example.to_json() for example in examples]


def run_reference_pilot(
    *,
    target,
    draft,
    codec,
    train_prompts: list[Prompt],
    test_prompts: list[Prompt],
    output_dir: str | Path,
    config: PilotConfig,
    eos_token_id: int | None,
    input_file: str | Path | None = None,
    model_metadata: dict | None = None,
) -> dict:
    """Collect baseline, warm up, refresh under the codec, then evaluate.

    The function refuses a preexisting output directory to prevent replacing
    earlier evidence. Checkpoints are written after each completed stage.
    """
    if not train_prompts or not test_prompts:
        raise ValueError("train and test prompts are required")
    train_ids = {p.id for p in train_prompts}
    test_ids = {p.id for p in test_prompts}
    if len(train_ids) != len(train_prompts) or len(test_ids) != len(test_prompts):
        raise ValueError("duplicate prompt IDs")
    if train_ids & test_ids or {
        p.input_ids for p in train_prompts
    } & {p.input_ids for p in test_prompts}:
        raise ValueError("train/test prompt leakage")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    input_digest = (
        hashlib.sha256(Path(input_file).read_bytes()).hexdigest() if input_file else None
    )
    protocol = {
        "scope": "local_full_prefix_reference_acceptance_only",
        "latency_claim_valid": False,
        "config": asdict(config),
        "codec": asdict(codec.config),
        "input_sha256": input_digest,
        "train_ids": [p.id for p in train_prompts],
        "test_ids": [p.id for p in test_prompts],
        "eos_token_id": eos_token_id,
        "frozen_target": True,
        "frozen_draft": True,
        "model_and_source": model_metadata,
    }
    _atomic_json(destination / "protocol.json", protocol)
    optimizer = torch.optim.AdamW(
        (p for p in codec.parameters() if p.requires_grad), lr=config.learning_rate
    )
    baseline_decoder = SpeculativeDecoder(
        target, draft, None, codec.config.window, eos_token_id
    )
    baseline_examples = collect_trajectories(
        baseline_decoder,
        [(p.id, list(p.input_ids)) for p in train_prompts],
        max_new_tokens=config.train_max_new_tokens,
        phase="baseline",
    )
    _atomic_json(destination / "baseline_trajectories.json", _phase_rows(baseline_examples))
    warmup_history = train_examples(
        examples=baseline_examples,
        target=target,
        draft=draft,
        codec=codec,
        optimizer=optimizer,
        steps=config.warmup_steps,
        prefix_weight=0.0,
        temperature=config.gumbel_temperature,
        seed=config.seed,
    )
    _atomic_json(destination / "warmup_history.json", warmup_history)
    torch.save(codec.state_dict(), destination / "after_warmup.pt")
    current_decoder = SpeculativeDecoder(
        target, draft, CodecEncoder(codec), codec.config.window, eos_token_id
    )
    refreshed_examples = collect_trajectories(
        current_decoder,
        [(p.id, list(p.input_ids)) for p in train_prompts],
        max_new_tokens=config.train_max_new_tokens,
        phase="refreshed",
    )
    _atomic_json(destination / "refreshed_trajectories.json", _phase_rows(refreshed_examples))
    refresh_history = train_examples(
        examples=refreshed_examples,
        target=target,
        draft=draft,
        codec=codec,
        optimizer=optimizer,
        steps=config.refresh_steps,
        prefix_weight=config.prefix_weight,
        temperature=config.gumbel_temperature,
        seed=config.seed + 1,
    )
    _atomic_json(destination / "refresh_history.json", refresh_history)
    torch.save(codec.state_dict(), destination / "final_codec.pt")
    codec.eval()
    comparison = evaluate_paired(
        target=target,
        draft=draft,
        codec=codec,
        prompts=test_prompts,
        window=codec.config.window,
        max_new_tokens=config.test_max_new_tokens,
        eos_token_id=eos_token_id,
        shuffled_control=len(test_prompts) >= 2,
        bootstrap_repetitions=config.bootstrap_repetitions,
        seed=config.seed,
    )
    _atomic_json(destination / "comparison.json", comparison)
    return comparison
