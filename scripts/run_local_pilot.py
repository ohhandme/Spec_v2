"""Run the bounded local correctness/acceptance pilot on pretokenized prompts.

Example:
  python scripts/run_local_pilot.py --target /models/Qwen3-8B \
    --draft /models/Qwen3-0.6B --dataset prompts.jsonl --output runs/pilot-1

Model directories must already exist locally. No model is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, FeedbackCodec
from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
from feedback_codec.pipeline import PilotConfig, load_prompt_jsonl, run_reference_pilot


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_file_hashes(root: Path) -> dict[str, str]:
    """Pin the actual local model files, including all weight shards."""
    tracked_names = {
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "model.safetensors.index.json",
    }
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and (path.name in tracked_names or path.suffix in (".safetensors", ".bin"))
    )
    if not any(path.suffix in (".safetensors", ".bin") for path in files):
        raise ValueError(f"model directory has no local weight files: {root}")
    return {str(path.relative_to(root)): _sha256_file(path) for path in files}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="local frozen target model")
    parser.add_argument("--draft", type=Path, required=True, help="local frozen draft model")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL id/split/input_ids")
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--refresh-steps", type=int, default=100)
    parser.add_argument("--train-max-new", type=int, default=32)
    parser.add_argument("--test-max-new", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prefix-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for path in (args.target, args.draft, args.dataset):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    torch.manual_seed(args.seed)

    # Each model loads its own tokenizer so a mismatch is detected rather than
    # hidden by supplying the target tokenizer to both backends.
    target = HFTargetBackend(args.target, None, device=device, dtype=dtype)
    draft_tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        str(args.draft),
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
        torch_dtype=dtype,
    ).to(device)
    codec = FeedbackCodec(
        CodecConfig(
            target_hidden_size=target.hidden_size,
            draft_hidden_size=int(draft_model.config.hidden_size),
            vocab_size=target.vocab_size,
        ),
        draft_model.get_output_embeddings(),
    ).to(device)
    draft = HFDraftBackend(
        draft_model, draft_tokenizer, codec, target=target, device=device, dtype=dtype
    )
    splits = load_prompt_jsonl(args.dataset, vocab_size=target.vocab_size)
    config = PilotConfig(
        warmup_steps=args.warmup_steps,
        refresh_steps=args.refresh_steps,
        train_max_new_tokens=args.train_max_new,
        test_max_new_tokens=args.test_max_new,
        learning_rate=args.lr,
        prefix_weight=args.prefix_weight,
        gumbel_temperature=args.temperature,
        seed=args.seed,
    )
    source_files = sorted((ROOT / "feedback_codec").glob("*.py")) + [Path(__file__)]
    metadata = {
        "target_path": str(args.target.resolve()),
        "draft_path": str(args.draft.resolve()),
        "target_file_sha256": _model_file_hashes(args.target),
        "draft_file_sha256": _model_file_hashes(args.draft),
        "target_model_type": target.model.config.model_type,
        "draft_model_type": draft.model.config.model_type,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }
    report = run_reference_pilot(
        target=target,
        draft=draft,
        codec=codec,
        train_prompts=splits["train"],
        test_prompts=splits["test"],
        output_dir=args.output,
        config=config,
        eos_token_id=target.tokenizer.eos_token_id,
        input_file=args.dataset,
        model_metadata=metadata,
    )
    print(json.dumps(report["paired_learned_minus_no_feedback"], ensure_ascii=False))


if __name__ == "__main__":
    main()
