"""Evaluate one frozen P1 arm on collected packed fixed-prefix validation probes.

The independent Target AR continuation and feature-provenance gate are sealed
in the collector artifact.  This command greedily rolls out a real Draft from
every common corrected prefix.  It reports no packed throughput or WAN speed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig
from feedback_codec.fixed_probe import ProbeExample
from feedback_codec.v2_upper_bound import MatchedUpperBound, UPPER_MODES
from feedback_codec.v2_p1_packed import (
    accepted_prefix, audit_evaluation_row, evaluate_one_window, load_packed_p1_corpus,
    prompt_means, sha256_file,
)
from scripts.train_packed_upper_p1 import (
    FrozenDraftFeatures, load_trainable_state, model_hashes, source_hashes,
    atomic_json,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--val-corpus", type=Path, required=True)
    parser.add_argument("--val-protocol", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--budget-lock", type=Path, required=True)
    parser.add_argument("--cost-lock", type=Path, required=True)
    parser.add_argument("--cost-freeze-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw_draft",) + UPPER_MODES, required=True)
    parser.add_argument("--ablation", choices=("none", "zero", "shuffle_stratified"),
                        default="none", help="F-H only; checkpoint weights stay fixed")
    parser.add_argument("--length-bin-size", type=int, default=128,
                        help="frozen corrected-prefix length bin for F-H shuffle")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-mib", type=int, default=22000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _feature_stratum(example: ProbeExample, length_bin_size: int) -> tuple[str, int, int]:
    o = example.observation
    return o.task_type, o.accepted_count, len(o.corrected_prefix) // length_bin_size


def _ablation_examples(examples: tuple[ProbeExample, ...], *, ablation: str,
                       seed: int, length_bin_size: int
                       ) -> tuple[list[tuple[ProbeExample, dict | None]], dict]:
    """Keep F weights fixed; a donor is always from another request in-stratum."""
    if length_bin_size <= 0:
        raise ValueError("length-bin-size must be positive")
    if ablation == "none":
        return [(example, None) for example in examples], {"windows": len(examples)}
    if ablation == "zero":
        return [(ProbeExample(replace(example.observation,
                                      target_hidden=torch.zeros_like(
                                          example.observation.target_hidden)),
                              example.target_future_ids), None) for example in examples], {
                                  "windows": len(examples)}
    strata: dict[tuple[str, int, int], list[ProbeExample]] = defaultdict(list)
    for example in examples:
        strata[_feature_stratum(example, length_bin_size)].append(example)
    rng = random.Random(seed + 9147)
    chosen: list[tuple[ProbeExample, dict]] = []
    for example in examples:
        candidates = [candidate for candidate in strata[
            _feature_stratum(example, length_bin_size)]
            if candidate.observation.prompt_id != example.observation.prompt_id]
        if not candidates:
            continue
        donor = candidates[rng.randrange(len(candidates))]
        o = example.observation
        donor_info = {"prompt_id": donor.observation.prompt_id,
                      "round_index": donor.observation.round_index,
                      "stratum": list(_feature_stratum(example, length_bin_size))}
        chosen.append((ProbeExample(replace(o, target_hidden=donor.observation.target_hidden),
                                    example.target_future_ids), donor_info))
    if not chosen:
        raise ValueError("no cross-request donors exist in locked F-H strata")
    return chosen, {"windows": len(chosen), "excluded_no_donor": len(examples) - len(chosen),
                    "coverage": len(chosen) / len(examples),
                    "stratum_fields": ["task_type", "accepted_count",
                                       "corrected_prefix_length_bin"],
                    "length_bin_size": length_bin_size,
                    "donor_seed": seed + 9147}


def _train_prompt_ids(path: Path) -> tuple[set[str], set[str]]:
    prompts: set[str] = set()
    families: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("split") != "train":
                    raise ValueError("training corpus contains a cross-split row")
                prompts.add(row["prompt_id"])
                families.add(row["family_id"])
    if not prompts:
        raise ValueError("empty training corpus")
    return prompts, families


@torch.inference_mode()
def _raw_row(draft: FrozenDraftFeatures, example: object,
             eos_token_id: int | None) -> dict:
    o = example.observation
    target_ids = tuple(example.target_future_ids)
    prefix = list(o.corrected_prefix)
    generated: list[int] = []
    for _ in range(min(o.window, o.remaining_budget)):
        output, _ = draft._forward_with_final_hidden(draft._ids(prefix))
        token = int(output.logits[0, -1].argmax(-1).item())
        generated.append(token)
        prefix.append(token)
        if eos_token_id is not None and token == eos_token_id:
            break
    accepted = accepted_prefix(generated, target_ids)
    teacher_hidden = draft.teacher_hidden(list(o.corrected_prefix), list(target_ids))
    teacher_logits = draft.model.get_output_embeddings()(teacher_hidden)
    labels = torch.tensor(target_ids, dtype=torch.long, device=teacher_logits.device)
    teacher_ce = F.cross_entropy(teacher_logits.float(), labels)
    if not torch.isfinite(teacher_ce).item():
        raise FloatingPointError("nonfinite RAW validation CE")
    return {
        "mode": "raw_draft", "prompt_id": o.prompt_id,
        "family_id": o.family_id, "task_type": o.task_type,
        "round_index": o.round_index, "accepted_count": accepted,
        "first_rejection_position": (accepted + 1 if accepted < min(
            len(generated), len(target_ids)) else None),
        "draft_ids": generated, "target_future_ids": list(target_ids),
        "teacher_forced_ce": float(teacher_ce.item()),
        "next_token_top1": int(bool(generated) and generated[0] == target_ids[0]),
        "primary_eligible": len(target_ids) == o.window and (
            eos_token_id is None or eos_token_id not in target_ids),
        "eos_match": ((eos_token_id in generated) == (eos_token_id in target_ids)
                      if eos_token_id is not None else True),
        "truncation_match": len(generated) == len(target_ids),
    }


def main() -> None:
    args = _args()
    if args.ablation != "none" and args.mode != "full_hidden":
        raise ValueError("Target Hidden ablations require the same FULL-H checkpoint")
    if args.output.exists() and not args.resume:
        raise FileExistsError("evaluation output exists; use exact --resume")
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError("resume requires an existing output directory")
    val = load_packed_p1_corpus(args.val_corpus, args.val_protocol, split="val")
    selected_examples, ablation_audit = _ablation_examples(
        val.examples, ablation=args.ablation, seed=args.seed,
        length_bin_size=args.length_bin_size)
    display_mode = (args.mode if args.ablation == "none" else
                    f"full_hidden_{args.ablation}")
    training = json.loads(args.training_protocol.read_text(encoding="utf-8"))
    if (training.get("scope") != "P1_packed_matched_upper_training"
            or training.get("collection_corpus_sha256") != sha256_file(args.train_corpus)
            or training.get("draft_file_sha256") != val.protocol["draft_file_sha256"]
            or training.get("target_file_sha256_from_collection") !=
               val.protocol["target_file_sha256"]
            or training.get("packed_p0_report_sha256") !=
               val.protocol["packed_p0_report_sha256"]
            or training.get("pre_val_budget_lock_sha256") != sha256_file(args.budget_lock)
            or training.get("p0_cost_lock_sha256") != sha256_file(args.cost_lock)
            or training.get("p0_cost_freeze_manifest_sha256") !=
               sha256_file(args.cost_freeze_manifest)
            or args.seed not in training.get("training_seeds", [])
            or training.get("source_sha256") != source_hashes()):
        raise ValueError("val collection differs from frozen training or source")
    if sha256_file(args.train_corpus) == val.corpus_sha256:
        raise ValueError("validation corpus is the training corpus")
    train_prompts, train_families = _train_prompt_ids(args.train_corpus)
    if any(example.observation.prompt_id in train_prompts or
           example.observation.family_id in train_families
           for example in val.examples):
        raise ValueError("val prompt/family overlaps training")
    if model_hashes(args.draft, val.protocol["draft_file_sha256"]) != val.protocol["draft_file_sha256"]:
        raise ValueError("Draft asset SHA differs from val collection")
    if args.mode == "raw_draft" and args.checkpoint is not None or (
            args.mode != "raw_draft" and args.checkpoint is None):
        raise ValueError("adapter arm needs one checkpoint; RAW needs none")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal P1 val requires real CUDA Draft")
    device = torch.device(args.device)
    index = 0 if device.index is None else device.index
    free_mib = torch.cuda.mem_get_info(index)[0] // (1024 * 1024)
    if free_mib < args.minimum_free_mib:
        raise RuntimeError(f"insufficient free GPU memory: {free_mib} MiB")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", torch_dtype=torch.bfloat16).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.draft), local_files_only=True, trust_remote_code=False)
    draft = FrozenDraftFeatures(model, tokenizer, device=device, dtype=torch.bfloat16)
    config = CodecConfig(
        target_hidden_size=val.examples[0].observation.target_hidden.numel(),
        draft_hidden_size=draft.hidden_size, vocab_size=draft.vocab_size, window=8)
    if asdict(config) != training.get("codec_config"):
        raise ValueError("val geometry differs from locked training model")
    adapter = None
    checkpoint_sha = None
    checkpoint_initial_sha = None
    if args.mode != "raw_draft":
        adapter = MatchedUpperBound(config, model.get_output_embeddings(),
                                    mode=args.mode).to(device)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if (payload.get("format") != "packed_upper_p1_trainable_v1"
                or payload.get("mode") != args.mode or payload.get("seed") != args.seed
                or payload.get("steps") != training["steps_per_arm"]
                or payload.get("training_protocol_sha256") !=
                   sha256_file(args.training_protocol)
                or payload.get("source_corpus_sha256") !=
                   training["collection_corpus_sha256"]
                or payload.get("draft_model_file_sha256") !=
                   val.protocol["draft_file_sha256"]
                or payload.get("exposure_order_sha256") !=
                   training["exposure_order_sha256"][str(args.seed)]
                or not isinstance(payload.get("initial_sha256"), str)
                or len(payload["initial_sha256"]) != 64):
            raise ValueError("adapter checkpoint provenance does not match P1 protocol")
        load_trainable_state(adapter, payload["state"])
        adapter.eval()
        checkpoint_sha = sha256_file(args.checkpoint)
        checkpoint_initial_sha = payload["initial_sha256"]
    config = {"scope": "P1_packed_fixed_prefix_eval_run_v2",
              "mode": display_mode, "base_mode": args.mode,
              "ablation": args.ablation,
              "ablation_audit": ablation_audit,
              "seed": args.seed,
              "collection_protocol_sha256": val.protocol_sha256,
              "val_corpus_sha256": val.corpus_sha256,
              "training_protocol_sha256": sha256_file(args.training_protocol),
              "checkpoint_sha256": checkpoint_sha,
              "checkpoint_initial_sha256": checkpoint_initial_sha,
              "evaluator_source_sha256": sha256_file(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    config_path = args.output / "run_config.json"
    if args.resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("evaluation resume configuration changed")
    else:
        atomic_json(config_path, config)
    request_dir = args.output / "requests"
    request_dir.mkdir(exist_ok=args.resume)
    rows: list[dict] = []
    for example, donor in selected_examples:
        key = [example.observation.prompt_id, example.observation.round_index]
        row_path = request_dir / f"{hashlib.sha256(json.dumps(key, ensure_ascii=False).encode('utf-8')).hexdigest()}.json"
        if row_path.exists():
            if not args.resume:
                raise FileExistsError("unexpected previously evaluated request")
            row = json.loads(row_path.read_text(encoding="utf-8"))
            if row.get("donor") != donor:
                raise ValueError("saved donor differs from locked F-H shuffle")
        else:
            row = (_raw_row(draft, example, val.protocol["eos_token_id"])
                   if adapter is None else evaluate_one_window(
                       adapter, draft, example,
                       eos_token_id=val.protocol["eos_token_id"]))
            row["mode"] = display_mode
            row["donor"] = donor
            audit_evaluation_row(row, example,
                                 eos_token_id=val.protocol["eos_token_id"],
                                 expected_mode=display_mode)
            atomic_json(row_path, row)
        audit_evaluation_row(row, example,
                             eos_token_id=val.protocol["eos_token_id"],
                             expected_mode=display_mode)
        rows.append(row)
    rows_file = args.output / "rows.jsonl"
    temporary = args.output / "rows.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")
    temporary.replace(rows_file)
    if len(rows) != len(selected_examples):
        raise RuntimeError("val rows incomplete")
    primary = prompt_means(rows, primary_only=True)
    all_windows = prompt_means(rows, primary_only=False)
    summary = {
        "scope": "P1_packed_fixed_prefix_validation",
        "claim": "acceptance_only_no_online_speed_or_WAN_claim",
        "mode": display_mode, "base_mode": args.mode,
        "ablation": args.ablation, "ablation_audit": ablation_audit,
        "seed": args.seed,
        "windows": len(rows),
        "primary_windows": sum(row["primary_eligible"] for row in rows),
        "excluded_fraction": 1 - sum(row["primary_eligible"] for row in rows) / len(rows),
        "primary_prompt_count": len(primary),
        "primary_mean_acceptance": (sum(primary.values()) / len(primary) if primary else None),
        "all_window_prompt_count": len(all_windows),
        "all_window_mean_acceptance": sum(all_windows.values()) / len(all_windows),
        "primary_mean_teacher_forced_ce": (
            sum(row["teacher_forced_ce"] for row in rows if row["primary_eligible"])
            / sum(row["primary_eligible"] for row in rows)
            if any(row["primary_eligible"] for row in rows) else None),
        "primary_next_token_top1": (
            sum(row["next_token_top1"] for row in rows if row["primary_eligible"])
            / sum(row["primary_eligible"] for row in rows)
            if any(row["primary_eligible"] for row in rows) else None),
        "all_window_mean_teacher_forced_ce": (
            sum(row["teacher_forced_ce"] for row in rows) / len(rows)),
        "all_window_next_token_top1": (
            sum(row["next_token_top1"] for row in rows) / len(rows)),
        "prompt_primary_means": primary,
        "prompt_all_window_means": all_windows,
        "collection_independent_ar_exact": True,
        "online_closed_loop_correctness_not_claimed": True,
        "collection_protocol_sha256": val.protocol_sha256,
        "val_corpus_sha256": val.corpus_sha256,
        "training_protocol_sha256": sha256_file(args.training_protocol),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_initial_sha256": checkpoint_initial_sha,
        "rows_sha256": sha256_file(rows_file),
        "source_sha256": {"scripts/evaluate_packed_upper_p1.py":
                          sha256_file(Path(__file__))},
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({"mode": display_mode, "seed": args.seed,
                      "primary_mean_acceptance": summary["primary_mean_acceptance"],
                      "rows": len(rows)}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
