"""One bounded train-only four-arm P1 wiring batch; no validation or test access."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_codec.codec import CodecConfig, FeedbackCodec
from feedback_codec.fixed_probe import ProbePrompt
from feedback_codec.hf_backend import HFDraftBackend, HFTargetBackend
from feedback_codec.packed_probe_numeric_v3 import (
    PACKED_PROBE_PROTOCOL_VERSION, NUMERICAL_RECHECK_TOP2_ULPS,
    canonical_bytes,
)
from feedback_codec.packed import NUMERICAL_RECHECK_BACKEND_V2
from feedback_codec.v10_four_arm import V10FourArmHead, V10_MODES
from feedback_codec.v10_train_smoke_reader import load_v10_train_smoke
from scripts.build_packed_probe_numeric_v3 import (
    _admit_idle_gpu, _checked_chunk, _prompt_digest,
)


PROGRAM_SHA = "07d1945490c84fb12d99f07b751bbe4f2b14a415fa571afb07fe2f805cc9348a"
PARENT_SHA = "1bfaeeb2f8699933a94a180633b5c361601630cb37409502d670158e3a55fdc2"
V9_P0_SHA = "9cde3357302eae18752b5f76c8d4c87416337d87bc956277f02c4d1fdab77ece"
FIXED_IDS = (
    "v2-train-ultrachat-7096255a802cf44eb7481c0d6dccdf479e4323c0ce44e38c7e217583b56f3328",
    "v2-train-ultrachat-a1761866ad950e95830390cc0d2f84e89d4aae7fc2a53e36338d66a13d25678e",
    "v2-train-dolly-databricks-dolly-15k.jsonl:6400",
    "v2-train-codealpaca-code_alpaca_20k.json:19632",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def atomic_torch(path: Path, payload: object) -> None:
    pending = path.with_name(path.name + ".tmp")
    torch.save(payload, pending)
    pending.replace(path)


def source_and_inputs(args: argparse.Namespace) -> tuple[list[ProbePrompt], dict, dict, dict]:
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_files = source["file_sha256"]
    if source["program_sha256"] != PROGRAM_SHA or digest(args.source_zip) != args.source_zip_sha256:
        raise ValueError("frozen v10 source/program SHA differs")
    for name, expected in source_files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or digest(path) != expected:
            raise ValueError(f"runtime source mismatch: {name}")
    if digest(args.parent) != PARENT_SHA:
        raise ValueError("parent tokenized train/val manifest changed")
    meta = json.loads(args.parent_meta.read_text(encoding="utf-8"))
    if (meta.get("output_sha256") != PARENT_SHA or meta.get("counts") !=
            {"train": 1000, "val": 200} or meta.get("sealed_test_loaded") is not False
            or meta.get("enable_thinking") is not False):
        raise ValueError("parent manifest metadata differs from train/val lock")
    if digest(args.v9_p0_report) != V9_P0_SHA:
        raise ValueError("accepted v9c P0 evidence SHA differs")
    found: dict[str, dict] = {}
    for line in args.parent.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") == "train" and row.get("id") in FIXED_IDS:
            if row["id"] in found:
                raise ValueError("duplicate fixed train prompt")
            found[row["id"]] = row
    if set(found) != set(FIXED_IDS):
        raise ValueError("four fixed prompts absent from parent train")
    stage_files = (args.v9_inputs / "stage1_second_failed_128.jsonl",
                   args.v9_inputs / "stage2_first_failed_128.jsonl",
                   args.v9_inputs / "stage4_fixed_extra2_128.jsonl")
    stage_hashes = ("155935f077a45552e318fc9aab190dc7a8951d34d62978d7095d98e6988dcc0f",
                    "f79611f4f7a47cabf48a6b39dc9a6e0877bad21e179484c2371de8853fe49072",
                    "ced4ee91d160e0cf2db85d7bc12f1e7f16c45a6d20f537a9e70650e5ec13da02")
    stage: dict[str, dict] = {}
    for path, expected in zip(stage_files, stage_hashes, strict=True):
        if digest(path) != expected:
            raise ValueError(f"v9 fixed input SHA differs: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            stage[row["id"]] = row
    if set(stage) != set(FIXED_IDS):
        raise ValueError("v9 fixed input IDs differ")
    prompts = []
    for key in FIXED_IDS:
        row = found[key]
        if (stage[key]["input_ids"] != row["input_ids"]
                or stage[key]["split"] != "train"):
            raise ValueError(f"v9 input differs from parent: {key}")
        prompts.append(ProbePrompt(
            id=key, family_id=row.get("family_id", row.get("prompt_family_sha256")),
            split="train", input_ids=tuple(row["input_ids"]),
            task_type=row.get("task_type") or row.get("source") or "unknown",
        ))
    prior = json.loads(args.prior_plan.read_text(encoding="utf-8"))
    if (prior.get("prompt_manifest_sha256") != PARENT_SHA
            or prior.get("packed_p0_report_sha256") !=
            "6a626705b9b7971aad5dc778888ee438e407dc78bb4ec1b2f5f23de1a13e7701"):
        raise ValueError("historical asset inventory is not the locked parent")
    model_hashes = {"target": prior["target_file_sha256"],
                    "draft": prior["draft_file_sha256"]}
    for side, root in (("target", args.target), ("draft", args.draft)):
        for name, expected in model_hashes[side].items():
            if digest(root / name) != expected:
                raise ValueError(f"{side} asset changed: {name}")
    for key, expected in meta["tokenizer_sha256"].items():
        side, name = key.split("/", 1)
        if name != "chat_template" and digest((args.target if side == "target" else args.draft) / name) != expected:
            raise ValueError(f"tokenizer asset differs: {key}")
    return prompts, meta, model_hashes, source_files


def trainable(model: V10FourArmHead) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        x = tensor.contiguous()
        h.update(name.encode()); h.update(str(tuple(x.shape)).encode())
        h.update(str(x.dtype).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def frozen_sample(model: torch.nn.Module) -> dict[str, object]:
    params = list(model.named_parameters())
    if any(p.requires_grad for _, p in params):
        raise ValueError("base model has trainable parameters")
    sample = {}
    for name, tensor in (params[0], params[len(params)//2], params[-1]):
        x = tensor.detach().flatten()
        part = torch.cat((x[:64], x[-64:])).float().cpu().numpy().tobytes()
        sample[name] = hashlib.sha256(part).hexdigest()
    return {"count": len(params), "numel": sum(p.numel() for _, p in params),
            "sample_sha256": sample}


def train_one(model: V10FourArmHead, draft: HFDraftBackend,
              observation: object, labels: tuple[int, ...], optimizer: torch.optim.Optimizer,
              seed: int) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        hidden = draft.teacher_hidden(list(observation.corrected_prefix), list(labels))
    state = model.prepare_observation(observation, draft)
    torch.manual_seed(seed)
    positions = torch.arange(1, len(labels)+1, device=hidden.device)
    logits = model.forward_logits(hidden.detach().unsqueeze(0), state, positions,
                                  temperature=1.0)
    loss = F.cross_entropy(logits.float().reshape(len(labels), -1),
                           torch.tensor(labels, device=logits.device))
    if not torch.isfinite(loss).item():
        raise FloatingPointError("nonfinite CE")
    loss.backward()
    grads = {name: float(p.grad.detach().float().norm().item())
             for name, p in model.named_parameters()
             if p.requires_grad and p.grad is not None}
    if not grads or any(not math.isfinite(v) for v in grads.values()):
        raise FloatingPointError("missing or nonfinite trainable gradient")
    optimizer.step()
    return {"loss": float(loss.detach().item()), "grad_norms": grads}


@torch.inference_mode()
def fixed_prefix_free(model: V10FourArmHead, draft: HFDraftBackend,
                      observation: object, labels: tuple[int, ...], eos: int) -> dict:
    model.eval()
    state = model.prepare_observation(observation, draft)
    prefix = list(observation.corrected_prefix)
    proposed = []
    for pos in range(1, 9):
        token = model.next_token(prefix, state, pos, draft)
        proposed.append(token)
        prefix.append(token)
        if token == eos:
            break
    accepted = 0
    for a, b in zip(proposed, labels):
        if a != b: break
        accepted += 1
    return {"prompt_id": observation.prompt_id, "round_index": observation.round_index,
            "proposed_ids": proposed, "target_future_ids": list(labels),
            "accepted_prefix": accepted, "eos": proposed[-1] == eos,
            "full_window": len(labels) == 8 and eos not in labels}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("source_manifest", "source_zip", "parent", "parent_meta",
                 "v9_p0_report", "v9_inputs", "prior_plan", "target", "draft", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--source-zip-sha256", required=True)
    parser.add_argument("--deadline-seconds", type=int, default=2700)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("v10 unique worker output already exists")
    args.output.mkdir(parents=True)
    begin = time.monotonic()
    status: dict[str, object] = {"state": "running", "pid": os.getpid(),
                                 "program_sha256": PROGRAM_SHA, "started_unix_s": time.time(),
                                 "stage_seconds": {}, "counts": {}}
    atomic_json(args.output / "status.json", status)
    stage_start = time.monotonic()

    def stage(name: str) -> None:
        nonlocal stage_start
        if torch.cuda.is_initialized(): torch.cuda.synchronize()
        now = time.monotonic()
        old = status.get("phase")
        if old: status["stage_seconds"][old] = now - stage_start
        status["phase"] = name; stage_start = now
        status["elapsed_seconds"] = now - begin
        atomic_json(args.output / "status.json", status)
        if now - begin >= args.deadline_seconds:
            raise TimeoutError("one-batch v10 GPU deadline reached")

    try:
        stage("source_and_input_integrity")
        prompts, meta, model_hashes, source_files = source_and_inputs(args)
        eos = meta["target_eos_token_id"]
        if eos != 151645: raise ValueError("EOS changed")
        plan = {"scope": "v10_four_fixed_train_smoke_plan", "program_sha256": PROGRAM_SHA,
                "source_zip_sha256": args.source_zip_sha256, "source_sha256": source_files,
                "parent_manifest_sha256": PARENT_SHA, "model_file_sha256": model_hashes,
                "tokenizer_sha256": meta["tokenizer_sha256"],
                "packed_p0_report_sha256": V9_P0_SHA,
                "prompt_records": [asdict(p) for p in prompts],
                "prompt_sha256": [_prompt_digest(p) for p in prompts],
                "window": 8, "max_new_tokens": 128,
                "max_probe_windows_per_prompt": 8, "eos_token_id": eos,
                "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
                "numerical_recheck_top2_ulps": 2, "shard_count": 1,
                "split": "train", "full_pilot_complete": False}
        atomic_json(args.output / "collection_plan.json", plan)
        plan_sha = digest(args.output / "collection_plan.json")
        status["gpu_admission"] = _admit_idle_gpu("cuda:0", 22000)
        atomic_json(args.output / "status.json", status)
        stage("load_models")
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        target = HFTargetBackend(args.target, None, device=device, dtype=torch.bfloat16)
        draft_tokenizer = AutoTokenizer.from_pretrained(str(args.draft), local_files_only=True,
                                                        trust_remote_code=False)
        draft_model = AutoModelForCausalLM.from_pretrained(
            str(args.draft), local_files_only=True, trust_remote_code=False,
            attn_implementation="eager", torch_dtype=torch.bfloat16).to(device)
        codec = FeedbackCodec(CodecConfig(target_hidden_size=target.hidden_size,
                                         draft_hidden_size=int(draft_model.config.hidden_size),
                                         vocab_size=target.vocab_size),
                              draft_model.get_output_embeddings()).to(device)
        draft = HFDraftBackend(draft_model, draft_tokenizer, codec, target=target,
                               device=device, dtype=torch.bfloat16)
        if (target.tokenizer.eos_token_id != eos
                or any(p.requires_grad for p in target.model.parameters())
                or any(p.requires_grad for p in draft_model.parameters())):
            raise ValueError("model/EOS/freeze contract failed")
        base_before = {"target": frozen_sample(target.model),
                       "draft": frozen_sample(draft_model)}
        atomic_json(args.output / "base_before.json", base_before)
        stage("collect_four_train_only")
        from feedback_codec.v10_collect_one import collect_one
        chunks_dir = args.output / "chunks"; chunks_dir.mkdir()
        aggregate = {"prompts": 0, "probe_windows": 0, "exact_requests": 0}
        chunk_paths = {}; chunk_sha = {}; prompt_input_sha = {}
        all_rows = []
        for prompt in prompts:
            if time.monotonic() - begin >= args.deadline_seconds:
                raise TimeoutError("collection reached v10 deadline")
            corpus, stream = collect_one(target, draft, prompt, eos_token_id=eos,
                                         max_new_tokens=128)
            atomic_json(args.output / (hashlib.sha256(prompt.id.encode()).hexdigest()
                                       + ".streams.json"), stream)
            if stream.get("exact") is not True:
                raise RuntimeError("packed SD differs from independent Target AR")
            chunk = {"prompt_id": prompt.id, "prompt_sha256": _prompt_digest(prompt),
                     "plan_sha256": plan_sha, "shard_index": 0, "shard_count": 1,
                     "eos_token_id": eos, "window": 8,
                     "max_probe_windows_per_prompt": 8, "capture_target_logits": True,
                     "feature_provenance_gate": True,
                     "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
                     "numerical_recheck_top2_ulps": NUMERICAL_RECHECK_TOP2_ULPS,
                     "gpu_name": torch.cuda.get_device_name(device),
                     "audit": corpus.audit, "request_audit": corpus.request_audit[0],
                     "examples": [row.to_json() for row in corpus.rows]}
            chunk["checksum_sha256"] = hashlib.sha256(canonical_bytes(chunk)).hexdigest()
            name = "chunks/" + hashlib.sha256(prompt.id.encode()).hexdigest() + ".json"
            path = args.output / name
            atomic_json(path, chunk)
            checked = _checked_chunk(path, prompt, plan_sha, 1, eos)
            chunk_paths[prompt.id] = name; chunk_sha[prompt.id] = digest(path)
            prompt_input_sha[prompt.id] = hashlib.sha256(canonical_bytes(list(prompt.input_ids))).hexdigest()
            all_rows.extend(checked["examples"])
            for key in aggregate:
                aggregate[key] += corpus.audit[key]
            status["counts"] = aggregate.copy()
            atomic_json(args.output / "status.json", status)
        if aggregate["probe_windows"] > 32 or aggregate["exact_requests"] != 4:
            raise ValueError("v10 collection count exceeds cap or misses request")
        corpus_file = args.output / "fixed_prefix.jsonl"
        pending = corpus_file.with_suffix(".jsonl.tmp")
        with pending.open("w", encoding="utf-8", newline="\n") as stream:
            for row in all_rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                        allow_nan=False) + "\n")
        pending.replace(corpus_file)
        wrapper = {"scope": "v10_train_smoke", "schema_version": "v10_train_smoke_v1",
                   "train_smoke": True, "split": "train", "full_pilot_complete": False,
                   "collection_protocol_version": PACKED_PROBE_PROTOCOL_VERSION,
                   "numeric_decision_backend": NUMERICAL_RECHECK_BACKEND_V2,
                   "numerical_recheck_top2_ulps": 2, "window": 8,
                   "max_new_tokens": 128, "max_probe_windows_per_prompt": 8,
                   "eos_token_id": eos, "parent_manifest_sha256": PARENT_SHA,
                   "collection_plan_sha256": plan_sha,
                   "source_zip_sha256": args.source_zip_sha256,
                   "source_sha256": source_files,
                   "model_file_sha256": model_hashes,
                   "tokenizer_sha256": meta["tokenizer_sha256"],
                   "packed_p0_report_sha256": V9_P0_SHA,
                   "fixed_prefix_sha256": digest(corpus_file),
                   "prompt_records": [asdict(p) for p in prompts],
                   "prompt_input_sha256": prompt_input_sha,
                   "chunk_paths": chunk_paths, "chunk_sha256": chunk_sha,
                   "shard_count": 1, "audit": aggregate}
        atomic_json(args.output / "collection_protocol.json", wrapper)
        stage("v3_reader_and_tiny_lock")
        loaded = load_v10_train_smoke(
            corpus_file, args.output / "collection_protocol.json",
            expected_parent_sha=PARENT_SHA, expected_plan_sha=plan_sha,
            expected_source_sha=args.source_zip_sha256,
            expected_model_hashes=model_hashes,
            expected_tokenizer_sha=meta["tokenizer_sha256"],
            expected_p0_sha=V9_P0_SHA)
        windows = list(loaded.training_windows())
        if len(windows) != 8 or any(len(w.labels) != 8 for w in windows):
            raise RuntimeError("INCOMPLETE_DATA: eight full L8 train windows absent")
        order = list(range(8)); random.Random(42).shuffle(order)
        config = CodecConfig(target_hidden_size=target.hidden_size,
                             draft_hidden_size=draft.hidden_size,
                             vocab_size=draft.vocab_size, window=8)
        tiny_lock = {"scope": "v10_train_smoke_only", "program_sha256": PROGRAM_SHA,
                     "source_zip_sha256": args.source_zip_sha256,
                     "plan_sha256": plan_sha,
                     "corpus_sha256": digest(corpus_file),
                     "protocol_sha256": digest(args.output / "collection_protocol.json"),
                     "seed": 42, "modes": list(V10_MODES),
                     "steps_each": 8, "window_order": order,
                     "learning_rate": 0.0001, "weight_decay": 0.01,
                     "temperature": 1.0, "codec_config": asdict(config),
                     "validation_or_test": 0}
        atomic_json(args.output / "tiny_training_lock.json", tiny_lock)
        stage("four_arm_training_reload_free_prefix_closed_loop")
        training = {}; fixed = {}; closed = {}; initial_sha = None
        from feedback_codec.v10_closed_loop import run_v10_closed_loop
        for mode in V10_MODES:
            stage("train_" + mode)
            torch.manual_seed(42)
            head = V10FourArmHead(config, draft_model.get_output_embeddings(),
                                  mode=mode,
                                  draft_embedding=draft_model.get_input_embeddings()).to(device)
            if any(p.requires_grad for p in head.core.repair.output_head.parameters()):
                raise ValueError("Draft lm_head became trainable")
            state0 = trainable(head)
            s0 = state_digest(state0)
            if initial_sha is None: initial_sha = s0
            if s0 != initial_sha:
                raise ValueError("four arms did not share initial weights")
            params = {n: p for n, p in head.named_parameters() if p.requires_grad}
            optimizer = torch.optim.AdamW(params.values(), lr=0.0001, weight_decay=0.01)
            history = []
            for i, index in enumerate(order):
                row = train_one(head, draft, windows[index].observation,
                                windows[index].labels, optimizer, 42+i+1)
                history.append({"step": i+1, "window_index": index,
                                "prompt_id": windows[index].observation.prompt_id,
                                "round_index": windows[index].observation.round_index,
                                **row})
                atomic_json(args.output / (mode + ".train_progress.json"),
                            {"mode": mode, "completed_steps": i+1, "history": history})
                if time.monotonic() - begin >= args.deadline_seconds:
                    raise TimeoutError("v10 deadline during training")
            checkpoint = args.output / (mode + ".trainable.pt")
            trained_state = trainable(head)
            atomic_torch(checkpoint, {"mode": mode, "state": trained_state,
                                      "tiny_lock_sha256": digest(args.output / "tiny_training_lock.json"),
                                      "initial_sha256": s0, "steps": 8})
            stage("reload_" + mode)
            eval_state = head.prepare_observation(windows[0].observation, draft)
            with torch.inference_mode():
                test_hidden = draft.teacher_hidden(
                    list(windows[0].observation.corrected_prefix),
                    list(windows[0].labels)).unsqueeze(0)
                before = head.forward_logits(test_hidden, eval_state,
                                             torch.arange(1, 9, device=device)).detach().float()
            del optimizer, head
            torch.cuda.empty_cache()
            torch.manual_seed(42)
            reloaded = V10FourArmHead(config, draft_model.get_output_embeddings(),
                                      mode=mode,
                                      draft_embedding=draft_model.get_input_embeddings()).to(device)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["tiny_lock_sha256"] != digest(args.output / "tiny_training_lock.json"):
                raise ValueError("checkpoint tiny lock changed")
            reloaded.load_trainable_state_dict(payload["state"])
            with torch.inference_mode():
                after = reloaded.forward_logits(
                    test_hidden, reloaded.prepare_observation(windows[0].observation, draft),
                    torch.arange(1, 9, device=device)).detach().float()
            if not torch.equal(before, after):
                raise RuntimeError("reload logits differ")
            reloaded.eval()
            training[mode] = {"initial_sha256": s0,
                              "trained_state_sha256": state_digest(trained_state),
                              "checkpoint_sha256": digest(checkpoint),
                              "trainable_parameter_names": sorted(params),
                              "steps": history, "reload_exact": True,
                              "parameter_budget": reloaded.parameter_budget()}
            atomic_json(args.output / (mode + ".training_receipt.json"), training[mode])
            stage("fixed_prefix_" + mode)
            all_prompt_windows = {}
            for row in loaded.windows:
                if len(row.labels) == 8 and eos not in row.labels:
                    all_prompt_windows.setdefault(row.observation.prompt_id, []).append(row)
            chosen = []
            for p in prompts:
                chosen.extend(all_prompt_windows.get(p.id, [])[:4])
            if len(chosen) > 16:
                raise RuntimeError("fixed-prefix evaluation exceeds v10 cap")
            fixed[mode] = [fixed_prefix_free(reloaded, draft, w.observation, w.labels, eos)
                           for w in chosen]
            atomic_json(args.output / (mode + ".fixed_prefix.json"), fixed[mode])
            stage("closed_loop_" + mode)
            closed[mode] = []
            for prompt in prompts[:2]:
                progress_path = args.output / (mode + "." +
                    hashlib.sha256(prompt.id.encode()).hexdigest() + ".closed_progress.json")
                def progress_hook(partial: dict) -> None:
                    atomic_json(progress_path, partial)
                run = run_v10_closed_loop(
                    target=target, draft=draft, head=reloaded, prompt=prompt,
                    max_new_tokens=32, eos_token_id=eos,
                    progress_hook=progress_hook).to_json()
                closed[mode].append(run)
                atomic_json(args.output / (mode + ".closed_loop.json"), closed[mode])
                if run.get("exact") is not True:
                    raise RuntimeError("trained high-bandwidth closed loop differs from independent AR")
                if time.monotonic() - begin >= args.deadline_seconds:
                    raise TimeoutError("v10 deadline during closed loop")
            del reloaded, trained_state, payload, params, state0, before, after, test_hidden
            torch.cuda.empty_cache()
        stage("postrun_integrity")
        base_after = {"target": frozen_sample(target.model),
                      "draft": frozen_sample(draft_model)}
        if base_after != base_before:
            raise RuntimeError("frozen Target or Draft parameter sample changed")
        atomic_json(args.output / "base_after.json", base_after)
        atomic_json(args.output / "final_receipt.json", {
            "decision": "P1_WIRING_PASS", "program_sha256": PROGRAM_SHA,
            "source_zip_sha256": args.source_zip_sha256,
            "collection_plan_sha256": plan_sha,
            "corpus_sha256": digest(corpus_file),
            "collection_protocol_sha256": digest(args.output / "collection_protocol.json"),
            "training_lock_sha256": digest(args.output / "tiny_training_lock.json"),
            "collection_audit": aggregate, "training": training,
            "fixed_prefix_counts": {m: len(fixed[m]) for m in V10_MODES},
            "closed_loop_counts": {m: len(closed[m]) for m in V10_MODES},
            "base_before_after_same": True,
            "elapsed_seconds": time.monotonic() - begin,
            "stage_seconds": status["stage_seconds"]})
        status["state"] = "completed"
    except Exception as exc:
        status["state"] = "failed"
        status["exception_type"] = type(exc).__name__
        status["exception"] = str(exc)
        status["traceback"] = traceback.format_exc()
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - begin
        status["ended_unix_s"] = time.time()
        atomic_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
