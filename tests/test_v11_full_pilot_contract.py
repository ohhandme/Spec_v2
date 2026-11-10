"""Focused CPU gates for the frozen full-pilot extension."""

from pathlib import Path
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from feedback_codec.codec import CodecConfig
from feedback_codec.decoding import TargetLogitSummary
from feedback_codec.fixed_probe import ProbeObservation
from feedback_codec.v10_four_arm import build_matched_v10_arms
from feedback_codec.v11_full_contract import (
    PARENT_SHA256, causal_window_id, checked_v10_import,
    locked_prompts, window_id,
)
from feedback_codec.v11_stats import paired_family_ci, prefix_acceptance
from feedback_codec.v11_teacher_diagnostic import run_teacher_diagnostic


REMOTE_PROJECT = os.environ.get("V11_TEST_PROJECT_ROOT")
if REMOTE_PROJECT:
    ROOT = Path(REMOTE_PROJECT)
    ROUND = ROOT / "runs/v10-p1-wiring-20261102/worker"
    PARENT = ROOT / "runs/p0-data-lock-20261101/pilot_tokenized.jsonl"
else:
    ROOT = Path(__file__).resolve().parents[3]
    ROUND = ROOT / "autoresearch/feedback64-v2-261031/rounds/v10-p1-wiring-20261102/evidence/worker"
    PARENT = ROOT / "autoresearch/feedback64-v2-261031/runs/p0-data-lock-20261101/pilot_tokenized.jsonl"


def test_locked_full_membership_and_immutable_v10_reference():
    import json

    prompts = locked_prompts(PARENT, PARENT.with_name("pilot_tokenized.jsonl.meta.json"))
    assert (len(prompts["train"]), len(prompts["val"])) == (1000, 200)
    protocol = json.loads((ROUND / "collection_protocol.json").read_text())
    windows, receipt = checked_v10_import(
        ROUND, parent_prompts=prompts["train"],
        source_zip_sha256=protocol["source_zip_sha256"],
        model_hashes=protocol["model_file_sha256"],
        tokenizer_sha256=protocol["tokenizer_sha256"],
        p0_sha256=protocol["packed_p0_report_sha256"],
        eos_token_id=protocol["eos_token_id"],
    )
    assert len(windows) == 32
    assert len(receipt["prompt_ids"]) == 4
    assert all(w.imported_v10 for w in windows)
    assert PARENT_SHA256 == protocol["parent_manifest_sha256"]
    first = windows[0]
    # An intervention ID must not change when only the held-out label changes.
    from dataclasses import replace
    assert causal_window_id(first) == causal_window_id(replace(
        first, labels=(first.labels[0] + 1,) + first.labels[1:]))
    assert window_id(first) != window_id(replace(
        first, labels=(first.labels[0] + 1,) + first.labels[1:]))


def _synthetic_diagnostic():
    embedding = nn.Embedding(64, 8)
    output = nn.Linear(8, 64, bias=False)
    embedding.requires_grad_(False)
    output.requires_grad_(False)
    draft = SimpleNamespace(
        model=SimpleNamespace(parameters=lambda: iter(()),
                              get_input_embeddings=lambda: embedding),
        teacher_hidden=lambda prefix, continuation: torch.ones(len(continuation), 8),
    )
    def summary(token):
        ids = (token,) + tuple(i for i in range(64) if i != token)[:31]
        return TargetLogitSummary(token, ids, (5.0,) + tuple(4.0 - i * .1 for i in range(31)),
                                  6.0, 1.0, 64)
    windows = []
    for i in range(8):
        token = i + 2
        obs = ProbeObservation(
            f"p{i}", f"f{i}", "train", "dolly" if i == 7 else "ultrachat", 0,
            (7,), tuple(range(8)), (7,), 0, 0, token, (7, token),
            torch.randn(16), (7,), 127, 8, summary(token),
        )
        windows.append(SimpleNamespace(observation=obs, labels=(3,)))
    heads = build_matched_v10_arms(
        CodecConfig(16, 8, 64, feature_dim=16, hidden_width=16),
        output, seed=42, draft_embedding=embedding)
    return heads, windows, draft


def test_teacher_leaf_and_local_control_are_distinct():
    heads, windows, draft = _synthetic_diagnostic()
    receipt = run_teacher_diagnostic(heads=heads, windows=windows, draft=draft)
    assert receipt["status"] == "passed"
    assert all(receipt["arms"][mode]["passed"] for mode in heads)
    assert receipt["arms"]["F-H"]["wrong_donor_NA_count"] == 1
    for mode in ("B", "C"):
        assert all(row["deltas"]["wrong"]["control_max_abs"] == 0
                   for row in receipt["arms"][mode]["windows"]
                   if "wrong" in row["deltas"])
    for mode in ("F-H", "F-CURRENT-TOP32"):
        assert any(row["teacher_feature_leaf_ce_gradient_norm"] > 0
                   for row in receipt["arms"][mode]["windows"])


def test_prompt_family_paired_ci_and_prefix_stop():
    scores = {}
    for source in ("ultrachat", "dolly"):
        for family in ("f1", "f2"):
            for prompt in ("a", "b"):
                for arm, value in (("F-H", 2.0), ("C", 1.0)):
                    scores[(42, source, family, prompt, arm)] = value
    result = paired_family_ci(scores, seed=42, left="F-H", right="C", draws=50)
    assert result["point"] == pytest.approx(1.0)
    assert result["ci95"] == [1.0, 1.0]
    assert result["prompt_count"] == 8
    assert prefix_acceptance([7, 8, 9], [7, 5, 9]) == 1


def test_val_statistics_uses_prompt_pairs_across_three_seeds(tmp_path, monkeypatch):
    import json
    from scripts import run_v11_full_pilot as driver

    # Exercise actual score aggregation while reducing only this CPU test's
    # bootstrap iteration count.  The frozen production constant remains 10k.
    original = driver.paired_family_ci
    monkeypatch.setattr(driver, "paired_family_ci", lambda *a, **k: original(
        *a, **{**k, "draws": 50}))
    prompts = [SimpleNamespace(id="p1", task_type="ultrachat"),
               SimpleNamespace(id="p2", task_type="dolly")]
    val_windows = []
    for prompt in prompts:
        observation = SimpleNamespace(prompt_id=prompt.id,
                                      task_type=prompt.task_type)
        val_windows.append(SimpleNamespace(observation=observation,
                                           labels=tuple(range(8))))
    for seed in (42, 43, 44):
        directory = tmp_path / "val_scores" / f"seed{seed}"
        directory.mkdir(parents=True)
        for prompt in prompts:
            rows = []
            for arm, acceptance in (("raw_draft", 1), ("B", 1), ("C", 1),
                                    ("F-H", 2), ("F-CURRENT-TOP32", 2),
                                    ("F-H-zero", 1),
                                    ("F-H-stratified_shuffle", 1)):
                rows.append({
                    "seed": seed, "source": prompt.task_type,
                    "family_id": prompt.id, "prompt_id": prompt.id,
                    "arm": arm, "window_id": prompt.id,
                    "full_window_eligible": True,
                    "acceptance": acceptance, "future_length": 8,
                    "first_rejection_position": acceptance,
                    "next_token_top1_exact": True,
                    "teacher_forced_CE_secondary": 1.0,
                })
            (directory / driver._chunk_name(prompt)).write_text(json.dumps({
                "seed": seed, "prompt_id": prompt.id, "rows": rows}), encoding="utf-8")
    result = driver._statistics(tmp_path, prompts, val_windows, eos=99)
    assert result["prompts_total"] == 2
    assert result["paired_seed_mean_intervals"]["F-H-C"]["point"] == 1.0
    assert result["teacher_correct_minus_stratified_shuffle_on_matched"]["42"]["point"] == 1.0
