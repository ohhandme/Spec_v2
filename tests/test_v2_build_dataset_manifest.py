"""CPU-only correctness tests for the frozen v2 prompt-sampling contract."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v2_build_dataset_manifest as builder  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


class TestManifestBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompts = {
            "ultrachat": [
                "What causes northern lights to appear over polar skies?",
                "Describe techniques for cultivating bonsai trees indoors.",
                "Explain why ancient ceramics survive in dry deserts.",
                "How do ocean tides influence coastal wetlands?",
                "What factors shape the migration of arctic seabirds?",
            ],
            "dolly": [
                "Outline a weekly strength routine for new athletes.",
                "Compare several methods for brewing loose leaf tea.",
                "Summarize safety checks for mountain hiking.",
                "How should a museum archive fragile photographs?",
            ],
            "codealpaca": [
                "Write Python to parse nested JSON objects.",
                "Create a SQL query for monthly revenue reporting.",
                "Implement a Rust iterator over binary records.",
                "Debug a JavaScript event listener memory leak.",
            ],
            "gsm8k": [
                "Mira packs seven apples in each blue box. How many apples are in four blue boxes?",
                "A baker divides eighty rolls among five trays. How many rolls are on each tray?",
                "Nora saves six coins every Tuesday. How many coins after nine Tuesdays?",
                "A shop sells twelve pencils each morning. How many pencils after eight mornings?",
            ],
        }
        self.plans = tuple(dataclasses.replace(
            plan, expected_rows=len(self.prompts[plan.key]),
            train_quota=1, val_quota=1,
            file_format="json_array" if plan.key == "codealpaca" else "jsonl",
            filenames=(f"{plan.key}.json" if plan.key == "codealpaca"
                       else f"{plan.key}.jsonl",),
        ) for plan in builder.SOURCE_PLANS)
        self.source_lock = self.root / "source_lock.json"
        self.active_path = self.root / "old_train_val.jsonl"
        self.selection_manifest = self.root / "old_selection_manifest.json"
        self.holdout_lock = self.root / "holdout_lock.json"
        self._write_source_lock()
        self._write_old_active_metadata()

    def _write_source_lock(self) -> None:
        sources = {}
        for plan in self.plans:
            path = self.root / plan.filenames[0]
            texts = self.prompts[plan.key]
            if plan.key == "ultrachat":
                rows = [{"prompt_id": f"uc-{i}", "prompt": text,
                         "messages": [{"role": "assistant", "content": "UNUSED_ASSISTANT"}]}
                        for i, text in enumerate(texts)]
            elif plan.key == "dolly":
                rows = [{"instruction": text, "context": "", "response": "UNUSED_RESPONSE",
                         "category": "general"} for text in texts]
            elif plan.key == "codealpaca":
                rows = [{"instruction": text, "input": "", "output": "UNUSED_OUTPUT"}
                        for text in texts]
            else:
                rows = [{"question": text, "answer": "UNUSED_ANSWER"} for text in texts]
            if plan.file_format == "json_array":
                write_json(path, rows)
            else:
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                        for row in rows), encoding="utf-8")
            sources[plan.key] = {
                "repo": plan.repo, "revision": plan.revision, "config": plan.config,
                "split": plan.split, "license": plan.license,
                "upstream_data_license": plan.upstream_data_license,
                "format": plan.file_format, "expected_rows": plan.expected_rows,
                "files": [{"path": path.name, "sha256": builder.sha256_file(path)}],
            }
        write_json(self.source_lock, {"schema": builder.SOURCE_LOCK_SCHEMA,
                                      "sources": sources})

    def _write_old_active_metadata(self, *, blocked_extra: list[str] | None = None,
                                   altered_val_family: bool = False) -> None:
        old_train = self.prompts["ultrachat"][0]
        old_val = self.prompts["ultrachat"][1]
        active = [
            {"split": "train", "source_id": "uc-0",
             "prompt_family_sha256": builder.family_hash(old_train)},
            {"split": "val", "source_id": "uc-1",
             "prompt_family_sha256": builder.family_hash(old_val)},
        ]
        self.active_path.write_text("".join(json.dumps(row) + "\n" for row in active),
                                    encoding="utf-8")
        retired_test_family = builder.family_hash("Synthetic retired test request")
        sealed_hash = sha(b"synthetic sealed file; builder never reads it")
        holdout = {
            "schema": "ultrachat_six_topic_holdout_lock_v2",
            "near_duplicate_threshold": 0.85,
            "new_test_manifest_sha256": sealed_hash,
            "new_test_source_ids": ["retired-uc"],
            "new_test_family_sha256": [retired_test_family],
            "new_val_source_ids": ["uc-1"],
            "new_val_family_sha256": [
                "0" * 64 if altered_val_family else builder.family_hash(old_val)],
            "blocked_train_source_ids_near_duplicate": [
                "uc-1", "retired-uc", *(blocked_extra or [])],
            "old_train_family_sha256_excluded": [builder.family_hash("Older prior train")],
            "old_val_family_sha256": [builder.family_hash("Older prior val")],
            "old_test_family_sha256": [builder.family_hash("Older retired test")],
        }
        write_json(self.holdout_lock, holdout)
        write_json(self.selection_manifest, {
            "schema": "ultrachat_six_topic_3000_v1",
            "train_val_prompts_sha256": builder.sha256_file(self.active_path),
            "holdout_lock_sha256": builder.sha256_file(self.holdout_lock),
            "counts": {"train": 1, "val": 1, "test": 0},
        })

    def _build(self, output: str = "out", *, plans: tuple[builder.SourcePlan, ...] | None = None
               ) -> dict:
        return builder.build(
            self.source_lock, self.active_path, self.selection_manifest,
            self.holdout_lock, self.root / output, plans=plans or self.plans,
            seed=20261101, expected_old_active_counts={"train": 1, "val": 1},
            expected_retired_test_count=1)

    def test_deterministic_split_manifest_and_answer_exclusion(self) -> None:
        first = self._build("out-a")
        second = self._build("out-b")
        for filename in ("datasets_manifest.json", "pilot_train.jsonl", "pilot_val.jsonl"):
            self.assertEqual((self.root / "out-a" / filename).read_bytes(),
                             (self.root / "out-b" / filename).read_bytes())
        self.assertEqual(first, second)
        self.assertEqual(first["counts"], {"train": 4, "val": 4})
        self.assertEqual(first["selection_order"], ["pilot_val", "pilot_train"])
        self.assertEqual(first["legacy_exclusions"]["retired_test_policy"],
                         "retired_never_used_for_v2")
        self.assertFalse(first["legacy_exclusions"][
            "retired_test_cross_source_near_duplicate_independence_proved"])
        self.assertTrue(first["retired_legacy_test_content_not_read"])
        self.assertEqual(first["sources"]["codealpaca"]["upstream_data_license"],
                         "CC-BY-NC-4.0")
        selected = (self.root / "out-a" / "pilot_train.jsonl").read_text(
            encoding="utf-8") + (self.root / "out-a" / "pilot_val.jsonl").read_text(
            encoding="utf-8")
        for reference in ("UNUSED_ASSISTANT", "UNUSED_RESPONSE", "UNUSED_OUTPUT",
                          "UNUSED_ANSWER"):
            self.assertNotIn(reference, selected)
        self.assertNotIn(self.prompts["ultrachat"][0], selected)
        self.assertNotIn(self.prompts["ultrachat"][1], selected)
        self.assertEqual(len({r["prompt_family_sha256"] for rows in first["selected"].values()
                              for r in rows}), 8)
        self.assertTrue(all("prompt" not in row for rows in first["selected"].values()
                            for row in rows))

    def test_old_train_val_near_duplicate_and_retired_family(self) -> None:
        rows, _ = builder.load_sources(self.source_lock, self.plans)
        index, blocked, provenance = builder.load_balanced3000_exclusions(
            rows["ultrachat"], self.active_path, self.selection_manifest,
            self.holdout_lock, expected_counts={"train": 1, "val": 1},
            expected_retired_test_count=1)
        old_val = self.prompts["ultrachat"][1]
        near = old_val[:-1] + "!"
        self.assertNotEqual(builder.family_hash(old_val), builder.family_hash(near))
        self.assertEqual(index.rejection_reason(builder.exact_hash(near),
                                                builder.family_hash(near),
                                                builder.char5gram_hashes(near)),
                         "char5gram_jaccard")
        retired_family = builder.family_hash("Synthetic retired test request")
        self.assertIn(retired_family, index.families)
        self.assertFalse(provenance["retired_test_cross_source_near_duplicate_independence_proved"])
        self.assertNotIn("uc-0", blocked)
        self.assertIn("uc-1", blocked)

    def test_prior_ultrachat_blocked_source_id_is_respected(self) -> None:
        self._write_old_active_metadata(blocked_extra=["uc-2"])
        manifest = self._build()
        ids = {row["source_row_id"] for rows in manifest["selected"].values()
               for row in rows if row["source"] == "ultrachat"}
        self.assertNotIn("uc-2", ids)

    def test_old_holdout_mismatch_hard_fails_without_output(self) -> None:
        self._write_old_active_metadata(altered_val_family=True)
        with self.assertRaisesRegex(ValueError, "validation differs"):
            self._build()
        self.assertFalse((self.root / "out").exists())

    def test_wrong_source_revision_and_file_sha_hard_fail(self) -> None:
        lock = json.loads(self.source_lock.read_text(encoding="utf-8"))
        lock["sources"]["gsm8k"]["revision"] = "main"
        write_json(self.source_lock, lock)
        with self.assertRaisesRegex(ValueError, "source identity"):
            self._build()
        lock["sources"]["gsm8k"]["revision"] = self.plans[-1].revision
        lock["sources"]["gsm8k"]["files"][0]["sha256"] = "0" * 64
        write_json(self.source_lock, lock)
        with self.assertRaisesRegex(ValueError, "source file SHA differs"):
            self._build()

    def test_quota_shortfall_hard_fails_without_partial_manifest(self) -> None:
        plans = tuple(dataclasses.replace(p, train_quota=4, val_quota=1)
                      if p.key == "dolly" else p for p in self.plans)
        with self.assertRaisesRegex(ValueError, "dolly train quota shortfall"):
            self._build(plans=plans)
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
