"""Synthetic, test-content-free checks for the six-topic pilot selector."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import select_balanced_pilot as pilot


def lock(**changes: object) -> dict:
    value = {
        "schema": "ultrachat_six_topic_holdout_lock_v2",
        "dataset": pilot.DATASET,
        "revision": pilot.REVISION,
        "frozen_before_selection": True,
        "holdout_source_split": "train_sft",
        "new_validation_manifest_sha256": "a" * 64,
        "new_test_manifest_sha256": "b" * 64,
        "near_duplicate_audit_sha256": "c" * 64,
        "near_duplicate_threshold": 0.85,
        "old_train_source_ids": ["old-tr"],
        "old_train_family_sha256_excluded": [pilot.family_hash("Calculate 1 + 1.")],
        "old_val_family_sha256": [pilot.family_hash("Old validation")],
        "old_test_family_sha256": [pilot.family_hash("Old test")],
        "new_val_family_sha256": [pilot.family_hash("New validation")],
        "new_test_family_sha256": [pilot.family_hash("New test")],
        "blocked_train_source_ids_near_duplicate": [],
        "reserved_train_source_ids": ["old-tr"],
    }
    value.update(changes)
    return value


def token_ids(text: str) -> list[int]:
    return [1, *(ord(char) for char in text), 2]


class TopicTests(unittest.TestCase):
    def test_each_topic_and_uncertain_labels(self) -> None:
        examples = {
            "code": "Write a Python script.",
            "creative_writing": "Write a poem about the sea.",
            "knowledge_qa": "Who was Marie Curie?",
            "summarization_transformation": "Summarize this passage.",
            "practical_howto": "Give me a recipe for soup.",
            "business_marketing": "Develop a marketing plan for a new bakery.",
        }
        for topic, text in examples.items():
            with self.subTest(topic=topic):
                self.assertEqual(pilot.classify_topic(text).label, topic)
        self.assertEqual(pilot.classify_topic(
            "Write a Python program to create a marketing plan.").label,
                         "ambiguous")
        self.assertEqual(pilot.classify_topic("The blue sky is vast.").label,
                         "unclassified")

    def test_six_topic_selection_and_holdout_exclusion(self) -> None:
        raw = {
            "c": "Write a Python script.",
            "w": "Write a poem about the sea.",
            "k": "Who was Marie Curie?",
            "s": "Summarize this passage.",
            "p": "Give me a recipe for soup.",
            "b": "Develop a marketing plan for a new bakery.",
            "holdout": "New test",
            "near": "Give me a recipe for rice.",
            "ambiguous": "Write a Python program to create a marketing plan.",
            "unclassified": "The blue sky is vast.",
            "invalid": None,
            "old-tr": "Calculate 1 + 1.",
            "old-family": "Calculate 2 + 2.",
            "old-near": "Calculate 1 + 1!",
        }
        selected, audit, sample = pilot.select_balanced(
            raw, lock(blocked_train_source_ids_near_duplicate=["near", "old-near"]),
            token_ids, seed=20, quota=1,
        )
        self.assertTrue(audit["gate_passed"])
        self.assertEqual(len(selected), 6)
        self.assertEqual({row["task_type"] for row in selected}, set(pilot.TOPICS))
        self.assertEqual(len({row["prompt_family_sha256"] for row in selected}), 6)
        self.assertEqual(audit["excluded_and_classification_counts"]["ambiguous"], 1)
        self.assertEqual(audit["excluded_and_classification_counts"]["unclassified"], 1)
        self.assertEqual(audit["excluded_and_classification_counts"]["reserved_family"], 2)
        self.assertEqual(audit["excluded_and_classification_counts"]["reserved_source_id"], 3)
        self.assertEqual(audit["overlap_with_old_train_prompts"], 0)
        self.assertEqual(len(sample), 8)

    def test_short_topic_stops_without_reallocation(self) -> None:
        raw = {"practical-one": "Give me a recipe for soup.",
               "code-one": "Write a Python script."}
        selected, audit, _ = pilot.select_balanced(raw, lock(), token_ids,
                                                   seed=20, quota=2)
        self.assertFalse(audit["gate_passed"])
        self.assertEqual(len(selected), 2)
        self.assertEqual(audit["missing_quota"]["practical_howto"], 1)
        self.assertEqual(audit["missing_quota"]["creative_writing"], 2)

    def test_fixed_rank_independent_of_raw_order(self) -> None:
        raw = {
            "x": "Give me a recipe for soup.",
            "y": "Give me a recipe for rice.",
            "z": "Give me a recipe for pasta.",
        }
        a = pilot.select_balanced(raw, lock(), token_ids, seed=20, quota=1)[0]
        b = pilot.select_balanced(dict(reversed(list(raw.items()))), lock(),
                                  token_ids, seed=20, quota=1)[0]
        self.assertEqual(a, b)

    def test_lock_and_family_normalization(self) -> None:
        self.assertEqual(pilot.family_hash("ＡＢＣ 42"), pilot.family_hash("abc 99"))
        with self.assertRaisesRegex(ValueError, "new_test_family_sha256"):
            pilot.validate_lock(lock(new_test_family_sha256=[]))
        with self.assertRaisesRegex(ValueError, "threshold"):
            pilot.validate_lock(lock(near_duplicate_threshold=0.9))
        with self.assertRaisesRegex(ValueError, "old train source IDs"):
            pilot.validate_lock(lock(reserved_train_source_ids=[]))
        with self.assertRaisesRegex(ValueError, "old_train_family_sha256_excluded"):
            pilot.validate_lock(lock(old_train_family_sha256_excluded=[]))

    def test_source_loader_reads_only_first_user(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "train.jsonl"
            rows = [
                {"prompt_id": "a", "messages": [
                    {"role": "user", "content": "Write a poem."},
                    {"role": "assistant", "content": "do not use"}]},
                {"prompt_id": "b", "messages": [
                    {"role": "assistant", "content": "bad"}]},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                            encoding="utf-8")
            loaded = pilot.load_train_rows(path, 2)
            self.assertEqual(loaded, {"a": "Write a poem.", "b": None})
            self.assertNotIn("do not use", str(loaded))


if __name__ == "__main__":
    unittest.main()
