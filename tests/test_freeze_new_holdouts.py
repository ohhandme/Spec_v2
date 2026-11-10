"""Synthetic holdout freeze and cross-split hygiene checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import freeze_new_holdouts as freeze
from select_balanced_pilot import TOPICS, family_hash


def token_ids(text: str) -> list[int]:
    return [1, *(ord(char) for char in text), 2]


def synthetic_classify(text: str) -> object:
    label = text.split(":", 1)[0].casefold()
    if label not in TOPICS:
        label = "unclassified"
    return SimpleNamespace(label=label, hits={label: [label]} if label in TOPICS else {})


class HoldoutTests(unittest.TestCase):
    def test_normalization_and_near_duplicate(self) -> None:
        self.assertTrue(freeze.near_duplicate(
            freeze.shingles("ＡＢＣ 123"), freeze.shingles("abc 999")))
        self.assertFalse(freeze.near_duplicate(
            freeze.shingles("a mathematical equation"),
            freeze.shingles("a fictional character")))

    def test_old_provenance_and_no_answer_use(self) -> None:
        train = {"tr": "Write a poem about rain.", "va": "Old validation"}
        test = {"te": "Old test"}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "old.jsonl"
            records = [
                {"split": split, "source_id": source_id,
                 "prompt_family_sha256": family_hash(text),
                 "input_ids": [9, 9, 9]}
                for split, source_id, text in (
                    ("train", "tr", train["tr"]),
                    ("val", "va", train["va"]),
                    ("test", "te", test["te"]),
                )
            ]
            path.write_text("\n".join(json.dumps(row) for row in records) + "\n",
                            encoding="utf-8")
            old = freeze.old_split_metadata(path, train, lambda _ids: test["te"],
                                             {"train": 1, "val": 1, "test": 1})
            self.assertEqual([item[0] for item in old["test"]], ["te"])
            self.assertNotIn("input_ids", str(old))

    def test_six_topics_new_val_test_and_train_blockers(self) -> None:
        old = {
            "train": [("old-tr", family_hash("An unrelated prior prompt"),
                       "An unrelated prior prompt")],
            "val": [("old-va", family_hash("Old validation"), "Old validation")],
            "test": [("old-te", family_hash("Old test"), "Old test")],
        }
        candidates = {
            f"{topic}-{suffix}": f"{topic}: {suffix}"
            for topic in TOPICS for suffix in
            ("alpha distinct words here", "zebra unrelated phrase now")
        }
        candidates["old-te"] = "Old test"
        val, test, audit, sample = freeze.select_holdouts(
            candidates, old, token_ids, seed=9,
            val_per_topic=1, test_per_topic=1,
            classify=synthetic_classify,
        )
        self.assertTrue(audit["gate_passed"])
        self.assertEqual(len(val), 6)
        self.assertEqual(len(test), 6)
        self.assertFalse(set(row["source_id"] for row in val) &
                         set(row["source_id"] for row in test))
        self.assertTrue(all("text" in row for row in sample))
        train = {
            "same-new": candidates[val[0]["source_id"]],
            "same-old": "Old validation",
            "same-old-train": "An unrelated prior prompt",
            "near-old-train": "An unrelated prior prompt!",
            "different": "Describe the weather today.",
        }
        train.update(candidates)
        blocked = freeze.blocked_near_duplicate_train(train, old,
                                                       val, test)
        self.assertIn("same-new", blocked)
        self.assertIn("same-old", blocked)
        self.assertIn("same-old-train", blocked)
        self.assertIn("near-old-train", blocked)
        self.assertNotIn("different", blocked)

    def test_quota_failure_is_explicit(self) -> None:
        old = {key: [] for key in ("train", "val", "test")}
        val, test, audit, _ = freeze.select_holdouts(
            {"one": f"{TOPICS[0]}: a"}, old, token_ids, seed=9,
            val_per_topic=1, test_per_topic=1, classify=synthetic_classify,
        )
        self.assertFalse(audit["gate_passed"])
        self.assertEqual(len(val) + len(test), 1)
        self.assertIn(f"test_{TOPICS[0]}", audit["missing_quota"])

    def test_old_sealed_user_decode_uses_only_first_turn(self) -> None:
        class FakeTokenizer:
            def decode(self, ids, skip_special_tokens=False):
                self_last = ids  # keep the shape in this synthetic decoder
                assert self_last == [1, 2, 3]
                assert skip_special_tokens is False
                return "<|im_start|>user\nWhat is a star?<|im_end|><|im_start|>assistant\n"

        text = freeze.decode_locked_qwen_user([1, 2, 3], FakeTokenizer())
        self.assertEqual(text, "What is a star?")


if __name__ == "__main__":
    unittest.main()
