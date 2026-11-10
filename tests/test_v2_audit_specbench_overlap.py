"""Focused CPU checks for the sealed benchmark overlap gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v2_audit_specbench_overlap as audit  # noqa: E402
import v2_build_dataset_manifest as builder  # noqa: E402


class TestOverlapAudit(unittest.TestCase):
    def test_normalization_matches_frozen_selection_rule(self) -> None:
        text = "PRICE 1,200.50  \u212a  year42\nJuly 77!"
        self.assertEqual(audit.normalize(text), builder.normalize_family(text))
        self.assertEqual(audit.fingerprints(text)[1], builder.family_hash(text))

    def test_near_duplicate_and_cross_split_are_reported(self) -> None:
        index = audit.Index()
        original = "Explain why ancient ceramics survive in dry deserts."
        index.add("pilot-1", "train", original)
        changed = "Explain why ancient ceramics survive in dry deserts!"
        matches = index.overlaps(changed)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["reason"], "char5_jaccard")
        self.assertGreaterEqual(matches[0]["similarity"], audit.THRESHOLD)
        self.assertEqual(index.overlaps(changed, other_split="train"), [])

    def test_distinct_topic_does_not_match(self) -> None:
        index = audit.Index()
        index.add("pilot-1", "train", "Explain ancient ceramics in dry deserts.")
        self.assertEqual(index.overlaps("Write Rust code for JSON parsing."), [])


if __name__ == "__main__":
    unittest.main()
