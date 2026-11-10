"""Small deterministic contracts for the versioned formal reservation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import v2_build_dataset_manifest as pilot  # noqa: E402
import v2_reserve_formal_after_pilot as formal  # noqa: E402


def fixtures():
    labels = {"ultrachat": "star cluster", "dolly": "orchid habitat",
              "codealpaca": "binary heap", "gsm8k": "river delta"}
    distinct = ("azure cosmos telescope", "luminous planetary formation",
                "emerald field ecology", "copper industrial archive",
                "violet manuscript translation")
    rows = {source: [{"source_row_id": f"{source}:{i}",
                      "source_file_sha256": "a" * 64,
                      "prompt": f"Explain {label} through {distinct[i]}."}
                     for i in range(5)]
            for source, label in labels.items()}
    quotas = {source: {split: 1 for split in formal.SPLITS} for source in labels}
    return rows, quotas


class FormalReservationTests(unittest.TestCase):
    def test_formal_val_precedes_train_and_is_disjoint(self):
        rows, quotas = fixtures()
        selected, _ = formal.select_formal(rows, pilot.SimilarityIndex(), set(),
                                           seed=20261101, quotas=quotas)
        self.assertEqual([len(selected[split]) for split in formal.SPLITS], [4, 4])
        self.assertTrue(
            {row["source_row_id"] for row in selected[formal.SPLITS[0]]}.isdisjoint(
                {row["source_row_id"] for row in selected[formal.SPLITS[1]]}))
        audit = formal.audit_formal(selected, pilot.SimilarityIndex())
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(
            audit["cross_prior_pilot_specbench_and_internal_overlap_count"], 0)

    def test_formal_quota_shortfall_is_not_silently_scaled(self):
        rows, quotas = fixtures()
        rows["gsm8k"] = []
        with self.assertRaisesRegex(ValueError, "quota shortfall"):
            formal.select_formal(rows, pilot.SimilarityIndex(), set(),
                                 seed=20261101, quotas=quotas)


if __name__ == "__main__":
    unittest.main()
