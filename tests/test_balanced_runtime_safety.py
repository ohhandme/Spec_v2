"""Focused local checks for sealed-test and four-shard safety gates."""

from __future__ import annotations

from argparse import Namespace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load_script("run_balanced_collection_queue.py")
unseal = load_script("unseal_final_test_manifest.py")


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                            for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BalancedRuntimeSafetyTests(unittest.TestCase):
    def test_four_shards_partition_prompt_ids_once(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = Path(folder) / "active.jsonl"
            rows = [{"id": f"prompt-{i}", "split": "train"} for i in range(3000)]
            rows += [{"id": f"val-{i}", "split": "val"} for i in range(120)]
            write_jsonl(path, rows)
            result = queue._expected(Namespace(dataset=path, split="train", shard_count=4))
            self.assertEqual(sum(map(len, result.values())), 3000)
            self.assertEqual(len(set().union(*result.values())), 3000)
            self.assertTrue(all(result.values()))

    def test_gpu_admission_needs_three_clean_samples(self):
        devices = {0: {"uuid": "g0", "free_mib": 24199, "utilization": 0},
                   1: {"uuid": "g1", "free_mib": 24199, "utilization": 0}}
        samples = [(devices, {}), (devices, {"g1": {9001}}), (devices, {})]
        with patch.object(queue, "_gpu_snapshot", side_effect=samples), patch.object(
            queue.time, "sleep", return_value=None
        ):
            self.assertEqual(queue.stable_idle_gpus(set()), [0])

    def test_test_file_is_not_read_when_val_gate_fails(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            root = Path(folder)
            active = root / "active.jsonl"
            meta = root / "meta.json"
            decision = root / "decision.json"
            active_sha = write_jsonl(active, [])
            sealed_sha = "b" * 64
            meta.write_text(json.dumps({"prompts_jsonl_sha256": active_sha,
                "counts": {"train": 6, "val": 6, "test": 0},
                "sealed_test_sha256": sealed_sha}), encoding="utf-8")
            decision.write_text(json.dumps({"schema": "balanced3000_validation_selection_v1",
                "state": "negative_stop", "active_prompts_sha256": active_sha,
                "sealed_test_sha256": sealed_sha}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation selection lock"):
                unseal.create_final_manifest(
                    active_path=active, active_meta_path=meta,
                    sealed_test_path=root / "nonexistent-sealed-test.jsonl",
                    sealed_test_sha=sealed_sha, selection_lock_path=decision,
                    output_dir=root / "final", expected_train=6, expected_val=6,
                    expected_test=6)

    def test_late_manifest_preserves_original_train_val_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            root = Path(folder)
            topics = [f"topic_{i}" for i in range(6)]
            def row(prefix: str, topic: str, split: str, token: int) -> dict:
                return {"id": f"{prefix}-{topic}", "task_type": topic,
                        "split": split, "input_ids": [token],
                        "prompt_family_sha256": hashlib.sha256(
                            f"{prefix}-{topic}".encode()).hexdigest()}
            active_rows = [row("train", topic, "train", 10 + index)
                           for index, topic in enumerate(topics)]
            active_rows += [row("val", topic, "val", 20 + index)
                            for index, topic in enumerate(topics)]
            test_rows = [row("test", topic, "test", 30 + index)
                         for index, topic in enumerate(topics)]
            active = root / "active.jsonl"
            sealed = root / "test.jsonl"
            meta = root / "meta.json"
            decision = root / "decision.json"
            active_sha = write_jsonl(active, active_rows)
            sealed_sha = write_jsonl(sealed, test_rows)
            meta.write_text(json.dumps({"prompts_jsonl_sha256": active_sha,
                "counts": {"train": 6, "val": 6, "test": 0},
                "sealed_test_sha256": sealed_sha}), encoding="utf-8")
            decision.write_text(json.dumps({
                "schema": "balanced3000_validation_selection_v1",
                "state": "selected_for_one_time_test",
                "active_prompts_sha256": active_sha,
                "sealed_test_sha256": sealed_sha,
                "gate": {"all_three_seeds_D_greater_C": True,
                         "D_separates_from_E": True,
                         "full_token_eos_truncation_correctness": True},
                "validation_report_sha256": {str(seed): "a" * 64
                                             for seed in (42, 43, 44)},
                "selected_candidate": "target_8B",
            }), encoding="utf-8")
            result = unseal.create_final_manifest(
                active_path=active, active_meta_path=meta,
                sealed_test_path=sealed, sealed_test_sha=sealed_sha,
                selection_lock_path=decision, output_dir=root / "final",
                expected_train=6, expected_val=6, expected_test=6)
            final = root / "final" / "train_val_test_prompts.jsonl"
            self.assertEqual(final.read_bytes(), active.read_bytes() + sealed.read_bytes())
            self.assertEqual(result["final_prompts_sha256"], unseal.sha256_file(final))


if __name__ == "__main__":
    unittest.main()
