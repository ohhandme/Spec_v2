"""Conversion tests: source assistant text must never enter prepared JSONL."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import convert_pinned_ultrachat as converter


class ConversionTests(unittest.TestCase):
    def test_first_user_only(self) -> None:
        row = converter.first_user_only({
            "prompt_id": "source-1",
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {"role": "assistant", "content": "SECRET ANSWER"},
            ],
        })
        self.assertEqual(row["messages"],
                         [{"role": "user", "content": "What is the weather?"}])
        self.assertNotIn("SECRET ANSWER", str(row))

    def test_tiny_parquet_conversion(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow unavailable in this Python runtime")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            source = root / "source.parquet"
            output = root / "converted.jsonl"
            records = [
                {"prompt_id": "a", "messages": [
                    {"role": "user", "content": "Write a poem."},
                    {"role": "assistant", "content": "SECRET ANSWER"}]},
                {"prompt_id": "b", "messages": [
                    {"role": "assistant", "content": "wrong first role"}]},
            ]
            pq.write_table(pa.Table.from_pylist(records), source)
            self.assertEqual(converter.convert([source], output, expected_rows=2), 2)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["messages"],
                             [{"role": "user", "content": "Write a poem."}])
            self.assertEqual(rows[1]["messages"], [])
            self.assertNotIn("SECRET ANSWER", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
