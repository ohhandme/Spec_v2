"""Convert verified official UltraChat Parquet to first-user-only JSONL.

The converter never writes assistant replies. Its input SHA manifest must be
frozen from the official revision before this command runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from select_balanced_pilot import DATASET, REVISION, sha256_file


def first_user_only(record: dict) -> dict:
    source_id = record.get("prompt_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("official row has no prompt_id")
    messages = record.get("messages")
    first = messages[0] if isinstance(messages, list) and messages else None
    if isinstance(first, dict) and first.get("role") == "user" and isinstance(
        first.get("content"), str
    ):
        return {"prompt_id": source_id,
                "messages": [{"role": "user", "content": first["content"]}]}
    return {"prompt_id": source_id, "messages": []}


def convert(paths: list[Path], output: Path, *, expected_rows: int) -> int:
    import pyarrow.parquet as pq

    seen: set[str] = set()
    rows = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1024,
                                              columns=["prompt_id", "messages"]):
                for record in batch.to_pylist():
                    compact = first_user_only(record)
                    source_id = compact["prompt_id"]
                    if source_id in seen:
                        raise ValueError(f"duplicate official prompt_id: {source_id}")
                    seen.add(source_id)
                    stream.write(json.dumps(compact, ensure_ascii=False,
                                            separators=(",", ":")) + "\n")
                    rows += 1
    if rows != expected_rows:
        raise ValueError(f"converted {rows} rows, expected {expected_rows}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, nargs=3, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--official-sha-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sha_lock = json.loads(args.official_sha_manifest.read_text(encoding="utf-8"))
    if sha_lock.get("dataset") != DATASET or sha_lock.get("revision") != REVISION:
        raise ValueError("official SHA lock dataset/revision differs")
    expected = sha_lock.get("parquet_sha256")
    if expected is None and isinstance(sha_lock.get("files"), dict):
        expected = {name: record.get("sha256") for name, record in
                    sha_lock["files"].items() if isinstance(record, dict)}
    if not isinstance(expected, dict):
        raise ValueError("official SHA lock lacks parquet_sha256")
    files = args.train_parquet + [args.test_parquet]
    if len({path.name for path in files}) != 4 or set(expected) != {path.name for path in files}:
        raise ValueError("official SHA lock must name exactly these four Parquet files")
    verified = {}
    for path in files:
        actual = sha256_file(path)
        if actual != expected[path.name]:
            raise ValueError(f"official Parquet SHA256 differs: {path.name}")
        verified[path.name] = actual
    args.output_dir.mkdir(parents=True)
    train_path = args.output_dir / "train_sft.jsonl"
    test_path = args.output_dir / "test_sft.jsonl"
    convert(args.train_parquet, train_path, expected_rows=207_865)
    convert([args.test_parquet], test_path, expected_rows=23_110)
    manifest = {
        "schema": "ultrachat_first_user_jsonl_v1",
        "dataset": DATASET, "revision": REVISION,
        "official_sha_manifest_sha256": sha256_file(args.official_sha_manifest),
        "parquet_sha256": verified,
        "train_sft_rows": 207_865, "test_sft_rows": 23_110,
        "train_sft_jsonl_sha256": sha256_file(train_path),
        "test_sft_jsonl_sha256": sha256_file(test_path),
        "assistant_reference_excluded": True,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "counts": [207_865, 23_110]}))


if __name__ == "__main__":
    main()
