"""Fetch a pinned Qwen3-8B mirror snapshot and verify every file by SHA256.

This is an operational asset transfer. The expected hashes come from the
previously verified Qwen/Hugging Face snapshot in qwen3-8b-expected.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


MODEL_ID = "Qwen/Qwen3-8B"
HF_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODELSCOPE_REVISION = "7c9709d23bd2136dac1d6ea1fe30f4107d681cd6"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if record.get("repo_id") != MODEL_ID or record.get("revision") != HF_REVISION:
        raise ValueError("Expected model identity does not match pinned snapshot")
    if {entry["file"]: entry["sha256"] for entry in record["weight_download_sources"]} != {
        item["path"]: item["sha256"]
        for item in record["files"] if item["path"].endswith(".safetensors")
    }:
        raise ValueError("Mirror weight hashes disagree with official snapshot")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    status = destination.parent / "qwen3-8b-download-status.json"
    lock = destination.parent / "qwen3-8b-download.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock_fd, f"{os.getpid()}\n".encode())
    os.close(lock_fd)
    started = time.time()
    files: list[dict] = []
    # ModelScope's pinned mirror omits the repository's documentation files.
    # They are not loaded by Transformers; all inference assets remain required.
    inference_files = [entry for entry in record["files"]
                       if entry["path"] not in {"LICENSE", "README.md"}]
    try:
        for entry in inference_files:
            name = entry["path"]
            if Path(name).name != name:
                raise ValueError(f"Unsafe manifest path: {name}")
            final = destination / name
            partial = destination / (name + ".part")
            expected_size = int(entry["bytes"])
            expected_sha = entry["sha256"]
            if final.exists():
                if final.stat().st_size != expected_size or sha256_file(final) != expected_sha:
                    raise ValueError(f"Unverified final asset exists: {name}")
            else:
                if partial.exists() and partial.stat().st_size > expected_size:
                    raise ValueError(f"Partial exceeds expected size: {name}")
                url = (
                    f"https://modelscope.cn/models/{MODEL_ID}/resolve/"
                    f"{MODELSCOPE_REVISION}/{name}"
                )
                for attempt in range(1, 13):
                    atomic_json(status, {
                        "state": "downloading", "repo_id": MODEL_ID,
                        "hf_revision": HF_REVISION,
                        "modelscope_revision": MODELSCOPE_REVISION,
                        "current_file": name, "attempt": attempt,
                        "partial_bytes": partial.stat().st_size if partial.exists() else 0,
                        "verified_files": files, "started_unix": started,
                    })
                    result = subprocess.run([
                        "curl", "--noproxy", "*", "-fL", "-sS", "-C", "-",
                        "--connect-timeout", "15", "--max-time", "1800",
                        "--speed-limit", "10240", "--speed-time", "120",
                        url, "-o", str(partial),
                    ], check=False)
                    if result.returncode == 0:
                        break
                    if attempt == 12:
                        raise RuntimeError(f"Transfer failed for {name}: curl {result.returncode}")
                    time.sleep(3)
                if partial.stat().st_size != expected_size:
                    raise ValueError(f"Size mismatch for {name}")
                actual_sha = sha256_file(partial)
                if actual_sha != expected_sha:
                    raise ValueError(f"SHA256 mismatch for {name}: {actual_sha}")
                os.replace(partial, final)
            files.append({"file": name, "bytes": expected_size, "sha256": expected_sha})
            atomic_json(status, {
                "state": "downloading", "repo_id": MODEL_ID,
                "hf_revision": HF_REVISION,
                "modelscope_revision": MODELSCOPE_REVISION,
                "current_file": None, "verified_files": files,
                "started_unix": started,
            })
        atomic_json(status, {
            "state": "complete", "repo_id": MODEL_ID,
            "hf_revision": HF_REVISION,
            "modelscope_revision": MODELSCOPE_REVISION,
            "verified_files": files, "omitted_non_inference_files": ["LICENSE", "README.md"],
            "total_bytes": sum(f["bytes"] for f in files),
            "started_unix": started, "finished_unix": time.time(),
        })
    except BaseException as exc:
        atomic_json(status, {
            "state": "failed", "repo_id": MODEL_ID,
            "hf_revision": HF_REVISION,
            "modelscope_revision": MODELSCOPE_REVISION,
            "current_file": name if "name" in locals() else None,
            "verified_files": files, "error": repr(exc),
            "started_unix": started, "finished_unix": time.time(),
        })
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
