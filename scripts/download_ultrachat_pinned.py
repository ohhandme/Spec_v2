"""Download only the pinned UltraChat SFT Parquet shards with LFS SHA checks.

This utility never opens the sealed test examples. It verifies the public
repository's pinned file content before any downstream sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import requests


REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
BASE = f"https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/{REVISION}/data/"
FILES = {
    "train_sft-00000-of-00003-a3ecf92756993583.parquet": (243999189, "afa8fa7426081b2a0e732fb50dbb5cd402a28ad5f0dbe66c0d996d63e7220727"),
    "train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet": (243897199, "c8230190bc8b29084c974ff32f05589b458fad1908fc4b2fb2b9e1e9f7921f03"),
    "train_sft-00002-of-00003-ee46ed25cfae92c6.parquet": (244131731, "6fe7d2a5e95cf660f972ecaf304aa5632d7f53384e4b0e0d0f44d9c96733c03e"),
    "test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet": (81176553, "c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(session: requests.Session, dest: Path, name: str, size: int, sha: str) -> None:
    final = dest / name
    partial = dest / (name + ".part")
    if final.exists():
        if final.stat().st_size != size or sha256_file(final) != sha:
            raise ValueError(f"Existing final file failed pinned SHA: {name}")
        print(json.dumps({"file": name, "state": "verified_existing", "bytes": size}), flush=True)
        return
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Partial file exceeds pinned size: {name}")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(BASE + name, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if offset:
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError(f"Server did not honor resume range for {name}")
                elif response.status_code != 200:
                    raise ValueError(f"Unexpected initial HTTP status for {name}: {response.status_code}")
                with partial.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if partial.stat().st_size != size:
                raise IOError(f"Incomplete {name}: {partial.stat().st_size}/{size}")
            actual = sha256_file(partial)
            if actual != sha:
                raise ValueError(f"Pinned LFS SHA mismatch for {name}: {actual}")
            os.replace(partial, final)
            print(json.dumps({"file": name, "state": "verified", "bytes": size, "sha256": actual}), flush=True)
            return
        except (requests.RequestException, IOError) as exc:
            print(json.dumps({"file": name, "attempt": attempt + 1, "error": type(exc).__name__,
                              "partial_bytes": partial.stat().st_size if partial.exists() else 0}), flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("Unreachable download retry state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for name, (size, sha) in FILES.items():
            download_one(session, args.output, name, size, sha)
    manifest = {"dataset": "HuggingFaceH4/ultrachat_200k", "revision": REVISION,
                "files": {name: {"bytes": size, "sha256": sha} for name, (size, sha) in FILES.items()}}
    pending = args.output / "manifest.json.tmp"
    pending.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, args.output / "manifest.json")
    print(json.dumps({"state": "complete", "files": len(FILES)}), flush=True)


if __name__ == "__main__":
    main()
