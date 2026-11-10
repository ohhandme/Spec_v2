"""Wait for verified Qwen3-8B and an idle 4090, then run the real P0 once.

This queue owns only its child process. It never stops unrelated GPU work.
P1 remains gated on the resulting real CUDA report and its source hashes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def available_gpu() -> list[int]:
    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=15,
    ).splitlines()
    process_lines = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"], text=True, timeout=15,
    ).splitlines()
    busy_uuids = {line.split(",", 1)[0].strip() for line in process_lines
                  if "," in line and line.strip()}
    idle = []
    for line in gpu_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        index, uuid, free_mib, utilization = fields
        if (uuid not in busy_uuids and int(free_mib) >= 22000
                and int(utilization) < 10):
            idle.append(int(index))
    return idle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--download-status", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--max-wait-seconds", type=int, default=86400)
    args = parser.parse_args()
    if args.max_wait_seconds <= 0:
        raise ValueError("max wait must be positive")
    if args.status.exists() or args.output.exists():
        raise FileExistsError("P0 queue/output already exists; inspect it before resuming")
    args.status.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    lock = args.status.with_suffix(".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.close(descriptor)
    try:
        consecutive: dict[int, int] = {}
        while True:
            if time.time() - started > args.max_wait_seconds:
                raise TimeoutError("model/GPU readiness exceeded bounded P0 wait")
            if args.download_status.exists():
                download = json.loads(args.download_status.read_text(encoding="utf-8"))
                if download.get("state") == "failed":
                    raise RuntimeError("verified Qwen3-8B download failed; inspect its log")
                ready = download.get("state") == "complete"
            else:
                ready = False
            if not ready:
                atomic_json(args.status, {"state": "waiting_for_verified_model",
                                          "pid": os.getpid(), "started_unix": started,
                                          "updated_unix": time.time()})
                time.sleep(30)
                continue
            idle = available_gpu()
            consecutive = {index: consecutive.get(index, 0) + 1 for index in idle}
            selected = next((index for index, count in sorted(consecutive.items())
                             if count >= 3), None)
            if selected is None:
                atomic_json(args.status, {"state": "waiting_for_idle_gpu",
                                          "pid": os.getpid(), "started_unix": started,
                                          "updated_unix": time.time(),
                                          "idle_consecutive": consecutive})
                time.sleep(5)
                continue
            args.output.parent.mkdir(parents=True, exist_ok=True)
            command = [str(args.python), str(ROOT / "scripts" / "run_real_p0.py"),
                       "--target", str(args.target), "--draft", str(args.draft),
                       "--target-download-status", str(args.download_status),
                       "--dataset", str(args.dataset),
                       "--dataset-manifest", str(args.dataset_manifest),
                       "--output", str(args.output), "--device", "cuda:0",
                       "--request-count", "4", "--max-new-tokens", "24"]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(selected)
            log_path = args.status.with_suffix(".log")
            with log_path.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                         stdout=log, stderr=subprocess.STDOUT)
                atomic_json(args.status, {"state": "running", "pid": os.getpid(),
                                          "worker_pid": child.pid, "physical_gpu": selected,
                                          "started_unix": started,
                                          "updated_unix": time.time(),
                                          "command": command, "log": str(log_path)})
                return_code = child.wait()
            report = args.output / "report.json"
            passed = (return_code == 0 and report.is_file()
                      and json.loads(report.read_text(encoding="utf-8")).get("state") == "passed")
            atomic_json(args.status, {"state": "passed" if passed else "failed",
                                      "pid": os.getpid(), "worker_pid": child.pid,
                                      "physical_gpu": selected,
                                      "started_unix": started,
                                      "completed_unix": time.time(),
                                      "return_code": return_code,
                                      "report": str(report) if passed else None,
                                      "log": str(log_path)})
            if not passed:
                raise RuntimeError("real P0 worker failed; preserve output/log for diagnosis")
            return
    except BaseException as exc:
        previous = (json.loads(args.status.read_text(encoding="utf-8"))
                    if args.status.exists() else {})
        atomic_json(args.status, previous | {"state": "failed", "pid": os.getpid(),
                                             "started_unix": started,
                                             "completed_unix": time.time(),
                                             "error": repr(exc),
                                             "traceback": traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
