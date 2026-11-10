"""Safely schedule four disjoint fixed-probe collection shards on idle GPUs.

Run once per split (`train`, then `val`). The collector's immutable plan and
atomic per-prompt chunks are authoritative. This queue never modifies them or
stops a process it cannot prove is its own worker. It deliberately refuses the
sealed test split; the final test is a separate post-validation operation.
"""

from __future__ import annotations

import argparse
from collections import Counter
try:
    import fcntl
except ImportError:  # importable for local static/unit checks; execution is Linux-only
    fcntl = None
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts" / "build_fixed_probe.py"
MIN_FREE_MIB = 22000
MAX_CONTENTION_RESTARTS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                  allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def _proc_identity(pid: int) -> tuple[int, list[str], int, str] | None:
    """Return Linux PID start ticks, argv, UID and state; handle a vanished PID."""
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        start_ticks = int(fields[19])  # field 22; fields[0] is original field 3
        state = fields[0]
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().rstrip(b"\0")
        command = [value.decode("utf-8", "replace") for value in argv.split(b"\0")]
        uid = (Path("/proc") / str(pid)).stat().st_uid
        return start_ticks, command, uid, state
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


def owned_worker_live(worker: dict) -> bool:
    observed = _proc_identity(worker["pid"])
    return bool(observed and observed[0] == worker["start_ticks"]
                and observed[1] == worker["command"]
                and observed[2] == os.getuid() and observed[3] != "Z")


def _gpu_snapshot() -> tuple[dict[int, dict], dict[str, set[int]]]:
    devices_raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    processes_raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    devices: dict[int, dict] = {}
    for line in devices_raw.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"unrecognized GPU row: {line!r}")
        index, uuid, free, utilization = fields
        devices[int(index)] = {"uuid": uuid, "free_mib": int(free),
                               "utilization": int(utilization)}
    processes: dict[str, set[int]] = {}
    for line in processes_raw.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"unrecognized compute-process row: {line!r}")
        uuid, pid = fields
        processes.setdefault(uuid, set()).add(int(pid))
    return devices, processes


def _eligible(devices: dict[int, dict], processes: dict[str, set[int]],
              occupied: set[int]) -> set[int]:
    return {index for index, row in devices.items()
            if index not in occupied and row["free_mib"] >= MIN_FREE_MIB
            and row["utilization"] < 10 and not processes.get(row["uuid"])}


def stable_idle_gpus(occupied: set[int], *, delay_seconds: float = 2.0) -> list[int]:
    eligible: set[int] | None = None
    for sample in range(3):
        devices, processes = _gpu_snapshot()
        now = _eligible(devices, processes, occupied)
        eligible = now if eligible is None else eligible & now
        if sample != 2:
            time.sleep(delay_seconds)
    return sorted(eligible or ())


def _identity(args: argparse.Namespace) -> dict:
    sources = sorted((ROOT / "feedback_codec").glob("*.py")) + [COLLECTOR, Path(__file__)]
    return {
        "scope": "balanced3000_fixed_probe_collection_queue_v1",
        "python": str(args.python.resolve()),
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "dataset_meta": str(args.dataset_meta.resolve()),
        "dataset_meta_sha256": sha256_file(args.dataset_meta),
        "real_p0_report": str(args.p0_report.resolve()),
        "real_p0_report_sha256": sha256_file(args.p0_report),
        "split": args.split, "shard_count": args.shard_count,
        "max_new_tokens": args.max_new_tokens,
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path)
                          for path in sources},
    }


def _expected(args: argparse.Namespace) -> dict[int, set[str]]:
    result = {index: set() for index in range(args.shard_count)}
    seen_ids: set[str] = set()
    with args.dataset.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank prompt row {number}")
            row = json.loads(line)
            if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen_ids:
                raise ValueError(f"invalid/duplicate prompt ID on row {number}")
            seen_ids.add(row["id"])
            if row.get("split") == args.split:
                raw = hashlib.sha256(row["id"].encode("utf-8")).hexdigest()
                index = int(raw, 16) % args.shard_count
                result[index].add(raw + ".json")
    if not all(result.values()):
        raise ValueError("one or more shards have no prompts")
    return result


def _chunk_counts(output: Path, expected: dict[int, set[str]]) -> dict[int, int]:
    present = {path.name for path in (output / "chunks").glob("*.json")}
    expected_union = set().union(*expected.values())
    if present - expected_union:
        raise ValueError("chunks directory contains foreign prompt IDs")
    return {index: len(names & present) for index, names in expected.items()}


def _common_command(args: argparse.Namespace) -> list[str]:
    return [str(args.python), str(COLLECTOR),
            "--target", str(args.target), "--draft", str(args.draft),
            "--dataset", str(args.dataset), "--dataset-meta", str(args.dataset_meta),
            "--p0-report", str(args.p0_report), "--split", args.split,
            "--thinking-mode", "disabled", "--output", str(args.output),
            "--shard-count", str(args.shard_count),
            "--max-new-tokens", str(args.max_new_tokens)]


def _run_cpu_stage(command: list[str], log_path: Path) -> int:
    with log_path.open("wb") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=ROOT, timeout=3600, check=False)
    return int(completed.returncode)


def _launch(args: argparse.Namespace, status: dict, shard: int, gpu: int) -> dict:
    lock_path = args.output / f"shard-{shard}.lock"
    if lock_path.exists():
        raise RuntimeError(f"shard-{shard} lock exists; inspect owner before retry")
    attempt = int(status["attempts"].get(str(shard), 0)) + 1
    command = _common_command(args) + [
        "--resume", "--shard-index", str(shard), "--device", f"cuda:{gpu}",
        "--minimum-free-mib", str(MIN_FREE_MIB),
    ]
    log_path = args.state_dir / f"shard-{shard}-attempt-{attempt}.log"
    log = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()
    observed = _proc_identity(process.pid)
    if observed is None or observed[1] != command or observed[2] != os.getuid():
        raise RuntimeError("launched worker process identity could not be verified")
    worker = {"pid": process.pid, "start_ticks": observed[0],
              "command": command, "gpu_index": gpu,
              "gpu_uuid": _gpu_snapshot()[0][gpu]["uuid"],
              "attempt": attempt, "log": str(log_path),
              "started_unix_s": time.time(), "stop_reason": None}
    status["attempts"][str(shard)] = attempt
    status["workers"][str(shard)] = worker
    return worker


def _stop_owned(worker: dict) -> bool:
    if not owned_worker_live(worker):
        return False
    os.killpg(worker["pid"], signal.SIGTERM)
    for _ in range(20):
        if not owned_worker_live(worker):
            return True
        time.sleep(0.5)
    if owned_worker_live(worker):
        os.killpg(worker["pid"], signal.SIGKILL)
    return True


def _clear_dead_owned_shard_lock(args: argparse.Namespace, status: dict,
                                 shard: int, worker: dict) -> None:
    lock_path = args.output / f"shard-{shard}.lock"
    if not lock_path.exists():
        return
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("pid") != worker["pid"] or payload.get("shard_index") != shard or (
        payload.get("plan_sha256") != status["plan_sha256"]
        or owned_worker_live(worker)
    ):
        raise RuntimeError("shard lock owner/plan/live-process mismatch; manual audit needed")
    lock_path.unlink()


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.split not in ("train", "val"):
        raise ValueError("sealed test is not scheduled by this queue")
    if args.shard_count != 4 or args.max_new_tokens != 32:
        raise ValueError("balanced-3000 protocol requires four shards and 32 new tokens")
    if args.max_wait_seconds <= 0 or args.poll_seconds <= 0:
        raise ValueError("queue wait/poll bounds must be positive")
    if not all(path.is_file() for path in (
        args.python, args.dataset, args.dataset_meta, args.p0_report, COLLECTOR,
    )) or not args.target.is_dir() or not args.draft.is_dir():
        raise FileNotFoundError("missing pinned Python/model/dataset/P0 input")
    meta = json.loads(args.dataset_meta.read_text(encoding="utf-8"))
    dataset_sha = sha256_file(args.dataset)
    if meta.get("prompts_jsonl_sha256") != dataset_sha or (
        meta.get("output_sha256", dataset_sha) != dataset_sha
        or meta.get("counts", {}).get("test") != 0
    ):
        raise ValueError("active manifest hash/test seal differs")
    p0 = json.loads(args.p0_report.read_text(encoding="utf-8"))
    if p0.get("state") != "passed" or p0.get("dataset_sha256") != dataset_sha or (
        p0.get("dtype") != "bfloat16"
    ):
        raise ValueError("matching real BF16 P0 gate is not passed")


def run(args: argparse.Namespace) -> None:
    if fcntl is None or os.name != "posix":
        raise RuntimeError("collection controller runs only on the Linux GPU host")
    _validate_inputs(args)
    expected = _expected(args)
    identity = _identity(args)
    if not args.resume:
        if args.state_dir.exists() or args.output.exists():
            raise FileExistsError("queue/output exists; inspect and use --resume")
        args.state_dir.mkdir(parents=True)
    elif not args.state_dir.is_dir() or not args.output.is_dir():
        raise FileNotFoundError("resume needs existing state and collection plan")
    lock_file = args.state_dir / "controller.lock"
    with lock_file.open("a+b") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = args.state_dir / "status.json"
        if args.resume:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("identity") != identity or (
                status.get("plan_sha256") != sha256_file(args.output / "collection_plan.json")
            ):
                raise ValueError("resume identity/plan SHA changed")
            if status.get("state") == "complete":
                final = json.loads((args.output / "collection_protocol.json").read_text(
                    encoding="utf-8"))
                if final.get("fixed_prefix_sha256") != sha256_file(
                    args.output / "fixed_prefix.jsonl"
                ):
                    raise ValueError("completed collection SHA changed")
                return
            if status.get("state") not in ("collecting", "waiting_for_idle_gpu",
                                           "merging"):
                raise RuntimeError("failed/timed-out queue requires evidence review")
        else:
            status = {"schema": "balanced3000_collection_queue_v1", "identity": identity,
                      "state": "initializing", "started_unix_s": time.time(),
                      "plan_sha256": None, "workers": {}, "attempts": {},
                      "events": []}
            atomic_json(status_path, status)
            rc = _run_cpu_stage(_common_command(args) + ["--initialize-only"],
                                args.state_dir / "initialize.log")
            if rc != 0:
                status["state"] = "initialization_failed"
                status["events"].append({"at": time.time(), "exit_code": rc,
                                         "stage": "initialize"})
                atomic_json(status_path, status)
                raise RuntimeError("collector initialization failed; inspect log")
            status["plan_sha256"] = sha256_file(args.output / "collection_plan.json")
            status["state"] = "waiting_for_idle_gpu"
            atomic_json(status_path, status)

        deadline = status["started_unix_s"] + args.max_wait_seconds
        while True:
            counts = _chunk_counts(args.output, expected)
            status["completed_per_shard"] = {str(k): counts[k] for k in expected}
            status["expected_per_shard"] = {str(k): len(expected[k]) for k in expected}
            devices, compute = _gpu_snapshot()
            for shard_key, worker in list(status["workers"].items()):
                shard = int(shard_key)
                if owned_worker_live(worker):
                    foreign = compute.get(worker["gpu_uuid"], set()) - {worker["pid"]}
                    if foreign:
                        _stop_owned(worker)
                        worker["stop_reason"] = "foreign_compute_process_on_same_gpu"
                        status["events"].append({"at": time.time(), "shard": shard,
                            "reason": worker["stop_reason"], "foreign_pids": sorted(foreign)})
                        _clear_dead_owned_shard_lock(args, status, shard, worker)
                        del status["workers"][shard_key]
                    continue
                _clear_dead_owned_shard_lock(args, status, shard, worker)
                del status["workers"][shard_key]
                # A worker can atomically finish its final chunk after our first
                # directory snapshot but before its process exit.
                counts = _chunk_counts(args.output, expected)
                if counts[shard] != len(expected[shard]):
                    status["state"] = "worker_failed_partial"
                    status["events"].append({"at": time.time(), "shard": shard,
                        "reason": "worker_exited_before_all_atomic_chunks",
                        "log": worker["log"]})
                    for other in status["workers"].values():
                        _stop_owned(other)
                    atomic_json(status_path, status)
                    raise RuntimeError(f"shard {shard} exited partial; preserve log/chunks")

            if all(counts[shard] == len(expected[shard]) for shard in expected) and (
                not status["workers"]
            ):
                status["state"] = "merging"
                atomic_json(status_path, status)
                rc = _run_cpu_stage(_common_command(args) + [
                    "--resume", "--merge-only", "--shard-index", "0",
                ], args.state_dir / "merge.log")
                if rc != 0:
                    status["state"] = "merge_failed"
                    status["events"].append({"at": time.time(), "exit_code": rc,
                                             "stage": "merge"})
                    atomic_json(status_path, status)
                    raise RuntimeError("merge failed; inspect log and atomic chunks")
                protocol = json.loads((args.output / "collection_protocol.json").read_text(
                    encoding="utf-8"))
                if protocol.get("fixed_prefix_sha256") != sha256_file(
                    args.output / "fixed_prefix.jsonl"
                ) or protocol.get("audit", {}).get("prompts") != sum(map(len, expected.values())):
                    raise ValueError("merged corpus/protocol coverage or SHA mismatch")
                status["state"] = "complete"
                status["completed_unix_s"] = time.time()
                status["collection_protocol_sha256"] = sha256_file(
                    args.output / "collection_protocol.json"
                )
                atomic_json(status_path, status)
                return

            if time.time() > deadline:
                status["state"] = "wait_timeout_needs_diagnosis"
                status["events"].append({"at": time.time(), "reason": "max_wait_seconds"})
                for worker in status["workers"].values():
                    _stop_owned(worker)
                atomic_json(status_path, status)
                raise TimeoutError("bounded queue timed out; inspect progress and logs")

            occupied = {worker["gpu_index"] for worker in status["workers"].values()}
            pending = [shard for shard in expected if counts[shard] < len(expected[shard])
                       and str(shard) not in status["workers"]]
            if pending:
                for gpu in stable_idle_gpus(occupied):
                    if not pending:
                        break
                    shard = pending.pop(0)
                    if int(status["attempts"].get(str(shard), 0)) >= (
                        1 + MAX_CONTENTION_RESTARTS
                    ):
                        status["state"] = "contention_restart_limit"
                        atomic_json(status_path, status)
                        raise RuntimeError("contention retry limit reached")
                    # Recheck after three admission samples, immediately before launch.
                    now_devices, now_compute = _gpu_snapshot()
                    if gpu not in _eligible(now_devices, now_compute, occupied):
                        pending.insert(0, shard)
                        continue
                    worker = _launch(args, status, shard, gpu)
                    occupied.add(gpu)
                    status["events"].append({"at": time.time(), "shard": shard,
                        "gpu_index": gpu, "pid": worker["pid"], "attempt": worker["attempt"]})
                    status["state"] = "collecting"
                    atomic_json(status_path, status)
            elif not status["workers"]:
                status["state"] = "waiting_for_idle_gpu"
            atomic_json(status_path, status)
            time.sleep(args.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-meta", type=Path, required=True)
    parser.add_argument("--p0-report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--max-wait-seconds", type=int, default=86400)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
