"""Apply externally measured GPU/network costs to recorded round counts.

The output is an assumption-based serial scenario, never observed speedup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feedback_codec.cost_model import CostInputs, estimate_serial_request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--assumptions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    inputs = CostInputs(**json.loads(args.assumptions.read_text(encoding="utf-8")))
    rows = [
        {"id": row["id"], "strategy": row["strategy"], **estimate_serial_request(row, inputs)}
        for row in comparison["rows"]
    ]
    summary = {
        strategy: mean(row["estimated_total_ms"] for row in rows if row["strategy"] == strategy)
        for strategy in sorted({row["strategy"] for row in rows})
    }
    report = {
        "kind": "hypothetical_serial_cost_scenario",
        "measured_latency": False,
        "assumptions": vars(inputs),
        "rows": rows,
        "mean_estimated_total_ms": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
