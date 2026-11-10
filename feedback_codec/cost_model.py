"""Explicitly hypothetical serial cloud/edge cost accounting.

Inputs must come from separate real GPU and network measurements.  This module
never treats the full-prefix reference backend's runtime as an acceleration
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CostInputs:
    rtt_ms: float
    uplink_bytes_per_second: float
    downlink_bytes_per_second: float
    draft_ms_per_token: float
    verify_ms_per_round: float
    cloud_encode_ms_per_message: float
    edge_decode_ms_per_message: float
    edge_repair_ms_per_token: float
    ordinary_reply_bytes_per_round: int = 6

    def __post_init__(self) -> None:
        for name in (
            "rtt_ms", "draft_ms_per_token", "verify_ms_per_round",
            "cloud_encode_ms_per_message", "edge_decode_ms_per_message",
            "edge_repair_ms_per_token",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("uplink_bytes_per_second", "downlink_bytes_per_second"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.ordinary_reply_bytes_per_round < 0:
            raise ValueError("ordinary_reply_bytes_per_round cannot be negative")


def estimate_serial_request(row: dict, inputs: CostInputs) -> dict[str, float | bool]:
    """Estimate one request from recorded *counts* and supplied cost inputs.

    The proposal assumes one 32-bit token ID per uploaded draft token, plus an
    explicit ordinary response cost, and precisely eight additional bytes per
    emitted feedback message.  It is a scenario calculation, not a measured
    latency or an asynchronous pipeline model.
    """
    rounds = row["verification_rounds"]
    proposed = row["proposed_draft_tokens"]
    feedback_bytes = row["feedback_bytes"]
    repair_positions = row.get("repair_positions", 0)
    if any(type(x) is not int or x < 0 for x in (rounds, proposed, feedback_bytes, repair_positions)):
        raise ValueError("request counts must be nonnegative integers")
    if feedback_bytes % 8 or repair_positions > proposed:
        raise ValueError("feedback and repair counts are inconsistent")
    uplink_bytes = proposed * 4
    downlink_bytes = rounds * inputs.ordinary_reply_bytes_per_round + feedback_bytes
    messages = feedback_bytes // 8
    breakdown = {
        "draft_ms": proposed * inputs.draft_ms_per_token,
        "target_verify_ms": rounds * inputs.verify_ms_per_round,
        "round_trip_ms": rounds * inputs.rtt_ms,
        "upload_ms": 1000.0 * uplink_bytes / inputs.uplink_bytes_per_second,
        "download_ms": 1000.0 * downlink_bytes / inputs.downlink_bytes_per_second,
        "feedback_encode_ms": messages * inputs.cloud_encode_ms_per_message,
        "feedback_decode_ms": messages * inputs.edge_decode_ms_per_message,
        "feedback_repair_ms": repair_positions * inputs.edge_repair_ms_per_token,
    }
    return {
        "assumption_based": True,
        "uplink_bytes": uplink_bytes,
        "downlink_bytes": downlink_bytes,
        **breakdown,
        "estimated_total_ms": sum(breakdown.values()),
    }
