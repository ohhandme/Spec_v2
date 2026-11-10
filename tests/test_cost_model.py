from feedback_codec.cost_model import CostInputs, estimate_serial_request


def test_eight_feedback_bytes_are_an_explicit_additional_cost():
    inputs = CostInputs(
        rtt_ms=10,
        uplink_bytes_per_second=1000,
        downlink_bytes_per_second=1000,
        draft_ms_per_token=1,
        verify_ms_per_round=2,
        cloud_encode_ms_per_message=0.5,
        edge_decode_ms_per_message=0.25,
        edge_repair_ms_per_token=0.1,
    )
    row = {
        "verification_rounds": 2,
        "proposed_draft_tokens": 6,
        "feedback_bytes": 8,
        "repair_positions": 3,
    }
    result = estimate_serial_request(row, inputs)
    assert result["assumption_based"] is True
    assert result["uplink_bytes"] == 24
    assert result["downlink_bytes"] == 20  # two ordinary replies plus eight feedback bytes
    assert abs(result["estimated_total_ms"] - (6 + 4 + 20 + 24 + 20 + 0.5 + 0.25 + 0.3)) < 1e-8
