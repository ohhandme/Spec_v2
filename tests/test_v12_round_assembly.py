import types

import pytest
import torch

from feedback_codec.v12_draft_observer import draft_step_stats
from feedback_codec.v12_round_assembly import assemble_request, digest


def _sample():
    prompt = types.SimpleNamespace(
        id="train-one", split="train", family_id="family-one",
        task_type="ultrachat", input_ids=(1, 2))
    first_draft = (11, 12, 3, 4, 5, 6, 7, 8)
    summary = types.SimpleNamespace(
        greedy_token_id=13,
        top_token_ids=tuple([13] + [x for x in range(16) if x != 13]),
        top_logits_fp16=tuple([3.0] + [1.0] * 15),
        logsumexp_fp32=3.5, vocab_size=16)
    first = types.SimpleNamespace(
        prefix_before=(1, 2), proposals=first_draft, accepted_count=1,
        committed_tokens=(11, 13), rejected_token=12,
        correction_token=13, bonus_token=None,
        observation_hidden=torch.ones(4), observation_logit_summary=summary)
    second = types.SimpleNamespace(
        prefix_before=(1, 2, 11, 13), proposals=(0,), accepted_count=1,
        committed_tokens=(0,), rejected_token=None,
        correction_token=None, bonus_token=None,
        observation_hidden=None, observation_logit_summary=None)
    steps = []
    for prefix, proposal in (([1, 2], first_draft), ([1, 2, 11, 13], (0,))):
        for i, token in enumerate(proposal):
            z = torch.zeros(16)
            z[token] = 2
            steps.append(draft_step_stats(
                z, token, prefix=prefix + list(proposal[:i]), position=i + 1))
    return dict(
        prompt=prompt, records=[first, second], draft_steps=steps,
        oracle_tokens=[11, 13, 0],
        oracle_steps=[{"greedy_token_id": x} for x in (11, 13, 0)],
        eos_token_id=0, dataset_lock_sha256="a" * 64,
        policy_id="raw", execution_id="locked",
        hidden_ref=lambda i, r, valid: {
            "path": "hidden/one.safetensors", "key": f"round_{i}",
            "shape": [4], "dtype": "torch.float32", "sha256": "b" * 64,
            "prefix_sha256": digest(valid)},
        oracle_ref=lambda offset, length: {
            "path": "oracle_tapes/one.json", "sha256": "c" * 64,
            "offset": offset, "length": length},
    )


def test_actual_round_links_and_future_are_separate():
    observations, labels = assemble_request(**_sample())
    assert len(observations) == len(labels) == 2
    assert observations[0]["first_reject_position"] == 2
    assert labels[0]["next_label_status"] == "terminal_eos_without_rejection"
    assert labels[0]["target_future"]["token_ids"] == [0]
    assert observations[1]["prefix_token_ids"] == observations[0]["prefix_after_commit_token_ids"]
    assert observations[1]["target_valid_hidden_ref"] is None
    assert labels[1]["next_label_status"] == "no_next_round"
    assert not any("future" in key or key.startswith("next_")
                   for row in observations for key in row)


def test_missing_proposal_step_or_wrong_oracle_fails():
    args = _sample()
    args["draft_steps"] = args["draft_steps"][:-1]
    with pytest.raises(ValueError, match="step count"):
        assemble_request(**args)
    args = _sample()
    args["oracle_tokens"][-1] = 5
    with pytest.raises(ValueError, match="packed stream"):
        assemble_request(**args)
