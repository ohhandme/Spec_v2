import json

import pytest
from safetensors.torch import load_file

from feedback_codec.v12_storage import file_sha256, persist_request
from test_v12_round_assembly import _sample


def _provenance(args):
    prompt = args["prompt"]
    return {
        "dataset_lock_sha256": args["dataset_lock_sha256"],
        "prompt_id": prompt.id, "family_id": prompt.family_id,
        "split": prompt.split, "source": prompt.task_type,
        "execution_id": args["execution_id"],
        "policy_id": args["policy_id"],
        "draft_checkpoint_sha256": "a" * 64,
        "feedback_policy_sha256": "b" * 64,
        "seed": 42, "model_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "chat_template_sha256": "e" * 64,
        "verifier_source_sha256": "f" * 64,
        "extension_source_sha256": "d" * 64,
        "observer_source_sha256": "1" * 64,
        "schema_lock_sha256": "e" * 64,
        "collection_phase": "train_collection",
        "max_new_tokens": 128,
    }


def test_request_files_are_separate_and_manifest_is_last(tmp_path):
    args = _sample()
    hidden_ref = args.pop("hidden_ref")
    oracle_ref = args.pop("oracle_ref")
    root = tmp_path / "train"
    result = persist_request(
        **args, root=root, source_sha256="d" * 64,
        schema_lock_sha256="e" * 64, audit={"exact": True},
        provenance=_provenance(args))
    manifest_path = root / "manifests" / f'{result["request_id"]}.json'
    assert manifest_path.exists()
    assert result["rounds"] == 2 and result["draft_positions"] == 9
    for ref in result["files"].values():
        if ref is not None:
            assert file_sha256(root / ref["path"]) == ref["sha256"]
    obs = [json.loads(line) for line in
           (root / result["files"]["observations"]["path"]).read_text().splitlines()]
    labels = [json.loads(line) for line in
              (root / result["files"]["labels"]["path"]).read_text().splitlines()]
    assert labels[0]["target_future"]["token_ids"] == [0]
    assert not any("target_future" in row or "next_round_id" in row for row in obs)
    hidden = load_file(str(root / result["files"]["hidden"]["path"]))
    assert list(hidden) == ["round_0"]
    assert result["files"]["oracle_tape"]["path"].startswith("oracle_tapes/")
    with pytest.raises(FileExistsError):
        persist_request(**args, root=root, source_sha256="d" * 64,
                        schema_lock_sha256="e" * 64, audit={"exact": True},
                        provenance=_provenance(args))
