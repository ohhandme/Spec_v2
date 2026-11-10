import json

from feedback_codec.v12_boundary_reader import BOUNDARY_FIELDS, load_boundary_inputs
from feedback_codec.v12_storage import persist_request
from test_v12_round_assembly import _sample
from test_v12_storage import _provenance


def test_online_reader_never_needs_future_labels_or_oracle(tmp_path):
    args = _sample()
    args.pop("hidden_ref")
    args.pop("oracle_ref")
    root = tmp_path / "train"
    result = persist_request(
        **args, root=root, source_sha256="d" * 64,
        schema_lock_sha256="e" * 64,
        audit={"exact": True}, provenance=_provenance(args))
    manifest = root / "manifests" / f'{result["request_id"]}.json'
    original = load_boundary_inputs(root, manifest)
    assert len(original) == 2
    assert all(set(row) == BOUNDARY_FIELDS for row in original)
    # The online reader's output stays byte-identical with those directories
    # inaccessible, showing that it does not read future targets by accident.
    (root / result["files"]["labels"]["path"]).unlink()
    (root / result["files"]["oracle_tape"]["path"]).unlink()
    assert load_boundary_inputs(root, manifest) == original
    assert "target_future" not in json.dumps(original)
