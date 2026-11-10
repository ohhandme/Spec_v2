"""Small CPU checks for the real-model P0 provenance and reproducibility gates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from feedback_codec.decoding import TargetStep
from scripts.run_real_p0 import verify_greedy_reproducibility, verify_tokenizer_provenance


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tokenizer_fixture(tmp_path: Path):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    for directory in (target, draft):
        directory.mkdir()
        (directory / "tokenizer.json").write_bytes(b"shared-token-ids")
        (directory / "tokenizer_config.json").write_bytes(b'{"chat_template":"locked"}')
    metadata = {
        "tokenizer_json_sha256": _sha(b"shared-token-ids"),
        "tokenizer_config_sha256": _sha(b'{"chat_template":"locked"}'),
        "chat_template_sha256": _sha(b"locked"),
    }
    return target, draft, metadata


def test_p0_accepts_matching_target_draft_prompt_tokenizers(tmp_path):
    target, draft, metadata = _tokenizer_fixture(tmp_path)
    hashes = verify_tokenizer_provenance(
        target, draft, SimpleNamespace(chat_template="locked"),
        SimpleNamespace(chat_template="locked"), metadata,
    )
    assert hashes["target_chat_template_sha256"] == metadata["chat_template_sha256"]
    assert hashes["draft_tokenizer_config.json_sha256"] == metadata["tokenizer_config_sha256"]


@pytest.mark.parametrize("role", ["target", "draft"])
def test_p0_rejects_tokenizer_config_drift(tmp_path, role):
    target, draft, metadata = _tokenizer_fixture(tmp_path)
    path = target if role == "target" else draft
    (path / "tokenizer_config.json").write_bytes(b'{"chat_template":"changed"}')
    with pytest.raises(RuntimeError, match=f"{role} tokenizer_config.json"):
        verify_tokenizer_provenance(
            target, draft, SimpleNamespace(chat_template="locked"),
            SimpleNamespace(chat_template="locked"), metadata,
        )


@pytest.mark.parametrize("role", ["target", "draft"])
def test_p0_rejects_live_chat_template_drift(tmp_path, role):
    target, draft, metadata = _tokenizer_fixture(tmp_path)
    target_template = "changed" if role == "target" else "locked"
    draft_template = "changed" if role == "draft" else "locked"
    with pytest.raises(RuntimeError, match=f"{role} chat template"):
        verify_tokenizer_provenance(
            target, draft, SimpleNamespace(chat_template=target_template),
            SimpleNamespace(chat_template=draft_template), metadata,
        )


class _GreedyTarget:
    def __init__(self, unstable: bool):
        self.calls = 0
        self.unstable = unstable

    def next(self, prefix):
        self.calls += 1
        token = 4 if self.unstable else 3
        return TargetStep(token_id=token, hidden=None)


def test_p0_accepts_repeated_identical_ar():
    verify_greedy_reproducibility(_GreedyTarget(False), [1], (3,), 1, 2)


def test_p0_rejects_nonreproducible_ar():
    with pytest.raises(RuntimeError, match="not reproducible"):
        verify_greedy_reproducibility(_GreedyTarget(True), [1], (3,), 1, 2)
