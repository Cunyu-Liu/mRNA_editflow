from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from scripts.r3x.artifact_io import (
    ArtifactIOError,
    atomic_publish_bytes,
    atomic_write_json_exclusive,
    read_exact_regular_file,
)


def test_atomic_publish_is_exact_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    digest = atomic_publish_bytes(path, b"complete-bytes")
    assert digest == hashlib.sha256(b"complete-bytes").hexdigest()
    assert read_exact_regular_file(path, digest) == b"complete-bytes"
    assert path.stat().st_nlink == 1
    with pytest.raises(ArtifactIOError, match="already exists"):
        atomic_publish_bytes(path, b"replacement")
    assert path.read_bytes() == b"complete-bytes"


def test_validation_failure_never_creates_final_name(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"

    def reject(_raw: bytes) -> None:
        raise ArtifactIOError("fixture rejection")

    with pytest.raises(ArtifactIOError, match="fixture rejection"):
        atomic_publish_bytes(path, b"not-published", replay_validator=reject)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_json_replays_schema_before_publish(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"

    def validate(payload) -> None:
        if set(payload) != {"schema_version", "value"}:
            raise ArtifactIOError("unexpected schema")

    digest = atomic_write_json_exclusive(
        path, {"schema_version": "fixture_v1", "value": 1}, validator=validate
    )
    assert read_exact_regular_file(path, digest).endswith(b"\n")
    with pytest.raises(ArtifactIOError, match="unexpected schema"):
        atomic_write_json_exclusive(
            tmp_path / "bad.json",
            {"schema_version": "fixture_v1", "value": 1, "extra": True},
            validator=validate,
        )
    assert not (tmp_path / "bad.json").exists()


def test_reader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"bytes")
    digest = hashlib.sha256(b"bytes").hexdigest()
    symlink = tmp_path / "symlink.bin"
    symlink.symlink_to(source)
    with pytest.raises(ArtifactIOError, match="symlink"):
        read_exact_regular_file(symlink, digest)
    hardlink = tmp_path / "hardlink.bin"
    os.link(source, hardlink)
    with pytest.raises(ArtifactIOError, match="single-link"):
        read_exact_regular_file(source, digest)
