"""Durable, no-overwrite artifact publication for ordinary-only R3X runs."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import errno
from pathlib import Path
from typing import Callable, Mapping

from .protocol import ProtocolHoldError, assert_ordinary_path


class ArtifactIOError(RuntimeError):
    pass


def canonical_json_bytes(payload: Mapping) -> bytes:
    try:
        return (
            json.dumps(
                dict(payload), sort_keys=True, indent=2,
                ensure_ascii=False, allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactIOError(f"artifact is not strict JSON: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_exact_regular_file(path: Path, expected_sha256: str) -> bytes:
    """Read one immutable regular single-link file and verify its exact bytes."""

    path = Path(path)
    try:
        assert_ordinary_path(path, purpose="R3X immutable artifact input")
    except ProtocolHoldError as exc:
        raise ArtifactIOError(str(exc)) from exc
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise ArtifactIOError(f"artifact input does not exist: {path}") from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise ArtifactIOError("artifact input cannot be a symlink") from exc
        raise ArtifactIOError(f"artifact input cannot be opened safely: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ArtifactIOError("artifact input must be a regular single-link file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            raw = handle.read()
            after = os.fstat(handle.fileno())
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or after.st_nlink != 1
            or before.st_size != after.st_size
        ):
            raise ArtifactIOError("artifact input metadata changed during exact read")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != str(expected_sha256).lower():
        raise ArtifactIOError(
            f"artifact SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return raw


def atomic_publish_bytes(
    path: Path,
    raw: bytes,
    *,
    replay_validator: Callable[[bytes], None] | None = None,
) -> str:
    """Publish fully materialized bytes atomically without overwriting a name.

    The final pathname is created with a hard-link no-replace operation only
    after the staging file has been completely written and fsynced.  The
    staging link is then removed, leaving the final artifact with link count 1.
    """

    path = Path(path)
    if not isinstance(raw, bytes) or not raw:
        raise ArtifactIOError("artifact bytes must be non-empty")
    try:
        assert_ordinary_path(path, purpose="R3X immutable artifact output")
    except ProtocolHoldError as exc:
        raise ArtifactIOError(str(exc)) from exc
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ArtifactIOError("artifact parent must be an existing real directory")
    if path.exists() or path.is_symlink():
        raise ArtifactIOError(f"artifact output already exists: {path}")
    if replay_validator is not None:
        replay_validator(raw)

    descriptor = -1
    staging: Path | None = None
    published = False
    try:
        descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".staging", dir=parent
        )
        staging = Path(staging_name)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if staging.is_symlink() or not stat.S_ISREG(staging.stat().st_mode):
            raise ArtifactIOError("staging artifact is not a regular file")
        try:
            os.link(staging, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ArtifactIOError(f"artifact output already exists: {path}") from exc
        published = True
        _fsync_directory(parent)
        staging.unlink()
        staging = None
        _fsync_directory(parent)
        digest = hashlib.sha256(raw).hexdigest()
        replay = read_exact_regular_file(path, digest)
        if replay != raw:
            raise ArtifactIOError("published artifact bytes changed during replay")
        if replay_validator is not None:
            replay_validator(replay)
        return digest
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if staging is not None:
            try:
                staging.unlink()
                _fsync_directory(parent)
            except FileNotFoundError:
                pass
        # A failure after the atomic link is fail-closed: never delete or
        # overwrite the visible final name automatically.
        if published:
            pass
        raise


def atomic_write_json_exclusive(
    path: Path,
    payload: Mapping,
    *,
    validator: Callable[[Mapping], object] | None = None,
) -> str:
    if validator is not None:
        validator(payload)
    raw = canonical_json_bytes(payload)

    def replay(data: bytes) -> None:
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactIOError(f"published JSON cannot be replayed: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ArtifactIOError("published JSON artifact must be an object")
        if validator is not None:
            validator(parsed)

    return atomic_publish_bytes(path, raw, replay_validator=replay)
