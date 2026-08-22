"""Fail closed on the superseded repository-root B0 split manifests."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_B0_SPLITS_DIR = (REPO_ROOT / "data/b0_splits").resolve()


class LegacyB0SplitLoadError(RuntimeError):
    """An active loader attempted to read superseded B0 split manifests."""


def reject_legacy_b0_splits(path: str | Path) -> Path:
    """Return a resolved non-legacy path or reject the superseded B0 root."""
    resolved = Path(path).resolve()
    if resolved == LEGACY_B0_SPLITS_DIR:
        raise LegacyB0SplitLoadError(
            "SUPERSEDED_NOT_LOADABLE: data/b0_splits is preserved as legacy "
            "evidence and cannot be used by an active loader; use the versioned "
            "V3.1 split registry/assignments instead"
        )
    return resolved
