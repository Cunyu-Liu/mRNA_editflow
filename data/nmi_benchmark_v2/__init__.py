"""Fail-closed loader and provenance contract for Benchmark v2.

The canonical record store is immutable after construction.  Role manifests
are lightweight indexes, so test labels are never copied into training views.
Final-test roles require an explicit ``allow_final_labels=True`` opt-in.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional

FINAL_ROLES = frozenset({
    "test_id", "test_family", "test_context", "test_assay", "test_ood",
})
ALL_ROLES = frozenset({"train", "val"}) | FINAL_ROLES
REQUIRED_SOURCE_MATCHED_FIELDS = (
    "source_id", "candidate_id", "source_sequence", "candidate_sequence",
    "edit_list", "edit_count", "measured_source", "measured_candidate",
    "measured_delta", "cargo", "cell_context", "assay", "batch", "replicate",
)


class FinalTestAccessError(PermissionError):
    """Raised when a final test manifest is opened without explicit consent."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: str | Path, *, allow_final_labels: bool = False) -> Dict:
    p = Path(path)
    obj = json.loads(p.read_text())
    role = str(obj.get("role", ""))
    if role not in ALL_ROLES:
        raise ValueError(f"unknown benchmark v2 role: {role!r}")
    if bool(obj.get("final_test", role in FINAL_ROLES)) and not allow_final_labels:
        raise FinalTestAccessError(
            f"{role} is final-test data; pass allow_final_labels=True only after freeze"
        )
    return obj


def iter_role_records(
    manifest_path: str | Path,
    *,
    allow_final_labels: bool = False,
    task_kind: Optional[str] = None,
) -> Iterator[Dict]:
    manifest_file = Path(manifest_path)
    manifest = load_manifest(manifest_file, allow_final_labels=allow_final_labels)
    root = manifest_file.parent.parent
    records_path = root / str(manifest["records_path"])
    index_path = root / str(manifest["index_path"])
    wanted = {
        line.strip() for line in index_path.read_text().splitlines() if line.strip()
    }
    if not wanted:
        return
    with records_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec.get("record_id")) in wanted:
                if task_kind is not None and str(rec.get("task_kind")) != task_kind:
                    continue
                yield rec


def manifest_sha256(path: str | Path) -> str:
    return _sha256(Path(path))


__all__ = [
    "ALL_ROLES", "FINAL_ROLES", "FinalTestAccessError", "load_manifest",
    "REQUIRED_SOURCE_MATCHED_FIELDS", "iter_role_records", "manifest_sha256",
]
