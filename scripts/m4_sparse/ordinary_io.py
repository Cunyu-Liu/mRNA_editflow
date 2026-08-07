"""Fail-closed JSONL I/O for ordinary-only R0--R2 workflows."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


FORBIDDEN_NAMESPACE_TERMS = ("sealed", "restricted", "gse246381")
IDENTITY_FIELDS = (
    "study", "asset_id", "accession", "dataset_id", "benchmark",
    "source_accession", "object_id",
)
ROLE_FIELDS = (
    "split", "partition", "data_role", "access_role", "future_use_role",
    "evaluation_role", "policy_role",
)

_PROTECTED_VALUE_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:sealed|restricted)|gse246381",
    flags=re.IGNORECASE,
)


def _contains_forbidden(value: object) -> bool:
    """Match protected namespaces without treating ``unsealed`` as sealed.

    Ordinary records are allowed to carry ordinary/negative boolean flags such
    as ``is_sealed=false``.  Protected role tokens, accessions and namespace
    references are rejected wherever they occur, including nested metadata.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return False
    return bool(_PROTECTED_VALUE_RE.search(str(value).strip()))


def _protected_path_component(value: object) -> bool:
    text = str(value or "").strip().lower()
    if "gse246381" in text:
        return True
    return text.startswith("sealed") or text.startswith("restricted")


def assert_ordinary_path(path: Path, purpose: str = "input") -> None:
    """Reject raw and resolved paths advertising a protected namespace."""
    path = Path(path)
    candidates = [path]
    try:
        candidates.append(path.resolve(strict=False))
    except OSError:
        pass
    bad = sorted({part for candidate in candidates for part in candidate.parts
                  if _protected_path_component(part)})
    if bad:
        raise ValueError(
            "ordinary-only %s refuses protected path component(s): %s"
            % (purpose, ", ".join(bad))
        )


def _walk_fields(value: object, path: str = "$") -> Iterable[Tuple[str, object]]:
    """Yield every nested key and scalar value with a deterministic JSON path."""
    if isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            child_path = "%s.%s" % (path, key)
            child = value[key]
            # A truthy protected key can otherwise hide behind a boolean value
            # (for example ``{"sealed": true}``).  Explicit false/None flags
            # remain admissible and are checked by the scalar branch below.
            if child not in (False, None, ""):
                yield child_path + "::<field_name>", key
            yield from _walk_fields(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_fields(child, "%s[%d]" % (path, index))
    else:
        yield path, value


def protected_row_reason(row: dict) -> Optional[str]:
    """Return the first protected identity/role reason, checking every field.

    The named identity/role checks provide a concise reason for common schema
    fields.  The recursive pass is the fail-closed backstop for nested or newly
    introduced metadata fields, so adding a schema alias cannot bypass the
    ordinary-only boundary.
    """
    if not isinstance(row, dict):
        return "non_object_row"
    for key in IDENTITY_FIELDS:
        value = row.get(key)
        if _contains_forbidden(value):
            return "protected_identity:%s=%s" % (key, value)
    for key in ROLE_FIELDS:
        value = row.get(key)
        if _contains_forbidden(value):
            return "protected_role:%s=%s" % (key, value)
    if row.get("is_sealed") is True or row.get("is_restricted") is True:
        return "protected_boolean_flag"
    for field_path, value in _walk_fields(row):
        if _contains_forbidden(value):
            if field_path.endswith("::<field_name>"):
                return "protected_field_name:%s" % field_path.split("::", 1)[0]
            return "protected_field_value:%s=%s" % (field_path, value)
    return None


def assert_ordinary_rows(rows: Iterable[dict], purpose: str = "input") -> None:
    """Enforce the row boundary at in-memory runner entrypoints as well."""
    for index, row in enumerate(rows):
        reason = protected_row_reason(row)
        if reason is not None:
            raise ValueError(
                "ordinary-only %s refused row %d: %s" % (purpose, index, reason)
            )


def load_ordinary_jsonl(path: Path, require_delta: bool = True) -> List[dict]:
    """Load an ordinary JSONL only after path and per-row boundary checks."""
    path = Path(path)
    assert_ordinary_path(path, purpose="input")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("ordinary JSONL row %d is not an object" % line_number)
            reason = protected_row_reason(row)
            if reason is not None:
                raise ValueError(
                    "ordinary-only input refused row %d: %s"
                    % (line_number, reason)
                )
            if require_delta and row.get("delta") is None:
                continue
            rows.append(row)
    return rows
