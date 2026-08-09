#!/usr/bin/env python3
"""Create a non-destructive reconciliation manifest for GSE114002.

This utility records the current compressed bytes of the ten public canonical
GSE114002 gzip files alongside the hashes declared by the original download
manifest.  Files named ``*.corrupt.bak*`` are inventoried as quarantined
evidence; they are never promoted, renamed, deleted, or used as canonical
inputs.

Reconciliation is provenance evidence only.  The emitted manifest explicitly
keeps scientific qualification, training, and model selection disabled.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import stat
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE114002"
MANIFEST_TYPE = "GSE114002_MANIFEST_RECONCILIATION"
SCHEMA_VERSION = "1.0.0"
RECONCILIATION_STATUS = "PROVENANCE_RECONCILED_NOT_QUALIFIED"

EXPECTED_CANONICAL_NAMES: tuple[str, ...] = (
    "GSM3130435_egfp_unmod_1.csv.gz",
    "GSM3130436_egfp_unmod_2.csv.gz",
    "GSM3130437_egfp_pseudo_1.csv.gz",
    "GSM3130438_egfp_pseudo_2.csv.gz",
    "GSM3130439_egfp_m1pseudo_1.csv.gz",
    "GSM3130440_egfp_m1pseudo_2.csv.gz",
    "GSM3130441_mcherry_1.csv.gz",
    "GSM3130442_mcherry_2.csv.gz",
    "GSM3130443_designed_library.csv.gz",
    "GSM4084997_varying_length_25to100.csv.gz",
)
FORBIDDEN_PATH_TOKENS: tuple[str, ...] = ("sealed", "restricted", "gse246381")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QUARANTINE_RE = re.compile(r"^(?P<canonical>.+\.csv\.gz)\.corrupt\.bak.*$")


class ReconciliationError(RuntimeError):
    """Raised when provenance inputs cannot be reconciled safely."""


class ScopeViolation(ReconciliationError):
    """Raised before payload reads when a path enters a forbidden scope."""


def _reject_forbidden_path(path: Path | str, *, label: str) -> None:
    text = os.fspath(path).casefold()
    matches = sorted(token for token in FORBIDDEN_PATH_TOKENS if token in text)
    if matches:
        raise ScopeViolation(
            f"{label} rejected before read; forbidden path token(s): "
            + ",".join(matches)
        )


def _resolve_paths_before_read(
    original_manifest_path: Path,
    canonical_directory: Path,
    output_path: Path,
) -> tuple[Path, Path, Path]:
    raw_paths = (
        (original_manifest_path.expanduser(), "original manifest path"),
        (canonical_directory.expanduser(), "canonical directory path"),
        (output_path.expanduser(), "output path"),
    )

    # Inspect every caller-supplied string before any filesystem operation.
    for raw_path, label in raw_paths:
        _reject_forbidden_path(raw_path, label=label)

    for raw_path, label in raw_paths[:2]:
        if raw_path.is_symlink():
            raise ReconciliationError(f"{label} must not be a symlink")
    if raw_paths[2][0].is_symlink():
        raise ReconciliationError("output path must not be a symlink")

    resolved = tuple(path.resolve(strict=False) for path, _ in raw_paths)
    for resolved_path, (_, label) in zip(resolved, raw_paths):
        _reject_forbidden_path(resolved_path, label=label)

    original, canonical_root, output = resolved
    if output in {original, canonical_root}:
        raise ReconciliationError("output path must be distinct from all inputs")
    return original, canonical_root, output


def _require_regular_file(path: Path, *, label: str) -> os.stat_result:
    _reject_forbidden_path(path, label=label)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ReconciliationError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ReconciliationError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise ReconciliationError(f"{label} must be a regular file: {path}")
    return info


def _require_directory(path: Path, *, label: str) -> None:
    _reject_forbidden_path(path, label=label)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ReconciliationError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ReconciliationError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise ReconciliationError(f"{label} must be a directory: {path}")


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def _gzip_integrity_passes(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            for _ in iter(lambda: handle.read(1 << 20), b""):
                pass
    except (gzip.BadGzipFile, EOFError, OSError, zlib.error):
        return False
    return True


def _inspect_compressed_file(path: Path, *, label: str) -> dict[str, Any]:
    before = _require_regular_file(path, label=label)
    observed_sha256, observed_bytes = _sha256_and_size(path)
    integrity_passes = _gzip_integrity_passes(path)
    after = _require_regular_file(path, label=label)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or observed_bytes != after.st_size:
        raise ReconciliationError(f"{label} changed during reconciliation: {path}")
    return {
        "compressed_sha256": observed_sha256,
        "compressed_bytes": observed_bytes,
        "gzip_integrity": "PASS" if integrity_passes else "FAIL",
    }


def _declared_bytes(entry: Mapping[str, Any]) -> int:
    for key in ("bytes", "byte_size", "size_bytes"):
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    raise ReconciliationError("original manifest entry lacks a valid declared byte count")


def _load_original_manifest(path: Path) -> tuple[dict[str, Any], str, int]:
    _require_regular_file(path, label="original manifest")
    raw = path.read_bytes()
    original_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconciliationError("original manifest is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ReconciliationError("original manifest root must be an object")
    if document.get("accession") != DATASET_ID:
        raise ReconciliationError(f"original manifest accession must be {DATASET_ID}")

    entries = document.get("files")
    if entries is None:
        entries = document.get("samples")
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_CANONICAL_NAMES):
        raise ReconciliationError("original manifest must declare exactly 10 canonical files")

    declared: dict[str, Any] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise ReconciliationError("original manifest file entries must be objects")
        name = raw_entry.get("name") or raw_entry.get("filename")
        if not isinstance(name, str) or not name:
            raise ReconciliationError("original manifest entry lacks a filename")
        _reject_forbidden_path(name, label="original manifest filename")
        if Path(name).is_absolute() or Path(name).name != name or ".." in Path(name).parts:
            raise ReconciliationError("original manifest filenames must be safe basenames")
        if name in declared:
            raise ReconciliationError(f"duplicate original manifest filename: {name}")
        declared_sha256 = raw_entry.get("sha256")
        if not isinstance(declared_sha256, str) or not SHA256_RE.fullmatch(declared_sha256):
            raise ReconciliationError(f"invalid original declared SHA-256 for {name}")
        declared[name] = {
            "sha256": declared_sha256,
            "bytes": _declared_bytes(raw_entry),
        }

    if set(declared) != set(EXPECTED_CANONICAL_NAMES):
        missing = sorted(set(EXPECTED_CANONICAL_NAMES) - set(declared))
        unexpected = sorted(set(declared) - set(EXPECTED_CANONICAL_NAMES))
        raise ReconciliationError(
            f"original manifest canonical filename set mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    return declared, original_sha256, len(raw)


def _inventory_directory(
    canonical_root: Path,
) -> tuple[dict[str, Path], list[tuple[str, Path]]]:
    _require_directory(canonical_root, label="canonical directory")
    children = sorted(canonical_root.iterdir(), key=lambda item: item.name)

    # Reject any sensitive-looking child name before opening or hashing it.
    for child in children:
        _reject_forbidden_path(child, label="canonical directory entry")

    canonical = {
        child.name: child
        for child in children
        if child.name.endswith(".csv.gz")
    }
    if set(canonical) != set(EXPECTED_CANONICAL_NAMES):
        missing = sorted(set(EXPECTED_CANONICAL_NAMES) - set(canonical))
        unexpected = sorted(set(canonical) - set(EXPECTED_CANONICAL_NAMES))
        raise ReconciliationError(
            f"current canonical gzip set mismatch; missing={missing}, unexpected={unexpected}"
        )

    quarantined: list[tuple[str, Path]] = []
    for child in children:
        match = QUARANTINE_RE.fullmatch(child.name)
        if not match:
            continue
        canonical_name = match.group("canonical")
        if canonical_name not in canonical:
            raise ReconciliationError(
                f"quarantine file has no canonical counterpart: {child.name}"
            )
        quarantined.append((canonical_name, child))
    return canonical, quarantined


def _closed_qualification() -> dict[str, Any]:
    return {
        "qualified": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "license_status": "UNKNOWN_BLOCKED",
        "exposure_status": "AUDIT_PENDING",
    }


def reconcile_gse114002_manifest(
    *,
    original_manifest_path: Path,
    canonical_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Reconcile current bytes and create ``output_path`` without overwriting.

    All three caller-supplied paths are scope-checked before any input read.
    The output path must not already exist.  Canonical or original inputs are
    only read; they are never renamed, removed, or modified.
    """

    original, canonical_root, output = _resolve_paths_before_read(
        Path(original_manifest_path), Path(canonical_directory), Path(output_path)
    )
    if output.exists():
        raise ReconciliationError(f"refusing to overwrite existing output: {output}")
    _require_directory(output.parent, label="output parent directory")

    declared, original_sha256, original_bytes = _load_original_manifest(original)
    canonical_paths, quarantined_paths = _inventory_directory(canonical_root)

    canonical_files: list[dict[str, Any]] = []
    for name in EXPECTED_CANONICAL_NAMES:
        inspection = _inspect_compressed_file(
            canonical_paths[name], label=f"canonical gzip {name}"
        )
        if inspection["gzip_integrity"] != "PASS":
            raise ReconciliationError(f"canonical gzip integrity failed: {name}")
        original_entry = declared[name]
        canonical_files.append(
            {
                "name": name,
                "path": str(canonical_paths[name]),
                "compressed_sha256": inspection["compressed_sha256"],
                "compressed_bytes": inspection["compressed_bytes"],
                "gzip_integrity": "PASS",
                "original_declared_sha256": original_entry["sha256"],
                "original_declared_bytes": original_entry["bytes"],
                "sha256_matches_original": (
                    inspection["compressed_sha256"] == original_entry["sha256"]
                ),
                "bytes_match_original": (
                    inspection["compressed_bytes"] == original_entry["bytes"]
                ),
            }
        )

    quarantine_files: list[dict[str, Any]] = []
    for canonical_name, quarantine_path in quarantined_paths:
        inspection = _inspect_compressed_file(
            quarantine_path, label=f"quarantine gzip {quarantine_path.name}"
        )
        quarantine_files.append(
            {
                "name": quarantine_path.name,
                "path": str(quarantine_path),
                "canonical_name": canonical_name,
                "compressed_sha256": inspection["compressed_sha256"],
                "compressed_bytes": inspection["compressed_bytes"],
                "gzip_integrity": inspection["gzip_integrity"],
                "disposition": "QUARANTINED_NOT_CANONICAL",
            }
        )

    mismatch_names = [
        item["name"] for item in canonical_files if not item["sha256_matches_original"]
    ]
    manifest: dict[str, Any] = {
        "contract_id": CONTRACT_ID,
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "dataset_id": DATASET_ID,
        "reconciliation_status": RECONCILIATION_STATUS,
        "qualification": _closed_qualification(),
        "original_manifest": {
            "path": str(original),
            "sha256": original_sha256,
            "bytes": original_bytes,
        },
        "canonical_directory": str(canonical_root),
        "canonical_files": canonical_files,
        "quarantine_files": quarantine_files,
        "summary": {
            "canonical_file_count": len(canonical_files),
            "canonical_gzip_pass_count": len(canonical_files),
            "canonical_sha256_match_original_count": (
                len(canonical_files) - len(mismatch_names)
            ),
            "canonical_sha256_mismatch_original_count": len(mismatch_names),
            "canonical_sha256_mismatch_names": mismatch_names,
            "quarantine_file_count": len(quarantine_files),
            "quarantine_gzip_pass_count": sum(
                item["gzip_integrity"] == "PASS" for item in quarantine_files
            ),
            "quarantine_gzip_fail_count": sum(
                item["gzip_integrity"] == "FAIL" for item in quarantine_files
            ),
        },
    }

    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ReconciliationError(f"refusing to overwrite existing output: {output}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Only a just-created incomplete output is removed; source assets and
        # the original manifest are never mutated.
        output.unlink(missing_ok=True)
        raise
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-manifest", required=True, type=Path)
    parser.add_argument("--canonical-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = reconcile_gse114002_manifest(
        original_manifest_path=args.original_manifest,
        canonical_directory=args.canonical_directory,
        output_path=args.output,
    )
    summary = {
        "dataset_id": manifest["dataset_id"],
        "output": str(args.output.resolve(strict=False)),
        "reconciliation_status": manifest["reconciliation_status"],
        "canonical_file_count": manifest["summary"]["canonical_file_count"],
        "quarantine_file_count": manifest["summary"]["quarantine_file_count"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
