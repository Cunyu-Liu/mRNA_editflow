from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT / "scripts" / "route_a_v3" / "reconcile_gse114002_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reconcile_gse114002_manifest", MODULE_PATH
)
assert SPEC and SPEC.loader
RECONCILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECONCILE)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _gzip_payload(label: str) -> bytes:
    return gzip.compress((label + "\n").encode("utf-8"), mtime=0)


def _make_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    canonical_root = tmp_path / "gse114002_public"
    canonical_root.mkdir()

    declared_entries: list[dict[str, object]] = []
    source_bytes: dict[str, bytes] = {}
    first_name = RECONCILE.EXPECTED_CANONICAL_NAMES[0]
    stale_declared_hash = _sha256_bytes(b"historical-corrupt-download")

    for name in RECONCILE.EXPECTED_CANONICAL_NAMES:
        payload = _gzip_payload(name)
        (canonical_root / name).write_bytes(payload)
        source_bytes[name] = payload
        declared_entries.append(
            {
                "name": name,
                "bytes": len(payload),
                "sha256": (
                    stale_declared_hash if name == first_name else _sha256_bytes(payload)
                ),
                "downloaded": True,
            }
        )

    for suffix, payload in (
        (".corrupt.bak", b"not-a-valid-gzip-backup-one"),
        (".corrupt.bak2", b"not-a-valid-gzip-backup-two"),
    ):
        quarantine_name = first_name + suffix
        (canonical_root / quarantine_name).write_bytes(payload)
        source_bytes[quarantine_name] = payload

    original_manifest = canonical_root / "manifest.json"
    manifest_payload = (
        json.dumps(
            {
                "accession": "GSE114002",
                "provider": "GEO",
                "files": declared_entries,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    original_manifest.write_bytes(manifest_payload)
    source_bytes[original_manifest.name] = manifest_payload
    return canonical_root, original_manifest, source_bytes


def _assert_sources_unchanged(root: Path, source_bytes: dict[str, bytes]) -> None:
    assert {name: (root / name).read_bytes() for name in source_bytes} == source_bytes


def test_reconciles_current_gzip_and_quarantine_without_qualification(
    tmp_path: Path,
) -> None:
    canonical_root, original_manifest, source_bytes = _make_fixture(tmp_path)
    output = tmp_path / "gse114002_manifest_reconciliation.v1.json"

    manifest = RECONCILE.reconcile_gse114002_manifest(
        original_manifest_path=original_manifest,
        canonical_directory=canonical_root,
        output_path=output,
    )

    assert set(manifest) == {
        "contract_id",
        "schema_version",
        "manifest_type",
        "dataset_id",
        "reconciliation_status",
        "qualification",
        "original_manifest",
        "canonical_directory",
        "canonical_files",
        "quarantine_files",
        "summary",
    }
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["dataset_id"] == "GSE114002"
    assert manifest["reconciliation_status"] == (
        "PROVENANCE_RECONCILED_NOT_QUALIFIED"
    )
    assert manifest["qualification"] == {
        "qualified": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "license_status": "UNKNOWN_BLOCKED",
        "exposure_status": "AUDIT_PENDING",
    }
    assert manifest["original_manifest"] == {
        "path": str(original_manifest.resolve()),
        "sha256": _sha256_bytes(source_bytes["manifest.json"]),
        "bytes": len(source_bytes["manifest.json"]),
    }

    canonical_files = manifest["canonical_files"]
    assert len(canonical_files) == 10
    assert [item["name"] for item in canonical_files] == list(
        RECONCILE.EXPECTED_CANONICAL_NAMES
    )
    assert all(item["gzip_integrity"] == "PASS" for item in canonical_files)
    repaired = canonical_files[0]
    assert repaired["sha256_matches_original"] is False
    assert repaired["bytes_match_original"] is True
    assert repaired["compressed_sha256"] == _sha256_bytes(
        source_bytes[repaired["name"]]
    )
    assert repaired["original_declared_sha256"] == _sha256_bytes(
        b"historical-corrupt-download"
    )
    assert all(item["sha256_matches_original"] for item in canonical_files[1:])

    quarantine = manifest["quarantine_files"]
    assert [item["name"] for item in quarantine] == [
        RECONCILE.EXPECTED_CANONICAL_NAMES[0] + ".corrupt.bak",
        RECONCILE.EXPECTED_CANONICAL_NAMES[0] + ".corrupt.bak2",
    ]
    assert all(item["gzip_integrity"] == "FAIL" for item in quarantine)
    assert all(item["disposition"] == "QUARANTINED_NOT_CANONICAL" for item in quarantine)
    for item in quarantine:
        assert item["compressed_sha256"] == _sha256_bytes(source_bytes[item["name"]])
        assert item["canonical_name"] == RECONCILE.EXPECTED_CANONICAL_NAMES[0]

    assert manifest["summary"] == {
        "canonical_file_count": 10,
        "canonical_gzip_pass_count": 10,
        "canonical_sha256_match_original_count": 9,
        "canonical_sha256_mismatch_original_count": 1,
        "canonical_sha256_mismatch_names": [
            RECONCILE.EXPECTED_CANONICAL_NAMES[0]
        ],
        "quarantine_file_count": 2,
        "quarantine_gzip_pass_count": 0,
        "quarantine_gzip_fail_count": 2,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    _assert_sources_unchanged(canonical_root, source_bytes)


def test_same_inputs_produce_same_versioned_manifest(tmp_path: Path) -> None:
    canonical_root, original_manifest, _ = _make_fixture(tmp_path)
    first_output = tmp_path / "reconciliation.v1.first.json"
    second_output = tmp_path / "reconciliation.v1.second.json"

    first = RECONCILE.reconcile_gse114002_manifest(
        original_manifest_path=original_manifest,
        canonical_directory=canonical_root,
        output_path=first_output,
    )
    second = RECONCILE.reconcile_gse114002_manifest(
        original_manifest_path=original_manifest,
        canonical_directory=canonical_root,
        output_path=second_output,
    )

    assert first == second
    assert first_output.read_bytes() == second_output.read_bytes()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    canonical_root, original_manifest, source_bytes = _make_fixture(tmp_path)
    output = tmp_path / "already_exists.json"
    output.write_bytes(b"preserve-me")

    with pytest.raises(RECONCILE.ReconciliationError, match="refusing to overwrite"):
        RECONCILE.reconcile_gse114002_manifest(
            original_manifest_path=original_manifest,
            canonical_directory=canonical_root,
            output_path=output,
        )

    assert output.read_bytes() == b"preserve-me"
    _assert_sources_unchanged(canonical_root, source_bytes)


@pytest.mark.parametrize(
    ("original", "canonical", "output"),
    [
        ("restricted/manifest.json", "missing-canonical", "out.json"),
        ("missing-manifest.json", "sealed_external/canonical", "out.json"),
        ("missing-manifest.json", "missing-canonical", "GSE246381/out.json"),
    ],
)
def test_forbidden_paths_fail_before_any_input_read(
    tmp_path: Path,
    original: str,
    canonical: str,
    output: str,
) -> None:
    with pytest.raises(RECONCILE.ScopeViolation, match="rejected before read"):
        RECONCILE.reconcile_gse114002_manifest(
            original_manifest_path=tmp_path / original,
            canonical_directory=tmp_path / canonical,
            output_path=tmp_path / output,
        )


def test_corrupt_canonical_gzip_fails_without_output(tmp_path: Path) -> None:
    canonical_root, original_manifest, _ = _make_fixture(tmp_path)
    broken_name = RECONCILE.EXPECTED_CANONICAL_NAMES[-1]
    (canonical_root / broken_name).write_bytes(b"not-gzip")
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(
        RECONCILE.ReconciliationError, match="canonical gzip integrity failed"
    ):
        RECONCILE.reconcile_gse114002_manifest(
            original_manifest_path=original_manifest,
            canonical_directory=canonical_root,
            output_path=output,
        )

    assert not output.exists()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_canonical_set_must_be_exactly_the_ten_known_gzip_files(
    tmp_path: Path,
    mutation: str,
) -> None:
    canonical_root, original_manifest, _ = _make_fixture(tmp_path)
    if mutation == "missing":
        (canonical_root / RECONCILE.EXPECTED_CANONICAL_NAMES[-1]).unlink()
    else:
        (canonical_root / "GSM9999999_unexpected.csv.gz").write_bytes(
            _gzip_payload("unexpected")
        )
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(RECONCILE.ReconciliationError, match="canonical gzip set mismatch"):
        RECONCILE.reconcile_gse114002_manifest(
            original_manifest_path=original_manifest,
            canonical_directory=canonical_root,
            output_path=output,
        )

    assert not output.exists()


def test_cli_writes_only_aggregate_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    canonical_root, original_manifest, _ = _make_fixture(tmp_path)
    output = tmp_path / "cli-output.v1.json"

    result = RECONCILE.main(
        [
            "--original-manifest",
            str(original_manifest),
            "--canonical-directory",
            str(canonical_root),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "canonical_file_count": 10,
        "dataset_id": "GSE114002",
        "output": str(output.resolve()),
        "quarantine_file_count": 2,
        "reconciliation_status": "PROVENANCE_RECONCILED_NOT_QUALIFIED",
    }
    assert output.is_file()
