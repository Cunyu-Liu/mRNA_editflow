"""Fail-closed FM0 checkpoint-byte provenance for the MK0 GPU path."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from mrna_editflow.core.mk0.acceptance import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GPU = _load("mk0_snapshot_gpu_test", "scripts/mk0/run_mk0_gpu_smoke.py")
FINALIZER = _load(
    "mk0_snapshot_finalizer_test", "scripts/mk0/finalize_mk0_acceptance.py"
)


def _fixture(tmp_path: Path):
    revision = "a" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"utrlm"}\n', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"frozen-foundation-weights\n")
    (snapshot / "license.md").write_text("AGPL-3.0 test fixture\n", encoding="utf-8")

    files = []
    for path in sorted(snapshot.iterdir()):
        files.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    fm0_root = tmp_path / "fm0"
    manifest_path = fm0_root / "evaluation" / "hash_license_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "task_id": "FM0-01",
        "manifest_kind": "foundation_checkpoint_hash_license",
        "model_id": "example/utrlm",
        "revision": revision,
        "snapshot_dir": str(snapshot),
        "expected_files": [record["filename"] for record in files],
        "missing_files": [],
        "files": files,
        "license": {
            "type": "agpl-3.0",
            "license_md_sha256": sha256_file(snapshot / "license.md"),
            "license_md_size": (snapshot / "license.md").stat().st_size,
        },
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    ledger = fm0_root / "artifact_checksums.sha256"
    ledger.write_text(
        f"{sha256_file(manifest_path)}  ./evaluation/hash_license_manifest.json\n",
        encoding="utf-8",
    )
    config = {
        "model": {"model_id": "example/utrlm", "revision": revision},
        "storage": {"snapshot_dir": str(snapshot)},
    }
    preflight = {
        "upstream": {
            "fm0_closure_root": str(fm0_root),
            "fm0_checksum_ledger_sha256": sha256_file(ledger),
        }
    }
    return snapshot, config, preflight, manifest_path, ledger


def test_gpu_and_finalizer_bind_exact_fm0_snapshot_bytes(tmp_path: Path) -> None:
    snapshot, config, preflight, manifest_path, ledger = _fixture(tmp_path)
    binding = GPU._validate_snapshot_binding(snapshot, config, preflight)
    binding["post_model_load_rehash_match"] = True
    expected_files = binding["files"]
    fm0 = {
        "checksum_ledger": {"sha256": sha256_file(ledger)},
        "foundation_checkpoint_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "model_id": binding["model_id"],
            "revision": binding["observed_revision"],
            "snapshot_dir": str(snapshot),
            "file_count": len(expected_files),
            "files": expected_files,
            "files_sha256": hashlib.sha256(
                canonical_json_bytes(expected_files)
            ).hexdigest(),
            "license": binding["license_binding"],
        },
    }
    evidence = FINALIZER.verify_foundation_snapshot_provenance(
        {"snapshot_binding": binding}, fm0
    )
    assert evidence["runner_and_finalizer_independent_rehash_match"] is True

    weight = snapshot / "model.safetensors"
    original_weight = weight.read_bytes()
    changed_weight = bytearray(original_weight)
    changed_weight[0] ^= 1
    weight.write_bytes(changed_weight)
    with pytest.raises(GPU.SmokeFailure, match="snapshot bytes differ"):
        GPU._validate_snapshot_binding(snapshot, config, preflight)
    with pytest.raises(FINALIZER.FinalizeFailure, match="snapshot bytes differ"):
        FINALIZER.verify_foundation_snapshot_provenance(
            {"snapshot_binding": binding}, fm0
        )

    weight.write_bytes(original_weight)
    extra = snapshot / "unbound.bin"
    extra.write_bytes(b"not-in-fm0\n")
    with pytest.raises(GPU.SmokeFailure, match="snapshot bytes differ"):
        GPU._validate_snapshot_binding(snapshot, config, preflight)
    extra.unlink()

    original_manifest = manifest_path.read_bytes()
    license_tamper = json.loads(original_manifest)
    license_tamper["license"]["type"] = "unknown"
    manifest_path.write_bytes(canonical_json_bytes(license_tamper))
    ledger.write_text(
        f"{sha256_file(manifest_path)}  ./evaluation/hash_license_manifest.json\n",
        encoding="utf-8",
    )
    preflight["upstream"]["fm0_checksum_ledger_sha256"] = sha256_file(ledger)
    with pytest.raises(GPU.SmokeFailure, match="license type"):
        GPU._validate_snapshot_binding(snapshot, config, preflight)

    for field, value in (
        ("license_md_sha256", "0" * 64),
        ("license_md_size", 1),
    ):
        byte_tamper = json.loads(original_manifest)
        byte_tamper["license"][field] = value
        manifest_path.write_bytes(canonical_json_bytes(byte_tamper))
        ledger.write_text(
            f"{sha256_file(manifest_path)}  ./evaluation/hash_license_manifest.json\n",
            encoding="utf-8",
        )
        preflight["upstream"]["fm0_checksum_ledger_sha256"] = sha256_file(ledger)
        with pytest.raises(GPU.SmokeFailure, match="license semantics"):
            GPU._validate_snapshot_binding(snapshot, config, preflight)

    manifest_path.write_bytes(original_manifest)
    ledger.write_text(
        f"{sha256_file(manifest_path)}  ./evaluation/hash_license_manifest.json\n",
        encoding="utf-8",
    )
    preflight["upstream"]["fm0_checksum_ledger_sha256"] = sha256_file(ledger)
    ledger.write_text("", encoding="utf-8")
    preflight["upstream"]["fm0_checksum_ledger_sha256"] = sha256_file(ledger)
    with pytest.raises(GPU.SmokeFailure, match="omits"):
        GPU._validate_snapshot_binding(snapshot, config, preflight)
