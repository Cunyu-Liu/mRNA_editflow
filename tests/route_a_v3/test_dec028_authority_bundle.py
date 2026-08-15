"""Focused fail-closed tests for the unactivated DEC028 successor bundle."""

from __future__ import annotations

import json
import shutil

import yaml


def _copy_static_bundle(validator, repo_root, target_root):
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    paths = set(validator.required_bundle_paths())
    paths.update(entry["path"] for entry in manifest["files"])
    for relative in paths:
        source = repo_root / relative
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _codes(issues):
    return {issue.code for issue in issues}


def test_dec028_pending_bundle_is_static_only_and_coherent(
    validator, repo_root, bundle_documents
):
    config, supersession, registries = bundle_documents
    assert validator._is_dec028_pending_authority_bundle(repo_root)
    assert validator.validate_dec028_pending_authority(
        repo_root, config, supersession, registries
    ) == []
    manifest = validator._load_json(repo_root, validator.REGISTRY_MANIFEST_PATH)
    assert manifest["manifest_status"] == validator.DEC028_PENDING_MANIFEST_STATUS
    assert manifest["pending_successor_amendment_decision_ids"] == ["V3-DEC-028"]


def test_dec028_pending_bundle_rejects_lock_relaxation(
    validator, repo_root, bundle_documents, tmp_path
):
    _copy_static_bundle(validator, repo_root, tmp_path)
    amendment_path = tmp_path / validator.DEC028_AMENDMENT_PATH
    amendment = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    amendment["locks"]["cuda_probe_allowed"] = True
    amendment_path.write_text(yaml.safe_dump(amendment, sort_keys=False), encoding="utf-8")
    config, supersession, registries = validator.load_bundle_documents(tmp_path)
    codes = _codes(
        validator.validate_dec028_pending_authority(
            tmp_path, config, supersession, registries
        )
    )
    assert "DEC028_STATIC_LEAF_DRIFT" in codes
    assert "DEC028_PENDING_AUTHORITY_BUNDLE" in codes


def test_dec028_pending_bundle_rejects_preallocated_runtime_event(
    validator, repo_root, bundle_documents, tmp_path
):
    _copy_static_bundle(validator, repo_root, tmp_path)
    interim_path = tmp_path / validator.A1_INTERIM_PATH
    interim = yaml.safe_load(interim_path.read_text(encoding="utf-8"))
    interim["dec028_current_disposition"]["fresh_runtime_event_id"] = "A1-EVT-061"
    interim_path.write_text(yaml.safe_dump(interim, sort_keys=False), encoding="utf-8")
    config, supersession, registries = validator.load_bundle_documents(tmp_path)
    codes = _codes(
        validator.validate_dec028_pending_authority(
            tmp_path, config, supersession, registries
        )
    )
    assert "DEC028_STATIC_LEAF_DRIFT" in codes
    assert "DEC028_PENDING_AUTHORITY_BUNDLE" in codes


def test_dec028_pending_bundle_rejects_premature_manifest_activation(
    validator, repo_root, bundle_documents, tmp_path
):
    _copy_static_bundle(validator, repo_root, tmp_path)
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pending_successor_amendment_decision_ids"] = []
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    config, supersession, registries = validator.load_bundle_documents(tmp_path)
    codes = _codes(
        validator.validate_dec028_pending_authority(
            tmp_path, config, supersession, registries
        )
    )
    assert "DEC028_PENDING_AUTHORITY_BUNDLE" in codes
