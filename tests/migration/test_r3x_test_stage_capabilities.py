"""Adversarial tests for feature-only and outcome-only stage capabilities."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from types import MappingProxyType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import scripts.r3x.test_stage_capabilities as stage_module  # noqa: E402
from scripts.r3x.test_stage_capabilities import (  # noqa: E402
    FEATURE_AUTHORITY_RELATIVE_PATH,
    FEATURE_MANIFEST_RELATIVE_PATH,
    FEATURE_OS_VERIFIER_ID,
    FEATURE_PROJECTION_RELATIVE_PATHS,
    FEATURE_SEMANTIC_VERIFIER_ID,
    FEATURE_STAGE_ROLE,
    FEATURE_STAGE_SCHEMA,
    OUTCOME_AUTHORITY_RELATIVE_PATH,
    OUTCOME_CLAIM_POLICY,
    OUTCOME_CLAIM_RELATIVE_PATH,
    OUTCOME_MANIFEST_RELATIVE_PATH,
    OUTCOME_OS_VERIFIER_ID,
    OUTCOME_SEMANTIC_VERIFIER_ID,
    OUTCOME_STAGE_ROLE,
    OUTCOME_STAGE_SCHEMA,
    SEMANTIC_TEST_PREDICTION_RELATIVE_PATH,
    TYPED_PRE_TEST_FREEZE_RELATIVE_PATH,
    TestStageCapabilityError as StageCapabilityError,
    canonical_feature_stage_manifest_bytes,
    canonical_outcome_stage_manifest_bytes,
    load_feature_stage_manifest_exact,
    load_outcome_stage_manifest_exact,
    read_internal_test_features_with_capability,
    release_ordinary_outcome_once,
    validate_feature_stage_manifest,
    validate_outcome_stage_manifest,
    verify_feature_stage_capability_runtime,
    verify_outcome_stage_capability_runtime,
    write_feature_stage_manifest_exclusive,
    write_outcome_stage_manifest_exclusive,
)
from scripts.r3x.stage_row_projections import (  # noqa: E402
    CONTEXT_SOURCE_CROSSWALK,
    INCLUSION_MANIFEST,
    SOURCE_COMPONENT_SPLIT_MANIFEST,
)


ROW_AUTHORITIES = (
    INCLUSION_MANIFEST,
    CONTEXT_SOURCE_CROSSWALK,
    SOURCE_COMPONENT_SPLIT_MANIFEST,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "ordinary_run"
    (root / FEATURE_MANIFEST_RELATIVE_PATH.parent).mkdir(parents=True)
    (root / OUTCOME_MANIFEST_RELATIVE_PATH.parent).mkdir(parents=True)
    (root / "commits").mkdir(parents=True)
    feature_bytes = b'{"row_id":"feature_fixture"}\n'
    outcome_bytes = b'{"row_id":"feature_fixture","delta":0.25}\n'
    (root / FEATURE_AUTHORITY_RELATIVE_PATH).write_bytes(feature_bytes)
    (root / OUTCOME_AUTHORITY_RELATIVE_PATH).write_bytes(outcome_bytes)
    return root


def _boundaries() -> dict:
    return {
        "sign_accuracy_gate": 0.60,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }


def _ref(path: str, label: str) -> dict:
    return {"relative_path": path, "sha256": _sha(label)}


def _feature_manifest(root: Path) -> dict:
    feature_sha = hashlib.sha256((root / FEATURE_AUTHORITY_RELATIVE_PATH).read_bytes()).hexdigest()
    return {
        "schema_version": FEATURE_STAGE_SCHEMA,
        "artifact_role": FEATURE_STAGE_ROLE,
        "status": "EXACT_MANIFEST_PREPARED_CAPABILITY_NOT_READY",
        "semantic_verifier_id": FEATURE_SEMANTIC_VERIFIER_ID,
        "semantic_verifier_status": "STRICT_SEMANTIC_VERIFIER_REGISTRY_REPLAY_REQUIRED",
        "os_capability_verifier_id": FEATURE_OS_VERIFIER_ID,
        "os_capability_status": "OS_MOUNT_CAPABILITY_ALLOWLIST_REPLAY_REQUIRED",
        "namespace_integrity_status": (
            "BLOCKED_PARENT_DIRECTORY_NOT_DESCRIPTOR_PINNED_"
            "PRODUCTION_RUNTIME_DISABLED"
        ),
        "capability_ready": False,
        "typed_pre_test_freeze_commit_ref": _ref(
            TYPED_PRE_TEST_FREEZE_RELATIVE_PATH,
            "typed-semantic-freeze",
        ),
        "feature_authority_ref": {
            "relative_path": FEATURE_AUTHORITY_RELATIVE_PATH,
            "sha256": feature_sha,
        },
        "feature_row_projection_refs": {
            name: _ref(FEATURE_PROJECTION_RELATIVE_PATHS[name], f"feature-projection:{name}")
            for name in ROW_AUTHORITIES
        },
        "feature_row_set_sha256": _sha("feature-row-set"),
        "boundaries": _boundaries(),
    }


def _outcome_manifest(root: Path) -> dict:
    outcome_sha = hashlib.sha256((root / OUTCOME_AUTHORITY_RELATIVE_PATH).read_bytes()).hexdigest()
    return {
        "schema_version": OUTCOME_STAGE_SCHEMA,
        "artifact_role": OUTCOME_STAGE_ROLE,
        "status": "EXACT_MANIFEST_PREPARED_CAPABILITY_NOT_READY",
        "semantic_verifier_id": OUTCOME_SEMANTIC_VERIFIER_ID,
        "semantic_verifier_status": "STRICT_SEMANTIC_VERIFIER_REGISTRY_REPLAY_REQUIRED",
        "os_capability_verifier_id": OUTCOME_OS_VERIFIER_ID,
        "os_capability_status": "OS_MOUNT_CAPABILITY_ALLOWLIST_REPLAY_REQUIRED",
        "namespace_integrity_status": (
            "BLOCKED_PARENT_DIRECTORY_NOT_DESCRIPTOR_PINNED_"
            "PRODUCTION_RUNTIME_DISABLED"
        ),
        "capability_ready": False,
        "semantic_test_prediction_commit_ref": _ref(
            SEMANTIC_TEST_PREDICTION_RELATIVE_PATH,
            "semantic-test-prediction",
        ),
        "outcome_authority_ref": {
            "relative_path": OUTCOME_AUTHORITY_RELATIVE_PATH,
            "sha256": outcome_sha,
        },
        "outcome_row_set_sha256": _sha("outcome-row-set"),
        "claim_relative_path": OUTCOME_CLAIM_RELATIVE_PATH,
        "claim_policy": OUTCOME_CLAIM_POLICY,
        "boundaries": _boundaries(),
    }


def _write_feature(root: Path) -> tuple[dict, Path, str]:
    payload = _feature_manifest(root)
    path = root / FEATURE_MANIFEST_RELATIVE_PATH
    digest = write_feature_stage_manifest_exclusive(root, path, payload)
    return payload, path, digest


def _write_outcome(root: Path) -> tuple[dict, Path, str]:
    payload = _outcome_manifest(root)
    path = root / OUTCOME_MANIFEST_RELATIVE_PATH
    digest = write_outcome_stage_manifest_exclusive(root, path, payload)
    return payload, path, digest


def _assert_hold(callable_, code: str) -> StageCapabilityError:
    with pytest.raises(StageCapabilityError) as caught:
        callable_()
    assert caught.value.code == code
    return caught.value


def _install_feature_verifiers(monkeypatch: pytest.MonkeyPatch, manifest: dict) -> None:
    semantic = {
        "schema_version": "r3x_strict_typed_pre_test_freeze_replay_evidence_v1",
        "verified_commit_sha256": manifest["typed_pre_test_freeze_commit_ref"]["sha256"],
        "authorized_feature_authority_sha256": manifest["feature_authority_ref"]["sha256"],
        "authorized_projection_sha256": {
            name: manifest["feature_row_projection_refs"][name]["sha256"]
            for name in ROW_AUTHORITIES
        },
        "authorized_row_set_sha256": manifest["feature_row_set_sha256"],
        "semantic_replay_passed": True,
        "outcome_refs_absent": True,
        "train_refs_absent": True,
        "mutable_selector_checkpoint_fields_absent": True,
    }
    os_evidence = {
        "schema_version": "r3x_feature_process_mount_allowlist_evidence_v1",
        "feature_authority_mounted": True,
        "train_authority_mounted": False,
        "outcome_authority_mounted": False,
        "sealed_authority_mounted": False,
        "mount_allowlist_replay_passed": True,
    }
    monkeypatch.setattr(
        stage_module,
        "_FEATURE_SEMANTIC_VERIFIER_REGISTRY",
        MappingProxyType({FEATURE_SEMANTIC_VERIFIER_ID: lambda _root, _ref: semantic}),
    )
    monkeypatch.setattr(
        stage_module,
        "_FEATURE_OS_VERIFIER_REGISTRY",
        MappingProxyType({FEATURE_OS_VERIFIER_ID: lambda _root: os_evidence}),
    )


def _install_outcome_verifiers(monkeypatch: pytest.MonkeyPatch, manifest: dict) -> None:
    semantic = {
        "schema_version": "r3x_strict_semantic_test_prediction_replay_evidence_v1",
        "verified_commit_sha256": manifest["semantic_test_prediction_commit_ref"]["sha256"],
        "authorized_outcome_authority_sha256": manifest["outcome_authority_ref"]["sha256"],
        "authorized_row_set_sha256": manifest["outcome_row_set_sha256"],
        "semantic_replay_passed": True,
        "predictions_frozen_semantically": True,
        "selector_checkpoint_identity_bound": True,
        "outcome_not_previously_accessed": True,
    }
    os_evidence = {
        "schema_version": "r3x_outcome_process_mount_allowlist_evidence_v1",
        "outcome_authority_mounted": True,
        "train_authority_mounted": False,
        "feature_authority_mounted": False,
        "model_config_checkpoint_mounted": False,
        "sealed_authority_mounted": False,
        "mount_allowlist_replay_passed": True,
    }
    monkeypatch.setattr(
        stage_module,
        "_OUTCOME_SEMANTIC_VERIFIER_REGISTRY",
        MappingProxyType({OUTCOME_SEMANTIC_VERIFIER_ID: lambda _root, _ref: semantic}),
    )
    monkeypatch.setattr(
        stage_module,
        "_OUTCOME_OS_VERIFIER_REGISTRY",
        MappingProxyType({OUTCOME_OS_VERIFIER_ID: lambda _root: os_evidence}),
    )


def _fixture_feature_verify(root: Path, path: Path, digest: str):
    return stage_module._verify_feature_stage_capability_fixture(
        root,
        path,
        digest,
        _fixture_token=stage_module._FIXTURE_ONLY_TOKEN,
    )


def _fixture_outcome_verify(root: Path, path: Path, digest: str):
    return stage_module._verify_outcome_stage_capability_fixture(
        root,
        path,
        digest,
        _fixture_token=stage_module._FIXTURE_ONLY_TOKEN,
    )


def _fixture_feature_read(root: Path, path: Path, digest: str) -> bytes:
    return stage_module._read_internal_test_features_fixture(
        root,
        path,
        digest,
        _fixture_token=stage_module._FIXTURE_ONLY_TOKEN,
    )


def _fixture_outcome_release(
    root: Path,
    path: Path,
    digest: str,
    **kwargs,
) -> dict:
    return stage_module._release_ordinary_outcome_fixture_once(
        root,
        path,
        digest,
        _fixture_token=stage_module._FIXTURE_ONLY_TOKEN,
        **kwargs,
    )


def test_feature_manifest_is_exact_canonical_exclusive_and_not_ready(run_root: Path) -> None:
    payload, path, digest = _write_feature(run_root)
    loaded = load_feature_stage_manifest_exact(run_root, path, digest)
    assert loaded == payload
    assert path.read_bytes() == canonical_feature_stage_manifest_bytes(payload)
    assert loaded["capability_ready"] is False
    assert loaded["namespace_integrity_status"].startswith(
        "BLOCKED_PARENT_DIRECTORY_NOT_DESCRIPTOR_PINNED"
    )
    assert loaded["boundaries"] == _boundaries()
    assert set(loaded["feature_row_projection_refs"]) == set(ROW_AUTHORITIES)
    forbidden = {
        "outcome_authority_ref",
        "train_authority_ref",
        "selector_ref",
        "checkpoint_ref",
        "model_config_ref",
    }
    assert not forbidden.intersection(loaded)
    _assert_hold(
        lambda: write_feature_stage_manifest_exclusive(run_root, path, payload),
        "HOLD_TEST_STAGE_WRITE",
    )


def test_outcome_manifest_is_exact_canonical_exclusive_and_not_ready(run_root: Path) -> None:
    payload, path, digest = _write_outcome(run_root)
    loaded = load_outcome_stage_manifest_exact(run_root, path, digest)
    assert loaded == payload
    assert path.read_bytes() == canonical_outcome_stage_manifest_bytes(payload)
    assert loaded["capability_ready"] is False
    assert loaded["namespace_integrity_status"].startswith(
        "BLOCKED_PARENT_DIRECTORY_NOT_DESCRIPTOR_PINNED"
    )
    assert loaded["boundaries"] == _boundaries()
    forbidden = {
        "feature_authority_ref",
        "train_authority_ref",
        "selector_ref",
        "checkpoint_ref",
        "model_config_ref",
        "feature_projection_refs",
    }
    assert not forbidden.intersection(loaded)
    _assert_hold(
        lambda: write_outcome_stage_manifest_exclusive(run_root, path, payload),
        "HOLD_TEST_STAGE_WRITE",
    )


@pytest.mark.parametrize(
    ("factory", "extra_field"),
    [
        (_feature_manifest, "outcome_authority_ref"),
        (_feature_manifest, "selector_ref"),
        (_feature_manifest, "checkpoint_ref"),
        (_outcome_manifest, "feature_authority_ref"),
        (_outcome_manifest, "train_authority_ref"),
        (_outcome_manifest, "model_config"),
    ],
)
def test_extra_or_cross_stage_fields_are_rejected(
    run_root: Path,
    factory,
    extra_field: str,
) -> None:
    payload = factory(run_root)
    payload[extra_field] = _ref("development/forbidden.json", extra_field)
    _assert_hold(
        lambda: (
            validate_feature_stage_manifest(payload)
            if factory is _feature_manifest
            else validate_outcome_stage_manifest(payload)
        ),
        "HOLD_TEST_STAGE_SCHEMA",
    )


@pytest.mark.parametrize(
    ("field_path", "wrong_path"),
    [
        ("feature_freeze", SEMANTIC_TEST_PREDICTION_RELATIVE_PATH),
        ("feature_authority", OUTCOME_AUTHORITY_RELATIVE_PATH),
        ("outcome_prediction", TYPED_PRE_TEST_FREEZE_RELATIVE_PATH),
        ("outcome_authority", FEATURE_AUTHORITY_RELATIVE_PATH),
    ],
)
def test_cross_stage_reference_paths_are_rejected(
    run_root: Path,
    field_path: str,
    wrong_path: str,
) -> None:
    if field_path.startswith("feature"):
        payload = _feature_manifest(run_root)
        field = (
            "typed_pre_test_freeze_commit_ref"
            if field_path == "feature_freeze"
            else "feature_authority_ref"
        )
        payload[field]["relative_path"] = wrong_path
        call = lambda: validate_feature_stage_manifest(payload)
    else:
        payload = _outcome_manifest(run_root)
        field = (
            "semantic_test_prediction_commit_ref"
            if field_path == "outcome_prediction"
            else "outcome_authority_ref"
        )
        payload[field]["relative_path"] = wrong_path
        call = lambda: validate_outcome_stage_manifest(payload)
    _assert_hold(call, "HOLD_TEST_STAGE_CROSS_STAGE_REF")


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (_feature_manifest, "sign_accuracy_gate", 0.59),
        (_outcome_manifest, "sign_accuracy_gate", 0.61),
        (_feature_manifest, "sign_gate_change_authorized", True),
        (_outcome_manifest, "ordinary_only", False),
        (_feature_manifest, "sealed_access", True),
        (_outcome_manifest, "downstream_unlock", True),
    ],
)
def test_sign_and_protected_boundaries_are_frozen(
    run_root: Path,
    factory,
    field: str,
    value: object,
) -> None:
    payload = factory(run_root)
    payload["boundaries"][field] = value
    _assert_hold(
        lambda: (
            validate_feature_stage_manifest(payload)
            if factory is _feature_manifest
            else validate_outcome_stage_manifest(payload)
        ),
        "HOLD_TEST_STAGE_SCHEMA",
    )


@pytest.mark.parametrize("token", ["sealed", "restricted", "GSE246381"])
def test_protected_reference_tokens_are_rejected(run_root: Path, token: str) -> None:
    payload = _outcome_manifest(run_root)
    payload["outcome_authority_ref"]["relative_path"] = f"authority_preparation/{token}/x.jsonl"
    _assert_hold(lambda: validate_outcome_stage_manifest(payload), "HOLD_PROTECTED_INPUT")


def test_path_escape_and_symlink_parent_are_rejected(run_root: Path) -> None:
    payload = _feature_manifest(run_root)
    _assert_hold(
        lambda: write_feature_stage_manifest_exclusive(
            run_root,
            run_root.parent / "escape.json",
            payload,
        ),
        "HOLD_TEST_STAGE_PATH",
    )

    real = run_root / "real_feature_stage"
    feature_parent = run_root / FEATURE_MANIFEST_RELATIVE_PATH.parent
    feature_parent.rename(real)
    feature_parent.symlink_to(real, target_is_directory=True)
    _assert_hold(
        lambda: write_feature_stage_manifest_exclusive(
            run_root,
            run_root / FEATURE_MANIFEST_RELATIVE_PATH,
            payload,
        ),
        "HOLD_TEST_STAGE_PATH",
    )


def test_noncanonical_hash_mismatch_and_hardlink_fail_closed(run_root: Path) -> None:
    payload = _feature_manifest(run_root)
    path = run_root / FEATURE_MANIFEST_RELATIVE_PATH
    compact = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(compact)
    _assert_hold(
        lambda: load_feature_stage_manifest_exact(
            run_root,
            path,
            hashlib.sha256(compact).hexdigest(),
        ),
        "HOLD_TEST_STAGE_NONCANONICAL",
    )
    _assert_hold(
        lambda: load_feature_stage_manifest_exact(run_root, path, _sha("wrong")),
        "HOLD_TEST_STAGE_HASH_MISMATCH",
    )

    path.unlink()
    target = path.parent / "target.json"
    raw = canonical_feature_stage_manifest_bytes(payload)
    target.write_bytes(raw)
    os.link(target, path)
    _assert_hold(
        lambda: load_feature_stage_manifest_exact(
            run_root,
            path,
            hashlib.sha256(raw).hexdigest(),
        ),
        "HOLD_TEST_STAGE_HASH_MISMATCH",
    )


def test_aliasing_row_set_or_cross_stage_commit_is_rejected(run_root: Path) -> None:
    feature = _feature_manifest(run_root)
    feature["feature_row_set_sha256"] = feature["feature_authority_ref"]["sha256"]
    _assert_hold(lambda: validate_feature_stage_manifest(feature), "HOLD_TEST_STAGE_ALIAS")

    outcome = _outcome_manifest(run_root)
    outcome["outcome_row_set_sha256"] = outcome[
        "semantic_test_prediction_commit_ref"
    ]["sha256"]
    _assert_hold(lambda: validate_outcome_stage_manifest(outcome), "HOLD_TEST_STAGE_ALIAS")


def test_missing_semantic_verifier_holds_before_feature_or_outcome_open(
    run_root: Path,
) -> None:
    _, feature_path, feature_sha = _write_feature(run_root)
    _, outcome_path, outcome_sha = _write_outcome(run_root)
    # Make both data authorities unreadable by hash.  The expected HOLD must
    # still be the missing semantic verifier, proving no data open occurred.
    (run_root / FEATURE_AUTHORITY_RELATIVE_PATH).write_bytes(b"changed feature\n")
    (run_root / OUTCOME_AUTHORITY_RELATIVE_PATH).write_bytes(b"changed outcome\n")
    _assert_hold(
        lambda: _fixture_feature_read(run_root, feature_path, feature_sha),
        "HOLD_FEATURE_STAGE_SEMANTIC_FREEZE_VERIFIER_NOT_CONNECTED",
    )
    _assert_hold(
        lambda: _fixture_outcome_release(run_root, outcome_path, outcome_sha),
        "HOLD_OUTCOME_STAGE_SEMANTIC_PREDICTION_VERIFIER_NOT_CONNECTED",
    )
    assert not (run_root / OUTCOME_CLAIM_RELATIVE_PATH).exists()


def test_semantic_verifier_alone_cannot_bypass_os_capability(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature, feature_path, feature_sha = _write_feature(run_root)
    outcome, outcome_path, outcome_sha = _write_outcome(run_root)
    _install_feature_verifiers(monkeypatch, feature)
    _install_outcome_verifiers(monkeypatch, outcome)
    monkeypatch.setattr(stage_module, "_FEATURE_OS_VERIFIER_REGISTRY", MappingProxyType({}))
    monkeypatch.setattr(stage_module, "_OUTCOME_OS_VERIFIER_REGISTRY", MappingProxyType({}))
    _assert_hold(
        lambda: _fixture_feature_verify(run_root, feature_path, feature_sha),
        "HOLD_FEATURE_STAGE_OS_CAPABILITY_NOT_VERIFIED",
    )
    _assert_hold(
        lambda: _fixture_outcome_verify(run_root, outcome_path, outcome_sha),
        "HOLD_OUTCOME_STAGE_OS_CAPABILITY_NOT_VERIFIED",
    )


def test_strict_registry_replay_can_read_only_feature_authority(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, path, digest = _write_feature(run_root)
    _install_feature_verifiers(monkeypatch, manifest)
    raw = _fixture_feature_read(run_root, path, digest)
    assert raw == (run_root / FEATURE_AUTHORITY_RELATIVE_PATH).read_bytes()


def test_bad_semantic_or_os_evidence_holds(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, path, digest = _write_feature(run_root)
    _install_feature_verifiers(monkeypatch, manifest)
    bad_semantic = {
        "schema_version": "r3x_strict_typed_pre_test_freeze_replay_evidence_v1",
        "verified_commit_sha256": _sha("wrong-freeze"),
        "authorized_feature_authority_sha256": manifest["feature_authority_ref"]["sha256"],
        "authorized_projection_sha256": {
            name: manifest["feature_row_projection_refs"][name]["sha256"]
            for name in ROW_AUTHORITIES
        },
        "authorized_row_set_sha256": manifest["feature_row_set_sha256"],
        "semantic_replay_passed": True,
        "outcome_refs_absent": True,
        "train_refs_absent": True,
        "mutable_selector_checkpoint_fields_absent": True,
    }
    monkeypatch.setattr(
        stage_module,
        "_FEATURE_SEMANTIC_VERIFIER_REGISTRY",
        MappingProxyType({FEATURE_SEMANTIC_VERIFIER_ID: lambda _root, _ref: bad_semantic}),
    )
    _assert_hold(
        lambda: _fixture_feature_verify(run_root, path, digest),
        "HOLD_FEATURE_STAGE_SEMANTIC_REPLAY",
    )

    _install_feature_verifiers(monkeypatch, manifest)
    bad_os = {
        "schema_version": "r3x_feature_process_mount_allowlist_evidence_v1",
        "feature_authority_mounted": True,
        "train_authority_mounted": True,
        "outcome_authority_mounted": False,
        "sealed_authority_mounted": False,
        "mount_allowlist_replay_passed": True,
    }
    monkeypatch.setattr(
        stage_module,
        "_FEATURE_OS_VERIFIER_REGISTRY",
        MappingProxyType({FEATURE_OS_VERIFIER_ID: lambda _root: bad_os}),
    )
    _assert_hold(
        lambda: _fixture_feature_verify(run_root, path, digest),
        "HOLD_FEATURE_STAGE_OS_CAPABILITY_NOT_VERIFIED",
    )


def test_outcome_claim_is_published_before_read_and_crash_is_not_retryable(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, path, digest = _write_outcome(run_root)
    _install_outcome_verifiers(monkeypatch, manifest)
    claim_path = run_root / OUTCOME_CLAIM_RELATIVE_PATH

    def crash_after_claim() -> None:
        assert claim_path.exists()
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        assert claim["status"] == "CLAIM_PUBLISHED_BEFORE_FIRST_OUTCOME_BYTE"
        assert claim["timestamp_used_as_ordering_evidence"] is False
        raise RuntimeError("synthetic crash after durable claim")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        _fixture_outcome_release(
            run_root,
            path,
            digest,
            _after_claim_hook=crash_after_claim,
        )
    assert claim_path.exists()

    # If retry reached the authority read, this mutation would produce a hash
    # error.  The durable claim must win first and consume the attempt.
    (run_root / OUTCOME_AUTHORITY_RELATIVE_PATH).write_bytes(b"changed after crash\n")
    _assert_hold(
        lambda: _fixture_outcome_release(run_root, path, digest),
        "HOLD_OUTCOME_RELEASE_ALREADY_CLAIMED_NO_RETRY",
    )


def test_successful_synthetic_outcome_release_is_claim_first_and_ordinary_only(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, path, digest = _write_outcome(run_root)
    _install_outcome_verifiers(monkeypatch, manifest)
    claim_path = run_root / OUTCOME_CLAIM_RELATIVE_PATH
    observed = {"claim_seen": False, "read_trace": None}

    def observe_claim() -> None:
        observed["claim_seen"] = claim_path.exists()

    def observe_read(trace: dict) -> None:
        assert claim_path.exists()
        observed["read_trace"] = dict(trace)

    result = _fixture_outcome_release(
        run_root,
        path,
        digest,
        _after_claim_hook=observe_claim,
        _read_trace_hook=observe_read,
    )
    assert observed["claim_seen"] is True
    expected_outcome = (run_root / OUTCOME_AUTHORITY_RELATIVE_PATH).read_bytes()
    assert result["outcome_bytes_sha256"] == hashlib.sha256(expected_outcome).hexdigest()
    assert result["outcome_byte_count"] == len(expected_outcome)
    assert "outcome_bytes" not in result
    assert observed["read_trace"] == {
        "claim_exists_before_read_completed": True,
        "outcome_bytes_sha256": hashlib.sha256(expected_outcome).hexdigest(),
        "outcome_byte_count": len(expected_outcome),
    }
    assert result["ordinary_only"] is True
    assert result["sealed_access"] is False
    assert result["downstream_unlock"] is False
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert not {
        "feature_authority_ref",
        "train_authority_ref",
        "model_config_ref",
        "checkpoint_ref",
    }.intersection(claim)


def test_claim_path_policy_and_verifier_registries_are_not_caller_selectable(
    run_root: Path,
) -> None:
    outcome = _outcome_manifest(run_root)
    outcome["claim_relative_path"] = "../retryable_claim.json"
    _assert_hold(lambda: validate_outcome_stage_manifest(outcome), "HOLD_TEST_STAGE_PATH")

    outcome = _outcome_manifest(run_root)
    outcome["claim_policy"] = "READ_THEN_BEST_EFFORT_CLAIM"
    _assert_hold(lambda: validate_outcome_stage_manifest(outcome), "HOLD_TEST_STAGE_SCHEMA")

    for function in (
        verify_feature_stage_capability_runtime,
        verify_outcome_stage_capability_runtime,
        release_ordinary_outcome_once,
    ):
        parameters = inspect.signature(function).parameters
        assert "registry" not in parameters
        assert "verifier" not in parameters


def test_parent_swap_is_an_explicit_non_unlockable_blocker(run_root: Path) -> None:
    feature = _feature_manifest(run_root)
    feature["namespace_integrity_status"] = "DESCRIPTOR_PINNED_PARENT_SWAP_PROTECTED"
    _assert_hold(lambda: validate_feature_stage_manifest(feature), "HOLD_TEST_STAGE_SCHEMA")

    outcome = _outcome_manifest(run_root)
    outcome["namespace_integrity_status"] = "PARENT_SWAP_SAFELY_IGNORED"
    _assert_hold(lambda: validate_outcome_stage_manifest(outcome), "HOLD_TEST_STAGE_SCHEMA")


def test_private_registry_monkeypatch_cannot_unlock_public_runtime(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature, feature_path, feature_sha = _write_feature(run_root)
    outcome, outcome_path, outcome_sha = _write_outcome(run_root)
    _install_feature_verifiers(monkeypatch, feature)
    _install_outcome_verifiers(monkeypatch, outcome)

    _assert_hold(
        lambda: verify_feature_stage_capability_runtime(
            run_root,
            feature_path,
            feature_sha,
        ),
        "HOLD_FEATURE_STAGE_PRODUCTION_RUNTIME_DISABLED",
    )
    _assert_hold(
        lambda: read_internal_test_features_with_capability(
            run_root,
            feature_path,
            feature_sha,
        ),
        "HOLD_FEATURE_STAGE_PRODUCTION_RUNTIME_DISABLED",
    )
    _assert_hold(
        lambda: verify_outcome_stage_capability_runtime(
            run_root,
            outcome_path,
            outcome_sha,
        ),
        "HOLD_OUTCOME_STAGE_PRODUCTION_RUNTIME_DISABLED",
    )
    _assert_hold(
        lambda: release_ordinary_outcome_once(run_root, outcome_path, outcome_sha),
        "HOLD_OUTCOME_AGGREGATE_EVALUATOR_NOT_CONNECTED",
    )
    assert not (run_root / OUTCOME_CLAIM_RELATIVE_PATH).exists()
    assert "_after_claim_hook" not in inspect.signature(
        release_ordinary_outcome_once
    ).parameters
