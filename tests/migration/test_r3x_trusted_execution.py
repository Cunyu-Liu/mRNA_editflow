"""Adversarial tests for pre-outcome code trust and execution replay."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3x.artifact_io import canonical_json_bytes  # noqa: E402
import scripts.r3x.trusted_execution as trust_module  # noqa: E402
from scripts.r3x.trusted_execution import (  # noqa: E402
    EXECUTION_CERTIFICATE_RELATIVE_PATH,
    EXECUTION_REPLAY_PASSED,
    HGB_VERIFIER_ID,
    PRODUCER_CAPABILITY_STATUS,
    PRODUCER_POLICY_ROLE,
    PRODUCER_POLICY_SCHEMA,
    PRODUCER_POLICY_STATUS,
    REQUIRED_PRODUCER_CAPABILITIES,
    TRUSTED_SNAPSHOT_RELATIVE_PATH,
    TRUST_ASSUMPTIONS,
    TRUST_LIMITATIONS,
    TrustedExecutionError,
    canonical_preconfigured_producer_policy_bytes,
    load_preconfigured_producer_policy_exact,
    load_trusted_code_snapshot_authority_exact,
    registered_execution_verifier_ids,
    replay_execution_qualification_certificate_exact,
    validate_execution_qualification_certificate,
    validate_preconfigured_producer_policy,
    write_execution_qualification_certificate_exclusive,
    write_trusted_code_snapshot_authority_exclusive,
)


PRODUCER_ID = "R3X_EXTERNAL_PREOUTCOME_SNAPSHOT_PRODUCER_V1"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "ordinary_run"
    (root / "development" / "trust").mkdir(parents=True)
    (root / "scripts" / "r3x").mkdir(parents=True)
    (root / "scripts" / "r3x" / "model.py").write_text(
        "MODEL_VERSION = 1\n",
        encoding="utf-8",
    )
    return root


def _code_entry(root: Path, relative_path: str = "scripts/r3x/model.py") -> dict:
    raw = (root / relative_path).read_bytes()
    return {"path": relative_path, "sha256": hashlib.sha256(raw).hexdigest()}


def _policy(root: Path, *, code_entry: dict | None = None) -> dict:
    files = [dict(code_entry or _code_entry(root))]
    return {
        "schema_version": PRODUCER_POLICY_SCHEMA,
        "artifact_role": PRODUCER_POLICY_ROLE,
        "status": PRODUCER_POLICY_STATUS,
        "policy_id": "R3X_PREOUTCOME_TRUST_POLICY_V1",
        "trusted_producer_id": PRODUCER_ID,
        "producer_capability_status": PRODUCER_CAPABILITY_STATUS,
        "producer_capability_evidence_sha256": _sha(
            "independently-reviewed-producer-capability-evidence"
        ),
        "producer_capabilities": dict(REQUIRED_PRODUCER_CAPABILITIES),
        "lineage": {
            "run_start_sha256": _sha("run-start"),
            "activation_sha256": _sha("external-preoutcome-activation"),
            "development_commit_sha256": None,
            "pre_test_freeze_state": "ABSENT_REQUIRED_AT_SNAPSHOT_PUBLICATION",
            "test_prediction_state": "ABSENT_REQUIRED_AT_SNAPSHOT_PUBLICATION",
            "ordinary_outcome_release_state": "ABSENT_REQUIRED_AT_SNAPSHOT_PUBLICATION",
            "outcome_release_claim_state": "ABSENT_REQUIRED_AT_SNAPSHOT_PUBLICATION",
            "commit_namespace_policy": "EXTERNALLY_SERIALIZED_APPEND_ONLY_NO_DELETE",
        },
        "approved_code_files": files,
        "approved_code_tree_sha256": trust_module.canonical_sha256(files),
        "trusted_snapshot_relative_path": TRUSTED_SNAPSHOT_RELATIVE_PATH,
        "allowed_execution_verifier_ids": [HGB_VERIFIER_ID],
        "trust_assumptions": list(TRUST_ASSUMPTIONS),
        "trust_limitations": list(TRUST_LIMITATIONS),
        "boundaries": {
            "sign_accuracy_gate": 0.60,
            "sign_gate_change_authorized": False,
            "ordinary_only": True,
            "sealed_access": False,
            "downstream_unlock": False,
            "training_ready": False,
        },
    }


def _install_policy(root: Path, payload: dict | None = None) -> tuple[Path, str]:
    policy = payload or _policy(root)
    raw = canonical_preconfigured_producer_policy_bytes(policy)
    path = root / "development" / "trust" / "preconfigured_producer_policy.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _install_unvalidated_policy(root: Path, payload: dict) -> tuple[Path, str]:
    raw = canonical_json_bytes(payload)
    path = root / "development" / "trust" / "preconfigured_producer_policy.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _write_snapshot(root: Path) -> tuple[Path, str, Path, str]:
    policy_path, policy_sha = _install_policy(root)
    snapshot_path, snapshot_sha = write_trusted_code_snapshot_authority_exclusive(
        root,
        producer_policy_path=policy_path,
        producer_policy_sha256=policy_sha,
        expected_producer_id=PRODUCER_ID,
    )
    return policy_path, policy_sha, snapshot_path, snapshot_sha


def _assert_hold(callable_, code: str) -> TrustedExecutionError:
    with pytest.raises(TrustedExecutionError) as caught:
        callable_()
    assert caught.value.code == code
    return caught.value


def test_snapshot_is_derived_exclusively_and_exactly_replayed(run_root: Path) -> None:
    policy_path, policy_sha, snapshot_path, snapshot_sha = _write_snapshot(run_root)

    snapshot = load_trusted_code_snapshot_authority_exact(
        run_root,
        snapshot_path,
        snapshot_sha,
        producer_policy_path=policy_path,
        producer_policy_sha256=policy_sha,
        expected_producer_id=PRODUCER_ID,
    )
    assert snapshot_path == run_root / TRUSTED_SNAPSHOT_RELATIVE_PATH
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == snapshot_sha
    assert snapshot["files"] == [_code_entry(run_root)]
    assert snapshot["preoutcome_publication_witness"] == {
        "ordering_basis": (
            "EXCLUSIVE_PUBLICATION_AFTER_NAMESPACE_ABSENCE_REPLAY_"
            "UNDER_EXTERNAL_SERIALIZATION"
        ),
        "freeze_and_outcome_namespace_absence_replayed": True,
        "timestamp_used_as_ordering_evidence": False,
    }
    assert snapshot["boundaries"] == {
        "sign_accuracy_gate": 0.60,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
        "training_ready": False,
    }
    assert "timestamp" not in snapshot["lineage"]
    assert list(inspect.signature(write_trusted_code_snapshot_authority_exclusive).parameters) == [
        "run_root",
        "producer_policy_path",
        "producer_policy_sha256",
        "expected_producer_id",
    ]

    alias = run_root / "development" / "trust" / "caller_selected_snapshot.json"
    alias.write_bytes(snapshot_path.read_bytes())
    _assert_hold(
        lambda: load_trusted_code_snapshot_authority_exact(
            run_root,
            alias,
            snapshot_sha,
            producer_policy_path=policy_path,
            producer_policy_sha256=policy_sha,
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUST_NAMESPACE",
    )

    _assert_hold(
        lambda: write_trusted_code_snapshot_authority_exclusive(
            run_root,
            producer_policy_path=policy_path,
            producer_policy_sha256=policy_sha,
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUSTED_EXECUTION_WRITE",
    )


def test_policy_digest_and_producer_identity_are_external_runtime_inputs(
    run_root: Path,
) -> None:
    policy_path, policy_sha = _install_policy(run_root)
    _assert_hold(
        lambda: load_preconfigured_producer_policy_exact(
            run_root,
            policy_path,
            _sha("caller-self-reported-different-root"),
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUSTED_EXECUTION_HASH_MISMATCH",
    )
    _assert_hold(
        lambda: load_preconfigured_producer_policy_exact(
            run_root,
            policy_path,
            policy_sha,
            expected_producer_id="DIFFERENT_PRECONFIGURED_PRODUCER",
        ),
        "HOLD_PRODUCER_IDENTITY_MISMATCH",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("internal_test_outcome_read", True),
        ("internal_test_feature_read", True),
        ("train_development_outcome_read", True),
        ("source_code_write", True),
        ("training_process_control", True),
        ("sealed_data_read", True),
    ],
)
def test_missing_independent_producer_capability_fails_closed(
    run_root: Path,
    field: str,
    value: bool,
) -> None:
    policy = _policy(run_root)
    policy["producer_capabilities"][field] = value
    policy_path, policy_sha = _install_unvalidated_policy(run_root, policy)
    _assert_hold(
        lambda: write_trusted_code_snapshot_authority_exclusive(
            run_root,
            producer_policy_path=policy_path,
            producer_policy_sha256=policy_sha,
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUSTED_EXECUTION_SCHEMA",
    )


def test_capability_status_or_evidence_cannot_be_omitted(run_root: Path) -> None:
    policy = _policy(run_root)
    policy["producer_capability_status"] = "CALLER_ASSERTED"
    policy_path, policy_sha = _install_unvalidated_policy(run_root, policy)
    _assert_hold(
        lambda: load_preconfigured_producer_policy_exact(
            run_root,
            policy_path,
            policy_sha,
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUSTED_EXECUTION_SCHEMA",
    )

    policy = _policy(run_root)
    policy["producer_capability_evidence_sha256"] = "missing"
    policy_path.unlink()
    path = policy_path
    raw = canonical_json_bytes(policy)
    path.write_bytes(raw)
    _assert_hold(
        lambda: load_preconfigured_producer_policy_exact(
            run_root,
            path,
            hashlib.sha256(raw).hexdigest(),
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUSTED_EXECUTION_SCHEMA",
    )


def test_snapshot_cannot_self_report_files_or_lineage(run_root: Path) -> None:
    policy_path, policy_sha, snapshot_path, snapshot_sha = _write_snapshot(run_root)
    malicious = json.loads(snapshot_path.read_text(encoding="utf-8"))
    malicious["files"][0]["sha256"] = _sha("caller-chosen-code")
    malicious["tree_sha256"] = trust_module.canonical_sha256(malicious["files"])
    malicious["lineage"]["activation_sha256"] = _sha("post-outcome-activation")
    snapshot_path.unlink()
    fake_path = snapshot_path
    fake_raw = canonical_json_bytes(malicious)
    fake_path.write_bytes(fake_raw)

    _assert_hold(
        lambda: load_trusted_code_snapshot_authority_exact(
            run_root,
            fake_path,
            hashlib.sha256(fake_raw).hexdigest(),
            producer_policy_path=policy_path,
            producer_policy_sha256=policy_sha,
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUSTED_CODE_POLICY_MISMATCH",
    )
    assert snapshot_sha != hashlib.sha256(fake_raw).hexdigest()


def test_code_bytes_are_replayed_at_snapshot_write_and_runtime_load(
    run_root: Path,
) -> None:
    policy_path, policy_sha = _install_policy(run_root)
    code = run_root / "scripts" / "r3x" / "model.py"
    code.write_text("MODEL_VERSION = 2\n", encoding="utf-8")
    _assert_hold(
        lambda: write_trusted_code_snapshot_authority_exclusive(
            run_root,
            producer_policy_path=policy_path,
            producer_policy_sha256=policy_sha,
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUSTED_EXECUTION_HASH_MISMATCH",
    )

    code.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    snapshot_path, snapshot_sha = write_trusted_code_snapshot_authority_exclusive(
        run_root,
        producer_policy_path=policy_path,
        producer_policy_sha256=policy_sha,
        expected_producer_id=PRODUCER_ID,
    )
    code.write_text("MODEL_VERSION = 3\n", encoding="utf-8")
    _assert_hold(
        lambda: load_trusted_code_snapshot_authority_exact(
            run_root,
            snapshot_path,
            snapshot_sha,
            producer_policy_path=policy_path,
            producer_policy_sha256=policy_sha,
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUSTED_EXECUTION_HASH_MISMATCH",
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "../escape.py",
        "/absolute/model.py",
        "data/model.py",
        "outputs/model.py",
        "tests/model.py",
        "scripts/../models/model.py",
        "scripts/r3x/model.json",
    ],
)
def test_policy_rejects_unsafe_code_paths(run_root: Path, relative_path: str) -> None:
    policy = _policy(
        run_root,
        code_entry={"path": relative_path, "sha256": _sha(relative_path)},
    )
    policy["approved_code_tree_sha256"] = trust_module.canonical_sha256(
        policy["approved_code_files"]
    )
    with pytest.raises(TrustedExecutionError) as caught:
        validate_preconfigured_producer_policy(policy)
    assert caught.value.code == "HOLD_TRUSTED_CODE_PATH"


def test_code_symlink_and_hardlink_are_rejected(run_root: Path) -> None:
    target = run_root / "scripts" / "r3x" / "model.py"
    alias = run_root / "scripts" / "r3x" / "alias.py"
    alias.symlink_to(target.name)
    policy = _policy(run_root, code_entry=_code_entry(run_root, "scripts/r3x/alias.py"))
    policy_path, policy_sha = _install_policy(run_root, policy)
    _assert_hold(
        lambda: write_trusted_code_snapshot_authority_exclusive(
            run_root,
            producer_policy_path=policy_path,
            producer_policy_sha256=policy_sha,
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUST_NAMESPACE",
    )

    alias.unlink()
    os.link(target, alias)
    policy = _policy(run_root, code_entry=_code_entry(run_root, "scripts/r3x/alias.py"))
    policy_path.unlink()
    second = policy_path
    raw = canonical_preconfigured_producer_policy_bytes(policy)
    second.write_bytes(raw)
    _assert_hold(
        lambda: write_trusted_code_snapshot_authority_exclusive(
            run_root,
            producer_policy_path=second,
            producer_policy_sha256=hashlib.sha256(raw).hexdigest(),
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUST_NAMESPACE",
    )


@pytest.mark.parametrize(
    "state_path",
    [
        "commits/02_PRE_TEST_FREEZE_COMMIT.json",
        "commits/02_TYPED_PRE_TEST_FREEZE_COMMIT.json",
        "commits/03_TEST_PREDICTION_COMMIT.json",
        "commits/03_SEMANTIC_TEST_PREDICTION_COMMIT.json",
        "commits/04_ORDINARY_TEST_OUTCOME_RELEASE_RECORD.json",
        "commits/.ORDINARY_TEST_OUTCOME_RELEASE_CLAIM.json",
        "commits/.ORDINARY_OUTCOME_RELEASE_CAPABILITY_CLAIM.json",
    ],
)
def test_snapshot_publication_requires_pre_freeze_pre_outcome_namespace(
    run_root: Path,
    state_path: str,
) -> None:
    assert state_path in trust_module.COMMIT_STATE_PATHS
    policy_path, policy_sha = _install_policy(run_root)
    state = run_root / state_path
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("already advanced\n", encoding="utf-8")
    _assert_hold(
        lambda: write_trusted_code_snapshot_authority_exclusive(
            run_root,
            producer_policy_path=policy_path,
            producer_policy_sha256=policy_sha,
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_PREOUTCOME_LINEAGE_ALREADY_ADVANCED",
    )


def test_policy_loader_requires_canonical_exact_single_link_bytes(run_root: Path) -> None:
    policy = _policy(run_root)
    compact = json.dumps(policy, sort_keys=True).encode("utf-8")
    path = run_root / "development" / "trust" / "preconfigured_producer_policy.json"
    path.write_bytes(compact)
    _assert_hold(
        lambda: load_preconfigured_producer_policy_exact(
            run_root,
            path,
            hashlib.sha256(compact).hexdigest(),
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUSTED_EXECUTION_NONCANONICAL",
    )

    path.unlink()
    canonical = canonical_preconfigured_producer_policy_bytes(policy)
    target = run_root / "development" / "trust" / "target.json"
    target.write_bytes(canonical)
    os.link(target, path)
    _assert_hold(
        lambda: load_preconfigured_producer_policy_exact(
            run_root,
            path,
            hashlib.sha256(canonical).hexdigest(),
            expected_producer_id=PRODUCER_ID,
        ),
        "HOLD_TRUST_NAMESPACE",
    )


def test_execution_certificate_is_registry_measured_and_replayed(run_root: Path) -> None:
    policy_path, policy_sha, snapshot_path, snapshot_sha = _write_snapshot(run_root)
    certificate_path, certificate_sha = write_execution_qualification_certificate_exclusive(
        run_root,
        producer_policy_path=policy_path,
        producer_policy_sha256=policy_sha,
        expected_producer_id=PRODUCER_ID,
        trusted_snapshot_path=snapshot_path,
        trusted_snapshot_sha256=snapshot_sha,
    )
    certificate = replay_execution_qualification_certificate_exact(
        run_root,
        certificate_path,
        certificate_sha,
        producer_policy_path=policy_path,
        producer_policy_sha256=policy_sha,
        expected_producer_id=PRODUCER_ID,
        trusted_snapshot_path=snapshot_path,
        trusted_snapshot_sha256=snapshot_sha,
    )

    assert certificate_path == run_root / EXECUTION_CERTIFICATE_RELATIVE_PATH
    assert certificate["verifier_id"] == HGB_VERIFIER_ID
    assert certificate["replay_status"] == EXECUTION_REPLAY_PASSED
    assert certificate["qualification_scope"] == {
        "method_id": "HISTOGRAM_GRADIENT_BOOSTING",
        "device_scope": "CPU_ONLY",
        "data_scope": "FIXED_SYNTHETIC_SMOKE_ONLY",
        "project_data_accessed": False,
        "gpu_initialized": False,
    }
    measurement = certificate["environment_measurement"]
    assert measurement["runtime"]["python_executable_sha256"]
    assert measurement["runtime"]["runtime_fingerprint_sha256"]
    assert [item["distribution"] for item in measurement["packages"]] == [
        "joblib",
        "numpy",
        "scikit-learn",
        "scipy",
        "threadpoolctl",
    ]
    assert all(item["package_fingerprint_sha256"] for item in measurement["packages"])
    smoke = measurement["synthetic_hgb_smoke"]
    environment_policy = measurement["child_environment_sanitization"]
    assert smoke["repeated_fit_exact_match"] is True
    assert smoke["finite_predictions"] is True
    assert environment_policy["schema_version"] == (
        trust_module.HGB_CHILD_ENVIRONMENT_POLICY_SCHEMA
    )
    assert environment_policy["parent_environment_copy_mode"] == (
        "EXPLICIT_KEY_LOOKUP_ONLY_NO_FULL_COPY"
    )
    assert environment_policy["passed_key_names_sha256"] == (
        trust_module.canonical_sha256(environment_policy["passed_key_names"])
    )
    assert smoke["received_environment_key_names_before_third_party_import"] == (
        environment_policy["passed_key_names"]
    )
    assert smoke["received_environment_key_names_sha256"] == environment_policy[
        "passed_key_names_sha256"
    ]
    assert environment_policy["pythonpath_passed"] is False
    assert environment_policy["unlisted_parent_environment_keys_passed"] is False
    assert environment_policy["sensitive_name_class_passed"] is False
    assert certificate["execution_capability_boundary"] == {
        "status": "BLOCKED_PENDING_OS_LEVEL_DEVELOPMENT_PROCESS_MOUNT_EVIDENCE",
        "internal_test_feature_authority_mount_status": "NOT_VERIFIED_UNMOUNTED",
        "internal_test_outcome_authority_mount_status": "NOT_VERIFIED_UNMOUNTED",
        "sealed_authority_mount_status": "NOT_VERIFIED_UNMOUNTED",
        "training_capability_status": (
            "SYNTHETIC_EXECUTABILITY_ONLY_NOT_TRAINING_CAPABILITY"
        ),
    }
    assert certificate["boundaries"]["training_ready"] is False
    assert certificate["execution_capability_boundary"]["status"].startswith(
        "BLOCKED_PENDING_OS_LEVEL"
    )
    assert certificate["boundaries"]["sign_accuracy_gate"] == 0.60


def test_hgb_child_environment_does_not_inherit_sensitive_parent_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dangerous_environment = {
        "PYTHONPATH": "/tmp/r3x-attacker-pythonpath-91f3",
        "PATH": "/tmp/r3x-attacker-bin-2d7a",
        "HOME": "/tmp/r3x-attacker-home-662c",
        "PROJECT_DATA_PATH": "/tmp/r3x-private-project-data-b91e",
        "DATABASE_URL": "postgresql://r3x-sensitive-database-4fa0",
        "HMAC_KEY": "r3x-sensitive-hmac-key-c392",
        "API_SECRET": "r3x-sensitive-api-secret-c159",
        "SERVICE_TOKEN": "r3x-sensitive-service-token-f843",
        "OPENAI_API_KEY": "r3x-sensitive-openai-key-b180",
    }
    for name, value in dangerous_environment.items():
        monkeypatch.setenv(name, value)

    measurement = trust_module._measure_hgb_cpu_v1()
    policy = measurement["child_environment_sanitization"]
    smoke = measurement["synthetic_hgb_smoke"]
    passed_names = policy["passed_key_names"]
    inherited_names = policy["platform_inherited_keys_present"]
    expected_names = sorted(
        set(trust_module.HGB_CHILD_FIXED_ENVIRONMENT) | set(inherited_names)
    )

    assert policy["platform_inherit_allowlist"] == list(
        trust_module.HGB_CHILD_PLATFORM_INHERIT_ALLOWLIST.get(
            policy["platform_system"],
            (),
        )
    )
    assert passed_names == expected_names
    assert policy["passed_key_names_sha256"] == trust_module.canonical_sha256(
        passed_names
    )
    assert smoke["received_environment_key_names_before_third_party_import"] == (
        passed_names
    )
    assert smoke["received_environment_key_names_sha256"] == policy[
        "passed_key_names_sha256"
    ]
    assert not set(dangerous_environment).intersection(passed_names)
    assert policy["pythonpath_passed"] is False
    assert policy["unlisted_parent_environment_keys_passed"] is False
    assert policy["sensitive_name_class_passed"] is False

    serialized_measurement = json.dumps(measurement, sort_keys=True)
    for secret_value in dangerous_environment.values():
        assert secret_value not in serialized_measurement


def test_execution_certificate_cannot_be_hash_only_or_use_caller_registry(
    run_root: Path,
) -> None:
    assert registered_execution_verifier_ids() == (HGB_VERIFIER_ID,)
    assert "registry" not in inspect.signature(
        replay_execution_qualification_certificate_exact
    ).parameters
    assert "environment_measurement" not in inspect.signature(
        write_execution_qualification_certificate_exclusive
    ).parameters

    policy_path, policy_sha, snapshot_path, snapshot_sha = _write_snapshot(run_root)
    _assert_hold(
        lambda: write_execution_qualification_certificate_exclusive(
            run_root,
            producer_policy_path=policy_path,
            producer_policy_sha256=policy_sha,
            expected_producer_id=PRODUCER_ID,
            trusted_snapshot_path=snapshot_path,
            trusted_snapshot_sha256=snapshot_sha,
            verifier_id="CALLER_SUPPLIED_ALWAYS_PASS_VERIFIER",
        ),
        "HOLD_EXECUTION_VERIFIER_NOT_PRECONFIGURED",
    )


def test_environment_measurement_tampering_fails_before_or_during_replay(
    run_root: Path,
) -> None:
    policy_path, policy_sha, snapshot_path, snapshot_sha = _write_snapshot(run_root)
    certificate_path, certificate_sha = write_execution_qualification_certificate_exclusive(
        run_root,
        producer_policy_path=policy_path,
        producer_policy_sha256=policy_sha,
        expected_producer_id=PRODUCER_ID,
        trusted_snapshot_path=snapshot_path,
        trusted_snapshot_sha256=snapshot_sha,
    )
    tampered = json.loads(certificate_path.read_text(encoding="utf-8"))
    tampered["environment_measurement"]["synthetic_hgb_smoke"][
        "prediction_sha256"
    ] = _sha("fabricated-pass")
    measurement = tampered["environment_measurement"]
    measurement["measurement_sha256"] = trust_module.canonical_sha256(
        {key: value for key, value in measurement.items() if key != "measurement_sha256"}
    )
    tampered["environment_measurement_sha256"] = trust_module.canonical_sha256(measurement)
    certificate_path.unlink()
    fake_path = certificate_path
    fake_raw = canonical_json_bytes(tampered)
    fake_path.write_bytes(fake_raw)

    _assert_hold(
        lambda: replay_execution_qualification_certificate_exact(
            run_root,
            fake_path,
            hashlib.sha256(fake_raw).hexdigest(),
            producer_policy_path=policy_path,
            producer_policy_sha256=policy_sha,
            expected_producer_id=PRODUCER_ID,
            trusted_snapshot_path=snapshot_path,
            trusted_snapshot_sha256=snapshot_sha,
        ),
        "HOLD_EXECUTION_ENVIRONMENT_DRIFT",
    )
    assert certificate_sha != hashlib.sha256(fake_raw).hexdigest()


def test_certificate_snapshot_binding_and_boundaries_cannot_be_changed(
    run_root: Path,
) -> None:
    policy_path, policy_sha, snapshot_path, snapshot_sha = _write_snapshot(run_root)
    certificate_path, _ = write_execution_qualification_certificate_exclusive(
        run_root,
        producer_policy_path=policy_path,
        producer_policy_sha256=policy_sha,
        expected_producer_id=PRODUCER_ID,
        trusted_snapshot_path=snapshot_path,
        trusted_snapshot_sha256=snapshot_sha,
    )
    policy = load_preconfigured_producer_policy_exact(
        run_root,
        policy_path,
        policy_sha,
        expected_producer_id=PRODUCER_ID,
    )
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["trusted_code_snapshot_ref"]["sha256"] = _sha("other-snapshot")
    _assert_hold(
        lambda: validate_execution_qualification_certificate(
            certificate,
            preconfigured_policy=policy,
            trusted_snapshot_sha256=snapshot_sha,
        ),
        "HOLD_EXECUTION_SNAPSHOT_BINDING",
    )

    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["boundaries"]["sign_accuracy_gate"] = 0.59
    _assert_hold(
        lambda: validate_execution_qualification_certificate(
            certificate,
            preconfigured_policy=policy,
            trusted_snapshot_sha256=snapshot_sha,
        ),
        "HOLD_TRUSTED_EXECUTION_SCHEMA",
    )


def test_certificate_is_development_only_not_a_staged_freeze_substitute(
    run_root: Path,
) -> None:
    policy_path, policy_sha, snapshot_path, snapshot_sha = _write_snapshot(run_root)
    certificate_path, certificate_sha = write_execution_qualification_certificate_exclusive(
        run_root,
        producer_policy_path=policy_path,
        producer_policy_sha256=policy_sha,
        expected_producer_id=PRODUCER_ID,
        trusted_snapshot_path=snapshot_path,
        trusted_snapshot_sha256=snapshot_sha,
    )
    certificate = replay_execution_qualification_certificate_exact(
        run_root,
        certificate_path,
        certificate_sha,
        producer_policy_path=policy_path,
        producer_policy_sha256=policy_sha,
        expected_producer_id=PRODUCER_ID,
        trusted_snapshot_path=snapshot_path,
        trusted_snapshot_sha256=snapshot_sha,
    )
    prohibited_freeze_substitutes = {
        "selector_ref",
        "predictor_provenance_refs",
        "frozen_checkpoint_refs",
        "authority_hashes",
        "nondata_authority_hashes",
        "internal_test_feature_authority_ref",
        "internal_test_outcome_authority_ref",
    }
    assert not prohibited_freeze_substitutes.intersection(certificate)
    assert certificate["boundaries"]["training_ready"] is False
    assert (
        "TRAINING_REMAINS_BLOCKED_UNTIL_ACTIVATION_BINDS_POLICY_SNAPSHOT_AND_CERTIFICATE"
        in certificate["trust_limitations"]
    )


def test_exact_hashing_does_not_claim_absence_of_hidden_channels(run_root: Path) -> None:
    policy = _policy(run_root)
    assert (
        "CODE_HASH_REPLAY_DOES_NOT_PROVE_ABSENCE_OF_COVERT_OR_SEMANTIC_CHANNELS"
        in policy["trust_limitations"]
    )
    # A semantically suspicious but policy-reviewed source is not magically
    # classified by filename/hash validation.  The external review assumption
    # stays visible and training readiness remains false.
    code = run_root / "scripts" / "r3x" / "model.py"
    code.write_text("ENCODED_CONSTANT = [1, 0, 1]\n", encoding="utf-8")
    policy = _policy(run_root)
    validated = validate_preconfigured_producer_policy(policy)
    assert validated["boundaries"]["training_ready"] is False
    assert validated["trust_limitations"] == list(TRUST_LIMITATIONS)
