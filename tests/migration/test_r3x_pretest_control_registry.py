"""Adversarial tests for the train/development-only pre-test control registry."""

from __future__ import annotations

import copy
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

import scripts.r3x.pretest_control_registry as registry_module  # noqa: E402
from scripts.r3x.pretest_control_registry import (  # noqa: E402
    ACTIVE,
    ARTIFACT_ROLE,
    BROKEN_PAIR_CONTROL_ID,
    BROKEN_PAIR_REASON,
    CONTROL_NOT_RUN,
    CONTROL_RUN,
    NAMESPACE_INTEGRITY_STATUS,
    OVERALL_STATUS,
    PROPOSED_NOT_ACTIVE,
    REGISTRY_RELATIVE_PATH,
    REQUIRED_CONTROL_IDS,
    REQUIRED_CONTROL_IDS_SHA256,
    SCHEMA,
    PretestControlRegistryError,
    canonical_pretest_control_registry_bytes,
    control_evidence_bundle_sha256,
    expected_verifier_id,
    load_pretest_control_registry_exact,
    require_pretest_control_registry,
    validate_pretest_control_registry,
    write_pretest_control_registry_exclusive,
)
from scripts.r3x.protocol import REQUIRED_NEGATIVE_CONTROLS, canonical_sha256  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "ordinary_run"
    (root / REGISTRY_RELATIVE_PATH.parent).mkdir(parents=True)
    return root


def _ref(path: str, label: str) -> dict:
    return {"relative_path": path, "sha256": _sha(label)}


def _control_root(control_id: str) -> str:
    return f"development/pretest_controls/{control_id.lower()}"


def _not_run_entry(
    control_id: str,
    *,
    reason: str = "CONTROL_EVIDENCE_NOT_AVAILABLE",
    activation_status: str = ACTIVE,
) -> dict:
    if control_id == BROKEN_PAIR_CONTROL_ID:
        reason = BROKEN_PAIR_REASON
        activation_status = PROPOSED_NOT_ACTIVE
    return {
        "control_id": control_id,
        "disposition": CONTROL_NOT_RUN,
        "activation_status": activation_status,
        "train_control_manifest_ref": _ref(
            f"{_control_root(control_id)}/train_control_manifest.json",
            f"{control_id}:not-run-manifest",
        ),
        "development_state_ref": None,
        "development_prediction_ref": None,
        "development_verifier_ref": None,
        "verifier_id": None,
        "evidence_bundle_sha256": None,
        "reason_code": reason,
    }


def _run_entry(control_id: str) -> dict:
    root = _control_root(control_id)
    entry = {
        "control_id": control_id,
        "disposition": CONTROL_RUN,
        "activation_status": ACTIVE,
        "train_control_manifest_ref": _ref(
            f"{root}/train_control_manifest.json",
            f"{control_id}:train-control-manifest",
        ),
        "development_state_ref": _ref(
            f"{root}/development_state.json",
            f"{control_id}:development-state",
        ),
        "development_prediction_ref": _ref(
            f"{root}/development_predictions.jsonl",
            f"{control_id}:development-prediction",
        ),
        "development_verifier_ref": _ref(
            f"{root}/development_verifier.json",
            f"{control_id}:development-verifier",
        ),
        "verifier_id": expected_verifier_id(control_id),
        "evidence_bundle_sha256": None,
        "reason_code": None,
    }
    entry["evidence_bundle_sha256"] = control_evidence_bundle_sha256(entry)
    return entry


def _payload(*, all_not_run: bool = False) -> dict:
    controls = []
    for control_id in REQUIRED_CONTROL_IDS:
        if all_not_run or control_id == BROKEN_PAIR_CONTROL_ID:
            controls.append(_not_run_entry(control_id))
        else:
            controls.append(_run_entry(control_id))
    return {
        "schema_version": SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "status": OVERALL_STATUS,
        "method_protocol_ref": _ref(
            "development/authorities/method_protocol.json",
            "method-protocol",
        ),
        "required_control_ids": list(REQUIRED_CONTROL_IDS),
        "required_control_ids_sha256": REQUIRED_CONTROL_IDS_SHA256,
        "train_development_manifest_ref": _ref(
            "development/pretest_controls/train_development_control_manifest.json",
            "train-development-control-manifest",
        ),
        "train_development_state_ref": _ref(
            "development/pretest_controls/train_development_control_state.json",
            "train-development-control-state",
        ),
        "controls": controls,
        "semantic_verifier_status": (
            "EXACT_DEVELOPMENT_VERIFIER_REGISTRY_REPLAY_REQUIRED"
        ),
        "namespace_integrity_status": NAMESPACE_INTEGRITY_STATUS,
        "broken_pair_activation_status": PROPOSED_NOT_ACTIVE,
        "all_required_controls_development_replayed": False,
        "can_satisfy_pretest_freeze": False,
        "boundaries": {
            "sign_accuracy_gate": 0.60,
            "sign_gate_change_authorized": False,
            "ordinary_only": True,
            "sealed_access": False,
            "downstream_unlock": False,
        },
    }


def _write_registry(root: Path, payload: dict | None = None) -> tuple[dict, Path, str]:
    registry = payload or _payload()
    path = root / REGISTRY_RELATIVE_PATH
    digest = write_pretest_control_registry_exclusive(root, path, registry)
    return registry, path, digest


def _assert_hold(callable_, code: str) -> PretestControlRegistryError:
    with pytest.raises(PretestControlRegistryError) as caught:
        callable_()
    assert caught.value.code == code
    return caught.value


def test_required_ids_are_derived_from_frozen_method_protocol_constant() -> None:
    assert REQUIRED_CONTROL_IDS == tuple(REQUIRED_NEGATIVE_CONTROLS)
    assert REQUIRED_CONTROL_IDS_SHA256 == canonical_sha256(REQUIRED_NEGATIVE_CONTROLS)
    assert REQUIRED_CONTROL_IDS[0] == BROKEN_PAIR_CONTROL_ID


def test_registry_is_canonical_exclusive_exact_and_always_hold(run_root: Path) -> None:
    payload, path, digest = _write_registry(run_root)
    loaded = load_pretest_control_registry_exact(run_root, path, digest)
    assert loaded == payload
    assert path.read_bytes() == canonical_pretest_control_registry_bytes(payload)
    assert loaded["status"] == OVERALL_STATUS
    assert loaded["can_satisfy_pretest_freeze"] is False
    assert loaded["all_required_controls_development_replayed"] is False
    assert loaded["boundaries"] == {
        "sign_accuracy_gate": 0.60,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    _assert_hold(
        lambda: write_pretest_control_registry_exclusive(run_root, path, payload),
        "HOLD_PRETEST_CONTROL_WRITE",
    )


def test_registry_contains_no_evaluation_metric_or_row_support_role(run_root: Path) -> None:
    raw = canonical_pretest_control_registry_bytes(_payload()).decode("utf-8").lower()
    for token in (
        "internal_test",
        "test_outcome",
        "row_support",
        "metric_value",
        "sign_accuracy_observed",
        "p_value",
        "effect_size",
    ):
        assert token not in raw


def test_not_run_is_typed_hold_never_pass(run_root: Path) -> None:
    payload = validate_pretest_control_registry(_payload(all_not_run=True))
    assert all(entry["disposition"] == CONTROL_NOT_RUN for entry in payload["controls"])
    assert "PASS" not in json.dumps(payload)
    assert payload["can_satisfy_pretest_freeze"] is False


def test_broken_pair_must_remain_proposed_not_active_not_run(run_root: Path) -> None:
    payload = _payload()
    broken = payload["controls"][0]
    assert broken == _not_run_entry(BROKEN_PAIR_CONTROL_ID)

    payload["controls"][0] = _run_entry(BROKEN_PAIR_CONTROL_ID)
    _assert_hold(
        lambda: validate_pretest_control_registry(payload),
        "HOLD_BROKEN_PAIR_CONTROL_PROPOSED_NOT_ACTIVE",
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered"])
def test_required_control_set_and_order_are_exact(run_root: Path, mutation: str) -> None:
    payload = _payload()
    if mutation == "missing":
        payload["controls"].pop()
    elif mutation == "extra":
        payload["controls"].append(copy.deepcopy(payload["controls"][-1]))
    else:
        payload["controls"][1], payload["controls"][2] = (
            payload["controls"][2],
            payload["controls"][1],
        )
    code = "HOLD_PRETEST_CONTROL_SET" if mutation != "reordered" else "HOLD_PRETEST_CONTROL_SCHEMA"
    _assert_hold(lambda: validate_pretest_control_registry(payload), code)


@pytest.mark.parametrize(
    "field",
    [
        "development_state_ref",
        "development_prediction_ref",
        "development_verifier_ref",
        "verifier_id",
        "evidence_bundle_sha256",
    ],
)
def test_completed_control_requires_every_exact_evidence_binding(
    run_root: Path,
    field: str,
) -> None:
    payload = _payload()
    payload["controls"][1][field] = None
    with pytest.raises(PretestControlRegistryError):
        validate_pretest_control_registry(payload)


def test_completed_control_bundle_and_verifier_identity_are_exact(run_root: Path) -> None:
    payload = _payload()
    payload["controls"][1]["evidence_bundle_sha256"] = _sha("stale-bundle")
    _assert_hold(
        lambda: validate_pretest_control_registry(payload),
        "HOLD_PRETEST_CONTROL_BINDING",
    )

    payload = _payload()
    payload["controls"][1]["verifier_id"] = "CALLER_ALWAYS_PASS"
    _assert_hold(
        lambda: validate_pretest_control_registry(payload),
        "HOLD_PRETEST_CONTROL_SCHEMA",
    )


def test_not_run_cannot_carry_completed_evidence(run_root: Path) -> None:
    payload = _payload(all_not_run=True)
    payload["controls"][1]["development_prediction_ref"] = _ref(
        f"{_control_root(REQUIRED_CONTROL_IDS[1])}/development_predictions.jsonl",
        "pretend-completed",
    )
    _assert_hold(
        lambda: validate_pretest_control_registry(payload),
        "HOLD_PRETEST_CONTROL_NOT_RUN_EVIDENCE",
    )


def test_not_run_reason_is_bounded_and_cannot_encode_results(run_root: Path) -> None:
    payload = _payload(all_not_run=True)
    payload["controls"][1]["reason_code"] = "SIGN_ACCURACY_0_73_P_VALUE_0_01"
    _assert_hold(
        lambda: validate_pretest_control_registry(payload),
        "HOLD_PRETEST_CONTROL_NOT_RUN_REASON",
    )


@pytest.mark.parametrize(
    "extra_field",
    [
        "ordinary_internal_test_row_binding_sha256",
        "metric",
        "row_support",
        "observed_result",
        "p_value",
    ],
)
def test_evaluation_or_metric_fields_are_not_schema_fields(
    run_root: Path,
    extra_field: str,
) -> None:
    payload = _payload()
    payload["controls"][1][extra_field] = _sha(extra_field)
    _assert_hold(
        lambda: validate_pretest_control_registry(payload),
        "HOLD_PRETEST_CONTROL_SCHEMA",
    )


@pytest.mark.parametrize("token", ["internal_test", "sealed", "restricted", "GSE246381"])
def test_evaluation_or_protected_reference_path_is_rejected(
    run_root: Path,
    token: str,
) -> None:
    payload = _payload()
    payload["method_protocol_ref"]["relative_path"] = f"development/{token}/protocol.json"
    _assert_hold(lambda: validate_pretest_control_registry(payload), "HOLD_PROTECTED_INPUT")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sign_accuracy_gate", 0.59),
        ("sign_accuracy_gate", 0.61),
        ("sign_gate_change_authorized", True),
        ("ordinary_only", False),
        ("sealed_access", True),
        ("downstream_unlock", True),
    ],
)
def test_sign_ordinary_sealed_and_downstream_boundaries_are_frozen(
    run_root: Path,
    field: str,
    value: object,
) -> None:
    payload = _payload()
    payload["boundaries"][field] = value
    _assert_hold(lambda: validate_pretest_control_registry(payload), "HOLD_PRETEST_CONTROL_SCHEMA")


def test_parent_swap_is_explicit_non_unlockable_blocker(run_root: Path) -> None:
    payload = _payload()
    assert payload["namespace_integrity_status"].startswith("BLOCKED_PARENT_DIRECTORY")
    payload["namespace_integrity_status"] = "DESCRIPTOR_PINNED_PARENT_SWAP_PROTECTED"
    _assert_hold(lambda: validate_pretest_control_registry(payload), "HOLD_PRETEST_CONTROL_SCHEMA")


def test_commitment_aliases_across_controls_are_rejected(run_root: Path) -> None:
    payload = _payload()
    payload["controls"][2]["train_control_manifest_ref"]["sha256"] = payload[
        "controls"
    ][1]["train_control_manifest_ref"]["sha256"]
    # Repair that entry's semantic bundle so only cross-entry alias detection remains.
    payload["controls"][2]["evidence_bundle_sha256"] = control_evidence_bundle_sha256(
        payload["controls"][2]
    )
    _assert_hold(lambda: validate_pretest_control_registry(payload), "HOLD_PRETEST_CONTROL_ALIAS")


def test_path_escape_symlink_and_hardlink_fail_closed(run_root: Path) -> None:
    payload = _payload()
    _assert_hold(
        lambda: write_pretest_control_registry_exclusive(
            run_root,
            run_root.parent / "escape.json",
            payload,
        ),
        "HOLD_PRETEST_CONTROL_PATH",
    )

    parent = run_root / REGISTRY_RELATIVE_PATH.parent
    moved = run_root / "moved_pretest_controls"
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    _assert_hold(
        lambda: write_pretest_control_registry_exclusive(
            run_root,
            run_root / REGISTRY_RELATIVE_PATH,
            payload,
        ),
        "HOLD_PRETEST_CONTROL_PATH",
    )


def test_noncanonical_hash_mismatch_and_hardlink_are_rejected(run_root: Path) -> None:
    payload = _payload()
    path = run_root / REGISTRY_RELATIVE_PATH
    compact = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(compact)
    _assert_hold(
        lambda: load_pretest_control_registry_exact(
            run_root,
            path,
            hashlib.sha256(compact).hexdigest(),
        ),
        "HOLD_PRETEST_CONTROL_NONCANONICAL",
    )
    _assert_hold(
        lambda: load_pretest_control_registry_exact(run_root, path, _sha("wrong")),
        "HOLD_PRETEST_CONTROL_HASH_MISMATCH",
    )

    path.unlink()
    target = path.parent / "target.json"
    raw = canonical_pretest_control_registry_bytes(payload)
    target.write_bytes(raw)
    os.link(target, path)
    _assert_hold(
        lambda: load_pretest_control_registry_exact(
            run_root,
            path,
            hashlib.sha256(raw).hexdigest(),
        ),
        "HOLD_PRETEST_CONTROL_HASH_MISMATCH",
    )


def test_required_run_control_without_registered_verifier_holds(run_root: Path) -> None:
    _, path, digest = _write_registry(run_root)
    _assert_hold(
        lambda: require_pretest_control_registry(run_root, path, digest),
        "HOLD_PRETEST_CONTROL_VERIFIER_NOT_CONNECTED",
    )


def test_all_not_run_holds_at_inactive_broken_pair(run_root: Path) -> None:
    _, path, digest = _write_registry(run_root, _payload(all_not_run=True))
    _assert_hold(
        lambda: require_pretest_control_registry(run_root, path, digest),
        "HOLD_BROKEN_PAIR_CONTROL_PROPOSED_NOT_ACTIVE",
    )


def test_exact_fixture_verifiers_still_cannot_overcome_inactive_broken_pair(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, path, digest = _write_registry(run_root)
    verifier_map = {}
    for entry in payload["controls"]:
        if entry["disposition"] != CONTROL_RUN:
            continue
        verifier_map[entry["verifier_id"]] = (
            lambda _root, candidate, registry=payload: registry_module._expected_verifier_evidence(
                registry,
                candidate,
            )
        )
    monkeypatch.setattr(
        registry_module,
        "_DEVELOPMENT_CONTROL_VERIFIER_REGISTRY",
        MappingProxyType(verifier_map),
    )
    _assert_hold(
        lambda: require_pretest_control_registry(run_root, path, digest),
        "HOLD_BROKEN_PAIR_CONTROL_PROPOSED_NOT_ACTIVE",
    )


def test_mismatched_verifier_evidence_holds(run_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload, path, digest = _write_registry(run_root)
    first_run = payload["controls"][1]
    bad = registry_module._expected_verifier_evidence(payload, first_run)
    bad["development_only_replay_passed"] = False
    monkeypatch.setattr(
        registry_module,
        "_DEVELOPMENT_CONTROL_VERIFIER_REGISTRY",
        MappingProxyType({first_run["verifier_id"]: lambda _root, _entry: bad}),
    )
    _assert_hold(
        lambda: require_pretest_control_registry(run_root, path, digest),
        "HOLD_PRETEST_CONTROL_VERIFIER_REPLAY",
    )


def test_runtime_does_not_accept_caller_registry_or_evaluation_inputs() -> None:
    parameters = inspect.signature(require_pretest_control_registry).parameters
    assert "registry" not in parameters
    assert "verifier" not in parameters
    assert "internal_test" not in " ".join(parameters)
    assert "metric" not in " ".join(parameters)
