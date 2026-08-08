from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.r3x.artifact_graph import make_artifact_ref
from scripts.r3x.artifact_io import atomic_publish_bytes, atomic_write_json_exclusive
from scripts.r3x.protocol import REQUIRED_ACTIVATION_AUTHORITIES
from scripts.r3x.run_complete import (
    ORDINARY_GO_CANDIDATE,
    SCIENTIFIC_HOLD,
    VERIFIER_ROLES,
    RunCompleteProtocolError,
    build_run_complete,
    replay_run_complete,
    replay_run_complete_payload,
    validate_run_complete,
    write_run_complete_exclusive,
)


METHOD_IDS = ["A_METHOD", "B_METHOD"]


def _publish_json(root: Path, relative: str, payload: dict) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = atomic_write_json_exclusive(path, payload)
    return make_artifact_ref(root, path, digest)


def _publish_bytes(root: Path, relative: str, raw: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = atomic_publish_bytes(path, raw)
    return make_artifact_ref(root, path, digest)


def _fixture(root: Path, *, status: str = SCIENTIFIC_HOLD) -> dict:
    root.mkdir()
    authority_refs = {
        name: _publish_bytes(
            root,
            f"authority_snapshots/{name}.artifact",
            f"frozen-authority:{name}".encode("utf-8"),
        )
        for name in REQUIRED_ACTIVATION_AUTHORITIES
    }
    authority_hashes = {
        f"{name}_sha256": authority_refs[name]["sha256"]
        for name in REQUIRED_ACTIVATION_AUTHORITIES
    }
    activation_ref = _publish_json(
        root,
        "authority_snapshots/ACTIVATION.json",
        {
            "schema_version": "r3_activation_manifest_v1",
            "status": "FROZEN",
            "mode": "ORDINARY_ONLY",
            "authorities": {
                name: {
                    "path": authority_refs[name]["relative_path"],
                    "sha256": authority_refs[name]["sha256"],
                    "status": "FROZEN",
                }
                for name in REQUIRED_ACTIVATION_AUTHORITIES
            },
        },
    )
    run_plan_ref = _publish_json(
        root,
        "R3X_RUN_PLAN.json",
        {
            "schema_version": "r3x_run_plan_v1",
            "activation_manifest_sha256": activation_ref["sha256"],
            "authority_hashes": authority_hashes,
            "ordinary_only": True,
            "sealed_access": False,
            "downstream_unlock": False,
        },
    )
    run_start_ref = _publish_json(
        root,
        "R3X_RUN_START.json",
        {
            "schema_version": "r3x_run_start_v1",
            "run_plan_sha256": run_plan_ref["sha256"],
            "activation_manifest_sha256": activation_ref["sha256"],
            "authority_hashes": authority_hashes,
            "sealed_access": False,
            "downstream_unlock": False,
        },
    )

    methods = {}
    for method_id in METHOD_IDS:
        methods[method_id] = {
            "method_id": method_id,
            "global_prediction_ref": _publish_json(
                root,
                f"methods/{method_id}/GLOBAL_PREDICTION_COMMIT.json",
                {
                    "schema_version": "fixture_global_prediction_v1",
                    "method_id": method_id,
                    "run_start_sha256": run_start_ref["sha256"],
                },
            ),
            "metrics_ref": _publish_json(
                root,
                f"methods/{method_id}/METRICS.json",
                {
                    "schema_version": "fixture_metrics_v1",
                    "method_id": method_id,
                    "macro_sign_accuracy": 0.65,
                },
            ),
            "checkpoint_ensemble_refs": {
                "primary_estimand": _publish_json(
                    root,
                    f"methods/{method_id}/primary_estimand/ENSEMBLE.json",
                    {
                        "schema_version": "fixture_ensemble_v1",
                        "method_id": method_id,
                        "run_start_sha256": run_start_ref["sha256"],
                    },
                )
            },
        }

    paired_ref = _publish_json(
        root,
        "R3X_PAIRED_COMPARISON.json",
        {"schema_version": "fixture_paired_v1", "status": "PASS"},
    )
    negative_ref = _publish_json(
        root,
        "R3X_NEGATIVE_CONTROL_SUMMARY.json",
        {
            "schema_version": "fixture_negative_controls_v1",
            "status": "ORDINARY_ONLY_COMPLETE",
            "sealed_access": False,
            "downstream_unlock": False,
        },
    )
    go = status == ORDINARY_GO_CANDIDATE
    report_ref = _publish_json(
        root,
        "R3X_REPORT.json",
        {
            "schema_version": "r3x_ordinary_run_report_v1",
            "status": (
                "ORDINARY_GO_EFFECT_CANDIDATE_EXISTS"
                if go
                else "ORDINARY_RUN_COMPLETE_SCIENTIFIC_HOLD"
            ),
            "activation_manifest_sha256": activation_ref["sha256"],
            "run_plan_sha256": run_plan_ref["sha256"],
            "run_start_sha256": run_start_ref["sha256"],
            "methods": {method_id: {"status": "RUN"} for method_id in METHOD_IDS},
            "GO_EFFECT": go,
            "sealed_access": False,
            "downstream_unlock": False,
        },
    )
    return build_run_complete(
        status=status,
        run_plan_ref=run_plan_ref,
        run_start_ref=run_start_ref,
        activation_manifest_ref=activation_ref,
        authority_refs=authority_refs,
        required_method_ids=METHOD_IDS,
        method_artifacts=methods,
        paired_artifact_ref=paired_ref,
        report_ref=report_ref,
        negative_control_summary_ref=negative_ref,
    )


def _all_verifiers():
    return {
        role: (lambda raw, metadata: bool(raw) and metadata["role"] in VERIFIER_ROLES)
        for role in VERIFIER_ROLES
    }


def test_hold_commit_replays_every_ref_and_is_atomic_no_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ordinary_run"
    payload = _fixture(root)
    output = root / "RUN_COMPLETE.json"
    digest = write_run_complete_exclusive(output, payload, run_root=root)
    reference = make_artifact_ref(root, output, digest)

    result = replay_run_complete(root, reference)
    assert result["declared_status"] == SCIENTIFIC_HOLD
    assert result["effective_status"] == SCIENTIFIC_HOLD
    assert result["go_candidate_verified"] is False
    assert result["hold_reasons"] == []
    assert result["verified_reference_count"] == 23
    assert result["sealed_access"] is False
    assert result["downstream_unlock"] is False

    original = output.read_bytes()
    with pytest.raises(RunCompleteProtocolError, match="already exists"):
        write_run_complete_exclusive(output, payload, run_root=root)
    assert output.read_bytes() == original


def test_schema_rejects_missing_refs_extra_fields_and_method_omission(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ordinary_run"
    payload = _fixture(root)

    extra = copy.deepcopy(payload)
    extra["metadata"] = {"claim": "GO"}
    with pytest.raises(RunCompleteProtocolError, match="strict schema"):
        validate_run_complete(extra)

    missing_metrics = copy.deepcopy(payload)
    del missing_metrics["method_artifacts"]["A_METHOD"]["metrics_ref"]
    with pytest.raises(RunCompleteProtocolError, match="strict schema"):
        validate_run_complete(missing_metrics)

    omitted_method = copy.deepcopy(payload)
    del omitted_method["method_artifacts"]["B_METHOD"]
    with pytest.raises(RunCompleteProtocolError, match="exactly cover"):
        validate_run_complete(omitted_method)

    missing_authority = copy.deepcopy(payload)
    del missing_authority["authority_refs"]["ready_dataset"]
    with pytest.raises(RunCompleteProtocolError, match="eleven"):
        validate_run_complete(missing_authority)


@pytest.mark.parametrize(
    "relative_path",
    ["../escape.json", "/outside.json", "sealed/evidence.json"],
)
def test_every_reference_uses_safe_ordinary_relative_paths(
    tmp_path: Path, relative_path: str
) -> None:
    root = tmp_path / "ordinary_run"
    payload = _fixture(root)
    payload["paired_artifact_ref"] = {
        "relative_path": relative_path,
        "sha256": "a" * 64,
    }
    with pytest.raises(RunCompleteProtocolError, match="paired_artifact_ref"):
        validate_run_complete(payload)


def test_tampered_and_missing_referenced_bytes_fail_before_commit(tmp_path: Path) -> None:
    root = tmp_path / "ordinary_run"
    payload = _fixture(root)
    paired_path = root / payload["paired_artifact_ref"]["relative_path"]
    paired_path.write_bytes(b"tampered-in-place")
    output = root / "RUN_COMPLETE.json"
    with pytest.raises(RunCompleteProtocolError, match="SHA-256 mismatch"):
        write_run_complete_exclusive(output, payload, run_root=root)
    assert not output.exists()

    second_root = tmp_path / "ordinary_run_missing"
    missing = _fixture(second_root)
    missing["method_artifacts"]["A_METHOD"]["metrics_ref"] = {
        "relative_path": "methods/A_METHOD/MISSING.json",
        "sha256": "f" * 64,
    }
    missing = validate_run_complete(missing)
    missing_output = second_root / "RUN_COMPLETE.json"
    with pytest.raises(RunCompleteProtocolError, match="does not exist"):
        write_run_complete_exclusive(
            missing_output, missing, run_root=second_root
        )
    assert not missing_output.exists()


def test_report_status_and_go_must_exactly_match_complete(tmp_path: Path) -> None:
    root = tmp_path / "ordinary_run"
    payload = _fixture(root)
    payload["status"] = ORDINARY_GO_CANDIDATE
    payload = validate_run_complete(payload)
    with pytest.raises(RunCompleteProtocolError, match="status is inconsistent"):
        replay_run_complete_payload(root, payload)


def test_go_claim_fails_closed_without_full_external_replay(tmp_path: Path) -> None:
    root = tmp_path / "ordinary_run"
    payload = _fixture(root, status=ORDINARY_GO_CANDIDATE)

    replay = replay_run_complete_payload(root, payload)
    assert replay["declared_status"] == ORDINARY_GO_CANDIDATE
    assert replay["effective_status"] == SCIENTIFIC_HOLD
    assert replay["go_candidate_verified"] is False
    assert {
        reason.split(":", 1)[0] for reason in replay["hold_reasons"]
    } == {
        "EXPECTED_METHOD_SET_REQUIRED_FOR_GO",
        "COMPLETE_EXTERNAL_VERIFIER_REGISTRY_REQUIRED_FOR_GO",
        "NONEMPTY_EXTERNAL_VERIFIER_CONTEXT_REQUIRED_FOR_GO",
    }

    refused = root / "RUN_COMPLETE_REFUSED_GO.json"
    with pytest.raises(RunCompleteProtocolError, match="failed closed"):
        write_run_complete_exclusive(refused, payload, run_root=root)
    assert not refused.exists()

    incomplete = _all_verifiers()
    del incomplete["report"]
    still_hold = replay_run_complete_payload(
        root,
        payload,
        expected_method_ids=METHOD_IDS,
        verifier_callbacks=incomplete,
        verifier_context={"ordinary_replay": True},
    )
    assert still_hold["effective_status"] == SCIENTIFIC_HOLD
    assert "report" in still_hold["hold_reasons"][0]


def test_go_candidate_requires_expected_methods_callbacks_and_context(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ordinary_run"
    payload = _fixture(root, status=ORDINARY_GO_CANDIDATE)
    callbacks = _all_verifiers()
    output = root / "RUN_COMPLETE.json"
    digest = write_run_complete_exclusive(
        output,
        payload,
        run_root=root,
        expected_method_ids=METHOD_IDS,
        verifier_callbacks=callbacks,
        verifier_context={"ordinary_replay": "full"},
    )
    reference = make_artifact_ref(root, output, digest)
    replay = replay_run_complete(
        root,
        reference,
        expected_method_ids=METHOD_IDS,
        verifier_callbacks=callbacks,
        verifier_context={"ordinary_replay": "full"},
    )
    assert replay["effective_status"] == ORDINARY_GO_CANDIDATE
    assert replay["go_candidate_verified"] is True
    assert replay["downstream_unlock"] is False
    assert replay["sealed_access"] is False

    with pytest.raises(RunCompleteProtocolError, match="expected method set"):
        replay_run_complete_payload(
            root,
            payload,
            expected_method_ids=["A_METHOD"],
            verifier_callbacks=callbacks,
            verifier_context={"ordinary_replay": "full"},
        )


def test_callback_rejection_and_self_reference_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "ordinary_run"
    payload = _fixture(root)
    with pytest.raises(RunCompleteProtocolError, match="rejected metrics"):
        replay_run_complete_payload(
            root,
            payload,
            verifier_callbacks={"metrics": lambda raw, metadata: False},
        )

    self_ref = copy.deepcopy(payload)
    self_ref["paired_artifact_ref"] = {
        "relative_path": "RUN_COMPLETE.json",
        "sha256": "a" * 64,
    }
    with pytest.raises(RunCompleteProtocolError, match="own output path"):
        write_run_complete_exclusive(
            root / "RUN_COMPLETE.json", self_ref, run_root=root
        )


def test_duplicate_json_key_in_run_complete_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ordinary_run"
    root.mkdir()
    raw = (
        b'{"schema_version":"r3x_run_complete_v1",'
        b'"schema_version":"r3x_run_complete_v1"}\n'
    )
    path = root / "RUN_COMPLETE.json"
    digest = atomic_publish_bytes(path, raw)
    reference = make_artifact_ref(root, path, digest)
    with pytest.raises(RunCompleteProtocolError, match="duplicate JSON key"):
        replay_run_complete(root, reference)
