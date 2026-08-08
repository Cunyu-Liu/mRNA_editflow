from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.r3x.artifact_graph import make_artifact_ref  # noqa: E402
from scripts.r3x.artifact_io import (  # noqa: E402
    atomic_publish_bytes,
    atomic_write_json_exclusive,
)
from scripts.r3x.control_summary import (  # noqa: E402
    ACTION_SUPPORT_QUALIFIED,
    ACTION_SUPPORT_NOT_QUALIFIED,
    ALGEBRAIC_INVARIANT,
    ARTIFACT_ROLE,
    B_INTERACTION,
    CONTROL_IDS,
    CONTROL_SPECS,
    DATA_CRITERION,
    HOLD,
    NegativeControlProtocolError,
    PASS,
    PASS_ALL,
    SCHEMA,
    TRAIN_DATA_PERTURBATION,
    build_not_run_negative_control_summary,
    load_negative_control_summary,
    validate_negative_control_summary,
    verify_negative_control_summary_references,
    write_negative_control_summary_exclusive,
)


TEST_BINDING = hashlib.sha256(b"ordinary-internal-test-row-binding").hexdigest()


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


def _method_protocol() -> dict:
    return {
        "schema_version": "r3_ordinary_experiment_protocol_v1",
        "negative_controls": list(CONTROL_IDS),
        "hard_boundaries": {
            "ordinary_only": True,
            "sealed_member_access": False,
            "O0_G1_E0_unlock": False,
        },
    }


def _manifest(control_id: str) -> dict:
    return {
        "schema_version": "R3X_NEGATIVE_CONTROL_MANIFEST_V1",
        "name": control_id,
        "scope": "ORDINARY_TRAIN_PARTITION_ONLY",
        "seed": 11,
        "n_rows": 20,
        "ordered_input_row_ids_sha256": hashlib.sha256(
            f"{control_id}-ordered".encode()
        ).hexdigest(),
        "input_sha256": hashlib.sha256(f"{control_id}-input".encode()).hexdigest(),
        "output_sha256": hashlib.sha256(f"{control_id}-output".encode()).hexdigest(),
        "partition_validation": "ALL_INPUT_AND_OUTPUT_ROWS_EQUAL_TRAIN",
        "test_partition_modified": False,
        "sealed_access": False,
    }


def _expected_methods(
    control_id: str, *, qualified: bool = False
) -> tuple[list[str], list[str]]:
    spec = CONTROL_SPECS[control_id]
    required = list(spec["base_methods"])
    if qualified and spec["optional_interaction"]:
        required.append(B_INTERACTION)
    excluded = (
        [B_INTERACTION]
        if spec["optional_interaction"] and not qualified
        else []
    )
    return required, excluded


def _action_policy() -> dict:
    return {
        "schema_version": "r3_action_support_policy_v1",
        "status": "FROZEN",
        "qualification_status": "PUBLICATION_QUALIFIED",
        "authority_hashes": {
            "ready_dataset_sha256": "1" * 64,
            "inclusion_manifest_sha256": "2" * 64,
            "context_source_crosswalk_sha256": "3" * 64,
            "endpoint_estimand_registry_sha256": "4" * 64,
            "source_component_split_manifest_sha256": "5" * 64,
        },
        "thresholds": {
            "min_independent_source_components_per_action": 20,
            "min_study_context_cells_per_action": 2,
            "max_single_component_dominance": 0.20,
        },
        "observed": {
            "min_independent_source_components_per_action": 25,
            "min_study_context_cells_per_action": 3,
            "max_single_component_dominance": 0.10,
            "qualified_action_strata": 8,
        },
    }


def _summary_fixture(root: Path, *, qualified: bool = False) -> dict:
    protocol_ref = _publish_json(root, "authority/method-protocol.json", _method_protocol())
    action_ref = (
        _publish_json(root, "authority/action-support.json", _action_policy())
        if qualified
        else None
    )
    controls = []
    for control_id in CONTROL_IDS:
        spec = CONTROL_SPECS[control_id]
        required, excluded = _expected_methods(control_id, qualified=qualified)
        manifest_ref = None
        if spec["control_class"] == TRAIN_DATA_PERTURBATION:
            manifest_ref = _publish_json(
                root, f"controls/{control_id}/train-manifest.json", _manifest(control_id)
            )
        results = []
        for method_id in required:
            evidence_ref = _publish_bytes(
                root,
                f"controls/{control_id}/evidence/{method_id}.bin",
                f"independent evidence for {control_id}/{method_id}\n".encode(),
            )
            results.append(
                {
                    "method_id": method_id,
                    "outcome": PASS,
                    "evidence_ref": evidence_ref,
                    "ordinary_internal_test_row_binding_sha256": (
                        None if spec["control_class"] == ALGEBRAIC_INVARIANT else TEST_BINDING
                    ),
                    "reason_code": None,
                }
            )
        controls.append(
            {
                "control_id": control_id,
                "control_class": spec["control_class"],
                "required_method_ids": required,
                "conditionally_excluded_method_ids": excluded,
                "outcome_criterion": spec["outcome_criterion"],
                "train_only_invariant": spec["train_only_invariant"],
                "train_control_manifest_ref": manifest_ref,
                "method_results": results,
                "status": "NOT_APPLICABLE" if not required else PASS,
            }
        )
    return {
        "schema_version": SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "method_protocol_ref": protocol_ref,
        "action_support_status": (
            ACTION_SUPPORT_QUALIFIED if qualified else ACTION_SUPPORT_NOT_QUALIFIED
        ),
        "action_support_policy_ref": action_ref,
        "ordinary_internal_test_row_binding_sha256": TEST_BINDING,
        "controls": controls,
        "overall_status": PASS_ALL,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }


def _verifier(raw: bytes, metadata: dict) -> dict:
    assert raw.startswith(b"independent evidence")
    is_data = metadata["control_class"] == TRAIN_DATA_PERTURBATION
    is_algebraic = metadata["control_class"] == ALGEBRAIC_INVARIANT
    return {
        "control_id": metadata["control_id"],
        "method_id": metadata["method_id"],
        "outcome": metadata["expected_outcome"],
        "ordinary_internal_test_row_binding_sha256": (
            None if is_algebraic else metadata["ordinary_internal_test_row_binding_sha256"]
        ),
        "train_only_verified": True if is_data else None,
        "train_control_manifest_sha256": (
            metadata["train_control_manifest_sha256"] if is_data else None
        ),
        "action_support_policy_sha256": (
            metadata["action_support_policy_sha256"]
            if metadata["method_id"] == B_INTERACTION
            else None
        ),
        "verification_scope": metadata["required_verification_scope"],
        "verified_partitions": metadata["required_verified_partitions"],
    }


def test_all_applicable_controls_require_exact_evidence_replay(tmp_path: Path) -> None:
    summary = _summary_fixture(tmp_path)
    normalized = verify_negative_control_summary_references(
        tmp_path, summary, evidence_verifier=_verifier
    )
    assert normalized["overall_status"] == PASS_ALL
    assert normalized["controls"][-1]["status"] == "NOT_APPLICABLE"
    assert normalized["controls"][-1]["conditionally_excluded_method_ids"] == [
        B_INTERACTION
    ]

    output = tmp_path / "closure" / "negative-control-summary.json"
    output.parent.mkdir()
    digest = write_negative_control_summary_exclusive(
        output,
        summary,
        run_root=tmp_path,
        evidence_verifier=_verifier,
    )
    loaded = load_negative_control_summary(
        output,
        digest,
        run_root=tmp_path,
        evidence_verifier=_verifier,
    )
    assert loaded == normalized
    with pytest.raises(NegativeControlProtocolError, match="already exists"):
        write_negative_control_summary_exclusive(
            output,
            summary,
            run_root=tmp_path,
            evidence_verifier=_verifier,
        )


def test_b_interaction_is_required_only_after_action_support_is_qualified(
    tmp_path: Path,
) -> None:
    summary = _summary_fixture(tmp_path, qualified=True)
    normalized = verify_negative_control_summary_references(
        tmp_path, summary, evidence_verifier=_verifier
    )
    assert normalized["overall_status"] == PASS_ALL
    for control in normalized["controls"]:
        if CONTROL_SPECS[control["control_id"]]["optional_interaction"]:
            assert B_INTERACTION in control["required_method_ids"]
            assert control["conditionally_excluded_method_ids"] == []
    assert normalized["controls"][-1]["status"] == PASS

    bad = copy.deepcopy(summary)
    bad["action_support_status"] = ACTION_SUPPORT_NOT_QUALIFIED
    with pytest.raises(NegativeControlProtocolError, match="cannot carry"):
        validate_negative_control_summary(bad)


def test_self_attested_bool_and_forged_pass_are_rejected(tmp_path: Path) -> None:
    summary = _summary_fixture(tmp_path)
    with pytest.raises(NegativeControlProtocolError, match="structured result"):
        verify_negative_control_summary_references(
            tmp_path,
            summary,
            evidence_verifier=lambda raw, metadata: True,
        )

    def contradict(raw: bytes, metadata: dict) -> dict:
        result = _verifier(raw, metadata)
        result["outcome"] = "FAIL"
        return result

    with pytest.raises(NegativeControlProtocolError, match="contradicts claimed outcome"):
        verify_negative_control_summary_references(
            tmp_path, summary, evidence_verifier=contradict
        )

    def development_only(raw: bytes, metadata: dict) -> dict:
        result = _verifier(raw, metadata)
        if metadata["control_class"] == TRAIN_DATA_PERTURBATION:
            result["verified_partitions"] = ["ORDINARY_DEVELOPMENT"]
        return result

    with pytest.raises(NegativeControlProtocolError, match="required partitions"):
        verify_negative_control_summary_references(
            tmp_path, summary, evidence_verifier=development_only
        )


def test_pass_without_evidence_or_independent_verifier_is_rejected(tmp_path: Path) -> None:
    summary = _summary_fixture(tmp_path)
    summary["controls"][0]["method_results"][0]["evidence_ref"] = None
    with pytest.raises(NegativeControlProtocolError, match="requires immutable evidence_ref"):
        validate_negative_control_summary(summary)

    summary = _summary_fixture(tmp_path / "second")
    with pytest.raises(NegativeControlProtocolError, match="independent verifier"):
        verify_negative_control_summary_references(
            tmp_path / "second", summary, evidence_verifier=None
        )


@pytest.mark.parametrize("outcome", ["NOT_RUN", "UNDEFINED"])
def test_not_run_or_undefined_cannot_claim_pass_all(tmp_path: Path, outcome: str) -> None:
    summary = _summary_fixture(tmp_path)
    result = summary["controls"][0]["method_results"][0]
    result.update(
        {
            "outcome": outcome,
            "evidence_ref": None,
            "ordinary_internal_test_row_binding_sha256": None,
            "reason_code": "CONTROL_EVIDENCE_NOT_AVAILABLE",
        }
    )
    summary["controls"][0]["status"] = HOLD
    with pytest.raises(NegativeControlProtocolError, match="recomputed status HOLD"):
        validate_negative_control_summary(summary)

    summary["overall_status"] = HOLD
    normalized = validate_negative_control_summary(summary)
    assert normalized["overall_status"] == HOLD
    assert normalized["controls"][0]["status"] == HOLD


def test_runner_can_publish_explicit_hold_before_any_control_execution(
    tmp_path: Path,
) -> None:
    protocol_ref = _publish_json(
        tmp_path, "authority/method-protocol.json", _method_protocol()
    )
    summary = build_not_run_negative_control_summary(
        method_protocol_ref=protocol_ref,
        ordinary_internal_test_row_binding_sha256=TEST_BINDING,
        reason_code="CONTROLS_NOT_EXECUTED",
    )
    assert summary["overall_status"] == HOLD
    assert all(
        control["status"] in {HOLD, "NOT_APPLICABLE"}
        for control in summary["controls"]
    )
    assert all(
        result["outcome"] == "NOT_RUN"
        for control in summary["controls"]
        for result in control["method_results"]
    )
    # With no PASS/FAIL claims there is no evidence callback to trust.  The
    # exact method protocol is still replayed and the result remains HOLD.
    replayed = verify_negative_control_summary_references(
        tmp_path, summary, evidence_verifier=None
    )
    assert replayed["overall_status"] == HOLD


def test_modified_internal_test_binding_is_rejected_by_schema_and_replay(tmp_path: Path) -> None:
    summary = _summary_fixture(tmp_path)
    summary["controls"][0]["method_results"][0][
        "ordinary_internal_test_row_binding_sha256"
    ] = hashlib.sha256(b"modified-test").hexdigest()
    with pytest.raises(NegativeControlProtocolError, match="changed ordinary internal-test"):
        validate_negative_control_summary(summary)

    summary = _summary_fixture(tmp_path / "replay")

    def modified_test(raw: bytes, metadata: dict) -> dict:
        result = _verifier(raw, metadata)
        if metadata["control_class"] != ALGEBRAIC_INVARIANT:
            result["ordinary_internal_test_row_binding_sha256"] = hashlib.sha256(
                b"modified-test"
            ).hexdigest()
        return result

    with pytest.raises(NegativeControlProtocolError, match="modified internal-test rows"):
        verify_negative_control_summary_references(
            tmp_path / "replay", summary, evidence_verifier=modified_test
        )


def test_train_manifest_cannot_modify_test_or_self_attest_wrong_scope(tmp_path: Path) -> None:
    summary = _summary_fixture(tmp_path)
    first = summary["controls"][0]
    original = first["train_control_manifest_ref"]
    bad = _manifest(first["control_id"])
    bad["test_partition_modified"] = True
    first["train_control_manifest_ref"] = _publish_json(
        tmp_path, "controls/bad-modified-test-manifest.json", bad
    )
    with pytest.raises(NegativeControlProtocolError, match="modified the test partition"):
        verify_negative_control_summary_references(
            tmp_path, summary, evidence_verifier=_verifier
        )
    assert original != first["train_control_manifest_ref"]


def test_hash_path_and_extra_field_are_fail_closed(tmp_path: Path) -> None:
    summary = _summary_fixture(tmp_path)
    summary["controls"][0]["unexpected"] = True
    with pytest.raises(NegativeControlProtocolError, match="strict schema"):
        validate_negative_control_summary(summary)

    summary = _summary_fixture(tmp_path / "hash")
    summary["method_protocol_ref"]["sha256"] = "0" * 64
    with pytest.raises(NegativeControlProtocolError, match="SHA-256 mismatch"):
        verify_negative_control_summary_references(
            tmp_path / "hash", summary, evidence_verifier=_verifier
        )

    summary = _summary_fixture(tmp_path / "path")
    summary["method_protocol_ref"]["relative_path"] = "../escape.json"
    with pytest.raises(NegativeControlProtocolError, match="canonical safe relative path"):
        validate_negative_control_summary(summary)


def test_protocol_must_list_exact_seven_controls(tmp_path: Path) -> None:
    summary = _summary_fixture(tmp_path)
    wrong_protocol = _method_protocol()
    wrong_protocol["negative_controls"] = list(CONTROL_IDS[:-1])
    summary["method_protocol_ref"] = _publish_json(
        tmp_path, "authority/wrong-method-protocol.json", wrong_protocol
    )
    with pytest.raises(NegativeControlProtocolError, match="exact seven"):
        verify_negative_control_summary_references(
            tmp_path, summary, evidence_verifier=_verifier
        )


def test_data_criterion_and_hard_boundaries_are_frozen(tmp_path: Path) -> None:
    summary = _summary_fixture(tmp_path)
    summary["controls"][0]["outcome_criterion"] = DATA_CRITERION + "_POST_HOC"
    with pytest.raises(NegativeControlProtocolError, match="criterion differs"):
        validate_negative_control_summary(summary)

    summary = _summary_fixture(tmp_path / "boundary")
    summary["sealed_access"] = True
    with pytest.raises(NegativeControlProtocolError, match="sealed"):
        validate_negative_control_summary(summary)
