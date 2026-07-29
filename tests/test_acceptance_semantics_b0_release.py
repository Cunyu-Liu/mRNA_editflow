from __future__ import annotations

import copy
from collections.abc import Callable

import pytest

from scripts.data.validate_b0_acceptance import validate_b0_acceptance
from scripts.execution.acceptance_semantics import validate_phase_acceptance
from tests.test_b0_tracks_and_acceptance_v2 import (
    _acceptance_bundle,
    _clean_privileged_evidence,
    _clean_track_audit,
)


FOUNDATION_PENDING = {
    "status": "UNKNOWN_PENDING_FM0",
    "foundation_selected": False,
    "checkpoint_sha256": None,
    "corpus_manifest_sha256": None,
    "audit_report_sha256": None,
    "clearance_evidence_complete": False,
    "allowed_claim": "NONE",
    "re_audit_required": True,
    "gate_applicable": False,
    "gate_passed": True,
}


def _production_reports() -> tuple[list[dict], list[dict]]:
    manifests, reports = _acceptance_bundle()
    for report_index, report in enumerate(reports):
        report.pop("metadata_axis_status")
        report["schema_version"] = "utr_b0_leakage_report.v2"
        report["partition_count"] = len(report["partitions"])
        report["gate_passed"] = True
        report["common_universe_binding"] = {
            "full_record_count": report["canonical_record_count"],
            "full_record_ids_sha256": report["canonical_record_ids_sha256"],
            "full_record_universe_sha256": report["structural_content_sha256"],
        }
        report["foundation_pretraining_overlap"] = copy.deepcopy(FOUNDATION_PENDING)
        report["structural_issues"] = []
        report["structural_records_path"] = "/b0/structural_records.jsonl"
        report["canonical_records_path"] = "/d1/canonical_records.jsonl"
        report["split_manifest_path"] = f"/b0/splits/split-{report_index}.json"
        report["split_manifest_bytes"] = report_index + 1
        report["foundation_exposure_path"] = None
        report["foundation_exposure_sha256"] = None
        report["recomputed_from_bound_structural_records"] = True
        report["canonical_manifest_exact_recomputation"] = True
        report["canonical_manifest_core_sha256"] = f"{report_index + 10:064x}"
        report["auditor_binding"] = {
            "schema_version": "utr_b0_leakage_auditor.v2",
            "entrypoint_path": "/repo/scripts/data/audit_b0_leakage.py",
            "entrypoint_sha256": "a" * 64,
            "canonical_auditor_path": ("/repo/data/utr_benchmark_v2/leakage.py"),
            "canonical_auditor_sha256": "b" * 64,
        }
        report["counts"].update(
            {
                "record_role_overlap_count": 0,
                "component_role_overlap_count": 0,
                "frozen_universe_issue_count": 0,
            }
        )
        for partition in report["partitions"]:
            heldout_study = (
                partition["partition_id"].split(":", 1)[1]
                if report["split_kind"] == "study_disjoint"
                else None
            )
            partition.update(
                {
                    "schema_version": ("utr_b0_partition_leakage_report.v2"),
                    "split_kind": report["split_kind"],
                    "region": report["region"],
                    "source_region": report["source_region"],
                    "target_region": report["target_region"],
                    "heldout_study": heldout_study,
                    "gate_passed": True,
                    "foundation_pretraining_overlap": copy.deepcopy(FOUNDATION_PENDING),
                    "examples": {},
                }
            )
            partition["counts"].update(
                {
                    "record_role_overlap_count": 0,
                    "component_role_overlap_count": 0,
                    "frozen_universe_issue_count": 0,
                }
            )
    return manifests, reports


def _production_payload(
    *,
    exposure_coverage: object = 1.0,
    track_count: object = 3,
    track_role_ambiguity_count: object = 0,
    expect_gate_passed: bool = True,
) -> dict:
    manifests, reports = _production_reports()
    track_audit = _clean_track_audit()
    track_audit.update(
        {
            "schema_version": "utr_track_role_audit.v2",
            "issues": [],
            "track_count": track_count,
            "track_role_ambiguity_count": track_role_ambiguity_count,
        }
    )
    label_seal, required_artifacts = _clean_privileged_evidence()
    label_seal.update(
        {
            "label_record_ids_sha256": "1" * 64,
            "label_record_count": 1,
        }
    )
    d1_binding = {
        "schema_version": "utr_b0_d1_exposure_binding.v2",
        "gate_passed": True,
        "failures": [],
        "d1_acceptance_path": "/d1/acceptance.json",
        "d1_acceptance_sha256": "b" * 64,
        "d1_build_manifest_path": "/d1/build_manifest.json",
        "d1_build_manifest_sha256": "c" * 64,
        "exposure_ledger_path": "/d1/data/data_exposure_ledger.jsonl",
        "exposure_ledger_bytes": 6,
        "exposure_ledger_sha256": "6" * 64,
        "ledger_semantics_valid": True,
    }
    required_artifacts.update(
        {
            "binding_manifest_path": "/b0/artifact_bindings.json",
            "binding_manifest_sha256": "f" * 64,
            "failures": [],
        }
    )
    required_artifacts["claims"]["schema_valid"] = True
    for index, (name, artifact) in enumerate(
        required_artifacts["artifacts"].items(),
        start=1,
    ):
        artifact["path"] = f"/b0/{name}-{index}"
    required_artifacts["artifacts"]["exposure_ledger"].update(
        {
            "path": d1_binding["exposure_ledger_path"],
            "bytes": d1_binding["exposure_ledger_bytes"],
            "sha256": d1_binding["exposure_ledger_sha256"],
        }
    )
    exposure_audit = {
        "coverage": exposure_coverage,
        "covered": 4,
        "expected": 4,
        "identity_level": "dataset_id",
        "missing": [],
        "extra": [
            "ENCSR854RUF_raw62",
            "GSE145046",
            "GSE149487",
            "GSE173083",
            "GSE207584",
            "GSE291719",
            "GSE330741",
            "MPRAu_processed_ENCSR854RUF",
        ],
        "required_ledger_identity_count": 12,
        "missing_from_required_ledger_scope": [],
        "outside_required_ledger_scope": [],
        "ledger_scope_gate_passed": True,
        "gate_passed": True,
    }
    result = validate_b0_acceptance(
        leakage_reports=reports,
        exposure_audit=exposure_audit,
        track_audit=track_audit,
        split_manifests=manifests,
        track_a_label_seal_audit=label_seal,
        required_artifact_audit=required_artifacts,
        d1_exposure_ledger_binding=d1_binding,
        supplied_leakage_reports_match_recomputation=True,
    )
    assert result["b0_gate_passed"] is expect_gate_passed
    result.update(
        {
            "exposure_ledger": exposure_audit,
            "track_role_audit": track_audit,
            "track_a_label_seal_audit": label_seal,
            "required_artifact_audit": required_artifacts,
            "d1_exposure_ledger_binding": d1_binding,
            "supplied_leakage_reports_match_recomputation": True,
            "recomputed_leakage_reports": reports,
            "supplied_leakage_report_files": [
                {
                    "path": f"/b0/supplied/report-{index}.json",
                    "bytes": index + 1,
                    "sha256": f"{index + 20:064x}",
                }
                for index in range(5)
            ],
        }
    )
    return result


def _errors(payload: dict) -> list[str]:
    return validate_phase_acceptance("B0", payload, require_pass=True)


def test_actual_validator_production_shape_passes_semantics() -> None:
    assert _errors(_production_payload()) == []


@pytest.mark.parametrize(
    ("kwargs", "failed_gate"),
    [
        ({"exposure_coverage": True}, "exposure_ledger_coverage_100_percent"),
        ({"track_count": 3.0}, "exact_three_tracks_present"),
        (
            {"track_role_ambiguity_count": False},
            "track_role_ambiguity_zero",
        ),
    ],
)
def test_generator_rejects_equal_but_wrong_scalar_types(
    kwargs: dict[str, object],
    failed_gate: str,
) -> None:
    payload = _production_payload(
        **kwargs,
        expect_gate_passed=False,
    )

    assert failed_gate in payload["failed_gates"]


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        (
            "leakage_report_count",
            5.0,
            "leakage_report_count must be exact integer 5",
        ),
        (
            "exposure_ledger_coverage",
            True,
            "exposure_ledger_coverage must be exact float 1.0",
        ),
        (
            "track_role_ambiguity_count",
            False,
            "track_role_ambiguity_count must be exact integer 0",
        ),
    ],
)
def test_observed_scalar_types_are_exact(
    field: str,
    value: object,
    needle: str,
) -> None:
    payload = _production_payload()
    payload["observed"][field] = value

    errors = _errors(payload)

    assert any(needle in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "value", "needle"),
    [
        (
            "exposure_ledger",
            "coverage",
            True,
            "exposure ledger audit did not pass",
        ),
        (
            "exposure_ledger",
            "coverage",
            1,
            "exposure ledger audit did not pass",
        ),
        (
            "exposure_ledger",
            "required_ledger_identity_count",
            12.0,
            "dataset exposure ledger scope is not exact",
        ),
        (
            "track_role_audit",
            "track_count",
            3.0,
            "track-role audit did not pass",
        ),
        (
            "track_role_audit",
            "track_role_ambiguity_count",
            False,
            "track-role audit did not pass",
        ),
    ],
)
def test_bound_source_scalar_types_are_exact(
    section: str,
    field: str,
    value: object,
    needle: str,
) -> None:
    payload = _production_payload()
    payload[section][field] = value

    errors = _errors(payload)

    assert any(needle in error for error in errors)


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (
            lambda payload: payload["track_role_audit"].__setitem__(
                "gate_passed", False
            ),
            "track-role audit did not pass",
        ),
        (
            lambda payload: payload["required_artifact_audit"]["artifacts"][
                "exposure_ledger"
            ].__setitem__("sha256", "9" * 64),
            "differs from its D1 binding",
        ),
        (
            lambda payload: payload["recomputed_leakage_reports"][0][
                "acceptance_gates"
            ].__setitem__("path_leakage_zero", False),
            "acceptance gates are not exact and all true",
        ),
        (
            lambda payload: payload.__setitem__(
                "supplied_leakage_reports_match_recomputation", False
            ),
            "differ from recomputation",
        ),
        (
            lambda payload: payload.__setitem__("claim_boundary", "B0 proves efficacy"),
            "claim_boundary is invalid",
        ),
        (
            lambda payload: payload["observed"]["foundation_states"][0].__setitem__(
                "gate_applicable", True
            ),
            "not exact UNKNOWN_PENDING_FM0 evidence",
        ),
    ],
)
def test_true_b0_gate_binding_recomputation_and_claim_drift_fail(
    mutate: Callable[[dict], None],
    needle: str,
) -> None:
    payload = _production_payload()
    mutate(payload)

    errors = _errors(payload)

    assert any(needle in error for error in errors)


def test_pending_fm0_false_descriptors_are_not_gate_failures() -> None:
    payload = _production_payload()
    false_paths = [
        state
        for state in payload["observed"]["foundation_states"]
        if state["foundation_selected"] is False
        and state["clearance_evidence_complete"] is False
        and state["gate_applicable"] is False
    ]

    assert len(false_paths) == 5
    assert _errors(payload) == []


def test_report_counts_require_exact_production_inventory() -> None:
    payload = _production_payload()
    del payload["recomputed_leakage_reports"][0]["counts"]["metadata_overlap_count"]

    errors = _errors(payload)

    assert any("counts field inventory is not exact" in error for error in errors)


def test_report_counts_reject_bool_values() -> None:
    payload = _production_payload()
    payload["recomputed_leakage_reports"][0]["counts"]["metadata_overlap_count"] = True

    errors = _errors(payload)

    assert any(
        "metadata_overlap_count is not a non-negative integer" in error
        for error in errors
    )


def test_partition_role_overlap_count_must_be_zero() -> None:
    payload = _production_payload()
    payload["recomputed_leakage_reports"][0]["partitions"][0]["counts"][
        "record_role_overlap_count"
    ] = 1

    errors = _errors(payload)

    assert any("record_role_overlap_count is not zero" in error for error in errors)


def test_report_counts_must_equal_partition_aggregate() -> None:
    payload = _production_payload()
    counts = payload["recomputed_leakage_reports"][0]["counts"]
    counts["metadata_overlap_count"] = 1
    counts["explained_metadata_overlap_count"] = 1

    errors = _errors(payload)

    assert any(
        "counts do not equal its partition aggregate" in error for error in errors
    )


@pytest.mark.parametrize("partition_count", [2.0, True])
def test_partition_count_requires_exact_int(partition_count: float | bool) -> None:
    payload = _production_payload()
    report = payload["recomputed_leakage_reports"][1]
    report["partition_count"] = partition_count

    errors = _errors(payload)

    assert any("partition count is invalid" in error for error in errors)


@pytest.mark.parametrize(
    ("report_index", "heldout_study"),
    [
        (1, None),
        (0, "GSE114002"),
    ],
)
def test_heldout_study_is_derived_from_split_identity(
    report_index: int,
    heldout_study: str | None,
) -> None:
    payload = _production_payload()
    payload["recomputed_leakage_reports"][report_index]["partitions"][0][
        "heldout_study"
    ] = heldout_study

    errors = _errors(payload)

    assert any(
        "heldout_study differs from its partition identity" in error for error in errors
    )
