from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.execution.acceptance_semantics import (
    D1_ACCEPTED_DATASETS,
    D1_BLOCKED_DATASETS,
    validate_phase_acceptance,
)
from tests.governance_fixtures import valid_d1_acceptance


def _production_shaped_acceptance(stage_root: Path) -> dict:
    payload = valid_d1_acceptance(stage_root)
    payload["builder_audit_validation"]["git_prelaunch_snapshot"] = {
        "captured_before_command": True,
        "clean": False,
    }
    payload["config_binding_validation"]["input_inventory_binding"] = {
        "legacy_inference_used": False,
    }
    return payload


def _errors(payload: dict) -> list[str]:
    return validate_phase_acceptance("D1", payload, require_pass=True)


def _provenance_check(payload: dict, *, status: str) -> dict:
    result = next(
        item for item in payload["dataset_results"] if item["status"] == status
    )
    return next(
        check
        for check in result["checks"]
        if check["name"] == "production_input_provenance_complete"
    )


def test_production_shaped_mutually_exclusive_false_values_pass(
    tmp_path: Path,
) -> None:
    payload = _production_shaped_acceptance(tmp_path)
    accepted = {
        item["dataset_id"]
        for item in payload["dataset_results"]
        if item["status"] == "accepted" and item["paper_eligible"] is True
    }
    blocked = {
        item["dataset_id"]
        for item in payload["dataset_results"]
        if item["status"] == "blocked" and item["paper_eligible"] is False
    }

    assert accepted == D1_ACCEPTED_DATASETS
    assert blocked == D1_BLOCKED_DATASETS
    assert _errors(payload) == []


def test_false_dataset_parent_check_still_fails(tmp_path: Path) -> None:
    payload = _production_shaped_acceptance(tmp_path)
    _provenance_check(payload, status="accepted")["passed"] = False

    errors = _errors(payload)

    assert any("has a failed check" in error for error in errors)


def test_false_required_artifact_semantic_check_still_fails(
    tmp_path: Path,
) -> None:
    payload = _production_shaped_acceptance(tmp_path)
    payload["required_artifact_validation"]["semantic_checks"][
        "all_path_bytes_sha_bindings_match"
    ] = False

    errors = _errors(payload)

    assert any("false nested gate predicates" in error for error in errors)


@pytest.mark.parametrize(
    ("status", "mutate"),
    [
        (
            "accepted",
            lambda detail: detail.__setitem__("metadata_only_provenance_passed", True),
        ),
        (
            "accepted",
            lambda detail: detail.__setitem__("blocked_or_excluded", True),
        ),
        (
            "accepted",
            lambda detail: detail["audit"].__setitem__("complete", False),
        ),
        (
            "blocked",
            lambda detail: detail.__setitem__("accepted_provenance_passed", True),
        ),
        (
            "blocked",
            lambda detail: detail.__setitem__("integrity_failures", ["bad"]),
        ),
    ],
)
def test_provenance_branch_contradictions_fail(
    tmp_path: Path,
    status: str,
    mutate: Callable[[dict], None],
) -> None:
    payload = _production_shaped_acceptance(tmp_path)
    detail = _provenance_check(payload, status=status)["detail"]
    mutate(detail)

    errors = _errors(payload)

    assert any("production provenance" in error for error in errors)


def test_unrelated_false_detail_gate_is_not_exempted(tmp_path: Path) -> None:
    payload = _production_shaped_acceptance(tmp_path)
    detail = _provenance_check(payload, status="blocked")["detail"]
    detail["unrelated_integrity_passed"] = False

    errors = _errors(payload)

    assert any(
        "unrelated_integrity_passed" in error
        and "false nested gate predicates" in error
        for error in errors
    )


def test_duplicate_provenance_checks_fail(tmp_path: Path) -> None:
    payload = _production_shaped_acceptance(tmp_path)
    result = next(
        item for item in payload["dataset_results"] if item["status"] == "accepted"
    )
    check = _provenance_check(payload, status="accepted")
    result["checks"].append(copy.deepcopy(check))

    errors = _errors(payload)

    assert any("duplicate production provenance checks" in error for error in errors)


def test_missing_provenance_check_fails(tmp_path: Path) -> None:
    payload = _production_shaped_acceptance(tmp_path)
    result = next(
        item for item in payload["dataset_results"] if item["status"] == "blocked"
    )
    result["checks"] = [
        check
        for check in result["checks"]
        if check["name"] != "production_input_provenance_complete"
    ]

    errors = _errors(payload)

    assert any(
        "missing the required production provenance check" in error for error in errors
    )


def test_blocked_dataset_promotion_fails_exact_disposition(tmp_path: Path) -> None:
    payload = _production_shaped_acceptance(tmp_path)
    result = next(
        item
        for item in payload["dataset_results"]
        if item["dataset_id"] in D1_BLOCKED_DATASETS
    )
    result["status"] = "accepted"
    result["paper_eligible"] = True

    errors = _errors(payload)

    assert any("was promoted from the frozen disposition" in error for error in errors)
    assert any("must not be paper eligible" in error for error in errors)


@pytest.mark.parametrize(
    "raw_files",
    [
        [None],
        [
            {
                "bytes": 1,
                "path": "/production/input.tsv",
                "sha256": "a" * 64,
            }
        ],
        [
            {
                "bytes": 1,
                "defaults": {},
                "delimiter": None,
                "format": "tsv",
                "path": "relative/input.tsv",
                "role": "sequence_and_provided_label_input",
                "sha256": "a" * 64,
                "sheet_name": None,
            }
        ],
        [
            {
                "bytes": True,
                "defaults": {},
                "delimiter": None,
                "format": "tsv",
                "path": "/production/input.tsv",
                "role": "sequence_and_provided_label_input",
                "sha256": "a" * 64,
                "sheet_name": None,
            }
        ],
        [
            {
                "bytes": 0,
                "defaults": {},
                "delimiter": None,
                "format": "tsv",
                "path": "/production/input.tsv",
                "role": "sequence_and_provided_label_input",
                "sha256": "a" * 64,
                "sheet_name": None,
            }
        ],
        [
            {
                "bytes": 1,
                "defaults": {},
                "delimiter": None,
                "format": "tsv",
                "path": "/production/input.tsv",
                "role": "sequence_and_provided_label_input",
                "sha256": "A" * 64,
                "sheet_name": None,
            }
        ],
    ],
)
def test_accepted_raw_file_references_fail_closed(
    tmp_path: Path,
    raw_files: list[object],
) -> None:
    payload = _production_shaped_acceptance(tmp_path)
    check = _provenance_check(payload, status="accepted")
    check["detail"]["audit"]["raw_files"] = raw_files

    errors = _errors(payload)

    assert any("audit.raw_files" in error for error in errors)
