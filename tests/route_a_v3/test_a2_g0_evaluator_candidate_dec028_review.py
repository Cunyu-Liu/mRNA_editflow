from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_PATH = (
    REPO_ROOT
    / "configs/route_a_v3_a2_g0_evaluator_candidate_dec028_review_v1.json"
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load() -> dict:
    return json.loads(REVIEW_PATH.read_text(encoding="utf-8"))


def test_dec028_a2_review_is_nonactive_and_preserves_state() -> None:
    review = _load()
    assert review["record_status"] == (
        "REVIEW_COMPLETE_NON_AUTHORITATIVE_NOT_ACTIVE_PROTOCOL"
    )
    assert review["authority_status"] == "NON_AUTHORITATIVE"
    context = review["authority_context"]
    assert context["decision_id"] == "V3-DEC-028"
    assert context["runtime_event_id"] == "A1-EVT-061"
    assert context["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    for key in (
        "training_allowed",
        "gpu_work_allowed",
        "model_selection_allowed",
        "a7_allowed",
    ):
        assert context[key] is False


def test_dec028_a2_review_binds_the_exact_forward_port_commit() -> None:
    candidate = _load()["candidate"]
    commit = candidate["implementation_commit"]
    assert _git("rev-parse", f"{commit}^") == candidate["implementation_expected_parent"]
    paths = sorted(
        line
        for line in _git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", commit
        ).splitlines()
        if line
    )
    assert paths == sorted(candidate["implementation_exact_changed_paths"])

    config = json.loads(
        (REPO_ROOT / "configs/route_a_v3_a2_g0_evaluator_candidate_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["candidate_id"] == candidate["candidate_id"]
    assert config["document_status"] == candidate["required_document_status"]
    assert config["authority_status"] == candidate["required_authority_status"]
    assert config["activation_state"] == candidate["required_activation_state"]


def test_dec028_a2_review_closes_ss1_semantics_without_promotion() -> None:
    review = _load()
    validated = set(review["independent_review"]["validated_behavior"])
    assert {
        "SYNTHETIC_ONLY_INPUT_SCOPE",
        "OUTCOME_BLIND_SOURCE_GROUP_AND_KNOWN_DUPLICATE_COMPONENT_SPLIT",
        "COMPONENT_DISJOINT_AGGREGATE_SPLIT_OUTPUT",
        "DIRECTION_NORMALIZED_CANDIDATE_MINUS_SOURCE_ENDPOINT_CONTRACT",
        "BIOLOGICAL_REPLICATE_STANDARD_ERROR_REQUIRED",
        "MISSING_AND_NONFINITE_NEVER_IMPUTED_ZERO",
    } <= validated
    assert review["independent_review"]["verdict"] == (
        "PASS_G0_SYNTHETIC_INTERFACE_ONLY_PARTIAL_NOT_ACTIVE"
    )
    assert review["interface_isolation"] == {
        "a6_candidate_is_review_evidence_for_a2": False,
        "a2_candidate_is_review_evidence_for_a6": False,
        "guide_output_is_evaluator_input": False,
        "model_selection_output_is_evaluator_input": False,
    }
    assert "A2_PHASE_PASS" in review["independent_review"]["explicit_noncoverage"]
