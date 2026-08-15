from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_PATH = (
    REPO_ROOT
    / "configs/route_a_v3_a6_learned_base_value_g0_candidate_dec028_review_v1.json"
)


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _load() -> dict:
    return json.loads(REVIEW_PATH.read_text(encoding="utf-8"))


def test_dec028_a6_review_is_nonactive_and_preserves_locks() -> None:
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
    assert context["current_future_g1_role"] == "GSE200304_SOURCE_RELATIVE_CRITIC_G1"
    for key in (
        "a6_learned_base_value_execution_authorized",
        "training_allowed",
        "gpu_work_allowed",
        "model_selection_allowed",
        "a7_allowed",
    ):
        assert context[key] is False


def test_dec028_a6_review_reuses_the_unchanged_ancestor_candidate() -> None:
    candidate = _load()["candidate"]
    commit = candidate["implementation_commit"]
    assert _git("rev-parse", f"{commit}^").stdout.strip() == candidate[
        "implementation_expected_parent"
    ]
    paths = sorted(
        line
        for line in _git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", commit
        ).stdout.splitlines()
        if line
    )
    assert paths == sorted(candidate["implementation_exact_changed_paths"])
    assert _git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode == 0
    assert _git(
        "diff", "--quiet", commit, "--", *candidate["implementation_exact_changed_paths"],
        check=False,
    ).returncode == 0

    config = json.loads((REPO_ROOT / candidate["config_path"]).read_text(encoding="utf-8"))
    parent = json.loads(
        (REPO_ROOT / candidate["parent_protocol_path"]).read_text(encoding="utf-8")
    )
    assert config["document_status"] == candidate["required_document_status"]
    assert config["authority_status"] == candidate["required_authority_status"]
    assert config["activation_state"] == candidate["required_activation_state"]
    assert parent["activation_state"] == candidate["parent_required_activation_state"]


def test_dec028_a6_review_closes_zero_update_but_not_learned_execution() -> None:
    review = _load()
    assert review["independent_review"]["verdict"] == (
        "PASS_G0_ZERO_UPDATE_INTERFACE_ONLY_PARTIAL_NOT_ACTIVE"
    )
    assert "ZERO_MODEL_OPTIMIZER_CUDA_CHECKPOINT_PARAMETER_UPDATE_OR_DATA_ROW_IO" in (
        review["independent_review"]["validated_behavior"]
    )
    noncoverage = set(review["independent_review"]["explicit_noncoverage"])
    assert {
        "SOURCE_RELATIVE_CRITIC_CALIBRATION_OR_LCB_MANIFEST",
        "A6_LEARNED_BASE_VALUE_SUCCESSOR_AUTHORITY",
        "A6_PASS_L3_CLAIM_OR_A7_UNLOCK",
    } <= noncoverage
    assert review["interface_isolation"] == {
        "a2_candidate_is_review_evidence_for_a6": False,
        "a6_candidate_is_review_evidence_for_a2": False,
        "source_relative_critic_is_implemented_by_this_candidate": False,
        "source_relative_critic_lcb_is_available": False,
    }
