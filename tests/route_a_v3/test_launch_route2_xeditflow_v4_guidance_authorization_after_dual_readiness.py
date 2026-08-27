from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditflow_v4_guidance_authorization_after_dual_readiness as launcher


HEAD = "a" * 40


def _critic_runtime(status: str) -> dict[str, object]:
    ready = status == "CRITIC_V4_READY_FOR_GUIDANCE"
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_runtime.v1",
        "status": status,
        "git_head": HEAD,
        "readiness": {
            "terminal_artifact_kind": "SUMMARY",
            "readiness_status": status,
            "guidance_authorized": ready,
        },
        "active_performance_output_read": False,
        "development_test_access_event_count_before_loso": 1,
        "development_test_outcome_reads_during_loso": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _setflow_runtime(
    *, eligible: bool = True, terminal: str = "SUMMARY", status: str = "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL"
) -> dict[str, object]:
    return {
        "schema_version": "route_a_v3_route2_xedit_v4_confirmation_posttraining_runtime.v1",
        "status": status,
        "git_head": HEAD,
        "eligible_components": ["setflow"] if eligible else ["critic"],
        "adjudications": {"setflow": {"terminal_artifact_kind": terminal}},
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_joint_authorization_requires_both_final_readiness_states() -> None:
    critic_state = launcher.critic_readiness_state(
        _critic_runtime("CRITIC_V4_READY_FOR_GUIDANCE"), {}, head=HEAD
    )
    setflow_state = launcher.setflow_readiness_state(
        _setflow_runtime(), {"status": "XEDITSETFLOW_V4_G0_READY"}, head=HEAD
    )
    assert critic_state == setflow_state == "READY"
    assert (
        launcher.guidance_authorization_decision(critic_state, setflow_state)
        == "AUTHORIZE_EXACT_V4_GUIDANCE"
    )


def test_joint_authorization_preserves_scientific_no_go_states() -> None:
    critic_state = launcher.critic_readiness_state(
        _critic_runtime("CRITIC_V4_NOT_READY_FOR_GUIDANCE"), None, head=HEAD
    )
    assert critic_state == "CRITIC_READINESS_NO_GO"
    assert launcher.guidance_authorization_decision(critic_state, "READY") == (
        "NOT_AUTHORIZED_CRITIC_READINESS_NO_GO"
    )
    setflow_state = launcher.setflow_readiness_state(
        _setflow_runtime(),
        {"status": "XEDITSETFLOW_V4_CONFIRMATION_NO_GO"},
        head=HEAD,
    )
    assert setflow_state == "SETFLOW_CONFIRMATION_NO_GO"
    assert launcher.guidance_authorization_decision("READY", setflow_state) == (
        "NOT_AUTHORIZED_SETFLOW_CONFIRMATION_NO_GO"
    )


def test_joint_authorization_distinguishes_screen_and_technical_blocks() -> None:
    assert launcher.setflow_readiness_state(
        _setflow_runtime(eligible=False), None, head=HEAD
    ) == "SETFLOW_SCREEN_NO_GO"
    assert launcher.setflow_readiness_state(
        _setflow_runtime(terminal="FAILURE"), None, head=HEAD
    ) == "SETFLOW_CONFIRMATION_TECHNICAL_FAILURE"
    assert launcher.critic_readiness_state(
        _critic_runtime("XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE"), None, head=HEAD
    ) == "CRITIC_LOSO_TECHNICAL_FAILURE"


def test_joint_authorization_rejects_active_or_protected_runtime() -> None:
    critic = _critic_runtime("CRITIC_V4_READY_FOR_GUIDANCE")
    critic["status"] = "XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING"
    with pytest.raises(Exception, match="not terminal"):
        launcher.critic_readiness_state(critic, {}, head=HEAD)
    setflow = _setflow_runtime()
    setflow["development_test_outcome_reads"] = 1
    with pytest.raises(Exception, match="protected boundary"):
        launcher.setflow_readiness_state(
            setflow, {"status": "XEDITSETFLOW_V4_G0_READY"}, head=HEAD
        )


def test_joint_authorization_entry_uses_existing_one_shot_authorizer() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    assert launcher.AUTHORIZER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/authorize_route2_xeditflow_v4_guidance.py"
    )


def test_joint_authorization_accepts_distinct_runtime_heads_and_paths() -> None:
    critic_head = "b" * 40
    setflow_head = "c" * 40
    critic = _critic_runtime("CRITIC_V4_READY_FOR_GUIDANCE")
    critic["git_head"] = critic_head
    setflow = _setflow_runtime()
    setflow["git_head"] = setflow_head
    assert launcher.critic_readiness_state(
        critic, {}, head=critic_head
    ) == "READY"
    assert launcher.setflow_readiness_state(
        setflow,
        {"status": "XEDITSETFLOW_V4_G0_READY"},
        head=setflow_head,
    ) == "READY"
    parameters = inspect.signature(launcher.run).parameters
    assert {
        "protocol_path",
        "critic_runtime_path",
        "critic_runtime_head",
        "setflow_runtime_path",
        "setflow_runtime_head",
        "decision_output",
    } <= set(parameters)
