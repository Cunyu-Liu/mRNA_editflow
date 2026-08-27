#!/usr/bin/env python3
"""Authorize V4 guidance once, only after both final readiness receipts pass."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


WORKTREE = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
AUTHORIZER = WORKTREE / "scripts/route_a_v3/authorize_route2_xeditflow_v4_guidance.py"
GUIDANCE_PROTOCOL = (
    WORKTREE / "configs/route_a_v3_route2_xeditflow_v4_guidance_protocol_v1.json"
)


class XEditFlowV4GuidanceAuthorizationLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowV4GuidanceAuthorizationLaunchError(message)


def command(
    arguments: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=check,
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def critic_readiness_state(
    runtime: Mapping[str, Any], readiness: Mapping[str, Any] | None, *, head: str
) -> str:
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_loso_runtime.v1"
        and runtime.get("git_head") == head
        and runtime.get("active_performance_output_read") is False
        and int(runtime.get("development_test_access_event_count_before_loso", -1))
        == 1
        and int(runtime.get("development_test_outcome_reads_during_loso", -1)) == 0
        and int(runtime.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 LOSO runtime changed or crossed a protected boundary",
    )
    status = runtime.get("status")
    if status == "XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE":
        return "CRITIC_LOSO_TECHNICAL_FAILURE"
    require(
        status in {
            "CRITIC_V4_READY_FOR_GUIDANCE",
            "CRITIC_V4_NOT_READY_FOR_GUIDANCE",
        },
        "Critic V4 LOSO runtime is not terminal",
    )
    receipt = runtime.get("readiness", {})
    require(
        receipt.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"},
        "Critic V4 readiness lacks an exact terminal artifact",
    )
    if status == "CRITIC_V4_NOT_READY_FOR_GUIDANCE":
        require(
            receipt.get("guidance_authorized") is False,
            "Critic V4 not-ready runtime unexpectedly authorizes guidance",
        )
        return "CRITIC_READINESS_NO_GO"
    require(
        receipt.get("terminal_artifact_kind") == "SUMMARY"
        and receipt.get("readiness_status") == "CRITIC_V4_READY_FOR_GUIDANCE"
        and receipt.get("guidance_authorized") is True
        and readiness is not None,
        "Critic V4 ready runtime lacks its passing readiness receipt",
    )
    return "READY"


def setflow_readiness_state(
    runtime: Mapping[str, Any], gate: Mapping[str, Any] | None, *, head: str
) -> str:
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xedit_v4_confirmation_posttraining_runtime.v1"
        and runtime.get("git_head") == head
        and runtime.get("active_performance_output_read") is False
        and int(runtime.get("development_test_outcome_reads", -1)) == 0
        and int(runtime.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow V4 confirmation runtime changed or crossed a protected boundary",
    )
    status = runtime.get("status")
    if status == "V4_CONFIRMATION_POSTTRAINING_TECHNICAL_FAILURE":
        return "SETFLOW_CONFIRMATION_TECHNICAL_FAILURE"
    require(
        status == "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL",
        "SetFlow V4 confirmation post-training runtime is not terminal",
    )
    if "setflow" not in runtime.get("eligible_components", []):
        return "SETFLOW_SCREEN_NO_GO"
    terminal = runtime.get("adjudications", {}).get("setflow", {}).get(
        "terminal_artifact_kind"
    )
    if terminal == "FAILURE":
        return "SETFLOW_CONFIRMATION_TECHNICAL_FAILURE"
    require(
        terminal == "SUMMARY" and gate is not None,
        "SetFlow V4 confirmation lacks its exact gate summary",
    )
    if gate.get("status") == "XEDITSETFLOW_V4_CONFIRMATION_NO_GO":
        return "SETFLOW_CONFIRMATION_NO_GO"
    require(
        gate.get("status") == "XEDITSETFLOW_V4_G0_READY",
        "SetFlow V4 confirmation gate has an unexpected status",
    )
    return "READY"


def guidance_authorization_decision(critic_state: str, setflow_state: str) -> str:
    if critic_state != "READY":
        return f"NOT_AUTHORIZED_{critic_state}"
    if setflow_state != "READY":
        return f"NOT_AUTHORIZED_{setflow_state}"
    return "AUTHORIZE_EXACT_V4_GUIDANCE"


def run(head: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "expected Git HEAD is invalid")
    require(
        PYTHON.is_file() and AUTHORIZER.is_file() and GUIDANCE_PROTOCOL.is_file(),
        "formal V4 guidance authorization runtime is absent",
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == head,
        "A100 worktree is not at expected HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )
    runtime_root = ROOT / f"experiments/xedit_v4/guidance_authorization_{head}"
    require(not runtime_root.exists(), "V4 guidance authorization decision exists")
    protocol = read_json(GUIDANCE_PROTOCOL)
    require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_protocol.v1",
        "V4 guidance protocol changed",
    )

    critic_runtime = read_json(
        ROOT / f"experiments/xedit_v4/loso_execution_{head}/runtime.json"
    )
    critic_readiness_path = Path(protocol["critic_readiness_path"])
    critic_readiness = (
        read_json(critic_readiness_path) if critic_readiness_path.is_file() else None
    )
    setflow_runtime = read_json(
        ROOT / f"experiments/xedit_v4/confirmation_posttraining_{head}/runtime.json"
    )
    setflow_gate_path = Path(protocol["setflow_confirmation_path"])
    setflow_gate = read_json(setflow_gate_path) if setflow_gate_path.is_file() else None
    critic_state = critic_readiness_state(
        critic_runtime, critic_readiness, head=head
    )
    setflow_state = setflow_readiness_state(
        setflow_runtime, setflow_gate, head=head
    )
    decision = guidance_authorization_decision(critic_state, setflow_state)
    result: dict[str, Any] = {
        "schema_version": (
            "route_a_v3_route2_xeditflow_v4_guidance_authorization_launch.v1"
        ),
        "status": decision,
        "git_head": head,
        "critic_readiness_state": critic_state,
        "setflow_readiness_state": setflow_state,
        "guidance_authorized": False,
        "guidance_training_or_sampling_launched": False,
        "development_test_reopened": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_reads": 0,
    }
    if decision != "AUTHORIZE_EXACT_V4_GUIDANCE":
        runtime_root.mkdir(parents=True)
        write_atomic(runtime_root / "decision.json", result)
        return result

    authorization_output = Path(protocol["authorization_output"])
    require(not authorization_output.exists(), "V4 guidance was already authorized")
    completed = command(
        [str(PYTHON), str(AUTHORIZER), "--protocol", str(GUIDANCE_PROTOCOL)],
        check=False,
    )
    require(
        authorization_output.is_file(),
        "V4 guidance authorizer did not publish its atomic authorization",
    )
    authorization = read_json(authorization_output)
    require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_authorization.v1"
        and authorization.get("status") == "XEDITFLOW_V4_GUIDANCE_AUTHORIZED"
        and authorization.get("critic_ready") is True
        and authorization.get("setflow_ready") is True
        and authorization.get("guidance_authorized") is True
        and authorization.get("development_test_reopened") is False
        and authorization.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and authorization.get("new_final_evaluation_authorized") is False
        and int(authorization.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4 joint guidance authorization changed",
    )
    runtime_root.mkdir(parents=True)
    result.update(
        {
            "status": "XEDITFLOW_V4_GUIDANCE_AUTHORIZED",
            "guidance_authorized": True,
            "authorizer_return_code": int(completed.returncode),
            "authorization_output": str(authorization_output),
        }
    )
    write_atomic(runtime_root / "decision.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_head), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
