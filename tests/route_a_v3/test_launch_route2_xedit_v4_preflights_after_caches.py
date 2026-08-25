from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xedit_v4_preflights_after_caches as launcher


def test_preflight_launcher_uses_current_head_formal_job_runner() -> None:
    assert launcher.PREFLIGHT_JOB_RUNNER == (
        launcher.WORKTREE / "scripts/route_a_v3/run_route2_xedit_v4_preflight_job.py"
    )
    assert launcher.PREFLIGHT_SEQUENCE_RUNNER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xedit_v4_preflight_sequence.py"
    )


def test_preflight_components_consume_exact_cache_terminals() -> None:
    components = launcher.component_paths()
    assert set(components) == {"critic", "setflow"}
    assert components["critic"]["cache_summary"].name == (
        "frozen_bottom_six_chunk_cache_v1.summary.json"
    )
    assert components["setflow"]["cache_summary"].name == (
        "source_token_cache_v3_adoption_receipt_v1.json"
    )
    assert components["critic"]["preflight"].name == (
        "preflight_route2_xeditcritic_v4.py"
    )
    assert components["setflow"]["preflight"].name == (
        "preflight_route2_xeditsetflow_v4.py"
    )
    for component in ("critic", "setflow"):
        assert Path(components[component]["output"]).parent.name == (
            "preflight_attempt_4"
        )
        assert Path(components[component]["failure"]).name == (
            "preflight.failure.json"
        )


def test_preflight_authorization_status_is_component_exact() -> None:
    assert launcher.expected_authorization_status("critic") == (
        "XEDITCRITIC_V4_PREFLIGHT_AUTHORIZED"
    )
    assert launcher.expected_authorization_status("setflow") == (
        "XEDITSETFLOW_V4_PREFLIGHT_AUTHORIZED"
    )
    with pytest.raises(Exception, match="unknown V4 preflight component"):
        launcher.expected_authorization_status("other")


def test_cache_summary_protected_reads_are_component_specific(tmp_path: Path) -> None:
    critic = tmp_path / "critic.json"
    critic.write_text(
        json.dumps(
            {
                "git_head": "a" * 40,
                "development_test_outcomes_accessed": False,
                "evaluation_outcomes_accessed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    launcher.require_summary(critic, expected_head="a" * 40, component="critic")

    setflow = tmp_path / "setflow.json"
    setflow.write_text(
        json.dumps(
            {
                "git_head": "a" * 40,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    launcher.require_summary(setflow, expected_head="a" * 40, component="setflow")

    setflow.write_text(
        json.dumps(
            {
                "git_head": "a" * 40,
                "development_test_outcome_reads": 1,
                "new_final_evaluation_outcome_reads": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="Development TEST read"):
        launcher.require_summary(
            setflow, expected_head="a" * 40, component="setflow"
        )


def test_preflight_launcher_keeps_runner_and_cache_heads_distinct() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "def run(\n    current_head: str,\n    experiment_head: str," in source
    assert 'expected_head=experiment_head' in source
    assert '"--cache-experiment-head"' in source
    assert '"git_head": current_head' in source
    assert '"experiment_head": experiment_head' in source


def test_preflight_gpu_availability_reports_both_selected_gaps_and_full_allowed_snapshot() -> None:
    free_memory = {
        0: 36999,
        1: 8775,
        2: 7805,
        3: 18870,
        4: 8728,
        5: 334,
        6: 36631,
        7: 22296,
    }
    with pytest.raises(
        launcher.XEditV4PreflightLaunchError,
        match=(
            r"Critic GPU0 has 36999 MiB free; requires at least 37000 MiB; "
            r"SetFlow GPU3 has 18870 MiB free; requires at least 20000 MiB; "
            r"allowed_gpu_free_memory_mib="
        ),
    ) as captured:
        launcher.require_preflight_gpu_availability(
            free_memory,
            critic_gpu=0,
            setflow_gpu=3,
            critic_minimum_free_mib=launcher.CRITIC_PREFLIGHT_MINIMUM_FREE_MIB,
            setflow_minimum_free_mib=20000,
        )
    message = str(captured.value)
    assert '"0": 36999' in message
    assert '"5": 334' in message
    assert '"6"' not in message
    assert '"7"' not in message


def test_preflight_gpu_availability_accepts_exact_thresholds() -> None:
    assert launcher.CRITIC_PREFLIGHT_MINIMUM_FREE_MIB == 37_000
    assert launcher.SETFLOW_PREFLIGHT_MINIMUM_FREE_MIB == 20_000
    launcher.require_preflight_gpu_availability(
        {0: 37000, 3: 20000},
        critic_gpu=0,
        setflow_gpu=3,
        critic_minimum_free_mib=launcher.CRITIC_PREFLIGHT_MINIMUM_FREE_MIB,
        setflow_minimum_free_mib=launcher.SETFLOW_PREFLIGHT_MINIMUM_FREE_MIB,
    )


def test_preflight_gpu_layout_distinguishes_concurrent_and_sequential_modes() -> None:
    launcher.require_preflight_gpu_layout(
        critic_gpu=0, setflow_gpu=3, sequential_single_gpu=False
    )
    launcher.require_preflight_gpu_layout(
        critic_gpu=0, setflow_gpu=0, sequential_single_gpu=True
    )
    with pytest.raises(Exception, match="distinct"):
        launcher.require_preflight_gpu_layout(
            critic_gpu=0, setflow_gpu=0, sequential_single_gpu=False
        )
    with pytest.raises(Exception, match="shared"):
        launcher.require_preflight_gpu_layout(
            critic_gpu=0, setflow_gpu=3, sequential_single_gpu=True
        )


def test_preflight_job_command_binds_component_gpu_and_terminal_paths(
    tmp_path: Path,
) -> None:
    paths = launcher.component_paths()["critic"]
    paths["gpu"] = 0
    command = launcher.preflight_job_command(
        "critic",
        paths,
        authorization=tmp_path / "authorization.json",
        runtime=tmp_path / "runtime.json",
        log=tmp_path / "preflight.log",
        current_head="a" * 40,
    )
    assert command[0] == str(launcher.PYTHON)
    assert command[1] == str(launcher.PREFLIGHT_JOB_RUNNER)
    assert command[command.index("--component") + 1] == "critic"
    assert command[command.index("--physical-gpu-index") + 1] == "0"
    assert command[command.index("--git-head") + 1] == "a" * 40
