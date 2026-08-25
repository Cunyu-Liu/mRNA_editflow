from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xedit_v4_preflights_after_caches as launcher


def test_preflight_launcher_uses_current_head_formal_job_runner() -> None:
    assert launcher.PREFLIGHT_JOB_RUNNER == (
        launcher.WORKTREE / "scripts/route_a_v3/run_route2_xedit_v4_preflight_job.py"
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
