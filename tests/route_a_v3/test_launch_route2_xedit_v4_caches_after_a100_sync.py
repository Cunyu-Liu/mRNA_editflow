from __future__ import annotations

from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xedit_v4_caches_after_a100_sync as launcher


def test_cache_launcher_uses_current_head_formal_job_runner() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    assert launcher.CACHE_JOB_RUNNER == (
        launcher.WORKTREE / "scripts/route_a_v3/run_route2_xedit_v4_cache_job.py"
    )


def test_cache_components_build_new_critic_cache_and_adopt_setflow_read_only() -> None:
    components = launcher.component_paths()
    assert set(components) == {"critic", "setflow"}
    assert components["critic"]["builder"].name == (
        "build_route2_mrnabert_bottom_six_cache_v4.py"
    )
    assert components["setflow"]["builder"].name == (
        "adopt_route2_xeditsetflow_v4_source_token_cache.py"
    )
    assert "build_route2_xeditsetflow_source_token_cache_v3.py" not in str(
        components["setflow"]["builder"]
    )
    assert components["setflow"]["summary"].name == (
        "source_token_cache_v3_adoption_receipt_v1.json"
    )


def test_cache_authorization_status_is_component_exact() -> None:
    assert launcher.expected_authorization_status("critic") == (
        "XEDITCRITIC_V4_CACHE_LAUNCH_AUTHORIZED"
    )
    assert launcher.expected_authorization_status("setflow") == (
        "XEDITSETFLOW_V4_CACHE_LAUNCH_AUTHORIZED"
    )
    with pytest.raises(Exception, match="unknown V4 cache component"):
        launcher.expected_authorization_status("other")
