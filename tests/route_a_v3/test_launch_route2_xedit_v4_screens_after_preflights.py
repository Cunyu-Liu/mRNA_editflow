from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xedit_v4_screens_after_preflights as launcher


def test_screen_launcher_uses_current_head_formal_scheduler() -> None:
    assert launcher.SCREEN_PACKAGE_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xedit_v4_screen_package_scheduler.py"
    )


def test_screen_run_ids_are_exact_frozen_packages() -> None:
    assert set(launcher.screen_run_ids()["critic"]) == {
        "c0_v4",
        "v4_full",
        "v4_source_only",
        "v4_edit_metadata_only",
        "v4_no_candidate_sequence",
        "v4_candidate_bundle_permutation",
        "v4_no_cross",
        "v4_no_moe",
    }
    assert launcher.screen_run_ids()["setflow"] == ["v4_full", "v4_single_mode"]


def test_screen_authorization_validation_reads_terminal_json(tmp_path: Path) -> None:
    authorization = tmp_path / "critic.json"
    authorization.write_text(
        json.dumps(
            {
                "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
                "authorized_git_head": "a" * 40,
                "cache_experiment_head": "b" * 40,
                "authorized_run_ids": launcher.screen_run_ids()["critic"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    launcher.validate_screen_authorization(
        authorization,
        component="critic",
        head="a" * 40,
        experiment_head="b" * 40,
    )

    payload = json.loads(authorization.read_text(encoding="utf-8"))
    payload["authorized_run_ids"] = payload["authorized_run_ids"][:-1]
    authorization.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="authorization content is invalid"):
        launcher.validate_screen_authorization(
            authorization,
            component="critic",
            head="a" * 40,
            experiment_head="b" * 40,
        )


def test_screen_authorization_status_is_component_exact() -> None:
    assert launcher.expected_authorization_status("critic") == (
        "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
    )
    assert launcher.expected_authorization_status("setflow") == (
        "XEDITSETFLOW_V4_SCREEN_LAUNCH_AUTHORIZED"
    )
    with pytest.raises(Exception, match="unknown V4 screen component"):
        launcher.expected_authorization_status("other")
