from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v402_gpu5_recovery as launcher


def _base_config() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_config.v1",
        "output_root": str(launcher.OLD_OUTPUT_ROOT),
        "screen_gate_output": str(launcher.OLD_OUTPUT_ROOT / "screen_gate.json"),
        "required_screen_runs": [
            {"run_id": run_id, "fixed": run_id} for run_id in launcher.RUN_IDS
        ],
        "architecture": {"model_width": 768, "edit_block_count": 12},
        "training": {"screen_seed": 20260907, "passes": 8},
        "screen_gate": {"minimum_task_macro_spearman": 0.30},
    }


def test_recovery_config_changes_only_two_output_paths() -> None:
    base = _base_config()
    root = Path("/mnt/recovery/exact-head")
    recovery = launcher.build_recovery_config(base, root)
    assert recovery["output_root"] == str(root)
    assert recovery["screen_gate_output"] == str(root / "screen_gate.json")
    for key in set(base) - {"output_root", "screen_gate_output"}:
        assert recovery[key] == base[key]


def test_recovery_config_rejects_scientific_drift() -> None:
    base = _base_config()
    recovery = launcher.build_recovery_config(base, Path("/mnt/recovery"))
    recovery["training"]["passes"] = 9
    with pytest.raises(Exception, match="scientific config field"):
        launcher.assert_scientific_config_unchanged(base, recovery)


def test_recovery_config_requires_exact_complete_ordered_eight_arm_package() -> None:
    base = _base_config()
    base["required_screen_runs"] = base["required_screen_runs"][:-1]
    with pytest.raises(Exception, match="arm order"):
        launcher.build_recovery_config(base, Path("/mnt/recovery"))


def test_recovery_authorization_is_trainer_compatible_and_v402_bound() -> None:
    preflight = {
        "git_head": "a" * 40,
    }
    authorization = launcher.build_launch_authorization(
        current_head="b" * 40,
        preflight=preflight,
    )
    assert authorization["schema_version"] == (
        "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
    )
    assert authorization["status"] == "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
    assert authorization["authorized_run_ids"] == list(launcher.RUN_IDS)
    assert authorization["v402_technical_recovery"] == {
        "single_complete_eight_arm_package": True,
        "physical_gpu_index": 5,
        "old_failure_artifacts_retained": True,
        "setflow_jobs_stopped_modified_or_restarted": False,
        "scientific_config_changed": False,
    }
    assert all(authorization["barriers"].values())
    assert authorization["development_test_outcome_reads"] == 0
    assert authorization["new_final_evaluation_outcome_reads"] == 0


def test_single_launch_marker_is_exclusive(tmp_path: Path) -> None:
    marker = tmp_path / "consumed.json"
    launcher.write_exclusive(marker, {"status": "CONSUMED"})
    assert json.loads(marker.read_text())["status"] == "CONSUMED"
    with pytest.raises(FileExistsError):
        launcher.write_exclusive(marker, {"status": "DUPLICATE"})


def test_recovery_worktree_and_gpu_are_exactly_isolated() -> None:
    assert launcher.WORKTREE.name == "route_a_v3_route2_v402_recovery_20260826"
    assert launcher.SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xeditcritic_v402_recovery_scheduler.py"
    )
    assert launcher.SINGLE_LAUNCH_MARKER.name == (
        "v402_recovery_launch_consumed.json"
    )
