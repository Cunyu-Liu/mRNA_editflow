from __future__ import annotations

from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditflow_v4_guidance_screen_after_authorization as launcher


def test_guidance_screen_assignment_uses_all_six_gpus_three_times() -> None:
    assignment = launcher.fixed_guidance_gpu_assignment()
    assert assignment == tuple(index % 6 for index in range(18))
    assert all(assignment.count(gpu) == 3 for gpu in range(6))


def test_guidance_screen_manifest_requires_exact_chain_and_final_paths(
    tmp_path: Path,
) -> None:
    fields = (
        "value_training_config_paths",
        "guidance_smc_config_paths",
        "guidance_critic_config_paths",
        "guidance_closed_config_paths",
        "guidance_open_metric_config_paths",
        "guidance_independent_evaluator_config_paths",
        "guidance_independent_evaluator_comparison_config_paths",
    )
    lengths = (6, 18, 18, 18, 18, 18, 18)
    manifest: dict[str, object] = {
        "schema_version": "route_a_v3_route2_xeditflow_v4_value_config_manifest.v1",
        "status": "XEDITFLOW_V4_VALUE_CONFIGS_PREPARED_NOT_STARTED",
        "base_flow_training_seed": 20260912,
        "rollout_job_count": 1,
        "critic_score_job_count": 1,
        "value_target_package_count": 6,
        "value_training_job_count": 6,
        "later_guidance_combination_count": 18,
        "guidance_gpu_assignment": list(
            launcher.fixed_guidance_gpu_assignment()
        ),
        "guidance_result_paths": [{} for _ in range(18)],
        "beta_max_used_in_value_target_or_training": False,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    for field, length in zip(fields, lengths, strict=True):
        paths = []
        for index in range(length):
            path = tmp_path / f"{field}_{index}.json"
            path.write_text("{}\n", encoding="utf-8")
            paths.append(str(path))
        manifest[field] = paths
    launcher.validate_manifest(manifest, config_root=tmp_path)
    manifest["guidance_gpu_assignment"] = [0] * 18
    with pytest.raises(Exception, match="manifest changed"):
        launcher.validate_manifest(manifest, config_root=tmp_path)


def test_guidance_screen_launcher_uses_formal_scheduler() -> None:
    assert launcher.SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xeditflow_v4_guidance_screen_scheduler.py"
    )
