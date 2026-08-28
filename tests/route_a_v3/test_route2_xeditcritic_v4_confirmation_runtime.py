from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.route_a_v3.adjudicate_route2_xeditcritic_v4_confirmation import (
    collect_critic_confirmation_payloads_v4,
    load_critic_confirmation_configs_v4,
)
from scripts.route_a_v3.authorize_route2_xeditcritic_v4_confirmation import (
    build_critic_confirmation_authorization_v4,
)
from tests.route_a_v3.test_train_route2_xeditcritic_v4 import (
    _authorization,
    _config,
    _preflight,
)


def _screen_gate() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_gate.v1",
        "status": "XEDITCRITIC_V4_SCREEN_PASS",
        "passed": True,
        "confirmation_authorized": True,
        "development_test_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_confirmation_authorizer_preserves_exact_scope_and_test_lock() -> None:
    config = _config()
    result = build_critic_confirmation_authorization_v4(
        config,
        _authorization(config),
        _preflight(),
        _screen_gate(),
        current_git_head="head",
    )
    assert result["authorized_seeds"] == [20260908, 20260909, 20260910]
    assert result["authorized_run_ids"] == ["v4_full", "c0_v4"]
    assert result["development_test_authorized"] is False
    gate = _screen_gate()
    gate["status"] = "XEDITCRITIC_V4_SCREEN_NO_GO"
    with pytest.raises(RuntimeError):
        build_critic_confirmation_authorization_v4(
            config,
            _authorization(config),
            _preflight(),
            gate,
            current_git_head="head",
        )


def test_terminal_collector_retains_failures_and_requires_both_matched_runs(
    tmp_path: Path,
) -> None:
    configs = {}
    for seed in (20260908, 20260909, 20260910):
        root = tmp_path / f"seed_{seed}"
        configs[seed] = {
            "output_root": str(root),
            "bootstrap_seed": seed * 100 + 1,
        }
        for run_id in ("v4_full", "c0_v4"):
            directory = root / run_id
            directory.mkdir(parents=True)
            (directory / "failure.json").write_text(
                json.dumps(
                    {
                        "development_test_outcome_reads": 0,
                        "new_final_evaluation_outcome_reads": 0,
                    }
                ),
                encoding="utf-8",
            )
    payloads, failures = collect_critic_confirmation_payloads_v4(configs)
    assert payloads == {}
    assert len(failures) == 6
    (tmp_path / "seed_20260910" / "c0_v4" / "failure.json").unlink()
    with pytest.raises(RuntimeError, match="not exactly terminal"):
        collect_critic_confirmation_payloads_v4(configs)


def test_config_loader_binds_head_seed_and_exact_paths(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    run_root = tmp_path / "runs"
    runner_head = "a" * 40
    config_paths = []
    config_root.mkdir()
    for seed in (20260908, 20260909, 20260910):
        path = config_root / f"seed_{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "training_seed": seed,
                    "run_stage": "CONFIRMATION",
                    "required_confirmation_run_ids": ["v4_full", "c0_v4"],
                    "confirmation_runner_git_head": runner_head,
                    "output_root": str(run_root / f"seed_{seed}"),
                }
            ),
            encoding="utf-8",
        )
        config_paths.append(str(path))
    manifest = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_confirmation_config_manifest.v1",
        "status": "THREE_MATCHED_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED",
        "required_seeds": [20260908, 20260909, 20260910],
        "required_run_ids": ["v4_full", "c0_v4"],
        "confirmation_runner_git_head": runner_head,
        "config_paths": config_paths,
    }
    configs = load_critic_confirmation_configs_v4(
        manifest, runtime_config_root=config_root, run_root=run_root
    )
    assert set(configs) == {20260908, 20260909, 20260910}

    drifted = json.loads((config_root / "seed_20260908.json").read_text())
    drifted["output_root"] = str(run_root / "seed_20260909")
    (config_root / "seed_20260908.json").write_text(json.dumps(drifted))
    with pytest.raises(RuntimeError, match="configs changed"):
        load_critic_confirmation_configs_v4(
            manifest, runtime_config_root=config_root, run_root=run_root
        )
