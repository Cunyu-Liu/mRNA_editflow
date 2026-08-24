from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.route_a_v3.adjudicate_route2_xeditsetflow_v4_confirmation import (
    collect_confirmation_terminal_artifacts_v4,
)


def _config(tmp_path: Path, seed: int) -> dict:
    root = tmp_path / f"seed_{seed}"
    return {
        "output_root": str(root),
        "validation_output_root": str(root / "validation"),
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_terminal_collector_requires_all_three_seeds_and_four_validations(
    tmp_path: Path,
) -> None:
    configs = {
        seed: _config(tmp_path, seed)
        for seed in (20260912, 20260913, 20260914)
    }
    for seed, config in configs.items():
        _write(
            Path(config["output_root"]) / "v4_full" / "training_summary.json",
            {
                "status": "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION",
                "run_stage": "CONFIRMATION",
                "seed": seed,
            },
        )
        for checkpoint_pass in (4, 6, 8, 10):
            _write(
                Path(config["validation_output_root"])
                / "v4_full"
                / f"pass_{checkpoint_pass}"
                / "validation_summary.json",
                {"seed": seed, "checkpoint_pass": checkpoint_pass},
            )
    summaries, failures = collect_confirmation_terminal_artifacts_v4(configs)
    assert set(summaries) == {20260912, 20260913, 20260914}
    assert all(set(rows) == {4, 6, 8, 10} for rows in summaries.values())
    assert failures == []
    missing = (
        Path(configs[20260914]["validation_output_root"])
        / "v4_full"
        / "pass_10"
        / "validation_summary.json"
    )
    missing.unlink()
    with pytest.raises(RuntimeError):
        collect_confirmation_terminal_artifacts_v4(configs)


def test_terminal_training_failure_is_retained_without_inventing_validations(
    tmp_path: Path,
) -> None:
    configs = {
        seed: _config(tmp_path, seed)
        for seed in (20260912, 20260913, 20260914)
    }
    for seed, config in configs.items():
        if seed == 20260912:
            _write(
                Path(config["output_root"]) / "v4_full" / "failure.json",
                {
                    "development_test_outcome_reads": 0,
                    "new_final_evaluation_outcome_reads": 0,
                },
            )
            continue
        _write(
            Path(config["output_root"]) / "v4_full" / "training_summary.json",
            {
                "status": "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION",
                "run_stage": "CONFIRMATION",
                "seed": seed,
            },
        )
        for checkpoint_pass in (4, 6, 8, 10):
            _write(
                Path(config["validation_output_root"])
                / "v4_full"
                / f"pass_{checkpoint_pass}"
                / "validation_summary.json",
                {"seed": seed, "checkpoint_pass": checkpoint_pass},
            )
    summaries, failures = collect_confirmation_terminal_artifacts_v4(configs)
    assert set(summaries) == {20260913, 20260914}
    assert len(failures) == 1
    assert failures[0]["training_seed"] == 20260912
