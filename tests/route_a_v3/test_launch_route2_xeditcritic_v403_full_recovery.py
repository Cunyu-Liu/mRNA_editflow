from pathlib import Path

import scripts.route_a_v3.launch_route2_xeditcritic_v403_full_recovery as launcher
from scripts.route_a_v3.launch_route2_xeditcritic_v403_full_recovery import (
    OLD_OUTPUT_ROOT,
    build_launch_authorization,
    build_recovery_config,
)


def test_v403_full_recovery_changes_only_output_paths() -> None:
    base = {
        "output_root": str(OLD_OUTPUT_ROOT),
        "screen_gate_output": str(OLD_OUTPUT_ROOT / "screen_gate.json"),
        "required_screen_runs": [
            {"run_id": "c0_v4"},
            {"run_id": "v4_full"},
        ],
        "training": {"screen_seed": 20260907, "pass_count": 8},
    }
    output = Path("/tmp/xeditcritic-v403-test-output")

    recovery = build_recovery_config(base, output)

    assert recovery["output_root"] == str(output)
    assert recovery["screen_gate_output"] == str(output / "screen_gate.json")
    assert recovery["required_screen_runs"] == base["required_screen_runs"]
    assert recovery["training"] == base["training"]


def test_v403_authorization_remains_compatible_with_frozen_screen_runner() -> None:
    config = {
        "required_screen_runs": [
            {"run_id": "c0_v4"},
            {"run_id": "v4_full"},
        ]
    }
    authorization = build_launch_authorization(
        config,
        {"git_head": "1" * 40},
        current_head="2" * 40,
        physical_gpu_index=3,
    )

    assert authorization["schema_version"] == (
        "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
    )
    assert authorization["authorized_git_head"] == "2" * 40
    assert authorization["preflight_runner_git_head"] == "1" * 40
    assert set(authorization["authorized_run_ids"]) == {"c0_v4", "v4_full"}
    assert authorization["v403_rng_replay_recovery"] == {
        "run_id": "v4_full",
        "physical_gpu_index": 3,
        "strict_full_model_rng_replay_smoke_passed": True,
        "scientific_config_changed": False,
        "historical_c0_reference_reused": True,
    }


def test_v403_smoke_and_launcher_record_memory_without_gating() -> None:
    launcher_source = Path(launcher.__file__).read_text(encoding="utf-8")
    smoke_source = Path(launcher.__file__).with_name(
        "smoke_route2_xeditcritic_v403_rng_replay.py"
    ).read_text(encoding="utf-8")

    for source in (launcher_source, smoke_source):
        assert '"free_memory_gate_applied": False' in source
        assert "required_free_memory_bytes" not in source
    assert (
        "selected GPU lacks the replay-smoke memory requirement"
        not in launcher_source
    )
    assert (
        "selected GPU free memory is below measured peak plus 2 GiB"
        not in smoke_source
    )
