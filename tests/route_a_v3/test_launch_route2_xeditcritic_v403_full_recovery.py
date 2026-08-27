from pathlib import Path

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
