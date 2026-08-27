from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditsetflow_v403_validation_recovery as launcher


def test_v403_setflow_launcher_resolves_its_own_repository_root() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]


def test_v403_validation_recovery_distributes_all_checkpoints_including_gpu5() -> None:
    assignments = launcher.validation_assignments([0, 5])
    assert set(assignments) == {0, 5}
    assert {
        row for rows in assignments.values() for row in rows
    } == set(launcher.VALIDATION_JOBS)
    assert sum(len(rows) for rows in assignments.values()) == 8
    assert 5 in assignments


def test_v403_validation_recovery_does_not_gate_on_free_memory() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert "required_free_memory_mib" not in source
    assert ">= required_free" not in source


def test_v403_recovery_is_validation_only_and_preserves_old_gate() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "train_route2_xeditsetflow_v4.py" not in source
    assert '"training_reused": True' in source
    assert '"parameter_update_count": 0' in source
    assert '"original_technical_gate"' in source
    assert launcher.SOURCE_SCREEN_HEAD == (
        "edad89392077a0cf56e84dfcf94335606dd2b05a"
    )


@pytest.mark.parametrize(
    "status",
    [
        "XEDITSETFLOW_V403_VALIDATION_RECOVERY_RUNNING",
        "XEDITSETFLOW_V403_VALIDATION_RECOVERY_FAILED",
    ],
)
def test_new_head_cannot_bypass_existing_canonical_setflow_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    runtime_root = tmp_path / "canonical-runtime"
    runtime_root.mkdir()
    (runtime_root / "runtime.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "route_a_v3_route2_xeditsetflow_v403_validation_recovery_runtime.v1"
                ),
                "status": status,
                "git_head": launcher.CANONICAL_RECOVERY_HEAD,
                "source_screen_head": launcher.SOURCE_SCREEN_HEAD,
                "experiment_head": launcher.EXPERIMENT_HEAD,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "CANONICAL_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(
        launcher, "CANONICAL_RECOVERY_ROOT", tmp_path / "canonical-output"
    )
    monkeypatch.setattr(
        launcher,
        "CANONICAL_RUNTIME_CONFIG",
        tmp_path / "canonical-runtime-config.json",
    )
    monkeypatch.setattr(
        launcher, "CANONICAL_LOG_ROOT", tmp_path / "canonical-logs"
    )
    monkeypatch.setattr(
        launcher,
        "CANONICAL_LAUNCH_MARKER",
        tmp_path / "canonical-launch-marker.json",
    )

    with pytest.raises(Exception, match="already RUNNING, terminal, or consumed"):
        launcher.require_canonical_attempt_unconsumed("9" * 40)


def test_canonical_setflow_guard_precedes_probe_and_new_head_paths() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    run_source = source[source.index("def run(current_head: str)") :]
    guard = run_source.index("require_canonical_attempt_unconsumed(")
    probe = run_source.index("gpu_free_memory_mib(")
    new_path = run_source.index('run_name = f"v403_validation_recovery_')
    assert guard < probe
    assert guard < new_path
