from __future__ import annotations

from pathlib import Path

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
