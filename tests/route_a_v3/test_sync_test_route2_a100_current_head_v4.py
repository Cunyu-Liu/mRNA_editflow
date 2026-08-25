from __future__ import annotations

from pathlib import Path

import scripts.route_a_v3.sync_test_route2_a100_current_head_v4 as sync


def test_critic_current_head_selection_includes_read_once_and_runner_regressions() -> None:
    assert (
        "tests/route_a_v3/test_adjudicate_route2_xeditcritic_v3_c3_v4_reference.py"
        in sync.CRITIC_TEST_PATTERNS
    )
    assert (
        "tests/route_a_v3/test_sync_test_route2_a100_current_head_v4.py"
        in sync.CRITIC_TEST_PATTERNS
    )
    assert sync.V332_TEST_PATTERNS == ("tests/route_a_v3/*v332*.py",)


def test_terminal_package_requires_exactly_one_terminal_artifact_per_run(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(sync, "EXPERIMENT_ROOT", tmp_path)
    for run_id in sync.RUN_IDS:
        directory = tmp_path / run_id
        directory.mkdir()
        (directory / "run_summary.json").write_text("{}\n", encoding="utf-8")
    assert sync.exact_terminal_package() is True

    incomplete = tmp_path / sync.RUN_IDS[0]
    (incomplete / "run_summary.json").unlink()
    assert sync.exact_terminal_package() is False
    (incomplete / "run_summary.json").write_text("{}\n", encoding="utf-8")
    (incomplete / "failure.json").write_text("{}\n", encoding="utf-8")
    assert sync.exact_terminal_package() is False


def test_test_file_selection_deduplicates_overlapping_patterns(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(sync, "WORKTREE", tmp_path)
    test_root = tmp_path / "tests" / "route_a_v3"
    test_root.mkdir(parents=True)
    target = test_root / "test_route2_xeditcritic_v4.py"
    target.write_text("def test_placeholder(): pass\n", encoding="utf-8")
    selected = sync.test_files(
        (
            "tests/route_a_v3/*xeditcritic_v4*.py",
            "tests/route_a_v3/test_route2_xeditcritic_v4.py",
        )
    )
    assert selected == ["tests/route_a_v3/test_route2_xeditcritic_v4.py"]
