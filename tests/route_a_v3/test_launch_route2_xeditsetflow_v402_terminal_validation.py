from __future__ import annotations

import scripts.route_a_v3.launch_route2_xeditsetflow_v402_terminal_validation as launcher


def test_v402_setflow_launcher_is_setflow_only_and_excludes_gpu5() -> None:
    assignments = launcher.validation_assignments()
    assert set(assignments) == {0, 1, 2, 3, 4}
    assert sum(len(rows) for rows in assignments.values()) == 8
    assert {
        row for rows in assignments.values() for row in rows
    } == {
        (run_id, checkpoint_pass)
        for run_id in ("v4_full", "v4_single_mode")
        for checkpoint_pass in (4, 6, 8, 10)
    }


def test_v402_setflow_launcher_binds_historical_screen_without_critic_adjudicator() -> None:
    source = open(launcher.__file__, encoding="utf-8").read()
    assert launcher.SOURCE_SCREEN_HEAD == "edad89392077a0cf56e84dfcf94335606dd2b05a"
    assert launcher.EXPERIMENT_HEAD == "a7ef72fac23cd5b25dcc6c8d560236b97fa8b09d"
    assert "adjudicate_route2_xeditcritic_v4_screen" not in source
    assert '"critic_failure_payload_reads": 0' in source
