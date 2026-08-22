import csv
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_three_track_results_table_v1 import build_table


ROOT = Path(__file__).resolve().parents[2]
COMMITTED_TABLE = ROOT / "docs/paper/route2_v332_three_track_results_table_v1.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_three_tracks_remain_separate_and_protected(tmp_path: Path) -> None:
    table = tmp_path / "three_track.csv"; audit_path = tmp_path / "audit.json"
    audit = build_table(table_path=table, audit_path=audit_path)
    rows = _rows(table); by_id = {row["row_id"]: row for row in rows}
    assert rows == _rows(COMMITTED_TABLE)
    assert len(rows) == len(by_id) == audit["row_count"] == 52
    assert audit["track_counts"] == {"NATIVE_REPRODUCTION": 10, "COMMON_SOURCE_RELATIVE_TASK": 12, "ARCH_CONTROLLED": 30}
    native = [row for row in rows if row["track"] == "NATIVE_REPRODUCTION"]
    assert all(row["primary_metric_value"] == row["secondary_metric_value"] == "" for row in native)
    assert all(row["headline_horizontal_comparison_eligible"] == "false" for row in native)
    assert audit["numeric_result_counts"] == {"NATIVE_REPRODUCTION": 0, "COMMON_SOURCE_RELATIVE_TASK": 9, "ARCH_CONTROLLED": 26}
    assert sum(row["headline_horizontal_comparison_eligible"] == "true" for row in rows) == 8
    assert by_id["B-10"]["result_status"] == "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS"
    assert by_id["B-11"]["primary_metric_value"] == "0.13171439492559175"
    assert by_id["C-06"]["primary_limitation"].startswith("Aligned A1 comparison inputs")
    assert by_id["C-27"]["result_status"] == by_id["C-28"]["result_status"] == "NOT_RUN_CRITIC_V2_NO_GO"
    assert by_id["C-29"]["result_status"] == "NOT_CAUSALLY_IDENTIFIABLE_FROM_CURRENT_TERMINAL_RUNS"
    assert by_id["C-30"]["result_status"] == "NOT_TERMINAL_MATCHED_CONTRAST"
    assert all(row["scientific_success_established"] == "false" for row in rows)
    assert all(row["development_test_accessed"] == "false" for row in rows)
    assert all(row["new_final_evaluation_accessed"] == "false" for row in rows)
    assert all(row["guided_executed"] == "false" for row in rows)
    assert audit["reporting_table_complete"] is True
    assert audit["three_track_benchmark_execution_complete"] is False
    assert audit["submission_ready"] is False
    assert audit["new_training_attempt_created"] is False
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    table = tmp_path / "three_track.csv"; audit = tmp_path / "audit.json"
    build_table(table_path=table, audit_path=audit)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_table(table_path=table, audit_path=audit)
