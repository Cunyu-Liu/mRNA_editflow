import csv
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_baseline_matrix_v1 import build_matrix


ROOT = Path(__file__).resolve().parents[2]
COMMITTED_TABLE = ROOT / "docs/paper/route2_v332_baseline_matrix_v1.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_builder_preserves_baseline_coverage_and_protected_boundaries(tmp_path: Path) -> None:
    table_path = tmp_path / "baseline_matrix.csv"
    audit_path = tmp_path / "baseline_matrix_audit.json"
    audit = build_matrix(table_path=table_path, audit_path=audit_path)
    rows = _rows(table_path)
    by_id = {row["matrix_row_id"]: row for row in rows}

    assert rows == _rows(COMMITTED_TABLE)
    assert len(rows) == len(by_id) == audit["row_count"] == 45
    assert audit["track_counts"] == {"GENERATION": 11, "PREDICTION": 34}
    assert audit["family_counts"] == {
        "CLASSICAL": 6,
        "INTERNAL_CONTROL": 11,
        "NEURAL": 7,
        "SEARCH_FLOW": 11,
        "TASK_SPECIFIC_FOUNDATION": 10,
    }
    assert by_id["P-IC-02"]["execution_status_v332"] == "MAPPED_COMPOSITE_DEVELOPMENT_CONTROL"
    assert by_id["P-CL-05"]["contract_coverage_status"] == "SATISFIED_COMPONENT_WITH_LIMIT"
    assert by_id["P-NN-03"]["implementation_id"].endswith("delta_anchored_position_aware_antisymmetric")
    assert by_id["P-NN-06"]["execution_status_v332"] == "CONFIGURED_NOT_TERMINAL_INDEPENDENT_BASELINE"
    assert by_id["P-NN-07"]["execution_status_v332"] == "CONFIGURED_NOT_TERMINAL_INDEPENDENT_BASELINE"

    external = [row for row in rows if row["matrix_row_id"].startswith("P-EXT-")]
    assert sum(row["execution_status_v332"] == "EXECUTED_COMMON_TASK_DEVELOPMENT_VALIDATION" for row in external) == 6
    assert {row["implementation_id"] for row in external if row["current_scope"] == "LITERATURE_ONLY"} == {
        "APARENT-Perturb", "Orthrus", "RiNALMo"
    }
    assert by_id["P-EXT-10"]["execution_status_v332"] == "TERMINAL_CRITIC_V2_NO_GO"

    terminal_generation = {
        row["implementation_id"]
        for row in rows
        if row["execution_status_v332"] == "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT"
    }
    assert terminal_generation == {
        "random_legal", "greedy", "beam", "genetic", "local_search",
        "generate_then_rerank", "unguided_learned_base_flow_g0",
    }
    assert by_id["G-02"]["execution_status_v332"] == "SMALL_SPACE_REFERENCE_NOT_FULL_COHORT_EXECUTION"
    assert by_id["G-09"]["execution_status_v332"] == "NOT_RUN_CRITIC_V2_NO_GO"
    assert by_id["G-10"]["execution_status_v332"] == "NOT_RUN_CRITIC_V2_NO_GO"
    assert by_id["G-11"]["execution_status_v332"] == "LITERATURE_ONLY_TASK_MISMATCH"

    forbidden_metrics = {
        "spearman", "mae", "ndcg", "uplift", "recovery", "wall_time_seconds",
        "generation_peak_vram_mb", "candidate_count", "nfe",
    }
    assert forbidden_metrics.isdisjoint(rows[0])
    assert all(row["headline_eligible_now"] == "false" for row in rows)
    assert all(row["development_test_accessed"] == "false" for row in rows)
    assert all(row["new_final_evaluation_accessed"] == "false" for row in rows)
    assert all(row["guided_executed"] == "false" for row in rows)
    assert audit["matrix_is_result_table"] is False
    assert audit["minimum_package_alignment"] == {
        "MBP-08": "COMPLETE_DEVELOPMENT_ONLY",
        "MBP-09": "COMPLETE_DEVELOPMENT_ONLY",
        "MBP-10": "PARTIAL_GUIDED_NOT_AUTHORIZED",
        "minimum_package_complete": False,
        "submission_ready": False,
    }
    assert audit["separate_future_artifacts"] == {
        "native_common_arch_three_track_results_table_built": False,
        "prediction_generation_matched_budget_numeric_matrix_built": False,
    }
    assert audit["new_training_attempt_created"] is False
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit


def test_builder_refuses_implicit_overwrite(tmp_path: Path) -> None:
    table_path = tmp_path / "baseline_matrix.csv"
    audit_path = tmp_path / "baseline_matrix_audit.json"
    build_matrix(table_path=table_path, audit_path=audit_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_matrix(table_path=table_path, audit_path=audit_path)
