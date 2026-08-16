from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/build_route2_generation_eligibility_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_route2_generation_eligibility_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(record_id: str, candidate: str, outcome: float):
    return {
        "record_id": record_id,
        "study_unit_id": "S",
        "source_id": "SRC",
        "biological_context_id": "CTX",
        "endpoint_id": "END",
        "region": "3UTR",
        "assay_id": "ASSAY",
        "source": "AAAA",
        "candidate": candidate,
        "edit_count": sum(base != "A" for base in candidate),
        "outcome": outcome,
    }


def test_generation_cohorts_use_fixed_budgets_and_no_canonical_credit() -> None:
    module = _load()
    key = ("S", "SRC", "CTX", "END")
    groups = {key: [_row("r1", "CAAA", 1.0), _row("r2", "GAAA", 2.0), _row("r3", "CCAA", 3.0)]}
    sources, measured, summary = module.build_eligibility(
        groups,
        requested_split="TEST",
        edit_budgets=(1, 3, 5),
        candidate_budget=8,
        minimum_measured_candidates=2,
    )
    assert [row["edit_budget"] for row in sources] == [1, 3, 5]
    assert all(row["generated_candidates_grant_canonical_credit"] is False for row in sources)
    assert all(row["evaluation_outcomes_included"] is False for row in sources)
    assert all(row["pool_assignment"] == "DEVELOPMENT" for row in measured)
    assert summary["evaluation_records_read"] == 0


def test_unequal_duplicate_candidate_outcomes_exclude_the_source_group() -> None:
    module = _load()
    ambiguous_key = ("S", "SRC", "CTX", "END")
    eligible_key = ("S", "SRC2", "CTX", "END")
    eligible_rows = [_row("r3", "CAAA", 1.0), _row("r4", "GAAA", 2.0)]
    for row in eligible_rows:
        row["source_id"] = "SRC2"
    groups = {
        ambiguous_key: [_row("r1", "CAAA", 1.0), _row("r2", "CAAA", 2.0)],
        eligible_key: eligible_rows,
    }
    sources, measured, summary = module.build_eligibility(
        groups,
        requested_split="TEST",
        edit_budgets=(1, 3, 5),
        candidate_budget=8,
        minimum_measured_candidates=2,
    )
    assert {row["source_id"] for row in sources} == {"SRC2"}
    assert {record_id for row in measured for record_id in row["canonical_record_ids"]} == {"r3", "r4"}
    assert summary["exclusions"]["AMBIGUOUS_DUPLICATE_CANDIDATE_OUTCOME_SOURCE_GROUP"] == 1


def test_too_small_measured_pool_is_excluded_with_reason() -> None:
    module = _load()
    key = ("S", "SRC", "CTX", "END")
    groups = {key: [_row("r1", "CAAA", 1.0)]}
    with pytest.raises(module.EligibilityError, match="no generation-eligible"):
        module.build_eligibility(
            groups,
            requested_split="TEST",
            edit_budgets=(1, 3, 5),
            candidate_budget=8,
            minimum_measured_candidates=2,
        )
