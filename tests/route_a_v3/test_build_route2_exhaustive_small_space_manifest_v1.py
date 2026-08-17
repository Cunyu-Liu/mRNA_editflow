from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/build_route2_exhaustive_small_space_manifest_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_route2_exhaustive_small_space_manifest_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(source_key: str, sequence: str, edit_budget: int) -> dict:
    return {
        "source_key": source_key,
        "source_sequence": sequence,
        "edit_budget": edit_budget,
        "candidate_budget": 32,
        "evaluation_outcomes_included": False,
        "generated_candidates_grant_canonical_credit": False,
    }


def test_selects_only_cohorts_fully_covered_by_both_limits() -> None:
    module = _module()
    selected, summary = module.build(
        [
            _row("fits", "AC", 1),
            _row("critic_limited", "ACGU", 2),
            _row("space_limited", "ACGUAC", 3),
        ],
        max_critic_forwards=10,
        exhaustive_space_limit=200,
    )
    assert [row["source_key"] for row in selected] == ["fits"]
    assert selected[0]["exhaustive_legal_space_size"] == 7
    assert summary["selected_source_cohort_count"] == 1
    assert summary["excluded_by_reason"] == {
        "ABOVE_EXHAUSTIVE_SPACE_LIMIT": 1,
        "ABOVE_MATCHED_CRITIC_FORWARD_BUDGET": 1,
    }
    assert summary["scientific_role"].endswith("NOT_FULL_COHORT_STRONGEST_SELECTOR")
