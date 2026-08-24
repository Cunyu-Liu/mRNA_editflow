from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.route_a_v3.preflight_route2_xeditsetflow_v4 import (
    SetFlowPreflightV4Error,
    require_preflight_authorization_v4,
    select_train_geometry_sources_v4,
)


def _authorization() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_preflight_authorization.v1",
        "status": "XEDITSETFLOW_V4_PREFLIGHT_AUTHORIZED",
        "authorized_git_head": "head",
        "barriers": {
            "all_five_c3_jobs_terminal": True,
            "c3_terminal_summaries_read_exactly_once": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            "source_token_cache_terminal_complete": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_preflight_authorization_requires_terminal_read_once_tests_and_cache() -> None:
    authorization = _authorization()
    require_preflight_authorization_v4(authorization, current_git_head="head")
    authorization["barriers"]["c3_terminal_summaries_read_exactly_once"] = False
    with pytest.raises(SetFlowPreflightV4Error):
        require_preflight_authorization_v4(
            authorization, current_git_head="head"
        )


def test_preflight_authorization_rejects_wrong_head_or_protected_read() -> None:
    authorization = _authorization()
    with pytest.raises(SetFlowPreflightV4Error):
        require_preflight_authorization_v4(
            authorization, current_git_head="other"
        )
    authorization["development_test_outcome_reads"] = 1
    with pytest.raises(SetFlowPreflightV4Error):
        require_preflight_authorization_v4(
            authorization, current_git_head="head"
        )


@dataclass(frozen=True)
class _Record:
    source_id: str
    source: str
    terminal_edit_sets: tuple[tuple[tuple[int, str], ...], ...]


def test_geometry_selection_uses_length_then_edit_count_then_id() -> None:
    records = [
        _Record(str(index), "A" * (20 + index), (((0, "C"),),))
        for index in range(9)
    ]
    records[7] = _Record("z", "A" * 40, (((0, "C"),),))
    records[8] = _Record("a", "A" * 40, (((0, "C"), (1, "G")),))
    selected = select_train_geometry_sources_v4(records)
    assert selected[:2] == [8, 7]
    assert len(selected) == 8
