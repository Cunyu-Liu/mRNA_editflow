from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.route_a_v3.preflight_route2_xeditcritic_v4 import (
    XEditCriticPreflightV4Error,
    build_preflight_vocabs_v4,
    preflight_example_v4,
    require_preflight_authorization_v4,
    select_train_geometry_records_v4,
)


ROOT = Path(__file__).resolve().parents[2]


def _row(record_id: str, split: str, length: int, edit_count: int) -> dict:
    source = "A" * length
    candidate = "C" * edit_count + "A" * (length - edit_count)
    return {
        "canonical_record_id": record_id,
        "split": split,
        "source_sequence": source,
        "candidate_sequence": candidate,
        "source_relative_edits": [
            {"position": index, "source_base": "A", "candidate_base": "C"}
            for index in range(edit_count)
        ],
        "source_group_id": "group",
        "task_id": "task",
        "study_unit_id": "study",
        "assay_id": "assay",
        "biological_context_id": "context",
        "region_id": 0,
        "endpoint_descriptor": {
            "quantity_family": "quantity",
            "measurement_form": "measurement",
            "numerator_family": None,
            "denominator_family": None,
        },
        "direction_normalized_delta": 999999.0,
    }


def _vocabs() -> dict:
    return {
        "study": {"__UNK__": 0, "study": 1},
        "assay": {"__UNK__": 0, "assay": 1},
        "context": {"__UNK__": 0, "context": 1},
        "quantity": {"__UNK__": 0, "quantity": 1},
        "measurement": {"__UNK__": 0, "measurement": 1},
        "numerator": {"__UNK__": 0, "__NONE__": 1},
        "denominator": {"__UNK__": 0, "__NONE__": 1},
    }


def test_preflight_binds_frozen_physical_gpu_scope_before_cuda() -> None:
    source = (
        ROOT / "scripts/route_a_v3/preflight_route2_xeditcritic_v4.py"
    ).read_text(encoding="utf-8")
    assert "require_physical_gpu_scope_v4(config, physical_gpu_index)" in source
    assert source.index(
        "require_physical_gpu_scope_v4(config, physical_gpu_index)"
    ) < source.index("device = require_cuda(physical_gpu_index)")


def test_preflight_authorization_requires_terminal_read_once_sync_tests_and_cache() -> None:
    authorization = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_preflight_authorization.v1",
        "status": "XEDITCRITIC_V4_PREFLIGHT_AUTHORIZED",
        "authorized_git_head": "head",
        "barriers": {
            "all_five_c3_jobs_terminal": True,
            "c3_terminal_summaries_read_exactly_once": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            "bottom_six_cache_terminal_complete": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    require_preflight_authorization_v4(authorization, current_git_head="head")
    authorization["barriers"]["bottom_six_cache_terminal_complete"] = False
    with pytest.raises(XEditCriticPreflightV4Error, match="barrier"):
        require_preflight_authorization_v4(
            authorization, current_git_head="head"
        )


def test_geometry_selection_uses_train_edit_and_length_only() -> None:
    rows = [
        _row(f"train-{index:02d}", "TRAIN", 100 + index, 1)
        for index in range(32)
    ]
    rows.extend(
        [
            _row("train-many-edits", "TRAIN", 80, 5),
            _row("validation-many-edits", "VALIDATION", 1000, 10),
        ]
    )
    selected = select_train_geometry_records_v4(rows)
    selected_ids = [row["canonical_record_id"] for row in selected]
    assert selected_ids[0] == "train-many-edits"
    assert "validation-many-edits" not in selected_ids
    assert len(selected_ids) == 32


def test_preflight_example_ignores_target_and_uses_structural_placeholders() -> None:
    row = _row("train", "TRAIN", 16, 2)
    example = preflight_example_v4(row, _vocabs())
    assert example["target"] == 0.0
    assert example["scaled_target"] == 0.0
    assert example["sample_weight"] == 1.0
    assert example["edits"] == ((0, "A", "C"), (1, "A", "C"))
    row["direction_normalized_delta"] = -999999.0
    changed_outcome = preflight_example_v4(row, _vocabs())
    assert torch.equal(changed_outcome.pop("source"), example.pop("source"))
    assert torch.equal(changed_outcome.pop("candidate"), example.pop("candidate"))
    assert changed_outcome == example


def test_preflight_vocab_is_outcome_free_and_matches_unknown_plus_sorted_categories() -> None:
    first = _row("a", "TRAIN", 16, 1)
    second = _row("b", "VALIDATION", 16, 1)
    second["study_unit_id"] = "another-study"
    second["endpoint_descriptor"]["quantity_family"] = "another-quantity"
    before = build_preflight_vocabs_v4([first, second])
    first["direction_normalized_delta"] = -1e30
    second["direction_normalized_delta"] = 1e30
    after = build_preflight_vocabs_v4([first, second])
    assert after == before
    assert before["study"] == {
        "__UNK__": 0,
        "another-study": 1,
        "study": 2,
    }
    assert before["quantity"] == {
        "__UNK__": 0,
        "another-quantity": 1,
        "quantity": 2,
    }
