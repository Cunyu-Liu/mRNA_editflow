from __future__ import annotations

import random

import pytest
import torch

from core.route2_source_token_cache_v3 import (
    SourceTokenCacheIndexV3,
    assemble_source_token_cache_v3,
)
from core.route2_xeditsetflow_training_v4 import (
    SetFlowSourceStateDatasetV4,
    collate_setflow_source_states_v4,
    expanded_source_state_batches_v4,
    setflow_source_records_from_projection_rows_v4,
    setflow_source_vocabs_v4,
)


def _row(record_id: str, edits: list[tuple[int, str]], *, duplicate: bool = False) -> dict:
    del duplicate
    source = "AAAAAA"
    candidate = list(source)
    for position, alt in edits:
        candidate[position] = alt
    return {
        "canonical_record_id": record_id,
        "split": "TRAIN",
        "source_sequence": source,
        "candidate_sequence": "".join(candidate),
        "source_relative_edits": [
            {"position": position, "source_base": source[position], "candidate_base": alt}
            for position, alt in edits
        ],
        "task_id": "task",
        "endpoint_id": "endpoint",
        "source_group_id": "legacy-group",
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
        "direction_normalized_delta": 123456.0,
    }


def _records():
    rows = [
        _row("candidate-a", [(0, "C")]),
        _row("candidate-a-duplicate", [(0, "C")]),
        _row("candidate-b", [(1, "G"), (2, "U")]),
        _row("candidate-c", [(3, "C"), (4, "G"), (5, "U")]),
    ]
    records, audit = setflow_source_records_from_projection_rows_v4(rows)
    return rows, records, audit


def _cache(rows) -> SourceTokenCacheIndexV3:
    source = rows[0]["source_sequence"]
    return SourceTokenCacheIndexV3(
        assemble_source_token_cache_v3(
            rows,
            sequence_to_index={source: 0},
            encoded_tokens={0: torch.arange(6 * 8).reshape(6, 8).float()},
            model_id="fixed",
            pretrained_parameter_count=10,
            attention_backend="fixed",
        )
    )


def test_source_grouping_deduplicates_terminal_sets_and_ignores_candidate_row_order() -> None:
    rows, records, audit = _records()
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    shuffled_records, shuffled_audit = setflow_source_records_from_projection_rows_v4(
        shuffled
    )
    assert records == shuffled_records
    assert len(records) == 1
    assert len(records[0].terminal_edit_sets) == 3
    assert records[0].candidate_row_count == 4
    assert records[0].duplicate_candidate_row_count == 1
    assert audit == shuffled_audit
    assert audit["outcome_value_access_count"] == 0


def test_four_states_use_empty_two_distinct_partial_anchors_and_completed_or_structural() -> None:
    _, records, _ = _records()
    dataset = SetFlowSourceStateDatasetV4(
        records, setflow_source_vocabs_v4(records), seed=20260911
    )
    dataset.set_pass(2)
    states = [dataset.state(0, slot) for slot in range(4)]
    assert [state["state_kind"] for state in states] == [
        "EMPTY",
        "PARTIAL",
        "PARTIAL",
        "COMPLETED_OR_STRUCTURAL",
    ]
    assert states[0]["selected_edit_set"] == ()
    assert states[1]["anchor_candidate_index"] != states[2]["anchor_candidate_index"]
    assert states[1]["partial_anchor_is_distinct_from_other_partial"] is True
    assert states[3]["common_stop_positive"] != states[3]["structural_budget_exhausted"]
    assert all(sum(state["remaining_count_soft_target"]) == 1.0 for state in states)
    assert [state["state_slot"] for state in states] == [0, 1, 2, 3]
    for state in states:
        assert tuple(
            records[0].terminal_edit_sets[candidate_index]
            for candidate_index in state[
                "compatible_canonical_candidate_indices"
            ]
        ) == state["compatible_terminal_edit_sets"]


def test_source_states_are_replayable_by_seed_and_pass() -> None:
    _, records, _ = _records()
    vocabs = setflow_source_vocabs_v4(records)
    first = SetFlowSourceStateDatasetV4(records, vocabs, seed=20260911)
    second = SetFlowSourceStateDatasetV4(records, vocabs, seed=20260911)
    first.set_pass(5)
    second.set_pass(5)
    assert [first.state(0, slot) for slot in range(4)] == [
        second.state(0, slot) for slot in range(4)
    ]


def test_source_state_collator_keeps_each_candidate_constraint_separate() -> None:
    rows, records, _ = _records()
    dataset = SetFlowSourceStateDatasetV4(
        records, setflow_source_vocabs_v4(records), seed=20260911
    )
    examples = [dataset.state(0, slot) for slot in range(4)]
    batch = collate_setflow_source_states_v4(examples, source_cache=_cache(rows))
    assert batch["source_tokens"].shape == (4, 6)
    assert batch["candidate_positive_action_mask"].shape[:2] == (4, 3)
    assert batch["candidate_valid_mask"][0].sum().item() == 3
    assert batch["candidate_positive_action_mask"][0, 0].sum().item() == 1
    assert batch["candidate_positive_action_mask"][0, 1].sum().item() == 2
    assert batch["candidate_positive_action_mask"][0, 2].sum().item() == 3
    assert torch.allclose(
        batch["remaining_count_soft_target"].sum(dim=1), torch.ones(4)
    )
    assert torch.equal(batch["state_slots"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(batch["source_occurrence_ids"], torch.zeros(4, dtype=torch.long))
    assert torch.equal(
        batch["canonical_candidate_indices"] >= 0,
        batch["candidate_valid_mask"],
    )
    for row_index, state in enumerate(examples):
        valid_count = int(batch["candidate_valid_mask"][row_index].sum().item())
        assert tuple(
            batch["canonical_candidate_indices"][row_index, :valid_count].tolist()
        ) == state["compatible_canonical_candidate_indices"]


def test_collator_isolates_repeated_source_draws_as_distinct_occurrences() -> None:
    rows, records, _ = _records()
    dataset = SetFlowSourceStateDatasetV4(
        records, setflow_source_vocabs_v4(records), seed=20260911
    )
    one_draw = [dataset.state(0, slot) for slot in range(4)]
    batch = collate_setflow_source_states_v4(
        one_draw + one_draw,
        source_cache=_cache(rows),
    )
    assert batch["source_ids"][:4] == batch["source_ids"][4:]
    assert torch.equal(
        batch["state_slots"],
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
    )
    assert torch.equal(
        batch["source_occurrence_ids"],
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )


def test_collator_rejects_noncontiguous_or_conflicting_canonical_state_blocks() -> None:
    rows, records, _ = _records()
    dataset = SetFlowSourceStateDatasetV4(
        records, setflow_source_vocabs_v4(records), seed=20260911
    )
    states = [dataset.state(0, slot) for slot in range(4)]
    with pytest.raises(RuntimeError, match="contiguous slot-0/1/2/3"):
        collate_setflow_source_states_v4(
            [states[0], states[2], states[1], states[3]],
            source_cache=_cache(rows),
        )

    poisoned = [dict(state) for state in states]
    poisoned[0]["compatible_canonical_candidate_indices"] = (0, 0, 2)
    with pytest.raises(RuntimeError, match="canonical candidate identities are invalid"):
        collate_setflow_source_states_v4(poisoned, source_cache=_cache(rows))


def test_expansion_never_splits_one_sources_four_states() -> None:
    batches = expanded_source_state_batches_v4(
        [list(range(8)), list(range(8, 11))]
    )
    assert [len(batch) for batch in batches] == [32, 12]
    assert all(len(batch) % 4 == 0 for batch in batches)
    assert all(
        {slot for source, slot in batch if source == source_index} == {0, 1, 2, 3}
        for batch in batches
        for source_index in {source for source, _ in batch}
    )
