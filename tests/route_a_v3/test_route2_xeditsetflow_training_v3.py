from __future__ import annotations

import torch

from core.route2_xeditsetflow_training_v3 import (
    BalancedTaskSourceGroupPassSamplerV3,
    SetMarginalStateDatasetV3,
    XEditSetFlowRecordV3,
    balanced_task_allocations_v3,
    collate_setflow_states_v3,
    expanded_state_batches_v3,
    setflow_records_from_projection_rows,
    setflow_vocabs,
)
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, assemble_source_token_cache_v3


def _record(edit_count: int, budget: int) -> XEditSetFlowRecordV3:
    edits = tuple((index, "C") for index in range(edit_count))
    return XEditSetFlowRecordV3(
        record_id=f"r{edit_count}", split="TRAIN", source="A" * 10,
        terminal_edits=edits, assigned_budget=budget, task="t", study="s",
        source_group=f"g{edit_count}", assay="a", context="c", region=0,
        quantity="q", measurement="m", numerator="n", denominator="d",
    )


def test_projection_conversion_does_not_access_outcome_value() -> None:
    class ProtectedRow(dict):
        def __getitem__(self, key):
            if key == "direction_normalized_delta":
                raise AssertionError("outcome value was accessed")
            return super().__getitem__(key)

    row = ProtectedRow(
        canonical_record_id="r", split="TRAIN", source_sequence="AAAA",
        source_relative_edits=[{"position": 1, "candidate_base": "C"}],
        task_id="t", study_unit_id="s", source_group_id="g", assay_id="a",
        biological_context_id="c", region_id=0,
        endpoint_descriptor={
            "quantity_family": "q", "measurement_form": "m",
            "numerator_family": None, "denominator_family": None,
        },
        direction_normalized_delta=object(),
    )
    records, summary = setflow_records_from_projection_rows([row])
    assert len(records) == 1
    assert summary["outcome_value_access_count"] == 0


def test_incomplete_subset_marks_all_remaining_edits_and_never_stop() -> None:
    record = _record(2, 3)
    dataset = SetMarginalStateDatasetV3(
        [record], setflow_vocabs([record]), seed=7, states_per_record=3
    )
    for slot in (0, 1):
        state = dataset.state(0, slot)
        assert state["remaining_true_edits"]
        assert state["stop_positive"] is False


def test_complete_state_distinguishes_stop_from_budget_exhaustion() -> None:
    with_stop = _record(2, 3)
    exhausted = _record(3, 3)
    records = [with_stop, exhausted]
    dataset = SetMarginalStateDatasetV3(
        records, setflow_vocabs(records), seed=7, states_per_record=3
    )
    stop_state = dataset.state(0, 2)
    exhausted_state = dataset.state(1, 2)
    assert stop_state["stop_positive"] is True
    assert stop_state["structural_budget_exhausted"] is False
    assert exhausted_state["stop_positive"] is False
    assert exhausted_state["structural_budget_exhausted"] is True


def test_subset_sampling_replays_for_fixed_seed_and_pass() -> None:
    record = _record(5, 5)
    first = SetMarginalStateDatasetV3(
        [record], setflow_vocabs([record]), seed=9, states_per_record=3
    )
    second = SetMarginalStateDatasetV3(
        [record], setflow_vocabs([record]), seed=9, states_per_record=3
    )
    first.set_pass(2)
    second.set_pass(2)
    assert [first.state(0, slot) for slot in range(3)] == [
        second.state(0, slot) for slot in range(3)
    ]


def test_expansion_keeps_multiple_states_and_avoids_terminal_only_batch() -> None:
    batches = expanded_state_batches_v3(
        [[0, 1, 2], [3, 4]], states_per_record=3, batch_size=7
    )
    flat = [item for batch in batches for item in batch]
    assert len(flat) == 15
    assert all({slot for index, slot in flat if index == record} == {0, 1, 2} for record in range(5))


def test_task_balancing_redistributes_only_after_small_task_repeat_cap() -> None:
    allocation = balanced_task_allocations_v3({"large": 10, "small": 1}, draw_count=11, repeat_cap=4)
    assert allocation == {"large": 7, "small": 4}
    records = [
        XEditSetFlowRecordV3(**{**_record(1, 1).__dict__, "record_id": f"a{i}", "task": "large", "source_group": f"lg{i % 2}"})
        for i in range(10)
    ] + [
        XEditSetFlowRecordV3(**{**_record(1, 1).__dict__, "record_id": "small", "task": "small", "source_group": "sg"})
    ]
    sampler = BalancedTaskSourceGroupPassSamplerV3(records, record_batch_size=4, seed=8, repeat_cap=4)
    flat = [index for batch in sampler.batches_for_pass() for index in batch]
    assert len(flat) == len(records)
    assert max(flat.count(index) for index in set(flat)) <= 4


def test_collator_aligns_ragged_source_tokens_and_positive_set() -> None:
    record = _record(2, 3)
    cache = SourceTokenCacheIndexV3(assemble_source_token_cache_v3(
        [{"canonical_record_id": record.record_id, "source_sequence": record.source}],
        sequence_to_index={record.source: 0},
        encoded_tokens={0: torch.arange(40, dtype=torch.float32).reshape(10, 4)},
        model_id="frozen", pretrained_parameter_count=100_000_001,
        attention_backend="OFFICIAL_PYTORCH_FALLBACK",
    ))
    dataset = SetMarginalStateDatasetV3([record], setflow_vocabs([record]), seed=5, states_per_record=2)
    incomplete = dataset.state(0, 0)
    complete = dataset.state(0, 1)
    batch = collate_setflow_states_v3([incomplete, complete], source_cache=cache)
    assert batch["source_pretrained_tokens"].shape == (2, 10, 4)
    assert batch["positive_action_mask"][0].sum().item() == len(incomplete["remaining_true_edits"])
    assert batch["positive_action_mask"][0].sum().item() >= 1
    assert batch["positive_action_mask"][1, 40]
    assert not batch["structural_budget_exhausted"].any()
