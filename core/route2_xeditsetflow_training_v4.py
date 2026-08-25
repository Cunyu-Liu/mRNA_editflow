"""Source-level terminal-set states and targets for XEditSetFlow V4."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3
from core.route2_xeditcritic_training_data_v3 import descriptor_category
from core.route2_xedit_v4_interfaces import SetFlowSourceBatchV4
from core.route2_xeditsetflow_training_v3 import (
    XEditSetFlowTrainingError,
    assigned_edit_budget,
)


TOKEN_V4 = {"A": 0, "C": 1, "G": 2, "U": 3}
PAD_V4 = 4
EXPECTED_VALIDATION_SOURCE_RECORD_COUNT_V4 = 15_327


class XEditSetFlowTrainingV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowTrainingV4Error(message)


@dataclass(frozen=True)
class SetFlowSourceRecordV4:
    source_id: str
    split: str
    source: str
    cache_record_id: str
    terminal_edit_sets: tuple[tuple[tuple[int, str], ...], ...]
    candidate_row_count: int
    duplicate_candidate_row_count: int
    task: str
    source_group: str
    endpoint_id: str
    studies: tuple[str, ...]
    assay: str
    context: str
    region: int
    quantity: str
    measurement: str
    numerator: str
    denominator: str

    @property
    def record_id(self) -> str:
        return self.source_id


def _terminal_edit_set(row: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    source = str(row["source_sequence"])
    edits = tuple(
        (int(edit["position"]), str(edit["candidate_base"]))
        for edit in row["source_relative_edits"]
    )
    _require(
        tuple(sorted(edits)) == edits
        and len({position for position, _ in edits}) == len(edits),
        "V4 terminal edit set is not sorted with unique positions",
    )
    _require(
        all(
            0 <= position < len(source)
            and alt in TOKEN_V4
            and alt != source[position]
            for position, alt in edits
        ),
        "V4 terminal edit set is not source-relative SUB",
    )
    return edits


def setflow_source_records_from_projection_rows_v4(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_budgets: Sequence[int] = (1, 3, 5),
) -> tuple[list[SetFlowSourceRecordV4], dict[str, int]]:
    """Group eligible rows by source semantics and equalize unique candidates."""

    _require(tuple(allowed_budgets) == (1, 3, 5), "SetFlow V4 budgets must remain 1/3/5")
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    input_count = 0
    eligible_count = 0
    over_budget = 0
    seen_records: set[str] = set()
    for row in rows:
        input_count += 1
        _require(row.get("split") in {"TRAIN", "VALIDATION"}, "protected split entered SetFlow V4")
        record_id = str(row["canonical_record_id"])
        _require(record_id not in seen_records, "SetFlow V4 projection record is duplicated")
        seen_records.add(record_id)
        edits = _terminal_edit_set(row)
        try:
            assigned_edit_budget(len(edits), allowed_budgets)
        except XEditSetFlowTrainingError:
            over_budget += 1
            continue
        eligible_count += 1
        key = (
            str(row["split"]),
            str(row["source_sequence"]),
            str(row["task_id"]),
            str(row["endpoint_id"]),
            str(row["biological_context_id"]),
        )
        groups.setdefault(key, []).append(row)
    records: list[SetFlowSourceRecordV4] = []
    single_candidate_sources = 0
    candidate_rows = 0
    unique_candidates = 0
    duplicate_rows = 0
    for key, members in sorted(groups.items()):
        split, source, task, endpoint_id, context = key
        descriptor_rows = [member["endpoint_descriptor"] for member in members]
        descriptor_signatures = {
            json.dumps(descriptor, sort_keys=True)
            for descriptor in descriptor_rows
        }
        _require(len(descriptor_signatures) == 1, "one SetFlow V4 source has conflicting endpoint descriptors")
        _require(len({str(member["assay_id"]) for member in members}) == 1, "one SetFlow V4 source has conflicting assays")
        _require(len({int(member["region_id"]) for member in members}) == 1, "one SetFlow V4 source has conflicting regions")
        terminal_sets = tuple(
            sorted(
                {_terminal_edit_set(member) for member in members},
                key=lambda edits: (len(edits), edits),
            )
        )
        _require(bool(terminal_sets), "SetFlow V4 source has no terminal candidate")
        descriptor = descriptor_rows[0]
        source_id = json.dumps(
            {
                "split": split,
                "source_sequence": source,
                "task_id": task,
                "endpoint_id": endpoint_id,
                "biological_context_id": context,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        row_count = len(members)
        duplicate_count = row_count - len(terminal_sets)
        if len(terminal_sets) == 1:
            single_candidate_sources += 1
        candidate_rows += row_count
        unique_candidates += len(terminal_sets)
        duplicate_rows += duplicate_count
        records.append(
            SetFlowSourceRecordV4(
                source_id=source_id,
                split=split,
                source=source,
                cache_record_id=min(str(member["canonical_record_id"]) for member in members),
                terminal_edit_sets=terminal_sets,
                candidate_row_count=row_count,
                duplicate_candidate_row_count=duplicate_count,
                task=task,
                source_group=source_id,
                endpoint_id=endpoint_id,
                studies=tuple(sorted({str(member["study_unit_id"]) for member in members})),
                assay=str(members[0]["assay_id"]),
                context=context,
                region=int(members[0]["region_id"]),
                quantity=str(descriptor["quantity_family"]),
                measurement=str(descriptor["measurement_form"]),
                numerator=descriptor_category(descriptor["numerator_family"]),
                denominator=descriptor_category(descriptor["denominator_family"]),
            )
        )
    _require(bool(records), "SetFlow V4 has no eligible source records")
    return records, {
        "input_candidate_row_count": input_count,
        "eligible_candidate_row_count": eligible_count,
        "over_budget_excluded_candidate_row_count": over_budget,
        "source_count": len(records),
        "candidate_row_count_after_budget_filter": candidate_rows,
        "unique_terminal_candidate_count": unique_candidates,
        "duplicate_candidate_row_count": duplicate_rows,
        "single_unique_candidate_source_count": single_candidate_sources,
        "outcome_value_access_count": 0,
    }


def setflow_source_vocabs_v4(
    records: Iterable[SetFlowSourceRecordV4],
) -> dict[str, dict[str, int]]:
    rows = list(records)
    fields = (
        "assay",
        "context",
        "quantity",
        "measurement",
        "numerator",
        "denominator",
    )
    return {
        field: {"__UNK__": 0}
        | {
            value: index + 1
            for index, value in enumerate(
                sorted({str(getattr(record, field)) for record in rows})
            )
        }
        for field in fields
    }


def _compatible_candidates(
    terminal_sets: Sequence[tuple[tuple[int, str], ...]],
    selected: set[tuple[int, str]],
    assigned_budget: int,
) -> tuple[tuple[tuple[int, str], ...], ...]:
    return tuple(
        edits
        for edits in terminal_sets
        if len(edits) <= assigned_budget and selected <= set(edits)
    )


class SetFlowSourceStateDatasetV4:
    """Exactly four source-level states per source and pass."""

    states_per_source = 4

    def __init__(
        self,
        records: Sequence[SetFlowSourceRecordV4],
        vocabs: Mapping[str, Mapping[str, int]],
        *,
        seed: int,
    ) -> None:
        _require(bool(records), "SetFlow V4 source-state dataset is empty")
        self.records = list(records)
        self.vocabs = vocabs
        self.seed = int(seed)
        self.pass_index = 0

    def set_pass(self, pass_index: int) -> None:
        _require(pass_index >= 0, "SetFlow V4 pass index is negative")
        self.pass_index = int(pass_index)

    def _candidate_order(self, source_index: int) -> list[int]:
        order = list(range(len(self.records[source_index].terminal_edit_sets)))
        random.Random(
            self.seed + self.pass_index * len(self.records) + source_index
        ).shuffle(order)
        return order

    def state(self, source_index: int, state_slot: int) -> dict[str, Any]:
        _require(0 <= state_slot < self.states_per_source, "SetFlow V4 state slot is outside 0..3")
        record = self.records[source_index]
        candidates = record.terminal_edit_sets
        order = self._candidate_order(source_index)
        if state_slot == 0:
            # The empty state uses the largest candidate budget so source-level
            # coverage sees every measured terminal set that is legal under the
            # source's maximum observed 1/3/5 budget.
            anchor_index = max(
                range(len(candidates)),
                key=lambda index: (
                    assigned_edit_budget(len(candidates[index]), (1, 3, 5)),
                    len(candidates[index]),
                    candidates[index],
                ),
            )
            selected: set[tuple[int, str]] = set()
            state_kind = "EMPTY"
        elif state_slot in {1, 2}:
            anchor_index = order[(state_slot - 1) % len(order)]
            anchor = candidates[anchor_index]
            rng = random.Random(
                self.seed
                + self.pass_index * len(self.records) * 8
                + source_index * 8
                + state_slot
            )
            subset_size = 0 if not anchor else rng.randrange(len(anchor))
            selected = set(rng.sample(list(anchor), subset_size))
            state_kind = "PARTIAL"
        else:
            anchor_index = order[3 % len(order)]
            selected = set(candidates[anchor_index])
            state_kind = "COMPLETED_OR_STRUCTURAL"
        anchor = candidates[anchor_index]
        budget = assigned_edit_budget(len(anchor), (1, 3, 5))
        remaining_budget = budget - len(selected)
        _require(remaining_budget >= 0, "SetFlow V4 selected edits exceed assigned budget")
        compatible = _compatible_candidates(candidates, selected, budget)
        _require(anchor in compatible, "SetFlow V4 anchor candidate is not compatible with its state")
        remaining_anchor = tuple(edit for edit in anchor if edit not in selected)
        structural = remaining_budget == 0
        common_stop = not remaining_anchor and remaining_budget > 0
        _require(not (structural and common_stop), "SetFlow V4 structural terminal was relabeled STOP")
        current = list(record.source)
        for position, alt in selected:
            current[position] = alt
        remaining_counts = [len(set(candidate) - selected) for candidate in compatible]
        count_target = [0.0] * 6
        for count in remaining_counts:
            _require(0 <= count <= 5, "SetFlow V4 remaining edit count is outside 0..5")
            count_target[count] += 1.0 / len(remaining_counts)
        return {
            "source_id": record.source_id,
            "record_id": record.cache_record_id,
            "task": record.task,
            "source_sequence": record.source,
            "current_sequence": "".join(current),
            "selected_edit_set": tuple(sorted(selected)),
            "anchor_terminal_edit_set": anchor,
            "anchor_candidate_index": anchor_index,
            "compatible_terminal_edit_sets": compatible,
            "partial_anchor_is_distinct_from_other_partial": len(candidates) >= 2,
            "state_kind": state_kind,
            "remaining_true_edits": remaining_anchor,
            "remaining_budget": remaining_budget,
            "assigned_budget": budget,
            "common_stop_positive": common_stop,
            "structural_budget_exhausted": structural,
            "remaining_count_soft_target": tuple(count_target),
            "sample_weight": 1.0,
            "assay": self.vocabs["assay"].get(record.assay, 0),
            "context": self.vocabs["context"].get(record.context, 0),
            "quantity": self.vocabs["quantity"].get(record.quantity, 0),
            "measurement": self.vocabs["measurement"].get(record.measurement, 0),
            "numerator": self.vocabs["numerator"].get(record.numerator, 0),
            "denominator": self.vocabs["denominator"].get(record.denominator, 0),
            "region": record.region,
        }


def expanded_source_state_batches_v4(
    source_batches: Iterable[Sequence[int]], *, batch_size: int = 32
) -> list[list[tuple[int, int]]]:
    _require(batch_size == 32, "SetFlow V4 state batch must equal 32")
    result: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for source_batch in source_batches:
        for source_index in source_batch:
            states = [(int(source_index), slot) for slot in range(4)]
            if current and len(current) + 4 > batch_size:
                result.append(current)
                current = []
            current.extend(states)
    if current:
        result.append(current)
    _require(all(len(batch) % 4 == 0 for batch in result), "SetFlow V4 split one source's four states")
    return result


def collate_setflow_source_states_v4(
    examples: Sequence[Mapping[str, Any]],
    *,
    source_cache: SourceTokenCacheIndexV3,
) -> SetFlowSourceBatchV4:
    _require(bool(examples), "SetFlow V4 state batch is empty")
    maximum_length = max(len(str(example["source_sequence"])) for example in examples)
    maximum_candidates = max(len(example["compatible_terminal_edit_sets"]) for example in examples)
    batch_size = len(examples)
    action_count = maximum_length * 4 + 1
    source = torch.full((batch_size, maximum_length), PAD_V4, dtype=torch.long)
    current = torch.full_like(source, PAD_V4)
    padding = torch.ones_like(source, dtype=torch.bool)
    pretrained = torch.zeros(
        (batch_size, maximum_length, int(source_cache.payload["embedding_width"])),
        dtype=torch.float16,
    )
    common_positive = torch.zeros((batch_size, action_count), dtype=torch.bool)
    candidate_positive = torch.zeros(
        (batch_size, maximum_candidates, action_count), dtype=torch.bool
    )
    candidate_valid = torch.zeros((batch_size, maximum_candidates), dtype=torch.bool)
    for row_index, example in enumerate(examples):
        source_sequence = str(example["source_sequence"])
        current_sequence = str(example["current_sequence"])
        _require(len(source_sequence) == len(current_sequence), "SetFlow V4 source/current length differs")
        length = len(source_sequence)
        source[row_index, :length] = torch.tensor(
            [TOKEN_V4[base] for base in source_sequence]
        )
        current[row_index, :length] = torch.tensor(
            [TOKEN_V4[base] for base in current_sequence]
        )
        padding[row_index, :length] = False
        cached = source_cache.tokens_for_record(str(example["record_id"]))
        _require(cached.shape[0] == length, "SetFlow V4 source cache does not align")
        pretrained[row_index, :length] = cached
        for position, alt in example["remaining_true_edits"]:
            common_positive[row_index, int(position) * 4 + TOKEN_V4[str(alt)]] = True
        if bool(example["common_stop_positive"]):
            common_positive[row_index, maximum_length * 4] = True
        selected = set(example["selected_edit_set"])
        remaining_budget = int(example["remaining_budget"])
        for candidate_index, candidate in enumerate(
            example["compatible_terminal_edit_sets"]
        ):
            candidate_valid[row_index, candidate_index] = True
            remaining = [edit for edit in candidate if edit not in selected]
            for position, alt in remaining:
                candidate_positive[
                    row_index,
                    candidate_index,
                    int(position) * 4 + TOKEN_V4[str(alt)],
                ] = True
            if not remaining and remaining_budget > 0:
                candidate_positive[
                    row_index, candidate_index, maximum_length * 4
                ] = True
    structural = torch.tensor(
        [bool(example["structural_budget_exhausted"]) for example in examples],
        dtype=torch.bool,
    )
    _require(
        bool(torch.all(common_positive.any(dim=1) == ~structural).item()),
        "SetFlow V4 common target presence differs from structural terminal",
    )
    _require(
        bool(
            torch.all(
                candidate_positive.any(dim=2)
                == (candidate_valid & ~structural[:, None])
            ).item()
        ),
        "SetFlow V4 per-candidate targets differ from structural terminal",
    )
    return {
        "source_ids": [str(example["source_id"]) for example in examples],
        "record_ids": [str(example["record_id"]) for example in examples],
        "task_ids": [str(example["task"]) for example in examples],
        "state_kinds": [str(example["state_kind"]) for example in examples],
        "source_tokens": source,
        "current_tokens": current,
        "padding_mask": padding,
        "source_pretrained_tokens": pretrained,
        "remaining_budget": torch.tensor(
            [int(example["remaining_budget"]) for example in examples]
        ),
        "quantity_ids": torch.tensor([int(example["quantity"]) for example in examples]),
        "measurement_ids": torch.tensor([int(example["measurement"]) for example in examples]),
        "numerator_ids": torch.tensor([int(example["numerator"]) for example in examples]),
        "denominator_ids": torch.tensor([int(example["denominator"]) for example in examples]),
        "assay_ids": torch.tensor([int(example["assay"]) for example in examples]),
        "context_ids": torch.tensor([int(example["context"]) for example in examples]),
        "region_ids": torch.tensor([int(example["region"]) for example in examples]),
        "common_positive_action_mask": common_positive,
        "candidate_positive_action_mask": candidate_positive,
        "candidate_valid_mask": candidate_valid,
        "remaining_count_soft_target": torch.tensor(
            [example["remaining_count_soft_target"] for example in examples],
            dtype=torch.float32,
        ),
        "structural_budget_exhausted": structural,
        "sample_weight": torch.tensor(
            [float(example["sample_weight"]) for example in examples],
            dtype=torch.float32,
        ),
    }
