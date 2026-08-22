"""Outcome-free set-state sampling and targets for XEditSetFlow V3."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Sampler

from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3
from core.route2_xeditcritic_training_data_v3 import descriptor_category


class XEditSetFlowTrainingError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowTrainingError(message)


@dataclass(frozen=True)
class XEditSetFlowRecordV3:
    record_id: str
    split: str
    source: str
    terminal_edits: tuple[tuple[int, str], ...]
    assigned_budget: int
    task: str
    study: str
    source_group: str
    assay: str
    context: str
    region: int
    quantity: str
    measurement: str
    numerator: str
    denominator: str


def assigned_edit_budget(edit_count: int, allowed_budgets: Sequence[int]) -> int:
    _require(edit_count >= 0, "edit count is negative")
    for budget in allowed_budgets:
        if edit_count <= budget:
            return int(budget)
    raise XEditSetFlowTrainingError("terminal edit set exceeds the largest budget")


def setflow_records_from_projection_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_budgets: Sequence[int] = (1, 3, 5),
) -> tuple[list[XEditSetFlowRecordV3], dict[str, int]]:
    """Build flow records without reading the projection target value."""

    _require(tuple(allowed_budgets) == (1, 3, 5), "SetFlow budgets must remain 1/3/5")
    records: list[XEditSetFlowRecordV3] = []
    skipped_over_budget = 0
    seen: set[str] = set()
    for row in rows:
        _require(row.get("split") in {"TRAIN", "VALIDATION"}, "protected split entered SetFlow records")
        record_id = str(row["canonical_record_id"])
        _require(record_id not in seen, "SetFlow projection record is duplicated")
        seen.add(record_id)
        source = str(row["source_sequence"])
        edits = tuple(
            (int(edit["position"]), str(edit["candidate_base"]))
            for edit in row["source_relative_edits"]
        )
        try:
            budget = assigned_edit_budget(len(edits), allowed_budgets)
        except XEditSetFlowTrainingError:
            skipped_over_budget += 1
            continue
        _require(
            tuple(sorted(edits)) == edits
            and len({position for position, _ in edits}) == len(edits),
            "terminal edit set is not a sorted unique-position set",
        )
        _require(
            all(
                0 <= position < len(source) and alt in "ACGU" and alt != source[position]
                for position, alt in edits
            ),
            "terminal edit set is not source-relative SUB",
        )
        descriptor = row["endpoint_descriptor"]
        records.append(
            XEditSetFlowRecordV3(
                record_id=record_id,
                split=str(row["split"]),
                source=source,
                terminal_edits=edits,
                assigned_budget=budget,
                task=str(row["task_id"]),
                study=str(row["study_unit_id"]),
                source_group=str(row["source_group_id"]),
                assay=str(row["assay_id"]),
                context=str(row["biological_context_id"]),
                region=int(row["region_id"]),
                quantity=str(descriptor["quantity_family"]),
                measurement=str(descriptor["measurement_form"]),
                numerator=descriptor_category(descriptor["numerator_family"]),
                denominator=descriptor_category(descriptor["denominator_family"]),
            )
        )
    _require(bool(records), "no SetFlow records satisfy the frozen edit budgets")
    return sorted(records, key=lambda record: record.record_id), {
        "input_record_count": len(seen),
        "eligible_record_count": len(records),
        "skipped_over_budget_count": skipped_over_budget,
        "outcome_value_access_count": 0,
    }


def setflow_vocabs(
    records: Iterable[XEditSetFlowRecordV3],
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


def task_source_group_weights_v3(
    records: Sequence[XEditSetFlowRecordV3],
) -> dict[str, float]:
    """Equal task, then equal source group, then equal rows within group."""

    groups_by_task: dict[str, set[str]] = {}
    group_sizes = Counter(record.source_group for record in records)
    for record in records:
        groups_by_task.setdefault(record.task, set()).add(record.source_group)
    raw = {
        record.record_id: 1.0
        / (
            len(groups_by_task)
            * len(groups_by_task[record.task])
            * group_sizes[record.source_group]
        )
        for record in records
    }
    scale = len(records) / sum(raw.values())
    return {record_id: value * scale for record_id, value in raw.items()}


class SetMarginalStateDatasetV3:
    """Multiple stochastic subsets per measured terminal edit set and pass."""

    def __init__(
        self,
        records: Sequence[XEditSetFlowRecordV3],
        vocabs: Mapping[str, Mapping[str, int]],
        *,
        seed: int,
        states_per_record: int = 3,
    ) -> None:
        _require(states_per_record >= 2, "each record needs multiple subset/progress states")
        self.records = list(records)
        self.vocabs = vocabs
        self.seed = int(seed)
        self.states_per_record = int(states_per_record)
        self.pass_index = 0
        # Task and source-group balance is enforced by the pass sampler.  Unit
        # loss weights avoid applying the same correction a second time.
        self.weights = {record.record_id: 1.0 for record in records}

    def set_pass(self, pass_index: int) -> None:
        _require(pass_index >= 0, "SetFlow pass index is negative")
        self.pass_index = int(pass_index)

    def state(self, record_index: int, state_slot: int) -> dict[str, Any]:
        record = self.records[record_index]
        _require(0 <= state_slot < self.states_per_record, "state slot is outside the frozen multiplicity")
        terminal = list(record.terminal_edits)
        rng = random.Random(
            self.seed
            + self.pass_index * len(self.records) * self.states_per_record
            + record_index * self.states_per_record
            + state_slot
        )
        if state_slot == self.states_per_record - 1:
            selected = set(terminal)
        elif terminal:
            subset_size = rng.randrange(len(terminal))
            selected = set(rng.sample(terminal, subset_size))
        else:
            selected = set()
        current = list(record.source)
        for position, alt in selected:
            current[position] = alt
        remaining_true_edits = tuple(
            edit for edit in record.terminal_edits if edit not in selected
        )
        remaining_budget = record.assigned_budget - len(selected)
        structural_budget_exhausted = remaining_budget == 0
        stop_positive = not remaining_true_edits and remaining_budget > 0
        _require(
            not (structural_budget_exhausted and stop_positive),
            "structural exhaustion was relabeled as STOP",
        )
        return {
            "record_id": record.record_id,
            "source_group": record.source_group,
            "task": record.task,
            "source_sequence": record.source,
            "current_sequence": "".join(current),
            "selected_edit_set": tuple(sorted(selected)),
            "remaining_true_edits": remaining_true_edits,
            "remaining_budget": remaining_budget,
            "assigned_budget": record.assigned_budget,
            "stop_positive": stop_positive,
            "structural_budget_exhausted": structural_budget_exhausted,
            "sample_weight": self.weights[record.record_id],
            "assay": self.vocabs["assay"].get(record.assay, 0),
            "context": self.vocabs["context"].get(record.context, 0),
            "quantity": self.vocabs["quantity"].get(record.quantity, 0),
            "measurement": self.vocabs["measurement"].get(record.measurement, 0),
            "numerator": self.vocabs["numerator"].get(record.numerator, 0),
            "denominator": self.vocabs["denominator"].get(record.denominator, 0),
            "region": record.region,
        }


def expanded_state_batches_v3(
    record_batches: Iterable[Sequence[int]],
    *,
    states_per_record: int,
    batch_size: int,
) -> list[list[tuple[int, int]]]:
    """Expand record draws while keeping each record's active/terminal states together."""

    _require(states_per_record >= 2 and batch_size >= states_per_record, "expanded state batch geometry is invalid")
    result: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for record_batch in record_batches:
        for record_index in record_batch:
            states = [
                (int(record_index), state_slot)
                for state_slot in range(states_per_record)
            ]
            if current and len(current) + len(states) > batch_size:
                result.append(current)
                current = []
            current.extend(states)
    if current:
        result.append(current)
    _require(
        all(
            any(not (state_slot == states_per_record - 1) for _, state_slot in batch)
            for batch in result
        ),
        "expanded batch contains only terminal state slots",
    )
    return result


def balanced_task_allocations_v3(
    task_sizes: Mapping[str, int], *, draw_count: int, repeat_cap: int = 4
) -> dict[str, int]:
    """Allocate equally over tasks, redistributing only after a task hits cap."""

    sizes = {str(task): int(size) for task, size in task_sizes.items()}
    _require(bool(sizes) and min(sizes.values()) > 0, "SetFlow task sizes are invalid")
    _require(draw_count > 0 and repeat_cap >= 1, "SetFlow draw geometry is invalid")
    _require(draw_count <= repeat_cap * sum(sizes.values()), "SetFlow draw count exceeds repeat capacity")
    allocations = {task: 0 for task in sizes}
    for _ in range(draw_count):
        eligible = [
            task for task, size in sizes.items()
            if allocations[task] < repeat_cap * size
        ]
        _require(bool(eligible), "SetFlow task allocation exhausted")
        selected = min(eligible, key=lambda task: (allocations[task], task))
        allocations[selected] += 1
    return allocations


class _BalancedGroupCycleV3:
    def __init__(
        self,
        groups: Mapping[str, Sequence[int]],
        *,
        rng: random.Random,
        repeat_cap: int,
    ) -> None:
        self.names = sorted(groups)
        rng.shuffle(self.names)
        self.rows = {name: list(groups[name]) for name in self.names}
        for values in self.rows.values():
            rng.shuffle(values)
        self.group_cursor = 0
        self.row_cursors = Counter()
        self.counts = Counter()
        self.repeat_cap = int(repeat_cap)

    def draw(self) -> int:
        for _ in range(sum(len(values) for values in self.rows.values()) * self.repeat_cap):
            group = self.names[self.group_cursor % len(self.names)]
            self.group_cursor += 1
            values = self.rows[group]
            for _ in range(len(values)):
                cursor = self.row_cursors[group]
                index = values[cursor % len(values)]
                self.row_cursors[group] += 1
                if self.counts[index] < self.repeat_cap:
                    self.counts[index] += 1
                    return index
        raise XEditSetFlowTrainingError("SetFlow source-group cycle is exhausted")


class BalancedTaskSourceGroupPassSamplerV3(Sampler[list[int]]):
    """One deterministic task/source-group-balanced record pass."""

    def __init__(
        self,
        records: Sequence[XEditSetFlowRecordV3],
        *,
        record_batch_size: int,
        seed: int,
        repeat_cap: int = 4,
    ) -> None:
        _require(bool(records) and record_batch_size > 0, "SetFlow sampler geometry is invalid")
        self.records = list(records)
        self.record_batch_size = int(record_batch_size)
        self.seed = int(seed)
        self.repeat_cap = int(repeat_cap)
        self.pass_index = 0
        self.allocations = balanced_task_allocations_v3(
            Counter(record.task for record in records),
            draw_count=len(records),
            repeat_cap=repeat_cap,
        )

    def set_pass(self, pass_index: int) -> None:
        _require(pass_index >= 0, "SetFlow sampler pass is negative")
        self.pass_index = int(pass_index)

    def batches_for_pass(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.pass_index)
        grouped: dict[str, dict[str, list[int]]] = {}
        for index, record in enumerate(self.records):
            grouped.setdefault(record.task, {}).setdefault(record.source_group, []).append(index)
        cycles = {
            task: _BalancedGroupCycleV3(groups, rng=rng, repeat_cap=self.repeat_cap)
            for task, groups in grouped.items()
        }
        draws = {
            task: [cycles[task].draw() for _ in range(self.allocations[task])]
            for task in sorted(grouped)
        }
        ordered: list[int] = []
        cursor = Counter()
        tasks = sorted(draws)
        while len(ordered) < len(self.records):
            for task in tasks:
                if cursor[task] < len(draws[task]):
                    ordered.append(draws[task][cursor[task]])
                    cursor[task] += 1
        batches = [
            ordered[start : start + self.record_batch_size]
            for start in range(0, len(ordered), self.record_batch_size)
        ]
        counts = Counter(ordered)
        _require(len(ordered) == len(self.records), "SetFlow pass draw count changed")
        _require(max(counts.values()) <= self.repeat_cap, "SetFlow record repeat cap was exceeded")
        return batches

    def __iter__(self):
        yield from self.batches_for_pass()

    def __len__(self) -> int:
        return math.ceil(len(self.records) / self.record_batch_size)


_TOKEN = {"A": 0, "C": 1, "G": 2, "U": 3}
_PAD = 4


def collate_setflow_states_v3(
    examples: Sequence[Mapping[str, Any]],
    *,
    source_cache: SourceTokenCacheIndexV3,
) -> dict[str, Any]:
    _require(bool(examples), "SetFlow state batch is empty")
    maximum = max(len(str(example["source_sequence"])) for example in examples)
    batch_size = len(examples)
    source = torch.full((batch_size, maximum), _PAD, dtype=torch.long)
    current = torch.full((batch_size, maximum), _PAD, dtype=torch.long)
    padding = torch.ones((batch_size, maximum), dtype=torch.bool)
    pretrained = torch.zeros(
        (batch_size, maximum, int(source_cache.payload["embedding_width"])),
        dtype=torch.float16,
    )
    positive = torch.zeros((batch_size, maximum * 4 + 1), dtype=torch.bool)
    for row_index, example in enumerate(examples):
        source_sequence = str(example["source_sequence"])
        current_sequence = str(example["current_sequence"])
        _require(len(source_sequence) == len(current_sequence), "SetFlow source/current length differs")
        length = len(source_sequence)
        source[row_index, :length] = torch.tensor([_TOKEN[base] for base in source_sequence])
        current[row_index, :length] = torch.tensor([_TOKEN[base] for base in current_sequence])
        padding[row_index, :length] = False
        cached = source_cache.tokens_for_record(str(example["record_id"]))
        _require(cached.shape[0] == length, "source-token cache does not align to the state")
        pretrained[row_index, :length] = cached
        for position, alt in example["remaining_true_edits"]:
            positive[row_index, int(position) * 4 + _TOKEN[str(alt)]] = True
        if bool(example["stop_positive"]):
            positive[row_index, maximum * 4] = True
    structural = torch.tensor(
        [bool(example["structural_budget_exhausted"]) for example in examples],
        dtype=torch.bool,
    )
    _require(
        bool(torch.all(positive.any(dim=1) == ~structural).item()),
        "SetFlow positive targets do not match structural terminal rows",
    )
    return {
        "record_ids": [str(example["record_id"]) for example in examples],
        "source_tokens": source,
        "current_tokens": current,
        "padding_mask": padding,
        "source_pretrained_tokens": pretrained,
        "remaining_budget": torch.tensor([int(example["remaining_budget"]) for example in examples]),
        "quantity_ids": torch.tensor([int(example["quantity"]) for example in examples]),
        "measurement_ids": torch.tensor([int(example["measurement"]) for example in examples]),
        "numerator_ids": torch.tensor([int(example["numerator"]) for example in examples]),
        "denominator_ids": torch.tensor([int(example["denominator"]) for example in examples]),
        "assay_ids": torch.tensor([int(example["assay"]) for example in examples]),
        "context_ids": torch.tensor([int(example["context"]) for example in examples]),
        "region_ids": torch.tensor([int(example["region"]) for example in examples]),
        "positive_action_mask": positive,
        "structural_budget_exhausted": structural,
        "sample_weight": torch.tensor([float(example["sample_weight"]) for example in examples], dtype=torch.float32),
    }
