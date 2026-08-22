"""Projection-only records, controls, and frozen pass sampler for Critic V3."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Sampler


RNA_TOKEN = {"A": 0, "C": 1, "G": 2, "U": 3}
PAD_TOKEN = 4
NONE_CATEGORY = "__NONE__"
UNKNOWN_CATEGORY = "__UNK__"


class XEditCriticTrainingDataError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticTrainingDataError(message)


def descriptor_category(value: Any) -> str:
    return NONE_CATEGORY if value is None else str(value)


@dataclass(frozen=True)
class XEditCriticRecordV3:
    record_id: str
    split: str
    source: str
    candidate: str
    edits: tuple[tuple[int, str, str], ...]
    target: float
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


def records_from_projection_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[XEditCriticRecordV3]:
    records: list[XEditCriticRecordV3] = []
    seen: set[str] = set()
    for row in rows:
        _require(row.get("split") in {"TRAIN", "VALIDATION"}, "protected split entered Critic V3 records")
        record_id = str(row["canonical_record_id"])
        _require(record_id not in seen, "projection record is duplicated")
        seen.add(record_id)
        source = str(row["source_sequence"])
        candidate = str(row["candidate_sequence"])
        _require(len(source) == len(candidate), "length-changing record entered Critic V3")
        descriptor = row["endpoint_descriptor"]
        edits = tuple(
            (
                int(edit["position"]),
                str(edit["source_base"]),
                str(edit["candidate_base"]),
            )
            for edit in row["source_relative_edits"]
        )
        _require(
            edits
            == tuple(
                (index, left, right)
                for index, (left, right) in enumerate(zip(source, candidate))
                if left != right
            ),
            "projection edit bundle differs from the sequence pair",
        )
        target = float(row["direction_normalized_delta"])
        _require(math.isfinite(target), "Critic V3 target is nonfinite")
        records.append(
            XEditCriticRecordV3(
                record_id=record_id,
                split=str(row["split"]),
                source=source,
                candidate=candidate,
                edits=edits,
                target=target,
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
    _require(bool(records), "Critic V3 projection records are empty")
    return sorted(records, key=lambda record: record.record_id)


def build_vocabs(records: Iterable[XEditCriticRecordV3]) -> dict[str, dict[str, int]]:
    rows = list(records)
    fields = (
        "study",
        "assay",
        "context",
        "quantity",
        "measurement",
        "numerator",
        "denominator",
    )
    return {
        field: {UNKNOWN_CATEGORY: 0}
        | {
            value: index + 1
            for index, value in enumerate(
                sorted({str(getattr(record, field)) for record in rows})
            )
        }
        for field in fields
    }


def capped_sqrt_task_allocations(
    task_sizes: Mapping[str, int],
    *,
    draw_count: int,
    repeat_cap: int = 4,
) -> dict[str, int]:
    """Allocate one pass in sqrt(task size) proportions without exceeding cap."""

    checked = {str(task): int(size) for task, size in task_sizes.items()}
    _require(bool(checked) and min(checked.values()) > 0, "task sizes are invalid")
    _require(draw_count > 0, "pass draw count must be positive")
    _require(repeat_cap >= 1, "repeat cap must be positive")
    _require(draw_count <= repeat_cap * sum(checked.values()), "draw count exceeds repeat capacity")
    allocations = {task: 0 for task in checked}
    weights = {task: math.sqrt(size) for task, size in checked.items()}
    # A weighted fair queue is deterministic and naturally redistributes draws
    # after a small task reaches its four-per-record capacity.
    for _ in range(draw_count):
        eligible = [
            task
            for task, size in checked.items()
            if allocations[task] < repeat_cap * size
        ]
        _require(bool(eligible), "task allocation exhausted unexpectedly")
        selected = min(
            eligible,
            key=lambda task: ((allocations[task] + 1) / weights[task], task),
        )
        allocations[selected] += 1
    _require(sum(allocations.values()) == draw_count, "task allocation count changed")
    return allocations


class _RecordCycle:
    def __init__(self, indices: Sequence[int], rng: random.Random, repeat_cap: int) -> None:
        self.indices = list(indices)
        rng.shuffle(self.indices)
        self.repeat_cap = repeat_cap
        self.counts = Counter()
        self.cursor = 0

    @property
    def available(self) -> bool:
        return sum(self.counts.values()) < self.repeat_cap * len(self.indices)

    def draw(self) -> int:
        _require(self.available, "record cycle is exhausted")
        for _ in range(len(self.indices)):
            index = self.indices[self.cursor % len(self.indices)]
            self.cursor += 1
            if self.counts[index] < self.repeat_cap:
                self.counts[index] += 1
                return index
        raise XEditCriticTrainingDataError("record cycle has capacity but no eligible row")


class _GroupCycle:
    def __init__(self, groups: Mapping[str, Sequence[int]], rng: random.Random, repeat_cap: int) -> None:
        names = sorted(groups)
        rng.shuffle(names)
        self.names = names
        self.groups = {
            name: _RecordCycle(groups[name], rng, repeat_cap) for name in names
        }
        self.cursor = 0

    @property
    def available(self) -> bool:
        return any(group.available for group in self.groups.values())

    def draw(self) -> int:
        _require(self.available, "source-group cycle is exhausted")
        for _ in range(len(self.names)):
            name = self.names[self.cursor % len(self.names)]
            self.cursor += 1
            if self.groups[name].available:
                return self.groups[name].draw()
        raise XEditCriticTrainingDataError("source-group cycle has capacity but no eligible group")


class _StudyCycle:
    def __init__(
        self,
        studies: Mapping[str, Mapping[str, Sequence[int]]],
        rng: random.Random,
        repeat_cap: int,
    ) -> None:
        names = sorted(studies)
        rng.shuffle(names)
        self.names = names
        self.studies = {
            name: _GroupCycle(studies[name], rng, repeat_cap) for name in names
        }
        self.cursor = 0

    def draw(self) -> int:
        _require(any(study.available for study in self.studies.values()), "study cycle is exhausted")
        for _ in range(len(self.names)):
            name = self.names[self.cursor % len(self.names)]
            self.cursor += 1
            if self.studies[name].available:
                return self.studies[name].draw()
        raise XEditCriticTrainingDataError("study cycle has capacity but no eligible study")


class SqrtTaskStudySourcePassSamplerV3(Sampler[list[int]]):
    """Task-homogeneous pass sampler with sqrt tasks and <=4 row repeats."""

    def __init__(
        self,
        records: Sequence[XEditCriticRecordV3],
        *,
        batch_size: int,
        seed: int,
        repeat_cap: int = 4,
    ) -> None:
        _require(bool(records), "sampler records are empty")
        _require(batch_size > 0, "batch size must be positive")
        self.records = list(records)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.repeat_cap = int(repeat_cap)
        self.pass_index = 0
        self.task_sizes = Counter(record.task for record in records)
        self.allocations = capped_sqrt_task_allocations(
            self.task_sizes,
            draw_count=len(records),
            repeat_cap=self.repeat_cap,
        )

    def set_pass(self, pass_index: int) -> None:
        _require(pass_index >= 0, "pass index is negative")
        self.pass_index = int(pass_index)

    def batches_for_pass(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.pass_index)
        hierarchy: dict[str, dict[str, dict[str, list[int]]]] = {}
        for index, record in enumerate(self.records):
            hierarchy.setdefault(record.task, {}).setdefault(record.study, {}).setdefault(
                record.source_group, []
            ).append(index)
        batches: list[list[int]] = []
        for task in sorted(hierarchy):
            cycle = _StudyCycle(hierarchy[task], rng, self.repeat_cap)
            draws = [cycle.draw() for _ in range(self.allocations[task])]
            # Keep the study/source-group cycle interleaving inside each batch;
            # globally sorting by length here would collapse ranking batches
            # back toward a single source group.
            task_batches = [
                draws[start : start + self.batch_size]
                for start in range(0, len(draws), self.batch_size)
            ]
            for batch in task_batches:
                rng.shuffle(batch)
            batches.extend(task_batches)
        rng.shuffle(batches)
        counts = Counter(index for batch in batches for index in batch)
        _require(sum(counts.values()) == len(self.records), "pass draw count changed")
        _require(max(counts.values()) <= self.repeat_cap, "record repeat cap was exceeded")
        _require(
            all(len({self.records[index].task for index in batch}) == 1 for batch in batches),
            "a training batch mixes tasks",
        )
        return batches

    def __iter__(self):
        yield from self.batches_for_pass()

    def __len__(self) -> int:
        return sum(
            math.ceil(allocation / self.batch_size)
            for allocation in self.allocations.values()
        )


def build_exact_source_task_candidate_bundle_permutation(
    records: Sequence[XEditCriticRecordV3], *, seed: int
) -> tuple[dict[str, str], dict[str, Any]]:
    """Permute the complete candidate/edit/cache bundle inside exact strata."""

    groups: dict[tuple[str, str], list[XEditCriticRecordV3]] = {}
    for record in records:
        groups.setdefault((record.source, record.task), []).append(record)
    rng = random.Random(seed)
    overrides: dict[str, str] = {}
    eligible_tasks: set[str] = set()
    changed = 0
    for (_source, task), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda record: record.record_id)
        shift_changes = {
            shift: sum(
                recipient.candidate != donor.candidate
                for recipient, donor in zip(
                    ordered, ordered[shift:] + ordered[:shift]
                )
            )
            for shift in range(1, len(ordered))
        }
        maximum_change = max(shift_changes.values())
        if maximum_change == 0:
            continue
        best_shifts = [
            shift for shift, change_count in shift_changes.items()
            if change_count == maximum_change
        ]
        shift = rng.choice(best_shifts)
        donors = ordered[shift:] + ordered[:shift]
        for recipient, donor in zip(ordered, donors):
            overrides[recipient.record_id] = donor.record_id
            changed += recipient.candidate != donor.candidate
        eligible_tasks.add(task)
    _require(bool(overrides) and changed > 0, "candidate bundle permutation changes no candidate")
    return overrides, {
        "schema_version": "route_a_v3_route2_candidate_bundle_permutation.v3",
        "exact_source_task_strata": True,
        "complete_candidate_bundle_permuted": True,
        "recipient_count": len(overrides),
        "changed_candidate_sequence_count": changed,
        "eligible_task_count": len(eligible_tasks),
        "eligible_tasks": sorted(eligible_tasks),
    }


def different_source_group_pairwise_logistic_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    source_groups: Sequence[str],
    task_ids: Sequence[str],
) -> torch.Tensor | None:
    """Same-task ranking loss whose every pair crosses source groups."""

    _require(predictions.ndim == targets.ndim == 1, "ranking tensors must be vectors")
    _require(len(predictions) == len(source_groups) == len(task_ids), "ranking bundle is misaligned")
    _require(len(set(task_ids)) == 1, "ranking batch is not task homogeneous")
    count = len(predictions)
    if count < 2:
        return None
    upper = torch.triu(
        torch.ones((count, count), dtype=torch.bool, device=predictions.device),
        diagonal=1,
    )
    cross_group = torch.tensor(
        [left != right for left in source_groups for right in source_groups],
        dtype=torch.bool,
        device=predictions.device,
    ).reshape(count, count)
    target_delta = targets[:, None] - targets[None, :]
    eligible = upper & cross_group & target_delta.ne(0)
    if not eligible.any():
        return None
    prediction_delta = predictions[:, None] - predictions[None, :]
    return torch.nn.functional.softplus(
        -target_delta.sign()[eligible] * prediction_delta[eligible]
    ).mean()
