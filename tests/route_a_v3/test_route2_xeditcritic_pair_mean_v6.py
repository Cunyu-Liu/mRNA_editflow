"""Focused unit tests for Critic V6 W1-a pair-mean / W1-c rank transform / W1-e eval extension.

All features default OFF; the tests prove both the default identity (V5
bit-identical) and the enabled semantics using tiny CPU fixtures.
"""

from __future__ import annotations

import math

import pytest

from core.route2_xeditcritic_training_data_v3 import XEditCriticRecordV3
from core.route2_xeditcritic_pair_mean_v6 import (
    MPRAU_TASK_ID,
    apply_pair_mean_targets_v6,
    apply_rank_gaussian_targets_v6,
    extended_validation_metrics_v6,
    pair_mean_target_map_v6,
    pair_key_v6,
    per_task_rank_gaussian_v6,
)


def _record(
    record_id: str,
    target: float,
    *,
    task: str = MPRAU_TASK_ID,
    context: str = "GM12878",
    source: str = "ACGU",
    candidate: str = "ACGU",
) -> XEditCriticRecordV3:
    return XEditCriticRecordV3(
        record_id=record_id,
        split="TRAIN",
        source=source,
        candidate=candidate,
        edits=(),
        target=target,
        task=task,
        study="ENCSR854RUF",
        source_group=f"{task}::SG::{context}",
        assay="MPRA",
        context=context,
        region=1,
        quantity="q",
        measurement="m",
        numerator=None,
        denominator=None,
    )


def test_pair_mean_map_default_matches_v5_identity():
    # Default pair_tasks enable only the MPRAU task, so non-MPRAU records keep
    # their own target (identity), which is exactly the V5 label.
    records = [
        _record("r1", 1.5, task="OTHER::region=0", source="ACGU", candidate="ACGU"),
        _record("r2", 2.5, task="OTHER::region=0", source="ACGU", candidate="ACGU"),
        _record("r3", 3.0, task="OTHER::region=0", source="ACGU", candidate="ACGU"),
    ]
    mapping = pair_mean_target_map_v6(records)  # pair_tasks=None -> MPRAU only
    assert mapping["r1"] == 1.5
    assert mapping["r2"] == 2.5
    assert mapping["r3"] == 3.0
    assert set(mapping) == {"r1", "r2", "r3"}


def test_pair_mean_collapses_six_cells_to_one_mean():
    variants = ["ACGU", "ACGA"]
    cells = ["GM12878", "HEK293FT", "HEPG2", "HMEC", "K562", "SKNSH"]
    records = []
    for variant_index, source in enumerate(variants):
        for cell_index, cell in enumerate(cells):
            records.append(
                _record(
                    f"v{variant_index}_c{cell_index}",
                    float(variant_index * 10 + cell_index),
                    context=cell,
                    source=source,
                    candidate=source,
                )
            )
    mapping = pair_mean_target_map_v6(records)
    # variant A mean = (0+1+2+3+4+5)/6 = 2.5 ; variant B = (10..15)/6 = 12.5
    assert mapping["v0_c0"] == pytest.approx(2.5)
    assert mapping["v0_c5"] == pytest.approx(2.5)
    assert mapping["v1_c0"] == pytest.approx(12.5)
    assert len({mapping[key] for key in mapping}) == 2


def test_pair_mean_rejects_incomplete_pair():
    records = [
        _record("r1", 1.0, context="GM12878"),
        _record("r2", 2.0, context="HEK293FT"),
        _record("r3", 3.0, context="HEPG2", source="GGGG", candidate="GGGU"),
    ]
    with pytest.raises(Exception):
        pair_mean_target_map_v6(records)


def test_pair_mean_apply_preserves_record_count_and_order():
    records = [
        _record(f"v0_c{i}", float(i), context=c)
        for i, c in enumerate(["GM12878", "HEK293FT", "HEPG2", "HMEC", "K562", "SKNSH"])
    ]
    relabelled, _ = apply_pair_mean_targets_v6(records)
    assert len(relabelled) == len(records)
    assert [r.record_id for r in relabelled] == [r.record_id for r in records]
    # every relabelled target equals the pair mean 2.5
    assert all(r.target == pytest.approx(2.5) for r in relabelled)


def test_pair_key_is_context_free():
    a = _record("a", 1.0, context="GM12878")
    b = _record("b", 2.0, context="HEK293FT")
    c = _record("c", 2.0, context="HEK293FT", source="GGGG", candidate="GGGU")
    assert pair_key_v6(a) == pair_key_v6(b)
    assert pair_key_v6(a) != pair_key_v6(c)


def test_per_task_rank_gaussian_is_monotone_and_invariant():
    values = [0.0, 0.1, 0.9, 5.0, -1.0]
    zs = per_task_rank_gaussian_v6(values)
    # exactly one Gaussian value per input; ties share a value
    assert len(zs) == len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    z_by_order = [zs[i] for i in order]
    assert z_by_order == sorted(z_by_order)
    assert all(math.isfinite(z) for z in zs)


def test_rank_gaussian_apply_maps_each_task_only():
    records = [_record(f"r{i}", float(v), task="T::region=0") for i, v in enumerate([1.0, 2.0, 9.0])]
    transformed, meta = apply_rank_gaussian_targets_v6(records, rank_tasks={"T::region=0"})
    assert meta["task_count"] == 1
    assert len(transformed) == 3
    raw = [r.target for r in records]
    zs = [r.target for r in transformed]
    assert zs == per_task_rank_gaussian_v6(raw)


def test_extended_validation_metrics_requires_aligned_bundles():
    with pytest.raises(Exception):
        extended_validation_metrics_v6([1.0, 2.0], [1.0], ["t"], ["g"], [("t", "s", "a", "c")])


def test_extended_validation_metrics_pair_mean_rho():
    targets = []
    predictions = []
    tasks = []
    groups = []
    keys = []
    for variant in range(10):
        pair_key = (MPRAU_TASK_ID, "ENCSR854RUF", f"src{variant}", f"cand{variant}")
        for cell_index in range(6):
            targets.append(float(variant * 0.1 + cell_index * 0.001))
            # nearly monotone prediction: perfect-ish ranking by variant means
            predictions.append(float(variant * 0.09 + cell_index * 0.0001))
            tasks.append(MPRAU_TASK_ID)
            groups.append(f"ENCSR854RUF::SG{variant}::CELL{cell_index}::{MPRAU_TASK_ID}")
            keys.append(pair_key)
    metrics = extended_validation_metrics_v6(
        targets,
        predictions,
        tasks,
        groups,
        keys,
        ceiling_by_task={MPRAU_TASK_ID: 0.683},
    )
    assert metrics["pair_mean_spearman_pooled_legacy"] is not None
    assert metrics["pair_mean_spearman_pooled_legacy"] > 0.8
    assert metrics["pair_mean_ceiling_ratio_pooled_legacy"] is not None
    assert metrics["pair_mean_ceiling_ratio_pooled_legacy"] > 1.0
    assert MPRAU_TASK_ID in metrics["tier_b_tasks"]
    assert metrics["schema_version"] == "route_a_v3_route2_xeditcritic_v6_extended_metrics.v2"
    # Task 3.1: the un-suffixed pooled key is gone (stale readers fail
    # loudly); single-task fixture pools to the only task's per-task value.
    assert "pair_mean_spearman" not in metrics
    assert metrics["pair_mean_spearman_by_task"][MPRAU_TASK_ID] == pytest.approx(
        metrics["pair_mean_spearman_pooled_legacy"]
    )
    assert metrics["pair_mean_pair_count_by_task"][MPRAU_TASK_ID] == 10
    assert metrics["pair_mean_pair_count_pooled_legacy"] == 10


def test_extended_validation_metrics_pair_mean_by_task_two_tasks():
    # Task 3.1 regression: two tasks with opposite pair-mean orderings must be
    # reported separately; the pooled legacy value mixes both task pools and
    # matches neither per-task column (the 2026-09-03 misreport mechanism).
    other_task = "OTHER_TASK::region=0"
    targets = []
    predictions = []
    tasks = []
    groups = []
    keys = []
    for variant in range(10):
        mprau_key = (MPRAU_TASK_ID, "ENCSR854RUF", f"src{variant}", f"cand{variant}")
        for cell_index in range(6):
            targets.append(float(variant * 0.1 + cell_index * 0.001))
            predictions.append(float(variant * 0.09 + cell_index * 0.0001))
            tasks.append(MPRAU_TASK_ID)
            groups.append(f"ENCSR854RUF::SG{variant}::CELL{cell_index}::{MPRAU_TASK_ID}")
            keys.append(mprau_key)
        other_key = (other_task, "STUDY2", f"src{variant}", f"cand{variant}")
        for rep in range(3):
            # anti-monotone predictions: per-task pair-mean rho = -1
            targets.append(1.0 + variant * 0.1 + rep * 0.001)
            predictions.append(1.8 - variant * 0.09 - rep * 0.0001)
            tasks.append(other_task)
            groups.append(f"STUDY2::SG{variant}::REP{rep}::{other_task}")
            keys.append(other_key)
    metrics = extended_validation_metrics_v6(
        targets,
        predictions,
        tasks,
        groups,
        keys,
        ceiling_by_task={MPRAU_TASK_ID: 0.683, other_task: 0.21},
    )
    by_task = metrics["pair_mean_spearman_by_task"]
    assert set(by_task) == {MPRAU_TASK_ID, other_task}
    assert by_task[MPRAU_TASK_ID] > 0.99
    assert by_task[other_task] < -0.99
    pooled = metrics["pair_mean_spearman_pooled_legacy"]
    assert pooled is not None
    assert pooled != by_task[MPRAU_TASK_ID]
    assert pooled != by_task[other_task]
    assert metrics["pair_mean_pair_count_by_task"] == {
        MPRAU_TASK_ID: 10,
        other_task: 10,
    }
    assert metrics["pair_mean_pair_count_pooled_legacy"] == 20
    ceilings = metrics["pair_mean_ceiling_by_task"]
    assert ceilings[MPRAU_TASK_ID] == pytest.approx(0.683)
    assert ceilings[other_task] == pytest.approx(0.21)
    ratios = metrics["pair_mean_ceiling_ratio_by_task"]
    assert ratios[MPRAU_TASK_ID] == pytest.approx(by_task[MPRAU_TASK_ID] / 0.683)
    assert ratios[other_task] == pytest.approx(by_task[other_task] / 0.21)
