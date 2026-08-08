from __future__ import annotations

import copy

import pytest

from scripts.r3x.estimands import (
    PREDICTION_SHARD_SCHEMA,
    build_primary_estimand_slices,
    merge_estimand_seed_runs,
    merge_prediction_shards,
    prediction_shard_from_seed_run,
    row_order_sha256,
)
from scripts.r3x.protocol import canonical_sha256
from scripts.r3x.report import ReportProtocolError


REGISTRY_SHA = "a" * 64


def _entry(estimand_id: str, study: str, endpoint: str) -> dict:
    return {
        "study": study,
        "endpoint": endpoint,
        "estimand_id": estimand_id,
        "raw_source_field": "source_value",
        "raw_candidate_field": "candidate_value",
        "raw_unit": "fixture_unit",
        "direction": "HIGHER_IS_MORE_STABLE",
        "transform": "IDENTITY_DIFFERENCE",
        "valid_domain": "FINITE",
        "role": "PRIMARY",
        "replicate_policy": "SINGLE_OBSERVATION_ONLY",
        "source_evidence_sha256": "e" * 64,
    }


def _registry() -> dict:
    return {
        "schema_version": "r3_endpoint_estimand_registry_v1",
        "status": "FROZEN",
        "entries": [
            _entry("estimand_a", "study_a", "endpoint_a"),
            _entry("estimand_b", "study_b", "endpoint_b"),
        ],
    }


def _rows() -> list[dict]:
    rows = []
    for estimand_id, study, endpoint, prefix in (
        ("estimand_a", "study_a", "endpoint_a", "a"),
        ("estimand_b", "study_b", "endpoint_b", "b"),
    ):
        entry = next(item for item in _registry()["entries"] if item["estimand_id"] == estimand_id)
        for number, partition in enumerate(("train", "development", "internal_test"), start=1):
            rows.append({
                "r3_input_row_sha256": prefix * 63 + str(number),
                "study": study,
                "endpoint": endpoint,
                "endpoint_id": endpoint,
                "estimand_id": estimand_id,
                "partition": partition,
                "estimand_status": "FROZEN_REGISTRY_APPLIED",
                "estimand_binding": {
                    "registry_sha256": REGISTRY_SHA,
                    **{key: value for key, value in entry.items() if key != "estimand_id"},
                },
            })
    return rows


def _slices(rows: list[dict] | None = None):
    return build_primary_estimand_slices(
        _rows() if rows is None else rows,
        _registry(),
        estimand_registry_sha256=REGISTRY_SHA,
    )


def _seed_run(estimand_slice, *, seed: int = 11, offset: float = 0.0) -> dict:
    row_ids = estimand_slice.row_ids_for("internal_test")
    return {
        "status": "RUN",
        "estimand_id": estimand_slice.estimand_id,
        "registry_sha256": estimand_slice.registry_sha256,
        "registry_entry_set_sha256": estimand_slice.registry_entry_set_sha256,
        "architecture_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "information_set_fingerprint": "f" * 64,
        "parameter_count": 10 if estimand_slice.estimand_id == "estimand_a" else 20,
        "seed": seed,
        "test_predictions": {
            "row_ids": list(row_ids),
            "row_set_sha256": row_order_sha256(row_ids),
            "mean": [offset + index + 0.25 for index in range(len(row_ids))],
            "logvar": [-1.0 for _ in row_ids],
        },
    }


def test_primary_estimand_slices_bind_registry_and_are_input_order_invariant() -> None:
    rows = _rows()
    forward = _slices(rows)
    reverse = _slices(list(reversed(rows)))
    assert [item.estimand_id for item in forward] == ["estimand_a", "estimand_b"]
    assert [item.partition_row_order_sha256 for item in forward] == [
        item.partition_row_order_sha256 for item in reverse
    ]
    assert [item.partition_binding_sha256 for item in forward] == [
        item.partition_binding_sha256 for item in reverse
    ]
    assert all(len(item.train) == len(item.development) == len(item.internal_test) == 1 for item in forward)
    expected_entry = _registry()["entries"][0]
    assert forward[0].registry_entry_set_sha256 == canonical_sha256([expected_entry])
    assert forward[0].row_ids_for("internal_test") == ("a" * 63 + "3",)


def test_primary_estimand_slices_fail_on_empty_partition_or_absent_registry_estimand() -> None:
    rows = [row for row in _rows() if not (
        row["estimand_id"] == "estimand_b" and row["partition"] == "development"
    )]
    with pytest.raises(ReportProtocolError, match="empty frozen partitions"):
        _slices(rows)

    only_a = [row for row in _rows() if row["estimand_id"] == "estimand_a"]
    with pytest.raises(ReportProtocolError, match="absent from the ready dataset"):
        _slices(only_a)


def test_primary_estimand_slices_fail_on_semantic_registry_drift() -> None:
    rows = _rows()
    rows[0]["estimand_binding"]["direction"] = "LOWER_IS_MORE_STABLE"
    with pytest.raises(ReportProtocolError, match="registry field direction"):
        _slices(rows)

    rows = _rows()
    rows[0]["estimand_id"] = "estimand_b"
    with pytest.raises(ReportProtocolError, match="semantically bound"):
        _slices(rows)


def test_prediction_shards_and_seed_runs_merge_in_global_canonical_order() -> None:
    slices = _slices()
    run_a = _seed_run(slices[0], offset=10.0)
    run_b = _seed_run(slices[1], offset=20.0)
    shard_a = prediction_shard_from_seed_run(slices[0], run_a)
    shard_b = prediction_shard_from_seed_run(slices[1], run_b)
    assert shard_a["schema_version"] == PREDICTION_SHARD_SCHEMA

    merged_predictions = merge_prediction_shards(slices, [shard_b, shard_a])
    assert merged_predictions["row_ids"] == ["a" * 63 + "3", "b" * 63 + "3"]
    assert merged_predictions["mean"] == [10.25, 20.25]
    assert merged_predictions["parameter_count"] == 30
    assert merged_predictions["parameter_count_scope"] == "SUM_OF_INDEPENDENT_ESTIMAND_MODELS"

    merged_run = merge_estimand_seed_runs(slices, [run_b, run_a])
    assert merged_run["seed"] == 11
    assert merged_run["test_predictions"]["row_ids"] == merged_predictions["row_ids"]
    assert merged_run["test_predictions"]["row_set_sha256"] == row_order_sha256(
        merged_predictions["row_ids"]
    )
    assert set(merged_run["estimand_runs"]) == {"estimand_a", "estimand_b"}


def test_seed_run_validation_rejects_row_hash_seed_and_registry_tampering() -> None:
    slices = _slices()
    run_a = _seed_run(slices[0])
    run_b = _seed_run(slices[1])

    tampered = copy.deepcopy(run_a)
    tampered["test_predictions"]["row_set_sha256"] = "0" * 64
    with pytest.raises(ReportProtocolError, match="canonical row-order hash"):
        prediction_shard_from_seed_run(slices[0], tampered)

    wrong_registry = copy.deepcopy(run_a)
    wrong_registry["registry_entry_set_sha256"] = "0" * 64
    with pytest.raises(ReportProtocolError, match="wrong registry entry set"):
        prediction_shard_from_seed_run(slices[0], wrong_registry)

    different_seed = copy.deepcopy(run_b)
    different_seed["seed"] = 23
    with pytest.raises(ReportProtocolError, match="different optimization seeds"):
        merge_estimand_seed_runs(slices, [run_a, different_seed])


def test_prediction_merge_requires_exactly_one_shard_per_estimand() -> None:
    slices = _slices()
    shard = prediction_shard_from_seed_run(slices[0], _seed_run(slices[0]))
    with pytest.raises(ReportProtocolError, match="exactly one shard"):
        merge_prediction_shards(slices, [shard])


def test_same_estimand_can_span_multiple_studies_when_semantics_match() -> None:
    registry = _registry()
    registry["entries"][1]["estimand_id"] = "estimand_a"
    rows = _rows()
    for row in rows:
        if row["study"] == "study_b":
            row["estimand_id"] = "estimand_a"
    slices = build_primary_estimand_slices(
        rows, registry, estimand_registry_sha256=REGISTRY_SHA,
    )
    assert len(slices) == 1
    assert slices[0].estimand_id == "estimand_a"
    assert slices[0].studies == ("study_a", "study_b")
    assert slices[0].endpoints == ("endpoint_a", "endpoint_b")


def test_same_estimand_rejects_inconsistent_scale_or_direction() -> None:
    registry = _registry()
    registry["entries"][1]["estimand_id"] = "estimand_a"
    registry["entries"][1]["raw_unit"] = "different_unit"
    rows = _rows()
    for row in rows:
        if row["study"] == "study_b":
            row["estimand_id"] = "estimand_a"
            row["estimand_binding"]["raw_unit"] = "different_unit"
    with pytest.raises(ReportProtocolError, match="inconsistent scientific semantics"):
        build_primary_estimand_slices(
            rows, registry, estimand_registry_sha256=REGISTRY_SHA,
        )
