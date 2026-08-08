from __future__ import annotations

import copy
import hashlib

import pytest

from scripts.r3x.development_estimands import build_development_estimand_slices
from scripts.r3x.protocol import ProtocolHoldError


def _id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _registry() -> dict:
    common = {
        "estimand_id": "shared_delta",
        "raw_source_field": "source_value",
        "raw_candidate_field": "candidate_value",
        "raw_unit": "normalized_unit",
        "direction": "HIGHER_IS_MORE_STABLE",
        "transform": "IDENTITY_DIFFERENCE",
        "valid_domain": "FINITE",
        "role": "PRIMARY",
        "replicate_policy": "SINGLE_OBSERVATION_ONLY",
        "source_evidence_sha256": "e" * 64,
    }
    return {
        "schema_version": "r3_endpoint_estimand_registry_v1",
        "status": "FROZEN",
        "entries": [
            {**common, "study": "study_a", "endpoint": "endpoint_shared"},
            {**common, "study": "study_b", "endpoint": "endpoint_shared"},
        ],
    }


def _row(study: str, partition: str, index: int) -> dict:
    source = "AAAA"
    candidate = "CAAA"
    return {
        "r3_input_row_sha256": _id(f"{study}-{partition}-{index}"),
        "record_id": f"row_{study}_{partition}_{index}",
        "partition": partition,
        "study": study,
        "endpoint": "endpoint_shared",
        "estimand_id": "shared_delta",
        "publication_context_id": f"context_{study}",
        "normalized_source_sequence": source,
        "normalized_candidate_sequence": candidate,
        "edit_list": [{"op": "SUB", "pos": 0, "ref": "A", "alt": "C"}],
        "edit_count": 1,
        "source_component_id": f"component_{study}_{index}",
        "source_group_id_v2": f"pool_{study}_{index}",
        "source_value": 1.0,
        "candidate_value": 1.25,
        "delta": 0.25,
    }


def _rows() -> list[dict]:
    return [
        _row(study, partition, index)
        for study in ("study_a", "study_b")
        for partition in ("train", "development")
        for index in range(2)
    ]


def _build(rows=None, registry=None):
    return build_development_estimand_slices(
        _rows() if rows is None else rows,
        _registry() if registry is None else registry,
        estimand_registry_sha256="a" * 64,
    )


def test_shared_estimand_builds_without_any_test_partition_surface() -> None:
    slices = _build()
    assert len(slices) == 1
    item = slices[0]
    assert item.estimand_id == "shared_delta"
    assert item.studies == ("study_a", "study_b")
    assert len(item.train) == len(item.development) == 4
    assert set(item.partition_row_set_sha256) == {"train", "development"}
    assert "internal_test" not in item.__dataclass_fields__
    with pytest.raises(ProtocolHoldError) as exc:
        item.rows_for("internal_test")
    assert exc.value.code == "HOLD_TEST_PARTITION_NOT_DISCLOSED"


@pytest.mark.parametrize("partition", ["internal_test", None, "renamed_test"])
def test_forbidden_or_unbound_partition_holds(partition) -> None:
    rows = _rows()
    rows[0]["partition"] = partition
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    expected = (
        "HOLD_TEST_PARTITION_PREMATURELY_DISCLOSED"
        if partition == "internal_test"
        else "HOLD_DEVELOPMENT_PARTITION_AUTHORITY"
    )
    assert exc.value.code == expected


def test_duplicate_or_non_sha_row_identity_holds() -> None:
    rows = _rows()
    rows[1]["r3_input_row_sha256"] = rows[0]["r3_input_row_sha256"]
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_ROW_IDENTITY"
    rows = _rows()
    rows[0]["r3_input_row_sha256"] = "not-a-sha"
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_ROW_IDENTITY"


def test_unregistered_or_mismatched_estimand_holds() -> None:
    rows = _rows()
    rows[0]["endpoint"] = "unknown_endpoint"
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_ESTIMAND_BINDING"
    rows = _rows()
    rows[0]["estimand_id"] = "post_hoc_estimand"
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_ESTIMAND_BINDING"


def test_every_registered_study_endpoint_requires_both_partitions() -> None:
    rows = [
        row for row in _rows()
        if not (row["study"] == "study_b" and row["partition"] == "development")
    ]
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_ESTIMAND_COVERAGE"
    assert exc.value.details["incomplete"] == {
        "study_b|endpoint_shared": ["development"]
    }


def test_shared_estimand_semantics_must_match_across_studies() -> None:
    registry = copy.deepcopy(_registry())
    registry["entries"][1]["transform"] = "NATURAL_LOG_RATIO"
    with pytest.raises(ProtocolHoldError) as exc:
        _build(registry=registry)
    assert exc.value.code == "HOLD_DEVELOPMENT_ESTIMAND_AUTHORITY"
    assert exc.value.details["field"] == "transform"


def test_protected_record_and_nonfinite_target_hold() -> None:
    rows = _rows()
    rows[0]["metadata"] = {"role": "sealed_final"}
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_PROTECTED_INPUT"
    rows = _rows()
    rows[0]["delta"] = float("nan")
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_TARGET"
