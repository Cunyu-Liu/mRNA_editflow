from __future__ import annotations

from scripts.route_a_v3.smoke_route2_xeditcritic_v402_recovery import (
    sampler_records_without_targets_v402,
)


def _row(split: str = "TRAIN") -> dict[str, object]:
    return {
        "canonical_record_id": "record-1",
        "split": split,
        "task_id": "task",
        "study_unit_id": "study",
        "source_group_id": "source-group",
        "assay_id": "assay",
        "biological_context_id": "context",
        "region_id": 1,
        "endpoint_descriptor": {
            "quantity_family": "quantity",
            "measurement_form": "measurement",
            "numerator_family": None,
            "denominator_family": None,
        },
    }


def test_sampler_smoke_metadata_builder_never_requires_a_target_key() -> None:
    records = sampler_records_without_targets_v402([_row()])
    assert len(records) == 1
    assert records[0].record_id == "record-1"
    assert records[0].target == 0.0


def test_sampler_smoke_metadata_builder_uses_train_only() -> None:
    records = sampler_records_without_targets_v402([_row(), _row("VALIDATION")])
    assert len(records) == 1
