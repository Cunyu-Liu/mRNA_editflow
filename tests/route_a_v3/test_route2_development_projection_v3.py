from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.route2_development_projection_v3 import (
    DevelopmentProjectionError,
    build_development_projection,
    extract_canonical_record_id,
    load_projection_rows,
)


ROOT = Path(__file__).resolve().parents[2]
DESCRIPTORS = ROOT / "configs/route_a_v3_route2_endpoint_descriptors_v1.json"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest_row(record_id: str, split: str) -> dict:
    return {
        "canonical_record_id": record_id,
        "pool_assignment": "DEVELOPMENT",
        "split": split,
        "study_unit_id": "GSE149487",
        "connected_source_component_id": f"component::{record_id}",
    }


def _canonical_row(record_id: str, endpoint: str, target: float) -> dict:
    return {
        "canonical_record_id": record_id,
        "pool_assignment": "DEVELOPMENT",
        "study_unit_id": "GSE149487",
        "source_id": "source-1",
        "source_sequence": "AAAA",
        "candidate_sequence": "ACAA",
        "region": "5UTR",
        "assay_id": "PLUMAGE_BARCODE_MPRA",
        "biological_context_id": "293T",
        "endpoint_id": endpoint,
        "direction_normalized_delta": target,
    }


def test_record_id_extraction_does_not_require_full_json_decode() -> None:
    protected = '{"canonical_record_id":"test", "direction_normalized_delta":NOT_JSON}\n'
    assert extract_canonical_record_id(protected) == "test"
    with pytest.raises(json.JSONDecodeError):
        json.loads(protected)


def test_projection_decodes_train_validation_but_not_test(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "canonical.jsonl"
    output = tmp_path / "projection"
    _write_jsonl(
        manifest,
        [
            _manifest_row("train", "TRAIN"),
            _manifest_row("validation", "VALIDATION"),
            _manifest_row("test", "TEST"),
        ],
    )
    train = _canonical_row("train", "te_log2_polysome_over_totalrna", 1.0)
    validation = _canonical_row(
        "validation", "transcript_log2_totalrna_over_dna", 2.0
    )
    protected = '{"canonical_record_id":"test", "direction_normalized_delta":NOT_JSON}\n'
    canonical.write_text(
        json.dumps(train) + "\n" + json.dumps(validation) + "\n" + protected,
        encoding="utf-8",
    )
    summary = build_development_projection(
        manifest_path=manifest,
        canonical_paths=[canonical],
        endpoint_descriptor_path=DESCRIPTORS,
        output_directory=output,
    )
    assert summary["canonical_full_decode_counts"] == {
        "TRAIN": 1,
        "VALIDATION": 1,
        "TEST": 0,
    }
    assert summary["development_test_outcomes_accessed"] is False
    rows = load_projection_rows(
        [output / "train.jsonl", output / "validation.jsonl"]
    )
    assert {row["canonical_record_id"] for row in rows} == {
        "train",
        "validation",
    }
    descriptors = {row["endpoint_id"]: row["endpoint_descriptor"] for row in rows}
    assert descriptors["te_log2_polysome_over_totalrna"]["quantity_family"] == "TRANSLATION_EFFICIENCY"
    assert descriptors["transcript_log2_totalrna_over_dna"]["quantity_family"] == "RNA_ABUNDANCE"
    assert all("outcome" not in key.lower() for value in descriptors.values() for key in value)


def test_projection_refuses_test_or_nonmodeling_split(tmp_path: Path) -> None:
    with pytest.raises(DevelopmentProjectionError, match="one-shot adjudicator"):
        build_development_projection(
            manifest_path=tmp_path / "unused-manifest",
            canonical_paths=[],
            endpoint_descriptor_path=DESCRIPTORS,
            output_directory=tmp_path / "unused-output",
            included_splits=("TEST",),
        )


def test_projection_loader_refuses_protected_split(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "route_a_v3_route2_development_projection.v3",
                "canonical_record_id": "test",
                "split": "TEST",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DevelopmentProjectionError, match="TRAIN/VALIDATION"):
        load_projection_rows([path], allowed_splits=("TEST",))
