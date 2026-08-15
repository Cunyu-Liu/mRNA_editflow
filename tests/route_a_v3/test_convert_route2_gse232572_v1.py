from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/convert_route2_gse232572_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse232572_converter_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("convert_route2_gse232572_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(expected: int = 2) -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["input"]["expected_row_count"] = expected
    return value


def _row(record_id: str, position: int = 82, standard_error=None) -> dict:
    source = "A" * 165
    candidate = source[:position] + "C" + source[position + 1 :]
    return {
        "record_id": record_id,
        "study": {
            "accession": "GSE232572",
            "independent_study_group_id": "GSE232573",
            "publication_doi": "10.1038/s41467-024-46795-7",
            "study_id": "GSE232572",
        },
        "region": "3UTR",
        "source": {
            "design_family_id": "family1",
            "gene_id": "GENE1",
            "locus_id": "locus1",
            "sequence_id": "source-seq1",
            "source_id": "source1",
        },
        "candidate": {"candidate_id": f"candidate-{record_id}", "design_id": record_id, "sequence_id": f"seq-{record_id}"},
        "source_sequence": source,
        "candidate_sequence": candidate,
        "edit_set": [
            {
                "alt_base": "C",
                "coordinate_system": "ZERO_BASED_SOURCE",
                "distance_from_region_start": position,
                "edit_id": f"edit-{record_id}",
                "position": position,
                "ref_base": "A",
                "region": "3UTR",
            }
        ],
        "biological_source_group_id": "GSE232572|GENE1|source1",
        "gene_group_id": "GENE1",
        "context": {
            "cell_type": "HeLa",
            "condition": "MapUTR transfection assay",
            "context_id": "GSE232572|HeLa",
            "observable_context": "HeLa cells",
        },
        "assay": {"assay_id": "GSE232572_MAPUTR_HELA", "assay_type": "MPRA_MAPUTR_RNA_DNA", "protocol_version": "Fu_et_al_2024"},
        "endpoint": {
            "beneficial_direction": "HIGHER_IS_BETTER",
            "endpoint_id": "GSE232572|ln_activity_ratio_alt_over_ref",
            "endpoint_name": "ln_activity_ratio_alt_over_ref",
        },
        "delta": 0.2,
        "standard_error": standard_error,
        "eligibility": {
            "reject_reason_code": "PUBLIC_ROW_REDISTRIBUTION_RIGHTS_UNKNOWN",
            "reject_reason_detail": "private only",
            "status": "DEVELOPMENT_ONLY",
        },
        "data_role": "ORDINARY_DEVELOPMENT",
        "provenance": {"dataset_id": "GSE232572"},
        "replicate": {"replicate_count": 3},
        "license": {"redistribution_allowed": False},
        "exposure": {"label_exposed": True, "sequence_exposed": True},
        "paper_faithful_transform": {"transform_id": "published_lnfc"},
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_converter_reclassifies_legacy_development_as_evaluation_reserved(tmp_path: Path) -> None:
    module = _module()
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"
    _write_rows(input_path, [_row("r1"), _row("r2", position=90)])
    summary = module.execute(_config(), input_path, output_dir)
    assert summary["status"] == "CONVERTED_EVALUATION_RESERVED"
    assert summary["converted_record_count"] == 2
    assert summary["missing_standard_error_count"] == 2
    assert summary["training_eligible_record_count"] == 0
    assert summary["model_selection_eligible_record_count"] == 0
    assert summary["zero_shot_result_recorded"] is False

    records = [json.loads(line) for line in (output_dir / "canonical_records.private.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {record["pool_assignment"] for record in records} == {"EVALUATION"}
    assert {record["qualification_class"] for record in records} == {"EVALUATION_RESERVED"}
    assert all(not record["training_eligible"] and not record["model_selection_eligible"] for record in records)
    assert all(record["biological_standard_error"] is None for record in records)
    assert all(record["legacy_data_role"] == "ORDINARY_DEVELOPMENT" for record in records)


def test_missing_standard_error_stays_null_and_finite_positive_is_preserved(tmp_path: Path) -> None:
    module = _module()
    assert module.convert_row(_row("missing"), _config(expected=1))["biological_standard_error"] is None
    assert module.convert_row(_row("present", standard_error=0.1), _config(expected=1))["biological_standard_error"] == 0.1
    invalid = _row("invalid", standard_error=0.0)
    with pytest.raises(module.RejectRow, match="INVALID_STANDARD_ERROR"):
        module.convert_row(invalid, _config(expected=1))


def test_edit_set_must_match_the_source_candidate_substitution() -> None:
    module = _module()
    invalid = _row("bad")
    invalid["edit_set"][0]["position"] = 81
    with pytest.raises(module.RejectRow, match="EDIT_SET_SEQUENCE_MISMATCH"):
        module.convert_row(invalid, _config(expected=1))


def test_evaluation_role_drift_is_rejected(tmp_path: Path) -> None:
    module = _module()
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"
    invalid = _row("bad")
    invalid["data_role"] = "TRAIN"
    _write_rows(input_path, [_row("good"), invalid])
    summary = module.execute(_config(), input_path, output_dir)
    assert summary["status"] == "PARTIAL_EVALUATION_RESERVED"
    rejects = json.loads((output_dir / "reject_summary.json").read_text(encoding="utf-8"))
    assert rejects["route2_reject_counts"] == {"LEGACY_DATA_ROLE_MISMATCH": 1}


def test_existing_output_and_qualified_counts_are_preserved(tmp_path: Path) -> None:
    module = _module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    module.validate_config(config)
    assert config["study"]["pool_assignment"] == "EVALUATION"
    assert not any(config["credit_policy"]["qualified_credit_delta"].values())
    assert config["credit_policy"]["qualified_counts_after_conversion"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    _write_rows(input_path, [_row("r1")])
    with pytest.raises(module.ConversionError, match="already exists"):
        module.execute(_config(expected=1), input_path, output_dir)
    assert marker.read_text(encoding="utf-8") == "keep"
