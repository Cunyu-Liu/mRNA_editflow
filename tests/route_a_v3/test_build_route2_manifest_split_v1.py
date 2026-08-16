from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/build_route2_manifest_split_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_route2_manifest_split_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(module, name: str, source: str, candidate: str, gene: str = ""):
    return module.RecordMetadata(
        canonical_record_id=f"record:{name}",
        study_unit_id="STUDY",
        pool_assignment="DEVELOPMENT",
        source_group_key=f"STUDY::{name}",
        source_id=name,
        source_sequence=source,
        candidate_sequence=candidate,
        gene_tokens=(gene,) if gene else (),
        stratum=("STUDY", "3UTR", "ENDPOINT"),
    )


def test_component_builder_binds_gene_exact_role_overlap_and_near_duplicate_sources() -> None:
    module = _module()
    source_a = "A" * 100
    source_b = "C" * 100
    source_c = "G" * 100
    near_c = "T" + source_c[1:]
    bridge = "AC" * 50
    records = [
        _record(module, "A", source_a, "T" + source_a[1:], "GENE1"),
        _record(module, "B", source_b, "A" + source_b[1:], "GENE1"),
        _record(module, "C", source_c, bridge),
        _record(module, "D", bridge, "T" + bridge[1:]),
        _record(module, "E", near_c, "A" + near_c[1:]),
    ]
    components, stats = module._build_development_components(records, 0.95)
    assert components["STUDY::A"] == components["STUDY::B"]
    assert components["STUDY::C"] == components["STUDY::D"]
    assert components["STUDY::C"] == components["STUDY::E"]
    assert stats["connected_component_count"] == 2
    assert stats["near_duplicate_source_sequence_pair_count"] >= 1


def test_grouped_assignment_keeps_components_intact_and_populates_all_splits() -> None:
    module = _module()
    records = []
    component_by_group = {}
    for component_index in range(6):
        for member_index in range(component_index + 1):
            name = f"C{component_index}M{member_index}"
            records.append(_record(module, name, ("ACGT" * 25), ("TGCA" * 25)))
            component_by_group[f"STUDY::{name}"] = f"COMPONENT{component_index}"
    assignment = module._assign_components(
        records,
        component_by_group,
        {"TRAIN": 0.7, "VALIDATION": 0.15, "TEST": 0.15},
        20260816,
    )
    assert set(assignment.values()) == {"TRAIN", "VALIDATION", "TEST"}
    assert len(assignment) == 6


def test_evaluation_overlap_is_tagged_without_reading_outcomes() -> None:
    module = _module()
    development = [_record(module, "D", "A" * 100, "C" * 100)]
    exact = _record(module, "E1", "G" * 100, "C" * 100)
    near = _record(module, "E2", "T" + "A" * 99, "G" * 100)
    evaluation = [
        module.RecordMetadata(
            **{**exact.__dict__, "pool_assignment": "EVALUATION", "canonical_record_id": "eval:exact"}
        ),
        module.RecordMetadata(
            **{**near.__dict__, "pool_assignment": "EVALUATION", "canonical_record_id": "eval:near"}
        ),
    ]
    exposure, stats = module._evaluation_exposure(development, evaluation, 0.95)
    assert exposure["eval:exact"]["exact_sequence_seen_in_development"] is True
    assert exposure["eval:near"]["near_duplicate_source_seen_in_development"] is True
    assert stats["evaluation_record_count"] == 2


def test_loso_excludes_other_study_records_connected_to_holdout() -> None:
    module = _module()
    first = _record(module, "A", "A" * 100, "C" * 100)
    second = module.RecordMetadata(**{
        **_record(module, "B", "G" * 100, "T" * 100).__dict__,
        "study_unit_id": "OTHER",
        "source_group_key": "OTHER::B",
    })
    bridge = module.RecordMetadata(**{
        **_record(module, "C", "C" * 100, "G" * 100).__dict__,
        "study_unit_id": "OTHER",
        "source_group_key": "OTHER::C",
    })
    third = module.RecordMetadata(**{
        **_record(module, "D", "T" * 100, "A" * 100).__dict__,
        "study_unit_id": "THIRD",
        "source_group_key": "THIRD::D",
    })
    records = [first, second, bridge, third]
    components = {
        "STUDY::A": "CONNECTED", "OTHER::C": "CONNECTED",
        "OTHER::B": "INDEPENDENT", "THIRD::D": "THIRD_COMPONENT",
    }
    folds = {row["holdout_study_unit_id"]: row for row in module._loso_definitions(records, components)}
    assert folds["STUDY"]["training_record_count"] == 2
    assert folds["STUDY"]["excluded_connected_other_study_record_count"] == 1


def test_zero_count_study_does_not_require_placeholder_canonical_file(tmp_path: Path) -> None:
    module = _module()
    records, stats = module._load_study({
        "study_unit_id": "ZERO_RECORD_STUDY",
        "pool_assignment": "EVALUATION",
        "canonical_records_path": str(tmp_path / "absent.jsonl"),
        "expected_canonical_record_count": 0,
        "gene_group_fields": [],
    })
    assert records == []
    assert stats["canonical_record_count"] == 0
    assert stats["source_group_count"] == 0

    with pytest.raises(module.ManifestError, match="canonical input absent"):
        module._load_study({
            "study_unit_id": "MISSING_NONZERO_STUDY",
            "pool_assignment": "EVALUATION",
            "canonical_records_path": str(tmp_path / "also-absent.jsonl"),
            "expected_canonical_record_count": 1,
            "gene_group_fields": [],
        })


def test_development_only_materialization_does_not_create_evaluation_manifest(tmp_path: Path) -> None:
    module = _module()
    studies = []
    sources = ("AAAA", "AAAC", "AACA", "ACAA", "CAAA", "AACC", "ACAC", "CACA")
    candidates = ("TTTT", "TTTG", "TTGT", "TGTT", "GTTT", "TTGG", "TGTG", "GTGT")
    for index, (source, candidate) in enumerate(zip(sources, candidates)):
        study_id = f"S{index}"
        canonical_path = tmp_path / f"{study_id}.jsonl"
        canonical_path.write_text(json.dumps({
            "canonical_record_id": f"record:{study_id}",
            "study_unit_id": study_id,
            "pool_assignment": "DEVELOPMENT",
            "source_id": f"source:{study_id}",
            "source_sequence": source,
            "candidate_sequence": candidate,
            "region": "3UTR",
            "endpoint_id": "ENDPOINT",
        }) + "\n")
        studies.append({
            "study_unit_id": study_id,
            "pool_assignment": "DEVELOPMENT",
            "canonical_records_path": str(canonical_path),
            "expected_canonical_record_count": 1,
            "gene_group_fields": [],
        })
    config = {
        "schema_version": "route_a_v3_route2_manifest_split.v1",
        "materialization_scope": "DEVELOPMENT_ONLY_PRE_EVALUATION_CLOSE",
        "studies": studies,
        "split_policy": {
            "unit": "CONNECTED_SOURCE_COMPONENT",
            "group_by_source_id": True,
            "group_by_gene_within_study": True,
            "group_by_exact_source_or_candidate_sequence": True,
            "group_by_near_duplicate_source_sequence": True,
            "near_duplicate_same_length_identity": 0.95,
            "ratios": {"TRAIN": 0.7, "VALIDATION": 0.15, "TEST": 0.15},
            "seed": 20260816,
        },
        "evaluation_policy": {
            "training_eligible": False,
            "model_selection_eligible": False,
            "outcome_metrics_computed": False,
        },
        "output": {
            "directory": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/unused-test-path",
            "development_manifest_filename": "development_manifest.jsonl",
            "evaluation_manifest_filename": "evaluation_manifest.jsonl",
            "loso_fold_definitions_filename": "loso_folds.jsonl",
            "summary_filename": "manifest_summary.json",
            "overwrite_allowed": False,
        },
    }
    module.validate_config(config)
    output_dir = tmp_path / "development_only"
    summary = module.execute(config, output_dir)
    assert summary["status"] == "ROUTE2_DEVELOPMENT_ONLY_MANIFEST_AND_GROUPED_SPLIT_MATERIALIZED"
    assert summary["evaluation_manifest_materialized"] is False
    assert summary["development_record_count"] == 8
    assert summary["evaluation_record_count"] == 0
    assert (output_dir / "development_manifest.jsonl").is_file()
    assert (output_dir / "loso_folds.jsonl").is_file()
    assert not (output_dir / "evaluation_manifest.jsonl").exists()
