from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from data.utr_benchmark_v2.records import REQUIRED_FIELDS
from data.utr_benchmark_v2.track_loader import CandidateStoreLabelError
from data.utr_benchmark_v2.track_loader import TrackContractError
from data.utr_benchmark_v2.track_loader import assert_candidate_store_label_free
from data.utr_benchmark_v2.track_loader import validate_generation_task
from scripts.data.build_b0_splits import validate_canonical_records_schema
from scripts.data.build_b0_splits import load_structural_jsonl
from tests.test_utr_record_v2 import _absolute_record
from tests.test_utr_record_v2 import _intervention_record


ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def _schema_ready_intervention() -> dict:
    record = _intervention_record()
    record.update(
        {
            "evidence_grade": "E2",
            "candidate_id": "candidate-1",
            "intermediate_sequences": [],
            "trajectory_provenance": {
                "status": "canonical_minimum_edit_alignment_between_endpoints",
                "observed_biological_path": False,
            },
            "scaffold_group": "scaffold-1",
            "barcode_batch": "NOT_APPLICABLE:GSE149487:no_barcode",
            "library_batch": "library-1",
            "library_design": "paired_reporter_library",
            "raw_source_sequence": "ACGU",
            "raw_candidate_sequence": "AGGU",
            "canonicalization_provenance": {"normalization": "identity"},
            "edit_script_ambiguity": {
                "equivalent_minimal_script_count": 1,
                "path_ambiguous": False,
            },
        }
    )
    return record


def _schema_ready_absolute() -> dict:
    record = _absolute_record()
    record.update(
        {
            "evidence_grade": "E2",
            "candidate_id": "candidate-absolute",
            "intermediate_sequences": [],
            "trajectory_provenance": {
                "status": "not_applicable_absolute_property_record",
                "observed_biological_path": False,
            },
            "scaffold_group": "scaffold-absolute",
            "barcode_batch": "NOT_APPLICABLE:GSE114002:no_barcode",
            "library_batch": "library-absolute",
            "library_design": "absolute_library",
            "raw_source_sequence": None,
            "raw_candidate_sequence": "ACGU",
            "canonicalization_provenance": {"normalization": "identity"},
            "edit_script_ambiguity": {
                "equivalent_minimal_script_count": 0,
                "path_ambiguous": False,
            },
        }
    )
    return record


def _edit_script_payload() -> dict:
    return {
        "schema_version": "edit_script.v2",
        "script_id": "script-1",
        "source_sequence": "ACGU",
        "candidate_sequence": "AGGU",
        "coordinate_system": "zero_based_current_state_before_operation",
        "path_status": "UNIQUE_MINIMAL_PATH",
        "path_provenance": {
            "observed": False,
            "evidence_type": "canonical_minimum_edit_alignment",
            "artifact_ref": "canonical_builder",
        },
        "operations": [{"op": "SUB", "pos": 1, "ref": "C", "alt": "G"}],
        "ambiguity": {
            "minimal_path_count": 1,
            "enumeration_capped": False,
            "selected_path_rule": "unique",
        },
        "intermediate_sequences": [],
    }


def _generation_task() -> dict:
    return {
        "schema_version": "generation_task.v2",
        "task_id": "task-1",
        "track_id": "track-a",
        "region": "five_utr",
        "source_id": "source-1",
        "source_sequence": "ACGU",
        "endpoint": "translation_efficiency",
        "candidate_id": None,
        "candidate_sequence": None,
        "legal_action_types": ["SUB", "STOP"],
        "max_edits": 2,
        "constraints": {
            "source_conditioned": True,
            "sequence_alphabet": "RNA",
            "allowed_operations": ["SUB", "STOP"],
            "min_length": 4,
            "max_length": 4,
        },
        "provenance": {
            "dataset_id": "dataset-1",
            "study_id": "study-1",
            "record_id": "record-1",
        },
    }


def _draft_validator(schema_name: str):
    jsonschema = pytest.importorskip("jsonschema")
    schemas = {
        name: _schema(name)
        for name in (
            "utr_edit_record.schema.json",
            "edit_script.schema.json",
            "generation_task.schema.json",
        )
    }
    store = {schema["$id"]: schema for schema in schemas.values()}
    schema = schemas[schema_name]
    resolver = jsonschema.RefResolver.from_schema(schema, store=store)
    return jsonschema.Draft202012Validator(schema, resolver=resolver)


def test_all_b0_schemas_are_draft_2020_12_and_fail_closed_at_top_level() -> None:
    for name in (
        "utr_edit_record.schema.json",
        "edit_script.schema.json",
        "generation_task.schema.json",
    ):
        schema = _schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False


def test_utr_record_schema_freezes_contract_canonical_and_grouping_fields() -> None:
    schema = _schema("utr_edit_record.schema.json")
    required = set(schema["required"])
    contract_fields = {
        "record_id",
        "dataset_id",
        "study_id",
        "assay_id",
        "context_id",
        "evidence_grade",
        "exposure_grade",
        "region",
        "organism",
        "cell_context",
        "reporter",
        "cargo",
        "endpoint",
        "timepoint",
        "source_id",
        "source_sequence",
        "candidate_sequence",
        "source_length",
        "candidate_length",
        "edit_script",
        "edit_types",
        "edit_positions",
        "reference_alleles",
        "alternate_alleles",
        "edit_count",
        "edit_distance",
        "source_value_raw",
        "candidate_value_raw",
        "delta_raw",
        "delta_normalized",
        "effect_standard_error",
        "replicate_count",
        "pair_type",
        "trajectory_observed",
        "trajectory_source",
        "paper_split",
        "canonical_split",
        "source_group",
        "gene_group",
        "study_group",
        "context_group",
        "sequence_cluster",
        "scaffold_group",
        "barcode_batch",
        "library_batch",
        "sequence_provenance",
        "label_provenance",
        "download_manifest",
        "license",
        "quality_flags",
        "historical_exposure",
    }
    assert contract_fields <= required
    assert set(REQUIRED_FIELDS) <= required
    assert schema["properties"]["region"]["enum"] == ["five_utr", "three_utr"]


def test_edit_operation_schema_matches_frozen_dynamic_coordinate_actions() -> None:
    operation = _schema("edit_script.schema.json")["$defs"]["operation"]
    assert operation["required"] == ["op", "pos", "ref", "alt"]
    assert operation["properties"]["op"]["enum"] == ["INS", "SUB", "DEL", "STOP"]


def test_generation_task_schema_has_no_label_or_value_fields() -> None:
    schema = _schema("generation_task.schema.json")
    serialized = json.dumps(schema["properties"], sort_keys=True).lower()
    for forbidden in (
        "source_value_raw",
        "candidate_value_raw",
        "delta_raw",
        "delta_normalized",
        "label",
    ):
        assert forbidden not in serialized


def test_candidate_store_label_scan_is_recursive_and_fail_closed() -> None:
    safe = {
        "task_id": "task-1",
        "source_id": "source-1",
        "source_sequence": "ACGU",
        "endpoint": "translation_efficiency",
        "candidate_id": "candidate-1",
        "candidate_sequence": "AGGU",
        "constraints": {"allowed_operations": ["SUB"]},
    }
    assert_candidate_store_label_free([safe])

    unsafe = dict(safe)
    unsafe["provenance"] = {"nested": {"delta_normalized": 1.25}}
    with pytest.raises(CandidateStoreLabelError, match="delta_normalized"):
        assert_candidate_store_label_free([unsafe])


def test_candidate_task_shape_has_no_unsealed_metadata_escape_hatch() -> None:
    task = _generation_task()
    validate_generation_task(task)
    task["metadata"] = {"x": 1}
    with pytest.raises(TrackContractError, match="unsealed fields"):
        validate_generation_task(task)


def test_generation_task_schema_seals_candidate_pair_and_rna_fields() -> None:
    validator = _draft_validator("generation_task.schema.json")
    valid = _generation_task()
    assert list(validator.iter_errors(valid)) == []

    mismatched_candidate_id = deepcopy(valid)
    mismatched_candidate_id["candidate_id"] = "candidate-1"
    assert list(validator.iter_errors(mismatched_candidate_id))

    mismatched_candidate_sequence = deepcopy(valid)
    mismatched_candidate_sequence["candidate_sequence"] = "AGGU"
    assert list(validator.iter_errors(mismatched_candidate_sequence))

    duplicate_motifs = deepcopy(valid)
    duplicate_motifs["constraints"]["forbidden_motifs"] = ["AUG", "AUG"]
    assert list(validator.iter_errors(duplicate_motifs))

    non_rna_motif = deepcopy(valid)
    non_rna_motif["constraints"]["forbidden_motifs"] = ["ATG"]
    assert list(validator.iter_errors(non_rna_motif))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda task: task["constraints"].update(
                {"allowed_operations": ["SUB"]}
            ),
            "allowed_operations",
        ),
        (
            lambda task: task["constraints"].update(
                {"min_length": 5, "max_length": 4}
            ),
            "min_length",
        ),
        (
            lambda task: task["constraints"].update(
                {"min_length": 5, "max_length": 8}
            ),
            "source_sequence length",
        ),
        (
            lambda task: task.update(
                {
                    "candidate_id": "candidate-1",
                    "candidate_sequence": "AAAA",
                    "max_edits": 1,
                }
            ),
            "edit distance",
        ),
        (
            lambda task: task.update(
                {
                    "candidate_id": "candidate-1",
                    "candidate_sequence": "ACGUA",
                }
            ),
            "candidate_sequence length",
        ),
    ],
)
def test_generation_task_runtime_closes_cross_field_action_and_length_semantics(
    mutation, message: str
) -> None:
    task = _generation_task()
    mutation(task)
    with pytest.raises(TrackContractError, match=message):
        validate_generation_task(task)


def test_draft202012_schema_accepts_real_d1_shape_and_rejects_fake_semantics() -> None:
    validator = _draft_validator("utr_edit_record.schema.json")
    valid = _schema_ready_intervention()
    assert list(validator.iter_errors(valid)) == []
    absolute = _schema_ready_absolute()
    assert list(validator.iter_errors(absolute)) == []

    fake_observed = deepcopy(valid)
    fake_observed["trajectory_source"] = "observed"
    fake_observed["trajectory_observed"] = False
    assert list(validator.iter_errors(fake_observed))

    fake_pair = deepcopy(valid)
    fake_pair["pair_type"] = "source_candidate_intervention"
    assert list(validator.iter_errors(fake_pair))

    ambiguous_alphabet = deepcopy(valid)
    ambiguous_alphabet["candidate_sequence"] = "ANGU"
    assert list(validator.iter_errors(ambiguous_alphabet))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", None),
        ("source_sequence", None),
        ("source_length", None),
        ("edit_script", None),
        ("edit_count", 0),
        ("edit_distance", None),
    ],
)
def test_intervention_schema_cannot_drop_source_or_edit_contract(
    field: str, value
) -> None:
    validator = _draft_validator("utr_edit_record.schema.json")
    payload = _schema_ready_intervention()
    payload[field] = value
    assert list(validator.iter_errors(payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", "invented-source"),
        ("source_sequence", "ACGU"),
        ("source_length", 4),
        (
            "edit_script",
            [{"op": "SUB", "pos": 1, "ref": "C", "alt": "G"}],
        ),
        ("edit_types", ["SUB"]),
        ("edit_positions", [1]),
        ("reference_alleles", ["C"]),
        ("alternate_alleles", ["G"]),
        ("edit_count", 1),
        ("edit_distance", 1),
        ("source_value_raw", 1.0),
        ("delta_raw", 0.4),
        ("delta_normalized", 0.4),
        ("raw_source_sequence", "ACGU"),
    ],
)
def test_absolute_property_schema_keeps_intervention_fields_null_or_empty(
    field: str, value
) -> None:
    validator = _draft_validator("utr_edit_record.schema.json")
    payload = _schema_ready_absolute()
    payload[field] = value
    assert list(validator.iter_errors(payload))


def test_record_schema_closes_trajectory_coupling_and_intermediate_claims() -> None:
    validator = _draft_validator("utr_edit_record.schema.json")

    latent_observed = _schema_ready_absolute()
    latent_observed["trajectory_observed"] = True
    assert list(validator.iter_errors(latent_observed))

    constructed_observed_coupling = _schema_ready_intervention()
    constructed_observed_coupling[
        "coupling_type"
    ] = "observed_endpoint_coupling"
    assert list(validator.iter_errors(constructed_observed_coupling))

    constructed_observed_intermediate = _schema_ready_intervention()
    constructed_observed_intermediate["intermediate_sequences"] = [
        {
            "sequence": "AGGU",
            "observed": True,
            "provenance": "constructed_alignment_step",
        }
    ]
    assert list(validator.iter_errors(constructed_observed_intermediate))

    missing_constructed_provenance = _schema_ready_intervention()
    missing_constructed_provenance["trajectory_provenance"] = {}
    assert list(validator.iter_errors(missing_constructed_provenance))


def test_pair_type_and_coupling_type_conditions_are_closed() -> None:
    validator = _draft_validator("utr_edit_record.schema.json")

    wrong_absolute_coupling = _schema_ready_absolute()
    wrong_absolute_coupling[
        "coupling_type"
    ] = "corruption_denoising_coupling"
    assert list(validator.iter_errors(wrong_absolute_coupling))

    unlabeled = _schema_ready_absolute()
    unlabeled.update(
        {
            "pair_type": "unlabeled_pretraining",
            "candidate_value_raw": None,
            "coupling_type": "corruption_denoising_coupling",
        }
    )
    assert list(validator.iter_errors(unlabeled)) == []
    unlabeled["coupling_type"] = "property_conditioned_target_coupling"
    assert list(validator.iter_errors(unlabeled))

    retrospective = _schema_ready_intervention()
    retrospective["pair_type"] = "retrospective_constructed_neighbor"
    assert list(validator.iter_errors(retrospective)) == []
    retrospective["trajectory_source"] = "observed"
    retrospective["trajectory_observed"] = True
    retrospective["coupling_type"] = "observed_endpoint_coupling"
    assert list(validator.iter_errors(retrospective))


def test_observed_edit_path_requires_explicit_observed_provenance() -> None:
    validator = _draft_validator("edit_script.schema.json")
    payload = _edit_script_payload()
    payload["path_status"] = "OBSERVED_TRAJECTORY"
    payload["path_provenance"] = {
        "observed": False,
        "evidence_type": "constructed_alignment",
        "artifact_ref": "canonical_builder",
    }
    assert list(validator.iter_errors(payload))

    payload["path_provenance"]["observed"] = True
    assert list(validator.iter_errors(payload))

    payload["path_provenance"]["evidence_type"] = "observed_trajectory"
    assert list(validator.iter_errors(payload)) == []

    payload["intermediate_sequences"] = [
        {
            "sequence": "AGGU",
            "observed": False,
            "provenance": "constructed_alignment_step",
        }
    ]
    assert list(validator.iter_errors(payload))


def test_constructed_edit_paths_cannot_fake_observed_intermediates() -> None:
    validator = _draft_validator("edit_script.schema.json")
    payload = _edit_script_payload()
    payload["path_status"] = "OBSERVED_ENDPOINT_ONLY_CONSTRUCTED_PATH"
    payload["path_provenance"]["evidence_type"] = "constructed_alignment"
    payload["intermediate_sequences"] = [
        {
            "sequence": "AGGU",
            "observed": False,
            "provenance": "canonical_alignment_step",
        }
    ]
    assert list(validator.iter_errors(payload)) == []

    payload["intermediate_sequences"][0]["observed"] = True
    assert list(validator.iter_errors(payload))


def test_edit_path_status_and_ambiguity_are_compatible() -> None:
    validator = _draft_validator("edit_script.schema.json")
    unique = _edit_script_payload()
    assert list(validator.iter_errors(unique)) == []

    unique["ambiguity"]["minimal_path_count"] = 2
    assert list(validator.iter_errors(unique))

    ambiguous = _edit_script_payload()
    ambiguous["path_status"] = "CANONICAL_PATH_AMBIGUOUS"
    ambiguous["ambiguity"]["minimal_path_count"] = 2
    ambiguous["ambiguity"]["selected_path_rule"] = "canonical_tie_break"
    assert list(validator.iter_errors(ambiguous)) == []

    ambiguous["ambiguity"]["minimal_path_count"] = 1
    assert list(validator.iter_errors(ambiguous))


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "SUB", "pos": 1, "ref": "CG", "alt": "A"},
        {"op": "SUB", "pos": 1, "ref": "C", "alt": "GU"},
        {"op": "SUB", "pos": 1, "ref": "C", "alt": "C"},
        {"op": "SUB", "pos": None, "ref": "C", "alt": "G"},
        {"op": "INS", "pos": 1, "ref": "C", "alt": "A"},
        {"op": "INS", "pos": 1, "ref": "", "alt": ""},
        {"op": "INS", "pos": None, "ref": "", "alt": "A"},
        {"op": "DEL", "pos": 1, "ref": "C", "alt": "A"},
        {"op": "DEL", "pos": 1, "ref": "", "alt": ""},
        {"op": "DEL", "pos": None, "ref": "C", "alt": ""},
        {"op": "STOP", "pos": 1, "ref": "", "alt": ""},
        {"op": "STOP", "pos": None, "ref": "A", "alt": ""},
        {"op": "SUB", "pos": 1, "ref": "N", "alt": "G"},
    ],
)
def test_edit_operation_schema_rejects_invalid_allele_or_position_shapes(
    operation: dict,
) -> None:
    validator = _draft_validator("edit_script.schema.json")
    payload = _edit_script_payload()
    payload["operations"] = [operation]
    assert list(validator.iter_errors(payload))


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "SUB", "pos": 1, "ref": "C", "alt": "G"},
        {"op": "INS", "pos": 1, "ref": "", "alt": "AG"},
        {"op": "DEL", "pos": 1, "ref": "CG", "alt": ""},
        {"op": "STOP", "pos": None, "ref": "", "alt": ""},
    ],
)
def test_edit_operation_schema_accepts_exact_frozen_action_shapes(
    operation: dict,
) -> None:
    validator = _draft_validator("edit_script.schema.json")
    payload = _edit_script_payload()
    payload["operations"] = [operation]
    assert list(validator.iter_errors(payload)) == []


def test_production_schema_validator_checks_every_jsonl_record(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    path = tmp_path / "canonical.jsonl"
    valid = _schema_ready_intervention()
    invalid = deepcopy(valid)
    invalid["pair_type"] = "invented_pair"
    path.write_text(
        json.dumps(valid) + "\n" + json.dumps(invalid) + "\n",
        encoding="utf-8",
    )
    report = validate_canonical_records_schema(path, ROOT / "schemas")
    assert report["record_count"] == 2
    assert report["invalid_record_count"] == 1
    assert report["status"] == "FAIL"


def test_structural_split_store_recursively_rejects_nested_values(
    tmp_path: Path,
) -> None:
    from data.utr_benchmark_v2.d1_builder import CANDIDATE_STORE_FIELDS

    canonical = _schema_ready_intervention()
    structural = {
        field: canonical[field]
        for field in CANDIDATE_STORE_FIELDS
        if field in canonical
    }
    path = tmp_path / "structural.jsonl"
    path.write_text(json.dumps(structural) + "\n", encoding="utf-8")
    assert len(load_structural_jsonl(path)) == 1

    unsafe = deepcopy(structural)
    unsafe["sequence_provenance"] = {
        "raw_artifact": "raw.tsv",
        "nested": {"delta_raw": 1.0},
    }
    path.write_text(json.dumps(unsafe) + "\n", encoding="utf-8")
    with pytest.raises(CandidateStoreLabelError, match="delta_raw"):
        load_structural_jsonl(path)
