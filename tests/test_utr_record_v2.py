from __future__ import annotations

from copy import deepcopy

import pytest

from data.utr_benchmark_v2.records import (
    CanonicalRecordError,
    CanonicalUTRRecord,
    canonical_record_id,
    validate_canonical_record,
)


def _intervention_record() -> dict:
    return {
        "record_id": "GSE149487:example:1",
        "dataset_id": "GSE149487",
        "study_id": "GSE149487",
        "assay_id": "plumage_te",
        "context_id": "prostate_context",
        "evidence_grade": "measured_paired",
        "exposure_grade": "previously_accessed",
        "region": "five_utr",
        "organism": "Homo sapiens",
        "cell_context": "prostate",
        "reporter": "reporter",
        "cargo": "reporter_cds",
        "endpoint": "translation_efficiency",
        "endpoint_provenance": {
            "raw_endpoint": "translation_efficiency",
            "transformation": "identity",
        },
        "timepoint": None,
        "source_id": "src:1",
        "source_sequence": "ACGU",
        "candidate_sequence": "AGGU",
        "source_length": 4,
        "candidate_length": 4,
        "edit_script": [
            {"op": "SUB", "pos": 1, "ref": "C", "alt": "G"},
        ],
        "edit_types": ["SUB"],
        "edit_positions": [1],
        "reference_alleles": ["C"],
        "alternate_alleles": ["G"],
        "edit_count": 1,
        "edit_distance": 1,
        "source_value_raw": 1.0,
        "candidate_value_raw": 1.4,
        "delta_raw": 0.4,
        "delta_normalized": 0.4,
        "effect_standard_error": 0.1,
        "replicate_count": 3,
        "pair_type": "true_wt_mutant",
        "trajectory_observed": False,
        "trajectory_source": "constructed",
        "trajectory_provenance": {},
        "coupling_type": "constructed_alignment_coupling",
        "paper_split": None,
        "canonical_split": None,
        "source_group": "src:1",
        "gene_group": "gene:1",
        "study_group": "GSE149487",
        "context_group": "prostate_context",
        "sequence_cluster": "cluster:1",
        "scaffold_group": "scaffold:reporter",
        "barcode_batch": "barcode_batch:GSE149487:not_reported",
        "library_batch": "library_batch:GSE149487:PLUMAGE",
        "sequence_provenance": {
            "raw_artifact": "raw/GSE149487.tsv",
            "processed_artifact": "processed/GSE149487.paper_clean.jsonl",
        },
        "label_provenance": {
            "raw_artifact": "raw/GSE149487.tsv",
            "columns": ["source_te", "candidate_te"],
        },
        "download_manifest": "data/p0/GSE149487/download_manifest.json",
        "license": "GEO_public_per_series_terms",
        "quality_flags": [],
        "historical_exposure": "previously_accessed_before_V2",
    }


def _absolute_record() -> dict:
    row = _intervention_record()
    row.update(
        {
            "record_id": "GSE114002:absolute:1",
            "dataset_id": "GSE114002",
            "study_id": "GSE114002",
            "source_id": None,
            "source_sequence": None,
            "candidate_sequence": "ACGU",
            "source_length": None,
            "candidate_length": 4,
            "edit_script": None,
            "edit_types": [],
            "edit_positions": [],
            "reference_alleles": [],
            "alternate_alleles": [],
            "edit_count": 0,
            "edit_distance": None,
            "source_value_raw": None,
            "candidate_value_raw": 1.4,
            "delta_raw": None,
            "delta_normalized": None,
            "pair_type": "absolute_property_only",
            "trajectory_source": "latent",
            "trajectory_observed": False,
            "coupling_type": "property_conditioned_target_coupling",
        }
    )
    return row


def test_valid_intervention_record_roundtrips() -> None:
    payload = _intervention_record()
    record = CanonicalUTRRecord.from_dict(payload)
    assert record.to_dict() == payload
    assert record.endpoint == "translation_efficiency"
    assert validate_canonical_record(payload) == payload
    assert canonical_record_id(payload) == canonical_record_id(deepcopy(payload))


def test_record_constructor_cannot_bypass_validation() -> None:
    payload = _intervention_record()
    payload["trajectory_observed"] = True
    with pytest.raises(CanonicalRecordError, match="constructed.*observed"):
        CanonicalUTRRecord(payload)


def test_valid_absolute_record_is_not_an_intervention() -> None:
    payload = _absolute_record()
    record = CanonicalUTRRecord.from_dict(payload)
    assert record.pair_type == "absolute_property_only"
    assert record.source_sequence is None
    assert record.edit_script is None


def test_unlabeled_pretraining_record_cannot_carry_a_label() -> None:
    payload = _absolute_record()
    payload.update(
        {
            "pair_type": "unlabeled_pretraining",
            "candidate_value_raw": None,
            "label_provenance": {},
        }
    )
    CanonicalUTRRecord.from_dict(payload)
    payload["candidate_value_raw"] = 1.0
    with pytest.raises(CanonicalRecordError, match="unlabeled_pretraining"):
        CanonicalUTRRecord.from_dict(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"pair_type": "absolute_property_only"},
        {"pair_type": "unlabeled_pretraining"},
    ],
)
def test_absolute_pair_cannot_masquerade_as_intervention(change: dict) -> None:
    payload = _intervention_record()
    payload.update(change)
    with pytest.raises(CanonicalRecordError, match="absolute"):
        CanonicalUTRRecord.from_dict(payload)


def test_absolute_pair_cannot_carry_delta() -> None:
    payload = _absolute_record()
    payload["delta_raw"] = 0.1
    with pytest.raises(CanonicalRecordError, match="absolute"):
        CanonicalUTRRecord.from_dict(payload)


def test_constructed_trajectory_cannot_be_marked_observed() -> None:
    payload = _intervention_record()
    payload["trajectory_observed"] = True
    with pytest.raises(CanonicalRecordError, match="constructed.*observed"):
        CanonicalUTRRecord.from_dict(payload)


def test_trajectory_source_and_coupling_type_must_agree() -> None:
    payload = _intervention_record()
    payload["coupling_type"] = "property_conditioned_target_coupling"
    with pytest.raises(CanonicalRecordError, match="incompatible"):
        CanonicalUTRRecord.from_dict(payload)


def test_observed_trajectory_requires_provenance_and_coupling() -> None:
    payload = _intervention_record()
    payload.update(
        {
            "trajectory_source": "observed",
            "trajectory_observed": True,
            "coupling_type": "observed_endpoint_coupling",
        }
    )
    with pytest.raises(CanonicalRecordError, match="trajectory_provenance"):
        CanonicalUTRRecord.from_dict(payload)

    payload["trajectory_provenance"] = {
        "raw_artifact": "raw/actions.tsv",
        "action_order_column": "step",
    }
    CanonicalUTRRecord.from_dict(payload)


@pytest.mark.parametrize(
    "endpoint",
    [
        "translation_efficiency|rna_abundance",
        "half_life;decay_rate",
        "expression_score",
    ],
)
def test_endpoints_must_remain_separate(endpoint: str) -> None:
    payload = _intervention_record()
    payload["endpoint"] = endpoint
    with pytest.raises(CanonicalRecordError, match="endpoint"):
        CanonicalUTRRecord.from_dict(payload)


def test_endpoint_and_data_provenance_are_required() -> None:
    for field in ("endpoint_provenance", "sequence_provenance", "download_manifest"):
        payload = _intervention_record()
        payload[field] = {}
        with pytest.raises(CanonicalRecordError, match=field):
            CanonicalUTRRecord.from_dict(payload)

    payload = _intervention_record()
    payload["label_provenance"] = {}
    with pytest.raises(CanonicalRecordError, match="label_provenance"):
        CanonicalUTRRecord.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    ["scaffold_group", "barcode_batch", "library_batch"],
)
def test_atomic_grouping_fields_are_required(field: str) -> None:
    payload = _intervention_record()
    del payload[field]
    with pytest.raises(CanonicalRecordError, match=field):
        CanonicalUTRRecord.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scaffold_group", ""),
        ("scaffold_group", "UNKNOWN"),
        ("barcode_batch", "NA"),
        ("library_batch", "N/A"),
        ("scaffold_group", "unscoped-reporter"),
        ("barcode_batch", ":missing-namespace"),
        ("library_batch", "library_batch:"),
    ],
)
def test_atomic_grouping_fields_require_scoped_nonplaceholder_ids(
    field: str, value: str
) -> None:
    payload = _intervention_record()
    payload[field] = value
    with pytest.raises(CanonicalRecordError, match=field):
        CanonicalUTRRecord.from_dict(payload)


def test_scoped_unknown_group_is_not_a_bare_placeholder() -> None:
    payload = _intervention_record()
    payload["barcode_batch"] = "barcode_batch:GSE149487:UNKNOWN"
    payload["library_batch"] = "NOT_APPLICABLE:GSE149487:no_library_batch"
    CanonicalUTRRecord.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_value_raw", float("nan")),
        ("source_value_raw", float("inf")),
        ("candidate_value_raw", float("-inf")),
        ("delta_raw", float("nan")),
        ("delta_normalized", float("inf")),
        ("effect_standard_error", float("-inf")),
    ],
)
def test_numeric_label_and_effect_fields_reject_nan_and_inf(
    field: str, value: float
) -> None:
    payload = _intervention_record()
    payload[field] = value
    with pytest.raises(CanonicalRecordError, match=field):
        CanonicalUTRRecord.from_dict(payload)


@pytest.mark.parametrize(
    "placeholder",
    [
        "",
        "UNKNOWN",
        "NA",
        "N/A",
        "not_opened_or_not_applicable",
        "raw:GSE149487:UNKNOWN",
    ],
)
def test_sequence_provenance_rejects_raw_placeholders(placeholder: str) -> None:
    payload = _intervention_record()
    payload["sequence_provenance"]["raw_artifact"] = placeholder
    with pytest.raises(CanonicalRecordError, match="raw"):
        CanonicalUTRRecord.from_dict(payload)


@pytest.mark.parametrize(
    "placeholder",
    ["", "UNKNOWN", "NA", "N/A", "processed:GSE149487:NA"],
)
def test_sequence_provenance_rejects_processed_placeholders(
    placeholder: str,
) -> None:
    payload = _intervention_record()
    payload["sequence_provenance"]["processed_artifact"] = placeholder
    with pytest.raises(CanonicalRecordError, match="processed"):
        CanonicalUTRRecord.from_dict(payload)


def test_nested_sequence_provenance_placeholders_fail_closed() -> None:
    payload = _intervention_record()
    payload["sequence_provenance"] = {
        "raw": {"path": "UNKNOWN", "sha256": "NA"},
        "processed": {"path": "processed/valid.jsonl"},
    }
    with pytest.raises(CanonicalRecordError, match="raw"):
        CanonicalUTRRecord.from_dict(payload)


@pytest.mark.parametrize(
    "placeholder",
    ["", "UNKNOWN", "NA", "N/A", "manifest:GSE149487:UNKNOWN"],
)
def test_download_manifest_rejects_placeholders(placeholder: str) -> None:
    payload = _intervention_record()
    payload["download_manifest"] = placeholder
    with pytest.raises(CanonicalRecordError, match="download_manifest"):
        CanonicalUTRRecord.from_dict(payload)


def test_nested_download_manifest_placeholders_fail_closed() -> None:
    payload = _intervention_record()
    payload["download_manifest"] = {
        "path": "UNKNOWN",
        "sha256": "NA",
    }
    with pytest.raises(CanonicalRecordError, match="download_manifest"):
        CanonicalUTRRecord.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "placeholder"),
    [
        ("license", ""),
        ("license", "UNKNOWN"),
        ("license", "NA"),
        ("license", "license:GSE149487:UNKNOWN"),
        ("historical_exposure", ""),
        ("historical_exposure", "N/A"),
        ("historical_exposure", "UNKNOWN"),
        ("historical_exposure", "UNKNOWN:GSE149487"),
    ],
)
def test_governance_provenance_rejects_bare_placeholders(
    field: str, placeholder: str
) -> None:
    payload = _intervention_record()
    payload[field] = placeholder
    with pytest.raises(CanonicalRecordError, match=field):
        CanonicalUTRRecord.from_dict(payload)


def test_delta_and_endpoint_mapping_fail_closed() -> None:
    payload = _intervention_record()
    payload["delta_raw"] = 9.0
    with pytest.raises(CanonicalRecordError, match="delta_raw"):
        CanonicalUTRRecord.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_length", 5),
        ("edit_count", 2),
        ("edit_types", ["DEL"]),
        ("edit_positions", [0]),
        ("reference_alleles", ["A"]),
        ("alternate_alleles", ["U"]),
        ("edit_distance", 2),
    ],
)
def test_derived_edit_fields_must_match_canonical_script(
    field: str, value: object
) -> None:
    payload = _intervention_record()
    payload[field] = value
    with pytest.raises(CanonicalRecordError, match=field):
        CanonicalUTRRecord.from_dict(payload)


def test_noncanonical_but_roundtripping_script_is_rejected() -> None:
    payload = _intervention_record()
    payload.update(
        {
            "source_sequence": "AAAA",
            "candidate_sequence": "AAA",
            "source_length": 4,
            "candidate_length": 3,
            "edit_script": [
                {"op": "DEL", "pos": 0, "ref": "A", "alt": ""},
            ],
            "edit_types": ["DEL"],
            "edit_positions": [0],
            "reference_alleles": ["A"],
            "alternate_alleles": [""],
            "source_value_raw": 1.0,
            "candidate_value_raw": 0.8,
            "delta_raw": -0.2,
            "delta_normalized": -0.2,
            "pair_type": "measured_indel_pair",
        }
    )
    with pytest.raises(CanonicalRecordError, match="canonical"):
        CanonicalUTRRecord.from_dict(payload)
