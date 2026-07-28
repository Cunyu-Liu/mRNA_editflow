from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from data.utr_benchmark_v2.d1_builder import (
    BLOCKED_DATASET_POLICIES,
    CANDIDATE_STORE_FORBIDDEN_FIELDS,
    D1_SCOPE_DATASETS,
    GSE217518_U3_SUFFIX_DNA,
    build_dataset_from_config,
    build_dataset_rows,
    candidate_store_label_paths,
    dataset_policy,
    extract_gse200304_exact_join,
    write_dataset_result,
)
from data.utr_benchmark_v2.edit_script import apply_edit_script
from data.utr_benchmark_v2.records import validate_canonical_record
from scripts.data.build_b0_splits import validate_canonical_records_schema
from scripts.data.validate_d1_acceptance import validate_d1_root
from scripts.data.validate_d1_acceptance import _validate_dataset
from scripts.data.build_d1_utr_benchmark import _parse_config, build_snapshot


ROOT = Path(__file__).resolve().parents[1]


def _strict_fixture_selection_policy() -> dict[str, object]:
    scope_path = ROOT / "data_registry/d1_dataset_scope_manifest.yaml"
    scope_bytes = scope_path.read_bytes()
    return {
        "candidate_final_labels_used_for_dataset_role_selection": False,
        "goal_contract_sha256": (
            "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"
        ),
        "dataset_scope_manifest": {
            "path": str(scope_path),
            "bytes": len(scope_bytes),
            "sha256": hashlib.sha256(scope_bytes).hexdigest(),
        },
    }


def _source(length: int = 50) -> str:
    return ("ACGU" * ((length + 3) // 4))[:length]


def _substitution(sequence: str, position: int, alternate: str) -> str:
    assert sequence[position] != alternate
    return sequence[:position] + alternate + sequence[position + 1 :]


def _provenance(tmp_path: Path, name: str) -> dict[str, object]:
    source = tmp_path / name
    source.write_text("fixture\n", encoding="utf-8")
    return {
        "input_path": str(source),
        "download_manifest": "fixture-download-manifest.json",
        "license": "fixture-only",
        "fixture": True,
    }


def test_dataset_policies_fail_closed_without_opening_blocked_labels():
    expected = {
        "GSE145046": "BLOCKED_SCAFFOLD_NOT_FROZEN",
        "GSE149487": "BLOCKED_EXACT_PAIR_MAPPING_UNRECOVERED",
        "ENCSR854RUF_raw62": "OBSERVATIONAL_ONLY_NOT_INTERVENTION",
        "GSE330741": "METADATA_ONLY_FINAL_LABELS_UNOPENED",
        "GSE291719": "METADATA_ONLY_FINAL_LABELS_UNOPENED",
    }
    for dataset_id, reason in expected.items():
        policy = dataset_policy(dataset_id)
        assert policy["status"] == "blocked"
        assert policy["reason_code"] == reason
        assert policy["read_final_labels"] is False
        assert dataset_id in BLOCKED_DATASET_POLICIES
    assert dataset_policy("GSE217518")["status"] == "eligible"


def test_gse114002_separates_anchored_snv_from_absolute_prior(tmp_path: Path):
    source = _source()
    candidate = _substitution(source, 7, "A")
    unanchored = "U" * 50
    rows = [
        {
            "row_id": "anchor",
            "library": "snv",
            "mother": source,
            "utr": source,
            "rl": "1.0",
        },
        {
            "row_id": "variant",
            "library": "snv",
            "mother": source,
            "utr": candidate,
            "rl": "2.5",
        },
        {
            "row_id": "absolute",
            "library": "random",
            "utr": "AUGCAUGC",
            "rl": "0.75",
        },
        {
            "row_id": "unanchored",
            "library": "snv",
            "mother": unanchored,
            "utr": _substitution(unanchored, 3, "A"),
            "rl": "4.0",
        },
    ]

    result = build_dataset_rows(
        "GSE114002",
        rows,
        provenance=_provenance(tmp_path, "gse114002.csv"),
        fixture_mode=True,
    )

    assert result["status"] == "accepted_fixture"
    assert result["accounting"] == {
        "total_input_rows": 4,
        "accepted_intervention_rows": 1,
        "accepted_absolute_rows": 1,
        "accepted_input_rows": 2,
        "auxiliary_source_anchor_rows": 1,
        "rejected_rows": 1,
        "accounted_rows": 4,
    }
    intervention = next(
        record
        for record in result["label_records"]
        if record["pair_type"] == "true_wt_mutant"
    )
    assert intervention["region"] == "five_utr"
    assert intervention["edit_types"] == ["SUB"]
    assert intervention["source_value_raw"] == 1.0
    assert intervention["candidate_value_raw"] == 2.5
    assert intervention["delta_raw"] == 1.5
    assert intervention["trajectory_observed"] is False
    assert intervention["label_provenance"]["status"] == "PROVIDED_LABEL_ONLY"
    assert result["label_reproduction"]["raw_reproduction_claim_allowed"] is False
    assert (
        apply_edit_script(intervention["source_sequence"], intervention["edit_script"])
        == intervention["candidate_sequence"]
    )

    absolute = next(
        record
        for record in result["label_records"]
        if record["pair_type"] == "absolute_property_only"
    )
    assert absolute["source_sequence"] is None
    assert absolute["edit_script"] is None
    assert absolute["edit_count"] == 0
    assert "ABSOLUTE_SEQUENCE_NOT_INTERVENTION" in absolute["quality_flags"]
    assert result["rejected_records"][0]["reason_code"] == "MISSING_SOURCE_ANCHOR"


def test_gse200304_enforces_201nt_single_substitution_and_discloses_count_gap(
    tmp_path: Path,
):
    source = _source(201)
    candidate = _substitution(source, 100, "C")
    rows = [
        {
            "row_id": "pair-1",
            "wt_sequence": source,
            "mutant_sequence": candidate,
            "endpoint": "translation_efficiency",
            "wt_value": "1.5",
            "mutant_value": "2.0",
            "gene": "GENE1",
        },
        {
            "row_id": "bad-length",
            "wt_sequence": source[:-1],
            "mutant_sequence": candidate[:-1],
            "endpoint": "translation_efficiency",
            "wt_value": "1.0",
            "mutant_value": "2.0",
        },
    ]
    result = build_dataset_rows(
        "GSE200304",
        rows,
        provenance=_provenance(tmp_path, "gse200304.tsv"),
        fixture_mode=True,
    )

    assert result["accounting"]["accepted_intervention_rows"] == 1
    assert result["accounting"]["rejected_rows"] == 1
    assert result["rejected_records"][0]["reason_code"] == "UNEXPECTED_PAIR_LENGTH"
    assert result["label_reproduction"]["status"] == "PROVIDED_LABEL_ONLY"
    assert result["paper_count_reconciliation"] == {
        "paper_reported_pairs": 6892,
        "production_expected_sequence_pairs": 6885,
        "production_expected_both_labeled_pairs": 6120,
        "production_expected_source_only_pairs": 192,
        "production_expected_candidate_only_pairs": 225,
        "production_expected_neither_labeled_pairs": 348,
        "fixture_observed_pairs": 1,
        "observed_label_coverage": {
            "both_labeled": 1,
            "source_only": 0,
            "candidate_only": 0,
            "neither_labeled": 0,
        },
        "known_discrepancy": 7,
        "status": "fixture_only_not_a_production_count_gate",
    }


def test_gse200304_exact_join_keeps_label_coverage_roles_separate():
    source = _source(201).replace("U", "T")
    candidate = _substitution(source, 10, "C")
    other_candidate = _substitution(source, 11, "A")
    constructs = [
        {"merged_id": "pair1_WT", "Type": "WT", "201bp": source},
        {"merged_id": "pair1_Mutant", "Type": "Mutant", "201bp": candidate},
        {"merged_id": "pair2_WT", "Type": "WT", "201bp": source},
        {
            "merged_id": "pair2_Mutant",
            "Type": "Mutant",
            "201bp": other_candidate,
        },
        {"merged_id": "control1", "Type": "Control", "201bp": source},
    ]
    labels = [
        {"Barcode": "pair1_WT", "Freq": "1.0"},
        {"Barcode": "pair1_Mutant", "Freq": "2.0"},
        {"Barcode": "pair2_Mutant", "Freq": "3.0"},
    ]
    rows, audit = extract_gse200304_exact_join(constructs, labels)
    assert audit["join_rule"] == "construct.merged_id == labels.Barcode"
    assert audit["sequence_pair_groups"] == 2
    assert audit["control_constructs"] == 1
    assert audit["pair_201nt_count"] == 2
    assert audit["pair_hamming_distribution"] == {"1": 2}
    paired = [row for row in rows if not row.get("_d1_rejection_reason")]
    assert (
        sum(
            row["source_value"] is not None and row["candidate_value"] is not None
            for row in paired
        )
        == 1
    )
    assert (
        sum(
            row["source_value"] is None and row["candidate_value"] is not None
            for row in paired
        )
        == 1
    )
    assert rows[-1]["_d1_rejection_reason"] == "CONTROL_NOT_WT_MUTANT_PAIR"
    lineage = audit["raw_row_lineage_records"]
    assert len(lineage) == len(constructs) + len(labels)
    assert audit["raw_row_lineage_summary"]["row_counts_by_table"] == {
        "construct_table": len(constructs),
        "processed_label_table": len(labels),
    }
    assert len({row["lineage_id"] for row in lineage}) == len(lineage)
    assert len({row["raw_row_key"] for row in lineage}) == len(lineage)
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", row["raw_row_fingerprint_sha256"])
        for row in lineage
    )


def test_gse246381_is_permanently_retrospective_and_never_trainable(
    tmp_path: Path,
):
    source = _source(75)
    candidate = _substitution(source, 19, "A")
    rows = [
        {
            "SeqID": "Variant;GENE1;Family=F1",
            "RefSequence": source,
            "AltSequence": candidate,
            "SYMBOL": "GENE1",
            "endpoint": "polysome_total_rna_logfc",
            "candidate_value": "-0.3",
            "reported_effect": "-0.27",
        }
    ]
    result = build_dataset_rows(
        "GSE246381",
        rows,
        provenance=_provenance(tmp_path, "gse246381.xlsx"),
        fixture_mode=True,
    )
    record = result["label_records"][0]
    assert record["exposure_grade"] == "E4"
    assert record["historical_exposure"] == ("historically_exposed_retrospective_E4")
    assert record["paper_split"] == "retrospective_only"
    assert record["canonical_split"] == "retrospective_only"
    assert "NO_TRAINING_OR_SELECTION" in record["quality_flags"]
    assert "reported_effect_raw" not in record
    assert record["label_provenance"]["reported_effect_raw"] == pytest.approx(-0.27)
    assert validate_canonical_record(record) == record
    canonical_path = tmp_path / "gse246381.canonical.jsonl"
    canonical_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    schema_report = validate_canonical_records_schema(
        canonical_path,
        Path(__file__).resolve().parents[1] / "schemas",
    )
    assert schema_report["invalid_record_count"] == 0
    assert schema_report["status"] == "PASS"
    assert result["allowed_uses"] == [
        "retrospective_stress_test",
        "diagnostics_without_selection",
    ]
    assert "training" in result["forbidden_uses"]

    without_reported_effect = build_dataset_rows(
        "GSE246381",
        [
            {
                "SeqID": "Variant;GENE2;Family=F2",
                "RefSequence": source,
                "AltSequence": _substitution(source, 23, "A"),
            }
        ],
        provenance=_provenance(tmp_path, "gse246381-no-effect.xlsx"),
        fixture_mode=True,
    )["label_records"][0]
    assert "reported_effect_raw" not in without_reported_effect["label_provenance"]


def test_gse217518_uses_unique_ref_mut_pairs_and_exact_observed_flanks(
    tmp_path: Path,
):
    source = _source(115)
    candidate = _substitution(source, 51, "A")
    suffix = GSE217518_U3_SUFFIX_DNA.replace("T", "U")
    base = "3M_fixture_NM_000001.1(GENE1):c.1A|G_transcript_utr3"
    rows = [
        {
            "seqName": f"{base}_Ref_1",
            "sequence": source + suffix,
            "halfLife": "10.0",
            "region": "three_utr",
        },
        {
            "seqName": f"{base}_Mut_1",
            "sequence": candidate + suffix,
            "halfLife": "12.5",
            "region": "three_utr",
        },
        {
            "seqName": "3M_unpaired_utr3_Ref_1",
            "sequence": source + suffix,
            "halfLife": "8.0",
            "region": "three_utr",
        },
    ]
    result = build_dataset_rows(
        "GSE217518",
        rows,
        provenance=_provenance(tmp_path, "SHdiNT_U3.csv"),
        fixture_mode=True,
    )
    assert result["accounting"]["accepted_intervention_rows"] == 1
    assert result["accounting"]["accepted_input_rows"] == 2
    assert result["accounting"]["rejected_rows"] == 1
    assert result["accounting"]["accounted_rows"] == 3
    assert result["rejected_records"][0]["reason_code"] == "UNPAIRED_ENDPOINT"
    record = result["label_records"][0]
    assert record["source_sequence"] == source
    assert record["candidate_sequence"] == candidate
    assert record["raw_source_sequence"] == source + suffix
    assert record["region"] == "three_utr"
    assert record["source_value_raw"] == 10.0
    assert record["candidate_value_raw"] == 12.5
    assert record["delta_raw"] == 2.5
    assert record["trajectory_source"] == "constructed"
    assert record["trajectory_observed"] is False
    assert (
        record["canonicalization_provenance"]["reference"]["rule"]
        == "trim_observed_fixed_U3_suffix_only"
    )


def test_mprau_is_blocked_when_frozen_reference_evidence_is_insufficient(
    tmp_path: Path,
):
    result = build_dataset_rows(
        "MPRAu_processed_ENCSR854RUF",
        [],
        provenance=_provenance(tmp_path, "mprau.xlsx"),
        fixture_mode=True,
        reference_audit={
            "status": "partial",
            "reference_coverage": 0.99,
            "roundtrip_fraction": 0.99,
        },
    )
    assert result["status"] == "blocked"
    assert result["reason_code"] == "BLOCKED_MPRAU_REFERENCE_NOT_FROZEN_100_PERCENT"
    assert result["candidate_records"] == []
    assert result["label_records"] == []
    assert result["paper_eligible"] is False


def test_mprau_config_does_not_open_input_before_reference_gate(tmp_path: Path):
    missing_final_labels = tmp_path / "must-not-be-opened.xlsx"
    result = build_dataset_from_config(
        {
            "dataset_id": "MPRAu_processed_ENCSR854RUF",
            "download_manifest": "metadata-only",
            "license": "ENCODE public",
            "reference_audit": {
                "status": "partial",
                "reference_coverage": 0.99,
                "roundtrip_fraction": 0.99,
            },
            "input_files": [
                {
                    "path": str(missing_final_labels),
                    "format": "xlsx",
                    "sheet_name": "final-labels",
                }
            ],
        },
        fixture_mode=True,
    )
    assert result["status"] == "blocked"
    assert result["input_provenance"]["input_files"] == []
    assert result["input_provenance"]["input_access"] == (
        "not_opened_due_to_fail_closed_dataset_policy"
    )
    assert not missing_final_labels.exists()


def test_production_provenance_is_strict_for_eligible_but_metadata_only_for_blocked(
    tmp_path: Path,
):
    source = _source(50)
    candidate = _substitution(source, 3, "A")
    input_path = tmp_path / "eligible.csv"
    input_path.write_text("fixture\n", encoding="utf-8")
    manifest_path = tmp_path / "download-manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    rows = [
        {"row_id": "a", "library": "snv", "mother": source, "utr": source, "rl": 1},
        {
            "row_id": "v",
            "library": "snv",
            "mother": source,
            "utr": candidate,
            "rl": 2,
        },
    ]
    eligible = build_dataset_rows(
        "GSE114002",
        rows,
        provenance={
            "input_path": str(input_path),
            "download_manifest": str(manifest_path),
            "license": "GEO public per-series terms",
        },
        fixture_mode=False,
    )
    assert eligible["paper_eligible"] is True

    missing_manifest = build_dataset_rows(
        "GSE114002",
        rows,
        provenance={
            "input_path": str(input_path),
            "download_manifest": str(tmp_path / "missing-manifest.json"),
            "license": "GEO public per-series terms",
        },
        fixture_mode=False,
    )
    assert missing_manifest["paper_eligible"] is False

    blocked = build_dataset_from_config(
        {
            "dataset_id": "GSE330741",
            "download_manifest": str(manifest_path),
            "license": "GEO public per-series terms",
            "input_files": [{"path": str(tmp_path / "must-not-open-final-labels.tsv")}],
        },
        fixture_mode=False,
    )
    blocked_root = write_dataset_result(blocked, tmp_path / "blocked-stage")
    validation = _validate_dataset(blocked_root, fixture_mode=False)
    assert validation["passed"] is True
    assert blocked["input_provenance"]["input_files"] == []


def test_writer_physically_separates_candidate_and_label_stores(tmp_path: Path):
    source = _source(201)
    candidate = _substitution(source, 5, "U")
    result = build_dataset_rows(
        "GSE200304",
        [
            {
                "row_id": "pair-1",
                "source_sequence": source,
                "candidate_sequence": candidate,
                "endpoint": "mrna_stability",
                "source_value": 1.0,
                "candidate_value": 1.2,
            }
        ],
        provenance=_provenance(tmp_path, "input.tsv"),
        fixture_mode=True,
    )
    dataset_root = write_dataset_result(result, tmp_path / "D1")
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    candidate_path = dataset_root / manifest["outputs"]["candidate_store"]["path"]
    label_path = dataset_root / manifest["outputs"]["label_store"]["path"]
    assert candidate_path.resolve() != label_path.resolve()

    candidates = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
    ]
    assert candidates
    for record in candidates:
        assert not (set(record) & CANDIDATE_STORE_FORBIDDEN_FIELDS)
    labels = [
        json.loads(line) for line in label_path.read_text(encoding="utf-8").splitlines()
    ]
    assert labels[0]["candidate_value_raw"] == 1.2
    assert manifest["outputs"]["candidate_store"]["sha256"]
    assert manifest["outputs"]["label_store"]["sha256"]


def test_candidate_policy_tamper_fails_even_after_resealing_self_hash(
    tmp_path: Path,
) -> None:
    source = _source(75)
    result = build_dataset_rows(
        "GSE246381",
        [
            {
                "SeqID": "Variant;GENE;Family=F",
                "RefSequence": source,
                "AltSequence": _substitution(source, 5, "U"),
            }
        ],
        provenance=_provenance(tmp_path, "gse246.xlsx"),
        fixture_mode=True,
    )
    dataset_root = write_dataset_result(result, tmp_path / "D1")
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_path = dataset_root / manifest["outputs"]["candidate_store"]["path"]
    candidates = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
    ]
    candidates[0]["canonical_split"] = "train"
    candidates[0]["historical_exposure"] = "untouched_E5"
    candidates[0]["quality_flags"] = []
    candidate_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    manifest["outputs"]["candidate_store"]["bytes"] = candidate_path.stat().st_size
    manifest["outputs"]["candidate_store"]["sha256"] = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = _validate_dataset(dataset_root, fixture_mode=True)
    checks = {check["name"]: check["passed"] for check in validation["checks"]}
    assert checks["output_integrity:candidate_store"] is True
    assert checks["candidate_store_label_free"] is True
    assert checks["candidate_store_exact_canonical_projection"] is False
    assert validation["passed"] is False


def test_forged_ambiguity_count_fails_after_label_store_reseal(
    tmp_path: Path,
) -> None:
    source = _source(201)
    result = build_dataset_rows(
        "GSE200304",
        [
            {
                "row_id": "pair",
                "source_sequence": source,
                "candidate_sequence": _substitution(source, 5, "U"),
                "endpoint": "stability",
                "source_value": 1.0,
                "candidate_value": 1.2,
            }
        ],
        provenance=_provenance(tmp_path, "ambiguity.tsv"),
        fixture_mode=True,
    )
    dataset_root = write_dataset_result(result, tmp_path / "D1")
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_path = dataset_root / manifest["outputs"]["label_store"]["path"]
    labels = [
        json.loads(line) for line in label_path.read_text(encoding="utf-8").splitlines()
    ]
    labels[0]["edit_script_ambiguity"]["equivalent_minimal_script_count"] += 1
    labels[0]["edit_script_ambiguity"]["path_ambiguous"] = True
    label_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in labels),
        encoding="utf-8",
    )
    manifest["outputs"]["label_store"]["bytes"] = label_path.stat().st_size
    manifest["outputs"]["label_store"]["sha256"] = hashlib.sha256(
        label_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = _validate_dataset(dataset_root, fixture_mode=True)
    checks = {check["name"]: check["passed"] for check in validation["checks"]}
    assert checks["output_integrity:label_store"] is True
    assert checks["canonical_edit_script_and_ambiguity_exactly_recomputed"] is False
    assert validation["passed"] is False


def test_paper_clean_replacement_cannot_pass_by_refreshing_its_self_hash(
    tmp_path: Path,
):
    source = _source(201)
    result = build_dataset_rows(
        "GSE200304",
        [
            {
                "row_id": "pair-1",
                "source_sequence": source,
                "candidate_sequence": _substitution(source, 5, "U"),
                "endpoint": "mrna_stability",
                "source_value": 1.0,
                "candidate_value": 1.2,
            }
        ],
        provenance=_provenance(tmp_path, "paper-clean.tsv"),
        fixture_mode=True,
    )
    dataset_root = write_dataset_result(result, tmp_path / "D1")
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paper_path = dataset_root / manifest["outputs"]["paper_clean"]["path"]
    paper_rows = [
        json.loads(line) for line in paper_path.read_text(encoding="utf-8").splitlines()
    ]
    paper_rows[0]["raw_source_sequence"] = "AAAA"
    paper_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in paper_rows),
        encoding="utf-8",
    )
    manifest["outputs"]["paper_clean"]["bytes"] = paper_path.stat().st_size
    manifest["outputs"]["paper_clean"]["sha256"] = hashlib.sha256(
        paper_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = _validate_dataset(dataset_root, fixture_mode=True)
    failed = {check["name"]: check["passed"] for check in validation["checks"]}
    assert failed["output_integrity:paper_clean"] is True
    assert failed["paper_clean_strict_content_and_layer_binding"] is False
    assert validation["passed"] is False


def test_missing_hardened_output_returns_structured_failure_instead_of_crashing(
    tmp_path: Path,
):
    source = _source(201)
    candidate = _substitution(source, 5, "U")
    result = build_dataset_rows(
        "GSE200304",
        [
            {
                "row_id": "legacy-pair",
                "source_sequence": source,
                "candidate_sequence": candidate,
                "endpoint": "stability",
                "source_value": 1.0,
                "candidate_value": 1.2,
            }
        ],
        provenance=_provenance(tmp_path, "legacy-output.tsv"),
        fixture_mode=True,
    )
    dataset_root = write_dataset_result(result, tmp_path / "D1")
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["outputs"]["raw_row_lineage"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = _validate_dataset(dataset_root, fixture_mode=True)
    assert validation["passed"] is False
    assert validation["status"] == "accepted_fixture"
    assert validation["paper_clean_validation"]["reason"] == (
        "required_outputs_missing"
    )
    assert validation["raw_row_lineage_validation"]["reason"] == (
        "required_outputs_missing"
    )


def test_gse200304_lineage_reopens_both_raw_tables_and_rejects_forged_disposition(
    tmp_path: Path,
):
    source = _source(201).replace("U", "T")
    candidate = _substitution(source, 5, "A")
    construct_path = tmp_path / "constructs.jsonl"
    label_path = tmp_path / "labels.jsonl"
    construct_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"merged_id": "pair_WT", "Type": "WT", "201bp": source},
                {
                    "merged_id": "pair_Mutant",
                    "Type": "Mutant",
                    "201bp": candidate,
                },
                {"merged_id": "control", "Type": "Control", "201bp": source},
            )
        ),
        encoding="utf-8",
    )
    label_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"Barcode": "pair_WT", "Freq": 1.0},
                {"Barcode": "pair_Mutant", "Freq": 2.0},
                {"Barcode": "control", "Freq": 3.0},
            )
        ),
        encoding="utf-8",
    )
    result = build_dataset_from_config(
        {
            "dataset_id": "GSE200304",
            "download_manifest": "fixture-manifest",
            "license": "fixture-only",
            "input_files": [
                {
                    "path": str(construct_path),
                    "format": "jsonl",
                    "role": "construct_table",
                },
                {
                    "path": str(label_path),
                    "format": "jsonl",
                    "role": "processed_label_table",
                },
            ],
        },
        fixture_mode=True,
    )
    dataset_root = write_dataset_result(result, tmp_path / "D1")
    clean = _validate_dataset(dataset_root, fixture_mode=True)
    assert clean["raw_row_lineage_validation"]["passed"] is True

    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lineage_path = dataset_root / manifest["outputs"]["raw_row_lineage"]["path"]
    rows = [
        json.loads(line)
        for line in lineage_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["disposition"] = "FORGED_BUT_SCHEMA_SHAPED"
    lineage_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest["outputs"]["raw_row_lineage"]["bytes"] = lineage_path.stat().st_size
    manifest["outputs"]["raw_row_lineage"]["sha256"] = hashlib.sha256(
        lineage_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    forged = _validate_dataset(dataset_root, fixture_mode=True)
    assert forged["raw_row_lineage_validation"]["passed"] is False
    assert (
        forged["raw_row_lineage_validation"]["checks"]["summary_exactly_recomputed"]
        is False
    )


def test_candidate_store_scans_nested_provenance_for_label_leakage(
    tmp_path: Path,
):
    assert candidate_store_label_paths(
        {
            "sequence_provenance": {
                "safe": {"sha256": "a" * 64},
                "nested": {"candidate_value": 9.0},
            }
        }
    ) == ["$.sequence_provenance.nested.candidate_value"]

    source = _source(201)
    candidate = _substitution(source, 5, "U")
    with pytest.raises(AssertionError, match="nested label-bearing"):
        build_dataset_rows(
            "GSE200304",
            [
                {
                    "row_id": "pair",
                    "source_sequence": source,
                    "candidate_sequence": candidate,
                    "endpoint": "stability",
                    "source_value": 1.0,
                    "candidate_value": 2.0,
                }
            ],
            provenance={
                **_provenance(tmp_path, "nested.tsv"),
                "nested": {"label": {"delta": 1.0}},
            },
            fixture_mode=True,
        )


def test_fixture_validation_is_structurally_green_but_never_a_scientific_gate(
    tmp_path: Path,
):
    stage_root = tmp_path / "D1"
    for dataset_id, length in (
        ("GSE114002", 50),
        ("GSE200304", 201),
        ("GSE246381", 75),
        ("GSE217518", 115),
    ):
        source = _source(length)
        candidate = _substitution(source, 3, "A")
        if dataset_id == "GSE114002":
            rows = [
                {
                    "row_id": "a",
                    "library": "snv",
                    "mother": source,
                    "utr": source,
                    "rl": 1,
                },
                {
                    "row_id": "v",
                    "library": "snv",
                    "mother": source,
                    "utr": candidate,
                    "rl": 2,
                },
            ]
        elif dataset_id == "GSE200304":
            rows = [
                {
                    "row_id": "v",
                    "source_sequence": source,
                    "candidate_sequence": candidate,
                    "endpoint": "mrna_stability",
                    "source_value": 1,
                    "candidate_value": 2,
                }
            ]
        elif dataset_id == "GSE246381":
            rows = [
                {
                    "SeqID": "Variant;GENE;Family=F",
                    "RefSequence": source,
                    "AltSequence": candidate,
                }
            ]
        else:
            suffix = GSE217518_U3_SUFFIX_DNA.replace("T", "U")
            rows = [
                {
                    "seqName": "3M_fixture(GENE)_utr3_Ref_1",
                    "sequence": source + suffix,
                    "halfLife": 1,
                    "region": "three_utr",
                },
                {
                    "seqName": "3M_fixture(GENE)_utr3_Mut_1",
                    "sequence": candidate + suffix,
                    "halfLife": 2,
                    "region": "three_utr",
                },
            ]
        result = build_dataset_rows(
            dataset_id,
            rows,
            provenance=_provenance(tmp_path, f"{dataset_id}.fixture"),
            fixture_mode=True,
        )
        write_dataset_result(result, stage_root)
    for dataset_id in sorted(
        D1_SCOPE_DATASETS - {"GSE114002", "GSE200304", "GSE246381", "GSE217518"}
    ):
        result = build_dataset_rows(
            dataset_id,
            (),
            provenance=_provenance(tmp_path, f"{dataset_id}.blocked"),
            fixture_mode=True,
        )
        write_dataset_result(result, stage_root)

    validation = validate_d1_root(
        stage_root,
        fixture_mode=True,
        require_global_stores=False,
    )
    assert validation["structural_validation_passed"] is True
    assert validation["phase_gate_passed"] is False
    assert validation["evidence_level"] == "fixture_only"


def test_full_build_emits_all_five_required_d1_artifacts(tmp_path: Path):
    source_50 = _source(50)
    candidate_50 = _substitution(source_50, 3, "A")
    source_201 = _source(201)
    candidate_201 = _substitution(source_201, 5, "U")
    source_75 = _source(75)
    candidate_75 = _substitution(source_75, 7, "A")
    source_115 = _source(115)
    candidate_115 = _substitution(source_115, 11, "A")
    suffix = GSE217518_U3_SUFFIX_DNA.replace("T", "U")

    payloads = {
        "GSE114002": [
            {
                "row_id": "anchor",
                "library": "snv",
                "mother": source_50,
                "utr": source_50,
                "rl": 1,
            },
            {
                "row_id": "variant",
                "library": "snv",
                "mother": source_50,
                "utr": candidate_50,
                "rl": 2,
            },
        ],
        "GSE200304": [
            {
                "row_id": "pair",
                "source_sequence": source_201,
                "candidate_sequence": candidate_201,
                "endpoint": "stability",
                "source_value": 1,
                "candidate_value": 2,
            }
        ],
        "GSE246381": [
            {
                "SeqID": "Variant;GENE;Family=F",
                "RefSequence": source_75,
                "AltSequence": candidate_75,
            }
        ],
        "GSE217518": [
            {
                "seqName": "3M_fixture(GENE)_utr3_Ref_1",
                "sequence": source_115 + suffix,
                "halfLife": 1,
                "region": "three_utr",
            },
            {
                "seqName": "3M_fixture(GENE)_utr3_Mut_1",
                "sequence": candidate_115 + suffix,
                "halfLife": 2,
                "region": "three_utr",
            },
        ],
    }
    datasets = []
    for dataset_id, rows in payloads.items():
        path = tmp_path / f"{dataset_id}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        datasets.append(
            {
                "dataset_id": dataset_id,
                "download_manifest": "fixture-manifest",
                "license": "fixture-only",
                "input_files": [{"path": str(path), "format": "jsonl"}],
            }
        )
    for dataset_id in sorted(D1_SCOPE_DATASETS - set(payloads)):
        datasets.append(
            {
                "dataset_id": dataset_id,
                "download_manifest": "not-opened",
                "license": "not-opened",
                "input_files": [],
            }
        )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "d1_build_config_v2",
                "stage_id": "D1_B0_20260728T160012Z_8862125",
                "selection_policy": _strict_fixture_selection_policy(),
                "datasets": datasets,
            }
        ),
        encoding="utf-8",
    )
    artifact_root = tmp_path / "contract-artifacts"
    snapshot_root = tmp_path / "snapshot"
    manifest = build_snapshot(
        config,
        snapshot_root,
        fixture_mode=True,
        artifact_root=artifact_root,
    )
    frozen_config_bytes = config.read_bytes()
    assert manifest["config_path"] == str(config.resolve())
    assert manifest["config_bytes"] == len(frozen_config_bytes)
    assert manifest["config_sha256"] == hashlib.sha256(frozen_config_bytes).hexdigest()
    expected = {
        "data/data_exposure_ledger.jsonl",
        "data/library_ascertainment_report.json",
        "data/edit_script_ambiguity_report.json",
        "data/measured_action_coverage_report.json",
        "reports/data_reproduction/summary.csv",
    }
    assert set(manifest["required_artifacts"]) == expected
    assert all((artifact_root / path).is_file() for path in expected)
    ambiguity = json.loads(
        (artifact_root / "data/edit_script_ambiguity_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert ambiguity["count_scope"] == ["minimum_cost_character_alignments"]
    assert ambiguity["constructed_paths_marked_observed"] == 0

    global_stores = manifest["global_stores"]
    label_meta = global_stores["canonical_label_store"]
    candidate_meta = global_stores["sealed_label_free_candidate_store"]
    label_path = snapshot_root / label_meta["path"]
    candidate_path = snapshot_root / candidate_meta["path"]
    labels = [
        json.loads(line)
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    candidates = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    label_ids = [record["record_id"] for record in labels]
    candidate_ids = [record["record_id"] for record in candidates]
    assert label_path.is_file()
    assert candidate_path.is_file()
    assert label_path.resolve() != candidate_path.resolve()
    assert label_ids == candidate_ids
    assert len(label_ids) == len(set(label_ids))
    assert all(not candidate_store_label_paths(record) for record in candidates)
    assert label_meta["records"] == len(labels)
    assert candidate_meta["records"] == len(candidates)
    assert label_meta["record_ids_sha256"] == candidate_meta["record_ids_sha256"]
    assert global_stores["blocked_dataset_records"] == 0

    validation = validate_d1_root(
        snapshot_root,
        fixture_mode=True,
        artifact_root=artifact_root,
    )
    assert validation["structural_validation_passed"] is True
    assert validation["phase_gate_passed"] is False
    assert validation["global_store_validation"]["passed"] is True
    assert validation["config_binding_validation"]["passed"] is True
    assert validation["required_artifact_validation"]["passed"] is True
    library = json.loads(
        (artifact_root / "data/library_ascertainment_report.json").read_text(
            encoding="utf-8"
        )
    )
    required_library_fields = set(library["required_dataset_fields"])
    required_library_audits = set(library["required_executed_audits"])
    assert set(library["datasets"]) == D1_SCOPE_DATASETS
    for entry in library["datasets"].values():
        assert required_library_fields <= set(entry)
        assert required_library_audits <= set(entry["executed_audits"])
        assert entry["claim_scope"] == "descriptive_ascertainment_only"
        assert entry["biological_desirability_claimed"] is False

    library_path = artifact_root / "data/library_ascertainment_report.json"
    frozen_library_bytes = library_path.read_bytes()
    build_manifest_path = snapshot_root / "build_manifest.json"
    frozen_build_manifest_bytes = build_manifest_path.read_bytes()
    replacement = {
        "schema_version": "d1_library_ascertainment_v2",
        "datasets": {dataset_id: {} for dataset_id in D1_SCOPE_DATASETS},
    }
    library_path.write_text(
        json.dumps(replacement, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refreshed_manifest = json.loads(frozen_build_manifest_bytes.decode("utf-8"))
    refreshed_manifest["required_artifacts"][
        "data/library_ascertainment_report.json"
    ].update(
        {
            "bytes": library_path.stat().st_size,
            "sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        }
    )
    build_manifest_path.write_text(
        json.dumps(refreshed_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replacement_validation = validate_d1_root(
        snapshot_root,
        fixture_mode=True,
        artifact_root=artifact_root,
    )
    replacement_artifacts = replacement_validation["required_artifact_validation"]
    assert (
        replacement_artifacts["binding_checks"][
            "data/library_ascertainment_report.json"
        ]
        is True
    )
    assert (
        replacement_artifacts["content_checks"][
            "data/library_ascertainment_report.json"
        ]
        is False
    )
    assert replacement_artifacts["passed"] is False
    assert replacement_validation["phase_gate_passed"] is False
    library_path.write_bytes(frozen_library_bytes)
    build_manifest_path.write_bytes(frozen_build_manifest_bytes)

    config.write_bytes(frozen_config_bytes + b"\n")
    stale_config_validation = validate_d1_root(
        snapshot_root,
        fixture_mode=True,
        artifact_root=artifact_root,
    )
    config_checks = stale_config_validation["config_binding_validation"]["checks"]
    assert stale_config_validation["structural_validation_passed"] is False
    assert config_checks["config_size_matches"] is False
    assert config_checks["config_sha256_matches"] is False
    config.write_bytes(frozen_config_bytes)

    tampered_candidates = [dict(record) for record in candidates]
    tampered_candidates[0]["nested"] = {"candidate_value": 99.0}
    candidate_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in tampered_candidates
        ),
        encoding="utf-8",
    )
    tampered_validation = validate_d1_root(
        snapshot_root,
        fixture_mode=True,
        artifact_root=artifact_root,
    )
    global_checks = tampered_validation["global_store_validation"]["checks"]
    assert tampered_validation["structural_validation_passed"] is False
    assert global_checks["candidate_store_integrity"] is False
    assert global_checks["candidate_store_recursively_label_free"] is False


def test_d1_config_rejects_scope_or_selection_semantic_drift(tmp_path: Path):
    datasets = [
        {
            "dataset_id": dataset_id,
            "download_manifest": "fixture",
            "license": "fixture",
            "input_files": [],
        }
        for dataset_id in sorted(D1_SCOPE_DATASETS)
    ]
    base = {
        "schema_version": "d1_build_config_v2",
        "stage_id": "D1_B0_20260728T160012Z_8862125",
        "selection_policy": _strict_fixture_selection_policy(),
        "datasets": datasets,
    }
    assert _parse_config(json.dumps(base).encode("utf-8"))["datasets"]

    missing = json.loads(json.dumps(base))
    missing["datasets"].pop()
    with pytest.raises(ValueError, match="exact-12"):
        _parse_config(json.dumps(missing).encode("utf-8"))

    label_selected = json.loads(json.dumps(base))
    label_selected["selection_policy"][
        "candidate_final_labels_used_for_dataset_role_selection"
    ] = True
    with pytest.raises(ValueError, match="final labels"):
        _parse_config(json.dumps(label_selected).encode("utf-8"))

    scope_text = (ROOT / "data_registry/d1_dataset_scope_manifest.yaml").read_text(
        encoding="utf-8"
    )
    drifted_scope_path = tmp_path / "drifted-scope.yaml"
    drifted_scope_path.write_text(
        scope_text.replace(
            "candidate_final_labels_used_for_role_selection: false",
            "candidate_final_labels_used_for_role_selection: true",
        ),
        encoding="utf-8",
    )
    drifted = json.loads(json.dumps(base))
    drifted_bytes = drifted_scope_path.read_bytes()
    drifted["selection_policy"]["dataset_scope_manifest"] = {
        "path": str(drifted_scope_path),
        "bytes": len(drifted_bytes),
        "sha256": hashlib.sha256(drifted_bytes).hexdigest(),
    }
    with pytest.raises(ValueError, match="not label-free"):
        _parse_config(json.dumps(drifted).encode("utf-8"))


def test_d1_pipeline_scaffolds_cover_every_scope_dataset_and_active_stage():
    pipeline_root = Path(__file__).resolve().parents[1] / "data" / "d1" / "pipelines"
    active_datasets = {
        "GSE114002",
        "GSE200304",
        "GSE217518",
        "GSE246381",
    }
    stage_files = {
        "download.py",
        "extract.py",
        "paper_clean.py",
        "canonical_clean.py",
        "build_source_candidate.py",
        "build_edit_scripts.py",
        "reproduce_labels.py",
        "audit_library_design.py",
        "audit_exposure.py",
    }
    for dataset_id in D1_SCOPE_DATASETS:
        dataset_root = pipeline_root / dataset_id
        assert (dataset_root / "README.md").is_file()
        assert (dataset_root / "manifest.yaml").is_file()
        if dataset_id in active_datasets:
            assert stage_files <= {
                path.name for path in dataset_root.iterdir() if path.is_file()
            }
            for stage_file in stage_files:
                stage = Path(stage_file).stem
                contents = (dataset_root / stage_file).read_text(encoding="utf-8")
                assert f'pipeline_stage_main("{dataset_id}", "{stage}")' in contents
