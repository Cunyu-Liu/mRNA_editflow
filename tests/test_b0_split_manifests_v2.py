from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from data.utr_benchmark_v2.edit_script import canonical_edit_script
from data.utr_benchmark_v2.split_graph import MissingGroupingMetadataError
from data.utr_benchmark_v2.split_graph import SplitGraphError
from data.utr_benchmark_v2.split_graph import build_atomic_components
from data.utr_benchmark_v2.split_graph import build_split_manifest
from data.utr_benchmark_v2.split_graph import expected_partition_ids
from data.utr_benchmark_v2.split_graph import partition_sha256
from scripts.data.build_b0_splits import _load_d1_acceptance_binding
from scripts.data.build_b0_splits import _frozen_ambiguity_binding_audit
from scripts.data.build_b0_splits import _projection_comparison
from scripts.data.build_b0_splits import load_structural_jsonl


def _code(index: int, width: int = 6) -> str:
    alphabet = "ACGU"
    value = index
    bases = []
    for _ in range(width):
        bases.append(alphabet[value % 4])
        value //= 4
    return "".join(reversed(bases))


def _record(
    index: int,
    *,
    region: str = "five_utr",
    dataset_id: str | None = None,
    study_group: str | None = None,
    source: str | None = None,
    candidate: str | None = None,
    source_group: str | None = None,
) -> dict:
    dataset = dataset_id or f"dataset-{index}"
    expanded_code = "".join(base * 6 for base in _code(index))
    source_sequence = source or (expanded_code + "AAAAAA")
    candidate_sequence = candidate or (expanded_code + "AAAAAC")
    actions = canonical_edit_script(source_sequence, candidate_sequence)
    study = study_group or dataset
    return {
        "record_id": f"record-{index}",
        "candidate_id": f"candidate-{index}",
        "dataset_id": dataset,
        "study_id": study,
        "assay_id": f"assay:{dataset}",
        "region": region,
        "source_id": f"source-{index}",
        "source_sequence": source_sequence,
        "candidate_sequence": candidate_sequence,
        "edit_script": [action.to_dict() for action in actions],
        "edit_distance": len(actions),
        "intermediate_sequences": [],
        "trajectory_observed": False,
        "source_group": source_group or f"source-group-{index}",
        "study_group": study,
        "sequence_cluster": f"cluster-{index}",
        "scaffold_group": f"scaffold-{index}",
        "gene_group": f"gene-{index}",
        "context_group": f"context-{index}",
        "barcode_batch": f"barcode-{index}",
        "library_batch": f"library-{index}",
        "quality_flags": [],
        # These fields must never influence grouping or assignment.
        "source_value_raw": float(index),
        "candidate_value_raw": float(index + 1),
        "delta_normalized": 1.0,
    }


def _cross_region_records() -> list[dict]:
    return [
        _record(0, dataset_id="GSE217518", region="five_utr"),
        _record(1, dataset_id="GSE217518", region="five_utr"),
        _record(2, dataset_id="GSE217518", region="three_utr"),
        _record(3, dataset_id="GSE217518", region="three_utr"),
        _record(4, dataset_id="GSE114002", region="five_utr"),
        _record(5, dataset_id="GSE114002", region="five_utr"),
        _record(6, dataset_id="GSE200304", region="three_utr"),
    ]


def test_source_assignment_is_label_independent_deterministic_and_hashed() -> None:
    records = [_record(index) for index in range(9)]
    first = build_split_manifest(
        records, region="five_utr", split_kind="source_disjoint"
    )
    changed_labels = copy.deepcopy(records)
    for record in changed_labels:
        record["source_value_raw"] = -9999.0
        record["candidate_value_raw"] = 9999.0
        record["delta_normalized"] = None
    second = build_split_manifest(
        list(reversed(changed_labels)),
        region="five_utr",
        split_kind="source_disjoint",
    )

    assert first["status"] == "READY"
    assert first["required_partition_ids"] == list(
        expected_partition_ids("source_disjoint", region="five_utr")
    )
    assert len(first["partitions"]) == 7
    assert {
        partition["independent_group_dimension"] for partition in first["partitions"]
    } == {
        "source_state",
        "sequence_cluster",
        "scaffold_group",
        "gene_group",
        "context_group",
        "barcode_batch",
        "library_batch",
    }
    partition = first["partitions"][0]
    assert partition["label_free_assignment"] is True
    assert partition["algorithm"]["uses_randomness"] is False
    assert (
        partition["algorithm"]["state_closure_scope"]
        == "frozen_d1_canonical_edit_script_prefixes_and_declared_intermediates"
    )
    assert partition["partition_sha256"] == partition_sha256(partition)
    assert partition["near_neighbor_binding"]["edit_distance_threshold"] == 5
    assert partition["near_neighbor_binding"]["candidate_generation_complete"] is True
    assert partition["component_roles"] == second["partitions"][0]["component_roles"]
    assert partition["roles"] == second["partitions"][0]["roles"]


def test_each_source_axis_is_independent_and_near_neighbors_never_split() -> None:
    records = [_record(index) for index in range(9)]
    source = records[0]["source_sequence"]
    near_source = "C" * 5 + source[5:]
    records[8]["source_sequence"] = near_source
    records[8]["candidate_sequence"] = near_source[:-1] + (
        "A" if near_source[-1] != "A" else "C"
    )
    records[8]["gene_group"] = records[0]["gene_group"]

    manifest = build_split_manifest(
        records, region="five_utr", split_kind="source_disjoint"
    )
    assert manifest["status"] == "READY"
    for partition in manifest["partitions"]:
        role_by_record = {
            record_id: role
            for role, record_ids in partition["roles"].items()
            for record_id in record_ids
        }
        assert role_by_record["record-0"] == role_by_record["record-8"]

    gene_partition = next(
        partition
        for partition in manifest["partitions"]
        if partition["independent_group_dimension"] == "gene_group"
    )
    assert "gene_group" in gene_partition["required_disjoint_axes"]


def test_atomic_component_uses_frozen_canonical_replay_states_not_metadata() -> None:
    ambiguous = _record(0, source="AC", candidate="CA")
    bridge = _record(1, source="CC", candidate="CU")
    independent = _record(2, source="GG", candidate="GU")
    # "CC" is a prefix state of D1's deterministic canonical AC -> CA script.
    components = build_atomic_components([ambiguous, bridge, independent])
    membership = {
        record_id: component.component_id
        for component in components
        for record_id in component.record_ids
    }
    assert membership["record-0"] == membership["record-1"]
    assert membership["record-2"] != membership["record-0"]
    assert all(
        component.ambiguity_scope
        == "frozen_d1_canonical_edit_script_prefixes_and_declared_intermediates"
        for component in components
    )

    # Shared metadata is audited later and must not redefine atomic state.
    metadata_only = [_record(10), _record(11)]
    for field in (
        "study_group",
        "sequence_cluster",
        "scaffold_group",
        "gene_group",
        "context_group",
        "barcode_batch",
        "library_batch",
    ):
        metadata_only[1][field] = metadata_only[0][field]
    assert len(build_atomic_components(metadata_only)) == 2


def test_constructed_intermediate_cannot_be_claimed_observed() -> None:
    record = _record(0)
    record["intermediate_sequences"] = [{"sequence": "ACGU", "observed": True}]
    with pytest.raises(
        SplitGraphError, match="constructed intermediate cannot be marked observed"
    ):
        build_atomic_components([record])


def test_missing_or_unknown_group_metadata_fails_closed() -> None:
    record = _record(0)
    del record["library_batch"]
    with pytest.raises(MissingGroupingMetadataError, match="library_batch"):
        build_atomic_components([record])

    record = _record(0)
    record["gene_group"] = "UNKNOWN"
    with pytest.raises(MissingGroupingMetadataError, match="gene_group"):
        build_atomic_components([record])


def test_study_disjoint_is_label_independent_loso_with_full_test_study() -> None:
    records = [
        _record(0, dataset_id="GSE114002"),
        _record(1, dataset_id="GSE114002"),
        _record(2, dataset_id="GSE217518"),
        _record(3, dataset_id="GSE217518"),
    ]
    retrospective = _record(50, dataset_id="GSE246381")
    retrospective.update(
        {
            "canonical_split": "retrospective_only",
            "paper_split": "retrospective_only",
            "quality_flags": ["NO_TRAINING_OR_SELECTION"],
        }
    )
    absolute = _record(51, dataset_id="GSE114002")
    absolute.update(
        {
            "pair_type": "absolute_property_only",
            "canonical_split": "absolute_prior_only",
            "source_id": None,
            "source_sequence": None,
            "edit_script": None,
            "edit_distance": None,
        }
    )
    manifest = build_split_manifest(
        records + [retrospective, absolute],
        region="five_utr",
        split_kind="study_disjoint",
    )

    assert manifest["status"] == "READY"
    assert set(manifest["required_partition_ids"]) == {
        "loso:GSE114002",
        "loso:GSE217518",
    }
    assert manifest["partitions"] == manifest["folds"]
    all_excluded = {
        row["record_id"]: row["reason"] for row in manifest["excluded_records"]
    }
    assert all_excluded["record-50"].startswith("retrospective_dataset")
    assert all_excluded["record-51"] == "non_intervention_pair_type"
    for fold in manifest["folds"]:
        assert fold["partition_sha256"] == partition_sha256(fold)
        heldout = fold["heldout_study"]
        test_ids = set(fold["roles"]["test"])
        expected_test_ids = {
            record["record_id"]
            for record in records
            if record["study_group"] == heldout
        }
        assert test_ids == expected_test_ids
        assert fold["selection_policy"]["outer_test_labels_read_for_selection"] is False
        assert fold["disjoint_scope"] == "test_vs_development"
        assert "descriptive" in fold["claim_boundary"]
        assert "record-50" not in {
            record_id for role in fold["roles"].values() for record_id in role
        }


def test_insufficient_studies_block_without_random_fallback() -> None:
    records = [
        _record(index, dataset_id="only-study", study_group="only-study")
        for index in range(6)
    ]
    manifest = build_split_manifest(
        records, region="five_utr", split_kind="study_disjoint"
    )
    assert manifest["status"] == "BLOCKED"
    assert any(
        reason.startswith("required_LOSO_studies_missing:")
        for reason in manifest["blocked_reasons"]
    )
    assert any(
        reason.startswith("unexpected_LOSO_studies_present:")
        for reason in manifest["blocked_reasons"]
    )
    assert len(manifest["partitions"]) == 2
    assert {partition["partition_id"] for partition in manifest["partitions"]} == set(
        manifest["required_partition_ids"]
    )
    assert all(partition["status"] == "BLOCKED" for partition in manifest["partitions"])
    assert manifest["algorithm"]["uses_randomness"] is False


def test_cross_region_manifest_preserves_all_required_strata_and_boundaries() -> None:
    manifest = build_split_manifest(
        _cross_region_records(),
        region=None,
        split_kind="cross_region_transfer",
        source_region="five_utr",
        target_region="three_utr",
    )
    assert manifest["status"] == "READY"
    assert set(manifest["required_partition_ids"]) == {
        "within_study:GSE217518:five_to_three",
        "within_study:GSE217518:three_to_five",
        "cross_study:GSE114002_five_to_GSE200304_three",
    }
    assert manifest["partitions"] == manifest["strata"]
    assert len(manifest["strata"]) == 3
    for stratum in manifest["strata"]:
        assert stratum["status"] == "READY"
        assert stratum["partition_sha256"] == partition_sha256(stratum)
        assert (
            stratum["selection_policy"]["outer_test_labels_read_for_selection"] is False
        )
        assert (
            stratum["aggregation_policy"]
            == "report_stratum_separately; do_not_pool_as_region_effect"
        )
    cross_study = next(
        stratum for stratum in manifest["strata"] if stratum["within_study"] is False
    )
    assert cross_study["required_metrics"] == ["endpoint_agnostic_generative_metrics"]
    assert cross_study["confounding_disclosure"] == [
        "study",
        "assay",
        "region",
    ]
    assert cross_study["estimand_required_overlap_axes"] == [
        "source_side_study_group",
        "source_side_assay_group",
        "source_side_region_group",
    ]
    assert "isolated region effect" in cross_study["claim_boundary"]


def test_cross_region_keeps_a_failed_required_stratum_and_blocks_outer() -> None:
    records = _cross_region_records()
    # A shared state joins the required within-study 5 -> 3 source/test sides.
    records[2]["source_sequence"] = records[0]["candidate_sequence"]
    actions = canonical_edit_script(
        records[2]["source_sequence"], records[2]["candidate_sequence"]
    )
    records[2]["edit_script"] = [action.to_dict() for action in actions]
    records[2]["edit_distance"] = len(actions)
    manifest = build_split_manifest(
        records,
        region=None,
        split_kind="cross_region_transfer",
        source_region="five_utr",
        target_region="three_utr",
    )
    assert manifest["status"] == "BLOCKED"
    assert len(manifest["strata"]) == 3
    failed = next(
        stratum
        for stratum in manifest["strata"]
        if stratum["partition_id"] == "within_study:GSE217518:five_to_three"
    )
    assert failed["status"] == "BLOCKED"
    assert "state_component_spans_source_and_target" in failed["blocked_reasons"]
    assert set(manifest["required_partition_ids"]) == {
        stratum["partition_id"] for stratum in manifest["strata"]
    }


def test_cross_region_forbidden_context_overlap_blocks_without_dropping_strata() -> (
    None
):
    records = _cross_region_records()
    records[2]["context_group"] = records[0]["context_group"]
    manifest = build_split_manifest(
        records,
        region=None,
        split_kind="cross_region_transfer",
        source_region="five_utr",
        target_region="three_utr",
    )
    assert manifest["status"] == "BLOCKED"
    assert len(manifest["strata"]) == 3
    assert {stratum["partition_id"] for stratum in manifest["strata"]} == set(
        manifest["required_partition_ids"]
    )
    assert any(
        "dense_graph_firewall_spans_source_and_target" in stratum["blocked_reasons"]
        for stratum in manifest["strata"]
    )


def test_d1_projection_detects_source_or_candidate_tamper_with_same_id() -> None:
    canonical = _record(0)
    canonical["source_value_raw"] = 1.5
    structural = {
        key: value
        for key, value in canonical.items()
        if key
        not in {
            "source_value_raw",
            "candidate_value_raw",
            "delta_normalized",
        }
    }
    assert _projection_comparison([canonical], [structural])["passed"] is True
    for field, replacement in (
        ("source_sequence", "UUUUUUAA"),
        ("candidate_sequence", "UUUUUUAC"),
    ):
        tampered = copy.deepcopy(structural)
        tampered[field] = replacement
        comparison = _projection_comparison([canonical], [tampered])
        assert comparison["passed"] is False
        assert comparison["mismatched_record_ids"] == ["record-0"]


def test_b0_binds_frozen_d1_ambiguity_without_state_reenumeration() -> None:
    record = _record(0, source="A" * 600, candidate="C" * 600)
    report = {
        "schema_version": "d1_edit_script_ambiguity_v2",
        "count_scope": ["minimum_cost_character_alignments"],
        "constructed_paths_marked_observed": 0,
        "datasets": {
            "dataset-0": {
                "records": 1,
                "records_with_quantified_ambiguity": 1,
                "ambiguous_records": 0,
                "max_equivalent_minimal_script_count": 1,
                "constructed_paths_marked_observed": 0,
                "count_scopes": ["minimum_cost_character_alignments"],
            }
        },
    }
    audit = _frozen_ambiguity_binding_audit([record], report)
    assert audit["passed"] is True
    assert audit["audit_mode"] == "frozen_d1_binding_no_b0_state_reenumeration"


def test_structural_loader_recursively_rejects_nested_label_escape(
    tmp_path: Path,
) -> None:
    record = _record(0)
    record.pop("source_value_raw")
    record.pop("candidate_value_raw")
    record.pop("delta_normalized")
    record["sequence_provenance"] = {"nested": {"candidate_value": 9.0}}
    path = tmp_path / "structural.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="label"):
        load_structural_jsonl(path)


def test_nonpassing_d1_phase_gate_cannot_form_a_binding(
    tmp_path: Path,
) -> None:
    canonical_path = tmp_path / "canonical.jsonl"
    canonical_path.write_text("{}\n", encoding="utf-8")
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "phase_gate_passed": False,
                "fixture_mode": False,
                "structural_validation_passed": True,
                "global_store_validation": {"passed": True},
                "required_artifact_validation": {"passed": True},
                "stage_d1_root": str(stage_root),
            }
        ),
        encoding="utf-8",
    )
    binding = _load_d1_acceptance_binding(canonical_path, acceptance_path)
    assert binding["passed"] is False
    assert binding["checks"]["phase_gate_passed"] is False
