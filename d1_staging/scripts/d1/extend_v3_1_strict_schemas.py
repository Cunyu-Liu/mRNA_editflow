#!/usr/bin/env python3
"""Extend active v3.1 schemas to the D1 contract's strict row fields.

The repository's C3 schemas were definitionally valid but still represented the
older compact D1 rows.  This migration preserves every pre-migration byte in a
separate archive, adds the contract-required provenance/lifecycle fields, and
regenerates the existing 21-schema manifest/checksum ledger.  It does not add a
22nd schema or change task/split frozen sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def st(enum=None, nullable=False):
    out = {"type": ["string", "null"] if nullable else "string"}
    if enum is not None:
        out["enum"] = enum
    return out


def num(nullable=False):
    return {"type": ["number", "null"] if nullable else "number"}


def integer(nullable=False):
    return {"type": ["integer", "null"] if nullable else "integer"}


def boolean(nullable=False):
    return {"type": ["boolean", "null"] if nullable else "boolean"}


def array(item=None, nullable=False):
    out = {"type": ["array", "null"] if nullable else "array"}
    if item is not None:
        out["items"] = item
    return out


def obj(nullable=False):
    return {"type": ["object", "null"] if nullable else "object"}


def extend(doc, props, required):
    doc.setdefault("properties", {}).update(props)
    doc["required"] = sorted(set(doc.get("required", [])) | set(required))


def extend_sequence(path: Path):
    d = json.loads(path.read_text())
    d["properties"]["sequence_scope"]["enum"] = sorted(set(d["properties"]["sequence_scope"]["enum"] + [
        "FULL_UTR", "UTR_WINDOW", "RANDOM_INSERT", "OLIGO_CONSTRUCT", "CDS_AUXILIARY",
    ]))
    d["properties"]["region_scope"]["enum"] = sorted(set(d["properties"]["region_scope"]["enum"] + [
        "CDS_AUX", "FULL_LENGTH_AUX",
    ]))
    extend(d, {
        "primary_asset_id": st(),
        "contributing_asset_ids": array(st()),
        "contributing_source_file_sha256s": array(st()),
        "contributor_set_sha256": st(),
        "sequence_reconstruction_rule_id": st(),
        "sequence_reconstruction_rule_sha256": st(),
        "source_record_id": st(),
        "source_row_locator": st(nullable=True),
        "normalized_sequence": st(nullable=True),
        "normalization_steps": array(st()),
        "alphabet_status": st(["EXACT_ACGT", "IUPAC_AMBIGUOUS", "INVALID"]),
        "model_sequence_eligible": boolean(),
        "invalid_symbol_status": st(["PASS", "QUARANTINED"]),
        "region": st(["5UTR", "3UTR", "CDS_AUX", "FULL_LENGTH_AUX"]),
        "species": st(),
        "reference_build": st(),
        "transcript_release": st(),
        "strand": st(["+", "-", "UNKNOWN", "NOT_APPLICABLE"]),
    }, [
        "primary_asset_id", "contributing_asset_ids", "contributing_source_file_sha256s",
        "contributor_set_sha256", "sequence_reconstruction_rule_id",
        "sequence_reconstruction_rule_sha256", "source_record_id", "source_row_locator",
        "raw_sequence_sha256", "normalized_sequence_sha256", "normalization_steps",
        "alphabet_status", "model_sequence_eligible", "invalid_symbol_status", "region",
        "sequence_scope", "species", "reference_build", "transcript_release", "strand",
        "original_length", "full_sequence_sha256",
    ])
    return d


def extend_functional(path: Path):
    d = json.loads(path.read_text())
    strict_observation = {
        "observation_candidate_id": st(),
        "canonical_status": st(["ACCEPTED", "SUPERSEDED", "REJECTED"]),
        "primary_label_asset_id": st(),
        "contributing_asset_ids": array(st()),
        "contributing_source_file_sha256s": array(st()),
        "contributor_set_sha256": st(),
        "parent_observation_id": st(nullable=True),
        "scientific_track": st(["E", "F", "AUX"]),
        "observation_role": st([
            "F_FUNCTION_LABEL", "E_SOURCE_MEASUREMENT", "E_CANDIDATE_MEASUREMENT",
            "E_NOEDIT_MEASUREMENT", "AUX_QC",
        ]),
        "primary_label_asset_id": st(),
        "source_file_sha256": st(),
        "source_record_id": st(),
        "source_row_locator": st(nullable=True),
        "raw_value": num(nullable=True),
        "normalized_value": num(nullable=True),
        "label_status": st(["OBSERVED", "DERIVED", "MISSING", "BELOW_COVERAGE", "QUARANTINED"]),
        "label_unit": st(),
        "label_transform": st(),
        "cell_context": st(nullable=True),
        "assay_context": st(),
        "promoter": st(nullable=True),
        "reporter_or_cargo": st(nullable=True),
        "rna_chemistry": st(nullable=True),
        "timepoint": st(nullable=True),
        "source_replicate_label": st(nullable=True),
        "sample_id": st(nullable=True),
        "biological_replicate_id": st(nullable=True),
        "technical_replicate_id": st(nullable=True),
        "barcode_id": st(nullable=True),
        "coverage_or_umi": num(nullable=True),
        "standard_error": num(nullable=True),
        "missingness_reason": st(nullable=True),
        "quality_flags": array(st()),
        "value": num(nullable=True),
        "unit": st(),
        "replicate": st(nullable=True),
    }
    extend(d, strict_observation, [
        "observation_id", "observation_candidate_id", "canonical_status", "sequence_id",
        "primary_label_asset_id", "contributing_asset_ids", "contributing_source_file_sha256s",
        "contributor_set_sha256", "source_file_sha256", "source_record_id", "source_row_locator",
        "scientific_track", "observation_role", "context_id", "endpoint_id", "raw_value",
        "normalized_value", "label_status", "label_unit", "label_transform",
        "source_replicate_label", "sample_id", "biological_replicate_id",
        "technical_replicate_id", "barcode_id", "cell_context", "assay_context", "promoter",
        "reporter_or_cargo", "rna_chemistry", "timepoint", "coverage_or_umi", "standard_error",
        "missingness_reason", "quality_flags", "value", "unit",
    ])

    candidate = d["$defs"]["FunctionalObservationCandidate"]
    extend(candidate, {
        "observation_candidate_id": st(),
        "asset_ids": array(st()),
        "contributing_source_file_sha256s": array(st()),
        "contributor_set_sha256": st(),
        "source_unit_ids": array(st()),
        "join_method_id": st(),
        "join_method_sha256": st(),
        "observation_acceptance_status": st(["CANDIDATE", "ACCEPTED", "AMBIGUOUS", "UNMATCHED", "REJECTED"]),
        "accepted_observation_id": st(nullable=True),
        "terminal_disposition_reason": st(nullable=True),
        "source_row_locators": array(st()),
        "evidence_ids": array(st()),
        "parent_candidate_id": st(nullable=True),
        "candidate_id": st(),
        "source": st(),
        "source_file_sha256": st(),
        "value": num(nullable=True),
        "lifecycle_status": st(["CANDIDATE", "ACCEPTED", "REJECTED"]),
    }, [
        "observation_candidate_id", "asset_ids", "contributing_source_file_sha256s",
        "contributor_set_sha256", "source_unit_ids", "sequence_id", "context_id", "endpoint_id",
        "join_method_id", "join_method_sha256", "observation_acceptance_status",
        "accepted_observation_id", "terminal_disposition_reason", "source_row_locators",
        "evidence_ids", "parent_candidate_id", "candidate_id", "source", "source_file_sha256",
        "lifecycle_status",
    ])
    endpoint = d["$defs"]["EndpointRegistryRow"]
    endpoint["properties"]["scaling"]["enum"] = sorted(set(endpoint["properties"]["scaling"]["enum"] + ["LOG2", "IDENTITY"]))
    extend(endpoint, {
        "biological_quantity": st(),
        "raw_field_mappings": obj(),
        "label_unit": st(),
        "directionality": st(["HIGHER_IS_BETTER", "LOWER_IS_BETTER", "TWO_SIDED", "NOT_APPLICABLE"]),
        "label_transform": st(),
        "comparability_scope": st(),
        "aggregation_rule_id": st(),
        "aggregation_rule_sha256": st(),
        "delta_rule_id": st(),
        "delta_rule_sha256": st(),
        "unknown_or_ambiguous_policy": st(),
        "record_sha256": st(),
    }, [
        "biological_quantity", "raw_field_mappings", "label_unit", "directionality",
        "label_transform", "comparability_scope", "aggregation_rule_id", "aggregation_rule_sha256",
        "delta_rule_id", "delta_rule_sha256", "unknown_or_ambiguous_policy", "record_sha256",
    ])
    return d


def extend_relation_candidate(path: Path):
    d = json.loads(path.read_text())
    extend(d, {
        "relation_candidate_id": st(),
        "parent_relation_candidate_id": st(nullable=True),
        "design_relation_group_id": st(),
        "contributing_asset_ids": array(st()),
        "contributing_source_file_sha256s": array(st()),
        "contributor_set_sha256": st(),
        "relation_context_key": st(),
        "context_id": st(),
        "endpoint_id": st(),
        "label_unit": st(),
        "label_transform": st(),
        "delta_rule_id": st(),
        "delta_rule_sha256": st(),
        "scientific_track": st(["E"]),
        "relation_acceptance_status": st(["CANDIDATE", "ACCEPTED", "AMBIGUOUS", "REJECTED"]),
        "relation_type": st(["EXACT_REF_ALT", "SOURCE_CANDIDATE", "NO_EDIT_CONTROL"]),
        "effect_evidence": st(["UNKNOWN", "SEQUENCE_ONLY", "CANDIDATE_ONLY", "BOTH_CROSS_CONTEXT", "BOTH_SAME_CONTEXT"]),
        "landscape_role": st(["SPARSE", "DENSE", "NOT_APPLICABLE"]),
        "future_use_role": st(),
        "pair_evidence_id": st(),
        "terminal_disposition_reason": st(nullable=True),
        "accepted_pair_id": st(nullable=True),
    }, [
        "relation_candidate_id", "parent_relation_candidate_id", "design_relation_group_id",
        "contributing_asset_ids", "contributing_source_file_sha256s", "contributor_set_sha256",
        "relation_context_key", "context_id", "endpoint_id", "label_unit", "label_transform",
        "delta_rule_id", "delta_rule_sha256", "scientific_track", "relation_acceptance_status",
        "relation_type", "effect_evidence", "landscape_role", "future_use_role",
        "source_sequence_id", "candidate_sequence_id", "pair_evidence_id", "terminal_disposition_reason",
        "accepted_pair_id", "candidate_id", "pairing_method", "evidence_id", "lifecycle_status",
    ])
    d["properties"]["source_sequence_id"] = st(nullable=True)
    d["properties"]["candidate_sequence_id"] = st(nullable=True)
    d["properties"]["lifecycle_status"]["enum"] = ["CANDIDATE", "ACCEPTED", "REJECTED"]
    return d


def extend_pair(path: Path):
    d = json.loads(path.read_text())
    extend(d, {
        "parent_pair_id": st(nullable=True),
        "relation_candidate_id": st(),
        "contributing_asset_ids": array(st()),
        "contributing_source_file_sha256s": array(st()),
        "contributor_set_sha256": st(),
        "context_id": st(),
        "endpoint_id": st(),
        "label_unit": st(),
        "label_transform": st(),
        "delta_rule_id": st(),
        "delta_rule_sha256": st(),
        "relation_acceptance_status": st(["ACCEPTED"]),
        "source_observation_id": st(nullable=True),
        "candidate_observation_id": st(nullable=True),
        "delta": num(nullable=True),
        "delta_standard_error": num(nullable=True),
        "same_assay_context": boolean(),
        "biological_parent_group": st(),
        "gene_group": st(nullable=True),
        "tile_family_group": st(nullable=True),
        "sequence_cluster_group": st(),
        "true_length_change": integer(),
        "minimum_edit_distance": integer(),
        "path_ambiguity_count_or_bound": {"type": ["integer", "number", "string"]},
        "pair_direction_verified": boolean(),
        "join_keys": array(st()),
        "pair_evidence_id": st(),
        "confirmatory_delta_eligible": boolean(),
        "link_view_eligible": boolean(),
        "permission_evidence_ids": array(st()),
        "exclusion_reason": st(nullable=True),
    }, [
        "parent_pair_id", "relation_candidate_id", "contributing_asset_ids",
        "contributing_source_file_sha256s", "contributor_set_sha256", "context_id", "endpoint_id",
        "label_unit", "label_transform", "delta_rule_id", "delta_rule_sha256",
        "relation_acceptance_status", "relation_type", "effect_evidence", "landscape_role",
        "future_use_role", "source_sequence_id", "candidate_sequence_id", "source_observation_id",
        "candidate_observation_id", "delta", "delta_standard_error", "same_assay_context",
        "biological_parent_group", "gene_group", "tile_family_group", "sequence_cluster_group",
        "true_length_change", "minimum_edit_distance", "path_ambiguity_count_or_bound",
        "pair_direction_verified", "pairing_method", "join_keys", "pair_evidence_id",
        "confirmatory_delta_eligible", "link_view_eligible", "permission_evidence_ids",
        "exclusion_reason", "pair_id", "candidate_id", "design_relation_group_id",
        "scientific_track", "immutable_base_future_use_role", "evidence_id",
    ])
    d["properties"]["source_sequence_id"] = st(nullable=True)
    d["properties"]["candidate_sequence_id"] = st(nullable=True)
    return d


def extend_group_registry(path: Path):
    d = json.loads(path.read_text())
    group_types = [
        "SAMPLE", "BIOLOGICAL_REPLICATE", "TECHNICAL_REPLICATE", "BARCODE", "BIOLOGICAL_PARENT",
        "GENE", "TRANSCRIPT", "TILE_FAMILY", "SEQUENCE_CLUSTER", "LIBRARY_LINEAGE",
        "EXPERIMENTAL_CONTEXT", "NO_EDIT_SAMPLING_FRAME",
    ]
    d["properties"]["grouping_atom"]["enum"] = sorted(set(d["properties"]["grouping_atom"]["enum"] + group_types))
    extend(d, {
        "group_type": st(group_types),
        "grouping_method": st(),
        "method_version": st(),
        "thresholds": obj(),
        "source_evidence": array(st()),
        "member_count": integer(),
        "ambiguous_membership": boolean(),
        "parent_group_id": st(nullable=True),
        "raw_context_values": obj(nullable=True),
        "context_components": obj(nullable=True),
        "ontology_ids": array(st()),
        "ontology_version": st(nullable=True),
        "mapping_status": st(["RESOLVED", "UNKNOWN", "AMBIGUOUS", "NOT_APPLICABLE"]),
        "mapping_rule_id": st(nullable=True),
        "mapping_rule_sha256": st(nullable=True),
    }, [
        "group_id", "group_type", "grouping_method", "method_version", "thresholds", "source_evidence",
        "member_count", "ambiguous_membership", "parent_group_id", "grouping_atom", "member_ids",
        "group_sha256",
    ])
    d["$defs"] = d.get("$defs", {})
    d["$defs"]["NoEditSamplingFrameRow"] = {
        "type": "object",
        "required": [
            "group_id", "group_type", "study_id", "asset_ids", "library_lineage_group_id",
            "sublibrary_or_design_stratum", "species", "region_scope", "context_id", "endpoint_id",
            "inclusion_mechanism", "identity_inclusion_rule_id", "identity_inclusion_rule_sha256",
            "nonidentity_inclusion_rule_id", "nonidentity_inclusion_rule_sha256",
            "inclusion_probability_status", "reweighting_rule_id", "reweighting_rule_sha256",
            "evidence_ids", "member_assignment_manifest_sha256", "frame_definition_sha256",
        ],
        "additionalProperties": False,
        "properties": {
            "group_id": st(), "group_type": st(["NO_EDIT_SAMPLING_FRAME"]),
            "study_id": st(), "asset_ids": array(st()), "library_lineage_group_id": st(),
            "sublibrary_or_design_stratum": st(), "species": st(), "region_scope": st(),
            "context_id": st(), "endpoint_id": st(), "inclusion_mechanism": st(),
            "identity_inclusion_rule_id": st(), "identity_inclusion_rule_sha256": st(),
            "nonidentity_inclusion_rule_id": st(), "nonidentity_inclusion_rule_sha256": st(),
            "inclusion_probability_status": st(["KNOWN", "RECOVERABLE", "UNKNOWN"]),
            "reweighting_rule_id": st(), "reweighting_rule_sha256": st(),
            "evidence_ids": array(st()), "member_assignment_manifest_sha256": st(),
            "frame_definition_sha256": st(),
        },
    }
    return d


def extend_group_assignment(path: Path):
    d = json.loads(path.read_text())
    extend(d, {"source_evidence_ids": array(st()), "member_locator": st(nullable=True)}, [
        "assignment_id", "object_id", "object_type", "group_id", "grouping_atom", "assignment_algorithm_id",
    ])
    d["properties"]["object_type"] = st()
    return d


def extend_exposure(path: Path):
    d = json.loads(path.read_text())
    object_types = ["SEQUENCE", "RELATION_CANDIDATE", "OBSERVATION_CANDIDATE", "PAIR", "OBSERVATION"]
    exposure = {
        "exposure_record_id": st(), "object_type": st(object_types),
        "project_sequence_analytic_exposure": st(["NONE_CONFIRMED", "PRESENT", "UNKNOWN"]),
        "project_sequence_analytic_use_types": array(st()),
        "project_label_analytic_exposure": st(["NONE_CONFIRMED", "PRESENT", "UNKNOWN"]),
        "project_label_analytic_use_types": array(st()),
        "pipeline_sequence_materialization": st(["ABSENT", "PRESENT", "UNKNOWN"]),
        "pipeline_label_materialization": st(["ABSENT", "PRESENT", "UNKNOWN"]),
        "foundation_overlap_requirement": st(["REQUIRED_FM0_A", "NOT_APPLICABLE_NO_EXTERNAL_WEIGHTS"]),
        "foundation_audit_scope_id": st(),
        "foundation_overlap_audit_status_at_baseline": st(["NOT_STARTED", "DEFERRED_TO_FM0_A", "NOT_APPLICABLE"]),
        "contributing_asset_ids": array(st()), "contributing_file_sha256s": array(st()),
        "rights_evidence_ids": array(st()), "rights_projection_rule_id": st(),
        "rights_projection_rule_sha256": st(), "rights_override_id": st(nullable=True),
        "rights_override_reviewer": st(nullable=True), "rights_override_scope": st(nullable=True),
        "rights_override_evidence_ids": array(st()), "rights_override_sha256": st(nullable=True),
        "permitted_model_training": st(["YES", "NO", "UNKNOWN"]),
        "permitted_evaluation": st(["YES", "NO", "UNKNOWN"]),
        "permitted_derived_release": st(["YES", "NO", "UNKNOWN"]),
        "permitted_raw_redistribution": st(["YES", "NO", "UNKNOWN"]),
        "evidence_ids": array(st()), "canonical_object_sha256": st(), "record_sha256": st(),
    }
    extend(d, exposure, list(exposure))
    def extend_effective(e):
        props = {
            "object_type": st(object_types), "projection_phase": st(["D1", "FM0_A", "B0_R", "G7", "MODEL_REBIND_OR_LATER"]),
            "snapshot_id": st(), "baseline_exposure_record_id": st(), "baseline_record_sha256": st(),
            "access_log_chain_root_sha256": st(), "foundation_exposure_ledger_manifest_sha256": st(),
            "as_of_event_id": st(), "effective_project_sequence_analytic_exposure": st(["NONE_CONFIRMED", "PRESENT", "UNKNOWN"]),
            "effective_project_sequence_use_types": array(st()), "effective_project_label_analytic_exposure": st(["NONE_CONFIRMED", "PRESENT", "UNKNOWN"]),
            "effective_project_label_use_types": array(st()), "final_access_status": st(["SEALED_UNOPENED", "FINAL_ACCESS_RESERVED", "FINAL_OPENED", "FINAL_INVALIDATED"]),
        }
        extend(e, props, list(props) + ["object_id", "effective_exposure", "projection_sha256", "chain_root_sha256"])
    extend_effective(d["$defs"]["EffectiveExposureProjection"])
    return d


def extend_use_role(path: Path):
    d = json.loads(path.read_text())
    strict = {
        "use_role_record_id": st(), "relation_candidate_id": st(), "pair_id": st(),
        "base_future_use_role": st(), "candidate_base_payload_sha256": st(),
        "pair_base_payload_sha256": st(), "canonical_manifest_sha256": st(), "record_sha256": st(),
    }
    extend(d, strict, list(strict))
    return d


def extend_transformation(path: Path):
    d = json.loads(path.read_text())
    strict = {
        "supersession_edge_id": st(), "object_type": st(["SEQUENCE", "RELATION_CANDIDATE", "PAIR", "OBSERVATION_CANDIDATE", "OBSERVATION"]),
        "old_object_id": st(), "new_object_id": st(), "old_object_sha256": st(), "new_object_sha256": st(),
        "reason": st(), "run_id": st(), "code_commit": st(),
    }
    extend(d, strict, list(strict) + ["edge_sha256", "config_hash"])
    edge = d["$defs"]["SupersessionEdge"]
    extend(edge, strict, list(strict) + ["edge_sha256", "config_hash"])
    proj = d["$defs"]["CurrentCanonicalObjectProjection"]
    projection = {
        "projection_record_id": st(), "run_id": st(), "canonical_snapshot_id": st(),
        "object_type": st(["SEQUENCE", "RELATION_CANDIDATE", "PAIR", "OBSERVATION_CANDIDATE", "OBSERVATION"]),
        "chain_root_object_id": st(), "chain_root_object_sha256": st(), "current_leaf_object_id": st(),
        "current_leaf_object_sha256": st(), "generation_index": integer(), "chain_length": integer(),
        "last_supersession_edge_id": st(nullable=True), "last_supersession_edge_sha256": st(nullable=True),
        "supersession_manifest_sha256": st(), "is_current_leaf_accepted": boolean(), "projection_sha256": st(),
    }
    extend(proj, projection, list(projection) + ["object_id", "active", "canonical_manifest_sha256", "record_sha256"])
    return d


def extend_rejection(path: Path):
    d = json.loads(path.read_text())
    extend(d, {
        "source_unit_id": st(nullable=True), "source_row_locator": st(nullable=True),
        "asset_ids": array(st()), "disposition_status": st(nullable=True),
        "terminal_disposition_reason": st(nullable=True),
    }, [])
    return d


def extend_reporter(path: Path):
    d = json.loads(path.read_text())
    extend(d, {
        "asset_id": st(nullable=True), "source_file_sha256": st(nullable=True),
        "assessment_status": st(nullable=True), "risk_reason": st(nullable=True),
        "evidence_ids": array(st()),
    }, [])
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-dir", type=Path, required=True)
    ap.add_argument("--archive-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--validator-files", nargs="+", type=Path, required=True)
    args = ap.parse_args()

    selected = {
        "sequence_entity.schema.json": extend_sequence,
        "functional_observation.schema.json": extend_functional,
        "utr_edit_relation_candidate.schema.json": extend_relation_candidate,
        "utr_edit_pair.schema.json": extend_pair,
        "group_registry.schema.json": extend_group_registry,
        "group_assignment.schema.json": extend_group_assignment,
        "exposure_record.schema.json": extend_exposure,
        "use_role.schema.json": extend_use_role,
        "transformation_edge.schema.json": extend_transformation,
        "rejection_record.schema.json": extend_rejection,
        "reporter_artifact_assessment.schema.json": extend_reporter,
    }
    old_schema_names = sorted(p.name for p in args.schema_dir.glob("*.schema.json"))
    archive_files = [args.schema_dir / "SCHEMA_MANIFEST.json", args.schema_dir / "SCHEMA_SHA256SUMS"]
    archive_files += [args.schema_dir / name for name in selected]
    archive_files += list(args.validator_files)
    for src in archive_files:
        rel = Path("schemas/v3_1") / src.name if src.parent == args.schema_dir else Path("contract_code") / src.name
        dest = args.archive_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    before = {str(src): sha256_file(src) for src in archive_files}
    for name, fn in selected.items():
        doc = fn(args.schema_dir / name)
        (args.schema_dir / name).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # The D1 contract explicitly requires the NoEditSamplingFrameRow $def.
    for path in args.validator_files:
        text = path.read_text(encoding="utf-8")
        marker = '    "functional_observation.schema.json": ["FunctionalObservationCandidate", "EndpointRegistryRow"],\n'
        addition = '    "group_registry.schema.json": ["NoEditSamplingFrameRow"],\n'
        if addition not in text:
            if marker not in text:
                raise RuntimeError(f"validator marker not found: {path}")
            text = text.replace(marker, addition + marker, 1)
            path.write_text(text, encoding="utf-8")

    schema_files = sorted(args.schema_dir.glob("*.schema.json"))
    old_manifest = json.loads((args.schema_dir / "SCHEMA_MANIFEST.json").read_text(encoding="utf-8"))
    entries = []
    for path in schema_files:
        entries.append({
            "$id": json.loads(path.read_text(encoding="utf-8"))["$id"],
            "contract_id": "utr_editflow_goal_v3.1_benchmark_first",
            "filename": path.name,
            "schema_version": "3.1",
            "sha256": sha256_file(path),
        })
    new_manifest = dict(old_manifest)
    new_manifest["schemas"] = entries
    (args.schema_dir / "SCHEMA_MANIFEST.json").write_text(json.dumps(new_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"{entry['sha256']}  {entry['filename']}" for entry in entries]
    (args.schema_dir / "SCHEMA_SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after_files = [args.schema_dir / name for name in selected]
    after_files += [args.schema_dir / "SCHEMA_MANIFEST.json", args.schema_dir / "SCHEMA_SHA256SUMS"]
    after_files += list(args.validator_files)
    report = {
        "artifact_kind": "V3_1_STRICT_SCHEMA_REVISION",
        "reason": "D1 contract required provenance/lifecycle/context/endpoint/delta/exposure fields absent from active schemas",
        "selected_schema_files": sorted(selected),
        "validator_files": [str(p) for p in args.validator_files],
        "archive_dir": str(args.archive_dir),
        "before_sha256": before,
        "after_sha256": {str(path): sha256_file(path) for path in after_files},
        "filename_set_unchanged": old_schema_names == sorted(p.name for p in schema_files),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
