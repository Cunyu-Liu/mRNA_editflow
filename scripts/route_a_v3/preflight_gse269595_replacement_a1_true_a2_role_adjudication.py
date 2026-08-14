#!/usr/bin/env python3
"""DEC024 aggregate-only GSE269595 A1-versus-true-A2 role preflight.

This producer is not a role assignment, qualification, credit decision, split,
training run, or model-selection run.  Its official path accepts exactly the
bound GEO processed table and author/publisher Table S5; both byte identities
must close before either parser runs.  Member material remains internal and the
sole output is aggregate.  The frozen DEC024 authority/runtime/projection
lineage, preceding GSE261709 producer, and this producer's own binding must
close before any data or output-path I/O.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping
from xml.etree import ElementTree


SCHEMA_VERSION = (
    "route_a_v3_gse269595_replacement_a1_true_a2_role_adjudication_preflight.v1"
)
PROTOCOL_ID = (
    "GSE269595_REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_V1"
)
DECISION_ID = "V3-DEC-024"
DATASET_ID = "GSE269595"
BIOPROJECT_ID = "PRJNA1122592"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"

STATUS_READY = "PREFLIGHT_COMPLETE_ROLE_STATUS_ONLY_NOT_ASSIGNED_NOT_QUALIFIED"
STATUS_BLOCKED = "BLOCKED_UNKNOWN_NOT_ASSIGNED_NOT_QUALIFIED"
STATUS_STOP = "STOP_FAIL_CLOSED_NOT_ASSIGNED_NOT_QUALIFIED"

CONFIG_PATH = (
    "configs/"
    "route_a_v3_gse269595_replacement_a1_true_a2_role_adjudication_preflight_v1.json"
)
SCRIPT_PATH = (
    "scripts/route_a_v3/"
    "preflight_gse269595_replacement_a1_true_a2_role_adjudication.py"
)
TEST_PATH = (
    "tests/route_a_v3/"
    "test_preflight_gse269595_replacement_a1_true_a2_role_adjudication.py"
)
EXACT3 = (CONFIG_PATH, SCRIPT_PATH, TEST_PATH)
RUNTIME_PATHS = (
    "configs/route_a_v3_dec024_authority_runtime_sync_v1.json",
    "scripts/route_a_v3/dec024_authority_runtime_sync.py",
    "tests/route_a_v3/test_dec024_authority_runtime_sync.py",
)
RUNTIME_IMPLEMENTATION_COMMIT = "f955ca5a1714af57f706ee2ddf0a6825ad4737de"
RUNTIME_BINDING_COMMIT = "e3c3416e24e0298ab792a1e0998018125c907ffa"
RUNTIME_IMPLEMENTATION_BLOBS = {
    RUNTIME_PATHS[0]: "3dbbdaed8458c6eb68af1c11d890b66c4116457b84df96dd84ef8622be0fd669",
    RUNTIME_PATHS[1]: "6d9614b53e160fe38bbf280c310d6adec773b790b48f93374448ff6b29e5bd3b",
    RUNTIME_PATHS[2]: "b22d92aeea7f3c0dc754dd739daa1d5025af64e96e86a84e7b0458b1788c3799",
}
RUNTIME_BINDING_BLOBS = {
    RUNTIME_PATHS[0]: "1b1a3159fc7b08aeb967983f3b651bdb8cba182829c22e767b6a9e9dad1fb7e1",
    RUNTIME_PATHS[1]: RUNTIME_IMPLEMENTATION_BLOBS[RUNTIME_PATHS[1]],
    RUNTIME_PATHS[2]: RUNTIME_IMPLEMENTATION_BLOBS[RUNTIME_PATHS[2]],
}
PROJECTION_COMMIT = "6df392e61d0d55b836c5baf84ce67f4aa9e7d1fe"
PROJECTION_EXACT4 = (
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
PROJECTION_BLOBS = {
    PROJECTION_EXACT4[0]: "06bfbcf468e28ee27f2f02210a0cad6719cb805cb441ea40142b7b837680b44b",
    PROJECTION_EXACT4[1]: "fde5f7150a6dbd8b3e1caa53c69beb5ddb8b7fb9f3242bd1ffe270b165a579b9",
    PROJECTION_EXACT4[2]: "4d4188f7777a2651c73b19e691be22687a73ad0da2abe8fc39f2c0e297ddc3a0",
    PROJECTION_EXACT4[3]: "aa7e5773d4353f4c8fb9a9afb6d5b9a3f3fb1ba035c6c54e6ae0d4eabac99c6b",
}
PREDECESSOR_PROTOCOL_ID = (
    "GSE261709_DEC024_AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_V1"
)
PREDECESSOR_PATHS = (
    "configs/route_a_v3_gse261709_dec024_aggregate_row_level_a1_qualification_preflight_v1.json",
    "scripts/route_a_v3/preflight_gse261709_dec024_aggregate_row_level_a1_qualification.py",
    "tests/route_a_v3/test_preflight_gse261709_dec024_aggregate_row_level_a1_qualification.py",
)
PREDECESSOR_IMPLEMENTATION_COMMIT = "1c36549f7d00f17b87f2afa6d4b86d7ca352f3d9"
PREDECESSOR_BINDING_COMMIT = "3fd73fb79c764a46bc502e0748e906926bdea9b6"
PREDECESSOR_IMPLEMENTATION_BLOBS = {
    PREDECESSOR_PATHS[0]: "9d91e5f6d2a185c6329730ca1ec44709ff5be760c09c5d3172fc5db22e68e5fd",
    PREDECESSOR_PATHS[1]: "63fc42708ba2b98e72e09ae331d912f68f37fb513e1f552e4be18e0b5190268d",
    PREDECESSOR_PATHS[2]: "f80213b8747849dce77359d277eb55870f4b3e539a303adcbcd5e33805b02baf",
}
PREDECESSOR_BINDING_BLOBS = {
    PREDECESSOR_PATHS[0]: "244509d8d9f35b542715e6f331845fb415c7387694757a2611781622b94c3d27",
    PREDECESSOR_PATHS[1]: PREDECESSOR_IMPLEMENTATION_BLOBS[PREDECESSOR_PATHS[1]],
    PREDECESSOR_PATHS[2]: PREDECESSOR_IMPLEMENTATION_BLOBS[PREDECESSOR_PATHS[2]],
}
AUTHORITY_COMMIT = "0bb84dffb1389b9eced7e92e36ef80b8a97ed0be"
AUTHORITY_EXPECTED_PARENT = "e5d089a43d194caf59369fd12c203c0694ba40c6"
AUTHORITY_EXACT12 = (
    "configs/route_a_v3.yaml",
    "configs/route_a_v3_a1_qualification.json",
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec024.yaml",
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml",
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_a6_interim.yaml",
    "docs/execution/route_a_v3_data_role_registry.yaml",
    "docs/execution/route_a_v3_decision_log.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "docs/execution/route_a_v3_task_registry.yaml",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
AUTHORITY_BLOBS = {
    "configs/route_a_v3.yaml": (
        "01cb7bb052da7459e946828b7c92dede8f257e3f13abba3534455b370ef09b74"
    ),
    "configs/route_a_v3_a1_qualification.json": (
        "46ce6ff3648f1abec47bfc9eb63045759ab8676b12fd3c38c5efe31d64f37c41"
    ),
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec024.yaml": (
        "163c5b744a8d68e6e0bd3afad378c3cd8611d42e7f8ff881291557049d908eac"
    ),
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml": (
        "d403a4aa7db9343848be74ae061b8196613525ff03600b1034864d7d803c7beb"
    ),
    "docs/execution/route_a_v3_a1_interim.yaml": (
        "0f35ae3b1b56a76174edac92fa578c0cdcd83a4a68c757717dbe155cf79e265a"
    ),
    "docs/execution/route_a_v3_a6_interim.yaml": (
        "c0002dac3ea470f69bf79e752a57041b9910e9583bddefd1625c6279f50e3081"
    ),
    "docs/execution/route_a_v3_data_role_registry.yaml": (
        "f62034239854d494c45196d2535895e9593cc38655d7b9042719fb912cf08e45"
    ),
    "docs/execution/route_a_v3_decision_log.yaml": (
        "06a031cf67ead3417942938a17f4783a6f2168866f54e1fc2abe1b7fa938c0c3"
    ),
    "docs/execution/route_a_v3_registry_manifest.json": (
        "1ff8ba5fce218794516b753628772a18ae93e6e933e08cc2c7174a09590cba3f"
    ),
    "docs/execution/route_a_v3_task_registry.yaml": (
        "c1d9920cc3d28c7ee63d9649a69d6b6856b9b25115f6ca4f1d392a067a0d5dae"
    ),
    "scripts/route_a_v3/validate_a0_bundle.py": (
        "d90908e4b2726df3621e5ef97fc290ec4af8464b7037cb000a2071886c586d3c"
    ),
    "tests/route_a_v3/test_a0_integrity_guards.py": (
        "625677fd9e639ae0d9c23b8fc49aa321c7e0e46c3a52784e3ff41efbcd93d1e1"
    ),
}
A6_G0_COMMIT = "8fde46ca7daa765fa3a8ad8ce24a3da82ce1a8d0"
A6_G0_STATUS = "FROZEN_NONAUTHORITATIVE_NO_SCIENCE_STATE_CHANGE"
A6_G0_EXACT4 = (
    "configs/route_a_v3_a6_learned_base_value_g0_implementation_candidate_v1.json",
    "docs/plans/2026-08-14-route-a-v3-a6-learned-g0-implementation-candidate-v1.md",
    "scripts/route_a_v3/a6_learned_base_value_g0_candidate.py",
    "tests/route_a_v3/test_a6_learned_base_value_g0_candidate.py",
)
A6_G0_BLOBS = {
    A6_G0_EXACT4[0]: (
        "f26ab89d8030f1c7ca91f1f60933475181b4270591532248daa4c8e1de8510f1"
    ),
    A6_G0_EXACT4[1]: (
        "371e9f2c581ec83120d0300f121d5354f4fc5d388b5afa72e4a1a0f9514595b9"
    ),
    A6_G0_EXACT4[2]: (
        "9a09df25b89ee8e08ffbb2c84d955fddffa2b93b3a5216dc3d8ee1af688fc891"
    ),
    A6_G0_EXACT4[3]: (
        "4c6ab5908f719989b42854aa73b915d7cc1d864879fa5ffc9652dcf1efb6becf"
    ),
}
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
REPORT_FILENAME = (
    "GSE269595_REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_V1.json"
)

ALLOWED_INPUT_FIELDS = (
    "PUBLIC_IDENTIFIER",
    "SOURCE_LOCUS_AND_SOURCE_FAMILY_ROLE",
    "SOURCE_AND_CANDIDATE_SEQUENCE",
    "APA_LOCATION_AND_INTRONIC_CONTEXT",
    "LEGAL_SUBSTITUTION_ANNOTATION",
    "ASSET_SAMPLE_RUN_ROLE",
    "HEADER_NAME_AND_SCHEMA_ROLE",
    "ASSET_DIMENSION_AND_AGGREGATE_COUNT",
    "ASSAY_GUIDE_AND_REPORTER_CONTEXT",
    "BIOLOGICAL_REPLICATE_ENDPOINT_AND_STANDARD_ERROR",
    "MISSINGNESS_AND_CENSORING_STATUS",
    "LICENSE_AND_REUSE_NOTICE",
    "APARENT_EXPOSURE_AND_MODEL_INPUT_ROUTE_CONTEXT",
)
ALLOWED_INTERNAL_USES = (
    "A1_VERSUS_TRUE_A2_ROLE_ELIGIBILITY_ADJUDICATION",
    "SOURCE_FAMILY_DISTRIBUTION_AUDIT",
    "INTRONIC_APA_EXCLUSION_AUDIT",
    "SOURCE_TO_CANDIDATE_LEGAL_SUBSTITUTION_REPLAY",
    "ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_AUDIT",
    "ASSAY_GUIDE_CONTEXT_ENDPOINT_AND_REPLICATE_SE_AUDIT",
    "APARENT_EXPOSURE_AND_RIGHTS_AUDIT",
    "SOURCE_GROUP_SPLIT_LEAKAGE_EFFECTIVE_N_AND_POWER_READINESS_AUDIT",
)
ALLOWED_AGGREGATE_OUTPUTS = (
    "GATE_STATUS_AND_REASON_COUNTS",
    "MUTUALLY_EXCLUSIVE_A1_TRUE_A2_ROLE_STATUS",
    "SOURCE_FAMILY_SIZE_AND_REGION_CLASS_HISTOGRAMS",
    "INTRONIC_APA_EXCLUSION_COUNTS",
    "LEGAL_SUBSTITUTION_REPLAY_COUNTS_AND_HISTOGRAMS",
    "ASSET_SAMPLE_RUN_ROLE_AND_SCHEMA_COVERAGE_COUNTS",
    "CONTEXT_GUIDE_REPLICATE_AND_STANDARD_ERROR_COVERAGE_HISTOGRAMS",
    "MISSING_CENSORING_RIGHTS_AND_EXPOSURE_GATE_STATUS",
    "POST_DEDUP_SOURCE_GROUP_EFFECTIVE_N_AND_POWER_READINESS_STATUS",
)
FORBIDDEN_OUTPUTS = (
    "MEMBER_IDENTIFIER",
    "ACTUAL_HEADER_NAME",
    "SOURCE_OR_CANDIDATE_SEQUENCE",
    "ROW_ENDPOINT_OR_EFFECT",
    "ROW_STANDARD_ERROR",
    "REPLICATE_IDENTIFIER",
    "SPLIT_ASSIGNMENT",
)
MPRA_HEADER = (
    "sample",
    "replicate",
    "perturbation",
    "distal_site",
    "barcode",
    "gene_id",
    "pas_id",
    "aim",
    "subaim",
    "experiment",
    "n_bc",
    "total",
    "distal",
    "proximal",
    "log_odds",
)
TABLE_S5_HEADER = (
    "gene_id",
    "pas_id",
    "type",
    "subtype",
    "experiment",
    "n_bc",
    "barcoded_seq_184bp",
)
OFFICIAL_MPRA_SPEC = {
    "canonical_url": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE269nnn/GSE269595/suppl/"
        "GSE269595_mpra_constructs_all_samples_proximal_site_usage.txt.gz"
    ),
    "byte_count": 5_910_248,
    "sha256": "6527ea54257b3f17ddb9df5977637e41e2ef16d926a27be4e29f56165acaa1de",
    "format": "GZIP_UTF8_WHITESPACE_TABLE",
    "exact_header_order": list(MPRA_HEADER),
    "exact_data_row_count": 366_780,
    "ordinary_public": True,
    "private_or_restricted": False,
    "asset_role": "OFFICIAL_GEO_PROCESSED_MPRA_COUNT_AND_PROXIMAL_USAGE_TABLE",
}
PUBLISHER_TABLE_S5_SPEC = {
    "canonical_url": (
        "https://ars.els-cdn.com/content/image/1-s2.0-S0092867424006457-mmc5.xlsx"
    ),
    "byte_count": 353_937,
    "sha256": "d350be818d87120216052645a5ffa97afeee898d32a877c74033dba0d0fa151a",
    "format": "XLSX_OOXML",
    "required_sheet": "MPRA oligo library",
    "required_sheet_dimension": "A1:G6114",
    "exact_header_order": list(TABLE_S5_HEADER),
    "exact_data_row_count": 6_113,
    "ordinary_public": True,
    "private_or_restricted": False,
    "asset_role": "AUTHOR_PUBLISHER_MPRA_OLIGO_LIBRARY_TABLE_S5",
}
ORDINARY_PUBLIC_INPUT_CONTRACT = {
    "caller_supplied_context_allowed": False,
    "both_asset_identities_must_close_before_any_parse": True,
    "official_mpra_processed_asset": OFFICIAL_MPRA_SPEC,
    "publisher_table_s5_asset": PUBLISHER_TABLE_S5_SPEC,
    "private_or_restricted_input_allowed": False,
    "raw_fastq_or_sra_member_payload_allowed": False,
    "persistent_member_level_intermediate_allowed": False,
}
BOUND_ROLE_CONTEXT_SEMANTICS = {
    "join_protocol": {
        "authoritative_member_join": (
            "GEO.barcode_EQUALS_TableS5.barcoded_seq_184bp_FIRST_20_NT"
        ),
        "required_join_cross_checks_exactly": [
            "gene_id",
            "pas_id",
            "experiment",
            "n_bc",
            "aim_EQUALS_type",
        ],
        "subtype_subaim_equality_required": False,
        "table_generic_subtype_may_be_refined_by_geo_subaim": True,
        "shared_metadata_join_allowed": False,
        "row_order_join_allowed": False,
        "every_member_and_processed_row_must_join_exactly_once": True,
    },
    "sequence_protocol": {
        "barcoded_sequence_length": 184,
        "barcode_prefix_length": 20,
        "candidate_construct_suffix_length": 164,
        "alphabet": "ACGT",
    },
    "source_replay_protocol": {
        "source_family_key_exactly": ["gene_id", "pas_id"],
        "candidate_design_key_exactly": [
            "gene_id",
            "pas_id",
            "experiment",
        ],
        "literal_source_experiment_value": "wt",
        "exactly_one_literal_source_construct_per_family_required": True,
        "declared_legal_edit_annotation_required": True,
        "source_or_edit_relation_may_be_inferred_from_row_order": False,
    },
    "processed_geometry_protocol": {
        "exact_sample_count": 12,
        "exact_distal_reporter_context_count": 5,
        "exact_sample_by_distal_contexts_per_barcode": 60,
        "sample_label_must_match_replicate_and_perturbation_fields": True,
        "sample_labels_establish_biological_independence": False,
        "reported_standard_error_present": False,
        "total_must_equal_distal_plus_proximal": True,
        "log_odds_formula": (
            "LOG2_PROXIMAL_DIVIDED_BY_DISTAL_WITH_SIGNED_INFINITY_AND_NA_FOR_ZERO_ZERO"
        ),
    },
    "authority_status": {
        "a1_primary_role_evidence": UNKNOWN,
        "true_a2_primary_role_evidence": UNKNOWN,
        "intronic_apa_context": UNKNOWN,
        "biological_replicate_independence_and_standard_error": UNKNOWN,
        "missing_censoring_and_selection": UNKNOWN,
        "license_and_reuse_rights": UNKNOWN,
        "aparent_prior_exposure": UNKNOWN,
        "near_duplicate_split_readiness": UNKNOWN,
    },
}
EXPECTED_SAMPLE_FIELDS = {
    "CSTF3gA-rep1": ("rep1", "CSTF3"),
    "CSTF3gA-rep2": ("rep2", "CSTF3"),
    "CSTF3gB-rep1": ("rep1", "CSTF3"),
    "CSTF3gB-rep2": ("rep2", "CSTF3"),
    "NTgA-rep1": ("rep1", "NT"),
    "NTgA-rep2": ("rep2", "NT"),
    "NTgB-rep1": ("rep1", "NT"),
    "NTgB-rep2": ("rep2", "NT"),
    "NUDT21gA-rep1": ("rep1", "NUDT21"),
    "NUDT21gA-rep2": ("rep2", "NUDT21"),
    "NUDT21gB-rep1": ("rep1", "NUDT21"),
    "NUDT21gB-rep2": ("rep2", "NUDT21"),
}
EXPECTED_DISTAL_CONTEXTS = (
    "CCT6A_moduleA",
    "CDK1_moduleB",
    "TMEM106C_moduleB",
    "TMEM237_moduleA",
    "bGH",
)
GATE_IDS = (
    "A1_VERSUS_TRUE_A2_ROLE_ELIGIBILITY_AND_MUTUAL_EXCLUSIVITY_CLOSED",
    "ORDINARY_PUBLIC_ASSET_IDENTITY_ROLE_AND_PROVENANCE_CLOSED",
    "SOURCE_FAMILY_DISTRIBUTION_AND_UNIQUE_SOURCE_ANCHOR_CLOSED",
    "INTRONIC_APA_EXCLUSION_CLOSED",
    "SOURCE_TO_CANDIDATE_LEGAL_SUBSTITUTION_REPLAY_CLOSED",
    "ASSAY_CONTEXT_GUIDE_ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED",
    "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED",
    "ASSET_SCHEMA_DIMENSION_AND_COVERAGE_CLOSED",
    "MISSING_CENSORING_AND_SELECTION_CLOSED",
    "APARENT_PRIOR_EXPOSURE_AND_MODEL_INPUT_ROUTE_CLOSED",
    "LICENSE_AND_REUSE_RIGHTS_CLOSED",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED",
    "POST_DEDUP_SOURCE_GROUP_EFFECTIVE_N_AND_PREFROZEN_POWER_FULL_CI_WIDTH_CLOSED",
)
A1_ROLE_VALUE = "SOURCE_RELATIVE_INTERVENTION_CONTRAST_PRIMARY"
TRUE_A2_ROLE_VALUE = "SOURCE_ANCHORED_MEASURED_DENSE_NEIGHBORHOOD_PRIMARY"
ALLOWED_REGIONS = ("EXONIC_3UTR", "TERMINAL_EXON_3UTR")
ALLOWED_DIRECTIONS = ("HIGHER_IS_BETTER", "LOWER_IS_BETTER")

NO_PROMOTION_LOCKS = {
    "changes_current_qualified_counts": False,
    "current_qualified_counts": {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    },
    "dataset_contribution": {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    },
    "registry_role": "AUDIT_ONLY",
    "role_assignment_allowed": False,
    "qualification_allowed": False,
    "canonical_materialization_allowed": False,
    "split_execution_allowed": False,
    "formal_qualification_power_gate_execution_allowed": False,
    "training_allowed": False,
    "gpu_work_allowed": False,
    "model_selection_allowed": False,
    "a6_pass": False,
    "l3_claim_established": False,
    "a7_allowed": False,
    "next_phase_authorized": False,
    "scientific_claim_status": "NOT_ESTABLISHED",
}


class PreflightError(RuntimeError):
    """Base error for protocol, binding, asset, or output closure."""


class ProtocolError(PreflightError):
    """The static protocol is not the frozen DEC024 candidate."""


class BindingNotFrozen(ProtocolError):
    """A grouped lifecycle binding is not fully frozen."""


class AssetError(PreflightError):
    """An authorized ordinary-public input cannot be parsed."""


class OutputError(PreflightError):
    """The sole aggregate artifact cannot be published safely."""


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ProtocolError(f"{label} contains a duplicate JSON key")
            value[key] = item
        return value

    def invalid_constant(_: str) -> None:
        raise ProtocolError(f"{label} contains a non-finite JSON number")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    return value


def _is_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _validate_digest_map(value: Any, paths: tuple[str, ...], *, label: str) -> None:
    mapping = _mapping(value, label=label)
    if set(mapping) != set(paths) or any(not _is_hex(item, 64) for item in mapping.values()):
        raise ProtocolError(f"{label} differs from the exact path/digest set")


def _runtime_dynamic_values(group: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        group.get("status"),
        group.get("implementation_commit"),
        group.get("implementation_expected_parent"),
        group.get("implementation_blob_sha256_by_path"),
        group.get("binding_commit"),
        group.get("binding_expected_parent"),
        group.get("binding_blob_sha256_by_path"),
    )


def _validate_runtime_group(group: Mapping[str, Any]) -> None:
    if group.get("protocol_id") != "ROUTE_A_V3_DEC024_AUTHORITY_RUNTIME_SYNC_V1":
        raise ProtocolError("authority-runtime protocol differs")
    if tuple(group.get("paths", ())) != RUNTIME_PATHS:
        raise ProtocolError("authority-runtime exact3 paths differ")
    if tuple(group.get("binding_exact_changed_paths", ())) != (RUNTIME_PATHS[0],):
        raise ProtocolError("authority-runtime binding must be config-only")
    if group.get("status") != BOUND:
        raise ProtocolError("authority-runtime status is not the frozen BOUND chain")
    _validate_digest_map(
        group.get("implementation_blob_sha256_by_path"),
        RUNTIME_PATHS,
        label="authority-runtime I blobs",
    )
    _validate_digest_map(
        group.get("binding_blob_sha256_by_path"),
        RUNTIME_PATHS,
        label="authority-runtime B blobs",
    )
    if (
        group.get("implementation_commit") != RUNTIME_IMPLEMENTATION_COMMIT
        or group.get("implementation_expected_parent") != A6_G0_COMMIT
        or group.get("implementation_blob_sha256_by_path")
        != RUNTIME_IMPLEMENTATION_BLOBS
    ):
        raise ProtocolError("authority-runtime I exact3 identity differs")
    if (
        group.get("binding_commit") != RUNTIME_BINDING_COMMIT
        or group.get("binding_expected_parent") != RUNTIME_IMPLEMENTATION_COMMIT
        or group.get("binding_blob_sha256_by_path") != RUNTIME_BINDING_BLOBS
    ):
        raise ProtocolError("authority-runtime B config-only identity differs")


def _validate_projection_group(group: Mapping[str, Any]) -> None:
    if (
        group.get("status") != BOUND
        or group.get("commit") != PROJECTION_COMMIT
        or group.get("expected_parent") != RUNTIME_BINDING_COMMIT
        or tuple(group.get("exact_changed_paths", ())) != PROJECTION_EXACT4
        or group.get("blob_sha256_by_path") != PROJECTION_BLOBS
        or group.get("changes_dec024_authority") is not False
        or group.get("changes_scientific_state") is not False
    ):
        raise ProtocolError("EVT058 current-projection P exact4 identity differs")


def _predecessor_dynamic_values(group: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        group.get("status"),
        group.get("implementation_commit"),
        group.get("implementation_blob_sha256_by_path"),
        group.get("binding_commit"),
        group.get("binding_expected_parent"),
        group.get("binding_blob_sha256_by_path"),
    )


def _validate_predecessor_group(group: Mapping[str, Any]) -> str:
    if group.get("protocol_id") != PREDECESSOR_PROTOCOL_ID:
        raise ProtocolError("GSE261709 predecessor protocol differs")
    if tuple(group.get("paths", ())) != PREDECESSOR_PATHS:
        raise ProtocolError("GSE261709 predecessor exact3 paths differ")
    if group.get("implementation_expected_parent") != PROJECTION_COMMIT:
        raise ProtocolError("GSE261709 predecessor I parent is not projection P")
    if tuple(group.get("implementation_exact_changed_paths", ())) != PREDECESSOR_PATHS:
        raise ProtocolError("GSE261709 predecessor I is not exact3")
    if tuple(group.get("binding_exact_changed_paths", ())) != (PREDECESSOR_PATHS[0],):
        raise ProtocolError("GSE261709 predecessor B must be config-only")
    dynamic = _predecessor_dynamic_values(group)
    if all(value == UNKNOWN for value in dynamic):
        return UNKNOWN
    if any(value == UNKNOWN for value in dynamic):
        raise ProtocolError("GSE261709 predecessor identity is partially bound")
    if group.get("status") != BOUND:
        raise ProtocolError("GSE261709 predecessor status is invalid")
    implementation = group.get("implementation_commit")
    binding = group.get("binding_commit")
    if not _is_hex(implementation, 40) or not _is_hex(binding, 40):
        raise ProtocolError("GSE261709 predecessor commit identity is invalid")
    if group.get("binding_expected_parent") != implementation:
        raise ProtocolError("GSE261709 predecessor B is not bound to I")
    _validate_digest_map(
        group.get("implementation_blob_sha256_by_path"),
        PREDECESSOR_PATHS,
        label="GSE261709 predecessor I blobs",
    )
    _validate_digest_map(
        group.get("binding_blob_sha256_by_path"),
        PREDECESSOR_PATHS,
        label="GSE261709 predecessor B blobs",
    )
    implementation_blobs = group["implementation_blob_sha256_by_path"]
    binding_blobs = group["binding_blob_sha256_by_path"]
    if any(
        binding_blobs[path] != implementation_blobs[path]
        for path in PREDECESSOR_PATHS[1:]
    ):
        raise ProtocolError("GSE261709 predecessor B changed script or test")
    if (
        implementation != PREDECESSOR_IMPLEMENTATION_COMMIT
        or group.get("implementation_blob_sha256_by_path")
        != PREDECESSOR_IMPLEMENTATION_BLOBS
    ):
        raise ProtocolError("GSE261709 predecessor I exact3 identity differs")
    if (
        binding != PREDECESSOR_BINDING_COMMIT
        or group.get("binding_blob_sha256_by_path") != PREDECESSOR_BINDING_BLOBS
    ):
        raise ProtocolError("GSE261709 predecessor B config-only identity differs")
    return BOUND


def _preflight_dynamic_values(group: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        group.get("status"),
        group.get("implementation_commit"),
        group.get("implementation_script_sha256"),
        group.get("implementation_test_sha256"),
    )


def _validate_preflight_group(group: Mapping[str, Any]) -> None:
    if group.get("implementation_script_path") != SCRIPT_PATH:
        raise ProtocolError("preflight script path differs")
    if group.get("implementation_test_path") != TEST_PATH:
        raise ProtocolError("preflight test path differs")
    if tuple(group.get("implementation_exact_changed_paths", ())) != EXACT3:
        raise ProtocolError("preflight implementation is not exact3")
    if tuple(group.get("binding_exact_changed_paths", ())) != (CONFIG_PATH,):
        raise ProtocolError("preflight binding must be config-only")
    expected_scalars = (
        "implementation_binding.preflight_group.status",
        "implementation_binding.preflight_group.implementation_commit",
        "implementation_binding.preflight_group.implementation_script_sha256",
        "implementation_binding.preflight_group.implementation_test_sha256",
    )
    if tuple(group.get("unknown_to_bound_scalar_paths", ())) != expected_scalars:
        raise ProtocolError("preflight four-scalar binding interface differs")
    dynamic = _preflight_dynamic_values(group)
    if all(value == UNKNOWN for value in dynamic):
        return
    if any(value == UNKNOWN for value in dynamic):
        raise ProtocolError("preflight four-scalar identity is partially bound")
    if group.get("status") != BOUND or not _is_hex(group.get("implementation_commit"), 40):
        raise ProtocolError("preflight implementation status/commit is invalid")
    if not _is_hex(group.get("implementation_script_sha256"), 64):
        raise ProtocolError("preflight script digest is invalid")
    if not _is_hex(group.get("implementation_test_sha256"), 64):
        raise ProtocolError("preflight test digest is invalid")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    identity = (
        protocol.get("schema_version"),
        protocol.get("protocol_id"),
        protocol.get("contract_id"),
        protocol.get("phase_id"),
        protocol.get("dataset_id"),
        protocol.get("bioproject_id"),
        protocol.get("decision_id"),
    )
    expected_identity = (
        SCHEMA_VERSION,
        PROTOCOL_ID,
        "mrna_xeditflow_route_a_v3",
        "A1",
        DATASET_ID,
        BIOPROJECT_ID,
        DECISION_ID,
    )
    if identity != expected_identity:
        raise ProtocolError("protocol identity differs")
    if protocol.get("protocol_status") != (
        "REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_ONLY_"
        "NOT_ROLE_ASSIGNMENT_NOT_QUALIFICATION"
    ):
        raise ProtocolError("protocol status broadens DEC024")

    binding = _mapping(protocol.get("implementation_binding"), label="implementation binding")
    if binding.get("binding_scheme") != (
        "DEC024_A_THEN_NONAUTHORITATIVE_A6_G0_EXACT4_THEN_RUNTIME_I_B_"
        "THEN_EVT058_PROJECTION_P_THEN_GSE261709_I_B_THEN_GSE269595_I_B_V1"
    ):
        raise ProtocolError("binding scheme differs")
    authority = _mapping(binding.get("authority_group"), label="authority group")
    if (
        authority.get("status") != BOUND
        or authority.get("authority_commit") != AUTHORITY_COMMIT
        or authority.get("authority_expected_parent") != AUTHORITY_EXPECTED_PARENT
        or tuple(authority.get("authority_exact_changed_paths", ())) != AUTHORITY_EXACT12
        or authority.get("authority_blob_sha256_by_path") != AUTHORITY_BLOBS
    ):
        raise ProtocolError("DEC024 A identity differs")
    a6_g0 = _mapping(
        binding.get("nonauthoritative_a6_g0_group"), label="A6 G0 lineage group"
    )
    if (
        a6_g0.get("status") != A6_G0_STATUS
        or a6_g0.get("commit") != A6_G0_COMMIT
        or a6_g0.get("expected_parent") != AUTHORITY_COMMIT
        or tuple(a6_g0.get("exact_changed_paths", ())) != A6_G0_EXACT4
        or a6_g0.get("blob_sha256_by_path") != A6_G0_BLOBS
        or a6_g0.get("authority_or_scientific_state_changed") is not False
    ):
        raise ProtocolError("nonauthoritative A6 G0 exact4 lineage differs")
    _validate_runtime_group(
        _mapping(binding.get("authority_runtime_group"), label="authority-runtime group")
    )
    _validate_projection_group(
        _mapping(binding.get("current_projection_group"), label="current-projection group")
    )
    predecessor_mode = _validate_predecessor_group(
        _mapping(binding.get("gse261709_predecessor_group"), label="GSE261709 predecessor group")
    )
    _validate_preflight_group(
        _mapping(binding.get("preflight_group"), label="preflight group")
    )
    own = binding["preflight_group"]
    if predecessor_mode != BOUND and any(
        value != UNKNOWN for value in _preflight_dynamic_values(own)
    ):
        raise ProtocolError("GSE269595 I cannot bind before GSE261709 predecessor B")

    decision = _mapping(protocol.get("decision_authority"), label="decision authority")
    if (
        decision.get("amendment_path")
        != "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec024.yaml"
        or decision.get("amendment_sha256") != AUTHORITY_BLOBS[AUTHORITY_EXACT12[2]]
        or decision.get("registry_role_must_remain") != "AUDIT_ONLY"
        or decision.get("authorized_role")
        != "REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_ONLY"
        or decision.get("authority_surface") != "ORDINARY_PUBLIC_ONLY"
        or tuple(decision.get("allowed_internal_input_field_classes_exactly", ()))
        != ALLOWED_INPUT_FIELDS
        or tuple(decision.get("allowed_internal_uses_exactly", ()))
        != ALLOWED_INTERNAL_USES
        or decision.get("allowed_output_class")
        != "AGGREGATE_GATE_COUNTS_HISTOGRAMS_ROLE_STATUS_ONLY"
        or tuple(decision.get("allowed_aggregate_outputs_exactly", ()))
        != ALLOWED_AGGREGATE_OUTPUTS
        or tuple(decision.get("forbidden_output_classes", ())) != FORBIDDEN_OUTPUTS
    ):
        raise ProtocolError("DEC024 authority surface differs")

    inputs = _mapping(protocol.get("ordinary_public_input_contract"), label="input contract")
    if inputs != ORDINARY_PUBLIC_INPUT_CONTRACT:
        raise ProtocolError("ordinary-public input boundary differs")
    semantics = _mapping(
        protocol.get("bound_role_context_semantics"), label="bound role/context semantics"
    )
    if semantics != BOUND_ROLE_CONTEXT_SEMANTICS:
        raise ProtocolError("bound asset role/context semantics differ")

    role = _mapping(protocol.get("role_adjudication_policy"), label="role policy")
    if (
        role.get("a1_role_evidence_value") != A1_ROLE_VALUE
        or role.get("true_a2_role_evidence_value") != TRUE_A2_ROLE_VALUE
        or role.get("exactly_one_primary_evidence_value_required") is not True
        or role.get("a1_role_may_be_presumed") is not False
        or role.get("true_a2_role_may_be_presumed") is not False
        or role.get("maximum_roles_if_later_qualified") != 1
        or role.get("a1_and_true_a2_double_credit_allowed") is not False
        or role.get("role_status_is_role_assignment_or_qualification") is not False
        or role.get("role_evidence_source_locator_required") is not True
    ):
        raise ProtocolError("role mutual-exclusivity policy differs")

    family = _mapping(protocol.get("source_family_policy"), label="family policy")
    if (
        family.get("minimum_distinct_candidates_per_source_family") != 3
        or family.get("exactly_one_source_anchor_per_family_context") is not True
        or family.get("source_family_membership_may_be_inferred_from_row_order") is not False
    ):
        raise ProtocolError("source-family policy differs")
    intronic = _mapping(protocol.get("intronic_apa_policy"), label="intronic policy")
    if (
        intronic.get("intronic_apa_exclusion_required") is not True
        or tuple(intronic.get("allowed_nonintronic_region_classes", ())) != ALLOWED_REGIONS
        or intronic.get("intronic_or_unknown_region_is_eligible") is not False
    ):
        raise ProtocolError("intronic APA policy differs")
    replay = _mapping(
        protocol.get("legal_substitution_replay_policy"), label="replay policy"
    )
    if (
        replay.get("source_to_candidate_edit_relation_may_be_presumed") is not False
        or replay.get("alphabet_after_normalization") != "ACGT"
        or replay.get("rna_u_normalizes_to_dna_t") is not True
        or replay.get("equal_length_required") is not True
        or replay.get("nonzero_substitution_count_required") is not True
        or replay.get("declared_edit_set_must_equal_replayed_edit_set") is not True
        or replay.get("indel_allowed") is not False
        or replay.get("route_a_budget_qualification_is_part_of_this_preflight") is not False
    ):
        raise ProtocolError("legal-substitution replay policy differs")

    replicate = _mapping(
        protocol.get("assay_endpoint_replicate_policy"), label="replicate policy"
    )
    if (
        replicate.get("minimum_independent_biological_replicates_per_candidate") != 2
        or replicate.get("technical_units_may_substitute_for_biological_replicates") is not False
        or replicate.get("replicate_unit_value_required") != "INDEPENDENT_BIOLOGICAL"
        or tuple(replicate.get("endpoint_direction_allowed", ())) != ALLOWED_DIRECTIONS
        or replicate.get("finite_endpoint_required") is not True
        or replicate.get("finite_nonnegative_standard_error_required") is not True
        or replicate.get("reported_standard_error_must_match_replicate_sample_standard_error")
        is not True
        or replicate.get("standard_error_absolute_tolerance") != 1e-12
        or replicate.get("standard_error_relative_tolerance") != 1e-6
        or replicate.get("context_reporter_and_guide_must_be_nonempty") is not True
    ):
        raise ProtocolError("endpoint/replicate/SE policy differs")
    missing = _mapping(protocol.get("missing_censoring_policy"), label="missing policy")
    if (
        missing.get("required_measurement_status") != "OBSERVED"
        or missing.get("required_censoring_status") != "NOT_CENSORED"
        or missing.get("required_selection_status")
        != "FULL_INTENDED_UNIVERSE_NOT_OUTCOME_SELECTED"
        or missing.get("missing_or_censored_measurement_may_be_treated_as_zero") is not False
    ):
        raise ProtocolError("missing/censoring policy differs")

    rights = _mapping(protocol.get("rights_policy"), label="rights policy")
    if rights != {
        "ordinary_public_locator_required": True,
        "license_or_reuse_notice_locator_required": True,
        "analysis_reuse_allowed_required": True,
        "aggregate_derived_reporting_allowed_required": True,
        "private_or_restricted_asset_allowed": False,
    }:
        raise ProtocolError("rights policy differs")
    exposure = _mapping(protocol.get("aparent_exposure_policy"), label="exposure policy")
    if exposure != {
        "closed_no_prior_exposure_value": "NONE_CONFIRMED_BY_AUTHORITATIVE_PROVENANCE",
        "required_future_model_input_route": (
            "SCRATCH_ONLY_NO_FOUNDATION_EXPOSURE_NO_MODEL_INPUT_UNTIL_QUALIFIED"
        ),
        "evidence_locator_required": True,
        "historical_analytic_or_checkpoint_exposure": UNKNOWN,
        "unknown_historical_exposure_is_gate_blocker": True,
    }:
        raise ProtocolError("APARENT exposure policy differs")

    split = _mapping(protocol.get("split_and_dedup_policy"), label="split policy")
    if (
        split.get("analysis_unit") != "POST_DEDUP_INDEPENDENT_SOURCE_GROUP"
        or split.get("required_grouping_level") != "NEAR_DUPLICATE_SOURCE_CLUSTER"
        or split.get("minimum_components_for_three_way_split_feasibility") != 3
        or split.get("outcome_blind_group_key_required") is not True
        or split.get("near_duplicate_group_key_required") is not True
        or split.get("split_readiness_audit_allowed") is not True
        or split.get("split_assignment_execution_allowed") is not False
        or split.get("split_assignment_output_allowed") is not False
    ):
        raise ProtocolError("split/leakage policy differs")
    power = _mapping(protocol.get("prefrozen_power_policy"), label="power policy")
    required_n = required_effective_n(
        rho=0.25, alpha=0.05, target_power=0.8, confidence=0.95, max_width=0.3
    )
    if (
        power.get("analysis_unit") != "POST_DEDUP_INDEPENDENT_SOURCE_GROUP"
        or power.get("minimum_effect_at_alternative") != 0.25
        or power.get("effect_metric") != "SPEARMAN_RHO"
        or power.get("alpha_two_sided") != 0.05
        or power.get("target_power") != 0.8
        or power.get("confidence_level") != 0.95
        or power.get("maximum_ci_full_width") != 0.3
        or power.get("required_effective_n_for_both_power_and_ci_width") != required_n
        or power.get("power_method")
        != "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN"
        or power.get("confidence_interval_method")
        != "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE"
        or power.get("row_candidate_replicate_or_run_count_may_substitute_for_source_group_n")
        is not False
        or power.get("formal_qualification_power_gate_execution_allowed") is not False
    ):
        raise ProtocolError("prefrozen source-group power policy differs")

    if tuple(protocol.get("gate_ids", ())) != GATE_IDS:
        raise ProtocolError("the DEC024 13-gate set differs")
    if protocol.get("initial_status_for_every_gate") != NOT_RUN:
        raise ProtocolError("gate initial status differs")
    if protocol.get("unknown_or_not_run_gate_is_pass") is not False:
        raise ProtocolError("unknown gate cannot pass")
    output = _mapping(protocol.get("output_contract"), label="output contract")
    if (
        output.get("sole_report_filename") != REPORT_FILENAME
        or output.get("overwrite_allowed") is not False
        or output.get("aggregate_only") is not True
        or any(
            output.get(key) is not False
            for key in (
                "member_identifier_output_allowed",
                "actual_header_names_output_allowed",
                "sequence_output_allowed",
                "row_endpoint_output_allowed",
                "row_effect_output_allowed",
                "row_standard_error_output_allowed",
                "replicate_identifier_output_allowed",
                "split_assignment_output_allowed",
            )
        )
    ):
        raise ProtocolError("aggregate-only output contract differs")
    if protocol.get("no_promotion_locks") != NO_PROMOTION_LOCKS:
        raise ProtocolError("no-promotion locks differ")
    claim = _mapping(protocol.get("claim_boundary"), label="claim boundary")
    if (
        claim.get("all_required_gates_passing_automatically_assigns_role") is not False
        or claim.get("all_required_gates_passing_automatically_qualifies_dataset") is not False
        or claim.get("role_status_is_credit") is not False
        or claim.get("separate_user_authority_required_for_role_assignment_qualification_or_counting")
        is not True
        or claim.get("double_credit_allowed") is not False
        or claim.get("training_or_model_selection_evidence_created") is not False
        or claim.get("scientific_claim_status") != "NOT_ESTABLISHED"
    ):
        raise ProtocolError("claim boundary differs")


def _normalise_preflight_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(protocol)
    group = value["implementation_binding"]["preflight_group"]
    group["status"] = UNKNOWN
    group["implementation_commit"] = UNKNOWN
    group["implementation_script_sha256"] = UNKNOWN
    group["implementation_test_sha256"] = UNKNOWN
    return value


def _require_bound_lifecycle(protocol: Mapping[str, Any]) -> None:
    binding = protocol["implementation_binding"]
    if binding["authority_group"]["status"] != BOUND:
        raise BindingNotFrozen("DEC024 A is not bound")
    runtime = binding["authority_runtime_group"]
    if any(value == UNKNOWN for value in _runtime_dynamic_values(runtime)):
        raise BindingNotFrozen(
            "DEC024 authority-runtime I/B identity is grouped UNKNOWN; stopped before data/output I/O"
        )
    if runtime["status"] != BOUND:
        raise BindingNotFrozen("DEC024 authority-runtime group is not BOUND")
    projection = binding["current_projection_group"]
    if projection["status"] != BOUND:
        raise BindingNotFrozen("EVT058 current-projection P is not BOUND")
    predecessor = binding["gse261709_predecessor_group"]
    if any(value == UNKNOWN for value in _predecessor_dynamic_values(predecessor)):
        raise BindingNotFrozen(
            "GSE261709 predecessor I/B identity is grouped UNKNOWN; stopped before data/output I/O"
        )
    if predecessor["status"] != BOUND:
        raise BindingNotFrozen("GSE261709 predecessor group is not BOUND")
    own = binding["preflight_group"]
    if any(value == UNKNOWN for value in _preflight_dynamic_values(own)):
        raise BindingNotFrozen(
            "GSE269595 implementation identity is grouped UNKNOWN; stopped before data/output I/O"
        )
    if own["status"] != BOUND:
        raise BindingNotFrozen("GSE269595 implementation group is not BOUND")


def _run_git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BindingNotFrozen("repository authority check failed") from exc
    return completed.stdout.strip()


def _git_blob(root: Path, commit: str, path: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BindingNotFrozen("a frozen repository blob is unavailable") from exc
    return completed.stdout


def _changed_paths(root: Path, commit: str) -> tuple[str, ...]:
    output = _run_git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
    return tuple(line for line in output.splitlines() if line)


def _verify_commit(
    root: Path,
    *,
    commit: str,
    expected_parent: str,
    changed_paths: tuple[str, ...],
    blobs: Mapping[str, str],
) -> None:
    if _run_git(root, "rev-parse", f"{commit}^") != expected_parent:
        raise BindingNotFrozen("frozen commit parent differs")
    if _changed_paths(root, commit) != changed_paths:
        raise BindingNotFrozen("frozen commit changed-path set differs")
    for path, digest in blobs.items():
        if hashlib.sha256(_git_blob(root, commit, path)).hexdigest() != digest:
            raise BindingNotFrozen("frozen commit blob differs")


def _default_binding_auditor(
    protocol: Mapping[str, Any], protocol_path: Path, protocol_bytes: bytes, root: Path
) -> dict[str, Any]:
    _require_bound_lifecycle(protocol)
    root = root.resolve()
    if protocol_path.resolve() != (root / CONFIG_PATH).resolve():
        raise BindingNotFrozen("protocol path is not the production config path")
    if Path(__file__).resolve() != (root / SCRIPT_PATH).resolve():
        raise BindingNotFrozen("executed script is not the production script path")

    authority = protocol["implementation_binding"]["authority_group"]
    _verify_commit(
        root,
        commit=authority["authority_commit"],
        expected_parent=authority["authority_expected_parent"],
        changed_paths=tuple(authority["authority_exact_changed_paths"]),
        blobs=authority["authority_blob_sha256_by_path"],
    )
    a6_g0 = protocol["implementation_binding"]["nonauthoritative_a6_g0_group"]
    _verify_commit(
        root,
        commit=a6_g0["commit"],
        expected_parent=a6_g0["expected_parent"],
        changed_paths=tuple(a6_g0["exact_changed_paths"]),
        blobs=a6_g0["blob_sha256_by_path"],
    )
    runtime = protocol["implementation_binding"]["authority_runtime_group"]
    _verify_commit(
        root,
        commit=runtime["implementation_commit"],
        expected_parent=runtime["implementation_expected_parent"],
        changed_paths=RUNTIME_PATHS,
        blobs=runtime["implementation_blob_sha256_by_path"],
    )
    _verify_commit(
        root,
        commit=runtime["binding_commit"],
        expected_parent=runtime["binding_expected_parent"],
        changed_paths=(RUNTIME_PATHS[0],),
        blobs=runtime["binding_blob_sha256_by_path"],
    )
    projection = protocol["implementation_binding"]["current_projection_group"]
    _verify_commit(
        root,
        commit=projection["commit"],
        expected_parent=projection["expected_parent"],
        changed_paths=tuple(projection["exact_changed_paths"]),
        blobs=projection["blob_sha256_by_path"],
    )
    predecessor = protocol["implementation_binding"]["gse261709_predecessor_group"]
    _verify_commit(
        root,
        commit=predecessor["implementation_commit"],
        expected_parent=predecessor["implementation_expected_parent"],
        changed_paths=PREDECESSOR_PATHS,
        blobs=predecessor["implementation_blob_sha256_by_path"],
    )
    _verify_commit(
        root,
        commit=predecessor["binding_commit"],
        expected_parent=predecessor["binding_expected_parent"],
        changed_paths=(PREDECESSOR_PATHS[0],),
        blobs=predecessor["binding_blob_sha256_by_path"],
    )

    own = protocol["implementation_binding"]["preflight_group"]
    implementation = own["implementation_commit"]
    _verify_commit(
        root,
        commit=implementation,
        expected_parent=predecessor["binding_commit"],
        changed_paths=EXACT3,
        blobs={
            SCRIPT_PATH: own["implementation_script_sha256"],
            TEST_PATH: own["implementation_test_sha256"],
            CONFIG_PATH: hashlib.sha256(_git_blob(root, implementation, CONFIG_PATH)).hexdigest(),
        },
    )
    implementation_protocol = _strict_json(
        _git_blob(root, implementation, CONFIG_PATH), label="implementation protocol"
    )
    if implementation_protocol != _normalise_preflight_binding(protocol):
        raise BindingNotFrozen("binding commit changed more than the four preflight scalars")
    if hashlib.sha256((root / SCRIPT_PATH).read_bytes()).hexdigest() != own[
        "implementation_script_sha256"
    ]:
        raise BindingNotFrozen("current script differs from the bound implementation")
    if hashlib.sha256((root / TEST_PATH).read_bytes()).hexdigest() != own[
        "implementation_test_sha256"
    ]:
        raise BindingNotFrozen("current focused test differs from the bound implementation")
    current_protocol = _strict_json(protocol_bytes, label="current protocol")
    if current_protocol != protocol:
        raise BindingNotFrozen("protocol bytes changed during binding audit")

    head = _run_git(root, "rev-parse", "HEAD")
    if _run_git(root, "rev-parse", f"{head}^") != implementation:
        raise BindingNotFrozen("current binding commit is not the direct child of implementation")
    if _changed_paths(root, head) != (CONFIG_PATH,):
        raise BindingNotFrozen("current binding commit is not config-only")
    branch = protocol["repository_authority"]["branch"]
    upstream = protocol["repository_authority"]["upstream_ref"]
    if _run_git(root, "branch", "--show-current") != branch:
        raise BindingNotFrozen("production branch differs")
    if _run_git(root, "status", "--porcelain"):
        raise BindingNotFrozen("production worktree is not clean")
    if _run_git(root, "rev-parse", upstream) != head:
        raise BindingNotFrozen("HEAD differs from the frozen upstream ref")
    if _run_git(root, "rev-parse", f"origin/{branch}") != head:
        raise BindingNotFrozen("HEAD differs from origin branch")
    return {
        "status": "BOUND_CLEAN_HEAD_EQUALS_UPSTREAM_EQUALS_ORIGIN",
        "authority_commit": AUTHORITY_COMMIT,
        "nonauthoritative_a6_g0_lineage_commit": A6_G0_COMMIT,
        "authority_runtime_binding_commit": runtime["binding_commit"],
        "current_projection_commit": projection["commit"],
        "gse261709_predecessor_binding_commit": predecessor["binding_commit"],
        "preflight_implementation_commit": implementation,
        "preflight_binding_commit": head,
        "binding_diff_is_config_only": True,
    }


def _read_bound_asset_bytes(
    path: Path, specification: Mapping[str, Any], *, label: str
) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AssetError(f"{label} could not be read") from exc
    if len(payload) != specification["byte_count"]:
        raise AssetError(f"{label} byte identity differs")
    if hashlib.sha256(payload).hexdigest() != specification["sha256"]:
        raise AssetError(f"{label} digest identity differs")
    return payload


_OOXML_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OOXML_DOC_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_OOXML_PACKAGE_REL_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)


def _xlsx_cell_column(reference: str) -> int:
    match = re.fullmatch(r"([A-Z]+)[1-9][0-9]*", reference)
    if match is None:
        raise AssetError("publisher Table S5 contains an invalid cell reference")
    column = 0
    for character in match.group(1):
        column = column * 26 + ord(character) - ord("A") + 1
    return column - 1


def _xlsx_text(
    cell: ElementTree.Element, shared_strings: tuple[str, ...]
) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or ""
            for node in cell.iter(f"{{{_OOXML_MAIN_NS}}}t")
        )
    value = cell.find(f"{{{_OOXML_MAIN_NS}}}v")
    text = "" if value is None or value.text is None else value.text
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except (IndexError, ValueError) as exc:
            raise AssetError("publisher Table S5 shared-string index is invalid") from exc
    return text


def _parse_publisher_table_s5(payload: bytes) -> list[dict[str, str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            relationships = ElementTree.fromstring(
                archive.read("xl/_rels/workbook.xml.rels")
            )
            relationship_targets = {
                relation.get("Id"): relation.get("Target")
                for relation in relationships.findall(
                    f"{{{_OOXML_PACKAGE_REL_NS}}}Relationship"
                )
            }
            matching_sheets = [
                sheet
                for sheet in workbook.findall(
                    f".//{{{_OOXML_MAIN_NS}}}sheet"
                )
                if sheet.get("name") == PUBLISHER_TABLE_S5_SPEC["required_sheet"]
            ]
            if len(matching_sheets) != 1:
                raise AssetError("publisher Table S5 required sheet is not unique")
            relationship_id = matching_sheets[0].get(
                f"{{{_OOXML_DOC_REL_NS}}}id"
            )
            target = relationship_targets.get(relationship_id)
            if not target:
                raise AssetError("publisher Table S5 sheet relationship is missing")
            worksheet_path = target.lstrip("/")
            if not worksheet_path.startswith("xl/"):
                worksheet_path = f"xl/{worksheet_path}"

            shared_strings: tuple[str, ...] = ()
            if "xl/sharedStrings.xml" in archive.namelist():
                shared_root = ElementTree.fromstring(
                    archive.read("xl/sharedStrings.xml")
                )
                shared_strings = tuple(
                    "".join(
                        node.text or ""
                        for node in item.iter(f"{{{_OOXML_MAIN_NS}}}t")
                    )
                    for item in shared_root.findall(
                        f"{{{_OOXML_MAIN_NS}}}si"
                    )
                )
            worksheet = ElementTree.fromstring(archive.read(worksheet_path))
    except AssetError:
        raise
    except (
        KeyError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
        ElementTree.ParseError,
    ) as exc:
        raise AssetError("publisher Table S5 is not the bound OOXML workbook") from exc

    dimension = worksheet.find(f"{{{_OOXML_MAIN_NS}}}dimension")
    if dimension is None or dimension.get("ref") != PUBLISHER_TABLE_S5_SPEC[
        "required_sheet_dimension"
    ]:
        raise AssetError("publisher Table S5 sheet dimension differs")
    sheet_data = worksheet.find(f"{{{_OOXML_MAIN_NS}}}sheetData")
    if sheet_data is None:
        raise AssetError("publisher Table S5 sheet data is missing")
    rows: list[tuple[str, ...]] = []
    for row in sheet_data.findall(f"{{{_OOXML_MAIN_NS}}}row"):
        values = [""] * len(TABLE_S5_HEADER)
        seen_columns: set[int] = set()
        for cell in row.findall(f"{{{_OOXML_MAIN_NS}}}c"):
            column = _xlsx_cell_column(cell.get("r", ""))
            if column >= len(values) or column in seen_columns:
                raise AssetError("publisher Table S5 row shape differs")
            seen_columns.add(column)
            values[column] = _xlsx_text(cell, shared_strings).strip()
        rows.append(tuple(values))
    if not rows or rows[0] != TABLE_S5_HEADER:
        raise AssetError("publisher Table S5 exact header differs")
    records = [dict(zip(TABLE_S5_HEADER, row)) for row in rows[1:]]
    if len(records) != PUBLISHER_TABLE_S5_SPEC["exact_data_row_count"]:
        raise AssetError("publisher Table S5 data-row count differs")
    return records


def _positive_integer(value: str, *, label: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise AssetError(f"{label} is not a positive integer")
    return int(value)


def _nonnegative_integer(value: str, *, label: str) -> int:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        raise AssetError(f"{label} is not a nonnegative integer")
    return int(value)


def _small_count_bin(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 2:
        return str(value)
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    return "11+"


def _edit_bin(value: int) -> str:
    if value <= 2:
        return str(value)
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    return "11+"


def _analyse_publisher_table(records: list[dict[str, str]]) -> dict[str, Any]:
    by_member_key: dict[str, dict[str, str]] = {}
    duplicate_member_key_count = 0
    design_to_constructs: dict[tuple[str, ...], set[str]] = defaultdict(set)
    design_to_declared_count: dict[tuple[str, ...], set[int]] = defaultdict(set)
    design_to_observed_count: Counter[tuple[str, ...]] = Counter()
    family_to_designs: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    family_to_sources: dict[tuple[str, ...], set[str]] = defaultdict(set)

    for record in records:
        sequence = record["barcoded_seq_184bp"].upper()
        if len(sequence) != 184 or re.fullmatch(r"[ACGT]{184}", sequence) is None:
            raise AssetError("publisher Table S5 has an invalid barcoded sequence")
        member_key = sequence[:20]
        if member_key in by_member_key:
            duplicate_member_key_count += 1
        else:
            by_member_key[member_key] = record
        construct = sequence[20:]
        family = (record["gene_id"], record["pas_id"])
        design = family + (record["experiment"],)
        declared_count = _positive_integer(record["n_bc"], label="Table S5 n_bc")
        design_to_constructs[design].add(construct)
        design_to_declared_count[design].add(declared_count)
        design_to_observed_count[design] += 1
        family_to_designs[family].add(design)
        if record["experiment"] == "wt":
            family_to_sources[family].add(construct)

    inconsistent_design_construct_count = sum(
        len(constructs) != 1 for constructs in design_to_constructs.values()
    )
    n_bc_mismatch_design_count = 0
    declared_member_total = 0
    for design, observed_count in design_to_observed_count.items():
        declared_values = design_to_declared_count[design]
        if len(declared_values) != 1:
            n_bc_mismatch_design_count += 1
            continue
        declared_count = next(iter(declared_values))
        declared_member_total += declared_count
        if declared_count != observed_count:
            n_bc_mismatch_design_count += 1

    candidate_count_by_family: Counter[tuple[str, ...]] = Counter()
    candidate_designs: list[tuple[str, ...]] = []
    for family, designs in family_to_designs.items():
        for design in designs:
            if design[-1] != "wt":
                candidate_designs.append(design)
                candidate_count_by_family[family] += 1

    missing_source_family_count = sum(
        len(family_to_sources.get(family, set())) == 0 for family in family_to_designs
    )
    ambiguous_source_family_count = sum(
        len(family_to_sources.get(family, set())) > 1 for family in family_to_designs
    )
    below_minimum_candidate_family_count = sum(
        candidate_count_by_family[family] < 3 for family in family_to_designs
    )
    invalid_family_count = sum(
        len(family_to_sources.get(family, set())) != 1
        or candidate_count_by_family[family] < 3
        for family in family_to_designs
    )
    family_size_histogram = Counter(
        _small_count_bin(candidate_count_by_family[family])
        for family in family_to_designs
    )

    replayable_candidate_count = 0
    source_unanchored_candidate_count = 0
    invalid_length_or_alphabet_count = 0
    zero_edit_candidate_count = 0
    edit_histogram: Counter[str] = Counter()
    for design in candidate_designs:
        family = design[:-1]
        sources = family_to_sources.get(family, set())
        constructs = design_to_constructs[design]
        if len(sources) != 1:
            source_unanchored_candidate_count += 1
            continue
        if len(constructs) != 1:
            invalid_length_or_alphabet_count += 1
            continue
        source = next(iter(sources))
        candidate = next(iter(constructs))
        if (
            len(source) != 164
            or len(candidate) != 164
            or re.fullmatch(r"[ACGT]{164}", source) is None
            or re.fullmatch(r"[ACGT]{164}", candidate) is None
        ):
            invalid_length_or_alphabet_count += 1
            continue
        edit_count = sum(left != right for left, right in zip(source, candidate))
        if edit_count == 0:
            zero_edit_candidate_count += 1
            continue
        replayable_candidate_count += 1
        edit_histogram[_edit_bin(edit_count)] += 1

    return {
        "by_member_key": by_member_key,
        "member_count": len(records),
        "unique_member_key_count": len(by_member_key),
        "duplicate_member_key_count": duplicate_member_key_count,
        "design_count": len(design_to_constructs),
        "source_family_count": len(family_to_designs),
        "source_anchored_family_count": sum(
            len(family_to_sources.get(family, set())) == 1
            for family in family_to_designs
        ),
        "missing_source_family_count": missing_source_family_count,
        "ambiguous_source_family_count": ambiguous_source_family_count,
        "below_minimum_candidate_family_count": below_minimum_candidate_family_count,
        "invalid_family_count": invalid_family_count,
        "candidate_design_count": len(candidate_designs),
        "family_size_histogram": dict(sorted(family_size_histogram.items())),
        "inconsistent_design_construct_count": inconsistent_design_construct_count,
        "observed_member_count_by_design_histogram": dict(
            sorted(
                Counter(
                    str(count) for count in design_to_observed_count.values()
                ).items()
            )
        ),
        "declared_member_total": declared_member_total,
        "n_bc_mismatch_design_count": n_bc_mismatch_design_count,
        "source_unanchored_candidate_count": source_unanchored_candidate_count,
        "sequence_diff_replayable_candidate_count": replayable_candidate_count,
        "invalid_length_or_alphabet_candidate_count": invalid_length_or_alphabet_count,
        "zero_edit_candidate_count": zero_edit_candidate_count,
        "edit_count_histogram": dict(sorted(edit_histogram.items())),
        "declared_legal_edit_annotation_candidate_count": 0,
    }


def _endpoint_formula_matches(
    total: int, distal: int, proximal: int, reported: str
) -> tuple[bool, str]:
    if total != distal + proximal:
        return False, "COUNT_EQUATION_MISMATCH"
    if distal == 0 and proximal == 0:
        return reported == "NA", "UNDEFINED_ZERO_TOTAL"
    if distal == 0:
        return reported == "Inf", "POSITIVE_INFINITY"
    if proximal == 0:
        return reported == "-Inf", "NEGATIVE_INFINITY"
    try:
        observed = float(reported)
    except ValueError:
        return False, "FINITE"
    expected = math.log2(proximal / distal)
    return (
        math.isfinite(observed)
        and math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12),
        "FINITE",
    )


def _parse_official_mpra(
    payload: bytes, table_by_member_key: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    context_index = {
        (sample, distal): index
        for index, (sample, distal) in enumerate(
            (sample, distal)
            for sample in EXPECTED_SAMPLE_FIELDS
            for distal in EXPECTED_DISTAL_CONTEXTS
        )
    }
    complete_context_mask = (1 << len(context_index)) - 1
    context_masks: dict[str, int] = defaultdict(int)
    context_row_counts: Counter[str] = Counter()
    seen_member_keys: set[str] = set()
    seen_samples: set[str] = set()
    seen_distal_contexts: set[str] = set()
    seen_perturbations: set[str] = set()
    unmatched_processed_row_count = 0
    join_crosscheck_mismatch_row_count = 0
    subtype_refinement_row_count = 0
    subtype_refinement_member_keys: set[str] = set()
    subtype_refinement_classes: set[str] = set()
    context_duplicate_row_count = 0
    context_label_mismatch_row_count = 0
    formula_mismatch_row_count = 0
    endpoint_class_counts: Counter[str] = Counter()
    row_count = 0

    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                header = handle.readline().rstrip("\r\n")
                if header != " ".join(MPRA_HEADER):
                    raise AssetError("official MPRA exact whitespace header differs")
                for line in handle:
                    row_count += 1
                    values = line.split()
                    if len(values) != len(MPRA_HEADER):
                        raise AssetError("official MPRA row width differs")
                    row = dict(zip(MPRA_HEADER, values))
                    member_key = row["barcode"]
                    record = table_by_member_key.get(member_key)
                    if record is None:
                        unmatched_processed_row_count += 1
                    else:
                        seen_member_keys.add(member_key)
                        if (
                            row["gene_id"] != record["gene_id"]
                            or row["pas_id"] != record["pas_id"]
                            or row["experiment"] != record["experiment"]
                            or row["n_bc"] != record["n_bc"]
                            or row["aim"] != record["type"]
                        ):
                            join_crosscheck_mismatch_row_count += 1
                        if row["subaim"] != record["subtype"]:
                            subtype_refinement_row_count += 1
                            if record["subtype"] == "none":
                                subtype_refinement_member_keys.add(member_key)
                                subtype_refinement_classes.add(row["subaim"])

                    sample = row["sample"]
                    distal_context = row["distal_site"]
                    seen_samples.add(sample)
                    seen_distal_contexts.add(distal_context)
                    seen_perturbations.add(row["perturbation"])
                    expected_fields = EXPECTED_SAMPLE_FIELDS.get(sample)
                    context_bit = context_index.get((sample, distal_context))
                    if (
                        expected_fields is None
                        or expected_fields
                        != (row["replicate"], row["perturbation"])
                        or context_bit is None
                    ):
                        context_label_mismatch_row_count += 1
                    elif member_key in table_by_member_key:
                        bit = 1 << context_bit
                        if context_masks[member_key] & bit:
                            context_duplicate_row_count += 1
                        context_masks[member_key] |= bit
                        context_row_counts[member_key] += 1

                    _positive_integer(row["n_bc"], label="official MPRA n_bc")
                    total = _nonnegative_integer(row["total"], label="official MPRA total")
                    distal = _nonnegative_integer(
                        row["distal"], label="official MPRA distal count"
                    )
                    proximal = _nonnegative_integer(
                        row["proximal"], label="official MPRA proximal count"
                    )
                    formula_matches, endpoint_class = _endpoint_formula_matches(
                        total, distal, proximal, row["log_odds"]
                    )
                    endpoint_class_counts[endpoint_class] += 1
                    if not formula_matches:
                        formula_mismatch_row_count += 1
    except AssetError:
        raise
    except (OSError, UnicodeError, gzip.BadGzipFile, EOFError) as exc:
        raise AssetError("official MPRA gzip table could not be parsed") from exc

    if row_count != OFFICIAL_MPRA_SPEC["exact_data_row_count"]:
        raise AssetError("official MPRA data-row count differs")
    incomplete_context_member_count = sum(
        context_masks.get(member_key, 0) != complete_context_mask
        or context_row_counts.get(member_key, 0) != len(context_index)
        for member_key in table_by_member_key
    )
    return {
        "row_count": row_count,
        "distinct_joined_member_count": len(seen_member_keys),
        "unmatched_processed_row_count": unmatched_processed_row_count,
        "unseen_publisher_member_count": len(table_by_member_key) - len(seen_member_keys),
        "join_crosscheck_mismatch_row_count": join_crosscheck_mismatch_row_count,
        "subtype_refinement_row_count": subtype_refinement_row_count,
        "subtype_refinement_member_count": len(subtype_refinement_member_keys),
        "subtype_refinement_class_count": len(subtype_refinement_classes),
        "sample_count": len(seen_samples),
        "perturbation_count": len(seen_perturbations),
        "distal_reporter_context_count": len(seen_distal_contexts),
        "context_duplicate_row_count": context_duplicate_row_count,
        "context_label_mismatch_row_count": context_label_mismatch_row_count,
        "incomplete_context_member_count": incomplete_context_member_count,
        "formula_mismatch_row_count": formula_mismatch_row_count,
        "endpoint_class_counts": dict(sorted(endpoint_class_counts.items())),
    }


def fisher_power(n: int, rho: float, alpha: float) -> float:
    if n <= 3:
        return 0.0
    null_se = 1.0 / math.sqrt(n - 3.0)
    alternative_se = math.sqrt(1.0 + rho**2 / 2.0) * null_se
    alternative_z = math.atanh(rho)
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0) * null_se
    return (
        1.0 - NormalDist().cdf((critical - alternative_z) / alternative_se)
        + NormalDist().cdf((-critical - alternative_z) / alternative_se)
    )


def fisher_ci_width(n: int, rho: float, confidence: float) -> float:
    if n <= 3:
        return 2.0
    critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    center = math.atanh(rho)
    alternative_se = math.sqrt(1.0 + rho**2 / 2.0) / math.sqrt(n - 3.0)
    radius = critical * alternative_se
    return math.tanh(center + radius) - math.tanh(center - radius)


def required_effective_n(
    *, rho: float, alpha: float, target_power: float, confidence: float, max_width: float
) -> int:
    for n in range(4, 10001):
        if fisher_power(n, rho, alpha) >= target_power and fisher_ci_width(
            n, rho, confidence
        ) <= max_width:
            return n
    raise ProtocolError("prefrozen power target has no bounded solution")


def _gate(status: str, reason: str) -> dict[str, str]:
    if status not in {PASS, FAIL, UNKNOWN}:
        raise AssertionError("invalid internal gate status")
    return {"status": status, "reason": reason}


def aggregate(
    protocol: Mapping[str, Any], official_mpra_asset: Path, publisher_table_s5_asset: Path
) -> dict[str, Any]:
    mpra_payload = _read_bound_asset_bytes(
        official_mpra_asset, OFFICIAL_MPRA_SPEC, label="official MPRA asset"
    )
    table_payload = _read_bound_asset_bytes(
        publisher_table_s5_asset,
        PUBLISHER_TABLE_S5_SPEC,
        label="publisher Table S5 asset",
    )
    # Both byte identities are now closed.  No parser call is permitted above.
    table_records = _parse_publisher_table_s5(table_payload)
    table = _analyse_publisher_table(table_records)
    mpra = _parse_official_mpra(mpra_payload, table["by_member_key"])

    join_closed = (
        table["duplicate_member_key_count"] == 0
        and mpra["unmatched_processed_row_count"] == 0
        and mpra["unseen_publisher_member_count"] == 0
        and mpra["join_crosscheck_mismatch_row_count"] == 0
        and mpra["context_duplicate_row_count"] == 0
        and mpra["incomplete_context_member_count"] == 0
    )
    family_closed = (
        table["source_family_count"] > 0
        and table["invalid_family_count"] == 0
        and table["inconsistent_design_construct_count"] == 0
    )
    replay_closed = (
        table["candidate_design_count"] > 0
        and table["source_unanchored_candidate_count"] == 0
        and table["invalid_length_or_alphabet_candidate_count"] == 0
        and table["zero_edit_candidate_count"] == 0
        and table["declared_legal_edit_annotation_candidate_count"]
        == table["candidate_design_count"]
    )
    nonfinite_or_undefined_endpoint_row_count = sum(
        count
        for endpoint_class, count in mpra["endpoint_class_counts"].items()
        if endpoint_class != "FINITE"
    )
    endpoint_closed = (
        join_closed
        and mpra["context_label_mismatch_row_count"] == 0
        and mpra["sample_count"] == len(EXPECTED_SAMPLE_FIELDS)
        and mpra["distal_reporter_context_count"] == len(EXPECTED_DISTAL_CONTEXTS)
        and mpra["formula_mismatch_row_count"] == 0
        and (
            protocol["assay_endpoint_replicate_policy"]["finite_endpoint_required"]
            is False
            or nonfinite_or_undefined_endpoint_row_count == 0
        )
    )
    schema_closed = (
        join_closed
        and table["member_count"] == PUBLISHER_TABLE_S5_SPEC["exact_data_row_count"]
        and mpra["row_count"] == OFFICIAL_MPRA_SPEC["exact_data_row_count"]
        and table["n_bc_mismatch_design_count"] == 0
        and table["inconsistent_design_construct_count"] == 0
    )

    role_gate = _gate(
        UNKNOWN,
        "A1_AND_TRUE_A2_PRIMARY_ROLE_AUTHORITY_BOTH_UNKNOWN",
    )
    identity_gate = _gate(
        PASS,
        "BOTH_ORDINARY_PUBLIC_CANONICAL_ASSET_BYTE_IDENTITIES_AND_ROLES_MATCHED",
    )
    family_gate = _gate(
        PASS if family_closed else FAIL,
        "EVERY_FAMILY_HAS_ONE_LITERAL_SOURCE_AND_AT_LEAST_THREE_CANDIDATE_DESIGNS"
        if family_closed
        else "LITERAL_SOURCE_ANCHOR_OR_MINIMUM_THREE_CANDIDATE_GEOMETRY_FAILED",
    )
    intronic_gate = _gate(UNKNOWN, "INTRONIC_APA_CONTEXT_NOT_PRESENT_IN_BOUND_ASSETS")
    replay_gate = _gate(
        PASS if replay_closed else FAIL,
        "ALL_CANDIDATE_DESIGNS_HAVE_ANCHORED_DECLARED_EXACT_SUBSTITUTION_REPLAY"
        if replay_closed
        else "SOURCE_ANCHOR_COVERAGE_OR_DECLARED_LEGAL_EDIT_AUTHORITY_FAILED",
    )
    endpoint_gate = _gate(
        PASS if endpoint_closed else FAIL,
        "BOUND_ASSAY_CONTEXTS_AND_COUNT_DERIVED_ENDPOINT_FORMULA_REPLAY_CLOSED"
        if endpoint_closed
        else (
            "NONFINITE_OR_UNDEFINED_ENDPOINT_PRESENT_WITHOUT_AUTHORITATIVE_CENSOR_RULE"
            if mpra["formula_mismatch_row_count"] == 0
            and nonfinite_or_undefined_endpoint_row_count > 0
            else "BOUND_ASSAY_CONTEXT_OR_ENDPOINT_FORMULA_REPLAY_FAILED"
        ),
    )
    replicate_gate = _gate(
        FAIL,
        "LABELS_DO_NOT_ESTABLISH_BIOLOGICAL_INDEPENDENCE_AND_REPORTED_SE_IS_ABSENT",
    )
    schema_gate = _gate(
        PASS if schema_closed else FAIL,
        "BOTH_ASSET_DIMENSIONS_JOIN_COVERAGE_AND_DECLARED_MULTIPLICITY_MATCH"
        if schema_closed
        else "DECLARED_MEMBER_MULTIPLICITY_OR_ASSET_COVERAGE_FAILED",
    )
    missing_gate = _gate(
        UNKNOWN,
        "MISSING_CENSORING_AND_OUTCOME_SELECTION_AUTHORITY_NOT_PRESENT",
    )
    exposure_gate = _gate(
        UNKNOWN,
        "APARENT_PRIOR_EXPOSURE_OR_MODEL_INPUT_ROUTE_UNKNOWN",
    )
    rights_gate = _gate(UNKNOWN, "LICENSE_OR_REUSE_RIGHTS_UNKNOWN")
    split_gate = _gate(
        UNKNOWN,
        "OUTCOME_BLIND_NEAR_DUPLICATE_GROUPING_AND_SPLIT_READINESS_UNKNOWN",
    )
    power_gate = _gate(
        UNKNOWN,
        "POST_DEDUP_SOURCE_GROUP_EFFECTIVE_N_NOT_IDENTIFIABLE",
    )

    gates = {
        GATE_IDS[0]: role_gate,
        GATE_IDS[1]: identity_gate,
        GATE_IDS[2]: family_gate,
        GATE_IDS[3]: intronic_gate,
        GATE_IDS[4]: replay_gate,
        GATE_IDS[5]: endpoint_gate,
        GATE_IDS[6]: replicate_gate,
        GATE_IDS[7]: schema_gate,
        GATE_IDS[8]: missing_gate,
        GATE_IDS[9]: exposure_gate,
        GATE_IDS[10]: rights_gate,
        GATE_IDS[11]: split_gate,
        GATE_IDS[12]: power_gate,
    }
    if any(gate["status"] == FAIL for gate in gates.values()):
        overall = STATUS_STOP
    elif any(gate["status"] == UNKNOWN for gate in gates.values()):
        overall = STATUS_BLOCKED
    else:
        overall = STATUS_READY

    role_status = (
        "NEITHER_CURRENTLY_JUSTIFIED_NOT_ASSIGNED_NOT_QUALIFIED"
    )

    gate_status_counts = Counter(gate["status"] for gate in gates.values())
    gate_reason_counts = Counter(gate["reason"] for gate in gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "bioproject_id": BIOPROJECT_ID,
        "decision_id": DECISION_ID,
        "protocol_status": protocol["protocol_status"],
        "status": overall,
        "mutually_exclusive_role_status": role_status,
        "geometry_compatibility_observation": (
            "PROVISIONAL_TRUE_A2_CANDIDATE_DENSE_NEIGHBORHOOD_ONLY_"
            "NOT_PRIMARY_ROLE_EVIDENCE_NOT_CREDIT"
        ),
        "aggregate_observation": {
            "asset_schema_geometry": {
                "verified_asset_identity_count": 2,
                "publisher_library_member_count": table["member_count"],
                "publisher_unique_member_key_count": table[
                    "unique_member_key_count"
                ],
                "publisher_duplicate_member_key_count": table[
                    "duplicate_member_key_count"
                ],
                "publisher_design_count": table["design_count"],
                "processed_measurement_row_count": mpra["row_count"],
                "processed_distinct_joined_member_count": mpra[
                    "distinct_joined_member_count"
                ],
                "processed_unmatched_row_count": mpra[
                    "unmatched_processed_row_count"
                ],
                "publisher_unseen_member_count": mpra[
                    "unseen_publisher_member_count"
                ],
                "join_crosscheck_mismatch_row_count": mpra[
                    "join_crosscheck_mismatch_row_count"
                ],
                "complete_context_member_count": table["member_count"]
                - mpra["incomplete_context_member_count"],
                "incomplete_context_member_count": mpra[
                    "incomplete_context_member_count"
                ],
                "context_duplicate_row_count": mpra["context_duplicate_row_count"],
                "generic_design_class_refinement_row_count": mpra[
                    "subtype_refinement_row_count"
                ],
                "generic_design_class_refinement_member_count": mpra[
                    "subtype_refinement_member_count"
                ],
                "generic_design_class_refinement_class_count": mpra[
                    "subtype_refinement_class_count"
                ],
                "declared_member_total": table["declared_member_total"],
                "observed_member_total": table["member_count"],
                "declared_multiplicity_mismatch_design_count": table[
                    "n_bc_mismatch_design_count"
                ],
                "observed_member_count_by_design_histogram": table[
                    "observed_member_count_by_design_histogram"
                ],
                "actual_header_names_reported_count": 0,
            },
            "source_family_geometry": {
                "candidate_source_family_count": table["source_family_count"],
                "literal_source_anchored_family_count": table[
                    "source_anchored_family_count"
                ],
                "literal_source_missing_family_count": table[
                    "missing_source_family_count"
                ],
                "literal_source_ambiguous_family_count": table[
                    "ambiguous_source_family_count"
                ],
                "below_minimum_candidate_family_count": table[
                    "below_minimum_candidate_family_count"
                ],
                "invalid_source_family_count": table["invalid_family_count"],
                "candidate_design_count": table["candidate_design_count"],
                "candidate_count_per_family_histogram": table[
                    "family_size_histogram"
                ],
            },
            "legal_substitution_replay": {
                "candidate_design_count": table["candidate_design_count"],
                "source_unanchored_candidate_count": table[
                    "source_unanchored_candidate_count"
                ],
                "sequence_diff_replayable_candidate_count": table[
                    "sequence_diff_replayable_candidate_count"
                ],
                "invalid_length_or_alphabet_candidate_count": table[
                    "invalid_length_or_alphabet_candidate_count"
                ],
                "zero_edit_candidate_count": table["zero_edit_candidate_count"],
                "declared_legal_edit_annotation_candidate_count": table[
                    "declared_legal_edit_annotation_candidate_count"
                ],
                "edit_count_histogram": table["edit_count_histogram"],
                "row_order_inference_count": 0,
            },
            "assay_endpoint_and_replicate_geometry": {
                "sample_label_count": mpra["sample_count"],
                "perturbation_label_count": mpra["perturbation_count"],
                "distal_reporter_context_count": mpra[
                    "distal_reporter_context_count"
                ],
                "sample_context_label_mismatch_row_count": mpra[
                    "context_label_mismatch_row_count"
                ],
                "count_and_endpoint_formula_mismatch_row_count": mpra[
                    "formula_mismatch_row_count"
                ],
                "endpoint_formula_replay_status": (
                    PASS if mpra["formula_mismatch_row_count"] == 0 else FAIL
                ),
                "nonfinite_or_undefined_endpoint_row_count": (
                    nonfinite_or_undefined_endpoint_row_count
                ),
                "endpoint_value_class_counts": mpra["endpoint_class_counts"],
                "biological_independence_authority_present": False,
                "reported_standard_error_field_present": False,
                "row_endpoint_effect_or_standard_error_reported_count": 0,
                "replicate_identifier_reported_count": 0,
            },
            "split_leakage_and_power_readiness": {
                "near_duplicate_grouping_authority_present": False,
                "post_dedup_independent_source_group_effective_n_identifiable": False,
                "required_effective_n_for_power_and_full_ci_width": protocol[
                    "prefrozen_power_policy"
                ]["required_effective_n_for_both_power_and_ci_width"],
                "analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
                "row_candidate_replicate_or_run_count_used_as_power_n": False,
                "split_assignment_output_count": 0,
            },
        },
        "gates": {gate_id: gates[gate_id] for gate_id in GATE_IDS},
        "aggregate_gate_summary": {
            "status_counts": dict(sorted(gate_status_counts.items())),
            "reason_counts": dict(sorted(gate_reason_counts.items())),
        },
        "internal_access_attestation": {
            "bound_ordinary_public_asset_identity_read_count": 2,
            "caller_supplied_context_asset_read_count": 0,
            "private_or_restricted_asset_read_count": 0,
            "raw_fastq_or_sra_member_payload_read_count": 0,
            "sealed_asset_contact_count": 0,
            "persistent_member_level_intermediate_count": 0,
            "member_identifier_sequence_row_effect_se_or_split_output_count": 0,
            "training_run_count": 0,
            "gpu_run_count": 0,
            "model_selection_count": 0,
        },
        "no_promotion_state": copy.deepcopy(protocol["no_promotion_locks"]),
        "claim_boundary": copy.deepcopy(protocol["claim_boundary"]),
    }


def _default_privacy_validator(report: Mapping[str, Any]) -> None:
    forbidden_keys = set(MPRA_HEADER) | set(TABLE_S5_HEADER) | {
        "member_id",
        "source_sequence",
        "candidate_sequence",
        "endpoint_value",
        "reported_standard_error",
        "biological_replicate_id",
        "split_assignment",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in forbidden_keys:
                    raise OutputError("aggregate report contains a forbidden member field")
                walk(item)
        elif isinstance(value, (list, tuple, set)):
            raise OutputError("aggregate report contains a member-like collection")
        elif isinstance(value, str) and re.fullmatch(r"[ACGT]{20,}", value):
            raise OutputError("aggregate report contains a sequence-like member value")

    walk(report)
    attestation = report.get("internal_access_attestation")
    if not isinstance(attestation, Mapping) or attestation.get(
        "member_identifier_sequence_row_effect_se_or_split_output_count"
    ) != 0:
        raise OutputError("aggregate-only access attestation is not closed")
    if report.get("no_promotion_state") != NO_PROMOTION_LOCKS:
        raise OutputError("aggregate report changes the no-promotion state")


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OutputError("aggregate report is not finite JSON") from exc


def _assert_output_target_available(output: Path) -> None:
    if output.name != REPORT_FILENAME:
        raise OutputError("output basename differs from the frozen sole artifact")
    if output.exists():
        raise OutputError("aggregate output already exists; overwrite is forbidden")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish(output: Path, payload: bytes) -> None:
    _assert_output_target_available(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise OutputError("aggregate output appeared during atomic publish") from exc
        except OSError as exc:
            raise OutputError("aggregate output could not be atomically published") from exc
        Path(temporary).unlink()
        temporary = None
        _fsync_directory(output.parent)
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass


def execute(
    protocol_path: Path,
    official_mpra_asset: Path,
    publisher_table_s5_asset: Path,
    output: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    protocol = _strict_json(protocol_bytes, label="protocol")
    _validate_protocol(protocol)
    root = Path(repo_root or PRODUCTION_REPO_ROOT)
    binding = _default_binding_auditor(protocol, protocol_path, protocol_bytes, root)
    _assert_output_target_available(output)
    report = aggregate(protocol, official_mpra_asset, publisher_table_s5_asset)
    report["binding"] = binding
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _default_privacy_validator(report)
    _atomic_publish(output, _json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--official-mpra-asset", required=True, type=Path)
    parser.add_argument("--publisher-table-s5-asset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = execute(
        arguments.protocol,
        arguments.official_mpra_asset,
        arguments.publisher_table_s5_asset,
        arguments.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "protocol_status": report["protocol_status"],
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
