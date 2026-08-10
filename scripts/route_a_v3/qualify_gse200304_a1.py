#!/usr/bin/env python3
"""Fail-closed aggregate-only GSE200304 Route-A-v3 gap qualifier.

This program audits the six currently present ordinary-public inputs.  It can
only publish an engineering-success bundle whose scientific disposition is
``BLOCKED_NOT_QUALIFIED``.  There is deliberately no activation, canonical
materialization, training, model-selection, network, or row-level output path.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


CONTRACT_ID = "mrna_xeditflow_route_a_v3"
SCHEMA_VERSION = "3.0.0"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_A1_AGGREGATE_GAP_QUALIFICATION_V1"
PROTOCOL_STATUS = "GO_QUALIFICATION_ONLY"
ACTIVATION_STATUS = "NO_GO_ACTIVATION_NOW"
QUALIFICATION_STATUS = "BLOCKED_NOT_QUALIFIED"
DATASET_ID = "GSE200304"
OUTPUT_ID = "ROUTE_A_V3_GSE200304_A1_BLOCKED_GAP_BUNDLE_V1"
SUCCESS_OUTCOME = "ENGINEERING_SUCCESS_BLOCKED_NOT_QUALIFIED"
FAILURE_OUTCOME = "FAIL_CLOSED"
PROTOCOL_BASENAME = "route_a_v3_gse200304_a1_qualification.json"
PUBLICATION_MARKER = "PUBLICATION_COMMIT.json"
SUCCESS_JSON_FILES = (
    "INPUT_INTEGRITY_AUDIT.json",
    "MECHANICAL_AUDIT.json",
    "QUALIFICATION_REPORT.json",
)
FAILURE_JSON_FILES = ("FAILURE_REPORT.json",)
SHA256SUMS_FILENAME = "SHA256SUMS"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MD5_RE = re.compile(r"[0-9a-f]{32}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SEQUENCE_RE = re.compile(r"[ACGT]+")
MAX_DECOMPRESSED_BYTES = 128 * 1024 * 1024
EXPECTED_DATA_ROOT = Path("/mnt/cunyuliu/mrna_editflow_p0/GSE200304")
FORBIDDEN_PATH_TOKENS = (
    "gse246381",
    "access_log",
    "sealed",
    "restricted",
    "sealed_external",
)
EXPECTED_MANIFEST_BYTES = 1115
EXPECTED_MANIFEST_SHA256 = "4a9a0b162f0731df6a5c15441b8984505e2ebaee260ad4e46f62636621125a8c"

EXPECTED_AUTHORITY = {
    "contract_path": "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md",
    "contract_sha256": "3ba224de6277edd67387913cf1c83a5e1344e0ad44ef196db07d0772b45c4d79",
    "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
    "staging_parent_head": "76cf61cddf0e8a0fb9d9055d9384ef55057d6f86",
    "data_role_registry_path": "docs/execution/route_a_v3_data_role_registry.yaml",
    "data_role_registry_sha256": "03a805c6441f0778225f9a8ec10feeadba23f572cd9ea7b234903384e6a902bf",
    "decision_log_path": "docs/execution/route_a_v3_decision_log.yaml",
    "decision_log_sha256": "fc177d710c3a737e860b7a2326bbdabd1e698bc9eea5838cecc362df519d2766",
    "a1_qualification_path": "configs/route_a_v3_a1_qualification.json",
    "a1_qualification_sha256": "1d348671de50c0fe8b155f8cc114d14a74360fe1a87f9d9bac5207ae794806c4",
}

EXPECTED_A1_GATE = {
    "required_qualified_ordinary_studies": 3,
    "required_qualified_a1_intervention_studies": 2,
    "required_qualified_true_a2_dense_studies": 1,
    "current_qualified_ordinary_studies": 0,
    "current_qualified_a1_intervention_studies": 0,
    "current_qualified_true_a2_dense_studies": 0,
    "this_protocol_ordinary_study_contribution": 0,
    "this_protocol_a1_study_contribution": 0,
    "this_protocol_true_a2_study_contribution": 0,
    "gate_pass": False,
}

IMPLEMENTATION_BINDING_UNKNOWN = {
    "status": "UNKNOWN_NOT_ASSERTED",
    "repository_root_rule": "PROTOCOL_PATH_PARENT_PARENT",
    "implementation_commit": "UNKNOWN_NOT_ASSERTED",
    "qualifier_path": "scripts/route_a_v3/qualify_gse200304_a1.py",
    "qualifier_blob_sha256": "UNKNOWN_NOT_ASSERTED",
    "test_path": "tests/route_a_v3/test_qualify_gse200304_a1.py",
    "test_blob_sha256": "UNKNOWN_NOT_ASSERTED",
    "post_implementation_allowed_changed_paths": [
        "configs/route_a_v3_gse200304_a1_qualification.json"
    ],
    "current_head_must_strictly_descend": True,
    "active_authority_must_be_ancestor": True,
    "clean_worktree_required": True,
    "running_script_must_match_qualifier_blob": True,
}

IMPLEMENTATION_BINDING_KEYS = set(IMPLEMENTATION_BINDING_UNKNOWN)

EXPECTED_SCOPE = {
    "ordinary_public_data_only": True,
    "region": "3UTR",
    "member_accessions": ["GSE200304", "GSE200302", "GSE200303", "GSE217530"],
    "maximum_independent_study_count": 1,
    "endpoint_or_member_accession_increases_study_count": False,
    "primary_accession": "GSE200302",
    "primary_measurement_families": ["High_Poly", "Low_Poly", "Total_RNA"],
    "primary_alleles": ["WT", "Mutant"],
    "primary_replicates": [1, 2, 3, 4, 5, 6],
    "qc_auxiliary_measurement_families": [
        "80S_RNA",
        "pDNA",
        "GSE200303_SMALL_PLASMID_FREQ",
        "GSE217530_IVT_TIMECOURSE",
    ],
    "legacy_freq_is_endpoint": False,
    "paper_native_raw_xtail_replay_status": "NOT_RUN",
    "qualified": False,
    "training_allowed": False,
    "model_selection_allowed": False,
    "canonical_materialization_allowed": False,
    "canonical_record_count": 0,
    "raw_rows_or_row_keys_may_be_output": False,
}

EXPECTED_ASSETS: tuple[dict[str, Any], ...] = (
    {
        "asset_id": "GSE200302_DESIGN",
        "accession": "GSE200302",
        "relative_path": "GSE200302/GSE200302_Twist_Oligo_Order_with_merged_ids.txt.gz",
        "bytes": 1091787,
        "sha256": "06b78231dcf02e6d42bc0abaf919d419630a6e1dd33d04ebe841ff03aa0e5f1f",
        "format": "GZIP_TSV",
        "role": "PRIMARY_DESIGN",
    },
    {
        "asset_id": "GSE200302_PROCESSED",
        "accession": "GSE200302",
        "relative_path": "GSE200302/GSE200302_log2_cpm_counts_all_samples.txt.gz",
        "bytes": 2843042,
        "sha256": "ed93162f9540676138cfba05af2841c90619ac4335eb55ee3d956a3cd8aace3c",
        "format": "GZIP_TSV",
        "role": "PRIMARY_PROCESSED_COMPANION",
    },
    {
        "asset_id": "GSE200303_RAW_TAR",
        "accession": "GSE200303",
        "relative_path": "GSE200303/GSE200303_RAW.tar",
        "bytes": 1187840,
        "sha256": "ff761e86c682cce13e95bea89951f98d167d3dc13d875e89825e97f366bc7617",
        "format": "TAR",
        "role": "AUXILIARY_ARCHIVE_BYTE_EQUIVALENCE",
    },
    {
        "asset_id": "GSE200303_DESIGN",
        "accession": "GSE200303",
        "relative_path": "GSE200303/GSM6030637_Twist_Oligo_Order_with_merged_ids.txt.gz",
        "bytes": 1091788,
        "sha256": "cc8c1b69f87e669bb45cdc34257e489352dad4659d193d9a9e29166a6e7d2d75",
        "format": "GZIP_TSV",
        "role": "AUXILIARY_DESIGN_QC",
    },
    {
        "asset_id": "GSE200303_SMALL_PLASMID",
        "accession": "GSE200303",
        "relative_path": "GSE200303/GSM6030637_log2_cpm_small_seq_on_plasmid.txt.gz",
        "bytes": 86041,
        "sha256": "54bc28cd55959d8b4ddfab4d2ea7126654256fe115fbcc363d141d9ff5d9a216",
        "format": "GZIP_TSV",
        "role": "AUXILIARY_QC_ONLY_LEGACY_FREQ",
    },
    {
        "asset_id": "GSE217530_IVT",
        "accession": "GSE217530",
        "relative_path": "GSE217530/GSE217530_log2_cpm_IVT_librray_April2022.txt.gz",
        "bytes": 3114235,
        "sha256": "570edec59bdfcbe56d34f221700dc0e5a3d419561c5f693d2c9d705d26ecf320",
        "format": "GZIP_TSV",
        "role": "AUXILIARY_QC_ONLY_IVT",
    },
)

EXPECTED_TAR_EQUIVALENCE = [
    {
        "member_name": "GSM6030637_Twist_Oligo_Order_with_merged_ids.txt.gz",
        "direct_asset_id": "GSE200303_DESIGN",
    },
    {
        "member_name": "GSM6030637_log2_cpm_small_seq_on_plasmid.txt.gz",
        "direct_asset_id": "GSE200303_SMALL_PLASMID",
    },
]

EXPECTED_DESIGN_CONTRACT = {
    "exact_header": ["ID", "Type", "201bp", "5' End", "3'End", "Full_Oligo", "merged_id"],
    "row_count": 13836,
    "type_counts": {"WT": 6885, "Mutant": 6885, "Control": 66},
    "unique_pair_count": 6885,
    "distinct_candidate_count": 6885,
    "distinct_wt_source_group_count": 6882,
    "singleton_source_pool_count": 6879,
    "two_candidate_source_pool_count": 3,
    "three_or_more_candidate_source_pool_count": 0,
    "ndcg_eligible_source_pool_count": 0,
    "control_id_count": 66,
    "pair_multiplicity": "EXACTLY_ONE_WT_AND_ONE_MUTANT",
    "control_multiplicity": "EXACTLY_ONE_ROW_PER_CONTROL_ID",
    "sequence_column": "201bp",
    "sequence_length": 201,
    "sequence_alphabet": "ACGT",
    "edit_rule": "EXACTLY_ONE_SNV",
    "identity_rule": "ID_EQUALS_STRIP_WT_OR_MUTANT_SUFFIX_FROM_MERGED_ID",
}

EXPECTED_PROCESSED_CONTRACT = {
    "key_column": "barcode",
    "allele_order": ["WT", "Mutant"],
    "measurement_family_order": ["80S_RNA", "High_Poly", "Low_Poly", "pDNA", "Total_RNA"],
    "sample_numbers_by_family": {
        "80S_RNA": [2, 6, 10, 14, 18, 22],
        "High_Poly": [4, 8, 12, 16, 20, 24],
        "Low_Poly": [3, 7, 11, 15, 19, 23],
        "pDNA": [25, 26, 27, 28, 29, 30],
        "Total_RNA": [1, 5, 9, 13, 17, 21],
    },
    "replicates": [1, 2, 3, 4, 5, 6],
    "row_count": 6772,
    "measurement_column_count": 60,
    "all_values_finite_and_nonmissing": True,
    "unique_key_required": True,
    "exact_design_pair_join_required": True,
    "outcome_blind_attrition_count": 113,
    "freq_column_allowed": False,
}

EXPECTED_SMALL_CONTRACT = {
    "exact_header": ["Barcode", "Freq"],
    "row_count": 12704,
    "complete_pair_count": 6120,
    "wt_only_pair_count": 192,
    "mutant_only_pair_count": 225,
    "neither_pair_count": 348,
    "control_row_count": 47,
    "unique_barcode_required": True,
    "all_numeric_values_finite_and_nonmissing": True,
    "role": "AUXILIARY_QC_ONLY_NOT_ENDPOINT",
}

EXPECTED_IVT_CONTRACT = {
    "key_column": "ids",
    "timepoint_order": ["12hr", "1hr", "24hr", "3hr", "6hr"],
    "sample_numbers_by_timepoint": {
        "12hr": [43, 44, 45, 46, 47, 48],
        "1hr": [25, 26, 27, 28, 29, 30],
        "24hr": [49, 50, 51, 52, 53, 54],
        "3hr": [31, 32, 33, 34, 35, 36],
        "6hr": [37, 38, 39, 40, 41, 42],
    },
    "replicates": [1, 2, 3, 4, 5, 6],
    "row_count": 13548,
    "complete_pair_count": 6774,
    "wt_only_pair_count": 0,
    "mutant_only_pair_count": 0,
    "missing_both_pair_count": 111,
    "control_row_count": 0,
    "measurement_column_count": 30,
    "unique_key_required": True,
    "all_values_finite_and_nonmissing": True,
    "role": "AUXILIARY_QC_ONLY_NOT_ENDPOINT",
}

EXPECTED_PAPER_EVIDENCE = {
    "pmc_id": "PMC10540565",
    "pubmed_id": "37516102",
    "paper_native_raw_xtail_replay_status": "NOT_RUN",
    "author_notebook_wt_df_mut_df_status": "UNDEFINED_VARIABLES",
    "author_notebook_descriptive_mutant_rep3_status": "ERRONEOUSLY_REUSES_REP2",
    "author_notebook_code_copy_allowed": False,
    "sam_to_count_lineage_status": "ABSENT",
    "checkpoint_specific_exposure_status": "UNKNOWN_NOT_ASSERTED",
    "outcome_blind_split_and_power_status": "NOT_FROZEN",
    "precomputed_gap_evidence_only": {
        "s3_mutation_unique_barcodes": 6772,
        "s3_comparison_count": 2,
        "s3_comparisons": ["HighPoly:RNA", "TotalPoly:RNA"],
        "s3_complete_stats_high_poly": 6538,
        "s3_complete_stats_total_poly": 6547,
        "s3_gene_nonmissing_rows": 13544,
        "s3_unique_genes": 1947,
        "companion_total_poly_spearman": 0.9556,
        "companion_total_poly_sign_agreement": 0.9462,
        "companion_high_poly_spearman": 0.9329,
        "companion_high_poly_sign_agreement": 0.9351,
        "companion_se_finite_positive_count": 6772,
        "status": "EVIDENCE_ONLY_NOT_A_PAPER_NATIVE_REPLAY_PASS",
    },
    "license_evidence": {
        "pmc_oa_api_declaration": "CC BY",
        "zenodo_code_archive_license_metadata": "other-open",
        "repository_explicit_license_file_status": "ABSENT",
        "artifact_specific_code_license_status": "UNKNOWN_FAIL_CLOSED",
    },
    "public_asset_bundle_lineage": {
        "bundle_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304/GSE200304_PUBLIC_ASSETS_20260810T143731P0800",
        "acquisition_manifest_sha256": "8318990d9e3b6a0e6265bf9d1e8bc20f56f0ecfd994e83d279e733258642100c",
        "sha256sums_sha256": "20da85cd34f0574829392b5de1d7c48cc9782219847f56ccc07dffd579d79f15",
        "publication_commit_sha256": "4742508195f28bf8c7ab1f7cb8bb0b68c32304f31b19c8f8979d098fa75786a5",
        "status": "PRESENT_IN_SEPARATE_COMMITTED_BUNDLE_NOT_INTEGRATED",
    },
    "ena_fastq_manifest_bundle_lineage": {
        "bundle_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304/GSE200304_ENA_FASTQ_MANIFEST_20260810T145631P0800",
        "canonical_tsv_sha256": "22cd317d961d07036cb2dad19555b5c2423671c33a76badeb7b325847ee68d7b",
        "source_tsv_sha256": "c4a0b6152ec2a3480f280d8498345196d5095ec54967525463fa81961f0f4ea1",
        "summary_sha256": "f92f944c825a255f3f1fb50f48cbf0e701980b7895101c1a2a6699d4b190e1e4",
        "terminal_marker_sha256": "d3eed4a9408543c77f47aa2a0d8cff59ebfe863c1e3c2d0bb2324d7910d6014b",
        "official_metadata_and_object_lengths_status": "VERIFIED_48_OBJECTS",
        "fastq_body_download_count": 0,
        "fastq_md5_local_recomputation_status": "NOT_RUN",
        "used_by_current_qualifier": False,
        "status": "PRESENT_IN_SEPARATE_COMMITTED_BUNDLE_NOT_CONSUMED",
    },
}

EXPECTED_A1_MASTER_REPORT_CONTRACT = {
    "required_report_fields": [
        "nominal_rows",
        "distinct_candidates",
        "biological_source_groups",
        "gene_groups",
        "study_groups",
        "eligible_multi_candidate_pools",
        "edit_count_strata",
        "replicate_and_se_coverage",
        "beneficial_and_noise_zone_balance",
        "post_dedup_effective_n",
        "foundation_exposure",
        "license_and_redistribution_status",
    ],
    "study_recovery_method": "SOURCE_CANDIDATE_ENDPOINT_CONTEXT_AND_GROUP_AUDIT",
    "metadata_only_recovery_allowed": False,
    "minimum_ndcg_pool_size": 3,
    "two_candidate_pool_use": "PAIRWISE_ONLY",
    "unknown_status_is_hard_block": True,
}

EXPECTED_FUTURE_ASSETS = [
    {
        "asset_id": "PMC_TABLE_S2",
        "locator": "https://pmc-oa-opendata.s3.amazonaws.com/PMC10540565.1/NIHMS1928233-supplement-3.csv",
        "filename": "NIHMS1928233-supplement-3.csv",
        "bytes": 7323186,
        "sha256": "812f3c983cb7c4f473200741ffd6d73bcab911c9e354934542e018e7b0cf8a6d",
        "shape_rows": 13850,
        "shape_columns": 6,
        "exact_header": ["ID", "Type", "201bp", "5' End", "3'End", "Full_Oligo"],
        "production_server_status": "PRESENT_IN_SEPARATE_COMMITTED_BUNDLE",
        "used_by_current_qualifier": False,
    },
    {
        "asset_id": "PMC_TABLE_S3",
        "locator": "https://pmc-oa-opendata.s3.amazonaws.com/PMC10540565.1/NIHMS1928233-supplement-4.xlsx",
        "filename": "NIHMS1928233-supplement-4.xlsx",
        "bytes": 864791,
        "sha256": "ec2aab60fcb0be87f2bcc1b1a5a1f786b23bb429edc9851a4034a3e8983dfa08",
        "sheets": [
            {"name": "S2A_Polysome_MPRA_Mut_Stats", "shape_rows": 13544, "shape_columns": 7},
            {"name": "S2B_Poly_MPRA_Control_Stats", "shape_rows": 29, "shape_columns": 13},
        ],
        "join_rule": "S3_BARCODE_EQUALS_DESIGN_ID",
        "production_server_status": "PRESENT_IN_SEPARATE_COMMITTED_BUNDLE",
        "used_by_current_qualifier": False,
    },
    {
        "asset_id": "ZENODO_CODE_ARCHIVE_V1_2",
        "locator": "https://doi.org/10.5281/zenodo.8007705",
        "filename": "slschuster/3UTRMutationalMPRA-v1.2.zip",
        "bytes": 46209,
        "provider_md5": "e20ae1ffbd05a5882fe7b3bb7c4700ca",
        "independent_sha256": "1c1b1979c1d5bd7fefa54e80a59f982228d0f1498eb0cff2883b753ee5eb0ae4",
        "contains_data": False,
        "explicit_code_license_status": "UNKNOWN_NOT_ASSERTED",
        "production_server_status": "PRESENT_IN_SEPARATE_COMMITTED_BUNDLE",
        "used_by_current_qualifier": False,
    },
]

EXPECTED_BLOCKERS = (
    "CURRENT_MANIFEST_NOT_BOUND_TO_FULL_SUPERSERIES",
    "FASTQ_CONTENT_INTEGRITY_NOT_VERIFIED",
    "RAW_COUNTS_ABSENT",
    "PUBLIC_ASSETS_ACQUIRED_NOT_INTEGRATED_IN_CURRENT_GAP_QUALIFIER",
    "ZENODO_EXPLICIT_CODE_LICENSE_UNKNOWN",
    "PAPER_NATIVE_RAW_XTAIL_REPLAY_NOT_RUN",
    "SAM_TO_COUNT_LINEAGE_ABSENT",
    "CHECKPOINT_SPECIFIC_EXPOSURE_UNKNOWN_NOT_ASSERTED",
    "OUTCOME_BLIND_SPLIT_AND_POWER_NOT_FROZEN",
    "CANONICAL_V3_NOT_MATERIALIZED",
    "REDISTRIBUTION_LOCATOR_HASH_AGGREGATE_ONLY_UNTIL_LICENSE_CLOSED",
)

IMPLEMENTATION_BINDING_BLOCKER = "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED"

EXPECTED_OUTPUT_CONTRACT = {
    "output_id": OUTPUT_ID,
    "aggregate_only": True,
    "closed_schema_required": True,
    "success_outcome": SUCCESS_OUTCOME,
    "failure_outcome": FAILURE_OUTCOME,
    "success_and_failure_mutually_exclusive": True,
    "success_files": [*SUCCESS_JSON_FILES, SHA256SUMS_FILENAME],
    "failure_files": [*FAILURE_JSON_FILES, SHA256SUMS_FILENAME],
    "terminal_marker": PUBLICATION_MARKER,
    "terminal_marker_written_last": True,
    "terminal_marker_binds_absolute_target_and_basename": True,
    "publication_state_enum": [
        "PARTIAL_PRECOMMIT",
        "COMMITTED_NOT_ACCEPTED",
        "COMMITTED_WITH_DURABILITY_WARNING",
        "COMMITTED_ACCEPTED",
    ],
    "directory_creation_mode": "ATOMIC_EXCLUSIVE_MKDIR",
    "member_write_mode": "O_EXCL_FILE_FSYNC",
    "no_overwrite": True,
    "canonical_or_row_level_output_path": "HARD_DISABLED_NOT_IMPLEMENTED",
    "allowed_output_information_classes": [
        "FIXED_PROTOCOL_ENUM",
        "AGGREGATE_COUNT",
        "BOOLEAN",
        "SHA256",
        "BYTE_COUNT",
        "PUBLIC_LOCATOR",
        "ACCESSION",
        "FIXED_EVIDENCE_STATUS",
    ],
    "disallowed_output_information_classes": [
        "ROW_ID",
        "BARCODE",
        "SEQUENCE",
        "RAW_ROW",
        "MEASUREMENT_VALUE",
        "LABEL",
        "CANONICAL_RECORD",
    ],
}

FAILURE_CODES = frozenset(
    {
        "PROTOCOL_INVALID",
        "INPUT_INTEGRITY_FAILED",
        "TABLE_AUDIT_FAILED",
        "PUBLICATION_FAILED",
    }
)

# Test-only fault-injection seam.  It is not exposed by the CLI and cannot
# change any scientific decision; it exists solely to prove post-snapshot
# namespace replacement is detected before parsing or publication.
_POST_VERIFIED_INPUT_SNAPSHOT_HOOK: Callable[[], None] | None = None
_PUBLICATION_FAULT_HOOK: Callable[[str], None] | None = None


class QualificationError(RuntimeError):
    """Closed-code failure that cannot be converted into a qualification PASS."""

    def __init__(self, message: str, *, code: str = "INPUT_INTEGRITY_FAILED") -> None:
        if code not in FAILURE_CODES:
            raise ValueError("failure code is outside the closed enum")
        super().__init__(message)
        self.code = code


class ScopeViolation(QualificationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="INPUT_INTEGRITY_FAILED")


class ProtocolError(QualificationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="PROTOCOL_INVALID")


class TableAuditError(QualificationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="TABLE_AUDIT_FAILED")


class PublicationError(QualificationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="PUBLICATION_FAILED")


class PublicationContention(PublicationError):
    """The exact output target already exists and must never be overwritten."""


class PartialPrecommitError(PublicationError):
    """A publication directory exists without a valid terminal commit marker."""

    publication_state = "PARTIAL_PRECOMMIT"


class CommittedNotAcceptedError(PublicationError):
    """A terminal marker exists, but the committed bundle cannot be accepted."""

    publication_state = "COMMITTED_NOT_ACCEPTED"

    def __init__(
        self,
        message: str,
        *,
        outcome: str,
        output_directory: Path,
        durability_warnings: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.output_directory = output_directory
        self.durability_warnings = tuple(durability_warnings)


FileIdentity = tuple[int, int, int, int, int, int]
DirectoryCoreIdentity = tuple[int, int, int]


@dataclass
class DirectoryBinding:
    path: Path
    fd: int
    full_identity: FileIdentity
    core_identity: DirectoryCoreIdentity
    label: str


@dataclass
class DesignState:
    pair_sequences: dict[str, dict[str, str]]
    control_ids: set[str]
    merged_ids: set[str]
    aggregate: dict[str, Any]


@dataclass
class AuxiliaryArmState:
    arms_by_pair: dict[str, set[str]]
    control_ids: set[str]
    complete_pair_ids: set[str]
    aggregate: dict[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _require_exact(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ProtocolError(f"{label} differs from the frozen production value")


def _require_exact_keys(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    observed = set(value)
    if observed != keys:
        raise ProtocolError(
            f"{label} keys are not closed; missing={sorted(keys - observed)}, "
            f"unexpected={sorted(observed - keys)}"
        )
    return value


def _expected_input_contract() -> dict[str, Any]:
    return {
        "data_root": str(EXPECTED_DATA_ROOT),
        "forbidden_path_tokens": [
            "GSE246381",
            "access_log",
            "sealed",
            "restricted",
            "sealed_external",
        ],
        "manifest_relative_path": "manifest.json",
        "manifest_expected_bytes": EXPECTED_MANIFEST_BYTES,
        "manifest_expected_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_expected_current_entry_count": 2,
        "manifest_full_superseries_binding_status": "INCOMPLETE_NOT_TRUSTED_AS_COMPLETE",
        "unique_input_asset_count": 6,
        "same_descriptor_verified_snapshot_required": True,
        "root_to_leaf_o_nofollow_required": True,
        "all_input_hashes_and_sizes_required_before_parsing": True,
        "network_access_allowed": False,
        "input_writes_allowed": False,
        "assets": [dict(asset) for asset in EXPECTED_ASSETS],
        "tar_member_equivalence": [dict(item) for item in EXPECTED_TAR_EQUIVALENCE],
        "tar_members_count_as_additional_unique_assets": False,
    }


def _validate_implementation_binding_document(value: Any) -> Mapping[str, Any]:
    binding = _require_exact_keys(
        value, IMPLEMENTATION_BINDING_KEYS, label="protocol.implementation_binding"
    )
    common = {
        "repository_root_rule": "PROTOCOL_PATH_PARENT_PARENT",
        "qualifier_path": "scripts/route_a_v3/qualify_gse200304_a1.py",
        "test_path": "tests/route_a_v3/test_qualify_gse200304_a1.py",
        "post_implementation_allowed_changed_paths": [
            "configs/route_a_v3_gse200304_a1_qualification.json"
        ],
        "current_head_must_strictly_descend": True,
        "active_authority_must_be_ancestor": True,
        "clean_worktree_required": True,
        "running_script_must_match_qualifier_blob": True,
    }
    for key, expected in common.items():
        if binding[key] != expected:
            raise ProtocolError(f"protocol.implementation_binding.{key} is not frozen")
    if binding["status"] == "UNKNOWN_NOT_ASSERTED":
        _require_exact(dict(binding), IMPLEMENTATION_BINDING_UNKNOWN, label="implementation binding")
    elif binding["status"] == "BOUND":
        if COMMIT_RE.fullmatch(str(binding["implementation_commit"])) is None:
            raise ProtocolError("implementation binding commit is invalid")
        for key in ("qualifier_blob_sha256", "test_blob_sha256"):
            if SHA256_RE.fullmatch(str(binding[key])) is None:
                raise ProtocolError(f"implementation binding {key} is invalid")
    else:
        raise ProtocolError("implementation binding status is outside the closed enum")
    return binding


def _expected_blockers_for_binding(binding: Mapping[str, Any]) -> list[str]:
    blockers = list(EXPECTED_BLOCKERS)
    if binding["status"] == "UNKNOWN_NOT_ASSERTED":
        blockers.append(IMPLEMENTATION_BINDING_BLOCKER)
    return blockers


def _validate_protocol(document: Mapping[str, Any]) -> None:
    top = _require_exact_keys(
        document,
        {
            "contract_id",
            "schema_version",
            "protocol_id",
            "protocol_status",
            "activation_status",
            "qualification_status",
            "dataset_id",
            "study_group_id",
            "authority",
            "implementation_binding",
            "a1_gate",
            "scope",
            "input_contract",
            "table_contract",
            "paper_and_external_evidence",
            "a1_master_report_contract",
            "expected_future_assets",
            "unresolved_blockers",
            "output_contract",
            "model_results_may_change_this_protocol",
        },
        label="protocol",
    )
    implementation_binding = _validate_implementation_binding_document(
        top["implementation_binding"]
    )
    for key, expected in {
        "contract_id": CONTRACT_ID,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_status": PROTOCOL_STATUS,
        "activation_status": ACTIVATION_STATUS,
        "qualification_status": QUALIFICATION_STATUS,
        "dataset_id": DATASET_ID,
        "study_group_id": "GSE200304_SUPERSERIES_ONE_STUDY",
        "authority": EXPECTED_AUTHORITY,
        "a1_gate": EXPECTED_A1_GATE,
        "scope": EXPECTED_SCOPE,
        "input_contract": _expected_input_contract(),
        "table_contract": {
            "design": EXPECTED_DESIGN_CONTRACT,
            "processed": EXPECTED_PROCESSED_CONTRACT,
            "small_plasmid": EXPECTED_SMALL_CONTRACT,
            "ivt": EXPECTED_IVT_CONTRACT,
        },
        "paper_and_external_evidence": EXPECTED_PAPER_EVIDENCE,
        "a1_master_report_contract": EXPECTED_A1_MASTER_REPORT_CONTRACT,
        "expected_future_assets": EXPECTED_FUTURE_ASSETS,
        "unresolved_blockers": _expected_blockers_for_binding(implementation_binding),
        "output_contract": EXPECTED_OUTPUT_CONTRACT,
        "model_results_may_change_this_protocol": False,
    }.items():
        _require_exact(top[key], expected, label=f"protocol.{key}")
    for key in (
        "contract_sha256",
        "data_role_registry_sha256",
        "decision_log_sha256",
        "a1_qualification_sha256",
    ):
        if SHA256_RE.fullmatch(str(EXPECTED_AUTHORITY[key])) is None:
            raise ProtocolError(f"authority {key} is not a SHA-256")
    for key in ("active_authority_commit", "staging_parent_head"):
        if COMMIT_RE.fullmatch(str(EXPECTED_AUTHORITY[key])) is None:
            raise ProtocolError(f"authority {key} is not a commit id")


def _git_capture(repository_root: Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repository_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ProtocolError("implementation-binding git executable is unavailable") from exc
    if completed.returncode != 0:
        raise ProtocolError("implementation-binding git query failed closed")
    return completed.stdout


def _git_is_ancestor(repository_root: Path, ancestor: str, descendant: str) -> bool:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(repository_root),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise ProtocolError("implementation-binding git executable is unavailable") from exc
    if completed.returncode not in {0, 1}:
        raise ProtocolError("implementation-binding ancestry query failed closed")
    return completed.returncode == 0


def _verify_implementation_binding(
    binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    repository_root: Path,
    *,
    running_script_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a two-commit executable binding without a self-hash fixed point."""

    _validate_implementation_binding_document(binding)
    if binding["status"] == "UNKNOWN_NOT_ASSERTED":
        return {
            "status": "UNKNOWN_NOT_ASSERTED",
            "verified": False,
            "implementation_commit": "UNKNOWN_NOT_ASSERTED",
            "current_head": "UNKNOWN_NOT_ASSERTED",
            "clean_worktree": "UNKNOWN_NOT_ASSERTED",
            "active_authority_ancestor": "UNKNOWN_NOT_ASSERTED",
            "staging_parent_ancestor": "UNKNOWN_NOT_ASSERTED",
            "current_head_strict_descendant": "UNKNOWN_NOT_ASSERTED",
            "authority_blob_hashes_match": "UNKNOWN_NOT_ASSERTED",
            "implementation_blob_hashes_match": "UNKNOWN_NOT_ASSERTED",
            "running_script_matches_bound_blob": "UNKNOWN_NOT_ASSERTED",
            "post_implementation_change_set_is_config_only": "UNKNOWN_NOT_ASSERTED",
        }

    root = _absolute_without_resolving(repository_root)
    current_head = _git_capture(root, ["rev-parse", "HEAD"]).decode("ascii").strip()
    if COMMIT_RE.fullmatch(current_head) is None:
        raise ProtocolError("implementation binding current HEAD is invalid")
    dirty = _git_capture(root, ["status", "--porcelain=v1"])
    if dirty:
        raise ProtocolError("implementation binding requires a clean worktree")
    active_commit = str(authority["active_authority_commit"])
    staging_parent = str(authority["staging_parent_head"])
    implementation_commit = str(binding["implementation_commit"])
    if active_commit == staging_parent or not _git_is_ancestor(
        root, active_commit, staging_parent
    ):
        raise ProtocolError("active authority is not a strict ancestor of staging parent")
    if staging_parent == implementation_commit or not _git_is_ancestor(
        root, staging_parent, implementation_commit
    ):
        raise ProtocolError("staging parent is not a strict ancestor of implementation commit")
    if current_head == implementation_commit or not _git_is_ancestor(
        root, implementation_commit, current_head
    ):
        raise ProtocolError("current HEAD is not a strict descendant of implementation commit")
    changed_paths = _git_capture(
        root,
        ["diff", "--name-only", implementation_commit, current_head, "--"],
    ).decode("utf-8").splitlines()
    if changed_paths != list(binding["post_implementation_allowed_changed_paths"]):
        raise ProtocolError("post-implementation commit set is not config-only")

    authority_bindings = (
        ("contract_path", "contract_sha256"),
        ("data_role_registry_path", "data_role_registry_sha256"),
        ("decision_log_path", "decision_log_sha256"),
        ("a1_qualification_path", "a1_qualification_sha256"),
    )
    for path_key, hash_key in authority_bindings:
        blob = _git_capture(root, ["show", f"{active_commit}:{authority[path_key]}"])
        if _sha256_bytes(blob) != authority[hash_key]:
            raise ProtocolError("active authority blob hash differs from frozen binding")

    for path_key, hash_key in (
        ("qualifier_path", "qualifier_blob_sha256"),
        ("test_path", "test_blob_sha256"),
    ):
        blob = _git_capture(root, ["show", f"{implementation_commit}:{binding[path_key]}"])
        if _sha256_bytes(blob) != binding[hash_key]:
            raise ProtocolError("implementation commit blob hash differs from frozen binding")
    qualifier_blob = _git_capture(
        root, ["show", f"{implementation_commit}:{binding['qualifier_path']}"]
    )
    script_path = Path(__file__) if running_script_path is None else running_script_path
    try:
        running_bytes = script_path.read_bytes()
    except OSError as exc:
        raise ProtocolError("running qualifier bytes cannot be captured") from exc
    if running_bytes != qualifier_blob:
        raise ProtocolError("running qualifier bytes differ from implementation binding")
    return {
        "status": "PASS_IMPLEMENTATION_BINDING",
        "verified": True,
        "implementation_commit": implementation_commit,
        "current_head": current_head,
        "clean_worktree": True,
        "active_authority_ancestor": True,
        "staging_parent_ancestor": True,
        "current_head_strict_descendant": True,
        "authority_blob_hashes_match": True,
        "implementation_blob_hashes_match": True,
        "running_script_matches_bound_blob": True,
        "post_implementation_change_set_is_config_only": True,
    }


def expected_processed_header(contract: Mapping[str, Any] | None = None) -> list[str]:
    spec = EXPECTED_PROCESSED_CONTRACT if contract is None else contract
    header = [str(spec["key_column"])]
    replicates = list(spec["replicates"])
    for allele in spec["allele_order"]:
        for family in spec["measurement_family_order"]:
            samples = spec["sample_numbers_by_family"][family]
            if len(samples) != len(replicates):
                raise ProtocolError("processed family sample and replicate counts differ")
            header.extend(
                f"{family}_{replicate}_S{sample}_{allele}"
                for replicate, sample in zip(replicates, samples)
            )
    return header


def expected_ivt_header(contract: Mapping[str, Any] | None = None) -> list[str]:
    spec = EXPECTED_IVT_CONTRACT if contract is None else contract
    header = [str(spec["key_column"])]
    replicates = list(spec["replicates"])
    for timepoint in spec["timepoint_order"]:
        samples = spec["sample_numbers_by_timepoint"][timepoint]
        if len(samples) != len(replicates):
            raise ProtocolError("IVT timepoint sample and replicate counts differ")
        header.extend(
            f"IVT_{timepoint}_{replicate}_S{sample}"
            for replicate, sample in zip(replicates, samples)
        )
    return header


def _reject_forbidden_path(path: Path | str, *, label: str) -> None:
    text = os.fspath(path).casefold()
    hits = sorted(token for token in FORBIDDEN_PATH_TOKENS if token in text)
    if hits:
        raise ScopeViolation(
            f"{label} rejected before payload read; forbidden token(s): {','.join(hits)}"
        )


def _absolute_without_resolving(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return Path(os.path.normpath(os.fspath(candidate)))
    return Path(os.path.abspath(os.fspath(candidate)))


def _safe_basename(name: str, *, label: str) -> str:
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ScopeViolation(f"{label} is not a safe basename")
    _reject_forbidden_path(name, label=label)
    return name


def _safe_relative_parts(value: str, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise ScopeViolation(f"{label} must be a nonempty relative path")
    _reject_forbidden_path(value, label=label)
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ScopeViolation(f"{label} is not a safe relative path")
    return tuple(_safe_basename(part, label=label) for part in pure.parts)


def _file_identity(info: os.stat_result) -> FileIdentity:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory_core_identity(info: os.stat_result) -> DirectoryCoreIdentity:
    return (info.st_dev, info.st_ino, info.st_mode)


def _open_directory_no_symlinks(path: Path | str, *, label: str) -> DirectoryBinding:
    absolute = _absolute_without_resolving(path)
    _reject_forbidden_path(absolute, label=label)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ScopeViolation(f"{label} requires O_NOFOLLOW and O_DIRECTORY")
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(absolute.anchor, flags)
    except OSError as exc:
        raise ScopeViolation(f"{label} filesystem root could not be opened safely") from exc
    try:
        for component in absolute.parts[1:]:
            _safe_basename(component, label=label)
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ScopeViolation(f"{label} contains a symlink or non-directory") from exc
            os.close(descriptor)
            descriptor = next_descriptor
        opened = os.fstat(descriptor)
        observed = os.stat(absolute, follow_symlinks=False)
        if not stat.S_ISDIR(opened.st_mode) or _file_identity(opened) != _file_identity(observed):
            raise ScopeViolation(f"{label} directory identity changed while opening")
        return DirectoryBinding(
            path=absolute,
            fd=descriptor,
            full_identity=_file_identity(opened),
            core_identity=_directory_core_identity(opened),
            label=label,
        )
    except Exception:
        os.close(descriptor)
        raise


def _assert_directory_binding(binding: DirectoryBinding, *, full: bool = True) -> None:
    descriptor_info = os.fstat(binding.fd)
    try:
        path_info = os.stat(binding.path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ScopeViolation(f"{binding.label} disappeared after verified snapshot") from exc
    if not stat.S_ISDIR(path_info.st_mode):
        raise ScopeViolation(f"{binding.label} is no longer a directory")
    if full:
        if _file_identity(descriptor_info) != binding.full_identity:
            raise ScopeViolation(f"{binding.label} descriptor identity changed")
        if _file_identity(path_info) != binding.full_identity:
            raise ScopeViolation(f"{binding.label} path identity changed")
    else:
        if _directory_core_identity(descriptor_info) != binding.core_identity:
            raise ScopeViolation(f"{binding.label} descriptor identity changed")
        if _directory_core_identity(path_info) != binding.core_identity:
            raise ScopeViolation(f"{binding.label} path identity changed")


def _open_relative_parent(root: DirectoryBinding, parts: Sequence[str], *, label: str) -> int:
    descriptor = os.dup(root.fd)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for component in parts:
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ScopeViolation(f"{label} parent component is not a non-symlink directory")
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ScopeViolation(f"{label} parent component could not be opened safely") from exc
            opened = os.fstat(next_descriptor)
            after = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if _file_identity(before) != _file_identity(opened) or _file_identity(after) != _file_identity(opened):
                os.close(next_descriptor)
                raise ScopeViolation(f"{label} parent identity changed while opening")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_relative_verified_snapshot(
    root: DirectoryBinding,
    relative_path: str,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[bytes, dict[str, Any], FileIdentity]:
    parts = _safe_relative_parts(relative_path, label=label)
    parent_fd = _open_relative_parent(root, parts[:-1], label=label)
    leaf = parts[-1]
    try:
        try:
            before = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise QualificationError(f"{label} is missing") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ScopeViolation(f"{label} must be a non-symlink regular file")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(leaf, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ScopeViolation(f"{label} could not be opened safely") from exc
        try:
            opened = os.fstat(descriptor)
            identity = _file_identity(opened)
            if not stat.S_ISREG(opened.st_mode) or identity != _file_identity(before):
                raise ScopeViolation(f"{label} identity changed while opening")
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    break
                digest.update(block)
                chunks.append(block)
            payload = b"".join(chunks)
            final = os.fstat(descriptor)
            after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if _file_identity(final) != identity or _file_identity(after) != identity:
                raise ScopeViolation(f"{label} changed during descriptor capture")
            if len(payload) != opened.st_size:
                raise ScopeViolation(f"{label} byte count changed during descriptor capture")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)
    observed_sha256 = digest.hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise QualificationError(f"{label} SHA-256 mismatch")
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise QualificationError(f"{label} byte count mismatch")
    return payload, {"sha256": observed_sha256, "bytes": len(payload)}, identity


def _assert_relative_identity(
    root: DirectoryBinding, relative_path: str, identity: FileIdentity, *, label: str
) -> None:
    parts = _safe_relative_parts(relative_path, label=label)
    parent_fd = _open_relative_parent(root, parts[:-1], label=label)
    try:
        info = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if _file_identity(info) != identity or not stat.S_ISREG(info.st_mode):
            raise ScopeViolation(f"{label} path identity changed after verified snapshot")
    finally:
        os.close(parent_fd)


def _read_path_verified_snapshot(
    path: Path | str,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    absolute = _absolute_without_resolving(path)
    parent = _open_directory_no_symlinks(absolute.parent, label=f"{label} parent")
    try:
        payload, provenance, identity = _read_relative_verified_snapshot(
            parent,
            absolute.name,
            label=label,
            expected_sha256=expected_sha256,
        )
        _assert_relative_identity(parent, absolute.name, identity, label=label)
        _assert_directory_binding(parent)
        return payload, provenance
    finally:
        os.close(parent.fd)


def _preflight_paths_before_read(
    protocol_path: Path | str, data_root: Path | str, output_directory: Path | str
) -> tuple[Path, Path, Path]:
    raw = (
        (Path(protocol_path), "protocol path"),
        (Path(data_root), "ordinary-public data root"),
        (Path(output_directory), "output path"),
    )
    # This lexical token loop intentionally precedes expanduser, stat, resolve,
    # open, and protocol loading.
    for path, label in raw:
        _reject_forbidden_path(path, label=label)
    absolute = tuple(_absolute_without_resolving(path) for path, _ in raw)
    for path, (_, label) in zip(absolute, raw):
        _reject_forbidden_path(path, label=label)
    protocol, root, output = absolute
    if protocol.name != PROTOCOL_BASENAME:
        raise ScopeViolation("protocol basename is outside the frozen allowlist")
    if root.name != "GSE200304":
        raise ScopeViolation("data-root basename is outside the frozen allowlist")
    if root != _absolute_without_resolving(EXPECTED_DATA_ROOT):
        raise ScopeViolation("data root differs from the single frozen ordinary-public root")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ScopeViolation("output path must not be inside the read-only input root")
    if output == root or output == protocol or output == protocol.parent:
        raise ScopeViolation("output path overlaps an input or authority path")
    _safe_basename(output.name, label="output directory basename")
    return protocol, root, output


def _load_protocol(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise ProtocolError("explicit protocol launch SHA-256 is invalid")
    raw, provenance = _read_path_verified_snapshot(
        path,
        label="GSE200304 production protocol",
        expected_sha256=expected_sha256,
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("production protocol is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("production protocol root must be an object")
    _validate_protocol(value)
    provenance["launch_expected_sha256"] = expected_sha256
    return value, provenance


def _decode_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"{label} root must be an object")
    return value


def _audit_root_manifest(payload: bytes, protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _decode_json_object(payload, label="root P0 manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        entries = manifest.get("samples")
    if not isinstance(entries, list) or any(not isinstance(entry, Mapping) for entry in entries):
        raise QualificationError("root P0 manifest must contain a files or samples object list")
    expected_count = protocol["input_contract"]["manifest_expected_current_entry_count"]
    if len(entries) != expected_count:
        raise QualificationError("root P0 manifest does not have the frozen incomplete entry count")
    assets = {asset["asset_id"]: asset for asset in protocol["input_contract"]["assets"]}
    expected = {
        Path(assets["GSE200303_DESIGN"]["relative_path"]).name: assets["GSE200303_DESIGN"],
        Path(assets["GSE200303_SMALL_PLASMID"]["relative_path"]).name: assets["GSE200303_SMALL_PLASMID"],
    }
    observed_names: set[str] = set()
    declared_sha_count = 0
    declared_size_count = 0
    for entry in entries:
        name = entry.get("name") or entry.get("filename")
        if not isinstance(name, str):
            raise QualificationError("root P0 manifest entry lacks a filename")
        _safe_basename(name, label="root P0 manifest filename")
        if name in observed_names or name not in expected:
            raise QualificationError("root P0 manifest filename set differs from the frozen two-entry set")
        observed_names.add(name)
        expected_asset = expected[name]
        if entry.get("sha256") != expected_asset["sha256"]:
            raise QualificationError("root P0 manifest entry SHA-256 differs from the frozen asset")
        declared_sha_count += 1
        size = entry.get("bytes", entry.get("size"))
        if size != expected_asset["bytes"]:
            raise QualificationError("root P0 manifest entry byte count differs from the frozen asset")
        declared_size_count += 1
    if observed_names != set(expected):
        raise QualificationError("root P0 manifest does not bind the expected two direct assets")
    return {
        "present": True,
        "declared_entry_count": len(entries),
        "declared_sha256_count": declared_sha_count,
        "declared_byte_count_count": declared_size_count,
        "bound_unique_asset_count": len(entries),
        "full_superseries_unique_asset_count": protocol["input_contract"]["unique_input_asset_count"],
        "full_superseries_binding_complete": False,
        "trusted_as_complete": False,
        "status": "INCOMPLETE_NOT_TRUSTED_AS_COMPLETE",
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
    }


def _parse_gzip_tsv(payload: bytes, *, label: str) -> tuple[list[str], list[list[str]]]:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as handle:
            decompressed = handle.read(MAX_DECOMPRESSED_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise TableAuditError(f"{label} is corrupt gzip data") from exc
    if len(decompressed) > MAX_DECOMPRESSED_BYTES:
        raise TableAuditError(f"{label} exceeds the decompression bound")
    try:
        text = decompressed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TableAuditError(f"{label} is not UTF-8 TSV") from exc
    reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t")
    try:
        header = next(reader)
    except StopIteration as exc:
        raise TableAuditError(f"{label} is empty") from exc
    rows = list(reader)
    if any(not row for row in rows):
        raise TableAuditError(f"{label} contains a blank physical row")
    return header, rows


def strip_design_suffix(value: str) -> tuple[str, str | None]:
    if value.endswith("_Mutant"):
        return value[: -len("_Mutant")], "Mutant"
    if value.endswith("_WT"):
        return value[: -len("_WT")], "WT"
    return value, None


def _hamming_distance(left: str, right: str) -> int | None:
    if len(left) != len(right):
        return None
    return sum(a != b for a, b in zip(left, right))


def _audit_design(payload: bytes, contract: Mapping[str, Any], *, label: str) -> DesignState:
    header, rows = _parse_gzip_tsv(payload, label=label)
    if header != contract["exact_header"]:
        raise TableAuditError(f"{label} header differs from the exact frozen header")
    if len(rows) != contract["row_count"]:
        raise TableAuditError(f"{label} row count differs from the frozen count")
    indices = {name: index for index, name in enumerate(header)}
    pairs: dict[str, dict[str, str]] = {}
    controls: set[str] = set()
    merged_ids: set[str] = set()
    type_counts = {"WT": 0, "Mutant": 0, "Control": 0}
    identity_equal_count = 0
    for row in rows:
        if len(row) != len(header):
            raise TableAuditError(f"{label} contains a ragged row")
        row_type = row[indices["Type"]]
        if row_type not in type_counts:
            raise TableAuditError(f"{label} contains an unrecognized or missing Type")
        identifier = row[indices["ID"]]
        sequence = row[indices["201bp"]]
        merged_id = row[indices["merged_id"]]
        if not identifier or not merged_id:
            raise TableAuditError(f"{label} contains a missing ID or merged_id")
        if len(sequence) != contract["sequence_length"] or SEQUENCE_RE.fullmatch(sequence) is None:
            raise TableAuditError(f"{label} contains a non-201nt ACGT sequence")
        if merged_id in merged_ids:
            raise TableAuditError(f"{label} contains a duplicate merged_id")
        merged_ids.add(merged_id)
        base, suffix_type = strip_design_suffix(merged_id)
        if base != identifier:
            raise TableAuditError(f"{label} violates ID == strip_suffix(merged_id)")
        identity_equal_count += 1
        type_counts[row_type] += 1
        if row_type == "Control":
            if suffix_type is not None or identifier in controls or identifier in pairs:
                raise TableAuditError(f"{label} control identity or multiplicity is invalid")
            controls.add(identifier)
            continue
        if suffix_type != row_type or identifier in controls:
            raise TableAuditError(f"{label} Type and merged_id suffix disagree")
        pair = pairs.setdefault(identifier, {})
        if row_type in pair:
            raise TableAuditError(f"{label} has duplicate WT or Mutant membership")
        pair[row_type] = sequence
    if type_counts != contract["type_counts"]:
        raise TableAuditError(f"{label} Type counts differ from the frozen counts")
    if len(pairs) != contract["unique_pair_count"] or len(controls) != contract["control_id_count"]:
        raise TableAuditError(f"{label} pair or control ID count differs from the frozen counts")
    for pair in pairs.values():
        if set(pair) != {"WT", "Mutant"}:
            raise TableAuditError(f"{label} has a missing WT/Mutant pair member")
        if _hamming_distance(pair["WT"], pair["Mutant"]) != 1:
            raise TableAuditError(f"{label} contains a pair that is not exactly one SNV")
    source_pools: dict[str, set[str]] = {}
    for pair in pairs.values():
        pool = source_pools.setdefault(pair["WT"], set())
        if pair["Mutant"] in pool:
            raise TableAuditError(f"{label} contains a duplicate candidate within a source pool")
        pool.add(pair["Mutant"])
    pool_sizes = [len(candidates) for candidates in source_pools.values()]
    source_geometry = {
        "distinct_candidate_count": sum(pool_sizes),
        "distinct_wt_source_group_count": len(source_pools),
        "singleton_source_pool_count": sum(size == 1 for size in pool_sizes),
        "two_candidate_source_pool_count": sum(size == 2 for size in pool_sizes),
        "three_or_more_candidate_source_pool_count": sum(size >= 3 for size in pool_sizes),
        "ndcg_eligible_source_pool_count": sum(size >= 3 for size in pool_sizes),
    }
    for key, observed in source_geometry.items():
        if observed != contract[key]:
            raise TableAuditError(f"{label} {key} differs from the frozen source geometry")
    return DesignState(
        pair_sequences=pairs,
        control_ids=controls,
        merged_ids=merged_ids,
        aggregate={
            "row_count": len(rows),
            "wt_row_count": type_counts["WT"],
            "mutant_row_count": type_counts["Mutant"],
            "control_row_count": type_counts["Control"],
            "unique_pair_count": len(pairs),
            **source_geometry,
            "unique_control_id_count": len(controls),
            "identity_equal_row_count": identity_equal_count,
            "sequence_length": contract["sequence_length"],
            "all_pairs_exactly_one_snv": True,
            "all_pairs_exactly_one_wt_one_mutant": True,
            "status": "PASS_MECHANICAL_DESIGN_AUDIT",
        },
    )


def _finite_number(text: str, *, label: str) -> float:
    if not isinstance(text, str) or not text.strip():
        raise TableAuditError(f"{label} contains a missing numeric value")
    try:
        value = float(text)
    except ValueError as exc:
        raise TableAuditError(f"{label} contains a nonnumeric value") from exc
    if not math.isfinite(value):
        raise TableAuditError(f"{label} contains a nonfinite value")
    return value


def _audit_processed(
    payload: bytes,
    contract: Mapping[str, Any],
    design_pair_ids: set[str],
    *,
    label: str,
) -> tuple[set[str], dict[str, Any]]:
    header, rows = _parse_gzip_tsv(payload, label=label)
    expected_header = expected_processed_header(contract)
    if header != expected_header:
        raise TableAuditError(f"{label} header differs from the exact 61-column contract")
    if any(name.casefold() == "freq" for name in header[1:]):
        raise TableAuditError(f"{label} may not use legacy Freq as an endpoint")
    if len(rows) != contract["row_count"]:
        raise TableAuditError(f"{label} row count differs from the frozen count")
    keys: set[str] = set()
    numeric_cells = 0
    for row in rows:
        if len(row) != len(header):
            raise TableAuditError(f"{label} contains a ragged row")
        key = row[0]
        if not key or key in keys:
            raise TableAuditError(f"{label} contains a missing or duplicate barcode")
        if key not in design_pair_ids:
            raise TableAuditError(f"{label} barcode does not exact-join a design pair ID")
        keys.add(key)
        for text in row[1:]:
            _finite_number(text, label=label)
            numeric_cells += 1
    attrition = len(design_pair_ids - keys)
    if len(keys) != contract["row_count"] or attrition != contract["outcome_blind_attrition_count"]:
        raise TableAuditError(f"{label} exact join or frozen attrition count differs")
    primary_families = set(EXPECTED_SCOPE["primary_measurement_families"])
    primary_column_count = sum(
        1
        for name in header[1:]
        if any(name.startswith(f"{family}_") for family in primary_families)
    )
    auxiliary_column_count = len(header) - 1 - primary_column_count
    return keys, {
        "row_count": len(rows),
        "unique_barcode_count": len(keys),
        "measurement_column_count": len(header) - 1,
        "finite_nonmissing_numeric_cell_count": numeric_cells,
        "missing_numeric_cell_count": 0,
        "nonfinite_numeric_cell_count": 0,
        "exact_design_pair_join_count": len(keys),
        "outcome_blind_attrition_count": attrition,
        "primary_measurement_column_count": primary_column_count,
        "qc_auxiliary_measurement_column_count": auxiliary_column_count,
        "legacy_freq_used_as_endpoint": False,
        "status": "PASS_MECHANICAL_PROCESSED_COMPANION_AUDIT",
    }


def _normalize_auxiliary_key(
    value: str,
    design_pair_ids: set[str],
    design_control_ids: set[str],
    *,
    label: str,
) -> tuple[str, str | None]:
    if not value:
        raise TableAuditError(f"{label} contains a missing key")
    base, allele = strip_design_suffix(value)
    if allele is None:
        if value not in design_control_ids:
            raise TableAuditError(f"{label} unsuffixed key is not a frozen control")
        return value, None
    if base not in design_pair_ids:
        raise TableAuditError(f"{label} allele key does not map to a frozen design pair")
    return base, allele


def _audit_small_plasmid(
    payload: bytes,
    contract: Mapping[str, Any],
    design_pair_ids: set[str],
    design_control_ids: set[str],
    *,
    label: str,
) -> AuxiliaryArmState:
    header, rows = _parse_gzip_tsv(payload, label=label)
    if header != contract["exact_header"]:
        raise TableAuditError(f"{label} header differs from exact Barcode/Freq")
    if len(rows) != contract["row_count"]:
        raise TableAuditError(f"{label} row count differs from the frozen count")
    raw_keys: set[str] = set()
    arms: dict[str, set[str]] = {}
    controls: set[str] = set()
    for row in rows:
        if len(row) != 2:
            raise TableAuditError(f"{label} contains a ragged row")
        key = row[0]
        if key in raw_keys:
            raise TableAuditError(f"{label} contains a duplicate Barcode")
        raw_keys.add(key)
        base, allele = _normalize_auxiliary_key(
            key, design_pair_ids, design_control_ids, label=label
        )
        if allele is None:
            controls.add(base)
        else:
            pair_arms = arms.setdefault(base, set())
            if allele in pair_arms:
                raise TableAuditError(f"{label} contains duplicate allele membership")
            pair_arms.add(allele)
        _finite_number(row[1], label=label)
    complete = {key for key, observed in arms.items() if observed == {"WT", "Mutant"}}
    wt_only = {key for key, observed in arms.items() if observed == {"WT"}}
    mutant_only = {key for key, observed in arms.items() if observed == {"Mutant"}}
    neither = design_pair_ids - set(arms)
    exact_counts = {
        "complete_pair_count": len(complete),
        "wt_only_pair_count": len(wt_only),
        "mutant_only_pair_count": len(mutant_only),
        "neither_pair_count": len(neither),
        "control_row_count": len(controls),
    }
    for key, observed in exact_counts.items():
        if observed != contract[key]:
            raise TableAuditError(f"{label} {key} differs from frozen arm completeness")
    aggregate = {
        "row_count": len(rows),
        "unique_barcode_count": len(raw_keys),
        "unique_design_id_join_count": len(set(arms) | controls),
        **exact_counts,
        "finite_nonmissing_numeric_cell_count": len(rows),
        "legacy_freq_column_present": True,
        "legacy_freq_used_as_endpoint": False,
        "role": "AUXILIARY_QC_ONLY_NOT_ENDPOINT",
        "status": "PASS_AUXILIARY_QC_TABLE_AUDIT",
    }
    return AuxiliaryArmState(arms, controls, complete, aggregate)


def _audit_ivt(
    payload: bytes,
    contract: Mapping[str, Any],
    design_pair_ids: set[str],
    design_control_ids: set[str],
    *,
    label: str,
) -> AuxiliaryArmState:
    header, rows = _parse_gzip_tsv(payload, label=label)
    if header != expected_ivt_header(contract):
        raise TableAuditError(f"{label} header differs from the exact 31-column contract")
    if len(rows) != contract["row_count"]:
        raise TableAuditError(f"{label} row count differs from the frozen count")
    raw_keys: set[str] = set()
    arms: dict[str, set[str]] = {}
    controls: set[str] = set()
    numeric_cells = 0
    for row in rows:
        if len(row) != len(header):
            raise TableAuditError(f"{label} contains a ragged row")
        key = row[0]
        if key in raw_keys:
            raise TableAuditError(f"{label} contains a duplicate ids value")
        raw_keys.add(key)
        base, allele = _normalize_auxiliary_key(
            key, design_pair_ids, design_control_ids, label=label
        )
        if allele is None:
            controls.add(base)
        else:
            pair_arms = arms.setdefault(base, set())
            if allele in pair_arms:
                raise TableAuditError(f"{label} contains duplicate allele membership")
            pair_arms.add(allele)
        for text in row[1:]:
            _finite_number(text, label=label)
            numeric_cells += 1
    complete = {key for key, observed in arms.items() if observed == {"WT", "Mutant"}}
    wt_only = {key for key, observed in arms.items() if observed == {"WT"}}
    mutant_only = {key for key, observed in arms.items() if observed == {"Mutant"}}
    missing_both = design_pair_ids - set(arms)
    exact_counts = {
        "complete_pair_count": len(complete),
        "wt_only_pair_count": len(wt_only),
        "mutant_only_pair_count": len(mutant_only),
        "missing_both_pair_count": len(missing_both),
        "control_row_count": len(controls),
    }
    for key, observed in exact_counts.items():
        if observed != contract[key]:
            raise TableAuditError(f"{label} {key} differs from frozen arm completeness")
    aggregate = {
        "row_count": len(rows),
        "unique_id_count": len(raw_keys),
        "unique_design_id_join_count": len(set(arms) | controls),
        **exact_counts,
        "measurement_column_count": len(header) - 1,
        "finite_nonmissing_numeric_cell_count": numeric_cells,
        "role": "AUXILIARY_QC_ONLY_NOT_ENDPOINT",
        "status": "PASS_AUXILIARY_QC_TABLE_AUDIT",
    }
    return AuxiliaryArmState(arms, controls, complete, aggregate)


def _audit_tar_equivalence(
    tar_payload: bytes,
    direct_payloads: Mapping[str, bytes],
    equivalence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {str(item["member_name"]): str(item["direct_asset_id"]) for item in equivalence}
    observed: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_payload), mode="r:") as archive:
            members = archive.getmembers()
            if len(members) != len(expected):
                raise QualificationError("GSE200303 tar member count differs from the exact contract")
            for member in members:
                pure = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or pure.is_absolute()
                    or len(pure.parts) != 1
                    or pure.name not in expected
                    or pure.name in observed
                ):
                    raise QualificationError("GSE200303 tar contains an unsafe or unexpected member")
                handle = archive.extractfile(member)
                if handle is None:
                    raise QualificationError("GSE200303 tar regular member cannot be read")
                payload = handle.read(member.size + 1)
                if len(payload) != member.size:
                    raise QualificationError("GSE200303 tar member size changed while reading")
                observed[pure.name] = payload
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise QualificationError("GSE200303 tar is corrupt") from exc
    if set(observed) != set(expected):
        raise QualificationError("GSE200303 tar member set differs from the exact contract")
    for name, asset_id in expected.items():
        if asset_id not in direct_payloads or observed[name] != direct_payloads[asset_id]:
            raise QualificationError("GSE200303 tar member is not byte-identical to its direct asset")
    return {
        "member_count": len(observed),
        "byte_identical_direct_member_count": len(observed),
        "additional_unique_asset_count": 0,
        "members_counted_as_additional_unique_assets": False,
        "status": "PASS_EXACT_TAR_TO_DIRECT_BYTE_EQUIVALENCE",
    }


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicationError(f"{label} must be a nonnegative integer")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise PublicationError(f"{label} must be a boolean")
    return value


def _validate_integrity_payload(value: Mapping[str, Any]) -> None:
    top = _require_exact_keys(
        value,
        {
            "contract_id",
            "protocol_id",
            "dataset_id",
            "status",
            "aggregate_only",
            "protocol_snapshot",
            "manifest",
            "assets",
            "unique_input_asset_count",
            "tar_equivalence",
            "network_accessed",
            "input_payload_writes",
        },
        label="integrity output",
    )
    for key, expected in {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": "PASS_INPUT_INTEGRITY_FOR_GAP_AUDIT",
        "aggregate_only": True,
        "unique_input_asset_count": 6,
        "network_accessed": False,
        "input_payload_writes": 0,
    }.items():
        if top[key] != expected:
            raise PublicationError(f"integrity output {key} is not frozen")
    protocol = _require_exact_keys(
        top["protocol_snapshot"],
        {"sha256", "bytes", "launch_expected_sha256"},
        label="integrity protocol snapshot",
    )
    if SHA256_RE.fullmatch(str(protocol["sha256"])) is None or protocol["sha256"] != protocol["launch_expected_sha256"]:
        raise PublicationError("integrity protocol snapshot SHA-256 is invalid")
    _nonnegative_int(protocol["bytes"], label="integrity protocol bytes")
    manifest = _require_exact_keys(
        top["manifest"],
        {
            "present",
            "declared_entry_count",
            "declared_sha256_count",
            "declared_byte_count_count",
            "bound_unique_asset_count",
            "full_superseries_unique_asset_count",
            "full_superseries_binding_complete",
            "trusted_as_complete",
            "status",
            "sha256",
            "bytes",
        },
        label="integrity manifest",
    )
    for key, expected in {
        "present": True,
        "declared_entry_count": 2,
        "declared_sha256_count": 2,
        "declared_byte_count_count": 2,
        "bound_unique_asset_count": 2,
        "full_superseries_unique_asset_count": 6,
        "full_superseries_binding_complete": False,
        "trusted_as_complete": False,
        "status": "INCOMPLETE_NOT_TRUSTED_AS_COMPLETE",
    }.items():
        if manifest[key] != expected:
            raise PublicationError(f"integrity manifest {key} is not frozen")
    if manifest["sha256"] != EXPECTED_MANIFEST_SHA256:
        raise PublicationError("integrity manifest SHA-256 differs from frozen bytes")
    if manifest["bytes"] != EXPECTED_MANIFEST_BYTES:
        raise PublicationError("integrity manifest byte count differs from frozen bytes")
    assets = top["assets"]
    if not isinstance(assets, list) or len(assets) != len(EXPECTED_ASSETS):
        raise PublicationError("integrity assets must contain six aggregate summaries")
    for asset, frozen in zip(assets, EXPECTED_ASSETS):
        item = _require_exact_keys(
            asset,
            {"asset_id", "accession", "relative_locator", "role", "format", "bytes", "sha256", "verified"},
            label="integrity asset summary",
        )
        expected_summary = {
            "asset_id": frozen["asset_id"],
            "accession": frozen["accession"],
            "relative_locator": frozen["relative_path"],
            "role": frozen["role"],
            "format": frozen["format"],
            "bytes": frozen["bytes"],
            "sha256": frozen["sha256"],
            "verified": True,
        }
        if dict(item) != expected_summary:
            raise PublicationError("integrity asset summary differs from exact frozen semantics")
    tar = _require_exact_keys(
        top["tar_equivalence"],
        {
            "member_count",
            "byte_identical_direct_member_count",
            "additional_unique_asset_count",
            "members_counted_as_additional_unique_assets",
            "status",
        },
        label="integrity tar equivalence",
    )
    expected_tar = {
        "member_count": 2,
        "byte_identical_direct_member_count": 2,
        "additional_unique_asset_count": 0,
        "members_counted_as_additional_unique_assets": False,
        "status": "PASS_EXACT_TAR_TO_DIRECT_BYTE_EQUIVALENCE",
    }
    if dict(tar) != expected_tar:
        raise PublicationError("integrity tar equivalence is not exact")


def _validate_mechanical_payload(value: Mapping[str, Any]) -> None:
    top = _require_exact_keys(
        value,
        {
            "contract_id",
            "protocol_id",
            "dataset_id",
            "status",
            "aggregate_only",
            "primary_design",
            "auxiliary_design",
            "processed",
            "small_plasmid",
            "ivt",
            "join_audit",
            "scope_audit",
        },
        label="mechanical output",
    )
    for key, expected in {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": "PASS_MECHANICAL_GAP_AUDIT",
        "aggregate_only": True,
    }.items():
        if top[key] != expected:
            raise PublicationError(f"mechanical output {key} is not frozen")
    design_keys = {
        "row_count",
        "wt_row_count",
        "mutant_row_count",
        "control_row_count",
        "unique_pair_count",
        "distinct_candidate_count",
        "distinct_wt_source_group_count",
        "singleton_source_pool_count",
        "two_candidate_source_pool_count",
        "three_or_more_candidate_source_pool_count",
        "ndcg_eligible_source_pool_count",
        "unique_control_id_count",
        "identity_equal_row_count",
        "sequence_length",
        "all_pairs_exactly_one_snv",
        "all_pairs_exactly_one_wt_one_mutant",
        "status",
    }
    for name in ("primary_design", "auxiliary_design"):
        design = _require_exact_keys(top[name], design_keys, label=f"mechanical {name}")
        expected = {
            "row_count": 13836,
            "wt_row_count": 6885,
            "mutant_row_count": 6885,
            "control_row_count": 66,
            "unique_pair_count": 6885,
            "distinct_candidate_count": 6885,
            "distinct_wt_source_group_count": 6882,
            "singleton_source_pool_count": 6879,
            "two_candidate_source_pool_count": 3,
            "three_or_more_candidate_source_pool_count": 0,
            "ndcg_eligible_source_pool_count": 0,
            "unique_control_id_count": 66,
            "identity_equal_row_count": 13836,
            "sequence_length": 201,
            "all_pairs_exactly_one_snv": True,
            "all_pairs_exactly_one_wt_one_mutant": True,
            "status": "PASS_MECHANICAL_DESIGN_AUDIT",
        }
        if dict(design) != expected:
            raise PublicationError(f"mechanical {name} aggregates differ from the exact contract")
    processed = _require_exact_keys(
        top["processed"],
        {
            "row_count",
            "unique_barcode_count",
            "measurement_column_count",
            "finite_nonmissing_numeric_cell_count",
            "missing_numeric_cell_count",
            "nonfinite_numeric_cell_count",
            "exact_design_pair_join_count",
            "outcome_blind_attrition_count",
            "primary_measurement_column_count",
            "qc_auxiliary_measurement_column_count",
            "legacy_freq_used_as_endpoint",
            "status",
        },
        label="mechanical processed",
    )
    for key in processed:
        if key.endswith("count"):
            _nonnegative_int(processed[key], label=f"mechanical processed {key}")
    expected_processed = {
        "row_count": 6772,
        "unique_barcode_count": 6772,
        "measurement_column_count": 60,
        "finite_nonmissing_numeric_cell_count": 6772 * 60,
        "missing_numeric_cell_count": 0,
        "nonfinite_numeric_cell_count": 0,
        "exact_design_pair_join_count": 6772,
        "outcome_blind_attrition_count": 113,
        "primary_measurement_column_count": 36,
        "qc_auxiliary_measurement_column_count": 24,
        "legacy_freq_used_as_endpoint": False,
        "status": "PASS_MECHANICAL_PROCESSED_COMPANION_AUDIT",
    }
    if dict(processed) != expected_processed:
        raise PublicationError("mechanical processed aggregates differ from the exact contract")
    small = _require_exact_keys(
        top["small_plasmid"],
        {
            "row_count",
            "unique_barcode_count",
            "unique_design_id_join_count",
            "complete_pair_count",
            "wt_only_pair_count",
            "mutant_only_pair_count",
            "neither_pair_count",
            "control_row_count",
            "finite_nonmissing_numeric_cell_count",
            "legacy_freq_column_present",
            "legacy_freq_used_as_endpoint",
            "role",
            "status",
        },
        label="mechanical small plasmid",
    )
    expected_small = {
        "row_count": 12704,
        "unique_barcode_count": 12704,
        "unique_design_id_join_count": 6584,
        "complete_pair_count": 6120,
        "wt_only_pair_count": 192,
        "mutant_only_pair_count": 225,
        "neither_pair_count": 348,
        "control_row_count": 47,
        "finite_nonmissing_numeric_cell_count": 12704,
        "legacy_freq_column_present": True,
        "legacy_freq_used_as_endpoint": False,
        "role": "AUXILIARY_QC_ONLY_NOT_ENDPOINT",
        "status": "PASS_AUXILIARY_QC_TABLE_AUDIT",
    }
    if dict(small) != expected_small:
        raise PublicationError("small-plasmid aggregates differ from frozen arm completeness")
    ivt = _require_exact_keys(
        top["ivt"],
        {
            "row_count",
            "unique_id_count",
            "unique_design_id_join_count",
            "complete_pair_count",
            "wt_only_pair_count",
            "mutant_only_pair_count",
            "missing_both_pair_count",
            "control_row_count",
            "measurement_column_count",
            "finite_nonmissing_numeric_cell_count",
            "role",
            "status",
        },
        label="mechanical IVT",
    )
    expected_ivt = {
        "row_count": 13548,
        "unique_id_count": 13548,
        "unique_design_id_join_count": 6774,
        "complete_pair_count": 6774,
        "wt_only_pair_count": 0,
        "mutant_only_pair_count": 0,
        "missing_both_pair_count": 111,
        "control_row_count": 0,
        "measurement_column_count": 30,
        "finite_nonmissing_numeric_cell_count": 13548 * 30,
        "role": "AUXILIARY_QC_ONLY_NOT_ENDPOINT",
        "status": "PASS_AUXILIARY_QC_TABLE_AUDIT",
    }
    if dict(ivt) != expected_ivt:
        raise PublicationError("IVT aggregates differ from frozen arm completeness")
    join = _require_exact_keys(
        top["join_audit"],
        {
            "gse200302_gse200303_design_pair_key_sets_equal",
            "gse200302_gse200303_control_id_sets_equal",
            "processed_barcode_equals_design_id_count",
            "design_id_equals_strip_suffix_merged_id_row_count",
            "processed_outcome_blind_attrition_count",
            "small_plasmid_design_join_id_count",
            "ivt_design_join_id_count",
            "three_modal_auxiliary_join_pair_count",
            "three_modal_join_role",
        },
        label="mechanical join audit",
    )
    for key, expected in {
        "gse200302_gse200303_design_pair_key_sets_equal": True,
        "gse200302_gse200303_control_id_sets_equal": True,
        "processed_barcode_equals_design_id_count": 6772,
        "design_id_equals_strip_suffix_merged_id_row_count": 13836,
        "processed_outcome_blind_attrition_count": 113,
        "three_modal_join_role": "AUXILIARY_QC_ONLY_NOT_PRIMARY_ENDPOINT",
    }.items():
        if join[key] != expected:
            raise PublicationError(f"mechanical join {key} is invalid")
    for key, expected in {
        "small_plasmid_design_join_id_count": 6584,
        "ivt_design_join_id_count": 6774,
        "three_modal_auxiliary_join_pair_count": 6120,
    }.items():
        if join[key] != expected:
            raise PublicationError(f"mechanical join {key} differs from frozen arm-complete join")
    scope = _require_exact_keys(
        top["scope_audit"],
        {
            "observed_member_accession_count",
            "maximum_independent_study_count",
            "ordinary_study_contribution",
            "a1_study_contribution",
            "true_a2_study_contribution",
            "primary_accession",
            "primary_measurement_column_count",
            "pDNA_gse200303_ivt_role",
            "legacy_freq_is_endpoint",
        },
        label="mechanical scope audit",
    )
    expected_scope = {
        "observed_member_accession_count": 4,
        "maximum_independent_study_count": 1,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "primary_accession": "GSE200302",
        "primary_measurement_column_count": 36,
        "pDNA_gse200303_ivt_role": "AUXILIARY_QC_ONLY_NOT_PRIMARY_ENDPOINT",
        "legacy_freq_is_endpoint": False,
    }
    if dict(scope) != expected_scope:
        raise PublicationError("mechanical scope audit differs from the hard boundary")


def _expected_a1_master_report() -> dict[str, Any]:
    return {
        "required_report_fields": list(
            EXPECTED_A1_MASTER_REPORT_CONTRACT["required_report_fields"]
        ),
        "study_recovery_method": "SOURCE_CANDIDATE_ENDPOINT_CONTEXT_AND_GROUP_AUDIT",
        "metadata_only_recovery_allowed": False,
        "minimum_ndcg_pool_size": 3,
        "two_candidate_pool_use": "PAIRWISE_ONLY",
        "unknown_status_is_hard_block": True,
        "nominal_rows": {
            "value": 6885,
            "unit": "WT_MUTANT_CANDIDATE_PAIRS",
            "status": "MECHANICALLY_VERIFIED",
        },
        "distinct_candidates": {
            "value": 6885,
            "status": "MECHANICALLY_VERIFIED_WITHIN_SOURCE_POOLS",
        },
        "biological_source_groups": {
            "value": 6882,
            "grouping_proxy": "DISTINCT_WT_SEQUENCE",
            "status": "MECHANICAL_PROXY_ONLY_NOT_BIOLOGICALLY_FROZEN",
        },
        "gene_groups": {
            "value": None,
            "status": "UNKNOWN_NOT_ASSERTED_CURRENT_QUALIFIER_DOES_NOT_INTEGRATE_S3",
        },
        "study_groups": {
            "value": 1,
            "member_accession_count": 4,
            "qualified_study_count": 0,
            "status": "ONE_SUPERSERIES_STUDY_NOT_QUALIFIED",
        },
        "eligible_multi_candidate_pools": {
            "singleton_pool_count": 6879,
            "two_candidate_pool_count": 3,
            "three_or_more_candidate_pool_count": 0,
            "ndcg_eligible_pool_count": 0,
            "pairwise_only_pool_count": 3,
            "status": "NO_NDCG_ELIGIBLE_POOL_TWO_CANDIDATE_POOLS_PAIRWISE_ONLY",
        },
        "edit_count_strata": {
            "exactly_one_snv_count": 6885,
            "other_edit_count": 0,
            "status": "MECHANICALLY_VERIFIED",
        },
        "replicate_and_se_coverage": {
            "processed_candidate_count": 6772,
            "replicates_per_allele_measurement_family": 6,
            "precomputed_gap_se_finite_positive_count": 6772,
            "paper_native_se_replay_status": "NOT_RUN",
            "status": "GAP_EVIDENCE_ONLY_NOT_QUALIFYING",
        },
        "beneficial_and_noise_zone_balance": {
            "beneficial_count": None,
            "noise_zone_count": None,
            "direction_rule_frozen": False,
            "margin_rule_frozen": False,
            "status": "UNKNOWN_NOT_ASSERTED_NOT_RUN",
        },
        "post_dedup_effective_n": {
            "wt_sequence_proxy_group_count": 6882,
            "effective_n": None,
            "independence_unit_frozen": False,
            "bootstrap_status": "NOT_RUN",
            "status": "UNKNOWN_NOT_ASSERTED",
        },
        "foundation_exposure": {
            "checkpoint": None,
            "exposure_status": "UNKNOWN_NOT_ASSERTED",
            "status": "HARD_BLOCKED",
        },
        "license_and_redistribution_status": {
            "artifact_specific_code_license_status": "UNKNOWN_FAIL_CLOSED",
            "redistribution_allowed": False,
            "allowed_mode": "PUBLIC_LOCATOR_HASH_AND_AGGREGATE_ONLY_UNTIL_LICENSE_CLOSED",
            "status": "HARD_BLOCKED",
        },
        "identification_audit": {
            "source": "WT_SEQUENCE_MECHANICAL_PROXY_ONLY_NOT_FROZEN_BIOLOGICAL_SOURCE",
            "candidate": "EXACT_ONE_SNV_MECHANICALLY_IDENTIFIED",
            "endpoint": "COMPANION_COLUMNS_IDENTIFIED_PAPER_NATIVE_REPLAY_NOT_RUN",
            "context": "UNKNOWN_NOT_ASSERTED",
            "group": "WT_SEQUENCE_PROXY_ONLY_NOT_FROZEN",
            "transform": "EXACT_ONE_SNV_MECHANICALLY_IDENTIFIED_ONLY",
            "overall": "NOT_QUALIFIED",
        },
    }


def _validate_implementation_binding_report(value: Any) -> Mapping[str, Any]:
    keys = {
        "status",
        "verified",
        "implementation_commit",
        "current_head",
        "clean_worktree",
        "active_authority_ancestor",
        "staging_parent_ancestor",
        "current_head_strict_descendant",
        "authority_blob_hashes_match",
        "implementation_blob_hashes_match",
        "running_script_matches_bound_blob",
        "post_implementation_change_set_is_config_only",
    }
    report = _require_exact_keys(value, keys, label="implementation binding report")
    if report["status"] == "UNKNOWN_NOT_ASSERTED":
        expected = _verify_implementation_binding(
            IMPLEMENTATION_BINDING_UNKNOWN, EXPECTED_AUTHORITY, Path(".")
        )
        if dict(report) != expected:
            raise PublicationError("unknown implementation binding report is not exact")
    elif report["status"] == "PASS_IMPLEMENTATION_BINDING":
        if report["verified"] is not True:
            raise PublicationError("bound implementation report is not verified")
        for key in ("implementation_commit", "current_head"):
            if COMMIT_RE.fullmatch(str(report[key])) is None:
                raise PublicationError("implementation report commit is invalid")
        for key in keys - {"status", "implementation_commit", "current_head"}:
            if report[key] is not True:
                raise PublicationError("bound implementation report contains a failed assertion")
        if report["implementation_commit"] == report["current_head"]:
            raise PublicationError("implementation report does not bind a strict descendant")
    else:
        raise PublicationError("implementation binding report status is outside the closed enum")
    return report


def _validate_report_payload(value: Mapping[str, Any]) -> None:
    top = _require_exact_keys(
        value,
        {
            "contract_id",
            "protocol_id",
            "dataset_id",
            "execution_outcome",
            "protocol_status",
            "activation_status",
            "qualification_status",
            "scientific_claim_status",
            "qualified",
            "ordinary_study_contribution",
            "a1_study_contribution",
            "true_a2_study_contribution",
            "canonical_record_count",
            "training_allowed",
            "model_selection_allowed",
            "canonical_materialization_allowed",
            "a1_gate",
            "implementation_binding",
            "paper_native_raw_xtail_replay_status",
            "public_asset_bundle_lineage",
            "ena_fastq_manifest_bundle_lineage",
            "current_qualifier_integrates_public_asset_bundle",
            "current_qualifier_integrates_ena_fastq_manifest_bundle",
            "precomputed_gap_evidence_only",
            "a1_master_report",
            "unresolved_blockers",
            "redistribution_mode",
            "aggregate_only",
        },
        label="qualification report",
    )
    expected_fixed = {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "execution_outcome": SUCCESS_OUTCOME,
        "protocol_status": PROTOCOL_STATUS,
        "activation_status": ACTIVATION_STATUS,
        "qualification_status": QUALIFICATION_STATUS,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "qualified": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "canonical_materialization_allowed": False,
        "a1_gate": EXPECTED_A1_GATE,
        "paper_native_raw_xtail_replay_status": "NOT_RUN",
        "public_asset_bundle_lineage": EXPECTED_PAPER_EVIDENCE["public_asset_bundle_lineage"],
        "ena_fastq_manifest_bundle_lineage": EXPECTED_PAPER_EVIDENCE[
            "ena_fastq_manifest_bundle_lineage"
        ],
        "current_qualifier_integrates_public_asset_bundle": False,
        "current_qualifier_integrates_ena_fastq_manifest_bundle": False,
        "precomputed_gap_evidence_only": EXPECTED_PAPER_EVIDENCE["precomputed_gap_evidence_only"],
        "a1_master_report": _expected_a1_master_report(),
        "redistribution_mode": "PUBLIC_LOCATOR_HASH_AND_AGGREGATE_ONLY_UNTIL_LICENSE_CLOSED",
        "aggregate_only": True,
    }
    for key, expected in expected_fixed.items():
        if top[key] != expected:
            raise PublicationError(f"qualification report {key} differs from hard-block schema")
    binding = _validate_implementation_binding_report(top["implementation_binding"])
    expected_blockers = list(EXPECTED_BLOCKERS)
    if binding["status"] == "UNKNOWN_NOT_ASSERTED":
        expected_blockers.append(IMPLEMENTATION_BINDING_BLOCKER)
    if top["unresolved_blockers"] != expected_blockers:
        raise PublicationError("qualification report blockers differ from binding state")


def _validate_failure_payload(value: Mapping[str, Any]) -> None:
    top = _require_exact_keys(
        value,
        {
            "contract_id",
            "protocol_id",
            "dataset_id",
            "execution_outcome",
            "status",
            "failure_code",
            "qualified",
            "ordinary_study_contribution",
            "a1_study_contribution",
            "true_a2_study_contribution",
            "canonical_record_count",
            "training_allowed",
            "model_selection_allowed",
            "success_bundle_published",
            "aggregate_only",
        },
        label="failure report",
    )
    for key, expected in {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "execution_outcome": FAILURE_OUTCOME,
        "status": "FAIL_CLOSED_BEFORE_SUCCESS_BUNDLE_PUBLICATION",
        "qualified": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "success_bundle_published": False,
        "aggregate_only": True,
    }.items():
        if top[key] != expected:
            raise PublicationError(f"failure report {key} is not frozen")
    if top["failure_code"] not in FAILURE_CODES:
        raise PublicationError("failure report code is outside the closed enum")


def _validate_closed_payloads(payloads: Mapping[str, Mapping[str, Any]], *, outcome: str) -> None:
    if outcome == SUCCESS_OUTCOME:
        if set(payloads) != set(SUCCESS_JSON_FILES):
            raise PublicationError("success payload filename set is not closed")
        _validate_integrity_payload(payloads["INPUT_INTEGRITY_AUDIT.json"])
        _validate_mechanical_payload(payloads["MECHANICAL_AUDIT.json"])
        _validate_report_payload(payloads["QUALIFICATION_REPORT.json"])
    elif outcome == FAILURE_OUTCOME:
        if set(payloads) != set(FAILURE_JSON_FILES):
            raise PublicationError("failure payload filename set is not closed")
        _validate_failure_payload(payloads["FAILURE_REPORT.json"])
    else:
        raise PublicationError("publication outcome is outside the success/failure enum")


def _safe_output_name(name: str) -> str:
    if name not in {
        *SUCCESS_JSON_FILES,
        *FAILURE_JSON_FILES,
        SHA256SUMS_FILENAME,
        PUBLICATION_MARKER,
    }:
        raise PublicationError("output filename is outside the closed allowlist")
    return name


def _publication_fault(phase: str) -> None:
    if _PUBLICATION_FAULT_HOOK is not None:
        _PUBLICATION_FAULT_HOOK(phase)


def _write_exclusive_at(
    directory_fd: int, name: str, payload: bytes, *, terminal: bool = False
) -> list[str]:
    _safe_output_name(name)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o640, dir_fd=directory_fd)
    except OSError as exc:
        raise PublicationError(f"exclusive publication member {name} could not be created") from exc
    error: Exception | None = None
    fully_written = False
    warnings: list[str] = []
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PublicationError("exclusive publication write made no progress")
            view = view[written:]
        fully_written = True
        try:
            if terminal:
                _publication_fault("terminal_marker_fsync")
            os.fsync(descriptor)
        except Exception as exc:
            if terminal:
                warnings.append("TERMINAL_MARKER_FSYNC_WARNING")
            else:
                error = PublicationError("ordinary publication member fsync failed")
                error.__cause__ = exc
    except Exception as exc:
        error = exc
    finally:
        try:
            if terminal:
                _publication_fault("terminal_marker_close")
        except Exception:
            if fully_written:
                warnings.append("TERMINAL_MARKER_CLOSE_WARNING")
            elif error is None:
                error = PublicationError("terminal marker close fault preceded complete write")
        try:
            os.close(descriptor)
        except OSError as exc:
            if fully_written and (terminal or error is None):
                warnings.append(
                    "TERMINAL_MARKER_CLOSE_WARNING"
                    if terminal
                    else "ORDINARY_MEMBER_CLOSE_WARNING"
                )
            elif error is None:
                error = PublicationError("exclusive publication member close failed")
                error.__cause__ = exc
    if error is not None:
        if isinstance(error, QualificationError):
            raise error
        raise PublicationError("exclusive publication member write failed") from error
    return warnings


def _read_output_member_at(directory_fd: int, name: str) -> bytes:
    _safe_output_name(name)
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PublicationError(f"published member {name} is not a regular file")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise PublicationError(f"published member {name} identity changed while opening")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        final = os.fstat(descriptor)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _file_identity(final) != _file_identity(opened) or _file_identity(after) != _file_identity(opened):
            raise PublicationError(f"published member {name} changed during validation")
        payload = b"".join(chunks)
        if len(payload) != opened.st_size:
            raise PublicationError(f"published member {name} byte count changed")
        return payload
    finally:
        os.close(descriptor)


def _publication_marker_document(
    *,
    outcome: str,
    member_payloads: Mapping[str, bytes],
    output_directory: Path,
) -> dict[str, Any]:
    names = sorted(member_payloads)
    target = _absolute_without_resolving(output_directory)
    return {
        "schema_version": "1.0.0",
        "record_type": "GSE200304_AGGREGATE_BUNDLE_PUBLICATION_COMMIT",
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "output_id": OUTPUT_ID,
        "execution_outcome": outcome,
        "bundle_member_names": names,
        "bundle_member_count": len(names),
        "sha256sums_sha256": _sha256_bytes(member_payloads[SHA256SUMS_FILENAME]),
        "target_binding_scheme": "SHA256_OF_UTF8_NORMALIZED_ABSOLUTE_TARGET_AND_BASENAME",
        "final_output_directory_name_sha256": _sha256_bytes(
            target.name.encode("utf-8")
        ),
        "final_output_target_sha256": _sha256_bytes(os.fspath(target).encode("utf-8")),
        "committed": True,
        "terminal_marker_written_last": True,
    }


def _validate_publication_commit(
    directory_fd: int,
    *,
    outcome: str,
    expected_member_payloads: Mapping[str, bytes],
    output_directory: Path,
) -> dict[str, Any]:
    marker_payload = _read_output_member_at(directory_fd, PUBLICATION_MARKER)
    try:
        marker = json.loads(marker_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError("terminal publication marker is not valid JSON") from exc
    if marker != _publication_marker_document(
        outcome=outcome,
        member_payloads=expected_member_payloads,
        output_directory=output_directory,
    ):
        raise PublicationError("terminal publication marker does not exactly bind the bundle")
    observed_names = set(os.listdir(directory_fd))
    expected_names = set(expected_member_payloads) | {PUBLICATION_MARKER}
    if observed_names != expected_names:
        raise PublicationError("published directory member set differs from the committed set")
    for name, expected in expected_member_payloads.items():
        observed = _read_output_member_at(directory_fd, name)
        if observed != expected:
            raise PublicationError(f"published member {name} differs from committed bytes")
    sums_payload = expected_member_payloads[SHA256SUMS_FILENAME]
    try:
        lines = sums_payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise PublicationError("SHA256SUMS is not ASCII") from exc
    expected_json_names = sorted(set(expected_member_payloads) - {SHA256SUMS_FILENAME})
    expected_lines = [
        f"{_sha256_bytes(expected_member_payloads[name])}  {name}" for name in expected_json_names
    ]
    if lines != expected_lines:
        raise PublicationError("SHA256SUMS does not exactly bind the JSON members")
    return marker


def validate_published_bundle(output_directory: Path) -> dict[str, Any]:
    """Default consumer: accept only a bundle committed at this exact target."""

    target = _absolute_without_resolving(output_directory)
    parent = _open_directory_no_symlinks(target.parent, label="published bundle parent")
    output_fd: int | None = None
    try:
        _safe_basename(target.name, label="published bundle basename")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        output_fd = os.open(target.name, flags, dir_fd=parent.fd)
        marker_raw = _read_output_member_at(output_fd, PUBLICATION_MARKER)
        try:
            marker = json.loads(marker_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicationError("terminal publication marker is not valid JSON") from exc
        outcome = marker.get("execution_outcome") if isinstance(marker, Mapping) else None
        if outcome == SUCCESS_OUTCOME:
            json_names = set(SUCCESS_JSON_FILES)
        elif outcome == FAILURE_OUTCOME:
            json_names = set(FAILURE_JSON_FILES)
        else:
            raise PublicationError("terminal marker outcome is outside the closed enum")
        expected_names = json_names | {SHA256SUMS_FILENAME, PUBLICATION_MARKER}
        if set(os.listdir(output_fd)) != expected_names:
            raise PublicationError("published bundle member names differ from closed outcome")
        member_payloads = {
            name: _read_output_member_at(output_fd, name)
            for name in sorted(expected_names - {PUBLICATION_MARKER})
        }
        _validate_publication_commit(
            output_fd,
            outcome=outcome,
            expected_member_payloads=member_payloads,
            output_directory=target,
        )
        decoded: dict[str, Mapping[str, Any]] = {}
        for name in json_names:
            try:
                value = json.loads(member_payloads[name].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PublicationError("published JSON member cannot be decoded") from exc
            if not isinstance(value, Mapping):
                raise PublicationError("published JSON member is not an object")
            decoded[name] = value
        _validate_closed_payloads(decoded, outcome=outcome)
        return {
            "publication_state": "COMMITTED_ACCEPTED",
            "committed": True,
            "accepted": True,
            "execution_outcome": outcome,
            "output_directory": os.fspath(target),
        }
    except OSError as exc:
        raise PublicationError("published bundle default-consumer I/O failed closed") from exc
    finally:
        if output_fd is not None:
            try:
                os.close(output_fd)
            finally:
                os.close(parent.fd)
        else:
            os.close(parent.fd)


def _publish_closed_bundle(
    output_directory: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    outcome: str,
) -> dict[str, Any]:
    _validate_closed_payloads(payloads, outcome=outcome)
    encoded: dict[str, bytes] = {name: _pretty_json_bytes(payloads[name]) for name in sorted(payloads)}
    sums = "".join(
        f"{_sha256_bytes(encoded[name])}  {name}\n" for name in sorted(encoded)
    ).encode("ascii")
    complete_payloads = {**encoded, SHA256SUMS_FILENAME: sums}
    output_directory = _absolute_without_resolving(output_directory)
    parent = _open_directory_no_symlinks(output_directory.parent, label="output parent")
    output_fd: int | None = None
    output_created = False
    committed = False
    accepted = False
    core_error: Exception | None = None
    warnings: list[str] = []
    opened_output: os.stat_result | None = None
    try:
        _safe_basename(output_directory.name, label="output directory basename")
        try:
            os.mkdir(output_directory.name, 0o700, dir_fd=parent.fd)
            output_created = True
        except FileExistsError as exc:
            raise PublicationContention("exclusive output target already exists") from exc
        except OSError as exc:
            raise PublicationError("exclusive output directory could not be created") from exc
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        output_fd = os.open(output_directory.name, flags, dir_fd=parent.fd)
        opened_output = os.fstat(output_fd)
        if not stat.S_ISDIR(opened_output.st_mode):
            raise PublicationError("exclusive output namespace is not a directory")
        # All ordinary members are O_EXCL-created and file-fsynced first.
        for name in sorted(complete_payloads):
            warnings.extend(_write_exclusive_at(output_fd, name, complete_payloads[name]))
        _publication_fault("precommit_output_fsync")
        os.fsync(output_fd)
        namespace = os.stat(
            output_directory.name, dir_fd=parent.fd, follow_symlinks=False
        )
        if _directory_core_identity(namespace) != _directory_core_identity(opened_output):
            raise PublicationError("output namespace changed before terminal commit")
        _publication_fault("precommit_parent_fsync")
        os.fsync(parent.fd)
        _assert_directory_binding(parent, full=False)
        # This is the sole terminal operation and must stay last.
        marker = _publication_marker_document(
            outcome=outcome,
            member_payloads=complete_payloads,
            output_directory=output_directory,
        )
        warnings.extend(
            _write_exclusive_at(
                output_fd,
                PUBLICATION_MARKER,
                _pretty_json_bytes(marker),
                terminal=True,
            )
        )
        committed = True

        validation_errors = 0
        for attempt in range(2):
            try:
                _publication_fault("post_marker_validation")
                _validate_publication_commit(
                    output_fd,
                    outcome=outcome,
                    expected_member_payloads=complete_payloads,
                    output_directory=output_directory,
                )
                _publication_fault("post_marker_stat")
                namespace = os.stat(
                    output_directory.name, dir_fd=parent.fd, follow_symlinks=False
                )
                if _directory_core_identity(namespace) != _directory_core_identity(
                    opened_output
                ):
                    raise PublicationError(
                        "output namespace identity changed after terminal commit"
                    )
                _assert_directory_binding(parent, full=False)
                accepted = True
                break
            except Exception:
                validation_errors += 1
                if attempt == 0:
                    warnings.append("POST_MARKER_ACCEPTANCE_RETRIED")
        if not accepted:
            warnings.append("POST_MARKER_ACCEPTANCE_PERSISTENT_FAILURE")

        for phase, descriptor, warning in (
            ("post_marker_output_fsync", output_fd, "POST_MARKER_OUTPUT_FSYNC_WARNING"),
            ("post_marker_parent_fsync", parent.fd, "POST_MARKER_PARENT_FSYNC_WARNING"),
        ):
            try:
                _publication_fault(phase)
                os.fsync(descriptor)
            except Exception:
                warnings.append(warning)
    except Exception as exc:
        core_error = exc
    finally:
        if output_fd is not None:
            try:
                if committed:
                    _publication_fault("post_marker_close_output")
            except Exception:
                warnings.append("POST_MARKER_OUTPUT_CLOSE_WARNING")
            try:
                os.close(output_fd)
            except OSError:
                if committed:
                    warnings.append("POST_MARKER_OUTPUT_CLOSE_WARNING")
                elif core_error is None:
                    core_error = PublicationError("precommit output descriptor close failed")
        try:
            if committed:
                _publication_fault("post_marker_close_parent")
        except Exception:
            warnings.append("POST_MARKER_PARENT_CLOSE_WARNING")
        try:
            os.close(parent.fd)
        except OSError:
            if committed:
                warnings.append("POST_MARKER_PARENT_CLOSE_WARNING")
            elif core_error is None:
                core_error = PublicationError("precommit parent descriptor close failed")

    warning_codes = sorted(set(warnings))
    if committed and not accepted:
        raise CommittedNotAcceptedError(
            "terminally committed bundle failed persistent acceptance checks",
            outcome=outcome,
            output_directory=output_directory,
            durability_warnings=warning_codes,
        )
    if committed:
        if core_error is not None:
            raise CommittedNotAcceptedError(
                "terminally committed bundle raised an unexpected post-commit exception",
                outcome=outcome,
                output_directory=output_directory,
                durability_warnings=warning_codes,
            ) from core_error
        publication_state = (
            "COMMITTED_WITH_DURABILITY_WARNING"
            if warning_codes
            else "COMMITTED_ACCEPTED"
        )
        return {
            "status": publication_state,
            "publication_state": publication_state,
            "execution_outcome": outcome,
            "published": True,
            "committed": True,
            "accepted": True,
            "durability_warning": bool(warning_codes),
            "durability_warning_codes": warning_codes,
            "qualified": False,
            "canonical_record_count": 0,
            "output_directory": str(output_directory),
            "member_count_including_terminal_marker": len(complete_payloads) + 1,
            "terminal_marker": PUBLICATION_MARKER,
            "terminal_marker_validated": True,
            "no_overwrite": True,
        }
    if output_created:
        if core_error is not None:
            raise PartialPrecommitError("publication stopped in PARTIAL_PRECOMMIT") from core_error
        raise PartialPrecommitError("publication stopped in PARTIAL_PRECOMMIT")
    if core_error is not None:
        if isinstance(core_error, QualificationError):
            raise core_error
        raise PublicationError("publication failed before output creation") from core_error
    raise PublicationError("publication failed before output creation")


def _build_success_payloads(
    *,
    protocol: Mapping[str, Any],
    protocol_provenance: Mapping[str, Any],
    manifest_payload: bytes,
    asset_payloads: Mapping[str, bytes],
    asset_provenance: Mapping[str, Mapping[str, Any]],
    implementation_binding_audit: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    manifest_audit = _audit_root_manifest(manifest_payload, protocol)
    tar_audit = _audit_tar_equivalence(
        asset_payloads["GSE200303_RAW_TAR"],
        asset_payloads,
        protocol["input_contract"]["tar_member_equivalence"],
    )
    table = protocol["table_contract"]
    primary_design = _audit_design(
        asset_payloads["GSE200302_DESIGN"], table["design"], label="GSE200302 design"
    )
    auxiliary_design = _audit_design(
        asset_payloads["GSE200303_DESIGN"], table["design"], label="GSE200303 design"
    )
    primary_pair_ids = set(primary_design.pair_sequences)
    if primary_pair_ids != set(auxiliary_design.pair_sequences):
        raise TableAuditError("GSE200302 and GSE200303 design pair-key sets differ")
    if primary_design.control_ids != auxiliary_design.control_ids:
        raise TableAuditError("GSE200302 and GSE200303 design control-ID sets differ")
    processed_ids, processed_audit = _audit_processed(
        asset_payloads["GSE200302_PROCESSED"],
        table["processed"],
        primary_pair_ids,
        label="GSE200302 processed companion",
    )
    small_state = _audit_small_plasmid(
        asset_payloads["GSE200303_SMALL_PLASMID"],
        table["small_plasmid"],
        primary_pair_ids,
        primary_design.control_ids,
        label="GSE200303 small-plasmid auxiliary table",
    )
    ivt_state = _audit_ivt(
        asset_payloads["GSE217530_IVT"],
        table["ivt"],
        primary_pair_ids,
        primary_design.control_ids,
        label="GSE217530 IVT auxiliary table",
    )
    assets = []
    for asset in protocol["input_contract"]["assets"]:
        observed = asset_provenance[asset["asset_id"]]
        assets.append(
            {
                "asset_id": asset["asset_id"],
                "accession": asset["accession"],
                "relative_locator": asset["relative_path"],
                "role": asset["role"],
                "format": asset["format"],
                "bytes": observed["bytes"],
                "sha256": observed["sha256"],
                "verified": True,
            }
        )
    integrity = {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": "PASS_INPUT_INTEGRITY_FOR_GAP_AUDIT",
        "aggregate_only": True,
        "protocol_snapshot": {
            "sha256": protocol_provenance["sha256"],
            "bytes": protocol_provenance["bytes"],
            "launch_expected_sha256": protocol_provenance["launch_expected_sha256"],
        },
        "manifest": manifest_audit,
        "assets": assets,
        "unique_input_asset_count": len(assets),
        "tar_equivalence": tar_audit,
        "network_accessed": False,
        "input_payload_writes": 0,
    }
    mechanical = {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": "PASS_MECHANICAL_GAP_AUDIT",
        "aggregate_only": True,
        "primary_design": primary_design.aggregate,
        "auxiliary_design": auxiliary_design.aggregate,
        "processed": processed_audit,
        "small_plasmid": small_state.aggregate,
        "ivt": ivt_state.aggregate,
        "join_audit": {
            "gse200302_gse200303_design_pair_key_sets_equal": True,
            "gse200302_gse200303_control_id_sets_equal": True,
            "processed_barcode_equals_design_id_count": len(processed_ids),
            "design_id_equals_strip_suffix_merged_id_row_count": primary_design.aggregate["identity_equal_row_count"],
            "processed_outcome_blind_attrition_count": len(primary_pair_ids - processed_ids),
            "small_plasmid_design_join_id_count": len(
                set(small_state.arms_by_pair) | small_state.control_ids
            ),
            "ivt_design_join_id_count": len(
                set(ivt_state.arms_by_pair) | ivt_state.control_ids
            ),
            "three_modal_auxiliary_join_pair_count": len(
                processed_ids
                & small_state.complete_pair_ids
                & ivt_state.complete_pair_ids
            ),
            "three_modal_join_role": "AUXILIARY_QC_ONLY_NOT_PRIMARY_ENDPOINT",
        },
        "scope_audit": {
            "observed_member_accession_count": len(protocol["scope"]["member_accessions"]),
            "maximum_independent_study_count": protocol["scope"]["maximum_independent_study_count"],
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "primary_accession": "GSE200302",
            "primary_measurement_column_count": processed_audit["primary_measurement_column_count"],
            "pDNA_gse200303_ivt_role": "AUXILIARY_QC_ONLY_NOT_PRIMARY_ENDPOINT",
            "legacy_freq_is_endpoint": False,
        },
    }
    evidence = protocol["paper_and_external_evidence"]
    a1_master_report = _expected_a1_master_report()
    a1_master_report["nominal_rows"]["value"] = primary_design.aggregate[
        "unique_pair_count"
    ]
    a1_master_report["distinct_candidates"]["value"] = primary_design.aggregate[
        "distinct_candidate_count"
    ]
    a1_master_report["biological_source_groups"]["value"] = primary_design.aggregate[
        "distinct_wt_source_group_count"
    ]
    a1_master_report["eligible_multi_candidate_pools"].update(
        {
            "singleton_pool_count": primary_design.aggregate["singleton_source_pool_count"],
            "two_candidate_pool_count": primary_design.aggregate[
                "two_candidate_source_pool_count"
            ],
            "three_or_more_candidate_pool_count": primary_design.aggregate[
                "three_or_more_candidate_source_pool_count"
            ],
            "ndcg_eligible_pool_count": primary_design.aggregate[
                "ndcg_eligible_source_pool_count"
            ],
        }
    )
    report = {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "execution_outcome": SUCCESS_OUTCOME,
        "protocol_status": PROTOCOL_STATUS,
        "activation_status": ACTIVATION_STATUS,
        "qualification_status": QUALIFICATION_STATUS,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "qualified": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "canonical_materialization_allowed": False,
        "a1_gate": dict(protocol["a1_gate"]),
        "implementation_binding": dict(implementation_binding_audit),
        "paper_native_raw_xtail_replay_status": "NOT_RUN",
        "public_asset_bundle_lineage": dict(evidence["public_asset_bundle_lineage"]),
        "ena_fastq_manifest_bundle_lineage": dict(
            evidence["ena_fastq_manifest_bundle_lineage"]
        ),
        "current_qualifier_integrates_public_asset_bundle": False,
        "current_qualifier_integrates_ena_fastq_manifest_bundle": False,
        "precomputed_gap_evidence_only": dict(evidence["precomputed_gap_evidence_only"]),
        "a1_master_report": a1_master_report,
        "unresolved_blockers": list(protocol["unresolved_blockers"]),
        "redistribution_mode": "PUBLIC_LOCATOR_HASH_AND_AGGREGATE_ONLY_UNTIL_LICENSE_CLOSED",
        "aggregate_only": True,
    }
    payloads = {
        "INPUT_INTEGRITY_AUDIT.json": integrity,
        "MECHANICAL_AUDIT.json": mechanical,
        "QUALIFICATION_REPORT.json": report,
    }
    _validate_closed_payloads(payloads, outcome=SUCCESS_OUTCOME)
    return payloads


def _qualify_preflighted(
    *,
    protocol_path: Path,
    protocol_sha256: str,
    data_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    try:
        protocol, protocol_provenance = _load_protocol(protocol_path, protocol_sha256)
        implementation_binding_audit = _verify_implementation_binding(
            protocol["implementation_binding"],
            protocol["authority"],
            protocol_path.parents[1],
        )
        root = _open_directory_no_symlinks(
            data_root, label="ordinary-public GSE200304 data root"
        )
        identities: dict[str, tuple[str, FileIdentity]] = {}
        try:
            manifest_relative = protocol["input_contract"]["manifest_relative_path"]
            manifest_payload, _, manifest_identity = _read_relative_verified_snapshot(
                root,
                manifest_relative,
                label="root incomplete P0 manifest",
                expected_sha256=protocol["input_contract"]["manifest_expected_sha256"],
                expected_bytes=protocol["input_contract"]["manifest_expected_bytes"],
            )
            identities["__manifest__"] = (manifest_relative, manifest_identity)
            asset_payloads: dict[str, bytes] = {}
            asset_provenance: dict[str, dict[str, Any]] = {}
            for asset in protocol["input_contract"]["assets"]:
                payload, provenance, identity = _read_relative_verified_snapshot(
                    root,
                    asset["relative_path"],
                    label=f"ordinary-public asset {asset['asset_id']}",
                    expected_sha256=asset["sha256"],
                    expected_bytes=asset["bytes"],
                )
                asset_payloads[asset["asset_id"]] = payload
                asset_provenance[asset["asset_id"]] = provenance
                identities[asset["asset_id"]] = (asset["relative_path"], identity)
            if _POST_VERIFIED_INPUT_SNAPSHOT_HOOK is not None:
                _POST_VERIFIED_INPUT_SNAPSHOT_HOOK()
            _assert_directory_binding(root)
            for asset_id, (relative, identity) in identities.items():
                _assert_relative_identity(
                    root, relative, identity, label=f"post-snapshot {asset_id}"
                )
        finally:
            os.close(root.fd)
    except QualificationError:
        raise
    except OSError as exc:
        raise QualificationError("input I/O failed closed") from exc
    payloads = _build_success_payloads(
        protocol=protocol,
        protocol_provenance=protocol_provenance,
        manifest_payload=manifest_payload,
        asset_payloads=asset_payloads,
        asset_provenance=asset_provenance,
        implementation_binding_audit=implementation_binding_audit,
    )
    return _publish_closed_bundle(output_directory, payloads, outcome=SUCCESS_OUTCOME)


def qualify_gse200304_a1(
    *,
    protocol_path: Path,
    protocol_sha256: str,
    data_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    protocol, root, output = _preflight_paths_before_read(
        protocol_path, data_root, output_directory
    )
    return _qualify_preflighted(
        protocol_path=protocol,
        protocol_sha256=protocol_sha256,
        data_root=root,
        output_directory=output,
    )


def _failure_payload(code: str) -> dict[str, dict[str, Any]]:
    if code not in FAILURE_CODES:
        code = "INPUT_INTEGRITY_FAILED"
    return {
        "FAILURE_REPORT.json": {
            "contract_id": CONTRACT_ID,
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "execution_outcome": FAILURE_OUTCOME,
            "status": "FAIL_CLOSED_BEFORE_SUCCESS_BUNDLE_PUBLICATION",
            "failure_code": code,
            "qualified": False,
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "canonical_record_count": 0,
            "training_allowed": False,
            "model_selection_allowed": False,
            "success_bundle_published": False,
            "aggregate_only": True,
        }
    }


def execute_qualification(
    *,
    protocol_path: Path,
    protocol_sha256: str,
    data_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    # Scope rejection is deliberately outside the failure publisher: a forbidden
    # token must terminate before any payload read or output filesystem write.
    protocol, root, output = _preflight_paths_before_read(
        protocol_path, data_root, output_directory
    )
    try:
        return _qualify_preflighted(
            protocol_path=protocol,
            protocol_sha256=protocol_sha256,
            data_root=root,
            output_directory=output,
        )
    except PublicationError:
        # A partially created publication directory is evidence and is never
        # overwritten or repurposed as a failure bundle.
        raise
    except QualificationError as exc:
        return _publish_closed_bundle(output, _failure_payload(exc.code), outcome=FAILURE_OUTCOME)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = execute_qualification(
            protocol_path=args.protocol,
            protocol_sha256=args.protocol_sha256,
            data_root=args.data_root,
            output_directory=args.output_directory,
        )
    except CommittedNotAcceptedError as exc:
        print(
            json.dumps(
                {
                    "status": "COMMITTED_NOT_ACCEPTED",
                    "publication_state": "COMMITTED_NOT_ACCEPTED",
                    "execution_outcome": exc.outcome,
                    "committed": True,
                    "accepted": False,
                    "durability_warning_codes": list(exc.durability_warnings),
                    "qualified": False,
                    "canonical_record_count": 0,
                },
                sort_keys=True,
            )
        )
        return 3
    except PartialPrecommitError:
        print(
            json.dumps(
                {
                    "status": "PARTIAL_PRECOMMIT",
                    "publication_state": "PARTIAL_PRECOMMIT",
                    "committed": False,
                    "accepted": False,
                    "qualified": False,
                    "canonical_record_count": 0,
                },
                sort_keys=True,
            )
        )
        return 3
    except QualificationError as exc:
        print(
            json.dumps(
                {
                    "status": FAILURE_OUTCOME,
                    "failure_code": exc.code,
                    "qualified": False,
                    "canonical_record_count": 0,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["execution_outcome"] == SUCCESS_OUTCOME else 2


if __name__ == "__main__":
    raise SystemExit(main())
