#!/usr/bin/env python3
"""Read-only static validator for the mRNA-XEditFlow Route A V3 A0 bundle.

The default command only reads Git-sized public authority/config/registry/schema
files below the selected repository root.  It never imports project training
code, initializes sealed state, follows restricted-store pointers, or imports
PyTorch.  ``--write-manifests`` is the sole opt-in write operation and rewrites
only the two deterministic schema manifest files in ``schemas/route_a_v3``.

An empty issue list means that the A0 static engineering contract is coherent;
it is not a scientific, data-qualification, model, guidance, or Route-A PASS.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml


CONTRACT_ID = "mrna_xeditflow_route_a_v3"
VERSION = "3.0.0"
CONFIG_STATUS = "ACTIVE_AUTHORITATIVE_CONTRACT"
SOURCE_CONTRACT_PATH = "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna v3.md"
SOURCE_CONTRACT_SHA256 = "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982"
GOAL_PATH = "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md"
CONFIG_PATH = "configs/route_a_v3.yaml"
A1_QUALIFICATION_CONFIG_PATH = "configs/route_a_v3_a1_qualification.json"
SUPERSESSION_PATH = "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml"
DEC019_AMENDMENT_PATH = "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec019.yaml"
DECISION_LOG_PATH = "docs/execution/route_a_v3_decision_log.yaml"
REGISTRY_MANIFEST_PATH = "docs/execution/route_a_v3_registry_manifest.json"
A1_INTERIM_PATH = "docs/execution/route_a_v3_a1_interim.yaml"
EXPECTED_A1_INTERIM_SHA256 = "552f705445a36df99a5ae071c85625c729ad2f69f0a375bb7b3118c3b400e16c"
GSE200304_DEC019_POST_ADJUDICATION_LEDGER_AT = "2026-08-11T20:32:48+08:00"
GSE200304_DEC019_POST_ADJUDICATION_MANIFEST_AT = "2026-08-11T20:35:15+08:00"
ACTIVE_AMENDMENT_DECISION_IDS = ["V3-DEC-017", "V3-DEC-018", "V3-DEC-019"]
DEC019_LEAF_AUTHORITY_SHA256 = {
    DEC019_AMENDMENT_PATH: "8c82e564398f0735fe4976f875fe91f053937b05044e5232e237694a2b36e1ca",
    DECISION_LOG_PATH: "b537a2ce19e4bb8b099f05df4ba383b56b8957cbc7be0b5954c9c11d741eb23b",
    "docs/execution/route_a_v3_data_role_registry.yaml": "4d14ebd1a6adc04a344165f775df8586ef9f8f0461fdcac08649d0644d9956f2",
    "docs/execution/route_a_v3_split_registry.yaml": "2764d471c09a27da889b690cac317ac582bf9f25b79b6a34ac491f2e0b434929",
    "docs/execution/route_a_v3_task_registry.yaml": "6c6659ef0e9ddbbbba002f77d39d388dbdacc7b383e98ebb30a1580d590d85b4",
    "docs/execution/route_a_v3_task_split_matrix.yaml": "dd340bcfb291138b862c5858daa28910c44689299647b468aedcc48b3d90b534",
    "docs/execution/route_a_v3_claim_evidence_matrix.yaml": "25b62c17320032c764f986892647d4548065cac3a6d42414f96737da3fb3cbad",
    A1_QUALIFICATION_CONFIG_PATH: "fe3f7736c1f64b362ebda683ca571fc1a84e1fff36aed3a9ae67272665ba2343",
}
GSE114002_DEC019_SUCCESSOR_CONFIG_PATH = (
    "configs/route_a_v3_gse114002_dec019_true_a2_activation_v2.json"
)
GSE114002_DEC019_SUCCESSOR_SCRIPT_PATH = (
    "scripts/route_a_v3/adjudicate_gse114002_dec019_true_a2.py"
)
GSE114002_DEC019_SUCCESSOR_TEST_PATH = (
    "tests/route_a_v3/test_adjudicate_gse114002_dec019_true_a2.py"
)
GSE114002_DEC019_SUCCESSOR_INITIAL_I_SHA256 = (
    "329461331a0a5d47f94dcafef502482f1416ffa37ef9d91cc0f70a9ed912513b"
)
GSE114002_DEC019_SUCCESSOR_CORE_SHA256 = (
    "1c3e4a7aa412e245f6f4680677db60b8241d7873fa126756791bdb0b58f9233a"
)
GSE114002_DEC019_SUCCESSOR_SCRIPT_SHA256 = (
    "20b1d6e7824921d31ea6a0ab5ecac93707ae3acf2789b11019574813e33c1b6c"
)
GSE114002_DEC019_SUCCESSOR_TEST_SHA256 = (
    "a1ae6e73ee4f9a44f5de6db9ff09be175ecd1648fc592ff2c9f073a7184302ee"
)
GSE200304_DEC019_SUCCESSOR_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v2.json"
)
GSE200304_DEC019_SUCCESSOR_SCRIPT_PATH = (
    "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1.py"
)
GSE200304_DEC019_SUCCESSOR_TEST_PATH = (
    "tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1.py"
)
GSE200304_DEC019_SUCCESSOR_INITIAL_I_SHA256 = (
    "d5f616c6802599bf803f351c817abbe87c91a79a0c98d39f737274fd7e21e1c1"
)
GSE200304_DEC019_SUCCESSOR_CORE_SHA256 = (
    "6cbc215d38adf3b3d15de314f674b2ae02b2f1a1a733cb4dec3d75d8f9480943"
)
GSE200304_DEC019_SUCCESSOR_SCRIPT_SHA256 = (
    "7ad39104d6fc908a23c538932a4ad9249e24131c383ce088f7010657d1c8191b"
)
GSE200304_DEC019_SUCCESSOR_TEST_SHA256 = (
    "a153c54c9b7fff228d4702e52e2cc9521afcbe14bfb5658fbc1b8fe96ae74c09"
)
GSE114002_DEC019_SUCCESSOR_LINEAGE_ID = (
    "gse114002_dec019_true_a2_successor_adjudicator_v2"
)
GSE200304_DEC019_SUCCESSOR_LINEAGE_ID = (
    "gse200304_dec019_reported_endpoint_a1_successor_adjudicator_v2"
)
GSE200304_DEC019_V3_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
)
GSE200304_DEC019_V3_SCRIPT_PATH = (
    "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
)
GSE200304_DEC019_V3_TEST_PATH = (
    "tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
)
GSE200304_DEC019_LINEAGE_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_dec019_canonical_row_lineage_gate_v1.json"
)
GSE200304_DEC019_LINEAGE_SCRIPT_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_canonical_row_lineage_gate.py"
)
GSE200304_DEC019_LINEAGE_TEST_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_canonical_row_lineage_gate.py"
)
GSE200304_DEC019_NEGATIVE_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_dec019_negative_gate_pack_v1.json"
)
GSE200304_DEC019_NEGATIVE_SCRIPT_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_negative_gate_pack.py"
)
GSE200304_DEC019_NEGATIVE_TEST_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_negative_gate_pack.py"
)
GSE200304_DEC019_V3_CONFIG_SHA256 = (
    "88fa21a08df60935f3d2d1bf44c6573889c22c110021146acf241fd92d6b5a13"
)
GSE200304_DEC019_V3_CONFIG_CORE_SHA256 = (
    "13394ac6a9b9ec6e6241d0d9b1048ecfa5c90874c7447991fc2a8248a574c170"
)
GSE200304_DEC019_V3_DESCRIPTOR_SET_SHA256 = (
    "14223d0193e4b3a4a3c1d98a5894849dd429e6eed021ff98e6697e73ac286a40"
)
GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256 = {
    GSE200304_DEC019_V3_SCRIPT_PATH: (
        "9cd4411fcb02e1feed913b799296351e38ab9071b9506611318645e41b8dbbfe"
    ),
    GSE200304_DEC019_V3_TEST_PATH: (
        "8e7b188cfa2e5015fa307acad980f9ff2f45145943384fcadb50d67b1263e1db"
    ),
    GSE200304_DEC019_LINEAGE_CONFIG_PATH: (
        "0904495aec2acd6470f3d827ab926bfa94b17ccf146e268df927c6515e38f527"
    ),
    GSE200304_DEC019_LINEAGE_SCRIPT_PATH: (
        "72f946063754b49fbc309465a1a44dac0c9b531eae6713e43ad4702cdbdbfe52"
    ),
    GSE200304_DEC019_LINEAGE_TEST_PATH: (
        "829bc7c70552b8c1aa53d7f7fc06592c1228a0bafe949d431fc419af38afc0e4"
    ),
    GSE200304_DEC019_NEGATIVE_CONFIG_PATH: (
        "fea1c56d21dc848b535c876b31799eb6ccca48ecf9c4d8a58a7dbc7f7187297e"
    ),
    GSE200304_DEC019_NEGATIVE_SCRIPT_PATH: (
        "3716ccf6492b067c374fde38f58d7b46e878dec7c58192b94030e05161e33205"
    ),
    GSE200304_DEC019_NEGATIVE_TEST_PATH: (
        "5523d3a1b5216963bd3793ba9ec3f8cf15d9a01867192ddf1eae32ac0e327948"
    ),
}
GSE200304_DEC019_LINEAGE_GATE_LINEAGE_ID = (
    "gse200304_dec019_canonical_row_lineage_gate_v1"
)
GSE200304_DEC019_NEGATIVE_GATE_PACK_LINEAGE_ID = (
    "gse200304_dec019_negative_gate_pack_v1"
)
GSE200304_DEC019_ADJUDICATION_LINEAGE_ID = (
    "gse200304_dec019_reported_endpoint_a1_adjudication_v3"
)
GSE200304_DEC019_POST_ADJUDICATION_BLOCKERS = [
    "BIOLOGICAL_GROUP_AUTHORITY_NOT_PASS",
    "CANONICAL_REPORTED_ENDPOINT_SEMANTICS_NOT_PASS",
    "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS",
    "LICENSE_RIGHTS_NOT_PASS",
    "OUTCOME_BLIND_SPLIT_LEAKAGE_NOT_PASS",
    "PREFROZEN_POWER_PRECISION_NOT_PASS",
    "ROW_REPLICATE_OR_VALID_SE_NOT_PASS",
]
GSE200304_DEC019_POST_ADJUDICATION_INPUT_STATUS_COUNTS = {
    "PASS": 1,
    "BLOCKED": 3,
    "UNKNOWN_NOT_ASSERTED": 2,
    "NOT_RUN": 2,
}
DEC019_SUCCESSOR_DYNAMIC_CONFIG_PATHS = frozenset(
    {
        GSE114002_DEC019_SUCCESSOR_CONFIG_PATH,
        GSE200304_DEC019_SUCCESSOR_CONFIG_PATH,
        GSE200304_DEC019_V3_CONFIG_PATH,
    }
)
GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_PATH = (
    "docs/execution/gse114002_public_authority_gap_audit_v1.json"
)
GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_SHA256 = (
    "3be184767bd297f2b50deff2b056e30e2229b970e9bbf0a9c3e5656e3147821f"
)
GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_STATUS = (
    "PUBLIC_AUTHORITY_GAPS_AUDITED_NOT_QUALIFIED"
)
GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_LINEAGE_ID = (
    "gse114002_public_authority_gap_audit_v1"
)
SCIENTIFIC_M0_HISTORY_PATH = "docs/contracts/history/mrna_v2_readiness_audit_20260807.md"
SCIENTIFIC_M0_HISTORY_SHA256 = "a8eb4f49ede793a8eae2037db9f46f044056d37610ec92482666a8242a52fa30"
SEALED_GUARD_PATH = "scripts/route_a_v3/sealed_guard.py"
SEALED_RUNNER_PATH = "scripts/e0x/run_e0x_final.py"
VALIDATOR_PATH = "scripts/route_a_v3/validate_a0_bundle.py"
GSE200302_ROLE_CONFIG_PATH = "configs/route_a_v3_gse200302_srr_role_authority.json"
GSE200302_ROLE_BUILDER_PATH = "scripts/route_a_v3/build_gse200302_srr_role_authority.py"
GSE200302_ROLE_TEST_PATH = "tests/route_a_v3/test_gse200302_srr_role_authority.py"
GSE200304_PUBLISHED_ENDPOINT_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_published_endpoint_a1.json"
)
GSE200304_PUBLISHED_ENDPOINT_SCRIPT_PATH = (
    "scripts/route_a_v3/qualify_gse200304_published_endpoint_a1.py"
)
GSE200304_PUBLISHED_ENDPOINT_TEST_PATH = (
    "tests/route_a_v3/test_qualify_gse200304_published_endpoint_a1.py"
)
INTEGRITY_GUARD_TEST_PATH = "tests/route_a_v3/test_a0_integrity_guards.py"
GSE200302_ROLE_ARTIFACT_ROOT = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200302/"
    "GSE200302_SRR_ROLE_AUTHORITY_20260810T230315P0800_e3b724d"
)
GSE200302_ROLE_BUNDLE_DIGEST = (
    "d3da1868158f0bb740d1e3ef2a84fa204d0ff492a2f841ff46d11396ddc8b430"
)
GSE200302_ROLE_PROTOCOL_CORE_SHA256 = (
    "d407504d42c390b32aaa0eff953c168b1e9cc4991afcd8530870144c78a1d526"
)
GSE200302_ROLE_PROTOCOL_SCHEMA = "route_a_v3_gse200302_srr_role_authority.v1"
GSE200302_ROLE_PROTOCOL_ID = "ROUTE_A_V3_GSE200302_SRR_ROLE_AUTHORITY_V1"
GSE200302_ROLE_BINDING_MODE = "TWO_COMMIT_CONFIG_ONLY_NON_SELF_REFERENTIAL_V1"
GSE200302_ROLE_BINDING_ACTIVATION_RULE = (
    "Commit this UNKNOWN protocol with the implementation script and test, then create exactly "
    "one separate config-only binding commit that changes only implementation_binding.status, "
    "implementation_commit, implementation_script_sha256, and implementation_test_sha256. Runtime "
    "fails before official-source or output access until that binding exists."
)
GSE200302_ROLE_MEASUREMENT_FAMILIES = ["High_Poly", "Low_Poly", "pDNA", "Total_RNA"]
GSE200302_ROLE_REPLICATES = [1, 2, 3, 4, 5, 6]
GSE200304_PUBLISHED_ENDPOINT_ARTIFACT_ID = (
    "GSE200304_PUBLISHED_ENDPOINT_A1_20260811T044050+0800_d06bb99"
)
GSE200304_PUBLISHED_ENDPOINT_ARTIFACT_ROOT = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
    "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
    f"{GSE200304_PUBLISHED_ENDPOINT_ARTIFACT_ID}"
)
GSE200304_PUBLISHED_ENDPOINT_BLOCKERS = [
    "OWNER_POLICY_FOR_PUBLISHED_ENDPOINT_USE_NOT_FROZEN",
    "CHECKPOINT_SPECIFIC_ENDPOINT_USE_NOT_CLEARED",
    "BIOLOGICAL_SOURCE_GROUP_AUTHORITY_NOT_CLOSED",
    "CURRENT_AUTHORITY_80S_BLOCKER_SCOPE_NOT_ROUTED_FOR_PUBLISHED_ENDPOINT_REUSE",
    "OUTCOME_BLIND_SPLIT_AND_LEAKAGE_POLICY_NOT_FROZEN",
    "POWER_AND_CONFIDENCE_INTERVAL_ADEQUACY_NOT_ESTABLISHED",
    "CANONICAL_REPORTED_ENDPOINT_SEMANTICS_NOT_ADJUDICATED",
    "ROW_LEVEL_REPLICATE_AND_STANDARD_ERROR_ADJUDICATION_NOT_CLOSED",
]
GSE200304_PUBLISHED_ENDPOINT_EXPECTED_RECORD = {
    "artifact_id": GSE200304_PUBLISHED_ENDPOINT_ARTIFACT_ID,
    "path": GSE200304_PUBLISHED_ENDPOINT_ARTIFACT_ROOT,
    "dataset_id": "GSE200304",
    "record_type": "PUBLISHED_ENDPOINT_AGGREGATE_EVIDENCE",
    "evidence_role": "AGGREGATE_AUDIT_ONLY_PENDING_OWNER_POLICY",
    "publication_state": "COMMITTED_ACCEPTED",
    "execution_outcome": "ENGINEERING_SUCCESS_IMMUTABLY_BLOCKED",
    "qualification_status": "BLOCKED_NOT_QUALIFIED",
    "accepted": True,
    "committed": True,
    "postcommit_warning_codes": [],
    "independent_consumer_validation_status": "PASS",
    "terminal_marker_written_last": True,
    "no_acceptance_critical_read_after_commit": True,
    "publication_closure": {
        "actual_directory_member_count": 5,
        "terminal_marker_declared_member_count_excluding_marker": 4,
        "terminal_marker_declared_member_names": [
            "INPUT_INTEGRITY_AUDIT.json",
            "PUBLISHED_ENDPOINT_AUDIT.json",
            "QUALIFICATION_REPORT.json",
            "SHA256SUMS",
        ],
        "sha256sums_listed_payload_count": 3,
        "sha256sums_listed_payload_names": [
            "INPUT_INTEGRITY_AUDIT.json",
            "PUBLISHED_ENDPOINT_AUDIT.json",
            "QUALIFICATION_REPORT.json",
        ],
    },
    "remote_authority": {
        "branch": "routea-v3-a1-20260810",
        "head_commit": "d06bb991ca9c9052671ee5c5ad7d92dfb69b0189",
        "origin_head_commit": "d06bb991ca9c9052671ee5c5ad7d92dfb69b0189",
        "worktree_and_index_clean": True,
    },
    "implementation_binding": {
        "status": "PASS_BOUND_IMPLEMENTATION",
        "implementation_commit": "84fc6932de32fe0de8e5ddf540e14dee62a2b723",
        "binding_commit": "d06bb991ca9c9052671ee5c5ad7d92dfb69b0189",
        "implementation_to_binding_diff_is_config_only": True,
        "protocol_config_path": GSE200304_PUBLISHED_ENDPOINT_CONFIG_PATH,
        "protocol_config_sha256": "92fc3a3859f7a8949ace67fa4b03a14e8ad102eb257d4f95cace01ea535b41af",
        "production_script_path": GSE200304_PUBLISHED_ENDPOINT_SCRIPT_PATH,
        "production_script_sha256": "687268524c7426eb4d3d450e71d13c7c478372162e0c084ffe90c8bb12764308",
        "focused_test_path": GSE200304_PUBLISHED_ENDPOINT_TEST_PATH,
        "focused_test_sha256": "173cad716fbdb2590e82ea54a91776ef61e7ab9eb7b596b694b5aa8609d44ad0",
    },
    "gate_snapshot": {
        "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0,
        "true_a2_dense_study_contribution": 0,
        "qualified": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "canonical_record_count": 0,
    },
    "access_and_materialization_boundary": {
        "raw_reads_or_alignments_opened": False,
        "raw_fastq_body_read_count": 0,
        "raw_replay_run_count": 0,
        "paper_native_xtail_replay_run_count": 0,
        "gpu_work_started": False,
        "row_level_payload_included": False,
        "row_identifier_payload_included": False,
        "sequence_payload_included": False,
        "effect_value_payload_included": False,
        "gene_payload_included": False,
        "barcode_payload_included": False,
        "annotation_label_payload_included": False,
        "canonical_read_count": 0,
        "canonical_write_count": 0,
    },
    "mechanical_aggregates": {
        "table_s2": {
            "raw_row_count": 13850,
            "unique_content_row_count": 13836,
            "exact_duplicate_excess_row_count": 14,
            "duplicated_pair_id_count": 7,
            "deduplicated_pair_count": 6885,
            "deduplicated_control_count": 66,
            "central_single_snv_pair_count": 6885,
            "design_orientation_counts": {
                "forward": 3497,
                "reverse_complement": 3388,
                "unresolved": 0,
            },
        },
        "table_s3": {
            "primary_data_row_count": 13544,
            "primary_pair_key_count": 6772,
            "total_poly_complete_pair_count": 6547,
            "total_poly_na_pair_count": 225,
            "high_poly_complete_pair_count": 6538,
            "high_poly_na_pair_count": 234,
            "table_s2_absent_from_table_s3_pair_count": 113,
            "post_dedup_primary_attrition_count": 338,
            "both_comparisons_complete_pair_count": 6538,
            "primary_only_complete_pair_count": 9,
            "secondary_only_complete_pair_count": 0,
            "neither_comparison_complete_pair_count": 225,
            "joined_orientation_counts": {
                "forward": 3451,
                "reverse_complement": 3321,
                "unresolved": 0,
            },
            "control_sheet_data_cell_read_count": 0,
            "translation_formula_cell_count": 13544,
            "translation_cached_string_cell_count": 13544,
            "translation_cached_values_role": "DESCRIPTIVE_ONLY_NOT_MEMBERSHIP_OR_GATE",
        },
        "endpoint_boundary": {
            "primary_complete_distinct_wt_201nt_proxy_group_count": 6544,
            "singleton_proxy_group_count": 6541,
            "two_candidate_proxy_group_count": 3,
            "biological_source_group_authority_closed": False,
            "study_level_reported_biological_replicate_count": 6,
            "row_level_effective_replicate_count": None,
            "standard_error": None,
            "power_effective_n": None,
            "true_a2_dense_pool_count": 0,
            "true_a2_dense_candidate_count": 0,
        },
    },
    "unresolved_blockers": GSE200304_PUBLISHED_ENDPOINT_BLOCKERS,
    "runtime_sync_status": "PENDING_NO_EVT_037",
}
GSE200304_PUBLISHED_ENDPOINT_EXPECTED_FILES = [
    {
        "path": f"{GSE200304_PUBLISHED_ENDPOINT_ARTIFACT_ROOT}/{name}",
        "bytes": size,
        "sha256": digest,
    }
    for name, size, digest in (
        ("INPUT_INTEGRITY_AUDIT.json", 3610, "e87723673dfea6dca654b670d1c05f331f240a53d52d81d1207fbfc50d9a4fe8"),
        ("PUBLISHED_ENDPOINT_AUDIT.json", 4981, "d849da8cc29a2a4419c85d69e5084736b6b41b03cac90263aa2620be3fe3acc7"),
        ("QUALIFICATION_REPORT.json", 2095, "006db8da47dc2bbc0c313a156ae16ab79a3f6aebe324d37806820ac9240b100d"),
        ("SHA256SUMS", 281, "e1720881f8bcfaaea1fef613dd4ee059c08da1bbd11bafc32a8fccdea0a43515"),
        ("PUBLICATION_COMMIT.json", 973, "f1e5d0752bcc12db0b0eaabe0e75efdb6f2c48dfba4c3bae6bff99a302194cfc"),
    )
]
GSE114002_ENDPOINT_GEOMETRY_CONFIG_PATH = (
    "configs/route_a_v3_gse114002_endpoint_geometry_reconciliation_v2.json"
)
GSE114002_ENDPOINT_GEOMETRY_SCRIPT_PATH = (
    "scripts/route_a_v3/audit_gse114002_endpoint_geometry_reconciliation_v2.py"
)
GSE114002_ENDPOINT_GEOMETRY_TEST_PATH = (
    "tests/route_a_v3/test_audit_gse114002_endpoint_geometry_reconciliation_v2.py"
)
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID = (
    "gse114002_endpoint_geometry_reconciliation_v2_attempt_001_failure"
)
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID = (
    "gse114002_endpoint_geometry_reconciliation_v2_attempt_002_mechanical_closure"
)
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_ARTIFACT_ID = (
    "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_20260811T073353P0800_998d030"
)
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_ARTIFACT_ID = (
    "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_20260811T075711P0800_a148c73"
)
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_ROOT = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
    f"{GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_ARTIFACT_ID}"
)
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_ROOT = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
    f"{GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_ARTIFACT_ID}"
)
GSE114002_ENDPOINT_GEOMETRY_SOURCE = {
    "path": "/mnt/cunyuliu/mrna_editflow_p0/GSE114002/GSM3130443_designed_library.csv.gz",
    "bytes": 17332142,
    "sha256": "b72ac298cb0f4d21f911d330c0def06f8d94f15d9f8cc22f3a50ae87a7ef7ee5",
    "aggregate_only": True,
    "row_or_sequence_payload_included": False,
}
GSE114002_ENDPOINT_GEOMETRY_ZERO_GATE = {
    "ordinary_study_contribution": 0,
    "a1_intervention_study_contribution": 0,
    "true_a2_dense_study_contribution": 0,
    "canonical_record_count": 0,
    "qualified": False,
    "true_a2_claim_established": False,
    "training_allowed": False,
    "model_selection_allowed": False,
    "next_phase_authorized": False,
    "canonical_materialization_allowed": False,
}
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_BLOCKERS = [
    "CHECKPOINT_SPECIFIC_EXPOSURE_UNKNOWN_NOT_ASSERTED",
    "FIELD_AND_BIOLOGICAL_SOURCE_AUTHORITY_UNKNOWN_NOT_ASSERTED",
    "FULL_CONSTRUCT_PREFIX_REPORTER_RNA_CHEMISTRY_UNKNOWN_NOT_ASSERTED",
    "HAMMING_DISTANCE_DISTRIBUTION_RECONCILIATION_MISMATCH",
    "LICENSE_AND_REDISTRIBUTION_RIGHTS_UNKNOWN_NOT_ASSERTED",
    "NEAR_DUPLICATE_SPLIT_AND_LEAKAGE_AUDIT_NOT_RUN",
    "OWNER_UNCERTAINTY_POLICY_UNKNOWN_NOT_ASSERTED",
    "PREFROZEN_GROUP_POWER_NOT_RUN",
    "TWO_STAGE_GLOBAL_NORMALIZATION_RECONCILIATION_MISMATCH",
]
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_BLOCKERS = [
    "CHECKPOINT_SPECIFIC_EXPOSURE_UNKNOWN_NOT_ASSERTED",
    "FIELD_AND_BIOLOGICAL_SOURCE_AUTHORITY_UNKNOWN_NOT_ASSERTED",
    "FULL_CONSTRUCT_PREFIX_REPORTER_RNA_CHEMISTRY_UNKNOWN_NOT_ASSERTED",
    "LICENSE_AND_REDISTRIBUTION_RIGHTS_UNKNOWN_NOT_ASSERTED",
    "NEAR_DUPLICATE_SPLIT_AND_LEAKAGE_AUDIT_NOT_RUN",
    "OWNER_UNCERTAINTY_POLICY_UNKNOWN_NOT_ASSERTED",
    "PREFROZEN_GROUP_POWER_NOT_RUN",
]
GSE114002_ENDPOINT_GEOMETRY_CLOSED_BLOCKERS = [
    "TWO_STAGE_GLOBAL_NORMALIZATION_RECONCILIATION_MISMATCH",
    "HAMMING_DISTANCE_DISTRIBUTION_RECONCILIATION_MISMATCH",
]
GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_EXPECTED_RECORD = {
    "path": GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_PATH,
    "bytes": 24861,
    "sha256": GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_SHA256,
    "dataset_id": "GSE114002",
    "record_id": "GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_V1",
    "record_type": "PUBLIC_AUTHORITY_GAP_AUDIT_AGGREGATE_ONLY",
    "evidence_role": "PUBLIC_AUTHORITY_SEARCH_AND_GAP_CLASSIFICATION_ONLY",
    "status": GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_STATUS,
    "audited_at": "2026-08-11T10:24:16+08:00",
    "endpoint_attempt_lineage_ids_preserved": [
        GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID,
        GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID,
    ],
    "predecessor_runtime_event_id": "A1-EVT-039",
    "predecessor_runtime_event_name": "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_ATTEMPT_LINEAGE_SYNCED_GATE_UNCHANGED",
    "source_registry_count": 16,
    "field_and_source_claim_count": 12,
    "construct_and_chemistry_claim_count": 5,
    "license_claim_count": 4,
    "checkpoint_family_count": 4,
    "engineering_items_closed_by_this_audit": [
        "PUBLIC_AUTHORITY_EVIDENCE_SEARCH_AND_GAP_CLASSIFICATION_COMPLETED"
    ],
    "science_blockers_closed_by_this_audit": [],
    "blocker_count": 7,
    "unresolved_blockers": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_BLOCKERS,
    "gate_snapshot": {
        "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0,
        "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0,
        "qualified": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    },
    "aggregate_only": True,
    "row_or_sequence_payload_included": False,
    "per_member_or_model_weight_hash_included": False,
    "restricted_or_sealed_contact": False,
    "runtime_sync_status": "PENDING_NO_EVT_040",
}
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_EXPECTED_RECORD = {
    "artifact_id": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_ARTIFACT_ID,
    "path": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_ROOT,
    "recorded_at": "2026-08-11T07:33:53+08:00",
    "dataset_id": "GSE114002",
    "record_type": "ENDPOINT_GEOMETRY_RECONCILIATION_AGGREGATE_EVIDENCE",
    "evidence_role": "HISTORICAL_FAILED_MECHANICAL_RECONCILIATION_PRESERVED",
    "publication_state": "COMMITTED_ACCEPTED",
    "status": "MECHANICAL_RECONCILIATION_FAILED_NOT_QUALIFIED",
    "is_current": False,
    "failure_preserved": True,
    "superseded_by_lineage_id": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID,
    "implementation_binding": {
        "status": "PASS_BOUND_IMPLEMENTATION",
        "implementation_commit": "78b06434a9b94eaf5149dff7f6bb6b2d58e76ade",
        "binding_commit": "998d030a51737bfa1e27580efe8b89e22ae39149",
        "implementation_to_binding_diff_is_config_only": True,
        "protocol_config_path": GSE114002_ENDPOINT_GEOMETRY_CONFIG_PATH,
        "protocol_config_bytes": 12936,
        "protocol_config_sha256": "1fe9eaaba3790b91da8b92612050c71c52125f6fe18c2a7815fd398a680650f1",
        "production_script_path": GSE114002_ENDPOINT_GEOMETRY_SCRIPT_PATH,
        "production_script_sha256": "29a510e0471803ce189fc3c66f6a0b1ad0d6c19b14db2f1e51cedff846ee40da",
        "focused_test_path": GSE114002_ENDPOINT_GEOMETRY_TEST_PATH,
        "focused_test_sha256": "973cce10449341df8f3d95a798789a224a797095e37118a1af43d723bd8a1d0e",
    },
    "source_provenance": GSE114002_ENDPOINT_GEOMETRY_SOURCE,
    "gate_snapshot": GSE114002_ENDPOINT_GEOMETRY_ZERO_GATE,
    "blocker_count": 9,
    "unresolved_blockers": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_BLOCKERS,
    "runtime_sync_status": "PENDING_NO_EVT_039",
    "terminal_marker_written_last": True,
}
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_EXPECTED_RECORD = {
    "artifact_id": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_ARTIFACT_ID,
    "path": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_ROOT,
    "recorded_at": "2026-08-11T07:57:11+08:00",
    "dataset_id": "GSE114002",
    "record_type": "ENDPOINT_GEOMETRY_RECONCILIATION_AGGREGATE_EVIDENCE",
    "evidence_role": "CURRENT_MECHANICAL_CLOSURE_NOT_STUDY_QUALIFICATION",
    "publication_state": "COMMITTED_ACCEPTED",
    "status": "MECHANICAL_ENDPOINT_RECONCILED_NOT_QUALIFIED",
    "is_current": True,
    "previous_attempt_lineage_id": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID,
    "previous_failure_preserved": True,
    "implementation_binding": {
        "status": "PASS_BOUND_IMPLEMENTATION",
        "implementation_commit": "1543f09e74643a9a36b89742c7a1cc458b6b0d56",
        "binding_commit": "a148c737101ed8d0e24209233b86a55e2710633e",
        "implementation_to_binding_diff_is_config_only": True,
        "protocol_config_path": GSE114002_ENDPOINT_GEOMETRY_CONFIG_PATH,
        "protocol_config_bytes": 13139,
        "protocol_config_sha256": "560c19c3cf6d2e41f8b05978584ce884dd2beb824cdca9392a597ff406120ff8",
        "production_script_path": GSE114002_ENDPOINT_GEOMETRY_SCRIPT_PATH,
        "production_script_sha256": "46e41b387357da0deae7139ac675638075cbae4fc24ac5c9583e969adfa8308d",
        "focused_test_path": GSE114002_ENDPOINT_GEOMETRY_TEST_PATH,
        "focused_test_sha256": "8a0b04b3305f50fb1685ab310efaead15a1e3f4ca5eae4abb01d7f44a9d7be29",
    },
    "source_provenance": GSE114002_ENDPOINT_GEOMETRY_SOURCE,
    "gate_snapshot": GSE114002_ENDPOINT_GEOMETRY_ZERO_GATE,
    "mechanical_diagnostics": {
        "eligible_provisional_pool_count": 959,
        "eligible_provisional_distinct_candidate_count": 3899,
        "diagnostic_only_not_effective_n": True,
        "diagnostic_only_not_study_count": True,
    },
    "mechanically_closed_from_attempt_001": GSE114002_ENDPOINT_GEOMETRY_CLOSED_BLOCKERS,
    "blocker_count": 7,
    "unresolved_blockers": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_BLOCKERS,
    "runtime_sync_status": "PENDING_NO_EVT_039",
    "terminal_marker_written_last": True,
}
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_EXPECTED_FILES = [
    {
        "path": f"{GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_ROOT}/{name}",
        "bytes": size,
        "sha256": digest,
    }
    for name, size, digest in (
        ("INPUT_INTEGRITY_AUDIT.json", 640, "d01377c1e05bc85beafba4893a05255e7f590c9518a8ee069fca607131d1b80b"),
        ("ENDPOINT_RECONCILIATION_AUDIT.json", 1907, "bd5f89751fa69d61aa124e7424c04f78ce75ce3cfab16145c5f47ef7799b38b9"),
        ("POOL_GEOMETRY_RECONCILIATION_AUDIT.json", 2250, "f4910ba655433076ddc44085770ef2d36df6169d2668030bcfbcb7372a46acc3"),
        ("QUALIFICATION_REPORT.json", 2450, "7f5269866be96deab3e41099181c53294f73e4e14c6a4138968a3e578b3c89d0"),
        ("SHA256SUMS", 392, "a7556e12c26c062698ffa0b03407ae521dc8238dfd5437820a5451674ee06d0e"),
        ("PUBLICATION_COMMIT.json", 1174, "07e17411b32c630b25e66f23212a65939c6d259a5fa554a260cc03505346bddf"),
    )
]
GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_EXPECTED_FILES = [
    {
        "path": f"{GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_ROOT}/{name}",
        "bytes": size,
        "sha256": digest,
    }
    for name, size, digest in (
        ("INPUT_INTEGRITY_AUDIT.json", 640, "d01377c1e05bc85beafba4893a05255e7f590c9518a8ee069fca607131d1b80b"),
        ("ENDPOINT_RECONCILIATION_AUDIT.json", 1910, "0d515639174ad0f1daa6dd9e46197984bd3fa43b769023a2d6d1cc2ea0d1e641"),
        ("POOL_GEOMETRY_RECONCILIATION_AUDIT.json", 2423, "40a7cef042e4b4f8db4d6db4be469a64fefad94d55f43ff54fbca2a580b23d21"),
        ("QUALIFICATION_REPORT.json", 2325, "34c2cc0c861286f8e22bf1ba4026d5f754254ffb1bd10a4b989a9546e874a9c3"),
        ("SHA256SUMS", 392, "9fbd4970a3974786cd246715dfef029bd0ad87718b4eff48266d355b003ef9f0"),
        ("PUBLICATION_COMMIT.json", 1172, "522568111b68fc68508dfb2cc82b48121c3f890a7a479ef3510fada56e38663a"),
    )
]
GSE149487_PLUMAGE_PROTOCOL_PATH = "configs/route_a_v3_gse149487_a1_qualification.json"
GSE149487_PLUMAGE_ASSET_MANIFEST_PATH = (
    "configs/route_a_v3_gse149487_asset_manifest_v2.json"
)
GSE149487_PLUMAGE_HELPER_PATH = (
    "scripts/route_a_v3/reconstruct_gse149487_plumage.py"
)
GSE149487_PLUMAGE_QUALIFIER_PATH = (
    "scripts/route_a_v3/qualify_gse149487_plumage.py"
)
GSE149487_PLUMAGE_TEST_PATH = (
    "tests/route_a_v3/test_qualify_gse149487_plumage.py"
)
GSE149487_PLUMAGE_PREFLIGHT_CONFIG_PATH = (
    "configs/route_a_v3_gse149487_external_evidence_roots_v1.json"
)
GSE149487_PLUMAGE_PREFLIGHT_SCRIPT_PATH = (
    "scripts/route_a_v3/preflight_gse149487_full_a1.py"
)
GSE149487_PLUMAGE_PREFLIGHT_TEST_PATH = (
    "tests/route_a_v3/test_preflight_gse149487_full_a1.py"
)
GSE149487_PLUMAGE_ASSET_MANIFEST_SHA256 = (
    "7105125e686f3bc6e99152b1cd86230aa6225d1a86cad1f8d7968aa99675a878"
)
GSE149487_PLUMAGE_HELPER_SHA256 = (
    "372d58a37de5d393bb9ef1a749ffc51c8835195f8f9d1f8c5f13bd718c2f336d"
)
GSE149487_PLUMAGE_QUALIFIER_SHA256 = (
    "15f162d9f687c740e592be405ea50ff6af30a2cd572970722ade287bf18ade1a"
)
GSE149487_PLUMAGE_TEST_SHA256 = (
    "fc8e2147d4aee480d7ff12b38b0b036ee8e65dec8be6ecf5d9a6b893f2a28fb9"
)
GSE149487_PLUMAGE_PREFLIGHT_CONFIG_SHA256 = (
    "d25a978c4603e180ca20534f3ba8b78c321e21c01fdf94c88b9ab65a0ed7ed8b"
)
GSE149487_PLUMAGE_PREFLIGHT_SCRIPT_SHA256 = (
    "a4178cf60803b0e93cd6fa6a1f39dd2837ce20f5b0e74bb0c4f4ef5e5c1a48bb"
)
GSE149487_PLUMAGE_PREFLIGHT_TEST_SHA256 = (
    "21173517b6c70000c7c704408573364467e1028d8a48de699d9d00274aa05c5c"
)
GSE149487_PLUMAGE_PREFLIGHT_ARTIFACT_ID = (
    "GSE149487_FULL_A1_STOP_BEFORE_DATA_PREFLIGHT_20260810T181439Z_aeecf0f"
)
GSE149487_PLUMAGE_PREFLIGHT_ARTIFACT_PATH = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
    "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
    f"{GSE149487_PLUMAGE_PREFLIGHT_ARTIFACT_ID}.json"
)
GSE149487_PLUMAGE_PREFLIGHT_ARTIFACT_SHA256 = (
    "c1d649c9470c7962dd3a0c1a0ebe3c771dbe7b066d5cc4ad91bcd45b19d9417d"
)
GSE149487_PLUMAGE_PREFLIGHT_BLOCKERS = [
    "CHECKPOINT_SPECIFIC_FOUNDATION_EXPOSURE_UNKNOWN_NOT_ASSERTED",
    "LICENSE_AND_REDISTRIBUTION_UNKNOWN_NOT_ASSERTED",
    "OUTCOME_BLIND_LONG_READ_MAPPING_PROVENANCE_UNKNOWN_NOT_ASSERTED",
    "PAPER_NATIVE_METHOD_NOT_REPRODUCED",
    "PAPER_NATIVE_METHOD_SOURCE_UNKNOWN_NOT_ASSERTED",
    "PAPER_NATIVE_MULTIPLE_TESTING_FAMILY_UNKNOWN_NOT_ASSERTED",
    "PREFROZEN_GROUP_POWER_OR_CI_GATE_FAILED",
    "PUBLISHED_RESULT_CROSSCHECK_UNKNOWN_NOT_ASSERTED",
    "RAW_KEY_UNCLASSIFIED_OUTCOME_BLIND_RECONCILIATION_NOT_ZERO",
    "UNADJUDICATED_OR_AMBIGUOUS_MAPPING_ROWS_PRESENT",
    "UNADJUDICATED_SEQUENCE_UNIVERSE_CLASSES_PRESENT",
]
GSE149487_PLUMAGE_ACTIVE_AUTHORITY_COMMIT = (
    "d328bf04c394d4960ac11058e079c063e09280af"
)
GSE149487_PLUMAGE_EXTERNAL_BLOCKERS = [
    "OUTCOME_BLIND_LONG_READ_MAPPING_PROVENANCE_UNKNOWN_NOT_ASSERTED",
    "PAPER_NATIVE_METHOD_SOURCE_UNKNOWN_NOT_ASSERTED",
    "PAPER_NATIVE_MULTIPLE_TESTING_FAMILY_UNKNOWN_NOT_ASSERTED",
    "PUBLISHED_RESULT_CROSSCHECK_UNKNOWN_NOT_ASSERTED",
    "LICENSE_AND_REDISTRIBUTION_UNKNOWN_NOT_ASSERTED",
    "CHECKPOINT_SPECIFIC_FOUNDATION_EXPOSURE_UNKNOWN_NOT_ASSERTED",
]
GSE149487_PLUMAGE_CURRENT_GATE_CONTRACT = {
    "qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
    "qualified": False,
    "training_allowed": False,
    "model_selection_allowed": False,
    "ordinary_study_contribution": 0,
    "a1_study_contribution": 0,
    "true_a2_study_contribution": 0,
    "canonical_record_count": 0,
    "next_phase_authorized": False,
}
GSE149487_PLUMAGE_QUALIFICATION_GATES = [
    "AUTHORITY_AND_CODE_TRUST_ROOTS",
    "EXACT_21_ASSET_MANIFEST_AND_PAYLOAD_INTEGRITY",
    "EXACT_18_TABLE_CONTEXT_ASSAY_REPLICATE_GRID",
    "WITHIN_CONTEXT_KEY_SET_ALIGNMENT_AND_MISSING_NOT_ZERO",
    "OUTCOME_BLIND_STRICT_SOURCE_CANDIDATE_MAPPING",
    "PAPER_NATIVE_TRANSFORM_TEST_AND_MULTIPLE_TESTING_REPRODUCTION",
    "THREE_BIOLOGICAL_REPLICATES_AND_ROUTE_A_SE",
    "CANONICAL_V3_SCHEMA_AND_HASH_LINEAGE",
    "LICENSE_AND_REDISTRIBUTION_AUDIT",
    "CHECKPOINT_SPECIFIC_FOUNDATION_EXPOSURE_AUDIT",
    "GROUP_AND_SEQUENCE_CLUSTER_LEAKAGE_AUDIT",
    "PREFROZEN_GROUP_POWER_SIMULATION",
]
GSE149487_PLUMAGE_NONBINDING_CORE_SHA256 = (
    "fa2f69f518f82ba815c1013655789c5dfd80235ceec7e809510bee93855e0aea"
)
GSE149487_PLUMAGE_PREFLIGHT_BINDING_SCHEME = (
    "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1"
)

REGISTRY_PATHS = {
    "task": "docs/execution/route_a_v3_task_registry.yaml",
    "data": "docs/execution/route_a_v3_data_role_registry.yaml",
    "baseline": "docs/execution/route_a_v3_baseline_registry.yaml",
    "split": "docs/execution/route_a_v3_split_registry.yaml",
    "matrix": "docs/execution/route_a_v3_task_split_matrix.yaml",
    "claim": "docs/execution/route_a_v3_claim_evidence_matrix.yaml",
}

REGISTRY_TYPES = {
    "task": "TASK_REGISTRY",
    "data": "DATA_ROLE_REGISTRY",
    "baseline": "BASELINE_REGISTRY",
    "split": "SPLIT_REGISTRY",
    "matrix": "TASK_SPLIT_MATRIX",
    "claim": "CLAIM_EVIDENCE_MATRIX",
}

SCHEMA_FILES = (
    "canonical_intervention_record.schema.json",
    "compute_ledger.schema.json",
    "gate_record.schema.json",
    "measured_candidate_pool.schema.json",
    "prediction_record.schema.json",
    "run_manifest.schema.json",
)
SCHEMA_DIR = "schemas/route_a_v3"
SCHEMA_MANIFEST = f"{SCHEMA_DIR}/SCHEMA_MANIFEST.json"
SCHEMA_SUMS = f"{SCHEMA_DIR}/SCHEMA_SHA256SUMS"

EXPECTED_PHASE_IDS = tuple(f"A{i}" for i in range(11))
EXPECTED_PHASE_DEPENDENCIES = {
    "A0": (),
    "A1": ("A0",),
    "A2": ("A1",),
    "A3": ("A2",),
    "A4": ("A3",),
    "A5": ("A4",),
    "A6": ("A0",),
    "A7": ("A5", "A6"),
    "A8": ("A7",),
    "A9": ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"),
    "A10": ("A9",),
}
EXPECTED_TASK_IDS = (
    "T5_SOURCE_RELATIVE_EFFECT",
    "T5_SELECTIVE_EFFECT",
    "T5_MEASURED_NEIGHBORHOOD_OPTIMIZATION",
    "T5_FIXED_BUDGET_MULTI_STEP_OPTIMIZATION",
    "FLOW_BASE_LEGAL_CTMC",
    "EXACT_GUIDANCE_TOY_GRAPH",
    "EXACT_GUIDANCE_MATCHED_COMPUTE",
    "TRANSFER_3UTR",
    "TRANSFER_CDS",
    "SEALED_EXTERNAL_ADJUDICATION",
)
EXPECTED_SPLIT_IDS = tuple(f"S{i}" for i in range(1, 10))
SEALED_DATASET_ID = "GSE246381"
SEALED_SPLIT_ID = "S6"
SEALED_TASK_ID = "SEALED_EXTERNAL_ADJUDICATION"
SEALED_A9_REPLACEMENT_PRECONDITIONS = (
    "ALL_ORDINARY_GATES_PASS",
    "SEALED_EVALUATOR_NO_STUB",
    "FULL_CONSUMED_ASSET_HASH_FREEZE",
    "FULL_RUNTIME_SOURCE_ENVIRONMENT_HASH_FREEZE",
    "TRANSACTIONAL_AGGREGATE_OUTPUT_BEFORE_COMPLETION",
    "INDEPENDENT_A9_READINESS_REVIEW_PASS",
    "SEPARATE_EXPLICIT_USER_AUTHORIZATION_FOR_A10",
)
TOY_TASK_ID = "EXACT_GUIDANCE_TOY_GRAPH"
TOY_SPLIT_ID = "S9"

EVIDENCE_STATUSES = {
    "NOT_RUN",
    "IN_PROGRESS",
    "PASS",
    "FAIL_CURRENT_PROTOCOL",
    "FAIL_REPAIRABLE",
    "BLOCKED_PENDING_PUBLIC_EVIDENCE",
    "TERMINATED_SAFELY_WITH_EVIDENCE",
}
CLAIM_STATUSES = {
    "NOT_ESTABLISHED",
    "ESTABLISHED",
    "INVALIDATED_CURRENT_FORMULATION",
    "PROHIBITED",
}

CPU_COMPUTE_CLASSES = frozenset(
    {
        "CPU_AUTHORITY",
        "CPU_DATA",
        "CPU_STATISTICS",
        "CPU_HASH_GIT",
        "CPU_SMALL_GRAPH_EXACT",
        "CPU_NUMERICAL_CHECKER",
        "CPU_UNIT_TEST",
    }
)
GPU_TRAIN_COMPUTE_CLASSES = frozenset(
    {
        "GPU_NEURAL_CRITIC_TRAIN",
        "GPU_BASE_FLOW_TRAIN",
        "GPU_GUIDANCE_VALUE_TRAIN",
        "GPU_FOUNDATION_FINETUNE",
    }
)
GPU_VALIDATION_COMPUTE_CLASSES = frozenset({"GPU_VALIDATION"})
GPU_COMPUTE_CLASSES = GPU_TRAIN_COMPUTE_CLASSES | GPU_VALIDATION_COMPUTE_CLASSES
RUN_COMPUTE_CLASSES = CPU_COMPUTE_CLASSES | GPU_COMPUTE_CLASSES
GPU_PRESTART_RUN_STATUSES = frozenset({"NOT_RUN", "QUEUED"})
GPU_FAILURE_RUN_STATUSES = frozenset(
    {
        "FAIL_CLOSED",
        "FAIL_CURRENT_PROTOCOL",
        "FAIL_REPAIRABLE",
        "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "TERMINATED",
        "TERMINATED_SAFELY_WITH_EVIDENCE",
    }
)
GPU_LIFECYCLE_RUN_STATUSES = GPU_PRESTART_RUN_STATUSES | GPU_FAILURE_RUN_STATUSES | {"IN_PROGRESS", "COMPLETED"}

# Hard-coded historical bindings prevent a modified supersession document from
# silently blessing modified predecessor bytes.
EXPECTED_PREDECESSOR_BINDINGS = {
    "LOCAL_PRE_V3_CONTRACT": {
        "path": "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna 最新合同-v2.md",
        "sha256": "9c79edd819e45551974bcfeb14a400dd504c55c0a7c869e456e638daf49f1c1e",
    },
    "REMOTE_XEDITFLOW_V1_1_CONTRACT": {
        "path": "docs/contracts/mrna_xeditflow_goal_v1_1.md",
        "sha256": "fc9c1c882efbaa4c1e86f4da2e1be64e219755fb9c5941da4b4309793d3d8c2f",
        "config_path": "configs/mrna_xeditflow_contract_v1_1.yaml",
        "config_sha256": "b3be70e765fb8285996487815ee6a4494ca4cc7fb503dae2901b40a0382d83cf",
        "claim_matrix_path": "docs/execution/xeditflow_claim_matrix.yaml",
        "claim_matrix_sha256": "6358c6caaeed58b44cf7c2f72a0038d299622e951fb65f4b9f8c516e1ad5b4b2",
    },
    "LOCAL_PRE_V3_READINESS_AUDIT": {
        "repository_copy_path": SCIENTIFIC_M0_HISTORY_PATH,
        "sha256": SCIENTIFIC_M0_HISTORY_SHA256,
    },
    "REPOSITORY_V3_1_DERIVED_COPY": {
        "path": "docs/contracts/utr_editflow_goal_v3_1.md",
        "sha256": "a7fda79fd6fea4d3020794e69cb966eb719ab8388406acc70b632f90d12a9cee",
    },
    "DECLARED_EXTERNAL_V3_1_AUTHORITY": {
        "path": "external_authority_declared_by_predecessor",
        "sha256": "ecc6c635f112575db2f14309c869a378fc31df8fb76c01dda0b54b832b4f8946",
    },
}

EXPECTED_HISTORICAL_GATE_BINDINGS = {
    "M0_GOVERNANCE": ("reports/migration/M0_READONLY_AUDIT.md", "3f90fb6970d2ccc1e3933b6ef97b746a43c80aca85c9b1ac7a663d6242b31635"),
    "M0_SCIENTIFIC_ORIGINAL": (SCIENTIFIC_M0_HISTORY_PATH, SCIENTIFIC_M0_HISTORY_SHA256),
    "O0": ("reports/migration/O0X_CLOSED_MEASURED_OPTIMIZATION_GATE.md", "ebbd0d1ae55fe302ebf673f5dfbea2461bb2e72304c958c9509f840ee0736bf5"),
    "G1": ("reports/migration/G1X_REAL_MRNA_GUIDANCE_GATE.md", "f183c08990b6c12752ca0be5c20de371358c1eba10b40c2c720519385dec840b"),
    "E0": ("reports/migration/E0X_PREREG_INTERNAL_GATE.md", "7ff4639764371b879b65b7fc100e03fe5e3da1216d584f8609e70fe78a6beedb"),
    "FINAL_MIGRATION": ("reports/migration/FINAL_MIGRATION_REPORT.md", "a987d8c292c3700754f77052cdfe7315cf656ff2ca32818b5267a6e9fff84b92"),
}

EXPECTED_DECISION_IDS = tuple(f"V3-DEC-{index:03d}" for index in range(1, 20))
EXPECTED_DECISION_DIMENSIONS = {
    "V3-DEC-001": "strategic_target",
    "V3-DEC-002": "evidence_and_claim_separation",
    "V3-DEC-003": "data_and_claim_scope",
    "V3-DEC-004": "edit_budget",
    "V3-DEC-005": "ordinary_study_qualification",
    "V3-DEC-006": "effect_uplift_metric",
    "V3-DEC-007": "secondary_region",
    "V3-DEC-008": "innovation_boundary",
    "V3-DEC-009": "a0_authority_base",
    "V3-DEC-010": "pre_v3_routea_run",
    "V3-DEC-011": "gpu_snapshot",
    "V3-DEC-012": "sealed_hard_disable",
    "V3-DEC-013": "commit_binding",
    "V3-DEC-014": "historical_m0_scientific_failure",
    "V3-DEC-015": "sealed_execution_freeze_hash_scope",
    "V3-DEC-016": "sealed_a0_phase_boundary",
    "V3-DEC-017": "gse145046_true_a2_role_and_a2_recovery",
    "V3-DEC-018": "gse200302_official_srr_role_authority_and_raw_replay_boundary",
    "V3-DEC-019": "a1_measurement_uncertainty_split_and_claim_boundary",
}

# Canonical per-entry digests make the accepted prefix genuinely append-only.
# A future DEC-020 requires an explicit validator update; rewriting any accepted
# DEC-001..019 entry while merely refreshing the registry manifest is rejected.
EXPECTED_DECISION_ENTRY_SHA256 = {
    "V3-DEC-001": "e00b87c7cd529b452ef6db96f982adfd419c3cf289f02d8795abaf09dae966f3",
    "V3-DEC-002": "bc5c0e6d1a68bf45e16529470b9c173b1fbbccab3789cc5a27e3033ee70590b1",
    "V3-DEC-003": "a3e45e7d4c382d63a092ccf3fff5cc23aa6d938be3d9a99a22663a5bd04e3fec",
    "V3-DEC-004": "b53d20748e285180b54a98a9710610d8798bbd52caae8966565c76e9367d76d4",
    "V3-DEC-005": "725509630b39c8f03c5927b0c8516553d6fdf815a25fa109da35836c364c0e2b",
    "V3-DEC-006": "1eefe8f30ae2bcbd62e7962ee56c41360b4682f1ad83b4c1a8af0213478ad7a3",
    "V3-DEC-007": "cf7ce474d29d9c6634e406a0cba48dec07ce08376f2161845c2b689726a00bbb",
    "V3-DEC-008": "01e115b916046d69090a328b442f1964410bf3159d0cc5980afe87bffaa15066",
    "V3-DEC-009": "8ab88b659376600fe361b77b2c80a50125a1fae882fac74b423f5cbbff7ee8f7",
    "V3-DEC-010": "e38ce6235048acef73a5d9826739b8665d482a7c3dc5541f8380160a5111dded",
    "V3-DEC-011": "853210b83267563c4d1b01fd0ceb2b6f1c6cba5e9bf606847748aa1085e7fb95",
    "V3-DEC-012": "2ccb85bd983353fd98874aab1faa1d677f352a52d0ac487ff02679a86b6d61ad",
    "V3-DEC-013": "02f22c2f09a8de22b8f9a4419b5b8fe877003db50b24d93af23317c92447e255",
    "V3-DEC-014": "92d27a394d258d8e189e378ca25e9ad7adc3f2a396bf2649d1981d1c20062e85",
    "V3-DEC-015": "2d45d836c04b39365df6528ad1972826af8a6595fcb0cc3e9568ecd9adcf56c2",
    "V3-DEC-016": "b980d623ca9de3439ef050fb1f6b0dd59ceeacf8c66b3b94bb7aade211380dca",
    "V3-DEC-017": "d3f4799501b4d0abb63c91105c4f46c5e3246bea9da708c813a1de7c30f3b11a",
    "V3-DEC-018": "c49c04371b02e4f66a42fa33670d8b164d40ad012c0db50b5f16fbaee0b539e4",
    "V3-DEC-019": "93b32f2dbbc261d87604dfff7ea7eca1f57f6d5e7124d7cbec130a87a5466260",
}

EXPECTED_REGISTRY_MANIFEST_PATH_ROLES = (
    (GOAL_PATH, "ACTIVE_AMENDED_CONTRACT"),
    (CONFIG_PATH, "EXECUTABLE_CONTRACT"),
    (DEC019_AMENDMENT_PATH, "DEC019_APPEND_ONLY_AUTHORITY_AMENDMENT"),
    (A1_QUALIFICATION_CONFIG_PATH, "A1_QUALIFICATION_ROOT_PROTOCOL"),
    (SUPERSESSION_PATH, "SUPERSESSION_LINEAGE"),
    (SCIENTIFIC_M0_HISTORY_PATH, "HISTORICAL_M0_SCIENTIFIC_FAILURE_EXACT_COPY"),
    (REGISTRY_PATHS["baseline"], "BASELINE_REGISTRY"),
    (REGISTRY_PATHS["claim"], "CLAIM_EVIDENCE_MATRIX"),
    (REGISTRY_PATHS["data"], "DATA_ROLE_REGISTRY"),
    (DECISION_LOG_PATH, "DECISION_AND_AMENDMENT_LOG"),
    (A1_INTERIM_PATH, "A1_ACTIVE_INTERIM_RECORD"),
    (REGISTRY_PATHS["split"], "SPLIT_REGISTRY"),
    (REGISTRY_PATHS["task"], "TASK_AND_PHASE_REGISTRY"),
    (REGISTRY_PATHS["matrix"], "TASK_SPLIT_MATRIX"),
    (SCHEMA_MANIFEST, "PUBLIC_SCHEMA_MANIFEST"),
    (SCHEMA_SUMS, "PUBLIC_SCHEMA_CHECKSUMS"),
    (SEALED_GUARD_PATH, "SEALED_HARD_DISABLE_GUARD"),
    (SEALED_RUNNER_PATH, "SEALED_RUNNER_GUARD_INTEGRATION"),
    (
        GSE149487_PLUMAGE_ASSET_MANIFEST_PATH,
        "GSE149487_PLUMAGE_21_ASSET_MANIFEST",
    ),
    (GSE149487_PLUMAGE_HELPER_PATH, "GSE149487_PLUMAGE_V4_HELPER"),
    (GSE149487_PLUMAGE_QUALIFIER_PATH, "GSE149487_PLUMAGE_FULL_A1_QUALIFIER"),
    (GSE149487_PLUMAGE_TEST_PATH, "GSE149487_PLUMAGE_FULL_A1_FOCUSED_TEST"),
    (
        GSE149487_PLUMAGE_PREFLIGHT_CONFIG_PATH,
        "GSE149487_PLUMAGE_EXTERNAL_EVIDENCE_ROOTS",
    ),
    (
        GSE149487_PLUMAGE_PREFLIGHT_SCRIPT_PATH,
        "GSE149487_PLUMAGE_STOP_BEFORE_DATA_PREFLIGHT",
    ),
    (
        GSE149487_PLUMAGE_PREFLIGHT_TEST_PATH,
        "GSE149487_PLUMAGE_STOP_BEFORE_DATA_PREFLIGHT_TEST",
    ),
    (GSE200302_ROLE_BUILDER_PATH, "GSE200302_OFFICIAL_ROLE_AUTHORITY_BUILDER"),
    (GSE200302_ROLE_TEST_PATH, "GSE200302_OFFICIAL_ROLE_AUTHORITY_FOCUSED_TEST"),
    (
        GSE200304_PUBLISHED_ENDPOINT_CONFIG_PATH,
        "GSE200304_PUBLISHED_ENDPOINT_A1_PROTOCOL",
    ),
    (
        GSE200304_PUBLISHED_ENDPOINT_SCRIPT_PATH,
        "GSE200304_PUBLISHED_ENDPOINT_A1_QUALIFIER",
    ),
    (
        GSE200304_PUBLISHED_ENDPOINT_TEST_PATH,
        "GSE200304_PUBLISHED_ENDPOINT_A1_FOCUSED_TEST",
    ),
    (
        GSE114002_ENDPOINT_GEOMETRY_CONFIG_PATH,
        "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_PROTOCOL",
    ),
    (
        GSE114002_ENDPOINT_GEOMETRY_SCRIPT_PATH,
        "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_AUDITOR",
    ),
    (
        GSE114002_ENDPOINT_GEOMETRY_TEST_PATH,
        "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_FOCUSED_TEST",
    ),
    (
        GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_PATH,
        "GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_AGGREGATE_EVIDENCE",
    ),
    (
        GSE114002_DEC019_SUCCESSOR_SCRIPT_PATH,
        "GSE114002_DEC019_TRUE_A2_SUCCESSOR_ADJUDICATOR",
    ),
    (
        GSE114002_DEC019_SUCCESSOR_TEST_PATH,
        "GSE114002_DEC019_TRUE_A2_SUCCESSOR_FOCUSED_TEST",
    ),
    (
        GSE200304_DEC019_SUCCESSOR_SCRIPT_PATH,
        "GSE200304_DEC019_REPORTED_ENDPOINT_A1_SUCCESSOR_ADJUDICATOR",
    ),
    (
        GSE200304_DEC019_SUCCESSOR_TEST_PATH,
        "GSE200304_DEC019_REPORTED_ENDPOINT_A1_SUCCESSOR_FOCUSED_TEST",
    ),
    (
        GSE200304_DEC019_V3_SCRIPT_PATH,
        "GSE200304_DEC019_REPORTED_ENDPOINT_A1_V3_ADJUDICATOR",
    ),
    (
        GSE200304_DEC019_V3_TEST_PATH,
        "GSE200304_DEC019_REPORTED_ENDPOINT_A1_V3_FOCUSED_TEST",
    ),
    (
        GSE200304_DEC019_LINEAGE_CONFIG_PATH,
        "GSE200304_DEC019_CANONICAL_ROW_LINEAGE_GATE_PROTOCOL",
    ),
    (
        GSE200304_DEC019_LINEAGE_SCRIPT_PATH,
        "GSE200304_DEC019_CANONICAL_ROW_LINEAGE_GATE_PRODUCER",
    ),
    (
        GSE200304_DEC019_LINEAGE_TEST_PATH,
        "GSE200304_DEC019_CANONICAL_ROW_LINEAGE_GATE_FOCUSED_TEST",
    ),
    (
        GSE200304_DEC019_NEGATIVE_CONFIG_PATH,
        "GSE200304_DEC019_NEGATIVE_GATE_PACK_PROTOCOL",
    ),
    (
        GSE200304_DEC019_NEGATIVE_SCRIPT_PATH,
        "GSE200304_DEC019_NEGATIVE_GATE_PACK_PRODUCER",
    ),
    (
        GSE200304_DEC019_NEGATIVE_TEST_PATH,
        "GSE200304_DEC019_NEGATIVE_GATE_PACK_FOCUSED_TEST",
    ),
    (INTEGRITY_GUARD_TEST_PATH, "A0_AUTHORITY_INTEGRITY_GUARD_TEST"),
    (VALIDATOR_PATH, "A0_STATIC_AND_SEMANTIC_VALIDATOR"),
)
MANDATORY_REGISTRY_MANIFEST_PATHS = frozenset(
    path for path, _role in EXPECTED_REGISTRY_MANIFEST_PATH_ROLES
)

PUBLIC_PREFIXES = {"configs", "docs", "reports", "schemas", "scripts", "tests"}
FORBIDDEN_PATH_PARTS = {"restricted", "restricted_store", "sealed_store", "access_log"}
CONFLICT_MARKERS = ("<" * 7, "=" * 7, ">" * 7)


@dataclass(frozen=True, order=True)
class Issue:
    """One deterministic validation failure."""

    code: str
    path: str
    detail: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _issue(issues: list[Issue], code: str, path: str, detail: str) -> None:
    issues.append(Issue(code=code, path=path, detail=detail))


def _safe_repo_path(repo_root: Path, relative: str, *, must_exist: bool = True) -> Path:
    """Resolve a public repository-relative path without following unsafe pointers."""

    raw = PurePosixPath(relative)
    if raw.is_absolute() or ".." in raw.parts or not raw.parts:
        raise ValueError(f"not a repository-relative public path: {relative!r}")
    if raw.parts[0] not in PUBLIC_PREFIXES:
        raise ValueError(f"path prefix is outside the public validation allowlist: {relative!r}")
    lowered = {part.lower() for part in raw.parts}
    if lowered & FORBIDDEN_PATH_PARTS:
        raise ValueError(f"restricted/sealed state path is not readable by A0 validator: {relative!r}")

    root = repo_root.resolve()
    candidate = repo_root.joinpath(*raw.parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {relative!r}") from exc
    if candidate.is_symlink():
        raise ValueError(f"symlink inputs are not followed by the A0 validator: {relative!r}")
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _read_text(repo_root: Path, relative: str) -> str:
    return _safe_repo_path(repo_root, relative).read_text(encoding="utf-8")


def _read_bytes(repo_root: Path, relative: str) -> bytes:
    return _safe_repo_path(repo_root, relative).read_bytes()


def _load_yaml(repo_root: Path, relative: str) -> Mapping[str, Any]:
    loaded = yaml.safe_load(_read_text(repo_root, relative))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"expected YAML mapping: {relative}")
    return loaded


def _load_json(repo_root: Path, relative: str) -> Mapping[str, Any]:
    loaded = json.loads(_read_text(repo_root, relative))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"expected JSON object: {relative}")
    return loaded


def required_bundle_paths() -> tuple[str, ...]:
    return (
        GOAL_PATH,
        CONFIG_PATH,
        A1_QUALIFICATION_CONFIG_PATH,
        SUPERSESSION_PATH,
        DEC019_AMENDMENT_PATH,
        DECISION_LOG_PATH,
        REGISTRY_MANIFEST_PATH,
        SCIENTIFIC_M0_HISTORY_PATH,
        SEALED_GUARD_PATH,
        SEALED_RUNNER_PATH,
        A1_INTERIM_PATH,
        GSE149487_PLUMAGE_PROTOCOL_PATH,
        GSE149487_PLUMAGE_ASSET_MANIFEST_PATH,
        GSE149487_PLUMAGE_HELPER_PATH,
        GSE149487_PLUMAGE_QUALIFIER_PATH,
        GSE149487_PLUMAGE_TEST_PATH,
        GSE149487_PLUMAGE_PREFLIGHT_CONFIG_PATH,
        GSE149487_PLUMAGE_PREFLIGHT_SCRIPT_PATH,
        GSE149487_PLUMAGE_PREFLIGHT_TEST_PATH,
        GSE200302_ROLE_CONFIG_PATH,
        GSE200302_ROLE_BUILDER_PATH,
        GSE200302_ROLE_TEST_PATH,
        GSE200304_PUBLISHED_ENDPOINT_CONFIG_PATH,
        GSE200304_PUBLISHED_ENDPOINT_SCRIPT_PATH,
        GSE200304_PUBLISHED_ENDPOINT_TEST_PATH,
        GSE114002_ENDPOINT_GEOMETRY_CONFIG_PATH,
        GSE114002_ENDPOINT_GEOMETRY_SCRIPT_PATH,
        GSE114002_ENDPOINT_GEOMETRY_TEST_PATH,
        GSE114002_DEC019_SUCCESSOR_CONFIG_PATH,
        GSE114002_DEC019_SUCCESSOR_SCRIPT_PATH,
        GSE114002_DEC019_SUCCESSOR_TEST_PATH,
        GSE200304_DEC019_SUCCESSOR_CONFIG_PATH,
        GSE200304_DEC019_SUCCESSOR_SCRIPT_PATH,
        GSE200304_DEC019_SUCCESSOR_TEST_PATH,
        GSE200304_DEC019_V3_CONFIG_PATH,
        *GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256,
        INTEGRITY_GUARD_TEST_PATH,
        *REGISTRY_PATHS.values(),
        *(f"{SCHEMA_DIR}/{name}" for name in SCHEMA_FILES),
        SCHEMA_MANIFEST,
        SCHEMA_SUMS,
    )


def validate_required_files(repo_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    for relative in required_bundle_paths():
        try:
            _safe_repo_path(repo_root, relative)
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "MISSING_OR_UNSAFE_FILE", relative, str(exc))
    return issues


def _metadata_ok(document: Mapping[str, Any], path: str, issues: list[Issue], *, registry_type: str | None = None) -> None:
    if document.get("contract_id") != CONTRACT_ID:
        _issue(issues, "CONTRACT_ID_MISMATCH", path, f"expected {CONTRACT_ID!r}, got {document.get('contract_id')!r}")
    if str(document.get("version")) != VERSION:
        _issue(issues, "VERSION_MISMATCH", path, f"expected {VERSION!r}, got {document.get('version')!r}")
    if str(document.get("schema_version")) != VERSION:
        _issue(issues, "SCHEMA_VERSION_MISMATCH", path, f"expected {VERSION!r}, got {document.get('schema_version')!r}")
    if registry_type is not None and document.get("registry_type") != registry_type:
        _issue(issues, "REGISTRY_TYPE_MISMATCH", path, f"expected {registry_type!r}, got {document.get('registry_type')!r}")


def _entry_ids(entries: Any, id_key: str, path: str, issues: list[Issue]) -> list[str]:
    if isinstance(entries, Mapping):
        ids = [str(key) for key in entries]
    elif isinstance(entries, list):
        ids = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or not isinstance(entry.get(id_key), str):
                _issue(issues, "INVALID_REGISTRY_ENTRY", path, f"entry {index} lacks string {id_key}")
                continue
            ids.append(entry[id_key])
    else:
        _issue(issues, "INVALID_REGISTRY_ENTRIES", path, "expected a list or mapping")
        return []
    if len(ids) != len(set(ids)):
        _issue(issues, "DUPLICATE_ID", path, f"duplicate {id_key} values")
    return ids


def _check_expected_closure(
    document: Mapping[str, Any],
    *,
    path: str,
    expected_key: str,
    entries_key: str,
    id_key: str,
    issues: list[Issue],
    fixed_expected: Sequence[str] | None = None,
) -> tuple[set[str], list[Mapping[str, Any]]]:
    expected_raw = document.get(expected_key)
    if not isinstance(expected_raw, list) or not all(isinstance(item, str) for item in expected_raw):
        _issue(issues, "INVALID_EXPECTED_ID_SET", path, f"{expected_key} must be a list of strings")
        expected: list[str] = []
    else:
        expected = list(expected_raw)
        if len(expected) != len(set(expected)):
            _issue(issues, "DUPLICATE_EXPECTED_ID", path, f"{expected_key} contains duplicates")

    entries_raw = document.get(entries_key)
    actual = _entry_ids(entries_raw, id_key, path, issues)
    if set(actual) != set(expected) or len(actual) != len(expected):
        _issue(issues, "EXPECTED_ID_CLOSURE", path, f"{entries_key} IDs do not equal {expected_key}")
    if fixed_expected is not None and (set(expected) != set(fixed_expected) or len(expected) != len(fixed_expected)):
        _issue(issues, "FROZEN_ID_SET_MISMATCH", path, f"{expected_key} differs from frozen V3 set")

    if isinstance(entries_raw, list):
        entries = [entry for entry in entries_raw if isinstance(entry, Mapping)]
    elif isinstance(entries_raw, Mapping):
        entries = []
        for key, value in entries_raw.items():
            if isinstance(value, Mapping):
                materialized = dict(value)
                materialized.setdefault(id_key, str(key))
                entries.append(materialized)
    else:
        entries = []
    return set(actual), entries


def _phase_dependency_map(entries: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        phase_id = entry.get("phase_id")
        depends = entry.get("depends_on")
        if isinstance(phase_id, str) and isinstance(depends, list) and all(isinstance(item, str) for item in depends):
            result[phase_id] = tuple(depends)
    return result


def validate_phase_dependencies(
    config_phase_entries: Sequence[Mapping[str, Any]],
    registry_phase_entries: Sequence[Mapping[str, Any]],
) -> list[Issue]:
    issues: list[Issue] = []
    config_map = _phase_dependency_map(config_phase_entries)
    registry_map = _phase_dependency_map(registry_phase_entries)
    frozen = {phase: tuple(deps) for phase, deps in EXPECTED_PHASE_DEPENDENCIES.items()}
    for label, mapping in ((CONFIG_PATH, config_map), (REGISTRY_PATHS["task"], registry_map)):
        if set(mapping) != set(EXPECTED_PHASE_IDS):
            _issue(issues, "PHASE_ID_CLOSURE", label, "phase IDs must be exactly A0 through A10")
            continue
        for phase_id, expected_deps in frozen.items():
            actual = mapping.get(phase_id, ())
            if set(actual) != set(expected_deps) or len(actual) != len(expected_deps):
                _issue(issues, "PHASE_DEPENDENCY_MISMATCH", label, f"{phase_id} depends_on {list(actual)!r}; expected {list(expected_deps)!r}")

        # Independent cycle/future-phase guard, even if the frozen map changes later.
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return False
            if node in visited:
                return True
            visiting.add(node)
            for dep in mapping.get(node, ()):
                if dep not in mapping or not visit(dep):
                    return False
            visiting.remove(node)
            visited.add(node)
            return True

        if not all(visit(phase) for phase in mapping):
            _issue(issues, "PHASE_DEPENDENCY_CYCLE_OR_UNKNOWN", label, "phase dependency graph is cyclic or references an unknown phase")

    if config_map and registry_map and config_map != registry_map:
        _issue(issues, "PHASE_DEPENDENCY_CROSS_FILE_MISMATCH", CONFIG_PATH, "config phase_plan and task registry phase_tasks differ")
    return issues


def _validate_authority_refs(registries: Mapping[str, Mapping[str, Any]], issues: list[Issue]) -> None:
    for name, document in registries.items():
        path = REGISTRY_PATHS[name]
        ref = document.get("authority_ref")
        if not isinstance(ref, Mapping):
            _issue(issues, "MISSING_AUTHORITY_REF", path, "authority_ref mapping is required")
            continue
        if ref.get("config_path") != CONFIG_PATH:
            _issue(issues, "AUTHORITY_CONFIG_PATH", path, f"authority_ref.config_path must be {CONFIG_PATH}")
        goal_path = ref.get("goal_path")
        if goal_path is not None and goal_path != GOAL_PATH:
            _issue(issues, "AUTHORITY_GOAL_PATH", path, f"authority_ref.goal_path must be {GOAL_PATH}")
        goal_sha = ref.get("goal_sha256")
        if goal_sha is not None and goal_sha != SOURCE_CONTRACT_SHA256:
            _issue(issues, "AUTHORITY_GOAL_HASH", path, "authority_ref.goal_sha256 is not the frozen V3 hash")


def validate_contract_authority(
    repo_root: Path,
    config: Mapping[str, Any],
    supersession: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    issues: list[Issue] = []
    _metadata_ok(config, CONFIG_PATH, issues)
    if config.get("status") != CONFIG_STATUS:
        _issue(issues, "CONFIG_NOT_ACTIVE_AUTHORITY", CONFIG_PATH, f"status must be {CONFIG_STATUS}")

    authority = config.get("authority")
    source_goal = authority.get("source_goal") if isinstance(authority, Mapping) else None
    if not isinstance(source_goal, Mapping):
        _issue(issues, "MISSING_SOURCE_GOAL_BINDING", CONFIG_PATH, "authority.source_goal mapping is required")
    else:
        expected = {
            "local_path": SOURCE_CONTRACT_PATH,
            "sha256": SOURCE_CONTRACT_SHA256,
            "repository_path": GOAL_PATH,
        }
        for key, value in expected.items():
            if source_goal.get(key) != value:
                _issue(issues, "SOURCE_GOAL_BINDING_MISMATCH", CONFIG_PATH, f"authority.source_goal.{key} must be {value!r}")
    if not isinstance(authority, Mapping) or authority.get("active_contract_count_required") != 1 or authority.get("authority_uniqueness_required") is not True:
        _issue(issues, "AUTHORITY_UNIQUENESS_POLICY", CONFIG_PATH, "exactly one active contract must be required")
    if not isinstance(authority, Mapping) or authority.get("active_amendment_decision_ids") != ACTIVE_AMENDMENT_DECISION_IDS:
        _issue(
            issues,
            "AUTHORITY_ACTIVE_AMENDMENT_LINEAGE",
            CONFIG_PATH,
            "authority.active_amendment_decision_ids must be exactly [V3-DEC-017, V3-DEC-018, V3-DEC-019]",
        )
    elif not _json_type_strict_equal(
        authority.get("active_amendment_paths"),
        {"V3-DEC-019": DEC019_AMENDMENT_PATH},
    ):
        _issue(
            issues,
            "AUTHORITY_ACTIVE_AMENDMENT_PATHS",
            CONFIG_PATH,
            "authority.active_amendment_paths must bind only the DEC-019 append-only amendment path",
        )

    try:
        goal_hash = sha256_bytes(_read_bytes(repo_root, GOAL_PATH))
        if goal_hash != SOURCE_CONTRACT_SHA256:
            _issue(issues, "ACTIVE_CONTRACT_HASH_MISMATCH", GOAL_PATH, f"got {goal_hash}, expected {SOURCE_CONTRACT_SHA256}")
        goal_text = _read_text(repo_root, GOAL_PATH)
        if "mRNA-XEditFlow Route A V3" not in goal_text:
            _issue(issues, "ACTIVE_CONTRACT_TITLE_MISSING", GOAL_PATH, "frozen contract title was not found")
    except (FileNotFoundError, ValueError) as exc:
        _issue(issues, "ACTIVE_CONTRACT_UNREADABLE", GOAL_PATH, str(exc))

    if supersession.get("active_contract") != CONTRACT_ID:
        _issue(issues, "SUPERSESSION_ACTIVE_CONTRACT", SUPERSESSION_PATH, f"active_contract must be {CONTRACT_ID}")
    if supersession.get("active_contract_path") != GOAL_PATH:
        _issue(issues, "SUPERSESSION_ACTIVE_PATH", SUPERSESSION_PATH, f"active_contract_path must be {GOAL_PATH}")
    if supersession.get("active_contract_sha256") != SOURCE_CONTRACT_SHA256:
        _issue(issues, "SUPERSESSION_ACTIVE_HASH", SUPERSESSION_PATH, "active contract hash is not frozen V3 hash")
    if supersession.get("active_contract_amendment_decision_ids") != ACTIVE_AMENDMENT_DECISION_IDS:
        _issue(
            issues,
            "SUPERSESSION_ACTIVE_AMENDMENT_LINEAGE",
            SUPERSESSION_PATH,
            "active contract amendment lineage must be exactly [V3-DEC-017, V3-DEC-018, V3-DEC-019]",
        )
    if not _json_type_strict_equal(
        supersession.get("active_contract_amendment_paths"),
        {"V3-DEC-019": DEC019_AMENDMENT_PATH},
    ):
        _issue(
            issues,
            "SUPERSESSION_ACTIVE_AMENDMENT_PATHS",
            SUPERSESSION_PATH,
            "active contract amendment paths must bind only DEC-019",
        )
    new_authority = supersession.get("new_authority")
    if not isinstance(new_authority, Mapping):
        _issue(issues, "SUPERSESSION_NEW_AUTHORITY", SUPERSESSION_PATH, "new_authority mapping is required")
    else:
        expected_new = {
            "contract_id": CONTRACT_ID,
            "version": VERSION,
            "contract_path": GOAL_PATH,
            "contract_sha256": SOURCE_CONTRACT_SHA256,
            "config_path": CONFIG_PATH,
        }
        for key, value in expected_new.items():
            if str(new_authority.get(key)) != value:
                _issue(issues, "SUPERSESSION_NEW_AUTHORITY", SUPERSESSION_PATH, f"new_authority.{key} must be {value!r}")
        if new_authority.get("active_amendment_decision_ids") != ACTIVE_AMENDMENT_DECISION_IDS:
            _issue(
                issues,
                "SUPERSESSION_NEW_AUTHORITY",
                SUPERSESSION_PATH,
                "new_authority.active_amendment_decision_ids must be exactly [V3-DEC-017, V3-DEC-018, V3-DEC-019]",
            )
        if not _json_type_strict_equal(
            new_authority.get("active_amendment_paths"),
            {"V3-DEC-019": DEC019_AMENDMENT_PATH},
        ):
            _issue(
                issues,
                "SUPERSESSION_NEW_AUTHORITY",
                SUPERSESSION_PATH,
                "new_authority.active_amendment_paths must bind only DEC-019",
            )
        actual_config_sha256 = sha256_bytes(_read_bytes(repo_root, CONFIG_PATH))
        if new_authority.get("config_sha256") != actual_config_sha256:
            _issue(
                issues,
                "SUPERSESSION_CONFIG_HASH",
                SUPERSESSION_PATH,
                f"new_authority.config_sha256 must bind current config bytes {actual_config_sha256}",
            )
        if new_authority.get("status") not in {CONFIG_STATUS, "ACTIVE_AUTHORITATIVE_CONTRACT_PENDING_A0_ACCEPTANCE"}:
            _issue(issues, "SUPERSESSION_NEW_AUTHORITY_STATUS", SUPERSESSION_PATH, "new authority status is neither active nor pending A0 acceptance")

    predecessors_raw = supersession.get("predecessors")
    predecessors = {
        item.get("record_id"): item
        for item in predecessors_raw
        if isinstance(predecessors_raw, list) and isinstance(item, Mapping) and isinstance(item.get("record_id"), str)
    } if isinstance(predecessors_raw, list) else {}
    if set(predecessors) != set(EXPECTED_PREDECESSOR_BINDINGS):
        _issue(issues, "PREDECESSOR_RECORD_CLOSURE", SUPERSESSION_PATH, "predecessor record IDs differ from the frozen set")
    for record_id, expected in EXPECTED_PREDECESSOR_BINDINGS.items():
        record = predecessors.get(record_id)
        if not isinstance(record, Mapping):
            continue
        for key, value in expected.items():
            if record.get(key) != value:
                _issue(issues, "PREDECESSOR_BINDING_MISMATCH", SUPERSESSION_PATH, f"{record_id}.{key} must be {value!r}")
        if "HISTORICAL" not in str(record.get("status", "")):
            _issue(issues, "PREDECESSOR_NOT_HISTORICAL", SUPERSESSION_PATH, f"{record_id} is not marked historical")

        # External declarations are metadata-only.  Only allowlisted, repository-
        # relative historical paths are byte-checked.
        for path_key, hash_key in (
            ("path", "sha256"),
            ("repository_copy_path", "sha256"),
            ("config_path", "config_sha256"),
            ("claim_matrix_path", "claim_matrix_sha256"),
        ):
            relative = record.get(path_key)
            frozen_hash = record.get(hash_key)
            if not isinstance(relative, str) or not isinstance(frozen_hash, str):
                continue
            if PurePosixPath(relative).is_absolute() or relative == "external_authority_declared_by_predecessor":
                continue
            try:
                actual = sha256_bytes(_read_bytes(repo_root, relative))
                if actual != frozen_hash:
                    _issue(issues, "HISTORICAL_BYTES_CHANGED", relative, f"got {actual}, expected {frozen_hash}")
            except (FileNotFoundError, ValueError) as exc:
                _issue(issues, "HISTORICAL_FILE_MISSING_OR_UNSAFE", relative, str(exc))

    gates_raw = supersession.get("historical_gate_records")
    gates = {
        item.get("gate_id"): item
        for item in gates_raw
        if isinstance(gates_raw, list) and isinstance(item, Mapping) and isinstance(item.get("gate_id"), str)
    } if isinstance(gates_raw, list) else {}
    if set(gates) != set(EXPECTED_HISTORICAL_GATE_BINDINGS):
        _issue(issues, "HISTORICAL_GATE_CLOSURE", SUPERSESSION_PATH, "historical gate IDs differ from frozen governance/scientific M0, O0, G1, E0 and final set")
    for gate_id, (relative, frozen_hash) in EXPECTED_HISTORICAL_GATE_BINDINGS.items():
        record = gates.get(gate_id)
        if not isinstance(record, Mapping):
            continue
        if record.get("path") != relative or record.get("sha256") != frozen_hash:
            _issue(issues, "HISTORICAL_GATE_BINDING_MISMATCH", SUPERSESSION_PATH, f"{gate_id} path/hash binding changed")
        if record.get("rerun_in_a0") is not False:
            _issue(issues, "HISTORICAL_GATE_RERUN_FORBIDDEN", SUPERSESSION_PATH, f"{gate_id}.rerun_in_a0 must be false")
        try:
            actual = sha256_bytes(_read_bytes(repo_root, relative))
            if actual != frozen_hash:
                _issue(issues, "HISTORICAL_BYTES_CHANGED", relative, f"got {actual}, expected {frozen_hash}")
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "HISTORICAL_FILE_MISSING_OR_UNSAFE", relative, str(exc))

    governance_m0 = gates.get("M0_GOVERNANCE")
    if not isinstance(governance_m0, Mapping) or governance_m0.get("repository_reported_status") != "PASS_AUTHORITY_AUDIT_ONLY":
        _issue(issues, "M0_GOVERNANCE_CONFLATED", SUPERSESSION_PATH, "M0_READONLY_AUDIT must remain a separate governance PASS_AUTHORITY_AUDIT_ONLY")
    scientific_m0 = gates.get("M0_SCIENTIFIC_ORIGINAL")
    if not isinstance(scientific_m0, Mapping):
        _issue(issues, "M0_SCIENTIFIC_RECORD_MISSING", SUPERSESSION_PATH, "original M0 scientific failure record is required")
    else:
        scientific_expected = {
            "status": "ORIGINAL_M0_EFFECT_GATE_FAIL",
            "macro_sign_accuracy": 0.510,
            "sign_accuracy_threshold": 0.60,
            "o0_valid": False,
            "g1_established": False,
            "sealed_evaluator_implemented": False,
            "rerun_in_a0": False,
        }
        for key, value in scientific_expected.items():
            if scientific_m0.get(key) != value:
                _issue(issues, "M0_SCIENTIFIC_BINDING", SUPERSESSION_PATH, f"M0_SCIENTIFIC_ORIGINAL.{key} must be {value!r}")
        sign = scientific_m0.get("macro_sign_accuracy")
        threshold = scientific_m0.get("sign_accuracy_threshold")
        if not isinstance(sign, (int, float)) or not isinstance(threshold, (int, float)) or not sign < threshold:
            _issue(issues, "M0_SCIENTIFIC_THRESHOLD", SUPERSESSION_PATH, "original macro sign accuracy must remain strictly below its 0.60 threshold")

    _validate_authority_refs(registries, issues)
    return issues


def validate_gse114002_public_authority_gap_audit(repo_root: Path) -> list[Issue]:
    """Freeze the aggregate-only public-authority gap audit without granting a gate."""

    issues: list[Issue] = []
    path = GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_PATH
    try:
        raw = _read_bytes(repo_root, path)
        audit = json.loads(raw.decode("utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_UNREADABLE", path, str(exc))
        return issues

    actual_sha256 = sha256_bytes(raw)
    if actual_sha256 != GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_SHA256:
        _issue(
            issues,
            "GSE114002_PUBLIC_GAP_AUDIT_CANONICAL_HASH",
            path,
            f"audit hash {actual_sha256} must remain {GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_SHA256}",
        )
    expected_top_keys = {
        "schema_version",
        "record_id",
        "record_type",
        "dataset_id",
        "status",
        "audited_at",
        "authority",
        "lineage",
        "scope_attestation",
        "source_registry",
        "field_and_source_claims",
        "merge_authority_claims",
        "construct_and_chemistry_claims",
        "license_claims",
        "checkpoint_family_exposure",
        "checkpoint_audit_rules",
        "closure_and_remaining_evidence",
        "gate_snapshot",
        "claim_boundary",
    }
    if type(audit) is not dict or set(audit) != expected_top_keys:
        _issue(
            issues,
            "GSE114002_PUBLIC_GAP_AUDIT_CLOSURE",
            path,
            f"top-level keys must be exactly {sorted(expected_top_keys)!r}",
        )
        if not isinstance(audit, Mapping):
            return issues

    for key, value in {
        "schema_version": "route_a_v3_gse114002_public_authority_gap_audit.v1",
        "record_id": "GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_V1",
        "record_type": "PUBLIC_AUTHORITY_GAP_AUDIT_AGGREGATE_ONLY",
        "dataset_id": "GSE114002",
        "status": GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_STATUS,
        "audited_at": "2026-08-11T10:24:16+08:00",
    }.items():
        _expect(audit, key, value, path, issues, "GSE114002_PUBLIC_GAP_AUDIT_METADATA")

    expected_authority = {
        "contract_path": GOAL_PATH,
        "contract_sha256": SOURCE_CONTRACT_SHA256,
        "baseline_registry_path": REGISTRY_PATHS["baseline"],
        "baseline_registry_sha256": "fb47324918b4cdd24a441e0c89909e4956b1f6a83ae64ef359af0bcc1a9371bf",
        "data_role_registry_path": REGISTRY_PATHS["data"],
        "data_role_registry_sha256": "746439ef5d88d8167176d19e9c675746fdc78984a66f6f123f77f6ec49523030",
        "interim_base_path": A1_INTERIM_PATH,
        "interim_base_sha256": "acadf7c36ba0a7601d1b610b664f7455dd1cce4878f05e17ce9b95b78810e464",
        "active_amendment_decision_ids": ["V3-DEC-017", "V3-DEC-018"],
        "repository_base_commit": "0fbec21589d424be7dda01610f9c540df959b4ee",
    }
    authority = audit.get("authority")
    if not isinstance(authority, Mapping):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_AUTHORITY", path, "authority must be a mapping")
    else:
        _expect_closed_mapping(
            authority,
            expected_authority,
            path,
            issues,
            "GSE114002_PUBLIC_GAP_AUDIT_AUTHORITY",
        )

    expected_lineage = {
        "historical_failed_attempt_lineage_id": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID,
        "current_mechanical_closure_lineage_id": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID,
        "predecessor_runtime_event_id": "A1-EVT-039",
        "predecessor_runtime_event_name": "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_ATTEMPT_LINEAGE_SYNCED_GATE_UNCHANGED",
        "runtime_sync_status": "PENDING_NO_EVT_040",
    }
    lineage = audit.get("lineage")
    if not isinstance(lineage, Mapping):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_LINEAGE", path, "lineage must be a mapping")
    else:
        _expect_closed_mapping(
            lineage,
            expected_lineage,
            path,
            issues,
            "GSE114002_PUBLIC_GAP_AUDIT_LINEAGE",
        )

    expected_scope = {
        "ordinary_public_sources_only": True,
        "aggregate_only": True,
        "ordinary_locator_metadata_only": True,
        "sequence_values_included": False,
        "row_identifier_values_included": False,
        "raw_label_values_included": False,
        "per_member_hashes_included": False,
        "model_weight_hashes_included": False,
        "real_row_level_payload_opened": False,
        "model_weight_payload_opened": False,
        "restricted_or_sealed_contact": False,
        "gse246381_contact": False,
        "qualifier_execution_count": 0,
        "training_run_count": 0,
        "gpu_work_count": 0,
        "model_selection_run_count": 0,
        "canonical_materialization_count": 0,
    }
    scope = audit.get("scope_attestation")
    if not isinstance(scope, Mapping):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_SCOPE", path, "scope_attestation must be a mapping")
    else:
        _expect_closed_mapping(
            scope,
            expected_scope,
            path,
            issues,
            "GSE114002_PUBLIC_GAP_AUDIT_SCOPE",
        )

    expected_sources = [
        {
            "source_id": "GEO_SERIES_GSE114002",
            "source_type": "PRIMARY_PUBLIC_SERIES_RECORD",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002",
            "immutable_revision": None,
            "ordinary_locator_metadata": {"accession": "GSE114002"},
            "authority_scope": "SERIES_SAMPLE_DESIGN_AND_REPLICATE_CONTEXT",
        },
        {
            "source_id": "GEO_DESIGNED_SAMPLE_GSM3130443",
            "source_type": "PRIMARY_PUBLIC_SAMPLE_RECORD",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3130443",
            "immutable_revision": None,
            "ordinary_locator_metadata": {
                "accession": "GSM3130443",
                "sample_role": "DESIGNED_LIBRARY",
            },
            "authority_scope": "DESIGNED_SAMPLE_AND_PROCESSED_ASSET_LOCATOR",
        },
        {
            "source_id": "GEO_DESIGNED_LIBRARY_CSV",
            "source_type": "PRIMARY_PUBLIC_PROCESSED_ASSET_LOCATOR",
            "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130443/suppl/GSM3130443_designed_library.csv.gz",
            "immutable_revision": None,
            "ordinary_locator_metadata": {
                "content_length_bytes": 17332142,
                "last_modified": "2018-05-03T16:09:43Z",
                "official_checksum_status": "NOT_PROVIDED",
            },
            "authority_scope": "ASSET_LOCATION_AND_TRANSPORT_METADATA_ONLY",
        },
        {
            "source_id": "PRIMARY_ARTICLE_PMC7100133",
            "source_type": "PRIMARY_RESEARCH_ARTICLE",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7100133/",
            "immutable_revision": None,
            "ordinary_locator_metadata": {
                "doi": "10.1038/s41587-019-0164-5",
                "pmcid": "PMC7100133",
            },
            "authority_scope": "DESIGN_CONSTRUCT_ASSAY_AND_SOURCE_RULES",
        },
        {
            "source_id": "PRIMARY_ARTICLE_NATURE",
            "source_type": "PUBLISHER_RECORD",
            "url": "https://www.nature.com/articles/s41587-019-0164-5",
            "immutable_revision": None,
            "ordinary_locator_metadata": {"doi": "10.1038/s41587-019-0164-5"},
            "authority_scope": "PUBLISHER_IDENTITY_AND_SUPPLEMENT_POINTER",
        },
        {
            "source_id": "AUTHOR_REPOSITORY_IMMUTABLE_COMMIT",
            "source_type": "AUTHOR_CODE_REPOSITORY",
            "url": "https://github.com/pjsample/human_5utr_modeling/tree/d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe",
            "immutable_revision": "d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe",
            "ordinary_locator_metadata": {"repository": "pjsample/human_5utr_modeling"},
            "authority_scope": "AUTHOR_CODE_AND_SAVED_MODEL_FAMILY_SURFACE",
        },
        {
            "source_id": "AUTHOR_NOTEBOOK_IMMUTABLE_COMMIT",
            "source_type": "AUTHOR_ANALYSIS_NOTEBOOK",
            "url": "https://github.com/pjsample/human_5utr_modeling/blob/d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe/human_5utrs/human_utr_modeling.ipynb",
            "immutable_revision": "d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe",
            "ordinary_locator_metadata": {
                "repository_relative_path": "human_5utrs/human_utr_modeling.ipynb"
            },
            "authority_scope": "FIELD_AND_MODEL_USE_DOCUMENTATION_SURFACE",
        },
        {
            "source_id": "AUTHOR_CODE_LICENSE_IMMUTABLE_COMMIT",
            "source_type": "AUTHOR_CODE_LICENSE",
            "url": "https://github.com/pjsample/human_5utr_modeling/blob/d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe/LICENSE",
            "immutable_revision": "d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe",
            "ordinary_locator_metadata": {"license_id": "GPL-3.0"},
            "authority_scope": "AUTHOR_CODE_ONLY_NOT_GEO_DATA",
        },
        {
            "source_id": "GEO_RIGHTS_DISCLAIMER",
            "source_type": "REPOSITORY_RIGHTS_POLICY",
            "url": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
            "immutable_revision": None,
            "ordinary_locator_metadata": {"repository": "NCBI_GEO"},
            "authority_scope": "GENERAL_GEO_ACCESS_AND_SUBMITTER_IP_CAVEAT",
        },
        {
            "source_id": "UTR_LM_PRIMARY_ARTICLE",
            "source_type": "PRIMARY_RESEARCH_ARTICLE",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11155392/",
            "immutable_revision": None,
            "ordinary_locator_metadata": {"pmcid": "PMC11155392"},
            "authority_scope": "UTR_LM_TRAINING_AND_TASK_DISCLOSURE_SURFACE",
        },
        {
            "source_id": "UTR_LM_HUGGINGFACE_EXACT_REVISION_README",
            "source_type": "PUBLIC_MODEL_CARD_AT_IMMUTABLE_REVISION",
            "url": "https://huggingface.co/multimolecule/utrlm-mrl/blob/79e23de069449e659696b5210f833c28ddd0de50/README.md",
            "immutable_revision": "79e23de069449e659696b5210f833c28ddd0de50",
            "ordinary_locator_metadata": {"model_id": "multimolecule/utrlm-mrl"},
            "authority_scope": "MODEL_CARD_DISCLOSURE_ONLY_NOT_MEMBERSHIP_AUDIT",
        },
        {
            "source_id": "MRNABERT_PRIMARY_ARTICLE",
            "source_type": "PRIMARY_RESEARCH_ARTICLE",
            "url": "https://www.nature.com/articles/s41467-025-65340-8",
            "immutable_revision": None,
            "ordinary_locator_metadata": {"article_id": "s41467-025-65340-8"},
            "authority_scope": "MRNABERT_PRETRAIN_AND_DOWNSTREAM_TASK_DISCLOSURE_SURFACE",
        },
        {
            "source_id": "MRNABERT_HUGGINGFACE_EXACT_REVISION",
            "source_type": "PUBLIC_MODEL_REPOSITORY_AT_IMMUTABLE_REVISION",
            "url": "https://huggingface.co/YYLY66/mRNABERT/tree/a1eb7df25804d23f08646e1cb996b234d7208a40",
            "immutable_revision": "a1eb7df25804d23f08646e1cb996b234d7208a40",
            "ordinary_locator_metadata": {"model_id": "YYLY66/mRNABERT"},
            "authority_scope": "MODEL_REPOSITORY_DISCLOSURE_ONLY_NOT_MEMBERSHIP_AUDIT",
        },
        {
            "source_id": "ORTHRUS_PRIMARY_ARTICLE",
            "source_type": "PRIMARY_RESEARCH_ARTICLE",
            "url": "https://www.nature.com/articles/s41592-026-03064-3",
            "immutable_revision": None,
            "ordinary_locator_metadata": {"article_id": "s41592-026-03064-3"},
            "authority_scope": "ORTHRUS_INPUT_CONTEXT_AND_TASK_DISCLOSURE_SURFACE",
        },
        {
            "source_id": "ORTHRUS_HUGGINGFACE_CANDIDATE_FAMILY",
            "source_type": "PUBLIC_MODEL_REPOSITORY_FAMILY_LOCATOR",
            "url": "https://huggingface.co/quietflamingo/orthrus-base-4-track",
            "immutable_revision": None,
            "ordinary_locator_metadata": {
                "model_id": "quietflamingo/orthrus-base-4-track"
            },
            "authority_scope": "CANDIDATE_FAMILY_LOCATOR_ONLY_NOT_ROUTE_A_BINDING",
        },
        {
            "source_id": "GEO_RANDOM_CHEMISTRY_SAMPLE_EXAMPLES",
            "source_type": "PRIMARY_PUBLIC_SAMPLE_SET_LOCATORS",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002",
            "immutable_revision": None,
            "ordinary_locator_metadata": {
                "example_accessions": ["GSM3130435", "GSM3130437", "GSM3130439"],
                "conditions": ["RANDOM_U", "RANDOM_PSI", "RANDOM_M1PSI"],
                "second_replicate_authority": "SERIES_RECORD",
            },
            "authority_scope": "RANDOM_LIBRARY_CHEMISTRY_AND_REPLICATE_CONTEXT_ONLY",
        },
    ]
    if not _json_type_strict_equal(audit.get("source_registry"), expected_sources):
        _issue(
            issues,
            "GSE114002_PUBLIC_GAP_AUDIT_SOURCE_REGISTRY",
            path,
            "source_registry must preserve the exact public URL, immutable-revision, and ordinary-locator registry",
        )

    def expected_claims(rows: Sequence[tuple[str, str, str, str, str]]) -> list[dict[str, str]]:
        return [
            {
                "claim_id": claim_id,
                "evidence_status": status,
                "finding": finding,
                "does_not_establish": boundary,
                "minimum_external_evidence_still_required": evidence,
            }
            for claim_id, status, finding, boundary, evidence in rows
        ]

    expected_field_claims = expected_claims(
        [
            ("FIELD_HEADER_PRESENT", "CONFIRMED", "THE_42_PUBLISHED_FIELD_NAMES_EXIST", "FIELD_VALUE_SEMANTICS_OR_ROW_LEVEL_AUTHORITY", "AUTHOR_FIELD_DICTIONARY_WITH_VALUE_DOMAINS"),
            ("LIBRARY_FIELD_SEMANTICS", "CONFIRMED_PARTIAL", "SUBLIBRARY_OR_DESIGN_FAMILY_CONTEXT_ONLY", "RNA_CHEMISTRY_OR_COMPLETE_DESIGN_DOMAIN", "COMPLETE_DESIGN_FAMILY_MANIFEST"),
            ("DESIGNED_FIELD_SEMANTICS", "CONFIRMED", "INTENDED_DESIGN_EXACT_MATCH_FLAG", "GENETIC_ALGORITHM_FAMILY_MEMBERSHIP", "NONE_FOR_THIS_NARROW_BOOLEAN_SEMANTIC"),
            ("INFO4_FIELD_SEMANTICS", "CONFIRMED_PARTIAL", "HUMAN_OR_VARIANT_COMMON_REFERENCE_VERSUS_VARIANT_CONTEXT", "COMPLETE_FIELD_DOMAIN_OR_ROW_LEVEL_JOIN_AUTHORITY", "AUTHOR_FIELD_DICTIONARY_AND_SOURCE_CROSSWALK"),
            ("INFO1_INFO2_INFO3_FIELD_SEMANTICS", "UNKNOWN_NOT_ASSERTED", "NO_PRIMARY_PUBLIC_DICTIONARY_LOCATED", "ANY_BIOLOGICAL_OR_DESIGN_SEMANTIC", "AUTHOR_FIELD_DICTIONARY"),
            ("MOTHER_AND_MATCH_SCORE_SEMANTICS", "REASONED_INFERENCE", "LIKELY_PARENT_OR_MATCHING_CONTEXT", "EXECUTABLE_PARENT_JOIN_OR_MATCHING_ALGORITHM", "AUTHOR_MATCHING_ALGORITHM_AND_VALIDATED_CROSSWALK"),
            ("ID_FIELD_AUTHORITY", "UNKNOWN_NOT_ASSERTED", "NAMESPACE_AND_UNIQUENESS_NOT_PUBLICLY_DEFINED", "STABLE_ROW_IDENTITY_OR_CROSS_ASSET_JOIN", "AUTHOR_NAMESPACE_AND_UNIQUENESS_SPECIFICATION"),
            ("SAMPLE_AUTHORITY", "CONFIRMED", "GEO_SERIES_AND_SAMPLE_RECORDS_BIND_ASSAY_CONDITION_CONTEXT", "ROW_LEVEL_SOURCE_OR_CHEMISTRY_FOR_DESIGNED_LIBRARY", "ROW_LEVEL_CROSSWALK_AND_DESIGNED_SAMPLE_CHEMISTRY_AUTHORITY"),
            ("HUMAN_SOURCE_RULE", "CONFIRMED_PARTIAL", "ENSEMBL_BIOMART_ANNOTATED_TIS_UPSTREAM_WINDOW", "ENSEMBL_RELEASE_GENOME_BUILD_OR_EXACT_QUERY", "IMMUTABLE_RELEASE_BUILD_QUERY_AND_SOURCE_SNAPSHOT"),
            ("VARIANT_SOURCE_RULE", "CONFIRMED_PARTIAL", "CLINVAR_VARIANTS_IN_SELECTED_REGIONS", "CLINVAR_RELEASE_GENOME_BUILD_OR_SOURCE_CHECKSUM", "IMMUTABLE_RELEASE_BUILD_AND_SOURCE_SNAPSHOT"),
            ("LOCUS_TRANSCRIPT_PARENT_MATCH_AUTHORITY", "UNKNOWN_NOT_ASSERTED", "NO_EXECUTABLE_PUBLIC_CROSSWALK_LOCATED", "BIOLOGICAL_SOURCE_GROUP_OR_PARENT_CHILD_MEMBERSHIP", "AUTHOR_VALIDATED_ROW_CROSSWALK"),
            ("DESIGN_FAMILY_AUTHORITY", "CONFIRMED_PARTIAL", "HIGH_LEVEL_DESIGN_FAMILY_DESCRIPTIONS_EXIST", "COMPLETE_ROW_LEVEL_FAMILY_MEMBERSHIP", "COMPLETE_DESIGN_FAMILY_MANIFEST"),
        ]
    )
    if not _json_type_strict_equal(audit.get("field_and_source_claims"), expected_field_claims):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_FIELD_CLAIMS", path, "field/source claims must preserve exact CONFIRMED, REASONED_INFERENCE, and UNKNOWN_NOT_ASSERTED boundaries")

    expected_merge = {
        "status": "UNKNOWN_NOT_ASSERTED",
        "public_field_dictionary_found": False,
        "public_executable_matching_algorithm_found": False,
        "public_row_crosswalk_found": False,
        "mother_and_match_score_may_be_used_as_join_authority": False,
        "reasoned_inference_may_be_promoted_to_confirmed": False,
        "minimum_external_evidence_still_required": [
            "AUTHOR_FIELD_DICTIONARY",
            "AUTHOR_MATCHING_ALGORITHM",
            "VALIDATED_ROW_CROSSWALK",
            "IMMUTABLE_SOURCE_SNAPSHOTS",
        ],
    }
    if not _json_type_strict_equal(audit.get("merge_authority_claims"), expected_merge):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_MERGE", path, "merge authority must remain unknown and may not promote inferred fields into join authority")

    expected_construct_claims = expected_claims(
        [
            ("DESIGNED_POOL_CONSTRUCT_CHAIN", "CONFIRMED", "DESIGNED_POOL_FIXED_25NT_PREFIX_EGFP_REPORTER_T7_IVT_CAP_POLYA_HEK293T_POLYSOME_CHAIN", "ROW_LEVEL_CONSTRUCT_SEQUENCE_OR_DESIGNED_SAMPLE_RNA_CHEMISTRY", "ACCESSION_SPECIFIC_CONSTRUCT_AND_CHEMISTRY_AUTHORITY"),
            ("DESIGNED_SAMPLE_RNA_CHEMISTRY", "UNKNOWN_NOT_ASSERTED", "NO_PRIMARY_PUBLIC_EXPLICIT_CHEMISTRY_DECLARATION_LOCATED_FOR_GSM3130443", "UNMODIFIED_U_CHEMISTRY", "AUTHOR_OR_ACCESSION_SPECIFIC_CHEMISTRY_DECLARATION"),
            ("DESIGNED_SAMPLE_UNMODIFIED_U", "REASONED_INFERENCE", "DEFAULT_UNMODIFIED_U_IS_PLAUSIBLE_FROM_CONTEXT_ONLY", "CONFIRMED_DESIGNED_SAMPLE_CHEMISTRY", "AUTHOR_OR_ACCESSION_SPECIFIC_CHEMISTRY_DECLARATION"),
            ("RANDOM_LIBRARY_CHEMISTRY_SPLIT", "CONFIRMED", "RANDOM_U_PSI_M1PSI_SAMPLES_AND_MODIFIED_UTP_SUBSTITUTION_ARE_DECLARED", "DESIGNED_SAMPLE_CHEMISTRY", "NONE_FOR_RANDOM_SAMPLE_LEVEL_CLASSIFICATION"),
            ("DESIGNED_CSV_ROW_LEVEL_CHEMISTRY_DISTINGUISHABILITY", "CONFIRMED", "FALSE", "ANY_ROW_LEVEL_CHEMISTRY_ASSIGNMENT", "EXTERNAL_ACCESSION_LEVEL_DESIGNED_SAMPLE_CHEMISTRY_AUTHORITY"),
        ]
    )
    if not _json_type_strict_equal(audit.get("construct_and_chemistry_claims"), expected_construct_claims):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_CONSTRUCT", path, "construct and chemistry claims must preserve the designed-sample chemistry unknown")

    expected_license_claims = expected_claims(
        [
            ("AUTHOR_CODE_LICENSE", "CONFIRMED", "GPL_3_0_AT_IMMUTABLE_AUTHOR_REPOSITORY_COMMIT", "GEO_DATA_LICENSE_OR_REDISTRIBUTION_RIGHTS", "NONE_FOR_CODE_ONLY"),
            ("GEO_GENERAL_ACCESS_POLICY", "CONFIRMED_PARTIAL", "PUBLIC_NCBI_ACCESS_WITH_SUBMITTER_IP_CAVEAT", "ACCESSION_SPECIFIC_SPDX_OR_REDISTRIBUTION_PERMISSION", "ACCESSION_SPECIFIC_DATA_RIGHTS_STATEMENT"),
            ("GSE114002_DATA_REDISTRIBUTION_RIGHTS", "UNKNOWN_NOT_ASSERTED", "NO_ACCESSION_SPECIFIC_DATA_LICENSE_LOCATED", "RIGHT_TO_REDISTRIBUTE_RAW_OR_DERIVED_DATA", "ACCESSION_SPECIFIC_DATA_LICENSE_OR_RIGHTSHOLDER_PERMISSION"),
            ("GPL_PROPAGATION_TO_GEO_DATA", "CONFIRMED", "GPL_CODE_LICENSE_DOES_NOT_AUTOMATICALLY_PROPAGATE_TO_GEO_DATA", "ANY_GEO_DATA_REDISTRIBUTION_RIGHT", "ACCESSION_SPECIFIC_DATA_LICENSE_OR_RIGHTSHOLDER_PERMISSION"),
        ]
    )
    if not _json_type_strict_equal(audit.get("license_claims"), expected_license_claims):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_LICENSE", path, "code GPL and GEO data redistribution authority must remain distinct")

    expected_checkpoints = [
        {
            "checkpoint_family": "OPTIMUS_5PRIME",
            "route_a_binding_status": "UNBOUND_FAMILY_NAME_ONLY",
            "ordinary_locator": None,
            "public_candidate_locator": "pjsample/human_5utr_modeling@saved_models",
            "public_candidate_variants": ["MAIN", "RETRAINED", "EVOLUTION"],
            "public_revision": "d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe",
            "head_semantics_status": "CHECKPOINT_DEPENDENT_NOT_CLOSED",
            "accession_exposure_status": "EXPOSED_ACCESSION_LEVEL_NOT_CHECKPOINT_SPECIFIC",
            "checkpoint_specific_exposure_status": "UNKNOWN_NOT_ASSERTED",
            "near_duplicate_exposure_status": "NOT_RUN",
            "label_exposure_status": "CHECKPOINT_DEPENDENT_NOT_CLOSED",
            "overall_blocker_status": "OPEN",
        },
        {
            "checkpoint_family": "UTR_LM",
            "route_a_binding_status": "UNBOUND_CURRENT_WITH_LEGACY_EXACT_SNAPSHOT_LOCATOR",
            "ordinary_locator": "/home/cunyuliu/.cache/huggingface/hub/models--multimolecule--utrlm-mrl/snapshots/79e23de069449e659696b5210f833c28ddd0de50",
            "public_candidate_locator": "multimolecule/utrlm-mrl",
            "public_candidate_variants": [
                "multimolecule/utrlm-mrl",
                "multimolecule/utrlm-te_el",
            ],
            "public_revision": "79e23de069449e659696b5210f833c28ddd0de50",
            "head_semantics_status": "CHECKPOINT_DEPENDENT_NOT_CLOSED",
            "accession_exposure_status": "EXPOSED_ACCESSION_LEVEL_NOT_CHECKPOINT_SPECIFIC",
            "checkpoint_specific_exposure_status": "UNKNOWN_NOT_ASSERTED",
            "near_duplicate_exposure_status": "NOT_RUN",
            "label_exposure_status": "CHECKPOINT_DEPENDENT_NOT_CLOSED",
            "overall_blocker_status": "OPEN",
        },
        {
            "checkpoint_family": "MRNABERT",
            "route_a_binding_status": "UNBOUND",
            "ordinary_locator": "/home/cunyuliu/.cache/huggingface/hub/models--YYLY66--mRNABERT/refs/main",
            "public_candidate_locator": "YYLY66/mRNABERT",
            "public_candidate_variants": ["YYLY66/mRNABERT"],
            "public_revision": "a1eb7df25804d23f08646e1cb996b234d7208a40",
            "head_semantics_status": "NOT_DECLARED_FOR_BASE_CHECKPOINT_DOWNSTREAM_FINE_TUNING_EXISTS",
            "accession_exposure_status": "NOT_DECLARED_DOES_NOT_ESTABLISH_ABSENCE",
            "checkpoint_specific_exposure_status": "UNKNOWN_NOT_ASSERTED",
            "near_duplicate_exposure_status": "NOT_RUN",
            "label_exposure_status": "NOT_DECLARED_DOES_NOT_ESTABLISH_ABSENCE",
            "overall_blocker_status": "OPEN",
        },
        {
            "checkpoint_family": "ORTHRUS",
            "route_a_binding_status": "UNBOUND_APPLICABILITY_PENDING",
            "ordinary_locator": None,
            "public_candidate_locator": "quietflamingo/orthrus-base-4-track",
            "public_candidate_variants": [
                "quietflamingo/orthrus-base-4-track",
                "quietflamingo/orthrus-large-4-track",
                "quietflamingo/orthrus-large-6-track",
            ],
            "public_revision": None,
            "head_semantics_status": "FULL_MATURE_RNA_INPUT_SHORT_5UTR_REPORTER_ADAPTER_PENDING",
            "accession_exposure_status": "NOT_DECLARED_DOES_NOT_ESTABLISH_ABSENCE",
            "checkpoint_specific_exposure_status": "UNKNOWN_NOT_ASSERTED",
            "near_duplicate_exposure_status": "NOT_RUN",
            "label_exposure_status": "NOT_DECLARED_DOES_NOT_ESTABLISH_ABSENCE",
            "overall_blocker_status": "OPEN",
        },
    ]
    checkpoints = audit.get("checkpoint_family_exposure")
    if not _json_type_strict_equal(checkpoints, expected_checkpoints):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_CHECKPOINTS", path, "four checkpoint rows must preserve accession-level versus checkpoint-specific uncertainty")
    if isinstance(checkpoints, list):
        forbidden_exposure_assertions = {"UNTOUCHED", "ZERO", "UNEXPOSED", "ABSENT", "PASS"}
        for index, row in enumerate(checkpoints):
            if not isinstance(row, Mapping):
                continue
            exact = row.get("checkpoint_specific_exposure_status")
            near = row.get("near_duplicate_exposure_status")
            if exact in forbidden_exposure_assertions or near in forbidden_exposure_assertions:
                _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_EXPOSURE_BYPASS", path, f"checkpoint row {index} may not assert {exact!r}/{near!r} without a complete version-matched audit")

    expected_checkpoint_rules = {
        "checkpoint_specific_minimum_identity": ["MODEL_ID", "IMMUTABLE_REVISION", "HEAD_SEMANTICS"],
        "revision_is_artifact_digest": False,
        "artifact_digest_is_training_membership_audit": False,
        "not_declared_establishes_absence": False,
        "untouched_or_zero_without_complete_member_audit_allowed": False,
        "label_exposure_scope": ["PRETRAIN", "FINE_TUNE", "HEAD_TRAINING", "MODEL_SELECTION"],
        "near_duplicate_audit_minimum_prefreeze": ["NORMALIZATION", "COMPARISON_UNIT", "METRIC", "THRESHOLD"],
    }
    if not _json_type_strict_equal(audit.get("checkpoint_audit_rules"), expected_checkpoint_rules):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_CHECKPOINT_RULES", path, "checkpoint audit rules must remain exact and fail closed")

    expected_minimum_evidence = [
        {"blocker": "CHECKPOINT_SPECIFIC_EXPOSURE_UNKNOWN_NOT_ASSERTED", "minimum_external_evidence": "SELECTED_MODEL_ID_IMMUTABLE_REVISION_HEAD_SEMANTICS_COMPLETE_VERSION_MATCHED_MEMBERSHIP_AND_PREFROZEN_NEAR_DUPLICATE_AUDIT"},
        {"blocker": "FIELD_AND_BIOLOGICAL_SOURCE_AUTHORITY_UNKNOWN_NOT_ASSERTED", "minimum_external_evidence": "AUTHOR_FIELD_DICTIONARY_MATCHING_ALGORITHM_IMMUTABLE_SOURCE_SNAPSHOTS_VALIDATED_ROW_CROSSWALK_AND_COMPLETE_DESIGN_FAMILY_MANIFEST"},
        {"blocker": "FULL_CONSTRUCT_PREFIX_REPORTER_RNA_CHEMISTRY_UNKNOWN_NOT_ASSERTED", "minimum_external_evidence": "ACCESSION_SPECIFIC_DESIGNED_SAMPLE_CONSTRUCT_PREFIX_REPORTER_CHAIN_AND_RNA_CHEMISTRY_AUTHORITY"},
        {"blocker": "LICENSE_AND_REDISTRIBUTION_RIGHTS_UNKNOWN_NOT_ASSERTED", "minimum_external_evidence": "ACCESSION_SPECIFIC_DATA_LICENSE_OR_RIGHTSHOLDER_PERMISSION_FOR_INTENDED_DERIVED_ARTIFACT_USE"},
        {"blocker": "NEAR_DUPLICATE_SPLIT_AND_LEAKAGE_AUDIT_NOT_RUN", "minimum_external_evidence": "OUTCOME_BLIND_PREFROZEN_GROUP_AND_NEAR_DUPLICATE_AUDIT_AFTER_SOURCE_GROUP_AUTHORITY_CLOSES"},
        {"blocker": "OWNER_UNCERTAINTY_POLICY_UNKNOWN_NOT_ASSERTED", "minimum_external_evidence": "OWNER_APPROVED_APPEND_ONLY_DECISION_LOG_ENTRY"},
        {"blocker": "PREFROZEN_GROUP_POWER_NOT_RUN", "minimum_external_evidence": "OUTCOME_BLIND_POWER_AND_CONFIDENCE_INTERVAL_AUDIT_ON_AUTHORIZED_BIOLOGICAL_SOURCE_GROUPS"},
    ]
    expected_closure = {
        "engineering_items_closed_by_this_audit": ["PUBLIC_AUTHORITY_EVIDENCE_SEARCH_AND_GAP_CLASSIFICATION_COMPLETED"],
        "science_blockers_closed_by_this_audit": [],
        "historical_mechanical_blockers_closed_before_this_audit": GSE114002_ENDPOINT_GEOMETRY_CLOSED_BLOCKERS,
        "remaining_science_blockers": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_BLOCKERS,
        "field_authority_subblockers": [
            "B_FIELD_DICTIONARY",
            "B_MATCHING_ALGORITHM",
            "B_SOURCE_SNAPSHOT",
            "B_ROW_CROSSWALK",
            "B_COMPLETE_DESIGN_FAMILY_MANIFEST",
            "B_DESIGNED_SAMPLE_CHEMISTRY",
            "B_DATA_LICENSE",
            "B_IMMUTABLE_DATA_HASH",
        ],
        "minimum_external_evidence_by_science_blocker": expected_minimum_evidence,
    }
    if not _json_type_strict_equal(audit.get("closure_and_remaining_evidence"), expected_closure):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_BLOCKERS", path, "the audit may close only its engineering search item and must retain all seven scientific blockers")

    expected_gate = {
        "qualified_independent_ordinary_studies": 0,
        "qualified_a1_studies": 0,
        "qualified_true_a2_dense_studies": 0,
        "canonical_record_count": 0,
        "qualified": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "phase_complete": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    }
    gate = audit.get("gate_snapshot")
    if not isinstance(gate, Mapping):
        _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_GATE", path, "gate_snapshot must be a mapping")
    else:
        _expect_closed_mapping(gate, expected_gate, path, issues, "GSE114002_PUBLIC_GAP_AUDIT_GATE")

    expected_boundary = (
        "This aggregate-only audit records which public authority statements are confirmed, inferred, or still unknown. "
        "It closes only completion of the public evidence search and gap classification engineering item. It does not close "
        "any of the seven scientific blockers, qualify GSE114002, establish a true-A2 claim, create canonical records, "
        "authorize training or model selection, or assert checkpoint exact or near-duplicate non-exposure."
    )
    _expect(audit, "claim_boundary", expected_boundary, path, issues, "GSE114002_PUBLIC_GAP_AUDIT_CLAIM_BOUNDARY")

    forbidden_keys = {
        "sequence",
        "sequences",
        "sequence_value",
        "sequence_hash",
        "sequence_sha256",
        "row_id",
        "row_ids",
        "row_identifier",
        "row_identifiers",
        "label_value",
        "label_values",
        "effect_value",
        "effect_values",
        "member_hash",
        "member_hashes",
        "member_sha256",
        "model_weight_hash",
        "model_weight_sha256",
        "checkpoint_sha256",
        "weight_sha256",
    }

    def scan_private_values(node: Any, json_path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                normalized = str(key).lower()
                if normalized in forbidden_keys:
                    _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_PRIVACY", path, f"forbidden payload-bearing key at {json_path}.{key}")
                scan_private_values(value, f"{json_path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                scan_private_values(value, f"{json_path}[{index}]")
        elif isinstance(node, str):
            compact = node.upper().replace(" ", "").replace("-", "").replace("_", "")
            if len(compact) >= 20 and set(compact) <= set("ACGTUN"):
                _issue(issues, "GSE114002_PUBLIC_GAP_AUDIT_PRIVACY", path, f"sequence-like value is forbidden at {json_path}")

    scan_private_values(audit, "$")
    return issues


def validate_registry_manifest(repo_root: Path) -> list[Issue]:
    """Verify every public bundle hash listed by the A0 registry manifest."""

    issues: list[Issue] = []
    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "REGISTRY_MANIFEST_UNREADABLE", REGISTRY_MANIFEST_PATH, str(exc))
        return issues

    expected_static_top = {
        "contract_id": CONTRACT_ID,
        "version": VERSION,
        "schema_version": "1.0.0",
        "contract_path": GOAL_PATH,
        "initial_contract_sha256": "d1c031aecdec710495f6861b380785cccd64663ac4bd97b4f479d6fdf372ea07",
        "contract_sha256": SOURCE_CONTRACT_SHA256,
        "active_amendment_decision_ids": ACTIVE_AMENDMENT_DECISION_IDS,
        "base_commit": "bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c",
        "manifest_status": "A1_DEC019_GSE200304_POST_ADJUDICATION_LEDGER_REGISTERED",
        "initial_generated_at": "2026-08-10T10:10:05+08:00",
        "generated_at": GSE200304_DEC019_POST_ADJUDICATION_MANIFEST_AT,
        "updated_at": GSE200304_DEC019_POST_ADJUDICATION_MANIFEST_AT,
        "sealed_contact": False,
    }
    expected_top_keys = set(expected_static_top) | {"files"}
    if type(manifest) is not dict or set(manifest) != expected_top_keys:
        _issue(
            issues,
            "REGISTRY_MANIFEST_CLOSURE",
            REGISTRY_MANIFEST_PATH,
            f"top-level keys must be exactly {sorted(expected_top_keys)!r}",
        )
    for key, value in expected_static_top.items():
        _expect(
            manifest,
            key,
            value,
            REGISTRY_MANIFEST_PATH,
            issues,
            "REGISTRY_MANIFEST_METADATA",
        )
    if manifest.get("sealed_contact") is not False:
        _issue(
            issues,
            "REGISTRY_MANIFEST_METADATA",
            REGISTRY_MANIFEST_PATH,
            "sealed_contact must be the JSON boolean false",
        )

    entries = manifest.get("files")
    if type(entries) is not list or not entries:
        _issue(issues, "REGISTRY_MANIFEST_FILES", REGISTRY_MANIFEST_PATH, "files must be a non-empty JSON array")
        return issues
    observed_path_roles: list[tuple[Any, Any]] = []
    by_path: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            _issue(issues, "REGISTRY_MANIFEST_ENTRY", REGISTRY_MANIFEST_PATH, f"files[{index}] is not an object")
            observed_path_roles.append((None, None))
            continue
        if set(entry) != {"path", "role", "sha256"}:
            _issue(
                issues,
                "REGISTRY_MANIFEST_CLOSURE",
                REGISTRY_MANIFEST_PATH,
                f"files[{index}] keys must be exactly ['path', 'role', 'sha256']",
            )
        relative = entry.get("path")
        declared = entry.get("sha256")
        role = entry.get("role")
        observed_path_roles.append((relative, role))
        if type(relative) is not str or type(declared) is not str or type(role) is not str or not role:
            _issue(issues, "REGISTRY_MANIFEST_ENTRY", REGISTRY_MANIFEST_PATH, f"files[{index}] requires path/role/sha256 strings")
            continue
        if relative in by_path:
            _issue(issues, "REGISTRY_MANIFEST_DUPLICATE", REGISTRY_MANIFEST_PATH, f"duplicate path {relative!r}")
            continue
        by_path[relative] = entry
        if relative == REGISTRY_MANIFEST_PATH:
            _issue(issues, "REGISTRY_MANIFEST_SELF_REFERENCE", REGISTRY_MANIFEST_PATH, "manifest may not hash itself")
            continue
        if len(declared) != 64 or any(ch not in "0123456789abcdef" for ch in declared):
            _issue(issues, "REGISTRY_MANIFEST_HASH_FORMAT", REGISTRY_MANIFEST_PATH, f"invalid SHA-256 for {relative}")
            continue
        try:
            actual = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "REGISTRY_MANIFEST_FILE_MISSING_OR_UNSAFE", relative, str(exc))
            continue
        if actual != declared:
            _issue(issues, "REGISTRY_MANIFEST_HASH_MISMATCH", relative, f"got {actual}, expected {declared}")

    if not _json_type_strict_equal(
        observed_path_roles,
        list(EXPECTED_REGISTRY_MANIFEST_PATH_ROLES),
    ):
        _issue(
            issues,
            "REGISTRY_MANIFEST_CLOSURE",
            REGISTRY_MANIFEST_PATH,
            "files must preserve the exact ordered path-to-role registry",
        )
    if set(by_path) != MANDATORY_REGISTRY_MANIFEST_PATHS:
        _issue(
            issues,
            "REGISTRY_MANIFEST_COVERAGE",
            REGISTRY_MANIFEST_PATH,
            "listed paths must equal the exact mandatory registry-manifest path set",
        )
    cyclic_dynamic_paths = set(by_path) & DEC019_SUCCESSOR_DYNAMIC_CONFIG_PATHS
    if cyclic_dynamic_paths:
        _issue(
            issues,
            "DEC019_SUCCESSOR_MANIFEST_CYCLE",
            REGISTRY_MANIFEST_PATH,
            "dynamic successor configs must be required and core-validated but not exact-hashed "
            f"by the static manifest: {sorted(cyclic_dynamic_paths)!r}",
        )
    try:
        goal_hash = sha256_bytes(_read_bytes(repo_root, GOAL_PATH))
        if goal_hash != manifest.get("contract_sha256"):
            _issue(issues, "REGISTRY_MANIFEST_CONTRACT_HASH", GOAL_PATH, "top-level contract hash does not match contract bytes")
    except (FileNotFoundError, ValueError) as exc:
        _issue(issues, "REGISTRY_MANIFEST_CONTRACT_MISSING", GOAL_PATH, str(exc))

    generated_at = manifest.get("generated_at")
    updated_at = manifest.get("updated_at")
    if generated_at != updated_at:
        _issue(issues, "REGISTRY_MANIFEST_TIME", REGISTRY_MANIFEST_PATH, "generated_at and updated_at must identify the same amended manifest bytes")
    try:
        manifest_updated = datetime.fromisoformat(str(updated_at))
        if manifest_updated.utcoffset() is None:
            raise ValueError("manifest updated_at has no UTC offset")
        interim = _load_yaml(repo_root, A1_INTERIM_PATH)
        interim_updated = datetime.fromisoformat(str(interim.get("updated_at")))
        if interim_updated.utcoffset() is None:
            raise ValueError("A1 interim updated_at has no UTC offset")
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        _issue(issues, "REGISTRY_MANIFEST_TIME", REGISTRY_MANIFEST_PATH, f"cannot validate causal timestamps: {exc}")
    else:
        if manifest_updated < interim_updated:
            _issue(
                issues,
                "REGISTRY_MANIFEST_TIME",
                REGISTRY_MANIFEST_PATH,
                "manifest updated_at must not predate the A1 interim bytes it hashes",
            )
    return issues


def validate_decision_log(decision_log: Mapping[str, Any]) -> list[Issue]:
    """Freeze required A0 decisions independently of the manifest hash."""

    issues: list[Issue] = []
    expected_metadata = {
        "schema_version": "1.0.0",
        "contract_id": CONTRACT_ID,
        "log_id": "ROUTE_A_V3_DECISIONS",
        "append_only": True,
        "created_at": "2026-08-10T00:32:15+08:00",
    }
    for key, value in expected_metadata.items():
        if decision_log.get(key) != value:
            _issue(issues, "DECISION_LOG_METADATA", DECISION_LOG_PATH, f"{key} must remain {value!r}")
    expected_top_level = {*expected_metadata, "decisions"}
    if set(decision_log) != expected_top_level:
        _issue(
            issues,
            "DECISION_LOG_TOP_LEVEL_SHAPE",
            DECISION_LOG_PATH,
            f"top-level keys must be exactly {sorted(expected_top_level)!r}",
        )
    raw = decision_log.get("decisions")
    if not isinstance(raw, list):
        _issue(issues, "DECISION_LOG_ENTRIES", DECISION_LOG_PATH, "decisions must be a list")
        return issues
    decisions = {
        entry.get("decision_id"): entry
        for entry in raw
        if isinstance(entry, Mapping) and isinstance(entry.get("decision_id"), str)
    }
    if len(decisions) != len(raw):
        _issue(issues, "DECISION_LOG_DUPLICATE_OR_INVALID", DECISION_LOG_PATH, "decision IDs must be unique strings")
    if set(decisions) != set(EXPECTED_DECISION_IDS):
        _issue(issues, "DECISION_LOG_ID_CLOSURE", DECISION_LOG_PATH, "decision IDs must be exactly V3-DEC-001 through V3-DEC-019")
    ordered_ids = [entry.get("decision_id") if isinstance(entry, Mapping) else None for entry in raw]
    if ordered_ids != list(EXPECTED_DECISION_IDS):
        _issue(issues, "DECISION_LOG_ORDER", DECISION_LOG_PATH, "accepted decision prefix must remain in exact DEC-001 through DEC-019 order")
    for decision_id, dimension in EXPECTED_DECISION_DIMENSIONS.items():
        entry = decisions.get(decision_id)
        if not isinstance(entry, Mapping):
            continue
        canonical = json.dumps(
            dict(entry),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        actual_digest = sha256_bytes(canonical)
        expected_digest = EXPECTED_DECISION_ENTRY_SHA256[decision_id]
        if actual_digest != expected_digest:
            _issue(
                issues,
                "DECISION_LOG_ENTRY_DRIFT",
                DECISION_LOG_PATH,
                f"{decision_id} canonical digest {actual_digest} does not match accepted prefix {expected_digest}",
            )
        if entry.get("dimension") != dimension:
            _issue(issues, "DECISION_LOG_DIMENSION", DECISION_LOG_PATH, f"{decision_id}.dimension must be {dimension!r}")
        if entry.get("sealed_contact") is not False:
            _issue(issues, "DECISION_LOG_SEALED_CONTACT", DECISION_LOG_PATH, f"{decision_id}.sealed_contact must be false")
        if entry.get("decision_type") not in {"DECISION", "AMENDMENT"}:
            _issue(issues, "DECISION_LOG_TYPE", DECISION_LOG_PATH, f"{decision_id} has invalid decision_type")
        if not isinstance(entry.get("evidence_refs"), list) or not entry.get("evidence_refs"):
            _issue(issues, "DECISION_LOG_EVIDENCE", DECISION_LOG_PATH, f"{decision_id} requires evidence_refs")

    exact_resolutions = {
        "V3-DEC-001": "ROUTE_A_FULL_XEDITFLOW",
        "V3-DEC-003": "PUBLIC_DATA_ONLY_NO_NEW_WETLAB_L4_PROHIBITED",
        "V3-DEC-004": "PRIMARY_K_1_3_5_SECONDARY_K_10",
        "V3-DEC-005": "AT_LEAST_3_ORDINARY_STUDIES_WITH_AT_LEAST_2_A1_AND_1_A2",
        "V3-DEC-009": "bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c",
    }
    for decision_id, resolution in exact_resolutions.items():
        entry = decisions.get(decision_id)
        if isinstance(entry, Mapping) and entry.get("resolution") != resolution:
            _issue(issues, "DECISION_LOG_RESOLUTION", DECISION_LOG_PATH, f"{decision_id}.resolution must remain {resolution!r}")

    required_resolution_tokens = {
        "V3-DEC-002": ("Evidence status", "claim status", "cannot be"),
        "V3-DEC-006": ("ADDITIVE_UPLIFT", "SOURCE_GROUP_BOOTSTRAP", "CI_LOWER_GT_0"),
        "V3-DEC-007": ("THREE_UTR_PRIORITY", "CDS_PARALLEL", "AT_LEAST_ONE"),
        "V3-DEC-008": ("identifiable public-intervention effect learning", "potential-consistent legal mRNA control", "no first-use claim"),
        "V3-DEC-010": ("PRE_V3_DEVELOPMENT_ONLY", "ineligible for V3 gates", "original M0/E0 failures"),
        "V3-DEC-011": ("No GPU is treated as free", "CPU-only"),
        "V3-DEC-012": ("SEALED_EXTERNAL_FINAL_ONLY", "may not write ACCESS_INTENT", "explicit user authorization"),
        "V3-DEC-013": ("second focused A0 activation record", "does not alter contract thresholds"),
        "V3-DEC-014": ("ORIGINAL_M0_EFFECT_GATE_FAIL", "0.510", "0.60", "unimplemented sealed evaluator"),
        "V3-DEC-015": ("independent", "A9 effective execution configuration snapshot", "no authorization pointers", "active config remains separately authority-bound and fail-closed"),
        "V3-DEC-016": (
            "Supersede every latent success path",
            "unconditional A0-A9 hard disable",
            "No configuration toggle",
            "authorization record",
            "readiness record",
            "execution manifest",
            "synthetic positive fixture",
            "A9 must replace this guard only after",
            "A10 still requires separate explicit user authorization",
        ),
        "V3-DEC-017": (
            "user-authorized Scheme A",
            "ABSOLUTE_AUXILIARY_ONLY",
            "TRUE_A2_NOT_QUALIFIED",
            "zero contribution",
            "A2_RECOVERY_CANDIDATE_NOT_QUALIFIED",
            "SEQUENCE_EXPOSED",
            "at least three ordinary studies",
            "at least two A1",
            "at least one genuine source-anchored true A2",
            "new genuine public A2 study is required",
            "no qualified count",
            "no GPU training",
            "no claim",
        ),
        "V3-DEC-018": (
            "COMMITTED_AND_ACCEPTED",
            "EXACT_OFFICIAL_SRR_ROLE_AUTHORITY_CLOSED",
            "24-run grid",
            "High_Poly",
            "Low_Poly",
            "pDNA",
            "Total_RNA",
            "EXACT_SRR_SAMPLE_ROLES_UNKNOWN",
            "17 historical blockers",
            "no 80S_RNA",
            "pDNA != 80S_RNA",
            "REQUIRED_80S_ROLE_AUTHORITY_ABSENT",
            "0/0/0",
            "qualified=false",
            "canonical record count remains 0",
            "PENDING_NO_EVT_035",
            "does not create EVT-035",
        ),
        "V3-DEC-019": (
            "root contract bytes",
            "power >= 0.80",
            "full CI width <= 0.30",
            "Candidate Hamming distances 1, 2, and 3",
            "global primary list [1,3,5]",
            "K5 is claim-boundary-only",
            "ordinary=1, A1=0, true-A2=1",
            "ordinary=1, A1=1, true-A2=0",
            "dataset-scoped",
            "GSE149487 three-biological-replicate",
            "qualifies no study",
            "establishes no scientific claim",
        ),
    }
    for decision_id, tokens in required_resolution_tokens.items():
        entry = decisions.get(decision_id)
        resolution = str(entry.get("resolution", "")) if isinstance(entry, Mapping) else ""
        missing = [token for token in tokens if token not in resolution]
        if missing:
            _issue(issues, "DECISION_LOG_KEY_DECISION", DECISION_LOG_PATH, f"{decision_id} resolution missing {missing!r}")

    amendment = decisions.get("V3-DEC-006")
    if isinstance(amendment, Mapping):
        if amendment.get("decision_type") != "AMENDMENT" or amendment.get("supersedes_decision_id") != "XE-DEC-008":
            _issue(issues, "DECISION_LOG_UPLIFT_AMENDMENT", DECISION_LOG_PATH, "V3-DEC-006 must remain the frozen amendment of XE-DEC-008")
        history = {str(value).lower() for value in amendment.get("historical_values_preserved", [])}
        if history != {"0.1322", "9.92x"}:
            _issue(issues, "DECISION_LOG_UPLIFT_HISTORY", DECISION_LOG_PATH, "old 0.1322 and 9.92x values must remain preserved")

    security_design = decisions.get("V3-DEC-015")
    if isinstance(security_design, Mapping):
        expected_security_fields = {
            "decision_type": "DECISION",
            "dimension": "sealed_execution_freeze_hash_scope",
            "status": "FROZEN_A0_SECURITY_DESIGN",
            "effective_phase": "A0",
            "requires_user_authorization": False,
            "sealed_contact": False,
        }
        for key, value in expected_security_fields.items():
            if security_design.get(key) != value:
                _issue(issues, "DECISION_LOG_SECURITY_DESIGN", DECISION_LOG_PATH, f"V3-DEC-015.{key} must remain {value!r}")
        evidence_refs = security_design.get("evidence_refs")
        required_refs = {
            CONFIG_PATH,
            SEALED_GUARD_PATH,
            SEALED_RUNNER_PATH,
        }
        if not isinstance(evidence_refs, list) or not required_refs <= set(evidence_refs):
            _issue(issues, "DECISION_LOG_SECURITY_EVIDENCE", DECISION_LOG_PATH, f"V3-DEC-015 evidence_refs must include {sorted(required_refs)!r}")

    phase_boundary = decisions.get("V3-DEC-016")
    if isinstance(phase_boundary, Mapping):
        expected_boundary_fields = {
            "decision_type": "AMENDMENT",
            "dimension": "sealed_a0_phase_boundary",
            "status": "FROZEN_A0_PHASE_BOUNDARY",
            "supersedes_decision_id": "V3-DEC-015",
            "effective_phase": "A0",
            "requires_user_authorization": False,
            "sealed_contact": False,
        }
        for key, value in expected_boundary_fields.items():
            if phase_boundary.get(key) != value:
                _issue(issues, "DECISION_LOG_A0_PHASE_BOUNDARY", DECISION_LOG_PATH, f"V3-DEC-016.{key} must remain {value!r}")
        evidence_refs = phase_boundary.get("evidence_refs")
        expected_refs = {
            GOAL_PATH,
            CONFIG_PATH,
            SEALED_GUARD_PATH,
            SEALED_RUNNER_PATH,
        }
        if not isinstance(evidence_refs, list) or set(evidence_refs) != expected_refs:
            _issue(issues, "DECISION_LOG_A0_PHASE_BOUNDARY_EVIDENCE", DECISION_LOG_PATH, f"V3-DEC-016 evidence_refs must be exactly {sorted(expected_refs)!r}")

    role_amendment = decisions.get("V3-DEC-017")
    if isinstance(role_amendment, Mapping):
        expected_role_fields = {
            "decision_type": "AMENDMENT",
            "dimension": "gse145046_true_a2_role_and_a2_recovery",
            "status": "FROZEN_USER_AUTHORIZED_A1_ROLE_AMENDMENT",
            "effective_phase": "A1",
            "requires_user_authorization": True,
            "user_authorization_status": "GRANTED",
            "preserves_decision_ids": ["V3-DEC-005"],
            "sealed_contact": False,
        }
        for key, value in expected_role_fields.items():
            if role_amendment.get(key) != value:
                _issue(issues, "DECISION_LOG_A1_ROLE_AMENDMENT", DECISION_LOG_PATH, f"V3-DEC-017.{key} must remain {value!r}")
        evidence_refs = role_amendment.get("evidence_refs")
        required_refs = {
            GOAL_PATH,
            REGISTRY_PATHS["data"],
            "configs/route_a_v3_gse145046_a2_audit.json",
        }
        if not isinstance(evidence_refs, list) or not required_refs <= set(evidence_refs):
            _issue(issues, "DECISION_LOG_A1_ROLE_AMENDMENT_EVIDENCE", DECISION_LOG_PATH, f"V3-DEC-017 evidence_refs must include {sorted(required_refs)!r}")
    official_role_authority = decisions.get("V3-DEC-018")
    if isinstance(official_role_authority, Mapping):
        expected_role_authority_fields = {
            "decision_type": "AMENDMENT",
            "dimension": "gse200302_official_srr_role_authority_and_raw_replay_boundary",
            "status": "FROZEN_A1_OFFICIAL_METADATA_ROLE_AUTHORITY",
            "effective_phase": "A1",
            "requires_user_authorization": False,
            "accepted_at": "2026-08-10T23:10:00+08:00",
            "preserves_decision_ids": ["V3-DEC-005", "V3-DEC-017"],
            "artifact_root": GSE200302_ROLE_ARTIFACT_ROOT,
            "publication_status": "COMMITTED_AND_ACCEPTED",
            "role_authority_status": "EXACT_OFFICIAL_SRR_ROLE_AUTHORITY_CLOSED",
            "prior_blocker": "EXACT_SRR_SAMPLE_ROLES_UNKNOWN",
            "prior_blocker_status": "CLOSED",
            "replacement_blocker": "REQUIRED_80S_ROLE_AUTHORITY_ABSENT",
            "replacement_blocker_status": "OPEN",
            "role_grid_status": "CONFLICT_WITH_CURRENT_80S_EXPECTATION",
            "pdna_may_substitute_for_80s_rna": False,
            "bundle_digest": GSE200302_ROLE_BUNDLE_DIGEST,
            "implementation_commit": "d042d7c1706a80821a19b78334985441bcf6eb86",
            "binding_commit": "e3b724d00a9e5263b99475b9744fc0bb68a3ab67",
            "independent_consumer_status": "ACCEPTED",
            "runtime_sync_status": "PENDING_NO_EVT_035",
            "sealed_contact": False,
        }
        for key, value in expected_role_authority_fields.items():
            if not _json_type_strict_equal(official_role_authority.get(key), value):
                _issue(
                    issues,
                    "DECISION_LOG_GSE200302_ROLE_AUTHORITY",
                    DECISION_LOG_PATH,
                    f"V3-DEC-018.{key} must remain {value!r}",
                )
        evidence_refs = official_role_authority.get("evidence_refs")
        required_refs = {
            GOAL_PATH,
            REGISTRY_PATHS["data"],
            GSE200302_ROLE_CONFIG_PATH,
            f"{GSE200302_ROLE_ARTIFACT_ROOT}/ROLE_AUTHORITY.json",
            f"{GSE200302_ROLE_ARTIFACT_ROOT}/PUBLICATION_COMMIT.json",
        }
        if not isinstance(evidence_refs, list) or set(evidence_refs) != required_refs:
            _issue(
                issues,
                "DECISION_LOG_GSE200302_ROLE_AUTHORITY_EVIDENCE",
                DECISION_LOG_PATH,
                f"V3-DEC-018 evidence_refs must be exactly {sorted(required_refs)!r}",
            )
    dec019 = decisions.get("V3-DEC-019")
    if isinstance(dec019, Mapping):
        expected_dec019_fields = {
            "decision_type": "AMENDMENT",
            "dimension": "a1_measurement_uncertainty_split_and_claim_boundary",
            "status": "FROZEN_USER_AUTHORIZED_A1_QUALIFICATION_POLICY",
            "effective_phase": "A1",
            "requires_user_authorization": True,
            "user_authorization_status": "GRANTED",
            "user_authorization_received_at": "2026-08-11T10:42:53+08:00",
            "preserves_decision_ids": ["V3-DEC-005", "V3-DEC-017", "V3-DEC-018"],
            "accepted_prefix_preserved_through": "V3-DEC-018",
            "amendment_path": DEC019_AMENDMENT_PATH,
            "current_qualified_counts": {
                "ordinary": 0,
                "a1": 0,
                "true_a2": 0,
                "canonical_records": 0,
            },
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "sealed_contact": False,
        }
        for key, value in expected_dec019_fields.items():
            if not _json_type_strict_equal(dec019.get(key), value):
                _issue(
                    issues,
                    "DECISION_LOG_DEC019",
                    DECISION_LOG_PATH,
                    f"V3-DEC-019.{key} must remain {value!r}",
                )
        expected_refs = {
            GOAL_PATH,
            DEC019_AMENDMENT_PATH,
            A1_QUALIFICATION_CONFIG_PATH,
            REGISTRY_PATHS["data"],
            REGISTRY_PATHS["split"],
            REGISTRY_PATHS["task"],
            REGISTRY_PATHS["claim"],
        }
        if not isinstance(dec019.get("evidence_refs"), list) or set(dec019["evidence_refs"]) != expected_refs:
            _issue(
                issues,
                "DECISION_LOG_DEC019_EVIDENCE",
                DECISION_LOG_PATH,
                f"V3-DEC-019 evidence_refs must be exactly {sorted(expected_refs)!r}",
            )
    return issues


def validate_scheme_a_data_roles(data_registry: Mapping[str, Any]) -> list[Issue]:
    """Freeze the user-authorized A1 Scheme-A role correction."""

    issues: list[Issue] = []
    path = REGISTRY_PATHS["data"]
    policy = data_registry.get("data_policy")
    expected_policy = {
        "ordinary_minimum_independent_studies": 3,
        "ordinary_minimum_a1_studies": 2,
        "ordinary_minimum_a2_dense_studies": 1,
    }
    if not isinstance(policy, Mapping):
        _issue(issues, "SCHEME_A_DATA_POLICY", path, "data_policy mapping is required")
    else:
        for key, value in expected_policy.items():
            if policy.get(key) != value:
                _issue(issues, "SCHEME_A_GATE_PRESERVATION", path, f"data_policy.{key} must remain {value!r}")

    expected_ordinary = {
        "GSE114002",
        "GSE149487",
        "GSE217518",
        "GSE200304",
        "ENCSR854RUF",
        "GSE232572",
        "GSE186455",
        "GSE207584",
    }
    ordinary = data_registry.get("ordinary_candidate_dataset_ids")
    if not isinstance(ordinary, list) or set(ordinary) != expected_ordinary or len(ordinary) != len(expected_ordinary):
        _issue(issues, "SCHEME_A_ORDINARY_CANDIDATES", path, "ordinary candidates must exclude GSE145046 and preserve the other eight candidates")
    if data_registry.get("absolute_auxiliary_dataset_ids") != ["GSE145046"]:
        _issue(issues, "SCHEME_A_ABSOLUTE_AUXILIARY", path, "absolute_auxiliary_dataset_ids must be exactly [GSE145046]")
    if data_registry.get("true_a2_recovery_candidate_dataset_ids") != ["GSE114002"]:
        _issue(issues, "SCHEME_A_TRUE_A2_RECOVERY", path, "true_a2_recovery_candidate_dataset_ids must be exactly [GSE114002]")

    rows = data_registry.get("datasets")
    by_id = {
        row.get("dataset_id"): row
        for row in rows
        if isinstance(rows, list) and isinstance(row, Mapping) and isinstance(row.get("dataset_id"), str)
    } if isinstance(rows, list) else {}
    gse145046 = by_id.get("GSE145046")
    expected_gse145046 = {
        "role": "AUDIT_ONLY",
        "qualification_status": "AUDIT_PENDING",
        "true_a2_qualification_status": "REJECTED_WITH_EVIDENCE",
        "qualified": False,
        "training_role": "EXCLUDED_PENDING_QUALIFICATION",
        "intended_role_if_qualified": "ABSOLUTE_AUXILIARY_FIXED_REPORTER_LANDSCAPE",
        "intended_evidence_grade_if_qualified": "AUXILIARY_ONLY_NOT_A1_OR_A2",
        "ordinary_gate_contribution": 0,
        "a1_gate_contribution": 0,
        "true_a2_gate_contribution": 0,
        "source_relative_confirmatory_evidence_allowed": False,
        "true_a2_evidence_status": "FAIL_CURRENT_PROTOCOL",
    }
    if not isinstance(gse145046, Mapping):
        _issue(issues, "SCHEME_A_GSE145046_MISSING", path, "GSE145046 data-role row is required")
    else:
        for key, value in expected_gse145046.items():
            if gse145046.get(key) != value:
                _issue(issues, "SCHEME_A_GSE145046_ROLE", path, f"GSE145046.{key} must remain {value!r}")
        expected_permanent_forbidden = {
            "ORDINARY_STUDY_GATE_CREDIT",
            "A1_GATE_CREDIT",
            "TRUE_A2_GATE_CREDIT",
            "SOURCE_RELATIVE_CONFIRMATORY_EVIDENCE",
        }
        if set(gse145046.get("permanently_forbidden_gate_uses", [])) != expected_permanent_forbidden:
            _issue(issues, "SCHEME_A_GSE145046_FORBIDDEN", path, "GSE145046 gate and confirmatory prohibitions must remain closed")

    gse114002 = by_id.get("GSE114002")
    if not isinstance(gse114002, Mapping):
        _issue(issues, "SCHEME_A_GSE114002_MISSING", path, "GSE114002 data-role row is required")
    else:
        intended = gse114002.get("intended_role_if_qualified")
        if not isinstance(intended, Mapping) or intended.get("designed_library") != "A2_SOURCE_ANCHORED_RECOVERY_CANDIDATE":
            _issue(issues, "SCHEME_A_GSE114002_RECOVERY_ROLE", path, "GSE114002 designed library must remain an A2 source-anchored recovery candidate")
        expected_gse114002 = {
            "qualified": False,
            "training_role": "EXCLUDED_PENDING_QUALIFICATION",
            "true_a2_qualification_status": "AUDIT_PENDING",
            "known_related_sequence_exposure_label": "SEQUENCE_EXPOSED",
            "future_use_boundary_if_qualified": "WITHIN_ASSAY_DEVELOPMENT_AND_OPTIMIZATION_ONLY_SEQUENCE_EXPOSED",
            "fallback_if_designed_library_not_qualifiable": "NEW_GENUINE_PUBLIC_A2_STUDY_REQUIRED",
        }
        for key, value in expected_gse114002.items():
            if gse114002.get(key) != value:
                _issue(issues, "SCHEME_A_GSE114002_BOUNDARY", path, f"GSE114002.{key} must remain {value!r}")

    if "GSE200302" in by_id:
        _issue(
            issues,
            "SCHEME_A_GSE200302_NOT_NEW_STUDY",
            path,
            "GSE200302 is a GSE200304 primary subseries authority, not a new dataset/study row",
        )
    gse200304 = by_id.get("GSE200304")
    if not isinstance(gse200304, Mapping):
        _issue(issues, "SCHEME_A_GSE200304_MISSING", path, "GSE200304 data-role row is required")
    else:
        expected_gse200304 = {
            "role": "AUDIT_ONLY",
            "qualification_status": "AUDIT_PENDING",
            "qualified": False,
            "training_role": "EXCLUDED_PENDING_QUALIFICATION",
            "mapping_status": "SERIES_MEMBER_SOURCE_CANDIDATE_MAPPING_PENDING",
            "primary_subseries_accession": "GSE200302",
            "role_authority_runtime_sync_status": "PENDING_NO_EVT_035",
            "role_authority_changes_dataset_qualification": False,
            "ordinary_gate_contribution": 0,
            "a1_gate_contribution": 0,
            "true_a2_gate_contribution": 0,
            "canonical_record_count": 0,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "role_authority_evidence_status": "PASS",
            "evidence_status": "NOT_RUN",
        }
        for key, value in expected_gse200304.items():
            if not _json_type_strict_equal(gse200304.get(key), value):
                _issue(
                    issues,
                    "SCHEME_A_GSE200304_ROLE_AUTHORITY",
                    path,
                    f"GSE200304.{key} must remain {value!r}",
                )
        official = gse200304.get("official_srr_role_authority")
        if not isinstance(official, Mapping):
            _issue(
                issues,
                "SCHEME_A_GSE200304_ROLE_AUTHORITY",
                path,
                "GSE200304.official_srr_role_authority must be a mapping",
            )
        else:
            _expect_closed_mapping(
                official,
                {
                    "status": "EXACT_OFFICIAL_SRR_ROLE_AUTHORITY_CLOSED",
                    "authority_level": "OFFICIAL_METADATA_ROLE_AUTHORITY_ONLY",
                    "target_series_accession": "GSE200302",
                    "bioproject_accession": "PRJNA824033",
                    "publication_status": "COMMITTED_AND_ACCEPTED",
                    "mapping_row_count": 24,
                    "experiment_join_row_count": 24,
                    "measurement_families": ["High_Poly", "Low_Poly", "pDNA", "Total_RNA"],
                    "replicates": [1, 2, 3, 4, 5, 6],
                    "forbidden_80s_alias_count": 0,
                    "artifact_path": GSE200302_ROLE_ARTIFACT_ROOT,
                    "bundle_digest": GSE200302_ROLE_BUNDLE_DIGEST,
                },
                path,
                issues,
                "SCHEME_A_GSE200304_ROLE_AUTHORITY",
            )
        raw_role = gse200304.get("raw_replay_role_authority")
        if not isinstance(raw_role, Mapping):
            _issue(
                issues,
                "SCHEME_A_GSE200304_ROLE_AUTHORITY",
                path,
                "GSE200304.raw_replay_role_authority must be a mapping",
            )
        else:
            _expect_closed_mapping(
                raw_role,
                {
                    "prior_blocker": "EXACT_SRR_SAMPLE_ROLES_UNKNOWN",
                    "prior_blocker_status": "CLOSED",
                    "replacement_blocker": "REQUIRED_80S_ROLE_AUTHORITY_ABSENT",
                    "replacement_blocker_status": "OPEN",
                    "role_grid_status": "CONFLICT_WITH_CURRENT_80S_EXPECTATION",
                    "pdna_may_substitute_for_80s_rna": False,
                },
                path,
                issues,
                "SCHEME_A_GSE200304_ROLE_AUTHORITY",
            )
        blockers = gse200304.get("blocking_requirements")
        if not isinstance(blockers, list) or "REQUIRED_80S_ROLE_AUTHORITY_ABSENT" in blockers:
            _issue(
                issues,
                "SCHEME_A_GSE200304_ROLE_AUTHORITY",
                path,
                "GSE200304 must preserve REQUIRED_80S_ROLE_AUTHORITY_ABSENT only inside the raw-replay authority; DEC-019 makes it nonblocking for the processed primary route",
            )
    return issues


def validate_registry_closure(
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    issues: list[Issue] = []
    for name, document in registries.items():
        _metadata_ok(document, REGISTRY_PATHS[name], issues, registry_type=REGISTRY_TYPES[name])

    task = registries["task"]
    phase_ids, phase_entries = _check_expected_closure(
        task,
        path=REGISTRY_PATHS["task"],
        expected_key="expected_phase_ids",
        entries_key="phase_tasks",
        id_key="phase_id",
        issues=issues,
        fixed_expected=EXPECTED_PHASE_IDS,
    )
    task_ids, task_entries = _check_expected_closure(
        task,
        path=REGISTRY_PATHS["task"],
        expected_key="expected_task_ids",
        entries_key="tasks",
        id_key="task_id",
        issues=issues,
        fixed_expected=EXPECTED_TASK_IDS,
    )
    data_ids, data_entries = _check_expected_closure(
        registries["data"],
        path=REGISTRY_PATHS["data"],
        expected_key="expected_dataset_ids",
        entries_key="datasets",
        id_key="dataset_id",
        issues=issues,
    )
    baseline_ids, baseline_entries = _check_expected_closure(
        registries["baseline"],
        path=REGISTRY_PATHS["baseline"],
        expected_key="expected_baseline_ids",
        entries_key="baselines",
        id_key="baseline_id",
        issues=issues,
    )
    split_ids, split_entries = _check_expected_closure(
        registries["split"],
        path=REGISTRY_PATHS["split"],
        expected_key="expected_split_ids",
        entries_key="splits",
        id_key="split_id",
        issues=issues,
        fixed_expected=EXPECTED_SPLIT_IDS,
    )
    claim_ids, claim_entries = _check_expected_closure(
        registries["claim"],
        path=REGISTRY_PATHS["claim"],
        expected_key="expected_claim_ids",
        entries_key="claims",
        id_key="claim_id",
        issues=issues,
    )
    del data_ids, baseline_ids, claim_ids  # sets are validated above; names aid review.

    config_phases_raw = config.get("phase_plan")
    config_phases = [item for item in config_phases_raw if isinstance(item, Mapping)] if isinstance(config_phases_raw, list) else []
    issues.extend(validate_phase_dependencies(config_phases, phase_entries))

    for entry in task_entries:
        owner = entry.get("phase_owner")
        if owner not in phase_ids:
            _issue(issues, "TASK_PHASE_FK", REGISTRY_PATHS["task"], f"task {entry.get('task_id')!r} references unknown phase {owner!r}")
    for entry in claim_entries:
        for phase_id in entry.get("required_phase_ids", []):
            if phase_id not in phase_ids:
                _issue(issues, "CLAIM_PHASE_FK", REGISTRY_PATHS["claim"], f"claim {entry.get('claim_id')!r} references unknown phase {phase_id!r}")
        for task_id in entry.get("required_task_ids", []):
            if task_id not in task_ids:
                _issue(issues, "CLAIM_TASK_FK", REGISTRY_PATHS["claim"], f"claim {entry.get('claim_id')!r} references unknown task {task_id!r}")

    matrix_doc = registries["matrix"]
    matrix = matrix_doc.get("matrix")
    if not isinstance(matrix, Mapping):
        _issue(issues, "INVALID_TASK_SPLIT_MATRIX", REGISTRY_PATHS["matrix"], "matrix must be a mapping")
        matrix = {}
    expected_matrix_tasks = matrix_doc.get("expected_task_ids")
    expected_matrix_splits = matrix_doc.get("expected_split_ids")
    if not isinstance(expected_matrix_tasks, list) or set(expected_matrix_tasks) != task_ids or len(expected_matrix_tasks) != len(task_ids):
        _issue(issues, "MATRIX_EXPECTED_TASK_CLOSURE", REGISTRY_PATHS["matrix"], "expected_task_ids must equal task registry IDs")
    if not isinstance(expected_matrix_splits, list) or set(expected_matrix_splits) != split_ids or len(expected_matrix_splits) != len(split_ids):
        _issue(issues, "MATRIX_EXPECTED_SPLIT_CLOSURE", REGISTRY_PATHS["matrix"], "expected_split_ids must equal split registry IDs")
    if set(matrix) != task_ids:
        _issue(issues, "MATRIX_TASK_CLOSURE", REGISTRY_PATHS["matrix"], "matrix row keys must equal task registry IDs")
    for task_id, assigned in matrix.items():
        if not isinstance(assigned, list) or not assigned:
            _issue(issues, "MATRIX_EMPTY_ASSIGNMENT", REGISTRY_PATHS["matrix"], f"task {task_id!r} must have a non-empty split list")
            continue
        unknown = set(assigned) - split_ids
        if unknown:
            _issue(issues, "MATRIX_SPLIT_FK", REGISTRY_PATHS["matrix"], f"task {task_id!r} references unknown splits {sorted(unknown)!r}")
        if len(assigned) != len(set(assigned)):
            _issue(issues, "MATRIX_DUPLICATE_SPLIT", REGISTRY_PATHS["matrix"], f"task {task_id!r} repeats a split")
        if task_id == SEALED_TASK_ID and set(assigned) != {SEALED_SPLIT_ID}:
            _issue(issues, "SEALED_TASK_SPLIT", REGISTRY_PATHS["matrix"], f"sealed task may reference only {SEALED_SPLIT_ID}")
        elif task_id != SEALED_TASK_ID and SEALED_SPLIT_ID in assigned:
            _issue(issues, "ORDINARY_TASK_USES_SEALED", REGISTRY_PATHS["matrix"], f"ordinary task {task_id!r} references {SEALED_SPLIT_ID}")
        if task_id == TOY_TASK_ID and set(assigned) != {TOY_SPLIT_ID}:
            _issue(issues, "TOY_TASK_SPLIT", REGISTRY_PATHS["matrix"], f"toy exact task may reference only {TOY_SPLIT_ID}")

    # A0 definitions must not smuggle a scientific PASS into any registry.
    for path, entries in (
        (REGISTRY_PATHS["task"], [*phase_entries, *task_entries]),
        (REGISTRY_PATHS["data"], data_entries),
        (REGISTRY_PATHS["baseline"], baseline_entries),
        (REGISTRY_PATHS["split"], split_entries),
        (REGISTRY_PATHS["claim"], claim_entries),
    ):
        for entry in entries:
            status = entry.get("evidence_status")
            if status is not None and status not in EVIDENCE_STATUSES:
                _issue(issues, "UNKNOWN_EVIDENCE_STATUS", path, f"{status!r} is outside the frozen vocabulary")
            if status == "PASS":
                _issue(issues, "A0_PREMATURE_SCIENTIFIC_PASS", path, f"entry {entry!r} is marked PASS")
            claim_status = entry.get("claim_status")
            if claim_status is not None and claim_status not in CLAIM_STATUSES:
                _issue(issues, "UNKNOWN_CLAIM_STATUS", path, f"{claim_status!r} is outside the frozen vocabulary")

    issues.extend(validate_scheme_a_data_roles(registries["data"]))

    return issues


def _mapping_entry(entries: Any, id_key: str, wanted: str) -> Mapping[str, Any] | None:
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get(id_key) == wanted:
                return entry
    elif isinstance(entries, Mapping):
        entry = entries.get(wanted)
        if isinstance(entry, Mapping):
            materialized = dict(entry)
            materialized.setdefault(id_key, wanted)
            return materialized
    return None


def _json_type_strict_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON-compatible values without Python's bool/int coercion."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(observed) == set(expected) and all(
            _json_type_strict_equal(observed[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _json_type_strict_equal(left, right)
            for left, right in zip(observed, expected)
        )
    return observed == expected


def _expect(mapping: Mapping[str, Any], key: str, value: Any, path: str, issues: list[Issue], code: str) -> None:
    if key not in mapping or not _json_type_strict_equal(mapping.get(key), value):
        _issue(issues, code, path, f"{key} must be {value!r}; got {mapping.get(key)!r}")


def _expect_closed_mapping(
    mapping: Mapping[str, Any],
    expected: Mapping[str, Any],
    path: str,
    issues: list[Issue],
    code: str,
) -> None:
    observed_keys = set(mapping)
    expected_keys = set(expected)
    if observed_keys != expected_keys:
        _issue(
            issues,
            code,
            path,
            f"mapping keys must be exactly {sorted(expected_keys)!r}; got {sorted(observed_keys)!r}",
        )
    for key, value in expected.items():
        _expect(mapping, key, value, path, issues, code)


def validate_dec019_authority(
    repo_root: Path,
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    """Freeze DEC-019 without granting qualification, training, or a claim."""

    issues: list[Issue] = []
    for relative, expected_sha256 in DEC019_LEAF_AUTHORITY_SHA256.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "DEC019_LEAF_AUTHORITY_UNREADABLE", relative, str(exc))
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "DEC019_LEAF_AUTHORITY_DRIFT",
                relative,
                f"leaf authority hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        amendment = _load_yaml(repo_root, DEC019_AMENDMENT_PATH)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        _issue(issues, "DEC019_AMENDMENT_LOAD", DEC019_AMENDMENT_PATH, str(exc))
        return issues
    try:
        qualification = _load_json(repo_root, A1_QUALIFICATION_CONFIG_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "DEC019_A1_QUALIFICATION_LOAD", A1_QUALIFICATION_CONFIG_PATH, str(exc))
        return issues

    expected_amendment_metadata = {
        "schema_version": "1.0.0",
        "amendment_id": "MRNA_XEDITFLOW_ROUTE_A_V3_DEC019",
        "decision_id": "V3-DEC-019",
        "contract_id": CONTRACT_ID,
        "contract_version": VERSION,
        "amends_contract_path": GOAL_PATH,
        "amends_contract_sha256": SOURCE_CONTRACT_SHA256,
        "amendment_mode": "APPEND_ONLY_AUTHORITY_COMPANION_ROOT_CONTRACT_BYTES_UNCHANGED",
        "status": "FROZEN_USER_AUTHORIZED_A1_QUALIFICATION_POLICY",
        "effective_phase": "A1",
        "requires_user_authorization": True,
    }
    expected_amendment_top_keys = set(expected_amendment_metadata) | {
        "user_authorization",
        "gate_preservation",
        "gse114002_designed_library_true_a2_route",
        "gse200304_published_processed_endpoint_a1_route",
        "uncertainty_and_power_authority",
        "split_freeze_boundary",
        "nonwaivable_authority",
        "historical_preservation",
    }
    if set(amendment) != expected_amendment_top_keys:
        _issue(
            issues,
            "DEC019_AMENDMENT_CLOSURE",
            DEC019_AMENDMENT_PATH,
            f"top-level keys must be exactly {sorted(expected_amendment_top_keys)!r}",
        )
    for key, value in expected_amendment_metadata.items():
        _expect(amendment, key, value, DEC019_AMENDMENT_PATH, issues, "DEC019_AMENDMENT_METADATA")
    for key, expected in (
        (
            "user_authorization",
            {
                "status": "GRANTED",
                "received_at": "2026-08-11T10:42:53+08:00",
                "source": "ACTIVE_CODEX_THREAD_OWNER_CONFIRMATION",
            },
        ),
        (
            "gate_preservation",
            {
                "minimum_independent_ordinary_studies": 3,
                "minimum_qualified_a1_studies": 2,
                "minimum_qualified_true_a2_dense_studies": 1,
                "changes_current_qualified_counts": False,
                "current_qualified_independent_ordinary_studies": 0,
                "current_qualified_a1_studies": 0,
                "current_qualified_true_a2_dense_studies": 0,
                "current_canonical_record_count": 0,
                "phase_complete": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
            },
        ),
        (
            "uncertainty_and_power_authority",
            {
                "analysis_unit": "BIOLOGICAL_SOURCE_GROUP",
                "bootstrap_unit": "BIOLOGICAL_SOURCE_GROUP",
                "target_power_minimum": 0.8,
                "confidence_level": 0.95,
                "full_ci_width_definition": "UPPER_MINUS_LOWER",
                "maximum_full_ci_width": 0.3,
                "thresholds_may_change_after_model_results": False,
                "technical_uncertainty_may_be_relabelled_biological_standard_error": False,
            },
        ),
        (
            "split_freeze_boundary",
            {
                "a1_freeze_required": [
                    "SOURCE_AUTHORITY",
                    "BIOLOGICAL_SOURCE_GROUP_ASSIGNMENT",
                    "NEAR_DUPLICATE_GRAPH",
                    "SPLIT_SALT",
                ],
                "a1_zero_leakage_audit_required": True,
                "a2_freeze_required": ["FINAL_BENCHMARK_MEMBERSHIP"],
                "outcome_or_model_result_may_change_a1_freeze": False,
                "selected_test_may_be_relabelled_untouched": False,
            },
        ),
        (
            "nonwaivable_authority",
            {
                "checkpoint_specific_exposure_may_be_waived": False,
                "license_or_redistribution_rights_may_be_waived": False,
                "accession_level_exposure_is_checkpoint_specific_exposure": False,
                "not_declared_establishes_absence": False,
                "uncertainty_routes_are_dataset_scoped_only": True,
                "global_replicate_or_standard_error_relaxation_allowed": False,
                "gse149487_three_biological_replicates_and_route_a_se_gate_changed": False,
                "other_dataset_specific_stricter_replicate_or_standard_error_gates_changed": False,
            },
        ),
        (
            "historical_preservation",
            {
                "root_contract_bytes_changed": False,
                "accepted_decision_entries_001_through_018_changed": False,
                "historical_gse114002_attempt_blockers_changed": False,
                "historical_gse200304_published_endpoint_blockers_changed": False,
                "prior_runtime_events_changed": False,
                "failure_evidence_deleted_or_relabelled": False,
                "decision_alone_qualifies_any_study": False,
                "decision_alone_materializes_canonical_records": False,
                "decision_alone_authorizes_training_or_model_selection": False,
            },
        ),
    ):
        observed = amendment.get(key)
        if not isinstance(observed, Mapping):
            _issue(issues, "DEC019_AMENDMENT_SEMANTICS", DEC019_AMENDMENT_PATH, f"{key} must be a mapping")
        else:
            _expect_closed_mapping(observed, expected, DEC019_AMENDMENT_PATH, issues, "DEC019_AMENDMENT_SEMANTICS")

    technical_uncertainty = {
        "use_scope": "QC_AND_WITHIN_ASSAY_TECHNICAL_DIAGNOSTIC_ONLY",
        "may_support_biological_standard_error": False,
        "may_support_power": False,
        "may_support_confidence_interval": False,
        "may_support_equivalence": False,
        "may_support_confirmatory_claim": False,
        "may_support_generalization_claim": False,
        "may_substitute_for_biological_standard_error": False,
    }
    expected_gse114002 = {
        "dataset_id": "GSE114002",
        "subset": "DESIGNED_LIBRARY_GSM3130443",
        "current_status": "NOT_QUALIFIED",
        "biological_replicate_status_may_remain": "ABSENT_BY_DESIGN",
        "paper_biological_standard_error_may_remain": None,
        "replicate_and_standard_error_absence_adjudication_scope": "GSE114002_DESIGNED_LIBRARY_ONLY",
        "replicate_and_standard_error_absence_adjudication_may_apply_to_other_datasets": False,
        "conditional_true_a2_eligibility": True,
        "ordinary_study_unit": "GSE114002_ONE_STUDY",
        "maximum_independent_ordinary_study_contribution_if_qualified": 1,
        "gsm_library_pool_or_candidate_may_count_as_independent_study": False,
        "maximum_true_a2_dense_study_contribution_if_qualified": 1,
        "a1_study_contribution_if_qualified": 0,
        "eligibility_scope_if_qualified": ["WITHIN_ASSAY_DEVELOPMENT", "WITHIN_ASSAY_OPTIMIZATION"],
        "confirmatory_contribution": 0,
        "generalization_contribution": 0,
        "technical_fraction_uncertainty": technical_uncertainty,
        "candidate_hamming_distance_eligibility_if_qualified": [1, 2, 3],
        "contract_primary_budget_reporting_if_qualified": [1, 3],
        "global_contract_primary_edit_budgets_retained": [1, 3, 5],
        "k5_role": "CLAIM_BOUNDARY_ONLY_NOT_QUALIFICATION_CREDIT",
        "nonwaivable_gates": [
            "FIELD_AND_BIOLOGICAL_SOURCE_AUTHORITY_CLOSED",
            "FULL_CONSTRUCT_PREFIX_REPORTER_AND_RNA_CHEMISTRY_CLOSED",
            "CHECKPOINT_SPECIFIC_EXPOSURE_CLOSED",
            "LICENSE_AND_REDISTRIBUTION_RIGHTS_CLOSED",
            "A1_SOURCE_GROUP_NEAR_DUPLICATE_GRAPH_AND_SALT_FROZEN",
            "A1_ZERO_LEAKAGE_AUDIT_PASS",
            "PREFROZEN_GROUP_POWER_AT_LEAST_0_80_PASS",
            "PREFROZEN_FULL_CI_WIDTH_AT_MOST_0_30_PASS",
        ],
    }
    observed_gse114002 = amendment.get("gse114002_designed_library_true_a2_route")
    if not isinstance(observed_gse114002, Mapping):
        _issue(issues, "DEC019_GSE114002_ROUTE", DEC019_AMENDMENT_PATH, "GSE114002 route must be a mapping")
    else:
        _expect_closed_mapping(observed_gse114002, expected_gse114002, DEC019_AMENDMENT_PATH, issues, "DEC019_GSE114002_ROUTE")

    expected_gse200304 = {
        "dataset_id": "GSE200304",
        "current_status": "NOT_QUALIFIED",
        "conditional_a1_eligibility": True,
        "ordinary_study_unit": "GSE200304_SUPERSERIES_ONE_STUDY",
        "maximum_independent_ordinary_study_contribution_if_qualified": 1,
        "maximum_a1_study_contribution_if_qualified": 1,
        "maximum_true_a2_dense_study_contribution_if_qualified": 0,
        "superseries_subseries_modality_endpoint_or_replicate_may_count_as_independent_study": False,
        "primary_measurement_route_if_qualified": "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT",
        "raw_replay_role": "REPRODUCIBILITY_AUXILIARY",
        "raw_replay_completion_required_for_primary_a1_measurement_route": False,
        "raw_replay_may_override_published_endpoint_without_adjudication": False,
        "processed_endpoint_engineering_success_is_qualification": False,
        "nonwaivable_gates": [
            "ROW_LEVEL_MULTI_ASSET_LINEAGE_AND_LOCATORS_CLOSED",
            "ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED",
            "BIOLOGICAL_SOURCE_GROUP_AUTHORITY_CLOSED",
            "REPLICATE_OR_VALID_STANDARD_ERROR_CLOSED",
            "LICENSE_AND_REDISTRIBUTION_RIGHTS_CLOSED",
            "CHECKPOINT_SPECIFIC_EXPOSURE_CLOSED",
            "A1_SOURCE_GROUP_NEAR_DUPLICATE_GRAPH_AND_SALT_FROZEN",
            "A1_ZERO_LEAKAGE_AUDIT_PASS",
            "PREFROZEN_GROUP_POWER_AT_LEAST_0_80_PASS",
            "PREFROZEN_FULL_CI_WIDTH_AT_MOST_0_30_PASS",
        ],
    }
    observed_gse200304 = amendment.get("gse200304_published_processed_endpoint_a1_route")
    if not isinstance(observed_gse200304, Mapping):
        _issue(issues, "DEC019_GSE200304_ROUTE", DEC019_AMENDMENT_PATH, "GSE200304 route must be a mapping")
    else:
        _expect_closed_mapping(observed_gse200304, expected_gse200304, DEC019_AMENDMENT_PATH, issues, "DEC019_GSE200304_ROUTE")

    expected_qualification_authority = {
        "contract_sha256": SOURCE_CONTRACT_SHA256,
        "base_commit": "fd722d5fa3c2538fce742b8942b1fb48e782760b",
        "active_amendment_decision_ids": ACTIVE_AMENDMENT_DECISION_IDS,
        "dec019_amendment_path": DEC019_AMENDMENT_PATH,
    }
    observed_q_authority = qualification.get("authority")
    if not isinstance(observed_q_authority, Mapping):
        _issue(issues, "DEC019_A1_QUALIFICATION_AUTHORITY", A1_QUALIFICATION_CONFIG_PATH, "authority must be a mapping")
    else:
        _expect_closed_mapping(observed_q_authority, expected_qualification_authority, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC019_A1_QUALIFICATION_AUTHORITY")

    gate = qualification.get("gate")
    if not isinstance(gate, Mapping):
        _issue(issues, "DEC019_A1_QUALIFICATION_GATE", A1_QUALIFICATION_CONFIG_PATH, "gate must be a mapping")
    else:
        for key, value in {
            "minimum_independent_ordinary_studies": 3,
            "minimum_qualified_a1_studies": 2,
            "minimum_qualified_a2_dense_studies": 1,
            "leakage_must_be_zero": True,
            "maximum_ci_full_width": 0.3,
        }.items():
            _expect(gate, key, value, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC019_A1_QUALIFICATION_GATE")
    power = qualification.get("power_prefreeze")
    if not isinstance(power, Mapping):
        _issue(issues, "DEC019_A1_QUALIFICATION_POWER", A1_QUALIFICATION_CONFIG_PATH, "power_prefreeze must be a mapping")
    else:
        for key, value in {
            "analysis_unit": "BIOLOGICAL_SOURCE_GROUP",
            "bootstrap_unit": "BIOLOGICAL_SOURCE_GROUP",
            "target_power": 0.8,
            "confidence_level": 0.95,
            "maximum_ci_full_width": 0.3,
            "model_results_may_change_this_rule": False,
        }.items():
            _expect(power, key, value, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC019_A1_QUALIFICATION_POWER")

    q_dec019 = qualification.get("dec019_qualification_authority")
    if not isinstance(q_dec019, Mapping):
        _issue(issues, "DEC019_A1_QUALIFICATION_POLICY", A1_QUALIFICATION_CONFIG_PATH, "dec019_qualification_authority must be a mapping")
    else:
        expected_q_keys = {
            "current_qualified_counts",
            "gse114002_designed_library",
            "gse200304_published_processed_endpoint",
            "split_freeze_boundary",
            "checkpoint_specific_exposure_may_be_waived",
            "license_or_redistribution_rights_may_be_waived",
            "uncertainty_routes_are_dataset_scoped_only",
            "global_replicate_or_standard_error_relaxation_allowed",
            "gse149487_three_biological_replicates_and_route_a_se_gate_changed",
            "other_dataset_specific_stricter_replicate_or_standard_error_gates_changed",
            "decision_alone_qualifies_any_study",
            "decision_alone_authorizes_training_or_model_selection",
        }
        if set(q_dec019) != expected_q_keys:
            _issue(issues, "DEC019_A1_QUALIFICATION_POLICY", A1_QUALIFICATION_CONFIG_PATH, f"DEC-019 policy keys must be exactly {sorted(expected_q_keys)!r}")
        _expect(q_dec019, "current_qualified_counts", {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0}, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC019_A1_QUALIFICATION_POLICY")
        q_gse114002 = q_dec019.get("gse114002_designed_library")
        expected_q_gse114002 = {
            key: value
            for key, value in expected_gse114002.items()
            if key not in {"dataset_id", "subset", "a1_study_contribution_if_qualified", "nonwaivable_gates"}
        }
        expected_q_gse114002["allowed_claim_scope_if_qualified"] = expected_q_gse114002.pop("eligibility_scope_if_qualified")
        expected_q_gse114002["all_other_nonwaivable_gates_must_pass"] = True
        if not isinstance(q_gse114002, Mapping):
            _issue(issues, "DEC019_A1_QUALIFICATION_GSE114002", A1_QUALIFICATION_CONFIG_PATH, "GSE114002 policy must be a mapping")
        else:
            _expect_closed_mapping(q_gse114002, expected_q_gse114002, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC019_A1_QUALIFICATION_GSE114002")
        q_gse200304 = q_dec019.get("gse200304_published_processed_endpoint")
        expected_q_gse200304 = {
            key: value
            for key, value in expected_gse200304.items()
            if key not in {"dataset_id", "raw_replay_may_override_published_endpoint_without_adjudication", "processed_endpoint_engineering_success_is_qualification", "nonwaivable_gates"}
        }
        expected_q_gse200304["all_other_nonwaivable_gates_must_pass"] = True
        if not isinstance(q_gse200304, Mapping):
            _issue(issues, "DEC019_A1_QUALIFICATION_GSE200304", A1_QUALIFICATION_CONFIG_PATH, "GSE200304 policy must be a mapping")
        else:
            _expect_closed_mapping(q_gse200304, expected_q_gse200304, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC019_A1_QUALIFICATION_GSE200304")
        _expect(
            q_dec019,
            "split_freeze_boundary",
            {
                "a1_freezes": ["SOURCE_AUTHORITY", "BIOLOGICAL_SOURCE_GROUP_ASSIGNMENT", "NEAR_DUPLICATE_GRAPH", "SPLIT_SALT"],
                "a1_zero_leakage_audit_required": True,
                "a2_freezes": ["FINAL_BENCHMARK_MEMBERSHIP"],
            },
            A1_QUALIFICATION_CONFIG_PATH,
            issues,
            "DEC019_A1_QUALIFICATION_SPLIT",
        )
        for key, value in {
            "checkpoint_specific_exposure_may_be_waived": False,
            "license_or_redistribution_rights_may_be_waived": False,
            "uncertainty_routes_are_dataset_scoped_only": True,
            "global_replicate_or_standard_error_relaxation_allowed": False,
            "gse149487_three_biological_replicates_and_route_a_se_gate_changed": False,
            "other_dataset_specific_stricter_replicate_or_standard_error_gates_changed": False,
            "decision_alone_qualifies_any_study": False,
            "decision_alone_authorizes_training_or_model_selection": False,
        }.items():
            _expect(q_dec019, key, value, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC019_A1_QUALIFICATION_POLICY")

    study_rules = qualification.get("study_rules")
    if isinstance(study_rules, Mapping):
        _expect(
            study_rules,
            "GSE145046",
            {
                "intended_grade": "ABSOLUTE_AUXILIARY_ONLY_NOT_A1_OR_A2",
                "region": "5UTR",
                "required_recovery": "PAPER_FAITHFUL_ENDPOINT_CONTEXT_RIGHTS_EXPOSURE_AND_AUXILIARY_SPLIT_CLOSURE",
                "legacy_single_input_record_may_qualify": False,
                "ordinary_study_contribution_if_qualified": 0,
                "a1_study_contribution_if_qualified": 0,
                "true_a2_dense_study_contribution_if_qualified": 0,
            },
            A1_QUALIFICATION_CONFIG_PATH,
            issues,
            "DEC019_A1_QUALIFICATION_GSE145046",
        )
    else:
        _issue(issues, "DEC019_A1_QUALIFICATION_STUDIES", A1_QUALIFICATION_CONFIG_PATH, "study_rules must be a mapping")

    ordinary = config.get("ordinary_data_minimum")
    if not isinstance(ordinary, Mapping):
        _issue(issues, "DEC019_ROOT_POWER_AUTHORITY", CONFIG_PATH, "ordinary_data_minimum must be a mapping")
    else:
        for key, value in {
            "independent_nonsealed_studies": 3,
            "a1_source_candidate_intervention_studies": 2,
            "a2_dense_candidate_neighborhood_studies": 1,
            "power_target": 0.8,
            "confidence_level": 0.95,
            "full_ci_width_definition": "UPPER_MINUS_LOWER",
            "maximum_full_ci_width": 0.3,
        }.items():
            _expect(ordinary, key, value, CONFIG_PATH, issues, "DEC019_ROOT_POWER_AUTHORITY")
    root_policy = config.get("a1_qualification_authority")
    if not isinstance(root_policy, Mapping):
        _issue(issues, "DEC019_ROOT_POLICY", CONFIG_PATH, "a1_qualification_authority must be a mapping")
    else:
        for key, value in {
            "current_qualified_counts": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
            "checkpoint_specific_exposure_may_be_waived": False,
            "license_or_redistribution_rights_may_be_waived": False,
            "uncertainty_routes_are_dataset_scoped_only": True,
            "global_replicate_or_standard_error_relaxation_allowed": False,
            "gse149487_three_biological_replicates_and_route_a_se_gate_changed": False,
            "other_dataset_specific_stricter_replicate_or_standard_error_gates_changed": False,
            "decision_alone_qualifies_any_study": False,
            "decision_alone_authorizes_training_or_model_selection": False,
        }.items():
            _expect(root_policy, key, value, CONFIG_PATH, issues, "DEC019_ROOT_POLICY")

    for name in ("data", "split", "task", "matrix", "claim"):
        ref = registries[name].get("authority_ref")
        if not isinstance(ref, Mapping):
            _issue(issues, "DEC019_REGISTRY_AUTHORITY", REGISTRY_PATHS[name], "authority_ref must be a mapping")
            continue
        _expect(ref, "active_amendment_decision_ids", ACTIVE_AMENDMENT_DECISION_IDS, REGISTRY_PATHS[name], issues, "DEC019_REGISTRY_AUTHORITY")
        _expect(ref, "dec019_amendment_path", DEC019_AMENDMENT_PATH, REGISTRY_PATHS[name], issues, "DEC019_REGISTRY_AUTHORITY")

    data = registries["data"]
    common_requirements = data.get("common_audit_requirements")
    if not isinstance(common_requirements, list) or "REPLICATE_AND_STANDARD_ERROR_OR_DATASET_SCOPED_ABSENCE_ADJUDICATION" not in common_requirements or "REPLICATE_AND_STANDARD_ERROR" in common_requirements:
        _issue(issues, "DEC019_DATASET_SCOPED_UNCERTAINTY", REGISTRY_PATHS["data"], "replicate/SE absence adjudication must remain dataset-scoped and may not become a global OR relaxation")
    data_rows = data.get("datasets")
    by_id = {
        row.get("dataset_id"): row
        for row in data_rows
        if isinstance(data_rows, list) and isinstance(row, Mapping) and isinstance(row.get("dataset_id"), str)
    } if isinstance(data_rows, list) else {}
    data_gse114002 = by_id.get("GSE114002")
    if not isinstance(data_gse114002, Mapping):
        _issue(issues, "DEC019_DATA_ROLE_GSE114002", REGISTRY_PATHS["data"], "GSE114002 row is required")
    else:
        _expect(data_gse114002, "dec019_owner_uncertainty_policy_status", "CLOSED_BY_V3_DEC_019", REGISTRY_PATHS["data"], issues, "DEC019_DATA_ROLE_GSE114002")
        _expect(data_gse114002, "current_dec019_blocker_count", 6, REGISTRY_PATHS["data"], issues, "DEC019_DATA_ROLE_GSE114002")
        route = data_gse114002.get("dec019_conditional_true_a2_route")
        if not isinstance(route, Mapping):
            _issue(issues, "DEC019_DATA_ROLE_GSE114002", REGISTRY_PATHS["data"], "conditional true-A2 route is required")
        else:
            for key, value in {
                "maximum_independent_ordinary_study_contribution_if_qualified": 1,
                "maximum_true_a2_dense_study_contribution_if_qualified": 1,
                "gsm_library_pool_or_candidate_may_count_as_independent_study": False,
                "candidate_hamming_distance_eligibility_if_qualified": [1, 2, 3],
                "contract_primary_budget_reporting_if_qualified": [1, 3],
                "global_contract_primary_edit_budgets_retained": [1, 3, 5],
                "k5_role": "CLAIM_BOUNDARY_ONLY_NOT_QUALIFICATION_CREDIT",
                "confirmatory_contribution": 0,
                "generalization_contribution": 0,
                "exposure_or_rights_waiver_allowed": False,
            }.items():
                _expect(route, key, value, REGISTRY_PATHS["data"], issues, "DEC019_DATA_ROLE_GSE114002")
    data_gse200304 = by_id.get("GSE200304")
    if not isinstance(data_gse200304, Mapping):
        _issue(issues, "DEC019_DATA_ROLE_GSE200304", REGISTRY_PATHS["data"], "GSE200304 row is required")
    else:
        route = data_gse200304.get("dec019_primary_measurement_route")
        if not isinstance(route, Mapping):
            _issue(issues, "DEC019_DATA_ROLE_GSE200304", REGISTRY_PATHS["data"], "processed-endpoint route is required")
        else:
            for key, value in {
                "owner_primary_route_policy_status": "CLOSED_BY_V3_DEC_019",
                "maximum_independent_ordinary_study_contribution_if_qualified": 1,
                "maximum_a1_study_contribution_if_qualified": 1,
                "maximum_true_a2_dense_study_contribution_if_qualified": 0,
                "superseries_subseries_modality_endpoint_or_replicate_may_count_as_independent_study": False,
                "primary_measurement_route_if_qualified": "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT",
                "raw_replay_role": "REPRODUCIBILITY_AUXILIARY",
                "required_80s_role_authority_is_primary_route_blocker": False,
                "checkpoint_specific_exposure_or_rights_waiver_allowed": False,
                "current_blocker_count": 8,
            }.items():
                _expect(route, key, value, REGISTRY_PATHS["data"], issues, "DEC019_DATA_ROLE_GSE200304")
        blockers = data_gse200304.get("blocking_requirements")
        if not isinstance(blockers, list) or "REQUIRED_80S_ROLE_AUTHORITY_ABSENT" in blockers:
            _issue(issues, "DEC019_GSE200304_RAW_REPLAY_AUXILIARY", REGISTRY_PATHS["data"], "80S remains a raw-replay blocker but may not block the processed primary route")

    split_rules = registries["split"].get("global_split_rules")
    if not isinstance(split_rules, Mapping):
        _issue(issues, "DEC019_SPLIT_FREEZE", REGISTRY_PATHS["split"], "global_split_rules must be a mapping")
    else:
        for key, value in {
            "freeze_phase": "STAGED_A1_QUALIFICATION_A2_FINAL_MEMBERSHIP",
            "a1_qualification_freeze_phase": "A1",
            "a1_qualification_freeze_required": ["SOURCE_AUTHORITY", "BIOLOGICAL_SOURCE_GROUP_ASSIGNMENT", "NEAR_DUPLICATE_GRAPH", "SPLIT_SALT"],
            "a1_zero_leakage_audit_required": True,
            "a1_current_freeze_status": "NOT_FROZEN",
            "a1_current_zero_leakage_audit_status": "NOT_RUN",
            "a2_final_benchmark_membership_freeze_phase": "A2",
            "a2_current_final_benchmark_membership_status": "NOT_FROZEN",
        }.items():
            _expect(split_rules, key, value, REGISTRY_PATHS["split"], issues, "DEC019_SPLIT_FREEZE")

    task_boundary = registries["task"].get("dec019_task_boundaries")
    if not isinstance(task_boundary, Mapping):
        _issue(issues, "DEC019_TASK_BOUNDARY", REGISTRY_PATHS["task"], "dec019_task_boundaries must be a mapping")
    else:
        for key, value in {
            "checkpoint_specific_exposure_may_be_waived": False,
            "license_or_redistribution_rights_may_be_waived": False,
            "uncertainty_routes_are_dataset_scoped_only": True,
            "global_replicate_or_standard_error_relaxation_allowed": False,
        }.items():
            _expect(task_boundary, key, value, REGISTRY_PATHS["task"], issues, "DEC019_TASK_BOUNDARY")

    matrix_boundary = registries["matrix"].get("dec019_split_freeze_boundary")
    if not isinstance(matrix_boundary, Mapping):
        _issue(issues, "DEC019_TASK_SPLIT_BOUNDARY", REGISTRY_PATHS["matrix"], "dec019 split boundary must be a mapping")
    else:
        _expect_closed_mapping(
            matrix_boundary,
            {
                "a1_freezes": ["SOURCE_AUTHORITY", "BIOLOGICAL_SOURCE_GROUP_ASSIGNMENT", "NEAR_DUPLICATE_GRAPH", "SPLIT_SALT"],
                "a1_zero_leakage_audit_required": True,
                "a2_freezes": ["FINAL_BENCHMARK_MEMBERSHIP"],
                "current_a1_freeze_status": "NOT_FROZEN",
                "current_a1_zero_leakage_audit_status": "NOT_RUN",
                "current_a2_final_membership_status": "NOT_FROZEN",
            },
            REGISTRY_PATHS["matrix"],
            issues,
            "DEC019_TASK_SPLIT_BOUNDARY",
        )

    claim_boundary = registries["claim"].get("dec019_claim_boundaries")
    if not isinstance(claim_boundary, Mapping):
        _issue(issues, "DEC019_CLAIM_BOUNDARY", REGISTRY_PATHS["claim"], "dec019_claim_boundaries must be a mapping")
    else:
        for key, value in {
            "checkpoint_specific_exposure_may_be_waived": False,
            "license_or_redistribution_rights_may_be_waived": False,
            "uncertainty_routes_are_dataset_scoped_only": True,
            "global_replicate_or_standard_error_relaxation_allowed": False,
            "decision_alone_establishes_any_claim": False,
        }.items():
            _expect(claim_boundary, key, value, REGISTRY_PATHS["claim"], issues, "DEC019_CLAIM_BOUNDARY")

    return issues


def _dec019_successor_core_sha256(config: Mapping[str, Any]) -> str:
    """Hash the closed successor config after excluding its dynamic I/B binding."""

    projection = dict(config)
    projection.pop("implementation_binding", None)
    payload = (
        json.dumps(
            projection,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    return sha256_bytes(payload)


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _gse200304_dec019_v3_config_core_sha256(config: Mapping[str, Any]) -> str:
    """Derive the v3 science core while excluding I/B and descriptor values."""

    projection = copy.deepcopy(dict(config))
    projection.pop("implementation_binding", None)
    descriptors = projection.get("evidence_descriptor_bindings")
    if type(descriptors) is dict:
        slots = descriptors.get("slots")
        projection["evidence_descriptor_bindings"] = {
            "binding_scheme": descriptors.get("binding_scheme"),
            "dynamic_scalar_suffixes": descriptors.get("dynamic_scalar_suffixes"),
            "all_descriptors_required_before_any_input_open": descriptors.get(
                "all_descriptors_required_before_any_input_open"
            ),
            "slots": [
                {"slot_id": slot.get("slot_id")}
                for slot in slots
            ]
            if type(slots) is list
            else slots,
        }
    payload = (
        json.dumps(
            projection,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    return sha256_bytes(payload)


def _gse200304_dec019_v3_descriptor_set_sha256(
    config: Mapping[str, Any],
) -> str:
    descriptors = copy.deepcopy(dict(config["evidence_descriptor_bindings"]))
    descriptors.pop("descriptor_set_sha256", None)
    payload = (
        json.dumps(
            descriptors,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    return sha256_bytes(payload)


def validate_gse200304_dec019_post_adjudication_registration(
    repo_root: Path,
) -> list[Issue]:
    """Bind the stable v3 leaves and dynamic D2 config without a manifest cycle."""

    issues: list[Issue] = []
    for relative, expected_sha256 in (
        GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256.items()
    ):
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(
                issues,
                "GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF",
                relative,
                str(exc),
            )
        else:
            if actual_sha256 != expected_sha256:
                _issue(
                    issues,
                    "GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF",
                    relative,
                    f"current bytes hash {actual_sha256} must remain {expected_sha256}",
                )

    try:
        config_bytes = _read_bytes(repo_root, GSE200304_DEC019_V3_CONFIG_PATH)
        config = _load_json(repo_root, GSE200304_DEC019_V3_CONFIG_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "GSE200304_DEC019_V3_DYNAMIC_CONFIG",
            GSE200304_DEC019_V3_CONFIG_PATH,
            str(exc),
        )
        return issues

    actual_config_sha256 = sha256_bytes(config_bytes)
    if actual_config_sha256 != GSE200304_DEC019_V3_CONFIG_SHA256:
        _issue(
            issues,
            "GSE200304_DEC019_V3_DYNAMIC_CONFIG",
            GSE200304_DEC019_V3_CONFIG_PATH,
            f"current full SHA {actual_config_sha256} must remain {GSE200304_DEC019_V3_CONFIG_SHA256}",
        )
    try:
        actual_core_sha256 = _gse200304_dec019_v3_config_core_sha256(config)
        actual_descriptor_sha256 = _gse200304_dec019_v3_descriptor_set_sha256(
            config
        )
    except (KeyError, TypeError, ValueError) as exc:
        _issue(
            issues,
            "GSE200304_DEC019_V3_DYNAMIC_CONFIG",
            GSE200304_DEC019_V3_CONFIG_PATH,
            f"cannot derive core/descriptor projections: {exc}",
        )
        return issues

    binding = config.get("implementation_binding")
    descriptors = config.get("evidence_descriptor_bindings")
    expected_binding_fields = {
        "status": "BOUND",
        "implementation_commit": "6d103877bbfb8e1196bfc22890bb239dcb87c3c8",
        "implementation_script_path": GSE200304_DEC019_V3_SCRIPT_PATH,
        "implementation_script_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
            GSE200304_DEC019_V3_SCRIPT_PATH
        ],
        "implementation_test_path": GSE200304_DEC019_V3_TEST_PATH,
        "implementation_test_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
            GSE200304_DEC019_V3_TEST_PATH
        ],
        "config_core_sha256": GSE200304_DEC019_V3_CONFIG_CORE_SHA256,
    }
    if not isinstance(binding, Mapping):
        _issue(
            issues,
            "GSE200304_DEC019_V3_CORE_BINDING",
            GSE200304_DEC019_V3_CONFIG_PATH,
            "implementation_binding must be a mapping",
        )
    else:
        for key, value in expected_binding_fields.items():
            _expect(
                binding,
                key,
                value,
                GSE200304_DEC019_V3_CONFIG_PATH,
                issues,
                "GSE200304_DEC019_V3_CORE_BINDING",
            )
    if actual_core_sha256 != GSE200304_DEC019_V3_CONFIG_CORE_SHA256:
        _issue(
            issues,
            "GSE200304_DEC019_V3_CORE_BINDING",
            GSE200304_DEC019_V3_CONFIG_PATH,
            f"derived core {actual_core_sha256} must remain {GSE200304_DEC019_V3_CONFIG_CORE_SHA256}",
        )
    if not isinstance(descriptors, Mapping):
        _issue(
            issues,
            "GSE200304_DEC019_V3_DESCRIPTOR_BINDING",
            GSE200304_DEC019_V3_CONFIG_PATH,
            "evidence_descriptor_bindings must be a mapping",
        )
    else:
        _expect(
            descriptors,
            "status",
            "BOUND",
            GSE200304_DEC019_V3_CONFIG_PATH,
            issues,
            "GSE200304_DEC019_V3_DESCRIPTOR_BINDING",
        )
        _expect(
            descriptors,
            "descriptor_set_sha256",
            GSE200304_DEC019_V3_DESCRIPTOR_SET_SHA256,
            GSE200304_DEC019_V3_CONFIG_PATH,
            issues,
            "GSE200304_DEC019_V3_DESCRIPTOR_BINDING",
        )
    if actual_descriptor_sha256 != GSE200304_DEC019_V3_DESCRIPTOR_SET_SHA256:
        _issue(
            issues,
            "GSE200304_DEC019_V3_DESCRIPTOR_BINDING",
            GSE200304_DEC019_V3_CONFIG_PATH,
            f"derived descriptor set {actual_descriptor_sha256} must remain {GSE200304_DEC019_V3_DESCRIPTOR_SET_SHA256}",
        )

    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "GSE200304_DEC019_POST_ADJUDICATION_DAG",
            REGISTRY_MANIFEST_PATH,
            str(exc),
        )
    else:
        entries = manifest.get("files")
        manifest_paths = {
            entry.get("path")
            for entry in entries
            if type(entries) is list and type(entry) is dict
        } if type(entries) is list else set()
        expected_static_paths = set(
            GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256
        )
        if GSE200304_DEC019_V3_CONFIG_PATH in manifest_paths:
            _issue(
                issues,
                "GSE200304_DEC019_POST_ADJUDICATION_DAG",
                REGISTRY_MANIFEST_PATH,
                "dynamic D2 config must remain outside the static manifest",
            )
        if not expected_static_paths.issubset(manifest_paths):
            _issue(
                issues,
                "GSE200304_DEC019_POST_ADJUDICATION_DAG",
                REGISTRY_MANIFEST_PATH,
                "all eight stable post-adjudication leaves must be statically registered",
            )
    return issues


def validate_dec019_successor_adjudicators(repo_root: Path) -> list[Issue]:
    """Validate stable scientific cores without creating an I-to-B hash cycle.

    The two config files are required public inputs, but their full hashes are
    intentionally absent from the static registry manifest.  Only the complete
    implementation-binding object may move from the initial UNKNOWN state to a
    later config-only BOUND state.  Every other config field is protected by a
    frozen core-projection SHA, while the implementation and focused-test bytes
    remain exact-hashed static leaves.
    """

    issues: list[Issue] = []
    specifications = (
        {
            "dataset_id": "GSE114002",
            "config_path": GSE114002_DEC019_SUCCESSOR_CONFIG_PATH,
            "script_path": GSE114002_DEC019_SUCCESSOR_SCRIPT_PATH,
            "test_path": GSE114002_DEC019_SUCCESSOR_TEST_PATH,
            "initial_i_sha256": GSE114002_DEC019_SUCCESSOR_INITIAL_I_SHA256,
            "core_sha256": GSE114002_DEC019_SUCCESSOR_CORE_SHA256,
            "script_sha256": GSE114002_DEC019_SUCCESSOR_SCRIPT_SHA256,
            "test_sha256": GSE114002_DEC019_SUCCESSOR_TEST_SHA256,
            "schema_version": "route_a_v3_gse114002_dec019_true_a2_activation.v2",
            "protocol_id": "ROUTE_A_V3_GSE114002_DEC019_TRUE_A2_ACTIVATION_V2",
            "blocked_status": "BLOCKED_DEC019_TRUE_A2_EVIDENCE_INCOMPLETE",
            "current_state": {
                "qualified": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "canonical_record_count": 0,
                "canonical_materialization_allowed": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "unresolved_blockers": [
                    "SOURCE_FIELD_AUTHORITY_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "CONSTRUCT_RNA_CHEMISTRY_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "CHECKPOINT_SPECIFIC_EXPOSURE_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "LICENSE_RIGHTS_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "OUTCOME_BLIND_SPLIT_LEAKAGE_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "PREFROZEN_POWER_PRECISION_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                ],
            },
        },
        {
            "dataset_id": "GSE200304",
            "config_path": GSE200304_DEC019_SUCCESSOR_CONFIG_PATH,
            "script_path": GSE200304_DEC019_SUCCESSOR_SCRIPT_PATH,
            "test_path": GSE200304_DEC019_SUCCESSOR_TEST_PATH,
            "initial_i_sha256": GSE200304_DEC019_SUCCESSOR_INITIAL_I_SHA256,
            "core_sha256": GSE200304_DEC019_SUCCESSOR_CORE_SHA256,
            "script_sha256": GSE200304_DEC019_SUCCESSOR_SCRIPT_SHA256,
            "test_sha256": GSE200304_DEC019_SUCCESSOR_TEST_SHA256,
            "schema_version": "route_a_v3_gse200304_dec019_reported_endpoint_a1_activation.v2",
            "protocol_id": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ACTIVATION_V2",
            "blocked_status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
            "current_state": {
                "qualified": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "canonical_record_count": 0,
                "canonical_materialization_allowed": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "unresolved_blockers": [
                    "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "CANONICAL_REPORTED_ENDPOINT_SEMANTICS_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "BIOLOGICAL_GROUP_AUTHORITY_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "ROW_REPLICATE_OR_VALID_SE_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "CHECKPOINT_SPECIFIC_EXPOSURE_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "LICENSE_RIGHTS_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "OUTCOME_BLIND_SPLIT_LEAKAGE_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                    "PREFROZEN_POWER_PRECISION_EVIDENCE_UNKNOWN_NOT_ASSERTED",
                ],
            },
        },
    )
    expected_top_keys = {
        "schema_version",
        "protocol_id",
        "contract_id",
        "phase_id",
        "dataset_id",
        "decision_id",
        "implementation_binding",
        "repository_authority",
        "core_authority",
        "policy_boundary",
        "current_external_state",
        "evidence_contract",
        "output_contract",
    }
    expected_binding_keys = {
        "binding_scheme",
        "status",
        "blocker_if_unbound",
        "implementation_commit",
        "implementation_script_path",
        "implementation_script_sha256",
        "implementation_test_path",
        "implementation_test_sha256",
        "config_core_sha256",
        "unknown_to_bound_scalar_paths",
    }
    expected_dynamic_paths = [
        "implementation_binding.status",
        "implementation_binding.implementation_commit",
        "implementation_binding.implementation_script_sha256",
        "implementation_binding.implementation_test_sha256",
    ]

    for spec in specifications:
        config_path = str(spec["config_path"])
        try:
            payload = _read_bytes(repo_root, config_path)

            def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON key {key!r}")
                    result[key] = value
                return result

            config = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {token!r}")
                ),
            )
            if type(config) is not dict:
                raise ValueError("successor config root must be a JSON object")
        except (FileNotFoundError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            _issue(issues, "DEC019_SUCCESSOR_CONFIG_UNREADABLE", config_path, str(exc))
            continue

        if set(config) != expected_top_keys:
            _issue(
                issues,
                "DEC019_SUCCESSOR_CONFIG_CLOSURE",
                config_path,
                f"top-level keys must be exactly {sorted(expected_top_keys)!r}",
            )
        for key, value in {
            "schema_version": spec["schema_version"],
            "protocol_id": spec["protocol_id"],
            "contract_id": CONTRACT_ID,
            "phase_id": "A1",
            "dataset_id": spec["dataset_id"],
            "decision_id": "V3-DEC-019",
        }.items():
            _expect(config, key, value, config_path, issues, "DEC019_SUCCESSOR_CONFIG_METADATA")

        try:
            observed_core_sha256 = _dec019_successor_core_sha256(config)
        except (TypeError, ValueError) as exc:
            _issue(issues, "DEC019_SUCCESSOR_CORE_UNHASHABLE", config_path, str(exc))
            observed_core_sha256 = None
        if observed_core_sha256 != spec["core_sha256"]:
            _issue(
                issues,
                "DEC019_SUCCESSOR_CORE_DRIFT",
                config_path,
                f"stable core {observed_core_sha256} must remain {spec['core_sha256']}",
            )

        binding = config.get("implementation_binding")
        if type(binding) is not dict or set(binding) != expected_binding_keys:
            _issue(
                issues,
                "DEC019_SUCCESSOR_BINDING_CLOSURE",
                config_path,
                f"implementation_binding keys must be exactly {sorted(expected_binding_keys)!r}",
            )
            continue
        for key, value in {
            "binding_scheme": "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
            "blocker_if_unbound": "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED",
            "implementation_script_path": spec["script_path"],
            "implementation_test_path": spec["test_path"],
            "config_core_sha256": spec["core_sha256"],
            "unknown_to_bound_scalar_paths": expected_dynamic_paths,
        }.items():
            _expect(binding, key, value, config_path, issues, "DEC019_SUCCESSOR_BINDING")

        status = binding.get("status")
        if status == "UNKNOWN_NOT_ASSERTED":
            for key in (
                "implementation_commit",
                "implementation_script_sha256",
                "implementation_test_sha256",
            ):
                _expect(
                    binding,
                    key,
                    "UNKNOWN_NOT_ASSERTED",
                    config_path,
                    issues,
                    "DEC019_SUCCESSOR_BINDING",
                )
            actual_i_sha256 = sha256_bytes(payload)
            if actual_i_sha256 != spec["initial_i_sha256"]:
                _issue(
                    issues,
                    "DEC019_SUCCESSOR_INITIAL_I_DRIFT",
                    config_path,
                    f"initial UNKNOWN config hash {actual_i_sha256} must remain {spec['initial_i_sha256']}",
                )
        elif status == "BOUND":
            if not _is_lower_hex(binding.get("implementation_commit"), 40):
                _issue(
                    issues,
                    "DEC019_SUCCESSOR_BINDING",
                    config_path,
                    "BOUND implementation_commit must be exactly 40 lowercase hex characters",
                )
            _expect(
                binding,
                "implementation_script_sha256",
                spec["script_sha256"],
                config_path,
                issues,
                "DEC019_SUCCESSOR_BINDING",
            )
            _expect(
                binding,
                "implementation_test_sha256",
                spec["test_sha256"],
                config_path,
                issues,
                "DEC019_SUCCESSOR_BINDING",
            )
        else:
            _issue(
                issues,
                "DEC019_SUCCESSOR_BINDING",
                config_path,
                "implementation binding status must be UNKNOWN_NOT_ASSERTED or BOUND",
            )

        current = config.get("current_external_state")
        if not isinstance(current, Mapping):
            _issue(
                issues,
                "DEC019_SUCCESSOR_CURRENT_STATE",
                config_path,
                "current_external_state must be a mapping",
            )
        else:
            _expect(
                current,
                "status",
                spec["blocked_status"],
                config_path,
                issues,
                "DEC019_SUCCESSOR_CURRENT_STATE",
            )
            for key, value in spec["current_state"].items():
                _expect(
                    current,
                    key,
                    value,
                    config_path,
                    issues,
                    "DEC019_SUCCESSOR_CURRENT_STATE",
                )

        for path_key, hash_key in (("script_path", "script_sha256"), ("test_path", "test_sha256")):
            relative = str(spec[path_key])
            try:
                actual = sha256_bytes(_read_bytes(repo_root, relative))
            except (FileNotFoundError, ValueError) as exc:
                _issue(issues, "DEC019_SUCCESSOR_STATIC_LEAF_UNREADABLE", relative, str(exc))
                continue
            if actual != spec[hash_key]:
                _issue(
                    issues,
                    "DEC019_SUCCESSOR_STATIC_LEAF_DRIFT",
                    relative,
                    f"static leaf {actual} must remain {spec[hash_key]}",
                )

    return issues


def _gse149487_plumage_nonbinding_core_sha256(protocol: Mapping[str, Any]) -> str:
    """Hash the full protocol after normalizing only the three permitted I/B scalars."""

    normalized = json.loads(json.dumps(protocol))
    authority = normalized.get("authority")
    binding = normalized.get("stop_before_data_preflight_binding")
    if not isinstance(authority, dict) or not isinstance(binding, dict):
        raise ValueError("protocol lacks the two implementation binding objects")
    authority["implementation_commit"] = "UNKNOWN_NOT_ASSERTED"
    binding["status"] = "UNKNOWN_NOT_ASSERTED"
    binding["implementation_commit"] = "UNKNOWN_NOT_ASSERTED"
    return sha256_bytes(
        json.dumps(
            normalized,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def validate_gse149487_plumage_protocol(repo_root: Path) -> list[Issue]:
    """Freeze the PLUMAGE pre-data boundary without hashing its dynamic protocol."""

    issues: list[Issue] = []
    path = GSE149487_PLUMAGE_PROTOCOL_PATH

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        protocol = json.loads(
            _read_bytes(repo_root, path).decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (FileNotFoundError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        _issue(
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_LOAD",
            path,
            f"protocol must be duplicate-free finite UTF-8 JSON: {exc}",
        )
        return issues
    if type(protocol) is not dict:
        _issue(
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_SHAPE",
            path,
            "protocol root must be an object",
        )
        return issues

    expected_top_keys = {
        "contract_id",
        "schema_version",
        "protocol_id",
        "protocol_status",
        "dataset_id",
        "dataset_alias",
        "study_group_id",
        "independent_study_count",
        "authority",
        "scope",
        "input_contract",
        "mapping",
        "paper_faithful_measurement_transform",
        "route_a_companion_summary",
        "canonical_v3",
        "license_and_redistribution",
        "foundation_exposure",
        "split_and_leakage",
        "power_prefreeze",
        "stop_before_data_preflight_binding",
        "current_gate_contract",
        "qualification_gates",
        "output_contract",
        "known_external_evidence_blockers",
        "model_results_may_change_this_protocol",
    }
    if set(protocol) != expected_top_keys:
        _issue(
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_SHAPE",
            path,
            f"top-level keys must be exactly {sorted(expected_top_keys)!r}",
        )
    expected_metadata = {
        "contract_id": CONTRACT_ID,
        "schema_version": VERSION,
        "protocol_id": "ROUTE_A_V3_GSE149487_PLUMAGE_FULL_A1_QUALIFICATION_V1",
        "protocol_status": "PREFROZEN_FAIL_CLOSED_BEFORE_FULL_RAW_JOIN_RESULTS",
        "dataset_id": "GSE149487",
        "dataset_alias": "PLUMAGE",
        "study_group_id": "PLUMAGE_LIM_2021",
        "independent_study_count": 1,
        "model_results_may_change_this_protocol": False,
    }
    for key, value in expected_metadata.items():
        _expect(
            protocol,
            key,
            value,
            path,
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_METADATA",
        )

    authority = protocol.get("authority")
    expected_authority = {
        "contract_path": GOAL_PATH,
        "initial_contract_sha256": "d1c031aecdec710495f6861b380785cccd64663ac4bd97b4f479d6fdf372ea07",
        "contract_sha256": SOURCE_CONTRACT_SHA256,
        "accepted_a0_base_commit": "fd722d5fa3c2538fce742b8942b1fb48e782760b",
        "active_authority_commit": GSE149487_PLUMAGE_ACTIVE_AUTHORITY_COMMIT,
        "active_amendment_decision_ids": ["V3-DEC-017", "V3-DEC-018"],
        "a1_qualification_path": "configs/route_a_v3_a1_qualification.json",
        "a1_qualification_sha256": "1d348671de50c0fe8b155f8cc114d14a74360fe1a87f9d9bac5207ae794806c4",
        "data_role_registry_path": REGISTRY_PATHS["data"],
        "data_role_registry_sha256": "746439ef5d88d8167176d19e9c675746fdc78984a66f6f123f77f6ec49523030",
        "decision_log_path": DECISION_LOG_PATH,
        "decision_log_sha256": "a5b041fab24d9a4309603a085fa3fcab936d69a899285bfa752689a2ee5fd4fd",
        "canonical_schema_path": "schemas/route_a_v3/canonical_intervention_record.schema.json",
        "canonical_schema_sha256": "5dc384d6c5714fb5834e83d8fafb51f712bbfcf7dfb632ad504f051b985af898",
        "asset_manifest_path": GSE149487_PLUMAGE_ASSET_MANIFEST_PATH,
        "asset_manifest_sha256": GSE149487_PLUMAGE_ASSET_MANIFEST_SHA256,
        "v4_helper_path": GSE149487_PLUMAGE_HELPER_PATH,
        "v4_helper_sha256": GSE149487_PLUMAGE_HELPER_SHA256,
        "qualifier_path": GSE149487_PLUMAGE_QUALIFIER_PATH,
        "qualifier_sha256": GSE149487_PLUMAGE_QUALIFIER_SHA256,
        "focused_test_path": GSE149487_PLUMAGE_TEST_PATH,
        "focused_test_sha256": GSE149487_PLUMAGE_TEST_SHA256,
    }
    expected_authority_keys = set(expected_authority) | {"implementation_commit"}
    if not isinstance(authority, Mapping) or set(authority) != expected_authority_keys:
        _issue(
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_AUTHORITY",
            path,
            f"authority keys must be exactly {sorted(expected_authority_keys)!r}",
        )
    else:
        for key, value in expected_authority.items():
            _expect(
                authority,
                key,
                value,
                path,
                issues,
                "GSE149487_PLUMAGE_PROTOCOL_AUTHORITY",
            )
        implementation = authority.get("implementation_commit")
        if implementation != "UNKNOWN_NOT_ASSERTED" and not (
            isinstance(implementation, str)
            and len(implementation) == 40
            and all(ch in "0123456789abcdef" for ch in implementation)
        ):
            _issue(
                issues,
                "GSE149487_PLUMAGE_PROTOCOL_AUTHORITY",
                path,
                "implementation_commit must be UNKNOWN_NOT_ASSERTED or a full lowercase object ID",
            )

        for path_key, hash_key in (
            ("contract_path", "contract_sha256"),
            ("a1_qualification_path", "a1_qualification_sha256"),
            ("data_role_registry_path", "data_role_registry_sha256"),
            ("decision_log_path", "decision_log_sha256"),
            ("canonical_schema_path", "canonical_schema_sha256"),
            ("asset_manifest_path", "asset_manifest_sha256"),
            ("v4_helper_path", "v4_helper_sha256"),
            ("qualifier_path", "qualifier_sha256"),
            ("focused_test_path", "focused_test_sha256"),
        ):
            if path_key in {
                "a1_qualification_path",
                "data_role_registry_path",
                "decision_log_path",
            }:
                # These hashes freeze the authority snapshot consumed by the
                # accepted historical PLUMAGE producer. DEC-019 advances the
                # live authority append-only; it must not rewrite this producer
                # or require its historical snapshot hashes to equal live bytes.
                continue
            relative = authority[path_key]
            try:
                actual = sha256_bytes(_read_bytes(repo_root, relative))
            except (FileNotFoundError, ValueError) as exc:
                _issue(
                    issues,
                    "GSE149487_PLUMAGE_PROTOCOL_AUTHORITY",
                    str(relative),
                    str(exc),
                )
                continue
            if actual != authority[hash_key]:
                _issue(
                    issues,
                    "GSE149487_PLUMAGE_PROTOCOL_AUTHORITY",
                    str(relative),
                    f"current hash {actual} must match protocol {hash_key} {authority[hash_key]}",
                )

    scope = protocol.get("scope")
    if not isinstance(scope, Mapping):
        _issue(issues, "GSE149487_PLUMAGE_PROTOCOL_GATES", path, "scope must be a mapping")
    else:
        for key, expected in (
            ("ordinary_public_data_only", True),
            ("training_allowed", False),
            ("model_selection_allowed", False),
            ("authority_update_allowed_by_qualifier", False),
        ):
            _expect(
                scope,
                key,
                expected,
                path,
                issues,
                "GSE149487_PLUMAGE_PROTOCOL_GATES",
            )

    mapping = protocol.get("mapping")
    if not isinstance(mapping, Mapping):
        _issue(issues, "GSE149487_PLUMAGE_PROTOCOL_EVIDENCE", path, "mapping must be a mapping")
    else:
        _expect(
            mapping,
            "outcome_blind_mapping_evidence_status",
            "UNKNOWN_NOT_ASSERTED",
            path,
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_EVIDENCE",
        )
        _expect(
            mapping,
            "membership_may_depend_on_measured_effect_or_significance",
            False,
            path,
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_MAPPING",
        )

    transform = protocol.get("paper_faithful_measurement_transform")
    if not isinstance(transform, Mapping):
        _issue(issues, "GSE149487_PLUMAGE_PROTOCOL_EVIDENCE", path, "paper transform must be a mapping")
    else:
        for key in (
            "method_source_status",
            "multiple_testing_family_status",
            "published_result_crosscheck_status",
        ):
            _expect(
                transform,
                key,
                "UNKNOWN_NOT_ASSERTED",
                path,
                issues,
                "GSE149487_PLUMAGE_PROTOCOL_EVIDENCE",
            )

    license_audit = protocol.get("license_and_redistribution")
    if not isinstance(license_audit, Mapping):
        _issue(issues, "GSE149487_PLUMAGE_PROTOCOL_EVIDENCE", path, "license audit must be a mapping")
    else:
        for key, expected in (
            ("audit_status", "UNKNOWN_NOT_ASSERTED"),
            ("unknown_status_blocks_qualification", True),
            ("license_id", "UNKNOWN_NOT_ASSERTED"),
            ("verified_at", "UNKNOWN_NOT_ASSERTED"),
        ):
            _expect(
                license_audit,
                key,
                expected,
                path,
                issues,
                "GSE149487_PLUMAGE_PROTOCOL_EVIDENCE",
            )

    exposure = protocol.get("foundation_exposure")
    if not isinstance(exposure, Mapping):
        _issue(issues, "GSE149487_PLUMAGE_PROTOCOL_EVIDENCE", path, "foundation exposure must be a mapping")
    else:
        for key, expected in (
            ("audit_status", "UNKNOWN_NOT_ASSERTED"),
            ("checkpoint_id", "UNKNOWN_NOT_ASSERTED"),
            ("checkpoint_sha256", "UNKNOWN_NOT_ASSERTED"),
            ("stratum", "DEVELOPMENT_ONLY"),
            ("sequence_exposed", True),
            ("label_exposed", True),
            ("audit_id", "UNKNOWN_NOT_ASSERTED"),
            ("unknown_checkpoint_blocks_qualification", True),
        ):
            _expect(
                exposure,
                key,
                expected,
                path,
                issues,
                "GSE149487_PLUMAGE_PROTOCOL_EVIDENCE",
            )

    canonical = protocol.get("canonical_v3")
    if not isinstance(canonical, Mapping):
        _issue(
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_CANONICAL",
            path,
            "canonical_v3 must be a mapping",
        )
    else:
        _expect(
            canonical,
            "materialize_only_when_every_qualification_gate_passes",
            True,
            path,
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_CANONICAL",
        )

    preflight_binding = protocol.get("stop_before_data_preflight_binding")
    expected_preflight_keys = {
        "binding_scheme",
        "status",
        "implementation_commit",
        "external_evidence_config_path",
        "external_evidence_config_sha256",
        "preflight_script_path",
        "preflight_script_sha256",
        "preflight_test_path",
        "preflight_test_sha256",
    }
    if not isinstance(preflight_binding, Mapping) or set(preflight_binding) != expected_preflight_keys:
        _issue(
            issues,
            "GSE149487_PLUMAGE_PREFLIGHT_BINDING",
            path,
            f"stop_before_data_preflight_binding keys must be exactly {sorted(expected_preflight_keys)!r}",
        )
    else:
        _expect(
            preflight_binding,
            "binding_scheme",
            GSE149487_PLUMAGE_PREFLIGHT_BINDING_SCHEME,
            path,
            issues,
            "GSE149487_PLUMAGE_PREFLIGHT_BINDING",
        )
        status = preflight_binding.get("status")
        implementation = preflight_binding.get("implementation_commit")
        qualifier_implementation = (
            authority.get("implementation_commit") if isinstance(authority, Mapping) else None
        )
        if status == "UNKNOWN_NOT_ASSERTED":
            if (
                implementation != "UNKNOWN_NOT_ASSERTED"
                or qualifier_implementation != "UNKNOWN_NOT_ASSERTED"
            ):
                _issue(
                    issues,
                    "GSE149487_PLUMAGE_PREFLIGHT_BINDING",
                    path,
                    "UNKNOWN preflight binding requires UNKNOWN commit and UNKNOWN qualifier implementation",
                )
        elif status == "BOUND":
            if not (
                isinstance(implementation, str)
                and len(implementation) == 40
                and all(ch in "0123456789abcdef" for ch in implementation)
                and qualifier_implementation == implementation
            ):
                _issue(
                    issues,
                    "GSE149487_PLUMAGE_PREFLIGHT_BINDING",
                    path,
                    "BOUND preflight binding requires one full lowercase qualifier/preflight implementation commit",
                )
        else:
            _issue(
                issues,
                "GSE149487_PLUMAGE_PREFLIGHT_BINDING",
                path,
                "preflight binding status must be UNKNOWN_NOT_ASSERTED or BOUND",
            )

        for path_key, hash_key, expected_path, expected_hash in (
            (
                "external_evidence_config_path",
                "external_evidence_config_sha256",
                GSE149487_PLUMAGE_PREFLIGHT_CONFIG_PATH,
                GSE149487_PLUMAGE_PREFLIGHT_CONFIG_SHA256,
            ),
            (
                "preflight_script_path",
                "preflight_script_sha256",
                GSE149487_PLUMAGE_PREFLIGHT_SCRIPT_PATH,
                GSE149487_PLUMAGE_PREFLIGHT_SCRIPT_SHA256,
            ),
            (
                "preflight_test_path",
                "preflight_test_sha256",
                GSE149487_PLUMAGE_PREFLIGHT_TEST_PATH,
                GSE149487_PLUMAGE_PREFLIGHT_TEST_SHA256,
            ),
        ):
            relative = preflight_binding.get(path_key)
            declared = preflight_binding.get(hash_key)
            if relative != expected_path or declared != expected_hash:
                _issue(
                    issues,
                    "GSE149487_PLUMAGE_PREFLIGHT_BINDING",
                    path,
                    f"{path_key}/{hash_key} must be the frozen repository path and SHA256",
                )
                continue
            try:
                actual = sha256_bytes(_read_bytes(repo_root, relative))
            except (FileNotFoundError, ValueError) as exc:
                _issue(
                    issues,
                    "GSE149487_PLUMAGE_PREFLIGHT_BINDING",
                    relative,
                    str(exc),
                )
                continue
            if actual != declared:
                _issue(
                    issues,
                    "GSE149487_PLUMAGE_PREFLIGHT_BINDING",
                    relative,
                    f"current hash {actual} must match declared {hash_key} {declared}",
                )

    gate = protocol.get("current_gate_contract")
    if not isinstance(gate, Mapping):
        _issue(issues, "GSE149487_PLUMAGE_PROTOCOL_GATES", path, "current_gate_contract must be a mapping")
    else:
        _expect_closed_mapping(
            gate,
            GSE149487_PLUMAGE_CURRENT_GATE_CONTRACT,
            path,
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_GATES",
        )
    _expect(
        protocol,
        "qualification_gates",
        GSE149487_PLUMAGE_QUALIFICATION_GATES,
        path,
        issues,
        "GSE149487_PLUMAGE_PROTOCOL_QUALIFICATION_GATES",
    )
    _expect(
        protocol,
        "known_external_evidence_blockers",
        GSE149487_PLUMAGE_EXTERNAL_BLOCKERS,
        path,
        issues,
        "GSE149487_PLUMAGE_PROTOCOL_EVIDENCE",
    )
    try:
        observed_core_sha256 = _gse149487_plumage_nonbinding_core_sha256(protocol)
    except (TypeError, ValueError) as exc:
        _issue(
            issues,
            "GSE149487_PLUMAGE_PROTOCOL_NONBINDING_CORE",
            path,
            f"non-binding core could not be canonicalized: {exc}",
        )
    else:
        if observed_core_sha256 != GSE149487_PLUMAGE_NONBINDING_CORE_SHA256:
            _issue(
                issues,
                "GSE149487_PLUMAGE_PROTOCOL_NONBINDING_CORE",
                path,
                "full qualifier semantics changed outside the three permitted I/B scalars",
            )
    return issues


def validate_gse200302_role_protocol(repo_root: Path) -> list[Issue]:
    """Validate the mutable binding separately from the immutable v1 protocol core."""

    issues: list[Issue] = []
    path = GSE200302_ROLE_CONFIG_PATH

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = _read_bytes(repo_root, path)
        protocol = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (FileNotFoundError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_LOAD",
            path,
            f"role protocol must be duplicate-free finite UTF-8 JSON: {exc}",
        )
        return issues
    if type(protocol) is not dict:
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_SHAPE",
            path,
            "role protocol root must be an object",
        )
        return issues

    expected_top_keys = {
        "schema_version",
        "protocol_id",
        "protocol_trust",
        "implementation_binding",
        "scope",
        "sources",
        "join_contract",
        "mapping_contract",
        "experiment_join_contract",
        "publication",
        "gate_contract",
        "execution_policy",
        "claim_boundary",
    }
    if set(protocol) != expected_top_keys:
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_SHAPE",
            path,
            f"top-level keys must be exactly {sorted(expected_top_keys)!r}",
        )

    core = {key: value for key, value in protocol.items() if key != "implementation_binding"}
    try:
        canonical_core = json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_CORE",
            path,
            f"non-binding core is not canonically serializable: {exc}",
        )
    else:
        observed_core_hash = sha256_bytes(canonical_core)
        if observed_core_hash != GSE200302_ROLE_PROTOCOL_CORE_SHA256:
            _issue(
                issues,
                "GSE200302_ROLE_PROTOCOL_CORE",
                path,
                f"non-binding core hash {observed_core_hash} must remain {GSE200302_ROLE_PROTOCOL_CORE_SHA256}",
            )

    if protocol.get("schema_version") != GSE200302_ROLE_PROTOCOL_SCHEMA:
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_SEMANTICS",
            path,
            f"schema_version must remain {GSE200302_ROLE_PROTOCOL_SCHEMA!r}",
        )
    if protocol.get("protocol_id") != GSE200302_ROLE_PROTOCOL_ID:
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_SEMANTICS",
            path,
            f"protocol_id must remain {GSE200302_ROLE_PROTOCOL_ID!r}",
        )

    trust = protocol.get("protocol_trust")
    if not isinstance(trust, Mapping):
        _issue(issues, "GSE200302_ROLE_PROTOCOL_SEMANTICS", path, "protocol_trust must be a mapping")
    else:
        _expect_closed_mapping(
            trust,
            {
                "canonicalization": "CANONICAL_SORTED_UTF8_V1",
                "core_projection_excluded_top_level_keys": ["implementation_binding"],
                "compiled_core_projection_required": True,
            },
            path,
            issues,
            "GSE200302_ROLE_PROTOCOL_SEMANTICS",
        )

    scope = protocol.get("scope")
    if not isinstance(scope, Mapping):
        _issue(issues, "GSE200302_ROLE_PROTOCOL_SEMANTICS", path, "scope must be a mapping")
    else:
        _expect_closed_mapping(
            scope,
            {
                "target_series_accession": "GSE200302",
                "bioproject_accession": "PRJNA824033",
                "authority_level": "OFFICIAL_METADATA_ROLE_AUTHORITY_ONLY",
                "ordinary_public_metadata_only": True,
            },
            path,
            issues,
            "GSE200302_ROLE_PROTOCOL_SEMANTICS",
        )

    mapping = protocol.get("mapping_contract")
    if not isinstance(mapping, Mapping):
        _issue(issues, "GSE200302_ROLE_PROTOCOL_GRID", path, "mapping_contract must be a mapping")
    else:
        for key, expected in (
            ("allowed_measurement_families", GSE200302_ROLE_MEASUREMENT_FAMILIES),
            ("replicates", GSE200302_ROLE_REPLICATES),
            ("forbidden_family_aliases", ["80S_RNA"]),
        ):
            _expect(mapping, key, expected, path, issues, "GSE200302_ROLE_PROTOCOL_GRID")
        rows = mapping.get("expected_rows")
        observed_grid: list[tuple[Any, Any]] = []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    observed_grid.append((row.get("measurement_family"), row.get("replicate")))
        expected_grid = {
            (family, replicate)
            for family in GSE200302_ROLE_MEASUREMENT_FAMILIES
            for replicate in GSE200302_ROLE_REPLICATES
        }
        if len(observed_grid) != 24 or len(set(observed_grid)) != 24 or set(observed_grid) != expected_grid:
            _issue(
                issues,
                "GSE200302_ROLE_PROTOCOL_GRID",
                path,
                "expected_rows must contain exactly the four official families by replicates 1 through 6",
            )

    gate = protocol.get("gate_contract")
    if not isinstance(gate, Mapping):
        _issue(issues, "GSE200302_ROLE_PROTOCOL_GATES", path, "gate_contract must be a mapping")
    else:
        _expect_closed_mapping(
            gate,
            {
                "role_authority_status": "EXACT_OFFICIAL_SRR_ROLE_AUTHORITY_CLOSED",
                "raw_replay_role_grid_status": "CONFLICT_WITH_CURRENT_80S_EXPECTATION",
                "qualified": False,
                "training_authorized": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "canonical_record_count": 0,
                "next_phase_authorized": False,
            },
            path,
            issues,
            "GSE200302_ROLE_PROTOCOL_GATES",
        )

    execution = protocol.get("execution_policy")
    if not isinstance(execution, Mapping):
        _issue(issues, "GSE200302_ROLE_PROTOCOL_GATES", path, "execution_policy must be a mapping")
    else:
        _expect_closed_mapping(
            execution,
            {
                "network_access_allowed": False,
                "subprocess_allowed": False,
                "fixed_argv_read_only_git_subprocess_allowed": True,
                "git_binary": "/usr/bin/git",
                "fastq_body_read_allowed": False,
                "sequence_output_allowed": False,
                "barcode_output_allowed": False,
                "training_label_output_allowed": False,
                "qualification_allowed": False,
                "canonical_materialization_allowed": False,
                "training_allowed": False,
                "next_phase_unlock_allowed": False,
            },
            path,
            issues,
            "GSE200302_ROLE_PROTOCOL_GATES",
        )

    def contains_key(value: Any, wanted: str) -> bool:
        if isinstance(value, Mapping):
            return wanted in value or any(contains_key(child, wanted) for child in value.values())
        if isinstance(value, list):
            return any(contains_key(child, wanted) for child in value)
        return False

    if contains_key(core, "model_selection_allowed"):
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_MODEL_SELECTION_FIELD",
            path,
            "the published v1 core must not gain a model_selection_allowed field; outer ledger authority fails it closed",
        )
    if contains_key(core, "pdna_may_substitute_for_80s_rna"):
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_80S_BOUNDARY",
            path,
            "the closed v1 core may not encode a pDNA-to-80S substitution field",
        )

    binding = protocol.get("implementation_binding")
    expected_binding_keys = {
        "status",
        "binding_mode",
        "implementation_commit",
        "implementation_script_repo_path",
        "implementation_script_sha256",
        "implementation_test_repo_path",
        "implementation_test_sha256",
        "protocol_repo_path",
        "activation_rule",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected_binding_keys:
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_BINDING",
            path,
            f"implementation_binding keys must be exactly {sorted(expected_binding_keys)!r}",
        )
        return issues
    expected_fixed_binding = {
        "binding_mode": GSE200302_ROLE_BINDING_MODE,
        "implementation_script_repo_path": GSE200302_ROLE_BUILDER_PATH,
        "implementation_test_repo_path": GSE200302_ROLE_TEST_PATH,
        "protocol_repo_path": GSE200302_ROLE_CONFIG_PATH,
        "activation_rule": GSE200302_ROLE_BINDING_ACTIVATION_RULE,
    }
    for key, expected in expected_fixed_binding.items():
        if not _json_type_strict_equal(binding.get(key), expected):
            _issue(
                issues,
                "GSE200302_ROLE_PROTOCOL_BINDING",
                path,
                f"implementation_binding.{key} must remain {expected!r}",
            )

    if binding.get("status") != "BOUND":
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_BINDING",
            path,
            "production implementation_binding.status must be BOUND",
        )
        return issues

    commit = binding.get("implementation_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        _issue(
            issues,
            "GSE200302_ROLE_PROTOCOL_BINDING",
            path,
            "BOUND implementation_commit must be a full lowercase object ID",
        )
    for binding_key, repo_path in (
        ("implementation_script_sha256", GSE200302_ROLE_BUILDER_PATH),
        ("implementation_test_sha256", GSE200302_ROLE_TEST_PATH),
    ):
        declared = binding.get(binding_key)
        if not _is_sha256(declared):
            _issue(
                issues,
                "GSE200302_ROLE_PROTOCOL_BINDING",
                path,
                f"BOUND {binding_key} must be a lowercase SHA256",
            )
            continue
        try:
            actual = sha256_bytes(_read_bytes(repo_root, repo_path))
        except (FileNotFoundError, ValueError) as exc:
            _issue(
                issues,
                "GSE200302_ROLE_PROTOCOL_BINDING",
                repo_path,
                str(exc),
            )
            continue
        if declared != actual:
            _issue(
                issues,
                "GSE200302_ROLE_PROTOCOL_BINDING",
                repo_path,
                f"current bytes hash {actual} must match bound {binding_key} {declared}",
            )
    return issues


def validate_a1_interim_lineage(
    repo_root: Path,
    interim: Mapping[str, Any],
) -> list[Issue]:
    """Bind the active A1 blocked record to Scheme A without granting a gate."""

    issues: list[Issue] = []
    path = A1_INTERIM_PATH
    try:
        actual_interim_hash = sha256_bytes(_read_bytes(repo_root, path))
    except (FileNotFoundError, ValueError) as exc:
        _issue(issues, "A1_INTERIM_UNREADABLE", path, str(exc))
        actual_interim_hash = None
    if actual_interim_hash is not None and actual_interim_hash != EXPECTED_A1_INTERIM_SHA256:
        _issue(
            issues,
            "A1_INTERIM_CANONICAL_HASH",
            path,
            f"active interim hash {actual_interim_hash} must remain {EXPECTED_A1_INTERIM_SHA256}",
        )

    expected_top = {
        "schema_version": "1.0.0",
        "contract_id": CONTRACT_ID,
        "contract_version": VERSION,
        "record_id": "ROUTE_A_V3_A1_INTERIM_20260810",
        "record_type": "A1_PUBLIC_DATA_QUALIFICATION_INTERIM",
        "phase_id": "A1",
        "record_status": "INTERIM_BLOCKED_NOT_PHASE_COMPLETE",
    }
    for key, value in expected_top.items():
        _expect(interim, key, value, path, issues, "A1_INTERIM_METADATA")
    expected_top_keys = {
        "schema_version",
        "contract_id",
        "contract_version",
        "record_id",
        "record_type",
        "phase_id",
        "record_status",
        "authority",
        "scope",
        "gate_snapshot",
        "dec019_current_disposition",
        "artifact_lineage",
        "dataset_boundary_summary",
        "boundary_deviation",
        "power_prefreeze",
        "claim_boundaries",
        "verification",
        "initial_generated_at",
        "generated_at",
        "updated_at",
        "updated_for_decision_id",
        "latest_authority_update_id",
        "latest_evidence_update_id",
    }
    if set(interim) != expected_top_keys:
        _issue(
            issues,
            "A1_INTERIM_METADATA",
            path,
            f"top-level keys must be exactly {sorted(expected_top_keys)!r}",
        )

    authority = interim.get("authority")
    if not isinstance(authority, Mapping):
        _issue(issues, "A1_INTERIM_AUTHORITY", path, "authority must be a mapping")
    else:
        expected_authority = {
            "contract_path": GOAL_PATH,
            "initial_contract_sha256": "d1c031aecdec710495f6861b380785cccd64663ac4bd97b4f479d6fdf372ea07",
            "contract_sha256": SOURCE_CONTRACT_SHA256,
            "active_amendment_decision_ids": ACTIVE_AMENDMENT_DECISION_IDS,
            "dec019_amendment_path": DEC019_AMENDMENT_PATH,
            "decision_log_path": DECISION_LOG_PATH,
            "data_role_registry_path": REGISTRY_PATHS["data"],
            "claim_evidence_matrix_path": REGISTRY_PATHS["claim"],
            "accepted_a0_activation_commit": "fd722d5fa3c2538fce742b8942b1fb48e782760b",
            "branch": "routea-v3-a1-20260810",
            "worktree": "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810",
            "run_id": "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5",
            "run_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5",
            "data_role_authority_remains": REGISTRY_PATHS["data"],
            "this_record_changes_dataset_qualification": False,
        }
        for key, value in expected_authority.items():
            _expect(authority, key, value, path, issues, "A1_INTERIM_AUTHORITY")
        for hash_key, relative in (
            ("dec019_amendment_sha256", DEC019_AMENDMENT_PATH),
            ("decision_log_sha256", DECISION_LOG_PATH),
            ("data_role_registry_sha256", REGISTRY_PATHS["data"]),
            ("claim_evidence_matrix_sha256", REGISTRY_PATHS["claim"]),
        ):
            try:
                actual = sha256_bytes(_read_bytes(repo_root, relative))
            except (FileNotFoundError, ValueError) as exc:
                _issue(issues, "A1_INTERIM_AUTHORITY_FILE", relative, str(exc))
            else:
                _expect(authority, hash_key, actual, path, issues, "A1_INTERIM_AUTHORITY_HASH")
        expected_authority_keys = set(expected_authority) | {
            "dec019_amendment_sha256",
            "decision_log_sha256",
            "data_role_registry_sha256",
            "claim_evidence_matrix_sha256",
        }
        if set(authority) != expected_authority_keys:
            _issue(
                issues,
                "A1_INTERIM_AUTHORITY",
                path,
                f"authority keys must be exactly {sorted(expected_authority_keys)!r}",
            )

    scope = interim.get("scope")
    if not isinstance(scope, Mapping):
        _issue(issues, "A1_INTERIM_SCOPE", path, "scope must be a mapping")
    else:
        expected_scope = {
            "ordinary_public_data_only": True,
            "included_dataset_ids": [
                "GSE145046",
                "GSE114002",
                "GSE149487",
                "GSE217518",
                "GSE200304",
                "ENCSR854RUF",
                "GSE232572",
                "GSE186455",
                "GSE207584",
            ],
            "absolute_auxiliary_dataset_ids": ["GSE145046"],
            "true_a2_recovery_candidate_dataset_ids": ["GSE114002"],
            "scheme_a_changes_qualified_counts": False,
            "excluded_dataset_ids": ["GSE246381"],
            "legacy_canonical_purpose": "GAP_INVENTORY_ONLY",
            "metadata_only_qualification_allowed": False,
            "training_allowed": False,
            "model_selection_allowed": False,
            "raw_sequence_or_label_payload_embedded": False,
            "record_contains_row_or_member_payload": False,
            "record_contains_sequence_values": False,
            "record_contains_raw_label_values": False,
            "training_started": False,
            "gpu_work_started": False,
            "model_selection_started": False,
            "sealed_evaluation_count": 0,
        }
        _expect_closed_mapping(
            scope, expected_scope, path, issues, "A1_INTERIM_SCOPE"
        )

    gate = interim.get("gate_snapshot")
    if not isinstance(gate, Mapping):
        _issue(issues, "A1_INTERIM_GATE", path, "gate_snapshot must be a mapping")
    else:
        expected_gate = {
            "decision": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "scientific_claim_status": "NOT_ESTABLISHED",
            "qualified_independent_ordinary_studies": 0,
            "required_independent_ordinary_studies": 3,
            "qualified_a1_studies": 0,
            "required_a1_studies": 2,
            "qualified_a2_dense_studies": 0,
            "required_a2_dense_studies": 1,
            "metadata_only_qualification_count": 0,
            "phase_complete": False,
            "next_phase_authorized": False,
            "a2_training_authorized": False,
        }
        _expect_closed_mapping(
            gate, expected_gate, path, issues, "A1_INTERIM_GATE"
        )

    dec019_disposition = interim.get("dec019_current_disposition")
    if not isinstance(dec019_disposition, Mapping):
        _issue(issues, "A1_INTERIM_DEC019", path, "dec019_current_disposition must be a mapping")
    else:
        expected_dec019_keys = {
            "decision_id",
            "status",
            "authority_only_not_study_qualification",
            "changes_historical_gse114002_attempt_or_audit_bytes",
            "changes_historical_gse200304_published_endpoint_bundle_bytes",
            "gse114002_designed_library",
            "gse200304_published_processed_endpoint",
            "split_freeze_boundary",
            "uncertainty_and_power",
            "gate_snapshot",
            "runtime_sync_status",
        }
        if set(dec019_disposition) != expected_dec019_keys:
            _issue(issues, "A1_INTERIM_DEC019", path, f"DEC-019 disposition keys must be exactly {sorted(expected_dec019_keys)!r}")
        for key, value in {
            "decision_id": "V3-DEC-019",
            "status": "FROZEN_USER_AUTHORIZED_A1_QUALIFICATION_POLICY",
            "authority_only_not_study_qualification": True,
            "changes_historical_gse114002_attempt_or_audit_bytes": False,
            "changes_historical_gse200304_published_endpoint_bundle_bytes": False,
            "split_freeze_boundary": {
                "a1_freezes": ["SOURCE_AUTHORITY", "BIOLOGICAL_SOURCE_GROUP_ASSIGNMENT", "NEAR_DUPLICATE_GRAPH", "SPLIT_SALT"],
                "a1_freeze_status": "NOT_FROZEN",
                "a1_zero_leakage_audit_status": "NOT_RUN",
                "a2_freezes": ["FINAL_BENCHMARK_MEMBERSHIP"],
                "a2_final_membership_status": "NOT_FROZEN",
            },
            "uncertainty_and_power": {
                "target_power_minimum": 0.8,
                "confidence_level": 0.95,
                "full_ci_width_definition": "UPPER_MINUS_LOWER",
                "maximum_full_ci_width": 0.3,
                "model_results_may_change_thresholds": False,
                "uncertainty_routes_are_dataset_scoped_only": True,
                "global_replicate_or_standard_error_relaxation_allowed": False,
                "gse149487_three_biological_replicates_and_route_a_se_gate_changed": False,
            },
            "gate_snapshot": {
                "qualified_independent_ordinary_studies": 0,
                "qualified_a1_studies": 0,
                "qualified_true_a2_dense_studies": 0,
                "canonical_record_count": 0,
                "phase_complete": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
            },
            "runtime_sync_status": "PENDING_NO_RUNTIME_EVENT",
        }.items():
            _expect(dec019_disposition, key, value, path, issues, "A1_INTERIM_DEC019")
        gse114002_current = dec019_disposition.get("gse114002_designed_library")
        if not isinstance(gse114002_current, Mapping):
            _issue(issues, "A1_INTERIM_DEC019_GSE114002", path, "current GSE114002 disposition must be a mapping")
        else:
            for key, value in {
                "owner_uncertainty_policy_status": "CLOSED_BY_V3_DEC_019",
                "current_status": "NOT_QUALIFIED",
                "maximum_independent_ordinary_study_contribution_if_qualified": 1,
                "maximum_a1_study_contribution_if_qualified": 0,
                "maximum_true_a2_dense_study_contribution_if_qualified": 1,
                "gsm_library_pool_or_candidate_may_count_as_independent_study": False,
                "candidate_hamming_distance_eligibility_if_qualified": [1, 2, 3],
                "contract_primary_budget_reporting_if_qualified": [1, 3],
                "global_contract_primary_edit_budgets_retained": [1, 3, 5],
                "k5_role": "CLAIM_BOUNDARY_ONLY_NOT_QUALIFICATION_CREDIT",
                "confirmatory_contribution": 0,
                "generalization_contribution": 0,
                "current_blocker_count": 6,
                "current_blockers": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_BLOCKERS[:5] + [GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_BLOCKERS[6]],
            }.items():
                _expect(gse114002_current, key, value, path, issues, "A1_INTERIM_DEC019_GSE114002")
            technical = gse114002_current.get("technical_fraction_uncertainty")
            if not isinstance(technical, Mapping) or not _json_type_strict_equal(
                technical,
                {
                    "use_scope": "QC_AND_WITHIN_ASSAY_TECHNICAL_DIAGNOSTIC_ONLY",
                    "may_support_biological_standard_error": False,
                    "may_support_power": False,
                    "may_support_confidence_interval": False,
                    "may_support_equivalence": False,
                    "may_support_confirmatory_claim": False,
                    "may_support_generalization_claim": False,
                    "may_substitute_for_biological_standard_error": False,
                },
            ):
                _issue(issues, "A1_INTERIM_DEC019_GSE114002", path, "technical uncertainty must remain closed QC-only")
        gse200304_current = dec019_disposition.get("gse200304_published_processed_endpoint")
        if not isinstance(gse200304_current, Mapping):
            _issue(issues, "A1_INTERIM_DEC019_GSE200304", path, "current GSE200304 disposition must be a mapping")
        else:
            expected_gse200304_current = {
                "owner_primary_route_policy_status": "CLOSED_BY_V3_DEC_019",
                "current_status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
                "ordinary_study_unit": "GSE200304_SUPERSERIES_ONE_STUDY",
                "maximum_independent_ordinary_study_contribution_if_qualified": 1,
                "maximum_a1_study_contribution_if_qualified": 1,
                "maximum_true_a2_dense_study_contribution_if_qualified": 0,
                "superseries_subseries_modality_endpoint_or_replicate_may_count_as_independent_study": False,
                "primary_measurement_route_if_qualified": "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT",
                "raw_replay_role": "REPRODUCIBILITY_AUXILIARY",
                "required_80s_role_authority_is_primary_route_blocker": False,
                "current_blocker_count": 7,
                "current_blockers": GSE200304_DEC019_POST_ADJUDICATION_BLOCKERS,
                "input_gate_count": 8,
                "input_status_counts": GSE200304_DEC019_POST_ADJUDICATION_INPUT_STATUS_COUNTS,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "positive_input_canonical_record_count": 6547,
                "canonical_record_count": 0,
                "qualified": False,
                "canonical_materialization_allowed": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            }
            _expect_closed_mapping(
                gse200304_current,
                expected_gse200304_current,
                path,
                issues,
                "A1_INTERIM_DEC019_GSE200304",
            )

    lineage = interim.get("artifact_lineage")
    if not isinstance(lineage, Mapping):
        _issue(issues, "A1_INTERIM_GSE200304_LINEAGE", path, "artifact_lineage must be a mapping")
    else:
        expected_all_lineage_ids = {
            "protocol",
            GSE114002_DEC019_SUCCESSOR_LINEAGE_ID,
            GSE200304_DEC019_SUCCESSOR_LINEAGE_ID,
            "collector",
            "legacy_gap_inventory_v1",
            "legacy_gap_inventory_v2",
            "gse114002_manifest_reconciliation_v1",
            GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID,
            GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID,
            GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_LINEAGE_ID,
            "gse149487_reconstruction_attempt_003_failure",
            "gse149487_lim6c_scale_diagnostic_v1",
            "gse149487_plumage_protocol",
            "gse149487_plumage_reconstruction_v4",
            "gse149487_full_a1_stop_before_data_preflight_v1",
            "gse145046_a2_audit_protocol",
            "gse145046_a2_formal_audit_v1",
            "a1_public_qualifiers_sync_v1",
            "gse200304_public_asset_bundle",
            "gse200304_ena_fastq_manifest_bundle",
            "gse200304_fastq_acquisition_v1",
            "gse200304_fastq_independent_consumer_verification_v1",
            "gse200304_qualifier_protocol",
            "gse200304_gap_qualification_attempt_001_failure",
            "gse200304_gap_qualification_attempt_002_failure",
            "gse200304_gap_qualification_attempt_003_failure",
            "gse200304_gap_qualification_v1",
            "gse200304_raw_replay_preflight_attempt_001_failure",
            "gse200304_raw_replay_preflight_v1",
            "gse200302_srr_role_authority_v1",
            "gse200304_published_endpoint_a1_v1",
            GSE200304_DEC019_LINEAGE_GATE_LINEAGE_ID,
            GSE200304_DEC019_NEGATIVE_GATE_PACK_LINEAGE_ID,
            GSE200304_DEC019_ADJUDICATION_LINEAGE_ID,
        }
        if set(lineage) != expected_all_lineage_ids:
            _issue(
                issues,
                "A1_INTERIM_LINEAGE_ID_SET",
                path,
                "artifact lineage IDs must remain the exact accepted closed set",
            )
        expected_non_gse200304_lineage = {
            "protocol": {
                "path": "configs/route_a_v3_a1_qualification.json",
                "sha256": "fe3f7736c1f64b362ebda683ca571fc1a84e1fff36aed3a9ae67272665ba2343",
                "status": "PREFROZEN_BEFORE_MODEL_RESULTS",
            },
            GSE114002_DEC019_SUCCESSOR_LINEAGE_ID: {
                "dataset_id": "GSE114002",
                "decision_id": "V3-DEC-019",
                "config_path": GSE114002_DEC019_SUCCESSOR_CONFIG_PATH,
                "initial_i_config_sha256": GSE114002_DEC019_SUCCESSOR_INITIAL_I_SHA256,
                "stable_core_projection_sha256": GSE114002_DEC019_SUCCESSOR_CORE_SHA256,
                "implementation_binding_lifecycle": "UNKNOWN_TO_CONFIG_ONLY_BOUND",
                "initial_implementation_binding_status": "UNKNOWN_NOT_ASSERTED",
                "dynamic_config_full_sha_in_static_manifest": False,
                "script_path": GSE114002_DEC019_SUCCESSOR_SCRIPT_PATH,
                "script_sha256": GSE114002_DEC019_SUCCESSOR_SCRIPT_SHA256,
                "focused_test_path": GSE114002_DEC019_SUCCESSOR_TEST_PATH,
                "focused_test_sha256": GSE114002_DEC019_SUCCESSOR_TEST_SHA256,
                "authority_only_not_study_qualification": True,
                "current_qualified": False,
                "current_ordinary_study_contribution": 0,
                "current_a1_study_contribution": 0,
                "current_true_a2_dense_study_contribution": 0,
                "current_canonical_record_count": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            },
            GSE200304_DEC019_SUCCESSOR_LINEAGE_ID: {
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "config_path": GSE200304_DEC019_SUCCESSOR_CONFIG_PATH,
                "initial_i_config_sha256": GSE200304_DEC019_SUCCESSOR_INITIAL_I_SHA256,
                "stable_core_projection_sha256": GSE200304_DEC019_SUCCESSOR_CORE_SHA256,
                "implementation_binding_lifecycle": "UNKNOWN_TO_CONFIG_ONLY_BOUND",
                "initial_implementation_binding_status": "UNKNOWN_NOT_ASSERTED",
                "dynamic_config_full_sha_in_static_manifest": False,
                "script_path": GSE200304_DEC019_SUCCESSOR_SCRIPT_PATH,
                "script_sha256": GSE200304_DEC019_SUCCESSOR_SCRIPT_SHA256,
                "focused_test_path": GSE200304_DEC019_SUCCESSOR_TEST_PATH,
                "focused_test_sha256": GSE200304_DEC019_SUCCESSOR_TEST_SHA256,
                "authority_only_not_study_qualification": True,
                "current_qualified": False,
                "current_ordinary_study_contribution": 0,
                "current_a1_study_contribution": 0,
                "current_true_a2_dense_study_contribution": 0,
                "current_canonical_record_count": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            },
            "collector": {
                "path": "scripts/route_a_v3/audit_a1_public_data.py",
                "sha256": "f5070e6cf6a884e6960654b03fab90f1aa5e1fe2508ee51aef46afce4d4da8ba",
                "purpose": "GAP_INVENTORY_ONLY",
                "may_auto_qualify_study": False,
            },
            "legacy_gap_inventory_v1": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/A1_LEGACY_GAP_INVENTORY.json",
                "sha256": "b3262b7db32a8b501c99491aa100575fcbe188e388ea4c59d2f459e5cbc1c350",
                "preserved": True,
            },
            "legacy_gap_inventory_v2": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/A1_LEGACY_GAP_INVENTORY_V2.json",
                "bytes": 62421,
                "sha256": "d1b371fd350f910a6de38e27c50a30f9c97c660085382f0ac384ac9ecdc0fdff",
                "purpose": "GAP_INVENTORY_ONLY",
                "embedded_report_id": "A1_ORDINARY_PUBLIC_DATA_GAP_INVENTORY_V1",
                "phase_complete": False,
                "contains_sequence_or_raw_label_payload": False,
                "establishes_qualification": False,
                "supersedes_v1_report_shape": True,
                "scientific_decision_changed_from_v1": False,
            },
            "gse114002_manifest_reconciliation_v1": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE114002_MANIFEST_RECONCILIATION_V1.json",
                "bytes": 7375,
                "sha256": "7a14ca5410b1f2aeeeba4d72acf48056cd2e4dff10e65cccb539a139f100700e",
                "status": "PROVENANCE_RECONCILED_NOT_QUALIFIED",
                "original_manifest_preserved": True,
                "canonical_payloads_preserved": True,
                "quarantine_evidence_preserved": True,
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
            },
            GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID: {
                **GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_EXPECTED_RECORD,
                "files": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_EXPECTED_FILES,
            },
            GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID: {
                **GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_EXPECTED_RECORD,
                "files": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_EXPECTED_FILES,
            },
            GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_LINEAGE_ID: (
                GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_EXPECTED_RECORD
            ),
            "gse149487_reconstruction_attempt_003_failure": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE149487_RECONSTRUCTION_ATTEMPT_003_FAILURE.json",
                "bytes": 2568,
                "sha256": "e05979690d92f463299698d5e78eaadfdcd3d05858a547bcd062f6245dcb7ba5",
                "status": "FAIL_CLOSED_BEFORE_OUTPUT_PUBLICATION",
                "failure_type": "NEGATIVE_VALUE_IN_DECLARED_CPM_INPUT",
                "failed_output_id_terminalized": True,
                "failure_deleted_or_concealed": False,
            },
            "gse149487_lim6c_scale_diagnostic_v1": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE149487_LIM6C_SCALE_DIAGNOSTIC_V1.json",
                "bytes": 3860,
                "sha256": "8a1551df2f603da7bdef11054bdeb322f380060256a1e997beca72a3bff0e41f",
                "adjudication": "PUBLISHED_LOG2_CPM_PER_BARCODE_NOT_LINEAR_CPM",
                "official_data_corruption_established": False,
            },
            "gse149487_plumage_protocol": {
                "path": "configs/route_a_v3_plumage_reconstruction.json",
                "sha256": "a7dd048e55f3b71dd90597ac95993cbe4f643c35b5a6bbc38509ce0d2f0fcc4a",
                "input_value_scale": "PUBLISHED_LOG2_CPM_PER_BARCODE",
                "original_cpm_minimum_inclusive": 0.5,
                "published_log2_cpm_minimum_inclusive": -1.0,
                "model_results_may_change_protocol": False,
            },
            "gse149487_plumage_reconstruction_v4": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE149487_PLUMAGE_293T_PARTIAL_RECONSTRUCTION_V4",
                "bytes": 581230429,
                "status": "DEVELOPMENT_RECONSTRUCTED_NOT_QUALIFIED",
                "sha256sums_sha256": "38d4c4fefd94fe5400c5dd5de2893efab4b039152f872b2ea67e1ed0ff65a000",
                "report_sha256": "c4b6ef08714cd640edaa1e698cc0c92d3c7702570265203b36de6bbc90ce58b6",
                "success_record_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE149487_RECONSTRUCTION_ATTEMPT_004_SUCCESS.json",
                "success_record_sha256": "c2d563cc09a14747e9d225766694c10922058cbb54c70fe084148409ff0cc0e4",
                "public_inventory_gate_applied": True,
                "canonical_record_count": 0,
                "development_companion_effect_record_count": 204,
                "paper_method_reproduced": False,
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
            },
            "gse145046_a2_audit_protocol": {
                "path": "configs/route_a_v3_gse145046_a2_audit.json",
                "sha256": "666c9ee86033a05a006171df963fa3d96b68430d9a9d4e817789e255b28b300d",
                "auditor_path": "scripts/route_a_v3/audit_gse145046_a2.py",
                "auditor_sha256": "9a6751f1a8dd17acde0330ffb85e8f083ab6a40d5c4391bf8e48806a831b32f2",
                "focused_test_path": "tests/route_a_v3/test_audit_gse145046_a2.py",
                "focused_test_sha256": "2cfdf82aee025b7f793b59d3c90a1c55c6477ef8a3971b3c86cc69cad2606275",
                "implementation_commit": "00aaa01b8376126ed71fc6c34f599fcf0b841a56",
                "canonical_protocol_trust_root_closed": True,
                "model_results_may_change_protocol": False,
            },
            "gse145046_a2_formal_audit_v1": {
                "audit_execution_id": "GSE145046_A2_FORMAL_AUDIT_V1_20260810T084313P0800",
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE145046_A2_FORMAL_AUDIT_V1_20260810T084313P0800.json",
                "bytes": 56208,
                "sha256": "e383711cd9ae88d83ad8f34575db4f0db4c9c1077cbda54420248b8da75ab836",
                "log_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE145046_A2_FORMAL_AUDIT_V1_20260810T084313P0800.log",
                "log_bytes": 427,
                "log_sha256": "9aa4930ce201a82ac899bb88ee48add5854b2c1b03f4f3c256ad9d04d311fe3b",
                "audit_execution_status": "COMPLETED",
                "payload_integrity_status": "PASS",
                "rpm_validation_status": "PASS",
                "aggregate_reconciliation_status": "MATCH",
                "dataset_qualification_status": "NOT_QUALIFIED",
                "recoverability_status": "CONDITIONALLY_RECOVERABLE_AS_ABSOLUTE_AUXILIARY",
                "a2_status": "NOT_TRUE_A2_FIXED_REPORTER_ABSOLUTE_AUXILIARY",
                "canonical_intervention_record_count": 0,
                "measured_candidate_pool_count": 0,
                "endpoint_values_materialized": False,
                "paper_method_reproduced": False,
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
            },
        }
        for lineage_id, expected_record in expected_non_gse200304_lineage.items():
            record = lineage.get(lineage_id)
            if not isinstance(record, Mapping):
                _issue(
                    issues,
                    "A1_INTERIM_LINEAGE",
                    path,
                    f"{lineage_id} must be a mapping",
                )
                continue
            _expect_closed_mapping(
                record,
                expected_record,
                path,
                issues,
                "A1_INTERIM_LINEAGE",
            )
        attempt_001 = lineage.get(
            GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID
        )
        attempt_002 = lineage.get(
            GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID
        )
        if isinstance(attempt_001, Mapping) and isinstance(attempt_002, Mapping):
            prior = attempt_001.get("unresolved_blockers")
            current = attempt_002.get("unresolved_blockers")
            if not (
                type(prior) is list
                and type(current) is list
                and set(prior) - set(current)
                == set(GSE114002_ENDPOINT_GEOMETRY_CLOSED_BLOCKERS)
                and set(current).issubset(set(prior))
            ):
                _issue(
                    issues,
                    "A1_INTERIM_GSE114002_ENDPOINT_GEOMETRY_HISTORY",
                    path,
                    "attempt 002 may close exactly the two frozen mechanical blockers and may not rewrite the preserved attempt 001 blocker set",
                )
        for relative, expected_sha256, expected_bytes in (
            (
                GSE114002_ENDPOINT_GEOMETRY_CONFIG_PATH,
                "560c19c3cf6d2e41f8b05978584ce884dd2beb824cdca9392a597ff406120ff8",
                13139,
            ),
            (
                GSE114002_ENDPOINT_GEOMETRY_SCRIPT_PATH,
                "46e41b387357da0deae7139ac675638075cbae4fc24ac5c9583e969adfa8308d",
                None,
            ),
            (
                GSE114002_ENDPOINT_GEOMETRY_TEST_PATH,
                "8a0b04b3305f50fb1685ab310efaead15a1e3f4ca5eae4abb01d7f44a9d7be29",
                None,
            ),
        ):
            try:
                payload = _read_bytes(repo_root, relative)
            except (FileNotFoundError, ValueError) as exc:
                _issue(
                    issues,
                    "A1_INTERIM_GSE114002_ENDPOINT_GEOMETRY_BINDING",
                    relative,
                    str(exc),
                )
                continue
            actual_sha256 = sha256_bytes(payload)
            if actual_sha256 != expected_sha256:
                _issue(
                    issues,
                    "A1_INTERIM_GSE114002_ENDPOINT_GEOMETRY_BINDING",
                    relative,
                    f"current bytes hash {actual_sha256} must remain {expected_sha256}",
                )
            if expected_bytes is not None and len(payload) != expected_bytes:
                _issue(
                    issues,
                    "A1_INTERIM_GSE114002_ENDPOINT_GEOMETRY_BINDING",
                    relative,
                    f"current byte count {len(payload)} must remain {expected_bytes}",
                )
        gse149487_preflight = lineage.get(
            "gse149487_full_a1_stop_before_data_preflight_v1"
        )
        if not isinstance(gse149487_preflight, Mapping):
            _issue(
                issues,
                "A1_INTERIM_GSE149487_PREFLIGHT",
                path,
                "gse149487_full_a1_stop_before_data_preflight_v1 must be a mapping",
            )
        else:
            _expect_closed_mapping(
                gse149487_preflight,
                {
                    "artifact_id": GSE149487_PLUMAGE_PREFLIGHT_ARTIFACT_ID,
                    "path": GSE149487_PLUMAGE_PREFLIGHT_ARTIFACT_PATH,
                    "bytes": 7218,
                    "sha256": GSE149487_PLUMAGE_PREFLIGHT_ARTIFACT_SHA256,
                    "schema_version": "route_a_v3_gse149487_stop_before_data_preflight.v1",
                    "protocol_id": "ROUTE_A_V3_GSE149487_STOP_BEFORE_DATA_PREFLIGHT_V1",
                    "dataset_id": "GSE149487",
                    "recorded_at_utc": "2026-08-10T18:14:39Z",
                    "outcome": "NOT_READY_FOR_STUDY_QUALIFICATION",
                    "ready_for_study_qualification": False,
                    "authority_audit": {
                        "status": "PASS_CONFIG_ONLY_BINDING_VERIFIED",
                        "accepted_a0_base_commit": "fd722d5fa3c2538fce742b8942b1fb48e782760b",
                        "active_authority_commit": GSE149487_PLUMAGE_ACTIVE_AUTHORITY_COMMIT,
                        "active_amendment_decision_ids": ["V3-DEC-017", "V3-DEC-018"],
                        "implementation_commit": "d10a42a564ecac2af048b39c05cbc863ebdacd02",
                        "binding_commit": "aeecf0f043a94f2e5a738807c6d13d92f16e129f",
                        "git_head": "aeecf0f043a94f2e5a738807c6d13d92f16e129f",
                        "origin_branch_head": "aeecf0f043a94f2e5a738807c6d13d92f16e129f",
                        "branch": "routea-v3-a1-20260810",
                        "git_clean": True,
                        "binding_scheme": "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
                        "qualifier_authority_implementation_commit": "d10a42a564ecac2af048b39c05cbc863ebdacd02",
                        "qualifier_config_sha256": "893ed3b1eea194472b2ae3b5a975dc29c37c2a970e940420ef0e2e455780f04f",
                        "qualifier_code_sha256": {
                            "qualifier_script": GSE149487_PLUMAGE_QUALIFIER_SHA256,
                            "qualifier_test": GSE149487_PLUMAGE_TEST_SHA256,
                        },
                        "preflight_blob_sha256": {
                            "external_evidence_config": GSE149487_PLUMAGE_PREFLIGHT_CONFIG_SHA256,
                            "preflight_script": GSE149487_PLUMAGE_PREFLIGHT_SCRIPT_SHA256,
                            "preflight_test": GSE149487_PLUMAGE_PREFLIGHT_TEST_SHA256,
                        },
                        "authority_sha256": {
                            "asset_manifest": GSE149487_PLUMAGE_ASSET_MANIFEST_SHA256,
                            "contract": SOURCE_CONTRACT_SHA256,
                            "data_role_registry": "746439ef5d88d8167176d19e9c675746fdc78984a66f6f123f77f6ec49523030",
                            "decision_log": "a5b041fab24d9a4309603a085fa3fcab936d69a899285bfa752689a2ee5fd4fd",
                            "raw_asset_registry": "b27356ef790b3296ef0c535da7f3aeabd6812f364809017905a2375d7405f3d7",
                        },
                    },
                    "inventory_audit": {
                        "status": "PASS_METADATA_ONLY_STOP_BEFORE_DATA",
                        "entry_count": 22,
                        "payload_asset_count": 21,
                        "geo_raw_count": 18,
                        "supplement_count": 3,
                        "manifest_count": 1,
                        "filename_set_sha256": "2763a8e4347b47ba1980730a5abb5cf48797e27f01ff2f906e1a441657fa271d",
                        "name_size_binding_sha256": "d44f9018b4b8ecd560eb89bd3e82cfd86366054dd135b6545002997cee6896c2",
                        "total_lstat_bytes": 70038944,
                        "hash_reverification": "NOT_RUN_STOP_BEFORE_DATA",
                        "manifest_open_count": 0,
                        "payload_hash_count": 0,
                        "payload_open_count": 0,
                        "scientific_processing_count": 0,
                    },
                    "environment_audit": {
                        "status": "PASS_PREFLIGHT_ENVIRONMENT_ONLY",
                        "output_absent": True,
                        "failure_absent": True,
                        "claim_absent": True,
                    },
                    "external_evidence_audit": {
                        "exact_public_description_to_barcode_map": "BLOCKED",
                        "pre_outcome_mapping_timing": "BLOCKED",
                        "long_read_identity_method_status": "VERIFIED_METHOD_SURFACE_ONLY",
                        "paper_exact_executable_method_source": "BLOCKED",
                        "mann_whitney_implementation": "BLOCKED",
                        "fdr_family_definition": "BLOCKED",
                        "published_190_vs_180_status": "AUTHOR_ADJUDICATION_REQUIRED",
                        "all_21_assets_license_status": "BLOCKED",
                        "foundation_checkpoint_family_count": 4,
                        "foundation_exact_overlap_closed_count": 0,
                        "foundation_near_duplicate_closed_count": 0,
                        "foundation_label_exposure_closed_count": 0,
                    },
                    "counters": {
                        "canonical_record_count": 0,
                        "manifest_open_count": 0,
                        "model_selection_run_count": 0,
                        "payload_hash_count": 0,
                        "payload_open_count": 0,
                        "qualifier_execution_count": 0,
                        "scientific_processing_count": 0,
                        "training_run_count": 0,
                    },
                    "gate_truth": {
                        "ordinary_study_contribution": 0,
                        "a1_study_contribution": 0,
                        "true_a2_study_contribution": 0,
                        "canonical_record_count": 0,
                        "qualified": False,
                        "ready_for_study_qualification": False,
                        "training_allowed": False,
                        "model_selection_allowed": False,
                        "next_phase_authorized": False,
                    },
                    "blockers": GSE149487_PLUMAGE_PREFLIGHT_BLOCKERS,
                    "historical_r4_closure": {
                        "bundle_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_GSE149487_PLUMAGE_FULL_QUAL_20260810T131156P0800_a859166_R4",
                        "status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
                        "exact_blockers": GSE149487_PLUMAGE_PREFLIGHT_BLOCKERS,
                        "qualification_report_sha256": "19df844b55ef7b8dbf53ba3044a51132bdea1f0d1dfa6809a720a2a83a7030b3",
                        "sha256sums_sha256": "c72c63c2090052657beaa797e3ba3196200f8cbc3e9c5a97cf1a4a04a4db3631",
                        "publication_commit_sha256": "3149001644cf1b21db74021b12ca1e887977a9d0d13deff3b2f57b18e4b64ca4",
                        "reference_only_not_reopened": True,
                        "rerun_is_qualification_path": False,
                    },
                    "protocol_provenance": {
                        "path": "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810/configs/route_a_v3_gse149487_external_evidence_roots_v1.json",
                        "bytes": 14255,
                        "sha256": GSE149487_PLUMAGE_PREFLIGHT_CONFIG_SHA256,
                        "binding_model": "IMMUTABLE_STATIC_BLOB_BOUND_FROM_QUALIFIER_CONFIG",
                    },
                    "claim_boundary": (
                        "This stop-before-data preflight may verify repository authority, environment readiness, "
                        "and descriptor-bound filenames and sizes only. It never opens or hashes the manifest or "
                        "any of the 21 data payloads, never executes the qualifier, and never establishes study "
                        "qualification, training permission, model-selection permission, checkpoint-specific "
                        "foundation exposure, or a scientific claim."
                    ),
                    "metadata_only": True,
                    "changes_qualification_gate": False,
                    "establishes_scientific_claim": False,
                },
                path,
                issues,
                "A1_INTERIM_GSE149487_PREFLIGHT",
            )
        raw_preflight_blockers = [
            "EXACT_SRR_SAMPLE_ROLES_UNKNOWN",
            "CONTROL_REFERENCE_U_TO_T_NORMALIZATION_UNKNOWN",
            "SAM_TO_COUNT_PAIRED_HANDLING_UNKNOWN",
            "SAM_TO_COUNT_MULTIMAP_POLICY_UNKNOWN",
            "SAM_TO_COUNT_FLAG_POLICY_UNKNOWN",
            "SAM_TO_COUNT_MAPQ_POLICY_UNKNOWN",
            "SAM_TO_COUNT_DUPLICATE_POLICY_UNKNOWN",
            "SAM_TO_COUNT_IDENTICAL_REFERENCE_TIE_POLICY_UNKNOWN",
            "XTAIL_6772_INCLUSION_POLICY_UNKNOWN",
            "PAPER_6892_VS_AUDIT_6885_DENOMINATOR_CONFLICT_UNRESOLVED",
            "EDGER_EXACT_VERSION_UNKNOWN",
            "DESEQ2_EXACT_VERSION_UNKNOWN",
            "XTAIL_DEPENDENCY_LOCK_UNKNOWN",
            "XTAIL_RNG_SEED_AND_STATE_UNKNOWN",
            "PRJNA824033_VS_GSE200304_PRJNA824026_IDENTITY_CONFLICT_UNKNOWN",
            "AUTHOR_CODE_REDISTRIBUTION_PERMISSION_UNKNOWN",
            "RAW_FASTQ_REDISTRIBUTION_PERMISSION_UNKNOWN",
        ]
        raw_preflight_outer_git_binding = {
            "implementation_commit": "c0b72723cb923a880741ca1b82166e777dcbe928",
            "binding_commit": "19376c077afaae51e184dd5e833255ad5b1e98c6",
            "head_commit": "19376c077afaae51e184dd5e833255ad5b1e98c6",
            "origin_head_commit": "19376c077afaae51e184dd5e833255ad5b1e98c6",
            "protocol_config_sha256": "99c70338e7ad8eaa2a6f0d8525bfb2f58ee7d99ba43245b8a146935b52c3fa28",
            "production_script_sha256": "b79b99d62c91c47c180dfd4bd5b57e2b599e5486d38e4825093fc4efbd22e91d",
            "focused_test_sha256": "541b2434fcab1e0b082fd09a200e544b9029724b7c2f16c82f4593a085a6125f",
            "protocol_core_sha256": "88762faebad2d3dda93066861166248cb77ec1494a02bef5103a6aeb88d31e31",
            "implementation_commit_is_ancestor_of_binding_commit": True,
            "implementation_to_binding_diff_is_config_only": True,
            "declared_hashes_match_binding_commit_blobs": True,
            "worktree_and_index_clean": True,
        }
        raw_preflight_reference_truth = {
            "record_count": 13836,
            "type_counts": {"Control": 66, "Mutant": 6885, "WT": 6885},
            "unique_sequence_count": 13832,
            "identical_sequence_group_count": 4,
            "strict_250_acgt_record_count": 13824,
            "u_containing_record_count": 12,
            "u_containing_record_count_by_type": {
                "Control": 12,
                "Mutant": 0,
                "WT": 0,
            },
            "u_base_count": 596,
            "u_base_count_by_type": {"Control": 596, "Mutant": 0, "WT": 0},
            "primary_wt_mutant_record_count": 13770,
            "primary_wt_mutant_strict_250_acgt_record_count": 13770,
            "primary_wt_mutant_all_strict_250_acgt": True,
            "u_to_t_normalization_applied": False,
            "control_row_exclusion_applied": False,
        }
        expected_gse200304_lineage = {
            GSE200304_DEC019_SUCCESSOR_LINEAGE_ID: {
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "config_path": GSE200304_DEC019_SUCCESSOR_CONFIG_PATH,
                "initial_i_config_sha256": GSE200304_DEC019_SUCCESSOR_INITIAL_I_SHA256,
                "stable_core_projection_sha256": GSE200304_DEC019_SUCCESSOR_CORE_SHA256,
                "implementation_binding_lifecycle": "UNKNOWN_TO_CONFIG_ONLY_BOUND",
                "initial_implementation_binding_status": "UNKNOWN_NOT_ASSERTED",
                "dynamic_config_full_sha_in_static_manifest": False,
                "script_path": GSE200304_DEC019_SUCCESSOR_SCRIPT_PATH,
                "script_sha256": GSE200304_DEC019_SUCCESSOR_SCRIPT_SHA256,
                "focused_test_path": GSE200304_DEC019_SUCCESSOR_TEST_PATH,
                "focused_test_sha256": GSE200304_DEC019_SUCCESSOR_TEST_SHA256,
                "authority_only_not_study_qualification": True,
                "current_qualified": False,
                "current_ordinary_study_contribution": 0,
                "current_a1_study_contribution": 0,
                "current_true_a2_dense_study_contribution": 0,
                "current_canonical_record_count": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            },
            GSE200304_DEC019_LINEAGE_GATE_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
                    "DEC019_GATE_EVIDENCE_V1/"
                    "GSE200304_DEC019_CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE_GATE.json"
                ),
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "gate_id": "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE",
                "status": "PASS",
                "accepted": True,
                "aggregate_only": True,
                "bytes": 4367,
                "sha256": "ecdd5a59cfba6aa3307d4e0bd6becbf4d8e88817ab05b524f9c1e1e1f215dd59",
                "positive_input_fact": {"canonical_record_count": 6547},
                "positive_input_fact_is_not_final_qualification_or_materialization": True,
                "producer_lineage": {
                    "initial_implementation_commit": "08f10e1e6a4194eca4f3cd78c31ff191597a4e41",
                    "initial_binding_commit": "de35ce44d7744b89c8b52291343d9f1d6ea674a0",
                    "repair_implementation_commit": "b700e43178b44c4fa925d45466e5c8e4ba823f83",
                    "repair_binding_commit": "c764c721b364e19916ba66698552eee86563dbfe",
                    "config_path": GSE200304_DEC019_LINEAGE_CONFIG_PATH,
                    "config_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_LINEAGE_CONFIG_PATH
                    ],
                    "script_path": GSE200304_DEC019_LINEAGE_SCRIPT_PATH,
                    "script_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_LINEAGE_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE200304_DEC019_LINEAGE_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_LINEAGE_TEST_PATH
                    ],
                },
            },
            GSE200304_DEC019_NEGATIVE_GATE_PACK_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
                    "GSE200304_DEC019_NEGATIVE_GATE_PACK_V1"
                ),
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "status": "COMMITTED_ACCEPTED_EXACT8",
                "aggregate_only": True,
                "gate_record_count": 7,
                "exact_member_count": 8,
                "gate_status_counts": {
                    "BLOCKED": 3,
                    "UNKNOWN_NOT_ASSERTED": 2,
                    "NOT_RUN": 2,
                },
                "producer_lineage": {
                    "initial_implementation_commit": "01f3f818937c97d9804b94413f54d3e654e35120",
                    "initial_binding_commit": "a677454ec78ad5df4a5880444b0764d42676025a",
                    "repair_implementation_commit": "8da6abaeddfadeca5542b997e7c0ba6501c1f1f7",
                    "repair_binding_commit": "981f001d6290cbc1b9b48a55d8e51a963e45d785",
                    "config_path": GSE200304_DEC019_NEGATIVE_CONFIG_PATH,
                    "config_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_NEGATIVE_CONFIG_PATH
                    ],
                    "script_path": GSE200304_DEC019_NEGATIVE_SCRIPT_PATH,
                    "script_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_NEGATIVE_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE200304_DEC019_NEGATIVE_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_NEGATIVE_TEST_PATH
                    ],
                },
            },
            GSE200304_DEC019_ADJUDICATION_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3"
                ),
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
                "publication_state": "COMMITTED_ACCEPTED",
                "aggregate_only": True,
                "input_gate_count": 8,
                "input_status_counts": GSE200304_DEC019_POST_ADJUDICATION_INPUT_STATUS_COUNTS,
                "blocker_count": 7,
                "blockers": GSE200304_DEC019_POST_ADJUDICATION_BLOCKERS,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "positive_input_canonical_record_count": 6547,
                "canonical_record_count": 0,
                "positive_input_fact_is_not_final_canonical_materialization": True,
                "qualified": False,
                "canonical_materialization_allowed": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "dynamic_config": {
                    "path": GSE200304_DEC019_V3_CONFIG_PATH,
                    "sha256": GSE200304_DEC019_V3_CONFIG_SHA256,
                    "config_core_sha256": GSE200304_DEC019_V3_CONFIG_CORE_SHA256,
                    "descriptor_set_sha256": GSE200304_DEC019_V3_DESCRIPTOR_SET_SHA256,
                    "exact_full_sha_in_static_manifest": False,
                },
                "consumer_lineage": {
                    "historical_implementation_commit": "4e200ed4048d5b112c6ac324d2376e8de1441419",
                    "historical_binding_commit": "e495c7ec5b6f00f14a18a4ffe0c5a6f2173bf2d8",
                    "commitment_schema_implementation_commit": "86d16c181fc9deaf83597da9c1523e4fea9c7493",
                    "commitment_schema_binding_commit": "0ad41b93eb3e145f5c509a0c2ec94988e147ac97",
                    "descriptor_d1_commit": "c61f3d06ab6cfbc54ff562738d95ba902865b54f",
                    "predecessor_i3_commit": "e829464d6ea1953b7a859ba5506946b9cb8e6384",
                    "implementation_i4_commit": "6d103877bbfb8e1196bfc22890bb239dcb87c3c8",
                    "binding_b4_commit": "6c42d8e1d75f70906afb7cde5704669b2c8ab6f7",
                    "descriptor_d2_commit": "c278f29a18b7858c85686fcec3857a992fd07d5f",
                    "script_path": GSE200304_DEC019_V3_SCRIPT_PATH,
                    "script_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_V3_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE200304_DEC019_V3_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_V3_TEST_PATH
                    ],
                },
            },
            "a1_public_qualifiers_sync_v1": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/A1_PUBLIC_QUALIFIERS_SYNC_V1.json",
                "sha256": "22eac457a5ccea5272b9e1b9ff4ded845c79c89449209f28d9aaa510f2ab59f5",
                "event_id": "A1-EVT-031",
                "event_at": "2026-08-10T13:26:47+08:00",
                "lineage_role": "PRIOR_PUBLIC_QUALIFIER_EVIDENCE_SYNC_WITH_GATE_UNCHANGED",
                "qualified_independent_ordinary_studies": 0,
                "qualified_a1_studies": 0,
                "qualified_a2_dense_studies": 0,
                "training_started": False,
                "next_phase_authorized": False,
            },
            "gse200304_public_asset_bundle": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304/GSE200304_PUBLIC_ASSETS_20260810T143731P0800",
                "acquisition_manifest_sha256": "8318990d9e3b6a0e6265bf9d1e8bc20f56f0ecfd994e83d279e733258642100c",
                "sha256sums_sha256": "20da85cd34f0574829392b5de1d7c48cc9782219847f56ccc07dffd579d79f15",
                "terminal_marker_sha256": "4742508195f28bf8c7ab1f7cb8bb0b68c32304f31b19c8f8979d098fa75786a5",
                "status": "PRESENT_IN_SEPARATE_COMMITTED_BUNDLE_NOT_INTEGRATED",
                "used_by_current_qualifier": False,
            },
            "gse200304_ena_fastq_manifest_bundle": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304/GSE200304_ENA_FASTQ_MANIFEST_20260810T145631P0800",
                "canonical_tsv_sha256": "22cd317d961d07036cb2dad19555b5c2423671c33a76badeb7b325847ee68d7b",
                "summary_sha256": "f92f944c825a255f3f1fb50f48cbf0e701980b7895101c1a2a6699d4b190e1e4",
                "terminal_marker_sha256": "d3eed4a9408543c77f47aa2a0d8cff59ebfe863c1e3c2d0bb2324d7910d6014b",
                "official_run_count": 24,
                "paired_fastq_object_count": 48,
                "declared_total_fastq_bytes": 12738938976,
                "fastq_body_download_count": 0,
                "fastq_md5_local_recomputation_status": "NOT_RUN",
                "official_metadata_and_object_lengths_status": "VERIFIED_48_OBJECTS",
                "metadata_only": True,
                "contains_fastq_body_payload": False,
                "status": "PRESENT_IN_SEPARATE_COMMITTED_BUNDLE_NOT_CONSUMED",
                "used_by_current_qualifier": False,
            },
            "gse200304_fastq_acquisition_v1": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304/GSE200304_FASTQ_ACQUISITION_20260810T165023P0800_e24d722",
                "target_subseries_accession": "GSE200302",
                "superseries_accession": "GSE200304",
                "bioproject_accession": "PRJNA824033",
                "publication_status": "FASTQ_ACQUISITION_COMMITTED",
                "implementation_commit": "7683cad77250fcb986d83a903d3e94b2eaea75de",
                "binding_commit": "e24d7225aecf098e7cddaa7a246e8bfea1a0730d",
                "implementation_script_sha256": "1b0d1c5db7e32475fb835cadb5d1805415447a490a1a83840bcb6e8518fa6340",
                "protocol_sha256": "e589a9ceccd469ee22eaddcf2f4f05e10a2a66c138a38ba30ee6795435d8f96a",
                "terminal_marker_sha256": "c0956cc8ce3e038ecc735a079fd53869376d5e6db42e46246f036446e03222ca",
                "verified_file_count": 48,
                "verified_run_count": 24,
                "verified_total_bytes": 12738938976,
                "repository_md5_verified_count": 48,
                "local_sha256_recorded_count": 48,
                "terminal_member_set_count_excluding_marker_and_operational_files": 100,
                "raw_fastq_body_present": True,
                "aggregate_only_ledger_entry": True,
                "paper_native_count_reconstruction_status": "NOT_RUN",
                "paper_native_xtail_replay_status": "NOT_RUN",
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            },
            "gse200304_fastq_independent_consumer_verification_v1": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_FASTQ_CONSUMER_VERIFY_20260810T191502P0800_e24d722",
                "publication_status": "INDEPENDENT_CONSUMER_VERIFICATION_COMMITTED",
                "first_descendant_head_attempt_status": "FAIL_CLOSED_REPLAY_ENVIRONMENT_BINDING_MISMATCH",
                "first_descendant_head": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
                "exact_producer_binding_attempt_status": "ALREADY_COMMITTED_VERIFIED",
                "exact_producer_binding_head": "e24d7225aecf098e7cddaa7a246e8bfea1a0730d",
                "producer_terminal_marker_sha256": "c0956cc8ce3e038ecc735a079fd53869376d5e6db42e46246f036446e03222ca",
                "verified_file_count": 48,
                "verified_run_count": 24,
                "verified_total_bytes": 12738938976,
                "repository_md5_verified_count": 48,
                "local_sha256_verified_count": 48,
                "terminal_member_closure_verified": True,
                "acceptance_scope": "TRANSPORT_AND_ACQUISITION_INTEGRITY_ONLY",
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            },
            "gse200304_qualifier_protocol": {
                "path": "configs/route_a_v3_gse200304_a1_qualification.json",
                "sha256": "0c7328735edbeed90ae04d5032b268c3b92c71e03031aa040bc03c2743b9e0a7",
                "qualifier_path": "scripts/route_a_v3/qualify_gse200304_a1.py",
                "qualifier_sha256": "49950a460079924d5e5b98b7a49bf2dc378a1cf82cba633d19b2bff0b52c9944",
                "focused_test_path": "tests/route_a_v3/test_qualify_gse200304_a1.py",
                "focused_test_sha256": "b21b0f497b4e2b9857b70d4ff83f2287a12b4f0944080f40fb24721682b15269",
                "implementation_commit": "b9697ef82ccb30f1d76a2baed1b3207f9ea056a6",
                "binding_commit": "46c608b219590cf844060a85ba0983bcf4c5a471",
                "qualification_execution_commit": "46c608b219590cf844060a85ba0983bcf4c5a471",
                "implementation_binding_status": "BOUND",
                "canonical_protocol_trust_root_closed": True,
                "model_results_may_change_protocol": False,
            },
            "gse200304_gap_qualification_attempt_001_failure": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_GAP_QUALIFICATION_20260810T155024P0800_bf14584",
                "bundled_status": "FAIL_CLOSED_BEFORE_SUCCESS_BUNDLE_PUBLICATION",
                "failure_report_bytes": 578,
                "failure_report_sha256": "248cdea9742d449ad3f5735b99cf2842477afa713d58733daa69b68ed1039bbf",
                "sha256sums_bytes": 86,
                "sha256sums_sha256": "c92383cd6b5e4426314aca7c6eecefcee9e26c87856c8f76d52677af10dd86da",
                "terminal_marker_bytes": 870,
                "terminal_marker_sha256": "55371492c30cfffd90bf091f229b34e54f17a99229ec23c2be8ed4d7bfbb9f7d",
                "preserved_without_overwrite": True,
                "diagnostic_reason": "NFS_STALE_PREOPEN_PARENT_METADATA_FALSE_REJECTION",
                "diagnostic_reason_provenance": "READ_ONLY_DIAGNOSTIC_REPLAY_NOT_BUNDLE_CLAIM",
            },
            "gse200304_gap_qualification_attempt_002_failure": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_GAP_QUALIFICATION_20260810T160803P0800_841b275",
                "bundled_status": "FAIL_CLOSED_BEFORE_SUCCESS_BUNDLE_PUBLICATION",
                "failure_report_bytes": 574,
                "failure_report_sha256": "fa8ae6fe50f8b2a493322b9c3902e2800e3fe5065caa6a4008a7f7cd2cf3b31f",
                "sha256sums_bytes": 86,
                "sha256sums_sha256": "c9a1321f6e93fe5992de45171084d916fb74a18aebd51690d3fcedf264abb18c",
                "terminal_marker_bytes": 870,
                "terminal_marker_sha256": "6b580898a8d0260e964d791bb472b6b8dabc300aa0459ec92179002e0dc3f4f2",
                "preserved_without_overwrite": True,
                "diagnostic_reason": "CONTROL_NON_ACGT_ROWS_INCORRECTLY_SUBJECTED_TO_PAIR_ALPHABET_GATE",
                "diagnostic_aggregate_control_non_acgt_count": 41,
                "diagnostic_reason_provenance": "READ_ONLY_DIAGNOSTIC_REPLAY_NOT_BUNDLE_CLAIM",
            },
            "gse200304_gap_qualification_attempt_003_failure": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_GAP_QUALIFICATION_20260810T162027P0800_8bb2106",
                "bundled_status": "FAIL_CLOSED_BEFORE_SUCCESS_BUNDLE_PUBLICATION",
                "failure_report_bytes": 574,
                "failure_report_sha256": "fa8ae6fe50f8b2a493322b9c3902e2800e3fe5065caa6a4008a7f7cd2cf3b31f",
                "sha256sums_bytes": 86,
                "sha256sums_sha256": "c9a1321f6e93fe5992de45171084d916fb74a18aebd51690d3fcedf264abb18c",
                "terminal_marker_bytes": 870,
                "terminal_marker_sha256": "24c6ae6890c0a061627b10a1207b9c0ba268d50c74a2a300e8527c3c30c5b764",
                "preserved_without_overwrite": True,
                "diagnostic_reason": "CONTROL_ID_INCORRECTLY_REQUIRED_TO_EQUAL_INDEPENDENT_CONTROL_MERGED_ID",
                "diagnostic_reason_provenance": "READ_ONLY_DIAGNOSTIC_REPLAY_NOT_BUNDLE_CLAIM",
            },
            "gse200304_gap_qualification_v1": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_GAP_QUALIFICATION_20260810T163429P0800_46c608b",
                "execution_outcome": "ENGINEERING_SUCCESS_BLOCKED_NOT_QUALIFIED",
                "qualification_status": "BLOCKED_NOT_QUALIFIED",
                "sha256sums_sha256": "f5c3bf069bb22878ee0b99d51810571d3b00bb37e03c3da0ff43138a650a0914",
                "qualification_report_sha256": "f2aaa99443c1df2eba30698ba46574974189102b8f65d0712286f56a85ea7e3f",
                "input_integrity_audit_sha256": "712451293571250ac196df8a190ab9ee82dc0729db59ef6aa61655c47e136cb3",
                "mechanical_audit_sha256": "142c88fa6e6db0ba73431cf0fd790e85f179a10612c906ad1ce651b9e4695ec9",
                "terminal_marker_sha256": "803042c2af9e72e4355e6decb25c3a349d03d961d987633a337fff41e3b58d1e",
                "aggregate_only": True,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "canonical_record_count": 0,
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
            },
            "gse200304_raw_replay_preflight_attempt_001_failure": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_RAW_REPLAY_PREFLIGHT_20260810T202615P0800_534882a",
                "execution_outcome": "FAIL_CLOSED",
                "status": "FAIL_CLOSED",
                "failure_code": "REFERENCE_AGGREGATE_INVALID",
                "aggregate_only": True,
                "raw_payload_included": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "canonical_record_count": 0,
                "qualified": False,
                "phase_complete": False,
                "training_started": False,
                "model_selection_started": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "success_bundle_published": False,
                "terminal_marker_written_last": True,
                "preserved_without_overwrite": True,
            },
            "gse200304_raw_replay_preflight_v1": {
                "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_RAW_REPLAY_PREFLIGHT_20260810T205200P0800_19376c0",
                "target_subseries_accession": "GSE200302",
                "superseries_accession": "GSE200304",
                "execution_outcome": "BLOCKED_PRE_EXECUTION_WITH_EVIDENCE",
                "status": "BLOCKED_PRE_EXECUTION_WITH_EVIDENCE",
                "aggregate_only": True,
                "hard_unknown_blocker_count": 17,
                "hard_unknown_blockers": raw_preflight_blockers,
                "fastq_body_read_count_by_preflight": 0,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "canonical_record_count": 0,
                "qualified": False,
                "phase_complete": False,
                "training_started": False,
                "model_selection_started": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "outer_git_binding": raw_preflight_outer_git_binding,
                "reference_aggregate_truth": raw_preflight_reference_truth,
                "claim_boundary": "AGGREGATE_ONLY_P0_PREFLIGHT_NOT_COUNT_REPLAY_XTAIL_REPLAY_QUALIFICATION_TRAINING_OR_PHASE_UNLOCK",
                "targeted_test_status": "PASS",
                "targeted_test_passed": 57,
                "targeted_test_failed": 0,
                "terminal_marker_written_last": True,
            },
            "gse200302_srr_role_authority_v1": {
                "path": GSE200302_ROLE_ARTIFACT_ROOT,
                "dataset_id": "GSE200304",
                "target_subseries_accession": "GSE200302",
                "bioproject_accession": "PRJNA824033",
                "record_type": "OFFICIAL_SRR_ROLE_AUTHORITY",
                "authority_level": "OFFICIAL_METADATA_ROLE_AUTHORITY_ONLY",
                "publication_status": "COMMITTED_AND_ACCEPTED",
                "status": "EXACT_OFFICIAL_SRR_ROLE_AUTHORITY_CLOSED",
                "bundle_digest": GSE200302_ROLE_BUNDLE_DIGEST,
                "mapping_row_count": 24,
                "experiment_join_row_count": 24,
                "measurement_families": ["High_Poly", "Low_Poly", "pDNA", "Total_RNA"],
                "replicates": [1, 2, 3, 4, 5, 6],
                "forbidden_80s_alias_count": 0,
                "prior_blocker": "EXACT_SRR_SAMPLE_ROLES_UNKNOWN",
                "prior_blocker_status": "CLOSED",
                "replacement_blocker": "REQUIRED_80S_ROLE_AUTHORITY_ABSENT",
                "replacement_blocker_status": "OPEN",
                "role_grid_status": "CONFLICT_WITH_CURRENT_80S_EXPECTATION",
                "pdna_may_substitute_for_80s_rna": False,
                "current_effective_raw_replay_blocker_count": 17,
                "publication_producer_protocol_config_path": GSE200302_ROLE_CONFIG_PATH,
                "publication_producer_protocol_config_sha256": "c39335b50bf9832f336ca62fdf839351c58f5fa050eaf36c62f050171cbf24b2",
                "publication_producer_script_path": GSE200302_ROLE_BUILDER_PATH,
                "publication_producer_script_sha256": "b057e4346fa9473a6da29d5f229761f2d65e76db7b510ba6477bd40dbe49183c",
                "publication_producer_focused_test_path": GSE200302_ROLE_TEST_PATH,
                "publication_producer_focused_test_sha256": "346e239c6628388b31ef3f8ea55b295f66347811a86fb28ddbdd6f268d894edd",
                "publication_producer_protocol_core_sha256": GSE200302_ROLE_PROTOCOL_CORE_SHA256,
                "artifact_intrinsic_model_selection_field_present": False,
                "artifact_intrinsic_model_selection_status": "NOT_ENCODED_FAIL_CLOSED",
                "implementation_commit": "d042d7c1706a80821a19b78334985441bcf6eb86",
                "binding_commit": "e3b724d00a9e5263b99475b9744fc0bb68a3ab67",
                "implementation_to_binding_diff_is_config_only": True,
                "commits_pushed_clean": True,
                "independent_consumer_status": "ACCEPTED",
                "runtime_sync_status": "PENDING_NO_EVT_035",
                "metadata_only": True,
                "fastq_body_read_count": 0,
                "contains_sequence_values": False,
                "contains_fastq_body_payload": False,
                "contains_raw_label_values": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "canonical_record_count": 0,
                "qualified": False,
                "training_authorized": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "terminal_marker_written_last": True,
            },
            "gse200304_published_endpoint_a1_v1": GSE200304_PUBLISHED_ENDPOINT_EXPECTED_RECORD,
        }
        expected_relevant_lineage_ids = set(expected_gse200304_lineage)
        observed_relevant_lineage_ids: set[str] = set()
        for lineage_id, record in lineage.items():
            if not isinstance(lineage_id, str):
                continue
            record_path = record.get("path", "") if isinstance(record, Mapping) else ""
            record_dataset = record.get("dataset_id") if isinstance(record, Mapping) else None
            if (
                lineage_id == "a1_public_qualifiers_sync_v1"
                or lineage_id.startswith("gse200304_")
                or record_dataset == "GSE200304"
                or "gse200304" in str(record_path).lower()
            ):
                observed_relevant_lineage_ids.add(lineage_id)
        if observed_relevant_lineage_ids != expected_relevant_lineage_ids:
            _issue(
                issues,
                "A1_INTERIM_GSE200304_LINEAGE_ID_SET",
                path,
                "GSE200304 lineage IDs must be exactly the closed accepted set; "
                f"got {sorted(observed_relevant_lineage_ids)!r}, "
                f"expected {sorted(expected_relevant_lineage_ids)!r}",
            )

        def _closed_files(
            root: str,
            members: Sequence[tuple[str, int, str]],
        ) -> list[dict[str, Any]]:
            return [
                {"path": f"{root}/{name}", "bytes": size, "sha256": digest}
                for name, size, digest in members
            ]

        public_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304/GSE200304_PUBLIC_ASSETS_20260810T143731P0800"
        ena_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304/GSE200304_ENA_FASTQ_MANIFEST_20260810T145631P0800"
        fastq_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304/GSE200304_FASTQ_ACQUISITION_20260810T165023P0800_e24d722"
        consumer_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_FASTQ_CONSUMER_VERIFY_20260810T191502P0800_e24d722"
        failure_001_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_GAP_QUALIFICATION_20260810T155024P0800_bf14584"
        failure_002_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_GAP_QUALIFICATION_20260810T160803P0800_841b275"
        failure_003_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_GAP_QUALIFICATION_20260810T162027P0800_8bb2106"
        final_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_GAP_QUALIFICATION_20260810T163429P0800_46c608b"
        raw_preflight_failure_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_RAW_REPLAY_PREFLIGHT_20260810T202615P0800_534882a"
        raw_preflight_final_root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_RAW_REPLAY_PREFLIGHT_20260810T205200P0800_19376c0"
        role_authority_root = GSE200302_ROLE_ARTIFACT_ROOT
        negative_gate_root = (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
            "GSE200304_DEC019_NEGATIVE_GATE_PACK_V1"
        )
        adjudication_v3_root = (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3"
        )
        expected_closed_files = {
            "gse200304_public_asset_bundle": _closed_files(
                public_root,
                (
                    ("ASSET_ACQUISITION_MANIFEST.json", 6426, "8318990d9e3b6a0e6265bf9d1e8bc20f56f0ecfd994e83d279e733258642100c"),
                    ("NCBI_PRJNA824033_RUNINFO.csv", 12042, "34bcedafebc41ee9ccd79483f331b62f2443df31d12691abc0a961a7201848f4"),
                    ("NIHMS1928233-supplement-3.csv", 7323186, "812f3c983cb7c4f473200741ffd6d73bcab911c9e354934542e018e7b0cf8a6d"),
                    ("NIHMS1928233-supplement-4.xlsx", 864791, "ec2aab60fcb0be87f2bcc1b1a5a1f786b23bb429edc9851a4034a3e8983dfa08"),
                    ("PUBLICATION_COMMIT.json", 1095, "4742508195f28bf8c7ab1f7cb8bb0b68c32304f31b19c8f8979d098fa75786a5"),
                    ("SHA256SUMS", 491, "20da85cd34f0574829392b5de1d7c48cc9782219847f56ccc07dffd579d79f15"),
                    ("slschuster_3UTRMutationalMPRA-v1.2.zip", 46209, "1c1b1979c1d5bd7fefa54e80a59f982228d0f1498eb0cff2883b753ee5eb0ae4"),
                ),
            ),
            "gse200304_ena_fastq_manifest_bundle": _closed_files(
                ena_root,
                (
                    ("ENA_PRJNA824033_FASTQ_FILES.canonical.tsv", 10388, "22cd317d961d07036cb2dad19555b5c2423671c33a76badeb7b325847ee68d7b"),
                    ("ENA_PRJNA824033_FASTQ_FILE_REPORT.source.tsv", 5998, "c4a0b6152ec2a3480f280d8498345196d5095ec54967525463fa81961f0f4ea1"),
                    ("MANIFEST_SUMMARY.json", 3135, "f92f944c825a255f3f1fb50f48cbf0e701980b7895101c1a2a6699d4b190e1e4"),
                    ("PUBLICATION_COMMIT.json", 1578, "d3eed4a9408543c77f47aa2a0d8cff59ebfe863c1e3c2d0bb2324d7910d6014b"),
                    ("SHA256SUMS", 307, "5217d3bd5494908d1886c6a00719014f4726ab3b61efde43184c2e475c6fdc78"),
                ),
            ),
            "gse200304_fastq_acquisition_v1": _closed_files(
                fastq_root,
                (
                    ("ACQUISITION_BINDING.json", 1584, "3d0681caaf864f18c9ae482b38e9e19a8cd09f0c326a76f3780623df84ab16cb"),
                    ("ACQUISITION_STATUS.json", 1418, "178708ad6f6d9de91b8c89aba63359822b274330d4050b574170eaec234ed4fd"),
                    ("FASTQ_INTEGRITY_MANIFEST.json", 20339, "87417e078dc6f47bec5404430a69ca72f18c03066ae2e24300d3a0642fbce167"),
                    ("PUBLICATION_COMMIT.json", 15875, "c0956cc8ce3e038ecc735a079fd53869376d5e6db42e46246f036446e03222ca"),
                    ("SHA256SUMS", 9493, "c20fb56dd116817db1aa1868da318e8ef4c038a9828d50004f88560e1b6cee3d"),
                ),
            ),
            "gse200304_fastq_independent_consumer_verification_v1": _closed_files(
                consumer_root,
                (
                    ("VERIFICATION_RECORD.json", 5539, "d316cfa617348457ba1f6a15c284c599a0b422dae85ea3f810cb5476806fb58e"),
                    ("PUBLICATION_COMMIT.json", 1472, "0189119470a9379c97b16533857e6c2f67dad6472509dd25247490e809f29e30"),
                    ("SHA256SUMS", 91, "968b11b3691b552d567d7461bf970871a7d120c231576df923ee28818c239b25"),
                ),
            ),
            "gse200304_gap_qualification_attempt_001_failure": _closed_files(
                failure_001_root,
                (
                    ("FAILURE_REPORT.json", 578, "248cdea9742d449ad3f5735b99cf2842477afa713d58733daa69b68ed1039bbf"),
                    ("PUBLICATION_COMMIT.json", 870, "55371492c30cfffd90bf091f229b34e54f17a99229ec23c2be8ed4d7bfbb9f7d"),
                    ("SHA256SUMS", 86, "c92383cd6b5e4426314aca7c6eecefcee9e26c87856c8f76d52677af10dd86da"),
                ),
            ),
            "gse200304_gap_qualification_attempt_002_failure": _closed_files(
                failure_002_root,
                (
                    ("FAILURE_REPORT.json", 574, "fa8ae6fe50f8b2a493322b9c3902e2800e3fe5065caa6a4008a7f7cd2cf3b31f"),
                    ("PUBLICATION_COMMIT.json", 870, "6b580898a8d0260e964d791bb472b6b8dabc300aa0459ec92179002e0dc3f4f2"),
                    ("SHA256SUMS", 86, "c9a1321f6e93fe5992de45171084d916fb74a18aebd51690d3fcedf264abb18c"),
                ),
            ),
            "gse200304_gap_qualification_attempt_003_failure": _closed_files(
                failure_003_root,
                (
                    ("FAILURE_REPORT.json", 574, "fa8ae6fe50f8b2a493322b9c3902e2800e3fe5065caa6a4008a7f7cd2cf3b31f"),
                    ("PUBLICATION_COMMIT.json", 870, "24c6ae6890c0a061627b10a1207b9c0ba268d50c74a2a300e8527c3c30c5b764"),
                    ("SHA256SUMS", 86, "c9a1321f6e93fe5992de45171084d916fb74a18aebd51690d3fcedf264abb18c"),
                ),
            ),
            "gse200304_gap_qualification_v1": _closed_files(
                final_root,
                (
                    ("INPUT_INTEGRITY_AUDIT.json", 3476, "712451293571250ac196df8a190ab9ee82dc0729db59ef6aa61655c47e136cb3"),
                    ("MECHANICAL_AUDIT.json", 5345, "142c88fa6e6db0ba73431cf0fd790e85f179a10612c906ad1ce651b9e4695ec9"),
                    ("PUBLICATION_COMMIT.json", 969, "803042c2af9e72e4355e6decb25c3a349d03d961d987633a337fff41e3b58d1e"),
                    ("QUALIFICATION_REPORT.json", 8080, "f2aaa99443c1df2eba30698ba46574974189102b8f65d0712286f56a85ea7e3f"),
                    ("SHA256SUMS", 273, "f5c3bf069bb22878ee0b99d51810571d3b00bb37e03c3da0ff43138a650a0914"),
                ),
            ),
            "gse200304_raw_replay_preflight_attempt_001_failure": _closed_files(
                raw_preflight_failure_root,
                (
                    ("FAILURE.json", 599, "80ad8eb024184eabfdfad84587377b08c2ecdef8edb3a18d74344f1c6724a92e"),
                    ("SHA256SUMS", 79, "82df59e4abcbfd09e6c79efacfd75990bd212f7bc3efaedce4253ffc18d1342d"),
                    ("PUBLICATION_COMMIT.json", 1104, "5e9ce0a7403f186199ef59ce167db2cfb6c7c04e199231a174ee1e812821dd30"),
                ),
            ),
            "gse200304_raw_replay_preflight_v1": _closed_files(
                raw_preflight_final_root,
                (
                    ("PREFLIGHT.json", 6907, "8b592b816b5c981a774e2d58364b424271eb28d2c3e4a16a300dac2f926e0f4f"),
                    ("SHA256SUMS", 81, "410398cb558115e41f6332d03f0acfde4ba43162a1dc1944d8205fe3b6ffdeae"),
                    ("PUBLICATION_COMMIT.json", 1156, "dde5847660e34d90cedcf38f69c619c2a8ba8f8470ddacdded3f7a290e5837b1"),
                ),
            ),
            "gse200302_srr_role_authority_v1": _closed_files(
                role_authority_root,
                (
                    ("GSE200302_SRR_ROLE_AUTHORITY.tsv", 1200, "f69fa9af134b421439a2a90c09c75cb300e2e833de143d829bafe4ef7a1d094d"),
                    ("GSE200302_SRR_SRX_ROLE_JOIN_AUTHORITY.tsv", 1509, "6684f3d1fde3666ac4bf07ff0aa29bd9b47240b5d6708fd8483aaa1d88a64ae4"),
                    ("ROLE_AUTHORITY.json", 4100, "4623dd996c927525aa4dbee6dc7b9bd0e87c0d5628ac65811f15affd67777dbc"),
                    ("SHA256SUMS", 293, "909d3ed6c15dea632622229aa3bae9840575000a8caf30e39165c1029ecfdec4"),
                    ("PUBLICATION_COMMIT.json", 2106, "35e8884db5b8e5734300e391715b87304766dbb8e4f888bf51ead5be8b5f83b3"),
                ),
            ),
            "gse200304_published_endpoint_a1_v1": GSE200304_PUBLISHED_ENDPOINT_EXPECTED_FILES,
            GSE200304_DEC019_NEGATIVE_GATE_PACK_LINEAGE_ID: _closed_files(
                negative_gate_root,
                (
                    (
                        "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE.json",
                        3921,
                        "2db95ec41d5e76a77d17104076c5823f5cc1f8646260964e92166f0faf440950",
                    ),
                    (
                        "GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_GATE.json",
                        4022,
                        "a2b4dd52f0fbe4cd31324d4b760b9fee104f719d67d4f6834b6c1ea3adffaf9e",
                    ),
                    (
                        "GSE200304_DEC019_CHECKPOINT_SPECIFIC_EXPOSURE_GATE.json",
                        3997,
                        "33b659a86eb8058adad922649b3c89a78ad65a61014826cd3ab28f1c4a6214f9",
                    ),
                    (
                        "GSE200304_DEC019_LICENSE_RIGHTS_GATE.json",
                        3919,
                        "7556db47e8b3ec9cdc0a7d795161cf4ded8679d0c04644005a475753c482aa28",
                    ),
                    (
                        "GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE.json",
                        4004,
                        "b3be463eb502b5b8a2b171ea763cffa8eba1449bf2c380aa3d16435733e35821",
                    ),
                    (
                        "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE.json",
                        3908,
                        "817165a2e9ea2e01efae2374a606334375b9ec3afafae665fefacf0c6779fc95",
                    ),
                    (
                        "GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_GATE.json",
                        3977,
                        "235683969801d375a59a0bde56af2a448afd2a56cd5224cbede1320c2c15026b",
                    ),
                    (
                        "PUBLICATION_COMMIT.json",
                        1256,
                        "bf2cad9cfdc3b6dfc537bc0bf302ad79e332c5cf5d341e8dfd7fa64675b423c4",
                    ),
                ),
            ),
            GSE200304_DEC019_ADJUDICATION_LINEAGE_ID: _closed_files(
                adjudication_v3_root,
                (
                    (
                        "ADJUDICATION_REPORT.json",
                        2486,
                        "62d2391bc61533f0374195605ba2a1e4ba3385f997b233087632f53901ae2de3",
                    ),
                    (
                        "INPUT_EVIDENCE_AUDIT.json",
                        3005,
                        "d84763040507e34f9c5913075ee306b4432a75389560bdeb85c9b6ba088809e6",
                    ),
                    (
                        "SHA256SUMS",
                        183,
                        "f856f14508db876aec3438a069d2aa0aacc92989153a33a05d54b59b4d256477",
                    ),
                    (
                        "PUBLICATION_COMMIT.json",
                        1055,
                        "6fb7c07c493ace456d4c4918fdc270986796c32e73e23f6901e34352d4bdf310",
                    ),
                ),
            ),
        }
        for lineage_id, expected_fields in expected_gse200304_lineage.items():
            record = lineage.get(lineage_id)
            if not isinstance(record, Mapping):
                _issue(
                    issues,
                    "A1_INTERIM_GSE200304_LINEAGE",
                    path,
                    f"{lineage_id} must be a mapping",
                )
                continue
            expected_record_keys = set(expected_fields)
            if lineage_id in expected_closed_files:
                expected_record_keys.add("files")
            if set(record) != expected_record_keys:
                _issue(
                    issues,
                    "A1_INTERIM_GSE200304_LINEAGE_KEYS",
                    path,
                    f"{lineage_id} keys must be exactly {sorted(expected_record_keys)!r}",
                )
            for key, value in expected_fields.items():
                _expect(
                    record,
                    key,
                    value,
                    path,
                    issues,
                    "A1_INTERIM_GSE200304_LINEAGE",
                )
            if lineage_id in expected_closed_files:
                _expect(
                    record,
                    "files",
                    expected_closed_files[lineage_id],
                    path,
                    issues,
                    "A1_INTERIM_GSE200304_CLOSED_FILES",
                )
        for relative, expected_sha256 in (
            (
                GSE200304_PUBLISHED_ENDPOINT_CONFIG_PATH,
                "92fc3a3859f7a8949ace67fa4b03a14e8ad102eb257d4f95cace01ea535b41af",
            ),
            (
                GSE200304_PUBLISHED_ENDPOINT_SCRIPT_PATH,
                "687268524c7426eb4d3d450e71d13c7c478372162e0c084ffe90c8bb12764308",
            ),
            (
                GSE200304_PUBLISHED_ENDPOINT_TEST_PATH,
                "173cad716fbdb2590e82ea54a91776ef61e7ab9eb7b596b694b5aa8609d44ad0",
            ),
        ):
            try:
                actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
            except (FileNotFoundError, ValueError) as exc:
                _issue(
                    issues,
                    "A1_INTERIM_GSE200304_PUBLISHED_ENDPOINT_BINDING",
                    relative,
                    str(exc),
                )
            else:
                if actual_sha256 != expected_sha256:
                    _issue(
                        issues,
                        "A1_INTERIM_GSE200304_PUBLISHED_ENDPOINT_BINDING",
                        relative,
                        f"current bytes hash {actual_sha256} must remain {expected_sha256}",
                    )

    summary = interim.get("dataset_boundary_summary")
    if not isinstance(summary, Mapping):
        _issue(issues, "A1_INTERIM_DATASET_BOUNDARY", path, "dataset_boundary_summary must be a mapping")
    else:
        expected_summary_keys = {
            "evidence_ref",
            "GSE114002",
            "GSE149487",
            "GSE145046",
            "GSE200304",
            "three_utr_candidates",
            "GSE207584",
            "qualified_a2_dense_neighborhoods",
        }
        if set(summary) != expected_summary_keys:
            _issue(
                issues,
                "A1_INTERIM_DATASET_BOUNDARY",
                path,
                "dataset boundary keys must remain the exact accepted set",
            )
        evidence_ref = summary.get("evidence_ref")
        if not isinstance(evidence_ref, Mapping):
            _issue(issues, "A1_INTERIM_DATASET_BOUNDARY", path, "evidence_ref must be a mapping")
        else:
            _expect_closed_mapping(
                evidence_ref,
                {
                    "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/A1_LEGACY_GAP_INVENTORY_V2.json",
                    "sha256": "d1b371fd350f910a6de38e27c50a30f9c97c660085382f0ac384ac9ecdc0fdff",
                },
                path,
                issues,
                "A1_INTERIM_DATASET_BOUNDARY",
            )
        gse145046 = summary.get("GSE145046")
        if not isinstance(gse145046, Mapping):
            _issue(issues, "A1_INTERIM_GSE145046", path, "GSE145046 boundary must be a mapping")
        else:
            expected_gse145046 = {
                "registry_qualification_status": "AUDIT_PENDING",
                "true_a2_qualification_status": "REJECTED_WITH_EVIDENCE",
                "classification": "CONDITIONALLY_RECOVERABLE_AS_ABSOLUTE_AUXILIARY",
                "a2_status": "NOT_TRUE_A2_FIXED_REPORTER_ABSOLUTE_AUXILIARY",
                "scheme_a_role": "ABSOLUTE_AUXILIARY_ONLY",
                "ordinary_gate_contribution": 0,
                "a1_gate_contribution": 0,
                "true_a2_gate_contribution": 0,
                "source_relative_confirmatory_evidence_allowed": False,
                "canonical_intervention_record_count": 0,
                "measured_candidate_pool_count": 0,
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "a1_inventory_qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
                "formal_audit_execution_status": "COMPLETED",
                "payload_integrity_status": "PASS",
                "rpm_validation_status": "PASS",
                "aggregate_reconciliation_status": "MATCH",
                "data_semantics": "FIXED_SCAFFOLD_ABSOLUTE_OUTCOMES_NOT_DIRECT_SOURCE_TO_CANDIDATE_INTERVENTIONS",
                "full_reporter_anchor_status": "NOT_CLOSED",
                "n10_locus_status": "CLOSED_AT_PRIMER_LEVEL",
                "decisive_remaining_blockers": [
                    "FULL_REPORTER_SOURCE_ANCHOR_NOT_IDENTIFIABLE",
                    "FACS_GATE_CONSTANTS_NOT_RECOVERED",
                    "IN_VIVO_HALF_LIFE_BASELINE_AND_AGGREGATION_NOT_RECOVERED",
                    "IN_VITRO_REPLICATE_AND_SE_NOT_IDENTIFIABLE",
                    "LICENSE_AND_REDISTRIBUTION_NOT_BOUND",
                    "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_CLOSED",
                    "DENSE_SPLIT_AND_HAMMING_MOAT_NOT_FROZEN",
                    "ABSOLUTE_OUTCOME_NOT_DIRECT_SOURCE_CANDIDATE_INTERVENTION",
                    "TRUE_A2_NOT_QUALIFIED",
                ],
            }
            _expect_closed_mapping(
                gse145046,
                expected_gse145046,
                path,
                issues,
                "A1_INTERIM_GSE145046",
            )
        gse114002 = summary.get("GSE114002")
        if not isinstance(gse114002, Mapping):
            _issue(issues, "A1_INTERIM_GSE114002", path, "GSE114002 boundary must be a mapping")
        else:
            expected_gse114002 = {
                "registry_qualification_status": "AUDIT_PENDING",
                "scheme_a_role": "A2_RECOVERY_CANDIDATE_NOT_QUALIFIED",
                "known_related_sequence_exposure_label": "SEQUENCE_EXPOSED",
                "future_use_boundary_if_qualified": "WITHIN_ASSAY_DEVELOPMENT_AND_OPTIMIZATION_ONLY_SEQUENCE_EXPOSED",
                "fallback_if_not_qualifiable": "NEW_GENUINE_PUBLIC_A2_STUDY_REQUIRED",
                "qualified": False,
                "a1_inventory_qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
                "p0_manifest_status": "INCOMPLETE_BLOCKED",
                "defect": "STALE_MANIFEST_HASH_DEFECT",
                "declared_file_hash_mismatch_count": 1,
                "provenance_reconciliation_status": "PROVENANCE_RECONCILED_NOT_QUALIFIED",
                "current_valid_payload_sha256": "23bbd468ff6c6905f11e7dfdd7509601730e0f99c8ad2a78f37f3dfe99c31719",
                "stale_declared_and_quarantined_payload_sha256": "d5baad2fcc6b59b572a1f3239bcf7910bd421fbbd4971f97b06671576ba7b0d7",
                "endpoint_geometry_reconciliation": {
                    "current_artifact_lineage_id": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_LINEAGE_ID,
                    "current_artifact_id": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_ARTIFACT_ID,
                    "current_status": "MECHANICAL_ENDPOINT_RECONCILED_NOT_QUALIFIED",
                    "previous_failure_artifact_lineage_id": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_001_LINEAGE_ID,
                    "previous_failure_preserved": True,
                    "mechanical_closure_only": True,
                    "mechanically_closed_from_previous_attempt": GSE114002_ENDPOINT_GEOMETRY_CLOSED_BLOCKERS,
                    "eligible_provisional_pool_count": 959,
                    "eligible_provisional_distinct_candidate_count": 3899,
                    "provisional_counts_are_diagnostic_only": True,
                    "blocker_count": 7,
                    "decisive_remaining_blockers": GSE114002_ENDPOINT_GEOMETRY_ATTEMPT_002_BLOCKERS,
                    "ordinary_study_contribution": 0,
                    "a1_intervention_study_contribution": 0,
                    "true_a2_dense_study_contribution": 0,
                    "canonical_record_count": 0,
                    "qualified": False,
                    "true_a2_claim_established": False,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "runtime_sync_status": "PENDING_NO_EVT_039",
                },
                "public_authority_gap_audit": {
                    "artifact_lineage_id": GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_LINEAGE_ID,
                    "record_id": "GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_V1",
                    "status": GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_STATUS,
                    "engineering_item_closed": "PUBLIC_AUTHORITY_EVIDENCE_SEARCH_AND_GAP_CLASSIFICATION_COMPLETED",
                    "science_blockers_closed_count": 0,
                    "decisive_remaining_blocker_count": 7,
                    "checkpoint_family_count": 4,
                    "accession_level_exposed_family_count": 2,
                    "not_declared_without_absence_family_count": 2,
                    "checkpoint_specific_exposure_closed_count": 0,
                    "near_duplicate_exposure_audit_completed_count": 0,
                    "designed_sample_chemistry_status": "UNKNOWN_NOT_ASSERTED",
                    "data_redistribution_rights_status": "UNKNOWN_NOT_ASSERTED",
                    "predecessor_runtime_event_id": "A1-EVT-039",
                    "ordinary_study_contribution": 0,
                    "a1_intervention_study_contribution": 0,
                    "true_a2_dense_study_contribution": 0,
                    "canonical_record_count": 0,
                    "qualified": False,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "runtime_sync_status": "PENDING_NO_EVT_040",
                },
            }
            _expect_closed_mapping(
                gse114002,
                expected_gse114002,
                path,
                issues,
                "A1_INTERIM_GSE114002",
            )
        gse149487 = summary.get("GSE149487")
        if not isinstance(gse149487, Mapping):
            _issue(issues, "A1_INTERIM_GSE149487", path, "GSE149487 boundary must be a mapping")
        else:
            expected_gse149487 = {
                "registry_qualification_status": "AUDIT_PENDING",
                "a1_inventory_qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
                "development_reconstruction_status": "DEVELOPMENT_RECONSTRUCTED_NOT_QUALIFIED",
                "input_value_scale": "PUBLISHED_LOG2_CPM_PER_BARCODE",
                "canonical_record_count": 0,
                "development_companion_effect_record_count": 204,
                "development_companion_nonnull_effect_record_count": 192,
                "raw_barcode_plaintext_match_count_in_outputs": 0,
                "stop_before_data_preflight": {
                    "artifact_lineage_id": "gse149487_full_a1_stop_before_data_preflight_v1",
                    "authority_status": "PASS_CONFIG_ONLY_BINDING_VERIFIED",
                    "inventory_status": "PASS_METADATA_ONLY_STOP_BEFORE_DATA",
                    "outcome": "NOT_READY_FOR_STUDY_QUALIFICATION",
                    "blocker_count": 11,
                    "manifest_open_count": 0,
                    "payload_hash_count": 0,
                    "payload_open_count": 0,
                    "scientific_processing_count": 0,
                    "qualifier_execution_count": 0,
                    "training_run_count": 0,
                    "model_selection_run_count": 0,
                    "canonical_record_count": 0,
                    "ordinary_study_contribution": 0,
                    "a1_study_contribution": 0,
                    "true_a2_study_contribution": 0,
                    "qualified": False,
                    "ready_for_study_qualification": False,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "changes_qualification_gate": False,
                    "historical_r4_reopened": False,
                },
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "decisive_remaining_blockers": [
                    "PC3_AND_18_GEO_RAW_COUNT_TABLE_JOIN_NOT_INCLUDED",
                    "SUPPLEMENTS_NOT_LISTED_IN_CURRENT_P0_MANIFEST",
                    "LICENSE_AND_REDISTRIBUTION_NOT_CLOSED",
                    "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_CLOSED",
                    "GROUP_LEAKAGE_AND_SPLIT_NOT_FROZEN",
                    "PAPER_NATIVE_MANN_WHITNEY_AND_MULTIPLE_TESTING_NOT_REPRODUCED",
                    "CANONICAL_INTERVENTION_RECORD_V3_NOT_MATERIALIZED",
                    "UNADJUDICATED_DESCRIPTION_CLASSES_EXCLUDED",
                    "UNADJUDICATED_6A_COORDINATE_CLASSES_EXCLUDED",
                ],
            }
            _expect_closed_mapping(
                gse149487,
                expected_gse149487,
                path,
                issues,
                "A1_INTERIM_GSE149487",
            )
        gse200304 = summary.get("GSE200304")
        if not isinstance(gse200304, Mapping):
            _issue(issues, "A1_INTERIM_GSE200304", path, "GSE200304 boundary must be a mapping")
        else:
            expected_gse200304 = {
                "registry_qualification_status": "AUDIT_PENDING",
                "a1_inventory_qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
                "qualification_execution_outcome": "ENGINEERING_SUCCESS_BLOCKED_NOT_QUALIFIED",
                "qualification_status": "BLOCKED_NOT_QUALIFIED",
                "ordinary_gate_contribution": 0,
                "a1_gate_contribution": 0,
                "true_a2_gate_contribution": 0,
                "nominal_intervention_pair_count": 6885,
                "distinct_candidate_count": 6885,
                "source_sequence_proxy_group_count": 6882,
                "singleton_source_pool_count": 6879,
                "two_candidate_source_pool_count": 3,
                "three_or_more_candidate_source_pool_count": 0,
                "ndcg_eligible_source_pool_count": 0,
                "processed_pair_count": 6772,
                "outcome_blind_attrition_count": 113,
                "small_plasmid_complete_pair_count": 6120,
                "ivt_complete_pair_count": 6774,
                "all_pairs_exactly_one_snv": True,
                "controls_excluded_from_source_candidate_geometry": True,
                "paper_native_raw_xtail_replay_status": "NOT_RUN",
                "fastq_acquisition_status": "COMMITTED_TRANSPORT_INTEGRITY_VERIFIED",
                "fastq_independent_consumer_status": "ALREADY_COMMITTED_VERIFIED",
                "raw_replay_preflight_status": "BLOCKED_PRE_EXECUTION_WITH_EVIDENCE",
                "raw_replay_preflight_hard_unknown_blocker_count": 17,
                "raw_replay_preflight_fastq_body_read_count": 0,
                "raw_replay_preflight_changes_qualification_gate": False,
                "primary_subseries_role_authority": {
                    "target_subseries_accession": "GSE200302",
                    "bioproject_accession": "PRJNA824033",
                    "publication_status": "COMMITTED_AND_ACCEPTED",
                    "status": "EXACT_OFFICIAL_SRR_ROLE_AUTHORITY_CLOSED",
                    "artifact_path": GSE200302_ROLE_ARTIFACT_ROOT,
                    "bundle_digest": GSE200302_ROLE_BUNDLE_DIGEST,
                    "mapping_row_count": 24,
                    "experiment_join_row_count": 24,
                    "measurement_families": ["High_Poly", "Low_Poly", "pDNA", "Total_RNA"],
                    "replicates": [1, 2, 3, 4, 5, 6],
                    "forbidden_80s_alias_count": 0,
                    "prior_blocker": "EXACT_SRR_SAMPLE_ROLES_UNKNOWN",
                    "prior_blocker_status": "CLOSED",
                    "historical_preflight_blocker_list_preserved": True,
                    "replacement_blocker": "REQUIRED_80S_ROLE_AUTHORITY_ABSENT",
                    "replacement_blocker_status": "OPEN",
                    "role_grid_status": "CONFLICT_WITH_CURRENT_80S_EXPECTATION",
                    "pdna_may_substitute_for_80s_rna": False,
                    "current_effective_raw_replay_blocker_count": 17,
                    "changes_qualification_gate": False,
                    "runtime_sync_status": "PENDING_NO_EVT_035",
                },
                "published_endpoint_evidence": {
                    "artifact_lineage_id": "gse200304_published_endpoint_a1_v1",
                    "publication_state": "COMMITTED_ACCEPTED",
                    "execution_outcome": "ENGINEERING_SUCCESS_IMMUTABLY_BLOCKED",
                    "independent_consumer_validation_status": "PASS",
                    "evidence_role": "AGGREGATE_AUDIT_ONLY_PENDING_OWNER_POLICY",
                    "primary_membership_pair_count": 6772,
                    "primary_finite_effect_pair_count": 6547,
                    "table_s2_absent_from_table_s3_pair_count": 113,
                    "primary_total_attrition_count": 338,
                    "primary_complete_distinct_wt_201nt_proxy_group_count": 6544,
                    "blocker_count": 8,
                    "ordinary_study_contribution": 0,
                    "a1_intervention_study_contribution": 0,
                    "true_a2_dense_study_contribution": 0,
                    "canonical_record_count": 0,
                    "qualified": False,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "changes_qualification_gate": False,
                    "runtime_sync_status": "PENDING_NO_EVT_037",
                },
                "sam_to_oligo_count_reconstruction_status": "UNKNOWN_NOT_ASSERTED",
                "acquisition_changes_qualification_gate": False,
                "source_grouping_status": "SEQUENCE_EQUALITY_PROXY_NOT_BIOLOGICALLY_FROZEN",
                "license_and_redistribution_status": "UNKNOWN_NOT_ASSERTED",
                "checkpoint_specific_foundation_exposure_status": "UNKNOWN_NOT_ASSERTED",
                "canonical_intervention_record_count": 0,
                "dec019_post_adjudication": {
                    "artifact_lineage_id": GSE200304_DEC019_ADJUDICATION_LINEAGE_ID,
                    "status": "BLOCKED",
                    "adjudication_status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
                    "input_gate_count": 8,
                    "input_status_counts": GSE200304_DEC019_POST_ADJUDICATION_INPUT_STATUS_COUNTS,
                    "blocker_count": 7,
                    "blockers": GSE200304_DEC019_POST_ADJUDICATION_BLOCKERS,
                    "ordinary_study_contribution": 0,
                    "a1_study_contribution": 0,
                    "true_a2_study_contribution": 0,
                    "positive_input_canonical_record_count": 6547,
                    "canonical_record_count": 0,
                    "positive_input_fact_is_not_final_canonical_materialization": True,
                    "qualified": False,
                    "canonical_materialization_allowed": False,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                },
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            }
            _expect_closed_mapping(
                gse200304,
                expected_gse200304,
                path,
                issues,
                "A1_INTERIM_GSE200304",
            )
        three_utr = summary.get("three_utr_candidates")
        if not isinstance(three_utr, Mapping):
            _issue(issues, "A1_INTERIM_THREE_UTR", path, "three_utr_candidates must be a mapping")
        else:
            _expect_closed_mapping(
                three_utr,
                {
                    "dataset_ids": [
                        "GSE217518",
                        "ENCSR854RUF",
                        "GSE200304",
                        "GSE232572",
                        "GSE186455",
                    ],
                    "registry_qualification_status": "AUDIT_PENDING",
                    "a1_inventory_qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
                    "qualified_studies": 0,
                    "transfer_claim_status": "NOT_ESTABLISHED",
                },
                path,
                issues,
                "A1_INTERIM_THREE_UTR",
            )
        gse207584 = summary.get("GSE207584")
        if not isinstance(gse207584, Mapping):
            _issue(issues, "A1_INTERIM_GSE207584", path, "GSE207584 boundary must be a mapping")
        else:
            _expect_closed_mapping(
                gse207584,
                {
                    "registry_qualification_status": "AUDIT_PENDING",
                    "a1_inventory_qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
                    "v3_per_variant_sequence_recovery_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
                    "required_recovery": "SYNONYMOUS_FAMILY_SEQUENCE_LABEL_AND_GROUP_LINEAGE",
                    "qualified": False,
                },
                path,
                issues,
                "A1_INTERIM_GSE207584",
            )
        _expect(summary, "qualified_a2_dense_neighborhoods", 0, path, issues, "A1_INTERIM_DATASET_BOUNDARY")

    boundary_deviation = interim.get("boundary_deviation")
    if not isinstance(boundary_deviation, Mapping):
        _issue(
            issues,
            "A1_INTERIM_BOUNDARY_DEVIATION",
            path,
            "boundary_deviation must be a mapping",
        )
    else:
        _expect_closed_mapping(
            boundary_deviation,
            {
                "count": 5,
                "classifications": [
                    "NON_SENSITIVE_AGGREGATE_METADATA_BOUNDARY_DEVIATION",
                    "NON_SENSITIVE_EXISTING_POLICY_TEXT_BOUNDARY_DEVIATION",
                    "ORDINARY_PUBLIC_OLIGO_PREVIEW_BOUNDARY_DEVIATION",
                    "NON_SENSITIVE_EXISTING_CODE_SYMBOL_BOUNDARY_DEVIATION_ADDENDUM",
                    "NON_SENSITIVE_PUBLIC_EXCLUDED_POLICY_LINE_BOUNDARY_DEVIATION_ADDENDUM",
                ],
                "descriptions": [
                    "ONE_PROHIBITED_STUDY_AGGREGATE_METADATA_ITEM_WAS_INCIDENTALLY_DISPLAYED_FROM_AN_OLD_MIXED_SPLIT_SUMMARY",
                    "SEVERAL_EXISTING_REGISTRY_OR_HISTORICAL_POLICY_LINES_WERE_INCIDENTALLY_DISPLAYED_FROM_ORDINARY_WORKTREE_DOCUMENTATION",
                    "FOUR_ORDINARY_PUBLIC_OLIGO_CELL_VALUES_WERE_INCIDENTALLY_DISPLAYED_DURING_A_COLUMN_TYPE_PREVIEW",
                    "ONE_EXISTING_EXCLUDED_STUDY_TEST_FUNCTION_NAME_WAS_DISPLAYED_DURING_A_FINAL_ORDINARY_CODE_SYMBOL_LISTING",
                    "ONE_NON_SENSITIVE_PUBLIC_EXCLUDED_DATASET_POLICY_LINE_WAS_INCIDENTALLY_DISPLAYED_DURING_PUBLIC_CONFIG_REVIEW",
                ],
                "inspection_stopped_after_detection": True,
                "restricted_or_sealed_path_accessed": False,
                "excluded_study_payload_contact": False,
                "restricted_or_sealed_member_content_read": False,
                "ordinary_public_oligo_cell_values_displayed": 4,
                "ordinary_worktree_excluded_study_test_function_names_displayed": 1,
                "ordinary_public_excluded_dataset_policy_lines_displayed": 1,
                "restricted_or_sealed_sequence_read": False,
                "raw_label_read": False,
                "runner_invoked": False,
                "access_intent_written": False,
                "evaluation_count": 0,
                "used_in_a1_reasoning": False,
                "evidence_deleted_or_concealed": False,
                "gate_decision_affected": False,
                "evidence_manifest_ref": {
                    "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/BOUNDARY_DEVIATIONS_MANIFEST.json",
                    "bytes": 2218,
                    "sha256": "e6c1b79c037058d21be33e0103a1d9740feac7cb0651f13033a01e0f8ff4bd47",
                },
            },
            path,
            issues,
            "A1_INTERIM_BOUNDARY_DEVIATION",
        )

    power_prefreeze = interim.get("power_prefreeze")
    if not isinstance(power_prefreeze, Mapping):
        _issue(
            issues,
            "A1_INTERIM_POWER_PREFREEZE",
            path,
            "power_prefreeze must be a mapping",
        )
    else:
        _expect_closed_mapping(
            power_prefreeze,
            {
                "source": "V3_DEC_019_USER_AUTHORIZED_ROOT_AUTHORITY_COMPANION",
                "contract_amendment": True,
                "status": "PREFROZEN_BEFORE_MODEL_RESULTS",
                "model_results_may_change_this_rule": False,
                "analysis_and_bootstrap_unit": "BIOLOGICAL_SOURCE_GROUP",
                "target_metric": "WITHIN_STUDY_SPEARMAN",
                "minimum_effect_at_alternative": 0.25,
                "alpha_two_sided": 0.05,
                "target_power": 0.8,
                "confidence_level": 0.95,
                "maximum_ci_full_width": 0.3,
                "contract_specifies_target_ci_full_width": True,
                "selected_under_contract_underspecification": False,
                "simulation_seed": 20260810,
                "bootstrap_resamples": 2000,
                "simulation_trials": 1000,
                "user_discussion_open": False,
            },
            path,
            issues,
            "A1_INTERIM_POWER_PREFREEZE",
        )

    claims = interim.get("claim_boundaries")
    if not isinstance(claims, Mapping):
        _issue(issues, "A1_INTERIM_CLAIMS", path, "claim_boundaries must be a mapping")
    else:
        expected_claims = {
            "gap_inventory_is_data_freeze": False,
            "provenance_reconciliation_is_study_qualification": False,
            "metadata_or_row_mass_establishes_effective_n": False,
            "endpoint_assay_region_or_replicate_increases_study_count": False,
            "engineering_tests_establish_scientific_claim": False,
            "smoke_or_proxy_result_may_be_final_scientific_conclusion": False,
            "gse145046_formal_audit_execution_is_study_qualification": False,
            "gse145046_fixed_scaffold_absolute_auxiliary_is_true_a2": False,
            "gse114002_mechanical_reconciliation_is_study_qualification": False,
            "gse114002_provisional_pool_or_candidate_counts_are_effective_n": False,
            "gse114002_mechanical_reconciliation_authorizes_training_or_model_selection": False,
            "gse114002_public_authority_gap_audit_is_study_qualification": False,
            "gse114002_public_authority_gap_audit_closes_any_science_blocker": False,
            "gse114002_accession_level_exposure_is_checkpoint_specific_exposure": False,
            "gse114002_checkpoint_not_declared_establishes_absence": False,
            "gse114002_author_code_gpl_licenses_geo_data": False,
            "dec019_authority_decision_is_gse114002_study_qualification": False,
            "gse114002_technical_fraction_uncertainty_is_biological_standard_error": False,
            "gse114002_technical_fraction_uncertainty_may_support_power_ci_equivalence_confirmatory_or_generalization": False,
            "gse114002_k2_candidate_hamming_distance_may_be_dropped": False,
            "gse114002_k5_may_receive_qualification_credit": False,
            "gse114002_gsm_library_pool_or_candidate_may_count_as_independent_study": False,
            "gse200304_engineering_success_is_study_qualification": False,
            "gse200304_fastq_acquisition_is_study_qualification": False,
            "gse200304_transport_integrity_is_paper_native_count_replay": False,
            "gse200304_sequence_proxy_groups_are_biological_source_groups": False,
            "gse200304_precomputed_aggregate_evidence_is_paper_native_xtail_replay": False,
            "dec019_authority_decision_is_gse200304_study_qualification": False,
            "gse200304_published_endpoint_engineering_success_is_study_qualification": False,
            "gse200304_subseries_modality_endpoint_or_replicate_may_count_as_independent_study": False,
            "gse200304_raw_replay_preflight_is_study_qualification": False,
            "gse200304_raw_replay_preflight_is_paper_native_count_replay": False,
            "gse200304_raw_replay_preflight_is_paper_native_xtail_replay": False,
            "gse200304_raw_replay_preflight_resolves_control_u_policy": False,
            "gse200302_srr_role_authority_is_raw_replay": False,
            "gse200302_srr_role_authority_is_study_qualification": False,
            "gse200302_srr_role_authority_authorizes_training_or_model_selection": False,
            "gse200302_pdna_may_substitute_for_80s_rna": False,
            "gse149487_stop_before_data_preflight_is_study_qualification": False,
            "gse149487_metadata_inventory_is_payload_integrity": False,
            "gse149487_stop_before_data_preflight_authorizes_training_or_model_selection": False,
            "gse149487_stop_before_data_preflight_establishes_scientific_claim": False,
            "gse149487_stop_before_data_preflight_reopens_historical_r4": False,
            "dec019_creates_global_replicate_or_standard_error_relaxation": False,
            "dec019_changes_gse149487_three_biological_replicates_and_route_a_se_gate": False,
            "dec019_allows_checkpoint_specific_exposure_or_rights_waiver": False,
            "dec019_a1_authority_freeze_is_final_benchmark_membership_freeze": False,
            "a1_phase_complete": False,
            "route_a_established": False,
        }
        _expect_closed_mapping(
            claims, expected_claims, path, issues, "A1_INTERIM_CLAIMS"
        )

    verification = interim.get("verification")
    if not isinstance(verification, Mapping):
        _issue(
            issues,
            "A1_INTERIM_VERIFICATION",
            path,
            "verification must be a mapping",
        )
    else:
        _expect_closed_mapping(
            verification,
            {
                "targeted_dec019_successor_adjudicator_tests": {
                    "status": "PASS",
                    "initial_i_state": {
                        "status": "PASS",
                        "passed": 60,
                        "failed": 0,
                    },
                    "synthetic_bound_state": {
                        "status": "PASS",
                        "passed": 60,
                        "failed": 0,
                    },
                },
                "dec019_successor_json_parse": "PASS",
                "dec019_successor_python_compile": "PASS",
                "dec019_successor_stable_core_projection_sha256_verified": True,
                "dec019_successor_initial_i_binding_fail_closed_before_evidence_or_output": True,
                "targeted_a1_tests": {"status": "PASS", "passed": 17, "failed": 0},
                "targeted_plumage_reconstruction_tests": {"status": "PASS", "passed": 32, "failed": 0},
                "targeted_gse149487_stop_before_data_preflight_tests": {
                    "status": "PASS",
                    "passed": 17,
                    "failed": 0,
                },
                "gse149487_stop_before_data_preflight_json_parse": "PASS",
                "gse149487_stop_before_data_preflight_sha256_verified": True,
                "gse149487_stop_before_data_preflight_closed_metadata_semantics": "PASS",
                "targeted_gse145046_a2_audit_tests": {"status": "PASS", "passed": 59, "failed": 0},
                "targeted_gse200304_a1_qualification_tests": {"status": "PASS", "passed": 60, "failed": 0},
                "targeted_gse200304_raw_replay_preflight_tests": {"status": "PASS", "passed": 57, "failed": 0},
                "gse200304_raw_replay_preflight_failure_bundle_validation": "PASS",
                "gse200304_raw_replay_preflight_final_bundle_validation": "PASS",
                "gse200304_final_bundle_validation": "PASS",
                "gse200304_final_bundle_default_consumer_validation": "PASS",
                "gse200304_closed_report_schema": "PASS",
                "gse200304_exact_file_set": "PASS",
                "gse200304_sha256s": "PASS",
                "gse200304_terminal_marker": "PASS",
                "gse200304_failure_bundles_preserved": "PASS",
                "gse200304_ena_manifest_closed_metadata_validation": "PASS",
                "gse200304_published_endpoint_bundle_validation": "PASS",
                "gse200304_published_endpoint_independent_consumer_validation": "PASS",
                "gse200304_published_endpoint_exact_file_set": "PASS",
                "gse200304_published_endpoint_sha256s": "PASS",
                "gse200304_published_endpoint_terminal_marker": "PASS",
                "gse200304_published_endpoint_gate_unchanged": "PASS",
                "gse200304_published_endpoint_producer_blob_binding": "PASS",
                "gse114002_public_authority_gap_audit_json_parse": "PASS",
                "gse114002_public_authority_gap_audit_closed_semantics": "PASS",
                "gse114002_public_authority_gap_audit_recursive_privacy_scan": "PASS",
                "gse145046_closed_report_schema": "PASS",
                "gse145046_payload_integrity": "PASS",
                "gse145046_rpm_validation": "PASS",
                "gse145046_aggregate_reconciliation": "MATCH",
                "plumage_v4_bundle_sha256s": "PASS",
                "plumage_v4_exact_file_set": "PASS",
                "plumage_v4_public_inventory_gate": "PASS",
                "plumage_v4_report_semantics": "PASS",
                "plumage_v4_raw_barcode_leakage_scan": "PASS",
                "plumage_v4_barcode_hash_rows": 283680,
                "gap_inventory_v2_json_parse": "PASS",
                "authority_yaml_parse": "PASS",
                "qualification_protocol_json_parse": "PASS",
                "qualification_protocol_sha256_verified": True,
                "legacy_gap_inventory_v2_sha256_verified": True,
                "run_manifest_draft7_compatibility_errors": 0,
                "full_repository_tests": {
                    "status": "NOT_RUN",
                    "reason": "INTERIM_TARGETED_MODULE_SCOPE",
                },
                "full_build": {
                    "status": "NOT_RUN",
                    "reason": "NO_BUILD_INTERFACE_CHANGED",
                },
                "full_lint": {
                    "status": "NOT_RUN",
                    "reason": "INTERIM_TARGETED_MODULE_SCOPE",
                },
                "e2e": {
                    "status": "NOT_RUN",
                    "reason": "A1_GATE_BLOCKED_AND_NO_TRAINING_OR_EVALUATION_AUTHORIZED",
                },
                "independent_review": {
                    "status": "COMPLETED_FINDINGS_APPLIED",
                    "scope": "INTERIM_RECORD_STATUS_CLAIM_BOUNDARIES_PLUMAGE_BLOCK_SEMANTICS_AND_LOG2_CPM_SCALE",
                },
            },
            path,
            issues,
            "A1_INTERIM_VERIFICATION",
        )

    _expect(interim, "initial_generated_at", "2026-08-10T06:30:58+08:00", path, issues, "A1_INTERIM_TIME")
    _expect(interim, "updated_for_decision_id", "V3-DEC-019", path, issues, "A1_INTERIM_TIME")
    _expect(interim, "latest_authority_update_id", "V3-DEC-019", path, issues, "A1_INTERIM_TIME")
    _expect(
        interim,
        "latest_evidence_update_id",
        "GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3",
        path,
        issues,
        "A1_INTERIM_TIME",
    )
    generated = interim.get("generated_at")
    updated = interim.get("updated_at")
    if (
        generated != GSE200304_DEC019_POST_ADJUDICATION_LEDGER_AT
        or updated != GSE200304_DEC019_POST_ADJUDICATION_LEDGER_AT
    ):
        _issue(
            issues,
            "A1_INTERIM_TIME",
            path,
            "generated_at and updated_at must remain the exact GSE200304 DEC-019 post-adjudication ledger timestamp",
        )
    if generated != updated:
        _issue(issues, "A1_INTERIM_TIME", path, "generated_at and updated_at must identify the same amended record bytes")
    try:
        updated_dt = datetime.fromisoformat(str(updated))
        audit_dt = datetime.fromisoformat("2026-08-10T08:43:13+08:00")
        amendment_dt = datetime.fromisoformat("2026-08-10T10:10:05+08:00")
        acquisition_dt = datetime.fromisoformat("2026-08-10T19:15:02+08:00")
        raw_preflight_dt = datetime.fromisoformat("2026-08-10T20:52:00+08:00")
        role_authority_dt = datetime.fromisoformat("2026-08-10T23:03:15+08:00")
        gse149487_preflight_dt = datetime.fromisoformat("2026-08-11T02:14:39+08:00")
        gse200304_published_endpoint_dt = datetime.fromisoformat("2026-08-11T04:40:50+08:00")
        gse114002_endpoint_geometry_dt = datetime.fromisoformat("2026-08-11T07:57:11+08:00")
        gse114002_public_gap_audit_dt = datetime.fromisoformat("2026-08-11T10:24:16+08:00")
        dec019_authorization_dt = datetime.fromisoformat("2026-08-11T10:42:53+08:00")
        dec019_successor_integration_dt = datetime.fromisoformat("2026-08-11T12:19:34+08:00")
        gse200304_positive_lineage_gate_dt = datetime.fromisoformat("2026-08-11T18:09:26+08:00")
        gse200304_negative_gate_pack_dt = datetime.fromisoformat("2026-08-11T19:42:43+08:00")
        gse200304_adjudication_v3_dt = datetime.fromisoformat("2026-08-11T19:50:38+08:00")
    except ValueError:
        _issue(issues, "A1_INTERIM_TIME", path, "updated_at must be an ISO-8601 timestamp with offset")
    else:
        if (
            updated_dt < audit_dt
            or updated_dt < amendment_dt
            or updated_dt < acquisition_dt
            or updated_dt < raw_preflight_dt
            or updated_dt < role_authority_dt
            or updated_dt < gse149487_preflight_dt
            or updated_dt < gse200304_published_endpoint_dt
            or updated_dt < gse114002_endpoint_geometry_dt
            or updated_dt < gse114002_public_gap_audit_dt
            or updated_dt < dec019_authorization_dt
            or updated_dt < dec019_successor_integration_dt
            or updated_dt < gse200304_positive_lineage_gate_dt
            or updated_dt < gse200304_negative_gate_pack_dt
            or updated_dt < gse200304_adjudication_v3_dt
        ):
            _issue(
                issues,
                "A1_INTERIM_TIME",
                path,
                "updated_at must follow all preserved evidence events, DEC-019 owner authorization, successor integration, and GSE200304 post-adjudication artifacts",
            )
    return issues


def validate_sealed_hard_disable(
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    issues: list[Issue] = []
    if "sealed_policy" in config:
        _issue(issues, "SEALED_POLICY_ALIAS_FORBIDDEN", CONFIG_PATH, "sealed controls must live only under the top-level sealed key")
    sealed = config.get("sealed")
    if not isinstance(sealed, Mapping):
        _issue(issues, "SEALED_BLOCK_MISSING", CONFIG_PATH, "top-level sealed mapping is required")
        return issues

    expected = {
        "dataset_id": SEALED_DATASET_ID,
        "role": "SEALED_EXTERNAL_FINAL_ONLY",
        "phase": "A10",
        "guard_mode": "A0_A9_UNCONDITIONAL_HARD_DISABLE",
        "latent_authorization_path_allowed": False,
        "evaluator_implementation_status": "A0_STUB_HARD_DISABLED",
        "a9_guard_replacement_required": True,
        "execution_enabled": False,
        "execution_authorized": False,
        "authorized": False,
        "access_intent_allowed": False,
        "ordinary_loader_returns_zero_rows": True,
        "in_task_activation": False,
        "in_metric_branch": False,
        "in_calibration": False,
        "in_model_selection": False,
        "final_evaluator_count_max": 1,
        "required_authorization_phase": "A10",
        "required_user_authorization": True,
        "automatic_execution_by_this_plan_allowed": False,
        "ordinary_activation_allowed": False,
        "training_allowed": False,
        "threshold_or_metric_setting_allowed": False,
        "calibration_allowed": False,
        "model_selection_allowed": False,
        "architecture_selection_allowed": False,
        "error_analysis_allowed_before_authorized_a10": False,
        "dry_run_may_write_access_intent": False,
        "custody_metadata_only_in_ordinary_registry": True,
    }
    for key, value in expected.items():
        _expect(sealed, key, value, CONFIG_PATH, issues, "SEALED_HARD_DISABLE")
    replacement_preconditions = sealed.get("a9_replacement_preconditions")
    if replacement_preconditions != list(SEALED_A9_REPLACEMENT_PRECONDITIONS):
        _issue(
            issues,
            "SEALED_A9_REPLACEMENT_PRECONDITIONS",
            CONFIG_PATH,
            "sealed.a9_replacement_preconditions must be the exact ordered seven-item A9 boundary",
        )
    required_prior = sealed.get("required_prior_phases")
    if not isinstance(required_prior, list) or set(required_prior) != set(EXPECTED_PHASE_IDS[:-1]) or len(required_prior) != 10:
        _issue(issues, "SEALED_PRIOR_PHASES", CONFIG_PATH, "sealed.required_prior_phases must be exactly A0 through A9")
    output = sealed.get("output")
    if not isinstance(output, Mapping):
        _issue(issues, "SEALED_OUTPUT_POLICY", CONFIG_PATH, "sealed.output mapping is required")
    else:
        _expect(output, "aggregate_only", True, CONFIG_PATH, issues, "SEALED_OUTPUT_POLICY")
        _expect(output, "row_level_labels_returned", False, CONFIG_PATH, issues, "SEALED_OUTPUT_POLICY")
    if sealed.get("authorization_record_path") is not None or sealed.get("authorization_record_sha256") is not None:
        _issue(issues, "SEALED_AUTHORIZATION_PREPOPULATED", CONFIG_PATH, "A0 must not pre-populate a sealed authorization record")

    data_entry = _mapping_entry(registries["data"].get("datasets"), "dataset_id", SEALED_DATASET_ID)
    if not isinstance(data_entry, Mapping):
        _issue(issues, "SEALED_DATASET_MISSING", REGISTRY_PATHS["data"], f"{SEALED_DATASET_ID} is required")
    else:
        data_expected = {
            "sealed": True,
            "role": "SEALED_EXTERNAL_FINAL_ONLY",
            "qualified": False,
            "training_role": "EXCLUDED_ALWAYS",
            "all_training_roles_excluded": True,
            "execution_enabled": False,
            "execution_authorized": False,
            "access_intent_allowed": False,
            "aggregate_only": True,
        }
        for key, value in data_expected.items():
            _expect(data_entry, key, value, REGISTRY_PATHS["data"], issues, "SEALED_DATA_ROLE")
        forbidden_uses = set(data_entry.get("forbidden_current_uses", []))
        required_forbidden = {
            "ORDINARY_ACTIVATION",
            "TRAINING",
            "HYPERPARAMETER_SELECTION",
            "THRESHOLD_OR_METRIC_SELECTION",
            "CALIBRATION",
            "MODEL_SELECTION",
            "ARCHITECTURE_SELECTION",
            "ERROR_ANALYSIS",
            "ROW_LEVEL_LABEL_INSPECTION",
        }
        if not required_forbidden <= forbidden_uses:
            _issue(issues, "SEALED_DATA_FORBIDDEN_USE_COVERAGE", REGISTRY_PATHS["data"], f"missing {sorted(required_forbidden - forbidden_uses)!r}")
        ordinary_ids = registries["data"].get("ordinary_candidate_dataset_ids")
        if isinstance(ordinary_ids, list) and SEALED_DATASET_ID in ordinary_ids:
            _issue(issues, "SEALED_DATASET_IN_ORDINARY_SET", REGISTRY_PATHS["data"], f"{SEALED_DATASET_ID} appears in ordinary_candidate_dataset_ids")

    split_entry = _mapping_entry(registries["split"].get("splits"), "split_id", SEALED_SPLIT_ID)
    if not isinstance(split_entry, Mapping):
        _issue(issues, "SEALED_SPLIT_MISSING", REGISTRY_PATHS["split"], f"{SEALED_SPLIT_ID} is required")
    else:
        split_expected = {
            "dataset_id": SEALED_DATASET_ID,
            "sealed": True,
            "execution_enabled": False,
            "execution_authorized": False,
            "access_intent_allowed": False,
            "ordinary_loader_returns_zero_rows": True,
            "in_task_activation": False,
            "in_metric_branch": False,
            "in_calibration": False,
            "in_model_selection": False,
            "final_evaluator_count_max": 1,
        }
        for key, value in split_expected.items():
            _expect(split_entry, key, value, REGISTRY_PATHS["split"], issues, "SEALED_SPLIT_POLICY")
        split_output = split_entry.get("output")
        if not isinstance(split_output, Mapping) or split_output.get("aggregate_only") is not True or split_output.get("row_level_labels_returned") is not False:
            _issue(issues, "SEALED_SPLIT_OUTPUT", REGISTRY_PATHS["split"], "S6 output must be aggregate-only and return no row-level labels")

    matrix = registries["matrix"]
    controls = matrix.get("sealed_controls")
    if not isinstance(controls, Mapping):
        _issue(issues, "SEALED_MATRIX_CONTROLS", REGISTRY_PATHS["matrix"], "sealed_controls mapping is required")
    else:
        controls_expected = {
            "sealed_split_id": SEALED_SPLIT_ID,
            "sealed_dataset_id": SEALED_DATASET_ID,
            "ordinary_loader_returns_zero_rows": True,
            "in_ordinary_task_activation": False,
            "in_metric_branch": False,
            "in_calibration": False,
            "in_model_selection": False,
            "in_architecture_selection": False,
            "in_threshold_selection": False,
        }
        for key, value in controls_expected.items():
            _expect(controls, key, value, REGISTRY_PATHS["matrix"], issues, "SEALED_MATRIX_CONTROLS")
    semantics = matrix.get("task_split_semantics")
    sealed_semantics = semantics.get(SEALED_TASK_ID) if isinstance(semantics, Mapping) else None
    if not isinstance(sealed_semantics, Mapping):
        _issue(issues, "SEALED_TASK_SEMANTICS", REGISTRY_PATHS["matrix"], "sealed task semantics are required")
    else:
        for key, value in {
            "execution_enabled": False,
            "execution_authorized": False,
            "access_intent_allowed": False,
            "required_authorization_phase": "A10",
            "required_user_authorization": True,
            "final_evaluator_count_max": 1,
        }.items():
            _expect(sealed_semantics, key, value, REGISTRY_PATHS["matrix"], issues, "SEALED_TASK_SEMANTICS")

    sealed_claim = _mapping_entry(registries["claim"].get("claims"), "claim_id", "L3_SEALED_EXTERNAL_ADJUDICATION")
    if not isinstance(sealed_claim, Mapping):
        _issue(issues, "SEALED_CLAIM_MISSING", REGISTRY_PATHS["claim"], "sealed adjudication claim is required")
    else:
        for key, value in {"execution_enabled": False, "execution_authorized": False, "access_intent_allowed": False, "evidence_status": "NOT_RUN", "claim_status": "NOT_ESTABLISHED"}.items():
            _expect(sealed_claim, key, value, REGISTRY_PATHS["claim"], issues, "SEALED_CLAIM_POLICY")
    return issues


def validate_l4_and_pre_v3(
    config: Mapping[str, Any],
    supersession: Mapping[str, Any],
    claim_registry: Mapping[str, Any],
) -> list[Issue]:
    issues: list[Issue] = []
    scope = config.get("scientific_scope")
    l4_policy = scope.get("l4_biological_or_therapeutic_claim") if isinstance(scope, Mapping) else None
    if not isinstance(l4_policy, Mapping) or l4_policy.get("allowed") is not False or l4_policy.get("status") != "PROHIBITED":
        _issue(issues, "L4_POLICY", CONFIG_PATH, "L4 biological/therapeutic claim must be permanently prohibited")
    if not isinstance(scope, Mapping) or scope.get("data_policy") != "PUBLIC_DATA_ONLY" or scope.get("new_wet_lab_experiments_allowed") is not False:
        _issue(issues, "PUBLIC_DATA_ONLY_POLICY", CONFIG_PATH, "V3 must remain public-data-only with no new wet lab")

    claims_raw = claim_registry.get("claims")
    l4 = _mapping_entry(claims_raw, "claim_id", "L4_BIOLOGICAL_THERAPEUTIC")
    if not isinstance(l4, Mapping):
        _issue(issues, "L4_CLAIM_MISSING", REGISTRY_PATHS["claim"], "L4 prohibited claim cell is required")
    else:
        for key, value in {"evidence_status": "NOT_RUN", "claim_status": "PROHIBITED", "public_data_only_policy": True, "new_wet_lab_allowed": False}.items():
            _expect(l4, key, value, REGISTRY_PATHS["claim"], issues, "L4_CLAIM_POLICY")
    if isinstance(claims_raw, list):
        for claim in claims_raw:
            if not isinstance(claim, Mapping) or claim.get("claim_id") == "L4_BIOLOGICAL_THERAPEUTIC":
                continue
            if claim.get("claim_status") != "NOT_ESTABLISHED" or claim.get("evidence_status") != "NOT_RUN":
                _issue(issues, "A0_CLAIM_PREMATURE", REGISTRY_PATHS["claim"], f"claim {claim.get('claim_id')!r} must start NOT_RUN/NOT_ESTABLISHED")

    history = config.get("historical_constraints")
    active_run = history.get("active_pre_v3_run") if isinstance(history, Mapping) else None
    if not isinstance(active_run, Mapping):
        _issue(issues, "PRE_V3_RUN_MISSING", CONFIG_PATH, "active_pre_v3_run historical record is required")
    else:
        expected = {
            "classification": "PRE_V3_DEVELOPMENT_ONLY",
            "may_complete_naturally": True,
            "stop_modify_or_migrate_allowed": False,
            "high_frequency_monitoring_allowed": False,
        }
        for key, value in expected.items():
            _expect(active_run, key, value, CONFIG_PATH, issues, "PRE_V3_RUN_POLICY")
        prohibited = set(active_run.get("prohibited_uses", []))
        required = {"SET_V3_GATE", "SELECT_V3_METRIC", "CLAIM_CANDIDATE_SPECIFIC_EFFECT", "OVERTURN_ORIGINAL_M0_FAILURE"}
        if not required <= prohibited:
            _issue(issues, "PRE_V3_PROHIBITED_USE_COVERAGE", CONFIG_PATH, f"missing {sorted(required - prohibited)!r}")

    gate_records = supersession.get("historical_gate_records")
    if isinstance(gate_records, list):
        for record in gate_records:
            if not isinstance(record, Mapping):
                continue
            if record.get("gate_id") in {"O0", "G1"} and "PRE_V3_DEVELOPMENT_ONLY" not in str(record.get("v3_interpretation")):
                _issue(issues, "PRE_V3_GATE_CLASSIFICATION", SUPERSESSION_PATH, f"{record.get('gate_id')} must remain PRE_V3_DEVELOPMENT_ONLY")
            if record.get("gate_id") == "E0":
                if record.get("repository_reported_status") != "NO_GO" or record.get("sealed_final_status") != "NOT_EXECUTED":
                    _issue(issues, "HISTORICAL_E0_REWRITE", SUPERSESSION_PATH, "E0 NO_GO and sealed NOT_EXECUTED must be preserved")
    return issues


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def validate_measured_candidate_pool_record(record: Mapping[str, Any]) -> list[Issue]:
    """Validate pool cross-field invariants without loading any dataset."""

    path = f"{SCHEMA_DIR}/measured_candidate_pool.schema.json"
    issues: list[Issue] = []
    forbidden_parallel = {"candidate_ids", "candidate_sequence_sha256s", "unique_candidate_count", "same_pool_constraints"}
    present = sorted(forbidden_parallel & set(record))
    if present:
        _issue(issues, "POOL_PARALLEL_REPRESENTATION", path, f"single candidates[] representation forbids {present!r}")
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        _issue(issues, "POOL_CANDIDATES", path, "candidates must be a list")
        return issues
    count = record.get("candidate_count")
    if type(count) is not int or count != len(candidates):
        _issue(issues, "POOL_COUNT_MISMATCH", path, f"candidate_count {count!r} != len(candidates) {len(candidates)}")
    pool_type = record.get("pool_type")
    if pool_type == "PAIRWISE_ONLY" and len(candidates) != 2:
        _issue(issues, "POOL_PAIRWISE_SIZE", path, "PAIRWISE_ONLY must contain exactly two candidates")
    if pool_type in {"NDCG_ELIGIBLE", "DENSE_NEIGHBORHOOD"} and len(candidates) < 3:
        _issue(issues, "POOL_RANKING_SIZE", path, "NDCG/dense pools require at least three candidates")

    ids: list[str] = []
    canonical_ids: list[str] = []
    hashes: list[str] = []
    sequences: list[str] = []
    common_keys = ("biological_source_group_id", "study_id", "assay_id", "context_id", "endpoint_id", "region")
    for key in common_keys:
        if not isinstance(record.get(key), str) or not record.get(key):
            _issue(issues, "POOL_COMMON_KEY", path, f"pool {key} must be a non-empty string")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            _issue(issues, "POOL_CANDIDATE_ENTRY", path, f"candidate {index} is not an object")
            continue
        candidate_id = candidate.get("id")
        canonical_id = candidate.get("canonical_record_id")
        sequence = candidate.get("sequence")
        sequence_hash = candidate.get("sequence_hash")
        if not isinstance(candidate_id, str) or not candidate_id:
            _issue(issues, "POOL_CANDIDATE_ID", path, f"candidate {index} has no non-empty id")
        else:
            ids.append(candidate_id)
        if not isinstance(canonical_id, str) or not canonical_id:
            _issue(issues, "POOL_CANONICAL_RECORD_ID", path, f"candidate {index} has no non-empty canonical_record_id")
        else:
            canonical_ids.append(canonical_id)
        if not isinstance(sequence, str) or not sequence:
            _issue(issues, "POOL_CANDIDATE_SEQUENCE", path, f"candidate {index} has no full sequence")
        else:
            sequences.append(sequence)
        if not _is_sha256(sequence_hash):
            _issue(issues, "POOL_CANDIDATE_HASH", path, f"candidate {index} sequence_hash is invalid")
        else:
            hashes.append(sequence_hash)
            if isinstance(sequence, str) and sha256_bytes(sequence.encode("utf-8")) != sequence_hash:
                _issue(issues, "POOL_CANDIDATE_HASH_MISMATCH", path, f"candidate {index} sequence_hash does not bind sequence")
        for key in common_keys:
            if candidate.get(key) != record.get(key):
                _issue(issues, "POOL_COMMON_KEY_MISMATCH", path, f"candidate {index}.{key} differs from pool {key}")
    if len(ids) != len(set(ids)):
        _issue(issues, "POOL_DUPLICATE_CANDIDATE_ID", path, "candidate IDs must be unique")
    if len(canonical_ids) != len(set(canonical_ids)):
        _issue(issues, "POOL_DUPLICATE_CANONICAL_RECORD_ID", path, "canonical record IDs must be unique within one endpoint pool")
    if len(hashes) != len(set(hashes)) or len(sequences) != len(set(sequences)):
        _issue(issues, "POOL_DUPLICATE_CANDIDATE_SEQUENCE", path, "candidate sequences and sequence hashes must be unique")
    return issues


def validate_compute_ledger_record(record: Mapping[str, Any]) -> list[Issue]:
    """Validate matched-compute arithmetic and frozen HPO/source/action bindings."""

    path = f"{SCHEMA_DIR}/compute_ledger.schema.json"
    issues: list[Issue] = []
    for key in ("source_pool_hash", "legal_action_space_hash"):
        if not _is_sha256(record.get(key)):
            _issue(issues, "COMPUTE_BINDING_HASH", path, f"{key} must be a lowercase SHA-256")
    budget = record.get("candidate_budget")
    count = record.get("candidate_count")
    unique = record.get("unique_candidate_count")
    if not all(type(value) is int and value >= 0 for value in (budget, count, unique)):
        _issue(issues, "COMPUTE_CANDIDATE_COUNTS", path, "candidate budget/count/unique count must be non-negative integers")
    elif not unique <= count <= budget:
        _issue(issues, "COMPUTE_CANDIDATE_INEQUALITY", path, "unique_candidate_count <= candidate_count <= candidate_budget is required")
    rate = record.get("unique_candidate_rate")
    expected_rate = (unique / count) if isinstance(count, int) and count > 0 and isinstance(unique, int) else 0.0
    if not isinstance(rate, (int, float)) or abs(float(rate) - expected_rate) > 1e-12:
        _issue(issues, "COMPUTE_UNIQUE_RATE", path, f"unique_candidate_rate must equal {expected_rate}")

    rule = record.get("forward_equivalent_rule")
    forward_fields = {
        "generator_nfe": "generator_weight",
        "critic_forwards": "critic_weight",
        "guidance_forwards": "guidance_weight",
        "reranker_forwards": "reranker_weight",
        "other_forwards": "other_weight",
    }
    if not isinstance(rule, Mapping) or not _is_sha256(rule.get("rule_sha256")):
        _issue(issues, "COMPUTE_FORWARD_RULE", path, "forward-equivalent rule and hash are required")
    else:
        canonical_rule = {
            key: rule.get(key)
            for key in (
                "rule_id",
                "generator_weight",
                "critic_weight",
                "guidance_weight",
                "reranker_weight",
                "other_weight",
            )
        }
        canonical_rule_bytes = json.dumps(
            canonical_rule,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        if sha256_bytes(canonical_rule_bytes) != rule.get("rule_sha256"):
            _issue(issues, "COMPUTE_FORWARD_RULE_HASH", path, "rule_sha256 must bind the frozen rule ID and weights")
        total = 0.0
        arithmetic_valid = True
        for count_key, weight_key in forward_fields.items():
            value = record.get(count_key)
            weight = rule.get(weight_key)
            if type(value) is not int or value < 0 or not isinstance(weight, (int, float)) or weight < 0:
                arithmetic_valid = False
                break
            total += value * float(weight)
        declared_total = record.get("total_forward_equivalents")
        if not arithmetic_valid or not isinstance(declared_total, (int, float)) or abs(float(declared_total) - total) > 1e-9:
            _issue(issues, "COMPUTE_FORWARD_EQUIVALENT", path, f"total_forward_equivalents must equal frozen weighted total {total}")

    seeds = record.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        _issue(issues, "COMPUTE_SEEDS", path, "seeds must be a non-empty unique non-negative integer list")
    hpo = record.get("hpo_budget")
    if not isinstance(hpo, Mapping):
        _issue(issues, "COMPUTE_HPO_BUDGET", path, "hpo_budget mapping is required")
    else:
        trials = hpo.get("trial_count")
        maximum = hpo.get("max_trials")
        if type(trials) is not int or type(maximum) is not int or not 0 <= trials <= maximum or maximum < 1:
            _issue(issues, "COMPUTE_HPO_TRIALS", path, "0 <= trial_count <= max_trials with max_trials >= 1 is required")
        if not _is_sha256(hpo.get("search_space_sha256")):
            _issue(issues, "COMPUTE_HPO_HASH", path, "HPO search space must be hash-bound")
        kind = hpo.get("budget_type")
        if kind not in {"MAX_TRIALS", "WALL_TIME_SECONDS", "FORWARD_EQUIVALENTS", "JOINT"}:
            _issue(issues, "COMPUTE_HPO_BUDGET_TYPE", path, "HPO budget_type is outside the frozen vocabulary")
        time_budget = hpo.get("time_budget_seconds")
        forward_budget = hpo.get("forward_equivalent_budget")
        if kind in {"WALL_TIME_SECONDS", "JOINT"} and (
            not isinstance(time_budget, (int, float)) or isinstance(time_budget, bool) or time_budget <= 0
        ):
            _issue(issues, "COMPUTE_HPO_TIME_BUDGET", path, "selected HPO time budget must be positive")
        if kind in {"FORWARD_EQUIVALENTS", "JOINT"} and (
            not isinstance(forward_budget, (int, float)) or isinstance(forward_budget, bool) or forward_budget <= 0
        ):
            _issue(issues, "COMPUTE_HPO_FORWARD_BUDGET", path, "selected HPO forward-equivalent budget must be positive")
    return issues


def validate_gate_record(record: Mapping[str, Any]) -> list[Issue]:
    """Validate gate decision/evidence/claim consistency and PASS sufficiency."""

    path = f"{SCHEMA_DIR}/gate_record.schema.json"
    issues: list[Issue] = []
    decision = record.get("decision")
    evidence = record.get("evidence_status")
    claim = record.get("claim_status")
    eligible = record.get("claim_eligible")
    expected_evidence = {
        "PASS": "PASS",
        "NOT_RUN": "NOT_RUN",
        "FAIL_CURRENT_PROTOCOL": "FAIL_CURRENT_PROTOCOL",
        "FAIL_REPAIRABLE": "FAIL_REPAIRABLE",
        "BLOCKED_PENDING_PUBLIC_EVIDENCE": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "TERMINATED_SAFELY_WITH_EVIDENCE": "TERMINATED_SAFELY_WITH_EVIDENCE",
    }
    if decision in expected_evidence and evidence != expected_evidence[decision]:
        _issue(issues, "GATE_DECISION_EVIDENCE", path, f"decision {decision!r} requires evidence_status {expected_evidence[decision]!r}")
    if evidence == "PASS" and decision != "PASS":
        _issue(issues, "GATE_EVIDENCE_WITHOUT_DECISION", path, "PASS evidence requires a PASS decision")
    non_pass_decisions = set(expected_evidence) - {"PASS"}
    if decision in non_pass_decisions and (eligible is not False or claim == "ESTABLISHED"):
        _issue(issues, "GATE_NONPASS_CLAIM", path, "a non-PASS decision must be claim-ineligible and not ESTABLISHED")
    if decision == "NOT_RUN" and (claim == "ESTABLISHED" or eligible is not False):
        _issue(issues, "GATE_NOT_RUN_CLAIM", path, "NOT_RUN must be claim-ineligible and not ESTABLISHED")
    if claim == "ESTABLISHED" and not (decision == "PASS" and evidence == "PASS" and eligible is True):
        _issue(issues, "GATE_ESTABLISHED_WITHOUT_PASS", path, "ESTABLISHED requires PASS evidence/decision and claim_eligible=true")
    if eligible is True and not (decision == "PASS" and evidence == "PASS"):
        _issue(issues, "GATE_ELIGIBLE_WITHOUT_PASS", path, "claim_eligible=true requires PASS")
    if decision == "PASS":
        run_ids = record.get("run_ids")
        seeds = record.get("seeds")
        results = record.get("per_study_results")
        if not isinstance(run_ids, list) or not run_ids or len(run_ids) != len(set(run_ids)):
            _issue(issues, "GATE_PASS_RUNS", path, "PASS requires at least one unique run ID")
        if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
            _issue(issues, "GATE_PASS_SEEDS", path, "PASS requires at least one unique seed")
        elif any(type(seed) is not int or seed < 0 for seed in seeds):
            _issue(issues, "GATE_PASS_SEEDS", path, "PASS seeds must be non-negative integers")
        if not isinstance(results, list) or not results:
            _issue(issues, "GATE_PASS_RESULTS", path, "PASS requires at least one result")
        if record.get("gate_family") == "CRITIC_EFFECT":
            study_ids = {
                result.get("study_id")
                for result in results or []
                if isinstance(result, Mapping) and isinstance(result.get("study_id"), str)
            }
            if len(study_ids) < 3:
                _issue(issues, "GATE_CRITIC_STUDIES", path, "CRITIC_EFFECT PASS requires at least three distinct studies")
            if not isinstance(seeds, list) or len(seeds) != 5 or len(set(seeds)) != 5:
                _issue(issues, "GATE_CRITIC_SEEDS", path, "CRITIC_EFFECT PASS requires exactly five unique seeds")
    if decision in non_pass_decisions - {"NOT_RUN"}:
        if not isinstance(record.get("failure_bundle"), Mapping):
            _issue(issues, "GATE_FAILURE_BUNDLE", path, "a failed, blocked or terminated gate requires a failure bundle")
        if not isinstance(record.get("next_route_a_recovery_task"), Mapping):
            _issue(issues, "GATE_RECOVERY_TASK", path, "a failed, blocked or terminated gate requires a Route-A recovery task")
    return issues


def validate_run_manifest_record(record: Mapping[str, Any]) -> list[Issue]:
    """Validate the closed CPU/GPU execution class and recovery semantics."""

    path = f"{SCHEMA_DIR}/run_manifest.schema.json"
    issues: list[Issue] = []
    gpu = record.get("gpu")
    environment = record.get("environment")
    status = record.get("run_status")
    evidence = record.get("evidence_status")
    claim = record.get("claim_status")
    compute_class = record.get("compute_class")
    parameter_updating = record.get("parameter_updating")
    failure = record.get("failure")
    cuda_failure = isinstance(failure, Mapping) and failure.get("failure_type") in {"CUDA_UNAVAILABLE", "CPU_FALLBACK"}

    if compute_class not in RUN_COMPUTE_CLASSES:
        _issue(issues, "RUN_COMPUTE_CLASS", path, "compute_class is outside the closed Route-A V3 vocabulary")
    if type(parameter_updating) is not bool:
        _issue(issues, "RUN_PARAMETER_UPDATING", path, "parameter_updating must be a boolean")
    if not isinstance(gpu, Mapping):
        _issue(issues, "RUN_GPU_RECORD", path, "every run requires an explicit GPU policy/usage record")
        gpu = {}
    if gpu.get("cuda_fail_closed") is not True or gpu.get("silent_cpu_fallback") is not False:
        _issue(issues, "RUN_GPU_FAIL_CLOSED_POLICY", path, "all compute classes must fail closed and forbid silent CPU fallback")

    if compute_class in CPU_COMPUTE_CLASSES:
        if parameter_updating is not False:
            _issue(issues, "RUN_CPU_PARAMETER_UPDATE", path, "CPU compute classes must set parameter_updating=false")
        if gpu.get("required") is not False or gpu.get("used") is not False:
            _issue(issues, "RUN_CPU_GPU_POLICY", path, "CPU compute classes must set gpu.required=false and gpu.used=false")
        if claim == "ESTABLISHED":
            _issue(issues, "RUN_CPU_CLAIM", path, "a CPU engineering/statistical run cannot itself establish a scientific claim")
    elif compute_class in GPU_TRAIN_COMPUTE_CLASSES:
        if parameter_updating is not True:
            _issue(issues, "RUN_GPU_TRAIN_PARAMETER_UPDATE", path, "GPU training classes must set parameter_updating=true")
        if gpu.get("required") is not True:
            _issue(issues, "RUN_GPU_REQUIRED_POLICY", path, "GPU training classes must set gpu.required=true")
    elif compute_class in GPU_VALIDATION_COMPUTE_CLASSES:
        if parameter_updating is not False:
            _issue(issues, "RUN_GPU_VALIDATION_PARAMETER_UPDATE", path, "GPU_VALIDATION must set parameter_updating=false")
        if gpu.get("required") is not True:
            _issue(issues, "RUN_GPU_REQUIRED_POLICY", path, "GPU_VALIDATION must set gpu.required=true")

    if parameter_updating is True and compute_class not in GPU_TRAIN_COMPUTE_CLASSES:
        _issue(issues, "RUN_PARAMETER_UPDATE_CLASS", path, "parameter_updating=true is legal only for the four GPU training classes")

    successful = status == "COMPLETED" or evidence == "PASS"
    if successful:
        if status != "COMPLETED":
            _issue(issues, "RUN_PASS_NOT_COMPLETED", path, "PASS evidence requires COMPLETED run status")
        ended_at = record.get("ended_at")
        if not isinstance(ended_at, str) or not ended_at:
            _issue(issues, "RUN_SUCCESS_ENDED_AT", path, "a completed/PASS run requires a non-empty ended_at timestamp")
        outputs = record.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            _issue(issues, "RUN_SUCCESS_OUTPUTS", path, "a completed/PASS run requires at least one output")
        else:
            for index, output in enumerate(outputs):
                valid_path = (
                    isinstance(output, Mapping)
                    and isinstance(output.get("absolute_path"), str)
                    and output.get("absolute_path", "").startswith("/")
                )
                if (
                    not isinstance(output, Mapping)
                    or output.get("status") != "COMPLETE"
                    or not _is_sha256(output.get("sha256"))
                    or not valid_path
                ):
                    _issue(issues, "RUN_SUCCESS_OUTPUT", path, f"output {index} must be COMPLETE with an absolute path and SHA-256")

    def validate_used_gpu_metadata() -> None:
        required_gpu = ("uuid", "model", "device", "driver_version", "cuda_version")
        if any(not isinstance(gpu.get(key), str) or not gpu.get(key) for key in required_gpu):
            _issue(issues, "RUN_GPU_METADATA", path, "gpu.used=true requires UUID/model/device/driver/CUDA metadata")
        device = gpu.get("device")
        valid_device = device == "cuda" or (
            isinstance(device, str) and device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
        )
        if not valid_device:
            _issue(issues, "RUN_GPU_DEVICE", path, "gpu.used=true requires a CUDA device")
        if type(gpu.get("peak_vram_bytes")) is not int or gpu.get("peak_vram_bytes") <= 0:
            _issue(issues, "RUN_GPU_VRAM", path, "gpu.used=true requires positive peak VRAM")
        if not isinstance(environment, Mapping) or not isinstance(environment.get("pytorch_version"), str) or not environment.get("pytorch_version"):
            _issue(issues, "RUN_PYTORCH_VERSION", path, "gpu.used=true requires a PyTorch version")

    if compute_class in GPU_COMPUTE_CLASSES:
        if status not in GPU_LIFECYCLE_RUN_STATUSES:
            _issue(issues, "RUN_GPU_LIFECYCLE_STATUS", path, "GPU compute_class has an unsupported lifecycle run_status")
        if gpu.get("required") is not True:
            _issue(issues, "RUN_GPU_REQUIRED_POLICY", path, "every GPU lifecycle state requires gpu.required=true")

        if status in GPU_PRESTART_RUN_STATUSES:
            if gpu.get("used") is not False:
                _issue(issues, "RUN_GPU_PRESTART_POLICY", path, "NOT_RUN/QUEUED GPU work must set gpu.used=false")
            if record.get("ended_at") is not None:
                _issue(issues, "RUN_GPU_PRESTART_ENDED_AT", path, "NOT_RUN/QUEUED GPU work must keep ended_at=null")
            if failure is not None or record.get("recovery") is not None:
                _issue(issues, "RUN_GPU_NONTERMINAL_FAILURE", path, "NOT_RUN/QUEUED GPU work cannot carry terminal failure/recovery records")
        elif status == "IN_PROGRESS":
            if gpu.get("used") is not True:
                _issue(issues, "RUN_GPU_IN_PROGRESS_POLICY", path, "IN_PROGRESS GPU work must set gpu.used=true")
            if record.get("ended_at") is not None:
                _issue(issues, "RUN_GPU_IN_PROGRESS_ENDED_AT", path, "IN_PROGRESS GPU work must keep ended_at=null")
            if failure is not None or record.get("recovery") is not None:
                _issue(issues, "RUN_GPU_NONTERMINAL_FAILURE", path, "IN_PROGRESS GPU work cannot carry terminal failure/recovery records")
        elif status == "COMPLETED":
            if gpu.get("used") is not True:
                _issue(issues, "RUN_GPU_SUCCESS_POLICY", path, "COMPLETED GPU work must set gpu.used=true")
            if failure is not None or record.get("recovery") is not None:
                _issue(issues, "RUN_GPU_COMPLETED_FAILURE", path, "COMPLETED GPU work cannot carry failure/recovery records")
        elif status in GPU_FAILURE_RUN_STATUSES and type(gpu.get("used")) is not bool:
            _issue(issues, "RUN_GPU_FAILURE_USAGE", path, "failed/terminated GPU work must truthfully record gpu.used as boolean")

        if gpu.get("used") is True:
            validate_used_gpu_metadata()
        elif gpu.get("used") is False:
            peak_vram = gpu.get("peak_vram_bytes")
            zero_or_null_vram = peak_vram is None or (type(peak_vram) is int and peak_vram == 0)
            if gpu.get("device") is not None or not zero_or_null_vram:
                _issue(issues, "RUN_GPU_UNUSED_TELEMETRY", path, "gpu.used=false requires device=null and peak_vram_bytes null or zero")

    if cuda_failure:
        if status not in {"FAIL_CLOSED", "TERMINATED", "TERMINATED_SAFELY_WITH_EVIDENCE"}:
            _issue(issues, "RUN_CUDA_FAILURE_STATUS", path, "CUDA unavailable/fallback must fail closed or terminate safely")
        if compute_class not in GPU_COMPUTE_CLASSES:
            _issue(issues, "RUN_CUDA_FAILURE_CLASS", path, "CUDA unavailable/fallback is valid only for an explicit GPU compute class")
        if gpu.get("required") is not True or gpu.get("used") is not False or gpu.get("cuda_fail_closed") is not True or gpu.get("silent_cpu_fallback") is not False:
            _issue(issues, "RUN_CUDA_FAILURE_GPU", path, "CUDA failure record must show no GPU use and no silent CPU fallback")
        if not isinstance(failure.get("failure_bundle_path"), str) or not failure.get("failure_bundle_path", "").startswith("/") or not _is_sha256(failure.get("failure_bundle_sha256")):
            _issue(issues, "RUN_CUDA_FAILURE_BUNDLE", path, "CUDA failure requires an absolute, hash-bound failure bundle")
        if not isinstance(record.get("recovery"), Mapping):
            _issue(issues, "RUN_FAILURE_RECOVERY", path, "CUDA unavailable/fallback requires a recovery record")
        if not isinstance(record.get("ended_at"), str) or not record.get("ended_at"):
            _issue(issues, "RUN_FAILURE_ENDED_AT", path, "CUDA unavailable/fallback requires ended_at")
    if evidence == "PASS" and status != "COMPLETED":
        _issue(issues, "RUN_EVIDENCE_STATUS", path, "PASS evidence requires COMPLETED run status")
    failure_statuses = GPU_FAILURE_RUN_STATUSES
    if status in failure_statuses:
        if not isinstance(record.get("ended_at"), str) or not record.get("ended_at"):
            _issue(issues, "RUN_FAILURE_ENDED_AT", path, "failed, blocked or terminated status requires ended_at")
        if not isinstance(failure, Mapping):
            _issue(issues, "RUN_FAILURE_RECORD", path, "failed, blocked or terminated status requires a failure record")
        elif not isinstance(failure.get("failure_bundle_path"), str) or not failure.get("failure_bundle_path", "").startswith("/") or not _is_sha256(failure.get("failure_bundle_sha256")):
            _issue(issues, "RUN_FAILURE_BUNDLE", path, "failed, blocked or terminated status requires an absolute, hash-bound failure bundle")
        if not isinstance(record.get("recovery"), Mapping):
            _issue(issues, "RUN_FAILURE_RECOVERY", path, "failed, blocked or terminated status requires a recovery record")
    return issues


def _object_schemas_without_closed_properties(node: Any, pointer: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(node, Mapping):
        node_type = node.get("type")
        is_object = node_type == "object" or (isinstance(node_type, list) and "object" in node_type)
        if is_object and node.get("additionalProperties") is not False:
            failures.append(pointer)
        for key, value in node.items():
            failures.extend(_object_schemas_without_closed_properties(value, f"{pointer}/{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            failures.extend(_object_schemas_without_closed_properties(value, f"{pointer}/{index}"))
    return failures


def _schema_structure_errors(schema: Mapping[str, Any]) -> list[str]:
    """Minimal draft-independent checks used when ``jsonschema`` is absent."""

    errors: list[str] = []
    defs = schema.get("$defs")
    known_defs = set(defs) if isinstance(defs, Mapping) else set()

    def walk(node: Any, pointer: str) -> None:
        if isinstance(node, Mapping):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name not in known_defs:
                    errors.append(f"{pointer}/$ref points to missing $defs/{name}")
            node_type = node.get("type")
            is_object = node_type == "object" or (isinstance(node_type, list) and "object" in node_type)
            if is_object:
                required = node.get("required", [])
                properties = node.get("properties", {})
                if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
                    errors.append(f"{pointer}/required is not a string list")
                elif not isinstance(properties, Mapping):
                    errors.append(f"{pointer}/properties is not an object")
                else:
                    missing = sorted(set(required) - set(properties))
                    if missing:
                        errors.append(f"{pointer}/required names missing properties {missing!r}")
            for key, value in node.items():
                walk(value, f"{pointer}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{pointer}/{index}")

    walk(schema, "$")
    return errors


def build_expected_schema_manifest(repo_root: Path) -> tuple[dict[str, Any], str]:
    entries: list[dict[str, str]] = []
    for filename in SCHEMA_FILES:
        relative = f"{SCHEMA_DIR}/{filename}"
        schema = _load_json(repo_root, relative)
        entries.append(
            {
                "$id": str(schema.get("$id", "")),
                "contract_id": str(schema.get("contract_id", "")),
                "filename": filename,
                "schema_version": str(schema.get("schema_version", "")),
                "sha256": sha256_bytes(_read_bytes(repo_root, relative)),
            }
        )
    manifest = {
        "contract_id": CONTRACT_ID,
        "manifest_version": VERSION,
        "schema_count": len(SCHEMA_FILES),
        "schema_version": VERSION,
        "schemas": entries,
    }
    sums = "".join(f"{entry['sha256']}  {entry['filename']}\n" for entry in entries)
    return manifest, sums


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_schema_manifests(repo_root: Path) -> None:
    """Opt-in deterministic write limited to the two schema manifest files."""

    schema_dir = _safe_repo_path(repo_root, SCHEMA_DIR, must_exist=False)
    if not schema_dir.is_dir() or schema_dir.is_symlink():
        raise FileNotFoundError(f"schema directory must already exist and not be a symlink: {schema_dir}")
    manifest, sums = build_expected_schema_manifest(repo_root)
    manifest_path = _safe_repo_path(repo_root, SCHEMA_MANIFEST, must_exist=False)
    sums_path = _safe_repo_path(repo_root, SCHEMA_SUMS, must_exist=False)
    manifest_path.write_bytes(_json_bytes(manifest))
    sums_path.write_bytes(sums.encode("utf-8"))


def validate_schema_manifest(repo_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    schemas: list[Mapping[str, Any]] = []
    ids: list[str] = []
    for filename in SCHEMA_FILES:
        relative = f"{SCHEMA_DIR}/{filename}"
        try:
            schema = _load_json(repo_root, relative)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            _issue(issues, "SCHEMA_UNREADABLE", relative, str(exc))
            continue
        schemas.append(schema)
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            _issue(issues, "SCHEMA_DRAFT", relative, "schema must use JSON Schema draft 2020-12")
        if schema.get("contract_id") != CONTRACT_ID or schema.get("schema_version") != VERSION:
            _issue(issues, "SCHEMA_AUTHORITY_METADATA", relative, "schema contract_id/schema_version mismatch")
        expected_id = f"https://github.com/Cunyu-Liu/mRNA_editflow/{relative}"
        if schema.get("$id") != expected_id:
            _issue(issues, "SCHEMA_ID", relative, f"$id must be {expected_id}")
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            _issue(issues, "SCHEMA_TOP_LEVEL_OPEN", relative, "top-level object must set additionalProperties=false")
        open_objects = _object_schemas_without_closed_properties(schema)
        if open_objects:
            _issue(issues, "SCHEMA_NESTED_OBJECT_OPEN", relative, f"object schemas missing additionalProperties=false: {open_objects!r}")
        structure_errors = _schema_structure_errors(schema)
        if structure_errors:
            _issue(issues, "SCHEMA_STRUCTURE", relative, f"structural errors: {structure_errors!r}")
        if not isinstance(schema.get("required"), list) or not isinstance(schema.get("properties"), Mapping):
            _issue(issues, "SCHEMA_STRUCTURE", relative, "top-level required/properties are required")
        if isinstance(schema.get("$id"), str):
            ids.append(schema["$id"])
    if len(ids) != len(set(ids)):
        _issue(issues, "SCHEMA_ID_DUPLICATE", SCHEMA_DIR, "schema $id values must be unique")
    actual_files = sorted(path.name for path in (repo_root / SCHEMA_DIR).glob("*.schema.json")) if (repo_root / SCHEMA_DIR).is_dir() else []
    if actual_files != sorted(SCHEMA_FILES):
        _issue(issues, "SCHEMA_FILENAME_SET", SCHEMA_DIR, f"expected exactly {list(SCHEMA_FILES)!r}, got {actual_files!r}")

    if len(schemas) != len(SCHEMA_FILES):
        return issues
    try:
        expected_manifest, expected_sums = build_expected_schema_manifest(repo_root)
        actual_manifest = _load_json(repo_root, SCHEMA_MANIFEST)
        actual_manifest_bytes = _read_bytes(repo_root, SCHEMA_MANIFEST)
        actual_sums = _read_text(repo_root, SCHEMA_SUMS)
        if actual_manifest != expected_manifest:
            _issue(issues, "SCHEMA_MANIFEST_CONTENT", SCHEMA_MANIFEST, "manifest metadata/hash entries are stale or malformed")
        if actual_manifest_bytes != _json_bytes(expected_manifest):
            _issue(issues, "SCHEMA_MANIFEST_ENCODING", SCHEMA_MANIFEST, "manifest must be deterministic sorted/indented JSON with a trailing LF")
        if actual_sums != expected_sums:
            _issue(issues, "SCHEMA_SUMS_CONTENT", SCHEMA_SUMS, "SCHEMA_SHA256SUMS is stale, unsorted or malformed")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "SCHEMA_MANIFEST_UNREADABLE", SCHEMA_DIR, str(exc))
    return issues


def validate_python_static_safety(repo_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    scripts_dir = repo_root / "scripts" / "route_a_v3"
    if not scripts_dir.is_dir():
        _issue(issues, "ROUTE_A_SCRIPT_DIR_MISSING", "scripts/route_a_v3", "script directory is required")
        return issues
    forbidden_import_roots = {"torch"}
    forbidden_import_fragments = {"e0x.sealed", "run_e0x_final"}
    forbidden_calls = {"SealedAccessState", "compare_and_append", "run_sealed_final", "append_intent", "reserve"}
    for path in sorted(scripts_dir.glob("*.py")):
        relative = path.relative_to(repo_root).as_posix()
        if path.is_symlink():
            _issue(issues, "UNSAFE_SCRIPT_SYMLINK", relative, "route_a_v3 scripts may not be symlinks")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError) as exc:
            _issue(issues, "PYTHON_AST_ERROR", relative, str(exc))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in forbidden_import_roots or any(fragment in alias.name for fragment in forbidden_import_fragments):
                        _issue(issues, "FORBIDDEN_IMPORT", relative, f"line {node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".", 1)[0]
                names = {alias.name for alias in node.names}
                if root in forbidden_import_roots or any(fragment in module for fragment in forbidden_import_fragments) or names & forbidden_calls:
                    _issue(issues, "FORBIDDEN_IMPORT", relative, f"line {node.lineno}: from {module} import {sorted(names)!r}")
            elif isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name in forbidden_calls:
                    _issue(issues, "FORBIDDEN_SEALED_STATE_CALL", relative, f"line {node.lineno}: call to {name}")
    return issues


def _ast_function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _ast_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _ast_call_names(node: ast.AST) -> set[str]:
    return {
        name
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and (name := _ast_call_name(child)) is not None
    }


def _first_executable_statement(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.stmt | None:
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body[0] if body else None


def _is_guard_call_statement(
    statement: ast.stmt,
    argument_name: str = "args",
) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    return (
        _ast_call_name(call) == "assert_sealed_final_authorized"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == argument_name
        and not call.keywords
    )


def _is_sealed_final_test(test: ast.AST) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Attribute)
        and isinstance(test.left.value, ast.Name)
        and test.left.value.id == "args"
        and test.left.attr == "mode"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "sealed-final"
    )


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for statement in tree.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                try:
                    return ast.literal_eval(statement.value)
                except (TypeError, ValueError):
                    value = statement.value
                    if (
                        isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Name)
                        and value.func.id == "frozenset"
                        and len(value.args) == 1
                        and not value.keywords
                    ):
                        try:
                            return frozenset(ast.literal_eval(value.args[0]))
                        except (TypeError, ValueError):
                            pass
                    return None
    return None


def validate_runner_and_guard_ast(repo_root: Path) -> list[Issue]:
    """Statically prove the A0--A9 unconditional sealed hard-disable boundary.

    The check parses source only.  It never imports either module and never
    reads config, authorization, readiness, invocation, restricted, or access
    state paths.
    """

    issues: list[Issue] = []
    try:
        runner_source = _read_text(repo_root, SEALED_RUNNER_PATH)
        guard_source = _read_text(repo_root, SEALED_GUARD_PATH)
        runner_tree = ast.parse(runner_source, filename=SEALED_RUNNER_PATH)
        guard_tree = ast.parse(guard_source, filename=SEALED_GUARD_PATH)
    except (FileNotFoundError, ValueError, OSError, SyntaxError, UnicodeDecodeError) as exc:
        _issue(issues, "SEALED_AST_UNREADABLE", SEALED_RUNNER_PATH, str(exc))
        return issues

    def import_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        return [node.module or ""]

    def exact_guard_import(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "scripts.route_a_v3.sealed_guard"
            and len(node.names) == 1
            and node.names[0].name == "assert_sealed_final_authorized"
            and node.names[0].asname is None
        )

    def exact_sealed_runtime_import(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "scripts.e0x"
            and len(node.names) == 1
            and node.names[0].name == "sealed"
            and node.names[0].asname is None
        )

    def exact_hard_disable_raise(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Raise)
            and node.cause is None
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "RouteAV3SealedHardDisabled"
            and len(node.exc.args) == 1
            and isinstance(node.exc.args[0], ast.Name)
            and node.exc.args[0].id == "HARD_DISABLED"
            and not node.exc.keywords
        )

    # Import provenance must be guarded before any local runtime module loads.
    # Only the tiny unconditional guard itself may be imported at module scope.
    project_import_nodes: list[ast.AST] = []
    eager_runtime_modules: set[str] = set()
    runtime_roots = {"torch", "numpy", "scipy", "sklearn"}
    for statement in runner_tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            modules = import_modules(node)
            if any(module == "scripts" or module.startswith("scripts.") for module in modules):
                project_import_nodes.append(node)
            eager_runtime_modules.update(
                module
                for module in modules
                if module.split(".", 1)[0] in runtime_roots
            )
    if len(project_import_nodes) != 1 or not exact_guard_import(project_import_nodes[0]):
        _issue(
            issues,
            "RUNNER_MODULE_PROJECT_IMPORT",
            SEALED_RUNNER_PATH,
            "module scope may import only assert_sealed_final_authorized from scripts.route_a_v3.sealed_guard",
        )
    if eager_runtime_modules:
        _issue(
            issues,
            "RUNNER_MODULE_RUNTIME_IMPORT",
            SEALED_RUNNER_PATH,
            f"runtime imports must remain behind the parsed-mode guard: {sorted(eager_runtime_modules)!r}",
        )

    main_function = _ast_function(runner_tree, "main")
    sealed_function = _ast_function(runner_tree, "run_sealed_final")
    if main_function is None:
        _issue(issues, "RUNNER_MAIN_MISSING", SEALED_RUNNER_PATH, "main function is required")
    else:
        parse_index = None
        for index, statement in enumerate(main_function.body):
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            value = statement.value
            if (
                isinstance(target, ast.Name)
                and target.id == "args"
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "parse_args"
                and not value.args
                and not value.keywords
            ):
                parse_index = index
                break
        if parse_index is None:
            _issue(issues, "RUNNER_PARSE_ARGS_MISSING", SEALED_RUNNER_PATH, "main must assign args directly from parse_args()")
        elif parse_index + 1 >= len(main_function.body):
            _issue(issues, "RUNNER_EARLY_GUARD_MISSING", SEALED_RUNNER_PATH, "sealed-final guard must immediately follow parse_args")
        else:
            guard_if = main_function.body[parse_index + 1]
            valid_if = (
                isinstance(guard_if, ast.If)
                and _is_sealed_final_test(guard_if.test)
                and len(guard_if.body) == 1
                and _is_guard_call_statement(guard_if.body[0])
                and not guard_if.orelse
            )
            if not valid_if:
                _issue(
                    issues,
                    "RUNNER_EARLY_GUARD_MISSING",
                    SEALED_RUNNER_PATH,
                    "parse_args must be followed by an exact one-statement sealed-final guard",
                )
            else:
                guard_line = guard_if.body[0].lineno
                sensitive_lines: list[int] = []
                observed_local_imports: set[tuple[str, tuple[str, ...]]] = set()
                observed_calls: set[str] = set()
                sensitive_arg_fields = {
                    "dataset",
                    "prereg",
                    "ckpt_dir",
                    "restricted",
                    "raw_seq_dir",
                    "out_dir",
                    "gpu",
                }
                for node in ast.walk(main_function):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        modules = import_modules(node)
                        if any(
                            module == "scripts"
                            or module.startswith("scripts.")
                            or module.split(".", 1)[0] in runtime_roots
                            for module in modules
                        ):
                            sensitive_lines.append(node.lineno)
                        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("scripts."):
                            observed_local_imports.add(
                                (node.module or "", tuple(alias.name for alias in node.names))
                            )
                    elif isinstance(node, ast.Call):
                        name = _ast_call_name(node)
                        if name is not None:
                            observed_calls.add(name)
                        if name in {
                            "load_prereg",
                            "load_rows",
                            "build_vocab",
                            "get_config",
                            "select_device",
                            "manual_seed",
                            "manual_seed_all",
                        }:
                            sensitive_lines.append(node.lineno)
                    elif (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "args"
                        and node.attr in sensitive_arg_fields
                    ):
                        sensitive_lines.append(node.lineno)
                required_local_imports = {
                    ("scripts.e0x", ("prereg",)),
                    ("scripts.m4_sparse", ("config",)),
                    ("scripts.m4_sparse.dataset", ("build_vocab",)),
                }
                if not required_local_imports <= observed_local_imports:
                    _issue(
                        issues,
                        "RUNNER_RUNTIME_IMPORT_CLOSURE",
                        SEALED_RUNNER_PATH,
                        f"main is missing frozen post-guard imports {sorted(required_local_imports - observed_local_imports)!r}",
                    )
                required_calls = {"load_prereg", "load_rows", "build_vocab", "get_config", "select_device"}
                if not required_calls <= observed_calls:
                    _issue(
                        issues,
                        "RUNNER_RUNTIME_ANCHORS",
                        SEALED_RUNNER_PATH,
                        f"main is missing runtime anchors {sorted(required_calls - observed_calls)!r}",
                    )
                if sensitive_lines and guard_line >= min(sensitive_lines):
                    _issue(
                        issues,
                        "RUNNER_GUARD_ORDER",
                        SEALED_RUNNER_PATH,
                        "main guard must precede local runtime imports, prereg, data, GPU, torch, and path use",
                    )

    if sealed_function is None:
        _issue(issues, "RUN_SEALED_FINAL_MISSING", SEALED_RUNNER_PATH, "run_sealed_final function is required")
    else:
        body = list(sealed_function.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        first = body[0] if body else None
        if first is None or not _is_guard_call_statement(first):
            _issue(
                issues,
                "RUN_SEALED_FIRST_GUARD",
                SEALED_RUNNER_PATH,
                "run_sealed_final must begin with the unconditional guard expression",
            )
        else:
            guard_line = first.lineno
            guard_calls = [
                node
                for node in ast.walk(sealed_function)
                if isinstance(node, ast.Call)
                and _ast_call_name(node) == "assert_sealed_final_authorized"
            ]
            if len(guard_calls) != 1:
                _issue(
                    issues,
                    "RUN_SEALED_FIRST_GUARD",
                    SEALED_RUNNER_PATH,
                    "run_sealed_final must contain exactly one guard call at its first executable statement",
                )
            second = body[1] if len(body) > 1 else None
            if second is None or not exact_sealed_runtime_import(second):
                _issue(
                    issues,
                    "RUN_SEALED_RUNTIME_IMPORT",
                    SEALED_RUNNER_PATH,
                    "the first post-guard statement must import scripts.e0x.sealed",
                )
            sensitive_lines: list[int] = []
            observed_calls: set[str] = set()
            restricted_seen = False
            for node in ast.walk(sealed_function):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    sensitive_lines.append(node.lineno)
                elif isinstance(node, ast.Call):
                    name = _ast_call_name(node)
                    if name is not None:
                        observed_calls.add(name)
                    if name in {"SealedAccessState", "append_intent", "reserve", "complete", "abort"}:
                        sensitive_lines.append(node.lineno)
                elif (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "args"
                    and node.attr == "restricted"
                ):
                    restricted_seen = True
                    sensitive_lines.append(node.lineno)
            required_protocol_calls = {"SealedAccessState", "append_intent", "reserve"}
            if not restricted_seen or not required_protocol_calls <= observed_calls:
                _issue(
                    issues,
                    "RUN_SEALED_PROTOCOL_ANCHORS",
                    SEALED_RUNNER_PATH,
                    "run_sealed_final must retain restricted state plus intent/reservation blockers behind the guard",
                )
            if sensitive_lines and guard_line >= min(sensitive_lines):
                _issue(
                    issues,
                    "RUN_SEALED_GUARD_ORDER",
                    SEALED_RUNNER_PATH,
                    "defense guard must precede sealed imports, restricted paths, state, intent, and reservation",
                )

    # A0--A9 deliberately contains no authorization implementation.  Freeze the
    # whole guard module shape so a helper, toggle, manifest read, or hidden
    # reachable return cannot be added beneath a superficially unchanged raise.
    module_body = list(guard_tree.body)
    if (
        module_body
        and isinstance(module_body[0], ast.Expr)
        and isinstance(module_body[0].value, ast.Constant)
        and isinstance(module_body[0].value.value, str)
    ):
        module_body = module_body[1:]
    exact_future = (
        len(module_body) >= 1
        and isinstance(module_body[0], ast.ImportFrom)
        and module_body[0].module == "__future__"
        and module_body[0].level == 0
        and len(module_body[0].names) == 1
        and module_body[0].names[0].name == "annotations"
        and module_body[0].names[0].asname is None
    )
    exact_constant = (
        len(module_body) >= 2
        and isinstance(module_body[1], ast.Assign)
        and len(module_body[1].targets) == 1
        and isinstance(module_body[1].targets[0], ast.Name)
        and module_body[1].targets[0].id == "HARD_DISABLED"
        and isinstance(module_body[1].value, ast.Constant)
        and module_body[1].value.value == "ROUTE_A_V3_SEALED_HARD_DISABLED_A0_A9"
    )
    exact_class = (
        len(module_body) >= 3
        and isinstance(module_body[2], ast.ClassDef)
        and module_body[2].name == "RouteAV3SealedHardDisabled"
        and len(module_body[2].bases) == 1
        and isinstance(module_body[2].bases[0], ast.Name)
        and module_body[2].bases[0].id == "RuntimeError"
        and not module_body[2].keywords
        and not module_body[2].decorator_list
    )
    exact_function_slot = (
        len(module_body) >= 4
        and isinstance(module_body[3], ast.FunctionDef)
        and module_body[3].name == "assert_sealed_final_authorized"
    )
    if not (
        len(module_body) == 4
        and exact_future
        and exact_constant
        and exact_class
        and exact_function_slot
    ):
        _issue(
            issues,
            "SEALED_GUARD_MODULE_SHAPE",
            SEALED_GUARD_PATH,
            "guard module must contain only future annotations, HARD_DISABLED, the exception class, and the guard function",
        )

    imports = [
        node
        for node in ast.walk(guard_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if len(imports) != 1 or not exact_future or imports[0] is not module_body[0]:
        _issue(
            issues,
            "SEALED_GUARD_IMPORT",
            SEALED_GUARD_PATH,
            "guard may import only __future__.annotations",
        )

    guard_classes = {
        node.name: node
        for node in guard_tree.body
        if isinstance(node, ast.ClassDef)
    }
    exception_class = guard_classes.get("RouteAV3SealedHardDisabled")
    if set(guard_classes) != {"RouteAV3SealedHardDisabled"} or exception_class is None:
        _issue(issues, "SEALED_GUARD_EXCEPTION", SEALED_GUARD_PATH, "exact hard-disable exception class is required")
    else:
        class_body = list(exception_class.body)
        if (
            class_body
            and isinstance(class_body[0], ast.Expr)
            and isinstance(class_body[0].value, ast.Constant)
            and isinstance(class_body[0].value.value, str)
        ):
            class_body = class_body[1:]
        if class_body or not exact_class:
            _issue(
                issues,
                "SEALED_GUARD_EXCEPTION",
                SEALED_GUARD_PATH,
                "hard-disable exception may contain only its docstring and RuntimeError base",
            )

    guard_functions = {
        node.name: node
        for node in guard_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    guard = guard_functions.get("assert_sealed_final_authorized")
    if set(guard_functions) != {"assert_sealed_final_authorized"} or not isinstance(guard, ast.FunctionDef):
        _issue(
            issues,
            "SEALED_GUARD_FUNCTIONS",
            SEALED_GUARD_PATH,
            "A0 guard module must expose exactly one synchronous guard function",
        )
    else:
        signature_ok = (
            not guard.decorator_list
            and not guard.args.posonlyargs
            and [argument.arg for argument in guard.args.args] == ["call_args", "repo_root"]
            and all(
                isinstance(argument.annotation, ast.Name)
                and argument.annotation.id == "object"
                for argument in guard.args.args
            )
            and guard.args.vararg is None
            and not guard.args.kwonlyargs
            and not guard.args.kw_defaults
            and guard.args.kwarg is None
            and len(guard.args.defaults) == 2
            and all(
                isinstance(default, ast.Constant) and default.value is None
                for default in guard.args.defaults
            )
            and isinstance(guard.returns, ast.Constant)
            and guard.returns.value is None
        )
        if not signature_ok:
            _issue(
                issues,
                "SEALED_GUARD_SIGNATURE",
                SEALED_GUARD_PATH,
                "guard signature must be undecorated (call_args: object = None, repo_root: object = None) -> None",
            )
        executable = list(guard.body)
        if (
            executable
            and isinstance(executable[0], ast.Expr)
            and isinstance(executable[0].value, ast.Constant)
            and isinstance(executable[0].value.value, str)
        ):
            executable = executable[1:]
        if len(executable) != 1 or not exact_hard_disable_raise(executable[0]):
            _issue(
                issues,
                "SEALED_GUARD_HARD_DISABLE_BODY",
                SEALED_GUARD_PATH,
                "guard must have exactly one executable statement: raise RouteAV3SealedHardDisabled(HARD_DISABLED)",
            )
        if any(isinstance(node, ast.Return) for node in ast.walk(guard)):
            _issue(
                issues,
                "SEALED_GUARD_REACHABLE_SUCCESS",
                SEALED_GUARD_PATH,
                "A0 guard must contain no return node or reachable success path",
            )
    return issues

def scan_conflict_markers(repo_root: Path) -> list[Issue]:
    issues: list[Issue] = []
    candidates = set(required_bundle_paths())
    for directory in ("scripts/route_a_v3", "tests/route_a_v3"):
        root = repo_root / directory
        if root.is_dir():
            for path in root.glob("*.py"):
                if path.is_file() and not path.is_symlink():
                    candidates.add(path.relative_to(repo_root).as_posix())
    for relative in sorted(candidates):
        try:
            text = _read_text(repo_root, relative)
        except (FileNotFoundError, UnicodeDecodeError, ValueError):
            continue
        for marker in CONFLICT_MARKERS:
            if marker in text:
                _issue(issues, "CONFLICT_MARKER", relative, f"contains {marker!r}")
    return issues


def load_bundle_documents(repo_root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    config = _load_yaml(repo_root, CONFIG_PATH)
    supersession = _load_yaml(repo_root, SUPERSESSION_PATH)
    registries = {name: _load_yaml(repo_root, path) for name, path in REGISTRY_PATHS.items()}
    return config, supersession, registries


def validate_bundle(repo_root: Path) -> list[Issue]:
    """Run all static checks and return deterministic failures without writing."""

    repo_root = repo_root.resolve()
    issues = validate_required_files(repo_root)
    issues.extend(validate_schema_manifest(repo_root))
    issues.extend(validate_registry_manifest(repo_root))
    issues.extend(validate_gse114002_public_authority_gap_audit(repo_root))
    issues.extend(validate_gse149487_plumage_protocol(repo_root))
    issues.extend(validate_gse200302_role_protocol(repo_root))
    issues.extend(validate_dec019_successor_adjudicators(repo_root))
    issues.extend(
        validate_gse200304_dec019_post_adjudication_registration(repo_root)
    )
    issues.extend(validate_python_static_safety(repo_root))
    issues.extend(validate_runner_and_guard_ast(repo_root))
    issues.extend(scan_conflict_markers(repo_root))
    try:
        config, supersession, registries = load_bundle_documents(repo_root)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        _issue(issues, "BUNDLE_DOCUMENT_LOAD", ".", str(exc))
        return sorted(set(issues))
    issues.extend(validate_contract_authority(repo_root, config, supersession, registries))
    issues.extend(validate_dec019_authority(repo_root, config, registries))
    try:
        decision_log = _load_yaml(repo_root, DECISION_LOG_PATH)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        _issue(issues, "DECISION_LOG_LOAD", DECISION_LOG_PATH, str(exc))
    else:
        issues.extend(validate_decision_log(decision_log))
    try:
        a1_interim = _load_yaml(repo_root, A1_INTERIM_PATH)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        _issue(issues, "A1_INTERIM_LOAD", A1_INTERIM_PATH, str(exc))
    else:
        issues.extend(validate_a1_interim_lineage(repo_root, a1_interim))
    issues.extend(validate_registry_closure(config, registries))
    issues.extend(validate_sealed_hard_disable(config, registries))
    issues.extend(validate_l4_and_pre_v3(config, supersession, registries["claim"]))
    return sorted(set(issues))


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root(), help="repository root (default: inferred from this script)")
    parser.add_argument("--write-manifests", action="store_true", help="opt in to deterministic schema manifest rewrite before validation")
    parser.add_argument("--json", action="store_true", dest="json_output", help="emit machine-readable validation result")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.write_manifests:
        write_schema_manifests(repo_root)
    issues = validate_bundle(repo_root)
    payload = {
        "contract_id": CONTRACT_ID,
        "version": VERSION,
        "validator_mode": "STATIC_READ_ONLY" if not args.write_manifests else "SCHEMA_MANIFEST_WRITE_THEN_STATIC_VALIDATE",
        "scientific_claim": "NOT_ASSERTED",
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(f"Route A V3 A0 static validation: {len(issues)} issue(s)")
        for issue in issues:
            print(f"[{issue.code}] {issue.path}: {issue.detail}")
        print("Scientific/data/model/guidance/Route-A PASS: NOT_ASSERTED")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
