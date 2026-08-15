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
DEC020_AMENDMENT_PATH = "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec020.yaml"
DEC021_AMENDMENT_PATH = "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec021.yaml"
DEC022_AMENDMENT_PATH = "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec022.yaml"
DEC023_AMENDMENT_PATH = "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec023.yaml"
DEC024_AMENDMENT_PATH = "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec024.yaml"
DEC027_AMENDMENT_PATH = "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec027.yaml"
DEC028_AMENDMENT_PATH = "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028.yaml"
DEC028_HUMAN_CONTRACT_PATH = "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028_single_study_mainline_contract.md"
DEC028_PROTOCOL_PATH = "configs/route_a_v3_single_study_mainline_v1.yaml"
DEC028_EXECUTION_PATH = "docs/execution/route_a_v3_single_study_mainline_v1.yaml"
DECISION_LOG_PATH = "docs/execution/route_a_v3_decision_log.yaml"
REGISTRY_MANIFEST_PATH = "docs/execution/route_a_v3_registry_manifest.json"
A1_INTERIM_PATH = "docs/execution/route_a_v3_a1_interim.yaml"
A6_INTERIM_PATH = "docs/execution/route_a_v3_a6_interim.yaml"
EXPECTED_A1_INTERIM_SHA256 = "15cda2717c69a13ace3567b24b97a68de9f7fc7c391aacb549380fa7f819c681"
EXPECTED_A6_INTERIM_SHA256 = "560f24b1a6fc3ef29f7507c59d4c2b62760c0405855b2a8671e5f0ce39ffe6b8"
CURRENT_ACTIVE_CONFIG_SHA256 = "6a6c97fe28b07738b42175183be556f36d5477d67c1180a69df75d0850790e41"
CURRENT_TASK_REGISTRY_SHA256 = "113c98a78644f5f5e432f59de7f8bc34f9956b13998e68b785f350cf5289d917"
DEC023_ACTIVE_TASK_REGISTRY_SHA256 = "210964e1a1c0b1166dab73e95e0243eee54d8470149aa3b3cb182f4f90e266b3"
A6_REGISTRATION_LEDGER_AT = "2026-08-13T18:55:00+08:00"
DEC021_AUTHORITY_LEDGER_AT = "2026-08-13T19:50:00+08:00"
DEC021_AUTHORITY_MANIFEST_AT = "2026-08-13T19:50:01+08:00"
DEC021_AUTHORITY_MANIFEST_STATUS = (
    "DEC021_GSE256185_PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY_"
    "REGISTERED_EVT051_SETTLED_A6_IN_PROGRESS_L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
GSE256185_PUBLIC_GEOMETRY_LEDGER_AT = "2026-08-13T21:10:30+08:00"
GSE256185_PUBLIC_GEOMETRY_MANIFEST_AT = "2026-08-13T21:10:31+08:00"
GSE256185_PUBLIC_GEOMETRY_EVIDENCE_UPDATE_ID = (
    "GSE256185_PUBLIC_IDENTIFIER_POOL_GEOMETRY_PREFLIGHT_V1"
)
GSE256185_PUBLIC_GEOMETRY_LINEAGE_ID = (
    "gse256185_public_identifier_pool_geometry_preflight_v1"
)
GSE256185_PUBLIC_GEOMETRY_MANIFEST_STATUS = (
    "DEC021_GSE256185_PUBLIC_IDENTIFIER_POOL_GEOMETRY_PREFLIGHT_REGISTERED_"
    "EVT053_SETTLED_EVIDENCE_RUNTIME_SYNCED_A1_INCOMPLETE_A6_IN_PROGRESS_"
    "L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
GSE256185_PUBLIC_GEOMETRY_RUNTIME_EVENT_ID = "A1-EVT-053"
DEC022_AUTHORITY_LEDGER_AT = "2026-08-13T22:20:00+08:00"
DEC022_AUTHORITY_MANIFEST_AT = "2026-08-13T22:20:01+08:00"
DEC022_AUTHORITY_MANIFEST_STATUS = (
    "DEC022_GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY_"
    "REGISTERED_EVT053_SETTLED_PENDING_FRESH_RUNTIME_EVENT_A1_INCOMPLETE_"
    "A6_IN_PROGRESS_L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
DEC022_PENDING_RUNTIME_EVENT_ID = "PENDING_FRESH_RUNTIME_EVENT_ID"
DEC022_AUTHORITY_RUNTIME_EVENT_ID = "A1-EVT-054"
GSE256185_ROW_PREFLIGHT_LEDGER_AT = "2026-08-13T23:35:00+08:00"
GSE256185_ROW_PREFLIGHT_MANIFEST_AT = "2026-08-13T23:57:32+08:00"
GSE256185_ROW_PREFLIGHT_EVIDENCE_UPDATE_ID = (
    "GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_V1"
)
GSE256185_ROW_PREFLIGHT_LINEAGE_ID = (
    "gse256185_aggregate_row_level_qualification_preflight_v1"
)
GSE256185_ROW_PREFLIGHT_MANIFEST_STATUS = (
    "DEC022_GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_REGISTERED_"
    "EVT055_SETTLED_EVIDENCE_RUNTIME_SYNCED_A1_INCOMPLETE_A6_IN_PROGRESS_"
    "L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
GSE256185_ROW_PREFLIGHT_RUNTIME_EVENT_ID = "A1-EVT-055"
DEC023_AUTHORITY_LEDGER_AT = "2026-08-14T10:15:00+08:00"
DEC023_AUTHORITY_MANIFEST_AT = "2026-08-14T10:15:01+08:00"
DEC023_AUTHORITY_MANIFEST_STATUS = (
    "DEC023_GSE261709_PUBLIC_SCHEMA_GEOMETRY_AND_GSE207584_DENSE_FAMILY_"
    "DUAL_AGGREGATE_ONLY_PREFLIGHT_REGISTERED_EVT055_SETTLED_PENDING_FRESH_"
    "UNALLOCATED_RUNTIME_EVENT_A1_INCOMPLETE_A6_IN_PROGRESS_L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
DEC023_PENDING_RUNTIME_EVENT_ID = "PENDING_FRESH_RUNTIME_EVENT_ID"
DEC023_AUTHORITY_RUNTIME_EVENT_ID = "A1-EVT-056"
DEC023_CURRENT_RUNTIME_EVENT_ID = "A1-EVT-057"
DEC023_DUAL_PREFLIGHT_EVIDENCE_LEDGER_AT = "2026-08-14T12:55:00+08:00"
DEC023_DUAL_PREFLIGHT_EVIDENCE_MANIFEST_AT = "2026-08-14T13:45:38+08:00"
DEC023_DUAL_PREFLIGHT_EVIDENCE_INTEGRATION_ID = (
    "DEC023_GSE261709_AND_GSE207584_DUAL_PREFLIGHT_FINAL_EVIDENCE_V1_"
    "LEDGER_REGISTRATION"
)
DEC023_DUAL_PREFLIGHT_EVIDENCE_UPDATE_ID = (
    "DEC023_GSE261709_AND_GSE207584_DUAL_PREFLIGHT_FINAL_EVIDENCE_V1"
)
DEC023_DUAL_PREFLIGHT_EVIDENCE_MANIFEST_STATUS = (
    "DEC023_GSE261709_PUBLIC_SCHEMA_GEOMETRY_AND_GSE207584_DENSE_FAMILY_"
    "DUAL_PREFLIGHT_EVIDENCE_REGISTERED_EVT057_SETTLED_EVIDENCE_RUNTIME_"
    "SYNCED_A1_INCOMPLETE_A6_IN_PROGRESS_L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
DEC024_AUTHORITY_MANIFEST_AT = "2026-08-14T17:30:00+08:00"
DEC024_AUTHORITY_MANIFEST_STATUS = (
    "DEC024_THREE_REPLACEMENT_AGGREGATE_ONLY_PREFLIGHT_AUTHORITIES_"
    "REGISTERED_EVT057_SETTLED_PENDING_FRESH_UNALLOCATED_RUNTIME_EVENT_"
    "A1_INCOMPLETE_A6_IN_PROGRESS_L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
DEC024_PENDING_RUNTIME_EVENT_ID = "PENDING_FRESH_RUNTIME_EVENT_ID"
DEC024_CURRENT_RUNTIME_EVENT_ID = "A1-EVT-058"
DEC024_CURRENT_MANIFEST_AT = "2026-08-14T18:35:00+08:00"
DEC024_CURRENT_MANIFEST_STATUS = (
    "DEC024_THREE_REPLACEMENT_AGGREGATE_ONLY_PREFLIGHT_AUTHORITIES_"
    "REGISTERED_EVT058_SETTLED_RUNTIME_SYNCED_A1_INCOMPLETE_A6_IN_PROGRESS_"
    "L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
DEC027_AUTHORITY_MANIFEST_AT = "2026-08-15T01:11:09+08:00"
DEC027_AUTHORITY_MANIFEST_STATUS = (
    "DEC027_BOUNDED_SIX_ROUTE_DATA_RESCUE_SPRINT_AUTHORIZED_EVT058_SETTLED_"
    "PENDING_FRESH_UNALLOCATED_RUNTIME_EVENT_A1_INCOMPLETE_A6_IN_PROGRESS_"
    "L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
DEC027_PENDING_RUNTIME_EVENT_ID = "PENDING_FRESH_RUNTIME_EVENT_ID"
DEC027_CURRENT_RUNTIME_EVENT_ID = "A1-EVT-059"
DEC027_PENDING_SUCCESSOR_RUNTIME_EVENT_LABEL = "A1-EVT-060"
DEC027_SIX_RESCUE_EVIDENCE_LEDGER_AT = "2026-08-15T05:05:00+08:00"
DEC027_SIX_RESCUE_EVIDENCE_MANIFEST_AT = "2026-08-15T05:05:01+08:00"
DEC027_SIX_RESCUE_EVIDENCE_INTEGRATION_ID = (
    "DEC027_SIX_TERMINAL_AGGREGATE_RESCUE_REPORTS_V1_LEDGER_REGISTRATION"
)
DEC027_SIX_RESCUE_EVIDENCE_UPDATE_ID = (
    "DEC027_SIX_TERMINAL_AGGREGATE_RESCUE_REPORTS_V1"
)
DEC027_SIX_RESCUE_EVIDENCE_MANIFEST_STATUS = (
    "DEC027_SIX_TERMINAL_AGGREGATE_RESCUE_REPORTS_REGISTERED_EVT059_SETTLED_"
    "PENDING_UNALLOCATED_EVT060_NO_PROMOTION_A1_INCOMPLETE_A6_IN_PROGRESS_"
    "L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
DEC027_EVT060_CURRENT_RUNTIME_EVENT_ID = "A1-EVT-060"
DEC027_EVT060_PROJECTION_LEDGER_AT = "2026-08-15T05:35:00+08:00"
DEC027_EVT060_PROJECTION_MANIFEST_AT = "2026-08-15T05:35:01+08:00"
DEC027_EVT060_PROJECTION_LINEAGE_ID = (
    "dec027_evt060_current_projection_settlement_v1"
)
DEC027_EVT060_PROJECTION_UPDATE_ID = (
    "DEC027_EVT060_CURRENT_PROJECTION_SETTLEMENT_V1"
)
DEC027_EVT060_PROJECTION_MANIFEST_STATUS = (
    "DEC027_EVT060_CURRENT_PROJECTION_SETTLED_RUNTIME_SYNCED_NO_EVT061_"
    "UNALLOCATED_NO_PROMOTION_A1_INCOMPLETE_A6_IN_PROGRESS_"
    "L3_NOT_ESTABLISHED_A7_NOT_RUN"
)
DEC027_EVT060_RUNTIME_CAS_PATHS = {
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/STATUS.json",
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/RUN_MANIFEST.json",
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/EVENT_LOG.jsonl",
}
DEC028_AUTHORITY_LEDGER_AT = "2026-08-15T17:15:00+08:00"
DEC028_AUTHORITY_MANIFEST_AT = "2026-08-15T20:00:17+08:00"
DEC028_AUTHORITY_MANIFEST_STATUS = (
    "DEC028_STANDARD_DEVELOPMENT_CRITIC_COMPLETED_NEGATIVE_RESULT_SS6_"
    "ENGINEERING_PASS_EVT061_SETTLED_CLAIM_NOT_ESTABLISHED_SEALED_UNTOUCHED"
)
DEC027_RESCUE_EXECUTION_ORDER = [
    "GSE217518_CORRECTED_A1_SUCCESSOR",
    "ENCSR854RUF_DATASET_SPECIFIC_A1_PREFLIGHT",
    "GSE232572_CORRECTED_A1_REPLAY",
    "GSE113849_DESIGNED_SNV_TRUE_A2_AGGREGATE_PREFLIGHT",
    "GSE269595_CORRECTED_ROLE_ADJUDICATION_SUCCESSOR",
    "GSE295080_INDEPENDENCE_OVERLAP_ADJUDICATION_ONLY",
]
DEC027_ROUTE_SECTION_AND_GATE_COUNT = {
    "GSE217518_CORRECTED_A1_SUCCESSOR": ("gse217518_corrected_a1_successor", 11),
    "ENCSR854RUF_DATASET_SPECIFIC_A1_PREFLIGHT": ("encsr854ruf_dataset_specific_a1_preflight", 11),
    "GSE232572_CORRECTED_A1_REPLAY": ("gse232572_corrected_a1_replay", 11),
    "GSE113849_DESIGNED_SNV_TRUE_A2_AGGREGATE_PREFLIGHT": ("gse113849_designed_snv_true_a2_preflight", 13),
    "GSE269595_CORRECTED_ROLE_ADJUDICATION_SUCCESSOR": ("gse269595_corrected_role_adjudication_successor", 13),
    "GSE295080_INDEPENDENCE_OVERLAP_ADJUDICATION_ONLY": ("gse295080_independence_overlap_adjudication", 7),
}
DEC027_SIX_RESCUE_LINEAGE_IDS = [
    "gse217518_corrected_a1_successor_aggregate_preflight_v1",
    "encsr854ruf_dec027_dataset_specific_a1_preflight_corrected_b4_v1",
    "gse232572_corrected_a1_replay_aggregate_preflight_v1",
    "gse113849_designed_snv_true_a2_aggregate_preflight_v1",
    "gse269595_corrected_role_adjudication_successor_aggregate_recompute_v1",
    "gse295080_independence_overlap_aggregate_preflight_v1",
]
DEC027_SIX_RESCUE_STATIC_LEAF_SHA256 = {
    "configs/route_a_v3_gse217518_corrected_a1_successor_candidate_v1.json": "c5acc8548ab8542ac029a420f21f1d8524bb0f255c6dd53c2d896c2838ce391f",
    "scripts/route_a_v3/preflight_gse217518_corrected_a1_successor_candidate.py": "9fa4464e1cc42baacdf39b4bae2427e1895269b8d6f4e1a05e1e944b0434f3fa",
    "tests/route_a_v3/test_preflight_gse217518_corrected_a1_successor_candidate.py": "bba0ee97e9ed2500f0155c8d1a776d185661b00d8c7fd48c3c0d718d53ccd097",
    "configs/route_a_v3_encsr854ruf_dec027_dataset_specific_a1_preflight_v1.json": "de9df6055a83f29351c4eba2dd895708de4890181e639e26f2228105f6c2cc07",
    "scripts/route_a_v3/preflight_encsr854ruf_dec027_dataset_specific_a1.py": "d5a1ef3e174f479404c3ca1b2dcac9b81c3848b2a7008c9333ca5f339f2d15c9",
    "tests/route_a_v3/test_preflight_encsr854ruf_dec027_dataset_specific_a1.py": "1e31a4dd3643f1c8a3b56e3e6bd0f99b1f01cac65b4c6c3b7113aad5c26ee5b2",
    "configs/route_a_v3_gse232572_corrected_a1_replay_v1.json": "c21821027d8a6806ee98d07177aacf5b7de3b007f15bd3d27bc6a6410bac3aab",
    "scripts/route_a_v3/replay_gse232572_corrected_a1.py": "132a7cf7ea9008f87e2d77a4ba51b2e0701f6ccec2b6e8782a6f645c26cdb466",
    "tests/route_a_v3/test_replay_gse232572_corrected_a1.py": "ea46cfbbdc3f8f0fd74149b862deabe3ce2c551c94f3a46edf891c1917dfca66",
    "configs/route_a_v3_gse113849_designed_snv_true_a2_preflight_v1.json": "464bf2da3988d3bce0a9edf978ac8fd2d88f07598e17d11cae7898ca3645758d",
    "scripts/route_a_v3/preflight_gse113849_designed_snv_true_a2.py": "44b934b828c8fa78aa37588006e7205d0f6277e463ebfe3b3834bbdfd022e23c",
    "tests/route_a_v3/test_preflight_gse113849_designed_snv_true_a2.py": "98630bc8e48c5a07e5f17cf60addbf4382b7e7c49f66ff81245e76062e8d0001",
    "configs/route_a_v3_gse269595_corrected_role_adjudication_successor_candidate_v1.json": "db58c162c282e90b2d1930b5526cea7d0751a2feb238a7c7b156627bd1bf8a26",
    "scripts/route_a_v3/preflight_gse269595_corrected_role_adjudication_successor_candidate.py": "d72bc64f68c72784f58739b6ac46a018e77a4e6034e55d690bf9b61b543c9c51",
    "tests/route_a_v3/test_preflight_gse269595_corrected_role_adjudication_successor_candidate.py": "69abb99c0c8710794f710bb7898b703e3e601237e949562c7910cbb4d7b800e7",
    "configs/route_a_v3_gse295080_independence_overlap_adjudication_v1.json": "368cf0f2fbfd85090518303b1fedfcc04fb21085f214a2729fa636806274a483",
    "scripts/route_a_v3/preflight_gse295080_independence_overlap_adjudication.py": "1eb410d58facdc22af1cfb86583c864e6a6f790699f6b89c13bc2303dbabb00c",
    "tests/route_a_v3/test_preflight_gse295080_independence_overlap_adjudication.py": "68e3b8c3908aa518eb531a269cebc1d82f9210d9f770deaa6019a3ce44b4a069",
}
DEC027_SIX_RESCUE_REPORT_PATHS = {
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE217518_CORRECTED_A1_SUCCESSOR_AGGREGATE_PREFLIGHT_V1/GSE217518_CORRECTED_A1_SUCCESSOR_AGGREGATE_PREFLIGHT_V1.json",
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/ENCSR854RUF_DEC027_DATASET_SPECIFIC_A1_PREFLIGHT_CORRECTED_B4_V1/ENCSR854RUF_DEC027_DATASET_SPECIFIC_A1_PREFLIGHT_RECORD_V1.json",
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE232572_CORRECTED_A1_REPLAY_B_0f2c008/GSE232572_CORRECTED_A1_REPLAY_AGGREGATE_PREFLIGHT.json",
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE113849_DESIGNED_SNV_TRUE_A2_B_6372ddc/GSE113849_DESIGNED_SNV_TRUE_A2_AGGREGATE_PREFLIGHT.json",
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE269595_CORRECTED_ROLE_ADJUDICATION_B2_19ca492/GSE269595_CORRECTED_ROLE_ADJUDICATION_SUCCESSOR_AGGREGATE_RECOMPUTE_V1.json",
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE295080_INDEPENDENCE_OVERLAP_B2_679a1c2/GSE295080_INDEPENDENCE_OVERLAP_AGGREGATE_PREFLIGHT_V1.json",
}
DEC027_SIX_RESCUE_REPORT_SPECS = {
    DEC027_SIX_RESCUE_LINEAGE_IDS[0]: {
        "route_id": "GSE217518_CORRECTED_A1_SUCCESSOR",
        "dataset_id": "GSE217518",
        "registry_status": "EXISTING_REGISTERED_STUDY_UNIT",
        "candidate_role": "CORRECTED_A1_SUCCESSOR_PREFLIGHT_ONLY",
        "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE217518_CORRECTED_A1_SUCCESSOR_AGGREGATE_PREFLIGHT_V1/GSE217518_CORRECTED_A1_SUCCESSOR_AGGREGATE_PREFLIGHT_V1.json",
        "bytes": 7833,
        "sha256": "03de0d423604518653a5188696d8186c82fa66e7858e7f498052fc67256e8884",
        "report_observed_at": "UNKNOWN_NOT_ASSERTED",
        "schema_version": "route_a_v3_gse217518_corrected_a1_successor_aggregate_preflight.v1",
        "protocol_id": "GSE217518_CORRECTED_A1_SUCCESSOR_CANDIDATE_V1",
        "status": "STOP_CORRECTED_PREFLIGHT_GATES_NOT_CLOSED",
        "gate_status_counts": {"PASS": 4, "BLOCKED": 4, "NOT_RUN": 3},
        "implementation_commit": "36b535f77b3f27bb872b182dcaf6c646d9781991",
        "binding_commit": "0a46400efee4ead95b1283df73d263f6f8033036",
        "static_paths": [
            "configs/route_a_v3_gse217518_corrected_a1_successor_candidate_v1.json",
            "scripts/route_a_v3/preflight_gse217518_corrected_a1_successor_candidate.py",
            "tests/route_a_v3/test_preflight_gse217518_corrected_a1_successor_candidate.py",
        ],
    },
    DEC027_SIX_RESCUE_LINEAGE_IDS[1]: {
        "route_id": "ENCSR854RUF_DATASET_SPECIFIC_A1_PREFLIGHT",
        "dataset_id": "ENCSR854RUF",
        "registry_status": "EXISTING_REGISTERED_STUDY_UNIT",
        "candidate_role": "DATASET_SPECIFIC_A1_PREFLIGHT_ONLY",
        "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/ENCSR854RUF_DEC027_DATASET_SPECIFIC_A1_PREFLIGHT_CORRECTED_B4_V1/ENCSR854RUF_DEC027_DATASET_SPECIFIC_A1_PREFLIGHT_RECORD_V1.json",
        "bytes": 11423,
        "sha256": "3753d6fc5fb4132e43e11f29f9c79a04078a592aaba760c1f8a6e6ed2c5fc6c2",
        "report_observed_at": "2026-08-15T01:40:00+08:00",
        "schema_version": "route_a_v3_encsr854ruf_dec027_dataset_specific_a1_preflight_record.v1",
        "protocol_id": "ENCSR854RUF_DEC027_DATASET_SPECIFIC_A1_PREFLIGHT_V1",
        "status": "TERMINAL_AGGREGATE_PREFLIGHT_STOP_NOT_QUALIFIED",
        "gate_status_counts": {"PASS": 3, "PARTIAL_OR_CONDITIONAL": 3, "FAIL": 1, "UNKNOWN_NOT_ASSERTED": 4},
        "implementation_commit": "53f426aef8b12e8dcbfaaf978fcfa7d1c7a911d2",
        "binding_commit": "56b39f966a272d8ea8022048855d2fcca0ee155a",
        "static_paths": [
            "configs/route_a_v3_encsr854ruf_dec027_dataset_specific_a1_preflight_v1.json",
            "scripts/route_a_v3/preflight_encsr854ruf_dec027_dataset_specific_a1.py",
            "tests/route_a_v3/test_preflight_encsr854ruf_dec027_dataset_specific_a1.py",
        ],
    },
    DEC027_SIX_RESCUE_LINEAGE_IDS[2]: {
        "route_id": "GSE232572_CORRECTED_A1_REPLAY",
        "dataset_id": "GSE232572",
        "registry_status": "EXISTING_REGISTERED_STUDY_UNIT",
        "candidate_role": "CORRECTED_A1_REPLAY_PREFLIGHT_ONLY",
        "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE232572_CORRECTED_A1_REPLAY_B_0f2c008/GSE232572_CORRECTED_A1_REPLAY_AGGREGATE_PREFLIGHT.json",
        "bytes": 9823,
        "sha256": "20451d763b8b2bc2658a4bf6163bbef8a2449759fa7fbe1ff5a29f9146cdae2c",
        "report_observed_at": "2026-08-15T04:05:39+08:00",
        "schema_version": "route_a_v3_gse232572_corrected_a1_replay.v1",
        "protocol_id": "GSE232572_CORRECTED_A1_REPLAY_V1",
        "status": "STOP_REMAINING_QUALIFICATION_GATES_NOT_CLOSED",
        "gate_status_counts": {"PASS": 7, "UNKNOWN_NOT_ASSERTED": 3, "NOT_RUN": 1},
        "implementation_commit": "d3dcae4c6ef53c52e942bb511946b52b952d3c7f",
        "binding_commit": "0f2c00868b6581edd9a429c7a8a67bb43f6b7776",
        "static_paths": [
            "configs/route_a_v3_gse232572_corrected_a1_replay_v1.json",
            "scripts/route_a_v3/replay_gse232572_corrected_a1.py",
            "tests/route_a_v3/test_replay_gse232572_corrected_a1.py",
        ],
    },
    DEC027_SIX_RESCUE_LINEAGE_IDS[3]: {
        "route_id": "GSE113849_DESIGNED_SNV_TRUE_A2_AGGREGATE_PREFLIGHT",
        "dataset_id": "GSE113849",
        "registry_status": "EXTERNAL_PREFLIGHT_CANDIDATE_ONLY_NOT_ACTIVE_STUDY_UNIT",
        "candidate_role": "DESIGNED_SNV_TRUE_A2_PREFLIGHT_ONLY",
        "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE113849_DESIGNED_SNV_TRUE_A2_B_6372ddc/GSE113849_DESIGNED_SNV_TRUE_A2_AGGREGATE_PREFLIGHT.json",
        "bytes": 16280,
        "sha256": "7ac51be90de8bbed2562e081a4063b6ed479f8a700d417623bfbefb269384839",
        "report_observed_at": "2026-08-15T04:14:00+08:00",
        "schema_version": "route_a_v3_gse113849_designed_snv_true_a2_preflight.v1",
        "protocol_id": "GSE113849_DESIGNED_SNV_TRUE_A2_AGGREGATE_PREFLIGHT_V1",
        "status": "STOP_REMAINING_QUALIFICATION_GATES_NOT_CLOSED",
        "gate_status_counts": {"PASS": 5, "PARTIAL_OR_CONDITIONAL": 2, "FAIL": 1, "UNKNOWN_NOT_ASSERTED": 4, "NOT_RUN": 1},
        "implementation_commit": "8dfca85f3311ede01f594662d13b126bc8e2fef2",
        "binding_commit": "6372ddcb4b006d587a40ce628f9e193324c28b17",
        "static_paths": [
            "configs/route_a_v3_gse113849_designed_snv_true_a2_preflight_v1.json",
            "scripts/route_a_v3/preflight_gse113849_designed_snv_true_a2.py",
            "tests/route_a_v3/test_preflight_gse113849_designed_snv_true_a2.py",
        ],
    },
    DEC027_SIX_RESCUE_LINEAGE_IDS[4]: {
        "route_id": "GSE269595_CORRECTED_ROLE_ADJUDICATION_SUCCESSOR",
        "dataset_id": "GSE269595",
        "registry_status": "EXISTING_REGISTERED_STUDY_UNIT",
        "candidate_role": "MUTUALLY_EXCLUSIVE_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_ONLY",
        "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE269595_CORRECTED_ROLE_ADJUDICATION_B2_19ca492/GSE269595_CORRECTED_ROLE_ADJUDICATION_SUCCESSOR_AGGREGATE_RECOMPUTE_V1.json",
        "bytes": 13272,
        "sha256": "7952a74690817f24c3dc1df1ccdb104b9997464ceb97f0320fddd227bd84ac4b",
        "report_observed_at": "2026-08-15T04:31:00+08:00",
        "schema_version": "route_a_v3_gse269595_corrected_role_adjudication_aggregate_recompute.v1",
        "protocol_id": "GSE269595_CORRECTED_ROLE_ADJUDICATION_SUCCESSOR_CANDIDATE_V1",
        "status": "STOP_CORRECTED_ROLE_ADJUDICATION_GATES_NOT_CLOSED",
        "gate_status_counts": {"PASS": 8, "BLOCKED": 3, "FAIL": 2},
        "implementation_commit": "95f32836f62db26f0302edbbb6443ae0a33918b3",
        "binding_commit": "19ca49229c9ff2814bad2c58b8b84be14624b7ea",
        "static_paths": [
            "configs/route_a_v3_gse269595_corrected_role_adjudication_successor_candidate_v1.json",
            "scripts/route_a_v3/preflight_gse269595_corrected_role_adjudication_successor_candidate.py",
            "tests/route_a_v3/test_preflight_gse269595_corrected_role_adjudication_successor_candidate.py",
        ],
    },
    DEC027_SIX_RESCUE_LINEAGE_IDS[5]: {
        "route_id": "GSE295080_INDEPENDENCE_OVERLAP_ADJUDICATION_ONLY",
        "dataset_id": "GSE295080",
        "registry_status": "EXTERNAL_PREFLIGHT_CANDIDATE_ONLY_NOT_ACTIVE_STUDY_UNIT",
        "candidate_role": "INDEPENDENCE_OVERLAP_ADJUDICATION_ONLY",
        "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE295080_INDEPENDENCE_OVERLAP_B2_679a1c2/GSE295080_INDEPENDENCE_OVERLAP_AGGREGATE_PREFLIGHT_V1.json",
        "bytes": 8989,
        "sha256": "f3f258cd89f58d42270e05b40a67c55fcd18511e6b17d9dc6711f62d0db5aa63",
        "report_observed_at": "2026-08-15T04:55:49+08:00",
        "schema_version": "route_a_v3_gse295080_independence_overlap_aggregate_record.v1",
        "protocol_id": "GSE295080_DEC027_INDEPENDENCE_OVERLAP_ADJUDICATION_ONLY_V1",
        "status": "STOP_NO_INDEPENDENT_CREDIT_AND_NO_ROW_LEVEL_AUTHORITY_REQUEST",
        "gate_status_counts": {"PASS": 3, "PARTIAL_OR_CONDITIONAL": 1, "FAIL": 1, "UNKNOWN_NOT_ASSERTED": 1, "BLOCKED_OR_STOP": 1},
        "implementation_commit": "845550e24c836872f4572abe10275db58f62554e",
        "binding_commit": "679a1c2ae89db7d6a9894f9299de7ce38b30ecdb",
        "static_paths": [
            "configs/route_a_v3_gse295080_independence_overlap_adjudication_v1.json",
            "scripts/route_a_v3/preflight_gse295080_independence_overlap_adjudication.py",
            "tests/route_a_v3/test_preflight_gse295080_independence_overlap_adjudication.py",
        ],
    },
}
DEC027_SIX_RESCUE_GATE_RESULTS = {
    "gse217518_corrected_a1_successor_aggregate_preflight_v1": {
        "PUBLIC_SOURCE_ASSET_IDENTITY_AND_PRIMARY_ROUTE_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "SOURCE_REFERENCE_TO_CANDIDATE_CROSSWALK_CLOSED": {"raw_status": "BLOCKED", "normalized_status": "BLOCKED"},
        "FULL_115BP_CONSTRUCT_REPORTER_AND_REGION_CONTEXT_CLOSED": {"raw_status": "BLOCKED", "normalized_status": "BLOCKED"},
        "ENDPOINT_DIRECTION_SCALE_TRANSFORM_AND_SEMANTICS_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "THREE_INDEPENDENT_BIOLOGICAL_EXPERIMENTS_AND_VALID_STANDARD_ERROR_CLOSED": {"raw_status": "BLOCKED", "normalized_status": "BLOCKED"},
        "MISSING_OUTLIER_QC_AND_SELECTION_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "LICENSE_AND_REUSE_RIGHTS_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "HISTORICAL_ANALYTIC_OR_CHECKPOINT_EXPOSURE_CLOSED": {"raw_status": "BLOCKED", "normalized_status": "BLOCKED"},
        "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED": {"raw_status": "NOT_RUN", "normalized_status": "NOT_RUN"},
        "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED": {"raw_status": "NOT_RUN", "normalized_status": "NOT_RUN"},
        "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED": {"raw_status": "NOT_RUN", "normalized_status": "NOT_RUN"},
    },
    "encsr854ruf_dec027_dataset_specific_a1_preflight_corrected_b4_v1": {
        "PUBLIC_SOURCE_ASSET_IDENTITY_AND_PRIMARY_ROUTE_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "SOURCE_REFERENCE_TO_CANDIDATE_CROSSWALK_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "FULL_REPORTER_AND_THREE_UTR_CONTEXT_CLOSED": {"raw_status": "PASS_FOR_COMPLETE_133BP_VARIABLE_INSERT_AND_FIXED_REPORTER_IDENTITY_ONLY", "normalized_status": "PARTIAL_OR_CONDITIONAL"},
        "ENDPOINT_DIRECTION_SCALE_TRANSFORM_AND_SEMANTICS_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED": {"raw_status": "PARTIAL_REPLICATE_INDEPENDENCE_CLOSED_REPORTED_LFCSE_PRESENT_EXACT_RECOMPUTATION_ENVIRONMENT_NOT_FROZEN", "normalized_status": "PARTIAL_OR_CONDITIONAL"},
        "MISSING_QC_AND_SELECTION_CLOSED": {"raw_status": "PARTIAL_DETERMINISTIC_FINITE_SUBSET_IDENTIFIED_REJECT_REASON_CLOSURE_NOT_BOUND", "normalized_status": "PARTIAL_OR_CONDITIONAL"},
        "LICENSE_AND_REUSE_RIGHTS_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "HISTORICAL_ANALYTIC_OR_CHECKPOINT_EXPOSURE_CLOSED": {"raw_status": "FAIL_KNOWN_HISTORICAL_MODEL_INPUT_WITHOUT_FULL_PRIOR_USE_ATTESTATION", "normalized_status": "FAIL"},
        "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
    },
    "gse232572_corrected_a1_replay_aggregate_preflight_v1": {
        "PUBLIC_SOURCE_ASSET_IDENTITY_AND_PRIMARY_ROUTE_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "REFERENCE_ALTERNATIVE_SOURCE_CANDIDATE_CROSSWALK_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "FULL_REPORTER_AND_THREE_UTR_CONTEXT_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "ENDPOINT_DIRECTION_SCALE_TRANSFORM_AND_SEMANTICS_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "THREE_INDEPENDENT_BIOLOGICAL_REPLICATES_AND_VALID_STANDARD_ERROR_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "MISSING_QC_AND_SELECTION_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "LICENSE_AND_REUSE_RIGHTS_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "HISTORICAL_ANALYTIC_OR_CHECKPOINT_EXPOSURE_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED": {"raw_status": "NOT_RUN", "normalized_status": "NOT_RUN"},
    },
    "gse113849_designed_snv_true_a2_aggregate_preflight_v1": {
        "PUBLIC_ASSET_LINEAGE_AND_INTENDED_UNIVERSE_CLOSED": {"raw_status": "PARTIAL_OR_CONDITIONAL", "normalized_status": "PARTIAL_OR_CONDITIONAL"},
        "SOURCE_TO_CANDIDATE_IDENTITY_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "LEGAL_SINGLE_SUBSTITUTION_EDIT_REPLAY_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "OUTCOME_BLIND_REPORTER_CONTEXT_RULE_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "DENSE_SOURCE_FAMILY_MINIMUM_THREE_CANDIDATES_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "ENDPOINT_DIRECTION_SCALE_TRANSFORM_AND_SEMANTICS_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED": {"raw_status": "FAIL", "normalized_status": "FAIL"},
        "MISSING_QC_AND_SELECTION_CLOSED": {"raw_status": "PARTIAL_OR_CONDITIONAL", "normalized_status": "PARTIAL_OR_CONDITIONAL"},
        "LICENSE_AND_REUSE_RIGHTS_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "HISTORICAL_ANALYTIC_OR_CHECKPOINT_EXPOSURE_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED": {"raw_status": "NOT_RUN", "normalized_status": "NOT_RUN"},
    },
    "gse269595_corrected_role_adjudication_successor_aggregate_recompute_v1": {
        "A1_VERSUS_TRUE_A2_ROLE_ELIGIBILITY_AND_MUTUAL_EXCLUSIVITY_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "ORDINARY_PUBLIC_ASSET_IDENTITY_ROLE_AND_PROVENANCE_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "ELIGIBLE_DENSE_SOURCE_FAMILY_DISTRIBUTION_AND_UNIQUE_SOURCE_ANCHOR_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "INTRONIC_APA_EXCLUSION_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "SOURCE_TO_CANDIDATE_LEGAL_SUBSTITUTION_REPLAY_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "ASSAY_CONTEXT_GUIDE_ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED": {"raw_status": "BLOCKED", "normalized_status": "BLOCKED"},
        "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED": {"raw_status": "BLOCKED", "normalized_status": "BLOCKED"},
        "ASSET_SCHEMA_DIMENSION_AND_COVERAGE_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "MISSING_CENSORING_AND_SELECTION_CLOSED": {"raw_status": "FAIL", "normalized_status": "FAIL"},
        "APARENT_PRIOR_EXPOSURE_AND_MODEL_INPUT_ROUTE_CLOSED": {"raw_status": "FAIL", "normalized_status": "FAIL"},
        "LICENSE_AND_REUSE_RIGHTS_CLOSED": {"raw_status": "BLOCKED", "normalized_status": "BLOCKED"},
        "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "POST_DEDUP_SOURCE_GROUP_EFFECTIVE_N_AND_PREFROZEN_POWER_FULL_CI_WIDTH_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
    },
    "gse295080_independence_overlap_aggregate_preflight_v1": {
        "PUBLIC_IDENTIFIER_ASSET_ROLE_AND_PROVENANCE_CLOSED": {"raw_status": "PASS", "normalized_status": "PASS"},
        "SOURCE_REFERENCE_ALTERNATIVE_MAPPING_SCHEMA_GEOMETRY_CLOSED": {"raw_status": "PASS_AGGREGATE_SCHEMA_GEOMETRY_ONLY_NOT_ROW_REPLAY", "normalized_status": "PASS"},
        "BIOLOGICAL_REPLICATE_LABEL_GEOMETRY_CLOSED": {"raw_status": "PASS_LABEL_GEOMETRY_ONLY_NOT_INDEPENDENCE_OR_STANDARD_ERROR", "normalized_status": "PASS"},
        "CROSS_DATASET_SOURCE_FAMILY_LIBRARY_OVERLAP_CLOSED": {"raw_status": "PARTIAL_GSE186455_EXACT_NAME_OVERLAP_CLOSED_THREE_REQUIRED_COMPARISONS_UNKNOWN", "normalized_status": "PARTIAL_OR_CONDITIONAL"},
        "INDEPENDENT_STUDY_OR_REUSED_LIBRARY_BOUNDARY_CLOSED": {"raw_status": "FAIL_FOR_INDEPENDENT_CREDIT_LIBRARY1_REUSED_LIBRARY_AND_LIBRARY2_BOUNDARY_NOT_CLOSED", "normalized_status": "FAIL"},
        "LICENSE_AND_REUSE_RIGHTS_CLOSED": {"raw_status": "UNKNOWN_NOT_ASSERTED", "normalized_status": "UNKNOWN_NOT_ASSERTED"},
        "OUTCOME_BLIND_NEXT_AUTHORITY_DISPOSITION_CLOSED": {"raw_status": "BLOCKED_OR_STOP", "normalized_status": "BLOCKED_OR_STOP"},
    },
}
GSE261709_PREFLIGHT_LINEAGE_ID = (
    "gse261709_public_identifier_asset_schema_aggregate_geometry_preflight_v1"
)
GSE261709_PREFLIGHT_CONFIG_PATH = (
    "configs/route_a_v3_gse261709_public_identifier_asset_schema_"
    "aggregate_geometry_preflight_v1.json"
)
GSE261709_PREFLIGHT_SCRIPT_PATH = (
    "scripts/route_a_v3/preflight_gse261709_public_identifier_asset_schema_"
    "aggregate_geometry.py"
)
GSE261709_PREFLIGHT_TEST_PATH = (
    "tests/route_a_v3/test_preflight_gse261709_public_identifier_asset_schema_"
    "aggregate_geometry.py"
)
GSE261709_PREFLIGHT_STATIC_LEAF_SHA256 = {
    GSE261709_PREFLIGHT_CONFIG_PATH: (
        "c1b72a24b12a54067f43be347f58060e7b5d382030afcd8d7eb178939e24c835"
    ),
    GSE261709_PREFLIGHT_SCRIPT_PATH: (
        "a6f3981a58cb4f22e0fb1d495070804e4d6d3fba3bbcb090730f19344a7a9089"
    ),
    GSE261709_PREFLIGHT_TEST_PATH: (
        "ada9d6220b49a9076e58d983eb8f41457d1010030c330dedcd8a164e2fb593dd"
    ),
}
GSE261709_PREFLIGHT_REPORT_PATH = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
    "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_GEOMETRY_PREFLIGHT_B3_d3177b0/"
    "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_PREFLIGHT_V1.json"
)
GSE261709_PREFLIGHT_REPORT_BYTES = 6748
GSE261709_PREFLIGHT_REPORT_SHA256 = (
    "ca68fb3a0e18e9c4989c3449f9b88c5112cc737e54ec026ce7c3b0df83386400"
)
GSE261709_PREFLIGHT_RECORDED_AT = "2026-08-14T04:42:22.199739Z"
GSE261709_PREFLIGHT_IMPLEMENTATION_COMMIT = (
    "ec9d6f393ff9bbe87b4b4591fc46d0b3c8cb9254"
)
GSE261709_PREFLIGHT_BINDING_COMMIT = (
    "d3177b0cd600ab4d8e5b64dbbe1e0aaeeb940153"
)
GSE261709_PREFLIGHT_GATE_RESULTS = {
    "OFFICIAL_IDENTIFIER_AND_CONTEXT": {
        "status": "PASS",
        "reason": "OFFICIAL_IDENTITIES_AND_CONTEXT_VISIBLE",
    },
    "ASSET_SAMPLE_AND_RUN_ROLE_AGGREGATE_GEOMETRY": {
        "status": "BLOCKED",
        "reason": "RUN_ROLE_OR_ARCHIVE_LISTING_GEOMETRY_NOT_CLOSED",
    },
    "HEADER_DIMENSION_AND_ASSET_LICENSE_NOTICE": {
        "status": "BLOCKED",
        "reason": "HEADER_DIMENSIONS_OR_ASSET_LEVEL_LICENSE_NOTICE_NOT_CLOSED",
    },
}
GSE207584_PREFLIGHT_LINEAGE_ID = (
    "gse207584_aggregate_dense_family_qualification_preflight_v1"
)
GSE207584_PREFLIGHT_CONFIG_PATH = (
    "configs/route_a_v3_gse207584_aggregate_dense_family_"
    "qualification_preflight_v1.json"
)
GSE207584_PREFLIGHT_SCRIPT_PATH = (
    "scripts/route_a_v3/preflight_gse207584_aggregate_dense_family_qualification.py"
)
GSE207584_PREFLIGHT_TEST_PATH = (
    "tests/route_a_v3/test_preflight_gse207584_aggregate_dense_family_qualification.py"
)
GSE207584_PREFLIGHT_STATIC_LEAF_SHA256 = {
    GSE207584_PREFLIGHT_CONFIG_PATH: (
        "65bae0b40a599f150d4a757887fb1c86239205ee56800f89d0bb9e10227cea43"
    ),
    GSE207584_PREFLIGHT_SCRIPT_PATH: (
        "4776085ad8bd459da92fc63d1d3c437b4e534f81d1e5ff55c167d17880f0c074"
    ),
    GSE207584_PREFLIGHT_TEST_PATH: (
        "81eccd24f16930df334e17b2cc8ae89906a91fba936ee1685d281079ba0b4df6"
    ),
}
GSE207584_PREFLIGHT_REPORT_PATH = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
    "GSE207584_AGGREGATE_DENSE_FAMILY_PREFLIGHT_B4_021ba2a/"
    "GSE207584_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_V1.json"
)
GSE207584_PREFLIGHT_REPORT_BYTES = 7755
GSE207584_PREFLIGHT_REPORT_SHA256 = (
    "a50329b862f41415b3ff33e8fc251d07457449e0d75f31501238eaf30feba6b1"
)
GSE207584_PREFLIGHT_RECORDED_AT = "2026-08-14T04:33:30.500454+00:00"
GSE207584_PREFLIGHT_IMPLEMENTATION_COMMIT = (
    "a136b9a7e5b218b24325bb45d112c348abd7adc5"
)
GSE207584_PREFLIGHT_BINDING_COMMIT = (
    "021ba2af69309b0b2b3b0dd13beffb5c31e6487b"
)
GSE207584_PREFLIGHT_GATE_RESULTS = {
    "INTENDED_UNIVERSE_MEMBERSHIP_CLOSED": {
        "status": "PASS_PREFLIGHT_ONLY",
        "reason": "INTENDED_DENOMINATOR_RETAINED_AND_OBSERVED_SUBSET_JOINED",
    },
    "SOURCE_TO_CANDIDATE_SYNONYMOUS_EDIT_REPLAY_CLOSED": {
        "status": "UNKNOWN_NOT_ASSERTED",
        "reason": "AUTHORITATIVE_SOURCE_MAPPING_AUTHORITY_NOT_BOUND",
    },
    "DENSE_FAMILY_AND_CONTEXT_CLOSED": {
        "status": "UNKNOWN_NOT_ASSERTED",
        "reason": "AUTHORITATIVE_SOURCE_MAPPING_AUTHORITY_NOT_BOUND",
    },
    "ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED": {
        "status": "FAIL_CLOSED",
        "reason": "DUPLICATE_MEASUREMENT_TUPLE_SEMANTICS_UNRESOLVED",
    },
    "THREE_BIOLOGICAL_REPLICATE_SLOPE_AND_VALID_STANDARD_ERROR_CLOSED": {
        "status": "FAIL_CLOSED",
        "reason": "DUPLICATE_MEASUREMENT_TUPLE_SEMANTICS_UNRESOLVED",
    },
    "MISSING_CENSORING_AND_COVERAGE_SELECTION_CLOSED": {
        "status": "UNKNOWN_NOT_ASSERTED",
        "reason": "DUPLICATE_MEASUREMENT_TUPLE_SEMANTICS_UNRESOLVED",
    },
    "LICENSE_AND_REUSE_RIGHTS_CLOSED": {
        "status": "UNKNOWN_NOT_ASSERTED",
        "reason": "ASSET_SPECIFIC_RIGHTS_NOT_BOUND",
    },
    "MODEL_INPUT_ROUTE_AND_SCRATCH_EXPOSURE_CLOSED": {
        "status": "UNKNOWN_NOT_ASSERTED",
        "reason": "DATASET_SPECIFIC_SCRATCH_EXPOSURE_POLICY_NOT_BOUND",
    },
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_CLOSED": {
        "status": "UNKNOWN_NOT_ASSERTED",
        "reason": "SOURCE_FAMILY_COMPONENTS_NOT_IDENTIFIABLE",
    },
    "POST_DEDUP_INDEPENDENT_EFFECTIVE_N_CLOSED": {
        "status": "UNKNOWN_NOT_ASSERTED",
        "reason": "POST_DEDUP_EFFECTIVE_N_NOT_IDENTIFIABLE",
    },
    "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED": {
        "status": "UNKNOWN_NOT_ASSERTED",
        "reason": "PREFROZEN_POWER_N_NOT_IDENTIFIABLE",
    },
}
DEC022_REQUIRED_GATE_IDS = [
    "STRICT_SINGLE_PARENT_CANDIDATE_UNIVERSE_CLOSED",
    "ROW_LEVEL_MULTI_ASSET_LINEAGE_AND_PRIMARY_MEASUREMENT_ROUTE_CLOSED",
    "SOURCE_AND_CANDIDATE_IDENTITY_CLOSED",
    "PARENT_TO_CANDIDATE_EDIT_REPLAY_CLOSED",
    "FAMILY_AND_CONTEXT_STRATIFICATION_CLOSED",
    "ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED",
    "ORIGINAL_UNIT_AND_PAPER_FAITHFUL_TRANSFORM_CLOSED",
    "BIOLOGICAL_SOURCE_GROUP_AUTHORITY_CLOSED",
    "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED",
    "LICENSE_AND_REDISTRIBUTION_RIGHTS_CLOSED",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_CLOSED",
    "PREFROZEN_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED",
    "PUBLIC_PROVENANCE_AND_PRIMARY_MEASUREMENT_ROUTE_CLOSED",
    "MODEL_INPUT_ROUTE_AND_ROUTE_CONDITIONAL_EXPOSURE_CLOSED",
    "BENEFICIAL_SIGNAL_VERSUS_MEASUREMENT_NOISE_CLOSED",
    "POST_DEDUP_INDEPENDENT_EFFECTIVE_N_CLOSED",
    "REJECT_REASON_AND_EXCLUSION_CLOSURE_CLOSED",
]
DEC023_GSE207584_REQUIRED_GATE_IDS = [
    "INTENDED_UNIVERSE_MEMBERSHIP_CLOSED",
    "SOURCE_TO_CANDIDATE_SYNONYMOUS_EDIT_REPLAY_CLOSED",
    "DENSE_FAMILY_AND_CONTEXT_CLOSED",
    "ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED",
    "THREE_BIOLOGICAL_REPLICATE_SLOPE_AND_VALID_STANDARD_ERROR_CLOSED",
    "MISSING_CENSORING_AND_COVERAGE_SELECTION_CLOSED",
    "LICENSE_AND_REUSE_RIGHTS_CLOSED",
    "MODEL_INPUT_ROUTE_AND_SCRATCH_EXPOSURE_CLOSED",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_CLOSED",
    "POST_DEDUP_INDEPENDENT_EFFECTIVE_N_CLOSED",
    "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED",
]
DEC024_GSE261709_REQUIRED_GATE_IDS = [
    "PUBLIC_PROCESSED_ASSET_IDENTITY_ROLE_PROVENANCE_AND_PRIMARY_MEASUREMENT_ROUTE_CLOSED",
    "BARCODE_ALLELE_TRANSCRIPT_SOURCE_AND_FULL_CONSTRUCT_JOIN_CLOSED",
    "SOURCE_CANDIDATE_IDENTITY_AND_DENSE_FAMILY_MINIMUM_THREE_CANDIDATES_CLOSED",
    "SOURCE_TO_CANDIDATE_LEGAL_EDIT_REPLAY_CLOSED",
    "ENDPOINT_DIRECTION_SCALE_EFFECT_AND_STANDARD_ERROR_SEMANTICS_CLOSED",
    "THREE_INDEPENDENT_BIOLOGICAL_REPLICATE_RNA_DNA_COUNTS_AND_VALID_STANDARD_ERROR_CLOSED",
    "MISSING_CENSORING_QC_AND_SELECTION_CLOSED",
    "LICENSE_AND_REUSE_RIGHTS_CLOSED",
    "HISTORICAL_ANALYTIC_OR_CHECKPOINT_EXPOSURE_CLOSED",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED",
    "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED",
    "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED",
]
DEC024_GSE269595_REQUIRED_GATE_IDS = [
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
]
DEC024_EMTAB10902_REQUIRED_GATE_IDS = [
    "INTENDED_UNIVERSE_MEMBERSHIP_CLOSED",
    "SOURCE_ANCHOR_IDENTITY_AND_FULL_REPORTER_CONTEXT_CLOSED",
    "SOURCE_TO_CANDIDATE_EDIT_REPLAY_CLOSED",
    "DENSE_SOURCE_FAMILY_MINIMUM_THREE_CANDIDATES_CLOSED",
    "ENDPOINT_DIRECTION_SCALE_UNIT_AND_SEMANTICS_CLOSED",
    "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED",
    "MISSING_CENSORING_QC_AND_SELECTION_CLOSED",
    "LICENSE_REUSE_RIGHTS_AND_EXPOSURE_CLOSED",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED",
    "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED",
    "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED",
]
GSE256185_ROW_PREFLIGHT_GATE_STATUSES = {
    "STRICT_SINGLE_PARENT_CANDIDATE_UNIVERSE_CLOSED": "PASS",
    "ROW_LEVEL_MULTI_ASSET_LINEAGE_AND_PRIMARY_MEASUREMENT_ROUTE_CLOSED": "PASS",
    "SOURCE_AND_CANDIDATE_IDENTITY_CLOSED": "PASS",
    "PARENT_TO_CANDIDATE_EDIT_REPLAY_CLOSED": "PARTIAL_FAIL_CURRENT_PROTOCOL",
    "FAMILY_AND_CONTEXT_STRATIFICATION_CLOSED": "PASS_LIMITED_VCE_CONTEXT_ONLY",
    "ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED": "PARTIAL_FAIL_CURRENT_PROTOCOL",
    "ORIGINAL_UNIT_AND_PAPER_FAITHFUL_TRANSFORM_CLOSED": "PASS_FOR_FINITE_ROWS_ONLY",
    "BIOLOGICAL_SOURCE_GROUP_AUTHORITY_CLOSED": "UNKNOWN_NOT_ASSERTED",
    "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED": "FAIL",
    "LICENSE_AND_REDISTRIBUTION_RIGHTS_CLOSED": "FAIL",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_CLOSED": "NOT_RUN_FORMAL",
    "PREFROZEN_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED": "INELIGIBLE_NOT_RUN",
    "PUBLIC_PROVENANCE_AND_PRIMARY_MEASUREMENT_ROUTE_CLOSED": "PASS_PUBLIC_ORIGIN_ONLY",
    "MODEL_INPUT_ROUTE_AND_ROUTE_CONDITIONAL_EXPOSURE_CLOSED": "CONDITIONAL_PENDING_ZERO_EXTERNAL_LEARNED_INPUT_RUNTIME_ATTESTATION",
    "BENEFICIAL_SIGNAL_VERSUS_MEASUREMENT_NOISE_CLOSED": "UNKNOWN_NOT_ASSERTED",
    "POST_DEDUP_INDEPENDENT_EFFECTIVE_N_CLOSED": "FAIL",
    "REJECT_REASON_AND_EXCLUSION_CLOSURE_CLOSED": "PASS_AGGREGATE_CLOSURE",
}
GSE256185_ROW_PREFLIGHT_CONFIG_PATH = (
    "configs/route_a_v3_gse256185_aggregate_row_level_qualification_preflight_v1.json"
)
GSE256185_ROW_PREFLIGHT_SCRIPT_PATH = (
    "scripts/route_a_v3/preflight_gse256185_aggregate_row_level_qualification.py"
)
GSE256185_ROW_PREFLIGHT_TEST_PATH = (
    "tests/route_a_v3/test_preflight_gse256185_aggregate_row_level_qualification.py"
)
GSE256185_ROW_PREFLIGHT_RUNTIME_CONFIG_PATH = (
    "configs/route_a_v3_gse256185_aggregate_row_level_qualification_"
    "preflight_runtime_sync_v1.json"
)
GSE256185_ROW_PREFLIGHT_STATIC_LEAF_SHA256 = {
    GSE256185_ROW_PREFLIGHT_CONFIG_PATH: (
        "50d37a4a5f49e0f9b458b35ada343aa8d77229fff5ac13bdff8ff88d78d4e7e8"
    ),
    GSE256185_ROW_PREFLIGHT_SCRIPT_PATH: (
        "20d0b4a3355fb98cf3c67268f314c0d1fa571034772074abdfbf9f6037c90f0f"
    ),
    GSE256185_ROW_PREFLIGHT_TEST_PATH: (
        "eaece480ead17f5ef3a8eb2662d3a0372e59910d07eee0eb77f1d55b4580e226"
    ),
}
GSE256185_ROW_PREFLIGHT_REPORT_PATH = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
    "GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_V1_"
    "20260813T232328P0800_4858156/"
    "GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_V1.json"
)
GSE256185_ROW_PREFLIGHT_REPORT_BYTES = 15214
GSE256185_ROW_PREFLIGHT_REPORT_SHA256 = (
    "6ee9c7de65422d3223347d8afbae49a37f71e21b20a932c9e50547c09c9d1a54"
)
GSE256185_ROW_PREFLIGHT_RECORDED_AT = "2026-08-13T15:23:47.452529Z"
GSE256185_ROW_PREFLIGHT_IMPLEMENTATION_COMMIT = (
    "2a577ef41b634a4c999740e12ed60ffb6aca80cb"
)
GSE256185_ROW_PREFLIGHT_BINDING_COMMIT = (
    "48581569b1ac86ec23d5662ea014f152384dc673"
)
GSE256185_PUBLIC_GEOMETRY_CONFIG_PATH = (
    "configs/route_a_v3_gse256185_public_identifier_pool_geometry_preflight_v1.json"
)
GSE256185_PUBLIC_GEOMETRY_SCRIPT_PATH = (
    "scripts/route_a_v3/preflight_gse256185_public_identifier_pool_geometry.py"
)
GSE256185_PUBLIC_GEOMETRY_TEST_PATH = (
    "tests/route_a_v3/test_preflight_gse256185_public_identifier_pool_geometry.py"
)
GSE256185_PUBLIC_GEOMETRY_RUNTIME_CONFIG_PATH = (
    "configs/route_a_v3_gse256185_public_identifier_pool_geometry_"
    "preflight_runtime_sync_v1.json"
)
GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF_SHA256 = {
    GSE256185_PUBLIC_GEOMETRY_CONFIG_PATH: (
        "cd5aca36925d60797f18eccedd5ea6fa253c36f93d6e24c832a452b86169a34f"
    ),
    GSE256185_PUBLIC_GEOMETRY_SCRIPT_PATH: (
        "300c80811b9e4159c011041741d36cdbd265993381e9dbe08741cb89c639c196"
    ),
    GSE256185_PUBLIC_GEOMETRY_TEST_PATH: (
        "6eb989b33e0b770f16ccd5b1e1c9118a8e78ed548cbb4814391097d793ac0892"
    ),
}
GSE256185_PUBLIC_GEOMETRY_REPORT_PATH = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
    "GSE256185_PUBLIC_IDENTIFIER_POOL_GEOMETRY_PREFLIGHT_V1_"
    "20260813T210613P0800_c95fe8e/"
    "GSE256185_PUBLIC_IDENTIFIER_POOL_GEOMETRY_PREFLIGHT_V1.json"
)
GSE256185_PUBLIC_GEOMETRY_REPORT_BYTES = 7441
GSE256185_PUBLIC_GEOMETRY_REPORT_SHA256 = (
    "06a8fed9599266ca170303705e6d78a1555e7b8949254e5355c7b17aff742db8"
)
GSE256185_PUBLIC_GEOMETRY_RECORDED_AT = "2026-08-13T13:06:51.607157Z"
GSE256185_AUTHORITY_COMMIT = "1ee575799a4b3289f9b7d684b4b31885dde0bd50"
GSE256185_AUTHORITY_RUNTIME_I1_COMMIT = (
    "2bd38ecf99002bb9583417adb2883375109d2759"
)
GSE256185_AUTHORITY_RUNTIME_I2_COMMIT = (
    "6d3508a5386b709b3ebc806d6915791a75ef4539"
)
GSE256185_AUTHORITY_RUNTIME_BINDING_COMMIT = (
    "e67be74d793a2a459b655ca11d38f86a9d52b7db"
)
GSE256185_PUBLIC_GEOMETRY_I1_COMMIT = (
    "fbf7c25e86e3b147df492ac1b934593e391a904a"
)
GSE256185_PUBLIC_GEOMETRY_I2_COMMIT = (
    "96dadcb67edd0d494f5b80965590a5e306cabbe1"
)
GSE256185_PUBLIC_GEOMETRY_B2_COMMIT = (
    "6d1922a286adfc4e9a14d920d46f0648a02317cd"
)
GSE256185_PUBLIC_GEOMETRY_IMPLEMENTATION_COMMIT = (
    "bb346b0c09a18ef844d576f3679caee419a0f0a0"
)
GSE256185_PUBLIC_GEOMETRY_BINDING_COMMIT = (
    "c95fe8e06f51c852c7ca1289d8df9d00a6754daa"
)
A6_PROTOCOL_ID = "ROUTE_A_V3_A6_CPU_EXACT_ABSORBING_DAG_V1"
A6_PROTOCOL_CONFIG_PATH = "configs/route_a_v3_a6_cpu_exact_absorbing_dag_v1.json"
A6_PRODUCER_PATH = "scripts/route_a_v3/run_a6_cpu_exact_absorbing_dag.py"
A6_FOCUSED_TEST_PATH = "tests/route_a_v3/test_a6_cpu_exact_absorbing_dag.py"
A6_STATIC_PRODUCER_LEAF_SHA256 = {
    A6_PROTOCOL_CONFIG_PATH: "84e9a3f21ac6293faa167eb08eb40e8886bfe43daaa374b2c7613fbc9baecab8",
    A6_PRODUCER_PATH: "4cc0e10784c218cf4fa18ede1280cb84e1b7daf2553db4d82e0ee88f71e0f7c8",
    A6_FOCUSED_TEST_PATH: "7e8253be856645f276689dc032f9581b9ea172faea929dff60ba07995c709e24",
}
A6_REPORT_EVIDENCE_ID = "A6_CPU_EXACT_ABSORBING_DAG_V1"
A6_REPORT_PATH = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A6/"
    "A6_CPU_EXACT_ABSORBING_DAG_V1/A6_CPU_EXACT_ABSORBING_DAG_REPORT.json"
)
A6_REPORT_BYTES = 14574
A6_REPORT_SHA256 = "e16e2536eb68b219644d64e40859c8e2270123c227994c0fa37e8b7084992e83"
A6_REPORT_RECORDED_AT = "2026-08-13T16:44:55+08:00"
A6_FROZEN_BASE_COMMIT = "db297787b3cd9f74908a1ae726cb64b19a9161fb"
A6_IMPLEMENTATION_COMMIT = "92ab710b3ddf570af305d94bdf68c36cff84aad2"
A6_BINDING_COMMIT = "de33f0d6337692409d17b2ac75e056e148815e72"
A6_GILLESPIE_PROTOCOL_ID = "ROUTE_A_V3_A6_CPU_LEGAL_CTMC_PARTIAL_V1"
A6_GILLESPIE_CONFIG_PATH = "configs/route_a_v3_a6_cpu_legal_ctmc_partial_v1.json"
A6_GILLESPIE_PRODUCER_PATH = "scripts/route_a_v3/run_a6_cpu_legal_ctmc_partial.py"
A6_GILLESPIE_FOCUSED_TEST_PATH = "tests/route_a_v3/test_a6_cpu_legal_ctmc_partial.py"
A6_GILLESPIE_STATIC_PRODUCER_LEAF_SHA256 = {
    A6_GILLESPIE_CONFIG_PATH: "c262ced2f9b7b0b951785a2a4dbbbc074c1bbc77a75d3229d7c2e82a0296da3a",
    A6_GILLESPIE_PRODUCER_PATH: "5a8f3454bd32bac60d5f574fde4bf364f738cd58086ea6ad6d523c27ea0ad4c4",
    A6_GILLESPIE_FOCUSED_TEST_PATH: "152172840f7f93efaab742bf4332568944c5121db2584a53a639724a47ba9a72",
}
A6_GILLESPIE_REPORT_EVIDENCE_ID = "A6_CPU_LEGAL_CTMC_PARTIAL_V1"
A6_GILLESPIE_REPORT_PATH = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A6/"
    "A6_CPU_LEGAL_CTMC_PARTIAL_V1/A6_CPU_LEGAL_CTMC_PARTIAL_REPORT.json"
)
A6_GILLESPIE_REPORT_BYTES = 4716
A6_GILLESPIE_REPORT_SHA256 = "2e3c264b8c9bd2ebc180704ed02528b22b4493db36f0525543d5efeba282192f"
A6_GILLESPIE_REPORT_RECORDED_AT = "2026-08-13T18:53:08+08:00"
A6_GILLESPIE_FROZEN_BASE_COMMIT = "a28bf3c67cf538a8754fde8505635b2ef2c3d68b"
A6_GILLESPIE_INITIAL_IMPLEMENTATION_COMMIT = "b5d729a1c75eb756ecf4f06fb8d073f7db394e46"
A6_GILLESPIE_INITIAL_BINDING_COMMIT = "edb261f2065eeab601c2464fc669426b6d329b8b"
A6_GILLESPIE_IMPLEMENTATION_COMMIT = "adc1e85eca4a597f45e583154c0565e8ef748db1"
A6_GILLESPIE_BINDING_COMMIT = "6dc3d38bd58ec6ee93de5971aef03138806a3586"
DEC020_AUTHORITY_LEDGER_AT = "2026-08-13T12:15:00+08:00"
DEC020_AUTHORITY_MANIFEST_AT = "2026-08-13T12:15:01+08:00"
DEC020_AUTHORITY_MANIFEST_STATUS = (
    "A1_DEC020_AUTHORITY_REGISTERED_EVT049_SETTLED_"
    "RUNTIME_SYNC_PENDING_FRESH_EVENT"
)
DEC020_RUNTIME_SYNC_STATUS = "PENDING_FRESH_EVENT_AFTER_SETTLED_EVT_049"
DEC020_PENDING_RUNTIME_EVENT_ID = "PENDING_FRESH_RUNTIME_EVENT_ID"
GSE200304_DEC020_V4_LEDGER_AT = "2026-08-13T14:40:00+08:00"
GSE200304_DEC020_V4_MANIFEST_AT = "2026-08-13T14:40:01+08:00"
GSE200304_DEC020_V4_EVIDENCE_UPDATE_ID = "GSE200304_DEC020_SCRATCH_ROUTE_REPORTED_ENDPOINT_A1_ADJUDICATION_V4"
GSE200304_DEC020_V4_LINEAGE_ID = "gse200304_dec020_scratch_route_reported_endpoint_a1_adjudication_v4"
GSE200304_DEC020_V4_MANIFEST_STATUS = "A1_GSE200304_DEC020_SCRATCH_ROUTE_V4_QUALIFIED_LEDGER_REGISTERED_EVT050_SETTLED_PENDING_FRESH_RUNTIME_EVENT"
GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH = "configs/route_a_v3_gse200304_dec020_reported_endpoint_a1_activation_v4.json"
GSE200304_DEC020_V4_SCRIPT_PATH = "scripts/route_a_v3/adjudicate_gse200304_dec020_reported_endpoint_a1_v4.py"
GSE200304_DEC020_V4_TEST_PATH = "tests/route_a_v3/test_adjudicate_gse200304_dec020_reported_endpoint_a1_v4.py"
GSE200304_DEC020_V4_STATIC_LEAF_SHA256 = {
    GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH: "de0f7f4fbf8194d112e185806a6e0a19ae006714c5f1d2786629c9c2b1c85869",
    GSE200304_DEC020_V4_SCRIPT_PATH: "744b5cb10633bf7ca6d1e0d723f7104fe94b1ff725a1f9b9884b9c7e120d7b16",
    GSE200304_DEC020_V4_TEST_PATH: "298304c19c3faa6103221913cb8b30e25fd56d3b55eb65ea04f2e8d82f7c3954",
}
GSE200304_DEC020_V4_IMPLEMENTATION_COMMIT = "89618c2aa8aa383286de24b6363849f9743f1685"
GSE200304_DEC020_V4_BINDING_COMMIT = "0eab96955764998f6cc767dbe9754091f9becd83"
GSE200304_DEC020_V4_CONFIG_CORE_SHA256 = "ae51c0682e1e45a062f3e1dcb7ee5d2defb684977591877650d9edcc06827545"
GSE200304_DEC020_V4_DESCRIPTOR_SET_SHA256 = "e682e3ed60fef02e894d533b647c4e06b7a58083e3dcce8b0d7b18f0e64cec03"
DEC020_SCRATCH_ROUTE = "SCRATCH_ONLY_NO_FOUNDATION_NO_EXTERNAL_LEARNED_INPUTS"
DEC020_FOUNDATION_ROUTE = "FOUNDATION_CHECKPOINT"
DEC020_ROUTE_GATE = "MODEL_INPUT_ROUTE_AND_ROUTE_CONDITIONAL_EXPOSURE_CLOSED"
DEC020_SCRATCH_EXPOSURE_STATUS = (
    "NOT_APPLICABLE_BY_FROZEN_NO_EXTERNAL_LEARNED_INPUT_ROUTE"
)
DEC020_FORBIDDEN_EXTERNAL_LEARNED_INPUTS = [
    "FOUNDATION_OR_OTHER_PRETRAINED_CHECKPOINT_WEIGHTS",
    "EXTERNALLY_TRAINED_INITIALIZATION_OR_WARM_START",
    "EXTERNALLY_LEARNED_TOKEN_EMBEDDINGS",
    "EXTERNALLY_TRAINED_ENCODER_ADAPTER_OR_HEAD",
    "EXTERNAL_LEARNED_REPRESENTATIONS_FEATURES_OR_LOGITS",
    "EXTERNAL_TEACHER_OR_DISTILLATION_TARGETS",
    "EXTERNAL_PSEUDOLABELS",
    "CHECKPOINT_DERIVED_CALIBRATION_OR_NORMALIZATION_STATISTICS",
    "LEARNED_RETRIEVAL_INDEX_RERANKER_OR_SCORE",
    "ANY_PARAMETER_TENSOR_FEATURE_OR_SCORE_LEARNED_FROM_EXTERNAL_DATA",
]
DEC020_CURRENT_ZERO_COUNTS = {
    "ordinary": 0,
    "a1": 0,
    "true_a2": 0,
    "canonical_records": 0,
}
DEC020_FUTURE_V4_REGISTRATION = {
    "lifecycle_status": "DEFERRED_TO_POST_AUTHORITY_IMPLEMENTATION_COMMIT_I",
    "expected_static_leaf_count": 3,
    "expected_static_leaf_roles": ["CONFIG", "SCRIPT", "FOCUSED_TEST"],
    "registered_static_leaf_count": 0,
    "registered_in_static_manifest": False,
    "leaf_paths": [],
    "leaf_sha256": [],
    "implementation_commit": "UNKNOWN_NOT_ASSERTED",
    "may_execute": False,
    "may_adjudicate": False,
}
GSE200304_DEC019_ONE_BLOCKER_LEDGER_AT = "2026-08-12T16:09:52+08:00"
GSE200304_DEC019_ONE_BLOCKER_MANIFEST_AT = "2026-08-12T16:09:53+08:00"
POST_FAIL_ACQUISITION_LEDGER_AT = "2026-08-12T19:30:00+08:00"
POST_FAIL_ACQUISITION_MANIFEST_AT = "2026-08-12T19:30:01+08:00"
POST_FAIL_ACQUISITION_EVIDENCE_UPDATE_ID = (
    "GSE200304_CHECKPOINT_EXPOSURE_FAIL_AND_"
    "GSE149487_PUBLIC_ASSET_ACQUISITION_V1"
)
GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LEDGER_AT = (
    "2026-08-12T23:55:00+08:00"
)
GSE232572_DEVELOPMENT_V3_MATERIALIZATION_MANIFEST_AT = (
    "2026-08-12T23:55:01+08:00"
)
GSE232572_DEVELOPMENT_V3_MATERIALIZATION_EVIDENCE_UPDATE_ID = (
    "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_V1"
)
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_LEDGER_AT = (
    "2026-08-13T01:03:09+08:00"
)
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_MANIFEST_AT = (
    "2026-08-13T01:03:10+08:00"
)
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_EVIDENCE_UPDATE_ID = (
    "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_V1"
)
ACTIVE_AMENDMENT_DECISION_IDS = [
    "V3-DEC-017",
    "V3-DEC-018",
    "V3-DEC-019",
    "V3-DEC-020",
    "V3-DEC-021",
    "V3-DEC-022",
    "V3-DEC-023",
    "V3-DEC-024",
    "V3-DEC-027",
    "V3-DEC-028",
]
ACTIVE_AMENDMENT_PATHS = {
    "V3-DEC-019": DEC019_AMENDMENT_PATH,
    "V3-DEC-020": DEC020_AMENDMENT_PATH,
    "V3-DEC-021": DEC021_AMENDMENT_PATH,
    "V3-DEC-022": DEC022_AMENDMENT_PATH,
    "V3-DEC-023": DEC023_AMENDMENT_PATH,
    "V3-DEC-024": DEC024_AMENDMENT_PATH,
    "V3-DEC-027": DEC027_AMENDMENT_PATH,
    "V3-DEC-028": DEC028_AMENDMENT_PATH,
}
DEC019_IMMUTABLE_LEAF_SHA256 = {
    DEC019_AMENDMENT_PATH: "8c82e564398f0735fe4976f875fe91f053937b05044e5232e237694a2b36e1ca",
}
DEC020_FROZEN_AUTHORITY_LEAF_SHA256 = {
    CONFIG_PATH: "c908ac57b7c9667398f616a0ccf7101b41451b80bf169e768131844d3b63a678",
    A1_QUALIFICATION_CONFIG_PATH: "ac1ed9e78bf88d916f5599e3a2e75e79df1504c16ba108a12f7e28cfd3da2e20",
    SUPERSESSION_PATH: "d7c0559742a44b4f0b6f8c941e734da52359c2733ba759ec2acd8ca40b07e62d",
    DEC020_AMENDMENT_PATH: "0cfbe6e35c2c7f3b19756b8aee41dc91b2a8f05b249a5b6e9cacf90185c56026",
    DECISION_LOG_PATH: "1332e789758a11687d3bcbbe95e0a5c7e852694e25ed90563d280006d94caced",
    "docs/execution/route_a_v3_data_role_registry.yaml": "d06bfcfb8d265153a44d270c7bc40e5dd462a5e3bdde631d91519c7d7e394852",
    "docs/execution/route_a_v3_split_registry.yaml": "52e1146027956e024dd6194ff18862e542e27fff81e8fc6b6d8aeaa972b8259c",
    "docs/execution/route_a_v3_task_registry.yaml": "bf3066a7534041374685e9ebe9ac8c840e53ceec1acbb076a72a758d397c63f2",
    "docs/execution/route_a_v3_task_split_matrix.yaml": "db23e96b6977339237956de57309d04a9e692bf937a8d34427d2e1b6cc150db8",
    "docs/execution/route_a_v3_claim_evidence_matrix.yaml": "9f5226ac78dd6c3848ba5ceb42742918de66ec459f951bb845ccaf21958a88f9",
}
DEC020_ACTIVE_AUTHORITY_LEAF_SHA256 = {
    **DEC020_FROZEN_AUTHORITY_LEAF_SHA256,
    "docs/execution/route_a_v3_claim_evidence_matrix.yaml": "214279390d09c2857735c9cfa041ce38c45a7542142d4b0941ad7c035a7ee81a",
}
DEC020_PRESERVED_AUTHORITY_LEAF_SHA256 = {
    DEC019_AMENDMENT_PATH: "8c82e564398f0735fe4976f875fe91f053937b05044e5232e237694a2b36e1ca",
    DEC020_AMENDMENT_PATH: "0cfbe6e35c2c7f3b19756b8aee41dc91b2a8f05b249a5b6e9cacf90185c56026",
    "docs/execution/route_a_v3_split_registry.yaml": "52e1146027956e024dd6194ff18862e542e27fff81e8fc6b6d8aeaa972b8259c",
    "docs/execution/route_a_v3_task_registry.yaml": "bf3066a7534041374685e9ebe9ac8c840e53ceec1acbb076a72a758d397c63f2",
    "docs/execution/route_a_v3_task_split_matrix.yaml": "db23e96b6977339237956de57309d04a9e692bf937a8d34427d2e1b6cc150db8",
    "docs/execution/route_a_v3_claim_evidence_matrix.yaml": "214279390d09c2857735c9cfa041ce38c45a7542142d4b0941ad7c035a7ee81a",
}
DEC020_AUTHORITY_LEDGER_PATHS = (
    A1_INTERIM_PATH,
    REGISTRY_MANIFEST_PATH,
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
DEC020_AUTHORITY_COMMIT_EXACT_CHANGED_PATHS = (
    *DEC020_FROZEN_AUTHORITY_LEAF_SHA256,
    *DEC020_AUTHORITY_LEDGER_PATHS,
)
DEC021_ACTIVE_AUTHORITY_LEAF_SHA256 = {
    CONFIG_PATH: "c9d685d6d300f0bcb5287401ce70ead3fedee940653d843290c94e3e3060b58f",
    A1_QUALIFICATION_CONFIG_PATH: "9344feca31c315ae273487347cc5275cb2433261838ea997035021a907a3cb8a",
    SUPERSESSION_PATH: "50b1811f8331c1ed76c69c31f29b55a6a659d37d4591b7711d0b0ae9beec43a0",
    DEC019_AMENDMENT_PATH: "8c82e564398f0735fe4976f875fe91f053937b05044e5232e237694a2b36e1ca",
    DEC020_AMENDMENT_PATH: "0cfbe6e35c2c7f3b19756b8aee41dc91b2a8f05b249a5b6e9cacf90185c56026",
    DEC021_AMENDMENT_PATH: "2a7b05e40434398b1d39396280ca019f48164fbb70b5b6058123bc653c400d3d",
    DECISION_LOG_PATH: "874d83156e337cc613b3df9f57bbd10f19a2585ffefbeb3e52d99aca696bc7ac",
    "docs/execution/route_a_v3_data_role_registry.yaml": "3579249154309a828f193c18785a9d6d9f4b325780cffbc014c2bf524b5b47d6",
    "docs/execution/route_a_v3_split_registry.yaml": "52e1146027956e024dd6194ff18862e542e27fff81e8fc6b6d8aeaa972b8259c",
    "docs/execution/route_a_v3_task_registry.yaml": "bf3066a7534041374685e9ebe9ac8c840e53ceec1acbb076a72a758d397c63f2",
    "docs/execution/route_a_v3_task_split_matrix.yaml": "db23e96b6977339237956de57309d04a9e692bf937a8d34427d2e1b6cc150db8",
    "docs/execution/route_a_v3_claim_evidence_matrix.yaml": "214279390d09c2857735c9cfa041ce38c45a7542142d4b0941ad7c035a7ee81a",
}
DEC021_AUTHORITY_EXACT_CHANGED_PATHS = (
    CONFIG_PATH,
    A1_QUALIFICATION_CONFIG_PATH,
    DEC021_AMENDMENT_PATH,
    SUPERSESSION_PATH,
    "docs/execution/route_a_v3_data_role_registry.yaml",
    DECISION_LOG_PATH,
    A1_INTERIM_PATH,
    REGISTRY_MANIFEST_PATH,
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
DEC021_PRESERVED_HISTORICAL_LEAF_SHA256 = {
    GOAL_PATH: SOURCE_CONTRACT_SHA256,
    DEC019_AMENDMENT_PATH: "8c82e564398f0735fe4976f875fe91f053937b05044e5232e237694a2b36e1ca",
    DEC020_AMENDMENT_PATH: "0cfbe6e35c2c7f3b19756b8aee41dc91b2a8f05b249a5b6e9cacf90185c56026",
    DEC021_AMENDMENT_PATH: "2a7b05e40434398b1d39396280ca019f48164fbb70b5b6058123bc653c400d3d",
    "docs/execution/route_a_v3_split_registry.yaml": "52e1146027956e024dd6194ff18862e542e27fff81e8fc6b6d8aeaa972b8259c",
    "docs/execution/route_a_v3_task_registry.yaml": "bf3066a7534041374685e9ebe9ac8c840e53ceec1acbb076a72a758d397c63f2",
    "docs/execution/route_a_v3_task_split_matrix.yaml": "db23e96b6977339237956de57309d04a9e692bf937a8d34427d2e1b6cc150db8",
    "docs/execution/route_a_v3_claim_evidence_matrix.yaml": "214279390d09c2857735c9cfa041ce38c45a7542142d4b0941ad7c035a7ee81a",
}
DEC022_ACTIVE_AUTHORITY_LEAF_SHA256 = {
    CONFIG_PATH: "66d5303d4644c5da401ca1e4295da854c0a51c4d98fcd92556dc1d0abae9e2cd",
    A1_QUALIFICATION_CONFIG_PATH: "baefc579eacfb5e43ded77c8443daade449e7cefeabeab1a1818b77800ac6184",
    SUPERSESSION_PATH: "eed3815b1825e2731cc60e03401f3046f440d76cd9c58d3a23f7cd0ef87ed4f8",
    DEC019_AMENDMENT_PATH: "8c82e564398f0735fe4976f875fe91f053937b05044e5232e237694a2b36e1ca",
    DEC020_AMENDMENT_PATH: "0cfbe6e35c2c7f3b19756b8aee41dc91b2a8f05b249a5b6e9cacf90185c56026",
    DEC021_AMENDMENT_PATH: "2a7b05e40434398b1d39396280ca019f48164fbb70b5b6058123bc653c400d3d",
    DEC022_AMENDMENT_PATH: "3496c3d72c62cef2e26e5414e06b3593656ea80599b91c56f228f81d6125e3c7",
    DECISION_LOG_PATH: "3a27abc9dbc4924e3a12e7662418e34472509a5345dec6948ae7279ca5455ee1",
    "docs/execution/route_a_v3_data_role_registry.yaml": "943233c41a4ec40cef65bbe3efda38ce8509d816e2e9a72e1f3ab4b4fe0f98e7",
    "docs/execution/route_a_v3_split_registry.yaml": "52e1146027956e024dd6194ff18862e542e27fff81e8fc6b6d8aeaa972b8259c",
    "docs/execution/route_a_v3_task_registry.yaml": "bf3066a7534041374685e9ebe9ac8c840e53ceec1acbb076a72a758d397c63f2",
    "docs/execution/route_a_v3_task_split_matrix.yaml": "db23e96b6977339237956de57309d04a9e692bf937a8d34427d2e1b6cc150db8",
    "docs/execution/route_a_v3_claim_evidence_matrix.yaml": "214279390d09c2857735c9cfa041ce38c45a7542142d4b0941ad7c035a7ee81a",
}
DEC022_AUTHORITY_EXACT_CHANGED_PATHS = (
    CONFIG_PATH,
    A1_QUALIFICATION_CONFIG_PATH,
    DEC022_AMENDMENT_PATH,
    SUPERSESSION_PATH,
    "docs/execution/route_a_v3_data_role_registry.yaml",
    DECISION_LOG_PATH,
    A1_INTERIM_PATH,
    REGISTRY_MANIFEST_PATH,
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
DEC023_FROZEN_AUTHORITY_LEAF_SHA256 = dict(DEC022_ACTIVE_AUTHORITY_LEAF_SHA256)
DEC023_FROZEN_AUTHORITY_LEAF_SHA256.update(
    {
        CONFIG_PATH: "df38455904d67f22a2fea1fb08a3314cd4fb120e91ea711427ad1689653ba8ce",
        A1_QUALIFICATION_CONFIG_PATH: "98de408ec423836efac75bcd75b4fd940e9fbd52a0bf1b3c397ea0c67e548740",
        SUPERSESSION_PATH: "0edfddd90ebea11db1cebb7084bdf28dd8a99426c3956a81cdfd7a4a9ccb12e2",
        DEC023_AMENDMENT_PATH: "44622c7f589d841105cb21d0b35219aa9163fe4d54350671106408d4c8439e4a",
        DECISION_LOG_PATH: "8e514512ccd63d87a596231b11183c06765ba50ba3736adc165f141da8fa13d0",
        "docs/execution/route_a_v3_data_role_registry.yaml": "bb577d4ce7d7dc673f41bb182b7868f66816c15a3ed4235c98e0839292e75d6b",
    }
)
DEC023_ACTIVE_AUTHORITY_LEAF_SHA256 = dict(DEC023_FROZEN_AUTHORITY_LEAF_SHA256)
DEC023_ACTIVE_AUTHORITY_LEAF_SHA256[
    "docs/execution/route_a_v3_task_registry.yaml"
] = DEC023_ACTIVE_TASK_REGISTRY_SHA256
DEC024_ACTIVE_AUTHORITY_LEAF_SHA256 = dict(DEC023_ACTIVE_AUTHORITY_LEAF_SHA256)
DEC024_ACTIVE_AUTHORITY_LEAF_SHA256.update(
    {
        CONFIG_PATH: "01cb7bb052da7459e946828b7c92dede8f257e3f13abba3534455b370ef09b74",
        A1_QUALIFICATION_CONFIG_PATH: "46ce6ff3648f1abec47bfc9eb63045759ab8676b12fd3c38c5efe31d64f37c41",
        SUPERSESSION_PATH: "d403a4aa7db9343848be74ae061b8196613525ff03600b1034864d7d803c7beb",
        DEC024_AMENDMENT_PATH: "163c5b744a8d68e6e0bd3afad378c3cd8611d42e7f8ff881291557049d908eac",
        DECISION_LOG_PATH: "06a031cf67ead3417942938a17f4783a6f2168866f54e1fc2abe1b7fa938c0c3",
        "docs/execution/route_a_v3_data_role_registry.yaml": "f62034239854d494c45196d2535895e9593cc38655d7b9042719fb912cf08e45",
        "docs/execution/route_a_v3_task_registry.yaml": "c1d9920cc3d28c7ee63d9649a69d6b6856b9b25115f6ca4f1d392a067a0d5dae",
    }
)
DEC027_ACTIVE_AUTHORITY_LEAF_SHA256 = dict(DEC024_ACTIVE_AUTHORITY_LEAF_SHA256)
DEC027_ACTIVE_AUTHORITY_LEAF_SHA256.update(
    {
        CONFIG_PATH: "c5ec7d236443b506c09fd3f09e149ce5d082daff618887989af6e59472727a27",
        A1_QUALIFICATION_CONFIG_PATH: "261339c38f4b8bbd48bf8f63f6a588be57af9f6229119e84bf661d7ee8f855db",
        SUPERSESSION_PATH: "1c4b6e29c09eb24798207138047b68909d7bda8bacc3a2eab8e17a7ca789b44b",
        DEC027_AMENDMENT_PATH: "2a27c296539e8e665873363778d91cc223f56f933a815e9509b10a7267f6b5c4",
        DECISION_LOG_PATH: "e0dc2a7fb186c5c8d00c1c5604602b1b2f87b26241191d4da55a405a02387e05",
        "docs/execution/route_a_v3_data_role_registry.yaml": "80217a8114286f84960237819ac5b2d5828afbc23af118576541cc7cee64ae4e",
        "docs/execution/route_a_v3_task_registry.yaml": "a64d0b8bb5eb466b06daa46ed109bd19901ee775910bc5cc9221c39ead63a4bc",
    }
)
DEC028_ACTIVE_AUTHORITY_LEAF_SHA256 = dict(DEC027_ACTIVE_AUTHORITY_LEAF_SHA256)
DEC028_ACTIVE_AUTHORITY_LEAF_SHA256.update(
    {
        CONFIG_PATH: "6a6c97fe28b07738b42175183be556f36d5477d67c1180a69df75d0850790e41",
        A1_QUALIFICATION_CONFIG_PATH: "0dfb6dba77ab80628451d0a87f09144864965a5ba65d1a05ce9a32ae3441c7bd",
        SUPERSESSION_PATH: "b13146dd852c8e8c6f5bf29972156d9c2f11c13bc09b2da5f5ea7f78c275cc93",
        DEC028_AMENDMENT_PATH: "98182778439b7f2da39fc26c32be6fe50f9710cb01682fb1ada6c5a440c0b26f",
        DEC028_HUMAN_CONTRACT_PATH: "1864f28a963f8feda113d4b5de5f92b23ebad58b9fe6471136466b9d314b0fd0",
        DEC028_PROTOCOL_PATH: "7d5c461189109f60b7bef2f96251327be641b0e154a05fe0ff53a973672bb47b",
        DEC028_EXECUTION_PATH: "0ad5f3a797110071dd01ea5ccbb3a37c1780b334442576aa671e5081882f5801",
        DECISION_LOG_PATH: "2248029779742897896f5a944d459d65444b959f5cde93f3a67bb91e16b425a0",
        "docs/execution/route_a_v3_data_role_registry.yaml": "b4b9949b3c41bcd501980d52f3cdafe37bf3d5a4b23b35dd9113337f1a1bb8e0",
        "docs/execution/route_a_v3_split_registry.yaml": "222a63d18e1ccca0a97ed3fdc2a3a656aefe283be3f3bdf23f19f5f1e04e2051",
        "docs/execution/route_a_v3_task_registry.yaml": "113c98a78644f5f5e432f59de7f8bc34f9956b13998e68b785f350cf5289d917",
        "docs/execution/route_a_v3_task_split_matrix.yaml": "46c122705ffec30efdbfb7f796c8866506bd07a847a6fcc8b622c4ada0ff50d0",
        "docs/execution/route_a_v3_claim_evidence_matrix.yaml": "ea188483008fcb6bb7df15de25b1ebb0264044eeee8d96d7c070d0b264ecea7e",
    }
)
DEC023_AUTHORITY_EXACT_CHANGED_PATHS = (
    CONFIG_PATH,
    A1_QUALIFICATION_CONFIG_PATH,
    DEC023_AMENDMENT_PATH,
    SUPERSESSION_PATH,
    "docs/execution/route_a_v3_data_role_registry.yaml",
    DECISION_LOG_PATH,
    A1_INTERIM_PATH,
    REGISTRY_MANIFEST_PATH,
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
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
GSE200304_UPSTREAM_AUTHORITY_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_upstream_authority_viability_v1.json"
)
GSE200304_UPSTREAM_AUTHORITY_SCRIPT_PATH = (
    "scripts/route_a_v3/produce_gse200304_upstream_authority_viability.py"
)
GSE200304_UPSTREAM_AUTHORITY_TEST_PATH = (
    "tests/route_a_v3/test_produce_gse200304_upstream_authority_viability.py"
)
GSE200304_DEC019_UPSTREAM_PASS_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_dec019_upstream_pass_gate_pack_v1.json"
)
GSE200304_DEC019_UPSTREAM_PASS_SCRIPT_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_upstream_pass_gate_pack.py"
)
GSE200304_DEC019_UPSTREAM_PASS_TEST_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_upstream_pass_gate_pack.py"
)
GSE200304_DEC019_V3_HISTORICAL_D2_CONFIG_SHA256 = (
    "88fa21a08df60935f3d2d1bf44c6573889c22c110021146acf241fd92d6b5a13"
)
GSE200304_DEC019_V3_HISTORICAL_D2_DESCRIPTOR_SET_SHA256 = (
    "14223d0193e4b3a4a3c1d98a5894849dd429e6eed021ff98e6697e73ac286a40"
)
GSE200304_DEC019_V3_HISTORICAL_CORE_SHA256 = (
    "13394ac6a9b9ec6e6241d0d9b1048ecfa5c90874c7447991fc2a8248a574c170"
)
GSE200304_DEC019_V3_HISTORICAL_D3_CONFIG_SHA256 = (
    "e7040fedd6e7217d402c36597c177f08fdf4921c55aced7379a1580c33c31891"
)
GSE200304_DEC019_V3_HISTORICAL_D3_DESCRIPTOR_SET_SHA256 = (
    "97e2d5ca135f2e5668ef513de0247d5973481f6532f99beac9e9d8d9a828148b"
)
GSE200304_DEC019_V3_CONFIG_SHA256 = (
    "0716e4f1a96280d7e33858df037e05d975119a82ff36ee6794f5ffac1c92bb44"
)
GSE200304_DEC019_V3_CONFIG_CORE_SHA256 = (
    "f2a3c59dddd047dd7f211926042d4da7120929e538234a97cd72041573bd0172"
)
GSE200304_DEC019_V3_DESCRIPTOR_SET_SHA256 = (
    "ef59cf8279848858f038bbc6ec6c194481661cacf3de271425517e018bb4e1cb"
)
GSE200304_DEC019_V3_HISTORICAL_SCRIPT_SHA256 = (
    "9cd4411fcb02e1feed913b799296351e38ab9071b9506611318645e41b8dbbfe"
)
GSE200304_DEC019_V3_HISTORICAL_TEST_SHA256 = (
    "8e7b188cfa2e5015fa307acad980f9ff2f45145943384fcadb50d67b1263e1db"
)
GSE200304_DEC019_GROUP_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_dec019_biological_group_authority_gate_v1.json"
)
GSE200304_DEC019_GROUP_SCRIPT_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_biological_group_authority_gate.py"
)
GSE200304_DEC019_GROUP_TEST_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_biological_group_authority_gate.py"
)
GSE200304_DEC019_SPLIT_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_dec019_outcome_blind_split_leakage_gate_v1.json"
)
GSE200304_DEC019_SPLIT_SCRIPT_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_outcome_blind_split_leakage_gate.py"
)
GSE200304_DEC019_SPLIT_TEST_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_outcome_blind_split_leakage_gate.py"
)
GSE200304_DEC019_POWER_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_dec019_prefrozen_power_precision_gate_v1.json"
)
GSE200304_DEC019_POWER_SCRIPT_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_prefrozen_power_precision_gate.py"
)
GSE200304_DEC019_POWER_TEST_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_prefrozen_power_precision_gate.py"
)
GSE200304_CHECKPOINT_EXPOSURE_FAIL_CONFIG_PATH = (
    "configs/route_a_v3_gse200304_dec019_"
    "checkpoint_exposure_fail_current_protocol_v1.json"
)
GSE200304_CHECKPOINT_EXPOSURE_FAIL_SCRIPT_PATH = (
    "scripts/route_a_v3/produce_gse200304_dec019_"
    "checkpoint_exposure_fail_current_protocol.py"
)
GSE200304_CHECKPOINT_EXPOSURE_FAIL_TEST_PATH = (
    "tests/route_a_v3/test_produce_gse200304_dec019_"
    "checkpoint_exposure_fail_current_protocol.py"
)
GSE149487_PUBLIC_ASSET_ACQUISITION_CONFIG_PATH = (
    "configs/route_a_v3_gse149487_public_asset_acquisition_v1.json"
)
GSE149487_PUBLIC_ASSET_ACQUISITION_SCRIPT_PATH = (
    "scripts/route_a_v3/acquire_gse149487_public_assets.py"
)
GSE149487_PUBLIC_ASSET_ACQUISITION_TEST_PATH = (
    "tests/route_a_v3/test_acquire_gse149487_public_assets.py"
)
GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_CONFIG_PATH = (
    "configs/route_a_v3_gse217518_public_authority_preflight_v1.json"
)
GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_SCRIPT_PATH = (
    "scripts/route_a_v3/preflight_gse217518_public_authority.py"
)
GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_TEST_PATH = (
    "tests/route_a_v3/test_preflight_gse217518_public_authority.py"
)
GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_RUNTIME_CONFIG_PATH = (
    "configs/route_a_v3_gse217518_public_authority_preflight_runtime_sync_v1.json"
)
GSE232572_PUBLIC_RECOVERY_AUDIT_CONFIG_PATH = (
    "configs/route_a_v3_gse232572_a1_recovery_v1.json"
)
GSE232572_PUBLIC_RECOVERY_AUDIT_SCRIPT_PATH = (
    "scripts/route_a_v3/recover_gse232572_a1.py"
)
GSE232572_PUBLIC_RECOVERY_AUDIT_TEST_PATH = (
    "tests/route_a_v3/test_recover_gse232572_a1.py"
)
GSE232572_PUBLIC_RECOVERY_AUDIT_RUNTIME_CONFIG_PATH = (
    "configs/route_a_v3_gse232572_public_recovery_audit_runtime_sync_v1.json"
)
GSE232572_DEVELOPMENT_V3_MATERIALIZATION_CONFIG_PATH = (
    "configs/route_a_v3_gse232572_development_v3_materialization_v1.json"
)
GSE232572_DEVELOPMENT_V3_MATERIALIZATION_SCRIPT_PATH = (
    "scripts/route_a_v3/materialize_gse232572_development_v3.py"
)
GSE232572_DEVELOPMENT_V3_MATERIALIZATION_TEST_PATH = (
    "tests/route_a_v3/test_materialize_gse232572_development_v3.py"
)
GSE232572_DEVELOPMENT_V3_MATERIALIZATION_RUNTIME_CONFIG_PATH = (
    "configs/route_a_v3_gse232572_development_v3_materialization_runtime_sync_v1.json"
)
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_CONFIG_PATH = (
    "configs/route_a_v3_gse232572_a1_qualification_authority_preflight_v1.json"
)
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_SCRIPT_PATH = (
    "scripts/route_a_v3/preflight_gse232572_a1_qualification_authority.py"
)
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_TEST_PATH = (
    "tests/route_a_v3/test_preflight_gse232572_a1_qualification_authority.py"
)
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_RUNTIME_CONFIG_PATH = (
    "configs/route_a_v3_gse232572_a1_qualification_authority_preflight_runtime_sync_v1.json"
)
POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256 = {
    GSE200304_CHECKPOINT_EXPOSURE_FAIL_CONFIG_PATH: (
        "0759e68b26a927e4acfebc55ac74541363122924734b4f75a55da2a931687404"
    ),
    GSE200304_CHECKPOINT_EXPOSURE_FAIL_SCRIPT_PATH: (
        "29973bf92e82bb4ca60ac3895c59878961bd60aefef11f722c5575f27e6c605c"
    ),
    GSE200304_CHECKPOINT_EXPOSURE_FAIL_TEST_PATH: (
        "6348fbc31efbb01cefbc4bde4b87cc15345f6bdaa98265692f83ab5e5fe6fd49"
    ),
    GSE149487_PUBLIC_ASSET_ACQUISITION_CONFIG_PATH: (
        "14e3e1b1f0b2711f340f8d9098e5bdbd8415cbf9bd4a2ad75f284da8f0d4d4da"
    ),
    GSE149487_PUBLIC_ASSET_ACQUISITION_SCRIPT_PATH: (
        "f14a7571a1e9681b69fb38fda8c9c5ecd322f73f85d8a704d1eb31ad391a1bf9"
    ),
    GSE149487_PUBLIC_ASSET_ACQUISITION_TEST_PATH: (
        "9dec87f0e1d3148a0d13132ed4cd2af37cb7c96110b557fd4e2c064a78cb103f"
    ),
}
GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256 = {
    GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_CONFIG_PATH: (
        "e85c4c49d35d0f7a2e6167eb4dc5dd1a12ee7e4f19638532e5500506da174398"
    ),
    GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_SCRIPT_PATH: (
        "b21540baca568156470a50d07989052a1c97357546bc339c5b90e8791f36080d"
    ),
    GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_TEST_PATH: (
        "ba192586174dccff21b83055608a9f9623fbf48d3b64deeb6768eb4055d69421"
    ),
}
GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF_SHA256 = {
    GSE232572_PUBLIC_RECOVERY_AUDIT_CONFIG_PATH: (
        "43519e11111fea141bbaa7b0eccaf0c4ed023155cdd02597aad426001132e826"
    ),
    GSE232572_PUBLIC_RECOVERY_AUDIT_SCRIPT_PATH: (
        "22a4487758bb2da9a7aaacea332124339a65f9c8a63f22de4538551ab139903c"
    ),
    GSE232572_PUBLIC_RECOVERY_AUDIT_TEST_PATH: (
        "9976a0c9650c0d8b5552bd2ed13c1fe850325828bab10dfff5092e7b8fc920ef"
    ),
}
GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF_SHA256 = {
    GSE232572_DEVELOPMENT_V3_MATERIALIZATION_CONFIG_PATH: (
        "b7b72f4ebc588a45a60debd910fbd669802e02f2a606d5acfab415c17ee6d4c3"
    ),
    GSE232572_DEVELOPMENT_V3_MATERIALIZATION_SCRIPT_PATH: (
        "b608844c3e5ca38037904a050c9312fc8ec2615d7b48d3126eceed31bec126a3"
    ),
    GSE232572_DEVELOPMENT_V3_MATERIALIZATION_TEST_PATH: (
        "2907e91da60d1ec58282e36ffb0149b69b54e3981fab9a6c9f0d49339a383ea5"
    ),
}
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256 = {
    GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_CONFIG_PATH: (
        "84ba2f629a29cea349ce47313fff6d8f209aa60769ebf19769c1ed42e4c31106"
    ),
    GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_SCRIPT_PATH: (
        "a2abaf57f72813b4d4a08abab275d88cf85332cd106fe324892cb02c2a1ea9ba"
    ),
    GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_TEST_PATH: (
        "2fcb96317f3b27b42f9274d297bac1427e4b19ee2ab5cb1bdfe18de417327732"
    ),
}
GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256 = {
    GSE200304_DEC019_V3_SCRIPT_PATH: (
        "7b79ca1a5fff8bd2640234fe30bdbf39533c52e73aa21a93341ed7ee8e34db53"
    ),
    GSE200304_DEC019_V3_TEST_PATH: (
        "a985a94c8258ea58dc7d83103050284aed41dcc4c2dcf997253367d0e6b1a1cf"
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
    GSE200304_UPSTREAM_AUTHORITY_CONFIG_PATH: (
        "c52688866026122488f1d8eef8d0bffebf864b99d78ddcc40c39a26221da76a1"
    ),
    GSE200304_UPSTREAM_AUTHORITY_SCRIPT_PATH: (
        "525635da3d84183e325a17f00fc7cece1517acbd9ce317c2cc4e26a4ba38f03d"
    ),
    GSE200304_UPSTREAM_AUTHORITY_TEST_PATH: (
        "78bca039152874a09dd6a31a0789b712c72b45e60cb9e99e72391809a1bd7035"
    ),
    GSE200304_DEC019_UPSTREAM_PASS_CONFIG_PATH: (
        "a241837bbd68a3c7321bfd96f1b3acf975cdc602762468b5ead161835778a7ae"
    ),
    GSE200304_DEC019_UPSTREAM_PASS_SCRIPT_PATH: (
        "b7ba11974f20472e77111a8f385d2595ad6600bd23d2fd5f969cfa8a91ed459a"
    ),
    GSE200304_DEC019_UPSTREAM_PASS_TEST_PATH: (
        "9e36b125589a38c7e264d15762d1f55af4264b02feb10b7db5e904dd0151dcbe"
    ),
    GSE200304_DEC019_GROUP_CONFIG_PATH: (
        "cdeb3e82656d727c0eb942299b32755aab4382a612911739df17546e464220dc"
    ),
    GSE200304_DEC019_GROUP_SCRIPT_PATH: (
        "c718bee3bedf782f1afe2d653f8a082b9770ed236349350e4ce473995ff35052"
    ),
    GSE200304_DEC019_GROUP_TEST_PATH: (
        "d2db133af013546b4c5d02edd53253190e6c24341012a688eb86a9b24395320b"
    ),
    GSE200304_DEC019_SPLIT_CONFIG_PATH: (
        "fb8450e384c68d97413e89b11645fd81af1518f62a7fd18dc7c4aca1063fa45e"
    ),
    GSE200304_DEC019_SPLIT_SCRIPT_PATH: (
        "f8e0705576487edcfe724fa466c6990ee77d0b2946b6607e1cb87f08aa6fcda6"
    ),
    GSE200304_DEC019_SPLIT_TEST_PATH: (
        "2068ce3dcde68b68b9f8cd99c1d4b39165ff8beb735a30cd9ba7c232d528c27d"
    ),
    GSE200304_DEC019_POWER_CONFIG_PATH: (
        "5daf682d0e5688968d51873dbd99886b34e47d011bdff9eb5b33baa496b40a42"
    ),
    GSE200304_DEC019_POWER_SCRIPT_PATH: (
        "e312e4393daf67e18029ef55b454c54f2ac4c08806a967cc3025187beacd469f"
    ),
    GSE200304_DEC019_POWER_TEST_PATH: (
        "e41f7b7a516dfb8d9a93530b44229d4da5e4ed6feb007c6e3564581528da5171"
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
GSE200304_UPSTREAM_AUTHORITY_LINEAGE_ID = (
    "gse200304_upstream_authority_viability_v1"
)
GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_LINEAGE_ID = (
    "gse200304_dec019_upstream_pass_gate_pack_v1"
)
GSE200304_DEC019_UPSTREAM_PASS_ADJUDICATION_LINEAGE_ID = (
    "gse200304_dec019_reported_endpoint_a1_adjudication_v3_upstream_pass_gate_pack_v1"
)
GSE200304_DEC019_GROUP_LINEAGE_ID = (
    "gse200304_dec019_biological_group_authority_gate_v1"
)
GSE200304_DEC019_SPLIT_LINEAGE_ID = (
    "gse200304_dec019_outcome_blind_split_leakage_gate_v1"
)
GSE200304_DEC019_POWER_LINEAGE_ID = (
    "gse200304_dec019_prefrozen_power_precision_gate_v1"
)
GSE200304_DEC019_ONE_BLOCKER_ADJUDICATION_LINEAGE_ID = (
    "gse200304_dec019_reported_endpoint_a1_adjudication_v3_group_split_power_pass_d6"
)
GSE200304_CHECKPOINT_EXPOSURE_FAIL_LINEAGE_ID = (
    "gse200304_dec019_checkpoint_exposure_fail_current_protocol_v1"
)
GSE149487_PUBLIC_ASSET_ACQUISITION_LINEAGE_ID = (
    "gse149487_public_asset_acquisition_v1"
)
GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_LINEAGE_ID = (
    "gse217518_public_authority_preflight_v1"
)
GSE232572_PUBLIC_RECOVERY_AUDIT_LINEAGE_ID = (
    "gse232572_public_recovery_audit_v1"
)
GSE232572_DEVELOPMENT_V3_MATERIALIZATION_FAILURE_LINEAGE_ID = (
    "gse232572_development_v3_materialization_attempt_001_failure"
)
GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LINEAGE_ID = (
    "gse232572_development_v3_materialization_v1"
)
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_LINEAGE_ID = (
    "gse232572_a1_qualification_authority_preflight_v1"
)
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_PASSES = [
    "AUTHORIZED_PRIMARY_MEASUREMENT_AND_PAPER_FAITHFUL_LNFC",
    "PUBLIC_RAW_LOCATOR_AND_LINEAGE",
    "SOURCE_CANDIDATE_CONTEXT_GROUP_RECONSTRUCTION",
]
GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_BLOCKERS = {
    "DATASET_SPECIFIC_QUALIFICATION_AUTHORITY_AND_CONSUMER_USER_APPROVAL": "MISSING_EXTERNAL_AUTHORITY",
    "OWNER_PROJECT_USE_AND_EXPOSURE_ATTESTATION": "MISSING_EXTERNAL_AUTHORITY",
    "CHECKPOINT_SPECIFIC_EXACT_AND_NEAR_EXPOSURE_AUDIT": "UNKNOWN_NOT_ASSERTED",
    "ASSET_LEVEL_QUALIFICATION_USE_RIGHTS_AND_RELEASE_SCOPE": "UNKNOWN_BLOCKED",
    "EXECUTABLE_BIOLOGICAL_SOURCE_GROUP_AUTHORITY": "NOT_CLOSED",
    "REPLICATE_OR_VALID_STANDARD_ERROR": "NOT_CLOSED",
    "ELIGIBLE_MULTI_CANDIDATE_POOLS": "UNKNOWN_NOT_ASSERTED",
    "A1_GROUP_SPLIT_NEAR_DUPLICATE_GRAPH_SALT_AND_ZERO_LEAKAGE": "NOT_RUN",
    "PREFROZEN_GROUP_EFFECTIVE_N_POWER_AND_FULL_CI_WIDTH": "NOT_RUN",
    "REQUIRED_QUALIFICATION_REPORT_FIELDS": "INCOMPLETE",
    "V3_CANONICAL_ADMISSION": "BLOCKED",
    "FINAL_CANONICAL_QUALIFIER_AND_ADJUDICATION": "NOT_RUN",
}
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
GSE200304_DEC019_UPSTREAM_PASS_BLOCKERS = [
    "BIOLOGICAL_GROUP_AUTHORITY_NOT_PASS",
    "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS",
    "OUTCOME_BLIND_SPLIT_LEAKAGE_NOT_PASS",
    "PREFROZEN_POWER_PRECISION_NOT_PASS",
]
GSE200304_DEC019_UPSTREAM_PASS_INPUT_STATUS_COUNTS = {
    "PASS": 4,
    "BLOCKED": 1,
    "UNKNOWN_NOT_ASSERTED": 1,
    "NOT_RUN": 2,
}
GSE200304_DEC019_ONE_BLOCKER_BLOCKERS = [
    "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS",
]
GSE200304_DEC019_ONE_BLOCKER_INPUT_STATUS_COUNTS = {
    "PASS": 7,
    "BLOCKED": 0,
    "UNKNOWN_NOT_ASSERTED": 1,
    "NOT_RUN": 0,
}
GSE200304_DEC019_ONE_BLOCKER_PASS_SLOT_IDS = [0, 1, 2, 3, 5, 6, 7]
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

EXPECTED_DECISION_IDS = tuple(f"V3-DEC-{index:03d}" for index in range(1, 25)) + ("V3-DEC-027", "V3-DEC-028")
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
    "V3-DEC-020": "gse200304_model_input_route_and_route_conditional_exposure",
    "V3-DEC-021": "gse256185_public_identifier_and_pool_geometry_preflight_scope",
    "V3-DEC-022": "gse256185_aggregate_row_level_qualification_preflight_scope",
    "V3-DEC-023": "gse261709_schema_geometry_and_gse207584_dense_family_dual_aggregate_only_preflight_scope",
    "V3-DEC-024": "gse261709_processed_a1_gse269595_role_adjudication_and_emtab10902_true_a2_replacement_preflight_scope",
    "V3-DEC-027": "bounded_six_route_data_rescue_sprint_and_conditional_claim_ladder_trigger",
    "V3-DEC-028": "owner_initiated_single_study_source_relative_operational_mainline",
}

# Canonical per-entry digests make the accepted prefix genuinely append-only.
# A future DEC-022 requires an explicit validator update; rewriting any accepted
# DEC-001..021 entry while merely refreshing the registry manifest is rejected.
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
    "V3-DEC-020": "15e31d6abc30f221659ded9ba734d9d90959d48ba05f02d03864814bc6ded75b",
    "V3-DEC-021": "9c033bf9343c13cd05e1615fdca6f08757a72759006ebad7b95452d1e2c73471",
    "V3-DEC-022": "8ef7609ce268520fab63a0a4c662e04104d1e5fb125d67314135444aaf0cfd7c",
    "V3-DEC-023": "fa777bec43eddc1269644aed2b23be09c3b33c870717101e7d00d7eb3c8d098b",
    "V3-DEC-024": "c71101c5edc6c2f3a8dcef537dfffc035c69439717eba5062edfaa398550caf9",
    "V3-DEC-027": "10945f65bddf732b8a5f754d54f74c607ccb9cd2b088848801380ed06405ad39",
    "V3-DEC-028": "7511cf322b94eb0854a9378ac48e0bfbba5022f9ea79730e0357e2a43acea363",
}

EXPECTED_REGISTRY_MANIFEST_PATH_ROLES = (
    (GOAL_PATH, "ACTIVE_AMENDED_CONTRACT"),
    (CONFIG_PATH, "EXECUTABLE_CONTRACT"),
    (DEC019_AMENDMENT_PATH, "DEC019_APPEND_ONLY_AUTHORITY_AMENDMENT"),
    (DEC020_AMENDMENT_PATH, "DEC020_APPEND_ONLY_AUTHORITY_AMENDMENT"),
    (DEC021_AMENDMENT_PATH, "DEC021_APPEND_ONLY_AUTHORITY_AMENDMENT"),
    (DEC022_AMENDMENT_PATH, "DEC022_APPEND_ONLY_AUTHORITY_AMENDMENT"),
    (DEC023_AMENDMENT_PATH, "DEC023_APPEND_ONLY_DUAL_PREFLIGHT_AUTHORITY_AMENDMENT"),
    (DEC024_AMENDMENT_PATH, "DEC024_APPEND_ONLY_REPLACEMENT_PREFLIGHT_AUTHORITY_AMENDMENT"),
    (DEC027_AMENDMENT_PATH, "DEC027_APPEND_ONLY_BOUNDED_RESCUE_SPRINT_AUTHORITY_AMENDMENT"),
    (DEC028_AMENDMENT_PATH, "DEC028_APPEND_ONLY_SINGLE_STUDY_MAINLINE_AUTHORITY_AMENDMENT"),
    (DEC028_HUMAN_CONTRACT_PATH, "DEC028_REVIEWED_HUMAN_READABLE_SINGLE_STUDY_CONTRACT"),
    (DEC028_PROTOCOL_PATH, "DEC028_SINGLE_STUDY_MAINLINE_EXECUTABLE_PROTOCOL"),
    (DEC028_EXECUTION_PATH, "DEC028_SINGLE_STUDY_MAINLINE_EXECUTION_STATE"),
    (A1_QUALIFICATION_CONFIG_PATH, "A1_QUALIFICATION_ROOT_PROTOCOL"),
    (SUPERSESSION_PATH, "SUPERSESSION_LINEAGE"),
    (SCIENTIFIC_M0_HISTORY_PATH, "HISTORICAL_M0_SCIENTIFIC_FAILURE_EXACT_COPY"),
    (REGISTRY_PATHS["baseline"], "BASELINE_REGISTRY"),
    (REGISTRY_PATHS["claim"], "CLAIM_EVIDENCE_MATRIX"),
    (REGISTRY_PATHS["data"], "DATA_ROLE_REGISTRY"),
    (DECISION_LOG_PATH, "DECISION_AND_AMENDMENT_LOG"),
    (A1_INTERIM_PATH, "A1_ACTIVE_INTERIM_RECORD"),
    (A6_INTERIM_PATH, "A6_ACTIVE_INTERIM_RECORD"),
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
    (
        GSE200304_UPSTREAM_AUTHORITY_CONFIG_PATH,
        "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_PROTOCOL",
    ),
    (
        GSE200304_UPSTREAM_AUTHORITY_SCRIPT_PATH,
        "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_PRODUCER",
    ),
    (
        GSE200304_UPSTREAM_AUTHORITY_TEST_PATH,
        "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_FOCUSED_TEST",
    ),
    (
        GSE200304_DEC019_UPSTREAM_PASS_CONFIG_PATH,
        "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_PROTOCOL",
    ),
    (
        GSE200304_DEC019_UPSTREAM_PASS_SCRIPT_PATH,
        "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_PRODUCER",
    ),
    (
        GSE200304_DEC019_UPSTREAM_PASS_TEST_PATH,
        "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_FOCUSED_TEST",
    ),
    (
        GSE200304_DEC019_GROUP_CONFIG_PATH,
        "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE_PROTOCOL",
    ),
    (
        GSE200304_DEC019_GROUP_SCRIPT_PATH,
        "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE_PRODUCER",
    ),
    (
        GSE200304_DEC019_GROUP_TEST_PATH,
        "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE_FOCUSED_TEST",
    ),
    (
        GSE200304_DEC019_SPLIT_CONFIG_PATH,
        "GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE_PROTOCOL",
    ),
    (
        GSE200304_DEC019_SPLIT_SCRIPT_PATH,
        "GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE_PRODUCER",
    ),
    (
        GSE200304_DEC019_SPLIT_TEST_PATH,
        "GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE_FOCUSED_TEST",
    ),
    (
        GSE200304_DEC019_POWER_CONFIG_PATH,
        "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE_PROTOCOL",
    ),
    (
        GSE200304_DEC019_POWER_SCRIPT_PATH,
        "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE_PRODUCER",
    ),
    (
        GSE200304_DEC019_POWER_TEST_PATH,
        "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE_FOCUSED_TEST",
    ),
    (
        GSE200304_CHECKPOINT_EXPOSURE_FAIL_CONFIG_PATH,
        "GSE200304_CHECKPOINT_EXPOSURE_FAIL_CURRENT_PROTOCOL_CONFIG",
    ),
    (
        GSE200304_CHECKPOINT_EXPOSURE_FAIL_SCRIPT_PATH,
        "GSE200304_CHECKPOINT_EXPOSURE_FAIL_CURRENT_PROTOCOL_PRODUCER",
    ),
    (
        GSE200304_CHECKPOINT_EXPOSURE_FAIL_TEST_PATH,
        "GSE200304_CHECKPOINT_EXPOSURE_FAIL_CURRENT_PROTOCOL_FOCUSED_TEST",
    ),
    (
        GSE149487_PUBLIC_ASSET_ACQUISITION_CONFIG_PATH,
        "GSE149487_PUBLIC_ASSET_ACQUISITION_CONFIG",
    ),
    (
        GSE149487_PUBLIC_ASSET_ACQUISITION_SCRIPT_PATH,
        "GSE149487_PUBLIC_ASSET_ACQUISITION_PRODUCER",
    ),
    (
        GSE149487_PUBLIC_ASSET_ACQUISITION_TEST_PATH,
        "GSE149487_PUBLIC_ASSET_ACQUISITION_FOCUSED_TEST",
    ),
    (
        GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_CONFIG_PATH,
        "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_CONFIG",
    ),
    (
        GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_SCRIPT_PATH,
        "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_PRODUCER",
    ),
    (
        GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_TEST_PATH,
        "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_FOCUSED_TEST",
    ),
    (
        GSE232572_PUBLIC_RECOVERY_AUDIT_CONFIG_PATH,
        "GSE232572_PUBLIC_RECOVERY_AUDIT_CONFIG",
    ),
    (
        GSE232572_PUBLIC_RECOVERY_AUDIT_SCRIPT_PATH,
        "GSE232572_PUBLIC_RECOVERY_AUDIT_PRODUCER",
    ),
    (
        GSE232572_PUBLIC_RECOVERY_AUDIT_TEST_PATH,
        "GSE232572_PUBLIC_RECOVERY_AUDIT_FOCUSED_TEST",
    ),
    (
        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_CONFIG_PATH,
        "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_CONFIG",
    ),
    (
        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_SCRIPT_PATH,
        "GSE232572_DEVELOPMENT_V3_MATERIALIZER",
    ),
    (
        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_TEST_PATH,
        "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_FOCUSED_TEST",
    ),
    (
        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_CONFIG_PATH,
        "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_CONFIG",
    ),
    (
        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_SCRIPT_PATH,
        "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_PRODUCER",
    ),
    (
        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_TEST_PATH,
        "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_FOCUSED_TEST",
    ),
    (GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH, "GSE200304_DEC020_SCRATCH_ROUTE_V4_CONFIG"),
    (GSE200304_DEC020_V4_SCRIPT_PATH, "GSE200304_DEC020_SCRATCH_ROUTE_V4_ADJUDICATOR"),
    (GSE200304_DEC020_V4_TEST_PATH, "GSE200304_DEC020_SCRATCH_ROUTE_V4_FOCUSED_TEST"),
    (
        GSE256185_PUBLIC_GEOMETRY_CONFIG_PATH,
        "GSE256185_PUBLIC_IDENTIFIER_POOL_GEOMETRY_PREFLIGHT_BOUND_CONFIG",
    ),
    (
        GSE256185_PUBLIC_GEOMETRY_SCRIPT_PATH,
        "GSE256185_PUBLIC_IDENTIFIER_POOL_GEOMETRY_PREFLIGHT_PRODUCER",
    ),
    (
        GSE256185_PUBLIC_GEOMETRY_TEST_PATH,
        "GSE256185_PUBLIC_IDENTIFIER_POOL_GEOMETRY_PREFLIGHT_FOCUSED_TEST",
    ),
    (
        GSE256185_ROW_PREFLIGHT_CONFIG_PATH,
        "GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_BOUND_CONFIG",
    ),
    (
        GSE256185_ROW_PREFLIGHT_SCRIPT_PATH,
        "GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_PRODUCER",
    ),
    (
        GSE256185_ROW_PREFLIGHT_TEST_PATH,
        "GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_FOCUSED_TEST",
    ),
    (
        GSE261709_PREFLIGHT_CONFIG_PATH,
        "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_PREFLIGHT_BOUND_CONFIG",
    ),
    (
        GSE261709_PREFLIGHT_SCRIPT_PATH,
        "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_PREFLIGHT_PRODUCER",
    ),
    (
        GSE261709_PREFLIGHT_TEST_PATH,
        "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_PREFLIGHT_FOCUSED_TEST",
    ),
    (
        GSE207584_PREFLIGHT_CONFIG_PATH,
        "GSE207584_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_BOUND_CONFIG",
    ),
    (
        GSE207584_PREFLIGHT_SCRIPT_PATH,
        "GSE207584_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_PRODUCER",
    ),
    (
        GSE207584_PREFLIGHT_TEST_PATH,
        "GSE207584_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_FOCUSED_TEST",
    ),
    (
        "configs/route_a_v3_gse217518_corrected_a1_successor_candidate_v1.json",
        "DEC027_GSE217518_CORRECTED_A1_SUCCESSOR_BOUND_CONFIG",
    ),
    (
        "scripts/route_a_v3/preflight_gse217518_corrected_a1_successor_candidate.py",
        "DEC027_GSE217518_CORRECTED_A1_SUCCESSOR_PRODUCER",
    ),
    (
        "tests/route_a_v3/test_preflight_gse217518_corrected_a1_successor_candidate.py",
        "DEC027_GSE217518_CORRECTED_A1_SUCCESSOR_FOCUSED_TEST",
    ),
    (
        "configs/route_a_v3_encsr854ruf_dec027_dataset_specific_a1_preflight_v1.json",
        "DEC027_ENCSR854RUF_DATASET_SPECIFIC_A1_PREFLIGHT_BOUND_CONFIG",
    ),
    (
        "scripts/route_a_v3/preflight_encsr854ruf_dec027_dataset_specific_a1.py",
        "DEC027_ENCSR854RUF_DATASET_SPECIFIC_A1_PREFLIGHT_PRODUCER",
    ),
    (
        "tests/route_a_v3/test_preflight_encsr854ruf_dec027_dataset_specific_a1.py",
        "DEC027_ENCSR854RUF_DATASET_SPECIFIC_A1_PREFLIGHT_FOCUSED_TEST",
    ),
    (
        "configs/route_a_v3_gse232572_corrected_a1_replay_v1.json",
        "DEC027_GSE232572_CORRECTED_A1_REPLAY_BOUND_CONFIG",
    ),
    (
        "scripts/route_a_v3/replay_gse232572_corrected_a1.py",
        "DEC027_GSE232572_CORRECTED_A1_REPLAY_PRODUCER",
    ),
    (
        "tests/route_a_v3/test_replay_gse232572_corrected_a1.py",
        "DEC027_GSE232572_CORRECTED_A1_REPLAY_FOCUSED_TEST",
    ),
    (
        "configs/route_a_v3_gse113849_designed_snv_true_a2_preflight_v1.json",
        "DEC027_GSE113849_DESIGNED_SNV_TRUE_A2_PREFLIGHT_BOUND_CONFIG",
    ),
    (
        "scripts/route_a_v3/preflight_gse113849_designed_snv_true_a2.py",
        "DEC027_GSE113849_DESIGNED_SNV_TRUE_A2_PREFLIGHT_PRODUCER",
    ),
    (
        "tests/route_a_v3/test_preflight_gse113849_designed_snv_true_a2.py",
        "DEC027_GSE113849_DESIGNED_SNV_TRUE_A2_PREFLIGHT_FOCUSED_TEST",
    ),
    (
        "configs/route_a_v3_gse269595_corrected_role_adjudication_successor_candidate_v1.json",
        "DEC027_GSE269595_CORRECTED_ROLE_ADJUDICATION_BOUND_CONFIG",
    ),
    (
        "scripts/route_a_v3/preflight_gse269595_corrected_role_adjudication_successor_candidate.py",
        "DEC027_GSE269595_CORRECTED_ROLE_ADJUDICATION_PRODUCER",
    ),
    (
        "tests/route_a_v3/test_preflight_gse269595_corrected_role_adjudication_successor_candidate.py",
        "DEC027_GSE269595_CORRECTED_ROLE_ADJUDICATION_FOCUSED_TEST",
    ),
    (
        "configs/route_a_v3_gse295080_independence_overlap_adjudication_v1.json",
        "DEC027_GSE295080_INDEPENDENCE_OVERLAP_BOUND_CONFIG",
    ),
    (
        "scripts/route_a_v3/preflight_gse295080_independence_overlap_adjudication.py",
        "DEC027_GSE295080_INDEPENDENCE_OVERLAP_PRODUCER",
    ),
    (
        "tests/route_a_v3/test_preflight_gse295080_independence_overlap_adjudication.py",
        "DEC027_GSE295080_INDEPENDENCE_OVERLAP_FOCUSED_TEST",
    ),
    (A6_PROTOCOL_CONFIG_PATH, "A6_CPU_EXACT_ABSORBING_DAG_BOUND_PROTOCOL"),
    (A6_PRODUCER_PATH, "A6_CPU_EXACT_ABSORBING_DAG_PRODUCER"),
    (A6_FOCUSED_TEST_PATH, "A6_CPU_EXACT_ABSORBING_DAG_FOCUSED_TEST"),
    (A6_GILLESPIE_CONFIG_PATH, "A6_CPU_LEGAL_CTMC_PARTIAL_BOUND_PROTOCOL"),
    (A6_GILLESPIE_PRODUCER_PATH, "A6_CPU_LEGAL_CTMC_PARTIAL_PRODUCER"),
    (A6_GILLESPIE_FOCUSED_TEST_PATH, "A6_CPU_LEGAL_CTMC_PARTIAL_FOCUSED_TEST"),
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
        DEC020_AMENDMENT_PATH,
        DEC021_AMENDMENT_PATH,
        DEC022_AMENDMENT_PATH,
        DECISION_LOG_PATH,
        REGISTRY_MANIFEST_PATH,
        SCIENTIFIC_M0_HISTORY_PATH,
        SEALED_GUARD_PATH,
        SEALED_RUNNER_PATH,
        A1_INTERIM_PATH,
        A6_INTERIM_PATH,
        *A6_STATIC_PRODUCER_LEAF_SHA256,
        *A6_GILLESPIE_STATIC_PRODUCER_LEAF_SHA256,
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
        *POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256,
        *GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256,
        *GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF_SHA256,
        *GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF_SHA256,
        *GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256,
        *GSE200304_DEC020_V4_STATIC_LEAF_SHA256,
        *GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF_SHA256,
        *GSE256185_ROW_PREFLIGHT_STATIC_LEAF_SHA256,
        *GSE261709_PREFLIGHT_STATIC_LEAF_SHA256,
        *GSE207584_PREFLIGHT_STATIC_LEAF_SHA256,
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
            "authority.active_amendment_decision_ids must preserve DEC019-DEC024 and append DEC027",
        )
    elif not _json_type_strict_equal(
        authority.get("active_amendment_paths"),
        ACTIVE_AMENDMENT_PATHS,
    ):
        _issue(
            issues,
            "AUTHORITY_ACTIVE_AMENDMENT_PATHS",
            CONFIG_PATH,
            "authority.active_amendment_paths must bind the exact DEC019-DEC024 prefix and DEC027 successor",
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
            "active contract amendment lineage must preserve DEC019-DEC024 and append DEC027",
        )
    if not _json_type_strict_equal(
        supersession.get("active_contract_amendment_paths"),
        ACTIVE_AMENDMENT_PATHS,
    ):
        _issue(
            issues,
            "SUPERSESSION_ACTIVE_AMENDMENT_PATHS",
            SUPERSESSION_PATH,
            "active contract amendment paths must bind the exact DEC019-DEC024 prefix and DEC027 successor",
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
                "new_authority.active_amendment_decision_ids must preserve DEC019-DEC024 and append DEC027",
            )
        if not _json_type_strict_equal(
            new_authority.get("active_amendment_paths"),
            ACTIVE_AMENDMENT_PATHS,
        ):
            _issue(
                issues,
                "SUPERSESSION_NEW_AUTHORITY",
                SUPERSESSION_PATH,
                "new_authority.active_amendment_paths must bind the exact DEC019-DEC024 prefix and DEC027 successor",
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
        "manifest_status": DEC028_AUTHORITY_MANIFEST_STATUS,
        "initial_generated_at": "2026-08-10T10:10:05+08:00",
        "generated_at": DEC028_AUTHORITY_MANIFEST_AT,
        "updated_at": DEC028_AUTHORITY_MANIFEST_AT,
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
    expected_dec020_registered_paths = (
        set(DEC020_AUTHORITY_COMMIT_EXACT_CHANGED_PATHS)
        - {REGISTRY_MANIFEST_PATH}
    )
    if not expected_dec020_registered_paths.issubset(by_path):
        _issue(
            issues,
            "DEC020_AUTHORITY_COMMIT_CLOSURE",
            REGISTRY_MANIFEST_PATH,
            "authority commit A must register all exact14 changed paths except "
            "the intentionally non-self-hashed registry manifest",
        )
    expected_dec021_registered_paths = set(DEC021_AUTHORITY_EXACT_CHANGED_PATHS) - {
        REGISTRY_MANIFEST_PATH
    }
    if not expected_dec021_registered_paths.issubset(by_path):
        _issue(
            issues,
            "DEC021_AUTHORITY_EXACT10_CLOSURE",
            REGISTRY_MANIFEST_PATH,
            "DEC021 exact10 must register all changed paths except the intentionally non-self-hashed registry manifest",
        )
    expected_dec022_registered_paths = set(DEC022_AUTHORITY_EXACT_CHANGED_PATHS) - {
        REGISTRY_MANIFEST_PATH
    }
    if not expected_dec022_registered_paths.issubset(by_path):
        _issue(
            issues,
            "DEC022_AUTHORITY_EXACT10_CLOSURE",
            REGISTRY_MANIFEST_PATH,
            "DEC022 exact10 must register all changed paths except the intentionally non-self-hashed registry manifest",
        )
    expected_dec023_registered_paths = set(DEC023_AUTHORITY_EXACT_CHANGED_PATHS) - {
        REGISTRY_MANIFEST_PATH
    }
    if not expected_dec023_registered_paths.issubset(by_path):
        _issue(
            issues,
            "DEC023_AUTHORITY_EXACT10_CLOSURE",
            REGISTRY_MANIFEST_PATH,
            "DEC023 exact10 must register all changed paths except the intentionally non-self-hashed registry manifest",
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
    observed_v4_paths = {
        relative for relative in by_path if relative in GSE200304_DEC020_V4_STATIC_LEAF_SHA256
    }
    if observed_v4_paths != set(GSE200304_DEC020_V4_STATIC_LEAF_SHA256):
        _issue(
            issues,
            "DEC020_V4_STATIC_REGISTRATION",
            REGISTRY_MANIFEST_PATH,
            "DEC020 V4 CONFIG/SCRIPT/FOCUSED_TEST leaves must be registered exactly once",
        )
    observed_gse256185_paths = set(by_path) & set(
        GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF_SHA256
    )
    if observed_gse256185_paths != set(
        GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF_SHA256
    ):
        _issue(
            issues,
            "GSE256185_PUBLIC_GEOMETRY_STATIC_REGISTRATION",
            REGISTRY_MANIFEST_PATH,
            "the bound preflight CONFIG/SCRIPT/FOCUSED_TEST leaves must appear exactly once",
        )
    if GSE256185_PUBLIC_GEOMETRY_RUNTIME_CONFIG_PATH in by_path:
        _issue(
            issues,
            "GSE256185_PUBLIC_GEOMETRY_RUNTIME_CONFIG_CYCLE",
            REGISTRY_MANIFEST_PATH,
            "a future dynamic evidence-runtime config must not enter the static manifest",
        )
    expected_a6_paths = {
        A6_INTERIM_PATH,
        *A6_STATIC_PRODUCER_LEAF_SHA256,
        *A6_GILLESPIE_STATIC_PRODUCER_LEAF_SHA256,
    }
    observed_a6_paths = set(by_path) & expected_a6_paths
    if observed_a6_paths != expected_a6_paths:
        _issue(
            issues,
            "A6_CPU_PARTIAL_STATIC_REGISTRATION",
            REGISTRY_MANIFEST_PATH,
            "A6 interim plus both registered CONFIG/SCRIPT/FOCUSED_TEST trios must appear exactly once",
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
        _issue(issues, "DECISION_LOG_ID_CLOSURE", DECISION_LOG_PATH, "decision IDs must preserve V3-DEC-001 through V3-DEC-024 and append V3-DEC-027")
    ordered_ids = [entry.get("decision_id") if isinstance(entry, Mapping) else None for entry in raw]
    if ordered_ids != list(EXPECTED_DECISION_IDS):
        _issue(issues, "DECISION_LOG_ORDER", DECISION_LOG_PATH, "accepted DEC-001 through DEC-024 prefix must remain exact and DEC-027 must be appended")
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
        "V3-DEC-020": (
            "SCRATCH_ONLY_NO_FOUNDATION_NO_EXTERNAL_LEARNED_INPUTS",
            "NOT_APPLICABLE_BY_FROZEN_NO_EXTERNAL_LEARNED_INPUT_ROUTE",
            "not PASS",
            "Retain FOUNDATION_CHECKPOINT",
            "No post-result route switch",
            "3/2/1 minima",
            "maximum 1/1/0 contribution",
            "rights gate remains PASS",
            "not untouched",
            "prior-analytic-use attestation remains required",
            "changes no current qualified count",
            "is not training evidence",
            "establishes no scientific claim",
        ),
        "V3-DEC-022": (
            "634 strict single-parent pools",
            "7292 strict candidate",
            "15 dual-parent",
            "two nonstrict grammar records",
            "reasoned 7294-candidate",
            "17 independently adjudicated fail-closed axes",
            "source-candidate identity",
            "edit replay",
            "independent biological replicate and valid SE",
            "outcome-blind near-duplicate split and zero leakage",
            "beneficial signal versus noise",
            "post-dedup effective N",
            "reject closure",
            "Do not output member IDs",
            "Do not presume source-to-candidate edits",
            "A1-EVT-053",
            "global 1/1/0 with 6547 canonical records",
            "GSE256185 remains 0/0/0 with zero canonical records",
            "pending a fresh, unallocated event ID",
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
    dec020 = decisions.get("V3-DEC-020")
    if isinstance(dec020, Mapping):
        expected_dec020_fields = {
            "decision_type": "AMENDMENT",
            "dimension": "gse200304_model_input_route_and_route_conditional_exposure",
            "status": "FROZEN_USER_AUTHORIZED_GSE200304_MODEL_INPUT_ROUTE_POLICY",
            "effective_phase": "A1",
            "requires_user_authorization": True,
            "user_authorization_status": "GRANTED",
            "user_authorization_received_at": "2026-08-13T12:07:11+08:00",
            "user_authorization_source": "ACTIVE_CODEX_THREAD_OWNER_CONFIRMATION",
            "preserves_decision_ids": list(EXPECTED_DECISION_IDS[:19]),
            "accepted_prefix_preserved_through": "V3-DEC-019",
            "amendment_path": DEC020_AMENDMENT_PATH,
            "selected_model_input_route": "SCRATCH_ONLY_NO_FOUNDATION_NO_EXTERNAL_LEARNED_INPUTS",
            "retained_model_input_route": "FOUNDATION_CHECKPOINT",
            "current_qualified_counts": {
                "ordinary": 0,
                "a1": 0,
                "true_a2": 0,
                "canonical_records": 0,
            },
            "phase_complete": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "decision_or_policy_alone_is_training_evidence": False,
            "existing_license_and_redistribution_rights_gate_status": "PASS",
            "rights_gate_reopened_by_dec020": False,
            "future_route_scoped_qualification_and_canonical_count_possible_after_all_gates_pass": True,
            "canonical_materialization_execution_authorized": False,
            "private_payload_access_allowed": False,
            "full_prior_analytic_use_attestation_required_before_predictor_protocol_promotion_or_training": True,
            "full_prior_analytic_use_attestation_completion_asserted_by_dec020": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "sealed_contact": False,
        }
        for key, value in expected_dec020_fields.items():
            if not _json_type_strict_equal(dec020.get(key), value):
                _issue(
                    issues,
                    "DECISION_LOG_DEC020",
                    DECISION_LOG_PATH,
                    f"V3-DEC-020.{key} must remain {value!r}",
                )
        expected_refs = {
            GOAL_PATH,
            DEC019_AMENDMENT_PATH,
            DEC020_AMENDMENT_PATH,
            CONFIG_PATH,
            A1_QUALIFICATION_CONFIG_PATH,
            REGISTRY_PATHS["data"],
            REGISTRY_PATHS["task"],
            REGISTRY_PATHS["claim"],
            REGISTRY_PATHS["split"],
            REGISTRY_PATHS["matrix"],
        }
        if not isinstance(dec020.get("evidence_refs"), list) or set(dec020["evidence_refs"]) != expected_refs:
            _issue(
                issues,
                "DECISION_LOG_DEC020_EVIDENCE",
                DECISION_LOG_PATH,
                f"V3-DEC-020 evidence_refs must be exactly {sorted(expected_refs)!r}",
            )
    dec021 = decisions.get("V3-DEC-021")
    if isinstance(dec021, Mapping):
        expected_dec021_fields = {
            "decision_type": "AMENDMENT",
            "dimension": "gse256185_public_identifier_and_pool_geometry_preflight_scope",
            "status": "FROZEN_USER_AUTHORIZED_GSE256185_PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
            "effective_phase": "A1",
            "requires_user_authorization": True,
            "user_authorization_status": "GRANTED",
            "user_authorization_received_at": "2026-08-13T19:45:00+08:00",
            "user_authorization_source": "ACTIVE_CODEX_THREAD_OWNER_CONFIRMATION",
            "preserves_decision_ids": list(EXPECTED_DECISION_IDS[:20]),
            "accepted_prefix_preserved_through": "V3-DEC-020",
            "amendment_path": DEC021_AMENDMENT_PATH,
            "dataset_id": "GSE256185",
            "preflight_role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
            "preflight_candidate_only_not_counting": True,
            "allowed_input_field_classes_exactly": ["IDENTIFIER", "ROLE", "CONTEXT"],
            "allowed_output_class": "AGGREGATE_POOL_GEOMETRY_ONLY",
            "sequence_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
            "edit_budget_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
            "effect_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
            "true_a2_status_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
            "qualification_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
            "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
            "changes_current_qualified_counts": False,
            "gse256185_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
            "phase_complete": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "qualifier_execution_allowed": False,
            "canonical_materialization_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "latest_settled_runtime_event_id": "A1-EVT-051",
            "settled_runtime_event_changed": False,
            "sealed_contact": False,
        }
        for key, value in expected_dec021_fields.items():
            if not _json_type_strict_equal(dec021.get(key), value):
                _issue(
                    issues,
                    "DECISION_LOG_DEC021",
                    DECISION_LOG_PATH,
                    f"V3-DEC-021.{key} must remain {value!r}",
                )
        expected_refs = {
            GOAL_PATH,
            DEC020_AMENDMENT_PATH,
            DEC021_AMENDMENT_PATH,
            CONFIG_PATH,
            A1_QUALIFICATION_CONFIG_PATH,
            REGISTRY_PATHS["data"],
        }
        if not isinstance(dec021.get("evidence_refs"), list) or set(dec021["evidence_refs"]) != expected_refs:
            _issue(
                issues,
                "DECISION_LOG_DEC021_EVIDENCE",
                DECISION_LOG_PATH,
                f"V3-DEC-021 evidence_refs must be exactly {sorted(expected_refs)!r}",
            )
    dec022 = decisions.get("V3-DEC-022")
    if isinstance(dec022, Mapping):
        expected_dec022_fields = {
            "decision_type": "AMENDMENT",
            "dimension": "gse256185_aggregate_row_level_qualification_preflight_scope",
            "status": "FROZEN_USER_AUTHORIZED_GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
            "effective_phase": "A1",
            "requires_user_authorization": True,
            "user_authorization_status": "GRANTED",
            "user_authorization_received_at": "2026-08-13T22:15:00+08:00",
            "user_authorization_source": "ACTIVE_CODEX_THREAD_OWNER_CONFIRMATION",
            "preserves_decision_ids": list(EXPECTED_DECISION_IDS[:21]),
            "accepted_prefix_preserved_through": "V3-DEC-021",
            "amendment_path": DEC022_AMENDMENT_PATH,
            "dataset_id": "GSE256185",
            "preflight_role": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
            "strict_single_parent_pool_count": 634,
            "strict_candidate_member_count": 7292,
            "two_candidate_strict_single_parent_group_count_excluded": 3,
            "dual_parent_group_count_excluded": 15,
            "nonstrict_grammar_record_count_excluded": 2,
            "reasoned_family_closure_candidate_count": 7294,
            "reasoned_family_closure_included": False,
            "allowed_input_field_classes_exactly": ["IDENTIFIER", "ROLE", "SEQUENCE", "ENDPOINT", "REPLICATE", "NECESSARY_CONTEXT"],
            "allowed_output_class": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
            "independent_gate_axis_count": 17,
            "all_required_gates_initial_status": "NOT_RUN",
            "unknown_or_not_run_gate_is_pass": False,
            "source_to_candidate_edit_relation_may_be_presumed": False,
            "row_or_member_output_allowed": False,
            "sequence_output_allowed": False,
            "row_effect_output_allowed": False,
            "replicate_identifier_output_allowed": False,
            "split_assignment_output_allowed": False,
            "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
            "changes_current_qualified_counts": False,
            "gse256185_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
            "gse256185_qualified": False,
            "phase_complete": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "qualifier_execution_allowed": False,
            "canonical_materialization_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "latest_settled_runtime_event_id": "A1-EVT-053",
            "settled_runtime_event_changed": False,
            "runtime_event_emitted": False,
            "runtime_sync_status": "PENDING_FRESH_EVENT_AFTER_SETTLED_EVT_053",
            "expected_next_runtime_event_id": DEC022_PENDING_RUNTIME_EVENT_ID,
            "next_runtime_event_id_preallocated": False,
            "sealed_contact": False,
        }
        for key, value in expected_dec022_fields.items():
            if not _json_type_strict_equal(dec022.get(key), value):
                _issue(
                    issues,
                    "DECISION_LOG_DEC022",
                    DECISION_LOG_PATH,
                    f"V3-DEC-022.{key} must remain {value!r}",
                )
        expected_refs = {
            GOAL_PATH,
            DEC021_AMENDMENT_PATH,
            DEC022_AMENDMENT_PATH,
            CONFIG_PATH,
            A1_QUALIFICATION_CONFIG_PATH,
            REGISTRY_PATHS["data"],
            A1_INTERIM_PATH,
        }
        if not isinstance(dec022.get("evidence_refs"), list) or set(dec022["evidence_refs"]) != expected_refs:
            _issue(
                issues,
                "DECISION_LOG_DEC022_EVIDENCE",
                DECISION_LOG_PATH,
                f"V3-DEC-022 evidence_refs must be exactly {sorted(expected_refs)!r}",
            )
    dec023 = decisions.get("V3-DEC-023")
    if isinstance(dec023, Mapping):
        expected_dec023_fields = {
            "decision_type": "AMENDMENT",
            "dimension": "gse261709_schema_geometry_and_gse207584_dense_family_dual_aggregate_only_preflight_scope",
            "status": "FROZEN_USER_AUTHORIZED_DUAL_AGGREGATE_ONLY_PREFLIGHT_NO_PROMOTION",
            "effective_phase": "A1",
            "requires_user_authorization": True,
            "user_authorization_status": "GRANTED",
            "user_authorization_received_at": "2026-08-14T10:05:00+08:00",
            "user_authorization_source": "ACTIVE_CODEX_THREAD_OWNER_CONFIRMATION",
            "preserves_decision_ids": list(EXPECTED_DECISION_IDS[:22]),
            "accepted_prefix_preserved_through": "V3-DEC-022",
            "amendment_path": DEC023_AMENDMENT_PATH,
            "gse261709_project_id": "PRJNA1088465",
            "gse261709_role": "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_PREFLIGHT_ONLY",
            "gse261709_member_or_body_read_count_required": 0,
            "gse261709_member_or_body_output_count_required": 0,
            "gse261709_actual_header_names_output_allowed": False,
            "gse261709_header_role_class_coverage_count_output_allowed": True,
            "gse261709_row_level_access_allowed": False,
            "gse207584_project_id": "PRJNA856272",
            "gse207584_role": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
            "gse207584_registry_role": "AUDIT_ONLY",
            "gse207584_independent_gate_axis_count": 11,
            "gse207584_all_required_gates_initial_status": "NOT_RUN",
            "gse207584_unknown_or_not_run_gate_is_pass": False,
            "gse207584_source_to_candidate_edit_relation_may_be_presumed": False,
            "gse207584_split_assignment_execution_allowed": False,
            "gse207584_aggregate_prefrozen_power_planning_calculation_allowed": True,
            "gse207584_aggregate_prefrozen_power_planning_alternative_spearman_rho": 0.25,
            "gse207584_aggregate_prefrozen_power_planning_method": "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN",
            "gse207584_aggregate_prefrozen_power_planning_confidence_interval_method": "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE",
            "gse207584_aggregate_prefrozen_power_planning_null_standard_error_formula": "1/sqrt(n-3)",
            "gse207584_aggregate_prefrozen_power_planning_alternative_standard_error_formula": "sqrt(1+rho^2/2)/sqrt(n-3)",
            "gse207584_aggregate_prefrozen_power_planning_working_distribution_assumption": "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO",
            "gse207584_aggregate_prefrozen_power_planning_alpha_two_sided": 0.05,
            "gse207584_aggregate_prefrozen_power_planning_target_power": 0.8,
            "gse207584_aggregate_prefrozen_power_planning_confidence_level": 0.95,
            "gse207584_aggregate_prefrozen_power_planning_maximum_full_ci_width": 0.3,
            "gse207584_aggregate_prefrozen_power_planning_required_effective_n_for_both_power_and_ci_width": 156,
            "gse207584_aggregate_prefrozen_power_planning_analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
            "gse207584_aggregate_prefrozen_power_planning_output_class": "AGGREGATE_ONLY",
            "gse207584_formal_qualification_power_gate_execution_allowed": False,
            "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
            "changes_current_qualified_counts": False,
            "gse261709_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
            "gse207584_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
            "gse261709_qualified": False,
            "gse207584_qualified": False,
            "qualifier_execution_allowed": False,
            "canonical_materialization_allowed": False,
            "split_execution_allowed": False,
            "formal_qualification_power_gate_execution_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "a7_allowed": False,
            "next_phase_authorized": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "latest_settled_runtime_event_id": "A1-EVT-055",
            "settled_runtime_event_changed": False,
            "runtime_event_emitted": False,
            "runtime_sync_status": "PENDING_FRESH_EVENT_AFTER_SETTLED_EVT_055",
            "expected_next_runtime_event_id": DEC023_PENDING_RUNTIME_EVENT_ID,
            "next_runtime_event_id_preallocated": False,
            "sealed_contact": False,
        }
        for key, value in expected_dec023_fields.items():
            if not _json_type_strict_equal(dec023.get(key), value):
                _issue(issues, "DECISION_LOG_DEC023", DECISION_LOG_PATH, f"V3-DEC-023.{key} must remain {value!r}")
        expected_refs = {
            GOAL_PATH,
            DEC022_AMENDMENT_PATH,
            DEC023_AMENDMENT_PATH,
            CONFIG_PATH,
            A1_QUALIFICATION_CONFIG_PATH,
            REGISTRY_PATHS["data"],
            A1_INTERIM_PATH,
        }
        if not isinstance(dec023.get("evidence_refs"), list) or set(dec023["evidence_refs"]) != expected_refs:
            _issue(issues, "DECISION_LOG_DEC023_EVIDENCE", DECISION_LOG_PATH, f"V3-DEC-023 evidence_refs must be exactly {sorted(expected_refs)!r}")
    dec024 = decisions.get("V3-DEC-024")
    if isinstance(dec024, Mapping):
        expected_dec024_fields = {
            "decision_type": "AMENDMENT",
            "dimension": "gse261709_processed_a1_gse269595_role_adjudication_and_emtab10902_true_a2_replacement_preflight_scope",
            "status": "FROZEN_USER_AUTHORIZED_REPLACEMENT_PREFLIGHT_ONLY_NO_PROMOTION",
            "effective_phase": "A1",
            "requires_user_authorization": True,
            "user_authorization_status": "GRANTED",
            "user_authorization_received_at": "2026-08-14T17:12:00+08:00",
            "user_authorization_source": "ACTIVE_CODEX_THREAD_OWNER_AUTONOMY_AND_REPLACEMENT_PREFLIGHT_DIRECTIVE",
            "preserves_decision_ids": list(EXPECTED_DECISION_IDS[:23]),
            "accepted_prefix_preserved_through": "V3-DEC-023",
            "amendment_path": DEC024_AMENDMENT_PATH,
            "gse261709_project_id": "PRJNA1088465",
            "gse261709_role": "AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_ONLY",
            "gse261709_authority_surface": "ORDINARY_PUBLIC_PROCESSED_ASSET_ONLY",
            "gse261709_raw_fastq_or_sra_member_payload_read_allowed": False,
            "gse261709_independent_gate_axis_count": 12,
            "gse269595_project_id": "PRJNA1122592",
            "gse269595_role": "REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_ONLY",
            "gse269595_maximum_roles_if_later_qualified": 1,
            "gse269595_a1_and_true_a2_double_credit_allowed": False,
            "gse269595_intronic_apa_exclusion_required": True,
            "gse269595_independent_gate_axis_count": 13,
            "emtab10902_alias": "N_ZIP",
            "emtab10902_role": "REPLACEMENT_TRUE_A2_CANDIDATE_PREFLIGHT_ONLY",
            "emtab10902_reported_source_group_count_approximate": 16,
            "emtab10902_reported_qc_design_row_count_reference_only": 5679,
            "emtab10902_reported_qc_design_rows_may_substitute_for_independent_n": False,
            "emtab10902_prefrozen_required_effective_n": 156,
            "emtab10902_power_infeasible_status_allowed": True,
            "future_use_route": "SCRATCH_ONLY_NO_FOUNDATION_EXPOSURE_NO_MODEL_INPUT_UNTIL_QUALIFIED",
            "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
            "unknown_historical_exposure_is_gate_blocker": True,
            "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
            "changes_current_qualified_counts": False,
            "gse261709_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
            "gse269595_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
            "emtab10902_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
            "gate_threshold_relaxation_authorized": False,
            "dataset_role_assignment_allowed": False,
            "qualification_allowed": False,
            "canonical_materialization_allowed": False,
            "split_execution_allowed": False,
            "formal_qualification_power_gate_execution_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "a7_allowed": False,
            "next_phase_authorized": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "latest_settled_runtime_event_id": "A1-EVT-057",
            "settled_runtime_event_changed": False,
            "runtime_event_emitted": False,
            "runtime_sync_status": "PENDING_FRESH_EVENT_AFTER_SETTLED_EVT_057",
            "expected_next_runtime_event_id": DEC024_PENDING_RUNTIME_EVENT_ID,
            "next_runtime_event_id_preallocated": False,
            "sealed_contact": False,
            "strategic_nonbinding_possible_future_combination": "EXISTING_GSE261709_AS_A1_PLUS_GSE269595_AS_TRUE_A2",
            "strategic_combination_requires_separate_formal_qualification_authority": True,
        }
        for key, value in expected_dec024_fields.items():
            if not _json_type_strict_equal(dec024.get(key), value):
                _issue(issues, "DECISION_LOG_DEC024", DECISION_LOG_PATH, f"V3-DEC-024.{key} must remain {value!r}")
        expected_refs = {
            GOAL_PATH,
            DEC023_AMENDMENT_PATH,
            DEC024_AMENDMENT_PATH,
            CONFIG_PATH,
            A1_QUALIFICATION_CONFIG_PATH,
            REGISTRY_PATHS["data"],
            REGISTRY_PATHS["task"],
            A1_INTERIM_PATH,
        }
        if not isinstance(dec024.get("evidence_refs"), list) or set(dec024["evidence_refs"]) != expected_refs:
            _issue(issues, "DECISION_LOG_DEC024_EVIDENCE", DECISION_LOG_PATH, f"V3-DEC-024 evidence_refs must be exactly {sorted(expected_refs)!r}")
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
    for relative, expected_sha256 in DEC019_IMMUTABLE_LEAF_SHA256.items():
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
        "dec020_amendment_path": DEC020_AMENDMENT_PATH,
        "dec021_amendment_path": DEC021_AMENDMENT_PATH,
        "dec022_amendment_path": DEC022_AMENDMENT_PATH,
        "dec023_amendment_path": DEC023_AMENDMENT_PATH,
        "dec024_amendment_path": DEC024_AMENDMENT_PATH,
        "dec027_amendment_path": DEC027_AMENDMENT_PATH,
        "dec028_amendment_path": DEC028_AMENDMENT_PATH,
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
            "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
            "checkpoint_specific_exposure_may_be_waived_for_foundation_checkpoint_route": False,
            "scratch_only_no_external_learned_inputs_is_checkpoint_exposure_waiver": False,
            "license_or_redistribution_rights_may_be_waived": False,
            "uncertainty_routes_are_dataset_scoped_only": True,
            "global_replicate_or_standard_error_relaxation_allowed": False,
            "gse149487_three_biological_replicates_and_route_a_se_gate_changed": False,
            "other_dataset_specific_stricter_replicate_or_standard_error_gates_changed": False,
            "decision_alone_qualifies_any_study": False,
            "decision_alone_authorizes_training_or_model_selection": False,
            "decision_or_policy_alone_is_training_evidence": False,
        }.items():
            _expect(root_policy, key, value, CONFIG_PATH, issues, "DEC019_ROOT_POLICY")

    for name in ("data", "split", "task", "matrix", "claim"):
        ref = registries[name].get("authority_ref")
        if not isinstance(ref, Mapping):
            _issue(issues, "DEC019_REGISTRY_AUTHORITY", REGISTRY_PATHS[name], "authority_ref must be a mapping")
            continue
        expected_ids = ACTIVE_AMENDMENT_DECISION_IDS
        _expect(ref, "active_amendment_decision_ids", expected_ids, REGISTRY_PATHS[name], issues, "DEC019_REGISTRY_AUTHORITY")
        _expect(ref, "dec019_amendment_path", DEC019_AMENDMENT_PATH, REGISTRY_PATHS[name], issues, "DEC019_REGISTRY_AUTHORITY")
        _expect(ref, "dec020_amendment_path", DEC020_AMENDMENT_PATH, REGISTRY_PATHS[name], issues, "DEC019_REGISTRY_AUTHORITY")
        if name in {"data", "task"}:
            _expect(ref, "dec021_amendment_path", DEC021_AMENDMENT_PATH, REGISTRY_PATHS[name], issues, "DEC021_REGISTRY_AUTHORITY")
            _expect(ref, "dec022_amendment_path", DEC022_AMENDMENT_PATH, REGISTRY_PATHS[name], issues, "DEC022_REGISTRY_AUTHORITY")
            _expect(ref, "dec023_amendment_path", DEC023_AMENDMENT_PATH, REGISTRY_PATHS[name], issues, "DEC023_REGISTRY_AUTHORITY")
            _expect(ref, "dec024_amendment_path", DEC024_AMENDMENT_PATH, REGISTRY_PATHS[name], issues, "DEC024_REGISTRY_AUTHORITY")
            _expect(ref, "dec027_amendment_path", DEC027_AMENDMENT_PATH, REGISTRY_PATHS[name], issues, "DEC027_REGISTRY_AUTHORITY")
        _expect(ref, "dec028_amendment_path", DEC028_AMENDMENT_PATH, REGISTRY_PATHS[name], issues, "DEC028_REGISTRY_AUTHORITY")

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
                "model_input_route_and_route_conditional_exposure_gate_required": True,
                "foundation_checkpoint_exposure_or_rights_waiver_allowed": False,
                "scratch_only_no_external_learned_inputs_is_checkpoint_exposure_waiver": False,
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


def validate_dec020_authority(
    repo_root: Path,
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    """Freeze DEC-020 as authority only; no route adjudication or V4 registration."""

    issues: list[Issue] = []
    for relative, expected_sha256 in DEC020_PRESERVED_AUTHORITY_LEAF_SHA256.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(
                issues,
                "DEC020_ACTIVE_AUTHORITY_LEAF_UNREADABLE",
                relative,
                str(exc),
            )
            continue
        allowed_sha256 = {expected_sha256}
        successor_sha256 = DEC028_ACTIVE_AUTHORITY_LEAF_SHA256.get(relative)
        if successor_sha256 is not None:
            allowed_sha256.add(successor_sha256)
        if relative == "docs/execution/route_a_v3_task_registry.yaml":
            allowed_sha256.add(CURRENT_TASK_REGISTRY_SHA256)
        if actual_sha256 not in allowed_sha256:
            _issue(
                issues,
                "DEC020_ACTIVE_AUTHORITY_LEAF_DRIFT",
                relative,
                f"active authority leaf hash {actual_sha256} must match the frozen DEC020 leaf or the current task-registry projection",
            )

    try:
        amendment = _load_yaml(repo_root, DEC020_AMENDMENT_PATH)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        _issue(issues, "DEC020_AMENDMENT_LOAD", DEC020_AMENDMENT_PATH, str(exc))
        return issues
    try:
        qualification = _load_json(repo_root, A1_QUALIFICATION_CONFIG_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "DEC020_A1_QUALIFICATION_LOAD",
            A1_QUALIFICATION_CONFIG_PATH,
            str(exc),
        )
        return issues

    expected_metadata = {
        "schema_version": "1.0.0",
        "amendment_id": "MRNA_XEDITFLOW_ROUTE_A_V3_DEC020",
        "decision_id": "V3-DEC-020",
        "contract_id": CONTRACT_ID,
        "contract_version": VERSION,
        "amends_contract_path": GOAL_PATH,
        "amends_contract_sha256": SOURCE_CONTRACT_SHA256,
        "predecessor_amendment_path": DEC019_AMENDMENT_PATH,
        "predecessor_amendment_sha256": DEC019_IMMUTABLE_LEAF_SHA256[
            DEC019_AMENDMENT_PATH
        ],
        "amendment_mode": (
            "APPEND_ONLY_AUTHORITY_COMPANION_ROOT_CONTRACT_AND_"
            "DEC019_BYTES_UNCHANGED"
        ),
        "status": "FROZEN_USER_AUTHORIZED_GSE200304_MODEL_INPUT_ROUTE_POLICY",
        "effective_phase": "A1",
        "requires_user_authorization": True,
    }
    expected_top_keys = set(expected_metadata) | {
        "user_authorization",
        "scope",
        "route_conditional_gate",
        "gse200304_gate_preservation",
        "prior_aggregate_design_use_disclosure",
        "authorization_projection",
        "historical_preservation",
    }
    if set(amendment) != expected_top_keys:
        _issue(
            issues,
            "DEC020_AMENDMENT_CLOSURE",
            DEC020_AMENDMENT_PATH,
            f"top-level keys must be exactly {sorted(expected_top_keys)!r}",
        )
    for key, value in expected_metadata.items():
        _expect(
            amendment,
            key,
            value,
            DEC020_AMENDMENT_PATH,
            issues,
            "DEC020_AMENDMENT_METADATA",
        )

    closed_sections = {
        "user_authorization": {
            "status": "GRANTED",
            "received_at": "2026-08-13T12:07:11+08:00",
            "source": "ACTIVE_CODEX_THREAD_OWNER_CONFIRMATION",
        },
        "scope": {
            "dataset_id": "GSE200304",
            "primary_measurement_route": "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT",
            "qualification_is_model_input_route_scoped": True,
            "current_qualification_route": DEC020_SCRATCH_ROUTE,
            "allowed_model_input_routes": [
                DEC020_SCRATCH_ROUTE,
                DEC020_FOUNDATION_ROUTE,
            ],
            "route_selected_before_model_results": True,
            "outcome_or_model_result_used_for_route_selection": False,
            "route_switch_after_model_results_allowed": False,
            "route_fallback_after_failure_allowed": False,
            "same_dataset_may_receive_duplicate_gate_credit_across_routes": False,
        },
        "gse200304_gate_preservation": {
            "minimum_independent_ordinary_studies": 3,
            "minimum_qualified_a1_studies": 2,
            "minimum_qualified_true_a2_dense_studies": 1,
            "maximum_independent_ordinary_study_contribution_if_qualified": 1,
            "maximum_a1_study_contribution_if_qualified": 1,
            "maximum_true_a2_dense_study_contribution_if_qualified": 0,
            "all_nonroute_nonwaivable_gates_remain_required": [
                "ROW_LEVEL_MULTI_ASSET_LINEAGE_AND_LOCATORS_CLOSED",
                "ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED",
                "BIOLOGICAL_SOURCE_GROUP_AUTHORITY_CLOSED",
                "REPLICATE_OR_VALID_STANDARD_ERROR_CLOSED",
                "LICENSE_AND_REDISTRIBUTION_RIGHTS_CLOSED",
                "A1_SOURCE_GROUP_NEAR_DUPLICATE_GRAPH_AND_SALT_FROZEN",
                "A1_ZERO_LEAKAGE_AUDIT_PASS",
                "PREFROZEN_GROUP_POWER_AT_LEAST_0_80_PASS",
                "PREFROZEN_FULL_CI_WIDTH_AT_MOST_0_30_PASS",
            ],
            "license_or_redistribution_rights_may_be_waived": False,
            "existing_license_and_redistribution_rights_gate_status": "PASS",
            "rights_gate_reopened_by_dec020": False,
            "target_power_minimum": 0.8,
            "maximum_full_ci_width": 0.3,
            "split_or_threshold_may_change_after_model_results": False,
        },
        "prior_aggregate_design_use_disclosure": {
            "status": "DISCLOSED_NOT_UNTOUCHED",
            "dataset_id": "GSE200304",
            "disclosed_use": (
                "AGGREGATE_PUBLIC_STRUCTURAL_AUTHORITY_INFORMED_PROTOCOL_DESIGN"
            ),
            "row_sequence_or_effect_payload_disclosed_by_this_amendment": False,
            "may_be_called_untouched": False,
            "claim_of_no_prior_influence_allowed": False,
            "selected_test_may_be_relabelled_untouched": False,
            "a2_final_benchmark_membership_frozen_by_this_amendment": False,
            "prospective_freeze_boundary": "DEC020_FORWARD",
            "full_prior_analytic_use_attestation_required_before_predictor_protocol_promotion_or_training": True,
            "full_prior_analytic_use_attestation_completion_asserted_by_dec020": False,
            "required_future_label": (
                "PRIOR_AGGREGATE_DESIGN_USE_DISCLOSED_NOT_UNTOUCHED"
            ),
        },
        "authorization_projection": {
            "changes_current_qualified_counts": False,
            "current_qualified_independent_ordinary_studies": 0,
            "current_qualified_a1_studies": 0,
            "current_qualified_true_a2_dense_studies": 0,
            "current_canonical_record_count": 0,
            "phase_complete": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "decision_alone_qualifies_gse200304": False,
            "decision_alone_materializes_canonical_records": False,
            "decision_alone_authorizes_adjudication_success": False,
            "decision_or_policy_alone_is_training_evidence": False,
            "future_route_scoped_qualification_and_canonical_count_possible_after_all_gates_pass": True,
            "canonical_materialization_execution_authorized": False,
            "private_payload_access_allowed": False,
        },
        "historical_preservation": {
            "root_contract_bytes_changed": False,
            "dec019_amendment_bytes_changed": False,
            "existing_dec019_activation_config_bytes_changed": False,
            "existing_dec019_adjudicator_bytes_changed": False,
            "existing_dec019_focused_test_bytes_changed": False,
            "existing_checkpoint_exposure_fail_record_changed_or_relabelled": False,
            "prior_runtime_events_changed": False,
            "failure_evidence_deleted_or_relabelled": False,
        },
    }
    for key, expected in closed_sections.items():
        observed = amendment.get(key)
        if not isinstance(observed, Mapping):
            _issue(
                issues,
                "DEC020_AMENDMENT_SEMANTICS",
                DEC020_AMENDMENT_PATH,
                f"{key} must be a mapping",
            )
        else:
            _expect_closed_mapping(
                observed,
                expected,
                DEC020_AMENDMENT_PATH,
                issues,
                "DEC020_AMENDMENT_SEMANTICS",
            )

    route_gate = amendment.get("route_conditional_gate")
    if not isinstance(route_gate, Mapping):
        _issue(
            issues,
            "DEC020_ROUTE_GATE",
            DEC020_AMENDMENT_PATH,
            "route_conditional_gate must be a mapping",
        )
    else:
        expected_gate_keys = {
            "gate_id",
            "gate_is_required",
            "unknown_or_mixed_route_passes",
            "scratch_route",
            "foundation_route",
        }
        if set(route_gate) != expected_gate_keys:
            _issue(
                issues,
                "DEC020_ROUTE_GATE",
                DEC020_AMENDMENT_PATH,
                f"route gate keys must be exactly {sorted(expected_gate_keys)!r}",
            )
        for key, value in {
            "gate_id": DEC020_ROUTE_GATE,
            "gate_is_required": True,
            "unknown_or_mixed_route_passes": False,
        }.items():
            _expect(
                route_gate,
                key,
                value,
                DEC020_AMENDMENT_PATH,
                issues,
                "DEC020_ROUTE_GATE",
            )
        scratch = route_gate.get("scratch_route")
        if not isinstance(scratch, Mapping):
            _issue(
                issues,
                "DEC020_SCRATCH_ROUTE",
                DEC020_AMENDMENT_PATH,
                "scratch_route must be a mapping",
            )
        else:
            _expect_closed_mapping(
                scratch,
                {
                    "route_id": DEC020_SCRATCH_ROUTE,
                    "current_status": "FROZEN_AUTHORIZED_NOT_YET_ADJUDICATED",
                    "checkpoint_specific_exposure_gate_applicable": False,
                    "checkpoint_specific_exposure_status_if_selected": DEC020_SCRATCH_EXPOSURE_STATUS,
                    "checkpoint_specific_exposure_pass_claimed": False,
                    "external_checkpoint_count_allowed": 0,
                    "external_learned_input_count_allowed": 0,
                    "allowed_parameter_initialization": "RANDOM_INITIALIZATION_ONLY",
                    "allowed_inputs": [
                        "QUALIFIED_ROUTE_A_TRAIN_PARTITION_AFTER_A2_FREEZE",
                        "DETERMINISTIC_NONLEARNED_TOKENIZATION_AND_TRANSFORMS",
                        "FIXED_NONLEARNED_BIOLOGICAL_FEATURES_AND_ACTION_GRAPH",
                        "PUBLIC_ARCHITECTURE_AND_METHOD_DESCRIPTIONS_WITHOUT_LEARNED_STATE",
                    ],
                    "forbidden_external_learned_inputs": DEC020_FORBIDDEN_EXTERNAL_LEARNED_INPUTS,
                    "route_contract_binding_required_before_adjudication": True,
                    "runtime_no_external_learned_input_attestation_required_before_any_learned_parameter_run": True,
                    "any_forbidden_external_learned_input_invalidates_route": True,
                    "invalidated_scratch_run_may_be_relabelled_foundation_without_re_adjudication": False,
                },
                DEC020_AMENDMENT_PATH,
                issues,
                "DEC020_SCRATCH_ROUTE",
            )
        foundation = route_gate.get("foundation_route")
        if not isinstance(foundation, Mapping):
            _issue(
                issues,
                "DEC020_FOUNDATION_ROUTE",
                DEC020_AMENDMENT_PATH,
                "foundation_route must be a mapping",
            )
        else:
            _expect_closed_mapping(
                foundation,
                {
                    "route_id": DEC020_FOUNDATION_ROUTE,
                    "current_status": "RETAINED_FAIL_CURRENT_PROTOCOL",
                    "existing_dec019_activation_and_adjudicator_retained": True,
                    "checkpoint_specific_exposure_gate_applicable": True,
                    "checkpoint_specific_exposure_may_be_waived": False,
                    "minimum_audited_checkpoint_count_for_pass": 1,
                    "empty_checkpoint_set_can_pass": False,
                    "required_true_fact_fields": [
                        "checkpoint_ids_and_revisions_frozen",
                        "checkpoint_artifact_digests_bound",
                        "exact_member_exposure_audit_pass",
                        "near_duplicate_exposure_audit_pass",
                    ],
                },
                DEC020_AMENDMENT_PATH,
                issues,
                "DEC020_FOUNDATION_ROUTE",
            )

    q_dec020 = qualification.get("dec020_model_input_route_authority")
    if not isinstance(q_dec020, Mapping):
        _issue(
            issues,
            "DEC020_A1_QUALIFICATION_POLICY",
            A1_QUALIFICATION_CONFIG_PATH,
            "dec020_model_input_route_authority must be a mapping",
        )
    else:
        for key, value in {
            "dataset_id": "GSE200304",
            "qualification_is_model_input_route_scoped": True,
            "selected_route": DEC020_SCRATCH_ROUTE,
            "allowed_routes": [DEC020_SCRATCH_ROUTE, DEC020_FOUNDATION_ROUTE],
            "route_selected_before_model_results": True,
            "outcome_or_model_result_used_for_route_selection": False,
            "route_switch_after_model_results_allowed": False,
            "route_fallback_after_failure_allowed": False,
            "same_dataset_may_receive_duplicate_gate_credit_across_routes": False,
            "model_input_route_gate": DEC020_ROUTE_GATE,
            "minimum_gate_counts_unchanged": {"ordinary": 3, "a1": 2, "true_a2": 1},
            "maximum_gse200304_contribution_if_qualified": {"ordinary": 1, "a1": 1, "true_a2": 0},
            "all_nonroute_nonwaivable_gates_must_pass": True,
            "existing_license_and_redistribution_rights_gate_status": "PASS",
            "rights_gate_reopened_by_dec020": False,
            "decision_alone_qualifies_gse200304": False,
            "future_route_scoped_qualification_and_canonical_count_possible_after_all_gates_pass": True,
            "canonical_materialization_execution_authorized": False,
            "decision_or_policy_alone_is_training_evidence": False,
            "private_payload_access_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }.items():
            _expect(
                q_dec020,
                key,
                value,
                A1_QUALIFICATION_CONFIG_PATH,
                issues,
                "DEC020_A1_QUALIFICATION_POLICY",
            )
        q_scratch = q_dec020.get("scratch_only_route")
        if not isinstance(q_scratch, Mapping):
            _issue(
                issues,
                "DEC020_A1_SCRATCH_ROUTE",
                A1_QUALIFICATION_CONFIG_PATH,
                "scratch_only_route must be a mapping",
            )
        else:
            for key, value in {
                "current_status": "FROZEN_AUTHORIZED_NOT_YET_ADJUDICATED",
                "checkpoint_specific_exposure_gate_applicable": False,
                "checkpoint_specific_exposure_status_if_selected": DEC020_SCRATCH_EXPOSURE_STATUS,
                "checkpoint_specific_exposure_pass_claimed": False,
                "external_checkpoint_count_allowed": 0,
                "external_learned_input_count_allowed": 0,
                "allowed_parameter_initialization": "RANDOM_INITIALIZATION_ONLY",
                "forbidden_external_learned_inputs": DEC020_FORBIDDEN_EXTERNAL_LEARNED_INPUTS,
                "runtime_no_external_learned_input_attestation_required_before_any_learned_parameter_run": True,
                "any_forbidden_external_learned_input_invalidates_route": True,
            }.items():
                _expect(
                    q_scratch,
                    key,
                    value,
                    A1_QUALIFICATION_CONFIG_PATH,
                    issues,
                    "DEC020_A1_SCRATCH_ROUTE",
                )
        q_foundation = q_dec020.get("foundation_checkpoint_route")
        if not isinstance(q_foundation, Mapping):
            _issue(
                issues,
                "DEC020_A1_FOUNDATION_ROUTE",
                A1_QUALIFICATION_CONFIG_PATH,
                "foundation_checkpoint_route must be a mapping",
            )
        else:
            for key, value in {
                "current_status": "RETAINED_FAIL_CURRENT_PROTOCOL",
                "existing_dec019_activation_and_adjudicator_retained": True,
                "checkpoint_specific_exposure_gate_applicable": True,
                "checkpoint_specific_exposure_may_be_waived": False,
                "minimum_audited_checkpoint_count_for_pass": 1,
                "empty_checkpoint_set_can_pass": False,
            }.items():
                _expect(
                    q_foundation,
                    key,
                    value,
                    A1_QUALIFICATION_CONFIG_PATH,
                    issues,
                    "DEC020_A1_FOUNDATION_ROUTE",
                )
        q_disclosure = q_dec020.get("prior_aggregate_design_use_disclosure")
        if not isinstance(q_disclosure, Mapping):
            _issue(
                issues,
                "DEC020_A1_PRIOR_USE",
                A1_QUALIFICATION_CONFIG_PATH,
                "prior-use disclosure must be a mapping",
            )
        else:
            for key, value in {
                "status": "DISCLOSED_NOT_UNTOUCHED",
                "may_be_called_untouched": False,
                "claim_of_no_prior_influence_allowed": False,
                "prospective_freeze_boundary": "DEC020_FORWARD",
                "full_prior_analytic_use_attestation_required_before_predictor_protocol_promotion_or_training": True,
                "full_prior_analytic_use_attestation_completion_asserted_by_dec020": False,
            }.items():
                _expect(
                    q_disclosure,
                    key,
                    value,
                    A1_QUALIFICATION_CONFIG_PATH,
                    issues,
                    "DEC020_A1_PRIOR_USE",
                )

    root_policy = config.get("a1_qualification_authority")
    root_gse200304 = (
        root_policy.get("gse200304") if isinstance(root_policy, Mapping) else None
    )
    if not isinstance(root_gse200304, Mapping):
        _issue(
            issues,
            "DEC020_ROOT_POLICY",
            CONFIG_PATH,
            "a1_qualification_authority.gse200304 must be a mapping",
        )
    else:
        for key, value in {
            "current_status": "NOT_QUALIFIED",
            "qualification_is_model_input_route_scoped": True,
            "current_qualification_route": DEC020_SCRATCH_ROUTE,
            "allowed_model_input_routes": [DEC020_SCRATCH_ROUTE, DEC020_FOUNDATION_ROUTE],
            "route_selected_before_model_results": True,
            "route_switch_after_model_results_allowed": False,
            "route_fallback_after_failure_allowed": False,
            "same_dataset_may_receive_duplicate_gate_credit_across_routes": False,
            "model_input_route_gate": DEC020_ROUTE_GATE,
            "prior_aggregate_design_use_disclosed": True,
            "untouched_claim_allowed": False,
            "claim_of_no_prior_influence_allowed": False,
            "prospective_freeze_boundary": "DEC020_FORWARD",
            "full_prior_analytic_use_attestation_required_before_predictor_protocol_promotion_or_training": True,
            "full_prior_analytic_use_attestation_completion_asserted_by_dec020": False,
            "existing_license_and_redistribution_rights_gate_status": "PASS",
            "rights_gate_reopened_by_dec020": False,
            "future_route_scoped_qualification_and_canonical_count_possible_after_all_gates_pass": True,
            "canonical_materialization_execution_authorized": False,
            "private_payload_access_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
        }.items():
            _expect(
                root_gse200304,
                key,
                value,
                CONFIG_PATH,
                issues,
                "DEC020_ROOT_POLICY",
            )

    data_rows = registries["data"].get("datasets")
    data_gse200304 = _mapping_entry(data_rows, "dataset_id", "GSE200304")
    data_route = (
        data_gse200304.get("dec020_model_input_route_policy")
        if isinstance(data_gse200304, Mapping)
        else None
    )
    if not isinstance(data_route, Mapping):
        _issue(
            issues,
            "DEC020_DATA_ROLE_POLICY",
            REGISTRY_PATHS["data"],
            "GSE200304 DEC020 route policy must be a mapping",
        )
    else:
        for key, value in {
            "selected_route": DEC020_SCRATCH_ROUTE,
            "retained_route": DEC020_FOUNDATION_ROUTE,
            "route_selected_before_model_results": True,
            "route_switch_after_model_results_allowed": False,
            "route_fallback_after_failure_allowed": False,
            "same_dataset_may_receive_duplicate_gate_credit_across_routes": False,
            "scratch_route_checkpoint_exposure_status": DEC020_SCRATCH_EXPOSURE_STATUS,
            "scratch_route_checkpoint_exposure_pass_claimed": False,
            "scratch_route_external_checkpoint_count_allowed": 0,
            "scratch_route_external_learned_input_count_allowed": 0,
            "scratch_route_parameter_initialization": "RANDOM_INITIALIZATION_ONLY",
            "scratch_route_runtime_attestation_required_before_any_learned_parameter_run": True,
            "foundation_route_status": "RETAINED_FAIL_CURRENT_PROTOCOL",
            "foundation_route_dec019_activation_and_adjudicator_retained": True,
            "foundation_route_minimum_audited_checkpoint_count_for_pass": 1,
            "foundation_route_empty_checkpoint_set_can_pass": False,
            "disclosure_status": "DISCLOSED_NOT_UNTOUCHED",
            "may_be_called_untouched": False,
            "claim_of_no_prior_influence_allowed": False,
            "prospective_freeze_boundary": "DEC020_FORWARD",
            "full_prior_analytic_use_attestation_required_before_predictor_protocol_promotion_or_training": True,
            "full_prior_analytic_use_attestation_completion_asserted_by_dec020": False,
            "existing_license_and_redistribution_rights_gate_status": "PASS",
            "rights_gate_reopened_by_dec020": False,
            "canonical_materialization_execution_authorized": False,
            "private_payload_access_allowed": False,
            "gpu_work_allowed": False,
            "sealed_contact_allowed": False,
            "decision_or_policy_alone_is_training_evidence": False,
        }.items():
            _expect(
                data_route,
                key,
                value,
                REGISTRY_PATHS["data"],
                issues,
                "DEC020_DATA_ROLE_POLICY",
            )

    simple_boundaries = (
        (
            "split",
            "dec020_model_input_route_split_boundary",
            {
                "changes_split_ids_or_assignment": False,
                "changes_a1_source_group_graph_or_salt_freeze": False,
                "changes_a2_final_benchmark_membership_freeze": False,
                "split_or_membership_may_change_after_model_results": False,
            },
        ),
        (
            "matrix",
            "dec020_model_input_route_task_split_boundary",
            {
                "changes_expected_task_ids": False,
                "changes_expected_split_ids": False,
                "changes_task_to_split_assignments": False,
                "changes_ordinary_or_sealed_task_boundary": False,
                "task_or_split_assignment_may_change_after_model_results": False,
            },
        ),
    )
    for registry_name, section_name, extra_expected in simple_boundaries:
        boundary = registries[registry_name].get(section_name)
        if not isinstance(boundary, Mapping):
            _issue(
                issues,
                "DEC020_REGISTRY_BOUNDARY",
                REGISTRY_PATHS[registry_name],
                f"{section_name} must be a mapping",
            )
            continue
        expected = {
            "dataset_id": "GSE200304",
            "selected_route": DEC020_SCRATCH_ROUTE,
            "retained_route": DEC020_FOUNDATION_ROUTE,
            "route_selected_before_model_results": True,
            "prior_aggregate_design_use": (
                "AGGREGATE_PUBLIC_STRUCTURAL_AUTHORITY_INFORMED_PROTOCOL_DESIGN"
            ),
            "disclosure_status": "DISCLOSED_NOT_UNTOUCHED",
            "may_be_called_untouched": False,
            "claim_of_no_prior_influence_allowed": False,
            "prospective_freeze_boundary": "DEC020_FORWARD",
            **extra_expected,
        }
        _expect_closed_mapping(
            boundary,
            expected,
            REGISTRY_PATHS[registry_name],
            issues,
            "DEC020_REGISTRY_BOUNDARY",
        )

    for registry_name, section_name in (
        ("task", "dec020_model_input_route_boundaries"),
        ("claim", "dec020_model_input_route_claim_boundaries"),
    ):
        boundary = registries[registry_name].get(section_name)
        if not isinstance(boundary, Mapping):
            _issue(
                issues,
                "DEC020_REGISTRY_POLICY",
                REGISTRY_PATHS[registry_name],
                f"{section_name} must be a mapping",
            )
            continue
        for key, value in {
            "dataset_id": "GSE200304",
            "selected_route": DEC020_SCRATCH_ROUTE,
            "retained_route": DEC020_FOUNDATION_ROUTE,
            "model_input_route_gate": DEC020_ROUTE_GATE,
            "scratch_route_checkpoint_exposure_status": DEC020_SCRATCH_EXPOSURE_STATUS,
            "scratch_route_checkpoint_exposure_pass_claimed": False,
            "scratch_route_external_checkpoint_count_allowed": 0,
            "scratch_route_external_learned_input_count_allowed": 0,
            "foundation_route_status": "RETAINED_FAIL_CURRENT_PROTOCOL",
            "foundation_route_minimum_audited_checkpoint_count_for_pass": 1,
            "foundation_route_empty_checkpoint_set_can_pass": False,
            "disclosure_status": "DISCLOSED_NOT_UNTOUCHED",
            "may_be_called_untouched": False,
            "claim_of_no_prior_influence_allowed": False,
            "prospective_freeze_boundary": "DEC020_FORWARD",
            "full_prior_analytic_use_attestation_required_before_predictor_protocol_promotion_or_training": True,
            "full_prior_analytic_use_attestation_completion_asserted_by_dec020": False,
            "existing_license_and_redistribution_rights_gate_status": "PASS",
            "rights_gate_reopened_by_dec020": False,
            "canonical_materialization_execution_authorized": False,
            "private_payload_access_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
            "decision_or_policy_alone_is_training_evidence": False,
        }.items():
            _expect(
                boundary,
                key,
                value,
                REGISTRY_PATHS[registry_name],
                issues,
                "DEC020_REGISTRY_POLICY",
            )

    return issues


def validate_dec021_authority(
    repo_root: Path,
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    """Preserve the completed DEC-021 authority bytes under its DEC-022 successor."""

    issues: list[Issue] = []
    for relative, expected_sha256 in DEC021_PRESERVED_HISTORICAL_LEAF_SHA256.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "DEC021_HISTORICAL_LEAF_UNREADABLE", relative, str(exc))
            continue
        allowed_sha256 = {expected_sha256}
        successor_sha256 = DEC028_ACTIVE_AUTHORITY_LEAF_SHA256.get(relative)
        if successor_sha256 is not None:
            allowed_sha256.add(successor_sha256)
        if relative == "docs/execution/route_a_v3_task_registry.yaml":
            allowed_sha256.add(CURRENT_TASK_REGISTRY_SHA256)
        if actual_sha256 not in allowed_sha256:
            _issue(
                issues,
                "DEC021_HISTORICAL_LEAF_DRIFT",
                relative,
                f"historical DEC021 leaf hash {actual_sha256} must remain {expected_sha256}, except for the current task-registry projection",
            )
    try:
        amendment = _load_yaml(repo_root, DEC021_AMENDMENT_PATH)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        _issue(issues, "DEC021_AMENDMENT_LOAD", DEC021_AMENDMENT_PATH, str(exc))
        return issues
    try:
        qualification = _load_json(repo_root, A1_QUALIFICATION_CONFIG_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "DEC021_A1_QUALIFICATION_LOAD", A1_QUALIFICATION_CONFIG_PATH, str(exc))
        return issues

    expected_metadata = {
        "schema_version": "1.0.0",
        "amendment_id": "MRNA_XEDITFLOW_ROUTE_A_V3_DEC021",
        "decision_id": "V3-DEC-021",
        "contract_id": CONTRACT_ID,
        "contract_version": VERSION,
        "amends_contract_path": GOAL_PATH,
        "amends_contract_sha256": SOURCE_CONTRACT_SHA256,
        "predecessor_amendment_path": DEC020_AMENDMENT_PATH,
        "predecessor_amendment_sha256": "0cfbe6e35c2c7f3b19756b8aee41dc91b2a8f05b249a5b6e9cacf90185c56026",
        "amendment_mode": "APPEND_ONLY_AUTHORITY_COMPANION_ROOT_CONTRACT_AND_DEC020_BYTES_UNCHANGED",
        "status": "FROZEN_USER_AUTHORIZED_GSE256185_PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
        "effective_phase": "A1",
        "requires_user_authorization": True,
    }
    expected_top_keys = set(expected_metadata) | {
        "user_authorization",
        "scope",
        "preflight_semantics",
        "authorization_projection",
        "historical_preservation",
    }
    if set(amendment) != expected_top_keys:
        _issue(
            issues,
            "DEC021_AMENDMENT_CLOSURE",
            DEC021_AMENDMENT_PATH,
            f"top-level keys must be exactly {sorted(expected_top_keys)!r}",
        )
    for key, value in expected_metadata.items():
        _expect(amendment, key, value, DEC021_AMENDMENT_PATH, issues, "DEC021_AMENDMENT_METADATA")

    expected_sections = {
        "user_authorization": {
            "status": "GRANTED",
            "received_at": "2026-08-13T19:45:00+08:00",
            "source": "ACTIVE_CODEX_THREAD_OWNER_CONFIRMATION",
        },
        "scope": {
            "dataset_id": "GSE256185",
            "role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
            "authority_surface": "ORDINARY_PUBLIC_ONLY",
            "allowed_input_field_classes_exactly": ["IDENTIFIER", "ROLE", "CONTEXT"],
            "allowed_output_class": "AGGREGATE_POOL_GEOMETRY_ONLY",
            "row_output_allowed": False,
            "sequence_output_allowed": False,
            "effect_output_allowed": False,
            "private_or_restricted_input_allowed": False,
            "sealed_contact_allowed": False,
        },
        "preflight_semantics": {
            "purpose": "TEST_WHETHER_PUBLIC_IDENTIFIER_ROLE_CONTEXT_FIELDS_SUPPORT_AGGREGATE_POOL_GEOMETRY_WITHOUT_QUALIFICATION",
            "preflight_candidate_only_not_counting": True,
            "identifier_evaluation": "AUTHORIZED",
            "role_evaluation": "AUTHORIZED",
            "context_evaluation": "AUTHORIZED",
            "aggregate_pool_geometry_evaluation": "AUTHORIZED",
            "sequence_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
            "edit_budget_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
            "effect_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
            "true_a2_status_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
            "qualification_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
            "output_may_contain_member_level_identifier": False,
            "output_may_contain_member_level_role": False,
            "output_may_contain_member_level_context": False,
            "output_may_contain_row_values": False,
            "output_may_contain_sequence_values": False,
            "output_may_contain_effect_values": False,
            "output_may_contain_edit_budget_values": False,
        },
        "authorization_projection": {
            "changes_current_qualified_counts": False,
            "current_qualified_independent_ordinary_studies": 1,
            "current_qualified_a1_studies": 1,
            "current_qualified_true_a2_dense_studies": 0,
            "current_canonical_record_count": 6547,
            "gse256185_ordinary_study_contribution": 0,
            "gse256185_a1_study_contribution": 0,
            "gse256185_true_a2_dense_study_contribution": 0,
            "gse256185_canonical_record_count": 0,
            "phase_complete": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "qualifier_execution_allowed": False,
            "canonical_materialization_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        },
        "historical_preservation": {
            "root_contract_bytes_changed": False,
            "dec019_amendment_bytes_changed": False,
            "dec020_amendment_bytes_changed": False,
            "prior_runtime_events_changed": False,
            "latest_settled_runtime_event_id": "A1-EVT-051",
            "evt051_settled_state_changed": False,
            "gse200304_current_qualification_changed": False,
            "failure_evidence_deleted_or_relabelled": False,
        },
    }
    for section_name, expected in expected_sections.items():
        observed = amendment.get(section_name)
        if not isinstance(observed, Mapping):
            _issue(issues, "DEC021_AMENDMENT_SEMANTICS", DEC021_AMENDMENT_PATH, f"{section_name} must be a mapping")
        else:
            _expect_closed_mapping(observed, expected, DEC021_AMENDMENT_PATH, issues, "DEC021_AMENDMENT_SEMANTICS")

    q_authority = qualification.get("authority")
    if not isinstance(q_authority, Mapping):
        _issue(issues, "DEC021_A1_QUALIFICATION_AUTHORITY", A1_QUALIFICATION_CONFIG_PATH, "authority must be a mapping")
    else:
        _expect(q_authority, "active_amendment_decision_ids", ACTIVE_AMENDMENT_DECISION_IDS, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC021_A1_QUALIFICATION_AUTHORITY")
        _expect(q_authority, "dec021_amendment_path", DEC021_AMENDMENT_PATH, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC021_A1_QUALIFICATION_AUTHORITY")

    q_scope = qualification.get("scope")
    if not isinstance(q_scope, Mapping):
        _issue(issues, "DEC021_A1_SCOPE", A1_QUALIFICATION_CONFIG_PATH, "scope must be a mapping")
    else:
        _expect(q_scope, "public_identifier_and_pool_geometry_preflight_only_dataset_ids", ["GSE256185"], A1_QUALIFICATION_CONFIG_PATH, issues, "DEC021_A1_SCOPE")
        included = q_scope.get("included_dataset_ids")
        if not isinstance(included, list) or "GSE256185" in included:
            _issue(issues, "DEC021_A1_SCOPE", A1_QUALIFICATION_CONFIG_PATH, "GSE256185 must remain outside qualification included_dataset_ids")

    expected_q = {
        "dataset_id": "GSE256185",
        "role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
        "user_authorization": {
            "status": "GRANTED",
            "received_at": "2026-08-13T19:45:00+08:00",
            "source": "ACTIVE_CODEX_THREAD_OWNER_CONFIRMATION",
        },
        "preflight_candidate_only_not_counting": True,
        "allowed_input_field_classes_exactly": ["IDENTIFIER", "ROLE", "CONTEXT"],
        "allowed_output_class": "AGGREGATE_POOL_GEOMETRY_ONLY",
        "sequence_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "edit_budget_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "effect_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "true_a2_status_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "qualification_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
        "changes_current_qualified_counts": False,
        "dataset_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
        "row_output_allowed": False,
        "sequence_output_allowed": False,
        "effect_output_allowed": False,
        "qualification_allowed": False,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "sealed_contact_allowed": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    observed_q = qualification.get("dec021_public_identifier_and_pool_geometry_preflight_authority")
    if not isinstance(observed_q, Mapping):
        _issue(issues, "DEC021_A1_POLICY", A1_QUALIFICATION_CONFIG_PATH, "DEC021 preflight authority must be a mapping")
    else:
        _expect_closed_mapping(observed_q, expected_q, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC021_A1_POLICY")

    root_policy = config.get("a1_qualification_authority")
    observed_root = root_policy.get("gse256185_public_identifier_and_pool_geometry_preflight") if isinstance(root_policy, Mapping) else None
    expected_root = {
        "current_status": "AUTHORIZED_NOT_RUN",
        "role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
        "preflight_candidate_only_not_counting": True,
        "allowed_input_field_classes_exactly": ["IDENTIFIER", "ROLE", "CONTEXT"],
        "allowed_output_class": "AGGREGATE_POOL_GEOMETRY_ONLY",
        "sequence_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "edit_budget_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "effect_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "true_a2_status_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "qualification_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0,
        "row_output_allowed": False,
        "sequence_output_allowed": False,
        "effect_output_allowed": False,
        "qualification_allowed": False,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "sealed_contact_allowed": False,
    }
    if not isinstance(observed_root, Mapping):
        _issue(issues, "DEC021_ROOT_POLICY", CONFIG_PATH, "GSE256185 preflight policy must be a mapping")
    else:
        _expect_closed_mapping(observed_root, expected_root, CONFIG_PATH, issues, "DEC021_ROOT_POLICY")
    if isinstance(root_policy, Mapping):
        _expect(root_policy, "current_qualified_counts", {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}, CONFIG_PATH, issues, "DEC021_ROOT_COUNTS")

    data = registries["data"]
    if "GSE256185" in set(data.get("ordinary_candidate_dataset_ids", [])):
        _issue(issues, "DEC021_DATA_ROLE", REGISTRY_PATHS["data"], "GSE256185 may not enter ordinary_candidate_dataset_ids")
    if "GSE256185" in set(data.get("true_a2_recovery_candidate_dataset_ids", [])):
        _issue(issues, "DEC021_DATA_ROLE", REGISTRY_PATHS["data"], "GSE256185 may not enter true_a2_recovery_candidate_dataset_ids")
    expected_row = {
        "dataset_id": "GSE256185",
        "aliases": ["PRJNA1078388"],
        "region_candidate": ["5UTR"],
        "sealed": False,
        "role": "AUDIT_ONLY",
        "qualification_status": "AUDIT_PENDING",
        "qualified": False,
        "training_role": "EXCLUDED_PENDING_QUALIFICATION",
        "intended_role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
        "preflight_candidate_only_not_counting": True,
        "mapping_status": "PUBLIC_IDENTIFIER_ROLE_CONTEXT_AND_AGGREGATE_POOL_GEOMETRY_PREFLIGHT_AUTHORIZED_NOT_RUN",
        "allowed_current_uses": ["OFFICIAL_PUBLIC_IDENTIFIER_ROLE_CONTEXT_AGGREGATE_AUDIT", "AGGREGATE_POOL_GEOMETRY_PREFLIGHT"],
        "forbidden_current_uses": ["ROW_OUTPUT", "SEQUENCE_OUTPUT", "EFFECT_OUTPUT", "EDIT_BUDGET_EVALUATION", "TRUE_A2_STATUS_EVALUATION", "QUALIFICATION", "CANONICAL_MATERIALIZATION", "TRAINING", "CALIBRATION", "MODEL_SELECTION", "NEXT_PHASE", "CONFIRMATORY_EVALUATION", "CLAIM_SUPPORT"],
        "allowed_input_field_classes_exactly": ["IDENTIFIER", "ROLE", "CONTEXT"],
        "allowed_output_class": "AGGREGATE_POOL_GEOMETRY_ONLY",
        "sequence_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "edit_budget_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "effect_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "true_a2_status_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "qualification_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
        "ordinary_gate_contribution": 0,
        "a1_gate_contribution": 0,
        "true_a2_gate_contribution": 0,
        "canonical_record_count": 0,
        "row_output_allowed": False,
        "sequence_output_allowed": False,
        "effect_output_allowed": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "sealed_contact_allowed": False,
        "evidence_status": "NOT_RUN",
    }
    observed_row = _mapping_entry(data.get("datasets"), "dataset_id", "GSE256185")
    if not isinstance(observed_row, Mapping):
        _issue(issues, "DEC021_DATA_ROLE", REGISTRY_PATHS["data"], "GSE256185 row is required")
    else:
        for key, value in {
            "dataset_id": "GSE256185",
            "aliases": ["PRJNA1078388"],
            "region_candidate": ["5UTR"],
            "sealed": False,
            "role": "AUDIT_ONLY",
            "qualification_status": "AUDIT_PENDING",
            "qualified": False,
            "training_role": "EXCLUDED_PENDING_QUALIFICATION",
            "preflight_candidate_only_not_counting": True,
            "ordinary_gate_contribution": 0,
            "a1_gate_contribution": 0,
            "true_a2_gate_contribution": 0,
            "canonical_record_count": 0,
            "row_output_allowed": False,
            "sequence_output_allowed": False,
            "effect_output_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
            "evidence_status": "NOT_RUN",
        }.items():
            _expect(
                observed_row,
                key,
                value,
                REGISTRY_PATHS["data"],
                issues,
                "DEC021_SUCCESSOR_PRESERVATION",
            )
    return issues


def validate_dec022_authority(
    repo_root: Path,
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    """Freeze the aggregate-only GSE256185 row-level preflight without promotion."""

    issues: list[Issue] = []
    for relative, expected_sha256 in DEC022_ACTIVE_AUTHORITY_LEAF_SHA256.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "DEC022_ACTIVE_AUTHORITY_LEAF_UNREADABLE", relative, str(exc))
            continue
        allowed_sha256 = {
            expected_sha256,
            DEC028_ACTIVE_AUTHORITY_LEAF_SHA256.get(relative, expected_sha256),
            DEC023_ACTIVE_AUTHORITY_LEAF_SHA256.get(relative, expected_sha256),
            DEC027_ACTIVE_AUTHORITY_LEAF_SHA256.get(
                relative,
                DEC024_ACTIVE_AUTHORITY_LEAF_SHA256.get(relative, expected_sha256),
            ),
        }
        if actual_sha256 not in allowed_sha256:
            _issue(
                issues,
                "DEC022_ACTIVE_AUTHORITY_LEAF_DRIFT",
                relative,
                f"active authority leaf hash {actual_sha256} must match the frozen DEC022 leaf or an accepted successor",
            )

    try:
        amendment = _load_yaml(repo_root, DEC022_AMENDMENT_PATH)
        qualification = _load_json(repo_root, A1_QUALIFICATION_CONFIG_PATH)
    except (FileNotFoundError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        _issue(issues, "DEC022_AUTHORITY_LOAD", DEC022_AMENDMENT_PATH, str(exc))
        return issues

    expected_metadata = {
        "schema_version": "1.0.0",
        "amendment_id": "MRNA_XEDITFLOW_ROUTE_A_V3_DEC022",
        "decision_id": "V3-DEC-022",
        "contract_id": CONTRACT_ID,
        "contract_version": VERSION,
        "amends_contract_path": GOAL_PATH,
        "amends_contract_sha256": SOURCE_CONTRACT_SHA256,
        "predecessor_amendment_path": DEC021_AMENDMENT_PATH,
        "predecessor_amendment_sha256": "2a7b05e40434398b1d39396280ca019f48164fbb70b5b6058123bc653c400d3d",
        "predecessor_authority_head": "c57f5aa937d33d7e5ec1c25d3e29b339628c6387",
        "amendment_mode": "APPEND_ONLY_AUTHORITY_COMPANION_ROOT_CONTRACT_AND_DEC021_BYTES_UNCHANGED",
        "status": "FROZEN_USER_AUTHORIZED_GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
        "effective_phase": "A1",
        "requires_user_authorization": True,
    }
    expected_top_keys = set(expected_metadata) | {
        "user_authorization",
        "predecessor_public_geometry_evidence",
        "candidate_universe",
        "authorized_internal_access",
        "aggregate_output_contract",
        "qualification_preflight_semantics",
        "fail_closed_gate_map",
        "authorization_projection",
        "runtime_successor",
        "historical_preservation",
    }
    if set(amendment) != expected_top_keys:
        _issue(
            issues,
            "DEC022_AMENDMENT_CLOSURE",
            DEC022_AMENDMENT_PATH,
            f"top-level keys must be exactly {sorted(expected_top_keys)!r}",
        )
    for key, value in expected_metadata.items():
        _expect(amendment, key, value, DEC022_AMENDMENT_PATH, issues, "DEC022_AMENDMENT_METADATA")

    _expect_closed_mapping(
        amendment.get("user_authorization") if isinstance(amendment.get("user_authorization"), Mapping) else {},
        {
            "status": "GRANTED",
            "received_at": "2026-08-13T22:15:00+08:00",
            "source": "ACTIVE_CODEX_THREAD_OWNER_CONFIRMATION",
        },
        DEC022_AMENDMENT_PATH,
        issues,
        "DEC022_USER_AUTHORIZATION",
    )
    expected_universe = {
        "dataset_id": "GSE256185",
        "universe_id": "GSE256185_DEC022_STRICT_SINGLE_PARENT_634_POOL_UNIVERSE",
        "selection_rule": "STRICT_GRAMMAR_EXACTLY_ONE_PARENT_AND_AT_LEAST_THREE_STRICT_CANDIDATES_PER_GROUP",
        "strict_single_parent_pool_count": 634,
        "strict_candidate_member_count": 7292,
        "membership_must_replay_to_exact_aggregate_counts_before_row_level_fields": True,
        "count_or_rule_drift_action": "STOP_BEFORE_ROW_LEVEL_FIELD_ACCESS",
        "two_candidate_strict_single_parent_group_count_excluded": 3,
        "dual_parent_group_count_excluded": 15,
        "nonstrict_grammar_record_count_excluded": 2,
        "reasoned_family_closure_candidate_count": 7294,
        "reasoned_family_closure_included": False,
        "dual_parent_or_nonstrict_future_inclusion_requires_new_explicit_authority": True,
        "grammar_role_alone_establishes_source_to_candidate_edit_relation": False,
    }
    observed_universe = amendment.get("candidate_universe")
    if not isinstance(observed_universe, Mapping):
        _issue(issues, "DEC022_UNIVERSE", DEC022_AMENDMENT_PATH, "candidate_universe must be a mapping")
    else:
        _expect_closed_mapping(observed_universe, expected_universe, DEC022_AMENDMENT_PATH, issues, "DEC022_UNIVERSE")

    expected_input_fields = ["IDENTIFIER", "ROLE", "SEQUENCE", "ENDPOINT", "REPLICATE", "NECESSARY_CONTEXT"]
    access = amendment.get("authorized_internal_access")
    if not isinstance(access, Mapping):
        _issue(issues, "DEC022_ACCESS", DEC022_AMENDMENT_PATH, "authorized_internal_access must be a mapping")
    else:
        for key, value in {
            "authority_surface": "ORDINARY_PUBLIC_ONLY",
            "allowed_input_field_classes_exactly": expected_input_fields,
            "internal_row_level_contrast_derivation_allowed": True,
            "internal_row_level_contrast_derivation_scope": "AGGREGATE_GATE_AUDIT_ONLY",
            "private_or_restricted_input_allowed": False,
            "sealed_contact_allowed": False,
            "gse246381_contact_allowed": False,
        }.items():
            _expect(access, key, value, DEC022_AMENDMENT_PATH, issues, "DEC022_ACCESS")

    output = amendment.get("aggregate_output_contract")
    if not isinstance(output, Mapping):
        _issue(issues, "DEC022_OUTPUT", DEC022_AMENDMENT_PATH, "aggregate_output_contract must be a mapping")
    else:
        for key, value in {
            "allowed_output_class": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
            "member_identifier_output_allowed": False,
            "member_role_output_allowed": False,
            "member_context_output_allowed": False,
            "row_output_allowed": False,
            "sequence_output_allowed": False,
            "row_effect_output_allowed": False,
            "replicate_identifier_output_allowed": False,
            "split_assignment_output_allowed": False,
            "canonical_record_output_allowed": False,
            "persistent_row_level_intermediate_allowed": False,
        }.items():
            _expect(output, key, value, DEC022_AMENDMENT_PATH, issues, "DEC022_OUTPUT")

    semantics = amendment.get("qualification_preflight_semantics")
    if not isinstance(semantics, Mapping):
        _issue(issues, "DEC022_SEMANTICS", DEC022_AMENDMENT_PATH, "qualification_preflight_semantics must be a mapping")
    else:
        for key, value in {
            "aggregate_row_level_qualification_preflight_execution_allowed": True,
            "dataset_qualification_decision_allowed": False,
            "a1_or_true_a2_assignment_allowed": False,
            "ordinary_a1_true_a2_credit_allowed": False,
            "canonical_materialization_allowed": False,
            "source_to_candidate_edit_relation_may_be_presumed": False,
            "gate_pass_requires_direct_evidence": True,
            "missing_ambiguous_or_conflicting_evidence_passes_gate": False,
            "all_required_gates_must_pass_before_separate_qualification_decision_may_be_requested": True,
            "all_required_gates_passing_automatically_qualifies_dataset": False,
            "separate_user_authority_required_for_qualification_or_counting": True,
        }.items():
            _expect(semantics, key, value, DEC022_AMENDMENT_PATH, issues, "DEC022_SEMANTICS")

    gates = amendment.get("fail_closed_gate_map")
    if not isinstance(gates, Mapping):
        _issue(issues, "DEC022_GATE_MAP", DEC022_AMENDMENT_PATH, "fail_closed_gate_map must be a mapping")
    else:
        _expect(gates, "required_gate_ids_exactly", DEC022_REQUIRED_GATE_IDS, DEC022_AMENDMENT_PATH, issues, "DEC022_GATE_MAP")
        for key, value in {
            "independent_axis_count": 17,
            "initial_status_for_every_gate": "NOT_RUN",
            "unknown_or_not_run_gate_is_pass": False,
            "target_power_minimum": 0.8,
            "confidence_level": 0.95,
            "maximum_full_ci_width": 0.3,
            "strict_universe_gate_and_reject_closure_gate_are_independently_adjudicated": True,
            "strict_universe_and_reject_closure_do_not_receive_duplicate_scientific_credit": True,
        }.items():
            _expect(gates, key, value, DEC022_AMENDMENT_PATH, issues, "DEC022_GATE_MAP")

    projection = amendment.get("authorization_projection")
    if not isinstance(projection, Mapping):
        _issue(issues, "DEC022_PROJECTION", DEC022_AMENDMENT_PATH, "authorization_projection must be a mapping")
    else:
        for key, value in {
            "changes_current_qualified_counts": False,
            "current_qualified_independent_ordinary_studies": 1,
            "current_qualified_a1_studies": 1,
            "current_qualified_true_a2_dense_studies": 0,
            "current_canonical_record_count": 6547,
            "gse256185_ordinary_study_contribution": 0,
            "gse256185_a1_study_contribution": 0,
            "gse256185_true_a2_dense_study_contribution": 0,
            "gse256185_canonical_record_count": 0,
            "gse256185_qualified": False,
            "phase_complete": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "qualifier_execution_allowed": False,
            "canonical_materialization_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }.items():
            _expect(projection, key, value, DEC022_AMENDMENT_PATH, issues, "DEC022_PROJECTION")

    runtime = amendment.get("runtime_successor")
    if not isinstance(runtime, Mapping):
        _issue(issues, "DEC022_RUNTIME", DEC022_AMENDMENT_PATH, "runtime_successor must be a mapping")
    else:
        _expect_closed_mapping(
            runtime,
            {
                "latest_settled_runtime_event_id": "A1-EVT-053",
                "settled_runtime_state_changed_by_authority_bytes": False,
                "runtime_event_emitted_by_authority_bytes": False,
                "runtime_sync_status": "PENDING_FRESH_EVENT_AFTER_SETTLED_EVT_053",
                "expected_next_runtime_event_id": DEC022_PENDING_RUNTIME_EVENT_ID,
                "next_runtime_event_id_preallocated": False,
                "scientific_state_change_expected": False,
            },
            DEC022_AMENDMENT_PATH,
            issues,
            "DEC022_RUNTIME",
        )

    q_scope = qualification.get("scope")
    if isinstance(q_scope, Mapping):
        _expect(q_scope, "aggregate_row_level_qualification_preflight_only_dataset_ids", ["GSE256185"], A1_QUALIFICATION_CONFIG_PATH, issues, "DEC022_A1_SCOPE")
        if "GSE256185" in set(q_scope.get("included_dataset_ids", [])):
            _issue(issues, "DEC022_A1_SCOPE", A1_QUALIFICATION_CONFIG_PATH, "GSE256185 must remain outside the qualification included set")
    else:
        _issue(issues, "DEC022_A1_SCOPE", A1_QUALIFICATION_CONFIG_PATH, "scope must be a mapping")

    q = qualification.get("dec022_aggregate_row_level_qualification_preflight_authority")
    if not isinstance(q, Mapping):
        _issue(issues, "DEC022_A1_POLICY", A1_QUALIFICATION_CONFIG_PATH, "DEC022 authority must be a mapping")
    else:
        for key, value in {
            "dataset_id": "GSE256185",
            "role": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
            "predecessor_decision_id": "V3-DEC-021",
            "predecessor_runtime_event_id": "A1-EVT-053",
            "allowed_input_field_classes_exactly": expected_input_fields,
            "allowed_output_class": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
            "required_fail_closed_gate_ids_exactly": DEC022_REQUIRED_GATE_IDS,
            "independent_gate_axis_count": 17,
            "initial_status_for_every_gate": "NOT_RUN",
            "unknown_or_not_run_gate_is_pass": False,
            "all_required_gates_must_pass": True,
            "all_gates_passing_automatically_qualifies_dataset": False,
            "source_to_candidate_edit_relation_may_be_presumed": False,
            "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
            "changes_current_qualified_counts": False,
            "dataset_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
            "member_identifier_output_allowed": False,
            "row_output_allowed": False,
            "sequence_output_allowed": False,
            "row_effect_output_allowed": False,
            "replicate_identifier_output_allowed": False,
            "split_assignment_output_allowed": False,
            "qualification_allowed": False,
            "canonical_materialization_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
            "runtime_sync_status": "PENDING_FRESH_EVENT_AFTER_SETTLED_EVT_053",
            "expected_next_runtime_event_id": DEC022_PENDING_RUNTIME_EVENT_ID,
            "next_runtime_event_id_preallocated": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }.items():
            _expect(q, key, value, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC022_A1_POLICY")

    root_policy = config.get("a1_qualification_authority")
    root = root_policy.get("gse256185_aggregate_row_level_qualification_preflight") if isinstance(root_policy, Mapping) else None
    if not isinstance(root, Mapping):
        _issue(issues, "DEC022_ROOT_POLICY", CONFIG_PATH, "DEC022 root policy must be a mapping")
    else:
        for key, value in {
            "current_status": "AUTHORIZED_NOT_RUN",
            "role": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
            "predecessor_decision_id": "V3-DEC-021",
            "predecessor_runtime_event_id": "A1-EVT-053",
            "strict_single_parent_pool_count": 634,
            "strict_candidate_member_count": 7292,
            "dual_parent_group_count_excluded": 15,
            "nonstrict_grammar_record_count_excluded": 2,
            "reasoned_family_closure_candidate_count": 7294,
            "reasoned_family_closure_included": False,
            "allowed_input_field_classes_exactly": expected_input_fields,
            "allowed_output_class": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
            "required_fail_closed_gate_ids_exactly": DEC022_REQUIRED_GATE_IDS,
            "independent_gate_axis_count": 17,
            "source_to_candidate_edit_relation_may_be_presumed": False,
            "dataset_qualification_decision_allowed": False,
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_dense_study_contribution": 0,
            "canonical_record_count": 0,
            "qualification_allowed": False,
            "canonical_materialization_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
        }.items():
            _expect(root, key, value, CONFIG_PATH, issues, "DEC022_ROOT_POLICY")

    data = registries["data"]
    if "GSE256185" in set(data.get("ordinary_candidate_dataset_ids", [])) or "GSE256185" in set(data.get("true_a2_recovery_candidate_dataset_ids", [])):
        _issue(issues, "DEC022_DATA_ROLE", REGISTRY_PATHS["data"], "GSE256185 may not receive candidate-set promotion")
    row = _mapping_entry(data.get("datasets"), "dataset_id", "GSE256185")
    if not isinstance(row, Mapping):
        _issue(issues, "DEC022_DATA_ROLE", REGISTRY_PATHS["data"], "GSE256185 row is required")
    else:
        for key, value in {
            "intended_role": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
            "mapping_status": "DEC022_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_AUTHORIZED_NOT_RUN",
            "allowed_input_field_classes_exactly": expected_input_fields,
            "allowed_output_class": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
            "strict_single_parent_pool_count": 634,
            "strict_candidate_member_count": 7292,
            "dual_parent_group_count_excluded": 15,
            "nonstrict_grammar_record_count_excluded": 2,
            "reasoned_family_closure_candidate_count": 7294,
            "reasoned_family_closure_included": False,
            "required_fail_closed_gate_ids_exactly": DEC022_REQUIRED_GATE_IDS,
            "independent_gate_axis_count": 17,
            "every_gate_initial_status": "NOT_RUN",
            "unknown_or_not_run_gate_is_pass": False,
            "source_to_candidate_edit_relation_may_be_presumed": False,
            "all_required_gates_passing_automatically_qualifies_dataset": False,
            "qualified": False,
            "ordinary_gate_contribution": 0,
            "a1_gate_contribution": 0,
            "true_a2_gate_contribution": 0,
            "canonical_record_count": 0,
            "row_output_allowed": False,
            "sequence_output_allowed": False,
            "effect_output_allowed": False,
            "replicate_identifier_output_allowed": False,
            "split_assignment_output_allowed": False,
            "qualification_allowed": False,
            "canonical_materialization_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
            "evidence_status": "NOT_RUN",
        }.items():
            _expect(row, key, value, REGISTRY_PATHS["data"], issues, "DEC022_DATA_ROLE")
    return issues


def validate_dec023_authority(
    repo_root: Path,
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    """Freeze both DEC023 aggregate-only preflights without any promotion."""

    issues: list[Issue] = []
    for relative, expected_sha256 in DEC023_ACTIVE_AUTHORITY_LEAF_SHA256.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "DEC023_ACTIVE_AUTHORITY_LEAF_UNREADABLE", relative, str(exc))
            continue
        allowed_sha256 = {
            expected_sha256,
            DEC028_ACTIVE_AUTHORITY_LEAF_SHA256.get(relative, expected_sha256),
            DEC027_ACTIVE_AUTHORITY_LEAF_SHA256.get(
                relative,
                DEC024_ACTIVE_AUTHORITY_LEAF_SHA256.get(relative, expected_sha256),
            ),
        }
        if actual_sha256 not in allowed_sha256:
            _issue(
                issues,
                "DEC023_ACTIVE_AUTHORITY_LEAF_DRIFT",
                relative,
                f"active authority leaf hash {actual_sha256} must match the frozen DEC023 leaf or an accepted successor",
            )

    try:
        amendment = _load_yaml(repo_root, DEC023_AMENDMENT_PATH)
        qualification = _load_json(repo_root, A1_QUALIFICATION_CONFIG_PATH)
        interim = _load_yaml(repo_root, A1_INTERIM_PATH)
    except (FileNotFoundError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        _issue(issues, "DEC023_AUTHORITY_LOAD", DEC023_AMENDMENT_PATH, str(exc))
        return issues

    expected_metadata = {
        "schema_version": "1.0.0",
        "amendment_id": "MRNA_XEDITFLOW_ROUTE_A_V3_DEC023",
        "decision_id": "V3-DEC-023",
        "contract_id": CONTRACT_ID,
        "contract_version": VERSION,
        "amends_contract_path": GOAL_PATH,
        "amends_contract_sha256": SOURCE_CONTRACT_SHA256,
        "predecessor_amendment_path": DEC022_AMENDMENT_PATH,
        "predecessor_amendment_sha256": DEC022_ACTIVE_AUTHORITY_LEAF_SHA256[DEC022_AMENDMENT_PATH],
        "predecessor_authority_head": "ae8e730d726754466e5c914d7ff962377607ac50",
        "amendment_mode": "APPEND_ONLY_DUAL_PREFLIGHT_AUTHORITY_COMPANION_ROOT_CONTRACT_AND_DEC001_THROUGH_DEC022_HISTORY_UNCHANGED",
        "status": "FROZEN_USER_AUTHORIZED_DUAL_AGGREGATE_ONLY_PREFLIGHT_NO_PROMOTION",
        "effective_phase": "A1",
        "requires_user_authorization": True,
    }
    for key, value in expected_metadata.items():
        _expect(amendment, key, value, DEC023_AMENDMENT_PATH, issues, "DEC023_AMENDMENT_METADATA")
    _expect_closed_mapping(
        amendment.get("user_authorization") if isinstance(amendment.get("user_authorization"), Mapping) else {},
        {"status": "GRANTED", "received_at": "2026-08-14T10:05:00+08:00", "source": "ACTIVE_CODEX_THREAD_OWNER_CONFIRMATION"},
        DEC023_AMENDMENT_PATH,
        issues,
        "DEC023_USER_AUTHORIZATION",
    )

    gse261_input_fields = [
        "PUBLIC_IDENTIFIER",
        "ASSET_SAMPLE_RUN_ROLE",
        "CONTEXT_METADATA",
        "HEADER_NAME",
        "ASSET_DIMENSION",
        "AGGREGATE_COUNT",
        "AGGREGATE_BYTE_COUNT",
        "LICENSE_NOTICE",
    ]
    gse261 = amendment.get("gse261709_public_schema_geometry_scope")
    if not isinstance(gse261, Mapping):
        _issue(issues, "DEC023_GSE261709_SCOPE", DEC023_AMENDMENT_PATH, "GSE261709 scope must be a mapping")
    else:
        for key, value in {
            "dataset_id": "GSE261709",
            "project_id": "PRJNA1088465",
            "role": "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_PREFLIGHT_ONLY",
            "authority_surface": "ORDINARY_PUBLIC_ONLY",
            "current_status": "AUTHORIZED_NOT_RUN",
            "preflight_execution_allowed": True,
            "allowed_input_field_classes_exactly": gse261_input_fields,
            "allowed_output_class": "AGGREGATE_IDENTIFIER_ASSET_SCHEMA_AND_GEOMETRY_ONLY",
            "allowed_aggregate_outputs_exactly": [
                "ASSET_SAMPLE_RUN_ROLE_COUNTS",
                "HEADER_ROLE_CLASS_COVERAGE_COUNTS",
                "ASSET_DIMENSION_COUNTS",
                "AGGREGATE_RECORD_AND_BYTE_COUNTS",
                "CONTEXT_STRATUM_COUNTS",
                "LICENSE_NOTICE_STATUS",
                "PREFLIGHT_GATE_STATUS_AND_REASON_COUNTS",
            ],
            "asset_body_read_allowed": False,
            "member_payload_read_allowed": False,
            "barcode_read_allowed": False,
            "variant_read_allowed": False,
            "transcript_read_allowed": False,
            "sequence_read_allowed": False,
            "row_effect_read_allowed": False,
            "standard_error_read_allowed": False,
            "member_identifier_output_allowed": False,
            "member_payload_output_allowed": False,
            "actual_header_names_output_allowed": False,
            "header_role_class_coverage_count_output_allowed": True,
            "row_output_allowed": False,
            "sequence_output_allowed": False,
            "row_effect_output_allowed": False,
            "standard_error_output_allowed": False,
            "split_assignment_output_allowed": False,
            "member_or_body_read_count_required": 0,
            "member_or_body_output_count_required": 0,
            "all_preflight_gates_passing_automatically_authorizes_row_level_access": False,
            "separate_user_authority_required_for_any_row_or_member_access": True,
        }.items():
            _expect(gse261, key, value, DEC023_AMENDMENT_PATH, issues, "DEC023_GSE261709_SCOPE")

    gse207_input_fields = [
        "DESIGNED_OBSERVED_MAPPING",
        "SOURCE_SEQUENCE",
        "CANDIDATE_SEQUENCE",
        "TWO_FIVE_EIGHT_HOUR_REPLICATE_ABUNDANCE",
        "NECESSARY_CONTEXT",
    ]
    gse207 = amendment.get("gse207584_dense_family_scope")
    if not isinstance(gse207, Mapping):
        _issue(issues, "DEC023_GSE207584_SCOPE", DEC023_AMENDMENT_PATH, "GSE207584 scope must be a mapping")
    else:
        for key, value in {
            "dataset_id": "GSE207584",
            "project_id": "PRJNA856272",
            "role": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
            "registry_role_must_remain": "AUDIT_ONLY",
            "current_status": "AUTHORIZED_NOT_RUN",
            "aggregate_dense_family_qualification_preflight_execution_allowed": True,
            "authority_surface": "ORDINARY_PUBLIC_ONLY",
            "allowed_internal_input_field_classes_exactly": gse207_input_fields,
            "allowed_output_class": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
            "required_fail_closed_gate_ids_exactly": DEC023_GSE207584_REQUIRED_GATE_IDS,
            "independent_gate_axis_count": 11,
            "initial_status_for_every_gate": "NOT_RUN",
            "unknown_or_not_run_gate_is_pass": False,
            "source_to_candidate_edit_relation_may_be_presumed": False,
            "intended_membership_may_be_inferred_from_detected_all_timepoints_subset": False,
            "biological_replicate_count_may_be_inflated_by_timepoints_or_technical_units": False,
            "three_biological_replicates_required": True,
            "missing_or_censored_measurement_may_be_treated_as_zero": False,
            "split_assignment_execution_allowed": False,
            "aggregate_prefrozen_power_planning_calculation_allowed": True,
            "aggregate_prefrozen_power_planning_alternative_spearman_rho": 0.25,
            "aggregate_prefrozen_power_planning_method": "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN",
            "aggregate_prefrozen_power_planning_confidence_interval_method": "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE",
            "aggregate_prefrozen_power_planning_null_standard_error_formula": "1/sqrt(n-3)",
            "aggregate_prefrozen_power_planning_alternative_standard_error_formula": "sqrt(1+rho^2/2)/sqrt(n-3)",
            "aggregate_prefrozen_power_planning_working_distribution_assumption": "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO",
            "aggregate_prefrozen_power_planning_alpha_two_sided": 0.05,
            "aggregate_prefrozen_power_planning_target_power": 0.8,
            "aggregate_prefrozen_power_planning_confidence_level": 0.95,
            "aggregate_prefrozen_power_planning_maximum_full_ci_width": 0.3,
            "aggregate_prefrozen_power_planning_required_effective_n_for_both_power_and_ci_width": 156,
            "aggregate_prefrozen_power_planning_analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
            "aggregate_prefrozen_power_planning_output_class": "AGGREGATE_ONLY",
            "formal_qualification_power_gate_execution_allowed": False,
            "member_identifier_output_allowed": False,
            "sequence_output_allowed": False,
            "row_abundance_output_allowed": False,
            "row_effect_output_allowed": False,
            "row_slope_output_allowed": False,
            "row_standard_error_output_allowed": False,
            "split_assignment_output_allowed": False,
            "all_required_gates_passing_automatically_qualifies_dataset": False,
            "separate_user_authority_required_for_qualification_or_counting": True,
        }.items():
            _expect(gse207, key, value, DEC023_AMENDMENT_PATH, issues, "DEC023_GSE207584_SCOPE")

    no_promotion = amendment.get("shared_no_promotion_boundary")
    if not isinstance(no_promotion, Mapping):
        _issue(issues, "DEC023_NO_PROMOTION", DEC023_AMENDMENT_PATH, "shared no-promotion boundary must be a mapping")
    else:
        for key, value in {
            "changes_current_qualified_counts": False,
            "current_qualified_independent_ordinary_studies": 1,
            "current_qualified_a1_studies": 1,
            "current_qualified_true_a2_dense_studies": 0,
            "current_canonical_record_count": 6547,
            "gse261709_qualified": False,
            "gse207584_qualified": False,
            "gse261709_enters_included_dataset_ids": False,
            "gse261709_enters_ordinary_candidate_dataset_ids": False,
            "gse261709_enters_true_a2_recovery_candidate_dataset_ids": False,
            "gse207584_enters_true_a2_recovery_candidate_dataset_ids": False,
            "dataset_qualification_decision_allowed": False,
            "qualifier_execution_allowed": False,
            "canonical_materialization_allowed": False,
            "split_execution_allowed": False,
            "formal_qualification_power_gate_execution_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "a7_allowed": False,
            "next_phase_authorized": False,
            "private_or_restricted_input_allowed": False,
            "sealed_contact_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }.items():
            _expect(no_promotion, key, value, DEC023_AMENDMENT_PATH, issues, "DEC023_NO_PROMOTION")
        for dataset in ("gse261709", "gse207584"):
            for suffix in ("ordinary_study_contribution", "a1_study_contribution", "true_a2_dense_study_contribution", "canonical_record_count"):
                _expect(no_promotion, f"{dataset}_{suffix}", 0, DEC023_AMENDMENT_PATH, issues, "DEC023_NO_PROMOTION")

    runtime = amendment.get("runtime_successor")
    _expect_closed_mapping(
        runtime if isinstance(runtime, Mapping) else {},
        {
            "latest_settled_runtime_event_id": "A1-EVT-055",
            "settled_runtime_state_changed_by_authority_bytes": False,
            "runtime_event_emitted_by_authority_bytes": False,
            "runtime_sync_status": "PENDING_FRESH_EVENT_AFTER_SETTLED_EVT_055",
            "expected_next_runtime_event_id": DEC023_PENDING_RUNTIME_EVENT_ID,
            "next_runtime_event_id_preallocated": False,
            "scientific_state_change_expected": False,
        },
        DEC023_AMENDMENT_PATH,
        issues,
        "DEC023_RUNTIME",
    )

    q_scope = qualification.get("scope")
    if not isinstance(q_scope, Mapping):
        _issue(issues, "DEC023_A1_SCOPE", A1_QUALIFICATION_CONFIG_PATH, "scope must be a mapping")
    else:
        _expect(q_scope, "public_identifier_asset_schema_and_aggregate_geometry_preflight_only_dataset_ids", ["GSE261709"], A1_QUALIFICATION_CONFIG_PATH, issues, "DEC023_A1_SCOPE")
        _expect(q_scope, "aggregate_dense_family_qualification_preflight_only_dataset_ids", ["GSE207584"], A1_QUALIFICATION_CONFIG_PATH, issues, "DEC023_A1_SCOPE")
        if "GSE261709" in set(q_scope.get("included_dataset_ids", [])):
            _issue(issues, "DEC023_A1_SCOPE", A1_QUALIFICATION_CONFIG_PATH, "GSE261709 must remain outside included_dataset_ids")
    q = qualification.get("dec023_dual_aggregate_only_preflight_authority")
    if not isinstance(q, Mapping):
        _issue(issues, "DEC023_A1_POLICY", A1_QUALIFICATION_CONFIG_PATH, "DEC023 authority must be a mapping")
    else:
        for key, value in {
            "latest_settled_runtime_event_id": "A1-EVT-055",
            "runtime_sync_status": "PENDING_FRESH_EVENT_AFTER_SETTLED_EVT_055",
            "expected_next_runtime_event_id": DEC023_PENDING_RUNTIME_EVENT_ID,
            "next_runtime_event_id_preallocated": False,
            "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
            "changes_current_qualified_counts": False,
            "qualification_allowed": False,
            "canonical_materialization_allowed": False,
            "split_execution_allowed": False,
            "formal_qualification_power_gate_execution_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "a7_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }.items():
            _expect(q, key, value, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC023_A1_POLICY")
        q261 = q.get("gse261709")
        q207 = q.get("gse207584")
        if not isinstance(q261, Mapping) or not isinstance(q207, Mapping):
            _issue(issues, "DEC023_A1_POLICY", A1_QUALIFICATION_CONFIG_PATH, "both dataset-specific DEC023 policies are required")
        else:
            for key, value in {
                "actual_header_names_output_allowed": False,
                "header_role_class_coverage_count_output_allowed": True,
            }.items():
                _expect(q261, key, value, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC023_A1_GSE261709")
            for key, value in {
                "aggregate_prefrozen_power_planning_calculation_allowed": True,
                "aggregate_prefrozen_power_planning_alternative_spearman_rho": 0.25,
                "aggregate_prefrozen_power_planning_method": "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN",
                "aggregate_prefrozen_power_planning_confidence_interval_method": "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE",
                "aggregate_prefrozen_power_planning_null_standard_error_formula": "1/sqrt(n-3)",
                "aggregate_prefrozen_power_planning_alternative_standard_error_formula": "sqrt(1+rho^2/2)/sqrt(n-3)",
                "aggregate_prefrozen_power_planning_working_distribution_assumption": "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO",
                "aggregate_prefrozen_power_planning_alpha_two_sided": 0.05,
                "aggregate_prefrozen_power_planning_target_power": 0.8,
                "aggregate_prefrozen_power_planning_confidence_level": 0.95,
                "aggregate_prefrozen_power_planning_maximum_full_ci_width": 0.3,
                "aggregate_prefrozen_power_planning_required_effective_n_for_both_power_and_ci_width": 156,
                "aggregate_prefrozen_power_planning_analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
                "aggregate_prefrozen_power_planning_output_class": "AGGREGATE_ONLY",
                "formal_qualification_power_gate_execution_allowed": False,
            }.items():
                _expect(q207, key, value, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC023_A1_GSE207584")

    root_policy = config.get("a1_qualification_authority")
    root261 = root_policy.get("gse261709_public_identifier_asset_schema_and_aggregate_geometry_preflight") if isinstance(root_policy, Mapping) else None
    root207 = root_policy.get("gse207584_aggregate_dense_family_qualification_preflight") if isinstance(root_policy, Mapping) else None
    if not isinstance(root261, Mapping) or not isinstance(root207, Mapping):
        _issue(issues, "DEC023_ROOT_POLICY", CONFIG_PATH, "both DEC023 root policies are required")
    else:
        for mapping, expected, code in (
            (root261, {"current_status": "AUTHORIZED_NOT_RUN", "dataset_id": "GSE261709", "project_id": "PRJNA1088465", "role": "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_PREFLIGHT_ONLY", "preflight_execution_allowed": True, "actual_header_names_output_allowed": False, "header_role_class_coverage_count_output_allowed": True, "asset_body_read_allowed": False, "member_payload_read_allowed": False, "member_or_body_read_count_required": 0, "member_or_body_output_count_required": 0, "row_level_access_allowed": False, "qualification_allowed": False, "split_execution_allowed": False, "power_execution_allowed": False, "training_allowed": False, "gpu_work_allowed": False, "model_selection_allowed": False, "a7_allowed": False, "next_phase_authorized": False}, "DEC023_ROOT_GSE261709"),
            (root207, {"current_status": "AUTHORIZED_NOT_RUN", "dataset_id": "GSE207584", "project_id": "PRJNA856272", "role": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY", "aggregate_dense_family_qualification_preflight_execution_allowed": True, "registry_role_must_remain": "AUDIT_ONLY", "required_fail_closed_gate_ids_exactly": DEC023_GSE207584_REQUIRED_GATE_IDS, "independent_gate_axis_count": 11, "initial_status_for_every_gate": "NOT_RUN", "unknown_or_not_run_gate_is_pass": False, "source_to_candidate_edit_relation_may_be_presumed": False, "split_assignment_execution_allowed": False, "aggregate_prefrozen_power_planning_calculation_allowed": True, "aggregate_prefrozen_power_planning_alternative_spearman_rho": 0.25, "aggregate_prefrozen_power_planning_method": "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN", "aggregate_prefrozen_power_planning_confidence_interval_method": "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE", "aggregate_prefrozen_power_planning_null_standard_error_formula": "1/sqrt(n-3)", "aggregate_prefrozen_power_planning_alternative_standard_error_formula": "sqrt(1+rho^2/2)/sqrt(n-3)", "aggregate_prefrozen_power_planning_working_distribution_assumption": "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO", "aggregate_prefrozen_power_planning_alpha_two_sided": 0.05, "aggregate_prefrozen_power_planning_target_power": 0.8, "aggregate_prefrozen_power_planning_confidence_level": 0.95, "aggregate_prefrozen_power_planning_maximum_full_ci_width": 0.3, "aggregate_prefrozen_power_planning_required_effective_n_for_both_power_and_ci_width": 156, "aggregate_prefrozen_power_planning_analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP", "aggregate_prefrozen_power_planning_output_class": "AGGREGATE_ONLY", "formal_qualification_power_gate_execution_allowed": False, "qualification_allowed": False, "training_allowed": False, "gpu_work_allowed": False, "model_selection_allowed": False, "a7_allowed": False, "next_phase_authorized": False}, "DEC023_ROOT_GSE207584"),
        ):
            for key, value in expected.items():
                _expect(mapping, key, value, CONFIG_PATH, issues, code)

    data = registries["data"]
    if "GSE261709" in set(data.get("ordinary_candidate_dataset_ids", [])) or "GSE261709" in set(data.get("true_a2_recovery_candidate_dataset_ids", [])):
        _issue(issues, "DEC023_DATA_ROLE", REGISTRY_PATHS["data"], "GSE261709 may not enter ordinary or true-A2 candidate sets")
    if "GSE207584" in set(data.get("true_a2_recovery_candidate_dataset_ids", [])):
        _issue(issues, "DEC023_DATA_ROLE", REGISTRY_PATHS["data"], "GSE207584 may not enter the true-A2 candidate set")
    row261 = _mapping_entry(data.get("datasets"), "dataset_id", "GSE261709")
    row207 = _mapping_entry(data.get("datasets"), "dataset_id", "GSE207584")
    if not isinstance(row261, Mapping) or not isinstance(row207, Mapping):
        _issue(issues, "DEC023_DATA_ROLE", REGISTRY_PATHS["data"], "both DEC023 dataset rows are required")
    else:
        for row, expected, code in (
            (row261, {"aliases": ["PRJNA1088465"], "role": "AUDIT_ONLY", "qualified": False, "intended_role": "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_PREFLIGHT_ONLY", "mapping_status": "DEC023_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_PREFLIGHT_AUTHORIZED_NOT_RUN", "allowed_input_field_classes_exactly": gse261_input_fields, "allowed_output_class": "AGGREGATE_IDENTIFIER_ASSET_SCHEMA_AND_GEOMETRY_ONLY", "actual_header_names_output_allowed": False, "header_role_class_coverage_count_output_allowed": True, "asset_body_read_allowed": False, "member_payload_read_allowed": False, "member_or_body_read_count_required": 0, "member_or_body_output_count_required": 0, "row_level_access_allowed": False, "ordinary_gate_contribution": 0, "a1_gate_contribution": 0, "true_a2_gate_contribution": 0, "canonical_record_count": 0, "qualification_allowed": False, "training_allowed": False, "gpu_work_allowed": False, "model_selection_allowed": False, "a7_allowed": False, "next_phase_authorized": False}, "DEC023_DATA_GSE261709"),
            (row207, {"aliases": ["ICODON", "PRJNA856272"], "role": "AUDIT_ONLY", "qualified": False, "intended_role": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY", "mapping_status": "DEC023_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_AUTHORIZED_NOT_RUN", "allowed_internal_input_field_classes_exactly": gse207_input_fields, "allowed_output_class": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY", "required_fail_closed_gate_ids_exactly": DEC023_GSE207584_REQUIRED_GATE_IDS, "independent_gate_axis_count": 11, "every_gate_initial_status": "NOT_RUN", "unknown_or_not_run_gate_is_pass": False, "source_to_candidate_edit_relation_may_be_presumed": False, "split_assignment_execution_allowed": False, "aggregate_prefrozen_power_planning_calculation_allowed": True, "aggregate_prefrozen_power_planning_alternative_spearman_rho": 0.25, "aggregate_prefrozen_power_planning_method": "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN", "aggregate_prefrozen_power_planning_confidence_interval_method": "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE", "aggregate_prefrozen_power_planning_null_standard_error_formula": "1/sqrt(n-3)", "aggregate_prefrozen_power_planning_alternative_standard_error_formula": "sqrt(1+rho^2/2)/sqrt(n-3)", "aggregate_prefrozen_power_planning_working_distribution_assumption": "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO", "aggregate_prefrozen_power_planning_alpha_two_sided": 0.05, "aggregate_prefrozen_power_planning_target_power": 0.8, "aggregate_prefrozen_power_planning_confidence_level": 0.95, "aggregate_prefrozen_power_planning_maximum_full_ci_width": 0.3, "aggregate_prefrozen_power_planning_required_effective_n_for_both_power_and_ci_width": 156, "aggregate_prefrozen_power_planning_analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP", "aggregate_prefrozen_power_planning_output_class": "AGGREGATE_ONLY", "formal_qualification_power_gate_execution_allowed": False, "ordinary_gate_contribution": 0, "a1_gate_contribution": 0, "true_a2_gate_contribution": 0, "canonical_record_count": 0, "qualification_allowed": False, "training_allowed": False, "gpu_work_allowed": False, "model_selection_allowed": False, "a7_allowed": False, "next_phase_authorized": False}, "DEC023_DATA_GSE207584"),
        ):
            for key, value in expected.items():
                _expect(row, key, value, REGISTRY_PATHS["data"], issues, code)

    disposition = interim.get("dec023_current_disposition")
    if not isinstance(disposition, Mapping):
        _issue(issues, "DEC023_INTERIM", A1_INTERIM_PATH, "DEC023 interim disposition must be a mapping")
    else:
        for key, value in {
            "decision_id": "V3-DEC-023",
            "status": "FROZEN_USER_AUTHORIZED_DUAL_AGGREGATE_ONLY_PREFLIGHT_NO_PROMOTION",
            "authority_only_not_study_qualification": True,
            "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
            "changes_current_qualified_counts": False,
            "qualification_allowed": False,
            "canonical_materialization_allowed": False,
            "split_execution_allowed": False,
            "formal_qualification_power_gate_execution_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "a7_allowed": False,
            "next_phase_authorized": False,
            "latest_settled_runtime_event_id": DEC023_CURRENT_RUNTIME_EVENT_ID,
            "settled_runtime_event_changed": True,
            "runtime_event_emitted": True,
            "runtime_sync_status": "SYNCED_EVT_057",
            "expected_next_runtime_event_id": DEC023_CURRENT_RUNTIME_EVENT_ID,
            "next_runtime_event_id_preallocated": False,
        }.items():
            _expect(disposition, key, value, A1_INTERIM_PATH, issues, "DEC023_INTERIM")
        interim261 = disposition.get("gse261709")
        interim207 = disposition.get("gse207584")
        if not isinstance(interim261, Mapping) or not isinstance(interim207, Mapping):
            _issue(issues, "DEC023_INTERIM", A1_INTERIM_PATH, "both dataset-specific DEC023 dispositions are required")
        else:
            for key, value in {
                "current_status": "STOP_PREFLIGHT_GATES_NOT_CLOSED",
                "actual_header_names_output_allowed": False,
                "header_role_class_coverage_count_output_allowed": True,
            }.items():
                _expect(interim261, key, value, A1_INTERIM_PATH, issues, "DEC023_INTERIM_GSE261709")
            for key, value in {
                "current_status": "STOP_CURRENT_PROTOCOL_NOT_QUALIFIED",
                "aggregate_prefrozen_power_planning_calculation_allowed": True,
                "aggregate_prefrozen_power_planning_alternative_spearman_rho": 0.25,
                "aggregate_prefrozen_power_planning_method": "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN",
                "aggregate_prefrozen_power_planning_confidence_interval_method": "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE",
                "aggregate_prefrozen_power_planning_null_standard_error_formula": "1/sqrt(n-3)",
                "aggregate_prefrozen_power_planning_alternative_standard_error_formula": "sqrt(1+rho^2/2)/sqrt(n-3)",
                "aggregate_prefrozen_power_planning_working_distribution_assumption": "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO",
                "aggregate_prefrozen_power_planning_alpha_two_sided": 0.05,
                "aggregate_prefrozen_power_planning_target_power": 0.8,
                "aggregate_prefrozen_power_planning_confidence_level": 0.95,
                "aggregate_prefrozen_power_planning_maximum_full_ci_width": 0.3,
                "aggregate_prefrozen_power_planning_required_effective_n_for_both_power_and_ci_width": 156,
                "aggregate_prefrozen_power_planning_analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
                "aggregate_prefrozen_power_planning_output_class": "AGGREGATE_ONLY",
                "formal_qualification_power_gate_execution_allowed": False,
            }.items():
                _expect(interim207, key, value, A1_INTERIM_PATH, issues, "DEC023_INTERIM_GSE207584")
        evidence_registration = disposition.get("evidence_registration")
        _expect_closed_mapping(
            evidence_registration if isinstance(evidence_registration, Mapping) else {},
            {
                "integration_id": DEC023_DUAL_PREFLIGHT_EVIDENCE_INTEGRATION_ID,
                "registered_lineage_ids_exactly": [
                    GSE261709_PREFLIGHT_LINEAGE_ID,
                    GSE207584_PREFLIGHT_LINEAGE_ID,
                ],
                "predecessor_runtime_event_id": DEC023_AUTHORITY_RUNTIME_EVENT_ID,
                "expected_next_runtime_event_id": DEC023_CURRENT_RUNTIME_EVENT_ID,
                "next_runtime_event_id_preallocated": False,
                "runtime_sync_status": "SYNCED_EVT_057",
                "runtime_event_emitted": True,
            },
            A1_INTERIM_PATH,
            issues,
            "DEC023_EVIDENCE_RUNTIME_BOUNDARY",
        )
    return issues


def validate_dec024_authority(
    repo_root: Path,
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    """Freeze three ordinary-public aggregate-only successor preflights."""

    issues: list[Issue] = []
    for relative, historical_sha256 in DEC024_ACTIVE_AUTHORITY_LEAF_SHA256.items():
        expected_sha256 = DEC028_ACTIVE_AUTHORITY_LEAF_SHA256.get(
            relative,
            DEC027_ACTIVE_AUTHORITY_LEAF_SHA256.get(relative, historical_sha256),
        )
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "DEC024_ACTIVE_AUTHORITY_LEAF_UNREADABLE", relative, str(exc))
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "DEC024_ACTIVE_AUTHORITY_LEAF_DRIFT",
                relative,
                f"active authority leaf hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        amendment = _load_yaml(repo_root, DEC024_AMENDMENT_PATH)
        qualification = _load_json(repo_root, A1_QUALIFICATION_CONFIG_PATH)
        interim = _load_yaml(repo_root, A1_INTERIM_PATH)
    except (FileNotFoundError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        _issue(issues, "DEC024_AUTHORITY_LOAD", DEC024_AMENDMENT_PATH, str(exc))
        return issues

    expected_top_keys = {
        "schema_version",
        "amendment_id",
        "decision_id",
        "contract_id",
        "contract_version",
        "amends_contract_path",
        "amends_contract_sha256",
        "predecessor_amendment_path",
        "predecessor_amendment_sha256",
        "predecessor_authority_head",
        "amendment_mode",
        "status",
        "effective_phase",
        "requires_user_authorization",
        "user_authorization",
        "gse261709_processed_a1_scope",
        "gse269595_role_adjudication_scope",
        "emtab10902_replacement_true_a2_scope",
        "shared_no_promotion_boundary",
        "strategic_nonbinding_route_note",
        "runtime_successor",
        "historical_preservation",
    }
    if set(amendment) != expected_top_keys:
        _issue(
            issues,
            "DEC024_AMENDMENT_CLOSURE",
            DEC024_AMENDMENT_PATH,
            f"top-level keys must be exactly {sorted(expected_top_keys)!r}",
        )
    for key, value in {
        "schema_version": "1.0.0",
        "amendment_id": "MRNA_XEDITFLOW_ROUTE_A_V3_DEC024",
        "decision_id": "V3-DEC-024",
        "contract_id": CONTRACT_ID,
        "contract_version": VERSION,
        "amends_contract_path": GOAL_PATH,
        "amends_contract_sha256": SOURCE_CONTRACT_SHA256,
        "predecessor_amendment_path": DEC023_AMENDMENT_PATH,
        "predecessor_amendment_sha256": DEC023_ACTIVE_AUTHORITY_LEAF_SHA256[
            DEC023_AMENDMENT_PATH
        ],
        "predecessor_authority_head": "e5d089a43d194caf59369fd12c203c0694ba40c6",
        "amendment_mode": "APPEND_ONLY_REPLACEMENT_PREFLIGHT_AUTHORITY_COMPANION_ROOT_CONTRACT_AND_DEC001_THROUGH_DEC023_HISTORY_UNCHANGED",
        "status": "FROZEN_USER_AUTHORIZED_REPLACEMENT_PREFLIGHT_ONLY_NO_PROMOTION",
        "effective_phase": "A1",
        "requires_user_authorization": True,
    }.items():
        _expect(amendment, key, value, DEC024_AMENDMENT_PATH, issues, "DEC024_AMENDMENT_METADATA")
    _expect_closed_mapping(
        amendment.get("user_authorization")
        if isinstance(amendment.get("user_authorization"), Mapping)
        else {},
        {
            "status": "GRANTED",
            "received_at": "2026-08-14T17:12:00+08:00",
            "source": "ACTIVE_CODEX_THREAD_OWNER_AUTONOMY_AND_REPLACEMENT_PREFLIGHT_DIRECTIVE",
        },
        DEC024_AMENDMENT_PATH,
        issues,
        "DEC024_USER_AUTHORIZATION",
    )

    gse261_inputs = [
        "PUBLIC_PROCESSED_ASSET_IDENTIFIER_AND_ROLE",
        "BARCODE_TO_ALLELE_TRANSCRIPT_SOURCE_MAPPING",
        "SOURCE_AND_CANDIDATE_SEQUENCE",
        "FULL_CONSTRUCT_AND_REPORTER_CONTEXT",
        "RNA_AND_DNA_THREE_BIOLOGICAL_REPLICATE_COUNTS",
        "ENDPOINT_EFFECT_AND_STANDARD_ERROR_FIELDS",
        "MISSINGNESS_CENSORING_AND_QC_STATUS",
        "LICENSE_AND_REUSE_NOTICE",
        "NECESSARY_CONTEXT",
    ]
    gse269_inputs = [
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
    ]
    emtab_inputs = [
        "PUBLIC_IDENTIFIER",
        "SOURCE_ANCHOR_AND_SOURCE_FAMILY_ROLE",
        "SOURCE_AND_CANDIDATE_SEQUENCE",
        "FULL_REPORTER_AND_ASSAY_CONTEXT",
        "DESIGNED_OBSERVED_MAPPING",
        "ENDPOINT_DIRECTION_SCALE_AND_UNIT",
        "BIOLOGICAL_REPLICATE_AND_STANDARD_ERROR",
        "MISSINGNESS_CENSORING_AND_QC_STATUS",
        "LICENSE_REUSE_AND_EXPOSURE_CONTEXT",
    ]
    future_use = "SCRATCH_ONLY_NO_FOUNDATION_EXPOSURE_NO_MODEL_INPUT_UNTIL_QUALIFIED"
    zero_contribution = {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0}
    current_counts = {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}

    scope_specs = (
        (
            "gse261709_processed_a1_scope",
            {
                "dataset_id": "GSE261709",
                "project_id": "PRJNA1088465",
                "predecessor_decision_id": "V3-DEC-023",
                "replacement_candidate_role": "REPLACEMENT_A1_CANDIDATE_PREFLIGHT_ONLY",
                "role": "AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_ONLY",
                "registry_role_must_remain": "AUDIT_ONLY",
                "current_status": "AUTHORIZED_NOT_RUN",
                "authority_surface": "ORDINARY_PUBLIC_PROCESSED_ASSET_ONLY",
                "aggregate_row_level_a1_qualification_preflight_execution_allowed": True,
                "allowed_internal_input_field_classes_exactly": gse261_inputs,
                "allowed_output_class": "AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_ONLY",
                "required_fail_closed_gate_ids_exactly": DEC024_GSE261709_REQUIRED_GATE_IDS,
                "independent_gate_axis_count": 12,
                "initial_status_for_every_gate": "NOT_RUN",
                "unknown_or_not_run_gate_is_pass": False,
                "source_to_candidate_edit_relation_may_be_presumed": False,
                "full_construct_and_source_join_required": True,
                "minimum_distinct_candidates_per_source_family": 3,
                "three_biological_replicates_required": True,
                "technical_units_may_substitute_for_biological_replicates": False,
                "processed_public_asset_body_read_allowed": True,
                "raw_fastq_or_sra_member_payload_read_allowed": False,
                "raw_archive_run_member_open_allowed": False,
                "split_assignment_execution_allowed": False,
                "formal_qualification_power_gate_execution_allowed": False,
                "future_use_route": future_use,
                "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
                "unknown_historical_exposure_is_gate_blocker": True,
                "member_identifier_output_allowed": False,
                "barcode_output_allowed": False,
                "sequence_output_allowed": False,
                "row_effect_output_allowed": False,
                "row_standard_error_output_allowed": False,
                "split_assignment_output_allowed": False,
                "all_required_gates_passing_automatically_qualifies_dataset": False,
                "separate_user_authority_required_for_qualification_or_counting": True,
            },
            "DEC024_GSE261709_SCOPE",
        ),
        (
            "gse269595_role_adjudication_scope",
            {
                "dataset_id": "GSE269595",
                "project_id": "PRJNA1122592",
                "replacement_candidate_role": "REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_ONLY",
                "role": "AGGREGATE_SOURCE_FAMILY_ASSET_SCHEMA_GEOMETRY_AND_ROLE_ADJUDICATION_PREFLIGHT_ONLY",
                "registry_role_must_remain": "AUDIT_ONLY",
                "current_status": "AUTHORIZED_NOT_RUN",
                "authority_surface": "ORDINARY_PUBLIC_ONLY",
                "aggregate_preflight_execution_allowed": True,
                "allowed_internal_input_field_classes_exactly": gse269_inputs,
                "allowed_output_class": "AGGREGATE_GATE_COUNTS_HISTOGRAMS_ROLE_STATUS_ONLY",
                "required_fail_closed_gate_ids_exactly": DEC024_GSE269595_REQUIRED_GATE_IDS,
                "independent_gate_axis_count": 13,
                "initial_status_for_every_gate": "NOT_RUN",
                "unknown_or_not_run_gate_is_pass": False,
                "a1_role_may_be_presumed": False,
                "true_a2_role_may_be_presumed": False,
                "maximum_roles_if_later_qualified": 1,
                "a1_and_true_a2_double_credit_allowed": False,
                "intronic_apa_exclusion_required": True,
                "source_to_candidate_edit_relation_may_be_presumed": False,
                "minimum_distinct_candidates_per_source_family": 3,
                "split_assignment_execution_allowed": False,
                "formal_qualification_power_gate_execution_allowed": False,
                "future_use_route": future_use,
                "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
                "unknown_historical_exposure_is_gate_blocker": True,
                "member_identifier_output_allowed": False,
                "sequence_output_allowed": False,
                "row_effect_output_allowed": False,
                "row_standard_error_output_allowed": False,
                "split_assignment_output_allowed": False,
                "all_required_gates_passing_automatically_qualifies_dataset": False,
                "separate_user_authority_required_for_role_assignment_qualification_or_counting": True,
            },
            "DEC024_GSE269595_SCOPE",
        ),
        (
            "emtab10902_replacement_true_a2_scope",
            {
                "dataset_id": "E-MTAB-10902",
                "public_alias": "N_ZIP",
                "replacement_candidate_role": "REPLACEMENT_TRUE_A2_CANDIDATE_PREFLIGHT_ONLY",
                "role": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
                "registry_role_must_remain": "AUDIT_ONLY",
                "current_status": "AUTHORIZED_NOT_RUN",
                "authority_surface": "ORDINARY_PUBLIC_ONLY",
                "aggregate_dense_family_qualification_preflight_execution_allowed": True,
                "allowed_internal_input_field_classes_exactly": emtab_inputs,
                "allowed_output_class": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
                "required_fail_closed_gate_ids_exactly": DEC024_EMTAB10902_REQUIRED_GATE_IDS,
                "independent_gate_axis_count": 11,
                "initial_status_for_every_gate": "NOT_RUN",
                "unknown_or_not_run_gate_is_pass": False,
                "source_anchor_may_be_inferred_from_row_order": False,
                "source_to_candidate_edit_relation_may_be_presumed": False,
                "full_reporter_context_required": True,
                "minimum_distinct_candidates_per_source_family": 3,
                "reported_source_group_count_approximate": 16,
                "reported_source_group_count_is_qualification_fact": False,
                "reported_qc_design_row_count_reference_only": 5679,
                "reported_qc_design_row_count_may_substitute_for_independent_source_group_n": False,
                "row_candidate_or_barcode_count_may_substitute_for_independent_source_group_n": False,
                "prefrozen_required_effective_n_for_power_and_full_ci_width": 156,
                "power_infeasible_status_allowed": True,
                "power_infeasible_status_is_qualification_or_credit": False,
                "split_assignment_execution_allowed": False,
                "formal_qualification_power_gate_execution_allowed": False,
                "future_use_route": future_use,
                "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
                "unknown_historical_exposure_is_gate_blocker": True,
                "member_identifier_output_allowed": False,
                "sequence_output_allowed": False,
                "row_effect_output_allowed": False,
                "row_standard_error_output_allowed": False,
                "split_assignment_output_allowed": False,
                "all_required_gates_passing_automatically_qualifies_dataset": False,
                "true_a2_status_may_be_presumed": False,
                "separate_user_authority_required_for_qualification_or_counting": True,
            },
            "DEC024_EMTAB10902_SCOPE",
        ),
    )
    for section_name, expected, code in scope_specs:
        section = amendment.get(section_name)
        if not isinstance(section, Mapping):
            _issue(issues, code, DEC024_AMENDMENT_PATH, f"{section_name} must be a mapping")
            continue
        for key, value in expected.items():
            _expect(section, key, value, DEC024_AMENDMENT_PATH, issues, code)

    shared = amendment.get("shared_no_promotion_boundary")
    expected_shared = {
        "changes_current_qualified_counts": False,
        "current_qualified_independent_ordinary_studies": 1,
        "current_qualified_a1_studies": 1,
        "current_qualified_true_a2_dense_studies": 0,
        "current_canonical_record_count": 6547,
        "gse261709_contribution": zero_contribution,
        "gse269595_contribution": zero_contribution,
        "emtab10902_contribution": zero_contribution,
        "gse261709_qualified": False,
        "gse269595_qualified": False,
        "emtab10902_qualified": False,
        "gse261709_enters_existing_qualification_candidate_lists": False,
        "gse269595_enters_existing_qualification_candidate_lists": False,
        "emtab10902_enters_existing_qualification_candidate_lists": False,
        "original_gate_minima": {"ordinary": 3, "a1": 2, "true_a2": 1},
        "gate_threshold_relaxation_authorized": False,
        "dataset_role_assignment_allowed": False,
        "dataset_qualification_decision_allowed": False,
        "qualifier_execution_allowed": False,
        "canonical_materialization_allowed": False,
        "split_execution_allowed": False,
        "formal_qualification_power_gate_execution_allowed": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "private_or_restricted_input_allowed": False,
        "sealed_contact_allowed": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    _expect_closed_mapping(
        shared if isinstance(shared, Mapping) else {},
        expected_shared,
        DEC024_AMENDMENT_PATH,
        issues,
        "DEC024_NO_PROMOTION_BOUNDARY",
    )
    for section_name, expected, code in (
        (
            "strategic_nonbinding_route_note",
            {
                "gse261709_dec023_geometry_scope_rewritten": False,
                "gse261709_dec024_processed_successor_added": True,
                "possible_future_combination": "EXISTING_GSE261709_AS_A1_PLUS_GSE269595_AS_TRUE_A2",
                "possible_future_combination_is_current_role_assignment_or_qualification": False,
                "separate_formal_qualification_authority_required_for_each_dataset": True,
                "double_credit_allowed": False,
            },
            "DEC024_STRATEGIC_NOTE",
        ),
        (
            "runtime_successor",
            {
                "latest_settled_runtime_event_id": "A1-EVT-057",
                "settled_runtime_state_changed_by_authority_bytes": False,
                "runtime_event_emitted_by_authority_bytes": False,
                "runtime_sync_status": "PENDING_FRESH_EVENT_AFTER_SETTLED_EVT_057",
                "expected_next_runtime_event_id": DEC024_PENDING_RUNTIME_EVENT_ID,
                "next_runtime_event_id_preallocated": False,
                "scientific_state_change_expected": False,
            },
            "DEC024_RUNTIME_BOUNDARY",
        ),
        (
            "historical_preservation",
            {
                "root_contract_bytes_changed": False,
                "dec001_through_dec023_history_changed": False,
                "dec023_amendment_bytes_changed": False,
                "predecessor_authority_head_rewritten": False,
                "prior_runtime_events_changed": False,
                "evt057_settled_state_changed": False,
                "existing_failure_evidence_deleted_or_relabelled": False,
                "existing_dataset_qualification_or_credit_changed": False,
                "g0_or_g1_gate_relaxation_added": False,
            },
            "DEC024_HISTORICAL_PRESERVATION",
        ),
    ):
        observed = amendment.get(section_name)
        _expect_closed_mapping(
            observed if isinstance(observed, Mapping) else {},
            expected,
            DEC024_AMENDMENT_PATH,
            issues,
            code,
        )

    root_policy = config.get("a1_qualification_authority")
    if not isinstance(root_policy, Mapping):
        _issue(issues, "DEC024_ROOT_POLICY", CONFIG_PATH, "a1_qualification_authority must be a mapping")
        root_policy = {}
    root_specs = (
        (
            "gse261709_dec024_processed_a1_qualification_preflight",
            "GSE261709",
            "PRJNA1088465",
            "AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_ONLY",
            gse261_inputs,
            DEC024_GSE261709_REQUIRED_GATE_IDS,
            12,
            "DEC024_ROOT_GSE261709",
        ),
        (
            "gse269595_dec024_role_adjudication_preflight",
            "GSE269595",
            "PRJNA1122592",
            "AGGREGATE_SOURCE_FAMILY_ASSET_SCHEMA_GEOMETRY_AND_ROLE_ADJUDICATION_PREFLIGHT_ONLY",
            gse269_inputs,
            DEC024_GSE269595_REQUIRED_GATE_IDS,
            13,
            "DEC024_ROOT_GSE269595",
        ),
        (
            "emtab10902_dec024_true_a2_preflight",
            "E-MTAB-10902",
            None,
            "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
            emtab_inputs,
            DEC024_EMTAB10902_REQUIRED_GATE_IDS,
            11,
            "DEC024_ROOT_EMTAB10902",
        ),
    )
    for policy_name, dataset_id, project_id, role, input_fields, gates, axis_count, code in root_specs:
        policy = root_policy.get(policy_name) if isinstance(root_policy, Mapping) else None
        if not isinstance(policy, Mapping):
            _issue(issues, code, CONFIG_PATH, f"{policy_name} must be a mapping")
            continue
        expected = {
            "current_status": "AUTHORIZED_NOT_RUN",
            "amendment_path": DEC024_AMENDMENT_PATH,
            "dataset_id": dataset_id,
            "role": role,
            "aggregate_preflight_execution_allowed": True,
            "allowed_internal_input_field_classes_exactly": input_fields,
            "required_fail_closed_gate_ids_exactly": gates,
            "independent_gate_axis_count": axis_count,
            "initial_status_for_every_gate": "NOT_RUN",
            "unknown_or_not_run_gate_is_pass": False,
            "source_to_candidate_edit_relation_may_be_presumed": False,
            "split_assignment_execution_allowed": False,
            "formal_qualification_power_gate_execution_allowed": False,
            "future_use_route": future_use,
            "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
            "unknown_historical_exposure_is_gate_blocker": True,
            "member_identifier_output_allowed": False,
            "sequence_output_allowed": False,
            "row_effect_output_allowed": False,
            "row_standard_error_output_allowed": False,
            "split_assignment_output_allowed": False,
            "qualification_allowed": False,
            "canonical_materialization_allowed": False,
            "contribution": zero_contribution,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "a7_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
        }
        if project_id is not None:
            expected["project_id"] = project_id
        for key, value in expected.items():
            _expect(policy, key, value, CONFIG_PATH, issues, code)
    root261 = root_policy.get("gse261709_dec024_processed_a1_qualification_preflight")
    if isinstance(root261, Mapping):
        for key, value in {
            "authority_surface": "ORDINARY_PUBLIC_PROCESSED_ASSET_ONLY",
            "minimum_distinct_candidates_per_source_family": 3,
            "full_construct_and_source_join_required": True,
            "three_biological_replicates_required": True,
            "processed_public_asset_body_read_allowed": True,
            "raw_fastq_or_sra_member_payload_read_allowed": False,
        }.items():
            _expect(root261, key, value, CONFIG_PATH, issues, "DEC024_ROOT_GSE261709")
    root269 = root_policy.get("gse269595_dec024_role_adjudication_preflight")
    if isinstance(root269, Mapping):
        for key, value in {
            "replacement_candidate_role": "REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_ONLY",
            "a1_role_may_be_presumed": False,
            "true_a2_role_may_be_presumed": False,
            "maximum_roles_if_later_qualified": 1,
            "a1_and_true_a2_double_credit_allowed": False,
            "intronic_apa_exclusion_required": True,
        }.items():
            _expect(root269, key, value, CONFIG_PATH, issues, "DEC024_ROOT_GSE269595")
    root_nzip = root_policy.get("emtab10902_dec024_true_a2_preflight")
    if isinstance(root_nzip, Mapping):
        for key, value in {
            "replacement_candidate_role": "REPLACEMENT_TRUE_A2_CANDIDATE_PREFLIGHT_ONLY",
            "full_reporter_context_required": True,
            "reported_source_group_count_approximate": 16,
            "reported_qc_design_row_count_reference_only": 5679,
            "reported_qc_design_row_count_may_substitute_for_independent_source_group_n": False,
            "prefrozen_required_effective_n_for_power_and_full_ci_width": 156,
            "power_infeasible_status_allowed": True,
            "power_infeasible_status_is_qualification_or_credit": False,
        }.items():
            _expect(root_nzip, key, value, CONFIG_PATH, issues, "DEC024_ROOT_EMTAB10902")

    q_policy = qualification.get("dec024_replacement_preflight_authority")
    if not isinstance(q_policy, Mapping):
        _issue(issues, "DEC024_A1_POLICY", A1_QUALIFICATION_CONFIG_PATH, "DEC024 policy must be a mapping")
    else:
        for key, value in {
            "current_qualified_counts": current_counts,
            "changes_current_qualified_counts": False,
            "dataset_role_assignment_allowed": False,
            "qualification_allowed": False,
            "canonical_materialization_allowed": False,
            "split_execution_allowed": False,
            "formal_qualification_power_gate_execution_allowed": False,
            "member_identifier_output_allowed": False,
            "sequence_output_allowed": False,
            "row_effect_or_standard_error_output_allowed": False,
            "split_assignment_output_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "a7_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }.items():
            _expect(q_policy, key, value, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC024_A1_POLICY")
        for dataset_key, root_name, code in (
            ("gse261709", "gse261709_dec024_processed_a1_qualification_preflight", "DEC024_A1_GSE261709"),
            ("gse269595", "gse269595_dec024_role_adjudication_preflight", "DEC024_A1_GSE269595"),
            ("emtab10902", "emtab10902_dec024_true_a2_preflight", "DEC024_A1_EMTAB10902"),
        ):
            q_dataset = q_policy.get(dataset_key)
            root_dataset = root_policy.get(root_name) if isinstance(root_policy, Mapping) else None
            if not isinstance(q_dataset, Mapping) or not isinstance(root_dataset, Mapping):
                _issue(issues, code, A1_QUALIFICATION_CONFIG_PATH, f"{dataset_key} DEC024 policy must be a mapping")
                continue
            for key in (
                "role",
                "allowed_internal_input_field_classes_exactly",
                "required_fail_closed_gate_ids_exactly",
                "independent_gate_axis_count",
                "future_use_route",
                "historical_analytic_or_checkpoint_exposure",
                "unknown_historical_exposure_is_gate_blocker",
            ):
                _expect(q_dataset, key, root_dataset.get(key), A1_QUALIFICATION_CONFIG_PATH, issues, code)

    q_scope = qualification.get("scope")
    if not isinstance(q_scope, Mapping):
        _issue(issues, "DEC024_A1_SCOPE", A1_QUALIFICATION_CONFIG_PATH, "scope must be a mapping")
    else:
        for key, value in {
            "processed_row_level_a1_qualification_preflight_only_dataset_ids": ["GSE261709"],
            "replacement_a1_or_true_a2_role_adjudication_preflight_only_dataset_ids": ["GSE269595"],
            "replacement_true_a2_dense_family_preflight_only_dataset_ids": ["E-MTAB-10902"],
            "training_allowed": False,
            "model_selection_allowed": False,
        }.items():
            _expect(q_scope, key, value, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC024_A1_SCOPE")

    data = registries["data"]
    for key, value in {
        "replacement_a1_processed_preflight_candidate_dataset_ids": ["GSE261709"],
        "replacement_a1_or_true_a2_role_adjudication_preflight_candidate_dataset_ids": ["GSE269595"],
        "replacement_true_a2_preflight_candidate_dataset_ids": ["E-MTAB-10902"],
    }.items():
        _expect(data, key, value, REGISTRY_PATHS["data"], issues, "DEC024_DATA_CANDIDATE_SCOPE")
    forbidden_promotions = {"GSE261709", "GSE269595", "E-MTAB-10902"}
    if forbidden_promotions & set(data.get("ordinary_candidate_dataset_ids", [])):
        _issue(issues, "DEC024_DATA_PROMOTION", REGISTRY_PATHS["data"], "DEC024 preflight candidates may not enter ordinary candidate credit")
    if {"GSE269595", "E-MTAB-10902"} & set(data.get("true_a2_recovery_candidate_dataset_ids", [])):
        _issue(issues, "DEC024_DATA_PROMOTION", REGISTRY_PATHS["data"], "DEC024 role/true-A2 preflight candidates may not enter true-A2 candidate credit")

    row_specs = (
        (
            "GSE261709",
            {
                "role": "AUDIT_ONLY",
                "qualified": False,
                "dec024_successor_role": "AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_ONLY",
                "dec024_allowed_internal_input_field_classes_exactly": gse261_inputs,
                "dec024_required_fail_closed_gate_ids_exactly": DEC024_GSE261709_REQUIRED_GATE_IDS,
                "dec024_independent_gate_axis_count": 12,
                "dec024_every_gate_initial_status": "NOT_RUN",
                "dec024_unknown_or_not_run_gate_is_pass": False,
                "dec024_processed_public_asset_body_read_allowed": True,
                "dec024_raw_fastq_or_sra_member_payload_read_allowed": False,
                "dec024_source_to_candidate_edit_relation_may_be_presumed": False,
                "future_use_route": future_use,
                "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
                "unknown_historical_exposure_is_gate_blocker": True,
                "dec024_member_identifier_output_allowed": False,
                "dec024_barcode_output_allowed": False,
                "dec024_sequence_output_allowed": False,
                "dec024_row_effect_output_allowed": False,
                "dec024_row_standard_error_output_allowed": False,
                "dec024_split_assignment_output_allowed": False,
            },
            "DEC024_DATA_GSE261709",
        ),
        (
            "GSE269595",
            {
                "role": "AUDIT_ONLY",
                "qualified": False,
                "intended_role": "REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_ONLY",
                "preflight_candidate_only_not_counting": True,
                "allowed_internal_input_field_classes_exactly": gse269_inputs,
                "required_fail_closed_gate_ids_exactly": DEC024_GSE269595_REQUIRED_GATE_IDS,
                "independent_gate_axis_count": 13,
                "every_gate_initial_status": "NOT_RUN",
                "unknown_or_not_run_gate_is_pass": False,
                "a1_role_may_be_presumed": False,
                "true_a2_role_may_be_presumed": False,
                "maximum_roles_if_later_qualified": 1,
                "a1_and_true_a2_double_credit_allowed": False,
                "intronic_apa_exclusion_required": True,
                "future_use_route": future_use,
                "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
                "unknown_historical_exposure_is_gate_blocker": True,
                "role_assignment_allowed": False,
            },
            "DEC024_DATA_GSE269595",
        ),
        (
            "E-MTAB-10902",
            {
                "role": "AUDIT_ONLY",
                "qualified": False,
                "intended_role": "REPLACEMENT_TRUE_A2_CANDIDATE_PREFLIGHT_ONLY",
                "preflight_candidate_only_not_counting": True,
                "allowed_internal_input_field_classes_exactly": emtab_inputs,
                "required_fail_closed_gate_ids_exactly": DEC024_EMTAB10902_REQUIRED_GATE_IDS,
                "independent_gate_axis_count": 11,
                "every_gate_initial_status": "NOT_RUN",
                "unknown_or_not_run_gate_is_pass": False,
                "full_reporter_context_required": True,
                "reported_source_group_count_approximate": 16,
                "reported_qc_design_row_count_reference_only": 5679,
                "reported_qc_design_row_count_may_substitute_for_independent_source_group_n": False,
                "row_candidate_or_barcode_count_may_substitute_for_independent_source_group_n": False,
                "prefrozen_required_effective_n_for_power_and_full_ci_width": 156,
                "power_infeasible_status_allowed": True,
                "power_infeasible_status_is_qualification_or_credit": False,
                "future_use_route": future_use,
                "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
                "unknown_historical_exposure_is_gate_blocker": True,
            },
            "DEC024_DATA_EMTAB10902",
        ),
    )
    for dataset_id, expected, code in row_specs:
        row = _mapping_entry(data.get("datasets"), "dataset_id", dataset_id)
        if not isinstance(row, Mapping):
            _issue(issues, code, REGISTRY_PATHS["data"], f"{dataset_id} row is required")
            continue
        for key, value in {
            **expected,
            "ordinary_gate_contribution": 0,
            "a1_gate_contribution": 0,
            "true_a2_gate_contribution": 0,
            "canonical_record_count": 0,
            "qualification_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "a7_allowed": False,
            "next_phase_authorized": False,
            "sealed_contact_allowed": False,
            "evidence_status": "NOT_RUN",
        }.items():
            _expect(row, key, value, REGISTRY_PATHS["data"], issues, code)

    task_boundary = registries["task"].get("dec024_replacement_preflight_boundaries")
    expected_task_boundary = {
        "latest_settled_runtime_event_id": "A1-EVT-057",
        "gse261709_role": "AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_ONLY",
        "gse261709_input_surface": "ORDINARY_PUBLIC_PROCESSED_ASSET_ONLY",
        "gse261709_raw_fastq_or_sra_member_payload_read_allowed": False,
        "gse269595_role": "REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_ONLY",
        "gse269595_maximum_roles_if_later_qualified": 1,
        "gse269595_double_credit_allowed": False,
        "emtab10902_role": "REPLACEMENT_TRUE_A2_CANDIDATE_PREFLIGHT_ONLY",
        "emtab10902_reported_source_group_count_approximate": 16,
        "emtab10902_reported_qc_design_rows_may_substitute_for_independent_n": False,
        "prefrozen_required_effective_n_for_power_and_full_ci_width": 156,
        "future_use_route": future_use,
        "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
        "unknown_historical_exposure_is_gate_blocker": True,
        "current_qualified_counts": current_counts,
        "changes_current_qualified_counts": False,
        "gate_threshold_relaxation_authorized": False,
        "dataset_role_assignment_allowed": False,
        "qualification_allowed": False,
        "canonical_materialization_allowed": False,
        "split_execution_allowed": False,
        "formal_qualification_power_gate_execution_allowed": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "sealed_contact_allowed": False,
    }
    _expect_closed_mapping(
        task_boundary if isinstance(task_boundary, Mapping) else {},
        expected_task_boundary,
        REGISTRY_PATHS["task"],
        issues,
        "DEC024_TASK_BOUNDARY",
    )

    disposition = interim.get("dec024_current_disposition")
    if not isinstance(disposition, Mapping):
        _issue(issues, "DEC024_INTERIM", A1_INTERIM_PATH, "DEC024 disposition must be a mapping")
    else:
        for key, value in {
            "decision_id": "V3-DEC-024",
            "status": "FROZEN_USER_AUTHORIZED_REPLACEMENT_PREFLIGHT_ONLY_NO_PROMOTION",
            "authority_only_not_study_qualification": True,
            "current_qualified_counts": current_counts,
            "changes_current_qualified_counts": False,
            "original_gate_minima": {"ordinary": 3, "a1": 2, "true_a2": 1},
            "gate_threshold_relaxation_authorized": False,
            "dataset_role_assignment_allowed": False,
            "qualification_allowed": False,
            "canonical_materialization_allowed": False,
            "split_execution_allowed": False,
            "formal_qualification_power_gate_execution_allowed": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "a7_allowed": False,
            "next_phase_authorized": False,
            "latest_settled_runtime_event_id": DEC024_CURRENT_RUNTIME_EVENT_ID,
            "settled_runtime_event_changed": True,
            "runtime_event_emitted": True,
            "runtime_sync_status": "SYNCED_EVT_058",
            "expected_next_runtime_event_id": DEC024_CURRENT_RUNTIME_EVENT_ID,
            "next_runtime_event_id_preallocated": False,
            "strategic_nonbinding_possible_future_combination": "EXISTING_GSE261709_AS_A1_PLUS_GSE269595_AS_TRUE_A2",
            "strategic_combination_requires_separate_formal_qualification_authority": True,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "sealed_contact_allowed": False,
        }.items():
            _expect(disposition, key, value, A1_INTERIM_PATH, issues, "DEC024_INTERIM")
        for dataset_key, role_key, role_value, axis_count, code in (
            ("gse261709", "preflight_role", "AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_ONLY", 12, "DEC024_INTERIM_GSE261709"),
            ("gse269595", "preflight_role", "REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_ONLY", 13, "DEC024_INTERIM_GSE269595"),
            ("emtab10902", "preflight_role", "REPLACEMENT_TRUE_A2_CANDIDATE_PREFLIGHT_ONLY", 11, "DEC024_INTERIM_EMTAB10902"),
        ):
            dataset = disposition.get(dataset_key)
            if not isinstance(dataset, Mapping):
                _issue(issues, code, A1_INTERIM_PATH, f"{dataset_key} disposition must be a mapping")
                continue
            for key, value in {
                "registry_role": "AUDIT_ONLY",
                role_key: role_value,
                "current_status": "AUTHORIZED_NOT_RUN",
                "independent_gate_axis_count": axis_count,
                "future_use_route": future_use,
                "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
                "unknown_historical_exposure_is_gate_blocker": True,
                "member_identifier_output_allowed": False,
                "sequence_output_allowed": False,
                "row_effect_or_standard_error_output_allowed": False,
                "split_assignment_output_allowed": False,
                "qualified": False,
                "contribution": zero_contribution,
            }.items():
                _expect(dataset, key, value, A1_INTERIM_PATH, issues, code)
        nzip = disposition.get("emtab10902")
        if isinstance(nzip, Mapping):
            for key, value in {
                "reported_source_group_count_approximate": 16,
                "reported_qc_design_row_count_reference_only": 5679,
                "reported_qc_design_rows_may_substitute_for_independent_n": False,
                "prefrozen_required_effective_n_for_power_and_full_ci_width": 156,
                "power_infeasible_status_allowed": True,
                "power_infeasible_status_is_qualification_or_credit": False,
            }.items():
                _expect(nzip, key, value, A1_INTERIM_PATH, issues, "DEC024_INTERIM_EMTAB10902")
        role_adjudication = disposition.get("gse269595")
        if isinstance(role_adjudication, Mapping):
            for key, value in {
                "a1_role_may_be_presumed": False,
                "true_a2_role_may_be_presumed": False,
                "maximum_roles_if_later_qualified": 1,
                "a1_and_true_a2_double_credit_allowed": False,
                "intronic_apa_exclusion_required": True,
            }.items():
                _expect(role_adjudication, key, value, A1_INTERIM_PATH, issues, "DEC024_INTERIM_GSE269595")
    return issues


def validate_dec027_authority(
    repo_root: Path,
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    """Freeze the ordered six-route rescue sprint without awarding credit."""

    issues: list[Issue] = []
    for relative, historical_sha256 in DEC027_ACTIVE_AUTHORITY_LEAF_SHA256.items():
        expected_sha256 = DEC028_ACTIVE_AUTHORITY_LEAF_SHA256.get(
            relative, historical_sha256
        )
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "DEC027_ACTIVE_AUTHORITY_LEAF_UNREADABLE", relative, str(exc))
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "DEC027_ACTIVE_AUTHORITY_LEAF_DRIFT",
                relative,
                f"active DEC027 authority leaf hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        amendment = _load_yaml(repo_root, DEC027_AMENDMENT_PATH)
        qualification = _load_json(repo_root, A1_QUALIFICATION_CONFIG_PATH)
        interim = _load_yaml(repo_root, A1_INTERIM_PATH)
        decision_log = _load_yaml(repo_root, DECISION_LOG_PATH)
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        _issue(issues, "DEC027_AUTHORITY_LOAD", DEC027_AMENDMENT_PATH, str(exc))
        return issues

    for key, value in {
        "schema_version": "1.0.0",
        "amendment_id": "MRNA_XEDITFLOW_ROUTE_A_V3_DEC027",
        "decision_id": "V3-DEC-027",
        "contract_id": CONTRACT_ID,
        "contract_version": VERSION,
        "amends_contract_path": GOAL_PATH,
        "amends_contract_sha256": SOURCE_CONTRACT_SHA256,
        "predecessor_amendment_path": DEC024_AMENDMENT_PATH,
        "predecessor_amendment_sha256": DEC024_ACTIVE_AUTHORITY_LEAF_SHA256[DEC024_AMENDMENT_PATH],
        "predecessor_authority_head": "b1ca33d852bad111ff31b4f60493d8c43c63d1a3",
        "amendment_mode": "APPEND_ONLY_BOUNDED_DATA_RESCUE_SPRINT_AND_CONDITIONAL_CLAIM_LADDER_TRIGGER",
        "status": "FROZEN_USER_AUTHORIZED_RESCUE_PREFLIGHT_ONLY_NO_AUTOMATIC_PROMOTION",
        "effective_phase": "A1",
        "requires_user_authorization": True,
    }.items():
        _expect(amendment, key, value, DEC027_AMENDMENT_PATH, issues, "DEC027_AMENDMENT_METADATA")
    _expect_closed_mapping(
        amendment.get("user_authorization")
        if isinstance(amendment.get("user_authorization"), Mapping)
        else {},
        {
            "status": "GRANTED",
            "received_at": "2026-08-15T01:11:08+08:00",
            "source": "ACTIVE_CODEX_THREAD_OWNER_BOUNDED_RESCUE_SPRINT_AND_STOP_RULE_DIRECTIVE",
        },
        DEC027_AMENDMENT_PATH,
        issues,
        "DEC027_USER_AUTHORIZATION",
    )

    current_counts = {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}
    preserved = amendment.get("preserved_full_route_a_target")
    if not isinstance(preserved, Mapping):
        _issue(issues, "DEC027_FULL_ROUTE_TARGET", DEC027_AMENDMENT_PATH, "preserved target must be a mapping")
    else:
        for key, value in {
            "status": "RETAINED_AS_HIGHEST_TARGET_DURING_RESCUE",
            "required_qualified_ordinary_studies": 3,
            "required_qualified_a1_studies": 2,
            "required_qualified_true_a2_studies": 1,
            "current_qualified_counts": current_counts,
            "gate_threshold_relaxation_authorized_by_this_decision": False,
            "credit_may_be_inferred_from_preflight_status": False,
            "separate_evidence_based_promotion_authority_required": True,
        }.items():
            _expect(preserved, key, value, DEC027_AMENDMENT_PATH, issues, "DEC027_FULL_ROUTE_TARGET")

    ordered = amendment.get("ordered_rescue_sprint")
    if not isinstance(ordered, Mapping):
        _issue(issues, "DEC027_ORDER", DEC027_AMENDMENT_PATH, "ordered_rescue_sprint must be a mapping")
    else:
        for key, value in {
            "execution_order_exactly": DEC027_RESCUE_EXECUTION_ORDER,
            "all_six_terminal_reports_required_before_stop_rule_evaluation": True,
            "ordinary_public_data_only": True,
            "aggregate_output_only": True,
            "private_or_sealed_access_allowed": False,
            "row_or_member_identifier_output_allowed": False,
            "sequence_output_allowed": False,
            "row_abundance_effect_slope_or_standard_error_output_allowed": False,
            "split_assignment_output_allowed": False,
            "persistent_member_level_intermediate_allowed": False,
        }.items():
            _expect(ordered, key, value, DEC027_AMENDMENT_PATH, issues, "DEC027_ORDER")

    root_policy = config.get("a1_qualification_authority")
    root = root_policy.get("dec027_bounded_rescue_sprint") if isinstance(root_policy, Mapping) else None
    q_policy = qualification.get("dec027_bounded_rescue_sprint_authority")
    task = registries["task"].get("dec027_bounded_rescue_sprint_boundaries")
    data = registries["data"]
    data_routes = data.get("dec027_rescue_route_overrides")
    disposition = interim.get("dec027_current_disposition")
    surfaces = (
        (root, CONFIG_PATH, "DEC027_ROOT"),
        (q_policy, A1_QUALIFICATION_CONFIG_PATH, "DEC027_A1_POLICY"),
        (task, REGISTRY_PATHS["task"], "DEC027_TASK_BOUNDARY"),
        (data_routes, REGISTRY_PATHS["data"], "DEC027_DATA_ROUTES"),
        (disposition, A1_INTERIM_PATH, "DEC027_INTERIM"),
    )
    for surface, path, code in surfaces:
        if not isinstance(surface, Mapping):
            _issue(issues, code, path, "DEC027 policy surface must be a mapping")
        else:
            _expect(surface, "execution_order_exactly", DEC027_RESCUE_EXECUTION_ORDER, path, issues, code)

    amendment_gate_map: dict[str, list[Any]] = {}
    for route_id, (section_name, gate_count) in DEC027_ROUTE_SECTION_AND_GATE_COUNT.items():
        section = amendment.get(section_name)
        if not isinstance(section, Mapping):
            _issue(issues, "DEC027_ROUTE_SCOPE", DEC027_AMENDMENT_PATH, f"{section_name} must be a mapping")
            continue
        gates = section.get("required_fail_closed_gate_ids_exactly")
        if not isinstance(gates, list) or len(gates) != gate_count or len(set(gates)) != gate_count:
            _issue(issues, "DEC027_ROUTE_GATES", DEC027_AMENDMENT_PATH, f"{route_id} must contain {gate_count} unique exact gates")
            continue
        amendment_gate_map[route_id] = gates
        if isinstance(root, Mapping):
            root_gates = root.get("route_gate_ids_exactly")
            observed = root_gates.get(route_id) if isinstance(root_gates, Mapping) else None
            if observed != gates:
                _issue(issues, "DEC027_ROOT_GATE_MAP", CONFIG_PATH, f"{route_id} gate list must equal the amendment")
        if isinstance(q_policy, Mapping):
            q_gates = q_policy.get("route_gate_ids_exactly")
            observed = q_gates.get(route_id) if isinstance(q_gates, Mapping) else None
            if observed != gates:
                _issue(issues, "DEC027_A1_GATE_MAP", A1_QUALIFICATION_CONFIG_PATH, f"{route_id} gate list must equal the amendment")
            q_counts = q_policy.get("route_gate_counts_exactly")
            observed_count = q_counts.get(route_id) if isinstance(q_counts, Mapping) else None
            if observed_count != gate_count:
                _issue(issues, "DEC027_A1_GATE_COUNT", A1_QUALIFICATION_CONFIG_PATH, f"{route_id} gate count must remain {gate_count}")
        if isinstance(disposition, Mapping):
            counts = disposition.get("route_gate_counts_exactly")
            observed_count = counts.get(route_id) if isinstance(counts, Mapping) else None
            if observed_count != gate_count:
                _issue(issues, "DEC027_INTERIM_GATE_COUNT", A1_INTERIM_PATH, f"{route_id} gate count must remain {gate_count}")

    corrections = {
        "gse232572_dense_multi_candidate_true_a2_gate_applicability": "NOT_APPLICABLE_FOR_A1_REPLAY",
        "gse269595_a1_and_true_a2_double_credit_allowed": False,
        "gse269595_two_candidate_family_is_pairwise_only_not_global_dense_failure": True,
        "gse269595_finite_endpoint_required": True,
        "gse113849_context_rule_must_be_frozen_before_endpoint_or_power": True,
        "gse295080_row_level_qualification_execution_allowed": False,
    }
    for surface, path, code in surfaces:
        if isinstance(surface, Mapping):
            for key, value in corrections.items():
                if key in surface:
                    _expect(surface, key, value, path, issues, code)

    amendment232 = amendment.get("gse232572_corrected_a1_replay")
    if isinstance(amendment232, Mapping):
        _expect(amendment232, "dense_multi_candidate_true_a2_gate_applicability", "NOT_APPLICABLE_FOR_A1_REPLAY", DEC027_AMENDMENT_PATH, issues, "DEC027_GSE232572_CORRECTION")
        _expect(amendment232, "dense_multi_candidate_gate_may_block_a1_replay", False, DEC027_AMENDMENT_PATH, issues, "DEC027_GSE232572_CORRECTION")
    amendment269 = amendment.get("gse269595_corrected_role_adjudication_successor")
    if isinstance(amendment269, Mapping):
        _expect(amendment269, "a1_and_true_a2_double_credit_allowed", False, DEC027_AMENDMENT_PATH, issues, "DEC027_GSE269595_CORRECTION")
        _expect(amendment269, "finite_endpoint_required", True, DEC027_AMENDMENT_PATH, issues, "DEC027_GSE269595_CORRECTION")
        dense_policy = amendment269.get("eligible_dense_family_policy")
        if isinstance(dense_policy, Mapping):
            _expect(dense_policy, "two_candidate_families_are_pairwise_only_and_excluded_from_dense_universe", True, DEC027_AMENDMENT_PATH, issues, "DEC027_GSE269595_CORRECTION")
            _expect(dense_policy, "one_two_candidate_family_may_fail_all_other_eligible_dense_families", False, DEC027_AMENDMENT_PATH, issues, "DEC027_GSE269595_CORRECTION")
    amendment113 = amendment.get("gse113849_designed_snv_true_a2_preflight")
    if isinstance(amendment113, Mapping):
        _expect(amendment113, "context_rule_must_be_frozen_before_endpoint_or_power_evaluation", True, DEC027_AMENDMENT_PATH, issues, "DEC027_GSE113849_CORRECTION")
        _expect(amendment113, "outcome_or_power_may_select_context_rule", False, DEC027_AMENDMENT_PATH, issues, "DEC027_GSE113849_CORRECTION")
        _expect(amendment113, "randomized_absolute_library_is_true_a2", False, DEC027_AMENDMENT_PATH, issues, "DEC027_GSE113849_CORRECTION")
    amendment295 = amendment.get("gse295080_independence_overlap_adjudication")
    if isinstance(amendment295, Mapping):
        _expect(amendment295, "row_level_qualification_execution_allowed", False, DEC027_AMENDMENT_PATH, issues, "DEC027_GSE295080_BOUNDARY")

    expected_stop = {
        "evaluate_only_after_all_six_terminal_reports_registered": True,
        "minimum_qualified_counts_after_separate_promotions": {"ordinary": 2, "a1": 2, "true_a2": 0},
        "floor_failure_logic": "QUALIFIED_ORDINARY_LT_2_OR_QUALIFIED_A1_LT_2",
        "true_a2_reachable_post_dedup_n_minimum": 156,
        "true_a2_reachable_power_minimum": 0.8,
        "true_a2_reachable_maximum_full_ci_width": 0.3,
        "trigger_logic": "RESCUE_FLOOR_FAILED_AND_NO_TRUE_A2_ROUTE_WITH_REACHABLE_POST_DEDUP_N_AND_POWER",
        "triggered_successor": "SINGLE_STUDY_SOURCE_RELATIVE_DEVELOPMENT_PLUS_ENGINEERING_THEORY",
        "triggered_successor_requires_separate_append_only_amendment": True,
        "full_route_a_remains_highest_inactive_target": True,
    }
    for surface, path, code in ((root, CONFIG_PATH, "DEC027_ROOT_STOP_RULE"), (q_policy, A1_QUALIFICATION_CONFIG_PATH, "DEC027_A1_STOP_RULE")):
        observed = surface.get("stop_rule") if isinstance(surface, Mapping) else None
        _expect_closed_mapping(observed if isinstance(observed, Mapping) else {}, expected_stop, path, issues, code)
    if isinstance(disposition, Mapping):
        observed = disposition.get("stop_rule")
        interim_stop = dict(expected_stop)
        interim_stop["prospective_triggered_successor"] = interim_stop.pop("triggered_successor")
        _expect_closed_mapping(observed if isinstance(observed, Mapping) else {}, interim_stop, A1_INTERIM_PATH, issues, "DEC027_INTERIM_STOP_RULE")

    no_promotion = amendment.get("no_promotion_state")
    authority_expected_locks = {
        "current_qualified_counts": current_counts,
        "changes_current_qualified_counts": False,
        "dataset_role_assignment_allowed": False,
        "qualification_allowed": False,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "latest_settled_runtime_event_id": DEC024_CURRENT_RUNTIME_EVENT_ID,
        "runtime_event_emitted": False,
        "expected_next_runtime_event_id": DEC027_PENDING_RUNTIME_EVENT_ID,
        "next_runtime_event_id_preallocated": False,
        "sealed_contact": False,
    }
    if not isinstance(no_promotion, Mapping):
        _issue(issues, "DEC027_NO_PROMOTION", DEC027_AMENDMENT_PATH, "no_promotion_state must be a mapping")
    else:
        for key, value in authority_expected_locks.items():
            _expect(no_promotion, key, value, DEC027_AMENDMENT_PATH, issues, "DEC027_NO_PROMOTION")
    if isinstance(q_policy, Mapping):
        for key, value in authority_expected_locks.items():
            if key in q_policy:
                _expect(q_policy, key, value, A1_QUALIFICATION_CONFIG_PATH, issues, "DEC027_A1_NO_PROMOTION")
    if isinstance(disposition, Mapping):
        interim_expected_locks = {
            **authority_expected_locks,
            "latest_settled_runtime_event_id": DEC027_EVT060_CURRENT_RUNTIME_EVENT_ID,
            "runtime_event_emitted": True,
            "expected_next_runtime_event_id": DEC027_PENDING_RUNTIME_EVENT_ID,
        }
        for key, value in interim_expected_locks.items():
            if key in disposition:
                _expect(disposition, key, value, A1_INTERIM_PATH, issues, "DEC027_INTERIM_NO_PROMOTION")

    expected_external = ["GSE113849", "GSE295080"]
    _expect(data, "dec027_external_preflight_candidate_only_dataset_ids", expected_external, REGISTRY_PATHS["data"], issues, "DEC027_EXTERNAL_CANDIDATES")
    registered_ids = set(data.get("expected_dataset_ids", []))
    rows = data.get("datasets")
    if isinstance(rows, list):
        registered_ids |= {row.get("dataset_id") for row in rows if isinstance(row, Mapping)}
    if registered_ids & set(expected_external):
        _issue(issues, "DEC027_EXTERNAL_CANDIDATE_PROMOTION", REGISTRY_PATHS["data"], "external preflight candidates may not enter registered study units before qualification")

    decisions = decision_log.get("decisions")
    if not isinstance(decisions, list) or not any(
        isinstance(entry, Mapping) and entry.get("decision_id") == "V3-DEC-027"
        for entry in decisions
    ):
        _issue(issues, "DEC027_DECISION_LOG", DECISION_LOG_PATH, "DEC027 must remain in the append-only decision prefix")
    manifest_paths = {
        entry.get("path"): entry
        for entry in manifest.get("files", [])
        if isinstance(entry, Mapping)
    }
    entry = manifest_paths.get(DEC027_AMENDMENT_PATH)
    if not isinstance(entry, Mapping) or entry.get("role") != "DEC027_APPEND_ONLY_BOUNDED_RESCUE_SPRINT_AUTHORITY_AMENDMENT":
        _issue(issues, "DEC027_MANIFEST", REGISTRY_MANIFEST_PATH, "DEC027 amendment must be registered with its exact role")
    for key, value in {
        "active_amendment_decision_ids": ACTIVE_AMENDMENT_DECISION_IDS,
        "sealed_contact": False,
    }.items():
        _expect(manifest, key, value, REGISTRY_MANIFEST_PATH, issues, "DEC027_MANIFEST")
    return issues


def validate_dec028_authority(
    repo_root: Path,
    config: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    """Validate the owner-initiated Single-study S0/P0 authority without unlocking execution."""

    issues: list[Issue] = []
    for relative, expected_sha256 in DEC028_ACTIVE_AUTHORITY_LEAF_SHA256.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "DEC028_ACTIVE_AUTHORITY_LEAF_UNREADABLE", relative, str(exc))
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "DEC028_ACTIVE_AUTHORITY_LEAF_DRIFT",
                relative,
                f"active DEC028 authority leaf hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        amendment = _load_yaml(repo_root, DEC028_AMENDMENT_PATH)
        protocol = _load_yaml(repo_root, DEC028_PROTOCOL_PATH)
        execution = _load_yaml(repo_root, DEC028_EXECUTION_PATH)
        qualification = _load_json(repo_root, A1_QUALIFICATION_CONFIG_PATH)
        interim = _load_yaml(repo_root, A1_INTERIM_PATH)
        decision_log = _load_yaml(repo_root, DECISION_LOG_PATH)
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        _issue(issues, "DEC028_AUTHORITY_LOAD", DEC028_AMENDMENT_PATH, str(exc))
        return issues

    for key, value in {
        "schema_version": "1.0.0",
        "contract_id": CONTRACT_ID,
        "decision_id": "V3-DEC-028",
        "decision_type": "APPEND_ONLY_AUTHORITY_AMENDMENT",
        "dimension": "owner_initiated_single_study_source_relative_operational_mainline",
        "status": "ACTIVE_AUTHORITY_AMENDMENT",
        "effective_phase": "SINGLE_STUDY_S0_AUTHORITY_AND_P0_CLOSURE",
    }.items():
        _expect(amendment, key, value, DEC028_AMENDMENT_PATH, issues, "DEC028_METADATA")

    owner = amendment.get("owner_authorization")
    expected_owner = {
        "status": "GRANTED",
        "source": "ACTIVE_CODEX_THREAD_OWNER_CONTRACT_EXECUTION_DIRECTIVE",
        "choice": "OWNER_INITIATED_PROSPECTIVE_OPERATIONAL_MAINLINE_CHOICE_NOT_DEC027_AUTOMATIC_TRIGGER",
        "dec027_automatic_trigger_claimed": False,
        "historical_gate_rewrite_allowed": False,
        "gate_threshold_relaxation_claimed": False,
        "new_credit_claimed": False,
    }
    _expect_closed_mapping(
        owner if isinstance(owner, Mapping) else {},
        expected_owner,
        DEC028_AMENDMENT_PATH,
        issues,
        "DEC028_OWNER_AUTHORITY",
    )

    counts = {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}
    locks = {
        "changes_current_qualified_counts": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "dataset_promotion_allowed": False,
        "qualification_allowed": False,
        "canonical_materialization_allowed": False,
        "private_row_access_allowed": False,
        "split_assignment_execution_allowed": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "sealed_access_authorized": False,
    }
    authority_scope = amendment.get("authority_scope")
    if not isinstance(authority_scope, Mapping):
        _issue(issues, "DEC028_SCOPE", DEC028_AMENDMENT_PATH, "authority_scope must be a mapping")
    else:
        _expect(authority_scope, "scope", "PROSPECTIVE_OPERATIONAL_MAINLINE_ONLY", DEC028_AMENDMENT_PATH, issues, "DEC028_SCOPE")
        _expect(authority_scope, "creates_second_root_contract", False, DEC028_AMENDMENT_PATH, issues, "DEC028_SCOPE")
        _expect(authority_scope, "strategic_target", "FULL_ROUTE_A_RETAINED_HIGHEST_INACTIVE_TARGET", DEC028_AMENDMENT_PATH, issues, "DEC028_SCOPE")
        _expect(authority_scope, "full_route_a_required_counts", {"ordinary": 3, "a1": 2, "true_a2": 1}, DEC028_AMENDMENT_PATH, issues, "DEC028_SCOPE")
        _expect(authority_scope, "current_qualified_counts", counts, DEC028_AMENDMENT_PATH, issues, "DEC028_SCOPE")
        _expect(authority_scope, "historical_gate_statuses_preserved", True, DEC028_AMENDMENT_PATH, issues, "DEC028_SCOPE")
        for key, value in locks.items():
            _expect(authority_scope, key, value, DEC028_AMENDMENT_PATH, issues, "DEC028_LOCKS")

    expected_roles = {
        "GSE200304": "PRIMARY_SINGLE_STUDY_SOURCE_RELATIVE_DEVELOPMENT",
        "GSE232572": "DEVELOPMENT_ROBUSTNESS_ONLY",
        "ENCSR854RUF": "DEVELOPMENT_EXPOSURE_POSITIVE_STRESS_ONLY",
        "GSE217518": "DEVELOPMENT_FAIL_CLOSED_CROSSWALK_STRESS_ONLY",
        "GSE149487": "DEVELOPMENT_RECONSTRUCTION_REGRESSION_ONLY",
        "GSE269595": "EXPLORATORY_DENSE_EVALUATOR_ONLY",
        "GSE114002": "EXPLORATORY_WITHIN_ASSAY_ORDINAL_EVALUATOR_ONLY",
        "E-MTAB-10902": "EXPLORATORY_SMALL_N_DENSE_SMOKE_ONLY",
        "GSE261709": "NEGATIVE_CONTROL_PROVENANCE_IDENTIFIABILITY",
        "GSE207584": "NEGATIVE_CONTROL_MAPPING_AND_CARTESIAN_RECONSTRUCTION",
        "GSE256185": "NEGATIVE_CONTROL_POOL_PARSER_REJECT_QA",
        "GSE145046": "NEGATIVE_CONTROL_ABSOLUTE_VS_SOURCE_RELATIVE",
        "GSE186455": "NEGATIVE_CONTROL_LIBRARY_INDEPENDENCE_REFERENCE",
        "GSE246381": "SEALED_EXTERNAL_FINAL_ONLY",
    }
    _expect(amendment, "study_use_roles", expected_roles, DEC028_AMENDMENT_PATH, issues, "DEC028_STUDY_ROLES")
    data_roles = registries["data"].get("dec028_single_study_use_roles")
    if not isinstance(data_roles, Mapping):
        _issue(issues, "DEC028_STUDY_ROLES", REGISTRY_PATHS["data"], "dec028_single_study_use_roles must be a mapping")
    else:
        _expect(data_roles, "role_field_is_orthogonal_to_qualification_status_and_credit", True, REGISTRY_PATHS["data"], issues, "DEC028_STUDY_ROLES")
        _expect(data_roles, "current_credit_unchanged", counts, REGISTRY_PATHS["data"], issues, "DEC028_STUDY_ROLES")
        for dataset_id, role in expected_roles.items():
            _expect(data_roles, dataset_id, role, REGISTRY_PATHS["data"], issues, "DEC028_STUDY_ROLES")

    dependency = amendment.get("execution_dependency")
    if not isinstance(dependency, Mapping):
        _issue(issues, "DEC028_DEPENDENCY", DEC028_AMENDMENT_PATH, "execution_dependency must be a mapping")
    else:
        _expect(dependency, "p0_required_pass_count", 11, DEC028_AMENDMENT_PATH, issues, "DEC028_DEPENDENCY")
        _expect(dependency, "p0_nonpass_action", "STOP_BEFORE_DATA_ROWS_CUDA_MODEL_OPTIMIZER_CHECKPOINT_PARAMETER_UPDATE_OR_TRAINING", DEC028_AMENDMENT_PATH, issues, "DEC028_DEPENDENCY")
        _expect(dependency, "materialization_authority_granted", False, DEC028_AMENDMENT_PATH, issues, "DEC028_DEPENDENCY")
        _expect(dependency, "g1_development_run_authorized", False, DEC028_AMENDMENT_PATH, issues, "DEC028_DEPENDENCY")

    p0 = amendment.get("p0_successor_contract")
    if not isinstance(p0, Mapping):
        _issue(issues, "DEC028_P0", DEC028_AMENDMENT_PATH, "p0_successor_contract must be a mapping")
    else:
        _expect(p0, "predecessor_result_preserved", {"pass": 3, "fail_closed": 7, "unknown_not_asserted": 1}, DEC028_AMENDMENT_PATH, issues, "DEC028_P0")
        _expect(p0, "predecessor_overwrite_allowed", False, DEC028_AMENDMENT_PATH, issues, "DEC028_P0")
        _expect(p0, "exact_gate_count", 11, DEC028_AMENDMENT_PATH, issues, "DEC028_P0")
        _expect(p0, "all_status_fields_must_equal_pass", True, DEC028_AMENDMENT_PATH, issues, "DEC028_P0")
        _expect(p0, "p0_2_success_shape", {"status": "PASS", "scope": "DISCLOSED_EXPOSED_DEVELOPMENT_ONLY", "predecessor_historical_status": "UNKNOWN_NOT_ASSERTED"}, DEC028_AMENDMENT_PATH, issues, "DEC028_P0")
        _expect(p0, "p0_7_pre_materialization_scope", {"freezes": ["GROUPING_ALGORITHM", "COMPONENT_RULES", "SALT", "DEVELOPMENT_SUBROLE_CONTRACT"], "split_assignment_count": 0}, DEC028_AMENDMENT_PATH, issues, "DEC028_P0")

    future = amendment.get("future_learned_execution")
    expected_future = {
        "current_authorized_execution_count": 0,
        "future_run_id": "GSE200304_SOURCE_RELATIVE_CRITIC_G1",
        "future_run_role": "SOURCE_RELATIVE_CRITIC_NOT_A6_LEARNED_BASE_VALUE",
        "separate_authority_required": True,
        "authorized_execution_count_if_later_granted": 1,
        "optimizer_fit_count_if_later_granted": 1,
        "fold_model_count_if_later_granted": 1,
        "checkpoint_count_if_later_granted": 1,
        "final_refit_count_if_later_granted": 0,
        "seed_count_if_later_granted": 1,
        "nested_cross_validation_authorized": False,
        "a6_learned_base_value_execution_authorized": False,
    }
    _expect_closed_mapping(
        future if isinstance(future, Mapping) else {},
        expected_future,
        DEC028_AMENDMENT_PATH,
        issues,
        "DEC028_G1",
    )

    for surface, path in (
        (config.get("a1_qualification_authority", {}).get("dec028_single_study_operational_mainline", {}), CONFIG_PATH),
        (qualification.get("dec028_single_study_operational_mainline_authority", {}), A1_QUALIFICATION_CONFIG_PATH),
        (interim.get("dec028_current_disposition", {}), A1_INTERIM_PATH),
    ):
        if not isinstance(surface, Mapping):
            _issue(issues, "DEC028_SURFACE", path, "DEC028 operational surface must be a mapping")
            continue
        _expect(surface, "current_qualified_counts", counts, path, issues, "DEC028_SURFACE")
        _expect(surface, "scientific_claim_status", "NOT_ESTABLISHED", path, issues, "DEC028_SURFACE")
        for key in ("training_allowed", "gpu_work_allowed", "model_selection_allowed", "a7_allowed", "next_phase_authorized"):
            _expect(surface, key, False, path, issues, "DEC028_SURFACE")

    _expect(protocol, "status", "STANDARD_DEVELOPMENT_CRITIC_COMPLETED_NEGATIVE_RESULT_SS6_ENGINEERING_PASS", DEC028_PROTOCOL_PATH, issues, "DEC028_PROTOCOL")
    _expect(protocol, "primary_study_unit", "GSE200304", DEC028_PROTOCOL_PATH, issues, "DEC028_PROTOCOL")
    _expect(protocol, "canonical_record_count", 6547, DEC028_PROTOCOL_PATH, issues, "DEC028_PROTOCOL")
    _expect(execution, "record_status", "STANDARD_DEVELOPMENT_CRITIC_COMPLETED_NEGATIVE_RESULT_SS6_ENGINEERING_PASS", DEC028_EXECUTION_PATH, issues, "DEC028_EXECUTION")
    _expect(execution, "predecessor_runtime_event_id", "A1-EVT-061", DEC028_EXECUTION_PATH, issues, "DEC028_EXECUTION")
    _expect(execution, "expected_next_runtime_event_id", "PENDING_FRESH_RUNTIME_EVENT_ID", DEC028_EXECUTION_PATH, issues, "DEC028_EXECUTION")
    _expect(execution, "next_runtime_event_id_preallocated", False, DEC028_EXECUTION_PATH, issues, "DEC028_EXECUTION")
    learned = protocol.get("learned_execution", {})
    for key, expected in {
        "g1_authorized": False,
        "authority_consumed": True,
        "terminal_status": "TERMINATED_SAFELY_WITH_EVIDENCE_NO_RETRY",
        "authorized_execution_count": 1,
        "launched_execution_count": 1,
        "optimizer_fit_attempt_count": 1,
        "optimizer_fit_count": 0,
        "parameter_update_count": 0,
        "checkpoint_count": 0,
        "final_refit_count": 0,
        "retry_authorized": False,
    }.items():
        _expect(learned, key, expected, DEC028_PROTOCOL_PATH, issues, "DEC028_G1_SETTLEMENT")
    development_policy = protocol.get("standard_development_policy", {})
    for key, expected in {
        "status": "ACTIVE_OWNER_DIRECTED_STANDARD_DEVELOPMENT",
        "global_run_limit": None,
        "resource_failure_retry_allowed": True,
        "successor_required_after_resource_failure": False,
        "fixed_gpu_index_or_uuid_required": False,
        "membership_split_model_and_evaluator_unchanged": True,
    }.items():
        _expect(development_policy, key, expected, DEC028_PROTOCOL_PATH, issues, "DEC028_DEVELOPMENT_POLICY")
    development_result = protocol.get("development_result", {})
    for key, expected in {
        "status": "COMPLETED_NEGATIVE_NO_PREDICTIVE_SIGNAL_ESTABLISHED",
        "parameter_update_count": 72,
        "checkpoint_count": 1,
        "test_source_group_count": 981,
        "multi_candidate_source_group_count": 3,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }.items():
        _expect(development_result, key, expected, DEC028_PROTOCOL_PATH, issues, "DEC028_DEVELOPMENT_RESULT")
    if abs(float(development_result.get("test_spearman", float("nan"))) - 0.001882286575573072) > 1e-15:
        _issue(issues, "DEC028_DEVELOPMENT_RESULT", DEC028_PROTOCOL_PATH, "development test Spearman differs")
    if abs(float(development_result.get("test_mae", float("nan"))) - 0.13537058487266992) > 1e-15:
        _issue(issues, "DEC028_DEVELOPMENT_RESULT", DEC028_PROTOCOL_PATH, "development test MAE differs")
    _expect(protocol.get("ss6_nonlearned_engineering", {}), "status", "PASS_SS6_NONLEARNED_ENGINEERING_REFERENCE", DEC028_PROTOCOL_PATH, issues, "DEC028_SS6")
    _expect(protocol.get("ss8_claim_adjudication", {}), "scientific_claim_status", "NOT_ESTABLISHED", DEC028_PROTOCOL_PATH, issues, "DEC028_SS8")
    _expect(protocol.get("ss8_claim_adjudication", {}), "learned_result_disposition", "NEGATIVE_NO_PREDICTIVE_SIGNAL_ESTABLISHED", DEC028_PROTOCOL_PATH, issues, "DEC028_SS8")
    _expect(protocol.get("ss9_sealed_boundary", {}), "payload_access_count", 0, DEC028_PROTOCOL_PATH, issues, "DEC028_SS9")

    dec027 = interim.get("dec027_current_disposition")
    adjudication = dec027.get("stop_rule_adjudication") if isinstance(dec027, Mapping) else None
    if not isinstance(adjudication, Mapping):
        _issue(issues, "DEC028_DEC027_HISTORY", A1_INTERIM_PATH, "DEC027 stop-rule adjudication must remain present")
    else:
        _expect(adjudication, "trigger_condition_met", False, A1_INTERIM_PATH, issues, "DEC028_DEC027_HISTORY")
        _expect(adjudication, "conditional_successor_activated", False, A1_INTERIM_PATH, issues, "DEC028_DEC027_HISTORY")

    decisions = decision_log.get("decisions")
    if not isinstance(decisions, list) or not decisions or not isinstance(decisions[-1], Mapping) or decisions[-1].get("decision_id") != "V3-DEC-028":
        _issue(issues, "DEC028_DECISION_LOG", DECISION_LOG_PATH, "DEC028 must be the append-only final decision after DEC027")

    manifest_paths = {
        entry.get("path"): entry
        for entry in manifest.get("files", [])
        if isinstance(entry, Mapping)
    }
    expected_roles_by_path = {
        DEC028_AMENDMENT_PATH: "DEC028_APPEND_ONLY_SINGLE_STUDY_MAINLINE_AUTHORITY_AMENDMENT",
        DEC028_HUMAN_CONTRACT_PATH: "DEC028_REVIEWED_HUMAN_READABLE_SINGLE_STUDY_CONTRACT",
        DEC028_PROTOCOL_PATH: "DEC028_SINGLE_STUDY_MAINLINE_EXECUTABLE_PROTOCOL",
        DEC028_EXECUTION_PATH: "DEC028_SINGLE_STUDY_MAINLINE_EXECUTION_STATE",
    }
    for relative, role in expected_roles_by_path.items():
        entry = manifest_paths.get(relative)
        if not isinstance(entry, Mapping) or entry.get("role") != role:
            _issue(issues, "DEC028_MANIFEST", REGISTRY_MANIFEST_PATH, f"{relative} must be registered as {role}")
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
    """Bind current static leaves and the dynamic one-blocker D6 config."""

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
                "dynamic D6 config must remain outside the static manifest",
            )
        if not expected_static_paths.issubset(manifest_paths):
            _issue(
                issues,
                "GSE200304_DEC019_POST_ADJUDICATION_DAG",
                REGISTRY_MANIFEST_PATH,
                "all twenty-three current static leaves must be registered",
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


def validate_post_fail_acquisition_registration(repo_root: Path) -> list[Issue]:
    """Freeze the two producer trios without reading either runtime artifact."""

    issues: list[Issue] = []
    for relative, expected_sha256 in POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(
                issues,
                "POST_FAIL_ACQUISITION_STATIC_LEAF",
                relative,
                str(exc),
            )
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "POST_FAIL_ACQUISITION_STATIC_LEAF",
                relative,
                f"current bytes hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "POST_FAIL_ACQUISITION_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            str(exc),
        )
    else:
        entries = manifest.get("files")
        manifest_paths = {
            entry.get("path")
            for entry in entries
            if isinstance(entries, list) and isinstance(entry, Mapping)
        }
        expected_static_paths = set(POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256)
        if not expected_static_paths.issubset(manifest_paths):
            _issue(
                issues,
                "POST_FAIL_ACQUISITION_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "all six producer leaves must be exact-hashed by the static manifest",
            )
        if REGISTRY_MANIFEST_PATH in manifest_paths:
            _issue(
                issues,
                "POST_FAIL_ACQUISITION_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the dynamic registry output must not hash itself",
            )
    return issues


def validate_gse217518_public_authority_preflight_registration(
    repo_root: Path,
) -> list[Issue]:
    """Freeze the producer exact3 while leaving the EVT046 config dynamic."""

    issues: list[Issue] = []
    for (
        relative,
        expected_sha256,
    ) in GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(
                issues,
                "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF",
                relative,
                str(exc),
            )
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF",
                relative,
                f"current bytes hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            str(exc),
        )
    else:
        entries = manifest.get("files")
        manifest_paths = {
            entry.get("path")
            for entry in entries
            if isinstance(entries, list) and isinstance(entry, Mapping)
        }
        expected_static_paths = set(
            GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256
        )
        if not expected_static_paths.issubset(manifest_paths):
            _issue(
                issues,
                "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the producer config, script, and focused test must be exact-hashed by the static manifest",
            )
        if GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_RUNTIME_CONFIG_PATH in manifest_paths:
            _issue(
                issues,
                "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the dynamic EVT046 runtime config must not enter the static manifest",
            )
        if REGISTRY_MANIFEST_PATH in manifest_paths:
            _issue(
                issues,
                "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the dynamic registry output must not hash itself",
            )
    return issues


def validate_gse232572_public_recovery_audit_registration(
    repo_root: Path,
) -> list[Issue]:
    """Freeze the audit producer exact3 while leaving the EVT047 config dynamic."""

    issues: list[Issue] = []
    for relative, expected_sha256 in (
        GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF_SHA256.items()
    ):
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(
                issues,
                "GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF",
                relative,
                str(exc),
            )
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF",
                relative,
                f"current bytes hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "GSE232572_PUBLIC_RECOVERY_AUDIT_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            str(exc),
        )
    else:
        entries = manifest.get("files")
        manifest_paths = {
            entry.get("path")
            for entry in entries
            if isinstance(entries, list) and isinstance(entry, Mapping)
        }
        expected_static_paths = set(
            GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF_SHA256
        )
        if not expected_static_paths.issubset(manifest_paths):
            _issue(
                issues,
                "GSE232572_PUBLIC_RECOVERY_AUDIT_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the producer config, script, and focused test must be exact-hashed by the static manifest",
            )
        if GSE232572_PUBLIC_RECOVERY_AUDIT_RUNTIME_CONFIG_PATH in manifest_paths:
            _issue(
                issues,
                "GSE232572_PUBLIC_RECOVERY_AUDIT_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the dynamic EVT047 runtime config must not enter the static manifest",
            )
        if REGISTRY_MANIFEST_PATH in manifest_paths:
            _issue(
                issues,
                "GSE232572_PUBLIC_RECOVERY_AUDIT_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the dynamic registry output must not hash itself",
            )
    return issues


def validate_gse232572_development_v3_materialization_registration(
    repo_root: Path,
) -> list[Issue]:
    """Freeze the materializer exact3 without registering its private row output."""

    issues: list[Issue] = []
    for relative, expected_sha256 in (
        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF_SHA256.items()
    ):
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(
                issues,
                "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF",
                relative,
                str(exc),
            )
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF",
                relative,
                f"current bytes hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            str(exc),
        )
    else:
        entries = manifest.get("files")
        manifest_paths = {
            entry.get("path")
            for entry in entries
            if isinstance(entries, list) and isinstance(entry, Mapping)
        }
        expected_static_paths = set(
            GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF_SHA256
        )
        if not expected_static_paths.issubset(manifest_paths):
            _issue(
                issues,
                "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the materializer config, script, and focused test must be exact-hashed by the static manifest",
            )
        forbidden_paths = {
            GSE232572_DEVELOPMENT_V3_MATERIALIZATION_RUNTIME_CONFIG_PATH,
            REGISTRY_MANIFEST_PATH,
        }
        if manifest_paths & forbidden_paths:
            _issue(
                issues,
                "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the dynamic EVT048 config and registry output must not enter the static manifest",
            )
        if any(
            isinstance(relative, str) and relative.endswith(".private.jsonl")
            for relative in manifest_paths
        ):
            _issue(
                issues,
                "GSE232572_DEVELOPMENT_V3_PRIVATE_JSONL_EXCLUDED",
                REGISTRY_MANIFEST_PATH,
                "private row JSONL must not be a registered public artifact",
            )
    return issues


def validate_gse232572_qualification_authority_preflight_registration(
    repo_root: Path,
) -> list[Issue]:
    """Freeze the aggregate-only preflight exact3 and keep EVT049 dynamic."""

    issues: list[Issue] = []
    for relative, expected_sha256 in (
        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256.items()
    ):
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(
                issues,
                "GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF",
                relative,
                str(exc),
            )
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF",
                relative,
                f"current bytes hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            str(exc),
        )
    else:
        entries = manifest.get("files")
        manifest_paths = {
            entry.get("path")
            for entry in entries
            if isinstance(entries, list) and isinstance(entry, Mapping)
        }
        expected_static_paths = set(
            GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256
        )
        if not expected_static_paths.issubset(manifest_paths):
            _issue(
                issues,
                "GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the preflight config, script, and focused test must be exact-hashed by the static manifest",
            )
        forbidden_paths = {
            GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_RUNTIME_CONFIG_PATH,
            REGISTRY_MANIFEST_PATH,
        }
        if manifest_paths & forbidden_paths:
            _issue(
                issues,
                "GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the dynamic EVT049 config and registry output must not enter the static manifest",
            )
        if any(
            isinstance(relative, str) and relative.endswith(".private.jsonl")
            for relative in manifest_paths
        ):
            _issue(
                issues,
                "GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_PRIVATE_ROW_EXCLUDED",
                REGISTRY_MANIFEST_PATH,
                "private row JSONL must not be registered by the aggregate-only preflight ledger",
            )
    return issues


def validate_gse256185_public_geometry_registration(
    repo_root: Path,
) -> list[Issue]:
    """Freeze the public aggregate preflight exact3 without reading runtime data."""

    issues: list[Issue] = []
    for relative, expected_sha256 in (
        GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF_SHA256.items()
    ):
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(
                issues,
                "GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF",
                relative,
                str(exc),
            )
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF",
                relative,
                f"current bytes hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "GSE256185_PUBLIC_GEOMETRY_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            str(exc),
        )
    else:
        entries = manifest.get("files")
        manifest_paths = {
            entry.get("path")
            for entry in entries
            if isinstance(entries, list) and isinstance(entry, Mapping)
        }
        expected_static_paths = set(
            GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF_SHA256
        )
        if not expected_static_paths.issubset(manifest_paths):
            _issue(
                issues,
                "GSE256185_PUBLIC_GEOMETRY_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the bound preflight config, script, and focused test must be exact-hashed by the static manifest",
            )
        forbidden_paths = {
            GSE256185_PUBLIC_GEOMETRY_RUNTIME_CONFIG_PATH,
            REGISTRY_MANIFEST_PATH,
        }
        if manifest_paths & forbidden_paths:
            _issue(
                issues,
                "GSE256185_PUBLIC_GEOMETRY_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the future dynamic evidence-runtime config and registry output must not enter the static manifest",
            )
    return issues


def validate_gse256185_row_preflight_registration(
    repo_root: Path,
) -> list[Issue]:
    """Freeze the DEC022 exact3 and exclude runtime/raw outputs from the static DAG."""

    issues: list[Issue] = []
    for relative, expected_sha256 in GSE256185_ROW_PREFLIGHT_STATIC_LEAF_SHA256.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "GSE256185_ROW_PREFLIGHT_STATIC_LEAF", relative, str(exc))
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "GSE256185_ROW_PREFLIGHT_STATIC_LEAF",
                relative,
                f"current bytes hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "GSE256185_ROW_PREFLIGHT_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            str(exc),
        )
    else:
        entries = manifest.get("files")
        manifest_paths = {
            entry.get("path")
            for entry in entries
            if isinstance(entries, list) and isinstance(entry, Mapping)
        }
        if not set(GSE256185_ROW_PREFLIGHT_STATIC_LEAF_SHA256).issubset(manifest_paths):
            _issue(
                issues,
                "GSE256185_ROW_PREFLIGHT_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "the bound config, producer, and focused test must be exact-hashed by the static manifest",
            )
        forbidden_paths = {
            GSE256185_ROW_PREFLIGHT_RUNTIME_CONFIG_PATH,
            REGISTRY_MANIFEST_PATH,
            GSE256185_ROW_PREFLIGHT_REPORT_PATH,
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/raw/GSE256185/GSE256185_CPMandRRS_VCE_Var.tsv.gz",
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/raw/GSE256185/GSE256185_DNAPool_ref.fa.gz",
        }
        if manifest_paths & forbidden_paths:
            _issue(
                issues,
                "GSE256185_ROW_PREFLIGHT_MANIFEST_DAG",
                REGISTRY_MANIFEST_PATH,
                "runtime config, aggregate report, raw assets, and registry output must remain outside the static manifest",
            )
        if any(
            isinstance(relative, str)
            and (relative.endswith(".private.jsonl") or relative.endswith(".rows.jsonl"))
            for relative in manifest_paths
        ):
            _issue(
                issues,
                "GSE256185_ROW_PREFLIGHT_DISCLOSURE",
                REGISTRY_MANIFEST_PATH,
                "row/member payload artifacts are forbidden from the aggregate-only registry",
            )
    return issues


def validate_dec023_dual_preflight_evidence_registration(
    repo_root: Path,
) -> list[Issue]:
    """Bind the six final producer leaves; keep both reports out of the static DAG."""

    issues: list[Issue] = []
    static_leaves = {
        **GSE261709_PREFLIGHT_STATIC_LEAF_SHA256,
        **GSE207584_PREFLIGHT_STATIC_LEAF_SHA256,
    }
    for relative, expected_sha256 in static_leaves.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "DEC023_DUAL_PREFLIGHT_STATIC_LEAF", relative, str(exc))
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "DEC023_DUAL_PREFLIGHT_STATIC_LEAF",
                relative,
                f"current bytes hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "DEC023_DUAL_PREFLIGHT_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            str(exc),
        )
        return issues

    entries = manifest.get("files")
    manifest_paths = {
        entry.get("path")
        for entry in entries
        if isinstance(entries, list) and isinstance(entry, Mapping)
    }
    if not set(static_leaves).issubset(manifest_paths):
        _issue(
            issues,
            "DEC023_DUAL_PREFLIGHT_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            "both bound configs, both producers, and both focused tests must be exact-hashed by the static manifest",
        )
    forbidden_paths = {
        REGISTRY_MANIFEST_PATH,
        GSE261709_PREFLIGHT_REPORT_PATH,
        GSE207584_PREFLIGHT_REPORT_PATH,
    }
    if manifest_paths & forbidden_paths:
        _issue(
            issues,
            "DEC023_DUAL_PREFLIGHT_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            "dynamic reports and the registry output must remain outside the static manifest",
        )
    if any(
        isinstance(relative, str)
        and (
            relative.endswith(".private.jsonl")
            or relative.endswith(".rows.jsonl")
            or relative.endswith(".members.jsonl")
        )
        for relative in manifest_paths
    ):
        _issue(
            issues,
            "DEC023_DUAL_PREFLIGHT_DISCLOSURE",
            REGISTRY_MANIFEST_PATH,
            "member or row payload artifacts are forbidden from the aggregate-only registry",
        )
    return issues


def validate_dec027_six_rescue_evidence_registration(
    repo_root: Path,
) -> list[Issue]:
    """Bind exactly 18 producer leaves; keep six terminal reports dynamic."""

    issues: list[Issue] = []
    for relative, expected_sha256 in DEC027_SIX_RESCUE_STATIC_LEAF_SHA256.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "DEC027_SIX_RESCUE_STATIC_LEAF", relative, str(exc))
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "DEC027_SIX_RESCUE_STATIC_LEAF",
                relative,
                f"current bytes hash {actual_sha256} must remain {expected_sha256}",
            )

    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "DEC027_SIX_RESCUE_MANIFEST_DAG", REGISTRY_MANIFEST_PATH, str(exc))
        return issues
    entries = manifest.get("files")
    manifest_paths = {
        entry.get("path")
        for entry in entries
        if isinstance(entries, list) and isinstance(entry, Mapping)
    }
    if set(DEC027_SIX_RESCUE_STATIC_LEAF_SHA256) - manifest_paths:
        _issue(
            issues,
            "DEC027_SIX_RESCUE_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            "all six bound configs, producers, and focused tests must be exact-hashed as exactly 18 static leaves",
        )
    if manifest_paths & (
        DEC027_SIX_RESCUE_REPORT_PATHS
        | DEC027_EVT060_RUNTIME_CAS_PATHS
        | {REGISTRY_MANIFEST_PATH}
    ):
        _issue(
            issues,
            "DEC027_SIX_RESCUE_MANIFEST_DAG",
            REGISTRY_MANIFEST_PATH,
            "six dynamic reports, EVT060 runtime CAS artifacts, and the registry output must remain outside the static manifest",
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


def _validate_gse256185_row_preflight_interim(
    interim: Mapping[str, Any], path: str, issues: list[Issue]
) -> None:
    lineage = interim.get("artifact_lineage")
    record = (
        lineage.get(GSE256185_ROW_PREFLIGHT_LINEAGE_ID)
        if isinstance(lineage, Mapping)
        else None
    )
    if not isinstance(record, Mapping):
        _issue(issues, "A1_INTERIM_GSE256185_ROW_PREFLIGHT", path, "aggregate result lineage is required")
        return
    expected_scalars = {
        "path": GSE256185_ROW_PREFLIGHT_REPORT_PATH,
        "bytes": GSE256185_ROW_PREFLIGHT_REPORT_BYTES,
        "sha256": GSE256185_ROW_PREFLIGHT_REPORT_SHA256,
        "recorded_at": GSE256185_ROW_PREFLIGHT_RECORDED_AT,
        "dataset_id": "GSE256185",
        "decision_id": "V3-DEC-022",
        "schema_version": "route_a_v3_gse256185_aggregate_row_level_qualification_preflight.v1",
        "protocol_id": "GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_V1",
        "status": "STOP_CURRENT_PROTOCOL_NOT_QUALIFIED",
        "authority_role": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
        "aggregate_only": True,
        "preflight_complete": True,
        "all_required_gates_pass": False,
        "required_gate_axis_count": 17,
        "qualified": False,
        "a1_complete": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "sole_next_external_action": "OBTAIN_AUTHORITATIVE_INDEPENDENT_BIOLOGICAL_REPLICATE_AND_ROW_LEVEL_VALID_STANDARD_ERROR_EVIDENCE_OR_AN_EXPLICIT_OWNER_DECISION_NOT_TO_USE_GSE256185_FOR_QUALIFICATION",
        "predecessor_runtime_event_id": DEC022_AUTHORITY_RUNTIME_EVENT_ID,
        "expected_next_runtime_event_id": GSE256185_ROW_PREFLIGHT_RUNTIME_EVENT_ID,
        "next_runtime_event_id_preallocated": False,
        "runtime_sync_status": "SYNCED_EVT_055",
        "aggregate_report_read_count_for_ledger": 1,
        "private_row_artifact_read_count_for_ledger": 0,
        "raw_asset_registered_artifact_count": 0,
    }
    expected_top_keys = set(expected_scalars) | {
        "required_gate_statuses",
        "aggregate_observation",
        "remaining_independent_blocker_classes",
        "scope_attestation",
        "current_qualified_counts",
        "gse256185_contribution",
        "producer_lineage",
    }
    if set(record) != expected_top_keys:
        _issue(issues, "A1_INTERIM_GSE256185_ROW_PREFLIGHT_CLOSURE", path, "aggregate result lineage keys must remain the exact aggregate-only set")
    for key, value in expected_scalars.items():
        _expect(record, key, value, path, issues, "A1_INTERIM_GSE256185_ROW_PREFLIGHT")
    _expect_closed_mapping(
        record.get("required_gate_statuses"),
        GSE256185_ROW_PREFLIGHT_GATE_STATUSES,
        path,
        issues,
        "A1_INTERIM_GSE256185_ROW_PREFLIGHT_GATES",
    )
    expected_observation = {
        "candidate_universe": {
            "total_body_row_count": 11404, "strict_grammar_row_count": 11402,
            "strict_group_count": 652, "strict_single_parent_group_count": 637,
            "review_pool_count": 634, "review_parent_row_count": 634,
            "review_candidate_row_count": 7292, "review_row_count": 7926,
            "strict_dual_parent_group_count_excluded": 15,
            "strict_single_parent_two_candidate_group_count_excluded": 3,
            "nonstrict_record_count_excluded": 2,
            "review_candidate_family_counts": {"win": 5124, "+CCC": 1090, "-CCC": 1078, "rand": 0},
        },
        "multi_asset_lineage": {
            "fasta_records_total": 51595, "tsv_rows_reviewed": 11404,
            "fasta_headers_matched_to_tsv_rows": 11404, "missing_fasta_header_count": 0,
            "transformed_sequences_exactly_matched": 11404, "transformed_sequence_mismatch_count": 0,
        },
        "edit_replay": {
            "candidate_rows_reviewed": 7292, "direct_total": 7136,
            "publisher_assisted_total": 153, "replay_closed_total": 7289,
            "unexplained_count": 3, "win_direct_count": 5044,
            "win_publisher_assisted_count": 80, "win_position_legal_count": 5124,
            "plus_ccc_direct_count": 1090, "minus_ccc_direct_count": 1002,
            "minus_ccc_publisher_assisted_count": 73,
            "expected_edit_length_delta_counts": {"+3": 218, "+6": 218, "+9": 218, "+12": 218, "+15": 218, "-3": 231, "-6": 5353, "-9": 231, "-12": 208, "-15": 179},
        },
        "endpoint_transform": {
            "rows_reviewed": 7926, "finite_endpoint_and_replicate_row_count": 7925,
            "nonfinite_or_undefined_row_count": 1, "formula_match_within_tolerance_count": 7925,
            "formula_mismatch_count": 0, "maximum_absolute_formula_difference": 2.7253181933417636e-09,
        },
        "eligible_after_row_preflight_exclusions": {
            "pool_count": 633, "parent_row_count": 633, "candidate_row_count": 7288,
            "row_count": 7921, "candidate_family_counts": {"win": 5123, "+CCC": 1090, "-CCC": 1075},
            "pool_family_counts": {"win": 185, "+CCC": 218, "-CCC": 230},
        },
        "beneficial_direction_counts": {
            "candidate_positive_count": 3790, "candidate_negative_count": 3498,
            "candidate_zero_count": 0, "mixed_direction_pool_count": 363,
            "positive_only_pool_count": 124, "negative_only_pool_count": 146,
            "zero_only_pool_count": 0,
        },
        "structural_dedup_and_split_feasibility": {
            "retained_row_count": 7921, "source_group_count": 633, "parent_row_count": 633,
            "candidate_row_count": 7288, "gene_count": 547, "genes_with_multiple_source_groups": 66,
            "exact_utr_cluster_count": 7902, "duplicate_exact_utr_cluster_count": 18,
            "rows_in_duplicate_exact_utr_clusters": 37, "cross_pool_exact_utr_cluster_count": 1,
            "distinct_within_pool_candidate_count": 7270,
        },
        "reject_closure": {
            "dual_parent_group_count": 15, "dual_parent_group_row_count": 3467,
            "two_candidate_group_count": 3, "two_candidate_group_row_count": 9,
            "nonstrict_record_count": 2, "unexplained_edit_affected_group_count": 1,
            "unexplained_edit_candidate_count": 3, "nonfinite_endpoint_affected_group_count": 1,
            "nonfinite_endpoint_row_count": 1, "edit_failure_and_nonfinite_affected_group_overlap_count": 0,
            "orphaned_parent_after_all_candidates_rejected_count": 1,
            "retained_row_count": 7921, "mutually_exclusive_row_reason_total": 11404,
        },
    }
    _expect_closed_mapping(record.get("aggregate_observation"), expected_observation, path, issues, "A1_INTERIM_GSE256185_ROW_PREFLIGHT_AGGREGATES")
    _expect_closed_mapping(record.get("current_qualified_counts"), {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}, path, issues, "A1_INTERIM_GSE256185_ROW_PREFLIGHT_COUNTS")
    _expect_closed_mapping(record.get("gse256185_contribution"), {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0}, path, issues, "A1_INTERIM_GSE256185_ROW_PREFLIGHT_COUNTS")
    expected_scope = {
        "ordinary_public_assets_read_count": 2, "aggregate_output_only": True,
        "candidate_universe_closed_before_row_level_field_access": True,
        "member_identifier_output_count": 0, "member_role_output_count": 0,
        "member_context_output_count": 0, "sequence_output_count": 0,
        "row_effect_output_count": 0, "raw_replicate_value_output_count": 0,
        "replicate_identifier_output_count": 0, "split_assignment_output_count": 0,
        "row_level_values_serialized_count": 0, "row_level_values_persisted_count": 0,
        "canonical_record_output_count": 0, "private_or_restricted_input_read_count": 0,
        "sealed_contact_count": 0, "gse246381_contact_count": 0,
    }
    _expect_closed_mapping(record.get("scope_attestation"), expected_scope, path, issues, "A1_INTERIM_GSE256185_ROW_PREFLIGHT_DISCLOSURE")
    expected_producer = {
        "authority_commit": "4fa39abca424bb6ff82e43a847332e92934b278b",
        "authority_runtime_binding_commit": "ab511527a110dc17bc3538ee5309600396693534",
        "implementation_commit": GSE256185_ROW_PREFLIGHT_IMPLEMENTATION_COMMIT,
        "binding_commit": GSE256185_ROW_PREFLIGHT_BINDING_COMMIT,
        "binding_diff_is_config_only": True,
        "config_path": GSE256185_ROW_PREFLIGHT_CONFIG_PATH,
        "config_sha256": GSE256185_ROW_PREFLIGHT_STATIC_LEAF_SHA256[GSE256185_ROW_PREFLIGHT_CONFIG_PATH],
        "script_path": GSE256185_ROW_PREFLIGHT_SCRIPT_PATH,
        "script_sha256": GSE256185_ROW_PREFLIGHT_STATIC_LEAF_SHA256[GSE256185_ROW_PREFLIGHT_SCRIPT_PATH],
        "focused_test_path": GSE256185_ROW_PREFLIGHT_TEST_PATH,
        "focused_test_sha256": GSE256185_ROW_PREFLIGHT_STATIC_LEAF_SHA256[GSE256185_ROW_PREFLIGHT_TEST_PATH],
    }
    _expect_closed_mapping(record.get("producer_lineage"), expected_producer, path, issues, "A1_INTERIM_GSE256185_ROW_PREFLIGHT_PRODUCER")
    expected_blockers = [
        "THREE_UNEXPLAINED_MINUS_CCC_EDIT_REPLAYS", "ONE_NONFINITE_ENDPOINT_ROW",
        "BIOLOGICAL_SOURCE_GROUP_AND_REPLICATE_INDEPENDENCE", "VALID_ROW_LEVEL_STANDARD_ERROR",
        "RAW_AND_DERIVED_REDISTRIBUTION_RIGHTS", "FORMAL_OUTCOME_BLIND_SPLIT_AND_ZERO_LEAKAGE",
        "PREFROZEN_POWER_AND_FULL_CI_WIDTH", "ZERO_EXTERNAL_LEARNED_INPUT_RUNTIME_ATTESTATION",
        "BENEFICIAL_SIGNAL_VERSUS_MEASUREMENT_NOISE", "POST_DEDUP_INDEPENDENT_EFFECTIVE_N",
    ]
    _expect(record, "remaining_independent_blocker_classes", expected_blockers, path, issues, "A1_INTERIM_GSE256185_ROW_PREFLIGHT_BLOCKERS")


def _validate_dec023_dual_preflight_interim(
    interim: Mapping[str, Any], path: str, issues: list[Issue]
) -> None:
    lineage = interim.get("artifact_lineage")
    if not isinstance(lineage, Mapping):
        _issue(
            issues,
            "A1_INTERIM_DEC023_DUAL_PREFLIGHT",
            path,
            "artifact_lineage must contain both final DEC023 preflight reports",
        )
        return

    common_expected = {
        "decision_id": "V3-DEC-023",
        "aggregate_only": True,
        "preflight_complete": True,
        "all_required_gates_pass": False,
        "current_qualified_counts": {
            "ordinary": 1,
            "a1": 1,
            "true_a2": 0,
            "canonical_records": 6547,
        },
        "contribution": {
            "ordinary": 0,
            "a1": 0,
            "true_a2": 0,
            "canonical_records": 0,
        },
        "qualified": False,
        "a1_complete": False,
        "qualification_allowed": False,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "predecessor_runtime_event_id": DEC023_AUTHORITY_RUNTIME_EVENT_ID,
        "expected_next_runtime_event_id": DEC023_CURRENT_RUNTIME_EVENT_ID,
        "next_runtime_event_id_preallocated": False,
        "runtime_sync_status": "SYNCED_EVT_057",
        "runtime_event_emitted": True,
        "aggregate_report_read_count_for_ledger": 1,
        "private_row_artifact_read_count_for_ledger": 0,
        "raw_asset_registered_artifact_count": 0,
    }
    dataset_specs = (
        (
            GSE261709_PREFLIGHT_LINEAGE_ID,
            {
                "path": GSE261709_PREFLIGHT_REPORT_PATH,
                "bytes": GSE261709_PREFLIGHT_REPORT_BYTES,
                "sha256": GSE261709_PREFLIGHT_REPORT_SHA256,
                "recorded_at": GSE261709_PREFLIGHT_RECORDED_AT,
                "dataset_id": "GSE261709",
                "schema_version": "route_a_v3_gse261709_public_identifier_asset_schema_aggregate_geometry_preflight.v1",
                "protocol_id": "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_PREFLIGHT_V1",
                "status": "STOP_PREFLIGHT_GATES_NOT_CLOSED",
                "artifact_type": "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_PREFLIGHT_ONLY",
                "authority_role": "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_PREFLIGHT_ONLY",
                "required_gate_axis_count": 3,
                "required_gate_status_counts": {"PASS": 1, "BLOCKED": 2},
                "required_gate_results": GSE261709_PREFLIGHT_GATE_RESULTS,
                "aggregate_observation": {
                    "official_identifier_count": 5,
                    "official_source_locator_count": 3,
                    "live_metadata_source_count": 2,
                    "total_sample_count": 7,
                    "biological_assay_sample_count": 6,
                    "input_pool_sample_count": 1,
                    "cell_context_count": 2,
                    "replicate_count_per_cell_context": 3,
                    "platform_count": 1,
                    "supplementary_archive_listing_count": 1,
                    "supplementary_archive_display_size": "690.0 Kb",
                    "run_count_or_status": "UNKNOWN_NOT_ASSERTED",
                    "header_name_count": 0,
                    "dimension_measure_count": 0,
                    "required_header_role_class_count": 6,
                    "observed_required_header_role_class_count": 0,
                    "applicable_asset_license_notice_count": 0,
                    "article_license_notice_count": 0,
                },
                "scope_attestation": {
                    "ordinary_public_only": True,
                    "single_aggregate_output_only": True,
                    "whole_small_metadata_response_transport_and_decode_count": 2,
                    "archive_listing_metadata_parsed_count": 1,
                    "archive_endpoint_access_count": 0,
                    "archive_download_count": 0,
                    "archive_member_listing_count": 0,
                    "archive_member_open_count": 0,
                    "payload_endpoint_access_count": 0,
                    "member_identifier_output_count": 0,
                    "header_name_output_count": 0,
                    "row_level_output_count": 0,
                    "sequence_output_count": 0,
                    "private_or_restricted_input_read_count": 0,
                    "sealed_contact_count": 0,
                    "persisted_artifact_count": 0,
                },
                "producer_lineage": {
                    "implementation_commit": GSE261709_PREFLIGHT_IMPLEMENTATION_COMMIT,
                    "binding_commit": GSE261709_PREFLIGHT_BINDING_COMMIT,
                    "binding_diff_is_config_only": True,
                    "config_path": GSE261709_PREFLIGHT_CONFIG_PATH,
                    "config_sha256": GSE261709_PREFLIGHT_STATIC_LEAF_SHA256[GSE261709_PREFLIGHT_CONFIG_PATH],
                    "script_path": GSE261709_PREFLIGHT_SCRIPT_PATH,
                    "script_sha256": GSE261709_PREFLIGHT_STATIC_LEAF_SHA256[GSE261709_PREFLIGHT_SCRIPT_PATH],
                    "focused_test_path": GSE261709_PREFLIGHT_TEST_PATH,
                    "focused_test_sha256": GSE261709_PREFLIGHT_STATIC_LEAF_SHA256[GSE261709_PREFLIGHT_TEST_PATH],
                },
            },
            "A1_INTERIM_GSE261709_PREFLIGHT",
        ),
        (
            GSE207584_PREFLIGHT_LINEAGE_ID,
            {
                "path": GSE207584_PREFLIGHT_REPORT_PATH,
                "bytes": GSE207584_PREFLIGHT_REPORT_BYTES,
                "sha256": GSE207584_PREFLIGHT_REPORT_SHA256,
                "recorded_at": GSE207584_PREFLIGHT_RECORDED_AT,
                "dataset_id": "GSE207584",
                "schema_version": "route_a_v3_gse207584_aggregate_dense_family_qualification_preflight.v1",
                "protocol_id": "GSE207584_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_V1",
                "status": "STOP_CURRENT_PROTOCOL_NOT_QUALIFIED",
                "artifact_type": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
                "authority_role": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
                "required_gate_axis_count": 11,
                "required_gate_status_counts": {
                    "PASS_PREFLIGHT_ONLY": 1,
                    "FAIL_CLOSED": 2,
                    "UNKNOWN_NOT_ASSERTED": 8,
                },
                "required_gate_results": GSE207584_PREFLIGHT_GATE_RESULTS,
                "aggregate_observation": {
                    "reference_construct_count": 1395,
                    "observed_unique_construct_count": 955,
                    "intended_not_observed_count": 440,
                    "observed_not_intended_count": 0,
                    "authoritative_mapping_candidate_count": 0,
                    "source_mapping_provided": False,
                    "observed_asset_body_row_count": 10227,
                    "duplicate_measurement_tuple_conflict_count": 9272,
                    "unresolved_construct_count": 952,
                    "biological_replicate_count": 3,
                    "independent_n_per_candidate": 3,
                    "timepoint_by_replicate_observation_count_per_candidate": 9,
                    "valid_endpoint_and_standard_error_count": 3,
                    "endpoint_direction": "HIGHER_IS_SLOWER_DECAY_AND_GREATER_STABILITY",
                    "post_dedup_independent_source_family_count": 0,
                    "required_effective_n_for_power_and_ci_width": 156,
                },
                "scope_attestation": {
                    "ordinary_public_asset_read_count": 2,
                    "aggregate_output_only": True,
                    "source_mapping_read_count": 0,
                    "member_identifier_output_count": 0,
                    "sequence_output_count": 0,
                    "row_measurement_output_count": 0,
                    "split_assignment_output_count": 0,
                    "training_count": 0,
                    "gpu_work_count": 0,
                    "model_selection_count": 0,
                    "private_or_restricted_input_read_count": 0,
                    "sealed_contact_count": 0,
                    "persisted_artifact_count": 0,
                },
                "producer_lineage": {
                    "implementation_commit": GSE207584_PREFLIGHT_IMPLEMENTATION_COMMIT,
                    "binding_commit": GSE207584_PREFLIGHT_BINDING_COMMIT,
                    "binding_diff_is_config_only": True,
                    "config_path": GSE207584_PREFLIGHT_CONFIG_PATH,
                    "config_sha256": GSE207584_PREFLIGHT_STATIC_LEAF_SHA256[GSE207584_PREFLIGHT_CONFIG_PATH],
                    "script_path": GSE207584_PREFLIGHT_SCRIPT_PATH,
                    "script_sha256": GSE207584_PREFLIGHT_STATIC_LEAF_SHA256[GSE207584_PREFLIGHT_SCRIPT_PATH],
                    "focused_test_path": GSE207584_PREFLIGHT_TEST_PATH,
                    "focused_test_sha256": GSE207584_PREFLIGHT_STATIC_LEAF_SHA256[GSE207584_PREFLIGHT_TEST_PATH],
                },
            },
            "A1_INTERIM_GSE207584_PREFLIGHT",
        ),
    )
    expected_top_keys = set(common_expected) | {
        "path",
        "bytes",
        "sha256",
        "recorded_at",
        "dataset_id",
        "schema_version",
        "protocol_id",
        "status",
        "artifact_type",
        "authority_role",
        "required_gate_axis_count",
        "required_gate_status_counts",
        "required_gate_results",
        "aggregate_observation",
        "scope_attestation",
        "producer_lineage",
    }
    for lineage_id, specific, code in dataset_specs:
        record = lineage.get(lineage_id)
        if not isinstance(record, Mapping):
            _issue(issues, code, path, f"missing lineage {lineage_id}")
            continue
        if set(record) != expected_top_keys:
            _issue(
                issues,
                f"{code}_CLOSURE",
                path,
                "final report lineage keys must remain the exact aggregate-only set",
            )
        for key, value in common_expected.items():
            if key in {"current_qualified_counts", "contribution"}:
                _expect_closed_mapping(record.get(key), value, path, issues, f"{code}_COUNTS")
            else:
                _expect(record, key, value, path, issues, code)
        for key, value in specific.items():
            if key in {
                "required_gate_status_counts",
                "required_gate_results",
                "aggregate_observation",
                "scope_attestation",
                "producer_lineage",
            }:
                _expect_closed_mapping(record.get(key), value, path, issues, f"{code}_{key.upper()}")
            else:
                _expect(record, key, value, path, issues, code)


def _expected_dec027_six_rescue_record(
    lineage_id: str, spec: Mapping[str, Any]
) -> dict[str, Any]:
    static_paths = spec["static_paths"]
    return {
        "path": spec["path"],
        "bytes": spec["bytes"],
        "sha256": spec["sha256"],
        "report_observed_at": spec["report_observed_at"],
        "decision_id": "V3-DEC-027",
        "route_id": spec["route_id"],
        "dataset_id": spec["dataset_id"],
        "registry_status": spec["registry_status"],
        "candidate_role": spec["candidate_role"],
        "schema_version": spec["schema_version"],
        "protocol_id": spec["protocol_id"],
        "status": spec["status"],
        "aggregate_only": True,
        "terminal_report_registered": True,
        "required_gate_axis_count": sum(spec["gate_status_counts"].values()),
        "required_gate_status_counts": spec["gate_status_counts"],
        "required_gate_results": DEC027_SIX_RESCUE_GATE_RESULTS[lineage_id],
        "producer_lineage": {
            "implementation_commit": spec["implementation_commit"],
            "binding_commit": spec["binding_commit"],
            "binding_diff_is_config_only": True,
            "config_path": static_paths[0],
            "config_sha256": DEC027_SIX_RESCUE_STATIC_LEAF_SHA256[static_paths[0]],
            "script_path": static_paths[1],
            "script_sha256": DEC027_SIX_RESCUE_STATIC_LEAF_SHA256[static_paths[1]],
            "focused_test_path": static_paths[2],
            "focused_test_sha256": DEC027_SIX_RESCUE_STATIC_LEAF_SHA256[static_paths[2]],
        },
        "current_qualified_counts": {
            "ordinary": 1,
            "a1": 1,
            "true_a2": 0,
            "canonical_records": 6547,
        },
        "contribution": {
            "ordinary": 0,
            "a1": 0,
            "true_a2": 0,
            "canonical_records": 0,
        },
        "qualified": False,
        "qualification_allowed": False,
        "role_assignment_allowed": False,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "active_registry_study_unit_count_changed": False,
        "member_or_row_payload_registered": False,
        "sequence_or_effect_payload_registered": False,
        "latest_settled_runtime_event_id": DEC027_CURRENT_RUNTIME_EVENT_ID,
        "expected_next_runtime_event_id": DEC027_PENDING_RUNTIME_EVENT_ID,
        "pending_successor_runtime_event_label": DEC027_PENDING_SUCCESSOR_RUNTIME_EVENT_LABEL,
        "next_runtime_event_id_preallocated": False,
        "runtime_sync_status": "EVIDENCE_REGISTERED_AFTER_EVT_059_PENDING_UNALLOCATED_EVT_060",
        "runtime_event_emitted": False,
        "aggregate_report_read_count_for_ledger": 1,
        "private_row_artifact_read_count_for_ledger": 0,
        "raw_asset_registered_artifact_count": 0,
    }


def _expected_dec027_evt060_projection_record() -> dict[str, Any]:
    run_root = (
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
        "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5"
    )
    return {
        "record_type": "DEC027_EVT060_CURRENT_PROJECTION_SETTLEMENT",
        "decision_id": "V3-DEC-027",
        "runtime_binding_commit": "6a7ebeae2a2ced43e29c9458601e43a19496416c",
        "aggregate_only": True,
        "event_id": DEC027_EVT060_CURRENT_RUNTIME_EVENT_ID,
        "event_count": 60,
        "runtime_sync_status": "SYNCED_EVT_060",
        "manifest_output_count": 266,
        "manifest_registered_artifact_count": 14,
        "all_six_terminal_reports_registered": True,
        "stop_rule_ready": True,
        "stop_rule_evaluated_by_this_event": False,
        "conditional_successor_activated": False,
        "current_qualified_counts": {
            "ordinary": 1,
            "a1": 1,
            "true_a2": 0,
            "canonical_records": 6547,
        },
        "scientific_claim_status": "NOT_ESTABLISHED",
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "stop_rule_trigger_condition_met": False,
        "successor_runtime_event_id": DEC027_PENDING_RUNTIME_EVENT_ID,
        "successor_runtime_event_id_preallocated": False,
        "evt061_emitted": False,
        "runtime_event_emitted_by_this_settlement": False,
        "runtime_cas": {
            "status": {
                "path": f"{run_root}/STATUS.json",
                "bytes": 32889,
                "sha256": "faad1d6bceecb8bece2a95bdb2420eb98cfb38cc9efabe6e38bb6d6f9f8fbced",
            },
            "run_manifest": {
                "path": f"{run_root}/RUN_MANIFEST.json",
                "bytes": 118291,
                "sha256": "700a285b61fdf13f69aecafecef8f590de4e44322bbdd0f528877027bd5de2f7",
            },
            "event_log": {
                "path": f"{run_root}/EVENT_LOG.jsonl",
                "bytes": 156074,
                "sha256": "91e4ec7a4a0b221ba641a544c90d213d6b2ce02ff076372a8bca34309c71d5e8",
            },
            "event_log_tail": {
                "identity": "EVENT_LOG_JSONL_CANONICAL_EVT060_TAIL_BYTES",
                "bytes": 5393,
                "sha256": "0895318bf3fec442ffdd3b914f3f51308e9b1e50e0cc847ff0a537e7c6638e47",
            },
        },
        "runtime_cas_artifacts_in_static_manifest": False,
        "private_or_sealed_payload_registered": False,
    }


def _validate_dec027_six_rescue_interim(
    interim: Mapping[str, Any], path: str, issues: list[Issue]
) -> None:
    lineage = interim.get("artifact_lineage")
    if not isinstance(lineage, Mapping):
        _issue(issues, "A1_INTERIM_DEC027_SIX_RESCUE", path, "artifact_lineage must be a mapping")
        return
    for lineage_id, spec in DEC027_SIX_RESCUE_REPORT_SPECS.items():
        record = lineage.get(lineage_id)
        if not isinstance(record, Mapping):
            _issue(issues, "A1_INTERIM_DEC027_SIX_RESCUE", path, f"missing lineage {lineage_id}")
            continue
        _expect_closed_mapping(
            record,
            _expected_dec027_six_rescue_record(lineage_id, spec),
            path,
            issues,
            "A1_INTERIM_DEC027_SIX_RESCUE",
        )
    settlement = lineage.get(DEC027_EVT060_PROJECTION_LINEAGE_ID)
    _expect_closed_mapping(
        settlement if isinstance(settlement, Mapping) else {},
        _expected_dec027_evt060_projection_record(),
        path,
        issues,
        "A1_INTERIM_DEC027_EVT060_PROJECTION",
    )

    disposition = interim.get("dec027_current_disposition")
    if not isinstance(disposition, Mapping):
        _issue(issues, "A1_INTERIM_DEC027_SIX_RESCUE", path, "dec027_current_disposition must be a mapping")
        return
    _expect(
        disposition,
        "route_statuses",
        {spec["route_id"]: spec["status"] for spec in DEC027_SIX_RESCUE_REPORT_SPECS.values()},
        path,
        issues,
        "A1_INTERIM_DEC027_SIX_RESCUE_ROUTE_STATUS",
    )
    expected_registration = {
        "integration_id": DEC027_SIX_RESCUE_EVIDENCE_INTEGRATION_ID,
        "registered_lineage_ids_exactly": DEC027_SIX_RESCUE_LINEAGE_IDS,
        "static_leaf_count": 18,
        "dynamic_report_count": 6,
        "latest_settled_runtime_event_id": DEC027_CURRENT_RUNTIME_EVENT_ID,
        "expected_next_runtime_event_id": DEC027_PENDING_RUNTIME_EVENT_ID,
        "pending_successor_runtime_event_label": DEC027_PENDING_SUCCESSOR_RUNTIME_EVENT_LABEL,
        "next_runtime_event_id_preallocated": False,
        "runtime_sync_status": "EVIDENCE_REGISTERED_AFTER_EVT_059_PENDING_UNALLOCATED_EVT_060",
        "runtime_event_emitted": False,
    }
    _expect_closed_mapping(
        disposition.get("evidence_registration")
        if isinstance(disposition.get("evidence_registration"), Mapping)
        else {},
        expected_registration,
        path,
        issues,
        "A1_INTERIM_DEC027_SIX_RESCUE_REGISTRATION",
    )
    expected_adjudication = {
        "all_six_terminal_reports_registered": True,
        "separate_promotion_count": 0,
        "qualified_counts_after_reports": {
            "ordinary": 1,
            "a1": 1,
            "true_a2": 0,
            "canonical_records": 6547,
        },
        "rescue_floor_met": False,
        "rescue_floor_failed": True,
        "gse269595_true_a2_reachability": {
            "effective_source_group_n": 363,
            "required_effective_n": 156,
            "planning_power": 0.9977590398119175,
            "target_power": 0.8,
            "planning_full_ci_width": 0.19610459396615842,
            "maximum_full_ci_width": 0.3,
            "reachable_for_stop_rule": True,
            "formal_qualification_power_run": False,
            "qualification_established": False,
            "role_assigned": False,
            "credit_change": 0,
        },
        "no_true_a2_route_with_reachable_post_dedup_n_and_power": False,
        "trigger_condition_met": False,
        "successor_amendment_triggered": False,
        "full_route_a_remains_highest_inactive_target": True,
        "stop_rule_ready": True,
        "stop_rule_evaluated_by_evt060": False,
        "conditional_successor_activated": False,
    }
    _expect_closed_mapping(
        disposition.get("stop_rule_adjudication")
        if isinstance(disposition.get("stop_rule_adjudication"), Mapping)
        else {},
        expected_adjudication,
        path,
        issues,
        "A1_INTERIM_DEC027_STOP_RULE_ADJUDICATION",
    )
    for key, value in {
        "latest_settled_runtime_event_id": DEC027_EVT060_CURRENT_RUNTIME_EVENT_ID,
        "settled_runtime_event_changed": True,
        "runtime_event_emitted": True,
        "runtime_sync_status": "SYNCED_EVT_060",
        "expected_next_runtime_event_id": DEC027_PENDING_RUNTIME_EVENT_ID,
        "next_runtime_event_id_preallocated": False,
    }.items():
        _expect(
            disposition,
            key,
            value,
            path,
            issues,
            "A1_INTERIM_DEC027_EVT060_CURRENT_PROJECTION",
        )
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
        "dec020_current_disposition",
        "dec021_current_disposition",
        "dec022_current_disposition",
        "dec023_current_disposition",
        "dec024_current_disposition",
        "dec027_current_disposition",
        "dec028_current_disposition",
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
            "dec020_amendment_path": DEC020_AMENDMENT_PATH,
            "dec021_amendment_path": DEC021_AMENDMENT_PATH,
            "dec022_amendment_path": DEC022_AMENDMENT_PATH,
            "dec023_amendment_path": DEC023_AMENDMENT_PATH,
            "dec024_amendment_path": DEC024_AMENDMENT_PATH,
            "dec027_amendment_path": DEC027_AMENDMENT_PATH,
            "dec028_amendment_path": DEC028_AMENDMENT_PATH,
            "decision_log_path": DECISION_LOG_PATH,
            "data_role_registry_path": REGISTRY_PATHS["data"],
            "claim_evidence_matrix_path": REGISTRY_PATHS["claim"],
            "accepted_a0_activation_commit": "fd722d5fa3c2538fce742b8942b1fb48e782760b",
            "branch": "routea-v3-a1-20260810",
            "worktree": "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810",
            "run_id": "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5",
            "run_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5",
            "data_role_authority_remains": REGISTRY_PATHS["data"],
            "this_record_changes_dataset_qualification": True,
        }
        for key, value in expected_authority.items():
            _expect(authority, key, value, path, issues, "A1_INTERIM_AUTHORITY")
        for hash_key, relative in (
            ("dec019_amendment_sha256", DEC019_AMENDMENT_PATH),
            ("dec020_amendment_sha256", DEC020_AMENDMENT_PATH),
            ("dec021_amendment_sha256", DEC021_AMENDMENT_PATH),
            ("dec022_amendment_sha256", DEC022_AMENDMENT_PATH),
            ("dec023_amendment_sha256", DEC023_AMENDMENT_PATH),
            ("dec024_amendment_sha256", DEC024_AMENDMENT_PATH),
            ("dec027_amendment_sha256", DEC027_AMENDMENT_PATH),
            ("dec028_amendment_sha256", DEC028_AMENDMENT_PATH),
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
            "dec020_amendment_sha256",
            "dec021_amendment_sha256",
            "dec022_amendment_sha256",
            "dec023_amendment_sha256",
            "dec024_amendment_sha256",
            "dec027_amendment_sha256",
            "dec028_amendment_sha256",
            "decision_log_sha256",
            "data_role_registry_sha256",
            "claim_evidence_matrix_sha256",
            "active_authority_leaf_sha256",
        }
        _expect(
            authority,
            "active_authority_leaf_sha256",
            DEC028_ACTIVE_AUTHORITY_LEAF_SHA256,
            path,
            issues,
            "A1_INTERIM_AUTHORITY_HASH",
        )
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
            "public_identifier_and_pool_geometry_preflight_only_dataset_ids": ["GSE256185"],
            "aggregate_row_level_qualification_preflight_only_dataset_ids": ["GSE256185"],
            "public_identifier_asset_schema_and_aggregate_geometry_preflight_only_dataset_ids": ["GSE261709"],
            "aggregate_dense_family_qualification_preflight_only_dataset_ids": ["GSE207584"],
            "processed_row_level_a1_qualification_preflight_only_dataset_ids": ["GSE261709"],
            "replacement_a1_or_true_a2_role_adjudication_preflight_only_dataset_ids": ["GSE269595"],
            "replacement_true_a2_dense_family_preflight_only_dataset_ids": ["E-MTAB-10902"],
            "dec027_existing_registered_rescue_dataset_ids": [
                "GSE217518",
                "ENCSR854RUF",
                "GSE232572",
                "GSE269595",
            ],
            "dec027_external_preflight_candidate_only_dataset_ids": [
                "GSE113849",
                "GSE295080",
            ],
            "dec028_primary_single_study_unit": "GSE200304",
            "dec028_operational_mainline": "SINGLE_STUDY_SOURCE_RELATIVE_DEVELOPMENT_ENGINEERING_THEORY",
            "scheme_a_changes_qualified_counts": True,
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
            "qualified_independent_ordinary_studies": 1,
            "required_independent_ordinary_studies": 3,
            "qualified_a1_studies": 1,
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
                "a1_freeze_status": "FROZEN",
                "a1_zero_leakage_audit_status": "PASS",
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
            "runtime_sync_status": "SYNCED_EVT_049",
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
                "current_blocker_count": 1,
                "current_blockers": GSE200304_DEC019_ONE_BLOCKER_BLOCKERS,
                "input_gate_count": 8,
                "input_status_counts": GSE200304_DEC019_ONE_BLOCKER_INPUT_STATUS_COUNTS,
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
                "predecessor_runtime_event_id": "A1-EVT-044",
                "expected_next_runtime_event_id": "A1-EVT-045",
                "runtime_sync_status": "SYNCED_EVT_045",
            }
            _expect_closed_mapping(
                gse200304_current,
                expected_gse200304_current,
                path,
                issues,
                "A1_INTERIM_DEC019_GSE200304",
            )

    dec020_disposition = interim.get("dec020_current_disposition")
    if not isinstance(dec020_disposition, Mapping):
        _issue(
            issues,
            "A1_INTERIM_DEC020",
            path,
            "dec020_current_disposition must be a mapping",
        )
    else:
        _expect_closed_mapping(
            dec020_disposition,
            {
                "decision_id": "V3-DEC-020",
                "status": "FROZEN_USER_AUTHORIZED_GSE200304_MODEL_INPUT_ROUTE_POLICY",
                "authority_only_not_study_qualification": True,
                "dataset_id": "GSE200304",
                "selected_route": DEC020_SCRATCH_ROUTE,
                "selected_route_status": "PASS_DEC020_SCRATCH_ROUTE_SCOPED_REPORTED_ENDPOINT_A1_QUALIFIED",
                "retained_route": DEC020_FOUNDATION_ROUTE,
                "retained_route_status": "RETAINED_FAIL_CURRENT_PROTOCOL",
                "route_gate": DEC020_ROUTE_GATE,
                "scratch_route_checkpoint_exposure_status": DEC020_SCRATCH_EXPOSURE_STATUS,
                "scratch_route_checkpoint_exposure_pass_claimed": False,
                "scratch_route_external_checkpoint_count_allowed": 0,
                "scratch_route_external_learned_input_count_allowed": 0,
                "scratch_route_parameter_initialization": "RANDOM_INITIALIZATION_ONLY",
                "scratch_route_runtime_attestation_required": True,
                "foundation_route_minimum_audited_checkpoint_count_for_pass": 1,
                "foundation_route_empty_checkpoint_set_can_pass": False,
                "foundation_route_checkpoint_exposure_may_be_waived": False,
                "prior_aggregate_design_use_status": "DISCLOSED_NOT_UNTOUCHED",
                "untouched_claim_allowed": False,
                "claim_of_no_prior_influence_allowed": False,
                "prospective_freeze_boundary": "DEC020_FORWARD",
                "full_prior_analytic_use_attestation_required_before_predictor_protocol_promotion_or_training": True,
                "full_prior_analytic_use_attestation_completed": False,
                "existing_license_and_redistribution_rights_gate_status": "PASS",
                "rights_gate_reopened_by_dec020": False,
                "current_qualified_counts": {
                    "ordinary": 1,
                    "a1": 1,
                    "true_a2": 0,
                    "canonical_records": 6547,
                },
                "qualified": True,
                "canonical_materialization_qualification_eligible": True,
                "canonical_materialization_execution_authorized": False,
                "private_payload_access_allowed": False,
                "row_level_payload_read_count": 0,
                "private_payload_read_count": 0,
                "sealed_payload_read_count": 0,
                "training_allowed": False,
                "gpu_work_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "latest_settled_runtime_event_id": "A1-EVT-051",
                "authority_runtime_sync": {
                    "predecessor_event_id": "A1-EVT-050",
                    "next_event_id": "A1-EVT-051",
                    "next_event_id_preallocated": False,
                    "status": "SYNCED_EVT_051",
                },
                "future_v4_successor_registration": {
                    "lifecycle_status": "ADJUDICATED_POST_IMPLEMENTATION_COMMIT_I_BOUND_PRODUCTION",
                    "expected_static_leaf_count": 3,
                    "expected_static_leaf_roles": ["CONFIG", "SCRIPT", "FOCUSED_TEST"],
                    "registered_static_leaf_count": 3,
                    "registered_in_static_manifest": True,
                    "leaf_paths": [GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH, GSE200304_DEC020_V4_SCRIPT_PATH, GSE200304_DEC020_V4_TEST_PATH],
                    "leaf_sha256": list(GSE200304_DEC020_V4_STATIC_LEAF_SHA256.values()),
                    "implementation_commit": GSE200304_DEC020_V4_IMPLEMENTATION_COMMIT,
                    "binding_commit": GSE200304_DEC020_V4_BINDING_COMMIT,
                    "may_execute": False,
                    "may_adjudicate": False,
                },
            },
            path,
            issues,
            "A1_INTERIM_DEC020",
        )

    dec021_disposition = interim.get("dec021_current_disposition")
    if not isinstance(dec021_disposition, Mapping):
        _issue(issues, "A1_INTERIM_DEC021", path, "dec021_current_disposition must be a mapping")
    else:
        _expect_closed_mapping(
            dec021_disposition,
            {
                "decision_id": "V3-DEC-021",
                "status": "FROZEN_USER_AUTHORIZED_GSE256185_PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
                "authority_only_not_study_qualification": True,
                "dataset_id": "GSE256185",
                "preflight_role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
                "preflight_candidate_only_not_counting": True,
                "allowed_input_field_classes_exactly": ["IDENTIFIER", "ROLE", "CONTEXT"],
                "allowed_output_class": "AGGREGATE_POOL_GEOMETRY_ONLY",
                "sequence_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                "edit_budget_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                "effect_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                "true_a2_status_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                "qualification_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
                "changes_current_qualified_counts": False,
                "gse256185_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
                "row_output_allowed": False,
                "sequence_output_allowed": False,
                "effect_output_allowed": False,
                "qualification_allowed": False,
                "canonical_materialization_allowed": False,
                "phase_complete": False,
                "training_allowed": False,
                "gpu_work_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "latest_settled_runtime_event_id": GSE256185_PUBLIC_GEOMETRY_RUNTIME_EVENT_ID,
                "settled_runtime_event_changed": True,
                "runtime_event_emitted": True,
                "sealed_contact_allowed": False,
            },
            path,
            issues,
            "A1_INTERIM_DEC021",
        )

    dec022_disposition = interim.get("dec022_current_disposition")
    if not isinstance(dec022_disposition, Mapping):
        _issue(issues, "A1_INTERIM_DEC022", path, "dec022_current_disposition must be a mapping")
    else:
        _expect_closed_mapping(
            dec022_disposition,
            {
                "decision_id": "V3-DEC-022",
                "status": "FROZEN_USER_AUTHORIZED_GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
                "authority_only_not_study_qualification": True,
                "dataset_id": "GSE256185",
                "preflight_role": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
                "predecessor_decision_id": "V3-DEC-021",
                "predecessor_runtime_event_id": "A1-EVT-053",
                "candidate_universe": {
                    "strict_single_parent_pool_count": 634,
                    "strict_candidate_member_count": 7292,
                    "two_candidate_strict_single_parent_group_count_excluded": 3,
                    "dual_parent_group_count_excluded": 15,
                    "nonstrict_grammar_record_count_excluded": 2,
                    "reasoned_family_closure_candidate_count": 7294,
                    "reasoned_family_closure_included": False,
                    "membership_must_replay_before_row_level_fields": True,
                    "drift_action": "STOP_BEFORE_ROW_LEVEL_FIELD_ACCESS",
                },
                "allowed_input_field_classes_exactly": ["IDENTIFIER", "ROLE", "SEQUENCE", "ENDPOINT", "REPLICATE", "NECESSARY_CONTEXT"],
                "allowed_output_class": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
                "required_fail_closed_gate_ids_exactly": DEC022_REQUIRED_GATE_IDS,
                "independent_gate_axis_count": 17,
                "initial_status_for_every_gate": "NOT_RUN",
                "unknown_or_not_run_gate_is_pass": False,
                "all_required_gates_must_pass": True,
                "all_required_gates_passing_automatically_qualifies_dataset": False,
                "source_to_candidate_edit_relation_may_be_presumed": False,
                "biological_replicate_independence_may_be_inferred_from_barcode_or_reaction_count": False,
                "technical_replicates_may_substitute_for_biological_replicates": False,
                "target_power_minimum": 0.8,
                "maximum_full_ci_width": 0.3,
                "current_qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
                "changes_current_qualified_counts": False,
                "gse256185_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
                "gse256185_qualified": False,
                "member_identifier_output_allowed": False,
                "row_output_allowed": False,
                "sequence_output_allowed": False,
                "row_effect_output_allowed": False,
                "replicate_identifier_output_allowed": False,
                "split_assignment_output_allowed": False,
                "qualification_allowed": False,
                "canonical_materialization_allowed": False,
                "phase_complete": False,
                "training_allowed": False,
                "gpu_work_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "latest_settled_runtime_event_id": GSE256185_ROW_PREFLIGHT_RUNTIME_EVENT_ID,
                "settled_runtime_event_changed": True,
                "runtime_event_emitted": True,
                "authority_runtime_sync_status": "SYNCED_EVT_054",
                "evidence_runtime_sync_status": "SYNCED_EVT_055",
                "expected_next_runtime_event_id": GSE256185_ROW_PREFLIGHT_RUNTIME_EVENT_ID,
                "next_runtime_event_id_preallocated": False,
                "sealed_contact_allowed": False,
            },
            path,
            issues,
            "A1_INTERIM_DEC022",
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
            GSE149487_PUBLIC_ASSET_ACQUISITION_LINEAGE_ID,
            GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_LINEAGE_ID,
            GSE232572_PUBLIC_RECOVERY_AUDIT_LINEAGE_ID,
            GSE232572_DEVELOPMENT_V3_MATERIALIZATION_FAILURE_LINEAGE_ID,
            GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LINEAGE_ID,
            GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_LINEAGE_ID,
            GSE256185_PUBLIC_GEOMETRY_LINEAGE_ID,
            GSE256185_ROW_PREFLIGHT_LINEAGE_ID,
            GSE261709_PREFLIGHT_LINEAGE_ID,
            GSE207584_PREFLIGHT_LINEAGE_ID,
            *DEC027_SIX_RESCUE_LINEAGE_IDS,
            DEC027_EVT060_PROJECTION_LINEAGE_ID,
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
            GSE200304_UPSTREAM_AUTHORITY_LINEAGE_ID,
            GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_LINEAGE_ID,
            GSE200304_DEC019_UPSTREAM_PASS_ADJUDICATION_LINEAGE_ID,
            GSE200304_DEC019_GROUP_LINEAGE_ID,
            GSE200304_DEC019_SPLIT_LINEAGE_ID,
            GSE200304_DEC019_POWER_LINEAGE_ID,
            GSE200304_DEC019_ONE_BLOCKER_ADJUDICATION_LINEAGE_ID,
            GSE200304_CHECKPOINT_EXPOSURE_FAIL_LINEAGE_ID,
            GSE200304_DEC020_V4_LINEAGE_ID,
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
                "sha256": DEC022_ACTIVE_AUTHORITY_LEAF_SHA256[
                    A1_QUALIFICATION_CONFIG_PATH
                ],
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
            GSE149487_PUBLIC_ASSET_ACQUISITION_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE149487/"
                    "GSE149487_PUBLIC_ASSETS_20260812T184152P0800/"
                    "GSE149487_PUBLIC_ASSET_ACQUISITION_V1.json"
                ),
                "bytes": 22790,
                "sha256": "0da2680906c5246d7d472632983d47b67223d55d63e11b5f44f9890574088242",
                "dataset_id": "GSE149487",
                "schema_version": "route_a_v3_gse149487_public_asset_acquisition.v1",
                "status": "STOPPED_WITH_PUBLIC_EVIDENCE_BLOCKER",
                "acquisition_status": "EXACT_21_ASSETS_ACQUIRED_AND_INTEGRITY_VERIFIED",
                "aggregate_metadata_only": True,
                "asset_count": 21,
                "geo_raw_count": 18,
                "supplement_count": 3,
                "total_verified_bytes": 70032274,
                "ready_for_full_qualifier_input": False,
                "ready_for_study_qualification": False,
                "qualified": False,
                "canonical_record_count": 0,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "producer_lineage": {
                    "implementation_commit": "e95e9bbfe099ec11a948019836560a23bb71e1b3",
                    "binding_commit": "7021118d8a27fc48beb0a1b0de1ce1059bbeb225",
                    "binding_diff_is_config_only": True,
                    "config_path": GSE149487_PUBLIC_ASSET_ACQUISITION_CONFIG_PATH,
                    "config_sha256": POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256[
                        GSE149487_PUBLIC_ASSET_ACQUISITION_CONFIG_PATH
                    ],
                    "script_path": GSE149487_PUBLIC_ASSET_ACQUISITION_SCRIPT_PATH,
                    "script_sha256": POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256[
                        GSE149487_PUBLIC_ASSET_ACQUISITION_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE149487_PUBLIC_ASSET_ACQUISITION_TEST_PATH,
                    "focused_test_sha256": POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256[
                        GSE149487_PUBLIC_ASSET_ACQUISITION_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-044",
                "expected_next_runtime_event_id": "A1-EVT-045",
                "runtime_sync_status": "SYNCED_EVT_045",
                "artifact_payload_read_count_for_ledger": 0,
            },
            GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE217518/"
                    "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_20260812T201139P0800/"
                    "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_V1.json"
                ),
                "bytes": 6517,
                "sha256": "4e43db6030ee0839edb011a35858ba52177a719be23b3cae1774b5aac58ac1c9",
                "dataset_id": "GSE217518",
                "schema_version": "route_a_v3_gse217518_public_authority_preflight.v1",
                "record_type": "PUBLIC_AUTHORITY_PREFLIGHT_AGGREGATE_ONLY",
                "status": "STOP_BEFORE_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER",
                "aggregate_only": True,
                "ready_for_ordinary_public_row_level_producer": False,
                "qualified": False,
                "canonical_record_count": 0,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "producer_lineage": {
                    "implementation_commit": "a0e8bd7c751f94e116546d6164ec2de4faeae924",
                    "binding_commit": "bcdbd5e0735e950be92cee557785d5f72d2013e9",
                    "binding_diff_is_config_only": True,
                    "remote_head_at_registration": "bcdbd5e0735e950be92cee557785d5f72d2013e9",
                    "config_path": GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_CONFIG_PATH,
                    "config_sha256": GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256[
                        GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_CONFIG_PATH
                    ],
                    "script_path": GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_SCRIPT_PATH,
                    "script_sha256": GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256[
                        GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_TEST_PATH,
                    "focused_test_sha256": GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256[
                        GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-045",
                "expected_next_runtime_event_id": "A1-EVT-046",
                "runtime_sync_status": "SYNCED_EVT_046",
                "artifact_payload_read_count_for_ledger": 0,
            },
            GSE232572_PUBLIC_RECOVERY_AUDIT_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE232572/"
                    "GSE232572_A1_PUBLIC_RECOVERY_AUDIT_20260812T215745P0800/"
                    "GSE232572_A1_RECOVERY_REPORT.json"
                ),
                "bytes": 3041,
                "sha256": "0542feabf00496eb3c353df82abe61048c2b95b9bceb4b0429ebc668cc99dbbd",
                "dataset_id": "GSE232572",
                "schema_version": "1.0.0",
                "artifact_type": "GSE232572_PUBLIC_RECOVERY_AUDIT_AGGREGATE_ONLY",
                "status": "DEVELOPMENT_PRIVATE_RECONSTRUCTION_COMPLETE_NOT_QUALIFIED",
                "registry_role": "AUDIT_ONLY",
                "qualification_status": "AUDIT_PENDING",
                "aggregate_only": True,
                "published_universe_row_count": 11929,
                "accepted_pair_count": 8068,
                "rejected_published_row_count": 3861,
                "rejection_reason_counts": {
                    "NO_UNIQUE_SEQUENCE_PAIR": 3404,
                    "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
                },
                "development_reconstruction_record_count": 8068,
                "canonical_materialization_allowed": False,
                "canonical_record_count": 0,
                "qualified": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "producer_lineage": {
                    "implementation_commit": "4d04c8729a4e5596782c19cee332bcb3beaf2031",
                    "config_inspected_predecessor_commit": "99b1fc1ffd65f1a1e45b4390d6d7ab32bdd0d06e",
                    "config_inspected_predecessor_is_binding_commit": False,
                    "config_path": GSE232572_PUBLIC_RECOVERY_AUDIT_CONFIG_PATH,
                    "config_sha256": GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF_SHA256[
                        GSE232572_PUBLIC_RECOVERY_AUDIT_CONFIG_PATH
                    ],
                    "script_path": GSE232572_PUBLIC_RECOVERY_AUDIT_SCRIPT_PATH,
                    "script_sha256": GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF_SHA256[
                        GSE232572_PUBLIC_RECOVERY_AUDIT_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE232572_PUBLIC_RECOVERY_AUDIT_TEST_PATH,
                    "focused_test_sha256": GSE232572_PUBLIC_RECOVERY_AUDIT_STATIC_LEAF_SHA256[
                        GSE232572_PUBLIC_RECOVERY_AUDIT_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-046",
                "expected_next_runtime_event_id": "A1-EVT-047",
                "runtime_sync_status": "SYNCED_EVT_047",
                "artifact_payload_read_count_for_ledger": 0,
            },
            GSE232572_DEVELOPMENT_V3_MATERIALIZATION_FAILURE_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE232572/"
                    "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_20260812T233151P0800/"
                    "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT.json"
                ),
                "bytes": 1415,
                "sha256": "97367fc5cb84bf0b9d6d4bc90d23aacb29ca0cbe91b06b888abb31d54b317fa7",
                "dataset_id": "GSE232572",
                "schema_version": "1.0.0",
                "artifact_type": "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_ATTEMPT_001_FAIL_CLOSED_EVIDENCE",
                "status": "STOP_BEFORE_DEVELOPMENT_V3_ROW_PRODUCTION",
                "scientific_disposition": "NOT_QUALIFIED",
                "aggregate_only": True,
                "failure_gate": "RECOVERY_AUTHORITY",
                "failure_code": "MATERIALIZER_INPUTS_DIVERGE_FROM_RECOVERY_CONFIG",
                "schema_valid_development_record_count": 0,
                "canonical_record_count": 0,
                "qualified": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_allowed": False,
                "failed_attempt_preserved": True,
                "historical_attempt_rewritten": False,
                "superseded_for_current_execution_by_lineage_id": GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LINEAGE_ID,
                "producer_lineage": {
                    "implementation_i2_commit": "5619dc39622de7f97f63811d51a0e04bdf668e48",
                    "binding_b2_commit": "89db6313c6331e767ac5074170e7ff5b3cab8e3e",
                    "implementation_exact_changed_paths": [
                        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_SCRIPT_PATH,
                        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_TEST_PATH,
                    ],
                    "binding_exact_changed_paths": [
                        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_CONFIG_PATH,
                    ],
                    "binding_diff_is_config_only": True,
                },
                "predecessor_runtime_event_id": "A1-EVT-047",
                "expected_next_runtime_event_id": "A1-EVT-048",
                "runtime_sync_status": "SYNCED_EVT_048",
                "artifact_payload_read_count_for_ledger": 0,
                "private_jsonl_read_count_for_ledger": 0,
                "private_jsonl_registered_artifact_count": 0,
            },
            GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE232572/"
                    "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_20260812T234243P0800/"
                    "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT.json"
                ),
                "bytes": 3601,
                "sha256": "007a70b9b43b71cbc21b2614473126e53ab760df824ac9d95cb14304cc647ef3",
                "dataset_id": "GSE232572",
                "schema_version": "1.0.0",
                "artifact_type": "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT_AGGREGATE_ONLY",
                "status": "DEVELOPMENT_V3_MATERIALIZED_NOT_QUALIFIED",
                "scientific_disposition": "SCHEMA_VALID_DEVELOPMENT_ONLY_NOT_CANONICALLY_QUALIFIED",
                "aggregate_only": True,
                "published_universe_row_count": 11929,
                "schema_valid_development_record_count": 8068,
                "rejected_published_row_count": 3861,
                "rejection_reason_counts": {
                    "NO_UNIQUE_SEQUENCE_PAIR": 3404,
                    "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
                },
                "canonical_materialization_allowed": False,
                "canonical_record_count": 0,
                "qualified": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "public_redistribution_status": "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT",
                "row_license_status": "UNKNOWN_BLOCKED",
                "redistribution_allowed": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_allowed": False,
                "failed_attempt_lineage_id": GSE232572_DEVELOPMENT_V3_MATERIALIZATION_FAILURE_LINEAGE_ID,
                "failed_attempt_preserved": True,
                "historical_attempt_rewritten": False,
                "producer_lineage": {
                    "implementation_i3_commit": "e923d7b992293ca7bb5889bf3c0b3bc6ce750e03",
                    "binding_b3_commit": "b982275c25b7158a5a543a5e0c9fd23728fa0961",
                    "implementation_exact_changed_paths": [
                        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_CONFIG_PATH,
                        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_SCRIPT_PATH,
                        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_TEST_PATH,
                    ],
                    "binding_exact_changed_paths": [
                        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_CONFIG_PATH,
                    ],
                    "binding_diff_is_config_only": True,
                    "config_path": GSE232572_DEVELOPMENT_V3_MATERIALIZATION_CONFIG_PATH,
                    "config_bytes": 9484,
                    "config_sha256": GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF_SHA256[
                        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_CONFIG_PATH
                    ],
                    "script_path": GSE232572_DEVELOPMENT_V3_MATERIALIZATION_SCRIPT_PATH,
                    "script_bytes": 55522,
                    "script_sha256": GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF_SHA256[
                        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE232572_DEVELOPMENT_V3_MATERIALIZATION_TEST_PATH,
                    "focused_test_bytes": 31585,
                    "focused_test_sha256": GSE232572_DEVELOPMENT_V3_MATERIALIZATION_STATIC_LEAF_SHA256[
                        GSE232572_DEVELOPMENT_V3_MATERIALIZATION_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-047",
                "expected_next_runtime_event_id": "A1-EVT-048",
                "runtime_sync_status": "SYNCED_EVT_048",
                "artifact_payload_read_count_for_ledger": 0,
                "private_jsonl_read_count_for_ledger": 0,
                "private_jsonl_registered_artifact_count": 0,
            },
            GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE232572/"
                    "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_20260813T010116P0800/"
                    "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT.json"
                ),
                "bytes": 9586,
                "sha256": "00776c808cfa3e9ba2cfdb92b866c5f7c1bc92ea3818d17687cb9a8521b30d71",
                "dataset_id": "GSE232572",
                "schema_version": "route_a_v3_gse232572_a1_qualification_authority_preflight.v1",
                "artifact_type": "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_AGGREGATE_ONLY",
                "protocol_id": "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_V1",
                "overall_decision": "BLOCKED_MISSING_EXTERNAL_AUTHORITY",
                "terminal_status": "STOP_BEFORE_PRIVATE_ROW_ACCESS_AND_CANONICAL_MATERIALIZATION",
                "aggregate_only": True,
                "registered_aggregate_pass_count": 3,
                "registered_aggregate_passes": GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_PASSES,
                "open_qualification_blocker_count": 12,
                "qualification_blocker_statuses": GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_BLOCKERS,
                "published_universe_row_count": 11929,
                "schema_valid_development_record_count": 8068,
                "rejected_published_row_count": 3861,
                "public_replicate_count": 3,
                "primary_label_standard_error": None,
                "checkpoint_specific_exposure": "UNKNOWN_NOT_ASSERTED",
                "sequence_exposure": "SEQUENCE_EXPOSED",
                "label_exposure": "LABEL_EXPOSED",
                "untouched_confirmatory": False,
                "public_redistribution_status": "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT",
                "row_license_status": "UNKNOWN_BLOCKED",
                "redistribution_allowed": False,
                "recommended_future_scope": "PRIVATE_CANONICAL_ONLY",
                "recommended_future_scope_approved": False,
                "future_contribution_authorization_status": "NOT_AUTHORIZED",
                "maximum_ordinary_study_contribution_if_fully_qualified": 1,
                "maximum_a1_study_contribution_if_fully_qualified": 1,
                "maximum_true_a2_study_contribution_if_fully_qualified": 0,
                "canonical_record_count": 0,
                "qualified": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_allowed": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "producer_lineage": {
                    "base_commit": "13baa39e87406b5bc81b7e236cee637f694bfd0f",
                    "initial_implementation_commit": "cb10350681a1f4fd7dbe5322d671d618d77aaebf",
                    "lifecycle_repair_implementation_commit": "8ee914723b0d97d8ca07bab9ae7aaa1114e049dd",
                    "binding_commit": "d0778b92c1b90456a84bce60c7b7c3e039bc1ff5",
                    "initial_implementation_exact_changed_paths": [
                        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_CONFIG_PATH,
                        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_SCRIPT_PATH,
                        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_TEST_PATH,
                    ],
                    "lifecycle_repair_exact_changed_paths": [
                        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_SCRIPT_PATH,
                        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_TEST_PATH,
                    ],
                    "binding_exact_changed_paths": [
                        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_CONFIG_PATH,
                    ],
                    "binding_diff_is_config_only": True,
                    "remote_head_at_registration": "d0778b92c1b90456a84bce60c7b7c3e039bc1ff5",
                    "config_path": GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_CONFIG_PATH,
                    "config_sha256": GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256[
                        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_CONFIG_PATH
                    ],
                    "script_path": GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_SCRIPT_PATH,
                    "script_sha256": GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256[
                        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_TEST_PATH,
                    "focused_test_sha256": GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_STATIC_LEAF_SHA256[
                        GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-048",
                "expected_next_runtime_event_id": "A1-EVT-049",
                "runtime_sync_status": "SYNCED_EVT_049",
                "artifact_payload_read_count_for_ledger": 0,
                "private_row_artifact_read_count_for_ledger": 0,
                "private_jsonl_registered_artifact_count": 0,
            },
            GSE256185_PUBLIC_GEOMETRY_LINEAGE_ID: {
                "path": GSE256185_PUBLIC_GEOMETRY_REPORT_PATH,
                "bytes": GSE256185_PUBLIC_GEOMETRY_REPORT_BYTES,
                "sha256": GSE256185_PUBLIC_GEOMETRY_REPORT_SHA256,
                "recorded_at": GSE256185_PUBLIC_GEOMETRY_RECORDED_AT,
                "dataset_id": "GSE256185",
                "decision_id": "V3-DEC-021",
                "schema_version": "route_a_v3_gse256185_public_identifier_pool_geometry_preflight.v1",
                "protocol_id": "GSE256185_PUBLIC_IDENTIFIER_POOL_GEOMETRY_PREFLIGHT_V1",
                "status": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_COMPLETE_NOT_QUALIFIED",
                "authority_role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
                "aggregate_only": True,
                "preflight_complete": True,
                "aggregate_pool_geometry": {
                    "total_body_row_count": 11404,
                    "group_count": 652,
                    "single_parent_group_count": 637,
                    "dual_parent_group_count": 15,
                    "single_parent_groups_with_at_least_3_candidate_rows": 634,
                    "strict_candidate_rows_in_at_least_3_candidate_groups": 7292,
                    "reasoned_family_closure_candidate_rows_in_at_least_3_candidate_groups": 7294,
                    "identifier_grammar_anomaly_counts": {
                        "MISSING_GROUP_ROLE_DELIMITER": 1,
                        "UNSIGNED_CCC_ROLE": 1,
                    },
                    "strict_axis_is_frozen_observed_identifier_grammar": True,
                    "reasoned_family_closure_axis_status": "REASONED_FAMILY_CLOSURE_NOT_PUBLISHER_EXPLICIT",
                    "reasoned_family_closure_axis_is_publisher_explicit": False,
                },
                "scope_attestation": {
                    "ordinary_public_only": True,
                    "raw_asset_registered": False,
                    "row_record_output_count": 0,
                    "member_identifier_output_count": 0,
                    "member_role_output_count": 0,
                    "member_context_output_count": 0,
                    "sequence_value_output_count": 0,
                    "effect_value_output_count": 0,
                    "private_or_restricted_input_read_count": 0,
                    "sealed_contact_count": 0,
                },
                "sequence_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                "edit_budget_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                "effect_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                "true_a2_status_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                "qualification_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                "current_qualified_counts": {
                    "ordinary": 1,
                    "a1": 1,
                    "true_a2": 0,
                    "canonical_records": 6547,
                },
                "gse256185_contribution": {
                    "ordinary": 0,
                    "a1": 0,
                    "true_a2": 0,
                    "canonical_records": 0,
                },
                "qualified": False,
                "a1_complete": False,
                "training_allowed": False,
                "gpu_work_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "sole_next_action": "STOP_NO_FURTHER_ACTION_AUTHORIZED_BY_V3_DEC_021",
                "producer_lineage": {
                    "authority_commit": GSE256185_AUTHORITY_COMMIT,
                    "authority_runtime_i1_commit": GSE256185_AUTHORITY_RUNTIME_I1_COMMIT,
                    "authority_runtime_i2_commit": GSE256185_AUTHORITY_RUNTIME_I2_COMMIT,
                    "authority_runtime_binding_commit": GSE256185_AUTHORITY_RUNTIME_BINDING_COMMIT,
                    "preflight_i1_commit": GSE256185_PUBLIC_GEOMETRY_I1_COMMIT,
                    "preflight_i2_commit": GSE256185_PUBLIC_GEOMETRY_I2_COMMIT,
                    "preflight_b2_commit": GSE256185_PUBLIC_GEOMETRY_B2_COMMIT,
                    "implementation_commit": GSE256185_PUBLIC_GEOMETRY_IMPLEMENTATION_COMMIT,
                    "binding_commit": GSE256185_PUBLIC_GEOMETRY_BINDING_COMMIT,
                    "binding_diff_is_config_only": True,
                    "config_path": GSE256185_PUBLIC_GEOMETRY_CONFIG_PATH,
                    "config_sha256": GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF_SHA256[
                        GSE256185_PUBLIC_GEOMETRY_CONFIG_PATH
                    ],
                    "script_path": GSE256185_PUBLIC_GEOMETRY_SCRIPT_PATH,
                    "script_sha256": GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF_SHA256[
                        GSE256185_PUBLIC_GEOMETRY_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE256185_PUBLIC_GEOMETRY_TEST_PATH,
                    "focused_test_sha256": GSE256185_PUBLIC_GEOMETRY_STATIC_LEAF_SHA256[
                        GSE256185_PUBLIC_GEOMETRY_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-052",
                "expected_next_runtime_event_id": GSE256185_PUBLIC_GEOMETRY_RUNTIME_EVENT_ID,
                "next_runtime_event_id_preallocated": False,
                "runtime_sync_status": "SYNCED_EVT_053",
                "aggregate_report_read_count_for_ledger": 1,
                "private_row_artifact_read_count_for_ledger": 0,
                "raw_asset_registered_artifact_count": 0,
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
                    "sha256": GSE200304_DEC019_V3_HISTORICAL_D2_CONFIG_SHA256,
                    "config_core_sha256": GSE200304_DEC019_V3_HISTORICAL_CORE_SHA256,
                    "descriptor_set_sha256": GSE200304_DEC019_V3_HISTORICAL_D2_DESCRIPTOR_SET_SHA256,
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
                    "script_sha256": GSE200304_DEC019_V3_HISTORICAL_SCRIPT_SHA256,
                    "focused_test_path": GSE200304_DEC019_V3_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_V3_HISTORICAL_TEST_SHA256,
                },
            },
            GSE200304_UPSTREAM_AUTHORITY_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
                    "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_V1"
                ),
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "status": "COMMITTED_ACCEPTED_EXACT6",
                "evidence_role": "PUBLIC_UPSTREAM_AUTHORITY_AND_VIABILITY_ONLY",
                "exact_member_count": 6,
                "final_target_sha256": "ad9b64166586813d86c99de49589fff565dbe24eb48d7d6aeb07808fb390dfaa",
                "endpoint_semantics_supported": True,
                "biological_replicate_branch_supported": True,
                "biological_replicate_count": 6,
                "standard_error_status": "ABSENT_NOT_DERIVED_NOT_USED",
                "rights_scope": "PRIVATE_CANONICAL_ONLY",
                "public_redistribution_authorized": False,
                "qualified": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "canonical_record_count": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "producer_lineage": {
                    "base_commit": "0b95ac77a44644e57cc4d0bfb31a9154238fdca6",
                    "initial_implementation_commit": "9844246dd4b3874a9ecfcf03a233278c5d3a02e0",
                    "repair_implementation_commit": "7e29c13ca778ffa27f3725f4bd1ea270630db044",
                    "binding_commit": "9c313d2793880edd2a4355ec3781e045cae27252",
                    "binding_diff_is_config_only": True,
                    "config_path": GSE200304_UPSTREAM_AUTHORITY_CONFIG_PATH,
                    "config_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_UPSTREAM_AUTHORITY_CONFIG_PATH
                    ],
                    "script_path": GSE200304_UPSTREAM_AUTHORITY_SCRIPT_PATH,
                    "script_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_UPSTREAM_AUTHORITY_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE200304_UPSTREAM_AUTHORITY_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_UPSTREAM_AUTHORITY_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-041",
                "expected_next_runtime_event_id": "A1-EVT-042",
                "runtime_sync_status": "PENDING_NO_EVT_042",
            },
            GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
                    "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_V1"
                ),
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "status": "COMMITTED_ACCEPTED_EXACT6",
                "aggregate_only": True,
                "gate_record_count": 3,
                "exact_member_count": 6,
                "gate_status_counts": {"PASS": 3},
                "pass_slot_ids": [1, 3, 5],
                "pass_gate_ids": [
                    "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
                    "ROW_REPLICATE_OR_VALID_SE",
                    "LICENSE_RIGHTS",
                ],
                "rights_scope": "PRIVATE_CANONICAL_ONLY",
                "public_redistribution_authorized": False,
                "qualified": False,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "canonical_record_count": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "producer_lineage": {
                    "initial_implementation_commit": "82ee5213e8c41e9d5decc2e423525bf8da858d32",
                    "binding_commit": "ae4813a11b7e65e3aa118178f5d0e3d850cb73b8",
                    "binding_diff_is_config_only": True,
                    "config_path": GSE200304_DEC019_UPSTREAM_PASS_CONFIG_PATH,
                    "config_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_UPSTREAM_PASS_CONFIG_PATH
                    ],
                    "script_path": GSE200304_DEC019_UPSTREAM_PASS_SCRIPT_PATH,
                    "script_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_UPSTREAM_PASS_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE200304_DEC019_UPSTREAM_PASS_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_UPSTREAM_PASS_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-041",
                "expected_next_runtime_event_id": "A1-EVT-042",
                "runtime_sync_status": "PENDING_NO_EVT_042",
            },
            GSE200304_DEC019_UPSTREAM_PASS_ADJUDICATION_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3_"
                    "UPSTREAM_PASS_GATE_PACK_V1"
                ),
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
                "publication_state": "COMMITTED_ACCEPTED",
                "aggregate_only": True,
                "historical_predecessor_adjudication_lineage_id": GSE200304_DEC019_ADJUDICATION_LINEAGE_ID,
                "input_gate_count": 8,
                "pass_slot_ids": [0, 1, 3, 5],
                "input_status_counts": GSE200304_DEC019_UPSTREAM_PASS_INPUT_STATUS_COUNTS,
                "blocker_count": 4,
                "blockers": GSE200304_DEC019_UPSTREAM_PASS_BLOCKERS,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "positive_input_canonical_record_count": 6547,
                "canonical_record_count": 0,
                "positive_input_fact_is_not_final_canonical_materialization": True,
                "qualified": False,
                "independent_reproduction_established": False,
                "canonical_materialization_allowed": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "dynamic_config": {
                    "path": GSE200304_DEC019_V3_CONFIG_PATH,
                    "sha256": GSE200304_DEC019_V3_HISTORICAL_D3_CONFIG_SHA256,
                    "config_core_sha256": GSE200304_DEC019_V3_HISTORICAL_CORE_SHA256,
                    "descriptor_set_sha256": GSE200304_DEC019_V3_HISTORICAL_D3_DESCRIPTOR_SET_SHA256,
                    "exact_full_sha_in_static_manifest": False,
                },
                "consumer_lineage": {
                    "descriptor_d2_commit": "c278f29a18b7858c85686fcec3857a992fd07d5f",
                    "upstream_authority_binding_commit": "9c313d2793880edd2a4355ec3781e045cae27252",
                    "upstream_pass_implementation_commit": "82ee5213e8c41e9d5decc2e423525bf8da858d32",
                    "upstream_pass_binding_commit": "ae4813a11b7e65e3aa118178f5d0e3d850cb73b8",
                    "descriptor_d3_commit": "8084a1e2b68eaf84bd4befb2f232759d7540b97c",
                    "descriptor_d3_parent_commit": "ae4813a11b7e65e3aa118178f5d0e3d850cb73b8",
                    "descriptor_d3_diff_is_exactly_one_config_path": True,
                    "script_path": GSE200304_DEC019_V3_SCRIPT_PATH,
                    "script_sha256": GSE200304_DEC019_V3_HISTORICAL_SCRIPT_SHA256,
                    "focused_test_path": GSE200304_DEC019_V3_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_V3_HISTORICAL_TEST_SHA256,
                },
                "predecessor_runtime_event_id": "A1-EVT-041",
                "expected_next_runtime_event_id": "A1-EVT-042",
                "runtime_sync_status": "PENDING_NO_EVT_042",
                "runtime_evidence_payload_read_count": 0,
            },
            GSE200304_DEC019_GROUP_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
                    "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE_V1"
                ),
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "status": "PASS",
                "publication_state": "COMMITTED_EXACT",
                "aggregate_only": True,
                "exact_member_count": 4,
                "private_member_count": 1,
                "private_member_payload_read_count": 0,
                "biological_group_count": 6544,
                "changes_study_qualification": False,
                "producer_lineage": {
                    "config_path": GSE200304_DEC019_GROUP_CONFIG_PATH,
                    "config_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_GROUP_CONFIG_PATH
                    ],
                    "script_path": GSE200304_DEC019_GROUP_SCRIPT_PATH,
                    "script_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_GROUP_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE200304_DEC019_GROUP_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_GROUP_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-043",
                "expected_next_runtime_event_id": "A1-EVT-044",
                "runtime_sync_status": "SYNCED_EVT_044",
            },
            GSE200304_DEC019_SPLIT_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
                    "GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE_V1"
                ),
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "status": "PASS",
                "publication_state": "COMMITTED_EXACT",
                "aggregate_only": True,
                "exact_member_count": 4,
                "private_member_count": 1,
                "private_member_payload_read_count": 0,
                "record_count": 6547,
                "biological_group_count": 6544,
                "connected_component_count": 1936,
                "maximum_component_size": 26,
                "outer_group_counts": [1309, 1309, 1309, 1309, 1308],
                "cross_fold_component_leakage_count": 0,
                "cross_fold_gene_leakage_count": 0,
                "cross_fold_hamming_leakage_count": 0,
                "cross_fold_jaccard_leakage_count": 0,
                "outcome_column_read_count": 0,
                "changes_study_qualification": False,
                "producer_lineage": {
                    "config_path": GSE200304_DEC019_SPLIT_CONFIG_PATH,
                    "config_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_SPLIT_CONFIG_PATH
                    ],
                    "script_path": GSE200304_DEC019_SPLIT_SCRIPT_PATH,
                    "script_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_SPLIT_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE200304_DEC019_SPLIT_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_SPLIT_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-043",
                "expected_next_runtime_event_id": "A1-EVT-044",
                "runtime_sync_status": "SYNCED_EVT_044",
            },
            GSE200304_DEC019_POWER_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
                    "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE_V1"
                ),
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "status": "PASS",
                "publication_state": "COMMITTED_EXACT",
                "aggregate_only": True,
                "exact_member_count": 2,
                "planning_only": True,
                "observed_model_results_used": False,
                "a2_final_membership_used": False,
                "planning_unit": "BIOLOGICAL_SOURCE_GROUP",
                "planning_group_count": 6544,
                "target_spearman_rho": 0.25,
                "two_sided_alpha": 0.05,
                "estimated_design_power": 1.0,
                "planned_full_ci_width": 0.04613579821079131,
                "target_power_minimum": 0.8,
                "maximum_full_ci_width": 0.3,
                "working_distribution_assumption": "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO",
                "changes_study_qualification": False,
                "producer_lineage": {
                    "config_path": GSE200304_DEC019_POWER_CONFIG_PATH,
                    "config_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_POWER_CONFIG_PATH
                    ],
                    "script_path": GSE200304_DEC019_POWER_SCRIPT_PATH,
                    "script_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_POWER_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE200304_DEC019_POWER_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_POWER_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-043",
                "expected_next_runtime_event_id": "A1-EVT-044",
                "runtime_sync_status": "SYNCED_EVT_044",
            },
            GSE200304_DEC019_ONE_BLOCKER_ADJUDICATION_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3_"
                    "GROUP_SPLIT_POWER_PASS_D6"
                ),
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
                "publication_state": "COMMITTED_EXACT",
                "aggregate_only": True,
                "historical_predecessor_adjudication_lineage_id": GSE200304_DEC019_UPSTREAM_PASS_ADJUDICATION_LINEAGE_ID,
                "settled_gate_lineage_ids": [
                    GSE200304_DEC019_GROUP_LINEAGE_ID,
                    GSE200304_DEC019_SPLIT_LINEAGE_ID,
                    GSE200304_DEC019_POWER_LINEAGE_ID,
                ],
                "input_gate_count": 8,
                "pass_slot_ids": GSE200304_DEC019_ONE_BLOCKER_PASS_SLOT_IDS,
                "input_status_counts": GSE200304_DEC019_ONE_BLOCKER_INPUT_STATUS_COUNTS,
                "blocker_count": 1,
                "blockers": GSE200304_DEC019_ONE_BLOCKER_BLOCKERS,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "positive_input_canonical_record_count": 6547,
                "canonical_record_count": 0,
                "positive_input_fact_is_not_final_canonical_materialization": True,
                "qualified": False,
                "independent_reproduction_established": False,
                "canonical_materialization_allowed": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "power_evidence_is_planning_only": True,
                "dynamic_config": {
                    "path": GSE200304_DEC019_V3_CONFIG_PATH,
                    "sha256": GSE200304_DEC019_V3_CONFIG_SHA256,
                    "config_core_sha256": GSE200304_DEC019_V3_CONFIG_CORE_SHA256,
                    "descriptor_set_sha256": GSE200304_DEC019_V3_DESCRIPTOR_SET_SHA256,
                    "exact_full_sha_in_static_manifest": False,
                },
                "consumer_lineage": {
                    "script_path": GSE200304_DEC019_V3_SCRIPT_PATH,
                    "script_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_V3_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE200304_DEC019_V3_TEST_PATH,
                    "focused_test_sha256": GSE200304_DEC019_POST_ADJUDICATION_STATIC_LEAF_SHA256[
                        GSE200304_DEC019_V3_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-043",
                "expected_next_runtime_event_id": "A1-EVT-044",
                "runtime_sync_status": "SYNCED_EVT_044",
                "runtime_evidence_payload_read_count": 0,
            },
            GSE200304_CHECKPOINT_EXPOSURE_FAIL_LINEAGE_ID: {
                "path": (
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
                    "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
                    "GSE200304_DEC019_CHECKPOINT_EXPOSURE_FAIL_CURRENT_PROTOCOL.json"
                ),
                "bytes": 10391,
                "sha256": "f2e111cd9c3c02726cf47e43220b371bbdd0fac9295f39043a9bc8a7b781da6c",
                "dataset_id": "GSE200304",
                "decision_id": "V3-DEC-019",
                "record_type": (
                    "GSE200304_DEC019_CHECKPOINT_EXPOSURE_FAIL_"
                    "CURRENT_PROTOCOL_AGGREGATE_ONLY_V1"
                ),
                "status": "FAIL_CURRENT_PROTOCOL",
                "aggregate_only": True,
                "considered_candidate_family_count": 4,
                "task_mismatch_candidate_family_count": 4,
                "current_public_executable_foundation_checkpoint_count": 0,
                "audited_checkpoint_count": 0,
                "current_exposure_gate_status": "UNKNOWN_NOT_ASSERTED",
                "exact_blocker": "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS",
                "changes_qualification_gate": False,
                "qualified": False,
                "scientific_claim_status": "NOT_ESTABLISHED",
                "canonical_record_count": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "producer_lineage": {
                    "implementation_commit": "5de083e19b6090c045854900532f68247d8b59c6",
                    "binding_commit": "d87631b16501072b45bef3016bdbaf00c87cc59f",
                    "binding_diff_is_config_only": True,
                    "config_path": GSE200304_CHECKPOINT_EXPOSURE_FAIL_CONFIG_PATH,
                    "config_sha256": POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256[
                        GSE200304_CHECKPOINT_EXPOSURE_FAIL_CONFIG_PATH
                    ],
                    "script_path": GSE200304_CHECKPOINT_EXPOSURE_FAIL_SCRIPT_PATH,
                    "script_sha256": POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256[
                        GSE200304_CHECKPOINT_EXPOSURE_FAIL_SCRIPT_PATH
                    ],
                    "focused_test_path": GSE200304_CHECKPOINT_EXPOSURE_FAIL_TEST_PATH,
                    "focused_test_sha256": POST_FAIL_ACQUISITION_STATIC_LEAF_SHA256[
                        GSE200304_CHECKPOINT_EXPOSURE_FAIL_TEST_PATH
                    ],
                },
                "predecessor_runtime_event_id": "A1-EVT-044",
                "expected_next_runtime_event_id": "A1-EVT-045",
                "runtime_sync_status": "SYNCED_EVT_045",
                "artifact_payload_read_count_for_ledger": 0,
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
        expected_gse200304_lineage[GSE200304_DEC020_V4_LINEAGE_ID] = {
            "path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_DEC020_SCRATCH_ROUTE_A1_ADJUDICATION_V4_20260813T143426P0800",
            "dataset_id": "GSE200304",
            "decision_id": "V3-DEC-020",
            "status": "PASS_DEC020_SCRATCH_ROUTE_SCOPED_REPORTED_ENDPOINT_A1_QUALIFIED",
            "publication_state": "COMMITTED_EXACT",
            "aggregate_only": True,
            "historical_predecessor_adjudication_lineage_id": GSE200304_DEC019_ONE_BLOCKER_ADJUDICATION_LINEAGE_ID,
            "input_gate_count": 7,
            "pass_slot_ids": [0, 1, 2, 3, 4, 5, 6],
            "input_status_counts": {"PASS": 7},
            "blocker_count": 0,
            "blockers": [],
            "ordinary_study_contribution": 1,
            "a1_study_contribution": 1,
            "true_a2_study_contribution": 0,
            "positive_input_canonical_record_count": 6547,
            "canonical_record_count": 6547,
            "positive_input_fact_is_not_final_canonical_materialization": True,
            "qualified": True,
            "independent_reproduction_established": False,
            "canonical_materialization_qualification_eligible": True,
            "canonical_materialization_allowed": False,
            "canonical_materialization_execution_authorized": False,
            "training_allowed": False,
            "gpu_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "foundation_route_status": "RETAINED_FAIL_CURRENT_PROTOCOL",
            "foundation_checkpoint_evidence_status": "UNKNOWN_NOT_ASSERTED",
            "private_payload_access_authorized": False,
            "row_level_payload_read_count": 0,
            "private_payload_read_count": 0,
            "sealed_payload_read_count": 0,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "dynamic_config": {
                "path": GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH,
                "config_core_sha256": GSE200304_DEC020_V4_CONFIG_CORE_SHA256,
                "descriptor_set_sha256": GSE200304_DEC020_V4_DESCRIPTOR_SET_SHA256,
                "exact_full_sha_in_static_manifest": True,
            },
            "consumer_lineage": {
                "implementation_commit": GSE200304_DEC020_V4_IMPLEMENTATION_COMMIT,
                "binding_commit": GSE200304_DEC020_V4_BINDING_COMMIT,
                "script_path": GSE200304_DEC020_V4_SCRIPT_PATH,
                "script_sha256": GSE200304_DEC020_V4_STATIC_LEAF_SHA256[GSE200304_DEC020_V4_SCRIPT_PATH],
                "focused_test_path": GSE200304_DEC020_V4_TEST_PATH,
                "focused_test_sha256": GSE200304_DEC020_V4_STATIC_LEAF_SHA256[GSE200304_DEC020_V4_TEST_PATH],
            },
            "predecessor_runtime_event_id": "A1-EVT-050",
            "expected_next_runtime_event_id": "A1-EVT-051",
            "runtime_sync_status": "SYNCED_EVT_051",
            "runtime_evidence_payload_read_count": 0,
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
        upstream_authority_root = (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
            "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_V1"
        )
        upstream_pass_gate_root = (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
            "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_V1"
        )
        upstream_pass_adjudication_root = (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3_"
            "UPSTREAM_PASS_GATE_PACK_V1"
        )
        group_gate_root = (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
            "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE_V1"
        )
        split_gate_root = (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
            "GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE_V1"
        )
        power_gate_root = (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/"
            "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE_V1"
        )
        one_blocker_adjudication_root = (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3_"
            "GROUP_SPLIT_POWER_PASS_D6"
        )
        v4_adjudication_root = (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "GSE200304_DEC020_SCRATCH_ROUTE_A1_ADJUDICATION_V4_20260813T143426P0800"
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
            GSE200304_UPSTREAM_AUTHORITY_LINEAGE_ID: _closed_files(
                upstream_authority_root,
                (
                    (
                        "PMC10540565_EUROPE_PMC_FULLTEXT.xml",
                        298763,
                        "4fe53c9ea58b5268b1014c0ef4b18cfbd7b5b3764f4c82542c065cb0aff5a7f0",
                    ),
                    (
                        "GSE200302_family.soft.gz",
                        4699,
                        "6df39a3406fe1bdf5a37345fee5605510ca1086fbce54d5aeeb934b562bb7d2e",
                    ),
                    (
                        "GSE200302_log2_cpm_counts_all_samples.txt.gz",
                        2843042,
                        "ed93162f9540676138cfba05af2841c90619ac4335eb55ee3d956a3cd8aace3c",
                    ),
                    (
                        "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT.json",
                        10427,
                        "997101dda5cbe3cf5a97bcfe9dda07150d11552decc159a2bc3cb96d9ebd0e45",
                    ),
                    (
                        "SHA256SUMS",
                        420,
                        "5f00d0d75ef8f12de5ed903a2c599498e5a6717f13a32b95d3f33765522ba371",
                    ),
                    (
                        "PUBLICATION_COMMIT.json",
                        1031,
                        "1dc26e885964bb15a2fad1ebb18e4ebf89fdf888e08e4b02058f396b2a4db664",
                    ),
                ),
            ),
            GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_LINEAGE_ID: _closed_files(
                upstream_pass_gate_root,
                (
                    (
                        "GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_GATE.json",
                        3988,
                        "6603803960b747126a5b6dfb7d56bf124d36144fa87667098813ccae2fe41ba3",
                    ),
                    (
                        "GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_GATE.json",
                        3944,
                        "dc0a08a1a6b389fcd4c982a7e52ad34ebc9cf67563482c6adc84a7c2c51b3d0f",
                    ),
                    (
                        "GSE200304_DEC019_LICENSE_RIGHTS_GATE.json",
                        3905,
                        "08cb30aeac3b6e1e989e0d379b0b51c83a7fcbea6f4a3bb0501b4529d3a5192c",
                    ),
                    (
                        "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT.json",
                        2171,
                        "bdfc4d8c7cf941e28e545cf70b33ac12cf0ca7fae02914b95c15bef46fef7cf2",
                    ),
                    (
                        "SHA256SUMS",
                        476,
                        "91cee112a8daa4fb562c76fe6a579146a9f1e7495785cbd24131c1032e6761c2",
                    ),
                    (
                        "PUBLICATION_COMMIT.json",
                        1362,
                        "f22e074f049db71e20fac05b58dad17953232b4d651aee460c3a3c27b3a185a3",
                    ),
                ),
            ),
            GSE200304_DEC019_UPSTREAM_PASS_ADJUDICATION_LINEAGE_ID: _closed_files(
                upstream_pass_adjudication_root,
                (
                    (
                        "ADJUDICATION_REPORT.json",
                        2359,
                        "cc1423d84add812380641998c4e36e7096c10eaaaf74ed12c3b781b45fc4cece",
                    ),
                    (
                        "INPUT_EVIDENCE_AUDIT.json",
                        2983,
                        "72d836ecb373fd3841c9c3f91b6777172d979831bea7042c9a3da30f16040352",
                    ),
                    (
                        "SHA256SUMS",
                        183,
                        "bff424c43fd392148a2d8417b171f325badb9adae1b2366010de4cfaca887dc6",
                    ),
                    (
                        "PUBLICATION_COMMIT.json",
                        1055,
                        "b4094b0621d50a18fe5ab64d1662b4ee95cfdd93c572e25f111e9b53d2586b42",
                    ),
                ),
            ),
            GSE200304_DEC019_GROUP_LINEAGE_ID: _closed_files(
                group_gate_root,
                (
                    (
                        "GSE200304_DEC019_BIOLOGICAL_GROUP_MAPPING_PRIVATE.json",
                        1336202,
                        "33abf14479671e264c3fbd8bae58e350d132d5c8f1cfdc65d766b1668b8f0229",
                    ),
                    (
                        "GSE200304_DEC019_BIOLOGICAL_GROUP_MAPPING_AUDIT.json",
                        1232,
                        "dc1cb607ba28e1cc34cd76bf6543c4394887302c745597033ef6dc7f338eb177",
                    ),
                    (
                        "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE.json",
                        4013,
                        "bbcc1f93f47190c7070755d8e892c07d43959985fee32f4f4ec3911fe6f7993b",
                    ),
                    (
                        "PUBLICATION_COMMIT.json",
                        693,
                        "7ba6701c48fe0c6abf9be2cee22b58c95aeaf594123fe89db6d6339af07b156f",
                    ),
                ),
            ),
            GSE200304_DEC019_SPLIT_LINEAGE_ID: _closed_files(
                split_gate_root,
                (
                    (
                        "GSE200304_DEC019_SPLIT_ASSIGNMENT_PRIVATE.json",
                        2094877,
                        "1e330df0fb34fcf860003389f380efae3969522dbf65755ee0d8081a50b3dbca",
                    ),
                    (
                        "GSE200304_DEC019_SPLIT_LEAKAGE_AUDIT.json",
                        5566,
                        "32780f6192a03133804b2b5518c2cd2df9a3e005cee0ba7c9cd6f9c10b0a9584",
                    ),
                    (
                        "GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE.json",
                        4120,
                        "8db5cb2d12c775be561c04049a381d6541502bd2fefeda12a981b65be240adf7",
                    ),
                    (
                        "PUBLICATION_COMMIT.json",
                        664,
                        "e0de4928b9127c58d5c90fa09a05d92afb379cf7ee9ff31ece1158042b208caa",
                    ),
                ),
            ),
            GSE200304_DEC019_POWER_LINEAGE_ID: _closed_files(
                power_gate_root,
                (
                    (
                        "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_AUDIT.json",
                        1782,
                        "874321a1a044e0f151c6c081482524758115e8bd9d6f03807ac685cee5e365ed",
                    ),
                    (
                        "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE.json",
                        4535,
                        "2432e1dfae92639e92060510aa2d9a413664d9b2d78f3655781d8e6bb8d27811",
                    ),
                ),
            ),
            GSE200304_DEC019_ONE_BLOCKER_ADJUDICATION_LINEAGE_ID: _closed_files(
                one_blocker_adjudication_root,
                (
                    (
                        "ADJUDICATION_REPORT.json",
                        2230,
                        "966a12e3ac44587bd6f4949d022fdc4b7de5e9cef4fd777f7348382a5a41ff73",
                    ),
                    (
                        "INPUT_EVIDENCE_AUDIT.json",
                        2974,
                        "3ce0496b5c49fa0ca82c0b0af66b7ce18de3939a7f56b7cd77a03f295604f1fe",
                    ),
                    (
                        "SHA256SUMS",
                        183,
                        "9b6f9f1e9042a28d9af267e2af33d2c1991898c7afca824542d9491b540ebc7f",
                    ),
                    (
                        "PUBLICATION_COMMIT.json",
                        1055,
                        "c96530cf37ff0eed803adf24bfde6e6db0058ddf8264c63acf1ff95700a32ca4",
                    ),
                ),
            ),
            GSE200304_DEC020_V4_LINEAGE_ID: _closed_files(
                v4_adjudication_root,
                (
                    ("ADJUDICATION_REPORT.json", 2812, "37ced466a905f8e71bb06ec55550a36d8c2e1d0ff8f03c6dde839562c1680d99"),
                    ("INPUT_EVIDENCE_AUDIT.json", 2891, "3fc8171d6f69fb27e3a454bf504efacddcddceb99fffe8b47b59c84c554d1d72"),
                    ("SHA256SUMS", 183, "c11d227abbf5700c7a9a1b403236e331377f23e20c7192b9effe7bd6428f3fb3"),
                    ("PUBLICATION_COMMIT.json", 1082, "6597863e9eef342e6539fa886d81b373157f8248baacd519d7b0ebc60a3af2ac"),
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

    _validate_gse256185_row_preflight_interim(interim, path, issues)
    _validate_dec023_dual_preflight_interim(interim, path, issues)
    _validate_dec027_six_rescue_interim(interim, path, issues)

    summary = interim.get("dataset_boundary_summary")
    if not isinstance(summary, Mapping):
        _issue(issues, "A1_INTERIM_DATASET_BOUNDARY", path, "dataset_boundary_summary must be a mapping")
    else:
        expected_summary_keys = {
            "evidence_ref",
            "GSE114002",
            "GSE149487",
            "GSE217518",
            "GSE232572",
            "GSE256185",
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
                "public_asset_acquisition": {
                    "artifact_lineage_id": GSE149487_PUBLIC_ASSET_ACQUISITION_LINEAGE_ID,
                    "status": "STOPPED_WITH_PUBLIC_EVIDENCE_BLOCKER",
                    "acquisition_status": "EXACT_21_ASSETS_ACQUIRED_AND_INTEGRITY_VERIFIED",
                    "asset_count": 21,
                    "geo_raw_count": 18,
                    "supplement_count": 3,
                    "total_verified_bytes": 70032274,
                    "ready_for_full_qualifier_input": False,
                    "ready_for_study_qualification": False,
                    "qualified": False,
                    "canonical_record_count": 0,
                    "ordinary_study_contribution": 0,
                    "a1_study_contribution": 0,
                    "true_a2_study_contribution": 0,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "changes_qualification_gate": False,
                    "predecessor_runtime_event_id": "A1-EVT-044",
                    "expected_next_runtime_event_id": "A1-EVT-045",
                    "runtime_sync_status": "SYNCED_EVT_045",
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
        gse217518 = summary.get("GSE217518")
        if not isinstance(gse217518, Mapping):
            _issue(
                issues,
                "A1_INTERIM_GSE217518",
                path,
                "GSE217518 boundary must be a mapping",
            )
        else:
            expected_gse217518 = {
                "registry_qualification_status": "AUDIT_PENDING",
                "a1_inventory_qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
                "public_authority_preflight": {
                    "artifact_lineage_id": GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_LINEAGE_ID,
                    "status": "STOP_BEFORE_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER",
                    "ready_for_ordinary_public_row_level_producer": False,
                    "qualified": False,
                    "canonical_record_count": 0,
                    "ordinary_study_contribution": 0,
                    "a1_study_contribution": 0,
                    "true_a2_study_contribution": 0,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "changes_qualification_gate": False,
                    "predecessor_runtime_event_id": "A1-EVT-045",
                    "expected_next_runtime_event_id": "A1-EVT-046",
                    "runtime_sync_status": "SYNCED_EVT_046",
                },
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            }
            _expect_closed_mapping(
                gse217518,
                expected_gse217518,
                path,
                issues,
                "A1_INTERIM_GSE217518",
            )
        gse232572 = summary.get("GSE232572")
        if not isinstance(gse232572, Mapping):
            _issue(
                issues,
                "A1_INTERIM_GSE232572",
                path,
                "GSE232572 boundary must be a mapping",
            )
        else:
            expected_gse232572 = {
                "registry_qualification_status": "AUDIT_PENDING",
                "a1_inventory_qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
                "public_recovery_audit": {
                    "artifact_lineage_id": GSE232572_PUBLIC_RECOVERY_AUDIT_LINEAGE_ID,
                    "artifact_type": "GSE232572_PUBLIC_RECOVERY_AUDIT_AGGREGATE_ONLY",
                    "status": "DEVELOPMENT_PRIVATE_RECONSTRUCTION_COMPLETE_NOT_QUALIFIED",
                    "registry_role": "AUDIT_ONLY",
                    "qualification_status": "AUDIT_PENDING",
                    "published_universe_row_count": 11929,
                    "accepted_pair_count": 8068,
                    "rejected_published_row_count": 3861,
                    "rejection_reason_counts": {
                        "NO_UNIQUE_SEQUENCE_PAIR": 3404,
                        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
                    },
                    "development_reconstruction_record_count": 8068,
                    "canonical_materialization_allowed": False,
                    "canonical_record_count": 0,
                    "qualified": False,
                    "ordinary_study_contribution": 0,
                    "a1_study_contribution": 0,
                    "true_a2_study_contribution": 0,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "changes_qualification_gate": False,
                    "predecessor_runtime_event_id": "A1-EVT-046",
                    "expected_next_runtime_event_id": "A1-EVT-047",
                    "runtime_sync_status": "SYNCED_EVT_047",
                },
                "development_v3_materialization": {
                    "failed_attempt_artifact_lineage_id": GSE232572_DEVELOPMENT_V3_MATERIALIZATION_FAILURE_LINEAGE_ID,
                    "current_artifact_lineage_id": GSE232572_DEVELOPMENT_V3_MATERIALIZATION_LINEAGE_ID,
                    "artifact_type": "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT_AGGREGATE_ONLY",
                    "status": "DEVELOPMENT_V3_MATERIALIZED_NOT_QUALIFIED",
                    "scientific_disposition": "SCHEMA_VALID_DEVELOPMENT_ONLY_NOT_CANONICALLY_QUALIFIED",
                    "failed_attempt_preserved": True,
                    "historical_attempt_rewritten": False,
                    "published_universe_row_count": 11929,
                    "schema_valid_development_record_count": 8068,
                    "rejected_published_row_count": 3861,
                    "rejection_reason_counts": {
                        "NO_UNIQUE_SEQUENCE_PAIR": 3404,
                        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
                    },
                    "canonical_materialization_allowed": False,
                    "canonical_record_count": 0,
                    "qualified": False,
                    "ordinary_study_contribution": 0,
                    "a1_study_contribution": 0,
                    "true_a2_study_contribution": 0,
                    "public_redistribution_status": "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT",
                    "row_license_status": "UNKNOWN_BLOCKED",
                    "redistribution_allowed": False,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_allowed": False,
                    "private_jsonl_read_count": 0,
                    "private_jsonl_registered_artifact_count": 0,
                    "changes_qualification_gate": False,
                    "predecessor_runtime_event_id": "A1-EVT-047",
                    "expected_next_runtime_event_id": "A1-EVT-048",
                    "runtime_sync_status": "SYNCED_EVT_048",
                },
                "qualification_authority_preflight": {
                    "artifact_lineage_id": GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_LINEAGE_ID,
                    "artifact_type": "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_AGGREGATE_ONLY",
                    "protocol_id": "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_V1",
                    "overall_decision": "BLOCKED_MISSING_EXTERNAL_AUTHORITY",
                    "terminal_status": "STOP_BEFORE_PRIVATE_ROW_ACCESS_AND_CANONICAL_MATERIALIZATION",
                    "aggregate_only": True,
                    "registered_aggregate_pass_count": 3,
                    "open_qualification_blocker_count": 12,
                    "published_universe_row_count": 11929,
                    "schema_valid_development_record_count": 8068,
                    "canonical_record_count": 0,
                    "qualified": False,
                    "ordinary_study_contribution": 0,
                    "a1_study_contribution": 0,
                    "true_a2_study_contribution": 0,
                    "checkpoint_specific_exposure": "UNKNOWN_NOT_ASSERTED",
                    "public_redistribution_status": "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT",
                    "row_license_status": "UNKNOWN_BLOCKED",
                    "recommended_future_scope": "PRIVATE_CANONICAL_ONLY",
                    "recommended_future_scope_approved": False,
                    "future_contribution_authorization_status": "NOT_AUTHORIZED",
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_allowed": False,
                    "changes_qualification_gate": False,
                    "predecessor_runtime_event_id": "A1-EVT-048",
                    "expected_next_runtime_event_id": "A1-EVT-049",
                    "runtime_sync_status": "SYNCED_EVT_049",
                },
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            }
            _expect_closed_mapping(
                gse232572,
                expected_gse232572,
                path,
                issues,
                "A1_INTERIM_GSE232572",
            )
        gse256185 = summary.get("GSE256185")
        if not isinstance(gse256185, Mapping):
            _issue(
                issues,
                "A1_INTERIM_GSE256185",
                path,
                "GSE256185 boundary must be a mapping",
            )
        else:
            expected_gse256185 = {
                "registry_qualification_status": "AUDIT_PENDING",
                "authority_role": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_ONLY",
                "active_authority_role": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
                "public_identifier_pool_geometry_preflight": {
                    "artifact_lineage_id": GSE256185_PUBLIC_GEOMETRY_LINEAGE_ID,
                    "status": "PUBLIC_IDENTIFIER_AND_POOL_GEOMETRY_PREFLIGHT_COMPLETE_NOT_QUALIFIED",
                    "aggregate_only": True,
                    "preflight_complete": True,
                    "total_body_row_count": 11404,
                    "group_count": 652,
                    "single_parent_group_count": 637,
                    "dual_parent_group_count": 15,
                    "single_parent_groups_with_at_least_3_candidate_rows": 634,
                    "strict_candidate_rows_in_at_least_3_candidate_groups": 7292,
                    "reasoned_family_closure_candidate_rows_in_at_least_3_candidate_groups": 7294,
                    "missing_group_role_delimiter_anomaly_count": 1,
                    "unsigned_ccc_role_anomaly_count": 1,
                    "sequence_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                    "edit_budget_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                    "effect_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                    "true_a2_status_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                    "qualification_evaluation": "OUT_OF_SCOPE_NOT_EVALUATED",
                    "qualified": False,
                    "canonical_record_count": 0,
                    "ordinary_study_contribution": 0,
                    "a1_study_contribution": 0,
                    "true_a2_study_contribution": 0,
                    "changes_qualification_gate": False,
                    "current_global_qualified_counts": {
                        "ordinary": 1,
                        "a1": 1,
                        "true_a2": 0,
                        "canonical_records": 6547,
                    },
                    "training_allowed": False,
                    "gpu_work_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "predecessor_runtime_event_id": "A1-EVT-052",
                    "expected_next_runtime_event_id": GSE256185_PUBLIC_GEOMETRY_RUNTIME_EVENT_ID,
                    "next_runtime_event_id_preallocated": False,
                    "runtime_sync_status": "SYNCED_EVT_053",
                },
                "aggregate_row_level_qualification_preflight": {
                    "artifact_lineage_id": GSE256185_ROW_PREFLIGHT_LINEAGE_ID,
                    "decision_id": "V3-DEC-022",
                    "protocol_id": "GSE256185_AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_V1",
                    "status": "STOP_CURRENT_PROTOCOL_NOT_QUALIFIED",
                    "preflight_complete": True,
                    "strict_single_parent_pool_count": 634,
                    "strict_candidate_member_count": 7292,
                    "dual_parent_group_count_excluded": 15,
                    "strict_single_parent_two_candidate_group_count_excluded": 3,
                    "nonstrict_grammar_record_count_excluded": 2,
                    "reasoned_family_closure_candidate_count": 7294,
                    "reasoned_family_closure_included": False,
                    "independent_fail_closed_gate_axis_count": 17,
                    "required_gate_statuses": GSE256185_ROW_PREFLIGHT_GATE_STATUSES,
                    "all_required_gates_pass": False,
                    "eligible_after_row_preflight_exclusions": {
                        "pool_count": 633,
                        "parent_row_count": 633,
                        "candidate_row_count": 7288,
                        "row_count": 7921,
                    },
                    "source_to_candidate_edit_relation_established": False,
                    "qualified": False,
                    "canonical_record_count": 0,
                    "ordinary_study_contribution": 0,
                    "a1_study_contribution": 0,
                    "true_a2_study_contribution": 0,
                    "output_class": "AGGREGATE_ROW_LEVEL_QUALIFICATION_PREFLIGHT_ONLY",
                    "row_or_member_output_allowed": False,
                    "sequence_output_allowed": False,
                    "row_effect_output_allowed": False,
                    "replicate_identifier_output_allowed": False,
                    "split_assignment_output_allowed": False,
                    "training_allowed": False,
                    "gpu_work_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "predecessor_runtime_event_id": DEC022_AUTHORITY_RUNTIME_EVENT_ID,
                    "expected_next_runtime_event_id": GSE256185_ROW_PREFLIGHT_RUNTIME_EVENT_ID,
                    "next_runtime_event_id_preallocated": False,
                    "runtime_sync_status": "SYNCED_EVT_055",
                },
                "qualified": False,
                "canonical_record_count": 0,
                "ordinary_study_contribution": 0,
                "a1_study_contribution": 0,
                "true_a2_study_contribution": 0,
                "training_allowed": False,
                "gpu_work_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            }
            _expect_closed_mapping(
                gse256185,
                expected_gse256185,
                path,
                issues,
                "A1_INTERIM_GSE256185",
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
                "source_grouping_status": "BIOLOGICAL_GROUP_AUTHORITY_PASS",
                "license_and_redistribution_status": "PRIVATE_CANONICAL_ONLY",
                "checkpoint_specific_foundation_exposure_status": "NOT_PASS",
                "canonical_intervention_record_count": 0,
                "upstream_authority_viability": {
                    "artifact_lineage_id": GSE200304_UPSTREAM_AUTHORITY_LINEAGE_ID,
                    "status": "COMMITTED_ACCEPTED_EXACT6",
                    "endpoint_semantics_supported": True,
                    "biological_replicate_branch_supported": True,
                    "biological_replicate_count": 6,
                    "standard_error_status": "ABSENT_NOT_DERIVED_NOT_USED",
                    "rights_scope": "PRIVATE_CANONICAL_ONLY",
                    "public_redistribution_authorized": False,
                    "changes_study_qualification": False,
                    "runtime_sync_status": "PENDING_NO_EVT_042",
                },
                "upstream_pass_gate_pack": {
                    "artifact_lineage_id": GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_LINEAGE_ID,
                    "status": "COMMITTED_ACCEPTED_EXACT6",
                    "pass_slot_ids": [1, 3, 5],
                    "gate_status_counts": {"PASS": 3},
                    "changes_study_qualification": False,
                    "runtime_sync_status": "PENDING_NO_EVT_042",
                },
                "biological_group_authority_gate": {
                    "artifact_lineage_id": GSE200304_DEC019_GROUP_LINEAGE_ID,
                    "status": "PASS",
                    "biological_group_count": 6544,
                    "changes_study_qualification": False,
                    "runtime_sync_status": "SYNCED_EVT_044",
                },
                "outcome_blind_split_leakage_gate": {
                    "artifact_lineage_id": GSE200304_DEC019_SPLIT_LINEAGE_ID,
                    "status": "PASS",
                    "connected_component_count": 1936,
                    "cross_fold_leakage_count": 0,
                    "outcome_column_read_count": 0,
                    "changes_study_qualification": False,
                    "runtime_sync_status": "SYNCED_EVT_044",
                },
                "prefrozen_power_precision_gate": {
                    "artifact_lineage_id": GSE200304_DEC019_POWER_LINEAGE_ID,
                    "status": "PASS",
                    "planning_only": True,
                    "observed_model_results_used": False,
                    "a2_final_membership_used": False,
                    "changes_study_qualification": False,
                    "runtime_sync_status": "SYNCED_EVT_044",
                },
                "dec019_post_adjudication": {
                    "artifact_lineage_id": GSE200304_DEC019_ONE_BLOCKER_ADJUDICATION_LINEAGE_ID,
                    "historical_predecessor_artifact_lineage_id": GSE200304_DEC019_UPSTREAM_PASS_ADJUDICATION_LINEAGE_ID,
                    "status": "BLOCKED",
                    "adjudication_status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
                    "input_gate_count": 8,
                    "pass_slot_ids": GSE200304_DEC019_ONE_BLOCKER_PASS_SLOT_IDS,
                    "input_status_counts": GSE200304_DEC019_ONE_BLOCKER_INPUT_STATUS_COUNTS,
                    "blocker_count": 1,
                    "blockers": GSE200304_DEC019_ONE_BLOCKER_BLOCKERS,
                    "ordinary_study_contribution": 0,
                    "a1_study_contribution": 0,
                    "true_a2_study_contribution": 0,
                    "positive_input_canonical_record_count": 6547,
                    "canonical_record_count": 0,
                    "positive_input_fact_is_not_final_canonical_materialization": True,
                    "qualified": False,
                    "independent_reproduction_established": False,
                    "canonical_materialization_allowed": False,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "power_evidence_is_planning_only": True,
                    "predecessor_runtime_event_id": "A1-EVT-043",
                    "expected_next_runtime_event_id": "A1-EVT-044",
                    "runtime_sync_status": "SYNCED_EVT_044",
                },
                "checkpoint_exposure_fail_current_protocol": {
                    "artifact_lineage_id": GSE200304_CHECKPOINT_EXPOSURE_FAIL_LINEAGE_ID,
                    "status": "FAIL_CURRENT_PROTOCOL",
                    "current_exposure_gate_status": "UNKNOWN_NOT_ASSERTED",
                    "current_public_executable_foundation_checkpoint_count": 0,
                    "audited_checkpoint_count": 0,
                    "blocker_count": 1,
                    "blockers": GSE200304_DEC019_ONE_BLOCKER_BLOCKERS,
                    "input_status_counts": GSE200304_DEC019_ONE_BLOCKER_INPUT_STATUS_COUNTS,
                    "qualified": False,
                    "canonical_record_count": 0,
                    "training_allowed": False,
                    "model_selection_allowed": False,
                    "next_phase_authorized": False,
                    "changes_qualification_gate": False,
                    "predecessor_runtime_event_id": "A1-EVT-044",
                    "expected_next_runtime_event_id": "A1-EVT-045",
                    "runtime_sync_status": "SYNCED_EVT_045",
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
            "gse200304_upstream_authority_viability_pack_is_study_qualification": False,
            "gse200304_three_upstream_pass_gates_are_complete_study_qualification": False,
            "gse200304_checkpoint_exposure_fail_current_protocol_is_exposure_gate_pass": False,
            "gse200304_six_biological_replicates_are_six_independent_studies": False,
            "gse200304_private_canonical_rights_authorize_public_redistribution": False,
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
            "gse149487_exact21_acquisition_is_study_qualification": False,
            "gse149487_exact21_acquisition_authorizes_training_model_selection_or_next_phase": False,
            "gse217518_public_authority_preflight_is_study_qualification": False,
            "gse217518_stop_authorizes_row_level_production_training_model_selection_or_next_phase": False,
            "gse232572_public_recovery_audit_is_study_qualification": False,
            "gse232572_development_reconstruction_record_count_is_canonical_record_count": False,
            "gse232572_development_v3_schema_valid_record_count_is_canonical_record_count": False,
            "gse232572_development_v3_materialization_authorizes_training_model_selection_or_next_phase": False,
            "gse232572_private_derivative_use_authorizes_public_redistribution": False,
            "gse232572_private_jsonl_is_registered_public_artifact": False,
            "dec019_creates_global_replicate_or_standard_error_relaxation": False,
            "dec019_changes_gse149487_three_biological_replicates_and_route_a_se_gate": False,
            "dec019_allows_checkpoint_specific_exposure_or_rights_waiver": False,
            "dec019_a1_authority_freeze_is_final_benchmark_membership_freeze": False,
            "dec020_authority_decision_is_gse200304_study_qualification": False,
            "dec020_scratch_checkpoint_exposure_not_applicable_is_exposure_gate_pass": False,
            "dec020_authority_decision_authorizes_v4_execution_or_adjudication": False,
            "dec020_authority_decision_authorizes_training_model_selection_or_next_phase": False,
            "dec020_prior_aggregate_design_use_may_be_called_untouched": False,
            "dec020_future_route_scoped_possibility_is_current_qualification_or_canonical_count": False,
            "dec020_deferred_v4_registration_is_active_implementation": False,
            "gse256185_public_identifier_pool_geometry_preflight_is_study_qualification": False,
            "gse256185_aggregate_pool_geometry_is_study_count_credit": False,
            "gse256185_total_body_row_count_is_canonical_record_count": False,
            "gse256185_reasoned_family_closure_axis_is_publisher_explicit": False,
            "gse256185_public_identifier_pool_geometry_preflight_establishes_true_a2": False,
            "gse256185_public_identifier_pool_geometry_preflight_authorizes_training_model_selection_or_next_phase": False,
            "dec022_authority_decision_is_gse256185_study_qualification": False,
            "dec022_strict_634_pool_universe_is_source_candidate_identity_evidence": False,
            "dec022_row_level_preflight_may_output_member_row_sequence_effect_replicate_or_split_values": False,
            "dec022_all_fail_closed_gates_passing_automatically_qualifies_gse256185": False,
            "dec022_strict_universe_and_reject_closure_receive_duplicate_scientific_credit": False,
            "dec022_authority_decision_authorizes_count_canonical_training_model_selection_or_next_phase": False,
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
                "targeted_gse200304_one_blocker_ledger_tests": {
                    "status": "PASS",
                    "scope": "CURRENT_D6_BINDING_FOUR_SETTLED_LINEAGE_NODES_EXACT14_AND_ONE_BLOCKER_COUNTS",
                },
                "gse200304_upstream_authority_exact6_descriptor_registration": "PASS",
                "gse200304_upstream_pass_gate_pack_exact6_descriptor_registration": "PASS",
                "gse200304_upstream_pass_adjudication_exact4_descriptor_registration": "PASS",
                "gse200304_biological_group_exact4_descriptor_registration": "PASS",
                "gse200304_outcome_blind_split_exact4_descriptor_registration": "PASS",
                "gse200304_prefrozen_power_exact2_descriptor_registration": "PASS",
                "gse200304_one_blocker_adjudication_exact4_descriptor_registration": "PASS",
                "gse200304_one_blocker_current_dynamic_config_binding": "PASS",
                "targeted_post_fail_acquisition_ledger_tests": {
                    "status": "PASS",
                    "scope": "TWO_AGGREGATE_EVIDENCE_RECORDS_SIX_STATIC_PRODUCER_LEAVES_EVT045_SETTLED",
                },
                "gse200304_checkpoint_exposure_fail_current_protocol_artifact_registration": "PASS",
                "gse149487_public_asset_acquisition_exact21_artifact_registration": "PASS",
                "post_fail_acquisition_six_static_leaf_registration": "PASS",
                "post_fail_acquisition_dynamic_output_cycle_policy": "PASS",
                "targeted_gse217518_registration_ledger_tests": {
                    "status": "PASS",
                    "scope": "ONE_AGGREGATE_STOP_REPORT_THREE_STATIC_PRODUCER_LEAVES_EVT046_SETTLED",
                },
                "gse217518_public_authority_preflight_artifact_registration": "PASS",
                "gse217518_public_authority_preflight_three_static_leaf_registration": "PASS",
                "gse217518_public_authority_preflight_dynamic_runtime_config_cycle_policy": "PASS",
                "targeted_gse232572_post_recovery_ledger_tests": {
                    "status": "PASS",
                    "scope": "ONE_AGGREGATE_AUDIT_REPORT_THREE_STATIC_PRODUCER_LEAVES_EVT047_SETTLED",
                },
                "gse232572_public_recovery_audit_artifact_registration": "PASS",
                "gse232572_public_recovery_audit_three_static_leaf_registration": "PASS",
                "gse232572_public_recovery_audit_dynamic_runtime_config_cycle_policy": "PASS",
                "targeted_gse232572_development_v3_materialization_ledger_tests": {
                    "status": "PASS",
                    "scope": "TWO_PUBLIC_AGGREGATE_REPORTS_THREE_STATIC_PRODUCER_LEAVES_EVT048_SETTLED",
                },
                "gse232572_development_v3_materialization_artifact_registration": "PASS",
                "gse232572_development_v3_materialization_three_static_leaf_registration": "PASS",
                "gse232572_development_v3_private_jsonl_excluded": "PASS",
                "targeted_gse232572_qualification_authority_preflight_ledger_tests": {
                    "status": "PASS",
                    "scope": "ONE_PUBLIC_AGGREGATE_STOP_REPORT_THREE_STATIC_PRODUCER_LEAVES_EVT049_SETTLED",
                },
                "gse232572_qualification_authority_preflight_artifact_registration": "PASS",
                "gse232572_qualification_authority_preflight_three_static_leaf_registration": "PASS",
                "gse232572_qualification_authority_preflight_private_row_excluded": "PASS",
                "targeted_gse200304_dec020_v4_post_adjudication_ledger_tests": {
                    "status": "PASS",
                    "scope": "EXACT7_PASS_QUALIFIED_TRUE_ORDINARY1_A1_1_A2_0_CANONICAL6547_EVT051_SETTLED_RUNTIME_SYNCED",
                },
                "gse200304_dec020_v4_static_leaf_registration": "PASS",
                "gse200304_dec020_v4_dynamic_config_binding": "PASS",
                "gse200304_dec020_v4_result": {
                    "status": "PASS",
                    "qualified": True,
                    "ordinary_study_contribution": 1,
                    "a1_study_contribution": 1,
                    "true_a2_study_contribution": 0,
                    "canonical_record_count": 6547,
                    "canonical_materialization_qualification_eligible": True,
                    "canonical_materialization_execution_authorized": False,
                    "gpu_allowed": False,
                    "foundation_route_status": "RETAINED_FAIL_CURRENT_PROTOCOL",
                    "foundation_checkpoint_evidence_status": "UNKNOWN_NOT_ASSERTED",
                    "private_payload_access_authorized": False,
                    "sealed_payload_read_count": 0,
                    "row_level_payload_read_count": 0,
                    "scientific_claim_status": "NOT_ESTABLISHED",
                },
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
                "one_blocker_ledger_independent_review": {
                    "status": "NOT_RUN",
                    "reason": "FOUR_FILE_APPEND_ONLY_LOW_TO_MEDIUM_RISK_BATCH_SELF_CHECKED_UNDER_GLOBAL_LIGHTWEIGHT_POLICY",
                },
                "post_fail_acquisition_ledger_independent_review": {
                    "status": "NOT_RUN",
                    "reason": "FOUR_FILE_APPEND_ONLY_LOW_TO_MEDIUM_RISK_BATCH_SELF_CHECKED_UNDER_GLOBAL_LIGHTWEIGHT_POLICY",
                },
                "gse217518_registration_ledger_independent_review": {
                    "status": "NOT_RUN",
                    "reason": "FOUR_FILE_APPEND_ONLY_LOW_TO_MEDIUM_RISK_BATCH_SELF_CHECKED_UNDER_GLOBAL_LIGHTWEIGHT_POLICY",
                },
                "gse232572_post_recovery_ledger_independent_review": {
                    "status": "NOT_RUN",
                    "reason": "FOUR_FILE_APPEND_ONLY_LOW_TO_MEDIUM_RISK_BATCH_SELF_CHECKED_UNDER_GLOBAL_LIGHTWEIGHT_POLICY",
                },
                "gse232572_development_v3_materialization_ledger_independent_review": {
                    "status": "NOT_RUN",
                    "reason": "FOUR_FILE_APPEND_ONLY_LOW_TO_MEDIUM_RISK_BATCH_SELF_CHECKED_UNDER_GLOBAL_LIGHTWEIGHT_POLICY",
                },
                "gse232572_qualification_authority_preflight_ledger_independent_review": {
                    "status": "NOT_RUN",
                    "reason": "FOUR_FILE_APPEND_ONLY_LOW_TO_MEDIUM_RISK_BATCH_SELF_CHECKED_UNDER_GLOBAL_LIGHTWEIGHT_POLICY",
                },
                "gse256185_post_preflight_ledger_independent_review": {
                    "status": "NOT_RUN",
                    "reason": "FOUR_FILE_APPEND_ONLY_LOW_TO_MEDIUM_RISK_BATCH_SELF_CHECKED_UNDER_GLOBAL_LIGHTWEIGHT_POLICY",
                },
                "targeted_dec020_authority_registry_tests": {
                    "status": "PASS",
                    "scope": "DEC020_EXACT10_AUTHORITY_PLUS_EXACT4_REGISTRY_EXACT3_V4_STATIC_EVT051_SETTLED_RUNTIME_SYNCED",
                },
                "dec020_active_authority_exact10_registration": "PASS",
                "dec020_v4_static_leaf_registration": "PASS",
                "dec020_runtime_sync_event_id": "A1-EVT-051",
                "targeted_dec021_authority_registry_tests": {
                    "status": "PASS",
                    "scope": "DEC021_EXACT10_AUTHORITY_GSE256185_PREFLIGHT_ONLY_EVT052_SETTLED",
                },
                "dec021_active_authority_exact10_registration": "PASS",
                "dec021_runtime_event_emitted": True,
                "targeted_dec022_authority_registry_tests": {
                    "status": "PASS",
                    "scope": "DEC022_EXACT10_AGGREGATE_ONLY_ROW_LEVEL_QUALIFICATION_PREFLIGHT_AUTHORITY_EVT054_SETTLED",
                },
                "dec022_active_authority_exact10_registration": "PASS",
                "dec022_independent_fail_closed_gate_axis_count": 17,
                "dec022_runtime_event_emitted": True,
                "dec022_authority_runtime_event_id": DEC022_AUTHORITY_RUNTIME_EVENT_ID,
                "targeted_gse256185_public_identifier_pool_geometry_preflight_ledger_tests": {
                    "status": "PASS",
                    "scope": "ONE_PUBLIC_AGGREGATE_REPORT_THREE_STATIC_PRODUCER_LEAVES_EVT053_SETTLED_RUNTIME_SYNCED",
                },
                "gse256185_public_identifier_pool_geometry_preflight_artifact_registration": "PASS",
                "gse256185_public_identifier_pool_geometry_preflight_three_static_leaf_registration": "PASS",
                "gse256185_public_identifier_pool_geometry_preflight_no_promotion_boundary": "PASS",
                "gse256185_public_identifier_pool_geometry_preflight_runtime_event_id": GSE256185_PUBLIC_GEOMETRY_RUNTIME_EVENT_ID,
                "targeted_gse256185_aggregate_row_level_qualification_preflight_ledger_tests": {
                    "status": "PASS",
                    "scope": "ONE_PUBLIC_AGGREGATE_REPORT_THREE_STATIC_PRODUCER_LEAVES_EXACT17_EVT055_SETTLED_EVIDENCE_RUNTIME_SYNCED",
                },
                "gse256185_aggregate_row_level_qualification_preflight_artifact_registration": "PASS",
                "gse256185_aggregate_row_level_qualification_preflight_three_static_leaf_registration": "PASS",
                "gse256185_aggregate_row_level_qualification_preflight_no_promotion_boundary": "PASS",
                "gse256185_aggregate_row_level_qualification_preflight_no_disclosure_boundary": "PASS",
                "gse256185_aggregate_row_level_qualification_preflight_evidence_runtime_event_id": GSE256185_ROW_PREFLIGHT_RUNTIME_EVENT_ID,
                "targeted_dec023_authority_registry_tests": {
                    "status": "PASS",
                    "scope": "DEC023_EXACT10_DUAL_AGGREGATE_ONLY_PREFLIGHT_AUTHORITY_EVT055_SETTLED_PENDING_FRESH_UNALLOCATED_RUNTIME_EVENT",
                },
                "dec023_active_authority_exact10_registration": "PASS",
                "dec023_gse261709_member_or_body_read_count_required": 0,
                "dec023_gse261709_member_or_body_output_count_required": 0,
                "dec023_gse261709_actual_header_names_output_allowed": False,
                "dec023_gse207584_independent_fail_closed_gate_axis_count": 11,
                "dec023_gse207584_aggregate_prefrozen_power_planning_calculation_allowed": True,
                "dec023_gse207584_aggregate_prefrozen_power_planning_method": "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN",
                "dec023_gse207584_aggregate_prefrozen_power_planning_confidence_interval_method": "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE",
                "dec023_gse207584_aggregate_prefrozen_power_planning_null_standard_error_formula": "1/sqrt(n-3)",
                "dec023_gse207584_aggregate_prefrozen_power_planning_alternative_standard_error_formula": "sqrt(1+rho^2/2)/sqrt(n-3)",
                "dec023_gse207584_aggregate_prefrozen_power_planning_required_effective_n_for_both_power_and_ci_width": 156,
                "dec023_gse207584_formal_qualification_power_gate_execution_allowed": False,
                "dec023_runtime_event_emitted": True,
                "dec023_current_runtime_event_id": DEC023_CURRENT_RUNTIME_EVENT_ID,
                "targeted_dec023_dual_preflight_evidence_ledger_tests": {
                    "status": "PASS",
                    "scope": "TWO_FINAL_AGGREGATE_STOP_REPORTS_SIX_STATIC_PRODUCER_LEAVES_EVT057_SETTLED_EVIDENCE_RUNTIME_SYNCED",
                },
                "dec023_dual_preflight_evidence_integration_id": DEC023_DUAL_PREFLIGHT_EVIDENCE_INTEGRATION_ID,
                "dec023_dual_preflight_registered_lineage_ids_exactly": [
                    GSE261709_PREFLIGHT_LINEAGE_ID,
                    GSE207584_PREFLIGHT_LINEAGE_ID,
                ],
                "dec023_dual_preflight_six_static_leaf_registration": "PASS",
                "dec023_dual_preflight_two_dynamic_report_lineage_registration": "PASS",
                "dec023_dual_preflight_no_promotion_boundary": "PASS",
                "dec023_dual_preflight_no_member_or_row_disclosure_boundary": "PASS",
                "dec023_dual_preflight_evidence_runtime_event_emitted": True,
                "dec023_dual_preflight_expected_next_runtime_event_id": DEC023_CURRENT_RUNTIME_EVENT_ID,
                "targeted_dec024_replacement_preflight_authority_tests": {
                    "status": "PASS",
                    "scope": "THREE_ORDINARY_PUBLIC_AGGREGATE_ONLY_PREFLIGHT_AUTHORITIES_EVT058_SETTLED_RUNTIME_SYNCED_NO_PROMOTION",
                },
                "dec024_gse261709_processed_only_no_raw_archive_boundary": "PASS",
                "dec024_gse269595_mutually_exclusive_role_adjudication_boundary": "PASS",
                "dec024_emtab10902_source_group_n_not_row_n_boundary": "PASS",
                "dec024_unknown_historical_exposure_remains_blocking": "PASS",
                "dec024_original_scientific_gate_3_2_1_unchanged": "PASS",
                "dec024_training_gpu_model_selection_a7_next_phase_all_unchanged_false": "PASS",
                "targeted_dec027_bounded_rescue_sprint_authority_tests": {
                    "status": "PASS",
                    "scope": "SIX_ORDERED_AGGREGATE_ONLY_RESCUE_PREFLIGHT_AUTHORITIES_EVT058_SETTLED_PENDING_FRESH_RUNTIME_NO_PROMOTION",
                },
                "dec027_active_authority_exact12_registration": "PASS",
                "dec027_full_route_a_3_2_1_target_retained": "PASS",
                "dec027_ordered_six_route_and_terminal_report_boundary": "PASS",
                "dec027_conditional_single_study_successor_stop_rule_frozen": "PASS",
                "dec027_training_gpu_model_selection_a7_next_phase_all_unchanged_false": "PASS",
                "targeted_dec027_six_rescue_evidence_ledger_tests": {
                    "status": "PASS",
                    "scope": "SIX_TERMINAL_AGGREGATE_REPORTS_EIGHTEEN_STATIC_PRODUCER_LEAVES_EVT059_SETTLED_PENDING_UNALLOCATED_EVT060_NO_PROMOTION",
                },
                "dec027_six_rescue_evidence_integration_id": DEC027_SIX_RESCUE_EVIDENCE_INTEGRATION_ID,
                "dec027_six_rescue_registered_lineage_ids_exactly": DEC027_SIX_RESCUE_LINEAGE_IDS,
                "dec027_six_rescue_eighteen_static_leaf_registration": "PASS",
                "dec027_six_rescue_six_dynamic_report_lineage_registration": "PASS",
                "dec027_six_rescue_no_promotion_boundary": "PASS",
                "dec027_six_rescue_ledger_runtime_event_id": DEC027_CURRENT_RUNTIME_EVENT_ID,
                "dec027_six_rescue_ledger_pending_successor_runtime_event_label": DEC027_PENDING_SUCCESSOR_RUNTIME_EVENT_LABEL,
                "dec027_six_rescue_successor_runtime_event_preallocated": False,
                "dec027_six_rescue_stop_rule_trigger_condition_met": False,
                "targeted_dec027_evt060_current_projection_tests": {
                    "status": "PASS",
                    "scope": "EVT060_EXACT_CAS_CURRENT_PROJECTION_SETTLED_NO_EVT061_NO_PROMOTION",
                },
                "dec027_evt060_projection_lineage_id": DEC027_EVT060_PROJECTION_LINEAGE_ID,
                "dec027_evt060_runtime_binding_commit": "6a7ebeae2a2ced43e29c9458601e43a19496416c",
                "dec027_evt060_current_runtime_event_id": DEC027_EVT060_CURRENT_RUNTIME_EVENT_ID,
                "dec027_evt060_expected_next_runtime_event_id": DEC027_PENDING_RUNTIME_EVENT_ID,
                "dec027_evt060_successor_runtime_event_preallocated": False,
                "dec027_evt060_runtime_cas_registration": "PASS",
                "dec027_evt060_counts_claim_and_locks_unchanged": "PASS",
                "dec027_evt060_stop_rule_ready": True,
                "dec027_evt060_stop_rule_evaluated_by_this_event": False,
                "dec027_evt060_conditional_successor_activated": False,
            },
            path,
            issues,
            "A1_INTERIM_VERIFICATION",
        )

    _expect(interim, "initial_generated_at", "2026-08-10T06:30:58+08:00", path, issues, "A1_INTERIM_TIME")
    _expect(interim, "updated_for_decision_id", "V3-DEC-028", path, issues, "A1_INTERIM_TIME")
    _expect(interim, "latest_authority_update_id", "V3-DEC-028", path, issues, "A1_INTERIM_TIME")
    _expect(
        interim,
        "latest_evidence_update_id",
        "DEC028_STATIC_AUTHORITY_PENDING_RUNTIME_SYNC",
        path,
        issues,
        "A1_INTERIM_TIME",
    )
    generated = interim.get("generated_at")
    updated = interim.get("updated_at")
    if (
        generated != DEC028_AUTHORITY_LEDGER_AT
        or updated != DEC028_AUTHORITY_LEDGER_AT
    ):
        _issue(
            issues,
            "A1_INTERIM_TIME",
            path,
            "generated_at and updated_at must identify the exact DEC028 static-authority ledger timestamp",
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
        gse149487_public_asset_acquisition_dt = datetime.fromisoformat(
            "2026-08-12T18:41:52+08:00"
        )
        gse217518_public_authority_preflight_dt = datetime.fromisoformat(
            "2026-08-12T20:11:39+08:00"
        )
        gse232572_public_recovery_audit_dt = datetime.fromisoformat(
            "2026-08-12T21:57:45+08:00"
        )
        gse232572_development_v3_materialization_dt = datetime.fromisoformat(
            "2026-08-12T23:42:43+08:00"
        )
        gse232572_qualification_authority_preflight_dt = datetime.fromisoformat(
            "2026-08-13T01:01:33+08:00"
        )
        gse256185_public_geometry_preflight_dt = datetime.fromisoformat(
            "2026-08-13T21:06:51+08:00"
        )
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
            or updated_dt < gse149487_public_asset_acquisition_dt
            or updated_dt < gse217518_public_authority_preflight_dt
            or updated_dt < gse232572_public_recovery_audit_dt
            or updated_dt < gse232572_development_v3_materialization_dt
            or updated_dt < gse232572_qualification_authority_preflight_dt
            or updated_dt < gse256185_public_geometry_preflight_dt
        ):
            _issue(
                issues,
                "A1_INTERIM_TIME",
                path,
                "updated_at must follow all preserved evidence events, DEC-019 owner authorization, successor integration, GSE200304 post-adjudication artifacts, GSE149487 exact21 acquisition, the GSE217518 public-authority preflight, the GSE232572 public-recovery audit, the GSE232572 development V3 materialization, the GSE232572 qualification-authority preflight, and the GSE256185 public geometry preflight",
            )
    return issues


def _expected_a6_cpu_exact_interim() -> dict[str, Any]:
    denominator = {
        "denominator_type": "FROZEN_SYNTHETIC_ENUMERATION_CASES",
        "fixed_case_count": 6,
        "case_ids": ["L2_B0", "L2_B1", "L2_B2", "L3_B0", "L3_B1", "L3_B2"],
        "total_state_count": 91,
        "total_complete_path_count": 108,
    }
    return {
        "schema_version": "1.0.0",
        "contract_id": CONTRACT_ID,
        "contract_version": VERSION,
        "record_id": "ROUTE_A_V3_A6_CPU_EXACT_ABSORBING_DAG_V1",
        "record_type": "A6_CPU_EXACT_PARTIAL_EVIDENCE_INTERIM",
        "phase_id": "A6",
        "record_status": "INTERIM_IN_PROGRESS_NOT_PHASE_COMPLETE",
        "authority": {
            "contract_path": GOAL_PATH,
            "contract_sha256": SOURCE_CONTRACT_SHA256,
            "active_config_path": CONFIG_PATH,
            "active_config_sha256": CURRENT_ACTIVE_CONFIG_SHA256,
            "task_registry_path": REGISTRY_PATHS["task"],
            "task_registry_sha256": DEC028_ACTIVE_AUTHORITY_LEAF_SHA256[
                REGISTRY_PATHS["task"]
            ],
            "claim_evidence_matrix_path": REGISTRY_PATHS["claim"],
            "claim_evidence_matrix_sha256": DEC028_ACTIVE_AUTHORITY_LEAF_SHA256[
                REGISTRY_PATHS["claim"]
            ],
            "branch": "routea-v3-a1-20260810",
            "worktree": "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810",
        },
        "artifact_lineage": {
            "producer_commits": {
                "frozen_base_commit": A6_FROZEN_BASE_COMMIT,
                "implementation_commit": A6_IMPLEMENTATION_COMMIT,
                "binding_commit": A6_BINDING_COMMIT,
            },
            "static_leaves": {
                "protocol": {
                    "path": A6_PROTOCOL_CONFIG_PATH,
                    "sha256": A6_STATIC_PRODUCER_LEAF_SHA256[
                        A6_PROTOCOL_CONFIG_PATH
                    ],
                },
                "producer": {
                    "path": A6_PRODUCER_PATH,
                    "sha256": A6_STATIC_PRODUCER_LEAF_SHA256[A6_PRODUCER_PATH],
                },
                "focused_test": {
                    "path": A6_FOCUSED_TEST_PATH,
                    "sha256": A6_STATIC_PRODUCER_LEAF_SHA256[
                        A6_FOCUSED_TEST_PATH
                    ],
                },
            },
            "dynamic_report": {
                "evidence_id": A6_REPORT_EVIDENCE_ID,
                "path": A6_REPORT_PATH,
                "bytes": A6_REPORT_BYTES,
                "sha256": A6_REPORT_SHA256,
                "recorded_at": A6_REPORT_RECORDED_AT,
                "registered_in_static_manifest": False,
            },
            "additional_partial_evidence": {
                "evidence_id": A6_GILLESPIE_REPORT_EVIDENCE_ID,
                "producer_commits": {
                    "frozen_base_commit": A6_GILLESPIE_FROZEN_BASE_COMMIT,
                    "initial_implementation_commit": (
                        A6_GILLESPIE_INITIAL_IMPLEMENTATION_COMMIT
                    ),
                    "initial_binding_commit": A6_GILLESPIE_INITIAL_BINDING_COMMIT,
                    "implementation_commit": A6_GILLESPIE_IMPLEMENTATION_COMMIT,
                    "binding_commit": A6_GILLESPIE_BINDING_COMMIT,
                },
                "static_leaves": {
                    "protocol": {
                        "path": A6_GILLESPIE_CONFIG_PATH,
                        "sha256": A6_GILLESPIE_STATIC_PRODUCER_LEAF_SHA256[
                            A6_GILLESPIE_CONFIG_PATH
                        ],
                    },
                    "producer": {
                        "path": A6_GILLESPIE_PRODUCER_PATH,
                        "sha256": A6_GILLESPIE_STATIC_PRODUCER_LEAF_SHA256[
                            A6_GILLESPIE_PRODUCER_PATH
                        ],
                    },
                    "focused_test": {
                        "path": A6_GILLESPIE_FOCUSED_TEST_PATH,
                        "sha256": A6_GILLESPIE_STATIC_PRODUCER_LEAF_SHA256[
                            A6_GILLESPIE_FOCUSED_TEST_PATH
                        ],
                    },
                },
                "dynamic_report": {
                    "path": A6_GILLESPIE_REPORT_PATH,
                    "bytes": A6_GILLESPIE_REPORT_BYTES,
                    "sha256": A6_GILLESPIE_REPORT_SHA256,
                    "recorded_at": A6_GILLESPIE_REPORT_RECORDED_AT,
                    "registered_in_static_manifest": False,
                },
            },
        },
        "run_state": {
            "protocol_id": A6_PROTOCOL_ID,
            "run_scope": "DEVELOPMENT_ONLY_CPU_EXACT_ABSORPTION_FIXTURE",
            "run_status": "PASS",
            "public_aggregate_only": True,
            "runtime_output_count": 3,
            "runtime_output_names": [
                "A6_CPU_EXACT_ABSORBING_DAG_REPORT.json",
                "RUN_MANIFEST.json",
                "EVENT_LOG.jsonl",
            ],
            "runtime_auxiliary_output_hashes_registered": False,
            "additional_partial_run": {
                "protocol_id": A6_GILLESPIE_PROTOCOL_ID,
                "run_scope": "SYNTHETIC_NONLEARNED_CPU_GILLESPIE_BASE_RECOVERY",
                "run_status": "PASS",
                "public_aggregate_only": True,
                "runtime_output_count": 3,
                "runtime_output_names": [
                    "A6_CPU_LEGAL_CTMC_PARTIAL_REPORT.json",
                    "RUN_MANIFEST.json",
                    "EVENT_LOG.jsonl",
                ],
                "runtime_auxiliary_output_hashes_registered": False,
            },
        },
        "phase_state": {
            "evidence_status": "IN_PROGRESS",
            "phase_complete": False,
        },
        "task_states": {
            "EXACT_GUIDANCE_TOY_GRAPH": {
                "evidence_status": "PASS",
                "result": "DEVELOPMENT_CPU_EXACT_FIXTURE_PASS",
                "scope": "SYNTHETIC_TIME_HOMOGENEOUS_CPU_EXACT",
            },
            "FLOW_BASE_LEGAL_CTMC": {
                "evidence_status": "IN_PROGRESS",
                "result": (
                    "DEVELOPMENT_CPU_NONLEARNED_GILLESPIE_REPLAY_PARTIAL_PASS"
                ),
                "scope": "SYNTHETIC_NONLEARNED_CPU_GILLESPIE_BASE_RECOVERY",
                "formal_task_pass_asserted": False,
            },
        },
        "claim_state": {
            "claim_id": "L3_LEGAL_POTENTIAL_CONSISTENT_XEDITFLOW",
            "evidence_status": "IN_PROGRESS",
            "claim_status": "NOT_ESTABLISHED",
        },
        "time_scope": {
            "clock_semantics": "CONTINUOUS_ALGORITHMIC_TIME",
            "rate_time_dependence": "NONE",
            "terminal_tilt_time_dependence": "NONE",
            "holding_time_law": "EXPONENTIAL_WITH_CURRENT_TOTAL_EXIT_RATE",
            "general_time_inhomogeneous_exactness": "NOT_RUN",
            "dp_time_key_rule": (
                "DP_KEYS_MAY_QUOTIENT_ALGORITHMIC_TIME_ONLY_BECAUSE_"
                "RATES_AND_TERMINAL_TILT_ARE_TIME_INVARIANT"
            ),
        },
        "denominator": denominator,
        "aggregate_maximum_metrics": {
            "guided_terminal_tv_vs_tilted_base": 5.139118297581291e-17,
            "guided_vs_base_total_exit_rate_relative_error": 2.0529242248115864e-16,
            "path_product_relative_error": 2.2266202935521747e-16,
            "true_per_rate_relative_error": 0.0,
        },
        "flow_base_partial_denominator": {
            "denominator_type": "FROZEN_SYNTHETIC_GILLESPIE_TRAJECTORIES",
            "kernel_case_id": "L2_B2",
            "trajectory_count": 20000,
            "replay_trajectory_count": 256,
            "random_seed": 2026081301,
        },
        "flow_base_partial_aggregate_metrics": {
            "terminal_distribution_tv": 0.011045317469385082,
            "initial_holding_time_mean_relative_error": 0.0017202797017030437,
            "replay_mismatch_count": 0,
            "legality_violation_count": 0,
            "source_reconstruction_violation_count": 0,
            "budget_violation_count": 0,
            "reedit_or_revert_violation_count": 0,
            "algorithmic_time_monotonicity_violation_count": 0,
            "alias_aggregation_violation_count": 0,
            "numerical_failure_count": 0,
        },
        "flow_base_partial_compute_ledger": {
            "device": "CPU",
            "learned_parameter_count": 0,
            "parameter_update_count": 0,
            "trajectory_count": 20000,
            "total_jump_count": 38089,
            "wall_clock_seconds": 24.46186525002122,
        },
        "flow_base_partial_terminal_cause_counts": {
            "BUDGET_EXHAUSTED": 14473,
            "EXPLICIT_STOP": 5527,
            "NO_LEGAL_ACTION": 0,
            "NUMERICAL_FAILURE": 0,
        },
        "required_evidence_coverage": {
            "exact_guidance_toy_graph": "PASS",
            "flow_base_legal_ctmc": "IN_PROGRESS",
            "general_time_inhomogeneous_exactness": "NOT_RUN",
            "learned_potential_approximation_error": "NOT_RUN",
            "ordinary_data_evidence": "NOT_RUN",
            "l3_evidence_status": "IN_PROGRESS",
        },
        "registry_boundary": {
            "task_registry_is_definition_only": True,
            "task_registry_a6_phase_evidence_status_remains": "NOT_RUN",
            "task_registry_exact_guidance_toy_graph_evidence_status_remains": "NOT_RUN",
            "task_registry_flow_base_legal_ctmc_evidence_status_remains": "NOT_RUN",
            "task_registry_changed_by_this_registration": False,
            "evidence_semantics_registered_only_in_a6_interim_and_claim_cell": True,
            "a1_interim_authority_pointer_rebind_only": True,
            "a1_runtime_event_emitted": False,
            "a1_runtime_science_or_counts_changed": False,
        },
        "boundaries": {
            "a6_pass_asserted": False,
            "formal_flow_base_task_pass_asserted": False,
            "l3_claim_established": False,
            "a7_evidence_status": "NOT_RUN",
            "a7_unlock": False,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "ordinary_data_read": False,
            "private_payload_access_allowed": False,
            "sealed_contact_allowed": False,
        },
        "generated_at": A6_REGISTRATION_LEDGER_AT,
        "updated_at": A6_REGISTRATION_LEDGER_AT,
    }


def validate_a6_cpu_exact_registration(
    repo_root: Path,
    registries: Mapping[str, Mapping[str, Any]],
) -> list[Issue]:
    """Validate the accumulated exact and Gillespie A6 partial evidence."""

    issues: list[Issue] = []
    try:
        interim = _load_yaml(repo_root, A6_INTERIM_PATH)
        actual_interim_sha256 = sha256_bytes(_read_bytes(repo_root, A6_INTERIM_PATH))
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        _issue(issues, "A6_INTERIM_LOAD", A6_INTERIM_PATH, str(exc))
        return issues
    if actual_interim_sha256 != EXPECTED_A6_INTERIM_SHA256:
        _issue(
            issues,
            "A6_INTERIM_CANONICAL_HASH",
            A6_INTERIM_PATH,
            f"active interim hash {actual_interim_sha256} must remain {EXPECTED_A6_INTERIM_SHA256}",
        )
    _expect_closed_mapping(
        interim,
        _expected_a6_cpu_exact_interim(),
        A6_INTERIM_PATH,
        issues,
        "A6_INTERIM_SEMANTICS",
    )

    registered_static_leaves = {
        **A6_STATIC_PRODUCER_LEAF_SHA256,
        **A6_GILLESPIE_STATIC_PRODUCER_LEAF_SHA256,
    }
    for relative, expected_sha256 in registered_static_leaves.items():
        try:
            actual_sha256 = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "A6_STATIC_LEAF_UNREADABLE", relative, str(exc))
            continue
        if actual_sha256 != expected_sha256:
            _issue(
                issues,
                "A6_STATIC_LEAF_DRIFT",
                relative,
                f"current bytes hash {actual_sha256} must remain {expected_sha256}",
            )

    denominator = _expected_a6_cpu_exact_interim()["denominator"]
    expected_exact_evidence_cell = {
        "evidence_id": A6_REPORT_EVIDENCE_ID,
        "artifact_path": A6_REPORT_PATH,
        "artifact_hash": A6_REPORT_SHA256,
        "artifact_bytes": A6_REPORT_BYTES,
        "recorded_at": A6_REPORT_RECORDED_AT,
        "analysis_intent": (
            "PROSPECTIVE_DEVELOPMENT_CPU_EXACT_FIXTURE_FOR_PARTIAL_A6_EVIDENCE"
        ),
        "denominator": denominator,
        "evidence_status": "PASS",
        "task_id": "EXACT_GUIDANCE_TOY_GRAPH",
        "result": "DEVELOPMENT_CPU_EXACT_FIXTURE_PASS",
        "scope": "SYNTHETIC_TIME_HOMOGENEOUS_CPU_EXACT",
        "producer_lineage": {
            "frozen_base_commit": A6_FROZEN_BASE_COMMIT,
            "implementation_commit": A6_IMPLEMENTATION_COMMIT,
            "binding_commit": A6_BINDING_COMMIT,
        },
        "establishes_a6_phase_pass": False,
        "establishes_l3_claim": False,
        "unlocks_a7": False,
    }
    expected_gillespie_evidence_cell = {
        "evidence_id": A6_GILLESPIE_REPORT_EVIDENCE_ID,
        "artifact_path": A6_GILLESPIE_REPORT_PATH,
        "artifact_hash": A6_GILLESPIE_REPORT_SHA256,
        "artifact_bytes": A6_GILLESPIE_REPORT_BYTES,
        "recorded_at": A6_GILLESPIE_REPORT_RECORDED_AT,
        "analysis_intent": (
            "PROSPECTIVE_DEVELOPMENT_CPU_NONLEARNED_GILLESPIE_REPLAY_"
            "FOR_PARTIAL_A6_EVIDENCE"
        ),
        "denominator": {
            "denominator_type": "FROZEN_SYNTHETIC_GILLESPIE_TRAJECTORIES",
            "kernel_case_id": "L2_B2",
            "trajectory_count": 20000,
            "replay_trajectory_count": 256,
            "random_seed": 2026081301,
        },
        "evidence_status": "PASS",
        "task_id": "FLOW_BASE_LEGAL_CTMC",
        "result": "DEVELOPMENT_CPU_NONLEARNED_GILLESPIE_REPLAY_PARTIAL_PASS",
        "scope": "SYNTHETIC_NONLEARNED_CPU_GILLESPIE_BASE_RECOVERY",
        "producer_lineage": {
            "frozen_base_commit": A6_GILLESPIE_FROZEN_BASE_COMMIT,
            "implementation_commit": A6_GILLESPIE_IMPLEMENTATION_COMMIT,
            "binding_commit": A6_GILLESPIE_BINDING_COMMIT,
        },
        "establishes_formal_task_pass": False,
        "establishes_a6_phase_pass": False,
        "establishes_l3_claim": False,
        "unlocks_a7": False,
    }
    claim_registry = registries.get("claim", {})
    l3_claim = _mapping_entry(
        claim_registry.get("claims"),
        "claim_id",
        "L3_LEGAL_POTENTIAL_CONSISTENT_XEDITFLOW",
    )
    if not isinstance(l3_claim, Mapping):
        _issue(
            issues,
            "A6_CLAIM_CELL",
            REGISTRY_PATHS["claim"],
            "L3 legal potential-consistent claim cell is required",
        )
    else:
        for key, expected in {
            "evidence_status": "IN_PROGRESS",
            "claim_status": "NOT_ESTABLISHED",
            "required_phase_ids": ["A6"],
            "required_task_ids": [
                "FLOW_BASE_LEGAL_CTMC",
                "EXACT_GUIDANCE_TOY_GRAPH",
            ],
            "evidence_cells": [
                expected_exact_evidence_cell,
                expected_gillespie_evidence_cell,
            ],
        }.items():
            _expect(
                l3_claim,
                key,
                expected,
                REGISTRY_PATHS["claim"],
                issues,
                "A6_CLAIM_CELL",
            )

    task_registry = registries.get("task", {})
    a6_phase = _mapping_entry(task_registry.get("phase_tasks"), "phase_id", "A6")
    exact_task = _mapping_entry(
        task_registry.get("tasks"),
        "task_id",
        "EXACT_GUIDANCE_TOY_GRAPH",
    )
    base_task = _mapping_entry(
        task_registry.get("tasks"),
        "task_id",
        "FLOW_BASE_LEGAL_CTMC",
    )
    for entry, label, expected in (
        (a6_phase, "A6 phase definition", {"evidence_status": "NOT_RUN"}),
        (
            exact_task,
            "EXACT_GUIDANCE_TOY_GRAPH definition",
            {"evidence_status": "NOT_RUN", "claim_status": "NOT_ESTABLISHED"},
        ),
        (
            base_task,
            "FLOW_BASE_LEGAL_CTMC definition",
            {"evidence_status": "NOT_RUN", "claim_status": "NOT_ESTABLISHED"},
        ),
    ):
        if not isinstance(entry, Mapping):
            _issue(
                issues,
                "A6_TASK_REGISTRY_BOUNDARY",
                REGISTRY_PATHS["task"],
                f"{label} is required",
            )
            continue
        for key, value in expected.items():
            _expect(
                entry,
                key,
                value,
                REGISTRY_PATHS["task"],
                issues,
                "A6_TASK_REGISTRY_BOUNDARY",
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
            expected_evidence_status = (
                "IN_PROGRESS"
                if claim.get("claim_id")
                == "L3_LEGAL_POTENTIAL_CONSISTENT_XEDITFLOW"
                else "NOT_RUN"
            )
            if (
                claim.get("claim_status") != "NOT_ESTABLISHED"
                or claim.get("evidence_status") != expected_evidence_status
            ):
                _issue(
                    issues,
                    "A0_CLAIM_PREMATURE",
                    REGISTRY_PATHS["claim"],
                    f"claim {claim.get('claim_id')!r} must remain "
                    f"{expected_evidence_status}/NOT_ESTABLISHED",
                )

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


def validate_gse200304_dec020_v4_post_adjudication_registration(
    repo_root: Path,
) -> list[Issue]:
    """Bind the DEC-020 V4 implementation and its settled aggregate outcome."""

    issues: list[Issue] = []
    for relative, expected_sha256 in GSE200304_DEC020_V4_STATIC_LEAF_SHA256.items():
        try:
            actual = sha256_bytes(_read_bytes(repo_root, relative))
        except (FileNotFoundError, ValueError) as exc:
            _issue(issues, "GSE200304_DEC020_V4_STATIC_LEAF", relative, str(exc))
        else:
            if actual != expected_sha256:
                _issue(issues, "GSE200304_DEC020_V4_STATIC_LEAF", relative, f"current bytes hash {actual} must remain {expected_sha256}")

    try:
        config = _load_json(repo_root, GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH)
        full_sha = sha256_bytes(_read_bytes(repo_root, GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "GSE200304_DEC020_V4_CONFIG", GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH, str(exc))
        return issues
    if config.get("dataset_id") != "GSE200304" or config.get("decision_id") != "V3-DEC-020":
        _issue(issues, "GSE200304_DEC020_V4_CONFIG", GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH, "dataset_id and decision_id must remain GSE200304/V3-DEC-020")
    binding = config.get("implementation_binding")
    expected_binding = {
        "status": "BOUND",
        "implementation_commit": GSE200304_DEC020_V4_IMPLEMENTATION_COMMIT,
        "implementation_script_path": GSE200304_DEC020_V4_SCRIPT_PATH,
        "implementation_script_sha256": GSE200304_DEC020_V4_STATIC_LEAF_SHA256[GSE200304_DEC020_V4_SCRIPT_PATH],
        "implementation_test_path": GSE200304_DEC020_V4_TEST_PATH,
        "implementation_test_sha256": GSE200304_DEC020_V4_STATIC_LEAF_SHA256[GSE200304_DEC020_V4_TEST_PATH],
        "config_core_sha256": GSE200304_DEC020_V4_CONFIG_CORE_SHA256,
    }
    if not isinstance(binding, Mapping):
        _issue(issues, "GSE200304_DEC020_V4_BINDING", GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH, "implementation_binding must be a mapping")
    else:
        for key, expected in expected_binding.items():
            _expect(binding, key, expected, GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH, issues, "GSE200304_DEC020_V4_BINDING")
    descriptors = config.get("evidence_descriptor_bindings")
    if not isinstance(descriptors, Mapping):
        _issue(issues, "GSE200304_DEC020_V4_DESCRIPTORS", GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH, "evidence_descriptor_bindings must be a mapping")
    else:
        _expect(descriptors, "status", "BOUND", GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH, issues, "GSE200304_DEC020_V4_DESCRIPTORS")
        _expect(descriptors, "descriptor_set_sha256", GSE200304_DEC020_V4_DESCRIPTOR_SET_SHA256, GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH, issues, "GSE200304_DEC020_V4_DESCRIPTORS")
    # The full config is intentionally not a self-referential manifest leaf check.
    if not full_sha or len(full_sha) != 64:
        _issue(issues, "GSE200304_DEC020_V4_CONFIG", GSE200304_DEC020_V4_DYNAMIC_CONFIG_PATH, "config full SHA-256 is unavailable")
    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "GSE200304_DEC020_V4_MANIFEST", REGISTRY_MANIFEST_PATH, str(exc))
    else:
        entries = manifest.get("files")
        paths = {entry.get("path") for entry in entries if isinstance(entry, Mapping)} if isinstance(entries, list) else set()
        if set(GSE200304_DEC020_V4_STATIC_LEAF_SHA256) - paths:
            _issue(issues, "GSE200304_DEC020_V4_MANIFEST", REGISTRY_MANIFEST_PATH, "V4 static leaves must be registered in the manifest")
        if GSE200304_DEC020_V4_LINEAGE_ID in paths:
            _issue(issues, "GSE200304_DEC020_V4_MANIFEST", REGISTRY_MANIFEST_PATH, "runtime lineage IDs are not static manifest paths")
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
    issues.extend(validate_post_fail_acquisition_registration(repo_root))
    issues.extend(
        validate_gse217518_public_authority_preflight_registration(repo_root)
    )
    issues.extend(validate_gse232572_public_recovery_audit_registration(repo_root))
    issues.extend(
        validate_gse232572_development_v3_materialization_registration(repo_root)
    )
    issues.extend(
        validate_gse232572_qualification_authority_preflight_registration(
            repo_root
        )
    )
    issues.extend(validate_gse200304_dec020_v4_post_adjudication_registration(repo_root))
    issues.extend(validate_gse256185_public_geometry_registration(repo_root))
    issues.extend(validate_gse256185_row_preflight_registration(repo_root))
    issues.extend(validate_dec023_dual_preflight_evidence_registration(repo_root))
    issues.extend(validate_dec027_six_rescue_evidence_registration(repo_root))
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
    issues.extend(validate_dec020_authority(repo_root, config, registries))
    issues.extend(validate_dec021_authority(repo_root, config, registries))
    issues.extend(validate_dec022_authority(repo_root, config, registries))
    issues.extend(validate_dec023_authority(repo_root, config, registries))
    issues.extend(validate_dec024_authority(repo_root, config, registries))
    issues.extend(validate_dec027_authority(repo_root, config, registries))
    issues.extend(validate_dec028_authority(repo_root, config, registries))
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
    issues.extend(validate_a6_cpu_exact_registration(repo_root, registries))
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
