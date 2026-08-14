#!/usr/bin/env python3
"""Bound aggregate-only corrected GSE269595 role-adjudication successor.

The corrected route keeps A1 and true-A2 mutually exclusive, excludes the
single two-candidate family only from the dense universe, preserves the
publisher's two-biological-replicate fact, and audits valid standard error
separately.  Non-finite endpoints are never treated as zero.

Production has one fixed built-in reader for the two publisher-grounded public
assets.  It remains fail-closed until DEC027 authority, the full runtime
history, the ordered GSE217518/ENCSR854RUF/GSE232572/GSE113849 predecessors,
and this exact3 implementation are bound.  The complete direct-parent Git
chain is audited before asset or output I/O.  One aggregate report is
published atomically without replacement; this route cannot qualify or credit
the study.
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
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    STAGING_ROOT
    / "configs/route_a_v3_gse269595_corrected_role_adjudication_successor_candidate_v1.json"
)
REPORT_PATH = (
    STAGING_ROOT
    / "reports/GSE269595_CORRECTED_ROLE_ADJUDICATION_SUCCESSOR_AGGREGATE_RECOMPUTE_V1.json"
)
REPORT_FILENAME = REPORT_PATH.name

SCHEMA_VERSION = (
    "route_a_v3_gse269595_corrected_role_adjudication_successor_candidate.v1"
)
PROTOCOL_ID = "GSE269595_CORRECTED_ROLE_ADJUDICATION_SUCCESSOR_CANDIDATE_V1"
REPORT_SCHEMA_VERSION = (
    "route_a_v3_gse269595_corrected_role_adjudication_aggregate_recompute.v1"
)
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
PASS = "PASS"
BLOCKED = "BLOCKED"
FAIL = "FAIL"

PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_BRANCH = "routea-v3-a1-20260810"
PRODUCTION_UPSTREAM = f"origin/{PRODUCTION_BRANCH}"
AUTHORITY_COMMIT = "3e0ad158a0b45b2f26ed82da3afe60667c712cd6"
AUTHORITY_PARENT = "b1ca33d852bad111ff31b4f60493d8c43c63d1a3"
RUNTIME_I1_COMMIT = "de40c58ab81fc06196be3bb9ffb5aa35d39c9d03"
RUNTIME_I2_COMMIT = "5d66e8dc83eb9966f7698ac0fc677f1b06af8ea6"
RUNTIME_B_COMMIT = "e60956cf59cbddc0406c5d116fb9714906db36e1"
BASE_EVENT = "A1-EVT-059"

AUTHORITY_EXACT12: tuple[str, ...] = (
    "configs/route_a_v3.yaml",
    "configs/route_a_v3_a1_qualification.json",
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec027.yaml",
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
    "configs/route_a_v3.yaml": "c5ec7d236443b506c09fd3f09e149ce5d082daff618887989af6e59472727a27",
    "configs/route_a_v3_a1_qualification.json": "261339c38f4b8bbd48bf8f63f6a588be57af9f6229119e84bf661d7ee8f855db",
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec027.yaml": "2a27c296539e8e665873363778d91cc223f56f933a815e9509b10a7267f6b5c4",
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml": "1c4b6e29c09eb24798207138047b68909d7bda8bacc3a2eab8e17a7ca789b44b",
    "docs/execution/route_a_v3_a1_interim.yaml": "fb50929ae2dfa0bdd1c50b003fc43c0e13b89baade203e8384c8b6ea3eba7b1e",
    "docs/execution/route_a_v3_a6_interim.yaml": "1d44bcfe8669a55dc42f619ed43178f0637e20a4297ca996ecab3f7165612769",
    "docs/execution/route_a_v3_data_role_registry.yaml": "80217a8114286f84960237819ac5b2d5828afbc23af118576541cc7cee64ae4e",
    "docs/execution/route_a_v3_decision_log.yaml": "e0dc2a7fb186c5c8d00c1c5604602b1b2f87b26241191d4da55a405a02387e05",
    "docs/execution/route_a_v3_registry_manifest.json": "73a39a566aa0310a80cc83f4eb17ddb95cabc87e5b070ef6484c05178ba32b75",
    "docs/execution/route_a_v3_task_registry.yaml": "a64d0b8bb5eb466b06daa46ed109bd19901ee775910bc5cc9221c39ead63a4bc",
    "scripts/route_a_v3/validate_a0_bundle.py": "81d1a8dc49375f53a1edd5c3f41625e734eacc31a4c328c8137340a041d77e65",
    "tests/route_a_v3/test_a0_integrity_guards.py": "106f847e957a40fffec4c1b57f8f572325ef96e8a3dc5729db68499e320f380b",
}
RUNTIME_CONFIG_PATH = "configs/route_a_v3_dec027_authority_runtime_sync_v1.json"
RUNTIME_SCRIPT_PATH = "scripts/route_a_v3/dec027_authority_runtime_sync.py"
RUNTIME_TEST_PATH = "tests/route_a_v3/test_dec027_authority_runtime_sync.py"
RUNTIME_EXACT3 = (RUNTIME_CONFIG_PATH, RUNTIME_SCRIPT_PATH, RUNTIME_TEST_PATH)
RUNTIME_I1_BLOBS = {
    RUNTIME_CONFIG_PATH: "cbb0c8c4fb2b47a1e1c1dd629d46c149b1c35299e8127dfd431a22666e40dfd5",
    RUNTIME_SCRIPT_PATH: "ae13a64ecbe10edd47eb403a328bdb2563f7d30e185d11e73d1b10f1955da4a1",
    RUNTIME_TEST_PATH: "de253087afd14136f188d3c525be76788cd214509d96971c4f11d2799c215117",
}
RUNTIME_I2_BLOBS = {
    RUNTIME_CONFIG_PATH: "3d5af87e7512568ed663b211c24a8586eeb9f03936a397cf2d2ddaeb2a21f57b",
    RUNTIME_SCRIPT_PATH: "44dcda8897e747cfe363668ddc23d8dd9c53a7f3ffab692a1bb4e7cf738973ca",
    RUNTIME_TEST_PATH: "ff250d4f011d8526e9a4a7bf13049f1f47346faa1c7ea512cbf447a6fb59ba4a",
}
RUNTIME_B_BLOBS = {
    RUNTIME_CONFIG_PATH: "e5c1f96ec57b220fd36ff4677deb37d6dc0be06e02f21af3837e17a51e91e5ee",
    RUNTIME_SCRIPT_PATH: RUNTIME_I2_BLOBS[RUNTIME_SCRIPT_PATH],
    RUNTIME_TEST_PATH: RUNTIME_I2_BLOBS[RUNTIME_TEST_PATH],
}

GSE217_CONFIG_PATH = "configs/route_a_v3_gse217518_corrected_a1_successor_candidate_v1.json"
GSE217_SCRIPT_PATH = "scripts/route_a_v3/preflight_gse217518_corrected_a1_successor_candidate.py"
GSE217_TEST_PATH = "tests/route_a_v3/test_preflight_gse217518_corrected_a1_successor_candidate.py"
GSE217_EXACT3 = (GSE217_CONFIG_PATH, GSE217_SCRIPT_PATH, GSE217_TEST_PATH)
ENCSR_CONFIG_PATH = "configs/route_a_v3_encsr854ruf_dec027_dataset_specific_a1_preflight_v1.json"
ENCSR_SCRIPT_PATH = "scripts/route_a_v3/preflight_encsr854ruf_dec027_dataset_specific_a1.py"
ENCSR_TEST_PATH = "tests/route_a_v3/test_preflight_encsr854ruf_dec027_dataset_specific_a1.py"
ENCSR_EXACT3 = (ENCSR_CONFIG_PATH, ENCSR_SCRIPT_PATH, ENCSR_TEST_PATH)
GSE232_CONFIG_PATH = "configs/route_a_v3_gse232572_corrected_a1_replay_v1.json"
GSE232_SCRIPT_PATH = "scripts/route_a_v3/replay_gse232572_corrected_a1.py"
GSE232_TEST_PATH = "tests/route_a_v3/test_replay_gse232572_corrected_a1.py"
GSE232_EXACT3 = (GSE232_CONFIG_PATH, GSE232_SCRIPT_PATH, GSE232_TEST_PATH)
GSE113_CONFIG_PATH = "configs/route_a_v3_gse113849_designed_snv_true_a2_preflight_v1.json"
GSE113_SCRIPT_PATH = "scripts/route_a_v3/preflight_gse113849_designed_snv_true_a2.py"
GSE113_TEST_PATH = "tests/route_a_v3/test_preflight_gse113849_designed_snv_true_a2.py"
GSE113_EXACT3 = (GSE113_CONFIG_PATH, GSE113_SCRIPT_PATH, GSE113_TEST_PATH)

GSE217_HISTORY: tuple[dict[str, Any], ...] = (
    {
        "step": "I1",
        "commit": "17a35f0f88cc988b938aaf25d94a8b32f0cacfc8",
        "expected_parent": RUNTIME_B_COMMIT,
        "exact_changed_paths": list(GSE217_EXACT3),
        "blob_sha256_by_path": {
            GSE217_CONFIG_PATH: "0aa3324d3cfdfd50837ea32a4d1efef754fe70abdab9805f373401f21a1ccb41",
            GSE217_SCRIPT_PATH: "6ca04bdc464ac30f1c3b83830b74c6621816bd25308e345f39e2c5ee94f21b4c",
            GSE217_TEST_PATH: "b08209856fb852991c1b795864304fcda62a4f63419197c068ec6d1f0fd34691",
        },
    },
    {
        "step": "I2",
        "commit": "6fbd63be6d0edb9f73cf2f85e446917d3c3ff100",
        "expected_parent": "17a35f0f88cc988b938aaf25d94a8b32f0cacfc8",
        "exact_changed_paths": list(GSE217_EXACT3),
        "blob_sha256_by_path": {
            GSE217_CONFIG_PATH: "de064e62e7031725908de8d09a1c6b7a2a36112868208ba0cb387a394062a1d8",
            GSE217_SCRIPT_PATH: "0c96d41e4f9ddd694be0be21aab3bcfcb938d3ccb996f3f7cb10ca0a9b69902e",
            GSE217_TEST_PATH: "c881350f5b6ab457af24a4004eedfa34e30d0c8fac7e1b99f74b37b9fe40ccc7",
        },
    },
    {
        "step": "B2",
        "commit": "c3611b0f2e8baeb83422bb07f5446b42edce90ef",
        "expected_parent": "6fbd63be6d0edb9f73cf2f85e446917d3c3ff100",
        "exact_changed_paths": [GSE217_CONFIG_PATH],
        "blob_sha256_by_path": {
            GSE217_CONFIG_PATH: "c808bdc6eb1ad8aaccd3d2ab483415ed0803da4683315b92ef3139f203f61e64",
            GSE217_SCRIPT_PATH: "0c96d41e4f9ddd694be0be21aab3bcfcb938d3ccb996f3f7cb10ca0a9b69902e",
            GSE217_TEST_PATH: "c881350f5b6ab457af24a4004eedfa34e30d0c8fac7e1b99f74b37b9fe40ccc7",
        },
    },
    {
        "step": "I3",
        "commit": "36b535f77b3f27bb872b182dcaf6c646d9781991",
        "expected_parent": "c3611b0f2e8baeb83422bb07f5446b42edce90ef",
        "exact_changed_paths": list(GSE217_EXACT3),
        "blob_sha256_by_path": {
            GSE217_CONFIG_PATH: "3355ba986f60268d3dd7b985ed31fe9df4aa2acbe9fd03c984956a1270279ff9",
            GSE217_SCRIPT_PATH: "9fa4464e1cc42baacdf39b4bae2427e1895269b8d6f4e1a05e1e944b0434f3fa",
            GSE217_TEST_PATH: "bba0ee97e9ed2500f0155c8d1a776d185661b00d8c7fd48c3c0d718d53ccd097",
        },
    },
    {
        "step": "B3",
        "commit": "0a46400efee4ead95b1283df73d263f6f8033036",
        "expected_parent": "36b535f77b3f27bb872b182dcaf6c646d9781991",
        "exact_changed_paths": [GSE217_CONFIG_PATH],
        "blob_sha256_by_path": {
            GSE217_CONFIG_PATH: "c5acc8548ab8542ac029a420f21f1d8524bb0f255c6dd53c2d896c2838ce391f",
            GSE217_SCRIPT_PATH: "9fa4464e1cc42baacdf39b4bae2427e1895269b8d6f4e1a05e1e944b0434f3fa",
            GSE217_TEST_PATH: "bba0ee97e9ed2500f0155c8d1a776d185661b00d8c7fd48c3c0d718d53ccd097",
        },
    },
)
GSE217_FINAL_B = GSE217_HISTORY[-1]["commit"]

ENCSR_HISTORY: tuple[dict[str, Any], ...] = (
    {
        "step": "I1",
        "commit": "c6132d8928df0a64be106b11ee62d225d77249ba",
        "expected_parent": GSE217_FINAL_B,
        "exact_changed_paths": list(ENCSR_EXACT3),
        "blob_sha256_by_path": {
            ENCSR_CONFIG_PATH: "e1d3747876818f5b0d2b47f4a185cc5fb0f1c6b141b25a1e635768cdde588e2c",
            ENCSR_SCRIPT_PATH: "d8f6517f935624204cfa8669c8322909734417b212287e867ea38d8e031881ec",
            ENCSR_TEST_PATH: "b59e94373fb02cb2a0e65b67183af9b2f3ddcab24bc8f72d4a636f9a781f4714",
        },
    },
    {
        "step": "I2",
        "commit": "5531907c9ede1a4323ffe884c47a410d9bcb946d",
        "expected_parent": "c6132d8928df0a64be106b11ee62d225d77249ba",
        "exact_changed_paths": list(ENCSR_EXACT3),
        "blob_sha256_by_path": {
            ENCSR_CONFIG_PATH: "e7f4adf157b638c10161c922d848c494aa6b3b50f8aed9c05c7111907bb691c8",
            ENCSR_SCRIPT_PATH: "4a5910cad545d4b699b2daf20933afe3e6512aff11b015cb6adb983f4911c247",
            ENCSR_TEST_PATH: "364b908433353451501b0419587d20d6702451ae87726231e3ac1800313e60b7",
        },
    },
    {
        "step": "B2",
        "commit": "e52a8d8614724574e3647c6cf0f84041221b76a0",
        "expected_parent": "5531907c9ede1a4323ffe884c47a410d9bcb946d",
        "exact_changed_paths": [ENCSR_CONFIG_PATH],
        "blob_sha256_by_path": {
            ENCSR_CONFIG_PATH: "2f3d688f463f5ee359ae76aa9111af2d9ee091f77a1c8037d2337feb49583045",
            ENCSR_SCRIPT_PATH: "4a5910cad545d4b699b2daf20933afe3e6512aff11b015cb6adb983f4911c247",
            ENCSR_TEST_PATH: "364b908433353451501b0419587d20d6702451ae87726231e3ac1800313e60b7",
        },
    },
    {
        "step": "I3",
        "commit": "c0f65f181ea797978d660ef3c918ee7318a51292",
        "expected_parent": "e52a8d8614724574e3647c6cf0f84041221b76a0",
        "exact_changed_paths": list(ENCSR_EXACT3),
        "blob_sha256_by_path": {
            ENCSR_CONFIG_PATH: "fe22477d631cc1ce08e9eed6cd0ca48b4723bf9757a16dd37c4a391cc3318263",
            ENCSR_SCRIPT_PATH: "6235c78ea8bcb008fb33b9d3356461bfb0446a516111d941c640e7a9933d6bac",
            ENCSR_TEST_PATH: "a7303cca44bbb0340251fb00640456fdea592bd461f49852510eb719a1c401c5",
        },
    },
    {
        "step": "B3",
        "commit": "d38f4b31cd5add04bbd7f3b839ff60590fa5fad2",
        "expected_parent": "c0f65f181ea797978d660ef3c918ee7318a51292",
        "exact_changed_paths": [ENCSR_CONFIG_PATH],
        "blob_sha256_by_path": {
            ENCSR_CONFIG_PATH: "f801ac603ccd8903921e2e513f5b363ce5112ed6b220e93ab6ddf896dcde3ceb",
            ENCSR_SCRIPT_PATH: "6235c78ea8bcb008fb33b9d3356461bfb0446a516111d941c640e7a9933d6bac",
            ENCSR_TEST_PATH: "a7303cca44bbb0340251fb00640456fdea592bd461f49852510eb719a1c401c5",
        },
    },
    {
        "step": "I4",
        "commit": "53f426aef8b12e8dcbfaaf978fcfa7d1c7a911d2",
        "expected_parent": "d38f4b31cd5add04bbd7f3b839ff60590fa5fad2",
        "exact_changed_paths": list(ENCSR_EXACT3),
        "blob_sha256_by_path": {
            ENCSR_CONFIG_PATH: "1ebbbe07339ccf8b6bf88d1e8fb946976d2065073c479affe8811af3cc1bd088",
            ENCSR_SCRIPT_PATH: "d5a1ef3e174f479404c3ca1b2dcac9b81c3848b2a7008c9333ca5f339f2d15c9",
            ENCSR_TEST_PATH: "1e31a4dd3643f1c8a3b56e3e6bd0f99b1f01cac65b4c6c3b7113aad5c26ee5b2",
        },
    },
    {
        "step": "B4",
        "commit": "56b39f966a272d8ea8022048855d2fcca0ee155a",
        "expected_parent": "53f426aef8b12e8dcbfaaf978fcfa7d1c7a911d2",
        "exact_changed_paths": [ENCSR_CONFIG_PATH],
        "blob_sha256_by_path": {
            ENCSR_CONFIG_PATH: "de9df6055a83f29351c4eba2dd895708de4890181e639e26f2228105f6c2cc07",
            ENCSR_SCRIPT_PATH: "d5a1ef3e174f479404c3ca1b2dcac9b81c3848b2a7008c9333ca5f339f2d15c9",
            ENCSR_TEST_PATH: "1e31a4dd3643f1c8a3b56e3e6bd0f99b1f01cac65b4c6c3b7113aad5c26ee5b2",
        },
    },
)
ENCSR_FINAL_B = ENCSR_HISTORY[-1]["commit"]

CONFIG_REPO_PATH = "configs/route_a_v3_gse269595_corrected_role_adjudication_successor_candidate_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/preflight_gse269595_corrected_role_adjudication_successor_candidate.py"
TEST_REPO_PATH = "tests/route_a_v3/test_preflight_gse269595_corrected_role_adjudication_successor_candidate.py"
EXACT3 = (CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH)
OWN_BINDING_FIELDS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
FUTURE_PREDECESSOR_FIELDS = (
    "status",
    "append_only_history",
    "terminal_binding_commit",
)
PUBLIC_ASSET_IDENTITIES = {
    "GSE269595_mpra_constructs_all_samples_proximal_site_usage.txt.gz": (
        5910248,
        "6527ea54257b3f17ddb9df5977637e41e2ef16d926a27be4e29f56165acaa1de",
    ),
    "1-s2.0-S0092867424006457-mmc5.xlsx": (
        353937,
        "d350be818d87120216052645a5ffa97afeee898d32a877c74033dba0d0fa151a",
    ),
}

GATE_IDS: tuple[str, ...] = (
    "A1_VERSUS_TRUE_A2_ROLE_ELIGIBILITY_AND_MUTUAL_EXCLUSIVITY_CLOSED",
    "ORDINARY_PUBLIC_ASSET_IDENTITY_ROLE_AND_PROVENANCE_CLOSED",
    "ELIGIBLE_DENSE_SOURCE_FAMILY_DISTRIBUTION_AND_UNIQUE_SOURCE_ANCHOR_CLOSED",
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

MPRA_HEADER: tuple[str, ...] = (
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
TABLE_S5_HEADER: tuple[str, ...] = (
    "gene_id",
    "pas_id",
    "type",
    "subtype",
    "experiment",
    "n_bc",
    "barcoded_seq_184bp",
)
EXPECTED_SAMPLE_FIELDS: dict[str, tuple[str, str]] = {
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
EXPECTED_DISTAL_CONTEXTS: tuple[str, ...] = (
    "CCT6A_moduleA",
    "CDK1_moduleB",
    "TMEM106C_moduleB",
    "TMEM237_moduleA",
    "bGH",
)

FORBIDDEN_OUTPUT_KEYS = {
    "member_id",
    "barcode",
    "variant",
    "source_sequence",
    "candidate_sequence",
    "sequence",
    "row_endpoint",
    "row_effect",
    "row_standard_error",
    "replicate_id",
    "split_assignment",
}

_OOXML_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OOXML_DOC_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_OOXML_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


class CandidateContractError(RuntimeError):
    """The corrected candidate protocol or aggregate observation differs."""


class BindingNotFrozen(RuntimeError):
    """Production execution was attempted before grouped bindings closed."""


class PublicAssetError(RuntimeError):
    """An ordinary-public asset differs from the frozen identity or schema."""


class OutputError(RuntimeError):
    """The single atomic report publication contract was not met."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise CandidateContractError(f"non-finite JSON constant: {token}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateContractError(f"cannot read strict candidate config: {path}") from exc
    if not isinstance(value, dict):
        raise CandidateContractError("candidate config root must be an object")
    return value


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_sha_map(value: Any, paths: Sequence[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(paths):
        raise CandidateContractError(f"{label} blob path closure differs")
    if any(not _is_hex(digest, 64) for digest in value.values()):
        raise CandidateContractError(f"{label} blob SHA-256 differs")


def _validate_authority_binding(binding: Mapping[str, Any]) -> None:
    if dict(binding) != {
        "status": BOUND,
        "authority_commit": AUTHORITY_COMMIT,
        "authority_expected_parent": AUTHORITY_PARENT,
        "authority_exact_changed_paths": list(AUTHORITY_EXACT12),
        "authority_blob_sha256_by_path": AUTHORITY_BLOBS,
    }:
        raise CandidateContractError("DEC027 authority exact12 binding differs")


def _validate_runtime_binding(binding: Mapping[str, Any]) -> None:
    expected = {
        "status": BOUND,
        "runtime_event_id": BASE_EVENT,
        "frozen_i1_commit": RUNTIME_I1_COMMIT,
        "frozen_i1_expected_parent": AUTHORITY_COMMIT,
        "frozen_i1_exact_changed_paths": list(RUNTIME_EXACT3),
        "frozen_i1_blob_sha256_by_path": RUNTIME_I1_BLOBS,
        "implementation_commit": RUNTIME_I2_COMMIT,
        "implementation_expected_parent": RUNTIME_I1_COMMIT,
        "implementation_exact_changed_paths": list(RUNTIME_EXACT3),
        "implementation_blob_sha256_by_path": RUNTIME_I2_BLOBS,
        "binding_commit": RUNTIME_B_COMMIT,
        "binding_expected_parent": RUNTIME_I2_COMMIT,
        "binding_exact_changed_paths": [RUNTIME_CONFIG_PATH],
        "binding_blob_sha256_by_path": RUNTIME_B_BLOBS,
    }
    if dict(binding) != expected:
        raise CandidateContractError("DEC027 runtime I1/I2/B2 binding differs")


def _validate_frozen_history(
    binding: Mapping[str, Any],
    *,
    name: str,
    history: Sequence[Mapping[str, Any]],
) -> None:
    expected = {
        "status": BOUND,
        "append_only_history": [dict(step) for step in history],
        "terminal_binding_commit": history[-1]["commit"],
    }
    if dict(binding) != expected:
        raise CandidateContractError(f"{name} frozen append-only history differs")


def _validate_append_only_history(
    history: Any,
    *,
    name: str,
    exact3: Sequence[str],
    config_path: str,
    expected_parent: str,
) -> str:
    if not isinstance(history, list) or len(history) < 2:
        raise CandidateContractError(f"{name} append-only history is incomplete")
    parent = expected_parent
    next_implementation = 1
    latest_unbound_implementation: int | None = None
    for index, step in enumerate(history):
        if not isinstance(step, Mapping) or set(step) != {
            "step",
            "commit",
            "expected_parent",
            "exact_changed_paths",
            "blob_sha256_by_path",
        }:
            raise CandidateContractError(f"{name} history step schema differs")
        label = step.get("step")
        match = re.fullmatch(r"([IB])([1-9][0-9]*)", str(label))
        if match is None:
            raise CandidateContractError(f"{name} history step label differs")
        kind, number = match.group(1), int(match.group(2))
        if kind == "I":
            if number != next_implementation:
                raise CandidateContractError(f"{name} implementation order differs")
            latest_unbound_implementation = number
            next_implementation += 1
            expected_paths = list(exact3)
        else:
            if latest_unbound_implementation != number:
                raise CandidateContractError(f"{name} binding order differs")
            latest_unbound_implementation = None
            expected_paths = [config_path]
        commit = step.get("commit")
        if not _is_hex(commit, 40) or step.get("expected_parent") != parent:
            raise CandidateContractError(f"{name} history direct parent differs")
        if step.get("exact_changed_paths") != expected_paths:
            raise CandidateContractError(f"{name} history changed paths differ")
        _validate_sha_map(
            step.get("blob_sha256_by_path"), exact3, f"{name} {label}"
        )
        parent = str(commit)
        if index == len(history) - 1 and kind != "B":
            raise CandidateContractError(f"{name} history lacks terminal binding")
    return parent


def _future_predecessor_mode(
    binding: Mapping[str, Any],
    *,
    name: str,
    exact3: Sequence[str],
    config_path: str,
    expected_parent: str | None,
    predecessor_mode: str,
) -> str:
    expected_keys = {
        "status",
        "implementation_exact_changed_paths",
        "binding_exact_changed_paths",
        "append_only_history",
        "terminal_binding_commit",
        "unknown_to_bound_fields",
    }
    if set(binding) != expected_keys:
        raise CandidateContractError(f"{name} binding group schema differs")
    if binding.get("unknown_to_bound_fields") != list(FUTURE_PREDECESSOR_FIELDS):
        raise CandidateContractError(f"{name} grouped binding fields differ")
    if binding.get("implementation_exact_changed_paths") != list(exact3):
        raise CandidateContractError(f"{name} implementation exact3 differs")
    if binding.get("binding_exact_changed_paths") != [config_path]:
        raise CandidateContractError(f"{name} binding changed path differs")
    dynamic = tuple(binding.get(field) for field in FUTURE_PREDECESSOR_FIELDS)
    if binding.get("status") == UNKNOWN:
        if dynamic != (UNKNOWN,) * len(FUTURE_PREDECESSOR_FIELDS):
            raise CandidateContractError(f"{name} binding is partially populated")
        return UNKNOWN
    if binding.get("status") != BOUND or predecessor_mode != BOUND:
        raise CandidateContractError(f"{name} predecessor is not bound")
    if expected_parent is None:
        raise CandidateContractError(f"{name} initial parent is unknown")
    terminal = _validate_append_only_history(
        binding.get("append_only_history"),
        name=name,
        exact3=exact3,
        config_path=config_path,
        expected_parent=expected_parent,
    )
    if binding.get("terminal_binding_commit") != terminal:
        raise CandidateContractError(f"{name} terminal binding differs")
    return BOUND


def _implementation_binding_mode(
    binding: Mapping[str, Any], *, predecessor_mode: str
) -> str:
    if binding.get("unknown_to_bound_fields") != list(OWN_BINDING_FIELDS):
        raise CandidateContractError("own four-scalar binding group differs")
    if binding.get("implementation_script_path") != SCRIPT_REPO_PATH:
        raise CandidateContractError("own implementation script path differs")
    if binding.get("implementation_test_path") != TEST_REPO_PATH:
        raise CandidateContractError("own implementation test path differs")
    if binding.get("implementation_exact_changed_paths") != list(EXACT3):
        raise CandidateContractError("own implementation exact3 differs")
    if binding.get("binding_exact_changed_paths") != [CONFIG_REPO_PATH]:
        raise CandidateContractError("own binding changed path differs")
    dynamic = tuple(binding.get(field) for field in OWN_BINDING_FIELDS)
    if binding.get("status") == UNKNOWN:
        if dynamic != (UNKNOWN,) * len(OWN_BINDING_FIELDS):
            raise CandidateContractError("own binding is partially populated")
        return UNKNOWN
    if binding.get("status") != BOUND:
        raise CandidateContractError("own binding status differs")
    if predecessor_mode != BOUND:
        raise CandidateContractError("own predecessor is not bound")
    if not _is_hex(binding.get("implementation_commit"), 40):
        raise CandidateContractError("own implementation commit differs")
    if not _is_hex(binding.get("implementation_script_sha256"), 64):
        raise CandidateContractError("own implementation script SHA-256 differs")
    if not _is_hex(binding.get("implementation_test_sha256"), 64):
        raise CandidateContractError("own implementation test SHA-256 differs")
    return BOUND


def _binding_modes(config: Mapping[str, Any]) -> dict[str, str]:
    bindings = config.get("bindings")
    expected_groups = {
        "authority",
        "runtime",
        "gse217518_predecessor",
        "encsr854ruf_predecessor",
        "gse232572_predecessor",
        "gse113849_predecessor",
        "implementation",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_groups:
        raise CandidateContractError("binding group closure differs")
    if any(not isinstance(value, Mapping) for value in bindings.values()):
        raise CandidateContractError("binding group must be an object")
    _validate_authority_binding(bindings["authority"])
    _validate_runtime_binding(bindings["runtime"])
    _validate_frozen_history(
        bindings["gse217518_predecessor"],
        name="GSE217518",
        history=GSE217_HISTORY,
    )
    _validate_frozen_history(
        bindings["encsr854ruf_predecessor"],
        name="ENCSR854RUF",
        history=ENCSR_HISTORY,
    )
    gse217_mode = BOUND
    encsr_mode = BOUND
    gse232_mode = _future_predecessor_mode(
        bindings["gse232572_predecessor"],
        name="GSE232572",
        exact3=GSE232_EXACT3,
        config_path=GSE232_CONFIG_PATH,
        expected_parent=ENCSR_FINAL_B,
        predecessor_mode=encsr_mode,
    )
    gse232_b = bindings["gse232572_predecessor"].get("terminal_binding_commit")
    gse113_mode = _future_predecessor_mode(
        bindings["gse113849_predecessor"],
        name="GSE113849",
        exact3=GSE113_EXACT3,
        config_path=GSE113_CONFIG_PATH,
        expected_parent=str(gse232_b) if gse232_mode == BOUND else None,
        predecessor_mode=gse232_mode,
    )
    own_mode = _implementation_binding_mode(
        bindings["implementation"], predecessor_mode=gse113_mode
    )
    return {
        "gse217518_predecessor": gse217_mode,
        "encsr854ruf_predecessor": encsr_mode,
        "gse232572_predecessor": gse232_mode,
        "gse113849_predecessor": gse113_mode,
        "implementation": own_mode,
    }


def validate_protocol(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "decision_id": "V3-DEC-027",
        "document_status": "DRAFT_CORRECTED_SUCCESSOR_NOT_ACTIVE_PROTOCOL",
        "dataset_id": "GSE269595",
        "project_id": "PRJNA1122592",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise CandidateContractError(f"protocol {key} differs")
    if config.get("required_gate_ids_exactly") != list(GATE_IDS):
        raise CandidateContractError("the DEC027 exact thirteen gate IDs differ")

    baseline = config.get("baseline")
    if not isinstance(baseline, Mapping) or dict(baseline) != {
        "remote_branch": PRODUCTION_BRANCH,
        "dec027_authority_head": AUTHORITY_COMMIT,
        "dec027_authority_parent": AUTHORITY_PARENT,
        "pre_dec027_projection_event": "A1-EVT-058",
        "bound_runtime_event": BASE_EVENT,
        "gse217518_final_binding_commit": GSE217_FINAL_B,
        "encsr854ruf_final_binding_commit": ENCSR_FINAL_B,
        "current_projection_status": "SETTLED_RUNTIME_CURRENT_PROJECTION",
    }:
        raise CandidateContractError("DEC027 baseline snapshot differs")

    repository = config.get("repository_authority")
    if not isinstance(repository, Mapping) or dict(repository) != {
        "production_repo_root": str(PRODUCTION_REPO_ROOT),
        "branch": PRODUCTION_BRANCH,
        "upstream_ref": PRODUCTION_UPSTREAM,
        "live_origin_head_required": True,
        "clean_worktree_and_index_required": True,
    }:
        raise CandidateContractError("repository authority differs")
    _binding_modes(config)

    activation = config.get("production_activation_rule")
    expected_activation = {
        "required_commit_chain": (
            "DEC027_A_TO_RUNTIME_I1_TO_RUNTIME_I2_TO_RUNTIME_B2_"
            "TO_GSE217518_I1_I2_B2_I3_B3_"
            "TO_ENCSR854RUF_I1_I2_B2_I3_B3_I4_B4_"
            "TO_GSE232572_APPEND_ONLY_HISTORY_"
            "TO_GSE113849_APPEND_ONLY_HISTORY_"
            "TO_GSE269595_I_TO_GSE269595_B"
        ),
        "predecessor_order": [
            "gse217518_predecessor",
            "encsr854ruf_predecessor",
            "gse232572_predecessor",
            "gse113849_predecessor",
        ],
        "all_binding_groups_must_be_bound": True,
        "gse269595_implementation_i_must_be_direct_child_of_gse113849_b": True,
        "clean_head_equals_upstream_equals_live_origin_required": True,
        "direct_parent_changed_path_and_blob_audit_required": True,
        "executing_script_and_focused_test_must_match_implementation_i": True,
        "binding_commit_may_change_only_the_four_own_binding_scalars": True,
        "fail_before_git_asset_or_output_while_any_predecessor_or_own_group_is_unknown": True,
    }
    if not isinstance(activation, Mapping) or dict(activation) != expected_activation:
        raise CandidateContractError("production activation rule differs")

    execution = config.get("execution_boundary")
    if not isinstance(execution, Mapping) or dict(execution) != {
        "production_mode": "SINGLE_BOUND_PRODUCTION_ENTRY_FIXED_BUILT_IN_PUBLIC_READER_ONLY",
        "ordinary_public_analysis_mode": "NO_SEPARATE_PUBLIC_ANALYSIS_BYPASS",
        "asset_argument_mode": "SINGLE_DIRECTORY_WITH_EXACT_TWO_FROZEN_BASENAMES",
        "identity_and_context_validation_before_parse": True,
        "ordinary_public_inputs_only": True,
        "private_or_sealed_access_allowed": False,
        "persistent_member_level_intermediate_allowed": False,
        "member_or_row_payload_output_allowed": False,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_allowed": False,
    }:
        raise CandidateContractError("single-entry execution boundary differs")

    assets = config.get("official_asset_contract")
    if not isinstance(assets, Mapping) or set(assets) != {
        "processed_mpra",
        "publisher_table_s5",
    }:
        raise CandidateContractError("official public asset closure differs")
    for name in ("processed_mpra", "publisher_table_s5"):
        contract = assets[name]
        if not isinstance(contract, Mapping):
            raise CandidateContractError(f"{name} asset contract is absent")
        filename = contract.get("filename")
        if filename not in PUBLIC_ASSET_IDENTITIES:
            raise CandidateContractError(f"{name} asset basename differs")
        if (contract.get("byte_count"), contract.get("sha256")) != (
            PUBLIC_ASSET_IDENTITIES[str(filename)]
        ):
            raise CandidateContractError(f"{name} asset identity differs")
    if assets["processed_mpra"].get("exact_data_row_count") != 366780:
        raise CandidateContractError("processed MPRA row geometry differs")
    if assets["publisher_table_s5"].get("required_sheet") != (
        "MPRA oligo library"
    ) or assets["publisher_table_s5"].get("required_sheet_dimension") != "A1:G6114":
        raise CandidateContractError("publisher Table S5 context differs")

    publisher = config.get("publisher_facts")
    if not isinstance(publisher, Mapping) or (
        publisher.get("model_scope") != "TANDEM_POLYA_SITE_USAGE"
        or publisher.get("endpoint_identity") != "PROXIMAL_POLYA_SITE_USAGE"
        or publisher.get("endpoint_scale")
        != "LOG2_PROXIMAL_READS_DIVIDED_BY_DISTAL_READS"
        or publisher.get("endpoint_direction")
        != "HIGHER_IS_MORE_PROXIMAL_SITE_USAGE"
        or publisher.get("biological_replicate_count_per_genetic_condition") != 2
        or publisher.get("source_locus_selection_uses_model_prediction_and_measured_perturbation_response")
        is not True
        or publisher.get("candidate_design_uses_aparent_prediction_or_interpretation")
        is not True
    ):
        raise CandidateContractError("publisher assay and exposure context differs")

    output = config.get("output_contract")
    if not isinstance(output, Mapping) or dict(output) != {
        "report_filename": REPORT_FILENAME,
        "aggregate_only": True,
        "atomic_no_replace_publication": True,
        "identical_existing_report_is_idempotent_success": True,
        "member_identifier_output_allowed": False,
        "sequence_output_allowed": False,
        "row_endpoint_or_effect_output_allowed": False,
        "row_standard_error_output_allowed": False,
        "replicate_identifier_output_allowed": False,
        "split_assignment_output_allowed": False,
    }:
        raise CandidateContractError("single aggregate output contract differs")

    decision = config.get("decision_snapshot")
    if not isinstance(decision, Mapping) or (
        decision.get("aggregate_preflight_execution_allowed") is not True
        or decision.get("qualification_allowed") is not False
        or decision.get("credit_may_be_inferred_from_preflight_status") is not False
    ):
        raise CandidateContractError("DEC027 decision boundary differs")

    corrected = config.get("corrected_successor_rules")
    if not isinstance(corrected, Mapping):
        raise CandidateContractError("corrected successor rules are absent")
    required_corrections = {
        "maximum_roles_if_later_qualified": 1,
        "a1_and_true_a2_double_credit_allowed": False,
        "minimum_distinct_candidates_per_dense_source": 3,
        "two_candidate_families_are_pairwise_only_and_excluded_from_dense_universe": True,
        "one_two_candidate_family_may_fail_all_other_dense_families": False,
        "finite_endpoint_required": True,
        "nonfinite_or_undefined_endpoint_requires_prefrozen_censoring_policy": True,
        "published_biological_replicate_count_reference_only": 2,
        "published_replicate_fact_may_be_hardcoded_false": False,
        "valid_standard_error_requires_independent_audit": True,
        "five_declared_multiplicity_discrepancies_are_schema_coverage_failure": False,
        "legal_sequence_diff_replay_may_be_failed_only_because_separate_annotation_column_is_absent": False,
    }
    if dict(corrected) != required_corrections:
        raise CandidateContractError("corrected successor rules differ")

    scope = config.get("scope")
    if not isinstance(scope, Mapping):
        raise CandidateContractError("scope is absent")
    for key in ("ordinary_public_only", "aggregate_output_only"):
        if scope.get(key) is not True:
            raise CandidateContractError(f"scope.{key} must be true")
    for key in (
        "member_identifier_output_allowed",
        "sequence_output_allowed",
        "row_endpoint_output_allowed",
        "row_effect_output_allowed",
        "row_standard_error_output_allowed",
        "replicate_identifier_output_allowed",
        "split_assignment_output_allowed",
        "persistent_member_level_intermediate_allowed",
        "private_or_sealed_access_allowed",
        "qualification_allowed",
        "credit_change_allowed",
        "canonical_materialization_allowed",
        "training_allowed",
        "gpu_work_allowed",
        "model_selection_allowed",
        "a7_allowed",
    ):
        if scope.get(key) is not False:
            raise CandidateContractError(f"scope.{key} must be false")

    power = config.get("split_and_power_policy")
    if not isinstance(power, Mapping) or (
        power.get("analysis_unit") != "POST_DEDUP_INDEPENDENT_SOURCE_GROUP"
        or power.get("near_duplicate_identity_threshold") != 0.8
        or power.get("alternative_spearman_rho") != 0.25
        or power.get("alpha_two_sided") != 0.05
        or power.get("target_power_minimum") != 0.8
        or power.get("confidence_level") != 0.95
        or power.get("maximum_full_ci_width") != 0.3
        or power.get("required_effective_n_reference") != 156
        or power.get("formal_qualification_power_execution_allowed") is not False
        or power.get("split_assignment_execution_allowed") is not False
    ):
        raise CandidateContractError("split/power policy differs")

    terminal = config.get("terminal_state")
    if not isinstance(terminal, Mapping) or terminal.get("qualified") is not False:
        raise CandidateContractError("terminal no-promotion state differs")
    if terminal.get("contribution") != {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }:
        raise CandidateContractError("dataset contribution lock differs")


def _require_production_bindings(config: Mapping[str, Any]) -> None:
    if any(mode != BOUND for mode in _binding_modes(config).values()):
        raise BindingNotFrozen(
            "ordered predecessor or own binding remains grouped UNKNOWN; "
            "production stops before Git/asset/output I/O"
        )


def _run_git(repo_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise CandidateContractError("Git is unavailable for binding audit") from exc
    if result.returncode != 0:
        raise CandidateContractError("Git binding audit command failed")
    return result.stdout.strip()


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{path}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise CandidateContractError("Git is unavailable for blob audit") from exc
    if result.returncode != 0:
        raise CandidateContractError("bound Git blob is not readable")
    return result.stdout


def _changed_paths(repo_root: Path, commit: str) -> tuple[str, ...]:
    value = _run_git(
        repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
    )
    return tuple(sorted(line for line in value.splitlines() if line))


def _verify_frozen_commit(
    repo_root: Path,
    *,
    label: str,
    commit: str,
    expected_parent: str,
    expected_paths: Sequence[str],
    expected_blobs: Mapping[str, str] | None = None,
) -> None:
    ancestry = _run_git(repo_root, "rev-list", "--parents", "-n", "1", commit)
    fields = ancestry.split()
    if len(fields) != 2 or fields[0] != commit or fields[1] != expected_parent:
        raise CandidateContractError(f"{label} direct parent differs")
    if _changed_paths(repo_root, commit) != tuple(sorted(expected_paths)):
        raise CandidateContractError(f"{label} changed paths differ")
    for path, expected_sha in (expected_blobs or {}).items():
        actual_sha = hashlib.sha256(_git_blob(repo_root, commit, path)).hexdigest()
        if actual_sha != expected_sha:
            raise CandidateContractError(f"{label} blob identity differs")


def _live_origin_head(repo_root: Path, branch: str) -> str:
    ref = f"refs/heads/{branch}"
    value = _run_git(repo_root, "ls-remote", "--exit-code", "--heads", "origin", ref)
    lines = [line.split() for line in value.splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0]) != 2 or lines[0][1] != ref:
        raise CandidateContractError("live origin branch resolution differs")
    commit = lines[0][0]
    if not _is_hex(commit, 40):
        raise CandidateContractError("live origin head is invalid")
    return commit


def _normalise_own_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    normalised = copy.deepcopy(dict(config))
    own = normalised["bindings"]["implementation"]
    for field in OWN_BINDING_FIELDS:
        own[field] = UNKNOWN
    return normalised


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateContractError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise CandidateContractError(f"{label} root is not an object")
    return value


def _audit_repository_bindings(
    config: Mapping[str, Any], config_path: Path, repo_root: Path
) -> dict[str, str]:
    """Audit A/runtime and every frozen append-only predecessor step."""

    _require_production_bindings(config)
    repository = config["repository_authority"]
    if repo_root.resolve() != Path(str(repository["production_repo_root"])).resolve():
        raise CandidateContractError("execution repository is not the frozen root")
    if config_path.resolve() != (repo_root / CONFIG_REPO_PATH).resolve():
        raise CandidateContractError("config path is outside the frozen repository")

    head = _run_git(repo_root, "rev-parse", "HEAD")
    upstream = _run_git(repo_root, "rev-parse", "@{upstream}")
    live_origin = _live_origin_head(repo_root, str(repository["branch"]))
    if head != upstream or head != live_origin:
        raise CandidateContractError("HEAD, upstream, and live origin differ")
    if _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") != (
        repository["branch"]
    ):
        raise CandidateContractError("production branch differs")
    if _run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}") != (
        repository["upstream_ref"]
    ):
        raise CandidateContractError("production upstream differs")
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise CandidateContractError("production worktree or index is dirty")

    bindings = config["bindings"]
    authority = bindings["authority"]
    runtime = bindings["runtime"]
    gse217 = bindings["gse217518_predecessor"]
    encsr = bindings["encsr854ruf_predecessor"]
    gse232 = bindings["gse232572_predecessor"]
    gse113 = bindings["gse113849_predecessor"]
    own = bindings["implementation"]
    gse217_b = str(gse217["terminal_binding_commit"])
    encsr_b = str(encsr["terminal_binding_commit"])
    gse232_b = str(gse232["terminal_binding_commit"])
    gse113_b = str(gse113["terminal_binding_commit"])
    own_i = str(own["implementation_commit"])

    chain: list[
        tuple[str, str, str, Sequence[str], Mapping[str, str] | None]
    ] = [
        (
            "DEC027_AUTHORITY_A",
            AUTHORITY_COMMIT,
            AUTHORITY_PARENT,
            AUTHORITY_EXACT12,
            authority["authority_blob_sha256_by_path"],
        ),
        (
            "DEC027_RUNTIME_I1",
            RUNTIME_I1_COMMIT,
            AUTHORITY_COMMIT,
            RUNTIME_EXACT3,
            runtime["frozen_i1_blob_sha256_by_path"],
        ),
        (
            "DEC027_RUNTIME_I2",
            RUNTIME_I2_COMMIT,
            RUNTIME_I1_COMMIT,
            RUNTIME_EXACT3,
            runtime["implementation_blob_sha256_by_path"],
        ),
        (
            "DEC027_RUNTIME_B2",
            RUNTIME_B_COMMIT,
            RUNTIME_I2_COMMIT,
            (RUNTIME_CONFIG_PATH,),
            runtime["binding_blob_sha256_by_path"],
        ),
    ]
    for dataset, history in (
        ("GSE217518", gse217["append_only_history"]),
        ("ENCSR854RUF", encsr["append_only_history"]),
        ("GSE232572", gse232["append_only_history"]),
        ("GSE113849", gse113["append_only_history"]),
    ):
        for step in history:
            chain.append(
                (
                    f"{dataset}_{step['step']}",
                    str(step["commit"]),
                    str(step["expected_parent"]),
                    tuple(step["exact_changed_paths"]),
                    step["blob_sha256_by_path"],
                )
            )
    chain.extend(
        [
        (
            "GSE269595_I",
            own_i,
            gse113_b,
            EXACT3,
            {
                SCRIPT_REPO_PATH: own["implementation_script_sha256"],
                TEST_REPO_PATH: own["implementation_test_sha256"],
            },
        ),
        ("GSE269595_B", head, own_i, (CONFIG_REPO_PATH,), None),
        ]
    )
    for label, commit, parent, paths, blobs in chain:
        _verify_frozen_commit(
            repo_root,
            label=label,
            commit=commit,
            expected_parent=parent,
            expected_paths=paths,
            expected_blobs=blobs,
        )

    implementation_config = _load_json_bytes(
        _git_blob(repo_root, own_i, CONFIG_REPO_PATH), "GSE269595_I_CONFIG"
    )
    if _normalise_own_binding(config) != implementation_config:
        raise CandidateContractError("GSE269595 B changed outside four own scalars")
    if config_path.read_bytes() != _git_blob(repo_root, head, CONFIG_REPO_PATH):
        raise CandidateContractError("working config differs from GSE269595 B")
    script_blob = _git_blob(repo_root, own_i, SCRIPT_REPO_PATH)
    test_blob = _git_blob(repo_root, own_i, TEST_REPO_PATH)
    executing_script = Path(__file__).resolve()
    if executing_script != (repo_root / SCRIPT_REPO_PATH).resolve():
        raise CandidateContractError("executing script is a stale copy")
    if executing_script.read_bytes() != script_blob:
        raise CandidateContractError("executing script differs from GSE269595 I")
    if (repo_root / TEST_REPO_PATH).read_bytes() != test_blob:
        raise CandidateContractError("focused test differs from GSE269595 I")
    return {
        "authority_commit": AUTHORITY_COMMIT,
        "runtime_binding_commit": RUNTIME_B_COMMIT,
        "gse217518_binding_commit": gse217_b,
        "encsr854ruf_binding_commit": encsr_b,
        "gse232572_binding_commit": gse232_b,
        "gse113849_binding_commit": gse113_b,
        "implementation_commit": own_i,
        "binding_commit": head,
    }


def _read_bound_public_asset(path: Path, spec: Mapping[str, Any]) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PublicAssetError(f"cannot read ordinary-public asset: {path}") from exc
    if len(payload) != spec["byte_count"]:
        raise PublicAssetError(f"ordinary-public asset byte count differs: {path}")
    if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise PublicAssetError(f"ordinary-public asset identity differs: {path}")
    return payload


def _xlsx_cell_column(reference: str) -> int:
    match = re.fullmatch(r"([A-Z]+)[1-9][0-9]*", reference)
    if match is None:
        raise PublicAssetError("publisher Table S5 cell reference differs")
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _xlsx_text(cell: ElementTree.Element, shared_strings: tuple[str, ...]) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(f"{{{_OOXML_MAIN_NS}}}t")
        )
    value = cell.find(f"{{{_OOXML_MAIN_NS}}}v")
    text = "" if value is None or value.text is None else value.text
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except (IndexError, ValueError) as exc:
            raise PublicAssetError("publisher Table S5 shared-string index differs") from exc
    return text


def _parse_table_s5(payload: bytes, spec: Mapping[str, Any]) -> list[dict[str, str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            relationships = ElementTree.fromstring(
                archive.read("xl/_rels/workbook.xml.rels")
            )
            targets = {
                relation.get("Id"): relation.get("Target")
                for relation in relationships.findall(
                    f"{{{_OOXML_PACKAGE_REL_NS}}}Relationship"
                )
            }
            sheets = [
                sheet
                for sheet in workbook.findall(f".//{{{_OOXML_MAIN_NS}}}sheet")
                if sheet.get("name") == spec["required_sheet"]
            ]
            if len(sheets) != 1:
                raise PublicAssetError("publisher Table S5 sheet closure differs")
            relation_id = sheets[0].get(f"{{{_OOXML_DOC_REL_NS}}}id")
            target = targets.get(relation_id)
            if not target:
                raise PublicAssetError("publisher Table S5 sheet relation is absent")
            worksheet_path = target.lstrip("/")
            if not worksheet_path.startswith("xl/"):
                worksheet_path = f"xl/{worksheet_path}"
            shared_strings: tuple[str, ...] = ()
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                shared_strings = tuple(
                    "".join(
                        node.text or ""
                        for node in item.iter(f"{{{_OOXML_MAIN_NS}}}t")
                    )
                    for item in root.findall(f"{{{_OOXML_MAIN_NS}}}si")
                )
            worksheet = ElementTree.fromstring(archive.read(worksheet_path))
    except PublicAssetError:
        raise
    except (KeyError, OSError, ValueError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise PublicAssetError("publisher Table S5 is not the expected OOXML file") from exc

    dimension = worksheet.find(f"{{{_OOXML_MAIN_NS}}}dimension")
    if dimension is None or dimension.get("ref") != spec["required_sheet_dimension"]:
        raise PublicAssetError("publisher Table S5 dimension differs")
    sheet_data = worksheet.find(f"{{{_OOXML_MAIN_NS}}}sheetData")
    if sheet_data is None:
        raise PublicAssetError("publisher Table S5 sheet data is absent")
    rows: list[tuple[str, ...]] = []
    for row in sheet_data.findall(f"{{{_OOXML_MAIN_NS}}}row"):
        values = [""] * len(TABLE_S5_HEADER)
        for cell in row.findall(f"{{{_OOXML_MAIN_NS}}}c"):
            column = _xlsx_cell_column(cell.get("r", ""))
            if column >= len(values):
                raise PublicAssetError("publisher Table S5 row width differs")
            values[column] = _xlsx_text(cell, shared_strings).strip()
        rows.append(tuple(values))
    if not rows or rows[0] != TABLE_S5_HEADER:
        raise PublicAssetError("publisher Table S5 header differs")
    records = [dict(zip(TABLE_S5_HEADER, row)) for row in rows[1:]]
    if len(records) != spec["exact_data_row_count"]:
        raise PublicAssetError("publisher Table S5 row count differs")
    return records


def _positive_integer(value: str, label: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise PublicAssetError(f"{label} is not a positive integer")
    return int(value)


def _nonnegative_integer(value: str, label: str) -> int:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        raise PublicAssetError(f"{label} is not a nonnegative integer")
    return int(value)


def _small_count_bin(value: int) -> str:
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


def _analyse_table(records: list[dict[str, str]], config: Mapping[str, Any]) -> dict[str, Any]:
    by_key: dict[str, dict[str, str]] = {}
    duplicate_key_count = 0
    design_constructs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    design_observed: Counter[tuple[str, str, str]] = Counter()
    design_declared: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    family_designs: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    family_sources: dict[tuple[str, str], set[str]] = defaultdict(set)

    for record in records:
        sequence = record["barcoded_seq_184bp"].upper()
        if len(sequence) != 184 or re.fullmatch(r"[ACGT]{184}", sequence) is None:
            raise PublicAssetError("publisher Table S5 construct alphabet/length differs")
        key = sequence[:20]
        if key in by_key:
            duplicate_key_count += 1
        else:
            by_key[key] = record
        construct = sequence[20:]
        family = (record["gene_id"], record["pas_id"])
        design = family + (record["experiment"],)
        design_constructs[design].add(construct)
        design_observed[design] += 1
        design_declared[design].add(_positive_integer(record["n_bc"], "Table S5 n_bc"))
        family_designs[family].add(design)
        if record["experiment"] == "wt":
            family_sources[family].add(construct)

    family_candidates: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for family, designs in family_designs.items():
        family_candidates[family] = {design for design in designs if design[-1] != "wt"}
    minimum_candidates = config["corrected_successor_rules"][
        "minimum_distinct_candidates_per_dense_source"
    ]
    eligible_families = {
        family
        for family in family_designs
        if len(family_sources.get(family, set())) == 1
        and len(family_candidates[family]) >= minimum_candidates
    }
    pairwise_only_families = {
        family
        for family in family_designs
        if len(family_sources.get(family, set())) == 1
        and len(family_candidates[family]) == 2
    }

    candidate_designs = {
        design for designs in family_candidates.values() for design in designs
    }
    replayable = 0
    unanchored = 0
    invalid_construct = 0
    zero_edit = 0
    edit_histogram: Counter[str] = Counter()
    for design in candidate_designs:
        family = design[:2]
        sources = family_sources.get(family, set())
        candidates = design_constructs[design]
        if len(sources) != 1:
            unanchored += 1
            continue
        if len(candidates) != 1:
            invalid_construct += 1
            continue
        source = next(iter(sources))
        candidate = next(iter(candidates))
        if len(source) != len(candidate) or len(source) != 164:
            invalid_construct += 1
            continue
        edit_count = sum(left != right for left, right in zip(source, candidate))
        if edit_count == 0:
            zero_edit += 1
            continue
        replayable += 1
        edit_histogram[_edit_bin(edit_count)] += 1

    mismatch_design_count = 0
    declared_total = 0
    for design, observed in design_observed.items():
        values = design_declared[design]
        if len(values) != 1:
            mismatch_design_count += 1
            continue
        declared = next(iter(values))
        declared_total += declared
        if declared != observed:
            mismatch_design_count += 1

    eligible_list = sorted(eligible_families)
    source_constructs = [next(iter(family_sources[family])) for family in eligible_list]
    parent = list(range(len(eligible_list)))
    near_parent = list(range(len(eligible_list)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    def near_find(index: int) -> int:
        while near_parent[index] != index:
            near_parent[index] = near_parent[near_parent[index]]
            index = near_parent[index]
        return index

    def near_union(left: int, right: int) -> None:
        left_root = near_find(left)
        right_root = near_find(right)
        if left_root != right_root:
            near_parent[right_root] = left_root

    for left in range(len(eligible_list)):
        for right in range(left + 1, len(eligible_list)):
            if eligible_list[left][0] == eligible_list[right][0]:
                union(left, right)

    length = config["split_and_power_policy"]["source_construct_length"]
    identity = config["split_and_power_policy"]["near_duplicate_identity_threshold"]
    maximum_distance = math.floor(length * (1.0 - identity))
    minimum_pairwise_distance: int | None = None
    near_duplicate_pair_count = 0
    for left in range(len(source_constructs)):
        for right in range(left + 1, len(source_constructs)):
            distance = sum(
                first != second
                for first, second in zip(source_constructs[left], source_constructs[right])
            )
            if minimum_pairwise_distance is None or distance < minimum_pairwise_distance:
                minimum_pairwise_distance = distance
            if distance <= maximum_distance:
                near_duplicate_pair_count += 1
                near_union(left, right)
                union(left, right)
    near_duplicate_components = len(
        {near_find(index) for index in range(len(eligible_list))}
    )
    effective_components = len({find(index) for index in range(len(eligible_list))})

    return {
        "_by_key": by_key,
        "_eligible_families": eligible_families,
        "_candidate_designs": candidate_designs,
        "member_count": len(records),
        "unique_member_key_count": len(by_key),
        "duplicate_member_key_count": duplicate_key_count,
        "design_count": len(design_constructs),
        "source_family_count": len(family_designs),
        "unique_source_anchor_family_count": sum(
            len(family_sources.get(family, set())) == 1 for family in family_designs
        ),
        "missing_source_anchor_family_count": sum(
            len(family_sources.get(family, set())) == 0 for family in family_designs
        ),
        "ambiguous_source_anchor_family_count": sum(
            len(family_sources.get(family, set())) > 1 for family in family_designs
        ),
        "eligible_dense_source_family_count": len(eligible_families),
        "pairwise_only_excluded_family_count": len(pairwise_only_families),
        "candidate_count_per_family_histogram": dict(
            sorted(
                Counter(
                    _small_count_bin(len(family_candidates[family]))
                    for family in family_designs
                ).items()
            )
        ),
        "candidate_design_count": len(candidate_designs),
        "sequence_diff_replayable_candidate_count": replayable,
        "source_unanchored_candidate_count": unanchored,
        "invalid_construct_candidate_count": invalid_construct,
        "zero_edit_candidate_count": zero_edit,
        "edit_count_histogram": dict(sorted(edit_histogram.items())),
        "declared_member_total": declared_total,
        "observed_member_total": len(records),
        "declared_multiplicity_mismatch_design_count": mismatch_design_count,
        "minimum_pairwise_source_hamming_distance": minimum_pairwise_distance,
        "near_duplicate_distance_threshold": maximum_distance,
        "near_duplicate_pair_count": near_duplicate_pair_count,
        "near_duplicate_component_count": near_duplicate_components,
        "post_gene_and_near_duplicate_effective_source_group_count": effective_components,
    }


def _endpoint_formula(total: int, distal: int, proximal: int, reported: str) -> tuple[bool, str]:
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


def _analyse_mpra(
    payload: bytes,
    table: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    by_key = table["_by_key"]
    eligible_families = table["_eligible_families"]
    candidate_designs = table["_candidate_designs"]
    context_index = {
        (sample, distal): index
        for index, (sample, distal) in enumerate(
            (sample, distal)
            for sample in EXPECTED_SAMPLE_FIELDS
            for distal in EXPECTED_DISTAL_CONTEXTS
        )
    }
    complete_mask = (1 << len(context_index)) - 1
    context_masks: dict[str, int] = defaultdict(int)
    context_counts: Counter[str] = Counter()
    seen_keys: set[str] = set()
    seen_samples: set[str] = set()
    seen_perturbations: set[str] = set()
    seen_distal_contexts: set[str] = set()
    endpoint_classes: Counter[str] = Counter()
    unmatched = 0
    join_mismatch = 0
    context_mismatch = 0
    context_duplicate = 0
    formula_mismatch = 0
    row_count = 0
    pooled: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0, 0])

    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                header = handle.readline().rstrip("\r\n")
                if header != " ".join(MPRA_HEADER):
                    raise PublicAssetError("official MPRA header differs")
                for line in handle:
                    row_count += 1
                    values = line.split()
                    if len(values) != len(MPRA_HEADER):
                        raise PublicAssetError("official MPRA row width differs")
                    row = dict(zip(MPRA_HEADER, values))
                    key = row["barcode"]
                    record = by_key.get(key)
                    if record is None:
                        unmatched += 1
                    else:
                        seen_keys.add(key)
                        if (
                            row["gene_id"] != record["gene_id"]
                            or row["pas_id"] != record["pas_id"]
                            or row["experiment"] != record["experiment"]
                            or row["n_bc"] != record["n_bc"]
                            or row["aim"] != record["type"]
                        ):
                            join_mismatch += 1

                    sample = row["sample"]
                    replicate = row["replicate"]
                    perturbation = row["perturbation"]
                    distal_context = row["distal_site"]
                    seen_samples.add(sample)
                    seen_perturbations.add(perturbation)
                    seen_distal_contexts.add(distal_context)
                    expected_sample = EXPECTED_SAMPLE_FIELDS.get(sample)
                    bit_index = context_index.get((sample, distal_context))
                    if (
                        expected_sample != (replicate, perturbation)
                        or bit_index is None
                    ):
                        context_mismatch += 1
                    elif record is not None:
                        bit = 1 << bit_index
                        if context_masks[key] & bit:
                            context_duplicate += 1
                        context_masks[key] |= bit
                        context_counts[key] += 1

                    _positive_integer(row["n_bc"], "official MPRA n_bc")
                    total = _nonnegative_integer(row["total"], "official MPRA total")
                    distal = _nonnegative_integer(row["distal"], "official MPRA distal")
                    proximal = _nonnegative_integer(row["proximal"], "official MPRA proximal")
                    matches, endpoint_class = _endpoint_formula(
                        total, distal, proximal, row["log_odds"]
                    )
                    endpoint_classes[endpoint_class] += 1
                    if not matches:
                        formula_mismatch += 1

                    if record is not None:
                        family = (record["gene_id"], record["pas_id"])
                        design = family + (record["experiment"],)
                        if family in eligible_families and design in candidate_designs:
                            pool_key = design + (perturbation, distal_context, replicate)
                            pooled[pool_key][0] += total
                            pooled[pool_key][1] += distal
                            pooled[pool_key][2] += proximal
    except PublicAssetError:
        raise
    except (OSError, UnicodeError, gzip.BadGzipFile, EOFError) as exc:
        raise PublicAssetError("official MPRA gzip table cannot be parsed") from exc

    expected_rows = config["official_asset_contract"]["processed_mpra"][
        "exact_data_row_count"
    ]
    if row_count != expected_rows:
        raise PublicAssetError("official MPRA data-row count differs")
    incomplete_context = sum(
        context_masks.get(key, 0) != complete_mask
        or context_counts.get(key, 0) != len(context_index)
        for key in by_key
    )

    by_endpoint: dict[tuple[Any, ...], dict[str, tuple[int, int, int]]] = defaultdict(dict)
    for key, counts in pooled.items():
        by_endpoint[key[:-1]][key[-1]] = tuple(counts)
    paired_endpoint_groups = 0
    both_replicates_publisher_qc_finite = 0
    independently_derivable_valid_se_groups = 0
    endpoint_groups_with_censored_or_low_umi_replicate = 0
    minimum_umi = config["publisher_facts"]["publisher_qc_minimum_total_umi"]
    for replicates in by_endpoint.values():
        if set(replicates) != {"rep1", "rep2"}:
            endpoint_groups_with_censored_or_low_umi_replicate += 1
            continue
        paired_endpoint_groups += 1
        values: list[float] = []
        eligible = True
        for total, distal, proximal in replicates.values():
            if total < minimum_umi or distal == 0 or proximal == 0:
                eligible = False
                break
            values.append(math.log2(proximal / distal))
        if not eligible:
            endpoint_groups_with_censored_or_low_umi_replicate += 1
            continue
        both_replicates_publisher_qc_finite += 1
        mean = sum(values) / 2.0
        sample_variance = sum((value - mean) ** 2 for value in values)
        standard_error = math.sqrt(sample_variance) / math.sqrt(2.0)
        if math.isfinite(standard_error) and standard_error >= 0.0:
            independently_derivable_valid_se_groups += 1

    return {
        "row_count": row_count,
        "distinct_joined_member_count": len(seen_keys),
        "unmatched_processed_row_count": unmatched,
        "unseen_publisher_member_count": len(by_key) - len(seen_keys),
        "join_crosscheck_mismatch_row_count": join_mismatch,
        "sample_count": len(seen_samples),
        "perturbation_count": len(seen_perturbations),
        "distal_reporter_context_count": len(seen_distal_contexts),
        "context_label_mismatch_row_count": context_mismatch,
        "context_duplicate_row_count": context_duplicate,
        "complete_context_member_count": len(by_key) - incomplete_context,
        "incomplete_context_member_count": incomplete_context,
        "formula_mismatch_row_count": formula_mismatch,
        "endpoint_value_class_counts": dict(sorted(endpoint_classes.items())),
        "paired_biological_replicate_endpoint_group_count": paired_endpoint_groups,
        "both_replicates_publisher_qc_finite_endpoint_group_count": (
            both_replicates_publisher_qc_finite
        ),
        "independently_derivable_valid_se_group_count": (
            independently_derivable_valid_se_groups
        ),
        "endpoint_group_with_censored_or_low_umi_replicate_count": (
            endpoint_groups_with_censored_or_low_umi_replicate
        ),
    }


def fisher_power(n: int, rho: float, alpha: float) -> float:
    if n <= 3:
        return 0.0
    null_standard_error = 1.0 / math.sqrt(n - 3.0)
    alternative_standard_error = (
        math.sqrt(1.0 + rho**2 / 2.0) * null_standard_error
    )
    alternative = math.atanh(rho)
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0) * null_standard_error
    normal = NormalDist()
    return 1.0 - normal.cdf(
        (critical - alternative) / alternative_standard_error
    ) + normal.cdf(
        (-critical - alternative) / alternative_standard_error
    )


def fisher_ci_width(n: int, rho: float, confidence: float) -> float:
    if n <= 3:
        return 2.0
    critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    center = math.atanh(rho)
    alternative_standard_error = math.sqrt(1.0 + rho**2 / 2.0) / math.sqrt(
        n - 3.0
    )
    radius = critical * alternative_standard_error
    return math.tanh(center + radius) - math.tanh(center - radius)


def inspect_official_public_assets(
    config: Mapping[str, Any], processed_mpra: Path, publisher_table_s5: Path
) -> dict[str, Any]:
    validate_protocol(config)
    assets = config["official_asset_contract"]
    mpra_payload = _read_bound_public_asset(processed_mpra, assets["processed_mpra"])
    table_payload = _read_bound_public_asset(
        publisher_table_s5, assets["publisher_table_s5"]
    )
    records = _parse_table_s5(table_payload, assets["publisher_table_s5"])
    table = _analyse_table(records, config)
    mpra = _analyse_mpra(mpra_payload, table, config)

    nonfinite = sum(
        count
        for value_class, count in mpra["endpoint_value_class_counts"].items()
        if value_class != "FINITE"
    )
    observation = {
        "source_mode": "LOCAL_DEC027_AUTHORIZED_ORDINARY_PUBLIC_AGGREGATE_RECOMPUTE_NOT_PRODUCTION_BINDING",
        "verified_asset_identity_count": 2,
        "asset_schema_geometry": {
            "publisher_library_member_count": table["member_count"],
            "publisher_unique_member_key_count": table["unique_member_key_count"],
            "publisher_duplicate_member_key_count": table["duplicate_member_key_count"],
            "publisher_design_count": table["design_count"],
            "processed_measurement_row_count": mpra["row_count"],
            "processed_distinct_joined_member_count": mpra[
                "distinct_joined_member_count"
            ],
            "processed_unmatched_row_count": mpra["unmatched_processed_row_count"],
            "publisher_unseen_member_count": mpra["unseen_publisher_member_count"],
            "join_crosscheck_mismatch_row_count": mpra[
                "join_crosscheck_mismatch_row_count"
            ],
            "complete_context_member_count": mpra["complete_context_member_count"],
            "incomplete_context_member_count": mpra["incomplete_context_member_count"],
            "context_duplicate_row_count": mpra["context_duplicate_row_count"],
            "declared_member_total": table["declared_member_total"],
            "observed_member_total": table["observed_member_total"],
            "declared_multiplicity_mismatch_design_count": table[
                "declared_multiplicity_mismatch_design_count"
            ],
        },
        "source_family_geometry": {
            "candidate_source_family_count": table["source_family_count"],
            "unique_source_anchor_family_count": table[
                "unique_source_anchor_family_count"
            ],
            "missing_source_anchor_family_count": table[
                "missing_source_anchor_family_count"
            ],
            "ambiguous_source_anchor_family_count": table[
                "ambiguous_source_anchor_family_count"
            ],
            "eligible_dense_source_family_count": table[
                "eligible_dense_source_family_count"
            ],
            "pairwise_only_excluded_family_count": table[
                "pairwise_only_excluded_family_count"
            ],
            "candidate_count_per_family_histogram": table[
                "candidate_count_per_family_histogram"
            ],
            "candidate_design_count": table["candidate_design_count"],
        },
        "legal_substitution_replay": {
            "candidate_design_count": table["candidate_design_count"],
            "sequence_diff_replayable_candidate_count": table[
                "sequence_diff_replayable_candidate_count"
            ],
            "source_unanchored_candidate_count": table[
                "source_unanchored_candidate_count"
            ],
            "invalid_construct_candidate_count": table[
                "invalid_construct_candidate_count"
            ],
            "zero_edit_candidate_count": table["zero_edit_candidate_count"],
            "edit_count_histogram": table["edit_count_histogram"],
            "row_order_inference_count": 0,
        },
        "assay_endpoint_replicate_geometry": {
            "sample_count": mpra["sample_count"],
            "perturbation_count": mpra["perturbation_count"],
            "distal_reporter_context_count": mpra[
                "distal_reporter_context_count"
            ],
            "sample_context_label_mismatch_row_count": mpra[
                "context_label_mismatch_row_count"
            ],
            "count_and_endpoint_formula_mismatch_row_count": mpra[
                "formula_mismatch_row_count"
            ],
            "endpoint_value_class_counts": mpra["endpoint_value_class_counts"],
            "nonfinite_or_undefined_endpoint_row_count": nonfinite,
            "publisher_asserted_biological_replicate_count": config[
                "publisher_facts"
            ]["biological_replicate_count_per_genetic_condition"],
            "publisher_reported_standard_error_field_present": config[
                "publisher_facts"
            ]["publisher_reports_row_standard_error"],
            "paired_biological_replicate_endpoint_group_count": mpra[
                "paired_biological_replicate_endpoint_group_count"
            ],
            "both_replicates_publisher_qc_finite_endpoint_group_count": mpra[
                "both_replicates_publisher_qc_finite_endpoint_group_count"
            ],
            "independently_derivable_valid_se_group_count": mpra[
                "independently_derivable_valid_se_group_count"
            ],
            "endpoint_group_with_censored_or_low_umi_replicate_count": mpra[
                "endpoint_group_with_censored_or_low_umi_replicate_count"
            ],
            "row_endpoint_effect_or_standard_error_reported_count": 0,
            "replicate_identifier_reported_count": 0,
        },
        "split_and_effective_n_geometry": {
            "outcome_blind_group_key_present": True,
            "minimum_pairwise_source_hamming_distance": table[
                "minimum_pairwise_source_hamming_distance"
            ],
            "near_duplicate_distance_threshold": table[
                "near_duplicate_distance_threshold"
            ],
            "near_duplicate_pair_count": table["near_duplicate_pair_count"],
            "near_duplicate_component_count": table[
                "near_duplicate_component_count"
            ],
            "post_gene_and_near_duplicate_effective_source_group_count": table[
                "post_gene_and_near_duplicate_effective_source_group_count"
            ],
            "split_assignment_output_count": 0,
        },
    }
    validate_aggregate_only(observation)
    return observation


def _gate(
    gate_id: str, status: str, reason_code: str, aggregate_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    if status not in {PASS, BLOCKED, FAIL}:
        raise CandidateContractError(f"invalid gate status for {gate_id}")
    return {
        "gate_id": gate_id,
        "status": status,
        "reason_code": reason_code,
        "aggregate_evidence": dict(aggregate_evidence),
    }


def _expected_geometry_projection(observation: Mapping[str, Any]) -> dict[str, Any]:
    schema = observation["asset_schema_geometry"]
    families = observation["source_family_geometry"]
    replay = observation["legal_substitution_replay"]
    endpoint = observation["assay_endpoint_replicate_geometry"]
    split = observation["split_and_effective_n_geometry"]
    return {
        "verified_asset_identity_count": observation["verified_asset_identity_count"],
        "publisher_library_member_count": schema["publisher_library_member_count"],
        "publisher_design_count": schema["publisher_design_count"],
        "processed_measurement_row_count": schema["processed_measurement_row_count"],
        "processed_distinct_joined_member_count": schema[
            "processed_distinct_joined_member_count"
        ],
        "processed_unmatched_row_count": schema["processed_unmatched_row_count"],
        "publisher_unseen_member_count": schema["publisher_unseen_member_count"],
        "join_crosscheck_mismatch_row_count": schema[
            "join_crosscheck_mismatch_row_count"
        ],
        "complete_context_member_count": schema["complete_context_member_count"],
        "candidate_source_family_count": families["candidate_source_family_count"],
        "eligible_dense_source_family_count": families[
            "eligible_dense_source_family_count"
        ],
        "pairwise_only_excluded_family_count": families[
            "pairwise_only_excluded_family_count"
        ],
        "candidate_design_count": families["candidate_design_count"],
        "sequence_diff_replayable_candidate_count": replay[
            "sequence_diff_replayable_candidate_count"
        ],
        "minimum_pairwise_source_hamming_distance": split[
            "minimum_pairwise_source_hamming_distance"
        ],
        "near_duplicate_component_count": split["near_duplicate_component_count"],
        "post_gene_group_effective_source_count": split[
            "post_gene_and_near_duplicate_effective_source_group_count"
        ],
        "nonfinite_or_undefined_endpoint_row_count": endpoint[
            "nonfinite_or_undefined_endpoint_row_count"
        ],
        "declared_multiplicity_mismatch_design_count": schema[
            "declared_multiplicity_mismatch_design_count"
        ],
    }


def evaluate_observation(
    config: Mapping[str, Any],
    observation: Mapping[str, Any],
    recorded_at: str = "NOT_RECORDED_NONPRODUCTION_VALIDATION",
) -> dict[str, Any]:
    validate_protocol(config)
    if not isinstance(recorded_at, str) or not recorded_at:
        raise CandidateContractError("recorded-at value is absent")
    validate_aggregate_only(observation)
    observed_projection = _expected_geometry_projection(observation)
    if observed_projection != config["expected_public_geometry"]:
        raise CandidateContractError(
            "public aggregate geometry differs from the DEC027 candidate snapshot"
        )

    schema = observation["asset_schema_geometry"]
    families = observation["source_family_geometry"]
    replay = observation["legal_substitution_replay"]
    endpoint = observation["assay_endpoint_replicate_geometry"]
    split = observation["split_and_effective_n_geometry"]
    publisher = config["publisher_facts"]
    power_policy = config["split_and_power_policy"]

    effective_n = split["post_gene_and_near_duplicate_effective_source_group_count"]
    power = fisher_power(
        effective_n,
        power_policy["alternative_spearman_rho"],
        power_policy["alpha_two_sided"],
    )
    ci_width = fisher_ci_width(
        effective_n,
        power_policy["alternative_spearman_rho"],
        power_policy["confidence_level"],
    )
    power_reachable = (
        effective_n >= power_policy["required_effective_n_reference"]
        and power >= power_policy["target_power_minimum"]
        and ci_width <= power_policy["maximum_full_ci_width"]
    )

    gates = [
        _gate(
            GATE_IDS[0],
            PASS,
            "TRUE_A2_DENSE_MEASURED_NEIGHBORHOOD_ELIGIBILITY_SELECTED_A1_EXCLUDED_BY_XOR_NOT_ROLE_ASSIGNMENT",
            {
                "a1_primary_role_evidence": False,
                "true_a2_primary_role_evidence": True,
                "maximum_roles_if_later_qualified": 1,
                "double_credit_allowed": False,
            },
        ),
        _gate(
            GATE_IDS[1],
            PASS,
            "TWO_ORDINARY_PUBLIC_PRIMARY_ASSETS_IDENTITY_ROLE_AND_PROVENANCE_CLOSED",
            {
                "verified_asset_identity_count": observation[
                    "verified_asset_identity_count"
                ],
                "processed_rows": schema["processed_measurement_row_count"],
                "publisher_rows": schema["publisher_library_member_count"],
            },
        ),
        _gate(
            GATE_IDS[2],
            PASS,
            "ONE_TWO_CANDIDATE_FAMILY_EXCLUDED_PAIRWISE_ONLY_OTHER_DENSE_FAMILIES_RETAINED",
            {
                "all_source_families": families["candidate_source_family_count"],
                "eligible_dense_source_families": families[
                    "eligible_dense_source_family_count"
                ],
                "pairwise_only_excluded_families": families[
                    "pairwise_only_excluded_family_count"
                ],
                "missing_source_anchor_families": families[
                    "missing_source_anchor_family_count"
                ],
                "ambiguous_source_anchor_families": families[
                    "ambiguous_source_anchor_family_count"
                ],
            },
        ),
        _gate(
            GATE_IDS[3],
            PASS,
            "PUBLISHER_APARENT_PERTURB_MPRA_SCOPE_IS_TANDEM_POLYA_NOT_INTRONIC_APA",
            {
                "publisher_model_scope": publisher["model_scope"],
                "publisher_mpra_locus_count": publisher["mpra_locus_count"],
            },
        ),
        _gate(
            GATE_IDS[4],
            PASS,
            "ALL_CANDIDATE_DESIGNS_REPLAY_AS_NONZERO_EQUAL_LENGTH_SUBSTITUTIONS_WITH_PUBLISHER_DESIGN_SEMANTICS",
            {
                "candidate_design_count": replay["candidate_design_count"],
                "replayable_candidate_design_count": replay[
                    "sequence_diff_replayable_candidate_count"
                ],
                "source_unanchored_candidate_count": replay[
                    "source_unanchored_candidate_count"
                ],
                "invalid_construct_candidate_count": replay[
                    "invalid_construct_candidate_count"
                ],
                "zero_edit_candidate_count": replay["zero_edit_candidate_count"],
                "row_order_inference_count": replay["row_order_inference_count"],
            },
        ),
        _gate(
            GATE_IDS[5],
            BLOCKED,
            "ENDPOINT_FORMULA_DIRECTION_AND_SCALE_CLOSED_BUT_EXACT_PREFROZEN_CENSOR_UNIVERSE_REPLAY_NOT_CLOSED",
            {
                "formula_mismatch_rows": endpoint[
                    "count_and_endpoint_formula_mismatch_row_count"
                ],
                "nonfinite_or_undefined_rows": endpoint[
                    "nonfinite_or_undefined_endpoint_row_count"
                ],
                "publisher_censor_policy_present": True,
                "exact_publisher_pooling_and_censor_replay_closed": publisher[
                    "exact_publisher_pooling_and_censor_replay_closed"
                ],
            },
        ),
        _gate(
            GATE_IDS[6],
            BLOCKED,
            "TWO_BIOLOGICAL_REPLICATES_PUBLISHER_CLOSED_VALID_SE_GROUPING_REMAINS_INDEPENDENT_AUDIT_BLOCKER",
            {
                "publisher_biological_replicates": endpoint[
                    "publisher_asserted_biological_replicate_count"
                ],
                "paired_replicate_endpoint_groups": endpoint[
                    "paired_biological_replicate_endpoint_group_count"
                ],
                "candidate_valid_se_group_count": endpoint[
                    "independently_derivable_valid_se_group_count"
                ],
                "publisher_reported_se_field_present": endpoint[
                    "publisher_reported_standard_error_field_present"
                ],
                "exact_valid_se_audit_closed": False,
            },
        ),
        _gate(
            GATE_IDS[7],
            PASS,
            "ASSET_DIMENSIONS_JOIN_AND_CONTEXT_COVERAGE_CLOSED_MULTIPLICITY_DISCREPANCIES_QUARANTINED_TO_QC",
            {
                "joined_members": schema["processed_distinct_joined_member_count"],
                "unmatched_rows": schema["processed_unmatched_row_count"],
                "unseen_publisher_members": schema["publisher_unseen_member_count"],
                "complete_context_members": schema["complete_context_member_count"],
                "declared_multiplicity_mismatch_designs": schema[
                    "declared_multiplicity_mismatch_design_count"
                ],
                "multiplicity_discrepancy_is_asset_coverage_failure": False,
            },
        ),
        _gate(
            GATE_IDS[8],
            FAIL,
            "PUBLISHER_CENSOR_POLICY_EXISTS_BUT_LOCUS_UNIVERSE_WAS_MODEL_AND_MEASURED_RESPONSE_SELECTED",
            {
                "publisher_qc_policy_present": True,
                "exact_censor_replay_closed": publisher[
                    "exact_publisher_pooling_and_censor_replay_closed"
                ],
                "source_locus_selection_uses_model_and_measured_response": publisher[
                    "source_locus_selection_uses_model_prediction_and_measured_perturbation_response"
                ],
                "nonfinite_or_undefined_rows": endpoint[
                    "nonfinite_or_undefined_endpoint_row_count"
                ],
            },
        ),
        _gate(
            GATE_IDS[9],
            FAIL,
            "APARENT_PREDICTION_INTERPRETATION_AND_RESPONSE_SELECTED_LIBRARY_IS_NOT_AN_UNEXPOSED_PRIMARY_BENCHMARK",
            {
                "aparent_guided_candidate_design": publisher[
                    "candidate_design_uses_aparent_prediction_or_interpretation"
                ],
                "aparent_and_measured_response_guided_locus_selection": publisher[
                    "source_locus_selection_uses_model_prediction_and_measured_perturbation_response"
                ],
                "future_model_input_route_closed": False,
            },
        ),
        _gate(
            GATE_IDS[10],
            BLOCKED,
            "PUBLIC_DOWNLOAD_AND_ARTICLE_COPYRIGHT_NOTICE_PRESENT_BUT_DATASET_SPECIFIC_REUSE_LICENSE_ABSENT",
            {
                "ordinary_public_download": True,
                "article_copyright_notice": publisher["article_copyright_notice"],
                "dataset_specific_reuse_license_notice_present": publisher[
                    "dataset_specific_reuse_license_notice_present"
                ],
            },
        ),
        _gate(
            GATE_IDS[11],
            PASS,
            "OUTCOME_BLIND_GENE_SOURCE_AND_NEAR_DUPLICATE_COMPONENTS_SUPPORT_ZERO_LEAKAGE_SPLIT_READINESS_NO_ASSIGNMENT_EXECUTED",
            {
                "near_duplicate_distance_threshold": split[
                    "near_duplicate_distance_threshold"
                ],
                "minimum_pairwise_source_hamming_distance": split[
                    "minimum_pairwise_source_hamming_distance"
                ],
                "near_duplicate_pair_count": split["near_duplicate_pair_count"],
                "effective_source_groups": effective_n,
                "split_assignment_output_count": split[
                    "split_assignment_output_count"
                ],
            },
        ),
        _gate(
            GATE_IDS[12],
            PASS if power_reachable else FAIL,
            (
                "POST_DEDUP_EFFECTIVE_N_POWER_AND_FULL_CI_WIDTH_REACHABLE_AGGREGATE_PREFLIGHT_ONLY"
                if power_reachable
                else "POST_DEDUP_EFFECTIVE_N_OR_PREFROZEN_POWER_PRECISION_NOT_REACHABLE"
            ),
            {
                "analysis_unit": power_policy["analysis_unit"],
                "effective_source_group_n": effective_n,
                "required_effective_n": power_policy[
                    "required_effective_n_reference"
                ],
                "planning_power": power,
                "target_power": power_policy["target_power_minimum"],
                "planning_full_ci_width": ci_width,
                "maximum_full_ci_width": power_policy["maximum_full_ci_width"],
                "formal_qualification_power_run": False,
            },
        ),
    ]
    if [gate["gate_id"] for gate in gates] != list(GATE_IDS):
        raise CandidateContractError("gate output order differs")
    counts = Counter(gate["status"] for gate in gates)
    all_pass = counts == Counter({PASS: len(GATE_IDS)})
    result_status = (
        "ALL_THIRTEEN_PREFLIGHT_GATES_PASS_PROMOTION_REQUEST_ONLY"
        if all_pass
        else "STOP_CORRECTED_ROLE_ADJUDICATION_GATES_NOT_CLOSED"
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": "GSE269595",
        "project_id": "PRJNA1122592",
        "decision_id": "V3-DEC-027",
        "observed_at_utc": recorded_at,
        "source_mode": observation["source_mode"],
        "result_status": result_status,
        "all_required_gates_pass": all_pass,
        "mutually_exclusive_role_disposition": {
            "a1_geometry_eligible": False,
            "true_a2_dense_measured_neighborhood_geometry_eligible": True,
            "recommended_role_if_later_independently_qualified": "TRUE_A2_ONLY",
            "role_assigned": False,
            "double_credit_allowed": False,
        },
        "corrected_predecessor_dispositions": {
            "single_two_candidate_family_fails_all_dense_families": False,
            "publisher_two_biological_replicates_hardcoded_absent": False,
            "five_multiplicity_discrepancies_fail_complete_asset_coverage": False,
            "all_replayable_sequence_diffs_fail_only_for_missing_separate_annotation_column": False,
            "nonfinite_endpoint_treated_as_zero": False,
        },
        "aggregate_geometry": copy.deepcopy(observation),
        "post_dedup_power_reachability": {
            "verdict": (
                "REACHABLE_FOR_PREFLIGHT_INFORMATION_GEOMETRY_NOT_FORMAL_QUALIFICATION"
                if power_reachable
                else "NOT_REACHABLE"
            ),
            "effective_source_group_n": effective_n,
            "required_effective_n": power_policy["required_effective_n_reference"],
            "planning_power": power,
            "target_power": power_policy["target_power_minimum"],
            "planning_full_ci_width": ci_width,
            "maximum_full_ci_width": power_policy["maximum_full_ci_width"],
            "formal_qualification_power_run": False,
        },
        "gates": gates,
        "gate_counts": {
            PASS: counts[PASS],
            BLOCKED: counts[BLOCKED],
            FAIL: counts[FAIL],
            "TOTAL": len(gates),
        },
        "remaining_blockers": [
            "EXACT_PREFROZEN_CENSOR_AND_ENDPOINT_UNIVERSE_REPLAY",
            "VALID_SE_GROUPING_AND_REPLAY",
            "OUTCOME_AND_MODEL_SELECTED_UNIVERSE_FOR_PRIMARY_CLAIM",
            "APARENT_PRIOR_EXPOSURE_AND_FUTURE_MODEL_INPUT_ROUTE",
            "DATASET_SPECIFIC_REUSE_LICENSE_NOTICE",
        ],
        "internal_access_attestation": {
            "ordinary_public_asset_read_count": 2,
            "private_or_sealed_asset_read_count": 0,
            "raw_fastq_or_sra_member_payload_read_count": 0,
            "persistent_member_level_intermediate_count": 0,
            "member_identifier_sequence_row_effect_se_or_split_output_count": 0,
            "split_assignment_execution_count": 0,
            "training_run_count": 0,
            "gpu_run_count": 0,
            "model_selection_count": 0,
        },
        "terminal_state": copy.deepcopy(config["terminal_state"]),
        "claim_boundary": {
            "role_eligibility_is_role_assignment": False,
            "all_gates_pass_automatically_qualifies_or_credits_dataset": False,
            "separate_promotion_authority_required": True,
            "scientific_claim_status": "NOT_ESTABLISHED",
        },
    }
    validate_aggregate_only(report)
    return report


def validate_aggregate_only(value: Any) -> None:
    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).lower() in FORBIDDEN_OUTPUT_KEYS:
                    raise CandidateContractError(f"forbidden member payload key: {key}")
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, str) and re.fullmatch(r"[ACGT]{20,}", item.upper()):
            raise CandidateContractError("sequence-like member payload is forbidden")

    walk(value)


def _write_report(output_dir: Path, report: Mapping[str, Any]) -> Path:
    validate_aggregate_only(report)
    try:
        payload = (
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OutputError("aggregate report is not finite JSON") from exc

    path = output_dir / REPORT_FILENAME
    directory_created = False
    temporary_path: Path | None = None
    try:
        if output_dir.exists():
            if not output_dir.is_dir():
                raise OutputError("output path is not a directory")
            entries = list(output_dir.iterdir())
            if entries:
                if len(entries) == 1 and entries[0] == path:
                    if path.read_bytes() == payload:
                        return path
                    raise OutputError("different report already exists")
                raise OutputError("output directory has an unexpected entry")
        else:
            output_dir.mkdir(parents=False, exist_ok=False)
            directory_created = True
            _fsync_directory(output_dir.parent)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{REPORT_FILENAME}.", suffix=".tmp", dir=output_dir
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            if path.read_bytes() == payload:
                temporary_path.unlink()
                temporary_path = None
                return path
            raise OutputError("different report appeared during publication") from exc
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(output_dir)
        if list(output_dir.iterdir()) != [path]:
            raise OutputError("single fixed report contract was violated")
        return path
    except OutputError:
        raise
    except OSError as exc:
        raise OutputError("atomic no-replace publication failed") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        if directory_created and output_dir.exists():
            try:
                if not any(output_dir.iterdir()):
                    output_dir.rmdir()
            except OSError:
                pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execute_production(
    *,
    config_path: Path,
    asset_dir: Path,
    output_dir: Path,
    recorded_at: str,
) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    validate_protocol(config)
    _require_production_bindings(config)
    binding_audit = _audit_repository_bindings(
        config, config_path, PRODUCTION_REPO_ROOT
    )
    assets = config["official_asset_contract"]
    processed_mpra = asset_dir / assets["processed_mpra"]["filename"]
    publisher_table_s5 = asset_dir / assets["publisher_table_s5"]["filename"]
    observation = inspect_official_public_assets(
        config, processed_mpra, publisher_table_s5
    )
    observation = copy.deepcopy(observation)
    observation["source_mode"] = (
        "BOUND_DEC027_PRODUCTION_ORDINARY_PUBLIC_AGGREGATE_RECOMPUTE"
    )
    report = evaluate_observation(config, observation, recorded_at)
    report["production_binding_audit"] = binding_audit
    validate_aggregate_only(report)
    output_path = _write_report(output_dir, report)
    return output_path, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recorded-at", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output_path, report = execute_production(
            config_path=args.config,
            asset_dir=args.asset_dir,
            output_dir=args.output_dir,
            recorded_at=args.recorded_at,
        )
    except (
        CandidateContractError,
        BindingNotFrozen,
        PublicAssetError,
        OutputError,
    ) as exc:
        print(
            json.dumps({"status": "STOP", "reason": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "output": str(output_path),
                "result_status": report["result_status"],
                "gate_counts": report["gate_counts"],
                "contribution": report["terminal_state"]["contribution"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
