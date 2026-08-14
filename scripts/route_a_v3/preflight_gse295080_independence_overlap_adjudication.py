#!/usr/bin/env python3
"""Bound aggregate-only GSE295080 independence/overlap adjudication.

This exact3 route does not perform row-level qualification.  Production has
one built-in reader set for the five frozen ordinary-public assets and remains
fail-closed until DEC027 authority, full runtime history, the complete frozen
GSE217518 and ENCSR854RUF histories, all three append-only predecessor histories,
and the GSE295080 frozen-I1/dynamic-I2/config-only-B2 lifecycle are bound.  The
complete direct-parent Git chain is audited before asset or output I/O.  It
publishes one aggregate JSON atomically and cannot qualify or credit GSE295080.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "route_a_v3_gse295080_independence_overlap_adjudication.v1"
REPORT_SCHEMA_VERSION = (
    "route_a_v3_gse295080_independence_overlap_aggregate_record.v1"
)
PROTOCOL_ID = "GSE295080_DEC027_INDEPENDENCE_OVERLAP_ADJUDICATION_ONLY_V1"
DATASET_ID = "GSE295080"
DECISION_ID = "V3-DEC-027"
AUTHORITY_COMMIT = "3e0ad158a0b45b2f26ed82da3afe60667c712cd6"
AUTHORITY_PARENT = "b1ca33d852bad111ff31b4f60493d8c43c63d1a3"
RUNTIME_I1_COMMIT = "de40c58ab81fc06196be3bb9ffb5aa35d39c9d03"
RUNTIME_I2_COMMIT = "5d66e8dc83eb9966f7698ac0fc677f1b06af8ea6"
RUNTIME_B_COMMIT = "e60956cf59cbddc0406c5d116fb9714906db36e1"
BASE_EVENT = "A1-EVT-059"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
REPORT_FILENAME = "GSE295080_INDEPENDENCE_OVERLAP_AGGREGATE_PREFLIGHT_V1.json"
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_BRANCH = "routea-v3-a1-20260810"
PRODUCTION_UPSTREAM = f"origin/{PRODUCTION_BRANCH}"

AUTHORITY_EXACT12 = (
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
GSE269_CONFIG_PATH = "configs/route_a_v3_gse269595_corrected_role_adjudication_successor_candidate_v1.json"
GSE269_SCRIPT_PATH = "scripts/route_a_v3/preflight_gse269595_corrected_role_adjudication_successor_candidate.py"
GSE269_TEST_PATH = "tests/route_a_v3/test_preflight_gse269595_corrected_role_adjudication_successor_candidate.py"
GSE269_EXACT3 = (GSE269_CONFIG_PATH, GSE269_SCRIPT_PATH, GSE269_TEST_PATH)

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
GSE217_FINAL_B = str(GSE217_HISTORY[-1]["commit"])

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
ENCSR_FINAL_B = str(ENCSR_HISTORY[-1]["commit"])

CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse295080_independence_overlap_adjudication_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/preflight_gse295080_independence_overlap_adjudication.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_preflight_gse295080_independence_overlap_adjudication.py"
)
EXACT3 = (CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH)
FROZEN_I1_COMMIT = "d422e27fe31ff66d8a9dd1faa8b0aef1d2cf352d"
FROZEN_I1_PARENT = "19ca49229c9ff2814bad2c58b8b84be14624b7ea"
FROZEN_I1_BLOBS = {
    CONFIG_REPO_PATH: "bab1148a045ddbf1e6f465f9dcb096c460f75e9929ef865375ccf51e32fdb6e5",
    SCRIPT_REPO_PATH: "d1db3a0e267aa6bbdb098ea665a8539253b344616d6861c5ae379ee33b3099ad",
    TEST_REPO_PATH: "1bd5efa9cf6e685e9e939c74e0e465c51321f223e71002cdab43ce8a40d33de2",
}
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

GATE_IDS = (
    "PUBLIC_IDENTIFIER_ASSET_ROLE_AND_PROVENANCE_CLOSED",
    "SOURCE_REFERENCE_ALTERNATIVE_MAPPING_SCHEMA_GEOMETRY_CLOSED",
    "BIOLOGICAL_REPLICATE_LABEL_GEOMETRY_CLOSED",
    "CROSS_DATASET_SOURCE_FAMILY_LIBRARY_OVERLAP_CLOSED",
    "INDEPENDENT_STUDY_OR_REUSED_LIBRARY_BOUNDARY_CLOSED",
    "LICENSE_AND_REUSE_RIGHTS_CLOSED",
    "OUTCOME_BLIND_NEXT_AUTHORITY_DISPOSITION_CLOSED",
)
COMPARISON_STUDIES = ("GSE186455", "ENCSR854RUF", "GSE217518", "GSE232572")
NORMALIZED_CLASSES = {
    "PASS",
    "PARTIAL_OR_CONDITIONAL",
    "FAIL",
    "UNKNOWN_NOT_ASSERTED",
    "BLOCKED_OR_STOP",
}
EXPECTED_STATUS_COUNTS = {
    "PASS": 3,
    "PARTIAL_OR_CONDITIONAL": 1,
    "FAIL": 1,
    "UNKNOWN_NOT_ASSERTED": 1,
    "BLOCKED_OR_STOP": 1,
    "TOTAL": 7,
}
INPUT_KEYS = (
    "stability_table",
    "author_reference_fasta",
    "geo_family_soft",
    "geo_file_inventory",
    "gse186455_processed_archive",
)
PUBLIC_ASSET_IDENTITIES = {
    "isompra_stability.tsv": (
        2151085,
        "1c9980dc7b9206bba620ed35b8d6efbe935f3ce9a15b19e97b7b0db917f93ffa",
    ),
    "isompra_ref.fasta": (
        255250,
        "e45db4e5169297bae8339ed4e73148f1e7e45309c17b5b1185e2910a111637df",
    ),
    "GSE295080_family.soft.gz": (
        21715,
        "72fd0496ad39dc66de5da8403369da202fc3679db6ae70d08d11b33a2bdd1a96",
    ),
    "GSE295080_filelist.txt": (
        22691,
        "2a7e22c9bc4f7e02888011b0fcc38109edbe68b6d7d7d345a16f87ec37a84f79",
    ),
    "GSE186455_RAW.tar": (
        2416640,
        "57685b8ce845629b2bc37603f94cb2c7baf452d4e798fba84325d12f80e67223",
    ),
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

FORBIDDEN_EXACT_REPORT_KEYS = {
    "member_id",
    "member_name",
    "row_id",
    "record_id",
    "element",
    "seqname",
    "sequence",
    "barcode",
    "sample_id",
    "replicate_id",
    "source_sequence",
    "candidate_sequence",
    "row_effect",
    "row_pvalue",
    "row_standard_error",
    "split_assignment",
}


class PreflightError(RuntimeError):
    """Base error with aggregate-safe messages only."""


class ProtocolError(PreflightError):
    """The reviewed candidate protocol is inconsistent."""


class ActivationBlocked(PreflightError):
    """Production binding is incomplete."""


class AssetError(PreflightError):
    """An ordinary-public input differs from the frozen contract."""


class ReplayError(PreflightError):
    """Aggregate replay geometry differs from the frozen candidate."""


class OutputError(PreflightError):
    """The single aggregate report cannot be written safely."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], *, label: str) -> None:
    if set(value) != set(expected):
        raise ProtocolError(f"{label} fields differ from the exact schema")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant {token}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError("protocol is not strict readable JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("protocol root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AssetError("ordinary-public asset is not readable") from exc
    return digest.hexdigest()


def _validate_sha_map(value: Any, paths: Sequence[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(paths):
        raise ProtocolError(f"{label} blob path closure differs")
    if any(not HEX64.fullmatch(str(digest)) for digest in value.values()):
        raise ProtocolError(f"{label} blob SHA-256 differs")


def _validate_authority_group(group: Mapping[str, Any]) -> None:
    if dict(group) != {
        "status": BOUND,
        "authority_commit": AUTHORITY_COMMIT,
        "authority_expected_parent": AUTHORITY_PARENT,
        "authority_exact_changed_paths": list(AUTHORITY_EXACT12),
        "authority_blob_sha256_by_path": AUTHORITY_BLOBS,
    }:
        raise ProtocolError("DEC027 authority exact12 binding differs")


def _validate_runtime_group(group: Mapping[str, Any]) -> None:
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
    if dict(group) != expected:
        raise ProtocolError("DEC027 runtime I1/I2/B2 binding differs")


def _validate_frozen_history(
    group: Mapping[str, Any],
    *,
    name: str,
    history: Sequence[Mapping[str, Any]],
) -> None:
    expected = {
        "status": BOUND,
        "append_only_history": [dict(step) for step in history],
        "terminal_binding_commit": history[-1]["commit"],
    }
    if dict(group) != expected:
        raise ProtocolError(f"{name} frozen append-only history differs")


def _validate_append_only_history(
    history: Any,
    *,
    name: str,
    exact3: Sequence[str],
    config_path: str,
    expected_parent: str,
) -> str:
    if not isinstance(history, list) or len(history) < 2:
        raise ProtocolError(f"{name} append-only history is incomplete")
    parent = expected_parent
    next_implementation = 1
    latest_unbound_implementation: int | None = None
    for index, raw_step in enumerate(history):
        if not isinstance(raw_step, Mapping) or set(raw_step) != {
            "step",
            "commit",
            "expected_parent",
            "exact_changed_paths",
            "blob_sha256_by_path",
        }:
            raise ProtocolError(f"{name} history step schema differs")
        step = raw_step
        label = step.get("step")
        match = re.fullmatch(r"([IB])([1-9][0-9]*)", str(label))
        if match is None:
            raise ProtocolError(f"{name} history step label differs")
        kind, number = match.group(1), int(match.group(2))
        if kind == "I":
            if number != next_implementation:
                raise ProtocolError(f"{name} implementation order differs")
            latest_unbound_implementation = number
            next_implementation += 1
            expected_paths = list(exact3)
        else:
            if latest_unbound_implementation != number:
                raise ProtocolError(f"{name} binding order differs")
            latest_unbound_implementation = None
            expected_paths = [config_path]
        commit = step.get("commit")
        if not HEX40.fullmatch(str(commit)) or step.get("expected_parent") != parent:
            raise ProtocolError(f"{name} history direct parent differs")
        if step.get("exact_changed_paths") != expected_paths:
            raise ProtocolError(f"{name} history changed paths differ")
        _validate_sha_map(
            step.get("blob_sha256_by_path"), exact3, f"{name} {label}"
        )
        parent = str(commit)
        if index == len(history) - 1 and kind != "B":
            raise ProtocolError(f"{name} history lacks terminal binding")
    return parent


def _future_predecessor_mode(
    group: Mapping[str, Any],
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
    if set(group) != expected_keys:
        raise ProtocolError(f"{name} binding group schema differs")
    if group.get("unknown_to_bound_fields") != list(FUTURE_PREDECESSOR_FIELDS):
        raise ProtocolError(f"{name} grouped fields differ")
    if group.get("implementation_exact_changed_paths") != list(exact3):
        raise ProtocolError(f"{name} implementation exact3 differs")
    if group.get("binding_exact_changed_paths") != [config_path]:
        raise ProtocolError(f"{name} binding path differs")
    dynamic = tuple(group.get(field) for field in FUTURE_PREDECESSOR_FIELDS)
    if group.get("status") == UNKNOWN:
        if dynamic != (UNKNOWN,) * len(FUTURE_PREDECESSOR_FIELDS):
            raise ProtocolError(f"{name} is partially bound")
        return UNKNOWN
    if group.get("status") != BOUND or predecessor_mode != BOUND:
        raise ProtocolError(f"{name} predecessor is not bound")
    if expected_parent is None:
        raise ProtocolError(f"{name} initial parent is unknown")
    terminal = _validate_append_only_history(
        group.get("append_only_history"),
        name=name,
        exact3=exact3,
        config_path=config_path,
        expected_parent=expected_parent,
    )
    if group.get("terminal_binding_commit") != terminal:
        raise ProtocolError(f"{name} terminal binding differs")
    return BOUND


def _own_mode(group: Mapping[str, Any], *, predecessor_mode: str) -> str:
    expected_keys = {
        "status",
        "frozen_i1_commit",
        "frozen_i1_expected_parent",
        "frozen_i1_exact_changed_paths",
        "frozen_i1_blob_sha256_by_path",
        "implementation_commit",
        "implementation_script_path",
        "implementation_script_sha256",
        "implementation_test_path",
        "implementation_test_sha256",
        "implementation_exact_changed_paths",
        "binding_exact_changed_paths",
        "unknown_to_bound_fields",
    }
    if set(group) != expected_keys:
        raise ProtocolError("own I1/I2 binding group schema differs")
    if group.get("frozen_i1_commit") != FROZEN_I1_COMMIT:
        raise ProtocolError("own frozen I1 commit differs")
    if group.get("frozen_i1_expected_parent") != FROZEN_I1_PARENT:
        raise ProtocolError("own frozen I1 parent differs")
    if group.get("frozen_i1_exact_changed_paths") != list(EXACT3):
        raise ProtocolError("own frozen I1 exact3 differs")
    if group.get("frozen_i1_blob_sha256_by_path") != FROZEN_I1_BLOBS:
        raise ProtocolError("own frozen I1 blobs differ")
    if group.get("unknown_to_bound_fields") != list(OWN_BINDING_FIELDS):
        raise ProtocolError("own four-scalar binding fields differ")
    if group.get("implementation_script_path") != SCRIPT_REPO_PATH:
        raise ProtocolError("own script path differs")
    if group.get("implementation_test_path") != TEST_REPO_PATH:
        raise ProtocolError("own test path differs")
    if group.get("implementation_exact_changed_paths") != list(EXACT3):
        raise ProtocolError("own implementation exact3 differs")
    if group.get("binding_exact_changed_paths") != [CONFIG_REPO_PATH]:
        raise ProtocolError("own binding path differs")
    dynamic = tuple(group.get(field) for field in OWN_BINDING_FIELDS)
    if group.get("status") == UNKNOWN:
        if dynamic != (UNKNOWN,) * len(OWN_BINDING_FIELDS):
            raise ProtocolError("own preflight group is partially bound")
        return UNKNOWN
    if group.get("status") != BOUND:
        raise ProtocolError("own preflight status differs")
    if predecessor_mode != BOUND:
        raise ProtocolError("own predecessor is not bound")
    if not HEX40.fullmatch(str(group.get("implementation_commit"))):
        raise ProtocolError("own I2 implementation commit differs")
    for field in ("implementation_script_sha256", "implementation_test_sha256"):
        if not HEX64.fullmatch(str(group.get(field))):
            raise ProtocolError(f"own I2 {field} differs")
    return BOUND


def _binding_modes(protocol: Mapping[str, Any]) -> dict[str, str]:
    binding = _mapping(protocol.get("implementation_binding"), label="implementation_binding")
    expected_keys = {
        "binding_scheme",
        "authority_group",
        "runtime_group",
        "gse217518_predecessor",
        "encsr854ruf_predecessor",
        "gse232572_predecessor",
        "gse113849_predecessor",
        "gse269595_predecessor",
        "own_preflight_group",
    }
    if set(binding) != expected_keys:
        raise ProtocolError("implementation binding group closure differs")
    if binding.get("binding_scheme") != (
        "DEC027_A_RUNTIME_I1_I2_B2_GSE217_I1_I2_B2_I3_B3_"
        "ENCSR_I1_I2_B2_I3_B3_I4_B4_FUTURE_GSE232_GSE113_GSE269_"
        "APPEND_ONLY_THEN_GSE295080_FROZEN_I1_DYNAMIC_I2_CONFIG_ONLY_B2"
    ):
        raise ProtocolError("binding scheme differs")
    for name in expected_keys - {"binding_scheme"}:
        _mapping(binding.get(name), label=name)
    _validate_authority_group(binding["authority_group"])
    _validate_runtime_group(binding["runtime_group"])
    _validate_frozen_history(
        binding["gse217518_predecessor"],
        name="GSE217518",
        history=GSE217_HISTORY,
    )
    _validate_frozen_history(
        binding["encsr854ruf_predecessor"],
        name="ENCSR854RUF",
        history=ENCSR_HISTORY,
    )
    gse217_mode = BOUND
    encsr_mode = BOUND
    gse232_mode = _future_predecessor_mode(
        binding["gse232572_predecessor"],
        name="GSE232572",
        exact3=GSE232_EXACT3,
        config_path=GSE232_CONFIG_PATH,
        expected_parent=ENCSR_FINAL_B,
        predecessor_mode=encsr_mode,
    )
    gse232_b = binding["gse232572_predecessor"].get("terminal_binding_commit")
    gse113_mode = _future_predecessor_mode(
        binding["gse113849_predecessor"],
        name="GSE113849",
        exact3=GSE113_EXACT3,
        config_path=GSE113_CONFIG_PATH,
        expected_parent=str(gse232_b) if gse232_mode == BOUND else None,
        predecessor_mode=gse232_mode,
    )
    gse113_b = binding["gse113849_predecessor"].get("terminal_binding_commit")
    gse269_mode = _future_predecessor_mode(
        binding["gse269595_predecessor"],
        name="GSE269595",
        exact3=GSE269_EXACT3,
        config_path=GSE269_CONFIG_PATH,
        expected_parent=str(gse113_b) if gse113_mode == BOUND else None,
        predecessor_mode=gse113_mode,
    )
    gse269_b = binding["gse269595_predecessor"].get("terminal_binding_commit")
    if gse269_mode == BOUND and gse269_b != FROZEN_I1_PARENT:
        raise ProtocolError("GSE295080 frozen I1 parent differs from GSE269595 B2")
    own_mode = _own_mode(
        binding["own_preflight_group"], predecessor_mode=gse269_mode
    )
    return {
        "gse217518_predecessor": gse217_mode,
        "encsr854ruf_predecessor": encsr_mode,
        "gse232572_predecessor": gse232_mode,
        "gse113849_predecessor": gse113_mode,
        "gse269595_predecessor": gse269_mode,
        "own_preflight_group": own_mode,
    }


def _count_gate_statuses(gates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = collections.Counter(str(gate["normalized_status"]) for gate in gates)
    return {
        "PASS": counts["PASS"],
        "PARTIAL_OR_CONDITIONAL": counts["PARTIAL_OR_CONDITIONAL"],
        "FAIL": counts["FAIL"],
        "UNKNOWN_NOT_ASSERTED": counts["UNKNOWN_NOT_ASSERTED"],
        "BLOCKED_OR_STOP": counts["BLOCKED_OR_STOP"],
        "TOTAL": len(gates),
    }


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    _exact_keys(
        protocol,
        (
            "schema_version",
            "protocol_id",
            "contract_id",
            "decision_id",
            "phase_id",
            "dataset_id",
            "project_id",
            "protocol_status",
            "fresh_baseline",
            "repository_authority",
            "implementation_binding",
            "production_activation_rule",
            "execution_boundary",
            "decision_boundary",
            "ordinary_public_source_chain",
            "ordinary_public_input_contract",
            "gate_contract",
            "expected_aggregate_replay",
            "frozen_gate_snapshot",
            "aggregate_output_contract",
            "scientific_state",
        ),
        label="protocol",
    )
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "decision_id": DECISION_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "project_id": "PRJNA1252772",
        "protocol_status": "DRAFT_PRODUCTION_HARDENED_NOT_ACTIVE_BINDINGS_UNKNOWN",
    }
    for key, expected in expected_scalars.items():
        if protocol.get(key) != expected:
            raise ProtocolError(f"{key} differs from the frozen value")

    baseline = _mapping(protocol.get("fresh_baseline"), label="fresh_baseline")
    if dict(baseline) != {
        "production_branch": PRODUCTION_BRANCH,
        "dec027_authority_commit": AUTHORITY_COMMIT,
        "dec027_authority_parent": AUTHORITY_PARENT,
        "pre_dec027_projection_event": "A1-EVT-058",
        "bound_runtime_event": BASE_EVENT,
        "current_projection_status": "SETTLED_RUNTIME_CURRENT_PROJECTION",
    }:
        raise ProtocolError("fresh DEC027/runtime baseline differs")

    repository = _mapping(
        protocol.get("repository_authority"), label="repository_authority"
    )
    if dict(repository) != {
        "production_repo_root": str(PRODUCTION_REPO_ROOT),
        "branch": PRODUCTION_BRANCH,
        "upstream_ref": PRODUCTION_UPSTREAM,
        "live_origin_head_required": True,
        "clean_worktree_and_index_required": True,
    }:
        raise ProtocolError("repository authority differs")

    _binding_modes(protocol)
    activation = _mapping(
        protocol.get("production_activation_rule"), label="production_activation_rule"
    )
    if dict(activation) != {
        "required_commit_chain": (
            "DEC027_A_TO_RUNTIME_I1_TO_RUNTIME_I2_TO_RUNTIME_B2_"
            "TO_GSE217518_I1_TO_GSE217518_I2_TO_GSE217518_B2_"
            "TO_GSE217518_I3_TO_GSE217518_B3_"
            "TO_ENCSR854RUF_I1_TO_ENCSR854RUF_I2_TO_ENCSR854RUF_B2_"
            "TO_ENCSR854RUF_I3_TO_ENCSR854RUF_B3_TO_ENCSR854RUF_I4_"
            "TO_ENCSR854RUF_B4_TO_GSE232572_APPEND_ONLY_I_B_HISTORY_"
            "TO_GSE113849_APPEND_ONLY_I_B_HISTORY_"
            "TO_GSE269595_APPEND_ONLY_I_B_HISTORY_"
            "TO_GSE295080_I1_TO_GSE295080_I2_TO_GSE295080_B2"
        ),
        "predecessor_order": [
            "gse217518_predecessor",
            "encsr854ruf_predecessor",
            "gse232572_predecessor",
            "gse113849_predecessor",
            "gse269595_predecessor",
        ],
        "all_binding_groups_must_be_bound": True,
        "gse295080_frozen_i1_must_be_direct_child_of_gse269595_b2": True,
        "gse295080_dynamic_i2_must_be_direct_child_of_frozen_i1": True,
        "clean_head_equals_upstream_equals_live_origin_required": True,
        "direct_parent_changed_path_and_blob_audit_required": True,
        "executing_script_and_focused_test_must_match_implementation_i": True,
        "binding_commit_may_change_only_the_four_own_binding_scalars": True,
        "fail_before_git_asset_or_output_while_any_predecessor_or_own_group_is_unknown": True,
    }:
        raise ProtocolError("production activation rule differs")

    execution = _mapping(
        protocol.get("execution_boundary"), label="execution_boundary"
    )
    if dict(execution) != {
        "production_mode": "SINGLE_BOUND_PRODUCTION_ENTRY_FIXED_BUILT_IN_PUBLIC_READERS_ONLY",
        "ordinary_public_analysis_mode": "NO_SEPARATE_LOCAL_OR_PUBLIC_ANALYSIS_BYPASS",
        "asset_argument_mode": "SINGLE_DIRECTORY_WITH_EXACT_FIVE_FROZEN_BASENAMES",
        "all_asset_identities_verified_before_any_parse": True,
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
        raise ProtocolError("execution boundary differs")

    boundary = _mapping(protocol.get("decision_boundary"), label="decision_boundary")
    if boundary.get("ordinary_public_local_aggregate_candidate_replay_allowed") is not False:
        raise ProtocolError("local aggregate replay bypass is enabled")
    if any(
        boundary.get(key) is not True
        for key in (
            "production_preflight_execution_allowed",
            "production_asset_read_allowed",
            "production_output_allowed",
        )
    ):
        raise ProtocolError("bound-only production permission differs")
    false_boundaries = (
        "network_download_allowed_by_candidate",
        "private_or_sealed_access_allowed",
        "row_level_qualification_execution_allowed",
        "member_sequence_or_row_value_output_allowed",
        "member_or_row_identifier_output_allowed",
        "barcode_output_allowed",
        "sequence_output_allowed",
        "row_effect_pvalue_or_standard_error_output_allowed",
        "split_assignment_output_allowed",
        "qualification_allowed",
        "ordinary_credit_change_allowed",
        "a1_credit_change_allowed",
        "true_a2_credit_change_allowed",
        "canonical_mutation_allowed",
        "training_allowed",
        "gpu_work_allowed",
        "cuda_or_device_probe_allowed",
        "model_selection_allowed",
        "a7_allowed",
        "next_phase_authorized",
        "all_gates_passing_automatically_qualifies_dataset",
    )
    if any(boundary.get(key) is not False for key in false_boundaries):
        raise ProtocolError("a prohibited boundary was enabled")
    if boundary.get("separate_row_level_authority_required_after_all_gates_pass") is not True:
        raise ProtocolError("separate row-level authority requirement was removed")
    if boundary.get("separate_promotion_authority_required_after_row_level_qualification") is not True:
        raise ProtocolError("separate promotion authority requirement was removed")

    inputs = _mapping(
        protocol.get("ordinary_public_input_contract"),
        label="ordinary_public_input_contract",
    )
    if tuple(inputs) != INPUT_KEYS:
        raise ProtocolError("ordinary-public input set differs")
    for key in INPUT_KEYS:
        contract = _mapping(inputs.get(key), label=f"input {key}")
        basename = contract.get("required_basename")
        if basename not in PUBLIC_ASSET_IDENTITIES:
            raise ProtocolError(f"input {key} basename differs")
        if (contract.get("byte_count"), contract.get("sha256")) != (
            PUBLIC_ASSET_IDENTITIES[str(basename)]
        ):
            raise ProtocolError(f"input {key} frozen identity differs")

    gate_contract = _mapping(protocol.get("gate_contract"), label="gate_contract")
    if tuple(gate_contract.get("gate_ids_exactly", ())) != GATE_IDS:
        raise ProtocolError("gate IDs differ from DEC027 exact7")
    if gate_contract.get("independent_gate_axis_count") != len(GATE_IDS):
        raise ProtocolError("independent gate axis count is not seven")
    if tuple(gate_contract.get("comparison_study_units_exactly", ())) != COMPARISON_STUDIES:
        raise ProtocolError("comparison studies differ from DEC027")
    if gate_contract.get("unknown_partial_blocked_or_stop_is_pass") is not False:
        raise ProtocolError("non-pass status may not count as pass")
    if gate_contract.get("row_level_replay_or_qualification_is_not_part_of_this_gate_set") is not True:
        raise ProtocolError("row-level qualification entered this preflight")
    if gate_contract.get("new_accession_endpoint_context_replicate_or_row_may_create_independent_study_credit") is not False:
        raise ProtocolError("non-study units may create independent credit")

    snapshot = _mapping(protocol.get("frozen_gate_snapshot"), label="frozen_gate_snapshot")
    gates = snapshot.get("gate_statuses")
    if not isinstance(gates, list) or len(gates) != len(GATE_IDS):
        raise ProtocolError("frozen snapshot does not contain exact7")
    if tuple(gate.get("gate_id") for gate in gates if isinstance(gate, dict)) != GATE_IDS:
        raise ProtocolError("frozen gate order differs")
    for gate in gates:
        _exact_keys(
            _mapping(gate, label="gate"),
            ("gate_id", "raw_status", "normalized_status", "fact_class", "reason_code"),
            label="gate",
        )
        if gate.get("normalized_status") not in NORMALIZED_CLASSES:
            raise ProtocolError("gate has an unmapped normalized status")
    if _count_gate_statuses(gates) != EXPECTED_STATUS_COUNTS:
        raise ProtocolError("frozen gate counts differ from 3/1/1/1/1")
    if snapshot.get("normalized_gate_counts") != EXPECTED_STATUS_COUNTS:
        raise ProtocolError("stored gate counts disagree with rows")
    if snapshot.get("p0_status") != "FAIL_CLOSED_STOP":
        raise ProtocolError("P0 must remain fail-closed STOP")
    if snapshot.get("p1_row_level_status") != "NOT_AUTHORIZED_NOT_RUN":
        raise ProtocolError("P1 row-level status was changed")

    output = _mapping(protocol.get("aggregate_output_contract"), label="output_contract")
    if output.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ProtocolError("report schema version differs")
    if output.get("report_filename") != REPORT_FILENAME:
        raise ProtocolError("report filename differs")
    if output.get("single_aggregate_json_only") is not True:
        raise ProtocolError("output is not one aggregate JSON")
    if output.get("atomic_no_replace_publication") is not True:
        raise ProtocolError("atomic no-replace publication is not frozen")
    if output.get("identical_existing_report_is_idempotent_success") is not True:
        raise ProtocolError("idempotent identical publication is not frozen")
    if output.get("aggregate_counts_histograms_and_gate_statuses_only") is not True:
        raise ProtocolError("aggregate-only output was relaxed")
    if output.get("row_level_successor_authority_request_eligible") is not False:
        raise ProtocolError("row-level successor request was pre-authorized")
    for key in (
        "member_identifier_allowed",
        "source_or_candidate_sequence_allowed",
        "barcode_allowed",
        "row_endpoint_effect_pvalue_or_standard_error_allowed",
        "replicate_identifier_allowed",
        "split_assignment_allowed",
    ):
        if output.get(key) is not False:
            raise ProtocolError(f"output boundary {key} was enabled")

    state = _mapping(protocol.get("scientific_state"), label="scientific_state")
    if state.get("current_qualified_counts") != {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }:
        raise ProtocolError("qualified count baseline changed")
    if state.get("gse295080_contribution") != {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }:
        raise ProtocolError("GSE295080 contribution is not zero")
    if state.get("qualified") is not False or state.get("independent_study_credit_allowed") is not False:
        raise ProtocolError("qualification or independent credit was enabled")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    validate_protocol(protocol)
    return protocol


def _require_all_bindings(protocol: Mapping[str, Any]) -> None:
    if any(mode != BOUND for mode in _binding_modes(protocol).values()):
        raise ActivationBlocked(
            "GROUPED_BINDINGS_UNKNOWN_FAIL_BEFORE_GIT_ASSET_OR_OUTPUT"
        )


def _run_git(repository_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(repository_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ActivationBlocked("BOUND_REPOSITORY_CHECK_FAILED_BEFORE_ASSET_IO") from exc
    return result.stdout.strip()


def _changed_paths(repository_root: Path, commit: str) -> tuple[str, ...]:
    output = _run_git(
        repository_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return tuple(sorted(line for line in output.splitlines() if line))


def _git_blob(repository_root: Path, commit: str, path: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(repository_root), "show", f"{commit}:{path}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ActivationBlocked("GIT_UNAVAILABLE_BEFORE_ASSET_IO") from exc
    if result.returncode != 0:
        raise ActivationBlocked("BOUND_GIT_BLOB_NOT_READABLE_BEFORE_ASSET_IO")
    return result.stdout


def _verify_frozen_commit(
    repository_root: Path,
    *,
    label: str,
    commit: str,
    expected_parent: str,
    expected_paths: Sequence[str],
    expected_blobs: Mapping[str, str] | None = None,
) -> None:
    ancestry = _run_git(
        repository_root, "rev-list", "--parents", "-n", "1", commit
    ).split()
    if len(ancestry) != 2 or ancestry[0] != commit or ancestry[1] != expected_parent:
        raise ActivationBlocked(f"{label}_DIRECT_PARENT_DIFFERS")
    if _changed_paths(repository_root, commit) != tuple(sorted(expected_paths)):
        raise ActivationBlocked(f"{label}_CHANGED_PATHS_DIFFER")
    for path, expected_sha in (expected_blobs or {}).items():
        actual = hashlib.sha256(_git_blob(repository_root, commit, path)).hexdigest()
        if actual != expected_sha:
            raise ActivationBlocked(f"{label}_BLOB_IDENTITY_DIFFERS")


def _live_origin_head(repository_root: Path, branch: str) -> str:
    ref = f"refs/heads/{branch}"
    value = _run_git(
        repository_root, "ls-remote", "--exit-code", "--heads", "origin", ref
    )
    lines = [line.split() for line in value.splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0]) != 2 or lines[0][1] != ref:
        raise ActivationBlocked("LIVE_ORIGIN_BRANCH_RESOLUTION_DIFFERS")
    commit = lines[0][0]
    if not HEX40.fullmatch(commit):
        raise ActivationBlocked("LIVE_ORIGIN_HEAD_INVALID")
    return commit


def _normalise_own_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    normalised = deepcopy(dict(protocol))
    own = normalised["implementation_binding"]["own_preflight_group"]
    for field in OWN_BINDING_FIELDS:
        own[field] = UNKNOWN
    return normalised


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ActivationBlocked(f"{label}_NOT_JSON") from exc
    if not isinstance(value, dict):
        raise ActivationBlocked(f"{label}_ROOT_NOT_OBJECT")
    return value


def _audit_bound_repository(
    protocol: Mapping[str, Any], protocol_path: Path, repository_root: Path
) -> dict[str, Any]:
    _require_all_bindings(protocol)
    repository = protocol["repository_authority"]
    if repository_root.resolve() != Path(
        str(repository["production_repo_root"])
    ).resolve():
        raise ActivationBlocked("EXECUTION_REPOSITORY_NOT_FROZEN_ROOT")
    if protocol_path.resolve() != (repository_root / CONFIG_REPO_PATH).resolve():
        raise ActivationBlocked("PROTOCOL_PATH_OUTSIDE_FROZEN_REPOSITORY")

    head = _run_git(repository_root, "rev-parse", "HEAD")
    upstream = _run_git(repository_root, "rev-parse", "@{upstream}")
    live_origin = _live_origin_head(repository_root, str(repository["branch"]))
    if head != upstream or head != live_origin:
        raise ActivationBlocked("HEAD_UPSTREAM_LIVE_ORIGIN_DIFFER")
    if _run_git(repository_root, "rev-parse", "--abbrev-ref", "HEAD") != (
        repository["branch"]
    ):
        raise ActivationBlocked("PRODUCTION_BRANCH_DIFFERS")
    if _run_git(repository_root, "rev-parse", "--abbrev-ref", "@{upstream}") != (
        repository["upstream_ref"]
    ):
        raise ActivationBlocked("PRODUCTION_UPSTREAM_DIFFERS")
    if _run_git(
        repository_root, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise ActivationBlocked("BOUND_WORKTREE_OR_INDEX_IS_NOT_CLEAN")

    binding = protocol["implementation_binding"]
    authority = binding["authority_group"]
    runtime = binding["runtime_group"]
    gse217 = binding["gse217518_predecessor"]
    encsr = binding["encsr854ruf_predecessor"]
    gse232 = binding["gse232572_predecessor"]
    gse113 = binding["gse113849_predecessor"]
    gse269 = binding["gse269595_predecessor"]
    own = binding["own_preflight_group"]
    gse217_history = tuple(gse217["append_only_history"])
    encsr_history = tuple(encsr["append_only_history"])
    gse232_history = tuple(gse232["append_only_history"])
    gse113_history = tuple(gse113["append_only_history"])
    gse269_history = tuple(gse269["append_only_history"])
    gse217_b = str(gse217_history[-1]["commit"])
    encsr_b = str(encsr_history[-1]["commit"])
    gse232_b = str(gse232_history[-1]["commit"])
    gse113_b = str(gse113_history[-1]["commit"])
    gse269_b = str(gse269_history[-1]["commit"])
    own_i2 = str(own["implementation_commit"])

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
        ("GSE217518", gse217_history),
        ("ENCSR854RUF", encsr_history),
        ("GSE232572", gse232_history),
        ("GSE113849", gse113_history),
        ("GSE269595", gse269_history),
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
                "GSE295080_I1",
                str(own["frozen_i1_commit"]),
                gse269_b,
                tuple(own["frozen_i1_exact_changed_paths"]),
                own["frozen_i1_blob_sha256_by_path"],
            ),
            (
                "GSE295080_I2",
                own_i2,
                str(own["frozen_i1_commit"]),
                EXACT3,
                {
                    SCRIPT_REPO_PATH: own["implementation_script_sha256"],
                    TEST_REPO_PATH: own["implementation_test_sha256"],
                },
            ),
            ("GSE295080_B2", head, own_i2, (CONFIG_REPO_PATH,), None),
        ]
    )
    for label, commit, parent, paths, blobs in chain:
        _verify_frozen_commit(
            repository_root,
            label=label,
            commit=commit,
            expected_parent=parent,
            expected_paths=paths,
            expected_blobs=blobs,
        )

    implementation_protocol = _load_json_bytes(
        _git_blob(repository_root, own_i2, CONFIG_REPO_PATH), "GSE295080_I2_CONFIG"
    )
    if _normalise_own_binding(protocol) != implementation_protocol:
        raise ActivationBlocked("GSE295080_B2_CHANGED_OUTSIDE_FOUR_OWN_SCALARS")
    if protocol_path.read_bytes() != _git_blob(repository_root, head, CONFIG_REPO_PATH):
        raise ActivationBlocked("WORKING_CONFIG_DIFFERS_FROM_GSE295080_B2")
    script_blob = _git_blob(repository_root, own_i2, SCRIPT_REPO_PATH)
    test_blob = _git_blob(repository_root, own_i2, TEST_REPO_PATH)
    executing_script = Path(__file__).resolve()
    if executing_script != (repository_root / SCRIPT_REPO_PATH).resolve():
        raise ActivationBlocked("EXECUTING_SCRIPT_IS_STALE_COPY")
    if executing_script.read_bytes() != script_blob:
        raise ActivationBlocked("EXECUTING_SCRIPT_DIFFERS_FROM_GSE295080_I2")
    if (repository_root / TEST_REPO_PATH).read_bytes() != test_blob:
        raise ActivationBlocked("FOCUSED_TEST_DIFFERS_FROM_GSE295080_I2")
    return {
        "head_equals_upstream_and_origin": True,
        "worktree_clean": True,
        "authority_commit": AUTHORITY_COMMIT,
        "runtime_binding_commit": RUNTIME_B_COMMIT,
        "gse217518_binding_commit": gse217_b,
        "encsr854ruf_binding_commit": encsr_b,
        "gse232572_binding_commit": gse232_b,
        "gse113849_binding_commit": gse113_b,
        "gse269595_binding_commit": gse269_b,
        "frozen_i1_commit": FROZEN_I1_COMMIT,
        "implementation_commit": own_i2,
        "binding_commit": head,
        "executing_script_matches_bound_implementation": True,
    }


def execute_production(
    *,
    protocol_path: Path,
    asset_dir: Path,
    output_dir: Path,
    recorded_at: str,
) -> tuple[Path, dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    if not isinstance(recorded_at, str) or not recorded_at:
        raise ProtocolError("recorded-at value is absent")
    _require_all_bindings(protocol)
    boundary = _mapping(protocol.get("decision_boundary"), label="decision_boundary")
    if boundary.get("production_preflight_execution_allowed") is not True:
        raise ActivationBlocked("PRODUCTION_EXECUTION_NOT_ENABLED_BY_BOUND_SUCCESSOR")
    repository_binding = _audit_bound_repository(
        protocol, protocol_path, PRODUCTION_REPO_ROOT
    )
    contracts = protocol["ordinary_public_input_contract"]
    asset_paths = {
        key: asset_dir / contracts[key]["required_basename"] for key in INPUT_KEYS
    }
    geometry = _build_actual_geometry(protocol, asset_paths)
    expected = _mapping(
        protocol.get("expected_aggregate_replay"), label="expected_aggregate_replay"
    )
    _same_json_shape(geometry, expected)
    report = _build_report(
        protocol,
        geometry,
        recorded_at,
        production_binding=repository_binding,
    )
    output_path = _write_report(report, output_dir)
    return output_path, report


def _verify_asset(path: Path, contract: Mapping[str, Any], *, label: str) -> None:
    if path.name != contract.get("required_basename"):
        raise AssetError(f"{label} basename differs")
    if not path.is_file():
        raise AssetError(f"{label} is not a regular readable asset")
    if path.stat().st_size != contract.get("byte_count"):
        raise AssetError(f"{label} byte count differs")
    if _sha256(path) != contract.get("sha256"):
        raise AssetError(f"{label} SHA-256 differs")


def _audit_stability(path: Path, contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, set[str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            expected_header = contract.get("required_header_names_exactly")
            if reader.fieldnames != expected_header:
                raise ReplayError("stability table header differs")
            row_count = 0
            all_designs: set[str] = set()
            all_families: set[str] = set()
            all_samples: set[str] = set()
            designs = {"1": set(), "2": set()}
            families = {"1": set(), "2": set()}
            rows_by_library = collections.Counter()
            type_counts = collections.Counter()
            for row in reader:
                row_count += 1
                library = row.get("Library", "")
                if library not in designs:
                    raise ReplayError("stability table has an unexpected library class")
                design = row.get("Element", "")
                family = row.get("familyID", "")
                sample = row.get("sampleID", "")
                variant_type = row.get("Type", "")
                if not design or not family or not sample or not variant_type:
                    raise ReplayError("stability mapping role contains an empty value")
                all_designs.add(design)
                all_families.add(family)
                all_samples.add(sample)
                designs[library].add(design)
                families[library].add(family)
                rows_by_library[library] += 1
                type_counts[variant_type] += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AssetError("stability table cannot be parsed") from exc

    aggregate = {
        "stability_table_row_count": row_count,
        "stability_table_header_count": len(expected_header),
        "all_unique_design_count": len(all_designs),
        "all_unique_family_label_count": len(all_families),
        "all_unique_sample_label_count": len(all_samples),
        "library1": {
            "row_count": rows_by_library["1"],
            "unique_design_count": len(designs["1"]),
            "duplicate_design_row_count": rows_by_library["1"] - len(designs["1"]),
            "unique_family_label_count": len(families["1"]),
        },
        "library2": {
            "row_count": rows_by_library["2"],
            "unique_design_count": len(designs["2"]),
            "duplicate_design_row_count": rows_by_library["2"] - len(designs["2"]),
            "unique_family_label_count": len(families["2"]),
        },
        "type_count_histogram": dict(sorted(type_counts.items())),
    }
    return aggregate, designs


def _audit_fasta(path: Path) -> dict[str, Any]:
    record_count = 0
    unique_headers: set[str] = set()
    length_counts: collections.Counter[int] = collections.Counter()
    current_length: int | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.rstrip("\r\n")
                if line.startswith(">"):
                    if current_length is not None:
                        length_counts[current_length] += 1
                    header = line[1:]
                    if not header:
                        raise ReplayError("author reference FASTA has an empty header")
                    unique_headers.add(header)
                    record_count += 1
                    current_length = 0
                else:
                    if current_length is None:
                        raise ReplayError("author reference FASTA begins before a header")
                    sequence = line.strip()
                    if not sequence or any(base not in "ACGTNacgtn" for base in sequence):
                        raise ReplayError("author reference FASTA has invalid sequence syntax")
                    current_length += len(sequence)
        if current_length is not None:
            length_counts[current_length] += 1
    except (OSError, UnicodeError) as exc:
        raise AssetError("author reference FASTA cannot be parsed") from exc
    return {
        "author_reference_fasta_record_count": record_count,
        "author_reference_fasta_unique_header_count": len(unique_headers),
        "author_reference_fasta_length_histogram": {
            str(length): count for length, count in sorted(length_counts.items())
        },
    }


def _audit_soft(path: Path) -> dict[str, int]:
    sample_count = 0
    titles: list[str] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
            for raw in handle:
                line = raw.rstrip("\r\n")
                if line.startswith("^SAMPLE = "):
                    sample_count += 1
                elif line.startswith("!Sample_title = "):
                    titles.append(line.split(" = ", 1)[1])
    except (OSError, UnicodeError, EOFError) as exc:
        raise AssetError("GEO family SOFT cannot be parsed") from exc
    if len(titles) != sample_count:
        raise ReplayError("GEO family SOFT sample-title geometry is incomplete")

    explicit_bio = sum("biolrep" in title.lower() for title in titles)
    explicit_technical = sum("techrep" in title.lower() for title in titles)
    generic = sum(bool(re.search(r"\brep\s*\d", title.lower())) for title in titles)
    return {
        "geo_sample_count": sample_count,
        "explicit_biological_replicate_label_sample_count": explicit_bio,
        "explicit_technical_replicate_label_sample_count": explicit_technical,
        "generic_replicate_label_sample_count": generic,
        "pre_splice_sample_count": sum("pre-splice" in title.lower() for title in titles),
        "post_splice_sample_count": sum("post-splice" in title.lower() for title in titles),
    }


def _audit_file_inventory(path: Path) -> dict[str, int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["#Archive/File", "Name", "Time", "Size", "Type"]:
                raise ReplayError("GEO file inventory header differs")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AssetError("GEO file inventory cannot be parsed") from exc
    kind_counts = collections.Counter(row["#Archive/File"] for row in rows)
    if set(kind_counts) != {"Archive", "File"}:
        raise ReplayError("GEO file inventory role classes differ")
    try:
        archive_bytes = sum(int(row["Size"]) for row in rows if row["#Archive/File"] == "Archive")
        member_bytes = sum(int(row["Size"]) for row in rows if row["#Archive/File"] == "File")
    except (KeyError, ValueError) as exc:
        raise ReplayError("GEO file inventory has a non-integer size") from exc
    return {
        "geo_file_inventory_record_count": len(rows),
        "geo_archive_count": kind_counts["Archive"],
        "geo_member_file_count": kind_counts["File"],
        "geo_declared_archive_bytes": archive_bytes,
        "geo_declared_member_bytes": member_bytes,
    }


def _audit_gse186455_archive(
    path: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    try:
        with tarfile.open(path, "r:*") as archive:
            regular = sorted((member for member in archive.getmembers() if member.isfile()), key=lambda item: item.name)
            if len(regular) < 2:
                raise ReplayError("GSE186455 processed archive lacks the two-member stability check")
            payloads: list[bytes] = []
            for selected in regular[:2]:
                handle = archive.extractfile(selected)
                if handle is None:
                    raise ReplayError("selected GSE186455 member is not readable")
                payloads.append(handle.read())
    except (OSError, tarfile.TarError) as exc:
        raise AssetError("GSE186455 processed archive cannot be parsed") from exc
    try:
        base_sets: list[set[str]] = []
        row_counts: list[int] = []
        for payload in payloads:
            if payload[:2] == b"\x1f\x8b":
                payload = gzip.decompress(payload)
            text = payload.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text), delimiter="\t")
            if reader.fieldnames != contract.get("processed_member_header_names_exactly"):
                raise ReplayError("GSE186455 selected member header differs")
            row_count = 0
            bases: set[str] = set()
            for row in reader:
                row_count += 1
                name = row.get("seqName", "")
                if "_" not in name or not name.rsplit("_", 1)[0]:
                    raise ReplayError("GSE186455 element-base derivation is not defined")
                bases.add(name.rsplit("_", 1)[0])
            row_counts.append(row_count)
            base_sets.append(bases)
    except (UnicodeError, csv.Error, EOFError) as exc:
        raise AssetError("GSE186455 selected member cannot be decoded") from exc
    return {
        "gse186455_processed_member_count": len(regular),
        "selected_gse186455_processed_member_row_count": row_counts[0],
        "selected_gse186455_unique_element_base_count": len(base_sets[0]),
        "first_two_gse186455_unique_element_base_count_each": [
            len(base_sets[0]),
            len(base_sets[1]),
        ],
        "first_two_gse186455_element_base_intersection_count": len(
            base_sets[0] & base_sets[1]
        ),
        "first_gse186455_member_only_element_base_count": len(
            base_sets[0] - base_sets[1]
        ),
        "second_gse186455_member_only_element_base_count": len(
            base_sets[1] - base_sets[0]
        ),
    }, base_sets[0]


def _build_actual_geometry(
    protocol: Mapping[str, Any], asset_paths: Mapping[str, Path]
) -> dict[str, Any]:
    inputs = _mapping(
        protocol.get("ordinary_public_input_contract"),
        label="ordinary_public_input_contract",
    )
    if tuple(asset_paths) != INPUT_KEYS:
        raise AssetError("provided ordinary-public input set differs")
    for key in INPUT_KEYS:
        _verify_asset(asset_paths[key], _mapping(inputs[key], label=key), label=key)

    stability, designs = _audit_stability(
        asset_paths["stability_table"],
        _mapping(inputs["stability_table"], label="stability_table"),
    )
    fasta = _audit_fasta(asset_paths["author_reference_fasta"])
    soft = _audit_soft(asset_paths["geo_family_soft"])
    inventory = _audit_file_inventory(asset_paths["geo_file_inventory"])
    reference, reference_bases = _audit_gse186455_archive(
        asset_paths["gse186455_processed_archive"],
        _mapping(inputs["gse186455_processed_archive"], label="gse186455_processed_archive"),
    )
    library1 = designs["1"]
    library2 = designs["2"]
    overlap = {
        "library1_exact_name_overlap_count": len(library1 & reference_bases),
        "library1_gse295080_only_count": len(library1 - reference_bases),
        "library1_gse186455_only_count": len(reference_bases - library1),
        "library2_exact_name_overlap_count": len(library2 & reference_bases),
        "library2_gse295080_only_count": len(library2 - reference_bases),
    }
    return {**stability, **fasta, **inventory, **soft, **reference, **overlap}


def _same_json_shape(actual: Any, expected: Any, *, path: str = "geometry") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ReplayError(f"{path} fields differ")
        for key in expected:
            _same_json_shape(actual[key], expected[key], path=f"{path}.{key}")
    elif actual != expected:
        raise ReplayError(f"{path} differs from the frozen aggregate")


def _assert_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise OutputError("aggregate report contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_EXACT_REPORT_KEYS:
                raise OutputError("aggregate report contains a forbidden member-level key")
            _assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite(child)


def _build_report(
    protocol: Mapping[str, Any],
    geometry: Mapping[str, Any],
    recorded_at: str,
    *,
    production_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = _mapping(protocol.get("frozen_gate_snapshot"), label="frozen_gate_snapshot")
    production = production_binding is not None
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "decision_id": DECISION_ID,
        "dataset_id": DATASET_ID,
        "project_id": "PRJNA1252772",
        "recorded_at": recorded_at,
        "record_status": (
            "TERMINAL_AGGREGATE_PREFLIGHT_BOUND_PRODUCTION"
            if production
            else "TERMINAL_AGGREGATE_CANDIDATE_NOT_PRODUCTION_BINDING"
        ),
        "source_mode": (
            "DEC027_BOUND_PRODUCTION_ORDINARY_PUBLIC_AGGREGATE_REPLAY"
            if production
            else "LOCAL_DEC027_AUTHORIZED_ORDINARY_PUBLIC_AGGREGATE_REPLAY_NOT_PRODUCTION_BINDING"
        ),
        "scope": {
            "role": "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_INDEPENDENCE_OVERLAP_ADJUDICATION_ONLY",
            "row_level_qualification_executed": False,
            "private_or_sealed_accessed": False,
            "training_or_gpu_or_model_selection_executed": False,
            "member_payload_serialized": False,
        },
        "input_verification": {
            "ordinary_public_asset_count": len(INPUT_KEYS),
            "all_five_assets_match_frozen_byte_count_and_sha256": True,
            "network_download_performed_by_candidate": False,
        },
        "production_binding": (
            dict(production_binding)
            if production_binding is not None
            else {"status": "NOT_PRODUCTION_BINDING"}
        ),
        "aggregate_geometry": dict(geometry),
        "required_cross_dataset_comparison_disposition": {
            "required_comparison_count": 4,
            "exact_name_comparison_complete_count": 1,
            "required_comparison_unknown_count": 3,
            "gse186455": "COMPLETE_LIBRARY1_FULL_OVERLAP_LIBRARY2_ZERO_EXACT_NAME_OVERLAP",
            "encsr854ruf": "UNKNOWN_NOT_ASSERTED",
            "gse217518": "UNKNOWN_NOT_ASSERTED",
            "gse232572": "UNKNOWN_NOT_ASSERTED",
        },
        "replicate_label_boundary": {
            "official_metadata_label_geometry_closed": True,
            "labels_establish_biological_independence": False,
            "labels_establish_valid_standard_error": False,
            "row_level_replicate_or_standard_error_audit_executed": False,
        },
        "independence_boundary": {
            "library1_reused_library": True,
            "library1_independent_credit_allowed": False,
            "library2_nonoverlap_with_gse186455_by_exact_name_only": True,
            "library2_independent_study_boundary_closed": False,
            "library2_independent_credit_allowed": False,
        },
        "scientific_gates": list(snapshot["gate_statuses"]),
        "normalized_gate_counts": dict(snapshot["normalized_gate_counts"]),
        "p0": {
            "status": snapshot["p0_status"],
            "all_seven_pass": False,
            "failure_consequence": "STOP_BEFORE_ROW_LEVEL_DATA_CUDA_OR_MODEL",
        },
        "p1": {
            "row_level_status": snapshot["p1_row_level_status"],
            "successor_authority_request_eligible": False,
        },
        "terminal_disposition": {
            "verdict": snapshot["aggregate_verdict"],
            "row_level_successor_authority_request": "DO_NOT_REQUEST_ON_CURRENT_EVIDENCE",
            "next_action": protocol["aggregate_output_contract"]["next_action"],
        },
        "scientific_state": dict(protocol["scientific_state"]),
    }
    _assert_finite(report)
    return report


def _write_report(report: Mapping[str, Any], output_dir: Path) -> Path:
    _assert_finite(report)
    try:
        payload = (
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OutputError("aggregate report is not finite JSON") from exc

    output_path = output_dir / REPORT_FILENAME
    directory_created = False
    temporary_path: Path | None = None
    try:
        if output_dir.exists():
            if not output_dir.is_dir():
                raise OutputError("aggregate output path is not a directory")
            entries = list(output_dir.iterdir())
            if entries:
                if len(entries) == 1 and entries[0] == output_path:
                    if output_path.read_bytes() == payload:
                        return output_path
                    raise OutputError("different aggregate report already exists")
                raise OutputError("aggregate output directory has an unexpected entry")
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
            os.link(temporary_path, output_path)
        except FileExistsError as exc:
            if output_path.read_bytes() == payload:
                temporary_path.unlink()
                temporary_path = None
                return output_path
            raise OutputError("different aggregate report appeared") from exc
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(output_dir)
        if list(output_dir.iterdir()) != [output_path]:
            raise OutputError("single fixed aggregate report contract was violated")
        return output_path
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
            protocol_path=args.config,
            asset_dir=args.asset_dir,
            output_dir=args.output_dir,
            recorded_at=args.recorded_at,
        )
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "record_status": report["record_status"],
                    "normalized_gate_counts": report["normalized_gate_counts"],
                    "p0_status": report["p0"]["status"],
                    "p1_row_level_status": report["p1"]["row_level_status"],
                    "verdict": report["terminal_disposition"]["verdict"],
                },
                sort_keys=True,
            )
        )
    except PreflightError as exc:
        print(f"STOP: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
