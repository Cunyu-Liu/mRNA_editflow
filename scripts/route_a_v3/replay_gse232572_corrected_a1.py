#!/usr/bin/env python3
"""Bound, aggregate-only corrected A1 replay for ordinary-public GSE232572.

Production has one entry point and one built-in reader.  Before Git, official
asset, or output I/O, the protocol requires the settled DEC027 authority and
runtime bindings, the ordered GSE217518 and ENCSR854RUF predecessor bindings,
and this producer's own exact3 binding.  A bound run then audits the complete
direct-parent Git chain and executing bytes before it validates and parses the
five frozen ordinary-public assets.  It writes one fixed aggregate JSON and
cannot qualify the study, award credit, create canonical rows, or authorize
training, GPU work, model selection, A7, or any later phase.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "route_a_v3_gse232572_corrected_a1_replay.v1"
PROTOCOL_ID = "GSE232572_CORRECTED_A1_REPLAY_V1"
DATASET_ID = "GSE232572"
REPORT_FILENAME = "GSE232572_CORRECTED_A1_REPLAY_AGGREGATE_PREFLIGHT.json"
UNKNOWN_BINDING_STATUS = "UNKNOWN_NOT_ASSERTED"
BOUND_BINDING_STATUS = "BOUND"

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
BASE_HEAD = AUTHORITY_COMMIT
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
AUTHORITY_BLOBS: dict[str, str] = {
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
GSE217_I1_COMMIT = "17a35f0f88cc988b938aaf25d94a8b32f0cacfc8"
GSE217_I2_COMMIT = "6fbd63be6d0edb9f73cf2f85e446917d3c3ff100"
GSE217_B2_COMMIT = "c3611b0f2e8baeb83422bb07f5446b42edce90ef"
GSE217_I3_COMMIT = "36b535f77b3f27bb872b182dcaf6c646d9781991"
GSE217_B3_COMMIT = "0a46400efee4ead95b1283df73d263f6f8033036"
GSE217_I1_BLOBS = {
    GSE217_CONFIG_PATH: "0aa3324d3cfdfd50837ea32a4d1efef754fe70abdab9805f373401f21a1ccb41",
    GSE217_SCRIPT_PATH: "6ca04bdc464ac30f1c3b83830b74c6621816bd25308e345f39e2c5ee94f21b4c",
    GSE217_TEST_PATH: "b08209856fb852991c1b795864304fcda62a4f63419197c068ec6d1f0fd34691",
}
GSE217_I2_BLOBS = {
    GSE217_CONFIG_PATH: "de064e62e7031725908de8d09a1c6b7a2a36112868208ba0cb387a394062a1d8",
    GSE217_SCRIPT_PATH: "0c96d41e4f9ddd694be0be21aab3bcfcb938d3ccb996f3f7cb10ca0a9b69902e",
    GSE217_TEST_PATH: "c881350f5b6ab457af24a4004eedfa34e30d0c8fac7e1b99f74b37b9fe40ccc7",
}
GSE217_B2_BLOBS = {
    GSE217_CONFIG_PATH: "c808bdc6eb1ad8aaccd3d2ab483415ed0803da4683315b92ef3139f203f61e64",
    GSE217_SCRIPT_PATH: GSE217_I2_BLOBS[GSE217_SCRIPT_PATH],
    GSE217_TEST_PATH: GSE217_I2_BLOBS[GSE217_TEST_PATH],
}
GSE217_I3_BLOBS = {
    GSE217_CONFIG_PATH: "3355ba986f60268d3dd7b985ed31fe9df4aa2acbe9fd03c984956a1270279ff9",
    GSE217_SCRIPT_PATH: "9fa4464e1cc42baacdf39b4bae2427e1895269b8d6f4e1a05e1e944b0434f3fa",
    GSE217_TEST_PATH: "bba0ee97e9ed2500f0155c8d1a776d185661b00d8c7fd48c3c0d718d53ccd097",
}
GSE217_B3_BLOBS = {
    GSE217_CONFIG_PATH: "c5acc8548ab8542ac029a420f21f1d8524bb0f255c6dd53c2d896c2838ce391f",
    GSE217_SCRIPT_PATH: GSE217_I3_BLOBS[GSE217_SCRIPT_PATH],
    GSE217_TEST_PATH: GSE217_I3_BLOBS[GSE217_TEST_PATH],
}

ENCSR_CONFIG_PATH = "configs/route_a_v3_encsr854ruf_dec027_dataset_specific_a1_preflight_v1.json"
ENCSR_SCRIPT_PATH = "scripts/route_a_v3/preflight_encsr854ruf_dec027_dataset_specific_a1.py"
ENCSR_TEST_PATH = "tests/route_a_v3/test_preflight_encsr854ruf_dec027_dataset_specific_a1.py"
ENCSR_EXACT3 = (ENCSR_CONFIG_PATH, ENCSR_SCRIPT_PATH, ENCSR_TEST_PATH)
ENCSR_I1_COMMIT = "c6132d8928df0a64be106b11ee62d225d77249ba"
ENCSR_I2_COMMIT = "5531907c9ede1a4323ffe884c47a410d9bcb946d"
ENCSR_B2_COMMIT = "e52a8d8614724574e3647c6cf0f84041221b76a0"
ENCSR_I3_COMMIT = "c0f65f181ea797978d660ef3c918ee7318a51292"
ENCSR_B3_COMMIT = "d38f4b31cd5add04bbd7f3b839ff60590fa5fad2"
ENCSR_I4_COMMIT = "53f426aef8b12e8dcbfaaf978fcfa7d1c7a911d2"
ENCSR_B4_COMMIT = "56b39f966a272d8ea8022048855d2fcca0ee155a"
ENCSR_I1_BLOBS = {
    ENCSR_CONFIG_PATH: "e1d3747876818f5b0d2b47f4a185cc5fb0f1c6b141b25a1e635768cdde588e2c",
    ENCSR_SCRIPT_PATH: "d8f6517f935624204cfa8669c8322909734417b212287e867ea38d8e031881ec",
    ENCSR_TEST_PATH: "b59e94373fb02cb2a0e65b67183af9b2f3ddcab24bc8f72d4a636f9a781f4714",
}
ENCSR_I2_BLOBS = {
    ENCSR_CONFIG_PATH: "e7f4adf157b638c10161c922d848c494aa6b3b50f8aed9c05c7111907bb691c8",
    ENCSR_SCRIPT_PATH: "4a5910cad545d4b699b2daf20933afe3e6512aff11b015cb6adb983f4911c247",
    ENCSR_TEST_PATH: "364b908433353451501b0419587d20d6702451ae87726231e3ac1800313e60b7",
}
ENCSR_B2_BLOBS = {
    ENCSR_CONFIG_PATH: "2f3d688f463f5ee359ae76aa9111af2d9ee091f77a1c8037d2337feb49583045",
    ENCSR_SCRIPT_PATH: ENCSR_I2_BLOBS[ENCSR_SCRIPT_PATH],
    ENCSR_TEST_PATH: ENCSR_I2_BLOBS[ENCSR_TEST_PATH],
}
ENCSR_I3_BLOBS = {
    ENCSR_CONFIG_PATH: "fe22477d631cc1ce08e9eed6cd0ca48b4723bf9757a16dd37c4a391cc3318263",
    ENCSR_SCRIPT_PATH: "6235c78ea8bcb008fb33b9d3356461bfb0446a516111d941c640e7a9933d6bac",
    ENCSR_TEST_PATH: "a7303cca44bbb0340251fb00640456fdea592bd461f49852510eb719a1c401c5",
}
ENCSR_B3_BLOBS = {
    ENCSR_CONFIG_PATH: "f801ac603ccd8903921e2e513f5b363ce5112ed6b220e93ab6ddf896dcde3ceb",
    ENCSR_SCRIPT_PATH: ENCSR_I3_BLOBS[ENCSR_SCRIPT_PATH],
    ENCSR_TEST_PATH: ENCSR_I3_BLOBS[ENCSR_TEST_PATH],
}
ENCSR_I4_BLOBS = {
    ENCSR_CONFIG_PATH: "1ebbbe07339ccf8b6bf88d1e8fb946976d2065073c479affe8811af3cc1bd088",
    ENCSR_SCRIPT_PATH: "d5a1ef3e174f479404c3ca1b2dcac9b81c3848b2a7008c9333ca5f339f2d15c9",
    ENCSR_TEST_PATH: "1e31a4dd3643f1c8a3b56e3e6bd0f99b1f01cac65b4c6c3b7113aad5c26ee5b2",
}
ENCSR_B4_BLOBS = {
    ENCSR_CONFIG_PATH: "de9df6055a83f29351c4eba2dd895708de4890181e639e26f2228105f6c2cc07",
    ENCSR_SCRIPT_PATH: ENCSR_I4_BLOBS[ENCSR_SCRIPT_PATH],
    ENCSR_TEST_PATH: ENCSR_I4_BLOBS[ENCSR_TEST_PATH],
}

CONFIG_REPO_PATH = "configs/route_a_v3_gse232572_corrected_a1_replay_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/replay_gse232572_corrected_a1.py"
TEST_REPO_PATH = "tests/route_a_v3/test_replay_gse232572_corrected_a1.py"
EXACT3 = (CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH)
OWN_BINDING_FIELDS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
FROZEN_GSE217_BINDING = {
    "status": BOUND_BINDING_STATUS,
    "i1_commit": GSE217_I1_COMMIT,
    "i1_expected_parent": RUNTIME_B_COMMIT,
    "i1_exact_changed_paths": list(GSE217_EXACT3),
    "i1_blob_sha256_by_path": GSE217_I1_BLOBS,
    "i2_commit": GSE217_I2_COMMIT,
    "i2_expected_parent": GSE217_I1_COMMIT,
    "i2_exact_changed_paths": list(GSE217_EXACT3),
    "i2_blob_sha256_by_path": GSE217_I2_BLOBS,
    "b2_commit": GSE217_B2_COMMIT,
    "b2_expected_parent": GSE217_I2_COMMIT,
    "b2_exact_changed_paths": [GSE217_CONFIG_PATH],
    "b2_blob_sha256_by_path": GSE217_B2_BLOBS,
    "i3_commit": GSE217_I3_COMMIT,
    "i3_expected_parent": GSE217_B2_COMMIT,
    "i3_exact_changed_paths": list(GSE217_EXACT3),
    "i3_blob_sha256_by_path": GSE217_I3_BLOBS,
    "b3_commit": GSE217_B3_COMMIT,
    "b3_expected_parent": GSE217_I3_COMMIT,
    "b3_exact_changed_paths": [GSE217_CONFIG_PATH],
    "b3_blob_sha256_by_path": GSE217_B3_BLOBS,
}
FROZEN_ENCSR_BINDING = {
    "status": BOUND_BINDING_STATUS,
    "i1_commit": ENCSR_I1_COMMIT,
    "i1_expected_parent": GSE217_B3_COMMIT,
    "i1_exact_changed_paths": list(ENCSR_EXACT3),
    "i1_blob_sha256_by_path": ENCSR_I1_BLOBS,
    "i2_commit": ENCSR_I2_COMMIT,
    "i2_expected_parent": ENCSR_I1_COMMIT,
    "i2_exact_changed_paths": list(ENCSR_EXACT3),
    "i2_blob_sha256_by_path": ENCSR_I2_BLOBS,
    "b2_commit": ENCSR_B2_COMMIT,
    "b2_expected_parent": ENCSR_I2_COMMIT,
    "b2_exact_changed_paths": [ENCSR_CONFIG_PATH],
    "b2_blob_sha256_by_path": ENCSR_B2_BLOBS,
    "i3_commit": ENCSR_I3_COMMIT,
    "i3_expected_parent": ENCSR_B2_COMMIT,
    "i3_exact_changed_paths": list(ENCSR_EXACT3),
    "i3_blob_sha256_by_path": ENCSR_I3_BLOBS,
    "b3_commit": ENCSR_B3_COMMIT,
    "b3_expected_parent": ENCSR_I3_COMMIT,
    "b3_exact_changed_paths": [ENCSR_CONFIG_PATH],
    "b3_blob_sha256_by_path": ENCSR_B3_BLOBS,
    "i4_commit": ENCSR_I4_COMMIT,
    "i4_expected_parent": ENCSR_B3_COMMIT,
    "i4_exact_changed_paths": list(ENCSR_EXACT3),
    "i4_blob_sha256_by_path": ENCSR_I4_BLOBS,
    "b4_commit": ENCSR_B4_COMMIT,
    "b4_expected_parent": ENCSR_I4_COMMIT,
    "b4_exact_changed_paths": [ENCSR_CONFIG_PATH],
    "b4_blob_sha256_by_path": ENCSR_B4_BLOBS,
}
PUBLIC_ASSET_IDENTITIES = {
    "GSE232572_C4Sp1.fasta.gz": (
        459691,
        "1c8908215d187ab73ac73dbedd2bf586af5105192aa9e5f5a3ea24b479a04cf8",
    ),
    "GSE232572_C4Sp2.fasta.gz": (
        459212,
        "aded1c42dac44c7f6320d7cbd14e39737d75c5acab64fdd84de2b9c403cdbe03",
    ),
    "GSE232572_C4Sp3.fasta.gz": (
        483778,
        "3e0203f48da7ecc98f9d04a07dd3122548919d7f273756f8e073fb9d559184eb",
    ),
    "GSE232572_RAW.tar": (
        1607680,
        "93afe7c213a2560e52ddb8918a8094b898112b28d5bf64b44279667956f6e5b1",
    ),
    "41467_2024_46795_MOESM4_ESM.xlsx": (
        5135110,
        "d2bdcafb68ae388f84da125d283dc0282cdc81545d704baf81730d7fd613f782",
    ),
}
ROOT_KEYS = {
    "schema_version",
    "protocol_id",
    "contract_id",
    "phase_id",
    "decision_id",
    "dataset_id",
    "study_id",
    "protocol_status",
    "base_snapshot",
    "repository_authority",
    "bindings",
    "production_activation_rule",
    "execution_boundary",
    "upstream_public_replay",
    "public_inputs",
    "publisher_authority",
    "a1_role_boundary",
    "expected_public_replay",
    "endpoint_contract",
    "source_group_and_power_boundary",
    "rights_boundary",
    "historical_exposure_boundary",
    "required_scientific_gate_ids_exactly",
    "output_contract",
    "claim_boundary",
}

REQUIRED_GATE_IDS = (
    "PUBLIC_SOURCE_ASSET_IDENTITY_AND_PRIMARY_ROUTE_CLOSED",
    "REFERENCE_ALTERNATIVE_SOURCE_CANDIDATE_CROSSWALK_CLOSED",
    "FULL_REPORTER_AND_THREE_UTR_CONTEXT_CLOSED",
    "ENDPOINT_DIRECTION_SCALE_TRANSFORM_AND_SEMANTICS_CLOSED",
    "THREE_INDEPENDENT_BIOLOGICAL_REPLICATES_AND_VALID_STANDARD_ERROR_CLOSED",
    "MISSING_QC_AND_SELECTION_CLOSED",
    "LICENSE_AND_REUSE_RIGHTS_CLOSED",
    "HISTORICAL_ANALYTIC_OR_CHECKPOINT_EXPOSURE_CLOSED",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED",
    "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED",
    "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED",
)
INAPPLICABLE_DENSE_GATE_ID = "ELIGIBLE_MULTI_CANDIDATE_POOLS"
FORBIDDEN_ASSET_NAME_FRAGMENTS = (
    ".private.",
    "sealed",
    "restricted",
    "gse246381",
    "access_log",
)
FORBIDDEN_REPORT_KEYS = {
    "member_id",
    "record_id",
    "source_sequence",
    "candidate_sequence",
    "sequence",
    "barcode",
    "row_effect",
    "row_standard_error",
    "replicate_log_ratios",
    "split_assignment",
    "split_or_bootstrap_assignment",
}


class ReplayError(RuntimeError):
    """Fail-closed replay error with a non-member-level reason code."""

    def __init__(self, gate: str, code: str):
        super().__init__(f"{gate}: {code}")
        self.gate = gate
        self.code = code


class ProtocolError(ReplayError):
    pass


class BindingNotReady(ReplayError):
    pass


class AssetError(ReplayError):
    pass


class ReplayInvariantError(ReplayError):
    pass


class OutputError(ReplayError):
    pass


def _require_mapping(value: Any, gate: str, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(gate, code)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("PROTOCOL", "CONFIG_NOT_READABLE_JSON") from exc
    if not isinstance(document, dict):
        raise ProtocolError("PROTOCOL", "CONFIG_ROOT_NOT_OBJECT")
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AssetError("ASSET_IDENTITY", "FILE_NOT_READABLE") from exc
    return digest.hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_authority_binding(binding: Mapping[str, Any]) -> None:
    expected = {
        "status": BOUND_BINDING_STATUS,
        "authority_commit": AUTHORITY_COMMIT,
        "authority_expected_parent": AUTHORITY_PARENT,
        "authority_exact_changed_paths": list(AUTHORITY_EXACT12),
        "authority_blob_sha256_by_path": AUTHORITY_BLOBS,
    }
    if dict(binding) != expected:
        raise ProtocolError("PROTOCOL", "DEC027_AUTHORITY_BINDING_DIFFERS")


def _validate_runtime_binding(binding: Mapping[str, Any]) -> None:
    if binding.get("status") != BOUND_BINDING_STATUS:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_NOT_BOUND")
    if binding.get("runtime_event_id") != BASE_EVENT:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_EVENT_DIFFERS")
    if binding.get("frozen_i1_commit") != RUNTIME_I1_COMMIT:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_I1_DIFFERS")
    if binding.get("frozen_i1_expected_parent") != AUTHORITY_COMMIT:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_I1_PARENT_DIFFERS")
    if binding.get("frozen_i1_exact_changed_paths") != list(RUNTIME_EXACT3):
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_I1_EXACT3_DIFFERS")
    if binding.get("frozen_i1_blob_sha256_by_path") != RUNTIME_I1_BLOBS:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_I1_BLOBS_DIFFER")
    if binding.get("implementation_commit") != RUNTIME_I2_COMMIT:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_I2_DIFFERS")
    if binding.get("implementation_expected_parent") != RUNTIME_I1_COMMIT:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_I2_PARENT_DIFFERS")
    if binding.get("implementation_exact_changed_paths") != list(RUNTIME_EXACT3):
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_I2_EXACT3_DIFFERS")
    if binding.get("implementation_blob_sha256_by_path") != RUNTIME_I2_BLOBS:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_I2_BLOBS_DIFFER")
    if binding.get("binding_commit") != RUNTIME_B_COMMIT:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_B_DIFFERS")
    if binding.get("binding_expected_parent") != RUNTIME_I2_COMMIT:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_B2_PARENT_DIFFERS")
    if binding.get("binding_exact_changed_paths") != [RUNTIME_CONFIG_PATH]:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_B2_PATH_DIFFERS")
    if binding.get("binding_blob_sha256_by_path") != RUNTIME_B_BLOBS:
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_B2_BLOBS_DIFFER")


def _validate_frozen_predecessor_lifecycle(
    binding: Mapping[str, Any], *, name: str, expected: Mapping[str, Any]
) -> None:
    if dict(binding) != dict(expected):
        raise ProtocolError("PROTOCOL", f"{name}_FULL_LIFECYCLE_BINDING_DIFFERS")


def _implementation_binding_mode(
    binding: Mapping[str, Any], *, predecessor_mode: str
) -> str:
    if binding.get("unknown_to_bound_fields") != list(OWN_BINDING_FIELDS):
        raise ProtocolError("PROTOCOL", "OWN_FOUR_SCALAR_GROUP_DIFFERS")
    if binding.get("implementation_script_path") != SCRIPT_REPO_PATH:
        raise ProtocolError("PROTOCOL", "OWN_SCRIPT_PATH_DIFFERS")
    if binding.get("implementation_test_path") != TEST_REPO_PATH:
        raise ProtocolError("PROTOCOL", "OWN_TEST_PATH_DIFFERS")
    if binding.get("implementation_exact_changed_paths") != list(EXACT3):
        raise ProtocolError("PROTOCOL", "OWN_I_EXACT3_DIFFERS")
    if binding.get("binding_exact_changed_paths") != [CONFIG_REPO_PATH]:
        raise ProtocolError("PROTOCOL", "OWN_B_PATH_DIFFERS")
    dynamic = tuple(binding.get(field) for field in OWN_BINDING_FIELDS)
    if binding.get("status") == UNKNOWN_BINDING_STATUS:
        if dynamic != (UNKNOWN_BINDING_STATUS,) * len(OWN_BINDING_FIELDS):
            raise ProtocolError("PROTOCOL", "OWN_PARTIAL_GROUP_FORBIDDEN")
        return UNKNOWN_BINDING_STATUS
    if binding.get("status") != BOUND_BINDING_STATUS:
        raise ProtocolError("PROTOCOL", "OWN_STATUS_INVALID")
    if predecessor_mode != BOUND_BINDING_STATUS:
        raise ProtocolError("PROTOCOL", "OWN_PREDECESSOR_NOT_BOUND")
    if not _is_hex(binding.get("implementation_commit"), 40):
        raise ProtocolError("PROTOCOL", "OWN_IMPLEMENTATION_COMMIT_INVALID")
    for field in ("implementation_script_sha256", "implementation_test_sha256"):
        if not _is_hex(binding.get(field), 64):
            raise ProtocolError("PROTOCOL", f"{field.upper()}_INVALID")
    return BOUND_BINDING_STATUS


def _binding_modes(protocol: Mapping[str, Any]) -> dict[str, str]:
    bindings = _require_mapping(
        protocol.get("bindings"), "PROTOCOL", "BINDINGS_NOT_OBJECT"
    )
    if set(bindings) != {
        "authority",
        "runtime",
        "gse217518_predecessor",
        "encsr854ruf_predecessor",
        "implementation",
    }:
        raise ProtocolError("PROTOCOL", "BINDING_GROUP_CLOSURE_DIFFERS")
    for name, value in bindings.items():
        _require_mapping(value, "PROTOCOL", f"{name.upper()}_NOT_OBJECT")
    _validate_authority_binding(bindings["authority"])
    _validate_runtime_binding(bindings["runtime"])
    _validate_frozen_predecessor_lifecycle(
        bindings["gse217518_predecessor"],
        name="GSE217518",
        expected=FROZEN_GSE217_BINDING,
    )
    _validate_frozen_predecessor_lifecycle(
        bindings["encsr854ruf_predecessor"],
        name="ENCSR854RUF",
        expected=FROZEN_ENCSR_BINDING,
    )
    own_mode = _implementation_binding_mode(
        bindings["implementation"], predecessor_mode=BOUND_BINDING_STATUS
    )
    return {
        "gse217518_predecessor": BOUND_BINDING_STATUS,
        "encsr854ruf_predecessor": BOUND_BINDING_STATUS,
        "implementation": own_mode,
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if set(protocol) != ROOT_KEYS:
        raise ProtocolError("PROTOCOL", "ROOT_KEY_CLOSURE_DIFFERS")
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "decision_id": "V3-DEC-027",
        "dataset_id": DATASET_ID,
        "study_id": DATASET_ID,
        "protocol_status": "DRAFT_CANDIDATE_NOT_ACTIVE_PROTOCOL",
    }
    for key, expected in expected_scalars.items():
        if protocol.get(key) != expected:
            raise ProtocolError("PROTOCOL", f"{key.upper()}_NOT_FROZEN")

    snapshot = _require_mapping(
        protocol.get("base_snapshot"), "PROTOCOL", "BASE_SNAPSHOT_NOT_OBJECT"
    )
    if snapshot.get("remote_branch") != PRODUCTION_BRANCH:
        raise ProtocolError("PROTOCOL", "BASE_BRANCH_NOT_FROZEN")
    if snapshot.get("dec027_authority_head") != AUTHORITY_COMMIT:
        raise ProtocolError("PROTOCOL", "AUTHORITY_HEAD_NOT_FROZEN")
    if snapshot.get("dec027_authority_parent") != AUTHORITY_PARENT:
        raise ProtocolError("PROTOCOL", "AUTHORITY_PARENT_NOT_FROZEN")
    if snapshot.get("pre_dec027_projection_event") != "A1-EVT-058":
        raise ProtocolError("PROTOCOL", "PRE_DEC027_EVENT_NOT_FROZEN")
    if snapshot.get("bound_runtime_event") != BASE_EVENT:
        raise ProtocolError("PROTOCOL", "RUNTIME_EVENT_NOT_FROZEN")
    credit = _require_mapping(
        snapshot.get("current_credit"), "PROTOCOL", "CURRENT_CREDIT_NOT_OBJECT"
    )
    if dict(credit) != {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }:
        raise ProtocolError("PROTOCOL", "CURRENT_CREDIT_NOT_ZERO")

    gate_ids = protocol.get("required_scientific_gate_ids_exactly")
    if not isinstance(gate_ids, list) or tuple(gate_ids) != REQUIRED_GATE_IDS:
        raise ProtocolError("PROTOCOL", "EXACT_ELEVEN_A1_GATES_NOT_FROZEN")
    if INAPPLICABLE_DENSE_GATE_ID in gate_ids:
        raise ProtocolError("PROTOCOL", "DENSE_TRUE_A2_GATE_MIXED_INTO_A1")

    role = _require_mapping(
        protocol.get("a1_role_boundary"), "PROTOCOL", "A1_ROLE_NOT_OBJECT"
    )
    if role.get("dense_multi_candidate_true_a2_gate_applicability") != (
        "NOT_APPLICABLE_FOR_A1_REPLAY"
    ):
        raise ProtocolError("PROTOCOL", "DENSE_GATE_NOT_EXACT_NA")
    if role.get("dense_multi_candidate_gate_may_block_a1_replay") is not False:
        raise ProtocolError("PROTOCOL", "DENSE_GATE_MAY_NOT_BLOCK_A1")
    if dict(_require_mapping(
        role.get("current_contribution"), "PROTOCOL", "CURRENT_CONTRIBUTION_NOT_OBJECT"
    )) != {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0}:
        raise ProtocolError("PROTOCOL", "CURRENT_CONTRIBUTION_NOT_ZERO")

    execution = _require_mapping(
        protocol.get("execution_boundary"), "PROTOCOL", "EXECUTION_BOUNDARY_NOT_OBJECT"
    )
    required_execution_false = (
        "private_or_sealed_access_allowed",
        "persistent_member_level_intermediate_allowed",
        "row_or_member_identifier_output_allowed",
        "sequence_output_allowed",
        "row_effect_or_standard_error_output_allowed",
        "split_assignment_output_allowed",
        "canonical_materialization_allowed",
        "training_allowed",
        "gpu_work_allowed",
        "model_selection_allowed",
        "a7_allowed",
        "next_phase_allowed",
    )
    if execution.get("ordinary_public_inputs_only") is not True:
        raise ProtocolError("PROTOCOL", "ORDINARY_PUBLIC_BOUNDARY_NOT_FROZEN")
    if any(execution.get(key) is not False for key in required_execution_false):
        raise ProtocolError("PROTOCOL", "PROHIBITED_EXECUTION_CAPABILITY_ENABLED")
    if execution.get("production_mode") != (
        "SINGLE_BOUND_PRODUCTION_ENTRY_FIXED_BUILT_IN_PUBLIC_READER_ONLY"
    ):
        raise ProtocolError("PROTOCOL", "PRODUCTION_FAIL_CLOSED_ORDER_NOT_FROZEN")
    if execution.get("ordinary_public_analysis_mode") != (
        "NO_SEPARATE_PUBLIC_ANALYSIS_BYPASS"
    ):
        raise ProtocolError("PROTOCOL", "PUBLIC_ANALYSIS_MODE_NOT_FROZEN")

    repository = _require_mapping(
        protocol.get("repository_authority"),
        "PROTOCOL",
        "REPOSITORY_AUTHORITY_NOT_OBJECT",
    )
    if dict(repository) != {
        "production_repo_root": str(PRODUCTION_REPO_ROOT),
        "branch": PRODUCTION_BRANCH,
        "upstream_ref": PRODUCTION_UPSTREAM,
        "live_origin_head_required": True,
        "clean_worktree_and_index_required": True,
    }:
        raise ProtocolError("PROTOCOL", "REPOSITORY_AUTHORITY_DIFFERS")
    _binding_modes(protocol)

    activation = _require_mapping(
        protocol.get("production_activation_rule"),
        "PROTOCOL",
        "PRODUCTION_ACTIVATION_RULE_NOT_OBJECT",
    )
    if activation.get("required_commit_chain") != (
        "DEC027_A_TO_RUNTIME_I1_TO_RUNTIME_I2_TO_RUNTIME_B2_"
        "TO_GSE217518_I1_TO_GSE217518_I2_TO_GSE217518_B2_"
        "TO_GSE217518_I3_TO_GSE217518_B3_"
        "TO_ENCSR854RUF_I1_TO_ENCSR854RUF_I2_TO_ENCSR854RUF_B2_"
        "TO_ENCSR854RUF_I3_TO_ENCSR854RUF_B3_"
        "TO_ENCSR854RUF_I4_TO_ENCSR854RUF_B4_"
        "TO_GSE232572_I_TO_GSE232572_B"
    ):
        raise ProtocolError("PROTOCOL", "PRODUCTION_COMMIT_CHAIN_DIFFERS")
    if activation.get("predecessor_order") != [
        "gse217518_predecessor",
        "encsr854ruf_predecessor",
    ]:
        raise ProtocolError("PROTOCOL", "PRODUCTION_PREDECESSOR_ORDER_DIFFERS")
    for key in (
        "all_binding_groups_must_be_bound",
        "gse232572_implementation_i_must_be_direct_child_of_encsr854ruf_b",
        "clean_head_equals_upstream_equals_live_origin_required",
        "direct_parent_changed_path_and_blob_audit_required",
        "executing_script_and_focused_test_must_match_implementation_i",
        "binding_commit_may_change_only_the_four_own_binding_scalars",
        "fail_before_git_asset_or_output_while_any_predecessor_or_own_group_is_unknown",
    ):
        if activation.get(key) is not True:
            raise ProtocolError("PROTOCOL", f"{key.upper()}_NOT_FROZEN")

    public_inputs = _require_mapping(
        protocol.get("public_inputs"), "PROTOCOL", "PUBLIC_INPUTS_NOT_OBJECT"
    )
    fasta_contracts = _require_mapping(
        public_inputs.get("fasta_by_subpool"),
        "PROTOCOL",
        "FASTA_INPUTS_NOT_OBJECT",
    )
    if set(fasta_contracts) != {"1", "2", "3"}:
        raise ProtocolError("PROTOCOL", "FASTA_SUBPOOL_CLOSURE_DIFFERS")
    asset_contracts = [
        _require_mapping(
            fasta_contracts[str(index)],
            "PROTOCOL",
            f"FASTA{index}_CONTRACT_NOT_OBJECT",
        )
        for index in (1, 2, 3)
    ] + [
        _require_mapping(public_inputs.get("raw_tar"), "PROTOCOL", "RAW_TAR_NOT_OBJECT"),
        _require_mapping(
            public_inputs.get("published_results"),
            "PROTOCOL",
            "PUBLISHED_RESULTS_NOT_OBJECT",
        ),
    ]
    if {str(item.get("filename")) for item in asset_contracts} != set(
        PUBLIC_ASSET_IDENTITIES
    ):
        raise ProtocolError("PROTOCOL", "PUBLIC_ASSET_FILENAME_CLOSURE_DIFFERS")
    for item in asset_contracts:
        filename = str(item.get("filename"))
        if (item.get("bytes"), item.get("sha256")) != PUBLIC_ASSET_IDENTITIES[filename]:
            raise ProtocolError("PROTOCOL", "PUBLIC_ASSET_IDENTITY_DIFFERS")

    publisher = _require_mapping(
        protocol.get("publisher_authority"),
        "PROTOCOL",
        "PUBLISHER_AUTHORITY_NOT_OBJECT",
    )
    expected_publisher_context = {
        "article_doi": "10.1038/s41467-024-46795-7",
        "article_license": "CC_BY_4_0",
        "reported_context": "HUMAN_HELA_EGFP_THREE_UTR_REPORTER",
        "reported_construct": (
            "200NT_OLIGO_WITH_164NT_FLANK_CENTERED_ON_VARIANT_AND_CLONING_ADAPTORS"
        ),
        "reported_biological_replicate_count": 3,
        "reported_primary_endpoint": (
            "AUTHOR_PUBLISHED_NATURAL_LOG_RELATIVE_ACTIVITY_ALT_OVER_REF"
        ),
    }
    if any(publisher.get(key) != value for key, value in expected_publisher_context.items()):
        raise ProtocolError("PROTOCOL", "PUBLISHER_CONTEXT_OR_ENDPOINT_DIFFERS")

    expected = _require_mapping(
        protocol.get("expected_public_replay"),
        "PROTOCOL",
        "EXPECTED_REPLAY_NOT_OBJECT",
    )
    expected_counts = {
        "published_universe_row_count": 11929,
        "accepted_reference_alternative_pair_count": 8068,
        "rejected_published_row_count": 3861,
        "accepted_pair_complete_raw_endpoint_count": 8068,
        "accepted_pair_incomplete_raw_endpoint_count": 0,
        "raw_auxiliary_defined_pair_count": 8068,
        "raw_auxiliary_zero_undefined_pair_count": 0,
        "exact_edit_distance": 1,
        "biological_replicate_count": 3,
    }
    for key, frozen in expected_counts.items():
        if expected.get(key) != frozen:
            raise ProtocolError("PROTOCOL", f"EXPECTED_{key.upper()}_NOT_FROZEN")
    if dict(_require_mapping(
        expected.get("rejection_reason_counts"),
        "PROTOCOL",
        "EXPECTED_REJECTION_COUNTS_NOT_OBJECT",
    )) != {
        "NO_UNIQUE_SEQUENCE_PAIR": 3404,
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
    }:
        raise ProtocolError("PROTOCOL", "EXPECTED_REJECTION_COUNTS_NOT_FROZEN")

    endpoint = _require_mapping(
        protocol.get("endpoint_contract"), "PROTOCOL", "ENDPOINT_CONTRACT_NOT_OBJECT"
    )
    if endpoint.get("replicate_derived_standard_error_formula") != (
        "SAMPLE_SD_OF_THREE_BIOLOGICAL_REPLICATE_LOG_RATIOS_DIVIDED_BY_SQRT_3"
    ):
        raise ProtocolError("PROTOCOL", "STANDARD_ERROR_FORMULA_NOT_FROZEN")
    if endpoint.get("pseudocount") is not None:
        raise ProtocolError("PROTOCOL", "PSEUDOCOUNT_NOT_NULL")
    if endpoint.get("zero_count_policy") != "UNDEFINED_NEVER_IMPUTE_ZERO":
        raise ProtocolError("PROTOCOL", "ZERO_POLICY_NOT_FROZEN")

    grouping = _require_mapping(
        protocol.get("source_group_and_power_boundary"),
        "PROTOCOL",
        "SOURCE_GROUP_BOUNDARY_NOT_OBJECT",
    )
    if grouping.get("formal_split_execution_allowed") is not False:
        raise ProtocolError("PROTOCOL", "SPLIT_EXECUTION_ENABLED")
    if grouping.get("formal_power_gate_execution_allowed") is not False:
        raise ProtocolError("PROTOCOL", "POWER_EXECUTION_ENABLED")
    if grouping.get("near_duplicate_rule_status") != UNKNOWN_BINDING_STATUS:
        raise ProtocolError("PROTOCOL", "NEAR_DUPLICATE_RULE_PREJUDGED")
    if grouping.get("post_near_duplicate_effective_n_status") != UNKNOWN_BINDING_STATUS:
        raise ProtocolError("PROTOCOL", "POST_DEDUP_N_PREJUDGED")

    output = _require_mapping(
        protocol.get("output_contract"), "PROTOCOL", "OUTPUT_CONTRACT_NOT_OBJECT"
    )
    if output.get("filename") != REPORT_FILENAME:
        raise ProtocolError("PROTOCOL", "OUTPUT_FILENAME_NOT_FROZEN")
    if output.get("single_aggregate_json_only") is not True:
        raise ProtocolError("PROTOCOL", "SINGLE_AGGREGATE_OUTPUT_NOT_FROZEN")
    if output.get("member_payload_allowed") is not False:
        raise ProtocolError("PROTOCOL", "MEMBER_PAYLOAD_ENABLED")
    if output.get("all_scientific_gates_pass_automatically_qualifies") is not False:
        raise ProtocolError("PROTOCOL", "AUTOMATIC_QUALIFICATION_ENABLED")
    if output.get("separate_promotion_authority_required") is not True:
        raise ProtocolError("PROTOCOL", "PROMOTION_AUTHORITY_NOT_REQUIRED")
    if output.get("atomic_no_replace_publication") is not True:
        raise ProtocolError("PROTOCOL", "ATOMIC_NO_REPLACE_NOT_FROZEN")
    if output.get("identical_existing_report_is_idempotent_success") is not True:
        raise ProtocolError("PROTOCOL", "IDEMPOTENT_PUBLICATION_NOT_FROZEN")


def _require_production_bindings(protocol: Mapping[str, Any]) -> None:
    modes = _binding_modes(protocol)
    missing = [name for name, mode in modes.items() if mode != BOUND_BINDING_STATUS]
    if missing:
        raise BindingNotReady(
            "PRODUCTION_BINDING",
            "ORDERED_PREDECESSOR_OR_OWN_GROUP_UNKNOWN_FAIL_BEFORE_GIT_ASSET_OUTPUT",
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
        raise ProtocolError("GIT_BINDING", "GIT_UNAVAILABLE") from exc
    if result.returncode != 0:
        raise ProtocolError("GIT_BINDING", "GIT_AUDIT_COMMAND_FAILED")
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
        raise ProtocolError("GIT_BINDING", "GIT_UNAVAILABLE") from exc
    if result.returncode != 0:
        raise ProtocolError("GIT_BINDING", "BOUND_BLOB_NOT_READABLE")
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
        raise ProtocolError("GIT_BINDING", f"{label}_DIRECT_PARENT_DIFFERS")
    if _changed_paths(repo_root, commit) != tuple(sorted(expected_paths)):
        raise ProtocolError("GIT_BINDING", f"{label}_CHANGED_PATHS_DIFFER")
    for path, expected_sha in (expected_blobs or {}).items():
        if hashlib.sha256(_git_blob(repo_root, commit, path)).hexdigest() != expected_sha:
            raise ProtocolError("GIT_BINDING", f"{label}_BLOB_IDENTITY_DIFFERS")


def _live_origin_head(repo_root: Path, branch: str) -> str:
    ref = f"refs/heads/{branch}"
    value = _run_git(repo_root, "ls-remote", "--exit-code", "--heads", "origin", ref)
    lines = [line.split() for line in value.splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0]) != 2 or lines[0][1] != ref:
        raise ProtocolError("GIT_BINDING", "LIVE_ORIGIN_BRANCH_RESOLUTION_DIFFERS")
    commit = lines[0][0]
    if not _is_hex(commit, 40):
        raise ProtocolError("GIT_BINDING", "LIVE_ORIGIN_HEAD_INVALID")
    return commit


def _normalise_own_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    normalised = deepcopy(dict(protocol))
    own = normalised["bindings"]["implementation"]
    for field in OWN_BINDING_FIELDS:
        own[field] = UNKNOWN_BINDING_STATUS
    return normalised


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("GIT_BINDING", f"{label}_NOT_STRICT_JSON") from exc
    if not isinstance(document, dict):
        raise ProtocolError("GIT_BINDING", f"{label}_ROOT_NOT_OBJECT")
    return document


def _audit_repository_bindings(
    protocol: Mapping[str, Any], protocol_path: Path, repo_root: Path
) -> dict[str, str]:
    """Audit the full A/runtime/GSE217/ENCSR lineage followed by own I/B."""

    _require_production_bindings(protocol)
    repository = _require_mapping(
        protocol.get("repository_authority"),
        "GIT_BINDING",
        "REPOSITORY_AUTHORITY_NOT_OBJECT",
    )
    if repo_root.resolve() != Path(str(repository["production_repo_root"])).resolve():
        raise ProtocolError("GIT_BINDING", "EXECUTION_REPOSITORY_NOT_FROZEN_ROOT")
    if protocol_path.resolve() != (repo_root / CONFIG_REPO_PATH).resolve():
        raise ProtocolError("GIT_BINDING", "PROTOCOL_PATH_OUTSIDE_FROZEN_REPOSITORY")

    head = _run_git(repo_root, "rev-parse", "HEAD")
    upstream = _run_git(repo_root, "rev-parse", "@{upstream}")
    live_origin = _live_origin_head(repo_root, str(repository["branch"]))
    if head != upstream or head != live_origin:
        raise ProtocolError("GIT_BINDING", "HEAD_UPSTREAM_LIVE_ORIGIN_DIFFER")
    if _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") != (
        repository["branch"]
    ):
        raise ProtocolError("GIT_BINDING", "PRODUCTION_BRANCH_DIFFERS")
    if _run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}") != (
        repository["upstream_ref"]
    ):
        raise ProtocolError("GIT_BINDING", "PRODUCTION_UPSTREAM_DIFFERS")
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ProtocolError("GIT_BINDING", "PRODUCTION_WORKTREE_OR_INDEX_DIRTY")

    bindings = protocol["bindings"]
    authority = bindings["authority"]
    runtime = bindings["runtime"]
    gse217 = bindings["gse217518_predecessor"]
    encsr = bindings["encsr854ruf_predecessor"]
    own = bindings["implementation"]
    own_i = str(own["implementation_commit"])

    chain = (
        ("DEC027_AUTHORITY_A", AUTHORITY_COMMIT, AUTHORITY_PARENT, AUTHORITY_EXACT12, authority["authority_blob_sha256_by_path"]),
        ("DEC027_RUNTIME_I1", RUNTIME_I1_COMMIT, AUTHORITY_COMMIT, RUNTIME_EXACT3, runtime["frozen_i1_blob_sha256_by_path"]),
        ("DEC027_RUNTIME_I2", RUNTIME_I2_COMMIT, RUNTIME_I1_COMMIT, RUNTIME_EXACT3, runtime["implementation_blob_sha256_by_path"]),
        ("DEC027_RUNTIME_B2", RUNTIME_B_COMMIT, RUNTIME_I2_COMMIT, (RUNTIME_CONFIG_PATH,), runtime["binding_blob_sha256_by_path"]),
        ("GSE217518_I1", GSE217_I1_COMMIT, RUNTIME_B_COMMIT, GSE217_EXACT3, gse217["i1_blob_sha256_by_path"]),
        ("GSE217518_I2", GSE217_I2_COMMIT, GSE217_I1_COMMIT, GSE217_EXACT3, gse217["i2_blob_sha256_by_path"]),
        ("GSE217518_B2", GSE217_B2_COMMIT, GSE217_I2_COMMIT, (GSE217_CONFIG_PATH,), gse217["b2_blob_sha256_by_path"]),
        ("GSE217518_I3", GSE217_I3_COMMIT, GSE217_B2_COMMIT, GSE217_EXACT3, gse217["i3_blob_sha256_by_path"]),
        ("GSE217518_B3", GSE217_B3_COMMIT, GSE217_I3_COMMIT, (GSE217_CONFIG_PATH,), gse217["b3_blob_sha256_by_path"]),
        ("ENCSR854RUF_I1", ENCSR_I1_COMMIT, GSE217_B3_COMMIT, ENCSR_EXACT3, encsr["i1_blob_sha256_by_path"]),
        ("ENCSR854RUF_I2", ENCSR_I2_COMMIT, ENCSR_I1_COMMIT, ENCSR_EXACT3, encsr["i2_blob_sha256_by_path"]),
        ("ENCSR854RUF_B2", ENCSR_B2_COMMIT, ENCSR_I2_COMMIT, (ENCSR_CONFIG_PATH,), encsr["b2_blob_sha256_by_path"]),
        ("ENCSR854RUF_I3", ENCSR_I3_COMMIT, ENCSR_B2_COMMIT, ENCSR_EXACT3, encsr["i3_blob_sha256_by_path"]),
        ("ENCSR854RUF_B3", ENCSR_B3_COMMIT, ENCSR_I3_COMMIT, (ENCSR_CONFIG_PATH,), encsr["b3_blob_sha256_by_path"]),
        ("ENCSR854RUF_I4", ENCSR_I4_COMMIT, ENCSR_B3_COMMIT, ENCSR_EXACT3, encsr["i4_blob_sha256_by_path"]),
        ("ENCSR854RUF_B4", ENCSR_B4_COMMIT, ENCSR_I4_COMMIT, (ENCSR_CONFIG_PATH,), encsr["b4_blob_sha256_by_path"]),
        ("GSE232572_I", own_i, ENCSR_B4_COMMIT, EXACT3, {SCRIPT_REPO_PATH: own["implementation_script_sha256"], TEST_REPO_PATH: own["implementation_test_sha256"]}),
        ("GSE232572_B", head, own_i, (CONFIG_REPO_PATH,), None),
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

    implementation_protocol = _load_json_bytes(
        _git_blob(repo_root, own_i, CONFIG_REPO_PATH), "OWN_I_PROTOCOL"
    )
    if _normalise_own_binding(protocol) != implementation_protocol:
        raise ProtocolError("GIT_BINDING", "OWN_B_CHANGED_OUTSIDE_FOUR_SCALARS")
    if protocol_path.read_bytes() != _git_blob(repo_root, head, CONFIG_REPO_PATH):
        raise ProtocolError("GIT_BINDING", "WORKING_PROTOCOL_DIFFERS_FROM_OWN_B")
    script_blob = _git_blob(repo_root, own_i, SCRIPT_REPO_PATH)
    test_blob = _git_blob(repo_root, own_i, TEST_REPO_PATH)
    executing_script = Path(__file__).resolve()
    if executing_script != (repo_root / SCRIPT_REPO_PATH).resolve():
        raise ProtocolError("GIT_BINDING", "EXECUTING_SCRIPT_IS_STALE_COPY")
    if executing_script.read_bytes() != script_blob:
        raise ProtocolError("GIT_BINDING", "EXECUTING_SCRIPT_DIFFERS_FROM_OWN_I")
    if (repo_root / TEST_REPO_PATH).read_bytes() != test_blob:
        raise ProtocolError("GIT_BINDING", "WORKING_TEST_DIFFERS_FROM_OWN_I")
    return {
        "authority_commit": AUTHORITY_COMMIT,
        "runtime_binding_commit": RUNTIME_B_COMMIT,
        "gse217518_binding_commit": GSE217_B3_COMMIT,
        "encsr854ruf_binding_commit": ENCSR_B4_COMMIT,
        "implementation_commit": own_i,
        "binding_commit": head,
    }


def _reject_nonpublic_asset_path(path: Path, label: str) -> None:
    name = path.name.lower()
    if any(fragment in name for fragment in FORBIDDEN_ASSET_NAME_FRAGMENTS):
        raise AssetError("PUBLIC_ASSET_SCOPE", f"{label.upper()}_PATH_NOT_ORDINARY_PUBLIC")


def _verify_file_identity(path: Path, contract: Mapping[str, Any], label: str) -> None:
    _reject_nonpublic_asset_path(path, label)
    if path.name != contract.get("filename"):
        raise AssetError("ASSET_IDENTITY", f"{label.upper()}_BASENAME_MISMATCH")
    if not path.is_file():
        raise AssetError("ASSET_IDENTITY", f"{label.upper()}_MISSING")
    expected_bytes = contract.get("bytes")
    expected_sha256 = contract.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ProtocolError("PROTOCOL", f"{label.upper()}_BYTE_COUNT_NOT_FROZEN")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ProtocolError("PROTOCOL", f"{label.upper()}_SHA256_NOT_FROZEN")
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise AssetError("ASSET_IDENTITY", f"{label.upper()}_NOT_STATABLE") from exc
    if actual_bytes != expected_bytes:
        raise AssetError("ASSET_IDENTITY", f"{label.upper()}_BYTE_COUNT_MISMATCH")
    if _sha256_file(path) != expected_sha256:
        raise AssetError("ASSET_IDENTITY", f"{label.upper()}_SHA256_MISMATCH")


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssetError("UPSTREAM_IMPLEMENTATION", "MODULE_NOT_LOADABLE")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssetError("UPSTREAM_IMPLEMENTATION", "MODULE_IMPORT_FAILED") from exc
    return module


def _verify_upstream_file(
    repo_root: Path, relative_path: str, expected_sha256: str, label: str
) -> Path:
    path = repo_root / PurePosixPath(relative_path)
    if not path.is_file():
        raise AssetError("UPSTREAM_IMPLEMENTATION", f"{label.upper()}_MISSING")
    if _sha256_file(path) != expected_sha256:
        raise AssetError("UPSTREAM_IMPLEMENTATION", f"{label.upper()}_SHA256_MISMATCH")
    return path


def _load_public_replay(
    *,
    protocol: Mapping[str, Any],
    repo_root: Path,
    fasta_paths: Mapping[int, Path],
    raw_tar: Path,
    published_results: Path,
) -> dict[str, Any]:
    upstream = _require_mapping(
        protocol.get("upstream_public_replay"),
        "PROTOCOL",
        "UPSTREAM_REPLAY_NOT_OBJECT",
    )
    recovery_script = _verify_upstream_file(
        repo_root,
        str(upstream.get("recovery_script_path")),
        str(upstream.get("recovery_script_sha256")),
        "recovery_script",
    )
    helper_path = _verify_upstream_file(
        repo_root,
        str(upstream.get("generic_fasta_helper_path")),
        str(upstream.get("generic_fasta_helper_sha256")),
        "generic_helper",
    )
    recovery_config_path = _verify_upstream_file(
        repo_root,
        str(upstream.get("recovery_config_path")),
        str(upstream.get("recovery_config_sha256")),
        "recovery_config",
    )

    public_inputs = _require_mapping(
        protocol.get("public_inputs"), "PROTOCOL", "PUBLIC_INPUTS_NOT_OBJECT"
    )
    fasta_contract = _require_mapping(
        public_inputs.get("fasta_by_subpool"), "PROTOCOL", "FASTA_INPUTS_NOT_OBJECT"
    )
    if set(fasta_paths) != {1, 2, 3}:
        raise AssetError("PUBLIC_ASSET_SCOPE", "EXACT_THREE_FASTA_PATHS_REQUIRED")
    for subpool in (1, 2, 3):
        _verify_file_identity(
            fasta_paths[subpool],
            _require_mapping(
                fasta_contract.get(str(subpool)),
                "PROTOCOL",
                f"FASTA{subpool}_CONTRACT_NOT_OBJECT",
            ),
            f"fasta{subpool}",
        )
    _verify_file_identity(
        raw_tar,
        _require_mapping(public_inputs.get("raw_tar"), "PROTOCOL", "RAW_TAR_NOT_OBJECT"),
        "raw_tar",
    )
    _verify_file_identity(
        published_results,
        _require_mapping(
            public_inputs.get("published_results"),
            "PROTOCOL",
            "PUBLISHED_RESULTS_NOT_OBJECT",
        ),
        "published_results",
    )

    recovery = _load_module(recovery_script, "gse232572_pinned_public_recovery")
    try:
        recovery_config = recovery._read_json(recovery_config_path)
        recovery._validate_config(recovery_config)
        helper = recovery._load_generic_helper(
            repo_root, str(upstream.get("generic_fasta_helper_path"))
        )
        records: list[dict[str, Any]] = []
        for subpool in (1, 2, 3):
            records.extend(recovery._read_fasta_records(fasta_paths[subpool], subpool, helper))
        if len({str(record["header"]) for record in records}) != len(records):
            raise ReplayInvariantError("PUBLIC_FASTA", "DUPLICATE_HEADER_ACROSS_SUBPOOLS")
        matrix_contract = _require_mapping(
            recovery_config.get("matrix_contract"),
            "UPSTREAM_CONFIG",
            "MATRIX_CONTRACT_NOT_OBJECT",
        )
        matrices = recovery._read_matrices(raw_tar, matrix_contract)
        result_contract = _require_mapping(
            recovery_config.get("published_result_contract"),
            "UPSTREAM_CONFIG",
            "PUBLISHED_RESULT_CONTRACT_NOT_OBJECT",
        )
        published = recovery._read_published_results(published_results, result_contract)
        pairs, rejection_counts = recovery._map_published_universe(records, published)
        complete_pairs, incomplete_count = recovery._accepted_pairs_with_complete_raw_endpoints(
            pairs, matrices
        )
    except ReplayError:
        raise
    except Exception as exc:
        gate = str(getattr(exc, "gate", "UPSTREAM_REPLAY"))
        code = str(getattr(exc, "code", "PINNED_PUBLIC_REPLAY_FAILED"))
        raise ReplayInvariantError(gate, code) from exc

    # The exact helper identity is checked above; retaining this reference makes
    # the dependency explicit without exporting any parsed sequence payload.
    if helper_path.name != "reconstruct_gse232572_sequences.py":
        raise ReplayInvariantError("UPSTREAM_IMPLEMENTATION", "HELPER_BASENAME_CHANGED")
    return {
        "published_count": len(published),
        "accepted_pairs": complete_pairs,
        "accepted_before_endpoint_count": len(pairs),
        "incomplete_endpoint_count": incomplete_count,
        "rejection_counts": dict(rejection_counts),
        "matrices": matrices,
        "published": published,
    }


def _direction(value: float) -> int:
    if abs(value) <= 1e-12:
        return 0
    return 1 if value > 0 else -1


def _increment_histogram(histogram: dict[str, int], value: float) -> None:
    if value == 0:
        histogram["0"] += 1
    elif value <= 0.1:
        histogram["(0,0.1]"] += 1
    elif value <= 0.25:
        histogram["(0.1,0.25]"] += 1
    elif value <= 0.5:
        histogram["(0.25,0.5]"] += 1
    elif value <= 1.0:
        histogram["(0.5,1.0]"] += 1
    else:
        histogram[">1.0"] += 1


def _gate(
    gate_id: str,
    status: str,
    reason_code: str,
    fact_class: str,
    aggregate_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "gate_id": gate_id,
        "status": status,
        "reason_code": reason_code,
        "fact_class": fact_class,
        "blocks_current_qualification": status != "PASS",
    }
    if aggregate_evidence is not None:
        row["aggregate_evidence"] = dict(aggregate_evidence)
    return row


def _aggregate_public_replay(
    protocol: Mapping[str, Any], replay: Mapping[str, Any], recorded_at: str
) -> dict[str, Any]:
    expected = _require_mapping(
        protocol.get("expected_public_replay"), "PROTOCOL", "EXPECTED_REPLAY_NOT_OBJECT"
    )
    published_count = int(replay.get("published_count", -1))
    accepted_pairs = replay.get("accepted_pairs")
    matrices = replay.get("matrices")
    published = replay.get("published")
    rejection_counts = replay.get("rejection_counts")
    if not isinstance(accepted_pairs, Sequence) or isinstance(accepted_pairs, (str, bytes)):
        raise ReplayInvariantError("REPLAY", "ACCEPTED_PAIRS_NOT_SEQUENCE")
    if not isinstance(matrices, Mapping) or not isinstance(published, Mapping):
        raise ReplayInvariantError("REPLAY", "ENDPOINT_INPUTS_NOT_MAPPINGS")
    if not isinstance(rejection_counts, Mapping):
        raise ReplayInvariantError("REPLAY", "REJECTION_COUNTS_NOT_OBJECT")

    actual_rejections = {
        "NO_UNIQUE_SEQUENCE_PAIR": int(rejection_counts.get("NO_UNIQUE_SEQUENCE_PAIR", -1)),
        "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": int(
            rejection_counts.get("AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS", -1)
        ),
    }
    accepted_before_endpoint_count = int(
        replay.get("accepted_before_endpoint_count", -1)
    )
    incomplete_endpoint_count = int(replay.get("incomplete_endpoint_count", -1))
    expected_rejections = dict(_require_mapping(
        expected.get("rejection_reason_counts"),
        "PROTOCOL",
        "EXPECTED_REJECTION_COUNTS_NOT_OBJECT",
    ))
    if published_count != int(expected["published_universe_row_count"]):
        raise ReplayInvariantError("PUBLIC_UNIVERSE", "PUBLISHED_UNIVERSE_COUNT_MISMATCH")
    if accepted_before_endpoint_count != int(
        expected["accepted_reference_alternative_pair_count"]
    ):
        raise ReplayInvariantError("CROSSWALK", "ACCEPTED_PAIR_COUNT_MISMATCH")
    if len(accepted_pairs) != int(expected["accepted_pair_complete_raw_endpoint_count"]):
        raise ReplayInvariantError("ENDPOINT", "COMPLETE_ENDPOINT_PAIR_COUNT_MISMATCH")
    if incomplete_endpoint_count != int(
        expected["accepted_pair_incomplete_raw_endpoint_count"]
    ):
        raise ReplayInvariantError("ENDPOINT", "INCOMPLETE_ENDPOINT_COUNT_MISMATCH")
    if actual_rejections != expected_rejections:
        raise ReplayInvariantError("CROSSWALK", "REJECTION_REASON_COUNTS_MISMATCH")
    if sum(actual_rejections.values()) != int(expected["rejected_published_row_count"]):
        raise ReplayInvariantError("CROSSWALK", "REJECTED_ROW_COUNT_MISMATCH")
    if published_count != accepted_before_endpoint_count + sum(actual_rejections.values()):
        raise ReplayInvariantError("CROSSWALK", "PUBLIC_UNIVERSE_NOT_PARTITIONED")

    source_group_candidate_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    unique_genes: set[str] = set()
    unique_reference_inserts: set[str] = set()
    se_histogram = {
        "0": 0,
        "(0,0.1]": 0,
        "(0.1,0.25]": 0,
        "(0.25,0.5]": 0,
        "(0.5,1.0]": 0,
        ">1.0": 0,
    }
    direction_histogram = {
        "published_positive": 0,
        "published_negative": 0,
        "published_zero": 0,
        "replicate_mean_direction_agrees": 0,
        "replicate_mean_direction_disagrees": 0,
    }
    exact_hamming_one_count = 0
    replicate_defined_count = 0
    replicate_zero_undefined_count = 0
    finite_se_count = 0

    for pair in accepted_pairs:
        if not isinstance(pair, Mapping):
            raise ReplayInvariantError("CROSSWALK", "PAIR_NOT_OBJECT")
        ref = pair.get("ref")
        alt = pair.get("alt")
        if not isinstance(ref, Mapping) or not isinstance(alt, Mapping):
            raise ReplayInvariantError("CROSSWALK", "PAIR_ALLELES_NOT_OBJECTS")
        ref_insert = str(ref.get("insert", ""))
        alt_insert = str(alt.get("insert", ""))
        if not ref_insert or len(ref_insert) != len(alt_insert):
            raise ReplayInvariantError("CROSSWALK", "PAIR_INSERT_LENGTH_MISMATCH")
        hamming = sum(left != right for left, right in zip(ref_insert, alt_insert))
        if hamming != int(expected["exact_edit_distance"]):
            raise ReplayInvariantError("CROSSWALK", "PAIR_NOT_EXACT_HAMMING_ONE")
        exact_hamming_one_count += 1

        key = pair.get("published_key")
        try:
            published_row = published[key]
        except (KeyError, TypeError) as exc:
            raise ReplayInvariantError("ENDPOINT", "PUBLISHED_LABEL_JOIN_MISSING") from exc
        try:
            published_lnfc = float(published_row["published_lnfc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayInvariantError("ENDPOINT", "PUBLISHED_LNFC_NOT_NUMERIC") from exc
        if not math.isfinite(published_lnfc):
            raise ReplayInvariantError("ENDPOINT", "PUBLISHED_LNFC_NOT_FINITE")

        try:
            subpool = int(ref["subpool_number"])
            ref_header = str(ref["header"])
            alt_header = str(alt["header"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayInvariantError("ENDPOINT", "PAIR_MATRIX_IDENTITY_INCOMPLETE") from exc
        replicate_values: list[float] = []
        has_zero = False
        for replicate in (1, 2, 3):
            try:
                dna_ref = float(matrices[(subpool, "DNA", replicate)][ref_header])
                dna_alt = float(matrices[(subpool, "DNA", replicate)][alt_header])
                rna_ref = float(matrices[(subpool, "RNA", replicate)][ref_header])
                rna_alt = float(matrices[(subpool, "RNA", replicate)][alt_header])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayInvariantError("ENDPOINT", "REQUIRED_REPLICATE_COUNT_MISSING") from exc
            values = (dna_ref, dna_alt, rna_ref, rna_alt)
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ReplayInvariantError("ENDPOINT", "REPLICATE_COUNT_NOT_FINITE_NONNEGATIVE")
            if any(value == 0 for value in values):
                has_zero = True
                break
            replicate_values.append(math.log((rna_alt / dna_alt) / (rna_ref / dna_ref)))
        if has_zero:
            replicate_zero_undefined_count += 1
            continue
        if len(replicate_values) != int(expected["biological_replicate_count"]):
            raise ReplayInvariantError("ENDPOINT", "EXACT_THREE_REPLICATES_NOT_CLOSED")
        if any(not math.isfinite(value) for value in replicate_values):
            raise ReplayInvariantError("ENDPOINT", "REPLICATE_LOG_RATIO_NOT_FINITE")
        replicate_defined_count += 1
        replicate_mean = statistics.fmean(replicate_values)
        replicate_se = statistics.stdev(replicate_values) / math.sqrt(3.0)
        if not math.isfinite(replicate_se) or replicate_se < 0:
            raise ReplayInvariantError("ENDPOINT", "REPLICATE_DERIVED_SE_NOT_FINITE")
        finite_se_count += 1
        _increment_histogram(se_histogram, replicate_se)

        published_direction = _direction(published_lnfc)
        if published_direction > 0:
            direction_histogram["published_positive"] += 1
        elif published_direction < 0:
            direction_histogram["published_negative"] += 1
        else:
            direction_histogram["published_zero"] += 1
        if _direction(replicate_mean) == published_direction:
            direction_histogram["replicate_mean_direction_agrees"] += 1
        else:
            direction_histogram["replicate_mean_direction_disagrees"] += 1

        try:
            group_key = (
                str(ref["gene"]),
                str(ref["source"]),
                str(ref["chr_pos"]),
                str(ref["strand"]),
                str(ref["orientation"]),
                ref_insert,
            )
        except KeyError as exc:
            raise ReplayInvariantError("SOURCE_GROUP", "SOURCE_GROUP_FIELDS_INCOMPLETE") from exc
        if any(not field for field in group_key):
            raise ReplayInvariantError("SOURCE_GROUP", "SOURCE_GROUP_FIELD_EMPTY")
        source_group_candidate_counts[group_key] += 1
        unique_genes.add(group_key[0])
        unique_reference_inserts.add(ref_insert)

    if replicate_defined_count != int(expected["raw_auxiliary_defined_pair_count"]):
        raise ReplayInvariantError("ENDPOINT", "DEFINED_REPLICATE_AUXILIARY_COUNT_MISMATCH")
    if replicate_zero_undefined_count != int(
        expected["raw_auxiliary_zero_undefined_pair_count"]
    ):
        raise ReplayInvariantError("ENDPOINT", "ZERO_UNDEFINED_AUXILIARY_COUNT_MISMATCH")
    if finite_se_count != len(accepted_pairs):
        raise ReplayInvariantError("ENDPOINT", "VALID_STANDARD_ERROR_COVERAGE_INCOMPLETE")
    if exact_hamming_one_count != len(accepted_pairs):
        raise ReplayInvariantError("CROSSWALK", "HAMMING_ONE_COVERAGE_INCOMPLETE")

    candidate_count_histogram: collections.Counter[str] = collections.Counter()
    for count in source_group_candidate_counts.values():
        if count == 1:
            candidate_count_histogram["1"] += 1
        elif count == 2:
            candidate_count_histogram["2"] += 1
        elif count == 3:
            candidate_count_histogram["3"] += 1
        else:
            candidate_count_histogram[">=4"] += 1

    scientific_gates = [
        _gate(
            REQUIRED_GATE_IDS[0],
            "PASS",
            "EXACT_FIVE_ORDINARY_PUBLIC_ASSETS_AND_PRIMARY_PUBLISHED_ROUTE_REPLAYED",
            "CONFIRMED_FACT",
            {"ordinary_public_asset_count": 5},
        ),
        _gate(
            REQUIRED_GATE_IDS[1],
            "PASS",
            "PUBLISHED_UNIVERSE_PARTITIONED_AND_8068_EXACT_HAMMING_ONE_REF_ALT_PAIRS_REPLAYED",
            "CONFIRMED_FACT",
            {
                "published_universe_row_count": published_count,
                "accepted_reference_alternative_pair_count": len(accepted_pairs),
                "rejected_published_row_count": sum(actual_rejections.values()),
                "rejection_reason_counts": actual_rejections,
                "exact_hamming_one_pair_count": exact_hamming_one_count,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[2],
            "PASS",
            "PUBLISHER_GROUNDED_HELA_EGFP_THREE_UTR_REPORTER_CONTEXT_CLOSED",
            "CONFIRMED_FACT",
            {
                "biological_context_count": 1,
                "reported_oligo_length_nt": 200,
                "reported_centered_flank_length_nt": 164,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[3],
            "PASS",
            "PUBLISHED_NATURAL_LOG_ALT_OVER_REF_ACTIVITY_DIRECTION_AND_RAW_REPLICATE_TRANSFORM_CLOSED",
            "CONFIRMED_FACT",
            direction_histogram,
        ),
        _gate(
            REQUIRED_GATE_IDS[4],
            "PASS",
            "THREE_BIOLOGICAL_REPLICATES_AND_REPLICATE_DERIVED_SAMPLE_SE_COVER_ALL_ACCEPTED_PAIRS",
            "CONFIRMED_FACT",
            {
                "biological_replicate_count": 3,
                "replicate_derived_se_defined_pair_count": finite_se_count,
                "replicate_derived_se_histogram": se_histogram,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[5],
            "PASS",
            "PUBLIC_UNIVERSE_REJECT_CLOSURE_COMPLETE_AND_ACCEPTED_ENDPOINTS_HAVE_NO_MISSING_OR_ZERO_COUNTS",
            "CONFIRMED_FACT",
            {
                "complete_raw_endpoint_pair_count": len(accepted_pairs),
                "incomplete_raw_endpoint_pair_count": incomplete_endpoint_count,
                "zero_undefined_raw_endpoint_pair_count": replicate_zero_undefined_count,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[6],
            "UNKNOWN_NOT_ASSERTED",
            "PUBLIC_READABILITY_AND_ARTICLE_CC_BY_DO_NOT_CLOSE_GEO_ASSET_QUALIFICATION_REUSE_OR_RELEASE_RIGHTS",
            "UNKNOWN_NOT_ASSERTED",
            {
                "ordinary_public_retrieval_closed": True,
                "article_and_publisher_supplement_license": "CC_BY_4_0",
                "geo_asset_specific_qualification_reuse_and_release_status": "UNKNOWN_NOT_ASSERTED",
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[7],
            "PASS",
            "FULL_DATA_ANALYTIC_EXPOSURE_CLOSED_AS_DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
            "CONFIRMED_FACT",
            {
                "sequence_exposure": "SEQUENCE_EXPOSED",
                "label_exposure": "LABEL_EXPOSED",
                "historical_analytic_use": "CONFIRMED_DEVELOPMENT_USE",
                "untouched_confirmatory": False,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[8],
            "UNKNOWN_NOT_ASSERTED",
            "OUTCOME_BLIND_SOURCE_GROUP_AND_NEAR_DUPLICATE_SPLIT_RULE_NOT_FROZEN_OR_EXECUTED",
            "UNKNOWN_NOT_ASSERTED",
            {
                "nominal_exact_source_group_count": len(source_group_candidate_counts),
                "formal_split_executed": False,
                "split_assignment_output_count": 0,
                "near_duplicate_rule_status": "UNKNOWN_NOT_ASSERTED",
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[9],
            "UNKNOWN_NOT_ASSERTED",
            "NOMINAL_EXACT_SOURCE_GROUP_GEOMETRY_REPLAYED_BUT_POST_NEAR_DUPLICATE_EFFECTIVE_N_NOT_CLOSED",
            "UNKNOWN_NOT_ASSERTED",
            {
                "nominal_exact_source_group_count": len(source_group_candidate_counts),
                "nominal_unique_reference_insert_count": len(unique_reference_inserts),
                "nominal_gene_group_count": len(unique_genes),
                "candidate_count_per_nominal_source_group_histogram": dict(
                    sorted(candidate_count_histogram.items())
                ),
                "post_near_duplicate_effective_n": "UNKNOWN_NOT_ASSERTED",
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[10],
            "NOT_RUN",
            "FORMAL_POWER_AND_FULL_CI_WIDTH_GATE_FORBIDDEN_UNTIL_POST_DEDUP_EFFECTIVE_N_IS_CLOSED",
            "NOT_RUN",
            {
                "required_effective_n_reference": 156,
                "alternative_spearman_rho": 0.25,
                "target_power_minimum": 0.8,
                "confidence_level": 0.95,
                "maximum_full_ci_width": 0.3,
                "formal_power_gate_executed": False,
            },
        ),
    ]
    if tuple(gate["gate_id"] for gate in scientific_gates) != REQUIRED_GATE_IDS:
        raise ReplayInvariantError("REPORT", "EXACT_ELEVEN_GATE_ORDER_CHANGED")

    status_counts = collections.Counter(str(gate["status"]) for gate in scientific_gates)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "study_id": DATASET_ID,
        "recorded_at": recorded_at,
        "execution_mode": "FIXED_BUILT_IN_ORDINARY_PUBLIC_AGGREGATE_REPLAY",
        "protocol_status": "DRAFT_CANDIDATE_NOT_ACTIVE_PROTOCOL",
        "terminal_status": "STOP_REMAINING_QUALIFICATION_GATES_NOT_CLOSED",
        "scientific_disposition": "NOT_QUALIFIED",
        "qualification_status": "AUDIT_PENDING",
        "qualified": False,
        "current_contribution": {
            "ordinary": 0,
            "a1": 0,
            "true_a2": 0,
            "canonical_records": 0,
        },
        "base_snapshot": {
            "dec027_authority_head": BASE_HEAD,
            "bound_runtime_event": BASE_EVENT,
        },
        "binding_status": {
            "authority": protocol["bindings"]["authority"]["status"],
            "runtime": protocol["bindings"]["runtime"]["status"],
            "gse217518_predecessor": protocol["bindings"][
                "gse217518_predecessor"
            ]["status"],
            "encsr854ruf_predecessor": protocol["bindings"][
                "encsr854ruf_predecessor"
            ]["status"],
            "implementation": protocol["bindings"]["implementation"]["status"],
        },
        "aggregate_replay_geometry": {
            "ordinary_public_asset_count": 5,
            "published_universe_row_count": published_count,
            "accepted_reference_alternative_pair_count": len(accepted_pairs),
            "rejected_published_row_count": sum(actual_rejections.values()),
            "rejection_reason_counts": actual_rejections,
            "complete_raw_endpoint_pair_count": len(accepted_pairs),
            "incomplete_raw_endpoint_pair_count": incomplete_endpoint_count,
            "exact_hamming_one_pair_count": exact_hamming_one_count,
            "biological_replicate_count": 3,
            "replicate_derived_se_defined_pair_count": finite_se_count,
            "replicate_derived_se_histogram": se_histogram,
            "nominal_exact_source_group_count": len(source_group_candidate_counts),
            "nominal_unique_reference_insert_count": len(unique_reference_inserts),
            "nominal_gene_group_count": len(unique_genes),
            "candidate_count_per_nominal_source_group_histogram": dict(
                sorted(candidate_count_histogram.items())
            ),
            "post_near_duplicate_effective_n": "UNKNOWN_NOT_ASSERTED",
        },
        "dense_multi_candidate_true_a2_gate": {
            "gate_id": INAPPLICABLE_DENSE_GATE_ID,
            "status": "NOT_APPLICABLE_FOR_A1_REPLAY",
            "blocks_a1_replay": False,
            "true_a2_contribution": 0,
        },
        "scientific_gate_status_counts": dict(sorted(status_counts.items())),
        "scientific_gates": scientific_gates,
        "remaining_blockers": [
            "GEO_ASSET_QUALIFICATION_REUSE_AND_RELEASE_RIGHTS_UNKNOWN_NOT_ASSERTED",
            "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_UNKNOWN_NOT_ASSERTED",
            "POST_NEAR_DUPLICATE_EFFECTIVE_N_UNKNOWN_NOT_ASSERTED",
            "PREFROZEN_POWER_AND_FULL_CI_WIDTH_NOT_RUN",
            "SEPARATE_PROMOTION_AUTHORITY_REQUIRED_EVEN_IF_ALL_GATES_LATER_PASS",
        ],
        "state_change": {
            "authority_changed": False,
            "credit_changed": False,
            "canonical_records_created": 0,
            "training_allowed": False,
            "gpu_work_allowed": False,
            "model_selection_allowed": False,
            "a7_allowed": False,
            "next_phase_allowed": False,
        },
        "payload_boundary": {
            "member_identifier_count_output": 0,
            "sequence_count_output": 0,
            "row_effect_count_output": 0,
            "row_standard_error_count_output": 0,
            "split_assignment_count_output": 0,
            "persistent_member_level_intermediate_count": 0,
        },
    }
    _assert_aggregate_only(report)
    return report


def _assert_aggregate_only(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            rendered_key = str(key).lower()
            if rendered_key in FORBIDDEN_REPORT_KEYS:
                raise ReplayInvariantError(
                    "REPORT_BOUNDARY", "FORBIDDEN_MEMBER_PAYLOAD_KEY"
                )
            _assert_aggregate_only(item, path + (rendered_key,))
    elif isinstance(value, list):
        for item in value:
            _assert_aggregate_only(item, path)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ReplayInvariantError("REPORT_BOUNDARY", "NONFINITE_JSON_NUMBER")


def _write_report(output_dir: Path, report: Mapping[str, Any]) -> Path:
    _assert_aggregate_only(report)
    try:
        payload = (
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OutputError("OUTPUT", "AGGREGATE_REPORT_NOT_FINITE_JSON") from exc

    output_path = output_dir / REPORT_FILENAME
    directory_created = False
    temporary_path: Path | None = None
    try:
        if output_dir.exists():
            if not output_dir.is_dir():
                raise OutputError("OUTPUT", "OUTPUT_PATH_NOT_DIRECTORY")
            entries = list(output_dir.iterdir())
            if entries:
                if len(entries) == 1 and entries[0] == output_path:
                    if output_path.read_bytes() == payload:
                        return output_path
                    raise OutputError("OUTPUT", "DIFFERENT_REPORT_ALREADY_EXISTS")
                raise OutputError("OUTPUT", "OUTPUT_DIRECTORY_HAS_UNEXPECTED_ENTRY")
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
            raise OutputError("OUTPUT", "DIFFERENT_REPORT_APPEARED") from exc
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(output_dir)
        if list(output_dir.iterdir()) != [output_path]:
            raise OutputError("OUTPUT", "SINGLE_FIXED_REPORT_CONTRACT_VIOLATED")
        return output_path
    except OutputError:
        raise
    except OSError as exc:
        raise OutputError("OUTPUT", "ATOMIC_NO_REPLACE_PUBLICATION_FAILED") from exc
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
    protocol_path: Path,
    asset_dir: Path,
    output_dir: Path,
    recorded_at: str,
) -> tuple[Path, dict[str, Any]]:
    """The sole production entry: binding audit precedes every asset/output read."""

    protocol = _load_json(protocol_path)
    _validate_protocol(protocol)
    _require_production_bindings(protocol)
    binding_audit = _audit_repository_bindings(
        protocol, protocol_path, PRODUCTION_REPO_ROOT
    )
    public_inputs = protocol["public_inputs"]
    fasta_contracts = public_inputs["fasta_by_subpool"]
    replay = _load_public_replay(
        protocol=protocol,
        repo_root=PRODUCTION_REPO_ROOT,
        fasta_paths={
            index: asset_dir / fasta_contracts[str(index)]["filename"]
            for index in (1, 2, 3)
        },
        raw_tar=asset_dir / public_inputs["raw_tar"]["filename"],
        published_results=asset_dir / public_inputs["published_results"]["filename"],
    )
    report = _aggregate_public_replay(protocol, replay, recorded_at)
    report["production_binding_audit"] = binding_audit
    _assert_aggregate_only(report)
    output_path = _write_report(output_dir, report)
    return output_path, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recorded-at", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output_path, report = execute_production(
            protocol_path=args.protocol,
            asset_dir=args.asset_dir,
            output_dir=args.output_dir,
            recorded_at=args.recorded_at,
        )
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "terminal_status": report["terminal_status"],
                    "scientific_gate_status_counts": report[
                        "scientific_gate_status_counts"
                    ],
                    "current_contribution": report["current_contribution"],
                },
                sort_keys=True,
            )
        )
        return 0
    except ReplayError as exc:
        print(
            json.dumps(
                {"status": "STOP", "gate": exc.gate, "reason_code": exc.code},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
