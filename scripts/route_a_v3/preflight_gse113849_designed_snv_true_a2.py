#!/usr/bin/env python3
"""Bound aggregate-only GSE113849 designed-SNV true-A2 preflight.

The context rule is part of the protocol and is validated before endpoint or
power fields are interpreted.  The randomized APARENT library is out of scope.
Production has one built-in reader and remains fail-closed until the DEC027
authority, full runtime history, complete GSE217518 and ENCSR854RUF append-only
histories, complete future GSE232572 append-only history, and this exact3
implementation are bound.  It audits that entire direct-parent Git chain before
reading the six frozen ordinary-public author assets or touching output.  It
writes exactly one aggregate JSON and cannot qualify or credit the study.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "route_a_v3_gse113849_designed_snv_true_a2_preflight.v1"
PROTOCOL_ID = "GSE113849_DESIGNED_SNV_TRUE_A2_AGGREGATE_PREFLIGHT_V1"
DATASET_ID = "GSE113849"
REPORT_FILENAME = "GSE113849_DESIGNED_SNV_TRUE_A2_AGGREGATE_PREFLIGHT.json"
SELECTED_CONTEXT_RULE = "ALL_PUBLISHER_ASSERTED_DESIGNED_THREE_UTR_REPORTER_SNVS"
EXPLICIT_UTR3_RULE = "PUBLIC_CONTEXT_FIELD_EXPLICIT_UTR3_ONLY"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
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
GSE217_HISTORY_LABELS = (
    "GSE217518_I1",
    "GSE217518_I2",
    "GSE217518_B2",
    "GSE217518_I3",
    "GSE217518_B3",
)
GSE217_HISTORY_COMMITS = (
    "17a35f0f88cc988b938aaf25d94a8b32f0cacfc8",
    "6fbd63be6d0edb9f73cf2f85e446917d3c3ff100",
    "c3611b0f2e8baeb83422bb07f5446b42edce90ef",
    "36b535f77b3f27bb872b182dcaf6c646d9781991",
    "0a46400efee4ead95b1283df73d263f6f8033036",
)
ENCSR_CONFIG_PATH = "configs/route_a_v3_encsr854ruf_dec027_dataset_specific_a1_preflight_v1.json"
ENCSR_SCRIPT_PATH = "scripts/route_a_v3/preflight_encsr854ruf_dec027_dataset_specific_a1.py"
ENCSR_TEST_PATH = "tests/route_a_v3/test_preflight_encsr854ruf_dec027_dataset_specific_a1.py"
ENCSR_EXACT3 = (ENCSR_CONFIG_PATH, ENCSR_SCRIPT_PATH, ENCSR_TEST_PATH)
ENCSR_HISTORY_LABELS = (
    "ENCSR854RUF_I1",
    "ENCSR854RUF_I2",
    "ENCSR854RUF_B2",
    "ENCSR854RUF_I3",
    "ENCSR854RUF_B3",
    "ENCSR854RUF_I4",
    "ENCSR854RUF_B4",
)
ENCSR_HISTORY_COMMITS = (
    "c6132d8928df0a64be106b11ee62d225d77249ba",
    "5531907c9ede1a4323ffe884c47a410d9bcb946d",
    "e52a8d8614724574e3647c6cf0f84041221b76a0",
    "c0f65f181ea797978d660ef3c918ee7318a51292",
    "d38f4b31cd5add04bbd7f3b839ff60590fa5fad2",
    "53f426aef8b12e8dcbfaaf978fcfa7d1c7a911d2",
    "56b39f966a272d8ea8022048855d2fcca0ee155a",
)
ENCSR_B4_COMMIT = ENCSR_HISTORY_COMMITS[-1]
GSE232_CONFIG_PATH = "configs/route_a_v3_gse232572_corrected_a1_replay_v1.json"
GSE232_SCRIPT_PATH = "scripts/route_a_v3/replay_gse232572_corrected_a1.py"
GSE232_TEST_PATH = "tests/route_a_v3/test_replay_gse232572_corrected_a1.py"
GSE232_EXACT3 = (GSE232_CONFIG_PATH, GSE232_SCRIPT_PATH, GSE232_TEST_PATH)
GSE232_HISTORY_LABELS = ("GSE232572_I1", "GSE232572_B1")
GSE232_HISTORY_COMMITS = (
    "d3dcae4c6ef53c52e942bb511946b52b952d3c7f",
    "0f2c00868b6581edd9a429c7a8a67bb43f6b7776",
)
GSE232_B1_COMMIT = GSE232_HISTORY_COMMITS[-1]

CONFIG_REPO_PATH = "configs/route_a_v3_gse113849_designed_snv_true_a2_preflight_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/preflight_gse113849_designed_snv_true_a2.py"
TEST_REPO_PATH = "tests/route_a_v3/test_preflight_gse113849_designed_snv_true_a2.py"
EXACT3 = (CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH)
OWN_BINDING_FIELDS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
HISTORY_GROUP_FIELDS = (
    "status",
    "append_only_history",
)
HISTORY_STEP_FIELDS = (
    "label",
    "kind",
    "commit",
    "expected_parent",
    "exact_changed_paths",
    "blob_sha256_by_path",
)

REQUIRED_GATE_IDS = (
    "PUBLIC_ASSET_LINEAGE_AND_INTENDED_UNIVERSE_CLOSED",
    "SOURCE_TO_CANDIDATE_IDENTITY_CLOSED",
    "LEGAL_SINGLE_SUBSTITUTION_EDIT_REPLAY_CLOSED",
    "OUTCOME_BLIND_REPORTER_CONTEXT_RULE_CLOSED",
    "DENSE_SOURCE_FAMILY_MINIMUM_THREE_CANDIDATES_CLOSED",
    "ENDPOINT_DIRECTION_SCALE_TRANSFORM_AND_SEMANTICS_CLOSED",
    "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED",
    "MISSING_QC_AND_SELECTION_CLOSED",
    "LICENSE_AND_REUSE_RIGHTS_CLOSED",
    "HISTORICAL_ANALYTIC_OR_CHECKPOINT_EXPOSURE_CLOSED",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED",
    "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED",
    "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED",
)

EXPECTED_INPUT_NAMES = (
    "designed_snv_table",
    "table_generator_notebook",
    "data_preparation_notebook",
    "replicate_notebook",
    "repository_readme",
    "repository_license",
)
PUBLIC_ASSET_IDENTITIES = {
    "apa_variant_mpra_all_snvs.csv": (
        5956224,
        "43cb3cc53433bfe2e7c713b71fabd7b49c021267a2140a21f75da38d3201826c",
    ),
    "analyze_aparent_designed_mpra_store_variant_table_legacy.ipynb": (
        5322,
        "84f9fc5e674879af4185fc5c8eb8665b9e93876c881b79e2c673570ac02889c9",
    ),
    "prepare_aparent_data.ipynb": (
        364785,
        "e5d4bf6a916230554d50d4e92367c5d54c1f39a5cb3e2827ce4b3ad34a3e621b",
    ),
    "analyze_aparent_designed_mpra_lofi_vs_hifi_legacy.ipynb": (
        149595,
        "48577e7c3bd1562a3859841e30e17042a7508fb58bc9469eeb411faafd457daa",
    ),
    "README.md": (
        8354,
        "8ef8342a4a3fa213a39f8a88a19d80231a8367388a4fa27ed5dad0f26c347016",
    ),
    "LICENSE": (
        1059,
        "f7ae7a693143e03c6c2eacda86977342823d5c9a337c2c1dac02d2b0a5a05718",
    ),
}
ROOT_KEYS = {
    "schema_version",
    "protocol_id",
    "contract_id",
    "decision_id",
    "phase_id",
    "dataset_id",
    "study_id",
    "project_alias",
    "protocol_status",
    "base_snapshot",
    "repository_authority",
    "bindings",
    "production_activation_rule",
    "execution_boundary",
    "public_authority_chain",
    "ordinary_public_inputs",
    "intended_universe",
    "outcome_blind_context_freeze",
    "table_contract",
    "expected_mechanical_replay",
    "replicate_and_uncertainty_boundary",
    "rights_boundary",
    "historical_exposure_boundary",
    "source_group_split_and_power_boundary",
    "required_fail_closed_gate_ids_exactly",
    "output_contract",
    "claim_boundary",
}
FORBIDDEN_ASSET_BASENAME_FRAGMENTS = (
    ".private.",
    "sealed",
    "restricted",
    "gse246381",
    "access_log",
)
FORBIDDEN_REPORT_KEYS = {
    "member_id",
    "row_id",
    "record_id",
    "gene",
    "clinvar_id",
    "source_sequence",
    "candidate_sequence",
    "wt_seq",
    "master_seq",
    "sequence",
    "barcode",
    "row_endpoint",
    "row_effect",
    "delta_logodds_true",
    "delta_logodds_pred",
    "delta_p_val",
    "row_pvalue",
    "row_standard_error",
    "split_assignment",
}


class PreflightError(RuntimeError):
    """Fail-closed error carrying only aggregate-safe reason codes."""

    def __init__(self, gate: str, code: str):
        super().__init__(f"{gate}: {code}")
        self.gate = gate
        self.code = code


class ProtocolError(PreflightError):
    pass


class BindingNotReady(PreflightError):
    pass


class AssetError(PreflightError):
    pass


class ReplayInvariantError(PreflightError):
    pass


class OutputError(PreflightError):
    pass


def _mapping(value: Any, gate: str, code: str) -> Mapping[str, Any]:
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
        raise AssetError("ASSET_IDENTITY", "ASSET_NOT_READABLE") from exc
    return digest.hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_sha_map(value: Any, paths: Sequence[str], code: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(paths):
        raise ProtocolError("PROTOCOL", f"{code}_PATH_CLOSURE_DIFFERS")
    if any(not _is_hex(digest, 64) for digest in value.values()):
        raise ProtocolError("PROTOCOL", f"{code}_SHA256_INVALID")


def _validate_authority_binding(binding: Mapping[str, Any]) -> None:
    if dict(binding) != {
        "status": BOUND,
        "authority_commit": AUTHORITY_COMMIT,
        "authority_expected_parent": AUTHORITY_PARENT,
        "authority_exact_changed_paths": list(AUTHORITY_EXACT12),
        "authority_blob_sha256_by_path": AUTHORITY_BLOBS,
    }:
        raise ProtocolError("PROTOCOL", "DEC027_AUTHORITY_BINDING_DIFFERS")


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
        raise ProtocolError("PROTOCOL", "DEC027_RUNTIME_I1_I2_B2_BINDING_DIFFERS")


def _history_binding_mode(
    binding: Mapping[str, Any],
    *,
    name: str,
    exact3: Sequence[str],
    config_path: str,
    expected_parent: str,
    predecessor_mode: str,
    allow_unknown: bool,
    expected_labels: Sequence[str] | None = None,
    expected_commits: Sequence[str] | None = None,
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    if set(binding) != set(HISTORY_GROUP_FIELDS):
        raise ProtocolError("PROTOCOL", f"{name}_HISTORY_GROUP_FIELDS_DIFFER")
    if binding.get("status") == UNKNOWN:
        if not allow_unknown or binding.get("append_only_history") != UNKNOWN:
            raise ProtocolError("PROTOCOL", f"{name}_PARTIAL_HISTORY_FORBIDDEN")
        return UNKNOWN, ()
    if binding.get("status") != BOUND:
        raise ProtocolError("PROTOCOL", f"{name}_STATUS_INVALID")
    if predecessor_mode != BOUND:
        raise ProtocolError("PROTOCOL", f"{name}_PREDECESSOR_NOT_BOUND")
    history = binding.get("append_only_history")
    if not isinstance(history, list) or not history:
        raise ProtocolError("PROTOCOL", f"{name}_BOUND_HISTORY_NOT_NONEMPTY_LIST")
    parent = expected_parent
    next_i_ordinal = 1
    last_i_ordinal: int | None = None
    previous_kind: str | None = None
    for index, raw_step in enumerate(history):
        step = _mapping(
            raw_step, "PROTOCOL", f"{name}_HISTORY_STEP_{index}_NOT_OBJECT"
        )
        if set(step) != set(HISTORY_STEP_FIELDS):
            raise ProtocolError("PROTOCOL", f"{name}_HISTORY_STEP_FIELDS_DIFFER")
        kind = step.get("kind")
        label = step.get("label")
        if kind not in {"I", "B"} or not isinstance(label, str):
            raise ProtocolError("PROTOCOL", f"{name}_HISTORY_LABEL_OR_KIND_INVALID")
        prefix = f"{name}_{kind}"
        ordinal_text = label.removeprefix(prefix)
        if not label.startswith(prefix) or not ordinal_text.isdigit():
            raise ProtocolError("PROTOCOL", f"{name}_HISTORY_LABEL_INVALID")
        ordinal = int(ordinal_text)
        if kind == "I":
            if ordinal != next_i_ordinal:
                raise ProtocolError("PROTOCOL", f"{name}_I_ORDINAL_NOT_APPEND_ONLY")
            next_i_ordinal += 1
            last_i_ordinal = ordinal
        elif previous_kind != "I" or ordinal != last_i_ordinal:
            raise ProtocolError("PROTOCOL", f"{name}_B_NOT_CONFIG_BINDING_FOR_PRIOR_I")
        commit = step.get("commit")
        if not _is_hex(commit, 40):
            raise ProtocolError("PROTOCOL", f"{name}_HISTORY_COMMIT_INVALID")
        if step.get("expected_parent") != parent:
            raise ProtocolError("PROTOCOL", f"{name}_HISTORY_DIRECT_PARENT_DIFFERS")
        expected_paths = list(exact3 if kind == "I" else (config_path,))
        if step.get("exact_changed_paths") != expected_paths:
            raise ProtocolError("PROTOCOL", f"{name}_{kind}_CHANGED_PATHS_DIFFER")
        _validate_sha_map(
            step.get("blob_sha256_by_path"), exact3, f"{name}_{label}_BLOBS"
        )
        parent = str(commit)
        previous_kind = str(kind)
    if history[0].get("kind") != "I" or history[-1].get("kind") != "B":
        raise ProtocolError("PROTOCOL", f"{name}_HISTORY_MUST_START_I_AND_END_B")
    labels = tuple(str(step["label"]) for step in history)
    commits = tuple(str(step["commit"]) for step in history)
    if expected_labels is not None and labels != tuple(expected_labels):
        raise ProtocolError("PROTOCOL", f"{name}_FROZEN_HISTORY_LABELS_DIFFER")
    if expected_commits is not None and commits != tuple(expected_commits):
        raise ProtocolError("PROTOCOL", f"{name}_FROZEN_HISTORY_COMMITS_DIFFER")
    return BOUND, tuple(history)


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
    if binding.get("status") == UNKNOWN:
        if dynamic != (UNKNOWN,) * len(OWN_BINDING_FIELDS):
            raise ProtocolError("PROTOCOL", "OWN_PARTIAL_GROUP_FORBIDDEN")
        return UNKNOWN
    if binding.get("status") != BOUND:
        raise ProtocolError("PROTOCOL", "OWN_STATUS_INVALID")
    if predecessor_mode != BOUND:
        raise ProtocolError("PROTOCOL", "OWN_PREDECESSOR_NOT_BOUND")
    if not _is_hex(binding.get("implementation_commit"), 40):
        raise ProtocolError("PROTOCOL", "OWN_IMPLEMENTATION_COMMIT_INVALID")
    for field in ("implementation_script_sha256", "implementation_test_sha256"):
        if not _is_hex(binding.get(field), 64):
            raise ProtocolError("PROTOCOL", f"{field.upper()}_INVALID")
    return BOUND


def _binding_modes(protocol: Mapping[str, Any]) -> dict[str, str]:
    bindings = _mapping(protocol.get("bindings"), "PROTOCOL", "BINDINGS_NOT_OBJECT")
    if set(bindings) != {
        "authority",
        "runtime",
        "gse217518_predecessor",
        "encsr854ruf_predecessor",
        "gse232572_predecessor",
        "implementation",
    }:
        raise ProtocolError("PROTOCOL", "BINDING_GROUP_CLOSURE_DIFFERS")
    for name, value in bindings.items():
        _mapping(value, "PROTOCOL", f"{name.upper()}_NOT_OBJECT")
    _validate_authority_binding(bindings["authority"])
    _validate_runtime_binding(bindings["runtime"])
    gse217_mode, _ = _history_binding_mode(
        bindings["gse217518_predecessor"],
        name="GSE217518",
        exact3=GSE217_EXACT3,
        config_path=GSE217_CONFIG_PATH,
        expected_parent=RUNTIME_B_COMMIT,
        predecessor_mode=BOUND,
        allow_unknown=False,
        expected_labels=GSE217_HISTORY_LABELS,
        expected_commits=GSE217_HISTORY_COMMITS,
    )
    encsr_mode, _ = _history_binding_mode(
        bindings["encsr854ruf_predecessor"],
        name="ENCSR854RUF",
        exact3=ENCSR_EXACT3,
        config_path=ENCSR_CONFIG_PATH,
        expected_parent=GSE217_HISTORY_COMMITS[-1],
        predecessor_mode=gse217_mode,
        allow_unknown=False,
        expected_labels=ENCSR_HISTORY_LABELS,
        expected_commits=ENCSR_HISTORY_COMMITS,
    )
    gse232_mode, _ = _history_binding_mode(
        bindings["gse232572_predecessor"],
        name="GSE232572",
        exact3=GSE232_EXACT3,
        config_path=GSE232_CONFIG_PATH,
        expected_parent=ENCSR_B4_COMMIT,
        predecessor_mode=encsr_mode,
        allow_unknown=True,
        expected_labels=GSE232_HISTORY_LABELS,
        expected_commits=GSE232_HISTORY_COMMITS,
    )
    own_mode = _implementation_binding_mode(
        bindings["implementation"], predecessor_mode=gse232_mode
    )
    return {
        "gse217518_predecessor": gse217_mode,
        "encsr854ruf_predecessor": encsr_mode,
        "gse232572_predecessor": gse232_mode,
        "implementation": own_mode,
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if set(protocol) != ROOT_KEYS:
        raise ProtocolError("PROTOCOL", "ROOT_KEY_CLOSURE_DIFFERS")
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "decision_id": "V3-DEC-027",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "study_id": DATASET_ID,
        "project_alias": "APARENT_DESIGNED_SNV_SUBSET",
        "protocol_status": "DRAFT_CANDIDATE_NOT_ACTIVE_PROTOCOL",
    }
    for key, expected in expected_scalars.items():
        if protocol.get(key) != expected:
            raise ProtocolError("PROTOCOL", f"{key.upper()}_NOT_FROZEN")

    snapshot = _mapping(
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
    contribution = _mapping(
        snapshot.get("gse113849_current_contribution"),
        "PROTOCOL",
        "CURRENT_CONTRIBUTION_NOT_OBJECT",
    )
    if dict(contribution) != {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }:
        raise ProtocolError("PROTOCOL", "GSE113849_CURRENT_CONTRIBUTION_NOT_ZERO")

    gates = protocol.get("required_fail_closed_gate_ids_exactly")
    if not isinstance(gates, list) or tuple(gates) != REQUIRED_GATE_IDS:
        raise ProtocolError("PROTOCOL", "EXACT_THIRTEEN_GATES_NOT_FROZEN")

    context = _mapping(
        protocol.get("outcome_blind_context_freeze"),
        "PROTOCOL",
        "CONTEXT_FREEZE_NOT_OBJECT",
    )
    if context.get("status") != "FROZEN_BEFORE_EXACT3_ENDPOINT_OR_POWER_EVALUATION":
        raise ProtocolError("PROTOCOL", "CONTEXT_NOT_FROZEN_BEFORE_ENDPOINT")
    if context.get("selected_rule") != SELECTED_CONTEXT_RULE:
        raise ProtocolError("PROTOCOL", "SELECTED_CONTEXT_RULE_CHANGED")
    if context.get("allowed_rules_exactly") != [SELECTED_CONTEXT_RULE, EXPLICIT_UTR3_RULE]:
        raise ProtocolError("PROTOCOL", "ALLOWED_CONTEXT_RULES_CHANGED")
    if context.get("selection_basis") != (
        "PUBLISHER_AND_GEO_ASSAY_CONTEXT_NOT_ENDPOINT_MAGNITUDE_DIRECTION_PVALUE_OR_POWER"
    ):
        raise ProtocolError("PROTOCOL", "CONTEXT_SELECTION_NOT_OUTCOME_BLIND")
    if context.get("outcome_or_power_may_change_selected_rule") is not False:
        raise ProtocolError("PROTOCOL", "OUTCOME_OR_POWER_MAY_CHANGE_CONTEXT")

    intended = _mapping(
        protocol.get("intended_universe"), "PROTOCOL", "INTENDED_UNIVERSE_NOT_OBJECT"
    )
    if intended.get("randomized_absolute_library") != "EXCLUDED_NOT_TRUE_A2":
        raise ProtocolError("PROTOCOL", "RANDOMIZED_LIBRARY_NOT_EXCLUDED")
    if intended.get("model_prediction_column_role") != (
        "NEVER_ENDPOINT_NEVER_ELIGIBILITY_NEVER_POWER"
    ):
        raise ProtocolError("PROTOCOL", "MODEL_PREDICTION_ROLE_NOT_FROZEN")

    table = _mapping(
        protocol.get("table_contract"), "PROTOCOL", "TABLE_CONTRACT_NOT_OBJECT"
    )
    if table.get("source_group_fields_exactly") != ["gene", "wt_seq"]:
        raise ProtocolError("PROTOCOL", "SOURCE_GROUP_FIELDS_NOT_FROZEN")
    if table.get("source_group_merge_rule") != (
        "IDENTICAL_SHORT_SOURCE_SEQUENCE_WITH_DIFFERENT_GENE_MAY_NOT_MERGE"
    ):
        raise ProtocolError("PROTOCOL", "CROSS_GENE_SOURCE_MERGE_NOT_FORBIDDEN")
    if table.get("minimum_distinct_candidates_per_dense_source") != 3:
        raise ProtocolError("PROTOCOL", "DENSE_MINIMUM_NOT_THREE")
    if table.get("prediction_field_may_be_read_for_gate_or_output") is not False:
        raise ProtocolError("PROTOCOL", "PREDICTION_FIELD_ENABLED")
    if table.get("pvalue_semantics") != (
        "POOLED_READ_COUNT_TWO_PROPORTION_Z_TEST_NOT_BIOLOGICAL_REPLICATE_STANDARD_ERROR"
    ):
        raise ProtocolError("PROTOCOL", "PVALUE_SEMANTICS_NOT_FROZEN")

    inputs = _mapping(
        protocol.get("ordinary_public_inputs"), "PROTOCOL", "PUBLIC_INPUTS_NOT_OBJECT"
    )
    if tuple(inputs) != EXPECTED_INPUT_NAMES:
        raise ProtocolError("PROTOCOL", "PUBLIC_INPUT_SET_NOT_EXACT")
    if {str(item.get("filename")) for item in inputs.values()} != set(
        PUBLIC_ASSET_IDENTITIES
    ):
        raise ProtocolError("PROTOCOL", "PUBLIC_ASSET_FILENAME_CLOSURE_DIFFERS")
    for value in inputs.values():
        contract = _mapping(value, "PROTOCOL", "PUBLIC_ASSET_CONTRACT_NOT_OBJECT")
        filename = str(contract.get("filename"))
        if (contract.get("bytes"), contract.get("sha256")) != (
            PUBLIC_ASSET_IDENTITIES[filename]
        ):
            raise ProtocolError("PROTOCOL", "PUBLIC_ASSET_IDENTITY_DIFFERS")

    authority_chain = _mapping(
        protocol.get("public_authority_chain"),
        "PROTOCOL",
        "PUBLIC_AUTHORITY_CHAIN_NOT_OBJECT",
    )
    article = _mapping(
        authority_chain.get("publisher_article"),
        "PROTOCOL",
        "PUBLISHER_ARTICLE_NOT_OBJECT",
    )
    sample = _mapping(
        authority_chain.get("designed_array_sample"),
        "PROTOCOL",
        "DESIGNED_ARRAY_SAMPLE_NOT_OBJECT",
    )
    repository_source = _mapping(
        authority_chain.get("author_repository"),
        "PROTOCOL",
        "AUTHOR_REPOSITORY_NOT_OBJECT",
    )
    if article.get("doi") != "10.1016/j.cell.2019.04.046":
        raise ProtocolError("PROTOCOL", "PUBLISHER_DOI_DIFFERS")
    if sample.get("accession") != "GSM3780566" or sample.get("cell_type") != (
        "HEK293T"
    ):
        raise ProtocolError("PROTOCOL", "DESIGNED_REPORTER_CONTEXT_DIFFERS")
    if repository_source.get("commit") != (
        "cea9ab754fbc4152ae77abb4dd82d898de872f0f"
    ):
        raise ProtocolError("PROTOCOL", "AUTHOR_REPOSITORY_COMMIT_DIFFERS")

    execution = _mapping(
        protocol.get("execution_boundary"), "PROTOCOL", "EXECUTION_BOUNDARY_NOT_OBJECT"
    )
    false_keys = (
        "private_or_sealed_access_allowed",
        "persistent_member_level_intermediate_allowed",
        "row_or_member_identifier_output_allowed",
        "sequence_output_allowed",
        "row_endpoint_effect_pvalue_or_standard_error_output_allowed",
        "split_assignment_output_allowed",
        "canonical_materialization_allowed",
        "training_allowed",
        "gpu_work_allowed",
        "model_selection_allowed",
        "a7_allowed",
        "next_phase_allowed",
    )
    if execution.get("ordinary_public_inputs_only") is not True:
        raise ProtocolError("PROTOCOL", "ORDINARY_PUBLIC_SCOPE_NOT_FROZEN")
    if any(execution.get(key) is not False for key in false_keys):
        raise ProtocolError("PROTOCOL", "PROHIBITED_CAPABILITY_ENABLED")
    if execution.get("production_mode") != (
        "SINGLE_BOUND_PRODUCTION_ENTRY_FIXED_BUILT_IN_PUBLIC_READER_ONLY"
    ):
        raise ProtocolError("PROTOCOL", "PRODUCTION_FAIL_ORDER_NOT_FROZEN")
    if execution.get("ordinary_public_analysis_mode") != (
        "NO_SEPARATE_PUBLIC_ANALYSIS_BYPASS"
    ):
        raise ProtocolError("PROTOCOL", "PUBLIC_ANALYSIS_MODE_NOT_FROZEN")

    repository = _mapping(
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

    activation = _mapping(
        protocol.get("production_activation_rule"),
        "PROTOCOL",
        "PRODUCTION_ACTIVATION_RULE_NOT_OBJECT",
    )
    if activation.get("required_commit_chain") != (
        "DEC027_A_TO_RUNTIME_I1_TO_RUNTIME_I2_TO_RUNTIME_B2_"
        "TO_GSE217518_I1_TO_GSE217518_I2_TO_GSE217518_B2_"
        "TO_GSE217518_I3_TO_GSE217518_B3_"
        "TO_ENCSR854RUF_I1_TO_ENCSR854RUF_I2_TO_ENCSR854RUF_B2_"
        "TO_ENCSR854RUF_I3_TO_ENCSR854RUF_B3_TO_ENCSR854RUF_I4_"
        "TO_ENCSR854RUF_B4_TO_GSE232572_COMPLETE_APPEND_ONLY_I_B_HISTORY_"
        "TO_GSE113849_I_TO_GSE113849_B"
    ):
        raise ProtocolError("PROTOCOL", "PRODUCTION_COMMIT_CHAIN_DIFFERS")
    if activation.get("predecessor_order") != [
        "gse217518_predecessor",
        "encsr854ruf_predecessor",
        "gse232572_predecessor",
    ]:
        raise ProtocolError("PROTOCOL", "PRODUCTION_PREDECESSOR_ORDER_DIFFERS")
    for key in (
        "all_binding_groups_must_be_bound",
        "gse113849_implementation_i_must_be_direct_child_of_gse232572_b",
        "clean_head_equals_upstream_equals_live_origin_required",
        "direct_parent_changed_path_and_blob_audit_required",
        "executing_script_and_focused_test_must_match_implementation_i",
        "binding_commit_may_change_only_the_four_own_binding_scalars",
        "fail_before_git_asset_or_output_while_any_predecessor_or_own_group_is_unknown",
    ):
        if activation.get(key) is not True:
            raise ProtocolError("PROTOCOL", f"{key.upper()}_NOT_FROZEN")

    split_power = _mapping(
        protocol.get("source_group_split_and_power_boundary"),
        "PROTOCOL",
        "SPLIT_POWER_BOUNDARY_NOT_OBJECT",
    )
    if split_power.get("analysis_unit") != "INDEPENDENT_POST_DEDUP_SOURCE_GROUP":
        raise ProtocolError("PROTOCOL", "POWER_ANALYSIS_UNIT_NOT_FROZEN")
    if split_power.get("formal_split_execution_allowed") is not False:
        raise ProtocolError("PROTOCOL", "FORMAL_SPLIT_EXECUTION_ENABLED")
    if split_power.get("formal_power_gate_execution_allowed") is not False:
        raise ProtocolError("PROTOCOL", "FORMAL_POWER_EXECUTION_ENABLED")
    if split_power.get("near_duplicate_graph_rule_status") != UNKNOWN:
        raise ProtocolError("PROTOCOL", "NEAR_DUPLICATE_RULE_PREJUDGED")
    if split_power.get("post_near_duplicate_effective_n_status") != UNKNOWN:
        raise ProtocolError("PROTOCOL", "POST_DEDUP_N_PREJUDGED")
    if split_power.get("prefrozen_required_effective_n_reference") != 156:
        raise ProtocolError("PROTOCOL", "REQUIRED_EFFECTIVE_N_NOT_156")

    uncertainty = _mapping(
        protocol.get("replicate_and_uncertainty_boundary"),
        "PROTOCOL",
        "UNCERTAINTY_BOUNDARY_NOT_OBJECT",
    )
    if uncertainty.get("published_biological_replicate_count") is not None:
        raise ProtocolError("PROTOCOL", "UNEXPECTED_UNCERTAINTY_FIELD")
    if uncertainty.get("delta_pvalue_may_substitute_for_replicate_standard_error") is not False:
        raise ProtocolError("PROTOCOL", "PVALUE_ALLOWED_AS_STANDARD_ERROR")
    if uncertainty.get("row_level_replicate_derived_standard_error_in_bound_table") is not False:
        raise ProtocolError("PROTOCOL", "ROW_STANDARD_ERROR_PREJUDGED_PRESENT")

    output = _mapping(
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
    if any(mode != BOUND for mode in modes.values()):
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
        own[field] = UNKNOWN
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
    """Audit the complete A/runtime/predecessor/GSE113 direct-parent history."""

    _require_production_bindings(protocol)
    repository = _mapping(
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
    gse232 = bindings["gse232572_predecessor"]
    own = bindings["implementation"]
    gse217_history = tuple(gse217["append_only_history"])
    encsr_history = tuple(encsr["append_only_history"])
    gse232_history = tuple(gse232["append_only_history"])
    gse217_b = str(gse217_history[-1]["commit"])
    encsr_b = str(encsr_history[-1]["commit"])
    gse232_b = str(gse232_history[-1]["commit"])
    own_i = str(own["implementation_commit"])

    chain: tuple[
        tuple[str, str, str, Sequence[str], Mapping[str, str] | None], ...
    ] = (
        ("DEC027_AUTHORITY_A", AUTHORITY_COMMIT, AUTHORITY_PARENT, AUTHORITY_EXACT12, authority["authority_blob_sha256_by_path"]),
        ("DEC027_RUNTIME_I1", RUNTIME_I1_COMMIT, AUTHORITY_COMMIT, RUNTIME_EXACT3, runtime["frozen_i1_blob_sha256_by_path"]),
        ("DEC027_RUNTIME_I2", RUNTIME_I2_COMMIT, RUNTIME_I1_COMMIT, RUNTIME_EXACT3, runtime["implementation_blob_sha256_by_path"]),
        ("DEC027_RUNTIME_B2", RUNTIME_B_COMMIT, RUNTIME_I2_COMMIT, (RUNTIME_CONFIG_PATH,), runtime["binding_blob_sha256_by_path"]),
    )
    for history in (gse217_history, encsr_history, gse232_history):
        chain += tuple(
            (
                str(step["label"]),
                str(step["commit"]),
                str(step["expected_parent"]),
                tuple(step["exact_changed_paths"]),
                step["blob_sha256_by_path"],
            )
            for step in history
        )
    chain += (
        (
            "GSE113849_I",
            own_i,
            gse232_b,
            EXACT3,
            {
                SCRIPT_REPO_PATH: own["implementation_script_sha256"],
                TEST_REPO_PATH: own["implementation_test_sha256"],
            },
        ),
        ("GSE113849_B", head, own_i, (CONFIG_REPO_PATH,), None),
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
        "gse217518_binding_commit": gse217_b,
        "encsr854ruf_binding_commit": encsr_b,
        "gse232572_binding_commit": gse232_b,
        "implementation_commit": own_i,
        "binding_commit": head,
    }


def _reject_nonpublic_asset(path: Path, label: str) -> None:
    basename = path.name.lower()
    if any(fragment in basename for fragment in FORBIDDEN_ASSET_BASENAME_FRAGMENTS):
        raise AssetError("PUBLIC_ASSET_SCOPE", f"{label.upper()}_NOT_ORDINARY_PUBLIC")


def _verify_asset(path: Path, contract: Mapping[str, Any], label: str) -> None:
    _reject_nonpublic_asset(path, label)
    if path.name != contract.get("filename"):
        raise AssetError("ASSET_IDENTITY", f"{label.upper()}_BASENAME_MISMATCH")
    if not path.is_file():
        raise AssetError("ASSET_IDENTITY", f"{label.upper()}_MISSING")
    expected_bytes = contract.get("bytes")
    expected_sha256 = contract.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ProtocolError("PROTOCOL", f"{label.upper()}_BYTES_NOT_FROZEN")
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


def _notebook_source(path: Path, label: str) -> str:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssetError("AUTHOR_LINEAGE", f"{label.upper()}_NOTEBOOK_NOT_READABLE") from exc
    if not isinstance(document, Mapping) or not isinstance(document.get("cells"), list):
        raise AssetError("AUTHOR_LINEAGE", f"{label.upper()}_NOTEBOOK_SHAPE_INVALID")
    sources: list[str] = []
    for cell in document["cells"]:
        if not isinstance(cell, Mapping) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(item, str) for item in source):
            sources.extend(source)
    if not sources:
        raise AssetError("AUTHOR_LINEAGE", f"{label.upper()}_HAS_NO_CODE_SOURCE")
    return "".join(sources)


def _require_tokens(text: str, tokens: Sequence[str], gate: str, code: str) -> None:
    if any(token not in text for token in tokens):
        raise ReplayInvariantError(gate, code)


def _validate_author_semantics(asset_paths: Mapping[str, Path]) -> dict[str, Any]:
    generator = _notebook_source(
        asset_paths["table_generator_notebook"], "table_generator"
    )
    _require_tokens(
        generator,
        (
            "included_experiments = ['acmg_apadb', 'acmg_polyadb', 'sensitive_genes', 'clinvar_wt', 'human_variant']",
            "human_variant_df = variant_df.query(filter_query).copy()",
            "human_variant_df.query(\"variant == 'snv'\")",
            "df_all_snvs.to_csv('predictions/apa_variant_mpra_all_snvs.csv'",
            "Collapsed over experiment replicates",
        ),
        "AUTHOR_LINEAGE",
        "TABLE_GENERATOR_SEMANTICS_NOT_CLOSED",
    )

    preparation = _notebook_source(
        asset_paths["data_preparation_notebook"], "data_preparation"
    )
    _require_tokens(
        preparation,
        (
            "seq_df_delta['delta_logodds_true'] = seq_df_delta['pooled_proximal_logodds_var'] - seq_df_delta['pooled_proximal_logodds_ref']",
            "def differential_prop_test(count_1, total_count_1, count_2, total_count_2)",
            "p_val = 2. * z_rv.sf(z_abs)",
            "seq_df_delta['delta_p_val'] = delta_p_vals",
            "array_version_var == array_version_ref",
        ),
        "AUTHOR_ENDPOINT",
        "ENDPOINT_OR_PVALUE_GENERATOR_SEMANTICS_NOT_CLOSED",
    )

    replicate = _notebook_source(asset_paths["replicate_notebook"], "replicate")
    _require_tokens(
        replicate,
        (
            "Biological replicate 1",
            "Biological replicate 2",
            "array_version == 'lofi'",
            "array_version == 'hifi'",
        ),
        "AUTHOR_REPLICATE",
        "TWO_BIOLOGICAL_REPLICATE_LABELS_NOT_CLOSED",
    )

    try:
        readme = asset_paths["repository_readme"].read_text(encoding="utf-8")
        license_text = asset_paths["repository_license"].read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AssetError("AUTHOR_LINEAGE", "README_OR_LICENSE_NOT_READABLE") from exc
    _require_tokens(
        readme,
        ("## Data Availability", "## Designed MPRA Analysis Notebooks"),
        "AUTHOR_LINEAGE",
        "README_DATA_ROUTE_NOT_CLOSED",
    )
    _require_tokens(
        license_text,
        ("software and associated documentation files", "Permission is hereby granted"),
        "RIGHTS",
        "SOFTWARE_LICENSE_SCOPE_NOT_READABLE",
    )
    return {
        "author_table_generator_filter_closed": True,
        "randomized_absolute_library_excluded_by_generator": True,
        "author_endpoint_transform_closed": True,
        "author_pvalue_is_pooled_count_z_test": True,
        "author_biological_replicate_label_count": 2,
        "repository_license_scope": "SOFTWARE_AND_ASSOCIATED_DOCUMENTATION",
    }


def _read_table(path: Path, protocol: Mapping[str, Any]) -> list[dict[str, str]]:
    table_contract = _mapping(
        protocol.get("table_contract"), "PROTOCOL", "TABLE_CONTRACT_NOT_OBJECT"
    )
    expected_header = table_contract.get("header_exactly")
    if not isinstance(expected_header, list):
        raise ProtocolError("PROTOCOL", "TABLE_HEADER_NOT_FROZEN")
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise AssetError("DESIGNED_SNV_TABLE", "TABLE_NOT_READABLE") from exc
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected_header:
            raise ReplayInvariantError("DESIGNED_SNV_TABLE", "TABLE_HEADER_MISMATCH")
        rows = []
        try:
            for row in reader:
                if None in row:
                    raise ReplayInvariantError(
                        "DESIGNED_SNV_TABLE", "TABLE_ROW_HAS_EXTRA_FIELDS"
                    )
                rows.append({str(key): str(value) for key, value in row.items()})
        except (csv.Error, UnicodeError) as exc:
            raise ReplayInvariantError(
                "DESIGNED_SNV_TABLE", "TABLE_PARSE_FAILED"
            ) from exc
    return rows


def _freeze_context_universe(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, str]]
) -> tuple[list[Mapping[str, str]], list[Mapping[str, str]]]:
    """Select context without reading endpoint, prediction, p-value, or power fields."""

    context = _mapping(
        protocol.get("outcome_blind_context_freeze"),
        "PROTOCOL",
        "CONTEXT_FREEZE_NOT_OBJECT",
    )
    if context.get("selected_rule") != SELECTED_CONTEXT_RULE:
        raise ProtocolError("CONTEXT", "SELECTED_CONTEXT_RULE_NOT_FROZEN")
    selected = list(rows)
    explicit_utr3 = [row for row in rows if row.get("sitetype") == "UTR3"]
    return selected, explicit_utr3


def _pool_bin(count: int) -> str:
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    if count <= 25:
        return "11-25"
    if count <= 50:
        return "26-50"
    if count <= 100:
        return "51-100"
    if count <= 200:
        return "101-200"
    return ">200"


def _geometry(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    source_to_candidates: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    source_sequence_to_genes: dict[str, set[str]] = collections.defaultdict(set)
    source_candidate_pairs: set[tuple[str, str]] = set()
    gene_count_values: set[str] = set()
    equal_length_count = 0
    hamming_one_count = 0
    position_consistent_count = 0
    missing_required_count = 0
    alphabet_valid_count = 0
    sitetype_histogram: collections.Counter[str] = collections.Counter()
    edit_position_bins: collections.Counter[str] = collections.Counter()

    for row in rows:
        gene = row.get("gene", "").strip()
        source = row.get("wt_seq", "").strip().upper()
        candidate = row.get("master_seq", "").strip().upper()
        endpoint = row.get("delta_logodds_true", "").strip()
        pvalue = row.get("delta_p_val", "").strip()
        if not source or not candidate or not endpoint or not pvalue:
            missing_required_count += 1
            continue
        if not gene:
            raise ReplayInvariantError("SOURCE_IDENTITY", "EMPTY_GENE_GROUP_FIELD")
        if set(source) <= set("ACGT") and set(candidate) <= set("ACGT"):
            alphabet_valid_count += 1
        else:
            raise ReplayInvariantError("EDIT_REPLAY", "SEQUENCE_ALPHABET_NOT_ACGT")
        if len(source) == len(candidate):
            equal_length_count += 1
        else:
            raise ReplayInvariantError("EDIT_REPLAY", "SOURCE_CANDIDATE_LENGTH_MISMATCH")
        differences = [
            index for index, (left, right) in enumerate(zip(source, candidate)) if left != right
        ]
        if len(differences) != 1:
            raise ReplayInvariantError("EDIT_REPLAY", "PAIR_NOT_EXACT_HAMMING_ONE")
        hamming_one_count += 1
        try:
            snv_position = int(row.get("snv_pos", ""))
        except ValueError as exc:
            raise ReplayInvariantError("EDIT_REPLAY", "SNV_POSITION_NOT_INTEGER") from exc
        if snv_position != differences[0]:
            raise ReplayInvariantError("EDIT_REPLAY", "SNV_POSITION_DISAGREES_WITH_SEQUENCE")
        position_consistent_count += 1
        if snv_position < 50:
            edit_position_bins["0-49"] += 1
        elif snv_position < 56:
            edit_position_bins["50-55"] += 1
        elif snv_position < 100:
            edit_position_bins["56-99"] += 1
        elif snv_position < 150:
            edit_position_bins["100-149"] += 1
        else:
            edit_position_bins[">=150"] += 1

        source_group = (gene, source)
        source_to_candidates[source_group].add(candidate)
        source_sequence_to_genes[source].add(gene)
        source_candidate_pairs.add((source, candidate))
        gene_count_values.add(gene)
        sitetype_histogram[row.get("sitetype", "").strip() or "EMPTY"] += 1

    pool_sizes = {group: len(candidates) for group, candidates in source_to_candidates.items()}
    dense_sizes = {group: size for group, size in pool_sizes.items() if size >= 3}
    pool_histogram: collections.Counter[str] = collections.Counter(
        _pool_bin(size) for size in pool_sizes.values()
    )
    dense_histogram: collections.Counter[str] = collections.Counter(
        _pool_bin(size) for size in dense_sizes.values()
    )
    dense_gene_count = len({group[0] for group in dense_sizes})
    return {
        "row_count": len(rows),
        "distinct_source_sequence_count": len(source_sequence_to_genes),
        "distinct_gene_source_group_count": len(source_to_candidates),
        "source_sequences_assigned_to_multiple_genes_count": sum(
            len(genes) > 1 for genes in source_sequence_to_genes.values()
        ),
        "distinct_source_candidate_pair_count": len(source_candidate_pairs),
        "equal_length_pair_count": equal_length_count,
        "exact_hamming_one_pair_count": hamming_one_count,
        "snv_position_consistent_pair_count": position_consistent_count,
        "alphabet_valid_pair_count": alphabet_valid_count,
        "missing_required_source_candidate_endpoint_pvalue_count": missing_required_count,
        "nominal_gene_count": len(gene_count_values),
        "dense_gene_source_group_count": len(dense_sizes),
        "dense_gene_count": dense_gene_count,
        "rows_in_dense_gene_source_groups": sum(dense_sizes.values()),
        "maximum_dense_candidate_pool_size": max(dense_sizes.values(), default=0),
        "candidate_pool_size_histogram": dict(sorted(pool_histogram.items())),
        "dense_candidate_pool_size_histogram": dict(sorted(dense_histogram.items())),
        "sitetype_histogram": dict(sorted(sitetype_histogram.items())),
        "edit_position_bin_histogram": dict(sorted(edit_position_bins.items())),
    }


def _endpoint_aggregates(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Interpret endpoint fields only after the outcome-blind context freeze."""

    sign_histogram = {"negative": 0, "zero": 0, "positive": 0}
    magnitude_histogram = {
        "[0,0.1)": 0,
        "[0.1,0.25)": 0,
        "[0.25,0.5)": 0,
        "[0.5,1.0)": 0,
        ">=1.0": 0,
    }
    pvalue_histogram = {
        "0": 0,
        "(0,1e-6]": 0,
        "(1e-6,0.01]": 0,
        "(0.01,0.05]": 0,
        "(0.05,1]": 0,
    }
    finite_endpoint_count = 0
    valid_pvalue_count = 0
    for row in rows:
        try:
            endpoint = float(row["delta_logodds_true"])
            pvalue = float(row["delta_p_val"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayInvariantError("ENDPOINT", "ENDPOINT_OR_PVALUE_NOT_NUMERIC") from exc
        if not math.isfinite(endpoint):
            raise ReplayInvariantError("ENDPOINT", "ENDPOINT_NOT_FINITE")
        if not math.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
            raise ReplayInvariantError("ENDPOINT", "PVALUE_NOT_FINITE_OR_OUT_OF_RANGE")
        finite_endpoint_count += 1
        valid_pvalue_count += 1
        if endpoint > 0:
            sign_histogram["positive"] += 1
        elif endpoint < 0:
            sign_histogram["negative"] += 1
        else:
            sign_histogram["zero"] += 1
        magnitude = abs(endpoint)
        if magnitude < 0.1:
            magnitude_histogram["[0,0.1)"] += 1
        elif magnitude < 0.25:
            magnitude_histogram["[0.1,0.25)"] += 1
        elif magnitude < 0.5:
            magnitude_histogram["[0.25,0.5)"] += 1
        elif magnitude < 1.0:
            magnitude_histogram["[0.5,1.0)"] += 1
        else:
            magnitude_histogram[">=1.0"] += 1
        if pvalue == 0:
            pvalue_histogram["0"] += 1
        elif pvalue <= 1e-6:
            pvalue_histogram["(0,1e-6]"] += 1
        elif pvalue <= 0.01:
            pvalue_histogram["(1e-6,0.01]"] += 1
        elif pvalue <= 0.05:
            pvalue_histogram["(0.01,0.05]"] += 1
        else:
            pvalue_histogram["(0.05,1]"] += 1
    return {
        "finite_endpoint_count": finite_endpoint_count,
        "valid_pvalue_count": valid_pvalue_count,
        "endpoint_sign_histogram": sign_histogram,
        "absolute_endpoint_magnitude_histogram": magnitude_histogram,
        "pooled_count_pvalue_histogram": pvalue_histogram,
        "replicate_derived_standard_error_field_count": 0,
        "prediction_field_read_or_used": False,
    }


def _assert_expected(
    protocol: Mapping[str, Any], full: Mapping[str, Any], utr3: Mapping[str, Any], endpoint: Mapping[str, Any]
) -> None:
    expected = _mapping(
        protocol.get("expected_mechanical_replay"),
        "PROTOCOL",
        "EXPECTED_REPLAY_NOT_OBJECT",
    )
    comparisons = {
        "table_row_count": full["row_count"],
        "distinct_source_sequence_count": full["distinct_source_sequence_count"],
        "distinct_gene_source_group_count": full["distinct_gene_source_group_count"],
        "source_sequences_assigned_to_multiple_genes_count": full[
            "source_sequences_assigned_to_multiple_genes_count"
        ],
        "distinct_source_candidate_pair_count": full[
            "distinct_source_candidate_pair_count"
        ],
        "equal_length_pair_count": full["equal_length_pair_count"],
        "exact_hamming_one_pair_count": full["exact_hamming_one_pair_count"],
        "snv_position_consistent_pair_count": full[
            "snv_position_consistent_pair_count"
        ],
        "missing_required_source_candidate_endpoint_pvalue_count": full[
            "missing_required_source_candidate_endpoint_pvalue_count"
        ],
        "finite_endpoint_count": endpoint["finite_endpoint_count"],
        "valid_pvalue_count": endpoint["valid_pvalue_count"],
        "dense_gene_source_group_count": full["dense_gene_source_group_count"],
        "rows_in_dense_gene_source_groups": full["rows_in_dense_gene_source_groups"],
        "maximum_dense_candidate_pool_size": full["maximum_dense_candidate_pool_size"],
        "explicit_utr3_row_count": utr3["row_count"],
        "explicit_utr3_gene_source_group_count": utr3[
            "distinct_gene_source_group_count"
        ],
        "explicit_utr3_dense_gene_source_group_count": utr3[
            "dense_gene_source_group_count"
        ],
        "explicit_utr3_rows_in_dense_gene_source_groups": utr3[
            "rows_in_dense_gene_source_groups"
        ],
        "explicit_utr3_maximum_candidate_pool_size": utr3[
            "maximum_dense_candidate_pool_size"
        ],
        "published_biological_replicate_count": 2,
        "replicate_derived_standard_error_field_count": endpoint[
            "replicate_derived_standard_error_field_count"
        ],
    }
    for key, actual in comparisons.items():
        if expected.get(key) != actual:
            raise ReplayInvariantError("EXPECTED_REPLAY", f"{key.upper()}_MISMATCH")


def _gate(
    gate_id: str,
    status: str,
    reason_code: str,
    fact_class: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "gate_id": gate_id,
        "status": status,
        "reason_code": reason_code,
        "fact_class": fact_class,
        "blocks_current_qualification": status != "PASS",
    }
    if evidence is not None:
        row["aggregate_evidence"] = dict(evidence)
    return row


def _build_report(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    semantics: Mapping[str, Any],
    recorded_at: str,
) -> dict[str, Any]:
    # Protocol validation freezes context before this function can interpret endpoints.
    selected_rows, explicit_utr3_rows = _freeze_context_universe(protocol, rows)
    full_geometry = _geometry(selected_rows)
    utr3_geometry = _geometry(explicit_utr3_rows)
    endpoint = _endpoint_aggregates(selected_rows)
    _assert_expected(protocol, full_geometry, utr3_geometry, endpoint)

    split_power = _mapping(
        protocol.get("source_group_split_and_power_boundary"),
        "PROTOCOL",
        "SPLIT_POWER_NOT_OBJECT",
    )
    nominal_n = int(full_geometry["dense_gene_source_group_count"])
    required_n = int(split_power["prefrozen_required_effective_n_reference"])
    nominal_headroom = nominal_n - required_n
    if nominal_n >= required_n:
        reachability_verdict = (
            "POTENTIALLY_REACHABLE_UNDER_FROZEN_REPORTER_CONTEXT_BUT_NOT_ESTABLISHED_"
            "POST_DEDUP_N_UNKNOWN_AND_REPLICATE_SE_GATE_FAILS"
        )
    else:
        reachability_verdict = (
            "NOT_NOMINALLY_REACHABLE_UNDER_FROZEN_CONTEXT_AND_FORMAL_POWER_NOT_RUN"
        )
    reachability = {
        "selected_context_rule": SELECTED_CONTEXT_RULE,
        "nominal_dense_source_group_count": nominal_n,
        "prefrozen_required_effective_n_reference": required_n,
        "nominal_headroom_before_near_duplicate_dedup": nominal_headroom,
        "post_near_duplicate_effective_n": UNKNOWN,
        "explicit_utr3_nonselected_sensitivity_dense_source_group_count": int(
            utr3_geometry["dense_gene_source_group_count"]
        ),
        "explicit_utr3_nonselected_sensitivity_gap_to_reference": int(
            utr3_geometry["dense_gene_source_group_count"]
        )
        - required_n,
        "formal_power_gate_executed": False,
        "formal_full_ci_width_gate_executed": False,
        "valid_replicate_derived_standard_error_route": False,
        "verdict": reachability_verdict,
    }

    gates = [
        _gate(
            REQUIRED_GATE_IDS[0],
            "PARTIAL_OR_CONDITIONAL",
            "COMMIT_BOUND_AUTHOR_TABLE_GENERATOR_AND_GEO_STUDY_ROUTE_CLOSED_BUT_GEO_TO_PREPARED_TABLE_ROW_LINEAGE_AND_REJECT_CLOSURE_UNKNOWN",
            "MIXED_CONFIRMED_AND_UNKNOWN",
            {
                "ordinary_public_commit_bound_input_count": 6,
                "author_table_generator_filter_closed": True,
                "randomized_absolute_library_excluded": True,
                "geo_to_prepared_table_row_crosswalk_status": UNKNOWN,
                "generator_prediction_inner_join_reject_closure_status": UNKNOWN,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[1],
            "PASS",
            "EXPLICIT_GENE_SOURCE_CANDIDATE_MAPPING_CLOSED_WITH_CROSS_GENE_SOURCE_SEQUENCE_MERGE_FORBIDDEN",
            "CONFIRMED_FACT",
            {
                "distinct_source_sequence_count": full_geometry[
                    "distinct_source_sequence_count"
                ],
                "distinct_gene_source_group_count": full_geometry[
                    "distinct_gene_source_group_count"
                ],
                "source_sequences_assigned_to_multiple_genes_count": full_geometry[
                    "source_sequences_assigned_to_multiple_genes_count"
                ],
                "distinct_source_candidate_pair_count": full_geometry[
                    "distinct_source_candidate_pair_count"
                ],
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[2],
            "PASS",
            "ALL_BOUND_DESIGNED_SNV_PAIRS_EQUAL_LENGTH_EXACT_HAMMING_ONE_AND_POSITION_CONSISTENT",
            "CONFIRMED_FACT",
            {
                "equal_length_pair_count": full_geometry["equal_length_pair_count"],
                "exact_hamming_one_pair_count": full_geometry[
                    "exact_hamming_one_pair_count"
                ],
                "snv_position_consistent_pair_count": full_geometry[
                    "snv_position_consistent_pair_count"
                ],
                "edit_position_bin_histogram": full_geometry[
                    "edit_position_bin_histogram"
                ],
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[3],
            "PASS",
            "PUBLISHER_GEO_REPORTER_CONTEXT_RULE_FROZEN_BEFORE_EXACT3_ENDPOINT_OR_POWER_EVALUATION",
            "CONFIRMED_FACT",
            {
                "selected_context_rule": SELECTED_CONTEXT_RULE,
                "selected_row_count": full_geometry["row_count"],
                "randomized_absolute_library_included": False,
                "sitetype_histogram": full_geometry["sitetype_histogram"],
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[4],
            "PASS",
            "MINIMUM_THREE_DISTINCT_MEASURED_CANDIDATES_CLOSED_FOR_SELECTED_DENSE_GENE_SOURCE_GROUPS",
            "CONFIRMED_FACT",
            {
                "dense_gene_source_group_count": full_geometry[
                    "dense_gene_source_group_count"
                ],
                "rows_in_dense_gene_source_groups": full_geometry[
                    "rows_in_dense_gene_source_groups"
                ],
                "maximum_dense_candidate_pool_size": full_geometry[
                    "maximum_dense_candidate_pool_size"
                ],
                "candidate_pool_size_histogram": full_geometry[
                    "candidate_pool_size_histogram"
                ],
                "dense_candidate_pool_size_histogram": full_geometry[
                    "dense_candidate_pool_size_histogram"
                ],
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[5],
            "PASS",
            "MEASURED_POOLED_PROXIMAL_ISOFORM_LOGODDS_CANDIDATE_MINUS_SOURCE_DIRECTION_AND_SCALE_CLOSED",
            "CONFIRMED_FACT",
            {
                "finite_endpoint_count": endpoint["finite_endpoint_count"],
                "endpoint_sign_histogram": endpoint["endpoint_sign_histogram"],
                "absolute_endpoint_magnitude_histogram": endpoint[
                    "absolute_endpoint_magnitude_histogram"
                ],
                "prediction_field_read_or_used": False,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[6],
            "FAIL",
            "BOUND_TABLE_COLLAPSES_TWO_AUTHOR_LABELED_BIOLOGICAL_REPLICATES_AND_HAS_NO_REPLICATE_ENDPOINT_OR_REPLICATE_DERIVED_STANDARD_ERROR_DELTA_PVALUE_IS_POOLED_COUNT_Z_TEST",
            "CONFIRMED_FACT",
            {
                "author_labeled_biological_replicate_count": 2,
                "row_level_replicate_endpoint_fields_present": False,
                "replicate_derived_standard_error_field_count": 0,
                "delta_pvalue_role": (
                    "POOLED_READ_COUNT_TWO_PROPORTION_Z_TEST_NOT_REPLICATE_STANDARD_ERROR"
                ),
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[7],
            "PARTIAL_OR_CONDITIONAL",
            "BOUND_TABLE_REQUIRED_FIELDS_AND_FINITE_ENDPOINTS_CLOSED_BUT_UPSTREAM_GEO_PREPARED_DATA_REJECT_AND_PREDICTION_INNER_JOIN_SELECTION_CLOSURE_UNKNOWN",
            "MIXED_CONFIRMED_AND_UNKNOWN",
            {
                "table_row_count": full_geometry["row_count"],
                "missing_required_source_candidate_endpoint_pvalue_count": full_geometry[
                    "missing_required_source_candidate_endpoint_pvalue_count"
                ],
                "finite_endpoint_count": endpoint["finite_endpoint_count"],
                "valid_pvalue_count": endpoint["valid_pvalue_count"],
                "generator_prediction_inner_join_reject_closure_status": UNKNOWN,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[8],
            UNKNOWN,
            "REPOSITORY_MIT_NOTICE_COVERS_SOFTWARE_AND_ASSOCIATED_DOCUMENTATION_AND_GEO_PUBLIC_READABILITY_DOES_NOT_CLOSE_DATASET_QUALIFICATION_REUSE_OR_RELEASE_RIGHTS",
            UNKNOWN,
            {
                "ordinary_public_retrieval_closed": True,
                "repository_license_scope": semantics["repository_license_scope"],
                "dataset_specific_reuse_and_release_rights": UNKNOWN,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[9],
            UNKNOWN,
            "AUTHOR_TABLE_CONTAINS_APARENT_PREDICTIONS_EXCLUDED_HERE_BUT_PROJECT_HISTORICAL_ANALYTIC_AND_CHECKPOINT_EXPOSURE_NOT_AUDITED",
            UNKNOWN,
            {
                "author_prediction_field_present": True,
                "author_prediction_field_used_by_preflight": False,
                "project_historical_analytic_use": UNKNOWN,
                "checkpoint_exact_or_near_exposure": UNKNOWN,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[10],
            UNKNOWN,
            "GENE_PLUS_SOURCE_GROUP_RULE_FROZEN_BUT_NEAR_DUPLICATE_GRAPH_SPLIT_SALT_AND_ZERO_LEAKAGE_AUDIT_NOT_FROZEN_OR_EXECUTED",
            UNKNOWN,
            {
                "source_group_rule": "GENE_PLUS_WT_SEQUENCE",
                "near_duplicate_graph_rule_status": UNKNOWN,
                "split_salt_status": UNKNOWN,
                "formal_split_executed": False,
                "split_assignment_output_count": 0,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[11],
            UNKNOWN,
            "NOMINAL_DENSE_SOURCE_GEOMETRY_HAS_HEADROOM_BUT_POST_NEAR_DUPLICATE_INDEPENDENT_EFFECTIVE_N_NOT_CLOSED",
            UNKNOWN,
            {
                "nominal_dense_source_group_count": nominal_n,
                "prefrozen_required_effective_n_reference": required_n,
                "nominal_headroom_before_near_duplicate_dedup": nominal_headroom,
                "post_near_duplicate_effective_n": UNKNOWN,
            },
        ),
        _gate(
            REQUIRED_GATE_IDS[12],
            "NOT_RUN",
            "FORMAL_POWER_AND_FULL_CI_WIDTH_EXECUTION_FORBIDDEN_UNTIL_POST_DEDUP_N_AND_VALID_UNCERTAINTY_ROUTE_CLOSE",
            "NOT_RUN",
            {
                "alternative_spearman_rho": 0.25,
                "alpha_two_sided": 0.05,
                "target_power_minimum": 0.8,
                "confidence_level": 0.95,
                "maximum_full_ci_width": 0.3,
                "formal_power_gate_executed": False,
                "reachability_verdict": reachability["verdict"],
            },
        ),
    ]
    if tuple(gate["gate_id"] for gate in gates) != REQUIRED_GATE_IDS:
        raise ReplayInvariantError("REPORT", "EXACT_THIRTEEN_GATE_ORDER_CHANGED")
    status_counts = collections.Counter(str(gate["status"]) for gate in gates)

    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "study_id": DATASET_ID,
        "recorded_at": recorded_at,
        "execution_mode": "ORDINARY_PUBLIC_ANALYSIS_STAGING_AGGREGATE_ONLY",
        "protocol_status": "DRAFT_CANDIDATE_NOT_ACTIVE_PROTOCOL",
        "terminal_status": "STOP_REMAINING_QUALIFICATION_GATES_NOT_CLOSED",
        "scientific_disposition": "NOT_QUALIFIED_EXTERNAL_PREFLIGHT_CANDIDATE_ONLY",
        "qualification_status": "NOT_QUALIFIED",
        "qualified": False,
        "active_registry_membership_changed": False,
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
            "gse232572_predecessor": protocol["bindings"][
                "gse232572_predecessor"
            ]["status"],
            "implementation": protocol["bindings"]["implementation"]["status"],
        },
        "randomized_absolute_library": {
            "status": "EXCLUDED_NOT_TRUE_A2",
            "included_row_count": 0,
            "may_contribute_true_a2": False,
        },
        "outcome_blind_context_rule": {
            "status": "FROZEN_BEFORE_EXACT3_ENDPOINT_OR_POWER_EVALUATION",
            "selected_rule": SELECTED_CONTEXT_RULE,
            "selection_used_endpoint_pvalue_or_power": False,
            "selected_row_count": full_geometry["row_count"],
        },
        "aggregate_selected_context_geometry": full_geometry,
        "aggregate_explicit_utr3_nonselected_sensitivity": utr3_geometry,
        "aggregate_endpoint_evidence": endpoint,
        "post_dedup_n_and_power_reachability": reachability,
        "scientific_gate_status_counts": dict(sorted(status_counts.items())),
        "scientific_gates": gates,
        "remaining_blockers": [
            "GEO_TO_COMMIT_BOUND_PREPARED_TABLE_ROW_LINEAGE_AND_REJECT_CLOSURE_PARTIAL",
            "INDEPENDENT_BIOLOGICAL_REPLICATE_DERIVED_STANDARD_ERROR_ROUTE_FAIL",
            "DATASET_SPECIFIC_QUALIFICATION_REUSE_AND_RELEASE_RIGHTS_UNKNOWN",
            "HISTORICAL_ANALYTIC_AND_CHECKPOINT_EXPOSURE_UNKNOWN",
            "NEAR_DUPLICATE_GRAPH_SPLIT_SALT_AND_ZERO_LEAKAGE_READINESS_UNKNOWN",
            "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_UNKNOWN",
            "FORMAL_POWER_AND_FULL_CI_WIDTH_NOT_RUN",
            "SEPARATE_EVIDENCE_BASED_PROMOTION_AUTHORITY_REQUIRED",
        ],
        "state_change": {
            "authority_changed": False,
            "registry_changed": False,
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
            "row_endpoint_effect_pvalue_count_output": 0,
            "row_standard_error_count_output": 0,
            "split_assignment_count_output": 0,
            "persistent_member_level_intermediate_count": 0,
        },
    }
    _assert_aggregate_only(report)
    return report


def _assert_aggregate_only(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_REPORT_KEYS:
                raise ReplayInvariantError(
                    "REPORT_BOUNDARY", "FORBIDDEN_MEMBER_PAYLOAD_KEY"
                )
            _assert_aggregate_only(child)
    elif isinstance(value, list):
        for child in value:
            _assert_aggregate_only(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ReplayInvariantError("REPORT_BOUNDARY", "NONFINITE_JSON_NUMBER")


def _load_and_replay(
    *, protocol: Mapping[str, Any], asset_paths: Mapping[str, Path]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    contracts = _mapping(
        protocol.get("ordinary_public_inputs"), "PROTOCOL", "PUBLIC_INPUTS_NOT_OBJECT"
    )
    if tuple(asset_paths) != EXPECTED_INPUT_NAMES:
        raise AssetError("PUBLIC_ASSET_SCOPE", "EXACT_SIX_PUBLIC_INPUTS_REQUIRED")
    for name in EXPECTED_INPUT_NAMES:
        _verify_asset(
            asset_paths[name],
            _mapping(contracts.get(name), "PROTOCOL", f"{name.upper()}_NOT_OBJECT"),
            name,
        )
    semantics = _validate_author_semantics(asset_paths)
    rows = _read_table(asset_paths["designed_snv_table"], protocol)
    return rows, semantics


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

    path = output_dir / REPORT_FILENAME
    directory_created = False
    temporary_path: Path | None = None
    try:
        if output_dir.exists():
            if not output_dir.is_dir():
                raise OutputError("OUTPUT", "OUTPUT_PATH_NOT_DIRECTORY")
            entries = list(output_dir.iterdir())
            if entries:
                if len(entries) == 1 and entries[0] == path:
                    if path.read_bytes() == payload:
                        return path
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
            os.link(temporary_path, path)
        except FileExistsError as exc:
            if path.read_bytes() == payload:
                temporary_path.unlink()
                temporary_path = None
                return path
            raise OutputError("OUTPUT", "DIFFERENT_REPORT_APPEARED") from exc
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(output_dir)
        if list(output_dir.iterdir()) != [path]:
            raise OutputError("OUTPUT", "SINGLE_FIXED_REPORT_CONTRACT_VIOLATED")
        return path
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
    protocol = _load_json(protocol_path)
    _validate_protocol(protocol)
    _require_production_bindings(protocol)
    binding_audit = _audit_repository_bindings(
        protocol, protocol_path, PRODUCTION_REPO_ROOT
    )
    contracts = protocol["ordinary_public_inputs"]
    asset_paths = {
        name: asset_dir / contracts[name]["filename"] for name in EXPECTED_INPUT_NAMES
    }
    rows, semantics = _load_and_replay(protocol=protocol, asset_paths=asset_paths)
    report = _build_report(protocol, rows, semantics, recorded_at)
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
                    "post_dedup_n_and_power_reachability": report[
                        "post_dedup_n_and_power_reachability"
                    ]["verdict"],
                },
                sort_keys=True,
            )
        )
        return 0
    except PreflightError as exc:
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
