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
SOURCE_CONTRACT_SHA256 = "3ba224de6277edd67387913cf1c83a5e1344e0ad44ef196db07d0772b45c4d79"
GOAL_PATH = "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md"
CONFIG_PATH = "configs/route_a_v3.yaml"
SUPERSESSION_PATH = "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml"
DECISION_LOG_PATH = "docs/execution/route_a_v3_decision_log.yaml"
REGISTRY_MANIFEST_PATH = "docs/execution/route_a_v3_registry_manifest.json"
A1_INTERIM_PATH = "docs/execution/route_a_v3_a1_interim.yaml"
EXPECTED_A1_INTERIM_SHA256 = "66b996c800af176fb04492396bba7fd7bf7cd15d20c8aa79744dafecefb27a5d"
SCIENTIFIC_M0_HISTORY_PATH = "docs/contracts/history/mrna_v2_readiness_audit_20260807.md"
SCIENTIFIC_M0_HISTORY_SHA256 = "a8eb4f49ede793a8eae2037db9f46f044056d37610ec92482666a8242a52fa30"
SEALED_GUARD_PATH = "scripts/route_a_v3/sealed_guard.py"
SEALED_RUNNER_PATH = "scripts/e0x/run_e0x_final.py"
VALIDATOR_PATH = "scripts/route_a_v3/validate_a0_bundle.py"

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

EXPECTED_DECISION_IDS = tuple(f"V3-DEC-{index:03d}" for index in range(1, 18))
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
}

# Canonical per-entry digests make the accepted prefix genuinely append-only.
# A future DEC-018 requires an explicit validator update; rewriting any accepted
# DEC-001..017 entry while merely refreshing the registry manifest is rejected.
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
}

MANDATORY_REGISTRY_MANIFEST_PATHS = {
    CONFIG_PATH,
    SUPERSESSION_PATH,
    DECISION_LOG_PATH,
    *REGISTRY_PATHS.values(),
    SCHEMA_MANIFEST,
    SCHEMA_SUMS,
    SCIENTIFIC_M0_HISTORY_PATH,
    SEALED_GUARD_PATH,
    SEALED_RUNNER_PATH,
    VALIDATOR_PATH,
    A1_INTERIM_PATH,
}

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
        SUPERSESSION_PATH,
        DECISION_LOG_PATH,
        REGISTRY_MANIFEST_PATH,
        SCIENTIFIC_M0_HISTORY_PATH,
        SEALED_GUARD_PATH,
        SEALED_RUNNER_PATH,
        A1_INTERIM_PATH,
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


def validate_registry_manifest(repo_root: Path) -> list[Issue]:
    """Verify every public bundle hash listed by the A0 registry manifest."""

    issues: list[Issue] = []
    try:
        manifest = _load_json(repo_root, REGISTRY_MANIFEST_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "REGISTRY_MANIFEST_UNREADABLE", REGISTRY_MANIFEST_PATH, str(exc))
        return issues

    expected_top = {
        "contract_id": CONTRACT_ID,
        "version": VERSION,
        "schema_version": "1.0.0",
        "contract_path": GOAL_PATH,
        "initial_contract_sha256": "d1c031aecdec710495f6861b380785cccd64663ac4bd97b4f479d6fdf372ea07",
        "contract_sha256": SOURCE_CONTRACT_SHA256,
        "active_amendment_decision_ids": ["V3-DEC-017"],
        "base_commit": "bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c",
        "manifest_status": "A1_SCHEME_A_AUTHORITY_REBIND",
        "initial_generated_at": "2026-08-10T10:10:05+08:00",
        "sealed_contact": False,
    }
    for key, value in expected_top.items():
        if manifest.get(key) != value:
            _issue(issues, "REGISTRY_MANIFEST_METADATA", REGISTRY_MANIFEST_PATH, f"{key} must be {value!r}")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        _issue(issues, "REGISTRY_MANIFEST_FILES", REGISTRY_MANIFEST_PATH, "files must be a non-empty list")
        return issues
    by_path: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            _issue(issues, "REGISTRY_MANIFEST_ENTRY", REGISTRY_MANIFEST_PATH, f"files[{index}] is not an object")
            continue
        relative = entry.get("path")
        declared = entry.get("sha256")
        role = entry.get("role")
        if not isinstance(relative, str) or not isinstance(declared, str) or not isinstance(role, str) or not role:
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

    missing = sorted(MANDATORY_REGISTRY_MANIFEST_PATHS - set(by_path))
    if missing:
        _issue(issues, "REGISTRY_MANIFEST_COVERAGE", REGISTRY_MANIFEST_PATH, f"missing mandatory paths {missing!r}")
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
        _issue(issues, "DECISION_LOG_ID_CLOSURE", DECISION_LOG_PATH, "decision IDs must be exactly V3-DEC-001 through V3-DEC-017")
    ordered_ids = [entry.get("decision_id") if isinstance(entry, Mapping) else None for entry in raw]
    if ordered_ids != list(EXPECTED_DECISION_IDS):
        _issue(issues, "DECISION_LOG_ORDER", DECISION_LOG_PATH, "accepted decision prefix must remain in exact DEC-001 through DEC-017 order")
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
            "active_amendment_decision_ids": ["V3-DEC-017"],
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

    lineage = interim.get("artifact_lineage")
    if not isinstance(lineage, Mapping):
        _issue(issues, "A1_INTERIM_GSE200304_LINEAGE", path, "artifact_lineage must be a mapping")
    else:
        expected_all_lineage_ids = {
            "protocol",
            "collector",
            "legacy_gap_inventory_v1",
            "legacy_gap_inventory_v2",
            "gse114002_manifest_reconciliation_v1",
            "gse149487_reconstruction_attempt_003_failure",
            "gse149487_lim6c_scale_diagnostic_v1",
            "gse149487_plumage_protocol",
            "gse149487_plumage_reconstruction_v4",
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
        }
        if set(lineage) != expected_all_lineage_ids:
            _issue(
                issues,
                "A1_INTERIM_LINEAGE_ID_SET",
                path,
                "artifact lineage IDs must remain the exact accepted closed set",
            )
        expected_gse200304_lineage = {
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
                "sam_to_oligo_count_reconstruction_status": "UNKNOWN_NOT_ASSERTED",
                "acquisition_changes_qualification_gate": False,
                "source_grouping_status": "SEQUENCE_EQUALITY_PROXY_NOT_BIOLOGICALLY_FROZEN",
                "license_and_redistribution_status": "UNKNOWN_NOT_ASSERTED",
                "checkpoint_specific_foundation_exposure_status": "UNKNOWN_NOT_ASSERTED",
                "canonical_intervention_record_count": 0,
                "qualified": False,
                "training_allowed": False,
                "model_selection_allowed": False,
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
            "gse200304_engineering_success_is_study_qualification": False,
            "gse200304_fastq_acquisition_is_study_qualification": False,
            "gse200304_transport_integrity_is_paper_native_count_replay": False,
            "gse200304_sequence_proxy_groups_are_biological_source_groups": False,
            "gse200304_precomputed_aggregate_evidence_is_paper_native_xtail_replay": False,
            "a1_phase_complete": False,
            "route_a_established": False,
        }
        _expect_closed_mapping(
            claims, expected_claims, path, issues, "A1_INTERIM_CLAIMS"
        )

    _expect(interim, "initial_generated_at", "2026-08-10T06:30:58+08:00", path, issues, "A1_INTERIM_TIME")
    _expect(interim, "updated_for_decision_id", "V3-DEC-017", path, issues, "A1_INTERIM_TIME")
    _expect(
        interim,
        "latest_evidence_update_id",
        "GSE200304_FASTQ_CONSUMER_VERIFY_20260810T191502P0800_e24d722",
        path,
        issues,
        "A1_INTERIM_TIME",
    )
    generated = interim.get("generated_at")
    updated = interim.get("updated_at")
    if generated != updated:
        _issue(issues, "A1_INTERIM_TIME", path, "generated_at and updated_at must identify the same amended record bytes")
    try:
        updated_dt = datetime.fromisoformat(str(updated))
        audit_dt = datetime.fromisoformat("2026-08-10T08:43:13+08:00")
        amendment_dt = datetime.fromisoformat("2026-08-10T10:10:05+08:00")
        acquisition_dt = datetime.fromisoformat("2026-08-10T19:15:02+08:00")
    except ValueError:
        _issue(issues, "A1_INTERIM_TIME", path, "updated_at must be an ISO-8601 timestamp with offset")
    else:
        if updated_dt < audit_dt or updated_dt < amendment_dt or updated_dt < acquisition_dt:
            _issue(issues, "A1_INTERIM_TIME", path, "updated_at must follow the formal audit, DEC-017 authorization, and FASTQ consumer evidence")
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
    issues.extend(validate_python_static_safety(repo_root))
    issues.extend(validate_runner_and_guard_ast(repo_root))
    issues.extend(scan_conflict_markers(repo_root))
    try:
        config, supersession, registries = load_bundle_documents(repo_root)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        _issue(issues, "BUNDLE_DOCUMENT_LOAD", ".", str(exc))
        return sorted(set(issues))
    issues.extend(validate_contract_authority(repo_root, config, supersession, registries))
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
