#!/usr/bin/env python3
"""Build physically sealed B0 Track A/B/C evaluation artifacts.

The role assignment is intentionally fixed before any label-bearing canonical
store is opened. Source-disjoint train records form Track C. Validation records
form Track B. Test atomic components that contain any structurally
non-measured-eligible record are rerouted intact to Track B; only the remaining
all-measured-eligible test components form Track A. Role selection never reads
effect values or directions and never splits an atomic component.

Track C exposes only the source, endpoint condition, legal actions, and a fixed
edit budget.  It never exposes the D1 candidate identity or sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.utr_benchmark_v2.split_graph import record_ids_sha256
from data.utr_benchmark_v2.split_graph import record_universe_sha256
from data.utr_benchmark_v2.split_graph import build_atomic_components
from data.utr_benchmark_v2.split_graph import expected_partition_ids
from data.utr_benchmark_v2.split_graph import SplitGraphError
from data.utr_benchmark_v2.d1_builder import D1_SCOPE_DATASETS
from data.utr_benchmark_v2.records import MEASURED_PAIR_TYPES
from data.utr_benchmark_v2.track_loader import audit_track_roles
from data.utr_benchmark_v2.track_loader import expected_generation_task
from data.utr_benchmark_v2.track_loader import ids_sha256
from data.utr_benchmark_v2.track_loader import load_track_manifest
from data.utr_benchmark_v2.track_loader import (
    privileged_verify_track_a_label_seal,
)
from scripts.data.audit_b0_leakage import (
    build_bound_structural_recompute_cache,
)
from scripts.data.audit_b0_leakage import (
    recompute_bound_leakage_report,
)
from scripts.data.build_b0_splits import _load_d1_acceptance_binding
from scripts.data.build_b0_splits import load_jsonl
from scripts.data.build_b0_splits import load_structural_jsonl
from scripts.data.build_b0_splits import sha256_file
from scripts.data.validate_b0_acceptance import EXPECTED_SPLIT_IDENTITIES
from scripts.data.validate_b0_acceptance import ZERO_GATE_COUNTS
from scripts.data.validate_b0_acceptance import _common_split_universe
from scripts.data.validate_b0_acceptance import (
    _derive_role_record_universe,
)
from scripts.data.validate_b0_acceptance import _partition_evidence_failures
from scripts.data.validate_b0_acceptance import _split_identity
from scripts.data.validate_b0_acceptance import _strict_pending_fm0
from scripts.data.validate_b0_acceptance import compute_exposure_coverage
from scripts.data.validate_b0_acceptance import validate_b0_acceptance
from scripts.data.validate_b0_acceptance import (
    validate_d1_exposure_ledger_binding,
)
from scripts.data.validate_b0_acceptance import validate_required_artifacts


TRACK_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("closed_measured_pool", "closed_measured_pool", "test"),
    ("heldout_generative", "heldout_generative", "validation"),
    ("open_legal_generation", "open_legal_generation", "train"),
)
ROLE_TO_TRACK = {split_role: track_type for _, track_type, split_role in TRACK_SPECS}
ACTION_ORDER = ("INS", "SUB", "DEL", "STOP")
OPEN_WORLD_EDIT_BUDGET = 5
RETROSPECTIVE_ROLE = "historically_exposed_retrospective_external_stress_test"
SPLIT_UNIVERSE_FIELDS = (
    "canonical_records_sha256",
    "structural_records_sha256",
    "structural_records_bytes",
    "canonical_record_ids_sha256",
    "canonical_record_count",
    "structural_record_ids_sha256",
    "structural_record_count",
    "structural_content_sha256",
)
TRACK_A_HIDDEN_LABEL_SCHEMA = (
    Path(__file__).resolve().parents[2] / "schemas" / "track_a_hidden_label.schema.json"
)


class B0ArtifactBuildError(ValueError):
    """A frozen B0 input or output violates the active contract."""


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B0ArtifactBuildError(
            f"invalid JSON input {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise B0ArtifactBuildError(f"{path} must contain one JSON object")
    return value


def _load_jsonl_allow_empty(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise B0ArtifactBuildError(
                        f"{path}:{line_number} must contain an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise B0ArtifactBuildError(
            f"invalid JSONL input {path}: {type(exc).__name__}: {exc}"
        ) from exc
    return rows


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")


def _write_yaml_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(payload),
            handle,
            sort_keys=True,
            allow_unicode=True,
        )


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )


def _file_ref(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise B0ArtifactBuildError(f"artifact does not exist: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _preflight_d1_without_opening_labels(
    *,
    d1_acceptance_path: Path,
    canonical_records_path: Path,
    structural_records_path: Path,
) -> Dict[str, Any]:
    """Validate the D1 gate and structural binding without reading labels."""

    acceptance = _load_json(d1_acceptance_path)
    required = {
        "phase_gate_passed": True,
        "fixture_mode": False,
        "structural_validation_passed": True,
    }
    mismatched = {
        key: acceptance.get(key)
        for key, expected in required.items()
        if acceptance.get(key) is not expected
    }
    if mismatched:
        raise B0ArtifactBuildError(
            f"D1 acceptance is not a frozen production PASS: {mismatched}"
        )
    for field in ("global_store_validation", "required_artifact_validation"):
        value = acceptance.get(field)
        if not isinstance(value, Mapping) or value.get("passed") is not True:
            raise B0ArtifactBuildError(f"D1 {field} is not PASS")

    stage_root = Path(str(acceptance.get("stage_d1_root") or ""))
    if not stage_root.is_absolute():
        raise B0ArtifactBuildError("D1 stage_d1_root must be absolute")
    build_manifest_path = stage_root / "build_manifest.json"
    build_manifest = _load_json(build_manifest_path)
    stores = build_manifest.get("global_stores")
    if not isinstance(stores, Mapping):
        raise B0ArtifactBuildError("D1 build manifest lacks global_stores")
    canonical_meta = stores.get("canonical_label_store")
    structural_meta = stores.get("sealed_label_free_candidate_store")
    if not isinstance(canonical_meta, Mapping) or not isinstance(
        structural_meta, Mapping
    ):
        raise B0ArtifactBuildError(
            "D1 build manifest lacks canonical/structural store bindings"
        )
    expected_canonical = stage_root / str(canonical_meta.get("path") or "")
    expected_structural = stage_root / str(structural_meta.get("path") or "")
    if canonical_records_path.resolve() != expected_canonical.resolve():
        raise B0ArtifactBuildError(
            "canonical input differs from the D1 build-manifest path"
        )
    if structural_records_path.resolve() != expected_structural.resolve():
        raise B0ArtifactBuildError(
            "structural input differs from the D1 build-manifest path"
        )
    if not canonical_records_path.is_file():
        raise B0ArtifactBuildError("D1 canonical label store is missing")
    if not structural_records_path.is_file():
        raise B0ArtifactBuildError("D1 structural store is missing")
    if structural_records_path.stat().st_size != structural_meta.get(
        "bytes"
    ) or _sha256(structural_records_path) != structural_meta.get("sha256"):
        raise B0ArtifactBuildError(
            "D1 structural store differs from its frozen manifest"
        )
    # Deliberately do not hash or read canonical_records_path here.  The full
    # D1 binding is revalidated only after the pre-label selection freeze.
    return {
        "acceptance": acceptance,
        "acceptance_path": d1_acceptance_path.resolve(),
        "build_manifest": build_manifest,
        "build_manifest_path": build_manifest_path.resolve(),
        "canonical_meta": dict(canonical_meta),
        "structural_meta": dict(structural_meta),
    }


def _load_and_validate_split_evidence(
    split_manifest_paths: Sequence[Path],
    leakage_report_paths: Sequence[Path],
    *,
    structural_records_path: Path,
    d1_acceptance_path: Path | None = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    set[str],
    set[str],
]:
    if len(split_manifest_paths) != 5 or len(leakage_report_paths) != 5:
        raise B0ArtifactBuildError(
            "B0 production requires exactly five split manifests and "
            "five leakage reports"
        )
    try:
        recompute_cache = build_bound_structural_recompute_cache(
            structural_records_path
        )
    except (OSError, ValueError) as exc:
        raise B0ArtifactBuildError(
            "bound structural recomputation cache failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    manifests: List[Dict[str, Any]] = []
    manifest_by_identity: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}
    manifest_path_by_identity: Dict[Tuple[str, str, Any], Path] = {}
    for path in split_manifest_paths:
        manifest = _load_json(path)
        manifest["_artifact_sha256"] = _sha256(path)
        identity = _split_identity(manifest)
        if identity in manifest_by_identity:
            raise B0ArtifactBuildError(f"duplicate split identity: {identity}")
        manifests.append(manifest)
        manifest_by_identity[identity] = manifest
        manifest_path_by_identity[identity] = path.resolve()
    if set(manifest_by_identity) != EXPECTED_SPLIT_IDENTITIES:
        raise B0ArtifactBuildError(
            "split inputs are not the exact five required B0 identities"
        )
    if any(manifest.get("status") != "READY" for manifest in manifests):
        raise B0ArtifactBuildError("every B0 split manifest must be READY")

    split_universe = _common_split_universe(manifests)
    if split_universe is None:
        raise B0ArtifactBuildError(
            "the five split manifests do not share one frozen D1 universe"
        )
    (
        role_universe,
        role_record_ids,
        full_record_ids,
        role_failures,
    ) = _derive_role_record_universe(manifest_by_identity, split_universe)
    if role_universe is None or role_failures:
        raise B0ArtifactBuildError(
            "split eligible/excluded accounting failed: "
            + ", ".join(sorted(set(role_failures)))
        )

    supplied_report_by_identity: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}
    for path in leakage_report_paths:
        report = _load_json(path)
        identity = _split_identity(report)
        if identity in supplied_report_by_identity:
            raise B0ArtifactBuildError(f"duplicate leakage-report identity: {identity}")
        supplied_report_by_identity[identity] = report
    if set(supplied_report_by_identity) != EXPECTED_SPLIT_IDENTITIES:
        raise B0ArtifactBuildError(
            "leakage inputs are not the exact five required B0 identities"
        )

    reports: List[Dict[str, Any]] = []
    report_by_identity: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}
    failures: List[str] = []
    for identity in EXPECTED_SPLIT_IDENTITIES:
        manifest = manifest_by_identity[identity]
        try:
            report = recompute_bound_leakage_report(
                structural_records_path,
                manifest_path_by_identity[identity],
                verify_canonical_store=False,
                expected_d1_acceptance_path=d1_acceptance_path,
                _cache=recompute_cache,
            )
        except (OSError, ValueError) as exc:
            raise B0ArtifactBuildError(
                "bound leakage recomputation failed for "
                f"{identity}: {type(exc).__name__}: {exc}"
            ) from exc
        supplied = supplied_report_by_identity[identity]
        if supplied != report:
            failures.append("supplied_leakage_report_differs_from_recomputation")
        reports.append(report)
        report_by_identity[identity] = report
        if report.get("split_manifest_sha256") != manifest["_artifact_sha256"]:
            failures.append("leakage_report_manifest_sha256")
        if any(
            report.get(field) != manifest.get(field) for field in SPLIT_UNIVERSE_FIELDS
        ):
            failures.append("leakage_report_manifest_universe")
    for gate_name, count_name in ZERO_GATE_COUNTS.items():
        for report in reports:
            counts = report.get("counts")
            gates = report.get("acceptance_gates")
            if (
                not isinstance(counts, Mapping)
                or not isinstance(gates, Mapping)
                or counts.get(count_name) != 0
                or gates.get(gate_name) is not True
            ):
                failures.append(gate_name)
    strict_foundation, _ = _strict_pending_fm0(reports)
    if not strict_foundation:
        failures.append("foundation_state_not_UNKNOWN_PENDING_FM0")
    failures.extend(
        _partition_evidence_failures(
            manifest_by_identity,
            report_by_identity,
            role_record_ids=role_record_ids,
            full_record_ids=full_record_ids,
        )
    )
    if failures:
        raise B0ArtifactBuildError(
            "split/leakage evidence is not production-ready: "
            + ", ".join(sorted(set(failures)))
        )
    return (
        manifests,
        reports,
        dict(split_universe),
        role_record_ids,
        full_record_ids,
    )


def _partition_for_source_manifest(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    region = str(manifest.get("region") or "")
    try:
        expected_ids = expected_partition_ids("source_disjoint", region=region)
    except SplitGraphError as exc:
        raise B0ArtifactBuildError(
            "source-disjoint manifest has an invalid region"
        ) from exc
    required_ids = manifest.get("required_partition_ids")
    partitions = manifest.get("partitions")
    if not (
        isinstance(required_ids, list)
        and tuple(required_ids) == expected_ids
        and len(required_ids) == len(set(required_ids))
        and isinstance(partitions, list)
        and len(partitions) == len(expected_ids)
    ):
        raise B0ArtifactBuildError(
            "source-disjoint manifest must preserve all seven frozen "
            "required axis partitions"
        )
    by_id: Dict[str, Mapping[str, Any]] = {}
    for partition in partitions:
        if not isinstance(partition, Mapping):
            raise B0ArtifactBuildError(
                "source-disjoint axis partition must be an object"
            )
        partition_id = partition.get("partition_id")
        if (
            not isinstance(partition_id, str)
            or not partition_id
            or partition_id in by_id
        ):
            raise B0ArtifactBuildError(
                "source-disjoint axis partition IDs must be unique"
            )
        by_id[partition_id] = partition
    if set(by_id) != set(expected_ids):
        raise B0ArtifactBuildError(
            "source-disjoint axis partitions differ from the frozen IDs"
        )
    blocked = sorted(
        partition_id
        for partition_id, partition in by_id.items()
        if partition.get("status") != "READY"
    )
    if blocked:
        raise B0ArtifactBuildError(
            "every frozen source-disjoint axis partition must be READY: "
            + ", ".join(blocked)
        )
    return by_id[f"source_disjoint:{region}"]


def _derive_fixed_track_roles(
    manifests: Sequence[Mapping[str, Any]],
    role_record_ids: set[str],
    eligible_records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    source_manifests = {
        str(manifest.get("region")): manifest
        for manifest in manifests
        if manifest.get("split_kind") == "source_disjoint"
    }
    if set(source_manifests) != {"five_utr", "three_utr"}:
        raise B0ArtifactBuildError(
            "fixed track roles require READY 5' and 3' source-disjoint splits"
        )
    initial_role_by_record: Dict[str, str] = {}
    partition_refs: Dict[str, Any] = {}
    for region in ("five_utr", "three_utr"):
        manifest = source_manifests[region]
        partition = _partition_for_source_manifest(manifest)
        roles = partition.get("roles")
        if not isinstance(roles, Mapping) or set(roles) != {
            "train",
            "validation",
            "test",
        }:
            raise B0ArtifactBuildError(
                f"{region} source-disjoint roles violate the sealed schema"
            )
        for split_role, raw_record_ids in roles.items():
            if not isinstance(raw_record_ids, list):
                raise B0ArtifactBuildError(
                    f"{region}:{split_role} role is not a record-ID list"
                )
            track_type = ROLE_TO_TRACK[split_role]
            for raw_record_id in raw_record_ids:
                record_id = str(raw_record_id)
                if record_id in initial_role_by_record:
                    raise B0ArtifactBuildError(
                        f"record has multiple track roles: {record_id}"
                    )
                initial_role_by_record[record_id] = track_type
        partition_refs[region] = {
            "split_manifest_sha256": manifest["_artifact_sha256"],
            "partition_id": partition["partition_id"],
            "partition_sha256": partition["partition_sha256"],
        }
    if set(initial_role_by_record) != role_record_ids:
        raise B0ArtifactBuildError(
            "fixed source-disjoint track roles do not cover the exact "
            "eligible intervention universe"
        )
    record_index = {str(record["record_id"]): record for record in eligible_records}
    if set(record_index) != role_record_ids:
        raise B0ArtifactBuildError(
            "structural records do not match the eligible role universe"
        )

    role_by_record: Dict[str, str] = {}
    rerouted_components: List[Dict[str, Any]] = []
    for component in build_atomic_components(eligible_records):
        initial_roles = {
            initial_role_by_record[record_id] for record_id in component.record_ids
        }
        if len(initial_roles) != 1:
            raise B0ArtifactBuildError(
                "one atomic state component spans multiple initial "
                f"source-disjoint roles: {component.component_id}"
            )
        initial_track = next(iter(initial_roles))
        component_records = [
            record_index[record_id] for record_id in component.record_ids
        ]
        nonmeasured_ids = sorted(
            str(record["record_id"])
            for record in component_records
            if record.get("pair_type") not in MEASURED_PAIR_TYPES
        )
        assigned_track = initial_track
        if initial_track == "closed_measured_pool" and nonmeasured_ids:
            assigned_track = "heldout_generative"
            rerouted_components.append(
                {
                    "component_id": component.component_id,
                    "component_record_ids_sha256": ids_sha256(
                        list(component.record_ids)
                    ),
                    "component_record_count": len(component.record_ids),
                    "nonmeasured_record_ids_sha256": ids_sha256(nonmeasured_ids),
                    "nonmeasured_record_count": len(nonmeasured_ids),
                    "from": "closed_measured_pool",
                    "to": "heldout_generative",
                    "reason": (
                        "atomic_test_component_contains_non_MEASURED_PAIR_TYPES"
                    ),
                }
            )
        for record_id in component.record_ids:
            if record_id in role_by_record:
                raise B0ArtifactBuildError(
                    f"atomic role routing duplicated record: {record_id}"
                )
            role_by_record[record_id] = assigned_track
    if set(role_by_record) != role_record_ids:
        raise B0ArtifactBuildError(
            "component-level safety routing does not cover the eligible universe"
        )

    counts = {
        track_type: sum(
            observed_track == track_type for observed_track in role_by_record.values()
        )
        for _, track_type, _ in TRACK_SPECS
    }
    if any(count == 0 for count in counts.values()):
        raise B0ArtifactBuildError(f"all three tracks must be non-empty: {counts}")
    evidence_counts = {
        track_type: {
            "measured_pair_type": sum(
                role_by_record[record_id] == track_type
                and record_index[record_id].get("pair_type") in MEASURED_PAIR_TYPES
                for record_id in role_by_record
            ),
            "structural_unmeasured_pair_type": sum(
                role_by_record[record_id] == track_type
                and record_index[record_id].get("pair_type") not in MEASURED_PAIR_TYPES
                for record_id in role_by_record
            ),
        }
        for _, track_type, _ in TRACK_SPECS
    }
    if evidence_counts["closed_measured_pool"]["structural_unmeasured_pair_type"] != 0:
        raise B0ArtifactBuildError(
            "component routing left a non-measured pair type in Track A"
        )
    return role_by_record, {
        "schema_version": "utr_b0_track_role_policy.v2",
        "selection_is_label_independent": True,
        "label_fields_read_for_selection": [],
        "structural_fields_read_for_measured_eligibility": ["pair_type"],
        "atomic_components_split": False,
        "measured_eligibility": {
            "allowed_pair_types": sorted(MEASURED_PAIR_TYPES),
            "requires_canonical_finite_pair_recheck_after_freeze": True,
            "effect_value_or_direction_used_for_selection": False,
        },
        "policy": {
            "source_disjoint_test_all_measured_component": ("closed_measured_pool"),
            "source_disjoint_test_component_with_any_nonmeasured_record": (
                "heldout_generative"
            ),
            "source_disjoint_validation": (
                "heldout_generative_b0_no_efficacy_conclusion"
            ),
            "source_disjoint_train": "open_legal_generation",
        },
        "partition_bindings": partition_refs,
        "record_counts": counts,
        "track_evidence_counts": evidence_counts,
        "rerouted_test_components": rerouted_components,
        "rerouted_test_component_count": len(rerouted_components),
        "rerouted_test_record_count": sum(
            row["component_record_count"] for row in rerouted_components
        ),
        "track_evidence_boundaries": {
            "closed_measured_pool": (
                "MEASURED_PAIR_TYPES plus late finite paired-label recheck"
            ),
            "heldout_generative": (
                "B0 freezes heldout generative tasks without an efficacy "
                "conclusion; future contract-gated formal generative "
                "evaluation remains allowed"
            ),
            "open_legal_generation": (
                "source and legal actions only; predicted/computational/"
                "proxy-supported evidence"
            ),
        },
        "track_b_b0_efficacy_conclusion_allowed": False,
        "track_b_future_formal_generative_evaluation_allowed": True,
    }


def _load_and_bind_structural_records(
    structural_records_path: Path,
    *,
    split_universe: Mapping[str, Any],
    full_record_ids: set[str],
    role_record_ids: set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = load_structural_jsonl(structural_records_path)
    observed_ids = {str(record.get("record_id") or "").strip() for record in records}
    if "" in observed_ids or len(observed_ids) != len(records):
        raise B0ArtifactBuildError(
            "D1 structural store has missing or duplicate record IDs"
        )
    checks = {
        "sha256": _sha256(structural_records_path)
        == split_universe["structural_records_sha256"],
        "bytes": structural_records_path.stat().st_size
        == split_universe["structural_records_bytes"],
        "count": len(records) == split_universe["structural_record_count"],
        "ids": record_ids_sha256(records)
        == split_universe["structural_record_ids_sha256"],
        "content": record_universe_sha256(records)
        == split_universe["structural_content_sha256"],
        "accounted_ids": observed_ids == full_record_ids,
    }
    if not all(checks.values()):
        raise B0ArtifactBuildError(
            f"structural store differs from split universe: {checks}"
        )
    eligible = [
        record for record in records if str(record["record_id"]) in role_record_ids
    ]
    if len(eligible) != len(role_record_ids):
        raise B0ArtifactBuildError(
            "not every eligible split record is present structurally"
        )
    for record in eligible:
        required = (
            "record_id",
            "candidate_id",
            "dataset_id",
            "study_id",
            "region",
            "source_id",
            "source_sequence",
            "candidate_sequence",
            "endpoint",
            "edit_types",
            "edit_count",
            "edit_distance",
        )
        if any(field not in record for field in required):
            raise B0ArtifactBuildError(
                f"eligible record lacks generation fields: {record.get('record_id')}"
            )
        if (
            not isinstance(record["source_id"], str)
            or not record["source_id"]
            or not isinstance(record["source_sequence"], str)
            or not record["source_sequence"]
            or not isinstance(record["candidate_sequence"], str)
            or not record["candidate_sequence"]
        ):
            raise B0ArtifactBuildError(
                f"eligible record is not an intervention: {record['record_id']}"
            )
    return records, eligible


def _record_scoped_candidate_id(record: Mapping[str, Any]) -> str:
    digest = _stable_sha256(
        {
            "record_id": record["record_id"],
            "d1_candidate_id": record["candidate_id"],
            "candidate_sequence": record["candidate_sequence"],
        }
    )
    return f"utr-editbench-v2:candidate:{digest[:24]}"


def _task_id(record_id: str, track_type: str) -> str:
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return f"utr-editbench-v2:{track_type}:{digest[:24]}"


def _observed_actions(record: Mapping[str, Any]) -> List[str]:
    raw = record.get("edit_types")
    if not isinstance(raw, list) or any(
        action not in {"INS", "SUB", "DEL"} for action in raw
    ):
        raise B0ArtifactBuildError(
            f"record has invalid edit_types: {record.get('record_id')}"
        )
    actions = [
        action for action in ACTION_ORDER if action in set(raw) or action == "STOP"
    ]
    if actions == ["STOP"]:
        raise B0ArtifactBuildError(
            f"eligible intervention has no edit action: {record.get('record_id')}"
        )
    return actions


def _generation_task(
    record: Mapping[str, Any],
    *,
    track_type: str,
) -> Dict[str, Any]:
    try:
        return expected_generation_task(record, track_type=track_type)
    except ValueError as exc:
        raise B0ArtifactBuildError(str(exc)) from exc


def _build_tasks(
    eligible_records: Sequence[Mapping[str, Any]],
    role_by_record: Mapping[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    tasks = {track_type: [] for _, track_type, _ in TRACK_SPECS}
    for record in sorted(eligible_records, key=lambda value: str(value["record_id"])):
        record_id = str(record["record_id"])
        track_type = role_by_record.get(record_id)
        if track_type not in tasks:
            raise B0ArtifactBuildError(f"record lacks a fixed track role: {record_id}")
        tasks[track_type].append(_generation_task(record, track_type=track_type))
    _assert_cross_track_separation(tasks)
    return tasks


def _assert_cross_track_separation(
    tasks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    identity_extractors = {
        "record_id": lambda task: task["provenance"]["record_id"],
        "task_id": lambda task: task["task_id"],
        "source_id": lambda task: task["source_id"],
        "source_sequence": lambda task: task["source_sequence"],
    }
    for identity_name, extractor in identity_extractors.items():
        by_track = {
            track_type: {str(extractor(task)) for task in track_tasks}
            for track_type, track_tasks in tasks.items()
        }
        track_types = sorted(by_track)
        for index, left in enumerate(track_types):
            for right in track_types[index + 1 :]:
                overlap = by_track[left] & by_track[right]
                if overlap:
                    raise B0ArtifactBuildError(
                        f"{identity_name} overlaps {left}/{right}: "
                        f"{sorted(overlap)[:3]}"
                    )
    candidate_sequences = {
        track_type: {
            str(task["candidate_sequence"])
            for task in track_tasks
            if task["candidate_sequence"] is not None
        }
        for track_type, track_tasks in tasks.items()
    }
    if (
        candidate_sequences["closed_measured_pool"]
        & candidate_sequences["heldout_generative"]
    ):
        raise B0ArtifactBuildError(
            "Track A/B candidate sequences overlap despite source-disjoint roles"
        )


def _universe_binding(
    tasks: Mapping[str, Sequence[Mapping[str, Any]]],
    split_universe: Mapping[str, Any],
) -> Dict[str, Any]:
    all_tasks = [task for track_type in sorted(tasks) for task in tasks[track_type]]
    record_ids = [str(task["provenance"]["record_id"]) for task in all_tasks]
    task_ids = [str(task["task_id"]) for task in all_tasks]
    candidate_ids = [
        str(task["candidate_id"])
        for task in all_tasks
        if task["candidate_id"] is not None
    ]
    source_ids = sorted({str(task["source_id"]) for task in all_tasks})
    if (
        len(record_ids) != len(set(record_ids))
        or len(task_ids) != len(set(task_ids))
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise B0ArtifactBuildError(
            "track task, record, or candidate identities are not unique"
        )
    return {
        "canonical_records_sha256": split_universe["canonical_records_sha256"],
        "structural_records_sha256": split_universe["structural_records_sha256"],
        "record_ids_sha256": ids_sha256(record_ids),
        "record_count": len(record_ids),
        "candidate_ids_sha256": ids_sha256(candidate_ids),
        "candidate_count": len(candidate_ids),
        "task_ids_sha256": ids_sha256(task_ids),
        "task_count": len(task_ids),
        "source_ids_sha256": ids_sha256(source_ids),
        "source_count": len(source_ids),
    }


def _validate_full_d1_binding(
    canonical_records_path: Path,
    d1_acceptance_path: Path,
) -> Dict[str, Any]:
    """Late privileged validation hook, intentionally easy to audit in tests."""

    return _load_d1_acceptance_binding(canonical_records_path, d1_acceptance_path)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_hidden_label_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    schema_path: Path = TRACK_A_HIDDEN_LABEL_SCHEMA,
) -> Dict[str, Any]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError(
            "jsonschema>=4.18 is required for Track A label sealing"
        ) from exc
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    failures: List[Dict[str, Any]] = []
    candidate_ids: List[str] = []
    record_ids: List[str] = []
    for index, row in enumerate(rows):
        row_errors = sorted(
            validator.iter_errors(row),
            key=lambda error: tuple(str(part) for part in error.path),
        )
        for error in row_errors:
            failures.append(
                {
                    "index": index,
                    "path": ".".join(str(part) for part in error.path),
                    "message": error.message,
                }
            )
        candidate_ids.append(str(row.get("candidate_id") or ""))
        record_ids.append(str(row.get("record_id") or ""))
        source_value = row.get("source_value_raw")
        candidate_value = row.get("candidate_value_raw")
        delta = row.get("delta_raw")
        if not all(
            _is_finite_number(value) for value in (source_value, candidate_value, delta)
        ):
            failures.append(
                {
                    "index": index,
                    "path": "paired_values",
                    "message": "source, candidate, and delta must all be finite",
                }
            )
        elif not math.isclose(
            float(delta),
            float(candidate_value) - float(source_value),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            failures.append(
                {
                    "index": index,
                    "path": "delta_raw",
                    "message": (
                        "delta_raw differs from candidate_value_raw - "
                        "source_value_raw"
                    ),
                }
            )
        for field in ("delta_normalized", "effect_standard_error"):
            value = row.get(field)
            if value is not None and not _is_finite_number(value):
                failures.append(
                    {
                        "index": index,
                        "path": field,
                        "message": f"{field} must be finite or null",
                    }
                )
    if (
        not rows
        or "" in candidate_ids
        or "" in record_ids
        or len(candidate_ids) != len(set(candidate_ids))
        or len(record_ids) != len(set(record_ids))
    ):
        failures.append(
            {
                "index": None,
                "path": "identity_universe",
                "message": "hidden label identities must be non-empty and unique",
            }
        )
    if failures:
        raise B0ArtifactBuildError(
            "Track A hidden labels violate the strict schema: "
            + json.dumps(failures[:5], sort_keys=True)
        )
    return {
        "schema_version": "utr_track_a_hidden_label_schema_audit.v2",
        "gate_passed": True,
        "schema_path": str(schema_path.resolve()),
        "schema_sha256": _sha256(schema_path),
        "record_count": len(rows),
        "record_ids_sha256": ids_sha256(record_ids),
        "candidate_count": len(rows),
        "candidate_ids_sha256": ids_sha256(candidate_ids),
        "paired_finite_measurements": len(rows),
        "delta_arithmetic_mismatch_count": 0,
    }


def _hidden_label_rows(
    canonical_records_path: Path,
    track_a_tasks: Sequence[Mapping[str, Any]],
    *,
    d1_acceptance_path: Path,
    split_universe: Mapping[str, Any],
    structural_records_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    binding = _validate_full_d1_binding(canonical_records_path, d1_acceptance_path)
    if binding.get("passed") is not True:
        raise B0ArtifactBuildError(
            "late full D1 canonical/structural binding did not pass"
        )
    canonical_binding = binding.get("canonical")
    structural_binding = binding.get("structural")
    if not isinstance(canonical_binding, Mapping) or not isinstance(
        structural_binding, Mapping
    ):
        raise B0ArtifactBuildError("late D1 binding is incomplete")
    expected = {
        "canonical_sha256": split_universe["canonical_records_sha256"],
        "structural_sha256": split_universe["structural_records_sha256"],
        "structural_path": str(structural_records_path.resolve()),
    }
    observed = {
        "canonical_sha256": canonical_binding.get("sha256"),
        "structural_sha256": structural_binding.get("sha256"),
        "structural_path": str(
            Path(str(structural_binding.get("path") or "")).resolve()
        ),
    }
    if observed != expected:
        raise B0ArtifactBuildError(
            f"late D1 binding differs from split universe: {observed}"
        )
    canonical = load_jsonl(canonical_records_path)
    canonical_index = {
        str(record.get("record_id") or ""): record for record in canonical
    }
    if (
        "" in canonical_index
        or len(canonical_index) != len(canonical)
        or _sha256(canonical_records_path) != split_universe["canonical_records_sha256"]
        or len(canonical) != split_universe["canonical_record_count"]
        or record_ids_sha256(canonical) != split_universe["canonical_record_ids_sha256"]
    ):
        raise B0ArtifactBuildError(
            "canonical label store changed or has duplicate record IDs"
        )

    rows: List[Dict[str, Any]] = []
    for task in track_a_tasks:
        record_id = str(task["provenance"]["record_id"])
        record = canonical_index.get(record_id)
        if record is None:
            raise B0ArtifactBuildError(
                f"Track A record is absent from canonical labels: {record_id}"
            )
        if (
            record.get("candidate_sequence") != task["candidate_sequence"]
            or record.get("source_sequence") != task["source_sequence"]
            or record.get("endpoint") != task["endpoint"]
            or record.get("dataset_id") != task["provenance"]["dataset_id"]
            or record.get("study_id") != task["provenance"]["study_id"]
            or record.get("source_id") != task["source_id"]
            or _record_scoped_candidate_id(record) != task["candidate_id"]
        ):
            raise B0ArtifactBuildError(
                f"Track A canonical task binding changed: {record_id}"
            )
        if record.get("pair_type") not in MEASURED_PAIR_TYPES:
            raise B0ArtifactBuildError(
                f"Track A pair_type is not measured-eligible: {record_id}"
            )
        source_value = record.get("source_value_raw")
        candidate_value = record.get("candidate_value_raw")
        delta = record.get("delta_raw")
        if not all(
            _is_finite_number(value) for value in (source_value, candidate_value, delta)
        ):
            raise B0ArtifactBuildError(
                f"Track A lacks finite paired measured values: {record_id}"
            )
        if not math.isclose(
            float(delta),
            float(candidate_value) - float(source_value),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise B0ArtifactBuildError(
                f"Track A measured delta arithmetic mismatch: {record_id}"
            )
        required_identity_fields = (
            "candidate_id",
            "dataset_id",
            "study_id",
            "assay_id",
            "context_id",
            "endpoint",
            "source_id",
        )
        if any(
            not isinstance(record.get(field), str) or not str(record[field]).strip()
            for field in required_identity_fields
        ):
            raise B0ArtifactBuildError(
                f"Track A canonical identity binding is incomplete: {record_id}"
            )
        rows.append(
            {
                "schema_version": "utr_track_a_hidden_label.v2",
                "candidate_id": task["candidate_id"],
                "canonical_candidate_id": record["candidate_id"],
                "record_id": record_id,
                "dataset_id": record["dataset_id"],
                "study_id": record["study_id"],
                "assay_id": record["assay_id"],
                "context_id": record["context_id"],
                "endpoint": record["endpoint"],
                "pair_type": record["pair_type"],
                "source_id": record["source_id"],
                "source_sequence_sha256": _sha256_text(str(record["source_sequence"])),
                "candidate_sequence_sha256": _sha256_text(
                    str(record["candidate_sequence"])
                ),
                "source_value_raw": source_value,
                "candidate_value_raw": candidate_value,
                "delta_raw": delta,
                "delta_normalized": record.get("delta_normalized"),
                "effect_standard_error": record.get("effect_standard_error"),
                "replicate_count": record.get("replicate_count"),
                "label_provenance": record.get("label_provenance"),
                "canonical_record_sha256": _stable_sha256(record),
                "measurement_evidence": "paired_finite_measured_endpoints",
            }
        )
    _validate_hidden_label_rows(rows)
    return rows, binding


def _track_manifest(
    *,
    track_type: str,
    candidate_store_path: Path,
    universe_binding: Mapping[str, Any],
    label_store_path: Path | None = None,
    freeze_proof_path: Path | None = None,
    hidden_label_schema_path: Path | None = None,
    selection_freeze_path: Path | None = None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "schema_version": "utr_track.v2",
        "track_id": track_type,
        "track_type": track_type,
        "candidate_store": {
            "path": candidate_store_path.name,
            "sha256": _sha256(candidate_store_path),
            "bytes": candidate_store_path.stat().st_size,
        },
        "candidate_store_contains_labels": False,
        "selection_access": {"labels": False},
        "retrospective_external_stress_datasets": [],
        "universe_binding": dict(universe_binding),
    }
    if track_type == "open_legal_generation":
        manifest["evaluation_budget_protocol"] = {
            "required_budgets": [1, 3, 5],
            "task_representation": "single_maximum_budget",
            "maximum_budget": 5,
            "report_each_budget_separately": True,
            "silent_budget_reduction_forbidden": True,
        }
    if track_type == "closed_measured_pool":
        if (
            label_store_path is None
            or freeze_proof_path is None
            or hidden_label_schema_path is None
            or selection_freeze_path is None
        ):
            raise B0ArtifactBuildError(
                "Track A requires labels, schema, selection freeze, and proof"
            )
        candidate_ids = [
            str(row["candidate_id"])
            for row in _load_jsonl_allow_empty(label_store_path)
        ]
        manifest["label_store"] = {
            "path": label_store_path.name,
            "sha256": _sha256(label_store_path),
            "bytes": label_store_path.stat().st_size,
            "access": "FROZEN_FINAL_ONLY",
            "candidate_id_field": "candidate_id",
            "candidate_ids_sha256": ids_sha256(candidate_ids),
            "candidate_count": len(candidate_ids),
            "schema": {
                "path": hidden_label_schema_path.name,
                "sha256": _sha256(hidden_label_schema_path),
                "bytes": hidden_label_schema_path.stat().st_size,
            },
            "selection_freeze": {
                "path": selection_freeze_path.name,
                "sha256": _sha256(selection_freeze_path),
                "bytes": selection_freeze_path.stat().st_size,
            },
            "freeze_proof": {
                "path": freeze_proof_path.name,
                "sha256": _sha256(freeze_proof_path),
            },
        }
    return manifest


def _claims(universe_binding: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "utr_b0_claims.v2",
        "universe_binding": dict(universe_binding),
        "foundation_status": "UNKNOWN_PENDING_FM0",
        "allowed_claim": "NONE",
        "requires_fm0_reaudit": True,
        "gse246381_role": RETROSPECTIVE_ROLE,
        "track_claims": {
            "closed_measured_pool": {
                "metric_name": "observed_pool_normalized_regret",
                "process_order_attestation_only": True,
                "forbidden_claims": ["full_legal_action_space_regret"],
            },
            "heldout_generative": {
                "b0_efficacy_conclusion_allowed": False,
                "future_formal_generative_evaluation_allowed": True,
                "evidence_scope": (
                    "B0 freezes heldout generative tasks without an efficacy "
                    "conclusion; later contract-gated formal generative "
                    "evaluation remains allowed"
                ),
                "forbidden_claims": ["measured_functional_improvement"],
            },
            "open_legal_generation": {
                "candidate_exposed": False,
                "required_evaluation_budgets": [1, 3, 5],
                "evidence_qualifiers": [
                    "predicted",
                    "computational",
                    "proxy-supported",
                ],
                "forbidden_claims": ["measured_improvement"],
            },
        },
        "allowed_claims": [
            "B0 structural benchmark, split, and track roles are frozen",
            "Track B has no efficacy conclusion at B0; future formal "
            "generative evaluation under later contract gates is allowed",
        ],
        "unsupported_capabilities": [
            "No final efficacy, SOTA, or full legal action-space result",
            "No foundation-model exposure clearance before FM0 re-audit",
            "No measured improvement claim for open-world Track C",
        ],
    }


def _data_card_body(
    *,
    universe_binding: Mapping[str, Any],
    track_audit: Mapping[str, Any],
    role_policy: Mapping[str, Any],
    exposure_sha256: str,
    role_matrix_sha256: str,
    claims_sha256: str,
    eligible_records: Sequence[Mapping[str, Any]],
) -> str:
    data_card_counts = track_audit.get("data_card_counts")
    if not isinstance(data_card_counts, Mapping):
        raise B0ArtifactBuildError("track audit lacks recomputed Data Card counts")
    if data_card_counts.get("track_evidence") != role_policy.get(
        "track_evidence_counts"
    ):
        raise B0ArtifactBuildError(
            "track audit and frozen role policy evidence counts differ"
        )
    front = {
        "schema_version": "utr_editbench_data_card.v2",
        "canonical_records_sha256": universe_binding["canonical_records_sha256"],
        "structural_records_sha256": universe_binding["structural_records_sha256"],
        "record_count": universe_binding["record_count"],
        "candidate_count": universe_binding["candidate_count"],
        "task_count": universe_binding["task_count"],
        "source_count": universe_binding["source_count"],
        "exposure_ledger_sha256": exposure_sha256,
        "track_role_matrix_sha256": role_matrix_sha256,
        "claims_sha256": claims_sha256,
        "required_evaluation_budgets": [1, 3, 5],
        "structured_facts": {
            "counts": dict(data_card_counts),
            "biases": {
                "proposal_space_uniform_coverage_claimed": False,
                "biological_trajectory_observed_claimed": False,
                "minimal_edit_paths_are_constructed": True,
            },
            "exposure": {
                "ledger_exact_d1_binding_required": True,
                "ledger_dataset_scope_count": len(D1_SCOPE_DATASETS),
                "foundation_status": "UNKNOWN_PENDING_FM0",
                "gse246381_role": RETROSPECTIVE_ROLE,
            },
            "claims": {
                "b0_efficacy_conclusion_allowed": False,
                "future_track_b_formal_generative_evaluation_allowed": True,
                "track_a_process_order_attestation_only": True,
                "track_c_measured_improvement_allowed": False,
            },
        },
    }
    region_counts = {
        region: sum(record.get("region") == region for record in eligible_records)
        for region in ("five_utr", "three_utr")
    }
    dataset_counts: Dict[str, int] = {}
    for record in eligible_records:
        dataset_id = str(record["dataset_id"])
        dataset_counts[dataset_id] = dataset_counts.get(dataset_id, 0) + 1
    track_counts = {
        str(row["track_type"]): int(row["task_count"]) for row in track_audit["tracks"]
    }
    body = [
        "---",
        yaml.safe_dump(front, sort_keys=True).rstrip(),
        "---",
        "# UTR EditBench v2 Data Card",
        "",
        "B0 freezes structural benchmark roles only. It does not report model "
        "efficacy or a final scientific result.",
        "",
        "## Counts",
        "",
        f"- Eligible intervention records/tasks: {universe_binding['record_count']}",
        f"- Known candidate tasks (Tracks A+B): {universe_binding['candidate_count']}",
        f"- Unique sources: {universe_binding['source_count']}",
        f"- Region counts: `{json.dumps(region_counts, sort_keys=True)}`",
        f"- Dataset counts: `{json.dumps(dataset_counts, sort_keys=True)}`",
        f"- Track task counts: `{json.dumps(track_counts, sort_keys=True)}`",
        "- Track evidence counts: "
        f"`{json.dumps(role_policy['track_evidence_counts'], sort_keys=True)}`",
        "",
        "## Biases",
        "",
        "- Records inherit each assay's library ascertainment, context, reporter, "
        "endpoint, and coverage biases; B0 does not reinterpret proposal-library "
        "support as uniform legal-action-space coverage.",
        "- Endpoint pairs define source/candidate coupling, not an observed "
        "biological trajectory; minimal paths remain constructed.",
        "- Track B includes source-disjoint validation records plus any intact "
        "test atomic component rerouted because it contains a structurally "
        "non-measured-eligible record. B0 draws no efficacy conclusion from "
        "Track B; future contract-gated formal generative evaluation remains "
        "allowed.",
        "",
        "## Exposure",
        "",
        "- Exposure-ledger coverage is complete for the frozen D1 universe.",
        "- Foundation sequence exposure remains `UNKNOWN_PENDING_FM0`; no model "
        "is declared unseen or cleared at B0.",
        "- GSE246381 remains historically exposed retrospective external stress "
        "evidence and has no normal Track A/B/C role.",
        "",
        "## Allowed Claims",
        "",
        "- The D1-bound structural universe, five split manifests, leakage "
        "evidence, and label-free A/B/C role assignment are frozen.",
        "- Track A may later use `observed_pool_normalized_regret` only after the "
        "separate model/candidate/path/statistical freeze and authorized label "
        "unlock. This is an execution-order attestation, not a result or "
        "performance claim.",
        "- Track C uses one maximum-budget task representation, but formal "
        "evaluation must report edit budgets 1, 3, and 5 separately; silent "
        "reduction to budget 1 is forbidden.",
        "- Track C outputs must be qualified as predicted, computational, or "
        "proxy-supported.",
        "",
        "## Unsupported Capabilities",
        "",
        "- B0 does not establish measured functional improvement, efficacy, SOTA, "
        "full legal action-space regret, foundation-model clearance, or "
        "open-world measured validation.",
        "- Smoke tests, proxy metrics, B0-stage Track B diagnostics, and training "
        "results cannot be promoted to final scientific conclusions.",
        "",
        "## Frozen Role Policy",
        "",
        "```yaml",
        yaml.safe_dump(dict(role_policy), sort_keys=True).rstrip(),
        "```",
        "",
    ]
    return "\n".join(body)


def _binding_path_value(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_b0_evaluation_artifacts(
    *,
    d1_acceptance_path: Path,
    canonical_records_path: Path,
    structural_records_path: Path,
    split_manifest_paths: Sequence[Path],
    leakage_report_paths: Sequence[Path],
    exposure_ledger_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    """Build and internally validate one immutable B0 artifact bundle."""

    d1 = _preflight_d1_without_opening_labels(
        d1_acceptance_path=d1_acceptance_path,
        canonical_records_path=canonical_records_path,
        structural_records_path=structural_records_path,
    )
    exposure_binding = validate_d1_exposure_ledger_binding(
        d1_acceptance_path,
        exposure_ledger_path,
    )
    if exposure_binding.get("gate_passed") is not True:
        raise B0ArtifactBuildError(
            "exposure ledger is not the exact D1-accepted artifact: "
            + ", ".join(exposure_binding.get("failures", []))
        )
    (
        split_manifests,
        leakage_reports,
        split_universe,
        role_record_ids,
        full_record_ids,
    ) = _load_and_validate_split_evidence(
        split_manifest_paths,
        leakage_report_paths,
        structural_records_path=structural_records_path,
        d1_acceptance_path=d1_acceptance_path,
    )
    canonical_meta = d1["canonical_meta"]
    structural_meta = d1["structural_meta"]
    if (
        canonical_meta.get("sha256") != split_universe["canonical_records_sha256"]
        or canonical_meta.get("records") != split_universe["canonical_record_count"]
        or structural_meta.get("sha256") != split_universe["structural_records_sha256"]
        or structural_meta.get("bytes") != split_universe["structural_records_bytes"]
        or structural_meta.get("records") != split_universe["structural_record_count"]
    ):
        raise B0ArtifactBuildError(
            "D1 build manifest and split universe bindings differ"
        )
    structural_records, eligible_records = _load_and_bind_structural_records(
        structural_records_path,
        split_universe=split_universe,
        full_record_ids=full_record_ids,
        role_record_ids=role_record_ids,
    )
    role_by_record, role_policy = _derive_fixed_track_roles(
        split_manifests, role_record_ids, eligible_records
    )
    tasks = _build_tasks(eligible_records, role_by_record)
    universe_binding = _universe_binding(tasks, split_universe)
    if universe_binding["record_ids_sha256"] != ids_sha256(
        sorted(role_record_ids)
    ) or universe_binding["record_count"] != len(role_record_ids):
        raise B0ArtifactBuildError(
            "track task universe differs from split eligible universe"
        )

    ledger_rows = _load_jsonl_allow_empty(exposure_ledger_path)
    exposure_audit = compute_exposure_coverage(
        structural_records,
        ledger_rows,
        identity_level="dataset_id",
        required_ledger_identities=sorted(D1_SCOPE_DATASETS),
    )
    if exposure_audit.get("gate_passed") is not True:
        raise B0ArtifactBuildError(
            "exposure ledger does not cover the full D1 dataset universe"
        )

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    leakage_root = output_root / "evaluation" / "leakage"
    tracks_root = output_root / "evaluation" / "tracks"
    claims_root = output_root / "evaluation" / "claims"
    data_card_root = output_root / "docs" / "data"
    role_policy_path = tracks_root / "track_role_policy.yaml"
    recomputed_report_paths: List[Path] = []
    for report in sorted(leakage_reports, key=_split_identity):
        split_kind, region, target_region = _split_identity(report)
        suffix = f"{region}_to_{target_region}" if target_region is not None else region
        report_path = leakage_root / f"{split_kind}.{suffix}.json"
        _write_json_exclusive(report_path, report)
        recomputed_report_paths.append(report_path)
    _write_yaml_exclusive(role_policy_path, role_policy)
    hidden_label_schema_path = (
        tracks_root / "closed_measured_pool.hidden_label.schema.json"
    )
    _write_text_exclusive(
        hidden_label_schema_path,
        TRACK_A_HIDDEN_LABEL_SCHEMA.read_text(encoding="utf-8"),
    )
    candidate_paths: Dict[str, Path] = {}
    for _, track_type, _ in TRACK_SPECS:
        path = tracks_root / f"{track_type}.tasks.jsonl"
        _write_jsonl_exclusive(path, tasks[track_type])
        candidate_paths[track_type] = path

    selection_freeze_path = tracks_root / "closed_measured_pool.selection.freeze.json"
    track_a_tasks = tasks["closed_measured_pool"]
    selection_freeze = {
        "schema_version": "utr_track_a_prelabel_selection_freeze.v2",
        "track_id": "closed_measured_pool",
        "candidate_store_path": str(candidate_paths["closed_measured_pool"].resolve()),
        "candidate_store_sha256": _sha256(candidate_paths["closed_measured_pool"]),
        "candidate_store_bytes": candidate_paths["closed_measured_pool"].stat().st_size,
        "selected_record_ids_sha256": ids_sha256(
            [str(task["provenance"]["record_id"]) for task in track_a_tasks]
        ),
        "selected_record_count": len(track_a_tasks),
        "selected_task_ids_sha256": ids_sha256(
            [str(task["task_id"]) for task in track_a_tasks]
        ),
        "selected_task_count": len(track_a_tasks),
        "structural_records_sha256": split_universe["structural_records_sha256"],
        "role_policy_path": str(role_policy_path.resolve()),
        "role_policy_sha256": _sha256(role_policy_path),
        "role_policy_bytes": role_policy_path.stat().st_size,
        "hidden_label_schema_path": str(hidden_label_schema_path.resolve()),
        "hidden_label_schema_sha256": _sha256(hidden_label_schema_path),
        "hidden_label_schema_bytes": hidden_label_schema_path.stat().st_size,
        "source_disjoint_partition_bindings": role_policy["partition_bindings"],
        "d1_acceptance": _file_ref(d1_acceptance_path),
        "d1_build_manifest": _file_ref(d1["build_manifest_path"]),
        "canonical_records_declared": {
            "path": str(canonical_records_path.resolve()),
            "sha256": canonical_meta["sha256"],
            "bytes": canonical_meta["bytes"],
            "record_count": canonical_meta["records"],
            "record_ids_sha256": canonical_meta["record_ids_sha256"],
        },
        "structural_records": {
            "path": str(structural_records_path.resolve()),
            "sha256": split_universe["structural_records_sha256"],
            "bytes": structural_records_path.stat().st_size,
            "record_count": split_universe["structural_record_count"],
            "record_ids_sha256": split_universe["structural_record_ids_sha256"],
            "structural_content_sha256": split_universe["structural_content_sha256"],
        },
        "canonical_label_store_opened": False,
        "selection_labels_hidden": True,
        "frozen_before_label_access": True,
    }
    # This exclusive write is deliberately the last operation before the late,
    # privileged canonical-label binding and read below.
    _write_json_exclusive(selection_freeze_path, selection_freeze)

    hidden_labels, full_d1_binding = _hidden_label_rows(
        canonical_records_path,
        track_a_tasks,
        d1_acceptance_path=d1_acceptance_path,
        split_universe=split_universe,
        structural_records_path=structural_records_path,
    )
    label_store_path = tracks_root / "closed_measured_pool.labels.hidden.jsonl"
    _write_jsonl_exclusive(label_store_path, hidden_labels)
    hidden_label_schema_audit = _validate_hidden_label_rows(hidden_labels)
    candidate_ids = [str(task["candidate_id"]) for task in track_a_tasks]
    label_candidate_ids = [str(row["candidate_id"]) for row in hidden_labels]
    freeze_proof = {
        "schema_version": "utr_track_a_label_freeze_proof.v2",
        "track_id": "closed_measured_pool",
        "candidate_store_sha256": _sha256(candidate_paths["closed_measured_pool"]),
        "candidate_store_bytes": candidate_paths["closed_measured_pool"].stat().st_size,
        "label_store_sha256": _sha256(label_store_path),
        "label_store_bytes": label_store_path.stat().st_size,
        "candidate_ids_sha256": ids_sha256(candidate_ids),
        "candidate_count": len(candidate_ids),
        "label_candidate_ids_sha256": ids_sha256(label_candidate_ids),
        "label_count": len(label_candidate_ids),
        "label_record_ids_sha256": ids_sha256(
            [str(row["record_id"]) for row in hidden_labels]
        ),
        "label_record_count": len(hidden_labels),
        "selection_freeze_sha256": _sha256(selection_freeze_path),
        "selection_freeze_bytes": selection_freeze_path.stat().st_size,
        "role_policy_sha256": _sha256(role_policy_path),
        "hidden_label_schema_sha256": _sha256(hidden_label_schema_path),
        "canonical_records_sha256": universe_binding["canonical_records_sha256"],
        "structural_records_sha256": universe_binding["structural_records_sha256"],
        "record_ids_sha256": universe_binding["record_ids_sha256"],
        "frozen_before_label_access": True,
        "selection_labels_hidden": True,
    }
    freeze_proof_path = tracks_root / "closed_measured_pool.labels.freeze.json"
    _write_json_exclusive(freeze_proof_path, freeze_proof)

    manifest_paths: Dict[str, Path] = {}
    for _, track_type, _ in TRACK_SPECS:
        path = tracks_root / f"{track_type}.yaml"
        payload = _track_manifest(
            track_type=track_type,
            candidate_store_path=candidate_paths[track_type],
            universe_binding=universe_binding,
            label_store_path=(
                label_store_path if track_type == "closed_measured_pool" else None
            ),
            freeze_proof_path=(
                freeze_proof_path if track_type == "closed_measured_pool" else None
            ),
            hidden_label_schema_path=(
                hidden_label_schema_path
                if track_type == "closed_measured_pool"
                else None
            ),
            selection_freeze_path=(
                selection_freeze_path if track_type == "closed_measured_pool" else None
            ),
        )
        _write_yaml_exclusive(path, payload)
        manifest_paths[track_type] = path

    loaded_tracks = [
        load_track_manifest(manifest_paths[track_type])
        for _, track_type, _ in TRACK_SPECS
    ]
    track_audit = audit_track_roles(
        loaded_tracks,
        eligible_records=eligible_records,
        expected_role_by_record=role_by_record,
    )
    if track_audit.get("gate_passed") is not True:
        raise B0ArtifactBuildError(
            "generated track roles are ambiguous: "
            + json.dumps(track_audit.get("issues", [])[:5])
        )
    label_seal_audit = privileged_verify_track_a_label_seal(
        manifest_paths["closed_measured_pool"],
        expected_role_policy=role_policy,
        expected_d1_acceptance_path=d1_acceptance_path,
    )

    role_matrix_path = tracks_root / "track_role_matrix.yaml"
    _write_yaml_exclusive(role_matrix_path, track_audit)
    claims_path = claims_root / "allowed_unsupported_claims.yaml"
    _write_yaml_exclusive(claims_path, _claims(universe_binding))
    data_card_path = data_card_root / "UTR_EditBench_v2_Data_Card.md"
    _write_text_exclusive(
        data_card_path,
        _data_card_body(
            universe_binding=universe_binding,
            track_audit=track_audit,
            role_policy=role_policy,
            exposure_sha256=_sha256(exposure_ledger_path),
            role_matrix_sha256=_sha256(role_matrix_path),
            claims_sha256=_sha256(claims_path),
            eligible_records=eligible_records,
        ),
    )

    bound_artifacts = {
        "exposure_ledger": (
            exposure_ledger_path.resolve(),
            "d1_data_exposure_ledger.v2",
        ),
        "track_role_matrix": (
            role_matrix_path,
            "utr_track_role_audit.v2",
        ),
        "data_card": (data_card_path, "utr_editbench_data_card.v2"),
        "claims": (claims_path, "utr_b0_claims.v2"),
    }
    artifact_bindings_path = output_root / "artifact_bindings.json"
    artifact_bindings = {
        "schema_version": "utr_b0_artifact_bindings.v2",
        "universe_binding": universe_binding,
        "artifacts": {
            name: {
                "path": _binding_path_value(path, output_root),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "schema_version": schema_version,
            }
            for name, (path, schema_version) in bound_artifacts.items()
        },
    }
    _write_json_exclusive(artifact_bindings_path, artifact_bindings)
    required_artifact_audit = validate_required_artifacts(
        artifact_bindings_path,
        track_audit=track_audit,
        exposure_ledger_path=exposure_ledger_path,
        role_policy=role_policy,
    )
    acceptance_preview = validate_b0_acceptance(
        leakage_reports=leakage_reports,
        exposure_audit=exposure_audit,
        track_audit=track_audit,
        split_manifests=split_manifests,
        track_a_label_seal_audit=label_seal_audit,
        required_artifact_audit=required_artifact_audit,
        d1_exposure_ledger_binding=exposure_binding,
        supplied_leakage_reports_match_recomputation=True,
    )
    if acceptance_preview.get("b0_gate_passed") is not True:
        raise B0ArtifactBuildError(
            "generated bundle fails B0 acceptance: "
            + ", ".join(acceptance_preview.get("failed_gates", []))
        )

    input_refs = {
        "d1_acceptance": _file_ref(d1_acceptance_path),
        "d1_build_manifest": _file_ref(d1["build_manifest_path"]),
        "canonical_records": _file_ref(canonical_records_path),
        "structural_records": _file_ref(structural_records_path),
        "exposure_ledger": _file_ref(exposure_ledger_path),
        "track_a_hidden_label_schema": _file_ref(TRACK_A_HIDDEN_LABEL_SCHEMA),
        "split_manifests": [_file_ref(path) for path in split_manifest_paths],
        "supplied_leakage_reports": [_file_ref(path) for path in leakage_report_paths],
    }
    output_paths = [
        role_policy_path,
        *recomputed_report_paths,
        hidden_label_schema_path,
        *candidate_paths.values(),
        selection_freeze_path,
        label_store_path,
        freeze_proof_path,
        *manifest_paths.values(),
        role_matrix_path,
        claims_path,
        data_card_path,
        artifact_bindings_path,
    ]
    build_manifest = {
        "schema_version": "utr_b0_evaluation_artifact_build.v2",
        "status": "PASS",
        "output_root": str(output_root),
        "role_policy": role_policy,
        "universe_binding": universe_binding,
        "inputs": input_refs,
        "outputs": [_file_ref(path) for path in output_paths],
        "prelabel_selection_freeze": _file_ref(selection_freeze_path),
        "track_role_policy": _file_ref(role_policy_path),
        "recomputed_leakage_reports": [
            _file_ref(path) for path in recomputed_report_paths
        ],
        "leakage_evidence_binding": {
            "supplied_reports_exactly_match_recomputation": True,
            "auditor_bindings": [
                report["auditor_binding"]
                for report in sorted(leakage_reports, key=_split_identity)
            ],
            "supplied_reports": [_file_ref(path) for path in leakage_report_paths],
            "recomputed_reports": [_file_ref(path) for path in recomputed_report_paths],
        },
        "track_a_hidden_label_schema_audit": hidden_label_schema_audit,
        "track_a_label_seal_audit": label_seal_audit,
        "required_artifact_audit": required_artifact_audit,
        "d1_exposure_ledger_binding": exposure_binding,
        "acceptance_preview": acceptance_preview,
        "full_d1_binding": {
            "passed": full_d1_binding.get("passed"),
            "d1_acceptance_sha256": full_d1_binding.get("d1_acceptance_sha256"),
            "build_manifest_sha256": full_d1_binding.get("build_manifest_sha256"),
        },
        "scientific_result_claimed": False,
        "track_b_b0_efficacy_conclusion_allowed": False,
        "track_b_future_formal_generative_evaluation_allowed": True,
        "foundation_status": "UNKNOWN_PENDING_FM0",
    }
    build_manifest_path = output_root / "build_manifest.json"
    _write_json_exclusive(build_manifest_path, build_manifest)
    return {
        "status": "PASS",
        "output_root": str(output_root),
        "build_manifest": str(build_manifest_path),
        "artifact_bindings": str(artifact_bindings_path),
        "track_manifests": {
            track_type: str(path) for track_type, path in manifest_paths.items()
        },
        "track_role_counts": role_policy["record_counts"],
        "b0_gate_preview_passed": True,
        "scientific_result_claimed": False,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-acceptance", type=Path, required=True)
    parser.add_argument("--canonical-records", type=Path, required=True)
    parser.add_argument("--structural-records", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, action="append", required=True)
    parser.add_argument("--leakage-report", type=Path, action="append", required=True)
    parser.add_argument("--exposure-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output_root_preexisted = args.output_root.resolve().exists()
    try:
        result = build_b0_evaluation_artifacts(
            d1_acceptance_path=args.d1_acceptance,
            canonical_records_path=args.canonical_records,
            structural_records_path=args.structural_records,
            split_manifest_paths=args.split_manifest,
            leakage_report_paths=args.leakage_report,
            exposure_ledger_path=args.exposure_ledger,
            output_root=args.output_root,
        )
    except Exception as exc:
        root = args.output_root.resolve()
        if not output_root_preexisted and root.is_dir():
            failure_path = root / "build_failure.json"
            if not failure_path.exists():
                try:
                    _write_json_exclusive(
                        failure_path,
                        {
                            "schema_version": ("utr_b0_evaluation_artifact_failure.v2"),
                            "status": "FAILED_WITH_EVIDENCE",
                            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "outputs_preserved": True,
                        },
                    )
                except OSError:
                    pass
        print(f"B0 evaluation artifact build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
