#!/usr/bin/env python3
"""Validate exact B0 gates, physical seals, and required artifact bindings."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.utr_benchmark_v2.leakage import normalize_foundation_exposure
from data.utr_benchmark_v2.d1_builder import ACTIVE_DATASET_POLICIES
from data.utr_benchmark_v2.d1_builder import D1_SCOPE_DATASETS
from data.utr_benchmark_v2.split_graph import METADATA_DIMENSIONS
from data.utr_benchmark_v2.split_graph import expected_partition_leakage_contract
from data.utr_benchmark_v2.track_loader import audit_track_roles
from data.utr_benchmark_v2.track_loader import load_track_manifest
from data.utr_benchmark_v2.track_loader import (
    privileged_verify_track_a_label_seal,
)
from scripts.data.audit_b0_leakage import load_json
from scripts.data.audit_b0_leakage import (
    build_bound_structural_recompute_cache,
)
from scripts.data.audit_b0_leakage import (
    recompute_bound_leakage_report,
)
from scripts.data.build_b0_splits import load_jsonl
from scripts.data.build_b0_splits import sha256_file
from scripts.data.build_b0_splits import write_json_exclusive


ZERO_GATE_COUNTS = {
    "unexplained_overlap_zero": "unexplained_overlap_count",
    "exact_source_overlap_zero": "exact_source_leakage_count",
    "exact_candidate_overlap_zero": "exact_candidate_leakage_count",
    "reverse_edge_leakage_zero": "reverse_edge_leakage_count",
    "path_leakage_zero": "path_leakage_count",
    "near_neighbor_leakage_zero": "near_neighbor_leakage_count",
    "final_endpoint_as_train_intermediate_zero": (
        "final_endpoint_as_train_intermediate_count"
    ),
    "required_axis_overlap_zero": "required_axis_overlap_count",
}
EXPECTED_SPLIT_IDENTITIES = {
    ("source_disjoint", "five_utr", None),
    ("study_disjoint", "five_utr", None),
    ("source_disjoint", "three_utr", None),
    ("study_disjoint", "three_utr", None),
    ("cross_region_transfer", "five_utr", "three_utr"),
}
EXPECTED_PARTITION_IDS = {
    ("source_disjoint", "five_utr", None): (
        "source_disjoint:five_utr",
        "sequence_cluster_disjoint:five_utr",
        "scaffold_disjoint:five_utr",
        "gene_disjoint:five_utr",
        "context_disjoint:five_utr",
        "barcode_batch_disjoint:five_utr",
        "library_batch_disjoint:five_utr",
    ),
    ("study_disjoint", "five_utr", None): (
        "loso:GSE114002",
        "loso:GSE217518",
    ),
    ("source_disjoint", "three_utr", None): (
        "source_disjoint:three_utr",
        "sequence_cluster_disjoint:three_utr",
        "scaffold_disjoint:three_utr",
        "gene_disjoint:three_utr",
        "context_disjoint:three_utr",
        "barcode_batch_disjoint:three_utr",
        "library_batch_disjoint:three_utr",
    ),
    ("study_disjoint", "three_utr", None): (
        "loso:GSE217518",
        "loso:GSE200304",
    ),
    ("cross_region_transfer", "five_utr", "three_utr"): (
        "within_study:GSE217518:five_to_three",
        "within_study:GSE217518:three_to_five",
        "cross_study:GSE114002_five_to_GSE200304_three",
    ),
}
_SOURCE_AXIS_BY_PARTITION_PREFIX = {
    "source_disjoint": "near_neighbor_cluster",
    "sequence_cluster_disjoint": "sequence_cluster",
    "scaffold_disjoint": "scaffold_group",
    "gene_disjoint": "gene_group",
    "context_disjoint": "context_group",
    "barcode_batch_disjoint": "barcode_batch",
    "library_batch_disjoint": "library_batch",
}
_SPLIT_UNIVERSE_FIELDS = (
    "canonical_records_sha256",
    "structural_records_sha256",
    "structural_records_bytes",
    "canonical_record_ids_sha256",
    "canonical_record_count",
    "structural_record_ids_sha256",
    "structural_record_count",
    "structural_content_sha256",
)
_BOUND_ARTIFACTS = {
    "exposure_ledger": "d1_data_exposure_ledger.v2",
    "track_role_matrix": "utr_track_role_audit.v2",
    "data_card": "utr_editbench_data_card.v2",
    "claims": "utr_b0_claims.v2",
}
_BINDING_REF_FIELDS = {"path", "sha256", "bytes", "schema_version"}
_CLAIMS_FIELDS = {
    "schema_version",
    "universe_binding",
    "foundation_status",
    "allowed_claim",
    "requires_fm0_reaudit",
    "gse246381_role",
    "track_claims",
    "allowed_claims",
    "unsupported_capabilities",
}
_RETROSPECTIVE_ROLE = "historically_exposed_retrospective_external_stress_test"


def _split_identity(payload: Mapping[str, Any]) -> Tuple[str, str, Any]:
    split_kind = str(payload.get("split_kind") or "")
    if split_kind == "cross_region_transfer":
        return (
            split_kind,
            str(payload.get("source_region") or ""),
            str(payload.get("target_region") or ""),
        )
    return (split_kind, str(payload.get("region") or ""), None)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def compute_exposure_coverage(
    records: Sequence[Mapping[str, Any]],
    ledger_entries: Sequence[Mapping[str, Any]],
    *,
    identity_level: str,
    required_ledger_identities: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Compute explicit record- or dataset-level exposure-ledger coverage.

    ``required_ledger_identities`` permits a frozen, audited ledger scope to be
    a strict superset of the record-bearing benchmark universe.  This is
    required for D1: blocked and excluded datasets remain in the exposure
    ledger even though they correctly contribute zero canonical B0 records.
    Arbitrary extras still fail closed because the observed ledger identities
    must equal the supplied frozen scope exactly.
    """

    if identity_level not in ("record_id", "dataset_id"):
        raise ValueError("identity_level must be record_id or dataset_id")
    if not records:
        raise ValueError("canonical record universe must be non-empty")
    if not ledger_entries:
        expected = sorted(
            {str(record.get(identity_level) or "").strip() for record in records}
        )
        return {
            "coverage": 0.0,
            "covered": 0,
            "expected": len(expected),
            "identity_level": identity_level,
            "missing": expected[:100],
            "extra": [],
            "gate_passed": False,
        }

    expected = {str(record.get(identity_level) or "").strip() for record in records}
    if "" in expected:
        raise ValueError(f"canonical records lack {identity_level}")
    required_scope: set[str] | None = None
    if required_ledger_identities is not None:
        required_scope = {
            str(identity).strip() for identity in required_ledger_identities
        }
        if "" in required_scope:
            raise ValueError("required exposure-ledger scope contains blank identity")
        if not expected <= required_scope:
            raise ValueError(
                "canonical exposure identities are outside the frozen ledger scope"
            )

    covered = set()
    observed_identities = set()
    for entry in ledger_entries:
        identity = str(entry.get(identity_level) or "").strip()
        if not identity:
            raise ValueError(f"exposure ledger mixes or omits frozen {identity_level}")
        if identity in observed_identities:
            raise ValueError(f"duplicate exposure ledger identity: {identity}")
        observed_identities.add(identity)
        disposition_values = [
            entry.get(key)
            for key in (
                "exposure_grade",
                "historical_exposure",
                "exposure_status",
                "status",
            )
            if key in entry
        ]
        if any(
            isinstance(value, str) and value.strip() for value in disposition_values
        ):
            covered.add(identity)
    matched = expected & covered
    missing = sorted(expected - covered)
    extra = sorted(observed_identities - expected)
    missing_from_required_scope = (
        sorted(required_scope - observed_identities)
        if required_scope is not None
        else []
    )
    outside_required_scope = (
        sorted(observed_identities - required_scope)
        if required_scope is not None
        else []
    )
    coverage = len(matched) / len(expected)
    exact_scope_gate = (
        not extra
        if required_scope is None
        else not missing_from_required_scope and not outside_required_scope
    )
    return {
        "coverage": coverage,
        "covered": len(matched),
        "expected": len(expected),
        "identity_level": identity_level,
        "missing": missing[:100],
        "extra": extra[:100],
        "required_ledger_identity_count": (
            len(required_scope) if required_scope is not None else None
        ),
        "missing_from_required_ledger_scope": missing_from_required_scope[:100],
        "outside_required_ledger_scope": outside_required_scope[:100],
        "ledger_scope_gate_passed": exact_scope_gate,
        "gate_passed": coverage == 1.0 and not missing and exact_scope_gate,
    }


def _common_split_universe(
    manifests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    if not manifests:
        return None
    universes = [
        {field: manifest.get(field) for field in _SPLIT_UNIVERSE_FIELDS}
        for manifest in manifests
    ]
    first = universes[0]
    if any(universe != first for universe in universes[1:]):
        return None
    if not all(
        _is_sha256(first[field])
        for field in (
            "canonical_records_sha256",
            "structural_records_sha256",
            "canonical_record_ids_sha256",
            "structural_record_ids_sha256",
            "structural_content_sha256",
        )
    ):
        return None
    if (
        not _nonnegative_int(first["canonical_record_count"])
        or first["canonical_record_count"] < 1
        or not _nonnegative_int(first["structural_records_bytes"])
        or first["structural_records_bytes"] < 1
        or first["structural_record_count"] != first["canonical_record_count"]
        or first["structural_record_ids_sha256"] != first["canonical_record_ids_sha256"]
    ):
        return None
    return first


def _ids_sha256(values: Sequence[str]) -> str:
    normalized = sorted(values)
    if len(normalized) != len(set(normalized)):
        raise ValueError("record identity universe contains duplicates")
    body = ("\n".join(normalized) + "\n") if normalized else ""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _accounted_record_sets(
    payload: Mapping[str, Any],
    *,
    full_count: int,
    full_ids_sha256: str,
) -> Tuple[set[str], set[str], List[str]]:
    failures: List[str] = []
    eligible_rows = payload.get("eligible_records")
    excluded_rows = payload.get("excluded_records")
    if not isinstance(eligible_rows, list) or not isinstance(excluded_rows, list):
        return set(), set(), ["record_accounting_missing"]

    def extract(rows: Sequence[Any], *, require_eligible_reason: bool) -> List[str]:
        values: List[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                failures.append("record_accounting_row_invalid")
                continue
            record_id = row.get("record_id")
            reason = row.get("reason")
            if (
                not isinstance(record_id, str)
                or not record_id.strip()
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                failures.append("record_accounting_row_invalid")
                continue
            if require_eligible_reason and reason != "eligible_intervention_record":
                failures.append("eligible_record_reason_invalid")
            values.append(record_id.strip())
        if len(values) != len(set(values)):
            failures.append("record_accounting_duplicate")
        return values

    eligible_values = extract(eligible_rows, require_eligible_reason=True)
    excluded_values = extract(excluded_rows, require_eligible_reason=False)
    eligible = set(eligible_values)
    excluded = set(excluded_values)
    if eligible & excluded:
        failures.append("eligible_excluded_overlap")
    accounted = eligible | excluded
    if (
        len(accounted) != full_count
        or _ids_sha256(sorted(accounted)) != full_ids_sha256
    ):
        failures.append("record_accounting_not_full_universe")
    if (
        payload.get("record_count") != len(eligible)
        or payload.get("record_ids_sha256") != _ids_sha256(sorted(eligible))
        or payload.get("excluded_record_count") != len(excluded)
    ):
        failures.append("record_accounting_summary_mismatch")
    return eligible, excluded, failures


def _derive_role_record_universe(
    manifest_by_identity: Mapping[Tuple[str, str, Any], Mapping[str, Any]],
    split_universe: Mapping[str, Any] | None,
) -> Tuple[Dict[str, Any] | None, set[str], set[str], List[str]]:
    failures: List[str] = []
    if split_universe is None:
        return None, set(), set(), ["full_split_universe_missing"]
    full_count = split_universe["canonical_record_count"]
    full_ids_sha = split_universe["canonical_record_ids_sha256"]
    accounting: Dict[Tuple[str, str, Any], Tuple[set[str], set[str]]] = {}
    full_ids: set[str] | None = None
    for identity in EXPECTED_SPLIT_IDENTITIES:
        manifest = manifest_by_identity.get(identity)
        if manifest is None:
            failures.append("required_manifest_missing_for_role_universe")
            continue
        eligible, excluded, local_failures = _accounted_record_sets(
            manifest,
            full_count=full_count,
            full_ids_sha256=full_ids_sha,
        )
        failures.extend(local_failures)
        accounting[identity] = (eligible, excluded)
        accounted = eligible | excluded
        if full_ids is None:
            full_ids = accounted
        elif accounted != full_ids:
            failures.append("split_accounting_full_identity_mismatch")

    five_source = accounting.get(("source_disjoint", "five_utr", None), (set(), set()))[
        0
    ]
    five_study = accounting.get(("study_disjoint", "five_utr", None), (set(), set()))[0]
    three_source = accounting.get(
        ("source_disjoint", "three_utr", None), (set(), set())
    )[0]
    three_study = accounting.get(("study_disjoint", "three_utr", None), (set(), set()))[
        0
    ]
    cross = accounting.get(
        ("cross_region_transfer", "five_utr", "three_utr"),
        (set(), set()),
    )[0]
    if five_source != five_study or three_source != three_study:
        failures.append("source_and_study_eligible_universe_mismatch")
    if five_source & three_source:
        failures.append("five_three_role_record_overlap")
    role_ids = five_source | three_source
    if cross != role_ids:
        failures.append("cross_region_eligible_universe_mismatch")
    if not role_ids:
        failures.append("eligible_track_role_universe_empty")
    global_excluded = (full_ids or set()) - role_ids
    universe = {
        "record_ids_sha256": _ids_sha256(sorted(role_ids)),
        "record_count": len(role_ids),
        "excluded_record_ids_sha256": _ids_sha256(sorted(global_excluded)),
        "excluded_record_count": len(global_excluded),
    }
    return (
        universe if not failures else None,
        role_ids,
        full_ids or set(),
        failures,
    )


def _track_split_universe_matches(
    track_audit: Mapping[str, Any],
    split_universe: Mapping[str, Any] | None,
    role_universe: Mapping[str, Any] | None,
) -> bool:
    if split_universe is None or role_universe is None:
        return False
    binding = track_audit.get("universe_binding")
    if not isinstance(binding, Mapping):
        return False
    return (
        binding.get("canonical_records_sha256")
        == split_universe["canonical_records_sha256"]
        and binding.get("structural_records_sha256")
        == split_universe["structural_records_sha256"]
        and binding.get("record_ids_sha256") == role_universe["record_ids_sha256"]
        and binding.get("record_count") == role_universe["record_count"]
    )


def _partition_sha256(partition: Mapping[str, Any]) -> str:
    frozen = {
        key: value for key, value in partition.items() if key != "partition_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            frozen,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _partition_map(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Mapping[str, Any]], bool]:
    raw = payload.get("partitions")
    if not isinstance(raw, list):
        return {}, False
    result: Dict[str, Mapping[str, Any]] = {}
    valid = True
    for item in raw:
        if not isinstance(item, Mapping):
            valid = False
            continue
        partition_id = item.get("partition_id")
        if (
            not isinstance(partition_id, str)
            or not partition_id
            or partition_id in result
        ):
            valid = False
            continue
        result[partition_id] = item
    return result, valid


def _partition_evidence_failures(
    manifest_by_identity: Mapping[Tuple[str, str, Any], Mapping[str, Any]],
    report_by_identity: Mapping[Tuple[str, str, Any], Mapping[str, Any]],
    *,
    role_record_ids: set[str],
    full_record_ids: set[str],
) -> List[str]:
    failures: List[str] = []
    for identity, expected_ids in EXPECTED_PARTITION_IDS.items():
        manifest = manifest_by_identity.get(identity)
        report = report_by_identity.get(identity)
        if manifest is None or report is None:
            continue
        required = manifest.get("required_partition_ids")
        report_required = report.get("required_partition_ids")
        if not (
            isinstance(required, list)
            and tuple(required) == expected_ids
            and len(required) == len(set(required))
            and isinstance(report_required, list)
            and tuple(report_required) == expected_ids
        ):
            failures.append("all_required_partitions_present")
        manifest_parts, manifest_shape_valid = _partition_map(manifest)
        report_parts, report_shape_valid = _partition_map(report)
        if not (
            manifest_shape_valid
            and report_shape_valid
            and tuple(manifest_parts) == expected_ids
            and tuple(report_parts) == expected_ids
        ):
            failures.append("all_required_partitions_present")

        split_kind = identity[0]
        if split_kind == "study_disjoint":
            if manifest.get("folds") != manifest.get("partitions"):
                failures.append("study_fold_alias_binding")
        elif split_kind == "cross_region_transfer":
            if manifest.get("strata") != manifest.get("partitions"):
                failures.append("cross_region_strata_alias_binding")

        for partition_id in expected_ids:
            partition = manifest_parts.get(partition_id)
            partition_report = report_parts.get(partition_id)
            if partition is None or partition_report is None:
                continue
            try:
                expected_leakage_contract = expected_partition_leakage_contract(
                    split_kind,
                    region=(
                        identity[1] if split_kind != "cross_region_transfer" else None
                    ),
                    partition_id=partition_id,
                )
            except ValueError:
                expected_leakage_contract = {}
                failures.append("per_partition_frozen_leakage_contract")
            if not expected_leakage_contract or any(
                partition.get(field) != expected
                for field, expected in expected_leakage_contract.items()
            ):
                failures.append("per_partition_frozen_leakage_contract")
            declared_sha = partition.get("partition_sha256")
            if (
                partition.get("status") != "READY"
                or not _is_sha256(declared_sha)
                or declared_sha != _partition_sha256(partition)
                or partition_report.get("split_partition_sha256") != declared_sha
            ):
                failures.append("split_partition_hash_binding")
            roles = partition.get("roles")
            if isinstance(roles, Mapping):
                if set(roles) != {"train", "validation", "test"}:
                    failures.append("partition_role_schema")
                role_occurrences: Dict[str, List[str]] = {}
                for role, raw_ids in roles.items():
                    if not isinstance(raw_ids, list):
                        failures.append("partition_role_schema")
                        continue
                    for raw_id in raw_ids:
                        record_id = str(raw_id)
                        role_occurrences.setdefault(record_id, []).append(str(role))
                if any(
                    len(set(role_names)) != 1
                    for role_names in role_occurrences.values()
                ):
                    failures.append("partition_record_role_ambiguity")
                for role in ("train", "validation"):
                    role_ids = roles.get(role, [])
                    if isinstance(role_ids, list) and any(
                        str(record_id).startswith("GSE246381:")
                        for record_id in role_ids
                    ):
                        failures.append("gse246381_retrospective_external_stress_only")
            else:
                failures.append("split_partition_hash_binding")
                role_occurrences = {}
            eligible, excluded, accounting_failures = _accounted_record_sets(
                partition,
                full_count=len(full_record_ids),
                full_ids_sha256=_ids_sha256(sorted(full_record_ids)),
            )
            if accounting_failures:
                failures.append("partition_reasoned_accounting")
            if not eligible <= role_record_ids:
                failures.append("partition_outside_eligible_role_universe")
            if set(role_occurrences) != eligible:
                failures.append("partition_roles_not_complete")
            if set(role_occurrences) & excluded:
                failures.append("excluded_record_has_track_or_split_role")
            counts = partition_report.get("counts")
            gates = partition_report.get("acceptance_gates")
            if not isinstance(counts, Mapping) or not isinstance(gates, Mapping):
                failures.append("per_partition_gate_evidence")
                continue
            for gate_name, count_name in ZERO_GATE_COUNTS.items():
                count = counts.get(count_name)
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count != 0
                    or gates.get(gate_name) is not True
                ):
                    failures.append(f"per_partition_{gate_name}")
            required_axes = partition.get("required_disjoint_axes")
            required_pairs = partition.get("required_disjoint_role_pairs")
            required_status = partition_report.get("required_axis_status")
            if not (
                isinstance(required_axes, list)
                and required_axes
                and len(required_axes) == len(set(required_axes))
                and isinstance(required_pairs, Mapping)
                and set(required_pairs) == set(required_axes)
                and isinstance(required_status, Mapping)
                and set(required_status) == set(required_axes)
            ):
                failures.append("per_partition_required_axis_schema")
            else:
                for axis in required_axes:
                    status = required_status.get(axis)
                    declared_axis_pairs = required_pairs.get(axis)
                    observed_axis_pairs = (
                        status.get("required_role_pairs")
                        if isinstance(status, Mapping)
                        else None
                    )
                    expected_declared_pairs = expected_leakage_contract.get(
                        "required_disjoint_role_pairs", {}
                    ).get(axis)
                    expected_report_pairs = (
                        [
                            list(pair)
                            for pair in sorted(
                                tuple(pair) for pair in expected_declared_pairs
                            )
                        ]
                        if isinstance(expected_declared_pairs, list)
                        else None
                    )
                    pair_bindings_match = (
                        isinstance(declared_axis_pairs, list)
                        and isinstance(observed_axis_pairs, list)
                        and declared_axis_pairs == expected_declared_pairs
                        and observed_axis_pairs == expected_report_pairs
                    )
                    if not (
                        isinstance(status, Mapping)
                        and status.get("raw_overlap_count") == 0
                        and status.get("gate_passed") is True
                        and pair_bindings_match
                    ):
                        failures.append("per_partition_required_axis_raw_overlap_zero")
                mandatory_firewalls = {
                    "near_neighbor_cluster",
                    "source_group",
                    "scaffold_group",
                }
                if not mandatory_firewalls <= set(required_axes):
                    failures.append("per_partition_dense_graph_firewalls_required")
                if split_kind == "source_disjoint":
                    prefix = partition_id.split(":", 1)[0]
                    expected_axis = _SOURCE_AXIS_BY_PARTITION_PREFIX.get(prefix)
                    if (
                        expected_axis is None
                        or expected_axis not in required_axes
                        or partition.get("axis_partition") is not True
                    ):
                        failures.append("per_partition_source_axis_binding")

            metadata_status = partition_report.get("metadata_axis_status")
            if not (
                isinstance(metadata_status, Mapping)
                and set(metadata_status) == set(METADATA_DIMENSIONS)
            ):
                failures.append("per_partition_metadata_axis_accounting")
            else:
                raw_total = 0
                explained_total = 0
                unexplained_total = 0
                for axis in METADATA_DIMENSIONS:
                    status = metadata_status.get(axis)
                    if not isinstance(status, Mapping):
                        failures.append("per_partition_metadata_axis_accounting")
                        continue
                    raw = status.get("overlap_count")
                    explained = status.get("explained_overlap_count")
                    unexplained = status.get("unexplained_overlap_count")
                    if (
                        not all(
                            isinstance(value, int)
                            and not isinstance(value, bool)
                            and value >= 0
                            for value in (raw, explained, unexplained)
                        )
                        or raw != explained + unexplained
                    ):
                        failures.append("per_partition_metadata_axis_accounting")
                        continue
                    raw_total += raw
                    explained_total += explained
                    unexplained_total += unexplained
                if (
                    raw_total != counts.get("metadata_overlap_count")
                    or explained_total != counts.get("explained_metadata_overlap_count")
                    or unexplained_total
                    != counts.get("unexplained_metadata_overlap_count")
                ):
                    failures.append("per_partition_metadata_axis_accounting")
            foundation_raw = partition_report.get("foundation_pretraining_overlap")
            foundation = (
                normalize_foundation_exposure(foundation_raw)
                if isinstance(foundation_raw, Mapping)
                else {}
            )
            if not (
                foundation.get("status") == "UNKNOWN_PENDING_FM0"
                and foundation.get("foundation_selected") is False
                and foundation.get("allowed_claim") == "NONE"
                and foundation.get("re_audit_required") is True
                and foundation.get("gate_passed") is True
                and gates.get("foundation_overlap_gate") is True
            ):
                failures.append("per_partition_foundation_state_unknown_pending_fm0")
    return failures


def _strict_pending_fm0(
    leakage_reports: Sequence[Mapping[str, Any]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    normalized: List[Dict[str, Any]] = []
    valid = bool(leakage_reports)
    for report in leakage_reports:
        raw = report.get("foundation_pretraining_overlap")
        if not isinstance(raw, Mapping):
            valid = False
            continue
        item = normalize_foundation_exposure(raw)
        normalized.append(item)
        gates = report.get("acceptance_gates")
        if not (
            item.get("status") == "UNKNOWN_PENDING_FM0"
            and item.get("foundation_selected") is False
            and item.get("allowed_claim") == "NONE"
            and item.get("re_audit_required") is True
            and item.get("gate_passed") is True
            and isinstance(gates, Mapping)
            and gates.get("foundation_overlap_gate") is True
        ):
            valid = False
    return valid and len(normalized) == len(leakage_reports), normalized


def _track_a_seal_matches(
    seal: Mapping[str, Any],
    track_audit: Mapping[str, Any],
    split_universe: Mapping[str, Any] | None,
    role_universe: Mapping[str, Any] | None,
    d1_exposure_ledger_binding: Mapping[str, Any] | None,
) -> bool:
    if not (
        seal.get("schema_version") == "utr_track_a_label_seal_audit.v2"
        and seal.get("gate_passed") is True
        and seal.get("candidate_label_bijection") is True
        and seal.get("record_label_bijection") is True
        and seal.get("strict_hidden_label_schema_passed") is True
        and seal.get("paired_finite_measured_labels") is True
        and seal.get("canonical_identity_binding_passed") is True
        and seal.get("d1_acceptance_binding_passed") is True
        and seal.get("current_d1_chain_binding_passed") is True
        and seal.get("role_policy_exact_binding_passed") is True
        and _is_sha256(seal.get("label_store_sha256"))
        and _nonnegative_int(seal.get("label_store_bytes"))
        and seal.get("label_store_bytes", 0) > 0
        and _is_sha256(seal.get("freeze_proof_sha256"))
        and _is_sha256(seal.get("selection_freeze_sha256"))
        and _is_sha256(seal.get("role_policy_sha256"))
        and _is_sha256(seal.get("hidden_label_schema_sha256"))
        and _is_sha256(seal.get("d1_acceptance_sha256"))
        and _is_sha256(seal.get("d1_build_manifest_sha256"))
    ):
        return False
    tracks = track_audit.get("tracks")
    if not isinstance(tracks, list):
        return False
    track_a_rows = [
        row
        for row in tracks
        if isinstance(row, Mapping) and row.get("track_type") == "closed_measured_pool"
    ]
    if len(track_a_rows) != 1:
        return False
    track_a = track_a_rows[0]
    if not (
        seal.get("track_id") == track_a.get("track_id")
        and seal.get("label_store_sha256") == track_a.get("label_store_sha256")
        and seal.get("label_store_bytes") == track_a.get("label_store_bytes")
        and seal.get("freeze_proof_sha256") == track_a.get("label_freeze_proof_sha256")
        and seal.get("selection_freeze_sha256")
        == track_a.get("selection_freeze_sha256")
        and seal.get("hidden_label_schema_sha256") == track_a.get("label_schema_sha256")
        and seal.get("candidate_ids_sha256") == track_a.get("candidate_ids_sha256")
        and seal.get("candidate_count") == track_a.get("candidate_count")
    ):
        return False
    return (
        split_universe is not None
        and role_universe is not None
        and seal.get("canonical_records_sha256")
        == split_universe["canonical_records_sha256"]
        and seal.get("structural_records_sha256")
        == split_universe["structural_records_sha256"]
        and seal.get("record_ids_sha256") == role_universe["record_ids_sha256"]
        and isinstance(d1_exposure_ledger_binding, Mapping)
        and seal.get("d1_acceptance_sha256")
        == d1_exposure_ledger_binding.get("d1_acceptance_sha256")
        and seal.get("d1_build_manifest_sha256")
        == d1_exposure_ledger_binding.get("d1_build_manifest_sha256")
    )


def _resolve_bound_file(root: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_absolute():
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"{field} escapes binding directory") from exc
    if not resolved.is_file():
        raise ValueError(f"{field} does not exist: {resolved}")
    return resolved


def _load_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain one mapping")
    return dict(value)


def _validate_exposure_ledger_shape(path: Path) -> bool:
    required = {
        "dataset_id",
        "status",
        "historical_exposure",
        "exposure_grade",
        "allowed_uses",
        "forbidden_uses",
        "read_final_labels",
        "provenance_complete",
        "reason_code",
    }
    try:
        rows = load_jsonl(path)
    except (OSError, ValueError):
        return False
    dataset_ids = []
    for row in rows:
        if set(row) != required:
            return False
        dataset_id = row.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            return False
        dataset_ids.append(dataset_id)
        allowed = row.get("allowed_uses")
        forbidden = row.get("forbidden_uses")
        if not (
            isinstance(allowed, list)
            and len(allowed) == len(set(allowed))
            and all(isinstance(value, str) and value.strip() for value in allowed)
            and isinstance(forbidden, list)
            and len(forbidden) == len(set(forbidden))
            and all(isinstance(value, str) and value.strip() for value in forbidden)
            and isinstance(row.get("read_final_labels"), bool)
            and isinstance(row.get("provenance_complete"), bool)
        ):
            return False
        status = row.get("status")
        if status not in {"accepted", "blocked"}:
            return False
        if not all(
            isinstance(row.get(field), str) and row[field].strip()
            for field in ("historical_exposure", "exposure_grade")
        ):
            return False
        reason = row.get("reason_code")
        if status == "accepted" and reason is not None:
            return False
        if status == "blocked" and (
            not isinstance(reason, str)
            or not reason.strip()
            or reason != reason.upper()
        ):
            return False
    row_by_dataset = {str(row["dataset_id"]): row for row in rows}
    retrospective = row_by_dataset.get("GSE246381")
    retrospective_policy = ACTIVE_DATASET_POLICIES["GSE246381"]
    retrospective_valid = (
        isinstance(retrospective, Mapping)
        and retrospective.get("historical_exposure")
        == retrospective_policy["historical_exposure"]
        and retrospective.get("exposure_grade")
        == retrospective_policy["exposure_grade"]
        and retrospective.get("allowed_uses") == retrospective_policy["allowed_uses"]
        and retrospective.get("forbidden_uses")
        == retrospective_policy["forbidden_uses"]
    )
    return (
        len(dataset_ids) == len(set(dataset_ids))
        and set(dataset_ids) == set(D1_SCOPE_DATASETS)
        and retrospective_valid
    )


def validate_d1_exposure_ledger_binding(
    d1_acceptance_path: Path,
    exposure_ledger_path: Path,
) -> Dict[str, Any]:
    """Bind the B0 ledger input to D1's accepted, recomputed artifact."""

    acceptance_path = d1_acceptance_path.resolve()
    ledger_path = exposure_ledger_path.resolve()
    failures: List[str] = []
    try:
        acceptance = load_json(acceptance_path)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return {
            "schema_version": "utr_b0_d1_exposure_binding.v2",
            "gate_passed": False,
            "failures": [f"d1_acceptance_invalid:{type(exc).__name__}"],
        }

    if not (
        acceptance.get("phase_gate_passed") is True
        and acceptance.get("fixture_mode") is False
        and acceptance.get("structural_validation_passed") is True
        and isinstance(acceptance.get("required_artifact_validation"), Mapping)
        and acceptance["required_artifact_validation"].get("passed") is True
    ):
        failures.append("d1_acceptance_gate")

    stage_root = Path(str(acceptance.get("stage_d1_root") or ""))
    expected_build_path = (stage_root / "build_manifest.json").resolve()
    required_validation = acceptance.get("required_artifact_validation")
    build_ref = (
        required_validation.get("build_manifest")
        if isinstance(required_validation, Mapping)
        else None
    )
    if (
        not isinstance(build_ref, Mapping)
        or set(build_ref) != {"path", "bytes", "sha256"}
        or Path(str(build_ref.get("path") or "")).resolve() != expected_build_path
        or not expected_build_path.is_file()
        or expected_build_path.stat().st_size != build_ref.get("bytes")
        or sha256_file(expected_build_path) != build_ref.get("sha256")
    ):
        failures.append("d1_build_manifest_binding")
        build_manifest: Dict[str, Any] = {}
    else:
        try:
            build_manifest = load_json(expected_build_path)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
            failures.append("d1_build_manifest_invalid")
            build_manifest = {}

    relative = "data/data_exposure_ledger.jsonl"
    required_artifacts = build_manifest.get("required_artifacts")
    ledger_ref = (
        required_artifacts.get(relative)
        if isinstance(required_artifacts, Mapping)
        else None
    )
    if (
        not isinstance(ledger_ref, Mapping)
        or set(ledger_ref) != {"path", "bytes", "sha256"}
        or not isinstance(ledger_ref.get("path"), str)
        or not Path(ledger_ref["path"]).is_absolute()
        or Path(ledger_ref["path"]).resolve() != ledger_path
        or not ledger_path.is_file()
        or ledger_path.stat().st_size != ledger_ref.get("bytes")
        or sha256_file(ledger_path) != ledger_ref.get("sha256")
    ):
        failures.append("d1_exposure_ledger_build_manifest_binding")

    accepted_artifacts = (
        required_validation.get("artifacts")
        if isinstance(required_validation, Mapping)
        else None
    )
    accepted_ledger = (
        accepted_artifacts.get(relative)
        if isinstance(accepted_artifacts, Mapping)
        else None
    )
    if not (
        isinstance(accepted_ledger, Mapping)
        and accepted_ledger.get("exists") is True
        and Path(str(accepted_ledger.get("path") or "")).resolve() == ledger_path
        and accepted_ledger.get("bytes")
        == (ledger_path.stat().st_size if ledger_path.is_file() else None)
        and accepted_ledger.get("sha256")
        == (sha256_file(ledger_path) if ledger_path.is_file() else None)
        and accepted_ledger.get("declared") == ledger_ref
    ):
        failures.append("d1_exposure_ledger_acceptance_binding")

    shape_valid = ledger_path.is_file() and _validate_exposure_ledger_shape(ledger_path)
    if not shape_valid:
        failures.append("d1_exposure_ledger_semantics")
    return {
        "schema_version": "utr_b0_d1_exposure_binding.v2",
        "gate_passed": not failures,
        "failures": sorted(set(failures)),
        "d1_acceptance_path": str(acceptance_path),
        "d1_acceptance_sha256": (
            sha256_file(acceptance_path) if acceptance_path.is_file() else None
        ),
        "d1_build_manifest_path": str(expected_build_path),
        "d1_build_manifest_sha256": (
            sha256_file(expected_build_path) if expected_build_path.is_file() else None
        ),
        "exposure_ledger_path": str(ledger_path),
        "exposure_ledger_bytes": (
            ledger_path.stat().st_size if ledger_path.is_file() else None
        ),
        "exposure_ledger_sha256": (
            sha256_file(ledger_path) if ledger_path.is_file() else None
        ),
        "ledger_semantics_valid": shape_valid,
    }


def _validate_claims(
    claims: Mapping[str, Any],
    universe_binding: Mapping[str, Any],
) -> Dict[str, Any]:
    valid = set(claims) == _CLAIMS_FIELDS
    valid = valid and claims.get("schema_version") == "utr_b0_claims.v2"
    valid = valid and claims.get("universe_binding") == universe_binding
    valid = valid and (
        claims.get("foundation_status") == "UNKNOWN_PENDING_FM0"
        and claims.get("allowed_claim") == "NONE"
        and claims.get("requires_fm0_reaudit") is True
        and claims.get("gse246381_role") == _RETROSPECTIVE_ROLE
    )
    allowed = claims.get("allowed_claims")
    unsupported = claims.get("unsupported_capabilities")
    valid = valid and (
        isinstance(allowed, list)
        and bool(allowed)
        and all(isinstance(value, str) and value.strip() for value in allowed)
        and isinstance(unsupported, list)
        and bool(unsupported)
        and all(isinstance(value, str) and value.strip() for value in unsupported)
    )
    serialized_allowed = " ".join(allowed or []).lower()
    if any(
        phrase in serialized_allowed
        for phrase in (
            "full legal action-space regret",
            "measured improvement",
            "foundation unseen",
            "state of the art",
            "sota",
        )
    ):
        valid = False
    track_claims = claims.get("track_claims")
    required_tracks = {
        "closed_measured_pool",
        "heldout_generative",
        "open_legal_generation",
    }
    valid = valid and (
        isinstance(track_claims, Mapping) and set(track_claims) == required_tracks
    )
    if isinstance(track_claims, Mapping):
        track_a = track_claims.get("closed_measured_pool")
        track_b = track_claims.get("heldout_generative")
        track_c = track_claims.get("open_legal_generation")
        valid = valid and (
            isinstance(track_a, Mapping)
            and track_a.get("metric_name") == "observed_pool_normalized_regret"
            and track_a.get("process_order_attestation_only") is True
            and "full_legal_action_space_regret" in track_a.get("forbidden_claims", [])
            and isinstance(track_b, Mapping)
            and track_b.get("b0_efficacy_conclusion_allowed") is False
            and track_b.get("future_formal_generative_evaluation_allowed") is True
            and "later contract-gated formal generative evaluation"
            in str(track_b.get("evidence_scope") or "")
            and "measured_functional_improvement" in track_b.get("forbidden_claims", [])
            and isinstance(track_c, Mapping)
            and track_c.get("required_evaluation_budgets") == [1, 3, 5]
            and set(track_c.get("evidence_qualifiers", []))
            == {"predicted", "computational", "proxy-supported"}
            and "measured_improvement" in track_c.get("forbidden_claims", [])
        )
    expected_track_claims = {
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
    }
    expected_allowed = [
        "B0 structural benchmark, split, and track roles are frozen",
        (
            "Track B has no efficacy conclusion at B0; future formal "
            "generative evaluation under later contract gates is allowed"
        ),
    ]
    expected_unsupported = [
        "No final efficacy, SOTA, or full legal action-space result",
        "No foundation-model exposure clearance before FM0 re-audit",
        "No measured improvement claim for open-world Track C",
    ]
    valid = valid and (
        track_claims == expected_track_claims
        and allowed == expected_allowed
        and unsupported == expected_unsupported
    )
    return {
        "schema_valid": bool(valid),
        "allowed_claims_present": isinstance(allowed, list) and bool(allowed),
        "unsupported_capabilities_present": (
            isinstance(unsupported, list) and bool(unsupported)
        ),
        "foundation_status": claims.get("foundation_status"),
        "allowed_claim": claims.get("allowed_claim"),
        "requires_fm0_reaudit": claims.get("requires_fm0_reaudit"),
        "gse246381_role": claims.get("gse246381_role"),
    }


def _parse_data_card_front_matter(text: str) -> Mapping[str, Any] | None:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    if not match:
        return None
    value = yaml.safe_load(match.group(1))
    return value if isinstance(value, Mapping) else None


def render_canonical_data_card(
    *,
    universe_binding: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    track_audit: Mapping[str, Any],
    role_policy: Mapping[str, Any],
) -> str:
    """Independently rebuild the one accepted Data Card byte sequence."""

    data_card_counts = track_audit.get("data_card_counts")
    if not isinstance(data_card_counts, Mapping):
        raise ValueError("track audit lacks recomputed Data Card counts")
    region_record_counts = data_card_counts.get("region_records")
    dataset_record_counts = data_card_counts.get("dataset_records")
    track_task_counts = data_card_counts.get("track_tasks")
    track_evidence_counts = data_card_counts.get("track_evidence")
    if not all(
        isinstance(value, Mapping)
        for value in (
            region_record_counts,
            dataset_record_counts,
            track_task_counts,
            track_evidence_counts,
        )
    ):
        raise ValueError("track audit Data Card counts are incomplete")
    if dict(track_evidence_counts) != role_policy.get("track_evidence_counts"):
        raise ValueError("track audit and frozen role policy evidence counts differ")

    region_counts = {
        region: int(region_record_counts.get(region, 0))
        for region in ("five_utr", "three_utr")
    }
    if set(region_record_counts) != set(region_counts):
        raise ValueError("Data Card region count scope is not exact")
    dataset_counts = {
        str(dataset_id): int(count)
        for dataset_id, count in sorted(dataset_record_counts.items())
    }
    track_counts = {
        str(track_type): int(count)
        for track_type, count in sorted(track_task_counts.items())
    }
    record_count = universe_binding.get("record_count")
    task_count = universe_binding.get("task_count")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or sum(region_counts.values()) != record_count
        or sum(dataset_counts.values()) != record_count
        or sum(track_counts.values()) != task_count
    ):
        raise ValueError("Data Card counts differ from the frozen universe")

    front = {
        "schema_version": "utr_editbench_data_card.v2",
        "canonical_records_sha256": universe_binding["canonical_records_sha256"],
        "structural_records_sha256": universe_binding["structural_records_sha256"],
        "record_count": record_count,
        "candidate_count": universe_binding["candidate_count"],
        "task_count": task_count,
        "source_count": universe_binding["source_count"],
        "exposure_ledger_sha256": artifact_hashes["exposure_ledger"],
        "track_role_matrix_sha256": artifact_hashes["track_role_matrix"],
        "claims_sha256": artifact_hashes["claims"],
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
                "gse246381_role": _RETROSPECTIVE_ROLE,
            },
            "claims": {
                "b0_efficacy_conclusion_allowed": False,
                "future_track_b_formal_generative_evaluation_allowed": True,
                "track_a_process_order_attestation_only": True,
                "track_c_measured_improvement_allowed": False,
            },
        },
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
        f"- Eligible intervention records/tasks: {record_count}",
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


def _validate_data_card(
    path: Path,
    *,
    universe_binding: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    track_audit: Mapping[str, Any],
    role_policy: Mapping[str, Any],
) -> bool:
    text = path.read_text(encoding="utf-8")
    front = _parse_data_card_front_matter(text)
    data_card_counts = track_audit.get("data_card_counts")
    if not isinstance(data_card_counts, Mapping):
        return False
    expected_front = {
        "schema_version": "utr_editbench_data_card.v2",
        "canonical_records_sha256": universe_binding["canonical_records_sha256"],
        "structural_records_sha256": universe_binding["structural_records_sha256"],
        "record_count": universe_binding["record_count"],
        "candidate_count": universe_binding["candidate_count"],
        "task_count": universe_binding["task_count"],
        "source_count": universe_binding["source_count"],
        "exposure_ledger_sha256": artifact_hashes["exposure_ledger"],
        "track_role_matrix_sha256": artifact_hashes["track_role_matrix"],
        "claims_sha256": artifact_hashes["claims"],
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
                "gse246381_role": _RETROSPECTIVE_ROLE,
            },
            "claims": {
                "b0_efficacy_conclusion_allowed": False,
                "future_track_b_formal_generative_evaluation_allowed": True,
                "track_a_process_order_attestation_only": True,
                "track_c_measured_improvement_allowed": False,
            },
        },
    }
    if dict(front or {}) != expected_front:
        return False
    try:
        canonical_text = render_canonical_data_card(
            universe_binding=universe_binding,
            artifact_hashes=artifact_hashes,
            track_audit=track_audit,
            role_policy=role_policy,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return text == canonical_text


def validate_required_artifacts(
    binding_manifest_path: Path,
    *,
    track_audit: Mapping[str, Any],
    exposure_ledger_path: Path,
    role_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Read and cryptographically bind every non-split B0 acceptance artifact."""

    binding_path = binding_manifest_path.resolve()
    failures: List[str] = []
    artifacts_audit: Dict[str, Dict[str, Any]] = {}
    try:
        binding = _load_mapping(binding_path)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return {
            "schema_version": "utr_b0_required_artifact_audit.v2",
            "gate_passed": False,
            "failures": [f"binding_manifest_invalid:{type(exc).__name__}"],
            "universe_binding": None,
            "artifacts": {},
            "claims": {},
        }
    if set(binding) != {"schema_version", "universe_binding", "artifacts"}:
        failures.append("binding_manifest_schema")
    if binding.get("schema_version") != "utr_b0_artifact_bindings.v2":
        failures.append("binding_manifest_schema_version")
    universe_binding = binding.get("universe_binding")
    if not isinstance(universe_binding, Mapping) or universe_binding != track_audit.get(
        "universe_binding"
    ):
        failures.append("binding_manifest_universe")
        universe_binding = {}
    references = binding.get("artifacts")
    if not isinstance(references, Mapping) or set(references) != set(_BOUND_ARTIFACTS):
        failures.append("binding_manifest_artifact_set")
        references = {}

    resolved: Dict[str, Path] = {}
    for name, expected_schema in _BOUND_ARTIFACTS.items():
        reference = references.get(name)
        item: Dict[str, Any] = {
            "exists": False,
            "bytes": None,
            "sha256": None,
            "schema_valid": False,
        }
        artifacts_audit[name] = item
        if not isinstance(reference, Mapping) or set(reference) != _BINDING_REF_FIELDS:
            failures.append(f"{name}:reference_schema")
            continue
        if reference.get("schema_version") != expected_schema:
            failures.append(f"{name}:declared_schema")
            continue
        if not (
            _is_sha256(reference.get("sha256"))
            and _nonnegative_int(reference.get("bytes"))
        ):
            failures.append(f"{name}:declared_hash_or_bytes")
            continue
        try:
            path = _resolve_bound_file(
                binding_path.parent,
                reference.get("path"),
                f"artifacts.{name}.path",
            )
        except ValueError:
            failures.append(f"{name}:missing")
            continue
        actual_sha = sha256_file(path)
        actual_bytes = path.stat().st_size
        item.update(
            {
                "path": str(path),
                "exists": True,
                "bytes": actual_bytes,
                "sha256": actual_sha,
            }
        )
        if actual_sha != reference["sha256"] or actual_bytes != reference["bytes"]:
            failures.append(f"{name}:hash_or_bytes_mismatch")
            continue
        resolved[name] = path

    if (
        "exposure_ledger" in resolved
        and resolved["exposure_ledger"] != exposure_ledger_path.resolve()
    ):
        failures.append("exposure_ledger:path_differs_from_acceptance_input")
    if "exposure_ledger" in resolved:
        valid = _validate_exposure_ledger_shape(resolved["exposure_ledger"])
        artifacts_audit["exposure_ledger"]["schema_valid"] = valid
        if not valid:
            failures.append("exposure_ledger:schema")

    if "track_role_matrix" in resolved:
        try:
            matrix = _load_mapping(resolved["track_role_matrix"])
            valid = matrix == dict(track_audit)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
            valid = False
        artifacts_audit["track_role_matrix"]["schema_valid"] = valid
        if not valid:
            failures.append("track_role_matrix:schema_or_content")

    claims_audit: Dict[str, Any] = {}
    if "claims" in resolved:
        try:
            claims = _load_mapping(resolved["claims"])
            claims_audit = _validate_claims(claims, universe_binding)
            valid = claims_audit["schema_valid"]
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
            valid = False
        artifacts_audit["claims"]["schema_valid"] = valid
        if not valid:
            failures.append("claims:schema_or_content")

    if "data_card" in resolved and all(
        name in resolved for name in ("exposure_ledger", "track_role_matrix", "claims")
    ):
        hashes = {
            name: artifacts_audit[name]["sha256"]
            for name in ("exposure_ledger", "track_role_matrix", "claims")
        }
        valid = _validate_data_card(
            resolved["data_card"],
            universe_binding=universe_binding,
            artifact_hashes=hashes,
            track_audit=track_audit,
            role_policy=role_policy,
        )
        artifacts_audit["data_card"]["schema_valid"] = valid
        if not valid:
            failures.append("data_card:schema_or_content")
    elif "data_card" in resolved:
        failures.append("data_card:dependencies_missing")

    return {
        "schema_version": "utr_b0_required_artifact_audit.v2",
        "binding_manifest_path": str(binding_path),
        "binding_manifest_sha256": sha256_file(binding_path),
        "gate_passed": not failures
        and all(
            item.get("exists") is True
            and item.get("schema_valid") is True
            and _is_sha256(item.get("sha256"))
            and _nonnegative_int(item.get("bytes"))
            and item.get("bytes", 0) > 0
            for item in artifacts_audit.values()
        ),
        "failures": sorted(set(failures)),
        "universe_binding": dict(universe_binding),
        "artifacts": artifacts_audit,
        "claims": claims_audit,
    }


def validate_b0_acceptance(
    *,
    leakage_reports: Sequence[Mapping[str, Any]],
    exposure_audit: Mapping[str, Any],
    track_audit: Mapping[str, Any],
    split_manifests: Sequence[Mapping[str, Any]],
    track_a_label_seal_audit: Mapping[str, Any] | None = None,
    required_artifact_audit: Mapping[str, Any] | None = None,
    d1_exposure_ledger_binding: Mapping[str, Any] | None = None,
    supplied_leakage_reports_match_recomputation: bool | None = None,
) -> Dict[str, Any]:
    """Combine the contract's exact B0 gates without proxy substitutions."""

    failed: List[str] = []
    if not leakage_reports:
        failed.append("leakage_reports_present")
    if supplied_leakage_reports_match_recomputation is not True:
        failed.append("supplied_leakage_reports_exactly_recomputed")
    for gate_name, count_name in ZERO_GATE_COUNTS.items():
        passed = bool(leakage_reports)
        for report in leakage_reports:
            counts = report.get("counts")
            gates = report.get("acceptance_gates")
            if not isinstance(counts, Mapping) or not isinstance(gates, Mapping):
                passed = False
                continue
            count = counts.get(count_name)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count != 0
                or gates.get(gate_name) is not True
            ):
                passed = False
        if not passed:
            failed.append(gate_name)

    strict_foundation, normalized_foundations = _strict_pending_fm0(leakage_reports)
    if not strict_foundation:
        failed.append("foundation_state_must_remain_unknown_pending_fm0")
    if (
        exposure_audit.get("coverage") != 1.0
        or exposure_audit.get("gate_passed") is not True
        or exposure_audit.get("identity_level") not in ("record_id", "dataset_id")
    ):
        failed.append("exposure_ledger_coverage_100_percent")
    if not (
        isinstance(d1_exposure_ledger_binding, Mapping)
        and d1_exposure_ledger_binding.get("schema_version")
        == "utr_b0_d1_exposure_binding.v2"
        and d1_exposure_ledger_binding.get("gate_passed") is True
        and d1_exposure_ledger_binding.get("ledger_semantics_valid") is True
    ):
        failed.append("d1_exposure_ledger_exact_binding")
    if (
        track_audit.get("track_role_ambiguity_count") != 0
        or track_audit.get("gate_passed") is not True
    ):
        failed.append("track_role_ambiguity_zero")
    if track_audit.get("identity_universe_complete") is not True:
        failed.append("track_identity_universe_complete")
    if (
        track_audit.get("eligible_identity_binding_checked") is not True
        or track_audit.get("eligible_identity_binding_complete") is not True
    ):
        failed.append("track_eligible_identity_universe_complete")
    if (
        track_audit.get("task_structural_binding_checked") is not True
        or track_audit.get("task_structural_binding_complete") is not True
    ):
        failed.append("track_tasks_exactly_bound_to_structural_records")
    track_rows = track_audit.get("tracks")
    observed_track_types = (
        {str(row.get("track_type")) for row in track_rows}
        if isinstance(track_rows, list)
        and all(isinstance(row, Mapping) for row in track_rows)
        else set()
    )
    if track_audit.get("track_count") != 3 or observed_track_types != {
        "closed_measured_pool",
        "heldout_generative",
        "open_legal_generation",
    }:
        failed.append("exact_three_tracks_present")
    track_c_rows = (
        [
            row
            for row in track_rows
            if isinstance(row, Mapping)
            and row.get("track_type") == "open_legal_generation"
        ]
        if isinstance(track_rows, list)
        else []
    )
    expected_budget_protocol = {
        "required_budgets": [1, 3, 5],
        "task_representation": "single_maximum_budget",
        "maximum_budget": 5,
        "report_each_budget_separately": True,
        "silent_budget_reduction_forbidden": True,
    }
    if (
        len(track_c_rows) != 1
        or track_c_rows[0].get("evaluation_budget_protocol") != expected_budget_protocol
    ):
        failed.append("track_c_budget_protocol_1_3_5")
    if track_audit.get("gse246381_role", _RETROSPECTIVE_ROLE) != _RETROSPECTIVE_ROLE:
        failed.append("gse246381_retrospective_only")
    if isinstance(track_rows, list):
        for row in track_rows:
            if (
                isinstance(row, Mapping)
                and row.get("track_type")
                in ("heldout_generative", "open_legal_generation")
                and row.get("retrospective_external_stress_datasets")
            ):
                failed.append("gse246381_retrospective_only")

    manifest_by_identity: Dict[Tuple[str, str, Any], Mapping[str, Any]] = {}
    duplicate_manifest_identities = set()
    for manifest in split_manifests:
        identity = _split_identity(manifest)
        if identity in manifest_by_identity:
            duplicate_manifest_identities.add(identity)
        manifest_by_identity[identity] = manifest
    if (
        set(manifest_by_identity) != EXPECTED_SPLIT_IDENTITIES
        or duplicate_manifest_identities
    ):
        failed.append("exact_five_split_manifests_present")
    if not split_manifests or any(
        manifest.get("status") != "READY" for manifest in split_manifests
    ):
        failed.append("all_split_manifests_ready")
    split_universe = _common_split_universe(split_manifests)
    if split_universe is None:
        failed.append("five_splits_share_one_frozen_universe")
    (
        role_universe,
        role_record_ids,
        full_record_ids,
        role_universe_failures,
    ) = _derive_role_record_universe(manifest_by_identity, split_universe)
    if role_universe_failures:
        failed.append("eligible_and_excluded_record_accounting")
    if not _track_split_universe_matches(track_audit, split_universe, role_universe):
        failed.append("track_split_universe_binding")

    report_by_identity: Dict[Tuple[str, str, Any], Mapping[str, Any]] = {}
    duplicate_report_identities = set()
    for report in leakage_reports:
        identity = _split_identity(report)
        if identity in report_by_identity:
            duplicate_report_identities.add(identity)
        report_by_identity[identity] = report
    if (
        set(report_by_identity) != EXPECTED_SPLIT_IDENTITIES
        or duplicate_report_identities
    ):
        failed.append("one_leakage_report_per_required_split")
    for identity in EXPECTED_SPLIT_IDENTITIES:
        manifest = manifest_by_identity.get(identity)
        report = report_by_identity.get(identity)
        if manifest is None or report is None:
            continue
        manifest_sha = manifest.get("_artifact_sha256")
        if (
            not _is_sha256(manifest_sha)
            or report.get("split_manifest_sha256") != manifest_sha
            or any(
                report.get(field) != manifest.get(field)
                for field in _SPLIT_UNIVERSE_FIELDS
            )
        ):
            failed.append("leakage_report_manifest_binding")
    failed.extend(
        _partition_evidence_failures(
            manifest_by_identity,
            report_by_identity,
            role_record_ids=role_record_ids,
            full_record_ids=full_record_ids,
        )
    )

    label_seal = track_a_label_seal_audit or {}
    if not _track_a_seal_matches(
        label_seal,
        track_audit,
        split_universe,
        role_universe,
        d1_exposure_ledger_binding,
    ):
        failed.append("track_a_privileged_label_seal")
    required = required_artifact_audit or {}
    required_artifacts_valid = (
        required.get("schema_version") == "utr_b0_required_artifact_audit.v2"
        and required.get("gate_passed") is True
        and required.get("universe_binding") == track_audit.get("universe_binding")
        and isinstance(required.get("artifacts"), Mapping)
        and set(required["artifacts"]) == set(_BOUND_ARTIFACTS)
        and all(
            isinstance(item, Mapping)
            and item.get("exists") is True
            and item.get("schema_valid") is True
            and _is_sha256(item.get("sha256"))
            and _nonnegative_int(item.get("bytes"))
            and item.get("bytes", 0) > 0
            for item in required["artifacts"].values()
        )
    )
    claims = required.get("claims")
    if not (
        required_artifacts_valid
        and isinstance(claims, Mapping)
        and claims.get("allowed_claims_present") is True
        and claims.get("unsupported_capabilities_present") is True
        and claims.get("foundation_status") == "UNKNOWN_PENDING_FM0"
        and claims.get("allowed_claim") == "NONE"
        and claims.get("requires_fm0_reaudit") is True
        and claims.get("gse246381_role") == _RETROSPECTIVE_ROLE
    ):
        failed.append("required_artifacts_bound_and_valid")

    failed = sorted(set(failed))
    return {
        "schema_version": "utr_b0_acceptance.v2",
        "b0_gate_passed": len(failed) == 0,
        "failed_gates": failed,
        "observed": {
            "leakage_report_count": len(leakage_reports),
            "exposure_ledger_coverage": exposure_audit.get("coverage"),
            "exposure_identity_level": exposure_audit.get("identity_level"),
            "track_role_ambiguity_count": track_audit.get("track_role_ambiguity_count"),
            "track_identity_universe_complete": track_audit.get(
                "identity_universe_complete"
            ),
            "track_eligible_identity_universe_complete": track_audit.get(
                "eligible_identity_binding_complete"
            ),
            "split_identities": [
                list(identity) for identity in sorted(manifest_by_identity)
            ],
            "split_universe": split_universe,
            "eligible_track_role_universe": role_universe,
            "eligible_track_role_universe_failures": sorted(
                set(role_universe_failures)
            ),
            "foundation_states": normalized_foundations,
        },
        "allowed_claim": "NONE",
        "requires_fm0_reaudit": True,
        "re_audit_required_before_foundation_use": True,
        "claim_boundary": (
            "B0 structural split acceptance is not an efficacy or SOTA result"
        ),
    }


def _read_jsonl_objects(path: Path) -> List[Dict[str, Any]]:
    return load_jsonl(path)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--d1-acceptance", type=Path, required=True)
    parser.add_argument("--exposure-ledger", type=Path, required=True)
    parser.add_argument(
        "--exposure-identity-level",
        choices=("record_id", "dataset_id"),
        required=True,
    )
    parser.add_argument("--split-manifest", type=Path, action="append", required=True)
    parser.add_argument("--leakage-report", type=Path, action="append", required=True)
    parser.add_argument("--track-manifest", type=Path, action="append", required=True)
    parser.add_argument("--artifact-bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    records = load_jsonl(args.records)
    exposure = compute_exposure_coverage(
        records,
        _read_jsonl_objects(args.exposure_ledger),
        identity_level=args.exposure_identity_level,
        required_ledger_identities=(
            sorted(D1_SCOPE_DATASETS)
            if args.exposure_identity_level == "dataset_id"
            else None
        ),
    )
    split_manifests = []
    split_path_by_identity: Dict[Tuple[str, str, Any], Path] = {}
    for path in args.split_manifest:
        manifest = load_json(path)
        manifest["_artifact_sha256"] = sha256_file(path)
        identity = _split_identity(manifest)
        if identity in split_path_by_identity:
            raise ValueError(f"duplicate split manifest identity: {identity}")
        split_path_by_identity[identity] = path.resolve()
        split_manifests.append(manifest)
    supplied_by_identity: Dict[Tuple[str, str, Any], Mapping[str, Any]] = {}
    for path in args.leakage_report:
        supplied = load_json(path)
        identity = _split_identity(supplied)
        if identity in supplied_by_identity:
            raise ValueError(f"duplicate leakage report identity: {identity}")
        supplied_by_identity[identity] = supplied
    if (
        set(split_path_by_identity) != EXPECTED_SPLIT_IDENTITIES
        or set(supplied_by_identity) != EXPECTED_SPLIT_IDENTITIES
    ):
        raise ValueError(
            "acceptance requires exactly the five frozen split/report identities"
        )
    recompute_cache = build_bound_structural_recompute_cache(args.records)
    leakage_reports = [
        recompute_bound_leakage_report(
            args.records,
            split_path_by_identity[identity],
            expected_d1_acceptance_path=args.d1_acceptance,
            _cache=recompute_cache,
        )
        for identity in sorted(EXPECTED_SPLIT_IDENTITIES)
    ]
    recomputed_by_identity = {
        _split_identity(report): report for report in leakage_reports
    }
    reports_match = all(
        supplied_by_identity[identity] == recomputed_by_identity[identity]
        for identity in EXPECTED_SPLIT_IDENTITIES
    )
    exposure_binding = validate_d1_exposure_ledger_binding(
        args.d1_acceptance,
        args.exposure_ledger,
    )
    tracks = [load_track_manifest(path) for path in args.track_manifest]
    track_a_paths = [
        Path(track.manifest_path)
        for track in tracks
        if track.track_type == "closed_measured_pool"
    ]
    if len(track_a_paths) != 1:
        raise ValueError("acceptance requires exactly one Track A manifest")
    manifest_by_identity = {
        _split_identity(manifest): manifest for manifest in split_manifests
    }
    split_universe = _common_split_universe(split_manifests)
    _, eligible_record_ids, _, _ = _derive_role_record_universe(
        manifest_by_identity, split_universe
    )
    eligible_records = [
        record
        for record in records
        if str(record.get("record_id") or "") in eligible_record_ids
    ]
    from scripts.data.build_b0_evaluation_artifacts import (
        _derive_fixed_track_roles,
    )

    role_by_record, role_policy = _derive_fixed_track_roles(
        split_manifests,
        eligible_record_ids,
        eligible_records,
    )
    label_seal = privileged_verify_track_a_label_seal(
        track_a_paths[0],
        expected_role_policy=role_policy,
        expected_d1_acceptance_path=args.d1_acceptance,
    )
    track_audit = audit_track_roles(
        tracks,
        eligible_records=eligible_records,
        expected_role_by_record=role_by_record,
    )
    required_artifacts = validate_required_artifacts(
        args.artifact_bindings,
        track_audit=track_audit,
        exposure_ledger_path=args.exposure_ledger,
        role_policy=role_policy,
    )
    result = validate_b0_acceptance(
        leakage_reports=leakage_reports,
        exposure_audit=exposure,
        track_audit=track_audit,
        split_manifests=split_manifests,
        track_a_label_seal_audit=label_seal,
        required_artifact_audit=required_artifacts,
        d1_exposure_ledger_binding=exposure_binding,
        supplied_leakage_reports_match_recomputation=reports_match,
    )
    result["exposure_ledger"] = exposure
    result["track_role_audit"] = track_audit
    result["track_a_label_seal_audit"] = label_seal
    result["required_artifact_audit"] = required_artifacts
    result["d1_exposure_ledger_binding"] = exposure_binding
    result["supplied_leakage_reports_match_recomputation"] = reports_match
    result["recomputed_leakage_reports"] = leakage_reports
    result["supplied_leakage_report_files"] = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in args.leakage_report
    ]
    write_json_exclusive(args.output, result)
    return 0 if result["b0_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
