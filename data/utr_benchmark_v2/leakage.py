"""Fail-closed structural leakage audit for UTR benchmark v2 partitions."""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .near_neighbors import NearNeighborClusters
from .split_graph import METADATA_DIMENSIONS
from .split_graph import ROLE_NAMES
from .split_graph import ROLE_PAIRS
from .split_graph import SplitGraphError
from .split_graph import build_atomic_components
from .split_graph import canonical_sequence
from .split_graph import expected_partition_leakage_contract
from .split_graph import expected_partition_ids
from .split_graph import global_near_neighbor_clusters
from .split_graph import intermediate_sequences
from .split_graph import metadata_values
from .split_graph import partition_sha256
from .split_graph import record_ids_sha256
from .split_graph import record_structural_sha256
from .split_graph import record_universe_sha256
from .split_graph import state_sequences


class LeakageAuditError(ValueError):
    """The split or record universe cannot be audited safely."""


def _stable_sha256(value: Any) -> str:
    import json

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _record_id(record: Mapping[str, Any]) -> str:
    value = str(record.get("record_id") or "").strip()
    if not value:
        raise LeakageAuditError("record_id must be non-empty")
    return value


def _role_pairs() -> Tuple[Tuple[str, str], ...]:
    return tuple(combinations(ROLE_NAMES, 2))


def _accounting_index(
    rows: Any,
    *,
    kind: str,
    universe_issues: List[Dict[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        universe_issues.append({"kind": f"{kind}_accounting_missing_or_invalid"})
        return {}
    index: Dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            universe_issues.append(
                {
                    "kind": f"{kind}_accounting_row_invalid",
                    "position": position,
                }
            )
            continue
        record_id = str(row.get("record_id") or "")
        if not record_id:
            universe_issues.append(
                {
                    "kind": f"{kind}_accounting_record_id_missing",
                    "position": position,
                }
            )
            continue
        if record_id in index:
            universe_issues.append(
                {
                    "kind": f"{kind}_accounting_duplicate_record_id",
                    "record_id": record_id,
                }
            )
        index[record_id] = row
    return index


def _binding_mismatch(
    issues: List[Dict[str, Any]],
    *,
    field: str,
    expected: Any,
    observed: Any,
) -> None:
    if observed != expected:
        issues.append(
            {
                "kind": "frozen_universe_binding_mismatch",
                "field": field,
                "expected": expected,
                "observed": observed,
            }
        )


def _records_by_role(
    records: Sequence[Mapping[str, Any]],
    split_partition: Mapping[str, Any],
    near_neighbors: NearNeighborClusters,
) -> Tuple[
    Dict[str, List[Mapping[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Resolve one partition while revalidating its exhaustive accounting."""

    if split_partition.get("status") != "READY":
        raise LeakageAuditError("cannot audit a split partition that is not READY")
    universe_issues: List[Dict[str, Any]] = []
    full_index: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        record_id = _record_id(record)
        if record_id in full_index:
            raise LeakageAuditError(f"duplicate canonical record_id: {record_id}")
        full_index[record_id] = record

    _binding_mismatch(
        universe_issues,
        field="full_record_count",
        expected=len(records),
        observed=split_partition.get("full_record_count"),
    )
    _binding_mismatch(
        universe_issues,
        field="full_record_ids_sha256",
        expected=record_ids_sha256(records),
        observed=split_partition.get("full_record_ids_sha256"),
    )
    _binding_mismatch(
        universe_issues,
        field="full_record_universe_sha256",
        expected=record_universe_sha256(records),
        observed=split_partition.get("full_record_universe_sha256"),
    )
    observed_partition_hash = split_partition.get("partition_sha256")
    expected_partition_hash = partition_sha256(split_partition)
    _binding_mismatch(
        universe_issues,
        field="partition_sha256",
        expected=expected_partition_hash,
        observed=observed_partition_hash,
    )
    _binding_mismatch(
        universe_issues,
        field="near_neighbor_binding",
        expected=dict(near_neighbors.binding),
        observed=split_partition.get("near_neighbor_binding"),
    )

    eligible_rows = _accounting_index(
        split_partition.get("eligible_records"),
        kind="eligible",
        universe_issues=universe_issues,
    )
    excluded_rows = _accounting_index(
        split_partition.get("excluded_records"),
        kind="excluded",
        universe_issues=universe_issues,
    )
    eligible_ids = set(eligible_rows)
    excluded_ids = set(excluded_rows)
    duplicate_accounting = sorted(eligible_ids & excluded_ids)
    if duplicate_accounting:
        universe_issues.append(
            {
                "kind": "record_in_eligible_and_excluded_accounting",
                "count": len(duplicate_accounting),
                "examples": duplicate_accounting[:20],
            }
        )
    missing_accounting = sorted(set(full_index) - eligible_ids - excluded_ids)
    extra_accounting = sorted((eligible_ids | excluded_ids) - set(full_index))
    if missing_accounting:
        universe_issues.append(
            {
                "kind": "full_records_missing_from_accounting",
                "count": len(missing_accounting),
                "examples": missing_accounting[:20],
            }
        )
    if extra_accounting:
        universe_issues.append(
            {
                "kind": "accounting_records_outside_full_universe",
                "count": len(extra_accounting),
                "examples": extra_accounting[:20],
            }
        )

    for kind, rows in (
        ("eligible", eligible_rows),
        ("excluded", excluded_rows),
    ):
        for record_id, row in rows.items():
            record = full_index.get(record_id)
            if record is None:
                continue
            expected_sha = record_structural_sha256(record)
            if row.get("structural_sha256") != expected_sha:
                universe_issues.append(
                    {
                        "kind": f"{kind}_record_structural_hash_mismatch",
                        "record_id": record_id,
                        "expected": expected_sha,
                        "observed": row.get("structural_sha256"),
                    }
                )
            reason = row.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                universe_issues.append(
                    {
                        "kind": f"{kind}_record_reason_missing",
                        "record_id": record_id,
                    }
                )
            if kind == "eligible" and reason != "eligible_intervention_record":
                universe_issues.append(
                    {
                        "kind": "eligible_record_reason_invalid",
                        "record_id": record_id,
                        "observed": reason,
                    }
                )

    eligible = [
        full_index[record_id] for record_id in sorted(eligible_ids & set(full_index))
    ]
    excluded_list = [
        dict(excluded_rows[record_id]) for record_id in sorted(excluded_rows)
    ]
    eligible_list = [
        dict(eligible_rows[record_id]) for record_id in sorted(eligible_rows)
    ]
    excluded_id_rows = [{"record_id": record_id} for record_id in sorted(excluded_rows)]
    reason_counts: Dict[str, int] = {}
    for row in excluded_list:
        reason = str(row.get("reason") or "")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    bindings = {
        "record_count": len(eligible),
        "record_ids_sha256": record_ids_sha256(eligible),
        "record_universe_sha256": record_universe_sha256(eligible),
        "eligible_record_accounting_sha256": _stable_sha256(eligible_list),
        "excluded_record_count": len(excluded_rows),
        "excluded_record_ids_sha256": record_ids_sha256(excluded_id_rows),
        "excluded_record_accounting_sha256": _stable_sha256(excluded_list),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "component_count": len(build_atomic_components(eligible)),
    }
    for field, expected in bindings.items():
        _binding_mismatch(
            universe_issues,
            field=field,
            expected=expected,
            observed=split_partition.get(field),
        )

    roles = split_partition.get("roles")
    if not isinstance(roles, Mapping):
        raise LeakageAuditError("split partition roles must be an object")
    extra_role_names = sorted(set(roles) - set(ROLE_NAMES))
    if extra_role_names:
        universe_issues.append(
            {"kind": "unknown_role_names", "roles": extra_role_names}
        )
    by_role: Dict[str, List[Mapping[str, Any]]] = {}
    occurrences: Dict[str, List[str]] = {}
    for role in ROLE_NAMES:
        ids = roles.get(role)
        if not isinstance(ids, list):
            raise LeakageAuditError(f"split role {role} must be a list")
        normalized_ids = [str(record_id) for record_id in ids]
        if len(normalized_ids) != len(set(normalized_ids)):
            universe_issues.append({"kind": "duplicate_id_within_role", "role": role})
        if not normalized_ids:
            universe_issues.append({"kind": "empty_required_role", "role": role})
        selected: List[Mapping[str, Any]] = []
        for record_id in normalized_ids:
            occurrences.setdefault(record_id, []).append(role)
            if record_id not in full_index:
                universe_issues.append(
                    {
                        "kind": "unknown_record_id",
                        "role": role,
                        "record_id": record_id,
                    }
                )
            elif record_id not in eligible_rows:
                universe_issues.append(
                    {
                        "kind": "ineligible_record_assigned",
                        "role": role,
                        "record_id": record_id,
                    }
                )
            else:
                selected.append(full_index[record_id])
        by_role[role] = selected

    assigned_ids = set(occurrences)
    missing_ids = sorted(eligible_ids - assigned_ids)
    extra_ids = sorted(assigned_ids - eligible_ids)
    if missing_ids:
        universe_issues.append(
            {
                "kind": "eligible_records_missing_from_roles",
                "count": len(missing_ids),
                "examples": missing_ids[:20],
            }
        )
    if extra_ids:
        universe_issues.append(
            {
                "kind": "role_records_outside_eligible_universe",
                "count": len(extra_ids),
                "examples": extra_ids[:20],
            }
        )
    duplicate_role_assignments = [
        {"record_id": record_id, "roles": sorted(set(role_names))}
        for record_id, role_names in sorted(occurrences.items())
        if len(set(role_names)) > 1
    ]

    actual_component_roles: Dict[str, str] = {}
    component_overlaps: List[Dict[str, Any]] = []
    components = build_atomic_components(eligible)
    for component in components:
        roles_for_component = {
            role
            for record_id in component.record_ids
            for role in occurrences.get(record_id, ())
        }
        if len(roles_for_component) > 1:
            component_overlaps.append(
                {
                    "component_id": component.component_id,
                    "roles": sorted(roles_for_component),
                }
            )
        elif len(roles_for_component) == 1:
            actual_component_roles[component.component_id] = next(
                iter(roles_for_component)
            )
    manifested_components = split_partition.get("component_roles")
    if not isinstance(manifested_components, Mapping):
        universe_issues.append({"kind": "component_roles_missing_or_invalid"})
    else:
        normalized_components = {
            str(component_id): str(role)
            for component_id, role in manifested_components.items()
        }
        if normalized_components != actual_component_roles:
            universe_issues.append(
                {
                    "kind": "component_roles_binding_mismatch",
                    "expected_count": len(actual_component_roles),
                    "observed_count": len(normalized_components),
                }
            )

    role_bindings = split_partition.get("role_bindings")
    if not isinstance(role_bindings, Mapping):
        universe_issues.append({"kind": "role_bindings_missing_or_invalid"})
    else:
        for role in ROLE_NAMES:
            role_records = by_role[role]
            expected = {
                "record_count": len(role_records),
                "record_ids_sha256": record_ids_sha256(role_records),
                "record_universe_sha256": record_universe_sha256(role_records),
            }
            if role_bindings.get(role) != expected:
                universe_issues.append(
                    {
                        "kind": "role_binding_mismatch",
                        "role": role,
                        "expected": expected,
                        "observed": role_bindings.get(role),
                    }
                )
    return (
        by_role,
        duplicate_role_assignments,
        universe_issues,
        component_overlaps,
    )


def _sequence_sets(records: Iterable[Mapping[str, Any]], field: str) -> Set[str]:
    if field == "source":
        return {
            canonical_sequence(record.get("source_sequence"), "source_sequence")
            for record in records
        }
    if field == "candidate":
        return {
            canonical_sequence(record.get("candidate_sequence"), "candidate_sequence")
            for record in records
        }
    if field == "intermediate":
        return {
            sequence
            for record in records
            for sequence in intermediate_sequences(record)
        }
    if field == "path":
        return {sequence for record in records for sequence in state_sequences(record)}
    raise LeakageAuditError(f"unknown sequence-set field: {field}")


def _cross_role_shared(
    by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    field: str,
) -> Set[Tuple[str, str, str]]:
    sets = {role: _sequence_sets(by_role[role], field) for role in ROLE_NAMES}
    shared: Set[Tuple[str, str, str]] = set()
    for left, right in _role_pairs():
        for sequence in sets[left] & sets[right]:
            shared.add((left, right, sequence))
    return shared


def _reverse_edges(
    by_role: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Set[Tuple[str, str, str, str]]:
    edges: Dict[str, Set[Tuple[str, str]]] = {}
    for role in ROLE_NAMES:
        edges[role] = {
            (
                canonical_sequence(record.get("source_sequence"), "source_sequence"),
                canonical_sequence(
                    record.get("candidate_sequence"),
                    "candidate_sequence",
                ),
            )
            for record in by_role[role]
        }
    reverse: Set[Tuple[str, str, str, str]] = set()
    for left, right in _role_pairs():
        for source, candidate in edges[left]:
            if (candidate, source) in edges[right]:
                reverse.add(
                    (
                        left,
                        right,
                        _sequence_digest(source),
                        _sequence_digest(candidate),
                    )
                )
        for source, candidate in edges[right]:
            if (candidate, source) in edges[left]:
                reverse.add(
                    (
                        left,
                        right,
                        _sequence_digest(candidate),
                        _sequence_digest(source),
                    )
                )
    return reverse


def _metadata_overlaps(
    by_role: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Set[Tuple[str, str, str, str]]:
    per_role: Dict[str, Dict[str, Set[str]]] = {}
    for role in ROLE_NAMES:
        dimensions: Dict[str, Set[str]] = {
            field: set() for field in METADATA_DIMENSIONS
        }
        for record in by_role[role]:
            for field, values in metadata_values(record).items():
                dimensions[field].update(values)
        per_role[role] = dimensions
    overlaps: Set[Tuple[str, str, str, str]] = set()
    for left, right in _role_pairs():
        for field in METADATA_DIMENSIONS:
            for value in per_role[left][field] & per_role[right][field]:
                overlaps.add((left, right, field, value))
    return overlaps


def _normalize_policy_pair(value: Any) -> Optional[Tuple[str, str]]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(role, str) for role in value)
    ):
        return None
    pair = tuple(value)
    if pair not in ROLE_PAIRS:
        return None
    return pair


def _classify_metadata_overlaps(
    overlaps: Set[Tuple[str, str, str, str]],
    policy: Any,
) -> Tuple[
    Set[Tuple[str, str, str, str]],
    Set[Tuple[str, str, str, str]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    issues: List[Dict[str, Any]] = []
    normalized: Dict[str, Set[Tuple[str, str]]] = {}
    policy_payload = policy if isinstance(policy, Mapping) else {}
    if set(policy_payload) != set(METADATA_DIMENSIONS):
        issues.append(
            {
                "kind": "metadata_overlap_policy_dimension_mismatch",
                "expected": list(METADATA_DIMENSIONS),
                "observed": sorted(str(key) for key in policy_payload),
            }
        )
    for field in METADATA_DIMENSIONS:
        entry = policy_payload.get(field)
        allowed: Set[Tuple[str, str]] = set()
        if not isinstance(entry, Mapping):
            issues.append(
                {
                    "kind": "metadata_overlap_policy_entry_missing",
                    "field": field,
                }
            )
        else:
            pairs = entry.get("allowed_role_pairs")
            justification = entry.get("justification")
            if not isinstance(pairs, list):
                issues.append(
                    {
                        "kind": "metadata_overlap_allowed_pairs_invalid",
                        "field": field,
                    }
                )
            else:
                for value in pairs:
                    pair = _normalize_policy_pair(value)
                    if pair is None:
                        issues.append(
                            {
                                "kind": "metadata_overlap_role_pair_invalid",
                                "field": field,
                                "value": value,
                            }
                        )
                    else:
                        allowed.add(pair)
            if (
                entry.get("unlisted_role_pairs") != "FORBIDDEN"
                or not isinstance(justification, str)
                or not justification.strip()
            ):
                issues.append(
                    {
                        "kind": "metadata_overlap_policy_not_fail_closed",
                        "field": field,
                    }
                )
        normalized[field] = allowed

    explained = {
        overlap
        for overlap in overlaps
        if (overlap[0], overlap[1]) in normalized.get(overlap[2], set())
    }
    unexplained = overlaps - explained
    axis_status = {
        field: {
            "overlap_count": sum(overlap[2] == field for overlap in overlaps),
            "explained_overlap_count": sum(
                overlap[2] == field for overlap in explained
            ),
            "unexplained_overlap_count": sum(
                overlap[2] == field for overlap in unexplained
            ),
            "allowed_role_pairs": [list(pair) for pair in sorted(normalized[field])],
            "status": (
                "PASS"
                if not any(overlap[2] == field for overlap in unexplained)
                else "FAIL"
            ),
        }
        for field in METADATA_DIMENSIONS
    }
    return explained, unexplained, axis_status, issues


def _near_neighbor_overlaps(
    by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    near_neighbors: NearNeighborClusters,
) -> Set[Tuple[str, str, str]]:
    clusters_by_role: Dict[str, Set[str]] = {}
    for role in ROLE_NAMES:
        clusters: Set[str] = set()
        for record in by_role[role]:
            record_id = _record_id(record)
            cluster_id = near_neighbors.record_clusters.get(record_id)
            if cluster_id is None:
                raise LeakageAuditError(
                    f"near-neighbor binding lacks role record {record_id}"
                )
            clusters.add(cluster_id)
        clusters_by_role[role] = clusters
    return {
        (left, right, cluster_id)
        for left, right in _role_pairs()
        for cluster_id in (clusters_by_role[left] & clusters_by_role[right])
    }


def _required_axis_audit(
    split_partition: Mapping[str, Any],
    metadata_overlaps: Set[Tuple[str, str, str, str]],
    near_overlaps: Set[Tuple[str, str, str]],
) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
    """Gate raw overlap on every independently declared disjoint axis."""

    issues: List[Dict[str, Any]] = []
    raw_axes = split_partition.get("required_disjoint_axes")
    allowed_axes = set(METADATA_DIMENSIONS) | {"near_neighbor_cluster"}
    if (
        not isinstance(raw_axes, list)
        or not raw_axes
        or not all(isinstance(axis, str) and axis for axis in raw_axes)
        or len(raw_axes) != len(set(raw_axes))
    ):
        issues.append({"kind": "required_disjoint_axes_invalid"})
        axes: Tuple[str, ...] = ()
    else:
        axes = tuple(raw_axes)
        unknown = sorted(set(axes) - allowed_axes)
        if unknown:
            issues.append(
                {
                    "kind": "required_disjoint_axes_unknown",
                    "axes": unknown,
                }
            )

    raw_pairs = split_partition.get("required_disjoint_role_pairs")
    pair_payload = raw_pairs if isinstance(raw_pairs, Mapping) else {}
    if set(pair_payload) != set(axes):
        issues.append(
            {
                "kind": "required_disjoint_role_pair_axes_mismatch",
                "expected": sorted(axes),
                "observed": sorted(str(key) for key in pair_payload),
            }
        )

    normalized_pairs: Dict[str, Set[Tuple[str, str]]] = {}
    for axis in axes:
        values = pair_payload.get(axis)
        pairs: Set[Tuple[str, str]] = set()
        if not isinstance(values, list) or not values:
            issues.append(
                {
                    "kind": "required_disjoint_role_pairs_invalid",
                    "axis": axis,
                }
            )
        else:
            for value in values:
                pair = _normalize_policy_pair(value)
                if pair is None:
                    issues.append(
                        {
                            "kind": "required_disjoint_role_pair_invalid",
                            "axis": axis,
                            "value": value,
                        }
                    )
                else:
                    pairs.add(pair)
        normalized_pairs[axis] = pairs

    if "near_neighbor_cluster" in axes and normalized_pairs.get(
        "near_neighbor_cluster"
    ) != set(ROLE_PAIRS):
        issues.append(
            {
                "kind": "near_neighbor_disjointness_not_all_role_pairs",
            }
        )
    for firewall_axis in ("source_group", "scaffold_group"):
        if firewall_axis in axes and normalized_pairs.get(firewall_axis) != set(
            ROLE_PAIRS
        ):
            issues.append(
                {
                    "kind": "dense_graph_firewall_not_all_role_pairs",
                    "axis": firewall_axis,
                }
            )

    axis_status: Dict[str, Any] = {}
    total_overlap_count = 0
    for axis in axes:
        pairs = normalized_pairs.get(axis, set())
        if axis == "near_neighbor_cluster":
            overlap_count = sum(
                (left, right) in pairs for left, right, _ in near_overlaps
            )
        else:
            overlap_count = sum(
                field == axis and (left, right) in pairs
                for left, right, field, _ in metadata_overlaps
            )
        total_overlap_count += overlap_count
        axis_status[axis] = {
            "required_role_pairs": [list(pair) for pair in sorted(pairs)],
            "raw_overlap_count": overlap_count,
            "gate_passed": overlap_count == 0,
        }

    axis_partition_map = {
        "source_state": "near_neighbor_cluster",
        "sequence_cluster": "sequence_cluster",
        "scaffold_group": "scaffold_group",
        "gene_group": "gene_group",
        "context_group": "context_group",
        "barcode_batch": "barcode_batch",
        "library_batch": "library_batch",
    }
    if split_partition.get("axis_partition") is True:
        dimension = split_partition.get("independent_group_dimension")
        expected_axis = axis_partition_map.get(str(dimension))
        if expected_axis is None or expected_axis not in axes:
            issues.append(
                {
                    "kind": "axis_partition_required_axis_mismatch",
                    "independent_group_dimension": dimension,
                    "expected_axis": expected_axis,
                }
            )
    return axis_status, total_overlap_count, issues


def normalize_foundation_exposure(
    foundation_exposure: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Normalize FM overlap status without turning unknown into no-overlap."""

    payload = dict(foundation_exposure or {})
    if "foundation_selected" in payload:
        selected = bool(payload["foundation_selected"])
    else:
        selection_status = str(payload.get("selection_status") or "").upper()
        selected = bool(selection_status) and selection_status not in {
            "NO_FOUNDATION_SELECTED",
            "NONE",
        }
    raw_status = str(payload.get("status") or "UNKNOWN_PENDING_FM0").upper()
    evidence_fields = (
        "checkpoint_sha256",
        "corpus_manifest_sha256",
        "audit_report_sha256",
    )
    evidence_complete = selected and all(
        isinstance(payload.get(field), str)
        and len(str(payload[field])) == 64
        and all(
            character in "0123456789abcdef" for character in str(payload[field]).lower()
        )
        for field in evidence_fields
    )
    if raw_status in ("CLEARED", "CLEARED_NO_OVERLAP", "NO_OVERLAP"):
        if evidence_complete:
            status = "CLEARED_NO_OVERLAP"
            allowed_claim = "FOUNDATION_OVERLAP_AUDITED"
            re_audit = False
        else:
            status = "INVALID_CLEARANCE_EVIDENCE"
            allowed_claim = "NONE"
            re_audit = True
    elif raw_status in ("OVERLAP", "OVERLAP_FOUND"):
        status = "OVERLAP_FOUND"
        allowed_claim = "NONE"
        re_audit = True
    else:
        status = "UNKNOWN_PENDING_FM0"
        allowed_claim = "NONE"
        re_audit = True
    gate_passed = status == "CLEARED_NO_OVERLAP" or (
        status == "UNKNOWN_PENDING_FM0" and not selected
    )
    return {
        "status": status,
        "foundation_selected": selected,
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "corpus_manifest_sha256": payload.get("corpus_manifest_sha256"),
        "audit_report_sha256": payload.get("audit_report_sha256"),
        "clearance_evidence_complete": evidence_complete,
        "allowed_claim": allowed_claim,
        "re_audit_required": re_audit,
        "gate_applicable": selected,
        "gate_passed": gate_passed,
    }


def audit_cross_role_leakage(
    records: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    foundation_exposure: Optional[Mapping[str, Any]] = None,
    *,
    _near_neighbors: Optional[NearNeighborClusters] = None,
) -> Dict[str, Any]:
    """Audit one partition's exact, path, intermediate, and metadata leakage."""

    try:
        near_neighbors = (
            _near_neighbors
            if _near_neighbors is not None
            else global_near_neighbor_clusters(records)
        )
        (
            by_role,
            duplicate_assignments,
            universe_issues,
            component_overlaps,
        ) = _records_by_role(records, split_manifest, near_neighbors)
        source_shared = _cross_role_shared(by_role, "source")
        candidate_shared = _cross_role_shared(by_role, "candidate")
        path_shared = _cross_role_shared(by_role, "path")
        near_shared = _near_neighbor_overlaps(by_role, near_neighbors)
        reverse = _reverse_edges(by_role)
        metadata = _metadata_overlaps(by_role)
        (
            explained_metadata,
            unexplained_metadata,
            metadata_axis_status,
            policy_issues,
        ) = _classify_metadata_overlaps(metadata, split_manifest.get("overlap_policy"))
        universe_issues.extend(policy_issues)
        (
            required_axis_status,
            required_axis_overlap_count,
            required_axis_issues,
        ) = _required_axis_audit(
            split_manifest,
            metadata,
            near_shared,
        )
        universe_issues.extend(required_axis_issues)
    except SplitGraphError as exc:
        raise LeakageAuditError(str(exc)) from exc

    train_intermediate = _sequence_sets(by_role["train"], "intermediate")
    heldout_candidates = _sequence_sets(
        by_role["validation"], "candidate"
    ) | _sequence_sets(by_role["test"], "candidate")
    final_as_intermediate = train_intermediate & heldout_candidates
    unexplained_count = (
        len(unexplained_metadata)
        + len(duplicate_assignments)
        + len(component_overlaps)
        + len(universe_issues)
    )
    foundation = normalize_foundation_exposure(foundation_exposure)
    counts = {
        "exact_source_leakage_count": len(source_shared),
        "exact_candidate_leakage_count": len(candidate_shared),
        "reverse_edge_leakage_count": len(reverse),
        "path_leakage_count": len(path_shared),
        "near_neighbor_leakage_count": len(near_shared),
        "final_endpoint_as_train_intermediate_count": len(final_as_intermediate),
        "metadata_overlap_count": len(metadata),
        "explained_metadata_overlap_count": len(explained_metadata),
        "unexplained_metadata_overlap_count": len(unexplained_metadata),
        "record_role_overlap_count": len(duplicate_assignments),
        "component_role_overlap_count": len(component_overlaps),
        "frozen_universe_issue_count": len(universe_issues),
        "required_axis_overlap_count": required_axis_overlap_count,
        "unexplained_overlap_count": unexplained_count,
    }
    gates = {
        "unexplained_overlap_zero": unexplained_count == 0,
        "exact_source_overlap_zero": len(source_shared) == 0,
        "exact_candidate_overlap_zero": len(candidate_shared) == 0,
        "reverse_edge_leakage_zero": len(reverse) == 0,
        "path_leakage_zero": len(path_shared) == 0,
        "near_neighbor_leakage_zero": len(near_shared) == 0,
        "final_endpoint_as_train_intermediate_zero": (len(final_as_intermediate) == 0),
        "required_axis_overlap_zero": (
            required_axis_overlap_count == 0 and not required_axis_issues
        ),
        "foundation_overlap_gate": foundation["gate_passed"],
    }
    return {
        "schema_version": "utr_b0_partition_leakage_report.v2",
        "partition_id": split_manifest.get("partition_id"),
        "split_partition_sha256": split_manifest.get("partition_sha256"),
        "split_kind": split_manifest.get("split_kind"),
        "region": split_manifest.get("region"),
        "source_region": split_manifest.get("source_region"),
        "target_region": split_manifest.get("target_region"),
        "heldout_study": split_manifest.get("heldout_study"),
        "counts": counts,
        "acceptance_gates": gates,
        "gate_passed": all(gates.values()),
        "foundation_pretraining_overlap": foundation,
        "metadata_axis_status": metadata_axis_status,
        "required_axis_status": required_axis_status,
        "examples": {
            "exact_source": [
                {
                    "roles": [left, right],
                    "sequence_sha256": _sequence_digest(sequence),
                }
                for left, right, sequence in sorted(source_shared)[:20]
            ],
            "exact_candidate": [
                {
                    "roles": [left, right],
                    "sequence_sha256": _sequence_digest(sequence),
                }
                for left, right, sequence in sorted(candidate_shared)[:20]
            ],
            "reverse_edge": [
                {
                    "roles": [left, right],
                    "source_sha256": source_sha,
                    "candidate_sha256": candidate_sha,
                }
                for left, right, source_sha, candidate_sha in sorted(reverse)[:20]
            ],
            "path": [
                {
                    "roles": [left, right],
                    "sequence_sha256": _sequence_digest(sequence),
                }
                for left, right, sequence in sorted(path_shared)[:20]
            ],
            "near_neighbor": [
                {
                    "roles": [left, right],
                    "cluster_id": cluster_id,
                }
                for left, right, cluster_id in sorted(near_shared)[:20]
            ],
            "final_endpoint_as_train_intermediate": [
                {"sequence_sha256": _sequence_digest(sequence)}
                for sequence in sorted(final_as_intermediate)[:20]
            ],
            "metadata_overlap": [
                {
                    "roles": [left, right],
                    "field": field,
                    "value": value,
                    "disposition": (
                        "EXPLAINED"
                        if (left, right, field, value) in explained_metadata
                        else "UNEXPLAINED"
                    ),
                }
                for left, right, field, value in sorted(metadata)[:20]
            ],
            "record_role_overlap": duplicate_assignments[:20],
            "component_role_overlap": component_overlaps[:20],
            "frozen_universe_issue": universe_issues[:20],
        },
    }


def _blocked_partition_report(
    partition: Mapping[str, Any],
    foundation_exposure: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    foundation = normalize_foundation_exposure(foundation_exposure)
    counts = {
        "exact_source_leakage_count": 0,
        "exact_candidate_leakage_count": 0,
        "reverse_edge_leakage_count": 0,
        "path_leakage_count": 0,
        "near_neighbor_leakage_count": 0,
        "final_endpoint_as_train_intermediate_count": 0,
        "metadata_overlap_count": 0,
        "explained_metadata_overlap_count": 0,
        "unexplained_metadata_overlap_count": 0,
        "record_role_overlap_count": 0,
        "component_role_overlap_count": 0,
        "frozen_universe_issue_count": 1,
        "required_axis_overlap_count": 0,
        "unexplained_overlap_count": 1,
    }
    gates = {
        "unexplained_overlap_zero": False,
        "exact_source_overlap_zero": False,
        "exact_candidate_overlap_zero": False,
        "reverse_edge_leakage_zero": False,
        "path_leakage_zero": False,
        "near_neighbor_leakage_zero": False,
        "final_endpoint_as_train_intermediate_zero": False,
        "required_axis_overlap_zero": False,
        "foundation_overlap_gate": foundation["gate_passed"],
    }
    return {
        "schema_version": "utr_b0_partition_leakage_report.v2",
        "partition_id": partition.get("partition_id"),
        "split_partition_sha256": partition.get("partition_sha256"),
        "split_kind": partition.get("split_kind"),
        "status": "BLOCKED",
        "blocked_reasons": list(partition.get("blocked_reasons", [])),
        "counts": counts,
        "acceptance_gates": gates,
        "gate_passed": False,
        "foundation_pretraining_overlap": foundation,
        "metadata_axis_status": {},
        "required_axis_status": {},
        "examples": {"frozen_universe_issue": [{"kind": "partition_blocked"}]},
    }


def audit_split_manifest(
    records: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    foundation_exposure: Optional[Mapping[str, Any]] = None,
    *,
    _near_neighbors: Optional[NearNeighborClusters] = None,
) -> Dict[str, Any]:
    """Audit every declared partition and aggregate only exact gate counts."""

    try:
        near_neighbors = _near_neighbors or global_near_neighbor_clusters(records)
    except SplitGraphError as exc:
        raise LeakageAuditError(str(exc)) from exc
    partitions = split_manifest.get("partitions")
    required_ids = split_manifest.get("required_partition_ids")
    structural_issues: List[Dict[str, Any]] = []
    if not isinstance(partitions, list) or not partitions:
        partitions = []
        structural_issues.append({"kind": "partitions_missing_or_empty"})
    if (
        not isinstance(required_ids, list)
        or not all(isinstance(value, str) and value for value in required_ids)
        or len(required_ids) != len(set(required_ids))
    ):
        required_ids = []
        structural_issues.append({"kind": "required_partition_ids_invalid"})
    try:
        frozen_required_ids = expected_partition_ids(
            str(split_manifest.get("split_kind") or ""),
            region=split_manifest.get("region"),
        )
    except SplitGraphError as exc:
        frozen_required_ids = ()
        structural_issues.append(
            {
                "kind": "frozen_partition_identity_invalid",
                "detail": str(exc),
            }
        )
    if tuple(required_ids) != frozen_required_ids:
        structural_issues.append(
            {
                "kind": "frozen_required_partition_set_mismatch",
                "expected": list(frozen_required_ids),
                "observed": list(required_ids),
            }
        )
    observed_ids = [
        str(partition.get("partition_id") or "")
        for partition in partitions
        if isinstance(partition, Mapping)
    ]
    if sorted(observed_ids) != sorted(required_ids):
        structural_issues.append(
            {
                "kind": "required_partition_set_mismatch",
                "required": sorted(required_ids),
                "observed": sorted(observed_ids),
            }
        )
    if len(observed_ids) != len(set(observed_ids)) or "" in observed_ids:
        structural_issues.append({"kind": "partition_ids_missing_or_duplicate"})

    expected_full_binding = {
        "full_record_count": len(records),
        "full_record_ids_sha256": record_ids_sha256(records),
        "full_record_universe_sha256": record_universe_sha256(records),
    }
    for field, expected in expected_full_binding.items():
        if split_manifest.get(field) != expected:
            structural_issues.append(
                {
                    "kind": "outer_full_universe_binding_mismatch",
                    "field": field,
                    "expected": expected,
                    "observed": split_manifest.get(field),
                }
            )
    if split_manifest.get("near_neighbor_binding") != dict(near_neighbors.binding):
        structural_issues.append({"kind": "outer_near_neighbor_binding_mismatch"})

    partition_reports: List[Dict[str, Any]] = []
    for partition in partitions:
        if not isinstance(partition, Mapping):
            structural_issues.append({"kind": "partition_entry_not_object"})
            continue
        try:
            expected_contract = expected_partition_leakage_contract(
                str(split_manifest.get("split_kind") or ""),
                region=split_manifest.get("region"),
                partition_id=str(partition.get("partition_id") or ""),
            )
        except SplitGraphError as exc:
            structural_issues.append(
                {
                    "kind": "frozen_partition_leakage_contract_invalid",
                    "partition_id": partition.get("partition_id"),
                    "detail": str(exc),
                }
            )
        else:
            mismatched_fields = [
                field
                for field, expected in expected_contract.items()
                if partition.get(field) != expected
            ]
            if mismatched_fields:
                structural_issues.append(
                    {
                        "kind": "frozen_partition_leakage_contract_mismatch",
                        "partition_id": partition.get("partition_id"),
                        "fields": mismatched_fields,
                    }
                )
        for field, expected in expected_full_binding.items():
            if partition.get(field) != expected:
                structural_issues.append(
                    {
                        "kind": "partition_common_universe_mismatch",
                        "partition_id": partition.get("partition_id"),
                        "field": field,
                    }
                )
        if partition.get("status") == "READY":
            partition_reports.append(
                audit_cross_role_leakage(
                    records,
                    partition,
                    foundation_exposure=foundation_exposure,
                    _near_neighbors=near_neighbors,
                )
            )
        else:
            partition_reports.append(
                _blocked_partition_report(partition, foundation_exposure)
            )

    count_names = (
        "exact_source_leakage_count",
        "exact_candidate_leakage_count",
        "reverse_edge_leakage_count",
        "path_leakage_count",
        "near_neighbor_leakage_count",
        "final_endpoint_as_train_intermediate_count",
        "metadata_overlap_count",
        "explained_metadata_overlap_count",
        "unexplained_metadata_overlap_count",
        "record_role_overlap_count",
        "component_role_overlap_count",
        "frozen_universe_issue_count",
        "required_axis_overlap_count",
        "unexplained_overlap_count",
    )
    counts = {
        name: sum(int(report["counts"].get(name, 0)) for report in partition_reports)
        for name in count_names
    }
    counts["frozen_universe_issue_count"] += len(structural_issues)
    counts["unexplained_overlap_count"] += len(structural_issues)
    gate_names = (
        "unexplained_overlap_zero",
        "exact_source_overlap_zero",
        "exact_candidate_overlap_zero",
        "reverse_edge_leakage_zero",
        "path_leakage_zero",
        "near_neighbor_leakage_zero",
        "final_endpoint_as_train_intermediate_zero",
        "required_axis_overlap_zero",
        "foundation_overlap_gate",
    )
    gates = {
        gate: (
            bool(partition_reports)
            and all(
                report["acceptance_gates"].get(gate) is True
                for report in partition_reports
            )
        )
        for gate in gate_names
    }
    gates["unexplained_overlap_zero"] = (
        gates["unexplained_overlap_zero"]
        and not structural_issues
        and counts["unexplained_overlap_count"] == 0
    )
    foundation = normalize_foundation_exposure(foundation_exposure)
    return {
        "schema_version": "utr_b0_leakage_report.v2",
        "split_kind": split_manifest.get("split_kind"),
        "region": split_manifest.get("region"),
        "source_region": split_manifest.get("source_region"),
        "target_region": split_manifest.get("target_region"),
        "required_partition_ids": list(required_ids),
        "partition_count": len(partition_reports),
        "partitions": partition_reports,
        "counts": counts,
        "acceptance_gates": gates,
        "gate_passed": all(gates.values()),
        "common_universe_binding": expected_full_binding,
        "foundation_pretraining_overlap": foundation,
        "structural_issues": structural_issues,
    }


__all__ = [
    "LeakageAuditError",
    "audit_cross_role_leakage",
    "audit_split_manifest",
    "normalize_foundation_exposure",
]
