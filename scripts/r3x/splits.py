"""Leakage-safe source-relation components and deterministic R3X splits."""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .protocol import canonical_sha256


HEX64 = re.compile(r"^[0-9a-f]{64}$")
PARTITIONS = ("train", "development", "internal_test")
SPLIT_ASSIGNMENT_ALGORITHM_ID = "SHA256_SEEDED_COMPONENT_ORDER_V1"
SPLIT_ASSIGNMENT_PARAMETERS = {
    "component_input_order": "LEXICOGRAPHIC_ASCENDING",
    "component_priority": "SHA256_CANONICAL_JSON_ALGORITHM_SEED_COMPONENT_ID",
    "partition_order": list(PARTITIONS),
    "partition_count_policy": "FLOOR_70_15_REMAINDER_WITH_MINIMUMS_V1",
    "train_fraction": 0.70,
    "development_fraction": 0.15,
}


class SplitHoldError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None):
        self.code = code
        self.details = dict(details or {})
        super().__init__(f"{code}: {message}")


def normalize_sequence(value: object) -> str:
    if not isinstance(value, str):
        raise SplitHoldError("HOLD_SEQUENCE_IDENTITY", "sequence must be a string")
    sequence = "".join(value.upper().replace("T", "U").split())
    if not sequence or any(base not in "ACGU" for base in sequence):
        raise SplitHoldError("HOLD_SEQUENCE_IDENTITY", "sequence must normalize to non-empty A/C/G/U")
    return sequence


def sequence_identity(value: object) -> str:
    return hashlib.sha256(normalize_sequence(value).encode("ascii")).hexdigest()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        # Lexicographic root makes the graph canonical independent of row order.
        lo, hi = sorted((a, b))
        self.parent[hi] = lo


def _row_id(row: Mapping) -> str:
    value = row.get("r3_input_row_sha256") or row.get("record_id")
    if not isinstance(value, str) or not value:
        raise SplitHoldError("HOLD_ROW_IDENTITY", "row lacks stable r3_input_row_sha256/record_id")
    return value


def _sequence(row: Mapping, normalized: str, fallback: str) -> tuple[str, str]:
    raw = row.get(normalized, row.get(fallback))
    seq = normalize_sequence(raw)
    digest = sequence_identity(seq)
    declared = row.get(normalized.replace("normalized_", "") + "_sha256")
    # Canonical field names in the R3 preparation bundle.
    if normalized == "normalized_source_sequence":
        declared = row.get("source_sequence_sha256", declared)
    else:
        declared = row.get("candidate_sequence_sha256", declared)
    if declared is not None and str(declared).lower() != digest:
        raise SplitHoldError(
            "HOLD_SEQUENCE_IDENTITY_MISMATCH",
            f"declared {normalized} hash does not match normalized bytes",
            details={"expected": digest, "declared": declared},
        )
    return seq, digest


def _atoms(row: Mapping) -> list[dict]:
    atoms = row.get("frozen_relation_atoms")
    if not isinstance(atoms, list) or not atoms:
        raise SplitHoldError("HOLD_RELATION_EVIDENCE", "frozen relation atoms are required")
    normalized = []
    for atom in atoms:
        if not isinstance(atom, Mapping):
            raise SplitHoldError("HOLD_RELATION_EVIDENCE", "relation atoms must be objects")
        if atom.get("status") != "FROZEN":
            raise SplitHoldError("HOLD_RELATION_EVIDENCE", "every relation atom must be FROZEN")
        atom_type = str(atom.get("atom_type") or "").strip()
        atom_id = str(atom.get("atom_id") or "").strip()
        member = str(atom.get("member") or "").strip().upper()
        evidence = str(atom.get("evidence_sha256") or "").strip().lower()
        if not atom_type or not atom_id or member not in {"SOURCE", "CANDIDATE", "PAIR", "BOTH"}:
            raise SplitHoldError("HOLD_RELATION_EVIDENCE", "relation atom fields are incomplete")
        if not HEX64.fullmatch(evidence):
            raise SplitHoldError("HOLD_RELATION_EVIDENCE", "relation atom evidence_sha256 is invalid")
        normalized.append({
            "atom_type": atom_type,
            "atom_id": atom_id,
            "member": member,
            "status": "FROZEN",
            "evidence_sha256": evidence,
        })
    return sorted(normalized, key=lambda atom: (
        atom["atom_type"], atom["atom_id"], atom["member"], atom["evidence_sha256"]
    ))


def build_source_components(rows: Sequence[Mapping]) -> dict:
    """Build the global transitive source/candidate/alias relation graph.

    `source_group_id_v2` is recorded only as a ranking-pool key and is never an
    edge.  This is essential because it is context/library scoped rather than a
    global biological source component.
    """

    rows = list(rows)
    if not rows:
        raise SplitHoldError("HOLD_EMPTY_SPLIT_INPUT", "component graph refuses empty rows")
    prepared = []
    seen_ids: set[str] = set()
    uf = _UnionFind()
    edges: set[tuple[str, str, str]] = set()
    atom_evidence: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not isinstance(row, Mapping):
            raise SplitHoldError("HOLD_INVALID_SPLIT_ROW", "split rows must be objects")
        row_id = _row_id(row)
        if row_id in seen_ids:
            raise SplitHoldError("HOLD_DUPLICATE_ROW_ID", f"duplicate split row {row_id}")
        seen_ids.add(row_id)
        source_seq, source_hash = _sequence(row, "normalized_source_sequence", "source_sequence")
        candidate_seq, candidate_hash = _sequence(row, "normalized_candidate_sequence", "candidate_sequence")
        pool = row.get("source_group_id_v2")
        if not isinstance(pool, str) or not pool:
            raise SplitHoldError("HOLD_RANKING_POOL", "source_group_id_v2 ranking pool is required")
        source_node = f"SEQ:{source_hash}"
        candidate_node = f"SEQ:{candidate_hash}"
        uf.union(source_node, candidate_node)
        edges.add(tuple(sorted((source_node, candidate_node))) + ("SOURCE_CANDIDATE_RELATION",))
        atoms = _atoms(row)
        for atom in atoms:
            atom_identity = {
                "atom_type": atom["atom_type"],
                "atom_id": atom["atom_id"],
            }
            # Logical relation identity must not depend on evidence-file
            # packaging.  Multiple frozen evidence hashes support one node.
            atom_node = f"ATOM:{canonical_sha256(atom_identity)}"
            atom_evidence[atom_node].add(atom["evidence_sha256"])
            members = (
                (source_node,) if atom["member"] == "SOURCE"
                else (candidate_node,) if atom["member"] == "CANDIDATE"
                else (source_node, candidate_node)
            )
            for member_node in members:
                uf.union(atom_node, member_node)
                edges.add(tuple(sorted((atom_node, member_node))) + ("FROZEN_ALIAS_ATOM",))
        prepared.append({
            "row_id": row_id,
            "source_sequence": source_seq,
            "candidate_sequence": candidate_seq,
            "source_sequence_sha256": source_hash,
            "candidate_sequence_sha256": candidate_hash,
            "source_node": source_node,
            "candidate_node": candidate_node,
            "ranking_pool_id": pool,
            "atoms": atoms,
            "study": str(row.get("study") or ""),
            "endpoint": str(row.get("endpoint") or row.get("endpoint_id") or ""),
            "context": str(row.get("publication_context_id") or row.get("context_id_v2") or ""),
        })

    nodes_by_root: dict[str, list[str]] = defaultdict(list)
    for node in sorted(uf.parent):
        nodes_by_root[uf.find(node)].append(node)
    component_id_by_root = {
        root: "SRCOMP_" + canonical_sha256(sorted(nodes))[:24]
        for root, nodes in nodes_by_root.items()
    }
    assignments = {}
    for item in sorted(prepared, key=lambda value: value["row_id"]):
        root = uf.find(item["source_node"])
        assignments[item["row_id"]] = {
            "source_component_id": component_id_by_root[root],
            "source_sequence_sha256": item["source_sequence_sha256"],
            "candidate_sequence_sha256": item["candidate_sequence_sha256"],
            "ranking_pool_id": item["ranking_pool_id"],
            "study": item["study"],
            "endpoint": item["endpoint"],
            "publication_context_id": item["context"],
        }
    rows_by_component: dict[str, list[str]] = defaultdict(list)
    for row_id, assignment in assignments.items():
        rows_by_component[assignment["source_component_id"]].append(row_id)
    components: dict[str, dict] = {}
    for root, nodes in sorted(nodes_by_root.items(), key=lambda item: component_id_by_root[item[0]]):
        component_id = component_id_by_root[root]
        members = rows_by_component[component_id]
        components[component_id] = {
            "node_ids": sorted(nodes),
            "row_ids": sorted(members),
            "node_membership_sha256": canonical_sha256(sorted(nodes)),
            "row_membership_sha256": canonical_sha256(sorted(members)),
        }
    graph = {
        "schema_version": "r3_source_relation_component_graph_v1",
        "split_unit": "GLOBAL_SOURCE_CANDIDATE_ALIAS_TRANSITIVE_COMPONENT",
        "ranking_pool_policy": "SAME_CONTEXT_RANKING_ONLY_NOT_SPLIT_UNIT",
        "n_rows": len(assignments),
        "n_components": len(components),
        "components": components,
        "row_assignments": assignments,
        "edges": [list(edge) for edge in sorted(edges)],
        "relation_atom_evidence": {
            node: sorted(values) for node, values in sorted(atom_evidence.items())
        },
    }
    graph["graph_sha256"] = canonical_sha256(graph)
    return graph


def _partition_counts(n_components: int) -> tuple[int, int, int]:
    if n_components < 3:
        return max(1, n_components), 0, max(0, n_components - 1)
    n_train = max(1, int(math.floor(0.70 * n_components)))
    n_development = max(1, int(math.floor(0.15 * n_components)))
    if n_train + n_development >= n_components:
        n_train = n_components - 2
        n_development = 1
    return n_train, n_development, n_components - n_train - n_development


def _require_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitHoldError("HOLD_INVALID_SPLIT_SEED", "split seed must be an integer")
    return value


def _ordered_components(component_ids: Sequence[str], seed: int) -> list[str]:
    """Return a runtime-independent, seed-bound component ordering."""

    seed = _require_seed(seed)
    return sorted(
        component_ids,
        key=lambda component_id: (
            canonical_sha256({
                "assignment_algorithm_id": SPLIT_ASSIGNMENT_ALGORITHM_ID,
                "seed": seed,
                "source_component_id": component_id,
            }),
            component_id,
        ),
    )


def _partition_by_component(component_ids: Sequence[str], seed: int) -> dict[str, str]:
    ordered = _ordered_components(component_ids, seed)
    n_train, n_development, _ = _partition_counts(len(ordered))
    return {
        component_id: (
            "train"
            if index < n_train
            else "development"
            if index < n_train + n_development
            else "internal_test"
        )
        for index, component_id in enumerate(ordered)
    }


def _split_counts(
    row_assignments: Mapping[str, Mapping],
    partition_by_component: Mapping[str, str],
) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    component_counts = {partition: 0 for partition in PARTITIONS}
    for partition in partition_by_component.values():
        component_counts[partition] += 1
    row_counts = {partition: 0 for partition in PARTITIONS}
    cell_row_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {partition: 0 for partition in PARTITIONS}
    )
    cell_components: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {partition: set() for partition in PARTITIONS}
    )
    for assignment in row_assignments.values():
        partition = assignment["partition"]
        row_counts[partition] += 1
        cell = "|".join((
            assignment["study"],
            assignment["endpoint"],
            assignment["publication_context_id"],
        ))
        cell_row_counts[cell][partition] += 1
        cell_components[cell][partition].add(assignment["source_component_id"])
    cell_component_counts = {
        cell: {
            partition: len(component_ids)
            for partition, component_ids in partitions.items()
        }
        for cell, partitions in sorted(cell_components.items())
    }
    return (
        component_counts,
        row_counts,
        {cell: dict(values) for cell, values in sorted(cell_row_counts.items())},
        cell_component_counts,
    )


def _scientific_scope(assignments: Mapping[str, Mapping]) -> tuple[str, bool]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for row in assignments.values():
        if row["study"] and row["endpoint"]:
            mapping[row["study"]].add(row["endpoint"])
    if len(mapping) < 2:
        return "DEVELOPMENT_ONLY_SINGLE_STUDY", False
    shared = set.intersection(*mapping.values())
    if not shared and len(mapping) == 2:
        return "DEVELOPMENT_ONLY_TWO_STUDY_NESTED_ENDPOINTS", False
    if not shared:
        return "DEVELOPMENT_ONLY_ENDPOINT_STUDY_CONFOUNDED", False
    # Endpoint-string overlap is only a candidate for an estimand registry
    # review.  A derivable split never self-confers publication eligibility.
    return "DEVELOPMENT_ONLY_SAME_ENDPOINT_ESTIMAND_AUTHORITY_REQUIRED", False


def build_split_manifest(
    rows: Sequence[Mapping],
    *,
    seed: int,
    publication_claim: bool = False,
    support_policy: Mapping[str, Any] | None = None,
    authority_bindings: Mapping[str, str] | None = None,
) -> dict:
    graph = build_source_components(rows)
    component_ids = sorted(graph["components"])
    seed = _require_seed(seed)
    frozen_support_policy: dict[str, Any]
    if publication_claim:
        if not isinstance(support_policy, Mapping) or support_policy.get("status") != "FROZEN":
            raise SplitHoldError(
                "HOLD_MISSING_SPLIT_SUPPORT_POLICY",
                "publication split requires a pre-frozen precision/support policy",
            )
        minimum = support_policy.get("minimum_components_per_cell_partition")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise SplitHoldError(
                "HOLD_INVALID_SPLIT_SUPPORT_POLICY",
                "minimum_components_per_cell_partition must be a positive integer",
            )
        per_cell_components: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for assignment in graph["row_assignments"].values():
            cell = (assignment["study"], assignment["endpoint"], assignment["publication_context_id"])
            per_cell_components[cell].add(assignment["source_component_id"])
        required_total = 3 * minimum
        bad = {
            "|".join(cell): len(values)
            for cell, values in per_cell_components.items()
            if len(values) < required_total
        }
        if bad:
            raise SplitHoldError(
                "HOLD_DEGENERATE_PUBLICATION_CELL",
                "publication cells lack the frozen per-partition component support",
                details={"component_counts": bad, "required_total": required_total},
            )
        frozen_support_policy = {
            "status": "FROZEN",
            "minimum_components_per_cell_partition": int(minimum),
            "authority_sha256": canonical_sha256(dict(support_policy)),
        }
    else:
        frozen_support_policy = {
            "status": "NOT_APPLICABLE_DEVELOPMENT",
            "minimum_components_per_cell_partition": None,
            "authority_sha256": None,
        }
    partition_by_component = _partition_by_component(component_ids, seed)
    row_assignments = {}
    for row_id, assignment in sorted(graph["row_assignments"].items()):
        row_assignments[row_id] = {
            **assignment,
            "partition": partition_by_component[assignment["source_component_id"]],
        }
    component_counts, row_counts, cell_row_counts, cell_component_counts = _split_counts(
        row_assignments, partition_by_component
    )
    scope, cross_study = _scientific_scope(row_assignments)
    manifest = {
        "schema_version": "r3_source_component_split_manifest_v1",
        "status": (
            "FROZEN"
            if authority_bindings is not None
            else "FROZEN_DERIVABLE_FIXTURE"
            if publication_claim
            else "DEVELOPMENT_DERIVABLE"
        ),
        "seed": seed,
        "assignment_algorithm_id": SPLIT_ASSIGNMENT_ALGORITHM_ID,
        "assignment_algorithm_parameters": dict(SPLIT_ASSIGNMENT_PARAMETERS),
        "split_unit": graph["split_unit"],
        "source_component_graph_sha256": graph["graph_sha256"],
        "ranking_pool_policy": graph["ranking_pool_policy"],
        "partition_component_counts": component_counts,
        "partition_row_counts": row_counts,
        "cell_row_counts": cell_row_counts,
        "cell_component_counts": cell_component_counts,
        "row_assignments": row_assignments,
        "publication_claim": bool(publication_claim),
        "support_policy": frozen_support_policy,
        "scientific_scope": scope,
        "publication_cross_study_evidence": cross_study,
        "sealed_access": False,
    }
    if authority_bindings is not None:
        required = {"ready_dataset_sha256", "inclusion_manifest_sha256"}
        if set(authority_bindings) != required or any(
            not HEX64.fullmatch(str(value)) for value in authority_bindings.values()
        ):
            raise SplitHoldError(
                "HOLD_INVALID_SPLIT_AUTHORITY_BINDING",
                "split authority bindings must contain exact dataset and inclusion hashes",
            )
        manifest.update({key: str(value) for key, value in authority_bindings.items()})
    if publication_claim:
        minimum = int(support_policy["minimum_components_per_cell_partition"])
        missing_support = {
            cell: dict(partitions)
            for cell, partitions in cell_component_counts.items()
            if any(value < minimum for value in partitions.values())
        }
        if missing_support:
            raise SplitHoldError(
                "HOLD_INSUFFICIENT_CELL_PARTITION_SUPPORT",
                "outcome-blind split does not satisfy frozen per-cell partition support",
                details={"support": missing_support, "minimum": minimum},
            )
        manifest["support_check"] = {
            "status": "PASS_FROZEN_MINIMUM_COMPONENT_SUPPORT",
            "minimum_components_per_cell_partition": minimum,
            "insufficient_cells": {},
        }
    else:
        manifest["support_check"] = {
            "status": "NOT_APPLICABLE_DEVELOPMENT",
            "minimum_components_per_cell_partition": None,
            "insufficient_cells": {},
        }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    audit_component_disjointness(manifest)
    return manifest


def verify_split_manifest_against_rows(rows: Sequence[Mapping], manifest: Mapping) -> dict:
    """Rebuild and replay the exact frozen split, not merely its disjointness."""

    if not isinstance(manifest, Mapping):
        raise SplitHoldError("HOLD_INVALID_SPLIT_MANIFEST", "split manifest must be an object")
    declared_hash = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    if not isinstance(declared_hash, str) or canonical_sha256(payload) != declared_hash:
        raise SplitHoldError(
            "HOLD_STALE_SPLIT_MANIFEST_HASH",
            "split manifest internal canonical hash is stale or missing",
        )
    if manifest.get("schema_version") != "r3_source_component_split_manifest_v1":
        raise SplitHoldError(
            "HOLD_UNREPLAYABLE_SPLIT_SCHEMA",
            "split manifest schema is not supported for deterministic replay",
        )
    if (
        manifest.get("assignment_algorithm_id") != SPLIT_ASSIGNMENT_ALGORITHM_ID
        or manifest.get("assignment_algorithm_parameters") != SPLIT_ASSIGNMENT_PARAMETERS
    ):
        raise SplitHoldError(
            "HOLD_UNREPLAYABLE_SPLIT_ALGORITHM",
            "split authority does not freeze the supported algorithm and exact parameters",
        )
    seed = _require_seed(manifest.get("seed"))
    graph = build_source_components(rows)
    if manifest.get("source_component_graph_sha256") != graph["graph_sha256"]:
        raise SplitHoldError(
            "HOLD_SPLIT_AUTHORITY_MISMATCH",
            "split graph hash does not match the ready-row relation graph",
        )
    declared = manifest.get("row_assignments")
    if not isinstance(declared, Mapping) or set(declared) != set(graph["row_assignments"]):
        raise SplitHoldError(
            "HOLD_SPLIT_AUTHORITY_MISMATCH",
            "split and rebuilt graph row identities differ",
        )
    partition_by_component = _partition_by_component(sorted(graph["components"]), seed)
    expected_assignments = {
        row_id: {
            **rebuilt,
            "partition": partition_by_component[rebuilt["source_component_id"]],
        }
        for row_id, rebuilt in sorted(graph["row_assignments"].items())
    }
    identity_fields = (
        "source_component_id",
        "source_sequence_sha256",
        "candidate_sequence_sha256",
        "ranking_pool_id",
        "study",
        "endpoint",
        "publication_context_id",
    )
    identity_mismatches = [
        row_id
        for row_id, expected in expected_assignments.items()
        if not isinstance(declared[row_id], Mapping)
        or any(declared[row_id].get(field) != expected[field] for field in identity_fields)
    ]
    if identity_mismatches:
        raise SplitHoldError(
            "HOLD_SPLIT_AUTHORITY_MISMATCH",
            "split row metadata/components do not match rebuilt identities",
            details={
                "row_ids": sorted(identity_mismatches)[:20],
                "n_mismatches": len(identity_mismatches),
            },
        )
    mismatches = [
        row_id
        for row_id, expected in expected_assignments.items()
        if not isinstance(declared[row_id], Mapping) or dict(declared[row_id]) != expected
    ]
    if mismatches:
        raise SplitHoldError(
            "HOLD_SPLIT_REPLAY_MISMATCH",
            "split row assignments do not match the deterministic seed replay",
            details={"row_ids": sorted(mismatches)[:20], "n_mismatches": len(mismatches)},
        )
    expected_counts = _split_counts(expected_assignments, partition_by_component)
    for field, expected in zip(
        (
            "partition_component_counts",
            "partition_row_counts",
            "cell_row_counts",
            "cell_component_counts",
        ),
        expected_counts,
    ):
        if manifest.get(field) != expected:
            raise SplitHoldError(
                "HOLD_SPLIT_REPLAY_MISMATCH",
                f"split {field} does not match the deterministic replay",
                details={"field": field, "expected": expected, "actual": manifest.get(field)},
            )
    publication_claim = manifest.get("publication_claim")
    if not isinstance(publication_claim, bool):
        raise SplitHoldError(
            "HOLD_UNREPLAYABLE_SPLIT_SCHEMA",
            "split manifest must freeze publication_claim as a boolean",
        )
    support_policy = manifest.get("support_policy")
    if not isinstance(support_policy, Mapping) or set(support_policy) != {
        "status", "minimum_components_per_cell_partition", "authority_sha256"
    }:
        raise SplitHoldError(
            "HOLD_UNREPLAYABLE_SPLIT_SCHEMA",
            "split manifest lacks an exact replayable support policy",
        )
    if publication_claim:
        minimum = support_policy.get("minimum_components_per_cell_partition")
        if (
            support_policy.get("status") != "FROZEN"
            or isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or minimum < 1
            or not HEX64.fullmatch(str(support_policy.get("authority_sha256") or ""))
        ):
            raise SplitHoldError(
                "HOLD_UNREPLAYABLE_SPLIT_SCHEMA",
                "publication split support policy is incomplete or unfrozen",
            )
        insufficient = {
            cell: dict(partitions)
            for cell, partitions in expected_counts[3].items()
            if any(value < minimum for value in partitions.values())
        }
        expected_support = {
            "status": "PASS_FROZEN_MINIMUM_COMPONENT_SUPPORT" if not insufficient
            else "FAIL_INSUFFICIENT_COMPONENT_SUPPORT",
            "minimum_components_per_cell_partition": minimum,
            "insufficient_cells": insufficient,
        }
        if insufficient:
            raise SplitHoldError(
                "HOLD_INSUFFICIENT_CELL_PARTITION_SUPPORT",
                "replayed publication split lacks frozen per-cell component support",
                details={"support": insufficient, "minimum": minimum},
            )
    else:
        if dict(support_policy) != {
            "status": "NOT_APPLICABLE_DEVELOPMENT",
            "minimum_components_per_cell_partition": None,
            "authority_sha256": None,
        }:
            raise SplitHoldError(
                "HOLD_UNREPLAYABLE_SPLIT_SCHEMA",
                "development split must carry the exact not-applicable support policy",
            )
        expected_support = {
            "status": "NOT_APPLICABLE_DEVELOPMENT",
            "minimum_components_per_cell_partition": None,
            "insufficient_cells": {},
        }
    if manifest.get("support_check") != expected_support:
        raise SplitHoldError(
            "HOLD_SPLIT_REPLAY_MISMATCH",
            "split support check does not match the replayed component support",
        )
    scope, cross_study = _scientific_scope(expected_assignments)
    for field, expected in (
        ("split_unit", graph["split_unit"]),
        ("ranking_pool_policy", graph["ranking_pool_policy"]),
        ("scientific_scope", scope),
        ("publication_cross_study_evidence", cross_study),
        ("sealed_access", False),
    ):
        if manifest.get(field) != expected:
            raise SplitHoldError(
                "HOLD_SPLIT_REPLAY_MISMATCH",
                f"split {field} does not match rebuilt ready-row evidence",
                details={"field": field, "expected": expected, "actual": manifest.get(field)},
            )
    audit_component_disjointness(manifest)
    return {
        "status": "PASS_REBUILT_COMPONENT_SPLIT",
        "n_rows": len(declared),
        "source_component_graph_sha256": graph["graph_sha256"],
        "assignment_algorithm_id": SPLIT_ASSIGNMENT_ALGORITHM_ID,
        "seed": seed,
        "partition_component_counts": expected_counts[0],
        "partition_row_counts": expected_counts[1],
    }


def audit_component_disjointness(manifest: Mapping) -> dict:
    assignments = manifest.get("row_assignments")
    if not isinstance(assignments, Mapping):
        raise SplitHoldError("HOLD_INVALID_SPLIT_MANIFEST", "manifest lacks row_assignments")
    partitions: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments.values():
        if not isinstance(assignment, Mapping):
            raise SplitHoldError("HOLD_INVALID_SPLIT_MANIFEST", "row assignment must be an object")
        component = assignment.get("source_component_id")
        partition = assignment.get("partition")
        if not isinstance(component, str) or not component or partition not in PARTITIONS:
            raise SplitHoldError("HOLD_INVALID_SPLIT_MANIFEST", "row assignment is incomplete")
        partitions[component].add(partition)
    overlap = sorted(component for component, values in partitions.items() if len(values) > 1)
    if overlap:
        raise SplitHoldError(
            "HOLD_COMPONENT_OVERLAP",
            "a source-relation component crosses partitions",
            details={"overlap_component_ids": overlap},
        )
    return {"status": "PASS_COMPONENT_DISJOINT", "overlap_component_ids": []}
