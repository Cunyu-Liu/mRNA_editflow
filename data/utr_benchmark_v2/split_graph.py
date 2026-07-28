"""Label-free, fail-closed split construction for UTR EditFlow benchmark v2.

Atomic components are defined *only* by source/candidate sequence states and
the complete sequence-state closure of every shortest primitive dynamic edit
execution order.  Study, scaffold, library, and other metadata are audited
separately under an explicit overlap policy; they never silently collapse the
complete benchmark into one component.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .near_neighbors import NearNeighborClusters
from .near_neighbors import NearNeighborError
from .near_neighbors import build_near_neighbor_clusters
from .path_states import ALGORITHM_ID as PATH_STATE_ALGORITHM_ID
from .path_states import MINIMUM_ALIGNMENT_COUNT_SCOPE
from .path_states import STATE_CLOSURE_SCOPE
from .path_states import STATE_PATH_COUNT_SCOPE
from .path_states import MinimumAlignmentStateClosure
from .path_states import PathStateError
from .path_states import minimum_alignment_state_closure


REGIONS = ("five_utr", "three_utr")
SPLIT_KINDS = ("source_disjoint", "study_disjoint", "cross_region_transfer")
ROLE_NAMES = ("train", "validation", "test")
ROLE_PAIRS = (
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
)

GROUPING_FIELD_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "source_group": ("source_group",),
    "study_group": ("study_group", "study_id"),
    "sequence_cluster": ("sequence_cluster",),
    "scaffold_group": ("scaffold_group", "reporter"),
    "gene_group": ("gene_group",),
    "context_group": ("context_group", "context_id"),
    "barcode_batch": ("barcode_batch",),
    "library_batch": ("library_batch",),
}
METADATA_DIMENSIONS = tuple(GROUPING_FIELD_ALIASES) + (
    "assay_group",
    "region_group",
)
SOURCE_AXIS_PARTITIONS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("source_disjoint", "source_state", None),
    (
        "sequence_cluster_disjoint",
        "sequence_cluster",
        "sequence_cluster",
    ),
    ("scaffold_disjoint", "scaffold_group", "scaffold_group"),
    ("gene_disjoint", "gene_group", "gene_group"),
    ("context_disjoint", "context_group", "context_group"),
    ("barcode_batch_disjoint", "barcode_batch", "barcode_batch"),
    ("library_batch_disjoint", "library_batch", "library_batch"),
)
_ALL_ROLE_PAIRS = tuple(ROLE_PAIRS)
_OUTER_TEST_ROLE_PAIRS = (
    ("train", "test"),
    ("validation", "test"),
)

_UNKNOWN_TOKENS = {
    "",
    "UNKNOWN",
    "MISSING",
    "NA",
    "N/A",
    "NONE",
    "NULL",
    "UNAVAILABLE",
    "UNRESOLVED",
    "TBD",
}
_NON_INTERVENTION_PAIR_TYPES = {
    "absolute_property_only",
    "unlabeled_pretraining",
}
_NON_PRIMARY_CANONICAL_SPLITS = {
    "absolute_prior_only",
    "retrospective_only",
}
_RETROSPECTIVE_DATASETS = {"GSE246381"}
_EXPECTED_LOSO_STUDIES = {
    "five_utr": ("GSE114002", "GSE217518"),
    "three_utr": ("GSE217518", "GSE200304"),
}
_NO_TRAINING_FLAGS = {
    "NO_TRAINING",
    "NO_TRAINING_OR_SELECTION",
    "RETROSPECTIVE_ONLY",
}


def expected_partition_ids(
    split_kind: str,
    *,
    region: Optional[str],
) -> Tuple[str, ...]:
    """Return the frozen, non-deletable partition set for one manifest."""

    if split_kind == "source_disjoint":
        if region not in REGIONS:
            raise SplitGraphError("source-disjoint manifest requires a region")
        return tuple(
            f"{partition_prefix}:{region}"
            for partition_prefix, _, _ in SOURCE_AXIS_PARTITIONS
        )
    if split_kind == "study_disjoint":
        if region not in REGIONS:
            raise SplitGraphError("study-disjoint manifest requires a region")
        return tuple(f"loso:{study}" for study in _EXPECTED_LOSO_STUDIES[region])
    if split_kind == "cross_region_transfer":
        return (
            "within_study:GSE217518:five_to_three",
            "within_study:GSE217518:three_to_five",
            "cross_study:GSE114002_five_to_GSE200304_three",
        )
    raise SplitGraphError(f"unknown split_kind: {split_kind}")


class SplitGraphError(ValueError):
    """Base class for deterministic split-graph contract failures."""


class MissingGroupingMetadataError(SplitGraphError):
    """A required grouping field is absent or scientifically unresolved."""


class DuplicateRecordIDError(SplitGraphError):
    """Canonical record identifiers must be unique."""


@dataclass(frozen=True)
class AtomicComponent:
    """One indivisible minimum-alignment state component."""

    component_id: str
    record_ids: Tuple[str, ...]
    sequence_nodes: Tuple[str, ...]
    regions: Tuple[str, ...]
    grouping_values: Mapping[str, Tuple[str, ...]]
    ambiguity_scope: str = MINIMUM_ALIGNMENT_COUNT_SCOPE


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _stable_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value)).hexdigest()


def canonical_sequence(value: Any, field: str = "sequence") -> str:
    """Return a canonical RNA identity without silently accepting emptiness."""

    if not isinstance(value, str):
        raise SplitGraphError(f"{field} must be a string")
    sequence = "".join(value.split()).upper().replace("T", "U")
    if not sequence:
        raise SplitGraphError(f"{field} must be non-empty")
    if any(base not in "ACGU" for base in sequence):
        raise SplitGraphError(f"{field} must use the canonical RNA alphabet A/C/G/U")
    return sequence


def _metadata_tokens(value: Any, field: str) -> Tuple[str, ...]:
    raw_values: Iterable[Any]
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = (value,)
    tokens: List[str] = []
    for raw in raw_values:
        if raw is None:
            raise MissingGroupingMetadataError(f"{field} is missing")
        token = str(raw).strip()
        upper = token.upper()
        if (
            upper in _UNKNOWN_TOKENS
            or upper.startswith("UNKNOWN:")
            or upper.startswith("MISSING:")
            or upper.startswith("UNRESOLVED:")
        ):
            raise MissingGroupingMetadataError(
                f"{field} is absent or unresolved: {token!r}"
            )
        if upper == "NOT_APPLICABLE":
            raise MissingGroupingMetadataError(
                f"{field}=NOT_APPLICABLE must include a scoped reason"
            )
        tokens.append(token)
    unique = tuple(sorted(set(tokens)))
    if not unique:
        raise MissingGroupingMetadataError(f"{field} has no values")
    return unique


def _nested_library_value(record: Mapping[str, Any], field: str) -> Any:
    library = record.get("library_design")
    if not isinstance(library, Mapping):
        return None
    aliases = {
        "barcode_batch": ("barcode_batch", "barcode_id", "barcode_group"),
        "library_batch": ("library_batch", "library_id", "batch_id"),
    }
    for alias in aliases.get(field, ()):
        if alias in library:
            return library[alias]
    return None


def grouping_values(record: Mapping[str, Any]) -> Dict[str, Tuple[str, ...]]:
    """Extract every contract-mandated grouping dimension or fail closed."""

    values: Dict[str, Tuple[str, ...]] = {}
    for canonical_field, aliases in GROUPING_FIELD_ALIASES.items():
        raw: Any = None
        found = False
        for alias in aliases:
            if alias in record:
                raw = record[alias]
                found = True
                break
        if not found and canonical_field in ("barcode_batch", "library_batch"):
            raw = _nested_library_value(record, canonical_field)
            found = raw is not None
        if not found:
            raise MissingGroupingMetadataError(
                f"{canonical_field} is required for fail-closed grouping"
            )
        values[canonical_field] = _metadata_tokens(raw, canonical_field)
    return values


def metadata_values(record: Mapping[str, Any]) -> Dict[str, Tuple[str, ...]]:
    """Return all axes that every leakage report must classify explicitly."""

    values = grouping_values(record)
    values["assay_group"] = _metadata_tokens(record.get("assay_id"), "assay_group")
    values["region_group"] = (_region(record),)
    return values


def _record_id(record: Mapping[str, Any]) -> str:
    value = str(record.get("record_id") or "").strip()
    if not value:
        raise SplitGraphError("record_id must be non-empty")
    return value


def _dataset_id(record: Mapping[str, Any]) -> str:
    value = str(record.get("dataset_id") or "").strip()
    if not value:
        raise SplitGraphError("dataset_id must be non-empty")
    return value


def _region(record: Mapping[str, Any]) -> str:
    value = str(record.get("region") or "").strip().lower()
    if value not in REGIONS:
        raise SplitGraphError(f"region must be one of {REGIONS}, got {value!r}")
    return value


def _candidate_store_fields() -> Tuple[str, ...]:
    # This lazy import binds structural identity to the exact D1 projection
    # without creating a module-import cycle.
    from .d1_builder import CANDIDATE_STORE_FIELDS

    return tuple(CANDIDATE_STORE_FIELDS)


def label_free_structural_projection(
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the exact D1 label-free candidate-store projection."""

    return {
        field: record[field] for field in _candidate_store_fields() if field in record
    }


def record_structural_sha256(record: Mapping[str, Any]) -> str:
    return _sha256_payload(label_free_structural_projection(record))


def record_ids_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Digest the complete sorted record identity universe."""

    record_ids = sorted(_record_id(record) for record in records)
    if len(record_ids) != len(set(record_ids)):
        raise DuplicateRecordIDError("record universe contains duplicate record_id")
    return hashlib.sha256(
        (("\n".join(record_ids) + "\n") if record_ids else "").encode("utf-8")
    ).hexdigest()


def record_universe_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Digest every D1 label-free structural record, independent of order."""

    signatures = sorted(
        (
            _record_id(record),
            record_structural_sha256(record),
        )
        for record in records
    )
    if len(signatures) != len({record_id for record_id, _ in signatures}):
        raise DuplicateRecordIDError("record universe contains duplicate record_id")
    return _sha256_payload(signatures)


def _declared_intermediate_sequences(
    record: Mapping[str, Any],
) -> Tuple[str, ...]:
    raw = record.get("intermediate_sequences", [])
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise SplitGraphError("intermediate_sequences must be a list")
    declared: List[str] = []
    trajectory_observed = record.get("trajectory_observed") is True
    for index, item in enumerate(raw):
        if isinstance(item, Mapping):
            if item.get("observed") is True and not trajectory_observed:
                raise SplitGraphError(
                    "constructed intermediate cannot be marked observed"
                )
            value = item.get("sequence")
        else:
            value = item
        declared.append(canonical_sequence(value, f"intermediate_sequences[{index}]"))
    return tuple(sorted(set(declared)))


def alignment_state_closure(
    record: Mapping[str, Any],
) -> MinimumAlignmentStateClosure:
    """Recompute exact shortest-action states while binding D1 ambiguity."""

    if record.get("trajectory_observed") is True:
        raise SplitGraphError(
            "B0 minimum-alignment closure cannot relabel an observed trajectory "
            "as a constructed path"
        )
    _declared_intermediate_sequences(record)
    source = canonical_sequence(record.get("source_sequence"), "source_sequence")
    candidate = canonical_sequence(
        record.get("candidate_sequence"), "candidate_sequence"
    )
    edit_distance = record.get("edit_distance")
    known_distance = (
        edit_distance
        if isinstance(edit_distance, int) and not isinstance(edit_distance, bool)
        else None
    )
    try:
        closure = minimum_alignment_state_closure(
            source,
            candidate,
            known_minimum_edit_count=known_distance,
        )
    except PathStateError as exc:
        raise SplitGraphError(str(exc)) from exc

    declared_ambiguity = record.get("edit_script_ambiguity")
    if isinstance(declared_ambiguity, Mapping):
        declared_scope = declared_ambiguity.get("count_scope")
        if (
            declared_scope is not None
            and declared_scope != MINIMUM_ALIGNMENT_COUNT_SCOPE
        ):
            raise SplitGraphError(
                "record ambiguity scope disagrees with the frozen D1 scope"
            )
        declared_count = declared_ambiguity.get("equivalent_minimal_script_count")
        if (
            declared_count is not None
            and declared_count != closure.minimum_alignment_count
        ):
            raise SplitGraphError(
                "record ambiguity count disagrees with recomputed alignment DAG"
            )
    if edit_distance is not None and edit_distance != closure.minimum_edit_count:
        raise SplitGraphError(
            "record edit_distance disagrees with minimum character alignment"
        )
    return closure


def intermediate_sequences(record: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return all constructed states inside any shortest dynamic edit path."""

    return alignment_state_closure(record).constructed_intermediate_states


def state_sequences(record: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return endpoints and every shortest dynamic-edit reachable state."""

    return alignment_state_closure(record).reachable_states


def _intrinsic_exclusion_reason(
    record: Mapping[str, Any],
) -> Optional[str]:
    dataset_id = _dataset_id(record)
    if dataset_id in _RETROSPECTIVE_DATASETS:
        return f"retrospective_dataset:{dataset_id}"
    pair_type = str(record.get("pair_type") or "")
    if pair_type in _NON_INTERVENTION_PAIR_TYPES:
        return "non_intervention_pair_type"
    canonical_split = str(record.get("canonical_split") or "")
    if canonical_split in _NON_PRIMARY_CANONICAL_SPLITS:
        return f"canonical_split:{canonical_split}"
    paper_split = str(record.get("paper_split") or "")
    if paper_split == "retrospective_only":
        return "paper_split:retrospective_only"
    quality_flags = {
        str(flag).upper()
        for flag in record.get("quality_flags", [])
        if isinstance(flag, str)
    }
    blocked_flags = sorted(quality_flags & _NO_TRAINING_FLAGS)
    if blocked_flags:
        return f"quality_flag:{blocked_flags[0]}"
    if record.get("source_id") is None or record.get("source_sequence") is None:
        return "missing_intervention_source"
    if record.get("candidate_sequence") is None:
        return "missing_candidate_sequence"
    if record.get("edit_script") is None:
        return "missing_intervention_edit_script"
    return None


def _accounting_row(
    record: Mapping[str, Any],
    *,
    reason: str,
) -> Dict[str, str]:
    return {
        "record_id": _record_id(record),
        "reason": reason,
        "structural_sha256": record_structural_sha256(record),
    }


def _select_records(
    records: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
    outside_reason: Callable[[Mapping[str, Any]], str],
) -> Tuple[List[Mapping[str, Any]], List[Dict[str, str]]]:
    selected: List[Mapping[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for record in records:
        record_id = _record_id(record)
        if record_id in seen:
            raise DuplicateRecordIDError(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        intrinsic = _intrinsic_exclusion_reason(record)
        if intrinsic is not None:
            excluded.append(_accounting_row(record, reason=intrinsic))
        elif predicate(record):
            selected.append(record)
        else:
            excluded.append(_accounting_row(record, reason=outside_reason(record)))
    return selected, sorted(excluded, key=lambda item: item["record_id"])


def select_split_eligible_records(
    records: Sequence[Mapping[str, Any]],
    *,
    regions: Sequence[str],
) -> Tuple[List[Mapping[str, Any]], List[Dict[str, str]]]:
    """Select intervention records and account for every other record."""

    requested = tuple(regions)
    if not requested or any(region not in REGIONS for region in requested):
        raise SplitGraphError(f"regions must be a non-empty subset of {REGIONS}")
    return _select_records(
        records,
        lambda record: _region(record) in requested,
        lambda record: f"outside_manifest_region:{_region(record)}",
    )


def global_near_neighbor_clusters(
    records: Sequence[Mapping[str, Any]],
) -> NearNeighborClusters:
    """Freeze exact edit-distance-5 clusters over the global eligible universe."""

    eligible, _ = select_split_eligible_records(records, regions=REGIONS)
    if not eligible:
        raise SplitGraphError(
            "global near-neighbor clustering has no split-eligible records"
        )
    record_states = {_record_id(record): state_sequences(record) for record in eligible}
    try:
        return build_near_neighbor_clusters(record_states)
    except NearNeighborError as exc:
        raise SplitGraphError(str(exc)) from exc


def build_atomic_components(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[AtomicComponent, ...]:
    """Build components from all shortest-action sequence states only."""

    normalized = list(records)
    if not normalized:
        return ()
    ids = [_record_id(record) for record in normalized]
    if len(ids) != len(set(ids)):
        raise DuplicateRecordIDError("duplicate record_id in component input")

    closures = [alignment_state_closure(record) for record in normalized]
    extracted_groups = [grouping_values(record) for record in normalized]
    uf = _UnionFind(len(normalized))
    first_by_sequence: Dict[str, int] = {}
    for index, closure in enumerate(closures):
        for sequence in closure.reachable_states:
            previous = first_by_sequence.get(sequence)
            if previous is None:
                first_by_sequence[sequence] = index
            else:
                uf.union(index, previous)

    member_indices: Dict[int, List[int]] = {}
    for index in range(len(normalized)):
        member_indices.setdefault(uf.find(index), []).append(index)

    components: List[AtomicComponent] = []
    for indices in member_indices.values():
        component_records = [normalized[index] for index in indices]
        component_record_ids = tuple(
            sorted(_record_id(record) for record in component_records)
        )
        sequences = tuple(
            sorted(
                {
                    sequence
                    for index in indices
                    for sequence in closures[index].reachable_states
                }
            )
        )
        regions = tuple(sorted({_region(record) for record in component_records}))
        combined_groups: Dict[str, Tuple[str, ...]] = {}
        for field in GROUPING_FIELD_ALIASES:
            combined_groups[field] = tuple(
                sorted(
                    {
                        token
                        for index in indices
                        for token in extracted_groups[index][field]
                    }
                )
            )
        component_digest = _sha256_payload(
            {
                "records": [
                    (
                        _record_id(record),
                        record_structural_sha256(record),
                        closures[index].reachable_states_sha256,
                        closures[index].minimum_alignment_count,
                        closures[index].minimum_state_path_count,
                    )
                    for index, record in zip(indices, component_records)
                ],
                "ambiguity_scope": MINIMUM_ALIGNMENT_COUNT_SCOPE,
                "state_closure_scope": STATE_CLOSURE_SCOPE,
                "state_path_count_scope": STATE_PATH_COUNT_SCOPE,
                "path_state_algorithm": PATH_STATE_ALGORITHM_ID,
            }
        )
        components.append(
            AtomicComponent(
                component_id="component:" + component_digest[:24],
                record_ids=component_record_ids,
                sequence_nodes=sequences,
                regions=regions,
                grouping_values=combined_groups,
            )
        )
    return tuple(sorted(components, key=lambda item: item.component_id))


def _reason_counts(
    excluded: Sequence[Mapping[str, str]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in excluded:
        reason = str(item["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _ambiguity_binding(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    closures = [alignment_state_closure(record) for record in records]
    return {
        "count_scope": MINIMUM_ALIGNMENT_COUNT_SCOPE,
        "state_closure_scope": STATE_CLOSURE_SCOPE,
        "state_path_count_scope": STATE_PATH_COUNT_SCOPE,
        "algorithm": PATH_STATE_ALGORITHM_ID,
        "records": len(closures),
        "ambiguous_records": sum(
            closure.minimum_alignment_count > 1 for closure in closures
        ),
        "max_equivalent_minimal_script_count": max(
            (closure.minimum_alignment_count for closure in closures),
            default=0,
        ),
        "max_minimum_state_path_count": max(
            (closure.minimum_state_path_count for closure in closures),
            default=0,
        ),
        "max_reachable_state_count": max(
            (closure.reachable_node_count for closure in closures),
            default=0,
        ),
        "max_reachable_transition_count": max(
            (closure.reachable_transition_count for closure in closures),
            default=0,
        ),
        "constructed_paths_marked_observed": 0,
        "path_state_closure_sha256": _sha256_payload(
            sorted(
                (
                    _record_id(record),
                    closure.minimum_alignment_count,
                    closure.minimum_state_path_count,
                    closure.reachable_transition_count,
                    closure.reachable_states_sha256,
                )
                for record, closure in zip(records, closures)
            )
        ),
    }


def _manifest_base(
    full_records: Sequence[Mapping[str, Any]],
    eligible: Sequence[Mapping[str, Any]],
    components: Sequence[AtomicComponent],
    near_neighbors: NearNeighborClusters,
    split_kind: str,
    region: Optional[str],
    excluded: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    eligible_rows = [
        _accounting_row(record, reason="eligible_intervention_record")
        for record in eligible
    ]
    if len(eligible_rows) + len(excluded) != len(full_records):
        raise AssertionError("eligible/excluded accounting is not exhaustive")
    excluded_ids = [{"record_id": str(item["record_id"])} for item in excluded]
    return {
        "schema_version": "utr_split_manifest.v2",
        "split_kind": split_kind,
        "region": region,
        "status": "READY",
        "blocked_reasons": [],
        "label_free_assignment": True,
        "candidate_store_contains_labels": False,
        "algorithm": {
            "name": "deterministic_all_shortest_action_component_partition_v3",
            "uses_randomness": False,
            "seed": None,
            "label_fields_read": [],
            "atomic_component_definition": (
                "source_candidate_and_all_shortest_primitive_dynamic_edit_states"
            ),
            "path_state_algorithm": PATH_STATE_ALGORITHM_ID,
            "state_closure_scope": STATE_CLOSURE_SCOPE,
            "constructed_states_claimed_observed": False,
        },
        "full_record_count": len(full_records),
        "full_record_ids_sha256": record_ids_sha256(full_records),
        "full_record_universe_sha256": record_universe_sha256(full_records),
        "record_count": len(eligible),
        "record_ids_sha256": record_ids_sha256(eligible),
        "component_count": len(components),
        "record_universe_sha256": record_universe_sha256(eligible),
        "eligible_records": sorted(eligible_rows, key=lambda item: item["record_id"]),
        "excluded_records": list(excluded),
        "eligible_record_accounting_sha256": _sha256_payload(
            sorted(eligible_rows, key=lambda item: item["record_id"])
        ),
        "excluded_record_count": len(excluded),
        "excluded_record_ids_sha256": record_ids_sha256(excluded_ids),
        "excluded_record_accounting_sha256": _sha256_payload(excluded),
        "exclusion_reason_counts": _reason_counts(excluded),
        "ambiguity_binding": _ambiguity_binding(eligible),
        "near_neighbor_binding": dict(near_neighbors.binding),
        "component_roles": {},
        "roles": {},
        "role_bindings": {},
    }


def _blocked_manifest(base: Dict[str, Any], reasons: Iterable[str]) -> Dict[str, Any]:
    base["status"] = "BLOCKED"
    base["blocked_reasons"] = sorted(set(str(reason) for reason in reasons))
    base["component_roles"] = {}
    base["roles"] = {}
    base["role_bindings"] = {}
    return base


def partition_sha256(partition: Mapping[str, Any]) -> str:
    """Hash the complete partition object, excluding only its self hash."""

    payload = {
        key: value for key, value in partition.items() if key != "partition_sha256"
    }
    return _sha256_payload(payload)


def _finalize_partition(partition: Dict[str, Any]) -> Dict[str, Any]:
    partition["partition_sha256"] = partition_sha256(partition)
    return partition


def _component_bundles(
    components: Sequence[AtomicComponent],
    grouping_field: str,
) -> Tuple[Tuple[AtomicComponent, ...], ...]:
    """Bundle atomic state components for a declared assignment dimension."""

    if grouping_field not in GROUPING_FIELD_ALIASES:
        raise SplitGraphError(f"unknown grouping field: {grouping_field}")
    uf = _UnionFind(len(components))
    first_by_token: Dict[str, int] = {}
    for index, component in enumerate(components):
        for token in component.grouping_values[grouping_field]:
            previous = first_by_token.get(token)
            if previous is None:
                first_by_token[token] = index
            else:
                uf.union(index, previous)
    indices_by_root: Dict[int, List[int]] = {}
    for index in range(len(components)):
        indices_by_root.setdefault(uf.find(index), []).append(index)
    bundles = [
        tuple(
            sorted(
                (components[index] for index in indices),
                key=lambda item: item.component_id,
            )
        )
        for indices in indices_by_root.values()
    ]
    return tuple(
        sorted(
            bundles,
            key=lambda bundle: tuple(component.component_id for component in bundle),
        )
    )


def _component_bundles_from_tokens(
    components: Sequence[AtomicComponent],
    tokens_by_component: Mapping[str, Sequence[str]],
) -> Tuple[Tuple[AtomicComponent, ...], ...]:
    """Union components by any predeclared label-free assignment token."""

    uf = _UnionFind(len(components))
    first_by_token: Dict[str, int] = {}
    for index, component in enumerate(components):
        tokens = tuple(tokens_by_component.get(component.component_id, ()))
        if not tokens:
            raise SplitGraphError(
                f"component {component.component_id} has no assignment tokens"
            )
        for token in tokens:
            previous = first_by_token.get(token)
            if previous is None:
                first_by_token[token] = index
            else:
                uf.union(index, previous)
    indices_by_root: Dict[int, List[int]] = {}
    for index in range(len(components)):
        indices_by_root.setdefault(uf.find(index), []).append(index)
    bundles = [
        tuple(
            sorted(
                (components[index] for index in indices),
                key=lambda item: item.component_id,
            )
        )
        for indices in indices_by_root.values()
    ]
    return tuple(
        sorted(
            bundles,
            key=lambda bundle: tuple(component.component_id for component in bundle),
        )
    )


def _component_firewall_tokens(
    component: AtomicComponent,
    near_neighbors: NearNeighborClusters,
) -> Tuple[str, ...]:
    """Return dense-graph firewall identities required in every split."""

    near_cluster_ids: set[str] = set()
    for record_id in component.record_ids:
        cluster_id = near_neighbors.record_clusters.get(record_id)
        if cluster_id is None:
            raise SplitGraphError(
                f"near-neighbor binding lacks eligible record {record_id}"
            )
        near_cluster_ids.add(cluster_id)
    return tuple(
        sorted(
            {
                *(f"near_neighbor:{value}" for value in near_cluster_ids),
                *(
                    f"source_group:{value}"
                    for value in component.grouping_values["source_group"]
                ),
                *(
                    f"scaffold_group:{value}"
                    for value in component.grouping_values["scaffold_group"]
                ),
            }
        )
    )


def _assignment_tokens(
    component: AtomicComponent,
    near_neighbors: NearNeighborClusters,
    axis_field: Optional[str],
) -> Tuple[str, ...]:
    tokens = set(_component_firewall_tokens(component, near_neighbors))
    if axis_field is not None:
        if axis_field not in GROUPING_FIELD_ALIASES:
            raise SplitGraphError(f"unknown assignment axis: {axis_field}")
        tokens.update(
            f"axis:{axis_field}:{value}"
            for value in component.grouping_values[axis_field]
        )
    return tuple(sorted(tokens))


def _assignment_bundles(
    components: Sequence[AtomicComponent],
    near_neighbors: NearNeighborClusters,
    *,
    axis_field: Optional[str],
) -> Tuple[Tuple[AtomicComponent, ...], ...]:
    tokens_by_component = {
        component.component_id: _assignment_tokens(
            component, near_neighbors, axis_field
        )
        for component in components
    }
    return _component_bundles_from_tokens(components, tokens_by_component)


def _firewall_overlap_tokens(
    left: Sequence[AtomicComponent],
    right: Sequence[AtomicComponent],
    near_neighbors: NearNeighborClusters,
) -> Set[str]:
    left_tokens = {
        token
        for component in left
        for token in _component_firewall_tokens(component, near_neighbors)
    }
    right_tokens = {
        token
        for component in right
        for token in _component_firewall_tokens(component, near_neighbors)
    }
    return left_tokens & right_tokens


def _cross_transfer_tokens(
    component: AtomicComponent,
    near_neighbors: NearNeighborClusters,
) -> Tuple[str, ...]:
    """Axes forbidden by the frozen cross-region estimand."""

    tokens = set(_component_firewall_tokens(component, near_neighbors))
    for field in (
        "sequence_cluster",
        "gene_group",
        "context_group",
        "barcode_batch",
        "library_batch",
    ):
        tokens.update(
            f"cross_strict:{field}:{value}"
            for value in component.grouping_values[field]
        )
    return tuple(sorted(tokens))


def _cross_transfer_bundles(
    components: Sequence[AtomicComponent],
    near_neighbors: NearNeighborClusters,
) -> Tuple[Tuple[AtomicComponent, ...], ...]:
    return _component_bundles_from_tokens(
        components,
        {
            component.component_id: _cross_transfer_tokens(component, near_neighbors)
            for component in components
        },
    )


def _cross_transfer_overlap_tokens(
    left: Sequence[AtomicComponent],
    right: Sequence[AtomicComponent],
    near_neighbors: NearNeighborClusters,
) -> Set[str]:
    left_tokens = {
        token
        for component in left
        for token in _cross_transfer_tokens(component, near_neighbors)
    }
    right_tokens = {
        token
        for component in right
        for token in _cross_transfer_tokens(component, near_neighbors)
    }
    return left_tokens & right_tokens


def _partition_bundles(
    bundles: Sequence[Sequence[AtomicComponent]],
    *,
    include_test: bool,
) -> Dict[str, Tuple[AtomicComponent, ...]]:
    minimum = 3 if include_test else 2
    if len(bundles) < minimum:
        raise SplitGraphError(
            f"at least {minimum} independent assignment bundles are required"
        )
    ordered = tuple(bundles)
    if include_test:
        n_test = max(1, int(round(len(ordered) * 0.15)))
        n_validation = max(1, int(round(len(ordered) * 0.15)))
        while n_test + n_validation >= len(ordered):
            if n_test >= n_validation and n_test > 1:
                n_test -= 1
            elif n_validation > 1:
                n_validation -= 1
            else:
                break
        bundle_roles = {
            "test": ordered[:n_test],
            "validation": ordered[n_test : n_test + n_validation],
            "train": ordered[n_test + n_validation :],
        }
    else:
        n_validation = max(1, int(round(len(ordered) * 0.15)))
        if n_validation >= len(ordered):
            n_validation = len(ordered) - 1
        bundle_roles = {
            "validation": ordered[:n_validation],
            "train": ordered[n_validation:],
            "test": (),
        }
    return {
        role: tuple(component for bundle in bundle_roles[role] for component in bundle)
        for role in ROLE_NAMES
    }


def _role_binding(
    record_ids: Sequence[str],
    record_index: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    selected = [record_index[record_id] for record_id in record_ids]
    return {
        "record_count": len(selected),
        "record_ids_sha256": record_ids_sha256(selected),
        "record_universe_sha256": record_universe_sha256(selected),
    }


def _materialize_roles(
    base: Dict[str, Any],
    component_roles: Mapping[str, Sequence[AtomicComponent]],
    record_index: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    roles: Dict[str, List[str]] = {}
    by_component: Dict[str, str] = {}
    for role in ROLE_NAMES:
        components = component_roles.get(role, ())
        roles[role] = sorted(
            record_id for component in components for record_id in component.record_ids
        )
        for component in components:
            if component.component_id in by_component:
                raise SplitGraphError("atomic component assigned to multiple roles")
            by_component[component.component_id] = role
    base["roles"] = roles
    base["component_roles"] = dict(sorted(by_component.items()))
    base["role_counts"] = {role: len(record_ids) for role, record_ids in roles.items()}
    base["role_bindings"] = {
        role: _role_binding(record_ids, record_index)
        for role, record_ids in roles.items()
    }
    return _finalize_partition(base)


def _policy_entry(
    *,
    allowed_pairs: Sequence[Tuple[str, str]],
    justification: str,
) -> Dict[str, Any]:
    canonical_pairs = [
        list(pair)
        for pair in ROLE_PAIRS
        if pair in set(tuple(value) for value in allowed_pairs)
    ]
    return {
        "allowed_role_pairs": canonical_pairs,
        "unlisted_role_pairs": "FORBIDDEN",
        "justification": justification,
    }


def _axis_disjoint_policy(axis_field: Optional[str]) -> Dict[str, Any]:
    strict_fields = {"source_group", "scaffold_group"}
    if axis_field is not None:
        strict_fields.add(axis_field)
    policy = {
        field: _policy_entry(
            allowed_pairs=(() if field in strict_fields else _ALL_ROLE_PAIRS),
            justification=(
                f"{field} is a required zero-overlap axis for this partition"
                if field in strict_fields
                else (
                    f"{field} is outside this independently reported axis "
                    "estimand; raw overlap remains disclosed and is tested "
                    "strictly in its own required partition when applicable"
                )
            ),
        )
        for field in METADATA_DIMENSIONS
    }
    return policy


def _study_disjoint_policy() -> Dict[str, Any]:
    policy = {
        field: _policy_entry(
            allowed_pairs=(
                () if field in {"source_group", "scaffold_group"} else _ALL_ROLE_PAIRS
            ),
            justification=(
                f"{field} is a dense-graph firewall across every role"
                if field in {"source_group", "scaffold_group"}
                else (
                    f"{field} is not the outer-LOSO estimand; raw overlap "
                    "remains disclosed"
                )
            ),
        )
        for field in METADATA_DIMENSIONS
    }
    policy["study_group"] = _policy_entry(
        allowed_pairs=(("train", "validation"),),
        justification=(
            "development roles may share study; outer test versus development "
            "is strictly study-disjoint"
        ),
    )
    return policy


def _cross_region_policy(*, within_study: bool) -> Dict[str, Any]:
    policy = {
        field: _policy_entry(
            allowed_pairs=(),
            justification=(
                f"{field} overlap is not required by the frozen transfer "
                "estimand and is therefore forbidden"
            ),
        )
        for field in METADATA_DIMENSIONS
    }
    policy["region_group"] = _policy_entry(
        allowed_pairs=(("train", "validation"),),
        justification=(
            "source-side development roles necessarily share the source "
            "region; target test uses the other frozen region"
        ),
    )
    policy["assay_group"] = _policy_entry(
        allowed_pairs=(_ALL_ROLE_PAIRS if within_study else (("train", "validation"),)),
        justification=(
            "the frozen within-study transfer stratum necessarily shares its "
            "study assay"
            if within_study
            else "source-side development roles share the source assay"
        ),
    )
    policy["study_group"] = _policy_entry(
        allowed_pairs=(_ALL_ROLE_PAIRS if within_study else (("train", "validation"),)),
        justification=(
            "within-study transfer necessarily shares study identity"
            if within_study
            else (
                "source-side development roles may share study; the target "
                "test study must remain distinct from development"
            )
        ),
    )
    return policy


def build_split_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    region: Optional[str],
    split_kind: str,
    source_region: str = "five_utr",
    target_region: str = "three_utr",
    _near_neighbors: Optional[NearNeighborClusters] = None,
) -> Dict[str, Any]:
    """Build a deterministic manifest, or preserve an explicit BLOCKED result."""

    if split_kind not in SPLIT_KINDS:
        raise SplitGraphError(
            f"split_kind must be one of {SPLIT_KINDS}, got {split_kind!r}"
        )
    full_records = list(records)
    record_ids_sha256(full_records)
    near_neighbors = _near_neighbors or global_near_neighbor_clusters(full_records)
    if split_kind == "cross_region_transfer":
        return _build_cross_region_manifest(
            full_records,
            near_neighbors=near_neighbors,
            source_region=source_region,
            target_region=target_region,
        )
    if region not in REGIONS:
        raise SplitGraphError(f"region must be one of {REGIONS}")
    if split_kind == "study_disjoint":
        return _build_study_disjoint_manifest(
            full_records,
            near_neighbors=near_neighbors,
            region=region,
        )
    return _build_source_disjoint_manifest(
        full_records,
        near_neighbors=near_neighbors,
        region=region,
    )


def _build_source_disjoint_manifest(
    full_records: Sequence[Mapping[str, Any]],
    *,
    near_neighbors: NearNeighborClusters,
    region: str,
) -> Dict[str, Any]:
    eligible, excluded = select_split_eligible_records(full_records, regions=(region,))
    components = build_atomic_components(eligible)
    outer = _manifest_base(
        full_records,
        eligible,
        components,
        near_neighbors,
        "source_disjoint",
        region,
        excluded,
    )
    partitions: List[Dict[str, Any]] = []
    failures: List[str] = []
    record_index = {_record_id(record): record for record in eligible}
    for partition_prefix, axis_name, axis_field in SOURCE_AXIS_PARTITIONS:
        partition = _manifest_base(
            full_records,
            eligible,
            components,
            near_neighbors,
            "source_disjoint",
            region,
            excluded,
        )
        partition_id = f"{partition_prefix}:{region}"
        required_axes = [
            "near_neighbor_cluster",
            "source_group",
            "scaffold_group",
        ]
        if axis_field is not None and axis_field not in required_axes:
            required_axes.append(axis_field)
        partition.update(
            {
                "partition_id": partition_id,
                "axis_partition": True,
                "independent_group_dimension": axis_name,
                "assignment_firewall_axes": [
                    "exact_shortest_path_state_component",
                    "near_neighbor_edit_distance_lte_5",
                    "source_group",
                    "scaffold_group",
                ],
                "required_disjoint_axes": required_axes,
                "required_disjoint_role_pairs": {
                    axis: [list(pair) for pair in _ALL_ROLE_PAIRS]
                    for axis in required_axes
                },
                "overlap_policy": _axis_disjoint_policy(axis_field),
            }
        )
        bundles = _assignment_bundles(
            components,
            near_neighbors,
            axis_field=axis_field,
        )
        partition["independent_group_count"] = len(bundles)
        if len(bundles) < 3:
            reason = f"{axis_name}_groups_insufficient"
            _finalize_partition(_blocked_manifest(partition, [reason]))
            failures.append(f"{partition_id}:{reason}")
        else:
            _materialize_roles(
                partition,
                _partition_bundles(bundles, include_test=True),
                record_index,
            )
        partitions.append(partition)

    required_partition_ids = [
        str(partition["partition_id"]) for partition in partitions
    ]
    outer.update(
        {
            "required_partition_ids": required_partition_ids,
            "partitions": partitions,
            "partition_count": len(partitions),
            "required_axis_partitions": [
                {
                    "partition_id": partition["partition_id"],
                    "axis": partition["independent_group_dimension"],
                }
                for partition in partitions
            ],
        }
    )
    outer["partitions_sha256"] = _sha256_payload(
        [
            {
                "partition_id": partition["partition_id"],
                "partition_sha256": partition["partition_sha256"],
            }
            for partition in partitions
        ]
    )
    if failures:
        outer["status"] = "BLOCKED"
        outer["blocked_reasons"] = sorted(failures)
    return outer


def _single_study(record: Mapping[str, Any]) -> str:
    studies = grouping_values(record)["study_group"]
    if len(studies) != 1:
        raise SplitGraphError("study-disjoint LOSO requires one study_group per record")
    return studies[0]


def _study_fold_contract_fields(
    heldout_study: str,
    claim_boundary: str,
) -> Dict[str, Any]:
    return {
        "fold_id": f"holdout:{heldout_study}",
        "partition_id": f"loso:{heldout_study}",
        "heldout_study": heldout_study,
        "independent_group_dimension": "study_group",
        "disjoint_scope": "test_vs_development",
        "assignment_firewall_axes": [
            "exact_shortest_path_state_component",
            "near_neighbor_edit_distance_lte_5",
            "source_group",
            "scaffold_group",
        ],
        "required_disjoint_axes": [
            "near_neighbor_cluster",
            "source_group",
            "scaffold_group",
            "study_group",
        ],
        "required_disjoint_role_pairs": {
            "near_neighbor_cluster": [list(pair) for pair in _ALL_ROLE_PAIRS],
            "source_group": [list(pair) for pair in _ALL_ROLE_PAIRS],
            "scaffold_group": [list(pair) for pair in _ALL_ROLE_PAIRS],
            "study_group": [list(pair) for pair in _OUTER_TEST_ROLE_PAIRS],
        },
        "selection_policy": {
            "assignment_label_independent": True,
            "outer_test_labels_read_for_selection": False,
            "validation_selected_from_development_only": True,
        },
        "claim_boundary": claim_boundary,
        "overlap_policy": _study_disjoint_policy(),
    }


def _build_study_disjoint_manifest(
    full_records: Sequence[Mapping[str, Any]],
    *,
    near_neighbors: NearNeighborClusters,
    region: str,
) -> Dict[str, Any]:
    eligible, excluded = select_split_eligible_records(full_records, regions=(region,))
    components = build_atomic_components(eligible)
    base = _manifest_base(
        full_records,
        eligible,
        components,
        near_neighbors,
        "study_disjoint",
        region,
        excluded,
    )
    observed_studies = {_single_study(record) for record in eligible}
    studies = list(_EXPECTED_LOSO_STUDIES[region])
    base.update(
        {
            "independent_group_dimension": "study_group",
            "independent_group_count": len(observed_studies),
            "disjoint_scope": "test_vs_development",
            "selection_policy": {
                "assignment_label_independent": True,
                "outer_test_labels_read_for_selection": False,
                "reported_aggregation": "per_fold_and_study_macro_only",
            },
            "claim_boundary": (
                "n=2 studies is descriptive; no population-level "
                "cross-study generalization claim"
            ),
            "folds": [],
        }
    )
    if observed_studies != set(studies):
        reasons = []
        missing = sorted(set(studies) - observed_studies)
        unexpected = sorted(observed_studies - set(studies))
        if missing:
            reasons.append("required_LOSO_studies_missing:" + ",".join(missing))
        if unexpected:
            reasons.append("unexpected_LOSO_studies_present:" + ",".join(unexpected))
        for heldout_study in studies:
            fold = _manifest_base(
                full_records,
                eligible,
                components,
                near_neighbors,
                "study_disjoint",
                region,
                excluded,
            )
            fold.update(
                _study_fold_contract_fields(heldout_study, base["claim_boundary"])
            )
            fold_reasons = ["frozen_study_universe_mismatch"]
            if heldout_study not in observed_studies:
                fold_reasons.append("heldout_study_missing")
            _finalize_partition(_blocked_manifest(fold, fold_reasons))
            base["folds"].append(fold)
        base["fold_count"] = len(base["folds"])
        base["required_partition_ids"] = [f"loso:{study}" for study in studies]
        base["partitions"] = base["folds"]
        base["folds_sha256"] = _sha256_payload(
            [
                {
                    "fold_id": fold["fold_id"],
                    "status": fold["status"],
                    "partition_sha256": fold["partition_sha256"],
                    "blocked_reasons": fold["blocked_reasons"],
                }
                for fold in base["folds"]
            ]
        )
        return _blocked_manifest(base, reasons)

    record_index = {_record_id(record): record for record in eligible}
    failures: List[str] = []
    for heldout_study in studies:
        fold = _manifest_base(
            full_records,
            eligible,
            components,
            near_neighbors,
            "study_disjoint",
            region,
            excluded,
        )
        fold.update(_study_fold_contract_fields(heldout_study, base["claim_boundary"]))
        test_components: List[AtomicComponent] = []
        development_components: List[AtomicComponent] = []
        fold_reasons: List[str] = []
        for component in components:
            component_studies = {
                _single_study(record_index[record_id])
                for record_id in component.record_ids
            }
            if heldout_study in component_studies:
                if component_studies != {heldout_study}:
                    fold_reasons.append(
                        "state_component_spans_test_and_development_studies"
                    )
                test_components.append(component)
            else:
                development_components.append(component)
        firewall_overlap = _firewall_overlap_tokens(
            test_components,
            development_components,
            near_neighbors,
        )
        if firewall_overlap:
            fold_reasons.append("dense_graph_firewall_spans_test_and_development")
        if not test_components:
            fold_reasons.append("heldout_study_has_no_test_records")
        development_bundles = _assignment_bundles(
            development_components,
            near_neighbors,
            axis_field=None,
        )
        if len(development_bundles) < 2:
            fold_reasons.append(
                "development_source_components_insufficient_for_train_validation"
            )
        if fold_reasons:
            _blocked_manifest(fold, fold_reasons)
            _finalize_partition(fold)
            failures.extend(f"{fold['fold_id']}:{reason}" for reason in fold_reasons)
        else:
            development_roles = _partition_bundles(
                development_bundles, include_test=False
            )
            development_roles["test"] = tuple(test_components)
            _materialize_roles(fold, development_roles, record_index)
        base["folds"].append(fold)

    base["fold_count"] = len(base["folds"])
    base["required_partition_ids"] = [f"loso:{study}" for study in studies]
    base["partitions"] = base["folds"]
    base["folds_sha256"] = _sha256_payload(
        [
            {
                "fold_id": fold["fold_id"],
                "status": fold["status"],
                "partition_sha256": fold.get("partition_sha256"),
                "blocked_reasons": fold["blocked_reasons"],
            }
            for fold in base["folds"]
        ]
    )
    if failures:
        return _blocked_manifest(base, failures)
    return base


_CROSS_REGION_STRATA = (
    {
        "stratum_id": "GSE217518_within_study_5to3",
        "partition_id": "within_study:GSE217518:five_to_three",
        "source_dataset": "GSE217518",
        "source_region": "five_utr",
        "target_dataset": "GSE217518",
        "target_region": "three_utr",
        "within_study": True,
    },
    {
        "stratum_id": "GSE217518_within_study_3to5",
        "partition_id": "within_study:GSE217518:three_to_five",
        "source_dataset": "GSE217518",
        "source_region": "three_utr",
        "target_dataset": "GSE217518",
        "target_region": "five_utr",
        "within_study": True,
    },
    {
        "stratum_id": "GSE114002_5to_GSE200304_3_cross_study",
        "partition_id": ("cross_study:GSE114002_five_to_GSE200304_three"),
        "source_dataset": "GSE114002",
        "source_region": "five_utr",
        "target_dataset": "GSE200304",
        "target_region": "three_utr",
        "within_study": False,
    },
)


def expected_partition_leakage_contract(
    split_kind: str,
    *,
    region: Optional[str],
    partition_id: str,
) -> Dict[str, Any]:
    """Return the frozen leakage axes, role pairs, and overlap policy."""

    expected_ids = expected_partition_ids(split_kind, region=region)
    if partition_id not in expected_ids:
        raise SplitGraphError(
            f"partition {partition_id!r} is not frozen for {split_kind}:{region}"
        )
    all_role_pairs = [list(pair) for pair in _ALL_ROLE_PAIRS]
    if split_kind == "source_disjoint":
        prefix = partition_id.split(":", 1)[0]
        try:
            _, axis_name, axis_field = next(
                row for row in SOURCE_AXIS_PARTITIONS if row[0] == prefix
            )
        except StopIteration as exc:
            raise SplitGraphError(
                f"unknown frozen source-axis partition: {partition_id}"
            ) from exc
        required_axes = [
            "near_neighbor_cluster",
            "source_group",
            "scaffold_group",
        ]
        if axis_field is not None and axis_field not in required_axes:
            required_axes.append(axis_field)
        return {
            "axis_partition": True,
            "independent_group_dimension": axis_name,
            "required_disjoint_axes": required_axes,
            "required_disjoint_role_pairs": {
                axis: list(all_role_pairs) for axis in required_axes
            },
            "overlap_policy": _axis_disjoint_policy(axis_field),
        }
    if split_kind == "study_disjoint":
        required_axes = [
            "near_neighbor_cluster",
            "source_group",
            "scaffold_group",
            "study_group",
        ]
        return {
            "independent_group_dimension": "study_group",
            "required_disjoint_axes": required_axes,
            "required_disjoint_role_pairs": {
                "near_neighbor_cluster": list(all_role_pairs),
                "source_group": list(all_role_pairs),
                "scaffold_group": list(all_role_pairs),
                "study_group": [list(pair) for pair in _OUTER_TEST_ROLE_PAIRS],
            },
            "overlap_policy": _study_disjoint_policy(),
        }
    if split_kind == "cross_region_transfer":
        spec = next(
            (
                item
                for item in _CROSS_REGION_STRATA
                if item["partition_id"] == partition_id
            ),
            None,
        )
        if spec is None:
            raise SplitGraphError(
                f"unknown frozen cross-region partition: {partition_id}"
            )
        within_study = bool(spec["within_study"])
        required_axes = [
            "near_neighbor_cluster",
            "source_group",
            "scaffold_group",
        ] + ([] if within_study else ["study_group"])
        required_pairs = {
            "near_neighbor_cluster": list(all_role_pairs),
            "source_group": list(all_role_pairs),
            "scaffold_group": list(all_role_pairs),
        }
        if not within_study:
            required_pairs["study_group"] = [
                list(pair) for pair in _OUTER_TEST_ROLE_PAIRS
            ]
        return {
            "required_disjoint_axes": required_axes,
            "required_disjoint_role_pairs": required_pairs,
            "overlap_policy": _cross_region_policy(within_study=within_study),
        }
    raise SplitGraphError(f"unknown split_kind: {split_kind}")


D1_SPLIT_BINDING_FIELDS = frozenset(
    {
        "d1_acceptance_path",
        "d1_acceptance_sha256",
        "d1_phase_gate_passed",
        "canonical_records_path",
        "canonical_records_sha256",
        "canonical_record_count",
        "canonical_record_ids_sha256",
        "structural_records_path",
        "structural_records_sha256",
        "structural_records_bytes",
        "structural_record_count",
        "structural_record_ids_sha256",
        "structural_content_sha256",
        "d1_ambiguity_report_path",
        "d1_ambiguity_report_sha256",
        "ambiguity_count_scope",
        "fresh_projection_comparison",
        "canonical_validation_report_path",
        "canonical_validation_report_sha256",
    }
)
_POST_BINDING_HASH_FIELDS = frozenset(
    {
        "_artifact_sha256",
        "partitions_sha256",
        "folds_sha256",
        "strata_sha256",
    }
)


def canonical_split_manifest_core(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Remove only audited D1 overlay/self-hash fields from a split manifest."""

    def normalized_partition(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            key: child
            for key, child in value.items()
            if key not in D1_SPLIT_BINDING_FIELDS and key != "partition_sha256"
        }

    return {
        key: (
            [normalized_partition(item) for item in value]
            if key in {"partitions", "folds", "strata"} and isinstance(value, list)
            else value
        )
        for key, value in manifest.items()
        if key not in D1_SPLIT_BINDING_FIELDS and key not in _POST_BINDING_HASH_FIELDS
    }


def _select_cross_region_stratum(
    full_records: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> Tuple[List[Mapping[str, Any]], List[Dict[str, str]]]:
    def matches(record: Mapping[str, Any]) -> bool:
        identity = (_dataset_id(record), _region(record))
        return identity in {
            (spec["source_dataset"], spec["source_region"]),
            (spec["target_dataset"], spec["target_region"]),
        }

    return _select_records(
        full_records,
        matches,
        lambda record: (
            f"outside_stratum_scope:{spec['stratum_id']}:"
            f"{_dataset_id(record)}:{_region(record)}"
        ),
    )


def _build_cross_region_stratum(
    full_records: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    near_neighbors: NearNeighborClusters,
) -> Dict[str, Any]:
    eligible, excluded = _select_cross_region_stratum(full_records, spec)
    components = build_atomic_components(eligible)
    stratum = _manifest_base(
        full_records,
        eligible,
        components,
        near_neighbors,
        "cross_region_transfer",
        None,
        excluded,
    )
    stratum.update(
        {
            **dict(spec),
            "selection_policy": {
                "assignment_label_independent": True,
                "outer_test_labels_read_for_selection": False,
                "validation_selected_from_source_side_only": True,
            },
            "overlap_policy": _cross_region_policy(
                within_study=bool(spec["within_study"])
            ),
            "assignment_firewall_axes": [
                "exact_shortest_path_state_component",
                "near_neighbor_edit_distance_lte_5",
                "source_group",
                "scaffold_group",
            ],
            "required_disjoint_axes": (
                [
                    "near_neighbor_cluster",
                    "source_group",
                    "scaffold_group",
                ]
                + ([] if spec["within_study"] else ["study_group"])
            ),
            "required_disjoint_role_pairs": {
                "near_neighbor_cluster": [list(pair) for pair in _ALL_ROLE_PAIRS],
                "source_group": [list(pair) for pair in _ALL_ROLE_PAIRS],
                "scaffold_group": [list(pair) for pair in _ALL_ROLE_PAIRS],
                **(
                    {}
                    if spec["within_study"]
                    else {
                        "study_group": [list(pair) for pair in _OUTER_TEST_ROLE_PAIRS]
                    }
                ),
            },
            "required_metrics": (
                ["endpoint_agnostic_generative_metrics"]
                if not spec["within_study"]
                else ["generative_metrics", "endpoint_metrics_by_stratum"]
            ),
            "estimand_required_overlap_axes": (
                ["study_group", "assay_group", "source_side_region_group"]
                if spec["within_study"]
                else [
                    "source_side_study_group",
                    "source_side_assay_group",
                    "source_side_region_group",
                ]
            ),
            "claim_boundary": (
                "study, assay, and region are confounded; report only "
                "endpoint-agnostic generative transfer and never an isolated "
                "region effect"
                if not spec["within_study"]
                else "within-study transfer; study overlap is predeclared"
            ),
            "confounding_disclosure": (
                ["study", "assay", "region"]
                if not spec["within_study"]
                else ["study", "assay", "region"]
            ),
            "aggregation_policy": (
                "report_stratum_separately; do_not_pool_as_region_effect"
            ),
        }
    )
    record_index = {_record_id(record): record for record in eligible}
    source_components: List[AtomicComponent] = []
    target_components: List[AtomicComponent] = []
    reasons: List[str] = []
    for component in components:
        sides = {
            (
                _dataset_id(record_index[record_id]),
                _region(record_index[record_id]),
            )
            for record_id in component.record_ids
        }
        source_identity = (
            spec["source_dataset"],
            spec["source_region"],
        )
        target_identity = (
            spec["target_dataset"],
            spec["target_region"],
        )
        if source_identity in sides and target_identity in sides:
            reasons.append("state_component_spans_source_and_target")
        if sides == {source_identity}:
            source_components.append(component)
        elif sides == {target_identity}:
            target_components.append(component)
        elif sides and not (source_identity in sides and target_identity in sides):
            reasons.append("component_has_unexpected_dataset_region_identity")

    firewall_overlap = _cross_transfer_overlap_tokens(
        source_components,
        target_components,
        near_neighbors,
    )
    if firewall_overlap:
        reasons.append("dense_graph_firewall_spans_source_and_target")
    source_bundles = _cross_transfer_bundles(
        source_components,
        near_neighbors,
    )
    if len(source_bundles) < 2:
        reasons.append("source_components_insufficient_for_train_validation")
    if not target_components:
        reasons.append("target_components_absent")
    if reasons:
        return _finalize_partition(_blocked_manifest(stratum, reasons))

    roles = _partition_bundles(source_bundles, include_test=False)
    roles["test"] = tuple(target_components)
    return _materialize_roles(stratum, roles, record_index)


def _build_cross_region_manifest(
    full_records: Sequence[Mapping[str, Any]],
    *,
    near_neighbors: NearNeighborClusters,
    source_region: str,
    target_region: str,
) -> Dict[str, Any]:
    if (source_region, target_region) != ("five_utr", "three_utr"):
        raise SplitGraphError(
            "the frozen cross-region manifest identity is five_utr -> three_utr; "
            "the required reverse direction is a named stratum in the same file"
        )
    eligible, excluded = select_split_eligible_records(full_records, regions=REGIONS)
    components = build_atomic_components(eligible)
    base = _manifest_base(
        full_records,
        eligible,
        components,
        near_neighbors,
        "cross_region_transfer",
        None,
        excluded,
    )
    base.update(
        {
            "source_region": source_region,
            "target_region": target_region,
            "required_stratum_ids": [
                spec["stratum_id"] for spec in _CROSS_REGION_STRATA
            ],
            "required_partition_ids": [
                spec["partition_id"] for spec in _CROSS_REGION_STRATA
            ],
            "strata": [],
            "aggregation_policy": (
                "all_required_strata_reported_separately; "
                "no_isolated_region_effect_claim"
            ),
        }
    )
    failures: List[str] = []
    for spec in _CROSS_REGION_STRATA:
        stratum = _build_cross_region_stratum(full_records, spec, near_neighbors)
        base["strata"].append(stratum)
        if stratum["status"] != "READY":
            failures.extend(
                f"{spec['stratum_id']}:{reason}"
                for reason in stratum["blocked_reasons"]
            )
    base["stratum_count"] = len(base["strata"])
    base["partitions"] = base["strata"]
    base["strata_sha256"] = _sha256_payload(
        [
            {
                "stratum_id": stratum["stratum_id"],
                "status": stratum["status"],
                "partition_sha256": stratum.get("partition_sha256"),
                "blocked_reasons": stratum["blocked_reasons"],
            }
            for stratum in base["strata"]
        ]
    )
    if failures:
        return _blocked_manifest(base, failures)
    return base


__all__ = [
    "GROUPING_FIELD_ALIASES",
    "METADATA_DIMENSIONS",
    "REGIONS",
    "ROLE_NAMES",
    "ROLE_PAIRS",
    "SOURCE_AXIS_PARTITIONS",
    "SPLIT_KINDS",
    "AtomicComponent",
    "DuplicateRecordIDError",
    "D1_SPLIT_BINDING_FIELDS",
    "MissingGroupingMetadataError",
    "SplitGraphError",
    "alignment_state_closure",
    "build_atomic_components",
    "build_split_manifest",
    "canonical_split_manifest_core",
    "canonical_sequence",
    "expected_partition_leakage_contract",
    "expected_partition_ids",
    "global_near_neighbor_clusters",
    "grouping_values",
    "intermediate_sequences",
    "label_free_structural_projection",
    "metadata_values",
    "record_ids_sha256",
    "record_structural_sha256",
    "record_universe_sha256",
    "partition_sha256",
    "select_split_eligible_records",
    "state_sequences",
]
