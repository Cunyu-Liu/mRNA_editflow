"""Label-sealed loader and complete role audit for B0 Track A/B/C.

The ordinary loader deliberately never resolves or opens the Track A label
store or its freeze proof.  The acceptance command must call the separate
privileged verifier after candidate selection is frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from .records import MEASURED_PAIR_TYPES


TRACK_TYPES = (
    "closed_measured_pool",
    "heldout_generative",
    "open_legal_generation",
)
OPEN_WORLD_EVALUATION_BUDGETS = (1, 3, 5)
OPEN_WORLD_MAX_EDIT_BUDGET = max(OPEN_WORLD_EVALUATION_BUDGETS)
_OPEN_WORLD_BUDGET_PROTOCOL = {
    "required_budgets": list(OPEN_WORLD_EVALUATION_BUDGETS),
    "task_representation": "single_maximum_budget",
    "maximum_budget": OPEN_WORLD_MAX_EDIT_BUDGET,
    "report_each_budget_separately": True,
    "silent_budget_reduction_forbidden": True,
}
_GENERATION_TASK_FIELDS = {
    "schema_version",
    "task_id",
    "track_id",
    "region",
    "source_id",
    "source_sequence",
    "endpoint",
    "candidate_id",
    "candidate_sequence",
    "legal_action_types",
    "max_edits",
    "constraints",
    "provenance",
}
_CONSTRAINT_FIELDS = {
    "source_conditioned",
    "sequence_alphabet",
    "allowed_operations",
    "min_length",
    "max_length",
    "forbidden_motifs",
}
_PROVENANCE_FIELDS = {"dataset_id", "study_id", "record_id"}
_RNA_ALPHABET = frozenset("ACGU")
_ACTION_TYPES = frozenset({"INS", "SUB", "DEL", "STOP"})
_TRACK_MANIFEST_BASE_FIELDS = {
    "schema_version",
    "track_id",
    "track_type",
    "candidate_store",
    "candidate_store_contains_labels",
    "selection_access",
    "retrospective_external_stress_datasets",
    "universe_binding",
}
_UNIVERSE_BINDING_FIELDS = {
    "canonical_records_sha256",
    "structural_records_sha256",
    "record_ids_sha256",
    "record_count",
    "candidate_ids_sha256",
    "candidate_count",
    "task_ids_sha256",
    "task_count",
    "source_ids_sha256",
    "source_count",
}
_LABEL_STORE_FIELDS = {
    "path",
    "sha256",
    "bytes",
    "access",
    "candidate_id_field",
    "candidate_ids_sha256",
    "candidate_count",
    "schema",
    "selection_freeze",
    "freeze_proof",
}
_FREEZE_PROOF_REF_FIELDS = {"path", "sha256"}
_SEALED_FILE_REF_FIELDS = {"path", "sha256", "bytes"}
_FREEZE_PROOF_FIELDS = {
    "schema_version",
    "track_id",
    "candidate_store_sha256",
    "candidate_store_bytes",
    "label_store_sha256",
    "label_store_bytes",
    "candidate_ids_sha256",
    "candidate_count",
    "label_candidate_ids_sha256",
    "label_count",
    "label_record_ids_sha256",
    "label_record_count",
    "selection_freeze_sha256",
    "selection_freeze_bytes",
    "role_policy_sha256",
    "hidden_label_schema_sha256",
    "canonical_records_sha256",
    "structural_records_sha256",
    "record_ids_sha256",
    "frozen_before_label_access",
    "selection_labels_hidden",
}
_RETROSPECTIVE_DATASET = "GSE246381"
_RETROSPECTIVE_ROLE = "historically_exposed_retrospective_external_stress_test"
_SELECTION_FREEZE_FIELDS = {
    "schema_version",
    "track_id",
    "candidate_store_path",
    "candidate_store_sha256",
    "candidate_store_bytes",
    "selected_record_ids_sha256",
    "selected_record_count",
    "selected_task_ids_sha256",
    "selected_task_count",
    "structural_records_sha256",
    "role_policy_path",
    "role_policy_sha256",
    "role_policy_bytes",
    "hidden_label_schema_path",
    "hidden_label_schema_sha256",
    "hidden_label_schema_bytes",
    "source_disjoint_partition_bindings",
    "d1_acceptance",
    "d1_build_manifest",
    "canonical_records_declared",
    "structural_records",
    "canonical_label_store_opened",
    "selection_labels_hidden",
    "frozen_before_label_access",
}
_CANONICAL_DECLARED_FIELDS = {
    "path",
    "sha256",
    "bytes",
    "record_count",
    "record_ids_sha256",
}
_STRUCTURAL_DECLARED_FIELDS = _CANONICAL_DECLARED_FIELDS | {"structural_content_sha256"}
_HIDDEN_LABEL_FIELDS = {
    "schema_version",
    "candidate_id",
    "canonical_candidate_id",
    "record_id",
    "dataset_id",
    "study_id",
    "assay_id",
    "context_id",
    "endpoint",
    "pair_type",
    "source_id",
    "source_sequence_sha256",
    "candidate_sequence_sha256",
    "source_value_raw",
    "candidate_value_raw",
    "delta_raw",
    "delta_normalized",
    "effect_standard_error",
    "replicate_count",
    "label_provenance",
    "canonical_record_sha256",
    "measurement_evidence",
}

_FORBIDDEN_EXACT = {
    "label",
    "labels",
    "target",
    "targets",
    "outcome",
    "outcomes",
    "fitness",
    "measurement",
    "measurements",
    "score",
    "scores",
    "source_value_raw",
    "candidate_value_raw",
    "delta_raw",
    "delta_normalized",
    "effect_standard_error",
    "replicate_count",
    "assay_result",
    "response",
    "value",
    "values",
    "effect",
    "effects",
}


class TrackContractError(ValueError):
    """A track manifest violates label sealing or role semantics."""


class CandidateStoreLabelError(TrackContractError):
    """The physical candidate store contains a forbidden label/value field."""


class CandidateStoreHashError(TrackContractError):
    """The candidate store differs from its frozen manifest checksum."""


@dataclass(frozen=True)
class LoadedTrack:
    manifest_path: str
    manifest_sha256: str
    track_id: str
    track_type: str
    candidate_store_path: str
    candidate_store_sha256: str
    candidate_store_bytes: int
    candidate_ids: Tuple[str, ...]
    task_ids: Tuple[str, ...]
    record_ids: Tuple[str, ...]
    source_ids: Tuple[str, ...]
    dataset_ids: Tuple[str, ...]
    candidate_count: int
    task_count: int
    record_count: int
    source_count: int
    label_store_access: str
    label_store_sha256: str | None
    label_store_bytes: int | None
    label_freeze_proof_sha256: str | None
    label_schema_sha256: str | None
    selection_freeze_sha256: str | None
    retrospective_external_stress_datasets: Tuple[str, ...]
    universe_binding: Mapping[str, Any]
    tasks: Tuple[Mapping[str, Any], ...]
    evaluation_budget_protocol: Mapping[str, Any] | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(values: Sequence[str]) -> str:
    normalized = sorted(values)
    if len(normalized) != len(set(normalized)):
        raise TrackContractError("identity universe contains duplicate IDs")
    body = ("\n".join(normalized) + "\n") if normalized else ""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _stable_payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def expected_generation_task(
    record: Mapping[str, Any],
    *,
    track_type: str,
) -> Dict[str, Any]:
    """Project one frozen structural record into its exact B0 task."""

    if track_type not in TRACK_TYPES:
        raise TrackContractError(f"unknown track type: {track_type}")
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
        raise TrackContractError(
            f"structural record lacks task fields: {record.get('record_id')}"
        )
    record_id = str(record["record_id"])
    source = str(record["source_sequence"])
    if track_type == "open_legal_generation":
        candidate_id = None
        candidate_sequence = None
        actions = ["INS", "SUB", "DEL", "STOP"]
        max_edits = OPEN_WORLD_MAX_EDIT_BUDGET
        minimum = max(1, len(source) - OPEN_WORLD_MAX_EDIT_BUDGET)
        maximum = len(source) + OPEN_WORLD_MAX_EDIT_BUDGET
    else:
        candidate_sequence = str(record["candidate_sequence"])
        candidate_digest = _stable_payload_sha256(
            {
                "record_id": record_id,
                "d1_candidate_id": record["candidate_id"],
                "candidate_sequence": candidate_sequence,
            }
        )
        candidate_id = f"utr-editbench-v2:candidate:{candidate_digest[:24]}"
        raw_actions = record.get("edit_types")
        if not isinstance(raw_actions, list) or any(
            action not in {"INS", "SUB", "DEL"} for action in raw_actions
        ):
            raise TrackContractError(f"record has invalid edit_types: {record_id}")
        actions = [
            action
            for action in ("INS", "SUB", "DEL", "STOP")
            if action in set(raw_actions) or action == "STOP"
        ]
        if actions == ["STOP"]:
            raise TrackContractError(
                f"eligible intervention has no edit action: {record_id}"
            )
        valid_counts = [
            value
            for value in (
                record.get("edit_count"),
                record.get("edit_distance"),
            )
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ]
        if not valid_counts:
            raise TrackContractError(f"record lacks a valid edit budget: {record_id}")
        max_edits = max(valid_counts)
        minimum = min(len(source), len(candidate_sequence))
        maximum = max(len(source), len(candidate_sequence))
    task_digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return {
        "schema_version": "generation_task.v2",
        "task_id": f"utr-editbench-v2:{track_type}:{task_digest[:24]}",
        "track_id": track_type,
        "region": record["region"],
        "source_id": record["source_id"],
        "source_sequence": source,
        "endpoint": record["endpoint"],
        "candidate_id": candidate_id,
        "candidate_sequence": candidate_sequence,
        "legal_action_types": actions,
        "max_edits": max_edits,
        "constraints": {
            "source_conditioned": True,
            "sequence_alphabet": "RNA",
            "allowed_operations": list(actions),
            "min_length": minimum,
            "max_length": maximum,
        },
        "provenance": {
            "dataset_id": record["dataset_id"],
            "study_id": record["study_id"],
            "record_id": record_id,
        },
    }


def _forbidden_key(key: str) -> bool:
    normalized = key.strip().lower()
    return (
        normalized in _FORBIDDEN_EXACT
        or "label" in normalized
        or "delta" in normalized
        or normalized.endswith("_value")
        or normalized.startswith("value_")
        or normalized.startswith("effect_")
        or normalized.endswith("_score")
    )


def _scan_label_fields(value: Any, path: str) -> List[str]:
    violations: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            text_key = str(key)
            child_path = f"{path}.{text_key}" if path else text_key
            if _forbidden_key(text_key):
                violations.append(child_path)
            violations.extend(_scan_label_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(_scan_label_fields(child, f"{path}[{index}]"))
    return violations


def assert_candidate_store_label_free(
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Recursively reject any physical label, endpoint score, or value field."""

    violations: List[str] = []
    for index, record in enumerate(records):
        violations.extend(_scan_label_fields(record, f"record[{index}]"))
    if violations:
        raise CandidateStoreLabelError(
            "candidate store contains forbidden label/value field: "
            + sorted(violations)[0]
        )


def _validate_rna(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.upper()
        or any(base not in _RNA_ALPHABET for base in value)
    ):
        raise TrackContractError(
            f"{field} must be a non-empty uppercase canonical RNA sequence"
        )


def _bounded_allowed_edit_distance(
    left: str,
    right: str,
    limit: int,
    allowed_operations: Sequence[str],
) -> int:
    """Return bounded distance using only explicitly allowed edit operations."""

    if abs(len(left) - len(right)) > limit:
        return limit + 1
    sentinel = limit + 1
    allowed = set(allowed_operations)
    previous: Dict[int, int] = {0: 0}
    if "INS" in allowed:
        previous.update(
            {column: column for column in range(1, min(len(right), limit) + 1)}
        )
    for row, left_base in enumerate(left, start=1):
        start = max(0, row - limit)
        end = min(len(right), row + limit)
        current: Dict[int, int] = {}
        if start == 0 and "DEL" in allowed and row <= limit:
            current[0] = row
        for column in range(max(1, start), end + 1):
            right_base = right[column - 1]
            costs: List[int] = []
            diagonal = previous.get(column - 1, sentinel)
            if left_base == right_base:
                costs.append(diagonal)
            elif "SUB" in allowed:
                costs.append(diagonal + 1)
            if "DEL" in allowed:
                costs.append(previous.get(column, sentinel) + 1)
            if "INS" in allowed:
                costs.append(current.get(column - 1, sentinel) + 1)
            if costs:
                current[column] = min(min(costs), sentinel)
        previous = current
    return min(previous.get(len(right), sentinel), sentinel)


def validate_generation_task(payload: Mapping[str, Any], index: int = 0) -> None:
    """Validate the physical candidate schema and hard edit constraints."""

    keys = set(payload)
    missing = _GENERATION_TASK_FIELDS - keys
    unknown = keys - _GENERATION_TASK_FIELDS
    if missing:
        raise TrackContractError(
            f"candidate store record {index} missing fields: "
            + ", ".join(sorted(missing))
        )
    if unknown:
        raise TrackContractError(
            f"candidate store record {index} has unsealed fields: "
            + ", ".join(sorted(unknown))
        )
    if payload["schema_version"] != "generation_task.v2":
        raise TrackContractError(
            f"candidate store record {index} has wrong schema_version"
        )
    for field in ("task_id", "track_id", "source_id", "endpoint"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise TrackContractError(
                f"candidate store record {index} {field} must be non-empty"
            )
    if payload["region"] not in ("five_utr", "three_utr"):
        raise TrackContractError(
            f"candidate store record {index} has non-canonical region"
        )
    source_sequence = payload["source_sequence"]
    _validate_rna(source_sequence, "source_sequence")
    candidate_id = payload["candidate_id"]
    candidate_sequence = payload["candidate_sequence"]
    if (candidate_id is None) != (candidate_sequence is None):
        raise TrackContractError(
            f"candidate store record {index} must seal candidate id/sequence together"
        )
    if candidate_id is not None:
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise TrackContractError(
                f"candidate store record {index} candidate_id must be non-empty or null"
            )
        _validate_rna(candidate_sequence, "candidate_sequence")

    actions = payload["legal_action_types"]
    if (
        not isinstance(actions, list)
        or not actions
        or len(actions) != len(set(actions))
        or any(action not in _ACTION_TYPES for action in actions)
    ):
        raise TrackContractError(
            f"candidate store record {index} legal_action_types are invalid"
        )
    max_edits = payload["max_edits"]
    if isinstance(max_edits, bool) or not isinstance(max_edits, int) or max_edits < 0:
        raise TrackContractError(
            f"candidate store record {index} max_edits must be non-negative"
        )

    constraints = payload["constraints"]
    if not isinstance(constraints, Mapping):
        raise TrackContractError(
            f"candidate store record {index} constraints must be an object"
        )
    required_constraints = _CONSTRAINT_FIELDS - {"forbidden_motifs"}
    if (
        not required_constraints <= set(constraints)
        or set(constraints) - _CONSTRAINT_FIELDS
    ):
        raise TrackContractError(
            f"candidate store record {index} constraints violate sealed schema"
        )
    if constraints["source_conditioned"] is not True:
        raise TrackContractError("source_conditioned must be true")
    if constraints["sequence_alphabet"] != "RNA":
        raise TrackContractError("sequence_alphabet must be RNA")
    allowed = constraints["allowed_operations"]
    if (
        not isinstance(allowed, list)
        or not allowed
        or len(allowed) != len(set(allowed))
        or any(action not in _ACTION_TYPES for action in allowed)
    ):
        raise TrackContractError("allowed_operations are invalid")
    if set(allowed) != set(actions):
        raise TrackContractError(
            "legal_action_types and constraints.allowed_operations must match"
        )
    for field in ("min_length", "max_length"):
        value = constraints[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TrackContractError(f"{field} must be a positive integer")
    minimum = constraints["min_length"]
    maximum = constraints["max_length"]
    if minimum > maximum:
        raise TrackContractError("min_length cannot exceed max_length")
    if not minimum <= len(source_sequence) <= maximum:
        raise TrackContractError("source_sequence length violates constraints")
    if candidate_sequence is not None:
        if not minimum <= len(candidate_sequence) <= maximum:
            raise TrackContractError("candidate_sequence length violates constraints")
        edit_distance = _bounded_allowed_edit_distance(
            source_sequence,
            candidate_sequence,
            max_edits,
            allowed,
        )
        if edit_distance > max_edits:
            raise TrackContractError(
                "candidate edit distance exceeds max_edits under " "allowed_operations"
            )
    forbidden_motifs = constraints.get("forbidden_motifs", [])
    if not isinstance(forbidden_motifs, list) or len(forbidden_motifs) != len(
        set(forbidden_motifs)
    ):
        raise TrackContractError("forbidden_motifs must be a unique list")
    for motif in forbidden_motifs:
        _validate_rna(motif, "forbidden_motifs[]")
        if candidate_sequence is not None and motif in candidate_sequence:
            raise TrackContractError("candidate_sequence contains a forbidden motif")

    provenance = payload["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != _PROVENANCE_FIELDS
        or any(
            not isinstance(provenance[field], str) or not provenance[field].strip()
            for field in _PROVENANCE_FIELDS
        )
    ):
        raise TrackContractError(
            f"candidate store record {index} provenance violates sealed schema"
        )


def _require_sha256(value: Any, field: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise TrackContractError(f"{field} must be a 64-character SHA-256")
    return text


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrackContractError(f"{field} must be a non-negative integer")
    return value


def _resolve_relative_file(root: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise TrackContractError(f"{field} must be a non-empty path")
    path = Path(raw)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise TrackContractError(f"{field} escapes manifest directory") from exc
    if not resolved.is_file():
        raise TrackContractError(f"{field} does not exist: {resolved}")
    return resolved


def _validate_universe_binding(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _UNIVERSE_BINDING_FIELDS:
        raise TrackContractError(
            "universe_binding differs from the sealed universe schema"
        )
    binding = dict(raw)
    for field in (
        "canonical_records_sha256",
        "structural_records_sha256",
        "record_ids_sha256",
        "candidate_ids_sha256",
        "task_ids_sha256",
        "source_ids_sha256",
    ):
        binding[field] = _require_sha256(binding[field], f"universe_binding.{field}")
    for field in (
        "record_count",
        "candidate_count",
        "task_count",
        "source_count",
    ):
        binding[field] = _require_nonnegative_int(
            binding[field], f"universe_binding.{field}"
        )
    if binding["record_count"] < 1 or binding["task_count"] < 1:
        raise TrackContractError(
            "universe_binding record/task universes must be non-empty"
        )
    return binding


def load_candidate_store(
    path: Path, expected_sha256: str, expected_bytes: int | None = None
) -> List[Dict[str, Any]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise CandidateStoreHashError(
            f"candidate store SHA mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise CandidateStoreHashError(
            "candidate store byte size differs from frozen manifest"
        )
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrackContractError(
                    f"candidate store line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise TrackContractError(
                    f"candidate store line {line_number} must be an object"
                )
            records.append(payload)
    if not records:
        raise TrackContractError("candidate store must contain at least one task")
    assert_candidate_store_label_free(records)
    for index, record in enumerate(records):
        validate_generation_task(record, index)
    return records


def _validate_declared_file_ref(raw: Any, field: str) -> Dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _SEALED_FILE_REF_FIELDS:
        raise TrackContractError(
            f"{field} differs from the sealed file-reference schema"
        )
    if not isinstance(raw["path"], str) or not raw["path"].strip():
        raise TrackContractError(f"{field}.path must be declared")
    return {
        "path": raw["path"],
        "sha256": _require_sha256(raw["sha256"], f"{field}.sha256"),
        "bytes": _require_nonnegative_int(raw["bytes"], f"{field}.bytes"),
    }


def _validate_declared_label_store(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _LABEL_STORE_FIELDS:
        raise TrackContractError(
            "label_store differs from the sealed label-store schema"
        )
    payload = dict(raw)
    if not isinstance(payload["path"], str) or not payload["path"].strip():
        raise TrackContractError("label_store.path must be declared")
    payload["sha256"] = _require_sha256(payload["sha256"], "label_store.sha256")
    payload["bytes"] = _require_nonnegative_int(payload["bytes"], "label_store.bytes")
    if payload["access"] != "FROZEN_FINAL_ONLY":
        raise TrackContractError(
            "closed pool label_store access must be FROZEN_FINAL_ONLY"
        )
    if payload["candidate_id_field"] != "candidate_id":
        raise TrackContractError("label_store.candidate_id_field must be candidate_id")
    payload["candidate_ids_sha256"] = _require_sha256(
        payload["candidate_ids_sha256"],
        "label_store.candidate_ids_sha256",
    )
    payload["candidate_count"] = _require_nonnegative_int(
        payload["candidate_count"], "label_store.candidate_count"
    )
    payload["schema"] = _validate_declared_file_ref(
        payload["schema"], "label_store.schema"
    )
    payload["selection_freeze"] = _validate_declared_file_ref(
        payload["selection_freeze"], "label_store.selection_freeze"
    )
    proof = payload["freeze_proof"]
    if not isinstance(proof, Mapping) or set(proof) != _FREEZE_PROOF_REF_FIELDS:
        raise TrackContractError(
            "label_store.freeze_proof differs from sealed reference schema"
        )
    if not isinstance(proof["path"], str) or not proof["path"].strip():
        raise TrackContractError("label_store.freeze_proof.path is required")
    payload["freeze_proof"] = {
        "path": proof["path"],
        "sha256": _require_sha256(proof["sha256"], "label_store.freeze_proof.sha256"),
    }
    return payload


def load_track_manifest(path: Path) -> LoadedTrack:
    """Load one label-free track without opening final labels or proof."""

    manifest_path = Path(path).resolve()
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TrackContractError("track manifest must be an object")
    if payload.get("schema_version") != "utr_track.v2":
        raise TrackContractError("track schema_version must be utr_track.v2")
    track_id = str(payload.get("track_id") or "").strip()
    if not track_id:
        raise TrackContractError("track_id must be non-empty")
    track_type = str(payload.get("track_type") or "")
    if track_type not in TRACK_TYPES:
        raise TrackContractError(f"track_type must be one of {TRACK_TYPES}")
    expected_manifest_fields = set(_TRACK_MANIFEST_BASE_FIELDS)
    if track_type == "closed_measured_pool":
        expected_manifest_fields.add("label_store")
    if track_type == "open_legal_generation":
        expected_manifest_fields.add("evaluation_budget_protocol")
    if set(payload) != expected_manifest_fields:
        raise TrackContractError(
            "track manifest fields differ from the sealed track schema"
        )
    if payload.get("candidate_store_contains_labels") is not False:
        raise TrackContractError(
            "candidate_store_contains_labels must be explicitly false"
        )
    selection_access = payload.get("selection_access")
    if (
        not isinstance(selection_access, Mapping)
        or set(selection_access) != {"labels"}
        or selection_access.get("labels") is not False
    ):
        raise TrackContractError(
            "selection_access must contain only the sealed labels=false field"
        )
    universe_binding = _validate_universe_binding(payload.get("universe_binding"))
    budget_protocol: Mapping[str, Any] | None = None
    if track_type == "open_legal_generation":
        raw_budget_protocol = payload.get("evaluation_budget_protocol")
        if raw_budget_protocol != _OPEN_WORLD_BUDGET_PROTOCOL:
            raise TrackContractError(
                "open_legal_generation must freeze separate evaluation "
                "budgets 1/3/5 under one maximum-budget task representation"
            )
        budget_protocol = dict(raw_budget_protocol)

    candidate_store = payload.get("candidate_store")
    if not isinstance(candidate_store, Mapping) or set(candidate_store) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise TrackContractError(
            "candidate_store must contain only path, sha256, and bytes"
        )
    expected_sha = _require_sha256(
        candidate_store.get("sha256"), "candidate_store.sha256"
    )
    expected_bytes = _require_nonnegative_int(
        candidate_store.get("bytes"), "candidate_store.bytes"
    )
    candidate_path = _resolve_relative_file(
        manifest_path.parent,
        candidate_store.get("path"),
        "candidate_store.path",
    )
    candidates = load_candidate_store(candidate_path, expected_sha, expected_bytes)

    task_ids: List[str] = []
    candidate_ids: List[str] = []
    record_ids: List[str] = []
    source_ids: List[str] = []
    dataset_ids: List[str] = []
    for index, candidate in enumerate(candidates):
        if candidate.get("track_id") != track_id:
            raise TrackContractError(
                f"candidate store record {index} track_id differs from manifest"
            )
        if track_type == "closed_measured_pool" and (
            candidate.get("candidate_id") is None
            or candidate.get("candidate_sequence") is None
        ):
            raise TrackContractError(
                "closed_measured_pool tasks require sealed candidate identities"
            )
        if track_type == "heldout_generative" and (
            candidate.get("candidate_id") is None
            or candidate.get("candidate_sequence") is None
        ):
            raise TrackContractError(
                "heldout_generative tasks require sealed heldout candidates"
            )
        if track_type == "open_legal_generation" and (
            candidate.get("candidate_id") is not None
            or candidate.get("candidate_sequence") is not None
        ):
            raise TrackContractError(
                "open_legal_generation tasks must keep candidate null"
            )
        action_types = set(candidate.get("legal_action_types", []))
        if (
            track_type
            in (
                "heldout_generative",
                "open_legal_generation",
            )
            and "STOP" not in action_types
        ):
            raise TrackContractError(
                f"{track_type} tasks require explicit STOP semantics"
            )
        if track_type == "open_legal_generation" and action_types != _ACTION_TYPES:
            raise TrackContractError(
                "open_legal_generation tasks require the full legal "
                "INS/SUB/DEL/STOP action set"
            )
        if track_type == "open_legal_generation":
            constraints = candidate["constraints"]
            source_length = len(candidate["source_sequence"])
            if not (
                candidate["max_edits"] == OPEN_WORLD_MAX_EDIT_BUDGET
                and constraints["min_length"]
                == max(1, source_length - OPEN_WORLD_MAX_EDIT_BUDGET)
                and constraints["max_length"]
                == source_length + OPEN_WORLD_MAX_EDIT_BUDGET
            ):
                raise TrackContractError(
                    "open_legal_generation tasks must encode the frozen "
                    "maximum edit budget 5; reports must stratify 1/3/5"
                )
        task_ids.append(str(candidate["task_id"]).strip())
        if candidate["candidate_id"] is not None:
            candidate_ids.append(str(candidate["candidate_id"]).strip())
        record_ids.append(str(candidate["provenance"]["record_id"]).strip())
        source_ids.append(str(candidate["source_id"]).strip())
        dataset_ids.append(str(candidate["provenance"]["dataset_id"]).strip())
    for field, values in (
        ("task_id", task_ids),
        ("candidate identity", candidate_ids),
        ("provenance.record_id", record_ids),
    ):
        if len(values) != len(set(values)):
            raise TrackContractError(f"candidate store contains duplicate {field}")

    retrospective = payload.get("retrospective_external_stress_datasets")
    if (
        not isinstance(retrospective, list)
        or len(retrospective) != len(set(retrospective))
        or any(item != _RETROSPECTIVE_DATASET for item in retrospective)
    ):
        raise TrackContractError(
            "retrospective_external_stress_datasets may contain only GSE246381"
        )
    observed_retrospective = _RETROSPECTIVE_DATASET in dataset_ids
    if track_type in ("heldout_generative", "open_legal_generation") and (
        observed_retrospective or retrospective
    ):
        raise TrackContractError(
            "GSE246381 is retrospective external stress only and cannot enter "
            "normal Track B/C"
        )
    if track_type == "closed_measured_pool" and (
        observed_retrospective != (_RETROSPECTIVE_DATASET in retrospective)
    ):
        raise TrackContractError(
            "GSE246381 must be explicitly and exactly declared as "
            "retrospective external stress"
        )

    label_access = "NONE"
    label_sha: str | None = None
    label_bytes: int | None = None
    proof_sha: str | None = None
    label_schema_sha: str | None = None
    selection_freeze_sha: str | None = None
    if track_type == "closed_measured_pool":
        label_store = _validate_declared_label_store(payload.get("label_store"))
        label_access = label_store["access"]
        label_sha = label_store["sha256"]
        label_bytes = label_store["bytes"]
        proof_sha = label_store["freeze_proof"]["sha256"]
        label_schema_sha = label_store["schema"]["sha256"]
        selection_freeze_sha = label_store["selection_freeze"]["sha256"]
        if label_store["candidate_count"] != len(candidate_ids) or label_store[
            "candidate_ids_sha256"
        ] != ids_sha256(candidate_ids):
            raise TrackContractError(
                "declared label-store candidate universe differs from Track A"
            )
    elif "label_store" in payload:
        raise TrackContractError(
            "generative tracks must not declare a measured label_store"
        )

    return LoadedTrack(
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        track_id=track_id,
        track_type=track_type,
        candidate_store_path=str(candidate_path),
        candidate_store_sha256=expected_sha,
        candidate_store_bytes=expected_bytes,
        candidate_ids=tuple(sorted(candidate_ids)),
        task_ids=tuple(sorted(task_ids)),
        record_ids=tuple(sorted(record_ids)),
        source_ids=tuple(sorted(set(source_ids))),
        dataset_ids=tuple(sorted(set(dataset_ids))),
        candidate_count=len(candidate_ids),
        task_count=len(task_ids),
        record_count=len(record_ids),
        source_count=len(set(source_ids)),
        label_store_access=label_access,
        label_store_sha256=label_sha,
        label_store_bytes=label_bytes,
        label_freeze_proof_sha256=proof_sha,
        label_schema_sha256=label_schema_sha,
        selection_freeze_sha256=selection_freeze_sha,
        retrospective_external_stress_datasets=tuple(retrospective),
        universe_binding=universe_binding,
        tasks=tuple(dict(candidate) for candidate in candidates),
        evaluation_budget_protocol=budget_protocol,
    )


def _load_jsonl_privileged(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrackContractError(
                    f"label store line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise TrackContractError(
                    f"label store line {line_number} must be an object"
                )
            rows.append(row)
    if not rows:
        raise TrackContractError("label store must contain at least one row")
    return rows


def _sequence_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _verify_file_ref(
    root: Path,
    raw: Any,
    field: str,
) -> Tuple[Path, Dict[str, Any]]:
    reference = _validate_declared_file_ref(raw, field)
    path = _resolve_relative_file(root, reference["path"], f"{field}.path")
    if (
        sha256_file(path) != reference["sha256"]
        or path.stat().st_size != reference["bytes"]
    ):
        raise TrackContractError(f"{field} differs from frozen sha256/bytes")
    return path, reference


def _load_json_mapping(path: Path, field: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrackContractError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TrackContractError(f"{field} must contain one object")
    return value


def _validate_declared_store(
    raw: Any,
    *,
    fields: set[str],
    field: str,
) -> Dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise TrackContractError(f"{field} differs from its sealed schema")
    payload = dict(raw)
    if not isinstance(payload["path"], str) or not payload["path"].strip():
        raise TrackContractError(f"{field}.path must be non-empty")
    payload["sha256"] = _require_sha256(payload["sha256"], f"{field}.sha256")
    payload["bytes"] = _require_nonnegative_int(payload["bytes"], f"{field}.bytes")
    payload["record_count"] = _require_nonnegative_int(
        payload["record_count"], f"{field}.record_count"
    )
    payload["record_ids_sha256"] = _require_sha256(
        payload["record_ids_sha256"], f"{field}.record_ids_sha256"
    )
    if "structural_content_sha256" in fields:
        payload["structural_content_sha256"] = _require_sha256(
            payload["structural_content_sha256"],
            f"{field}.structural_content_sha256",
        )
    return payload


def privileged_verify_track_a_label_seal(
    manifest_path: Path,
    *,
    expected_role_policy: Mapping[str, Any],
    expected_d1_acceptance_path: Path,
) -> Dict[str, Any]:
    """Privileged acceptance-only verification of the complete Track A seal.

    The role policy and D1 acceptance path come from independently recomputed
    B0/D1 inputs.  A self-consistent reseal of a weakened policy or a different
    passing D1 run is therefore not sufficient for acceptance.
    """

    loaded = load_track_manifest(manifest_path)
    if loaded.track_type != "closed_measured_pool":
        raise TrackContractError("privileged label seal verification requires Track A")
    path = Path(loaded.manifest_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    label_store = _validate_declared_label_store(payload["label_store"])

    label_path = _resolve_relative_file(
        path.parent, label_store["path"], "label_store.path"
    )
    actual_label_sha = sha256_file(label_path)
    actual_label_bytes = label_path.stat().st_size
    if (
        actual_label_sha != label_store["sha256"]
        or actual_label_bytes != label_store["bytes"]
    ):
        raise TrackContractError("label store differs from its frozen sha256/bytes")
    schema_path, schema_ref = _verify_file_ref(
        path.parent, label_store["schema"], "label_store.schema"
    )
    selection_path, selection_ref = _verify_file_ref(
        path.parent,
        label_store["selection_freeze"],
        "label_store.selection_freeze",
    )
    selection = _load_json_mapping(selection_path, "Track A selection freeze")
    if set(selection) != _SELECTION_FREEZE_FIELDS:
        raise TrackContractError(
            "Track A selection freeze differs from the sealed schema"
        )
    if not (
        selection.get("schema_version") == "utr_track_a_prelabel_selection_freeze.v2"
        and selection.get("track_id") == loaded.track_id
        and selection.get("candidate_store_path") == loaded.candidate_store_path
        and selection.get("candidate_store_sha256") == loaded.candidate_store_sha256
        and selection.get("candidate_store_bytes") == loaded.candidate_store_bytes
        and selection.get("selected_record_ids_sha256") == ids_sha256(loaded.record_ids)
        and selection.get("selected_record_count") == loaded.record_count
        and selection.get("selected_task_ids_sha256") == ids_sha256(loaded.task_ids)
        and selection.get("selected_task_count") == loaded.task_count
        and selection.get("structural_records_sha256")
        == loaded.universe_binding["structural_records_sha256"]
        and selection.get("canonical_label_store_opened") is False
        and selection.get("selection_labels_hidden") is True
        and selection.get("frozen_before_label_access") is True
    ):
        raise TrackContractError(
            "Track A selection freeze does not bind the candidate selection"
        )

    role_policy_path = _resolve_relative_file(
        path.parent,
        selection.get("role_policy_path"),
        "selection_freeze.role_policy_path",
    )
    if not (
        selection.get("role_policy_sha256") == sha256_file(role_policy_path)
        and selection.get("role_policy_bytes") == role_policy_path.stat().st_size
    ):
        raise TrackContractError("Track A selection freeze role-policy binding changed")
    try:
        role_policy = yaml.safe_load(role_policy_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TrackContractError("Track role policy is invalid YAML") from exc
    if not isinstance(role_policy, Mapping) or dict(role_policy) != dict(
        expected_role_policy
    ):
        raise TrackContractError(
            "Track role policy differs from independent structural recomputation"
        )
    measured_policy = (
        role_policy.get("measured_eligibility", {})
        if isinstance(role_policy, Mapping)
        else {}
    )
    evidence_counts = (
        role_policy.get("track_evidence_counts", {})
        if isinstance(role_policy, Mapping)
        else {}
    )
    track_a_counts = (
        evidence_counts.get("closed_measured_pool", {})
        if isinstance(evidence_counts, Mapping)
        else {}
    )
    if not (
        isinstance(role_policy, Mapping)
        and role_policy.get("selection_is_label_independent") is True
        and role_policy.get("label_fields_read_for_selection") == []
        and role_policy.get("atomic_components_split") is False
        and measured_policy.get("effect_value_or_direction_used_for_selection") is False
        and track_a_counts.get("structural_unmeasured_pair_type") == 0
    ):
        raise TrackContractError(
            "Track role policy does not prove label-free measured routing"
        )

    if not (
        selection.get("hidden_label_schema_path") == str(schema_path)
        and selection.get("hidden_label_schema_sha256") == schema_ref["sha256"]
        and selection.get("hidden_label_schema_bytes") == schema_ref["bytes"]
    ):
        raise TrackContractError("selection freeze hidden-label schema binding changed")
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise TrackContractError(
            "jsonschema>=4.18 is required for privileged Track A verification"
        ) from exc
    hidden_schema = _load_json_mapping(schema_path, "Track A hidden-label schema")
    Draft202012Validator.check_schema(hidden_schema)
    hidden_validator = Draft202012Validator(hidden_schema)

    d1_acceptance_path, d1_acceptance_ref = _verify_file_ref(
        path.parent,
        selection.get("d1_acceptance"),
        "selection_freeze.d1_acceptance",
    )
    expected_d1_acceptance_path = expected_d1_acceptance_path.resolve()
    if d1_acceptance_path != expected_d1_acceptance_path:
        raise TrackContractError(
            "Track A seal references a different D1 acceptance artifact"
        )
    d1_acceptance = _load_json_mapping(d1_acceptance_path, "D1 acceptance")
    if not (
        d1_acceptance.get("phase_gate_passed") is True
        and d1_acceptance.get("fixture_mode") is False
        and d1_acceptance.get("structural_validation_passed") is True
        and isinstance(d1_acceptance.get("global_store_validation"), Mapping)
        and d1_acceptance["global_store_validation"].get("passed") is True
        and isinstance(d1_acceptance.get("required_artifact_validation"), Mapping)
        and d1_acceptance["required_artifact_validation"].get("passed") is True
    ):
        raise TrackContractError("D1 acceptance is no longer a production PASS")
    d1_build_path, d1_build_ref = _verify_file_ref(
        path.parent,
        selection.get("d1_build_manifest"),
        "selection_freeze.d1_build_manifest",
    )
    stage_root = Path(str(d1_acceptance.get("stage_d1_root") or ""))
    if (
        not stage_root.is_absolute()
        or d1_build_path != (stage_root / "build_manifest.json").resolve()
    ):
        raise TrackContractError("D1 acceptance/build-manifest path binding changed")
    d1_build = _load_json_mapping(d1_build_path, "D1 build manifest")
    global_stores = d1_build.get("global_stores")
    if not isinstance(global_stores, Mapping):
        raise TrackContractError("D1 build manifest lacks global stores")
    canonical_meta = global_stores.get("canonical_label_store")
    structural_meta = global_stores.get("sealed_label_free_candidate_store")
    if not isinstance(canonical_meta, Mapping) or not isinstance(
        structural_meta, Mapping
    ):
        raise TrackContractError("D1 global-store bindings are incomplete")
    canonical_declared = _validate_declared_store(
        selection.get("canonical_records_declared"),
        fields=_CANONICAL_DECLARED_FIELDS,
        field="selection_freeze.canonical_records_declared",
    )
    structural_declared = _validate_declared_store(
        selection.get("structural_records"),
        fields=_STRUCTURAL_DECLARED_FIELDS,
        field="selection_freeze.structural_records",
    )
    canonical_path = _resolve_relative_file(
        path.parent,
        canonical_declared["path"],
        "selection_freeze.canonical_records_declared.path",
    )
    structural_path = _resolve_relative_file(
        path.parent,
        structural_declared["path"],
        "selection_freeze.structural_records.path",
    )
    expected_canonical_path = (
        stage_root / str(canonical_meta.get("path") or "")
    ).resolve()
    expected_structural_path = (
        stage_root / str(structural_meta.get("path") or "")
    ).resolve()
    if canonical_path != expected_canonical_path or structural_path != (
        expected_structural_path
    ):
        raise TrackContractError("selection freeze differs from D1 global-store paths")
    for declared, meta, store_path, universe_field in (
        (
            canonical_declared,
            canonical_meta,
            canonical_path,
            "canonical_records_sha256",
        ),
        (
            structural_declared,
            structural_meta,
            structural_path,
            "structural_records_sha256",
        ),
    ):
        if not (
            declared["sha256"] == meta.get("sha256")
            and declared["bytes"] == meta.get("bytes")
            and declared["record_count"] == meta.get("records")
            and declared["record_ids_sha256"] == meta.get("record_ids_sha256")
            and sha256_file(store_path) == declared["sha256"]
            and store_path.stat().st_size == declared["bytes"]
            and declared["sha256"] == loaded.universe_binding[universe_field]
        ):
            raise TrackContractError(f"D1 {universe_field} binding changed")

    canonical_rows = _load_jsonl_privileged(canonical_path)
    canonical_index: Dict[str, Dict[str, Any]] = {}
    for row in canonical_rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise TrackContractError("D1 canonical store contains a missing record_id")
        if record_id in canonical_index:
            raise TrackContractError("D1 canonical store contains duplicate record_id")
        canonical_index[record_id] = row
    if (
        len(canonical_rows) != canonical_declared["record_count"]
        or ids_sha256(list(canonical_index)) != canonical_declared["record_ids_sha256"]
    ):
        raise TrackContractError(
            "D1 canonical record universe differs from its manifest"
        )

    candidate_rows = load_candidate_store(
        Path(loaded.candidate_store_path),
        loaded.candidate_store_sha256,
        loaded.candidate_store_bytes,
    )
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows}
    rows = _load_jsonl_privileged(label_path)
    label_candidate_ids: List[str] = []
    label_record_ids: List[str] = []
    for index, row in enumerate(rows):
        schema_errors = sorted(
            hidden_validator.iter_errors(row),
            key=lambda error: tuple(str(part) for part in error.path),
        )
        if schema_errors or set(row) != _HIDDEN_LABEL_FIELDS:
            message = (
                schema_errors[0].message
                if schema_errors
                else "hidden-label fields differ from the sealed schema"
            )
            raise TrackContractError(
                f"label store row {index} violates strict schema: {message}"
            )
        candidate_id = row["candidate_id"]
        record_id = row["record_id"]
        task = candidate_by_id.get(candidate_id)
        canonical = canonical_index.get(record_id)
        if task is None or canonical is None:
            raise TrackContractError(
                f"label store row {index} lacks task/canonical identity"
            )
        expected_identity = {
            "canonical_candidate_id": canonical.get("candidate_id"),
            "dataset_id": canonical.get("dataset_id"),
            "study_id": canonical.get("study_id"),
            "assay_id": canonical.get("assay_id"),
            "context_id": canonical.get("context_id"),
            "endpoint": canonical.get("endpoint"),
            "pair_type": canonical.get("pair_type"),
            "source_id": canonical.get("source_id"),
            "source_sequence_sha256": _sequence_sha256(
                str(canonical.get("source_sequence") or "")
            ),
            "candidate_sequence_sha256": _sequence_sha256(
                str(canonical.get("candidate_sequence") or "")
            ),
            "source_value_raw": canonical.get("source_value_raw"),
            "candidate_value_raw": canonical.get("candidate_value_raw"),
            "delta_raw": canonical.get("delta_raw"),
            "delta_normalized": canonical.get("delta_normalized"),
            "effect_standard_error": canonical.get("effect_standard_error"),
            "replicate_count": canonical.get("replicate_count"),
            "label_provenance": canonical.get("label_provenance"),
            "canonical_record_sha256": _stable_payload_sha256(canonical),
            "measurement_evidence": "paired_finite_measured_endpoints",
        }
        if any(row.get(key) != value for key, value in expected_identity.items()):
            raise TrackContractError(
                f"label store row {index} differs from D1 canonical identity"
            )
        if not (
            task["provenance"]["record_id"] == record_id
            and task["provenance"]["dataset_id"] == row["dataset_id"]
            and task["provenance"]["study_id"] == row["study_id"]
            and task["source_id"] == row["source_id"]
            and task["endpoint"] == row["endpoint"]
            and _sequence_sha256(task["source_sequence"])
            == row["source_sequence_sha256"]
            and _sequence_sha256(task["candidate_sequence"])
            == row["candidate_sequence_sha256"]
        ):
            raise TrackContractError(
                f"label store row {index} differs from frozen Track A task"
            )
        source_value = row["source_value_raw"]
        candidate_value = row["candidate_value_raw"]
        delta = row["delta_raw"]
        if not (
            row["pair_type"] in MEASURED_PAIR_TYPES
            and all(
                _is_finite_number(value)
                for value in (source_value, candidate_value, delta)
            )
            and math.isclose(
                float(delta),
                float(candidate_value) - float(source_value),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and (
                row["delta_normalized"] is None
                or _is_finite_number(row["delta_normalized"])
            )
            and (
                row["effect_standard_error"] is None
                or _is_finite_number(row["effect_standard_error"])
            )
        ):
            raise TrackContractError(
                f"label store row {index} lacks a finite paired measurement"
            )
        label_candidate_ids.append(candidate_id)
        label_record_ids.append(record_id)
    if (
        len(label_candidate_ids) != len(set(label_candidate_ids))
        or len(label_record_ids) != len(set(label_record_ids))
        or set(label_candidate_ids) != set(loaded.candidate_ids)
        or set(label_record_ids) != set(loaded.record_ids)
    ):
        raise TrackContractError(
            "label store is not a record/candidate bijection with Track A"
        )
    label_ids_sha = ids_sha256(label_candidate_ids)
    label_record_ids_sha = ids_sha256(label_record_ids)
    if (
        len(label_candidate_ids) != label_store["candidate_count"]
        or label_ids_sha != label_store["candidate_ids_sha256"]
    ):
        raise TrackContractError(
            "label store candidate universe differs from its manifest"
        )

    proof_ref = label_store["freeze_proof"]
    proof_path = _resolve_relative_file(
        path.parent,
        proof_ref["path"],
        "label_store.freeze_proof.path",
    )
    actual_proof_sha = sha256_file(proof_path)
    if actual_proof_sha != proof_ref["sha256"]:
        raise TrackContractError("label freeze proof SHA mismatch")
    proof = _load_json_mapping(proof_path, "label freeze proof")
    if set(proof) != _FREEZE_PROOF_FIELDS:
        raise TrackContractError(
            "label freeze proof differs from the sealed proof schema"
        )
    expected = {
        "schema_version": "utr_track_a_label_freeze_proof.v2",
        "track_id": loaded.track_id,
        "candidate_store_sha256": loaded.candidate_store_sha256,
        "candidate_store_bytes": loaded.candidate_store_bytes,
        "label_store_sha256": actual_label_sha,
        "label_store_bytes": actual_label_bytes,
        "candidate_ids_sha256": ids_sha256(loaded.candidate_ids),
        "candidate_count": loaded.candidate_count,
        "label_candidate_ids_sha256": label_ids_sha,
        "label_count": len(label_candidate_ids),
        "label_record_ids_sha256": label_record_ids_sha,
        "label_record_count": len(label_record_ids),
        "selection_freeze_sha256": selection_ref["sha256"],
        "selection_freeze_bytes": selection_ref["bytes"],
        "role_policy_sha256": selection["role_policy_sha256"],
        "hidden_label_schema_sha256": schema_ref["sha256"],
        "canonical_records_sha256": loaded.universe_binding["canonical_records_sha256"],
        "structural_records_sha256": loaded.universe_binding[
            "structural_records_sha256"
        ],
        "record_ids_sha256": loaded.universe_binding["record_ids_sha256"],
        "frozen_before_label_access": True,
        "selection_labels_hidden": True,
    }
    if proof != expected:
        raise TrackContractError(
            "label freeze proof does not bind the actual files and universes"
        )
    return {
        "schema_version": "utr_track_a_label_seal_audit.v2",
        "track_id": loaded.track_id,
        "gate_passed": True,
        "candidate_label_bijection": True,
        "record_label_bijection": True,
        "strict_hidden_label_schema_passed": True,
        "paired_finite_measured_labels": True,
        "canonical_identity_binding_passed": True,
        "d1_acceptance_binding_passed": True,
        "current_d1_chain_binding_passed": True,
        "role_policy_exact_binding_passed": True,
        "label_store_sha256": actual_label_sha,
        "label_store_bytes": actual_label_bytes,
        "freeze_proof_sha256": actual_proof_sha,
        "selection_freeze_sha256": selection_ref["sha256"],
        "role_policy_sha256": selection["role_policy_sha256"],
        "hidden_label_schema_sha256": schema_ref["sha256"],
        "d1_acceptance_sha256": d1_acceptance_ref["sha256"],
        "d1_build_manifest_sha256": d1_build_ref["sha256"],
        "candidate_ids_sha256": label_ids_sha,
        "candidate_count": len(label_candidate_ids),
        "label_record_ids_sha256": label_record_ids_sha,
        "label_record_count": len(label_record_ids),
        "canonical_records_sha256": loaded.universe_binding["canonical_records_sha256"],
        "structural_records_sha256": loaded.universe_binding[
            "structural_records_sha256"
        ],
        "record_ids_sha256": loaded.universe_binding["record_ids_sha256"],
    }


def audit_track_roles(
    tracks: Sequence[LoadedTrack],
    *,
    eligible_records: Sequence[Mapping[str, Any]] | None = None,
    expected_role_by_record: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Require exact A/B/C roles and complete four-identity universe coverage."""

    issues: List[Dict[str, Any]] = []
    by_type: Dict[str, List[LoadedTrack]] = {
        track_type: [] for track_type in TRACK_TYPES
    }
    seen_track_ids: Dict[str, int] = {}
    for track in tracks:
        by_type[track.track_type].append(track)
        seen_track_ids[track.track_id] = seen_track_ids.get(track.track_id, 0) + 1
    for track_id, count in sorted(seen_track_ids.items()):
        if count != 1:
            issues.append(
                {
                    "kind": "duplicate_track_id",
                    "track_id": track_id,
                    "count": count,
                }
            )
    for track_type in TRACK_TYPES:
        count = len(by_type[track_type])
        if count != 1:
            issues.append(
                {
                    "kind": "track_type_cardinality",
                    "track_type": track_type,
                    "count": count,
                    "expected": 1,
                }
            )

    identity_fields = {
        "candidate": "candidate_ids",
        "task": "task_ids",
        "record": "record_ids",
        "source": "source_ids",
    }
    for left_type, right_type in combinations(TRACK_TYPES, 2):
        for identity_kind, attribute in identity_fields.items():
            left_ids = {
                value
                for track in by_type[left_type]
                for value in getattr(track, attribute)
            }
            right_ids = {
                value
                for track in by_type[right_type]
                for value in getattr(track, attribute)
            }
            for identity in sorted(left_ids & right_ids):
                issues.append(
                    {
                        "kind": f"{identity_kind}_role_overlap",
                        f"{identity_kind}_id": identity,
                        "track_types": [left_type, right_type],
                    }
                )

    bindings = [dict(track.universe_binding) for track in tracks]
    common_binding: Dict[str, Any] | None = bindings[0] if bindings else None
    if not bindings or any(binding != common_binding for binding in bindings):
        issues.append({"kind": "track_universe_binding_mismatch"})
        common_binding = None

    identity_universes: Dict[str, Dict[str, Any]] = {}
    for identity_kind, attribute in identity_fields.items():
        values = sorted(
            {value for track in tracks for value in getattr(track, attribute)}
        )
        identity_universes[identity_kind] = {
            "count": len(values),
            "ids_sha256": ids_sha256(values),
        }
    identity_universe_complete = common_binding is not None
    if common_binding is not None:
        for identity_kind in ("record", "candidate", "task", "source"):
            observed = identity_universes[identity_kind]
            if (
                observed["count"] != common_binding[f"{identity_kind}_count"]
                or observed["ids_sha256"]
                != common_binding[f"{identity_kind}_ids_sha256"]
            ):
                identity_universe_complete = False
                issues.append(
                    {
                        "kind": f"{identity_kind}_universe_binding_mismatch",
                        "expected_count": common_binding[f"{identity_kind}_count"],
                        "observed_count": observed["count"],
                    }
                )

    eligible_identity_binding_checked = eligible_records is not None
    eligible_identity_binding_complete = False
    if eligible_records is not None:
        canonical_record_ids: List[str] = []
        canonical_source_ids = set()
        for index, record in enumerate(eligible_records):
            record_id = record.get("record_id")
            if not isinstance(record_id, str) or not record_id.strip():
                issues.append(
                    {
                        "kind": "canonical_record_id_missing",
                        "index": index,
                    }
                )
                continue
            canonical_record_ids.append(record_id.strip())
            source_id = record.get("source_id")
            if source_id is not None:
                if not isinstance(source_id, str) or not source_id.strip():
                    issues.append(
                        {
                            "kind": "canonical_source_id_invalid",
                            "record_id": record_id,
                        }
                    )
                else:
                    canonical_source_ids.add(source_id.strip())
        if len(canonical_record_ids) != len(set(canonical_record_ids)):
            issues.append({"kind": "canonical_record_id_duplicate"})
        observed_record_ids = {
            record_id for track in tracks for record_id in track.record_ids
        }
        observed_source_ids = {
            source_id for track in tracks for source_id in track.source_ids
        }
        observed_task_count = sum(track.task_count for track in tracks)
        eligible_identity_binding_complete = (
            len(canonical_record_ids) == len(set(canonical_record_ids))
            and observed_record_ids == set(canonical_record_ids)
            and observed_source_ids == canonical_source_ids
            and observed_task_count == len(canonical_record_ids)
        )
        if observed_record_ids != set(canonical_record_ids):
            issues.append(
                {
                    "kind": "canonical_record_role_universe_mismatch",
                    "expected_count": len(set(canonical_record_ids)),
                    "observed_count": len(observed_record_ids),
                }
            )
        if observed_source_ids != canonical_source_ids:
            issues.append(
                {
                    "kind": "canonical_source_role_universe_mismatch",
                    "expected_count": len(canonical_source_ids),
                    "observed_count": len(observed_source_ids),
                }
            )
        if observed_task_count != len(canonical_record_ids):
            issues.append(
                {
                    "kind": "canonical_task_role_universe_mismatch",
                    "expected_count": len(canonical_record_ids),
                    "observed_count": observed_task_count,
                }
            )

    task_binding_checked = (
        eligible_records is not None and expected_role_by_record is not None
    )
    task_binding_complete = False
    if task_binding_checked:
        record_index = {
            str(record.get("record_id") or ""): record
            for record in eligible_records or ()
        }
        expected_role_map = {
            str(record_id): str(track_type)
            for record_id, track_type in expected_role_by_record.items()
        }
        if (
            "" in record_index
            or len(record_index) != len(eligible_records or ())
            or set(record_index) != set(expected_role_map)
            or any(
                track_type not in TRACK_TYPES
                for track_type in expected_role_map.values()
            )
        ):
            issues.append({"kind": "expected_task_role_universe_invalid"})
        observed_tasks: Dict[str, Tuple[str, Mapping[str, Any]]] = {}
        for track in tracks:
            for task in track.tasks:
                provenance = task.get("provenance")
                record_id = (
                    str(provenance.get("record_id") or "")
                    if isinstance(provenance, Mapping)
                    else ""
                )
                if not record_id or record_id in observed_tasks:
                    issues.append(
                        {
                            "kind": "task_record_binding_duplicate_or_missing",
                            "record_id": record_id,
                        }
                    )
                    continue
                observed_tasks[record_id] = (track.track_type, task)
        for record_id in sorted(set(record_index) | set(observed_tasks)):
            expected_record = record_index.get(record_id)
            observed = observed_tasks.get(record_id)
            expected_track = expected_role_map.get(record_id)
            if expected_record is None or observed is None or expected_track is None:
                issues.append(
                    {
                        "kind": "task_record_binding_universe_mismatch",
                        "record_id": record_id,
                    }
                )
                continue
            observed_track, observed_task = observed
            if observed_track != expected_track:
                issues.append(
                    {
                        "kind": "task_frozen_role_mismatch",
                        "record_id": record_id,
                        "expected_track_type": expected_track,
                        "observed_track_type": observed_track,
                    }
                )
                continue
            try:
                expected_task = expected_generation_task(
                    expected_record,
                    track_type=expected_track,
                )
            except TrackContractError as exc:
                issues.append(
                    {
                        "kind": "structural_task_projection_invalid",
                        "record_id": record_id,
                        "detail": str(exc),
                    }
                )
                continue
            if dict(observed_task) != expected_task:
                issues.append(
                    {
                        "kind": "task_structural_binding_mismatch",
                        "record_id": record_id,
                        "track_type": expected_track,
                    }
                )
        task_binding_complete = not any(
            issue["kind"]
            in {
                "expected_task_role_universe_invalid",
                "task_record_binding_duplicate_or_missing",
                "task_record_binding_universe_mismatch",
                "task_frozen_role_mismatch",
                "structural_task_projection_invalid",
                "task_structural_binding_mismatch",
            }
            for issue in issues
        )

    region_counts: Dict[str, int] = {}
    dataset_record_counts: Dict[str, int] = {}
    track_task_counts: Dict[str, int] = {}
    for track in tracks:
        track_task_counts[track.track_type] = track_task_counts.get(
            track.track_type, 0
        ) + len(track.tasks)
        for task in track.tasks:
            region = str(task.get("region") or "")
            region_counts[region] = region_counts.get(region, 0) + 1
            provenance = task.get("provenance")
            dataset_id = (
                str(provenance.get("dataset_id") or "")
                if isinstance(provenance, Mapping)
                else ""
            )
            dataset_record_counts[dataset_id] = (
                dataset_record_counts.get(dataset_id, 0) + 1
            )
    track_evidence_counts = {
        track_type: {
            "measured_pair_type": 0,
            "structural_unmeasured_pair_type": 0,
        }
        for track_type in TRACK_TYPES
    }
    if eligible_records is not None and expected_role_by_record is not None:
        for record in eligible_records:
            record_id = str(record.get("record_id") or "")
            track_type = expected_role_by_record.get(record_id)
            if track_type not in track_evidence_counts:
                continue
            key = (
                "measured_pair_type"
                if record.get("pair_type") in MEASURED_PAIR_TYPES
                else "structural_unmeasured_pair_type"
            )
            track_evidence_counts[track_type][key] += 1

    return {
        "schema_version": "utr_track_role_audit.v2",
        "track_count": len(tracks),
        "track_role_ambiguity_count": len(issues),
        "identity_universe_complete": identity_universe_complete,
        "eligible_identity_binding_checked": (eligible_identity_binding_checked),
        "eligible_identity_binding_complete": (eligible_identity_binding_complete),
        "task_structural_binding_checked": task_binding_checked,
        "task_structural_binding_complete": task_binding_complete,
        "gate_passed": len(issues) == 0 and identity_universe_complete,
        "issues": issues[:100],
        "universe_binding": common_binding,
        "identity_universes": identity_universes,
        "data_card_counts": {
            "region_records": dict(sorted(region_counts.items())),
            "dataset_records": dict(sorted(dataset_record_counts.items())),
            "track_tasks": dict(sorted(track_task_counts.items())),
            "track_evidence": track_evidence_counts,
        },
        "gse246381_role": _RETROSPECTIVE_ROLE,
        "tracks": [
            {
                "track_id": track.track_id,
                "track_type": track.track_type,
                "manifest_sha256": track.manifest_sha256,
                "candidate_count": track.candidate_count,
                "candidate_ids_sha256": ids_sha256(track.candidate_ids),
                "task_count": track.task_count,
                "task_ids_sha256": ids_sha256(track.task_ids),
                "record_count": track.record_count,
                "record_ids_sha256": ids_sha256(track.record_ids),
                "source_count": track.source_count,
                "source_ids_sha256": ids_sha256(track.source_ids),
                "candidate_store_sha256": track.candidate_store_sha256,
                "candidate_store_bytes": track.candidate_store_bytes,
                "label_store_access": track.label_store_access,
                "label_store_sha256": track.label_store_sha256,
                "label_store_bytes": track.label_store_bytes,
                "label_freeze_proof_sha256": (track.label_freeze_proof_sha256),
                "label_schema_sha256": track.label_schema_sha256,
                "selection_freeze_sha256": (track.selection_freeze_sha256),
                "retrospective_external_stress_datasets": list(
                    track.retrospective_external_stress_datasets
                ),
                "evaluation_budget_protocol": (
                    dict(track.evaluation_budget_protocol)
                    if track.evaluation_budget_protocol is not None
                    else None
                ),
            }
            for track in sorted(tracks, key=lambda item: item.track_type)
        ],
    }
