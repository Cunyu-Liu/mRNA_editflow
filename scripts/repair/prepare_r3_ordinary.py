"""Prepare a fail-closed ordinary-only R3 inclusion bundle.

This script does not discover publication contexts and does not reinterpret
indels as substitutions. It binds context only through a byte-hash-pinned,
versioned frozen crosswalk; derives a SUB script only for the narrow,
deterministic incomplete-script case; and keeps out-of-contract edit budgets
outside the primary inclusion file.

Example::

    python -m scripts.repair.prepare_r3_ordinary \
      --dataset artifacts/b0x/effect_dataset.jsonl \
      --crosswalk configs/repair/r3_context_crosswalk_v1.json \
      --crosswalk-sha256 <sha256> \
      --out-dir artifacts/repair/r3_ordinary_<run_id>
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse.dataset import (  # noqa: E402
    IncompatibleEditEncodingError,
    encode_pair,
)
from scripts.m4_sparse.ordinary_io import (  # noqa: E402
    assert_ordinary_path,
    assert_ordinary_rows,
    load_ordinary_jsonl,
)


PREPARATION_VERSION = "r3_ordinary_preparation_v1"
CROSSWALK_SCHEMA_VERSION = "r3_ordinary_context_crosswalk_v1"
ESTIMAND_SCHEMA_VERSION = "r3_endpoint_estimand_registry_v1"
SOURCE_GROUP_VERSION = "source_group_id_v2"
EDIT_REPAIR_METHOD = "DERIVED_STRING_TRANSFORM_FROM_ACCEPTED_RELATION_V1"
PRIMARY_EDIT_BUDGETS = (1, 3, 5)
REPAIRABLE_EDIT_COUNTS = (1, 2, 3)
MAX_REPAIR_SPAN = 2
READY_FILE = "r3_ordinary_primary_ready.jsonl"
HOLD_FILE = "r3_ordinary_hold.jsonl"
MANIFEST_FILE = "r3_ordinary_manifest.json"

CROSSWALK_ATOMS = (
    "study",
    "endpoint",
    "assay_arm",
    "library",
    "source_asset",
    "region",
    "source_anchor_rule",
)
CROSSWALK_REQUIRED_FIELDS = CROSSWALK_ATOMS + (
    "publication_context_id",
    "context_evidence_id",
    "source_relation_evidence_id",
    "allowed_row_relation_evidence",
)

ESTIMAND_REQUIRED_FIELDS = (
    "study",
    "endpoint",
    "raw_source_field",
    "raw_candidate_field",
    "raw_unit",
    "direction",
    "transform",
    "valid_domain",
    "role",
    "replicate_policy",
    "source_citation",
    "source_evidence_sha256",
)
ALLOWED_DIRECTIONS = (
    "HIGHER_IS_MORE_STABLE",
    "LOWER_IS_MORE_STABLE",
    "SIGNED_EFFECT_AS_REPORTED",
)
ALLOWED_TRANSFORMS = (
    "IDENTITY_DIFFERENCE",
    "NATURAL_LOG_RATIO",
    "DECAY_RATE_DIFFERENCE",
)
ALLOWED_DOMAINS = ("FINITE", "STRICTLY_POSITIVE")
ALLOWED_ESTIMAND_ROLES = (
    "PRIMARY",
    "SECONDARY_SENSITIVITY",
    "DEVELOPMENT",
)
ALLOWED_REPLICATE_POLICIES = (
    "SINGLE_OBSERVATION_ONLY",
    "PRESERVE_FOR_HIERARCHICAL_MODEL",
    "PRE_REGISTERED_AGGREGATE_WITH_SE",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_sequence(value: Any) -> str:
    return _text(value).upper().replace("T", "U")


def _is_study_sentinel(context_id: str, study: str) -> bool:
    def token(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    return token(context_id) == "ctx_%s" % token(study)


def _crosswalk_key(row: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(_text(row.get(field)) for field in CROSSWALK_ATOMS)


def load_frozen_crosswalk(path: Path, expected_sha256: str) -> dict:
    """Load a frozen crosswalk only when its exact bytes are pre-bound."""
    path = Path(path)
    assert_ordinary_path(path, purpose="context crosswalk input")
    expected = _text(expected_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("crosswalk expected SHA-256 must be 64 lowercase hex")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            "context crosswalk SHA-256 mismatch: expected %s got %s"
            % (expected, actual)
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("context crosswalk must be a JSON object")
    if payload.get("schema_version") != CROSSWALK_SCHEMA_VERSION:
        raise ValueError("unsupported context crosswalk schema_version")
    if payload.get("status") != "FROZEN":
        raise ValueError("context crosswalk must have status=FROZEN")
    for field in ("crosswalk_id", "version", "frozen_at_utc"):
        if not _text(payload.get(field)):
            raise ValueError("context crosswalk is missing %s" % field)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("context crosswalk entries must be a list")
    index: Dict[Tuple[str, ...], dict] = {}
    for entry_index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                "context crosswalk entry %d is not an object" % entry_index
            )
        assert_ordinary_rows(
            [raw_entry], purpose="context crosswalk entry"
        )
        entry = dict(raw_entry)
        missing = [
            field for field in CROSSWALK_REQUIRED_FIELDS
            if not _text(entry.get(field))
        ]
        if missing:
            raise ValueError(
                "context crosswalk entry %d missing: %s"
                % (entry_index, ",".join(missing))
            )
        if _is_study_sentinel(
            _text(entry["publication_context_id"]), _text(entry["study"])
        ):
            raise ValueError(
                "ctx_<study> sentinel is not a publication context"
            )
        allowed_relation_evidence = entry.get(
            "allowed_row_relation_evidence"
        )
        if not isinstance(
            allowed_relation_evidence, list
        ) or not allowed_relation_evidence:
            raise ValueError(
                "context crosswalk entry requires a non-empty "
                "allowed_row_relation_evidence list"
            )
        normalized_relation_evidence = []
        seen_relation_ids = set()
        for relation in allowed_relation_evidence:
            if not isinstance(relation, dict):
                raise ValueError(
                    "allowed row relation evidence must be objects"
                )
            relation_id = _text(relation.get("evidence_id"))
            relation_sha256 = _text(relation.get("evidence_sha256")).lower()
            relation_status = _text(relation.get("relation_status"))
            if not relation_id or relation_id in seen_relation_ids:
                raise ValueError(
                    "allowed row relation evidence ids must be unique"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", relation_sha256):
                raise ValueError(
                    "allowed row relation evidence requires SHA-256"
                )
            if relation_status != "ACCEPTED":
                raise ValueError(
                    "allowed row relation evidence must be ACCEPTED"
                )
            seen_relation_ids.add(relation_id)
            normalized_relation_evidence.append({
                "evidence_id": relation_id,
                "evidence_sha256": relation_sha256,
                "relation_status": relation_status,
            })
        entry["allowed_row_relation_evidence"] = sorted(
            normalized_relation_evidence,
            key=lambda item: item["evidence_id"],
        )
        key = _crosswalk_key(entry)
        if key in index:
            raise ValueError(
                "duplicate or ambiguous context crosswalk key: %s" % (key,)
            )
        index[key] = entry
    return {
        "path": str(path),
        "sha256": actual,
        "crosswalk_id": payload["crosswalk_id"],
        "version": payload["version"],
        "frozen_at_utc": payload["frozen_at_utc"],
        "entries": entries,
        "index": index,
    }


def load_frozen_estimand_registry(path: Path, expected_sha256: str) -> dict:
    """Load an exact-byte frozen endpoint estimand registry."""
    path = Path(path)
    assert_ordinary_path(path, purpose="endpoint estimand registry input")
    expected = _text(expected_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("estimand registry SHA-256 must be 64 lowercase hex")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            "endpoint estimand registry SHA-256 mismatch: expected %s got %s"
            % (expected, actual)
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("endpoint estimand registry must be a JSON object")
    if payload.get("schema_version") != ESTIMAND_SCHEMA_VERSION:
        raise ValueError("unsupported endpoint estimand registry schema")
    if payload.get("status") != "FROZEN":
        raise ValueError("endpoint estimand registry must have status=FROZEN")
    for field in ("registry_id", "version", "frozen_at_utc"):
        if not _text(payload.get(field)):
            raise ValueError("endpoint estimand registry missing %s" % field)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("endpoint estimand registry entries must be a list")
    index = {}
    for entry_index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                "endpoint estimand entry %d is not an object" % entry_index
            )
        assert_ordinary_rows(
            [raw_entry], purpose="endpoint estimand registry entry"
        )
        entry = dict(raw_entry)
        missing = [
            field for field in ESTIMAND_REQUIRED_FIELDS
            if not _text(entry.get(field))
        ]
        if missing:
            raise ValueError(
                "endpoint estimand entry %d missing: %s"
                % (entry_index, ",".join(missing))
            )
        if entry["direction"] not in ALLOWED_DIRECTIONS:
            raise ValueError("invalid endpoint estimand direction")
        if entry["transform"] not in ALLOWED_TRANSFORMS:
            raise ValueError("invalid endpoint estimand transform")
        if entry["valid_domain"] not in ALLOWED_DOMAINS:
            raise ValueError("invalid endpoint estimand valid_domain")
        if entry["role"] not in ALLOWED_ESTIMAND_ROLES:
            raise ValueError("invalid endpoint estimand role")
        if entry["replicate_policy"] not in ALLOWED_REPLICATE_POLICIES:
            raise ValueError("invalid endpoint estimand replicate_policy")
        if entry["role"] == "PRIMARY" and entry[
            "transform"
        ] != "IDENTITY_DIFFERENCE":
            raise ValueError(
                "PRIMARY endpoint estimand must preserve raw endpoint "
                "semantics with IDENTITY_DIFFERENCE"
            )
        if entry["transform"] in (
            "NATURAL_LOG_RATIO", "DECAY_RATE_DIFFERENCE"
        ) and entry["role"] != "SECONDARY_SENSITIVITY":
            raise ValueError(
                "log-ratio and decay-rate transforms are "
                "SECONDARY_SENSITIVITY only"
            )
        if entry["study"] == "GSE217518" and entry[
            "role"
        ] == "PRIMARY" and entry["direction"] != "HIGHER_IS_MORE_STABLE":
            raise ValueError(
                "GSE217518 primary half-life direction must be "
                "HIGHER_IS_MORE_STABLE"
            )
        if entry["raw_source_field"] != "source_value" or entry[
            "raw_candidate_field"
        ] != "candidate_value":
            raise ValueError(
                "R3 v1 estimand registry only accepts raw source_value and "
                "candidate_value fields"
            )
        evidence_hash = _text(entry["source_evidence_sha256"]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", evidence_hash):
            raise ValueError("invalid estimand source_evidence_sha256")
        key = (_text(entry["study"]), _text(entry["endpoint"]))
        if key in index:
            raise ValueError("duplicate endpoint estimand key: %s" % (key,))
        index[key] = entry
    return {
        "path": str(path),
        "sha256": actual,
        "registry_id": payload["registry_id"],
        "version": payload["version"],
        "frozen_at_utc": payload["frozen_at_utc"],
        "entries": entries,
        "index": index,
    }


def _row_atom(row: Mapping[str, Any], field: str) -> str:
    direct = _text(row.get(field))
    if direct:
        return direct
    canonical = row.get("canonical_provenance")
    if not isinstance(canonical, dict):
        return ""
    if field in ("region", "library", "source_asset"):
        direct = _text(canonical.get(field))
        if direct:
            return direct
        metadata = canonical.get("metadata")
        if isinstance(metadata, dict):
            aliases = {
                "library": ("library",),
                "source_asset": ("source_file", "source_asset"),
                "region": ("region",),
            }
            for alias in aliases[field]:
                value = _text(metadata.get(alias))
                if value:
                    return value
    return ""


def _bind_context(row: dict, crosswalk: dict) -> Tuple[Optional[dict], List[str]]:
    atoms = {field: _row_atom(row, field) for field in CROSSWALK_ATOMS}
    missing = sorted(field for field, value in atoms.items() if not value)
    if missing:
        return None, [
            "MISSING_CONTEXT_ATOM:%s" % field for field in missing
        ]
    key = tuple(atoms[field] for field in CROSSWALK_ATOMS)
    entry = crosswalk["index"].get(key)
    if entry is None:
        return None, ["NO_FROZEN_CONTEXT_BINDING"]
    publication_context = _text(entry["publication_context_id"])
    if _is_study_sentinel(publication_context, atoms["study"]):
        return None, ["SENTINEL_PUBLICATION_CONTEXT_REJECTED"]
    pair_provenance = row.get("pair_provenance")
    row_relation_evidence_id = (
        _text(pair_provenance.get("evidence_id"))
        if isinstance(pair_provenance, dict) else ""
    )
    row_relation_evidence_sha256 = (
        _text(pair_provenance.get("evidence_sha256")).lower()
        if isinstance(pair_provenance, dict) else ""
    )
    row_relation_status = (
        _text(pair_provenance.get("relation_status"))
        if isinstance(pair_provenance, dict) else ""
    )
    if not row_relation_evidence_id:
        return None, ["MISSING_ROW_SOURCE_RELATION_EVIDENCE"]
    if not re.fullmatch(r"[0-9a-f]{64}", row_relation_evidence_sha256):
        return None, ["MISSING_ROW_SOURCE_RELATION_EVIDENCE_HASH"]
    if row_relation_status != "ACCEPTED":
        return None, ["ROW_SOURCE_RELATION_NOT_ACCEPTED"]
    allowed_relation_map = {
        relation["evidence_id"]: relation["evidence_sha256"]
        for relation in entry.get("allowed_row_relation_evidence", [])
    }
    if allowed_relation_map.get(
        row_relation_evidence_id
    ) != row_relation_evidence_sha256:
        return None, ["ROW_SOURCE_RELATION_EVIDENCE_NOT_BOUND"]
    return {
        "context_id": publication_context,
        "assay_context": publication_context,
        "context_binding": {
            "schema_version": CROSSWALK_SCHEMA_VERSION,
            "crosswalk_id": crosswalk["crosswalk_id"],
            "crosswalk_version": crosswalk["version"],
            "crosswalk_sha256": crosswalk["sha256"],
            "context_evidence_id": entry["context_evidence_id"],
            "source_relation_evidence_id": entry[
                "source_relation_evidence_id"
            ],
            "row_relation_evidence_id": row_relation_evidence_id,
            "row_relation_evidence_sha256": row_relation_evidence_sha256,
            "row_relation_status": row_relation_status,
            "allowed_row_relation_evidence_sha256": _json_sha256(
                entry["allowed_row_relation_evidence"]
            ),
            "matched_atoms": atoms,
        },
    }, []


def _functional_observation_reasons(row: Mapping[str, Any]) -> List[str]:
    observations = row.get("functional_observations")
    if not isinstance(observations, list) or not observations:
        return ["MISSING_FUNCTIONAL_OBSERVATIONS"]
    reasons = []
    seen_slots: Dict[Tuple[str, ...], set] = defaultdict(set)
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            reasons.append("INVALID_FUNCTIONAL_OBSERVATION_OBJECT")
            continue
        try:
            value = float(observation.get("value"))
        except (TypeError, ValueError):
            reasons.append("NONNUMERIC_FUNCTIONAL_OBSERVATION")
            continue
        if not math.isfinite(value):
            reasons.append("NONFINITE_FUNCTIONAL_OBSERVATION")
        row_hash = _text(observation.get("input_row_sha256"))
        if not re.fullmatch(r"[0-9a-f]{64}", row_hash):
            reasons.append("MISSING_FUNCTIONAL_OBSERVATION_PROVENANCE_HASH")
        replicate = observation.get("replicate")
        if replicate is None:
            replicate = observation.get("replicate_id")
        slot = (
            _text(observation.get("sequence_id")),
            _text(observation.get("endpoint_id")),
            _text(observation.get("context_id")),
            _text(observation.get("assay_context")),
            _canonical_json(replicate),
            _text(observation.get("unit")),
        )
        seen_slots[slot].add(value)
    if any(len(values) > 1 for values in seen_slots.values()):
        reasons.append("CONFLICTING_FUNCTIONAL_OBSERVATION_VALUES")
    return sorted(set(reasons))


def _replay_substitutions(source: str, edits: Sequence[dict]) -> str:
    sequence = list(source)
    for edit in edits:
        sequence[int(edit["pos"])] = _normalize_sequence(edit["token"])
    return "".join(sequence)


def _edit_repair(row: Mapping[str, Any]) -> dict:
    """Verify the existing SUB script or derive only the narrow safe case."""
    source = _normalize_sequence(row.get("source_sequence"))
    candidate = _normalize_sequence(row.get("candidate_sequence"))
    original = row.get("edit_list")
    result = {
        "status": "HOLD",
        "reason": None,
        "source_sequence": source,
        "candidate_sequence": candidate,
        "edit_list": copy.deepcopy(original),
        "observed_edit_count": None,
        "edit_repair": None,
    }
    if not source or not candidate:
        result["reason"] = "EMPTY_SEQUENCE"
        return result
    if len(source) != len(candidate):
        result["reason"] = "UNEQUAL_LENGTH"
        return result
    if set(source) - set("ACGU") or set(candidate) - set("ACGU"):
        result["reason"] = "NON_ACGU_SEQUENCE"
        return result
    if not isinstance(original, list):
        result["reason"] = "INVALID_EDIT_SCRIPT"
        return result
    if any(
        not isinstance(edit, dict) or edit.get("op") != "SUB"
        for edit in original
    ):
        result["reason"] = "NON_SUB_EDIT"
        return result
    actual_positions = [
        index for index, (src, cand) in enumerate(zip(source, candidate))
        if src != cand
    ]
    result["observed_edit_count"] = len(actual_positions)
    try:
        encode_pair(source, candidate, original)
    except IncompatibleEditEncodingError as exc:
        message = str(exc)
        if "does not exactly match source/candidate differences" not in message:
            result["reason"] = "UNREPAIRABLE_EDIT_SCRIPT:%s" % message
            return result
        if original != []:
            result["reason"] = (
                "NONEMPTY_MISMATCHED_SUB_SCRIPT_NOT_REPAIRABLE"
            )
            return result
    else:
        result["status"] = "VERIFIED_EXISTING_SUB_SCRIPT"
        result["edit_list"] = copy.deepcopy(original)
        return result

    if len(actual_positions) not in REPAIRABLE_EDIT_COUNTS:
        result["reason"] = "INCOMPLETE_EDIT_SCRIPT_OUTSIDE_REPAIR_COUNT"
        return result
    span = actual_positions[-1] - actual_positions[0]
    if span > MAX_REPAIR_SPAN:
        result["reason"] = "INCOMPLETE_EDIT_SCRIPT_SPAN_EXCEEDS_2"
        return result
    derived = [
        {"op": "SUB", "pos": position, "token": candidate[position]}
        for position in actual_positions
    ]
    if _replay_substitutions(source, derived) != candidate:
        result["reason"] = "DERIVED_SUB_REPLAY_MISMATCH"
        return result
    encode_pair(source, candidate, derived)
    result["status"] = EDIT_REPAIR_METHOD
    result["edit_list"] = derived
    result["edit_repair"] = {
        "method": EDIT_REPAIR_METHOD,
        "reason": "INCOMPLETE_EDIT_SCRIPT",
        "original_edit_list": copy.deepcopy(original),
        "original_edit_list_sha256": _json_sha256(original),
        "derived_edit_list_sha256": _json_sha256(derived),
        "source_sequence_sha256": hashlib.sha256(
            source.encode("ascii")
        ).hexdigest(),
        "candidate_sequence_sha256": hashlib.sha256(
            candidate.encode("ascii")
        ).hexdigest(),
        "observed_difference_positions": actual_positions,
        "observed_difference_count": len(actual_positions),
        "observed_difference_span": span,
        "sub_replay_exact": True,
    }
    return result


def _budget_disposition(observed_edit_count: Optional[int]) -> str:
    if observed_edit_count in PRIMARY_EDIT_BUDGETS:
        return "PRIMARY_PREREGISTERED_BUDGET_%d" % observed_edit_count
    if observed_edit_count is not None and (
        observed_edit_count in (2, 4) or observed_edit_count > 5
    ):
        return "DEVELOPMENT_OUT_OF_CONTRACT"
    return "HOLD_OUTSIDE_PREREGISTERED_BUDGET"


def _finite_label_reasons(row: Mapping[str, Any]) -> List[str]:
    reasons = []
    for field in ("candidate_value", "delta"):
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            reasons.append("MISSING_OR_NONNUMERIC_%s" % field.upper())
            continue
        if not math.isfinite(value):
            reasons.append("NONFINITE_%s" % field.upper())
    return reasons


def _apply_estimand(row: dict, registry: Optional[dict]) -> List[str]:
    """Bind and, only when explicit, derive the registered endpoint target."""
    study = _text(row.get("study"))
    endpoint = _text(row.get("endpoint"))
    row["raw_source_value"] = row.get("source_value")
    row["raw_candidate_value"] = row.get("candidate_value")
    row["raw_delta"] = row.get("delta")
    entry = (
        registry["index"].get((study, endpoint))
        if registry is not None else None
    )
    if entry is None:
        row["estimand_status"] = "HOLD_MISSING_FROZEN_REGISTRY"
        return ["MISSING_FROZEN_ENDPOINT_ESTIMAND"]

    if entry.get("role") == "PRIMARY" and entry.get(
        "transform"
    ) != "IDENTITY_DIFFERENCE":
        row["estimand_status"] = "HOLD_INVALID_PRIMARY_TRANSFORM"
        return ["PRIMARY_ESTIMAND_MUST_PRESERVE_RAW_ENDPOINT"]
    if entry.get("transform") in (
        "NATURAL_LOG_RATIO", "DECAY_RATE_DIFFERENCE"
    ) and entry.get("role") != "SECONDARY_SENSITIVITY":
        row["estimand_status"] = "HOLD_INVALID_TRANSFORM_ROLE"
        return ["TRANSFORM_MUST_BE_SECONDARY_SENSITIVITY"]
    if study == "GSE217518" and entry.get("role") == "PRIMARY" and entry.get(
        "direction"
    ) != "HIGHER_IS_MORE_STABLE":
        row["estimand_status"] = "HOLD_INVALID_HALF_LIFE_DIRECTION"
        return ["GSE217518_PRIMARY_DIRECTION_MUST_BE_HIGHER_IS_MORE_STABLE"]

    try:
        source = float(row.get(entry["raw_source_field"]))
        candidate = float(row.get(entry["raw_candidate_field"]))
    except (TypeError, ValueError):
        row["estimand_status"] = "HOLD_INVALID_RAW_VALUE"
        row["delta"] = None
        return ["ENDPOINT_ESTIMAND_INVALID_RAW_VALUE"]
    if not math.isfinite(source) or not math.isfinite(candidate):
        row["estimand_status"] = "HOLD_INVALID_RAW_VALUE"
        row["delta"] = None
        return ["ENDPOINT_ESTIMAND_NONFINITE_RAW_VALUE"]
    transform = entry["transform"]
    domain = entry["valid_domain"]
    if domain == "STRICTLY_POSITIVE" and (
        source <= 0.0 or candidate <= 0.0
    ):
        row["estimand_status"] = "HOLD_OUTSIDE_VALID_DOMAIN"
        row["delta"] = None
        return ["ENDPOINT_ESTIMAND_REQUIRES_POSITIVE_VALUES"]
    if transform in ("NATURAL_LOG_RATIO", "DECAY_RATE_DIFFERENCE") and (
        source <= 0.0 or candidate <= 0.0
    ):
        row["estimand_status"] = "HOLD_OUTSIDE_TRANSFORM_DOMAIN"
        row["delta"] = None
        return ["ENDPOINT_TRANSFORM_REQUIRES_POSITIVE_VALUES"]

    if transform == "IDENTITY_DIFFERENCE":
        raw_effect = candidate - source
        formula = "candidate_value - source_value"
    elif transform == "NATURAL_LOG_RATIO":
        raw_effect = math.log(candidate / source)
        formula = "ln(candidate_value / source_value)"
    elif transform == "DECAY_RATE_DIFFERENCE":
        raw_effect = math.log(2.0) / candidate - math.log(2.0) / source
        formula = "ln(2)/candidate_value - ln(2)/source_value"
    else:  # Registry validation makes this unreachable; retain fail closed.
        row["estimand_status"] = "HOLD_UNKNOWN_TRANSFORM"
        row["delta"] = None
        return ["UNKNOWN_ENDPOINT_ESTIMAND_TRANSFORM"]

    direction = entry["direction"]
    if direction == "HIGHER_IS_MORE_STABLE":
        multiplier = -1.0 if transform == "DECAY_RATE_DIFFERENCE" else 1.0
    elif direction == "LOWER_IS_MORE_STABLE":
        multiplier = 1.0 if transform == "DECAY_RATE_DIFFERENCE" else -1.0
    elif direction == "SIGNED_EFFECT_AS_REPORTED":
        multiplier = 1.0
    else:
        row["estimand_status"] = "HOLD_INVALID_DIRECTION"
        row["delta"] = None
        return ["INVALID_ENDPOINT_ESTIMAND_DIRECTION"]
    derived = multiplier * raw_effect
    if not math.isfinite(derived):
        row["estimand_status"] = "HOLD_NONFINITE_DERIVED_TARGET"
        row["delta"] = None
        return ["NONFINITE_DERIVED_ENDPOINT_ESTIMAND"]
    row["delta"] = derived
    row["estimand_status"] = "FROZEN_REGISTRY_APPLIED"
    row["estimand_binding"] = {
        "schema_version": ESTIMAND_SCHEMA_VERSION,
        "registry_id": registry["registry_id"],
        "registry_version": registry["version"],
        "registry_sha256": registry["sha256"],
        "study": study,
        "endpoint": endpoint,
        "raw_source_field": entry["raw_source_field"],
        "raw_candidate_field": entry["raw_candidate_field"],
        "raw_unit": entry["raw_unit"],
        "direction": direction,
        "transform": transform,
        "valid_domain": domain,
        "role": entry["role"],
        "replicate_policy": entry["replicate_policy"],
        "source_citation": entry["source_citation"],
        "source_evidence_sha256": entry["source_evidence_sha256"],
        "formula": formula,
        "direction_multiplier": multiplier,
        "raw_transform_value": raw_effect,
        "derived_delta": derived,
    }
    if entry["role"] == "SECONDARY_SENSITIVITY":
        return ["ENDPOINT_ESTIMAND_SECONDARY_SENSITIVITY_ONLY"]
    if entry["role"] != "PRIMARY":
        return ["ENDPOINT_ESTIMAND_DEVELOPMENT_ONLY"]
    return []


def _source_group_id(row: Mapping[str, Any]) -> Tuple[Optional[str], List[str]]:
    source = _normalize_sequence(row.get("source_sequence"))
    atoms = {
        "study": _text(row.get("study")),
        "publication_context": _text(row.get("context_id")),
        "library": _row_atom(row, "library"),
        "source_asset": _row_atom(row, "source_asset"),
        "source_anchor_rule": _text(row.get("source_anchor_rule")),
        "source_anchor_identity": _text(row.get("source_anchor_identity")),
        "normalized_source_sequence_sha256": (
            hashlib.sha256(source.encode("ascii")).hexdigest()
            if source else ""
        ),
    }
    missing = sorted(field for field, value in atoms.items() if not value)
    if missing:
        return None, [
            "MISSING_SOURCE_GROUP_ATOM:%s" % field for field in missing
        ]
    return "%s:%s" % (SOURCE_GROUP_VERSION, _json_sha256(atoms)), []


def _group_report(rows: Sequence[dict], field: str) -> dict:
    groups: Dict[str, List[dict]] = defaultdict(list)
    missing = 0
    for row in rows:
        key = _text(row.get(field))
        if not key:
            missing += 1
            continue
        groups[key].append(row)
    row_repeat_groups = 0
    row_repeat_rows = 0
    multi = 0
    multi_rows = 0
    conflict_free_multi = 0
    conflict_free_multi_rows = 0
    max_candidates = 0
    conflicts = 0
    missing_or_nonfinite = 0
    for group_rows in groups.values():
        if len(group_rows) >= 2:
            row_repeat_groups += 1
            row_repeat_rows += len(group_rows)
        candidates = {
            _normalize_sequence(row.get("candidate_sequence"))
            for row in group_rows
            if _normalize_sequence(row.get("candidate_sequence"))
        }
        max_candidates = max(max_candidates, len(candidates))
        if len(candidates) >= 2:
            multi += 1
            multi_rows += len(group_rows)
        source_values = set()
        invalid_source_value = False
        for row in group_rows:
            value = row.get("source_value")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                invalid_source_value = True
                continue
            if math.isfinite(numeric):
                source_values.add(numeric)
            else:
                invalid_source_value = True
        if invalid_source_value:
            missing_or_nonfinite += 1
        if len(source_values) > 1:
            conflicts += 1
        elif len(candidates) >= 2 and not invalid_source_value:
            conflict_free_multi += 1
            conflict_free_multi_rows += len(group_rows)
    return {
        "group_field": field,
        "n_groups": len(groups),
        "n_rows_missing_group": missing,
        "n_groups_with_row_repeats": row_repeat_groups,
        "n_rows_in_row_repeat_groups": row_repeat_rows,
        "n_groups_with_at_least_2_distinct_candidates": multi,
        "n_rows_in_multi_candidate_groups": multi_rows,
        "n_conflict_free_distinct_candidate_groups": conflict_free_multi,
        "n_rows_in_conflict_free_distinct_candidate_groups": (
            conflict_free_multi_rows
        ),
        "max_distinct_candidates": max_candidates,
        "n_groups_with_conflicting_source_value": conflicts,
        "n_groups_with_missing_or_nonfinite_source_value": (
            missing_or_nonfinite
        ),
        "publication_qualified": False,
        "qualification_note": (
            "Row repeats are not candidate diversity. Distinct-candidate, "
            "finite-source and conflict-free counts remain non-publication "
            "until all frozen context, relation and endpoint gates pass."
        ),
    }


def prepare_rows(
    rows: Sequence[dict],
    crosswalk: dict,
    estimand_registry: Optional[dict] = None,
) -> Tuple[List[dict], List[dict], dict]:
    """Prepare deterministic ready/HOLD partitions from ordinary rows."""
    assert_ordinary_rows(rows, purpose="R3 ordinary preparation input")
    prepared_rows = []
    input_hashes = set()
    context_counts = Counter()
    repair_counts = Counter()
    budget_counts = Counter()
    estimand_counts = Counter()
    for raw_row in rows:
        row = copy.deepcopy(raw_row)
        input_hash = _json_sha256(raw_row)
        if input_hash in input_hashes:
            raise ValueError("duplicate ordinary input row hash: %s" % input_hash)
        input_hashes.add(input_hash)
        row["r3_input_row_sha256"] = input_hash
        row["input_context_aliases"] = {
            "context_id": raw_row.get("context_id"),
            "assay_context": raw_row.get("assay_context"),
        }
        row.pop("context_id", None)
        row.pop("assay_context", None)
        reasons = []

        observation_reasons = _functional_observation_reasons(row)
        reasons.extend(observation_reasons)

        binding, context_reasons = _bind_context(row, crosswalk)
        reasons.extend(context_reasons)
        if binding is not None:
            row.update(binding)
            context_counts["BOUND_FROZEN_CROSSWALK"] += 1
        else:
            context_counts["HOLD"] += 1

        edit = _edit_repair(row)
        repair_counts[edit["status"] if edit["status"] != "HOLD" else edit["reason"]] += 1
        row["normalized_source_sequence"] = edit["source_sequence"]
        row["normalized_candidate_sequence"] = edit["candidate_sequence"]
        row["observed_edit_count"] = edit["observed_edit_count"]
        if edit["status"] == "HOLD":
            reasons.append(edit["reason"] or "UNCLASSIFIED_EDIT_HOLD")
        else:
            row["edit_list"] = edit["edit_list"]
            row["edit_count"] = edit["observed_edit_count"]
            row["edit_script_status"] = edit["status"]
            if edit["edit_repair"] is not None:
                row["edit_repair"] = edit["edit_repair"]

        budget = _budget_disposition(edit["observed_edit_count"])
        row["edit_budget_disposition"] = budget
        budget_counts[budget] += 1
        if not budget.startswith("PRIMARY_PREREGISTERED_BUDGET_"):
            reasons.append(budget)

        estimand_reasons = _apply_estimand(row, estimand_registry)
        reasons.extend(estimand_reasons)
        estimand_counts[row.get("estimand_status") or "MISSING"] += 1
        reasons.extend(_finite_label_reasons(row))
        try:
            finite_source_value = math.isfinite(float(row.get("source_value")))
        except (TypeError, ValueError):
            finite_source_value = False
        if not finite_source_value:
            reasons.append("MISSING_OR_NONFINITE_SOURCE_VALUE_FOR_PRIMARY_POOL")
        anchor_status = _text(row.get("source_anchor_status"))
        if anchor_status != "MEASURED_SINGLE_VALUE":
            reasons.append("UNQUALIFIED_SOURCE_ANCHOR_STATUS:%s" % (
                anchor_status or "MISSING"
            ))
        if not isinstance(row.get("source_anchor_evidence"), list) or not row.get("source_anchor_evidence"):
            reasons.append("MISSING_SOURCE_ANCHOR_EVIDENCE")

        source_group_id = None
        if binding is not None:
            source_group_id, source_group_reasons = _source_group_id(row)
            reasons.extend(source_group_reasons)
        if source_group_id is not None:
            row["source_group_id_v2"] = source_group_id
            row["source_group_definition_version"] = SOURCE_GROUP_VERSION
            row["legacy_source_id"] = (
                row.get("legacy_source_id") or row.get("source_id")
            )
            row["source_id"] = source_group_id
            row["source_id_status"] = "FROZEN_CONTEXT_BOUND_SOURCE_GROUP_V2"

        row["r3_hold_reasons"] = sorted(set(reason for reason in reasons if reason))
        prepared_rows.append(row)

    # Replicates are not independent primary rows. Until a frozen registry and
    # a bound downstream hierarchical/aggregation implementation exist, every
    # multi-replicate pair+endpoint+context group remains out of primary.
    replicate_groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in prepared_rows:
        group_key = (
            _text(row.get("pair_id") or row.get("record_id")),
            _text(row.get("endpoint")),
            _text(row.get("context_id")),
        )
        if all(group_key):
            replicate_groups[group_key].append(row)
    multi_replicate_group_keys = set()
    multi_replicate_row_count = 0
    for group_key, group_rows in replicate_groups.items():
        replicate_tokens = set()
        for row in group_rows:
            observations = row.get("functional_observations") or []
            row_explicit_replicates = set()
            for observation in observations:
                replicate = observation.get("replicate")
                if replicate is None:
                    replicate = observation.get("replicate_id")
                if replicate is not None:
                    row_explicit_replicates.add(
                        "replicate:%s" % _canonical_json(replicate)
                    )
            if row_explicit_replicates:
                replicate_tokens.update(row_explicit_replicates)
            else:
                # Ref/alt component observations within one effect row are
                # not replicates merely because their observation ids differ.
                # Separate unlabelled effect rows remain conservatively
                # distinct until a frozen replicate relation is available.
                replicate_tokens.add(
                    "unlabelled_effect_row:%s" % row["r3_input_row_sha256"]
                )
        if len(replicate_tokens) <= 1:
            continue
        multi_replicate_group_keys.add(group_key)
        multi_replicate_row_count += len(group_rows)
        for row in group_rows:
            binding = row.get("estimand_binding")
            policy = (
                binding.get("replicate_policy")
                if isinstance(binding, dict) else None
            )
            if policy == "PRESERVE_FOR_HIERARCHICAL_MODEL":
                reason = "HOLD_REQUIRES_REPLICATE_MODEL"
            elif policy == "PRE_REGISTERED_AGGREGATE_WITH_SE":
                reason = "HOLD_REQUIRES_REPLICATE_AGGREGATION_IMPLEMENTATION"
            elif policy == "SINGLE_OBSERVATION_ONLY":
                reason = "MULTI_REPLICATE_VIOLATES_SINGLE_OBSERVATION_POLICY"
            else:
                reason = "HOLD_MULTI_REPLICATE_WITHOUT_FROZEN_POLICY"
            row["r3_hold_reasons"] = sorted(set(
                row["r3_hold_reasons"] + [reason]
            ))

    # A value conflict is a group-level property. No member of a conflicting
    # source group may enter the primary file.
    by_v2: Dict[str, List[dict]] = defaultdict(list)
    for row in prepared_rows:
        if row.get("source_group_id_v2"):
            by_v2[row["source_group_id_v2"]].append(row)
    conflict_group_ids = set()
    for group_id, group_rows in by_v2.items():
        values = set()
        for row in group_rows:
            try:
                value = float(row.get("source_value"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.add(value)
        if len(values) > 1:
            conflict_group_ids.add(group_id)
    for row in prepared_rows:
        if row.get("source_group_id_v2") in conflict_group_ids:
            row["r3_hold_reasons"] = sorted(set(
                row["r3_hold_reasons"] + ["SOURCE_GROUP_VALUE_CONFLICT"]
            ))

    ready = []
    hold = []
    for row in prepared_rows:
        if not row["r3_hold_reasons"]:
            row["r3_disposition"] = (
                "PREREGISTERED_BUDGET_CANDIDATE_ORDINARY"
            )
            ready.append(row)
        else:
            budget_is_development = (
                row.get("edit_budget_disposition")
                == "DEVELOPMENT_OUT_OF_CONTRACT"
            )
            row["r3_disposition"] = (
                "DEVELOPMENT_OUT_OF_CONTRACT"
                if budget_is_development and row["r3_hold_reasons"] == [
                    "DEVELOPMENT_OUT_OF_CONTRACT"
                ]
                else "HOLD"
            )
            hold.append(row)
    ready = sorted(ready, key=lambda row: row["r3_input_row_sha256"])
    hold = sorted(hold, key=lambda row: row["r3_input_row_sha256"])
    assert_ordinary_rows(ready, purpose="R3 primary-ready output")
    assert_ordinary_rows(hold, purpose="R3 HOLD output")

    hold_reasons = Counter(
        reason for row in hold for reason in row["r3_hold_reasons"]
    )
    report = {
        "preparation_version": PREPARATION_VERSION,
        "status": (
            "ORDINARY_CANDIDATES_PREPARED_NOT_PUBLICATION_QUALIFIED"
            if ready else "HOLD_NO_PREREGISTERED_CANDIDATE_ROWS"
        ),
        "ordinary_only": True,
        "sealed_rows_seen": 0,
        "n_input_rows": len(rows),
        "n_primary_ready_rows": len(ready),
        "n_hold_rows": len(hold),
        "context_status_counts": dict(sorted(context_counts.items())),
        "edit_repair_status_counts": dict(sorted(repair_counts.items())),
        "edit_budget_disposition_counts": dict(sorted(budget_counts.items())),
        "endpoint_estimand_status_counts": dict(
            sorted(estimand_counts.items())
        ),
        "hold_reason_counts": dict(sorted(hold_reasons.items())),
        "primary_edit_budgets": list(PRIMARY_EDIT_BUDGETS),
        "legacy_source_groups": _group_report(
            prepared_rows, "legacy_source_id"
        ),
        "source_groups_v2": _group_report(
            prepared_rows, "source_group_id_v2"
        ),
        "primary_ready_source_groups_v2": _group_report(
            ready, "source_group_id_v2"
        ),
        "n_source_group_value_conflicts": len(conflict_group_ids),
        "replicate_groups": {
            "n_multi_replicate_groups": len(multi_replicate_group_keys),
            "n_rows_in_multi_replicate_groups": multi_replicate_row_count,
            "policy": (
                "No replicate averaging or independent-row weighting is "
                "performed by R3 preparation."
            ),
        },
    }
    report["primary_ready_source_groups_v2"]["publication_qualified"] = False
    report["primary_ready_source_groups_v2"]["development_availability"] = (
        "DEV_AVAILABLE" if ready else "NOT_AVAILABLE"
    )
    report["primary_ready_source_groups_v2"]["qualification_note"] = (
        "Preparation never confers publication qualification. Independent "
        "pre-registered minima and four-hash verification are still required."
    )
    return ready, hold, report


def _jsonl_bytes(rows: Iterable[dict]) -> bytes:
    return b"".join(
        (_canonical_json(row) + "\n").encode("utf-8") for row in rows
    )


def write_bundle_exclusive(
    out_dir: Path,
    ready: Sequence[dict],
    hold: Sequence[dict],
    report: dict,
    dataset_path: Path,
    crosswalk: dict,
    estimand_registry: Optional[dict] = None,
) -> dict:
    """Write ready/HOLD/manifest files without overwriting any prior bytes."""
    out_dir = Path(out_dir)
    assert_ordinary_path(out_dir, purpose="R3 output directory")
    paths = {
        "primary_ready": out_dir / READY_FILE,
        "hold": out_dir / HOLD_FILE,
        "manifest": out_dir / MANIFEST_FILE,
    }
    for path in paths.values():
        assert_ordinary_path(path, purpose="R3 output")
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "R3 preparation refuses to overwrite: %s" % ", ".join(existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    ready_bytes = _jsonl_bytes(ready)
    hold_bytes = _jsonl_bytes(hold)
    output_hashes = {
        "primary_ready": hashlib.sha256(ready_bytes).hexdigest(),
        "hold": hashlib.sha256(hold_bytes).hexdigest(),
    }
    manifest = dict(report)
    manifest.update({
        "input_dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "frozen_context_crosswalk": {
            "path": crosswalk["path"],
            "sha256": crosswalk["sha256"],
            "crosswalk_id": crosswalk["crosswalk_id"],
            "version": crosswalk["version"],
            "frozen_at_utc": crosswalk["frozen_at_utc"],
        },
        "outputs": {
            "primary_ready": {
                "path": str(paths["primary_ready"]),
                "sha256": output_hashes["primary_ready"],
                "n_rows": len(ready),
            },
            "hold": {
                "path": str(paths["hold"]),
                "sha256": output_hashes["hold"],
                "n_rows": len(hold),
            },
        },
    })
    manifest["frozen_endpoint_estimand_registry"] = (
        {
            "path": estimand_registry["path"],
            "sha256": estimand_registry["sha256"],
            "registry_id": estimand_registry["registry_id"],
            "version": estimand_registry["version"],
            "frozen_at_utc": estimand_registry["frozen_at_utc"],
        }
        if estimand_registry is not None else {
            "status": "NOT_PROVIDED",
            "effect": "ALL_PRIMARY_ROWS_HOLD",
        }
    )
    manifest_bytes = (
        json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")
    # Pre-render all bytes before the first exclusive create.
    with paths["primary_ready"].open("xb") as handle:
        handle.write(ready_bytes)
    with paths["hold"].open("xb") as handle:
        handle.write(hold_bytes)
    with paths["manifest"].open("xb") as handle:
        handle.write(manifest_bytes)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--crosswalk-sha256", required=True)
    parser.add_argument("--estimand-registry", type=Path)
    parser.add_argument("--estimand-registry-sha256")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    assert_ordinary_path(args.dataset, purpose="R3 ordinary dataset input")
    rows = load_ordinary_jsonl(args.dataset, require_delta=False)
    crosswalk = load_frozen_crosswalk(
        args.crosswalk, args.crosswalk_sha256
    )
    if bool(args.estimand_registry) != bool(args.estimand_registry_sha256):
        parser.error(
            "--estimand-registry and --estimand-registry-sha256 must be "
            "provided together"
        )
    estimand_registry = (
        load_frozen_estimand_registry(
            args.estimand_registry, args.estimand_registry_sha256
        )
        if args.estimand_registry else None
    )
    ready, hold, report = prepare_rows(
        rows, crosswalk, estimand_registry
    )
    manifest = write_bundle_exclusive(
        args.out_dir, ready, hold, report, args.dataset, crosswalk,
        estimand_registry,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
