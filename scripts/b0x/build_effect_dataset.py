"""B0-X effect dataset builder.

Loads the rebuilt D1 staging pairs + sequence entities + functional
observations and joins them with the read-only main-repo canonical records
(which carry the actual sequence text, edit scripts and candidate labels), then
derives the source-relative effect delta per asset per config.DELTA_SPECS.

No data is fabricated.  When a source anchor is genuinely not derivable, the
record is emitted with delta_source_status="source_anchor_unavailable" and a
null delta (excluded from delta-based ranking but present in the manifest).

Usage:
    python -m scripts.b0x.build_effect_dataset --out-dir artifacts/b0x
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from scripts.m4_sparse.ordinary_io import (  # noqa: E402
    assert_ordinary_path,
    assert_ordinary_rows,
)

CAND_SUFFIX = "__cand"
SRC_SUFFIX = "__src"
OBSERVATION_SCHEMA_VERSION = "lossless_functional_observations_v1"
EFFECT_DATASET_SCHEMA_VERSION = (
    "b0x_effect_row_v3_diff_ref_alt_independent_replay"
)
EFFECT_DATASET_MANIFEST_VERSION = (
    "b0x_effect_dataset_manifest_v3_diff_ref_alt_replay"
)
IMPLEMENTATION_BINDING_SCHEMA_VERSION = "b0x_implementation_binding_v1"
EFFECT_ROW_QUALIFICATION_STATUS = (
    "MECHANICAL_REPLAY_REQUIRES_MATCHING_V3_MANIFEST_SCIENTIFIC_HOLD"
)
MANIFEST_QUALIFICATION_STATUS = (
    "MECHANICAL_REPLAY_VERIFIED_WITH_MATCHING_V3_MANIFEST_SCIENTIFIC_HOLD"
)
DIFF_REF_ALT_JOIN_POLICY_SCHEMA_VERSION = (
    "diff_ref_alt_join_policy_v2_explicit_null_semantics"
)
DIFF_REF_ALT_JOIN_AUDIT_SCHEMA_VERSION = (
    "diff_ref_alt_join_audit_v3_independent_replay"
)
DIFF_REF_ALT_JOIN_KEY_FIELDS = (
    "context_id", "assay_context", "replicate",
)
DIFF_REF_ALT_JOIN_POLICY_KEYS = {
    "schema_version",
    "policy_id",
    "key_fields",
    "context_match",
    "context_required_non_null",
    "assay_context_match",
    "assay_context_null_policy",
    "replicate_match",
    "replicate_field",
    "replicate_null_policy",
    "ref_allowed_raw_unit_tokens",
    "alt_allowed_raw_unit_tokens",
    "comparison_scale_id",
    "raw_unit_semantics",
}
DIFF_REF_ALT_JOIN_AUDIT_KEYS = {
    "schema_version",
    "verification_status",
    "join_policy",
    "join_policy_sha256",
    "policy_id",
    "comparison_scale_id",
    "n_materialized_pairs",
    "n_output_pairs",
    "n_output_join_rows",
    "n_expected_missing_both_pairs",
    "n_consumed_missing_both_rows",
    "n_ref_observation_groups",
    "n_alt_observation_groups",
    "n_input_side_groups",
    "n_consumed_ref_observation_groups",
    "n_consumed_alt_observation_groups",
    "n_consumed_side_groups",
    "n_matched_groups",
    "n_unmatched_ref_groups",
    "n_unmatched_alt_groups",
    "n_missing_both_rows",
    "n_derived_rows",
    "n_unavailable_rows",
    "expected_output_rows_from_group_conservation",
    "join_disposition_counts",
    "delta_unavailable_reason_counts",
    "replicate_pairing_status_counts",
    "ref_raw_unit_token_group_counts",
    "alt_raw_unit_token_group_counts",
    "expected_input_side_group_multiset_sha256",
    "consumed_output_side_group_multiset_sha256",
    "expected_missing_both_multiset_sha256",
    "consumed_missing_both_multiset_sha256",
    "materialized_pair_registry_sha256",
    "output_pair_registry_sha256",
}
IMPLEMENTATION_BINDING_KEYS = {
    "schema_version",
    "build_effect_dataset_path",
    "build_effect_dataset_sha256",
    "config_path",
    "config_sha256",
    "delta_specs_sha256",
    "diff_ref_alt_policy_sha256_by_study",
    "effect_dataset_schema_version",
    "manifest_version",
    "diff_ref_alt_join_policy_schema_version",
    "diff_ref_alt_replay_audit_schema_version",
    "ordinary_only",
    "scientific_unit_authority_created",
    "estimand_authority_created",
}
DIFF_REF_ALT_RAW_UNIT_SEMANTICS = (
    "ENDPOINT_SPECIFIC_RAW_LABEL_TOKEN_NOT_PHYSICAL_UNIT"
)

# ``relation_acceptance_status`` is the canonical D1 authority.  The legacy
# ``relation_status`` field is retained only as an input consistency check; it
# must never promote a row whose canonical authority is absent.
RELATION_ACCEPTANCE_STATUS_MAP = {
    "CANDIDATE": "CANDIDATE",
    "ACCEPTED": "ACCEPTED",
    "AMBIGUOUS": "AMBIGUOUS",
    "REJECTED": "REJECTED",
}
RELATION_STATUS_AUTHORITY_FIELD = "relation_acceptance_status"
MISSING_RELATION_STATUS_SOURCE = "MISSING_NOT_ASSERTED"


class RelationStatusConflictError(ValueError):
    """Raised when canonical and legacy relation statuses disagree."""


def _study_of(seq_id: str) -> str:
    return seq_id.split("_")[0]


def load_pairs(pairs_path: Path) -> List[dict]:
    pairs_path = Path(pairs_path)
    assert_ordinary_path(pairs_path, purpose="pair input")
    pairs = []
    active = set(config.ACTIVE_STUDIES)
    for line in pairs_path.open(encoding="utf-8"):
        r = json.loads(line)
        assert_ordinary_rows([r], purpose="pair input")
        st = _study_of(r["candidate_sequence_id"])
        if st in active and r.get("scientific_track") == "E":
            pairs.append(r)
    return pairs


def load_canonical(records_path: Path, needed_record_ids: set) -> Dict[str, dict]:
    """Stream canonical records; keep only needed record_ids."""
    records_path = Path(records_path)
    assert_ordinary_path(records_path, purpose="canonical input")
    out: Dict[str, dict] = {}
    for line in records_path.open(encoding="utf-8"):
        r = json.loads(line)
        rid = r.get("record_id")
        if rid in needed_record_ids:
            assert_ordinary_rows([r], purpose="canonical input")
            out[rid] = r
    # Also load reconstructed pairs (rebuilt assets not in the pre-rebuild
    # main-repo canonical records file).
    for study, rp_path in config.RECONSTRUCTED_PAIRS.items():
        if not Path(rp_path).exists():
            continue
        assert_ordinary_path(Path(rp_path), purpose="reconstructed pair input")
        for line in Path(rp_path).open(encoding="utf-8"):
            r = json.loads(line)
            rid = r.get("record_id")
            if rid in needed_record_ids:
                assert_ordinary_rows(
                    [r], purpose="reconstructed pair input"
                )
                out[rid] = r
    return out


class ObservationConflictError(ValueError):
    """Raised when ordinary observation rows cannot be resolved losslessly."""


class DiffRefAltJoinPolicyError(ObservationConflictError):
    """Raised when an explicit ref/alt comparison policy cannot be replayed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DiffRefAltJoinPolicyError(
            "%s must be an original canonical non-empty string" % label
        )
    return value


def _canonical_token_list(value: Any, *, label: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise DiffRefAltJoinPolicyError(
            "%s must be a non-empty list" % label
        )
    tokens = []
    for raw in value:
        token = _optional_text(raw)
        if token is None or token != raw:
            raise DiffRefAltJoinPolicyError(
                "%s must contain canonical non-empty strings" % label
            )
        tokens.append(token)
    if tokens != sorted(set(tokens)):
        raise DiffRefAltJoinPolicyError(
            "%s must be sorted and duplicate-free" % label
        )
    return tokens


def validate_diff_ref_alt_join_policy(spec: Mapping[str, Any]) -> dict:
    """Validate the exact, versioned unit-compatibility authority for one pair.

    Raw ``unit`` values remain provenance.  They are never stripped from the
    global observation identity and never made comparable by endpoint-name
    heuristics.  A ref/alt subtraction is legal only when this explicit policy
    maps both raw unit-token allowlists to one registered comparison scale.
    """
    if not isinstance(spec, Mapping) or spec.get("type") != "diff_ref_alt":
        raise DiffRefAltJoinPolicyError(
            "diff_ref_alt join policy requires a diff_ref_alt spec"
        )
    raw = spec.get("join_policy")
    if not isinstance(raw, Mapping) or set(raw) != DIFF_REF_ALT_JOIN_POLICY_KEYS:
        actual = sorted(raw) if isinstance(raw, Mapping) else type(raw).__name__
        raise DiffRefAltJoinPolicyError(
            "diff_ref_alt join_policy fields differ from the exact schema: %s"
            % actual
        )
    policy = dict(raw)
    if policy["schema_version"] != DIFF_REF_ALT_JOIN_POLICY_SCHEMA_VERSION:
        raise DiffRefAltJoinPolicyError(
            "unsupported diff_ref_alt join policy schema"
        )
    for field in ("policy_id", "comparison_scale_id"):
        _canonical_nonempty_string(policy.get(field), label=field)
    ref_endpoint = _canonical_nonempty_string(
        spec.get("ref_endpoint"), label="ref_endpoint"
    )
    alt_endpoint = _canonical_nonempty_string(
        spec.get("alt_endpoint"), label="alt_endpoint"
    )
    if ref_endpoint == alt_endpoint:
        raise DiffRefAltJoinPolicyError(
            "diff_ref_alt ref_endpoint and alt_endpoint must differ"
        )
    if policy["key_fields"] != list(DIFF_REF_ALT_JOIN_KEY_FIELDS):
        raise DiffRefAltJoinPolicyError(
            "diff_ref_alt join key must be exact context/assay/replicate"
        )
    expected_exact = {
        "context_match": "EXACT",
        "assay_context_match": "EXACT",
        "replicate_match": "EXACT_CANONICAL_TOKEN_INCLUDING_NULL",
        "replicate_field": "replicate",
        "assay_context_null_policy": (
            "ALLOW_EXACT_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
        ),
        "replicate_null_policy": (
            "ALLOW_EXACT_NULL_NO_REPLICATE_CLAIM"
        ),
        "raw_unit_semantics": DIFF_REF_ALT_RAW_UNIT_SEMANTICS,
    }
    for field, expected in expected_exact.items():
        if policy.get(field) != expected:
            raise DiffRefAltJoinPolicyError(
                "diff_ref_alt %s must equal %s" % (field, expected)
            )
    if policy.get("context_required_non_null") is not True:
        raise DiffRefAltJoinPolicyError(
            "diff_ref_alt context_required_non_null must be true"
        )
    policy["ref_allowed_raw_unit_tokens"] = _canonical_token_list(
        policy["ref_allowed_raw_unit_tokens"],
        label="ref_allowed_raw_unit_tokens",
    )
    policy["alt_allowed_raw_unit_tokens"] = _canonical_token_list(
        policy["alt_allowed_raw_unit_tokens"],
        label="alt_allowed_raw_unit_tokens",
    )
    if set(policy["ref_allowed_raw_unit_tokens"]) & set(
        policy["alt_allowed_raw_unit_tokens"]
    ):
        raise DiffRefAltJoinPolicyError(
            "diff_ref_alt ref/alt raw-unit allowlists must be disjoint"
        )
    return policy


def validate_diff_ref_alt_policy_registry(
    specs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, dict]:
    """Validate cross-study uniqueness of all configured paired policies."""
    policies = {}
    policy_owners: Dict[str, str] = {}
    scale_owners: Dict[str, str] = {}
    for study in sorted(specs):
        spec = specs[study]
        if spec.get("type") != "diff_ref_alt":
            continue
        policy = validate_diff_ref_alt_join_policy(spec)
        for value, owners, label in (
            (policy["policy_id"], policy_owners, "policy_id"),
            (
                policy["comparison_scale_id"],
                scale_owners,
                "comparison_scale_id",
            ),
        ):
            previous = owners.get(value)
            if previous is not None and previous != study:
                raise DiffRefAltJoinPolicyError(
                    "diff_ref_alt %s %s is shared by %s and %s"
                    % (label, value, previous, study)
                )
            owners[value] = study
        policies[study] = policy
    return policies


def build_pair_provenance(pair: Mapping[str, Any]) -> dict:
    """Resolve pair relation status from the canonical D1 authority.

    The original pair fields remain present in the returned provenance.  A
    legacy ``relation_status`` value is copied to
    ``legacy_relation_status_input`` before the canonical output field is
    written.  When the canonical authority is absent, the output remains
    ``UNREVIEWED`` even if the legacy value says ``ACCEPTED``.
    """
    raw_canonical = pair.get(RELATION_STATUS_AUTHORITY_FIELD)
    if raw_canonical is None:
        canonical = None
    elif (
        not isinstance(raw_canonical, str)
        or not raw_canonical
        or raw_canonical.strip() != raw_canonical
    ):
        raise ValueError(
            "canonical relation_acceptance_status must be an original "
            "trim-equal non-empty string"
        )
    else:
        canonical = raw_canonical
    legacy = _optional_text(pair.get("relation_status"))

    if canonical is None:
        resolved = "UNREVIEWED"
        source_field = MISSING_RELATION_STATUS_SOURCE
    else:
        try:
            resolved = RELATION_ACCEPTANCE_STATUS_MAP[canonical]
        except KeyError:
            raise ValueError(
                "unsupported canonical relation_acceptance_status: %s"
                % canonical
            ) from None
        source_field = RELATION_STATUS_AUTHORITY_FIELD
        if legacy is not None and legacy != resolved:
            raise RelationStatusConflictError(
                "canonical relation_acceptance_status %s conflicts with "
                "legacy relation_status %s" % (canonical, legacy)
            )

    provenance = dict(pair)
    if legacy is not None:
        provenance["legacy_relation_status_input"] = legacy
    provenance["evidence_sha256"] = _json_sha256(pair)
    provenance["relation_status"] = resolved
    provenance["relation_status_source_field"] = source_field
    return provenance


def summarize_pair_relation_provenance(
    provenances: Sequence[Mapping[str, Any]],
) -> dict:
    """Return canonical, order-independent relation provenance counts."""
    statuses: Counter[str] = Counter()
    source_fields: Counter[str] = Counter()
    for provenance in provenances:
        status = _optional_text(provenance.get("relation_status"))
        source_field = _optional_text(
            provenance.get("relation_status_source_field")
        )
        if status is None or source_field is None:
            raise ValueError(
                "resolved pair provenance is missing relation status source"
            )
        statuses[status] += 1
        source_fields[source_field] += 1
    return {
        "relation_status_counts": dict(sorted(statuses.items())),
        "relation_status_source_field_counts": dict(
            sorted(source_fields.items())
        ),
    }


def _replicate_token(row: Mapping[str, Any]) -> str:
    value = row.get("replicate")
    if value is None:
        value = row.get("replicate_id")
    return _canonical_json(value)


def _observation_slot(row: Mapping[str, Any]) -> Tuple[str, ...]:
    """Identity at which more than one numeric value is a hard conflict.

    Observation ids are deliberately not part of this key: two rows with
    different ids but the same biological slot must not silently overwrite
    one another. Distinct, explicitly labelled replicates remain distinct.
    """
    return (
        str(row.get("sequence_id") or ""),
        str(row.get("endpoint_id") or ""),
        str(row.get("context_id") or ""),
        str(row.get("assay_context") or ""),
        _replicate_token(row),
        str(row.get("unit") or ""),
    )


def load_candidate_observations(
    obs_path: Path,
    cand_seq_ids: Dict[str, set],
) -> Dict[str, Dict[str, List[dict]]]:
    """Load candidate observations without endpoint-level reduction.

    Rows retain context, replicate, value, unit and provenance fields plus a
    stable row hash. Explicitly labelled replicates are preserved. Distinct
    values in the same biological slot are rejected rather than averaged or
    resolved by input order.
    """
    obs_path = Path(obs_path)
    assert_ordinary_path(obs_path, purpose="functional observation input")
    out: Dict[str, Dict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    values_by_slot: Dict[Tuple[str, ...], set] = defaultdict(set)
    rows_by_observation_id: Dict[str, str] = {}
    with obs_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ObservationConflictError(
                    "functional observation row %d is not an object"
                    % line_number
                )
            if "upstream_declared_input_row_sha256" in row:
                raise ObservationConflictError(
                    "functional observation row %d uses the builder-reserved "
                    "upstream hash namespace" % line_number
                )
            assert_ordinary_rows(
                [row], purpose="functional observation input"
            )
            sid = row.get("sequence_id")
            if not isinstance(sid, str) or not sid:
                raise ObservationConflictError(
                    "functional observation row %d has no sequence_id"
                    % line_number
                )
            study = _study_of(sid)
            if study not in cand_seq_ids or sid not in cand_seq_ids[study]:
                continue
            endpoint = row.get("endpoint_id")
            if not isinstance(endpoint, str) or not endpoint:
                raise ObservationConflictError(
                    "functional observation row %d has no endpoint_id"
                    % line_number
                )
            try:
                value = float(row.get("value"))
            except (TypeError, ValueError):
                raise ObservationConflictError(
                    "functional observation row %d has a non-numeric value"
                    % line_number
                ) from None
            if not math.isfinite(value):
                raise ObservationConflictError(
                    "functional observation row %d has a non-finite value"
                    % line_number
                )
            preserved = dict(row)
            upstream_declared_hash = preserved.pop(
                "input_row_sha256", None
            )
            if upstream_declared_hash is not None:
                # Never trust an input-supplied value as the builder's own
                # identity. Preserve it under a non-authoritative namespace,
                # then mint a fresh self-verifiable hash below.
                preserved["upstream_declared_input_row_sha256"] = (
                    upstream_declared_hash
                )
            preserved["value"] = value
            # The declared identity is self-verifiable from the exact
            # normalized preserved row after removing this reserved field.
            preserved["input_row_sha256"] = _json_sha256(preserved)
            obs_id = _optional_text(row.get("observation_id"))
            if obs_id:
                previous = rows_by_observation_id.get(obs_id)
                current = preserved["input_row_sha256"]
                if previous is not None and previous != current:
                    raise ObservationConflictError(
                        "observation_id %s maps to conflicting rows" % obs_id
                    )
                rows_by_observation_id[obs_id] = current
            slot = _observation_slot(row)
            values_by_slot[slot].add(value)
            if len(values_by_slot[slot]) > 1:
                raise ObservationConflictError(
                    "multiple values for functional observation slot %s"
                    % (slot,)
                )
            out[sid][endpoint].append(preserved)
    result: Dict[str, Dict[str, List[dict]]] = {}
    for sid in sorted(out):
        result[sid] = {}
        for endpoint in sorted(out[sid]):
            result[sid][endpoint] = sorted(
                out[sid][endpoint], key=_canonical_json
            )
    return result


# Compatibility name retained for callers. Unlike the historical function it
# returns lossless observation lists, never endpoint -> scalar reduction.
load_candidate_values = load_candidate_observations


def _observation_join_key(
    row: Mapping[str, Any],
) -> Tuple[str, str, str, str]:
    return (
        str(row.get("context_id") or ""),
        str(row.get("assay_context") or ""),
        _replicate_token(row),
        str(row.get("unit") or ""),
    )


def _observation_groups(
    index: Mapping[str, Mapping[str, Any]], sid: str, endpoint: str,
) -> Dict[Tuple[str, str, str, str], dict]:
    raw = index.get(sid, {}).get(endpoint, [])
    if isinstance(raw, (int, float)):
        raw = [{
            "sequence_id": sid,
            "endpoint_id": endpoint,
            "value": float(raw),
        }]
    elif isinstance(raw, dict):
        raw = [raw]
    groups: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for row in raw or []:
        groups[_observation_join_key(row)].append(dict(row))
    resolved = {}
    for key in sorted(groups):
        rows = sorted(groups[key], key=_canonical_json)
        values = {float(row["value"]) for row in rows}
        if len(values) != 1:
            raise ObservationConflictError(
                "unresolved multiple values for %s/%s at %s"
                % (sid, endpoint, key)
            )
        resolved[key] = {"value": next(iter(values)), "rows": rows}
    return resolved


def _paired_observation_join_key(
    row: Mapping[str, Any], policy: Mapping[str, Any],
) -> Tuple[str, str, str, str]:
    """Return the exact paired key after explicit unit-scale qualification.

    The raw unit token is deliberately absent from this derived key because it
    has already been checked against a side-specific allowlist.  The registered
    comparison-scale identity replaces it; the global observation key above is
    unchanged for every other caller.
    """
    for field in ("context_id", "assay_context", "replicate"):
        if field not in row:
            raise DiffRefAltJoinPolicyError(
                "paired observation must explicitly carry %s" % field
            )
    context_id = row["context_id"]
    if context_id is None:
        raise DiffRefAltJoinPolicyError(
            "paired observation context_id cannot be null under policy"
        )
    _canonical_nonempty_string(context_id, label="context_id")
    assay_context = row["assay_context"]
    if assay_context is not None:
        _canonical_nonempty_string(
            assay_context, label="assay_context"
        )
    elif policy["assay_context_null_policy"] != (
        "ALLOW_EXACT_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
    ):
        raise DiffRefAltJoinPolicyError(
            "null assay_context is not allowed by paired policy"
        )
    replicate = row["replicate"]
    if replicate is None and policy["replicate_null_policy"] != (
        "ALLOW_EXACT_NULL_NO_REPLICATE_CLAIM"
    ):
        raise DiffRefAltJoinPolicyError(
            "null replicate is not allowed by paired policy"
        )
    return (
        _canonical_json(context_id),
        _canonical_json(assay_context),
        _canonical_json(replicate),
        str(policy["comparison_scale_id"]),
    )


def _paired_observation_groups(
    index: Mapping[str, Mapping[str, Any]],
    sid: str,
    endpoint: str,
    *,
    side: str,
    policy: Mapping[str, Any],
) -> Dict[Tuple[str, str, str, str], dict]:
    """Group one ref/alt side under an explicit raw-unit compatibility policy."""
    if side not in {"ref", "alt"}:
        raise DiffRefAltJoinPolicyError("paired observation side is invalid")
    raw = index.get(sid, {}).get(endpoint, [])
    if isinstance(raw, (int, float)):
        raise DiffRefAltJoinPolicyError(
            "diff_ref_alt requires structured observations with raw unit provenance"
        )
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, list):
        raise DiffRefAltJoinPolicyError(
            "diff_ref_alt endpoint observations must be a list"
        )
    allowed_field = "%s_allowed_raw_unit_tokens" % side
    allowed = set(policy[allowed_field])
    groups: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for row_index, raw_row in enumerate(raw):
        if not isinstance(raw_row, Mapping):
            raise DiffRefAltJoinPolicyError(
                "diff_ref_alt observation %s/%s row %d is not an object"
                % (sid, endpoint, row_index)
            )
        row = dict(raw_row)
        declared_sid = row.get("sequence_id")
        if declared_sid not in (None, sid):
            raise DiffRefAltJoinPolicyError(
                "diff_ref_alt observation sequence_id disagrees with its index"
            )
        declared_endpoint = row.get("endpoint_id")
        if declared_endpoint not in (None, endpoint):
            raise DiffRefAltJoinPolicyError(
                "diff_ref_alt observation endpoint_id disagrees with its index"
            )
        raw_unit = row.get("unit")
        if not isinstance(raw_unit, str) or raw_unit not in allowed:
            raise DiffRefAltJoinPolicyError(
                "unregistered %s raw unit token for %s/%s: %r"
                % (side, sid, endpoint, raw_unit)
            )
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            raise ObservationConflictError(
                "non-numeric paired observation for %s/%s" % (sid, endpoint)
            ) from None
        if not math.isfinite(value):
            raise ObservationConflictError(
                "non-finite paired observation for %s/%s" % (sid, endpoint)
            )
        row["value"] = value
        groups[_paired_observation_join_key(row, policy)].append(row)

    resolved = {}
    for key in sorted(groups):
        rows = sorted(groups[key], key=_canonical_json)
        values = {float(row["value"]) for row in rows}
        if len(values) != 1:
            raise ObservationConflictError(
                "unresolved multiple %s values for %s/%s at %s"
                % (side, sid, endpoint, key)
            )
        resolved[key] = {
            "value": next(iter(values)),
            "rows": rows,
            "raw_unit_tokens": sorted({str(row["unit"]) for row in rows}),
        }
    return resolved


def _unmatched_ref_alt_reason(
    side: str,
    key: Tuple[str, str, str, str],
    opposite_keys: Sequence[Tuple[str, str, str, str]],
) -> str:
    """Classify one unpaired slot without relaxing exact pairing."""
    missing = "MISSING_ALT_ENDPOINT" if side == "ref" else "MISSING_REF_ENDPOINT"
    if not opposite_keys:
        return missing
    context, assay_context, replicate_token, comparison_scale_id = key
    same_context = [item for item in opposite_keys if item[0] == context]
    if not same_context:
        return "REF_ALT_CONTEXT_ID_MISMATCH"
    same_assay = [
        item for item in same_context if item[1] == assay_context
    ]
    if not same_assay:
        return "REF_ALT_ASSAY_CONTEXT_MISMATCH"
    same_replicate = [
        item for item in same_assay if item[2] == replicate_token
    ]
    if not same_replicate:
        return "REF_ALT_REPLICATE_MISMATCH"
    if not any(item[3] == comparison_scale_id for item in same_replicate):
        return "REF_ALT_COMPARISON_SCALE_MISMATCH"
    return missing


def _replicate_pairing_status(replicate_token: str) -> str:
    if replicate_token == "null":
        return "EXACT_NULL_SLOT_NO_REPLICATE_CLAIM"
    return "EXACT_EXPLICIT_REPLICATE_TOKEN"


def _normalize_sequence(value: Any) -> str:
    return str(value or "").strip().upper().replace("T", "U")


def _source_anchor_id(study: str, anchor_kind: str, raw_identity: Any) -> str:
    payload = {
        "study": study,
        "anchor_kind": anchor_kind,
        "raw_identity": raw_identity,
    }
    return "%s:%s" % (anchor_kind, _json_sha256(payload))


def _resolved_anchor(
    study: str,
    anchor_kind: str,
    raw_identity: Any,
    values: Sequence[float],
    evidence: Sequence[dict],
) -> dict:
    unique = sorted({float(value) for value in values})
    if not unique:
        status = "MISSING_MEASURED_VALUE"
        value = None
    elif len(unique) == 1:
        status = "MEASURED_SINGLE_VALUE"
        value = unique[0]
    else:
        status = "CONFLICTING_MEASURED_VALUES"
        value = None
    return {
        "source_anchor_identity": _source_anchor_id(
            study, anchor_kind, raw_identity
        ),
        "source_anchor_status": status,
        "value": value,
        "measured_values": unique,
        "source_anchor_evidence": sorted(
            [dict(item) for item in evidence], key=_canonical_json
        ),
    }


def _coerce_anchor(
    raw: Any, study: str, anchor_kind: str, raw_identity: Any,
) -> dict:
    if isinstance(raw, dict) and "source_anchor_status" in raw:
        return dict(raw)
    if raw is None:
        return _resolved_anchor(
            study, anchor_kind, raw_identity, [], []
        )
    return _resolved_anchor(
        study,
        anchor_kind,
        raw_identity,
        [float(raw)],
        [{"legacy_scalar_input": float(raw)}],
    )


def load_gse114002_source_anchors() -> Dict[Tuple[str, str], dict]:
    """Lossless WT rl anchor per (library, normalized mother sequence).

    Conflicting measured values are retained as evidence and made unusable;
    they are never averaged into an apparently unique source anchor.
    """
    import pandas as pd
    df = pd.read_csv(config.GSE114002_RAW_LIB, low_memory=False)
    anchor_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    evidence: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    raw_file_sha256 = sha256_file(Path(config.GSE114002_RAW_LIB))
    for row_index, (library, seq, rl) in enumerate(zip(
        df["library"].astype(str),
        df["utr"].astype(str),
        pd.to_numeric(df["rl"], errors="coerce"),
    )):
        if pd.notna(rl) and len(seq) > 10:
            key = (str(library).strip(), _normalize_sequence(seq))
            value = float(rl)
            anchor_values[key].append(value)
            evidence[key].append({
                "source_asset": str(config.GSE114002_RAW_LIB),
                "source_asset_sha256": raw_file_sha256,
                "raw_row_index": row_index,
                "library": key[0],
                "raw_sequence_sha256": hashlib.sha256(
                    key[1].encode("ascii")
                ).hexdigest(),
                "value": value,
                "endpoint": "rl",
            })
    return {
        key: _resolved_anchor(
            "GSE114002", "RAW_LIBRARY_RL", key,
            anchor_values[key], evidence[key],
        )
        for key in sorted(anchor_values)
    }


def load_gse200304_source_anchors() -> Dict[str, dict]:
    """WT Freq anchor per merged_id without barcode overwrite."""
    import pandas as pd
    cdf = pd.read_csv(config.GSE200304_COUNT, sep="\t", compression="infer")
    raw_count_sha256 = sha256_file(Path(config.GSE200304_COUNT))
    twist_sha256 = sha256_file(Path(config.GSE200304_TWIST))
    bc_freq: Dict[str, List[float]] = defaultdict(list)
    bc_evidence: Dict[str, List[dict]] = defaultdict(list)
    for row_index, (bc, fq) in enumerate(zip(
        cdf["Barcode"].astype(str),
        pd.to_numeric(cdf["Freq"], errors="coerce"),
    )):
        if pd.notna(fq):
            value = float(fq)
            bc_freq[bc].append(value)
            bc_evidence[bc].append({
                "source_asset": str(config.GSE200304_COUNT),
                "source_asset_sha256": raw_count_sha256,
                "raw_row_index": row_index,
                "barcode": bc,
                "value": value,
                "endpoint": "Freq",
            })
    # Map WT merged_id -> Freq. merged_id format "chr2:69461620_G-C_WT".
    tdf = pd.read_csv(config.GSE200304_TWIST, sep="\t", compression="infer")
    wt_freq: Dict[str, dict] = {}
    for mid in tdf["merged_id"].astype(str):
        if mid.endswith("_WT"):
            relation_evidence = list(bc_evidence.get(mid, [])) + [{
                "relation_source_asset": str(config.GSE200304_TWIST),
                "relation_source_asset_sha256": twist_sha256,
                "merged_id": mid,
                "relation": "WT_BARCODE_IDENTITY",
            }]
            wt_freq[mid] = _resolved_anchor(
                "GSE200304",
                "RAW_COUNT_FREQ",
                mid,
                bc_freq.get(mid, []),
                relation_evidence,
            )
    return wt_freq


def _attach_observation_provenance(
    record: dict, observations: Sequence[dict],
) -> None:
    rows = sorted([dict(row) for row in observations], key=_canonical_json)
    record["functional_observation_schema_version"] = (
        OBSERVATION_SCHEMA_VERSION
    )
    record["functional_observations"] = rows
    record["functional_observation_hashes"] = [
        row.get("input_row_sha256") or _json_sha256(row) for row in rows
    ]
    record["observation_ids"] = sorted({
        str(row["observation_id"]) for row in rows
        if row.get("observation_id") not in (None, "")
    })

    def unique_text(*names: str) -> Optional[str]:
        values = {
            str(row.get(name)).strip()
            for row in rows for name in names
            if row.get(name) not in (None, "")
        }
        return next(iter(values)) if len(values) == 1 else None

    # This is a raw assay arm used only as a crosswalk atom. It is explicitly
    # not a publication context_id.
    record["assay_arm"] = unique_text(
        "assay_arm", "context_id", "assay_context"
    )
    record["observation_context_id"] = unique_text("context_id")
    record["observation_assay_context"] = unique_text("assay_context")
    replicate_tokens = {_replicate_token(row) for row in rows}
    record["replicate"] = (
        json.loads(next(iter(replicate_tokens)))
        if len(replicate_tokens) == 1 else None
    )
    record["unit"] = unique_text("unit")


def _attach_source_anchor(record: dict, anchor: dict) -> None:
    record["source_anchor_identity"] = anchor.get(
        "source_anchor_identity"
    )
    record["source_anchor_status"] = anchor.get("source_anchor_status")
    record["source_anchor_evidence"] = anchor.get(
        "source_anchor_evidence", []
    )
    record["source_anchor_measured_values"] = anchor.get(
        "measured_values", []
    )


def derive_delta(study: str, spec: dict, record: dict,
                 cand_values: Mapping[str, Mapping[str, Any]],
                 src_anchor_gse114002: Mapping[str, Any],
                 src_anchor_gse200304: Mapping[str, Any],
                 cand_seq_id: str) -> List[dict]:
    """Return lossless delta records, one per resolvable observation slot."""
    cand_seq = record.get("candidate_sequence")
    src_seq = record.get("source_sequence")
    edit_list = record.get("edit_script", [])
    metadata = record.get("metadata", {}) or {}

    def base(source_anchor_rule: str):
        legacy_source_id = cand_seq_id[:-len(CAND_SUFFIX)] + SRC_SUFFIX
        return {
            "effect_dataset_schema_version": EFFECT_DATASET_SCHEMA_VERSION,
            "qualification_status": EFFECT_ROW_QUALIFICATION_STATUS,
            "pair_id": None,
            "record_id": record.get("record_id"),
            "study": study,
            "benchmark": spec["benchmark"],
            "source_sequence": src_seq,
            "candidate_sequence": cand_seq,
            "edit_list": edit_list,
            "edit_count": len(edit_list),
            "source_id": legacy_source_id,
            "legacy_source_id": legacy_source_id,
            "source_id_status": (
                "LEGACY_MECHANICAL_NOT_PUBLICATION_GROUP"
            ),
            "candidate_id": cand_seq_id,
            "region": record.get("region"),
            "library": metadata.get("library"),
            "source_asset": metadata.get("source_file"),
            "source_anchor_rule": source_anchor_rule,
            "canonical_provenance": {
                "record_id": record.get("record_id"),
                "dataset": record.get("dataset"),
                "accession": record.get("accession"),
                "region": record.get("region"),
                "metadata": metadata,
                "canonical_record_sha256": _json_sha256(record),
            },
        }

    t = spec["type"]

    if t == "diff_cand_src":
        endpoint = spec["cand_endpoint"]
        groups = _observation_groups(cand_values, cand_seq_id, endpoint)
        source_anchor_rule = "diff_cand_src:%s" % spec["src_source"]
        if spec["src_source"] == "raw_library":
            raw_identity = (
                str(metadata.get("library") or "").strip(),
                _normalize_sequence(src_seq),
            )
            raw_anchor = src_anchor_gse114002.get(raw_identity)
            # Synthetic/backward-compatible scalar maps are accepted only at
            # this pure-function boundary. Production loaders use the strict
            # (library, normalized sequence) key above.
            if raw_anchor is None:
                raw_anchor = src_anchor_gse114002.get(src_seq)
            anchor = _coerce_anchor(
                raw_anchor, study, "RAW_LIBRARY_RL", raw_identity
            )
        elif spec["src_source"] == "raw_count":
            # GSE200304: source anchor is the _WT barcode of this pair.
            mid = metadata.get("merged_id")
            wt_id = None
            if mid:
                wt_id = (
                    str(mid)[:-len("_Mutant")] + "_WT"
                    if str(mid).endswith("_Mutant")
                    else str(mid) + "_WT"
                )
            anchor = _coerce_anchor(
                src_anchor_gse200304.get(wt_id) if wt_id else None,
                study,
                "RAW_COUNT_FREQ",
                wt_id,
            )
        else:
            raise ValueError(
                "unknown diff_cand_src source %s" % spec["src_source"]
            )
        slots = groups or {("", "", "null", ""): {
            "value": None, "rows": [],
        }}
        out = []
        for key in sorted(slots):
            observation = slots[key]
            cand_val = observation["value"]
            src_val = anchor.get("value")
            rec = base(source_anchor_rule)
            rec["endpoint"] = endpoint
            rec["candidate_value"] = cand_val
            rec["source_value"] = src_val
            rec["delta"] = (
                cand_val - src_val
                if cand_val is not None and src_val is not None else None
            )
            rec["delta_source_status"] = (
                "derived" if rec["delta"] is not None
                else "source_anchor_unavailable"
            )
            _attach_observation_provenance(rec, observation["rows"])
            _attach_source_anchor(rec, anchor)
            out.append(rec)
        return out

    if t == "diff_ref_alt":
        ref_endpoint = spec["ref_endpoint"]
        alt_endpoint = spec["alt_endpoint"]
        policy = validate_diff_ref_alt_join_policy(spec)
        policy_sha256 = _json_sha256(policy)
        ref_groups = _paired_observation_groups(
            cand_values,
            cand_seq_id,
            ref_endpoint,
            side="ref",
            policy=policy,
        )
        alt_groups = _paired_observation_groups(
            cand_values,
            cand_seq_id,
            alt_endpoint,
            side="alt",
            policy=policy,
        )
        keys = sorted(set(ref_groups) | set(alt_groups)) or [
            (
                "null",
                "null",
                "null",
                policy["comparison_scale_id"],
            )
        ]
        out = []
        for key in keys:
            ref_present = key in ref_groups
            alt_present = key in alt_groups
            ref_group = ref_groups.get(key, {
                "value": None, "rows": [], "raw_unit_tokens": [],
            })
            alt_group = alt_groups.get(key, {
                "value": None, "rows": [], "raw_unit_tokens": [],
            })
            ref = ref_group["value"]
            alt = alt_group["value"]
            if ref_present and alt_present:
                join_disposition = "MATCHED"
                unavailable_reason = None
            elif ref_present:
                join_disposition = "UNMATCHED_REF_ONLY"
                unavailable_reason = _unmatched_ref_alt_reason(
                    "ref", key, sorted(alt_groups)
                )
            elif alt_present:
                join_disposition = "UNMATCHED_ALT_ONLY"
                unavailable_reason = _unmatched_ref_alt_reason(
                    "alt", key, sorted(ref_groups)
                )
            else:
                join_disposition = "MISSING_BOTH"
                unavailable_reason = "MISSING_REF_AND_ALT_ENDPOINT"

            source_anchor_rule = (
                "diff_ref_alt:%s:%s"
                % (ref_endpoint, policy["policy_id"])
            )
            rec = base(source_anchor_rule)
            rec["endpoint"] = "%s/%s" % (ref_endpoint, alt_endpoint)
            rec["source_value"] = ref
            rec["candidate_value"] = alt
            rec["delta"] = (
                alt - ref if ref_present and alt_present else None
            )
            rec["delta_source_status"] = (
                "derived" if rec["delta"] is not None
                else "source_anchor_unavailable"
            )
            rec["delta_unavailable_reason"] = unavailable_reason
            rec["diff_ref_alt_join_disposition"] = join_disposition
            rec["diff_ref_alt_join_policy_schema_version"] = policy[
                "schema_version"
            ]
            rec["diff_ref_alt_join_policy_id"] = policy["policy_id"]
            rec["diff_ref_alt_join_policy_sha256"] = policy_sha256
            rec["comparison_scale_id"] = policy["comparison_scale_id"]
            rec["ref_observation_group_present"] = ref_present
            rec["alt_observation_group_present"] = alt_present
            rec["ref_observation_unit_tokens"] = ref_group[
                "raw_unit_tokens"
            ]
            rec["alt_observation_unit_tokens"] = alt_group[
                "raw_unit_tokens"
            ]
            rec["observation_join_key"] = {
                "context_id": json.loads(key[0]),
                "assay_context": json.loads(key[1]),
                "replicate": json.loads(key[2]),
                "comparison_scale_id": key[3],
            }
            rec["replicate_pairing_status"] = (
                _replicate_pairing_status(key[2])
            )
            observations = ref_group["rows"] + alt_group["rows"]
            _attach_observation_provenance(rec, observations)
            # Paired policy reads only the explicit ``replicate`` field.  Do
            # not let the generic provenance helper's legacy replicate_id
            # fallback promote a mechanical id into a replicate claim.
            rec["replicate"] = json.loads(key[2])
            anchor = _resolved_anchor(
                study,
                "FUNCTIONAL_REF_ENDPOINT",
                {
                    "source_sequence_sha256": hashlib.sha256(
                        _normalize_sequence(src_seq).encode("ascii")
                    ).hexdigest(),
                    "ref_endpoint": ref_endpoint,
                    "join_policy_id": policy["policy_id"],
                    "join_policy_sha256": policy_sha256,
                    "observation_join_key": rec["observation_join_key"],
                },
                [] if ref is None else [ref],
                ref_group["rows"],
            )
            _attach_source_anchor(rec, anchor)
            out.append(rec)
        return out

    if t == "log2fc":
        endpoint = spec["endpoint"]
        groups = _observation_groups(cand_values, cand_seq_id, endpoint)
        slots = groups or {("", "", "null", ""): {
            "value": None, "rows": [],
        }}
        out = []
        for key in sorted(slots):
            observation = slots[key]
            value = observation["value"]
            rec = base("log2fc_direct:%s" % endpoint)
            rec["endpoint"] = endpoint
            rec["source_value"] = None
            rec["candidate_value"] = value
            rec["delta"] = value
            rec["delta_source_status"] = (
                "log2fc_direct" if value is not None
                else "source_anchor_unavailable"
            )
            _attach_observation_provenance(rec, observation["rows"])
            anchor = _resolved_anchor(
                study,
                "DIRECT_EFFECT_OBSERVATION",
                {
                    "source_sequence_sha256": hashlib.sha256(
                        _normalize_sequence(src_seq).encode("ascii")
                    ).hexdigest(),
                    "endpoint": endpoint,
                    "observation_join_key": key,
                },
                [] if value is None else [value],
                observation["rows"],
            )
            _attach_source_anchor(rec, anchor)
            out.append(rec)
        return out

    if t == "diff_wt_meta":
        out = []
        for ep_spec in spec["endpoints"]:
            endpoint = ep_spec["cand_endpoint"]
            groups = _observation_groups(cand_values, cand_seq_id, endpoint)
            slots = groups or {("", "", "null", ""): {
                "value": None, "rows": [],
            }}
            raw_wt = metadata.get(ep_spec["wt_meta"])
            wt_values = [] if raw_wt is None else [float(raw_wt)]
            anchor = _resolved_anchor(
                study,
                "CANONICAL_WT_METADATA",
                {
                    "source_sequence_sha256": hashlib.sha256(
                        _normalize_sequence(src_seq).encode("ascii")
                    ).hexdigest(),
                    "metadata_field": ep_spec["wt_meta"],
                },
                wt_values,
                [{
                    "canonical_record_sha256": _json_sha256(record),
                    "record_id": record.get("record_id"),
                    "metadata_field": ep_spec["wt_meta"],
                    "value": raw_wt,
                }] if raw_wt is not None else [],
            )
            for key in sorted(slots):
                observation = slots[key]
                cand_val = observation["value"]
                wt = anchor.get("value")
                rec = base("diff_wt_meta:%s" % ep_spec["wt_meta"])
                rec["endpoint"] = endpoint
                rec["source_value"] = wt
                rec["candidate_value"] = cand_val
                rec["delta"] = (
                    cand_val - wt
                    if cand_val is not None and wt is not None else None
                )
                rec["delta_source_status"] = (
                    "derived" if rec["delta"] is not None
                    else "source_anchor_unavailable"
                )
                _attach_observation_provenance(
                    rec, observation["rows"]
                )
                _attach_source_anchor(rec, anchor)
                out.append(rec)
        return out

    raise ValueError(f"unknown delta type {t}")


def _replay_input_row_hash(row: Mapping[str, Any]) -> str:
    declared = row.get("input_row_sha256")
    if declared is None:
        return _json_sha256(row)
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(char not in "0123456789abcdef" for char in declared)
    ):
        raise DiffRefAltJoinPolicyError(
            "functional observation input_row_sha256 is not canonical"
        )
    preserved = dict(row)
    preserved.pop("input_row_sha256")
    if _json_sha256(preserved) != declared:
        raise DiffRefAltJoinPolicyError(
            "functional observation content disagrees with input_row_sha256"
        )
    return declared


def _replay_exact_join_key(
    row: Mapping[str, Any], policy: Mapping[str, Any],
) -> Tuple[str, str, str, str]:
    """Independently reconstruct the exact key without replicate_id fallback."""
    for field in ("context_id", "assay_context", "replicate"):
        if field not in row:
            raise DiffRefAltJoinPolicyError(
                "replay observation must explicitly carry %s" % field
            )
    context_id = row["context_id"]
    if context_id is None or policy["context_required_non_null"] is not True:
        raise DiffRefAltJoinPolicyError(
            "replay requires a non-null context_id"
        )
    _canonical_nonempty_string(context_id, label="replay context_id")
    assay_context = row["assay_context"]
    if assay_context is None:
        if policy["assay_context_null_policy"] != (
            "ALLOW_EXACT_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
        ):
            raise DiffRefAltJoinPolicyError(
                "replay policy does not allow null assay_context"
            )
    else:
        _canonical_nonempty_string(
            assay_context, label="replay assay_context"
        )
    replicate = row[policy["replicate_field"]]
    if replicate is None and policy["replicate_null_policy"] != (
        "ALLOW_EXACT_NULL_NO_REPLICATE_CLAIM"
    ):
        raise DiffRefAltJoinPolicyError(
            "replay policy does not allow a null replicate slot"
        )
    return (
        _canonical_json(context_id),
        _canonical_json(assay_context),
        _canonical_json(replicate),
        policy["comparison_scale_id"],
    )


def _replay_side_groups(
    rows: Any,
    *,
    sid: str,
    endpoint: str,
    side: str,
    policy: Mapping[str, Any],
) -> Dict[Tuple[str, str, str, str], dict]:
    """Rebuild side groups independently from preserved observation evidence."""
    if side not in {"ref", "alt"}:
        raise DiffRefAltJoinPolicyError("replay side is invalid")
    if isinstance(rows, Mapping):
        rows = [rows]
    if not isinstance(rows, list):
        raise DiffRefAltJoinPolicyError(
            "replay observations must be a structured list"
        )
    allowed = set(policy["%s_allowed_raw_unit_tokens" % side])
    grouped: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise DiffRefAltJoinPolicyError(
                "replay functional observation is not an object"
            )
        row = dict(raw_row)
        if row.get("sequence_id") != sid:
            raise DiffRefAltJoinPolicyError(
                "replay observation sequence_id mismatch"
            )
        if row.get("endpoint_id") != endpoint:
            raise DiffRefAltJoinPolicyError(
                "replay observation endpoint mismatch"
            )
        raw_unit = row.get("unit")
        if not isinstance(raw_unit, str) or raw_unit not in allowed:
            raise DiffRefAltJoinPolicyError(
                "replay observation raw unit is outside %s policy" % side
            )
        value = row.get("value")
        if isinstance(value, bool):
            raise DiffRefAltJoinPolicyError(
                "replay observation value cannot be boolean"
            )
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise DiffRefAltJoinPolicyError(
                "replay observation value is non-numeric"
            ) from None
        if not math.isfinite(value):
            raise DiffRefAltJoinPolicyError(
                "replay observation value is non-finite"
            )
        row["value"] = value
        grouped[_replay_exact_join_key(row, policy)].append(row)

    resolved = {}
    for key in sorted(grouped):
        evidence = sorted(grouped[key], key=_canonical_json)
        values = {float(row["value"]) for row in evidence}
        if len(values) != 1:
            raise DiffRefAltJoinPolicyError(
                "replay found non-unique finite values in one side group"
            )
        resolved[key] = {
            "value": next(iter(values)),
            "rows": evidence,
            "input_row_hashes": sorted(
                _replay_input_row_hash(row) for row in evidence
            ),
            "raw_unit_tokens": sorted({
                str(row["unit"]) for row in evidence
            }),
        }
    return resolved


def _replay_unmatched_reason(
    side: str,
    key: Tuple[str, str, str, str],
    opposite_keys: Sequence[Tuple[str, str, str, str]],
) -> str:
    """Independently classify an orphan without invoking derive logic."""
    if side not in {"ref", "alt"}:
        raise DiffRefAltJoinPolicyError("replay orphan side is invalid")
    if not opposite_keys:
        return (
            "MISSING_ALT_ENDPOINT" if side == "ref"
            else "MISSING_REF_ENDPOINT"
        )
    context, assay_context, replicate, scale = key
    same_context = [item for item in opposite_keys if item[0] == context]
    if not same_context:
        return "REF_ALT_CONTEXT_ID_MISMATCH"
    same_assay = [
        item for item in same_context if item[1] == assay_context
    ]
    if not same_assay:
        return "REF_ALT_ASSAY_CONTEXT_MISMATCH"
    same_replicate = [
        item for item in same_assay if item[2] == replicate
    ]
    if not same_replicate:
        return "REF_ALT_REPLICATE_MISMATCH"
    if not any(item[3] == scale for item in same_replicate):
        return "REF_ALT_COMPARISON_SCALE_MISMATCH"
    return (
        "MISSING_ALT_ENDPOINT" if side == "ref"
        else "MISSING_REF_ENDPOINT"
    )


def _join_key_object(key: Tuple[str, str, str, str]) -> dict:
    return {
        "context_id": json.loads(key[0]),
        "assay_context": json.loads(key[1]),
        "replicate": json.loads(key[2]),
        "comparison_scale_id": key[3],
    }


def _side_group_hash(
    *,
    study: str,
    pair_id: str,
    candidate_id: str,
    side: str,
    endpoint: str,
    key: Tuple[str, str, str, str],
    group: Mapping[str, Any],
) -> str:
    return _json_sha256({
        "study": study,
        "pair_id": pair_id,
        "candidate_id": candidate_id,
        "side": side,
        "endpoint": endpoint,
        "comparison_key": _join_key_object(key),
        "value": group["value"],
        "raw_unit_tokens": group["raw_unit_tokens"],
        "input_row_hashes": group["input_row_hashes"],
    })


def _counter_multiset_sha256(counter: Mapping[str, int]) -> str:
    return _json_sha256([
        {"item_sha256": key, "count": int(counter[key])}
        for key in sorted(counter)
    ])


def _assert_exact_optional_finite_number(
    actual: Any, expected: Optional[float], *, label: str,
) -> None:
    if expected is None:
        if actual is not None:
            raise DiffRefAltJoinPolicyError(
                "%s must be null under independent replay" % label
            )
        return
    if isinstance(actual, bool):
        raise DiffRefAltJoinPolicyError(
            "%s cannot be boolean" % label
        )
    try:
        actual_float = float(actual)
    except (TypeError, ValueError):
        raise DiffRefAltJoinPolicyError(
            "%s must be numeric" % label
        ) from None
    if not math.isfinite(actual_float) or actual_float != expected:
        raise DiffRefAltJoinPolicyError(
            "%s disagrees with independent replay" % label
        )


def summarize_diff_ref_alt_join_audits(
    records: Sequence[Mapping[str, Any]],
    specs: Mapping[str, Mapping[str, Any]],
    *,
    cand_values: Mapping[str, Mapping[str, Any]],
    materialized_pairs: Sequence[Mapping[str, Any]],
) -> dict:
    """Replay-verify output evidence against the raw input group multiset.

    Self-reported flags, values, keys and deltas are never used to establish
    correctness.  The audit independently reconstructs every output group from
    ``functional_observations`` and compares its hash multiset with a second
    reconstruction from ``cand_values`` for every materialized pair.
    """
    policies = validate_diff_ref_alt_policy_registry(specs)
    descriptors_by_study: Dict[str, List[dict]] = defaultdict(list)
    descriptor_keys = set()
    for raw_descriptor in materialized_pairs:
        if not isinstance(raw_descriptor, Mapping):
            raise DiffRefAltJoinPolicyError(
                "materialized pair descriptor is not an object"
            )
        study = _canonical_nonempty_string(
            raw_descriptor.get("study"), label="materialized study"
        )
        pair_id = _canonical_nonempty_string(
            raw_descriptor.get("pair_id"), label="materialized pair_id"
        )
        candidate_id = _canonical_nonempty_string(
            raw_descriptor.get("candidate_sequence_id"),
            label="materialized candidate_sequence_id",
        )
        if study not in policies:
            raise DiffRefAltJoinPolicyError(
                "materialized diff_ref_alt pair has no registered policy"
            )
        descriptor_key = (study, pair_id, candidate_id)
        if descriptor_key in descriptor_keys:
            raise DiffRefAltJoinPolicyError(
                "duplicate materialized diff_ref_alt pair descriptor"
            )
        descriptor_keys.add(descriptor_key)
        descriptors_by_study[study].append({
            "study": study,
            "pair_id": pair_id,
            "candidate_sequence_id": candidate_id,
        })

    audits = {}
    for study in sorted(policies):
        spec = specs[study]
        policy = policies[study]
        policy_sha256 = _json_sha256(policy)
        ref_endpoint = spec["ref_endpoint"]
        alt_endpoint = spec["alt_endpoint"]
        descriptors = sorted(
            descriptors_by_study.get(study, []), key=_canonical_json
        )
        expected_groups_by_pair = {}
        expected_side_hashes: Counter[str] = Counter()
        expected_missing_both: Counter[str] = Counter()
        expected_ref_groups = 0
        expected_alt_groups = 0
        expected_matched_groups = 0
        expected_unmatched_ref_groups = 0
        expected_unmatched_alt_groups = 0
        for descriptor in descriptors:
            pair_id = descriptor["pair_id"]
            candidate_id = descriptor["candidate_sequence_id"]
            endpoint_map = cand_values.get(candidate_id, {})
            if not isinstance(endpoint_map, Mapping):
                raise DiffRefAltJoinPolicyError(
                    "candidate observation index is not a mapping"
                )
            ref_groups = _replay_side_groups(
                endpoint_map.get(ref_endpoint, []),
                sid=candidate_id,
                endpoint=ref_endpoint,
                side="ref",
                policy=policy,
            )
            alt_groups = _replay_side_groups(
                endpoint_map.get(alt_endpoint, []),
                sid=candidate_id,
                endpoint=alt_endpoint,
                side="alt",
                policy=policy,
            )
            descriptor_key = (pair_id, candidate_id)
            expected_groups_by_pair[descriptor_key] = {
                "ref": ref_groups,
                "alt": alt_groups,
            }
            expected_ref_groups += len(ref_groups)
            expected_alt_groups += len(alt_groups)
            exact_keys = set(ref_groups) & set(alt_groups)
            expected_matched_groups += len(exact_keys)
            expected_unmatched_ref_groups += len(
                set(ref_groups) - exact_keys
            )
            expected_unmatched_alt_groups += len(
                set(alt_groups) - exact_keys
            )
            for side, endpoint, groups in (
                ("ref", ref_endpoint, ref_groups),
                ("alt", alt_endpoint, alt_groups),
            ):
                for key, group in groups.items():
                    expected_side_hashes[_side_group_hash(
                        study=study,
                        pair_id=pair_id,
                        candidate_id=candidate_id,
                        side=side,
                        endpoint=endpoint,
                        key=key,
                        group=group,
                    )] += 1
            if not ref_groups and not alt_groups:
                expected_missing_both[_json_sha256(descriptor)] += 1

        study_rows = [
            row for row in records if row.get("study") == study
        ]
        consumed_side_hashes: Counter[str] = Counter()
        consumed_missing_both: Counter[str] = Counter()
        dispositions: Counter[str] = Counter()
        unavailable_reasons: Counter[str] = Counter()
        replicate_statuses: Counter[str] = Counter()
        ref_unit_tokens: Counter[str] = Counter()
        alt_unit_tokens: Counter[str] = Counter()
        consumed_ref_groups = 0
        consumed_alt_groups = 0
        n_derived = 0
        n_unavailable = 0
        output_pair_registry = set()

        for row in study_rows:
            if not isinstance(row, Mapping):
                raise DiffRefAltJoinPolicyError(
                    "effect output row is not an object"
                )
            if row.get("effect_dataset_schema_version") != (
                EFFECT_DATASET_SCHEMA_VERSION
            ):
                raise DiffRefAltJoinPolicyError(
                    "effect output row has a stale schema version"
                )
            if row.get("qualification_status") != (
                EFFECT_ROW_QUALIFICATION_STATUS
            ):
                raise DiffRefAltJoinPolicyError(
                    "effect output row does not require its v3 manifest"
                )
            if row.get("functional_observation_schema_version") != (
                OBSERVATION_SCHEMA_VERSION
            ):
                raise DiffRefAltJoinPolicyError(
                    "effect output functional-observation schema mismatch"
                )
            if row.get("diff_ref_alt_join_policy_schema_version") != (
                DIFF_REF_ALT_JOIN_POLICY_SCHEMA_VERSION
            ):
                raise DiffRefAltJoinPolicyError(
                    "effect output join-policy schema mismatch"
                )
            pair_id = _canonical_nonempty_string(
                row.get("pair_id"), label="output pair_id"
            )
            candidate_id = _canonical_nonempty_string(
                row.get("candidate_id"), label="output candidate_id"
            )
            descriptor_key = (pair_id, candidate_id)
            if descriptor_key not in expected_groups_by_pair:
                raise DiffRefAltJoinPolicyError(
                    "effect output row has no materialized pair descriptor"
                )
            output_pair_registry.add(descriptor_key)
            if row.get("endpoint") != "%s/%s" % (
                ref_endpoint, alt_endpoint
            ):
                raise DiffRefAltJoinPolicyError(
                    "effect output endpoint disagrees with paired policy"
                )
            if row.get("diff_ref_alt_join_policy_id") != policy["policy_id"]:
                raise DiffRefAltJoinPolicyError(
                    "effect output policy-id mismatch"
                )
            if row.get("diff_ref_alt_join_policy_sha256") != policy_sha256:
                raise DiffRefAltJoinPolicyError(
                    "effect output policy-hash mismatch"
                )
            if row.get("comparison_scale_id") != policy[
                "comparison_scale_id"
            ]:
                raise DiffRefAltJoinPolicyError(
                    "effect output comparison-scale mismatch"
                )

            observations = row.get("functional_observations")
            if not isinstance(observations, list):
                raise DiffRefAltJoinPolicyError(
                    "effect output lacks functional_observations evidence"
                )
            ref_rows = []
            alt_rows = []
            for observation in observations:
                if not isinstance(observation, Mapping):
                    raise DiffRefAltJoinPolicyError(
                        "effect observation evidence is not an object"
                    )
                endpoint = observation.get("endpoint_id")
                if endpoint == ref_endpoint:
                    ref_rows.append(observation)
                elif endpoint == alt_endpoint:
                    alt_rows.append(observation)
                else:
                    raise DiffRefAltJoinPolicyError(
                        "effect observation has an unexpected endpoint"
                    )
            replay_ref = _replay_side_groups(
                ref_rows,
                sid=candidate_id,
                endpoint=ref_endpoint,
                side="ref",
                policy=policy,
            )
            replay_alt = _replay_side_groups(
                alt_rows,
                sid=candidate_id,
                endpoint=alt_endpoint,
                side="alt",
                policy=policy,
            )
            if len(replay_ref) > 1 or len(replay_alt) > 1:
                raise DiffRefAltJoinPolicyError(
                    "one effect row consumed more than one group per side"
                )
            ref_item = next(iter(replay_ref.items()), None)
            alt_item = next(iter(replay_alt.items()), None)
            if ref_item and alt_item and ref_item[0] != alt_item[0]:
                raise DiffRefAltJoinPolicyError(
                    "one effect row combined non-exact ref/alt keys"
                )
            ref_present = ref_item is not None
            alt_present = alt_item is not None
            ref_key, ref_group = ref_item if ref_item else (None, None)
            alt_key, alt_group = alt_item if alt_item else (None, None)
            key = ref_key or alt_key or (
                "null", "null", "null", policy["comparison_scale_id"]
            )
            expected_for_pair = expected_groups_by_pair[descriptor_key]
            if ref_present and alt_present:
                disposition = "MATCHED"
                reason = None
            elif ref_present:
                if key in expected_for_pair["alt"]:
                    raise DiffRefAltJoinPolicyError(
                        "output split an exact ref/alt match into orphans"
                    )
                disposition = "UNMATCHED_REF_ONLY"
                reason = _replay_unmatched_reason(
                    "ref", key, sorted(expected_for_pair["alt"])
                )
            elif alt_present:
                if key in expected_for_pair["ref"]:
                    raise DiffRefAltJoinPolicyError(
                        "output split an exact ref/alt match into orphans"
                    )
                disposition = "UNMATCHED_ALT_ONLY"
                reason = _replay_unmatched_reason(
                    "alt", key, sorted(expected_for_pair["ref"])
                )
            else:
                disposition = "MISSING_BOTH"
                reason = "MISSING_REF_AND_ALT_ENDPOINT"

            ref_value = ref_group["value"] if ref_group else None
            alt_value = alt_group["value"] if alt_group else None
            delta = alt_value - ref_value if ref_present and alt_present else None
            if delta is not None and not math.isfinite(delta):
                raise DiffRefAltJoinPolicyError(
                    "independently replayed delta is non-finite"
                )
            _assert_exact_optional_finite_number(
                row.get("source_value"), ref_value, label="source_value"
            )
            _assert_exact_optional_finite_number(
                row.get("candidate_value"), alt_value,
                label="candidate_value",
            )
            _assert_exact_optional_finite_number(
                row.get("delta"), delta, label="delta"
            )
            expected_status = (
                "derived" if delta is not None
                else "source_anchor_unavailable"
            )
            if row.get("delta_source_status") != expected_status:
                raise DiffRefAltJoinPolicyError(
                    "delta_source_status disagrees with independent replay"
                )
            if row.get("delta_unavailable_reason") != reason:
                raise DiffRefAltJoinPolicyError(
                    "delta_unavailable_reason disagrees with replay"
                )
            if row.get("diff_ref_alt_join_disposition") != disposition:
                raise DiffRefAltJoinPolicyError(
                    "join disposition disagrees with independent replay"
                )
            if row.get("ref_observation_group_present") is not ref_present:
                raise DiffRefAltJoinPolicyError(
                    "ref presence flag disagrees with independent replay"
                )
            if row.get("alt_observation_group_present") is not alt_present:
                raise DiffRefAltJoinPolicyError(
                    "alt presence flag disagrees with independent replay"
                )
            expected_ref_units = (
                ref_group["raw_unit_tokens"] if ref_group else []
            )
            expected_alt_units = (
                alt_group["raw_unit_tokens"] if alt_group else []
            )
            if row.get("ref_observation_unit_tokens") != expected_ref_units:
                raise DiffRefAltJoinPolicyError(
                    "ref raw-unit provenance disagrees with replay"
                )
            if row.get("alt_observation_unit_tokens") != expected_alt_units:
                raise DiffRefAltJoinPolicyError(
                    "alt raw-unit provenance disagrees with replay"
                )
            if row.get("observation_join_key") != _join_key_object(key):
                raise DiffRefAltJoinPolicyError(
                    "observation_join_key disagrees with independent replay"
                )
            if row.get("replicate") != json.loads(key[2]):
                raise DiffRefAltJoinPolicyError(
                    "top-level replicate used a non-policy fallback"
                )
            replicate_status = _replicate_pairing_status(key[2])
            if row.get("replicate_pairing_status") != replicate_status:
                raise DiffRefAltJoinPolicyError(
                    "replicate pairing status disagrees with replay"
                )
            replay_hashes = sorted(
                (ref_group["input_row_hashes"] if ref_group else [])
                + (alt_group["input_row_hashes"] if alt_group else [])
            )
            if sorted(row.get("functional_observation_hashes") or []) != (
                replay_hashes
            ):
                raise DiffRefAltJoinPolicyError(
                    "functional observation hashes disagree with replay"
                )
            expected_observation_ids = sorted({
                str(observation["observation_id"])
                for observation in observations
                if observation.get("observation_id") not in (None, "")
            })
            if row.get("observation_ids") != expected_observation_ids:
                raise DiffRefAltJoinPolicyError(
                    "observation_ids disagree with replayed evidence"
                )
            expected_context = (
                json.loads(key[0]) if observations else None
            )
            expected_assay_context = (
                json.loads(key[1]) if observations else None
            )
            if row.get("observation_context_id") != expected_context:
                raise DiffRefAltJoinPolicyError(
                    "top-level observation_context_id disagrees with replay"
                )
            if row.get("observation_assay_context") != (
                expected_assay_context
            ):
                raise DiffRefAltJoinPolicyError(
                    "top-level observation_assay_context disagrees with replay"
                )
            raw_units = sorted({
                str(observation["unit"]) for observation in observations
            })
            expected_top_unit = raw_units[0] if len(raw_units) == 1 else None
            if row.get("unit") != expected_top_unit:
                raise DiffRefAltJoinPolicyError(
                    "top-level unit disagrees with raw-unit replay"
                )

            expected_anchor = _resolved_anchor(
                study,
                "FUNCTIONAL_REF_ENDPOINT",
                {
                    "source_sequence_sha256": hashlib.sha256(
                        _normalize_sequence(
                            row.get("source_sequence")
                        ).encode("ascii")
                    ).hexdigest(),
                    "ref_endpoint": ref_endpoint,
                    "join_policy_id": policy["policy_id"],
                    "join_policy_sha256": policy_sha256,
                    "observation_join_key": _join_key_object(key),
                },
                [] if ref_value is None else [ref_value],
                ref_group["rows"] if ref_group else [],
            )
            expected_anchor_fields = {
                "source_anchor_identity": expected_anchor[
                    "source_anchor_identity"
                ],
                "source_anchor_status": expected_anchor[
                    "source_anchor_status"
                ],
                "source_anchor_evidence": expected_anchor[
                    "source_anchor_evidence"
                ],
                "source_anchor_measured_values": expected_anchor[
                    "measured_values"
                ],
            }
            for field, expected_value in expected_anchor_fields.items():
                if row.get(field) != expected_value:
                    raise DiffRefAltJoinPolicyError(
                        "%s disagrees with replayed ref anchor" % field
                    )

            for side, endpoint, item in (
                ("ref", ref_endpoint, ref_item),
                ("alt", alt_endpoint, alt_item),
            ):
                if item is None:
                    continue
                group_key, group = item
                consumed_side_hashes[_side_group_hash(
                    study=study,
                    pair_id=pair_id,
                    candidate_id=candidate_id,
                    side=side,
                    endpoint=endpoint,
                    key=group_key,
                    group=group,
                )] += 1
            if disposition == "MISSING_BOTH":
                consumed_missing_both[_json_sha256({
                    "study": study,
                    "pair_id": pair_id,
                    "candidate_sequence_id": candidate_id,
                })] += 1

            consumed_ref_groups += int(ref_present)
            consumed_alt_groups += int(alt_present)
            dispositions[disposition] += 1
            if reason is not None:
                unavailable_reasons[reason] += 1
            replicate_statuses[replicate_status] += 1
            ref_unit_tokens.update(expected_ref_units)
            alt_unit_tokens.update(expected_alt_units)
            n_derived += int(delta is not None)
            n_unavailable += int(delta is None)

        if consumed_side_hashes != expected_side_hashes:
            raise DiffRefAltJoinPolicyError(
                "output side-group hash multiset does not equal raw input"
            )
        if consumed_missing_both != expected_missing_both:
            raise DiffRefAltJoinPolicyError(
                "missing-both output multiset does not equal materialized pairs"
            )
        materialized_registry = {
            (item["pair_id"], item["candidate_sequence_id"])
            for item in descriptors
        }
        if output_pair_registry != materialized_registry:
            raise DiffRefAltJoinPolicyError(
                "output pair registry does not equal materialized pair registry"
            )
        n_matched = dispositions["MATCHED"]
        n_missing_both = dispositions["MISSING_BOTH"]
        if (
            n_matched != expected_matched_groups
            or dispositions["UNMATCHED_REF_ONLY"]
            != expected_unmatched_ref_groups
            or dispositions["UNMATCHED_ALT_ONLY"]
            != expected_unmatched_alt_groups
            or n_missing_both != sum(expected_missing_both.values())
        ):
            raise DiffRefAltJoinPolicyError(
                "output join dispositions disagree with raw-input replay"
            )
        expected_output_rows = (
            expected_ref_groups + expected_alt_groups
            - expected_matched_groups + sum(expected_missing_both.values())
        )
        if len(study_rows) != expected_output_rows:
            raise DiffRefAltJoinPolicyError(
                "replay-verified side-group conservation failed for %s" % study
            )

        audit = {
            "schema_version": DIFF_REF_ALT_JOIN_AUDIT_SCHEMA_VERSION,
            "verification_status": "INDEPENDENT_REPLAY_VERIFIED",
            "join_policy": policy,
            "join_policy_sha256": policy_sha256,
            "policy_id": policy["policy_id"],
            "comparison_scale_id": policy["comparison_scale_id"],
            "n_materialized_pairs": len(descriptors),
            "n_output_pairs": len(output_pair_registry),
            "n_output_join_rows": len(study_rows),
            "n_expected_missing_both_pairs": sum(
                expected_missing_both.values()
            ),
            "n_consumed_missing_both_rows": sum(
                consumed_missing_both.values()
            ),
            "n_ref_observation_groups": expected_ref_groups,
            "n_alt_observation_groups": expected_alt_groups,
            "n_input_side_groups": (
                expected_ref_groups + expected_alt_groups
            ),
            "n_consumed_ref_observation_groups": consumed_ref_groups,
            "n_consumed_alt_observation_groups": consumed_alt_groups,
            "n_consumed_side_groups": (
                consumed_ref_groups + consumed_alt_groups
            ),
            "n_matched_groups": n_matched,
            "n_unmatched_ref_groups": dispositions["UNMATCHED_REF_ONLY"],
            "n_unmatched_alt_groups": dispositions["UNMATCHED_ALT_ONLY"],
            "n_missing_both_rows": n_missing_both,
            "n_derived_rows": n_derived,
            "n_unavailable_rows": n_unavailable,
            "expected_output_rows_from_group_conservation": (
                expected_output_rows
            ),
            "join_disposition_counts": dict(sorted(dispositions.items())),
            "delta_unavailable_reason_counts": dict(
                sorted(unavailable_reasons.items())
            ),
            "replicate_pairing_status_counts": dict(
                sorted(replicate_statuses.items())
            ),
            "ref_raw_unit_token_group_counts": dict(
                sorted(ref_unit_tokens.items())
            ),
            "alt_raw_unit_token_group_counts": dict(
                sorted(alt_unit_tokens.items())
            ),
            "expected_input_side_group_multiset_sha256": (
                _counter_multiset_sha256(expected_side_hashes)
            ),
            "consumed_output_side_group_multiset_sha256": (
                _counter_multiset_sha256(consumed_side_hashes)
            ),
            "expected_missing_both_multiset_sha256": (
                _counter_multiset_sha256(expected_missing_both)
            ),
            "consumed_missing_both_multiset_sha256": (
                _counter_multiset_sha256(consumed_missing_both)
            ),
            "materialized_pair_registry_sha256": _json_sha256(sorted(
                [list(item) for item in materialized_registry]
            )),
            "output_pair_registry_sha256": _json_sha256(sorted(
                [list(item) for item in output_pair_registry]
            )),
        }
        if set(audit) != DIFF_REF_ALT_JOIN_AUDIT_KEYS:
            raise DiffRefAltJoinPolicyError(
                "internal replay audit schema keys drifted"
            )
        audits[study] = audit
    return audits


def build(out_dir: Path) -> dict:
    diff_ref_alt_policies = validate_diff_ref_alt_policy_registry(
        config.DELTA_SPECS
    )
    staging = Path(config.STAGING_DIR)
    pairs_path = staging / "utr_edit_pairs.jsonl"
    observations_path = staging / "functional_observations.jsonl"
    pairs = load_pairs(pairs_path)

    # Map pair -> candidate_seq_id / source_seq_id / study
    needed_record_ids = set()
    cand_seq_ids_by_study: Dict[str, set] = defaultdict(set)
    for p in pairs:
        cid = p["candidate_sequence_id"]
        rid = cid[:-len(CAND_SUFFIX)]
        needed_record_ids.add(rid)
        cand_seq_ids_by_study[_study_of(cid)].add(cid)

    canonical = load_canonical(Path(config.CANONICAL_RECORDS), needed_record_ids)

    cand_values = load_candidate_observations(
        observations_path, cand_seq_ids_by_study
    )

    src_114 = load_gse114002_source_anchors() if "GSE114002" in config.ACTIVE_STUDIES else {}
    src_200 = load_gse200304_source_anchors() if "GSE200304" in config.ACTIVE_STUDIES else {}

    records = []
    materialized_pair_provenances = []
    materialized_diff_ref_alt_pairs = []
    per_asset = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        cid = p["candidate_sequence_id"]
        rid = cid[:-len(CAND_SUFFIX)]
        study = _study_of(cid)
        # Resolve authority before any downstream skip so a contradictory
        # active pair can never be hidden by a missing canonical record.
        pair_provenance = build_pair_provenance(p)
        rec0 = canonical.get(rid)
        if rec0 is None:
            per_asset[study]["missing_canonical"] += 1
            continue
        spec = config.DELTA_SPECS[study]
        if spec.get("type") == "diff_ref_alt":
            materialized_diff_ref_alt_pairs.append({
                "study": study,
                "pair_id": p["pair_id"],
                "candidate_sequence_id": cid,
            })
        deltas = derive_delta(study, spec, rec0, cand_values,
                              src_114, src_200, cid)
        materialized_pair_provenances.append(pair_provenance)
        for d in deltas:
            d["pair_id"] = p["pair_id"]
            d["pair_provenance"] = dict(pair_provenance)
            records.append(d)
            per_asset[study]["total"] += 1
            per_asset[study][d["delta_source_status"]] += 1

    records = sorted(records, key=_canonical_json)
    assert_ordinary_rows(records, purpose="effect dataset output")
    diff_ref_alt_join_audits = summarize_diff_ref_alt_join_audits(
        records,
        config.DELTA_SPECS,
        cand_values=cand_values,
        materialized_pairs=materialized_diff_ref_alt_pairs,
    )

    input_paths = {
        "pairs": pairs_path,
        "functional_observations": observations_path,
        "canonical": Path(config.CANONICAL_RECORDS),
    }
    if "GSE114002" in config.ACTIVE_STUDIES:
        input_paths["gse114002_raw_library"] = Path(
            config.GSE114002_RAW_LIB
        )
    if "GSE200304" in config.ACTIVE_STUDIES:
        input_paths["gse200304_raw_count"] = Path(config.GSE200304_COUNT)
        input_paths["gse200304_twist"] = Path(config.GSE200304_TWIST)
    for study, raw_path in sorted(config.RECONSTRUCTED_PAIRS.items()):
        path = Path(raw_path)
        if path.exists():
            input_paths["reconstructed_pairs_%s" % study] = path

    anchor_status_counts = defaultdict(int)
    for row in records:
        anchor_status_counts[str(row.get("source_anchor_status"))] += 1

    relation_provenance_summary = {
        "authority_field": RELATION_STATUS_AUTHORITY_FIELD,
        "materialized_pairs": summarize_pair_relation_provenance(
            materialized_pair_provenances
        ),
        "effect_records": summarize_pair_relation_provenance([
            row["pair_provenance"] for row in records
        ]),
    }
    policy_hashes = {
        study: _json_sha256(policy)
        for study, policy in sorted(diff_ref_alt_policies.items())
    }
    if {
        study: audit["join_policy_sha256"]
        for study, audit in sorted(diff_ref_alt_join_audits.items())
    } != policy_hashes:
        raise DiffRefAltJoinPolicyError(
            "replay audit policy hashes disagree with implementation binding"
        )
    anticipated_effect_sha256 = hashlib.sha256()
    for row in records:
        anticipated_effect_sha256.update(
            (_canonical_json(row) + "\n").encode("utf-8")
        )
    anticipated_effect_sha256_hex = anticipated_effect_sha256.hexdigest()
    builder_path = Path(__file__).resolve()
    config_path = Path(config.__file__).resolve()
    implementation_binding = {
        "schema_version": IMPLEMENTATION_BINDING_SCHEMA_VERSION,
        "build_effect_dataset_path": str(builder_path),
        "build_effect_dataset_sha256": sha256_file(builder_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "delta_specs_sha256": _json_sha256(config.DELTA_SPECS),
        "diff_ref_alt_policy_sha256_by_study": policy_hashes,
        "effect_dataset_schema_version": EFFECT_DATASET_SCHEMA_VERSION,
        "manifest_version": EFFECT_DATASET_MANIFEST_VERSION,
        "diff_ref_alt_join_policy_schema_version": (
            DIFF_REF_ALT_JOIN_POLICY_SCHEMA_VERSION
        ),
        "diff_ref_alt_replay_audit_schema_version": (
            DIFF_REF_ALT_JOIN_AUDIT_SCHEMA_VERSION
        ),
        "ordinary_only": True,
        "scientific_unit_authority_created": False,
        "estimand_authority_created": False,
    }
    if set(implementation_binding) != IMPLEMENTATION_BINDING_KEYS:
        raise DiffRefAltJoinPolicyError(
            "internal implementation binding schema keys drifted"
        )

    manifest = {
        "manifest_version": EFFECT_DATASET_MANIFEST_VERSION,
        "effect_dataset_schema_version": EFFECT_DATASET_SCHEMA_VERSION,
        "functional_observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "qualification": {
            "status": MANIFEST_QUALIFICATION_STATUS,
            "dataset_without_manifest": "NOT_QUALIFIED",
            "replay_audit_completed_before_dataset_write": True,
            "scientific_qualification": False,
            "training_ready": False,
            "split_ready": False,
        },
        "implementation_binding": implementation_binding,
        "n_records": len(records),
        "n_pairs_scanned": len(pairs),
        "per_asset": {k: dict(v) for k, v in per_asset.items()},
        "source_anchor_status_counts": dict(
            sorted(anchor_status_counts.items())
        ),
        "pair_relation_provenance": relation_provenance_summary,
        "diff_ref_alt_join_audits": diff_ref_alt_join_audits,
        "legacy_source_lineage": {
            "status": "NOT_PUBLICATION_GROUPING",
            "n_unique_legacy_source_ids": len({
                row.get("legacy_source_id") for row in records
                if row.get("legacy_source_id")
            }),
            "n_unique_source_anchor_identities": len({
                row.get("source_anchor_identity") for row in records
                if row.get("source_anchor_identity")
            }),
            "required_next_step": (
                "R3 ordinary preparation must bind a frozen context and "
                "construct source_group_id_v2 before publication ranking."
            ),
        },
        "sha256": anticipated_effect_sha256_hex,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(input_paths.items())
        },
        "config": {
            "staging": str(staging),
            "canonical": config.CANONICAL_RECORDS,
            "split": config.PRIMARY_SPLIT,
            "seed": config.SEED,
        },
    }
    out_dir = Path(out_dir)
    assert_ordinary_path(out_dir, purpose="effect dataset output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "effect_dataset.jsonl"
    man_path = out_dir / "effect_dataset_manifest.json"
    assert_ordinary_path(out_path, purpose="effect dataset output")
    assert_ordinary_path(man_path, purpose="effect dataset manifest output")
    if out_path.exists() or man_path.exists():
        raise FileExistsError(
            "effect builder refuses to overwrite existing outputs: %s, %s"
            % (out_path, man_path)
        )
    with out_path.open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(_canonical_json(row) + "\n")
    if sha256_file(out_path) != anticipated_effect_sha256_hex:
        raise IOError(
            "written effect dataset differs from pre-write audited bytes"
        )
    with man_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False
        ) + "\n")
    return manifest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/b0x"))
    args = ap.parse_args()
    man = build(args.out_dir)
    print(json.dumps({k: (v if k != "per_asset" else v) for k, v in man.items() if k != "sha256"}, indent=2))
    print("sha256:", man["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
