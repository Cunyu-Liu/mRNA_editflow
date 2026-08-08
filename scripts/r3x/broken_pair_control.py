"""Inactive candidate protocol for a replayable broken-pair train control.

This module deliberately does not integrate with the active negative-control
gate.  It specifies and replays a candidate tensorizer that paired-sequence
models can consume without pretending that a deliberately broken pair has an
edit script.  Activation requires a future, independently frozen versioned
policy authority and a separate protocol revision.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .protocol import ProtocolHoldError, assert_ordinary_record


POLICY_SCHEMA = "r3x_broken_pair_tensorizer_policy_candidate_v1"
POLICY_ARTIFACT_ROLE = "ORDINARY_TRAIN_BROKEN_PAIR_TENSORIZER_POLICY_CANDIDATE"
MANIFEST_SCHEMA = "r3x_broken_pair_tensor_manifest_candidate_v1"
MANIFEST_ARTIFACT_ROLE = "ORDINARY_TRAIN_BROKEN_PAIR_TENSOR_CANDIDATE"
TENSOR_ROW_SCHEMA = "r3x_broken_pair_tensor_row_candidate_v1"
ROW_IDENTITY_SCHEMA = "r3x_broken_pair_target_free_row_identity_v1"
CONTROL_ID = "SOURCE_CANDIDATE_PAIRING_SHUFFLE"
INACTIVE_STATUS = "PROPOSED_NOT_ACTIVE"

FROZEN_SEEDS = (11, 23, 42, 67, 101)
ELIGIBLE_METHOD_IDS = (
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
    "R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION",
    "A_EXACT_POTENTIAL_DIFFERENCE",
    "A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL",
)
PROHIBITED_METHOD_FAMILIES = (
    "B_SPARSE_EDIT_SET_ADDITIVE",
    "B_SPARSE_EDIT_SET_INTERACTION",
    "SPARSE_EDIT_SET",
)
REQUIRED_FUTURE_AUTHORITY_BINDINGS = (
    "TARGET_FREE_ORDINARY_TRAIN_PROJECTION_AUTHORITY",
    "ORDINARY_TRAIN_SPLIT_MEMBERSHIP_AUTHORITY",
    "VERSIONED_BROKEN_PAIR_POLICY_AUTHORITY",
)
CONDITION_KEYS = (
    "study",
    "estimand_id",
    "endpoint",
    "publication_context_id",
    "region",
    "edit_count",
)
CELL_KEYS = CONDITION_KEYS[:-1]
INPUT_ROW_KEYS = {
    "row_id",
    "partition",
    "source_sequence",
    "candidate_sequence",
    "target",
    "condition",
}
TENSOR_INPUT_KEYS = INPUT_ROW_KEYS
TENSOR_ROW_KEYS = {
    "schema_version",
    "receiver_row_id",
    "donor_row_id",
    "tensor_input",
    "audit",
}
TENSOR_AUDIT_KEYS = {
    "stratum_id",
    "receiver_original_candidate_sha256",
    "donor_candidate_sha256",
    "receiver_binding_sha256",
    "broken_pair_sha256",
    "candidate_changed",
    "donor_differs",
    "equal_length_acgu",
    "edit_metadata_tensor_input",
    "source_candidate_replay_required",
}
STRATUM_SUMMARY_KEYS = {
    "stratum_id",
    "cell",
    "source_length",
    "candidate_length",
    "edit_count",
    "n_rows",
    "input_candidate_marginal_sha256",
    "output_candidate_marginal_sha256",
    "permutation_sha256",
    "status",
    "reason",
}
MANIFEST_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "candidate_execution_status",
    "control_id",
    "method_id",
    "seed",
    "policy",
    "policy_sha256",
    "partition_scope",
    "input_rows",
    "input_rows_sha256",
    "tensor_rows",
    "tensor_rows_sha256",
    "n_rows",
    "n_strata",
    "ordered_row_ids_sha256",
    "receiver_identity_source_target_condition_sha256",
    "input_candidate_marginal_sha256",
    "output_candidate_marginal_sha256",
    "permutation_sha256",
    "strata",
    "hold_reason",
    "original_edit_metadata_tensor_input",
    "outcome_mutation",
    "development_access",
    "internal_test_access",
    "sealed_access",
    "can_satisfy_pre_test_freeze",
    "authority_binding_status",
    "required_future_authority_bindings",
    "required_future_activation",
}

HEX64 = re.compile(r"^[0-9a-f]{64}$")


class BrokenPairControlError(RuntimeError):
    """Fail-closed candidate-policy or tensorizer violation."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


def _fail(code: str, message: str) -> None:
    raise BrokenPairControlError(code, message)


def _compact_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BrokenPairControlError(
            "HOLD_NON_CANONICAL_BROKEN_PAIR_JSON", str(exc)
        ) from exc


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_compact_json_bytes(value)).hexdigest()


def _exact_keys(value: object, expected: set[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _fail("HOLD_BROKEN_PAIR_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _fail(
            "HOLD_BROKEN_PAIR_SCHEMA",
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}",
        )
    return value


def _policy_template() -> dict[str, Any]:
    return {
        "schema_version": POLICY_SCHEMA,
        "artifact_role": POLICY_ARTIFACT_ROLE,
        "status": INACTIVE_STATUS,
        "control_id": CONTROL_ID,
        "policy_version": "CANDIDATE_V1_REQUIRES_FUTURE_AUTHORITY_ACTIVATION",
        "scope": "ORDINARY_TRAIN_PARTITION_ONLY",
        "eligible_method_ids": list(ELIGIBLE_METHOD_IDS),
        "prohibited_method_families": list(PROHIBITED_METHOD_FAMILIES),
        "fixed_seeds": list(FROZEN_SEEDS),
        "stratum_fields": [
            *CELL_KEYS,
            "source_length",
            "candidate_length",
            "edit_count",
        ],
        "permutation_algorithm": ("SHA256_SEEDED_CANONICAL_BIPARTITE_DERANGEMENT_V1"),
        "candidate_marginal_policy": "EXACT_MULTISET_PRESERVATION_WITHIN_STRATUM",
        "receiver_fields_preserved": [
            "row_id",
            "partition",
            "source_sequence",
            "target",
            "condition",
        ],
        "row_identity_policy": (
            "SHA256_CANONICAL_TARGET_FREE_SOURCE_CANDIDATE_CONDITION_V1;"
            "ARBITRARY_OR_OUTCOME_BEARING_ROW_IDS_PROHIBITED"
        ),
        "tensor_input_fields": [
            "row_id",
            "partition",
            "source_sequence",
            "candidate_sequence",
            "target",
            "condition",
        ],
        "edit_metadata_policy": "PROHIBITED_FROM_TENSOR_INPUT",
        "sequence_policy": (
            "SOURCE_AND_DONOR_CANDIDATE_MUST_BE_EQUAL_LENGTH_STRICT_ACGU;"
            "EDIT_REPLAY_IS_INTENTIONALLY_UNDEFINED"
        ),
        "support_policy": (
            "EVERY_STRATUM_REQUIRES_A_PERFECT_DONOR_DERANGEMENT_WITH_"
            "DONOR_ROW_DIFFERENT_AND_CANDIDATE_SEQUENCE_CHANGED;OTHERWISE_NOT_RUN"
        ),
        "activation_status": "NO_VERSIONED_POLICY_AUTHORITY_ACTIVE",
        "required_future_authority_bindings": list(REQUIRED_FUTURE_AUTHORITY_BINDINGS),
        "pre_test_freeze_effect": "CANNOT_SATISFY_OR_ADVANCE_PRE_TEST_FREEZE",
        "ordinary_only": True,
        "development_access": False,
        "internal_test_access": False,
        "sealed_access": False,
    }


def build_candidate_policy() -> dict[str, Any]:
    """Return the sole exact inactive policy payload."""

    return copy.deepcopy(_policy_template())


def validate_candidate_policy(policy: Mapping) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        _fail("HOLD_BROKEN_PAIR_POLICY_SCHEMA", "candidate policy must be an object")
    expected = _policy_template()
    if _compact_json_bytes(dict(policy)) != _compact_json_bytes(expected):
        _fail(
            "HOLD_BROKEN_PAIR_POLICY_SCHEMA",
            "candidate policy differs from the exact inactive schema",
        )
    return expected


def candidate_policy_sha256(policy: Mapping) -> str:
    normalized = validate_candidate_policy(policy)
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def _strict_acgu(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(base not in "ACGU" for base in value)
    ):
        _fail(
            "HOLD_BROKEN_PAIR_SEQUENCE",
            f"{label} must be non-empty uppercase A/C/G/U",
        )
    return value


def _condition(value: object, *, row_id: str) -> dict[str, Any]:
    condition = _exact_keys(value, set(CONDITION_KEYS), label=f"condition for {row_id}")
    normalized: dict[str, Any] = {}
    for field in CELL_KEYS:
        item = condition[field]
        if not isinstance(item, str) or not item or item != item.strip():
            _fail(
                "HOLD_BROKEN_PAIR_CONDITION",
                f"{row_id} condition {field} must be a non-empty canonical string",
            )
        normalized[field] = item
    edit_count = condition["edit_count"]
    if (
        isinstance(edit_count, bool)
        or not isinstance(edit_count, int)
        or edit_count not in {1, 3, 5}
    ):
        _fail(
            "HOLD_BROKEN_PAIR_CONDITION",
            f"{row_id} edit_count must be one of 1, 3, or 5",
        )
    normalized["edit_count"] = edit_count
    return normalized


def _target_free_identity_payload(
    source_sequence: str,
    candidate_sequence: str,
    condition: Mapping,
) -> dict[str, Any]:
    return {
        "schema_version": ROW_IDENTITY_SCHEMA,
        "partition": "train",
        "source_sequence": source_sequence,
        "candidate_sequence": candidate_sequence,
        "condition": copy.deepcopy(dict(condition)),
    }


def derive_target_free_row_id(
    *,
    source_sequence: object,
    candidate_sequence: object,
    condition: object,
) -> str:
    """Derive the only row identity accepted by the candidate tensorizer.

    Target and arbitrary caller labels are deliberately absent, so frozen-seed
    donor assignment cannot change when only receiver outcomes change.
    """

    source = _strict_acgu(source_sequence, label="row identity source")
    candidate = _strict_acgu(candidate_sequence, label="row identity candidate")
    if len(source) != len(candidate):
        _fail(
            "HOLD_BROKEN_PAIR_SEQUENCE",
            "row identity source and candidate must have equal length",
        )
    normalized_condition = _condition(condition, row_id="target-free row identity")
    return _json_sha256(
        _target_free_identity_payload(source, candidate, normalized_condition)
    )


def _normalize_input_rows(rows: Sequence[Mapping]) -> list[dict[str, Any]]:
    raw_rows = list(rows)
    if not raw_rows:
        _fail("HOLD_EMPTY_BROKEN_PAIR_TRAIN_INPUT", "tensorizer refuses empty input")

    # This first pass intentionally touches no sequence, target, condition, or
    # other member field.  Non-train data is rejected before it can be parsed.
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            _fail("HOLD_BROKEN_PAIR_SCHEMA", f"input row {index} must be an object")
        partition = row.get("partition")
        if type(partition) is not str or partition != "train":
            _fail(
                "HOLD_BROKEN_PAIR_TRAIN_ONLY",
                "broken-pair tensorizer accepts partition=train only",
            )

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_rows):
        exact = _exact_keys(row, INPUT_ROW_KEYS, label=f"input row {index}")
        try:
            assert_ordinary_record(exact, purpose="broken-pair train tensor input")
        except ProtocolHoldError as exc:
            raise BrokenPairControlError(
                "HOLD_BROKEN_PAIR_ORDINARY_ONLY", str(exc)
            ) from exc
        row_id = exact["row_id"]
        if not isinstance(row_id, str) or not HEX64.fullmatch(row_id):
            _fail(
                "HOLD_BROKEN_PAIR_ROW_IDENTITY",
                "row_id must be a lowercase SHA-256 identity",
            )
        source = _strict_acgu(exact["source_sequence"], label=f"{row_id} source")
        candidate = _strict_acgu(
            exact["candidate_sequence"], label=f"{row_id} candidate"
        )
        if len(source) != len(candidate):
            _fail(
                "HOLD_BROKEN_PAIR_SEQUENCE",
                f"{row_id} source and candidate must have equal length",
            )
        condition = _condition(exact["condition"], row_id=row_id)
        expected_row_id = _json_sha256(
            _target_free_identity_payload(source, candidate, condition)
        )
        if row_id != expected_row_id or row_id in seen:
            _fail(
                "HOLD_BROKEN_PAIR_ROW_IDENTITY",
                "row_id must be the unique canonical target-free identity hash",
            )
        seen.add(row_id)
        target = exact["target"]
        try:
            finite_target = (
                not isinstance(target, bool)
                and isinstance(target, (int, float))
                and math.isfinite(float(target))
            )
        except (OverflowError, TypeError, ValueError):
            finite_target = False
        if not finite_target:
            _fail(
                "HOLD_BROKEN_PAIR_TARGET",
                f"{row_id} receiver target must be a finite number",
            )
        normalized.append(
            {
                "row_id": row_id,
                "partition": "train",
                "source_sequence": source,
                "candidate_sequence": candidate,
                "target": target,
                "condition": condition,
            }
        )
    return sorted(normalized, key=lambda item: item["row_id"])


def _validate_method(method_id: object) -> str:
    if not isinstance(method_id, str) or not method_id:
        _fail("HOLD_BROKEN_PAIR_METHOD", "method_id must be a non-empty string")
    if method_id.startswith("B_") or "SPARSE_EDIT_SET" in method_id:
        _fail(
            "HOLD_BROKEN_PAIR_B_SPARSE_EDIT_SET_PROHIBITED",
            "B/SparseEditSet cannot consume a tensor without exact edit metadata",
        )
    if method_id not in ELIGIBLE_METHOD_IDS:
        _fail(
            "HOLD_BROKEN_PAIR_METHOD",
            f"method is not preregistered for this candidate tensorizer: {method_id}",
        )
    return method_id


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in FROZEN_SEEDS:
        _fail(
            "HOLD_BROKEN_PAIR_SEED",
            f"seed must be one of the fixed seeds {list(FROZEN_SEEDS)}",
        )
    return seed


def _stratum_payload(row: Mapping) -> dict[str, Any]:
    condition = row["condition"]
    return {
        "cell": {field: condition[field] for field in CELL_KEYS},
        "source_length": len(row["source_sequence"]),
        "candidate_length": len(row["candidate_sequence"]),
        "edit_count": condition["edit_count"],
    }


def _donor_priority(seed: int, stratum_id: str, receiver: str, donor: str) -> str:
    return _json_sha256(
        {
            "algorithm": "SHA256_SEEDED_CANONICAL_BIPARTITE_DERANGEMENT_V1",
            "seed": seed,
            "stratum_id": stratum_id,
            "receiver_row_id": receiver,
            "donor_row_id": donor,
        }
    )


def _perfect_derangement(
    group: Sequence[Mapping], *, seed: int, stratum_id: str
) -> dict[str, str] | None:
    """Return a deterministic perfect receiver->donor matching if one exists."""

    donor_for_receiver: dict[str, str] = {}
    receiver_for_donor: dict[str, str] = {}
    by_id = {str(row["row_id"]): row for row in group}
    receiver_ids = sorted(by_id)
    donor_orders = {
        receiver_id: sorted(
            (
                donor_id
                for donor_id in receiver_ids
                if donor_id != receiver_id
                and by_id[donor_id]["candidate_sequence"]
                != by_id[receiver_id]["candidate_sequence"]
            ),
            key=lambda donor_id: (
                _donor_priority(seed, stratum_id, receiver_id, donor_id),
                donor_id,
            ),
        )
        for receiver_id in receiver_ids
    }

    def augment(root_receiver_id: str) -> bool:
        # Use an iterative alternating-path search.  A recursive Kuhn search
        # can hit Python's recursion limit on a scientifically valid large
        # stratum and turn available support into an implementation crash.
        receiver_queue = [root_receiver_id]
        seen_receivers = {root_receiver_id}
        parent_receiver_for_donor: dict[str, str] = {}
        free_donor_id: str | None = None
        queue_index = 0
        while queue_index < len(receiver_queue) and free_donor_id is None:
            receiver_id = receiver_queue[queue_index]
            queue_index += 1
            for donor_id in donor_orders[receiver_id]:
                if donor_id in parent_receiver_for_donor:
                    continue
                parent_receiver_for_donor[donor_id] = receiver_id
                matched_receiver = receiver_for_donor.get(donor_id)
                if matched_receiver is None:
                    free_donor_id = donor_id
                    break
                if matched_receiver not in seen_receivers:
                    seen_receivers.add(matched_receiver)
                    receiver_queue.append(matched_receiver)
        if free_donor_id is None:
            return False

        donor_id: str | None = free_donor_id
        while donor_id is not None:
            receiver_id = parent_receiver_for_donor[donor_id]
            previous_donor = donor_for_receiver.get(receiver_id)
            receiver_for_donor[donor_id] = receiver_id
            donor_for_receiver[receiver_id] = donor_id
            donor_id = previous_donor
        return True

    for receiver_id in receiver_ids:
        if not augment(receiver_id):
            return None
    if set(donor_for_receiver) != set(receiver_ids):
        return None
    if set(donor_for_receiver.values()) != set(receiver_ids):
        return None
    return dict(sorted(donor_for_receiver.items()))


def _receiver_binding(row: Mapping) -> dict[str, Any]:
    return {
        "row_id": row["row_id"],
        "partition": row["partition"],
        "source_sequence": row["source_sequence"],
        "target": row["target"],
        "condition": copy.deepcopy(row["condition"]),
    }


def _tensor_row(
    receiver: Mapping,
    donor: Mapping,
    *,
    stratum_id: str,
) -> dict[str, Any]:
    tensor_input = {
        "row_id": receiver["row_id"],
        "partition": "train",
        "source_sequence": receiver["source_sequence"],
        "candidate_sequence": donor["candidate_sequence"],
        "target": receiver["target"],
        "condition": copy.deepcopy(receiver["condition"]),
    }
    receiver_binding = _receiver_binding(receiver)
    return {
        "schema_version": TENSOR_ROW_SCHEMA,
        "receiver_row_id": receiver["row_id"],
        "donor_row_id": donor["row_id"],
        "tensor_input": tensor_input,
        "audit": {
            "stratum_id": stratum_id,
            "receiver_original_candidate_sha256": hashlib.sha256(
                receiver["candidate_sequence"].encode("ascii")
            ).hexdigest(),
            "donor_candidate_sha256": hashlib.sha256(
                donor["candidate_sequence"].encode("ascii")
            ).hexdigest(),
            "receiver_binding_sha256": _json_sha256(receiver_binding),
            "broken_pair_sha256": _json_sha256(tensor_input),
            "candidate_changed": True,
            "donor_differs": True,
            "equal_length_acgu": True,
            "edit_metadata_tensor_input": False,
            "source_candidate_replay_required": False,
        },
    }


def _candidate_marginal_payload(
    groups: Mapping[str, Sequence[Mapping]], *, tensor_rows: Sequence[Mapping] | None
) -> list[dict[str, Any]]:
    by_stratum_output: dict[str, list[str]] = {}
    if tensor_rows is not None:
        for row in tensor_rows:
            stratum_id = row["audit"]["stratum_id"]
            by_stratum_output.setdefault(stratum_id, []).append(
                row["tensor_input"]["candidate_sequence"]
            )
    payload = []
    for stratum_id, group in sorted(groups.items()):
        candidates = (
            [str(row["candidate_sequence"]) for row in group]
            if tensor_rows is None
            else by_stratum_output.get(stratum_id, [])
        )
        payload.append(
            {"stratum_id": stratum_id, "candidate_sequences": sorted(candidates)}
        )
    return payload


def _assemble_manifest(
    input_rows: list[dict[str, Any]],
    *,
    seed: int,
    method_id: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    stratum_payloads: dict[str, dict[str, Any]] = {}
    for row in input_rows:
        payload = _stratum_payload(row)
        stratum_id = _json_sha256(payload)
        groups.setdefault(stratum_id, []).append(row)
        stratum_payloads[stratum_id] = payload
    for group in groups.values():
        group.sort(key=lambda item: item["row_id"])

    tensor_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    permutations: list[dict[str, Any]] = []
    failed_reasons: list[str] = []
    for stratum_id, group in sorted(groups.items()):
        payload = stratum_payloads[stratum_id]
        input_marginal = _json_sha256(
            sorted(str(row["candidate_sequence"]) for row in group)
        )
        matching = _perfect_derangement(group, seed=seed, stratum_id=stratum_id)
        if matching is None:
            reason = (
                "no perfect donor derangement with donor_row_id different and "
                "candidate_sequence changed"
            )
            failed_reasons.append(f"{stratum_id}:{reason}")
            summaries.append(
                {
                    "stratum_id": stratum_id,
                    **payload,
                    "n_rows": len(group),
                    "input_candidate_marginal_sha256": input_marginal,
                    "output_candidate_marginal_sha256": None,
                    "permutation_sha256": None,
                    "status": "NOT_RUN_WITH_REASON",
                    "reason": reason,
                }
            )
            continue

        by_id = {row["row_id"]: row for row in group}
        group_tensor_rows = [
            _tensor_row(
                by_id[receiver_id],
                by_id[donor_id],
                stratum_id=stratum_id,
            )
            for receiver_id, donor_id in sorted(matching.items())
        ]
        output_marginal = _json_sha256(
            sorted(
                row["tensor_input"]["candidate_sequence"] for row in group_tensor_rows
            )
        )
        if output_marginal != input_marginal:
            _fail(
                "HOLD_BROKEN_PAIR_MARGINAL",
                "internal error: perfect matching changed candidate marginal",
            )
        permutation_payload = [
            {"receiver_row_id": receiver, "donor_row_id": donor}
            for receiver, donor in sorted(matching.items())
        ]
        permutation_sha = _json_sha256(permutation_payload)
        permutations.extend(
            {"stratum_id": stratum_id, **item} for item in permutation_payload
        )
        tensor_rows.extend(group_tensor_rows)
        summaries.append(
            {
                "stratum_id": stratum_id,
                **payload,
                "n_rows": len(group),
                "input_candidate_marginal_sha256": input_marginal,
                "output_candidate_marginal_sha256": output_marginal,
                "permutation_sha256": permutation_sha,
                "status": "CANDIDATE_TENSORS_BUILT",
                "reason": None,
            }
        )

    if failed_reasons:
        # A control is atomic across strata.  Partial perturbations would
        # silently change its estimand, so no tensor row is released.
        tensor_rows = []
        permutations = []
        atomic_reason = (
            "atomic control held because at least one stratum lacks a perfect "
            "donor derangement; no partial tensor artifact is released"
        )
        summaries = [
            {
                **summary,
                "output_candidate_marginal_sha256": None,
                "permutation_sha256": None,
                "status": "NOT_RUN_WITH_REASON",
                "reason": (
                    summary["reason"]
                    if summary["status"] == "NOT_RUN_WITH_REASON"
                    else atomic_reason
                ),
            }
            for summary in summaries
        ]
        candidate_execution_status = "NOT_RUN_WITH_REASON"
        hold_reason = "PRE_TEST_FREEZE_HOLD; NOT_RUN_IS_NOT_PASS; " + "; ".join(
            sorted(failed_reasons)
        )
        output_marginal_sha = None
        permutation_sha = None
    else:
        tensor_rows.sort(key=lambda item: item["receiver_row_id"])
        candidate_execution_status = "CANDIDATE_TENSORS_BUILT"
        hold_reason = None
        output_marginal_sha = _json_sha256(
            _candidate_marginal_payload(groups, tensor_rows=tensor_rows)
        )
        permutation_sha = _json_sha256(sorted(permutations, key=_compact_json_bytes))

    ordered_ids = [row["row_id"] for row in input_rows]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "artifact_role": MANIFEST_ARTIFACT_ROLE,
        "status": INACTIVE_STATUS,
        "candidate_execution_status": candidate_execution_status,
        "control_id": CONTROL_ID,
        "method_id": method_id,
        "seed": seed,
        "policy": copy.deepcopy(policy),
        "policy_sha256": candidate_policy_sha256(policy),
        "partition_scope": "ORDINARY_TRAIN_PARTITION_ONLY",
        "input_rows": copy.deepcopy(input_rows),
        "input_rows_sha256": _json_sha256(input_rows),
        "tensor_rows": tensor_rows,
        "tensor_rows_sha256": _json_sha256(tensor_rows),
        "n_rows": len(input_rows),
        "n_strata": len(groups),
        "ordered_row_ids_sha256": _json_sha256(ordered_ids),
        "receiver_identity_source_target_condition_sha256": _json_sha256(
            [_receiver_binding(row) for row in input_rows]
        ),
        "input_candidate_marginal_sha256": _json_sha256(
            _candidate_marginal_payload(groups, tensor_rows=None)
        ),
        "output_candidate_marginal_sha256": output_marginal_sha,
        "permutation_sha256": permutation_sha,
        "strata": summaries,
        "hold_reason": hold_reason,
        "original_edit_metadata_tensor_input": False,
        "outcome_mutation": False,
        "development_access": False,
        "internal_test_access": False,
        "sealed_access": False,
        "can_satisfy_pre_test_freeze": False,
        "authority_binding_status": "NOT_BOUND_CANDIDATE_ONLY",
        "required_future_authority_bindings": list(REQUIRED_FUTURE_AUTHORITY_BINDINGS),
        "required_future_activation": (
            "INDEPENDENT_FROZEN_VERSIONED_BROKEN_PAIR_POLICY_AUTHORITY_"
            "AND_PROTOCOL_REVISION"
        ),
    }


def build_broken_pair_candidate(
    rows: Sequence[Mapping],
    *,
    seed: int,
    method_id: str,
    policy: Mapping | None = None,
) -> dict[str, Any]:
    """Build a deterministic train-only candidate artifact.

    A successfully constructed candidate remains ``PROPOSED_NOT_ACTIVE`` and
    cannot satisfy the existing PRE_TEST_FREEZE gate.
    """

    normalized_policy = validate_candidate_policy(
        build_candidate_policy() if policy is None else policy
    )
    normalized_method = _validate_method(method_id)
    normalized_seed = _validate_seed(seed)
    normalized_rows = _normalize_input_rows(rows)
    manifest = _assemble_manifest(
        normalized_rows,
        seed=normalized_seed,
        method_id=normalized_method,
        policy=normalized_policy,
    )
    return validate_broken_pair_manifest(manifest)


def _validate_tensor_rows(
    tensor_rows: object, *, input_by_id: Mapping[str, Mapping]
) -> None:
    if not isinstance(tensor_rows, list):
        _fail("HOLD_BROKEN_PAIR_SCHEMA", "tensor_rows must be a list")
    seen: set[str] = set()
    for index, raw in enumerate(tensor_rows):
        row = _exact_keys(raw, TENSOR_ROW_KEYS, label=f"tensor row {index}")
        if row.get("schema_version") != TENSOR_ROW_SCHEMA:
            _fail("HOLD_BROKEN_PAIR_SCHEMA", "tensor row schema is unsupported")
        receiver_id = row.get("receiver_row_id")
        donor_id = row.get("donor_row_id")
        if receiver_id not in input_by_id or donor_id not in input_by_id:
            _fail("HOLD_BROKEN_PAIR_ROW_IDENTITY", "tensor row names an unknown row")
        if receiver_id in seen or receiver_id == donor_id:
            _fail(
                "HOLD_BROKEN_PAIR_DERANGEMENT",
                "receiver identities must be unique and donor must differ",
            )
        seen.add(receiver_id)
        tensor_input = _exact_keys(
            row.get("tensor_input"), TENSOR_INPUT_KEYS, label="tensor_input"
        )
        audit = _exact_keys(row.get("audit"), TENSOR_AUDIT_KEYS, label="tensor audit")
        receiver = input_by_id[receiver_id]
        donor = input_by_id[donor_id]
        expected_tensor = {
            "row_id": receiver_id,
            "partition": "train",
            "source_sequence": receiver["source_sequence"],
            "candidate_sequence": donor["candidate_sequence"],
            "target": receiver["target"],
            "condition": receiver["condition"],
        }
        if _compact_json_bytes(tensor_input) != _compact_json_bytes(expected_tensor):
            _fail(
                "HOLD_BROKEN_PAIR_RECEIVER_MUTATION",
                "tensor row changed receiver identity/source/target/condition",
            )
        if donor["candidate_sequence"] == receiver["candidate_sequence"]:
            _fail(
                "HOLD_BROKEN_PAIR_DERANGEMENT",
                "tensor row did not change the candidate branch",
            )
        if len(receiver["source_sequence"]) != len(donor["candidate_sequence"]):
            _fail(
                "HOLD_BROKEN_PAIR_SEQUENCE", "broken pair has unequal sequence length"
            )
        expected_audit = _tensor_row(
            receiver,
            donor,
            stratum_id=audit.get("stratum_id"),
        )["audit"]
        if _compact_json_bytes(audit) != _compact_json_bytes(expected_audit):
            _fail("HOLD_BROKEN_PAIR_REPLAY", "tensor audit does not replay")


def validate_broken_pair_manifest(payload: Mapping) -> dict[str, Any]:
    manifest = _exact_keys(payload, MANIFEST_KEYS, label="broken-pair manifest")
    exact_scalars = {
        "schema_version": MANIFEST_SCHEMA,
        "artifact_role": MANIFEST_ARTIFACT_ROLE,
        "status": INACTIVE_STATUS,
        "control_id": CONTROL_ID,
        "partition_scope": "ORDINARY_TRAIN_PARTITION_ONLY",
        "original_edit_metadata_tensor_input": False,
        "outcome_mutation": False,
        "development_access": False,
        "internal_test_access": False,
        "sealed_access": False,
        "can_satisfy_pre_test_freeze": False,
        "authority_binding_status": "NOT_BOUND_CANDIDATE_ONLY",
        "required_future_authority_bindings": list(REQUIRED_FUTURE_AUTHORITY_BINDINGS),
        "required_future_activation": (
            "INDEPENDENT_FROZEN_VERSIONED_BROKEN_PAIR_POLICY_AUTHORITY_"
            "AND_PROTOCOL_REVISION"
        ),
    }
    for field, expected in exact_scalars.items():
        if manifest.get(field) != expected:
            _fail(
                "HOLD_BROKEN_PAIR_SCHEMA",
                f"manifest {field} must remain {expected!r}",
            )
    policy = validate_candidate_policy(manifest.get("policy"))
    if manifest.get("policy_sha256") != candidate_policy_sha256(policy):
        _fail("HOLD_BROKEN_PAIR_POLICY_HASH", "candidate policy hash mismatch")
    method_id = _validate_method(manifest.get("method_id"))
    seed = _validate_seed(manifest.get("seed"))
    input_rows_raw = manifest.get("input_rows")
    if not isinstance(input_rows_raw, list):
        _fail("HOLD_BROKEN_PAIR_SCHEMA", "manifest input_rows must be a list")
    input_rows = _normalize_input_rows(input_rows_raw)
    input_by_id = {row["row_id"]: row for row in input_rows}
    _validate_tensor_rows(manifest.get("tensor_rows"), input_by_id=input_by_id)
    strata = manifest.get("strata")
    if not isinstance(strata, list):
        _fail("HOLD_BROKEN_PAIR_SCHEMA", "manifest strata must be a list")
    for index, summary in enumerate(strata):
        _exact_keys(summary, STRATUM_SUMMARY_KEYS, label=f"stratum summary {index}")

    expected = _assemble_manifest(
        input_rows,
        seed=seed,
        method_id=method_id,
        policy=policy,
    )
    if _compact_json_bytes(dict(manifest)) != _compact_json_bytes(expected):
        _fail(
            "HOLD_BROKEN_PAIR_REPLAY",
            "manifest differs from deterministic tensorizer replay",
        )
    return expected


def _parse_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BrokenPairControlError(
            "HOLD_BROKEN_PAIR_JSON", f"{label} is not strict JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        _fail("HOLD_BROKEN_PAIR_JSON", f"{label} must be an object")
    return payload


def write_candidate_policy_exclusive(path: Path, policy: Mapping | None = None) -> str:
    normalized = validate_candidate_policy(
        build_candidate_policy() if policy is None else policy
    )
    try:
        return atomic_write_json_exclusive(
            Path(path), normalized, validator=validate_candidate_policy
        )
    except ArtifactIOError as exc:
        raise BrokenPairControlError("HOLD_BROKEN_PAIR_ARTIFACT_IO", str(exc)) from exc


def replay_candidate_policy(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        raw = read_exact_regular_file(Path(path), expected_sha256)
    except ArtifactIOError as exc:
        raise BrokenPairControlError("HOLD_BROKEN_PAIR_ARTIFACT_IO", str(exc)) from exc
    payload = _parse_json_object(raw, label="candidate policy")
    normalized = validate_candidate_policy(payload)
    if canonical_json_bytes(normalized) != raw:
        _fail(
            "HOLD_BROKEN_PAIR_NONCANONICAL", "candidate policy bytes are not canonical"
        )
    return normalized


def write_broken_pair_manifest_exclusive(path: Path, payload: Mapping) -> str:
    normalized = validate_broken_pair_manifest(payload)
    try:
        return atomic_write_json_exclusive(
            Path(path), normalized, validator=validate_broken_pair_manifest
        )
    except ArtifactIOError as exc:
        raise BrokenPairControlError("HOLD_BROKEN_PAIR_ARTIFACT_IO", str(exc)) from exc


def replay_broken_pair_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        raw = read_exact_regular_file(Path(path), expected_sha256)
    except ArtifactIOError as exc:
        raise BrokenPairControlError("HOLD_BROKEN_PAIR_ARTIFACT_IO", str(exc)) from exc
    payload = _parse_json_object(raw, label="broken-pair manifest")
    normalized = validate_broken_pair_manifest(payload)
    if canonical_json_bytes(normalized) != raw:
        _fail("HOLD_BROKEN_PAIR_NONCANONICAL", "manifest bytes are not canonical")
    return normalized


__all__ = [
    "BrokenPairControlError",
    "CONTROL_ID",
    "ELIGIBLE_METHOD_IDS",
    "FROZEN_SEEDS",
    "INACTIVE_STATUS",
    "MANIFEST_SCHEMA",
    "POLICY_SCHEMA",
    "PROHIBITED_METHOD_FAMILIES",
    "REQUIRED_FUTURE_AUTHORITY_BINDINGS",
    "build_broken_pair_candidate",
    "build_candidate_policy",
    "candidate_policy_sha256",
    "derive_target_free_row_id",
    "replay_broken_pair_manifest",
    "replay_candidate_policy",
    "validate_broken_pair_manifest",
    "validate_candidate_policy",
    "write_broken_pair_manifest_exclusive",
    "write_candidate_policy_exclusive",
]
