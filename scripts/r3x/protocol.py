"""Fail-closed protocol primitives for the R3 ordinary-only route.

The legacy v1 activation graph contains one monolithic ready dataset with
train, development, and internal-test outcomes.  Its validator is therefore an
outcome-bearing development audit, not a blind pre-test capability.  Real or
confirmatory callers must first pass the staged-disclosure guard in this
module; that guard currently fails closed until physically separate
train/development outcomes, internal-test features, and internal-test outcomes
are implemented and predictor provenance is bound into the selector.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, NoReturn, Optional


ROUTE_AUTHORIZATION_SCHEMA = "r3_route_authorization_v1"
ACTIVATION_MANIFEST_SCHEMA = "r3_activation_manifest_v1"
STAGED_DISCLOSURE_SCHEMA = "r3_activation_manifest_v2_staged_disclosure"
STAGED_DISCLOSURE_HOLD_CODE = (
    "HOLD_STAGED_DISCLOSURE_AND_SELECTOR_PROVENANCE_SCHEMA_REQUIRED_"
    "BEFORE_READY_OUTCOME_VALIDATION"
)
AUTHORIZED_ROUTE = "BENCHMARK_CRITIC_ROUTE_A_THEN_B_ORDINARY_ONLY"
REQUIRED_NEGATIVE_CONTROLS = [
    "SOURCE_CANDIDATE_PAIRING_SHUFFLE",
    "CONTEXT_LABEL_PERMUTATION",
    "A_IDENTITY_AND_SWAP_INVARIANTS",
    "SOURCE_CONTEXT_SHUFFLE_WITHIN_ACTION",
    "EDIT_POSITION_SHUFFLE_WITHIN_SOURCE",
    "TARGET_TOKEN_SHUFFLE_WITHIN_CELL",
    "B_ADDITIVE_VERSUS_INTERACTION",
]
AUTHORIZED_SEQUENCE = (
    "PUBLICATION_GRADE_STRONG_BASELINES",
    "A_EXACT_POTENTIAL_DIFFERENCE",
    "A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL",
    "B_SPARSE_EDIT_SET_ADDITIVE",
    "B_SPARSE_EDIT_SET_INTERACTION_ONLY_IF_SEPARATELY_QUALIFIED",
)
SIGN_ACCURACY_GATE = 0.60
REQUIRED_ACTIVATION_AUTHORITIES = (
    "route_authorization",
    "ready_dataset",
    "inclusion_manifest",
    "context_source_crosswalk",
    "endpoint_estimand_registry",
    "source_component_split_manifest",
    "metric_zero_policy_registry",
    "model_optimizer_seed_config",
    "analysis_code_tree_manifest",
    "method_protocol",
    "paired_strongest_baseline_comparison_policy",
)

REQUIRED_STAGED_DISCLOSURE_CAPABILITIES = frozenset({
    "train_development_authority",
    "internal_test_feature_authority",
    "internal_test_outcome_authority",
    "selector_predictor_provenance_bound",
    "test_outcome_release_requires_qualified_selector",
    "physical_separation_verified",
})

# Deliberately private acknowledgement for outcome-bearing legacy audit tests.
# The supported activation API never possesses or forwards this object.
_LEGACY_DEVELOPMENT_AUDIT_ACKNOWLEDGEMENT = object()

AUTHORITY_SCHEMAS = {
    "inclusion_manifest": "r3x_inclusion_manifest_v1",
    "context_source_crosswalk": "r3_ordinary_context_crosswalk_v1",
    "endpoint_estimand_registry": "r3_endpoint_estimand_registry_v1",
    "source_component_split_manifest": "r3_source_component_split_manifest_v1",
    "metric_zero_policy_registry": "r3x_metric_zero_policy_registry_v1",
    "model_optimizer_seed_config": "r3x_model_optimizer_seed_config_v1",
    "analysis_code_tree_manifest": "r3x_analysis_code_tree_manifest_v1",
    "method_protocol": "r3_ordinary_experiment_protocol_v1",
    "paired_strongest_baseline_comparison_policy": (
        "r3x_paired_strongest_baseline_comparison_policy_v1"
    ),
}

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_PROTECTED_VALUE_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:sealed|restricted)(?:$|[^a-z0-9])|gse246381",
    flags=re.IGNORECASE,
)


class ProtocolHoldError(RuntimeError):
    """Structured non-GO result raised before a prohibited or unbound action."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {self.message}")


def canonical_json(value: Any) -> str:
    """Return the single canonical JSON representation used for payload IDs."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolHoldError(
            "HOLD_INVALID_JSON",
            "value is not canonical finite JSON",
            details={"error": str(exc)},
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except (OSError, IOError) as exc:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY",
            f"authority bytes are unavailable: {path}",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    return digest.hexdigest()


def read_exact_bytes(path: Path, expected_sha256: Any, *, purpose: str) -> bytes:
    """Read one immutable byte snapshot and verify its pre-bound digest.

    Hashing and parsing separate opens creates a time-of-check/time-of-use gap.
    Every authority consumer therefore receives the exact bytes that were
    hashed in this function.
    """

    path = Path(path)
    assert_ordinary_path(path, purpose=purpose)
    expected = _require_sha256(expected_sha256, label=purpose)
    descriptor = None
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError("authority must be a regular single-link file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            raw = handle.read()
    except (OSError, IOError) as exc:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY",
            f"authority bytes are unavailable: {path}",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ProtocolHoldError(
            "HOLD_HASH_MISMATCH",
            f"{purpose} SHA-256 mismatch",
            details={"path": str(path), "expected": expected, "actual": actual},
        )
    return raw


def _json_from_exact_bytes(raw: bytes, *, path: Path, purpose: str) -> dict:
    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY",
            f"{purpose} is not readable JSON",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", f"{purpose} must be a JSON object"
        )
    return payload


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if not _HEX64_RE.fullmatch(text):
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY",
            f"{label} expected SHA-256 must be 64 lowercase hex",
            details={"value": str(value)},
        )
    return text


def _is_protected_path_component(value: object) -> bool:
    text = str(value or "").strip().lower()
    if "gse246381" in text:
        return True
    return text.startswith("sealed") or text.startswith("restricted")


def assert_ordinary_path(path: Path, purpose: str = "input") -> None:
    """Reject raw or resolved paths that advertise a protected namespace."""
    path = Path(path)
    candidates = [path]
    try:
        candidates.append(path.resolve(strict=False))
    except OSError:
        pass
    bad = sorted(
        {
            part
            for candidate in candidates
            for part in candidate.parts
            if _is_protected_path_component(part)
        }
    )
    if bad:
        raise ProtocolHoldError(
            "HOLD_PROTECTED_INPUT",
            f"ordinary-only {purpose} refuses protected path component(s): "
            + ", ".join(bad),
            details={"path": str(path), "components": bad},
        )


def _walk_scalars(value: object, path: str = "$") -> Iterable[tuple[str, object]]:
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            child = value[key]
            child_path = f"{path}.{key}"
            # Truthy protected keys must not hide behind booleans.  Explicit
            # false/None flags such as is_sealed=false remain admissible.
            if child not in (False, None, ""):
                yield child_path + "::<field_name>", key
            yield from _walk_scalars(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_scalars(child, f"{path}[{index}]")
    else:
        yield path, value


def _contains_protected(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float)):
        return False
    return bool(_PROTECTED_VALUE_RE.search(str(value).strip()))


def assert_ordinary_record(record: Mapping[str, Any], purpose: str = "input") -> None:
    """Reject protected identities/roles anywhere in a nested ordinary record."""
    if not isinstance(record, Mapping):
        raise ProtocolHoldError(
            "HOLD_PROTECTED_INPUT", f"ordinary-only {purpose} requires an object"
        )
    if record.get("is_sealed") is True or record.get("is_restricted") is True:
        raise ProtocolHoldError(
            "HOLD_PROTECTED_INPUT",
            f"ordinary-only {purpose} refuses a protected boolean flag",
        )
    for field_path, value in _walk_scalars(record):
        if _contains_protected(value):
            raise ProtocolHoldError(
                "HOLD_PROTECTED_INPUT",
                f"ordinary-only {purpose} refuses protected content at {field_path}",
                details={"field_path": field_path, "value": str(value)},
            )


def _load_exact_json(
    path: Path,
    expected_sha256: Any,
    *,
    purpose: str,
    scan_protected_contents: bool = True,
) -> dict:
    path = Path(path)
    raw = read_exact_bytes(path, expected_sha256, purpose=purpose)
    payload = _json_from_exact_bytes(raw, path=path, purpose=purpose)
    if scan_protected_contents:
        assert_ordinary_record(payload, purpose=purpose)
    return payload


def require_staged_disclosure_before_outcome_access(
    activation_manifest_path: Path,
    activation_manifest_sha256: str,
    *,
    prospective_output_dir: Optional[Path] = None,
) -> NoReturn:
    """Refuse monolithic outcome access until staged protocol v2 exists.

    Only the activation manifest itself is read and hash-verified here.  The
    function intentionally does not call :func:`validate_activation_manifest`,
    open any ready-dataset authority, create an output directory, or accept a
    metadata-only assertion that staged disclosure is complete.  Once a real
    v2 validator and release state machine exist, this unconditional HOLD must
    be replaced by their independent exact-byte replay; it must never be
    relaxed to trusting the capability booleans inspected below.
    """

    if prospective_output_dir is not None:
        output = Path(prospective_output_dir)
        assert_ordinary_path(output, purpose="prospective staged R3X output")
        if output.exists():
            raise ProtocolHoldError(
                "HOLD_OUTPUT_EXISTS_BEFORE_STAGED_DISCLOSURE_PREFLIGHT",
                "prospective output must not exist before staged disclosure",
                details={"path": str(output)},
            )
    manifest = _load_exact_json(
        Path(activation_manifest_path),
        activation_manifest_sha256,
        purpose="pre-disclosure activation capability probe",
        scan_protected_contents=False,
    )
    staged = manifest.get("staged_disclosure")
    capability_keys = set(staged) if isinstance(staged, Mapping) else set()
    missing = sorted(REQUIRED_STAGED_DISCLOSURE_CAPABILITIES - capability_keys)
    invalid = sorted(
        key
        for key in REQUIRED_STAGED_DISCLOSURE_CAPABILITIES & capability_keys
        if not isinstance(staged[key], Mapping)
        or staged[key].get("status") != "VERIFIED"
        or not isinstance(staged[key].get("evidence_ref"), Mapping)
    ) if isinstance(staged, Mapping) else []
    unverified = sorted(
        REQUIRED_STAGED_DISCLOSURE_CAPABILITIES - set(missing) - set(invalid)
    )
    raise ProtocolHoldError(
        STAGED_DISCLOSURE_HOLD_CODE,
        "current protocol validator opens a monolithic outcome-bearing "
        "ready_dataset and is not authorized for real confirmatory execution",
        details={
            "activation_schema": manifest.get("schema_version"),
            "required_schema": STAGED_DISCLOSURE_SCHEMA,
            "missing_capabilities": missing,
            "invalid_capabilities": invalid,
            "unverified_capabilities": unverified,
            "ordinary_internal_test_feature_access": False,
            "ordinary_internal_test_outcome_access": False,
            "ordinary_internal_test_predictions_generated": False,
            "sealed_access": False,
            "downstream_unlock": False,
        },
    )


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise ProtocolHoldError(
            "HOLD_ROUTE_AUTHORIZATION",
            f"route authorization {label} must be {expected!r}, got {value!r}",
            details={"field": label, "expected": expected, "actual": value},
        )


def load_route_authorization(
    path: Path,
    expected_sha256: str,
    *,
    require_training_ready: bool = False,
) -> dict:
    """Validate the exact user authorization without expanding its scope."""
    # The authorization document itself must describe the prohibited sealed
    # boundary, so scanning every prose string for the word "sealed" would
    # reject the very policy that keeps it disabled.  Its exact schema and
    # every protected capability are instead validated field-by-field below.
    payload = _load_exact_json(
        path,
        expected_sha256,
        purpose="route authorization input",
        scan_protected_contents=False,
    )
    return _validate_route_authorization_payload(
        payload, require_training_ready=require_training_ready
    )


def _validate_route_authorization_payload(
    payload: Mapping[str, Any], *, require_training_ready: bool
) -> dict:
    """Validate a route payload already parsed from an exact byte snapshot."""

    _require_exact(
        payload.get("schema_version"), ROUTE_AUTHORIZATION_SCHEMA, label="schema_version"
    )
    _require_exact(payload.get("selected_route"), AUTHORIZED_ROUTE, label="route")
    _require_exact(
        tuple(payload.get("authorized_sequence") or ()),
        AUTHORIZED_SEQUENCE,
        label="authorized_sequence",
    )
    _require_exact(
        payload.get("code_implementation_authorized"),
        True,
        label="implementation authorization",
    )
    _require_exact(
        payload.get("ordinary_training_authorized_when_all_inputs_frozen"),
        True,
        label="conditional ordinary training authorization",
    )

    status = payload.get("status")
    if status not in {
        "USER_AUTHORIZED_INPUTS_NOT_YET_QUALIFIED",
        "USER_AUTHORIZED_INPUTS_FROZEN",
    }:
        raise ProtocolHoldError(
            "HOLD_ROUTE_AUTHORIZATION",
            f"route authorization status is not recognized: {status!r}",
        )

    boundaries = payload.get("hard_boundaries")
    if not isinstance(boundaries, Mapping):
        raise ProtocolHoldError(
            "HOLD_ROUTE_AUTHORIZATION", "route authorization lacks hard_boundaries"
        )
    expected_boundaries = {
        "ordinary_only": True,
        "sealed_member_access": False,
        "formal_sealed_evaluator": False,
        "sign_accuracy_gate": SIGN_ACCURACY_GATE,
        "sign_gate_change_authorized": False,
        "post_hoc_metric_threshold_seed_or_endpoint_selection": False,
        "legacy_signed_ratio_enrichment": "PROHIBITED",
        "o0_g1_unlock": False,
        "authority_contract_overwrite": False,
    }
    for field, expected in expected_boundaries.items():
        actual = boundaries.get(field)
        if field == "sign_accuracy_gate":
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                actual = None
            elif not math.isclose(float(actual), SIGN_ACCURACY_GATE, abs_tol=0.0):
                actual = float(actual)
        if actual != expected:
            raise ProtocolHoldError(
                "HOLD_ROUTE_AUTHORIZATION",
                f"hard boundary {field} must remain {expected!r}, got {actual!r}",
                details={"field": field, "expected": expected, "actual": actual},
            )

    c3 = payload.get("c3_redline")
    if not isinstance(c3, Mapping) or c3.get("active_contract_mutation_authorized") is not False:
        raise ProtocolHoldError(
            "HOLD_ROUTE_AUTHORIZATION",
            "C3 authorization must remain draft-only",
        )

    if require_training_ready:
        unclosed = payload.get("required_unclosed_inputs")
        missing = list(unclosed) if isinstance(unclosed, list) else ["INVALID_UNCLOSED_LIST"]
        ready = payload.get("currently_ready_for_real_training") is True
        if status != "USER_AUTHORIZED_INPUTS_FROZEN" or not ready or missing:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY",
                "route is authorized but real ordinary training inputs are not all frozen",
                details={"missing": missing, "status": status, "ready": ready},
            )
    return dict(payload)


def _resolve_reference_path(manifest_path: Path, value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", "activation authority reference has no path"
        )
    path = Path(text)
    return path if path.is_absolute() else manifest_path.parent / path


def _verify_frozen_json_authority(name: str, raw: bytes, path: Path) -> dict | None:
    """Verify authority-specific schema from the same bytes that were hashed."""
    if name == "ready_dataset":
        # Dataset bytes may be JSONL; labels are intentionally not parsed here.
        return None
    payload = _json_from_exact_bytes(raw, path=path, purpose=f"{name} authority")
    expected_schema = AUTHORITY_SCHEMAS.get(name)
    if expected_schema is None or payload.get("schema_version") != expected_schema:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY",
            f"{name} has an unsupported authority schema",
            details={
                "path": str(path),
                "expected_schema": expected_schema,
                "actual_schema": payload.get("schema_version"),
            },
        )
    if payload.get("status") != "FROZEN":
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY",
            f"{name} must declare status=FROZEN",
            details={"path": str(path), "status": payload.get("status") if isinstance(payload, dict) else None},
        )
    assert_ordinary_record(payload, purpose=f"{name} authority")
    _validate_authority_semantics(name, payload)
    return payload


def _require_nonempty_list(payload: Mapping[str, Any], field: str, *, name: str) -> list:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", f"{name} requires a non-empty {field} list"
        )
    return value


def _validate_authority_semantics(name: str, payload: Mapping[str, Any]) -> None:
    if name == "inclusion_manifest":
        count = payload.get("n_ready_rows")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "inclusion manifest requires n_ready_rows >= 1"
            )
        for field in (
            "ready_dataset_sha256", "ready_row_ids_sha256",
            "context_source_crosswalk_sha256", "endpoint_estimand_registry_sha256",
            "ordinary_membership_registry_sha256",
        ):
            _require_sha256(payload.get(field), label=f"inclusion manifest {field}")
        if payload.get("protected_member_count") != 0 or payload.get("ordinary_only") is not True:
            raise ProtocolHoldError(
                "HOLD_PROTECTED_INPUT", "inclusion manifest must prove ordinary-only zero protected members"
            )
        rows = _require_nonempty_list(payload, "rows", name=name)
        if len(rows) != count:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "inclusion row registry count is inconsistent"
            )
        seen = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ProtocolHoldError(
                    "HOLD_INPUT_AUTHORITY", "inclusion rows must be objects"
                )
            row_id = str(row.get("row_id") or "")
            if not _HEX64_RE.fullmatch(row_id) or row_id in seen:
                raise ProtocolHoldError(
                    "HOLD_INPUT_AUTHORITY", "inclusion row IDs must be unique SHA-256 values"
                )
            seen.add(row_id)
            _require_sha256(row.get("row_content_sha256"), label="inclusion row content")
            if row.get("role") != "ORDINARY_PRIMARY_READY":
                raise ProtocolHoldError(
                    "HOLD_PROTECTED_INPUT", "inclusion row role is not ordinary primary-ready"
                )
        computed_row_ids = hashlib.sha256(
            json.dumps(sorted(seen), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if computed_row_ids != payload.get("ready_row_ids_sha256"):
            raise ProtocolHoldError(
                "HOLD_AUTHORITY_CROSS_BINDING", "inclusion ready row ID digest is stale"
            )
    elif name == "context_source_crosswalk":
        entries = _require_nonempty_list(payload, "entries", name=name)
        required = {
            "study", "endpoint", "assay_arm", "library", "source_asset",
            "region", "source_anchor_rule", "publication_context_id",
            "context_evidence_id", "source_relation_evidence_id",
            "allowed_row_relation_evidence",
        }
        for entry in entries:
            if not isinstance(entry, Mapping) or any(not str(entry.get(field) or "").strip() for field in required):
                raise ProtocolHoldError(
                    "HOLD_INPUT_AUTHORITY", "crosswalk entry is semantically incomplete"
                )
            context = str(entry["publication_context_id"]).lower().replace("-", "_")
            study = str(entry["study"]).lower().replace("-", "_")
            if context == f"ctx_{study}":
                raise ProtocolHoldError(
                    "HOLD_INPUT_AUTHORITY", "study sentinel is not a publication context"
                )
    elif name == "endpoint_estimand_registry":
        entries = _require_nonempty_list(payload, "entries", name=name)
        required = {
            "study", "endpoint", "estimand_id", "raw_source_field",
            "raw_candidate_field", "raw_unit", "direction", "transform",
            "valid_domain", "role", "replicate_policy", "source_evidence_sha256",
        }
        for entry in entries:
            if not isinstance(entry, Mapping) or any(not str(entry.get(field) or "").strip() for field in required):
                raise ProtocolHoldError(
                    "HOLD_INPUT_AUTHORITY", "endpoint estimand entry is semantically incomplete"
                )
            _require_sha256(
                entry.get("source_evidence_sha256"), label="estimand source evidence"
            )
    elif name == "source_component_split_manifest":
        if payload.get("split_unit") != "GLOBAL_SOURCE_CANDIDATE_ALIAS_TRANSITIVE_COMPONENT":
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "split manifest uses the wrong split unit"
            )
        if not isinstance(payload.get("row_assignments"), Mapping) or not payload["row_assignments"]:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "split manifest requires row assignments"
            )
        for field in (
            "manifest_sha256",
            "seed",
            "assignment_algorithm_id",
            "assignment_algorithm_parameters",
            "partition_component_counts",
            "partition_row_counts",
            "cell_row_counts",
            "cell_component_counts",
            "publication_claim",
            "support_policy",
            "support_check",
        ):
            if field not in payload:
                raise ProtocolHoldError(
                    "HOLD_UNREPLAYABLE_SPLIT_SCHEMA",
                    f"split manifest lacks deterministic replay field {field}",
                )
    elif name == "metric_zero_policy_registry":
        required_exact = {
            "zero_policy": "EXCLUDE_TRUE_ZERO",
            "primary_cell_definition": "STUDY_ESTIMAND_PUBLICATION_CONTEXT_EDIT_COUNT",
            "component_weighting": "EQUAL_COMPONENT_THEN_EQUAL_ROW_WITHIN_COMPONENT",
            "ranking_aggregation": "WITHIN_SOURCE_POOL_THEN_CELL_MACRO",
            "undefined_primary_cell_policy": "HOLD_PRIMARY_CELL_UNDEFINED",
            "seed_policy": "FIXED_FIVE_MODEL_PREDICTION_ENSEMBLE",
            "selection_tie_policy": "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK",
        }
        for field, expected in required_exact.items():
            if payload.get(field) != expected:
                raise ProtocolHoldError(
                    "HOLD_INPUT_AUTHORITY", f"metric registry {field} must be {expected}"
                )
    elif name == "model_optimizer_seed_config":
        if payload.get("seeds") != [11, 23, 42, 67, 101]:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "run config must freeze the five authorized seeds"
            )
        for field in ("optimizer", "loss", "checkpoint_selection", "model"):
            if not isinstance(payload.get(field), Mapping) or not payload[field]:
                raise ProtocolHoldError(
                    "HOLD_INPUT_AUTHORITY", f"run config requires a non-empty {field} object"
                )
        if payload.get("seed_aggregation") != "FIXED_PREDICTION_ENSEMBLE":
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "run config must freeze prediction-level ensemble aggregation"
            )
        if payload.get("estimand_training_scope") != "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND":
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "raw endpoint scales may not share one training loss"
            )
        dropout = payload.get("condition_unknown_dropout")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 < float(dropout) < 1.0:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "run config requires training-only UNK dropout in (0,1)"
            )
        if payload.get("checkpoint_selection", {}).get("rule") != (
            "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH"
        ):
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "run config checkpoint rule is not the frozen rule"
            )
    elif name == "method_protocol":
        if payload.get("selected_route") != AUTHORIZED_ROUTE or payload.get("real_ordinary_training_ready") is not True:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "method protocol is not frozen ready for the authorized route"
            )
        if payload.get("negative_controls") != REQUIRED_NEGATIVE_CONTROLS:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY",
                "method protocol must bind the exact ordered preregistered negative-control family",
            )
        boundaries = payload.get("hard_boundaries")
        if (
            not isinstance(boundaries, Mapping)
            or boundaries.get("ordinary_only") is not True
            or boundaries.get("sealed_member_access") is not False
            or boundaries.get("O0_G1_E0_unlock") is not False
        ):
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY",
                "method protocol must preserve ordinary-only, sealed-locked, downstream-locked boundaries",
            )
    elif name == "paired_strongest_baseline_comparison_policy":
        for field in (
            "policy_id", "primary_candidate_model_id", "strongest_baseline_id",
            "information_set", "comparison_family_id",
        ):
            if not str(payload.get(field) or "").strip():
                raise ProtocolHoldError(
                    "HOLD_INPUT_AUTHORITY", f"paired comparison policy lacks {field}"
                )
        required_methods = _require_nonempty_list(
            payload, "required_baseline_method_ids", name=name
        )
        minimum_suite = {
            "ZERO_DELTA", "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
            "EDIT_ONLY_RIDGE", "CANDIDATE_KMER_RIDGE",
            "PAIRED_LOCAL_GLOBAL_KMER_RIDGE", "PAIRED_FEATURE_NEAREST_NEIGHBOR",
            "FULL_PAIR_UNRESTRICTED_SIAMESE",
        }
        if not minimum_suite.issubset(set(required_methods)):
            raise ProtocolHoldError(
                "HOLD_BASELINE_SUITE_NOT_QUALIFIED",
                "paired policy omits required publication baseline families",
            )
        if payload.get("strongest_baseline_id") == "R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION":
            raise ProtocolHoldError(
                "HOLD_BASELINE_SUITE_NOT_QUALIFIED",
                "R2-style adaptation lacks a frozen architecture crosswalk and semantic study-proxy audit",
            )
        members = _require_nonempty_list(
            payload, "comparison_family_member_ids", name=name
        )
        if (
            any(not isinstance(member, str) or not member for member in members)
            or len(members) != len(set(members))
        ):
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY",
                "paired comparison family member IDs must be unique non-empty strings",
            )
        primary_member = payload.get("primary_comparison_member_id")
        if primary_member not in members:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY",
                "paired primary comparison member is absent from the frozen family",
            )
        inference = payload.get("primary_sign_inference")
        if not isinstance(inference, Mapping) or set(inference) != {
            "metric_id", "alpha", "confidence", "bootstrap", "permutation"
        }:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY",
                "paired policy must freeze the complete primary sign inference schema",
            )
        if inference.get("metric_id") != "MACRO_COMPONENT_CELL_BALANCED_SIGN_ACCURACY":
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "paired primary utility must be the frozen sign metric"
            )
        alpha = inference.get("alpha")
        confidence = inference.get("confidence")
        if (
            isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not 0.0 < float(alpha) < 1.0
            or not math.isclose(float(confidence), 1.0 - float(alpha), abs_tol=1e-12)
        ):
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "paired alpha/confidence settings are invalid"
            )
        bootstrap = inference.get("bootstrap")
        permutation = inference.get("permutation")
        if not isinstance(bootstrap, Mapping) or set(bootstrap) != {
            "n_resamples", "seed", "min_components_per_cell", "max_attempt_multiplier"
        }:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "paired bootstrap policy is incomplete"
            )
        if not isinstance(permutation, Mapping) or set(permutation) != {
            "n_permutations", "seed", "exact_component_limit", "min_components_per_cell"
        }:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "paired permutation policy is incomplete"
            )
        if (
            isinstance(bootstrap.get("n_resamples"), bool)
            or not isinstance(bootstrap.get("n_resamples"), int)
            or bootstrap["n_resamples"] < 2000
            or isinstance(permutation.get("n_permutations"), bool)
            or not isinstance(permutation.get("n_permutations"), int)
            or permutation["n_permutations"] < 10000
            or bootstrap.get("min_components_per_cell") != 10
            or permutation.get("min_components_per_cell") != 10
            or permutation.get("exact_component_limit") != 20
        ):
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY",
                "paired publication inference requires >=2000 bootstrap, >=10000 permutation, "
                "exact cap 20, and >=10 components per cell",
            )


def _validate_code_tree(payload: Mapping[str, Any], manifest_path: Path) -> dict:
    root_text = str(payload.get("root_path") or "").strip()
    if not root_text:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", "analysis code tree lacks root_path"
        )
    root = Path(root_text)
    if not root.is_absolute():
        root = manifest_path.parent / root
    assert_ordinary_path(root, purpose="analysis code tree root")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", "analysis code tree requires a non-empty files list"
        )
    normalized = []
    seen = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "analysis code tree entries must be objects"
            )
        relative = Path(str(entry.get("path") or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "analysis code tree paths must be safe relative paths"
            )
        key = relative.as_posix()
        if key in seen:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", "analysis code tree paths must be unique"
            )
        seen.add(key)
        expected = _require_sha256(entry.get("sha256"), label=f"analysis code {key}")
        read_exact_bytes(root / relative, expected, purpose=f"analysis code {key}")
        normalized.append({"path": key, "sha256": expected})
    normalized.sort(key=lambda item: item["path"])
    declared_tree = _require_sha256(
        payload.get("tree_sha256"), label="analysis code tree"
    )
    actual_tree = canonical_sha256(normalized)
    if declared_tree != actual_tree:
        raise ProtocolHoldError(
            "HOLD_HASH_MISMATCH",
            "analysis code tree semantic digest mismatch",
            details={"expected": declared_tree, "actual": actual_tree},
        )
    return {"root_path": str(root), "tree_sha256": actual_tree, "n_files": len(normalized)}


def _ready_rows_from_exact_bytes(raw: bytes, *, path: Path) -> list[dict]:
    """Parse only the structural ready-row authority needed by activation."""

    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    rows: list[dict] = []
    seen: set[str] = set()
    try:
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=unique_object)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not an object")
            assert_ordinary_record(row, purpose=f"ready dataset line {line_number}")
            row_id = str(row.get("r3_input_row_sha256") or "")
            if not _HEX64_RE.fullmatch(row_id) or row_id in seen:
                raise ValueError(
                    f"line {line_number} has a missing, invalid, or duplicate r3_input_row_sha256"
                )
            seen.add(row_id)
            rows.append(row)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY",
            "ready dataset is not valid structural JSONL",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not rows:
        raise ProtocolHoldError("HOLD_INPUT_AUTHORITY", "ready dataset contains no rows")
    return rows


def _verified_ready_study_maps(
    rows: list[dict],
    *,
    inclusion: Mapping[str, Any],
    crosswalk: Mapping[str, Any],
    estimand_registry: Mapping[str, Any],
    split: Mapping[str, Any],
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[tuple[str, str]]],
    dict,
]:
    """Cross-check ready bytes and derive scope maps from verified authority content."""

    inclusion_rows = inclusion.get("rows")
    if not isinstance(inclusion_rows, list):
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", "inclusion manifest lacks row-level entries"
        )
    inclusion_index = {
        str(entry.get("row_id") or ""): entry
        for entry in inclusion_rows
        if isinstance(entry, Mapping)
    }
    if len(inclusion_index) != len(inclusion_rows) or len(inclusion_index) != len(rows):
        raise ProtocolHoldError(
            "HOLD_AUTHORITY_CROSS_BINDING",
            "ready rows and inclusion entries are not one-to-one",
        )
    ready_ids = {str(row["r3_input_row_sha256"]) for row in rows}
    if set(inclusion_index) != ready_ids:
        raise ProtocolHoldError(
            "HOLD_AUTHORITY_CROSS_BINDING",
            "ready row identities differ from the inclusion authority",
        )
    for row in rows:
        row_id = str(row["r3_input_row_sha256"])
        if inclusion_index[row_id].get("row_content_sha256") != canonical_sha256(row):
            raise ProtocolHoldError(
                "HOLD_AUTHORITY_CROSS_BINDING",
                "ready row content does not match its inclusion digest",
                details={"row_id": row_id},
            )

    estimand_entries = estimand_registry.get("entries")
    if not isinstance(estimand_entries, list):
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", "endpoint estimand registry lacks entries"
        )
    estimand_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in estimand_entries:
        if not isinstance(entry, Mapping) or entry.get("role") != "PRIMARY":
            continue
        key = (str(entry.get("study") or ""), str(entry.get("endpoint") or ""))
        if "" in key or key in estimand_index:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY",
                "primary endpoint estimand keys must be complete and unique",
            )
        estimand_index[key] = entry

    crosswalk_entries = crosswalk.get("entries")
    if not isinstance(crosswalk_entries, list):
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", "context/source crosswalk lacks entries"
        )
    crosswalk_cells = {
        (
            str(entry.get("study") or ""),
            str(entry.get("endpoint") or ""),
            str(entry.get("publication_context_id") or ""),
        )
        for entry in crosswalk_entries
        if isinstance(entry, Mapping)
    }

    endpoint_map: dict[str, set[str]] = {}
    estimand_map: dict[str, set[str]] = {}
    endpoint_estimand_map: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        study = str(row.get("study") or "").strip()
        endpoint = str(row.get("endpoint") or row.get("endpoint_id") or "").strip()
        context = str(
            row.get("publication_context_id") or row.get("context_id_v2") or ""
        ).strip()
        estimand_id = str(row.get("estimand_id") or "").strip()
        if not all((study, endpoint, context, estimand_id)):
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY",
                "ready row lacks study, endpoint, context, or estimand identity",
                details={"row_id": row["r3_input_row_sha256"]},
            )
        registry_entry = estimand_index.get((study, endpoint))
        if registry_entry is None or registry_entry.get("estimand_id") != estimand_id:
            raise ProtocolHoldError(
                "HOLD_AUTHORITY_CROSS_BINDING",
                "ready row is not bound to its frozen primary endpoint estimand",
                details={"row_id": row["r3_input_row_sha256"]},
            )
        if (study, endpoint, context) not in crosswalk_cells:
            raise ProtocolHoldError(
                "HOLD_AUTHORITY_CROSS_BINDING",
                "ready row is absent from the frozen context/source crosswalk",
                details={"row_id": row["r3_input_row_sha256"]},
            )
        endpoint_map.setdefault(study, set()).add(endpoint)
        estimand_map.setdefault(study, set()).add(estimand_id)
        endpoint_estimand_map.setdefault(study, set()).add((endpoint, estimand_id))

    try:
        # Local import avoids a module-import cycle: splits uses canonical_sha256.
        from .splits import SplitHoldError, verify_split_manifest_against_rows

        split_replay = verify_split_manifest_against_rows(rows, split)
    except SplitHoldError as exc:
        raise ProtocolHoldError(exc.code, str(exc), details=exc.details) from exc
    return endpoint_map, estimand_map, endpoint_estimand_map, split_replay


def _normalized_study_map(value: Any, *, label: str) -> dict[str, set[str]]:
    if not isinstance(value, Mapping) or not value:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", f"activation requires a non-empty {label}"
        )
    normalized: dict[str, set[str]] = {}
    for study, raw_values in value.items():
        study_id = str(study or "").strip()
        if not study_id or not isinstance(raw_values, list) or not raw_values:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", f"{label} contains an empty study/value set"
            )
        values = {str(item or "").strip() for item in raw_values}
        if "" in values:
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY", f"{label} contains an empty value"
            )
        normalized[study_id] = values
    return normalized


def _activation_scientific_scope(
    payload: Mapping[str, Any],
    *,
    verified_endpoint_map: Mapping[str, set[str]],
    verified_estimand_map: Mapping[str, set[str]],
    verified_endpoint_estimand_map: Mapping[str, set[tuple[str, str]]],
) -> str:
    asserted_endpoint_map = _normalized_study_map(
        payload.get("study_endpoint_map"), label="study_endpoint_map"
    )
    endpoint_map = {
        str(study): set(values) for study, values in verified_endpoint_map.items()
    }
    if asserted_endpoint_map != endpoint_map:
        raise ProtocolHoldError(
            "HOLD_ACTIVATION_SCOPE_MISMATCH",
            "activation study_endpoint_map disagrees with verified ready-row authorities",
            details={
                "asserted": {key: sorted(values) for key, values in asserted_endpoint_map.items()},
                "verified": {key: sorted(values) for key, values in endpoint_map.items()},
            },
        )
    asserted_estimands = payload.get("study_estimand_map")
    if asserted_estimands is not None:
        asserted_estimand_map = _normalized_study_map(
            asserted_estimands, label="study_estimand_map"
        )
        if asserted_estimand_map != verified_estimand_map:
            raise ProtocolHoldError(
                "HOLD_ACTIVATION_SCOPE_MISMATCH",
                "activation study_estimand_map disagrees with verified registry bindings",
            )
    if len(endpoint_map) < 2:
        if payload.get("claim_scope") == "PUBLICATION_ELIGIBLE_SAME_ESTIMAND":
            raise ProtocolHoldError(
                "HOLD_SINGLE_STUDY_PUBLICATION_SCOPE",
                "a single ready study cannot activate publication-eligible scope",
            )
        return "DEVELOPMENT_ONLY_SINGLE_STUDY"
    shared_endpoint = set.intersection(*endpoint_map.values())
    if not shared_endpoint:
        if payload.get("claim_scope") == "PUBLICATION_ELIGIBLE_SAME_ESTIMAND":
            raise ProtocolHoldError(
                "HOLD_TWO_STUDY_NESTED_ENDPOINTS" if len(endpoint_map) == 2
                else "HOLD_ESTIMAND_NOT_REPLICATED_ACROSS_STUDIES",
                "publication scope cannot be identified without a shared estimand/endpoint",
            )
        if len(endpoint_map) == 2:
            return "DEVELOPMENT_ONLY_TWO_STUDY_NESTED_ENDPOINTS"
        return "DEVELOPMENT_ONLY_ENDPOINT_STUDY_CONFOUNDED"
    if payload.get("claim_scope") != "PUBLICATION_ELIGIBLE_SAME_ESTIMAND":
        return "DEVELOPMENT_ONLY_ACTIVATION_SCOPE"
    estimand_map = {str(study): set(values) for study, values in verified_estimand_map.items()}
    if asserted_estimands is None:
        raise ProtocolHoldError(
            "HOLD_ESTIMAND_NOT_REPLICATED_ACROSS_STUDIES",
            "publication activation must explicitly bind the verified study_estimand_map",
        )
    if set(estimand_map) != set(endpoint_map) or not set.intersection(
        *estimand_map.values()
    ):
        raise ProtocolHoldError(
            "HOLD_ESTIMAND_NOT_REPLICATED_ACROSS_STUDIES",
            "publication scope requires one frozen estimand replicated across studies",
        )
    if (
        set(verified_endpoint_estimand_map) != set(endpoint_map)
        or not set.intersection(*verified_endpoint_estimand_map.values())
    ):
        raise ProtocolHoldError(
            "HOLD_ESTIMAND_NOT_REPLICATED_ACROSS_STUDIES",
            "publication scope requires the same endpoint-estimand binding in every verified study",
        )
    return "PUBLICATION_ELIGIBLE_SAME_ESTIMAND"


def _require_binding(payload: Mapping[str, Any], field: str, expected: str, *, name: str) -> None:
    actual = str(payload.get(field) or "").strip().lower()
    if actual != expected:
        raise ProtocolHoldError(
            "HOLD_AUTHORITY_CROSS_BINDING",
            f"{name}.{field} does not bind the activated authority bytes",
            details={"field": field, "expected": expected, "actual": actual},
        )


def _validate_authority_cross_bindings(
    payloads: Mapping[str, Mapping[str, Any] | None],
    digests: Mapping[str, str],
) -> None:
    inclusion = payloads["inclusion_manifest"]
    split = payloads["source_component_split_manifest"]
    run_config = payloads["model_optimizer_seed_config"]
    paired_policy = payloads["paired_strongest_baseline_comparison_policy"]
    assert isinstance(inclusion, Mapping)
    assert isinstance(split, Mapping)
    assert isinstance(run_config, Mapping)
    assert isinstance(paired_policy, Mapping)

    for field, authority in (
        ("ready_dataset_sha256", "ready_dataset"),
        ("context_source_crosswalk_sha256", "context_source_crosswalk"),
        ("endpoint_estimand_registry_sha256", "endpoint_estimand_registry"),
    ):
        _require_binding(inclusion, field, digests[authority], name="inclusion_manifest")
    for field, authority in (
        ("ready_dataset_sha256", "ready_dataset"),
        ("inclusion_manifest_sha256", "inclusion_manifest"),
    ):
        _require_binding(split, field, digests[authority], name="source_component_split_manifest")

    authority_bindings = run_config.get("authority_sha256")
    required_run_bindings = {
        name: digest
        for name, digest in digests.items()
        if name not in {"model_optimizer_seed_config"}
    }
    if not isinstance(authority_bindings, Mapping) or dict(authority_bindings) != required_run_bindings:
        raise ProtocolHoldError(
            "HOLD_AUTHORITY_CROSS_BINDING",
            "model/optimizer config must exactly bind every other activated authority",
        )

    if paired_policy.get("baseline_suite_status") != "PUBLICATION_QUALIFIED":
        raise ProtocolHoldError(
            "HOLD_BASELINE_SUITE_NOT_QUALIFIED",
            "real A execution requires a publication-qualified strong-baseline suite",
        )
    primary = paired_policy.get("primary_candidate_model_id")
    if primary not in {
        "A_EXACT_POTENTIAL_DIFFERENCE",
        "B_SPARSE_EDIT_SET_ADDITIVE",
    }:
        raise ProtocolHoldError(
            "HOLD_MODEL_FAMILY_SELECTION_POLICY",
            "paired policy must freeze one primary candidate model before test access",
        )
    for field, authority in (
        ("ready_dataset_sha256", "ready_dataset"),
        ("inclusion_manifest_sha256", "inclusion_manifest"),
        ("source_component_split_manifest_sha256", "source_component_split_manifest"),
        ("metric_zero_policy_registry_sha256", "metric_zero_policy_registry"),
    ):
        _require_binding(paired_policy, field, digests[authority], name="paired_policy")


def validate_activation_manifest(
    path: Path,
    expected_sha256: Optional[str],
    *,
    prospective_output_dir: Optional[Path] = None,
) -> NoReturn:
    """Public activation entrypoint; currently always stops before outcomes.

    The old v1 activation format binds a monolithic outcome-bearing ready file.
    Calling this public name therefore cannot activate training or publication
    scope.  Explicit legacy-development audits use the deliberately named
    private legacy-development audit helper instead.
    """

    require_staged_disclosure_before_outcome_access(
        path,
        expected_sha256,
        prospective_output_dir=prospective_output_dir,
    )
    raise ProtocolHoldError(
        "HOLD_STAGED_DISCLOSURE_STATE_MACHINE_NOT_IMPLEMENTED",
        "staged disclosure guard returned without a stage-specific validator",
    )


def _audit_legacy_outcome_bearing_activation_manifest(
    path: Path,
    expected_sha256: Optional[str],
    *,
    prospective_output_dir: Optional[Path] = None,
    _development_audit_acknowledgement: object = None,
) -> dict:
    """Audit a legacy v1 outcome-bearing graph for development diagnostics.

    This validator opens and parses the monolithic ``ready_dataset``, including
    internal-test outcomes.  It is retained for explicit development audits and
    protocol tests only; it is not an activation, blind pre-test capability, or
    publication-scope grant.  Real runners, materializers, and preflight
    entrypoints use the public fail-closed activation entrypoint instead.
    """
    if (
        _development_audit_acknowledgement
        is not _LEGACY_DEVELOPMENT_AUDIT_ACKNOWLEDGEMENT
    ):
        raise ProtocolHoldError(
            "HOLD_LEGACY_DEVELOPMENT_AUDIT_ACKNOWLEDGEMENT_REQUIRED",
            "private outcome-bearing legacy audit requires explicit test-only acknowledgement",
        )
    manifest_path = Path(path)
    if prospective_output_dir is not None:
        output = Path(prospective_output_dir)
        assert_ordinary_path(output, purpose="prospective output")
        if output.exists():
            raise ProtocolHoldError(
                "HOLD_OUTPUT_EXISTS",
                "prospective output must not exist before exclusive run creation",
                details={"path": str(output)},
            )
    payload = _load_exact_json(
        manifest_path,
        expected_sha256,
        purpose="activation manifest input",
    )
    if payload.get("schema_version") != ACTIVATION_MANIFEST_SCHEMA:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", "unsupported activation manifest schema"
        )
    if payload.get("status") != "FROZEN":
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", "activation manifest must declare status=FROZEN"
        )
    if payload.get("mode") != "ORDINARY_ONLY":
        raise ProtocolHoldError(
            "HOLD_PROTECTED_INPUT", "activation mode must be ORDINARY_ONLY"
        )
    authorities = payload.get("authorities")
    if not isinstance(authorities, Mapping):
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", "activation manifest lacks authorities"
        )
    missing = [name for name in REQUIRED_ACTIVATION_AUTHORITIES if name not in authorities]
    if missing:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY",
            "activation manifest is incomplete",
            details={"missing": missing},
        )
    unexpected = sorted(set(authorities) - set(REQUIRED_ACTIVATION_AUTHORITIES))
    if unexpected:
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY",
            "activation manifest contains unregistered authority entries",
            details={"unexpected": unexpected},
        )

    verified: Dict[str, dict] = {}
    authority_payloads: Dict[str, Any] = {}
    authority_digests: Dict[str, str] = {}
    for name in REQUIRED_ACTIVATION_AUTHORITIES:
        reference = authorities[name]
        if not isinstance(reference, Mapping) or reference.get("status") != "FROZEN":
            raise ProtocolHoldError(
                "HOLD_INPUT_AUTHORITY",
                f"activation reference {name} must declare status=FROZEN",
            )
        authority_path = _resolve_reference_path(manifest_path, reference.get("path"))
        assert_ordinary_path(authority_path, purpose=f"{name} authority")
        expected = _require_sha256(reference.get("sha256"), label=f"{name} authority")
        raw = read_exact_bytes(
            authority_path, expected, purpose=f"{name} authority"
        )
        actual = hashlib.sha256(raw).hexdigest()
        code_tree = None
        if name == "route_authorization":
            route_payload = _json_from_exact_bytes(
                raw, path=authority_path, purpose="route authorization authority"
            )
            _validate_route_authorization_payload(
                route_payload, require_training_ready=True
            )
            authority_payloads[name] = route_payload
        elif name == "ready_dataset":
            authority_payloads[name] = _ready_rows_from_exact_bytes(
                raw, path=authority_path
            )
        else:
            authority_payloads[name] = _verify_frozen_json_authority(
                name, raw, authority_path
            )
            if name == "analysis_code_tree_manifest":
                assert authority_payloads[name] is not None
                code_tree = _validate_code_tree(
                    authority_payloads[name], authority_path
                )
        verified[name] = {"path": str(authority_path), "sha256": actual}
        if code_tree is not None:
            verified[name]["semantic_tree"] = code_tree
        authority_digests[name] = actual

    _validate_authority_cross_bindings(authority_payloads, authority_digests)

    ready_rows = authority_payloads["ready_dataset"]
    inclusion = authority_payloads["inclusion_manifest"]
    crosswalk = authority_payloads["context_source_crosswalk"]
    estimand_registry = authority_payloads["endpoint_estimand_registry"]
    split = authority_payloads["source_component_split_manifest"]
    if (
        not isinstance(ready_rows, list)
        or not isinstance(inclusion, Mapping)
        or not isinstance(crosswalk, Mapping)
        or not isinstance(estimand_registry, Mapping)
        or not isinstance(split, Mapping)
    ):
        raise ProtocolHoldError(
            "HOLD_INPUT_AUTHORITY", "verified scientific authorities have invalid types"
        )
    (
        verified_endpoint_map,
        verified_estimand_map,
        verified_endpoint_estimand_map,
        split_replay,
    ) = (
        _verified_ready_study_maps(
            ready_rows,
            inclusion=inclusion,
            crosswalk=crosswalk,
            estimand_registry=estimand_registry,
            split=split,
        )
    )

    if payload.get("claim_scope") == "PUBLICATION_CROSS_STUDY":
        raise ProtocolHoldError(
            "HOLD_UNSUPPORTED_CLAIM_SCOPE",
            "publication activation must use the exact SAME_ESTIMAND scope enum",
        )
    scientific_scope = _activation_scientific_scope(
        payload,
        verified_endpoint_map=verified_endpoint_map,
        verified_estimand_map=verified_estimand_map,
        verified_endpoint_estimand_map=verified_endpoint_estimand_map,
    )
    return {
        "schema_version": ACTIVATION_MANIFEST_SCHEMA,
        "status": "LEGACY_OUTCOME_BEARING_DEVELOPMENT_AUDIT_COMPLETE",
        "activation_manifest_path": str(manifest_path),
        "activation_manifest_sha256": _require_sha256(
            expected_sha256, label="activation manifest input"
        ),
        "verified_authorities": verified,
        "verified_study_endpoint_map": {
            study: sorted(values) for study, values in sorted(verified_endpoint_map.items())
        },
        "verified_study_estimand_map": {
            study: sorted(values) for study, values in sorted(verified_estimand_map.items())
        },
        "verified_study_endpoint_estimand_map": {
            study: [
                {"endpoint": endpoint, "estimand_id": estimand_id}
                for endpoint, estimand_id in sorted(values)
            ]
            for study, values in sorted(verified_endpoint_estimand_map.items())
        },
        "verified_split_replay": split_replay,
        "scientific_scope": "DEVELOPMENT_ONLY_LEGACY_OUTCOME_BEARING_AUDIT",
        "legacy_claim_scope_diagnostic": scientific_scope,
        "sealed_access": False,
        "downstream_unlock": False,
    }
