"""Development-only neural *state* artifacts for staged R3X execution.

The legacy checkpoint schema binds an ``internal_test`` row-set hash.  This
schema intentionally cannot: it is created before either internal-test
authority is opened and binds only train/development rows and authorities.

This binary torch artifact is deliberately distinct from the canonical JSON
checkpoint *envelope* used by :mod:`scripts.r3x.predictor_provenance`.  The
envelope may reference this file as opaque state, but the two formats must
never share a schema identifier.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch

from .artifact_io import ArtifactIOError, atomic_publish_bytes, read_exact_regular_file
from .checkpoint import ALLOWED_TRAINING_VARIANTS, FROZEN_SEEDS
from .models import ModelConfig
from .protocol import ProtocolHoldError, assert_ordinary_path
from .staged_disclosure import REQUIRED_DEVELOPMENT_AUTHORITY_DIGEST_KEYS


SCHEMA = "r3x_development_neural_state_v2"
ARTIFACT_ROLE = "ORDINARY_DEVELOPMENT_NEURAL_STATE"
SCIENTIFIC_SCOPE = "DEVELOPMENT_ONLY_STAGED_PRETEST"
SIGN_GATE = 0.60
REQUIRED_DEVELOPMENT_DATA_AUTHORITY_HASHES = (
    *REQUIRED_DEVELOPMENT_AUTHORITY_DIGEST_KEYS,
)
REQUIRED_DEVELOPMENT_NONDATA_AUTHORITY_HASHES = (
    "route_authorization",
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
# Compatibility export for callers that need the complete development-stage
# binding list.  Data and non-data authorities remain separate in the payload.
REQUIRED_DEVELOPMENT_AUTHORITY_HASHES = (
    *REQUIRED_DEVELOPMENT_DATA_AUTHORITY_HASHES,
    *REQUIRED_DEVELOPMENT_NONDATA_AUTHORITY_HASHES,
)
PARTITION_KEYS = {"train", "development"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_METHOD = re.compile(r"^[A-Z][A-Z0-9_]*$")
_KEYS = {
    "schema_version",
    "artifact_role",
    "run_start_sha256",
    "run_plan_sha256",
    "development_activation_sha256",
    "development_data_authority_hashes",
    "development_nondata_authority_hashes",
    "estimand_binding_sha256",
    "method_id",
    "training_variant_id",
    "architecture_id",
    "model_config",
    "model_config_sha256",
    "condition_vocab",
    "condition_vocab_sha256",
    "training_config_sha256",
    "information_set",
    "information_set_fingerprint",
    "seed",
    "scientific_scope",
    "selected_epoch",
    "checkpoint_selection_sha256",
    "partition_row_set_sha256",
    "sign_gate",
    "sign_decision_rule",
    "ordinary_only",
    "internal_test_feature_access",
    "internal_test_outcome_access",
    "sealed_access",
    "downstream_unlock",
    "state_shapes",
    "state_dtypes",
    "state_dict",
}


class DevelopmentCheckpointError(RuntimeError):
    pass


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise DevelopmentCheckpointError(f"{label} must be 64 lowercase hex")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _authority_hashes(
    value: Mapping,
    *,
    required: tuple[str, ...],
    label: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise DevelopmentCheckpointError(
            f"{label} hashes are incomplete or unexpected"
        )
    return {
        name: _digest(value[name], label=name)
        for name in required
    }


def _partition_hashes(value: Mapping) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != PARTITION_KEYS:
        raise DevelopmentCheckpointError(
            "development checkpoint accepts only train/development row-set hashes"
        )
    return {
        partition: _digest(value[partition], label=f"{partition} row-set SHA-256")
        for partition in ("train", "development")
    }


def _condition_vocab(value: Mapping, cfg: ModelConfig) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or set(value) != {"endpoint", "context"}:
        raise DevelopmentCheckpointError(
            "condition vocab requires exact endpoint/context mappings"
        )
    normalized = {}
    for axis, capacity in (("endpoint", cfg.n_endpoints), ("context", cfg.n_contexts)):
        raw = value[axis]
        if not isinstance(raw, Mapping) or raw.get("__UNK__") != 0:
            raise DevelopmentCheckpointError(
                f"condition vocab {axis} must bind __UNK__ to zero"
            )
        mapping = {}
        for key, index in raw.items():
            if (
                not isinstance(key, str)
                or not key
                or key != key.strip()
                or isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
            ):
                raise DevelopmentCheckpointError(
                    f"condition vocab {axis} contains an invalid entry"
                )
            mapping[key] = int(index)
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise DevelopmentCheckpointError(
                f"condition vocab {axis} indices must be contiguous from zero"
            )
        if len(mapping) > capacity:
            raise DevelopmentCheckpointError(
                f"condition vocab {axis} exceeds model capacity"
            )
        normalized[axis] = dict(sorted(mapping.items()))
    return normalized


def build_development_checkpoint(
    model: torch.nn.Module,
    cfg: ModelConfig,
    *,
    run_start_sha256: str,
    run_plan_sha256: str,
    development_activation_sha256: str,
    development_data_authority_hashes: Mapping,
    development_nondata_authority_hashes: Mapping,
    estimand_binding_sha256: str,
    method_id: str,
    training_variant_id: str,
    training_config_sha256: str,
    condition_vocab: Mapping,
    information_set: str,
    information_set_fingerprint: str,
    seed: int,
    selected_epoch: int,
    checkpoint_selection_sha256: str,
    partition_row_set_sha256: Mapping,
) -> dict:
    architecture = getattr(model, "architecture_id", None)
    if not isinstance(architecture, str) or not architecture:
        raise DevelopmentCheckpointError("model lacks architecture_id")
    if not isinstance(method_id, str) or not _METHOD.fullmatch(method_id):
        raise DevelopmentCheckpointError("method_id is not a safe frozen identifier")
    if training_variant_id not in ALLOWED_TRAINING_VARIANTS:
        raise DevelopmentCheckpointError("training_variant_id is not frozen")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in FROZEN_SEEDS:
        raise DevelopmentCheckpointError("seed is not one of the five frozen seeds")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch < 1
    ):
        raise DevelopmentCheckpointError("selected_epoch must be positive")
    if not isinstance(information_set, str) or not information_set:
        raise DevelopmentCheckpointError("information_set is required")
    model_config = asdict(cfg)
    normalized_vocab = _condition_vocab(condition_vocab, cfg)
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    if not state:
        raise DevelopmentCheckpointError("model state is empty")
    return {
        "schema_version": SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "run_start_sha256": _digest(run_start_sha256, label="run_start_sha256"),
        "run_plan_sha256": _digest(run_plan_sha256, label="run_plan_sha256"),
        "development_activation_sha256": _digest(
            development_activation_sha256,
            label="development_activation_sha256",
        ),
        "development_data_authority_hashes": _authority_hashes(
            development_data_authority_hashes,
            required=REQUIRED_DEVELOPMENT_DATA_AUTHORITY_HASHES,
            label="development data authority",
        ),
        "development_nondata_authority_hashes": _authority_hashes(
            development_nondata_authority_hashes,
            required=REQUIRED_DEVELOPMENT_NONDATA_AUTHORITY_HASHES,
            label="development non-data authority",
        ),
        "estimand_binding_sha256": _digest(
            estimand_binding_sha256, label="estimand_binding_sha256"
        ),
        "method_id": method_id,
        "training_variant_id": training_variant_id,
        "architecture_id": architecture,
        "model_config": model_config,
        "model_config_sha256": _canonical_sha256(model_config),
        "condition_vocab": normalized_vocab,
        "condition_vocab_sha256": _canonical_sha256(normalized_vocab),
        "training_config_sha256": _digest(
            training_config_sha256, label="training_config_sha256"
        ),
        "information_set": information_set,
        "information_set_fingerprint": _digest(
            information_set_fingerprint, label="information_set_fingerprint"
        ),
        "seed": seed,
        "scientific_scope": SCIENTIFIC_SCOPE,
        "selected_epoch": selected_epoch,
        "checkpoint_selection_sha256": _digest(
            checkpoint_selection_sha256,
            label="checkpoint_selection_sha256",
        ),
        "partition_row_set_sha256": _partition_hashes(
            partition_row_set_sha256
        ),
        "sign_gate": SIGN_GATE,
        "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        "ordinary_only": True,
        "internal_test_feature_access": False,
        "internal_test_outcome_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
        "state_shapes": {key: list(value.shape) for key, value in state.items()},
        "state_dtypes": {key: str(value.dtype) for key, value in state.items()},
        "state_dict": state,
    }


def validate_development_checkpoint(
    payload: Mapping,
    *,
    expected_architecture_id: str,
    expected_development_data_authority_hashes: Mapping,
    expected_development_nondata_authority_hashes: Mapping,
    expected_run_start_sha256: str | None = None,
    expected_run_plan_sha256: str | None = None,
    expected_development_activation_sha256: str | None = None,
    expected_method_id: str | None = None,
    expected_training_variant_id: str | None = None,
    expected_seed: int | None = None,
    expected_estimand_binding_sha256: str | None = None,
    expected_information_set_fingerprint: str | None = None,
    expected_partition_row_set_sha256: Mapping | None = None,
) -> dict:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA:
        raise DevelopmentCheckpointError("unsupported development checkpoint schema")
    if set(payload) != _KEYS:
        raise DevelopmentCheckpointError(
            f"development checkpoint keys differ from schema: {sorted(set(payload) ^ _KEYS)}"
        )
    if payload.get("artifact_role") != ARTIFACT_ROLE:
        raise DevelopmentCheckpointError("development checkpoint role changed")
    if payload.get("scientific_scope") != SCIENTIFIC_SCOPE:
        raise DevelopmentCheckpointError("development checkpoint scope escalated")
    for field in (
        "ordinary_only",
        "internal_test_feature_access",
        "internal_test_outcome_access",
        "sealed_access",
        "downstream_unlock",
    ):
        expected = field == "ordinary_only"
        if payload.get(field) is not expected:
            raise DevelopmentCheckpointError(f"development boundary changed: {field}")
    if payload.get("sign_gate") != SIGN_GATE:
        raise DevelopmentCheckpointError("sign gate changed")
    if payload.get("sign_decision_rule") != "RAW_PREDICTED_DELTA_MEAN_GE_ZERO":
        raise DevelopmentCheckpointError("sign decision rule changed")
    if payload.get("architecture_id") != expected_architecture_id:
        raise DevelopmentCheckpointError("architecture mismatch")
    actual_data_authorities = _authority_hashes(
        payload.get("development_data_authority_hashes"),
        required=REQUIRED_DEVELOPMENT_DATA_AUTHORITY_HASHES,
        label="development data authority",
    )
    expected_data_authorities = _authority_hashes(
        expected_development_data_authority_hashes,
        required=REQUIRED_DEVELOPMENT_DATA_AUTHORITY_HASHES,
        label="expected development data authority",
    )
    if actual_data_authorities != expected_data_authorities:
        raise DevelopmentCheckpointError(
            "development data authority hashes mismatch"
        )
    actual_nondata_authorities = _authority_hashes(
        payload.get("development_nondata_authority_hashes"),
        required=REQUIRED_DEVELOPMENT_NONDATA_AUTHORITY_HASHES,
        label="development non-data authority",
    )
    expected_nondata_authorities = _authority_hashes(
        expected_development_nondata_authority_hashes,
        required=REQUIRED_DEVELOPMENT_NONDATA_AUTHORITY_HASHES,
        label="expected development non-data authority",
    )
    if actual_nondata_authorities != expected_nondata_authorities:
        raise DevelopmentCheckpointError(
            "development non-data authority hashes mismatch"
        )
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise DevelopmentCheckpointError("model_config must be an object")
    try:
        cfg = ModelConfig(**dict(model_config))
    except (TypeError, ValueError) as exc:
        raise DevelopmentCheckpointError(f"model_config is invalid: {exc}") from exc
    if payload.get("model_config_sha256") != _canonical_sha256(dict(model_config)):
        raise DevelopmentCheckpointError("model_config digest is stale")
    vocab = _condition_vocab(payload.get("condition_vocab"), cfg)
    if payload.get("condition_vocab_sha256") != _canonical_sha256(vocab):
        raise DevelopmentCheckpointError("condition vocab digest is stale")
    partitions = _partition_hashes(payload.get("partition_row_set_sha256"))
    state = payload.get("state_dict")
    shapes = payload.get("state_shapes")
    dtypes = payload.get("state_dtypes")
    if (
        not isinstance(state, Mapping)
        or not state
        or not isinstance(shapes, Mapping)
        or not isinstance(dtypes, Mapping)
        or set(state) != set(shapes)
        or set(state) != set(dtypes)
    ):
        raise DevelopmentCheckpointError("model state metadata is incomplete")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise DevelopmentCheckpointError(f"state value is not a tensor: {key}")
        if shapes[key] != list(value.shape) or dtypes[key] != str(value.dtype):
            raise DevelopmentCheckpointError(f"state metadata mismatch: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise DevelopmentCheckpointError(f"state contains non-finite values: {key}")
    expected_digests = (
        ("run_start_sha256", expected_run_start_sha256),
        ("run_plan_sha256", expected_run_plan_sha256),
        ("development_activation_sha256", expected_development_activation_sha256),
        ("estimand_binding_sha256", expected_estimand_binding_sha256),
        ("information_set_fingerprint", expected_information_set_fingerprint),
    )
    for field, expected in expected_digests:
        if expected is not None and payload.get(field) != _digest(expected, label=f"expected {field}"):
            raise DevelopmentCheckpointError(f"{field} mismatch")
    if expected_method_id is not None and payload.get("method_id") != expected_method_id:
        raise DevelopmentCheckpointError("method_id mismatch")
    if (
        expected_training_variant_id is not None
        and payload.get("training_variant_id") != expected_training_variant_id
    ):
        raise DevelopmentCheckpointError("training_variant_id mismatch")
    if expected_seed is not None and payload.get("seed") != expected_seed:
        raise DevelopmentCheckpointError("seed mismatch")
    if (
        expected_partition_row_set_sha256 is not None
        and partitions != _partition_hashes(expected_partition_row_set_sha256)
    ):
        raise DevelopmentCheckpointError("partition row-set hashes mismatch")
    return dict(payload)


def save_development_checkpoint_exclusive(path: Path, payload: Mapping) -> str:
    path = Path(path)
    try:
        assert_ordinary_path(path, purpose="development checkpoint output")
    except ProtocolHoldError as exc:
        raise DevelopmentCheckpointError(str(exc)) from exc
    if not path.parent.is_dir():
        raise DevelopmentCheckpointError("checkpoint parent must already exist")
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)

    def replay(raw: bytes) -> None:
        try:
            decoded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise DevelopmentCheckpointError(
                f"development checkpoint cannot be safely replayed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(decoded, dict) or decoded.get("schema_version") != SCHEMA:
            raise DevelopmentCheckpointError("development checkpoint replay schema changed")

    try:
        return atomic_publish_bytes(
            path, buffer.getvalue(), replay_validator=replay
        )
    except ArtifactIOError as exc:
        raise DevelopmentCheckpointError(str(exc)) from exc


def load_development_checkpoint_exact(path: Path, expected_sha256: str) -> dict:
    path = Path(path)
    try:
        assert_ordinary_path(path, purpose="development checkpoint input")
        raw = read_exact_regular_file(path, expected_sha256)
    except (ProtocolHoldError, ArtifactIOError) as exc:
        raise DevelopmentCheckpointError(str(exc)) from exc
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DevelopmentCheckpointError(
            f"development checkpoint cannot be safely decoded: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DevelopmentCheckpointError("development checkpoint is not an object")
    return payload
