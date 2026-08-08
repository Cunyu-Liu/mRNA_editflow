"""Hash- and schema-bound R3X checkpoint payloads."""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch

from .models import ModelConfig
from .artifact_io import ArtifactIOError, atomic_publish_bytes, read_exact_regular_file
from .protocol import ProtocolHoldError, assert_ordinary_path


SCHEMA = "r3x_critic_checkpoint_v2"
REQUIRED_AUTHORITY_HASHES = (
    "route_authorization_sha256",
    "ready_dataset_sha256",
    "inclusion_manifest_sha256",
    "context_source_crosswalk_sha256",
    "endpoint_estimand_registry_sha256",
    "source_component_split_manifest_sha256",
    "metric_zero_policy_registry_sha256",
    "model_optimizer_seed_config_sha256",
    "analysis_code_tree_manifest_sha256",
    "method_protocol_sha256",
    "paired_strongest_baseline_comparison_policy_sha256",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FROZEN_SEEDS = {11, 23, 42, 67, 101}
ALLOWED_TRAINING_VARIANTS = {
    "PRIMARY",
    "SOURCE_CANDIDATE_PAIRING_SHUFFLE",
    "CONTEXT_LABEL_PERMUTATION",
    "SOURCE_CONTEXT_SHUFFLE_WITHIN_ACTION",
    "EDIT_POSITION_SHUFFLE_WITHIN_SOURCE",
    "TARGET_TOKEN_SHUFFLE_WITHIN_CELL",
}
CHECKPOINT_KEYS = {
    "schema_version", "artifact_role", "run_start_sha256", "run_plan_sha256",
    "activation_manifest_sha256", "authority_hashes", "estimand_binding_sha256",
    "method_id", "training_variant_id", "architecture_id", "model_config",
    "model_config_sha256", "condition_vocab", "condition_vocab_sha256",
    "training_config_sha256", "information_set",
    "information_set_fingerprint", "seed", "scientific_scope", "selected_epoch",
    "checkpoint_selection_sha256", "partition_row_set_sha256", "sign_gate",
    "sign_decision_rule", "ordinary_only", "sealed_access", "downstream_unlock",
    "state_shapes", "state_dtypes", "state_dict",
}


class CheckpointProtocolError(RuntimeError):
    pass


def _hashes(values: Mapping) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise CheckpointProtocolError("authority_hashes must be an object")
    result = {}
    for name in REQUIRED_AUTHORITY_HASHES:
        value = str(values.get(name) or "").lower()
        if not HEX64.fullmatch(value):
            raise CheckpointProtocolError(f"missing or invalid {name}")
        result[name] = value
    extra = sorted(set(values) - set(REQUIRED_AUTHORITY_HASHES))
    if extra:
        raise CheckpointProtocolError(f"unexpected authority hashes: {extra}")
    return result


def _digest(value: object, label: str) -> str:
    text = str(value or "").lower()
    if not HEX64.fullmatch(text):
        raise CheckpointProtocolError(f"missing or invalid {label}")
    return text


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _condition_vocab(value: object, cfg: ModelConfig) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or set(value) != {"endpoint", "context"}:
        raise CheckpointProtocolError(
            "condition_vocab must contain exactly endpoint and context mappings"
        )
    normalized: dict[str, dict[str, int]] = {}
    for axis, capacity in (("endpoint", cfg.n_endpoints), ("context", cfg.n_contexts)):
        raw = value.get(axis)
        if not isinstance(raw, Mapping) or not raw:
            raise CheckpointProtocolError(f"condition_vocab {axis} mapping is empty")
        if raw.get("__UNK__") != 0:
            raise CheckpointProtocolError(
                f"condition_vocab {axis} must bind explicit __UNK__ to zero"
            )
        mapping: dict[str, int] = {}
        for key, index in raw.items():
            if not isinstance(key, str) or not key or key != key.strip():
                raise CheckpointProtocolError(
                    f"condition_vocab {axis} keys must be canonical non-empty strings"
                )
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise CheckpointProtocolError(
                    f"condition_vocab {axis} indices must be non-negative integers"
                )
            mapping[key] = int(index)
        indices = sorted(mapping.values())
        if indices != list(range(len(mapping))) or len(indices) != len(set(indices)):
            raise CheckpointProtocolError(
                f"condition_vocab {axis} indices must be unique and contiguous from zero"
            )
        if len(mapping) > capacity:
            raise CheckpointProtocolError(
                f"condition_vocab {axis} exceeds the frozen model capacity"
            )
        normalized[axis] = dict(sorted(mapping.items()))
    return normalized


def build_checkpoint(
    model: torch.nn.Module,
    cfg: ModelConfig,
    *,
    authority_hashes: Mapping,
    information_set: str,
    seed: int,
    scientific_scope: str,
    run_start_sha256: str,
    run_plan_sha256: str,
    activation_manifest_sha256: str,
    estimand_binding_sha256: str,
    method_id: str,
    training_variant_id: str,
    training_config_sha256: str,
    condition_vocab: Mapping,
    information_set_fingerprint: str,
    selected_epoch: int,
    checkpoint_selection_sha256: str,
    partition_row_set_sha256: Mapping[str, str],
) -> dict:
    architecture = getattr(model, "architecture_id", None)
    if not isinstance(architecture, str) or not architecture:
        raise CheckpointProtocolError("model lacks architecture_id")
    if information_set != "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR":
        raise CheckpointProtocolError("A/B checkpoint information set must be paired without measured anchor")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in FROZEN_SEEDS:
        raise CheckpointProtocolError("seed must be one of the five frozen seeds")
    if not isinstance(scientific_scope, str) or not scientific_scope:
        raise CheckpointProtocolError("scientific_scope is required")
    if not isinstance(method_id, str) or not method_id:
        raise CheckpointProtocolError("method_id is required")
    if training_variant_id not in ALLOWED_TRAINING_VARIANTS:
        raise CheckpointProtocolError("training_variant_id is not frozen")
    if isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int) or selected_epoch < 1:
        raise CheckpointProtocolError("selected_epoch must be a positive integer")
    partition_hashes = dict(partition_row_set_sha256) if isinstance(partition_row_set_sha256, Mapping) else {}
    if set(partition_hashes) != {"train", "development", "internal_test"}:
        raise CheckpointProtocolError("partition row-set hashes are incomplete or unexpected")
    partition_hashes = {
        key: _digest(value, f"{key} row-set SHA-256")
        for key, value in partition_hashes.items()
    }
    model_config = asdict(cfg)
    normalized_vocab = _condition_vocab(condition_vocab, cfg)
    state_dict = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    return {
        "schema_version": SCHEMA,
        "artifact_role": "ORDINARY_ESTIMAND_METHOD_VARIANT_SEED_CHECKPOINT",
        "run_start_sha256": _digest(run_start_sha256, "run_start_sha256"),
        "run_plan_sha256": _digest(run_plan_sha256, "run_plan_sha256"),
        "activation_manifest_sha256": _digest(
            activation_manifest_sha256, "activation_manifest_sha256"
        ),
        "architecture_id": architecture,
        "model_config": model_config,
        "model_config_sha256": _canonical_sha256(model_config),
        "condition_vocab": normalized_vocab,
        "condition_vocab_sha256": _canonical_sha256(normalized_vocab),
        "training_config_sha256": _digest(
            training_config_sha256, "training_config_sha256"
        ),
        "authority_hashes": _hashes(authority_hashes),
        "estimand_binding_sha256": _digest(
            estimand_binding_sha256, "estimand_binding_sha256"
        ),
        "method_id": method_id,
        "training_variant_id": training_variant_id,
        "information_set": information_set,
        "information_set_fingerprint": _digest(
            information_set_fingerprint, "information_set_fingerprint"
        ),
        "seed": seed,
        "scientific_scope": scientific_scope,
        "selected_epoch": selected_epoch,
        "checkpoint_selection_sha256": _digest(
            checkpoint_selection_sha256, "checkpoint_selection_sha256"
        ),
        "partition_row_set_sha256": partition_hashes,
        "sign_gate": 0.60,
        "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
        "state_shapes": {key: list(value.shape) for key, value in state_dict.items()},
        "state_dtypes": {key: str(value.dtype) for key, value in state_dict.items()},
        "state_dict": state_dict,
    }


def validate_checkpoint(
    payload: Mapping,
    *,
    expected_architecture_id: str,
    expected_authority_hashes: Mapping,
    expected_model_config: ModelConfig | None = None,
    expected_condition_vocab: Mapping | None = None,
    expected_run_start_sha256: str | None = None,
    expected_run_plan_sha256: str | None = None,
    expected_activation_manifest_sha256: str | None = None,
    expected_training_config_sha256: str | None = None,
    expected_checkpoint_selection_sha256: str | None = None,
    expected_seed: int | None = None,
    expected_scientific_scope: str | None = None,
    expected_state_shapes: Mapping[str, tuple[int, ...]] | None = None,
    expected_method_id: str | None = None,
    expected_training_variant_id: str | None = None,
    expected_estimand_binding_sha256: str | None = None,
    expected_information_set_fingerprint: str | None = None,
    expected_partition_row_set_sha256: Mapping[str, str] | None = None,
) -> dict:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA:
        raise CheckpointProtocolError("unsupported checkpoint schema")
    if set(payload) != CHECKPOINT_KEYS:
        raise CheckpointProtocolError(
            f"checkpoint keys differ from strict schema: {sorted(set(payload) ^ CHECKPOINT_KEYS)}"
        )
    if payload.get("artifact_role") != "ORDINARY_ESTIMAND_METHOD_VARIANT_SEED_CHECKPOINT":
        raise CheckpointProtocolError("checkpoint artifact role is invalid")
    for field in (
        "run_start_sha256", "run_plan_sha256", "activation_manifest_sha256",
        "estimand_binding_sha256", "model_config_sha256", "condition_vocab_sha256",
        "training_config_sha256", "information_set_fingerprint",
        "checkpoint_selection_sha256",
    ):
        _digest(payload.get(field), field)
    if payload.get("architecture_id") != expected_architecture_id:
        raise CheckpointProtocolError(
            f"architecture mismatch: expected {expected_architecture_id}, got {payload.get('architecture_id')}"
        )
    expected = _hashes(expected_authority_hashes)
    actual = _hashes(payload.get("authority_hashes"))
    for name in REQUIRED_AUTHORITY_HASHES:
        if actual[name] != expected[name]:
            raise CheckpointProtocolError(
                f"{name} mismatch: expected {expected[name]}, got {actual[name]}"
            )
    if payload.get("information_set") != "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR":
        raise CheckpointProtocolError("checkpoint information set is invalid")
    if payload.get("sign_gate") != 0.60:
        raise CheckpointProtocolError("checkpoint sign gate must remain 0.60")
    if payload.get("sign_decision_rule") != "RAW_PREDICTED_DELTA_MEAN_GE_ZERO":
        raise CheckpointProtocolError("checkpoint sign decision rule changed")
    if payload.get("sealed_access") is not False or payload.get("downstream_unlock") is not False:
        raise CheckpointProtocolError("checkpoint cannot carry sealed/downstream permission")
    if payload.get("ordinary_only") is not True:
        raise CheckpointProtocolError("checkpoint must remain ordinary-only")
    if payload.get("training_variant_id") not in ALLOWED_TRAINING_VARIANTS:
        raise CheckpointProtocolError("checkpoint training variant is not frozen")
    if payload.get("seed") not in FROZEN_SEEDS:
        raise CheckpointProtocolError("checkpoint seed is not frozen")
    if (isinstance(payload.get("selected_epoch"), bool)
            or not isinstance(payload.get("selected_epoch"), int)
            or payload["selected_epoch"] < 1):
        raise CheckpointProtocolError("checkpoint selected_epoch is invalid")
    if not isinstance(payload.get("state_dict"), Mapping) or not payload["state_dict"]:
        raise CheckpointProtocolError("checkpoint lacks model state")
    if payload.get("model_config_sha256") != _canonical_sha256(payload.get("model_config")):
        raise CheckpointProtocolError("checkpoint model config digest is stale")
    try:
        payload_cfg = ModelConfig(**dict(payload.get("model_config") or {}))
    except (TypeError, ValueError) as exc:
        raise CheckpointProtocolError(f"checkpoint model config is invalid: {exc}") from exc
    normalized_vocab = _condition_vocab(payload.get("condition_vocab"), payload_cfg)
    if payload.get("condition_vocab_sha256") != _canonical_sha256(normalized_vocab):
        raise CheckpointProtocolError("checkpoint condition vocab digest is stale")
    partition_hashes = payload.get("partition_row_set_sha256")
    if not isinstance(partition_hashes, Mapping) or set(partition_hashes) != {
        "train", "development", "internal_test"
    }:
        raise CheckpointProtocolError("checkpoint partition hashes are incomplete")
    for key, value in partition_hashes.items():
        _digest(value, f"{key} row-set SHA-256")
    state = payload["state_dict"]
    shapes = payload.get("state_shapes")
    dtypes = payload.get("state_dtypes")
    if not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping) or set(state) != set(shapes) or set(state) != set(dtypes):
        raise CheckpointProtocolError("checkpoint state metadata keys mismatch")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise CheckpointProtocolError(f"checkpoint state value is not a tensor: {key}")
        if shapes[key] != list(value.shape) or dtypes[key] != str(value.dtype):
            raise CheckpointProtocolError(f"checkpoint state metadata mismatch for {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise CheckpointProtocolError(
                f"checkpoint state contains non-finite values for {key}"
            )
    if expected_model_config is not None and payload.get("model_config") != asdict(expected_model_config):
        raise CheckpointProtocolError("checkpoint model config mismatch")
    if expected_condition_vocab is not None:
        expected_cfg = expected_model_config or payload_cfg
        if normalized_vocab != _condition_vocab(expected_condition_vocab, expected_cfg):
            raise CheckpointProtocolError("checkpoint condition vocab mismatch")
    expected_digests = (
        ("run_start_sha256", expected_run_start_sha256),
        ("run_plan_sha256", expected_run_plan_sha256),
        ("activation_manifest_sha256", expected_activation_manifest_sha256),
        ("training_config_sha256", expected_training_config_sha256),
        ("checkpoint_selection_sha256", expected_checkpoint_selection_sha256),
    )
    for field, expected_value in expected_digests:
        if (
            expected_value is not None
            and payload.get(field) != _digest(expected_value, f"expected {field}")
        ):
            raise CheckpointProtocolError(f"checkpoint {field} mismatch")
    if expected_seed is not None and payload.get("seed") != expected_seed:
        raise CheckpointProtocolError("checkpoint seed mismatch")
    if expected_scientific_scope is not None and payload.get("scientific_scope") != expected_scientific_scope:
        raise CheckpointProtocolError("checkpoint scientific scope mismatch")
    if expected_method_id is not None and payload.get("method_id") != expected_method_id:
        raise CheckpointProtocolError("checkpoint method ID mismatch")
    if expected_training_variant_id is not None and payload.get("training_variant_id") != expected_training_variant_id:
        raise CheckpointProtocolError("checkpoint training variant mismatch")
    if expected_estimand_binding_sha256 is not None and payload.get("estimand_binding_sha256") != _digest(expected_estimand_binding_sha256, "expected estimand binding"):
        raise CheckpointProtocolError("checkpoint estimand binding mismatch")
    if expected_information_set_fingerprint is not None and payload.get("information_set_fingerprint") != _digest(expected_information_set_fingerprint, "expected information-set fingerprint"):
        raise CheckpointProtocolError("checkpoint information-set fingerprint mismatch")
    if expected_partition_row_set_sha256 is not None and dict(partition_hashes) != dict(expected_partition_row_set_sha256):
        raise CheckpointProtocolError("checkpoint partition row-set binding mismatch")
    if expected_state_shapes is not None:
        state = payload["state_dict"]
        if set(state) != set(expected_state_shapes):
            raise CheckpointProtocolError("checkpoint state_dict keys mismatch")
        for key, expected_shape in expected_state_shapes.items():
            value = state[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(expected_shape):
                raise CheckpointProtocolError(f"checkpoint state shape mismatch for {key}")
    return dict(payload)


def save_checkpoint_exclusive(path: Path, payload: Mapping) -> str:
    path = Path(path)
    try:
        assert_ordinary_path(path, purpose="R3X checkpoint output")
    except ProtocolHoldError as exc:
        raise CheckpointProtocolError(str(exc)) from exc
    if not path.parent.is_dir():
        raise CheckpointProtocolError("checkpoint parent must already exist after successful preflight")
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    raw = buffer.getvalue()

    def replay(data: bytes) -> None:
        try:
            decoded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointProtocolError(
                f"checkpoint bytes cannot be safely replayed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise CheckpointProtocolError("checkpoint replay is not an object")

    try:
        return atomic_publish_bytes(path, raw, replay_validator=replay)
    except ArtifactIOError as exc:
        raise CheckpointProtocolError(str(exc)) from exc


def load_checkpoint_exact(path: Path, expected_sha256: str) -> dict:
    path = Path(path)
    try:
        assert_ordinary_path(path, purpose="R3X checkpoint input")
    except ProtocolHoldError as exc:
        raise CheckpointProtocolError(str(exc)) from exc
    try:
        raw = read_exact_regular_file(path, expected_sha256)
    except ArtifactIOError as exc:
        raise CheckpointProtocolError(str(exc)) from exc
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointProtocolError(
            f"checkpoint payload cannot be safely decoded: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointProtocolError("checkpoint payload is not an object")
    return payload
