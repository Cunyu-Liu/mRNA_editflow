"""Versioned checkpoint helpers for the repaired R2 M4 model.

R2 adds a reserved unknown-context embedding and changes sequence
preprocessing from prefix truncation to a shared edit-centred window.  A
legacy checkpoint can therefore be shape-compatible in some configurations
while still being semantically incompatible.  These helpers make that
boundary explicit and fail before model construction or scoring.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Optional

from .config import CHECKPOINT_SCHEMA_VERSION, INPUT_SCHEMA_VERSION


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a checkpoint predates or disagrees with the R2 schema."""


def vocab_sha256(vocab: Dict[str, Dict[str, int]]) -> str:
    payload = json.dumps(
        vocab, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_checkpoint_payload(state_dict, cfg, held_out_study: str,
                             vocab: Dict[str, Dict[str, int]]) -> dict:
    """Build the complete, self-describing R2 checkpoint payload."""
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "state_dict": state_dict,
        "cfg": cfg,
        "held_out_study": held_out_study,
        "vocab": vocab,
        "vocab_sha256": vocab_sha256(vocab),
    }
    return validate_checkpoint_payload(payload)


def validate_checkpoint_payload(payload: object,
                                expected_vocab: Optional[Dict] = None) -> dict:
    """Validate schema and optional vocabulary identity, returning payload.

    Missing schema fields are treated as a legacy R1-or-earlier checkpoint and
    rejected.  This is intentionally not an automatic migration: the R2 UNK
    embeddings and edit-centred preprocessing must be learned by retraining.
    """
    if not isinstance(payload, dict):
        raise IncompatibleCheckpointError("M4 checkpoint payload is not a mapping")
    schema = payload.get("checkpoint_schema_version")
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError(
            "incompatible M4 checkpoint schema %r; expected %r; retrain R2 "
            "instead of reusing a legacy checkpoint"
            % (schema, CHECKPOINT_SCHEMA_VERSION)
        )
    input_schema = payload.get("input_schema_version")
    if input_schema != INPUT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError(
            "incompatible M4 input schema %r; expected %r"
            % (input_schema, INPUT_SCHEMA_VERSION)
        )
    required = {"state_dict", "cfg", "held_out_study", "vocab", "vocab_sha256"}
    missing = sorted(required - set(payload))
    if missing:
        raise IncompatibleCheckpointError(
            "R2 M4 checkpoint missing fields: " + ", ".join(missing)
        )
    if not isinstance(payload["state_dict"], Mapping) or not payload["state_dict"]:
        raise IncompatibleCheckpointError(
            "R2 M4 checkpoint state_dict is missing or empty"
        )
    if not str(payload["held_out_study"] or "").strip():
        raise IncompatibleCheckpointError(
            "R2 M4 checkpoint held_out_study is empty"
        )
    if not isinstance(payload["vocab"], dict):
        raise IncompatibleCheckpointError("R2 M4 checkpoint vocab is not a mapping")
    try:
        actual_vocab_sha = vocab_sha256(payload["vocab"])
    except (TypeError, ValueError, AttributeError) as exc:
        raise IncompatibleCheckpointError(
            "R2 M4 checkpoint vocabulary is not canonical JSON"
        ) from exc
    if payload["vocab_sha256"] != actual_vocab_sha:
        raise IncompatibleCheckpointError("M4 checkpoint vocabulary hash mismatch")
    if expected_vocab is not None and vocab_sha256(expected_vocab) != actual_vocab_sha:
        raise IncompatibleCheckpointError(
            "runtime M4 vocabulary does not match the checkpoint vocabulary"
        )
    cfg = payload["cfg"]
    if getattr(cfg, "CHECKPOINT_SCHEMA_VERSION", None) != CHECKPOINT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError("checkpoint cfg is not bound to the R2 schema")
    if getattr(cfg, "INPUT_SCHEMA_VERSION", None) != INPUT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError("checkpoint cfg is not bound to the R2 input schema")
    for namespace, cfg_field, state_key in (
        ("study", "N_STUDIES", "study_emb.weight"),
        ("endpoint", "N_ENDPOINTS", "endpoint_emb.weight"),
        ("benchmark", "N_BENCHMARKS", "bench_emb.weight"),
    ):
        mapping = payload["vocab"].get(namespace)
        if not isinstance(mapping, dict) or "__UNK__" not in mapping:
            raise IncompatibleCheckpointError(
                "checkpoint vocab namespace %s has no reserved __UNK__ id"
                % namespace
            )
        if (not mapping
                or any(not isinstance(key, str) for key in mapping)
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in mapping.values())
                or sorted(mapping.values()) != list(range(len(mapping)))):
            raise IncompatibleCheckpointError(
                "checkpoint vocab namespace %s ids are not a contiguous bijection"
                % namespace
            )
        if getattr(cfg, cfg_field, None) != len(mapping):
            raise IncompatibleCheckpointError(
                "checkpoint %s embedding count does not match vocab" % namespace
            )
        if int(mapping["__UNK__"]) != len(mapping) - 1:
            raise IncompatibleCheckpointError(
                "checkpoint %s __UNK__ id is not the frozen final id" % namespace
            )
        weight = payload["state_dict"].get(state_key)
        if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
            raise IncompatibleCheckpointError(
                "checkpoint %s embedding tensor is missing" % namespace
            )
        if int(weight.shape[0]) != len(mapping):
            raise IncompatibleCheckpointError(
                "checkpoint %s embedding tensor does not match vocab" % namespace
            )
    if getattr(cfg, "MAX_SEQ_LEN", None) != 100 or getattr(cfg, "NUC_ORDER", None) != "ACGU":
        raise IncompatibleCheckpointError(
            "checkpoint nucleotide/window contract differs from R2"
        )
    if getattr(cfg, "TARGET", None) not in ("delta", "candidate_value"):
        raise IncompatibleCheckpointError(
            "checkpoint regression TARGET is not supported by R2"
        )
    if (getattr(cfg, "TARGET", None) == "candidate_value"
            and getattr(cfg, "ANCHOR_AT_TEST", None) is not True):
        raise IncompatibleCheckpointError(
            "candidate_value checkpoint is missing the measured-anchor contract"
        )
    if getattr(cfg, "SIGN_DECISION_RULE", None) != "sign_head_logit_ge_0_r2_prospective_v1":
        raise IncompatibleCheckpointError("checkpoint sign-decision rule is not frozen R2")
    if getattr(cfg, "SIGN_CALIBRATION_METHOD", None) != "temperature_scaling_grouped_dev_v1":
        raise IncompatibleCheckpointError(
            "checkpoint sign calibration method is not frozen R2"
        )
    temperature = getattr(cfg, "SIGN_TEMPERATURE", None)
    if (not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature)) or float(temperature) <= 0):
        raise IncompatibleCheckpointError("checkpoint sign temperature is invalid")
    return payload


def load_checkpoint(path: Path, device, expected_vocab: Optional[Dict] = None) -> dict:
    import torch

    payload = torch.load(Path(path), map_location=str(device), weights_only=False)
    return validate_checkpoint_payload(payload, expected_vocab=expected_vocab)
