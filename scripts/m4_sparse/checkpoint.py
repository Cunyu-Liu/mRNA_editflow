"""Versioned checkpoint helpers for the R3-P0 M4 model.

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

from .config import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_SELECTION_POLICY,
    DEV_SIGN_ACCURACY_GATE,
    INPUT_SCHEMA_VERSION,
    RANK_PAIR_DEFINITION,
    RANK_SCORE_DEFINITION,
    SIGN_CALIBRATION_METHOD,
    SIGN_DECISION_RULE,
    SIGN_SCORE_DEFINITION,
)


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a checkpoint predates or disagrees with the R3-P0 schema."""


class RankSupervisionUnavailableError(IncompatibleCheckpointError):
    """Raised when a ranking/G1 consumer requests an effect-only checkpoint."""


def vocab_sha256(vocab: Dict[str, Dict[str, int]]) -> str:
    payload = json.dumps(
        vocab, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_checkpoint_payload(state_dict, cfg, held_out_study: str,
                             vocab: Dict[str, Dict[str, int]],
                             rank_supervision: Dict,
                             dev_selection: Dict) -> dict:
    """Build the complete, self-describing R3-P0 checkpoint payload."""
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "state_dict": state_dict,
        "cfg": cfg,
        "held_out_study": held_out_study,
        "vocab": vocab,
        "vocab_sha256": vocab_sha256(vocab),
        "rank_supervision_status": rank_supervision.get("status"),
        "qualified_rank_pair_count": rank_supervision.get(
            "qualified_rank_pair_count"),
        "eligible_rank_group_count": rank_supervision.get(
            "eligible_rank_group_count"),
        "rank_pair_definition": rank_supervision.get("pair_definition"),
        "rank_score_definition": rank_supervision.get("score_definition"),
        "dev_selection": dev_selection,
    }
    return validate_checkpoint_payload(payload)


def validate_checkpoint_payload(payload: object,
                                expected_vocab: Optional[Dict] = None) -> dict:
    """Validate schema and optional vocabulary identity, returning payload.

    Missing schema fields are treated as a legacy checkpoint and rejected.
    This is intentionally not an automatic migration: removal of independent
    sign/rank heads and the R3-P0 selection policy require retraining.
    """
    if not isinstance(payload, dict):
        raise IncompatibleCheckpointError("M4 checkpoint payload is not a mapping")
    schema = payload.get("checkpoint_schema_version")
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError(
            "incompatible M4 checkpoint schema %r; expected %r; retrain R3-P0 "
            "instead of reusing a legacy checkpoint"
            % (schema, CHECKPOINT_SCHEMA_VERSION)
        )
    input_schema = payload.get("input_schema_version")
    if input_schema != INPUT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError(
            "incompatible M4 input schema %r; expected %r"
            % (input_schema, INPUT_SCHEMA_VERSION)
        )
    required = {
        "state_dict", "cfg", "held_out_study", "vocab", "vocab_sha256",
        "rank_supervision_status", "qualified_rank_pair_count",
        "eligible_rank_group_count", "rank_pair_definition",
        "rank_score_definition", "dev_selection",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise IncompatibleCheckpointError(
            "R3-P0 M4 checkpoint missing fields: " + ", ".join(missing)
        )
    if not isinstance(payload["state_dict"], Mapping) or not payload["state_dict"]:
        raise IncompatibleCheckpointError(
            "R3-P0 M4 checkpoint state_dict is missing or empty"
        )
    if not str(payload["held_out_study"] or "").strip():
        raise IncompatibleCheckpointError(
            "R3-P0 M4 checkpoint held_out_study is empty"
        )
    if not isinstance(payload["vocab"], dict):
        raise IncompatibleCheckpointError("R3-P0 M4 checkpoint vocab is not a mapping")
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
        raise IncompatibleCheckpointError("checkpoint cfg is not bound to the R3-P0 schema")
    if getattr(cfg, "INPUT_SCHEMA_VERSION", None) != INPUT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError("checkpoint cfg is not bound to the R3-P0 input schema")
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
            "checkpoint nucleotide/window contract differs from R3-P0"
        )
    if getattr(cfg, "TARGET", None) not in ("delta", "candidate_value"):
        raise IncompatibleCheckpointError(
            "checkpoint regression TARGET is not supported by R3-P0"
        )
    if (getattr(cfg, "TARGET", None) == "candidate_value"
            and getattr(cfg, "ANCHOR_AT_TEST", None) is not True):
        raise IncompatibleCheckpointError(
            "candidate_value checkpoint is missing the measured-anchor contract"
        )
    if getattr(cfg, "RANK_PAIR_DEFINITION", None) != RANK_PAIR_DEFINITION:
        raise IncompatibleCheckpointError("checkpoint rank-pair definition is not frozen R3-P0")
    if getattr(cfg, "RANK_SCORE_DEFINITION", None) != RANK_SCORE_DEFINITION:
        raise IncompatibleCheckpointError("checkpoint rank score is not predicted delta mean")
    if getattr(cfg, "SIGN_SCORE_DEFINITION", None) != SIGN_SCORE_DEFINITION:
        raise IncompatibleCheckpointError("checkpoint sign score is not frozen mu/sigma")
    if getattr(cfg, "SIGN_DECISION_RULE", None) != SIGN_DECISION_RULE:
        raise IncompatibleCheckpointError("checkpoint sign-decision rule is not frozen R3-P0")
    if getattr(cfg, "SIGN_CALIBRATION_METHOD", None) != SIGN_CALIBRATION_METHOD:
        raise IncompatibleCheckpointError(
            "checkpoint sign calibration method is not probability-only R3-P0"
        )
    if getattr(cfg, "CHECKPOINT_SELECTION_POLICY", None) != CHECKPOINT_SELECTION_POLICY:
        raise IncompatibleCheckpointError(
            "checkpoint selection policy is not frozen study-balanced R3-P0"
        )
    cfg_sign_gate = getattr(cfg, "DEV_SIGN_ACCURACY_GATE", None)
    if (isinstance(cfg_sign_gate, bool)
            or not isinstance(cfg_sign_gate, (int, float))
            or not math.isfinite(float(cfg_sign_gate))
            or float(cfg_sign_gate) != float(DEV_SIGN_ACCURACY_GATE)):
        raise IncompatibleCheckpointError(
            "checkpoint dev sign-accuracy gate differs from frozen 0.60"
        )
    temperature = getattr(cfg, "SIGN_TEMPERATURE", None)
    if (not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature)) or float(temperature) <= 0):
        raise IncompatibleCheckpointError("checkpoint sign temperature is invalid")
    forbidden_heads = [
        str(key) for key in payload["state_dict"]
        if "rank_head" in str(key) or "sign_head" in str(key)
    ]
    if forbidden_heads:
        raise IncompatibleCheckpointError(
            "R3-P0 checkpoint contains forbidden independent sign/rank heads"
        )
    _validate_rank_supervision(payload)
    _validate_dev_selection(payload["dev_selection"])
    return payload


def _validate_rank_supervision(payload: Mapping) -> None:
    count = payload.get("qualified_rank_pair_count")
    groups = payload.get("eligible_rank_group_count")
    if (isinstance(count, bool) or not isinstance(count, int) or count < 0
            or isinstance(groups, bool) or not isinstance(groups, int)
            or groups < 0):
        raise IncompatibleCheckpointError(
            "checkpoint qualified rank pair/group counts are invalid"
        )
    status = payload.get("rank_supervision_status")
    expected_status = (
        "QUALIFIED" if count > 0 else "NOT_RUN_ZERO_QUALIFIED_PAIRS"
    )
    if status != expected_status:
        raise IncompatibleCheckpointError(
            "checkpoint rank supervision status/count are inconsistent"
        )
    if payload.get("rank_pair_definition") != RANK_PAIR_DEFINITION:
        raise IncompatibleCheckpointError(
            "checkpoint qualified rank pair definition is incompatible"
        )
    if payload.get("rank_score_definition") != RANK_SCORE_DEFINITION:
        raise IncompatibleCheckpointError(
            "checkpoint rank score definition is incompatible"
        )


def _validate_dev_selection(selection: object) -> None:
    if not isinstance(selection, Mapping):
        raise IncompatibleCheckpointError(
            "checkpoint dev selection evidence is not a mapping"
        )
    if selection.get("policy") != CHECKPOINT_SELECTION_POLICY:
        raise IncompatibleCheckpointError(
            "checkpoint dev selection policy is incompatible"
        )
    if selection.get("sign_accuracy_gate") != DEV_SIGN_ACCURACY_GATE:
        raise IncompatibleCheckpointError(
            "checkpoint dev selection sign gate differs from frozen 0.60"
        )
    epoch = selection.get("selected_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise IncompatibleCheckpointError(
            "checkpoint selected epoch is invalid"
        )
    metrics = selection.get("selected_metrics")
    if not isinstance(metrics, Mapping):
        raise IncompatibleCheckpointError(
            "checkpoint selected dev metrics are missing"
        )
    required_metrics = (
        "study_macro_sign_accuracy", "study_macro_delta_spearman",
        "study_macro_gaussian_nll", "study_coverage",
    )
    for name in required_metrics:
        value = metrics.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise IncompatibleCheckpointError(
                "checkpoint selected dev metric %s is missing/non-finite" % name
            )
    if float(metrics["study_coverage"]) != 1.0:
        raise IncompatibleCheckpointError(
            "checkpoint selected dev metrics do not have full study coverage"
        )
    if float(metrics["study_macro_sign_accuracy"]) < DEV_SIGN_ACCURACY_GATE:
        raise IncompatibleCheckpointError(
            "checkpoint selected dev sign accuracy is below frozen 0.60 gate"
        )


def require_rank_supervision(payload: Mapping) -> Mapping:
    """Fail closed before any ranking or G1 use of an effect-only checkpoint."""
    _validate_rank_supervision(payload)
    if (payload.get("rank_supervision_status") != "QUALIFIED"
            or int(payload.get("qualified_rank_pair_count", 0)) <= 0):
        raise RankSupervisionUnavailableError(
            "ranking/G1 requires qualified within-pool rank pairs; checkpoint "
            "is effect-only with zero qualified pairs"
        )
    return payload


def load_checkpoint(path: Path, device, expected_vocab: Optional[Dict] = None,
                    require_rank: bool = False) -> dict:
    import torch

    payload = torch.load(Path(path), map_location=str(device), weights_only=False)
    payload = validate_checkpoint_payload(payload, expected_vocab=expected_vocab)
    if require_rank:
        require_rank_supervision(payload)
    return payload
