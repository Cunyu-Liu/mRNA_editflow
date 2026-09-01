"""V5 successor runtime for XEditSetFlow base-model repair (guided-generation prep).

V5 reuses the frozen architecture classes from V4 but parameterizes per-arm:
run role, architecture profile (A1 small 8-15M vs V4 98-100M), coverage
weight (F2 ablation), mode count (F4), and training schedule (F1 early stop).
All switches default to V4-S1 behavior unless explicitly set in the V5 screen
config, so frozen V4/V4-S1 code paths remain bit-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from core.route2_xeditsetflow_v4 import XEditSetFlowV4


class XEditSetFlowRuntimeV5Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowRuntimeV5Error(message)


@dataclass(frozen=True)
class SetFlowScreenRunSpecV5:
    run_id: str
    mode_count: int
    mode_information_weight: float
    coverage_weight: float
    selectable: bool
    architecture_profile: str


def screen_run_spec_v5(
    config: Mapping[str, Any], run_id: str
) -> SetFlowScreenRunSpecV5:
    """Resolve one V5 arm without relabeling its external run identity."""

    rows = config.get("required_screen_runs")
    _require(isinstance(rows, list) and bool(rows), "SetFlow V5 screen runs are absent")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("run_id")) == str(run_id)
    ]
    _require(len(matches) == 1, "run id is not one frozen SetFlow V5 screen run")
    row = matches[0]
    spec = SetFlowScreenRunSpecV5(
        run_id=str(row["run_id"]),
        mode_count=int(row["mode_count"]),
        mode_information_weight=float(row["mode_information_weight"]),
        coverage_weight=float(row.get("coverage_weight", 0.50)),
        selectable=bool(row["selectable"]),
        architecture_profile=str(row.get("architecture_profile", "V4_FULL")),
    )
    _require(
        spec.mode_count >= 1 and spec.coverage_weight >= 0.0,
        "SetFlow V5 run spec has an invalid mode/coverage field",
    )
    return spec


def _apply_architecture_profile(
    architecture: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    profiles = architecture.get("architecture_profiles")
    _require(
        isinstance(profiles, Mapping) and profile in profiles,
        f"SetFlow V5 architecture profile {profile} is undeclared",
    )
    merged = {**dict(architecture)}
    merged.update(profiles[profile])
    return merged


def build_setflow_screen_model_v5(
    config: Mapping[str, Any],
    vocabs: Mapping[str, Mapping[str, int]],
    *,
    run_id: str,
) -> tuple[XEditSetFlowV4, dict[str, Any]]:
    """Build the V5 model: same class family as V4, size driven by profile."""

    spec = screen_run_spec_v5(config, run_id)
    architecture = _apply_architecture_profile(
        config["architecture"], spec.architecture_profile
    )
    expected_vocab = architecture["formal_endpoint_vocab_cardinalities"]
    for field in (
        "assay",
        "context",
        "quantity",
        "measurement",
        "numerator",
        "denominator",
    ):
        _require(
            len(vocabs[field]) == int(expected_vocab[field]),
            f"SetFlow V5 frozen {field} vocabulary cardinality changed",
        )
    model = XEditSetFlowV4(
        assay_count=len(vocabs["assay"]),
        context_count=len(vocabs["context"]),
        quantity_count=len(vocabs["quantity"]),
        measurement_count=len(vocabs["measurement"]),
        numerator_count=len(vocabs["numerator"]),
        denominator_count=len(vocabs["denominator"]),
        pretrained_width=int(architecture["frozen_source_mrnabert_width"]),
        model_width=int(architecture["model_width"]),
        depth=int(architecture["depth"]),
        heads=int(architecture["attention_heads"]),
        ffn_width=int(architecture["ffn_width"]),
        local_attention_window=int(architecture["local_attention_window"]),
        mode_count=spec.mode_count,
        mode_residual_rank=int(architecture["mode_residual_rank"]),
        stop_bottleneck_width=int(architecture["stop_bottleneck_width"]),
        dropout=float(architecture["dropout"]),
        activation_checkpointing=True,
    )
    capacity = require_setflow_v5_trainable_parameter_range(model, spec)
    return model, capacity


def require_setflow_v5_trainable_parameter_range(
    model: XEditSetFlowV4,
    spec: SetFlowScreenRunSpecV5,
) -> dict[str, Any]:
    """V5 pre-registered parameter bands (A1 profile 5M-20M; V4 80-150M)."""

    count = int(model.trainable_parameter_count)
    if spec.architecture_profile == "A1":
        return require_setflow_v5_parameter_band(
            model, spec, minimum=5_000_000, maximum=20_000_000
        )
    return require_setflow_v5_parameter_band(
        model, spec, minimum=80_000_000, maximum=150_000_000
    )


def require_setflow_v5_parameter_band(
    model: XEditSetFlowV4,
    spec: SetFlowScreenRunSpecV5,
    *,
    minimum: int,
    maximum: int,
) -> dict[str, Any]:
    count = int(model.trainable_parameter_count)
    _require(minimum <= count <= maximum, "SetFlow V5 trainable count is outside its band")
    return {
        "trainable_parameter_count": count,
        "minimum": minimum,
        "maximum": maximum,
        "architecture_profile": spec.architecture_profile,
        "module_counts": model.parameter_counts_by_module(),
        "passed": True,
    }


def setflow_v5_learning_rate_factor(
    update_index: int,
    *,
    total_updates: int,
    warmup_fraction: float = 0.05,
) -> float:
    _require(total_updates > 1, "SetFlow V5 update budget is too small")
    _require(0 <= update_index < total_updates, "SetFlow V5 update index is outside budget")
    _require(0.0 < warmup_fraction < 1.0, "SetFlow V5 warmup fraction changed")
    warmup_updates = max(1, math.ceil(total_updates * warmup_fraction))
    if update_index < warmup_updates:
        return float(update_index + 1) / warmup_updates
    progress = float(update_index + 1 - warmup_updates) / float(
        total_updates - warmup_updates
    )
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def gate_b0_convergence_judgment(
    pass_rows: Sequence[Mapping[str, Any]],
    *,
    window: int = 2,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Gate B0: last two passes' train-loss relative drop < 5% (converged)."""

    _require(bool(pass_rows) and window >= 1, "SetFlow V5 convergence rows are empty")
    losses = [float(row["mean_train_total_loss"]) for row in pass_rows[-window:]]
    earlier = losses[0]
    latest = losses[-1]
    _require(math.isfinite(earlier) and earlier > 0.0, "SetFlow V5 early loss is invalid")
    relative_drop = (earlier - latest) / earlier
    converged = bool(relative_drop < tolerance)
    return {
        "gate": "B0",
        "convergence_window_passes": window,
        "window_train_losses": losses,
        "relative_drop_over_window": relative_drop,
        "tolerance_relative_drop": tolerance,
        "converged": converged,
        "rule": "LAST_2_PASS_RELATIVE_DROP_LT_5_PERCENT",
    }


def _require_source_token_preflight_identity_v5(
    config: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    from core.route2_source_token_cache_v3 import (
        require_source_token_cache_identity_receipt_v3,
    )

    require_source_token_cache_identity_receipt_v3(
        preflight.get("source_token_cache_identity"),
        expected_model_id="YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
        expected_record_count=84218,
        expected_unique_source_count=19303,
        expected_token_count=2817781,
        expected_maximum_source_length=837,
        expected_embedding_width=int(
            config["architecture"]["frozen_source_mrnabert_width"]
        ),
    )


def require_setflow_v5_screen_launch_authorization(
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    source_data_audit: Mapping[str, Any],
    *,
    run_id: str,
    current_git_head: str,
) -> None:
    spec = screen_run_spec_v5(config, run_id)
    frozen_run_ids = {str(row["run_id"]) for row in config["required_screen_runs"]}
    _require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v5_screen_launch_authorization.v1",
        "SetFlow V5 launch authorization schema is absent",
    )
    _require(
        authorization.get("status") == "XEDITSETFLOW_V5_SCREEN_LAUNCH_AUTHORIZED",
        "SetFlow V5 screen launch is not authorized",
    )
    _require(
        str(authorization.get("authorized_git_head")) == str(current_git_head),
        "SetFlow V5 authorization is for another Git HEAD",
    )
    _require(
        set(authorization.get("authorized_run_ids", [])) == frozen_run_ids,
        "SetFlow V5 authorization does not cover the exact frozen package",
    )
    _require(spec.run_id in frozen_run_ids, "SetFlow V5 requested run is not authorized")
    barriers = authorization.get("barriers", {})
    required_true = (
        "a100_current_head_focused_tests_passed",
        "source_token_cache_terminal_complete",
        "source_level_data_audit_passed",
        "formal_parameter_preflight_passed",
    )
    _require(
        all(barriers.get(key) is True for key in required_true),
        "a SetFlow V5 launch barrier is not satisfied",
    )
    _require(
        preflight.get("status") == "XEDITSETFLOW_V5_PREFLIGHT_PASS"
        and preflight.get("passed") is True,
        "formal SetFlow V5 preflight did not pass",
    )
    _require(
        source_data_audit.get("status") == "XEDITSETFLOW_V5_SOURCE_LEVEL_DATA_AUDIT_PASS",
        "SetFlow V5 source-level data audit did not pass",
    )
    _require(
        int(source_data_audit.get("validation_source_count", -1))
        == int(config["data_geometry"]["expected_validation_source_record_count"]),
        "SetFlow V5 Validation source-record inventory changed",
    )
    for payload, name in (
        (authorization, "authorization"),
        (preflight, "preflight"),
        (source_data_audit, "source data audit"),
    ):
        _require(
            int(payload.get("development_test_outcome_reads", -1)) == 0,
            f"SetFlow V5 {name} reports a Development TEST read",
        )
        _require(
            int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
            f"SetFlow V5 {name} reports a new Evaluation read",
        )
