"""Frozen screen runtime helpers for XEditSetFlow V4."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.route2_source_token_cache_v3 import (
    require_source_token_cache_identity_receipt_v3,
)
from core.route2_xeditsetflow_v4 import (
    XEditSetFlowV4,
    require_setflow_v4_trainable_parameter_range,
)


class XEditSetFlowRuntimeV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowRuntimeV4Error(message)


@dataclass(frozen=True)
class SetFlowScreenRunSpecV4:
    run_id: str
    mode_count: int
    mode_information_weight: float
    selectable: bool


def screen_run_spec_v4(
    config: Mapping[str, Any], run_id: str
) -> SetFlowScreenRunSpecV4:
    matches = [
        row
        for row in config["required_screen_runs"]
        if str(row["run_id"]) == str(run_id)
    ]
    _require(len(matches) == 1, "run id is not one frozen SetFlow V4 screen run")
    row = matches[0]
    spec = SetFlowScreenRunSpecV4(
        run_id=str(row["run_id"]),
        mode_count=int(row["mode_count"]),
        mode_information_weight=float(row["mode_information_weight"]),
        selectable=bool(row["selectable"]),
    )
    _require(
        (spec.run_id, spec.mode_count, spec.mode_information_weight, spec.selectable)
        in {
            ("v4_full", 8, 0.05, True),
            ("v4_single_mode", 1, 0.0, False),
        },
        "SetFlow V4 screen role changed",
    )
    return spec


def build_setflow_screen_model_v4(
    config: Mapping[str, Any],
    vocabs: Mapping[str, Mapping[str, int]],
    *,
    run_id: str,
) -> tuple[XEditSetFlowV4, dict[str, Any]]:
    spec = screen_run_spec_v4(config, run_id)
    architecture = config["architecture"]
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
            f"SetFlow V4 frozen {field} vocabulary cardinality changed",
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
    capacity = require_setflow_v4_trainable_parameter_range(model)
    expected_count = int(
        architecture[
            "formal_full_trainable_parameter_count"
            if spec.mode_count == 8
            else "formal_single_mode_trainable_parameter_count"
        ]
    )
    _require(
        int(capacity["trainable_parameter_count"]) == expected_count,
        "SetFlow V4 exact formal parameter count changed",
    )
    return model, capacity


def pad_source_batches_v4(
    batches: Sequence[Sequence[int]],
    *,
    source_count: int,
    sources_per_batch: int = 8,
    repeat_cap: int = 4,
) -> list[list[int]]:
    """Deterministically fill only short source batches to 8 sources / 32 states."""

    _require(source_count >= sources_per_batch, "SetFlow V4 has fewer than eight TRAIN sources")
    _require(bool(batches), "SetFlow V4 source sampler emitted no batch")
    flattened = [int(index) for batch in batches for index in batch]
    _require(
        all(0 <= index < source_count for index in flattened),
        "SetFlow V4 source sampler emitted an invalid index",
    )
    counts = Counter(flattened)
    _require(max(counts.values(), default=0) <= repeat_cap, "SetFlow V4 source repeat cap was already exceeded")
    result: list[list[int]] = []
    cursor = 0
    for raw in batches:
        batch = [int(index) for index in raw]
        _require(0 < len(batch) <= sources_per_batch, "SetFlow V4 source batch geometry changed")
        while len(batch) < sources_per_batch:
            selected: int | None = None
            for prefer_distinct in (True, False):
                for _ in range(len(flattened)):
                    candidate = flattened[cursor % len(flattened)]
                    cursor += 1
                    if counts[candidate] >= repeat_cap:
                        continue
                    if prefer_distinct and candidate in batch:
                        continue
                    selected = candidate
                    break
                if selected is not None:
                    break
            _require(selected is not None, "SetFlow V4 cannot fill final batch within repeat cap")
            batch.append(selected)
            counts[selected] += 1
        result.append(batch)
    _require(all(len(batch) == sources_per_batch for batch in result), "SetFlow V4 physical batch is not 32 states")
    _require(max(counts.values()) <= repeat_cap, "SetFlow V4 source repeat cap was exceeded during final fill")
    return result


def setflow_v4_learning_rate_factor(
    update_index: int,
    *,
    total_updates: int,
    warmup_fraction: float = 0.05,
) -> float:
    _require(total_updates > 1, "SetFlow V4 update budget is too small")
    _require(0 <= update_index < total_updates, "SetFlow V4 update index is outside budget")
    _require(0.0 < warmup_fraction < 1.0, "SetFlow V4 warmup fraction changed")
    warmup_updates = max(1, math.ceil(total_updates * warmup_fraction))
    if update_index < warmup_updates:
        return float(update_index + 1) / warmup_updates
    progress = float(update_index + 1 - warmup_updates) / float(
        total_updates - warmup_updates
    )
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _require_source_token_preflight_identity_v4(
    config: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
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


def require_setflow_v4_screen_launch_authorization(
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    source_data_audit: Mapping[str, Any],
    *,
    run_id: str,
    current_git_head: str,
) -> None:
    spec = screen_run_spec_v4(config, run_id)
    frozen_run_ids = {str(row["run_id"]) for row in config["required_screen_runs"]}
    _require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_screen_launch_authorization.v1",
        "SetFlow V4 launch authorization schema is absent",
    )
    _require(
        authorization.get("status") == "XEDITSETFLOW_V4_SCREEN_LAUNCH_AUTHORIZED",
        "SetFlow V4 screen launch is not authorized",
    )
    _require(
        str(authorization.get("authorized_git_head")) == str(current_git_head),
        "SetFlow V4 authorization is for another Git HEAD",
    )
    _require(
        str(authorization.get("preflight_runner_git_head"))
        == str(preflight.get("git_head")),
        "SetFlow V4 authorization is bound to another preflight runner HEAD",
    )
    _require(
        set(authorization.get("authorized_run_ids", [])) == frozen_run_ids,
        "SetFlow V4 authorization does not cover the exact frozen package",
    )
    _require(spec.run_id in frozen_run_ids, "SetFlow V4 requested run is not authorized")
    barriers = authorization.get("barriers", {})
    required_true = (
        "all_five_c3_jobs_terminal",
        "c3_terminal_summaries_read_exactly_once",
        "a100_current_head_focused_tests_passed",
        "a100_current_head_v332_tests_passed",
        "source_token_cache_terminal_complete",
        "source_level_data_audit_passed",
        "formal_parameter_preflight_passed",
    )
    _require(
        all(barriers.get(key) is True for key in required_true),
        "a SetFlow V4 launch barrier is not satisfied",
    )
    _require(
        preflight.get("status") == "XEDITSETFLOW_V4_PREFLIGHT_PASS"
        and preflight.get("passed") is True,
        "formal SetFlow V4 preflight did not pass",
    )
    _require(
        int(preflight.get("full_trainable_parameter_count", -1))
        == int(config["architecture"]["formal_full_trainable_parameter_count"]),
        "SetFlow V4 full preflight count changed",
    )
    _require(
        int(preflight.get("single_mode_trainable_parameter_count", -1))
        == int(config["architecture"]["formal_single_mode_trainable_parameter_count"]),
        "SetFlow V4 single-mode preflight count changed",
    )
    _require_source_token_preflight_identity_v4(config, preflight)
    _require(
        source_data_audit.get("status") == "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS",
        "SetFlow V4 source-level data audit did not pass",
    )
    _require(
        int(source_data_audit.get("validation_source_count", -1))
        == int(config["data_geometry"]["expected_validation_source_record_count"]),
        "SetFlow V4 Validation source-record inventory changed",
    )
    _require(
        int(source_data_audit.get("train_source_count", 0)) >= 8,
        "SetFlow V4 TRAIN source cohort is too small",
    )
    for payload, name in (
        (authorization, "authorization"),
        (preflight, "preflight"),
        (source_data_audit, "source data audit"),
    ):
        _require(
            int(payload.get("development_test_outcome_reads", -1)) == 0,
            f"SetFlow V4 {name} reports a Development TEST read",
        )
        _require(
            int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
            f"SetFlow V4 {name} reports a new Evaluation read",
        )


def require_setflow_v4_confirmation_launch_authorization(
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    source_data_audit: Mapping[str, Any],
    screen_gate: Mapping[str, Any],
    *,
    run_id: str,
    current_git_head: str,
) -> None:
    seed = config.get("training_seed")
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_runtime.v1"
        and config.get("run_stage") == "CONFIRMATION",
        "SetFlow V4 confirmation runtime config identity changed",
    )
    _require(
        isinstance(seed, int)
        and not isinstance(seed, bool)
        and seed in {20260912, 20260913, 20260914},
        "SetFlow V4 confirmation seed is undeclared",
    )
    _require(
        run_id == config.get("selected_model") == "v4_full",
        "SetFlow V4 confirmation attempted a non-full model",
    )
    _require(
        screen_gate.get("status") == "XEDITSETFLOW_V4_SCREEN_PASS"
        and screen_gate.get("confirmation_authorized") is True
        and screen_gate.get("confirmation_seeds")
        == [20260912, 20260913, 20260914],
        "SetFlow V4 screen gate does not authorize confirmation",
    )
    _require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_launch_authorization.v1"
        and authorization.get("status")
        == "XEDITSETFLOW_V4_CONFIRMATION_LAUNCH_AUTHORIZED",
        "SetFlow V4 confirmation authorization is absent",
    )
    _require(
        str(authorization.get("authorized_git_head")) == str(current_git_head)
        and authorization.get("authorized_seeds")
        == [20260912, 20260913, 20260914]
        and authorization.get("authorized_run_id") == "v4_full",
        "SetFlow V4 confirmation authorization scope changed",
    )
    barriers = authorization.get("barriers", {})
    required = (
        "screen_gate_passed",
        "a100_current_head_focused_tests_passed",
        "a100_current_head_v332_tests_passed",
        "source_token_cache_terminal_complete",
        "source_level_data_audit_passed",
        "formal_parameter_preflight_passed",
    )
    _require(
        all(barriers.get(key) is True for key in required),
        "a SetFlow V4 confirmation barrier is not satisfied",
    )
    _require(
        preflight.get("status") == "XEDITSETFLOW_V4_PREFLIGHT_PASS"
        and preflight.get("passed") is True
        and int(preflight.get("full_trainable_parameter_count", -1))
        == int(config["architecture"]["formal_full_trainable_parameter_count"]),
        "SetFlow V4 confirmation preflight identity changed",
    )
    _require_source_token_preflight_identity_v4(config, preflight)
    _require(
        source_data_audit.get("status")
        == "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS"
        and int(source_data_audit.get("validation_source_count", -1))
        == int(config["data_geometry"]["expected_validation_source_record_count"]),
        "SetFlow V4 confirmation source-data identity changed",
    )
    for payload, name in (
        (authorization, "authorization"),
        (preflight, "preflight"),
        (source_data_audit, "source data audit"),
        (screen_gate, "screen gate"),
    ):
        _require(
            int(payload.get("development_test_outcome_reads", -1)) == 0
            and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
            f"SetFlow V4 confirmation {name} reports a protected read",
        )
