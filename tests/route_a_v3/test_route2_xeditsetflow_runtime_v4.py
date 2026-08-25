from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from core.route2_xeditsetflow_runtime_v4 import (
    XEditSetFlowRuntimeV4Error,
    build_setflow_screen_model_v4,
    pad_source_batches_v4,
    require_setflow_v4_confirmation_launch_authorization,
    require_setflow_v4_screen_launch_authorization,
    screen_run_spec_v4,
    setflow_v4_learning_rate_factor,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _vocabs() -> dict[str, dict[str, int]]:
    sizes = _config()["architecture"]["formal_endpoint_vocab_cardinalities"]
    return {
        field: {f"value_{index}": index for index in range(size)}
        for field, size in sizes.items()
    }


def _source_token_cache_receipt() -> dict:
    return {
        "model_id": "YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
        "record_count": 84218,
        "unique_source_count": 19303,
        "token_count": 2817781,
        "maximum_source_length": 837,
        "embedding_width": 768,
        "tokenization_policy": "UTR_SINGLE_NUCLEOTIDE_SPACE_SEPARATED_DNA_ALPHABET_ONE_LEADING_SPECIAL",
        "chunk_policy": "ONE_COMPLETE_CHUNK_MAXIMUM_1000_NUCLEOTIDES",
    }


def test_screen_specs_and_formal_models_use_exact_real_vocab_capacity() -> None:
    config = _config()
    full_spec = screen_run_spec_v4(config, "v4_full")
    single_spec = screen_run_spec_v4(config, "v4_single_mode")
    assert (full_spec.mode_count, full_spec.selectable) == (8, True)
    assert (single_spec.mode_count, single_spec.selectable) == (1, False)
    full, full_capacity = build_setflow_screen_model_v4(
        config, _vocabs(), run_id="v4_full"
    )
    assert full_capacity["trainable_parameter_count"] == 100_099_998
    del full
    single, single_capacity = build_setflow_screen_model_v4(
        config, _vocabs(), run_id="v4_single_mode"
    )
    assert single_capacity["trainable_parameter_count"] == 98_628_717


def test_source_batch_fill_makes_exact_32_state_batches_with_repeat_cap() -> None:
    padded = pad_source_batches_v4(
        [list(range(8)), list(range(8, 13))], source_count=13
    )
    assert all(len(batch) == 8 for batch in padded)
    counts = {index: sum(index in batch for batch in padded) for index in range(13)}
    assert max(counts.values()) <= 4
    assert len(padded) * 8 * 4 == 64


def test_learning_rate_is_warmup_then_cosine_to_exact_ten_percent() -> None:
    values = [
        setflow_v4_learning_rate_factor(index, total_updates=200)
        for index in range(200)
    ]
    assert values[0] == pytest.approx(0.1)
    assert values[9] == pytest.approx(1.0)
    assert values[10] < 1.0
    assert values[-1] == pytest.approx(0.1)
    assert all(math.isfinite(value) and value > 0 for value in values)


def _authorization(*, head: str = "abc") -> tuple[dict, dict, dict]:
    authorization = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_screen_launch_authorization.v1",
        "status": "XEDITSETFLOW_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": head,
        "authorized_run_ids": ["v4_full", "v4_single_mode"],
        "barriers": {
            "all_five_c3_jobs_terminal": True,
            "c3_terminal_summaries_read_exactly_once": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            "source_token_cache_terminal_complete": True,
            "source_level_data_audit_passed": True,
            "formal_parameter_preflight_passed": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    preflight = {
        "status": "XEDITSETFLOW_V4_PREFLIGHT_PASS",
        "passed": True,
        "full_trainable_parameter_count": 100_099_998,
        "single_mode_trainable_parameter_count": 98_628_717,
        "source_token_cache_identity": _source_token_cache_receipt(),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    data = {
        "status": "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS",
        "train_source_count": 100,
        "validation_source_count": 15_327,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    return authorization, preflight, data


def test_launch_authorization_requires_every_barrier_same_head_and_zero_reads() -> None:
    config = _config()
    authorization, preflight, data = _authorization()
    require_setflow_v4_screen_launch_authorization(
        config,
        authorization,
        preflight,
        data,
        run_id="v4_full",
        current_git_head="abc",
    )
    authorization["barriers"]["all_five_c3_jobs_terminal"] = False
    with pytest.raises(XEditSetFlowRuntimeV4Error):
        require_setflow_v4_screen_launch_authorization(
            config,
            authorization,
            preflight,
            data,
            run_id="v4_full",
            current_git_head="abc",
        )


def test_launch_authorization_rejects_protected_read_before_any_run() -> None:
    config = _config()
    authorization, preflight, data = _authorization()
    data["development_test_outcome_reads"] = 1
    with pytest.raises(XEditSetFlowRuntimeV4Error):
        require_setflow_v4_screen_launch_authorization(
            config,
            authorization,
            preflight,
            data,
            run_id="v4_single_mode",
            current_git_head="abc",
        )


def test_launch_authorization_rejects_missing_tensor_cache_identity_receipt() -> None:
    config = _config()
    authorization, preflight, data = _authorization()
    del preflight["source_token_cache_identity"]
    with pytest.raises(Exception, match="cache identity receipt is absent"):
        require_setflow_v4_screen_launch_authorization(
            config,
            authorization,
            preflight,
            data,
            run_id="v4_full",
            current_git_head="abc",
        )


def test_confirmation_authorization_is_full_only_three_seed_and_screen_gated() -> None:
    config = {
        **_config(),
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_confirmation_runtime.v1",
        "run_stage": "CONFIRMATION",
        "training_seed": 20260912,
        "selected_model": "v4_full",
    }
    authorization = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_confirmation_launch_authorization.v1",
        "status": "XEDITSETFLOW_V4_CONFIRMATION_LAUNCH_AUTHORIZED",
        "authorized_git_head": "abc",
        "authorized_seeds": [20260912, 20260913, 20260914],
        "authorized_run_id": "v4_full",
        "barriers": {
            "screen_gate_passed": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            "source_token_cache_terminal_complete": True,
            "source_level_data_audit_passed": True,
            "formal_parameter_preflight_passed": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    preflight = {
        "status": "XEDITSETFLOW_V4_PREFLIGHT_PASS",
        "passed": True,
        "full_trainable_parameter_count": 100_099_998,
        "source_token_cache_identity": _source_token_cache_receipt(),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    data = {
        "status": "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS",
            "validation_source_count": 15_327,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    gate = {
        "status": "XEDITSETFLOW_V4_SCREEN_PASS",
        "confirmation_authorized": True,
        "confirmation_seeds": [20260912, 20260913, 20260914],
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    require_setflow_v4_confirmation_launch_authorization(
        config,
        authorization,
        preflight,
        data,
        gate,
        run_id="v4_full",
        current_git_head="abc",
    )
    with pytest.raises(XEditSetFlowRuntimeV4Error):
        require_setflow_v4_confirmation_launch_authorization(
            config,
            authorization,
            preflight,
            data,
            gate,
            run_id="v4_single_mode",
            current_git_head="abc",
        )
