from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.route_a_v3.train_route2_xeditcritic_v4 import (
    XEditCriticTrainingV4RunnerError,
    _write_atomic_terminal_v4,
    critic_v4_run_stage_seed,
    evaluation_index_batches_v4,
    require_confirmation_launch_authorization_v4,
    require_posttest_launch_authorization_v4,
    require_screen_launch_authorization_v4,
    screen_run_spec_v4,
    split_posttest_records_v4,
    posttest_selection_policy_v4,
)
from core.route2_xeditcritic_training_data_v3 import XEditCriticRecordV3


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_critic_training_terminal_artifact_is_atomic_and_exact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run_summary.json"
    payload = {"status": "TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE"}
    _write_atomic_terminal_v4(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not output.with_suffix(output.suffix + ".partial").exists()

    stale = tmp_path / "failure.json.partial"
    stale.write_text("interrupted", encoding="utf-8")
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="partial terminal"):
        _write_atomic_terminal_v4(tmp_path / "failure.json", payload)
    source = (ROOT / "scripts/route_a_v3/train_route2_xeditcritic_v4.py").read_text()
    assert 'if not (output_directory / "run_summary.json").exists()' in source


def _authorization(config: dict) -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1",
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": "head",
        "preflight_runner_git_head": "head",
        "authorized_run_ids": [
            row["run_id"] for row in config["required_screen_runs"]
        ],
        "barriers": {
            "all_five_c3_jobs_terminal": True,
            "c3_terminal_summaries_read_exactly_once": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            "bottom_six_cache_terminal_complete": True,
            "formal_parameter_preflight_passed": True,
            "formal_memory_preflight_passed": True,
            "cache_online_equivalence_passed": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _preflight() -> dict:
    return {
        "status": "XEDITCRITIC_V4_PREFLIGHT_PASS",
        "passed": True,
        "git_head": "head",
        "selected_physical_batch": 8,
        "trainable_parameter_count": 170_481_733,
        "selected_peak_allocated_gib": 29.0,
        "bottom_six_cache_identity": {
            "model_id": "YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
            "record_count": 107873,
            "unique_sequence_count": 43730,
            "embedding_width": 768,
            "frozen_encoder_blocks": [0, 1, 2, 3, 4, 5],
            "trainable_encoder_blocks": [6, 7, 8, 9, 10, 11],
            "chunk_length": 1000,
            "chunk_overlap": 64,
            "local_context_radius": 32,
            "special_token_offset": 1,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_all_training_stages_bind_frozen_physical_gpu_scope_before_cuda() -> None:
    source = (
        ROOT / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
    ).read_text(encoding="utf-8")
    assert "require_physical_gpu_scope_v4(config, physical_gpu_index)" in source
    assert source.index(
        "require_physical_gpu_scope_v4(config, physical_gpu_index)"
    ) < source.index("device = require_cuda(physical_gpu_index)")


def test_screen_run_specs_bind_all_exact_frozen_controls_and_ablations() -> None:
    config = _config()
    permutation = screen_run_spec_v4(config, "v4_candidate_bundle_permutation")
    assert permutation.control_mode == "CANDIDATE_BUNDLE_PERMUTATION"
    assert permutation.candidate_bundle_permutation is True
    assert permutation.selectable is False
    assert screen_run_spec_v4(config, "v4_no_cross").mechanism_mode == "NO_CROSS"
    assert screen_run_spec_v4(config, "v4_full").selectable is True
    assert screen_run_spec_v4(config, "c0_v4").model_kind == "C0-V4"
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="exact frozen"):
        screen_run_spec_v4(config, "v4_unregistered")


def test_validation_batch_padding_does_not_add_measured_rows() -> None:
    batches = evaluation_index_batches_v4(18, 8)
    assert [valid for _, valid in batches] == [8, 8, 2]
    assert all(len(indices) == 8 for indices, _ in batches)
    assert sum(valid for _, valid in batches) == 18
    assert batches[-1][0][:2] == [16, 17]
    assert batches[-1][0][2:] == [0, 1, 2, 3, 4, 5]


def test_launch_authorization_requires_every_c3_sync_cache_and_preflight_barrier() -> None:
    config = _config()
    authorization = _authorization(config)
    require_screen_launch_authorization_v4(
        config,
        authorization,
        _preflight(),
        run_id="v4_full",
        physical_batch_size=8,
        current_git_head="head",
    )
    authorization["barriers"]["c3_terminal_summaries_read_exactly_once"] = False
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="barrier"):
        require_screen_launch_authorization_v4(
            config,
            authorization,
            _preflight(),
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="head",
        )


def test_launch_authorization_accepts_low_positive_peak_but_rejects_upper_memory_drift() -> None:
    config = _config()
    authorization = _authorization(config)
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="another Git HEAD"):
        require_screen_launch_authorization_v4(
            config,
            authorization,
            _preflight(),
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="different",
        )
    preflight = _preflight()
    preflight["selected_peak_allocated_gib"] = 19.9
    require_screen_launch_authorization_v4(
        config,
        authorization,
        preflight,
        run_id="v4_full",
        physical_batch_size=8,
        current_git_head="head",
    )
    preflight["selected_peak_allocated_gib"] = 35.1
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="above 35"):
        require_screen_launch_authorization_v4(
            config,
            authorization,
            preflight,
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="head",
        )
    preflight = _preflight()
    preflight["development_test_outcome_reads"] = 1
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="TEST read"):
        require_screen_launch_authorization_v4(
            config,
            authorization,
            preflight,
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="head",
        )

    preflight = _preflight()
    del preflight["bottom_six_cache_identity"]
    with pytest.raises(Exception, match="cache identity receipt is absent"):
        require_screen_launch_authorization_v4(
            config,
            authorization,
            preflight,
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="head",
        )


def test_confirmation_authorization_is_exact_three_seed_full_c0_scope() -> None:
    config = _config()
    config.update(
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_confirmation_runtime.v1",
            "run_stage": "CONFIRMATION",
            "training_seed": 20260908,
            "required_confirmation_run_ids": ["v4_full", "c0_v4"],
        }
    )
    authorization = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_confirmation_launch_authorization.v1",
        "status": "XEDITCRITIC_V4_CONFIRMATION_LAUNCH_AUTHORIZED",
        "authorized_git_head": "head",
        "authorized_seeds": [20260908, 20260909, 20260910],
        "authorized_run_ids": ["v4_full", "c0_v4"],
        "barriers": {
            "screen_gate_passed": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            "bottom_six_cache_terminal_complete": True,
            "formal_parameter_preflight_passed": True,
            "formal_memory_preflight_passed": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    screen_gate = {
        "status": "XEDITCRITIC_V4_SCREEN_PASS",
        "passed": True,
        "confirmation_authorized": True,
        "development_test_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    assert critic_v4_run_stage_seed(config, "v4_full") == (
        "CONFIRMATION",
        20260908,
    )
    require_confirmation_launch_authorization_v4(
        config,
        authorization,
        _preflight(),
        screen_gate,
        run_id="v4_full",
        physical_batch_size=8,
        current_git_head="head",
    )
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="scope"):
        critic_v4_run_stage_seed(config, "v4_no_moe")
    preflight = _preflight()
    preflight["development_test_outcome_reads"] = 1
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="protected read"):
        require_confirmation_launch_authorization_v4(
            config,
            authorization,
            preflight,
            screen_gate,
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="head",
        )


def _record(record_id: str, study: str, split: str) -> XEditCriticRecordV3:
    return XEditCriticRecordV3(
        record_id=record_id,
        split=split,
        source="AAAA",
        candidate="ACAA",
        edits=((1, "A", "C"),),
        target=1.0,
        task="task",
        study=study,
        source_group=f"source::{record_id}",
        assay="assay",
        context="context",
        region=0,
        quantity="quantity",
        measurement="measurement",
        numerator="numerator",
        denominator="denominator",
    )


def test_posttest_record_splits_refit_all_and_loso_by_study() -> None:
    records = [
        _record("a", "GSE200304", "TRAIN"),
        _record("b", "GSE269595", "VALIDATION"),
        _record("c", "GSE269595", "TRAIN"),
    ]
    train, validation = split_posttest_records_v4(
        records, run_stage="REFIT", held_out_study=None
    )
    assert len(train) == 3 and validation == []
    assert {record.split for record in train} == {"TRAIN"}
    train, validation = split_posttest_records_v4(
        records, run_stage="LOSO", held_out_study="GSE269595"
    )
    assert [record.record_id for record in train] == ["a"]
    assert [record.record_id for record in validation] == ["b", "c"]
    assert {record.split for record in validation} == {"VALIDATION"}
    assert (
        posttest_selection_policy_v4("LOSO")
        == "FINAL_PASS_8_FIXED_NO_TEST_OR_VALIDATION_SELECTION"
    )


def test_posttest_launch_authorization_requires_atomic_test_and_exact_scope(
    tmp_path: Path,
) -> None:
    config = _config()
    config.update(
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_runtime.v1",
            "run_stage": "REFIT",
            "training_seed": 20260908,
            "required_posttest_run_ids": ["v4_full"],
            "held_out_study": None,
        }
    )
    authorization = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_launch_authorization.v1",
        "status": "XEDITCRITIC_V4_REFIT_LAUNCH_AUTHORIZED",
        "authorized_stage": "REFIT",
        "authorized_git_head": "head",
        "authorized_seeds": [20260908, 20260909, 20260910],
        "authorized_run_ids": ["v4_full"],
        "authorized_held_out_studies": [],
        "atomic_frozen_test_passed": True,
        "all_three_refits_complete": False,
        "development_test_access_event_count_before_posttest": 1,
        "development_test_outcome_reads_during_posttest": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    three = {
        "status": "XEDITCRITIC_V4_THREE_SEED_PASS",
        "required_seeds": [20260908, 20260909, 20260910],
        "development_test_authorized": True,
        "atomic_development_test_only": True,
    }
    receipt = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_authorization_receipt.v1",
        "status": "XEDITCRITIC_V4_POSTTEST_AUTHORIZED",
        "required_seeds": [20260908, 20260909, 20260910],
        "frozen_test_gate_status": "XEDITCRITIC_V4_FROZEN_TEST_PASS",
        "all_development_refit_authorized": True,
        "development_test_access_event_count": 1,
        "general_test_projection_persisted": False,
        "test_bottom_six_cache_persisted": False,
        "development_test_metrics_in_receipt": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    require_posttest_launch_authorization_v4(
        config,
        authorization,
        _preflight(),
        three,
        receipt,
        run_id="v4_full",
        physical_batch_size=8,
        current_git_head="head",
    )
    receipt["test_bottom_six_cache_persisted"] = True
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="receipt"):
        require_posttest_launch_authorization_v4(
            config,
            authorization,
            _preflight(),
            three,
            receipt,
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="head",
        )
