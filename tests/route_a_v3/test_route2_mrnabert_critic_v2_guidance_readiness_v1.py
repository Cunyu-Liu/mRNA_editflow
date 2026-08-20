from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILDER_SCRIPT = (
    ROOT
    / "scripts/route_a_v3/build_route2_mrnabert_critic_v2_guidance_readiness_input_v1.py"
)
ADJUDICATOR_SCRIPT = (
    ROOT
    / "scripts/route_a_v3/adjudicate_route2_mrnabert_critic_v2_readiness_v1.py"
)
GUIDED_SCRIPT = ROOT / "scripts/route_a_v3/run_route2_guided_xeditflow_v1.py"
MATCHED_SCRIPT = (
    ROOT / "scripts/route_a_v3/run_route2_mrnabert_matched_search_suite_v1.py"
)
COMPARISON_SCRIPT = (
    ROOT
    / "scripts/route_a_v3/run_route2_mrnabert_generation_comparison_suite_v1.py"
)
PROTOCOL_PATHS = {
    "readiness": ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_protocol_v1.json",
    "control": ROOT / "configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json",
    "three_seed": ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol_v1.json",
    "frozen_test": ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol_v1.json",
    "refit": ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol_v1.json",
    "primary_loso": ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json",
    "baseline_loso": ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol_v1.json",
    "loso_aggregation": ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_loso_aggregation_protocol_v1.json",
}
REWARD_POLICY = ROOT / "configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json"
STUDIES = (
    "GSE200304",
    "GSE114002",
    "GSE149487",
    "GSE217518",
    "ENCSR854RUF",
    "GSE186455",
    "GSE269595",
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modules():
    return (
        _load(BUILDER_SCRIPT, "critic_v2_readiness_builder_test"),
        _load(ADJUDICATOR_SCRIPT, "critic_v2_readiness_adjudicator_test"),
    )


def _protocols() -> dict[str, dict]:
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in PROTOCOL_PATHS.items()
    }


def _control_adjudication() -> dict:
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_control_adjudication.v1",
        "status": "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS",
        "supports_three_frozen_seeds": True,
        "checks": {
            "full_beats_strongest_same_information_baseline": True,
            "full_beats_source_only_macro": True,
            "full_beats_source_edit_metadata_macro": True,
            "full_beats_permutation_on_supported_tasks": True,
        },
        "frozen_confirmation_seeds": [20260822, 20260823, 20260824],
        "development_test_opened": False,
        "evaluation_opened": False,
        "guided_generation_authorized": False,
    }


def _three_seed_adjudication() -> dict:
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_three_seed_adjudication.v1",
        "status": "CRITIC_V2_THREE_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST",
        "supports_single_frozen_development_test": True,
        "checks": {
            "control_adjudication_supports_three_frozen_seeds": True,
            "all_seed_metrics_finite": True,
            "all_seed_prediction_spreads_positive": True,
            "all_seed_task_macros_replay": True,
            "all_seed_spread_ratios_replay": True,
            "all_three_seed_margins_over_strongest_baseline_positive": True,
        },
        "seed_results": [
            {
                "seed": seed,
                "margin_over_strongest_same_information_baseline": 0.01,
                "nonfinite_metric_detected": False,
                "mean_collapse_detected": False,
            }
            for seed in (20260822, 20260823, 20260824)
        ],
        "development_test_opened": False,
        "evaluation_opened": False,
        "guided_generation_authorized": False,
    }


def _test_config(protocols: dict[str, dict]) -> dict:
    config = dict(protocols["frozen_test"]["frozen_training_policy"])
    config.update(
        {
            "scientific_role": "CRITIC_V2_SINGLE_FROZEN_DEVELOPMENT_TEST",
            "result_stage": "FROZEN_DEVELOPMENT_TEST",
            "seed": 20260823,
            "validation_checkpoint_selection_before_test": "BEST_VALIDATION",
            "checkpoint_selection": "FINAL_EPOCH",
            "test_used_for_checkpoint_selection": False,
            "test_used_for_model_or_policy_selection": False,
            "evaluation_outcomes_accessed": False,
        }
    )
    return config


def _cuda_fields(gpu: int) -> dict:
    return {
        "device": f"cuda:{gpu}",
        "physical_gpu_index": gpu,
        "cpu_fallback_used": False,
        "cuda_training_tensors_verified": True,
        "cuda_device_index": gpu,
        "cuda_device_uuid": f"GPU-{gpu}",
        "cuda_total_memory_mb": 40960.0,
    }


def _test_summary() -> dict:
    return {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "result_stage": "FROZEN_DEVELOPMENT_TEST",
        "seed": 20260823,
        "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "candidate_control": "NONE",
        "checkpoint_selection": "FINAL_EPOCH",
        "selected_epoch": 100,
        "final_training_epoch": 100,
        "development_test_outcomes_evaluated": True,
        "development_validation_folded_into_training": True,
        "record_counts": {"TRAIN": 107873, "TEST": 18292},
        "test_metrics": {"task_macro_spearman": -0.8},
        "evaluation_outcomes_read": 0,
        "optimizer_steps": 674200,
        "parameter_changed": True,
        **_cuda_fields(2),
    }


def _refit_config(protocols: dict[str, dict]) -> dict:
    policy = protocols["refit"]["frozen_model_training_policy"]
    config = {key: value for key, value in policy.items() if key != "checkpoint_selection_before_test"}
    config.update(
        {
            "scientific_role": "CRITIC_V2_FINAL_ALL_DEVELOPMENT_REFIT",
            "result_stage": "FINAL_ALL_DEVELOPMENT_REFIT",
            "development_record_scope": "ALL_126165",
            "checkpoint_selection": "FINAL_EPOCH",
            "refit_model_selection_performed": False,
            "test_metrics_used_for_refit_selection": False,
            "evaluation_outcomes_accessed": False,
        }
    )
    return config


def _refit_summary() -> dict:
    return {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "result_stage": "FINAL_ALL_DEVELOPMENT_REFIT",
        "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "candidate_control": "NONE",
        "checkpoint_selection": "FINAL_EPOCH",
        "selected_epoch": 100,
        "final_training_epoch": 100,
        "record_counts": {"TRAIN": 126165},
        "development_validation_folded_into_training": True,
        "development_test_record_count_withheld": 0,
        "test_metrics": None,
        "evaluation_outcomes_read": 0,
        "optimizer_steps": 788600,
        "parameter_changed": True,
        **_cuda_fields(3),
    }


def _loso_results() -> list[dict]:
    return [
        {
            "schema_version": "route_a_v3_route2_loso_aggregation.v1",
            "status": "LOSO_MODEL_BASELINE_ALIGNED_COMPLETE",
            "seed": seed,
            "study_count": 7,
            "aligned_study_count": 7,
            "undefined_study_count": 0,
            "development_inventory_study_count": 8,
            "zero_record_development_studies": ["GSE256185"],
            "model_macro_spearman": 0.2,
            "baseline_macro_spearman": 0.1,
            "macro_improvement": 0.1,
            "per_study": [
                {
                    "study_unit_id": study,
                    "model_task_macro_spearman": 0.2,
                    "baseline_task_macro_spearman": 0.1,
                    "improvement": 0.1,
                    "failure_reasons": [],
                }
                for study in STUDIES
            ],
            "failure_reasons": [],
            "all_model_training_gpu_provenance_verified": True,
            "development_test_preserved": True,
            "evaluation_studies_included": 0,
        }
        for seed in (20260822, 20260823, 20260824)
    ]


def _online_encoder() -> dict:
    return {
        "schema_version": "route_a_v3_route2_mrnabert_online_encoder_validation.v1",
        "status": "ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE",
        "novel_candidate_encoding_supported": True,
        "frozen_parameter_count": 113389056,
        "evaluation_records_read": 0,
        "maximum_absolute_difference": 0.004,
        "absolute_tolerance": 0.01,
    }


def _flow_training() -> dict:
    return {
        "status": "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE",
        "optimizer_steps": 100,
        "seed": 11,
        "parameter_changed": True,
        "torch_device": "cuda:0",
        "physical_gpu_index": 0,
        "cpu_fallback_used": False,
        "cuda_training_tensors_verified": True,
        "cuda_device_index": 0,
        "cuda_device_uuid": "GPU-flow-train",
        "cuda_total_memory_mb": 40960.0,
        "evaluation_records_read": 0,
        "biological_optimization_established": False,
    }


def _flow_validation() -> dict:
    return {
        "hard_legality_rate": 1.0,
        "edit_budget_violation_count": 0,
        "trajectory_replay_failure_count": 0,
        "distinguishable_terminal_causes": [
            "EXPLICIT_STOP",
            "BUDGET_EXHAUSTED",
            "NO_LEGAL_ACTION",
            "NUMERICAL_FAILURE",
        ],
        "small_graph_reference": {
            "status": "PASS",
            "total_variation": 0.0,
            "tolerance": 1e-12,
        },
        "device": "cuda:1",
        "physical_gpu_index": 1,
        "cpu_fallback_used": False,
        "cuda_device_index": 1,
        "cuda_device_uuid": "GPU-flow-validation",
        "cuda_total_memory_mb": 40960.0,
        "trajectory_sampling_device": "cuda:1",
        "checkpoint_gpu_parameter_update_provenance_verified": True,
        "checkpoint_training_device": "cuda:0",
        "checkpoint_training_physical_gpu_index": 0,
        "checkpoint_cpu_fallback_used": False,
        "checkpoint_training_seed": 11,
        "checkpoint_training_optimizer_steps": 100,
        "checkpoint_parameter_changed": True,
        "checkpoint_cuda_training_tensors_verified": True,
        "checkpoint_training_cuda_device_index": 0,
        "checkpoint_training_cuda_device_uuid": "GPU-flow-train",
        "checkpoint_training_cuda_total_memory_mb": 40960.0,
        "evaluation_outcomes_read": 0,
        "biological_optimization_established": False,
    }


def _builder_kwargs(tmp_path: Path) -> dict:
    protocols = _protocols()
    refit_checkpoint = tmp_path / "delta_predictor_checkpoint.pt"
    refit_checkpoint.write_bytes(b"critic")
    flow_checkpoint = tmp_path / "base_flow.pt"
    flow_checkpoint.write_bytes(b"flow")
    return {
        "protocols": protocols,
        "control_adjudication": _control_adjudication(),
        "three_seed_adjudication": _three_seed_adjudication(),
        "frozen_test_config": _test_config(protocols),
        "frozen_test_summary": _test_summary(),
        "refit_config": _refit_config(protocols),
        "refit_summary": _refit_summary(),
        "refit_checkpoint": refit_checkpoint,
        "loso_results": _loso_results(),
        "reward_policy": json.loads(REWARD_POLICY.read_text(encoding="utf-8")),
        "online_encoder_validation": _online_encoder(),
        "flow_training_summary": _flow_training(),
        "flow_validation_summary": _flow_validation(),
        "flow_checkpoint": flow_checkpoint,
    }


def test_complete_synthetic_packet_unlocks_only_logical_dual_readiness(
    tmp_path: Path,
) -> None:
    builder, adjudicator = _modules()
    payload = builder.build_input(**_builder_kwargs(tmp_path))
    result = adjudicator.adjudicate(payload)
    assert payload["critic"]["frozen_test_summary"]["test_metrics"][
        "task_macro_spearman"
    ] < 0.0
    assert result["critic_status"] == "CRITIC_READY_FOR_GUIDANCE"
    assert result["flow_status"] == "FLOW_G0_READY"
    assert result["guided_unlocked"] is True
    assert result["biological_optimization_established"] is False


def test_test_metric_value_is_report_only_not_a_readiness_threshold(
    tmp_path: Path,
) -> None:
    builder, adjudicator = _modules()
    kwargs = _builder_kwargs(tmp_path)
    kwargs["frozen_test_summary"]["test_metrics"]["task_macro_spearman"] = -999.0
    result = adjudicator.adjudicate(builder.build_input(**kwargs))
    assert result["critic_checks"][
        "single_frozen_test_complete_without_test_selection"
    ] is True
    assert result["critic_status"] == "CRITIC_READY_FOR_GUIDANCE"


def test_one_nonpositive_loso_seed_keeps_critic_and_guidance_closed(
    tmp_path: Path,
) -> None:
    builder, adjudicator = _modules()
    kwargs = _builder_kwargs(tmp_path)
    row = kwargs["loso_results"][1]
    row["model_macro_spearman"] = 0.05
    row["macro_improvement"] = -0.05
    result = adjudicator.adjudicate(builder.build_input(**kwargs))
    assert result["critic_checks"]["three_complete_matched_loso_aggregations"] is True
    assert result["critic_checks"]["all_loso_seed_improvements_positive"] is False
    assert result["critic_status"] == "CRITIC_NOT_READY_FOR_GUIDANCE"
    assert result["flow_status"] == "FLOW_G0_READY"
    assert result["guided_unlocked"] is False


@pytest.mark.parametrize(
    ("section", "mutation", "check"),
    [
        ("control_adjudication", ("supports_three_frozen_seeds", False), "critic_v2_control_gate_pass"),
        ("three_seed_adjudication", ("supports_single_frozen_development_test", False), "critic_v2_three_seed_gate_pass"),
        ("frozen_test_summary", ("selected_epoch", 99), "single_frozen_test_complete_without_test_selection"),
        ("refit_summary", ("record_counts", {"TRAIN": 126164}), "all_126165_refit_complete"),
        ("reward_policy", ("uncertainty_in_guidance", "LEARNED_LOG_VARIANCE"), "guidance_reward_policy_frozen"),
        ("online_encoder_validation", ("novel_candidate_encoding_supported", False), "generated_candidate_online_encoder_ready"),
    ],
)
def test_each_critic_dependency_failure_keeps_guidance_closed(
    tmp_path: Path, section: str, mutation: tuple[str, object], check: str
) -> None:
    builder, adjudicator = _modules()
    payload = builder.build_input(**_builder_kwargs(tmp_path))
    payload["critic"][section][mutation[0]] = mutation[1]
    result = adjudicator.adjudicate(payload)
    assert result["critic_checks"][check] is False
    assert result["critic_status"] == "CRITIC_NOT_READY_FOR_GUIDANCE"
    assert result["guided_unlocked"] is False


def test_flow_failure_keeps_dual_readiness_closed(tmp_path: Path) -> None:
    builder, adjudicator = _modules()
    payload = builder.build_input(**_builder_kwargs(tmp_path))
    payload["flow"]["validation_summary"]["hard_legality_rate"] = 0.99
    result = adjudicator.adjudicate(payload)
    assert result["critic_status"] == "CRITIC_READY_FOR_GUIDANCE"
    assert result["flow_status"] == "FLOW_G0_NOT_READY"
    assert result["guided_unlocked"] is False


def test_evaluation_contamination_keeps_readiness_closed(tmp_path: Path) -> None:
    builder, adjudicator = _modules()
    payload = builder.build_input(**_builder_kwargs(tmp_path))
    payload["critic"]["refit_summary"]["evaluation_outcomes_read"] = 1
    result = adjudicator.adjudicate(payload)
    assert result["critic_checks"]["evaluation_not_used"] is False
    assert result["critic_status"] == "CRITIC_NOT_READY_FOR_GUIDANCE"
    assert result["guided_unlocked"] is False


def test_builder_rejects_wrong_loso_seed_or_missing_checkpoint(
    tmp_path: Path,
) -> None:
    builder, _ = _modules()
    kwargs = _builder_kwargs(tmp_path)
    kwargs["loso_results"][2]["seed"] = 999
    with pytest.raises(builder.CriticV2ReadinessInputError, match="seed set differs"):
        builder.build_input(**kwargs)

    kwargs = _builder_kwargs(tmp_path)
    kwargs["refit_checkpoint"].unlink()
    with pytest.raises(builder.CriticV2ReadinessInputError, match="checkpoint is absent"):
        builder.build_input(**kwargs)


def test_builder_rejects_protocol_drift(tmp_path: Path) -> None:
    builder, _ = _modules()
    kwargs = _builder_kwargs(tmp_path)
    kwargs["protocols"]["readiness"]["required_seeds"] = [20260822, 20260823, 999]
    with pytest.raises(builder.CriticV2ReadinessInputError, match="seed set differs"):
        builder.build_input(**kwargs)


def test_write_input_once_refuses_overwrite(tmp_path: Path) -> None:
    builder, _ = _modules()
    output = tmp_path / "readiness.json"
    builder.write_input_once({"schema_version": "x"}, output)
    with pytest.raises(builder.CriticV2ReadinessInputError, match="already exists"):
        builder.write_input_once({"schema_version": "x"}, output)


def test_real_readiness_output_contract_closes_the_v2_development_pipeline(
    tmp_path: Path,
) -> None:
    builder, adjudicator = _modules()
    guided = _load(GUIDED_SCRIPT, "critic_v2_guided_contract_test")
    matched = _load(MATCHED_SCRIPT, "critic_v2_matched_contract_test")
    comparison = _load(COMPARISON_SCRIPT, "critic_v2_comparison_contract_test")

    payload = builder.build_input(**_builder_kwargs(tmp_path))
    readiness_result = adjudicator.adjudicate(payload)

    guided_config = json.loads(
        (
            ROOT
            / "configs/route_a_v3_route2_mrnabert_critic_v2_guided_xeditflow_development_gpu0_v1.json"
        ).read_text(encoding="utf-8")
    )
    payload["critic"]["refit_checkpoint"] = guided_config[
        "critic_checkpoint_path"
    ]
    payload["flow"]["checkpoint"] = guided_config["base_flow_checkpoint_path"]
    guided.validate_guided_config(guided_config)
    guided.validate_readiness(payload, readiness_result, guided_config)

    matched_config = json.loads(
        (
            ROOT
            / "configs/route_a_v3_route2_mrnabert_critic_v2_matched_search_development_gpu0_v1.json"
        ).read_text(encoding="utf-8")
    )
    guided_summary = {
        "schema_version": guided.GUIDED_CONFIG_SCHEMA,
        "status": "GUIDED_XEDITFLOW_DEVELOPMENT_COMPLETE",
        "matched_search_budget_rule": matched_config["critic_budget_rule"],
        "per_source_compute_path": matched_config[
            "guided_compute_by_source_path"
        ],
        "evaluation_outcomes_read": 0,
        "generated_candidates_grant_canonical_credit": False,
        "biological_optimization_established": False,
    }
    budgets = matched.validate_inputs(
        matched_config,
        payload,
        readiness_result,
        {
            "schema_version": (
                "route_a_v3_route2_independent_generation_evaluator_adjudication.v1"
            ),
            "status": "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
            "development_test_outcomes_accessed": False,
            "evaluation_outcomes_accessed": False,
        },
        guided_summary,
        [{"source_key": "S1", "matched_search_critic_forward_budget": 101}],
        [{"source_key": "S1"}],
    )
    assert budgets == {"S1": 101}

    comparison_config = json.loads(
        (
            ROOT
            / "configs/route_a_v3_route2_mrnabert_critic_v2_generation_comparison_development_gpu0_v1.json"
        ).read_text(encoding="utf-8")
    )
    comparison.validate_config_boundary(comparison_config)
    assert comparison.GUIDED_METHOD == guided.GUIDED_METHOD_ID
