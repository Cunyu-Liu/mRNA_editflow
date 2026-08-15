from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/gse200304_source_relative_critic_g1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_gse200304_source_relative_critic_g1_implementation_candidate_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("critic_g1_candidate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inactive_config(module):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["activation_state"] = module.INACTIVE
    config["future_activation_requirements"]["current_requirement_count_satisfied"] = 0
    config["activation_binding"] = {key: None for key in config["activation_binding"]}
    module.validate_config(config)
    return config


def test_inactive_candidate_validates_without_importing_torch() -> None:
    before = set(sys.modules)
    module = _module()
    config = _inactive_config(module)
    result = module.validate_only(config)
    assert result["status"] == "PASS_STATIC_IMPLEMENTATION_CONTRACT_NOT_ACTIVE_NOT_RUN"
    assert result["activation_state"] == module.INACTIVE
    assert result["data_rows_read"] == 0
    assert result["model_constructions"] == 0
    assert result["cuda_touches"] == 0
    assert result["parameter_updates"] == 0
    assert result["outputs_written"] == 0
    assert "torch" not in set(sys.modules) - before


def test_inactive_run_stops_before_asset_model_cuda_and_output(tmp_path: Path) -> None:
    module = _module()
    config = _inactive_config(module)
    missing_asset = tmp_path / "missing-assets"
    output = tmp_path / "output"
    with pytest.raises(module.InactiveAuthorityError, match="stop before data"):
        module.run_once(config, missing_asset, output)
    assert not missing_asset.exists()
    assert not output.exists()


def test_full_length_antisymmetric_architecture_and_single_fit_are_frozen() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    input_contract = config["input_contract"]
    assert input_contract["context_vector_definition"].startswith("SIXTEEN_CONTIGUOUS_201NT_POSITION_BINS")
    assert input_contract["edit_feature_definition"].startswith("THREE_POSITIONS_CENTER_MINUS_ONE_CENTER")
    assert input_contract["effect_definition"].startswith("MEAN_OVER_SIX_PAIRED_BIOLOGICAL_REPLICATES")
    assert input_contract["standard_error_definition"].endswith("FINITE_POSITIVE")
    model = config["model_contract"]
    assert model["fixed_prefix_truncation_allowed"] is False
    assert model["full_length_dynamic_padding_required"] is True
    assert model["mean_construction"] == "HALF_FORWARD_MINUS_REVERSE_PAIR_SCORE"
    assert model["external_learned_input_count"] == 0
    run = config["single_fit_contract"]
    assert [
        run["authorized_execution_count"],
        run["optimizer_fit_count"],
        run["fold_model_count"],
        run["checkpoint_count"],
        run["final_refit_count"],
        run["seed_count"],
    ] == [1, 1, 1, 1, 0, 1]
    assert run["terminal_checkpoint_only"] is True
    assert run["early_stopping_allowed"] is False
    assert run["best_checkpoint_selection_allowed"] is False
    assert run["hyperparameter_search_allowed"] is False
    assert run["automatic_retry_allowed"] is False
    evaluator = config["evaluator_and_baseline_contract"]
    assert evaluator["primary_metric"] == "WITHIN_STUDY_SOURCE_GROUP_EQUAL_WEIGHT_SPEARMAN"
    assert evaluator["baseline_set"] == [
        "TRAIN_SOURCE_GROUP_EQUAL_WEIGHT_GLOBAL_MEAN",
        "TRAIN_DIRECTED_EDIT_TYPE_MEAN",
        "TRAIN_GC_AND_LENGTH_LINEAR_RIDGE_FIXED_ALPHA_1",
        "TRAIN_15MER_COUNT_RIDGE_FIXED_ALPHA_10",
    ]
    assert evaluator["kmer_feature_dimension"] == 4096
    assert evaluator["guide_or_model_selection_output_allowed"] is False
    assert evaluator["test_feedback_allowed"] is False


def test_static_source_contains_real_model_and_input_paths_but_no_current_execution() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (isinstance(node, ast.Import) and any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and (node.module == "torch" or str(node.module).startswith("torch.")))
        )
    ]
    assert forbidden_imports == []
    functions = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {
        "_load_rows_and_split",
        "_torch_components",
        "_batch_tensors",
        "_predict_rows",
        "_fit_and_evaluate_baselines",
        "_group_equal_metrics",
        "_coverage_risk",
        "_repository_audit",
        "_cuda_binding_audit",
        "_write_terminal_outputs",
        "_write_failure",
        "run_once",
        "require_active_before_operational_io",
    } <= functions
    assert "FullLengthEncoder" in source
    assert "SourceRelativeCritic" in source
    assert "src_key_padding_mask" in source
    assert "[:max_len]" not in source
    module = _module()
    config = _inactive_config(module)
    assert config["future_activation_requirements"]["current_requirement_count_satisfied"] == 0
    assert set(config["activation_binding"].values()) == {None}
    assert set(value for key, value in config["current_truth"].items() if key != "scientific_claim_status") <= {0, False}


def test_source_group_equal_metric_is_not_row_weighted_and_constant_rank_stops() -> None:
    module = _module()
    predictions = [
        {"source_group": "g1", "observed": 0.0, "calibrated_mean": 0.0, "predicted_scale": 1.0},
        {"source_group": "g1", "observed": 100.0, "calibrated_mean": 100.0, "predicted_scale": 1.0},
        {"source_group": "g2", "observed": 60.0, "calibrated_mean": 55.0, "predicted_scale": 1.0},
        {"source_group": "g3", "observed": 70.0, "calibrated_mean": 65.0, "predicted_scale": 1.0},
    ]
    metrics = module._group_equal_metrics(predictions)
    assert metrics["source_group_count"] == 3
    assert metrics["mae"] == pytest.approx(10.0 / 3.0)
    constant = [dict(item, calibrated_mean=1.0) for item in predictions]
    with pytest.raises(module.ContractError, match="constant-rank terminal"):
        module._group_equal_metrics(constant)


def test_calibration_and_coverage_risk_are_frozen_from_calibration_only() -> None:
    module = _module()
    calibration = [
        {"source_group": f"g{i}", "observed": float(2 * i + 1), "predicted_mean": float(i), "predicted_scale": float(i + 1)}
        for i in range(4)
    ]
    slope, intercept = module._calibration_line(calibration)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    module._apply_calibration(calibration, slope, intercept)
    terminal = [dict(item) for item in calibration]
    risk = module._coverage_risk(calibration, terminal, [1.0, 0.5])
    assert [item["target_retained_fraction"] for item in risk] == [1.0, 0.5]
    assert risk[1]["terminal_retained_group_count"] == 2


def test_directed_edit_and_hashed_15mer_features_are_orientation_sensitive() -> None:
    module = _module()
    source = "A" * 100 + "C" + "A" * 100
    candidate = "A" * 100 + "G" + "A" * 100
    row = {"source_sequence": source, "candidate_sequence": candidate}
    reverse = {"source_sequence": candidate, "candidate_sequence": source}
    assert module._directed_edit_type(row) == "C>G"
    forward_features = module._kmer_delta_features(row, 15, 4096)
    reverse_features = module._kmer_delta_features(reverse, 15, 4096)
    assert forward_features
    assert reverse_features == {key: -value for key, value in forward_features.items()}


def test_future_active_shape_is_explicit_and_complete() -> None:
    module = _module()
    config = _inactive_config(module)
    config["activation_state"] = module.ACTIVE
    config["future_activation_requirements"]["current_requirement_count_satisfied"] = 5
    for key in config["activation_binding"]:
        config["activation_binding"][key] = 1 if key in {"materialized_rows_bytes", "split_assignments_bytes", "cuda_physical_index"} else "BOUND"
    config["activation_binding"]["materialized_rows_sha256"] = "a" * 64
    config["activation_binding"]["split_assignments_sha256"] = "b" * 64
    config["activation_binding"]["implementation_commit"] = "c" * 40
    config["activation_binding"]["materialized_rows_path"] = "/private/rows.jsonl"
    config["activation_binding"]["split_assignments_path"] = "/private/split.json"
    config["activation_binding"]["python_executable"] = "/runtime/bin/python"
    config["activation_binding"]["output_directory"] = "/private/output"
    config["activation_binding"]["cuda_uuid"] = "GPU-00000000-0000-0000-0000-000000000000"
    module.validate_config(config)
    assert module.validate_only(config)["status"] == "PASS_ACTIVE_EXACTLY_ONE_RUN_AUTHORITY_STATIC_VALIDATION_NOT_RUN"
