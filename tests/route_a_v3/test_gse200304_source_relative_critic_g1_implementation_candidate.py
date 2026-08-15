from __future__ import annotations

import ast
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


def test_inactive_candidate_validates_without_importing_torch() -> None:
    before = set(sys.modules)
    module = _module()
    config = module.load_config(CONFIG_PATH)
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
    config = module.load_config(CONFIG_PATH)
    missing_asset = tmp_path / "missing-assets"
    output = tmp_path / "output"
    with pytest.raises(module.InactiveAuthorityError, match="stop before data"):
        module.run_once(config, missing_asset, output)
    assert not missing_asset.exists()
    assert not output.exists()


def test_full_length_antisymmetric_architecture_and_single_fit_are_frozen() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
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


def test_static_source_contains_real_model_and_input_paths_but_no_current_execution() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {
        "_load_rows_and_split",
        "_torch_components",
        "_batch_tensors",
        "_predict_rows",
        "_write_terminal_outputs",
        "_write_failure",
        "run_once",
        "require_active_before_operational_io",
    } <= functions
    assert "FullLengthEncoder" in source
    assert "SourceRelativeCritic" in source
    assert "src_key_padding_mask" in source
    assert "[:max_len]" not in source
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["future_activation_requirements"]["current_requirement_count_satisfied"] == 0
    assert set(config["activation_binding"].values()) == {None}
    assert set(value for key, value in config["current_truth"].items() if key != "scientific_claim_status") <= {0, False}
