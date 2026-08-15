from __future__ import annotations

import ast
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/freeze_gse200304_dec028_ss4_split_evaluator.py"
CONFIG = ROOT / "configs/route_a_v3_dec028_gse200304_ss4_split_evaluator_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("ss4", SCRIPT); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def _config(): return json.loads(CONFIG.read_text())


def test_authority_freezes_one_outcome_blind_split_and_no_g1() -> None:
    config = _config(); split = config["split_contract"]
    assert config["authority"]["authorized_execution_count"] == 1
    assert split["roles"] == ["TRAIN", "CALIBRATION", "TEST"]
    assert split["historical_fold_roles_reused"] is False and split["outcome_columns_used"] == []
    assert split["retry_or_resalt_allowed"] is False
    assert all(config["authority"][key] is False for key in ("model_authorized", "cuda_authorized", "optimizer_authorized", "training_authorized", "g1_authorized", "sealed_access_authorized"))


def test_component_assignment_is_deterministic_indivisible_and_targeted() -> None:
    module = _module(); sizes = {"a": 10, "b": 7, "c": 4, "d": 3, "e": 2, "f": 1}
    target = {"TRAIN": .7, "CALIBRATION": .15, "TEST": .15}; roles = ["TRAIN", "CALIBRATION", "TEST"]
    first = module.assign_components(sizes, "salt", target, roles); second = module.assign_components(sizes, "salt", target, roles)
    assert first == second and set(first) == set(sizes) and set(first.values()) == set(roles)


def test_canonical_locator_matches_frozen_domain_shape() -> None:
    module = _module(); value = module.canonical_locator("GENE:10_C-T")
    assert len(value) == 64 and set(value) <= set("0123456789abcdef")


def test_unknown_binding_stops_before_private_inputs(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.SplitError, match="not BOUND"): module.audit_repository(_config())
    assert not (tmp_path / "output").exists()


def test_evaluator_and_baselines_are_frozen_without_execution() -> None:
    config = _config(); evaluator = config["evaluator_contract"]
    assert evaluator["primary_metric"] == "WITHIN_STUDY_SOURCE_GROUP_EQUAL_WEIGHT_SPEARMAN"
    assert len(evaluator["baseline_set"]) == 4
    assert evaluator["guide_or_model_selection_output_allowed"] is False
    assert config["current_truth"]["metric_execution_count"] == 0
    assert config["current_truth"]["baseline_fit_count"] == 0
    assert config["outputs"]["private_assignment_filename"] == "GSE200304_SINGLE_STUDY_SPLIT_ASSIGNMENTS_PRIVATE.json"


def test_no_model_cuda_or_training_imports_and_mutation_fails_closed() -> None:
    module = _module(); tree = ast.parse(SCRIPT.read_text()); imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module: imported.add(node.module)
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    config = deepcopy(_config()); config["authority"]["g1_authorized"] = True
    with pytest.raises(module.SplitError, match="g1_authorized"): module.validate_config(config)
