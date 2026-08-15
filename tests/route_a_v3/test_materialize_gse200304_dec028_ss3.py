from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/route_a_v3/materialize_gse200304_dec028_ss3.py"
CONFIG = REPO_ROOT / "configs/route_a_v3_dec028_gse200304_ss3_materialization_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("ss3_materializer", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_authority_is_one_materialization_only_and_all_downstream_actions_false() -> None:
    config = _config()
    assert config["authority"]["authorized_execution_count"] == 1
    assert all(config["authority"][key] is False for key in ("split_authorized", "model_authorized", "cuda_authorized", "optimizer_authorized", "training_authorized", "g1_authorized", "sealed_access_authorized"))
    assert config["row_contract"]["membership_frozen_before_replicate_effect_calculation"] is True
    assert (config["row_contract"]["expected_public_join_count"], config["row_contract"]["expected_na_exclusion_count"], config["row_contract"]["expected_materialized_count"]) == (6772, 225, 6547)


def test_unknown_binding_stops_before_any_asset_or_output(tmp_path: Path) -> None:
    module = _module()
    config = _config()
    with pytest.raises(module.MaterializationError, match="not BOUND"):
        module.audit_repository(config)
    assert not (tmp_path / "assets").exists()
    assert not (tmp_path / "output").exists()


def test_orientation_context_edit_and_replicate_effect_are_replayable() -> None:
    module = _module()
    source = "A" * 100 + "C" + "A" * 100
    candidate = "A" * 100 + "T" + "A" * 100
    normalized = module._normalize_pair("GENE:10_C-T", source, candidate)
    assert normalized[:2] == (source, candidate)
    assert len(module._context_vector(source, candidate)) == 64
    edit = module._edit_features(source, candidate)
    assert len(edit) == 12 and math.isclose(sum(edit), 0.0)
    header = ["barcode"]
    values = []
    for arm in ("WT", "Mutant"):
        for role in ("High_Poly", "Low_Poly", "Total_RNA"):
            for replicate in range(1, 7):
                header.append(f"{role}_{replicate}_S{replicate}_{arm}")
                values.append(1.0 + (0.5 if arm == "Mutant" and role == "High_Poly" else 0.0))
    effects = module._replicate_effects(header, values)
    assert len(effects) == 6 and all(math.isfinite(item) for item in effects)


def test_build_rows_stops_if_positive_se_does_not_close() -> None:
    module = _module()
    source = "A" * 100 + "C" + "A" * 100
    candidate = "A" * 100 + "T" + "A" * 100
    key = "GENE:10_C-T"
    header = ["barcode"]
    values = []
    for arm in ("WT", "Mutant"):
        for role in ("High_Poly", "Low_Poly", "Total_RNA"):
            for replicate in range(1, 7):
                header.append(f"{role}_{replicate}_S{replicate}_{arm}")
                values.append(1.0)
    with pytest.raises(module.MaterializationError, match="public join count differs"):
        module.build_rows({key: (source, candidate)}, {key}, header, {key: values})


def test_source_has_no_model_cuda_optimizer_or_training_import() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "split_assignment_count\": 0" in source
    assert "cuda_touch_count\": 0" in source
    assert "g1_launched\": False" in source


def test_config_semantic_mutation_fails_closed() -> None:
    module = _module()
    config = deepcopy(_config())
    config["authority"]["g1_authorized"] = True
    with pytest.raises(module.MaterializationError, match="g1_authorized"):
        module.validate_config(config)
    config = _config()
    config["row_contract"]["standard_error_definition"] = "ARBITRARY"
    with pytest.raises(module.MaterializationError, match="standard_error_definition"):
        module.validate_config(config)
