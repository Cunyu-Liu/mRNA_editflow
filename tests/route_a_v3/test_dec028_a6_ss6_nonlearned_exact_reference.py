from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/run_dec028_a6_ss6_nonlearned_exact_reference.py"
CONFIG = ROOT / "configs/route_a_v3_dec028_a6_ss6_nonlearned_exact_reference_v1.json"


def module():
    spec = importlib.util.spec_from_file_location("ss6_reference", SCRIPT)
    value = importlib.util.module_from_spec(spec); sys.modules[spec.name] = value; spec.loader.exec_module(value)
    return value


def test_static_boundary_has_no_torch_data_cuda_or_model_surface() -> None:
    source = SCRIPT.read_text(encoding="utf-8"); tree = ast.parse(source)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported_from = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name == "torch" or str(name).startswith("torch.") for name in imports | imported_from)
    assert "nvidia-smi" not in source
    assert "subprocess" not in imports


def test_state_geometry_is_source_anchored_acyclic_and_budgeted() -> None:
    value = module()
    for budget, expected in ((1, 6), (3, 26), (5, 32)):
        states = value.states(5, budget)
        assert len(states) == expected
        for state in states:
            assert sum(state) <= budget
            for target, rate, _ in value.raw_actions(state, budget, 0, 0, 0.0001):
                assert rate >= 0.0001
                assert target == value.STOP or sum(target) == sum(state) + 1


def test_alias_aggregation_and_base_recovery_are_exact() -> None:
    value = module(); state = (0, 0, 0, 0, 0)
    raw = value.raw_actions(state, 3, 4, 1, 0.0001)
    aggregated = value.aggregated_transitions(state, 3, 4, 1, 0.0001)
    assert len(raw) == 11
    assert len(aggregated) == 6
    assert abs(sum(rate for _, rate, _ in raw) - sum(aggregated.values())) < 1e-15


def test_dp_and_independent_enumeration_match_one_graph() -> None:
    value = module(); all_states = value.states(5, 3) + [value.STOP]; source = (0, 0, 0, 0, 0)
    initial = {state: 0.0 for state in all_states}; initial[source] = 1.0
    left = value.dp_segment(initial, all_states, 3, 7, 0, 0.35, 0.0001)
    right = value.enumeration_segment(initial, all_states, 3, 7, 0, 0.35, 0.0001)
    assert value.total_variation(left, right) <= 1e-12
    assert abs(sum(left.values()) - 1.0) <= 1e-12


def test_full_96_graph_suite_passes_and_keeps_all_locks() -> None:
    value = module(); report = value.run_suite(value.load_config(CONFIG))
    assert report["status"] == "PASS_SS6_NONLEARNED_ENGINEERING_REFERENCE"
    assert report["graph_count"] == 96
    assert report["maximum_terminal_total_variation"] <= 1e-12
    assert report["maximum_mass_error"] <= 1e-12
    assert report["maximum_base_rate_recovery_error"] <= 1e-12
    assert report["illegal_edge_count"] == 0
    assert report["future_learned_execution_authorized"] is False
    assert report["critic_lcb_manifest_available"] is False
    assert report["project_data_row_reads"] == 0
    assert report["torch_imports"] == 0
    assert report["cuda_touches"] == 0
    assert report["parameter_updates"] == 0
    assert report["scientific_claim_status"] == "NOT_ESTABLISHED"
