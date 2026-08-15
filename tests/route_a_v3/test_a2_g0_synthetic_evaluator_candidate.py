from __future__ import annotations

import ast
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/route_a_v3/a2_g0_synthetic_evaluator_candidate.py"
CONFIG_PATH = ROOT / "configs/route_a_v3_a2_g0_evaluator_candidate_v1.json"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("a2_g0_synthetic_evaluator_candidate_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_is_non_authoritative_and_runtime_truth_is_zero() -> None:
    runner = _load()
    config = runner.load_candidate_config()
    assert config["document_status"] == "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL"
    assert config["authority_status"] == "NON_AUTHORITATIVE"
    assert config["activation_state"] == "INACTIVE_EVALUATOR_CANDIDATE"
    assert set(config["runtime_truth"].values()) <= {0, False}
    assert "REAL_EVALUATION" in config["forbidden_operations"]


def test_validate_only_uses_only_synthetic_fixture_and_emits_no_files(tmp_path: Path, capsys, monkeypatch) -> None:
    runner = _load()
    monkeypatch.chdir(tmp_path)
    assert runner.main(["--validate-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "VALIDATE_ONLY_SYNTHETIC_G0"
    assert result["synthetic_fixture_only"] is True
    assert result["component_disjoint"] is True
    assert result["outcome_values_consumed_for_split"] is False
    assert result["final_assignment_materialized"] is False
    assert result["candidate_minus_source_valid"] is True
    assert result["missing_or_nonfinite_coerced_to_zero"] is False
    assert result["runtime_truth"]["project_rows_read"] == 0
    assert result["runtime_truth"]["cuda_probe_calls"] == 0
    assert result["a2_pass_asserted"] is False
    assert result["scientific_claim_established"] is False
    assert list(tmp_path.iterdir()) == []


def test_synthetic_split_is_component_disjoint_and_outcome_blind() -> None:
    runner = _load()
    edges = (
        runner.SyntheticEdge("SYN_SOURCE_A", "SYN_CANDIDATE_A"),
        runner.SyntheticEdge("SYN_SOURCE_A", "SYN_CANDIDATE_B"),
        runner.SyntheticEdge("SYN_SOURCE_B", "SYN_CANDIDATE_C"),
    )
    split = runner.outcome_blind_synthetic_split(edges, fold_count=2)
    assert split.outcome_values_consumed is False
    assert split.final_assignment_materialized is False
    assert split.components == (
        ("SYN_CANDIDATE_A", "SYN_CANDIDATE_B", "SYN_SOURCE_A"),
        ("SYN_CANDIDATE_C", "SYN_SOURCE_B"),
    )
    assert split.fold_by_component == (0, 1)


@pytest.mark.parametrize(
    ("source_endpoint", "candidate_endpoint", "biological_se", "replicate_count", "direction", "expected"),
    [
        (None, 1.0, 0.1, 2, "CANDIDATE_MINUS_SOURCE", "missing"),
        (1.0, float("nan"), 0.1, 2, "CANDIDATE_MINUS_SOURCE", "non-finite"),
        (1.0, 1.2, None, 2, "CANDIDATE_MINUS_SOURCE", "missing"),
        (1.0, 1.2, -0.1, 2, "CANDIDATE_MINUS_SOURCE", "nonnegative"),
        (1.0, 1.2, 0.1, 1, "CANDIDATE_MINUS_SOURCE", "at least two"),
        (1.0, 1.2, 0.1, 2, "SOURCE_MINUS_CANDIDATE", "candidate-minus-source"),
    ],
)
def test_missing_nonfinite_and_invalid_biological_se_fail_closed(
    source_endpoint, candidate_endpoint, biological_se, replicate_count, direction, expected
) -> None:
    runner = _load()
    with pytest.raises(runner.G0CandidateError, match=expected):
        runner.validate_direction_normalized_synthetic_endpoint(
            source_id="SYN_SOURCE_A",
            candidate_id="SYN_CANDIDATE_A",
            source_endpoint=source_endpoint,
            candidate_endpoint=candidate_endpoint,
            biological_se=biological_se,
            biological_replicate_count=replicate_count,
            endpoint_direction=direction,
        )


def test_direction_normalization_is_candidate_minus_source() -> None:
    runner = _load()
    result = runner.validate_direction_normalized_synthetic_endpoint(
        source_id="SYN_SOURCE_A",
        candidate_id="SYN_CANDIDATE_A",
        source_endpoint=1.0,
        candidate_endpoint=1.25,
        biological_se=0.1,
        biological_replicate_count=2,
        endpoint_direction="CANDIDATE_MINUS_SOURCE",
    )
    assert result.candidate_minus_source == pytest.approx(0.25)
    assert result.endpoint_direction == "CANDIDATE_MINUS_SOURCE"


@pytest.mark.parametrize(
    "edges",
    [
        (("SYN_SOURCE_A", "SYN_SOURCE_A"),),
        (("REAL_SOURCE_A", "SYN_CANDIDATE_A"),),
        (
            ("SYN_SOURCE_A", "SYN_CANDIDATE_A"),
            ("SYN_SOURCE_A", "SYN_CANDIDATE_A"),
        ),
    ],
)
def test_synthetic_component_contract_rejects_ambiguous_or_non_synthetic_edges(edges) -> None:
    runner = _load()
    with pytest.raises(runner.G0CandidateError):
        runner.synthetic_connected_components(tuple(runner.SyntheticEdge(*edge) for edge in edges))


def test_candidate_has_no_torch_network_subprocess_csv_or_runtime_write_imports() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "argparse",
        "dataclasses",
        "json",
        "math",
        "pathlib",
        "typing",
    }
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert ".write_text(" not in source
    assert ".write_bytes(" not in source
    assert "torch" not in imported_roots


def test_candidate_config_rejects_runtime_or_boundary_drift() -> None:
    runner = _load()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutated = deepcopy(config)
    mutated["runtime_truth"]["project_rows_read"] = 1
    with pytest.raises(runner.G0CandidateError, match="nonzero"):
        runner.validate_candidate_config(mutated)
    mutated = deepcopy(config)
    mutated["synthetic_evaluator_contract"]["real_evaluation_allowed"] = True
    with pytest.raises(runner.G0CandidateError, match="boundary"):
        runner.validate_candidate_config(mutated)
