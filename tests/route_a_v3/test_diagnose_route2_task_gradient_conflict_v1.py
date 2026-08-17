import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/diagnose_route2_task_gradient_conflict_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("route2_task_gradient_diagnostic_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _adjudication(status="EXPLORATORY_SCREEN_DOES_NOT_SUPPORT_CONFIRMATION"):
    return {
        "schema_version": "route_a_v3_route2_method_repair_screen_adjudication.v2",
        "status": status,
        "fresh_confirmation_seeds": [],
        "evaluation_used_for_selection": False,
        "development_test_used_for_selection": False,
    }


def test_gradient_cosine_matrix_reports_alignment_and_conflict() -> None:
    module = _load()
    gradients = {
        "task_a": torch.tensor([1.0, 0.0]),
        "task_b": torch.tensor([0.5, 0.0]),
        "task_c": torch.tensor([-1.0, 0.0]),
    }
    matrix = module.cosine_matrix(gradients)
    assert matrix["task_a"]["task_b"] == pytest.approx(1.0)
    assert matrix["task_a"]["task_c"] == pytest.approx(-1.0)
    assert matrix["task_a"]["task_a"] == pytest.approx(1.0)


def test_gradient_cosine_matrix_rejects_zero_gradient() -> None:
    module = _load()
    with pytest.raises(module.TaskGradientDiagnosticError, match="zero"):
        module.cosine_matrix({"task_a": torch.ones(2), "task_b": torch.zeros(2)})


def test_diagnostic_requires_terminal_no_go_before_execution() -> None:
    module = _load()
    module.validate_terminal_no_go(_adjudication())
    with pytest.raises(module.TaskGradientDiagnosticError, match="not allowed"):
        module.validate_terminal_no_go(
            _adjudication("EXPLORATORY_SCREEN_SUPPORTS_FRESH_SEED_CONFIRMATION")
            | {"fresh_confirmation_seeds": [1, 2, 3]}
        )


def test_evenly_spaced_batches_cover_first_and_last() -> None:
    module = _load()

    class Sampler:
        batches = [[index] for index in range(10)]

    assert module.evenly_spaced_batches(Sampler(), 3) == [[0], [4], [9]]


def test_diagnostic_source_has_no_optimizer_or_evaluation_loader() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "optimizer.step" not in text
    assert "load_evaluation" not in text
    assert 'record.split == "TRAIN"' in text
    assert '"parameter_updates": 0' in text
