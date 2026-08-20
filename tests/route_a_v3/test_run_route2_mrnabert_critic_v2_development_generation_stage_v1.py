from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "scripts/route_a_v3/run_route2_mrnabert_critic_v2_development_generation_stage_v1.py"
)
GUIDED_TEMPLATE = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_guided_xeditflow_development_gpu0_v1.json"
)
MATCHED_TEMPLATE = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_matched_search_development_gpu0_v1.json"
)
COMPARISON_TEMPLATE = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_generation_comparison_development_gpu0_v1.json"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "critic_v2_development_generation_stage_runner", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_selects_most_free_eligible_gpu_with_stable_tie_break() -> None:
    runner = _load_runner()
    memory = {0: 1000, 1: 5000, 2: 7000, 3: 7000, 4: 2000, 5: 6000}
    assert runner.select_gpu(memory, 4096) == 2
    with pytest.raises(runner.CriticV2GenerationStageError, match="enough free memory"):
        runner.select_gpu({gpu: 4095 for gpu in runner.GPU_CANDIDATES}, 4096)


def test_runtime_configs_change_only_selected_gpu_binding() -> None:
    runner = _load_runner()
    templates = {
        "guided": _read(GUIDED_TEMPLATE),
        "matched": _read(MATCHED_TEMPLATE),
        "comparison": _read(COMPARISON_TEMPLATE),
    }
    payloads = runner.build_runtime_payloads(
        templates["guided"], templates["matched"], templates["comparison"], 4
    )
    for name, template in templates.items():
        expected = deepcopy(template)
        expected["device"] = "cuda:4"
        expected["physical_gpu_index"] = 4
        assert payloads[name] == expected
        assert payloads[name]["evaluation_outcomes_accessed"] is False
        assert payloads[name]["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert payloads["comparison"]["generated_candidates_grant_canonical_credit"] is False
    assert payloads["comparison"]["evaluation_release_state"] == "CLOSED"
    assert (
        payloads["comparison"]["guided_method_id"]
        == runner.guided_runner.GUIDED_METHOD_ID
    )


def test_runtime_configs_are_written_once(tmp_path: Path) -> None:
    runner = _load_runner()
    payloads = {
        "guided": {"stage": "guided"},
        "matched": {"stage": "matched"},
        "comparison": {"stage": "comparison"},
    }
    runtime_root = tmp_path / "runtime"
    paths = runner.write_runtime_payloads_once(payloads, runtime_root)
    assert list(paths) == ["guided", "matched", "comparison"]
    assert {name: _read(path) for name, path in paths.items()} == payloads
    with pytest.raises(
        runner.CriticV2GenerationStageError, match="runtime root already exists"
    ):
        runner.write_runtime_payloads_once(payloads, runtime_root)


def test_source_gates_writes_and_orders_development_only_children() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    main = source[source.index("def main") :]
    assert main.index("guided_runner.validate_readiness") < main.index(
        "write_runtime_payloads_once"
    )
    assert main.index('"guided",') < main.index('"matched",') < main.index(
        '"comparison",'
    )
    assert '"development_test_opened": False' in main
    assert '"evaluation_opened": False' in main
    assert '"generated_candidates_grant_canonical_credit": False' in main
    assert '"biological_optimization_established": False' in main
