from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/build_route2_generation_baseline_selection_input_v2.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_generation_selection_input_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_composes_exact_required_method_set(tmp_path: Path) -> None:
    module = _load()
    methods = ["random_legal", "beam"]
    evaluation_root = tmp_path / "evaluations"
    evaluation_root.mkdir()
    jobs_rows = []
    for method_id in methods:
        evaluation = {"method": method_id, "kind": "evaluation"}
        (evaluation_root / f"{method_id}_evaluation_v2.json").write_text(
            json.dumps(evaluation), encoding="utf-8"
        )
        output = tmp_path / f"{method_id}.jsonl"
        output.with_suffix(output.suffix + ".summary.json").write_text(
            json.dumps({"method": method_id, "kind": "summary"}), encoding="utf-8"
        )
        jobs_rows.append({"method_id": method_id, "output_path": str(output)})
    protocol = {
        "schema_version": "route_a_v3_route2_generation_matched_compute_repair_protocol.v1",
        "required_method_ids": methods,
        "independent_evaluation_output_root": str(evaluation_root),
        "selection_bootstrap_iterations": 10000,
        "selection_bootstrap_seed": 20260816,
        "forward_equivalent_budget_per_source": 320,
        "search_critic_forward_budget_per_source": 256,
    }
    jobs = {
        "schema_version": "route_a_v3_route2_generation_independent_evaluator_jobs.v1",
        "jobs": jobs_rows,
    }
    result = module.build(protocol, jobs)
    assert result["required_method_ids"] == methods
    assert result["forward_equivalent_budget_per_source"] == 320
    assert result["critic_forward_budget_per_source"] == 256
    assert [row["method_id"] for row in result["baseline_evaluations"]] == methods
    assert result["baseline_evaluations"][0]["evaluation"]["kind"] == "evaluation"
    assert result["baseline_evaluations"][0]["independent_evaluator_summary"]["kind"] == "summary"
