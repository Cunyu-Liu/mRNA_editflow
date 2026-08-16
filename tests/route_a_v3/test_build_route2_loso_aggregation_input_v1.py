from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/build_route2_loso_aggregation_input_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_route2_loso_aggregation_input_v1_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_build_collects_one_completed_summary_and_aligned_evaluations(tmp_path: Path) -> None:
    module = _load()
    run_root = tmp_path / "runs"
    evaluation_root = tmp_path / "evaluations"
    seed = 17
    for study in module.EXPECTED_LOSO_STUDIES:
        lower = study.lower()
        completed = run_root / f"delta_main_2m_lr3e4_loso_{lower}_seed{seed}_v1"
        _write(completed / "training_summary.json", {"study": study, "seed": seed})
        (run_root / f"delta_main_2m_lr3e4_loso_{lower}_seed{seed}_failed").mkdir(parents=True)
        _write(evaluation_root / f"{lower}_model_vs_strongest_evaluation_v1.json", {"kind": "model"})
        _write(evaluation_root / f"{lower}_strongest_only_evaluation_v1.json", {"kind": "baseline"})

    payload = module.build(seed, run_root, evaluation_root)
    assert payload["seed"] == seed
    assert len(payload["model_results"]) == 7
    assert len(payload["baseline_results"]) == 7
    assert payload["zero_record_development_studies"] == ["GSE256185"]
    assert {row["training_summary"]["study"] for row in payload["model_results"]} == set(module.EXPECTED_LOSO_STUDIES)


def test_build_rejects_multiple_completed_summaries_for_one_study(tmp_path: Path) -> None:
    module = _load()
    module.EXPECTED_LOSO_STUDIES = ("GSE200304",)
    run_root = tmp_path / "runs"
    for suffix in ("v1", "recovery"):
        _write(
            run_root / f"delta_main_2m_lr3e4_loso_gse200304_seed17_{suffix}" / "training_summary.json",
            {"seed": 17},
        )
    with pytest.raises(module.LosoInputBuildError, match="found 2"):
        module.build(17, run_root, tmp_path / "evaluations")
