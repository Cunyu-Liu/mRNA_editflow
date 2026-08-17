import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/route_a_v3/build_route2_test_preserving_loso_aggregation_inputs_v1.py"
SPEC = importlib.util.spec_from_file_location("build_loso_inputs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _summary(study: str, seed: int, model_kind: str):
    return {
        "seed": seed,
        "model_kind": model_kind,
        "loso_holdout_study_unit_id": study,
        "development_test_outcomes_evaluated": False,
        "test_metrics": None,
        "evaluation_outcomes_read": 0,
        "validation_metrics": {
            "task_count": 1,
            "task_spearman_defined_count": 1,
            "task_macro_spearman": 0.2,
        },
    }


def test_builds_three_aligned_inputs(tmp_path: Path):
    model_root = tmp_path / "model"
    baseline_root = tmp_path / "baseline"
    for seed, gpu in MODULE.SEED_GPU_PAIRS:
        for study in MODULE.HOLDOUT_STUDIES:
            model_path = model_root / study / f"seed{seed}_gpu{gpu}_huber_v1/training_summary.json"
            baseline_path = baseline_root / study / f"seed{seed}_gpu{gpu}_global_scaled_v1/training_summary.json"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text(json.dumps(_summary(study, seed, "model")))
            baseline_path.write_text(json.dumps(_summary(study, seed, "baseline")))
    payloads = MODULE.build_inputs(model_root, baseline_root, loss_kind="huber")
    assert set(payloads) == {20260822, 20260823, 20260824}
    for payload in payloads.values():
        assert len(payload["model_results"]) == 7
        assert len(payload["baseline_results"]) == 7
        assert payload["zero_record_development_studies"] == ["GSE256185"]
        assert payload["development_test_opened"] is False
