from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_route2_mrnabert_signal_controls_v1.py"
PROTOCOL = ROOT / "configs/route_a_v3_route2_mrnabert_signal_control_gate_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location("mrnabert_signal_control_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASKS = [
    "MEAN_RIBOSOME_LOAD::region=0",
    "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1",
    "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1",
    "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1",
    "RNA_HALF_LIFE_MINUTES::region=0",
    "RNA_HALF_LIFE_MINUTES::region=1",
    "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1",
    "te_log2_polysome_over_totalrna::region=0",
    "transcript_log2_totalrna_over_dna::region=0",
]


def _summary(*, kind: str, candidate_control: str, values: list[float]) -> dict:
    return {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "result_stage": "HPO_VALIDATION_ONLY",
        "evaluation_outcomes_read": 0,
        "development_test_outcomes_evaluated": False,
        "test_metrics": None,
        "loss_kind": "huber",
        "model_kind": kind,
        "candidate_control": candidate_control,
        "seed": 17,
        "trainable_parameter_count": 9_342_914,
        "validation_metrics": {
            "task_macro_spearman": sum(values) / len(values),
            "task_macro_standardized_mae": 1.5,
            "task_metrics": {
                task: {"spearman": value} for task, value in zip(TASKS, values)
            },
        },
    }


def _inputs():
    import json

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    comparison = {
        "selected_loss_for_controls": "huber",
        "development_test_opened": False,
        "evaluation_opened": False,
    }
    primary = _summary(
        kind="delta_pretrained_mrnabert_edit_centered_antisymmetric",
        candidate_control="NONE",
        values=[0.30, 0.20, 0.40, 0.15, 0.10, 0.05, 0.18, 0.12, 0.08],
    )
    permutation = _summary(
        kind="delta_pretrained_mrnabert_edit_centered_antisymmetric",
        candidate_control="WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION",
        values=[0.10, 0.21, 0.20, 0.14, 0.11, 0.04, 0.17, 0.10, 0.07],
    )
    source = _summary(
        kind="delta_pretrained_mrnabert_edit_centered_source_only_control",
        candidate_control="NONE",
        values=[0.10, 0.10, 0.20, 0.05, 0.02, 0.01, 0.08, 0.02, 0.01],
    )
    return protocol, comparison, primary, permutation, source


def test_controls_support_final_seed_confirmation() -> None:
    module = _load()
    result = module.adjudicate(*_inputs())
    assert result["supports_final_seed_confirmation"] is True
    assert result["source_only_control"]["task_win_count"] >= 5
    assert result["candidate_permutation_control"]["eligible_task_win_count"] == 2
    assert result["development_test_opened"] is False
    assert result["evaluation_opened"] is False
    assert result["guided_generation_authorized"] is False


def test_failure_against_each_control_blocks_confirmation() -> None:
    module = _load()
    for arm in ("baseline", "source", "permutation"):
        protocol, comparison, primary, permutation, source = _inputs()
        if arm == "baseline":
            protocol["strongest_same_information_baseline"]["task_macro_spearman"] = 0.9
        elif arm == "source":
            source["validation_metrics"]["task_metrics"] = deepcopy(
                primary["validation_metrics"]["task_metrics"]
            )
            source["validation_metrics"]["task_macro_spearman"] = primary[
                "validation_metrics"
            ]["task_macro_spearman"]
        else:
            for task in (
                "MEAN_RIBOSOME_LOAD::region=0",
                "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1",
            ):
                permutation["validation_metrics"]["task_metrics"][task]["spearman"] = 0.9
            values = [
                row["spearman"]
                for row in permutation["validation_metrics"]["task_metrics"].values()
            ]
            permutation["validation_metrics"]["task_macro_spearman"] = sum(values) / len(values)
        result = module.adjudicate(protocol, comparison, primary, permutation, source)
        assert result["supports_final_seed_confirmation"] is False


def test_test_or_evaluation_access_is_rejected() -> None:
    module = _load()
    protocol, comparison, primary, permutation, source = _inputs()
    primary["evaluation_outcomes_read"] = 1
    try:
        module.adjudicate(protocol, comparison, primary, permutation, source)
    except module.SignalControlError:
        pass
    else:
        raise AssertionError("Evaluation-contaminated control adjudication was accepted")
