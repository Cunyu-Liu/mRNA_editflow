from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILDER = (
    ROOT
    / "scripts/route_a_v3/build_route2_mrnabert_critic_v2_loso_aggregation_inputs_v1.py"
)
AGGREGATOR = ROOT / "scripts/route_a_v3/aggregate_route2_loso_v1.py"
PRIMARY_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json"
)
BASELINE_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol_v1.json"
)
AGGREGATION_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_loso_aggregation_protocol_v1.json"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protocols(tmp_path: Path) -> tuple[dict, dict, dict]:
    primary = json.loads(PRIMARY_PROTOCOL.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_PROTOCOL.read_text(encoding="utf-8"))
    aggregation = json.loads(AGGREGATION_PROTOCOL.read_text(encoding="utf-8"))
    primary["run_root"] = str(tmp_path / "primary")
    baseline["run_root"] = str(tmp_path / "baseline")
    return primary, baseline, aggregation


def _summary(study: str, seed: int, gpu: int, model_kind: str, value: float) -> dict:
    return {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "model_kind": model_kind,
        "candidate_control": "NONE",
        "run_mode": "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY",
        "result_stage": "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS",
        "loso_holdout_study_unit_id": study,
        "loso_development_test_preserved": True,
        "development_test_outcomes_evaluated": False,
        "development_test_record_count_withheld": 18292,
        "test_metrics": None,
        "seed": seed,
        "optimizer_steps": 10,
        "parameter_changed": True,
        "cuda_training_tensors_verified": True,
        "physical_gpu_index": gpu,
        "device": f"cuda:{gpu}",
        "cpu_fallback_used": False,
        "cuda_device_index": gpu,
        "cuda_device_uuid": f"GPU-{gpu}",
        "cuda_total_memory_mb": 40960.0,
        "evaluation_outcomes_read": 0,
        "validation_metrics": {
            "task_count": 1,
            "task_spearman_defined_count": 1,
            "task_macro_spearman": value,
        },
    }


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _configs(tmp_path: Path):
    module = _load(BUILDER, "critic_v2_loso_input_fixture")
    primary_protocol, baseline_protocol, aggregation_protocol = _protocols(tmp_path)
    primary_configs = []
    baseline_configs = []
    for study in module.HOLDOUT_STUDIES:
        for seed in module.FINAL_SEEDS:
            gpu = module.assigned_gpu(study, seed)
            primary_output = Path(primary_protocol["run_root"]) / study / f"seed{seed}_gpu{gpu}"
            baseline_output = Path(baseline_protocol["run_root"]) / study / f"seed{seed}_gpu{gpu}"
            primary_id = f"mrnabert_critic_v2_loso_{study.lower()}_seed{seed}"
            primary_configs.append(
                {
                    "scientific_role": "CRITIC_V2_TEST_PRESERVING_CROSS_STUDY_TRANSFER",
                    "model_kind": module.PRIMARY_KIND,
                    "candidate_control": "NONE",
                    "run_mode": "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY",
                    "result_stage": "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS",
                    "development_record_scope": "TRAIN_VALIDATION_ONLY_TEST_WITHHELD",
                    "loso_holdout_study_unit_id": study,
                    "seed": seed,
                    "physical_gpu_index": gpu,
                    "device": f"cuda:{gpu}",
                    "output_directory": str(primary_output),
                    "baseline_id": primary_id,
                    "loso_protocol_schema_version": module.PRIMARY_PROTOCOL_SCHEMA,
                    "development_test_outcomes_accessed": False,
                    "test_metrics_used_for_loso_selection": False,
                    "evaluation_outcomes_accessed": False,
                }
            )
            baseline_configs.append(
                {
                    "scientific_role": "CRITIC_V2_STRONGEST_BASELINE_TEST_PRESERVING_LOSO",
                    "model_kind": module.BASELINE_KIND,
                    "candidate_control": "NONE",
                    "run_mode": "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY",
                    "result_stage": "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS",
                    "development_record_scope": "TRAIN_VALIDATION_ONLY_TEST_WITHHELD",
                    "loso_holdout_study_unit_id": study,
                    "seed": seed,
                    "physical_gpu_index": gpu,
                    "device": f"cuda:{gpu}",
                    "output_directory": str(baseline_output),
                    "baseline_id": f"global_scaled_critic_v2_loso_{study.lower()}_seed{seed}",
                    "matched_baseline_loso_protocol_schema_version": module.BASELINE_PROTOCOL_SCHEMA,
                    "paired_primary_baseline_id": primary_id,
                    "paired_primary_output_directory": str(primary_output),
                    "development_test_outcomes_accessed": False,
                    "test_metrics_used_for_loso_selection": False,
                    "evaluation_outcomes_accessed": False,
                }
            )
            _write_summary(
                primary_output / "training_summary.json",
                _summary(study, seed, gpu, module.PRIMARY_KIND, 0.2),
            )
            _write_summary(
                baseline_output / "training_summary.json",
                _summary(study, seed, gpu, module.BASELINE_KIND, 0.1),
            )
    return (
        module,
        primary_configs,
        baseline_configs,
        primary_protocol,
        baseline_protocol,
        aggregation_protocol,
    )


def test_builds_three_v2_inputs_accepted_by_shared_aggregator(tmp_path: Path) -> None:
    (
        module,
        primary_configs,
        baseline_configs,
        primary_protocol,
        baseline_protocol,
        aggregation_protocol,
    ) = _configs(tmp_path)
    aggregator = _load(AGGREGATOR, "critic_v2_loso_shared_aggregator")
    payloads = module.build_inputs(
        primary_configs,
        baseline_configs,
        primary_protocol,
        baseline_protocol,
        aggregation_protocol,
    )
    assert set(payloads) == set(module.FINAL_SEEDS)
    for seed, payload in payloads.items():
        result = aggregator.aggregate(payload)
        assert result["seed"] == seed
        assert result["status"] == "LOSO_MODEL_BASELINE_ALIGNED_COMPLETE"
        assert result["study_count"] == 7
        assert result["macro_improvement"] == pytest.approx(0.1)
        assert result["development_test_preserved"] is True
        assert result["evaluation_studies_included"] == 0


def test_rejects_primary_baseline_pairing_drift(tmp_path: Path) -> None:
    module, primary, baseline, primary_protocol, baseline_protocol, aggregation = _configs(tmp_path)
    baseline[0]["paired_primary_baseline_id"] = "wrong"
    with pytest.raises(module.CriticV2LosoInputError, match="pairing differs"):
        module.build_inputs(primary, baseline, primary_protocol, baseline_protocol, aggregation)


def test_rejects_nonterminal_or_protected_summary(tmp_path: Path) -> None:
    module, primary, baseline, primary_protocol, baseline_protocol, aggregation = _configs(tmp_path)
    summary_path = Path(primary[0]["output_directory"]) / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["development_test_outcomes_evaluated"] = True
    _write_summary(summary_path, summary)
    with pytest.raises(module.CriticV2LosoInputError, match="protected outcome"):
        module.build_inputs(primary, baseline, primary_protocol, baseline_protocol, aggregation)


def test_rejects_aggregation_protocol_drift(tmp_path: Path) -> None:
    module, primary, baseline, primary_protocol, baseline_protocol, aggregation = _configs(tmp_path)
    aggregation["required_seeds"] = [20260822, 20260823, 999]
    with pytest.raises(module.CriticV2LosoInputError, match="seed or study"):
        module.build_inputs(primary, baseline, primary_protocol, baseline_protocol, aggregation)


def test_write_inputs_once_refuses_existing_input_or_result_root(tmp_path: Path) -> None:
    module = _load(BUILDER, "critic_v2_loso_input_writer")
    payloads = {seed: {"seed": seed} for seed in module.FINAL_SEEDS}
    input_root = tmp_path / "inputs"
    result_root = tmp_path / "results"
    paths = module.write_inputs_once(payloads, input_root, result_root)
    assert len(paths) == 3
    with pytest.raises(module.CriticV2LosoInputError, match="input root already exists"):
        module.write_inputs_once(payloads, input_root, result_root)

    second_input = tmp_path / "inputs-second"
    result_root.mkdir()
    with pytest.raises(module.CriticV2LosoInputError, match="output root already exists"):
        module.write_inputs_once(payloads, second_input, result_root)
