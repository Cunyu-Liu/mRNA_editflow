from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/run_route2_classical_prediction_baselines_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_route2_classical_prediction_baselines_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(index: int, source_value, candidate_value):
    source = "A" * 50
    candidate = source[:index] + "C" + source[index + 1:]
    return {
        "canonical_record_id": f"R{index}",
        "study_unit_id": "S",
        "region": "3UTR",
        "endpoint_id": "E",
        "assay_id": "A",
        "biological_context_id": "C",
        "source_id": f"SOURCE{index}",
        "source_sequence": source,
        "candidate_sequence": candidate,
        "edit_operations": [{"type": "SUB", "position_zero_based": index, "ref": "A", "alt": "C"}],
        "target": float(index) / 10.0,
        "source_endpoint_value": source_value,
        "candidate_endpoint_value": candidate_value,
    }


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for label-fitted classical baseline tests")
    return torch.device(f"cuda:{int(os.environ.get('ROUTE2_TEST_CUDA_INDEX', '0'))}")


def test_feature_modes_keep_candidate_only_separate_from_source_relative_inputs() -> None:
    module = _module()
    record = _record(3, 0.0, 0.3)
    candidate_only = module._numeric_features(record, "candidate_only")
    source_only = module._numeric_features(record, "source_only")
    source_centered = module._numeric_features(record, "source_centered")
    full = module._numeric_features(record, "full")
    assert len(candidate_only) == 84
    assert len(source_only) == 84
    assert len(source_centered) == 84 + 84 + 17
    assert len(full) == 84 * 3 + 17
    assert module._kmer_vector(record["candidate_sequence"]) is module._kmer_vector(record["candidate_sequence"])
    assert len(module._numeric_features(record, "edit_position_only")) == 5
    assert len(module._numeric_features(record, "ref_alt_only")) == 12
    assert len(module._numeric_features(record, "context_only")) == 0


def test_study_then_source_group_weights_equalize_both_levels() -> None:
    module = _module()
    rows = []
    for study, source, count in (("A", "A1", 4), ("A", "A2", 2), ("B", "B1", 1)):
        for index in range(count):
            rows.append({
                "study_unit_id": study,
                "source_id": source,
                "biological_context_id": "C",
                "endpoint_id": "E",
                "row": index,
            })
    weights = module._training_weights(rows, "STUDY_THEN_SOURCE_GROUP_EQUAL")
    assert weights is not None
    assert float(weights.mean()) == pytest.approx(1.0)
    study_totals = {
        study: sum(weight for row, weight in zip(rows, weights) if row["study_unit_id"] == study)
        for study in ("A", "B")
    }
    assert study_totals["A"] == pytest.approx(study_totals["B"])
    group_totals = {
        source: sum(weight for row, weight in zip(rows, weights) if row["source_id"] == source)
        for source in ("A1", "A2", "B1")
    }
    assert group_totals["A1"] == pytest.approx(group_totals["A2"])


def test_gc_mfe_motif_uses_real_vienna_mfe_and_is_cached() -> None:
    module = _module()
    record = _record(3, 0.0, 0.3)
    values = module._numeric_features(record, "gc_mfe_motif")
    assert len(values) == (2 + len(module.MOTIFS)) * 3
    assert np.isfinite(values).all()
    assert module._gc_mfe_motif_vector(record["source_sequence"]) is module._gc_mfe_motif_vector(record["source_sequence"])


def test_absolute_difference_excludes_missing_endpoints_instead_of_imputing_zero() -> None:
    module = _module()
    train = [
        _record(1, 1.0, 1.2),
        _record(2, 2.0, 2.5),
        _record(3, None, None),
    ]
    predictions, _artifact = module._fit_predict_absolute(
        train,
        train,
        {"alpha": 1.0, "minimum_complete_training_records": 2},
        _cuda_device(),
    )
    assert len(predictions) == 3
    assert np.isfinite(predictions).all()
    with pytest.raises(module.BaselineError, match="too small"):
        module._fit_predict_absolute(
            train,
            train,
            {"alpha": 1.0, "minimum_complete_training_records": 3},
            _cuda_device(),
        )


def test_xgboost_refuses_to_claim_all_server_cores(monkeypatch) -> None:
    module = _module()
    records = [_record(index, float(index), float(index) + 0.1) for index in range(1, 5)]
    with pytest.raises(module.BaselineError, match="thread cap"):
        module._fit_predict_xgboost(records, records, {
            "n_estimators": 2,
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_jobs": 96,
            "seed": 1,
        }, torch.device("cuda:0"))


def test_group_means_fall_back_without_cross_split_source_leakage() -> None:
    module = _module()
    train = [_record(1, 1.0, 1.1), _record(2, 2.0, 2.1)]
    target = [_record(3, 3.0, 3.1)]
    predictions, artifact = module._fit_predict_group_mean(
        train, target, ["study_unit_id", "source_id"], _cuda_device()
    )
    assert predictions.tolist() == pytest.approx([0.15])
    assert artifact["global_mean"] == pytest.approx(0.15)


def test_candidate_permutation_stays_within_source_pool() -> None:
    module = _module()
    records = [_record(index, float(index), float(index) + 0.1) for index in range(1, 5)]
    for record in records:
        record["source_id"] = "SHARED"
    permuted, changed = module._permute_candidates_within_source(records, seed=7)
    assert changed > 0
    assert {record["candidate_sequence"] for record in permuted} == {record["candidate_sequence"] for record in records}
    assert all(record["source_sequence"] == records[0]["source_sequence"] for record in permuted)


def test_prediction_writer_preserves_exact_validation_membership(tmp_path: Path) -> None:
    module = _module()
    records = [_record(1, 1.0, 1.1), _record(2, 2.0, 2.1)]
    path = tmp_path / "validation_predictions.jsonl"
    module._write_predictions(path, "ridge", records, np.asarray([0.1, 0.2]))
    import json
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["canonical_record_id"] for row in rows] == ["R1", "R2"]
    assert all(row["baseline_id"] == "ridge" for row in rows)


def test_manifest_loader_keeps_component_metadata_for_loso(tmp_path: Path) -> None:
    module = _module()
    import json
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({
        "canonical_record_id": "r", "pool_assignment": "DEVELOPMENT", "split": "TRAIN",
        "study_unit_id": "S", "connected_source_component_id": "COMP",
    }) + "\n")
    assert module.load_manifest(path)["r"] == {
        "split": "TRAIN", "study_unit_id": "S", "connected_source_component_id": "COMP"
    }


def test_source_group_equal_weights_do_not_let_dense_groups_dominate() -> None:
    module = _module()
    records = [_record(index, 0.0, 0.1) for index in range(1, 5)]
    records[0]["source_id"] = "SMALL"
    for record in records[1:]:
        record["source_id"] = "DENSE"
    weights = module._training_weights(records, "SOURCE_GROUP_EQUAL")
    assert weights is not None
    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(weights[1:].sum())


def test_task_macro_spearman_requires_every_task_to_be_defined() -> None:
    module = _module()
    records = []
    predictions = []
    for endpoint, values in (("E1", [0.0, 1.0, 2.0]), ("E2", [2.0, 1.0, 0.0])):
        for index, target in enumerate(values):
            records.append({"target": target, "region": "3UTR", "endpoint_id": endpoint})
            predictions.append(float(index))
    assert module._task_macro_spearman(records, np.asarray(predictions)) == pytest.approx(0.0)
    predictions[-3:] = [1.0, 1.0, 1.0]
    assert module._task_macro_spearman(records, np.asarray(predictions)) is None


def test_cuda_preflight_refuses_cpu_device_without_fitting(monkeypatch) -> None:
    module = _module()
    with pytest.raises(module.BaselineError, match="explicit CUDA"):
        module.require_cuda_device("cpu", 0)


def test_cuda_ridge_fit_and_prediction_remain_on_gpu() -> None:
    module = _module()
    device = _cuda_device()
    x_train = torch.tensor([[0.0], [1.0], [2.0]], device=device)
    y_train = torch.tensor([0.0, 1.0, 2.0], device=device)
    prediction, artifact = module._ridge_cuda(
        x_train,
        y_train,
        x_train,
        alpha=0.1,
        sample_weight=torch.ones(3, device=device),
    )
    assert prediction.is_cuda and prediction.device == device
    assert artifact["optimizer"] == "CUDA_WEIGHTED_NORMAL_EQUATION"


def test_cuda_elastic_net_fit_and_prediction_remain_on_gpu() -> None:
    module = _module()
    device = _cuda_device()
    x_train = torch.arange(20, dtype=torch.float32, device=device).reshape(-1, 1) / 10.0
    y_train = 0.25 + 0.75 * x_train[:, 0]
    prediction, artifact = module._elastic_net_cuda(
        x_train,
        y_train,
        x_train,
        alpha=1e-3,
        l1_ratio=0.5,
        sample_weight=torch.ones(len(x_train), device=device),
        max_iter=2000,
        tolerance=1e-5,
        seed=7,
    )
    assert prediction.is_cuda and prediction.device == device
    assert artifact["optimizer"] == "CUDA_FISTA"
    assert artifact["converged"] is True


def test_frozen_ridge_artifact_predicts_without_targets_or_refitting(tmp_path: Path) -> None:
    module = _module()
    device = _cuda_device()
    train = [_record(index, float(index), float(index) + 0.1) for index in range(1, 6)]
    baseline = {"kind": "ridge", "feature_mode": "full"}
    expected, artifact = module._fit_predict_linear(
        train, train, "full", "ridge", {"alpha": 1.0}, device
    )
    target = [{key: value for key, value in record.items() if key != "target"} for record in train]
    artifact_path = tmp_path / "ridge.joblib"
    artifact = module.bind_artifact_identity({**artifact}, {**baseline, "baseline_id": "ridge"})
    module.joblib.dump(artifact, artifact_path)
    reloaded = module.joblib.load(artifact_path)
    observed = module.predict_from_frozen_artifact(baseline, reloaded, target, device)
    assert observed == pytest.approx(expected)
    assert "encoder" not in artifact
    assert artifact["encoder_spec"]["type"] == "FEATURE"
    assert reloaded["artifact_baseline_id"] == "ridge"


def test_frozen_group_mean_uses_global_value_for_unseen_evaluation_context() -> None:
    module = _module()
    device = _cuda_device()
    train = [_record(1, 1.0, 1.1), _record(2, 2.0, 2.1)]
    predictions, artifact = module._fit_predict_group_mean(
        train, train, ["study_unit_id", "source_id"], device
    )
    assert np.isfinite(predictions).all()
    target = [{key: value for key, value in _record(3, 3.0, 3.1).items() if key != "target"}]
    target[0]["study_unit_id"] = "UNSEEN"
    observed = module.predict_from_frozen_artifact(
        {"kind": "group_mean"}, artifact, target, device
    )
    assert observed.tolist() == pytest.approx([artifact["global_mean"]])


def test_xgboost_reports_actual_cuda_booster_device() -> None:
    module = _module()
    device = _cuda_device()
    records = [_record(index, float(index), float(index) + 0.1) for index in range(1, 9)]
    predictions, artifact = module._fit_predict_xgboost(records, records, {
        "n_estimators": 2,
        "max_depth": 2,
        "learning_rate": 0.1,
        "n_jobs": 1,
        "seed": 1,
    }, device)
    assert np.isfinite(predictions).all()
    assert artifact["booster_device"] == str(device)


def test_top_level_completion_status_cannot_hide_not_run_baselines() -> None:
    module = _module()
    complete = {"ridge": {"status": "COMPLETED_DEVELOPMENT_ONLY"}}
    assert module._completion_summary(complete, "FIXED_GROUPED_SPLIT") == {
        "status": "CLASSICAL_DEVELOPMENT_BASELINES_COMPLETED",
        "completed_baseline_count": 1,
        "not_run_baseline_count": 0,
    }
    partial = {**complete, "absolute": {"status": "NOT_RUN"}}
    assert module._completion_summary(partial, "FIXED_GROUPED_SPLIT")["status"].endswith("PARTIAL_WITH_NOT_RUN")
    with pytest.raises(module.BaselineError, match="no classical baseline completed"):
        module._completion_summary({"ridge": {"status": "NOT_RUN"}}, "FIXED_GROUPED_SPLIT")


def test_completion_recognizes_validation_and_frozen_test_stages() -> None:
    module = _module()
    for status in (
        "COMPLETED_DEVELOPMENT_VALIDATION_ONLY",
        "COMPLETED_FROZEN_DEVELOPMENT_TEST",
        "COMPLETED_DEVELOPMENT_LOSO",
    ):
        result = module._completion_summary({"ridge": {"status": status}}, "FIXED_GROUPED_SPLIT")
        assert result["completed_baseline_count"] == 1
        assert result["not_run_baseline_count"] == 0


def test_classical_hpo_persists_live_progress_and_completed_artifacts(tmp_path: Path) -> None:
    module = _module()
    device = _cuda_device()
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "canonical.jsonl"
    manifest_rows = []
    canonical_rows = []
    for index, split in enumerate(("TRAIN", "VALIDATION", "TEST"), start=1):
        record = _record(index, float(index), float(index) + 0.1)
        manifest_rows.append({
            "canonical_record_id": record["canonical_record_id"],
            "pool_assignment": "DEVELOPMENT",
            "split": split,
            "study_unit_id": "S",
            "connected_source_component_id": f"COMP{index}",
        })
        canonical_rows.append({
            **record,
            "pool_assignment": "DEVELOPMENT",
            "direction_normalized_delta": record.pop("target"),
        })
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    canonical.write_text("".join(json.dumps(row) + "\n" for row in canonical_rows))
    output = tmp_path / "classical_run"
    summary = module.execute({
        "evaluation_outcomes_accessed": False,
        "cpu_thread_cap": 1,
        "device": str(device),
        "physical_gpu_index": device.index,
        "minimum_free_gpu_memory_bytes": 0,
        "development_manifest_path": str(manifest),
        "canonical_paths": [str(canonical)],
        "run_mode": "FIXED_GROUPED_SPLIT",
        "result_stage": "HPO_VALIDATION_ONLY",
        "baselines": [{"baseline_id": "mean", "kind": "mean", "parameter_grid": {}}],
    }, output)
    assert summary["completed_baseline_count"] == 1
    for name in ("train.log", "metrics.jsonl", "config.yaml", "summary.json", "final_summary.json"):
        assert (output / name).is_file(), name
    assert "HPO_TRIAL_COMPLETED" in (output / "metrics.jsonl").read_text()
    assert (output / "mean/model.joblib").is_file()


def test_hpo_manifest_withholds_development_test_before_outcome_loading() -> None:
    module = _module()
    manifest = {
        "train": {"split": "TRAIN"},
        "validation": {"split": "VALIDATION"},
        "test": {"split": "TEST"},
    }
    selected, withheld = module.manifest_for_result_stage(
        manifest, "FIXED_GROUPED_SPLIT", "HPO_VALIDATION_ONLY"
    )
    assert set(selected) == {"train", "validation"}
    assert withheld == 1
    selected, withheld = module.manifest_for_result_stage(
        manifest, "FIXED_GROUPED_SPLIT", "FROZEN_DEVELOPMENT_TEST"
    )
    assert set(selected) == set(manifest)
    assert withheld == 0
    selected, withheld = module.manifest_for_result_stage(
        manifest,
        "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY",
        "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_PARAMETERS",
    )
    assert set(selected) == {"train", "validation"}
    assert withheld == 1
    with pytest.raises(module.BaselineError, match="invalid result_stage"):
        module.manifest_for_result_stage(manifest, "FIXED_GROUPED_SPLIT", "")
