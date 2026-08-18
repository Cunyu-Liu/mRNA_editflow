from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/audit_route2_mrnabert_uncertainty_absorption_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("uncertainty_absorption_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def create_inputs(root: Path) -> tuple[list[Path], Path]:
    manifest = root / "manifest.jsonl"
    canonical = root / "canonical.jsonl"
    record_ids = [f"r{index}" for index in range(8)]
    write_jsonl(
        manifest,
        [
            {
                "canonical_record_id": record_id,
                "split": "VALIDATION",
            }
            for record_id in record_ids
        ],
    )
    write_jsonl(
        canonical,
        [
            {
                "canonical_record_id": record_id,
                "pool_assignment": "DEVELOPMENT",
                "direction_normalized_delta": float(index % 4),
                "endpoint_id": "endpoint_a" if index < 4 else "endpoint_b",
                "region": "5UTR" if index < 4 else "3UTR",
            }
            for index, record_id in enumerate(record_ids)
        ],
    )
    summaries = []
    loss_offsets = {
        "huber": 0.0,
        "fixed_variance_gaussian_nll": 0.1,
        "learned_variance_gaussian_nll": 0.2,
    }
    module = load_module()
    for loss, offset in loss_offsets.items():
        run = root / loss
        run.mkdir()
        config = {
            "development_manifest": str(manifest),
            "canonical_paths": [str(canonical)],
        }
        (run / "training_config.json").write_text(json.dumps(config))
        predictions = []
        targets = []
        means = []
        task_keys = []
        for index, record_id in enumerate(record_ids):
            target = float(index % 4)
            mean = target + offset * (-1 if index % 2 else 1)
            predictions.append(
                {
                    "canonical_record_id": record_id,
                    "predicted_standardized_delta": mean,
                    "target_scale": 1.0,
                    "predicted_variance": (0.25 + 0.1 * index) if loss.startswith("learned") else None,
                }
            )
            targets.append(target)
            means.append(mean)
            task_keys.append("endpoint_a::region=0" if index < 4 else "endpoint_b::region=1")
        write_jsonl(run / "validation_predictions.jsonl", predictions)
        metric = module.np.mean(
            [
                module.rank_correlation(module.np.asarray(targets[:4]), module.np.asarray(means[:4])),
                module.rank_correlation(module.np.asarray(targets[4:]), module.np.asarray(means[4:])),
            ]
        )
        standardized_mae = module.np.mean(module.np.abs(module.np.asarray(means) - module.np.asarray(targets)))
        summary = {
            "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
            "result_stage": "HPO_VALIDATION_ONLY",
            "evaluation_outcomes_read": 0,
            "test_metrics": None,
            "loss_kind": loss,
            "baseline_id": loss,
            "selected_epoch": 4,
            "validation_metrics": {
                "task_macro_spearman": float(metric),
                "task_macro_standardized_mae": float(standardized_mae),
            },
        }
        summary_path = run / "training_summary.json"
        summary_path.write_text(json.dumps(summary))
        summaries.append(summary_path)
    comparison = root / "comparison.json"
    comparison.write_text(json.dumps({"selected_loss_for_controls": "huber"}))
    return summaries, comparison


def test_audit_uses_task_standardized_spread_and_keeps_selection_fixed(tmp_path: Path) -> None:
    module = load_module()
    summaries, comparison = create_inputs(tmp_path)
    result = module.audit(summaries, comparison)
    assert result["status"].endswith("AUDIT_COMPLETE")
    assert result["selected_loss_for_controls"] == "huber"
    assert result["selection_rule_unchanged"] is True
    assert result["development_validation_record_count"] == 8
    learned = next(row for row in result["rows"] if row["loss_kind"].startswith("learned"))
    assert learned["task_count"] == 2
    assert learned["task_macro_uncertainty_absolute_residual_spearman"] is not None
    assert result["development_test_opened"] is False
    assert result["evaluation_opened"] is False


def test_audit_rejects_prediction_membership_drift(tmp_path: Path) -> None:
    module = load_module()
    summaries, comparison = create_inputs(tmp_path)
    rows = [json.loads(line) for line in (summaries[0].parent / "validation_predictions.jsonl").read_text().splitlines()]
    write_jsonl(summaries[0].parent / "validation_predictions.jsonl", rows[:-1])
    with pytest.raises(module.UncertaintyAuditError, match="exactly cover"):
        module.audit(summaries, comparison)


def test_audit_accepts_float32_metric_roundoff_but_rejects_material_drift(tmp_path: Path) -> None:
    module = load_module()
    summaries, comparison = create_inputs(tmp_path)
    summary_path = summaries[0]
    summary = json.loads(summary_path.read_text())
    recorded = summary["validation_metrics"]["task_macro_standardized_mae"]
    summary["validation_metrics"]["task_macro_standardized_mae"] = (
        recorded + module.FLOAT32_METRIC_RECONSTRUCTION_ATOL / 2
    )
    summary_path.write_text(json.dumps(summary))
    module.audit(summaries, comparison)

    summary["validation_metrics"]["task_macro_standardized_mae"] = (
        recorded + module.FLOAT32_METRIC_RECONSTRUCTION_ATOL * 2
    )
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(module.UncertaintyAuditError, match="standardized MAE"):
        module.audit(summaries, comparison)
