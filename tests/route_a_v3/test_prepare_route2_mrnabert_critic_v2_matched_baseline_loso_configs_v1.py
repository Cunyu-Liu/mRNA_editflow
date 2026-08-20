from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_matched_baseline_loso_configs_v1.py"
)
BASE_CONFIG = (
    ROOT / "configs/route_a_v3_route2_method_repair_global_scaled_seed20260821_gpu0_v1.json"
)
PRIMARY_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json"
)
BASELINE_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol_v1.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "critic_v2_matched_baseline_loso_prepare_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(module) -> tuple[dict, list[dict], dict, dict]:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    primary_protocol = json.loads(PRIMARY_PROTOCOL.read_text(encoding="utf-8"))
    baseline_protocol = json.loads(BASELINE_PROTOCOL.read_text(encoding="utf-8"))
    run_root = Path(primary_protocol["run_root"])
    primary_configs = []
    for study, seed, gpu in module.loso_assignments():
        config = dict(primary_protocol["frozen_loso_training_policy"])
        config.update(
            {
                "scientific_role": "CRITIC_V2_TEST_PRESERVING_CROSS_STUDY_TRANSFER",
                "result_stage": "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS",
                "run_mode": "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY",
                "model_kind": module.PRIMARY_KIND,
                "candidate_control": "NONE",
                "baseline_id": f"mrnabert_critic_v2_loso_{study.lower()}_seed{seed}",
                "loso_holdout_study_unit_id": study,
                "seed": seed,
                "physical_gpu_index": gpu,
                "device": f"cuda:{gpu}",
                "output_directory": str(
                    run_root / study / f"seed{seed}_gpu{gpu}"
                ),
                "loso_protocol_schema_version": primary_protocol["schema_version"],
                "development_record_scope": "TRAIN_VALIDATION_ONLY_TEST_WITHHELD",
                "development_test_outcomes_accessed": False,
                "test_metrics_used_for_loso_selection": False,
                "all_development_refit_completed_before_loso": True,
                "evaluation_outcomes_accessed": False,
            }
        )
        primary_configs.append(config)
    return base, primary_configs, primary_protocol, baseline_protocol


def test_builds_exact_native_baseline_configs_paired_to_every_primary_fold() -> None:
    module = _load()
    base, primary_configs, primary_protocol, baseline_protocol = _inputs(module)
    configs = module.build_configs(
        base, primary_configs, primary_protocol, baseline_protocol
    )

    assert len(configs) == 21
    for baseline, primary, assignment in zip(
        configs, primary_configs, module.loso_assignments()
    ):
        study, seed, gpu = assignment
        assert (
            baseline["loso_holdout_study_unit_id"],
            baseline["seed"],
            baseline["physical_gpu_index"],
        ) == (study, seed, gpu)
        assert baseline["paired_primary_baseline_id"] == primary["baseline_id"]
        assert (
            baseline["paired_primary_output_directory"]
            == primary["output_directory"]
        )
        assert baseline["device"] == primary["device"]
        assert baseline["run_mode"] == primary["run_mode"]
        assert baseline["result_stage"] == primary["result_stage"]
        assert baseline["development_test_outcomes_accessed"] is False
        assert baseline["test_metrics_used_for_loso_selection"] is False
        assert baseline["evaluation_outcomes_accessed"] is False
        assert baseline["checkpoint_selection"] == "FINAL_EPOCH"
        assert baseline["epochs"] == 8
        assert baseline["training_precision"] == "FP32"
        assert baseline["loss_kind"] == "huber_plus_pairwise"
        assert baseline["model_kind"] == module.BASELINE_KIND


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("baseline_id", "substitute", "substituted"),
        ("model_kind", "other", "model kind differs"),
        ("loss_kind", "huber", "baseline frozen policy differs: loss_kind"),
        ("development_test_outcomes_accessed", True, "Development TEST entered"),
        ("evaluation_outcomes_accessed", True, "Evaluation entered"),
    ],
)
def test_rejects_baseline_substitution_policy_drift_or_contamination(
    field: str, value: object, match: str
) -> None:
    module = _load()
    inputs = list(_inputs(module))
    inputs[0][field] = value
    with pytest.raises(
        module.CriticV2MatchedBaselineLosoPreparationError, match=match
    ):
        module.build_configs(*inputs)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "exactly 21"),
        ("duplicate", "duplicated"),
        ("gpu", "primary LOSO GPU differs"),
        ("test", "primary LOSO TEST boundary differs"),
        ("refit", "bypassed the refit gate"),
    ],
)
def test_rejects_missing_duplicate_or_contaminated_primary_pairing(
    mutation: str, match: str
) -> None:
    module = _load()
    inputs = list(_inputs(module))
    if mutation == "missing":
        inputs[1].pop()
    elif mutation == "duplicate":
        inputs[1][-1] = dict(inputs[1][0])
    elif mutation == "gpu":
        inputs[1][0]["physical_gpu_index"] = 5
    elif mutation == "test":
        inputs[1][0]["development_test_outcomes_accessed"] = True
    else:
        inputs[1][0]["all_development_refit_completed_before_loso"] = False
    with pytest.raises(
        module.CriticV2MatchedBaselineLosoPreparationError, match=match
    ):
        module.build_configs(*inputs)


def test_rejects_protocol_pairing_drift() -> None:
    module = _load()
    inputs = list(_inputs(module))
    inputs[3]["pairing_policy"] = "STUDY_ONLY"
    with pytest.raises(
        module.CriticV2MatchedBaselineLosoPreparationError,
        match="pairing policy differs",
    ):
        module.build_configs(*inputs)


@pytest.mark.parametrize("existing_target", ["config_root", "run"])
def test_write_configs_once_refuses_existing_targets(
    tmp_path: Path, existing_target: str
) -> None:
    module = _load()
    config_root = tmp_path / "runtime"
    run_directory = tmp_path / "runs" / "GSE200304" / "seed20260822_gpu0"
    configs = [
        {
            "baseline_id": "global_scaled_critic_v2_loso_gse200304_seed20260822",
            "output_directory": str(run_directory),
        }
    ]
    if existing_target == "config_root":
        config_root.mkdir()
        match = "config root already exists"
    else:
        run_directory.mkdir(parents=True)
        match = "run directory already exists"
    with pytest.raises(
        module.CriticV2MatchedBaselineLosoPreparationError, match=match
    ):
        module.write_configs_once(configs, config_root)


def test_write_configs_once_writes_all_without_creating_runs(tmp_path: Path) -> None:
    module = _load()
    configs = module.build_configs(*_inputs(module))
    config_root = tmp_path / "runtime"
    rewritten = []
    for config in configs:
        row = dict(config)
        row["output_directory"] = str(
            tmp_path
            / "runs"
            / row["loso_holdout_study_unit_id"]
            / f"seed{row['seed']}_gpu{row['physical_gpu_index']}"
        )
        rewritten.append(row)
    paths = module.write_configs_once(rewritten, config_root)
    assert len(paths) == 21
    assert len(list(config_root.glob("*.json"))) == 21
    assert all(not Path(row["output_directory"]).exists() for row in rewritten)
