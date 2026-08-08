from __future__ import annotations

import copy

import pytest

from scripts.r3x.estimands import build_primary_estimand_slices
from scripts.r3x.dataset import _synthetic_engineering_materialization_scope
from scripts.r3x.artifact_graph import replay_ensemble_manifest, replay_seed_commit
from scripts.r3x.models import ExactPotentialCritic, ModelConfig
from scripts.r3x.report import ReportProtocolError
from scripts.r3x.run_ordinary import (
    FROZEN_SEEDS,
    _aggregate_seed_runs,
    _baseline_methods_per_estimand,
    _canonical_partition_rows,
    _row_set_sha256,
    _train_neural_family_per_estimand,
    _validate_estimand_model_capacity,
    synthetic_rows,
)


@pytest.fixture(autouse=True)
def _synthetic_materialization_capability():
    with _synthetic_engineering_materialization_scope():
        yield


def _fixture():
    rows = synthetic_rows(24)
    entries = []
    for estimand_id, endpoint, indices, target in (
        ("estimand_a", "endpoint_a", range(0, 12), 10.0),
        ("estimand_b", "endpoint_b", range(12, 24), -10.0),
    ):
        entries.append({
            "study": "synthetic_study",
            "endpoint": endpoint,
            "estimand_id": estimand_id,
            "raw_source_field": "source_value",
            "raw_candidate_field": "candidate_value",
            "raw_unit": f"unit_{estimand_id}",
            "direction": "HIGHER_IS_MORE_STABLE",
            "transform": "IDENTITY_DIFFERENCE",
            "valid_domain": "FINITE",
            "role": "PRIMARY",
            "replicate_policy": "SINGLE_OBSERVATION_ONLY",
            "source_evidence_sha256": "e" * 64,
        })
        partitions = ("train", "development", "internal_test")
        for local_index, row_index in enumerate(indices):
            row = rows[row_index]
            row["endpoint"] = endpoint
            row["endpoint_id"] = endpoint
            row["estimand_id"] = estimand_id
            row["partition"] = partitions[local_index % 3]
            row["source_component_id"] = f"component_{row_index}"
            row["delta"] = target
            row["candidate_value"] = float(row["source_value"]) + target
    registry = {
        "schema_version": "r3_endpoint_estimand_registry_v1",
        "status": "FROZEN",
        "entries": entries,
    }
    slices = build_primary_estimand_slices(
        rows, registry, estimand_registry_sha256="a" * 64,
    )
    return rows, registry, slices


def test_capacity_and_scalar_baselines_are_independent_per_estimand() -> None:
    _, _, slices = _fixture()
    cfg = ModelConfig(
        d_model=8, n_heads=2, n_layers=1, ff_dim=16, max_len=8,
        n_endpoints=2, n_contexts=2, max_edits=5, pair_rank=2,
    )
    capacity = _validate_estimand_model_capacity(slices, cfg)
    assert capacity["status"] == "PASS_PER_ESTIMAND_TRAIN_ONLY_VOCAB_CAPACITY"
    test_rows = _canonical_partition_rows(slices, "internal_test")
    methods = _baseline_methods_per_estimand(slices, test_rows)
    prior = methods["SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR"]
    prediction_by_id = {
        row["r3_input_row_sha256"]: prediction
        for row, prediction in zip(test_rows, prior["predictions"])
    }
    for item in slices:
        expected_positive = item.estimand_id == "estimand_a"
        for row_id in item.row_ids_for("internal_test"):
            assert (prediction_by_id[row_id] > 0.0) is expected_positive
    assert prior["estimand_training_scope"] == "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND"


def _seed_run(seed: int, rows, *, reversed_ids: bool = False) -> dict:
    ids = [row["r3_input_row_sha256"] for row in rows]
    if reversed_ids:
        ids = list(reversed(ids))
    values = [0.25] * len(rows)
    return {
        "seed": seed,
        "architecture_id": "fixture_architecture",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "information_set_fingerprint": "f" * 64,
        "parameter_count": 100,
        "test_metrics": {"macro": {
            "macro_sign_accuracy": 0.5,
            "macro_delta_spearman": 0.25,
            "macro_rmse": 1.0,
            "macro_mae": 0.8,
            "macro_gaussian_nll": 1.2,
        }},
        "test_predictions": {
            "row_ids": ids,
            "row_set_sha256": _row_set_sha256(rows),
            "mean": values,
            "logvar": [0.0] * len(rows),
        },
    }


def test_seed_aggregation_rejects_shared_wrong_order_and_duplicate_seed() -> None:
    _, _, slices = _fixture()
    test_rows = _canonical_partition_rows(slices, "internal_test")
    wrong = [_seed_run(seed, test_rows, reversed_ids=True) for seed in FROZEN_SEEDS]
    with pytest.raises(ReportProtocolError, match="supplied test-row order"):
        _aggregate_seed_runs(wrong, test_rows)

    valid = [_seed_run(seed, test_rows) for seed in FROZEN_SEEDS]
    duplicate = copy.deepcopy(valid)
    duplicate[-1]["seed"] = 11
    with pytest.raises(ReportProtocolError, match="every frozen seed exactly once"):
        _aggregate_seed_runs(duplicate, test_rows)


def test_capacity_preflight_rejects_phantom_or_insufficient_condition_rows() -> None:
    _, _, slices = _fixture()
    too_large = ModelConfig(
        d_model=8, n_heads=2, n_layers=1, ff_dim=16, max_len=8,
        n_endpoints=3, n_contexts=2, max_edits=5, pair_rank=2,
    )
    with pytest.raises(ReportProtocolError, match="one UNK row"):
        _validate_estimand_model_capacity(slices, too_large)


def test_neural_family_trains_five_models_per_estimand_then_merges_by_row() -> None:
    _, _, slices = _fixture()
    test_rows = _canonical_partition_rows(slices, "internal_test")
    cfg = ModelConfig(
        d_model=8, n_heads=2, n_layers=1, ff_dim=16, max_len=8,
        n_endpoints=2, n_contexts=2, max_edits=5, pair_rank=2,
    )
    result = _train_neural_family_per_estimand(
        factory=lambda: ExactPotentialCritic(cfg),
        slices=slices,
        global_test_rows=test_rows,
        cfg=cfg,
        seeds=FROZEN_SEEDS,
        epochs=1,
        batch_size=4,
        device_name="cpu",
        condition_unknown_dropout=0.15,
        optimizer_config={
            "name": "Adam", "learning_rate": 0.002, "weight_decay": 0.0001,
        },
        loss_config={
            "huber_weight": 1.0, "nll_weight": 0.25,
            "sign_weight": 0.25, "sign_temperature": 1.0,
        },
    )
    assert result["models_per_row"] == 5
    assert result["total_fitted_models"] == 10
    assert result["n_primary_estimands"] == 2
    assert result["ensemble_prediction"]["row_ids"] == [
        row["r3_input_row_sha256"] for row in test_rows
    ]
    assert set(result["parameter_count_by_estimand"]) == {"estimand_a", "estimand_b"}


def test_artifact_mode_scores_and_ensembles_only_through_committed_bytes(
    tmp_path,
) -> None:
    _, _, slices = _fixture()
    test_rows = _canonical_partition_rows(slices, "internal_test")
    cfg = ModelConfig(
        d_model=8, n_heads=2, n_layers=1, ff_dim=16, max_len=8,
        n_endpoints=2, n_contexts=2, max_edits=5, pair_rank=2,
    )
    authority_names = (
        "route_authorization_sha256", "ready_dataset_sha256",
        "inclusion_manifest_sha256", "context_source_crosswalk_sha256",
        "endpoint_estimand_registry_sha256",
        "source_component_split_manifest_sha256",
        "metric_zero_policy_registry_sha256",
        "model_optimizer_seed_config_sha256",
        "analysis_code_tree_manifest_sha256", "method_protocol_sha256",
        "paired_strongest_baseline_comparison_policy_sha256",
    )
    context = {
        "run_root": tmp_path,
        "run_start_sha256": "1" * 64,
        "run_plan_sha256": "2" * 64,
        "activation_manifest_sha256": "3" * 64,
        "authority_hashes": {
            name: format(index + 1, "x") * 64
            for index, name in enumerate(authority_names)
        },
        "training_config_sha256": "d" * 64,
        "scientific_scope": "DEVELOPMENT_ONLY_FIXTURE",
    }
    result = _train_neural_family_per_estimand(
        factory=lambda: ExactPotentialCritic(cfg),
        slices=slices,
        global_test_rows=test_rows,
        cfg=cfg,
        seeds=FROZEN_SEEDS,
        epochs=1,
        batch_size=4,
        device_name="cpu",
        condition_unknown_dropout=0.15,
        optimizer_config={
            "name": "Adam", "learning_rate": 0.002, "weight_decay": 0.0001,
        },
        loss_config={
            "huber_weight": 1.0, "nll_weight": 0.25,
            "sign_weight": 0.25, "sign_temperature": 1.0,
        },
        method_id="A_EXACT_POTENTIAL_DIFFERENCE",
        artifact_context=context,
    )
    assert result["ensemble_prediction"]["source"] == (
        "EXACT_PER_ESTIMAND_ENSEMBLE_COMMIT_REPLAY"
    )
    for estimand_id, refs in result["ensemble_artifact_refs_by_estimand"].items():
        replayed = replay_ensemble_manifest(tmp_path, refs["manifest"])
        assert replayed["manifest"]["estimand_binding_sha256"] == next(
            item.partition_binding_sha256
            for item in slices if item.estimand_id == estimand_id
        )
    for seed_run in result["seed_runs"].values():
        for estimand_run in seed_run["estimand_runs"].values():
            replay_seed_commit(
                tmp_path, estimand_run["artifact_refs"]["seed_commit"],
                expected_seed=seed_run["seed"],
            )
