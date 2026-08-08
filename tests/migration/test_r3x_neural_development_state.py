from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import scripts.r3x.neural_development_state as neural_state
from scripts.r3x.information_sets import information_set_fingerprint
from scripts.r3x.models import ModelConfig
from scripts.r3x.neural_development_state import (
    NeuralDevelopmentFitConfig,
    NeuralDevelopmentStateError,
    build_neural_source_component_assignment,
    fit_neural_development_five_seed_ensemble,
    fit_neural_development_state,
    load_neural_development_state_exact,
    load_neural_source_component_assignment_exact,
    rebuild_and_match_neural_development_state,
    recompute_neural_development_predictions,
    save_neural_development_state_exclusive,
    validate_neural_development_ensemble,
    validate_neural_development_state,
    write_neural_source_component_assignment_exclusive,
)
from scripts.r3x.neural_method_registry import (
    A0_CONTROL_METHOD_ID,
    A0_METHOD_ID,
    A1_CONTROL_METHOD_ID,
    A1_METHOD_ID,
    A1_UNK_ONLY_METHOD_ID,
    A1_WRONG_LOCAL_METHOD_ID,
    B_ADDITIVE_METHOD_ID,
    B_MATCHED_CONTROL_METHOD_ID,
    B_UNK_ONLY_METHOD_ID,
    B_WRONG_LOCAL_METHOD_ID,
    EDIT_AWARE_FULL_PAIR_METHOD_ID,
    FULL_PAIR_METHOD_ID,
    registry_sha256,
)
from scripts.r3x.predictor_provenance import (
    DEVELOPMENT_PARTITION,
    FROZEN_NEURAL_SEEDS,
    NON_DATA_AUTHORITY_HASH_KEYS,
    PREDICTOR_INPUT_ROW_SCHEMA,
    TRAIN_PARTITION,
)


RUN_SHA = "1" * 64
ACTIVATION_SHA = "2" * 64
OUTCOMES_SHA = "3" * 64
ESTIMAND = "ESTIMAND_A"


def _cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=5,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        max_len=8,
        n_endpoints=3,
        n_contexts=3,
        max_edits=5,
        pair_rank=2,
    )


def _fit_config(**updates) -> NeuralDevelopmentFitConfig:
    values = {
        "epochs": 1,
        "batch_size": 3,
        "learning_rate": 2e-3,
        "weight_decay": 1e-4,
        "huber_weight": 1.0,
        "nll_weight": 0.25,
        "sign_weight": 0.25,
        "sign_temperature": 1.0,
        "gradient_clip_norm": 5.0,
        "condition_unknown_dropout": 0.15,
        "max_rows_per_fit": 100,
        "max_sequence_tokens_per_batch": 10_000,
        "max_forward_equivalents_per_fit": 10_000,
        "max_total_forward_equivalents_five_seeds": 50_000,
        "max_peak_parameter_bytes_per_fit": 200_000_000,
        "max_wall_time_seconds_per_fit": 120.0,
        "device": "cpu",
    }
    values.update(updates)
    return NeuralDevelopmentFitConfig(**values)


def _authorities() -> dict[str, str]:
    return {
        name: hashlib.sha256(f"authority/{name}".encode()).hexdigest()
        for name in NON_DATA_AUTHORITY_HASH_KEYS
    }


def _edits(source: str, positions: tuple[int, ...]) -> tuple[str, list[dict]]:
    candidate = list(source)
    edits = []
    next_base = {"A": "C", "C": "G", "G": "U", "U": "A"}
    for position in positions:
        ref = source[position]
        alt = next_base[ref]
        candidate[position] = alt
        edits.append({"op": "SUB", "pos": position, "ref": ref, "alt": alt})
    return "".join(candidate), edits


def _row(
    partition: str,
    index: int,
    *,
    cell: str,
    positions: tuple[int, ...],
    target: float | None,
    source: str = "ACGUACGU",
) -> dict:
    candidate, edits = _edits(source, positions)
    return {
        "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
        "row_id": hashlib.sha256(f"{partition}/{index}".encode()).hexdigest(),
        "partition": partition,
        "estimand_id": ESTIMAND,
        "cell_id": cell,
        "predictor_input": {
            "endpoint": "mrna_abundance",
            "publication_context_id": "cell_line_a|assay_a|library_a",
            "region": "5UTR",
            "normalized_source_sequence": source,
            "normalized_candidate_sequence": candidate,
            "edit_list": edits,
            "edit_count": len(edits),
        },
        "training_target": target,
    }


def _fixture_rows(
    *, source: str = "ACGUACGU"
) -> tuple[list[dict], list[dict], dict[str, float]]:
    budgets = ((0,), (0, 2, 4), (0, 1, 2, 3, 4))
    targets = (-1.0, 0.5, 1.5)
    train = []
    development = []
    dev_targets = {}
    for cell_index, cell in enumerate(("ESTIMAND_A:CELL_A", "ESTIMAND_A:CELL_B")):
        for offset, (positions, target) in enumerate(zip(budgets, targets, strict=True)):
            index = 3 * cell_index + offset
            train.append(
                _row(
                    TRAIN_PARTITION,
                    index,
                    cell=cell,
                    positions=positions,
                    target=target + 0.1 * cell_index,
                    source=source,
                )
            )
            dev = _row(
                DEVELOPMENT_PARTITION,
                index,
                cell=cell,
                positions=positions,
                target=None,
                source=source,
            )
            development.append(dev)
            dev_targets[dev["row_id"]] = target - 0.1 * cell_index
    return train, development, dev_targets


def _assignment(train: list[dict], development: list[dict]) -> dict:
    authorities = _authorities()
    assignments = []
    for index, row in enumerate((*train, *development)):
        assignments.append({
            "row_id": row["row_id"],
            "partition": row["partition"],
            "source_component_id": f"GLOBAL_COMPONENT_{index % 4}",
        })
    return build_neural_source_component_assignment(
        train,
        development,
        assignments,
        information_set_fingerprint_sha256=information_set_fingerprint(
            "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
        ),
        train_development_outcomes_sha256=OUTCOMES_SHA,
        non_data_authority_hashes=authorities,
        source_component_split_manifest_source_sha256=authorities[
            "source_component_split_manifest_sha256"
        ],
        source_component_split_manifest_projection_sha256="4" * 64,
    )


def _fit(method_id: str = FULL_PAIR_METHOD_ID, *, seed: int = 11) -> dict:
    train, development, targets = _fixture_rows()
    return fit_neural_development_state(
        method_id,
        _cfg(),
        _fit_config(),
        train,
        development,
        targets,
        _assignment(train, development),
        seed=seed,
        expected_registry_sha256=registry_sha256(),
        run_start_sha256=RUN_SHA,
        development_activation_sha256=ACTIVATION_SHA,
    )


def _fit_wrong_local(method_id: str, *, seed: int = 11) -> dict:
    train, development, targets = _fixture_rows(source="ACGU" * 5)
    cfg = ModelConfig(
        vocab_size=5,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        max_len=20,
        n_endpoints=3,
        n_contexts=3,
        max_edits=5,
        pair_rank=2,
    )
    return fit_neural_development_state(
        method_id,
        cfg,
        _fit_config(),
        train,
        development,
        targets,
        _assignment(train, development),
        seed=seed,
        expected_registry_sha256=registry_sha256(),
        run_start_sha256=RUN_SHA,
        development_activation_sha256=ACTIVATION_SHA,
    )


@pytest.fixture(scope="module")
def fitted_state() -> dict:
    return _fit()


def test_fullpair_state_binds_rows_actions_sampler_registry_and_no_future_stage(
    fitted_state: dict,
) -> None:
    state = validate_neural_development_state(fitted_state)
    assert state["method_id"] == FULL_PAIR_METHOD_ID
    assert state["registry_sha256"] == registry_sha256()
    assert state["information_set"] == "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
    assert state["allowed_edit_counts"] == [1, 3, 5]
    assert state["action_policy"].endswith("EDIT_COUNT_1_3_5_V1")
    assert state["checkpoint_selection"]["sign_gate"] == 0.60
    assert state["ordinary_only"] is True
    assert state["internal_test_access"] is False
    assert state["sealed_access"] is False
    assert state["downstream_unlock"] is False
    assert state["execution_authorized"] is False
    assert state["independent_authority_verified"] is False
    assert state["training_ready"] is False
    assert state["publication_evidence"] is False
    assert state["resource_preflight"]["status"] == "PASS_BEFORE_MODEL_CONSTRUCTION"
    assert state["compiled_binding"]["parameter_match"]["matched"] is True
    assert len(state["sampler_epoch_manifests"]) == 1
    assert state["sampler_epoch_manifests"][0]["n_cells"] == 2


def test_component_assignment_exact_write_load_is_bound_to_both_partitions(
    tmp_path: Path,
) -> None:
    train, development, _ = _fixture_rows()
    assignment = _assignment(train, development)
    authorities = _authorities()
    development_dir = tmp_path / "development" / "assignments"
    development_dir.mkdir(parents=True)
    path = development_dir / "components.json"
    kwargs = {
        "expected_train_rows": train,
        "expected_development_rows": development,
        "expected_information_set_fingerprint": information_set_fingerprint(
            "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
        ),
        "expected_train_development_outcomes_sha256": OUTCOMES_SHA,
        "expected_non_data_authority_hashes": authorities,
        "expected_source_component_split_manifest_source_sha256": authorities[
            "source_component_split_manifest_sha256"
        ],
        "expected_source_component_split_manifest_projection_sha256": "4" * 64,
    }
    digest = write_neural_source_component_assignment_exclusive(
        path, assignment, **kwargs
    )
    loaded = load_neural_source_component_assignment_exact(
        path, digest, **kwargs
    )
    assert loaded == assignment
    other_development = copy.deepcopy(development)
    other_development[0]["row_id"] = "f" * 64
    with pytest.raises(NeuralDevelopmentStateError, match="partition_row_bindings"):
        load_neural_source_component_assignment_exact(
            path,
            digest,
            **{**kwargs, "expected_development_rows": other_development},
        )


@pytest.mark.parametrize(
    "method_id",
    [
        EDIT_AWARE_FULL_PAIR_METHOD_ID,
        A0_METHOD_ID,
        A0_CONTROL_METHOD_ID,
        A1_METHOD_ID,
        A1_CONTROL_METHOD_ID,
    ],
)
def test_generic_kernel_fits_every_registered_route_a_and_matched_control(
    method_id: str,
) -> None:
    state = _fit(method_id)
    assert state["method_id"] == method_id
    assert len(state["development_predictions"]) == 6
    assert state["compiled_binding"]["action_budget_validated_by_compiler"] is False
    if state["compiled_binding"]["matched_reference_method_id"] is not None:
        assert state["compiled_binding"]["parameter_match"]["matched"] is True


@pytest.mark.parametrize(
    "method_id",
    [
        A1_UNK_ONLY_METHOD_ID,
        B_ADDITIVE_METHOD_ID,
        B_MATCHED_CONTROL_METHOD_ID,
        B_UNK_ONLY_METHOD_ID,
    ],
)
def test_generic_kernel_fits_new_development_only_a_b_methods(
    method_id: str,
) -> None:
    state = _fit(method_id)
    assert state["method_id"] == method_id
    assert state["training_variant_id"] == state["compiled_binding"][
        "training_variant_id"
    ]
    assert len(state["development_predictions"]) == 6
    assert state["compiled_binding"]["execution_authorized"] is False
    assert state["compiled_binding"]["training_ready"] is False
    assert state["compiled_binding"]["publication"] is False
    assert state["compiled_binding"]["publication_evidence"] is False
    if state["compiled_binding"]["matched_reference_method_id"] is not None:
        assert state["compiled_binding"]["parameter_match"]["matched"] is True


@pytest.mark.parametrize(
    "method_id",
    [A1_WRONG_LOCAL_METHOD_ID, B_WRONG_LOCAL_METHOD_ID],
)
def test_wrong_local_methods_fit_only_when_every_bound_row_supports_relocation(
    method_id: str,
) -> None:
    state = _fit_wrong_local(method_id)
    assert state["method_id"] == method_id
    assert "WRONG_LOCAL_POSITION" in state["training_variant_id"]
    assert state["compiled_binding"]["parameter_match"]["gap_fraction"] == 0.0


def test_wrong_local_support_hold_occurs_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, development, targets = _fixture_rows()

    def construction_must_not_run(*args, **kwargs):
        raise AssertionError("wrong-local model constructed before support preflight")

    monkeypatch.setattr(
        neural_state, "compile_registered_development_model", construction_must_not_run
    )
    with pytest.raises(
        NeuralDevelopmentStateError,
        match="HOLD_WRONG_LOCAL_INSUFFICIENT_COMPATIBLE_POSITION_SUPPORT",
    ):
        fit_neural_development_state(
            B_WRONG_LOCAL_METHOD_ID,
            _cfg(),
            _fit_config(),
            train,
            development,
            targets,
            _assignment(train, development),
            seed=11,
            expected_registry_sha256=registry_sha256(),
            run_start_sha256=RUN_SHA,
            development_activation_sha256=ACTIVATION_SHA,
        )


def test_b_additive_state_exact_save_load_and_prediction_recompute(
    tmp_path: Path,
) -> None:
    state = _fit(B_ADDITIVE_METHOD_ID)
    development_dir = tmp_path / "development" / "b_additive"
    development_dir.mkdir(parents=True)
    path = development_dir / "seed_11.pt"
    digest = save_neural_development_state_exclusive(path, state)
    loaded = load_neural_development_state_exact(path, digest)
    _train, development, _targets = _fixture_rows()
    assert recompute_neural_development_predictions(loaded, development) == loaded[
        "development_predictions"
    ]


def test_exact_save_load_prediction_recompute_and_refit(
    tmp_path: Path, fitted_state: dict
) -> None:
    development_dir = tmp_path / "development" / "neural"
    development_dir.mkdir(parents=True)
    path = development_dir / "seed_11.pt"
    digest = save_neural_development_state_exclusive(path, fitted_state)
    loaded = load_neural_development_state_exact(path, digest)
    train, development, targets = _fixture_rows()
    recomputed = recompute_neural_development_predictions(loaded, development)
    assert recomputed == loaded["development_predictions"]
    rebuilt = rebuild_and_match_neural_development_state(
        loaded,
        train,
        development,
        targets,
        _assignment(train, development),
    )
    assert rebuilt["state_tensor_sha256"] == loaded["state_tensor_sha256"]
    with pytest.raises(NeuralDevelopmentStateError):
        save_neural_development_state_exclusive(path, fitted_state)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(NeuralDevelopmentStateError, match="SHA-256 mismatch"):
        load_neural_development_state_exact(path, digest)


def test_fixed_five_seed_ensemble_uses_total_variance_and_no_best_seed() -> None:
    train, development, targets = _fixture_rows()
    states, ensemble = fit_neural_development_five_seed_ensemble(
        EDIT_AWARE_FULL_PAIR_METHOD_ID,
        _cfg(),
        _fit_config(),
        train,
        development,
        targets,
        _assignment(train, development),
        expected_registry_sha256=registry_sha256(),
        run_start_sha256=RUN_SHA,
        development_activation_sha256=ACTIVATION_SHA,
    )
    assert [state["seed"] for state in states] == list(FROZEN_NEURAL_SEEDS)
    validated = validate_neural_development_ensemble(ensemble)
    assert validated["seeds"] == list(FROZEN_NEURAL_SEEDS)
    assert len(validated["seed_state_tensor_sha256"]) == 5
    assert len(validated["ensemble_predictions"]) == len(development)
    assert all(
        row["predicted_delta_logvar"] is not None
        for row in validated["ensemble_predictions"]
    )
    assert all(
        state["checkpoint_selection"]["best_seed_selection"] is False
        for state in states
    )


def test_b_additive_uses_the_same_frozen_five_seed_state_protocol() -> None:
    train, development, targets = _fixture_rows()
    states, ensemble = fit_neural_development_five_seed_ensemble(
        B_ADDITIVE_METHOD_ID,
        _cfg(),
        _fit_config(),
        train,
        development,
        targets,
        _assignment(train, development),
        expected_registry_sha256=registry_sha256(),
        run_start_sha256=RUN_SHA,
        development_activation_sha256=ACTIVATION_SHA,
    )
    assert [state["seed"] for state in states] == list(FROZEN_NEURAL_SEEDS)
    assert all(state["method_id"] == B_ADDITIVE_METHOD_ID for state in states)
    assert ensemble["training_variant_id"] == "PRIMARY"
    assert ensemble["method_id"] == B_ADDITIVE_METHOD_ID


def test_resource_failure_occurs_before_registered_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, development, targets = _fixture_rows()

    def construction_must_not_run(*args, **kwargs):
        raise AssertionError("registered model constructed before resource preflight")

    monkeypatch.setattr(
        neural_state, "compile_registered_development_model", construction_must_not_run
    )
    with pytest.raises(NeuralDevelopmentStateError, match="before model construction"):
        fit_neural_development_state(
            FULL_PAIR_METHOD_ID,
            _cfg(),
            _fit_config(max_rows_per_fit=1),
            train,
            development,
            targets,
            _assignment(train, development),
            seed=11,
            expected_registry_sha256=registry_sha256(),
            run_start_sha256=RUN_SHA,
            development_activation_sha256=ACTIVATION_SHA,
        )


def test_false_actual_config_parameter_match_fails_before_optimizer_or_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, development, targets = _fixture_rows()

    def optimizer_must_not_run(*args, **kwargs):
        raise AssertionError("optimizer constructed after false parameter match")

    monkeypatch.setattr(neural_state.torch.optim, "Adam", optimizer_must_not_run)
    cfg = ModelConfig(
        vocab_size=5,
        d_model=8,
        n_heads=1,
        n_layers=1,
        ff_dim=8,
        max_len=8,
        n_endpoints=3,
        n_contexts=3,
        max_edits=5,
        pair_rank=2,
    )
    with pytest.raises(NeuralDevelopmentStateError, match="parameter-matched"):
        fit_neural_development_state(
            EDIT_AWARE_FULL_PAIR_METHOD_ID,
            cfg,
            _fit_config(),
            train,
            development,
            targets,
            _assignment(train, development),
            seed=11,
            expected_registry_sha256=registry_sha256(),
            run_start_sha256=RUN_SHA,
            development_activation_sha256=ACTIVATION_SHA,
        )


def test_actual_action_replay_and_row_component_coverage_fail_closed() -> None:
    train, development, targets = _fixture_rows()
    broken = copy.deepcopy(train)
    broken[0]["predictor_input"]["normalized_candidate_sequence"] = broken[0][
        "predictor_input"
    ]["normalized_source_sequence"]
    with pytest.raises(NeuralDevelopmentStateError):
        fit_neural_development_state(
            FULL_PAIR_METHOD_ID,
            _cfg(),
            _fit_config(),
            broken,
            development,
            targets,
            _assignment(train, development),
            seed=11,
            expected_registry_sha256=registry_sha256(),
            run_start_sha256=RUN_SHA,
            development_activation_sha256=ACTIVATION_SHA,
        )
    missing = _assignment(train, development)
    missing["assignments"] = missing["assignments"][:-1]
    with pytest.raises(NeuralDevelopmentStateError):
        fit_neural_development_state(
            FULL_PAIR_METHOD_ID,
            _cfg(),
            _fit_config(),
            train,
            development,
            targets,
            missing,
            seed=11,
            expected_registry_sha256=registry_sha256(),
            run_start_sha256=RUN_SHA,
            development_activation_sha256=ACTIVATION_SHA,
        )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda state: state.__setitem__("seed", 12), "seed is not frozen"),
        (lambda state: state.__setitem__("registry_sha256", "0" * 64), "registry binding is stale"),
        (lambda state: state.__setitem__("information_set_fingerprint", "0" * 64), "fingerprint is stale"),
        (lambda state: state.__setitem__("sealed_access", True), "ordinary development boundary"),
        (lambda state: state["state_dict"][next(iter(state["state_dict"]))].view(-1).__setitem__(0, float("nan")), "non-finite"),
    ],
)
def test_state_tampering_fails_closed(fitted_state: dict, mutator, match: str) -> None:
    state = copy.deepcopy(fitted_state)
    mutator(state)
    with pytest.raises(NeuralDevelopmentStateError, match=match):
        validate_neural_development_state(state)


def test_state_paths_must_be_development_only(tmp_path: Path, fitted_state: dict) -> None:
    with pytest.raises(NeuralDevelopmentStateError, match="literal development"):
        save_neural_development_state_exclusive(tmp_path / "state.pt", fitted_state)
    protected = tmp_path / "development" / "sealed" / "state.pt"
    protected.parent.mkdir(parents=True)
    with pytest.raises(NeuralDevelopmentStateError, match="protected namespace"):
        save_neural_development_state_exclusive(protected, fitted_state)
