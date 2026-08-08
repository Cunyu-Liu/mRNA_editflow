from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.r3x.artifact_graph import (
    make_artifact_ref,
    read_artifact_ref,
    write_prediction_jsonl_exclusive,
)
from scripts.r3x.continuation_neural_evidence import (
    ContinuationNeuralEvidenceError,
    build_continuation_neural_evidence,
    load_continuation_neural_evidence_exact,
    validate_continuation_neural_evidence,
    write_continuation_neural_evidence_exclusive,
)
from scripts.r3x.information_sets import information_set_fingerprint
from scripts.r3x.models import ModelConfig
from scripts.r3x.neural_development_state import (
    NeuralDevelopmentFitConfig,
    build_neural_source_component_assignment,
    fit_neural_development_five_seed_ensemble,
    save_neural_development_state_exclusive,
)
from scripts.r3x.neural_method_registry import FULL_PAIR_METHOD_ID, registry_sha256
from scripts.r3x.predictor_provenance import (
    DEVELOPMENT_PARTITION,
    ENSEMBLE_RULE,
    ENSEMBLE_VARIANCE_FORMULA,
    FROZEN_NEURAL_SEEDS,
    NEURAL_KIND,
    NON_DATA_AUTHORITY_HASH_KEYS,
    PREDICTOR_INPUT_ROW_SCHEMA,
    TRAIN_PARTITION,
    VARIANCE_FLOOR,
    PredictorVerifier,
    build_development_checkpoint,
    build_development_checkpoint_selection,
    build_predictor_provenance,
    parse_predictor_input_jsonl,
    write_development_checkpoint_exclusive,
    write_development_checkpoint_selection_exclusive,
    write_predictor_input_jsonl_exclusive,
    write_predictor_provenance_exclusive,
)


RUN_SHA = "1" * 64
ACTIVATION_SHA = "2" * 64
OUTCOMES_SHA = "3" * 64
ESTIMAND = "ESTIMAND_A"
ESTIMAND_SHA = "4" * 64


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


def _fit_config() -> NeuralDevelopmentFitConfig:
    return NeuralDevelopmentFitConfig(
        epochs=1,
        batch_size=3,
        max_rows_per_fit=100,
        max_sequence_tokens_per_batch=10_000,
        max_forward_equivalents_per_fit=10_000,
        max_total_forward_equivalents_five_seeds=50_000,
        max_peak_parameter_bytes_per_fit=200_000_000,
        max_wall_time_seconds_per_fit=120.0,
        device="cpu",
    )


def _authorities() -> dict[str, str]:
    return {
        name: hashlib.sha256(f"authority/{name}".encode()).hexdigest()
        for name in NON_DATA_AUTHORITY_HASH_KEYS
    }


def _row(partition: str, index: int, *, cell: str, positions: tuple[int, ...], target):
    source = "ACGUACGU"
    candidate = list(source)
    edits = []
    next_base = {"A": "C", "C": "G", "G": "U", "U": "A"}
    for position in positions:
        ref = source[position]
        alt = next_base[ref]
        candidate[position] = alt
        edits.append({"op": "SUB", "pos": position, "ref": ref, "alt": alt})
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
            "normalized_candidate_sequence": "".join(candidate),
            "edit_list": edits,
            "edit_count": len(edits),
        },
        "training_target": target,
    }


def _rows():
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
                )
            )
            dev = _row(
                DEVELOPMENT_PARTITION,
                index,
                cell=cell,
                positions=positions,
                target=None,
            )
            development.append(dev)
            dev_targets[dev["row_id"]] = target - 0.1 * cell_index
    return train, development, dev_targets


def _row_set_sha(rows) -> str:
    row_ids = sorted(row["row_id"] for row in rows)
    return hashlib.sha256(
        json.dumps(row_ids, separators=(",", ":")).encode()
    ).hexdigest()


def _ref(run_root: Path, path: Path, digest: str) -> dict:
    return make_artifact_ref(run_root, path, digest)


@pytest.fixture(scope="module")
def neural_run(tmp_path_factory):
    run_root = tmp_path_factory.mktemp("ordinary_neural_run")
    development_root = run_root / "development"
    development_root.mkdir()
    train, development, dev_targets = _rows()
    authorities = _authorities()
    assignments = [
        {
            "row_id": row["row_id"],
            "partition": row["partition"],
            "source_component_id": f"COMPONENT_{index}",
        }
        for index, row in enumerate((*train, *development))
    ]
    assignment = build_neural_source_component_assignment(
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
        source_component_split_manifest_projection_sha256="5" * 64,
    )
    states, ensemble = fit_neural_development_five_seed_ensemble(
        FULL_PAIR_METHOD_ID,
        _cfg(),
        _fit_config(),
        train,
        development,
        dev_targets,
        assignment,
        expected_registry_sha256=registry_sha256(),
        run_start_sha256=RUN_SHA,
        development_activation_sha256=ACTIVATION_SHA,
    )

    state_refs = []
    for state in states:
        state_dir = development_root / f"seed_{state['seed']}"
        state_dir.mkdir()
        path = state_dir / "neural_state.pt"
        digest = save_neural_development_state_exclusive(path, state)
        state_refs.append(_ref(run_root, path, digest))

    cells = []
    state_prediction_maps = [
        {row["row_id"]: row for row in state["development_predictions"]}
        for state in states
    ]
    ensemble_map = {
        row["row_id"]: row for row in ensemble["ensemble_predictions"]
    }
    for cell in ("ESTIMAND_A:CELL_A", "ESTIMAND_A:CELL_B"):
        safe_cell = cell.replace(":", "_")
        cell_dir = development_root / safe_cell
        cell_dir.mkdir()
        train_cell = [row for row in train if row["cell_id"] == cell]
        dev_cell = [row for row in development if row["cell_id"] == cell]
        train_path = cell_dir / "train_rows.jsonl"
        dev_path = cell_dir / "development_rows.jsonl"
        train_ref = _ref(
            run_root,
            train_path,
            write_predictor_input_jsonl_exclusive(train_path, train_cell),
        )
        dev_ref = _ref(
            run_root,
            dev_path,
            write_predictor_input_jsonl_exclusive(dev_path, dev_cell),
        )
        seed_artifacts = []
        for index, (seed, state, state_ref) in enumerate(
            zip(FROZEN_NEURAL_SEEDS, states, state_refs, strict=True)
        ):
            seed_dir = cell_dir / f"seed_{seed}"
            seed_dir.mkdir()
            selection = build_development_checkpoint_selection(
                run_start_sha256=RUN_SHA,
                activation_manifest_sha256=ACTIVATION_SHA,
                train_development_outcomes_sha256=OUTCOMES_SHA,
                non_data_authority_hashes=authorities,
                method_id=FULL_PAIR_METHOD_ID,
                estimand_id=ESTIMAND,
                estimand_binding_sha256=ESTIMAND_SHA,
                cell_id=cell,
                seed=seed,
                train_row_set_sha256=_row_set_sha(train_cell),
                development_row_set_sha256=_row_set_sha(dev_cell),
                predictor_code_sha256=authorities[
                    "analysis_code_tree_manifest_sha256"
                ],
                predictor_config_sha256=authorities[
                    "model_optimizer_seed_config_sha256"
                ],
                selected_epoch=state["selected_epoch"],
            )
            selection_path = seed_dir / "checkpoint_selection.json"
            selection_ref = _ref(
                run_root,
                selection_path,
                write_development_checkpoint_selection_exclusive(
                    selection_path, selection
                ),
            )
            checkpoint = build_development_checkpoint(
                run_start_sha256=RUN_SHA,
                activation_manifest_sha256=ACTIVATION_SHA,
                train_development_outcomes_sha256=OUTCOMES_SHA,
                non_data_authority_hashes=authorities,
                method_id=FULL_PAIR_METHOD_ID,
                estimand_id=ESTIMAND,
                estimand_binding_sha256=ESTIMAND_SHA,
                cell_id=cell,
                seed=seed,
                train_row_set_sha256=_row_set_sha(train_cell),
                development_row_set_sha256=_row_set_sha(dev_cell),
                information_set_fingerprint=information_set_fingerprint(
                    "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
                ),
                predictor_code_sha256=authorities[
                    "analysis_code_tree_manifest_sha256"
                ],
                predictor_config_sha256=authorities[
                    "model_optimizer_seed_config_sha256"
                ],
                checkpoint_selection_ref=selection_ref,
                state_ref=state_ref,
                architecture_id=state["architecture_id"],
            )
            checkpoint_path = seed_dir / "checkpoint.json"
            checkpoint_ref = _ref(
                run_root,
                checkpoint_path,
                write_development_checkpoint_exclusive(checkpoint_path, checkpoint),
            )
            predictions = [
                state_prediction_maps[index][row["row_id"]] for row in dev_cell
            ]
            prediction_path = seed_dir / "development_predictions.jsonl"
            prediction_ref = _ref(
                run_root,
                prediction_path,
                write_prediction_jsonl_exclusive(prediction_path, predictions),
            )
            seed_artifacts.append(
                {
                    "seed": seed,
                    "checkpoint_selection_ref": selection_ref,
                    "checkpoint_ref": checkpoint_ref,
                    "development_prediction_ref": prediction_ref,
                }
            )
        ensemble_rows = [ensemble_map[row["row_id"]] for row in dev_cell]
        ensemble_path = cell_dir / "ensemble_predictions.jsonl"
        ensemble_ref = _ref(
            run_root,
            ensemble_path,
            write_prediction_jsonl_exclusive(ensemble_path, ensemble_rows),
        )
        cells.append(
            {
                "cell_id": cell,
                "train_rows_ref": train_ref,
                "development_rows_ref": dev_ref,
                "prediction_jsonl_ref": ensemble_ref,
                "predictor_artifacts": {
                    "kind": NEURAL_KIND,
                    "seeds": list(FROZEN_NEURAL_SEEDS),
                    "seed_artifacts": seed_artifacts,
                    "ensemble_prediction_ref": ensemble_ref,
                    "ensemble_rule": ENSEMBLE_RULE,
                    "ensemble_variance_formula": ENSEMBLE_VARIANCE_FORMULA,
                    "variance_floor": VARIANCE_FLOOR,
                },
            }
        )

    def recompute(root: Path, cell: dict):
        rows = parse_predictor_input_jsonl(
            read_artifact_ref(root, cell["development_rows_ref"])
        )
        row_ids = [row["row_id"] for row in rows]
        return {
            str(seed): [state_prediction_maps[index][row_id] for row_id in row_ids]
            for index, seed in enumerate(FROZEN_NEURAL_SEEDS)
        }

    code_sha = authorities["analysis_code_tree_manifest_sha256"]
    config_sha = authorities["model_optimizer_seed_config_sha256"]
    verifier = PredictorVerifier(
        verifier_id="NEURAL_STATE_TEST_VERIFIER_V1",
        code_sha256="6" * 64,
        expected_predictor_code_sha256=code_sha,
        expected_predictor_config_sha256=config_sha,
        recompute=recompute,
    )
    provenance = build_predictor_provenance(
        run_root,
        run_start_sha256=RUN_SHA,
        activation_manifest_sha256=ACTIVATION_SHA,
        train_development_outcomes_sha256=OUTCOMES_SHA,
        non_data_authority_hashes=authorities,
        method_id=FULL_PAIR_METHOD_ID,
        information_set_fingerprint=information_set_fingerprint(
            "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
        ),
        predictor_code_sha256=code_sha,
        predictor_config_sha256=config_sha,
        verifier=verifier,
        estimands=[
            {
                "estimand_id": ESTIMAND,
                "estimand_binding_sha256": ESTIMAND_SHA,
                "cells": cells,
            }
        ],
    )
    provenance_path = development_root / "predictor_provenance.json"
    provenance_sha = write_predictor_provenance_exclusive(
        run_root,
        provenance_path,
        provenance,
        verifiers={verifier.verifier_id: verifier},
    )
    return {
        "run_root": run_root,
        "provenance_ref": _ref(run_root, provenance_path, provenance_sha),
        "states": states,
    }


def test_receipt_exact_loads_states_recomputes_ensemble_and_binds_train_rows(
    neural_run,
) -> None:
    receipt = build_continuation_neural_evidence(
        neural_run["run_root"], neural_run["provenance_ref"]
    )
    estimand = receipt["estimands"][0]
    assert receipt["method_id"] == FULL_PAIR_METHOD_ID
    assert estimand["seeds"] == list(FROZEN_NEURAL_SEEDS)
    assert estimand["partition_row_bindings"]["train"]["n_rows"] == 6
    assert estimand["partition_row_bindings"]["development"]["n_rows"] == 6
    assert estimand["state_tensor_sha256"] == [
        state["state_tensor_sha256"] for state in neural_run["states"]
    ]
    assert receipt["best_seed_selection"] is False
    assert receipt["independent_train_refit_verified"] is False
    assert receipt["independent_component_assignment_replay_verified"] is False
    assert receipt["independent_development_target_replay_verified"] is False
    assert receipt["internal_test_access"] is False
    assert receipt["sealed_access"] is False
    assert receipt["publication_evidence"] is False
    assert receipt["downstream_unlock"] is False


def test_receipt_exact_write_and_load_replays_all_refs(neural_run) -> None:
    receipt = build_continuation_neural_evidence(
        neural_run["run_root"], neural_run["provenance_ref"]
    )
    path = neural_run["run_root"] / "development" / "neural_receipt.json"
    digest = write_continuation_neural_evidence_exclusive(
        neural_run["run_root"], path, receipt
    )
    reference = _ref(neural_run["run_root"], path, digest)
    assert load_continuation_neural_evidence_exact(
        neural_run["run_root"], reference
    ) == receipt


def test_receipt_schema_rejects_best_seed_and_protected_refs(neural_run) -> None:
    receipt = build_continuation_neural_evidence(
        neural_run["run_root"], neural_run["provenance_ref"]
    )
    changed = copy.deepcopy(receipt)
    changed["best_seed_selection"] = True
    with pytest.raises(ContinuationNeuralEvidenceError, match="boundary"):
        validate_continuation_neural_evidence(changed)
    changed = copy.deepcopy(receipt)
    changed["estimands"][0]["state_refs"][0]["relative_path"] = (
        "sealed/state.pt"
    )
    with pytest.raises(ContinuationNeuralEvidenceError, match="PROTECTED_INPUT"):
        validate_continuation_neural_evidence(changed)


def test_receipt_replay_rejects_state_or_generic_prediction_mismatch(
    neural_run, monkeypatch
) -> None:
    import scripts.r3x.continuation_neural_evidence as evidence

    original = evidence.recompute_neural_development_predictions

    def changed_predictions(state, rows):
        result = original(state, rows)
        result[0] = {**result[0], "predicted_delta_mean": result[0]["predicted_delta_mean"] + 0.5}
        return result

    monkeypatch.setattr(
        evidence, "recompute_neural_development_predictions", changed_predictions
    )
    with pytest.raises(
        ContinuationNeuralEvidenceError, match="seed prediction bytes differ"
    ):
        build_continuation_neural_evidence(
            neural_run["run_root"], neural_run["provenance_ref"]
        )
