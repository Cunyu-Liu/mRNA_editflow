from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch

import scripts.r3x.typed_safe_authorities as typed_module

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3x.artifact_graph import (
    make_artifact_ref,
    read_artifact_ref,
    write_prediction_jsonl_exclusive,
)
from scripts.r3x.artifact_io import atomic_publish_bytes, canonical_json_bytes
from scripts.r3x.baseline_selection import (
    DEVELOPMENT_PARTITION,
    SELECTION_SCHEMA,
    build_development_baseline_evidence,
    build_development_baseline_selection,
    write_development_baseline_evidence_exclusive,
    write_development_baseline_selection_exclusive,
    write_development_evaluation_jsonl_exclusive,
)
from scripts.r3x.development_checkpoint import (
    build_development_checkpoint as build_development_model_state,
    load_development_checkpoint_exact as load_development_model_state_exact,
    save_development_checkpoint_exclusive,
)
from scripts.r3x.models import ExactPotentialCritic, ModelConfig
from scripts.r3x.predictor_provenance import (
    DEVELOPMENT_PARTITION as PREDICTOR_DEVELOPMENT_PARTITION,
    ENSEMBLE_RULE,
    ENSEMBLE_VARIANCE_FORMULA,
    FROZEN_NEURAL_SEEDS,
    NEURAL_KIND,
    NON_DATA_AUTHORITY_HASH_KEYS,
    PREDICTOR_INPUT_ROW_SCHEMA,
    SCALAR_KIND,
    TRAIN_PARTITION,
    VARIANCE_FLOOR,
    PredictorVerifier,
    build_development_checkpoint as build_checkpoint_envelope,
    build_development_checkpoint_selection,
    build_predictor_provenance,
    canonical_predictor_provenance_bytes,
    parse_predictor_input_jsonl,
    write_development_checkpoint_exclusive as write_checkpoint_envelope_exclusive,
    write_development_checkpoint_selection_exclusive,
    write_predictor_input_jsonl_exclusive,
    write_predictor_provenance_exclusive,
)
from scripts.r3x.protocol import REQUIRED_NEGATIVE_CONTROLS
from scripts.r3x.typed_pretest_freeze import (
    CONTROL_NOT_RUN,
    CONTROL_REPLAY_ACTIVE,
    CONTROL_REPLAY_RESULT_SCHEMA,
    DIRECT_CONTROL_VERIFIER_SCOPE,
    NEURAL_SEED,
    PRETEST_CONTROL_REGISTRY_SCHEMA_REQUIRED,
    REMAINING_BLOCKERS,
    SCALAR_FIT,
    SHIPPED_CAPABILITY_STATUS,
    UNIT_REGISTRY_HOLD,
    UNIT_REGISTRY_READY,
    PretestControlVerifier,
    PretestFreezeExpectations,
    TypedPretestFreezeError,
    build_typed_pretest_freeze,
    build_typed_unit_registry,
    canonical_typed_pretest_freeze_bytes,
    canonical_typed_unit_registry_bytes,
    load_typed_pretest_freeze_exact,
    load_typed_unit_registry_exact,
    typed_unit_registry_ref,
    validate_typed_pretest_freeze,
    validate_typed_unit_registry,
    write_typed_pretest_freeze_exclusive,
    write_typed_unit_registry_exclusive,
)
from scripts.r3x.typed_safe_authorities import (
    PAIRED_BASELINE_POLICY,
    ROLES,
    SCHEMAS,
    TYPED_DEVELOPMENT_SAFE,
    write_typed_safe_authority_exclusive,
)


BASELINE_IDS = list(typed_module.FROZEN_BASELINE_METHOD_ORDER)
FULL_PAIR = "FULL_PAIR_UNRESTRICTED_SIAMESE"
RUN_START_SHA = hashlib.sha256(b"run-start").hexdigest()
RUN_PLAN_SHA = hashlib.sha256(b"run-plan").hexdigest()
DEVELOPMENT_CAPABILITY_SHA = hashlib.sha256(b"development-capability").hexdigest()
TRAIN_DEVELOPMENT_SHA = hashlib.sha256(b"train-development-outcomes").hexdigest()
INFORMATION_SET_SHA = hashlib.sha256(b"primary-information-set").hexdigest()
ESTIMAND_ID = "ESTIMAND_A"
ESTIMAND_SHA = hashlib.sha256(b"estimand-a").hexdigest()
CELL_ID = "ESTIMAND_A:CELL_1"
TRUTHS = (-2.0, -1.0, 0.2, 1.0, 2.0)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _nondata_without_policy() -> dict[str, str]:
    return {
        name: _sha(f"authority:{name}")
        for name in NON_DATA_AUTHORITY_HASH_KEYS
        if name != "paired_strongest_baseline_comparison_policy_sha256"
    }


def _paired_policy(nondata: dict[str, str]) -> dict:
    projections = {
        "inclusion_manifest_development_projection": nondata[
            "inclusion_manifest_sha256"
        ],
        "context_source_crosswalk_development_projection": nondata[
            "context_source_crosswalk_sha256"
        ],
        "source_component_split_manifest_development_projection": nondata[
            "source_component_split_manifest_sha256"
        ],
    }
    family = ["primary_candidate_vs_development_selected_baseline_sign"]
    return {
        "schema_version": SCHEMAS[PAIRED_BASELINE_POLICY],
        "artifact_role": ROLES[PAIRED_BASELINE_POLICY],
        "status": TYPED_DEVELOPMENT_SAFE,
        "authority_name": PAIRED_BASELINE_POLICY,
        "source_authority_sha256": _sha("source:paired-policy"),
        "train_development_outcomes_sha256": TRAIN_DEVELOPMENT_SHA,
        "development_projection_hashes": projections,
        "sign_accuracy_gate": 0.60,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "training_ready": False,
        "sealed_access": False,
        "downstream_unlock": False,
        "policy_id": "R3X_PAIRED_STRONGEST_BASELINE_POLICY_V1",
        "baseline_suite_status": "PRETEST_POLICY_FROZEN",
        "primary_candidate_model_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "comparison_family_id": "R3X_PRIMARY_PAIRED_FAMILY_V1",
        "comparison_family_member_ids": family,
        "primary_comparison_member_id": family[0],
        "eligible_baseline_method_ids": list(BASELINE_IDS),
        "required_baseline_method_ids": list(BASELINE_IDS),
        "development_selector_required": True,
        "development_selector_schema": SELECTION_SCHEMA,
        "frozen_environment_execution_qualification_sha256": _sha(
            "execution-qualification"
        ),
        "histogram_gradient_boosting_execution_qualification_asserted": True,
        "execution_qualification_replay_status": "CERTIFICATE_HASH_BOUND_NOT_REPLAYED",
        "foundation_baseline_status": "NOT_RUN_EXPOSURE_LICENSE_QUALIFICATION_UNKNOWN",
        "primary_sign_inference": {
            "metric_id": "MACRO_COMPONENT_CELL_BALANCED_SIGN_ACCURACY",
            "alpha": 0.05,
            "confidence": 0.95,
            "bootstrap": {
                "n_resamples": 2000,
                "seed": 23,
                "min_components_per_cell": 10,
                "max_attempt_multiplier": 100,
            },
            "permutation": {
                "n_permutations": 10000,
                "seed": 42,
                "exact_component_limit": 20,
                "min_components_per_cell": 10,
            },
        },
    }


def _row_id(pool: str, index: int) -> str:
    return hashlib.sha256(f"{ESTIMAND_ID}/{CELL_ID}/{pool}/{index}".encode()).hexdigest()


def _evaluation_rows() -> list[dict]:
    rows = []
    for pool_index in (1, 2):
        pool = f"POOL_{pool_index}"
        for index, truth in enumerate(TRUTHS):
            rows.append(
                {
                    "schema_version": "r3x_development_evaluation_row_v1",
                    "row_id": _row_id(pool, index),
                    "partition": DEVELOPMENT_PARTITION,
                    "estimand_id": ESTIMAND_ID,
                    "cell_id": CELL_ID,
                    "true_delta": truth,
                    "source_component_id": f"COMPONENT_{pool_index}",
                    "source_group_id_v2": pool,
                }
            )
    return rows


def _predictor_input(position: int) -> dict:
    source = "ACGUACGUACGUACGU"
    ref = source[position]
    alt = {"A": "C", "C": "G", "G": "U", "U": "A"}[ref]
    candidate = list(source)
    candidate[position] = alt
    return {
        "endpoint": "mrna_abundance",
        "publication_context_id": "cell_a.assay_a.library_a",
        "region": "5UTR",
        "normalized_source_sequence": source,
        "normalized_candidate_sequence": "".join(candidate),
        "edit_list": [{"op": "SUB", "pos": position, "ref": ref, "alt": alt}],
        "edit_count": 1,
    }


def _predictor_rows(partition: str) -> list[dict]:
    if partition == TRAIN_PARTITION:
        return [
            {
                "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
                "row_id": _sha(f"train-row:{index}"),
                "partition": TRAIN_PARTITION,
                "estimand_id": ESTIMAND_ID,
                "cell_id": CELL_ID,
                "predictor_input": _predictor_input(index),
                "training_target": float(TRUTHS[index]),
            }
            for index in range(len(TRUTHS))
        ]
    rows = []
    for pool_index in (1, 2):
        pool = f"POOL_{pool_index}"
        for index in range(len(TRUTHS)):
            rows.append(
                {
                    "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
                    "row_id": _row_id(pool, index),
                    "partition": PREDICTOR_DEVELOPMENT_PARTITION,
                    "estimand_id": ESTIMAND_ID,
                    "cell_id": CELL_ID,
                    "predictor_input": _predictor_input(index),
                    "training_target": None,
                }
            )
    return rows


def _row_set_sha(rows: list[dict]) -> str:
    raw = json.dumps(
        sorted(row["row_id"] for row in rows), separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _prediction_rows(values: dict[str, float], *, logvar: float | None) -> list[dict]:
    return [
        {
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row_id,
            "predicted_delta_mean": float(value),
            "predicted_delta_logvar": logvar,
        }
        for row_id, value in values.items()
    ]


def _write_json_ref(run_root: Path, path: Path, payload: dict) -> dict:
    digest = atomic_publish_bytes(path, canonical_json_bytes(payload))
    return make_artifact_ref(run_root, path, digest)


def _state_nondata(authorities: dict[str, str]) -> dict[str, str]:
    return {
        name.removesuffix("_sha256"): authorities[name]
        for name in NON_DATA_AUTHORITY_HASH_KEYS
    }


def _cfg() -> ModelConfig:
    return ModelConfig(
        d_model=8,
        n_heads=2,
        n_layers=1,
        ff_dim=16,
        max_len=16,
        n_endpoints=2,
        n_contexts=2,
        max_edits=5,
        pair_rank=2,
    )


def _first_state_value(payload: dict) -> float:
    for tensor in payload["state_dict"].values():
        if tensor.is_floating_point() and tensor.numel():
            return float(tensor.reshape(-1)[0].item())
    raise AssertionError("synthetic state lacks a floating tensor")


def _materialize_scalar_predictor(
    run_root: Path,
    development: Path,
    method: str,
    authorities: dict[str, str],
) -> dict:
    method_root = development / "predictors" / method.lower()
    method_root.mkdir(parents=True)
    train_rows = _predictor_rows(TRAIN_PARTITION)
    development_rows = _predictor_rows(PREDICTOR_DEVELOPMENT_PARTITION)
    train_path = method_root / "train_rows.jsonl"
    development_path = method_root / "development_rows.jsonl"
    train_sha = write_predictor_input_jsonl_exclusive(train_path, train_rows)
    development_sha = write_predictor_input_jsonl_exclusive(
        development_path, development_rows
    )
    values = {
        row["row_id"]: TRUTHS[
            int(row["predictor_input"]["edit_list"][0]["pos"])
        ]
        - 1.1
        for row in development_rows
    }
    state_ref = _write_json_ref(
        run_root, method_root / "fitted_state.json", {"predictions": values}
    )
    config_ref = _write_json_ref(
        run_root,
        method_root / "fit_config.json",
        {"method_id": method, "fit_scope": "TRAIN_ONLY"},
    )
    prediction_path = method_root / "development_predictions.jsonl"
    prediction_sha = write_prediction_jsonl_exclusive(
        prediction_path, _prediction_rows(values, logvar=None)
    )
    prediction_ref = make_artifact_ref(run_root, prediction_path, prediction_sha)

    def recompute(root: Path, cell: dict) -> dict:
        state = json.loads(
            read_artifact_ref(root, cell["predictor_artifacts"]["fitted_state_ref"])
        )
        rows = parse_predictor_input_jsonl(
            read_artifact_ref(root, cell["development_rows_ref"])
        )
        return {
            "prediction": _prediction_rows(
                {row["row_id"]: state["predictions"][row["row_id"]] for row in rows},
                logvar=None,
            )
        }

    verifier = PredictorVerifier(
        verifier_id=f"{method}.SYNTHETIC_EXACT_REPLAY_V1",
        code_sha256=_sha(f"verifier:{method}"),
        expected_predictor_code_sha256=authorities[
            "analysis_code_tree_manifest_sha256"
        ],
        expected_predictor_config_sha256=authorities[
            "model_optimizer_seed_config_sha256"
        ],
        recompute=recompute,
    )
    provenance = build_predictor_provenance(
        run_root,
        run_start_sha256=RUN_START_SHA,
        activation_manifest_sha256=DEVELOPMENT_CAPABILITY_SHA,
        train_development_outcomes_sha256=TRAIN_DEVELOPMENT_SHA,
        non_data_authority_hashes=authorities,
        method_id=method,
        information_set_fingerprint=INFORMATION_SET_SHA,
        predictor_code_sha256=authorities["analysis_code_tree_manifest_sha256"],
        predictor_config_sha256=authorities[
            "model_optimizer_seed_config_sha256"
        ],
        verifier=verifier,
        estimands=[
            {
                "estimand_id": ESTIMAND_ID,
                "estimand_binding_sha256": ESTIMAND_SHA,
                "cells": [
                    {
                        "cell_id": CELL_ID,
                        "train_rows_ref": make_artifact_ref(
                            run_root, train_path, train_sha
                        ),
                        "development_rows_ref": make_artifact_ref(
                            run_root, development_path, development_sha
                        ),
                        "prediction_jsonl_ref": prediction_ref,
                        "predictor_artifacts": {
                            "kind": SCALAR_KIND,
                            "fitted_state_ref": state_ref,
                            "fit_config_ref": config_ref,
                        },
                    }
                ],
            }
        ],
    )
    provenance_path = method_root / "predictor_provenance.json"
    provenance_sha = write_predictor_provenance_exclusive(
        run_root,
        provenance_path,
        provenance,
        verifiers={verifier.verifier_id: verifier},
    )
    return {
        "provenance": provenance,
        "provenance_ref": make_artifact_ref(
            run_root, provenance_path, provenance_sha
        ),
        "prediction_ref": prediction_ref,
        "verifier": verifier,
    }


def _materialize_neural_predictor(
    run_root: Path,
    development: Path,
    authorities: dict[str, str],
) -> dict:
    method_root = development / "predictors" / FULL_PAIR.lower()
    method_root.mkdir(parents=True)
    train_rows = _predictor_rows(TRAIN_PARTITION)
    development_rows = _predictor_rows(PREDICTOR_DEVELOPMENT_PARTITION)
    train_path = method_root / "train_rows.jsonl"
    development_path = method_root / "development_rows.jsonl"
    train_sha = write_predictor_input_jsonl_exclusive(train_path, train_rows)
    development_sha = write_predictor_input_jsonl_exclusive(
        development_path, development_rows
    )
    train_set = _row_set_sha(train_rows)
    development_set = _row_set_sha(development_rows)
    code_sha = authorities["analysis_code_tree_manifest_sha256"]
    config_sha = authorities["model_optimizer_seed_config_sha256"]
    seed_artifacts = []
    seed_predictions = []
    state_paths = []
    for seed_index, seed in enumerate(FROZEN_NEURAL_SEEDS, start=1):
        seed_root = method_root / f"seed_{seed}"
        seed_root.mkdir()
        selection = build_development_checkpoint_selection(
            run_start_sha256=RUN_START_SHA,
            activation_manifest_sha256=DEVELOPMENT_CAPABILITY_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_SHA,
            non_data_authority_hashes=authorities,
            method_id=FULL_PAIR,
            estimand_id=ESTIMAND_ID,
            estimand_binding_sha256=ESTIMAND_SHA,
            cell_id=CELL_ID,
            seed=seed,
            train_row_set_sha256=train_set,
            development_row_set_sha256=development_set,
            predictor_code_sha256=code_sha,
            predictor_config_sha256=config_sha,
            selected_epoch=seed_index,
        )
        selection_path = seed_root / "checkpoint_selection.json"
        selection_sha = write_development_checkpoint_selection_exclusive(
            selection_path, selection
        )
        selection_ref = make_artifact_ref(run_root, selection_path, selection_sha)

        torch.manual_seed(seed)
        model = ExactPotentialCritic(_cfg())
        state_payload = build_development_model_state(
            model,
            _cfg(),
            run_start_sha256=RUN_START_SHA,
            run_plan_sha256=RUN_PLAN_SHA,
            development_activation_sha256=DEVELOPMENT_CAPABILITY_SHA,
            development_data_authority_hashes={
                "train_development_outcomes": TRAIN_DEVELOPMENT_SHA
            },
            development_nondata_authority_hashes=_state_nondata(authorities),
            estimand_binding_sha256=ESTIMAND_SHA,
            method_id=FULL_PAIR,
            training_variant_id="PRIMARY",
            training_config_sha256=config_sha,
            condition_vocab={
                "endpoint": {"__UNK__": 0, "mrna_abundance": 1},
                "context": {"__UNK__": 0, "cell_a.assay_a.library_a": 1},
            },
            information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
            information_set_fingerprint=INFORMATION_SET_SHA,
            seed=seed,
            selected_epoch=seed_index,
            checkpoint_selection_sha256=selection_sha,
            partition_row_set_sha256={
                "train": train_set,
                "development": development_set,
            },
        )
        state_path = seed_root / "development_model_state.pt"
        state_sha = save_development_checkpoint_exclusive(state_path, state_payload)
        state_ref = make_artifact_ref(run_root, state_path, state_sha)
        state_paths.append(state_path)
        checkpoint = build_checkpoint_envelope(
            run_start_sha256=RUN_START_SHA,
            activation_manifest_sha256=DEVELOPMENT_CAPABILITY_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_SHA,
            non_data_authority_hashes=authorities,
            method_id=FULL_PAIR,
            estimand_id=ESTIMAND_ID,
            estimand_binding_sha256=ESTIMAND_SHA,
            cell_id=CELL_ID,
            seed=seed,
            train_row_set_sha256=train_set,
            development_row_set_sha256=development_set,
            information_set_fingerprint=INFORMATION_SET_SHA,
            predictor_code_sha256=code_sha,
            predictor_config_sha256=config_sha,
            checkpoint_selection_ref=selection_ref,
            state_ref=state_ref,
            architecture_id=state_payload["architecture_id"],
        )
        checkpoint_path = seed_root / "checkpoint_envelope.json"
        checkpoint_sha = write_checkpoint_envelope_exclusive(
            checkpoint_path, checkpoint
        )
        checkpoint_ref = make_artifact_ref(
            run_root, checkpoint_path, checkpoint_sha
        )
        offset = 0.001 * _first_state_value(state_payload)
        values = {
            row["row_id"]: TRUTHS[
                int(row["predictor_input"]["edit_list"][0]["pos"])
            ]
            + offset
            for row in development_rows
        }
        prediction_rows = _prediction_rows(values, logvar=math.log(0.5))
        prediction_path = seed_root / "development_predictions.jsonl"
        prediction_sha = write_prediction_jsonl_exclusive(
            prediction_path, prediction_rows
        )
        prediction_ref = make_artifact_ref(
            run_root, prediction_path, prediction_sha
        )
        seed_predictions.append(prediction_rows)
        seed_artifacts.append(
            {
                "seed": seed,
                "checkpoint_selection_ref": selection_ref,
                "checkpoint_ref": checkpoint_ref,
                "development_prediction_ref": prediction_ref,
            }
        )

    rows_by_seed = [
        {row["row_id"]: row for row in rows} for rows in seed_predictions
    ]
    ensemble_rows = []
    for row in development_rows:
        means = [mapping[row["row_id"]]["predicted_delta_mean"] for mapping in rows_by_seed]
        logvars = [mapping[row["row_id"]]["predicted_delta_logvar"] for mapping in rows_by_seed]
        mean = math.fsum(means) / len(means)
        variance = math.fsum(
            math.exp(logvar) + value * value
            for value, logvar in zip(means, logvars)
        ) / len(means) - mean * mean
        ensemble_rows.append(
            {
                "schema_version": "r3x_prediction_row_v1",
                "row_id": row["row_id"],
                "predicted_delta_mean": mean,
                "predicted_delta_logvar": math.log(max(variance, VARIANCE_FLOOR)),
            }
        )
    ensemble_path = method_root / "ensemble_predictions.jsonl"
    ensemble_sha = write_prediction_jsonl_exclusive(ensemble_path, ensemble_rows)
    ensemble_ref = make_artifact_ref(run_root, ensemble_path, ensemble_sha)

    def recompute(root: Path, cell: dict) -> dict:
        rows = parse_predictor_input_jsonl(
            read_artifact_ref(root, cell["development_rows_ref"])
        )
        result = {}
        for item in cell["predictor_artifacts"]["seed_artifacts"]:
            envelope = json.loads(read_artifact_ref(root, item["checkpoint_ref"]))
            state_path = root / envelope["state_ref"]["relative_path"]
            state = load_development_model_state_exact(
                state_path, envelope["state_ref"]["sha256"]
            )
            offset = 0.001 * _first_state_value(state)
            values = {
                row["row_id"]: TRUTHS[
                    int(row["predictor_input"]["edit_list"][0]["pos"])
                ]
                + offset
                for row in rows
            }
            result[str(item["seed"])] = _prediction_rows(
                values, logvar=math.log(0.5)
            )
        return result

    verifier = PredictorVerifier(
        verifier_id="FULL_PAIR.SYNTHETIC_EXACT_REPLAY_V1",
        code_sha256=_sha("verifier:full-pair"),
        expected_predictor_code_sha256=code_sha,
        expected_predictor_config_sha256=config_sha,
        recompute=recompute,
    )
    provenance = build_predictor_provenance(
        run_root,
        run_start_sha256=RUN_START_SHA,
        activation_manifest_sha256=DEVELOPMENT_CAPABILITY_SHA,
        train_development_outcomes_sha256=TRAIN_DEVELOPMENT_SHA,
        non_data_authority_hashes=authorities,
        method_id=FULL_PAIR,
        information_set_fingerprint=INFORMATION_SET_SHA,
        predictor_code_sha256=code_sha,
        predictor_config_sha256=config_sha,
        verifier=verifier,
        estimands=[
            {
                "estimand_id": ESTIMAND_ID,
                "estimand_binding_sha256": ESTIMAND_SHA,
                "cells": [
                    {
                        "cell_id": CELL_ID,
                        "train_rows_ref": make_artifact_ref(
                            run_root, train_path, train_sha
                        ),
                        "development_rows_ref": make_artifact_ref(
                            run_root, development_path, development_sha
                        ),
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
                ],
            }
        ],
    )
    provenance_path = method_root / "predictor_provenance.json"
    provenance_sha = write_predictor_provenance_exclusive(
        run_root,
        provenance_path,
        provenance,
        verifiers={verifier.verifier_id: verifier},
    )
    return {
        "provenance": provenance,
        "provenance_ref": make_artifact_ref(
            run_root, provenance_path, provenance_sha
        ),
        "prediction_ref": ensemble_ref,
        "verifier": verifier,
        "state_paths": state_paths,
    }


def _control_verifier() -> PretestControlVerifier:
    verifier_id = "SYNTHETIC_PRETEST_CONTROL_REPLAY_V1"
    code_sha = _sha("control-verifier")

    def replay(root: Path, reference: dict, context: dict) -> dict:
        payload = json.loads(read_artifact_ref(root, reference))
        if payload != {
            "active": True,
            "control_id": payload.get("control_id"),
            "development_capability_sha256": DEVELOPMENT_CAPABILITY_SHA,
            "ordinary_only": True,
            "schema_version": "r3x_synthetic_pretest_control_v1",
        }:
            raise RuntimeError("synthetic control artifact changed")
        return {
            "schema_version": CONTROL_REPLAY_RESULT_SCHEMA,
            "control_id": payload["control_id"],
            "status": CONTROL_REPLAY_ACTIVE,
            "control_artifact_sha256": reference["sha256"],
            "run_start_sha256": context["run_start_sha256"],
            "development_capability_sha256": context[
                "development_capability_sha256"
            ],
            "ordinary_only": True,
            "internal_test_feature_access": False,
            "internal_test_outcome_access": False,
            "sealed_access": False,
            "downstream_unlock": False,
        }

    return PretestControlVerifier(
        verifier_id=verifier_id, code_sha256=code_sha, replay=replay
    )


def _required_layout() -> list[dict]:
    return [
        {
            "estimand_id": ESTIMAND_ID,
            "cells": [
                {
                    "cell_id": CELL_ID,
                    "registered_source_group_ids_v2": ["POOL_1", "POOL_2"],
                }
            ],
        }
    ]


@pytest.fixture(scope="module")
def strict_bundle(tmp_path_factory) -> dict:
    tmp_path = tmp_path_factory.mktemp("typed_pretest")
    run_root = tmp_path / "ordinary_run"
    development = run_root / "development"
    (development / "authorities").mkdir(parents=True)
    (development / "predictors").mkdir()
    (development / "controls").mkdir()

    nondata = _nondata_without_policy()
    policy_path = development / "authorities" / "paired_policy.json"
    policy_sha = write_typed_safe_authority_exclusive(
        run_root, policy_path, _paired_policy(nondata)
    )
    policy_ref = make_artifact_ref(run_root, policy_path, policy_sha)
    nondata["paired_strongest_baseline_comparison_policy_sha256"] = policy_sha
    authorities = {
        name: nondata[name] for name in NON_DATA_AUTHORITY_HASH_KEYS
    }
    expectations = PretestFreezeExpectations(
        run_start_sha256=RUN_START_SHA,
        run_plan_sha256=RUN_PLAN_SHA,
        development_capability_sha256=DEVELOPMENT_CAPABILITY_SHA,
        train_development_outcomes_sha256=TRAIN_DEVELOPMENT_SHA,
        non_data_authority_hashes=authorities,
        information_set_fingerprint=INFORMATION_SET_SHA,
        required_estimand_cells=_required_layout(),
    )

    evaluation_path = development / "evaluation_rows.jsonl"
    evaluation_sha = write_development_evaluation_jsonl_exclusive(
        evaluation_path, _evaluation_rows()
    )
    evaluation_ref = make_artifact_ref(run_root, evaluation_path, evaluation_sha)

    predictor_refs = {}
    predictor_payloads = {}
    prediction_refs = {}
    predictor_verifiers = {}
    state_paths = []
    for method in BASELINE_IDS:
        result = (
            _materialize_neural_predictor(run_root, development, authorities)
            if method == FULL_PAIR
            else _materialize_scalar_predictor(
                run_root, development, method, authorities
            )
        )
        predictor_refs[method] = result["provenance_ref"]
        predictor_payloads[method] = result["provenance"]
        prediction_refs[method] = result["prediction_ref"]
        verifier = result["verifier"]
        predictor_verifiers[verifier.verifier_id] = verifier
        state_paths.extend(result.get("state_paths", []))

    evidence_refs = {}
    for method in BASELINE_IDS:
        evidence = build_development_baseline_evidence(
            run_root,
            run_start_sha256=RUN_START_SHA,
            activation_manifest_sha256=DEVELOPMENT_CAPABILITY_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_SHA,
            non_data_authority_hashes=authorities,
            method_id=method,
            information_set_fingerprint=INFORMATION_SET_SHA,
            predictor_provenance_ref=predictor_refs[method],
            predictor_verifiers=predictor_verifiers,
            estimands=[
                {
                    "estimand_id": ESTIMAND_ID,
                    "cells": [
                        {
                            "cell_id": CELL_ID,
                            "prediction_jsonl_ref": prediction_refs[method],
                            "development_evaluation_jsonl_ref": evaluation_ref,
                        }
                    ],
                }
            ],
        )
        evidence_path = (
            development
            / "predictors"
            / method.lower()
            / "development_evidence.json"
        )
        evidence_sha = write_development_baseline_evidence_exclusive(
            run_root,
            evidence_path,
            evidence,
            predictor_verifiers=predictor_verifiers,
        )
        evidence_refs[method] = make_artifact_ref(
            run_root, evidence_path, evidence_sha
        )

    selector = build_development_baseline_selection(
        run_root,
        run_start_sha256=RUN_START_SHA,
        activation_manifest_sha256=DEVELOPMENT_CAPABILITY_SHA,
        train_development_outcomes_sha256=TRAIN_DEVELOPMENT_SHA,
        non_data_authority_hashes=authorities,
        eligible_method_order=BASELINE_IDS,
        development_evidence_refs=evidence_refs,
        required_estimand_cells=_required_layout(),
        predictor_verifiers=predictor_verifiers,
    )
    selector_path = development / "baseline_selection.json"
    selector_sha = write_development_baseline_selection_exclusive(
        run_root,
        selector_path,
        selector,
        predictor_verifiers=predictor_verifiers,
    )
    selector_ref = make_artifact_ref(run_root, selector_path, selector_sha)

    control_verifier = _control_verifier()
    control_registry = {control_verifier.verifier_id: control_verifier}
    control_specs = {}
    for control_id in REQUIRED_NEGATIVE_CONTROLS:
        control_path = development / "controls" / f"{control_id.lower()}.json"
        control_ref = _write_json_ref(
            run_root,
            control_path,
            {
                "active": True,
                "control_id": control_id,
                "development_capability_sha256": DEVELOPMENT_CAPABILITY_SHA,
                "ordinary_only": True,
                "schema_version": "r3x_synthetic_pretest_control_v1",
            },
        )
        control_specs[control_id] = {
            "status": "RUN",
            "control_artifact_ref": control_ref,
            "control_verifier_id": control_verifier.verifier_id,
        }

    registry = build_typed_unit_registry(
        run_root,
        paired_policy_ref=policy_ref,
        baseline_predictor_provenance_refs=predictor_refs,
        control_specifications=control_specs,
        expectations=expectations,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_registry,
    )
    registry_path = development / "typed_unit_registry.json"
    registry_sha = write_typed_unit_registry_exclusive(
        run_root,
        registry_path,
        registry,
        expectations=expectations,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_registry,
    )
    registry_ref = typed_unit_registry_ref(run_root, registry_path, registry_sha)
    freeze = build_typed_pretest_freeze(
        run_root,
        typed_unit_registry_ref=registry_ref,
        development_selector_ref=selector_ref,
        expectations=expectations,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_registry,
    )
    freeze_path = development / "typed_pretest_freeze.json"
    freeze_sha = write_typed_pretest_freeze_exclusive(
        run_root,
        freeze_path,
        freeze,
        expectations=expectations,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_registry,
    )
    return {
        "run_root": run_root,
        "development": development,
        "policy_ref": policy_ref,
        "authorities": authorities,
        "expectations": expectations,
        "predictor_refs": predictor_refs,
        "predictor_payloads": predictor_payloads,
        "predictor_verifiers": predictor_verifiers,
        "control_specs": control_specs,
        "control_verifiers": control_registry,
        "selector_ref": selector_ref,
        "registry": registry,
        "registry_ref": registry_ref,
        "registry_path": registry_path,
        "registry_sha": registry_sha,
        "freeze": freeze,
        "freeze_path": freeze_path,
        "freeze_sha": freeze_sha,
        "state_paths": state_paths,
    }


def test_exact_registry_uses_typed_non_cartesian_units_and_freeze_stays_hold(
    strict_bundle: dict,
) -> None:
    registry = strict_bundle["registry"]
    assert registry["status"] == UNIT_REGISTRY_READY
    baseline_units = [unit for unit in registry["units"] if unit["unit_kind"] in {SCALAR_FIT, NEURAL_SEED}]
    assert sum(unit["unit_kind"] == SCALAR_FIT for unit in baseline_units) == 7
    assert sum(unit["unit_kind"] == NEURAL_SEED for unit in baseline_units) == 5
    assert len(baseline_units) == 12
    assert [
        unit["seed"] for unit in baseline_units if unit["unit_kind"] == NEURAL_SEED
    ] == list(FROZEN_NEURAL_SEEDS)
    assert registry["eligible_baseline_method_order"] == BASELINE_IDS
    assert registry["control_authority_scope"] == DIRECT_CONTROL_VERIFIER_SCOPE
    assert (
        registry["pretest_control_registry_schema_required"]
        == PRETEST_CONTROL_REGISTRY_SCHEMA_REQUIRED
    )
    assert registry["pretest_control_registry_v1_replayed"] is False
    assert registry["pretest_control_registry_can_satisfy_freeze"] is False
    assert registry["control_qualification_passed"] is False
    assert registry["feature_stage_acceptable"] is False

    loaded_registry = load_typed_unit_registry_exact(
        strict_bundle["run_root"],
        strict_bundle["registry_path"],
        strict_bundle["registry_sha"],
        expectations=strict_bundle["expectations"],
        predictor_verifiers=strict_bundle["predictor_verifiers"],
        control_verifiers=strict_bundle["control_verifiers"],
    )
    assert loaded_registry == registry
    assert strict_bundle["registry_path"].read_bytes() == canonical_typed_unit_registry_bytes(registry)

    loaded_freeze = load_typed_pretest_freeze_exact(
        strict_bundle["run_root"],
        strict_bundle["freeze_path"],
        strict_bundle["freeze_sha"],
        expectations=strict_bundle["expectations"],
        predictor_verifiers=strict_bundle["predictor_verifiers"],
        control_verifiers=strict_bundle["control_verifiers"],
    )
    assert loaded_freeze == strict_bundle["freeze"]
    assert loaded_freeze["training_ready"] is False
    assert loaded_freeze["capability_ready"] is False
    assert loaded_freeze["shipped_capability_status"] == SHIPPED_CAPABILITY_STATUS
    assert loaded_freeze["shipped_capability_status"] == (
        "HOLD_PENDING_PRETEST_CONTROL_REGISTRY_DEVELOPMENT_CONTROL_EVIDENCE_"
        "EXECUTION_AND_ACTIVATION_BINDING"
    )
    assert loaded_freeze["downstream_unlock"] is False
    assert loaded_freeze["remaining_blockers"] == list(REMAINING_BLOCKERS)
    assert "PRETEST_CONTROL_REGISTRY_V1_REQUIRED" in loaded_freeze["remaining_blockers"]
    assert (
        "PRETEST_CONTROL_REGISTRY_CAN_SATISFY_FREEZE_FALSE"
        in loaded_freeze["remaining_blockers"]
    )
    assert (
        "ALL_REQUIRED_DEVELOPMENT_CONTROL_EVIDENCE_NOT_ESTABLISHED"
        in loaded_freeze["remaining_blockers"]
    )
    assert all(
        "CONTROL_SUMMARY_PASS" not in blocker
        for blocker in loaded_freeze["remaining_blockers"]
    )
    assert loaded_freeze["control_authority_scope"] == DIRECT_CONTROL_VERIFIER_SCOPE
    assert loaded_freeze["pretest_control_registry_v1_replayed"] is False
    assert loaded_freeze["pretest_control_registry_can_satisfy_freeze"] is False
    assert loaded_freeze["control_qualification_passed"] is False
    assert loaded_freeze["feature_stage_acceptable"] is False
    assert loaded_freeze["strict_replay"] == {
        "typed_paired_policy_replayed": True,
        "selector_replayed": True,
        "predictor_provenance_replayed": True,
        "checkpoint_selection_replayed": True,
        "development_model_state_replayed": True,
        "required_controls_replayed": False,
        "synthetic_control_scaffold_replayed": True,
        "pretest_control_registry_v1_replayed": False,
        "control_qualification_passed": False,
        "baseline_method_count": 8,
        "scalar_fit_unit_count": 7,
        "neural_seed_unit_count": 5,
        "control_run_unit_count": 7,
    }
    assert strict_bundle["freeze_path"].read_bytes() == canonical_typed_pretest_freeze_bytes(loaded_freeze)
    serialized = json.dumps(loaded_freeze, sort_keys=True).lower()
    assert "test_support" not in serialized
    assert "internal_test_feature_ref" not in serialized
    assert "internal_test_outcome_ref" not in serialized
    assert "gse246381" not in serialized

    with pytest.raises(TypedPretestFreezeError, match="already exists"):
        write_typed_pretest_freeze_exclusive(
            strict_bundle["run_root"],
            strict_bundle["freeze_path"],
            loaded_freeze,
            expectations=strict_bundle["expectations"],
            predictor_verifiers=strict_bundle["predictor_verifiers"],
            control_verifiers=strict_bundle["control_verifiers"],
        )


def test_fake_four_field_selector_and_provenance_fail_closed(
    strict_bundle: dict,
) -> None:
    development = strict_bundle["development"]
    fake_selector_ref = _write_json_ref(
        strict_bundle["run_root"],
        development / "fake_selector.json",
        {
            "schema_version": "fake_selector_v1",
            "status": "PASS",
            "selected_method_id": FULL_PAIR,
            "sha256": _sha("self-reported-selector"),
        },
    )
    with pytest.raises(TypedPretestFreezeError, match="selector failed strict replay"):
        build_typed_pretest_freeze(
            strict_bundle["run_root"],
            typed_unit_registry_ref=strict_bundle["registry_ref"],
            development_selector_ref=fake_selector_ref,
            expectations=strict_bundle["expectations"],
            predictor_verifiers=strict_bundle["predictor_verifiers"],
            control_verifiers=strict_bundle["control_verifiers"],
        )

    fake_provenance_ref = _write_json_ref(
        strict_bundle["run_root"],
        development / "fake_predictor_provenance.json",
        {
            "schema_version": "fake_provenance_v1",
            "method_id": BASELINE_IDS[0],
            "status": "PASS",
            "sha256": _sha("self-reported-provenance"),
        },
    )
    refs = copy.deepcopy(strict_bundle["predictor_refs"])
    refs[BASELINE_IDS[0]] = fake_provenance_ref
    with pytest.raises(TypedPretestFreezeError, match="predictor provenance failed exact replay"):
        build_typed_unit_registry(
            strict_bundle["run_root"],
            paired_policy_ref=strict_bundle["policy_ref"],
            baseline_predictor_provenance_refs=refs,
            control_specifications=strict_bundle["control_specs"],
            expectations=strict_bundle["expectations"],
            predictor_verifiers=strict_bundle["predictor_verifiers"],
            control_verifiers=strict_bundle["control_verifiers"],
        )


def test_arbitrary_checkpoint_and_model_state_bytes_cannot_enter_registry(
    strict_bundle: dict,
) -> None:
    run_root = strict_bundle["run_root"]
    development = strict_bundle["development"]
    arbitrary_checkpoint_path = development / "arbitrary_checkpoint.bin"
    arbitrary_checkpoint_sha = atomic_publish_bytes(
        arbitrary_checkpoint_path, b"checkpoint:arbitrary-unit"
    )
    arbitrary_checkpoint_ref = make_artifact_ref(
        run_root, arbitrary_checkpoint_path, arbitrary_checkpoint_sha
    )
    forged = copy.deepcopy(strict_bundle["predictor_payloads"][FULL_PAIR])
    forged["estimands"][0]["cells"][0]["predictor_artifacts"]["seed_artifacts"][0][
        "checkpoint_ref"
    ] = arbitrary_checkpoint_ref
    forged_path = development / "forged_arbitrary_checkpoint_provenance.json"
    forged_sha = atomic_publish_bytes(
        forged_path, canonical_predictor_provenance_bytes(forged)
    )
    forged_ref = make_artifact_ref(run_root, forged_path, forged_sha)
    refs = copy.deepcopy(strict_bundle["predictor_refs"])
    refs[FULL_PAIR] = forged_ref
    with pytest.raises(TypedPretestFreezeError, match="predictor provenance failed exact replay"):
        build_typed_unit_registry(
            run_root,
            paired_policy_ref=strict_bundle["policy_ref"],
            baseline_predictor_provenance_refs=refs,
            control_specifications=strict_bundle["control_specs"],
            expectations=strict_bundle["expectations"],
            predictor_verifiers=strict_bundle["predictor_verifiers"],
            control_verifiers=strict_bundle["control_verifiers"],
        )

    provenance = copy.deepcopy(strict_bundle["predictor_payloads"][FULL_PAIR])
    original_checkpoint_ref = provenance["estimands"][0]["cells"][0][
        "predictor_artifacts"
    ]["seed_artifacts"][0]["checkpoint_ref"]
    checkpoint = json.loads(read_artifact_ref(run_root, original_checkpoint_ref))
    arbitrary_state_path = development / "arbitrary_model_state.bin"
    arbitrary_state_sha = atomic_publish_bytes(
        arbitrary_state_path, b"model-state:not-a-development-checkpoint"
    )
    checkpoint["state_ref"] = make_artifact_ref(
        run_root, arbitrary_state_path, arbitrary_state_sha
    )
    forged_checkpoint_path = development / "forged_checkpoint_envelope.json"
    forged_checkpoint_ref = _write_json_ref(
        run_root, forged_checkpoint_path, checkpoint
    )
    provenance["estimands"][0]["cells"][0]["predictor_artifacts"]["seed_artifacts"][0][
        "checkpoint_ref"
    ] = forged_checkpoint_ref
    forged_state_provenance_path = development / "forged_state_provenance.json"
    forged_state_sha = atomic_publish_bytes(
        forged_state_provenance_path,
        canonical_predictor_provenance_bytes(provenance),
    )
    refs[FULL_PAIR] = make_artifact_ref(
        run_root, forged_state_provenance_path, forged_state_sha
    )
    with pytest.raises(TypedPretestFreezeError, match="predictor provenance failed exact replay|model state"):
        build_typed_unit_registry(
            run_root,
            paired_policy_ref=strict_bundle["policy_ref"],
            baseline_predictor_provenance_refs=refs,
            control_specifications=strict_bundle["control_specs"],
            expectations=strict_bundle["expectations"],
            predictor_verifiers=strict_bundle["predictor_verifiers"],
            control_verifiers=strict_bundle["control_verifiers"],
        )


def test_required_control_not_run_inactive_missing_verifier_and_posttest_summary_hold(
    strict_bundle: dict,
) -> None:
    specs = copy.deepcopy(strict_bundle["control_specs"])
    first_control = REQUIRED_NEGATIVE_CONTROLS[0]
    specs[first_control] = {
        "status": "NOT_RUN",
        "reason_code": "SYNTHETIC_CONTROL_NOT_RUN",
    }
    registry = build_typed_unit_registry(
        strict_bundle["run_root"],
        paired_policy_ref=strict_bundle["policy_ref"],
        baseline_predictor_provenance_refs=strict_bundle["predictor_refs"],
        control_specifications=specs,
        expectations=strict_bundle["expectations"],
        predictor_verifiers=strict_bundle["predictor_verifiers"],
        control_verifiers=strict_bundle["control_verifiers"],
    )
    assert registry["status"] == UNIT_REGISTRY_HOLD
    assert any(unit["unit_kind"] == CONTROL_NOT_RUN for unit in registry["units"])
    hold_path = strict_bundle["development"] / "not_run_unit_registry.json"
    hold_sha = write_typed_unit_registry_exclusive(
        strict_bundle["run_root"],
        hold_path,
        registry,
        expectations=strict_bundle["expectations"],
        predictor_verifiers=strict_bundle["predictor_verifiers"],
        control_verifiers=strict_bundle["control_verifiers"],
    )
    hold_ref = typed_unit_registry_ref(
        strict_bundle["run_root"], hold_path, hold_sha
    )
    with pytest.raises(TypedPretestFreezeError, match="required controls are not all active"):
        build_typed_pretest_freeze(
            strict_bundle["run_root"],
            typed_unit_registry_ref=hold_ref,
            development_selector_ref=strict_bundle["selector_ref"],
            expectations=strict_bundle["expectations"],
            predictor_verifiers=strict_bundle["predictor_verifiers"],
            control_verifiers=strict_bundle["control_verifiers"],
        )

    with pytest.raises(TypedPretestFreezeError, match="missing registered verifier"):
        build_typed_unit_registry(
            strict_bundle["run_root"],
            paired_policy_ref=strict_bundle["policy_ref"],
            baseline_predictor_provenance_refs=strict_bundle["predictor_refs"],
            control_specifications=strict_bundle["control_specs"],
            expectations=strict_bundle["expectations"],
            predictor_verifiers=strict_bundle["predictor_verifiers"],
            control_verifiers={},
        )

    original = next(iter(strict_bundle["control_verifiers"].values()))

    def inactive(root: Path, reference: dict, context: dict) -> dict:
        result = original.replay(root, reference, context)
        return {**result, "status": "INACTIVE"}

    inactive_verifier = PretestControlVerifier(
        verifier_id=original.verifier_id,
        code_sha256=original.code_sha256,
        replay=inactive,
    )
    with pytest.raises(TypedPretestFreezeError, match="exact active replay result"):
        build_typed_unit_registry(
            strict_bundle["run_root"],
            paired_policy_ref=strict_bundle["policy_ref"],
            baseline_predictor_provenance_refs=strict_bundle["predictor_refs"],
            control_specifications=strict_bundle["control_specs"],
            expectations=strict_bundle["expectations"],
            predictor_verifiers=strict_bundle["predictor_verifiers"],
            control_verifiers={inactive_verifier.verifier_id: inactive_verifier},
        )

    posttest_path = strict_bundle["development"] / "controls" / "posttest_summary.json"
    posttest_ref = _write_json_ref(
        strict_bundle["run_root"],
        posttest_path,
        {
            "ordinary_internal_test_row_binding_sha256": _sha("test-row-binding"),
            "overall_status": "PASS_ALL_PREREGISTERED_CONTROLS",
        },
    )
    posttest_specs = copy.deepcopy(strict_bundle["control_specs"])
    posttest_specs[first_control] = {
        "status": "RUN",
        "control_artifact_ref": posttest_ref,
        "control_verifier_id": original.verifier_id,
    }
    with pytest.raises(TypedPretestFreezeError, match="test/outcome/support content"):
        build_typed_unit_registry(
            strict_bundle["run_root"],
            paired_policy_ref=strict_bundle["policy_ref"],
            baseline_predictor_provenance_refs=strict_bundle["predictor_refs"],
            control_specifications=posttest_specs,
            expectations=strict_bundle["expectations"],
            predictor_verifiers=strict_bundle["predictor_verifiers"],
            control_verifiers=strict_bundle["control_verifiers"],
        )


def test_schema_sign_gate_cross_stage_and_old_cartesian_tampering_fail(
    strict_bundle: dict,
) -> None:
    lowered = copy.deepcopy(strict_bundle["freeze"])
    lowered["sign_accuracy_gate"] = 0.50
    with pytest.raises(TypedPretestFreezeError, match="boundary changed"):
        validate_typed_pretest_freeze(lowered)

    injected = copy.deepcopy(strict_bundle["freeze"])
    injected["internal_test_support_sha256"] = _sha("test-support")
    with pytest.raises(TypedPretestFreezeError, match="exact schema"):
        validate_typed_pretest_freeze(injected)

    forged_control_pass = copy.deepcopy(strict_bundle["freeze"])
    forged_control_pass["pretest_control_registry_v1_replayed"] = True
    forged_control_pass["pretest_control_registry_can_satisfy_freeze"] = True
    forged_control_pass["control_qualification_passed"] = True
    forged_control_pass["feature_stage_acceptable"] = True
    with pytest.raises(TypedPretestFreezeError, match="boundary changed"):
        validate_typed_pretest_freeze(forged_control_pass)

    forged_replay_pass = copy.deepcopy(strict_bundle["freeze"])
    forged_replay_pass["strict_replay"]["required_controls_replayed"] = True
    with pytest.raises(TypedPretestFreezeError, match="replay summary is not exact"):
        validate_typed_pretest_freeze(forged_replay_pass)

    cartesian = copy.deepcopy(strict_bundle["registry"])
    scalar = next(unit for unit in cartesian["units"] if unit["unit_kind"] == SCALAR_FIT)
    forged = {
        **scalar,
        "unit_id": f"BASELINE:{scalar['method_id']}:NEURAL_SEED:11",
        "unit_kind": NEURAL_SEED,
        "seed": 11,
    }
    cartesian["units"].insert(1, forged)
    with pytest.raises(TypedPretestFreezeError, match="exactly one deterministic scalar-fit"):
        validate_typed_unit_registry(cartesian)

    wrong_hash = copy.deepcopy(strict_bundle["registry"])
    wrong_hash["paired_policy_ref"]["sha256"] = "0" * 64
    with pytest.raises(
        TypedPretestFreezeError,
        match="HOLD_TYPED_PRETEST_PAIRED_POLICY",
    ):
        write_typed_unit_registry_exclusive(
            strict_bundle["run_root"],
            strict_bundle["development"] / "wrong_policy_registry.json",
            wrong_hash,
            expectations=strict_bundle["expectations"],
            predictor_verifiers=strict_bundle["predictor_verifiers"],
            control_verifiers=strict_bundle["control_verifiers"],
        )
