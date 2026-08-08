from __future__ import annotations

import json
import copy
import hashlib
from functools import lru_cache
from pathlib import Path

import pytest

import scripts.r3x.run_ordinary as ordinary_runner
from scripts.r3x.artifact_graph import (
    make_artifact_ref,
    read_artifact_ref,
    write_prediction_jsonl_exclusive,
)
from scripts.r3x.artifact_io import (
    atomic_publish_bytes,
    atomic_write_json_exclusive,
    canonical_json_bytes,
)
from scripts.r3x.baseline_selection import (
    DEVELOPMENT_PARTITION,
    HOLD_STATUS as BASELINE_SELECTION_HOLD_STATUS,
    QUALIFIED_STATUS as BASELINE_SELECTION_QUALIFIED_STATUS,
    build_development_baseline_evidence,
    build_development_baseline_selection,
    write_development_baseline_evidence_exclusive,
    write_development_baseline_selection_exclusive,
    write_development_evaluation_jsonl_exclusive,
)
from scripts.r3x.predictor_provenance import (
    DEVELOPMENT_PARTITION as PREDICTOR_DEVELOPMENT_PARTITION,
    NON_DATA_AUTHORITY_HASH_KEYS,
    PREDICTOR_INPUT_ROW_SCHEMA,
    SCALAR_KIND,
    TRAIN_PARTITION,
    PredictorVerifier,
    build_predictor_provenance,
    parse_predictor_input_jsonl,
    write_predictor_input_jsonl_exclusive,
    write_predictor_provenance_exclusive,
)
from scripts.r3x.control_summary import (
    ACTION_SUPPORT_QUALIFIED,
    ALGEBRAIC_INVARIANT,
    ARTIFACT_ROLE as CONTROL_ARTIFACT_ROLE,
    B_INTERACTION,
    CONTROL_IDS,
    CONTROL_SPECS,
    PASS as CONTROL_PASS,
    PASS_ALL as CONTROL_PASS_ALL,
    SCHEMA as CONTROL_SCHEMA,
    TRAIN_DATA_PERTURBATION,
    build_not_run_negative_control_summary,
    write_negative_control_summary_exclusive,
)
from scripts.r3x.paired_artifact import (
    build_paired_comparison_artifact,
    build_verified_paired_reference,
    canonical_paired_artifact_bytes,
    write_paired_comparison_artifact_exclusive,
)
from scripts.r3x.statistics import PRIMARY_PAIRED_METRIC_ID
from scripts.r3x.report import (
    ReportProtocolError,
    evaluate_effect_gate,
    validate_run_report,
    write_json_exclusive,
)
from scripts.r3x.run_ordinary import run_synthetic_engineering_fixture


def metrics(sign: float = 0.65, spearman: float = 0.35) -> dict:
    return {
        "macro_sign_accuracy": sign,
        "macro_delta_spearman": spearman,
        "macro_rmse": 0.5,
        "macro_mae": 0.4,
        "macro_gaussian_nll": 0.8,
    }


def selection(*, passing: bool = True) -> dict:
    return {
        "schema_version": "r3x_selection_metric_result_v1",
        "status": "FROZEN_EVALUATED",
        "metric_id": "ORACLE_NORMALIZED_SELECTION_GAIN",
        "tie_policy": "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK",
        "value": 0.5 if passing else 0.1,
        "threshold": 0.3,
    }


@lru_cache(maxsize=2)
def _comparison_bundle(passing: bool) -> tuple[dict, dict]:
    rows = []
    candidate = []
    baseline = []
    for index in range(40):
        truth = 1.0 if index % 2 == 0 else -1.0
        rows.append({
            "r3_input_row_sha256": f"{index + 1:064x}",
            "record_id": f"row_{index}",
            "study": "study",
            "estimand_id": "estimand",
            "context_id_v2": "ctx",
            "actual_edit_count": 1,
            "source_component_id": f"component_{index}",
            "delta": truth,
        })
        candidate.append(truth if index < 26 else -truth)
        baseline.append(candidate[-1] if not passing else -truth)
    policy = {
        "schema_version": "r3x_paired_strongest_baseline_comparison_policy_v1",
        "status": "FROZEN",
        "policy_id": "policy",
        "primary_candidate_model_id": "A",
        "strongest_baseline_id": "FULL_PAIR_UNRESTRICTED_SIAMESE",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "comparison_family_id": "family",
        "comparison_family_member_ids": ["a_vs_strongest_sign"],
        "primary_comparison_member_id": "a_vs_strongest_sign",
        "primary_sign_inference": {
            "metric_id": PRIMARY_PAIRED_METRIC_ID,
            "alpha": 0.05,
            "confidence": 0.95,
            "bootstrap": {
                "n_resamples": 2000, "seed": 23,
                "min_components_per_cell": 10, "max_attempt_multiplier": 100,
            },
            "permutation": {
                "n_permutations": 10000, "seed": 42,
                "exact_component_limit": 20, "min_components_per_cell": 10,
            },
        },
    }
    build_kwargs = {
        "policy": policy, "policy_sha256": "1" * 64,
        "authority_sha256": {"ready_dataset_sha256": "2" * 64},
        "candidate_model_id": "A",
        "baseline_model_id": "FULL_PAIR_UNRESTRICTED_SIAMESE",
        "candidate_information_set_fingerprint": "a" * 64,
        "baseline_information_set_fingerprint": "a" * 64,
        "candidate_prediction_artifact_sha256": "3" * 64,
        "baseline_prediction_artifact_sha256": "4" * 64,
        "additional_family_raw_p_values": {},
        "additional_family_evidence_sha256": {},
    }
    payload = build_paired_comparison_artifact(
        rows, candidate, baseline, **build_kwargs,
    )
    digest = hashlib.sha256(canonical_paired_artifact_bytes(payload)).hexdigest()
    reference = build_verified_paired_reference(
        artifact_relative_path="R3X_PAIRED_COMPARISON_V2.json",
        artifact_sha256=digest,
        artifact_payload=payload,
    )
    context = {
        "rows": rows,
        "candidate_prediction": candidate,
        "baseline_prediction": baseline,
        "build_kwargs": build_kwargs,
    }
    return reference, context


def _comparison_fixture(passing: bool) -> dict:
    return copy.deepcopy(_comparison_bundle(passing)[0])


def _comparison_context(passing: bool = True) -> dict:
    return copy.deepcopy(_comparison_bundle(passing)[1])


def _materialize_comparison(tmp_path: Path, passing: bool = True) -> tuple[dict, dict]:
    reference = _comparison_fixture(passing)
    path = tmp_path / reference["artifact_relative_path"]
    digest = write_paired_comparison_artifact_exclusive(
        path, reference["artifact_payload"]
    )
    assert digest == reference["artifact_sha256"]
    return reference, _comparison_context(passing)


BASELINE_METHOD_ORDER = [
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
    "PAIRED_FEATURE_NEAREST_NEIGHBOR",
    "HISTOGRAM_GRADIENT_BOOSTING",
]
BASELINE_RUN_START_SHA = "c" * 64
BASELINE_ACTIVATION_SHA = "d" * 64
BASELINE_TRAIN_DEVELOPMENT_OUTCOMES_SHA = "b" * 64
BASELINE_INFORMATION_SET_FINGERPRINT = "a" * 64


def _baseline_primary_predictor_input() -> dict:
    return {
        "endpoint": "ESTIMAND_ENDPOINT",
        "publication_context_id": "CELL_CONTEXT",
        "region": "5UTR",
        "normalized_source_sequence": "ACGUAC",
        "normalized_candidate_sequence": "AGGUAC",
        "edit_list": [{"op": "SUB", "pos": 1, "ref": "C", "alt": "G"}],
        "edit_count": 1,
    }


def _baseline_authorities() -> dict[str, str]:
    return {
        name: format(index + 1, "x") * 64
        for index, name in enumerate(NON_DATA_AUTHORITY_HASH_KEYS)
    }


def _baseline_predictor_verifier(method_id: str) -> PredictorVerifier:
    authorities = _baseline_authorities()

    def recompute(run_root: Path, cell: dict) -> dict:
        state = json.loads(
            read_artifact_ref(
                run_root, cell["predictor_artifacts"]["fitted_state_ref"]
            )
        )
        rows = parse_predictor_input_jsonl(
            read_artifact_ref(run_root, cell["development_rows_ref"])
        )
        return {"prediction": [{
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row["row_id"],
            "predicted_delta_mean": float(state["predictions"][row["row_id"]]),
            "predicted_delta_logvar": None,
        } for row in rows]}

    return PredictorVerifier(
        verifier_id=f"{method_id}.REPORT_FIXTURE_REPLAY_V1",
        code_sha256=hashlib.sha256(f"report/{method_id}".encode()).hexdigest(),
        expected_predictor_code_sha256=authorities[
            "analysis_code_tree_manifest_sha256"
        ],
        expected_predictor_config_sha256=authorities[
            "model_optimizer_seed_config_sha256"
        ],
        recompute=recompute,
    )


def _baseline_required_cells() -> list[dict]:
    return [{
        "estimand_id": "ESTIMAND",
        "cells": [{
            "cell_id": "ESTIMAND:CELL",
            "registered_source_group_ids_v2": ["POOL_1", "POOL_2"],
        }],
    }]


def _baseline_development_rows() -> list[dict]:
    rows = []
    for pool_index in (1, 2):
        for row_index, truth in enumerate((-2.0, -1.0, 0.2, 1.0, 2.0)):
            row_id = hashlib.sha256(
                f"baseline/{pool_index}/{row_index}".encode("utf-8")
            ).hexdigest()
            rows.append({
                "schema_version": "r3x_development_evaluation_row_v1",
                "row_id": row_id,
                "partition": DEVELOPMENT_PARTITION,
                "estimand_id": "ESTIMAND",
                "cell_id": "ESTIMAND:CELL",
                "true_delta": truth,
                "source_component_id": f"COMPONENT_{pool_index}",
                "source_group_id_v2": f"POOL_{pool_index}",
            })
    return rows


def _materialize_baseline_selection(
    run_root: Path,
    *,
    selected_method_id: str = "FULL_PAIR_UNRESTRICTED_SIAMESE",
    qualified: bool = True,
    information_set_fingerprint: str = BASELINE_INFORMATION_SET_FINGERPRINT,
) -> tuple[dict, dict, dict]:
    development = run_root / "development"
    development.mkdir(parents=True, exist_ok=True)
    evaluation_rows = _baseline_development_rows()
    evaluation_path = development / "evaluation.jsonl"
    evaluation_sha = write_development_evaluation_jsonl_exclusive(
        evaluation_path, evaluation_rows
    )
    evaluation_ref = make_artifact_ref(run_root, evaluation_path, evaluation_sha)
    evidence_refs = {}
    predictor_verifiers = {}
    for method_id in BASELINE_METHOD_ORDER:
        method_root = development / method_id.lower()
        method_root.mkdir()
        if not qualified:
            offset = -2.1
        elif method_id == selected_method_id:
            offset = 0.1
        elif method_id == "FULL_PAIR_UNRESTRICTED_SIAMESE":
            offset = -1.1
        else:
            offset = -2.1
        predictions = [{
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row["row_id"],
            "predicted_delta_mean": row["true_delta"] + offset,
            "predicted_delta_logvar": None,
        } for row in evaluation_rows]
        prediction_path = method_root / "prediction.jsonl"
        prediction_sha = write_prediction_jsonl_exclusive(
            prediction_path, predictions
        )
        prediction_ref = make_artifact_ref(
            run_root, prediction_path, prediction_sha
        )
        train_input_rows = [{
            "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
            "row_id": hashlib.sha256(
                f"report/{method_id}/train/{index}".encode()
            ).hexdigest(),
            "partition": TRAIN_PARTITION,
            "estimand_id": "ESTIMAND",
            "cell_id": "ESTIMAND:CELL",
            "predictor_input": _baseline_primary_predictor_input(),
            "training_target": float(index),
        } for index in range(1, 4)]
        development_input_rows = [{
            "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
            "row_id": prediction["row_id"],
            "partition": PREDICTOR_DEVELOPMENT_PARTITION,
            "estimand_id": "ESTIMAND",
            "cell_id": "ESTIMAND:CELL",
            "predictor_input": _baseline_primary_predictor_input(),
            "training_target": None,
        } for index, prediction in enumerate(predictions, start=1)]
        train_path = method_root / "train_rows.jsonl"
        development_path = method_root / "development_rows.jsonl"
        train_sha = write_predictor_input_jsonl_exclusive(
            train_path, train_input_rows
        )
        development_sha = write_predictor_input_jsonl_exclusive(
            development_path, development_input_rows
        )
        state_path = method_root / "fitted_state.json"
        state_sha = atomic_publish_bytes(
            state_path,
            canonical_json_bytes({
                "predictions": {
                    row["row_id"]: row["predicted_delta_mean"]
                    for row in predictions
                }
            }),
        )
        config_path = method_root / "fit_config.json"
        config_sha = atomic_publish_bytes(
            config_path,
            canonical_json_bytes({"method_id": method_id, "scope": "TRAIN_ONLY"}),
        )
        verifier = _baseline_predictor_verifier(method_id)
        predictor_verifiers[verifier.verifier_id] = verifier
        provenance = build_predictor_provenance(
            run_root,
            run_start_sha256=BASELINE_RUN_START_SHA,
            activation_manifest_sha256=BASELINE_ACTIVATION_SHA,
            train_development_outcomes_sha256=(
                BASELINE_TRAIN_DEVELOPMENT_OUTCOMES_SHA
            ),
            non_data_authority_hashes=_baseline_authorities(),
            method_id=method_id,
            information_set_fingerprint=information_set_fingerprint,
            predictor_code_sha256=_baseline_authorities()[
                "analysis_code_tree_manifest_sha256"
            ],
            predictor_config_sha256=_baseline_authorities()[
                "model_optimizer_seed_config_sha256"
            ],
            verifier=verifier,
            estimands=[{
                "estimand_id": "ESTIMAND",
                "estimand_binding_sha256": hashlib.sha256(
                    b"report/ESTIMAND"
                ).hexdigest(),
                "cells": [{
                    "cell_id": "ESTIMAND:CELL",
                    "train_rows_ref": make_artifact_ref(
                        run_root, train_path, train_sha
                    ),
                    "development_rows_ref": make_artifact_ref(
                        run_root, development_path, development_sha
                    ),
                    "prediction_jsonl_ref": prediction_ref,
                    "predictor_artifacts": {
                        "kind": SCALAR_KIND,
                        "fitted_state_ref": make_artifact_ref(
                            run_root, state_path, state_sha
                        ),
                        "fit_config_ref": make_artifact_ref(
                            run_root, config_path, config_sha
                        ),
                    },
                }],
            }],
        )
        provenance_path = method_root / "predictor_provenance.json"
        provenance_sha = write_predictor_provenance_exclusive(
            run_root,
            provenance_path,
            provenance,
            verifiers=predictor_verifiers,
        )
        provenance_ref = make_artifact_ref(
            run_root, provenance_path, provenance_sha
        )
        evidence = build_development_baseline_evidence(
            run_root,
            run_start_sha256=BASELINE_RUN_START_SHA,
            activation_manifest_sha256=BASELINE_ACTIVATION_SHA,
            train_development_outcomes_sha256=(
                BASELINE_TRAIN_DEVELOPMENT_OUTCOMES_SHA
            ),
            non_data_authority_hashes=_baseline_authorities(),
            method_id=method_id,
            information_set_fingerprint=information_set_fingerprint,
            predictor_provenance_ref=provenance_ref,
            predictor_verifiers=predictor_verifiers,
            estimands=[{
                "estimand_id": "ESTIMAND",
                "cells": [{
                    "cell_id": "ESTIMAND:CELL",
                    "prediction_jsonl_ref": prediction_ref,
                    "development_evaluation_jsonl_ref": evaluation_ref,
                }],
            }],
        )
        evidence_path = method_root / "evidence.json"
        evidence_sha = write_development_baseline_evidence_exclusive(
            run_root,
            evidence_path,
            evidence,
            predictor_verifiers=predictor_verifiers,
        )
        evidence_refs[method_id] = make_artifact_ref(
            run_root, evidence_path, evidence_sha
        )
    payload = build_development_baseline_selection(
        run_root,
        run_start_sha256=BASELINE_RUN_START_SHA,
        activation_manifest_sha256=BASELINE_ACTIVATION_SHA,
        train_development_outcomes_sha256=(
            BASELINE_TRAIN_DEVELOPMENT_OUTCOMES_SHA
        ),
        non_data_authority_hashes=_baseline_authorities(),
        eligible_method_order=BASELINE_METHOD_ORDER,
        development_evidence_refs=evidence_refs,
        required_estimand_cells=_baseline_required_cells(),
        predictor_verifiers=predictor_verifiers,
    )
    selection_path = development / "baseline-selection.json"
    selection_sha = write_development_baseline_selection_exclusive(
        run_root,
        selection_path,
        payload,
        predictor_verifiers=predictor_verifiers,
    )
    reference = make_artifact_ref(run_root, selection_path, selection_sha)
    context = {
        "expected_run_start_sha256": BASELINE_RUN_START_SHA,
        "expected_activation_manifest_sha256": BASELINE_ACTIVATION_SHA,
        "expected_train_development_outcomes_sha256": (
            BASELINE_TRAIN_DEVELOPMENT_OUTCOMES_SHA
        ),
        "expected_non_data_authority_hashes": _baseline_authorities(),
        "expected_eligible_method_order": list(BASELINE_METHOD_ORDER),
        "expected_required_estimand_cells": _baseline_required_cells(),
        "predictor_verifiers": predictor_verifiers,
    }
    return reference, context, payload


CONTROL_TEST_BINDING = hashlib.sha256(
    json.dumps(
        sorted(f"{index + 1:064x}" for index in range(40)),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()
CONTROL_VERIFIER_CONTEXT = {
    "authority_source": "INDEPENDENT_REPORT_TEST_CONTEXT",
    "expected_internal_test_row_binding_sha256": CONTROL_TEST_BINDING,
}


def _publish_json(root: Path, relative_path: str, payload: dict) -> dict:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = atomic_write_json_exclusive(path, payload)
    return make_artifact_ref(root, path, digest)


def _publish_bytes(root: Path, relative_path: str, raw: bytes) -> dict:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = atomic_publish_bytes(path, raw)
    return make_artifact_ref(root, path, digest)


def _control_method_protocol() -> dict:
    return {
        "schema_version": "r3_ordinary_experiment_protocol_v1",
        "negative_controls": list(CONTROL_IDS),
        "hard_boundaries": {
            "ordinary_only": True,
            "sealed_member_access": False,
            "O0_G1_E0_unlock": False,
        },
    }


def _control_action_policy() -> dict:
    return {
        "schema_version": "r3_action_support_policy_v1",
        "status": "FROZEN",
        "qualification_status": "PUBLICATION_QUALIFIED",
        "authority_hashes": {
            "ready_dataset_sha256": "1" * 64,
            "inclusion_manifest_sha256": "2" * 64,
            "context_source_crosswalk_sha256": "3" * 64,
            "endpoint_estimand_registry_sha256": "4" * 64,
            "source_component_split_manifest_sha256": "5" * 64,
        },
        "thresholds": {
            "min_independent_source_components_per_action": 20,
            "min_study_context_cells_per_action": 2,
            "max_single_component_dominance": 0.20,
        },
        "observed": {
            "min_independent_source_components_per_action": 25,
            "min_study_context_cells_per_action": 3,
            "max_single_component_dominance": 0.10,
            "qualified_action_strata": 8,
        },
    }


def _control_manifest(control_id: str) -> dict:
    return {
        "schema_version": "R3X_NEGATIVE_CONTROL_MANIFEST_V1",
        "name": control_id,
        "scope": "ORDINARY_TRAIN_PARTITION_ONLY",
        "seed": 11,
        "n_rows": 20,
        "ordered_input_row_ids_sha256": hashlib.sha256(
            f"{control_id}-ordered".encode()
        ).hexdigest(),
        "input_sha256": hashlib.sha256(f"{control_id}-input".encode()).hexdigest(),
        "output_sha256": hashlib.sha256(f"{control_id}-output".encode()).hexdigest(),
        "partition_validation": "ALL_INPUT_AND_OUTPUT_ROWS_EQUAL_TRAIN",
        "test_partition_modified": False,
        "sealed_access": False,
    }


def _control_verifier_result(raw: bytes, metadata: dict) -> dict:
    assert raw.startswith(b"independent report-gate control evidence")
    is_train_control = metadata["control_class"] == TRAIN_DATA_PERTURBATION
    is_algebraic = metadata["control_class"] == ALGEBRAIC_INVARIANT
    return {
        "control_id": metadata["control_id"],
        "method_id": metadata["method_id"],
        "outcome": metadata["expected_outcome"],
        "ordinary_internal_test_row_binding_sha256": (
            None if is_algebraic else metadata["ordinary_internal_test_row_binding_sha256"]
        ),
        "train_only_verified": True if is_train_control else None,
        "train_control_manifest_sha256": (
            metadata["train_control_manifest_sha256"] if is_train_control else None
        ),
        "action_support_policy_sha256": (
            metadata["action_support_policy_sha256"]
            if metadata["method_id"] == B_INTERACTION
            else None
        ),
        "verification_scope": metadata["required_verification_scope"],
        "verified_partitions": metadata["required_verified_partitions"],
    }


def _control_verifier(raw: bytes, metadata: dict) -> dict:
    assert metadata["verifier_context"] == CONTROL_VERIFIER_CONTEXT
    return _control_verifier_result(raw, metadata)


def _control_verifier_accepting_external_context(raw: bytes, metadata: dict) -> dict:
    return _control_verifier_result(raw, metadata)


def _materialize_passing_controls(
    run_root: Path,
    *,
    test_binding: str = CONTROL_TEST_BINDING,
) -> tuple[dict, dict]:
    protocol_ref = _publish_json(
        run_root, "controls/authority/method-protocol.json", _control_method_protocol()
    )
    action_ref = _publish_json(
        run_root, "controls/authority/action-support.json", _control_action_policy()
    )
    controls = []
    for control_id in CONTROL_IDS:
        spec = CONTROL_SPECS[control_id]
        required = list(spec["base_methods"])
        if spec["optional_interaction"]:
            required.append(B_INTERACTION)
        manifest_ref = None
        if spec["control_class"] == TRAIN_DATA_PERTURBATION:
            manifest_ref = _publish_json(
                run_root,
                f"controls/manifests/{control_id}.json",
                _control_manifest(control_id),
            )
        results = []
        for method_id in required:
            evidence_ref = _publish_bytes(
                run_root,
                f"controls/evidence/{control_id}/{method_id}.bin",
                f"independent report-gate control evidence {control_id}/{method_id}\n".encode(),
            )
            results.append({
                "method_id": method_id,
                "outcome": CONTROL_PASS,
                "evidence_ref": evidence_ref,
                "ordinary_internal_test_row_binding_sha256": (
                    None
                    if spec["control_class"] == ALGEBRAIC_INVARIANT
                    else test_binding
                ),
                "reason_code": None,
            })
        controls.append({
            "control_id": control_id,
            "control_class": spec["control_class"],
            "required_method_ids": required,
            "conditionally_excluded_method_ids": [],
            "outcome_criterion": spec["outcome_criterion"],
            "train_only_invariant": spec["train_only_invariant"],
            "train_control_manifest_ref": manifest_ref,
            "method_results": results,
            "status": CONTROL_PASS,
        })
    summary = {
        "schema_version": CONTROL_SCHEMA,
        "artifact_role": CONTROL_ARTIFACT_ROLE,
        "method_protocol_ref": protocol_ref,
        "action_support_status": ACTION_SUPPORT_QUALIFIED,
        "action_support_policy_ref": action_ref,
        "ordinary_internal_test_row_binding_sha256": test_binding,
        "controls": controls,
        "overall_status": CONTROL_PASS_ALL,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    path = run_root / "controls/negative-control-summary.json"
    digest = write_negative_control_summary_exclusive(
        path,
        summary,
        run_root=run_root,
        evidence_verifier=_control_verifier,
        verifier_context=CONTROL_VERIFIER_CONTEXT,
    )
    return make_artifact_ref(run_root, path, digest), CONTROL_VERIFIER_CONTEXT.copy()


def _materialize_not_run_controls(run_root: Path) -> tuple[dict, dict]:
    protocol_ref = _publish_json(
        run_root, "hold-controls/authority/method-protocol.json", _control_method_protocol()
    )
    summary = build_not_run_negative_control_summary(
        method_protocol_ref=protocol_ref,
        ordinary_internal_test_row_binding_sha256=CONTROL_TEST_BINDING,
        reason_code="CONTROL_EXECUTION_NOT_COMPLETED",
    )
    path = run_root / "hold-controls/negative-control-summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = write_negative_control_summary_exclusive(
        path,
        summary,
        run_root=run_root,
        evidence_verifier=None,
    )
    return make_artifact_ref(run_root, path, digest), CONTROL_VERIFIER_CONTEXT.copy()


def comparison(*, same_information_set: bool = True, passing: bool = True) -> dict:
    reference = copy.deepcopy(_comparison_fixture(passing))
    if not same_information_set:
        reference["artifact_payload"]["baseline_information_set_fingerprint"] = "b" * 64
    return reference


def test_gate_is_conjunctive_and_sign_cannot_be_substituted() -> None:
    out = evaluate_effect_gate(
        candidate_model_id="A",
        candidate_metrics=metrics(sign=0.59, spearman=0.99),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=comparison(),
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
    )
    assert out["status"] == "HOLD_M0_REPAIR"
    assert out["checks"]["macro_sign_accuracy"]["pass"] is False
    assert out["checks"]["macro_delta_spearman"]["pass"] is True
    assert out["GO_EFFECT"] is False


def test_selection_metric_policy_is_required_even_when_other_metrics_pass() -> None:
    out = evaluate_effect_gate(
        candidate_model_id="A",
        candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=comparison(),
        selection_metric_policy={"status": "HOLD_PENDING_SEPARATE_PREREGISTRATION", "pass": False},
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
    )
    assert out["status"] == "HOLD_SELECTION_METRIC_POLICY"
    assert out["GO_EFFECT"] is False


def test_two_study_nested_endpoint_scope_can_never_be_publication_go() -> None:
    out = evaluate_effect_gate(
        candidate_model_id="A",
        candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=comparison(),
        selection_metric_policy=selection(),
        scientific_scope="DEVELOPMENT_ONLY_TWO_STUDY_NESTED_ENDPOINTS",
    )
    assert out["status"] == "DEVELOPMENT_ONLY_TWO_STUDY_NESTED_ENDPOINTS"
    assert out["GO_EFFECT"] is False


def test_publication_gate_can_pass_only_as_single_model_single_information_set(
    tmp_path: Path,
) -> None:
    reference, verification_context = _materialize_comparison(tmp_path)
    control_reference, control_context = _materialize_passing_controls(tmp_path)
    baseline_reference, baseline_context, baseline_payload = (
        _materialize_baseline_selection(tmp_path)
    )
    assert baseline_payload["status"] == BASELINE_SELECTION_QUALIFIED_STATUS
    out = evaluate_effect_gate(
        candidate_model_id="A",
        candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=reference,
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        ordinary_run_root=tmp_path,
        paired_verification_context=verification_context,
        development_baseline_selection_reference=baseline_reference,
        development_baseline_selection_verification_context=baseline_context,
        negative_control_summary_reference=control_reference,
        negative_control_evidence_verifier=_control_verifier,
        negative_control_verifier_context=control_context,
    )
    assert out["status"] == "GO_EFFECT"
    assert out["GO_EFFECT"] is True
    assert out["downstream_unlock"] is False
    unequal = evaluate_effect_gate(
        candidate_model_id="A",
        candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=comparison(same_information_set=False),
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        ordinary_run_root=tmp_path,
        paired_verification_context=verification_context,
        development_baseline_selection_reference=baseline_reference,
        development_baseline_selection_verification_context=baseline_context,
        negative_control_summary_reference=control_reference,
        negative_control_evidence_verifier=_control_verifier,
        negative_control_verifier_context=control_context,
    )
    assert unequal["GO_EFFECT"] is False
    assert unequal["checks"]["strongest_same_information_set_baseline_win"]["pass"] is False


def test_in_memory_paired_payload_alone_can_never_drive_go() -> None:
    out = evaluate_effect_gate(
        candidate_model_id="A",
        candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=comparison(),
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
    )
    paired = out["checks"]["strongest_same_information_set_baseline_win"]
    assert out["GO_EFFECT"] is False
    assert paired["pass"] is False
    assert paired["status"] == "HOLD_PAIRED_FILE_VERIFICATION_CONTEXT_REQUIRED"


def _evaluate_with_development_baseline_selector(
    run_root: Path,
    reference: dict | None,
    verification_context: dict | None,
) -> dict:
    paired_reference, paired_context = _materialize_comparison(run_root)
    control_reference, control_context = _materialize_passing_controls(run_root)
    return evaluate_effect_gate(
        candidate_model_id="A",
        candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=paired_reference,
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        ordinary_run_root=run_root,
        paired_verification_context=paired_context,
        development_baseline_selection_reference=reference,
        development_baseline_selection_verification_context=verification_context,
        negative_control_summary_reference=control_reference,
        negative_control_evidence_verifier=_control_verifier,
        negative_control_verifier_context=control_context,
    )


def test_development_baseline_selector_reference_and_external_context_are_required(
    tmp_path: Path,
) -> None:
    missing_reference_root = tmp_path / "missing-reference"
    missing_reference_root.mkdir()
    missing_reference = _evaluate_with_development_baseline_selector(
        missing_reference_root, None, None
    )
    check = missing_reference["checks"][
        "development_only_strongest_baseline_selection"
    ]
    assert missing_reference["GO_EFFECT"] is False
    assert missing_reference["status"] == "HOLD_DEVELOPMENT_BASELINE_SELECTION"
    assert check["status"] == (
        "HOLD_DEVELOPMENT_BASELINE_SELECTION_REFERENCE_REQUIRED"
    )

    missing_context_root = tmp_path / "missing-context"
    missing_context_root.mkdir()
    selector_ref, _, _ = _materialize_baseline_selection(missing_context_root)
    missing_context = _evaluate_with_development_baseline_selector(
        missing_context_root, selector_ref, None
    )
    context_check = missing_context["checks"][
        "development_only_strongest_baseline_selection"
    ]
    assert missing_context["GO_EFFECT"] is False
    assert context_check["status"] == (
        "HOLD_DEVELOPMENT_BASELINE_SELECTION_CONTEXT_REQUIRED"
    )

    malformed_reference_root = tmp_path / "malformed-reference"
    malformed_reference_root.mkdir()
    malformed_ref, malformed_context, _ = _materialize_baseline_selection(
        malformed_reference_root
    )
    malformed_ref["self_attested_pass"] = True
    malformed = _evaluate_with_development_baseline_selector(
        malformed_reference_root, malformed_ref, malformed_context
    )
    malformed_check = malformed["checks"][
        "development_only_strongest_baseline_selection"
    ]
    assert malformed["GO_EFFECT"] is False
    assert malformed_check["status"] == (
        "HOLD_DEVELOPMENT_BASELINE_SELECTION_REFERENCE_INVALID"
    )


def test_hold_or_different_selected_baseline_can_never_drive_go(
    tmp_path: Path,
) -> None:
    hold_root = tmp_path / "hold"
    hold_root.mkdir()
    hold_ref, hold_context, hold_payload = _materialize_baseline_selection(
        hold_root, qualified=False
    )
    assert hold_payload["status"] == BASELINE_SELECTION_HOLD_STATUS
    held = _evaluate_with_development_baseline_selector(
        hold_root, hold_ref, hold_context
    )
    held_check = held["checks"][
        "development_only_strongest_baseline_selection"
    ]
    assert held["GO_EFFECT"] is False
    assert held_check["pass"] is False
    assert held_check["status"] == BASELINE_SELECTION_HOLD_STATUS

    different_root = tmp_path / "different"
    different_root.mkdir()
    different_ref, different_context, different_payload = (
        _materialize_baseline_selection(
            different_root,
            selected_method_id="PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
        )
    )
    assert different_payload["status"] == BASELINE_SELECTION_QUALIFIED_STATUS
    assert different_payload["selected_method_id"] == (
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"
    )
    different = _evaluate_with_development_baseline_selector(
        different_root, different_ref, different_context
    )
    different_check = different["checks"][
        "development_only_strongest_baseline_selection"
    ]
    assert different["GO_EFFECT"] is False
    assert different_check["selected_method_matches_paired_baseline"] is False
    assert different_check["paired_baseline_id"] == (
        "FULL_PAIR_UNRESTRICTED_SIAMESE"
    )


def test_selector_information_fingerprint_and_exact_bytes_are_gate_bound(
    tmp_path: Path,
) -> None:
    fingerprint_root = tmp_path / "fingerprint"
    fingerprint_root.mkdir()
    fingerprint_ref, fingerprint_context, _ = _materialize_baseline_selection(
        fingerprint_root, information_set_fingerprint="b" * 64
    )
    fingerprint = _evaluate_with_development_baseline_selector(
        fingerprint_root, fingerprint_ref, fingerprint_context
    )
    fingerprint_check = fingerprint["checks"][
        "development_only_strongest_baseline_selection"
    ]
    assert fingerprint["GO_EFFECT"] is False
    assert fingerprint_check["same_information_set_fingerprint"] is False

    tampered_root = tmp_path / "tampered"
    tampered_root.mkdir()
    tampered_ref, tampered_context, _ = _materialize_baseline_selection(
        tampered_root
    )
    (tampered_root / tampered_ref["relative_path"]).write_bytes(b"{}\n")
    tampered = _evaluate_with_development_baseline_selector(
        tampered_root, tampered_ref, tampered_context
    )
    tampered_check = tampered["checks"][
        "development_only_strongest_baseline_selection"
    ]
    assert tampered["GO_EFFECT"] is False
    assert tampered_check["status"] == (
        "HOLD_DEVELOPMENT_BASELINE_SELECTION_FILE_VERIFICATION_FAILED"
    )
    assert "SHA-256 mismatch" in tampered_check["reason"]


@pytest.mark.parametrize(
    "binding",
    ["run_start", "activation", "authorities", "eligible_order", "required_cells"],
)
def test_selector_replay_binds_every_external_authority_input(
    tmp_path: Path, binding: str
) -> None:
    run_root = tmp_path / binding
    run_root.mkdir()
    selector_ref, selector_context, _ = _materialize_baseline_selection(run_root)
    selector_context = copy.deepcopy(selector_context)
    if binding == "run_start":
        selector_context["expected_run_start_sha256"] = "e" * 64
    elif binding == "activation":
        selector_context["expected_activation_manifest_sha256"] = "e" * 64
    elif binding == "authorities":
        first_authority = NON_DATA_AUTHORITY_HASH_KEYS[0]
        selector_context["expected_non_data_authority_hashes"][
            first_authority
        ] = "e" * 64
    elif binding == "eligible_order":
        selector_context["expected_eligible_method_order"] = [
            BASELINE_METHOD_ORDER[1],
            BASELINE_METHOD_ORDER[0],
            BASELINE_METHOD_ORDER[2],
        ]
    elif binding == "required_cells":
        selector_context["expected_required_estimand_cells"][0]["cells"][0][
            "registered_source_group_ids_v2"
        ] = ["POOL_1", "POOL_2", "POOL_3"]
    out = _evaluate_with_development_baseline_selector(
        run_root, selector_ref, selector_context
    )
    check = out["checks"]["development_only_strongest_baseline_selection"]
    assert out["GO_EFFECT"] is False
    assert check["status"] == (
        "HOLD_DEVELOPMENT_BASELINE_SELECTION_FILE_VERIFICATION_FAILED"
    )


def test_selector_verification_context_rejects_extra_self_attested_fields(
    tmp_path: Path,
) -> None:
    selector_ref, selector_context, _ = _materialize_baseline_selection(tmp_path)
    selector_context["selected_baseline_pass"] = True
    out = _evaluate_with_development_baseline_selector(
        tmp_path, selector_ref, selector_context
    )
    check = out["checks"]["development_only_strongest_baseline_selection"]
    assert out["GO_EFFECT"] is False
    assert check["status"] == (
        "HOLD_DEVELOPMENT_BASELINE_SELECTION_CONTEXT_INVALID"
    )


def test_missing_tampered_and_escaping_paired_files_hold(tmp_path: Path) -> None:
    reference = _comparison_fixture(True)
    context = _comparison_context(True)

    missing = evaluate_effect_gate(
        candidate_model_id="A", candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=reference,
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        ordinary_run_root=tmp_path,
        paired_verification_context=context,
    )
    assert missing["GO_EFFECT"] is False
    assert missing["checks"]["strongest_same_information_set_baseline_win"]["status"] == (
        "HOLD_PAIRED_FILE_VERIFICATION_FAILED"
    )

    materialized, context = _materialize_comparison(tmp_path)
    artifact = tmp_path / materialized["artifact_relative_path"]
    artifact.write_bytes(b"{}\n")
    tampered = evaluate_effect_gate(
        candidate_model_id="A", candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=materialized,
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        ordinary_run_root=tmp_path,
        paired_verification_context=context,
    )
    assert tampered["GO_EFFECT"] is False
    assert "SHA-256 mismatch" in tampered["checks"][
        "strongest_same_information_set_baseline_win"
    ]["reason"]

    run_root = tmp_path / "run"
    run_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(canonical_paired_artifact_bytes(materialized["artifact_payload"]))
    escaping = copy.deepcopy(materialized)
    escaping["artifact_relative_path"] = "../outside.json"
    escaped = evaluate_effect_gate(
        candidate_model_id="A", candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=escaping,
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        ordinary_run_root=run_root,
        paired_verification_context=context,
    )
    assert escaped["GO_EFFECT"] is False
    assert "safe and canonical" in escaped["checks"][
        "strongest_same_information_set_baseline_win"
    ]["reason"]


def _evaluate_with_control_reference(
    run_root: Path,
    control_reference: dict,
    *,
    control_verifier=_control_verifier,
    control_context: dict | None = None,
) -> dict:
    paired_reference, paired_context = _materialize_comparison(run_root)
    return evaluate_effect_gate(
        candidate_model_id="A",
        candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=paired_reference,
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        ordinary_run_root=run_root,
        paired_verification_context=paired_context,
        negative_control_summary_reference=control_reference,
        negative_control_evidence_verifier=control_verifier,
        negative_control_verifier_context=(
            CONTROL_VERIFIER_CONTEXT.copy()
            if control_context is None
            else control_context
        ),
    )


def test_negative_control_summary_requires_strict_exact_file_and_independent_replay(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing = _evaluate_with_control_reference(
        missing_root,
        {"relative_path": "controls/missing.json", "sha256": "a" * 64},
    )
    missing_check = missing["checks"]["preregistered_negative_controls"]
    assert missing["GO_EFFECT"] is False
    assert missing["status"] == "HOLD_NEGATIVE_CONTROLS"
    assert missing_check["status"] == "HOLD_NEGATIVE_CONTROL_FILE_VERIFICATION_FAILED"

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    invalid = _evaluate_with_control_reference(
        invalid_root,
        {
            "relative_path": "controls/summary.json",
            "sha256": "a" * 64,
            "self_attested_status": "PASS",
        },
    )
    assert invalid["GO_EFFECT"] is False
    assert invalid["checks"]["preregistered_negative_controls"]["status"] == (
        "HOLD_NEGATIVE_CONTROL_SUMMARY_REFERENCE_INVALID"
    )

    no_verifier_root = tmp_path / "no-verifier"
    no_verifier_root.mkdir()
    no_verifier_ref, _ = _materialize_passing_controls(no_verifier_root)
    no_verifier = _evaluate_with_control_reference(
        no_verifier_root,
        no_verifier_ref,
        control_verifier=None,
    )
    assert no_verifier["GO_EFFECT"] is False
    assert no_verifier["checks"]["preregistered_negative_controls"]["status"] == (
        "HOLD_NEGATIVE_CONTROL_EVIDENCE_VERIFIER_REQUIRED"
    )

    no_context_root = tmp_path / "no-context"
    no_context_root.mkdir()
    no_context_ref, _ = _materialize_passing_controls(no_context_root)
    paired_reference, paired_context = _materialize_comparison(no_context_root)
    no_context = evaluate_effect_gate(
        candidate_model_id="A",
        candidate_metrics=metrics(),
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=paired_reference,
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        ordinary_run_root=no_context_root,
        paired_verification_context=paired_context,
        negative_control_summary_reference=no_context_ref,
        negative_control_evidence_verifier=_control_verifier,
        negative_control_verifier_context=None,
    )
    assert no_context["GO_EFFECT"] is False
    assert no_context["checks"]["preregistered_negative_controls"]["status"] == (
        "HOLD_NEGATIVE_CONTROL_VERIFIER_CONTEXT_REQUIRED"
    )

    tampered_root = tmp_path / "tampered"
    tampered_root.mkdir()
    tampered_ref, _ = _materialize_passing_controls(tampered_root)
    (tampered_root / tampered_ref["relative_path"]).write_bytes(b"{}\n")
    tampered = _evaluate_with_control_reference(tampered_root, tampered_ref)
    assert tampered["GO_EFFECT"] is False
    tampered_check = tampered["checks"]["preregistered_negative_controls"]
    assert tampered_check["status"] == "HOLD_NEGATIVE_CONTROL_FILE_VERIFICATION_FAILED"
    assert "SHA-256 mismatch" in tampered_check["reason"]


def test_negative_control_test_rows_must_match_candidate_global_prediction(
    tmp_path: Path,
) -> None:
    control_reference, _ = _materialize_passing_controls(
        tmp_path, test_binding="f" * 64
    )
    out = _evaluate_with_control_reference(
        tmp_path,
        control_reference,
        control_context=CONTROL_VERIFIER_CONTEXT.copy(),
    )
    check = out["checks"]["preregistered_negative_controls"]
    assert out["GO_EFFECT"] is False
    assert out["status"] == "HOLD_NEGATIVE_CONTROLS"
    assert check["status"] == "HOLD_NEGATIVE_CONTROL_TEST_ROW_BINDING_MISMATCH"
    assert check["observed_internal_test_row_binding_sha256"] == "f" * 64
    assert check["expected_internal_test_row_binding_sha256"] == CONTROL_TEST_BINDING

    context_root = tmp_path / "wrong-external-context"
    context_root.mkdir()
    matching_control_reference, _ = _materialize_passing_controls(context_root)
    wrong_binding_context = {
        "authority_source": "INDEPENDENT_REPORT_TEST_CONTEXT",
        "expected_internal_test_row_binding_sha256": "f" * 64,
    }
    wrong_context = _evaluate_with_control_reference(
        context_root,
        matching_control_reference,
        control_verifier=_control_verifier_accepting_external_context,
        control_context=wrong_binding_context,
    )
    context_check = wrong_context["checks"]["preregistered_negative_controls"]
    assert wrong_context["GO_EFFECT"] is False
    assert context_check["status"] == (
        "HOLD_NEGATIVE_CONTROL_VERIFIER_CONTEXT_TEST_BINDING_MISMATCH"
    )
    assert context_check["candidate_internal_test_row_binding_sha256"] == (
        CONTROL_TEST_BINDING
    )
    assert context_check["verifier_context_internal_test_row_binding_sha256"] == "f" * 64


def test_not_run_control_summary_stays_hold_without_invoking_completed_evidence_verifier(
    tmp_path: Path,
) -> None:
    control_reference, control_context = _materialize_not_run_controls(tmp_path)

    def reject_if_called(_raw: bytes, _metadata: dict) -> dict:
        raise AssertionError("NOT_RUN summary must not invoke completed-evidence verifier")

    out = _evaluate_with_control_reference(
        tmp_path,
        control_reference,
        control_verifier=reject_if_called,
        control_context=control_context,
    )
    check = out["checks"]["preregistered_negative_controls"]
    assert out["GO_EFFECT"] is False
    assert out["status"] == "HOLD_NEGATIVE_CONTROLS"
    assert check["pass"] is False
    assert check["status"] == "HOLD"
    assert check["control_count"] == 7


def test_go_report_requires_external_run_file_and_recomputation_context(
    tmp_path: Path,
) -> None:
    reference, context = _materialize_comparison(tmp_path)
    control_reference, control_context = _materialize_passing_controls(tmp_path)
    baseline_reference, baseline_context, _ = _materialize_baseline_selection(
        tmp_path
    )
    candidate_metrics = metrics()
    gate = evaluate_effect_gate(
        candidate_model_id="A", candidate_metrics=candidate_metrics,
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        strongest_baseline_comparison=reference,
        selection_metric_policy=selection(),
        scientific_scope="PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        ordinary_run_root=tmp_path,
        paired_verification_context=context,
        development_baseline_selection_reference=baseline_reference,
        development_baseline_selection_verification_context=baseline_context,
        negative_control_summary_reference=control_reference,
        negative_control_evidence_verifier=_control_verifier,
        negative_control_verifier_context=control_context,
    )
    assert gate["GO_EFFECT"] is True
    report = {
        "schema_version": "r3x_ordinary_run_report_v1",
        "status": "ORDINARY_GO_EFFECT_CANDIDATE_EXISTS",
        "scientific_scope": "PUBLICATION_ELIGIBLE_SAME_ESTIMAND",
        "sign_gate": 0.60,
        "active_contract_mutation": False,
        "sealed_access": False,
        "downstream_unlock": False,
        "GO_EFFECT": True,
        "methods": {
            "A": {
                "status": "RUN",
                "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
                "metrics": {"macro": candidate_metrics},
            },
        },
        "effect_gates": {"A": gate},
        "effect_gate_inputs": {
            "A": {
                "candidate_metrics": candidate_metrics,
                "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
                "strongest_baseline_comparison": reference,
                "selection_metric_policy": selection(),
                "development_baseline_selection_reference": baseline_reference,
                "negative_control_summary_reference": control_reference,
            },
        },
    }
    assert validate_run_report(
        report,
        ordinary_run_root=tmp_path,
        paired_verification_context_by_method={"A": context},
        development_baseline_selection_verification_context_by_method={
            "A": baseline_context
        },
        negative_control_evidence_verifier_by_method={"A": _control_verifier},
        negative_control_verifier_context_by_method={"A": control_context},
    )["GO_EFFECT"] is True
    missing_control_input = copy.deepcopy(report)
    del missing_control_input["effect_gate_inputs"]["A"][
        "negative_control_summary_reference"
    ]
    with pytest.raises(ReportProtocolError, match="strict schema"):
        validate_run_report(
            missing_control_input,
            ordinary_run_root=tmp_path,
            paired_verification_context_by_method={"A": context},
            development_baseline_selection_verification_context_by_method={
                "A": baseline_context
            },
            negative_control_evidence_verifier_by_method={"A": _control_verifier},
            negative_control_verifier_context_by_method={"A": control_context},
        )
    malformed_control_input = copy.deepcopy(report)
    malformed_control_input["effect_gate_inputs"]["A"][
        "negative_control_summary_reference"
    ]["self_attested_pass"] = True
    with pytest.raises(ReportProtocolError, match="summary reference differs"):
        validate_run_report(
            malformed_control_input,
            ordinary_run_root=tmp_path,
            paired_verification_context_by_method={"A": context},
            development_baseline_selection_verification_context_by_method={
                "A": baseline_context
            },
            negative_control_evidence_verifier_by_method={"A": _control_verifier},
            negative_control_verifier_context_by_method={"A": control_context},
        )
    with pytest.raises(ReportProtocolError, match="strict recomputation"):
        validate_run_report(
            report,
            ordinary_run_root=tmp_path,
            paired_verification_context_by_method={"A": context},
            development_baseline_selection_verification_context_by_method={
                "A": baseline_context
            },
            negative_control_verifier_context_by_method={"A": control_context},
        )
    with pytest.raises(ReportProtocolError, match="strict recomputation"):
        validate_run_report(report)
    (tmp_path / reference["artifact_relative_path"]).write_bytes(b"{}\n")
    with pytest.raises(ReportProtocolError, match="strict recomputation"):
        validate_run_report(
            report,
            ordinary_run_root=tmp_path,
            paired_verification_context_by_method={"A": context},
            development_baseline_selection_verification_context_by_method={
                "A": baseline_context
            },
            negative_control_evidence_verifier_by_method={"A": _control_verifier},
            negative_control_verifier_context_by_method={"A": control_context},
        )


def test_synthetic_pipeline_is_engineering_only_and_exercises_a_then_b(tmp_path: Path) -> None:
    output = tmp_path / "synthetic-r3x"
    report = run_synthetic_engineering_fixture(output_dir=output, seed=11, epochs=1, n_rows=30)
    assert report["status"] == "SYNTHETIC_ENGINEERING_ONLY"
    assert report["sealed_access"] is False
    assert report["GO_EFFECT"] is False
    assert report["downstream_unlock"] is False
    assert report["execution_order"][:3] == [
        "PUBLICATION_GRADE_STRONG_BASELINES",
        "FULL_PAIR_UNRESTRICTED_SIAMESE",
        "R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION",
    ]
    assert report["methods"]["FULL_PAIR_UNRESTRICTED_SIAMESE"]["status"] == "RUN"
    assert report["methods"]["R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION"]["status"] == "RUN"
    assert report["methods"]["R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION"]["strongest_baseline_eligible"] is False
    assert report["methods"]["FULL_PAIR_UNRESTRICTED_SIAMESE"]["parameter_gap_fraction_vs_A"] <= 0.05
    paired = [
        report["methods"][method_id]
        for method_id in (
            "FULL_PAIR_UNRESTRICTED_SIAMESE",
            "R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION",
            "A_EXACT_POTENTIAL_DIFFERENCE",
        )
    ]
    assert len({method["information_set_fingerprint"] for method in paired}) == 1
    assert len({method["test_predictions"]["row_set_sha256"] for method in paired}) == 1
    assert "B_SPARSE_EDIT_SET_ADDITIVE" in report["methods"]
    assert report["methods"]["B_SPARSE_EDIT_SET_INTERACTION"]["status"] == "NOT_RUN_ACTION_SUPPORT"
    assert (output / "R3X_RUN_REPORT.json").is_file()
    validate_run_report(json.loads((output / "R3X_RUN_REPORT.json").read_text()))


def test_synthetic_runner_executes_neural_families_in_authorized_order(monkeypatch) -> None:
    trace = []

    def fake_train(model, *args, **kwargs):
        trace.append(model.architecture_id)
        return {
            "status": "RUN",
            "architecture_id": model.architecture_id,
            "information_set": model.information_set,
            "parameter_count": ordinary_runner.count_parameters(model),
        }

    monkeypatch.setattr(ordinary_runner, "_train_neural", fake_train)
    report = ordinary_runner.run_synthetic_engineering_fixture(
        output_dir=None, seed=11, epochs=1, n_rows=30
    )
    assert report["status"] == "SYNTHETIC_ENGINEERING_ONLY"
    assert trace == [
        "R3X_FULL_PAIR_UNRESTRICTED_SIAMESE_V1",
        "R3X_R2_STYLE_NO_EXPLICIT_STUDY_CROSS_ATTENTION_ADAPTATION_V1",
        "R3X_A_EXACT_POTENTIAL_DIFFERENCE_V1",
        "R3X_A_PARAMETER_MATCHED_UNRESTRICTED_SIAMESE_V1",
        "R3X_B_SPARSE_EDIT_SET_ADDITIVE_V1",
    ]


def test_report_write_is_exclusive_and_protected_paths_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_json_exclusive(path, {"status": "TEST"})
    with pytest.raises(ReportProtocolError, match="exists"):
        write_json_exclusive(path, {"status": "OVERWRITE"})
    protected = tmp_path / "sealed" / "report.json"
    with pytest.raises(ReportProtocolError, match="PROTECTED"):
        write_json_exclusive(protected, {"status": "NO"})


def test_unknown_scope_and_forged_go_report_are_rejected() -> None:
    with pytest.raises(ReportProtocolError, match="unknown scientific scope"):
        evaluate_effect_gate(
            candidate_model_id="A", candidate_metrics=metrics(),
            information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
            strongest_baseline_comparison=comparison(),
            selection_metric_policy=selection(), scientific_scope="ORDINARY_SCOPE_UNKNOWN",
        )
    forged = {
        "schema_version": "r3x_ordinary_run_report_v1",
        "status": "ORDINARY_GO_EFFECT_CANDIDATE_EXISTS",
        "scientific_scope": "DEVELOPMENT_ONLY_SINGLE_STUDY",
        "sign_gate": 0.01,
        "active_contract_mutation": True,
        "sealed_access": False,
        "downstream_unlock": False,
        "GO_EFFECT": True,
        "methods": {"fake": {"status": "RUN"}},
        "effect_gates": {},
    }
    with pytest.raises(ReportProtocolError):
        validate_run_report(forged)


def test_method_protocol_freezes_route_but_keeps_input_bindings_on_hold() -> None:
    protocol = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/r3x/r3_ordinary_experiment_protocol_v1.json")
        .read_text(encoding="utf-8")
    )
    assert protocol["status"] == "FROZEN_METHOD_PROTOCOL_INPUT_BINDINGS_PENDING"
    assert protocol["seeds"] == [11, 23, 42, 67, 101]
    assert protocol["effect_gate"]["macro_sign_accuracy"] == ">=0.60"
    assert protocol["effect_gate"]["stable_top_selection_metric"].startswith("HOLD")
    assert protocol["models"]["B_sparse_edit_set_interaction"]["status"].startswith("NOT_RUN")
    assert protocol["hard_boundaries"]["sealed_member_access"] is False
    assert protocol["hard_boundaries"]["sign_gate_change"] is False
