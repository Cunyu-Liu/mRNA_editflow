from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.r3x.artifact_graph import (
    PRIMARY_INFORMATION_SET,
    make_artifact_ref,
    read_artifact_ref,
    write_prediction_jsonl_exclusive,
)
from scripts.r3x.artifact_io import atomic_publish_bytes, canonical_json_bytes
from scripts.r3x.baseline_selection import (
    DEVELOPMENT_PARTITION,
    HOLD_STATUS,
    METRIC_POLICY,
    MINIMUM_REQUIRED_METHOD_IDS,
    QUALIFIED_STATUS,
    RUNNER_INTEGRATION_REQUIREMENTS,
    SELECTION_SCHEMA,
    BaselineSelectionError,
    build_development_baseline_evidence,
    build_development_baseline_selection,
    canonical_development_baseline_evidence_bytes,
    canonical_development_baseline_selection_bytes,
    load_development_baseline_selection_exact,
    replay_development_baseline_evidence,
    validate_development_baseline_evidence,
    validate_development_baseline_selection,
    write_development_baseline_evidence_exclusive,
    write_development_evaluation_jsonl_exclusive,
    write_development_baseline_selection_exclusive,
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


METHOD_ORDER = [
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
    "PAIRED_FEATURE_NEAREST_NEIGHBOR",
    "HISTOGRAM_GRADIENT_BOOSTING",
]
RUN_START_SHA = "c" * 64
ACTIVATION_SHA = "d" * 64
INFORMATION_FINGERPRINT = "e" * 64
TRAIN_DEVELOPMENT_OUTCOMES_SHA = "b" * 64


def _primary_predictor_input(estimand_id: str, cell_id: str) -> dict:
    return {
        "endpoint": f"{estimand_id}_ENDPOINT",
        "publication_context_id": f"{cell_id}_CONTEXT",
        "region": "5UTR",
        "normalized_source_sequence": "ACGUAC",
        "normalized_candidate_sequence": "AGGUAC",
        "edit_list": [{"op": "SUB", "pos": 1, "ref": "C", "alt": "G"}],
        "edit_count": 1,
    }


def _authorities() -> dict[str, str]:
    return {
        name: format(index + 1, "x") * 64
        for index, name in enumerate(NON_DATA_AUTHORITY_HASH_KEYS)
    }


def _default_modes() -> dict[str, str]:
    return {
        # FullPair/HGB: sign=.60, rho=1.0; ridge: sign=.80, rho=.60;
        # nearest neighbor: sign=.40, rho=1.0.  All are recomputed from rows.
        "FULL_PAIR_UNRESTRICTED_SIAMESE": "SHIFT_1_1",
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE": "SCRAMBLED_HIGH_SIGN",
        "PAIRED_FEATURE_NEAREST_NEIGHBOR": "SHIFT_2_1",
        "HISTOGRAM_GRADIENT_BOOSTING": "SHIFT_1_1",
    }


def _row_id(
    estimand: str, cell: str, pool: str, index: int, suffix: str = ""
) -> str:
    return hashlib.sha256(
        f"{estimand}/{cell}/{pool}/{index}/{suffix}".encode("utf-8")
    ).hexdigest()


def _evaluation_rows(
    estimand: str, cell: str, *, suffix: str = ""
) -> list[dict]:
    rows = []
    truths = (-2.0, -1.0, 0.2, 1.0, 2.0)
    for pool_index in (1, 2):
        pool = f"POOL_{pool_index}"
        component = f"COMPONENT_{pool_index}"
        for index, true_delta in enumerate(truths, start=1):
            rows.append({
                "schema_version": "r3x_development_evaluation_row_v1",
                "row_id": _row_id(estimand, cell, pool, index, suffix),
                "partition": DEVELOPMENT_PARTITION,
                "estimand_id": estimand,
                "cell_id": f"{estimand}:{cell}",
                "true_delta": true_delta,
                "source_component_id": component,
                "source_group_id_v2": pool,
            })
    return rows


def _prediction_rows(
    estimand: str,
    cell: str,
    *,
    mode: str = "SHIFT_1_1",
    suffix: str = "",
) -> list[dict]:
    rows = []
    scramble = (-2.0, -1.0, 2.0, 0.2, -0.5)
    low_rank_sign_0_4 = (-1.0, -2.0, -3.0, -4.0, -5.0)
    higher_rank_sign_0_2 = (0.1, 0.2, -3.0, -2.0, 1.0)
    evaluations = _evaluation_rows(estimand, cell, suffix=suffix)
    for row_index, evaluation in enumerate(evaluations):
        true_delta = evaluation["true_delta"]
        pool_position = row_index % 5
        if mode == "SHIFT_1_1":
            prediction = true_delta - 1.1
        elif mode == "SCRAMBLED_HIGH_SIGN":
            prediction = scramble[pool_position]
        elif mode == "SHIFT_2_1":
            prediction = true_delta - 2.1
        elif mode == "PLUS_0_1":
            prediction = true_delta + 0.1
        elif mode == "PLUS_0_2":
            prediction = true_delta + 0.2
        elif mode == "LOW_RANK_SIGN_0_4":
            prediction = low_rank_sign_0_4[pool_position]
        elif mode == "HIGHER_RANK_SIGN_0_2":
            prediction = higher_rank_sign_0_2[pool_position]
        elif mode == "NEGATED":
            prediction = -true_delta
        else:
            raise AssertionError(f"unknown fixture prediction mode: {mode}")
        rows.append({
            "schema_version": "r3x_prediction_row_v1",
            "row_id": evaluation["row_id"],
            "predicted_delta_mean": prediction,
            "predicted_delta_logvar": None,
        })
    return rows


def _cell_input(
    prediction_ref: dict,
    evaluation_ref: dict,
    cell_id: str,
) -> dict:
    return {
        "cell_id": cell_id,
        "prediction_jsonl_ref": prediction_ref,
        "development_evaluation_jsonl_ref": evaluation_ref,
    }


def _required_layout() -> list[dict]:
    return [
        {
            "estimand_id": estimand_id,
            "cells": [
                {
                    "cell_id": f"{estimand_id}:{cell_id}",
                    "registered_source_group_ids_v2": ["POOL_1", "POOL_2"],
                }
                for cell_id in ("CELL_1", "CELL_2")
            ],
        }
        for estimand_id in ("ESTIMAND_A", "ESTIMAND_B")
    ]


def _write_json_ref(run_root: Path, path: Path, payload: dict) -> dict:
    digest = atomic_publish_bytes(path, canonical_json_bytes(payload))
    return make_artifact_ref(run_root, path, digest)


def _predictor_verifier(method_id: str) -> PredictorVerifier:
    authorities = _authorities()

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
        verifier_id=f"{method_id}.EXACT_REPLAY_V1",
        code_sha256=hashlib.sha256(
            f"verifier/{method_id}".encode("utf-8")
        ).hexdigest(),
        expected_predictor_code_sha256=authorities[
            "analysis_code_tree_manifest_sha256"
        ],
        expected_predictor_config_sha256=authorities[
            "model_optimizer_seed_config_sha256"
        ],
        recompute=recompute,
    )


def _materialize(
    tmp_path: Path,
    *,
    mode_by_method: dict[str, str] | None = None,
    fingerprint_by_method: dict[str, str] | None = None,
    row_suffix_by_method: dict[str, str] | None = None,
    omit_last_cell_for: str | None = None,
) -> dict:
    run_root = tmp_path / "ordinary_run"
    development = run_root / "development"
    development.mkdir(parents=True)
    mode_by_method = mode_by_method or _default_modes()
    fingerprint_by_method = fingerprint_by_method or {}
    row_suffix_by_method = row_suffix_by_method or {}
    evidence_refs = {}
    evidence_payloads = {}
    predictor_provenance_refs = {}
    predictor_verifiers = {}
    prediction_paths = {}
    evaluation_paths = {}
    common_evaluation_refs = {}
    evaluation_root = development / "evaluation"
    evaluation_root.mkdir()
    for estimand_id in ("ESTIMAND_A", "ESTIMAND_B"):
        for cell_id in ("CELL_1", "CELL_2"):
            evaluation_path = evaluation_root / f"{estimand_id}_{cell_id}.jsonl"
            evaluation_sha = write_development_evaluation_jsonl_exclusive(
                evaluation_path, _evaluation_rows(estimand_id, cell_id)
            )
            common_evaluation_refs[(estimand_id, cell_id)] = make_artifact_ref(
                run_root, evaluation_path, evaluation_sha
            )
            evaluation_paths[("COMMON", estimand_id, cell_id)] = evaluation_path
    for method in METHOD_ORDER:
        method_root = development / method.lower()
        method_root.mkdir()
        estimands = []
        provenance_estimands = []
        state_predictions = {}
        verifier = _predictor_verifier(method)
        predictor_verifiers[verifier.verifier_id] = verifier
        for estimand_id in ("ESTIMAND_A", "ESTIMAND_B"):
            cells = []
            provenance_cells = []
            for cell_id in ("CELL_1", "CELL_2"):
                if (
                    (method == omit_last_cell_for or omit_last_cell_for == "ALL")
                    and estimand_id == "ESTIMAND_B"
                    and cell_id == "CELL_2"
                ):
                    continue
                suffix = row_suffix_by_method.get(method, "")
                if suffix:
                    evaluation_path = method_root / f"{estimand_id}_{cell_id}_evaluation.jsonl"
                    evaluation_sha = write_development_evaluation_jsonl_exclusive(
                        evaluation_path,
                        _evaluation_rows(estimand_id, cell_id, suffix=suffix),
                    )
                    evaluation_ref = make_artifact_ref(
                        run_root, evaluation_path, evaluation_sha
                    )
                    evaluation_paths[(method, estimand_id, cell_id)] = evaluation_path
                else:
                    evaluation_ref = common_evaluation_refs[(estimand_id, cell_id)]
                prediction_path = method_root / f"{estimand_id}_{cell_id}.jsonl"
                prediction_rows = _prediction_rows(
                    estimand_id,
                    cell_id,
                    mode=mode_by_method[method],
                    suffix=suffix,
                )
                prediction_sha = write_prediction_jsonl_exclusive(
                    prediction_path,
                    prediction_rows,
                )
                prediction_paths[(method, estimand_id, cell_id)] = prediction_path
                prediction_ref = make_artifact_ref(
                    run_root, prediction_path, prediction_sha
                )
                cells.append(
                    _cell_input(
                        prediction_ref,
                        evaluation_ref,
                        f"{estimand_id}:{cell_id}",
                    )
                )
                frozen_cell_id = f"{estimand_id}:{cell_id}"
                train_input_rows = [{
                    "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
                    "row_id": hashlib.sha256(
                        f"{method}/{estimand_id}/{cell_id}/train/{index}".encode()
                    ).hexdigest(),
                    "partition": TRAIN_PARTITION,
                    "estimand_id": estimand_id,
                    "cell_id": frozen_cell_id,
                    "predictor_input": _primary_predictor_input(
                        estimand_id, cell_id
                    ),
                    "training_target": float(index),
                } for index in range(1, 4)]
                development_input_rows = [{
                    "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
                    "row_id": row["row_id"],
                    "partition": PREDICTOR_DEVELOPMENT_PARTITION,
                    "estimand_id": estimand_id,
                    "cell_id": frozen_cell_id,
                    "predictor_input": _primary_predictor_input(
                        estimand_id, cell_id
                    ),
                    "training_target": None,
                } for index, row in enumerate(prediction_rows, start=1)]
                train_path = method_root / f"{estimand_id}_{cell_id}_train.jsonl"
                development_path = (
                    method_root / f"{estimand_id}_{cell_id}_development.jsonl"
                )
                train_sha = write_predictor_input_jsonl_exclusive(
                    train_path, train_input_rows
                )
                development_sha = write_predictor_input_jsonl_exclusive(
                    development_path, development_input_rows
                )
                state_predictions.update({
                    row["row_id"]: row["predicted_delta_mean"]
                    for row in prediction_rows
                })
                provenance_cells.append({
                    "cell_id": frozen_cell_id,
                    "train_rows_ref": make_artifact_ref(
                        run_root, train_path, train_sha
                    ),
                    "development_rows_ref": make_artifact_ref(
                        run_root, development_path, development_sha
                    ),
                    "prediction_jsonl_ref": prediction_ref,
                })
            estimands.append({"estimand_id": estimand_id, "cells": cells})
            provenance_estimands.append({
                "estimand_id": estimand_id,
                "estimand_binding_sha256": hashlib.sha256(
                    f"estimand/{estimand_id}".encode()
                ).hexdigest(),
                "cells": provenance_cells,
            })
        state_ref = _write_json_ref(
            run_root,
            method_root / "fitted_state.json",
            {"predictions": state_predictions},
        )
        fit_config_ref = _write_json_ref(
            run_root,
            method_root / "fit_config.json",
            {"method_id": method, "fit_scope": "TRAIN_ONLY"},
        )
        for estimand in provenance_estimands:
            for cell in estimand["cells"]:
                cell["predictor_artifacts"] = {
                    "kind": SCALAR_KIND,
                    "fitted_state_ref": state_ref,
                    "fit_config_ref": fit_config_ref,
                }
        method_fingerprint = fingerprint_by_method.get(
            method, INFORMATION_FINGERPRINT
        )
        provenance = build_predictor_provenance(
            run_root,
            run_start_sha256=RUN_START_SHA,
            activation_manifest_sha256=ACTIVATION_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
            method_id=method,
            information_set_fingerprint=method_fingerprint,
            predictor_code_sha256=_authorities()[
                "analysis_code_tree_manifest_sha256"
            ],
            predictor_config_sha256=_authorities()[
                "model_optimizer_seed_config_sha256"
            ],
            verifier=verifier,
            estimands=provenance_estimands,
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
        predictor_provenance_refs[method] = provenance_ref
        evidence = build_development_baseline_evidence(
            run_root,
            run_start_sha256=RUN_START_SHA,
            activation_manifest_sha256=ACTIVATION_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
            method_id=method,
            information_set_fingerprint=method_fingerprint,
            predictor_provenance_ref=provenance_ref,
            predictor_verifiers=predictor_verifiers,
            estimands=estimands,
        )
        evidence_path = method_root / "development_evidence.json"
        evidence_sha = write_development_baseline_evidence_exclusive(
            run_root,
            evidence_path,
            evidence,
            predictor_verifiers=predictor_verifiers,
        )
        evidence_refs[method] = make_artifact_ref(
            run_root, evidence_path, evidence_sha
        )
        evidence_payloads[method] = evidence
    return {
        "run_root": run_root,
        "evidence_refs": evidence_refs,
        "evidence_payloads": evidence_payloads,
        "prediction_paths": prediction_paths,
        "evaluation_paths": evaluation_paths,
        "predictor_provenance_refs": predictor_provenance_refs,
        "predictor_verifiers": predictor_verifiers,
    }


def _select(
    bundle: dict,
    order: list[str] | None = None,
    evidence_refs: dict[str, dict] | None = None,
) -> dict:
    return build_development_baseline_selection(
        bundle["run_root"],
        run_start_sha256=RUN_START_SHA,
        activation_manifest_sha256=ACTIVATION_SHA,
        train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
        non_data_authority_hashes=_authorities(),
        eligible_method_order=order or METHOD_ORDER,
        development_evidence_refs=(
            bundle["evidence_refs"] if evidence_refs is None else evidence_refs
        ),
        required_estimand_cells=_required_layout(),
        predictor_verifiers=bundle["predictor_verifiers"],
    )


def test_selector_uses_raw_sign_then_spearman_without_lowering_gate(
    tmp_path: Path,
) -> None:
    selection = _select(_materialize(tmp_path))
    assert selection["status"] == QUALIFIED_STATUS
    assert selection["selected_method_id"] == "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"
    assert selection["selected_development_metrics"] == pytest.approx({
        "macro_component_cell_sign_accuracy": 0.80,
        "macro_within_pool_spearman": 0.60,
        "macro_component_cell_mae": 1.02,
    })
    records = {record["method_id"]: record for record in selection["method_selection_records"]}
    assert records["PAIRED_LOCAL_GLOBAL_KMER_RIDGE"]["lexicographic_selection_tuple"][:2] == pytest.approx([
        0.80,
        0.60,
    ])
    assert records["PAIRED_FEATURE_NEAREST_NEIGHBOR"][
        "lexicographic_selection_tuple"
    ][0] == pytest.approx(0.40)
    assert selection["comparator_selected"] is True
    assert selection["baseline_sign_gate_met"] is True
    assert selection["strongest_same_information_claim"] is True
    assert selection["can_satisfy_pretest_freeze"] is True
    assert selection["metric_policy"] == METRIC_POLICY
    assert selection["partition"] == DEVELOPMENT_PARTITION


def test_mae_then_prefrozen_method_order_break_exact_ties(tmp_path: Path) -> None:
    modes = {
        method: "PLUS_0_2"
        for method in METHOD_ORDER
    }
    modes["PAIRED_LOCAL_GLOBAL_KMER_RIDGE"] = "PLUS_0_1"
    bundle = _materialize(tmp_path / "mae", mode_by_method=modes)
    assert _select(bundle)["selected_method_id"] == "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"

    tied = {
        method: "PLUS_0_1"
        for method in METHOD_ORDER
    }
    tied_bundle = _materialize(tmp_path / "order", mode_by_method=tied)
    custom_order = [
        "PAIRED_FEATURE_NEAREST_NEIGHBOR",
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
        "HISTOGRAM_GRADIENT_BOOSTING",
        "FULL_PAIR_UNRESTRICTED_SIAMESE",
    ]
    selection = _select(tied_bundle, order=custom_order)
    assert selection["selected_method_id"] == "PAIRED_FEATURE_NEAREST_NEIGHBOR"
    assert selection["eligible_method_order"] == custom_order


def test_no_method_meeting_sign_gate_is_explicit_hold_not_silent_qualification(
    tmp_path: Path,
) -> None:
    modes = {
        "FULL_PAIR_UNRESTRICTED_SIAMESE": "LOW_RANK_SIGN_0_4",
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE": "HIGHER_RANK_SIGN_0_2",
        "PAIRED_FEATURE_NEAREST_NEIGHBOR": "NEGATED",
        "HISTOGRAM_GRADIENT_BOOSTING": "HIGHER_RANK_SIGN_0_2",
    }
    selection = _select(_materialize(tmp_path, mode_by_method=modes))
    assert selection["status"] == HOLD_STATUS
    # A recovery comparator remains deterministic, but it cannot satisfy the
    # unchanged gate, pre-test freeze, or downstream unlock.
    assert selection["selected_method_id"] == "FULL_PAIR_UNRESTRICTED_SIAMESE"
    assert selection["selected_development_metrics"][
        "macro_component_cell_sign_accuracy"
    ] == pytest.approx(0.40)
    assert selection["selected_development_metrics"][
        "macro_within_pool_spearman"
    ] == pytest.approx(-1.0)
    ridge = next(
        record
        for record in selection["method_selection_records"]
        if record["method_id"] == "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"
    )
    assert ridge["development_metrics"][
        "macro_component_cell_sign_accuracy"
    ] == pytest.approx(0.20)
    assert ridge["development_metrics"][
        "macro_within_pool_spearman"
    ] == pytest.approx(0.20)
    assert selection["comparator_selected"] is True
    assert selection["baseline_sign_gate_met"] is False
    assert selection["strongest_same_information_claim"] is True
    assert selection["can_satisfy_pretest_freeze"] is False
    assert selection["downstream_unlock"] is False


def test_exact_point_60_boundary_qualifies_without_lowering_gate(
    tmp_path: Path,
) -> None:
    modes = {method: "SHIFT_2_1" for method in METHOD_ORDER}
    modes["FULL_PAIR_UNRESTRICTED_SIAMESE"] = "SHIFT_1_1"
    selection = _select(_materialize(tmp_path, mode_by_method=modes))

    assert selection["selected_method_id"] == "FULL_PAIR_UNRESTRICTED_SIAMESE"
    assert selection["selected_development_metrics"][
        "macro_component_cell_sign_accuracy"
    ] == pytest.approx(0.60)
    assert selection["baseline_sign_gate_met"] is True
    assert selection["status"] == QUALIFIED_STATUS
    assert selection["can_satisfy_pretest_freeze"] is True


def test_recovery_state_cannot_unlock_existing_report_or_pretest_consumers(
    tmp_path: Path,
) -> None:
    from scripts.r3x import report as report_module
    from scripts.r3x import typed_pretest_freeze as pretest_module

    modes = {method: "SHIFT_2_1" for method in METHOD_ORDER}
    bundle = _materialize(tmp_path, mode_by_method=modes)
    selection = _select(bundle)
    path = bundle["run_root"] / "development" / "recovery-selection.json"
    digest = write_development_baseline_selection_exclusive(
        bundle["run_root"],
        path,
        selection,
        predictor_verifiers=bundle["predictor_verifiers"],
    )
    reference = make_artifact_ref(bundle["run_root"], path, digest)
    context = {
        "expected_run_start_sha256": RUN_START_SHA,
        "expected_activation_manifest_sha256": ACTIVATION_SHA,
        "expected_train_development_outcomes_sha256": (
            TRAIN_DEVELOPMENT_OUTCOMES_SHA
        ),
        "expected_non_data_authority_hashes": _authorities(),
        "expected_eligible_method_order": list(METHOD_ORDER),
        "expected_required_estimand_cells": _required_layout(),
        "predictor_verifiers": bundle["predictor_verifiers"],
    }
    fingerprint = selection["information_set_fingerprint"]
    passed, report_state = report_module._development_baseline_selection_pass(
        reference,
        bundle["run_root"],
        context,
        PRIMARY_INFORMATION_SET,
        {
            "baseline_id": selection["selected_method_id"],
            "candidate_information_set_fingerprint": fingerprint,
            "baseline_information_set_fingerprint": fingerprint,
            "information_set": PRIMARY_INFORMATION_SET,
        },
    )

    assert passed is False
    assert report_state["status"] == HOLD_STATUS
    assert selection["status"] != report_module.DEVELOPMENT_BASELINE_QUALIFIED_STATUS
    assert selection["status"] != pretest_module.QUALIFIED_STATUS
    assert selection["can_satisfy_pretest_freeze"] is False
    assert selection["downstream_unlock"] is False


def test_evidence_is_complete_recomputed_and_contains_only_dev_boundaries(
    tmp_path: Path,
) -> None:
    bundle = _materialize(tmp_path)
    evidence = bundle["evidence_payloads"]["FULL_PAIR_UNRESTRICTED_SIAMESE"]
    aggregate = evidence["aggregate_development_metrics"]
    assert {key: aggregate[key] for key in ("n_estimands", "n_cells", "n_rows", "complete")} == {
        "n_estimands": 2,
        "n_cells": 4,
        "n_rows": 40,
        "complete": True,
    }
    assert aggregate["macro_component_cell_sign_accuracy"] == pytest.approx(0.60)
    assert aggregate["macro_within_pool_spearman"] == pytest.approx(1.0)
    assert aggregate["macro_component_cell_mae"] == pytest.approx(1.1)
    assert evidence["test_access"] is False
    assert evidence["sealed_access"] is False
    assert evidence["downstream_unlock"] is False
    assert '"internal_test"' not in json.dumps(evidence, sort_keys=True)
    assert "before any internal-test loader" in RUNNER_INTEGRATION_REQUIREMENTS
    assert MINIMUM_REQUIRED_METHOD_IDS == set(METHOD_ORDER)


def test_missing_hgb_or_any_required_evidence_is_recovery_hold(
    tmp_path: Path,
) -> None:
    bundle = _materialize(tmp_path)
    hgb_recovery = None
    for missing_method in METHOD_ORDER:
        incomplete_refs = dict(bundle["evidence_refs"])
        incomplete_refs.pop(missing_method)
        missing_evidence = _select(bundle, evidence_refs=incomplete_refs)
        assert missing_evidence["status"] == HOLD_STATUS
        assert missing_evidence["comparator_selected"] is True
        assert missing_evidence["strongest_same_information_claim"] is False
        assert missing_evidence["can_satisfy_pretest_freeze"] is False
        assert missing_evidence["missing_required_method_ids"] == []
        assert missing_evidence["missing_required_evidence_method_ids"] == [
            missing_method
        ]
        assert missing_evidence["missing_eligible_evidence_method_ids"] == [
            missing_method
        ]
        if missing_method == "HISTOGRAM_GRADIENT_BOOSTING":
            hgb_recovery = missing_evidence

    assert hgb_recovery is not None
    missing_hgb_refs = dict(bundle["evidence_refs"])
    missing_hgb_refs.pop("HISTOGRAM_GRADIENT_BOOSTING")

    missing_order = _select(
        bundle,
        order=METHOD_ORDER[:-1],
        evidence_refs=missing_hgb_refs,
    )
    assert missing_order["status"] == HOLD_STATUS
    assert missing_order["missing_required_method_ids"] == [
        "HISTOGRAM_GRADIENT_BOOSTING"
    ]
    assert missing_order["strongest_same_information_claim"] is False
    assert missing_order["can_satisfy_pretest_freeze"] is False


def test_information_fingerprint_row_order_and_complete_layout_must_match(
    tmp_path: Path,
) -> None:
    mismatch = _materialize(
        tmp_path / "fingerprint",
        fingerprint_by_method={"PAIRED_FEATURE_NEAREST_NEIGHBOR": "f" * 64},
    )
    with pytest.raises(BaselineSelectionError, match="information-set fingerprint"):
        _select(mismatch)

    row_mismatch = _materialize(
        tmp_path / "rows",
        row_suffix_by_method={"PAIRED_FEATURE_NEAREST_NEIGHBOR": "DIFFERENT"},
    )
    with pytest.raises(BaselineSelectionError, match="rows and order"):
        _select(row_mismatch)

    cell_mismatch = _materialize(
        tmp_path / "cells",
        omit_last_cell_for="PAIRED_FEATURE_NEAREST_NEIGHBOR",
    )
    with pytest.raises(BaselineSelectionError, match="rows and order"):
        _select(cell_mismatch)

    common_omission = _materialize(
        tmp_path / "common_omission", omit_last_cell_for="ALL"
    )
    with pytest.raises(BaselineSelectionError, match="every frozen estimand/cell"):
        _select(common_omission)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda row: row.__setitem__("true_delta", float("nan")), "finite"),
        (lambda row: row.__setitem__("partition", "ORDINARY_INTERNAL_TEST"), "another partition"),
        (lambda row: row.__setitem__("source_component_id", ""), "canonical identifier"),
        (lambda row: row.__setitem__("self_attested_metric", 1.0), "strict schema"),
    ],
)
def test_development_evaluation_rows_are_strict_and_cannot_self_attest_metrics(
    tmp_path: Path, mutator, match: str
) -> None:
    run_root = tmp_path / "ordinary_run"
    dev = run_root / "development"
    dev.mkdir(parents=True)
    rows = _evaluation_rows("E", "C")
    mutator(rows[0])
    with pytest.raises(BaselineSelectionError, match=match):
        write_development_evaluation_jsonl_exclusive(dev / "evaluation.jsonl", rows)


def test_metric_completeness_status_and_duplicate_rows_fail_closed(tmp_path: Path) -> None:
    run_root = tmp_path / "ordinary_run"
    dev = run_root / "development"
    dev.mkdir(parents=True)
    short_evaluation_rows = [
        row
        for index, row in enumerate(_evaluation_rows("E", "C"))
        if index % 5 < 2
    ]
    short_prediction_rows = [
        row
        for row in _prediction_rows("E", "C")
        if row["row_id"] in {item["row_id"] for item in short_evaluation_rows}
    ]
    evaluation_path = dev / "short_evaluation.jsonl"
    evaluation_sha = write_development_evaluation_jsonl_exclusive(
        evaluation_path, short_evaluation_rows
    )
    prediction_path = dev / "short_prediction.jsonl"
    prediction_sha = write_prediction_jsonl_exclusive(
        prediction_path, short_prediction_rows
    )
    first = _cell_input(
        make_artifact_ref(run_root, prediction_path, prediction_sha),
        make_artifact_ref(run_root, evaluation_path, evaluation_sha),
        "E:C",
    )
    with pytest.raises(BaselineSelectionError, match="Spearman metric is incomplete"):
        build_development_baseline_evidence(
            run_root,
            run_start_sha256=RUN_START_SHA,
            activation_manifest_sha256=ACTIVATION_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
            method_id=METHOD_ORDER[0],
            information_set_fingerprint=INFORMATION_FINGERPRINT,
            predictor_provenance_ref={},
            predictor_verifiers={},
            estimands=[{"estimand_id": "E", "cells": [first]}],
        )

    evaluation_path = dev / "evaluation.jsonl"
    evaluation_sha = write_development_evaluation_jsonl_exclusive(
        evaluation_path, _evaluation_rows("E", "C")
    )
    prediction_path = dev / "prediction.jsonl"
    prediction_sha = write_prediction_jsonl_exclusive(
        prediction_path, _prediction_rows("E", "C")
    )
    first = _cell_input(
        make_artifact_ref(run_root, prediction_path, prediction_sha),
        make_artifact_ref(run_root, evaluation_path, evaluation_sha),
        "E:C",
    )
    second = copy.deepcopy(first)
    with pytest.raises(BaselineSelectionError, match="multiple cells"):
        build_development_baseline_evidence(
            run_root,
            run_start_sha256=RUN_START_SHA,
            activation_manifest_sha256=ACTIVATION_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
            method_id=METHOD_ORDER[0],
            information_set_fingerprint=INFORMATION_FINGERPRINT,
            predictor_provenance_ref={},
            predictor_verifiers={},
            estimands=[{"estimand_id": "E", "cells": [first, second]}],
        )


def test_prediction_and_evidence_refs_must_use_development_namespace(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "ordinary_run"
    run_root.mkdir()
    wrong = run_root / "validation"
    wrong.mkdir()
    development = run_root / "development"
    development.mkdir()
    prediction = wrong / "pred.jsonl"
    digest = write_prediction_jsonl_exclusive(
        prediction, _prediction_rows("E", "C")
    )
    evaluation = development / "evaluation.jsonl"
    evaluation_digest = write_development_evaluation_jsonl_exclusive(
        evaluation, _evaluation_rows("E", "C")
    )
    cell = _cell_input(
        make_artifact_ref(run_root, prediction, digest),
        make_artifact_ref(run_root, evaluation, evaluation_digest),
        "E:C",
    )
    with pytest.raises(BaselineSelectionError, match="development namespace"):
        build_development_baseline_evidence(
            run_root,
            run_start_sha256=RUN_START_SHA,
            activation_manifest_sha256=ACTIVATION_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
            method_id=METHOD_ORDER[0],
            information_set_fingerprint=INFORMATION_FINGERPRINT,
            predictor_provenance_ref={},
            predictor_verifiers={},
            estimands=[{"estimand_id": "E", "cells": [cell]}],
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.__setitem__(
            "selected_method_id", "FULL_PAIR_UNRESTRICTED_SIAMESE"
        ),
        lambda payload: payload.__setitem__("status", HOLD_STATUS),
        lambda payload: payload["metric_policy"].__setitem__("sign_gate", 0.59),
        lambda payload: payload.__setitem__("test_access", True),
        lambda payload: payload.__setitem__("sealed_access", True),
        lambda payload: payload.__setitem__("downstream_unlock", True),
        lambda payload: payload.__setitem__("comparator_selected", False),
        lambda payload: payload.__setitem__("baseline_sign_gate_met", False),
        lambda payload: payload.__setitem__(
            "strongest_same_information_claim", False
        ),
        lambda payload: payload.__setitem__("can_satisfy_pretest_freeze", False),
        lambda payload: payload["missing_required_evidence_method_ids"].append(
            "FABRICATED_MISSING_METHOD"
        ),
        lambda payload: payload["method_selection_records"][0].__setitem__(
            "lexicographic_selection_tuple", [1.0, 1.0, 0.0, 0.0]
        ),
        lambda payload: payload["development_layout"][0]["cells"][0].__setitem__(
            "row_set_sha256", "0" * 64
        ),
        lambda payload: payload.__setitem__("posthoc_metric", 1.0),
    ],
)
def test_selector_strict_schema_or_recomputation_rejects_tampering(
    tmp_path: Path, mutator
) -> None:
    selection = copy.deepcopy(_select(_materialize(tmp_path)))
    mutator(selection)
    with pytest.raises(BaselineSelectionError):
        validate_development_baseline_selection(selection)


def test_evidence_aggregate_tampering_is_recomputed(tmp_path: Path) -> None:
    evidence = copy.deepcopy(
        _materialize(tmp_path)["evidence_payloads"][METHOD_ORDER[0]]
    )
    evidence["aggregate_development_metrics"][
        "macro_component_cell_sign_accuracy"
    ] = 1.0
    with pytest.raises(BaselineSelectionError, match="do not match"):
        validate_development_baseline_evidence(evidence)


def test_missing_predictor_ref_and_v1_evidence_or_selector_never_qualify(
    tmp_path: Path,
) -> None:
    bundle = _materialize(tmp_path)
    evidence = copy.deepcopy(bundle["evidence_payloads"][METHOD_ORDER[0]])
    evidence.pop("predictor_provenance_ref")
    with pytest.raises(BaselineSelectionError, match="strict schema"):
        validate_development_baseline_evidence(evidence)

    evidence = copy.deepcopy(bundle["evidence_payloads"][METHOD_ORDER[0]])
    evidence["schema_version"] = "r3x_development_baseline_evidence_v1"
    with pytest.raises(BaselineSelectionError, match="unsupported"):
        validate_development_baseline_evidence(evidence)

    assert SELECTION_SCHEMA == "r3x_development_strongest_baseline_selection_v3"
    selection = copy.deepcopy(_select(bundle))
    selection["schema_version"] = "r3x_development_strongest_baseline_selection_v2"
    with pytest.raises(BaselineSelectionError, match="unsupported"):
        validate_development_baseline_selection(selection)

    selection["schema_version"] = "r3x_development_strongest_baseline_selection_v1"
    with pytest.raises(BaselineSelectionError, match="unsupported"):
        validate_development_baseline_selection(selection)


def test_selector_replays_truth_rows_and_rejects_self_attested_cell_metrics(
    tmp_path: Path,
) -> None:
    bundle = _materialize(tmp_path)
    method = METHOD_ORDER[0]
    fabricated = copy.deepcopy(bundle["evidence_payloads"][method])
    for estimand in fabricated["estimands"]:
        for cell in estimand["cells"]:
            cell["component_cell_sign_accuracy"] = 1.0
    fabricated["aggregate_development_metrics"][
        "macro_component_cell_sign_accuracy"
    ] = 1.0
    # Schema-only validation can establish internal consistency, but the
    # scientific selector always performs exact label/prediction replay.
    validate_development_baseline_evidence(fabricated)
    fabricated_path = (
        bundle["run_root"] / "development" / "fabricated_evidence.json"
    )
    with pytest.raises(BaselineSelectionError, match="do not replay exact truth"):
        write_development_baseline_evidence_exclusive(
            bundle["run_root"],
            fabricated_path,
            fabricated,
            predictor_verifiers=bundle["predictor_verifiers"],
        )
    # Simulate an externally forged but internally consistent artifact.  The
    # public exact writer would reject it before publication.
    fabricated_path.write_bytes(
        canonical_development_baseline_evidence_bytes(fabricated)
    )
    fabricated_sha = hashlib.sha256(fabricated_path.read_bytes()).hexdigest()
    refs = dict(bundle["evidence_refs"])
    refs[method] = make_artifact_ref(
        bundle["run_root"], fabricated_path, fabricated_sha
    )
    with pytest.raises(BaselineSelectionError, match="do not replay exact truth"):
        build_development_baseline_selection(
            bundle["run_root"],
            run_start_sha256=RUN_START_SHA,
            activation_manifest_sha256=ACTIVATION_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
            eligible_method_order=METHOD_ORDER,
            development_evidence_refs=refs,
            required_estimand_cells=_required_layout(),
            predictor_verifiers=bundle["predictor_verifiers"],
        )


def test_exact_writer_loader_replays_all_evidence_and_prediction_hashes(
    tmp_path: Path,
) -> None:
    bundle = _materialize(tmp_path)
    selection = _select(bundle)
    path = bundle["run_root"] / "baseline_selection.json"
    digest = write_development_baseline_selection_exclusive(
        bundle["run_root"],
        path,
        selection,
        predictor_verifiers=bundle["predictor_verifiers"],
    )
    assert path.read_bytes() == canonical_development_baseline_selection_bytes(selection)
    loaded = load_development_baseline_selection_exact(
        bundle["run_root"],
        path,
        digest,
        expected_run_start_sha256=RUN_START_SHA,
        expected_activation_manifest_sha256=ACTIVATION_SHA,
        expected_train_development_outcomes_sha256=(
            TRAIN_DEVELOPMENT_OUTCOMES_SHA
        ),
        expected_non_data_authority_hashes=_authorities(),
        expected_eligible_method_order=METHOD_ORDER,
        expected_required_estimand_cells=_required_layout(),
        predictor_verifiers=bundle["predictor_verifiers"],
    )
    assert loaded == selection
    with pytest.raises(BaselineSelectionError, match="already exists"):
        write_development_baseline_selection_exclusive(
            bundle["run_root"],
            path,
            selection,
            predictor_verifiers=bundle["predictor_verifiers"],
        )
    with pytest.raises(BaselineSelectionError, match="SHA-256 mismatch"):
        load_development_baseline_selection_exact(
            bundle["run_root"],
            path,
            "0" * 64,
            predictor_verifiers=bundle["predictor_verifiers"],
        )

    # Exact selector replay detects later mutation of any prediction it binds.
    prediction_path = bundle["prediction_paths"][(
        METHOD_ORDER[0],
        "ESTIMAND_A",
        "CELL_1",
    )]
    prediction_path.write_bytes(b'{"tampered":true}\n')
    with pytest.raises(BaselineSelectionError, match="SHA-256 mismatch"):
        load_development_baseline_selection_exact(
            bundle["run_root"],
            path,
            digest,
            predictor_verifiers=bundle["predictor_verifiers"],
        )


def test_evidence_exact_replay_and_canonical_bytes(tmp_path: Path) -> None:
    bundle = _materialize(tmp_path)
    method = METHOD_ORDER[0]
    evidence = replay_development_baseline_evidence(
        bundle["run_root"],
        bundle["evidence_refs"][method],
        predictor_verifiers=bundle["predictor_verifiers"],
        expected_run_start_sha256=RUN_START_SHA,
        expected_activation_manifest_sha256=ACTIVATION_SHA,
        expected_train_development_outcomes_sha256=(
            TRAIN_DEVELOPMENT_OUTCOMES_SHA
        ),
        expected_non_data_authority_hashes=_authorities(),
        expected_method_id=method,
        expected_information_set_fingerprint=INFORMATION_FINGERPRINT,
    )
    evidence_path = (
        bundle["run_root"]
        / bundle["evidence_refs"][method]["relative_path"]
    )
    assert evidence_path.read_bytes() == canonical_development_baseline_evidence_bytes(
        evidence
    )


def test_exact_selector_replay_detects_development_truth_mutation(
    tmp_path: Path,
) -> None:
    bundle = _materialize(tmp_path)
    selection = _select(bundle)
    path = bundle["run_root"] / "selection.json"
    digest = write_development_baseline_selection_exclusive(
        bundle["run_root"],
        path,
        selection,
        predictor_verifiers=bundle["predictor_verifiers"],
    )
    evaluation_path = bundle["evaluation_paths"][(
        "COMMON",
        "ESTIMAND_A",
        "CELL_1",
    )]
    evaluation_path.write_bytes(b'{"tampered_true_delta":999}\n')
    with pytest.raises(BaselineSelectionError, match="SHA-256 mismatch"):
        load_development_baseline_selection_exact(
            bundle["run_root"],
            path,
            digest,
            predictor_verifiers=bundle["predictor_verifiers"],
        )


def test_exact_loader_rejects_duplicate_keys_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "ordinary_run"
    run_root.mkdir()
    duplicate = run_root / "duplicate.json"
    duplicate.write_bytes(b'{"schema_version":"first","schema_version":"second"}\n')
    digest = hashlib.sha256(duplicate.read_bytes()).hexdigest()
    with pytest.raises(BaselineSelectionError, match="duplicate JSON key"):
        load_development_baseline_selection_exact(
            run_root, duplicate, digest, predictor_verifiers={}
        )

    bundle = _materialize(tmp_path / "canonical")
    selection = _select(bundle)
    noncanonical = bundle["run_root"] / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(selection, sort_keys=False, separators=(",", ":")),
        encoding="utf-8",
    )
    noncanonical_digest = hashlib.sha256(noncanonical.read_bytes()).hexdigest()
    with pytest.raises(BaselineSelectionError, match="not canonical"):
        load_development_baseline_selection_exact(
            bundle["run_root"],
            noncanonical,
            noncanonical_digest,
            predictor_verifiers=bundle["predictor_verifiers"],
        )
