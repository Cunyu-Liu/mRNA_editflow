from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

import scripts.r3x.scalar_fitted_state as scalar_module
from scripts.r3x.baselines import (
    candidate_kmer_ridge,
    edit_only_ridge,
    group_balanced_prior,
    nearest_neighbor,
    paired_kmer_ridge,
    zero_predictor,
)
from scripts.r3x.predictor_provenance import (
    DEVELOPMENT_PARTITION,
    NON_DATA_AUTHORITY_HASH_KEYS,
    PREDICTOR_INPUT_ROW_SCHEMA,
    TRAIN_PARTITION,
    normalize_predictor_input_rows,
)
from scripts.r3x.information_sets import information_set_fingerprint
from scripts.r3x.scalar_fitted_state import (
    EXACT_EIGHT_COMPLETE,
    FORMAL_WEIGHT_POLICY,
    HGB_BLOCKER,
    HGB_METHOD_ID,
    METHOD_INFORMATION_SETS,
    NN_SCALING_POLICY,
    NN_TIE_RULE,
    RIDGE_CV_RULE,
    RIDGE_CV_SCORING_POLICY,
    RIDGE_TRAINING_POLICY,
    SCALAR_FEATURE_SCHEMA,
    SCALAR_FIT_CONFIG_SCHEMA,
    SCALAR_FITTED_STATE_SCHEMA,
    SUPPORTED_SCALAR_METHOD_IDS,
    ScalarFittedStateError,
    _formal_cell_component_row_weighted_mse,
    _formal_cell_component_row_weights,
    build_train_source_component_assignment,
    canonical_scalar_fitted_state_bytes,
    canonical_train_source_component_assignment_bytes,
    fit_scalar_fitted_state,
    load_scalar_fitted_state_exact,
    load_train_source_component_assignment_exact,
    recompute_scalar_predictions,
    validate_scalar_fitted_state,
    validate_train_source_component_assignment,
    write_scalar_fitted_state_exclusive,
    write_train_source_component_assignment_exclusive,
)


DEFAULT_ASSIGNMENT_METHOD_ID = "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"
TRAIN_DEVELOPMENT_OUTCOMES_SHA = "b" * 64
SOURCE_SPLIT_SHA = "c" * 64
SOURCE_PROJECTION_SHA = "d" * 64
ESTIMAND_ID = "ESTIMAND_A"

METHOD_RUNNERS = {
    "ZERO_DELTA": zero_predictor,
    "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR": group_balanced_prior,
    "EDIT_ONLY_RIDGE": edit_only_ridge,
    "CANDIDATE_KMER_RIDGE": candidate_kmer_ridge,
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE": paired_kmer_ridge,
    "PAIRED_FEATURE_NEAREST_NEIGHBOR": nearest_neighbor,
}


def _method_fingerprint(method_id: str) -> str:
    return information_set_fingerprint(METHOD_INFORMATION_SETS[method_id])


def _authorities() -> dict[str, str]:
    values = {
        key: format(index + 1, "x") * 64
        for index, key in enumerate(NON_DATA_AUTHORITY_HASH_KEYS)
    }
    values["source_component_split_manifest_sha256"] = SOURCE_SPLIT_SHA
    return values


def _typed_row(
    index: int,
    *,
    partition: str,
    target: float | None,
    estimand_id: str = ESTIMAND_ID,
    row_namespace: str = "main",
) -> dict:
    source = "ACGUACGU"
    position = index % len(source)
    alt = {"A": "C", "C": "G", "G": "U", "U": "A"}[source[position]]
    candidate = source[:position] + alt + source[position + 1 :]
    return {
        "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
        "row_id": hashlib.sha256(
            f"{row_namespace}/{partition}/{index}".encode("utf-8")
        ).hexdigest(),
        "partition": partition,
        "estimand_id": estimand_id,
        "cell_id": f"CELL_{index % 2}",
        "predictor_input": {
            "endpoint": f"ENDPOINT_{index % 2}",
            "publication_context_id": f"CONTEXT_{index % 3}",
            "region": "5UTR",
            "normalized_source_sequence": source,
            "normalized_candidate_sequence": candidate,
            "edit_list": [{
                "op": "SUB",
                "pos": position,
                "ref": source[position],
                "alt": alt,
            }],
            "edit_count": 1,
        },
        "training_target": target,
    }


def _train_rows(n_rows: int = 8) -> list[dict]:
    targets = (-2.0, -1.0, 1.0, 2.0, 3.0, 4.0, -0.5, 0.75)
    return [
        _typed_row(
            index,
            partition=TRAIN_PARTITION,
            target=float(targets[index % len(targets)]),
        )
        for index in range(n_rows)
    ]


def _development_rows() -> list[dict]:
    return [
        _typed_row(
            index,
            partition=DEVELOPMENT_PARTITION,
            target=None,
            row_namespace="development",
        )
        for index in (11, 9, 10)
    ]


def _component_by_row_id(rows: list[dict]) -> dict[str, str]:
    canonical = normalize_predictor_input_rows(rows)
    return {
        row["row_id"]: f"COMPONENT_{index % 4}"
        for index, row in enumerate(canonical)
    }


def _assignment_rows(rows: list[dict]) -> list[dict]:
    components = _component_by_row_id(rows)
    return [
        {"row_id": row_id, "source_component_id": component}
        for row_id, component in reversed(list(components.items()))
    ]


def _publish_assignment(
    tmp_path: Path,
    train_rows: list[dict],
    assignment_rows: list[dict] | None = None,
    *,
    method_id: str = DEFAULT_ASSIGNMENT_METHOD_ID,
) -> tuple[Path, str, dict]:
    del method_id  # grouping is intentionally independent of method information set
    development = tmp_path / "development"
    development.mkdir(parents=True, exist_ok=True)
    payload = build_train_source_component_assignment(
        train_rows,
        _assignment_rows(train_rows) if assignment_rows is None else assignment_rows,
        train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
        non_data_authority_hashes=_authorities(),
        source_component_split_manifest_source_sha256=SOURCE_SPLIT_SHA,
        source_component_split_manifest_projection_sha256=SOURCE_PROJECTION_SHA,
    )
    path = development / "train-component-assignment.json"
    digest = write_train_source_component_assignment_exclusive(
        path,
        payload,
        expected_train_rows=train_rows,
        expected_train_development_outcomes_sha256=(
            TRAIN_DEVELOPMENT_OUTCOMES_SHA
        ),
        expected_non_data_authority_hashes=_authorities(),
        expected_source_component_split_manifest_source_sha256=SOURCE_SPLIT_SHA,
        expected_source_component_split_manifest_projection_sha256=(
            SOURCE_PROJECTION_SHA
        ),
    )
    return path, digest, payload


def _fit_kwargs(
    method_id: str, assignment_path: Path, assignment_sha: str
) -> dict:
    values = {
        "information_set_fingerprint": _method_fingerprint(method_id),
        "train_development_outcomes_sha256": TRAIN_DEVELOPMENT_OUTCOMES_SHA,
        "non_data_authority_hashes": _authorities(),
    }
    if method_id != "ZERO_DELTA":
        values.update({
            "component_assignment_path": assignment_path,
            "component_assignment_sha256": assignment_sha,
            "source_component_split_manifest_source_sha256": SOURCE_SPLIT_SHA,
            "source_component_split_manifest_projection_sha256": (
                SOURCE_PROJECTION_SHA
            ),
        })
    return values


def _writer_kwargs(
    method_id: str,
    train_rows: list[dict],
    assignment_path: Path,
    assignment_sha: str,
) -> dict:
    values = {
        "expected_train_rows": train_rows,
        "expected_method_id": method_id,
        "expected_information_set_fingerprint": _method_fingerprint(method_id),
        "expected_train_development_outcomes_sha256": (
            TRAIN_DEVELOPMENT_OUTCOMES_SHA
        ),
        "expected_non_data_authority_hashes": _authorities(),
    }
    if method_id != "ZERO_DELTA":
        values.update({
            "component_assignment_path": assignment_path,
            "expected_component_assignment_sha256": assignment_sha,
            "expected_source_component_split_manifest_source_sha256": (
                SOURCE_SPLIT_SHA
            ),
            "expected_source_component_split_manifest_projection_sha256": (
                SOURCE_PROJECTION_SHA
            ),
        })
    return values


def _legacy_rows(
    rows: list[dict], component_by_row_id: dict[str, str], *, train: bool
) -> list[dict]:
    result = []
    for row in normalize_predictor_input_rows(rows):
        inputs = row["predictor_input"]
        value = {
            "record_id": row["row_id"],
            "normalized_source_sequence": inputs["normalized_source_sequence"],
            "normalized_candidate_sequence": inputs[
                "normalized_candidate_sequence"
            ],
            "edit_list": copy.deepcopy(inputs["edit_list"]),
            "edit_count": inputs["edit_count"],
            "source_component_id": component_by_row_id.get(
                row["row_id"], "DEVELOPMENT_COMPONENT"
            ),
            "endpoint": inputs["endpoint"],
            "publication_context_id": inputs["publication_context_id"],
            "region": inputs["region"],
        }
        if train:
            value["delta"] = float(row["training_target"])
        result.append(value)
    return result


def test_all_six_scalar_states_are_exactly_replayable_under_v2_policies(
    tmp_path: Path,
) -> None:
    train_rows = _train_rows()
    development_rows = _development_rows()
    assert set(METHOD_RUNNERS) == SUPPORTED_SCALAR_METHOD_IDS
    for method_id in METHOD_RUNNERS:
        assignment_path, assignment_sha, _ = _publish_assignment(
            tmp_path / method_id.lower(), train_rows, method_id=method_id
        )
        state = fit_scalar_fitted_state(
            method_id,
            list(reversed(train_rows)),
            **_fit_kwargs(method_id, assignment_path, assignment_sha),
        )
        replay = recompute_scalar_predictions(
            state, list(reversed(development_rows))
        )
        actual = np.asarray(
            [row["predicted_delta_mean"] for row in replay], dtype=np.float64
        )
        assert np.all(np.isfinite(actual)), method_id
        if method_id == "ZERO_DELTA":
            assert np.array_equal(actual, np.zeros(len(actual)))
            assert state["fit_config"]["training_weight_policy"] == (
                "NOT_USED_ZERO_DELTA"
            )
        else:
            assert state["fit_config"]["training_weight_policy"] == (
                FORMAL_WEIGHT_POLICY
            )
            assert state["fit_config"]["weighted_risk_normalization"] == (
                "SUM_TO_ONE"
            )
        assert [row["row_id"] for row in replay] == [
            row["row_id"] for row in normalize_predictor_input_rows(development_rows)
        ]
        assert all(row["predicted_delta_logvar"] is None for row in replay)


def test_row_order_is_canonical_and_does_not_change_estimand(tmp_path: Path) -> None:
    train_rows = _train_rows()
    assignment_path, assignment_sha, _ = _publish_assignment(tmp_path, train_rows)
    forward = fit_scalar_fitted_state(
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
        train_rows,
        **_fit_kwargs(
            "PAIRED_LOCAL_GLOBAL_KMER_RIDGE", assignment_path, assignment_sha
        ),
    )
    reverse = fit_scalar_fitted_state(
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
        list(reversed(train_rows)),
        **_fit_kwargs(
            "PAIRED_LOCAL_GLOBAL_KMER_RIDGE", assignment_path, assignment_sha
        ),
    )
    assert forward == reverse
    assert forward["fit_config"]["row_order_rule"] == "CANONICAL_ASCENDING_ROW_ID"

    mixed = copy.deepcopy(train_rows)
    mixed[0]["estimand_id"] = "ESTIMAND_B"
    with pytest.raises(ScalarFittedStateError, match="one training estimand"):
        fit_scalar_fitted_state(
            "ZERO_DELTA",
            mixed,
            **_fit_kwargs("ZERO_DELTA", assignment_path, assignment_sha),
        )


def test_assignment_is_exact_train_only_target_free_projection(tmp_path: Path) -> None:
    train_rows = _train_rows()
    path, digest, payload = _publish_assignment(tmp_path, train_rows)
    loaded = load_train_source_component_assignment_exact(
        path,
        digest,
        expected_train_rows=list(reversed(train_rows)),
        expected_train_development_outcomes_sha256=(
            TRAIN_DEVELOPMENT_OUTCOMES_SHA
        ),
        expected_non_data_authority_hashes=_authorities(),
        expected_source_component_split_manifest_source_sha256=SOURCE_SPLIT_SHA,
        expected_source_component_split_manifest_projection_sha256=(
            SOURCE_PROJECTION_SHA
        ),
    )
    assert loaded == payload
    assert path.read_bytes() == canonical_train_source_component_assignment_bytes(
        payload
    )
    encoded = json.dumps(payload, sort_keys=True)
    assert "information_set_fingerprint" not in payload
    assert payload["assignment_scope"] == (
        "INFORMATION_SET_INDEPENDENT_SHARED_GLOBAL_COMPONENT_GROUPING_V1"
    )
    assert {item["partition"] for item in payload["assignments"]} == {
        TRAIN_PARTITION
    }
    assert {
        item["row_id"]: item["cell_id"] for item in payload["assignments"]
    } == {
        row["row_id"]: row["cell_id"]
        for row in normalize_predictor_input_rows(train_rows)
    }
    assert "training_target" not in encoded
    assert "normalized_source_sequence" not in encoded
    assert "normalized_candidate_sequence" not in encoded
    assert payload["internal_test_access"] is False
    assert payload["sealed_access"] is False
    assert payload["downstream_unlock"] is False


def test_prior_and_grouped_methods_refuse_raw_or_missing_component_mapping(
    tmp_path: Path,
) -> None:
    train_rows = _train_rows()
    with pytest.raises(ScalarFittedStateError, match="component-assignment authority"):
        fit_scalar_fitted_state(
            "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
            train_rows,
            information_set_fingerprint=_method_fingerprint(
                "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR"
            ),
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
        )
    with pytest.raises(ScalarFittedStateError, match="assignments must be a sequence"):
        build_train_source_component_assignment(
            train_rows,
            _component_by_row_id(train_rows),
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
            source_component_split_manifest_source_sha256=SOURCE_SPLIT_SHA,
            source_component_split_manifest_projection_sha256=SOURCE_PROJECTION_SHA,
        )


def test_exact_state_writer_loader_refits_and_rejects_swaps(tmp_path: Path) -> None:
    train_rows = _train_rows()
    assignment_path, assignment_sha, _ = _publish_assignment(tmp_path, train_rows)
    method_id = "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"
    state = fit_scalar_fitted_state(
        method_id,
        train_rows,
        **_fit_kwargs(method_id, assignment_path, assignment_sha),
    )
    state_path = tmp_path / "development" / "paired-ridge-state.json"
    writer_kwargs = _writer_kwargs(
        method_id, train_rows, assignment_path, assignment_sha
    )
    digest = write_scalar_fitted_state_exclusive(
        state_path, state, **writer_kwargs
    )
    loaded = load_scalar_fitted_state_exact(
        state_path, digest, **writer_kwargs
    )
    assert loaded == state
    assert state_path.read_bytes() == canonical_scalar_fitted_state_bytes(state)
    with pytest.raises(ScalarFittedStateError, match="already exists"):
        write_scalar_fitted_state_exclusive(
            state_path, state, **writer_kwargs
        )

    fabricated = copy.deepcopy(state)
    fabricated["state"]["coef"][0] += 1.0
    with pytest.raises(ScalarFittedStateError, match="independent exact train-row refit"):
        write_scalar_fitted_state_exclusive(
            tmp_path / "development" / "fabricated-state.json",
            fabricated,
            **writer_kwargs,
        )

    swapped_rows = copy.deepcopy(train_rows)
    swapped_rows[0]["training_target"] += 7.0
    swapped_kwargs = _writer_kwargs(
        method_id, swapped_rows, assignment_path, assignment_sha
    )
    with pytest.raises(ScalarFittedStateError, match="binding mismatch"):
        load_scalar_fitted_state_exact(
            state_path, digest, **swapped_kwargs
        )
    with pytest.raises(ScalarFittedStateError, match="method binding mismatch"):
        load_scalar_fitted_state_exact(
            state_path,
            digest,
            **{
                **writer_kwargs,
                "expected_method_id": "EDIT_ONLY_RIDGE",
            },
        )


def test_nearest_neighbor_replay_is_bitwise_chunk_invariant(tmp_path: Path) -> None:
    train_rows = _train_rows()
    assignment_path, assignment_sha, _ = _publish_assignment(tmp_path, train_rows)
    state = fit_scalar_fitted_state(
        "PAIRED_FEATURE_NEAREST_NEIGHBOR",
        train_rows,
        **_fit_kwargs(
            "PAIRED_FEATURE_NEAREST_NEIGHBOR", assignment_path, assignment_sha
        ),
    )
    one = recompute_scalar_predictions(
        state, _development_rows(), nearest_neighbor_train_chunk_size=1
    )
    three = recompute_scalar_predictions(
        state, _development_rows(), nearest_neighbor_train_chunk_size=3
    )
    large = recompute_scalar_predictions(
        state, _development_rows(), nearest_neighbor_train_chunk_size=4096
    )
    assert one == three == large
    with pytest.raises(ScalarFittedStateError, match="positive integer"):
        recompute_scalar_predictions(
            state, _development_rows(), nearest_neighbor_train_chunk_size=0
        )


def test_large_n_ridge_state_contains_no_n_by_n_object(tmp_path: Path) -> None:
    n_rows = 600
    train_rows = _train_rows(n_rows)
    assignment_path, assignment_sha, _ = _publish_assignment(
        tmp_path, train_rows, method_id="EDIT_ONLY_RIDGE"
    )
    state = fit_scalar_fitted_state(
        "EDIT_ONLY_RIDGE",
        train_rows,
        **_fit_kwargs("EDIT_ONLY_RIDGE", assignment_path, assignment_sha),
    )
    ridge = state["state"]
    assert ridge["solver"] == "PRIMAL_SMALLER_FEATURE_SYSTEM"
    assert ridge["square_system_dimension"] == ridge["n_features"]
    assert ridge["square_system_dimension"] < n_rows
    assert len(ridge["coef"]) == ridge["n_features"]
    assert len(ridge["scaler_mean"]) == ridge["n_features"]
    assert len(ridge["scaler_scale"]) == ridge["n_features"]
    assert "kernel" not in ridge
    assert "dual" not in ridge
    assert not any(
        isinstance(value, list)
        and len(value) == n_rows
        and value
        and isinstance(value[0], list)
        for value in ridge.values()
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows[0].__setitem__("training_target", float("nan")),
        lambda rows: rows.append(copy.deepcopy(rows[0])),
        lambda rows: rows[0].__setitem__("partition", "ORDINARY_INTERNAL_TEST"),
        lambda rows: rows[0].__setitem__("sealed_target", 1.0),
        lambda rows: rows[0]["predictor_input"].__setitem__("endpoint_id", "alias"),
    ],
)
def test_train_rows_reject_nan_duplicates_protected_and_alias_fields(
    tmp_path: Path, mutator
) -> None:
    rows = copy.deepcopy(_train_rows())
    mutator(rows)
    with pytest.raises(ScalarFittedStateError):
        fit_scalar_fitted_state(
            "ZERO_DELTA",
            rows,
            information_set_fingerprint=_method_fingerprint("ZERO_DELTA"),
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
        )


def test_development_replay_accepts_no_target_or_protected_fields(
    tmp_path: Path,
) -> None:
    train_rows = _train_rows()
    state = fit_scalar_fitted_state(
        "ZERO_DELTA",
        train_rows,
        **{
            key: value
            for key, value in _fit_kwargs("ZERO_DELTA", Path(), "").items()
        },
    )
    development = _development_rows()
    replay = recompute_scalar_predictions(state, development)
    assert all(row["predicted_delta_mean"] == 0.0 for row in replay)

    with_target = copy.deepcopy(development)
    with_target[0]["training_target"] = 10.0
    with pytest.raises(ScalarFittedStateError, match="cannot contain a training target"):
        recompute_scalar_predictions(state, with_target)
    protected = copy.deepcopy(development)
    protected[0]["internal_test_target"] = 10.0
    with pytest.raises(ScalarFittedStateError, match="strict schema"):
        recompute_scalar_predictions(state, protected)


def test_unknown_extra_old_forged_and_noncanonical_states_fail_closed(
    tmp_path: Path,
) -> None:
    train_rows = _train_rows()
    state = fit_scalar_fitted_state(
        "ZERO_DELTA",
        train_rows,
        information_set_fingerprint=_method_fingerprint("ZERO_DELTA"),
        train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
        non_data_authority_hashes=_authorities(),
    )
    old = copy.deepcopy(state)
    old["schema_version"] = "r3x_scalar_fitted_state_v1"
    with pytest.raises(ScalarFittedStateError, match="unsupported"):
        validate_scalar_fitted_state(old)
    extra = copy.deepcopy(state)
    extra["self_attested_pass"] = True
    with pytest.raises(ScalarFittedStateError, match="strict schema"):
        validate_scalar_fitted_state(extra)
    nan_state = copy.deepcopy(state)
    nan_state["state"]["constant"] = float("nan")
    with pytest.raises(ScalarFittedStateError):
        validate_scalar_fitted_state(nan_state)

    development = tmp_path / "development"
    development.mkdir()
    duplicate_path = development / "duplicate.json"
    duplicate_path.write_bytes(b'{"schema_version":"a","schema_version":"b"}\n')
    duplicate_sha = hashlib.sha256(duplicate_path.read_bytes()).hexdigest()
    with pytest.raises(ScalarFittedStateError, match="duplicate JSON key"):
        load_scalar_fitted_state_exact(
            duplicate_path,
            duplicate_sha,
            **_writer_kwargs("ZERO_DELTA", train_rows, Path(), ""),
        )

    noncanonical_path = development / "noncanonical.json"
    noncanonical_path.write_text(
        json.dumps(state, sort_keys=False, separators=(",", ":")),
        encoding="utf-8",
    )
    noncanonical_sha = hashlib.sha256(noncanonical_path.read_bytes()).hexdigest()
    with pytest.raises(ScalarFittedStateError, match="not canonical"):
        load_scalar_fitted_state_exact(
            noncanonical_path,
            noncanonical_sha,
            **_writer_kwargs("ZERO_DELTA", train_rows, Path(), ""),
        )


def test_assignment_tampering_and_protected_paths_fail_closed(tmp_path: Path) -> None:
    train_rows = _train_rows()
    _, _, payload = _publish_assignment(tmp_path / "valid", train_rows)
    missing = copy.deepcopy(payload)
    missing["assignments"].pop()
    with pytest.raises(ScalarFittedStateError):
        validate_train_source_component_assignment(missing)
    extra = copy.deepcopy(payload)
    extra["assignments"][0]["target"] = 1.0
    with pytest.raises(ScalarFittedStateError, match="strict schema"):
        validate_train_source_component_assignment(extra)
    wrong_source = copy.deepcopy(payload)
    wrong_source["source_component_split_manifest_source_sha256"] = "e" * 64
    with pytest.raises(ScalarFittedStateError, match="non-data authorities"):
        validate_train_source_component_assignment(wrong_source)

    protected_dir = tmp_path / "development" / "sealed_inputs"
    protected_dir.mkdir(parents=True)
    with pytest.raises(ScalarFittedStateError, match="protected path"):
        write_train_source_component_assignment_exclusive(
            protected_dir / "assignment.json",
            payload,
            expected_train_rows=train_rows,
            expected_train_development_outcomes_sha256=(
                TRAIN_DEVELOPMENT_OUTCOMES_SHA
            ),
            expected_non_data_authority_hashes=_authorities(),
            expected_source_component_split_manifest_source_sha256=SOURCE_SPLIT_SHA,
            expected_source_component_split_manifest_projection_sha256=(
                SOURCE_PROJECTION_SHA
            ),
        )


def test_hgb_is_explicitly_outside_scalar_module_and_exact_eight_is_not_claimed() -> None:
    assert HGB_METHOD_ID not in SUPPORTED_SCALAR_METHOD_IDS
    assert EXACT_EIGHT_COMPLETE is False
    assert "separately trusted sklearn" in HGB_BLOCKER
    with pytest.raises(ScalarFittedStateError, match="separately trusted sklearn"):
        fit_scalar_fitted_state(
            HGB_METHOD_ID,
            _train_rows(),
            information_set_fingerprint=_method_fingerprint(
                DEFAULT_ASSIGNMENT_METHOD_ID
            ),
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
        )


def test_schema_and_boundary_fields_are_frozen(tmp_path: Path) -> None:
    state = fit_scalar_fitted_state(
        "ZERO_DELTA",
        _train_rows(),
        information_set_fingerprint=_method_fingerprint("ZERO_DELTA"),
        train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
        non_data_authority_hashes=_authorities(),
    )
    assert state["schema_version"] == SCALAR_FITTED_STATE_SCHEMA
    assert state["internal_test_access"] is False
    assert state["sealed_access"] is False
    assert state["downstream_unlock"] is False
    assert state["train_row_binding"]["n_rows"] == len(_train_rows())
    assert state["feature_schema"]["schema_version"] == SCALAR_FEATURE_SCHEMA
    assert state["fit_config"]["schema_version"] == SCALAR_FIT_CONFIG_SCHEMA
    assert state["schema_version"] == "r3x_scalar_fitted_state_v2"


def _row_dominance_fixture() -> tuple[list[dict], list[dict], list[dict]]:
    template = _typed_row(
        0, partition=TRAIN_PARTITION, target=0.0, row_namespace="dominance"
    )
    rows: list[dict] = []
    assignments: list[dict] = []
    cells: list[str] = []
    components: list[str] = []
    for index in range(102):
        row = copy.deepcopy(template)
        row["row_id"] = hashlib.sha256(
            f"dominance/train/{index}".encode("utf-8")
        ).hexdigest()
        if index < 100:
            row["cell_id"] = "CELL_A"
            row["training_target"] = 1.0
            component = "COMPONENT_A"
        else:
            row["cell_id"] = "CELL_B"
            row["training_target"] = -1.0
            component = f"COMPONENT_{'B' if index == 100 else 'C'}"
        rows.append(row)
        assignments.append({
            "row_id": row["row_id"],
            "source_component_id": component,
        })
        cells.append(row["cell_id"])
        components.append(component)
    development = copy.deepcopy(template)
    development["row_id"] = hashlib.sha256(
        b"dominance/development/0"
    ).hexdigest()
    development["partition"] = DEVELOPMENT_PARTITION
    development["training_target"] = None
    return rows, assignments, [development]


def test_formal_weights_remove_100_to_1_row_dominance_for_ridge_prior_and_nn(
    tmp_path: Path,
) -> None:
    rows, assignments, development = _row_dominance_fixture()
    authority_by_method = {
        method_id: _publish_assignment(
            tmp_path / method_id.lower(),
            rows,
            assignments,
            method_id=method_id,
        )[:2]
        for method_id in (
            "EDIT_ONLY_RIDGE",
            "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
            "PAIRED_FEATURE_NEAREST_NEIGHBOR",
        )
    }
    components_by_id = {
        item["row_id"]: item["source_component_id"] for item in assignments
    }
    canonical = normalize_predictor_input_rows(rows)
    cells = [row["cell_id"] for row in canonical]
    components = [components_by_id[row["row_id"]] for row in canonical]
    weights = _formal_cell_component_row_weights(cells, components)
    cell_a_mass = math.fsum(
        float(weight)
        for weight, cell in zip(weights, cells, strict=True)
        if cell == "CELL_A"
    )
    assert math.fsum(float(value) for value in weights) == pytest.approx(1.0)
    assert cell_a_mass == pytest.approx(0.5)
    assert float(np.mean([row["training_target"] for row in rows])) == (
        pytest.approx(0.9607843137254902)
    )

    ridge = fit_scalar_fitted_state(
        "EDIT_ONLY_RIDGE",
        rows,
        **_fit_kwargs(
            "EDIT_ONLY_RIDGE", *authority_by_method["EDIT_ONLY_RIDGE"]
        ),
    )
    assert ridge["state"]["y_mean"] == pytest.approx(0.0, abs=1e-15)
    assert np.asarray(ridge["state"]["coef"]) == pytest.approx(0.0, abs=1e-14)

    prior = fit_scalar_fitted_state(
        "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
        rows,
        **_fit_kwargs(
            "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
            *authority_by_method["SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR"],
        ),
    )
    assert prior["fit_config"]["prior_training_policy"] == (
        "CONDITION_LOCAL_AND_GLOBAL_FALLBACK_CELL_COMPONENT_ROW_MEAN_V2"
    )
    assert prior["state"]["fallback_value"] == pytest.approx(0.0, abs=1e-15)
    assert prior["state"]["condition_formal_values"][0]["formal_value"] == (
        pytest.approx(0.0, abs=1e-15)
    )

    nn = fit_scalar_fitted_state(
        "PAIRED_FEATURE_NEAREST_NEIGHBOR",
        rows,
        **_fit_kwargs(
            "PAIRED_FEATURE_NEAREST_NEIGHBOR",
            *authority_by_method["PAIRED_FEATURE_NEAREST_NEIGHBOR"],
        ),
    )
    for fitted in (ridge, prior, nn):
        prediction = recompute_scalar_predictions(fitted, development)
        assert prediction[0]["predicted_delta_mean"] == pytest.approx(
            0.0, abs=1e-12
        )


def test_formal_oof_metric_is_50_not_row_weighted_0_990099() -> None:
    observed = np.zeros(101, dtype=np.float64)
    prediction = np.zeros(101, dtype=np.float64)
    prediction[-1] = 10.0
    cells = ["CELL_A"] * 100 + ["CELL_B"]
    components = ["COMPONENT_A"] * 100 + ["COMPONENT_B"]
    row_weighted = float(np.mean((prediction - observed) ** 2))
    formal = _formal_cell_component_row_weighted_mse(
        observed, prediction, cells, components
    )
    assert row_weighted == pytest.approx(0.9900990099009901)
    assert formal == pytest.approx(50.0)


def test_six_global_components_use_five_folds_and_one_formal_oof_score(
    tmp_path: Path,
) -> None:
    rows = _train_rows(6)
    assignments = [
        {
            "row_id": row["row_id"],
            "source_component_id": f"GLOBAL_COMPONENT_{index}",
        }
        for index, row in enumerate(rows)
    ]
    assignment_path, assignment_sha, _ = _publish_assignment(
        tmp_path, rows, assignments, method_id="EDIT_ONLY_RIDGE"
    )
    state = fit_scalar_fitted_state(
        "EDIT_ONLY_RIDGE",
        rows,
        **_fit_kwargs("EDIT_ONLY_RIDGE", assignment_path, assignment_sha),
    )
    ridge = state["state"]
    assert ridge["cv_status"] == (
        "TRAIN_ONLY_GLOBAL_SOURCE_COMPONENT_GROUPED_FORMAL_OOF_CV"
    )
    assert ridge["cv_n_global_source_components"] == 6
    assert ridge["cv_n_folds"] == 5
    assert len(ridge["cv_formal_oof_weighted_mse"]) == 5
    assert state["fit_config"]["ridge_cv_rule"] == RIDGE_CV_RULE
    assert state["fit_config"]["ridge_cv_scoring_policy"] == (
        RIDGE_CV_SCORING_POLICY
    )


def test_replication_inside_one_cell_component_does_not_change_predictions(
    tmp_path: Path,
) -> None:
    base_rows = _train_rows(6)
    base_assignments = [
        {
            "row_id": row["row_id"],
            "source_component_id": f"GLOBAL_COMPONENT_{index}",
        }
        for index, row in enumerate(base_rows)
    ]
    replicated_rows = copy.deepcopy(base_rows)
    replicated_assignments = copy.deepcopy(base_assignments)
    for index in range(20):
        clone = copy.deepcopy(base_rows[0])
        clone["row_id"] = hashlib.sha256(
            f"replication/{index}".encode("utf-8")
        ).hexdigest()
        replicated_rows.append(clone)
        replicated_assignments.append({
            "row_id": clone["row_id"],
            "source_component_id": "GLOBAL_COMPONENT_0",
        })
    for method_id in (
        "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
        "EDIT_ONLY_RIDGE",
        "PAIRED_FEATURE_NEAREST_NEIGHBOR",
    ):
        base_path, base_sha, _ = _publish_assignment(
            tmp_path / method_id.lower() / "base",
            base_rows,
            base_assignments,
            method_id=method_id,
        )
        replicated_path, replicated_sha, _ = _publish_assignment(
            tmp_path / method_id.lower() / "replicated",
            replicated_rows,
            replicated_assignments,
            method_id=method_id,
        )
        base = fit_scalar_fitted_state(
            method_id,
            base_rows,
            **_fit_kwargs(method_id, base_path, base_sha),
        )
        replicated = fit_scalar_fitted_state(
            method_id,
            replicated_rows,
            **_fit_kwargs(method_id, replicated_path, replicated_sha),
        )
        base_prediction = recompute_scalar_predictions(
            base, _development_rows()
        )
        replicated_prediction = recompute_scalar_predictions(
            replicated, _development_rows()
        )
        assert [row["predicted_delta_mean"] for row in replicated_prediction] == (
            pytest.approx(
                [row["predicted_delta_mean"] for row in base_prediction],
                rel=1e-10,
                abs=1e-10,
            )
        )


def test_v2_state_binds_design_ids_and_exact_weighting_policies(
    tmp_path: Path,
) -> None:
    rows, assignments, _ = _row_dominance_fixture()
    ridge_assignment_path, ridge_assignment_sha, _ = _publish_assignment(
        tmp_path / "ridge",
        rows,
        assignments,
        method_id="EDIT_ONLY_RIDGE",
    )
    ridge = fit_scalar_fitted_state(
        "EDIT_ONLY_RIDGE",
        rows,
        **_fit_kwargs(
            "EDIT_ONLY_RIDGE", ridge_assignment_path, ridge_assignment_sha
        ),
    )
    design = ridge["training_design_binding"]
    assert design["training_weight_policy"] == FORMAL_WEIGHT_POLICY
    assert design["cell_ids"] == ["CELL_A", "CELL_B"]
    assert design["source_component_ids"] == [
        "COMPONENT_A",
        "COMPONENT_B",
        "COMPONENT_C",
    ]
    assert all(len(design[field]) == 64 for field in (
        "design_sha256",
        "formal_weights_sha256",
        "ordered_cell_ids_sha256",
        "ordered_source_component_ids_sha256",
    ))
    assert ridge["fit_config"]["ridge_training_policy"] == (
        RIDGE_TRAINING_POLICY
    )

    nn_assignment_path, nn_assignment_sha, _ = _publish_assignment(
        tmp_path / "nn",
        rows,
        assignments,
        method_id="PAIRED_FEATURE_NEAREST_NEIGHBOR",
    )
    nn = fit_scalar_fitted_state(
        "PAIRED_FEATURE_NEAREST_NEIGHBOR",
        rows,
        **_fit_kwargs(
            "PAIRED_FEATURE_NEAREST_NEIGHBOR",
            nn_assignment_path,
            nn_assignment_sha,
        ),
    )
    assert nn["fit_config"]["nearest_neighbor_scaling_policy"] == (
        NN_SCALING_POLICY
    )
    assert nn["fit_config"]["nearest_neighbor_tie_rule"] == NN_TIE_RULE
    assert sorted(set(nn["state"]["train_cell_ids"])) == [
        "CELL_A",
        "CELL_B",
    ]

    forged = copy.deepcopy(ridge)
    forged["training_design_binding"]["training_weight_policy"] = (
        "ROW_WEIGHTED"
    )
    with pytest.raises(ScalarFittedStateError, match="wrong formal weight policy"):
        validate_scalar_fitted_state(forged)


def test_each_scalar_method_uses_its_registered_canonical_fingerprint() -> None:
    fingerprints = {
        method_id: _method_fingerprint(method_id)
        for method_id in sorted(SUPPORTED_SCALAR_METHOD_IDS)
    }
    assert fingerprints["ZERO_DELTA"] == fingerprints[
        "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR"
    ]
    assert fingerprints["PAIRED_LOCAL_GLOBAL_KMER_RIDGE"] == fingerprints[
        "PAIRED_FEATURE_NEAREST_NEIGHBOR"
    ]
    assert len(set(fingerprints.values())) == 4
    assert all(len(value) == 64 for value in fingerprints.values())


@pytest.mark.parametrize("claimed", ["a" * 64, "not-a-fingerprint"])
@pytest.mark.parametrize("method_id", sorted(SUPPORTED_SCALAR_METHOD_IDS))
def test_fit_rejects_old_or_malformed_self_reported_fingerprint(
    method_id: str, claimed: str
) -> None:
    match = "canonical fingerprint" if len(claimed) == 64 else "64 lowercase hex"
    with pytest.raises(ScalarFittedStateError, match=match):
        fit_scalar_fitted_state(
            method_id,
            _train_rows(),
            information_set_fingerprint=claimed,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=_authorities(),
        )


def test_one_shared_assignment_serves_all_four_information_sets_while_state_fingerprint_is_method_bound(
    tmp_path: Path,
) -> None:
    rows = _train_rows()
    shared_path, shared_sha, shared = _publish_assignment(tmp_path, rows)
    methods = (
        "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
        "EDIT_ONLY_RIDGE",
        "CANDIDATE_KMER_RIDGE",
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
        "PAIRED_FEATURE_NEAREST_NEIGHBOR",
    )
    states = {
        method_id: fit_scalar_fitted_state(
            method_id,
            rows,
            **_fit_kwargs(method_id, shared_path, shared_sha),
        )
        for method_id in methods
    }
    assert len({
        state["source_component_assignment_binding"]["sha256"]
        for state in states.values()
    }) == 1
    assert {
        states[method_id]["information_set_fingerprint"]
        for method_id in methods
    } == {
        _method_fingerprint(method_id) for method_id in methods
    }

    state = states["EDIT_ONLY_RIDGE"]
    forged = copy.deepcopy(state)
    forged["information_set_fingerprint"] = _method_fingerprint(
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"
    )
    with pytest.raises(ScalarFittedStateError, match="canonical fingerprint"):
        validate_scalar_fitted_state(forged)

    tampered = copy.deepcopy(shared)
    tampered["assignments"][0]["cell_id"] = "FORGED_CELL"
    with pytest.raises(ScalarFittedStateError, match="row/cell/partition binding"):
        scalar_module._verify_assignment_expected_bindings(
            validate_train_source_component_assignment(tampered),
            expected_train_rows=rows,
            expected_train_development_outcomes_sha256=(
                TRAIN_DEVELOPMENT_OUTCOMES_SHA
            ),
            expected_non_data_authority_hashes=_authorities(),
            expected_source_component_split_manifest_source_sha256=(
                SOURCE_SPLIT_SHA
            ),
            expected_source_component_split_manifest_projection_sha256=(
                SOURCE_PROJECTION_SHA
            ),
        )


def test_one_shared_assignment_serves_multiple_estimand_specific_fits(
    tmp_path: Path,
) -> None:
    first = _train_rows()
    second = [
        _typed_row(
            index,
            partition=TRAIN_PARTITION,
            target=float(index - 3),
            estimand_id="ESTIMAND_B",
            row_namespace="second-estimand",
        )
        for index in range(8)
    ]
    all_rows = first + second
    assignment_path, assignment_sha, _ = _publish_assignment(
        tmp_path, all_rows
    )
    states = []
    for rows, estimand_id in ((first, ESTIMAND_ID), (second, "ESTIMAND_B")):
        state = fit_scalar_fitted_state(
            "EDIT_ONLY_RIDGE",
            rows,
            component_assignment_authority_rows=all_rows,
            **_fit_kwargs(
                "EDIT_ONLY_RIDGE", assignment_path, assignment_sha
            ),
        )
        assert state["estimand_id"] == estimand_id
        assert state["source_component_assignment_binding"]["sha256"] == (
            assignment_sha
        )
        states.append(state)
    assert states[0]["train_row_binding"] != states[1]["train_row_binding"]


def test_v1_method_bound_assignment_fails_closed(tmp_path: Path) -> None:
    rows = _train_rows()
    _, _, payload = _publish_assignment(tmp_path, rows)
    legacy = copy.deepcopy(payload)
    legacy["schema_version"] = "r3x_train_source_component_assignment_v1"
    legacy["information_set_fingerprint"] = _method_fingerprint(
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"
    )
    legacy.pop("assignment_scope")
    with pytest.raises(ScalarFittedStateError, match="v1.*rejected"):
        validate_train_source_component_assignment(legacy)


def test_exact_loader_rejects_wrong_expected_canonical_fingerprint(
    tmp_path: Path,
) -> None:
    rows = _train_rows()
    assignment_path, assignment_sha, _ = _publish_assignment(
        tmp_path, rows, method_id="EDIT_ONLY_RIDGE"
    )
    state = fit_scalar_fitted_state(
        "EDIT_ONLY_RIDGE",
        rows,
        **_fit_kwargs("EDIT_ONLY_RIDGE", assignment_path, assignment_sha),
    )
    state_path = tmp_path / "development" / "edit-state.json"
    kwargs = _writer_kwargs(
        "EDIT_ONLY_RIDGE", rows, assignment_path, assignment_sha
    )
    digest = write_scalar_fitted_state_exclusive(state_path, state, **kwargs)
    wrong = {
        **kwargs,
        "expected_information_set_fingerprint": _method_fingerprint(
            "CANDIDATE_KMER_RIDGE"
        ),
    }
    with pytest.raises(ScalarFittedStateError, match="canonical fingerprint"):
        load_scalar_fitted_state_exact(state_path, digest, **wrong)
