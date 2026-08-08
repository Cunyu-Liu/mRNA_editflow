from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from scripts.r3x.artifact_io import atomic_write_json_exclusive
from scripts.r3x.baselines import fit_optional_hist_gradient_boosting
from scripts.r3x.hgb_fitted_state import (
    EXACT_EIGHT_COMPLETE,
    FIXTURE_NAMESPACE_STATUS,
    HGB_ASSIGNMENT_SCHEMA,
    HGB_BINNING_WEIGHT_POLICY,
    HGB_ENVIRONMENT_SCHEMA,
    HGB_FITTED_STATE_SCHEMA,
    HGB_LEAF_SUPPORT_POLICY,
    HGB_LOSS_SAMPLE_WEIGHT_POLICY,
    HGB_MISSING_FEATURE_POLICY,
    HGB_REPLICATION_INVARIANCE,
    HGB_SAMPLE_WEIGHT_NORMALIZATION,
    HGB_WEIGHT_SUM_ALGORITHM,
    READINESS_BLOCKER,
    TRAINING_READY,
    HGBFittedStateError,
    _normalize_loss_sample_weights,
    _recompute_hgb_predictions_untrusted_state_for_tests_only,
    build_hgb_cell_component_assignment,
    canonical_hgb_fitted_state_bytes,
    fit_hgb_fitted_state,
    load_and_recompute_hgb_predictions_exact,
    load_hgb_cell_component_assignment_exact,
    load_hgb_environment_qualification_exact,
    load_hgb_fitted_state_exact,
    observe_hgb_environment_qualification,
    recompute_hgb_training_weights,
    validate_hgb_cell_component_assignment,
    validate_hgb_environment_qualification,
    validate_hgb_fitted_state,
    write_hgb_cell_component_assignment_exclusive,
    write_hgb_environment_qualification_exclusive,
    write_hgb_fitted_state_exclusive,
)
from scripts.r3x.information_sets import InformationSet, information_set_fingerprint
from scripts.r3x.predictor_provenance import (
    DEVELOPMENT_PARTITION,
    NON_DATA_AUTHORITY_HASH_KEYS,
    PREDICTOR_INPUT_ROW_SCHEMA,
    TRAIN_PARTITION,
    normalize_predictor_input_rows,
)


INFORMATION_FINGERPRINT = information_set_fingerprint(
    InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR
)
TRAIN_DEVELOPMENT_SHA = "b" * 64
COMPONENT_PROJECTION_SHA = "c" * 64
CELL_SOURCE_SHA = "d" * 64
CELL_PROJECTION_SHA = "e" * 64
ESTIMAND_ID = "SYNTHETIC_ESTIMAND_A"


def _authorities() -> dict[str, str]:
    return {
        key: hashlib.sha256(f"authority/{key}".encode("utf-8")).hexdigest()
        for key in NON_DATA_AUTHORITY_HASH_KEYS
    }


def _cell_sha(cell: dict) -> str:
    ordered_tuple = [
        cell["study_id"],
        cell["estimand_id"],
        cell["publication_context_id"],
        cell["edit_count"],
    ]
    raw = json.dumps(
        ordered_tuple,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _typed_row(
    index: int,
    *,
    partition: str,
    target: float | None,
    namespace: str,
) -> dict:
    rotations = ("ACGU", "CGUA", "GUAC", "UACG")
    source = (rotations[index % 4] * 16)[:64]
    position = (index * 7 + index // 4) % len(source)
    alt = {"A": "C", "C": "G", "G": "U", "U": "A"}[source[position]]
    candidate = source[:position] + alt + source[position + 1 :]
    context_index = 0 if index < 90 else 1
    return {
        "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
        "row_id": hashlib.sha256(
            f"{namespace}/{partition}/{index}".encode("utf-8")
        ).hexdigest(),
        "partition": partition,
        "estimand_id": ESTIMAND_ID,
        "cell_id": f"TYPED_CELL_{context_index}",
        "predictor_input": {
            "endpoint": f"ENDPOINT_{index % 3}",
            "publication_context_id": f"CONTEXT_{context_index}",
            "region": "UTR5",
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


def _train_rows() -> list[dict]:
    return [
        _typed_row(
            index,
            partition=TRAIN_PARTITION,
            target=float(((index * 13) % 19) - 9) / 3.0,
            namespace="train",
        )
        for index in range(100)
    ]


def _development_rows() -> list[dict]:
    return [
        _typed_row(
            index,
            partition=DEVELOPMENT_PARTITION,
            target=None,
            namespace="development",
        )
        for index in range(82, 94)
    ]


def _assignment_rows(train_rows: list[dict]) -> list[dict]:
    result = []
    for row in reversed(normalize_predictor_input_rows(train_rows)):
        context = row["predictor_input"]["publication_context_id"]
        context_index = int(context.rsplit("_", 1)[1])
        cell = {
            "study_id": f"SYNTHETIC_STUDY_{context_index}",
            "estimand_id": row["estimand_id"],
            "publication_context_id": context,
            "edit_count": row["predictor_input"]["edit_count"],
        }
        # The large cell has nine components; the small cell has one.  This
        # lets the weighting test distinguish cell, component, and row mass.
        component = (
            f"COMPONENT_{int(row['row_id'][:8], 16) % 9}"
            if context_index == 0
            else "COMPONENT_SMALL"
        )
        result.append({
            "row_id": row["row_id"],
            "typed_cell_id": row["cell_id"],
            "source_component_id": component,
            "formal_cell": cell,
            "formal_cell_sha256": _cell_sha(cell),
        })
    return result


def _assignment_build_kwargs() -> dict:
    authorities = _authorities()
    return {
        "information_set_fingerprint": INFORMATION_FINGERPRINT,
        "train_development_outcomes_sha256": TRAIN_DEVELOPMENT_SHA,
        "non_data_authority_hashes": authorities,
        "source_component_assignment_source_sha256": authorities[
            "source_component_split_manifest_sha256"
        ],
        "source_component_assignment_projection_sha256": (
            COMPONENT_PROJECTION_SHA
        ),
        "formal_cell_assignment_source_sha256": CELL_SOURCE_SHA,
        "formal_cell_assignment_projection_sha256": CELL_PROJECTION_SHA,
    }


def _assignment_expected_kwargs(train_rows: list[dict]) -> dict:
    return {
        f"expected_{key}": value
        for key, value in {
            "train_rows": train_rows,
            **_assignment_build_kwargs(),
        }.items()
    }


def _fit_kwargs(bundle: dict) -> dict:
    return {
        "cell_component_assignment_path": bundle["assignment_path"],
        "cell_component_assignment_sha256": bundle["assignment_sha"],
        "environment_qualification_path": bundle["environment_path"],
        "environment_qualification_sha256": bundle["environment_sha"],
        **_assignment_build_kwargs(),
    }


def _state_expected_kwargs(bundle: dict) -> dict:
    values = {
        "train_rows": bundle["train_rows"],
        "cell_component_assignment_path": bundle["assignment_path"],
        "cell_component_assignment_sha256": bundle["assignment_sha"],
        "environment_qualification_path": bundle["environment_path"],
        "environment_qualification_sha256": bundle["environment_sha"],
        **_assignment_build_kwargs(),
    }
    return {f"expected_{key}": value for key, value in values.items()}


@pytest.fixture(scope="module")
def fitted_bundle(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("hgb_state") / "development"
    root.mkdir()
    train_rows = _train_rows()
    assignment = build_hgb_cell_component_assignment(
        train_rows, _assignment_rows(train_rows), **_assignment_build_kwargs()
    )
    assignment_path = root / "hgb-cell-component-assignment.json"
    assignment_sha = write_hgb_cell_component_assignment_exclusive(
        assignment_path,
        assignment,
        **_assignment_expected_kwargs(train_rows),
    )
    environment = observe_hgb_environment_qualification()
    environment_path = root / "hgb-environment.json"
    environment_sha = write_hgb_environment_qualification_exclusive(
        environment_path, environment
    )
    bundle = {
        "root": root,
        "train_rows": train_rows,
        "assignment": assignment,
        "assignment_path": assignment_path,
        "assignment_sha": assignment_sha,
        "environment": environment,
        "environment_path": environment_path,
        "environment_sha": environment_sha,
    }
    state = fit_hgb_fitted_state(train_rows, **_fit_kwargs(bundle))
    state_path = root / "hgb-fitted-state.json"
    state_sha = write_hgb_fitted_state_exclusive(
        state_path,
        state,
        **_state_expected_kwargs(bundle),
    )
    bundle.update({"state": state, "state_path": state_path, "state_sha": state_sha})
    return bundle


def _legacy_rows(
    rows: list[dict],
    *,
    component_by_id: dict[str, str] | None = None,
) -> list[dict]:
    result = []
    for row in normalize_predictor_input_rows(rows):
        inputs = row["predictor_input"]
        flat = {
            "record_id": row["row_id"],
            "endpoint": inputs["endpoint"],
            "publication_context_id": inputs["publication_context_id"],
            "region": inputs["region"],
            "normalized_source_sequence": inputs["normalized_source_sequence"],
            "normalized_candidate_sequence": inputs["normalized_candidate_sequence"],
            "edit_list": copy.deepcopy(inputs["edit_list"]),
            "edit_count": inputs["edit_count"],
        }
        if row["partition"] == TRAIN_PARTITION:
            flat["delta"] = row["training_target"]
            flat["source_component_id"] = component_by_id[row["row_id"]]
        result.append(flat)
    return result


def test_readiness_is_explicitly_false_and_environment_is_observed_only(
    fitted_bundle: dict,
) -> None:
    assert EXACT_EIGHT_COMPLETE is False
    assert TRAINING_READY is False
    assert fitted_bundle["state"]["exact_eight_complete"] is False
    assert fitted_bundle["state"]["training_ready"] is False
    assert fitted_bundle["state"]["readiness_blocker"] == READINESS_BLOCKER
    assert fitted_bundle["state"]["fixture_namespace_status"] == (
        FIXTURE_NAMESPACE_STATUS
    )
    assert fitted_bundle["environment"]["trust_root_binding_status"] == (
        "NOT_CONNECTED_TO_REAL_RUN_PLAN_TRUST_ROOT"
    )
    assert fitted_bundle["environment"]["schema_version"] == HGB_ENVIRONMENT_SCHEMA

    real_named_rows = copy.deepcopy(fitted_bundle["train_rows"])
    for row in real_named_rows:
        row["estimand_id"] = "REAL_ESTIMAND"
    with pytest.raises(HGBFittedStateError, match="positive fixture namespace"):
        fit_hgb_fitted_state(real_named_rows, **_fit_kwargs(fitted_bundle))

    # A positive namespace is deliberately not allowed to upgrade readiness.
    assert fitted_bundle["state"]["fit_config"]["fixture_namespace_status"] == (
        "POSITIVE_FIXTURE_NAMESPACE_ONLY_NOT_PROVENANCE_OR_TRUST_ROOT"
    )


def test_wrong_information_set_fingerprint_fails_build_fit_state_and_load(
    fitted_bundle: dict,
) -> None:
    wrong = "a" * 64
    assert wrong != INFORMATION_FINGERPRINT

    build_kwargs = _assignment_build_kwargs()
    build_kwargs["information_set_fingerprint"] = wrong
    with pytest.raises(HGBFittedStateError, match="canonical fingerprint"):
        build_hgb_cell_component_assignment(
            _train_rows(), _assignment_rows(_train_rows()), **build_kwargs
        )

    fit_kwargs = _fit_kwargs(fitted_bundle)
    fit_kwargs["information_set_fingerprint"] = wrong
    with pytest.raises(HGBFittedStateError, match="canonical fingerprint"):
        fit_hgb_fitted_state(fitted_bundle["train_rows"], **fit_kwargs)

    forged = copy.deepcopy(fitted_bundle["state"])
    forged["information_set_fingerprint"] = wrong
    with pytest.raises(HGBFittedStateError, match="canonical fingerprint"):
        validate_hgb_fitted_state(forged)

    expected = _state_expected_kwargs(fitted_bundle)
    expected["expected_information_set_fingerprint"] = wrong
    with pytest.raises(HGBFittedStateError, match="canonical fingerprint"):
        load_hgb_fitted_state_exact(
            fitted_bundle["state_path"], fitted_bundle["state_sha"], **expected
        )


def test_formal_cell_then_component_then_row_weights_give_each_cell_half_mass(
    fitted_bundle: dict,
) -> None:
    rows = normalize_predictor_input_rows(fitted_bundle["train_rows"])
    weights = recompute_hgb_training_weights(rows, fitted_bundle["assignment"])
    by_id = {
        item["row_id"]: item for item in fitted_bundle["assignment"]["assignments"]
    }
    mass = {}
    component_mass = {}
    for row, weight in zip(rows, weights, strict=True):
        assignment = by_id[row["row_id"]]
        cell = assignment["formal_cell_sha256"]
        component = assignment["source_component_id"]
        mass[cell] = mass.get(cell, 0.0) + float(weight)
        component_mass[(cell, component)] = (
            component_mass.get((cell, component), 0.0) + float(weight)
        )
    assert len(mass) == 2
    assert np.allclose(sorted(mass.values()), [50.0, 50.0], rtol=0, atol=1e-12)
    for cell in mass:
        values = [value for (key, _), value in component_mass.items() if key == cell]
        assert np.allclose(values, np.full(len(values), 50.0 / len(values)))
    binding = fitted_bundle["state"]["training_weight_binding"]
    assert binding["training_weight_policy"] == HGB_LOSS_SAMPLE_WEIGHT_POLICY
    assert binding["sample_weight_normalization"] == (
        HGB_SAMPLE_WEIGHT_NORMALIZATION
    )
    assert binding["weight_sum_algorithm"] == HGB_WEIGHT_SUM_ALGORITHM
    assert binding["replication_invariance"] == HGB_REPLICATION_INVARIANCE
    assert binding["sample_weight_sum"] == math.fsum(float(x) for x in weights)

    config = fitted_bundle["state"]["fit_config"]
    assert config["binning_weight_policy"] == HGB_BINNING_WEIGHT_POLICY
    assert config["leaf_support_policy"] == HGB_LEAF_SUPPORT_POLICY
    assert config["missing_feature_policy"] == HGB_MISSING_FEATURE_POLICY
    assert config["replication_invariance"] == "NOT_ASSERTED"


def test_large_loss_weight_normalization_uses_stable_fsum_scale() -> None:
    n_rows = 500_000
    raw = np.resize(
        np.asarray([1.0 / 7.0, 1.0 / 113.0, 1.0 / 791.0]),
        n_rows,
    )
    normalized = _normalize_loss_sample_weights(raw, n_rows=n_rows)
    assert normalized.shape == (n_rows,)
    assert np.all(np.isfinite(normalized))
    assert np.all(normalized > 0.0)
    assert math.fsum(float(value) for value in normalized) == pytest.approx(
        float(n_rows), rel=0.0, abs=1e-9
    )


def test_json_tree_replay_matches_public_sklearn_baseline_on_development(
    fitted_bundle: dict,
) -> None:
    development_rows = _development_rows()
    replay = load_and_recompute_hgb_predictions_exact(
        fitted_bundle["state_path"],
        fitted_bundle["state_sha"],
        development_rows,
        **_state_expected_kwargs(fitted_bundle),
    )
    component_by_id = {
        item["row_id"]: item["source_component_id"]
        for item in fitted_bundle["assignment"]["assignments"]
    }
    train = _legacy_rows(fitted_bundle["train_rows"], component_by_id=component_by_id)
    development = _legacy_rows(development_rows)
    cell_ids = [
        item["formal_cell_sha256"]
        for item in fitted_bundle["assignment"]["assignments"]
    ]
    with threadpool_limits(limits=1, user_api="openmp"):
        baseline = fit_optional_hist_gradient_boosting(
            train,
            development,
            execution_authorized=True,
            minimum_train_rows=100,
            training_cell_ids=cell_ids,
        )
    assert baseline.status == "RUN"
    assert np.array_equal(replay, baseline.predictions)


def test_exact_loader_refits_and_returns_canonical_state(fitted_bundle: dict) -> None:
    loaded = load_hgb_fitted_state_exact(
        fitted_bundle["state_path"],
        fitted_bundle["state_sha"],
        **_state_expected_kwargs(fitted_bundle),
    )
    assert loaded == fitted_bundle["state"]
    assert canonical_hgb_fitted_state_bytes(loaded) == fitted_bundle[
        "state_path"
    ].read_bytes()


def test_assignment_exact_loader_binds_formal_projection(fitted_bundle: dict) -> None:
    loaded = load_hgb_cell_component_assignment_exact(
        fitted_bundle["assignment_path"],
        fitted_bundle["assignment_sha"],
        **_assignment_expected_kwargs(fitted_bundle["train_rows"]),
    )
    assert loaded == fitted_bundle["assignment"]
    assert loaded["n_typed_cells"] == 2
    assert loaded["n_formal_cells"] == 2
    assert loaded["typed_cell_to_formal_cell_mapping_sha256"] == (
        fitted_bundle["state"]["cell_component_assignment_binding"][
            "typed_cell_to_formal_cell_mapping_sha256"
        ]
    )


def test_assignment_rejects_one_typed_cell_split_into_multiple_formal_cells() -> None:
    rows = _train_rows()
    assignments = _assignment_rows(rows)
    target_typed_cell = assignments[0]["typed_cell_id"]
    item = next(
        value for value in assignments
        if value["typed_cell_id"] == target_typed_cell
    )
    item["formal_cell"]["study_id"] = "SYNTHETIC_STUDY_SPLIT"
    item["formal_cell_sha256"] = _cell_sha(item["formal_cell"])
    with pytest.raises(
        HGBFittedStateError,
        match="one typed_cell_id maps to multiple formal cells",
    ):
        build_hgb_cell_component_assignment(
            rows, assignments, **_assignment_build_kwargs()
        )


def test_assignment_rejects_multiple_typed_cells_merged_into_one_formal_cell() -> None:
    rows = _train_rows()
    normalized = normalize_predictor_input_rows(rows)
    context_zero = [
        row for row in normalized
        if row["predictor_input"]["publication_context_id"] == "CONTEXT_0"
    ]
    context_zero[0]["cell_id"] = "TYPED_CELL_ALIAS"
    assignments = _assignment_rows(context_zero + [
        row for row in normalized
        if row["predictor_input"]["publication_context_id"] != "CONTEXT_0"
    ])
    with pytest.raises(
        HGBFittedStateError,
        match="one formal cell maps to multiple typed_cell_id values",
    ):
        build_hgb_cell_component_assignment(
            normalized, assignments, **_assignment_build_kwargs()
        )


def test_tampered_leaf_cannot_enter_exact_evidence_prediction(
    fitted_bundle: dict,
) -> None:
    tampered = copy.deepcopy(fitted_bundle["state"])
    leaf = next(
        node
        for tree in tampered["model_state"]["trees"]
        for node in tree["nodes"]
        if node["is_leaf"]
    )
    leaf["value"] += 1.0
    development_rows = _development_rows()
    trusted = load_and_recompute_hgb_predictions_exact(
        fitted_bundle["state_path"],
        fitted_bundle["state_sha"],
        development_rows,
        **_state_expected_kwargs(fitted_bundle),
    )
    untrusted = _recompute_hgb_predictions_untrusted_state_for_tests_only(
        tampered, development_rows
    )
    assert not np.array_equal(trusted, untrusted)

    path = fitted_bundle["root"] / "tampered-node-state.json"
    digest = atomic_write_json_exclusive(
        path, tampered, validator=validate_hgb_fitted_state
    )
    with pytest.raises(HGBFittedStateError, match="differs from exact training-row refit"):
        load_and_recompute_hgb_predictions_exact(
            path,
            digest,
            development_rows,
            **_state_expected_kwargs(fitted_bundle),
        )


def test_changed_training_row_content_is_rejected(fitted_bundle: dict) -> None:
    changed = copy.deepcopy(fitted_bundle["train_rows"])
    changed[0]["training_target"] += 1.0
    expected = _state_expected_kwargs(fitted_bundle)
    expected["expected_train_rows"] = changed
    with pytest.raises(HGBFittedStateError, match="train_row_binding binding mismatch"):
        load_hgb_fitted_state_exact(
            fitted_bundle["state_path"], fitted_bundle["state_sha"], **expected
        )


def test_tampered_formal_cell_mapping_cannot_replace_bound_assignment(
    fitted_bundle: dict,
) -> None:
    assignment_rows = copy.deepcopy(fitted_bundle["assignment"]["assignments"])
    target_typed_cell = assignment_rows[0]["typed_cell_id"]
    for item in assignment_rows:
        if item["typed_cell_id"] == target_typed_cell:
            item["formal_cell"]["study_id"] = "SYNTHETIC_STUDY_TAMPERED"
            item["formal_cell_sha256"] = _cell_sha(item["formal_cell"])
    tampered = build_hgb_cell_component_assignment(
        fitted_bundle["train_rows"],
        assignment_rows,
        **_assignment_build_kwargs(),
    )
    path = fitted_bundle["root"] / "tampered-cell-assignment.json"
    digest = atomic_write_json_exclusive(
        path, tampered, validator=validate_hgb_cell_component_assignment
    )
    expected = _state_expected_kwargs(fitted_bundle)
    expected["expected_cell_component_assignment_path"] = path
    expected["expected_cell_component_assignment_sha256"] = digest
    with pytest.raises(HGBFittedStateError, match="assignment SHA mismatch"):
        load_hgb_fitted_state_exact(
            fitted_bundle["state_path"], fitted_bundle["state_sha"], **expected
        )


def test_tampered_sklearn_version_is_rejected_against_current_runtime(
    fitted_bundle: dict,
) -> None:
    tampered = copy.deepcopy(fitted_bundle["environment"])
    tampered["sklearn_version"] = "999.0.synthetic"
    path = fitted_bundle["root"] / "tampered-environment.json"
    digest = atomic_write_json_exclusive(
        path, tampered, validator=validate_hgb_environment_qualification
    )
    with pytest.raises(HGBFittedStateError, match="does not match current runtime"):
        load_hgb_environment_qualification_exact(path, digest)


def test_weight_policy_categorical_nonfinite_old_schema_and_extra_keys_reject(
    fitted_bundle: dict,
) -> None:
    policy = copy.deepcopy(fitted_bundle["state"])
    policy["training_weight_binding"]["training_weight_policy"] = "ROW_UNIFORM"
    with pytest.raises(HGBFittedStateError, match="weight policy"):
        validate_hgb_fitted_state(policy)

    categorical = copy.deepcopy(fitted_bundle["state"])
    categorical["model_state"]["trees"][0]["nodes"][0]["is_categorical"] = True
    with pytest.raises(HGBFittedStateError, match="categorical"):
        validate_hgb_fitted_state(categorical)

    nonfinite = copy.deepcopy(fitted_bundle["state"])
    nonfinite["model_state"]["baseline_prediction"] = float("nan")
    with pytest.raises(HGBFittedStateError, match="finite"):
        validate_hgb_fitted_state(nonfinite)

    for schema in (
        "r3x_hgb_numeric_fitted_state_v0",
        "r3x_hgb_numeric_fitted_state_v1",
    ):
        old = copy.deepcopy(fitted_bundle["state"])
        old["schema_version"] = schema
        with pytest.raises(HGBFittedStateError, match="unsupported"):
            validate_hgb_fitted_state(old)

    old_assignment = copy.deepcopy(fitted_bundle["assignment"])
    old_assignment["schema_version"] = (
        "r3x_train_hgb_formal_cell_component_assignment_v1"
    )
    with pytest.raises(HGBFittedStateError, match="unsupported"):
        validate_hgb_cell_component_assignment(old_assignment)

    old_environment = copy.deepcopy(fitted_bundle["environment"])
    old_environment["schema_version"] = (
        "r3x_hgb_observed_environment_qualification_v1"
    )
    with pytest.raises(HGBFittedStateError, match="unsupported"):
        validate_hgb_environment_qualification(old_environment)

    extra = copy.deepcopy(fitted_bundle["state"])
    extra["caller_prediction"] = [1.0]
    with pytest.raises(HGBFittedStateError, match="strict schema"):
        validate_hgb_fitted_state(extra)


def test_assignment_rejects_bare_cell_id_instead_of_formal_tuple(
    fitted_bundle: dict,
) -> None:
    bad = copy.deepcopy(fitted_bundle["assignment"])
    bad["assignments"][0]["formal_cell"] = "CALLER_CELL_A"
    with pytest.raises(HGBFittedStateError, match="must be an object"):
        validate_hgb_cell_component_assignment(bad)


def test_exclusive_canonical_io_refuses_overwrite_and_protected_paths(
    fitted_bundle: dict,
    tmp_path: Path,
) -> None:
    with pytest.raises(HGBFittedStateError, match="already exists"):
        write_hgb_fitted_state_exclusive(
            fitted_bundle["state_path"],
            fitted_bundle["state"],
            **_state_expected_kwargs(fitted_bundle),
        )
    protected = tmp_path / "sealed" / "development" / "state.json"
    with pytest.raises(HGBFittedStateError, match="protected path"):
        write_hgb_fitted_state_exclusive(
            protected,
            fitted_bundle["state"],
            **_state_expected_kwargs(fitted_bundle),
        )


def test_no_pickle_joblib_or_caller_pass_surface() -> None:
    source = Path(
        "scripts/r3x/hgb_fitted_state.py"
    ).read_text(encoding="utf-8").lower()
    assert "import pickle" not in source
    assert "import joblib" not in source
    assert "caller_prediction" not in source
    assert "caller_pass" not in source
    assert "def recompute_hgb_predictions(" not in source
    assert "load_and_recompute_hgb_predictions_exact" in source
    assert HGB_FITTED_STATE_SCHEMA.endswith("_v2")
    assert HGB_ASSIGNMENT_SCHEMA.endswith("_v2")
    assert HGB_ENVIRONMENT_SCHEMA.endswith("_v2")
