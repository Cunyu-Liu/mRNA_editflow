"""Adversarial tests for typed development-safe non-data authorities."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import scripts.r3x.typed_safe_authorities as typed_module  # noqa: E402
from scripts.r3x.artifact_io import canonical_json_bytes  # noqa: E402
from scripts.r3x.baseline_selection import SELECTION_SCHEMA  # noqa: E402
from scripts.r3x.protocol import (  # noqa: E402
    AUTHORIZED_ROUTE,
    AUTHORIZED_SEQUENCE,
    REQUIRED_NEGATIVE_CONTROLS,
    canonical_sha256,
)
from scripts.r3x.typed_safe_authorities import (  # noqa: E402
    ANALYSIS_CODE_TREE_MANIFEST,
    ENDPOINT_ESTIMAND_REGISTRY,
    METRIC_ZERO_POLICY_REGISTRY,
    METHOD_PROTOCOL,
    MODEL_OPTIMIZER_SEED_CONFIG,
    PAIRED_BASELINE_POLICY,
    ROUTE_AUTHORIZATION,
    SUPPORTED_AUTHORITIES,
    TYPED_DEVELOPMENT_SAFE,
    TYPED_DEVELOPMENT_SAFE_BUNDLE_NOT_TRAINING_READY,
    TypedSafeAuthorityError,
    canonical_typed_safe_authority_bytes,
    load_typed_safe_authority_exact,
    load_typed_safe_bundle_exact,
    validate_typed_safe_authority,
    validate_typed_safe_bundle,
    write_typed_safe_authority_exclusive,
)


METHOD_IDS = sorted(
    {
        "ZERO_DELTA",
        "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
        "EDIT_ONLY_RIDGE",
        "CANDIDATE_KMER_RIDGE",
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
        "PAIRED_FEATURE_NEAREST_NEIGHBOR",
        "HISTOGRAM_GRADIENT_BOOSTING",
        "FULL_PAIR_UNRESTRICTED_SIAMESE",
        "A_EXACT_POTENTIAL_DIFFERENCE",
        "A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL",
        "B_SPARSE_EDIT_SET_ADDITIVE",
    }
)
BASELINE_IDS = list(typed_module.FROZEN_BASELINE_METHOD_ORDER)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "ordinary_run"
    (root / "development" / "authorities").mkdir(parents=True)
    (root / "scripts" / "r3x").mkdir(parents=True)
    (root / "scripts" / "r3x" / "train_model.py").write_text(
        "MODEL_VERSION = 1\n",
        encoding="utf-8",
    )
    return root


def _common(authority_name: str) -> dict:
    return {
        "schema_version": typed_module.SCHEMAS[authority_name],
        "artifact_role": typed_module.ROLES[authority_name],
        "status": TYPED_DEVELOPMENT_SAFE,
        "authority_name": authority_name,
        "source_authority_sha256": _sha(f"source:{authority_name}"),
        "train_development_outcomes_sha256": _sha("train-development-outcomes"),
        "development_projection_hashes": {
            "inclusion_manifest_development_projection": _sha(
                "inclusion-development"
            ),
            "context_source_crosswalk_development_projection": _sha(
                "crosswalk-development"
            ),
            "source_component_split_manifest_development_projection": _sha(
                "split-development"
            ),
        },
        "sign_accuracy_gate": 0.60,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "training_ready": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }


def _route_payload() -> dict:
    return _common(ROUTE_AUTHORIZATION) | {
        "authorization_id": "R3X_A_THEN_B_ORDINARY_ONLY_V1",
        "selected_route": AUTHORIZED_ROUTE,
        "authorized_sequence": list(AUTHORIZED_SEQUENCE),
        "code_implementation_authorized": True,
        "ordinary_training_authorized_when_all_inputs_frozen": True,
        "c3_positioning": "PROPOSED_NOT_ACTIVE_DGM_DOOB_DERIVED_ADAPTATION",
        "hard_boundaries": {
            "ordinary_only": True,
            "sealed_member_access": False,
            "formal_sealed_evaluator": False,
            "sign_accuracy_gate": 0.60,
            "sign_gate_change_authorized": False,
            "post_hoc_metric_threshold_seed_or_endpoint_selection": False,
            "legacy_signed_ratio_enrichment": "PROHIBITED",
            "o0_g1_unlock": False,
            "authority_contract_overwrite": False,
            "active_contract_mutation": False,
        },
    }


def _estimand_payload() -> dict:
    return _common(ENDPOINT_ESTIMAND_REGISTRY) | {
        "entries": [
            {
                "study": "ordinary_study_a",
                "endpoint": "half_life",
                "estimand_id": "variant_minus_source_half_life",
                "raw_source_field": "source_half_life",
                "raw_candidate_field": "candidate_half_life",
                "raw_unit": "hours",
                "direction": "HIGHER_IS_MORE_STABLE",
                "transform": "IDENTITY_DIFFERENCE",
                "valid_domain": "FINITE",
                "role": "PRIMARY",
                "replicate_policy": "SINGLE_OBSERVATION_ONLY",
            },
            {
                "study": "ordinary_study_b",
                "endpoint": "translation_efficiency",
                "estimand_id": "variant_minus_source_translation_efficiency",
                "raw_source_field": "source_translation_efficiency",
                "raw_candidate_field": "candidate_translation_efficiency",
                "raw_unit": "relative_units",
                "direction": "HIGHER_IS_MORE_STABLE",
                "transform": "IDENTITY_DIFFERENCE",
                "valid_domain": "FINITE",
                "role": "PRIMARY",
                "replicate_policy": "SINGLE_OBSERVATION_ONLY",
            },
        ]
    }


def _metric_payload() -> dict:
    return _common(METRIC_ZERO_POLICY_REGISTRY) | {
        "zero_policy": "EXCLUDE_TRUE_ZERO",
        "primary_cell_definition": "STUDY_ESTIMAND_PUBLICATION_CONTEXT_EDIT_COUNT",
        "component_weighting": "EQUAL_COMPONENT_THEN_EQUAL_ROW_WITHIN_COMPONENT",
        "ranking_aggregation": "WITHIN_SOURCE_POOL_THEN_CELL_MACRO",
        "undefined_primary_cell_policy": "HOLD_PRIMARY_CELL_UNDEFINED",
        "seed_policy": "FIXED_FIVE_MODEL_PREDICTION_ENSEMBLE",
        "selection_tie_policy": "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK",
        "primary_sign_gate": 0.60,
    }


def _config_payload() -> dict:
    return _common(MODEL_OPTIMIZER_SEED_CONFIG) | {
        "seeds": [11, 23, 42, 67, 101],
        "seed_aggregation": "FIXED_PREDICTION_ENSEMBLE",
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
        "condition_unknown_dropout": 0.15,
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "loss": {
            "huber_weight": 1.0,
            "nll_weight": 1.0,
            "sign_weight": 1.0,
            "sign_temperature": 1.0,
        },
        "checkpoint_selection": {
            "rule": "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH",
            "selection_partition": "development",
        },
        "model": {
            "architecture_id": "PAIRED_POTENTIAL_DIFFERENCE_V1",
            "max_sequence_length": 4096,
            "embedding_dim": 256,
            "hidden_dim": 512,
            "num_layers": 4,
            "dropout": 0.10,
        },
        "frozen_environment_execution_qualification_sha256": _sha(
            "frozen-environment-execution-qualification"
        ),
        "histogram_gradient_boosting_execution_qualification_asserted": True,
        "execution_qualification_replay_status": (
            "CERTIFICATE_HASH_BOUND_NOT_REPLAYED"
        ),
        "foundation_baseline_status": (
            "NOT_RUN_EXPOSURE_LICENSE_QUALIFICATION_UNKNOWN"
        ),
    }


def _code_payload(
    root: Path,
    relative_path: str = "scripts/r3x/train_model.py",
) -> dict:
    snapshot = _trusted_code_snapshot(root, relative_path)
    entries = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(snapshot.items())
    ]
    return _common(ANALYSIS_CODE_TREE_MANIFEST) | {
        "code_root_relative_path": ".",
        "files": entries,
        "tree_sha256": canonical_sha256(entries),
        "trusted_code_snapshot_sha256": canonical_sha256(entries),
    }


def _trusted_code_snapshot(
    root: Path,
    relative_path: str = "scripts/r3x/train_model.py",
) -> dict[str, str]:
    raw = (root / relative_path).read_bytes()
    return {relative_path: hashlib.sha256(raw).hexdigest()}


def _method_payload() -> dict:
    return _common(METHOD_PROTOCOL) | {
        "selected_route": AUTHORIZED_ROUTE,
        "execution_order": list(AUTHORIZED_SEQUENCE),
        "seeds": [11, 23, 42, 67, 101],
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
        "method_ids": list(METHOD_IDS),
        "negative_controls": list(REQUIRED_NEGATIVE_CONTROLS),
        "training_policy": {
            "sampler": "STUDY_ENDPOINT_CONTEXT_AND_SOURCE_COMPONENT_BALANCED",
            "mean_loss": "HUBER",
            "uncertainty_loss": "GAUSSIAN_NLL",
            "sign_probability": "NORMAL_CDF_OF_MU_OVER_SIGMA",
            "sign_bce_variance_gradient": "DETACHED",
            "hard_sign_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
            "rank_loss": "DISABLED_UNLESS_SEPARATE_PUBLICATION_RANK_POLICY_QUALIFIED",
            "checkpoint_selection": "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH",
            "selection_partition": "development",
        },
        "compute_matching": {
            "same_training_rows": True,
            "same_seeds": True,
            "same_epoch_and_optimizer_budget": True,
            "a_control_parameter_difference_max_fraction": 0.05,
            "report_parameter_count": True,
            "report_total_forward_equivalents": True,
            "report_wall_time_and_peak_memory": True,
        },
        "stop_rules": [
            "ANY_PROTECTED_INPUT",
            "ANY_COMPONENT_PARTITION_OVERLAP",
            "INCOMPLETE_TYPED_AUTHORITY",
            "A_INVARIANT_OR_MATCHED_CONTROL_FAILURE",
            "B_CONTROL_OR_ADDITIVE_FAILURE",
            "PRIMARY_SIGN_GATE_BELOW_0_60",
            "POST_HOC_CHANGE_REQUIRED",
        ],
        "hard_boundaries": {
            "ordinary_only": True,
            "sealed_member_access": False,
            "formal_sealed_evaluator": False,
            "O0_G1_E0_unlock": False,
            "active_contract_mutation": False,
            "sign_gate_change_authorized": False,
        },
    }


def _paired_payload() -> dict:
    family = ["primary_candidate_vs_development_selected_baseline_sign"]
    return _common(PAIRED_BASELINE_POLICY) | {
        "policy_id": "R3X_PAIRED_STRONGEST_BASELINE_POLICY_V1",
        "baseline_suite_status": "PRETEST_POLICY_FROZEN",
        "primary_candidate_model_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "comparison_family_id": "R3X_PRIMARY_PAIRED_FAMILY_V1",
        "comparison_family_member_ids": family,
        "primary_comparison_member_id": (
            "primary_candidate_vs_development_selected_baseline_sign"
        ),
        "eligible_baseline_method_ids": list(BASELINE_IDS),
        "required_baseline_method_ids": list(BASELINE_IDS),
        "development_selector_required": True,
        "development_selector_schema": SELECTION_SCHEMA,
        "frozen_environment_execution_qualification_sha256": _sha(
            "frozen-environment-execution-qualification"
        ),
        "histogram_gradient_boosting_execution_qualification_asserted": True,
        "execution_qualification_replay_status": (
            "CERTIFICATE_HASH_BOUND_NOT_REPLAYED"
        ),
        "foundation_baseline_status": (
            "NOT_RUN_EXPOSURE_LICENSE_QUALIFICATION_UNKNOWN"
        ),
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


def _payload(authority_name: str, root: Path) -> dict:
    factories = {
        ROUTE_AUTHORIZATION: _route_payload,
        ENDPOINT_ESTIMAND_REGISTRY: _estimand_payload,
        METRIC_ZERO_POLICY_REGISTRY: _metric_payload,
        MODEL_OPTIMIZER_SEED_CONFIG: _config_payload,
        METHOD_PROTOCOL: _method_payload,
        PAIRED_BASELINE_POLICY: _paired_payload,
    }
    if authority_name == ANALYSIS_CODE_TREE_MANIFEST:
        return _code_payload(root)
    return factories[authority_name]()


def _bundle(root: Path) -> dict[str, dict]:
    return {name: _payload(name, root) for name in SUPPORTED_AUTHORITIES}


def _assert_hold(callable_, code: str) -> TypedSafeAuthorityError:
    with pytest.raises(TypedSafeAuthorityError) as caught:
        callable_()
    assert caught.value.code == code
    return caught.value


@pytest.mark.parametrize("authority_name", SUPPORTED_AUTHORITIES)
def test_all_seven_authorities_validate_write_and_exact_load(
    run_root: Path,
    authority_name: str,
) -> None:
    payload = _payload(authority_name, run_root)
    code_snapshot = (
        _trusted_code_snapshot(run_root)
        if authority_name == ANALYSIS_CODE_TREE_MANIFEST
        else None
    )
    validated = validate_typed_safe_authority(
        payload,
        trusted_code_snapshot=code_snapshot,
    )
    assert validated["status"] == TYPED_DEVELOPMENT_SAFE
    assert validated["sign_accuracy_gate"] == 0.60
    assert validated["training_ready"] is False
    assert validated["sealed_access"] is False
    assert validated["downstream_unlock"] is False

    path = run_root / "development" / "authorities" / f"{authority_name}.json"
    digest = write_typed_safe_authority_exclusive(
        run_root,
        path,
        payload,
        trusted_code_snapshot=code_snapshot,
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert path.read_bytes() == canonical_typed_safe_authority_bytes(
        payload,
        trusted_code_snapshot=code_snapshot,
    )
    assert path.read_bytes() == canonical_json_bytes(payload)
    assert load_typed_safe_authority_exact(
        run_root,
        path,
        digest,
        expected_authority_name=authority_name,
        trusted_code_snapshot=code_snapshot,
    ) == payload


def test_writer_is_exclusive_and_authority_name_is_bound(run_root: Path) -> None:
    payload = _route_payload()
    path = run_root / "development" / "authorities" / "route.json"
    digest = write_typed_safe_authority_exclusive(run_root, path, payload)

    _assert_hold(
        lambda: write_typed_safe_authority_exclusive(run_root, path, payload),
        "HOLD_TYPED_SAFE_WRITE",
    )
    _assert_hold(
        lambda: load_typed_safe_authority_exact(
            run_root,
            path,
            digest,
            expected_authority_name=METHOD_PROTOCOL,
        ),
        "HOLD_TYPED_SAFE_BINDING",
    )


@pytest.mark.parametrize("extra_key", ["payload", "rows", "values", "observed_results"])
def test_no_generic_or_row_like_top_level_payload(
    run_root: Path,
    extra_key: str,
) -> None:
    payload = _config_payload()
    payload[extra_key] = [{"opaque": "support"}]
    _assert_hold(
        lambda: validate_typed_safe_authority(payload),
        "HOLD_TYPED_SAFE_SCHEMA",
    )


def test_nested_result_support_and_source_evidence_are_not_schema_fields(
    run_root: Path,
) -> None:
    estimand = _estimand_payload()
    estimand["entries"][0]["source_evidence_sha256"] = _sha("hidden-support")
    _assert_hold(
        lambda: validate_typed_safe_authority(estimand),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    config = _config_payload()
    config["optimizer"]["observed_results"] = [0.7, 0.8]
    _assert_hold(
        lambda: validate_typed_safe_authority(config),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    paired = _paired_payload()
    paired["primary_sign_inference"]["bootstrap"]["observed_values"] = [0.61]
    _assert_hold(
        lambda: validate_typed_safe_authority(paired),
        "HOLD_TYPED_SAFE_SCHEMA",
    )


def test_data_bindings_are_exactly_train_development_and_three_projections(
    run_root: Path,
) -> None:
    payload = _metric_payload()
    payload["development_projection_hashes"]["internal_test_features"] = _sha(
        "not-allowed"
    )
    _assert_hold(
        lambda: validate_typed_safe_authority(payload),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    payload = _metric_payload()
    payload["development_projection_hashes"][
        "inclusion_manifest_development_projection"
    ] = payload["train_development_outcomes_sha256"]
    _assert_hold(
        lambda: validate_typed_safe_authority(payload),
        "HOLD_TYPED_SAFE_BINDING",
    )

    payload = _metric_payload()
    payload["train_development_outcomes_sha256"] = "INTERNAL_TEST_OUTCOMES"
    _assert_hold(
        lambda: validate_typed_safe_authority(payload),
        "HOLD_TYPED_SAFE_SCHEMA",
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("sign_accuracy_gate", 0.59, "HOLD_TYPED_SAFE_SCHEMA"),
        ("sign_gate_change_authorized", True, "HOLD_TYPED_SAFE_SCHEMA"),
        ("ordinary_only", False, "HOLD_TYPED_SAFE_SCHEMA"),
        ("training_ready", True, "HOLD_TYPED_SAFE_SCHEMA"),
        ("sealed_access", True, "HOLD_TYPED_SAFE_SCHEMA"),
        ("downstream_unlock", True, "HOLD_TYPED_SAFE_SCHEMA"),
    ],
)
def test_sign_sealed_and_downstream_boundaries_are_immutable(
    field: str,
    value: object,
    expected_code: str,
) -> None:
    payload = _route_payload()
    payload[field] = value
    _assert_hold(
        lambda: validate_typed_safe_authority(payload),
        expected_code,
    )


def test_bool_int_aliases_and_nonfinite_numbers_are_rejected() -> None:
    route = _route_payload()
    route["hard_boundaries"]["sealed_member_access"] = 0
    _assert_hold(
        lambda: validate_typed_safe_authority(route),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    method = _method_payload()
    method["compute_matching"]["same_training_rows"] = 1
    _assert_hold(
        lambda: validate_typed_safe_authority(method),
        "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
    )

    config = _config_payload()
    config["optimizer"]["learning_rate"] = float("nan")
    _assert_hold(
        lambda: validate_typed_safe_authority(config),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    config = _config_payload()
    config["model"]["num_layers"] = True
    _assert_hold(
        lambda: validate_typed_safe_authority(config),
        "HOLD_TYPED_SAFE_SCHEMA",
    )


def test_unsupported_estimand_and_incomplete_method_fail_closed() -> None:
    estimand = _estimand_payload()
    estimand["entries"][0]["transform"] = "ARBITRARY_PROGRAM"
    _assert_hold(
        lambda: validate_typed_safe_authority(estimand),
        "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
    )

    estimand = _estimand_payload()
    estimand["entries"] = list(reversed(estimand["entries"]))
    _assert_hold(
        lambda: validate_typed_safe_authority(estimand),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    method = _method_payload()
    method["method_ids"].remove("B_SPARSE_EDIT_SET_ADDITIVE")
    _assert_hold(
        lambda: validate_typed_safe_authority(method),
        "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
    )


def test_paired_policy_requires_frozen_pretest_family_not_result_claims() -> None:
    paired = _paired_payload()
    paired["baseline_suite_status"] = "PUBLICATION_QUALIFIED"
    _assert_hold(
        lambda: validate_typed_safe_authority(paired),
        "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
    )

    paired = _paired_payload()
    paired["required_baseline_method_ids"].remove("EDIT_ONLY_RIDGE")
    _assert_hold(
        lambda: validate_typed_safe_authority(paired),
        "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
    )

    paired = _paired_payload()
    paired["eligible_baseline_method_ids"][0:2] = reversed(
        paired["eligible_baseline_method_ids"][0:2]
    )
    paired["required_baseline_method_ids"][0:2] = reversed(
        paired["required_baseline_method_ids"][0:2]
    )
    _assert_hold(
        lambda: validate_typed_safe_authority(paired),
        "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
    )

    paired = _paired_payload()
    paired["histogram_gradient_boosting_execution_qualification_asserted"] = False
    _assert_hold(
        lambda: validate_typed_safe_authority(paired),
        "HOLD_BASELINE_SUITE_NOT_QUALIFIED",
    )

    paired = _paired_payload()
    paired["strongest_baseline_id"] = "FULL_PAIR_UNRESTRICTED_SIAMESE"
    _assert_hold(
        lambda: validate_typed_safe_authority(paired),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    paired = _paired_payload()
    paired["development_selector_required"] = False
    _assert_hold(
        lambda: validate_typed_safe_authority(paired),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    paired = _paired_payload()
    paired["primary_candidate_model_id"] = "B_SPARSE_EDIT_SET_ADDITIVE"
    _assert_hold(
        lambda: validate_typed_safe_authority(paired),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    paired = _paired_payload()
    paired["comparison_family_member_ids"].append("zzz_encoded_061_042_117")
    _assert_hold(
        lambda: validate_typed_safe_authority(paired),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    paired = _paired_payload()
    paired["primary_sign_inference"]["alpha"] = 0.20
    paired["primary_sign_inference"]["confidence"] = 0.80
    _assert_hold(
        lambda: validate_typed_safe_authority(paired),
        "HOLD_TYPED_SAFE_SCHEMA",
    )


def test_c3_is_only_a_not_active_proposal() -> None:
    route = _route_payload()
    route["c3_positioning"] = "DGM_DOOB_DERIVED_LEGAL_EDIT_ADAPTATION"
    _assert_hold(
        lambda: validate_typed_safe_authority(route),
        "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
    )


def test_environment_qualification_is_hash_bound_and_foundation_stays_not_run() -> None:
    config = _config_payload()
    config["frozen_environment_execution_qualification_sha256"] = config[
        "source_authority_sha256"
    ]
    _assert_hold(
        lambda: validate_typed_safe_authority(config),
        "HOLD_TYPED_SAFE_BINDING",
    )

    config = _config_payload()
    config["histogram_gradient_boosting_execution_qualification_asserted"] = False
    _assert_hold(
        lambda: validate_typed_safe_authority(config),
        "HOLD_BASELINE_SUITE_NOT_QUALIFIED",
    )

    config = _config_payload()
    config["foundation_baseline_status"] = "RUN"
    _assert_hold(
        lambda: validate_typed_safe_authority(config),
        "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "../escape.py",
        "/absolute/model.py",
        "data/model.py",
        "datasets/model.py",
        "outputs/model.py",
        "checkpoints/model.py",
        "results/model.py",
        "artifacts/model.py",
        "tests/model.py",
        "scripts/r3x/config.json",
        "scripts/../models/model.py",
    ],
)
def test_code_manifest_rejects_unsafe_paths_and_extensions(
    run_root: Path,
    relative_path: str,
) -> None:
    entry = {"path": relative_path, "sha256": _sha(relative_path)}
    payload = _common(ANALYSIS_CODE_TREE_MANIFEST) | {
        "code_root_relative_path": ".",
        "files": [entry],
        "tree_sha256": canonical_sha256([entry]),
        "trusted_code_snapshot_sha256": canonical_sha256([entry]),
    }
    with pytest.raises(TypedSafeAuthorityError) as caught:
        validate_typed_safe_authority(payload)
    assert caught.value.code in {
        "HOLD_TYPED_SAFE_PATH",
        "HOLD_TYPED_SAFE_PROHIBITED_CONTENT",
        "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
    }


def test_code_manifest_rejects_symlink_and_hardlink_files(run_root: Path) -> None:
    target = run_root / "scripts" / "r3x" / "train_model.py"
    symlink = run_root / "scripts" / "r3x" / "alias.py"
    symlink.symlink_to(target.name)
    payload = _code_payload(run_root, "scripts/r3x/alias.py")
    snapshot = _trusted_code_snapshot(run_root, "scripts/r3x/alias.py")
    path = run_root / "development" / "authorities" / "symlink-code.json"
    _assert_hold(
        lambda: write_typed_safe_authority_exclusive(
            run_root,
            path,
            payload,
            trusted_code_snapshot=snapshot,
        ),
        "HOLD_TYPED_SAFE_PATH",
    )

    symlink.unlink()
    hardlink = run_root / "scripts" / "r3x" / "hardlink.py"
    os.link(target, hardlink)
    payload = _code_payload(run_root, "scripts/r3x/hardlink.py")
    snapshot = _trusted_code_snapshot(run_root, "scripts/r3x/hardlink.py")
    path = run_root / "development" / "authorities" / "hardlink-code.json"
    _assert_hold(
        lambda: write_typed_safe_authority_exclusive(
            run_root,
            path,
            payload,
            trusted_code_snapshot=snapshot,
        ),
        "HOLD_TYPED_SAFE_PATH",
    )


def test_code_manifest_rejects_symlink_directory(run_root: Path) -> None:
    (run_root / "models").symlink_to(run_root / "scripts" / "r3x", target_is_directory=True)
    payload = _code_payload(run_root, "models/train_model.py")
    snapshot = _trusted_code_snapshot(run_root, "models/train_model.py")
    path = run_root / "development" / "authorities" / "symlink-dir-code.json"
    _assert_hold(
        lambda: write_typed_safe_authority_exclusive(
            run_root,
            path,
            payload,
            trusted_code_snapshot=snapshot,
        ),
        "HOLD_TYPED_SAFE_PATH",
    )


def test_code_bytes_are_replayed_on_write_and_load(run_root: Path) -> None:
    payload = _code_payload(run_root)
    snapshot = _trusted_code_snapshot(run_root)
    path = run_root / "development" / "authorities" / "code.json"
    digest = write_typed_safe_authority_exclusive(
        run_root,
        path,
        payload,
        trusted_code_snapshot=snapshot,
    )

    (run_root / "scripts" / "r3x" / "train_model.py").write_text(
        "MODEL_VERSION = 2\n",
        encoding="utf-8",
    )
    _assert_hold(
        lambda: load_typed_safe_authority_exact(
            run_root,
            path,
            digest,
            expected_authority_name=ANALYSIS_CODE_TREE_MANIFEST,
            trusted_code_snapshot=snapshot,
        ),
        "HOLD_TYPED_SAFE_HASH_MISMATCH",
    )


def test_code_manifest_rejects_prohibited_support_in_code_bytes(run_root: Path) -> None:
    code = run_root / "scripts" / "r3x" / "train_model.py"
    code.write_text('PARTITION = "internal_test"\n', encoding="utf-8")
    payload = _code_payload(run_root)
    snapshot = _trusted_code_snapshot(run_root)
    path = run_root / "development" / "authorities" / "code.json"
    _assert_hold(
        lambda: write_typed_safe_authority_exclusive(
            run_root,
            path,
            payload,
            trusted_code_snapshot=snapshot,
        ),
        "HOLD_TYPED_SAFE_PROHIBITED_CONTENT",
    )


def test_code_tree_digest_must_match_sorted_entries(run_root: Path) -> None:
    payload = _code_payload(run_root)
    payload["tree_sha256"] = _sha("stale-tree")
    _assert_hold(
        lambda: validate_typed_safe_authority(payload),
        "HOLD_TYPED_SAFE_HASH_MISMATCH",
    )


def test_code_manifest_cannot_self_certify_without_external_snapshot(
    run_root: Path,
) -> None:
    payload = _code_payload(run_root)
    _assert_hold(
        lambda: validate_typed_safe_authority(payload),
        "HOLD_EXTERNAL_CODE_SNAPSHOT_REVIEW_REQUIRED",
    )

    labels = run_root / "scripts" / "r3x" / "ordinary_labels.py"
    labels.write_text("LABELS = [0.4, -0.7, 1.2]\n", encoding="utf-8")
    payload = _code_payload(run_root, "scripts/r3x/ordinary_labels.py")
    path = run_root / "development" / "authorities" / "labels-code.json"
    _assert_hold(
        lambda: write_typed_safe_authority_exclusive(run_root, path, payload),
        "HOLD_EXTERNAL_CODE_SNAPSHOT_REVIEW_REQUIRED",
    )


def test_code_manifest_must_equal_external_snapshot(run_root: Path) -> None:
    payload = _code_payload(run_root)
    snapshot = _trusted_code_snapshot(run_root)
    snapshot["scripts/r3x/train_model.py"] = _sha("different-approved-bytes")
    _assert_hold(
        lambda: validate_typed_safe_authority(
            payload,
            trusted_code_snapshot=snapshot,
        ),
        "HOLD_EXTERNAL_CODE_SNAPSHOT_REVIEW_REQUIRED",
    )


def test_code_replay_pins_each_directory_descriptor_against_replacement(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = Path("scripts/r3x/train_model.py")
    expected = (run_root / relative).read_bytes()
    expected_sha = hashlib.sha256(expected).hexdigest()
    original_directory = run_root / "scripts" / "r3x"
    pinned_directory = run_root / "scripts" / "r3x-pinned"
    outside_directory = run_root.parent / "outside-code"
    outside_directory.mkdir()
    (outside_directory / "train_model.py").write_text(
        "MODEL_VERSION = 999\n",
        encoding="utf-8",
    )

    real_open = typed_module.os.open
    replaced = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "r3x" and dir_fd is not None and not replaced:
            replaced = True
            original_directory.rename(pinned_directory)
            original_directory.symlink_to(outside_directory, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(typed_module.os, "open", racing_open)
    replayed = typed_module._read_code_exact(run_root, relative, expected_sha)
    assert replaced is True
    assert replayed == expected
    assert replayed != (outside_directory / "train_model.py").read_bytes()


def test_bundle_without_external_code_snapshot_remains_hold(run_root: Path) -> None:
    _assert_hold(
        lambda: validate_typed_safe_bundle(_bundle(run_root)),
        "HOLD_EXTERNAL_CODE_SNAPSHOT_REVIEW_REQUIRED",
    )


def test_artifacts_must_stay_under_real_development_path(run_root: Path) -> None:
    payload = _route_payload()
    outside = run_root / "authorities"
    outside.mkdir()
    _assert_hold(
        lambda: write_typed_safe_authority_exclusive(
            run_root,
            outside / "route.json",
            payload,
        ),
        "HOLD_TYPED_SAFE_PATH",
    )

    real = run_root / "development" / "real" / "nested"
    real.mkdir(parents=True)
    alias = run_root / "development" / "alias"
    alias.symlink_to(real.parent, target_is_directory=True)
    _assert_hold(
        lambda: write_typed_safe_authority_exclusive(
            run_root,
            alias / "nested" / "route.json",
            payload,
        ),
        "HOLD_TYPED_SAFE_PATH",
    )


def test_loader_rejects_symlink_hardlink_and_noncanonical_json(run_root: Path) -> None:
    payload = _route_payload()
    path = run_root / "development" / "authorities" / "route.json"
    digest = write_typed_safe_authority_exclusive(run_root, path, payload)

    symlink = path.with_name("route-symlink.json")
    symlink.symlink_to(path.name)
    _assert_hold(
        lambda: load_typed_safe_authority_exact(run_root, symlink, digest),
        "HOLD_TYPED_SAFE_PATH",
    )

    hardlink = path.with_name("route-hardlink.json")
    os.link(path, hardlink)
    _assert_hold(
        lambda: load_typed_safe_authority_exact(run_root, path, digest),
        "HOLD_TYPED_SAFE_PATH",
    )

    hardlink.unlink()
    noncanonical = path.with_name("route-noncanonical.json")
    raw = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    noncanonical.write_bytes(raw)
    noncanonical_sha = hashlib.sha256(raw).hexdigest()
    _assert_hold(
        lambda: load_typed_safe_authority_exact(
            run_root,
            noncanonical,
            noncanonical_sha,
        ),
        "HOLD_TYPED_SAFE_JSON",
    )


def test_loader_rejects_duplicate_json_keys(run_root: Path) -> None:
    path = run_root / "development" / "authorities" / "duplicate.json"
    raw = b'{"authority_name":"route_authorization","authority_name":"method_protocol"}'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    _assert_hold(
        lambda: load_typed_safe_authority_exact(run_root, path, digest),
        "HOLD_TYPED_SAFE_JSON",
    )


def test_authority_loader_holds_if_parent_namespace_is_replaced(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _route_payload()
    path = run_root / "development" / "authorities" / "route.json"
    digest = write_typed_safe_authority_exclusive(run_root, path, payload)
    original = path.parent
    pinned = original.with_name("authorities-pinned")
    outside = run_root.parent / "outside-authorities-load"
    outside.mkdir()
    (outside / path.name).write_text('{"attacker":true}\n', encoding="utf-8")

    real_open = typed_module.os.open
    replaced = False

    def racing_open(name, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        if dir_fd is None:
            descriptor = real_open(name, flags, mode)
        else:
            descriptor = real_open(name, flags, mode, dir_fd=dir_fd)
        if name == "authorities" and dir_fd is not None and not replaced:
            replaced = True
            original.rename(pinned)
            original.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(typed_module.os, "open", racing_open)
    _assert_hold(
        lambda: load_typed_safe_authority_exact(run_root, path, digest),
        "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
    )
    assert replaced is True


def test_authority_writer_holds_if_parent_namespace_is_replaced(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _route_payload()
    path = run_root / "development" / "authorities" / "route-race.json"
    original = path.parent
    pinned = original.with_name("authorities-pinned")
    outside = run_root.parent / "outside-authorities-write"
    outside.mkdir()

    real_open = typed_module.os.open
    replaced = False

    def racing_open(name, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        if dir_fd is None:
            descriptor = real_open(name, flags, mode)
        else:
            descriptor = real_open(name, flags, mode, dir_fd=dir_fd)
        if name == "authorities" and dir_fd is not None and not replaced:
            replaced = True
            original.rename(pinned)
            original.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(typed_module.os, "open", racing_open)
    _assert_hold(
        lambda: write_typed_safe_authority_exclusive(run_root, path, payload),
        "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
    )
    assert replaced is True
    assert not (outside / path.name).exists()
    assert (pinned / path.name).is_file()


def test_unknown_or_overcomplex_authority_is_hold_unsupported() -> None:
    _assert_hold(
        lambda: validate_typed_safe_authority(
            {
                "authority_name": "arbitrary_full_authority",
                "payload": {"rows": [], "values": []},
            }
        ),
        "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
    )


def test_bundle_cross_binds_all_seven_without_claiming_training_ready(
    run_root: Path,
) -> None:
    result = validate_typed_safe_bundle(
        _bundle(run_root),
        trusted_code_snapshot=_trusted_code_snapshot(run_root),
    )
    assert result["status"] == TYPED_DEVELOPMENT_SAFE_BUNDLE_NOT_TRAINING_READY
    assert result["training_ready"] is False
    assert (
        result["execution_qualification_replay_status"]
        == "CERTIFICATE_HASH_BOUND_NOT_REPLAYED"
    )
    assert result["remaining_required_typed_row_authorities"] == [
        "inclusion_manifest_development_projection",
        "context_source_crosswalk_development_projection",
        "source_component_split_manifest_development_projection",
    ]
    assert set(result["authorities"]) == set(SUPPORTED_AUTHORITIES)


def test_bundle_rejects_missing_and_mixed_run_authorities(run_root: Path) -> None:
    bundle = _bundle(run_root)
    bundle.pop(METHOD_PROTOCOL)
    _assert_hold(
        lambda: validate_typed_safe_bundle(
            bundle,
            trusted_code_snapshot=_trusted_code_snapshot(run_root),
        ),
        "HOLD_TYPED_SAFE_SCHEMA",
    )

    bundle = _bundle(run_root)
    bundle[METHOD_PROTOCOL]["train_development_outcomes_sha256"] = _sha(
        "another-run-train-development"
    )
    _assert_hold(
        lambda: validate_typed_safe_bundle(
            bundle,
            trusted_code_snapshot=_trusted_code_snapshot(run_root),
        ),
        "HOLD_TYPED_SAFE_BUNDLE_BINDING",
    )

    bundle = _bundle(run_root)
    bundle[METHOD_PROTOCOL]["development_projection_hashes"][
        "inclusion_manifest_development_projection"
    ] = _sha("another-run-inclusion")
    _assert_hold(
        lambda: validate_typed_safe_bundle(
            bundle,
            trusted_code_snapshot=_trusted_code_snapshot(run_root),
        ),
        "HOLD_TYPED_SAFE_BUNDLE_BINDING",
    )


def test_bundle_rejects_source_alias_and_execution_qualification_drift(
    run_root: Path,
) -> None:
    bundle = _bundle(run_root)
    bundle[METHOD_PROTOCOL]["source_authority_sha256"] = bundle[
        ROUTE_AUTHORIZATION
    ]["source_authority_sha256"]
    _assert_hold(
        lambda: validate_typed_safe_bundle(
            bundle,
            trusted_code_snapshot=_trusted_code_snapshot(run_root),
        ),
        "HOLD_TYPED_SAFE_BUNDLE_BINDING",
    )

    bundle = _bundle(run_root)
    bundle[PAIRED_BASELINE_POLICY][
        "frozen_environment_execution_qualification_sha256"
    ] = _sha("different-execution-qualification")
    _assert_hold(
        lambda: validate_typed_safe_bundle(
            bundle,
            trusted_code_snapshot=_trusted_code_snapshot(run_root),
        ),
        "HOLD_TYPED_SAFE_BUNDLE_BINDING",
    )


def test_bundle_exact_loader_replays_every_artifact(run_root: Path) -> None:
    refs = {}
    code_snapshot = _trusted_code_snapshot(run_root)
    for name, payload in _bundle(run_root).items():
        relative = Path("development") / "authorities" / f"bundle-{name}.json"
        digest = write_typed_safe_authority_exclusive(
            run_root,
            run_root / relative,
            payload,
            trusted_code_snapshot=(
                code_snapshot if name == ANALYSIS_CODE_TREE_MANIFEST else None
            ),
        )
        refs[name] = {"path": relative.as_posix(), "sha256": digest}

    result = load_typed_safe_bundle_exact(
        run_root,
        refs,
        trusted_code_snapshot=code_snapshot,
    )
    assert result["status"] == TYPED_DEVELOPMENT_SAFE_BUNDLE_NOT_TRAINING_READY
    assert result["training_ready"] is False

    bad_refs = copy.deepcopy(refs)
    bad_refs[ROUTE_AUTHORIZATION]["path"] = "../route.json"
    _assert_hold(
        lambda: load_typed_safe_bundle_exact(
            run_root,
            bad_refs,
            trusted_code_snapshot=code_snapshot,
        ),
        "HOLD_TYPED_SAFE_PATH",
    )


def test_common_prohibited_string_scan_catches_disguised_test_support() -> None:
    payload = _config_payload()
    payload["model"]["architecture_id"] = "MODEL_WITH_INTERNAL_TEST_SUPPORT"
    _assert_hold(
        lambda: validate_typed_safe_authority(payload),
        "HOLD_TYPED_SAFE_PROHIBITED_CONTENT",
    )


def test_validation_does_not_mutate_caller_payload(run_root: Path) -> None:
    payload = _payload(METHOD_PROTOCOL, run_root)
    before = copy.deepcopy(payload)
    assert validate_typed_safe_authority(payload) == before
    assert payload == before
