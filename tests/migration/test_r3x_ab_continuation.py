from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

import scripts.r3x.ab_continuation as ab
from scripts.r3x.artifact_graph import make_artifact_ref
from scripts.r3x.models import ModelConfig
from scripts.r3x.neural_method_registry import (
    A0_METHOD_ID,
    A1_CONTROL_METHOD_ID,
    A1_METHOD_ID,
    A1_UNK_ONLY_METHOD_ID,
    A1_WRONG_LOCAL_METHOD_ID,
    B_ADDITIVE_METHOD_ID,
    B_MATCHED_CONTROL_METHOD_ID,
    B_UNK_ONLY_METHOD_ID,
    B_WRONG_LOCAL_METHOD_ID,
    registry_sha256,
)


SHA = "a" * 64
INFO_SHA = "b" * 64
CONFIG_SHA = "c" * 64


def _ref(name: str, digit: str = "d") -> dict:
    return {
        "relative_path": f"development/{name}.json",
        "sha256": digit * 64,
    }


def _fake_registry() -> dict:
    entries = []
    methods = {
        A1_METHOD_ID: ("ARCH_A_CANDIDATE", "ROUTE_A_CANDIDATE"),
        A0_METHOD_ID: ("ARCH_A_REFERENCE", "ROUTE_A_MECHANISM_REFERENCE"),
        A1_CONTROL_METHOD_ID: ("ARCH_A_MATCHED", "ROUTE_A_MATCHED_CONTROL"),
        A1_WRONG_LOCAL_METHOD_ID: (
            "ARCH_A_WRONG",
            "ROUTE_A_WRONG_LOCAL_CONTROL",
        ),
        A1_UNK_ONLY_METHOD_ID: ("ARCH_A_UNK", "ROUTE_A_UNK_CONTROL"),
        B_ADDITIVE_METHOD_ID: ("ARCH_B_ADDITIVE", "ROUTE_B_ADDITIVE"),
        B_MATCHED_CONTROL_METHOD_ID: (
            "ARCH_B_MATCHED",
            "ROUTE_B_MATCHED_CONTROL",
        ),
        B_WRONG_LOCAL_METHOD_ID: (
            "ARCH_B_WRONG",
            "ROUTE_B_WRONG_LOCAL_CONTROL",
        ),
        B_UNK_ONLY_METHOD_ID: ("ARCH_B_UNK", "ROUTE_B_UNK_CONTROL"),
    }
    for method_id, (architecture_id, scientific_role) in methods.items():
        entries.append({
            "method_id": method_id,
            "architecture_id": architecture_id,
            "scientific_role": scientific_role,
        })
    return {"methods": entries}


def _method_spec(
    method_id: str,
    architecture_id: str,
    scientific_role: str,
    target: str | None = None,
) -> dict:
    return {
        "method_id": method_id,
        "architecture_id": architecture_id,
        "scientific_role": scientific_role,
        "requires_parameter_match_to_role": target,
    }


def _policy(monkeypatch: pytest.MonkeyPatch) -> dict:
    fake_registry = _fake_registry()
    fake_sha = hashlib.sha256(b"future-complete-registry").hexdigest()
    monkeypatch.setattr(ab, "canonical_registry_payload", lambda: fake_registry)
    monkeypatch.setattr(ab, "registry_sha256", lambda: fake_sha)
    a_roles = {
        "candidate": _method_spec(
            A1_METHOD_ID, "ARCH_A_CANDIDATE", "ROUTE_A_CANDIDATE"
        ),
        "mechanism_reference": _method_spec(
            A0_METHOD_ID, "ARCH_A_REFERENCE", "ROUTE_A_MECHANISM_REFERENCE"
        ),
        "matched_control": _method_spec(
            A1_CONTROL_METHOD_ID,
            "ARCH_A_MATCHED",
            "ROUTE_A_MATCHED_CONTROL",
            "candidate",
        ),
        "wrong_local_control": _method_spec(
            A1_WRONG_LOCAL_METHOD_ID,
            "ARCH_A_WRONG",
            "ROUTE_A_WRONG_LOCAL_CONTROL",
            "candidate",
        ),
        "unk_control": _method_spec(
            A1_UNK_ONLY_METHOD_ID,
            "ARCH_A_UNK",
            "ROUTE_A_UNK_CONTROL",
            "candidate",
        ),
    }
    b_roles = {
        "candidate": _method_spec(
            B_ADDITIVE_METHOD_ID, "ARCH_B_ADDITIVE", "ROUTE_B_ADDITIVE"
        ),
        "a_reference": _method_spec(
            A1_METHOD_ID, "ARCH_A_CANDIDATE", "ROUTE_A_CANDIDATE"
        ),
        "matched_control": _method_spec(
            B_MATCHED_CONTROL_METHOD_ID,
            "ARCH_B_MATCHED",
            "ROUTE_B_MATCHED_CONTROL",
            "candidate",
        ),
        "wrong_local_control": _method_spec(
            B_WRONG_LOCAL_METHOD_ID,
            "ARCH_B_WRONG",
            "ROUTE_B_WRONG_LOCAL_CONTROL",
            "candidate",
        ),
        "unk_control": _method_spec(
            B_UNK_ONLY_METHOD_ID,
            "ARCH_B_UNK",
            "ROUTE_B_UNK_CONTROL",
            "candidate",
        ),
    }
    payload = {
        "schema_version": ab.POLICY_SCHEMA,
        "artifact_role": ab.POLICY_ROLE,
        "status": "FROZEN_PRE_OUTCOME",
        "policy_id": "R3X_AB_CONTINUATION_TEST_POLICY",
        "partition": "ORDINARY_DEVELOPMENT",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "selector_schema": "r3x_development_strongest_baseline_selection_v3",
        "development_selector_ref": _ref("selector", "2"),
        "sign_gate": 0.60,
        "registered_seed_ids": [11, 23, 42, 67, 101],
        "ensemble_rule": "FIXED_FIVE_MODEL_TOTAL_VARIANCE_ENSEMBLE",
        "neural_evidence_schema": ab.CONTINUATION_NEURAL_EVIDENCE_SCHEMA,
        "registry_schema": ab.REGISTRY_SCHEMA,
        "registry_sha256": fake_sha,
        "maximum_parameter_gap_fraction": 0.05,
        "paired_metric_id": "MACRO_COMPONENT_CELL_BALANCED_SIGN_ACCURACY",
        "paired_family_alpha": 0.05,
        "paired_confidence": 0.95,
        "paired_bootstrap": {
            "n_resamples": 2000,
            "seed": 23,
            "min_components_per_cell": 10,
            "max_attempt_multiplier": 20,
        },
        "paired_permutation": {
            "n_permutations": 10000,
            "seed": 42,
            "exact_component_limit": 20,
            "min_components_per_cell": 10,
        },
        "matched_control_direction": ab.MATCHED_DIRECTION,
        "control_nonreplication_rule": ab.CONTROL_NONREPLICATION_RULE,
        "a_method_roles": a_roles,
        "b_method_roles": b_roles,
        "a_model_binding_refs": _role_refs(ab.A_ROLE_ORDER, "a_binding"),
        "b_model_binding_refs": _role_refs(ab.B_ROLE_ORDER, "b_binding"),
        "a_comparisons": [
            {
                "comparison_id": comparison_id,
                "left_role": left,
                "right_role": right,
                "rule": ab.POSITIVE_RULE,
            }
            for comparison_id, left, right in ab.A_COMPARISON_LAYOUT
        ],
        "b_comparisons": [
            {
                "comparison_id": comparison_id,
                "left_role": left,
                "right_role": right,
                "rule": ab.POSITIVE_RULE,
            }
            for comparison_id, left, right in ab.B_COMPARISON_LAYOUT
        ],
        "b_interaction_status": ab.B_INTERACTION_STATUS,
        "c3_status": ab.C3_STATUS,
        "c3_guidance_position": ab.C3_GUIDANCE_POSITION,
        "ordinary_development_only": True,
        "internal_test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return ab.validate_continuation_policy(payload)


def _selector() -> dict:
    return {
        "status": "HOLD_RECOVERY_ONLY_NON_QUALIFIED_COMPARATOR",
        "selected_method_id": "BASELINE",
        "can_satisfy_pretest_freeze": False,
        "development_evidence_refs": {"BASELINE": _ref("baseline", "e")},
        "run_start_sha256": "1" * 64,
        "activation_manifest_sha256": "2" * 64,
        "train_development_outcomes_sha256": "3" * 64,
        "non_data_authority_hashes": {
            "model_optimizer_seed_config_sha256": CONFIG_SHA,
            "other_authority_sha256": "4" * 64,
        },
        "information_set_fingerprint": INFO_SHA,
    }


def _rows() -> list[dict]:
    rows = []
    for index in range(10):
        rows.append({
            "r3_input_row_sha256": f"{index + 1:064x}",
            "study": "R3X_ORDINARY_DEVELOPMENT",
            "estimand_id": "ESTIMAND",
            "context_id_v2": "CELL",
            "actual_edit_count": 1,
            "source_component_id": f"COMPONENT_{index:02d}",
            "delta": 1.0 if index % 2 == 0 else -1.0,
        })
    return rows


def _bundle(
    method_id: str,
    predictions: list[float],
    *,
    architecture_id: str,
    five_seed: bool = True,
) -> dict:
    rows = _rows()
    row_ids = [row["r3_input_row_sha256"] for row in rows]
    component_assignment = [
        {
            "row_id": row["r3_input_row_sha256"],
            "cell_id": "ESTIMAND::CELL",
            "source_component_id": row["source_component_id"],
        }
        for row in rows
    ]
    sign = sum(
        (prediction >= 0.0) == (row["delta"] > 0.0)
        for prediction, row in zip(predictions, rows)
    ) / len(rows)
    return {
        "evidence": {
            "method_id": method_id,
            "information_set_fingerprint": INFO_SHA,
            "non_data_authority_hashes": {
                "model_optimizer_seed_config_sha256": CONFIG_SHA,
            },
            "aggregate_development_metrics": {
                "macro_component_cell_sign_accuracy": sign,
            },
        },
        "provenance": {},
        "five_seed_verified": five_seed,
        "checkpoint_architecture_ids": [architecture_id] if five_seed else [],
        "rows": rows,
        "predictions": predictions,
        "registered_cell_ids": ["ESTIMAND::CELL"],
        "ordered_row_ids_sha256": ab._canonical_sha256(row_ids),
        "row_set_sha256": ab._canonical_sha256(sorted(row_ids)),
        "component_assignment_sha256": ab._canonical_sha256(component_assignment),
    }


def _binding(role_spec: dict, role_specs: dict) -> dict:
    target = role_spec["requires_parameter_match_to_role"]
    compiled = {
        "training_variant_id": "PRIMARY",
        "control_intervention": "NONE",
        "model_config_sha256": CONFIG_SHA,
        "matched_reference_method_id": (
            None if target is None else role_specs[target]["method_id"]
        ),
        "parameter_match": (
            None
            if target is None
            else {
                "matched": True,
                "gap_fraction": 0.01,
                "maximum_gap_fraction": 0.05,
            }
        ),
    }
    return {
        "method_id": role_spec["method_id"],
        "architecture_id": role_spec["architecture_id"],
        "scientific_role": role_spec["scientific_role"],
        "registry_schema": ab.REGISTRY_SCHEMA,
        "registry_sha256": ab.registry_sha256(),
        "information_set_fingerprint": INFO_SHA,
        "predictor_config_sha256": CONFIG_SHA,
        "compiled_binding": compiled,
        "compiled_binding_sha256": ab._canonical_sha256(compiled),
    }


def _role_refs(roles: tuple[str, ...], prefix: str) -> dict[str, dict]:
    return {role: _ref(f"{prefix}_{role}", format(index + 5, "x")) for index, role in enumerate(roles)}


def _install_replays(
    monkeypatch: pytest.MonkeyPatch,
    policy: dict,
    *,
    route: str,
    candidate_predictions: list[float] | None = None,
    wrong_local_predictions: list[float] | None = None,
    five_seed_overrides: dict[str, bool] | None = None,
    independent_train_refit_verified: bool = False,
) -> tuple[dict, dict, dict, dict]:
    selector = _selector()
    role_order = ab.A_ROLE_ORDER if route == "A" else ab.B_ROLE_ORDER
    role_specs = policy[f"{route.lower()}_method_roles"]
    truth = [row["delta"] for row in _rows()]
    inverse = [-value for value in truth]
    bundles = {}
    for role in role_order:
        spec = role_specs[role]
        if role == "candidate":
            predictions = candidate_predictions or truth
        elif role == "wrong_local_control" and wrong_local_predictions is not None:
            predictions = wrong_local_predictions
        else:
            predictions = inverse
        bundles[spec["method_id"]] = _bundle(
            spec["method_id"],
            predictions,
            architecture_id=spec["architecture_id"],
            five_seed=(five_seed_overrides or {}).get(role, True),
        )
    bundles["BASELINE"] = _bundle(
        "BASELINE", inverse, architecture_id="BASELINE_ARCH", five_seed=False
    )
    bindings = {
        role: _binding(role_specs[role], role_specs) for role in role_order
    }
    evidence_refs = _role_refs(role_order, f"{route.lower()}_evidence")
    binding_refs = _role_refs(role_order, f"{route.lower()}_binding")
    neural_refs = _role_refs(role_order, f"{route.lower()}_neural_receipt")
    provenance_refs = _role_refs(role_order, f"{route.lower()}_predictor_provenance")
    ensemble_refs = _role_refs(role_order, f"{route.lower()}_ensemble_prediction")
    for role in role_order:
        bundle = bundles[role_specs[role]["method_id"]]
        bundle["evidence"]["predictor_provenance_ref"] = provenance_refs[role]
        bundle["provenance"] = {
            "estimands": [
                {
                    "estimand_id": "ESTIMAND",
                    "cells": [
                        {
                            "cell_id": "CELL",
                            "prediction_jsonl_ref": ensemble_refs[role],
                        }
                    ],
                }
            ]
        }
    binding_by_path = {
        reference["relative_path"]: bindings[role]
        for role, reference in binding_refs.items()
    }

    monkeypatch.setattr(ab, "load_continuation_policy_exact", lambda *args, **kwargs: policy)
    monkeypatch.setattr(ab, "_load_selector", lambda *args, **kwargs: selector)

    def load_bundle(run_root, reference, *, selector, method_id, predictor_verifiers):
        return bundles[method_id]

    monkeypatch.setattr(ab, "_load_method_bundle", load_bundle)
    monkeypatch.setattr(
        ab,
        "load_model_binding_evidence_exact",
        lambda run_root, reference: binding_by_path[reference["relative_path"]],
    )
    receipt_by_path = {}
    for role, reference in neural_refs.items():
        bundle = bundles[role_specs[role]["method_id"]]
        binding = bindings[role]
        receipt_by_path[reference["relative_path"]] = {
            "predictor_provenance_ref": provenance_refs[role],
            "method_id": role_specs[role]["method_id"],
            "architecture_id": binding["architecture_id"],
            "registry_schema": binding["registry_schema"],
            "registry_sha256": binding["registry_sha256"],
            "compiled_binding_sha256": binding["compiled_binding_sha256"],
            "model_config_sha256": binding["compiled_binding"][
                "model_config_sha256"
            ],
            "fit_config_sha256": "d" * 64,
            "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
            "information_set_fingerprint": INFO_SHA,
            "predictor_config_authority_sha256": CONFIG_SHA,
            "run_start_sha256": "1" * 64,
            "activation_manifest_sha256": "2" * 64,
            "train_development_outcomes_sha256": "3" * 64,
            "non_data_authority_hashes": _selector()["non_data_authority_hashes"],
            "registered_cell_ids": bundle["registered_cell_ids"],
            "continuation_development_ordered_row_ids_sha256": bundle[
                "ordered_row_ids_sha256"
            ],
            "continuation_development_row_set_sha256": bundle["row_set_sha256"],
            "training_variant_id": binding["compiled_binding"][
                "training_variant_id"
            ],
            "control_intervention": binding["compiled_binding"][
                "control_intervention"
            ],
            "ensemble_rule": "FIXED_FIVE_MODEL_TOTAL_VARIANCE_ENSEMBLE",
            "estimands": [
                {
                    "estimand_id": "ESTIMAND",
                    "seeds": [11, 23, 42, 67, 101],
                    "cells": [
                        {
                            "cell_id": "CELL",
                            "ensemble_prediction_ref": ensemble_refs[role],
                        }
                    ],
                }
            ],
            "independent_train_refit_verified": independent_train_refit_verified,
            "independent_component_assignment_replay_verified": (
                independent_train_refit_verified
            ),
            "independent_development_target_replay_verified": (
                independent_train_refit_verified
            ),
        }
    monkeypatch.setattr(
        ab,
        "load_continuation_neural_evidence_exact",
        lambda run_root, reference: receipt_by_path[reference["relative_path"]],
    )
    return evidence_refs, binding_refs, bundles, neural_refs


def _selector_context() -> dict:
    return {
        "expected_run_start_sha256": "1" * 64,
        "expected_activation_manifest_sha256": "2" * 64,
        "expected_train_development_outcomes_sha256": "3" * 64,
        "expected_non_data_authority_hashes": {},
        "expected_eligible_method_order": [],
        "expected_required_estimand_cells": [],
        "predictor_verifiers": {},
    }


def test_policy_freezes_gate_c3_registry_roles_and_comparison_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    assert policy["sign_gate"] == 0.60
    assert policy["c3_status"] == "PROPOSED_REDLINE_NOT_ACTIVE"
    assert policy["b_interaction_status"] == "NOT_RUN_ACTION_SUPPORT"
    assert policy["registered_seed_ids"] == [11, 23, 42, 67, 101]
    assert [item["comparison_id"] for item in policy["a_comparisons"]] == [
        item[0] for item in ab.A_COMPARISON_LAYOUT
    ]

    changed = copy.deepcopy(policy)
    changed["schema_version"] = "r3x_ab_continuation_policy_v1"
    with pytest.raises(ab.ABContinuationError, match="unsupported"):
        ab.validate_continuation_policy(changed)
    changed = copy.deepcopy(policy)
    changed["schema_version"] = "r3x_ab_continuation_policy_v2"
    with pytest.raises(ab.ABContinuationError, match="unsupported"):
        ab.validate_continuation_policy(changed)
    changed = copy.deepcopy(policy)
    changed["sign_gate"] = 0.59
    with pytest.raises(ab.ABContinuationError, match="0.60"):
        ab.validate_continuation_policy(changed)
    changed = copy.deepcopy(policy)
    changed["c3_status"] = "ACTIVE"
    with pytest.raises(ab.ABContinuationError, match="C3"):
        ab.validate_continuation_policy(changed)
    changed = copy.deepcopy(policy)
    changed["posthoc_comparison"] = True
    with pytest.raises(ab.ABContinuationError, match="strict schema"):
        ab.validate_continuation_policy(changed)


def test_current_registry_has_exact_development_controls_but_no_b_interaction() -> None:
    methods = {entry["method_id"] for entry in ab.canonical_registry_payload()["methods"]}
    assert {
        A1_WRONG_LOCAL_METHOD_ID,
        A1_UNK_ONLY_METHOD_ID,
        B_ADDITIVE_METHOD_ID,
        B_MATCHED_CONTROL_METHOD_ID,
        B_WRONG_LOCAL_METHOD_ID,
        B_UNK_ONLY_METHOD_ID,
    }.issubset(methods)
    assert not any("INTERACTION" in method for method in methods)
    assert ab.B_INTERACTION_STATUS == "NOT_RUN_ACTION_SUPPORT"


def test_real_registry_roles_compile_into_the_frozen_a_b_policy_layout() -> None:
    entries = {
        entry["method_id"]: entry
        for entry in ab.canonical_registry_payload()["methods"]
    }

    def role(method_id: str, target: str | None = None) -> dict:
        entry = entries[method_id]
        return {
            "method_id": method_id,
            "architecture_id": entry["architecture_id"],
            "scientific_role": entry["scientific_role"],
            "requires_parameter_match_to_role": target,
        }

    a_roles = {
        "candidate": role(A1_METHOD_ID),
        "mechanism_reference": role(A0_METHOD_ID),
        "matched_control": role(A1_CONTROL_METHOD_ID, "candidate"),
        "wrong_local_control": role(A1_WRONG_LOCAL_METHOD_ID, "candidate"),
        "unk_control": role(A1_UNK_ONLY_METHOD_ID, "candidate"),
    }
    b_roles = {
        "candidate": role(B_ADDITIVE_METHOD_ID),
        "a_reference": role(A1_METHOD_ID),
        "matched_control": role(B_MATCHED_CONTROL_METHOD_ID, "candidate"),
        "wrong_local_control": role(B_WRONG_LOCAL_METHOD_ID, "candidate"),
        "unk_control": role(B_UNK_ONLY_METHOD_ID, "candidate"),
    }
    assert set(
        ab._method_specs(
            a_roles,
            route="A",
            role_order=ab.A_ROLE_ORDER,
            registry_entries=entries,
        )
    ) == set(ab.A_ROLE_ORDER)
    assert set(
        ab._method_specs(
            b_roles,
            route="B",
            role_order=ab.B_ROLE_ORDER,
            registry_entries=entries,
        )
    ) == set(ab.B_ROLE_ORDER)


@pytest.mark.parametrize(
    "route_key,left_role,right_role",
    [
        ("a_method_roles", "candidate", "mechanism_reference"),
        ("b_method_roles", "matched_control", "wrong_local_control"),
    ],
)
def test_policy_rejects_exact_registry_method_role_swaps(
    monkeypatch: pytest.MonkeyPatch,
    route_key: str,
    left_role: str,
    right_role: str,
) -> None:
    policy = _policy(monkeypatch)
    changed = copy.deepcopy(policy)
    changed[route_key][left_role], changed[route_key][right_role] = (
        changed[route_key][right_role],
        changed[route_key][left_role],
    )
    with pytest.raises(ab.ABContinuationError, match="frozen exact registry role"):
        ab.validate_continuation_policy(changed)


def test_model_binding_recompiles_real_registry_and_rejects_self_report() -> None:
    cfg = ModelConfig(
        vocab_size=5,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        max_len=16,
        n_endpoints=3,
        n_contexts=4,
        max_edits=5,
        pair_rank=2,
    )
    torch.manual_seed(918273)
    rng_before = torch.random.get_rng_state().clone()
    payload = ab.build_model_binding_evidence(
        method_id=A1_CONTROL_METHOD_ID,
        model_config=cfg,
        expected_registry_sha256=registry_sha256(),
        predictor_config_sha256=CONFIG_SHA,
        information_set_fingerprint=INFO_SHA,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert payload["parameter_match_qualified"] is True
    assert payload["compiled_binding"]["parameter_match"]["gap_fraction"] <= 0.05
    changed = copy.deepcopy(payload)
    changed["parameter_match_qualified"] = False
    with pytest.raises(ab.ABContinuationError, match="not recomputable"):
        ab.validate_model_binding_evidence(changed)
    changed = copy.deepcopy(payload)
    changed["compiled_binding"]["parameter_count"] += 1
    with pytest.raises(ab.ABContinuationError, match="differs from exact registry"):
        ab.validate_model_binding_evidence(changed)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    "method_id,expected_match",
    [
        (A1_WRONG_LOCAL_METHOD_ID, True),
        (A1_UNK_ONLY_METHOD_ID, True),
        (B_ADDITIVE_METHOD_ID, None),
        (B_MATCHED_CONTROL_METHOD_ID, True),
        (B_WRONG_LOCAL_METHOD_ID, True),
        (B_UNK_ONLY_METHOD_ID, True),
    ],
)
def test_new_a_b_model_bindings_are_recompiled_not_self_attested(
    method_id: str,
    expected_match: bool | None,
) -> None:
    cfg = ModelConfig(
        vocab_size=5,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        max_len=20,
        n_endpoints=3,
        n_contexts=4,
        max_edits=5,
        pair_rank=2,
    )
    payload = ab.build_model_binding_evidence(
        method_id=method_id,
        model_config=cfg,
        expected_registry_sha256=registry_sha256(),
        predictor_config_sha256=CONFIG_SHA,
        information_set_fingerprint=INFO_SHA,
    )
    assert payload["parameter_match_qualified"] is expected_match
    assert payload["compiled_binding"]["execution_authorized"] is False
    assert payload["compiled_binding"]["training_ready"] is False
    assert payload["compiled_binding"]["publication"] is False
    assert payload["compiled_binding"]["publication_evidence"] is False
    assert payload["compiled_binding"]["b_interaction_status"] == (
        "NOT_RUN_ACTION_SUPPORT"
    )


def test_selector_and_model_binding_refs_are_pre_outcome_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, _, _ = _install_replays(monkeypatch, policy, route="A")
    with pytest.raises(ab.ABContinuationError, match="selector ref differs"):
        ab.build_a_continuation_decision(
            Path("/tmp/ordinary_run"),
            policy_ref=_ref("policy", "1"),
            selector_ref=_ref("other_selector", "f"),
            method_evidence_refs=evidence_refs,
            model_binding_refs=binding_refs,
            selector_context=_selector_context(),
            predictor_verifiers={},
        )
    changed_bindings = copy.deepcopy(binding_refs)
    changed_bindings["candidate"] = _ref("postoutcome_candidate_binding", "f")
    with pytest.raises(ab.ABContinuationError, match="pre-outcome policy freeze"):
        ab.build_a_continuation_decision(
            Path("/tmp/ordinary_run"),
            policy_ref=_ref("policy", "1"),
            selector_ref=_ref("selector", "2"),
            method_evidence_refs=evidence_refs,
            model_binding_refs=changed_bindings,
            selector_context=_selector_context(),
            predictor_verifiers={},
        )


def test_generic_a_evidence_cannot_go_without_exact_neural_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, _, _ = _install_replays(monkeypatch, policy, route="A")
    decision = ab.build_a_continuation_decision(
        Path("/tmp/ordinary_run"),
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        method_evidence_refs=evidence_refs,
        model_binding_refs=binding_refs,
        selector_context=_selector_context(),
        predictor_verifiers={},
    )
    assert decision["decision"] == "HOLD"
    assert decision["status"] == ab.HOLD_STATUS
    assert decision["reason_codes"] == [
        f"MISSING_NEURAL_STATE_RECEIPT:{role}" for role in ab.A_ROLE_ORDER
    ]
    assert decision["comparator_qualification"] == "RECOVERY_ONLY"
    assert decision["candidate_macro_sign_accuracy"] is None
    assert decision["sign_gate"] == 0.60
    assert decision["sign_gate_met"] is None
    assert decision["paired_family"] is None
    assert decision["scientific_checks"]["exact_neural_state_receipts"] is None
    assert decision["internal_test_access"] is False
    assert decision["sealed_access"] is False
    assert decision["downstream_unlock"] is False


def test_complete_state_load_receipts_without_independent_refit_remain_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, _, neural_refs = _install_replays(
        monkeypatch, policy, route="A"
    )
    decision = ab.build_a_continuation_decision(
        Path("/tmp/ordinary_run"),
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        method_evidence_refs=evidence_refs,
        model_binding_refs=binding_refs,
        neural_evidence_refs=neural_refs,
        selector_context=_selector_context(),
        predictor_verifiers={},
    )
    assert decision["decision"] == "HOLD"
    assert decision["reason_codes"] == [
        *[
            f"NEURAL_RECEIPT_NO_INDEPENDENT_TRAIN_REFIT:{role}"
            for role in ab.A_ROLE_ORDER
        ],
        *[
            f"NEURAL_RECEIPT_NO_INDEPENDENT_COMPONENT_ASSIGNMENT_REPLAY:{role}"
            for role in ab.A_ROLE_ORDER
        ],
        *[
            f"NEURAL_RECEIPT_NO_INDEPENDENT_DEVELOPMENT_TARGET_REPLAY:{role}"
            for role in ab.A_ROLE_ORDER
        ],
    ]
    assert decision["verified_neural_receipt_roles"] == list(ab.A_ROLE_ORDER)
    assert decision["scientific_checks"]["exact_neural_state_receipts"] is True
    assert (
        decision["scientific_checks"]["independent_train_refit_verified"] is False
    )
    assert (
        decision["scientific_checks"][
            "independent_component_assignment_replay_verified"
        ]
        is False
    )
    assert (
        decision["scientific_checks"][
            "independent_development_target_replay_verified"
        ]
        is False
    )
    assert decision["scientific_checks"]["same_fit_config_across_roles"] is True
    assert decision["bindings"] is None
    assert decision["candidate_macro_sign_accuracy"] is None
    assert decision["sign_gate_met"] is None
    assert decision["paired_family"] is None

    old = copy.deepcopy(decision)
    old["schema_version"] = "r3x_a_continuation_decision_v1"
    with pytest.raises(ab.ABContinuationError, match="schema/role"):
        ab.validate_continuation_decision(old)


def test_swapped_receipt_provenance_or_ensemble_ref_is_hold_without_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, _, neural_refs = _install_replays(
        monkeypatch, policy, route="A"
    )
    original_load = ab.load_continuation_neural_evidence_exact

    def swapped_provenance(run_root, reference):
        receipt = copy.deepcopy(original_load(run_root, reference))
        if reference == neural_refs["candidate"]:
            receipt["predictor_provenance_ref"] = _ref(
                "swapped_predictor_provenance", "f"
            )
            receipt["estimands"][0]["cells"][0][
                "ensemble_prediction_ref"
            ] = _ref("swapped_ensemble_prediction", "e")
        return receipt

    monkeypatch.setattr(
        ab, "load_continuation_neural_evidence_exact", swapped_provenance
    )
    decision = ab.build_a_continuation_decision(
        Path("/tmp/ordinary_run"),
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        method_evidence_refs=evidence_refs,
        model_binding_refs=binding_refs,
        neural_evidence_refs=neural_refs,
        selector_context=_selector_context(),
        predictor_verifiers={},
    )
    assert decision["decision"] == "HOLD"
    assert decision["reason_codes"] == [
        "NEURAL_STATE_RECEIPT_BINDING_MISMATCH:candidate"
    ]
    assert decision["bindings"] is None
    assert decision["candidate_macro_sign_accuracy"] is None
    assert decision["paired_family"] is None


def test_missing_control_is_hold_not_scientific_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, _, _ = _install_replays(monkeypatch, policy, route="A")
    evidence_refs.pop("unk_control")
    decision = ab.build_a_continuation_decision(
        Path("/tmp/ordinary_run"),
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        method_evidence_refs=evidence_refs,
        model_binding_refs=binding_refs,
        selector_context=_selector_context(),
        predictor_verifiers={},
    )
    assert decision["decision"] == "HOLD"
    assert decision["status"] == ab.HOLD_STATUS
    assert decision["reason_codes"] == [
        "MISSING_METHOD_EVIDENCE:unk_control",
        *[f"MISSING_NEURAL_STATE_RECEIPT:{role}" for role in ab.A_ROLE_ORDER],
    ]
    assert decision["candidate_macro_sign_accuracy"] is None


def test_seed_metric_average_or_non_five_seed_candidate_is_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, _, neural_refs = _install_replays(
        monkeypatch,
        policy,
        route="A",
        five_seed_overrides={"candidate": False},
    )
    decision = ab.build_a_continuation_decision(
        Path("/tmp/ordinary_run"),
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        method_evidence_refs=evidence_refs,
        model_binding_refs=binding_refs,
        neural_evidence_refs=neural_refs,
        selector_context=_selector_context(),
        predictor_verifiers={},
    )
    assert decision["decision"] == "HOLD"
    assert decision["scientific_checks"]["five_seed_ensembles"] is False
    assert decision["reason_codes"] == [
        "NOT_FIXED_FIVE_SEED_PREDICTION_ENSEMBLE:candidate"
    ]


def test_sign_gate_or_control_replication_is_scientific_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    truth = [row["delta"] for row in _rows()]
    half_correct = truth[:5] + [-value for value in truth[5:]]
    evidence_refs, binding_refs, _, neural_refs = _install_replays(
        monkeypatch,
        policy,
        route="A",
        candidate_predictions=half_correct,
        independent_train_refit_verified=True,
    )
    decision = ab.build_a_continuation_decision(
        Path("/tmp/ordinary_run"),
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        method_evidence_refs=evidence_refs,
        model_binding_refs=binding_refs,
        neural_evidence_refs=neural_refs,
        selector_context=_selector_context(),
        predictor_verifiers={},
    )
    assert decision["decision"] == "FAIL"
    assert decision["sign_gate_met"] is False
    assert "CANDIDATE_SIGN_ACCURACY_BELOW_0.60" in decision["reason_codes"]

    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, _, neural_refs = _install_replays(
        monkeypatch,
        policy,
        route="A",
        wrong_local_predictions=truth,
        independent_train_refit_verified=True,
    )
    decision = ab.build_a_continuation_decision(
        Path("/tmp/ordinary_run"),
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        method_evidence_refs=evidence_refs,
        model_binding_refs=binding_refs,
        neural_evidence_refs=neural_refs,
        selector_context=_selector_context(),
        predictor_verifiers={},
    )
    assert decision["decision"] == "FAIL"
    assert decision["sign_gate_met"] is True
    assert decision["scientific_checks"]["wrong_local_does_not_replicate"] is False
    assert "FAILED_PAIRED_COMPARISON:A_CANDIDATE_VS_WRONG_LOCAL_CONTROL" in decision["reason_codes"]


def test_component_assignment_or_extra_role_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, bundles, neural_refs = _install_replays(monkeypatch, policy, route="A")
    bundles[A1_CONTROL_METHOD_ID]["component_assignment_sha256"] = "0" * 64
    with pytest.raises(ab.ABContinuationError, match="cells/rows/components"):
        ab.build_a_continuation_decision(
            Path("/tmp/ordinary_run"),
            policy_ref=_ref("policy", "1"),
            selector_ref=_ref("selector", "2"),
            method_evidence_refs=evidence_refs,
            model_binding_refs=binding_refs,
            neural_evidence_refs=neural_refs,
            selector_context=_selector_context(),
            predictor_verifiers={},
        )

    evidence_refs["posthoc_control"] = _ref("posthoc", "f")
    with pytest.raises(ab.ABContinuationError, match="unknown roles"):
        ab.build_a_continuation_decision(
            Path("/tmp/ordinary_run"),
            policy_ref=_ref("policy", "1"),
            selector_ref=_ref("selector", "2"),
            method_evidence_refs=evidence_refs,
            model_binding_refs=binding_refs,
            neural_evidence_refs=neural_refs,
            selector_context=_selector_context(),
            predictor_verifiers={},
        )


def test_b_is_hold_until_exact_a_go_then_can_go_additive_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, bundles, neural_refs = _install_replays(
        monkeypatch,
        policy,
        route="B",
        independent_train_refit_verified=True,
    )
    a_hold = {
        "route": "A",
        "decision": "HOLD",
        "candidate_method_id": A1_METHOD_ID,
        "policy_ref": _ref("policy", "1"),
        "selector_ref": _ref("selector", "2"),
    }
    monkeypatch.setattr(ab, "load_continuation_decision_exact", lambda *args, **kwargs: a_hold)
    held = ab.build_b_continuation_decision(
        Path("/tmp/ordinary_run"),
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        a_decision_ref=_ref("a_decision", "3"),
        method_evidence_refs={},
        model_binding_refs={},
        selector_context=_selector_context(),
        predictor_verifiers={},
        a_replay_context={},
    )
    assert held["status"] == ab.HOLD_A_STATUS
    assert held["reason_codes"] == ["A_CONTINUATION_NOT_GO"]

    candidate = bundles[B_ADDITIVE_METHOD_ID]
    a_go = {
        "route": "A",
        "decision": "GO",
        "candidate_method_id": A1_METHOD_ID,
        "policy_ref": _ref("policy", "1"),
        "selector_ref": _ref("selector", "2"),
        "bindings": {
            "run_start_sha256": "1" * 64,
            "activation_manifest_sha256": "2" * 64,
            "train_development_outcomes_sha256": "3" * 64,
            "non_data_authority_hashes": _selector()["non_data_authority_hashes"],
            "information_set_fingerprint": INFO_SHA,
            "registered_cell_ids_sha256": ab._canonical_sha256(
                candidate["registered_cell_ids"]
            ),
            "ordered_row_ids_sha256": candidate["ordered_row_ids_sha256"],
            "row_set_sha256": candidate["row_set_sha256"],
            "component_assignment_sha256": candidate[
                "component_assignment_sha256"
            ],
        },
    }
    monkeypatch.setattr(ab, "load_continuation_decision_exact", lambda *args, **kwargs: a_go)
    go = ab.build_b_continuation_decision(
        Path("/tmp/ordinary_run"),
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        a_decision_ref=_ref("a_decision", "3"),
        method_evidence_refs=evidence_refs,
        model_binding_refs=binding_refs,
        neural_evidence_refs=neural_refs,
        selector_context=_selector_context(),
        predictor_verifiers={},
        a_replay_context={},
    )
    assert go["decision"] == "GO"
    assert go["scientific_checks"]["a_prerequisite_go"] is True
    assert go["b_interaction_status"] == "NOT_RUN_ACTION_SUPPORT"


def test_decision_validator_rejects_status_metric_holm_or_extra_field_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, _, neural_refs = _install_replays(
        monkeypatch,
        policy,
        route="A",
        independent_train_refit_verified=True,
    )
    decision = ab.build_a_continuation_decision(
        Path("/tmp/ordinary_run"),
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        method_evidence_refs=evidence_refs,
        model_binding_refs=binding_refs,
        neural_evidence_refs=neural_refs,
        selector_context=_selector_context(),
        predictor_verifiers={},
    )
    changed = copy.deepcopy(decision)
    changed["status"] = ab.FAIL_STATUS
    with pytest.raises(ab.ABContinuationError, match="status/decision"):
        ab.validate_continuation_decision(changed)
    changed = copy.deepcopy(decision)
    changed["candidate_macro_sign_accuracy"] = 0.61
    with pytest.raises(ab.ABContinuationError):
        # Internal schema can recompute the gate but exact load is needed to
        # detect a different still-passing metric.  Force a visible internal
        # contradiction here; exact replay is tested below.
        changed["sign_gate_met"] = False
        ab.validate_continuation_decision(changed)
    changed = copy.deepcopy(decision)
    first = changed["paired_family"]["comparison_order"][0]
    changed["paired_family"]["holm_adjusted_p_values"][first] = 0.049
    with pytest.raises(ab.ABContinuationError, match="Holm"):
        ab.validate_continuation_decision(changed)
    changed = copy.deepcopy(decision)
    changed["self_reported_pass"] = True
    with pytest.raises(ab.ABContinuationError, match="strict schema"):
        ab.validate_continuation_decision(changed)


def test_canonical_writer_loader_rebuilds_and_refuses_overwrite_or_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy(monkeypatch)
    evidence_refs, binding_refs, _, _ = _install_replays(monkeypatch, policy, route="A")
    decision = ab.build_a_continuation_decision(
        tmp_path,
        policy_ref=_ref("policy", "1"),
        selector_ref=_ref("selector", "2"),
        method_evidence_refs=evidence_refs,
        model_binding_refs=binding_refs,
        selector_context=_selector_context(),
        predictor_verifiers={},
    )
    development = tmp_path / "development"
    development.mkdir()
    path = development / "a_decision.json"
    replay_context = {
        "selector_context": _selector_context(),
        "predictor_verifiers": {},
        "a_replay_context": None,
    }
    monkeypatch.setattr(ab, "build_a_continuation_decision", lambda *args, **kwargs: decision)
    digest = ab.write_continuation_decision_exclusive(
        tmp_path, path, decision, replay_context=replay_context
    )
    reference = make_artifact_ref(tmp_path, path, digest)
    assert ab.load_continuation_decision_exact(
        tmp_path, reference, replay_context=replay_context
    ) == decision
    with pytest.raises(ab.ABContinuationError, match="already exists"):
        ab.write_continuation_decision_exclusive(
            tmp_path, path, decision, replay_context=replay_context
        )

    monkeypatch.setattr(
        ab,
        "build_a_continuation_decision",
        lambda *args, **kwargs: {**decision, "candidate_macro_sign_accuracy": 0.9},
    )
    with pytest.raises(ab.ABContinuationError, match="does not match exact replay"):
        ab.load_continuation_decision_exact(
            tmp_path, reference, replay_context=replay_context
        )


def test_refs_reject_test_sealed_and_non_development_namespaces() -> None:
    for path in (
        "internal_test/a.json",
        "development/internal_test/a.json",
        "development/sealed/a.json",
        "development/GSE246381/a.json",
    ):
        with pytest.raises(ab.ABContinuationError):
            ab._development_ref(
                {"relative_path": path, "sha256": SHA}, label="prohibited"
            )
