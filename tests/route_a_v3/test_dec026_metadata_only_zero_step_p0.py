from __future__ import annotations

import ast
import builtins
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/dec026_metadata_only_zero_step_p0.py"
)


def _load_candidate():
    name = "route_a_v3_dec026_zero_step_p0_validator_candidate_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _group(evidence: dict) -> dict:
    return {"declared_status": "PASS", "evidence": evidence}


def _complete_package() -> dict:
    executable_checks = {
        name: {
            "binding_id": f"binding::{name.lower()}",
            "binding_status": "BOUND_EXECUTABLE_NOT_RUN",
        }
        for name in (
            "LEGAL_CTMC_PRODUCTION_INTERFACE",
            "STOP_SUPPORT_FLOOR_ALIAS_BUDGET",
            "BASE_RECOVERY",
            "LEARNED_POTENTIAL_APPROXIMATION_ERROR",
            "LEGALITY",
            "TRAJECTORY_REPLAY",
            "PROVENANCE",
            "FAILURE_BUNDLE",
        )
    }
    successor_bindings = {
        name: f"binding::{name}"
        for name in (
            "executable",
            "configuration",
            "exact_reference",
            "environment",
            "input_branch_from_p0_1",
            "split",
            "seed",
            "optimizer",
            "budget",
            "device",
            "outputs",
            "stop_and_failure_destinations",
        )
    }
    return {
        "schema_version": "route_a_v3_dec026_zero_step_p0_metadata_package.v1",
        "package_role": "STATIC_METADATA_AND_AUTHORITY_BINDINGS_ONLY",
        "groups": {
            "P0.1": _group(
                {
                    "binding_route": "BOUND_EXISTING_CANONICAL_ASSET",
                    "binding": {
                        "authoritative_locator": "authority::canonical-6547",
                        "schema_binding_id": "binding::canonical-schema",
                        "row_count": 6547,
                        "membership_rule_id": "binding::canonical-membership",
                        "producing_provenance_id": "binding::canonical-provenance",
                        "materialization_authority_id": None,
                        "materialization_destination": None,
                        "materialization_exercised": False,
                        "independent_review_status": "INDEPENDENT_REVIEW_PASS",
                    },
                }
            ),
            "P0.2": _group(
                {
                    "authority_binding_record_id": "authority::prior-use",
                    "attestation_status": "FULL_PRIOR_ANALYTIC_USE_ATTESTATION_PASS",
                    "bound_input_count": 2,
                    "derived_asset_count": 1,
                    "attested_input_count": 2,
                    "attested_derived_asset_count": 1,
                    "unknown_history_count": 0,
                    "partial_history_count": 0,
                    "inferred_history_count": 0,
                    "dataset_level_only_history_count": 0,
                    "independent_review_status": "INDEPENDENT_REVIEW_PASS",
                }
            ),
            "P0.3": _group(
                {
                    "authority_binding_record_id": "authority::exposure-role",
                    "participating_study_count": 1,
                    "study_level_binding_count": 1,
                    "gse200304_included": True,
                    "all_study_role": "EXPOSED_DEVELOPMENT_ONLY",
                    "untouched_role_count": 0,
                    "sealed_role_count": 0,
                    "confirmatory_role_count": 0,
                    "later_confirmatory_reuse_eligible_count": 0,
                    "independent_review_status": "INDEPENDENT_REVIEW_PASS",
                }
            ),
            "P0.4": _group(
                {
                    "rights_authority_record_id": "authority::rights",
                    "input_count": 2,
                    "internal_processing_authorized_count": 2,
                    "training_authorized_count": 2,
                    "evaluation_authorized_count": 2,
                    "public_access_only_count": 0,
                    "redistribution_only_count": 0,
                    "permission_basis": "EXPLICIT_INTENDED_INTERNAL_PROCESS_TRAIN_EVALUATE",
                    "independent_review_status": "INDEPENDENT_REVIEW_PASS",
                }
            ),
            "P0.5": _group(
                {
                    "row_contract_binding_id": "binding::scientific-row-contract",
                    "contract_member_count": 6547,
                    "required_authoritative_fields": [
                        "source",
                        "candidate",
                        "endpoint_transform",
                        "endpoint_direction",
                        "biological_source_group",
                        "context",
                        "rights",
                        "exposure",
                        "membership",
                    ],
                    "missing_required_field_allowed": False,
                    "inferred_identity_allowed": False,
                    "enforcement_binding_id": "binding::row-contract-enforcement",
                    "independent_review_status": "INDEPENDENT_REVIEW_PASS",
                }
            ),
            "P0.6": _group(
                {
                    "route_binding_id": "binding::scratch-route",
                    "route": "SCRATCH_ONLY_NO_EXTERNAL_LEARNED_INPUTS",
                    "initialization": "RANDOM_INITIALIZATION",
                    "foundation_input_count": 0,
                    "warm_start_input_count": 0,
                    "resumed_checkpoint_count": 0,
                    "previously_failed_checkpoint_count": 0,
                    "external_learned_input_count": 0,
                    "checkpoint_reads_before_first_update": 0,
                    "independent_review_status": "INDEPENDENT_REVIEW_PASS",
                }
            ),
            "P0.7": _group(
                {
                    "split_authority_record_id": "authority::split",
                    "split_binding_id": "binding::prospective-split",
                    "frozen_before_label_bearing_access": True,
                    "outcome_blind": True,
                    "assignment_unit": "SOURCE_GROUP_WITH_KNOWN_DUPLICATE_COMPONENTS",
                    "source_group_disjoint": True,
                    "known_duplicate_disjoint": True,
                    "membership_adjustable_from_model_results": False,
                    "membership_adjustable_from_endpoint_results": False,
                    "independent_review_status": "INDEPENDENT_REVIEW_PASS",
                }
            ),
            "P0.8": _group(
                {
                    "policy_binding_id": "binding::single-run-policy",
                    "run_count": 1,
                    "seed": 20260814,
                    "architecture_binding_id": "binding::architecture",
                    "optimizer_binding_id": "binding::optimizer",
                    "learning_rate": 0.0001,
                    "schedule_binding_id": "binding::schedule",
                    "compute_budget_binding_id": "binding::compute-budget",
                    "checkpoint_emission_retention_rule_id": "binding::checkpoint-policy",
                    "terminal_checkpoint_rule_id": "binding::terminal-checkpoint",
                    "stop_rule_id": "binding::stop-rule",
                    "cuda_device_ownership_rule_id": "binding::device-ownership",
                    "aggregate_metric_set_id": "binding::aggregate-metrics",
                    "independent_exact_reference_binding_id": "binding::exact-reference",
                    "alternative_count_after_results": 0,
                    "selection_after_results_allowed": False,
                    "independent_review_status": "INDEPENDENT_REVIEW_PASS",
                }
            ),
            "P0.9": _group(
                {
                    "gate_bundle_binding_id": "binding::scientific-gates",
                    "checks": executable_checks,
                    "deferred_check_count": 0,
                    "independent_review_status": "INDEPENDENT_REVIEW_PASS",
                }
            ),
            "P0.10": _group(
                {
                    "parent_g0_candidate_id": (
                        "ROUTE_A_V3_A6_LEARNED_BASE_VALUE_G0_IMPLEMENTATION_CANDIDATE_V1"
                    ),
                    "successor_implementation_id": "candidate::future-active-successor",
                    "review_status": "INDEPENDENT_REVIEW_PASS",
                    "successor_state": "REVIEWED_READY_FOR_FUTURE_ACTIVE_SUCCESSOR",
                    "future_activation_scope": "ACTIVE_FOR_THIS_G1_ONE_RUN_ONLY",
                    "bindings": successor_bindings,
                    "draft_implementation": False,
                    "partial_binding": False,
                    "configurable_placeholder_count": 0,
                }
            ),
            "P0.11": _group(
                {
                    "lock_binding_id": "binding::state-locks",
                    "pre_p0_lock_state": {
                        "training_allowed": False,
                        "gpu_work_allowed": False,
                        "parameter_updates_allowed": False,
                    },
                    "p0_01_to_p0_10_required_pass_count": 10,
                    "conditional_one_run_atomic_unlock": True,
                    "conditional_unlock_scope": "ACTIVE_FOR_THIS_G1_ONE_RUN_ONLY",
                    "persistent_locks": {
                        "model_selection_allowed": False,
                        "qualification_change_allowed": False,
                        "credit_change_allowed": False,
                        "canonical_mutation_allowed": False,
                        "a6_pass_allowed": False,
                        "l3_claim_allowed": False,
                        "a7_unlock_allowed": False,
                        "private_or_sealed_access_allowed": False,
                        "scientific_claim_allowed": False,
                    },
                    "independent_review_status": "INDEPENDENT_REVIEW_PASS",
                }
            ),
        },
    }


def _assert_zero_touchpoints(runner, sentinel) -> None:
    assert sentinel.counts == {name: 0 for name in runner.FORBIDDEN_TOUCHPOINTS}
    sentinel.assert_zero()


def test_protocol_is_active_only_for_metadata_p0_and_lists_exact_eleven_groups() -> None:
    runner = _load_candidate()
    config = runner.load_candidate_config()
    assert config["document_status"] == (
        "FROZEN_USER_AUTHORIZED_METADATA_ONLY_ZERO_STEP_P0"
    )
    assert config["authority"]["decision_status"] == (
        "GRANTED_METADATA_ONLY_ZERO_STEP_P0"
    )
    assert config["authority"]["validator_activation_state"] == (
        "ACTIVE_METADATA_ONLY_ZERO_STEP_P0"
    )
    assert config["authority"]["validator_may_publish_exactly_one_aggregate_record"]
    assert config["authority"]["validator_may_launch_g1"] is False
    assert config["authority"]["g1_launch_requires_all_eleven_pass"] is True
    assert [(item["gate_id"], item["gate_name"]) for item in config["p0_groups"]] == list(
        runner.P0_GROUPS
    )
    assert len(config["p0_groups"]) == 11
    assert all(value is False for value in config["retained_locks"].values())
    binding = config["implementation_binding"]
    dynamic = {
        binding["status"],
        binding["implementation_commit"],
        binding["implementation_script_sha256"],
        binding["implementation_test_sha256"],
    }
    assert dynamic == {runner.UNKNOWN} or (
        binding["status"] == runner.BOUND
        and runner._is_hex(binding["implementation_commit"], 40)
        and runner._is_hex(binding["implementation_script_sha256"], 64)
        and runner._is_hex(binding["implementation_test_sha256"], 64)
    )


def test_current_metadata_submission_stops_with_three_pass_seven_fail_one_unknown() -> None:
    runner = _load_candidate()
    config = runner.load_protocol()
    sentinel = runner.ForbiddenTouchpointSentinel()
    result = runner.evaluate_zero_step_p0(
        config["current_submission"],
        candidate_config=config,
        touchpoints=sentinel,
    )
    assert result["result_status"] == (
        "ZERO_STEP_P0_FAILURE_STOP_BEFORE_DATA_CUDA_MODEL"
    )
    counts = runner._gate_counts(result)
    assert counts == {
        "pass": 3,
        "fail_closed": 7,
        "unknown_not_asserted": 1,
        "total": 11,
    }
    statuses = {item["gate_id"]: item["status"] for item in result["gate_statuses"]}
    assert {gate_id for gate_id, status in statuses.items() if status == "PASS"} == {
        "P0.3",
        "P0.6",
        "P0.11",
    }
    assert statuses["P0.2"] == "UNKNOWN_NOT_ASSERTED"
    assert statuses["P0.10"] == "FAIL_CLOSED_SUCCESSOR_NOT_BOUND"
    _assert_zero_touchpoints(runner, sentinel)


def test_aggregate_record_is_one_file_and_contains_no_member_or_execution_payload(
    tmp_path: Path,
) -> None:
    runner = _load_candidate()
    config = runner.load_protocol()
    result = runner.evaluate_zero_step_p0(
        config["current_submission"], candidate_config=config
    )
    record = runner.build_aggregate_record(config, result)
    output_dir = tmp_path / "one-record"
    report_path = runner.write_aggregate_record(output_dir, record)
    assert report_path.name == runner.REPORT_FILENAME
    assert [path.name for path in output_dir.iterdir()] == [runner.REPORT_FILENAME]
    loaded = runner.load_json_object(report_path)
    assert loaded == record
    assert loaded["g1_one_run_eligible"] is False
    assert loaded["g1_launched"] is False
    rendered = report_path.read_text(encoding="utf-8").lower()
    for forbidden in (
        "member_id",
        "sequence",
        "endpoint_value",
        "split_assignment",
        "model_state",
        "optimizer_state",
        "checkpoint_path",
        "device_uuid",
    ):
        assert forbidden not in rendered


def test_success_shaped_metadata_is_eligible_but_never_launches_or_touches_operations(
    monkeypatch,
) -> None:
    runner = _load_candidate()
    config = runner.load_candidate_config()
    package = _complete_package()
    sentinel = runner.ForbiddenTouchpointSentinel()

    def forbidden_file_read(*args, **kwargs):
        raise AssertionError("zero-step evaluation attempted file I/O")

    monkeypatch.setattr(Path, "read_bytes", forbidden_file_read)
    monkeypatch.setattr(Path, "read_text", forbidden_file_read)
    monkeypatch.setattr(builtins, "open", forbidden_file_read)

    result = runner.evaluate_zero_step_p0(
        package,
        candidate_config=config,
        touchpoints=sentinel,
    )
    assert result["result_status"] == (
        "ZERO_STEP_P0_PASS_G1_ONE_RUN_ELIGIBLE_NOT_LAUNCHED"
    )
    assert result["authority_status"] == "GRANTED_METADATA_ONLY_ZERO_STEP_P0"
    assert len(result["gate_statuses"]) == 11
    assert {item["status"] for item in result["gate_statuses"]} == {"PASS"}
    assert set(result) == {"result_status", "authority_status", "gate_statuses"}
    assert all(set(item) == {"gate_id", "gate_name", "status"} for item in result["gate_statuses"])
    _assert_zero_touchpoints(runner, sentinel)


@pytest.mark.parametrize("gate_id", [f"P0.{index}" for index in range(1, 12)])
def test_each_of_the_eleven_groups_fails_closed_when_partial(gate_id: str) -> None:
    runner = _load_candidate()
    config = runner.load_candidate_config()
    package = _complete_package()
    evidence = package["groups"][gate_id]["evidence"]
    evidence.pop(next(iter(evidence)))
    sentinel = runner.ForbiddenTouchpointSentinel()
    result = runner.evaluate_zero_step_p0(
        package,
        candidate_config=config,
        touchpoints=sentinel,
    )
    statuses = {item["gate_id"]: item["status"] for item in result["gate_statuses"]}
    assert statuses[gate_id] == "FAIL_CLOSED_PARTIAL_GROUP"
    assert set(statuses) == {f"P0.{index}" for index in range(1, 12)}
    assert result["result_status"] == (
        "ZERO_STEP_P0_FAILURE_STOP_BEFORE_DATA_CUDA_MODEL"
    )
    _assert_zero_touchpoints(runner, sentinel)


@pytest.mark.parametrize(
    ("gate_id", "mutator"),
    [
        ("P0.1", lambda value: value["binding"].update(row_count=6546)),
        ("P0.2", lambda value: value.update(unknown_history_count=1)),
        ("P0.3", lambda value: value.update(all_study_role="UNTOUCHED_CONFIRMATORY")),
        ("P0.4", lambda value: value.update(training_authorized_count=1)),
        ("P0.5", lambda value: value.update(inferred_identity_allowed=True)),
        ("P0.6", lambda value: value.update(warm_start_input_count=1)),
        ("P0.7", lambda value: value.update(source_group_disjoint=False)),
        ("P0.8", lambda value: value.update(run_count=2)),
        ("P0.9", lambda value: value.update(deferred_check_count=1)),
        ("P0.10", lambda value: value.update(partial_binding=True)),
        (
            "P0.11",
            lambda value: value["persistent_locks"].update(model_selection_allowed=True),
        ),
    ],
)
def test_each_group_rejects_a_complete_but_wrong_semantic_state(
    gate_id: str, mutator
) -> None:
    runner = _load_candidate()
    config = runner.load_candidate_config()
    package = _complete_package()
    mutator(package["groups"][gate_id]["evidence"])
    sentinel = runner.ForbiddenTouchpointSentinel()
    result = runner.evaluate_zero_step_p0(
        package,
        candidate_config=config,
        touchpoints=sentinel,
    )
    statuses = {item["gate_id"]: item["status"] for item in result["gate_statuses"]}
    assert statuses[gate_id] == "FAIL_CLOSED_SEMANTIC_MISMATCH"
    _assert_zero_touchpoints(runner, sentinel)


def test_p01_unexercised_membership_preserving_authority_is_the_only_other_route() -> None:
    runner = _load_candidate()
    config = runner.load_candidate_config()
    package = _complete_package()
    package["groups"]["P0.1"]["evidence"] = {
        "binding_route": "BOUND_UNEXERCISED_MEMBERSHIP_PRESERVING_MATERIALIZATION_AUTHORITY",
        "binding": {
            "authoritative_source_locator": "authority::source",
            "frozen_membership_rule_id": "binding::membership",
            "destination_locator": "destination::future-single-materialization",
            "materialization_authority_id": "authority::future-single-materialization",
            "frozen_member_count": 6547,
            "materialization_exercised": False,
            "member_add_allowed": False,
            "member_drop_allowed": False,
            "member_relabel_allowed": False,
            "member_deduplicate_allowed": False,
            "member_resample_allowed": False,
            "observed_data_selection_allowed": False,
            "independent_review_status": "INDEPENDENT_REVIEW_PASS",
        },
    }
    sentinel = runner.ForbiddenTouchpointSentinel()
    result = runner.evaluate_zero_step_p0(
        package,
        candidate_config=config,
        touchpoints=sentinel,
    )
    status = next(item for item in result["gate_statuses"] if item["gate_id"] == "P0.1")
    assert status["status"] == "PASS"
    assert result["result_status"] == (
        "ZERO_STEP_P0_PASS_G1_ONE_RUN_ELIGIBLE_NOT_LAUNCHED"
    )
    _assert_zero_touchpoints(runner, sentinel)


def test_missing_nonpass_and_unexpected_groups_have_distinct_fail_closed_statuses() -> None:
    runner = _load_candidate()
    config = runner.load_candidate_config()

    missing = _complete_package()
    missing["groups"].pop("P0.4")
    result = runner.evaluate_zero_step_p0(missing, candidate_config=config)
    statuses = {item["gate_id"]: item["status"] for item in result["gate_statuses"]}
    assert statuses["P0.4"] == "FAIL_CLOSED_MISSING_GROUP"

    nonpass = _complete_package()
    nonpass["groups"]["P0.4"]["declared_status"] = "UNKNOWN_NOT_ASSERTED"
    result = runner.evaluate_zero_step_p0(nonpass, candidate_config=config)
    statuses = {item["gate_id"]: item["status"] for item in result["gate_statuses"]}
    assert statuses["P0.4"] == "UNKNOWN_NOT_ASSERTED"

    unexpected = _complete_package()
    unexpected["groups"]["P0.12"] = {"declared_status": "PASS", "evidence": {}}
    result = runner.evaluate_zero_step_p0(unexpected, candidate_config=config)
    assert {item["status"] for item in result["gate_statuses"]} == {
        "FAIL_CLOSED_UNEXPECTED_GROUP_SCOPE"
    }


def test_failure_result_cannot_echo_member_or_execution_payload() -> None:
    runner = _load_candidate()
    config = runner.load_candidate_config()
    package = _complete_package()
    package["groups"]["P0.5"]["evidence"]["sequence"] = "payload-must-not-be-read-or-echoed"
    result = runner.evaluate_zero_step_p0(package, candidate_config=config)
    rendered = repr(result)
    assert "payload-must-not-be-read-or-echoed" not in rendered
    assert "sequence" not in rendered
    assert "endpoint" not in rendered.lower()
    statuses = {item["gate_id"]: item["status"] for item in result["gate_statuses"]}
    assert statuses["P0.5"] == "FAIL_CLOSED_PARTIAL_GROUP"


def test_preexisting_nonzero_touchpoint_aborts_before_metadata_evaluation() -> None:
    runner = _load_candidate()
    config = runner.load_candidate_config()
    sentinel = runner.ForbiddenTouchpointSentinel()
    sentinel.counts["CUDA_PROBE"] = 1
    with pytest.raises(runner.ForbiddenTouchpointError, match="nonzero"):
        runner.evaluate_zero_step_p0(
            _complete_package(),
            candidate_config=config,
            touchpoints=sentinel,
        )


def test_boolean_cannot_masquerade_as_a_numeric_run_or_count_binding() -> None:
    runner = _load_candidate()
    config = runner.load_candidate_config()
    package = _complete_package()
    package["groups"]["P0.8"]["evidence"]["run_count"] = True
    result = runner.evaluate_zero_step_p0(package, candidate_config=config)
    statuses = {item["gate_id"]: item["status"] for item in result["gate_statuses"]}
    assert statuses["P0.8"] == "FAIL_CLOSED_SEMANTIC_MISMATCH"


def test_source_has_no_data_cuda_model_optimizer_checkpoint_or_training_surface() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "subprocess",
        "typing",
    }
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden in (
        ".write_text(",
        ".write_bytes(",
        "socket",
        "urllib",
        "requests",
        "import torch",
        "torch.cuda",
        "pickle",
        "os.system",
    ):
        assert forbidden not in source


def test_mutating_active_authority_or_locks_invalidates_protocol() -> None:
    runner = _load_candidate()
    config = runner.load_candidate_config()
    authority_mutation = deepcopy(config)
    authority_mutation["authority"]["decision_status"] = "NOT_GRANTED"
    with pytest.raises(runner.CandidateContractError, match="authority"):
        runner.evaluate_zero_step_p0(
            _complete_package(), candidate_config=authority_mutation
        )

    lock_mutation = deepcopy(config)
    lock_mutation["retained_locks"]["training_allowed"] = True
    with pytest.raises(runner.CandidateContractError, match="locks"):
        runner.evaluate_zero_step_p0(_complete_package(), candidate_config=lock_mutation)
