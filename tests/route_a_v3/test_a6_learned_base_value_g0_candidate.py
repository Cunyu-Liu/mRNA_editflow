from __future__ import annotations

import ast
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/a6_learned_base_value_g0_candidate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runner():
    return _load(SCRIPT_PATH, "a6_learned_base_value_g0_candidate_test")


def _valid_input_manifest(protocol: dict) -> dict:
    return {
        "schema_version": "route_a_v3_a6_g0_aggregate_input_contract_manifest.v1",
        "aggregate_only": True,
        "contains_member_payload": False,
        "data_scope": protocol["ordinary_public_data_contract"]["data_scope"],
        "record_role": protocol["ordinary_public_data_contract"]["allowed_record_role"],
        "qualification_status": "QUALIFIED_UNDER_FUTURE_ACTIVE_AUTHORITY",
        "declared_fields": protocol["ordinary_public_data_contract"]["allowed_fields"],
        "forbidden_roles_present": [],
        "split": {
            "parent_split_authority": protocol["split_contract"]["parent_split_authority"],
            "frozen": True,
            "label_blind": True,
            "components_indivisible": True,
            "assignment_unit": protocol["split_contract"]["assignment_unit"],
            "development_subroles": protocol["split_contract"]["development_subroles"],
            "leakage_counts": {
                "source_group": 0,
                "exact_sequence": 0,
                "near_duplicate": 0,
                "reverse_edge": 0,
                "candidate": 0,
                "study_context": 0,
            },
            "retry_after_labels_or_results": False,
            "outer_test_label_accessed": False,
        },
        "rights": {
            "qualification_use_authorized": True,
            "private_processing_and_evaluation_authorized": True,
            "rights_status": "PASS_UNDER_FUTURE_ACTIVE_AUTHORITY",
            "raw_or_member_level_redistribution_allowed": False,
        },
        "exposure": {
            "model_input_route": "SCRATCH_ONLY_RANDOM_INITIALIZATION",
            "pretrained_foundation_checkpoints": [],
            "pretrained_weights": [],
            "warm_start_checkpoints": [],
            "external_learned_embeddings": [],
            "external_pretraining_corpora": [],
            "checkpoint_loads_before_first_update": 0,
        },
    }


def test_parent_draft_remains_static_only_and_candidate_is_non_authoritative() -> None:
    runner = _runner()
    protocol = runner.load_protocol()
    candidate = runner.load_candidate_config()
    assert protocol["document_status"] == "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL"
    assert protocol["authority_status"] == "NON_AUTHORITATIVE"
    assert protocol["implementation_scope"] == "PROTOCOL_SCHEMA_STATIC_VALIDATOR_AND_FOCUSED_TEST_ONLY"
    assert protocol["forbidden_state_changes"]["training_allowed"] is False
    assert candidate["activation_state"] == "INACTIVE_IMPLEMENTATION_CANDIDATE"
    assert candidate["runtime_truth"]["parameter_updates"] == 0
    assert candidate["runtime_truth"]["gpu_runs"] == 0


def test_validate_only_builds_shape_and_manifest_plans_with_zero_side_effects(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    runner = _runner()
    monkeypatch.chdir(tmp_path)
    assert runner.main(["--validate-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "VALIDATE_ONLY_ZERO_UPDATE_DRY_RUN"
    assert result["architecture_plan"]["state_vector_width"] == 154
    assert result["architecture_plan"]["action_vector_width"] == 225
    assert result["architecture_plan"]["parameter_tensors_constructed"] == 0
    assert result["future_manifest_plan"]["files_created"] == 0
    assert result["kernel_loaded"] is False
    assert result["torch_imported"] is False
    assert result["cuda_probe_invoked"] is False
    assert set(result["audit"].values()) == {0}
    assert list(tmp_path.iterdir()) == []


def test_aggregate_input_contract_interface_accepts_only_closed_metadata() -> None:
    runner = _runner()
    protocol = runner.load_protocol()
    result = runner.validate_aggregate_input_contract_manifest(_valid_input_manifest(protocol), protocol)
    assert result == {
        "status": "PASS_INTERFACE_METADATA_ONLY_NOT_DATA_QUALIFICATION",
        "member_payload_reads": 0,
        "ordinary_row_reads": 0,
        "private_row_reads": 0,
        "sealed_row_reads": 0,
    }


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda value: value.update(contains_member_payload=True), "member payload"),
        (lambda value: value.update(record_role="OUTER_TEST"), "record role"),
        (lambda value: value.update(qualification_status="UNKNOWN"), "qualification"),
        (lambda value: value["split"]["leakage_counts"].update(source_group=1), "leakage"),
        (lambda value: value["rights"].update(private_processing_and_evaluation_authorized=False), "rights"),
        (lambda value: value["exposure"]["pretrained_weights"].append("forbidden.ckpt"), "exposure"),
    ],
)
def test_input_role_split_rights_and_exposure_fail_closed(mutator, expected: str) -> None:
    runner = _runner()
    protocol = runner.load_protocol()
    manifest = deepcopy(_valid_input_manifest(protocol))
    mutator(manifest)
    with pytest.raises(runner.ContractError, match=expected):
        runner.validate_aggregate_input_contract_manifest(manifest, protocol)


def test_existing_kernel_is_reused_for_source_anchor_stop_alias_support_and_budget() -> None:
    runner = _runner()
    protocol = runner.load_protocol()
    candidate = runner.load_candidate_config()
    dependency = candidate["existing_kernel_dependency"]
    kernel = _load(
        STAGING_ROOT / dependency["script_path_from_repo"],
        "a6_existing_cpu_exact_kernel_for_g0_test",
    )
    kernel_config = kernel.load_config(STAGING_ROOT / dependency["config_path_from_repo"])
    case = next(item for item in kernel_config["fixed_cases"] if item["case_id"] == "L2_B2")
    state = kernel.initial_state(case, kernel_config)
    plans = runner.canonical_action_plans(kernel, state, kernel_config, protocol)
    assert plans
    assert len({plan.next_state for plan in plans}) == len(plans)
    assert {plan.action_type for plan in plans} == {"SOURCE_BASE_TO_ALT_BASE", "STOP"}
    assert all(plan.support_floor == 1e-8 for plan in plans)
    assert all(len(plan.raw_alias_ids) >= 2 for plan in plans)
    for plan in plans:
        child = plan.next_state
        assert child.source_sequence == state.source_sequence
        if plan.action_type == "STOP":
            assert child.source_relative_edit_set == state.source_relative_edit_set
        else:
            assert child.net_edit_count == state.net_edit_count + 1
            assert child.remaining_budget == state.remaining_budget - 1
    assert [runner.assign_primary_budget(value, protocol) for value in (0, 1, 2, 3, 4, 5)] == [1, 1, 3, 3, 5, 5]
    with pytest.raises(runner.ContractError, match="exceeds"):
        runner.assign_primary_budget(6, protocol)


def test_pure_rate_terminal_and_guided_interfaces_match_frozen_formulas() -> None:
    runner = _runner()
    protocol = runner.load_protocol()
    base = runner.normalized_base_rates({"edit": -100.0, "stop": 0.0}, protocol)
    assert set(base) == {"edit", "stop"}
    assert all(value > 0.0 for value in base.values())
    assert sum(base.values()) == pytest.approx(1.0, abs=1e-12)
    weight, terminal_potential = runner.terminal_boundary(9.0, protocol)
    assert weight == pytest.approx(54.598150033144236)
    assert terminal_potential == pytest.approx(4.0)
    guided = runner.guided_generator_plan(base, 0.25, {"edit": 0.5, "stop": terminal_potential})
    assert guided.diagonal == pytest.approx(-guided.total_exit_hazard)
    assert all(rate > 0.0 for _, rate in guided.off_diagonal)


@pytest.mark.parametrize(
    "operation",
    ["request_training", "request_optimizer_step", "request_checkpoint_write", "cuda_ownership_preflight"],
)
def test_execution_and_cuda_requests_fail_before_callback_and_keep_zero(operation: str) -> None:
    runner = _runner()
    protocol = runner.load_protocol()
    calls = []

    def forbidden_callback():
        calls.append("CALLED")
        raise AssertionError("inactive G0 crossed the authority barrier")

    function = getattr(runner, operation)
    with pytest.raises(runner.InactiveAuthorityError, match="parameter_updates=0; gpu_runs=0; outputs=0"):
        function(protocol, runner.ExecutionAuthority(), forbidden_callback)
    assert calls == []
    runner.ZeroUpdateAudit().assert_zero()


def test_candidate_has_no_torch_subprocess_network_or_file_write_imports() -> None:
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
        "importlib",
        "json",
        "math",
        "pathlib",
        "sys",
        "types",
        "typing",
    }
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert ".write_text(" not in source
    assert ".write_bytes(" not in source
    assert "torch" not in imported_roots
