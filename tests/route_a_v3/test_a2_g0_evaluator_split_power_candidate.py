from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    STAGING_ROOT / "scripts/route_a_v3/a2_g0_evaluator_split_power_candidate.py"
)


def _load():
    name = "route_a_v3_a2_g0_candidate_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _structural_fixture() -> list[dict]:
    return [
        {
            "record_key": "synthetic_r1",
            "source_group": "synthetic_s1",
            "known_duplicate_record_keys": [],
        },
        {
            "record_key": "synthetic_r2",
            "source_group": "synthetic_s1",
            "known_duplicate_record_keys": [],
        },
        {
            "record_key": "synthetic_r3",
            "source_group": "synthetic_s2",
            "known_duplicate_record_keys": ["synthetic_r4"],
        },
        {
            "record_key": "synthetic_r4",
            "source_group": "synthetic_s3",
            "known_duplicate_record_keys": [],
        },
        {
            "record_key": "synthetic_r5",
            "source_group": "synthetic_s4",
            "known_duplicate_record_keys": [],
        },
        {
            "record_key": "synthetic_r6",
            "source_group": "synthetic_s5",
            "known_duplicate_record_keys": [],
        },
        {
            "record_key": "synthetic_r7",
            "source_group": "synthetic_s6",
            "known_duplicate_record_keys": [],
        },
        {
            "record_key": "synthetic_r8",
            "source_group": "synthetic_s7",
            "known_duplicate_record_keys": [],
        },
    ]


def _endpoint_manifest(module) -> dict:
    return {
        "schema_version": "route_a_v3_a2_g0_endpoint_effect_se_manifest.v1",
        "input_scope": module.SYNTHETIC_SCOPE,
        "endpoint_name": "synthetic_activity",
        "endpoint_scale": "synthetic_log2_units",
        "endpoint_transform": "LOG2",
        "endpoint_direction": "HIGHER_IS_BETTER",
        "direction_multiplier": 1,
        "effect_definition": (
            "DIRECTION_NORMALIZED_CANDIDATE_MINUS_SOURCE_ON_TRANSFORMED_ENDPOINT_SCALE"
        ),
        "effect_analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
        "standard_error_estimator": "BIOLOGICAL_REPLICATE_MEAN_STANDARD_ERROR",
        "standard_error_analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
        "independent_replicate_unit": "BIOLOGICAL_REPLICATE",
        "minimum_independent_biological_replicates": 3,
        "technical_replicates_may_count_as_biological": False,
        "missing_policy": "EXCLUDE_ONLY_BY_PREFROZEN_RULE_NEVER_IMPUTE_ZERO",
        "nonfinite_policy": "EXCLUDE_ONLY_BY_PREFROZEN_RULE_NEVER_IMPUTE_ZERO",
        "censoring_or_selection_rule_prefrozen": True,
    }


def _effect_rows() -> list[dict]:
    return [
        {
            "source_group_key": f"synthetic_group_{index}",
            "predicted_direction_normalized_effect": float(index),
            "observed_direction_normalized_effect": float(index),
            "observed_standard_error": 0.1 + index / 100.0,
        }
        for index in range(1, 6)
    ]


def test_candidate_config_is_inactive_non_authoritative_and_preserves_state() -> None:
    module = _load()
    config = module.load_candidate_config()
    assert config["document_status"] == "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL"
    assert config["authority_status"] == "NON_AUTHORITATIVE"
    assert config["activation_state"] == "INACTIVE_G0_IMPLEMENTATION_CANDIDATE"
    assert config["governance_boundary"]["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert config["governance_boundary"]["changes_current_qualified_counts"] is False
    assert config["governance_boundary"]["final_a2_membership_frozen"] is False


def test_validate_only_reads_zero_rows_writes_no_artifacts_and_reproduces_n156(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load()
    monkeypatch.chdir(tmp_path)
    assert module.main(["--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == module.CONFIG_PASS
    assert report["mode"] == "VALIDATE_ONLY_ZERO_PROJECT_ROW_IO_NO_RUNTIME_ARTIFACTS"
    assert set(report["validate_only_truth"].values()) == {0}
    planning = report["power_precision_planning"]
    assert planning["required_effective_n"] == 156
    assert planning["n_155_both_thresholds_pass"] is False
    assert planning["n_156_both_thresholds_pass"] is True
    assert planning["n_156_estimated_design_power"] >= 0.80
    assert planning["n_156_planned_full_confidence_interval_width"] <= 0.30
    assert report["scientific_gate_status"] == "NOT_RUN"
    assert list(tmp_path.iterdir()) == []


def test_validate_only_is_the_only_command_line_mode() -> None:
    module = _load()
    with pytest.raises(module.ContractError, match="--validate-only only"):
        module.main([])


def test_candidate_rejects_added_execution_permission_or_hidden_nested_authority() -> None:
    module = _load()
    config = module.load_candidate_config()
    extra_permission = deepcopy(config)
    extra_permission["permitted_g0_operations"].append("READ_REAL_A2_ROWS")
    with pytest.raises(module.ContractError, match="permitted G0"):
        module.validate_candidate_config(extra_permission)
    hidden_authority = deepcopy(config)
    hidden_authority["governance_boundary"]["training_allowed"] = True
    with pytest.raises(module.ContractError, match="governance boundary keys"):
        module.validate_candidate_config(hidden_authority)


def test_bonett_wright_boundary_is_first_pass_at_156_independent_groups() -> None:
    module = _load()
    kwargs = {
        "alternative_spearman_rho": 0.25,
        "two_sided_alpha": 0.05,
        "confidence_level": 0.95,
        "target_power": 0.80,
        "maximum_full_ci_width": 0.30,
    }
    plan_155 = module.bonett_wright_fisher_z_plan(155, **kwargs)
    plan_156 = module.bonett_wright_fisher_z_plan(156, **kwargs)
    assert plan_155["power_pass"] is True
    assert plan_155["precision_pass"] is False
    assert plan_156["power_pass"] is True
    assert plan_156["precision_pass"] is True
    assert module.minimum_effective_n_for_power_and_precision(**kwargs) == 156
    assert plan_156["analysis_unit"] == "POST_DEDUP_INDEPENDENT_SOURCE_GROUP"
    assert plan_156["formal_qualification_gate_executed"] is False


@pytest.mark.parametrize("bad_n", [True, 3, 3.5])
def test_power_plan_rejects_invalid_effective_n(bad_n) -> None:
    module = _load()
    with pytest.raises(module.ContractError, match="effective N"):
        module.bonett_wright_fisher_z_plan(
            bad_n,
            alternative_spearman_rho=0.25,
            two_sided_alpha=0.05,
            confidence_level=0.95,
            target_power=0.80,
            maximum_full_ci_width=0.30,
        )


def test_structural_graph_merges_source_and_duplicate_edges_without_keys_in_output() -> None:
    module = _load()
    config = module.load_candidate_config()
    summary = module.build_structural_graph_summary(
        _structural_fixture(), scope_token=module.SYNTHETIC_SCOPE, config=config
    )
    assert summary["status"] == module.SYNTHETIC_PASS
    assert summary["record_count"] == 8
    assert summary["source_group_count"] == 7
    assert summary["known_duplicate_edge_count"] == 1
    assert summary["connected_component_count"] == 6
    assert summary["component_size_histogram"] == {"1": 4, "2": 2}
    assert summary["outcome_fields_read"] == 0
    assert summary["record_or_component_keys_included"] is False
    serialized = json.dumps(summary)
    assert "synthetic_r" not in serialized
    assert "synthetic_s" not in serialized


def test_outcome_blind_split_is_component_disjoint_and_aggregate_only() -> None:
    module = _load()
    config = module.load_candidate_config()
    plan = module.generate_outcome_blind_split_plan(
        _structural_fixture(), scope_token=module.SYNTHETIC_SCOPE, config=config
    )
    assert plan["status"] == module.SYNTHETIC_PASS
    assert plan["outcome_blind"] is True
    assert plan["record_count"] == 8
    assert len(plan["fold_aggregate_counts"]) == 5
    assert all(item["record_count"] > 0 for item in plan["fold_aggregate_counts"])
    assert sum(item["record_count"] for item in plan["fold_aggregate_counts"]) == 8
    assert sum(item["component_count"] for item in plan["fold_aggregate_counts"]) == 6
    assert plan["source_group_cross_fold_count"] == 0
    assert plan["known_duplicate_cross_fold_count"] == 0
    assert plan["component_cross_fold_count"] == 0
    assert plan["record_or_component_keys_included"] is False
    assert plan["split_assignments_included"] is False
    assert plan["synthetic_salt_is_final_a2_salt"] is False
    assert plan["final_a2_membership_status"] == "NOT_FROZEN"
    serialized = json.dumps(plan)
    assert "synthetic_r1" not in serialized
    assert "synthetic_s1" not in serialized


def test_split_structural_input_rejects_outcomes_and_unknown_duplicate_references() -> None:
    module = _load()
    config = module.load_candidate_config()
    with_outcome = deepcopy(_structural_fixture())
    with_outcome[0]["effect"] = 0.5
    with pytest.raises(module.ContractError, match="keys differ"):
        module.generate_outcome_blind_split_plan(
            with_outcome, scope_token=module.SYNTHETIC_SCOPE, config=config
        )
    unknown_duplicate = deepcopy(_structural_fixture())
    unknown_duplicate[0]["known_duplicate_record_keys"] = ["missing_record"]
    with pytest.raises(module.ContractError, match="not present"):
        module.generate_outcome_blind_split_plan(
            unknown_duplicate, scope_token=module.SYNTHETIC_SCOPE, config=config
        )
    real_looking_key = deepcopy(_structural_fixture())
    real_looking_key[0]["record_key"] = "GSE_REAL_MEMBER"
    with pytest.raises(module.ContractError, match="synthetic prefix"):
        module.generate_outcome_blind_split_plan(
            real_looking_key, scope_token=module.SYNTHETIC_SCOPE, config=config
        )


def test_split_fails_when_indivisible_components_cannot_fill_all_folds() -> None:
    module = _load()
    config = module.load_candidate_config()
    records = _structural_fixture()[:4]
    with pytest.raises(module.ContractError, match="fewer connected components"):
        module.generate_outcome_blind_split_plan(
            records, scope_token=module.SYNTHETIC_SCOPE, config=config
        )


def test_endpoint_effect_se_manifest_closes_direction_biological_se_and_missingness() -> None:
    module = _load()
    config = module.load_candidate_config()
    result = module.validate_endpoint_effect_se_manifest(
        _endpoint_manifest(module),
        scope_token=module.SYNTHETIC_SCOPE,
        config=config,
    )
    assert result == {
        "status": module.SYNTHETIC_PASS,
        "endpoint_direction_and_transform_closed": True,
        "effect_definition_closed": True,
        "biological_standard_error_contract_closed": True,
        "missing_nonfinite_and_censoring_policy_closed": True,
        "endpoint_or_effect_values_read": 0,
        "member_identifiers_included": False,
        "scientific_gate_status": "NOT_RUN",
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda item: item.update(endpoint_direction="LOWER_IS_BETTER"), "multiplier"),
        (
            lambda item: item.update(minimum_independent_biological_replicates=2),
            "fewer than three",
        ),
        (
            lambda item: item.update(technical_replicates_may_count_as_biological=True),
            "technical replicates",
        ),
        (lambda item: item.update(censoring_or_selection_rule_prefrozen=False), "not prefrozen"),
        (lambda item: item.update(missing_policy="MISSING_IS_ZERO"), "missing-value"),
    ],
)
def test_endpoint_effect_se_manifest_fails_closed(mutator, message: str) -> None:
    module = _load()
    config = module.load_candidate_config()
    manifest = _endpoint_manifest(module)
    mutator(manifest)
    with pytest.raises(module.ContractError, match=message):
        module.validate_endpoint_effect_se_manifest(
            manifest, scope_token=module.SYNTHETIC_SCOPE, config=config
        )


def test_synthetic_evaluator_returns_only_aggregate_metric_values() -> None:
    module = _load()
    config = module.load_candidate_config()
    result = module.evaluate_synthetic_effects(
        _effect_rows(),
        _endpoint_manifest(module),
        scope_token=module.SYNTHETIC_SCOPE,
        config=config,
    )
    assert result["status"] == module.SYNTHETIC_PASS
    assert result["independent_source_group_count"] == 5
    assert result["primary_metric"] == "WITHIN_STUDY_SPEARMAN"
    assert result["within_study_spearman"] == pytest.approx(1.0)
    assert result["mean_absolute_effect_error"] == pytest.approx(0.0)
    assert result["mean_observed_standard_error"] == pytest.approx(0.13)
    assert result["one_source_group_one_vote"] is True
    assert result["member_or_source_group_keys_included"] is False
    assert result["row_level_effects_or_standard_errors_included"] is False
    assert result["model_or_checkpoint_selection_allowed"] is False
    assert result["formal_qualification_or_power_gate_executed"] is False
    serialized = json.dumps(result)
    assert "synthetic_group" not in serialized


@pytest.mark.parametrize(
    ("mutator", "exception_name", "message"),
    [
        (
            lambda rows: rows[1].update(source_group_key=rows[0]["source_group_key"]),
            "ContractError",
            "duplicated",
        ),
        (
            lambda rows: rows[0].update(observed_standard_error=0.0),
            "ContractError",
            "strictly positive",
        ),
        (
            lambda rows: rows[0].update(observed_direction_normalized_effect=math.nan),
            "ContractError",
            "nonfinite",
        ),
        (
            lambda rows: [
                row.update(predicted_direction_normalized_effect=1.0) for row in rows
            ],
            "MetricUndefinedError",
            "undefined",
        ),
    ],
)
def test_synthetic_evaluator_fails_closed(mutator, exception_name: str, message: str) -> None:
    module = _load()
    config = module.load_candidate_config()
    rows = _effect_rows()
    mutator(rows)
    exception = getattr(module, exception_name)
    with pytest.raises(exception, match=message):
        module.evaluate_synthetic_effects(
            rows,
            _endpoint_manifest(module),
            scope_token=module.SYNTHETIC_SCOPE,
            config=config,
        )


@pytest.mark.parametrize(
    "function_name",
    [
        "build_structural_graph_summary",
        "generate_outcome_blind_split_plan",
    ],
)
def test_structural_helpers_reject_any_non_synthetic_scope(function_name: str) -> None:
    module = _load()
    config = module.load_candidate_config()
    function = getattr(module, function_name)
    with pytest.raises(module.ContractError, match="synthetic-test-fixture"):
        function(_structural_fixture(), scope_token="REAL_DATA", config=config)


def test_source_has_no_data_training_gpu_network_or_file_write_surface() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "argparse",
        "collections",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "statistics",
        "typing",
    }
    assert ".write_text(" not in source
    assert ".write_bytes(" not in source
    assert "subprocess" not in imported_roots
    assert "socket" not in imported_roots
    assert "requests" not in imported_roots
    assert "torch" not in imported_roots
    assert "pandas" not in imported_roots
    assert "numpy" not in imported_roots
