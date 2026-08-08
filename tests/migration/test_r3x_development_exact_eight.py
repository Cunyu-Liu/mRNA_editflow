"""Adversarial tests for the independent development exact-eight orchestrator."""

from __future__ import annotations

import ast
import copy
import hashlib
import os
import stat
from pathlib import Path

import pytest

import scripts.r3x.development_exact_eight as exact_module
import scripts.r3x.scalar_fitted_state as scalar_module
import tests.migration.test_r3x_development_run_plan as plan_test
from scripts.r3x.artifact_io import (
    atomic_write_json_exclusive,
    canonical_json_bytes,
)
from scripts.r3x.development_run_plan import (
    FROZEN_BASELINE_METHOD_ORDER,
    ORDINARY_DEVELOPMENT,
    ORDINARY_TRAIN,
    write_development_run_plan_exclusive,
)
from scripts.r3x.information_sets import information_set_fingerprint
from scripts.r3x.neural_method_registry import registry_sha256
from scripts.r3x.predictor_provenance import (
    NON_DATA_AUTHORITY_HASH_KEYS,
    PREDICTOR_INPUT_ROW_SCHEMA,
    write_predictor_input_jsonl_exclusive,
)
from scripts.r3x.scalar_fitted_state import (
    METHOD_INFORMATION_SETS,
    build_train_source_component_assignment,
    validate_train_source_component_assignment,
    write_train_source_component_assignment_exclusive,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _ref(root: Path, path: Path) -> dict[str, str]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _predictor_row(inventory_row: dict, index: int) -> dict:
    source = "ACGUACGUACGU"
    position = index % len(source)
    alt = {"A": "C", "C": "G", "G": "U", "U": "A"}[source[position]]
    candidate = source[:position] + alt + source[position + 1 :]
    target = None
    if inventory_row["partition"] == ORDINARY_TRAIN:
        target = float(((index * 7) % 19) - 9) / 3.0
    return {
        "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
        "row_id": inventory_row["row_id"],
        "partition": inventory_row["partition"],
        "estimand_id": inventory_row["estimand_id"],
        "cell_id": f"CELL_{inventory_row['study']}",
        "predictor_input": {
            "endpoint": inventory_row["endpoint"],
            "publication_context_id": f"CONTEXT_{index % 3}",
            "region": "5UTR",
            "normalized_source_sequence": source,
            "normalized_candidate_sequence": candidate,
            "edit_list": [
                {
                    "op": "SUB",
                    "pos": position,
                    "ref": source[position],
                    "alt": alt,
                }
            ],
            "edit_count": 1,
        },
        "training_target": target,
    }


def _authorities(plan: dict) -> dict[str, str]:
    values = {
        key: _sha(f"caller-prebound:{key}")
        for key in NON_DATA_AUTHORITY_HASH_KEYS
    }
    source = plan["authority_source_sha256"]
    values.update(
        {
            "inclusion_manifest_sha256": source["inclusion_manifest"],
            "context_source_crosswalk_sha256": source[
                "context_source_crosswalk"
            ],
            "endpoint_estimand_registry_sha256": source[
                "endpoint_estimand_registry"
            ],
            "source_component_split_manifest_sha256": source[
                "source_component_split_manifest"
            ],
            "model_optimizer_seed_config_sha256": source[
                "executable_config"
            ],
            "analysis_code_tree_manifest_sha256": plan["executable_config"][
                "code_tree_sha256"
            ],
            "paired_strongest_baseline_comparison_policy_sha256": source[
                "paired_baseline_policy"
            ],
        }
    )
    return values


@pytest.fixture
def exact_bundle(tmp_path: Path) -> dict:
    prepared = plan_test.prepared.__wrapped__(tmp_path)
    executable_config = copy.deepcopy(prepared.payloads["executable_config"])
    executable_config["model"].update({
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 1,
        "ff_dim": 32,
        "max_len": 16,
        "n_endpoints": 3,
        "n_contexts": 4,
        "pair_rank": 2,
    })
    executable_config["training"].update({
        "epochs": 1,
        "batch_size": 64,
    })
    prepared.replace("executable_config", executable_config)
    root = prepared.root
    for relative in (
        "development/predictor_inputs",
        "development/targets",
        "development/assignments",
        "development/exact_eight/fixture-run",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    plan = prepared.build()
    plan_path = root / "development/run_plans/exact-eight-plan.json"
    plan_sha = write_development_run_plan_exclusive(root, plan_path, plan)

    inventory = prepared.payloads["row_inventory"]
    rows = [
        _predictor_row(item, index)
        for index, item in enumerate(inventory["rows"])
    ]
    train_rows = [row for row in rows if row["partition"] == ORDINARY_TRAIN]
    development_rows = [
        row for row in rows if row["partition"] == ORDINARY_DEVELOPMENT
    ]
    train_path = root / "development/predictor_inputs/train-rows.jsonl"
    development_path = (
        root / "development/predictor_inputs/development-rows.jsonl"
    )
    train_sha = write_predictor_input_jsonl_exclusive(train_path, train_rows)
    development_sha = write_predictor_input_jsonl_exclusive(
        development_path, development_rows
    )
    target_rows = [
        {
            "row_id": row["row_id"],
            "estimand_id": row["estimand_id"],
            "target": (
                (-1.0 if index % 2 else 1.0) * (1.0 + index % 5) / 5.0
            ),
        }
        for index, row in enumerate(development_rows)
    ]
    targets_payload = {
        "schema_version": exact_module.DEVELOPMENT_TARGETS_SCHEMA,
        "artifact_role": "ORDINARY_DEVELOPMENT_EXACT_TARGETS",
        "status": "COMPLETE_EXACT_DEVELOPMENT_TARGET_BINDING",
        "train_development_outcomes_sha256": plan[
            "train_development_outcomes_sha256"
        ],
        "development_projection_hashes": copy.deepcopy(
            plan["development_projection_hashes"]
        ),
        "partition": ORDINARY_DEVELOPMENT,
        "partition_row_set_sha256": plan["partition_row_set_sha256"][
            ORDINARY_DEVELOPMENT
        ],
        "targets": target_rows,
        "targets_sha256": plan_test.canonical_sha256(target_rows),
        "ordinary_only": True,
        "protected_partition_access": False,
        "future_stage_access": False,
        "execution_authorized": False,
        "publication_evidence": False,
    }
    targets_path = root / "development/targets/development-targets.json"
    targets_sha = atomic_write_json_exclusive(
        targets_path,
        targets_payload,
        validator=exact_module.validate_development_targets,
    )

    authorities = _authorities(plan)
    component_by_id = {
        item["row_id"]: item["source_component_id"]
        for item in inventory["rows"]
        if item["partition"] == ORDINARY_TRAIN
    }
    assignment_payload = build_train_source_component_assignment(
        train_rows,
        [
            {
                "row_id": row["row_id"],
                "source_component_id": component_by_id[row["row_id"]],
            }
            for row in reversed(train_rows)
        ],
        train_development_outcomes_sha256=plan[
            "train_development_outcomes_sha256"
        ],
        non_data_authority_hashes=authorities,
        source_component_split_manifest_source_sha256=plan[
            "authority_source_sha256"
        ]["source_component_split_manifest"],
        source_component_split_manifest_projection_sha256=plan[
            "development_projection_hashes"
        ]["source_component_split_manifest_development_projection"],
    )
    assignment_path = root / "development/assignments/shared-v2.json"
    assignment_sha = write_train_source_component_assignment_exclusive(
        assignment_path,
        assignment_payload,
        expected_train_rows=train_rows,
        expected_train_development_outcomes_sha256=plan[
            "train_development_outcomes_sha256"
        ],
        expected_non_data_authority_hashes=authorities,
        expected_source_component_split_manifest_source_sha256=plan[
            "authority_source_sha256"
        ]["source_component_split_manifest"],
        expected_source_component_split_manifest_projection_sha256=plan[
            "development_projection_hashes"
        ]["source_component_split_manifest_development_projection"],
    )
    request = {
        "schema_version": exact_module.REQUEST_SCHEMA,
        "run_id": "fixture-run",
        "run_plan_ref": {
            "relative_path": plan_path.relative_to(root).as_posix(),
            "sha256": plan_sha,
        },
        "train_predictor_input_ref": {
            "relative_path": train_path.relative_to(root).as_posix(),
            "sha256": train_sha,
        },
        "development_predictor_input_ref": {
            "relative_path": development_path.relative_to(root).as_posix(),
            "sha256": development_sha,
        },
        "development_targets_ref": {
            "relative_path": targets_path.relative_to(root).as_posix(),
            "sha256": targets_sha,
        },
        "shared_component_assignment_ref": {
            "relative_path": assignment_path.relative_to(root).as_posix(),
            "sha256": assignment_sha,
        },
        "expected_non_data_authority_hashes": authorities,
        "expected_neural_registry_sha256": registry_sha256(),
        "output_relative_dir": "development/exact_eight/fixture-run",
        "ordinary_only": True,
        "primary_sign_gate": 0.60,
        "sign_gate_change_authorized": False,
    }
    return {
        "root": root,
        "prepared": prepared,
        "plan": plan,
        "train_rows": train_rows,
        "development_rows": development_rows,
        "targets": targets_payload,
        "targets_path": targets_path,
        "assignment": assignment_payload,
        "assignment_path": assignment_path,
        "request": request,
    }


def _assert_hold(callable_, code: str) -> exact_module.DevelopmentExactEightError:
    with pytest.raises(exact_module.DevelopmentExactEightError) as caught:
        callable_()
    assert caught.value.code == code
    return caught.value


def test_module_is_independent_of_historical_runner_and_protected_helpers() -> None:
    tree = ast.parse(Path(exact_module.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.endswith("run_ordinary") for name in imports)
    assert not any("internal_test" in name or "sealed" in name for name in imports)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.__setitem__(
            "sealed_ref", {"relative_path": "never/read", "sha256": "a" * 64}
        ),
        lambda request: request["train_predictor_input_ref"].__setitem__(
            "relative_path", "development/predictor_inputs/outcome-rows.jsonl"
        ),
        lambda request: request.__setitem__(
            "internal\u200b_test_ref", {"relative_path": "never/read", "sha256": "a" * 64}
        ),
    ],
)
def test_protected_request_or_path_fails_before_first_content_read(
    exact_bundle: dict, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    request = copy.deepcopy(exact_bundle["request"])
    mutation(request)
    reads = 0

    def forbidden_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("content read happened before protected request rejection")

    monkeypatch.setattr(exact_module, "load_development_run_plan_exact", forbidden_read)
    _assert_hold(
        lambda: exact_module.execute_development_exact_eight(
            exact_bundle["root"], request
        ),
        "HOLD_DEVELOPMENT_EXACT_EIGHT_PROTECTED_REQUEST",
    )
    assert reads == 0


def test_execute_is_exact_ordered_partial_and_full_replayable(
    exact_bundle: dict,
) -> None:
    result, result_ref = exact_module.execute_development_exact_eight(
        exact_bundle["root"], exact_bundle["request"]
    )
    assert result["status"] == exact_module.PARTIAL_STATUS
    assert result["method_order"] == list(FROZEN_BASELINE_METHOD_ORDER)
    assert result["method_order"] != sorted(result["method_order"])
    assert result["completion"] == {
        "complete_method_count": 7,
        "held_method_count": 1,
        "exact_eight_complete": False,
    }
    assert result["primary_sign_gate"] == 0.60
    assert result["resource_budget_binding"]["executed_scalar_fit_units"] > 0
    assert result["resource_budget_binding"][
        "neural_forward_equivalents_per_scalar_fit"
    ] == 0
    assert result["resource_budget_binding"][
        "executed_neural_fit_units"
    ] == 10
    assert result["resource_budget_binding"][
        "neural_forward_equivalents"
    ] > 0
    status_by_method = {
        item["method_id"]: item["status"] for item in result["methods"]
    }
    assert status_by_method["HISTOGRAM_GRADIENT_BOOSTING"] == (
        exact_module.METHOD_HOLD_STATUS
    )
    assert status_by_method["FULL_PAIR_UNRESTRICTED_SIAMESE"] == (
        exact_module.NEURAL_COMPLETE_STATUS
    )
    assert all(
        status_by_method[method_id] == exact_module.SCALAR_COMPLETE_STATUS
        for method_id in scalar_module.SUPPORTED_SCALAR_METHOD_IDS
    )
    for method_id in FROZEN_BASELINE_METHOD_ORDER:
        expected_set = METHOD_INFORMATION_SETS.get(
            method_id, "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
        )
        assert result["information_set_fingerprints"][method_id] == (
            information_set_fingerprint(expected_set)
        )
    for record in result["methods"]:
        path = exact_bundle["root"] / record["evidence_ref"]["relative_path"]
        metadata = os.lstat(path)
        assert stat.S_ISREG(metadata.st_mode)
        assert metadata.st_nlink == 1
        evidence = exact_module._load_method_evidence_exact(
            exact_bundle["root"], record["evidence_ref"]
        )
        assert evidence["shared_component_assignment_ref"] == (
            result["request"]["shared_component_assignment_ref"]
        )
        if record["status"] == exact_module.SCALAR_COMPLETE_STATUS:
            assert all(
                "fitted_state_ref" in item and "prediction_ref" in item
                for item in evidence["estimands"]
            )
        elif record["status"] == exact_module.NEURAL_COMPLETE_STATUS:
            assert evidence["method_kind"] == "NEURAL"
            assert evidence["neural_registry_sha256"] == registry_sha256()
            assert evidence["resource_accounting"]["fit_units"] == 10
            assert all(
                len(item["seed_states"]) == 5
                and "derived_component_assignment_ref" in item
                and "ensemble_ref" in item
                and "prediction_ref" in item
                for item in evidence["estimands"]
            )
        else:
            assert record["method_id"] == "HISTOGRAM_GRADIENT_BOOSTING"
            assert all(
                "fitted_state_ref" not in item and "prediction_ref" not in item
                for item in evidence["estimands"]
            )
    replayed = exact_module.load_development_exact_eight_result_exact(
        exact_bundle["root"], result_ref
    )
    assert replayed == result


def test_complete_status_forgery_and_method_fingerprint_forgery_fail_closed(
    exact_bundle: dict,
) -> None:
    result, _ = exact_module.execute_development_exact_eight(
        exact_bundle["root"], exact_bundle["request"]
    )
    forged = copy.deepcopy(result)
    forged["status"] = exact_module.COMPLETE_STATUS
    forged["completion"]["exact_eight_complete"] = True
    _assert_hold(
        lambda: exact_module.validate_development_exact_eight_result(forged),
        "HOLD_DEVELOPMENT_EXACT_EIGHT_COMPLETION",
    )

    first = result["methods"][0]
    evidence = exact_module._load_method_evidence_exact(
        exact_bundle["root"], first["evidence_ref"]
    )
    evidence["information_set_fingerprint"] = "a" * 64
    _assert_hold(
        lambda: exact_module.validate_development_exact_eight_method_evidence(
            evidence
        ),
        "HOLD_DEVELOPMENT_EXACT_EIGHT_INFORMATION_SET",
    )


def test_row_set_mismatch_holds_before_any_method_artifact(
    exact_bundle: dict,
) -> None:
    root = exact_bundle["root"]
    output = root / "development/exact_eight/row-mismatch"
    output.mkdir()
    path = root / "development/predictor_inputs/development-missing-row.jsonl"
    digest = write_predictor_input_jsonl_exclusive(
        path, exact_bundle["development_rows"][:-1]
    )
    request = copy.deepcopy(exact_bundle["request"])
    request["run_id"] = "row-mismatch"
    request["output_relative_dir"] = "development/exact_eight/row-mismatch"
    request["development_predictor_input_ref"] = {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": digest,
    }
    _assert_hold(
        lambda: exact_module.execute_development_exact_eight(root, request),
        "HOLD_DEVELOPMENT_EXACT_EIGHT_ROW_SET",
    )
    assert list(output.iterdir()) == []


def test_assignment_cell_tamper_and_v1_assignment_hold_before_fit(
    exact_bundle: dict,
) -> None:
    root = exact_bundle["root"]
    for run_id, legacy in (("cell-tamper", False), ("legacy-v1", True)):
        output = root / f"development/exact_eight/{run_id}"
        output.mkdir()
        payload = copy.deepcopy(exact_bundle["assignment"])
        if legacy:
            payload["schema_version"] = "r3x_train_source_component_assignment_v1"
            payload["information_set_fingerprint"] = information_set_fingerprint(
                "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
            )
            payload.pop("assignment_scope")
            validator = None
        else:
            payload["assignments"][0]["cell_id"] = "FORGED_CELL"
            validator = validate_train_source_component_assignment
        path = root / f"development/assignments/{run_id}.json"
        digest = atomic_write_json_exclusive(path, payload, validator=validator)
        request = copy.deepcopy(exact_bundle["request"])
        request["run_id"] = run_id
        request["output_relative_dir"] = f"development/exact_eight/{run_id}"
        request["shared_component_assignment_ref"] = {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": digest,
        }
        _assert_hold(
            lambda: exact_module.execute_development_exact_eight(root, request),
            "HOLD_DEVELOPMENT_EXACT_EIGHT_ASSIGNMENT",
        )
        assert list(output.iterdir()) == []


def test_v3_run_plan_bytes_fail_closed_before_fit(exact_bundle: dict) -> None:
    root = exact_bundle["root"]
    output = root / "development/exact_eight/old-plan"
    output.mkdir()
    old = copy.deepcopy(exact_bundle["plan"])
    old["schema_version"] = "r3x_development_run_plan_v3"
    core = dict(old)
    core.pop("plan_sha256")
    old["plan_sha256"] = plan_test.canonical_sha256(core)
    path = root / "development/run_plans/old-v3.json"
    path.write_bytes(canonical_json_bytes(old))
    request = copy.deepcopy(exact_bundle["request"])
    request["run_id"] = "old-plan"
    request["output_relative_dir"] = "development/exact_eight/old-plan"
    request["run_plan_ref"] = _ref(root, path)
    _assert_hold(
        lambda: exact_module.execute_development_exact_eight(root, request),
        "HOLD_DEVELOPMENT_EXACT_EIGHT_RUN_PLAN",
    )
    assert list(output.iterdir()) == []


def test_neural_registry_and_target_authority_fail_before_method_artifacts(
    exact_bundle: dict,
) -> None:
    root = exact_bundle["root"]
    registry_output = root / "development/exact_eight/registry-drift"
    registry_output.mkdir()
    request = copy.deepcopy(exact_bundle["request"])
    request["run_id"] = "registry-drift"
    request["output_relative_dir"] = "development/exact_eight/registry-drift"
    request["expected_neural_registry_sha256"] = "0" * 64
    _assert_hold(
        lambda: exact_module.execute_development_exact_eight(root, request),
        "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REGISTRY",
    )
    assert list(registry_output.iterdir()) == []

    target_output = root / "development/exact_eight/target-authority-drift"
    target_output.mkdir()
    target_payload = copy.deepcopy(exact_bundle["targets"])
    target_payload["train_development_outcomes_sha256"] = "1" * 64
    target_path = root / "development/targets/drifted-authority.json"
    target_sha = atomic_write_json_exclusive(
        target_path,
        target_payload,
        validator=exact_module.validate_development_targets,
    )
    request = copy.deepcopy(exact_bundle["request"])
    request["run_id"] = "target-authority-drift"
    request["output_relative_dir"] = (
        "development/exact_eight/target-authority-drift"
    )
    request["development_targets_ref"] = {
        "relative_path": target_path.relative_to(root).as_posix(),
        "sha256": target_sha,
    }
    _assert_hold(
        lambda: exact_module.execute_development_exact_eight(root, request),
        "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
    )
    assert list(target_output.iterdir()) == []


def test_fullpair_ensemble_byte_tamper_fails_exact_result_replay(
    exact_bundle: dict,
) -> None:
    result, result_ref = exact_module.execute_development_exact_eight(
        exact_bundle["root"], exact_bundle["request"]
    )
    neural_record = next(
        item
        for item in result["methods"]
        if item["method_id"] == "FULL_PAIR_UNRESTRICTED_SIAMESE"
    )
    evidence = exact_module._load_method_evidence_exact(
        exact_bundle["root"], neural_record["evidence_ref"]
    )
    ensemble_path = exact_bundle["root"] / evidence["estimands"][0][
        "ensemble_ref"
    ]["relative_path"]
    ensemble_path.write_bytes(ensemble_path.read_bytes() + b"\n")
    _assert_hold(
        lambda: exact_module.load_development_exact_eight_result_exact(
            exact_bundle["root"], result_ref
        ),
        "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
    )


def test_result_and_intermediate_names_are_exclusive(exact_bundle: dict) -> None:
    exact_module.execute_development_exact_eight(
        exact_bundle["root"], exact_bundle["request"]
    )
    error = _assert_hold(
        lambda: exact_module.execute_development_exact_eight(
            exact_bundle["root"], exact_bundle["request"]
        ),
        "HOLD_DEVELOPMENT_EXACT_EIGHT_SCALAR_REPLAY",
    )
    assert "already exists" in error.message
