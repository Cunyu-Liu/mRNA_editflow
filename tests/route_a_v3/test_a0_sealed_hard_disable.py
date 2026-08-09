"""Static and runtime-free A0 sealed hard-disable tests."""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


def _load_sealed_guard(repo_root):
    path = repo_root / "scripts" / "route_a_v3" / "sealed_guard.py"
    spec = importlib.util.spec_from_file_location("route_a_v3_test_sealed_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a0_sealed_policy_is_hard_disabled(validator, bundle_documents):
    config, _, registries = bundle_documents
    assert validator.validate_sealed_hard_disable(config, registries) == []


def test_enabling_sealed_execution_is_rejected(validator, bundle_documents):
    source_config, _, registries = bundle_documents
    config = deepcopy(source_config)
    config["sealed"]["execution_enabled"] = True
    codes = {issue.code for issue in validator.validate_sealed_hard_disable(config, registries)}
    assert "SEALED_HARD_DISABLE" in codes


def test_a0_guard_mode_and_a9_replacement_boundary_are_frozen(
    validator,
    bundle_documents,
):
    source_config, _, registries = bundle_documents
    config = deepcopy(source_config)
    config["sealed"]["guard_mode"] = "TOGGLE_AUTHORIZATION"
    config["sealed"]["latent_authorization_path_allowed"] = True
    config["sealed"]["evaluator_implementation_status"] = "IMPLEMENTED"
    config["sealed"]["a9_guard_replacement_required"] = False
    config["sealed"]["a9_replacement_preconditions"] = [
        "SEPARATE_EXPLICIT_USER_AUTHORIZATION_FOR_A10",
    ]
    codes = {
        issue.code
        for issue in validator.validate_sealed_hard_disable(config, registries)
    }
    assert "SEALED_HARD_DISABLE" in codes
    assert "SEALED_A9_REPLACEMENT_PRECONDITIONS" in codes


def test_gse246381_ordinary_or_training_role_is_rejected(validator, bundle_documents):
    config, _, source_registries = bundle_documents
    registries = deepcopy(source_registries)
    sealed = next(row for row in registries["data"]["datasets"] if row["dataset_id"] == validator.SEALED_DATASET_ID)
    sealed["role"] = "AUDIT_ONLY"
    sealed["training_role"] = "CRITIC_AUX"
    sealed["all_training_roles_excluded"] = False
    registries["data"]["ordinary_candidate_dataset_ids"].append(validator.SEALED_DATASET_ID)
    codes = {issue.code for issue in validator.validate_sealed_hard_disable(config, registries)}
    assert "SEALED_DATA_ROLE" in codes
    assert "SEALED_DATASET_IN_ORDINARY_SET" in codes


def test_authorization_record_must_be_empty_at_a0(validator, bundle_documents):
    source_config, _, registries = bundle_documents
    config = deepcopy(source_config)
    config["sealed"]["authorization_record_path"] = "docs/execution/not_authorized.yaml"
    config["sealed"]["authorization_record_sha256"] = "0" * 64
    codes = {issue.code for issue in validator.validate_sealed_hard_disable(config, registries)}
    assert "SEALED_AUTHORIZATION_PREPOPULATED" in codes


def test_route_a_static_validator_imports_no_torch_or_sealed_state(validator, repo_root):
    assert validator.validate_python_static_safety(repo_root) == []


def test_guard_unconditionally_rejects_without_inspecting_arguments(repo_root):
    guard = _load_sealed_guard(repo_root)

    class ExplosiveCall:
        def __getattribute__(self, name):
            raise AssertionError("A0 guard inspected call argument " + name)

    with pytest.raises(guard.RouteAV3SealedHardDisabled) as caught:
        guard.assert_sealed_final_authorized(
            ExplosiveCall(),
            repo_root=ExplosiveCall(),
        )
    assert str(caught.value) == "ROUTE_A_V3_SEALED_HARD_DISABLED_A0_A9"


def test_forged_toggle_authorization_and_manifests_still_rejected_without_io(
    tmp_path,
    repo_root,
):
    guard = _load_sealed_guard(repo_root)
    nonexistent = tmp_path / "does-not-exist"
    forged_call = {
        "mode": "sealed-final",
        "dataset": nonexistent / "dataset.jsonl",
        "prereg": nonexistent / "prereg.yaml",
        "ckpt_dir": nonexistent / "checkpoints",
        "restricted": nonexistent / "restricted",
        "raw_seq_dir": nonexistent / "raw",
        "out_dir": nonexistent / "output",
        "n_perm": 1,
        "seed": 0,
        "gpu": "cuda:1",
        "execution_enabled": True,
        "execution_authorized": True,
        "authorized": True,
        "access_intent_allowed": True,
        "authorization_record_path": nonexistent / "authorization.yaml",
        "readiness_record_path": nonexistent / "readiness.yaml",
        "execution_manifest_path": nonexistent / "execution_manifest.yaml",
    }
    with pytest.raises(
        guard.RouteAV3SealedHardDisabled,
        match="^ROUTE_A_V3_SEALED_HARD_DISABLED_A0_A9$",
    ):
        guard.assert_sealed_final_authorized(forged_call, repo_root=nonexistent)
    assert not nonexistent.exists()
    assert not list(tmp_path.rglob("ACCESS_LOG.jsonl"))
    assert not list(tmp_path.rglob("*.lock"))


def test_guard_ast_has_only_unconditional_raise_and_no_reachable_return(repo_root):
    guard_path = repo_root / "scripts" / "route_a_v3" / "sealed_guard.py"
    tree = ast.parse(guard_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "assert_sealed_final_authorized"
    )
    executable = function.body[1:] if (
        function.body
        and isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
    ) else function.body
    assert len(executable) == 1
    assert isinstance(executable[0], ast.Raise)
    assert not any(isinstance(node, ast.Return) for node in ast.walk(function))
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert len(imports) == 1
    assert isinstance(imports[0], ast.ImportFrom)
    assert imports[0].module == "__future__"


def test_runner_guards_before_runtime_imports_paths_and_state(repo_root):
    runner_path = repo_root / "scripts" / "e0x" / "run_e0x_final.py"
    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    top_project_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("scripts.")
    ]
    assert [node.module for node in top_project_imports] == [
        "scripts.route_a_v3.sealed_guard"
    ]

    main = functions["main"]
    parse_index = next(
        index
        for index, statement in enumerate(main.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "parse_args"
    )
    guard_if = main.body[parse_index + 1]
    assert isinstance(guard_if, ast.If)
    assert ast.unparse(guard_if.test) == "args.mode == 'sealed-final'"
    assert len(guard_if.body) == 1
    assert isinstance(guard_if.body[0], ast.Expr)
    assert isinstance(guard_if.body[0].value, ast.Call)
    assert getattr(guard_if.body[0].value.func, "id", None) == (
        "assert_sealed_final_authorized"
    )

    sealed = functions["run_sealed_final"]
    sealed_executable = sealed.body[1:]  # skip docstring
    assert isinstance(sealed_executable[0], ast.Expr)
    assert isinstance(sealed_executable[0].value, ast.Call)
    assert getattr(sealed_executable[0].value.func, "id", None) == (
        "assert_sealed_final_authorized"
    )

    guard_line = guard_if.body[0].lineno
    sensitive_fragments = (
        "from scripts.e0x import prereg",
        "from scripts.m4_sparse import config as C",
        "from scripts.m4_sparse.dataset import build_vocab",
        "prereg.load_prereg",
        "load_rows(args.dataset)",
        "select_device(cfg, args.gpu)",
        "import torch",
    )
    for fragment in sensitive_fragments:
        line = next(
            index
            for index, text in enumerate(source.splitlines(), start=1)
            if fragment in text and index >= main.lineno
        )
        assert guard_line < line

    sealed_guard_line = sealed_executable[0].lineno
    for fragment in (
        "from scripts.e0x import sealed",
        'args.restricted / "ACCESS_LOG.jsonl"',
        "sealed.SealedAccessState(access_log)",
        "sm.append_intent",
        "sm.reserve",
    ):
        line = next(
            index
            for index, text in enumerate(source.splitlines(), start=1)
            if fragment in text and index >= sealed.lineno
        )
        assert sealed_guard_line < line
