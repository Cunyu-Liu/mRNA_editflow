"""Negative tests for decision, manifest, supersession and sealed AST drift."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy


def _codes(issues):
    return {issue.code for issue in issues}


def _copy_runner_and_guard(validator, repo_root, tmp_path):
    runner_target = tmp_path / validator.SEALED_RUNNER_PATH
    guard_target = tmp_path / validator.SEALED_GUARD_PATH
    runner_target.parent.mkdir(parents=True)
    guard_target.parent.mkdir(parents=True)
    shutil.copy2(repo_root / validator.SEALED_RUNNER_PATH, runner_target)
    shutil.copy2(repo_root / validator.SEALED_GUARD_PATH, guard_target)
    return runner_target, guard_target


def test_decision_log_requires_all_ids_and_historical_m0_decision(validator, repo_root):
    decision_log = validator._load_yaml(repo_root, validator.DECISION_LOG_PATH)
    assert validator.validate_decision_log(decision_log) == []

    bypass = deepcopy(decision_log)
    bypass["decisions"] = [row for row in bypass["decisions"] if row["decision_id"] != "V3-DEC-014"]
    codes = _codes(validator.validate_decision_log(bypass))
    assert "DECISION_LOG_ID_CLOSURE" in codes
    assert "DECISION_LOG_KEY_DECISION" in codes


def test_decision_015_historical_security_design_is_preserved(validator, repo_root):
    decision_log = validator._load_yaml(repo_root, validator.DECISION_LOG_PATH)
    assert validator.validate_decision_log(decision_log) == []

    bypass = deepcopy(decision_log)
    decision = next(row for row in bypass["decisions"] if row["decision_id"] == "V3-DEC-015")
    decision["status"] = "MUTABLE"
    decision["effective_phase"] = "A10"
    decision["requires_user_authorization"] = True
    decision["sealed_contact"] = True
    decision["evidence_refs"].remove(validator.SEALED_GUARD_PATH)
    codes = _codes(validator.validate_decision_log(bypass))
    assert "DECISION_LOG_SECURITY_DESIGN" in codes
    assert "DECISION_LOG_SECURITY_EVIDENCE" in codes


def test_decision_016_superseding_a0_boundary_is_frozen(validator, repo_root):
    decision_log = validator._load_yaml(repo_root, validator.DECISION_LOG_PATH)
    assert validator.validate_decision_log(decision_log) == []

    bypass = deepcopy(decision_log)
    decision = next(row for row in bypass["decisions"] if row["decision_id"] == "V3-DEC-016")
    decision["status"] = "MUTABLE"
    decision["supersedes_decision_id"] = "V3-DEC-014"
    decision["effective_phase"] = "A9"
    decision["requires_user_authorization"] = True
    decision["sealed_contact"] = True
    decision["resolution"] = decision["resolution"].replace(
        "unconditional A0-A9 hard disable",
        "conditional toggle",
    )
    decision["evidence_refs"].remove(validator.SEALED_RUNNER_PATH)
    codes = _codes(validator.validate_decision_log(bypass))
    assert "DECISION_LOG_A0_PHASE_BOUNDARY" in codes
    assert "DECISION_LOG_A0_PHASE_BOUNDARY_EVIDENCE" in codes
    assert "DECISION_LOG_KEY_DECISION" in codes


def test_supersession_config_and_m0_scientific_failure_are_bound(validator, repo_root, bundle_documents):
    config, source_supersession, registries = bundle_documents
    supersession = deepcopy(source_supersession)
    supersession["new_authority"]["config_sha256"] = "0" * 64
    m0 = next(row for row in supersession["historical_gate_records"] if row["gate_id"] == "M0_SCIENTIFIC_ORIGINAL")
    m0["macro_sign_accuracy"] = 0.61
    m0["o0_valid"] = True
    codes = _codes(validator.validate_contract_authority(repo_root, config, supersession, registries))
    assert "SUPERSESSION_CONFIG_HASH" in codes
    assert "M0_SCIENTIFIC_BINDING" in codes
    assert "M0_SCIENTIFIC_THRESHOLD" in codes


def test_runner_and_guard_ast_are_fail_closed(validator, repo_root):
    assert validator.validate_runner_and_guard_ast(repo_root) == []


def test_runner_ast_rejects_guard_moved_after_runtime_import(
    validator,
    repo_root,
    tmp_path,
):
    runner_target, _ = _copy_runner_and_guard(
        validator,
        repo_root,
        tmp_path,
    )
    source = runner_target.read_text(encoding="utf-8")
    guard_block = (
        '    if args.mode == "sealed-final":\n'
        "        assert_sealed_final_authorized(args)\n\n"
    )
    source = source.replace(guard_block, "", 1)
    source = source.replace(
        "    from scripts.e0x import prereg\n",
        "    from scripts.e0x import prereg\n" + guard_block,
        1,
    )
    runner_target.write_text(source, encoding="utf-8")
    codes = _codes(validator.validate_runner_and_guard_ast(tmp_path))
    assert "RUNNER_EARLY_GUARD_MISSING" in codes


def test_runner_ast_rejects_eager_project_import(
    validator,
    repo_root,
    tmp_path,
):
    runner_target, _ = _copy_runner_and_guard(
        validator,
        repo_root,
        tmp_path,
    )
    source = runner_target.read_text(encoding="utf-8").replace(
        "from scripts.route_a_v3.sealed_guard import assert_sealed_final_authorized\n",
        "from scripts.route_a_v3.sealed_guard import assert_sealed_final_authorized\n"
        "from scripts.m4_sparse import config as EAGER_CONFIG\n",
        1,
    )
    runner_target.write_text(source, encoding="utf-8")
    assert "RUNNER_MODULE_PROJECT_IMPORT" in _codes(
        validator.validate_runner_and_guard_ast(tmp_path)
    )


def test_runner_ast_requires_defense_guard_expression_first(
    validator,
    repo_root,
    tmp_path,
):
    runner_target, _ = _copy_runner_and_guard(
        validator,
        repo_root,
        tmp_path,
    )
    source = runner_target.read_text(encoding="utf-8").replace(
        "    assert_sealed_final_authorized(args)\n"
        "    from scripts.e0x import sealed\n",
        "    authorization = assert_sealed_final_authorized(args)\n"
        "    from scripts.e0x import sealed\n",
        1,
    )
    runner_target.write_text(source, encoding="utf-8")
    assert "RUN_SEALED_FIRST_GUARD" in _codes(
        validator.validate_runner_and_guard_ast(tmp_path)
    )


def test_guard_ast_rejects_toggle_and_reachable_return(
    validator,
    repo_root,
    tmp_path,
):
    _, guard_target = _copy_runner_and_guard(
        validator,
        repo_root,
        tmp_path,
    )
    source = guard_target.read_text(encoding="utf-8").replace(
        "    raise RouteAV3SealedHardDisabled(HARD_DISABLED)\n",
        '    if getattr(call_args, "execution_authorized", False):\n'
        "        return None\n"
        "    raise RouteAV3SealedHardDisabled(HARD_DISABLED)\n",
        1,
    )
    guard_target.write_text(source, encoding="utf-8")
    codes = _codes(validator.validate_runner_and_guard_ast(tmp_path))
    assert "SEALED_GUARD_HARD_DISABLE_BODY" in codes
    assert "SEALED_GUARD_REACHABLE_SUCCESS" in codes


def test_guard_ast_rejects_any_path_or_manifest_import(
    validator,
    repo_root,
    tmp_path,
):
    _, guard_target = _copy_runner_and_guard(
        validator,
        repo_root,
        tmp_path,
    )
    source = guard_target.read_text(encoding="utf-8").replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\nfrom pathlib import Path\n",
        1,
    )
    guard_target.write_text(source, encoding="utf-8")
    codes = _codes(validator.validate_runner_and_guard_ast(tmp_path))
    assert "SEALED_GUARD_MODULE_SHAPE" in codes
    assert "SEALED_GUARD_IMPORT" in codes


def test_registry_manifest_detects_every_listed_hash_drift(validator, tmp_path, monkeypatch):
    goal_bytes = b"fixture Route A V3 authority\n"
    goal_hash = validator.sha256_bytes(goal_bytes)
    monkeypatch.setattr(validator, "SOURCE_CONTRACT_SHA256", goal_hash)
    goal_path = tmp_path / validator.GOAL_PATH
    goal_path.parent.mkdir(parents=True)
    goal_path.write_bytes(goal_bytes)

    entries = []
    for index, relative in enumerate(sorted(validator.MANDATORY_REGISTRY_MANIFEST_PATHS)):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"fixture-{index}-{relative}\n".encode("utf-8")
        path.write_bytes(data)
        entries.append({"path": relative, "role": "FIXTURE", "sha256": validator.sha256_bytes(data)})
    manifest = {
        "contract_id": validator.CONTRACT_ID,
        "version": validator.VERSION,
        "contract_path": validator.GOAL_PATH,
        "contract_sha256": goal_hash,
        "base_commit": "bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c",
        "sealed_contact": False,
        "files": entries,
    }
    manifest_path = tmp_path / validator.REGISTRY_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validator.validate_registry_manifest(tmp_path) == []

    config_path = tmp_path / validator.CONFIG_PATH
    config_path.write_bytes(config_path.read_bytes() + b"drift")
    assert "REGISTRY_MANIFEST_HASH_MISMATCH" in _codes(validator.validate_registry_manifest(tmp_path))
