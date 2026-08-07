"""R0 governance, authority identity, and lineage fail-closed tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BINDING = REPO_ROOT / "docs/contracts/xeditflow_v2_source_binding.yaml"
REPAIR_LOG = REPO_ROOT / "docs/execution/xeditflow_repair_decision_log.yaml"
SCOPE = REPO_ROOT / "configs/repair/r0_r2_repair_scope.yaml"
BUILDER = REPO_ROOT / "scripts/repair/build_r0_r2_lineage_manifest.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_builder():
    spec = importlib.util.spec_from_file_location("r0_lineage_builder", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lineage_builder_direct_cli_is_importable():
    completed = subprocess.run(
        [sys.executable, str(BUILDER), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--ordinary-artifact" in completed.stdout


def test_source_binding_has_one_executable_authority_without_equivalence_claim():
    binding = yaml.safe_load(BINDING.read_text(encoding="utf-8"))
    assert binding["source_proposal"]["sha256"] == (
        "9c79edd819e45551974bcfeb14a400dd504c55c0a7c869e456e638daf49f1c1e"
    )
    assert binding["source_proposal"]["authority_role"] == "SOURCE_PROPOSAL_NOT_ACTIVE"
    active = binding["active_executable_authority"]
    assert active["authority_role"] == "ACTIVE_EXECUTABLE_AUTHORITY"
    assert active["sha256"] == _sha256(REPO_ROOT / active["path"])
    assert active["config_sha256"] == _sha256(REPO_ROOT / active["config_path"])
    assert binding["relationship"]["classification"] == "DERIVED_NOT_BYTE_EQUIVALENT"
    assert binding["relationship"]["complete_section_crosswalk_proven"] is False
    assert binding["relationship"]["executable_authority_count"] == 1


def test_repair_overlay_retains_original_failure_gate_and_sealed_prohibition():
    log = yaml.safe_load(REPAIR_LOG.read_text(encoding="utf-8"))
    inv = log["invariants"]
    assert inv["original_m0_effect_gate_result"] == "FAIL_RETAINED"
    assert inv["sign_accuracy_threshold"] == 0.60
    assert inv["sign_gate_amendment"] == "NOT_AUTHORIZED"
    assert inv["sealed_analytic_member_access"] == "FORBIDDEN_DURING_R0_R2"
    assert inv["formal_sealed_evaluator_invocation"] == "FORBIDDEN_DURING_R0_R2"
    assert inv["git_push"] == "NOT_AUTHORIZED"
    assert _sha256(REPO_ROOT / log["historical_decision_log"]) == log[
        "historical_decision_log_sha256"
    ]


def _init_fixture_repo(root: Path, base_commit_placeholder: str = "") -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "fixture@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "fixture"], check=True)
    (root / "scripts/repair").mkdir(parents=True)
    (root / "scripts/repair/fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def test_lineage_builder_hashes_tracked_scope_and_explicit_ordinary_input(tmp_path: Path):
    module = _load_builder()
    repo = tmp_path / "repo"
    repo.mkdir()
    head = _init_fixture_repo(repo)
    scope = {
        "repair_goal": "FIXTURE",
        "base_commit": head,
        "source_binding": "fixture-binding",
        "repair_decision_log": "fixture-log",
        "target_state": "REPAIR_IMPLEMENTED_NOT_SCIENTIFIC_GO",
        "tracked_scope_globs": ["scripts/repair/*.py"],
        "forbidden_input_tokens": ["sealed", "restricted", "gse246381"],
    }
    scope_path = repo / "scope.yaml"
    scope_path.write_text(yaml.safe_dump(scope), encoding="utf-8")
    ordinary = tmp_path / "ordinary_effects.json"
    ordinary.write_text(json.dumps({"delta": 1.0}), encoding="utf-8")

    manifest = module.build_manifest(
        repo_root=repo,
        scope_path=scope_path,
        run_id="fixture-run",
        ordinary_artifacts=[ordinary],
    )
    assert manifest["head_commit"] == head
    assert [r["path"] for r in manifest["tracked_files"]] == ["scripts/repair/fixture.py"]
    assert manifest["ordinary_external_inputs"][0]["sha256"] == _sha256(ordinary)
    assert manifest["sealed_safety"] == {
        "formal_evaluator_invoked": "UNKNOWN_NOT_ASSERTED",
        "access_intent_written": "UNKNOWN_NOT_ASSERTED",
        "reservation_written": "UNKNOWN_NOT_ASSERTED",
        "restricted_or_sealed_member_input_count": "UNKNOWN_NOT_ASSERTED",
        "enforcement": (
            "ordinary inputs passed path and JSONL row guards; runtime "
            "zero-access requires independent before/after access-log evidence"),
    }
    payload_hash = manifest.pop("manifest_payload_sha256")
    assert payload_hash == module.canonical_sha256(manifest)


@pytest.mark.parametrize("token", ["sealed", "restricted", "GSE246381"])
def test_lineage_builder_rejects_forbidden_external_path_before_open(tmp_path: Path, token: str):
    module = _load_builder()
    path = tmp_path / f"{token}_members.jsonl"
    # The path need not exist: rejection must occur before stat/open.
    with pytest.raises(ValueError, match="rejected before open"):
        module._reject_forbidden_path(path, ["sealed", "restricted", "gse246381"])
