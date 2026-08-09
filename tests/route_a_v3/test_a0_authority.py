"""Focused tests for V3 authority binding and read-only validator behavior."""

from __future__ import annotations

from pathlib import Path


def test_authority_constants_and_source_hash_are_frozen(validator, bundle_documents):
    config, supersession, registries = bundle_documents
    assert validator.CONTRACT_ID == "mrna_xeditflow_route_a_v3"
    assert validator.VERSION == "3.0.0"
    assert validator.CONFIG_STATUS == "ACTIVE_AUTHORITATIVE_CONTRACT"
    assert config["contract_id"] == validator.CONTRACT_ID
    assert config["version"] == validator.VERSION
    assert config["status"] == validator.CONFIG_STATUS
    assert config["authority"]["source_goal"] == {
        "local_path": validator.SOURCE_CONTRACT_PATH,
        "sha256": validator.SOURCE_CONTRACT_SHA256,
        "repository_path": validator.GOAL_PATH,
    }
    assert supersession["active_contract"] == validator.CONTRACT_ID
    assert supersession["active_contract_path"] == validator.GOAL_PATH
    assert supersession["active_contract_sha256"] == validator.SOURCE_CONTRACT_SHA256
    assert supersession["new_authority"]["status"] in {
        validator.CONFIG_STATUS,
        "ACTIVE_AUTHORITATIVE_CONTRACT_PENDING_A0_ACCEPTANCE",
    }
    assert all(doc["contract_id"] == validator.CONTRACT_ID for doc in registries.values())


def test_predecessor_bindings_are_historical_and_hash_frozen(validator, bundle_documents):
    _, supersession, _ = bundle_documents
    records = {record["record_id"]: record for record in supersession["predecessors"]}
    assert set(records) == set(validator.EXPECTED_PREDECESSOR_BINDINGS)
    for record_id, expected in validator.EXPECTED_PREDECESSOR_BINDINGS.items():
        record = records[record_id]
        assert "HISTORICAL" in record["status"]
        for key, value in expected.items():
            assert record[key] == value

    gates = {record["gate_id"]: record for record in supersession["historical_gate_records"]}
    assert set(gates) == set(validator.EXPECTED_HISTORICAL_GATE_BINDINGS)
    for gate_id, (path, sha256) in validator.EXPECTED_HISTORICAL_GATE_BINDINGS.items():
        assert gates[gate_id]["path"] == path
        assert gates[gate_id]["sha256"] == sha256
        assert gates[gate_id]["rerun_in_a0"] is False


def test_authority_validator_has_no_binding_mismatch_in_staging(validator, repo_root, bundle_documents):
    """A staging-only tree may omit preserved base files; metadata must still agree.

    After overlay on the base worktree, the allowed missing-file issues disappear
    and the same function additionally verifies every historical byte hash.
    """

    config, supersession, registries = bundle_documents
    issues = validator.validate_contract_authority(repo_root, config, supersession, registries)
    allowed_staging_only = {
        "ACTIVE_CONTRACT_UNREADABLE",
        "HISTORICAL_FILE_MISSING_OR_UNSAFE",
    }
    unexpected = [issue for issue in issues if issue.code not in allowed_staging_only]
    assert unexpected == []


def test_default_static_checks_do_not_call_path_write_apis(validator, repo_root, bundle_documents, monkeypatch):
    def forbidden_write(*args, **kwargs):
        raise AssertionError("default validation attempted a write")

    monkeypatch.setattr(Path, "write_bytes", forbidden_write)
    monkeypatch.setattr(Path, "write_text", forbidden_write)
    config, supersession, registries = bundle_documents
    assert validator.validate_registry_closure(config, registries) == []
    assert validator.validate_sealed_hard_disable(config, registries) == []
    assert validator.validate_l4_and_pre_v3(config, supersession, registries["claim"]) == []
    assert validator.validate_schema_manifest(repo_root) == []
    assert validator.scan_conflict_markers(repo_root) == []
