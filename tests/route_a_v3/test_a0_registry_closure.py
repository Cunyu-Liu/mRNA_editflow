"""Expected-ID, foreign-key and A0-through-A10 dependency tests."""

from __future__ import annotations

from copy import deepcopy


def test_registry_expected_ids_and_foreign_keys_close(validator, bundle_documents):
    config, _, registries = bundle_documents
    assert validator.validate_registry_closure(config, registries) == []


def test_phase_dependencies_match_frozen_route_a_dag(validator, bundle_documents):
    config, _, registries = bundle_documents
    issues = validator.validate_phase_dependencies(config["phase_plan"], registries["task"]["phase_tasks"])
    assert issues == []
    assert {row["phase_id"] for row in config["phase_plan"]} == set(validator.EXPECTED_PHASE_IDS)
    assert {row["phase_id"] for row in registries["task"]["phase_tasks"]} == set(validator.EXPECTED_PHASE_IDS)


def test_unknown_split_fk_is_rejected(validator, bundle_documents):
    config, _, source_registries = bundle_documents
    registries = deepcopy(source_registries)
    registries["matrix"]["matrix"]["T5_SOURCE_RELATIVE_EFFECT"].append("S_UNKNOWN")
    codes = {issue.code for issue in validator.validate_registry_closure(config, registries)}
    assert "MATRIX_SPLIT_FK" in codes


def test_ordinary_task_cannot_reference_sealed_split(validator, bundle_documents):
    config, _, source_registries = bundle_documents
    registries = deepcopy(source_registries)
    registries["matrix"]["matrix"]["T5_SOURCE_RELATIVE_EFFECT"].append(validator.SEALED_SPLIT_ID)
    codes = {issue.code for issue in validator.validate_registry_closure(config, registries)}
    assert "ORDINARY_TASK_USES_SEALED" in codes


def test_missing_phase_dependency_is_rejected(validator, bundle_documents):
    config, _, registries = bundle_documents
    config_phases = deepcopy(config["phase_plan"])
    a7 = next(row for row in config_phases if row["phase_id"] == "A7")
    a7["depends_on"] = ["A5"]
    codes = {issue.code for issue in validator.validate_phase_dependencies(config_phases, registries["task"]["phase_tasks"])}
    assert "PHASE_DEPENDENCY_MISMATCH" in codes
    assert "PHASE_DEPENDENCY_CROSS_FILE_MISMATCH" in codes
