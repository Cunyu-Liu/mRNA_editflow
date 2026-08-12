from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py"
CONFIG = ROOT / "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
SPEC = importlib.util.spec_from_file_location("gse200304_dec019_adjudicator_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADJ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADJ)

SYNTHETIC_IMPLEMENTATION_COMMIT = "9" * 40
SYNTHETIC_BINDING_COMMIT = "8" * 40
SYNTHETIC_DESCENDANT_COMMIT = "7" * 40
SYNTHETIC_SCRIPT_SHA256 = "6" * 64
SYNTHETIC_TEST_SHA256 = "5" * 64
FORCE_BOUND_CONFIGS_ENV = "G200_V3_TEST_FORCE_BOUND_CONFIGS"


def _read_test_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if os.environ.get(FORCE_BOUND_CONFIGS_ENV) == "1":
        value["implementation_binding"].update(
            {
                "status": "BOUND",
                "implementation_commit": SYNTHETIC_IMPLEMENTATION_COMMIT,
                "implementation_script_sha256": SYNTHETIC_SCRIPT_SHA256,
                "implementation_test_sha256": SYNTHETIC_TEST_SHA256,
            }
        )
    return value


def read_config() -> dict[str, Any]:
    return _read_test_config(CONFIG)


def bind_implementation(
    config: Mapping[str, Any] | None = None,
    *,
    implementation_commit: str = SYNTHETIC_IMPLEMENTATION_COMMIT,
    script_sha256: str = SYNTHETIC_SCRIPT_SHA256,
    test_sha256: str = SYNTHETIC_TEST_SHA256,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(config if config is not None else read_config()))
    binding = value["implementation_binding"]
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": implementation_commit,
            "implementation_script_sha256": script_sha256,
            "implementation_test_sha256": test_sha256,
        }
    )
    return value


def unknown_implementation(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    binding = value["implementation_binding"]
    binding.update(
        {
            "status": ADJ.UNKNOWN,
            "implementation_commit": ADJ.UNKNOWN,
            "implementation_script_sha256": ADJ.UNKNOWN,
            "implementation_test_sha256": ADJ.UNKNOWN,
        }
    )
    return value


def refresh_descriptor_hash(config: dict[str, Any]) -> None:
    config["evidence_descriptor_bindings"]["descriptor_set_sha256"] = (
        ADJ.descriptor_set_sha256(config)
    )


def refresh_science_core(config: dict[str, Any]) -> str:
    digest = ADJ.config_core_sha256(config)
    config["implementation_binding"]["config_core_sha256"] = digest
    return digest


def privacy() -> dict[str, bool]:
    return {
        "contains_row_level_payload": False,
        "contains_sequence": False,
        "contains_row_identifier": False,
        "contains_raw_label_or_effect": False,
        "contains_member_identifiers_or_hashes": False,
    }


def provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = config["evidence_contract"]["required_predecessor_authority"]
    return {
        "producer_protocol_id": "SYNTHETIC_GSE200304_GATE_PRODUCER_V1",
        "producer_commit": "4" * 40,
        "producer_script_sha256": "5" * 64,
        "source_bundle_id": predecessor["bundle_id"],
        "source_bundle_root_or_target_sha256": predecessor[
            "terminal_marker_final_output_target_sha256"
        ],
        "predecessor_authority": copy.deepcopy(predecessor),
        "acceptance_authority": copy.deepcopy(
            config["evidence_contract"]["gate_record_provenance_contract"][
                "acceptance_authority"
            ]
        ),
    }


def passing_facts() -> dict[str, dict[str, Any]]:
    return {
        "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE": {
            "deterministic_row_locator_frozen": True,
            "table_s2_hash_bound": True,
            "table_s3_hash_bound": True,
            "s2_s3_join_rule_frozen": True,
            "multi_asset_lineage_closed": True,
            "locator_lineage_commitment_algorithm": (
                ADJ.LOCATOR_LINEAGE_COMMITMENT_ALGORITHM
            ),
            "locator_lineage_merkle_root_sha256": "a" * 64,
            "canonical_record_count": 6120,
            "processed_pair_count": 6772,
            "raw_replay_role": "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE",
            "raw_replay_status": "NOT_RUN",
            "independent_raw_reproduction_claimed": False,
        },
        "CANONICAL_REPORTED_ENDPOINT_SEMANTICS": {
            "author_published_processed_endpoint_is_primary": True,
            "endpoint_id_frozen": True,
            "endpoint_direction_frozen": True,
            "endpoint_scale_frozen": True,
            "contrast_and_transform_frozen": True,
            "paper_faithful_mapping_closed": True,
        },
        "BIOLOGICAL_GROUP_AUTHORITY": {
            "biological_group_id_frozen": True,
            "study_unit_is_gse200304": True,
            "gse200302_is_subseries_not_independent_study": True,
            "group_mapping_hash_bound": True,
        },
        "ROW_REPLICATE_OR_VALID_SE": {
            "replicate_or_valid_standard_error_present": True,
            "replicate_count_or_effective_n_frozen": True,
            "standard_error_semantics_frozen": True,
            "technical_uncertainty_not_substituted_for_biological_se": True,
        },
        "CHECKPOINT_SPECIFIC_EXPOSURE": {
            "checkpoint_ids_and_revisions_frozen": True,
            "checkpoint_artifact_digests_bound": True,
            "exact_member_exposure_audit_pass": True,
            "near_duplicate_exposure_audit_pass": True,
            "audited_checkpoint_count": 4,
        },
        "LICENSE_RIGHTS": {
            "rights_source_authority_closed": True,
            "qualification_use_allowed": True,
            "private_canonical_materialization_allowed": True,
            "redistribution_scope": "PRIVATE_CANONICAL_ONLY",
        },
        "OUTCOME_BLIND_SPLIT_LEAKAGE": {
            "a1_source_graph_frozen": True,
            "a1_group_graph_frozen": True,
            "a1_near_duplicate_graph_frozen": True,
            "split_salt_hash_bound": True,
            "outcome_blind_assignment": True,
            "leakage_audit_pass": True,
            "final_benchmark_membership_deferred_to_a2": True,
        },
        "PREFROZEN_POWER_PRECISION": {
            "analysis_unit": "BIOLOGICAL_GROUP",
            "bootstrap_unit": "BIOLOGICAL_GROUP",
            "observed_power": 0.8,
            "full_confidence_interval_width": 0.3,
            "prefrozen_before_model_results": True,
        },
    }


def gate_record(
    config: Mapping[str, Any],
    slot_id: str,
    *,
    status: str = "PASS",
    facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    is_pass = status == "PASS"
    record = {
        "schema_version": ADJ.EVIDENCE_SCHEMA_VERSION,
        "record_type": ADJ.EVIDENCE_RECORD_TYPE,
        "contract_id": ADJ.CONTRACT_ID,
        "decision_id": ADJ.DECISION_ID,
        "dataset_id": ADJ.DATASET_ID,
        "gate_id": slot_id,
        "status": status,
        "accepted": True,
        "aggregate_only": True,
        "privacy": privacy(),
        "provenance": provenance(config),
        "facts": copy.deepcopy(dict(facts)) if is_pass and facts is not None else None,
        "unknown_fields": [] if is_pass else sorted(ADJ.FACT_KEYS[slot_id]),
        "reason_codes": [] if is_pass else [f"{slot_id}_NOT_ESTABLISHED"],
    }
    if is_pass and slot_id == "BIOLOGICAL_GROUP_AUTHORITY":
        record["provenance"][ADJ.GROUP_MAPPING_COMMITMENT_KEY] = "b" * 64
    return record


def materialize_evidence(
    root: Path,
    config: dict[str, Any],
    *,
    statuses: Mapping[str, str] | None = None,
    fact_updates: Mapping[str, Mapping[str, Any]] | None = None,
    record_updates: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    root.mkdir()
    statuses = statuses or {}
    facts = passing_facts()
    for slot_id, updates in (fact_updates or {}).items():
        facts[slot_id].update(updates)
    descriptor_by_slot = {
        slot["slot_id"]: slot
        for slot in config["evidence_descriptor_bindings"]["slots"]
    }
    paths: dict[str, Path] = {}
    for slot in config["evidence_contract"]["slots"]:
        slot_id = slot["slot_id"]
        record = gate_record(
            config,
            slot_id,
            status=statuses.get(slot_id, "PASS"),
            facts=facts[slot_id],
        )
        record.update(copy.deepcopy(dict((record_updates or {}).get(slot_id, {}))))
        payload = ADJ.json_bytes(record)
        path = root / slot["allowed_basename"]
        path.write_bytes(payload)
        descriptor_by_slot[slot_id].update(
            {"absolute_path": str(path), "sha256": ADJ.sha256(payload), "bytes": len(payload)}
        )
        paths[slot_id] = path
    config["evidence_descriptor_bindings"]["status"] = "BOUND"
    refresh_descriptor_hash(config)
    return paths


def read_report(output: Path) -> dict[str, Any]:
    return json.loads((output / "ADJUDICATION_REPORT.json").read_text(encoding="utf-8"))


def _evidence_slot(config: Mapping[str, Any], slot_id: str) -> Mapping[str, Any]:
    return next(
        slot
        for slot in config["evidence_contract"]["slots"]
        if slot["slot_id"] == slot_id
    )


def test_group_negative_record_keeps_exact_seven_key_provenance() -> None:
    config = bind_implementation()
    slot = _evidence_slot(config, "BIOLOGICAL_GROUP_AUTHORITY")
    record = gate_record(
        config,
        "BIOLOGICAL_GROUP_AUTHORITY",
        status="BLOCKED",
    )
    assert set(record["provenance"]) == ADJ.PROVENANCE_KEYS
    accepted = ADJ._validate_gate_record(ADJ.json_bytes(record), slot, config)
    assert accepted["status"] == "BLOCKED"


def test_group_pass_without_mapping_commitment_is_rejected() -> None:
    config = bind_implementation()
    slot = _evidence_slot(config, "BIOLOGICAL_GROUP_AUTHORITY")
    record = gate_record(
        config,
        "BIOLOGICAL_GROUP_AUTHORITY",
        facts=passing_facts()["BIOLOGICAL_GROUP_AUTHORITY"],
    )
    record["provenance"].pop(ADJ.GROUP_MAPPING_COMMITMENT_KEY)
    with pytest.raises(ADJ.AdjudicationError, match="provenance.*keys differ"):
        ADJ._validate_gate_record(ADJ.json_bytes(record), slot, config)


def test_group_pass_exact_eight_provenance_is_accepted_and_passes_gate() -> None:
    config = bind_implementation()
    ADJ.validate_static_config(config)
    assert config["evidence_contract"]["gate_record_provenance_contract"][
        "biological_group_pass_requires_mapping_commitment_sha256"
    ] is True
    slot = _evidence_slot(config, "BIOLOGICAL_GROUP_AUTHORITY")
    record = gate_record(
        config,
        "BIOLOGICAL_GROUP_AUTHORITY",
        facts=passing_facts()["BIOLOGICAL_GROUP_AUTHORITY"],
    )
    assert set(record["provenance"]) == (
        ADJ.PROVENANCE_KEYS | {ADJ.GROUP_MAPPING_COMMITMENT_KEY}
    )
    accepted = ADJ._validate_gate_record(ADJ.json_bytes(record), slot, config)
    assert ADJ._slot_gate_pass(slot["slot_id"], accepted["facts"]) is True


def test_commitment_must_be_hex64_and_is_forbidden_on_other_slots() -> None:
    config = bind_implementation()
    group_slot = _evidence_slot(config, "BIOLOGICAL_GROUP_AUTHORITY")
    malformed = gate_record(
        config,
        "BIOLOGICAL_GROUP_AUTHORITY",
        facts=passing_facts()["BIOLOGICAL_GROUP_AUTHORITY"],
    )
    malformed["provenance"][ADJ.GROUP_MAPPING_COMMITMENT_KEY] = "not-a-digest"
    with pytest.raises(ADJ.AdjudicationError, match="mapping commitment"):
        ADJ._validate_gate_record(ADJ.json_bytes(malformed), group_slot, config)

    endpoint_slot = _evidence_slot(
        config,
        "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
    )
    other = gate_record(
        config,
        "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
        facts=passing_facts()["CANONICAL_REPORTED_ENDPOINT_SEMANTICS"],
    )
    other["provenance"][ADJ.GROUP_MAPPING_COMMITMENT_KEY] = "b" * 64
    with pytest.raises(ADJ.AdjudicationError, match="provenance.*keys differ"):
        ADJ._validate_gate_record(ADJ.json_bytes(other), endpoint_slot, config)


def test_static_current_and_synthetic_i_b_freeze_f492_successor_chain_and_truth() -> None:
    current_config = read_config()
    ADJ.validate_static_config(current_config)
    current_binding = current_config["implementation_binding"]
    if current_binding["status"] == "BOUND":
        ADJ.validate_implementation_binding(current_config)
        assert current_binding["implementation_commit"] != ADJ.UNKNOWN
        assert current_binding["implementation_script_sha256"] != ADJ.UNKNOWN
        assert current_binding["implementation_test_sha256"] != ADJ.UNKNOWN
    else:
        assert current_binding["status"] == ADJ.UNKNOWN
        assert all(
            current_binding[key] == ADJ.UNKNOWN
            for key in (
                "implementation_commit",
                "implementation_script_sha256",
                "implementation_test_sha256",
            )
        )

    config_i = unknown_implementation(current_config)
    ADJ.validate_static_config(config_i)
    config_b = bind_implementation(config_i)
    ADJ.validate_static_config(config_b)
    assert config_i["repository_authority"]["base_commit"] == ADJ.REPAIR_BASE_COMMIT
    assert config_i["repository_authority"]["implementation_commit_expected_parent"] == ADJ.REPAIR_BASE_COMMIT
    assert ADJ.REPAIR_BASE_COMMIT == "f4922af6dfcd6e8b63064fe8d819edb3971da1fb"
    assert ADJ.REPAIR_BASE_COMMIT != ADJ.HISTORICAL_BINDING_COMMIT
    assert ADJ.BINDING_CONFIG_REPO_PATHS == (ADJ.CONFIG_REPO_PATH,)
    assert ADJ.EXPECTED_IMPLEMENTATION_FILES == {
        ADJ.CONFIG_REPO_PATH: (ADJ.SCRIPT_REPO_PATH, ADJ.TEST_REPO_PATH)
    }
    assert config_i["repository_authority"][
        "implementation_commit_exact_changed_paths"
    ] == [ADJ.CONFIG_REPO_PATH, ADJ.SCRIPT_REPO_PATH, ADJ.TEST_REPO_PATH]
    assert config_i["repository_authority"][
        "binding_commit_exact_changed_paths"
    ] == [ADJ.CONFIG_REPO_PATH]
    assert current_config["evidence_contract"]["evidence_schema_version"] == ADJ.EVIDENCE_SCHEMA_VERSION
    assert ADJ.EVIDENCE_SCHEMA_VERSION.endswith(".v3")
    assert ADJ.EVIDENCE_RECORD_TYPE.endswith("_V3")
    state = config_i["current_external_state"]
    assert len(state["unresolved_blockers"]) == 8
    assert (state["qualified"], state["ordinary_study_contribution"], state["a1_study_contribution"], state["true_a2_study_contribution"], state["canonical_record_count"]) == (False, 0, 0, 0, 0)
    assert state["training_allowed"] is state["model_selection_allowed"] is state["next_phase_authorized"] is False


def test_current_disk_i_or_b_preserves_current_bound_descriptor_state() -> None:
    current = json.loads(CONFIG.read_text(encoding="utf-8"))
    ADJ.validate_static_config(current)
    current_binding = current["implementation_binding"]
    if current_binding["status"] == "BOUND":
        ADJ.validate_implementation_binding(current)
    else:
        assert current_binding["status"] == ADJ.UNKNOWN

    config_i = unknown_implementation(current)
    ADJ.validate_static_config(config_i)
    binding = config_i["implementation_binding"]
    assert binding["status"] == ADJ.UNKNOWN
    assert {
        binding[key]
        for key in (
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    } == {ADJ.UNKNOWN}
    descriptors = config_i["evidence_descriptor_bindings"]
    assert descriptors["status"] == "BOUND"
    assert descriptors["descriptor_set_sha256"] == ADJ.descriptor_set_sha256(config_i)
    assert ADJ._derived_descriptor_status(config_i) == "BOUND"
    assert all(ADJ._descriptor_slot_bound(slot) for slot in descriptors["slots"])

    if current_binding["status"] == "BOUND":
        config_b = bind_implementation(
            config_i,
            implementation_commit=current_binding["implementation_commit"],
            script_sha256=current_binding["implementation_script_sha256"],
            test_sha256=current_binding["implementation_test_sha256"],
        )
        assert config_b == current
    else:
        config_b = bind_implementation(config_i)
    ADJ.validate_static_config(config_b)
    ADJ._validate_i_to_b_config_pair(
        config_i,
        config_b,
        config_path=ADJ.CONFIG_REPO_PATH,
        implementation_commit=config_b["implementation_binding"]["implementation_commit"],
    )
    assert config_b["evidence_descriptor_bindings"] == descriptors


def test_other_seven_gate_semantics_are_unchanged_under_v3_record_identity() -> None:
    expected_fact_keys = {
        "CANONICAL_REPORTED_ENDPOINT_SEMANTICS": {
            "author_published_processed_endpoint_is_primary",
            "endpoint_id_frozen",
            "endpoint_direction_frozen",
            "endpoint_scale_frozen",
            "contrast_and_transform_frozen",
            "paper_faithful_mapping_closed",
        },
        "BIOLOGICAL_GROUP_AUTHORITY": {
            "biological_group_id_frozen",
            "study_unit_is_gse200304",
            "gse200302_is_subseries_not_independent_study",
            "group_mapping_hash_bound",
        },
        "ROW_REPLICATE_OR_VALID_SE": {
            "replicate_or_valid_standard_error_present",
            "replicate_count_or_effective_n_frozen",
            "standard_error_semantics_frozen",
            "technical_uncertainty_not_substituted_for_biological_se",
        },
        "CHECKPOINT_SPECIFIC_EXPOSURE": {
            "checkpoint_ids_and_revisions_frozen",
            "checkpoint_artifact_digests_bound",
            "exact_member_exposure_audit_pass",
            "near_duplicate_exposure_audit_pass",
            "audited_checkpoint_count",
        },
        "LICENSE_RIGHTS": {
            "rights_source_authority_closed",
            "qualification_use_allowed",
            "private_canonical_materialization_allowed",
            "redistribution_scope",
        },
        "OUTCOME_BLIND_SPLIT_LEAKAGE": {
            "a1_source_graph_frozen",
            "a1_group_graph_frozen",
            "a1_near_duplicate_graph_frozen",
            "split_salt_hash_bound",
            "outcome_blind_assignment",
            "leakage_audit_pass",
            "final_benchmark_membership_deferred_to_a2",
        },
        "PREFROZEN_POWER_PRECISION": {
            "analysis_unit",
            "bootstrap_unit",
            "observed_power",
            "full_confidence_interval_width",
            "prefrozen_before_model_results",
        },
    }
    assert {slot_id: ADJ.FACT_KEYS[slot_id] for slot_id in ADJ.SLOT_IDS[1:]} == (
        expected_fact_keys
    )

    config = bind_implementation()
    facts = passing_facts()
    for slot_id in ADJ.SLOT_IDS[1:]:
        ADJ._validate_fact_types(slot_id, facts[slot_id])
        assert ADJ._slot_gate_pass(slot_id, facts[slot_id]) is True
        record_v3 = gate_record(config, slot_id, facts=facts[slot_id])
        record_v2_identity = copy.deepcopy(record_v3)
        record_v2_identity["schema_version"] = (
            "route_a_v3_dec019_aggregate_gate_evidence.v2"
        )
        record_v2_identity["record_type"] = (
            "ROUTE_A_V3_DEC019_ACCEPTED_AGGREGATE_GATE_EVIDENCE_V2"
        )
        record_v2_identity["provenance"]["acceptance_authority"]["rule"] = (
            "CONFIG_HASH_BOUND_ACCEPTED_AGGREGATE_GATE_RECORD_V2"
        )
        assert ADJ._semantic_diff_paths(record_v2_identity, record_v3) == {
            "schema_version",
            "record_type",
            "provenance.acceptance_authority.rule",
        }


def test_i_to_b_changes_exactly_four_binding_scalars() -> None:
    config_i = unknown_implementation(read_config())
    config_b = bind_implementation(config_i)
    ADJ._validate_i_to_b_config_pair(
        config_i,
        config_b,
        config_path=ADJ.CONFIG_REPO_PATH,
        implementation_commit=SYNTHETIC_IMPLEMENTATION_COMMIT,
    )
    assert ADJ._semantic_diff_paths(config_i, config_b) == set(ADJ.EXPECTED_I_TO_B_SCALAR_PATHS)


def test_descriptor_values_do_not_change_science_core_but_science_rehash_forgery_fails() -> None:
    config = bind_implementation()
    original_core = ADJ.config_core_sha256(config)
    descriptor = config["evidence_descriptor_bindings"]["slots"][0]
    descriptor.update({"absolute_path": "/tmp/synthetic.json", "sha256": "6" * 64, "bytes": 10})
    refresh_descriptor_hash(config)
    assert ADJ.config_core_sha256(config) == original_core
    ADJ.validate_static_config(config)

    forged = copy.deepcopy(config)
    forged["policy_boundary"]["minimum_power"] = 0.81
    refresh_science_core(forged)
    with pytest.raises(ADJ.AdjudicationError, match="compiled authority"):
        ADJ.validate_static_config(forged)
    with pytest.raises(ADJ.BindingError, match="science core"):
        ADJ._validate_descriptor_only_transition(config, forged, config_path=ADJ.CONFIG_REPO_PATH)


def test_descriptor_triple_and_status_are_closed() -> None:
    config = bind_implementation()
    descriptor = config["evidence_descriptor_bindings"]["slots"][0]
    descriptor["sha256"] = ADJ.UNKNOWN
    refresh_descriptor_hash(config)
    with pytest.raises(ADJ.AdjudicationError, match="partially bound triple"):
        ADJ.validate_static_config(config)


def test_unknown_implementation_stops_before_evidence_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("evidence/output must remain untouched")

    monkeypatch.setattr(ADJ, "_read_verified_evidence", forbidden)
    monkeypatch.setattr(ADJ, "_preflight_output", forbidden)
    with pytest.raises(ADJ.BindingError, match="stopped before evidence input"):
        ADJ.adjudicate(unknown_implementation(read_config()), tmp_path / "never")
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("partial", [False, True])
def test_unbound_or_partial_descriptor_set_reads_zero_and_emits_exact_eight_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial: bool,
) -> None:
    config = bind_implementation()
    descriptors = config["evidence_descriptor_bindings"]
    if partial:
        descriptors["status"] = "PARTIALLY_BOUND"
        descriptors["slots"][0].update(
            {
                "absolute_path": ADJ.UNKNOWN,
                "sha256": ADJ.UNKNOWN,
                "bytes": ADJ.UNKNOWN,
            }
        )
        refresh_descriptor_hash(config)
    else:
        descriptors["status"] = "UNBOUND"
        for descriptor in descriptors["slots"]:
            descriptor.update(
                {
                    "absolute_path": ADJ.UNKNOWN,
                    "sha256": ADJ.UNKNOWN,
                    "bytes": ADJ.UNKNOWN,
                }
            )
        refresh_descriptor_hash(config)
    assert ADJ._derived_descriptor_status(config) == (
        "PARTIALLY_BOUND" if partial else "UNBOUND"
    )
    opened = 0

    def forbidden_read(*_args: Any, **_kwargs: Any) -> bytes:
        nonlocal opened
        opened += 1
        raise AssertionError("no descriptor may open until all eight bind")

    monkeypatch.setattr(ADJ, "_read_verified_evidence", forbidden_read)
    result = ADJ.adjudicate(config, tmp_path / f"blocked-{partial}")
    assert opened == 0
    assert result["blockers"] == config["current_external_state"]["unresolved_blockers"]
    assert (result["qualified"], result["ordinary_study_contribution"], result["a1_study_contribution"], result["true_a2_study_contribution"], result["canonical_record_count"]) == (False, 0, 0, 0, 0)
    audit = json.loads((tmp_path / f"blocked-{partial}" / "INPUT_EVIDENCE_AUDIT.json").read_text())
    assert audit["opened_input_count"] == 0


def test_eight_bound_negative_records_are_accepted_as_blocked_not_zero_facts(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        statuses={slot_id: "BLOCKED" for slot_id in ADJ.SLOT_IDS},
    )
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert result["blockers"] == sorted(
        slot["blocker_if_not_pass"] for slot in config["evidence_contract"]["slots"]
    )
    assert (result["qualified"], result["ordinary_study_contribution"], result["a1_study_contribution"], result["true_a2_study_contribution"], result["canonical_record_count"]) == (False, 0, 0, 0, 0)
    audit = json.loads((tmp_path / "out" / "INPUT_EVIDENCE_AUDIT.json").read_text())
    assert audit["opened_input_count"] == 8
    assert {slot["gate_status"] for slot in audit["slots"]} == {"BLOCKED"}


def test_negative_numeric_zero_cannot_masquerade_as_unknown(tmp_path: Path) -> None:
    config = bind_implementation()
    slot_id = ADJ.SLOT_IDS[0]
    materialize_evidence(
        tmp_path / "evidence",
        config,
        statuses={slot_id: ADJ.UNKNOWN},
        record_updates={slot_id: {"facts": {"canonical_record_count": 0}}},
    )
    with pytest.raises(ADJ.AdjudicationError, match="facts=null|numeric zero"):
        ADJ.adjudicate(config, tmp_path / "out")


def test_negative_unknown_fields_include_commitment_keys_and_are_exact(
    tmp_path: Path,
) -> None:
    config = bind_implementation()
    slot_id = ADJ.SLOT_IDS[0]
    paths = materialize_evidence(
        tmp_path / "evidence",
        config,
        statuses={slot_id: "NOT_RUN"},
    )
    path = paths[slot_id]
    record = json.loads(path.read_text())
    assert record["unknown_fields"] == sorted(ADJ.FACT_KEYS[slot_id])
    assert {
        "locator_lineage_commitment_algorithm",
        "locator_lineage_merkle_root_sha256",
    }.issubset(record["unknown_fields"])

    record["unknown_fields"].remove("locator_lineage_merkle_root_sha256")
    payload = ADJ.json_bytes(record)
    path.write_bytes(payload)
    descriptor = config["evidence_descriptor_bindings"]["slots"][0]
    descriptor.update({"sha256": ADJ.sha256(payload), "bytes": len(payload)})
    refresh_descriptor_hash(config)
    with pytest.raises(ADJ.AdjudicationError, match="negative unknown fields"):
        ADJ.adjudicate(config, tmp_path / "out")


def test_lineage_pass_requires_commitment_root_and_fixed_algorithm(
    tmp_path: Path,
) -> None:
    slot_id = ADJ.SLOT_IDS[0]

    missing_config = bind_implementation()
    missing_facts = passing_facts()[slot_id]
    missing_facts.pop("locator_lineage_merkle_root_sha256")
    materialize_evidence(
        tmp_path / "missing-evidence",
        missing_config,
        record_updates={slot_id: {"facts": missing_facts}},
    )
    with pytest.raises(ADJ.AdjudicationError, match="facts keys differ"):
        ADJ.adjudicate(missing_config, tmp_path / "missing-out")

    nonhex_config = bind_implementation()
    materialize_evidence(
        tmp_path / "nonhex-evidence",
        nonhex_config,
        fact_updates={
            slot_id: {"locator_lineage_merkle_root_sha256": "A" * 64}
        },
    )
    with pytest.raises(ADJ.AdjudicationError, match="lowercase SHA-256"):
        ADJ.adjudicate(nonhex_config, tmp_path / "nonhex-out")

    wrong_algorithm_config = bind_implementation()
    paths = materialize_evidence(
        tmp_path / "algorithm-evidence",
        wrong_algorithm_config,
        fact_updates={
            slot_id: {"locator_lineage_commitment_algorithm": "UNSCOPED_MERKLE_V1"}
        },
    )
    descriptor = wrong_algorithm_config["evidence_descriptor_bindings"]["slots"][0]
    payload = paths[slot_id].read_bytes()
    assert descriptor["sha256"] == ADJ.sha256(payload)
    assert wrong_algorithm_config["evidence_descriptor_bindings"][
        "descriptor_set_sha256"
    ] == ADJ.descriptor_set_sha256(wrong_algorithm_config)
    with pytest.raises(ADJ.AdjudicationError, match="commitment algorithm differs"):
        ADJ.adjudicate(wrong_algorithm_config, tmp_path / "algorithm-out")


def test_merkle_root_changes_canonical_gate_bytes_without_member_or_row_payload() -> None:
    config = bind_implementation()
    slot_id = ADJ.SLOT_IDS[0]
    facts_a = passing_facts()[slot_id]
    facts_b = copy.deepcopy(facts_a)
    facts_b["locator_lineage_merkle_root_sha256"] = "b" * 64
    record_a = gate_record(config, slot_id, facts=facts_a)
    record_b = gate_record(config, slot_id, facts=facts_b)
    payload_a = ADJ.json_bytes(record_a)
    payload_b = ADJ.json_bytes(record_b)

    assert ADJ._semantic_diff_paths(record_a, record_b) == {
        "facts.locator_lineage_merkle_root_sha256"
    }
    assert payload_a != payload_b
    assert ADJ.sha256(payload_a) != ADJ.sha256(payload_b)
    assert record_a["facts"]["locator_lineage_merkle_root_sha256"] == "a" * 64
    assert record_a["privacy"] == privacy()
    assert not {
        "table_s2_sha256",
        "table_s3_sha256",
        "locator_leaf_sha256s",
        "member_hashes",
        "row_values",
    }.intersection(record_a["facts"])


def test_missing_or_nonexact_predecessor_provenance_is_rejected(tmp_path: Path) -> None:
    config = bind_implementation()
    paths = materialize_evidence(tmp_path / "evidence", config)
    slot_id = ADJ.SLOT_IDS[0]
    path = paths[slot_id]
    record = json.loads(path.read_text())
    record["provenance"]["predecessor_authority"]["members"][0]["sha256"] = "0" * 64
    payload = ADJ.json_bytes(record)
    path.write_bytes(payload)
    descriptor = next(slot for slot in config["evidence_descriptor_bindings"]["slots"] if slot["slot_id"] == slot_id)
    descriptor.update({"sha256": ADJ.sha256(payload), "bytes": len(payload)})
    refresh_descriptor_hash(config)
    with pytest.raises(ADJ.AdjudicationError, match="predecessor authority"):
        ADJ.adjudicate(config, tmp_path / "out")


def test_copied_predecessor_target_identity_cannot_self_prove(tmp_path: Path) -> None:
    config = bind_implementation()
    slot_id = ADJ.SLOT_IDS[0]
    materialize_evidence(
        tmp_path / "evidence",
        config,
        record_updates={
            slot_id: {
                "provenance": {
                    **provenance(config),
                    "source_bundle_root_or_target_sha256": "0" * 64,
                }
            }
        },
    )
    with pytest.raises(ADJ.AdjudicationError, match="source target identity"):
        ADJ.adjudicate(config, tmp_path / "out")


def test_v2_acceptance_rule_is_rejected_even_after_synchronized_rehash(
    tmp_path: Path,
) -> None:
    config = bind_implementation()
    slot_id = "CANONICAL_REPORTED_ENDPOINT_SEMANTICS"
    forged_provenance = provenance(config)
    forged_provenance["acceptance_authority"]["rule"] = (
        "CONFIG_HASH_BOUND_ACCEPTED_AGGREGATE_GATE_RECORD_V2"
    )
    paths = materialize_evidence(
        tmp_path / "evidence",
        config,
        record_updates={slot_id: {"provenance": forged_provenance}},
    )
    descriptor = next(
        item
        for item in config["evidence_descriptor_bindings"]["slots"]
        if item["slot_id"] == slot_id
    )
    payload = paths[slot_id].read_bytes()
    assert descriptor["sha256"] == ADJ.sha256(payload)
    assert config["evidence_descriptor_bindings"]["descriptor_set_sha256"] == (
        ADJ.descriptor_set_sha256(config)
    )
    with pytest.raises(ADJ.AdjudicationError, match="acceptance authority"):
        ADJ.adjudicate(config, tmp_path / "out")


def test_pass_without_provenance_is_rejected(tmp_path: Path) -> None:
    config = bind_implementation()
    slot_id = ADJ.SLOT_IDS[0]
    materialize_evidence(
        tmp_path / "evidence",
        config,
        record_updates={slot_id: {"provenance": None}},
    )
    with pytest.raises(ADJ.AdjudicationError, match="provenance"):
        ADJ.adjudicate(config, tmp_path / "out")


def test_all_eight_pass_qualifies_one_dataset_only_and_binds_output_authority(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(tmp_path / "evidence", config)
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert result["status"] == ADJ.SUCCESS_STATUS
    assert (result["qualified"], result["ordinary_study_contribution"], result["a1_study_contribution"], result["true_a2_study_contribution"], result["canonical_record_count"]) == (True, 1, 1, 0, 6120)
    assert result["independent_raw_reproduction_established"] is False
    report = read_report(tmp_path / "out")
    assert report["authority_provenance"]["historical_binding_commit"] == ADJ.HISTORICAL_BINDING_COMMIT
    assert report["authority_provenance"]["predecessor_authority_sha256"] == ADJ.sha256(
        ADJ.json_bytes(config["evidence_contract"]["required_predecessor_authority"])
    )
    assert report["evidence_descriptor_set_sha256"] == config["evidence_descriptor_bindings"]["descriptor_set_sha256"]
    assert report["training_allowed"] is report["model_selection_allowed"] is report["next_phase_authorized"] is False


@pytest.mark.parametrize(
    ("updates", "blocker"),
    [
        ({"observed_power": 0.799}, "POWER_LT_0_80"),
        ({"full_confidence_interval_width": 0.301}, "FULL_CI_WIDTH_GT_0_30"),
    ],
)
def test_power_and_precision_thresholds_remain_separate(
    tmp_path: Path,
    updates: Mapping[str, Any],
    blocker: str,
) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        fact_updates={"PREFROZEN_POWER_PRECISION": updates},
    )
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert blocker in result["blockers"]
    assert result["canonical_record_count"] == 0


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"observed_power": 1.5}, "observed power"),
        ({"observed_power": -0.1}, "observed power"),
        ({"full_confidence_interval_width": -0.2}, "CI width"),
    ],
)
def test_power_and_precision_impossible_domains_are_rejected(
    tmp_path: Path,
    updates: Mapping[str, Any],
    message: str,
) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        fact_updates={"PREFROZEN_POWER_PRECISION": updates},
    )
    with pytest.raises(ADJ.AdjudicationError, match=message):
        ADJ.adjudicate(config, tmp_path / "out")


def test_hash_drift_and_evidence_parent_symlink_are_rejected(tmp_path: Path) -> None:
    config = bind_implementation()
    paths = materialize_evidence(tmp_path / "evidence", config)
    paths["LICENSE_RIGHTS"].write_bytes(paths["LICENSE_RIGHTS"].read_bytes() + b" ")
    with pytest.raises(ADJ.AdjudicationError, match="byte count differs|SHA differs"):
        ADJ.adjudicate(config, tmp_path / "hash-out")

    config = bind_implementation()
    real = tmp_path / "real"
    materialize_evidence(real, config)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    for descriptor in config["evidence_descriptor_bindings"]["slots"]:
        basename = next(slot["allowed_basename"] for slot in config["evidence_contract"]["slots"] if slot["slot_id"] == descriptor["slot_id"])
        descriptor["absolute_path"] = str(linked / basename)
    refresh_descriptor_hash(config)
    with pytest.raises(ADJ.AdjudicationError, match="symlink or non-directory"):
        ADJ.adjudicate(config, tmp_path / "symlink-out")


def test_publisher_is_exact_idempotent_and_partial_marker_is_not_truth(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        statuses={slot_id: "BLOCKED" for slot_id in ADJ.SLOT_IDS},
    )
    output = tmp_path / "out"
    assert ADJ.adjudicate(config, output)["publication_status"] == "PUBLISHED"
    assert ADJ.adjudicate(config, output)["publication_status"] == "EXISTING_EXACT"

    partial = tmp_path / "partial"

    def fail(point: str) -> None:
        if point == "before_commit_marker":
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        ADJ.adjudicate(config, partial, fault_injector=fail)
    assert partial.is_dir() and not (partial / ADJ.COMMIT_MARKER).exists()
    with pytest.raises(ADJ.PartialPublicationError):
        ADJ.inspect_committed_bundle(partial)


def test_publisher_directory_swap_cannot_write_outside_trusted_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "A1"
    trusted.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    output = trusted / "bundle"
    moved = trusted / "moved-partial"
    monkeypatch.setattr(ADJ, "TRUSTED_A1_OUTPUT_ROOT", trusted)
    config = bind_implementation()
    authority = ADJ._synthetic_authority_provenance(config)
    report = ADJ._blocked_report(
        config,
        config["current_external_state"]["unresolved_blockers"],
        authority,
    )
    audit = ADJ._input_audit(config, None, authority)
    payloads = ADJ._build_bundle(config, output, report, audit)
    swapped = False

    def swap_after_first_member(point: str) -> None:
        nonlocal swapped
        if not swapped and point.startswith("after_"):
            swapped = True
            output.rename(moved)
            output.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ADJ.PublicationError, match="identity changed"):
        ADJ._publish_bundle(
            output,
            payloads,
            production=True,
            config=config,
            authority_provenance=authority,
            fault_injector=swap_after_first_member,
        )
    assert swapped is True
    assert list(outside.iterdir()) == []
    assert not (moved / ADJ.COMMIT_MARKER).exists()


def test_publisher_parent_root_rename_swap_is_rejected_before_commit_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "A1"
    trusted.mkdir()
    moved_root = tmp_path / "A1-moved"
    output = trusted / "bundle"
    monkeypatch.setattr(ADJ, "TRUSTED_A1_OUTPUT_ROOT", trusted)
    config = bind_implementation()
    authority = ADJ._synthetic_authority_provenance(config)
    report = ADJ._blocked_report(
        config,
        config["current_external_state"]["unresolved_blockers"],
        authority,
    )
    audit = ADJ._input_audit(config, None, authority)
    payloads = ADJ._build_bundle(config, output, report, audit)
    swapped = False

    def swap_parent_before_marker(point: str) -> None:
        nonlocal swapped
        if not swapped and point == "before_commit_marker":
            swapped = True
            trusted.rename(moved_root)
            trusted.mkdir()

    with pytest.raises(ADJ.PublicationError, match="output parent identity changed"):
        ADJ._publish_bundle(
            output,
            payloads,
            production=True,
            config=config,
            authority_provenance=authority,
            fault_injector=swap_parent_before_marker,
        )
    assert swapped is True
    assert list(trusted.iterdir()) == []
    assert not (moved_root / "bundle" / ADJ.COMMIT_MARKER).exists()


def test_inspector_directory_swap_is_detected_while_reads_stay_fd_anchored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        statuses={slot_id: "BLOCKED" for slot_id in ADJ.SLOT_IDS},
    )
    authority = ADJ._synthetic_authority_provenance(config)
    output = tmp_path / "bundle"
    ADJ.adjudicate(config, output)
    moved = tmp_path / "moved"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_read = ADJ._read_regular_at
    calls = 0

    def swapping_read(directory_fd: int, name: str) -> bytes:
        nonlocal calls
        payload = original_read(directory_fd, name)
        calls += 1
        if calls == 1:
            output.rename(moved)
            output.symlink_to(outside, target_is_directory=True)
        return payload

    monkeypatch.setattr(ADJ, "_read_regular_at", swapping_read)
    with pytest.raises(ADJ.PublicationError, match="identity changed"):
        ADJ.inspect_committed_bundle(
            output,
            config=config,
            expected_authority_provenance=authority,
        )
    assert list(outside.iterdir()) == []


def test_inspector_parent_root_rename_swap_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "A1"
    trusted.mkdir()
    output = trusted / "bundle"
    moved_root = tmp_path / "A1-moved"
    monkeypatch.setattr(ADJ, "TRUSTED_A1_OUTPUT_ROOT", trusted)
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        statuses={slot_id: "BLOCKED" for slot_id in ADJ.SLOT_IDS},
    )
    authority = ADJ._synthetic_authority_provenance(config)
    ADJ.adjudicate(config, output)
    original_read = ADJ._read_regular_at
    calls = 0

    def swapping_read(directory_fd: int, name: str) -> bytes:
        nonlocal calls
        payload = original_read(directory_fd, name)
        calls += 1
        if calls == 1:
            trusted.rename(moved_root)
            trusted.mkdir()
        return payload

    monkeypatch.setattr(ADJ, "_read_regular_at", swapping_read)
    with pytest.raises(ADJ.PublicationError, match="output parent identity changed"):
        ADJ.inspect_committed_bundle(
            output,
            production=True,
            config=config,
            expected_authority_provenance=authority,
        )
    assert list(trusted.iterdir()) == []
    assert (moved_root / "bundle" / ADJ.COMMIT_MARKER).is_file()


def test_inspector_rejects_rehashed_semantically_forged_bundle(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        statuses={slot_id: "BLOCKED" for slot_id in ADJ.SLOT_IDS},
    )
    authority = ADJ._synthetic_authority_provenance(config)
    output = tmp_path / "bundle"
    ADJ.adjudicate(config, output)
    report = json.loads((output / "ADJUDICATION_REPORT.json").read_text())
    report.pop("authority_provenance")
    report["qualified"] = True
    report["canonical_record_count"] = 999999
    report_payload = ADJ.json_bytes(report)
    (output / "ADJUDICATION_REPORT.json").write_bytes(report_payload)
    audit_payload = (output / "INPUT_EVIDENCE_AUDIT.json").read_bytes()
    sums = (
        f"{ADJ.sha256(report_payload)}  ADJUDICATION_REPORT.json\n"
        f"{ADJ.sha256(audit_payload)}  INPUT_EVIDENCE_AUDIT.json\n"
    ).encode("ascii")
    (output / "SHA256SUMS").write_bytes(sums)
    marker = json.loads((output / ADJ.COMMIT_MARKER).read_text())
    marker["sha256sums_sha256"] = ADJ.sha256(sums)
    (output / ADJ.COMMIT_MARKER).write_bytes(ADJ.json_bytes(marker))
    with pytest.raises(ADJ.AdjudicationError, match="published report|closed schema"):
        ADJ.inspect_committed_bundle(
            output,
            config=config,
            expected_authority_provenance=authority,
        )


def test_inspector_recomputes_bound_evidence_and_rejects_fully_rehashed_false_success(
    tmp_path: Path,
) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        fact_updates={
            "PREFROZEN_POWER_PRECISION": {"observed_power": 0.79},
        },
    )
    authority = ADJ._synthetic_authority_provenance(config)
    expected_report, audit = ADJ._recompute_adjudication_outputs(config, authority)
    assert expected_report["status"] == ADJ.BLOCKED_STATUS
    assert "POWER_LT_0_80" in expected_report["blockers"]
    assert {slot["gate_status"] for slot in audit["slots"]} == {"PASS"}

    output = tmp_path / "forged-bundle"
    forged_report = ADJ._success_report(
        config,
        6120,
        False,
        authority,
    )
    payloads = ADJ._build_bundle(config, output, forged_report, audit)
    output.mkdir()
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)

    with pytest.raises(ADJ.PublicationError, match="recomputed adjudication"):
        ADJ.inspect_committed_bundle(
            output,
            config=config,
            expected_authority_provenance=authority,
        )


def test_inspector_rejects_rehashed_blocked_report_with_success_values(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        statuses={slot_id: "BLOCKED" for slot_id in ADJ.SLOT_IDS},
    )
    authority = ADJ._synthetic_authority_provenance(config)
    output = tmp_path / "bundle"
    ADJ.adjudicate(config, output)
    report = json.loads((output / "ADJUDICATION_REPORT.json").read_text())
    report["qualified"] = True
    report["canonical_record_count"] = 999999
    report_payload = ADJ.json_bytes(report)
    (output / "ADJUDICATION_REPORT.json").write_bytes(report_payload)
    audit_payload = (output / "INPUT_EVIDENCE_AUDIT.json").read_bytes()
    sums = (
        f"{ADJ.sha256(report_payload)}  ADJUDICATION_REPORT.json\n"
        f"{ADJ.sha256(audit_payload)}  INPUT_EVIDENCE_AUDIT.json\n"
    ).encode("ascii")
    (output / "SHA256SUMS").write_bytes(sums)
    marker = json.loads((output / ADJ.COMMIT_MARKER).read_text())
    marker["sha256sums_sha256"] = ADJ.sha256(sums)
    (output / ADJ.COMMIT_MARKER).write_bytes(ADJ.json_bytes(marker))
    with pytest.raises(ADJ.AdjudicationError, match="blocked report qualified"):
        ADJ.inspect_committed_bundle(
            output,
            config=config,
            expected_authority_provenance=authority,
        )


def test_inspector_rejects_rehashed_blocked_report_claiming_reproduction(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        statuses={slot_id: "BLOCKED" for slot_id in ADJ.SLOT_IDS},
    )
    authority = ADJ._synthetic_authority_provenance(config)
    output = tmp_path / "bundle"
    ADJ.adjudicate(config, output)
    report = json.loads((output / "ADJUDICATION_REPORT.json").read_text())
    report["independent_raw_reproduction_established"] = True
    report_payload = ADJ.json_bytes(report)
    (output / "ADJUDICATION_REPORT.json").write_bytes(report_payload)
    audit_payload = (output / "INPUT_EVIDENCE_AUDIT.json").read_bytes()
    sums = (
        f"{ADJ.sha256(report_payload)}  ADJUDICATION_REPORT.json\n"
        f"{ADJ.sha256(audit_payload)}  INPUT_EVIDENCE_AUDIT.json\n"
    ).encode("ascii")
    (output / "SHA256SUMS").write_bytes(sums)
    marker = json.loads((output / ADJ.COMMIT_MARKER).read_text())
    marker["sha256sums_sha256"] = ADJ.sha256(sums)
    (output / ADJ.COMMIT_MARKER).write_bytes(ADJ.json_bytes(marker))
    with pytest.raises(
        ADJ.AdjudicationError,
        match="blocked report independent_raw_reproduction_established",
    ):
        ADJ.inspect_committed_bundle(
            output,
            config=config,
            expected_authority_provenance=authority,
        )


def test_production_inspect_requires_direct_child_of_trusted_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = tmp_path / "A1"
    trusted.mkdir()
    monkeypatch.setattr(ADJ, "TRUSTED_A1_OUTPUT_ROOT", trusted)
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        statuses={slot_id: "BLOCKED" for slot_id in ADJ.SLOT_IDS},
    )
    output = trusted / "bundle"
    ADJ.adjudicate(config, output)
    authority = ADJ._synthetic_authority_provenance(config)
    assert ADJ.inspect_committed_bundle(
        output,
        production=True,
        config=config,
        expected_authority_provenance=authority,
    )["publication_status"] == "COMMITTED_EXACT"

    nested = trusted / "nested"
    nested.mkdir()
    with pytest.raises(ADJ.ScopeViolation, match="direct child"):
        ADJ.inspect_committed_bundle(
            nested / "bundle",
            production=True,
            config=config,
            expected_authority_provenance=authority,
        )


@pytest.mark.parametrize("lifecycle", ["I", "B_DESCENDANT"])
def test_validate_authority_cli_mode_touches_no_evidence_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: str,
) -> None:
    config, _ = _authority_fixture(tmp_path, monkeypatch, lifecycle=lifecycle)
    monkeypatch.setattr(ADJ, "load_production_config", lambda: config)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("authority-only mode touched evidence/output")

    monkeypatch.setattr(ADJ, "_read_verified_evidence", forbidden)
    monkeypatch.setattr(ADJ, "_preflight_output", forbidden)
    monkeypatch.setattr(ADJ, "_publish_bundle", forbidden)
    assert ADJ.main(["--validate-authority"]) == 0


def _authority_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lifecycle: str,
    base_ancestor_ok: bool = True,
    invalid_implementation_changed_paths: bool = False,
    invalid_binding_changed_paths: bool = False,
    drift_successor_i_descriptor: bool = False,
    drift_successor_i_core_authority: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    repo = tmp_path / f"repo-{lifecycle}"
    repo.mkdir()
    branch = "routea-v3-a1-20260810"
    repair_base = ADJ.REPAIR_BASE_COMMIT
    implementation = SYNTHETIC_IMPLEMENTATION_COMMIT
    binding_commit = SYNTHETIC_BINDING_COMMIT
    descendant = SYNTHETIC_DESCENDANT_COMMIT
    head = implementation if lifecycle == "I" else descendant
    script_payload = b"synthetic successor G200 script\n"
    test_payload = b"synthetic successor G200 test\n"

    config_i = unknown_implementation(read_config())
    repository = config_i["repository_authority"]
    repository["production_repo_root"] = str(repo)
    core_authority = config_i["core_authority"]
    for path_key, digest_key in (
        ("root_contract_path", "root_contract_sha256"),
        ("amendment_path", "amendment_sha256"),
        ("decision_log_path", "decision_log_sha256"),
        ("data_role_registry_path", "data_role_registry_sha256"),
        ("split_registry_path", "split_registry_sha256"),
        ("task_registry_path", "task_registry_sha256"),
        ("task_split_matrix_path", "task_split_matrix_sha256"),
        ("claim_evidence_matrix_path", "claim_evidence_matrix_sha256"),
        ("a1_qualification_path", "a1_qualification_sha256"),
    ):
        relative_path = core_authority[path_key]
        payload = f"synthetic core authority: {relative_path}\n".encode()
        target = repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        core_authority[digest_key] = ADJ.sha256(payload)

    base_config = copy.deepcopy(config_i)
    base_config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "6d103877bbfb8e1196bfc22890bb239dcb87c3c8",
            "implementation_script_sha256": (
                "9cd4411fcb02e1feed913b799296351e38ab9071b9506611318645e41b8dbbfe"
            ),
            "implementation_test_sha256": (
                "8e7b188cfa2e5015fa307acad980f9ff2f45145943384fcadb50d67b1263e1db"
            ),
        }
    )
    base_config["repository_authority"].update(
        {
            "base_commit": ADJ.PREDECESSOR_I3_COMMIT,
            "implementation_commit_expected_parent": ADJ.PREDECESSOR_I3_COMMIT,
        }
    )
    base_config["evidence_contract"]["gate_record_provenance_contract"].pop(
        "biological_group_pass_requires_mapping_commitment_sha256"
    )
    base_config["implementation_binding"]["config_core_sha256"] = (
        ADJ.config_core_sha256(base_config)
    )

    if drift_successor_i_descriptor:
        config_i["evidence_descriptor_bindings"]["slots"][1].update(
            {
                "absolute_path": "/synthetic/successor-i-descriptor-drift.json",
                "sha256": "b" * 64,
                "bytes": 2,
            }
        )
        refresh_descriptor_hash(config_i)
    if drift_successor_i_core_authority:
        authority_path = core_authority["root_contract_path"]
        authority_payload = b"synthetic synchronized successor-I core drift\n"
        (repo / authority_path).write_bytes(authority_payload)
        core_authority["root_contract_sha256"] = ADJ.sha256(authority_payload)

    frozen_core = refresh_science_core(config_i)
    config_b = bind_implementation(
        config_i,
        implementation_commit=implementation,
        script_sha256=ADJ.sha256(script_payload),
        test_sha256=ADJ.sha256(test_payload),
    )
    config_current = config_i if lifecycle == "I" else config_b

    monkeypatch.setattr(ADJ, "PRODUCTION_REPO_ROOT", repo)
    monkeypatch.setattr(
        ADJ,
        "FROZEN_CONFIG_CORE_SHA256_BY_PATH",
        {ADJ.GSE200304_CONFIG_REPO_PATH: frozen_core},
    )
    monkeypatch.setattr(ADJ, "FROZEN_CONFIG_CORE_SHA256", frozen_core)
    current_path = repo / ADJ.CONFIG_REPO_PATH
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_bytes(ADJ.json_bytes(config_current))
    expected_i_paths = repository["implementation_commit_exact_changed_paths"]

    def fake_git(_repo: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return head
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return branch
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", f"refs/remotes/origin/{branch}"):
            return head
        parents = {
            f"{implementation}^": repair_base,
            f"{binding_commit}^": implementation,
            f"{descendant}^": binding_commit,
        }
        if len(args) == 2 and args[0] == "rev-parse" and args[1] in parents:
            return parents[args[1]]
        if len(args) == 2 and args[0] == "rev-parse":
            return args[1]
        if args[:2] == ("merge-base", "--is-ancestor"):
            if not base_ancestor_ok and args[2] == repair_base and args[3] == head:
                raise ADJ.BindingError("synthetic non-ancestor")
            return ""
        if args[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            commit = args[4]
            if commit == implementation:
                if invalid_implementation_changed_paths:
                    return f"{ADJ.CONFIG_REPO_PATH}\n{ADJ.SCRIPT_REPO_PATH}"
                return "\n".join(expected_i_paths)
            if commit == binding_commit:
                if invalid_binding_changed_paths:
                    return f"{ADJ.CONFIG_REPO_PATH}\n{ADJ.SCRIPT_REPO_PATH}"
                return "\n".join(ADJ.BINDING_CONFIG_REPO_PATHS)
            if commit == descendant:
                return "docs/execution/unrelated_runtime_event.json"
        if args[:4] == ("rev-list", "--parents", "-n", "1"):
            commit = args[4]
            parent_by_commit = {
                implementation: repair_base,
                binding_commit: implementation,
                descendant: binding_commit,
            }
            return f"{commit} {parent_by_commit[commit]}"
        if args[:3] == ("rev-list", "--ancestry-path", "--reverse"):
            if args[3] == f"{implementation}..{head}":
                return f"{binding_commit}\n{descendant}" if lifecycle != "I" else ""
            if args[3] == f"{binding_commit}..{head}":
                return descendant if lifecycle != "I" else ""
        raise AssertionError(f"unexpected git call: {args!r}")

    def fake_git_bytes(_repo: Path, *args: str) -> bytes:
        assert args[0] == "show"
        commit, path = args[1].split(":", 1)
        if path == ADJ.CONFIG_REPO_PATH:
            if commit == repair_base:
                return ADJ.json_bytes(base_config)
            if commit == implementation:
                return ADJ.json_bytes(config_i)
            return ADJ.json_bytes(config_b)
        payloads = {
            ADJ.SCRIPT_REPO_PATH: script_payload,
            ADJ.TEST_REPO_PATH: test_payload,
        }
        return payloads[path]

    monkeypatch.setattr(ADJ, "_git", fake_git)
    monkeypatch.setattr(ADJ, "_git_bytes", fake_git_bytes)
    return config_current, {
        "head": head,
        "implementation": implementation,
        "binding": binding_commit,
        "repair_base": repair_base,
    }

@pytest.mark.parametrize("lifecycle", ["I", "B_DESCENDANT"])
def test_production_authority_supports_exact_i_and_bound_descendant_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: str,
) -> None:
    config, commits = _authority_fixture(tmp_path, monkeypatch, lifecycle=lifecycle)
    result = ADJ.validate_production_authority(config)
    assert result["repair_base_commit"] == commits["repair_base"]
    assert result["historical_binding_commit"] == ADJ.HISTORICAL_BINDING_COMMIT
    if lifecycle == "I":
        assert result["lifecycle_state"] == "REPAIR_I_IMPLEMENTATION_UNBOUND"
        assert result["repair_binding_commit"] == ADJ.UNKNOWN
    else:
        assert result["lifecycle_state"] == "REPAIR_B_BOUND_OR_DESCRIPTOR_DESCENDANT"
        assert result["repair_binding_commit"] == commits["binding"]
        assert result["current_head"] == commits["head"]


def test_production_authority_rejects_nonancestor_f492_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _authority_fixture(
        tmp_path,
        monkeypatch,
        lifecycle="B_DESCENDANT",
        base_ancestor_ok=False,
    )
    with pytest.raises(ADJ.BindingError, match="non-ancestor"):
        ADJ.validate_production_authority(config)


@pytest.mark.parametrize(
    ("invalid_implementation_changed_paths", "invalid_binding_changed_paths", "match"),
    [
        (True, False, "exact three-file"),
        (False, True, "exact config-only"),
    ],
)
def test_production_authority_rejects_nonexact_i_or_b_changed_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_implementation_changed_paths: bool,
    invalid_binding_changed_paths: bool,
    match: str,
) -> None:
    config, _ = _authority_fixture(
        tmp_path,
        monkeypatch,
        lifecycle="B_DESCENDANT",
        invalid_implementation_changed_paths=invalid_implementation_changed_paths,
        invalid_binding_changed_paths=invalid_binding_changed_paths,
    )
    with pytest.raises(ADJ.BindingError, match=match):
        ADJ.validate_production_authority(config)


@pytest.mark.parametrize("lifecycle", ["I", "B_DESCENDANT"])
def test_production_authority_rejects_synchronized_successor_i_descriptor_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: str,
) -> None:
    config, _ = _authority_fixture(
        tmp_path,
        monkeypatch,
        lifecycle=lifecycle,
        drift_successor_i_descriptor=True,
    )
    with pytest.raises(ADJ.BindingError, match="descriptor binding drifted"):
        ADJ.validate_production_authority(config)

    core_drift_root = tmp_path / "synchronized-core-authority-drift"
    core_drift_root.mkdir()
    core_drift_config, _ = _authority_fixture(
        core_drift_root,
        monkeypatch,
        lifecycle=lifecycle,
        drift_successor_i_core_authority=True,
    )
    with pytest.raises(ADJ.BindingError, match="exact allowlist"):
        ADJ.validate_production_authority(core_drift_config)
