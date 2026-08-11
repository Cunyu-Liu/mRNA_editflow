from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1.py"
CONFIG = ROOT / "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v2.json"
SPEC = importlib.util.spec_from_file_location("gse200304_dec019_adjudicator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADJ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADJ)

TEST_CONFIG_STATE_ENV = "ROUTE_A_V3_DEC019_TEST_CONFIG_STATE"
IMPLEMENTATION_DYNAMIC_KEYS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)


def unknown_implementation(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    binding = value["implementation_binding"]
    for key in IMPLEMENTATION_DYNAMIC_KEYS:
        binding[key] = ADJ.UNKNOWN
    return value


def _synthetic_bound_implementation(config: Mapping[str, Any]) -> dict[str, Any]:
    value = unknown_implementation(config)
    binding = value["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = "2" * 64
    binding["implementation_test_sha256"] = "3" * 64
    return value


def read_config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    override = os.environ.get(TEST_CONFIG_STATE_ENV)
    if override is None:
        return value
    if override == "SYNTHETIC_BOUND":
        return _synthetic_bound_implementation(value)
    raise AssertionError(f"unsupported {TEST_CONFIG_STATE_ENV}: {override}")


def refresh_core(config: dict[str, Any]) -> None:
    config["implementation_binding"]["config_core_sha256"] = ADJ.config_core_sha256(config)


def bind_implementation(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = _synthetic_bound_implementation(
        config if config is not None else read_config()
    )
    refresh_core(value)
    return value


def assert_exact_unknown_implementation(config: Mapping[str, Any]) -> None:
    binding = config["implementation_binding"]
    assert {key: binding[key] for key in IMPLEMENTATION_DYNAMIC_KEYS} == {
        key: ADJ.UNKNOWN for key in IMPLEMENTATION_DYNAMIC_KEYS
    }


def privacy() -> dict[str, bool]:
    return {
        "contains_row_level_payload": False,
        "contains_sequence": False,
        "contains_row_identifier": False,
        "contains_raw_label_or_effect": False,
        "contains_member_identifiers_or_hashes": False,
    }


def passing_facts() -> dict[str, dict[str, Any]]:
    return {
        "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE": {
            "deterministic_row_locator_frozen": True,
            "table_s2_hash_bound": True,
            "table_s3_hash_bound": True,
            "s2_s3_join_rule_frozen": True,
            "multi_asset_lineage_closed": True,
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


def gate_record(slot_id: str, facts: Mapping[str, Any], *, status: str = "PASS") -> dict[str, Any]:
    return {
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
        "facts": copy.deepcopy(dict(facts)),
    }


def materialize_evidence(
    root: Path,
    config: dict[str, Any],
    *,
    statuses: Mapping[str, str] | None = None,
    fact_updates: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    root.mkdir()
    statuses = statuses or {}
    facts = passing_facts()
    for slot_id, updates in (fact_updates or {}).items():
        facts[slot_id].update(updates)
    paths: dict[str, Path] = {}
    for slot in config["evidence_contract"]["slots"]:
        slot_id = slot["slot_id"]
        payload = ADJ.json_bytes(
            gate_record(slot_id, facts[slot_id], status=statuses.get(slot_id, "PASS"))
        )
        path = root / slot["allowed_basename"]
        path.write_bytes(payload)
        slot["absolute_path"] = str(path)
        slot["sha256"] = ADJ.sha256(payload)
        slot["bytes"] = len(payload)
        paths[slot_id] = path
    refresh_core(config)
    return paths


def read_report(output: Path) -> dict[str, Any]:
    return json.loads((output / "ADJUDICATION_REPORT.json").read_text(encoding="utf-8"))


def walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key.casefold())
            keys.update(walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(walk_keys(child))
    return keys


def test_static_current_config_accepts_real_i_or_b_and_builds_exact_i_fixture() -> None:
    current = read_config()
    ADJ.validate_static_config(current)
    assert current["implementation_binding"]["config_core_sha256"] == (
        "6cbc215d38adf3b3d15de314f674b2ae02b2f1a1a733cb4dec3d75d8f9480943"
    )
    assert ADJ.config_core_sha256(current) == ADJ.FROZEN_CONFIG_CORE_SHA256
    if current["implementation_binding"]["status"] == ADJ.UNKNOWN:
        assert_exact_unknown_implementation(current)
    else:
        assert current["implementation_binding"]["status"] == "BOUND"
        ADJ.validate_implementation_binding(current)

    config_i = unknown_implementation(current)
    assert_exact_unknown_implementation(config_i)
    ADJ.validate_static_config(config_i)
    assert len(current["current_external_state"]["unresolved_blockers"]) == 8
    assert [slot["slot_id"] for slot in current["evidence_contract"]["slots"]] == list(ADJ.SLOT_IDS)
    assert ADJ.config_core_sha256(bind_implementation(config_i)) == ADJ.config_core_sha256(config_i)
    policy = current["policy_boundary"]
    assert policy["raw_replay_not_run_blocks_a1_qualification"] is False
    assert policy["raw_replay_not_run_allows_independent_reproduction_claim"] is False
    assert policy["maximum_study_contribution_per_dataset"] == 1
    assert policy["gsm_pool_subseries_modality_endpoint_replicate_may_multiply_study_count"] is False


def test_exact_i_to_b_pair_accepts_only_the_four_dynamic_scalars() -> None:
    config_i = unknown_implementation(read_config())
    config_b = bind_implementation(config_i)
    ADJ._validate_i_to_b_config_pair(
        config_i,
        config_b,
        config_path=ADJ.CONFIG_REPO_PATH,
        implementation_commit="1" * 40,
    )


def test_synchronized_rehashed_i_and_b_scientific_core_forgery_is_rejected() -> None:
    forged_i = unknown_implementation(read_config())
    forged_i["policy_boundary"]["minimum_power"] = 0.81
    refresh_core(forged_i)
    forged_b = bind_implementation(forged_i)
    with pytest.raises(ADJ.BindingError, match="frozen core"):
        ADJ._validate_i_to_b_config_pair(
            forged_i,
            forged_b,
            config_path=ADJ.CONFIG_REPO_PATH,
            implementation_commit="1" * 40,
        )


def test_parent_i_core_drift_is_rejected_even_after_rehash() -> None:
    drifted_i = unknown_implementation(read_config())
    drifted_i["policy_boundary"]["minimum_power"] = 0.81
    refresh_core(drifted_i)
    config_b = bind_implementation(unknown_implementation(read_config()))
    with pytest.raises(ADJ.BindingError, match="frozen core"):
        ADJ._validate_i_to_b_config_pair(
            drifted_i,
            config_b,
            config_path=ADJ.CONFIG_REPO_PATH,
            implementation_commit="1" * 40,
        )


def test_extra_implementation_binding_field_is_rejected() -> None:
    config_i = unknown_implementation(read_config())
    config_b = bind_implementation(config_i)
    config_b["implementation_binding"]["extra_binding_claim"] = True
    with pytest.raises(ADJ.BindingError, match="binding schema differs"):
        ADJ._validate_i_to_b_config_pair(
            config_i,
            config_b,
            config_path=ADJ.CONFIG_REPO_PATH,
            implementation_commit="1" * 40,
        )


def test_same_test_lifecycle_runs_from_current_and_synthetic_b() -> None:
    current = read_config()
    synthetic_b = bind_implementation(unknown_implementation(current))
    for starting_config in (current, synthetic_b):
        ADJ.validate_static_config(starting_config)
        config_i = unknown_implementation(starting_config)
        assert_exact_unknown_implementation(config_i)
        config_b = bind_implementation(config_i)
        ADJ._validate_i_to_b_config_pair(
            config_i,
            config_b,
            config_path=ADJ.CONFIG_REPO_PATH,
            implementation_commit="1" * 40,
        )


def test_unknown_implementation_stops_before_evidence_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_read(*_args: Any, **_kwargs: Any) -> bytes:
        nonlocal called
        called = True
        raise AssertionError("must not read")

    monkeypatch.setattr(ADJ, "_read_verified_evidence", forbidden_read)
    output = tmp_path / "out"
    with pytest.raises(ADJ.BindingError, match="stopped before evidence input"):
        ADJ.adjudicate(unknown_implementation(read_config()), output)
    assert called is False
    assert not output.exists()


def test_bound_but_unbound_evidence_commits_zero_read_eight_blocker_bundle(tmp_path: Path) -> None:
    config = bind_implementation()
    output = tmp_path / "blocked"
    result = ADJ.adjudicate(config, output)
    assert result["status"] == ADJ.BLOCKED_STATUS
    assert result["blockers"] == config["current_external_state"]["unresolved_blockers"]
    assert (
        result["ordinary_study_contribution"],
        result["a1_study_contribution"],
        result["true_a2_study_contribution"],
        result["canonical_record_count"],
    ) == (0, 0, 0, 0)
    audit = json.loads((output / "INPUT_EVIDENCE_AUDIT.json").read_text())
    assert audit["opened_input_count"] == 0
    assert audit["row_level_payload_read_count"] == 0


def test_all_eight_gates_pass_with_raw_not_run_qualifies_without_reproduction_claim(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(tmp_path / "evidence", config)
    output = tmp_path / "success"
    result = ADJ.adjudicate(config, output)
    assert result["status"] == ADJ.SUCCESS_STATUS
    assert result["qualified"] is True
    assert (
        result["ordinary_study_contribution"],
        result["a1_study_contribution"],
        result["true_a2_study_contribution"],
    ) == (1, 1, 0)
    assert result["canonical_record_count"] == 6120
    assert result["independent_raw_reproduction_established"] is False
    report = read_report(output)
    assert report["primary_measurement_route"] == "AUTHOR_PUBLISHED_PROCESSED_ENDPOINT"
    assert report["raw_replay_role"] == "REPRODUCIBILITY_AUXILIARY_NOT_QUALIFICATION_PREREQUISITE"
    assert report["training_allowed"] is False
    assert report["model_selection_allowed"] is False
    assert report["next_phase_authorized"] is False


@pytest.mark.parametrize(
    ("slot_id", "blocker"),
    [
        ("CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE", "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE_NOT_PASS"),
        ("CANONICAL_REPORTED_ENDPOINT_SEMANTICS", "CANONICAL_REPORTED_ENDPOINT_SEMANTICS_NOT_PASS"),
        ("BIOLOGICAL_GROUP_AUTHORITY", "BIOLOGICAL_GROUP_AUTHORITY_NOT_PASS"),
        ("ROW_REPLICATE_OR_VALID_SE", "ROW_REPLICATE_OR_VALID_SE_NOT_PASS"),
        ("CHECKPOINT_SPECIFIC_EXPOSURE", "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS"),
        ("LICENSE_RIGHTS", "LICENSE_RIGHTS_NOT_PASS"),
        ("OUTCOME_BLIND_SPLIT_LEAKAGE", "OUTCOME_BLIND_SPLIT_LEAKAGE_NOT_PASS"),
        ("PREFROZEN_POWER_PRECISION", "PREFROZEN_POWER_PRECISION_NOT_PASS"),
    ],
)
def test_each_of_eight_decisive_gates_independently_blocks_all_credit(
    tmp_path: Path, slot_id: str, blocker: str
) -> None:
    config = bind_implementation()
    materialize_evidence(tmp_path / "evidence", config, statuses={slot_id: "FAIL"})
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert blocker in result["blockers"]
    assert (result["ordinary_study_contribution"], result["a1_study_contribution"], result["true_a2_study_contribution"]) == (0, 0, 0)
    assert result["canonical_record_count"] == 0


def test_auxiliary_not_run_cannot_claim_independent_reproduction(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        fact_updates={
            "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE": {
                "raw_replay_status": "NOT_RUN",
                "independent_raw_reproduction_claimed": True,
            }
        },
    )
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert "RAW_REPLAY_INDEPENDENT_REPRODUCTION_CLAIM_INVALID" in result["blockers"]
    assert result["qualified"] is False


def test_completed_auxiliary_raw_replay_may_establish_independent_reproduction(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        fact_updates={
            "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE": {
                "raw_replay_status": "PASS_INDEPENDENT_REPRODUCTION",
                "independent_raw_reproduction_claimed": True,
            }
        },
    )
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert result["qualified"] is True
    assert result["independent_raw_reproduction_established"] is True


@pytest.mark.parametrize(
    ("updates", "blocker"),
    [
        ({"observed_power": 0.799}, "POWER_LT_0_80"),
        ({"full_confidence_interval_width": 0.301}, "FULL_CI_WIDTH_GT_0_30"),
    ],
)
def test_power_and_full_ci_width_thresholds_are_separate_closed_gates(
    tmp_path: Path, updates: Mapping[str, Any], blocker: str
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


def test_pairs_subseries_and_modalities_cannot_multiply_single_dataset_credit(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        fact_updates={
            "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE": {
                "canonical_record_count": 1000000,
                "processed_pair_count": 1000000,
            }
        },
    )
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert result["qualified"] is True
    assert (result["ordinary_study_contribution"], result["a1_study_contribution"], result["true_a2_study_contribution"]) == (1, 1, 0)


def test_type_strict_rejects_boolean_record_count(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        fact_updates={"CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE": {"canonical_record_count": True}},
    )
    with pytest.raises(ADJ.AdjudicationError, match="must be an integer"):
        ADJ.adjudicate(config, tmp_path / "out")


def test_hash_drift_and_parent_symlink_are_rejected(tmp_path: Path) -> None:
    config = bind_implementation()
    paths = materialize_evidence(tmp_path / "evidence", config)
    paths["LICENSE_RIGHTS"].write_bytes(paths["LICENSE_RIGHTS"].read_bytes() + b" ")
    with pytest.raises(ADJ.AdjudicationError, match="byte count differs|SHA differs"):
        ADJ.adjudicate(config, tmp_path / "hash_out")

    config = bind_implementation()
    real_root = tmp_path / "real"
    materialize_evidence(real_root, config)
    symlink_root = tmp_path / "linked"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    for slot in config["evidence_contract"]["slots"]:
        slot["absolute_path"] = str(symlink_root / slot["allowed_basename"])
    refresh_core(config)
    with pytest.raises(ADJ.AdjudicationError, match="symlink or non-directory"):
        ADJ.adjudicate(config, tmp_path / "symlink_out")


def test_privacy_schema_rejects_barcode_and_outputs_are_aggregate_safe(tmp_path: Path) -> None:
    config = bind_implementation()
    paths = materialize_evidence(tmp_path / "evidence", config)
    path = paths["CANONICAL_REPORTED_ENDPOINT_SEMANTICS"]
    record = json.loads(path.read_text())
    record["facts"]["barcode"] = "synthetic-row-identifier"
    payload = ADJ.json_bytes(record)
    path.write_bytes(payload)
    slot = next(slot for slot in config["evidence_contract"]["slots"] if slot["slot_id"] == "CANONICAL_REPORTED_ENDPOINT_SEMANTICS")
    slot["sha256"], slot["bytes"] = ADJ.sha256(payload), len(payload)
    refresh_core(config)
    with pytest.raises(ADJ.AdjudicationError, match="closed schema|forbidden key"):
        ADJ.adjudicate(config, tmp_path / "bad")

    config = bind_implementation()
    materialize_evidence(tmp_path / "evidence_ok", config)
    output = tmp_path / "ok"
    ADJ.adjudicate(config, output)
    forbidden = {key.casefold() for key in config["output_contract"]["forbidden_output_keys"]}
    for name in ADJ.OUTPUT_JSON_NAMES:
        value = json.loads((output / name).read_text())
        assert walk_keys(value).isdisjoint(forbidden)


def test_publisher_is_idempotent_exact_and_preserves_partial_failure(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(tmp_path / "evidence", config)
    output = tmp_path / "out"
    assert ADJ.adjudicate(config, output)["publication_status"] == "PUBLISHED"
    assert ADJ.adjudicate(config, output)["publication_status"] == "EXISTING_EXACT"
    assert ADJ.inspect_committed_bundle(output)["scientific_status"] == ADJ.SUCCESS_STATUS

    partial = tmp_path / "partial"

    def fail_before_marker(point: str) -> None:
        if point == "before_commit_marker":
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        ADJ.adjudicate(config, partial, fault_injector=fail_before_marker)
    assert partial.is_dir()
    assert not (partial / ADJ.COMMIT_MARKER).exists()
    with pytest.raises(ADJ.PartialPublicationError):
        ADJ.adjudicate(config, partial)


def test_full_config_rehashes_across_i_to_b_while_core_projection_does_not() -> None:
    config_i = unknown_implementation(read_config())
    config_b = bind_implementation(config_i)
    assert ADJ.sha256(ADJ.json_bytes(config_i)) != ADJ.sha256(ADJ.json_bytes(config_b))
    assert ADJ.config_core_sha256(config_i) == ADJ.config_core_sha256(config_b)
    assert config_i["repository_authority"]["binding_commit_exact_changed_paths"] == [
        "configs/route_a_v3_gse114002_dec019_true_a2_activation_v2.json",
        "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v2.json",
    ]
