from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_gse114002_dec019_true_a2.py"
CONFIG = ROOT / "configs/route_a_v3_gse114002_dec019_true_a2_activation_v2.json"
SPEC = importlib.util.spec_from_file_location("gse114002_dec019_adjudicator", SCRIPT)
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


def legacy_geometry() -> dict[str, Any]:
    return {
        "contract_id": ADJ.CONTRACT_ID,
        "protocol_id": "ROUTE_A_V3_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2",
        "dataset_id": ADJ.DATASET_ID,
        "status": "MECHANICAL_ENDPOINT_RECONCILED_NOT_QUALIFIED",
        "qualified": False,
        "data_role": "A2_RECOVERY_CANDIDATE_NOT_QUALIFIED",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0,
        "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "true_a2_claim_established": False,
        "aggregate_only": True,
        "blockers": [
            "CHECKPOINT_SPECIFIC_EXPOSURE_UNKNOWN_NOT_ASSERTED",
            "FIELD_AND_BIOLOGICAL_SOURCE_AUTHORITY_UNKNOWN_NOT_ASSERTED",
            "FULL_CONSTRUCT_PREFIX_REPORTER_RNA_CHEMISTRY_UNKNOWN_NOT_ASSERTED",
            "LICENSE_AND_REDISTRIBUTION_RIGHTS_UNKNOWN_NOT_ASSERTED",
            "NEAR_DUPLICATE_SPLIT_AND_LEAKAGE_AUDIT_NOT_RUN",
            "OWNER_UNCERTAINTY_POLICY_UNKNOWN_NOT_ASSERTED",
            "PREFROZEN_GROUP_POWER_NOT_RUN",
        ],
        "protocol_provenance": {"aggregate_only": True},
        "source_provenance": {"aggregate_only": True},
        "implementation_binding": {
            "status": "PASS_BOUND_IMPLEMENTATION",
            "verified": True,
            "implementation_commit": "4" * 40,
            "binding_commit": "5" * 40,
        },
    }


def passing_facts() -> dict[str, dict[str, Any]]:
    return {
        "SOURCE_FIELD_AUTHORITY": {
            "field_dictionary_closed": True,
            "mother_join_semantics_closed": True,
            "source_snapshot_hash_bound": True,
            "row_crosswalk_hash_bound": True,
            "complete_design_family_manifest_closed": True,
            "unsafe_ambiguous_fields_excluded_from_join": True,
            "source_anchored_pool_count": 959,
            "canonical_record_count": 3899,
            "minimum_distinct_edited_candidates_per_source": 3,
            "k5_dense_pool_count": 0,
            "k5_used_as_qualification_gate": False,
        },
        "CONSTRUCT_RNA_CHEMISTRY": {
            "full_25nt_prefix_authority_closed": True,
            "reporter_identity_authority_closed": True,
            "designed_sample_rna_chemistry_closed": True,
            "assay_context_id_frozen": True,
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
            "analysis_unit": "BIOLOGICAL_SOURCE_GROUP",
            "bootstrap_unit": "BIOLOGICAL_SOURCE_GROUP",
            "observed_power": 0.8,
            "full_confidence_interval_width": 0.3,
            "prefrozen_before_model_results": True,
            "biological_replicate_status": "ABSENT_BY_DESIGN",
            "paper_standard_error_status": "ABSENT",
            "technical_uncertainty_used_as_biological_standard_error": False,
            "technical_fraction_uncertainty_role": "QC_ONLY_WITHIN_ASSAY_DIAGNOSTIC",
            "technical_fraction_uncertainty_used_for_observed_power": False,
            "technical_fraction_uncertainty_used_for_full_confidence_interval": False,
            "technical_fraction_uncertainty_used_for_equivalence": False,
            "technical_fraction_uncertainty_used_for_confirmatory_evidence": False,
            "technical_fraction_uncertainty_used_for_generalization_evidence": False,
            "uncertainty_basis": "BIOLOGICAL_SOURCE_GROUP_RESAMPLING_WITHOUT_TECHNICAL_FRACTION_UNCERTAINTY",
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
        record = legacy_geometry() if slot_id == "MECHANICAL_ENDPOINT_GEOMETRY" else gate_record(
            slot_id,
            facts[slot_id],
            status=statuses.get(slot_id, "PASS"),
        )
        payload = ADJ.json_bytes(record)
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
        "1c3e4a7aa412e245f6f4680677db60b8241d7873fa126756791bdb0b58f9233a"
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
    mutated = bind_implementation(config_i)
    assert ADJ.config_core_sha256(mutated) == ADJ.config_core_sha256(config_i)
    policy = current["policy_boundary"]
    assert policy["eligible_edit_distances"] == [1, 2, 3]
    assert policy["primary_reporting_k_values"] == [1, 3]
    assert policy["global_reporting_k_values"] == [1, 3, 5]
    assert policy["technical_uncertainty_prohibited_uses"] == [
        "BIOLOGICAL_STANDARD_ERROR",
        "POWER",
        "CONFIDENCE_INTERVAL",
        "EQUIVALENCE",
        "CONFIRMATORY_EVIDENCE",
        "GENERALIZATION_EVIDENCE",
    ]
    assert (
        current["evidence_contract"]["slots"][0]["sha256"]
        == "34c2cc0c861286f8e22bf1ba4026d5f754254ffb1bd10a4b989a9546e874a9c3"
    )


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


def test_unknown_implementation_stops_before_any_evidence_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_read(*_args: Any, **_kwargs: Any) -> bytes:
        nonlocal called
        called = True
        raise AssertionError("evidence must not be read")

    monkeypatch.setattr(ADJ, "_read_verified_evidence", forbidden_read)
    output = tmp_path / "out"
    with pytest.raises(ADJ.BindingError, match="stopped before evidence input"):
        ADJ.adjudicate(unknown_implementation(read_config()), output)
    assert called is False
    assert not output.exists()


def test_bound_but_future_evidence_unbound_publishes_zero_read_blocked_bundle(tmp_path: Path) -> None:
    config = bind_implementation()
    output = tmp_path / "blocked"
    result = ADJ.adjudicate(config, output)
    assert result["status"] == ADJ.BLOCKED_STATUS
    assert (
        result["ordinary_study_contribution"],
        result["a1_study_contribution"],
        result["true_a2_study_contribution"],
        result["canonical_record_count"],
    ) == (0, 0, 0, 0)
    assert result["blockers"] == config["current_external_state"]["unresolved_blockers"]
    audit = json.loads((output / "INPUT_EVIDENCE_AUDIT.json").read_text())
    assert audit["opened_input_count"] == 0
    assert audit["row_level_payload_read_count"] == 0
    assert ADJ.inspect_committed_bundle(output)["scientific_status"] == ADJ.BLOCKED_STATUS


def test_all_gates_pass_activates_exactly_one_ordinary_and_one_true_a2_not_a1(tmp_path: Path) -> None:
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
    ) == (1, 0, 1)
    assert result["canonical_record_count"] == 3899
    assert result["training_allowed"] is False
    report = read_report(output)
    assert report["confirmatory_contribution"] == 0
    assert report["generalization_contribution"] == 0
    assert report["within_assay_development_and_optimization_only"] is True
    assert report["technical_uncertainty_is_biological_standard_error"] is False
    assert report["k5_is_qualification_gate"] is False


@pytest.mark.parametrize(
    ("slot_id", "blocker"),
    [
        ("SOURCE_FIELD_AUTHORITY", "SOURCE_FIELD_AUTHORITY_NOT_PASS"),
        ("CONSTRUCT_RNA_CHEMISTRY", "CONSTRUCT_RNA_CHEMISTRY_NOT_PASS"),
        ("CHECKPOINT_SPECIFIC_EXPOSURE", "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS"),
        ("LICENSE_RIGHTS", "LICENSE_RIGHTS_NOT_PASS"),
        ("OUTCOME_BLIND_SPLIT_LEAKAGE", "OUTCOME_BLIND_SPLIT_LEAKAGE_NOT_PASS"),
        ("PREFROZEN_POWER_PRECISION", "PREFROZEN_POWER_PRECISION_NOT_PASS"),
    ],
)
def test_each_remaining_dec019_gate_independently_blocks_all_credit(
    tmp_path: Path, slot_id: str, blocker: str
) -> None:
    config = bind_implementation()
    materialize_evidence(tmp_path / "evidence", config, statuses={slot_id: "FAIL"})
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert blocker in result["blockers"]
    assert (result["ordinary_study_contribution"], result["a1_study_contribution"], result["true_a2_study_contribution"]) == (0, 0, 0)
    assert result["canonical_record_count"] == 0


@pytest.mark.parametrize(
    ("slot_id", "updates", "blocker"),
    [
        ("SOURCE_FIELD_AUTHORITY", {"minimum_distinct_edited_candidates_per_source": 2}, "SOURCE_ANCHORED_K3_NEIGHBORHOOD_NOT_ESTABLISHED"),
        ("SOURCE_FIELD_AUTHORITY", {"k5_used_as_qualification_gate": True}, "K5_USED_AS_QUALIFICATION_GATE"),
        ("PREFROZEN_POWER_PRECISION", {"analysis_unit": "SOURCE_POOL"}, "PREFROZEN_POWER_PRECISION_NOT_PASS"),
        ("PREFROZEN_POWER_PRECISION", {"bootstrap_unit": "SOURCE_POOL"}, "PREFROZEN_POWER_PRECISION_NOT_PASS"),
        ("PREFROZEN_POWER_PRECISION", {"observed_power": 0.799}, "POWER_LT_0_80"),
        ("PREFROZEN_POWER_PRECISION", {"full_confidence_interval_width": 0.301}, "FULL_CI_WIDTH_GT_0_30"),
        ("PREFROZEN_POWER_PRECISION", {"technical_uncertainty_used_as_biological_standard_error": True}, "TECHNICAL_UNCERTAINTY_MISREPRESENTED_AS_BIOLOGICAL_SE"),
    ],
)
def test_policy_boundaries_are_fail_closed(
    tmp_path: Path, slot_id: str, updates: Mapping[str, Any], blocker: str
) -> None:
    config = bind_implementation()
    materialize_evidence(tmp_path / "evidence", config, fact_updates={slot_id: updates})
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert blocker in result["blockers"]
    assert result["canonical_record_count"] == 0


@pytest.mark.parametrize(
    "usage_field",
    [
        "technical_fraction_uncertainty_used_for_observed_power",
        "technical_fraction_uncertainty_used_for_full_confidence_interval",
        "technical_fraction_uncertainty_used_for_equivalence",
        "technical_fraction_uncertainty_used_for_confirmatory_evidence",
        "technical_fraction_uncertainty_used_for_generalization_evidence",
    ],
)
def test_technical_fraction_uncertainty_is_qc_only_and_cannot_support_claims(
    tmp_path: Path, usage_field: str
) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        fact_updates={"PREFROZEN_POWER_PRECISION": {usage_field: True}},
    )
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert "TECHNICAL_FRACTION_UNCERTAINTY_USED_OUTSIDE_QC" in result["blockers"]
    assert (
        result["ordinary_study_contribution"],
        result["a1_study_contribution"],
        result["true_a2_study_contribution"],
        result["canonical_record_count"],
    ) == (0, 0, 0, 0)


def test_k5_zero_does_not_block_and_pool_count_cannot_multiply_study_credit(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        fact_updates={
            "SOURCE_FIELD_AUTHORITY": {
                "source_anchored_pool_count": 100000,
                "canonical_record_count": 1000000,
                "k5_dense_pool_count": 0,
            }
        },
    )
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert result["qualified"] is True
    assert (result["ordinary_study_contribution"], result["a1_study_contribution"], result["true_a2_study_contribution"]) == (1, 0, 1)


def test_type_strict_rejects_boolean_canonical_count(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(
        tmp_path / "evidence",
        config,
        fact_updates={"SOURCE_FIELD_AUTHORITY": {"canonical_record_count": True}},
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


def test_privacy_schema_rejects_sequence_key_and_outputs_have_no_forbidden_keys(tmp_path: Path) -> None:
    config = bind_implementation()
    paths = materialize_evidence(tmp_path / "evidence", config)
    source_path = paths["SOURCE_FIELD_AUTHORITY"]
    source = json.loads(source_path.read_text())
    source["facts"]["sequence"] = "ACGT" * 8
    payload = ADJ.json_bytes(source)
    source_path.write_bytes(payload)
    slot = next(slot for slot in config["evidence_contract"]["slots"] if slot["slot_id"] == "SOURCE_FIELD_AUTHORITY")
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
        assert "ACGTACGTACGTACGTACGT" not in (output / name).read_text()


def test_publisher_is_idempotent_exact_and_refuses_partial_recovery(tmp_path: Path) -> None:
    config = bind_implementation()
    materialize_evidence(tmp_path / "evidence", config)
    output = tmp_path / "out"
    first = ADJ.adjudicate(config, output)
    second = ADJ.adjudicate(config, output)
    assert first["publication_status"] == "PUBLISHED"
    assert second["publication_status"] == "EXISTING_EXACT"

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
