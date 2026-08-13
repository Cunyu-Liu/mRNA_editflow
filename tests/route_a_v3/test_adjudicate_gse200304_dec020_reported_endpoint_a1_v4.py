from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_gse200304_dec020_reported_endpoint_a1_v4.py"
CONFIG = ROOT / "configs/route_a_v3_gse200304_dec020_reported_endpoint_a1_activation_v4.json"
SPEC = importlib.util.spec_from_file_location("gse200304_dec020_adjudicator_v4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADJ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADJ)

SYNTHETIC_IMPLEMENTATION_COMMIT = "9" * 40
SYNTHETIC_SCRIPT_SHA256 = "8" * 64
SYNTHETIC_TEST_SHA256 = "7" * 64
RUNTIME_I1_COMMIT = ADJ.AUTHORITY_RUNTIME_I1_COMMIT
RUNTIME_I_COMMIT = "17d0f570bdfb4bf4a3e5ff34cb1d3aa11a2cccdd"
RUNTIME_B_COMMIT = "fb21121525ca13692a4619115f09e99fd99c122a"
V4_I_COMMIT = "3" * 40
V4_B_COMMIT = "4" * 40
RUNTIME_SCRIPT_SHA256 = "fe4c7f19eecef91b9fd1340f4ea8258e644a6b0139d2880a8a84640bd4721862"
RUNTIME_TEST_SHA256 = "2909a34d6c1b152f91bfe0a336290066dc73bee8411db70814068e288a9378b1"


def read_runtime_i_config() -> dict[str, Any]:
    candidates = (
        ROOT / ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH,
        ROOT.parent
        / "g200_dec020_authority_runtime_sync_staging"
        / ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH,
    )
    for candidate in candidates:
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            value["implementation_binding"].update(
                {
                    "status": ADJ.UNKNOWN,
                    "implementation_commit": ADJ.UNKNOWN,
                    "implementation_script_sha256": ADJ.UNKNOWN,
                    "implementation_test_sha256": ADJ.UNKNOWN,
                }
            )
            return value
    raise AssertionError("authority-runtime config fixture is unavailable")


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def unknown_i_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = copy.deepcopy(dict(config if config is not None else read_config()))
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        value["implementation_binding"][key] = ADJ.UNKNOWN
    return value


def bind_implementation(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = unknown_i_config(config)
    value["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": SYNTHETIC_IMPLEMENTATION_COMMIT,
            "implementation_script_sha256": SYNTHETIC_SCRIPT_SHA256,
            "implementation_test_sha256": SYNTHETIC_TEST_SHA256,
        }
    )
    return value


def bind_runtime_lifecycle(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = copy.deepcopy(dict(config if config is not None else read_config()))
    value["repository_authority"].update(
        {
            "authority_runtime_implementation_commit": RUNTIME_I_COMMIT,
            "authority_runtime_binding_commit": RUNTIME_B_COMMIT,
            "authority_runtime_implementation_script_sha256": RUNTIME_SCRIPT_SHA256,
            "authority_runtime_implementation_test_sha256": RUNTIME_TEST_SHA256,
            "base_commit": RUNTIME_B_COMMIT,
            "implementation_commit_expected_parent": RUNTIME_B_COMMIT,
        }
    )
    return value


def unbind_runtime_lifecycle(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = copy.deepcopy(dict(config if config is not None else read_config()))
    for key in ADJ.AUTHORITY_RUNTIME_DYNAMIC_FIELDS:
        value["repository_authority"][key] = ADJ.UNKNOWN
    return value


def bind_production_v4(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = bind_runtime_lifecycle(config)
    value["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": V4_I_COMMIT,
            "implementation_script_sha256": SYNTHETIC_SCRIPT_SHA256,
            "implementation_test_sha256": SYNTHETIC_TEST_SHA256,
        }
    )
    return value


def unbind_descriptors(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    descriptors = value["evidence_descriptor_bindings"]
    for slot in descriptors["slots"]:
        slot.update(
            {
                "absolute_path": ADJ.UNKNOWN,
                "sha256": ADJ.UNKNOWN,
                "bytes": ADJ.UNKNOWN,
            }
        )
    descriptors["status"] = "UNBOUND"
    descriptors["descriptor_set_sha256"] = ADJ.descriptor_set_sha256(value)
    return value


def privacy() -> dict[str, bool]:
    return {
        "contains_row_level_payload": False,
        "contains_sequence": False,
        "contains_row_identifier": False,
        "contains_raw_label_or_effect": False,
        "contains_member_identifiers_or_hashes": False,
    }


def provenance(config: Mapping[str, Any], slot_id: str) -> dict[str, Any]:
    predecessor = config["evidence_contract"]["required_predecessor_authority"]
    value = {
        "producer_protocol_id": "SYNTHETIC_DEC019_AGGREGATE_GATE_PRODUCER_V1",
        "producer_commit": "4" * 40,
        "producer_script_sha256": "5" * 64,
        "source_bundle_id": predecessor["bundle_id"],
        "source_bundle_root_or_target_sha256": predecessor[
            "terminal_marker_final_output_target_sha256"
        ],
        "predecessor_authority": copy.deepcopy(predecessor),
        "acceptance_authority": copy.deepcopy(ADJ.EVIDENCE_ACCEPTANCE_AUTHORITY),
    }
    if slot_id == "BIOLOGICAL_GROUP_AUTHORITY":
        value[ADJ.GROUP_MAPPING_COMMITMENT_KEY] = "6" * 64
    if slot_id == "OUTCOME_BLIND_SPLIT_LEAKAGE":
        value[ADJ.SPLIT_ASSIGNMENT_COMMITMENT_KEY] = "3" * 64
    return value


def passing_facts() -> dict[str, dict[str, Any]]:
    return {
        "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE": {
            "deterministic_row_locator_frozen": True,
            "table_s2_hash_bound": True,
            "table_s3_hash_bound": True,
            "s2_s3_join_rule_frozen": True,
            "multi_asset_lineage_closed": True,
            "locator_lineage_commitment_algorithm": ADJ.LOCATOR_LINEAGE_COMMITMENT_ALGORITHM,
            "locator_lineage_merkle_root_sha256": "a" * 64,
            "canonical_record_count": 6547,
            "processed_pair_count": 6547,
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
            "analysis_unit": ADJ.POWER_ANALYSIS_UNIT,
            "bootstrap_unit": ADJ.POWER_ANALYSIS_UNIT,
            "evaluation_population": ADJ.POWER_EVALUATION_POPULATION,
            "evaluation_group_count": 6544,
            "target_metric": ADJ.POWER_TARGET_METRIC,
            "alternative_spearman_rho": 0.25,
            "two_sided_alpha": 0.05,
            "power_method": ADJ.POWER_METHOD,
            "working_distribution_assumption": ADJ.POWER_WORKING_DISTRIBUTION_ASSUMPTION,
            "estimated_design_power": 1.0,
            "confidence_level": 0.95,
            "confidence_interval_method": ADJ.POWER_CI_METHOD,
            "planned_full_confidence_interval_width": 0.04613579821079131,
            "prefrozen_before_model_results": True,
        },
    }


def gate_record(
    config: Mapping[str, Any],
    slot_id: str,
    *,
    facts: Mapping[str, Any] | None = None,
    status: str = "PASS",
) -> dict[str, Any]:
    is_pass = status == "PASS"
    return {
        "schema_version": ADJ.EVIDENCE_SCHEMA_VERSION,
        "record_type": ADJ.EVIDENCE_RECORD_TYPE,
        "contract_id": ADJ.CONTRACT_ID,
        "decision_id": ADJ.EVIDENCE_DECISION_ID,
        "dataset_id": ADJ.DATASET_ID,
        "gate_id": slot_id,
        "status": status,
        "accepted": True,
        "aggregate_only": True,
        "privacy": privacy(),
        "provenance": provenance(config, slot_id),
        "facts": copy.deepcopy(dict(facts)) if is_pass and facts is not None else None,
        "unknown_fields": [] if is_pass else sorted(ADJ.FACT_KEYS[slot_id]),
        "reason_codes": [] if is_pass else ["SYNTHETIC_NOT_PASS"],
    }


RecordMutator = Callable[[str, dict[str, Any]], None]


def bind_records(
    tmp_path: Path,
    config: Mapping[str, Any] | None = None,
    *,
    mutator: RecordMutator | None = None,
) -> dict[str, Any]:
    value = bind_implementation(config)
    facts = passing_facts()
    contract_slots = {
        slot["slot_id"]: slot for slot in value["evidence_contract"]["slots"]
    }
    descriptors = {
        slot["slot_id"]: slot
        for slot in value["evidence_descriptor_bindings"]["slots"]
    }
    for slot_id in ADJ.SLOT_IDS:
        record = gate_record(value, slot_id, facts=facts[slot_id])
        if mutator is not None:
            mutator(slot_id, record)
        payload = ADJ.json_bytes(record)
        path = tmp_path / contract_slots[slot_id]["allowed_basename"]
        path.write_bytes(payload)
        descriptors[slot_id].update(
            {
                "absolute_path": str(path),
                "sha256": ADJ.sha256(payload),
                "bytes": len(payload),
            }
        )
    value["evidence_descriptor_bindings"]["status"] = "BOUND"
    value["evidence_descriptor_bindings"]["descriptor_set_sha256"] = (
        ADJ.descriptor_set_sha256(value)
    )
    return value


def read_report(output: Path) -> dict[str, Any]:
    return json.loads((output / "ADJUDICATION_REPORT.json").read_text(encoding="utf-8"))


def read_audit(output: Path) -> dict[str, Any]:
    return json.loads((output / "INPUT_EVIDENCE_AUDIT.json").read_text(encoding="utf-8"))


def publish_with_authority(
    config: Mapping[str, Any], output: Path, authority: Mapping[str, Any]
) -> None:
    report, audit = ADJ.recompute_adjudication_outputs(config, authority)
    assert ADJ._publish_bundle(output, ADJ._build_bundle(config, report, audit)) == (
        "PUBLISHED_NEW"
    )


def install_valid_production_git_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    config: Mapping[str, Any],
) -> tuple[
    dict[str, list[str]],
    dict[tuple[str, str], str],
    dict[tuple[str, ...], str],
]:
    runtime_i_config = read_runtime_i_config()
    runtime_b_config = copy.deepcopy(runtime_i_config)
    runtime_b_config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": RUNTIME_I_COMMIT,
            "implementation_script_sha256": RUNTIME_SCRIPT_SHA256,
            "implementation_test_sha256": RUNTIME_TEST_SHA256,
        }
    )
    v4_i_config = copy.deepcopy(dict(config))
    v4_i_config["implementation_binding"].update(
        {
            "status": ADJ.UNKNOWN,
            "implementation_commit": ADJ.UNKNOWN,
            "implementation_script_sha256": ADJ.UNKNOWN,
            "implementation_test_sha256": ADJ.UNKNOWN,
        }
    )
    parents = {
        ADJ.AUTHORITY_COMMIT: ADJ.PRE_AUTHORITY_BASE_COMMIT,
        RUNTIME_I1_COMMIT: ADJ.AUTHORITY_COMMIT,
        RUNTIME_I_COMMIT: RUNTIME_I1_COMMIT,
        RUNTIME_B_COMMIT: RUNTIME_I_COMMIT,
        V4_I_COMMIT: RUNTIME_B_COMMIT,
        V4_B_COMMIT: V4_I_COMMIT,
    }
    changed_paths = {
        ADJ.AUTHORITY_COMMIT: list(ADJ.EXPECTED_AUTHORITY_A_PATHS),
        RUNTIME_I1_COMMIT: list(ADJ.EXPECTED_AUTHORITY_RUNTIME_I1_PATHS),
        RUNTIME_I_COMMIT: list(ADJ.EXPECTED_AUTHORITY_RUNTIME_I_PATHS),
        RUNTIME_B_COMMIT: list(ADJ.EXPECTED_AUTHORITY_RUNTIME_B_PATHS),
        V4_I_COMMIT: list(ADJ.EXPECTED_I_PATHS),
        V4_B_COMMIT: list(ADJ.EXPECTED_B_PATHS),
    }
    sha_overrides: dict[tuple[str, str], str] = {}
    git_text_overrides: dict[tuple[str, ...], str] = {}
    i1_document = {
        "implementation_binding": {
            "status": ADJ.UNKNOWN,
            "implementation_commit": ADJ.UNKNOWN,
            "implementation_script_sha256": ADJ.UNKNOWN,
            "implementation_test_sha256": ADJ.UNKNOWN,
        }
    }
    i1_prefix = ADJ.json_bytes(i1_document)
    i1_config_payload = i1_prefix + b" " * (
        ADJ.AUTHORITY_RUNTIME_I1_CONFIG_BYTES - len(i1_prefix)
    )

    def fake_git_text(repo: Path, *args: str) -> str:
        del repo
        if args in git_text_overrides:
            return git_text_overrides[args]
        if args == ("status", "--porcelain"):
            return ""
        if args == ("branch", "--show-current"):
            return config["repository_authority"]["branch"]
        if args in {
            ("rev-parse", "HEAD"),
            ("rev-parse", "@{u}"),
            (
                "rev-parse",
                f"refs/remotes/origin/{config['repository_authority']['branch']}",
            ),
        }:
            return V4_B_COMMIT
        raise AssertionError(f"unexpected git text query: {args}")

    def fake_show_sha(repo: Path, commit: str, path: str) -> str:
        del repo
        override = sha_overrides.get((commit, path))
        if override is not None:
            return override
        authority_blobs = {
            **ADJ.ROOT_CONTRACT_FROZEN_BLOB,
            **ADJ.AUTHORITY_A_EXACT_CHANGED_BLOBS,
        }
        if path in authority_blobs and commit in {ADJ.AUTHORITY_COMMIT, V4_B_COMMIT}:
            return authority_blobs[path]
        if path in ADJ.PREDECESSOR_V3_FROZEN_BLOBS and commit == V4_B_COMMIT:
            return ADJ.PREDECESSOR_V3_FROZEN_BLOBS[path]
        if path == ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH and commit == RUNTIME_I1_COMMIT:
            return ADJ.AUTHORITY_RUNTIME_I1_CONFIG_SHA256
        if path == ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH and commit == RUNTIME_I_COMMIT:
            return ADJ.AUTHORITY_RUNTIME_I2_CONFIG_SHA256
        if path == ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH and commit in {
            RUNTIME_B_COMMIT,
            V4_B_COMMIT,
        }:
            return ADJ.AUTHORITY_RUNTIME_B2_CONFIG_SHA256
        if path == ADJ.AUTHORITY_RUNTIME_SCRIPT_REPO_PATH and commit == RUNTIME_I1_COMMIT:
            return ADJ.AUTHORITY_RUNTIME_I1_SCRIPT_SHA256
        if path == ADJ.AUTHORITY_RUNTIME_TEST_REPO_PATH and commit == RUNTIME_I1_COMMIT:
            return ADJ.AUTHORITY_RUNTIME_I1_TEST_SHA256
        if path == ADJ.AUTHORITY_RUNTIME_SCRIPT_REPO_PATH and commit in {
            RUNTIME_I_COMMIT,
            RUNTIME_B_COMMIT,
            V4_B_COMMIT,
        }:
            return RUNTIME_SCRIPT_SHA256
        if path == ADJ.AUTHORITY_RUNTIME_TEST_REPO_PATH and commit in {
            RUNTIME_I_COMMIT,
            RUNTIME_B_COMMIT,
            V4_B_COMMIT,
        }:
            return RUNTIME_TEST_SHA256
        if path == ADJ.SCRIPT_REPO_PATH and commit == V4_B_COMMIT:
            return SYNTHETIC_SCRIPT_SHA256
        if path == ADJ.TEST_REPO_PATH and commit == V4_B_COMMIT:
            return SYNTHETIC_TEST_SHA256
        raise AssertionError(f"unexpected blob query: {commit}:{path}")

    def fake_show_json(repo: Path, commit: str, path: str) -> dict[str, Any]:
        del repo
        if path == ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH:
            if commit == RUNTIME_I_COMMIT:
                return copy.deepcopy(runtime_i_config)
            if commit in {RUNTIME_B_COMMIT, V4_B_COMMIT}:
                return copy.deepcopy(runtime_b_config)
        if path == ADJ.CONFIG_REPO_PATH:
            if commit == V4_I_COMMIT:
                return copy.deepcopy(v4_i_config)
            if commit == V4_B_COMMIT:
                return copy.deepcopy(dict(config))
        if path == ADJ.PREDECESSOR_V3_CONFIG_REPO_PATH and commit == V4_B_COMMIT:
            return {
                "current_external_state": {
                    "status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE"
                }
            }
        raise AssertionError(f"unexpected JSON query: {commit}:{path}")

    def fake_git(repo: Path, *args: str) -> bytes:
        del repo
        if args == (
            "show",
            f"{RUNTIME_I1_COMMIT}:{ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH}",
        ):
            return i1_config_payload
        raise AssertionError(f"unexpected raw Git query: {args}")

    monkeypatch.setattr(ADJ, "_git_text", fake_git_text)
    monkeypatch.setattr(ADJ, "_commit_parent", lambda repo, commit: parents[commit])
    monkeypatch.setattr(
        ADJ, "_changed_paths", lambda repo, commit: sorted(changed_paths[commit])
    )
    monkeypatch.setattr(ADJ, "_show_sha", fake_show_sha)
    monkeypatch.setattr(ADJ, "_show_json", fake_show_json)
    monkeypatch.setattr(ADJ, "_git", fake_git)
    return changed_paths, sha_overrides, git_text_overrides


def test_v4_is_exact_seven_slot_successor_and_v3_is_frozen_unchanged() -> None:
    config = read_config()
    ADJ.validate_static_config(config)
    assert tuple(slot["slot_id"] for slot in config["evidence_contract"]["slots"]) == ADJ.SLOT_IDS
    assert len(ADJ.SLOT_IDS) == 7
    assert "CHECKPOINT_SPECIFIC_EXPOSURE" not in ADJ.SLOT_IDS
    predecessor = config["repository_authority"]["predecessor_v3_preservation"]
    assert predecessor["expected_external_status"] == "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE"
    assert predecessor["checkpoint_negative_slot_retained_in_v3"] is True
    assert predecessor["checkpoint_negative_slot_relabelled_or_consumed_by_v4"] is False
    assert {item["path"]: item["sha256"] for item in predecessor["frozen_blobs"]} == ADJ.PREDECESSOR_V3_FROZEN_BLOBS


def test_exact_seven_pass_scratch_route_qualifies_one_one_zero_and_6547(tmp_path: Path) -> None:
    config = bind_records(tmp_path)
    output = tmp_path / "v4-success"
    result = ADJ.adjudicate(config, output)
    assert result["publication_status"] == "PUBLISHED_NEW"
    assert (
        result["qualified"],
        result["ordinary_study_contribution"],
        result["a1_study_contribution"],
        result["true_a2_study_contribution"],
        result["canonical_record_count"],
    ) == (True, 1, 1, 0, 6547)
    report = read_report(output)
    audit = read_audit(output)
    assert report["foundation_route_status"] == ADJ.FOUNDATION_ROUTE_STATUS
    assert (
        report["foundation_checkpoint_evidence_status"]
        == ADJ.FOUNDATION_CHECKPOINT_EVIDENCE_STATUS
    )
    assert audit["opened_input_count"] == 7
    assert audit["excluded_checkpoint_record_open_count"] == 0
    assert [slot["slot_id"] for slot in audit["slots"]] == list(ADJ.SLOT_IDS)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("external_checkpoint_count", 1),
        ("external_learned_input_count", 1),
        ("pretrained_weights_present", True),
        ("warm_start_present", True),
        ("external_embedding_present", True),
        ("external_learned_feature_present", True),
        ("external_logits_present", True),
        ("external_teacher_or_distillation_target_present", True),
        ("external_pseudolabel_present", True),
        ("checkpoint_derived_statistic_present", True),
        ("learned_retrieval_or_reranker_present", True),
    ],
)
def test_any_checkpoint_weight_feature_logits_teacher_or_pseudolabel_input_rejects(
    field: str, invalid: Any
) -> None:
    config = read_config()
    config["model_input_route_contract"]["scratch_route"][field] = invalid
    with pytest.raises(ADJ.AdjudicationError, match="scratch route"):
        ADJ.validate_route_contract(config)


@pytest.mark.parametrize(
    "field",
    [
        "outcome_or_model_result_used_for_route_selection",
        "route_switch_after_model_results_allowed",
        "route_fallback_after_failure_allowed",
        "same_dataset_duplicate_credit_across_routes_allowed",
    ],
)
def test_post_result_route_change_fallback_or_duplicate_credit_rejects(field: str) -> None:
    config = read_config()
    config["model_input_route_contract"][field] = True
    with pytest.raises(ADJ.AdjudicationError, match="route"):
        ADJ.validate_route_contract(config)


def test_foundation_predicate_is_retained_and_empty_or_unknown_never_passes() -> None:
    foundation = read_config()["model_input_route_contract"]["foundation_route"]
    assert foundation["status"] == ADJ.FOUNDATION_ROUTE_STATUS
    assert foundation["checkpoint_evidence_status"] == ADJ.UNKNOWN
    assert foundation["checkpoint_specific_exposure_gate_applicable"] is True
    assert foundation["minimum_audited_checkpoint_count_for_pass"] == 1
    assert foundation["empty_checkpoint_set_can_pass"] is False
    assert ADJ.foundation_route_passes(foundation) is False
    empty = copy.deepcopy(foundation)
    empty.update({"status": "PASS", "audited_checkpoint_count": 0})
    assert ADJ.foundation_route_passes(empty) is False


@pytest.mark.parametrize("variant", ["missing", "mixed", "foundation"])
def test_missing_mixed_or_foundation_route_is_not_a_scratch_pass(variant: str) -> None:
    config = read_config()
    route = config["model_input_route_contract"]
    if variant == "missing":
        del route["selected_route"]
    elif variant == "mixed":
        route["selected_route"] = [ADJ.SCRATCH_ROUTE, "FOUNDATION_CHECKPOINT"]
    else:
        route["selected_route"] = "FOUNDATION_CHECKPOINT"
    with pytest.raises(ADJ.AdjudicationError):
        ADJ.validate_route_contract(config)


def test_all_seven_records_require_exact_reused_dec019_provenance(tmp_path: Path) -> None:
    def mutate(slot_id: str, record: dict[str, Any]) -> None:
        if slot_id == "LICENSE_RIGHTS":
            record["provenance"]["acceptance_authority"]["decision_id"] = "V3-DEC-020"

    config = bind_records(tmp_path, mutator=mutate)
    with pytest.raises(ADJ.AdjudicationError, match="reused DEC019 acceptance authority"):
        ADJ.adjudicate(config, tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_success_is_qualification_not_payload_materialization_training_or_next(tmp_path: Path) -> None:
    config = bind_records(tmp_path)
    output = tmp_path / "boundary"
    ADJ.adjudicate(config, output)
    report = read_report(output)
    assert report["canonical_materialization_qualification_eligible"] is True
    assert report["private_payload_access_authorized"] is False
    assert report["canonical_materialization_execution_authorized"] is False
    assert report["row_level_payload_read_count"] == 0
    assert report["private_payload_read_count"] == 0
    assert report["sealed_payload_read_count"] == 0
    assert report["training_allowed"] is False
    assert report["model_selection_allowed"] is False
    assert report["gpu_allowed"] is False
    assert report["next_phase_authorized"] is False
    assert report["scientific_claim_status"] == "NOT_ESTABLISHED"


def test_new_v4_output_is_exclusive_idempotent_and_never_overwrites_old(tmp_path: Path) -> None:
    config = bind_records(tmp_path)
    old = tmp_path / "GSE200304_DEC019_OLD_OUTPUT"
    old.mkdir()
    sentinel = old / "PUBLICATION_COMMIT.json"
    sentinel.write_text("old-v3-evidence\n", encoding="utf-8")
    with pytest.raises(ADJ.PublicationError):
        ADJ.adjudicate(config, old)
    assert sentinel.read_text(encoding="utf-8") == "old-v3-evidence\n"
    output = tmp_path / "GSE200304_DEC020_NEW_OUTPUT"
    assert ADJ.adjudicate(config, output)["publication_status"] == "PUBLISHED_NEW"
    assert ADJ.adjudicate(config, output)["publication_status"] == "IDEMPOTENT_EXISTING_EXACT"


def test_unbound_descriptor_set_reads_zero_and_emits_seven_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = unbind_descriptors(bind_implementation())
    calls = 0

    def forbidden_read(*args: Any, **kwargs: Any) -> bytes:
        nonlocal calls
        calls += 1
        raise AssertionError("unbound descriptors must stop before reads")

    monkeypatch.setattr(ADJ, "_read_verified_evidence", forbidden_read)
    output = tmp_path / "blocked"
    result = ADJ.adjudicate(config, output)
    assert calls == 0
    assert (result["qualified"], result["canonical_record_count"]) == (False, 0)
    assert len(result["blockers"]) == 7
    audit = read_audit(output)
    assert audit["opened_input_count"] == 0
    assert all(slot["input_opened"] is False for slot in audit["slots"])


def test_unknown_implementation_stops_before_evidence_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = unknown_i_config()
    monkeypatch.setattr(
        ADJ,
        "_read_verified_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not read")),
    )
    output = tmp_path / "never"
    with pytest.raises(ADJ.BindingError, match="implementation binding is UNKNOWN"):
        ADJ.adjudicate(config, output)
    assert not output.exists()


def test_canonical_count_must_be_exact_6547_not_merely_positive(tmp_path: Path) -> None:
    def mutate(slot_id: str, record: dict[str, Any]) -> None:
        if slot_id == "CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE":
            record["facts"]["canonical_record_count"] = 6546

    config = bind_records(tmp_path, mutator=mutate)
    result = ADJ.adjudicate(config, tmp_path / "count-blocked")
    assert result["qualified"] is False
    assert result["canonical_record_count"] == 0
    assert "CANONICAL_RECORD_COUNT_NOT_6547" in result["blockers"]


def test_i_to_b_is_exact_four_scalars_and_dynamic_lifecycle_group_is_closed() -> None:
    config_i = unknown_i_config()
    repository = config_i["repository_authority"]
    assert ADJ._runtime_group_is_bound(repository)
    assert repository["authority_runtime_i1_commit"] == RUNTIME_I1_COMMIT
    assert config_i["core_authority"]["authority_commit"] == ADJ.AUTHORITY_COMMIT
    config_b = bind_implementation(config_i)
    ADJ.validate_i_to_b(config_i, config_b)
    assert ADJ._semantic_diff_paths(config_i, config_b) == set(
        ADJ.EXPECTED_I_TO_B_SCALAR_PATHS
    )

    unbound = unbind_runtime_lifecycle(config_i)
    ADJ.validate_static_config(unbound)
    partial = copy.deepcopy(unbound)
    partial["repository_authority"]["base_commit"] = "1" * 40
    with pytest.raises(ADJ.BindingError, match="partial or invalid"):
        ADJ.validate_static_config(partial)

    bound = bind_runtime_lifecycle(config_i)
    ADJ.validate_static_config(bound)
    assert ADJ.config_core_sha256(bound) == ADJ.FROZEN_CONFIG_CORE_SHA256


def test_disk_config_is_valid_in_either_i_or_b_state() -> None:
    config = read_config()
    ADJ.validate_static_config(config)
    binding = config["implementation_binding"]
    if binding["status"] == "BOUND":
        assert ADJ.sha256(SCRIPT.read_bytes()) == binding["implementation_script_sha256"]
        assert ADJ.sha256(Path(__file__).read_bytes()) == binding["implementation_test_sha256"]
    else:
        assert binding["status"] == ADJ.UNKNOWN
        assert all(
            binding[key] == ADJ.UNKNOWN
            for key in (
                "implementation_commit",
                "implementation_script_sha256",
                "implementation_test_sha256",
            )
        )
    normalized = unknown_i_config(config)
    ADJ.validate_static_config(normalized)
    assert ADJ.config_core_sha256(normalized) == ADJ.FROZEN_CONFIG_CORE_SHA256


def test_static_authority_is_exact_A_and_exact14_not_fake_or_path_drift() -> None:
    config = read_config()
    assert config["core_authority"]["authority_commit"] == ADJ.AUTHORITY_COMMIT
    assert config["core_authority"]["authority_commit_exact_changed_paths"] == (
        ADJ.EXPECTED_AUTHORITY_A_PATHS
    )
    fake_a = copy.deepcopy(config)
    fake_a["core_authority"]["authority_commit"] = "f" * 40
    with pytest.raises(ADJ.AdjudicationError, match="authority_commit"):
        ADJ.validate_static_config(fake_a)
    path_drift = copy.deepcopy(config)
    path_drift["core_authority"]["authority_commit_exact_changed_paths"] = (
        ADJ.EXPECTED_AUTHORITY_A_PATHS[:-1]
    )
    with pytest.raises(ADJ.AdjudicationError, match="exact_changed_paths"):
        ADJ.validate_static_config(path_drift)


def test_production_authority_accepts_exact_A_runtime_I_B_and_v4_I_B(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_production_v4()
    install_valid_production_git_lifecycle(monkeypatch, config)
    observed = ADJ.validate_production_authority(config)
    assert observed["mode"] == "PRODUCTION"
    assert observed["lifecycle_state"] == "V4_B_BOUND_EXACT_HEAD"
    assert observed["authority_commit"] == ADJ.AUTHORITY_COMMIT
    assert observed["authority_runtime_i1_commit"] == RUNTIME_I1_COMMIT
    assert observed["authority_runtime_implementation_commit"] == RUNTIME_I_COMMIT
    assert observed["authority_runtime_binding_commit"] == RUNTIME_B_COMMIT
    assert observed["implementation_commit"] == V4_I_COMMIT
    assert observed["binding_commit"] == V4_B_COMMIT


@pytest.mark.parametrize(
    "variant",
    [
        "authority_path_set",
        "inserted_authority_drift",
        "runtime_i1_path",
        "runtime_i_path",
        "runtime_b_extra",
    ],
)
def test_production_authority_rejects_A_or_runtime_lifecycle_drift(
    variant: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_production_v4()
    changed_paths, sha_overrides, git_text_overrides = install_valid_production_git_lifecycle(
        monkeypatch, config
    )
    if variant == "authority_path_set":
        changed_paths[ADJ.AUTHORITY_COMMIT] = ADJ.EXPECTED_AUTHORITY_A_PATHS[:-1]
        match = "authority A changed-path"
    elif variant == "inserted_authority_drift":
        drift_path = ADJ.EXPECTED_AUTHORITY_A_PATHS[0]
        sha_overrides[(V4_B_COMMIT, drift_path)] = "0" * 64
        match = "drifted after A"
    elif variant == "runtime_i1_path":
        changed_paths[RUNTIME_I1_COMMIT] = [ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH]
        match = "runtime I1 changed-path"
    elif variant == "runtime_i_path":
        changed_paths[RUNTIME_I_COMMIT] = [ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH]
        match = "runtime I2 changed-path"
    else:
        changed_paths[RUNTIME_B_COMMIT] = [
            ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH,
            ADJ.AUTHORITY_RUNTIME_SCRIPT_REPO_PATH,
        ]
        match = "runtime B is not config-only"
    with pytest.raises(ADJ.BindingError, match=match):
        ADJ.validate_production_authority(config)


def test_production_authority_rejects_origin_or_runtime_config_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_production_v4()
    _, sha_overrides, git_text_overrides = install_valid_production_git_lifecycle(
        monkeypatch, config
    )
    origin_query = (
        "rev-parse",
        f"refs/remotes/origin/{config['repository_authority']['branch']}",
    )
    git_text_overrides[origin_query] = "0" * 40
    with pytest.raises(ADJ.BindingError, match="HEAD/upstream/origin"):
        ADJ.validate_production_authority(config)
    git_text_overrides.pop(origin_query)
    sha_overrides[(RUNTIME_I_COMMIT, ADJ.AUTHORITY_RUNTIME_CONFIG_REPO_PATH)] = (
        "0" * 64
    )
    with pytest.raises(ADJ.BindingError, match="config identity"):
        ADJ.validate_production_authority(config)


def test_authority_runtime_B_binds_I_and_differs_by_exact_four_scalars() -> None:
    runtime_i = read_runtime_i_config()
    runtime_b = copy.deepcopy(runtime_i)
    runtime_b["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": RUNTIME_I_COMMIT,
            "implementation_script_sha256": RUNTIME_SCRIPT_SHA256,
            "implementation_test_sha256": RUNTIME_TEST_SHA256,
        }
    )
    ADJ._validate_authority_runtime_i_to_b(
        runtime_i,
        runtime_b,
        runtime_implementation_commit=RUNTIME_I_COMMIT,
        runtime_script_sha256=RUNTIME_SCRIPT_SHA256,
        runtime_test_sha256=RUNTIME_TEST_SHA256,
    )
    extra_change = copy.deepcopy(runtime_b)
    extra_change["successor_scientific_state"]["qualified"] = True
    with pytest.raises(ADJ.BindingError, match="config core"):
        ADJ._validate_authority_runtime_i_to_b(
            runtime_i,
            extra_change,
            runtime_implementation_commit=RUNTIME_I_COMMIT,
            runtime_script_sha256=RUNTIME_SCRIPT_SHA256,
            runtime_test_sha256=RUNTIME_TEST_SHA256,
        )
    wrong_binding = copy.deepcopy(runtime_b)
    wrong_binding["implementation_binding"]["implementation_commit"] = "f" * 40
    with pytest.raises(ADJ.BindingError, match="implementation_commit differs"):
        ADJ._validate_authority_runtime_i_to_b(
            runtime_i,
            wrong_binding,
            runtime_implementation_commit=RUNTIME_I_COMMIT,
            runtime_script_sha256=RUNTIME_SCRIPT_SHA256,
            runtime_test_sha256=RUNTIME_TEST_SHA256,
        )


def test_private_or_sealed_descriptor_path_rejects_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = bind_records(tmp_path)
    descriptor = config["evidence_descriptor_bindings"]["slots"][0]
    descriptor["absolute_path"] = "/mnt/cunyuliu/sealed/forbidden.json"
    config["evidence_descriptor_bindings"]["descriptor_set_sha256"] = (
        ADJ.descriptor_set_sha256(config)
    )
    calls = 0

    def forbidden_read(*args: Any, **kwargs: Any) -> bytes:
        nonlocal calls
        calls += 1
        raise AssertionError("sealed path must reject before open")

    monkeypatch.setattr(ADJ, "_read_verified_evidence", forbidden_read)
    with pytest.raises(ADJ.ScopeViolation, match="forbidden token"):
        ADJ.adjudicate(config, tmp_path / "never-sealed")
    assert calls == 0
    assert not (tmp_path / "never-sealed").exists()


def test_synthetic_bundle_inspector_is_nonproduction_and_exact4(tmp_path: Path) -> None:
    config = bind_records(tmp_path)
    output = tmp_path / "inspect"
    ADJ.adjudicate(config, output)
    inspected = ADJ.inspect_committed_bundle(output)
    assert {item.name for item in output.iterdir()} == {
        "ADJUDICATION_REPORT.json",
        "INPUT_EVIDENCE_AUDIT.json",
        "SHA256SUMS",
        "PUBLICATION_COMMIT.json",
    }
    assert inspected == {
        "publication_status": "COMMITTED_SYNTHETIC_NON_PRODUCTION",
        "production_registerable": False,
        "status": ADJ.SUCCESS_STATUS,
        "qualified": True,
        "canonical_record_count": 6547,
        "private_payload_access_authorized": False,
        "canonical_materialization_execution_authorized": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def test_blocked_synthetic_bundle_remains_nonproduction(tmp_path: Path) -> None:
    config = unbind_descriptors(bind_implementation())
    output = tmp_path / "blocked-inspect"
    ADJ.adjudicate(config, output)
    inspected = ADJ.inspect_committed_bundle(output)
    assert inspected["publication_status"] == "COMMITTED_SYNTHETIC_NON_PRODUCTION"
    assert inspected["production_registerable"] is False
    assert inspected["qualified"] is False
    assert inspected["status"] == ADJ.BLOCKED_STATUS
    assert inspected["canonical_record_count"] == 0
    assert inspected["scientific_claim_status"] == "NOT_ESTABLISHED"


def production_authority_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "PRODUCTION",
        "lifecycle_state": "V4_B_BOUND_EXACT_HEAD",
        "pre_authority_base_commit": ADJ.PRE_AUTHORITY_BASE_COMMIT,
        "authority_commit": ADJ.AUTHORITY_COMMIT,
        "authority_runtime_i1_commit": RUNTIME_I1_COMMIT,
        "authority_runtime_implementation_commit": RUNTIME_I_COMMIT,
        "authority_runtime_binding_commit": RUNTIME_B_COMMIT,
        "implementation_commit": V4_I_COMMIT,
        "binding_commit": V4_B_COMMIT,
        "current_head": V4_B_COMMIT,
        "config_core_sha256": ADJ.FROZEN_CONFIG_CORE_SHA256,
        "evidence_descriptor_set_sha256": config["evidence_descriptor_bindings"][
            "descriptor_set_sha256"
        ],
    }


def test_production_provenance_exact_bundle_is_registerable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = bind_records(tmp_path)
    config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": V4_I_COMMIT,
            "implementation_script_sha256": SYNTHETIC_SCRIPT_SHA256,
            "implementation_test_sha256": SYNTHETIC_TEST_SHA256,
        }
    )
    install_valid_production_git_lifecycle(monkeypatch, config)
    output = tmp_path / "production-inspect"
    publish_with_authority(config, output, production_authority_provenance(config))
    inspected = ADJ.inspect_committed_bundle(output, config=config)
    assert inspected["publication_status"] == "COMMITTED_EXACT"
    assert inspected["production_registerable"] is True
    assert inspected["qualified"] is True
    assert inspected["scientific_claim_status"] == "NOT_ESTABLISHED"


def test_inspector_rejects_qualified_bundle_without_exact_seven(tmp_path: Path) -> None:
    config = bind_records(tmp_path)
    authority = production_authority_provenance(config)
    report, audit = ADJ.recompute_adjudication_outputs(config, authority)
    audit["mode"] = "NO_INPUT_READ_DESCRIPTOR_SET_INCOMPLETE"
    audit["opened_input_count"] = 0
    for slot in audit["slots"]:
        slot.update(
            {
                "descriptor_bound": False,
                "input_opened": False,
                "hash_verified": False,
                "gate_status": ADJ.UNKNOWN,
            }
        )
    output = tmp_path / "forged-no-input-pass"
    ADJ._publish_bundle(output, ADJ._build_bundle(config, report, audit))
    with pytest.raises(ADJ.AdjudicationError, match="exact-seven PASS"):
        ADJ.inspect_committed_bundle(output, config=config)


@pytest.mark.parametrize(
    ("mode", "lifecycle"),
    [
        ("PRODUCTION", "BOUND_CONFIG_SYNTHETIC_EXECUTION"),
        ("SYNTHETIC_NON_PRODUCTION", "V4_B_BOUND_EXACT_HEAD"),
    ],
)
def test_inspector_rejects_mismatched_authority_mode_and_lifecycle(
    mode: str, lifecycle: str, tmp_path: Path
) -> None:
    config = bind_records(tmp_path)
    authority = production_authority_provenance(config)
    authority.update({"mode": mode, "lifecycle_state": lifecycle})
    output = tmp_path / f"mismatch-{mode}"
    publish_with_authority(config, output, authority)
    with pytest.raises(ADJ.PublicationError, match="mode/lifecycle"):
        ADJ.inspect_committed_bundle(output)
