from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_gse114002_dec019_true_a2_v3.py"
CONFIG = ROOT / "configs/route_a_v3_gse114002_dec019_true_a2_activation_v3.json"
G200_CONFIG = (
    ROOT
    / "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json"
)
SPEC = importlib.util.spec_from_file_location("gse114002_dec019_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADJ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADJ)

SYNTHETIC_IMPLEMENTATION_COMMIT = "1" * 40
SYNTHETIC_SCRIPT_SHA256 = "2" * 64
SYNTHETIC_TEST_SHA256 = "3" * 64
FORCE_BOUND_CONFIGS_ENV = "G114_V3_TEST_FORCE_BOUND_CONFIGS"


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


def read_g200_config() -> dict[str, Any]:
    return _read_test_config(G200_CONFIG)


def bind_implementation(
    config: Mapping[str, Any] | None = None,
    *,
    implementation: str = SYNTHETIC_IMPLEMENTATION_COMMIT,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(config if config is not None else read_config()))
    binding = result["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = implementation
    binding["implementation_script_sha256"] = SYNTHETIC_SCRIPT_SHA256
    binding["implementation_test_sha256"] = SYNTHETIC_TEST_SHA256
    return result


def unbind_implementation(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    binding = result["implementation_binding"]
    binding["status"] = ADJ.UNKNOWN
    binding["implementation_commit"] = ADJ.UNKNOWN
    binding["implementation_script_sha256"] = ADJ.UNKNOWN
    binding["implementation_test_sha256"] = ADJ.UNKNOWN
    return result


def refresh_descriptors(config: dict[str, Any]) -> None:
    config["evidence_descriptor_bindings"]["status"] = ADJ._descriptor_status(config)
    config["evidence_descriptor_bindings"][
        "descriptor_set_sha256"
    ] = ADJ.descriptor_set_sha256(config)


def descriptor(config: Mapping[str, Any], slot_id: str) -> dict[str, Any]:
    return next(
        slot
        for slot in config["evidence_descriptor_bindings"]["slots"]
        if slot["slot_id"] == slot_id
    )


def privacy() -> dict[str, bool]:
    return {
        "contains_row_level_payload": False,
        "contains_sequence": False,
        "contains_row_identifier": False,
        "contains_raw_label_or_effect": False,
        "contains_member_identifiers_or_hashes": False,
    }


def provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    required = config["evidence_contract"]["required_predecessor_authority"]
    return {
        "producer_protocol_id": "ROUTE_A_V3_SYNTHETIC_AGGREGATE_GATE_PRODUCER_V1",
        "producer_commit": "4" * 40,
        "producer_script_sha256": "5" * 64,
        "source_bundle_id": required["bundle_id"],
        "source_bundle_root_or_target_sha256": required[
            "evt040_runtime_sync_artifact_sha256"
        ],
        "predecessor_authority": copy.deepcopy(required),
        "acceptance_authority": copy.deepcopy(
            config["evidence_contract"]["gate_record_provenance_contract"][
                "acceptance_authority"
            ]
        ),
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
        "blockers": ["HISTORICAL_MECHANICAL_ONLY_NOT_QUALIFIED"],
        "protocol_provenance": {"aggregate_only": True},
        "source_provenance": {"aggregate_only": True},
        "implementation_binding": {
            "status": "PASS_BOUND_IMPLEMENTATION",
            "verified": True,
            "implementation_commit": "6" * 40,
            "binding_commit": "7" * 40,
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


def gate_record(
    config: Mapping[str, Any], slot_id: str, *, status: str = "PASS"
) -> dict[str, Any]:
    is_pass = status == "PASS"
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
        "provenance": provenance(config),
        "facts": copy.deepcopy(passing_facts()[slot_id]) if is_pass else None,
        "unknown_fields": [] if is_pass else sorted(ADJ.FACT_KEYS[slot_id]),
        "reason_codes": [] if is_pass else ["PUBLIC_AUTHORITY_NOT_CLOSED"],
    }


def materialize_all(
    root: Path,
    config: dict[str, Any],
    *,
    statuses: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    root.mkdir()
    statuses = statuses or {}
    paths: dict[str, Path] = {}
    for slot in config["evidence_contract"]["slots"]:
        slot_id = slot["slot_id"]
        value = (
            legacy_geometry()
            if slot_id == "MECHANICAL_ENDPOINT_GEOMETRY"
            else gate_record(config, slot_id, status=statuses.get(slot_id, "PASS"))
        )
        payload = ADJ.json_bytes(value)
        path = root / slot["allowed_basename"]
        path.write_bytes(payload)
        dynamic = descriptor(config, slot_id)
        dynamic["absolute_path"] = str(path)
        dynamic["sha256"] = ADJ.sha256(payload)
        dynamic["bytes"] = len(payload)
        paths[slot_id] = path
    refresh_descriptors(config)
    return paths


def rewrite_record(config: dict[str, Any], slot_id: str, path: Path, value: Any) -> None:
    payload = ADJ.json_bytes(value)
    path.write_bytes(payload)
    dynamic = descriptor(config, slot_id)
    dynamic["sha256"] = ADJ.sha256(payload)
    dynamic["bytes"] = len(payload)
    refresh_descriptors(config)


def test_checked_in_i_or_bound_state_and_hashes_are_closed() -> None:
    config = read_config()
    ADJ.validate_static_config(config)
    binding = config["implementation_binding"]
    assert binding["status"] in {ADJ.UNKNOWN, "BOUND"}
    if binding["status"] == ADJ.UNKNOWN:
        assert {
            binding["implementation_commit"],
            binding["implementation_script_sha256"],
            binding["implementation_test_sha256"],
        } == {ADJ.UNKNOWN}
    else:
        ADJ.validate_implementation_binding(config)
    assert config["repository_authority"]["base_commit"] == (
        "139c4e8d9749ae93ed90924bb527127cf2bbf553"
    )
    assert config["repository_authority"]["historical_dec019_binding"][
        "binding_commit"
    ] == "78827501c7efcef28550b04876c98206d94d4808"
    assert ADJ.config_core_sha256(config) == (
        "6a2955a9c76edbff45aa79c8c71cf3262cbfc631472b345845b5a612a909d67d"
    )
    assert ADJ.descriptor_set_sha256(config) == (
        "65d837cc84b2ca3c4f04b13b4fc38677805507d2df7f6fcd78bd398a81f238d6"
    )


def test_i_to_b_changes_only_four_binding_scalars() -> None:
    config_i = unbind_implementation(read_config())
    config_b = bind_implementation(config_i)
    ADJ._validate_i_to_b_config_pair(
        config_i,
        config_b,
        config_path=ADJ.CONFIG_REPO_PATH,
        implementation_commit="1" * 40,
    )
    assert ADJ.config_core_sha256(config_i) == ADJ.config_core_sha256(config_b)


def test_descriptor_binding_does_not_change_science_core() -> None:
    before = read_config()
    after = copy.deepcopy(before)
    dynamic = descriptor(after, "SOURCE_FIELD_AUTHORITY")
    dynamic.update(
        {
            "absolute_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/x/GSE114002_DEC019_SOURCE_FIELD_AUTHORITY_GATE.json",
            "sha256": "8" * 64,
            "bytes": 123,
        }
    )
    refresh_descriptors(after)
    assert ADJ.config_core_sha256(before) == ADJ.config_core_sha256(after)
    ADJ._validate_descriptor_only_config_pair(
        before, after, config_path=ADJ.CONFIG_REPO_PATH
    )


def test_scientific_change_and_rehash_is_rejected() -> None:
    before = read_config()
    forged = copy.deepcopy(before)
    forged["policy_boundary"]["minimum_power"] = 0.81
    forged["implementation_binding"]["config_core_sha256"] = ADJ.config_core_sha256(
        forged
    )
    with pytest.raises(ADJ.BindingError, match="science core"):
        ADJ._validate_descriptor_only_config_pair(
            before, forged, config_path=ADJ.CONFIG_REPO_PATH
        )


def test_mixed_descriptor_state_is_rejected() -> None:
    config = read_config()
    descriptor(config, "SOURCE_FIELD_AUTHORITY")["absolute_path"] = "/tmp/x"
    with pytest.raises(ADJ.AdjudicationError, match="mixed or invalid"):
        ADJ._descriptor_status(config)


def test_unknown_implementation_stops_before_evidence_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ADJ,
        "_read_verified_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("evidence read")
        ),
    )
    output = tmp_path / "out"
    with pytest.raises(ADJ.BindingError, match="stopped before evidence"):
        ADJ.adjudicate(unbind_implementation(read_config()), output)
    assert not output.exists()


def test_current_partial_descriptors_zero_read_and_exact_six_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = bind_implementation()
    monkeypatch.setattr(
        ADJ,
        "_read_verified_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("evidence read")
        ),
    )
    result = ADJ.adjudicate(config, tmp_path / "blocked")
    assert result["blockers"] == config["current_external_state"][
        "unresolved_blockers"
    ]
    assert len(result["blockers"]) == 6
    assert (
        result["ordinary_study_contribution"],
        result["a1_study_contribution"],
        result["true_a2_study_contribution"],
        result["canonical_record_count"],
        result["training_allowed"],
        result["next_phase_authorized"],
    ) == (0, 0, 0, 0, False, False)
    audit = json.loads((tmp_path / "blocked/INPUT_EVIDENCE_AUDIT.json").read_text())
    assert audit["opened_input_count"] == 0


def test_all_unbound_and_additional_partial_binding_still_zero_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def forbidden(*_args: Any, **_kwargs: Any) -> bytes:
        nonlocal calls
        calls += 1
        raise AssertionError("evidence read")

    monkeypatch.setattr(ADJ, "_read_verified_evidence", forbidden)
    all_unbound = bind_implementation()
    for item in all_unbound["evidence_descriptor_bindings"]["slots"]:
        item.update(
            {"absolute_path": ADJ.UNKNOWN, "sha256": ADJ.UNKNOWN, "bytes": ADJ.UNKNOWN}
        )
    refresh_descriptors(all_unbound)
    result = ADJ.adjudicate(all_unbound, tmp_path / "all_unbound")
    assert len(result["blockers"]) == 7

    partial = bind_implementation()
    descriptor(partial, "SOURCE_FIELD_AUTHORITY").update(
        {"absolute_path": "/tmp/source.json", "sha256": "9" * 64, "bytes": 1}
    )
    refresh_descriptors(partial)
    ADJ.adjudicate(partial, tmp_path / "partial")
    assert calls == 0


def test_all_bound_negative_records_are_read_and_remain_exactly_blocked(
    tmp_path: Path,
) -> None:
    config = bind_implementation()
    statuses = {slot_id: ADJ.UNKNOWN for slot_id in ADJ.FUTURE_SLOT_IDS}
    materialize_all(tmp_path / "evidence", config, statuses=statuses)
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert result["blockers"] == sorted(
        slot["blocker_if_not_pass"]
        for slot in config["evidence_contract"]["slots"][1:]
    )
    assert len(result["blockers"]) == 6
    assert result["canonical_record_count"] == 0
    audit = json.loads((tmp_path / "out/INPUT_EVIDENCE_AUDIT.json").read_text())
    assert audit["opened_input_count"] == 7


def test_negative_record_cannot_encode_unknown_numeric_as_zero(tmp_path: Path) -> None:
    config = bind_implementation()
    paths = materialize_all(
        tmp_path / "evidence",
        config,
        statuses={"PREFROZEN_POWER_PRECISION": ADJ.UNKNOWN},
    )
    bad = gate_record(config, "PREFROZEN_POWER_PRECISION", status=ADJ.UNKNOWN)
    bad["facts"] = copy.deepcopy(passing_facts()["PREFROZEN_POWER_PRECISION"])
    bad["facts"]["observed_power"] = 0.0
    rewrite_record(config, "PREFROZEN_POWER_PRECISION", paths["PREFROZEN_POWER_PRECISION"], bad)
    with pytest.raises(ADJ.AdjudicationError, match="negative facts"):
        ADJ.adjudicate(config, tmp_path / "out")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("observed_power", -0.01, r"\[0, 1\]"),
        ("observed_power", 1.01, r"\[0, 1\]"),
        ("full_confidence_interval_width", -0.01, ">= 0"),
    ],
)
def test_pass_power_and_ci_values_have_physical_ranges(
    tmp_path: Path, field: str, value: float, message: str
) -> None:
    config = bind_implementation()
    paths = materialize_all(tmp_path / "evidence", config)
    record = gate_record(config, "PREFROZEN_POWER_PRECISION")
    record["facts"][field] = value
    rewrite_record(
        config,
        "PREFROZEN_POWER_PRECISION",
        paths["PREFROZEN_POWER_PRECISION"],
        record,
    )
    with pytest.raises(ADJ.AdjudicationError, match=message):
        ADJ.adjudicate(config, tmp_path / "out")


def test_missing_or_drifted_predecessor_provenance_is_rejected(tmp_path: Path) -> None:
    config = bind_implementation()
    paths = materialize_all(tmp_path / "evidence", config)
    source = gate_record(config, "SOURCE_FIELD_AUTHORITY")
    del source["provenance"]
    rewrite_record(config, "SOURCE_FIELD_AUTHORITY", paths["SOURCE_FIELD_AUTHORITY"], source)
    with pytest.raises(ADJ.AdjudicationError, match="closed schema"):
        ADJ.adjudicate(config, tmp_path / "missing")

    config = bind_implementation()
    paths = materialize_all(tmp_path / "evidence2", config)
    source = gate_record(config, "SOURCE_FIELD_AUTHORITY")
    source["provenance"]["predecessor_authority"][
        "predecessor_runtime_event_id"
    ] = "A1-EVT-039"
    rewrite_record(config, "SOURCE_FIELD_AUTHORITY", paths["SOURCE_FIELD_AUTHORITY"], source)
    with pytest.raises(ADJ.AdjudicationError, match="predecessor authority"):
        ADJ.adjudicate(config, tmp_path / "drift")


def test_all_pass_preserves_scientific_ceiling_and_zero_runtime_authority(
    tmp_path: Path,
) -> None:
    config = bind_implementation()
    materialize_all(tmp_path / "evidence", config)
    result = ADJ.adjudicate(config, tmp_path / "out")
    assert result["status"] == ADJ.SUCCESS_STATUS
    assert result["qualified"] is True
    assert (
        result["ordinary_study_contribution"],
        result["a1_study_contribution"],
        result["true_a2_study_contribution"],
    ) == (1, 0, 1)
    assert result["canonical_record_count"] == 3899
    assert (
        result["training_allowed"],
        result["model_selection_allowed"],
        result["next_phase_authorized"],
    ) == (False, False, False)
    report = json.loads((tmp_path / "out/ADJUDICATION_REPORT.json").read_text())
    assert report["confirmatory_contribution"] == 0
    assert report["generalization_contribution"] == 0
    assert report["authority_provenance"]["validation_mode"] == (
        "SYNTHETIC_NONPRODUCTION"
    )


def test_validate_authority_cli_has_no_evidence_or_output_access(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = bind_implementation()
    monkeypatch.setattr(ADJ, "load_production_config", lambda: config)
    monkeypatch.setattr(
        ADJ,
        "validate_production_authority",
        lambda _config: {"validation_mode": "PRODUCTION_GIT_AUTHORITY"},
    )
    monkeypatch.setattr(
        ADJ,
        "_read_verified_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("evidence access")
        ),
    )
    monkeypatch.setattr(
        ADJ,
        "_preflight_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("output access")
        ),
    )
    assert ADJ.main(["--validate-authority"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["evidence_opened_count"] == 0
    assert result["output_access_count"] == 0


def test_production_inspect_rejects_outside_trusted_root_before_open(tmp_path: Path) -> None:
    with pytest.raises(ADJ.ScopeViolation, match="direct child"):
        ADJ.inspect_committed_bundle(
            tmp_path / "outside",
            production=True,
            config=bind_implementation(),
        )


def test_inspect_and_publisher_reject_symlinked_ancestor(tmp_path: Path) -> None:
    config = bind_implementation()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ADJ.AdjudicationError, match="symlink"):
        ADJ.adjudicate(config, linked_parent / "out")

    output = real_parent / "bundle"
    ADJ.adjudicate(config, output)
    linked_output = tmp_path / "linked_bundle"
    linked_output.symlink_to(output, target_is_directory=True)
    with pytest.raises(ADJ.AdjudicationError, match="symlink|opened safely"):
        ADJ.inspect_committed_bundle(linked_output, config=config)


def test_directory_entry_swap_is_detected_by_inode_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    parent_fd = ADJ._open_directory_root_to_leaf(parent, label="parent")
    child_fd = ADJ._open_child_directory(parent_fd, "child", label="child")
    real_stat = os.stat

    def swapped_stat(*args: Any, **kwargs: Any) -> os.stat_result:
        observed = real_stat(*args, **kwargs)
        values = list(observed)
        values[1] += 1
        return os.stat_result(values)

    monkeypatch.setattr(ADJ.os, "stat", swapped_stat)
    try:
        with pytest.raises(ADJ.PublicationError, match="identity changed"):
            ADJ._assert_named_directory_identity(
                parent_fd, "child", child_fd, label="child"
            )
    finally:
        os.close(child_fd)
        os.close(parent_fd)


def test_production_output_is_direct_child_and_parent_rename_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_implementation()
    trusted = tmp_path / "trusted_a1"
    nested = trusted / "nested"
    nested.mkdir(parents=True)
    monkeypatch.setattr(ADJ, "TRUSTED_A1_ROOT", trusted)
    with pytest.raises(ADJ.ScopeViolation, match="direct child"):
        ADJ._preflight_output(nested / "out", config, production=True)

    parent = tmp_path / "publish_parent"
    parent.mkdir()
    moved_parent = tmp_path / "publish_parent_moved"

    def rename_parent(point: str) -> None:
        if point == "after_ADJUDICATION_REPORT.json":
            parent.rename(moved_parent)
            parent.mkdir()

    with pytest.raises(ADJ.PublicationError, match="canonical identity changed"):
        ADJ.adjudicate(config, parent / "out", fault_injector=rename_parent)
    assert not (parent / "out" / ADJ.COMMIT_MARKER).exists()
    assert not (moved_parent / "out" / ADJ.COMMIT_MARKER).exists()


def test_inspector_rejects_parent_rename_after_retained_fd_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_implementation()
    parent = tmp_path / "inspect_parent"
    output = parent / "out"
    parent.mkdir()
    ADJ.adjudicate(config, output)
    moved_parent = tmp_path / "inspect_parent_moved"
    original_read = ADJ._read_regular_at
    renamed = False

    def read_then_rename(directory_fd: int, name: str) -> bytes:
        nonlocal renamed
        payload = original_read(directory_fd, name)
        if not renamed:
            renamed = True
            parent.rename(moved_parent)
            parent.mkdir()
        return payload

    monkeypatch.setattr(ADJ, "_read_regular_at", read_then_rename)
    with pytest.raises(ADJ.PublicationError, match="canonical identity changed"):
        ADJ.inspect_committed_bundle(output, config=config)


def test_publication_idempotence_and_partial_marker_recovery(tmp_path: Path) -> None:
    config = bind_implementation()
    output = tmp_path / "out"
    first = ADJ.adjudicate(config, output)
    second = ADJ.adjudicate(config, output)
    assert first["publication_status"] == "PUBLISHED"
    assert second["publication_status"] == "EXISTING_EXACT"

    partial = tmp_path / "partial"

    def fail(point: str) -> None:
        if point == "before_commit_marker":
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        ADJ.adjudicate(config, partial, fault_injector=fail)
    assert partial.is_dir()
    assert not (partial / ADJ.COMMIT_MARKER).exists()
    with pytest.raises(ADJ.PartialPublicationError):
        ADJ.adjudicate(config, partial)


def test_inspect_rejects_self_consistent_semantic_forgery(tmp_path: Path) -> None:
    config = bind_implementation()
    output = tmp_path / "out"
    ADJ.adjudicate(config, output)
    report = json.loads((output / "ADJUDICATION_REPORT.json").read_text())
    report["qualified"] = True
    (output / "ADJUDICATION_REPORT.json").write_bytes(ADJ.json_bytes(report))
    sums = "".join(
        f"{ADJ.sha256((output / name).read_bytes())}  {name}\n"
        for name in sorted(ADJ.OUTPUT_JSON_NAMES)
    ).encode("ascii")
    (output / "SHA256SUMS").write_bytes(sums)
    marker = json.loads((output / ADJ.COMMIT_MARKER).read_text())
    marker["sha256sums_sha256"] = ADJ.sha256(sums)
    (output / ADJ.COMMIT_MARKER).write_bytes(ADJ.json_bytes(marker))
    with pytest.raises(ADJ.AdjudicationError, match="published report truth qualified"):
        ADJ.inspect_committed_bundle(output, config=config)


def test_inspect_rebuilds_evidence_truth_and_rejects_blocked_to_success_rehash(
    tmp_path: Path,
) -> None:
    config = bind_implementation()
    evidence = materialize_all(tmp_path / "evidence", config)
    power = gate_record(config, "PREFROZEN_POWER_PRECISION")
    power["facts"]["observed_power"] = 0.79
    rewrite_record(
        config,
        "PREFROZEN_POWER_PRECISION",
        evidence["PREFROZEN_POWER_PRECISION"],
        power,
    )
    output = tmp_path / "out"
    result = ADJ.adjudicate(config, output)
    assert result["status"] == ADJ.BLOCKED_STATUS

    blocked = json.loads((output / "ADJUDICATION_REPORT.json").read_text())
    audit = json.loads((output / "INPUT_EVIDENCE_AUDIT.json").read_text())
    assert all(
        slot["gate_status"] == "PASS"
        for slot in audit["slots"][1:]
    )
    forged_report = ADJ._success_report(
        config,
        3899,
        blocked["authority_provenance"],
    )
    forged_payloads = ADJ._build_bundle(config, output, forged_report, audit)
    for name, payload in forged_payloads.items():
        (output / name).write_bytes(payload)

    with pytest.raises(
        ADJ.PublicationError,
        match="evidence-derived expected bundle",
    ):
        ADJ.inspect_committed_bundle(output, config=config)


def test_blocked_and_success_report_truth_matrix_is_closed_in_inspector(
    tmp_path: Path,
) -> None:
    config = bind_implementation()
    blocked_output = tmp_path / "blocked"
    ADJ.adjudicate(config, blocked_output)
    blocked = json.loads((blocked_output / "ADJUDICATION_REPORT.json").read_text())
    blocked_audit = json.loads(
        (blocked_output / "INPUT_EVIDENCE_AUDIT.json").read_text()
    )
    blocked_marker = json.loads((blocked_output / ADJ.COMMIT_MARKER).read_text())
    blocked_mutations = {
        "qualified": True,
        "data_role": "TRUE_A2_QUALIFIED",
        "ordinary_study_contribution": 1,
        "a1_study_contribution": 1,
        "true_a2_study_contribution": 1,
        "confirmatory_contribution": 1,
        "generalization_contribution": 1,
        "canonical_record_count": 1,
        "canonical_materialization_allowed": True,
        "within_assay_development_and_optimization_only": False,
        "technical_uncertainty_is_biological_standard_error": True,
        "k5_is_qualification_gate": True,
        "training_allowed": True,
        "model_selection_allowed": True,
        "next_phase_authorized": True,
        "scientific_claim_status": "ESTABLISHED",
        "aggregate_only": False,
        "blockers": [],
    }
    for key, invalid in blocked_mutations.items():
        forged = copy.deepcopy(blocked)
        forged[key] = invalid
        with pytest.raises(ADJ.AdjudicationError):
            ADJ._validate_published_bundle_semantics(
                forged,
                blocked_audit,
                blocked_marker,
                config=config,
                expected_authority_provenance=blocked["authority_provenance"],
            )

    success_config = bind_implementation()
    materialize_all(tmp_path / "evidence", success_config)
    success_output = tmp_path / "success"
    ADJ.adjudicate(success_config, success_output)
    success = json.loads((success_output / "ADJUDICATION_REPORT.json").read_text())
    success_audit = json.loads(
        (success_output / "INPUT_EVIDENCE_AUDIT.json").read_text()
    )
    success_marker = json.loads((success_output / ADJ.COMMIT_MARKER).read_text())
    success_mutations = {
        "qualified": False,
        "data_role": "TRUE_A2_RECOVERY_CANDIDATE_NOT_QUALIFIED",
        "ordinary_study_contribution": 2,
        "a1_study_contribution": 1,
        "true_a2_study_contribution": 2,
        "confirmatory_contribution": 1,
        "generalization_contribution": 1,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "within_assay_development_and_optimization_only": False,
        "technical_uncertainty_is_biological_standard_error": True,
        "k5_is_qualification_gate": True,
        "training_allowed": True,
        "model_selection_allowed": True,
        "next_phase_authorized": True,
        "scientific_claim_status": "ESTABLISHED",
        "aggregate_only": False,
        "blockers": ["SOURCE_FIELD_AUTHORITY_NOT_PASS"],
    }
    for key, invalid in success_mutations.items():
        forged = copy.deepcopy(success)
        forged[key] = invalid
        with pytest.raises(ADJ.AdjudicationError):
            ADJ._validate_published_bundle_semantics(
                forged,
                success_audit,
                success_marker,
                config=success_config,
                expected_authority_provenance=success["authority_provenance"],
            )


def test_authority_accepts_descendant_head_and_repair_i_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_implementation()
    g200 = bind_implementation(read_g200_config())
    implementation = config["implementation_binding"]["implementation_commit"]
    repair_b = "8" * 40
    head = "9" * 40
    old_b = config["repository_authority"]["historical_dec019_binding"][
        "binding_commit"
    ]
    i_configs = {
        ADJ.CONFIG_REPO_PATH: unbind_implementation(config),
        ADJ.GSE200304_CONFIG_REPO_PATH: unbind_implementation(g200),
    }
    b_configs = {ADJ.CONFIG_REPO_PATH: config, ADJ.GSE200304_CONFIG_REPO_PATH: g200}
    file_payloads: dict[str, bytes] = {}
    for config_path, (script_path, test_path) in ADJ.EXPECTED_IMPLEMENTATION_FILES.items():
        for path in (script_path, test_path):
            file_payloads[path] = f"implementation:{path}".encode()
        b_configs[config_path]["implementation_binding"][
            "implementation_script_sha256"
        ] = ADJ.sha256(file_payloads[script_path])
        b_configs[config_path]["implementation_binding"][
            "implementation_test_sha256"
        ] = ADJ.sha256(file_payloads[test_path])
        i_configs[config_path] = unbind_implementation(b_configs[config_path])

    old_payloads = {
        item["path"]: f"historical:{item['path']}".encode()
        for item in config["repository_authority"]["historical_dec019_binding"][
            "frozen_successor_blobs"
        ]
    }
    real_sha256 = ADJ.sha256
    historical_digest = {
        payload: item["sha256"]
        for item in config["repository_authority"]["historical_dec019_binding"][
            "frozen_successor_blobs"
        ]
        for payload in [old_payloads[item["path"]]]
    }
    monkeypatch.setattr(
        ADJ, "sha256", lambda payload: historical_digest.get(payload, real_sha256(payload))
    )
    monkeypatch.setattr(ADJ, "validate_implementation_binding", lambda _config: None)
    monkeypatch.setattr(ADJ, "_validate_i_to_b_config_pair", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ADJ, "_verify_repo_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ADJ,
        "_read_verified_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authority validator opened evidence")
        ),
    )
    monkeypatch.setattr(
        ADJ,
        "_preflight_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authority validator accessed output")
        ),
    )
    ancestor_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        ADJ,
        "_require_ancestor",
        lambda _repo, ancestor, descendant, *, label: ancestor_calls.append(
            (ancestor, descendant, label)
        ),
    )

    old_i = config["repository_authority"]["historical_dec019_binding"][
        "implementation_commit"
    ]
    old_base = config["repository_authority"]["historical_dec019_binding"][
        "base_commit"
    ]

    def fake_git(_repo: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return head
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return config["repository_authority"]["branch"]
        if args == ("status", "--porcelain"):
            return ""
        if args == (
            "rev-parse",
            f"refs/remotes/origin/{config['repository_authority']['branch']}",
        ):
            return head
        if args == ("rev-parse", f"{old_i}^"):
            return old_base
        if args == ("rev-parse", f"{old_b}^"):
            return old_i
        if args == ("rev-parse", f"{implementation}^"):
            return config["repository_authority"]["base_commit"]
        if args == ("rev-parse", f"{repair_b}^"):
            return implementation
        if args == ("rev-list", "--parents", "-n", "1", old_i):
            return f"{old_i} {old_base}"
        if args == ("rev-list", "--parents", "-n", "1", old_b):
            return f"{old_b} {old_i}"
        if args == ("rev-list", "--parents", "-n", "1", implementation):
            return f"{implementation} {config['repository_authority']['base_commit']}"
        if args == ("rev-list", "--parents", "-n", "1", repair_b):
            return f"{repair_b} {implementation}"
        if args == ("rev-list", "--parents", "-n", "1", head):
            return f"{head} {repair_b}"
        if args[:2] == ("diff-tree", "--no-commit-id"):
            commit = args[-1]
            if commit == implementation:
                return "\n".join(
                    config["repository_authority"][
                        "implementation_commit_exact_changed_paths"
                    ]
                )
            if commit == repair_b:
                return "\n".join(ADJ.BINDING_CONFIG_REPO_PATHS)
        if args[:3] == ("rev-list", "--ancestry-path", "--reverse"):
            if args[3] == f"{implementation}..{head}":
                return f"{repair_b}\n{head}"
            if args[3] == f"{repair_b}..{head}":
                return head
        if args[0] == "log":
            return ""
        if args[:2] == ("diff-tree", "--no-commit-id") and args[-1] == head:
            return ""
        raise AssertionError(args)

    def fake_git_bytes(_repo: Path, *args: str) -> bytes:
        assert args[0] == "show"
        commit, path = args[1].split(":", 1)
        if path in old_payloads:
            return old_payloads[path]
        if path in i_configs and commit == implementation:
            return ADJ.json_bytes(i_configs[path])
        if path in b_configs and commit in {repair_b, head}:
            return ADJ.json_bytes(b_configs[path])
        if path in file_payloads and commit in {implementation, head}:
            return file_payloads[path]
        raise AssertionError((commit, path))

    original_read_bytes = Path.read_bytes

    def fake_read_bytes(path: Path) -> bytes:
        text = str(path)
        for config_path, value in b_configs.items():
            if text.endswith(config_path):
                return ADJ.json_bytes(value)
        return original_read_bytes(path)

    monkeypatch.setattr(ADJ, "_git", fake_git)
    monkeypatch.setattr(ADJ, "_git_bytes", fake_git_bytes)
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    authority = ADJ.validate_production_authority(config)
    assert authority["repair_binding_commit"] == repair_b
    assert authority["current_head_commit"] == head
    assert (old_b, head, "historical DEC-019 binding") in ancestor_calls
    assert (
        old_b,
        config["repository_authority"]["base_commit"],
        "repair base",
    ) in ancestor_calls


def test_authority_validates_exact_repair_i_without_evidence_or_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = unbind_implementation(read_config())
    implementation = "a" * 40
    config["repository_authority"]["historical_dec019_binding"][
        "frozen_successor_blobs"
    ] = []
    monkeypatch.setattr(ADJ, "validate_static_config", lambda _config: None)
    monkeypatch.setattr(ADJ, "_verify_repo_file", lambda *_args, **_kwargs: None)
    parents: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        ADJ,
        "_require_single_parent",
        lambda _repo, commit, parent, *, label: parents.append((commit, parent, label)),
    )
    monkeypatch.setattr(ADJ, "_require_ancestor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ADJ,
        "_read_verified_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authority validator opened evidence")
        ),
    )
    monkeypatch.setattr(
        ADJ,
        "_preflight_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authority validator accessed output")
        ),
    )

    def fake_git(_repo: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return implementation
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return config["repository_authority"]["branch"]
        if args == ("status", "--porcelain"):
            return ""
        if args == (
            "rev-parse",
            f"refs/remotes/origin/{config['repository_authority']['branch']}",
        ):
            return implementation
        if args[:2] == ("diff-tree", "--no-commit-id"):
            return "\n".join(
                config["repository_authority"][
                    "implementation_commit_exact_changed_paths"
                ]
            )
        raise AssertionError(args)

    monkeypatch.setattr(ADJ, "_git", fake_git)
    monkeypatch.setattr(
        ADJ,
        "_git_bytes",
        lambda _repo, *args: ADJ.json_bytes(config)
        if args == ("show", f"{implementation}:{ADJ.CONFIG_REPO_PATH}")
        else (_ for _ in ()).throw(AssertionError(args)),
    )
    original_read_bytes = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: ADJ.json_bytes(config)
        if str(path).endswith(ADJ.CONFIG_REPO_PATH)
        else original_read_bytes(path),
    )
    authority = ADJ.validate_production_authority(config)
    assert authority["lifecycle_state"] == "REPAIR_I_IMPLEMENTATION_UNBOUND"
    assert authority["repair_implementation_commit"] == implementation
    assert authority["repair_binding_commit"] == ADJ.UNKNOWN
    assert (
        implementation,
        config["repository_authority"]["base_commit"],
        "repair I",
    ) in parents


def test_authority_rejects_non_descendant_historical_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_implementation()
    monkeypatch.setattr(ADJ, "validate_implementation_binding", lambda _config: None)
    monkeypatch.setattr(ADJ, "_require_single_parent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ADJ,
        "_git",
        lambda _repo, *args: (
            "a" * 40
            if args == ("rev-parse", "HEAD")
            else config["repository_authority"]["branch"]
            if args == ("rev-parse", "--abbrev-ref", "HEAD")
            else ""
            if args == ("status", "--porcelain")
            else "a" * 40
            if args[0] == "rev-parse" and args[1].startswith("refs/remotes/")
            else config["repository_authority"]["historical_dec019_binding"][
                "base_commit"
            ]
            if args == (
                "rev-parse",
                f"{config['repository_authority']['historical_dec019_binding']['implementation_commit']}^",
            )
            else config["repository_authority"]["historical_dec019_binding"][
                "implementation_commit"
            ]
            if args == (
                "rev-parse",
                f"{config['repository_authority']['historical_dec019_binding']['binding_commit']}^",
            )
            else (_ for _ in ()).throw(AssertionError(args))
        ),
    )
    monkeypatch.setattr(
        ADJ,
        "_require_ancestor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ADJ.BindingError("historical DEC-019 binding is not an ancestor")
        ),
    )
    with pytest.raises(ADJ.BindingError, match="not an ancestor"):
        ADJ.validate_production_authority(config)
