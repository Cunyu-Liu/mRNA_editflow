from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"
S1_PATH = (
    ROOT
    / "configs/route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_v1.json"
)
BASE = json.loads(BASE_PATH.read_text(encoding="utf-8"))
S1 = json.loads(S1_PATH.read_text(encoding="utf-8"))


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_base_v4_screen_remains_the_read_only_terminal_reference() -> None:
    assert BASE["schema_version"] == "route_a_v3_route2_xeditsetflow_v4_screen_config.v1"
    assert BASE["status"] == "FROZEN_BEFORE_V4_PARAMETER_UPDATE_OR_VALIDATION_GENERATION_READ"
    assert [row["run_id"] for row in BASE["required_screen_runs"]] == [
        "v4_full",
        "v4_single_mode",
    ]
    assert "s1_mechanics" not in BASE
    assert S1["base_screen_config"] == str(BASE_PATH.relative_to(ROOT))
    assert S1["legacy_v403_conclusion"] == {
        "gate_status": "XEDITSETFLOW_V4_SCREEN_NO_GO",
        "confirmation_authorized": False,
        "artifacts_immutable": True,
        "reinterpreted_as_s1_result": False,
    }


def test_s1_changes_only_the_declared_mechanics_not_v4_geometry_or_schedule() -> None:
    for key in (
        "data_geometry",
        "architecture",
        "training",
        "validation_generation",
        "terminal_f2_reference",
    ):
        assert S1[key] == BASE[key]

    assert S1["objective"]["common_set_marginal_weight"] == BASE["objective"][
        "common_set_marginal_weight"
    ] == 1.0
    assert S1["objective"]["source_candidate_coverage_weight"] == BASE[
        "objective"
    ]["source_candidate_coverage_weight"] == 0.5
    assert S1["objective"]["remaining_count_weight"] == BASE["objective"][
        "remaining_count_weight"
    ] == 0.2
    assert S1["objective"]["full_mode_information_weight"] == BASE["objective"][
        "full_mode_information_weight"
    ] == 0.05
    assert S1["objective"]["single_mode_information_weight"] == BASE[
        "objective"
    ]["single_mode_information_weight"] == 0.0


def test_s1_run_identity_weight_and_no_sweep_are_exact() -> None:
    assert S1["schema_version"] == (
        "route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_config.v1"
    )
    assert S1["amendment_identity"] == (
        "XEDITSETFLOW_V4_S1_CROSS_STATE_CANDIDATE_MODE_RESPONSIBILITY"
    )
    assert S1["objective"]["identity"] == S1["amendment_identity"]
    assert S1["objective"]["cross_state_candidate_mode_responsibility_weight"] == 0.05
    assert S1["objective"]["cross_state_candidate_mode_responsibility_weight_sweep"] is False
    assert [row["run_id"] for row in S1["required_screen_runs"]] == [
        "v4_s1_full",
        "v4_s1_single_mode",
    ]
    assert [row["mode_count"] for row in S1["required_screen_runs"]] == [8, 1]
    assert [row["mode_information_weight"] for row in S1["required_screen_runs"]] == [
        0.05,
        0.0,
    ]
    assert all(
        row["cross_state_candidate_mode_responsibility_weight"] == 0.05
        for row in S1["required_screen_runs"]
    )
    assert S1["training"]["screen_seed"] == 20260911
    assert S1["confirmation"]["additional_seed_authorized"] is False


def test_s1_candidate_state_occurrence_and_forward_kl_semantics_are_exact() -> None:
    mechanics = S1["s1_mechanics"]
    assert mechanics["canonical_candidate_identity"] == (
        "ZERO_BASED_INDEX_IN_DUPLICATE_COLLAPSED_TERMINAL_EDIT_SETS_"
        "SORTED_BY_EDIT_COUNT_THEN_EDIT_TUPLE"
    )
    assert mechanics["canonical_candidate_identity_uses_train_targets_only"] is True
    assert mechanics["outcome_value_or_row_order_used_in_candidate_identity"] is False
    assert [row["state_slot"] for row in mechanics["four_state_slots"]] == [0, 1, 2, 3]
    assert [row["state_kind"] for row in mechanics["four_state_slots"]] == [
        "EMPTY",
        "PARTIAL",
        "PARTIAL",
        "COMPLETED_OR_STRUCTURAL",
    ]
    assert mechanics["root_state_slot"] == 0
    assert mechanics["root_candidate_mode_posterior"] == "SOFT_NORMALIZED_MODE_POSTERIOR"
    assert mechanics["root_candidate_mode_posterior_gradient"] == "DETACHED"
    assert mechanics["padded_repeat_of_same_source_gets_new_occurrence_identity"] is True
    assert mechanics["divergence"] == (
        "FORWARD_KL_DETACHED_ROOT_TARGET_TO_CURRENT_STATE_POSTERIOR"
    )
    assert mechanics["reduction_order"] == [
        "STATE_MEAN",
        "CANONICAL_CANDIDATE_MEAN",
        "SOURCE_OCCURRENCE_MEAN",
    ]
    assert mechanics["single_mode_value"] == 0.0


def test_s1_reuses_every_absolute_and_relative_v4_screen_threshold() -> None:
    base_checkpoint = dict(BASE["checkpoint_eligibility_and_selection"])
    s1_checkpoint = dict(S1["checkpoint_eligibility_and_selection"])
    assert base_checkpoint.pop("no_eligible_checkpoint_status") == (
        "XEDITSETFLOW_V4_SCREEN_NO_GO"
    )
    assert s1_checkpoint.pop("no_eligible_checkpoint_status") == (
        "XEDITSETFLOW_V4_S1_SCREEN_NO_GO"
    )
    assert s1_checkpoint == base_checkpoint

    for key, value in BASE["screen_gate"].items():
        if key == "failure_status":
            continue
        assert S1["screen_gate"][key] == value
    assert S1["screen_gate"]["success_status"] == "XEDITSETFLOW_V4_S1_SCREEN_PASS"
    assert S1["screen_gate"]["failure_status"] == "XEDITSETFLOW_V4_S1_SCREEN_NO_GO"


def test_s1_gpu_policy_is_cuda_only_without_a_memory_gate() -> None:
    assert S1["gpu_policy"]["physical_gpu_scope"] == [0, 1, 2, 3, 4, 5]
    assert S1["gpu_policy"]["cuda_bf16_only"] is True
    assert S1["gpu_policy"]["cpu_fallback"] is False
    assert S1["gpu_policy"]["free_or_estimated_memory_gate"] is False
    assert S1["gpu_policy"]["free_or_estimated_memory_sorting"] is False
    assert S1["gpu_policy"]["memory_values_are_diagnostic_only"] is True


def test_terminal_fact_audits_preserve_old_nogo_and_pause_critic_successors() -> None:
    setflow = _json(
        "audits/route_a_v3_route2_xeditsetflow_v403_recovered_screen_terminal_nogo_v1.json"
    )
    facts = setflow["terminal_facts"]
    assert facts["runtime_status"] == (
        "XEDITSETFLOW_V403_VALIDATION_RECOVERY_AND_GATE_TERMINAL"
    )
    assert (facts["validation_job_count"], facts["unique_summary_count"]) == (8, 8)
    assert facts["gate_status"] == "XEDITSETFLOW_V4_SCREEN_NO_GO"
    assert facts["confirmation_authorized"] is False
    assert facts["protected_outcome_reads"] == 0
    assert setflow["scientific_interpretation"]["failed_subchecks_not_invented"] is True
    assert setflow["immutability"]["runtime_and_gate_are_read_only"] is True

    critic = _json("audits/route_a_v3_route2_xeditcritic_v403_full_terminal_v1.json")
    facts = critic["terminal_facts"]
    assert critic["terminal_summary_path"].endswith("/v4_full/run_summary.json")
    assert facts["runtime_status"] == "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL"
    assert facts["terminal_artifact_kind"] == "SUMMARY"
    assert (facts["seed"], facts["completed_passes"], facts["optimizer_update_count"]) == (
        20260907,
        8,
        22416,
    )
    assert facts["physical_gpu_index"] == 5
    assert facts["training_precision"] == "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE"
    assert facts["cuda_used"] is True and facts["cpu_fallback_used"] is False
    assert facts["protected_outcome_reads"] == 0
    assert critic["successor_decision"]["controls_started"] is False


def test_s1_freeze_audit_binds_config_runner_and_unchanged_gates() -> None:
    audit = _json(
        "audits/route_a_v3_route2_xeditsetflow_v4_s1_freeze_and_runner_v1.json"
    )
    assert audit["status"] == "XEDITSETFLOW_V4_S1_PROTOCOL_AND_RUNNER_FROZEN_NO_ATTEMPT"
    assert audit["authority"]["s1_config"] == str(S1_PATH.relative_to(ROOT))
    assert audit["new_family"]["run_ids"] == ["v4_s1_full", "v4_s1_single_mode"]
    assert audit["new_family"]["legacy_v403_retry"] is False
    assert audit["mechanism_delta"]["weight"] == 0.05
    assert audit["mechanism_delta"]["weight_sweep_authorized"] is False
    assert audit["runner"]["launcher"].endswith(
        "launch_route2_xeditsetflow_s1_screen_after_v403_terminal.py"
    )
    assert audit["runner"]["free_or_estimated_memory_gate"] is False
    assert audit["frozen_gate"]["maximum_common_validation_nll"] == 2.06809
    assert audit["frozen_gate"]["minimum_source_macro_recovery"] == 0.35
    assert audit["execution_state"]["optimizer_attempt_started"] is False
    assert audit["execution_state"]["development_test_outcome_reads"] == 0
    assert audit["execution_state"]["new_final_evaluation_outcome_reads"] == 0


def test_protocol_and_execution_docs_append_s1_without_claiming_a_result() -> None:
    protocol = (
        ROOT / "docs/paper/route2_xedit_v4_prospective_experiments_protocol_v1.md"
    ).read_text(encoding="utf-8")
    assert protocol.count("## V4-S1 prospective SetFlow mechanics amendment") == 1
    assert "`XEDITSETFLOW_V4_SCREEN_NO_GO` with `confirmation_authorized=false`" in protocol
    assert "state mean, then canonical-candidate mean, then source-\noccurrence mean" in protocol
    assert "common NLL at most 2.06809" in protocol
    assert "S1 screen evidence cannot by itself establish an excellent Development result" in protocol

    index = (ROOT / "docs/execution/CURRENT_EXECUTION_INDEX.md").read_text(
        encoding="utf-8"
    )
    assert "configs/route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_v1.json" in index
    assert "XEDITSETFLOW_V4_S1_SCREEN_NO_GO" in index
    assert "launch_route2_xeditsetflow_s1_screen_after_v403_terminal.py" in index
    assert "显存只作诊断" in index

    rapid = (
        ROOT / "docs/execution/route_a_v3_route2_rapid_iteration_log_20260827.md"
    ).read_text(encoding="utf-8")
    attempts = (
        ROOT / "docs/execution/route_a_v3_route2_training_attempt_table_20260817.md"
    ).read_text(encoding="utf-8")
    assert "## Iteration 11 — Freeze the independent SetFlow V4-S1 mechanics successor" in rapid
    assert "no S1 optimizer, GPU Validation, Development TEST" in rapid
    assert "## SetFlow V4-S1 prospective mechanics screen freeze（2026-08-28）" in attempts
    assert "当前 optimizer attempt started=false" in attempts


def test_protected_outcome_and_claim_boundaries_remain_closed() -> None:
    assert S1["development_test_outcomes_accessed"] is False
    assert S1["new_final_evaluation_outcomes_accessed"] is False
    assert S1["screen_gate"]["maximum_development_test_outcome_reads"] == 0
    assert S1["screen_gate"]["maximum_new_evaluation_outcome_reads"] == 0
    assert S1["confirmation"]["legacy_v403_recovered_confirmation_launcher_authorized"] is False
    assert S1["claim_boundary"] == (
        "S1_SCREEN_IS_NOT_FINAL_SCIENTIFIC_EVIDENCE_AND_CANNOT_ESTABLISH_AN_"
        "EXCELLENT_DEVELOPMENT_OR_EXTERNAL_EVALUATION_RESULT"
    )
