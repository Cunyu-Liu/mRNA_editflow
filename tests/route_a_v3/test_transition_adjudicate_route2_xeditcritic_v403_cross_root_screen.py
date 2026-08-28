from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.route_a_v3.transition_adjudicate_route2_xeditcritic_v403_cross_root_screen as transition

CONTROL_HEAD = "c" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _summary(
    run_id: str, authorization_path: Path, *, protected_reads: int = 0
) -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_screen_run.v1"
            if run_id in {"c0_v4", "v4_full"}
            else "route_a_v3_route2_xeditcritic_v4_screen_run.v2"
        ),
        "status": "TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE",
        "run_id": run_id,
        "launch_authorization_path": str(authorization_path),
        "development_test_outcome_reads": protected_reads,
        "new_final_evaluation_outcome_reads": 0,
    }


def _authorization(run_id: str, authorized_git_head: str) -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
        ),
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": authorized_git_head,
        "authorized_run_ids": list(transition.ARM_ORDER),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _config(tmp_path: Path) -> tuple[Path, dict]:
    reference_path = tmp_path / "c3_reference.json"
    preflight_path = tmp_path / "preflight.json"
    _write_json(
        reference_path,
        {
            "status": "C3_V4_REFERENCE_READ_ONCE_COMPLETE",
            "terminal_summaries_read_count": 5,
            "c3_reference_task_macro_spearman": 0.2,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    _write_json(
        preflight_path,
        {
            "status": "XEDITCRITIC_V4_PREFLIGHT_PASS",
            "passed": True,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    config = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_config.v1",
        "status": "FROZEN_BEFORE_V4_PARAMETER_UPDATE_OR_VALIDATION_OUTCOME_READ",
        "runner_git_head": CONTROL_HEAD,
        "required_screen_runs": [
            {"run_id": run_id} for run_id in transition.ARM_ORDER
        ],
        "c3_read_once_reference_adjudication": str(reference_path),
        "preflight_output": str(preflight_path),
    }
    config_path = tmp_path / "screen_config.json"
    _write_json(config_path, config)
    return config_path, config


def _arm_sources(tmp_path: Path) -> dict[str, transition.ArmSource]:
    sources: dict[str, transition.ArmSource] = {}
    for run_id in transition.ARM_ORDER:
        head = (
            transition.C0_GIT_HEAD
            if run_id == "c0_v4"
            else transition.TRAINING_GIT_HEAD
            if run_id == "v4_full"
            else CONTROL_HEAD
        )
        role = (
            "HISTORICAL_MATCHED_C0_TERMINAL_SUMMARY"
            if run_id == "c0_v4"
            else (
                "CURRENT_V403_REPAIRED_FULL_TERMINAL_SUMMARY"
                if run_id == "v4_full"
                else "CURRENT_HEAD_CONTROL_TERMINAL_SUMMARY"
            )
        )
        sources[run_id] = transition.ArmSource(
            tmp_path / "arms" / run_id / "run_summary.json",
            role,
        )
        _write_json(
            tmp_path / "authorizations" / run_id / "launch_authorization.json",
            _authorization(run_id, head),
        )
    return sources


def _write_arm_summary(
    source: transition.ArmSource,
    run_id: str,
    *,
    protected_reads: int = 0,
) -> None:
    root = source.summary_path.parents[2]
    authorization_path = (
        root / "authorizations" / run_id / "launch_authorization.json"
    )
    _write_json(
        source.summary_path,
        _summary(
            run_id,
            authorization_path,
            protected_reads=protected_reads,
        ),
    )


def _write_barriers(tmp_path: Path) -> tuple[Path, Path]:
    full_terminal_audit = tmp_path / "full_terminal_audit.json"
    _write_json(
        full_terminal_audit,
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v403_full_terminal.v1"
            ),
            "status": "XEDITCRITIC_V403_FULL_TERMINAL_SUMMARY_RECORDED",
            "evidence_scope": (
                "TERMINAL_FACTS_ALREADY_CONSUMED_BY_THE_LOW_FREQUENCY_"
                "HEARTBEAT_ONLY"
            ),
            "runtime_path": str(transition.CURRENT_FULL_RUNTIME),
            "output_root": str(
                transition.CURRENT_FULL_OUTPUT_ROOT / "v4_full"
            ),
            "terminal_summary_path": str(
                transition.CURRENT_FULL_OUTPUT_ROOT
                / "v4_full/run_summary.json"
            ),
            "terminal_facts": {
                "authorization_git_head": transition.TRAINING_GIT_HEAD,
                "runtime_status": "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL",
                "run_id": "v4_full",
                "terminal_artifact_kind": "SUMMARY",
                "seed": 20260907,
                "completed_passes": 8,
                "selected_pass": 8,
                "optimizer_update_count": 22416,
                "physical_batch_size": 32,
                "effective_batch_size": 32,
                "cuda_used": True,
                "device_class": "A100",
                "training_precision": (
                    "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE"
                ),
                "cpu_fallback_used": False,
                "protected_outcome_reads": 0,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
            "claim_boundary": {
                "single_arm_terminal_summary_is_not_a_screen_pass": True,
                "single_arm_terminal_summary_is_not_final_scientific_evidence": True,
                "model_advantage_established": False,
                "submission_ready": False,
            },
        },
    )
    control_runtime = tmp_path / "control_runtime.json"
    _write_json(
        control_runtime,
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v403_"
                "control_recovery_runtime.v1"
            ),
            "status": (
                "XEDITCRITIC_V403_CONTROL_RECOVERY_"
                "ALL_SIX_SUMMARIES_TERMINAL"
            ),
            "training_code_git_head": CONTROL_HEAD,
            "ordered_control_run_ids": list(transition.CONTROL_RUN_IDS),
            "jobs": {
                run_id: {
                    "status": "TERMINAL_SUMMARY",
                    "terminal_artifact_kind": "SUMMARY",
                    "return_code": 0,
                }
                for run_id in transition.CONTROL_RUN_IDS
            },
            "full_retrained": False,
            "c0_retrained": False,
            "old_v402_stopped_process_resumed": False,
            "free_memory_gate_applied": False,
            "terminal_artifact_payloads_read_by_scheduler": 0,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    return full_terminal_audit, control_runtime


def test_gate_is_not_called_until_all_eight_exact_summaries_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _ = _config(tmp_path)
    sources = _arm_sources(tmp_path)
    for run_id, source in sources.items():
        if run_id != "v4_no_moe":
            _write_arm_summary(source, run_id)
    full_terminal_audit, control_runtime = _write_barriers(tmp_path)
    calls: list[object] = []

    def forbidden_gate(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("frozen gate must not run on seven arms")

    monkeypatch.setattr(transition, "evaluate_xeditcritic_v4_screen", forbidden_gate)
    output = tmp_path / "new_gate/screen_gate.json"
    with pytest.raises(Exception, match="v4_no_moe is not exact terminal SUMMARY"):
        transition.run(
            expected_control_runner_head=CONTROL_HEAD,
            config_path=config_path,
            arm_sources=sources,
            full_terminal_audit_path=full_terminal_audit,
            control_runtime_path=control_runtime,
            legacy_gate_path=tmp_path / "legacy_gate.json",
            output_path=output,
        )

    assert calls == []
    assert not output.exists()
    assert not output.with_suffix(".json.partial").exists()


def test_complete_cross_root_package_calls_frozen_gate_once_and_preserves_old_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config = _config(tmp_path)
    sources = _arm_sources(tmp_path)
    for run_id, source in sources.items():
        _write_arm_summary(source, run_id)
    full_terminal_audit, control_runtime = _write_barriers(tmp_path)
    legacy_gate = tmp_path / "legacy_gate.json"
    legacy_payload = '{"status":"LEGACY_NO_GO"}\n'
    legacy_gate.write_text(legacy_payload, encoding="utf-8")
    output = tmp_path / "new_gate/screen_gate.json"
    calls: list[dict] = []
    read_paths: list[Path] = []
    original_read_json = transition.read_json

    def tracked_read_json(path: Path):
        read_paths.append(Path(path))
        return original_read_json(path)

    def frozen_gate(
        received_config,
        summaries,
        *,
        c3_reference_spearman,
        preflight,
        terminal_provenance,
        expected_training_git_heads,
    ):
        calls.append(
            {
                "config": received_config,
                "summaries": summaries,
                "reference": c3_reference_spearman,
                "preflight": preflight,
                "terminal_provenance": terminal_provenance,
                "expected_training_git_heads": expected_training_git_heads,
            }
        )
        return {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_gate.v1",
            "status": "XEDITCRITIC_V4_SCREEN_PASS",
            "passed": True,
            "confirmation_authorized": True,
            "development_test_authorized": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }

    monkeypatch.setattr(transition, "evaluate_xeditcritic_v4_screen", frozen_gate)
    monkeypatch.setattr(transition, "read_json", tracked_read_json)
    result = transition.run(
        expected_control_runner_head=CONTROL_HEAD,
        config_path=config_path,
        arm_sources=sources,
        full_terminal_audit_path=full_terminal_audit,
        control_runtime_path=control_runtime,
        legacy_gate_path=legacy_gate,
        output_path=output,
    )

    assert len(calls) == 1
    assert calls[0]["config"] == config
    assert tuple(calls[0]["summaries"]) == transition.ARM_ORDER
    assert tuple(calls[0]["terminal_provenance"]) == transition.ARM_ORDER
    assert all(
        calls[0]["terminal_provenance"][run_id]["run_id"] == run_id
        for run_id in transition.ARM_ORDER
    )
    assert calls[0]["reference"] == 0.2
    assert result["cross_root_transition"]["ordered_run_ids"] == list(
        transition.ARM_ORDER
    )
    assert set(result["cross_root_transition"]["arm_sources"]) == set(
        transition.ARM_ORDER
    )
    provenance = result["cross_root_transition"]["arm_sources"]
    assert provenance["c0_v4"]["authorized_git_head"] == transition.C0_GIT_HEAD
    assert provenance["v4_full"]["authorized_git_head"] == transition.TRAINING_GIT_HEAD
    assert all(
        provenance[run_id]["authorized_git_head"] == CONTROL_HEAD
        for run_id in transition.CONTROL_RUN_IDS
    )
    assert calls[0]["expected_training_git_heads"] == {
        run_id: CONTROL_HEAD for run_id in transition.CONTROL_RUN_IDS
    }
    assert all(
        row["run_id_authorization_verified"] is True
        and row["authorization_protected_outcome_reads_verified_zero"] is True
        and row["launch_authorization_status"]
        == "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
        for row in provenance.values()
    )
    assert result["cross_root_transition"]["terminal_summary_payloads_read"] == 8
    assert read_paths.count(full_terminal_audit) == 1
    assert transition.CURRENT_FULL_RUNTIME not in read_paths
    assert read_paths.count(sources["c0_v4"].summary_path) == 1
    assert read_paths.count(sources["v4_full"].summary_path) == 1
    assert (
        result["cross_root_transition"][
            "historical_terminal_payloads_read_before_cross_root"
        ]
        == 0
    )
    assert (
        result["cross_root_transition"]["full_terminal_audit_path"]
        == str(full_terminal_audit)
    )
    assert result["cross_root_transition"]["scientific_thresholds_changed"] is False
    assert result["development_test_outcome_reads"] == 0
    assert result["new_final_evaluation_outcome_reads"] == 0
    assert legacy_gate.read_text(encoding="utf-8") == legacy_payload
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_protected_read_or_ambiguous_arm_map_prevents_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _ = _config(tmp_path)
    sources = _arm_sources(tmp_path)
    for run_id, source in sources.items():
        _write_arm_summary(
            source,
            run_id,
            protected_reads=1 if run_id == "v4_no_cross" else 0,
        )
    full_terminal_audit, control_runtime = _write_barriers(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(
        transition,
        "evaluate_xeditcritic_v4_screen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with pytest.raises(Exception, match="Development TEST read"):
        transition.run(
            expected_control_runner_head=CONTROL_HEAD,
            config_path=config_path,
            arm_sources=sources,
            full_terminal_audit_path=full_terminal_audit,
            control_runtime_path=control_runtime,
            legacy_gate_path=tmp_path / "legacy_gate.json",
            output_path=tmp_path / "protected/screen_gate.json",
        )
    assert calls == []

    clean_sources = _arm_sources(tmp_path / "clean")
    for run_id, source in clean_sources.items():
        _write_arm_summary(source, run_id)
    ambiguous = dict(clean_sources)
    ambiguous["unexpected_arm"] = transition.ArmSource(
        tmp_path / "unexpected/run_summary.json",
        "UNEXPECTED",
    )
    with pytest.raises(Exception, match="exact ordered eight-arm package"):
        transition.collect_arm_summaries(
            json.loads(config_path.read_text(encoding="utf-8")),
            ambiguous,
            expected_control_runner_head=CONTROL_HEAD,
        )


@pytest.mark.parametrize(
    ("run_id", "field", "invalid_value", "error"),
    [
        (
            "v4_no_cross",
            "schema_version",
            "wrong.schema",
            "launch authorization identity",
        ),
        (
            "v4_no_cross",
            "status",
            "NOT_AUTHORIZED",
            "launch authorization identity",
        ),
        (
            "c0_v4",
            "authorized_git_head",
            transition.TRAINING_GIT_HEAD,
            "launch authorization identity",
        ),
        (
            "v4_no_cross",
            "authorized_git_head",
            transition.C0_GIT_HEAD,
            "launch authorization identity",
        ),
        (
            "v4_no_cross",
            "authorized_run_ids",
            [run_id for run_id in transition.ARM_ORDER if run_id != "v4_no_cross"],
            "launch authorization identity",
        ),
        (
            "v4_no_cross",
            "development_test_outcome_reads",
            1,
            "Development TEST read",
        ),
    ],
)
def test_each_arm_requires_authoritative_launch_authorization(
    tmp_path: Path,
    run_id: str,
    field: str,
    invalid_value: object,
    error: str,
) -> None:
    _, config = _config(tmp_path)
    sources = _arm_sources(tmp_path)
    for current_run_id, source in sources.items():
        _write_arm_summary(source, current_run_id)
    authorization_path = Path(
        json.loads(sources[run_id].summary_path.read_text(encoding="utf-8"))[
            "launch_authorization_path"
        ]
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization[field] = invalid_value
    _write_json(authorization_path, authorization)

    with pytest.raises(Exception, match=error):
        transition.collect_arm_summaries(
            config,
            sources,
            expected_control_runner_head=CONTROL_HEAD,
        )


def test_controls_cannot_reuse_historical_f34_v1_evidence(tmp_path: Path) -> None:
    _, config = _config(tmp_path)
    sources = _arm_sources(tmp_path)
    for run_id, source in sources.items():
        _write_arm_summary(source, run_id)
    run_id = "v4_no_moe"
    summary = json.loads(sources[run_id].summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = "route_a_v3_route2_xeditcritic_v4_screen_run.v1"
    _write_json(sources[run_id].summary_path, summary)
    with pytest.raises(Exception, match="terminal summary identity"):
        transition.collect_arm_summaries(
            config,
            sources,
            expected_control_runner_head=CONTROL_HEAD,
        )

    with pytest.raises(Exception, match="new exact licensed HEAD"):
        transition.collect_arm_summaries(
            config,
            sources,
            expected_control_runner_head=transition.TRAINING_GIT_HEAD,
        )
