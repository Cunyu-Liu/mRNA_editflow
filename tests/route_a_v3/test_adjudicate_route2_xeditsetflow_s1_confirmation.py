from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.route_a_v3.adjudicate_route2_xeditsetflow_s1_confirmation as adjudicator


HEAD = "c" * 40


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _package(tmp_path: Path) -> tuple[dict, dict, Path]:
    gate = tmp_path / "posttraining" / "confirmation_gate.json"
    config_paths = []
    for seed in adjudicator.CONFIRMATION_SEEDS:
        output_root = tmp_path / f"training/seed_{seed}"
        validation_root = tmp_path / f"validation/seed_{seed}"
        config_path = tmp_path / f"configs/seed_{seed}.json"
        config = {
            "schema_version": adjudicator.CONFIRMATION_RUNTIME_SCHEMA,
            "status": adjudicator.CONFIRMATION_RUNTIME_STATUS,
            "run_stage": "CONFIRMATION",
            "training_seed": seed,
            "selected_model": adjudicator.CONFIRMATION_RUN_ID,
            "confirmation_runner_git_head": HEAD,
            "output_root": str(output_root),
            "validation_output_root": str(validation_root),
            "confirmation_gate_output": str(gate),
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
        _write(config_path, config)
        config_paths.append(str(config_path))
        training_directory = output_root / adjudicator.CONFIRMATION_RUN_ID
        checkpoint_paths = {
            str(checkpoint_pass): str(
                training_directory / f"pass_{checkpoint_pass}.pt"
            )
            for checkpoint_pass in adjudicator.CHECKPOINT_PASSES
        }
        _write(
            training_directory / "training_summary.json",
            {
                "schema_version": (
                    "route_a_v3_route2_xeditsetflow_v4_s1_training_summary.v1"
                ),
                "status": "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION",
                "run_stage": "CONFIRMATION",
                "run_id": "v4_s1_full",
                "selected_model": "v4_s1_full",
                "seed": seed,
                "objective_identity": adjudicator.OBJECTIVE_IDENTITY,
                "cross_state_candidate_mode_responsibility_weight": .05,
                "training_precision": "BF16",
                "torch_device": "cuda:0",
                "device_name": "NVIDIA A100-SXM4-80GB",
                "cpu_fallback_used": False,
                "saved_checkpoint_paths": checkpoint_paths,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        for checkpoint_pass in adjudicator.CHECKPOINT_PASSES:
            output = (
                validation_root
                / adjudicator.CONFIRMATION_RUN_ID
                / f"pass_{checkpoint_pass}"
            )
            summary_path = output / "validation_summary.json"
            _write(
                summary_path,
                {
                    "schema_version": (
                        "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint_validation.v1"
                    ),
                    "status": "TERMINAL_XEDITSETFLOW_V4_S1_CHECKPOINT_VALIDATION_COMPLETE",
                    "run_stage": "CONFIRMATION",
                    "run_id": "v4_s1_full",
                    "selected_model": "v4_s1_full",
                    "seed": seed,
                    "checkpoint_pass": checkpoint_pass,
                    "objective_identity": adjudicator.OBJECTIVE_IDENTITY,
                    "cross_state_candidate_mode_responsibility_weight": .05,
                    "checkpoint_path": checkpoint_paths[str(checkpoint_pass)],
                    "training_summary_path": str(
                        training_directory / "training_summary.json"
                    ),
                    "validation_summary_path": str(summary_path),
                    "precision": "BF16",
                    "torch_device": "cuda:0",
                    "device_name": "NVIDIA A100-SXM4-80GB",
                    "cpu_fallback_used": False,
                    "parameter_update_count": 0,
                    "development_test_outcome_reads": 0,
                    "new_final_evaluation_outcome_reads": 0,
                },
            )
    manifest = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_config_manifest.v1"
        ),
        "status": "THREE_S1_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED",
        "confirmation_runner_git_head": HEAD,
        "selected_model": "v4_s1_full",
        "required_seeds": list(adjudicator.CONFIRMATION_SEEDS),
        "training_job_count": 3,
        "single_mode_training_job_count": 0,
        "checkpoint_validation_job_count": 12,
        "config_paths": config_paths,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    terminal_f2 = tmp_path / "terminal_f2.json"
    _write(terminal_f2, {"reference": True})
    protocol = {
        "schema_version": adjudicator.CONFIRMATION_PROTOCOL_SCHEMA,
        "status": adjudicator.CONFIRMATION_PROTOCOL_STATUS,
        "selected_model": "v4_s1_full",
        "required_seeds": list(adjudicator.CONFIRMATION_SEEDS),
        "additional_seed_authorized": False,
        "terminal_f2_validation_summary": str(terminal_f2),
        "runner_outputs": {
            "confirmation_gate_output_template": str(gate).replace(HEAD, "{runner_git_head}")
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    return protocol, manifest, gate


def _ready_result() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_confirmation_gate.v1",
        "status": "XEDITSETFLOW_V4_G0_READY",
        "required_seeds": list(adjudicator.CONFIRMATION_SEEDS),
        "selected_model": "v4_s1_full",
        "objective_identity": adjudicator.OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": .05,
        "seed_results": {
            str(seed): {"selected_checkpoint_path": f"seed_{seed}.pt"}
            for seed in adjudicator.CONFIRMATION_SEEDS
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_complete_twelve_summary_package_calls_s1_gate_and_emits_v4_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, manifest, gate = _package(tmp_path)
    observed = {}

    def decide(configs, summaries, terminal_f2):
        observed["seeds"] = tuple(configs)
        observed["summary_count"] = sum(len(rows) for rows in summaries.values())
        observed["terminal_f2"] = terminal_f2
        return _ready_result()

    monkeypatch.setattr(adjudicator, "adjudicate_setflow_confirmation_s1", decide)
    result, output = adjudicator.adjudicate_complete_package_s1(protocol, manifest)
    assert result["status"] == "XEDITSETFLOW_V4_G0_READY"
    assert result["schema_version"] == (
        "route_a_v3_route2_xeditsetflow_v4_confirmation_gate.v1"
    )
    assert output == gate
    assert observed["seeds"] == adjudicator.CONFIRMATION_SEEDS
    assert observed["summary_count"] == 12
    adjudicator.write_terminal_gate_s1(output, result)
    assert json.loads(output.read_text())["status"] == "XEDITSETFLOW_V4_G0_READY"


def test_missing_validation_is_technical_error_not_scientific_no_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, manifest, gate = _package(tmp_path)
    config = json.loads(Path(manifest["config_paths"][0]).read_text())
    missing = (
        Path(config["validation_output_root"])
        / "v4_s1_full/pass_4/validation_summary.json"
    )
    missing.unlink()
    monkeypatch.setattr(
        adjudicator,
        "adjudicate_setflow_confirmation_s1",
        lambda *_args: pytest.fail("scientific adjudication must not run"),
    )
    with pytest.raises(
        adjudicator.XEditSetFlowS1ConfirmationAdjudicationError,
        match="not uniquely SUMMARY-terminal",
    ):
        adjudicator.adjudicate_complete_package_s1(protocol, manifest)
    assert not gate.exists()
    source = Path(adjudicator.__file__).read_text()
    assert "confirmation_technical_failure" not in source


def test_each_seed_checkpoint_lineage_is_bound_before_scientific_adjudication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, manifest, _gate = _package(tmp_path)
    config = json.loads(Path(manifest["config_paths"][1]).read_text())
    summary_path = (
        Path(config["validation_output_root"])
        / "v4_s1_full/pass_6/validation_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    summary["checkpoint_path"] = "/wrong/seed/checkpoint.pt"
    summary_path.write_text(json.dumps(summary))
    monkeypatch.setattr(
        adjudicator,
        "adjudicate_setflow_confirmation_s1",
        lambda *_args: pytest.fail("scientific adjudication must not run"),
    )
    with pytest.raises(
        adjudicator.XEditSetFlowS1ConfirmationAdjudicationError,
        match="Validation lineage changed",
    ):
        adjudicator.adjudicate_complete_package_s1(protocol, manifest)
