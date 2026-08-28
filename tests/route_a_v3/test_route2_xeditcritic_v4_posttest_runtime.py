from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.route2_xeditcritic_gate_v4 import (
    CONFIRMATION_SEEDS_V4,
    LOSO_STUDIES_V4,
)
from scripts.route_a_v3.adjudicate_route2_xeditcritic_v4_posttest import (
    adjudicate_loso_jobs_v4,
    adjudicate_refits_v4,
)
from scripts.route_a_v3.adjudicate_route2_xeditcritic_v4_readiness import (
    compose_readiness_v4,
)
from scripts.route_a_v3 import prepare_route2_xeditcritic_v4_posttest_configs as prepare
from scripts.route_a_v3.authorize_route2_xeditcritic_v4_posttest import (
    build_posttest_authorization_v4,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER_HEAD = "a" * 40


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _protocol(tmp_path: Path) -> dict:
    protocol = json.loads(
        (
            ROOT
            / "configs/route_a_v3_route2_xeditcritic_v4_posttest_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    three = tmp_path / "three.json"
    receipt = tmp_path / "receipt.json"
    _write(
        three,
        {
            "status": "XEDITCRITIC_V4_THREE_SEED_PASS",
            "required_seeds": list(CONFIRMATION_SEEDS_V4),
            "development_test_authorized": True,
            "atomic_development_test_only": True,
        },
    )
    _write(
        receipt,
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_authorization_receipt.v1",
            "status": "XEDITCRITIC_V4_POSTTEST_AUTHORIZED",
            "required_seeds": list(CONFIRMATION_SEEDS_V4),
            "frozen_test_gate_status": "XEDITCRITIC_V4_FROZEN_TEST_PASS",
            "all_development_refit_authorized": True,
            "development_test_access_event_count": 1,
            "general_test_projection_persisted": False,
            "test_bottom_six_cache_persisted": False,
            "development_test_metrics_in_receipt": False,
            "new_final_evaluation_outcomes_accessed": False,
        },
    )
    protocol["three_seed_gate_path"] = str(three)
    protocol["posttest_authorization_receipt_path"] = str(receipt)
    protocol["all_development_refit"]["run_root"] = str(tmp_path / "refit-runs")
    protocol["test_preserving_loso"]["run_root"] = str(tmp_path / "loso-runs")
    return protocol


def _base() -> dict:
    return {
        "data_geometry": {
            "expected_record_count": 107873,
            "expected_train_count": 89580,
            "expected_validation_count": 18293,
            "withheld_development_test_record_count": 18292,
            "pass_count": 8,
            "updates_per_pass": 2802,
            "total_optimizer_updates": 22416,
            "maximum_record_repeats_per_pass": 4,
            "effective_batch_size": 32,
        },
        "required_screen_runs": [
            {
                "run_id": "v4_full",
                "model": "V4-FULL",
                "control": "NONE",
                "mechanism": "FULL",
                "selectable": True,
            },
            {
                "run_id": "c0_v4",
                "model": "C0-V4",
                "control": "NONE",
                "mechanism": "RAW_BASELINE",
                "selectable": False,
            },
        ],
        "training": {},
        "gpu_policy": {"physical_gpu_scope": [0, 1, 2, 3, 4, 5]},
        "memory_preflight": {},
    }


def test_v4_posttest_preparers_emit_three_refits_and_42_paired_loso(
    tmp_path: Path, monkeypatch
) -> None:
    protocol = _protocol(tmp_path)
    records = [
        SimpleNamespace(study=study)
        for study in LOSO_STUDIES_V4
        for _ in range(2)
    ]
    monkeypatch.setattr(prepare, "_load_records", lambda _: records)
    monkeypatch.setattr(prepare, "_updates_per_pass", lambda records, seed: 123)
    refit = prepare.prepare_refit_configs_v4(
        protocol, _base(), runner_git_head=RUNNER_HEAD
    )
    assert refit["job_count"] == 3
    assert {job["run_id"] for job in refit["jobs"]} == {"v4_full"}
    assert {job["config"]["data_geometry"]["total_optimizer_updates"] for job in refit["jobs"]} == {984}

    completed_refit = {
        "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "completed_refit_count": 3,
        "refit_pass_count": 8,
        "loso_authorized": True,
        "manifest_path": str(tmp_path / "refit-manifest.json"),
    }
    loso = prepare.prepare_loso_configs_v4(
        protocol,
        _base(),
        completed_refit,
        runner_git_head=RUNNER_HEAD,
    )
    assert loso["job_count"] == 42
    identities = {
        (job["seed"], job["held_out_study"], job["run_id"])
        for job in loso["jobs"]
    }
    assert len(identities) == 42
    assert all(
        job["config"]["held_out_study_scale_policy"]
        == "UNKNOWN_STUDY_SCALE_FIXED_1"
        for job in loso["jobs"]
    )


def test_v4_posttest_config_manifest_is_atomically_published(tmp_path: Path) -> None:
    output = tmp_path / "runtime-configs"
    payload = {
        "status": "XEDITCRITIC_V4_REFIT_CONFIGS_PREPARED_NOT_STARTED",
        "jobs": [
            {
                "seed": 20260908,
                "run_id": "v4_full",
                "config": {"run_stage": "REFIT", "training_seed": 20260908},
            }
        ],
    }
    prepare.write_manifest_v4(payload, output)
    assert not output.with_name(output.name + ".partial").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["runtime_config_paths"]) == 1
    assert Path(manifest["runtime_config_paths"][0]).is_file()

    stale_output = tmp_path / "stale-configs"
    stale = stale_output.with_name(stale_output.name + ".partial")
    stale.mkdir()
    try:
        prepare.write_manifest_v4(payload, stale_output)
    except Exception as error:
        assert "partial posttest config root exists" in str(error)
    else:
        raise AssertionError("stale posttest staging directory was overwritten")
    assert stale.is_dir()


def test_v4_posttest_authorizer_keeps_stage_seed_and_holdout_scope(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    refit = {
        "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "completed_refit_count": 3,
        "refit_pass_count": 8,
        "loso_authorized": True,
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    result = build_posttest_authorization_v4(
        protocol,
        stage="LOSO",
        current_git_head="head",
        refit_manifest=refit,
    )
    assert result["authorized_seeds"] == list(CONFIRMATION_SEEDS_V4)
    assert result["authorized_run_ids"] == ["v4_full", "c0_v4"]
    assert result["authorized_held_out_studies"] == list(LOSO_STUDIES_V4)
    assert result["development_test_outcome_reads_during_posttest"] == 0


def _parameter_update_job(
    tmp_path: Path,
    *,
    stage: str,
    seed: int,
    run_id: str,
    train_count: int,
    validation_count: int,
    updates: int,
    held_out_study: str | None = None,
    rho: float | None = None,
) -> dict:
    output_root = tmp_path / f"{stage.lower()}-{seed}-{held_out_study or 'all'}"
    output_directory = output_root / run_id
    summary_path = output_directory / "run_summary.json"
    failure_path = output_directory / "failure.json"
    checkpoint_path = output_directory / "final_pass_8_checkpoint.pt"
    attempt_path = output_directory / "training_attempt.json"
    config_path = tmp_path / "configs" / f"{stage}-{seed}-{held_out_study or 'all'}.json"
    config = {
        **_base(),
        "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_runtime.v1",
        "run_stage": stage,
        "training_seed": seed,
        "posttest_runner_git_head": RUNNER_HEAD,
        "required_posttest_run_ids": (
            ["v4_full"] if stage == "REFIT" else ["v4_full", "c0_v4"]
        ),
        "output_root": str(output_root),
        "held_out_study": held_out_study,
        "held_out_study_scale_policy": (
            "NOT_APPLICABLE_ALL_DEVELOPMENT_REFIT"
            if held_out_study is None
            else "UNKNOWN_STUDY_SCALE_FIXED_1"
        ),
        "data_geometry": {
            **_base()["data_geometry"],
            "expected_train_count": train_count,
            "expected_validation_count": validation_count,
            "pass_count": 8,
            "updates_per_pass": updates // 8,
            "total_optimizer_updates": updates,
        },
    }
    _write(config_path, config)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint-present")
    scope = (
        "NOT_CLAIMED_DIFFERENT_C0_ARCHITECTURE"
        if run_id == "c0_v4"
        else "SHARED_V4_CONSTRUCTOR_WITHIN_IDENTICAL_ARCHITECTURE"
    )
    common = {
        "seed": seed,
        "parameter_initialization_seed": seed,
        "parameter_initialization_seed_applied_before_model_construction": True,
        "parameter_initialization_tensor_identity_scope": scope,
        "training_git_head": RUNNER_HEAD,
        "cuda_available": True,
        "cuda_device_name": "NVIDIA A100-SXM4-80GB",
        "a100_device_verified": True,
        "bf16_supported": True,
        "cpu_fallback_used": False,
        "output_directory": str(output_directory),
        "training_summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "training_attempt_path": str(attempt_path),
    }
    summary = {
        **common,
        "schema_version": f"route_a_v3_route2_xeditcritic_v4_{stage.lower()}_run.v2",
        "status": f"TERMINAL_XEDITCRITIC_V4_{stage}_RUN_COMPLETE",
        "run_stage": stage,
        "run_id": run_id,
        "held_out_study": held_out_study,
        "held_out_study_scale_policy": config["held_out_study_scale_policy"],
        "physical_gpu_index": 0,
        "cuda_device": "cuda:0",
        "precision": "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE",
        "train_record_count": train_count,
        "validation_record_count": validation_count,
        "pass_count": 8,
        "selected_pass": 8,
        "update_count": updates,
        "physical_batch_size": 8,
        "effective_batch_size": 32,
        "parameter_changed": True,
        "selection_policy": "FINAL_PASS_8_FIXED_NO_TEST_OR_VALIDATION_SELECTION",
        "final_validation": (
            {"task_macro_spearman": rho} if rho is not None else None
        ),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    _write(summary_path, summary)
    _write(
        attempt_path,
        {
            **common,
            "status": "COMPLETED",
            "code_commit": RUNNER_HEAD,
            "device": "cuda:0",
            "training_precision": "BF16",
            "optimizer_steps": updates,
            "selected_epoch": 8,
        },
    )
    return {
        "seed": seed,
        "run_id": run_id,
        "held_out_study": held_out_study,
        "config_path": str(config_path),
        "summary_path": str(summary_path),
        "failure_path": str(failure_path),
    }


def test_v4_refit_adjudicator_requires_exact_three_terminal_jobs(tmp_path: Path) -> None:
    jobs = [
        _parameter_update_job(
            tmp_path,
            stage="REFIT",
            seed=seed,
            run_id="v4_full",
            train_count=107_873,
            validation_count=0,
            updates=984,
        )
        for seed in CONFIRMATION_SEEDS_V4
    ]
    result = adjudicate_refits_v4(
        {
            "status": "XEDITCRITIC_V4_REFIT_CONFIGS_PREPARED_NOT_STARTED",
            "required_seeds": list(CONFIRMATION_SEEDS_V4),
            "runner_git_head": RUNNER_HEAD,
            "refit_pass_count": 8,
            "jobs": jobs,
        }
    )
    assert result["status"] == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
    assert result["loso_authorized"] is True
    assert {
        row["physical_batch_size"] for row in result["checkpoints"]
    } == {8}


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("parameter_initialization_seed", 7),
        ("parameter_initialization_seed_applied_before_model_construction", False),
        ("cuda_available", False),
        ("a100_device_verified", False),
        ("bf16_supported", False),
        ("cpu_fallback_used", True),
        ("training_git_head", "b" * 40),
        ("update_count", 0),
        ("checkpoint_path", "/wrong/checkpoint.pt"),
    ),
)
def test_v4_refit_rejects_missing_or_drifted_training_evidence(
    tmp_path: Path, field: str, invalid: object
) -> None:
    job = _parameter_update_job(
        tmp_path,
        stage="REFIT",
        seed=20260908,
        run_id="v4_full",
        train_count=107_873,
        validation_count=0,
        updates=984,
    )
    summary_path = Path(job["summary_path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = invalid
    _write(summary_path, summary)
    manifest = {
        "status": "XEDITCRITIC_V4_REFIT_CONFIGS_PREPARED_NOT_STARTED",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "runner_git_head": RUNNER_HEAD,
        "refit_pass_count": 8,
        "jobs": [job]
        + [
            _parameter_update_job(
                tmp_path,
                stage="REFIT",
                seed=seed,
                run_id="v4_full",
                train_count=107_873,
                validation_count=0,
                updates=984,
            )
            for seed in (20260909, 20260910)
        ],
    }
    with pytest.raises(Exception, match="parameter-update evidence"):
        adjudicate_refits_v4(manifest)


def _loso_manifest(tmp_path: Path) -> dict:
    jobs = []
    for seed in CONFIRMATION_SEEDS_V4:
        for study in LOSO_STUDIES_V4:
            for run_id, rho in (("v4_full", 0.35), ("c0_v4", 0.20)):
                jobs.append(
                    _parameter_update_job(
                        tmp_path,
                        stage="LOSO",
                        seed=seed,
                        run_id=run_id,
                        train_count=90_000,
                        validation_count=17_873,
                        updates=800,
                        held_out_study=study,
                        rho=rho,
                    )
                )
    return {
        "status": "XEDITCRITIC_V4_LOSO_CONFIGS_PREPARED_NOT_STARTED",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "runner_git_head": RUNNER_HEAD,
        "held_out_studies": list(LOSO_STUDIES_V4),
        "jobs": jobs,
    }


def test_v4_loso_collector_and_readiness_close_exact_terminal_package(tmp_path: Path) -> None:
    loso = adjudicate_loso_jobs_v4(_loso_manifest(tmp_path))
    assert loso["loso_gate"]["status"] == "XEDITCRITIC_V4_LOSO_PASS"
    three = {
        "status": "XEDITCRITIC_V4_THREE_SEED_PASS",
        "development_test_authorized": True,
        "atomic_development_test_only": True,
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
    }
    receipt = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_authorization_receipt.v1",
        "status": "XEDITCRITIC_V4_POSTTEST_AUTHORIZED",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "frozen_test_gate_status": "XEDITCRITIC_V4_FROZEN_TEST_PASS",
        "all_development_refit_authorized": True,
        "development_test_access_event_count": 1,
        "general_test_projection_persisted": False,
        "test_bottom_six_cache_persisted": False,
        "development_test_metrics_in_receipt": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    refit = {
        "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "completed_refit_count": 3,
        "refit_pass_count": 8,
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    readiness = compose_readiness_v4(three, receipt, refit, loso)
    assert readiness["status"] == "CRITIC_V4_READY_FOR_GUIDANCE"
    assert readiness["guidance_authorized"] is True

    receipt["status"] = "XEDITCRITIC_V4_POSTTEST_NOT_AUTHORIZED"
    try:
        compose_readiness_v4(three, receipt, refit, loso)
    except Exception as error:
        assert "posttest receipt is absent" in str(error)
    else:
        raise AssertionError("non-authorizing frozen TEST receipt reached readiness")


def test_v4_loso_technical_failure_is_terminal_no_go(tmp_path: Path) -> None:
    manifest = _loso_manifest(tmp_path)
    failed = manifest["jobs"][0]
    Path(failed["summary_path"]).unlink()
    _write(
        Path(failed["failure_path"]),
        {
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    result = adjudicate_loso_jobs_v4(manifest)
    assert result["loso_gate"]["status"] == "XEDITCRITIC_V4_LOSO_NO_GO"
    assert result["loso_gate"]["guidance_readiness_authorized"] is False
