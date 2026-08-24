from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
    refit = prepare.prepare_refit_configs_v4(protocol, _base())
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
    loso = prepare.prepare_loso_configs_v4(protocol, _base(), completed_refit)
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


def test_v4_refit_adjudicator_requires_exact_three_terminal_jobs(tmp_path: Path) -> None:
    jobs = []
    for seed in CONFIRMATION_SEEDS_V4:
        summary = tmp_path / f"refit-{seed}.json"
        failure = tmp_path / f"refit-{seed}-failure.json"
        _write(
            summary,
            {
                "status": "TERMINAL_XEDITCRITIC_V4_REFIT_RUN_COMPLETE",
                "run_stage": "REFIT",
                "run_id": "v4_full",
                "seed": seed,
                "train_record_count": 107873,
                "validation_record_count": 0,
                "pass_count": 8,
                "selected_pass": 8,
                "physical_batch_size": 8,
                "checkpoint_path": f"/mnt/{seed}.pt",
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        jobs.append(
            {
                "seed": seed,
                "summary_path": str(summary),
                "failure_path": str(failure),
            }
        )
    result = adjudicate_refits_v4(
        {
            "status": "XEDITCRITIC_V4_REFIT_CONFIGS_PREPARED_NOT_STARTED",
            "required_seeds": list(CONFIRMATION_SEEDS_V4),
            "refit_pass_count": 8,
            "jobs": jobs,
        }
    )
    assert result["status"] == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
    assert result["loso_authorized"] is True
    assert {
        row["physical_batch_size"] for row in result["checkpoints"]
    } == {8}


def _loso_manifest(tmp_path: Path) -> dict:
    jobs = []
    for seed in CONFIRMATION_SEEDS_V4:
        for study in LOSO_STUDIES_V4:
            for run_id, rho in (("v4_full", 0.35), ("c0_v4", 0.20)):
                summary = tmp_path / f"{seed}-{study}-{run_id}.json"
                failure = tmp_path / f"{seed}-{study}-{run_id}-failure.json"
                _write(
                    summary,
                    {
                        "status": "TERMINAL_XEDITCRITIC_V4_LOSO_RUN_COMPLETE",
                        "run_stage": "LOSO",
                        "run_id": run_id,
                        "seed": seed,
                        "held_out_study": study,
                        "held_out_study_scale_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
                        "pass_count": 8,
                        "selected_pass": 8,
                        "final_validation": {"task_macro_spearman": rho},
                        "development_test_outcome_reads": 0,
                        "new_final_evaluation_outcome_reads": 0,
                    },
                )
                jobs.append(
                    {
                        "seed": seed,
                        "held_out_study": study,
                        "run_id": run_id,
                        "summary_path": str(summary),
                        "failure_path": str(failure),
                    }
                )
    return {
        "status": "XEDITCRITIC_V4_LOSO_CONFIGS_PREPARED_NOT_STARTED",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
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
        "frozen_test_gate_status": "XEDITCRITIC_V4_FROZEN_TEST_PASS",
        "all_development_refit_authorized": True,
        "development_test_metrics_in_receipt": False,
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
