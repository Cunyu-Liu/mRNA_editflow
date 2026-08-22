from __future__ import annotations

import json
from types import SimpleNamespace

from core.route2_xeditcritic_ledger_v3 import POSTTEST_STUDIES_V3
from scripts.route_a_v3.adjudicate_route2_xeditcritic_v3_posttest import (
    adjudicate_loso_jobs_v3,
    adjudicate_refits_v3,
)
from scripts.route_a_v3 import prepare_route2_xeditcritic_v3_posttest_configs as prepare


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _authority(tmp_path):
    three = tmp_path / "three.json"
    atomic = tmp_path / "atomic.json"
    _write(three, {"status": "XEDITCRITIC_V3_THREE_SEED_PASS", "selected_arm": "C2"})
    _write(atomic, {
        "status": "ATOMIC_FROZEN_DEVELOPMENT_TEST_TERMINAL",
        "frozen_test_gate": {
            "status": "XEDITCRITIC_V3_FROZEN_TEST_PASS",
            "all_development_refit_authorized": True,
        },
    })
    return three, atomic


def _base(tmp_path):
    three, atomic = _authority(tmp_path)
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_posttest_prepare.v1",
        "three_seed_gate_path": str(three),
        "atomic_frozen_test_path": str(atomic),
        "projection_paths": ["/mnt/train.jsonl", "/mnt/validation.jsonl"],
        "edit_site_cache": "/mnt/cache.pt",
        "expected_record_count": 107873,
        "experiment_ledger_path": "/mnt/ledger.csv",
        "physical_gpu_indices": [0, 1, 2],
        "training_template": {
            "screen_seed": 20260830,
            "batch_size": 32,
            "head_learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "huber_delta": 1.0,
        },
    }


def test_refit_freezes_confirmation_selected_pass_median_and_three_jobs(tmp_path) -> None:
    config = {**_base(tmp_path), "mode": "REFIT", "output_root": "/mnt/refit"}
    specs = []
    for seed, selected_pass in zip((20260831, 20260901, 20260902), (7, 8, 8), strict=True):
        path = tmp_path / f"seed{seed}.json"
        _write(path, {
            "status": "TERMINAL_CONFIRMATION_ARM_COMPLETE",
            "arm": "C2", "seed": seed, "selected_pass": selected_pass,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        })
        specs.append({"seed": seed, "summary_path": str(path)})
    config["confirmation_candidate_summaries"] = specs
    manifest = prepare.prepare_refit_configs_v3(config)
    assert manifest["refit_pass_count"] == 8
    assert manifest["job_count"] == 3
    assert {job["config"]["expected_train_count"] for job in manifest["jobs"]} == {107873}
    assert {job["config"]["expected_validation_count"] for job in manifest["jobs"]} == {0}


class _StudyRecords:
    def __len__(self):
        return 107873

    def __iter__(self):
        base = 107873 // 7
        remaining = 107873 - base * 7
        for index, study in enumerate(POSTTEST_STUDIES_V3):
            for _ in range(base + int(index < remaining)):
                yield SimpleNamespace(study=study)


def test_loso_preparer_emits_three_by_seven_by_two_paired_jobs(tmp_path, monkeypatch) -> None:
    config = {**_base(tmp_path), "mode": "LOSO", "output_root": "/mnt/loso"}
    refit = tmp_path / "refit.json"
    _write(refit, {
        "status": "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "required_seeds": [20260831, 20260901, 20260902],
        "completed_refit_count": 3,
        "refit_pass_count": 8,
    })
    config["refit_manifest_path"] = str(refit)
    monkeypatch.setattr(prepare, "load_projection_rows", lambda paths: [])
    monkeypatch.setattr(prepare, "records_from_projection_rows", lambda rows: _StudyRecords())
    manifest = prepare.prepare_loso_configs_v3(config)
    assert manifest["job_count"] == 42
    assert set(manifest["held_out_studies"]) == set(POSTTEST_STUDIES_V3)
    identities = {(job["seed"], job["held_out_study"], job["arm"]) for job in manifest["jobs"]}
    assert len(identities) == 42


def test_refit_and_loso_adjudicators_close_exact_artifact_inventories(tmp_path) -> None:
    refit_jobs = []
    for seed in (20260831, 20260901, 20260902):
        path = tmp_path / f"refit-{seed}.json"
        _write(path, {
            "status": "TERMINAL_REFIT_ARM_COMPLETE", "arm": "C2", "seed": seed,
            "train_record_count": 107873, "validation_record_count": 0,
            "selected_pass": 8, "checkpoint_path": f"/mnt/{seed}.pt",
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        })
        refit_jobs.append({"seed": seed, "summary_path": str(path)})
    refit = adjudicate_refits_v3({
        "status": "XEDITCRITIC_V3_REFIT_CONFIGS_PREPARED",
        "selected_arm": "C2", "refit_pass_count": 8, "jobs": refit_jobs,
    })
    assert refit["completed_refit_count"] == 3 and refit["loso_authorized"] is True

    loso_jobs = []
    for seed in (20260831, 20260901, 20260902):
        for study in POSTTEST_STUDIES_V3:
            for arm, value in (("C2", 0.4), ("C0", 0.2)):
                path = tmp_path / f"{seed}-{study}-{arm}.json"
                _write(path, {
                    "status": "TERMINAL_LOSO_ARM_COMPLETE", "seed": seed,
                    "arm": arm, "held_out_study": study,
                    "held_out_study_scale_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
                    "final_validation": {"task_macro_spearman": value},
                    "development_test_outcomes_accessed": False,
                    "new_final_evaluation_outcomes_accessed": False,
                })
                loso_jobs.append({
                    "seed": seed, "held_out_study": study, "arm": arm,
                    "summary_path": str(path),
                })
    result = adjudicate_loso_jobs_v3({
        "status": "XEDITCRITIC_V3_LOSO_CONFIGS_PREPARED",
        "selected_arm": "C2", "jobs": loso_jobs,
    })
    assert result["loso_gate"]["status"] == "XEDITCRITIC_V3_LOSO_PASS"
    assert result["loso_gate"]["guidance_readiness_authorized"] is True
