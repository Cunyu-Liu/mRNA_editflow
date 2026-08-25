from __future__ import annotations

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v4_loso_after_refits as launcher


def _manifest() -> dict[str, object]:
    jobs = [
        {"seed": seed, "held_out_study": study, "run_id": run_id}
        for seed in launcher.SEEDS
        for study in launcher.STUDIES
        for run_id in ("v4_full", "c0_v4")
    ]
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_job_manifest.v1",
        "status": "XEDITCRITIC_V4_LOSO_CONFIGS_PREPARED_NOT_STARTED",
        "required_seeds": list(launcher.SEEDS),
        "held_out_studies": list(launcher.STUDIES),
        "job_count": 42,
        "jobs": jobs,
    }


def test_loso_manifest_is_exact_42_paired_jobs() -> None:
    assert len(launcher.validate_loso_manifest(_manifest())) == 42
    payload = _manifest()
    payload["jobs"] = payload["jobs"][:-1]
    with pytest.raises(Exception, match="job set changed"):
        launcher.validate_loso_manifest(payload)


def test_loso_gpu_selection_uses_all_eligible_zero_to_five() -> None:
    free = {0: 31_000, 1: 36_000, 2: 35_000, 3: 20_000, 4: 34_000, 5: 33_000, 6: 90_000}
    assert launcher.eligible_loso_gpus(free, required_mib=30_000) == (1, 2, 4, 5, 0)
    with pytest.raises(Exception, match="no GPU 0–5"):
        launcher.eligible_loso_gpus(
            {gpu: 10_000 for gpu in range(8)}, required_mib=30_000
        )


def test_loso_launcher_uses_formal_current_head_scheduler() -> None:
    assert launcher.LOSO_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xeditcritic_v4_loso_scheduler.py"
    )
