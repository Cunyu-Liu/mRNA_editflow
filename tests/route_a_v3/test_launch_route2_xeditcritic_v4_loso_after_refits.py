from __future__ import annotations

from pathlib import Path

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


def test_loso_gpu_selection_uses_frozen_protocol_order_without_memory_gate() -> None:
    inventory = {gpu: 1 for gpu in range(6)}
    assert launcher.eligible_loso_gpus((5, 2, 0, 1, 3, 4), inventory) == (
        5, 2, 0, 1, 3, 4
    )
    with pytest.raises(Exception, match="absent"):
        launcher.eligible_loso_gpus((0, 5), {0: 100_000})


def test_loso_launcher_uses_formal_current_head_scheduler() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    assert launcher.LOSO_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xeditcritic_v4_loso_scheduler.py"
    )


def test_loso_records_memory_without_filtering_or_sorting() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert '"diagnostic_peak_plus_two_gib_mib"' in source
    assert "key=lambda gpu" not in source
