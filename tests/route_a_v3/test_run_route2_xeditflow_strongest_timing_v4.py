from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

import scripts.route_a_v3.run_route2_xeditflow_strongest_timing_v4 as timing


def test_producer_cuda_failure_is_bridged_to_declared_final_job_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "timing"
    declared_failure = tmp_path / "failures" / "strongest_timing.failed.json"
    config = {
        "strongest_generation_baseline_path": str(tmp_path / "strongest.json"),
        "baseline_selection_input_path": str(tmp_path / "selection.json"),
        "output_dir": str(output_dir),
        "source_manifest_path": str(tmp_path / "sources.jsonl"),
        "guiding_checkpoint_path": str(tmp_path / "critic.pt"),
        "device": "cuda:2",
        "physical_gpu_index": 2,
        "critic_forward_budget_per_source": 320,
        "beam_width": 16,
        "genetic_population_size": 32,
        "oversample_factor": 8,
        "exhaustive_space_limit": 4096,
        "seed": 20260816,
    }
    original_json = timing._json
    monkeypatch.setattr(
        timing,
        "_json",
        lambda path: (
            original_json(path)
            if str(path).endswith(".failed.json")
            else {}
        ),
    )
    monkeypatch.setattr(
        timing, "validate_strongest_timing_config_v4", lambda *args: None
    )

    def fail_after_producer_evidence(command, **kwargs):
        candidate = Path(command[command.index("--output") + 1])
        producer_failure = candidate.with_suffix(
            candidate.suffix + ".failed.json"
        )
        producer_failure.write_text(
            json.dumps(
                {
                    "status": "CUDA_SILENT_CPU_FALLBACK",
                    "device": "cuda:2",
                    "physical_gpu_index": 2,
                    "cpu_fallback_used": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(timing.subprocess, "run", fail_after_producer_evidence)
    with pytest.raises(subprocess.CalledProcessError):
        timing.run(config, failure_path=declared_failure)

    evidence = json.loads(declared_failure.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == (
        "route_a_v3_route2_xeditflow_v4_final_job_failure.v1"
    )
    assert evidence["status"] == (
        "TERMINAL_STRONGEST_TIMING_PRODUCER_FAILURE"
    )
    assert evidence["job_key"] == "strongest_timing"
    assert evidence["producer_failure_evidence_present"] is True
    assert evidence["producer_failure_evidence"]["status"] == (
        "CUDA_SILENT_CPU_FALLBACK"
    )
    assert evidence["producer_failure_evidence"]["physical_gpu_index"] == 2
    assert evidence["cpu_fallback_used"] is True
    assert not (output_dir / "run_summary.json").exists()
