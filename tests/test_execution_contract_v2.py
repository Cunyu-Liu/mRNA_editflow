from pathlib import Path

import json
import yaml

from scripts.execution.monitor_run import evaluate
from scripts.execution.preflight import collect


ROOT = Path(__file__).resolve().parents[1]


def test_execution_contract_freezes_artifacts_and_monitor_cadence():
    contract = yaml.safe_load(
        (ROOT / "configs/execution_contract.yaml").read_text(encoding="utf-8")
    )
    assert contract["parent_contract_id"] == "mrna_editflow_single_active_contract"
    assert contract["formal_neural_training"]["device"] == "cuda"
    assert contract["formal_neural_training"]["cpu_fallback_allowed"] is False
    assert len(contract["formal_neural_training"]["health_fields"]) == 8
    assert contract["monitoring"]["initial_health_check_after_minutes"] == 3
    assert contract["monitoring"]["initial_health_check_deadline_minutes"] == 5
    assert contract["monitoring"]["semantic_check_min_interval_minutes"] == 30
    assert contract["monitoring"]["continuous_tail_forbidden"] is True
    for required in (
        "status.json",
        "run_manifest.json",
        "logs/metrics.jsonl",
        "logs/system_metrics.jsonl",
        "logs/cuda_preflight.json",
        "checkpoints/checksums.sha256",
    ):
        assert required in contract["required_artifacts"]


def test_run_manifest_schema_requires_provenance_and_gpu_evidence():
    schema = json.loads(
        (ROOT / "schemas/run_manifest.schema.json").read_text(encoding="utf-8")
    )
    required = set(schema["required"])
    assert {
        "goal_contract",
        "data_manifest_sha256",
        "split_manifest_sha256",
        "foundation_checkpoint_sha256",
        "exposure_ledger_version",
        "gpu",
        "artifact_checksums",
    } <= required
    assert schema["properties"]["gpu"]["properties"]["max_memory_allocated"][
        "minimum"
    ] == 1


def test_monitor_fails_closed_on_non_finite_metric(tmp_path):
    metrics = tmp_path / "logs/metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"step": 1, "loss": NaN}\n', encoding="utf-8")
    report = evaluate(
        tmp_path,
        now_epoch=metrics.stat().st_mtime,
        stall_seconds=900,
    )
    assert report["state_recommendation"] == "SAFE_PAUSED"
    assert report["reasons"] == ["NON_FINITE_METRIC"]
    assert report["automatic_process_termination"] is False


def test_monitor_reports_stall_without_killing_process(tmp_path):
    metrics = tmp_path / "logs/metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"step": 1, "loss": 1.0}\n', encoding="utf-8")
    report = evaluate(
        tmp_path,
        now_epoch=metrics.stat().st_mtime + 901,
        stall_seconds=900,
    )
    assert report["state_recommendation"] == "SAFE_PAUSED"
    assert report["reasons"] == ["STALL_HEARTBEAT"]
    assert report["automatic_process_termination"] is False


def test_preflight_records_memory_disk_git_process_and_gpu_evidence(tmp_path):
    report = collect(tmp_path)
    assert report["mutations_performed"] == 0
    assert set(report["cpu_memory"]) == {
        "total_bytes",
        "available_bytes",
        "swap_total_bytes",
        "swap_free_bytes",
    }
    assert report["disk"]["free_bytes"] > 0
    assert {
        "git_head",
        "git_status",
        "nvidia_smi",
        "nvidia_smi_summary",
        "gpu_processes",
        "processes",
    } <= set(report["commands"])
