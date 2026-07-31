from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.execution.validate_stage_manifest import GOAL_SHA256, validate


ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    return {
        "artifact_type": "stage_manifest",
        "schema_version": "utr_stage_manifest.v1",
        "stage_id": "D1_B0_20260728T160012Z_8862125",
        "phase_ids": ["D1", "B0"],
        "captured_at_utc": "2026-07-28T16:00:12Z",
        "workload_class": "NON_NEURAL_DATA_BENCHMARK",
        "goal_contract": {
            "id": "mrna_editflow_single_active_contract",
            "sha256": GOAL_SHA256,
            "source_path_local": "/local/goal.md",
            "repository_snapshot": "docs/contracts/mrna_editflow_contract.md",
        },
        "remote": {
            "host": "36.137.135.49",
            "port": 22,
            "user": "cunyuliu",
            "hostname": "server",
            "original_project_root": "/home/cunyuliu/mrna_editflow_goal/mrna_editflow",
            "isolated_worktree": "/mnt/cunyuliu/worktree",
            "external_data_root": "/mnt/cunyuliu/data",
        },
        "git": {
            "original": {
                "path": "/original",
                "branch": "existing",
                "head": "1" * 40,
                "clean": False,
                "dirty_diff_sha256": "2" * 64,
            },
            "isolated": {
                "path": "/isolated",
                "branch": "stage",
                "head": "3" * 40,
                "clean": True,
                "dirty_diff_sha256": None,
            },
        },
        "protection": {
            "protected_processes": [
                {"pid": 123, "kind": "existing_download", "action": "observe_only"}
            ],
            "processes_terminated": 0,
            "original_worktree_mutations": 0,
            "existing_results_overwritten": 0,
        },
        "resources": {
            "gpu": {
                "count": 8,
                "model": "NVIDIA A100-PCIE-40GB",
                "driver": "580.126.09",
                "formal_neural_work_planned": False,
            },
            "memory": {
                "total_bytes": 100,
                "available_bytes": 90,
                "swap_total_bytes": 10,
                "swap_free_bytes": 0,
            },
            "disk": {
                "home": {"path": "/home", "available_bytes": 100},
                "mnt": {"path": "/mnt", "available_bytes": 200},
            },
        },
        "data_state": {
            "input_inventory": "artifacts/stages/example/D1/input_inventory.json",
            "existing_artifact_inventory": "artifacts/stages/example/inventory.json",
            "encode_reconstruction": {
                "expected_files": 62,
                "verified_files": 61,
                "complete": False,
                "role": "OBSERVATIONAL_ONLY",
            },
        },
        "execution_boundary": {
            "formal_neural_activity_started": False,
            "gpu_validation_started": False,
            "cuda_fallback_events": 0,
            "gpu_requirement_status": "NOT_APPLICABLE_NO_NEURAL_WORK",
            "smoke_or_proxy_is_final_evidence": False,
            "d1_required_before_b0": True,
        },
    }


def test_stage_manifest_schema_is_valid_json_schema():
    schema = json.loads(
        (ROOT / "schemas/stage_manifest.schema.json").read_text(encoding="utf-8")
    )
    assert schema["properties"]["workload_class"]["const"] == (
        "NON_NEURAL_DATA_BENCHMARK"
    )
    assert (
        schema["properties"]["execution_boundary"]["properties"][
            "formal_neural_activity_started"
        ]["const"]
        is False
    )


def test_valid_non_neural_stage_manifest_passes():
    assert validate(_manifest()) == []


def test_schema_rejects_invalid_time_remote_and_extra_fields():
    manifest = _manifest()
    manifest["captured_at_utc"] = "2026-07-28T16:00:12"
    manifest["remote"] = {}
    manifest["unsealed_assertion"] = True
    errors = validate(manifest)
    assert any(
        "schema:captured_at_utc" in error and "date-time" in error for error in errors
    )
    assert any(error.startswith("schema:remote") for error in errors)
    assert any(
        "schema:<root>" in error and "Additional properties" in error
        for error in errors
    )


def test_stage_manifest_allows_auditable_retry_suffix():
    manifest = _manifest()
    manifest["stage_id"] += "_A2"
    assert validate(manifest) == []


def test_stage_id_rejects_invalid_calendar_timestamp():
    manifest = _manifest()
    manifest["stage_id"] = "D1_B0_20261340T250061Z_8862125"
    assert any(
        "invalid UTC calendar timestamp" in error for error in validate(manifest)
    )


def test_dirty_original_requires_diff_hash():
    manifest = _manifest()
    manifest["git"]["original"]["dirty_diff_sha256"] = None
    assert any("dirty state requires" in error for error in validate(manifest))


def test_stage_manifest_cannot_fabricate_gpu_work():
    manifest = _manifest()
    manifest["execution_boundary"]["gpu_validation_started"] = True
    assert any("gpu_validation_started" in error for error in validate(manifest))


def test_original_state_and_jobs_are_fail_closed():
    manifest = _manifest()
    manifest["protection"]["processes_terminated"] = 1
    assert any("processes_terminated" in error for error in validate(manifest))


def test_encode_cannot_be_promoted_to_intervention():
    manifest = copy.deepcopy(_manifest())
    manifest["data_state"]["encode_reconstruction"]["role"] = "INTERVENTION"
    assert any("OBSERVATIONAL_ONLY" in error for error in validate(manifest))
