"""Negative cross-field tests for the four public runtime record contracts."""

from __future__ import annotations

import json
from copy import deepcopy


def _codes(issues):
    return {issue.code for issue in issues}


def _pool(validator):
    common = {
        "biological_source_group_id": "source-group-1",
        "study_id": "study-1",
        "assay_id": "assay-1",
        "context_id": "context-1",
        "endpoint_id": "endpoint-1",
        "region": "3UTR",
    }
    candidates = []
    for index, sequence in enumerate(("AAAA", "CCCC", "GGGG"), start=1):
        candidates.append(
            {
                "id": f"candidate-{index}",
                "canonical_record_id": f"record-{index}",
                "sequence": sequence,
                "sequence_hash": validator.sha256_bytes(sequence.encode("utf-8")),
                **common,
            }
        )
    return {"pool_type": "NDCG_ELIGIBLE", "candidate_count": 3, "candidates": candidates, **common}


def _ledger(validator):
    rule_without_hash = {
        "rule_id": "forward-v1",
        "generator_weight": 1.0,
        "critic_weight": 2.0,
        "guidance_weight": 3.0,
        "reranker_weight": 4.0,
        "other_weight": 5.0,
    }
    rule_hash = validator.sha256_bytes(
        json.dumps(rule_without_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    return {
        "source_pool_hash": "1" * 64,
        "legal_action_space_hash": "2" * 64,
        "candidate_budget": 4,
        "candidate_count": 3,
        "unique_candidate_count": 2,
        "unique_candidate_rate": 2 / 3,
        "generator_nfe": 2,
        "critic_forwards": 3,
        "guidance_forwards": 5,
        "reranker_forwards": 7,
        "other_forwards": 11,
        "total_forward_equivalents": 2 + 6 + 15 + 28 + 55,
        "forward_equivalent_rule": {**rule_without_hash, "rule_sha256": rule_hash},
        "seeds": [11, 22],
        "hpo_budget": {
            "trial_count": 2,
            "max_trials": 4,
            "budget_type": "JOINT",
            "time_budget_seconds": 120.0,
            "forward_equivalent_budget": 500.0,
            "search_space_sha256": "3" * 64,
        },
    }


def _critic_gate():
    return {
        "gate_family": "CRITIC_EFFECT",
        "claim_eligible": True,
        "decision": "PASS",
        "evidence_status": "PASS",
        "claim_status": "ESTABLISHED",
        "run_ids": ["run-1"],
        "seeds": [1, 2, 3, 4, 5],
        "per_study_results": [
            {"study_id": "study-1"},
            {"study_id": "study-2"},
            {"study_id": "study-3"},
        ],
        "failure_bundle": None,
        "next_route_a_recovery_task": None,
    }


def _completed_run():
    return {
        "compute_class": "GPU_VALIDATION",
        "parameter_updating": False,
        "run_status": "COMPLETED",
        "evidence_status": "PASS",
        "claim_status": "NOT_ESTABLISHED",
        "gpu": {
            "required": True,
            "used": True,
            "cuda_fail_closed": True,
            "silent_cpu_fallback": False,
            "uuid": "GPU-uuid",
            "model": "A100",
            "device": "cuda:0",
            "driver_version": "550.0",
            "cuda_version": "12.1",
            "peak_vram_bytes": 1024,
        },
        "environment": {"pytorch_version": "2.5.0"},
        "outputs": [{"status": "COMPLETE", "absolute_path": "/tmp/result.json", "sha256": "4" * 64}],
        "failure": None,
        "recovery": None,
        "ended_at": "2026-08-10T01:00:00+08:00",
    }


def _completed_cpu(compute_class):
    return {
        "compute_class": compute_class,
        "parameter_updating": False,
        "run_status": "COMPLETED",
        "evidence_status": "PASS",
        "claim_status": "NOT_ESTABLISHED",
        "gpu": {
            "required": False,
            "used": False,
            "cuda_fail_closed": True,
            "silent_cpu_fallback": False,
            "uuid": None,
            "model": None,
            "device": None,
            "driver_version": None,
            "cuda_version": None,
            "peak_vram_bytes": None,
        },
        "environment": {"pytorch_version": None},
        "outputs": [{"status": "COMPLETE", "absolute_path": "/tmp/cpu-result.json", "sha256": "6" * 64}],
        "failure": None,
        "recovery": None,
        "ended_at": "2026-08-10T01:00:00+08:00",
    }


def _gpu_lifecycle_record(status, *, used, failure_type=None):
    active_metadata = {
        "uuid": "GPU-lifecycle-uuid" if used else None,
        "model": "A100" if used else None,
        "device": "cuda:0" if used else None,
        "driver_version": "550.0" if used else None,
        "cuda_version": "12.1" if used else None,
        "peak_vram_bytes": 2048 if used else None,
    }
    failure_status = status in {
        "FAIL_CLOSED",
        "FAIL_CURRENT_PROTOCOL",
        "FAIL_REPAIRABLE",
        "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "TERMINATED",
        "TERMINATED_SAFELY_WITH_EVIDENCE",
    }
    return {
        "compute_class": "GPU_NEURAL_CRITIC_TRAIN",
        "parameter_updating": True,
        "run_status": status,
        "evidence_status": status if status in {"FAIL_CURRENT_PROTOCOL", "FAIL_REPAIRABLE", "BLOCKED_PENDING_PUBLIC_EVIDENCE", "TERMINATED_SAFELY_WITH_EVIDENCE"} else ("IN_PROGRESS" if status == "IN_PROGRESS" else "NOT_RUN"),
        "claim_status": "NOT_ESTABLISHED",
        "gpu": {
            "required": True,
            "used": used,
            "cuda_fail_closed": True,
            "silent_cpu_fallback": False,
            **active_metadata,
        },
        "environment": {"pytorch_version": "2.5.0" if used else None},
        "outputs": [],
        "failure": (
            {
                "failure_type": failure_type or "COMPUTE",
                "detail": "frozen failure fixture",
                "failure_bundle_path": "/tmp/failure.json",
                "failure_bundle_sha256": "7" * 64,
                "last_valid_checkpoint_sha256": None,
            }
            if failure_status
            else None
        ),
        "recovery": (
            {
                "recovery_branch": "recover-gpu",
                "next_route_a_task_id": "RECOVER_GPU",
                "hypothesis": "retry only after the frozen failure is repaired",
                "new_run_id_required": True,
            }
            if failure_status
            else None
        ),
        "ended_at": "2026-08-10T01:00:00+08:00" if failure_status else None,
    }


def test_measured_pool_single_representation_and_common_endpoint(validator):
    record = _pool(validator)
    assert validator.validate_measured_candidate_pool_record(record) == []

    record["candidate_ids"] = [row["id"] for row in record["candidates"]]
    record["candidate_count"] = 2
    record["candidates"][1]["endpoint_id"] = "endpoint-bypass"
    record["candidates"][2]["id"] = record["candidates"][0]["id"]
    codes = _codes(validator.validate_measured_candidate_pool_record(record))
    assert {
        "POOL_PARALLEL_REPRESENTATION",
        "POOL_COUNT_MISMATCH",
        "POOL_COMMON_KEY_MISMATCH",
        "POOL_DUPLICATE_CANDIDATE_ID",
    } <= codes


def test_measured_pool_cardinality_and_sequence_hash_are_enforced(validator):
    record = _pool(validator)
    record["pool_type"] = "PAIRWISE_ONLY"
    record["candidates"][0]["sequence_hash"] = "0" * 64
    codes = _codes(validator.validate_measured_candidate_pool_record(record))
    assert "POOL_PAIRWISE_SIZE" in codes
    assert "POOL_CANDIDATE_HASH_MISMATCH" in codes


def test_compute_ledger_arithmetic_and_bindings_are_enforced(validator):
    record = _ledger(validator)
    assert validator.validate_compute_ledger_record(record) == []

    record["source_pool_hash"] = "not-a-hash"
    record["unique_candidate_count"] = 5
    record["unique_candidate_rate"] = 0.1
    record["total_forward_equivalents"] += 1
    record["forward_equivalent_rule"]["critic_weight"] = 99
    record["hpo_budget"]["trial_count"] = 10
    record["hpo_budget"]["time_budget_seconds"] = 0
    codes = _codes(validator.validate_compute_ledger_record(record))
    assert {
        "COMPUTE_BINDING_HASH",
        "COMPUTE_CANDIDATE_INEQUALITY",
        "COMPUTE_UNIQUE_RATE",
        "COMPUTE_FORWARD_RULE_HASH",
        "COMPUTE_FORWARD_EQUIVALENT",
        "COMPUTE_HPO_TRIALS",
        "COMPUTE_HPO_TIME_BUDGET",
    } <= codes


def test_gate_pass_and_critic_sufficiency_are_enforced(validator):
    record = _critic_gate()
    assert validator.validate_gate_record(record) == []

    record["seeds"] = [1, 1]
    record["per_study_results"] = [{"study_id": "study-1"}, {"study_id": "study-1"}]
    codes = _codes(validator.validate_gate_record(record))
    assert {"GATE_PASS_SEEDS", "GATE_CRITIC_SEEDS", "GATE_CRITIC_STUDIES"} <= codes


def test_gate_not_run_and_failure_cannot_establish_claim(validator):
    not_run = {
        "decision": "NOT_RUN",
        "evidence_status": "PASS",
        "claim_status": "ESTABLISHED",
        "claim_eligible": True,
    }
    codes = _codes(validator.validate_gate_record(not_run))
    assert {"GATE_DECISION_EVIDENCE", "GATE_NONPASS_CLAIM", "GATE_NOT_RUN_CLAIM"} <= codes

    failed = {
        "decision": "FAIL_CURRENT_PROTOCOL",
        "evidence_status": "FAIL_CURRENT_PROTOCOL",
        "claim_status": "NOT_ESTABLISHED",
        "claim_eligible": False,
        "failure_bundle": None,
        "next_route_a_recovery_task": None,
    }
    codes = _codes(validator.validate_gate_record(failed))
    assert {"GATE_FAILURE_BUNDLE", "GATE_RECOVERY_TASK"} <= codes


def test_completed_or_pass_run_requires_real_gpu_and_complete_outputs(validator):
    record = _completed_run()
    assert validator.validate_run_manifest_record(record) == []

    record["gpu"]["required"] = False
    record["gpu"]["used"] = True
    record["gpu"]["uuid"] = None
    record["gpu"]["device"] = "cpu"
    record["gpu"]["peak_vram_bytes"] = 0
    record["environment"]["pytorch_version"] = None
    record["outputs"] = [{"status": "PARTIAL", "absolute_path": "relative", "sha256": None}]
    codes = _codes(validator.validate_run_manifest_record(record))
    assert {
        "RUN_GPU_REQUIRED_POLICY",
        "RUN_GPU_METADATA",
        "RUN_GPU_DEVICE",
        "RUN_GPU_VRAM",
        "RUN_PYTORCH_VERSION",
        "RUN_SUCCESS_OUTPUT",
    } <= codes


def test_cpu_authority_data_and_exact_runs_may_complete_without_gpu(validator):
    for compute_class in ("CPU_AUTHORITY", "CPU_DATA", "CPU_SMALL_GRAPH_EXACT"):
        assert validator.validate_run_manifest_record(_completed_cpu(compute_class)) == []


def test_cpu_run_cannot_lie_about_gpu_parameter_updates_or_claim(validator):
    record = _completed_cpu("CPU_STATISTICS")
    record["parameter_updating"] = True
    record["gpu"]["required"] = True
    record["gpu"]["used"] = True
    record["claim_status"] = "ESTABLISHED"
    codes = _codes(validator.validate_run_manifest_record(record))
    assert {
        "RUN_CPU_PARAMETER_UPDATE",
        "RUN_CPU_GPU_POLICY",
        "RUN_CPU_CLAIM",
        "RUN_PARAMETER_UPDATE_CLASS",
    } <= codes


def test_gpu_training_class_cannot_bypass_parameter_or_gpu_metadata(validator):
    record = _completed_run()
    record["compute_class"] = "GPU_NEURAL_CRITIC_TRAIN"
    record["parameter_updating"] = False
    record["gpu"].update(
        {
            "required": False,
            "used": True,
            "uuid": None,
            "model": None,
            "device": None,
            "driver_version": None,
            "cuda_version": None,
            "peak_vram_bytes": None,
        }
    )
    codes = _codes(validator.validate_run_manifest_record(record))
    assert {
        "RUN_GPU_TRAIN_PARAMETER_UPDATE",
        "RUN_GPU_REQUIRED_POLICY",
        "RUN_GPU_METADATA",
        "RUN_GPU_DEVICE",
        "RUN_GPU_VRAM",
    } <= codes


def test_gpu_not_run_and_queued_do_not_fake_gpu_use(validator):
    for status in ("NOT_RUN", "QUEUED"):
        record = _gpu_lifecycle_record(status, used=False)
        assert record["parameter_updating"] is True
        assert validator.validate_run_manifest_record(record) == []

        record["gpu"]["used"] = True
        record["gpu"].update(
            {
                "uuid": "GPU-fake",
                "model": "A100",
                "device": "cuda:0",
                "driver_version": "550",
                "cuda_version": "12.1",
                "peak_vram_bytes": 1,
            }
        )
        record["environment"]["pytorch_version"] = "2.5"
        record["ended_at"] = "2026-08-10T01:00:00+08:00"
        codes = _codes(validator.validate_run_manifest_record(record))
        assert {"RUN_GPU_PRESTART_POLICY", "RUN_GPU_PRESTART_ENDED_AT"} <= codes


def test_gpu_in_progress_requires_real_gpu_but_no_end_timestamp(validator):
    record = _gpu_lifecycle_record("IN_PROGRESS", used=True)
    assert validator.validate_run_manifest_record(record) == []

    record["gpu"].update({"used": False, "device": None, "peak_vram_bytes": 0})
    record["ended_at"] = "2026-08-10T01:00:00+08:00"
    codes = _codes(validator.validate_run_manifest_record(record))
    assert {"RUN_GPU_IN_PROGRESS_POLICY", "RUN_GPU_IN_PROGRESS_ENDED_AT"} <= codes


def test_non_cuda_failure_records_pre_and_post_gpu_initialization(validator):
    pre_init = _gpu_lifecycle_record("FAIL_REPAIRABLE", used=False, failure_type="COMPUTE")
    post_init = _gpu_lifecycle_record("FAIL_REPAIRABLE", used=True, failure_type="OOM")
    assert validator.validate_run_manifest_record(pre_init) == []
    assert validator.validate_run_manifest_record(post_init) == []

    pre_init["gpu"]["device"] = "cuda:0"
    pre_init["gpu"]["peak_vram_bytes"] = 4096
    assert "RUN_GPU_UNUSED_TELEMETRY" in _codes(validator.validate_run_manifest_record(pre_init))

    post_init["gpu"]["uuid"] = None
    post_init["gpu"]["device"] = None
    post_init["gpu"]["peak_vram_bytes"] = 0
    post_init["environment"]["pytorch_version"] = None
    codes = _codes(validator.validate_run_manifest_record(post_init))
    assert {"RUN_GPU_METADATA", "RUN_GPU_DEVICE", "RUN_GPU_VRAM", "RUN_PYTORCH_VERSION"} <= codes


def test_cuda_unavailable_or_fallback_must_fail_closed_with_bundle(validator):
    record = {
        "compute_class": "GPU_NEURAL_CRITIC_TRAIN",
        "parameter_updating": True,
        "run_status": "FAIL_CLOSED",
        "evidence_status": "FAIL_CURRENT_PROTOCOL",
        "claim_status": "NOT_ESTABLISHED",
        "gpu": {
            "required": True,
            "used": False,
            "cuda_fail_closed": True,
            "silent_cpu_fallback": False,
            "uuid": None,
            "model": None,
            "device": None,
            "driver_version": None,
            "cuda_version": None,
            "peak_vram_bytes": None,
        },
        "failure": {
            "failure_type": "CUDA_UNAVAILABLE",
            "failure_bundle_path": "/tmp/failure.json",
            "failure_bundle_sha256": "5" * 64,
        },
        "recovery": {"next_route_a_task_id": "RECOVER_GPU"},
        "ended_at": "2026-08-10T01:00:00+08:00",
    }
    assert validator.validate_run_manifest_record(record) == []

    record["run_status"] = "COMPLETED"
    record["gpu"]["used"] = True
    record["failure"]["failure_bundle_sha256"] = None
    record["recovery"] = None
    codes = _codes(validator.validate_run_manifest_record(record))
    assert {"RUN_CUDA_FAILURE_STATUS", "RUN_CUDA_FAILURE_GPU", "RUN_CUDA_FAILURE_BUNDLE", "RUN_FAILURE_RECOVERY"} <= codes


def test_schema_exposes_required_cross_field_contracts(validator, repo_root):
    schema_dir = repo_root / validator.SCHEMA_DIR
    compute = json.loads((schema_dir / "compute_ledger.schema.json").read_text(encoding="utf-8"))
    assert {"source_pool_hash", "legal_action_space_hash", "unique_candidate_rate", "hpo_budget"} <= set(compute["required"])
    gate = json.loads((schema_dir / "gate_record.schema.json").read_text(encoding="utf-8"))
    assert {"gate_family", "claim_eligible"} <= set(gate["required"])
    pool = json.loads((schema_dir / "measured_candidate_pool.schema.json").read_text(encoding="utf-8"))
    assert "candidates" in pool["required"]
    assert not {"candidate_ids", "candidate_sequence_sha256s", "same_pool_constraints"} & set(pool["properties"])
    run = json.loads((schema_dir / "run_manifest.schema.json").read_text(encoding="utf-8"))
    assert {"compute_class", "parameter_updating"} <= set(run["required"])
    assert set(run["properties"]["compute_class"]["enum"]) == set(validator.RUN_COMPUTE_CLASSES)
