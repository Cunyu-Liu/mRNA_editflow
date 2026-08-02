"""Negative tests for fail-closed MK0 evidence aggregation."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
from argparse import Namespace
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

from mrna_editflow.core.mk0.acceptance import (
    canonical_json_bytes,
    gate_result_from_runtime_binding,
    sha256_file,
    verify_bound_file,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(
    (ROOT / "configs" / "math" / "math_kernel_v1.yaml").read_text(encoding="utf-8")
)


def _load_finalizer():
    path = ROOT / "scripts" / "mk0" / "finalize_mk0_acceptance.py"
    spec = importlib.util.spec_from_file_location("mk0_finalizer_integrity_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINALIZER = _load_finalizer()


def _load_gpu_runner():
    path = ROOT / "scripts" / "mk0" / "run_mk0_gpu_smoke.py"
    spec = importlib.util.spec_from_file_location("mk0_gpu_runner_integrity_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding(
    gate: dict[str, Any], digest: str, *, failed: bool = False
) -> dict[str, Any]:
    sample_count = gate["sample_count"]
    if sample_count == "RUNTIME_REQUIRED":
        sample_count = 7
    return {
        "gate_id": gate["id"],
        "name": gate["name"],
        "passed": not failed,
        "test_domain": gate["domain"],
        "exhaustive_or_sampled": gate["coverage"],
        "sample_count": sample_count,
        "dtype": str(gate["dtype"]),
        "atol": gate["atol"],
        "rtol": gate["rtol"],
        "seed": gate["seed"],
        "failure_count": int(failed),
        "failure_denominator": sample_count,
        "artifact_path": gate["artifact_path"],
        "artifact_sha256": digest,
        "metrics": {"observed_failure_count": int(failed)},
    }


def _synthetic_gpu_role_query_audit(
    *,
    run_id: str,
    goal_sha256: str,
    implementation_commit: str,
    run_manifest_path: Path,
    run_manifest_sha256: str,
    preflight: dict[str, Any],
    source_binding: dict[str, Any],
) -> dict[str, Any]:
    source_files = source_binding["tracked_source_files"]
    exact_identities = FINALIZER._bound_gpu_role_identities(source_binding)

    def exact_line(source_file: str, qualname: str) -> int:
        lines = FINALIZER._ast_function_lines(source_file)
        assert qualname in lines
        return lines[qualname]

    phase_records: list[dict[str, Any]] = []
    total_counts = {
        f"{category}_call_count": 0 for category in FINALIZER.GPU_ROLE_CALL_CATEGORIES
    }
    total_counts.update(
        {
            "repository_python_call_count": 0,
            "external_python_call_count": 0,
            "unknown_external_call_count": 0,
            "total_python_call_count": 0,
        }
    )
    for phase_id, phase_kind, entrypoint, required in FINALIZER.GPU_ROLE_PHASE_SPECS:
        inventory: list[dict[str, Any]] = []
        if "rate_interface" in required:
            inventory.append(
                {
                    "source_file": "core/mk0/foundation_fusion.py",
                    "function_qualname": "FoundationFusionRateField.forward",
                    "first_lineno": exact_line(
                        "core/mk0/foundation_fusion.py",
                        "FoundationFusionRateField.forward",
                    ),
                    "categories": ["rate_interface"],
                    "call_count": 2,
                }
            )
        if "sampler_interface" in required:
            sampler_name = (
                "paper_first_order_parallel"
                if "paper" in phase_id
                else "constrained_single_event_first_order"
            )
            inventory.append(
                {
                    "source_file": "core/mk0/samplers.py",
                    "function_qualname": sampler_name,
                    "first_lineno": exact_line("core/mk0/samplers.py", sampler_name),
                    "categories": ["sampler_interface"],
                    "call_count": 1,
                }
            )
        if "generator_interface" in required:
            inventory.append(
                {
                    "source_file": "scripts/mk0/run_mk0_gpu_smoke.py",
                    "function_qualname": "_run_forced_action_arm",
                    "first_lineno": exact_line(
                        "scripts/mk0/run_mk0_gpu_smoke.py",
                        "_run_forced_action_arm",
                    ),
                    "categories": ["generator_interface"],
                    "call_count": 1,
                }
            )
        inventory.sort(
            key=lambda record: (
                record["source_file"],
                record["function_qualname"],
                record["first_lineno"],
                tuple(record["categories"]),
            )
        )
        category_counts = {
            category: sum(
                record["call_count"]
                for record in inventory
                if category in record["categories"]
            )
            for category in FINALIZER.GPU_ROLE_CALL_CATEGORIES
        }
        repository_calls = sum(record["call_count"] for record in inventory)
        thread_inventory = [
            {
                "thread_id": 1,
                "thread_name": "MainThread",
                "repository_call_count": repository_calls,
                "external_call_count": 0,
                "total_python_call_count": repository_calls,
            }
        ]
        phase = {
            "phase_id": phase_id,
            "phase_kind": phase_kind,
            "entrypoint": entrypoint,
            "required_call_categories": list(required),
            "completed": True,
            "phase_status": "PASS",
            "failure_reason": None,
            "repository_python_call_count": repository_calls,
            "external_python_call_count": 0,
            "unknown_external_call_count": 0,
            "total_python_call_count": repository_calls,
            "python_thread_count": 1,
            "preexisting_noncurrent_python_thread_count": 0,
            "preexisting_noncurrent_python_threads": [],
            "unjoined_new_thread_ids": [],
            **{
                f"{category}_call_count": count
                for category, count in category_counts.items()
            },
            "call_inventory": inventory,
            "record_stream_sha256": hashlib.sha256(
                canonical_json_bytes(inventory)
            ).hexdigest(),
            "external_call_inventory": [],
            "external_record_stream_sha256": hashlib.sha256(
                canonical_json_bytes([])
            ).hexdigest(),
            "thread_inventory": thread_inventory,
            "thread_record_stream_sha256": hashlib.sha256(
                canonical_json_bytes(thread_inventory)
            ).hexdigest(),
        }
        phase_records.append(phase)
        total_counts["repository_python_call_count"] += repository_calls
        total_counts["total_python_call_count"] += repository_calls
        for category, count in category_counts.items():
            total_counts[f"{category}_call_count"] += count

    interface_sources = {
        **{
            label: "scripts/mk0/run_mk0_gpu_smoke.py"
            for label in FINALIZER.GPU_ROLE_INTERFACE_LABELS
            if label.startswith("gpu_runner.")
        },
        **{
            label: "core/mk0/foundation_fusion.py"
            for label in FINALIZER.GPU_ROLE_INTERFACE_LABELS
            if label.startswith("foundation_fusion.")
        },
        **{
            label: "core/mk0/samplers.py"
            for label in FINALIZER.GPU_ROLE_INTERFACE_LABELS
            if label.startswith("samplers.")
        },
    }
    interface_qualnames = {
        "gpu_runner.forced_action_arm": "_run_forced_action_arm",
        "gpu_runner.paper_sampler_route": "_run_official_paper_sampler_route",
        "gpu_runner.primary_sampler_integration": "_run_primary_gpu_sampler_integration",
        "gpu_runner.target_alignment_leakage_audit": "_audit_target_alignment_leakage",
        "gpu_runner.dynamic_current_encoding_audit": "_audit_dynamic_current_encoding",
        "foundation_fusion.rate_field_forward": "FoundationFusionRateField.forward",
        "foundation_fusion.official_paper_adapter": "OfficialPaperRateAdapter.__call__",
        "samplers.constrained_primary": "constrained_single_event_first_order",
        "samplers.paper_parallel": "paper_first_order_parallel",
        "samplers.replay_constrained": "replay_constrained_result",
        "samplers.replay_paper": "replay_paper_result",
    }
    interfaces = [
        {
            "interface": label,
            "source_file": interface_sources[label],
            "source_file_sha256": source_files[interface_sources[label]],
            "function_qualname": interface_qualnames[label],
            "first_lineno": exact_line(
                interface_sources[label], interface_qualnames[label]
            ),
            "role_categories": list(
                FINALIZER._gpu_role_categories(
                    interface_sources[label],
                    interface_qualnames[label],
                    exact_line(interface_sources[label], interface_qualnames[label]),
                    exact_identities,
                )
            ),
            "parameters": ["state"],
            "prohibited_parameters": [],
        }
        for label in FINALIZER.GPU_ROLE_INTERFACE_LABELS
    ]
    stream_material = [
        {
            "phase_id": phase["phase_id"],
            "call_inventory": phase["call_inventory"],
            "external_call_inventory": phase["external_call_inventory"],
            "thread_inventory": phase["thread_inventory"],
            "preexisting_noncurrent_python_threads": phase[
                "preexisting_noncurrent_python_threads"
            ],
        }
        for phase in phase_records
    ]
    return {
        "schema_version": "mk0_gpu_post_role_query_audit_v1",
        "status": "PASS",
        "placement": "after_all_formal_gpu_generator_rate_sampler_phases_before_support_publication",
        "runtime_instrumentation": "sys_setprofile_python_calls",
        "thread_scope": (
            "formal_gpu_main_and_new_threading_threads_with_"
            "preexisting_noncurrent_threads_forbidden"
        ),
        "external_call_policy": {
            "unknown_external_calls": "FAIL_CLOSED",
            "stdlib_root": str(FINALIZER.STDLIB_ROOT),
            "site_package_roots_excluded_from_stdlib": [
                str(path) for path in FINALIZER.SITE_PACKAGE_ROOTS
            ],
            "frozen_foundation_module_prefixes": list(
                FINALIZER.FROZEN_FOUNDATION_EXTERNAL_MODULE_PREFIXES
            ),
        },
        "run_id": run_id,
        "goal_sha256": goal_sha256,
        "implementation_commit": implementation_commit,
        "run_manifest": {
            "path": str(run_manifest_path),
            "sha256": run_manifest_sha256,
        },
        "preflight": preflight,
        "source_binding_sha256": hashlib.sha256(
            canonical_json_bytes(source_binding)
        ).hexdigest(),
        "tracked_source_files_sha256": source_binding["tracked_source_files_sha256"],
        "required_phase_ids": [item[0] for item in FINALIZER.GPU_ROLE_PHASE_SPECS],
        "formal_gpu_phase_count": len(phase_records),
        "completed_phase_count": len(phase_records),
        "phase_records": phase_records,
        "record_stream_sha256": hashlib.sha256(
            canonical_json_bytes(stream_material)
        ).hexdigest(),
        "audited_interfaces": interfaces,
        "audited_interface_count": len(interfaces),
        "interface_failure_count": 0,
        **total_counts,
        "all_role_query_counts_zero": True,
        "formal_gpu_computation_complete": True,
        "all_external_calls_allowlisted": True,
        "all_new_threads_joined": True,
        "all_preexisting_noncurrent_threads_absent": True,
    }


def test_runtime_binding_requires_exact_config_hash_and_metrics() -> None:
    gate = CONFIG["acceptance"]["gates"][0]
    digest = "a" * 64
    binding = _binding(gate, digest)
    result = gate_result_from_runtime_binding(
        binding, gate, actual_artifact_sha256=digest
    )
    assert result.passed
    assert result.failure_count == 0
    assert result.metrics == {"observed_failure_count": 0}

    failed = _binding(gate, digest, failed=True)
    result = gate_result_from_runtime_binding(
        failed, gate, actual_artifact_sha256=digest
    )
    assert not result.passed
    assert result.failure_count == 1

    for mutation in (
        {"artifact_sha256": "b" * 64},
        {"test_domain": "substituted_domain"},
        {"sample_count": binding["sample_count"] + 1},
        {"metrics": {}},
        {"passed": False},
        {"seed": float(binding["seed"])},
        {"atol": False if gate["atol"] == 0.0 else gate["atol"]},
    ):
        with pytest.raises(ValueError):
            gate_result_from_runtime_binding(
                {**binding, **mutation}, gate, actual_artifact_sha256=digest
            )


def test_bound_file_fails_on_path_size_hash_or_byte_substitution(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected.json"
    alternate = tmp_path / "alternate.json"
    expected.write_bytes(b"evidence\n")
    alternate.write_bytes(b"evidence\n")
    digest = sha256_file(expected)
    evidence = verify_bound_file(
        expected,
        expected_path=expected,
        expected_sha256=digest,
        expected_size_bytes=9,
    )
    assert evidence["sha256"] == digest
    with pytest.raises(ValueError, match="path substitution"):
        verify_bound_file(
            alternate,
            expected_path=expected,
            expected_sha256=digest,
            expected_size_bytes=9,
        )
    with pytest.raises(ValueError, match="size drift"):
        verify_bound_file(
            expected,
            expected_path=expected,
            expected_sha256=digest,
            expected_size_bytes=10,
        )
    expected.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="SHA-256 drift"):
        verify_bound_file(expected, expected_path=expected, expected_sha256=digest)


def test_checksum_ledger_rejects_tamper_duplicate_and_path_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "artifact.json"
    artifact.write_bytes(b"{}\n")
    digest = sha256_file(artifact)
    ledger = root / "checksums.sha256"
    ledger.write_text(f"{digest}  ./artifact.json\n", encoding="utf-8")
    report = FINALIZER.verify_checksum_ledger(root, ledger)
    assert report["verified_entry_count"] == 1

    artifact.write_bytes(b"tamper\n")
    with pytest.raises(FINALIZER.FinalizeFailure, match="checksum mismatch"):
        FINALIZER.verify_checksum_ledger(root, ledger)
    artifact.write_bytes(b"{}\n")
    ledger.write_text(
        f"{digest}  artifact.json\n{digest}  artifact.json\n", encoding="utf-8"
    )
    with pytest.raises(FINALIZER.FinalizeFailure, match="duplicate"):
        FINALIZER.verify_checksum_ledger(root, ledger)
    (root / "nested").mkdir()
    ledger.write_text(
        f"{digest}  artifact.json\n{digest}  nested/../artifact.json\n",
        encoding="utf-8",
    )
    with pytest.raises(FINALIZER.FinalizeFailure, match="duplicate"):
        FINALIZER.verify_checksum_ledger(root, ledger)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}\n")
    ledger.write_text(f"{digest}  ../outside.json\n", encoding="utf-8")
    with pytest.raises(FINALIZER.FinalizeFailure, match="escaped"):
        FINALIZER.verify_checksum_ledger(root, ledger)


def test_gpu_summary_publish_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    runner = _load_gpu_runner()
    summary_path = tmp_path / "summary" / "gpu_acceptance_summary.json"
    payload = {"schema_version": "test_v1", "status": "PASS", "value": 7}
    expected_bytes = canonical_json_bytes(payload)
    expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()

    observed_sha256 = runner._write_canonical_atomic_exclusive(  # noqa: SLF001
        summary_path, payload
    )
    assert observed_sha256 == expected_sha256
    assert summary_path.read_bytes() == expected_bytes
    assert not list(summary_path.parent.glob(f".{summary_path.name}.*.tmp"))

    original_bytes = summary_path.read_bytes()
    with pytest.raises(FileExistsError):
        runner._write_canonical_atomic_exclusive(  # noqa: SLF001
            summary_path, {**payload, "status": "SUBSTITUTED"}
        )
    assert summary_path.read_bytes() == original_bytes
    assert not list(summary_path.parent.glob(f".{summary_path.name}.*.tmp"))


def test_cuda_uuid_mapping_rejects_ordinal_or_properties_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_gpu_runner()
    first = "GPU-00000000-0000-0000-0000-000000000000"
    second = "GPU-11111111-1111-1111-1111-111111111111"

    class Completed:
        stdout = f"0, {first}\n1, {second}\n"

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Completed())

    class Properties:
        uuid = "00000000-0000-0000-0000-000000000000"
        total_memory = 40 * 1024**3

    monkeypatch.setattr(
        runner.torch.cuda, "get_device_properties", lambda device: Properties()
    )
    monkeypatch.setattr(runner.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda device: "Test GPU")
    monkeypatch.setattr(
        runner.torch.cuda, "get_device_capability", lambda device: (8, 0)
    )
    monkeypatch.setattr(runner.torch.backends.cudnn, "version", lambda: 90100)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", first)
    identity = runner._cuda_identity(runner.torch.device("cuda:0"))
    assert identity["device_uuid"] == first
    assert identity["torch_device_properties_uuid_normalized"] == first

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(runner.SmokeFailure, match="single physical GPU UUID"):
        runner._cuda_identity(runner.torch.device("cuda:0"))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", second)
    with pytest.raises(runner.SmokeFailure, match="PyTorch CUDA UUID differs"):
        runner._cuda_identity(runner.torch.device("cuda:0"))


def test_final_gate_assembly_uses_bindings_and_detects_support_tamper(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    cpu_bindings: dict[str, dict[str, Any]] = {}
    gpu_bindings: dict[str, dict[str, Any]] = {}
    for gate in CONFIG["acceptance"]["gates"]:
        support = run_root / gate["artifact_path"]
        support.parent.mkdir(parents=True, exist_ok=True)
        if not support.exists():
            support.write_bytes(canonical_json_bytes({"support": support.name}))
        binding = _binding(gate, sha256_file(support))
        (gpu_bindings if gate["id"] in FINALIZER.GPU_GATE_IDS else cpu_bindings)[
            gate["id"]
        ] = binding
    cpu = {
        "gate_bindings": cpu_bindings,
        "failed_gate_ids": [],
        "status": "PASS_CPU_GATES_PENDING_GPU",
    }
    gpu = {
        "gate_bindings": gpu_bindings,
        "failed_gate_ids": [],
        "status": "PASS_GPU_GATES",
    }
    records = FINALIZER.validated_gate_results(CONFIG, run_root, cpu, gpu)
    assert len(records) == 35 and all(record.passed for record in records)

    tampered = run_root / CONFIG["acceptance"]["gates"][0]["artifact_path"]
    tampered.write_bytes(b"substitution\n")
    with pytest.raises(FINALIZER.FinalizeFailure, match="substituted or tampered"):
        FINALIZER.validated_gate_results(CONFIG, run_root, cpu, gpu)


def test_failed_runtime_binding_cannot_be_summarized_as_pass(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    cpu_bindings: dict[str, dict[str, Any]] = {}
    gpu_bindings: dict[str, dict[str, Any]] = {}
    for gate in CONFIG["acceptance"]["gates"]:
        support = run_root / gate["artifact_path"]
        support.parent.mkdir(parents=True, exist_ok=True)
        if not support.exists():
            support.write_bytes(canonical_json_bytes({"support": support.name}))
        binding = _binding(gate, sha256_file(support), failed=gate["id"] == "M17")
        (gpu_bindings if gate["id"] in FINALIZER.GPU_GATE_IDS else cpu_bindings)[
            gate["id"]
        ] = binding
    cpu = {
        "gate_bindings": cpu_bindings,
        "failed_gate_ids": [],
        "status": "PASS_CPU_GATES_PENDING_GPU",
    }
    gpu = {
        "gate_bindings": gpu_bindings,
        "failed_gate_ids": [],
        "status": "PASS_GPU_GATES",
    }
    with pytest.raises(FINALIZER.FinalizeFailure, match="failed-gate summary drift"):
        FINALIZER.validated_gate_results(CONFIG, run_root, cpu, gpu)


def test_cpu_gpu_sidecars_bind_run_source_preflight_and_support_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        FINALIZER,
        "verify_cpu_pytest_evidence",
        lambda *_args, **_kwargs: {"synthetic_fixture": True},
    )
    run_root = tmp_path / "run"
    artifact_dir = run_root / "artifacts" / "mk0"
    artifact_dir.mkdir(parents=True)
    run_id = "MK0_INTEGRITY_TEST"
    goal = "a" * 64
    commit = "b" * 40
    tracked_source_files = {
        relative: sha256_file(ROOT / relative)
        for relative in (
            "core/mk0/foundation_fusion.py",
            "core/mk0/samplers.py",
            "scripts/mk0/run_mk0_gpu_smoke.py",
        )
    }
    source_binding = {
        "test_source_binding": True,
        "tracked_source_files": tracked_source_files,
        "tracked_source_files_sha256": hashlib.sha256(
            canonical_json_bytes(tracked_source_files)
        ).hexdigest(),
    }
    preflight = {
        "path": str(tmp_path / "preflight.json"),
        "sha256": "c" * 64,
        "safety": {
            "unrelated_processes_killed": 0,
            "existing_results_overwritten": 0,
            "final_labels_read": False,
            "neural_forward_executed": False,
            "downstream_stage_started": False,
        },
    }
    run_manifest = {
        "schema_version": "mk0_run_manifest_v3",
        "run_id": run_id,
        "goal_sha256": goal,
        "implementation_commit": commit,
        "run_root": str(run_root.resolve()),
        "source_binding": source_binding,
        "preflight": preflight,
        "exact_commands": {"cpu_acceptance": {"argv": [sys.executable]}},
        "final_labels_accessed": False,
        "downstream_stage_started": False,
    }
    (run_root / "run_manifest.json").write_bytes(canonical_json_bytes(run_manifest))
    run_manifest_sha = sha256_file(run_root / "run_manifest.json")
    role_query_audit = _synthetic_gpu_role_query_audit(
        run_id=run_id,
        goal_sha256=goal,
        implementation_commit=commit,
        run_manifest_path=run_root / "run_manifest.json",
        run_manifest_sha256=run_manifest_sha,
        preflight=preflight,
        source_binding=source_binding,
    )

    inner = {
        gate_id: {
            "passed": True,
            "sample_count": 1,
            "failure_count": 0,
            "failure_denominator": 1,
        }
        for gate_id in FINALIZER.GPU_GATE_IDS
    }
    foundation = {
        "run_id": run_id,
        "status": "PASS",
        "cuda": {
            "cpu_fallback_observed": False,
            "cpu_fallback_allowed": False,
            "cuda_tensor_evidence": True,
            "max_memory_allocated_bytes": 1,
        },
        "gate_bindings": {gate_id: inner[gate_id] for gate_id in ("M31", "M32", "M35")},
        "post_gpu_role_query_audit": role_query_audit,
    }
    leakage = {
        "run_id": run_id,
        "status": "PASS",
        "gate_binding": inner["M05"],
    }
    (artifact_dir / "foundation_fusion_audit.json").write_bytes(
        canonical_json_bytes(foundation)
    )
    (artifact_dir / "target_alignment_leakage_audit.json").write_bytes(
        canonical_json_bytes(leakage)
    )
    for name in FINALIZER.CPU_SUPPORT:
        (artifact_dir / name).write_bytes(canonical_json_bytes({"name": name}))
    cpu_hashes = {
        name: sha256_file(artifact_dir / name) for name in FINALIZER.CPU_SUPPORT
    }
    gpu_hashes = {
        name: sha256_file(artifact_dir / name) for name in FINALIZER.GPU_SUPPORT
    }
    cpu = {
        "schema_version": "mk0_cpu_gate_results_v2",
        "run_id": run_id,
        "status": "PASS_CPU_GATES_PENDING_GPU",
        "goal_sha256": goal,
        "implementation_commit": commit,
        "source_binding": source_binding,
        "preflight": preflight,
        "run_manifest_path": "run_manifest.json",
        "run_manifest_sha256": run_manifest_sha,
        "artifact_hashes": cpu_hashes,
        "gate_bindings": {
            "M34": _binding(
                next(
                    gate
                    for gate in CONFIG["acceptance"]["gates"]
                    if gate["id"] == "M34"
                ),
                cpu_hashes["critic_role_audit.json"],
            )
        },
        "failed_gate_ids": [],
        "pending_gpu_gate_ids": ["M05", "M31", "M32", "M35"],
    }
    gpu = {
        "schema_version": "mk0_gpu_gate_results_v1",
        "run_id": run_id,
        "status": "PASS_GPU_GATES",
        "goal_sha256": goal,
        "implementation_commit": commit,
        "source_binding": source_binding,
        "preflight": preflight,
        "run_manifest_sha256": run_manifest_sha,
        "artifact_hashes": gpu_hashes,
        "post_gpu_role_query_audit": {
            "schema_version": "mk0_gpu_post_role_query_binding_v1",
            "qualifies_cpu_gate_id": "M34",
            "support_artifact": "foundation_fusion_audit.json",
            "support_artifact_sha256": gpu_hashes["foundation_fusion_audit.json"],
            "audit_sha256": hashlib.sha256(
                canonical_json_bytes(role_query_audit)
            ).hexdigest(),
            "record_stream_sha256": role_query_audit["record_stream_sha256"],
            "formal_gpu_phase_count": len(FINALIZER.GPU_ROLE_PHASE_SPECS),
            "all_role_query_counts_zero": True,
        },
        "gate_bindings": {
            gate_id: {
                "metrics": {
                    "support_gate_binding_sha256": hashlib.sha256(
                        canonical_json_bytes(inner[gate_id])
                    ).hexdigest()
                }
            }
            for gate_id in FINALIZER.GPU_GATE_IDS
        },
        "failed_gate_ids": [],
    }
    (artifact_dir / FINALIZER.CPU_RESULTS).write_bytes(canonical_json_bytes(cpu))
    (artifact_dir / FINALIZER.GPU_RESULTS).write_bytes(canonical_json_bytes(gpu))
    cpu_results_sha = sha256_file(artifact_dir / FINALIZER.CPU_RESULTS)
    gpu_results_sha = sha256_file(artifact_dir / FINALIZER.GPU_RESULTS)
    source_binding_sha = hashlib.sha256(
        canonical_json_bytes(source_binding)
    ).hexdigest()
    summary_dir = run_root / "summary"
    summary_dir.mkdir()
    cpu_summary = {
        "schema_version": "mk0_cpu_acceptance_summary_v2",
        "run_id": run_id,
        "status": cpu["status"],
        "evidence_level": "E0_MATH_ENGINEERING_ONLY",
        "goal_sha256": goal,
        "implementation_commit": commit,
        "run_root": str(run_root.resolve()),
        "run_manifest": {
            "path": str(run_root / "run_manifest.json"),
            "sha256": run_manifest_sha,
        },
        "preflight": preflight,
        "source_binding": source_binding,
        "source_binding_sha256": source_binding_sha,
        "cpu_gate_results": {
            "path": str(artifact_dir / FINALIZER.CPU_RESULTS),
            "sha256": cpu_results_sha,
        },
        "cpu_gate_results_sha256": cpu_results_sha,
        "artifact_count": len(cpu_hashes) + 1,
        "artifact_hashes": {**cpu_hashes, FINALIZER.CPU_RESULTS: cpu_results_sha},
        "failed_gate_ids": [],
        "pending_gpu_gate_ids": cpu["pending_gpu_gate_ids"],
    }
    gpu_summary = {
        "schema_version": "mk0_gpu_acceptance_summary_v1",
        "run_id": run_id,
        "status": gpu["status"],
        "evidence_level": "E0_MATH_ENGINEERING_ONLY",
        "goal_sha256": goal,
        "implementation_commit": commit,
        "run_root": str(run_root.resolve()),
        "run_manifest": {
            "path": str(run_root / "run_manifest.json"),
            "sha256": run_manifest_sha,
        },
        "preflight": preflight,
        "source_binding": source_binding,
        "source_binding_sha256": source_binding_sha,
        "gpu_gate_results": {
            "path": str(artifact_dir / FINALIZER.GPU_RESULTS),
            "sha256": gpu_results_sha,
        },
        "artifact_count": len(gpu_hashes) + 1,
        "artifact_hashes": {**gpu_hashes, FINALIZER.GPU_RESULTS: gpu_results_sha},
        "failed_gate_ids": [],
    }
    (summary_dir / FINALIZER.CPU_SUMMARY).write_bytes(canonical_json_bytes(cpu_summary))
    gpu_summary_path = summary_dir / FINALIZER.GPU_SUMMARY
    gpu_summary_path.write_bytes(canonical_json_bytes(gpu_summary))
    observed = FINALIZER.verify_runner_results(
        artifact_dir,
        run_root.resolve(),
        run_id=run_id,
        goal_sha256=goal,
        implementation_commit=commit,
        preflight=preflight,
        source_binding=source_binding,
    )
    assert observed[0]["run_id"] == observed[1]["run_id"] == run_id
    assert observed[4]["gpu_post_role_query_audit"]["qualifies_cpu_gate_id"] == "M34"

    role_verify_kwargs = {
        "run_id": run_id,
        "goal_sha256": goal,
        "implementation_commit": commit,
        "run_manifest_path": run_root / "run_manifest.json",
        "run_manifest_sha256": run_manifest_sha,
        "preflight": preflight,
        "source_binding": source_binding,
        "foundation_sha256": gpu_hashes["foundation_fusion_audit.json"],
    }
    missing_audit_foundation = copy.deepcopy(foundation)
    missing_audit_foundation.pop("post_gpu_role_query_audit")
    with pytest.raises(FINALIZER.FinalizeFailure, match="audit is missing"):
        FINALIZER.verify_gpu_post_role_query_audit(
            missing_audit_foundation, gpu, **role_verify_kwargs
        )

    missing_phase_foundation = copy.deepcopy(foundation)
    missing_phase_foundation["post_gpu_role_query_audit"]["phase_records"].pop()
    with pytest.raises(FINALIZER.FinalizeFailure, match="phase coverage is incomplete"):
        FINALIZER.verify_gpu_post_role_query_audit(
            missing_phase_foundation, gpu, **role_verify_kwargs
        )

    fake_qualname_foundation = copy.deepcopy(foundation)
    fake_qualname_record = fake_qualname_foundation["post_gpu_role_query_audit"][
        "phase_records"
    ][0]["call_inventory"][0]
    fake_qualname_record["function_qualname"] = "FakeWrapper.forward"
    with pytest.raises(FINALIZER.FinalizeFailure, match="category substitution"):
        FINALIZER.verify_gpu_post_role_query_audit(
            fake_qualname_foundation, gpu, **role_verify_kwargs
        )

    fake_line_foundation = copy.deepcopy(foundation)
    fake_line_record = fake_line_foundation["post_gpu_role_query_audit"][
        "phase_records"
    ][0]["call_inventory"][0]
    fake_line_record["first_lineno"] = 1
    with pytest.raises(FINALIZER.FinalizeFailure, match="category substitution"):
        FINALIZER.verify_gpu_post_role_query_audit(
            fake_line_foundation, gpu, **role_verify_kwargs
        )

    unknown_external_foundation = copy.deepcopy(foundation)
    unknown_external_phase = unknown_external_foundation["post_gpu_role_query_audit"][
        "phase_records"
    ][0]
    unknown_external_phase["external_call_inventory"] = [
        {
            "module_name": "thirdparty.benign_sdk",
            "source_file": str(tmp_path / "benign_sdk.py"),
            "function_qualname": "calculate",
            "first_lineno": 7,
            "classification": "unknown_external",
            "categories": [],
            "call_count": 1,
        }
    ]
    unknown_external_phase["external_record_stream_sha256"] = hashlib.sha256(
        canonical_json_bytes(unknown_external_phase["external_call_inventory"])
    ).hexdigest()
    unknown_external_phase["external_python_call_count"] = 1
    unknown_external_phase["unknown_external_call_count"] = 1
    unknown_external_phase["total_python_call_count"] += 1
    with pytest.raises(FINALIZER.FinalizeFailure, match="unknown external call"):
        FINALIZER.verify_gpu_post_role_query_audit(
            unknown_external_foundation, gpu, **role_verify_kwargs
        )

    thread_drift_foundation = copy.deepcopy(foundation)
    thread_drift_phase = thread_drift_foundation["post_gpu_role_query_audit"][
        "phase_records"
    ][0]
    thread_drift_phase["thread_inventory"][0]["repository_call_count"] -= 1
    thread_drift_phase["thread_inventory"][0]["total_python_call_count"] -= 1
    thread_drift_phase["thread_record_stream_sha256"] = hashlib.sha256(
        canonical_json_bytes(thread_drift_phase["thread_inventory"])
    ).hexdigest()
    with pytest.raises(FINALIZER.FinalizeFailure, match="thread/call inventory"):
        FINALIZER.verify_gpu_post_role_query_audit(
            thread_drift_foundation, gpu, **role_verify_kwargs
        )

    nonzero_query_foundation = copy.deepcopy(foundation)
    nonzero_audit = nonzero_query_foundation["post_gpu_role_query_audit"]
    first_phase = nonzero_audit["phase_records"][0]
    first_phase["call_inventory"].append(
        {
            "source_file": "scripts/mk0/run_mk0_gpu_smoke.py",
            "function_qualname": "formal_critic_query",
            "first_lineno": 999999,
            "categories": ["critic_query"],
            "call_count": 1,
        }
    )
    first_phase["call_inventory"].sort(
        key=lambda record: (
            record["source_file"],
            record["function_qualname"],
            record["first_lineno"],
            tuple(record["categories"]),
        )
    )
    first_phase["repository_python_call_count"] += 1
    first_phase["critic_query_call_count"] = 1
    first_phase["record_stream_sha256"] = hashlib.sha256(
        canonical_json_bytes(first_phase["call_inventory"])
    ).hexdigest()
    nonzero_audit["repository_python_call_count"] += 1
    nonzero_audit["critic_query_call_count"] = 1
    nonzero_audit["all_role_query_counts_zero"] = False
    nonzero_audit["record_stream_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            [
                {
                    "phase_id": phase["phase_id"],
                    "call_inventory": phase["call_inventory"],
                }
                for phase in nonzero_audit["phase_records"]
            ]
        )
    ).hexdigest()
    with pytest.raises(FINALIZER.FinalizeFailure, match="prohibited critic_query"):
        FINALIZER.verify_gpu_post_role_query_audit(
            nonzero_query_foundation, gpu, **role_verify_kwargs
        )

    audit_hash_gpu = copy.deepcopy(gpu)
    audit_hash_gpu["post_gpu_role_query_audit"]["audit_sha256"] = "0" * 64
    with pytest.raises(FINALIZER.FinalizeFailure, match="audit digest drift"):
        FINALIZER.verify_gpu_post_role_query_audit(
            foundation, audit_hash_gpu, **role_verify_kwargs
        )

    support_hash_gpu = copy.deepcopy(gpu)
    support_hash_gpu["post_gpu_role_query_audit"]["support_artifact_sha256"] = "0" * 64
    with pytest.raises(FINALIZER.FinalizeFailure, match="support hash drift"):
        FINALIZER.verify_gpu_post_role_query_audit(
            foundation, support_hash_gpu, **role_verify_kwargs
        )

    gpu_summary_path.write_bytes(
        canonical_json_bytes({**gpu_summary, "implementation_commit": "d" * 40})
    )
    with pytest.raises(FINALIZER.FinalizeFailure, match="GPU summary commit drift"):
        FINALIZER.verify_runner_results(
            artifact_dir,
            run_root.resolve(),
            run_id=run_id,
            goal_sha256=goal,
            implementation_commit=commit,
            preflight=preflight,
            source_binding=source_binding,
        )
    gpu_summary_path.write_bytes(canonical_json_bytes(gpu_summary))

    (artifact_dir / "foundation_fusion_audit.json").write_bytes(b"tampered\n")
    with pytest.raises(FINALIZER.FinalizeFailure, match="support hash drift"):
        FINALIZER.verify_runner_results(
            artifact_dir,
            run_root.resolve(),
            run_id=run_id,
            goal_sha256=goal,
            implementation_commit=commit,
            preflight=preflight,
            source_binding=source_binding,
        )


@pytest.mark.parametrize(
    ("error", "expected_reason"),
    [(KeyboardInterrupt(), "KeyboardInterrupt"), (RuntimeError(), "RuntimeError")],
)
def test_finalizer_main_writes_nonempty_standard_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: BaseException,
    expected_reason: str,
) -> None:
    run_root = tmp_path / "finalizer_failure"
    (run_root / "artifacts" / "mk0").mkdir(parents=True)
    (run_root / "logs").mkdir()
    (run_root / "logs" / "events.jsonl").write_text("", encoding="utf-8")
    (run_root / "logs" / "stderr.log").write_text("", encoding="utf-8")
    args = Namespace(
        run_root=run_root,
        run_id="MK0_FAILURE_REASON_TEST",
        parent_run_id=None,
        goal_sha256="a" * 64,
        implementation_commit="b" * 40,
        fm0_closure_root=tmp_path,
        d1_data=tmp_path,
        d1_ledger=tmp_path,
        preflight_record=tmp_path,
    )
    observed: dict[str, Any] = {}

    monkeypatch.setattr(FINALIZER, "parse_args", lambda _argv: args)

    def raise_failure(_condition: Any, _message: str) -> None:
        raise error

    monkeypatch.setattr(FINALIZER, "require", raise_failure)
    monkeypatch.setattr(
        FINALIZER,
        "write_failed_sentinel",
        lambda _root, **kwargs: observed.update(kwargs),
    )

    assert FINALIZER.main([]) == 1
    assert observed["reason"] == expected_reason
    persisted = json.loads(
        (run_root / "failure" / "finalize_failure.json").read_text(encoding="utf-8")
    )
    assert persisted["exception_message"] == expected_reason


def test_absent_fm0_terminal_prerequisite_fails_closed(tmp_path: Path) -> None:
    fm0 = tmp_path / "fm0"
    fm0.mkdir()
    marker = fm0 / "only_one_file"
    marker.write_text("incomplete\n", encoding="utf-8")
    digest = sha256_file(marker)
    (fm0 / "artifact_checksums.sha256").write_text(
        f"{digest}  ./only_one_file\n", encoding="utf-8"
    )
    with pytest.raises(FINALIZER.FinalizeFailure, match="omits a required"):
        FINALIZER.verify_fm0_b0_d1(
            fm0,
            goal_sha256="a" * 64,
            implementation_commit="b" * 40,
            d1_data=tmp_path / "missing-data",
            d1_ledger=tmp_path / "missing-ledger",
        )


def test_preflight_run_substitution_fails_before_acceptance(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "mk0_preflight_v1",
                "run_id": "WRONG_RUN",
                "parent_run_id": None,
                "goal_sha256": "a" * 64,
                "mode": "read_only_metadata_and_hashes",
            }
        )
    )
    with pytest.raises(FINALIZER.FinalizeFailure, match="run ID drift"):
        FINALIZER.verify_preflight(
            preflight,
            run_id="EXPECTED_RUN",
            goal_sha256="a" * 64,
            implementation_commit="b" * 40,
            fm0={},
            parent_run_id=None,
        )


def _preflight_inventory(root: Path) -> dict[str, Any]:
    entries = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        stat = child.lstat()
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "kind": "directory" if child.is_dir() else "file",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "root": str(root),
        "entry_count": len(entries),
        "entries": entries,
        "inventory_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
        "recursive": False,
        "metadata_only": True,
    }


def _complete_preflight_observations(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    worktree = tmp_path / "worktree"
    main_repo = tmp_path / "main"
    mnt_root = tmp_path / "mnt"
    fm0_root = mnt_root / "mrna_editflow_fm0_runs" / "closure"
    for path in (
        worktree,
        main_repo / "data",
        main_repo / "data_registry",
        main_repo / "artifacts",
        fm0_root,
    ):
        path.mkdir(parents=True)
        (path / "observed").write_text("metadata\n", encoding="utf-8")
    (mnt_root / "mrna_editflow_mk0_runs").mkdir()
    worktree_list = (
        f"worktree {main_repo}\nHEAD {'a' * 40}\nbranch refs/heads/protected-main\n\n"
        f"worktree {worktree}\nHEAD {'f' * 40}\nbranch refs/heads/mk0-test\n"
    )
    gpu_processes = [
        {
            "pid": 101,
            "process_name": "python",
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "used_memory_mib": 1024,
            "owner": "test-user",
            "owner_uid": 1000,
            "owner_resolution": "RESOLVED_FROM_PROC_STATUS",
        }
    ]
    user_processes = [
        {"pid": 101, "stat": "Sl", "elapsed": "01:02", "command": "python"},
        {"pid": 202, "stat": "Ss", "elapsed": "02:03", "command": "sshd"},
    ]
    collector = ROOT / "scripts" / "mk0" / "record_mk0_preflight.py"
    report = {
        "collector": {
            "path": str(collector),
            "sha256": sha256_file(collector),
            "pid": 303,
        },
        "worktree": {
            "path": str(worktree),
            "branch": "mk0-test",
            "head": "f" * 40,
            "status_porcelain": "",
        },
        "protected_main_repo": {
            "path": str(main_repo),
            "branch": "protected-main",
            "head": "a" * 40,
            "status_porcelain": " M user-file",
            "worktree_list_sha256": hashlib.sha256(
                worktree_list.encode("utf-8")
            ).hexdigest(),
            "worktree_list_porcelain": worktree_list,
        },
        "resources": {
            "home_filesystem": {
                "path": str(worktree),
                "total_bytes": 100,
                "used_bytes": 40,
                "free_bytes": 60,
                "reserved_bytes": 0,
            },
            "mnt_filesystem": {
                "path": str(mnt_root),
                "total_bytes": 200,
                "used_bytes": 50,
                "free_bytes": 150,
                "reserved_bytes": 0,
            },
            "cpu_memory": {
                "total_bytes": 1024,
                "available_bytes": 768,
                "free_bytes": 512,
                "swap_total_bytes": 256,
                "swap_free_bytes": 128,
            },
            "framework": {
                "python_version": "3.10.20",
                "python_implementation": "CPython",
                "python_executable": str(Path(sys.executable).resolve()),
                "torch_version": "2.6.0+cu124",
                "torch_cuda_build_version": "12.4",
                "torch_cudnn_version": 90100,
                "torch_cuda_is_available": True,
                "torch_cuda_device_count": 1,
            },
            "driver_supported_cuda_version": "12.4",
            "nvidia_smi_exit_code": 0,
            "nvidia_smi_stdout_sha256": "7" * 64,
            "nvidia_smi_stderr_sha256": "8" * 64,
            "nvidia_smi_l_exit_code": 0,
            "nvidia_smi_l_stdout_sha256": hashlib.sha256(
                "GPU 0: Test GPU (UUID: GPU-00000000-0000-0000-0000-000000000000)".encode(
                    "utf-8"
                )
            ).hexdigest(),
            "nvidia_smi_l_stderr_sha256": "9" * 64,
            "mig_instance_count": 0,
            "mig_instances": [],
            "mig_instances_sha256": hashlib.sha256(
                canonical_json_bytes([])
            ).hexdigest(),
            "gpus": [
                {
                    "index": 0,
                    "name": "Test GPU",
                    "uuid": "GPU-00000000-0000-0000-0000-000000000000",
                    "driver_version": "550.54.15",
                    "memory_total_mib": 40960,
                    "memory_used_mib": 1024,
                    "memory_free_mib": 39936,
                    "utilization_gpu_percent": 25,
                }
            ],
            "gpu_query_exit_code": 0,
            "gpu_query_stderr_sha256": "c" * 64,
            "gpu_compute_process_query_exit_code": 0,
            "gpu_compute_process_query_stderr_sha256": "1" * 64,
            "gpu_compute_process_count": 1,
            "gpu_compute_processes": gpu_processes,
            "gpu_compute_process_metadata_sha256": hashlib.sha256(
                canonical_json_bytes(gpu_processes)
            ).hexdigest(),
            "gpu_process_policy": "no process killed; any card with sufficient free memory may be used per user authorization",
            "current_user_process_count": 2,
            "current_user_processes": user_processes,
            "current_user_process_query_exit_code": 0,
            "current_user_process_query_stderr_sha256": "d" * 64,
            "current_user_process_metadata_sha256": hashlib.sha256(
                canonical_json_bytes(user_processes)
            ).hexdigest(),
        },
        "inventory": {
            "project_data": {
                "data": _preflight_inventory(main_repo / "data"),
                "data_registry": _preflight_inventory(main_repo / "data_registry"),
            },
            "existing_artifacts": {
                "main_repo_artifacts": _preflight_inventory(main_repo / "artifacts"),
                "fm0_closure": _preflight_inventory(fm0_root),
                "mnt_mrna_editflow_roots": _preflight_inventory(mnt_root),
            },
        },
    }
    return report, {
        "worktree": worktree,
        "main_repo": main_repo,
        "mnt_root": mnt_root,
        "fm0_root": fm0_root,
    }


def test_preflight_requires_project_tasks_gpu_disk_data_and_artifact_evidence(
    tmp_path: Path,
) -> None:
    report, paths = _complete_preflight_observations(tmp_path)
    kwargs = {
        "expected_worktree": paths["worktree"],
        "expected_main_repo": paths["main_repo"],
        "expected_mnt_root": paths["mnt_root"],
        "expected_fm0_root": paths["fm0_root"],
    }
    FINALIZER._validate_preflight_observation_fields(report, **kwargs)

    missing_main = copy.deepcopy(report)
    del missing_main["protected_main_repo"]
    with pytest.raises(FINALIZER.FinalizeFailure, match="protected main repo"):
        FINALIZER._validate_preflight_observation_fields(missing_main, **kwargs)

    missing_gpu = copy.deepcopy(report)
    missing_gpu["resources"]["gpus"] = []
    with pytest.raises(FINALIZER.FinalizeFailure, match="GPU inventory"):
        FINALIZER._validate_preflight_observation_fields(missing_gpu, **kwargs)

    failed_gpu_process_probe = copy.deepcopy(report)
    failed_gpu_process_probe["resources"]["gpu_compute_process_query_exit_code"] = 1
    with pytest.raises(FINALIZER.FinalizeFailure, match="GPU process query"):
        FINALIZER._validate_preflight_observation_fields(
            failed_gpu_process_probe, **kwargs
        )

    failed_process_probe = copy.deepcopy(report)
    failed_process_probe["resources"]["current_user_process_query_exit_code"] = 1
    with pytest.raises(FINALIZER.FinalizeFailure, match="process query"):
        FINALIZER._validate_preflight_observation_fields(failed_process_probe, **kwargs)

    missing_data = copy.deepcopy(report)
    del missing_data["inventory"]["project_data"]
    with pytest.raises(FINALIZER.FinalizeFailure, match="project data inventory"):
        FINALIZER._validate_preflight_observation_fields(missing_data, **kwargs)

    tampered_artifacts = copy.deepcopy(report)
    tampered_artifacts["inventory"]["existing_artifacts"]["fm0_closure"][
        "entry_count"
    ] = 0
    with pytest.raises(FINALIZER.FinalizeFailure, match="entry count drift"):
        FINALIZER._validate_preflight_observation_fields(tampered_artifacts, **kwargs)

    tampered_collector = copy.deepcopy(report)
    tampered_collector["collector"]["sha256"] = "0" * 64
    with pytest.raises(FINALIZER.FinalizeFailure, match="collector source hash drift"):
        FINALIZER._validate_preflight_observation_fields(tampered_collector, **kwargs)

    tampered_worktree_inventory = copy.deepcopy(report)
    tampered_worktree_inventory["protected_main_repo"][
        "worktree_list_porcelain"
    ] += "\nforged\n"
    with pytest.raises(FINALIZER.FinalizeFailure, match="inventory digest drift"):
        FINALIZER._validate_preflight_observation_fields(
            tampered_worktree_inventory, **kwargs
        )

    invalid_gpu_uuid = copy.deepcopy(report)
    invalid_gpu_uuid["resources"]["gpus"][0]["uuid"] = "not-a-gpu-uuid"
    with pytest.raises(FINALIZER.FinalizeFailure, match="GPU UUID"):
        FINALIZER._validate_preflight_observation_fields(invalid_gpu_uuid, **kwargs)

    tampered_gpu_process_count = copy.deepcopy(report)
    tampered_gpu_process_count["resources"]["gpu_compute_process_count"] = 2
    with pytest.raises(FINALIZER.FinalizeFailure, match="records/count drift"):
        FINALIZER._validate_preflight_observation_fields(
            tampered_gpu_process_count, **kwargs
        )

    tampered_gpu_process_digest = copy.deepcopy(report)
    tampered_gpu_process_digest["resources"]["gpu_compute_process_metadata_sha256"] = (
        "0" * 64
    )
    with pytest.raises(FINALIZER.FinalizeFailure, match="record digest drift"):
        FINALIZER._validate_preflight_observation_fields(
            tampered_gpu_process_digest, **kwargs
        )

    tampered_user_process_digest = copy.deepcopy(report)
    tampered_user_process_digest["resources"][
        "current_user_process_metadata_sha256"
    ] = ("0" * 64)
    with pytest.raises(FINALIZER.FinalizeFailure, match="record digest drift"):
        FINALIZER._validate_preflight_observation_fields(
            tampered_user_process_digest, **kwargs
        )

    missing_framework = copy.deepcopy(report)
    del missing_framework["resources"]["framework"]
    with pytest.raises(FINALIZER.FinalizeFailure, match="framework inventory"):
        FINALIZER._validate_preflight_observation_fields(missing_framework, **kwargs)

    cuda_unavailable = copy.deepcopy(report)
    cuda_unavailable["resources"]["framework"]["torch_cuda_is_available"] = False
    with pytest.raises(FINALIZER.FinalizeFailure, match="CUDA availability"):
        FINALIZER._validate_preflight_observation_fields(cuda_unavailable, **kwargs)

    missing_cpu_ram = copy.deepcopy(report)
    del missing_cpu_ram["resources"]["cpu_memory"]
    with pytest.raises(FINALIZER.FinalizeFailure, match="CPU RAM inventory"):
        FINALIZER._validate_preflight_observation_fields(missing_cpu_ram, **kwargs)

    invalid_driver = copy.deepcopy(report)
    invalid_driver["resources"]["gpus"][0]["driver_version"] = "unknown"
    with pytest.raises(FINALIZER.FinalizeFailure, match="driver version"):
        FINALIZER._validate_preflight_observation_fields(invalid_driver, **kwargs)

    missing_owner = copy.deepcopy(report)
    missing_owner["resources"]["gpu_compute_processes"][0]["owner"] = None
    missing_owner["resources"]["gpu_compute_process_metadata_sha256"] = hashlib.sha256(
        canonical_json_bytes(missing_owner["resources"]["gpu_compute_processes"])
    ).hexdigest()
    with pytest.raises(FINALIZER.FinalizeFailure, match="owner is invalid"):
        FINALIZER._validate_preflight_observation_fields(missing_owner, **kwargs)


def test_preflight_gpu_execution_identity_is_cross_bound(tmp_path: Path) -> None:
    report, _ = _complete_preflight_observations(tmp_path)
    preflight = tmp_path / "preflight.json"
    preflight.write_bytes(canonical_json_bytes(report))
    foundation = {
        "cuda": {
            "device_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "device_name": "Test GPU",
            "logical_device_index": 0,
            "cuda_visible_devices": "0",
            "python_version": "3.10.20",
            "torch_version": "2.6.0+cu124",
            "torch_cuda_version": "12.4",
        }
    }
    binding = FINALIZER.verify_preflight_gpu_execution_identity(
        preflight,
        foundation,
        expected_preflight_sha256=sha256_file(preflight),
    )
    assert binding["formal_cuda_device_was_present_at_preflight"] is True

    wrong_uuid = copy.deepcopy(foundation)
    wrong_uuid["cuda"]["device_uuid"] = "GPU-11111111-1111-1111-1111-111111111111"
    with pytest.raises(FINALIZER.FinalizeFailure, match="absent from the preflight"):
        FINALIZER.verify_preflight_gpu_execution_identity(
            preflight,
            wrong_uuid,
            expected_preflight_sha256=sha256_file(preflight),
        )

    wrong_name = copy.deepcopy(foundation)
    wrong_name["cuda"]["device_name"] = "Substituted GPU"
    with pytest.raises(FINALIZER.FinalizeFailure, match="name differs"):
        FINALIZER.verify_preflight_gpu_execution_identity(
            preflight,
            wrong_name,
            expected_preflight_sha256=sha256_file(preflight),
        )

    wrong_framework = copy.deepcopy(foundation)
    wrong_framework["cuda"]["torch_version"] = "2.5.0"
    with pytest.raises(FINALIZER.FinalizeFailure, match="torch version differs"):
        FINALIZER.verify_preflight_gpu_execution_identity(
            preflight,
            wrong_framework,
            expected_preflight_sha256=sha256_file(preflight),
        )


def test_live_preflight_authenticity_rechecks_stable_resource_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, paths = _complete_preflight_observations(tmp_path)

    class Usage:
        def __init__(self, total: int, free: int) -> None:
            self.total = total
            self.free = free

    def fake_disk_usage(path: Path) -> Usage:
        if Path(path) == paths["worktree"]:
            return Usage(100, 60)
        return Usage(200, 150)

    def fake_probe(command: list[str], *, cwd: Path | None = None) -> str:
        if command[:3] == ["git", "branch", "--show-current"]:
            return "mk0-test" if cwd == paths["worktree"] else "protected-main"
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "f" * 40 if cwd == paths["worktree"] else "a" * 40
        if command[:3] == ["git", "status", "--porcelain=v1"]:
            return "" if cwd == paths["worktree"] else " M user-file"
        if command[:3] == ["git", "worktree", "list"]:
            return report["protected_main_repo"]["worktree_list_porcelain"]
        if (
            command[0] == "nvidia-smi"
            and "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu"
            in command
        ):
            return "0, Test GPU, GPU-00000000-0000-0000-0000-000000000000, 550.54.15, 40960, 1024, 39936, 25"
        if command == ["nvidia-smi"]:
            return "NVIDIA-SMI 550.54.15 Driver Version: 550.54.15 CUDA Version: 12.4"
        if command == ["nvidia-smi", "-L"]:
            return "GPU 0: Test GPU (UUID: GPU-00000000-0000-0000-0000-000000000000)"
        raise AssertionError(command)

    monkeypatch.setattr(FINALIZER.shutil, "disk_usage", fake_disk_usage)
    monkeypatch.setattr(FINALIZER, "_run_read_only_text", fake_probe)
    monkeypatch.setattr(
        FINALIZER,
        "_read_linux_cpu_memory_total",
        lambda: report["resources"]["cpu_memory"]["total_bytes"],
    )
    FINALIZER._validate_live_preflight_authenticity(
        report,
        expected_worktree=paths["worktree"],
        expected_main_repo=paths["main_repo"],
        expected_mnt_root=paths["mnt_root"],
    )

    report["resources"]["gpus"][0]["driver_version"] = "551.00"
    with pytest.raises(FINALIZER.FinalizeFailure, match="GPU identity"):
        FINALIZER._validate_live_preflight_authenticity(
            report,
            expected_worktree=paths["worktree"],
            expected_main_repo=paths["main_repo"],
            expected_mnt_root=paths["mnt_root"],
        )


def test_runner_and_finalizer_sources_do_not_mechanically_declare_gate_passes() -> None:
    cpu_source = (ROOT / "scripts" / "mk0" / "run_mk0_cpu_acceptance.py").read_text(
        encoding="utf-8"
    )
    finalizer_source = inspect.getsource(FINALIZER.validated_gate_results)
    assert "passed_cpu_gates =" not in cpu_source
    assert '"passed": failure_count == 0' in cpu_source
    assert "gate_result_from_runtime_binding" in finalizer_source
    assert "passed=True" not in finalizer_source
