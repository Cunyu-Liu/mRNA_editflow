#!/usr/bin/env python3
"""Fail-closed MK0 GPU smoke for the real frozen UTR-LM fusion path.

This runner is intentionally small and synthetic.  It verifies GPU execution,
foundation freezing, dynamic-current full re-encoding, target/alignment
non-interference, and one differentiable positive-rate case for each of
INS/SUB/DEL/STOP.  It does not train an Edit Flow model and it does not access
development or final labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
import sysconfig
import threading
from types import ModuleType
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

# Required by PyTorch deterministic CUDA matrix multiplication.  This must be
# set before the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[2]
_bootstrap_path = REPO_ROOT / "scripts" / "mk0" / "strict_worktree_import.py"
_bootstrap_module = ModuleType("_mk0_strict_worktree_import")
_bootstrap_module.__file__ = str(_bootstrap_path)
_bootstrap_module.__cached__ = None
exec(
    compile(
        _bootstrap_path.read_bytes(),
        str(_bootstrap_path),
        "exec",
        dont_inherit=True,
        optimize=0,
    ),
    _bootstrap_module.__dict__,
)

import torch
import yaml

with _bootstrap_module.strict_worktree_package_import(REPO_ROOT):
    from mrna_editflow.core.mk0.acceptance import (
        canonical_json_bytes,
        gate_result_from_runtime_binding,
        sha256_file,
    )
    from mrna_editflow.core.mk0.foundation_fusion import (
        FoundationFusionRateField,
        OfficialPaperRateAdapter,
        load_official_utrlm,
        require_neural_cuda,
    )
    from mrna_editflow.core.mk0.run_contract import (
        append_event,
        append_jsonl,
        append_text,
        resume_failure_closure_if_present,
        update_status,
        validate_terminal_chain,
        write_failed_sentinel,
    )
    from mrna_editflow.core.mk0.state_action import (
        apply_action,
        enumerate_legal_actions,
        is_legal,
    )
    from mrna_editflow.core.mk0.samplers import (
        constrained_single_event_first_order,
        paper_first_order_parallel,
        replay_constrained_result,
        replay_paper_result,
    )
    from mrna_editflow.core.mk0.training_boundary import (
        EditFlowTrainingExample,
        canonical_rate_input_bytes,
        rate_input_state,
    )
    from mrna_editflow.core.mk0.types import (
        ALPHABET,
        ActionType,
        AtomicAction,
        EditState,
        TokenOrigin,
    )


SEED = 20260802
FLOAT32_ATOL = 1.0e-5
FLOAT32_RTOL = 1.0e-4
FOUNDATION_FILENAME = "foundation_fusion_audit.json"
LEAKAGE_FILENAME = "target_alignment_leakage_audit.json"
FAILURE_FILENAME = "mk0_gpu_smoke_failure.json"
GPU_RESULTS_FILENAME = "mk0_gpu_gate_results.json"
GPU_SUMMARY_FILENAME = "gpu_acceptance_summary.json"
FM0_CONFIG_PATH = REPO_ROOT / "configs" / "fm0_utrlm_config.yaml"
MATH_CONFIG_PATH = REPO_ROOT / "configs" / "math" / "math_kernel_v1.yaml"
FM0_HASH_LICENSE_RELATIVE = Path("evaluation/hash_license_manifest.json")

FORMAL_GPU_ROLE_PHASE_SPECS = (
    {
        "phase_id": "generator_rate_official_frozen_arm",
        "phase_kind": "generator_rate",
        "entrypoint": "_run_forced_action_arm",
        "required_call_categories": ("generator_interface", "rate_interface"),
    },
    {
        "phase_id": "generator_rate_from_scratch_control_arm",
        "phase_kind": "generator_rate",
        "entrypoint": "_run_forced_action_arm",
        "required_call_categories": ("generator_interface", "rate_interface"),
    },
    {
        "phase_id": "sampler_paper_official_foundation",
        "phase_kind": "sampler_rate",
        "entrypoint": "_run_official_paper_sampler_route",
        "required_call_categories": ("sampler_interface", "rate_interface"),
    },
    {
        "phase_id": "sampler_primary_official_foundation",
        "phase_kind": "sampler_rate",
        "entrypoint": "_run_primary_gpu_sampler_integration",
        "required_call_categories": ("sampler_interface", "rate_interface"),
    },
    {
        "phase_id": "rate_target_alignment_leakage_audit",
        "phase_kind": "rate_audit",
        "entrypoint": "_audit_target_alignment_leakage",
        "required_call_categories": ("rate_interface",),
    },
    {
        "phase_id": "rate_dynamic_current_encoding_audit",
        "phase_kind": "rate_audit",
        "entrypoint": "_audit_dynamic_current_encoding",
        "required_call_categories": ("rate_interface",),
    },
)

ROLE_QUERY_CATEGORIES = (
    "critic_query",
    "guidance_query",
    "final_evaluator_query",
)

FROZEN_FOUNDATION_EXTERNAL_MODULE_PREFIXES = (
    "torch",
    "transformers",
    "multimolecule",
    "tokenizers",
    "numpy",
    "safetensors",
    "huggingface_hub",
    "packaging",
)

ROLE_PROHIBITED_TOKENS = (
    "critic",
    "guidance",
    "evaluator",
    "final_evaluator",
    "reward",
    "rerank",
    "selector",
)

STDLIB_ROOT = Path(sysconfig.get_path("stdlib")).resolve()
SITE_PACKAGE_ROOTS = tuple(
    sorted(
        {
            Path(value).resolve()
            for key in ("purelib", "platlib")
            if (value := sysconfig.get_path(key))
        },
        key=str,
    )
)

FORMAL_GPU_ROLE_INTERFACE_LABELS = (
    "gpu_runner.forced_action_arm",
    "gpu_runner.paper_sampler_route",
    "gpu_runner.primary_sampler_integration",
    "gpu_runner.target_alignment_leakage_audit",
    "gpu_runner.dynamic_current_encoding_audit",
    "foundation_fusion.rate_field_forward",
    "foundation_fusion.official_paper_adapter",
    "samplers.constrained_primary",
    "samplers.paper_parallel",
    "samplers.replay_constrained",
    "samplers.replay_paper",
)


class SmokeFailure(RuntimeError):
    """A fail-closed MK0 GPU-smoke invariant violation."""


def _module_matches_prefix(module_name: str, prefix: str) -> bool:
    return module_name == prefix or module_name.startswith(f"{prefix}.")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _external_prohibited_categories(
    module_name: str,
    source_file: str,
    qualname: str,
) -> tuple[str, ...]:
    lower = f"{module_name} {source_file} {qualname}".lower()
    categories: set[str] = set()
    if "critic" in lower:
        categories.add("critic_query")
    if "guidance" in lower:
        categories.add("guidance_query")
    if "evaluator" in lower:
        categories.add("final_evaluator_query")
    if any(token in lower for token in ("reward", "rerank", "selector")):
        categories.add("final_evaluator_query")
    return tuple(sorted(categories))


def _external_call_classification(
    module_name: str,
    source_file: str,
    qualname: str,
) -> tuple[str, tuple[str, ...]]:
    categories = _external_prohibited_categories(
        module_name,
        source_file,
        qualname,
    )
    if categories:
        return "prohibited_role", categories
    if any(
        _module_matches_prefix(module_name, prefix)
        for prefix in FROZEN_FOUNDATION_EXTERNAL_MODULE_PREFIXES
    ):
        return "frozen_foundation_stack_allowlist", ()
    root_module = module_name.partition(".")[0]
    if root_module in sys.stdlib_module_names:
        if source_file.startswith(("<built-in", "<frozen")):
            return "stdlib_allowlist", ()
        try:
            source_path = Path(source_file).resolve(strict=True)
        except OSError:
            return "unknown_external", ()
        in_site_packages = any(
            _path_is_within(source_path, root) for root in SITE_PACKAGE_ROOTS
        )
        if _path_is_within(source_path, STDLIB_ROOT) and not in_site_packages:
            return "stdlib_allowlist", ()
    return "unknown_external", ()


class _FormalGpuRoleQueryRecorder:
    """Profile the actual formal GPU Python call chain phase by phase.

    This is deliberately runtime instrumentation rather than a post-hoc set of
    zero constants.  Only calls whose code lives in the source-bound repository
    enter the inventory.  The resulting aggregate inventory remains compact,
    while its counts and canonical digest are independently checkable by the
    finalizer.
    """

    def __init__(
        self,
        formal_binding: Mapping[str, Any],
        *,
        run_id: str,
        interface_records: Sequence[Mapping[str, Any]],
        role_code_metadata: Mapping[Any, tuple[str, str, int, tuple[str, ...]]],
    ) -> None:
        self._formal_binding = formal_binding
        self._run_id = run_id
        self._interface_records = [dict(record) for record in interface_records]
        self._role_code_metadata = dict(role_code_metadata)
        self._phase_records: list[dict[str, Any]] = []
        self._active_inventory: (
            dict[tuple[str, str, int, tuple[str, ...]], int] | None
        ) = None
        self._active_external_inventory: (
            dict[tuple[str, str, str, int, str, tuple[str, ...]], int] | None
        ) = None
        self._active_thread_inventory: dict[tuple[int, str], dict[str, int]] | None = (
            None
        )
        self._inventory_lock = threading.Lock()
        self._code_metadata_cache: dict[
            Any, tuple[str, str, int, tuple[str, ...]] | None
        ] = {}
        self._external_code_metadata_cache: dict[
            tuple[Any, str], tuple[str, str, str, int, str, tuple[str, ...]]
        ] = {}

    @staticmethod
    def _prohibited_categories(relative: str, qualname: str) -> tuple[str, ...]:
        categories: set[str] = set()
        lower = f"{relative} {qualname}".lower()
        if "critic" in lower:
            categories.add("critic_query")
        if "guidance" in lower:
            categories.add("guidance_query")
        if "final_evaluator" in lower or ("final" in lower and "evaluator" in lower):
            categories.add("final_evaluator_query")
        return tuple(sorted(categories))

    def _metadata_for_code(
        self, code: Any
    ) -> tuple[str, str, int, tuple[str, ...]] | None:
        if code in self._code_metadata_cache:
            return self._code_metadata_cache[code]
        exact_metadata = self._role_code_metadata.get(code)
        if exact_metadata is not None:
            self._code_metadata_cache[code] = exact_metadata
            return exact_metadata
        try:
            source = Path(code.co_filename).resolve()
            relative = str(source.relative_to(REPO_ROOT))
        except (OSError, ValueError):
            self._code_metadata_cache[code] = None
            return None
        qualname = str(getattr(code, "co_qualname", code.co_name))
        metadata = (
            relative,
            qualname,
            int(code.co_firstlineno),
            self._prohibited_categories(relative, qualname),
        )
        self._code_metadata_cache[code] = metadata
        return metadata

    def _external_metadata_for_frame(
        self,
        frame: Any,
    ) -> tuple[str, str, str, int, str, tuple[str, ...]]:
        code = frame.f_code
        module_name = str(frame.f_globals.get("__name__", "<unknown>"))
        cache_key = (code, module_name)
        if cache_key in self._external_code_metadata_cache:
            return self._external_code_metadata_cache[cache_key]
        source_file = str(code.co_filename)
        if not source_file.startswith("<"):
            source_file = str(Path(source_file).resolve())
        qualname = str(getattr(code, "co_qualname", code.co_name))
        classification, categories = _external_call_classification(
            module_name,
            source_file,
            qualname,
        )
        metadata = (
            module_name,
            source_file,
            qualname,
            int(code.co_firstlineno),
            classification,
            categories,
        )
        self._external_code_metadata_cache[cache_key] = metadata
        return metadata

    def _profile(self, frame: Any, event: str, _argument: Any) -> None:
        if event != "call" or self._active_inventory is None:
            return
        metadata = self._metadata_for_code(frame.f_code)
        current = threading.current_thread()
        thread_key = (int(threading.get_ident()), current.name)
        with self._inventory_lock:
            if self._active_inventory is None:
                return
            if metadata is not None:
                self._active_inventory[metadata] = (
                    self._active_inventory.get(metadata, 0) + 1
                )
                kind = "repository_call_count"
            else:
                external_metadata = self._external_metadata_for_frame(frame)
                if self._active_external_inventory is None:
                    return
                self._active_external_inventory[external_metadata] = (
                    self._active_external_inventory.get(external_metadata, 0) + 1
                )
                kind = "external_call_count"
            if self._active_thread_inventory is not None:
                thread_counts = self._active_thread_inventory.setdefault(
                    thread_key,
                    {"repository_call_count": 0, "external_call_count": 0},
                )
                thread_counts[kind] += 1

    def run_phase(self, phase_id: str, callback: Callable[[], Any]) -> Any:
        specs = {item["phase_id"]: item for item in FORMAL_GPU_ROLE_PHASE_SPECS}
        if phase_id not in specs:
            raise SmokeFailure(f"unregistered formal GPU role-audit phase: {phase_id}")
        if phase_id in {item["phase_id"] for item in self._phase_records}:
            raise SmokeFailure(f"formal GPU phase executed more than once: {phase_id}")
        if sys.getprofile() is not None:
            raise SmokeFailure(
                "an external Python profiler would invalidate GPU role audit"
            )
        threading_getprofile = getattr(threading, "getprofile", lambda: None)
        previous_thread_profile = threading_getprofile()
        if previous_thread_profile is not None:
            raise SmokeFailure(
                "an external threading profiler would invalidate GPU role audit"
            )
        self._active_inventory = {}
        self._active_external_inventory = {}
        self._active_thread_inventory = {}
        completed = False
        callback_error: BaseException | None = None
        result: Any = None
        threading.setprofile(self._profile)
        current_thread = threading.current_thread()
        preexisting_noncurrent_threads = [
            {
                "thread_id": (int(thread.ident) if thread.ident is not None else None),
                "native_id": (
                    int(thread.native_id)
                    if getattr(thread, "native_id", None) is not None
                    else None
                ),
                "thread_name": thread.name,
                "daemon": bool(thread.daemon),
            }
            for thread in sorted(
                (
                    thread
                    for thread in threading.enumerate()
                    if thread is not current_thread and thread.is_alive()
                ),
                key=lambda thread: (
                    thread.name,
                    int(thread.ident) if thread.ident is not None else -1,
                ),
            )
        ]
        try:
            if not preexisting_noncurrent_threads:
                sys.setprofile(self._profile)
                try:
                    result = callback()
                    completed = True
                except BaseException as error:
                    callback_error = error
                finally:
                    sys.setprofile(None)
        finally:
            threading.setprofile(previous_thread_profile)
        with self._inventory_lock:
            active_inventory = self._active_inventory or {}
            active_external_inventory = self._active_external_inventory or {}
            active_thread_inventory = self._active_thread_inventory or {}
            self._active_inventory = None
            self._active_external_inventory = None
            self._active_thread_inventory = None
        inventory = [
            {
                "source_file": relative,
                "function_qualname": qualname,
                "first_lineno": first_lineno,
                "categories": list(categories),
                "call_count": call_count,
            }
            for (relative, qualname, first_lineno, categories), call_count in sorted(
                active_inventory.items()
            )
        ]
        external_inventory = [
            {
                "module_name": module_name,
                "source_file": source_file,
                "function_qualname": qualname,
                "first_lineno": first_lineno,
                "classification": classification,
                "categories": list(categories),
                "call_count": call_count,
            }
            for (
                module_name,
                source_file,
                qualname,
                first_lineno,
                classification,
                categories,
            ), call_count in sorted(active_external_inventory.items())
        ]
        thread_inventory = [
            {
                "thread_id": thread_id,
                "thread_name": thread_name,
                **counts,
                "total_python_call_count": sum(counts.values()),
            }
            for (thread_id, thread_name), counts in sorted(
                active_thread_inventory.items()
            )
        ]
        # A formal phase is allowed to begin only when the current execution
        # thread is the sole live Python thread.  Do not take a second
        # "baseline" snapshot here: a thread that starts between two snapshots
        # could otherwise be mistaken for preexisting audited state and remain
        # alive after the callback without failing the phase.
        unjoined_new_threads = sorted(
            {
                int(thread.ident)
                for thread in threading.enumerate()
                if thread is not current_thread
                and thread.is_alive()
                and thread.ident is not None
            }
        )
        category_counts = {
            category: sum(
                int(record["call_count"])
                for record in (*inventory, *external_inventory)
                if category in record["categories"]
            )
            for category in (
                "generator_interface",
                "rate_interface",
                "sampler_interface",
                *ROLE_QUERY_CATEGORIES,
            )
        }
        spec = specs[phase_id]
        failure_reason: str | None = None
        if preexisting_noncurrent_threads:
            failure_reason = (
                "formal GPU phase started with unauditable preexisting noncurrent "
                f"Python threads: {phase_id}"
            )
        elif callback_error is not None:
            failure_reason = (
                "formal GPU phase callback failed: "
                f"{phase_id}: {type(callback_error).__name__}: {callback_error}"
            )
        elif not completed or not inventory:
            failure_reason = f"formal GPU role-audit phase was not observed: {phase_id}"
        else:
            for category in spec["required_call_categories"]:
                if category_counts[category] <= 0:
                    failure_reason = (
                        f"formal GPU phase lacks {category} calls: {phase_id}"
                    )
                    break
        if failure_reason is None and any(
            category_counts[category] != 0 for category in ROLE_QUERY_CATEGORIES
        ):
            failure_reason = f"prohibited role query observed in GPU phase: {phase_id}"
        unknown_external_call_count = sum(
            int(record["call_count"])
            for record in external_inventory
            if record["classification"] == "unknown_external"
        )
        if failure_reason is None and unknown_external_call_count:
            failure_reason = (
                f"unknown external Python calls observed in GPU phase: {phase_id}"
            )
        if failure_reason is None and unjoined_new_threads:
            failure_reason = (
                f"formal GPU phase left new Python threads running: {phase_id}"
            )
        external_python_call_count = sum(
            int(record["call_count"]) for record in external_inventory
        )
        record = {
            "phase_id": phase_id,
            "phase_kind": spec["phase_kind"],
            "entrypoint": spec["entrypoint"],
            "required_call_categories": list(spec["required_call_categories"]),
            "completed": completed,
            "phase_status": "PASS" if failure_reason is None else "FAILED",
            "failure_reason": failure_reason,
            "repository_python_call_count": sum(
                int(item["call_count"]) for item in inventory
            ),
            "external_python_call_count": external_python_call_count,
            "unknown_external_call_count": unknown_external_call_count,
            "total_python_call_count": sum(
                int(item["call_count"]) for item in inventory
            )
            + external_python_call_count,
            "python_thread_count": len(thread_inventory),
            "preexisting_noncurrent_python_thread_count": len(
                preexisting_noncurrent_threads
            ),
            "preexisting_noncurrent_python_threads": preexisting_noncurrent_threads,
            "unjoined_new_thread_ids": unjoined_new_threads,
            **{f"{key}_call_count": value for key, value in category_counts.items()},
            "call_inventory": inventory,
            "record_stream_sha256": _canonical_hash(inventory),
            "external_call_inventory": external_inventory,
            "external_record_stream_sha256": _canonical_hash(external_inventory),
            "thread_inventory": thread_inventory,
            "thread_record_stream_sha256": _canonical_hash(thread_inventory),
        }
        self._phase_records.append(record)
        if failure_reason is not None:
            failure = SmokeFailure(failure_reason)
            failure.partial_phase_evidence = self.partial_evidence()
            if callback_error is not None:
                raise failure from callback_error
            raise failure
        return result

    def partial_evidence(self) -> dict[str, Any]:
        """Return auditable phase records without claiming formal completion."""

        expected_ids = [item["phase_id"] for item in FORMAL_GPU_ROLE_PHASE_SPECS]
        count_keys = (
            "repository_python_call_count",
            "external_python_call_count",
            "unknown_external_call_count",
            "total_python_call_count",
            "generator_interface_call_count",
            "rate_interface_call_count",
            "sampler_interface_call_count",
            "critic_query_call_count",
            "guidance_query_call_count",
            "final_evaluator_query_call_count",
        )
        totals = {
            key: sum(int(phase[key]) for phase in self._phase_records)
            for key in count_keys
        }
        stream_material = [
            {
                "phase_id": phase["phase_id"],
                "phase_status": phase["phase_status"],
                "call_inventory": phase["call_inventory"],
                "external_call_inventory": phase["external_call_inventory"],
                "thread_inventory": phase["thread_inventory"],
                "preexisting_noncurrent_python_threads": phase[
                    "preexisting_noncurrent_python_threads"
                ],
            }
            for phase in self._phase_records
        ]
        source_binding = self._formal_binding.get("source_binding", {})
        return {
            "schema_version": "mk0_gpu_role_query_partial_evidence_v1",
            "status": "FAILED_WITH_PARTIAL_EVIDENCE",
            "runtime_instrumentation": "sys_setprofile_python_calls",
            "thread_scope": (
                "formal_gpu_main_and_new_threading_threads_with_"
                "preexisting_noncurrent_threads_forbidden"
            ),
            "run_id": self._run_id,
            "goal_sha256": self._formal_binding.get("goal_sha256"),
            "implementation_commit": self._formal_binding.get("implementation_commit"),
            "required_phase_ids": expected_ids,
            "observed_phase_ids": [phase["phase_id"] for phase in self._phase_records],
            "formal_gpu_phase_count": len(self._phase_records),
            "completed_phase_count": sum(
                bool(phase["completed"]) for phase in self._phase_records
            ),
            "failed_phase_count": sum(
                phase["phase_status"] == "FAILED" for phase in self._phase_records
            ),
            "phase_records": self._phase_records,
            "record_stream_sha256": _canonical_hash(stream_material),
            "audited_interfaces": self._interface_records,
            "audited_interface_count": len(self._interface_records),
            "tracked_source_files_sha256": source_binding.get(
                "tracked_source_files_sha256"
            ),
            **totals,
            "formal_gpu_computation_complete": False,
        }

    def finalize(self) -> dict[str, Any]:
        expected_ids = [item["phase_id"] for item in FORMAL_GPU_ROLE_PHASE_SPECS]
        observed_ids = [item["phase_id"] for item in self._phase_records]
        if observed_ids != expected_ids:
            failure = SmokeFailure(
                f"formal GPU role-audit phase coverage drift: {observed_ids}"
            )
            failure.partial_phase_evidence = self.partial_evidence()
            raise failure
        if any(phase["phase_status"] != "PASS" for phase in self._phase_records):
            failure = SmokeFailure("formal GPU role-audit includes a failed phase")
            failure.partial_phase_evidence = self.partial_evidence()
            raise failure
        interface_records = self._interface_records
        interface_failures = sum(
            bool(record["prohibited_parameters"]) for record in interface_records
        )
        if interface_failures:
            failure = SmokeFailure(
                "formal GPU interface exposes a prohibited role input"
            )
            failure.partial_phase_evidence = self.partial_evidence()
            raise failure
        totals = {
            key: sum(int(phase[key]) for phase in self._phase_records)
            for key in (
                "repository_python_call_count",
                "external_python_call_count",
                "unknown_external_call_count",
                "total_python_call_count",
                "generator_interface_call_count",
                "rate_interface_call_count",
                "sampler_interface_call_count",
                "critic_query_call_count",
                "guidance_query_call_count",
                "final_evaluator_query_call_count",
            )
        }
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
            for phase in self._phase_records
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
                "stdlib_root": str(STDLIB_ROOT),
                "site_package_roots_excluded_from_stdlib": [
                    str(path) for path in SITE_PACKAGE_ROOTS
                ],
                "frozen_foundation_module_prefixes": list(
                    FROZEN_FOUNDATION_EXTERNAL_MODULE_PREFIXES
                ),
            },
            "run_id": self._run_id,
            "goal_sha256": self._formal_binding["goal_sha256"],
            "implementation_commit": self._formal_binding["implementation_commit"],
            "run_manifest": {
                "path": self._formal_binding["run_manifest_path"],
                "sha256": self._formal_binding["run_manifest_sha256"],
            },
            "preflight": self._formal_binding["preflight"],
            "source_binding_sha256": _canonical_hash(
                self._formal_binding["source_binding"]
            ),
            "tracked_source_files_sha256": self._formal_binding["source_binding"][
                "tracked_source_files_sha256"
            ],
            "required_phase_ids": expected_ids,
            "formal_gpu_phase_count": len(self._phase_records),
            "completed_phase_count": sum(
                bool(phase["completed"]) for phase in self._phase_records
            ),
            "phase_records": self._phase_records,
            "record_stream_sha256": _canonical_hash(stream_material),
            "audited_interfaces": interface_records,
            "audited_interface_count": len(interface_records),
            "interface_failure_count": interface_failures,
            **totals,
            "all_role_query_counts_zero": all(
                totals[f"{category}_call_count"] == 0
                for category in ROLE_QUERY_CATEGORIES
            ),
            "formal_gpu_computation_complete": True,
            "all_external_calls_allowlisted": totals["unknown_external_call_count"]
            == 0,
            "all_new_threads_joined": all(
                not phase["unjoined_new_thread_ids"] for phase in self._phase_records
            ),
            "all_preexisting_noncurrent_threads_absent": all(
                phase["preexisting_noncurrent_python_thread_count"] == 0
                and phase["preexisting_noncurrent_python_threads"] == []
                for phase in self._phase_records
            ),
        }


_ACTIVE_ROLE_QUERY_RECORDER: _FormalGpuRoleQueryRecorder | None = None


def _standard_failure_reason(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _attach_active_partial_phase_evidence(error: BaseException) -> None:
    if getattr(error, "partial_phase_evidence", None) is not None:
        return
    recorder = _ACTIVE_ROLE_QUERY_RECORDER
    if recorder is None:
        return
    try:
        error.partial_phase_evidence = recorder.partial_evidence()
    except BaseException:
        pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_canonical_exclusive(path: Path, value: Any) -> str:
    """Write canonical JSON without ever replacing an existing artifact."""

    data = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def _write_canonical_atomic_exclusive(path: Path, value: Any) -> str:
    """Atomically publish canonical JSON without replacing an existing path."""

    data = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # link(2) is an atomic create-if-absent publication.  Unlike rename,
        # it cannot replace evidence already present at the final path.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(data).hexdigest()


def _write_failure_best_effort(
    output_dir: Path,
    *,
    run_id: str,
    snapshot_dir: str,
    device: str,
    error: BaseException,
) -> Path | None:
    partial_phase_evidence = getattr(error, "partial_phase_evidence", None)
    payload = {
        "schema_version": "mk0_gpu_smoke_failure_v1",
        "run_id": run_id,
        "status": "FAILED_WITH_EVIDENCE",
        "evidence_level": "E0_MATH_ENGINEERING_ONLY",
        "created_at_utc": _utc_now(),
        "requested_snapshot_dir": snapshot_dir,
        "requested_device": device,
        "exception_type": type(error).__name__,
        "exception_message": _standard_failure_reason(error),
        "traceback": traceback.format_exc(),
        "partial_phase_evidence": partial_phase_evidence,
        "cpu_fallback_allowed": False,
        "scientific_claims": {
            "functional_improvement": False,
            "matched_budget_superiority": False,
            "paper_success": False,
        },
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates = [output_dir / FAILURE_FILENAME]
        candidates.extend(
            output_dir / f"mk0_gpu_smoke_failure.{index}.json"
            for index in range(1, 1000)
        )
        for candidate in candidates:
            try:
                _write_canonical_exclusive(candidate, payload)
                return candidate
            except FileExistsError:
                continue
    except Exception:
        return None
    return None


def _load_frozen_fm0_config() -> dict[str, Any]:
    with FM0_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SmokeFailure("FM0 frozen config did not parse as an object")
    if config.get("meta", {}).get("frozen") is not True:
        raise SmokeFailure("FM0 model config is not marked frozen")
    return config


def _hash_file(path: Path) -> str:
    return sha256_file(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SmokeFailure(f"JSON artifact is not an object: {path}")
    return value


def _git(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as error:
        raise SmokeFailure(f"Git binding command failed: {arguments}") from error
    return completed.stdout.rstrip("\n")


def _validate_formal_source_binding(
    *,
    run_id: str,
    goal_sha256: str,
    implementation_commit: str,
    run_manifest_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    """Bind the GPU evidence to the same clean implementation as CPU gates."""

    if not re.fullmatch(r"[0-9a-f]{64}", goal_sha256):
        raise SmokeFailure("--goal-sha256 is not a lowercase SHA-256")
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise SmokeFailure("--implementation-commit is not a full Git commit")
    math_config = yaml.safe_load(MATH_CONFIG_PATH.read_text(encoding="utf-8"))
    if math_config["contract"]["sha256"] != goal_sha256:
        raise SmokeFailure("GPU goal hash differs from frozen math-kernel contract")

    manifest_path = run_manifest_path.expanduser().resolve(strict=True)
    preflight_resolved = preflight_path.expanduser().resolve(strict=True)
    run_manifest = _load_json_object(manifest_path)
    if run_manifest.get("run_id") != run_id:
        raise SmokeFailure("run manifest run_id differs from GPU run")
    if run_manifest.get("goal_sha256") != goal_sha256:
        raise SmokeFailure("run manifest goal hash differs from GPU run")
    if run_manifest.get("implementation_commit") != implementation_commit:
        raise SmokeFailure("run manifest implementation commit differs from GPU run")
    run_root = Path(str(run_manifest.get("run_root", ""))).resolve(strict=True)
    if not run_root.is_dir() or manifest_path != run_root / "run_manifest.json":
        raise SmokeFailure("run manifest path/root binding is invalid")

    source_binding = run_manifest.get("source_binding")
    if not isinstance(source_binding, dict):
        raise SmokeFailure("run manifest lacks source_binding")
    expected_source_keys = {
        "repo_root",
        "git_commit",
        "git_tree",
        "git_status_porcelain",
        "tracked_source_file_count",
        "tracked_source_files_sha256",
        "tracked_source_files",
    }
    if set(source_binding) != expected_source_keys:
        raise SmokeFailure("run manifest source_binding shape drift")
    if source_binding["repo_root"] != str(REPO_ROOT):
        raise SmokeFailure("source_binding repo root differs from GPU checkout")
    if source_binding["git_commit"] != implementation_commit:
        raise SmokeFailure("source_binding commit differs from GPU commit")
    if _git("rev-parse", "HEAD") != implementation_commit:
        raise SmokeFailure("GPU checkout HEAD differs from implementation commit")
    if _git("status", "--porcelain=v1", "--untracked-files=all") != "":
        raise SmokeFailure("GPU formal run requires a completely clean worktree")
    tree = _git("rev-parse", "HEAD^{tree}")
    if source_binding["git_tree"] != tree:
        raise SmokeFailure("GPU checkout tree differs from source binding")
    if source_binding["git_status_porcelain"] != "":
        raise SmokeFailure("source binding does not certify a clean tree")

    tracked = source_binding["tracked_source_files"]
    if not isinstance(tracked, dict) or not tracked:
        raise SmokeFailure("source binding tracked-file inventory is empty")
    if source_binding["tracked_source_file_count"] != len(tracked):
        raise SmokeFailure("source binding tracked-file count mismatch")
    if source_binding["tracked_source_files_sha256"] != _canonical_hash(tracked):
        raise SmokeFailure("source binding inventory digest mismatch")
    for relative, expected_sha256 in sorted(tracked.items()):
        if not isinstance(relative, str) or not re.fullmatch(
            r"[0-9a-f]{64}", str(expected_sha256)
        ):
            raise SmokeFailure("malformed source binding entry")
        path = (REPO_ROOT / relative).resolve(strict=True)
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as error:
            raise SmokeFailure("source binding path escapes checkout") from error
        if _hash_file(path) != expected_sha256:
            raise SmokeFailure(f"current source hash differs for {relative}")
        committed = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "show",
                f"{implementation_commit}:{relative}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != expected_sha256:
            raise SmokeFailure(f"committed source hash differs for {relative}")

    preflight = run_manifest.get("preflight")
    if not isinstance(preflight, dict):
        raise SmokeFailure("run manifest lacks preflight binding")
    if Path(str(preflight.get("path"))).resolve() != preflight_resolved:
        raise SmokeFailure("run manifest preflight path differs from CLI path")
    if preflight.get("sha256") != _hash_file(preflight_resolved):
        raise SmokeFailure("preflight artifact digest mismatch")
    preflight_payload = _load_json_object(preflight_resolved)
    if preflight_payload.get("run_id") != run_id:
        raise SmokeFailure("preflight run_id differs from GPU run")
    if preflight_payload.get("parent_run_id") != run_manifest.get("parent_run_id"):
        raise SmokeFailure("preflight parent run differs from GPU manifest")
    if preflight.get("parent_run_id") != run_manifest.get("parent_run_id"):
        raise SmokeFailure("preflight binding parent differs from GPU manifest")
    if preflight_payload.get("goal_sha256") != goal_sha256:
        raise SmokeFailure("preflight goal hash differs from GPU run")
    observed_preflight_head = preflight_payload.get("worktree", {}).get("head")
    if preflight.get("preflight_worktree_head") != observed_preflight_head:
        raise SmokeFailure(
            "preflight binding does not match its recorded worktree head"
        )

    return {
        "goal_sha256": goal_sha256,
        "implementation_commit": implementation_commit,
        "run_root": str(run_root),
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": _hash_file(manifest_path),
        "preflight": preflight,
        "preflight_payload": preflight_payload,
        "source_binding": source_binding,
    }


def _snapshot_manifest(snapshot_dir: Path) -> list[dict[str, Any]]:
    files = sorted(path for path in snapshot_dir.rglob("*") if path.is_file())
    if not files:
        raise SmokeFailure("frozen UTR-LM snapshot contains no readable files")
    return [
        {
            "path": str(path.relative_to(snapshot_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": _hash_file(path),
        }
        for path in files
    ]


def _fm0_checkpoint_manifest_binding(
    preflight_payload: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    upstream = preflight_payload.get("upstream")
    if not isinstance(upstream, Mapping):
        raise SmokeFailure("preflight lacks the FM0 upstream binding")
    fm0_root = Path(str(upstream.get("fm0_closure_root", ""))).resolve(strict=True)
    ledger = fm0_root / "artifact_checksums.sha256"
    if not ledger.is_file():
        raise SmokeFailure("FM0 checksum ledger is missing")
    if _hash_file(ledger) != upstream.get("fm0_checksum_ledger_sha256"):
        raise SmokeFailure("FM0 checksum ledger differs from the preflight binding")

    expected_manifest_sha256 = None
    seen_paths: set[str] = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (?:\./)?(.+)", line)
        if match is None:
            raise SmokeFailure("FM0 checksum ledger has a malformed line")
        relative = match.group(2)
        if relative in seen_paths:
            raise SmokeFailure("FM0 checksum ledger contains a duplicate path")
        seen_paths.add(relative)
        if relative == FM0_HASH_LICENSE_RELATIVE.as_posix():
            expected_manifest_sha256 = match.group(1)
    if expected_manifest_sha256 is None:
        raise SmokeFailure("FM0 ledger omits the checkpoint hash/license manifest")

    manifest_path = fm0_root / FM0_HASH_LICENSE_RELATIVE
    if _hash_file(manifest_path) != expected_manifest_sha256:
        raise SmokeFailure("FM0 checkpoint manifest differs from its closure ledger")
    return fm0_root, ledger, _load_json_object(manifest_path)


def _validate_snapshot_binding(
    snapshot_dir: Path,
    config: Mapping[str, Any],
    preflight_payload: Mapping[str, Any],
) -> dict[str, Any]:
    model_config = config["model"]
    storage_config = config["storage"]
    expected_revision = str(model_config["revision"])
    resolved = snapshot_dir.expanduser().resolve(strict=True)
    configured = Path(str(storage_config["snapshot_dir"])).resolve(strict=True)
    if resolved != configured:
        raise SmokeFailure(
            "provided snapshot is not the exact path frozen by fm0_utrlm_config.yaml"
        )
    if resolved.name != expected_revision:
        raise SmokeFailure(
            f"snapshot revision mismatch: {resolved.name!r} != {expected_revision!r}"
        )
    config_json = resolved / "config.json"
    if not config_json.is_file():
        raise SmokeFailure("frozen snapshot is missing config.json")
    weight_candidates = tuple(resolved.glob("*.safetensors")) + tuple(
        resolved.glob("pytorch_model*.bin")
    )
    if not weight_candidates:
        raise SmokeFailure("frozen snapshot has no checkpoint weight file")
    fm0_root, fm0_ledger, checkpoint_manifest = _fm0_checkpoint_manifest_binding(
        preflight_payload
    )
    if checkpoint_manifest.get("task_id") != "FM0-01":
        raise SmokeFailure("FM0 checkpoint manifest task identity drift")
    if checkpoint_manifest.get("manifest_kind") != "foundation_checkpoint_hash_license":
        raise SmokeFailure("FM0 checkpoint manifest kind drift")
    if checkpoint_manifest.get("model_id") != str(model_config["model_id"]):
        raise SmokeFailure("FM0 checkpoint manifest model ID drift")
    if checkpoint_manifest.get("revision") != expected_revision:
        raise SmokeFailure("FM0 checkpoint manifest revision drift")
    if (
        Path(str(checkpoint_manifest.get("snapshot_dir", ""))).resolve(strict=True)
        != resolved
    ):
        raise SmokeFailure("FM0 checkpoint manifest snapshot path drift")
    if checkpoint_manifest.get("missing_files") != []:
        raise SmokeFailure("FM0 checkpoint manifest records missing files")

    expected_names = checkpoint_manifest.get("expected_files")
    expected_records = checkpoint_manifest.get("files")
    if not isinstance(expected_names, list) or not expected_names:
        raise SmokeFailure("FM0 checkpoint expected-file inventory is empty")
    if not isinstance(expected_records, list) or not expected_records:
        raise SmokeFailure("FM0 checkpoint file bindings are empty")
    normalized_expected: list[dict[str, Any]] = []
    seen_expected: set[str] = set()
    for record in expected_records:
        if not isinstance(record, Mapping):
            raise SmokeFailure("FM0 checkpoint file binding is malformed")
        filename = record.get("filename")
        size_bytes = record.get("size_bytes")
        sha256 = record.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or filename in seen_expected
            or Path(filename).name != filename
        ):
            raise SmokeFailure("FM0 checkpoint filename is invalid or duplicated")
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise SmokeFailure("FM0 checkpoint file size is invalid")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise SmokeFailure("FM0 checkpoint file hash is invalid")
        seen_expected.add(filename)
        normalized_expected.append(
            {"path": filename, "size_bytes": size_bytes, "sha256": sha256}
        )
    normalized_expected.sort(key=lambda record: record["path"])
    if set(expected_names) != seen_expected or len(expected_names) != len(
        seen_expected
    ):
        raise SmokeFailure("FM0 expected-file list differs from its hash bindings")
    license_binding = checkpoint_manifest.get("license")
    if not isinstance(license_binding, Mapping):
        raise SmokeFailure("FM0 checkpoint license binding is missing")
    if str(license_binding.get("type", "")).lower() != "agpl-3.0":
        raise SmokeFailure("FM0 checkpoint license type drift")
    license_file = next(
        (record for record in normalized_expected if record["path"] == "license.md"),
        None,
    )
    if license_file is None:
        raise SmokeFailure("FM0 checkpoint inventory omits license.md")
    if (
        license_binding.get("license_md_sha256") != license_file["sha256"]
        or license_binding.get("license_md_size") != license_file["size_bytes"]
    ):
        raise SmokeFailure("FM0 license semantics differ from license.md bytes")

    file_manifest = _snapshot_manifest(resolved)
    if file_manifest != normalized_expected:
        raise SmokeFailure("frozen UTR-LM snapshot bytes differ from FM0 closure")
    manifest_path = fm0_root / FM0_HASH_LICENSE_RELATIVE
    return {
        "model_id": str(model_config["model_id"]),
        "expected_revision": expected_revision,
        "observed_revision": resolved.name,
        "snapshot_dir": str(resolved),
        "snapshot_matches_frozen_config": True,
        "fm0_config_path": str(FM0_CONFIG_PATH),
        "fm0_config_sha256": _hash_file(FM0_CONFIG_PATH),
        "file_count": len(file_manifest),
        "files": file_manifest,
        "snapshot_manifest_sha256": _canonical_hash(file_manifest),
        "fm0_closure_root": str(fm0_root),
        "fm0_checksum_ledger_path": str(fm0_ledger),
        "fm0_checksum_ledger_sha256": _hash_file(fm0_ledger),
        "fm0_hash_license_manifest_path": str(manifest_path),
        "fm0_hash_license_manifest_sha256": _hash_file(manifest_path),
        "fm0_expected_snapshot_manifest_sha256": _canonical_hash(normalized_expected),
        "fm0_expected_snapshot_file_count": len(normalized_expected),
        "snapshot_bytes_match_fm0_closure": True,
        "license_binding": {
            "type": "agpl-3.0",
            "license_md_sha256": license_file["sha256"],
            "license_md_size": license_file["size_bytes"],
        },
    }


def _normalise_device(device_text: str) -> torch.device:
    device = require_neural_cuda(device_text)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.index < 0 or device.index >= torch.cuda.device_count():
        raise SmokeFailure(
            f"requested logical CUDA index {device.index} is unavailable"
        )
    torch.cuda.set_device(device)
    probe = torch.ones(1, device=device, dtype=torch.float32)
    if probe.device.type != "cuda" or probe.dtype != torch.float32:
        raise SmokeFailure("CUDA FP32 tensor allocation silently fell back")
    return device


def _nvidia_smi_uuid(logical_index: int) -> str | None:
    """Map a logical CUDA ordinal to a canonical nvidia-smi GPU UUID."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    by_index: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2:
            by_index[parts[0]] = parts[1]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [item.strip() for item in visible.split(",") if item.strip()]
        if logical_index < len(entries):
            physical = entries[logical_index]
            if physical.startswith("GPU-"):
                return physical if physical in set(by_index.values()) else None
            if physical.startswith("MIG-"):
                return None
            return by_index.get(physical)
    return by_index.get(str(logical_index))


def _normalise_gpu_uuid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if re.fullmatch(r"[0-9A-Fa-f-]{36}", text) is not None:
        text = f"GPU-{text}"
    if re.fullmatch(r"GPU-[0-9A-Fa-f-]{36}", text) is None:
        return None
    return text


def _cuda_identity(device: torch.device) -> dict[str, Any]:
    index = int(device.index)
    properties = torch.cuda.get_device_properties(device)
    uuid_value = getattr(properties, "uuid", None)
    properties_uuid = str(uuid_value) if uuid_value is not None else None
    # nvidia-smi is also the preflight source of truth.  Resolve the logical
    # CUDA index through CUDA_VISIBLE_DEVICES and always publish that canonical
    # GPU- UUID so PyTorch version-specific UUID formatting cannot break (or
    # weaken) the finalizer's independent cross-binding.
    uuid_text = _nvidia_smi_uuid(index)
    if not uuid_text or re.fullmatch(r"GPU-[0-9A-Fa-f-]{36}", uuid_text) is None:
        raise SmokeFailure(
            "canonical nvidia-smi CUDA device UUID could not be recorded"
        )
    normalized_properties_uuid = _normalise_gpu_uuid(properties_uuid)
    if (
        normalized_properties_uuid is not None
        and normalized_properties_uuid != uuid_text
    ):
        raise SmokeFailure(
            "PyTorch CUDA UUID differs from the CUDA_VISIBLE_DEVICES/nvidia-smi mapping"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_entries = (
        [item.strip() for item in visible.split(",") if item.strip()] if visible else []
    )
    if len(visible_entries) != 1 or visible_entries[0] != uuid_text or index != 0:
        raise SmokeFailure(
            "formal GPU smoke requires CUDA_VISIBLE_DEVICES=<single physical GPU UUID> and cuda:0"
        )
    return {
        "requested_device": str(device),
        "logical_device_index": index,
        "current_device_index": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(device),
        "device_uuid": uuid_text,
        "torch_device_properties_uuid": properties_uuid,
        "torch_device_properties_uuid_normalized": normalized_properties_uuid,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
    }


def _parameter_inventory(module: torch.nn.Module) -> dict[str, Any]:
    parameters = tuple(module.parameters())
    floating = tuple(
        parameter for parameter in parameters if parameter.is_floating_point()
    )
    devices = sorted({str(parameter.device) for parameter in parameters})
    dtypes = sorted({str(parameter.dtype) for parameter in floating})
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_devices": devices,
        "floating_parameter_dtypes": dtypes,
        "all_parameters_cuda": bool(parameters)
        and all(parameter.device.type == "cuda" for parameter in parameters),
        "all_floating_parameters_fp32": bool(floating)
        and all(parameter.dtype == torch.float32 for parameter in floating),
    }


def _assert_module_cuda_fp32(
    module: torch.nn.Module, *, trainable_expected: bool, label: str
) -> dict[str, Any]:
    inventory = _parameter_inventory(module)
    if not inventory["all_parameters_cuda"]:
        raise SmokeFailure(f"{label} contains a CPU parameter")
    if not inventory["all_floating_parameters_fp32"]:
        raise SmokeFailure(f"{label} contains a non-FP32 floating parameter")
    trainable = int(inventory["trainable_parameter_count"])
    if trainable_expected and trainable <= 0:
        raise SmokeFailure(f"{label} unexpectedly has no trainable parameters")
    if not trainable_expected and trainable != 0:
        raise SmokeFailure(f"{label} foundation parameters are not frozen")
    return inventory


def _architecture_signature(model: torch.nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        raise SmokeFailure("UTR-LM instance has no architecture config")
    names = (
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
    )
    return {name: getattr(config, name) for name in names}


def _assert_frozen_config_architecture(
    signature: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    expected = config["model"]
    for name, observed in signature.items():
        if name in expected and observed != expected[name]:
            raise SmokeFailure(
                f"UTR-LM architecture drift for {name}: {observed!r} != {expected[name]!r}"
            )


def _clone_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed_parameter_count(
    before: Sequence[torch.Tensor], module: torch.nn.Module
) -> int:
    after = tuple(module.parameters())
    if len(before) != len(after):
        raise SmokeFailure("module parameter structure changed during GPU smoke")
    return sum(
        not torch.equal(reference, parameter.detach())
        for reference, parameter in zip(before, after)
    )


def _gradient_inventory(module: torch.nn.Module) -> dict[str, Any]:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    all_cuda = bool(gradients) and all(
        gradient.device.type == "cuda" for gradient in gradients
    )
    all_fp32 = bool(gradients) and all(
        gradient.dtype == torch.float32 for gradient in gradients
    )
    all_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    total_l1 = (
        float(sum(gradient.detach().abs().sum() for gradient in gradients).item())
        if gradients
        else 0.0
    )
    return {
        "gradient_tensor_count": len(gradients),
        "all_gradients_cuda": all_cuda,
        "all_gradients_fp32": all_fp32,
        "all_gradients_finite": all_finite,
        "gradient_l1_norm": total_l1,
    }


def _attach_foundation_forward_telemetry(
    foundation: torch.nn.Module,
) -> dict[str, Any]:
    telemetry: dict[str, Any] = {
        "forward_calls": 0,
        "input_tensor_devices": [],
        "input_tensor_dtypes": [],
        "output_tensor_devices": [],
        "output_tensor_dtypes": [],
        "all_input_tensors_cuda": True,
        "all_hidden_outputs_cuda_fp32": True,
    }

    def remember(key: str, value: str) -> None:
        values = telemetry[key]
        if value not in values:
            values.append(value)

    def pre_hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        tensors = [value for value in args if isinstance(value, torch.Tensor)]
        tensors.extend(
            value for value in kwargs.values() if isinstance(value, torch.Tensor)
        )
        if not tensors:
            raise SmokeFailure("UTR-LM forward received no tensor inputs")
        for tensor in tensors:
            remember("input_tensor_devices", str(tensor.device))
            remember("input_tensor_dtypes", str(tensor.dtype))
            if tensor.device.type != "cuda":
                telemetry["all_input_tensors_cuda"] = False
                raise SmokeFailure("UTR-LM forward received a CPU tensor")

    def post_hook(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor):
            raise SmokeFailure("UTR-LM forward returned no last_hidden_state tensor")
        remember("output_tensor_devices", str(hidden.device))
        remember("output_tensor_dtypes", str(hidden.dtype))
        telemetry["forward_calls"] += 1
        if hidden.device.type != "cuda" or hidden.dtype != torch.float32:
            telemetry["all_hidden_outputs_cuda_fp32"] = False
            raise SmokeFailure("UTR-LM hidden output is not CUDA FP32")

    foundation.register_forward_pre_hook(pre_hook, with_kwargs=True)
    foundation.register_forward_hook(post_hook, with_kwargs=True)
    return telemetry


def _forced_action_sequence() -> tuple[AtomicAction, ...]:
    return (
        AtomicAction(ActionType.INS, 2, "A"),
        AtomicAction(ActionType.SUB, 0, "C"),
        AtomicAction(ActionType.DEL, 3),
        AtomicAction(ActionType.STOP),
    )


def _run_forced_action_arm(
    *,
    name: str,
    foundation: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    min_length: int = 1,
    max_length: int = 32,
) -> tuple[FoundationFusionRateField, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(SEED + 101)
    torch.cuda.manual_seed_all(SEED + 101)
    rate_field = FoundationFusionRateField(
        foundation,
        tokenizer,
        device=device,
        min_length=min_length,
        max_length=max_length,
    )
    foundation_inventory = _assert_module_cuda_fp32(
        rate_field.foundation, trainable_expected=False, label=f"{name} foundation"
    )
    rate_head_inventory = _assert_module_cuda_fp32(
        rate_field.rate_head, trainable_expected=True, label=f"{name} rate head"
    )
    if rate_field.foundation.training:
        raise SmokeFailure(f"{name} frozen foundation is not in eval mode")
    forward_telemetry = _attach_foundation_forward_telemetry(rate_field.foundation)

    foundation_before = _clone_parameters(rate_field.foundation)
    head_before = _clone_parameters(rate_field.rate_head)
    optimizer = torch.optim.AdamW(
        rate_field.rate_head.parameters(), lr=1.0e-3, weight_decay=0.0
    )
    optimizer.zero_grad(set_to_none=True)

    state = EditState.initial(
        "ACGUACGU",
        region="5UTR",
        context={"assay": "synthetic_gpu_smoke", "endpoint": "engineering_only"},
        target_condition="increase",
        budget=4,
    )
    times = (0.10, 0.30, 0.60, 0.80)
    records: list[dict[str, Any]] = []
    rates: list[torch.Tensor] = []
    initial_current_calls = rate_field.current_encode_calls
    for time, action in zip(times, _forced_action_sequence()):
        if not is_legal(state, action, min_length=min_length, max_length=max_length):
            raise SmokeFailure(f"forced {action.key} action is not legal")
        before_hash = state.state_hash
        before_current = state.current
        rate = rate_field(state, time, actions=(action,))[action]
        if rate.device.type != "cuda" or rate.dtype != torch.float32:
            raise SmokeFailure(f"forced {action.key} rate is not CUDA FP32")
        if not bool(torch.isfinite(rate)) or not bool(rate > 0.0):
            raise SmokeFailure(f"forced {action.key} rate is not finite and positive")
        rates.append(rate)
        transition = apply_action(
            state, action, min_length=min_length, max_length=max_length
        )
        state = transition.after
        records.append(
            {
                "action": action.to_dict(),
                "action_key": action.key,
                "time": time,
                "rate": float(rate.detach().item()),
                "rate_device": str(rate.device),
                "rate_dtype": str(rate.dtype),
                "rate_is_finite_positive": True,
                "state_hash_before": before_hash,
                "state_hash_after": state.state_hash,
                "current_before": before_current,
                "current_after": state.current,
                "remaining_budget_after": state.remaining_budget,
            }
        )

    # A positive sum has a guaranteed non-zero output-bias gradient through
    # softplus while retaining all four action-rate computation graphs.
    accumulated_loss = torch.stack(rates).sum()
    if (
        accumulated_loss.device.type != "cuda"
        or accumulated_loss.dtype != torch.float32
    ):
        raise SmokeFailure(f"{name} accumulated loss is not CUDA FP32")
    if not bool(torch.isfinite(accumulated_loss)):
        raise SmokeFailure(f"{name} accumulated loss is non-finite")
    accumulated_loss.backward()
    gradient_inventory = _gradient_inventory(rate_field.rate_head)
    if not (
        gradient_inventory["all_gradients_cuda"]
        and gradient_inventory["all_gradients_fp32"]
        and gradient_inventory["all_gradients_finite"]
        and gradient_inventory["gradient_l1_norm"] > 0.0
    ):
        raise SmokeFailure(f"{name} rate-head backward evidence is invalid")
    if any(
        parameter.grad is not None for parameter in rate_field.foundation.parameters()
    ):
        raise SmokeFailure(f"{name} frozen foundation received a gradient")
    optimizer.step()
    torch.cuda.synchronize(device)

    foundation_changes = _changed_parameter_count(
        foundation_before, rate_field.foundation
    )
    head_changes = _changed_parameter_count(head_before, rate_field.rate_head)
    if foundation_changes != 0:
        raise SmokeFailure(f"{name} frozen foundation parameters changed")
    if head_changes <= 0:
        raise SmokeFailure(f"{name} optimizer step changed no rate-head parameter")
    forced_current_calls = rate_field.current_encode_calls - initial_current_calls
    if forced_current_calls != 4:
        raise SmokeFailure(
            f"{name} expected four full current re-encodes, observed {forced_current_calls}"
        )
    if state.phase.value != "HALTED" or state.termination_reason is None:
        raise SmokeFailure(f"{name} forced STOP did not halt the state")

    return (
        rate_field,
        {
            "arm": name,
            "foundation_class": type(rate_field.foundation).__name__,
            "foundation_module": type(rate_field.foundation).__module__,
            "foundation_inventory": foundation_inventory,
            "rate_head_inventory": rate_head_inventory,
            "foundation_parameter_change_count": foundation_changes,
            "foundation_gradient_tensor_count": sum(
                parameter.grad is not None
                for parameter in rate_field.foundation.parameters()
            ),
            "rate_head_parameter_change_count": head_changes,
            "gradient_evidence": gradient_inventory,
            "optimizer": {
                "class": type(optimizer).__name__,
                "learning_rate": 1.0e-3,
                "step_count": 1,
            },
            "accumulated_loss": float(accumulated_loss.detach().item()),
            "forced_action_count": len(records),
            "forced_action_types": [record["action"]["kind"] for record in records],
            "forced_actions": records,
            "positive_finite_rate_count": sum(
                record["rate_is_finite_positive"] for record in records
            ),
            "full_current_reencode_calls_during_forced_sequence": forced_current_calls,
            "source_encode_calls_after_forced_sequence": rate_field.source_encode_calls,
            "current_encode_calls_after_forced_sequence": rate_field.current_encode_calls,
            "incremental_update_enabled": False,
            "final_phase": state.phase.value,
            "final_termination_reason": state.termination_reason.value,
        },
        forward_telemetry,
    )


def _synthetic_replacement(index: int) -> tuple[str, dict[str, Any]]:
    digits = []
    value = index + 1
    for _ in range(8):
        digits.append(ALPHABET[value % len(ALPHABET)])
        value //= len(ALPHABET)
    target = "".join(digits)
    alignment = {
        "synthetic_case": index,
        "target": target,
        "source_positions": list(range(7, -1, -1)),
        "switch_clock_order": [(index + offset) % 8 for offset in range(8)],
    }
    return target, alignment


def _audit_target_alignment_leakage(
    rate_field: FoundationFusionRateField,
) -> dict[str, Any]:
    signature = inspect.signature(FoundationFusionRateField.forward)
    parameter_names = list(signature.parameters)
    prohibited = {
        "z_aux",
        "target",
        "target_sequence",
        "target_alignment",
        "alignment",
        "remaining_target_edits",
    }
    present = sorted(prohibited.intersection(parameter_names))
    if present:
        raise SmokeFailure(f"rate-field interface exposes prohibited inputs: {present}")

    state = EditState.initial(
        "ACGUACGU",
        region="5UTR",
        context={"assay": "synthetic_leakage_audit"},
        target_condition="increase",
        budget=4,
    )
    actions = tuple(
        sorted(
            (
                AtomicAction(ActionType.STOP),
                AtomicAction(ActionType.INS, 0, "A"),
                AtomicAction(ActionType.SUB, 0, "C"),
                AtomicAction(ActionType.DEL, 0),
            ),
            key=lambda action: action.key,
        )
    )
    if not all(
        is_legal(
            state,
            action,
            min_length=rate_field.min_length,
            max_length=rate_field.max_length,
        )
        for action in actions
    ):
        raise SmokeFailure("target-leakage audit fixture contains an illegal action")

    calls_before = rate_field.current_encode_calls
    baseline_example = EditFlowTrainingExample(
        inference_state=state,
        training_auxiliary={
            "target_sequence": "AAAAAAAA",
            "target_alignment": {"synthetic_case": "baseline"},
            "remaining_target_edits": 0,
        },
    )
    baseline_rate_input = rate_input_state(baseline_example)
    baseline_input_bytes = canonical_rate_input_bytes(baseline_example)
    baseline = rate_field(baseline_rate_input, 0.5, actions=actions)
    cases: list[dict[str, Any]] = []
    failures = 0
    max_abs_difference = 0.0
    for index in range(64):
        target, alignment = _synthetic_replacement(index)
        # Exercise the actual training-example -> inference-state boundary.
        # The paired auxiliary changes on every case; the exact bytes supplied
        # to the neural field must remain identical.
        example = EditFlowTrainingExample(
            inference_state=state,
            training_auxiliary={
                "target_sequence": target,
                "target_alignment": alignment,
                "remaining_target_edits": (index % 8) + 1,
            },
        )
        observed_input_bytes = canonical_rate_input_bytes(example)
        rate_state = rate_input_state(example)
        input_bytes_equal = observed_input_bytes == baseline_input_bytes
        input_state_equal = rate_state.state_hash == baseline_rate_input.state_hash
        observed = rate_field(rate_state, 0.5, actions=actions)
        exact_equal = all(
            torch.equal(baseline[action], observed[action]) for action in actions
        )
        differences = [
            float((baseline[action] - observed[action]).abs().detach().item())
            for action in actions
        ]
        case_max = max(differences)
        max_abs_difference = max(max_abs_difference, case_max)
        case_pass = input_bytes_equal and input_state_equal and exact_equal
        failures += int(not case_pass)
        cases.append(
            {
                "case_id": index,
                "replacement_target_sha256": _sha256_text(target),
                "replacement_alignment_sha256": _canonical_hash(alignment),
                "training_example_boundary_exercised": True,
                "canonical_rate_input_sha256": hashlib.sha256(
                    observed_input_bytes
                ).hexdigest(),
                "canonical_rate_input_bytes_equal": input_bytes_equal,
                "inference_state_hash_equal": input_state_equal,
                "replacement_passed_to_rate_field": False,
                "exact_rate_vector_equal": exact_equal,
                "max_abs_difference": case_max,
                "passed": case_pass,
            }
        )

    constructor_rejections: list[dict[str, Any]] = []
    for key in (
        "z_aux",
        "target_sequence",
        "target_alignment",
        "remaining_target_edits",
        "target_derived_embedding",
    ):
        rejected = False
        diagnostic = None
        try:
            EditState.initial(
                "ACGUACGU",
                context={"assay": "synthetic_leakage_audit", key: "secret"},
                budget=4,
            )
        except ValueError as error:
            rejected = True
            diagnostic = str(error)
        if not rejected:
            raise SmokeFailure(f"training-only context key was accepted: {key}")
        constructor_rejections.append(
            {"key": key, "rejected": rejected, "diagnostic": diagnostic}
        )
    call_delta = rate_field.current_encode_calls - calls_before
    if call_delta != 65:
        raise SmokeFailure(
            f"leakage audit expected 65 full current re-encodes, observed {call_delta}"
        )
    if failures:
        raise SmokeFailure(
            f"{failures}/64 target/alignment replacements altered the rate vector"
        )
    return {
        "schema_version": "target_alignment_leakage_audit_v1",
        "status": "PASS",
        "evidence_level": "E0_MATH_ENGINEERING_ONLY",
        "seed": SEED,
        "test_domain": "64_identical_inference_states_with_replaced_synthetic_target_and_alignment_payloads",
        "coverage": "sampled_paired",
        "sample_count": 64,
        "failure_count": failures,
        "failure_denominator": 64,
        "dtype": "float32_gpu_and_float64_cpu_stub",
        "atol": FLOAT32_ATOL,
        "rtol": FLOAT32_RTOL,
        "exact_equality_also_required": True,
        "max_abs_difference": max_abs_difference,
        "rate_interface_parameters": parameter_names,
        "prohibited_interface_parameters_present": present,
        "inference_state_hash": state.state_hash,
        "canonical_rate_input_sha256": hashlib.sha256(baseline_input_bytes).hexdigest(),
        "constructor_boundary_rejections": constructor_rejections,
        "action_keys": [action.key for action in actions],
        "full_current_reencode_calls": call_delta,
        "cases": cases,
        "gate_binding": {
            "id": "M05",
            "name": "target_alignment_leakage",
            "test_domain": "64_identical_inference_states_with_permuted_or_replaced_Z_aux",
            "coverage": "sampled_paired",
            "exhaustive_or_sampled": "sampled_paired",
            "sample_count": 64,
            "failure_count": failures,
            "failure_denominator": 64,
            "dtype": "float32_gpu_and_float64_cpu_stub",
            "atol": FLOAT32_ATOL,
            "rtol": FLOAT32_RTOL,
            "seed": SEED,
            "passed": failures == 0,
            "metrics": {
                "maximum_absolute_rate_difference": max_abs_difference,
                "canonical_input_mismatch_count": sum(
                    not case["canonical_rate_input_bytes_equal"] for case in cases
                ),
                "constructor_rejection_failure_count": sum(
                    not item["rejected"] for item in constructor_rejections
                ),
            },
        },
        "scientific_claims": {
            "functional_improvement": False,
            "matched_budget_superiority": False,
            "paper_success": False,
        },
    }


def _direct_action_representation(
    rate_field: FoundationFusionRateField,
    state: EditState,
    action: AtomicAction,
) -> dict[str, torch.Tensor]:
    """Independent position/gap gather oracle over real foundation tensors."""

    source_tokens = rate_field._encode_tokens(state.source, source_cache=True)
    current_tokens = rate_field._encode_tokens(state.current, source_cache=False)
    zero = torch.zeros(
        rate_field.hidden_size, device=rate_field.device, dtype=torch.float32
    )
    aligned_tokens = torch.stack(
        [
            (
                source_tokens[int(ref.source_index)]
                if ref.origin == TokenOrigin.SOURCE
                else zero
            )
            for ref in state.mapping.tokens
        ]
    )

    def gap(tokens: torch.Tensor, index: int) -> torch.Tensor:
        neighbours: list[torch.Tensor] = []
        if index > 0:
            neighbours.append(tokens[index - 1])
        if index < tokens.shape[0]:
            neighbours.append(tokens[index])
        if not neighbours:
            return zero
        return torch.stack(neighbours).mean(dim=0)

    if action.kind == ActionType.INS:
        position = int(action.position)
        current_local = gap(current_tokens, position)
        source_aligned_local = gap(aligned_tokens, position)
    elif action.kind in {ActionType.SUB, ActionType.DEL}:
        position = int(action.position)
        current_local = current_tokens[position]
        source_aligned_local = aligned_tokens[position]
    else:
        current_local = current_tokens.mean(dim=0)
        source_aligned_local = aligned_tokens.mean(dim=0)
    return {
        "current_local": current_local.detach().clone(),
        "source_aligned_local": source_aligned_local.detach().clone(),
    }


def _inverse_local_audit_action(
    before: EditState, mutation: AtomicAction
) -> AtomicAction:
    position = int(mutation.position)
    if mutation.kind == ActionType.INS:
        return AtomicAction(ActionType.DEL, position)
    if mutation.kind == ActionType.SUB:
        return AtomicAction(ActionType.SUB, position, before.current[position])
    if mutation.kind == ActionType.DEL:
        return AtomicAction(ActionType.INS, position, before.current[position])
    raise SmokeFailure("M31 mutation fixture unexpectedly contains STOP")


def _audit_dynamic_current_encoding(
    rate_field: FoundationFusionRateField,
) -> dict[str, Any]:
    base = EditState.initial(
        "ACGUACGU",
        region="5UTR",
        context={"assay": "synthetic_stale_state_audit"},
        target_condition="increase",
        budget=8,
    )
    stop = AtomicAction(ActionType.STOP)
    mutations = tuple(
        sorted(
            enumerate_legal_actions(
                base,
                min_length=rate_field.min_length,
                max_length=rate_field.max_length,
                include_stop=False,
            ),
            key=lambda action: action.key,
        )[:64]
    )
    if len(mutations) != 64:
        raise SmokeFailure("M31 could not construct 64 legal mutation fixtures")
    calls_before = rate_field.current_encode_calls
    baseline_rate = rate_field(base, 0.4, actions=(stop,))[stop]
    cases: list[dict[str, Any]] = []
    failures = 0
    minimum_abs_rate_delta = float("inf")
    representation_failures = 0
    mutation_type_counts = {
        kind.value: 0 for kind in (ActionType.INS, ActionType.SUB, ActionType.DEL)
    }
    for index, mutation in enumerate(mutations):
        mutated = apply_action(
            base,
            mutation,
            min_length=rate_field.min_length,
            max_length=rate_field.max_length,
        ).after
        audit_action = _inverse_local_audit_action(base, mutation)
        if not is_legal(
            mutated,
            audit_action,
            min_length=rate_field.min_length,
            max_length=rate_field.max_length,
        ):
            raise SmokeFailure("M31 inverse local-audit action is not legal")
        observed_rate = rate_field(mutated, 0.4, actions=(stop,))[stop]
        production_representation = rate_field.action_representation_audit(
            mutated, 0.4, audit_action
        )
        oracle_representation = _direct_action_representation(
            rate_field, mutated, audit_action
        )
        local_exact = all(
            torch.equal(production_representation[key], oracle_representation[key])
            for key in ("current_local", "source_aligned_local")
        )
        local_cuda_fp32 = all(
            tensor.device.type == "cuda" and tensor.dtype == torch.float32
            for tensor in production_representation.values()
        )
        delta = float((baseline_rate - observed_rate).abs().detach().item())
        rate_changed = not torch.equal(baseline_rate, observed_rate)
        representation_failures += int(not (local_exact and local_cuda_fp32))
        case_pass = rate_changed and local_exact and local_cuda_fp32
        failures += int(not case_pass)
        minimum_abs_rate_delta = min(minimum_abs_rate_delta, delta)
        mutation_type_counts[mutation.kind.value] += 1
        cases.append(
            {
                "case_id": index,
                "source_sha256": _sha256_text(mutated.source),
                "current_sha256": _sha256_text(mutated.current),
                "state_hash": mutated.state_hash,
                "mutation_action": mutation.to_dict(),
                "local_audit_action": audit_action.to_dict(),
                "mapping_current_length_consistent": len(mutated.mapping.tokens)
                == len(mutated.current),
                "stop_rate_changed_exactly": rate_changed,
                "absolute_rate_delta": delta,
                "production_local_gather_matches_independent_oracle": local_exact,
                "local_representations_cuda_fp32": local_cuda_fp32,
                "passed": case_pass,
            }
        )
    call_delta = rate_field.current_encode_calls - calls_before
    expected_call_delta = 1 + 3 * len(mutations)
    if call_delta != expected_call_delta:
        raise SmokeFailure(
            f"stale-state audit expected {expected_call_delta} full current re-encodes, observed {call_delta}"
        )
    if failures:
        raise SmokeFailure(
            f"{failures}/64 current mutations yielded an exactly stale STOP rate"
        )
    return {
        "test_domain": "64_source_fixed_current_mutated_paired_rate_checks",
        "coverage": "sampled_paired",
        "sample_count": 64,
        "failure_count": failures,
        "failure_denominator": 64,
        "dtype": "float32_gpu",
        "atol": FLOAT32_ATOL,
        "rtol": FLOAT32_RTOL,
        "exact_rate_change_required": True,
        "minimum_absolute_rate_delta": minimum_abs_rate_delta,
        "full_current_reencode_calls": call_delta,
        "representation_oracle_failure_count": representation_failures,
        "mutation_action_type_counts": mutation_type_counts,
        "cases": cases,
    }


def _run_official_paper_sampler_route(
    rate_field: FoundationFusionRateField,
) -> dict[str, Any]:
    """Exercise the actual paper sampler through the official-only adapter."""

    adapter = OfficialPaperRateAdapter(rate_field)
    stop = AtomicAction(ActionType.STOP)

    def stop_only_official(state: EditState, time: float) -> dict[AtomicAction, float]:
        all_rates = adapter(state, time)
        # Scaling is an explicitly labelled sampling oracle used only to force
        # the STOP branch in this E0 integration check.  The underlying rate
        # is still produced by the real CUDA foundation/head path.
        return {stop: max(64.0, 64.0 * all_rates[stop])}

    result = paper_first_order_parallel(
        EditState.initial(
            "ACGUACGU",
            context={"assay": "synthetic_paper_route", "endpoint": "engineering_only"},
            budget=4,
        ),
        stop_only_official,
        step_size=0.25,
        min_length=rate_field.min_length,
        max_length=rate_field.max_length,
        seed=SEED + 3200,
    )
    replay_ok = replay_paper_result(result, stop_only_official)
    if not replay_ok:
        raise SmokeFailure("paper-mode real-foundation sampler replay failed")
    if (
        result.final_state.termination_reason is None
        or result.final_state.termination_reason.value != "LEARNED_STOP"
    ):
        raise SmokeFailure("paper-mode real-foundation route did not sample STOP")
    telemetry = dict(adapter.telemetry)
    if (
        telemetry["paper_rate_calls"] <= 0
        or telemetry["official_foundation_forward_calls"] <= 0
    ):
        raise SmokeFailure("paper-mode adapter observed no official foundation calls")
    if telemetry["placeholder_foundation_forward_calls"] != 0:
        raise SmokeFailure("paper-mode adapter observed a placeholder foundation call")
    return {
        **telemetry,
        "sampler": result.sampler,
        "sampler_step_count": len(result.steps),
        "sampler_replay_passed": replay_ok,
        "termination_reason": result.final_state.termination_reason.value,
        "exact_gillespie": result.exact_gillespie,
    }


def _run_primary_gpu_sampler_integration(
    rate_field: FoundationFusionRateField,
) -> dict[str, Any]:
    """Force all four action types through the primary sampler and GPU field."""

    plan = _forced_action_sequence()
    gpu_rate_calls = 0

    def planned_gpu_rates(state: EditState, time: float) -> dict[AtomicAction, float]:
        nonlocal gpu_rate_calls
        index = state.history.executed
        action = plan[index] if index < len(plan) - 1 else plan[-1]
        if not is_legal(
            state,
            action,
            min_length=rate_field.min_length,
            max_length=rate_field.max_length,
        ):
            raise SmokeFailure(f"planned sampler action became illegal: {action.key}")
        tensor_rate = rate_field(state, time, actions=(action,))[action]
        if tensor_rate.device.type != "cuda" or tensor_rate.dtype != torch.float32:
            raise SmokeFailure("primary sampler rate did not originate on CUDA FP32")
        if not bool(torch.isfinite(tensor_rate)) or not bool(tensor_rate > 0.0):
            raise SmokeFailure("primary sampler received a nonpositive neural rate")
        gpu_rate_calls += 1
        return {action: max(64.0, 64.0 * float(tensor_rate.detach().item()))}

    initial = EditState.initial(
        "ACGUACGU",
        region="5UTR",
        context={"assay": "synthetic_primary_sampler", "endpoint": "engineering_only"},
        target_condition="increase",
        # Four keeps STOP a learned event after three edits rather than an
        # administrative budget termination.
        budget=4,
    )
    result = constrained_single_event_first_order(
        initial,
        planned_gpu_rates,
        step_size=0.125,
        stability_hazard=0.05,
        min_length=rate_field.min_length,
        max_length=rate_field.max_length,
        seed=SEED + 3500,
    )
    replay_ok = replay_constrained_result(result, planned_gpu_rates)
    sampled_actions = tuple(
        step.selected_action
        for step in result.steps
        if step.selected_action is not None
    )
    if sampled_actions != plan:
        raise SmokeFailure(
            "primary GPU sampler did not traverse INS/SUB/DEL/STOP in order"
        )
    if not replay_ok:
        raise SmokeFailure("primary GPU sampler deterministic replay failed")
    return {
        "sampler": result.sampler,
        "sampler_step_count": len(result.steps),
        "gpu_rate_field_call_count": gpu_rate_calls,
        "sampled_action_keys": [action.key for action in sampled_actions],
        "sampled_action_types": [action.kind.value for action in sampled_actions],
        "sampler_replay_passed": replay_ok,
        "final_state_hash": result.final_state.state_hash,
        "termination_reason": result.final_state.termination_reason.value,
        "exact_gillespie": result.exact_gillespie,
    }


def _configure_numerics() -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    cuda_autocast = bool(torch.is_autocast_enabled("cuda"))
    if torch.is_autocast_enabled() or cuda_autocast:
        raise SmokeFailure("autocast/AMP was active during MK0 GPU smoke setup")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise SmokeFailure("TF32 could not be disabled")
    return {
        "default_dtype": str(torch.get_default_dtype()),
        "amp_enabled": False,
        "cuda_autocast_enabled": cuda_autocast,
        "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_enabled": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "master_seed": SEED,
    }


def _formal_gpu_role_interfaces() -> (
    tuple[tuple[str, Callable[..., Any], tuple[str, ...]], ...]
):
    """Return the exact production callables used for role attribution."""

    return (
        (
            "gpu_runner.forced_action_arm",
            _run_forced_action_arm,
            ("generator_interface",),
        ),
        ("gpu_runner.paper_sampler_route", _run_official_paper_sampler_route, ()),
        (
            "gpu_runner.primary_sampler_integration",
            _run_primary_gpu_sampler_integration,
            (),
        ),
        (
            "gpu_runner.target_alignment_leakage_audit",
            _audit_target_alignment_leakage,
            (),
        ),
        (
            "gpu_runner.dynamic_current_encoding_audit",
            _audit_dynamic_current_encoding,
            (),
        ),
        (
            "foundation_fusion.rate_field_forward",
            FoundationFusionRateField.forward,
            ("rate_interface",),
        ),
        (
            "foundation_fusion.official_paper_adapter",
            OfficialPaperRateAdapter.__call__,
            ("rate_interface",),
        ),
        (
            "samplers.constrained_primary",
            constrained_single_event_first_order,
            ("sampler_interface",),
        ),
        (
            "samplers.paper_parallel",
            paper_first_order_parallel,
            ("sampler_interface",),
        ),
        (
            "samplers.replay_constrained",
            replay_constrained_result,
            ("sampler_interface",),
        ),
        (
            "samplers.replay_paper",
            replay_paper_result,
            ("sampler_interface",),
        ),
    )


def _formal_gpu_role_interface_records() -> tuple[
    list[dict[str, Any]],
    dict[Any, tuple[str, str, int, tuple[str, ...]]],
]:
    """Audit interfaces and bind categories to exact code-object identities."""

    interfaces = _formal_gpu_role_interfaces()
    labels = tuple(label for label, _interface, _categories in interfaces)
    if labels != FORMAL_GPU_ROLE_INTERFACE_LABELS:
        raise SmokeFailure("formal GPU role-interface inventory/order drift")
    records: list[dict[str, Any]] = []
    role_code_metadata: dict[Any, tuple[str, str, int, tuple[str, ...]]] = {}
    for label, interface, categories in interfaces:
        source_file = inspect.getsourcefile(interface)
        if source_file is None:
            raise SmokeFailure(f"formal GPU interface lacks source: {label}")
        resolved = Path(source_file).resolve(strict=True)
        try:
            relative = str(resolved.relative_to(REPO_ROOT))
        except ValueError as error:
            raise SmokeFailure(
                f"formal GPU interface source escaped repository: {label}"
            ) from error
        code = getattr(interface, "__code__", None)
        if code is None:
            raise SmokeFailure(f"formal GPU interface lacks Python code: {label}")
        try:
            code_source = Path(code.co_filename).resolve(strict=True)
        except OSError as error:
            raise SmokeFailure(
                f"formal GPU interface code source is unreadable: {label}"
            ) from error
        if code_source != resolved:
            raise SmokeFailure(
                f"formal GPU interface source/code origin differs: {label}"
            )
        qualname = str(getattr(interface, "__qualname__", interface.__name__))
        metadata = (
            relative,
            qualname,
            int(code.co_firstlineno),
            tuple(sorted(categories)),
        )
        if code in role_code_metadata:
            raise SmokeFailure(f"formal GPU interface code identity reused: {label}")
        role_code_metadata[code] = metadata
        parameters = list(inspect.signature(interface).parameters)
        prohibited = sorted(
            parameter
            for parameter in parameters
            if any(token in parameter.lower() for token in ROLE_PROHIBITED_TOKENS)
        )
        records.append(
            {
                "interface": label,
                "source_file": relative,
                "source_file_sha256": _hash_file(resolved),
                "function_qualname": qualname,
                "first_lineno": int(code.co_firstlineno),
                "role_categories": list(metadata[3]),
                "parameters": parameters,
                "prohibited_parameters": prohibited,
            }
        )
    return records, role_code_metadata


def _validate_formal_runtime_interface_origins(
    formal_binding: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[Any, tuple[str, str, int, tuple[str, ...]]],
]:
    """Fail before model load if a runtime callable escaped the bound tree."""

    records, role_code_metadata = _formal_gpu_role_interface_records()
    tracked = formal_binding["source_binding"]["tracked_source_files"]
    for record in records:
        relative = record["source_file"]
        if relative not in tracked:
            raise SmokeFailure(
                f"formal GPU runtime interface is not source-bound: {record['interface']}"
            )
        if record["source_file_sha256"] != tracked[relative]:
            raise SmokeFailure(
                f"formal GPU runtime interface hash differs: {record['interface']}"
            )
    if len(role_code_metadata) != len(records):
        raise SmokeFailure("formal GPU runtime code-identity inventory drift")
    return records, role_code_metadata


def run_gpu_smoke(
    *,
    output_dir: Path,
    run_id: str,
    snapshot_dir: Path,
    device_text: str,
    goal_sha256: str,
    implementation_commit: str,
    run_manifest_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    global _ACTIVE_ROLE_QUERY_RECORDER
    _ACTIVE_ROLE_QUERY_RECORDER = None
    if not run_id.strip():
        raise SmokeFailure("--run-id must not be empty")
    if any(
        (output_dir / name).exists()
        for name in (FOUNDATION_FILENAME, LEAKAGE_FILENAME, GPU_RESULTS_FILENAME)
    ):
        raise SmokeFailure("refusing to overwrite an existing MK0 GPU audit artifact")

    formal_binding = _validate_formal_source_binding(
        run_id=run_id,
        goal_sha256=goal_sha256,
        implementation_commit=implementation_commit,
        run_manifest_path=run_manifest_path,
        preflight_path=preflight_path,
    )
    interface_records, role_code_metadata = _validate_formal_runtime_interface_origins(
        formal_binding
    )
    run_root = Path(formal_binding["run_root"])
    expected_output_dir = (run_root / "artifacts" / "mk0").resolve(strict=True)
    if output_dir.expanduser().resolve(strict=True) != expected_output_dir:
        raise SmokeFailure(
            "GPU output directory is not the bound MK0 artifact directory"
        )
    gpu_summary_path = run_root / "summary" / GPU_SUMMARY_FILENAME
    if gpu_summary_path.exists():
        raise SmokeFailure("refusing to overwrite an existing GPU acceptance summary")
    math_config = yaml.safe_load(MATH_CONFIG_PATH.read_text(encoding="utf-8"))
    config = _load_frozen_fm0_config()
    snapshot_binding = _validate_snapshot_binding(
        snapshot_dir, config, formal_binding["preflight_payload"]
    )
    device = _normalise_device(device_text)
    numerical_policy = _configure_numerics()
    cuda_identity = _cuda_identity(device)
    torch.cuda.reset_peak_memory_stats(device)

    official, official_tokenizer = load_official_utrlm(
        snapshot_binding["snapshot_dir"], device=device, from_scratch=False, seed=SEED
    )
    control, control_tokenizer = load_official_utrlm(
        snapshot_binding["snapshot_dir"], device=device, from_scratch=True, seed=SEED
    )
    post_load_snapshot_binding = _validate_snapshot_binding(
        snapshot_dir, config, formal_binding["preflight_payload"]
    )
    if post_load_snapshot_binding != snapshot_binding:
        raise SmokeFailure("foundation snapshot changed during model loading")
    snapshot_binding = {
        **snapshot_binding,
        "post_model_load_rehash_match": True,
    }
    official_signature = _architecture_signature(official)
    control_signature = _architecture_signature(control)
    _assert_frozen_config_architecture(official_signature, config)
    _assert_frozen_config_architecture(control_signature, config)
    if official_signature != control_signature:
        raise SmokeFailure(
            "from-scratch control architecture differs from frozen UTR-LM"
        )
    official_count = sum(parameter.numel() for parameter in official.parameters())
    control_count = sum(parameter.numel() for parameter in control.parameters())
    expected_count = int(config["model"]["num_parameters"])
    if official_count != expected_count:
        raise SmokeFailure(
            f"frozen bare-encoder parameter-count drift: {official_count} != {expected_count}"
        )
    if official_count != control_count:
        raise SmokeFailure(
            "from-scratch control parameter count differs from frozen UTR-LM"
        )
    if all(
        torch.equal(left.detach(), right.detach())
        for left, right in zip(official.parameters(), control.parameters())
    ):
        raise SmokeFailure(
            "from-scratch control unexpectedly equals checkpoint weights"
        )

    role_query_recorder = _FormalGpuRoleQueryRecorder(
        formal_binding,
        run_id=run_id,
        interface_records=interface_records,
        role_code_metadata=role_code_metadata,
    )
    _ACTIVE_ROLE_QUERY_RECORDER = role_query_recorder
    official_field, official_arm, official_forward_telemetry = (
        role_query_recorder.run_phase(
            "generator_rate_official_frozen_arm",
            lambda: _run_forced_action_arm(
                name="official_frozen_utrlm",
                foundation=official,
                tokenizer=official_tokenizer,
                device=device,
            ),
        )
    )
    control_field, control_arm, control_forward_telemetry = (
        role_query_recorder.run_phase(
            "generator_rate_from_scratch_control_arm",
            lambda: _run_forced_action_arm(
                name="same_architecture_from_scratch_control",
                foundation=control,
                tokenizer=control_tokenizer,
                device=device,
            ),
        )
    )

    paper_route = role_query_recorder.run_phase(
        "sampler_paper_official_foundation",
        lambda: _run_official_paper_sampler_route(official_field),
    )
    primary_sampler_integration = role_query_recorder.run_phase(
        "sampler_primary_official_foundation",
        lambda: _run_primary_gpu_sampler_integration(official_field),
    )
    leakage = role_query_recorder.run_phase(
        "rate_target_alignment_leakage_audit",
        lambda: _audit_target_alignment_leakage(official_field),
    )
    leakage["run_id"] = run_id
    leakage["goal_sha256"] = goal_sha256
    leakage["implementation_commit"] = implementation_commit
    leakage["source_binding_sha256"] = formal_binding["source_binding"][
        "tracked_source_files_sha256"
    ]
    dynamic_current = role_query_recorder.run_phase(
        "rate_dynamic_current_encoding_audit",
        lambda: _audit_dynamic_current_encoding(official_field),
    )
    post_gpu_role_query_audit = role_query_recorder.finalize()
    torch.cuda.synchronize(device)
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    current_memory = int(torch.cuda.memory_allocated(device))
    if peak_memory <= 0:
        raise SmokeFailure("CUDA peak-memory evidence is zero")

    placeholder_calls = int(paper_route["placeholder_foundation_forward_calls"])
    real_foundation_calls = (
        official_field.source_encode_calls
        + official_field.current_encode_calls
        + control_field.source_encode_calls
        + control_field.current_encode_calls
    )
    if real_foundation_calls <= 0:
        raise SmokeFailure("no real UTR-LM foundation forward was observed")
    expected_official_calls = (
        official_field.source_encode_calls + official_field.current_encode_calls
    )
    expected_control_calls = (
        control_field.source_encode_calls + control_field.current_encode_calls
    )
    if official_forward_telemetry["forward_calls"] != expected_official_calls:
        raise SmokeFailure("official UTR-LM forward telemetry/count mismatch")
    if control_forward_telemetry["forward_calls"] != expected_control_calls:
        raise SmokeFailure("control UTR-LM forward telemetry/count mismatch")
    if not (
        official_forward_telemetry["all_input_tensors_cuda"]
        and official_forward_telemetry["all_hidden_outputs_cuda_fp32"]
        and control_forward_telemetry["all_input_tensors_cuda"]
        and control_forward_telemetry["all_hidden_outputs_cuda_fp32"]
    ):
        raise SmokeFailure("a UTR-LM neural forward left CUDA FP32")
    paper_mode_real_calls = int(paper_route["official_foundation_forward_calls"])
    if paper_mode_real_calls <= 0:
        raise SmokeFailure("paper-mode official forward-call count is zero")
    if type(official_field.foundation).__name__ != "UtrLmModel":
        raise SmokeFailure("official foundation is not the real UtrLmModel class")

    foundation_audit = {
        "schema_version": "mk0_foundation_fusion_audit_v1",
        "run_id": run_id,
        "status": "PASS",
        "evidence_level": "E0_MATH_ENGINEERING_ONLY",
        "created_at_utc": _utc_now(),
        "implementation_binding": {
            "goal_sha256": goal_sha256,
            "implementation_commit": implementation_commit,
            "git_tree": formal_binding["source_binding"]["git_tree"],
            "tracked_source_files_sha256": formal_binding["source_binding"][
                "tracked_source_files_sha256"
            ],
            "run_manifest_sha256": formal_binding["run_manifest_sha256"],
            "preflight_sha256": formal_binding["preflight"]["sha256"],
            "repo_root": str(REPO_ROOT),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": _hash_file(Path(__file__).resolve()),
            "foundation_fusion_path": str(
                (REPO_ROOT / "core" / "mk0" / "foundation_fusion.py").resolve()
            ),
            "foundation_fusion_sha256": _hash_file(
                REPO_ROOT / "core" / "mk0" / "foundation_fusion.py"
            ),
        },
        "snapshot_binding": snapshot_binding,
        "cuda": {
            **cuda_identity,
            "cpu_fallback_allowed": False,
            "cpu_fallback_observed": False,
            "cuda_tensor_evidence": True,
            "memory_allocated_bytes_at_end": current_memory,
            "max_memory_allocated_bytes": peak_memory,
        },
        "numerical_policy": numerical_policy,
        "architecture_equivalence": {
            "same_architecture": True,
            "official_signature": official_signature,
            "from_scratch_signature": control_signature,
            "signature_sha256": _canonical_hash(official_signature),
            "frozen_expected_parameter_count": expected_count,
            "official_parameter_count": official_count,
            "from_scratch_parameter_count": control_count,
            "checkpoint_and_control_weights_differ": True,
        },
        "arms": {
            "official_frozen": official_arm,
            "same_architecture_from_scratch_control": control_arm,
        },
        "paper_sampler_official_foundation_route": paper_route,
        "primary_gpu_sampler_integration": primary_sampler_integration,
        "dynamic_current_encoding_audit": dynamic_current,
        "post_gpu_role_query_audit": post_gpu_role_query_audit,
        "full_reencode_accounting": {
            "incremental_update_enabled": False,
            "official_source_encode_calls": official_field.source_encode_calls,
            "official_current_encode_calls": official_field.current_encode_calls,
            "control_source_encode_calls": control_field.source_encode_calls,
            "control_current_encode_calls": control_field.current_encode_calls,
            "current_embedding_recomputed_after_each_forced_edit": True,
            "source_only_cache_reused_as_current": False,
        },
        "neural_forward_telemetry": {
            "official_frozen": official_forward_telemetry,
            "same_architecture_from_scratch_control": control_forward_telemetry,
        },
        "placeholder_audit": {
            "foundation_forward_mode": "real_utrlm_full_reencode",
            "real_utrlm_foundation_forward_calls": real_foundation_calls,
            "paper_mode_real_utrlm_forward_calls": paper_mode_real_calls,
            "placeholder_foundation_forward_calls": placeholder_calls,
            "placeholder_foundation_detected": False,
            "runtime_classified_by_official_adapter": True,
        },
        "gate_bindings": {
            "M31": {
                "name": "source_only_stale_state_encoding",
                **{
                    key: dynamic_current[key]
                    for key in (
                        "test_domain",
                        "coverage",
                        "sample_count",
                        "failure_count",
                        "failure_denominator",
                        "dtype",
                        "atol",
                        "rtol",
                    )
                },
                "exhaustive_or_sampled": dynamic_current["coverage"],
                "seed": SEED,
                "passed": dynamic_current["failure_count"] == 0,
                "metrics": {
                    "minimum_absolute_rate_delta": dynamic_current[
                        "minimum_absolute_rate_delta"
                    ],
                    "full_current_reencode_calls": dynamic_current[
                        "full_current_reencode_calls"
                    ],
                    "representation_oracle_failure_count": dynamic_current[
                        "representation_oracle_failure_count"
                    ],
                    "mutation_action_type_counts": dynamic_current[
                        "mutation_action_type_counts"
                    ],
                },
            },
            "M32": {
                "name": "paper_mode_placeholder_foundation",
                "test_domain": "all_paper_mode_foundation_forward_calls_in_gpu_smoke",
                "coverage": "exhaustive_log_audit",
                "exhaustive_or_sampled": "exhaustive_log_audit",
                "sample_count": paper_mode_real_calls,
                "failure_count": placeholder_calls,
                "failure_denominator": paper_mode_real_calls,
                "dtype": "discrete",
                "atol": 0.0,
                "rtol": 0.0,
                "seed": SEED,
                "passed": placeholder_calls == 0,
                "metrics": {
                    "paper_rate_calls": paper_route["paper_rate_calls"],
                    "official_foundation_forward_calls": paper_mode_real_calls,
                    "placeholder_foundation_forward_calls": placeholder_calls,
                    "paper_sampler_replay_passed": paper_route["sampler_replay_passed"],
                },
            },
            "M35": {
                "name": "gpu_tiny_smoke_forced_INS_SUB_DEL_STOP",
                "test_domain": "one_legal_nonzero_oracle_rate_case_per_action_with_forward_backward",
                "coverage": "forced_exhaustive_action_types",
                "exhaustive_or_sampled": "forced_exhaustive_action_types",
                "sample_count": 4,
                "failure_count": 0,
                "failure_denominator": 4,
                "dtype": "float32_cuda",
                "atol": FLOAT32_ATOL,
                "rtol": FLOAT32_RTOL,
                "seed": SEED,
                "passed": True,
                "metrics": {
                    "primary_sampler_integration": primary_sampler_integration,
                    "official_backward_action_types": official_arm[
                        "forced_action_types"
                    ],
                    "official_gradient_l1_norm": official_arm["gradient_evidence"][
                        "gradient_l1_norm"
                    ],
                },
            },
        },
        "claim_boundary": {
            "scope": "E0_math_and_engineering_GPU_smoke_only",
            "training_run": False,
            "development_labels_accessed": False,
            "final_labels_accessed": False,
            "functional_improvement_claim": False,
            "matched_budget_superiority_claim": False,
            "exact_gillespie_claim": False,
            "paper_success_claim": False,
        },
    }

    foundation_path = output_dir / FOUNDATION_FILENAME
    leakage_path = output_dir / LEAKAGE_FILENAME
    foundation_sha256 = _write_canonical_exclusive(foundation_path, foundation_audit)
    leakage_sha256 = _write_canonical_exclusive(leakage_path, leakage)
    artifact_hashes = {
        FOUNDATION_FILENAME: foundation_sha256,
        LEAKAGE_FILENAME: leakage_sha256,
    }
    post_role_query_binding = {
        "schema_version": "mk0_gpu_post_role_query_binding_v1",
        "qualifies_cpu_gate_id": "M34",
        "support_artifact": FOUNDATION_FILENAME,
        "support_artifact_sha256": foundation_sha256,
        "audit_sha256": _canonical_hash(post_gpu_role_query_audit),
        "record_stream_sha256": post_gpu_role_query_audit["record_stream_sha256"],
        "formal_gpu_phase_count": post_gpu_role_query_audit["formal_gpu_phase_count"],
        "all_role_query_counts_zero": post_gpu_role_query_audit[
            "all_role_query_counts_zero"
        ],
    }
    support_bindings = {
        "M05": leakage["gate_binding"],
        **foundation_audit["gate_bindings"],
    }
    gates_by_id = {gate["id"]: gate for gate in math_config["acceptance"]["gates"]}
    support_file_by_gate = {
        "M05": LEAKAGE_FILENAME,
        "M31": FOUNDATION_FILENAME,
        "M32": FOUNDATION_FILENAME,
        "M35": FOUNDATION_FILENAME,
    }
    gpu_gate_bindings: dict[str, dict[str, Any]] = {}
    for gate_id in ("M05", "M31", "M32", "M35"):
        gate_config = gates_by_id[gate_id]
        inner = support_bindings[gate_id]
        support_name = support_file_by_gate[gate_id]
        observed_metrics = dict(inner.get("metrics", {}))
        observed_metrics["support_gate_binding_sha256"] = hashlib.sha256(
            canonical_json_bytes(inner)
        ).hexdigest()
        gpu_gate_bindings[gate_id] = {
            "gate_id": gate_id,
            "name": gate_config["name"],
            "passed": bool(inner["passed"]),
            "test_domain": gate_config["domain"],
            "exhaustive_or_sampled": gate_config["coverage"],
            "sample_count": int(inner["sample_count"]),
            "dtype": str(gate_config["dtype"]),
            "atol": gate_config["atol"],
            "rtol": gate_config["rtol"],
            "seed": int(gate_config["seed"]),
            "failure_count": int(inner["failure_count"]),
            "failure_denominator": int(inner["failure_denominator"]),
            "artifact_path": gate_config["artifact_path"],
            "artifact_sha256": artifact_hashes[support_name],
            "metrics": observed_metrics,
        }
        gate_result_from_runtime_binding(
            gpu_gate_bindings[gate_id],
            gate_config,
            actual_artifact_sha256=artifact_hashes[support_name],
        )
    failed_gate_ids = [
        gate_id
        for gate_id, binding in gpu_gate_bindings.items()
        if not binding["passed"]
    ]
    gpu_results = {
        "schema_version": "mk0_gpu_gate_results_v1",
        "run_id": run_id,
        "status": "PASS_GPU_GATES" if not failed_gate_ids else "FAILED_WITH_EVIDENCE",
        "goal_sha256": goal_sha256,
        "implementation_commit": implementation_commit,
        "run_manifest_sha256": formal_binding["run_manifest_sha256"],
        "preflight": formal_binding["preflight"],
        "source_binding": formal_binding["source_binding"],
        "artifact_hashes": artifact_hashes,
        "post_gpu_role_query_audit": post_role_query_binding,
        "gate_bindings": gpu_gate_bindings,
        "failed_gate_ids": failed_gate_ids,
    }
    gpu_results_path = output_dir / GPU_RESULTS_FILENAME
    gpu_results_sha256 = _write_canonical_exclusive(gpu_results_path, gpu_results)
    if failed_gate_ids:
        raise SmokeFailure(f"GPU gate failures: {failed_gate_ids}")
    gpu_artifact_hashes = {
        **artifact_hashes,
        GPU_RESULTS_FILENAME: gpu_results_sha256,
    }
    gpu_summary = {
        "schema_version": "mk0_gpu_acceptance_summary_v1",
        "run_id": run_id,
        "status": "PASS_GPU_GATES",
        "evidence_level": "E0_MATH_ENGINEERING_ONLY",
        "created_at_utc": _utc_now(),
        "goal_sha256": goal_sha256,
        "implementation_commit": implementation_commit,
        "run_root": str(run_root),
        "run_manifest": {
            "path": formal_binding["run_manifest_path"],
            "sha256": formal_binding["run_manifest_sha256"],
        },
        "preflight": formal_binding["preflight"],
        "source_binding": formal_binding["source_binding"],
        "source_binding_sha256": _canonical_hash(formal_binding["source_binding"]),
        "gpu_gate_results": {
            "path": str(gpu_results_path.resolve()),
            "sha256": gpu_results_sha256,
        },
        "artifact_count": len(gpu_artifact_hashes),
        "artifact_hashes": gpu_artifact_hashes,
        "failed_gate_ids": [],
        "cuda": {
            "device_uuid": cuda_identity["device_uuid"],
            "max_memory_allocated_bytes": peak_memory,
            "cpu_fallback_observed": False,
        },
    }
    gpu_summary_sha256 = _write_canonical_atomic_exclusive(
        gpu_summary_path, gpu_summary
    )
    _ACTIVE_ROLE_QUERY_RECORDER = None
    return {
        "status": "PASS",
        "run_id": run_id,
        "foundation_fusion_audit": {
            "path": str(foundation_path.resolve()),
            "sha256": foundation_sha256,
        },
        "target_alignment_leakage_audit": {
            "path": str(leakage_path.resolve()),
            "sha256": leakage_sha256,
        },
        "gpu_gate_results": {
            "path": str(gpu_results_path.resolve()),
            "sha256": gpu_results_sha256,
        },
        "gpu_acceptance_summary": {
            "path": str(gpu_summary_path.resolve()),
            "sha256": gpu_summary_sha256,
        },
        "max_memory_allocated_bytes": peak_memory,
        "device_uuid": cuda_identity["device_uuid"],
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed MK0 real-UTR-LM CUDA smoke"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--goal-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--preflight-record", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--device", required=True)
    return parser.parse_args(argv)


def _validate_gpu_launch_contract(
    args: argparse.Namespace, raw_argv: list[str]
) -> tuple[Path, dict[str, Any]]:
    manifest_path = args.run_manifest.expanduser().resolve(strict=True)
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != "mk0_run_manifest_v3":
        raise SmokeFailure("formal GPU launch requires the section-19 v3 manifest")
    if manifest.get("run_id") != args.run_id:
        raise SmokeFailure("GPU launch/run manifest ID drift")
    if Path.cwd().resolve(strict=True) != REPO_ROOT:
        raise SmokeFailure("formal GPU smoke must be launched from the worktree")
    run_root = Path(str(manifest.get("run_root", ""))).resolve(strict=True)
    if manifest_path != run_root / "run_manifest.json":
        raise SmokeFailure("GPU launch manifest path/root drift")
    command = manifest.get("exact_commands", {}).get("gpu_smoke")
    if not isinstance(command, dict):
        raise SmokeFailure("GPU exact-command registration is absent")
    actual_argv = [sys.executable, str(Path(__file__).resolve()), *raw_argv]
    if command.get("argv") != actual_argv:
        raise SmokeFailure("actual GPU argv differs from the registered exact command")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if command.get("environment") != {"CUDA_VISIBLE_DEVICES": visible}:
        raise SmokeFailure("actual GPU environment differs from the registered command")
    if manifest.get("gpu_uuid") != visible:
        raise SmokeFailure("GPU launch UUID differs from the registered manifest")
    status = _load_json_object(run_root / "status.json")
    if status.get("state") != "CPU_VERIFIED_PENDING_GPU" or status.get("terminal"):
        raise SmokeFailure("GPU launch requires the nonterminal CPU-verified state")
    command_record = {
        "schema_version": "mk0_gpu_command_v1",
        "run_id": args.run_id,
        "created_at_utc": _utc_now(),
        "argv": actual_argv,
        "environment": {"CUDA_VISIBLE_DEVICES": visible},
        "pid": os.getpid(),
        "cwd": str(REPO_ROOT),
    }
    _write_canonical_atomic_exclusive(
        run_root / "provenance" / "gpu_command.json", command_record
    )
    append_event(
        run_root,
        "GPU_SMOKE_STARTED",
        run_id=args.run_id,
        pid=os.getpid(),
        gpu_uuid=visible,
    )
    append_text(
        run_root / "logs" / "stdout.log",
        f"{_utc_now()} GPU smoke started on {visible}\n",
    )
    return run_root, manifest


def _candidate_bound_run_root(args: argparse.Namespace) -> Path | None:
    """Find only the manifest-bound root safe for terminal failure evidence."""

    try:
        manifest_path = args.run_manifest.expanduser().resolve(strict=True)
        manifest = _load_json_object(manifest_path)
        run_root = Path(str(manifest.get("run_root", ""))).resolve(strict=True)
        output_dir = args.output_dir.expanduser().resolve()
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        manifest_path != run_root / "run_manifest.json"
        or output_dir != run_root / "artifacts" / "mk0"
        or manifest.get("schema_version") != "mk0_run_manifest_v3"
        or not isinstance(manifest.get("run_id"), str)
    ):
        return None
    return run_root


def main(argv: Iterable[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw_argv)
    run_root = _candidate_bound_run_root(args)
    bound_run_id: str | None = None
    if run_root is not None:
        try:
            bound_run_id = _load_json_object(run_root / "run_manifest.json").get(
                "run_id"
            )
        except BaseException:
            bound_run_id = None
    if run_root is not None and isinstance(bound_run_id, str):
        terminal_before = (run_root / "FAILED").exists() or (run_root / "DONE").exists()
        terminal = resume_failure_closure_if_present(run_root, run_id=bound_run_id)
        if terminal is not None:
            requested_matches = args.run_id == bound_run_id
            try:
                print(
                    json.dumps(
                        {
                            "status": (
                                f"ALREADY_{terminal}"
                                if requested_matches
                                else "TERMINAL_BOUND_TO_DIFFERENT_REQUEST"
                            ),
                            "run_id": bound_run_id,
                            "requested_run_id": args.run_id,
                            "tree_mutated": not terminal_before,
                            "tree_mutated_only_to_finish_failure_closure": (
                                not terminal_before and terminal == "FAILED"
                            ),
                        },
                        sort_keys=True,
                    )
                )
            except BaseException:
                pass
            if not requested_matches:
                return 2
            return 0 if terminal == "DONE" else 1
    try:
        validated_run_root, _ = _validate_gpu_launch_contract(args, raw_argv)
        if run_root is None or run_root != validated_run_root:
            raise SmokeFailure("safe GPU failure root differs from validated run root")
        summary = run_gpu_smoke(
            output_dir=args.output_dir,
            run_id=args.run_id,
            snapshot_dir=args.snapshot_dir,
            device_text=args.device,
            goal_sha256=args.goal_sha256,
            implementation_commit=args.implementation_commit,
            run_manifest_path=args.run_manifest,
            preflight_path=args.preflight_record,
        )
        append_jsonl(
            run_root / "logs" / "metrics.jsonl",
            {
                "created_at_utc": _utc_now(),
                "stage": "GPU_SMOKE",
                "status": "PASS_GPU_GATES",
                "gpu_uuid": summary["device_uuid"],
                "max_memory_allocated_bytes": summary["max_memory_allocated_bytes"],
                "cpu_fallback_count": 0,
                "forced_actions": ["INS", "SUB", "DEL", "STOP"],
            },
        )
        append_jsonl(
            run_root / "logs" / "system_metrics.jsonl",
            {
                "created_at_utc": _utc_now(),
                "event": "GPU_SMOKE_COMPLETED",
                "gpu_uuid": summary["device_uuid"],
                "max_memory_allocated_bytes": summary["max_memory_allocated_bytes"],
                "pid": os.getpid(),
            },
        )
        update_status(
            run_root,
            run_id=args.run_id,
            state="GPU_VERIFIED_PENDING_FINALIZER",
            terminal=False,
            stop_reason="GPU_GATES_PASSED_FINALIZER_NOT_STARTED",
        )
        append_text(
            run_root / "logs" / "stdout.log",
            f"{_utc_now()} GPU smoke passed; finalizer pending\n",
        )
    except BaseException as error:
        _attach_active_partial_phase_evidence(error)
        failure_reason = _standard_failure_reason(error)
        failure_path = None
        if run_root is not None:
            try:
                failure_path = _write_failure_best_effort(
                    args.output_dir,
                    run_id=bound_run_id or args.run_id,
                    snapshot_dir=str(args.snapshot_dir),
                    device=args.device,
                    error=error,
                )
            except BaseException:
                failure_path = None
        try:
            print(
                json.dumps(
                    {
                        "status": "FAILED_WITH_EVIDENCE",
                        "error": failure_reason,
                        "failure_artifact": (
                            str(failure_path) if failure_path else None
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        except BaseException:
            pass
        if run_root is not None:
            root_failure = run_root / "failure" / "gpu_smoke_failure.json"
            try:
                if not root_failure.exists():
                    _write_canonical_atomic_exclusive(
                        root_failure,
                        {
                            "schema_version": "mk0_gpu_smoke_failure_v2",
                            "run_id": bound_run_id,
                            "requested_run_id": args.run_id,
                            "status": "FAILED_WITH_EVIDENCE",
                            "created_at_utc": _utc_now(),
                            "exception_type": type(error).__name__,
                            "exception_message": failure_reason,
                            "traceback": traceback.format_exc(),
                            "partial_phase_evidence": getattr(
                                error, "partial_phase_evidence", None
                            ),
                            "support_failure_path": (
                                str(failure_path) if failure_path is not None else None
                            ),
                            "cpu_fallback_allowed": False,
                            "final_labels_accessed": False,
                        },
                    )
            except BaseException:
                pass
            try:
                append_text(
                    run_root / "logs" / "stderr.log",
                    f"{_utc_now()} GPU smoke failed: {failure_reason}\n",
                )
            except BaseException:
                pass
            try:
                write_failed_sentinel(
                    run_root,
                    run_id=bound_run_id or args.run_id,
                    stage="GPU_SMOKE",
                    reason=failure_reason,
                    exit_code=1,
                )
            except BaseException as closure_error:
                print(
                    f"GPU failure closure also failed: {closure_error}",
                    file=sys.stderr,
                )
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
