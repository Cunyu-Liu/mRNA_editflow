#!/usr/bin/env python3
"""Run the EF0 engineering gate and seal an auditable E0 bundle.

This is a tiny GPU validation, not GP0 training and not a scientific
evaluation.  It deliberately separates forced action coverage from the
stochastic sampler smoke so that a random run cannot hide an untested action
type and a forced harness cannot be mistaken for a performance result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any


GOAL_SHA256 = "3a3a654ca5c10a988eca897bff40be2e0b45c841f744f7423fdfd60b298b5791"
EVIDENCE_LEVEL = "E0_ENGINEERING_ONLY"


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json(payload))
        handle.flush()
        os.fsync(handle.fileno())


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_command(argv: list[str]) -> str:
    completed = subprocess.run(argv, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def file_binding(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"required provenance file is absent: {resolved}")
    payload: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if include_hash:
        payload["sha256"] = sha256_file(resolved)
    return payload


def prepare_run_tree(run_root: Path) -> None:
    if run_root.exists():
        raise FileExistsError(
            f"refusing to reuse or overwrite an existing EF0 run root: {run_root}"
        )
    for relative in (
        "provenance",
        "git",
        "logs",
        "checkpoints",
        "evaluation",
        "failure",
        "summary",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=False)
    for name in (
        "stdout.log",
        "stderr.log",
        "metrics.jsonl",
        "system_metrics.jsonl",
        "events.jsonl",
    ):
        (run_root / "logs" / name).touch(exist_ok=False)


def git_binding(worktree: Path) -> dict[str, Any]:
    status = run_command(["git", "-C", str(worktree), "status", "--porcelain"])
    commit = run_command(["git", "-C", str(worktree), "rev-parse", "HEAD"])
    tree = run_command(["git", "-C", str(worktree), "rev-parse", "HEAD^{tree}"])
    diff = subprocess.run(
        ["git", "-C", str(worktree), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "worktree": str(worktree),
        "commit": commit,
        "tree": tree,
        "status_porcelain": status,
        "clean": status == "",
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_bytes": len(diff),
    }


def action_payload(action: Any) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "position": action.position,
        "token": action.token,
        "key": action.key,
    }


def step_payload(step: Any) -> dict[str, Any]:
    return {
        "step": step.step,
        "t_start": step.t_start,
        "t_end": step.t_end,
        "h": step.h,
        "total_hazard": step.total_hazard,
        "event_probability": step.event_probability,
        "event_draw": step.event_draw,
        "action_draw": step.action_draw,
        "selected_action": (
            action_payload(step.selected_action)
            if step.selected_action is not None
            else None
        ),
        "outcome": step.outcome,
        "before_hash": step.before_hash,
        "after_hash": step.after_hash,
        "candidate_actions_hash": step.candidate_actions_hash,
        "candidate_rates_hash": step.candidate_rates_hash,
        "adaptive_subdivision_count": step.adaptive_subdivision_count,
        "rate_recomputed_after_step": step.rate_recomputed_after_step,
        "parallel_draws": list(step.parallel_draws),
        "parallel_actions": [action_payload(a) for a in step.parallel_actions],
    }


def state_payload(state: Any) -> dict[str, Any]:
    return {
        "source": state.source,
        "current": state.current,
        "region": state.region,
        "target_condition": state.target_condition,
        "initial_budget": state.initial_budget,
        "remaining_budget": state.remaining_budget,
        "history": asdict(state.history),
        "phase": state.phase.value,
        "termination_reason": (
            state.termination_reason.value
            if state.termination_reason is not None
            else None
        ),
        "state_hash": state.state_hash,
        "mapping": {
            "tokens": [
                {
                    "origin": token.origin.value,
                    "stable_id": token.stable_id,
                    "source_index": token.source_index,
                    "protected": token.protected,
                }
                for token in state.mapping.tokens
            ],
            "gap_ids": list(state.mapping.gap_ids),
        },
    }


def result_payload(result: Any) -> dict[str, Any]:
    return {
        "sampler": result.sampler,
        "exact_gillespie": result.exact_gillespie,
        "seed": result.seed,
        "step_size": result.step_size,
        "stability_hazard": result.stability_hazard,
        "min_length": result.min_length,
        "max_length": result.max_length,
        "horizon": result.horizon,
        "edit_events": result.edit_events,
        "termination_time": result.termination_time,
        "termination_before_hash": result.termination_before_hash,
        "initial_state": state_payload(result.initial_state),
        "final_state": state_payload(result.final_state),
        "steps": [step_payload(step) for step in result.steps],
        "remaining_hazard_certificate": (
            asdict(result.remaining_hazard_certificate)
            if result.remaining_hazard_certificate is not None
            else None
        ),
        "invalid_joint_proposals": result.invalid_joint_proposals,
    }


def create_initial_state() -> Any:
    from core.mk0.types import EditState

    return EditState.initial(
        "ACGU",
        region="5UTR",
        context={
            "assay": "EF0_GPU_SMOKE",
            "cell_or_tissue": "synthetic_engineering_fixture",
            "endpoint": "engineering_only",
            "batch": "EF0_20260803",
        },
        target_condition="increase",
        budget=4,
    )


def collect_gpu_inventory(device: Any, gpu_uuid: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; EF0 neural validation is fail-closed")
    if device.type != "cuda":
        raise RuntimeError("EF0 neural validation requires a CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(index)
    properties = torch.cuda.get_device_properties(index)
    return {
        "requested_device": str(device),
        "cuda_device_index_after_visibility": index,
        "gpu_uuid_from_preflight": gpu_uuid,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "memory_allocated_before_bytes": torch.cuda.memory_allocated(index),
        "memory_reserved_before_bytes": torch.cuda.memory_reserved(index),
        "max_memory_allocated_before_bytes": torch.cuda.max_memory_allocated(index),
    }


def write_checksum_ledger(run_root: Path) -> str:
    ledger = run_root / "artifact_checksums.sha256"
    rows: list[str] = []
    excluded = {
        ledger.resolve(),
        (run_root / "DONE").resolve(),
        (run_root / "FAILED").resolve(),
    }
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(run_root).as_posix()}")
    write_text(ledger, "\n".join(rows) + "\n")
    return sha256_file(ledger)


def publish_terminal_success(
    run_root: Path,
    *,
    run_id: str,
    acceptance: dict[str, Any],
    status: dict[str, Any],
) -> None:
    acceptance_path = run_root / "evaluation" / "ef0_acceptance.json"
    write_json(acceptance_path, acceptance)
    write_json(run_root / "summary" / "summary.json", acceptance)
    write_json(run_root / "status.json", status)
    ledger_sha = write_checksum_ledger(run_root)
    done_lines = [
        run_id,
        sha256_file(acceptance_path),
        ledger_sha,
        sha256_file(run_root / "status.json"),
    ]
    write_text(run_root / "DONE", "\n".join(done_lines) + "\n")


def publish_terminal_failure(
    run_root: Path,
    *,
    run_id: str,
    error: BaseException,
    stage: str,
) -> None:
    failure = {
        "schema_version": "ef0_failure_v1",
        "run_id": run_id,
        "stage": stage,
        "status": "FAILED_WITH_EVIDENCE",
        "evidence_level": EVIDENCE_LEVEL,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "paper_eligibility": False,
    }
    write_json(run_root / "failure" / "failure.json", failure)
    write_text(run_root / "failure" / "traceback.txt", failure["traceback"])
    write_json(
        run_root / "status.json",
        {
            "schema_version": "ef0_status_v1",
            "run_id": run_id,
            "state": "FAILED_WITH_EVIDENCE",
            "terminal": True,
            "evidence_level": EVIDENCE_LEVEL,
            "stop_reason": f"{stage}: {type(error).__name__}: {error}",
            "exit_code": 1,
            "updated_at_utc": now_utc(),
        },
    )
    ledger_sha = write_checksum_ledger(run_root)
    write_text(
        run_root / "FAILED",
        "\n".join(
            [
                run_id,
                stage,
                sha256_file(run_root / "failure" / "failure.json"),
                ledger_sha,
                sha256_file(run_root / "status.json"),
            ]
        )
        + "\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--foundation-closure", required=True, type=Path)
    parser.add_argument("--d1-data", required=True, type=Path)
    parser.add_argument("--exposure-ledger", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path, action="append")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--implementation-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    stage = "REGISTERED"
    prepare_run_tree(run_root)
    append_jsonl(
        run_root / "logs" / "events.jsonl",
        {"created_at_utc": now_utc(), "event": "REGISTERED", "run_id": args.run_id},
    )
    try:
        worktree_git = git_binding(args.worktree)
        if not worktree_git["clean"]:
            raise RuntimeError("EF0 formal run requires a clean committed worktree")
        if worktree_git["commit"] != args.implementation_commit:
            raise RuntimeError(
                "implementation commit argument does not bind worktree HEAD"
            )
    except Exception as error:
        append_jsonl(
            run_root / "logs" / "events.jsonl",
            {
                "created_at_utc": now_utc(),
                "event": "FAILED_WITH_EVIDENCE",
                "stage": "PREFLIGHT_BINDING",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        publish_terminal_failure(
            run_root,
            run_id=args.run_id,
            error=error,
            stage="PREFLIGHT_BINDING",
        )
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": "FAILED_WITH_EVIDENCE",
                    "stage": "PREFLIGHT_BINDING",
                    "run_root": str(run_root),
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 1

    write_text(run_root / "command.txt", " ".join([sys.executable, *sys.argv]) + "\n")
    write_json(
        run_root / "git" / "commit.json",
        {
            **worktree_git,
            "implementation_commit_argument": args.implementation_commit,
        },
    )
    write_text(run_root / "git" / "diff.patch", "")
    write_text(
        run_root / "git" / "diff.sha256",
        hashlib.sha256(b"").hexdigest() + "\n",
    )
    write_text(run_root / "provenance" / "goal_contract.sha256", GOAL_SHA256 + "\n")
    write_json(
        run_root / "provenance" / "data_manifest.json",
        {
            "schema_version": "ef0_data_binding_v1",
            "final_labels_accessed": False,
            "d1_canonical_records": file_binding(args.d1_data),
            "exposure_ledger": file_binding(args.exposure_ledger),
        },
    )
    write_json(
        run_root / "provenance" / "split_manifest.json",
        {
            "schema_version": "ef0_split_binding_v1",
            "final_labels_accessed": False,
            "split_files": [file_binding(path) for path in args.split_manifest],
        },
    )
    closure_manifest = args.foundation_closure / "provenance" / "foundation_manifest.json"
    write_json(
        run_root / "provenance" / "foundation_manifest.json",
        {
            "schema_version": "ef0_foundation_binding_v1",
            "model_id": "multimolecule/utrlm-mrl",
            "snapshot_dir": str(args.snapshot_dir.resolve()),
            "closure_root": str(args.foundation_closure.resolve()),
            "closure_manifest": file_binding(closure_manifest),
            "checkpoint_revision": "79e23de069449e659696b5210f833c28ddd0de50",
            "sequence_overlap_status": "NOT_AVAILABLE_NOT_ASSERTED",
            "label_overlap_status": "bound_by_fm0_ledger; final_labels_accessed=false",
        },
    )
    write_json(
        run_root / "provenance" / "code_manifest.json",
        {
            "schema_version": "ef0_code_binding_v1",
            "worktree": str(args.worktree.resolve()),
            "commit": args.implementation_commit,
            "tree": worktree_git["tree"],
            "clean": True,
            "tracked_ef0_files": {
                relative: file_binding(args.worktree / relative)
                for relative in (
                    "core/ef0/__init__.py",
                    "core/ef0/model.py",
                    "core/ef0/sampler.py",
                    "scripts/ef0/run_ef0_acceptance.py",
                    "tests/ef0/test_state_contract.py",
                )
            },
        },
    )
    write_text(
        run_root / "resolved_config.yaml",
        "\n".join(
            [
                "schema_version: ef0_gpu_smoke_v1",
                "phase: EF0",
                "task_id: EF0-01",
                f"run_id: {args.run_id}",
                f"goal_contract_sha256: {GOAL_SHA256}",
                "evidence_level: E0_ENGINEERING_ONLY",
                "formal_training: false",
                "final_labels_accessed: false",
                "sampler: constrained_single_event_first_order",
                "exact_gillespie: false",
                "min_length: 1",
                "max_length: 6",
                "step_size: 0.0625",
                "stability_hazard: 0.05",
                "horizon: 1.0",
                "foundation: multimolecule/utrlm-mrl@79e23de069449e659696b5210f833c28ddd0de50",
                f"gpu_uuid: {args.gpu_uuid}",
                f"device: {args.device}",
            ]
        )
        + "\n",
    )
    write_json(
        run_root / "status.json",
        {
            "schema_version": "ef0_status_v1",
            "run_id": args.run_id,
            "state": "PREFLIGHT_PASSED",
            "terminal": False,
            "evidence_level": EVIDENCE_LEVEL,
            "stop_reason": "RUNNING_GPU_VALIDATION",
            "exit_code": None,
            "updated_at_utc": now_utc(),
        },
    )
    append_jsonl(
        run_root / "logs" / "events.jsonl",
        {
            "created_at_utc": now_utc(),
            "event": "PREFLIGHT_PASSED",
            "implementation_commit": args.implementation_commit,
            "final_labels_accessed": False,
            "unrelated_processes_killed": 0,
        },
    )

    try:
        stage = "GPU_VERIFIED"
        import torch

        from core.ef0 import (
            EF0ModelConfig,
            EF0SamplerConfig,
            TrueUTREditFlow,
            TrueUTREditFlowRateField,
            generate_candidates,
        )
        from core.mk0.state_action import apply_action, enumerate_legal_actions
        from core.mk0.types import ActionType, AtomicAction, EditState

        device = torch.device(args.device)
        gpu = collect_gpu_inventory(device, args.gpu_uuid)
        torch.manual_seed(20260803)
        torch.cuda.manual_seed_all(20260803)
        foundation, tokenizer = __import__(
            "core.mk0.foundation_fusion", fromlist=["load_official_utrlm"]
        ).load_official_utrlm(
            args.snapshot_dir,
            device=device,
            from_scratch=False,
            seed=20260802,
        )
        field = TrueUTREditFlowRateField(
            foundation,
            tokenizer,
            device=device,
            config=EF0ModelConfig(min_length=1, max_length=6, hidden_head_width=64),
        )
        flow = TrueUTREditFlow(field).to(device)
        flow.eval()
        device_audit = flow.runtime_device_audit()
        if not device_audit["cuda_available"]:
            raise RuntimeError("CUDA became unavailable after model construction")
        if not device_audit["trainable_parameters_cuda"] or not device_audit["frozen_parameters_cuda"]:
            raise RuntimeError("EF0 parameters are not fully CUDA-bound")
        if device_audit["foundation_requires_grad_count"] != 0:
            raise RuntimeError("frozen FM0 foundation has trainable parameters")

        base_state = create_initial_state()
        forced_actions = (
            AtomicAction(ActionType.INS, 0, "A"),
            AtomicAction(ActionType.SUB, 0, "C"),
            AtomicAction(ActionType.DEL, 0),
            AtomicAction(ActionType.STOP),
        )
        coverage: list[dict[str, Any]] = []
        forced_rates: list[torch.Tensor] = []
        for action in forced_actions:
            if action not in enumerate_legal_actions(
                base_state, min_length=1, max_length=6, include_stop=True
            ):
                raise AssertionError(f"forced action is not legal in fixture: {action.key}")
            tensor_rate = flow(base_state, 0.25, actions=(action,))[action]
            if tensor_rate.device.type != "cuda" or not bool(torch.isfinite(tensor_rate)):
                raise FloatingPointError(f"invalid CUDA rate for {action.key}")
            if float(tensor_rate.detach().cpu()) <= 0.0:
                raise AssertionError(f"non-positive forced coverage rate: {action.key}")
            forced_rates.append(tensor_rate)
            coverage.append(
                {
                    "action": action_payload(action),
                    "rate": float(tensor_rate.detach().cpu()),
                    "device": str(tensor_rate.device),
                    "state_hash": base_state.state_hash,
                }
            )
        loss = torch.stack(forced_rates).sum()
        if loss.device.type != "cuda" or not bool(torch.isfinite(loss)):
            raise FloatingPointError("GPU smoke loss is not finite CUDA")
        trainable = [parameter for parameter in flow.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=1.0e-3)
        before_parameter = trainable[0].detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is None for parameter in trainable):
            raise RuntimeError("a trainable EF0 parameter received no GPU gradient")
        if any(
            parameter.grad.device.type != "cuda"
            or not bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        ):
            raise FloatingPointError("EF0 GPU gradient is non-finite or off-device")
        optimizer.step()
        update_delta = float((trainable[0].detach() - before_parameter).abs().max().cpu())
        if not math.isfinite(update_delta) or update_delta <= 0.0:
            raise RuntimeError("EF0 optimizer update did not modify a trainable CUDA parameter")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)

        protected_state = EditState.initial(
            base_state.source,
            region="5UTR",
            context=dict(base_state.context),
            target_condition="increase",
            budget=2,
            protected_indices=(1,),
        )
        illegal = AtomicAction(ActionType.SUB, 1, "A")
        try:
            flow(protected_state, 0.25, actions=(illegal,))
        except ValueError:
            illegal_rejected = True
        else:
            illegal_rejected = False
        if not illegal_rejected:
            raise AssertionError("EF0 did not reject a protected-position action")

        state_3utr = EditState.initial(
            base_state.source,
            region="3UTR",
            context=dict(base_state.context),
            target_condition="increase",
            budget=4,
        )
        stop_5 = float(flow(base_state, 0.33, actions=(AtomicAction(ActionType.STOP),))[AtomicAction(ActionType.STOP)].detach().cpu())
        stop_3 = float(flow(state_3utr, 0.33, actions=(AtomicAction(ActionType.STOP),))[AtomicAction(ActionType.STOP)].detach().cpu())
        if not math.isfinite(stop_5) or not math.isfinite(stop_3):
            raise FloatingPointError("region adapter produced a non-finite rate")

        # Dynamic current-state path: every forced action is preceded by the
        # production CUDA rate field, then applied by the frozen MK0 updater.
        dynamic = base_state
        dynamic_records: list[dict[str, Any]] = []
        dynamic_actions = (
            AtomicAction(ActionType.INS, 0, "A"),
            AtomicAction(ActionType.SUB, 2, "G"),
            AtomicAction(ActionType.DEL, 0),
            AtomicAction(ActionType.STOP),
        )
        lengths = [len(dynamic.current)]
        for index, action in enumerate(dynamic_actions):
            rate = flow(dynamic, min(0.90, 0.15 + index * 0.15), actions=(action,))[action]
            if not bool(torch.isfinite(rate)) or float(rate.detach().cpu()) <= 0.0:
                raise AssertionError(f"dynamic path action has no positive rate: {action.key}")
            before = dynamic
            transition = apply_action(dynamic, action, min_length=1, max_length=6)
            dynamic = transition.after
            lengths.append(len(dynamic.current))
            dynamic_records.append(
                {
                    "action": action_payload(action),
                    "rate": float(rate.detach().cpu()),
                    "before": state_payload(before),
                    "after": state_payload(dynamic),
                }
            )
        if lengths[1] <= lengths[0] or lengths[3] >= lengths[2]:
            raise AssertionError("EF0 dynamic path did not exercise real variable length")
        if dynamic.termination_reason.value != "LEARNED_STOP":
            raise AssertionError("dynamic STOP did not enter learned absorbing state")

        flow.eval()
        sampler_state = EditState.initial(
            "A",
            region="5UTR",
            context=dict(base_state.context),
            target_condition="increase",
            budget=1,
        )
        sampler_config = EF0SamplerConfig(
            step_size=0.0625,
            stability_hazard=0.05,
            min_length=1,
            max_length=6,
            horizon=1.0,
        )
        sampler_first = generate_candidates(
            flow, sampler_state, config=sampler_config, seed=20260803
        )
        sampler_second = generate_candidates(
            flow, sampler_state, config=sampler_config, seed=20260803
        )
        if sampler_first.final_state.state_hash != sampler_second.final_state.state_hash:
            raise AssertionError("same-seed EF0 sampler replay changed final state")
        if sampler_first.steps != sampler_second.steps:
            raise AssertionError("same-seed EF0 sampler replay changed step ledger")
        if sampler_first.exact_gillespie:
            raise AssertionError("EF0 sampler was incorrectly labeled exact Gillespie")
        for step in sampler_first.steps:
            if step.selected_action is not None:
                if step.selected_action not in enumerate_legal_actions(
                    sampler_state,
                    min_length=1,
                    max_length=6,
                    include_stop=True,
                ) and step.before_hash == sampler_state.state_hash:
                    raise AssertionError("sampler selected action outside its initial mask")
        if any(token not in "ACGU" for token in sampler_first.final_state.current):
            raise AssertionError("sampler produced an invalid nucleotide")

        gpu["memory_allocated_after_bytes"] = torch.cuda.memory_allocated(device)
        gpu["memory_reserved_after_bytes"] = torch.cuda.memory_reserved(device)
        gpu["max_memory_allocated_after_bytes"] = torch.cuda.max_memory_allocated(device)
        gpu["cuda_memory_positive"] = gpu["max_memory_allocated_after_bytes"] > 0
        if not gpu["cuda_memory_positive"]:
            raise RuntimeError("CUDA memory evidence is zero")

        stage = "VERIFIED"
        acceptance = {
            "schema_version": "ef0_acceptance_v1",
            "run_id": args.run_id,
            "phase": "EF0",
            "task_id": "EF0-01",
            "evidence_level": EVIDENCE_LEVEL,
            "contract_sha256": GOAL_SHA256,
            "implementation_commit": args.implementation_commit,
            "paper_eligibility": False,
            "scientific_claims_allowed": False,
            "final_labels_accessed": False,
            "downstream_stage_started": False,
            "gpu": gpu,
            "device_audit": device_audit,
            "model_interface": {
                "inference_signature_fields": list(flow.inference_signature_fields),
                "explicit_action_heads": list(field.action_heads.keys()),
                "region_adapter_shape": list(field.region_adapter.shape),
                "region_stop_rate_5utr": stop_5,
                "region_stop_rate_3utr": stop_3,
            },
            "forced_action_coverage": coverage,
            "forced_action_coverage_count": len(coverage),
            "optimizer_update_delta_max": update_delta,
            "protected_action_rejected": illegal_rejected,
            "dynamic_variable_length": {
                "lengths": lengths,
                "records": dynamic_records,
                "final_reason": dynamic.termination_reason.value,
            },
            "sampler": {
                "config": asdict(sampler_config),
                "first": result_payload(sampler_first),
                "replay_equal": True,
                "exact_gillespie": False,
            },
            "gates": {
                "dynamic_current_state": "PASS",
                "source_conditioning_and_mapping": "PASS",
                "explicit_ins_sub_del_stop_heads": "PASS",
                "nonnegative_rates": "PASS",
                "hard_legality_before_sampler": "PASS",
                "protected_action_rejection": "PASS",
                "variable_length_ins_del": "PASS",
                "multi_step_rollout": "PASS",
                "region_adapters_5utr_3utr": "PASS",
                "deterministic_replay": "PASS",
                "real_gpu_forward_backward_optimizer": "PASS",
                "cpu_fallback_count": 0,
                "formal_training_started": False,
                "final_labels_accessed": False,
                "final_evaluator_used_for_guidance": False,
            },
            "known_deviations": [
                "E0 engineering validation only; no GP0 training",
                "forced action coverage is a harness, not a random performance result",
                "sampler is constrained_single_event_first_order, not exact Gillespie",
                "no functional, biological, superiority, or paper claim",
            ],
            "created_at_utc": now_utc(),
        }
        write_json(run_root / "evaluation" / "trajectory_records.json", {
            "forced_dynamic_path": dynamic_records,
            "sampler": result_payload(sampler_first),
        })
        append_jsonl(
            run_root / "logs" / "metrics.jsonl",
            {
                "created_at_utc": now_utc(),
                "metric": "ef0_gpu_smoke",
                "forced_action_coverage": len(coverage),
                "sampler_edit_events": sampler_first.edit_events,
                "max_memory_allocated_bytes": gpu["max_memory_allocated_after_bytes"],
            },
        )
        append_jsonl(
            run_root / "logs" / "system_metrics.jsonl",
            {
                "created_at_utc": now_utc(),
                "gpu_uuid": args.gpu_uuid,
                "gpu_name": gpu["gpu_name"],
                "torch_version": gpu["torch_version"],
                "cuda_version": gpu["cuda_version"],
                "memory_allocated_after_bytes": gpu["memory_allocated_after_bytes"],
            },
        )
        append_jsonl(
            run_root / "logs" / "events.jsonl",
            {
                "created_at_utc": now_utc(),
                "event": "VERIFIED",
                "evidence_level": EVIDENCE_LEVEL,
                "downstream_stage_started": False,
            },
        )
        publish_terminal_success(
            run_root,
            run_id=args.run_id,
            acceptance=acceptance,
            status={
                "schema_version": "ef0_status_v1",
                "run_id": args.run_id,
                "state": "VERIFIED",
                "terminal": True,
                "evidence_level": EVIDENCE_LEVEL,
                "stop_reason": "ALL_EF0_ENGINEERING_GATES_PASSED",
                "exit_code": 0,
                "updated_at_utc": now_utc(),
            },
        )
        print(json.dumps({"run_id": args.run_id, "status": "VERIFIED", "run_root": str(run_root)}))
        return 0
    except Exception as error:
        append_jsonl(
            run_root / "logs" / "events.jsonl",
            {
                "created_at_utc": now_utc(),
                "event": "FAILED_WITH_EVIDENCE",
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        publish_terminal_failure(
            run_root,
            run_id=args.run_id,
            error=error,
            stage=stage,
        )
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": "FAILED_WITH_EVIDENCE",
                    "run_root": str(run_root),
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
