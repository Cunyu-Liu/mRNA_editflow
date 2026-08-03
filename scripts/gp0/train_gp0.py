#!/usr/bin/env python3
"""GPU-only GP0 trainer with contract-bound data and fail-closed gates.

The command has two deliberately different modes:

* ``development`` requires an explicit small record cap and produces only
  development evidence;
* ``formal`` refuses to start unless the current D1/B0/exposure/resource
  bindings are complete.  In particular it cannot turn the stale 130,047-row
  B0 card into a formal run against the current 134,059 paired D1 records.

No command in this file writes a ``DONE`` marker.  GP0 acceptance requires
held-out evaluation and the contract's five-seed aggregation/finalizer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    CONTRACT_ID,
    CONTRACT_SHA256,
    EXPECTED_D1_ACCESSIONS,
    EXPECTED_D1_PAIRED,
    PHASE_ID,
    SCIENTIFIC_QUESTION_ID,
    TASK_ID,
    GP0GateError,
    PairedRecord,
    append_jsonl,
    artifact_checksums,
    build_rate_field,
    file_binding,
    load_b0_binding,
    load_exposure_policy,
    load_split_binding,
    require_cuda_device,
    runtime_model_audit,
    scan_d1_and_select,
    sha256_file,
    validate_formal_data_binding,
    write_json,
    make_training_example,
    assert_cuda_rates,
)


VARIANTS = (
    "from-scratch",
    "frozen-foundation",
    "lora-adapter",
    "no-source",
    "no-time",
    "no-indel",
    "no-STOP",
    "fixed-length",
    "no-region-adapter",
)


def _run(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except OSError as error:
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(error)}


def _git_binding(repo: Path) -> dict[str, Any]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    status = _run(["git", "status", "--short", "--branch"], cwd=repo)
    return {
        "head": head["stdout"].strip(),
        "head_returncode": head["returncode"],
        "status": status["stdout"],
        "status_returncode": status["returncode"],
    }


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:7]


def _new_run_dir(base: Path, variant: str, seed: int, git_head: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"GP0_{variant.replace('-', '_')}_{stamp}_{_short_hash(git_head)}_s{seed}"
    path = base / run_id
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return path


def _public_split_binding(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": binding["schema_version"],
        "paths": binding["paths"],
        "combined_sha256": binding["combined_sha256"],
        "record_count": binding["record_count"],
        "split_counts": binding["split_counts"],
    }


def _base_manifest(args: argparse.Namespace, run_dir: Path, git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "gp0_run_manifest_v1",
        "status": "STARTING",
        "run_id": run_dir.name,
        "parent_run_id": args.parent_run_id,
        "mode": args.mode,
        "variant": args.variant,
        "goal_contract": {"id": CONTRACT_ID, "sha256": CONTRACT_SHA256},
        "scientific_question_id": SCIENTIFIC_QUESTION_ID,
        "phase_id": PHASE_ID,
        "task_id": TASK_ID,
        "git_commit": git["head"],
        "git_binding": git,
        "formal_seed_requirement": {"required_seed_count": 5, "current_seed": args.seed},
        "scientific_scope": {
            "target": "source-conditioned legal variable-length UTR edit distribution before function conditioning",
            "target_condition_policy": "fixed maintain; labels and measured effects are not read",
            "regions": ["5UTR", "3UTR"],
            "evidence_boundary": "development or formal computational training only; no biological conclusion",
        },
        "hyperparameters": {
            "seed": args.seed,
            "epochs": args.epochs,
            "max_records": args.max_records,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "log_every": args.log_every,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "hidden_head_width": args.hidden_head_width,
            "adapter_rank": args.adapter_rank,
            "time_policy": args.time_policy,
            "sequence_alphabet_policy": args.sequence_alphabet_policy,
        },
        "inputs": {
            "data_manifest_sha256": None,
            "split_manifest_sha256": None,
            "foundation_checkpoint": str(args.foundation_snapshot),
            "foundation_checkpoint_sha256": args.foundation_checkpoint_sha256,
            "exposure_ledger_version": "data_exposure_ledger.jsonl",
        },
        "resource_request": {
            "device": args.device,
            "gpu_uuid": args.gpu_uuid,
            "resource_allocation_id": args.resource_allocation_id,
            "allow_shared_gpu": bool(args.allow_shared_gpu),
        },
        "artifacts": {"run_dir": str(run_dir)},
        "gate_blockers": [],
    }


def _finalize(run_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "status.json", {"status": manifest["status"], "run_id": manifest["run_id"]})
    (run_dir / "artifact_checksums.sha256").write_text(
        artifact_checksums(run_dir), encoding="utf-8"
    )


def _fail(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    status: str,
    blockers: list[dict[str, Any]],
    error: BaseException | None = None,
) -> int:
    manifest["status"] = status
    manifest["gate_blockers"] = blockers
    if error is not None:
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
    write_json(run_dir / "failure_evidence.json", {
        "schema_version": "gp0_failure_evidence_v1",
        "status": status,
        "blockers": blockers,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
        "done_marker_written": False,
    })
    _finalize(run_dir, manifest)
    return 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "formal"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256")
    parser.add_argument("--exposure-ledger", type=Path, required=True)
    parser.add_argument("--exposure-ledger-sha256")
    parser.add_argument("--split-manifest", type=Path, action="append", required=True)
    parser.add_argument("--split-role", choices=("train", "val", "test"), default="train")
    parser.add_argument("--b0-card", type=Path)
    parser.add_argument("--b0-split-summary", type=Path)
    parser.add_argument("--foundation-manifest", type=Path, required=True)
    parser.add_argument("--foundation-snapshot", type=Path, required=True)
    parser.add_argument("--foundation-checkpoint-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--resource-allocation-id")
    parser.add_argument("--parent-run-id")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="bind data/policy/split metadata and stop before model/GPU initialization",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--hidden-head-width", type=int, default=128)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--time-policy", choices=("initial_only", "stochastic"), default="stochastic")
    parser.add_argument(
        "--sequence-alphabet-policy",
        choices=("strict_rna", "dna_t_to_rna_u"),
        default="dna_t_to_rna_u",
    )
    return parser.parse_args(argv)


def _check_cli(args: argparse.Namespace) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if args.seed < 0:
        blockers.append({"id": "SEED_INVALID"})
    if args.epochs < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        blockers.append({"id": "HYPERPARAMETER_INVALID"})
    if args.min_length < 1 or args.max_length < args.min_length:
        blockers.append({"id": "LENGTH_BOUND_INVALID"})
    if args.log_every < 1:
        blockers.append({"id": "MONITOR_INTERVAL_INVALID"})
    if args.mode == "development":
        if args.max_records is None or args.max_records < 1:
            blockers.append({"id": "DEVELOPMENT_MAX_RECORDS_REQUIRED"})
        if args.max_records is not None and args.max_records > 128:
            blockers.append({"id": "DEVELOPMENT_MAX_RECORDS_TOO_LARGE", "max": 128})
    else:
        if args.max_records is not None or args.max_steps is not None:
            blockers.append({"id": "FORMAL_CAP_NOT_ALLOWED"})
        if not args.b0_card or not args.b0_split_summary:
            blockers.append({"id": "FORMAL_B0_BINDING_REQUIRED"})
        if not args.resource_allocation_id or not args.gpu_uuid:
            blockers.append({"id": "FORMAL_GPU_ALLOCATION_BINDING_REQUIRED"})
    if args.max_steps is not None and args.max_steps < 1:
        blockers.append({"id": "MAX_STEPS_INVALID"})
    return blockers


def _prepare_variant_records(args: argparse.Namespace, records: list[PairedRecord]) -> tuple[list[PairedRecord], dict[str, Any]]:
    if args.variant not in {"no-indel", "fixed-length"}:
        return records, {"input_records": len(records), "kept_records": len(records), "dropped": {}}
    kept: list[PairedRecord] = []
    dropped = 0
    for record in records:
        if len(record.source) == len(record.candidate):
            kept.append(record)
        else:
            dropped += 1
    return kept, {
        "input_records": len(records),
        "kept_records": len(kept),
        "dropped": {"length_changing_pairs": dropped},
        "filter": "same source/candidate length is required for no-indel/fixed-length",
    }


def _check_formal_variant(records: list[PairedRecord], args: argparse.Namespace) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if args.variant in {"no-indel", "fixed-length"}:
        changing = sum(len(row.source) != len(row.candidate) for row in records)
        if changing:
            blockers.append({"id": "VARIANT_LENGTH_POLICY_CONFLICT", "length_changing_records": changing})
    return blockers


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _objective_uniform_loss(example: Any, *, min_length: int, max_length: int, device: Any) -> float:
    import torch

    from mrna_editflow.core.mk0.bregman import edit_flow_loss
    from mrna_editflow.core.mk0.state_action import enumerate_legal_actions
    from mrna_editflow.core.mk0.types import ActionType

    legal = enumerate_legal_actions(
        example.state,
        min_length=min_length,
        max_length=max_length,
        include_stop=False,
    )
    rates = {action: torch.ones((), device=device, dtype=torch.float32) for action in legal}
    loss = edit_flow_loss(
        example.state,
        rates,
        example.oracle,
        min_length=min_length,
        max_length=max_length,
    )
    if isinstance(loss, torch.Tensor):
        return float(loss.detach().cpu().item())
    return float(loss)


def _probe_optimizer(field: Any, example: Any, optimizer: Any, *, min_length: int, max_length: int) -> dict[str, Any]:
    import torch

    assert_cuda_rates(field(example.state, example.time))
    optimizer.zero_grad(set_to_none=True)
    rates = field(example.state, example.time)
    probe = sum(rates.values())
    if not isinstance(probe, torch.Tensor) or probe.device.type != "cuda":
        raise GP0GateError("GP0 probe loss did not stay on CUDA")
    if not bool(torch.isfinite(probe)):
        raise GP0GateError("GP0 probe loss is NaN/Inf")
    probe.backward()
    grad_norm_sq = 0.0
    finite_grads = True
    for parameter in field.parameters():
        if parameter.grad is not None:
            finite_grads = finite_grads and bool(torch.isfinite(parameter.grad).all())
            grad_norm_sq += float(parameter.grad.detach().float().pow(2).sum().cpu())
    if not finite_grads or not math.isfinite(grad_norm_sq):
        raise GP0GateError("GP0 probe backward produced non-finite gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "probe_loss": float(probe.detach().cpu().item()),
        "probe_grad_l2": math.sqrt(grad_norm_sq),
        "probe_optimizer_step": True,
    }


def train(args: argparse.Namespace) -> int:
    cli_blockers = _check_cli(args)
    git = _git_binding(args.repo)
    run_dir = _new_run_dir(args.run_root, args.variant, args.seed, git["head"] or "unknown")
    manifest = _base_manifest(args, run_dir, git)
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "status.json", {"status": "STARTING", "run_id": run_dir.name})
    if cli_blockers:
        return _fail(run_dir, manifest, status="BLOCKED_BEFORE_DATA_BINDING", blockers=cli_blockers)

    try:
        if not args.data.exists() or not args.exposure_ledger.exists():
            raise GP0GateError("D1 data or exposure ledger is missing")
        if not args.foundation_manifest.exists() or not args.foundation_snapshot.exists():
            raise GP0GateError("foundation manifest or checkpoint directory is missing")
        split_binding = load_split_binding(args.split_manifest)
        all_record_ids = set(split_binding["record_roles"])
        exposure_policy = load_exposure_policy(args.exposure_ledger, all_record_ids)
        split_binding_summary = _public_split_binding(split_binding)
        b0_binding = load_b0_binding(args.b0_card, args.b0_split_summary)
        records, selected_summary = scan_d1_and_select(
            args.data,
            split_binding=split_binding,
            requested_split=args.split_role,
            exposure_policy=exposure_policy,
            allow_forbidden_for_development=False,
            max_records=args.max_records,
            sequence_alphabet_policy=args.sequence_alphabet_policy,
        )
        data_binding = file_binding(args.data, supplied_sha256=args.data_sha256)
        ledger_binding = file_binding(args.exposure_ledger, supplied_sha256=args.exposure_ledger_sha256)
        foundation_binding = file_binding(args.foundation_manifest)
        manifest["inputs"].update({
            "data_manifest_sha256": data_binding.get("sha256"),
            "split_manifest_sha256": split_binding_summary["combined_sha256"],
            "data": data_binding,
            "exposure_ledger": ledger_binding,
            "split_binding": split_binding_summary,
            "b0_binding": b0_binding,
            "foundation_manifest": foundation_binding,
        })
        manifest["data_summary"] = {"d1": selected_summary, "selected_split": args.split_role}
        manifest["exposure_policy_summary"] = {
            "bound_record_count": len(exposure_policy),
            "forbidden_accessions_present": sorted(
                {row["accession"] for row in exposure_policy.values() if row["accession"] in {"GSE246381"}}
            ),
            "labels_read": False,
        }
        write_json(run_dir / "provenance.json", {
            "goal_contract": manifest["goal_contract"],
            "scientific_question_id": SCIENTIFIC_QUESTION_ID,
            "phase_id": PHASE_ID,
            "task_id": TASK_ID,
            "git_commit": git["head"],
            "data_manifest_sha256": data_binding.get("sha256"),
            "split_manifest_sha256": split_binding_summary["combined_sha256"],
            "foundation_checkpoint": str(args.foundation_snapshot),
            "foundation_checkpoint_sha256": args.foundation_checkpoint_sha256,
            "exposure_ledger_version": "data_exposure_ledger.jsonl",
            "labels_read": False,
            "raw_sequences_written_to_artifacts": False,
        })

        blockers: list[dict[str, Any]] = []
        if args.mode == "formal":
            blockers.extend(
                validate_formal_data_binding(
                    d1_summary=selected_summary,
                    b0_binding=b0_binding,
                    selected_summary=selected_summary,
                    exposure_policy=exposure_policy,
                )
            )
            if args.foundation_checkpoint_sha256 != "4646f79e76d970ed51aefad811777390ecd43a3e3e5ed6372780583d3be1541":
                blockers.append({"id": "FOUNDATION_CHECKPOINT_NOT_ACCEPTED_FM0_SHA256"})
            required = [REPO_ROOT / "scripts/gp0/train_gp0.py", REPO_ROOT / "scripts/gp0/evaluate_gp0.py"]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                blockers.append({"id": "GP0_FORMAL_ENTRYPOINT_MISSING", "paths": missing})
            blockers.extend(_check_formal_variant(records, args))
        if args.preflight_only:
            manifest["preflight_only"] = True
            manifest["status"] = "PRECHECK_ONLY_BLOCKED" if blockers else "PRECHECK_ONLY_NO_TRAINING"
            manifest["training_started"] = False
            manifest["gpu_forward_started"] = False
            if blockers:
                return _fail(run_dir, manifest, status="PRECHECK_ONLY_BLOCKED", blockers=blockers)
            _finalize(run_dir, manifest)
            return 0
        if blockers:
            return _fail(run_dir, manifest, status="BLOCKED_BEFORE_GPU_TRAINING", blockers=blockers)
        if not records:
            return _fail(run_dir, manifest, status="BLOCKED_NO_ADMITTED_RECORDS", blockers=[{"id": "NO_ADMITTED_PAIRED_RECORDS"}])

        records, variant_filter = _prepare_variant_records(args, records)
        manifest["variant_data_filter"] = variant_filter
        if not records:
            return _fail(run_dir, manifest, status="BLOCKED_VARIANT_EMPTY", blockers=[{"id": "VARIANT_HAS_NO_RECORDS_AFTER_FILTER"}])
        if args.mode == "formal" and variant_filter.get("dropped"):
            return _fail(run_dir, manifest, status="BLOCKED_VARIANT_DATA_FILTER", blockers=[{"id": "FORMAL_VARIANT_WOULD_DROP_RECORDS", "filter": variant_filter}])

        _seed_everything(args.seed)
        gpu = require_cuda_device(args.device)
        if args.gpu_uuid and args.gpu_uuid not in gpu["nvidia_smi"]:
            return _fail(run_dir, manifest, status="BLOCKED_GPU_UUID_MISMATCH", blockers=[{"id": "GPU_UUID_MISMATCH", "requested": args.gpu_uuid, "observed": gpu["nvidia_smi"]}])
        manifest["resource_audit"] = gpu
        import torch

        device = torch.device(args.device)
        field = build_rate_field(
            variant=args.variant,
            snapshot_dir=args.foundation_snapshot,
            device=device,
            seed=args.seed,
            min_length=args.min_length,
            max_length=args.max_length,
            hidden_head_width=args.hidden_head_width,
            adapter_rank=args.adapter_rank,
        )
        field.train()
        model_audit = runtime_model_audit(field)
        if not model_audit["cuda_available"] or not model_audit["all_parameters_cuda"]:
            raise GP0GateError("model parameter/device audit failed closed")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in field.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        manifest["model_audit_before_training"] = model_audit
        rng = random.Random(args.seed ^ 0x5EED)
        probe_record = next((record for record in records if record.edit_distance > 0), records[0])
        probe_example = make_training_example(
            probe_record,
            rng=rng,
            min_length=args.min_length,
            max_length=args.max_length,
            time_policy=args.time_policy,
        )
        probe = _probe_optimizer(field, probe_example, optimizer, min_length=args.min_length, max_length=args.max_length)
        write_json(run_dir / "gpu_probe.json", {
            "schema_version": "gp0_gpu_probe_v1",
            "status": "PASS_ENGINEERING_ONLY",
            "resource": gpu,
            "model": runtime_model_audit(field),
            "probe": probe,
            "cpu_fallback_count": 0,
            "scientific_claim_allowed": False,
        })

        from mrna_editflow.core.mk0.bregman import edit_flow_loss

        metrics_path = run_dir / "logs" / "metrics.jsonl"
        failures_path = run_dir / "logs" / "failures.jsonl"
        order = list(records)
        global_step = 0
        update_count = 0
        zero_target_count = 0
        example_failure_count = 0
        losses: list[float] = []
        baseline_losses: list[float] = []
        for epoch in range(args.epochs):
            rng.shuffle(order)
            for record_index, record in enumerate(order):
                if args.max_steps is not None and global_step >= args.max_steps:
                    break
                if record.edit_distance == 0:
                    zero_target_count += 1
                    continue
                try:
                    example = make_training_example(
                        record,
                        rng=rng,
                        min_length=args.min_length,
                        max_length=args.max_length,
                        time_policy=args.time_policy,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    rates = field(example.state, example.time)
                    assert_cuda_rates(rates)
                    edit_rates = {
                        action: rate
                        for action, rate in rates.items()
                        if action.kind.value != "STOP"
                    }
                    loss = edit_flow_loss(
                        example.state,
                        edit_rates,
                        example.oracle,
                        min_length=args.min_length,
                        max_length=args.max_length,
                    )
                    if not isinstance(loss, torch.Tensor) or loss.device.type != "cuda":
                        raise GP0GateError("GP0 Bregman loss left CUDA or became a Python scalar")
                    if not bool(torch.isfinite(loss)):
                        raise GP0GateError("GP0 Bregman loss is NaN/Inf")
                    baseline_losses.append(_objective_uniform_loss(example, min_length=args.min_length, max_length=args.max_length, device=device))
                    loss.backward()
                    gradients = [parameter.grad for parameter in field.parameters() if parameter.grad is not None]
                    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                        raise GP0GateError("GP0 training backward produced missing/non-finite gradients")
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in field.parameters() if parameter.requires_grad], args.gradient_clip
                    ).detach().cpu())
                    if not math.isfinite(grad_norm):
                        raise GP0GateError("GP0 gradient norm is NaN/Inf")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    loss_value = float(loss.detach().cpu().item())
                    if not math.isfinite(loss_value):
                        raise GP0GateError("GP0 loss conversion failed")
                    losses.append(loss_value)
                    update_count += 1
                    global_step += 1
                    if global_step % args.log_every == 0:
                        append_jsonl(metrics_path, {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": loss_value,
                            "mean_loss": sum(losses[-args.log_every:]) / len(losses[-args.log_every:]),
                            "mean_uniform_baseline_loss": sum(baseline_losses[-args.log_every:]) / len(baseline_losses[-args.log_every:]),
                            "gradient_l2_after_clip": grad_norm,
                            "cuda": True,
                            "cpu_fallback_count": 0,
                        })
                except Exception as error:
                    example_failure_count += 1
                    append_jsonl(failures_path, {
                        "step": global_step,
                        "epoch": epoch,
                        "record_id_sha256": hashlib.sha256(record.record_id.encode()).hexdigest(),
                        "accession": record.accession,
                        "region": record.region,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "done_marker_written": False,
                    })
                    raise
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if not losses:
            return _fail(run_dir, manifest, status="BLOCKED_NO_NONZERO_EDIT_OBJECTIVE", blockers=[{"id": "NO_NONZERO_EDIT_OBJECTIVE", "zero_target_count": zero_target_count}])
        checkpoint = run_dir / "checkpoints" / "model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": "gp0_checkpoint_v1",
            "model_state": field.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "variant": args.variant,
            "seed": args.seed,
            "model_audit": runtime_model_audit(field),
            "hyperparameters": manifest["hyperparameters"],
        }, checkpoint)
        checkpoint_sha = sha256_file(checkpoint)
        manifest["artifacts"].update({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "metrics": str(metrics_path),
            "failures": str(failures_path),
            "gpu_probe": str(run_dir / "gpu_probe.json"),
        })
        manifest["training_summary"] = {
            "epochs_requested": args.epochs,
            "steps_completed": global_step,
            "updates": update_count,
            "zero_target_count": zero_target_count,
            "example_failure_count": example_failure_count,
            "mean_model_bregman_loss": sum(losses) / len(losses),
            "mean_uniform_corruption_baseline_loss": sum(baseline_losses) / len(baseline_losses),
            "lower_is_better": True,
            "formal_scientific_acceptance": False,
        }
        manifest["status"] = "FORMAL_TRAINING_PENDING_HELDOUT_EVALUATION" if args.mode == "formal" else "DEVELOPMENT_EVIDENCE_TRAINED"
        _finalize(run_dir, manifest)
        return 0
    except Exception as error:
        return _fail(
            run_dir,
            manifest,
            status="FAILED_WITH_EVIDENCE",
            blockers=[{"id": "TRAINING_EXCEPTION", "type": type(error).__name__, "message": str(error)}],
            error=error,
        )


def main(argv: list[str] | None = None) -> int:
    return train(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
