#!/usr/bin/env python3
"""Held-out GP0 evaluator with explicit scientific-claim boundaries.

The evaluator reports the MK0 edit-flow objective against a uniform legal
corruption reference, plus hard-mask/source-budget/diversity diagnostics.  It
does not declare GP0 accepted: the contract has no license to convert a proxy
objective, a sampler proxy, or one seed into a scientific conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import random
import sys
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
    GP0GateError,
    PairedRecord,
    append_jsonl,
    artifact_checksums,
    assert_cuda_rates,
    build_rate_field,
    file_binding,
    load_exposure_policy,
    load_split_binding,
    make_training_example,
    require_cuda_device,
    runtime_model_audit,
    scan_d1_and_select,
    sha256_file,
    write_json,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--exposure-ledger", type=Path)
    parser.add_argument("--split-manifest", type=Path, action="append")
    parser.add_argument("--split-role", choices=("train", "val", "test"), default="test")
    parser.add_argument("--foundation-snapshot", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--samples-per-record", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args(argv)


def _resolve_inputs(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    inputs = manifest.get("inputs", {})
    data = args.data or Path(inputs.get("data", {}).get("path", ""))
    ledger = args.exposure_ledger or Path(inputs.get("exposure_ledger", {}).get("path", ""))
    snapshot = args.foundation_snapshot or Path(inputs.get("foundation_checkpoint", ""))
    split_paths = args.split_manifest
    if not split_paths:
        split_paths = [Path(item["binding"]["path"]) for item in inputs.get("split_binding", {}).get("paths", [])]
    if not data.exists() or not ledger.exists() or not snapshot.exists() or not split_paths:
        raise GP0GateError("evaluation inputs cannot be resolved from CLI/run manifest")
    if args.samples_per_record < 1 or args.samples_per_record > 32:
        raise GP0GateError("samples-per-record must lie in [1,32]")
    return {"data": data, "ledger": ledger, "snapshot": snapshot, "split_paths": split_paths}


def _uniform_loss(example: Any, *, min_length: int, max_length: int, device: Any) -> float:
    import torch

    from mrna_editflow.core.mk0.bregman import edit_flow_loss
    from mrna_editflow.core.mk0.state_action import enumerate_legal_actions

    legal = enumerate_legal_actions(example.state, min_length=min_length, max_length=max_length, include_stop=False)
    rates = {action: torch.ones((), device=device, dtype=torch.float32) for action in legal}
    value = edit_flow_loss(example.state, rates, example.oracle, min_length=min_length, max_length=max_length)
    return float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)


def _sample_fixed_time(field: Any, record: PairedRecord, *, min_length: int, max_length: int, rng: random.Random) -> dict[str, Any]:
    """A legal-action draw used only for hard-mask/diversity diagnostics."""

    from mrna_editflow.core.mk0.state_action import apply_action, enumerate_legal_actions
    from mrna_editflow.core.mk0.types import ActionType, EditState

    state = EditState.initial(
        record.source,
        region=record.region,
        context={"assay": "unspecified", "cell_or_tissue": "unspecified", "endpoint": "unspecified", "batch": None},
        target_condition="maintain",
        budget=max(1, record.edit_distance),
    )
    steps = 0
    violation = None
    try:
        while state.phase.value == "ACTIVE" and steps <= state.initial_budget:
            rates = field.rate_fn(state, 0.0)
            legal = set(enumerate_legal_actions(state, min_length=min_length, max_length=max_length, include_stop=True))
            if set(rates) - legal:
                raise GP0GateError("rate_fn returned a hard-illegal action")
            ordered = [(action, float(rate)) for action, rate in rates.items() if action in legal and rate > 0.0]
            total = math.fsum(rate for _, rate in ordered)
            if not math.isfinite(total) or total <= 0.0:
                raise GP0GateError("sampler proxy encountered zero/invalid legal hazard")
            draw = rng.random() * total
            cumulative = 0.0
            selected = ordered[-1][0]
            for action, rate in sorted(ordered, key=lambda item: item[0].key):
                cumulative += rate
                if draw <= cumulative:
                    selected = action
                    break
            transition = apply_action(state, selected, min_length=min_length, max_length=max_length)
            state = transition.after
            steps += 1
            if selected.kind == ActionType.STOP:
                break
        return {
            "status": "PASS",
            "candidate_sha256": hashlib.sha256(state.current.encode("ascii")).hexdigest(),
            "candidate_length": len(state.current),
            "edit_events": state.history.executed,
            "termination": state.termination_reason.value if state.termination_reason else None,
            "hard_constraint_violation": False,
        }
    except Exception as error:
        violation = error
    return {
        "status": "FAIL",
        "candidate_sha256": None,
        "candidate_length": None,
        "edit_events": steps,
        "termination": None,
        "hard_constraint_violation": True,
        "error_type": type(violation).__name__ if violation else "UnknownError",
        "error_message": str(violation) if violation else "unknown",
    }


def _region_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "unique_candidate_count": 0, "unique_ratio": None}
    hashes = [row["candidate_sha256"] for row in rows if row.get("candidate_sha256")]
    lengths = [row["candidate_length"] for row in rows if row.get("candidate_length") is not None]
    edits = [row["edit_events"] for row in rows if row.get("edit_events") is not None]
    counts: dict[str, int] = {}
    for value in hashes:
        counts[value] = counts.get(value, 0) + 1
    return {
        "sample_count": len(rows),
        "successful_sample_count": len(hashes),
        "unique_candidate_count": len(set(hashes)),
        "unique_ratio": len(set(hashes)) / len(hashes) if hashes else None,
        "max_same_candidate_frequency": max(counts.values()) if counts else None,
        "mean_candidate_length": sum(lengths) / len(lengths) if lengths else None,
        "mean_edit_events": sum(edits) / len(edits) if edits else None,
        "hard_constraint_violations": sum(bool(row.get("hard_constraint_violation")) for row in rows),
    }


def evaluate(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise GP0GateError("run manifest does not exist")
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status", "").startswith("FAILED") or manifest.get("status", "").startswith("BLOCKED"):
        raise GP0GateError("cannot evaluate a failed or blocked trainer run")
    checkpoint_path = Path(manifest.get("artifacts", {}).get("checkpoint", ""))
    if not checkpoint_path.exists():
        raise GP0GateError("trained checkpoint is not bound in run manifest")
    inputs = _resolve_inputs(args, manifest)
    split_binding = load_split_binding(inputs["split_paths"])
    policy = load_exposure_policy(inputs["ledger"], set(split_binding["record_roles"]))
    records, selected_summary = scan_d1_and_select(
        inputs["data"],
        split_binding=split_binding,
        requested_split=args.split_role,
        exposure_policy=policy,
        allow_forbidden_for_development=False,
        max_records=args.max_records,
        sequence_alphabet_policy=str(manifest.get("hyperparameters", {}).get("sequence_alphabet_policy", "dna_t_to_rna_u")),
    )
    if not records:
        raise GP0GateError("held-out evaluation split has no admitted paired records")

    import torch

    resource = require_cuda_device(args.device)
    if args.gpu_uuid and args.gpu_uuid not in resource["nvidia_smi"]:
        raise GP0GateError("evaluation GPU UUID does not match nvidia-smi binding")
    hp = manifest.get("hyperparameters", {})
    variant = manifest["variant"]
    field = build_rate_field(
        variant=variant,
        snapshot_dir=inputs["snapshot"],
        device=torch.device(args.device),
        seed=int(manifest.get("formal_seed_requirement", {}).get("current_seed", manifest.get("hyperparameters", {}).get("seed", args.seed))),
        min_length=int(hp.get("min_length", 1)),
        max_length=int(hp.get("max_length", 256)),
        hidden_head_width=int(hp.get("hidden_head_width", 128)),
        adapter_rank=int(hp.get("adapter_rank", 8)),
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device), weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only keyword
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device))
    if checkpoint.get("variant") != variant:
        raise GP0GateError("checkpoint variant does not match run manifest")
    field.load_state_dict(checkpoint["model_state"], strict=True)
    field.eval()
    audit = runtime_model_audit(field)
    if not audit["all_parameters_cuda"]:
        raise GP0GateError("evaluation model/device audit failed closed")

    min_length = int(hp.get("min_length", 1))
    max_length = int(hp.get("max_length", 256))
    time_policy = str(hp.get("time_policy", "stochastic"))
    seed = int(manifest.get("hyperparameters", {}).get("seed", args.seed))
    rng = random.Random(seed ^ 0xEA71)
    objective_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, record in enumerate(records):
            if record.edit_distance == 0:
                continue
            try:
                example = make_training_example(
                    record,
                    rng=rng,
                    min_length=min_length,
                    max_length=max_length,
                    time_policy=time_policy,
                )
                rates = field(example.state, example.time)
                assert_cuda_rates(rates)
                edit_rates = {action: rate for action, rate in rates.items() if action.kind.value != "STOP"}
                from mrna_editflow.core.mk0.bregman import edit_flow_loss

                model_value = edit_flow_loss(example.state, edit_rates, example.oracle, min_length=min_length, max_length=max_length)
                model_loss = float(model_value.detach().cpu().item()) if isinstance(model_value, torch.Tensor) else float(model_value)
                baseline_loss = _uniform_loss(example, min_length=min_length, max_length=max_length, device=torch.device(args.device))
                objective_rows.append({
                    "region": record.region,
                    "model_loss": model_loss,
                    "uniform_corruption_baseline_loss": baseline_loss,
                    "model_lower_than_uniform": model_loss < baseline_loss,
                })
                for sample_index in range(args.samples_per_record):
                    sample = _sample_fixed_time(field, record, min_length=min_length, max_length=max_length, rng=rng)
                    sample["region"] = record.region
                    sample["source_record_id_sha256"] = hashlib.sha256(record.record_id.encode()).hexdigest()
                    sample["sample_index"] = sample_index
                    sampling_rows.append(sample)
            except Exception as error:
                failures.append({
                    "record_index": index,
                    "record_id_sha256": hashlib.sha256(record.record_id.encode()).hexdigest(),
                    "accession": record.accession,
                    "region": record.region,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                })
                break

    by_region = {
        region: _region_metrics([row for row in sampling_rows if row["region"] == region])
        for region in ("5UTR", "3UTR")
    }
    model_mean = sum(row["model_loss"] for row in objective_rows) / len(objective_rows) if objective_rows else None
    baseline_mean = sum(row["uniform_corruption_baseline_loss"] for row in objective_rows) / len(objective_rows) if objective_rows else None
    objective_by_region = {}
    for region in ("5UTR", "3UTR"):
        subset = [row for row in objective_rows if row["region"] == region]
        objective_by_region[region] = {
            "n": len(subset),
            "model_mean_loss": sum(row["model_loss"] for row in subset) / len(subset) if subset else None,
            "uniform_mean_loss": sum(row["uniform_corruption_baseline_loss"] for row in subset) / len(subset) if subset else None,
            "model_lower_than_uniform_fraction": sum(row["model_lower_than_uniform"] for row in subset) / len(subset) if subset else None,
        }
    gates = {
        "heldout_objective_finite": bool(objective_rows) and all(math.isfinite(row["model_loss"]) and math.isfinite(row["uniform_corruption_baseline_loss"]) for row in objective_rows),
        "heldout_objective_beats_uniform_on_mean": model_mean is not None and baseline_mean is not None and model_mean < baseline_mean,
        "hard_constraints": not failures and all(row["hard_constraint_violation"] is False for row in sampling_rows),
        "source_preservation_and_budget": not failures,
        "mode_collapse": "NOT_ASSESSED_NO_FROZEN_CONTRACT_THRESHOLD",
        "five_seed_aggregation": "NOT_RUN_BY_SINGLE_SEED_EVALUATOR",
        "formal_scientific_acceptance": False,
    }
    evaluation = {
        "schema_version": "gp0_evaluation_v1",
        "status": "DEVELOPMENT_EVIDENCE_ONLY" if manifest.get("mode") == "development" else "FORMAL_EVALUATION_PENDING_FINALIZER",
        "run_id": manifest["run_id"],
        "goal_contract": {"id": CONTRACT_ID, "sha256": CONTRACT_SHA256},
        "scientific_question_id": manifest["scientific_question_id"],
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "heldout_data": {
            "split_role": args.split_role,
            "selected_summary": selected_summary,
            "data_path": str(inputs["data"]),
            "data_sha256": sha256_file(inputs["data"]),
            "split_manifest_sha256": split_binding["combined_sha256"],
        },
        "resource_audit": resource,
        "model_audit": audit,
        "objective": {
            "semantics": "lower is better; exact MK0 edit-flow Bregman loss evaluated on held-out pairs",
            "model_mean_loss": model_mean,
            "uniform_corruption_baseline_mean_loss": baseline_mean,
            "by_region": objective_by_region,
        },
        "sampling_proxy": {
            "sampler": "fixed_time_legal_action_draw",
            "time": 0.0,
            "semantics": "engineering hard-mask/diversity proxy; not exact CTMC and not biological evidence",
            "by_region": by_region,
        },
        "gates": gates,
        "failures": failures,
        "done_marker_written": False,
    }
    write_json(run_dir / "evaluation.json", evaluation)
    if failures:
        write_json(run_dir / "evaluation_failure_evidence.json", {"failures": failures, "done_marker_written": False})
    manifest["artifacts"]["evaluation"] = str(run_dir / "evaluation.json")
    manifest["evaluation_summary"] = {
        "status": evaluation["status"],
        "heldout_split": args.split_role,
        "formal_scientific_acceptance": False,
        "gates": gates,
    }
    manifest["status"] = "DEVELOPMENT_EVIDENCE_EVALUATED" if manifest.get("mode") == "development" else "FORMAL_EVALUATION_PENDING_GP0_FINALIZER"
    write_json(manifest_path, manifest)
    write_json(run_dir / "status.json", {"status": manifest["status"], "run_id": manifest["run_id"]})
    (run_dir / "artifact_checksums.sha256").write_text(artifact_checksums(run_dir), encoding="utf-8")
    return 0 if not failures else 2


def main(argv: list[str] | None = None) -> int:
    parsed = _parse_args(argv)
    try:
        return evaluate(parsed)
    except Exception as error:
        run_dir = parsed.run_dir.resolve()
        if run_dir.exists():
            write_json(run_dir / "evaluation_failure_evidence.json", {
                "status": "FAILED_WITH_EVIDENCE",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "done_marker_written": False,
            })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
