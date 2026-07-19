"""Read-only health audit for long-running Stage A profile JSONL files.

This module never imports process-control libraries and never writes to profile,
checkpoint, or configuration inputs.  Its verdict is advisory only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Optional, Sequence

import numpy as np

from mrna_editflow.data.split_contract import sha256_file

VERDICTS = ("healthy_to_continue", "manual_review_required", "stop_recommended")


def _finite_numbers(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _rolling_loss(rows: Sequence[Mapping[str, object]], window: int) -> dict[str, object]:
    values = _finite_numbers(rows[-max(1, int(window)):], "loss")
    if not values:
        return {"count": 0, "mean": None, "median": None, "slope_per_step": None, "mad": None}
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, np.asarray(values), 1)[0]) if len(values) >= 2 else 0.0
    center = median(values)
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(center),
        "slope_per_step": slope,
        "mad": float(median([abs(value - center) for value in values])),
    }


def _load_profile(path: str) -> list[dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is invalid JSON") from exc
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _load_config(path: Optional[str]) -> dict[str, object]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _scientific_warnings(
    config: Mapping[str, object],
    *,
    split_provenance_present: bool,
    held_out_curve_present: bool,
) -> list[str]:
    model = config.get("model", {})
    model = model if isinstance(model, Mapping) else {}
    validity = config.get("scientific_validity", {})
    validity = validity if isinstance(validity, Mapping) else {}
    warnings = []
    if validity.get("records_role_restricted") is not True:
        warnings.append("unrestricted_full_corpus_training")
    if validity.get("records_pretruncated") is not False:
        warnings.append("pretruncated_records")
    if bool(model.get("use_aux_struct")) and float(model.get("aux_loss_weight", 0.0)) > 0:
        warnings.append("auxiliary_zero_target_config_in_running_snapshot")
    if not split_provenance_present:
        warnings.append("missing_split_provenance")
    if not held_out_curve_present:
        warnings.append("missing_held_out_curve")
    return warnings


def audit_profile(
    profile_path: str,
    *,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    process_command: Optional[str] = None,
    target_steps: int = 100000,
    rolling_window: int = 500,
    held_out_curve_present: bool = False,
    split_provenance_present: bool = False,
) -> dict[str, object]:
    """Audit one profile without mutating any input artifact."""
    before_stat = os.stat(profile_path)
    rows = _load_profile(profile_path)
    after_stat = os.stat(profile_path)
    if (before_stat.st_size, before_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
        raise RuntimeError("profile changed while it was being audited; retry a stable read")
    config = _load_config(config_path)
    losses = _finite_numbers(rows, "loss")
    grad_norms = _finite_numbers(rows, "grad_norm")
    throughputs = _finite_numbers(rows, "samples_per_s")
    retry_counts = Counter(int(row.get("retries", 0)) for row in rows)
    amp_enabled = [bool(row.get("amp_enabled")) for row in rows]
    amp_fallback = [bool(row.get("amp_fallback_used")) for row in rows]
    oom_count = sum(int(row.get("oom_reductions", 0)) for row in rows)
    nonfinite_loss = sum(row.get("finite_loss") is False for row in rows)
    nonfinite_grad = sum(row.get("finite_grad") is False for row in rows)
    last_step = max((int(row.get("step", 0)) for row in rows), default=0)
    rough_steps_per_s = None
    eta_seconds = None
    if throughputs and rows:
        recent_batch = max(1, int(rows[-1].get("batch_size", 1)))
        rough_steps_per_s = float(median(throughputs) / recent_batch)
        if rough_steps_per_s > 0:
            eta_seconds = max(0.0, target_steps - last_step) / rough_steps_per_s
    checkpoint_exists = bool(checkpoint_path and os.path.isfile(checkpoint_path))
    warnings = _scientific_warnings(
        config,
        split_provenance_present=split_provenance_present,
        held_out_curve_present=held_out_curve_present,
    )
    reasons = []
    if nonfinite_loss or nonfinite_grad:
        reasons.append("nonfinite_loss_or_gradient_observed")
    rolling = _rolling_loss(rows, rolling_window)
    if losses and rolling["median"] is not None and float(rolling["median"]) > 10 * max(min(losses), 1e-12):
        reasons.append("recent_loss_far_above_best_observed")
    fallback_fraction = float(sum(amp_fallback) / len(amp_fallback)) if amp_fallback else 0.0
    amp_fraction = float(sum(amp_enabled) / len(amp_enabled)) if amp_enabled else 0.0
    grad_p99 = _quantile(grad_norms, 0.99)
    retry_p90 = _quantile([float(value) for value in retry_counts.elements()], 0.90)
    if fallback_fraction > 0.25:
        reasons.append("frequent_amp_fallback")
    if retry_p90 is not None and retry_p90 >= 3:
        reasons.append("frequent_step_retries")
    if grad_p99 is not None and grad_p99 > 100:
        reasons.append("large_preclip_gradient_norms")
    if not held_out_curve_present:
        reasons.append("no_held_out_curve")
    if nonfinite_loss or nonfinite_grad:
        verdict = "stop_recommended"
    elif reasons or warnings:
        verdict = "manual_review_required"
    else:
        verdict = "healthy_to_continue"
    return {
        "profile_path": str(Path(profile_path).resolve()),
        "profile_sha256": sha256_file(profile_path),
        "profile_rows": len(rows),
        "last_completed_step": last_step,
        "last_update_timestamp": dt.datetime.fromtimestamp(
            after_stat.st_mtime, tz=dt.timezone.utc
        ).isoformat(),
        "rolling_loss": rolling,
        "loss_all": {
            "mean": float(mean(losses)) if losses else None,
            "median": float(median(losses)) if losses else None,
            "nonfinite_count": nonfinite_loss,
        },
        "amp": {
            "enabled_fraction": amp_fraction,
            "fallback_fraction": fallback_fraction,
        },
        "retries": {
            "distribution": {str(key): value for key, value in sorted(retry_counts.items())},
            "p90": retry_p90,
            "oom_reduction_count": oom_count,
        },
        "preclip_gradient_norm": {
            "median": _quantile(grad_norms, 0.50),
            "p90": _quantile(grad_norms, 0.90),
            "p99": grad_p99,
            "max": max(grad_norms) if grad_norms else None,
            "nonfinite_count": nonfinite_grad,
        },
        "throughput": {
            "samples_per_s_median": _quantile(throughputs, 0.50),
            "samples_per_s_mean": float(mean(throughputs)) if throughputs else None,
            "rough_steps_per_s": rough_steps_per_s,
            "eta_seconds_rough_estimate": eta_seconds,
            "eta_is_estimate": True,
        },
        "checkpoint": {
            "path": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
            "exists": checkpoint_exists,
            "sha256": sha256_file(checkpoint_path) if checkpoint_exists else None,
        },
        "held_out_curve_exists": bool(held_out_curve_present),
        "process_command_supplied": process_command,
        "scientific_validity_warnings": warnings,
        "verdict": verdict,
        "verdict_reasons": sorted(set(reasons)),
        "advisory_only": True,
        "biological_progress_inferred_from_training_loss": False,
    }


def audit_stage_a_health(
    profile_paths: Sequence[str],
    *,
    config_path: Optional[str] = None,
    checkpoint_paths: Optional[Sequence[Optional[str]]] = None,
    process_commands: Optional[Sequence[Optional[str]]] = None,
    target_steps: int = 100000,
    rolling_window: int = 500,
) -> dict[str, object]:
    checkpoints = list(checkpoint_paths or [None] * len(profile_paths))
    commands = list(process_commands or [None] * len(profile_paths))
    checkpoints.extend([None] * (len(profile_paths) - len(checkpoints)))
    commands.extend([None] * (len(profile_paths) - len(commands)))
    runs = [
        audit_profile(
            profile,
            config_path=config_path,
            checkpoint_path=checkpoints[index],
            process_command=commands[index],
            target_steps=target_steps,
            rolling_window=rolling_window,
        )
        for index, profile in enumerate(profile_paths)
    ]
    severity = {"healthy_to_continue": 0, "manual_review_required": 1, "stop_recommended": 2}
    overall = max((row["verdict"] for row in runs), key=lambda item: severity[item], default="manual_review_required")
    return {
        "artifact_kind": "stage_a_health_audit",
        "schema_version": 1,
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "overall_verdict": overall,
        "runs": runs,
        "advisory_only": True,
        "process_control_actions": [],
        "claim_policy": "Training loss is an engineering health signal, not evidence of biological progress.",
    }


def write_reports(payload: Mapping[str, object], out_json: str, out_md: str) -> None:
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("# Stage A Health Audit\n\n")
        fh.write(f"- Overall verdict: `{payload.get('overall_verdict')}`\n")
        fh.write("- Advisory only: `true`; no process-control action was taken.\n")
        fh.write("- Training loss is not interpreted as biological progress.\n\n")
        fh.write("| Run | Step | Verdict | AMP fallback | Grad P99 | ETA estimate |\n")
        fh.write("|---|---:|---|---:|---:|---:|\n")
        for run in payload.get("runs", []):
            if not isinstance(run, Mapping):
                continue
            fh.write(
                f"| `{Path(str(run.get('profile_path'))).name}` | {run.get('last_completed_step')} | "
                f"`{run.get('verdict')}` | {run.get('amp', {}).get('fallback_fraction')} | "
                f"{run.get('preclip_gradient_norm', {}).get('p99')} | "
                f"{run.get('throughput', {}).get('eta_seconds_rough_estimate')} |\n"
            )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", action="append", default=None)
    parser.add_argument("--process-command", action="append", default=None)
    parser.add_argument("--target-steps", type=int, default=100000)
    parser.add_argument("--rolling-window", type=int, default=500)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = audit_stage_a_health(
        args.profile,
        config_path=args.config,
        checkpoint_paths=args.checkpoint,
        process_commands=args.process_command,
        target_steps=args.target_steps,
        rolling_window=args.rolling_window,
    )
    write_reports(payload, args.out_json, args.out_md)
    print(json.dumps({"overall_verdict": payload["overall_verdict"], "out_json": args.out_json, "out_md": args.out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERDICTS",
    "audit_profile",
    "audit_stage_a_health",
    "write_reports",
    "main",
]
