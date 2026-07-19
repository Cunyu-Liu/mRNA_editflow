"""Dry-run registry for external SOTA baseline integration.

This module is the executable bridge between the static SOTA landscape and the
paper-grade benchmark pipeline. It does **not** fabricate metrics for external
methods. Instead, it records whether an external baseline is locally executable,
which command or environment variable would be used, what protocol differences
remain, and where future real-run artifacts must be written.

Outputs follow the project-wide external SOTA contract:

* ``summary.json``: model-by-model status and protocol metadata.
* ``runtime.json``: reproducibility/runtime metadata for the dry-run itself.
* ``table.md``: compact human-readable readiness table.

Complexity is ``O(M * C)`` where ``M`` is the number of requested models and
``C`` is the small fixed number of candidate command names probed per model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import time
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from mrna_editflow.baselines.external_models import (
    get_external_result,
)
from mrna_editflow.baselines.external_sota_input_pack import EXTERNAL_BENCHMARK_MODELS


_COMMAND_CANDIDATES: dict[str, tuple[str, ...]] = {
    "LinearDesign": ("LINEARDESIGN_BIN", "lineardesign", "LinearDesign"),
    "EnsembleDesign": ("ENSEMBLEDESIGN_BIN", "ensembledesign", "EnsembleDesign"),
    "codonGPT": ("CODONGPT_BIN", "codongpt", "codonGPT"),
    "UTailoR": ("UTAILOR_BIN", "utailor", "UTailoR"),
    "UTRGAN": ("UTRGAN_BIN", "utrgan", "UTRGAN"),
    "mRNA-LM": ("MRNA_LM_BIN", "mrna-lm", "mRNA-LM"),
    "Helix-mRNA": ("HELIX_MRNA_BIN", "helix-mrna", "Helix-mRNA"),
    "CodonFM": ("CODONFM_BIN", "codonfm", "CodonFM"),
    "Prot2RNA": ("PROT2RNA_BIN", "prot2rna", "Prot2RNA"),
    "mRNA2vec": ("MRNA2VEC_BIN", "mrna2vec", "mRNA2vec"),
    "StructmRNA": ("STRUCTMRNA_BIN", "structmrna", "StructmRNA"),
}


@dataclass(frozen=True)
class ExternalSOTADryRunRow:
    """One external SOTA readiness row."""

    model_name: str
    status: str
    task_id: str
    family: str
    citation: str
    expected_inputs: str
    expected_outputs: str
    protocol_difference: str
    executable: Optional[str]
    executable_source: Optional[str]
    command_candidates: tuple[str, ...]
    candidate_audit: tuple[Mapping[str, object], ...]
    metrics: Mapping[str, float]
    notes: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["command_candidates"] = list(self.command_candidates)
        payload["candidate_audit"] = [dict(item) for item in self.candidate_audit]
        return payload


def _is_executable_file(path: str) -> bool:
    """Return whether ``path`` points to a real executable file."""
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _probe_candidate(candidate: str) -> tuple[Optional[str], Optional[str], dict[str, object]]:
    """Resolve an env-var or PATH command candidate.

    Uppercase candidates ending in ``_BIN`` are interpreted as environment
    variables. Other candidates are searched through ``PATH`` with
    :func:`shutil.which`.
    """
    if candidate.endswith("_BIN") and candidate.upper() == candidate:
        value = os.environ.get(candidate)
        audit: dict[str, object] = {
            "candidate": candidate,
            "kind": "env",
            "env_value_set": bool(value),
            "resolved": None,
            "status": "env_unset",
        }
        if not value:
            return None, None, audit
        expanded = os.path.abspath(os.path.expanduser(value))
        audit["resolved"] = expanded
        if _is_executable_file(expanded):
            audit["status"] = "executable"
            return expanded, f"env:{candidate}", audit
        which_value = shutil.which(value)
        if which_value and _is_executable_file(which_value):
            audit["resolved"] = which_value
            audit["status"] = "executable_via_path"
            return which_value, f"env:{candidate}", audit
        audit["status"] = "not_executable"
        audit["exists"] = os.path.exists(expanded)
        audit["is_file"] = os.path.isfile(expanded)
        audit["is_executable"] = os.access(expanded, os.X_OK)
        return None, None, audit
    resolved = shutil.which(candidate)
    if resolved:
        return resolved, "PATH", {
            "candidate": candidate,
            "kind": "path",
            "resolved": resolved,
            "status": "executable",
        }
    return None, None, {
        "candidate": candidate,
        "kind": "path",
        "resolved": None,
        "status": "path_not_found",
    }


def _file_sha256(path: Optional[str]) -> Optional[str]:
    """Return the SHA256 digest for an input file, or ``None`` if absent.

    Dataset identity must be tied to file bytes rather than a mutable filename.
    The streaming hash costs ``O(B)`` time for ``B`` bytes and ``O(1)`` memory.
    """
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_jsonl_records(path: Optional[str]) -> Optional[int]:
    """Count JSONL records in ``path`` if present. Complexity: ``O(N)`` lines."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _effective_record_count(total: Optional[int], limit: Optional[int]) -> Optional[int]:
    if total is None:
        return None
    return min(total, limit) if limit is not None else total


def _hardware_summary(label: Optional[str] = None) -> dict[str, object]:
    """Return host-level hardware metadata used by dry-run and real adapters."""
    return {
        "label": label,
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "platform": platform.platform(),
    }


def _command_candidates(model_name: str) -> tuple[str, ...]:
    return _COMMAND_CANDIDATES.get(model_name, (f"{model_name.upper().replace('-', '_')}_BIN", model_name))


def dry_run_external_sota(
    *,
    models: Optional[Sequence[str]] = None,
    task_id: str = "T5",
    records_jsonl: Optional[str] = None,
    out_dir: str,
    limit: Optional[int] = None,
    split_name: str = "unspecified",
    seed: Optional[int] = None,
    hardware_label: Optional[str] = None,
    write_artifacts: bool = True,
) -> dict[str, object]:
    """Probe external SOTA readiness and optionally write standard artifacts."""
    start = time.perf_counter()
    dataset_sha256 = _file_sha256(records_jsonl)
    record_count_total = _count_jsonl_records(records_jsonl)
    record_count_effective = _effective_record_count(record_count_total, limit)
    hardware = _hardware_summary(hardware_label)
    requested = list(models) if models else list(EXTERNAL_BENCHMARK_MODELS)
    rows: list[ExternalSOTADryRunRow] = []
    for model_name in requested:
        result = get_external_result(
            model_name,
            task_id=task_id,
            extra_note=(
                "Dry-run only: no external executable was run and no metric is "
                "claimed unless status is executable_ready."
            ),
        )
        candidates = _command_candidates(result.model_name)
        executable = None
        executable_source = None
        candidate_audit: list[Mapping[str, object]] = []
        for candidate in candidates:
            executable, executable_source, audit = _probe_candidate(candidate)
            candidate_audit.append(audit)
            if executable:
                break
        status = "executable_ready" if executable else "not_configured"
        notes = result.notes
        if not executable:
            notes = (
                f"{notes} No executable found via candidates {list(candidates)}. "
                "Set the documented *_BIN environment variable or install the "
                "tool on PATH before claiming external SOTA metrics."
            )
        rows.append(
            ExternalSOTADryRunRow(
                model_name=result.model_name,
                status=status,
                task_id=task_id,
                family=result.family,
                citation=result.citation,
                expected_inputs=result.expected_inputs,
                expected_outputs=result.expected_outputs,
                protocol_difference=result.protocol_difference,
                executable=executable,
                executable_source=executable_source,
                command_candidates=candidates,
                candidate_audit=tuple(candidate_audit),
                metrics=dict(result.metrics),
                notes=notes,
            )
        )

    elapsed = time.perf_counter() - start
    runtime = {
        "elapsed_s": float(elapsed),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "records_jsonl": records_jsonl,
        "records_jsonl_exists": bool(records_jsonl and os.path.exists(records_jsonl)),
        "dataset_sha256": dataset_sha256,
        "record_count_total": record_count_total,
        "record_count_effective": record_count_effective,
        "split_name": split_name,
        "seed": seed,
        "hardware": hardware,
        "limit": limit,
        "task_id": task_id,
        "n_models": len(rows),
    }
    dataset = {
        "records_jsonl": records_jsonl,
        "exists": bool(records_jsonl and os.path.exists(records_jsonl)),
        "sha256": dataset_sha256,
        "record_count_total": record_count_total,
        "record_count_effective": record_count_effective,
        "limit": limit,
        "split_name": split_name,
        "seed": seed,
    }
    summary: dict[str, object] = {
        "status": "dry_run_complete",
        "task_id": task_id,
        "records_jsonl": records_jsonl,
        "limit": limit,
        "dataset": dataset,
        "hardware": hardware,
        "n_models": len(rows),
        "n_executable_ready": sum(row.status == "executable_ready" for row in rows),
        "n_not_configured": sum(row.status == "not_configured" for row in rows),
        "rows": [row.to_dict() for row in rows],
        "artifact_contract": {
            "summary_json": "external SOTA status/protocol metadata",
            "runtime_json": "dry-run runtime and environment metadata",
            "table_md": "human-readable readiness table",
            "real_metric_policy": (
                "Do not report accuracy/F1/TE/runtime metrics for an external "
                "method until its row status is executable_ready and a real "
                "adapter writes measured outputs under benchmark/external_sota/."
            ),
            "required_real_run_metadata": [
                "dataset.sha256",
                "dataset.split_name",
                "dataset.seed",
                "dataset.record_count_effective",
                "runtime.elapsed_s",
                "hardware",
            ],
        },
    }
    if write_artifacts:
        write_external_sota_artifacts(summary, runtime, out_dir)
    return {"summary": summary, "runtime": runtime}


def write_external_sota_artifacts(
    summary: Mapping[str, object],
    runtime: Mapping[str, object],
    out_dir: str,
) -> dict[str, str]:
    """Write ``summary.json``, ``runtime.json`` and ``table.md``."""
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")
    runtime_path = os.path.join(out_dir, "runtime.json")
    table_path = os.path.join(out_dir, "table.md")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    with open(runtime_path, "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, sort_keys=True)
    write_external_sota_table(summary, table_path)
    return {
        "summary_json": summary_path,
        "runtime_json": runtime_path,
        "table_md": table_path,
    }


def write_external_sota_table(summary: Mapping[str, object], path: str) -> str:
    """Write a Markdown readiness table."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = summary.get("rows", [])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# External SOTA Dry-Run Readiness\n\n")
        fh.write(f"Task: `{summary.get('task_id', '')}`\n\n")
        dataset = summary.get("dataset", {})
        if isinstance(dataset, Mapping):
            fh.write(
                "Dataset audit: "
                f"split=`{dataset.get('split_name', '')}`, "
                f"seed=`{dataset.get('seed')}`, "
                f"records=`{dataset.get('record_count_effective')}` / "
                f"{dataset.get('record_count_total')}, "
                f"sha256=`{dataset.get('sha256') or 'NA'}`\n\n"
            )
        fh.write("| Model | Status | Family | Executable | Command candidates | Protocol note |\n")
        fh.write("|---|---|---|---|---|---|\n")
        if isinstance(rows, Sequence):
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                candidates = row.get("command_candidates", [])
                if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                    candidate_text = ", ".join(str(x) for x in candidates)
                else:
                    candidate_text = str(candidates)
                audit_rows = row.get("candidate_audit", [])
                if isinstance(audit_rows, Sequence) and not isinstance(audit_rows, (str, bytes)):
                    audit_text = "; ".join(
                        f"{item.get('candidate')}={item.get('status')}"
                        for item in audit_rows
                        if isinstance(item, Mapping)
                    )
                else:
                    audit_text = ""
                fh.write(
                    f"| {row.get('model_name', '')} | `{row.get('status', '')}` | "
                    f"{row.get('family', '')} | `{row.get('executable') or ''}` | "
                    f"`{candidate_text}` | {row.get('protocol_difference', '')} "
                    f"Candidate audit: {audit_text}. |\n"
                )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task-id", default="T5")
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-name", default="unspecified")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hardware-label", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = dry_run_external_sota(
        models=args.models,
        task_id=args.task_id,
        records_jsonl=args.records_jsonl,
        out_dir=args.out_dir,
        limit=args.limit,
        split_name=args.split_name,
        seed=args.seed,
        hardware_label=args.hardware_label,
        write_artifacts=False,
    )
    paths = write_external_sota_artifacts(payload["summary"], payload["runtime"], args.out_dir)
    print(json.dumps(paths, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExternalSOTADryRunRow",
    "dry_run_external_sota",
    "write_external_sota_artifacts",
    "write_external_sota_table",
    "main",
]
