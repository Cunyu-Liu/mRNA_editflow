"""Run the official LinearDesign executable on the external SOTA input pack.

This adapter invokes a configured LinearDesign wrapper for every protein target,
parses the official sequence/MFE/CAI output, restores the native terminal stop
codon for the project's valid-CDS contract, and writes the measured-output
schema consumed by :mod:`mrna_editflow.eval.audit_external_sota_real_runs`.

The adapter does not claim that MEF beats LinearDesign. It only produces
auditable external baseline measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Mapping, Optional, Sequence

from mrna_editflow.core.constants import CODON_TABLE, STOP_CODONS, is_valid_cds, translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.prepare_codon import gc3_fraction
from mrna_editflow.eval.metrics import cai, codon_weights_from_reference, gc_fraction


CLAIM_POLICY = (
    "This artifact contains measured outputs from the configured official "
    "LinearDesign executable. It supports a matched CDS baseline table, but "
    "does not establish MEF superiority, wet-lab efficacy, or full-length "
    "de novo generation."
)
MODEL_NAME = "LinearDesign"
TASK_FAMILY = "cds_protein_conditioned"
RNA_ALPHABET = "ACGU"
ALL_CODONS = tuple("".join(parts) for parts in product(RNA_ALPHABET, repeat=3))
ALL_CODON_PAIRS = tuple(f"{a}|{b}" for a in ALL_CODONS for b in ALL_CODONS)
SEQUENCE_RE = re.compile(r"mRNA sequence:\s*([ACGUTacgut]+)")
STRUCTURE_RE = re.compile(r"mRNA structure:\s*([().]+)")
MFE_RE = re.compile(r"mRNA folding free energy:\s*([-+]?\d+(?:\.\d+)?)")
CAI_RE = re.compile(r"mRNA CAI:\s*([-+]?\d+(?:\.\d+)?)")


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(dict(payload))
    return rows


def _write_json(payload: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def _write_jsonl(rows: Sequence[Mapping[str, object]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def _append_jsonl(row: Mapping[str, object], path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _normalise_rna(seq: object) -> str:
    return "".join(str(seq or "").upper().replace("T", "U").split())


def _codons(cds: str) -> list[str]:
    return [cds[i : i + 3] for i in range(0, len(cds) - len(cds) % 3, 3)]


def _protein_identity(a: str, b: str) -> float:
    a = a.rstrip("*")
    b = b.rstrip("*")
    denom = max(len(a), len(b), 1)
    return float(sum(1 for x, y in zip(a, b) if x == y) / denom)


def _distribution(tokens: Sequence[str], keys: Sequence[str], eps: float = 1e-8) -> dict[str, float]:
    counts = Counter({key: float(eps) for key in keys})
    for token in tokens:
        if token in counts:
            counts[token] += 1.0
    total = float(sum(counts.values()))
    return {key: float(counts[key] / total) for key in keys}


def _kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    return float(
        sum(
            float(pv) * math.log(float(pv) / max(float(q.get(key, 0.0)), 1e-12))
            for key, pv in p.items()
            if pv > 0
        )
    )


def _codon_kl(designed_cds: str, native_cds: str, *, pairs: bool = False) -> float:
    designed = _codons(designed_cds)
    native = _codons(native_cds)
    if pairs:
        designed_tokens = [f"{a}|{b}" for a, b in zip(designed, designed[1:])]
        native_tokens = [f"{a}|{b}" for a, b in zip(native, native[1:])]
        keys = ALL_CODON_PAIRS
    else:
        designed_tokens = designed
        native_tokens = native
        keys = ALL_CODONS
    return _kl(_distribution(designed_tokens, keys), _distribution(native_tokens, keys))


def _resolve_pack_file(summary_path: str, summary: Mapping[str, object], key: str, default_name: str) -> str:
    outputs = summary.get("outputs", {})
    outputs = outputs if isinstance(outputs, Mapping) else {}
    value = outputs.get(key)
    if isinstance(value, str):
        candidate = value if os.path.isabs(value) else os.path.join(os.path.dirname(summary_path), value)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(os.path.dirname(summary_path), default_name)


def _parse_output(stdout: str) -> dict[str, object]:
    sequence_match = SEQUENCE_RE.search(stdout)
    if not sequence_match:
        raise ValueError("LinearDesign output did not contain an mRNA sequence")
    structure_match = STRUCTURE_RE.search(stdout)
    mfe_match = MFE_RE.search(stdout)
    cai_match = CAI_RE.search(stdout)
    return {
        "designed_cds_without_stop": _normalise_rna(sequence_match.group(1)),
        "structure": structure_match.group(1) if structure_match else None,
        "mfe_without_stop": float(mfe_match.group(1)) if mfe_match else None,
        "external_reported_cai": float(cai_match.group(1)) if cai_match else None,
    }


def _executable_version(executable: str) -> str:
    proc = subprocess.run(
        [executable, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    text = proc.stdout.strip()
    return text or f"unknown exit={proc.returncode}"


def _run_one(
    row: Mapping[str, object],
    *,
    executable: str,
    timeout_s: float,
    codon_weights: Mapping[str, float],
) -> dict[str, object]:
    start = time.perf_counter()
    protein = str(row.get("protein_target") or "").rstrip("*")
    native_cds = _normalise_rna(row.get("native_cds"))
    proc = subprocess.run(
        [executable],
        input=protein + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_s,
    )
    wall_clock_s = float(time.perf_counter() - start)
    if proc.returncode != 0:
        raise RuntimeError(f"LinearDesign exit={proc.returncode}: {proc.stdout[-1000:]}")
    parsed = _parse_output(proc.stdout)
    designed_without_stop = str(parsed["designed_cds_without_stop"])
    native_codons = _codons(native_cds)
    terminal_stop = native_codons[-1] if native_codons and native_codons[-1] in STOP_CODONS else "UAA"
    designed_codons = _codons(designed_without_stop)
    designed_cds = (
        designed_without_stop
        if designed_codons and designed_codons[-1] in STOP_CODONS
        else designed_without_stop + terminal_stop
    )
    designed_protein = translate(designed_cds)
    target_with_stop = str(row.get("protein_target_with_stop") or protein)
    identity = _protein_identity(target_with_stop, designed_protein)
    valid = bool(is_valid_cds(designed_cds))
    return {
        "transcript_id": row.get("transcript_id"),
        "model_name": MODEL_NAME,
        "designed_cds": designed_cds,
        "wall_clock_s": wall_clock_s,
        "valid_cds": valid,
        "protein_identity": float(identity),
        "protein_identity_exact_1": bool(identity == 1.0),
        "cai": float(cai(designed_cds, codon_weights)),
        "gc": float(gc_fraction(designed_cds)),
        "gc3": float(gc3_fraction(designed_cds)),
        "codon_usage_kl_vs_native": float(_codon_kl(designed_cds, native_cds)),
        "codon_pair_kl_vs_native": float(_codon_kl(designed_cds, native_cds, pairs=True)),
        "mfe": parsed.get("mfe_without_stop"),
        "structure_proxy": parsed.get("structure"),
        "external_reported_cai": parsed.get("external_reported_cai"),
        "native_cds": native_cds,
        "postprocessing": {
            "terminal_stop_appended": bool(not (designed_codons and designed_codons[-1] in STOP_CODONS)),
            "terminal_stop": terminal_stop,
            "mfe_excludes_appended_stop": True,
        },
    }


def run_lineardesign_adapter(
    *,
    input_pack_summary: str,
    executable: str,
    out_dir: str,
    limit: Optional[int] = None,
    workers: int = 1,
    timeout_s: float = 300.0,
    progress_jsonl: Optional[str] = None,
) -> dict[str, object]:
    """Run LinearDesign and write a real-run artifact contract."""
    start = time.perf_counter()
    input_pack = _load_json(input_pack_summary)
    cds_inputs_path = _resolve_pack_file(
        input_pack_summary,
        input_pack,
        "cds_protein_jsonl",
        "cds_protein_inputs.jsonl",
    )
    input_rows = _load_jsonl(cds_inputs_path)
    selected = input_rows[:limit] if limit is not None else input_rows
    reference_records = [
        MRNARecord(str(row.get("transcript_id")), "", _normalise_rna(row.get("native_cds")), "")
        for row in selected
    ]
    weights = codon_weights_from_reference(reference_records)
    version = _executable_version(executable)
    if progress_jsonl and os.path.exists(progress_jsonl):
        os.remove(progress_jsonl)
    _append_jsonl(
        {
            "time": time.time(),
            "event": "start",
            "model_name": MODEL_NAME,
            "n_inputs": len(selected),
            "workers": workers,
        },
        progress_jsonl,
    )

    outputs_by_index: dict[int, dict[str, object]] = {}
    failures: list[dict[str, object]] = []

    def execute(index: int, row: Mapping[str, object]) -> tuple[int, dict[str, object]]:
        return index, _run_one(
            row,
            executable=executable,
            timeout_s=timeout_s,
            codon_weights=weights,
        )

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        future_map = {
            pool.submit(execute, index, row): (index, row)
            for index, row in enumerate(selected)
        }
        completed = 0
        for future in as_completed(future_map):
            index, row = future_map[future]
            try:
                result_index, output = future.result()
                outputs_by_index[result_index] = output
            except Exception as exc:
                failures.append(
                    {
                        "record_index": index,
                        "transcript_id": row.get("transcript_id"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed += 1
            if completed == len(selected) or completed % 25 == 0:
                _append_jsonl(
                    {
                        "time": time.time(),
                        "event": "progress",
                        "completed": completed,
                        "n_inputs": len(selected),
                        "n_outputs": len(outputs_by_index),
                        "n_failures": len(failures),
                    },
                    progress_jsonl,
                )

    outputs = [outputs_by_index[index] for index in sorted(outputs_by_index)]
    elapsed_s = float(time.perf_counter() - start)
    os.makedirs(out_dir, exist_ok=True)
    outputs_path = os.path.join(out_dir, "cds_outputs.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")
    failures_path = os.path.join(out_dir, "failures.jsonl")
    _write_jsonl(outputs, outputs_path)
    _write_jsonl(failures, failures_path)

    def mean(key: str) -> float:
        values = [float(row[key]) for row in outputs if isinstance(row.get(key), (int, float))]
        return float(sum(values) / len(values)) if values else 0.0

    input_pack_outputs = input_pack.get("outputs", {})
    input_pack_outputs = input_pack_outputs if isinstance(input_pack_outputs, Mapping) else {}
    dataset = input_pack.get("dataset", {})
    dataset = dataset if isinstance(dataset, Mapping) else {}
    summary: dict[str, object] = {
        "artifact_kind": "external_sota_real_run_summary",
        "claim_policy": CLAIM_POLICY,
        "model_name": MODEL_NAME,
        "task_family": TASK_FAMILY,
        "protocol_fidelity": "official_code_single_lambda_1_no_lambda_sweep",
        "protocol_fidelity_sufficient_for_sota_reproduction": False,
        "input_pack": {
            "summary_sha256": _sha256_file(input_pack_summary),
            "cds_protein_jsonl_sha256": input_pack_outputs.get("cds_protein_jsonl_sha256"),
            "utr5_jsonl_sha256": input_pack_outputs.get("utr5_jsonl_sha256"),
        },
        "dataset": {
            "records_jsonl_sha256": dataset.get("records_jsonl_sha256"),
            "split_name": dataset.get("split_name"),
            "seed": dataset.get("seed"),
        },
        "runtime": {
            "elapsed_s": elapsed_s,
            "workers": int(workers),
            "timeout_s": float(timeout_s),
        },
        "hardware": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "executable": {
            "path": os.path.abspath(executable),
            "version": version,
            "sha256": _sha256_file(executable),
        },
        "adapter": {
            "module": "mrna_editflow.baselines.external_lineardesign_adapter",
            "terminal_stop_policy": "append native terminal stop when official output omits stop",
            "mfe_scope": "official reported MFE excludes adapter-appended terminal stop",
        },
        "n_inputs": len(selected),
        "n_outputs": len(outputs),
        "n_failures": len(failures),
        "mean_wall_clock_s": mean("wall_clock_s"),
        "valid_cds_fraction": (
            float(sum(bool(row.get("valid_cds")) for row in outputs) / len(outputs))
            if outputs
            else 0.0
        ),
        "protein_identity_exact_1_fraction": (
            float(sum(bool(row.get("protein_identity_exact_1")) for row in outputs) / len(outputs))
            if outputs
            else 0.0
        ),
        "mean_cai": mean("cai"),
        "mean_gc": mean("gc"),
        "mean_gc3": mean("gc3"),
        "mean_mfe_without_stop": mean("mfe"),
        "mean_codon_usage_kl_vs_native": mean("codon_usage_kl_vs_native"),
        "mean_codon_pair_kl_vs_native": mean("codon_pair_kl_vs_native"),
        "outputs": {
            "cds_outputs_jsonl": outputs_path,
            "cds_outputs_sha256": _sha256_file(outputs_path),
            "failures_jsonl": failures_path,
            "failures_sha256": _sha256_file(failures_path),
            "progress_jsonl": progress_jsonl,
        },
    }
    _write_json(summary, summary_path)
    _append_jsonl(
        {
            "time": time.time(),
            "event": "complete",
            "n_inputs": len(selected),
            "n_outputs": len(outputs),
            "n_failures": len(failures),
            "elapsed_s": elapsed_s,
        },
        progress_jsonl,
    )
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack-summary", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--progress-jsonl", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_lineardesign_adapter(
        input_pack_summary=args.input_pack_summary,
        executable=args.executable,
        out_dir=args.out_dir,
        limit=args.limit,
        workers=args.workers,
        timeout_s=args.timeout_s,
        progress_jsonl=args.progress_jsonl,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CLAIM_POLICY", "run_lineardesign_adapter", "main"]
