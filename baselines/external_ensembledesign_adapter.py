"""Run official EnsembleDesign on the standardized protein input pack.

The official driver accepts a protein FASTA, launches stochastic optimization,
and returns the minimum-ensemble-free-energy coding sequence across runs. This
adapter executes one input row per subprocess so a long head1024 benchmark can
resume at record granularity. Every completed row is appended immediately to
an auditable JSONL file.

Budgeted settings and paper-default settings are deliberately distinguished.
Only beam=200, iterations=30, runs=20, lr=0.03, epsilon=0.5 is marked as
protocol-faithful. A smaller run can populate a descriptive metric table but
cannot support an external SOTA reproduction claim.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Mapping, Optional, Sequence

from mrna_editflow.baselines.external_lineardesign_adapter import (
    _codon_kl,
    _codons,
    _load_json,
    _load_jsonl,
    _normalise_rna,
    _protein_identity,
    _resolve_pack_file,
    _sha256_file,
    _write_json,
    _write_jsonl,
)
from mrna_editflow.core.constants import STOP_CODONS, is_valid_cds, translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.prepare_codon import gc3_fraction
from mrna_editflow.eval.metrics import (
    cai,
    codon_weights_from_reference,
    gc_fraction,
)


CLAIM_POLICY = (
    "This artifact contains measured outputs from official EnsembleDesign. "
    "Budgeted runs support descriptive CDS metrics only. They do not establish "
    "paper-faithful reproduction, MEF superiority, or wet-lab efficacy."
)
MODEL_NAME = "EnsembleDesign"
TASK_FAMILY = "cds_protein_conditioned"
OUTPUT_RE = re.compile(
    r">(?P<id>[^\n|]+)\|Ensemble Free Energy:\s*"
    r"(?P<energy>[-+]?\d+(?:\.\d+)?)\s*kcal/mol\s*\n"
    r"(?P<sequence>[ACGUTacgut]+)"
)


def _append_jsonl(
    row: Mapping[str, object],
    path: Optional[str],
    lock: Optional[threading.Lock] = None,
) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if lock is None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")
        return
    with lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _executable_version(executable: str) -> str:
    proc = subprocess.run(
        [executable, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    return proc.stdout.strip() or f"unknown exit={proc.returncode}"


def _parse_output(stdout: str, expected_id: str) -> tuple[str, float]:
    match = OUTPUT_RE.search(stdout)
    if not match:
        raise ValueError("EnsembleDesign output did not contain a result record")
    if match.group("id") != expected_id:
        raise ValueError(
            f"EnsembleDesign output id mismatch: "
            f"{match.group('id')} != {expected_id}"
        )
    return _normalise_rna(match.group("sequence")), float(match.group("energy"))


def _paper_default(
    *,
    beam_size: int,
    num_iters: int,
    num_runs: int,
    learning_rate: float,
    epsilon: float,
) -> bool:
    return bool(
        int(beam_size) == 200
        and int(num_iters) == 30
        and int(num_runs) == 20
        and abs(float(learning_rate) - 0.03) <= 1e-12
        and abs(float(epsilon) - 0.5) <= 1e-12
    )


def _run_one(
    index: int,
    row: Mapping[str, object],
    *,
    executable: str,
    timeout_s: float,
    beam_size: int,
    num_iters: int,
    num_runs: int,
    learning_rate: float,
    epsilon: float,
    codon_weights: Mapping[str, float],
) -> tuple[int, dict[str, object]]:
    transcript_id = str(row.get("transcript_id"))
    safe_id = f"row_{index:06d}"
    protein = str(row.get("protein_target") or "").rstrip("*")
    native_cds = _normalise_rna(row.get("native_cds"))
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"mef_ensemble_{index:06d}_") as tmp:
        fasta_path = os.path.join(tmp, "input.fasta")
        output_dir = os.path.join(tmp, "outputs")
        with open(fasta_path, "w", encoding="utf-8") as fh:
            fh.write(f">{safe_id}\n{protein}\n")
        proc = subprocess.run(
            [
                executable,
                "--fasta",
                fasta_path,
                "--output_dir",
                output_dir,
                "--beam_size",
                str(int(beam_size)),
                "--num_iters",
                str(int(num_iters)),
                "--num_runs",
                str(int(num_runs)),
                "--num_threads",
                "1",
                "--lr",
                str(float(learning_rate)),
                "--epsilon",
                str(float(epsilon)),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_s,
        )
    wall_clock_s = float(time.perf_counter() - start)
    if proc.returncode != 0:
        raise RuntimeError(
            f"EnsembleDesign exit={proc.returncode}: {proc.stdout[-2000:]}"
        )
    designed_without_stop, ensemble_energy = _parse_output(proc.stdout, safe_id)
    native_codons = _codons(native_cds)
    terminal_stop = (
        native_codons[-1]
        if native_codons and native_codons[-1] in STOP_CODONS
        else "UAA"
    )
    designed_codons = _codons(designed_without_stop)
    terminal_stop_appended = not (
        designed_codons and designed_codons[-1] in STOP_CODONS
    )
    designed_cds = (
        designed_without_stop + terminal_stop
        if terminal_stop_appended
        else designed_without_stop
    )
    designed_protein = translate(designed_cds)
    target_with_stop = str(row.get("protein_target_with_stop") or protein)
    identity = _protein_identity(target_with_stop, designed_protein)
    return index, {
        "transcript_id": transcript_id,
        "model_name": MODEL_NAME,
        "designed_cds": designed_cds,
        "wall_clock_s": wall_clock_s,
        "valid_cds": bool(is_valid_cds(designed_cds)),
        "protein_identity": float(identity),
        "protein_identity_exact_1": bool(identity == 1.0),
        "cai": float(cai(designed_cds, codon_weights)),
        "gc": float(gc_fraction(designed_cds)),
        "gc3": float(gc3_fraction(designed_cds)),
        "codon_usage_kl_vs_native": float(
            _codon_kl(designed_cds, native_cds)
        ),
        "codon_pair_kl_vs_native": float(
            _codon_kl(designed_cds, native_cds, pairs=True)
        ),
        "ensemble_free_energy": float(ensemble_energy),
        "native_cds": native_cds,
        "postprocessing": {
            "terminal_stop_appended": terminal_stop_appended,
            "terminal_stop": terminal_stop,
            "ensemble_free_energy_excludes_appended_stop": True,
        },
    }


def run_ensembledesign_adapter(
    *,
    input_pack_summary: str,
    executable: str,
    out_dir: str,
    limit: Optional[int] = None,
    workers: int = 1,
    timeout_s: float = 3600.0,
    beam_size: int = 100,
    num_iters: int = 3,
    num_runs: int = 1,
    learning_rate: float = 0.03,
    epsilon: float = 0.5,
    rescue_beam_size: Optional[int] = None,
    resume: bool = False,
    progress_jsonl: Optional[str] = None,
) -> dict[str, object]:
    """Run EnsembleDesign and persist every completed row for safe resumption."""
    start = time.perf_counter()
    pack = _load_json(input_pack_summary)
    cds_inputs_path = _resolve_pack_file(
        input_pack_summary,
        pack,
        "cds_protein_jsonl",
        "cds_protein_inputs.jsonl",
    )
    input_rows = _load_jsonl(cds_inputs_path)
    selected = input_rows[:limit] if limit is not None else input_rows
    reference_records = [
        MRNARecord(
            str(row.get("transcript_id")),
            "",
            _normalise_rna(row.get("native_cds")),
            "",
        )
        for row in selected
    ]
    weights = codon_weights_from_reference(reference_records)
    os.makedirs(out_dir, exist_ok=True)
    outputs_path = os.path.join(out_dir, "cds_outputs.jsonl")
    failures_path = os.path.join(out_dir, "failures.jsonl")
    state_path = os.path.join(out_dir, "run_state.json")
    summary_path = os.path.join(out_dir, "summary.json")
    progress_path = progress_jsonl or os.path.join(
        out_dir, "progress.jsonl"
    )
    config = {
        "input_pack_summary_sha256": _sha256_file(input_pack_summary),
        "beam_size": int(beam_size),
        "num_iters": int(num_iters),
        "num_runs": int(num_runs),
        "learning_rate": float(learning_rate),
        "epsilon": float(epsilon),
        "rescue_beam_size": (
            int(rescue_beam_size)
            if rescue_beam_size is not None
            else None
        ),
    }
    existing: dict[str, dict[str, object]] = {}
    if resume and os.path.exists(state_path):
        state = _load_json(state_path)
        state_config = state.get("config")
        legacy_config = dict(config)
        legacy_config.pop("rescue_beam_size")
        if state_config not in (config, legacy_config):
            raise ValueError("resume state config/input-pack mismatch")
        for row in _load_jsonl(outputs_path):
            transcript_id = str(row.get("transcript_id"))
            if transcript_id:
                existing[transcript_id] = row
    else:
        for path in (
            outputs_path,
            failures_path,
            progress_path,
            summary_path,
        ):
            if os.path.exists(path):
                os.remove(path)
    _write_json(
        {
            "artifact_kind": "external_ensembledesign_run_state",
            "config": config,
            "n_inputs": len(selected),
        },
        state_path,
    )
    pending = [
        (index, row)
        for index, row in enumerate(selected)
        if str(row.get("transcript_id")) not in existing
    ]
    append_lock = threading.Lock()
    _append_jsonl(
        {
            "time": time.time(),
            "event": "start_or_resume",
            "n_inputs": len(selected),
            "n_completed_existing": len(existing),
            "n_pending": len(pending),
            "workers": int(workers),
            "config": config,
        },
        progress_path,
    )
    outputs_by_id = dict(existing)
    failures: list[dict[str, object]] = []

    def execute(
        index: int,
        row: Mapping[str, object],
    ) -> tuple[int, dict[str, object]]:
        execution_start = time.perf_counter()
        try:
            _result_index, output = _run_one(
                index,
                row,
                executable=executable,
                timeout_s=timeout_s,
                beam_size=beam_size,
                num_iters=num_iters,
                num_runs=num_runs,
                learning_rate=learning_rate,
                epsilon=epsilon,
                codon_weights=weights,
            )
            primary_error = None
            effective_beam_size = int(beam_size)
        except Exception as primary_exc:
            if (
                rescue_beam_size is None
                or int(rescue_beam_size) == int(beam_size)
            ):
                raise
            primary_error = (
                f"{type(primary_exc).__name__}: {primary_exc}"
            )
            try:
                _result_index, output = _run_one(
                    index,
                    row,
                    executable=executable,
                    timeout_s=timeout_s,
                    beam_size=int(rescue_beam_size),
                    num_iters=num_iters,
                    num_runs=num_runs,
                    learning_rate=learning_rate,
                    epsilon=epsilon,
                    codon_weights=weights,
                )
            except Exception as rescue_exc:
                raise RuntimeError(
                    f"primary beam={int(beam_size)} failed: "
                    f"{primary_error}; rescue beam="
                    f"{int(rescue_beam_size)} failed: "
                    f"{type(rescue_exc).__name__}: {rescue_exc}"
                ) from rescue_exc
            effective_beam_size = int(rescue_beam_size)

        postprocessing = output.get("postprocessing")
        postprocessing = (
            dict(postprocessing)
            if isinstance(postprocessing, Mapping)
            else {}
        )
        postprocessing.update(
            {
                "primary_beam_size": int(beam_size),
                "effective_beam_size": effective_beam_size,
                "beam_rescue_used": primary_error is not None,
                "primary_error": primary_error,
            }
        )
        output["postprocessing"] = postprocessing
        output["wall_clock_s"] = float(
            time.perf_counter() - execution_start
        )
        return _result_index, output

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        future_map = {
            pool.submit(execute, index, row): (index, row)
            for index, row in pending
        }
        completed_now = 0
        for future in as_completed(future_map):
            index, source = future_map[future]
            try:
                _result_index, output = future.result()
                transcript_id = str(output.get("transcript_id"))
                outputs_by_id[transcript_id] = output
                _append_jsonl(output, outputs_path, append_lock)
            except Exception as exc:
                failure = {
                    "record_index": index,
                    "transcript_id": source.get("transcript_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                _append_jsonl(failure, failures_path, append_lock)
            completed_now += 1
            if completed_now % 5 == 0 or completed_now == len(pending):
                _append_jsonl(
                    {
                        "time": time.time(),
                        "event": "progress",
                        "completed_now": completed_now,
                        "n_pending_at_start": len(pending),
                        "n_outputs_total": len(outputs_by_id),
                        "n_failures_now": len(failures),
                    },
                    progress_path,
                    append_lock,
                )

    order = {
        str(row.get("transcript_id")): index
        for index, row in enumerate(selected)
    }
    outputs = sorted(
        outputs_by_id.values(),
        key=lambda row: order.get(str(row.get("transcript_id")), len(order)),
    )
    _write_jsonl(outputs, outputs_path)
    elapsed_s = float(time.perf_counter() - start)
    n_historical_failed_adapter_attempts = (
        len(_load_jsonl(failures_path))
        if os.path.exists(failures_path)
        else 0
    )
    n_beam_rescued = sum(
        bool(
            row.get("postprocessing", {}).get("beam_rescue_used")
        )
        for row in outputs
        if isinstance(row.get("postprocessing"), Mapping)
    )

    def mean(key: str) -> float:
        values = [
            float(row[key])
            for row in outputs
            if isinstance(row.get(key), (int, float))
        ]
        return float(sum(values) / len(values)) if values else 0.0

    pack_outputs = pack.get("outputs", {})
    pack_outputs = pack_outputs if isinstance(pack_outputs, Mapping) else {}
    dataset = pack.get("dataset", {})
    dataset = dataset if isinstance(dataset, Mapping) else {}
    faithful = bool(
        n_beam_rescued == 0
        and _paper_default(
            beam_size=beam_size,
            num_iters=num_iters,
            num_runs=num_runs,
            learning_rate=learning_rate,
            epsilon=epsilon,
        )
    )
    summary: dict[str, object] = {
        "artifact_kind": "external_sota_real_run_summary",
        "claim_policy": CLAIM_POLICY,
        "model_name": MODEL_NAME,
        "task_family": TASK_FAMILY,
        "protocol_fidelity": (
            "official_code_paper_default_beam200_iter30_runs20"
            if faithful
            else (
                f"official_code_budgeted_beam{int(beam_size)}_"
                f"iter{int(num_iters)}_runs{int(num_runs)}_"
                "vs_paper_default_beam200_iter30_runs20"
                + (
                    f"_with_beam{int(rescue_beam_size)}_"
                    f"search_error_rescue_{n_beam_rescued}_rows"
                    if n_beam_rescued
                    and rescue_beam_size is not None
                    else ""
                )
            )
        ),
        "protocol_fidelity_sufficient_for_sota_reproduction": faithful,
        "input_pack": {
            "summary_sha256": _sha256_file(input_pack_summary),
            "cds_protein_jsonl_sha256": pack_outputs.get(
                "cds_protein_jsonl_sha256"
            ),
            "utr5_jsonl_sha256": pack_outputs.get("utr5_jsonl_sha256"),
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
            **config,
        },
        "hardware": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "executable": {
            "path": os.path.abspath(executable),
            "version": _executable_version(executable),
            "sha256": _sha256_file(executable),
        },
        "adapter": {
            "module": (
                "mrna_editflow.baselines.external_ensembledesign_adapter"
            ),
            "resume_granularity": "one input record",
            "terminal_stop_policy": (
                "append native terminal stop when official output omits stop"
            ),
            "ensemble_energy_scope": (
                "official EFE excludes adapter-appended terminal stop"
            ),
            "failure_log_semantics": (
                "failures.jsonl is append-only attempt history; "
                "summary.n_failures counts unresolved input rows"
            ),
        },
        "n_inputs": len(selected),
        "n_outputs": len(outputs),
        "n_failures": len(selected) - len(outputs),
        "n_historical_failed_adapter_attempts": (
            n_historical_failed_adapter_attempts
        ),
        "n_beam_rescued": int(n_beam_rescued),
        "beam_rescue_fraction": (
            float(n_beam_rescued / len(outputs)) if outputs else 0.0
        ),
        "rescue_beam_size": (
            int(rescue_beam_size)
            if rescue_beam_size is not None
            else None
        ),
        "mean_wall_clock_s": mean("wall_clock_s"),
        "valid_cds_fraction": (
            float(
                sum(bool(row.get("valid_cds")) for row in outputs)
                / len(outputs)
            )
            if outputs
            else 0.0
        ),
        "protein_identity_exact_1_fraction": (
            float(
                sum(
                    bool(row.get("protein_identity_exact_1"))
                    for row in outputs
                )
                / len(outputs)
            )
            if outputs
            else 0.0
        ),
        "mean_cai": mean("cai"),
        "mean_gc": mean("gc"),
        "mean_gc3": mean("gc3"),
        "mean_ensemble_free_energy": mean("ensemble_free_energy"),
        "mean_codon_usage_kl_vs_native": mean(
            "codon_usage_kl_vs_native"
        ),
        "mean_codon_pair_kl_vs_native": mean(
            "codon_pair_kl_vs_native"
        ),
        "outputs": {
            "cds_outputs_jsonl": outputs_path,
            "cds_outputs_sha256": _sha256_file(outputs_path),
            "failures_jsonl": failures_path,
            "failures_sha256": _sha256_file(failures_path),
            "progress_jsonl": progress_path,
            "run_state_json": state_path,
        },
    }
    _write_json(summary, summary_path)
    _append_jsonl(
        {
            "time": time.time(),
            "event": "complete",
            "n_inputs": len(selected),
            "n_outputs": len(outputs),
            "n_failures": len(selected) - len(outputs),
            "elapsed_s": elapsed_s,
        },
        progress_path,
    )
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack-summary", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--beam-size", type=int, default=100)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--rescue-beam-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-jsonl", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_ensembledesign_adapter(
        input_pack_summary=args.input_pack_summary,
        executable=args.executable,
        out_dir=args.out_dir,
        limit=args.limit,
        workers=args.workers,
        timeout_s=args.timeout_s,
        beam_size=args.beam_size,
        num_iters=args.num_iters,
        num_runs=args.num_runs,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        rescue_beam_size=args.rescue_beam_size,
        resume=args.resume,
        progress_jsonl=args.progress_jsonl,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CLAIM_POLICY", "run_ensembledesign_adapter", "main"]
