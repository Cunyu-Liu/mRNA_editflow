"""Run official UTailoR on its declared 25-100 nt input domain."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Mapping, Optional, Sequence

from mrna_editflow.baselines.external_utrgan_adapter import (
    _load_json,
    _load_jsonl,
    _normalise_rna,
    _resolve_pack_file,
    _sha256_file,
    _write_json,
    _write_jsonl,
)
from mrna_editflow.eval.metrics import edit_distance
from mrna_editflow.eval.oracle import score_utr


CLAIM_POLICY = (
    "This artifact contains measured official UTailoR outputs on the strict "
    "25-100 nt domain declared by its public web tool. It uses a shared MEF "
    "proxy oracle for comparison and is not wet-lab evidence or proof of MEF "
    "superiority. The public archives contain no explicit license file, so "
    "redistribution rights are not assumed."
)
MODEL_NAME = "UTailoR"
TASK_FAMILY = "utr5_only"
ELIGIBILITY_POLICY = "official_input_length_25_100_strict"


def _version(executable: str) -> str:
    proc = subprocess.run(
        [executable, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    return proc.stdout.strip() or f"unknown exit={proc.returncode}"


def run_utailor_adapter(
    *,
    input_pack_summary: str,
    executable: str,
    out_dir: str,
    limit: Optional[int] = None,
    timeout_s: float = 1800.0,
) -> dict[str, object]:
    """Run one batched official UTailoR inference over eligible input rows."""
    start = time.perf_counter()
    pack = _load_json(input_pack_summary)
    utr_inputs_path = _resolve_pack_file(
        input_pack_summary,
        pack,
        "utr5_jsonl",
        "utr5_inputs.jsonl",
    )
    all_rows = _load_jsonl(utr_inputs_path)
    selected = all_rows[:limit] if limit is not None else all_rows
    eligible = [
        row
        for row in selected
        if 25 <= len(_normalise_rna(row.get("native_five_utr"))) <= 100
    ]
    eligible_ids = {str(row.get("transcript_id")) for row in eligible}
    ineligible = [
        {
            "transcript_id": row.get("transcript_id"),
            "native_utr_length": len(
                _normalise_rna(row.get("native_five_utr"))
            ),
            "reason": "outside_official_25_100_nt_domain",
        }
        for row in selected
        if str(row.get("transcript_id")) not in eligible_ids
    ]
    if not eligible:
        raise ValueError("UTailoR input selection has no eligible 25-100 nt rows")

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "official_raw_output.json")
    stdout_path = os.path.join(out_dir, "official_stdout.log")
    xlsx_path = os.path.join(out_dir, "official_report.xlsx")
    task_id = f"mef_{os.getpid()}_{int(time.time())}"
    with tempfile.TemporaryDirectory(prefix="mef_utailor_") as tmp:
        fasta_path = os.path.join(tmp, "eligible.fasta")
        temp_raw_path = os.path.join(tmp, "raw.json")
        with open(fasta_path, "w", encoding="utf-8") as fh:
            for row in eligible:
                transcript_id = str(row.get("transcript_id"))
                utr = _normalise_rna(row.get("native_five_utr"))
                fh.write(f">{transcript_id}\n{utr}\n")
        proc = subprocess.run(
            [
                executable,
                "--input-fasta",
                fasta_path,
                "--output-json",
                temp_raw_path,
                "--task-id",
                task_id,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"UTailoR exit={proc.returncode}: {proc.stdout[-4000:]}"
            )
        raw = _load_json(temp_raw_path)
        shutil.copyfile(temp_raw_path, raw_path)
        official_xlsx = raw.get("official_xlsx")
        if isinstance(official_xlsx, str) and os.path.exists(official_xlsx):
            shutil.copyfile(official_xlsx, xlsx_path)
    elapsed_s = float(time.perf_counter() - start)
    with open(stdout_path, "w", encoding="utf-8") as fh:
        fh.write(proc.stdout)

    raw_records = raw.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("UTailoR raw output records must be a list")
    raw_by_id = {
        str(row.get("Sequence name")): row
        for row in raw_records
        if isinstance(row, Mapping)
    }
    if set(raw_by_id) != eligible_ids:
        raise ValueError(
            "UTailoR output coverage mismatch for eligible transcript ids"
        )

    wall_per_row = elapsed_s / len(eligible)
    outputs: list[dict[str, object]] = []
    for source in eligible:
        transcript_id = str(source.get("transcript_id"))
        official = raw_by_id[transcript_id]
        native_utr = _normalise_rna(source.get("native_five_utr"))
        designed_utr = _normalise_rna(official.get("Optimized UTR"))
        cds = _normalise_rna(source.get("fixed_cds_context"))
        native_score = score_utr(native_utr, cds[:12])
        candidate_score = score_utr(designed_utr, cds[:12])
        features = candidate_score.get("features", {})
        features = features if isinstance(features, Mapping) else {}
        utr_distance = int(edit_distance(designed_utr, native_utr))
        outputs.append(
            {
                "transcript_id": transcript_id,
                "model_name": MODEL_NAME,
                "designed_five_utr": designed_utr,
                "wall_clock_s": float(wall_per_row),
                "cds_unchanged": True,
                "three_utr_unchanged": True,
                "protein_identity_exact_1": True,
                "te_proxy": float(candidate_score["ensemble_te"]),
                "te_proxy_delta_vs_native": float(
                    candidate_score["ensemble_te"]
                    - native_score["ensemble_te"]
                ),
                "uaug_count": float(features.get("uaug_count", 0.0)),
                "kozak_score": float(features.get("kozak", 0.0)),
                "start_accessibility_proxy": float(
                    features.get("start_accessibility", 0.0)
                ),
                "external_utailor_original_rl": float(
                    official.get("Original RL", 0.0)
                ),
                "external_utailor_optimized_rl": float(
                    official.get("Optimized RL", 0.0)
                ),
                "external_utailor_optimized_rl_50nt": float(
                    official.get("Optimized RL_50nt", 0.0)
                ),
                "external_utailor_increased_rl": float(
                    official.get("Increased RL", 0.0)
                ),
                "utr_edit_distance_vs_native": utr_distance,
                "normalized_utr_edit_distance_vs_native": float(
                    utr_distance / max(len(designed_utr), len(native_utr), 1)
                ),
                "designed_utr_length": len(designed_utr),
                "native_utr_length": len(native_utr),
                "utr_length_delta": len(designed_utr) - len(native_utr),
                "exact_native_utr_match": bool(designed_utr == native_utr),
                "native_te_proxy": float(native_score["ensemble_te"]),
                "native_five_utr": native_utr,
                "fixed_cds_context": cds,
                "fixed_three_utr_context": _normalise_rna(
                    source.get("fixed_three_utr_context")
                ),
            }
        )

    outputs_path = os.path.join(out_dir, "utr5_outputs.jsonl")
    ineligible_path = os.path.join(out_dir, "ineligible_inputs.jsonl")
    _write_jsonl(outputs, outputs_path)
    _write_jsonl(ineligible, ineligible_path)

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
    summary: dict[str, object] = {
        "artifact_kind": "external_sota_real_run_summary",
        "claim_policy": CLAIM_POLICY,
        "model_name": MODEL_NAME,
        "task_family": TASK_FAMILY,
        "protocol_fidelity": (
            "official_public_code_and_weights_strict_25_100_nt_shared_public_"
            "subset_not_paper_dataset"
        ),
        "protocol_fidelity_sufficient_for_sota_reproduction": False,
        "eligibility": {
            "policy": ELIGIBILITY_POLICY,
            "min_length": 25,
            "max_length": 100,
            "n_total_inputs": len(selected),
            "n_eligible_inputs": len(eligible),
            "n_ineligible_inputs": len(ineligible),
            "ineligible_inputs_jsonl": ineligible_path,
            "ineligible_inputs_sha256": _sha256_file(ineligible_path),
        },
        "license": {
            "status": "not_present_in_public_archives",
            "redistribution_rights_assumed": False,
            "internal_research_execution_only": True,
        },
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
            "batch_size": len(eligible),
            "timeout_s": float(timeout_s),
        },
        "hardware": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "configured_cuda_visible_devices": os.environ.get(
                "UTAILOR_CUDA_VISIBLE_DEVICES"
            ),
        },
        "executable": {
            "path": os.path.abspath(executable),
            "version": _version(executable),
            "sha256": _sha256_file(executable),
        },
        "adapter": {
            "module": "mrna_editflow.baselines.external_utailor_adapter",
            "official_runner_module": (
                "mrna_editflow.baselines.external_utailor_runner"
            ),
            "fixed_regions": ["CDS", "3UTR"],
            "shared_proxy_oracle": "mrna_editflow.eval.oracle.score_utr",
        },
        "n_inputs": len(selected),
        "n_eligible_inputs": len(eligible),
        "n_ineligible_inputs": len(ineligible),
        "n_outputs": len(outputs),
        "n_failures": 0,
        "mean_wall_clock_s": mean("wall_clock_s"),
        "cds_unchanged_fraction": 1.0,
        "three_utr_unchanged_fraction": 1.0,
        "protein_identity_exact_1_fraction": 1.0,
        "mean_te_proxy_delta_vs_native": mean(
            "te_proxy_delta_vs_native"
        ),
        "mean_te_proxy": mean("te_proxy"),
        "mean_uaug_count": mean("uaug_count"),
        "mean_kozak_score": mean("kozak_score"),
        "mean_start_accessibility_proxy": mean(
            "start_accessibility_proxy"
        ),
        "mean_external_utailor_original_rl": mean(
            "external_utailor_original_rl"
        ),
        "mean_external_utailor_optimized_rl": mean(
            "external_utailor_optimized_rl"
        ),
        "mean_external_utailor_increased_rl": mean(
            "external_utailor_increased_rl"
        ),
        "mean_utr_edit_distance_vs_native": mean(
            "utr_edit_distance_vs_native"
        ),
        "mean_normalized_utr_edit_distance_vs_native": mean(
            "normalized_utr_edit_distance_vs_native"
        ),
        "mean_designed_utr_length": mean("designed_utr_length"),
        "mean_native_utr_length": mean("native_utr_length"),
        "mean_utr_length_delta": mean("utr_length_delta"),
        "exact_native_utr_match_fraction": float(
            sum(bool(row.get("exact_native_utr_match")) for row in outputs)
            / len(outputs)
        ),
        "unique_designed_utr_fraction": float(
            len({str(row.get("designed_five_utr")) for row in outputs})
            / len(outputs)
        ),
        "outputs": {
            "utr5_outputs_jsonl": outputs_path,
            "utr5_outputs_sha256": _sha256_file(outputs_path),
            "official_raw_output_json": raw_path,
            "official_raw_output_sha256": _sha256_file(raw_path),
            "official_stdout_log": stdout_path,
            "official_stdout_sha256": _sha256_file(stdout_path),
            "official_report_xlsx": (
                xlsx_path if os.path.exists(xlsx_path) else None
            ),
            "official_report_xlsx_sha256": _sha256_file(xlsx_path),
        },
    }
    _write_json(summary, os.path.join(out_dir, "summary.json"))
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack-summary", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_utailor_adapter(
        input_pack_summary=args.input_pack_summary,
        executable=args.executable,
        out_dir=args.out_dir,
        limit=args.limit,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CLAIM_POLICY", "ELIGIBILITY_POLICY", "run_utailor_adapter", "main"]
