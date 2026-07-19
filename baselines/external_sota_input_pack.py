"""Build standardized input packs for external SOTA baselines.

The external SOTA dry-run answers "is the executable available?". This module
answers the next reproducibility question: "what exact inputs should that
executable consume once it is installed?" It writes CDS/protein-conditioned and
5'UTR-only JSONL input packs with dataset SHA, split metadata, expected output
schemas and claim boundaries.

No external executable is called here and no external metric is fabricated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from mrna_editflow.core.constants import is_valid_cds, translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl


CLAIM_POLICY = (
    "External SOTA input packs are reproducibility/preflight artifacts only. "
    "They define the exact input rows and expected output schema for external "
    "tools, but do not imply that LinearDesign, EnsembleDesign, codonGPT, "
    "Prot2RNA, UTailoR or UTRGAN has been executed, reproduced, or beaten."
)

CDS_MODELS = ("LinearDesign", "EnsembleDesign", "codonGPT", "Prot2RNA")
UTR_MODELS = ("UTailoR", "UTRGAN")
EXTERNAL_BENCHMARK_MODELS = CDS_MODELS + UTR_MODELS


@dataclass(frozen=True)
class ExternalInputPackPaths:
    """Output paths written by :func:`build_external_sota_input_pack`."""

    summary_json: str
    table_md: str
    cds_protein_jsonl: str
    utr5_jsonl: str
    metric_schema_json: str


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_lengths(record: MRNARecord) -> dict[str, int]:
    return {
        "five_utr_nt": len(record.five_utr),
        "cds_nt": len(record.cds),
        "three_utr_nt": len(record.three_utr),
        "full_nt": len(record.seq),
    }


def _gc_fraction(seq: str) -> float:
    """Return GC fraction without importing heavy metric dependencies."""
    return float((seq.count("G") + seq.count("C")) / len(seq)) if seq else 0.0


def _cds_row(record: MRNARecord, index: int, split_name: str, seed: Optional[int]) -> dict[str, object]:
    protein_with_stop = translate(record.cds)
    return {
        "row_kind": "cds_protein_conditioned_external_input",
        "record_index": index,
        "transcript_id": record.transcript_id,
        "species": record.species,
        "split_name": split_name,
        "seed": seed,
        "protein_target": protein_with_stop[:-1] if protein_with_stop.endswith("*") else protein_with_stop,
        "protein_target_with_stop": protein_with_stop,
        "native_cds": record.cds,
        "five_utr_context": record.five_utr,
        "three_utr_context": record.three_utr,
        "native_full_transcript": record.seq,
        "lengths": _record_lengths(record),
        "native_cds_gc": _gc_fraction(record.cds),
        "native_full_gc": _gc_fraction(record.seq),
        "required_hard_constraints": {
            "valid_cds": True,
            "protein_identity_exact_1": True,
            "reading_frame_preserved": True,
            "terminal_stop_required": True,
            "no_internal_stop": True,
        },
        "expected_models": list(CDS_MODELS),
    }


def _utr5_row(record: MRNARecord, index: int, split_name: str, seed: Optional[int]) -> dict[str, object]:
    return {
        "row_kind": "utr5_external_input",
        "record_index": index,
        "transcript_id": record.transcript_id,
        "species": record.species,
        "split_name": split_name,
        "seed": seed,
        "native_five_utr": record.five_utr,
        "fixed_cds_context": record.cds,
        "fixed_three_utr_context": record.three_utr,
        "native_full_transcript": record.seq,
        "lengths": _record_lengths(record),
        "native_5utr_gc": _gc_fraction(record.five_utr),
        "native_full_gc": _gc_fraction(record.seq),
        "required_hard_constraints": {
            "cds_unchanged": True,
            "three_utr_unchanged": True,
            "protein_identity_exact_1": True,
            "reading_frame_preserved": True,
        },
        "expected_models": list(UTR_MODELS),
    }


def external_metric_schema() -> dict[str, object]:
    """Return the required output schema for future real external adapters."""
    return {
        "artifact_kind": "external_sota_metric_schema",
        "claim_policy": CLAIM_POLICY,
        "cds_protein_conditioned": {
            "models": list(CDS_MODELS),
            "required_output_jsonl_fields": [
                "transcript_id",
                "model_name",
                "designed_cds",
                "wall_clock_s",
                "valid_cds",
                "protein_identity",
                "protein_identity_exact_1",
                "cai",
                "gc",
                "gc3",
                "codon_usage_kl_vs_native",
                "codon_pair_kl_vs_native",
            ],
            "optional_output_jsonl_fields": [
                "mfe",
                "ensemble_free_energy",
                "structure_proxy",
                "reward",
                "notes",
            ],
            "required_summary_fields": [
                "n_inputs",
                "n_outputs",
                "n_failures",
                "mean_wall_clock_s",
                "valid_cds_fraction",
                "protein_identity_exact_1_fraction",
                "mean_cai",
                "mean_gc",
                "mean_gc3",
            ],
        },
        "utr5_only": {
            "models": list(UTR_MODELS),
            "required_output_jsonl_fields": [
                "transcript_id",
                "model_name",
                "designed_five_utr",
                "wall_clock_s",
                "cds_unchanged",
                "three_utr_unchanged",
                "protein_identity_exact_1",
                "te_proxy",
                "te_proxy_delta_vs_native",
                "uaug_count",
                "kozak_score",
                "start_accessibility_proxy",
            ],
            "optional_output_jsonl_fields": [
                "mfe",
                "motif_notes",
                "online_service_request_id",
                "notes",
            ],
            "required_summary_fields": [
                "n_inputs",
                "n_outputs",
                "n_failures",
                "mean_wall_clock_s",
                "cds_unchanged_fraction",
                "three_utr_unchanged_fraction",
                "protein_identity_exact_1_fraction",
                "mean_te_proxy_delta_vs_native",
            ],
        },
        "required_real_run_metadata": [
            "input_pack.summary_sha256",
            "input_pack.cds_protein_jsonl_sha256",
            "input_pack.utr5_jsonl_sha256",
            "dataset.records_jsonl_sha256",
            "dataset.split_name",
            "dataset.seed",
            "runtime.elapsed_s",
            "hardware",
            "executable.path",
            "executable.version",
        ],
    }


def _write_jsonl(rows: Sequence[Mapping[str, object]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def _write_json(payload: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def _write_table(summary: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dataset = summary.get("dataset", {})
    dataset = dataset if isinstance(dataset, Mapping) else {}
    outputs = summary.get("outputs", {})
    outputs = outputs if isinstance(outputs, Mapping) else {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# External SOTA Input Pack\n\n")
        fh.write(f"- Claim policy: {summary.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Dataset: split=`{dataset.get('split_name')}`, seed=`{dataset.get('seed')}`, "
            f"records=`{dataset.get('record_count_effective')}` / `{dataset.get('record_count_total')}`, "
            f"sha256=`{dataset.get('records_jsonl_sha256')}`\n"
        )
        fh.write(
            f"- CDS/protein rows: `{summary.get('n_cds_protein_rows')}`; "
            f"UTR5 rows: `{summary.get('n_utr5_rows')}`; skipped invalid CDS: `{summary.get('n_skipped_invalid_cds')}`\n\n"
        )
        fh.write("| Pack | Models | Path | SHA256 | Required real-run outputs |\n")
        fh.write("|---|---|---|---|---|\n")
        fh.write(
            f"| CDS/protein-conditioned | `{', '.join(CDS_MODELS)}` | "
            f"`{outputs.get('cds_protein_jsonl')}` | `{outputs.get('cds_protein_jsonl_sha256')}` | "
            "`designed_cds`, `wall_clock_s`, `protein_identity`, `CAI`, `GC/GC3`, codon KL |\n"
        )
        fh.write(
            f"| 5'UTR-only | `{', '.join(UTR_MODELS)}` | "
            f"`{outputs.get('utr5_jsonl')}` | `{outputs.get('utr5_jsonl_sha256')}` | "
            "`designed_five_utr`, `TE proxy`, `uAUG`, `Kozak`, unchanged CDS/3'UTR |\n"
        )
    return path


def build_external_sota_input_pack(
    *,
    records_jsonl: str,
    out_dir: str,
    limit: Optional[int] = None,
    split_name: str = "unspecified",
    seed: Optional[int] = None,
) -> dict[str, object]:
    """Build external SOTA input packs and return the summary payload."""
    start = time.perf_counter()
    records = load_records_jsonl(records_jsonl)
    selected = records[:limit] if limit is not None else records

    cds_rows: list[dict[str, object]] = []
    utr_rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for index, record in enumerate(selected):
        if not is_valid_cds(record.cds):
            skipped.append(
                {
                    "record_index": index,
                    "transcript_id": record.transcript_id,
                    "reason": "invalid_cds",
                }
            )
            continue
        cds_rows.append(_cds_row(record, index, split_name, seed))
        utr_rows.append(_utr5_row(record, index, split_name, seed))

    os.makedirs(out_dir, exist_ok=True)
    paths = ExternalInputPackPaths(
        summary_json=os.path.join(out_dir, "summary.json"),
        table_md=os.path.join(out_dir, "table.md"),
        cds_protein_jsonl=os.path.join(out_dir, "cds_protein_inputs.jsonl"),
        utr5_jsonl=os.path.join(out_dir, "utr5_inputs.jsonl"),
        metric_schema_json=os.path.join(out_dir, "metric_schema.json"),
    )
    _write_jsonl(cds_rows, paths.cds_protein_jsonl)
    _write_jsonl(utr_rows, paths.utr5_jsonl)
    metric_schema = external_metric_schema()
    _write_json(metric_schema, paths.metric_schema_json)

    summary: dict[str, object] = {
        "artifact_kind": "external_sota_input_pack",
        "claim_policy": CLAIM_POLICY,
        "dataset": {
            "records_jsonl": records_jsonl,
            "records_jsonl_exists": os.path.exists(records_jsonl),
            "records_jsonl_sha256": _sha256_file(records_jsonl),
            "record_count_total": len(records),
            "record_count_effective": len(selected),
            "split_name": split_name,
            "seed": seed,
            "limit": limit,
        },
        "models": {
            "cds_protein_conditioned": list(CDS_MODELS),
            "utr5_only": list(UTR_MODELS),
        },
        "n_cds_protein_rows": len(cds_rows),
        "n_utr5_rows": len(utr_rows),
        "n_skipped_invalid_cds": len(skipped),
        "skipped_records": skipped[:50],
        "outputs": asdict(paths),
        "metric_schema": metric_schema,
        "elapsed_s": float(time.perf_counter() - start),
        "ready_for_external_real_run": bool(cds_rows and utr_rows),
        "ready_for_external_sota_claim": False,
    }
    _write_json(summary, paths.summary_json)
    # Add self-SHA after the first write, then rewrite once.
    summary["outputs"]["summary_json_sha256"] = _sha256_file(paths.summary_json)
    summary["outputs"]["cds_protein_jsonl_sha256"] = _sha256_file(paths.cds_protein_jsonl)
    summary["outputs"]["utr5_jsonl_sha256"] = _sha256_file(paths.utr5_jsonl)
    summary["outputs"]["metric_schema_json_sha256"] = _sha256_file(paths.metric_schema_json)
    _write_json(summary, paths.summary_json)
    summary["outputs"]["summary_json_sha256"] = _sha256_file(paths.summary_json)
    _write_json(summary, paths.summary_json)
    _write_table(summary, paths.table_md)
    summary["outputs"]["table_md_sha256"] = _sha256_file(paths.table_md)
    _write_json(summary, paths.summary_json)
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-name", default="unspecified")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = build_external_sota_input_pack(
        records_jsonl=args.records_jsonl,
        out_dir=args.out_dir,
        limit=args.limit,
        split_name=args.split_name,
        seed=args.seed,
    )
    print(json.dumps({"summary_json": summary["outputs"]["summary_json"], "table_md": summary["outputs"]["table_md"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "CDS_MODELS",
    "UTR_MODELS",
    "EXTERNAL_BENCHMARK_MODELS",
    "ExternalInputPackPaths",
    "build_external_sota_input_pack",
    "external_metric_schema",
    "main",
]
