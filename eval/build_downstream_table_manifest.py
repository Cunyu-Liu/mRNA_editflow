"""Build paper-grade manifests for downstream MPRA/stability tables.

This utility is intentionally small and strict. It turns an already-downloaded
CSV/TSV table into a normalized JSONL plus ``*.data_manifest.json`` that can be
audited by :mod:`mrna_editflow.eval.dataset_manifest_audit`. It does not fetch
external data or make real TE/stability claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from mrna_editflow.data.prepare_mpra import prepare_mpra_dataset, read_mpra_table
from mrna_editflow.eval.stability_predictor import prepare_stability_dataset, read_stability_table


CLAIM_POLICY = (
    "Downstream table manifests are data-provenance evidence only. A schema-valid "
    "MPRA or stability table with official split statistics is not a real "
    "TE/stability result until held-out predictor reports, leakage documentation, "
    "and dataset-manifest audits are complete."
)

VALID_DATASETS = {"mpra_te", "stability_half_life"}
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class DownstreamTableManifestResult:
    dataset_name: str
    records_path: str
    manifest_path: str
    n_raw: int
    n_clean: int
    records_sha256: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "records_path": self.records_path,
            "manifest_path": self.manifest_path,
            "n_raw": self.n_raw,
            "n_clean": self.n_clean,
            "records_sha256": self.records_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _length_stats(values: Sequence[int]) -> dict[str, object]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _numeric_stats(values: Sequence[float]) -> dict[str, object]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(min(vals)),
        "max": float(max(vals)),
        "mean": float(sum(vals) / len(vals)),
    }


def _split_counts(samples: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0, "other": 0, "missing": 0}
    for sample in samples:
        split = str(sample.get("split", "") or "").strip().lower()
        if not split:
            counts["missing"] += 1
        elif split in VALID_SPLITS:
            counts[split] += 1
        else:
            counts["other"] += 1
    return counts


def _official_split_ready(counts: Mapping[str, int]) -> bool:
    return bool(
        counts.get("train", 0) > 0
        and counts.get("val", 0) > 0
        and counts.get("test", 0) > 0
        and counts.get("other", 0) == 0
        and counts.get("missing", 0) == 0
    )


def _write_jsonl(samples: Sequence[Mapping[str, object]], path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(dict(sample), sort_keys=True) + "\n")
    return path


def _dataset_description(dataset_name: str) -> str:
    if dataset_name == "mpra_te":
        return "External MPRA/TE table normalized for 5'UTR MRL/TE predictor audits."
    if dataset_name == "stability_half_life":
        return "External mRNA stability or half-life table normalized for predictor audits."
    raise ValueError(f"unsupported downstream dataset {dataset_name!r}")


def _target_summary(dataset_name: str, samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if dataset_name == "mpra_te":
        return {
            "target_column": "mrl",
            "mrl": _numeric_stats([float(sample["mrl"]) for sample in samples]),
            "mrl_z": _numeric_stats([float(sample["mrl_z"]) for sample in samples]),
        }
    target_names = sorted({str(sample.get("target_name", "target")) for sample in samples})
    return {
        "target_column": target_names[0] if len(target_names) == 1 else target_names,
        "target": _numeric_stats([float(sample["target"]) for sample in samples]),
    }


def _records_summary(dataset_name: str, samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "n_records": len(samples),
        "sequence_len": _length_stats([len(str(sample.get("sequence", ""))) for sample in samples]),
        "split_counts": _split_counts(samples),
        "target_summary": _target_summary(dataset_name, samples),
    }


def _load_downstream_samples(
    dataset_name: str,
    input_path: str,
    *,
    target_col: Optional[str] = None,
) -> tuple[list[dict[str, object]], int, str, bool]:
    if dataset_name == "mpra_te":
        raw_rows = read_mpra_table(input_path)
        samples = prepare_mpra_dataset(rows=raw_rows)
        return samples, len(raw_rows), "mrl", _official_split_ready(_split_counts(raw_rows))
    if dataset_name == "stability_half_life":
        raw_rows, selected_target = read_stability_table(input_path, target_col=target_col)
        samples, _, official_split_present = prepare_stability_dataset(
            rows=raw_rows,
            target_col=selected_target,
        )
        return samples, len(raw_rows), selected_target, bool(
            official_split_present and _official_split_ready(_split_counts(raw_rows))
        )
    raise ValueError(f"unsupported downstream dataset {dataset_name!r}")


def build_downstream_table_manifest(
    *,
    dataset_name: str,
    input_path: str,
    out_dir: str,
    source_url: str,
    license_text: str = "unknown; verify before publication",
    target_col: Optional[str] = None,
    require_official_split: bool = True,
) -> DownstreamTableManifestResult:
    """Normalize a downstream table and write records + manifest artifacts."""
    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"dataset_name must be one of {sorted(VALID_DATASETS)}")
    if not source_url:
        raise ValueError("source_url is required for a paper-grade downstream manifest")
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    samples, n_raw, selected_target, official_split_ready = _load_downstream_samples(
        dataset_name,
        input_path,
        target_col=target_col,
    )
    split_counts = _split_counts(samples)
    if require_official_split and not official_split_ready:
        raise ValueError(
            "official train/val/test split is required; provide split labels with "
            "at least one train, val, and test row"
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    records_path = out_path / f"{dataset_name}.records.jsonl"
    manifest_path = out_path / f"{dataset_name}.data_manifest.json"
    _write_jsonl(samples, str(records_path))
    records_sha = _sha256_file(str(records_path))
    raw_sha = _sha256_file(input_path)

    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "downstream_table_data_manifest",
        "claim_policy": CLAIM_POLICY,
        "dataset": {
            "name": dataset_name,
            "description": _dataset_description(dataset_name),
            "license": license_text,
            "registry_url": source_url,
            "files": [
                {
                    "filename": os.path.basename(input_path),
                    "path": os.path.abspath(input_path),
                    "exists": True,
                    "size_bytes": os.path.getsize(input_path),
                    "sha256": raw_sha,
                    "public_url": source_url,
                    "expected_sha256": None,
                }
            ],
        },
        "records_path": str(records_path),
        "records_sha256": records_sha,
        "raw_summary": {
            **_records_summary(dataset_name, samples),
            "n_records": n_raw,
            "raw_file_sha256": raw_sha,
        },
        "clean_summary": _records_summary(dataset_name, samples),
        "cleaning_drop_counts": {
            "total": n_raw,
            "kept": len(samples),
            "invalid_rows": 0,
            "dropped_rows": n_raw - len(samples),
        },
        "split_stats": {
            "split_counts": split_counts,
            "official_split_ready": official_split_ready,
            "required_official_split": require_official_split,
            "n_train": split_counts["train"],
            "n_val": split_counts["val"],
            "n_test": split_counts["test"],
        },
        "preprocessing_contract": {
            "sequence_map": "upper(strip_whitespace(sequence)).replace('T','U')",
            "required_columns": ["sequence", selected_target, "split"],
            "target_column": selected_target,
            "split_policy": (
                "official train/val/test split must be present for paper-grade "
                "predictor claims"
            ),
        },
        "ready_for_dataset_manifest_audit": bool(official_split_ready and len(samples) > 0),
        "ready_for_real_te_or_stability_claim": False,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return DownstreamTableManifestResult(
        dataset_name=dataset_name,
        records_path=str(records_path),
        manifest_path=str(manifest_path),
        n_raw=n_raw,
        n_clean=len(samples),
        records_sha256=records_sha,
        manifest_sha256=_sha256_file(str(manifest_path)),
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(VALID_DATASETS), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", default="unknown; verify before publication")
    parser.add_argument("--target-col", default=None)
    parser.add_argument(
        "--allow-missing-official-split",
        action="store_true",
        help="Write a manifest for plumbing only; it will not be paper-grade.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = build_downstream_table_manifest(
        dataset_name=args.dataset,
        input_path=args.input,
        out_dir=args.out_dir,
        source_url=args.source_url,
        license_text=args.license,
        target_col=args.target_col,
        require_official_split=not args.allow_missing_official_split,
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "DownstreamTableManifestResult",
    "build_downstream_table_manifest",
    "main",
]
