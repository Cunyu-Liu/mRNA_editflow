#!/usr/bin/env python3
"""Materialize GSE114002 identity/support rows from every verified raw file.

This is intentionally additive: the legacy D1 adapter already materializes
non-identity designed-library pairs.  This helper only emits identity rows
from that library and observational support rows from the remaining raw
files, while recording invalid/unmapped rows instead of dropping them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd


SEQ_RE = re.compile(r"^[ACGTU]+$", re.IGNORECASE)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize(value: str) -> tuple[str, list[str], str]:
    raw = cell_text(value)
    if not raw:
        return "", [], "EMPTY"
    steps = ["STRIP"]
    upper = raw.upper()
    if upper != raw:
        steps.append("UPPERCASE")
    if "U" in upper:
        steps.append("U_TO_T")
    if not SEQ_RE.fullmatch(upper):
        return "", steps, "INVALID_SYMBOL"
    return upper.replace("U", "T"), steps, "PASS"


def finite_float(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def make_record(record_id, accession, region, source, candidate, labels, metadata):
    if source is None:
        return {
            "record_id": record_id,
            "dataset": "sample2019",
            "accession": accession,
            "region": region,
            "source_sequence": None,
            "candidate_sequence": candidate,
            "edit_script": [],
            "edit_script_verified": True,
            "edit_distance": 0,
            "n_ins": 0,
            "n_del": 0,
            "n_sub": 0,
            "path_ambiguity": 1,
            "labels": labels,
            "metadata": metadata,
        }
    return {
        "record_id": record_id,
        "dataset": "sample2019",
        "accession": accession,
        "region": region,
        "source_sequence": source,
        "candidate_sequence": candidate,
        "edit_script": [],
        "edit_script_verified": True,
        "edit_distance": 0,
        "n_ins": 0,
        "n_del": 0,
        "n_sub": 0,
        "path_ambiguity": 1,
        "labels": labels,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dispositions", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.dispositions.parent.mkdir(parents=True, exist_ok=True)
    emitted = {"identity": 0, "observational": 0}
    dispositions = {}

    with args.output.open("w", encoding="utf-8") as out, args.dispositions.open(
        "w", encoding="utf-8"
    ) as rej:
        for path in sorted(args.raw_dir.glob("*.csv.gz")):
            if ".corrupt." in path.name:
                dispositions["CORRUPT_BACKUP_EXCLUDED"] = dispositions.get(
                    "CORRUPT_BACKUP_EXCLUDED", 0
                ) + 1
                continue
            df = pd.read_csv(path, compression="infer", low_memory=False)
            is_designed = path.name == "GSM3130443_designed_library.csv.gz"
            for zero_based, row in df.iterrows():
                locator = f"{path.name}#row={int(zero_based) + 2}"
                raw_utr = cell_text(row.get("utr"))
                candidate, candidate_steps, candidate_status = normalize(raw_utr)
                raw_mother = cell_text(row.get("mother")) if is_designed else ""
                source, source_steps, source_status = normalize(raw_mother) if is_designed else ("", [], "NOT_APPLICABLE")
                base = {
                    "source_file": path.name,
                    "source_row_locator": locator,
                    "record_type": "identity" if is_designed and source and candidate and source == candidate else "observational",
                    "adapter_id": "GSE114002_ALL_FILES_IDENTITY_SUPPORT_V1",
                    "adapter_config_sha256": "RAW_CELL_STRICT_ACGTU_NORMALIZE_V1",
                    "raw_candidate_sequence_sha256": sha256_text(raw_utr) if raw_utr else None,
                    "raw_source_sequence_sha256": sha256_text(raw_mother) if raw_mother else None,
                    "candidate_normalization_steps": candidate_steps,
                    "source_normalization_steps": source_steps,
                    "candidate_alphabet_status": candidate_status,
                    "source_alphabet_status": source_status,
                }
                if candidate_status != "PASS" or (is_designed and source_status not in ("PASS", "EMPTY")):
                    reason = "INVALID_SYMBOL" if "INVALID_SYMBOL" in (candidate_status, source_status) else "EMPTY_SEQUENCE"
                    row_payload = {
                        "accession": "GSE114002",
                        "source_file": path.name,
                        "source_row_locator": locator,
                        "disposition": "QUARANTINED",
                        "reason": reason,
                        "candidate_alphabet_status": candidate_status,
                        "source_alphabet_status": source_status,
                    }
                    rej.write(json.dumps(row_payload, sort_keys=True) + "\n")
                    dispositions[reason] = dispositions.get(reason, 0) + 1
                    continue
                label = finite_float(row.get("rl"))
                labels = {"rl": label} if label is not None else {}
                rid = f"GSE114002_FULL_{path.stem}_{int(zero_based)}"
                if is_designed and source and candidate:
                    if source != candidate:
                        # Non-identity designed rows are already emitted by the
                        # legacy designed-library adapter; record the unit here
                        # for reconciliation without duplicating the pair.
                        dispositions["DESIGNED_NON_IDENTITY_ALREADY_MATERIALIZED"] = dispositions.get(
                            "DESIGNED_NON_IDENTITY_ALREADY_MATERIALIZED", 0
                        ) + 1
                        continue
                    rec = make_record(rid, "GSE114002", "5'UTR", source, candidate, labels, base)
                    emitted["identity"] += 1
                else:
                    rec = make_record(rid, "GSE114002", "5'UTR", None, candidate, labels, base)
                    emitted["observational"] += 1
                out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "adapter_id": "GSE114002_ALL_FILES_IDENTITY_SUPPORT_V1",
        "emitted": emitted,
        "dispositions": dispositions,
        "output": str(args.output),
        "dispositions_path": str(args.dispositions),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
