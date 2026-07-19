"""Offline MPRA-like 5'UTR task preparation for mRNA-EditFlow.

The public MPRA path accepts simple CSV/TSV files with at least
``sequence,mrl`` columns and an optional official ``split`` column. Tests and
smoke runs do not depend on external data: without an input file this module
derives a small MPRA-like 5'UTR regression set from
``download_mrna.synthesize_corpus``.

Returned samples are plain dictionaries so they can be written as JSONL or fed
directly into lightweight training/evaluation code.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from mrna_editflow.data.download_mrna import synthesize_corpus

_VALID_RNA = frozenset("ACGU")


def normalise_rna_sequence(sequence: str) -> str:
    """Uppercase a DNA/RNA sequence, strip whitespace, and map ``T`` to ``U``."""
    seq = "".join(str(sequence).split()).upper().replace("T", "U")
    if not seq:
        raise ValueError("sequence must be non-empty")
    bad = set(seq) - _VALID_RNA
    if bad:
        raise ValueError(f"sequence contains non-ACGU characters: {sorted(bad)}")
    return seq


def zscore(values: Sequence[float]) -> List[float]:
    """Population z-score with a finite zero-variance fallback."""
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(var)
    if std == 0.0 or not math.isfinite(std):
        return [0.0 for _ in values]
    return [(x - mean) / std for x in values]


def _guess_delimiter(path: Path) -> str:
    if path.suffix.lower() in {".tsv", ".tab"}:
        return "\t"
    if path.suffix.lower() == ".csv":
        return ","
    with open(path, "r", encoding="utf-8", newline="") as fh:
        sample = fh.read(2048)
    return "\t" if sample.count("\t") > sample.count(",") else ","


def read_mpra_table(
    path: str,
    sequence_col: str = "sequence",
    mrl_col: str = "mrl",
    split_col: str = "split",
) -> List[dict]:
    """Read a minimal MPRA CSV/TSV table.

    ``split`` is optional and, when present, is preserved by
    :func:`prepare_mpra_dataset` rather than overwritten by random splitting.
    """
    fp = Path(path)
    delim = _guess_delimiter(fp)
    rows: List[dict] = []
    with open(fp, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = {sequence_col, mrl_col} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            seq = normalise_rna_sequence(row[sequence_col])
            try:
                mrl = float(row[mrl_col])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {i} has non-numeric MRL: {row[mrl_col]!r}") from exc
            split = row.get(split_col, "")
            rows.append(
                {
                    "sample_id": row.get("sample_id") or row.get("id") or f"mpra_{i:05d}",
                    "sequence": seq,
                    "mrl": mrl,
                    "split": split.strip().lower() if split else "",
                    "source": str(fp),
                }
            )
    return rows


def _synthetic_mrl(sequence: str) -> float:
    """Deterministic translation-efficiency proxy for synthetic MPRA labels."""
    gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
    kozak = 1.0 if sequence.endswith("GCCACC") or "GCCACC" in sequence[-12:] else 0.0
    upstream_aug = float(sequence.count("AUG"))
    length_penalty = abs(len(sequence) - 70) / 100.0
    return 2.0 + 0.8 * kozak + 0.6 * gc - 0.35 * upstream_aug - length_penalty


def synthesize_mpra_rows(n: int = 32, seed: int = 0) -> List[dict]:
    """Build a deterministic offline 5'UTR MPRA-like table."""
    rows: List[dict] = []
    for i, rec in enumerate(synthesize_corpus(n, seed=seed)):
        seq = normalise_rna_sequence(rec.five_utr)
        rows.append(
            {
                "sample_id": f"synthetic_mpra_{i:05d}",
                "sequence": seq,
                "mrl": _synthetic_mrl(seq),
                "split": "",
                "source": "synthetic",
                "transcript_id": rec.transcript_id,
            }
        )
    return rows


def _fallback_split(i: int, n: int) -> str:
    frac = (i + 0.5) / max(1, n)
    if frac < 0.8:
        return "train"
    if frac < 0.9:
        return "val"
    return "test"


def prepare_mpra_dataset(
    path: Optional[str] = None,
    rows: Optional[Iterable[Mapping[str, object]]] = None,
    n_synthetic: int = 32,
    seed: int = 0,
    output_jsonl: Optional[str] = None,
) -> List[dict]:
    """Return normalised MPRA samples with finite MRL z-scores.

    Input precedence is ``rows`` > ``path`` > synthetic fallback. Official
    split labels in the input are preserved; otherwise a deterministic
    train/val/test split is assigned by row order.
    """
    if rows is not None:
        raw_rows = [dict(r) for r in rows]
    elif path is not None:
        raw_rows = read_mpra_table(path)
    else:
        raw_rows = synthesize_mpra_rows(n_synthetic, seed=seed)

    prepared: List[dict] = []
    for i, row in enumerate(raw_rows):
        seq = normalise_rna_sequence(str(row["sequence"]))
        mrl = float(row["mrl"])
        split = str(row.get("split", "") or "").strip().lower()
        prepared.append(
            {
                "sample_id": str(row.get("sample_id") or row.get("id") or f"mpra_{i:05d}"),
                "sequence": seq,
                "mrl": mrl,
                "split": split,
                "source": str(row.get("source", "memory")),
            }
        )

    zs = zscore([s["mrl"] for s in prepared])
    n = len(prepared)
    for i, sample in enumerate(prepared):
        sample["mrl_z"] = float(zs[i])
        if not sample["split"]:
            sample["split"] = _fallback_split(i, n)

    if output_jsonl is not None:
        write_jsonl(prepared, output_jsonl)
    return prepared


def write_jsonl(samples: Sequence[Mapping[str, object]], path: str) -> None:
    """Write samples as UTF-8 JSONL."""
    with open(path, "w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(dict(sample), sort_keys=True) + "\n")


__all__ = [
    "normalise_rna_sequence",
    "zscore",
    "read_mpra_table",
    "synthesize_mpra_rows",
    "prepare_mpra_dataset",
    "write_jsonl",
]
