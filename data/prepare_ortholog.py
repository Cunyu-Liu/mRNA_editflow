"""Ortholog-coupling task preparation.

Real ortholog tables can be loaded from local CSV/TSV/JSONL files, but the
default path is fully offline: synthetic homolog variants are generated from
``synthesize_corpus`` or supplied ``MRNARecord`` objects. CDS pairs are
codon-boundary safe and preserve the translated protein.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from mrna_editflow.core.constants import translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.augment import motif_preserving_utr_perturb, synonymously_perturb_cds
from mrna_editflow.data.download_mrna import synthesize_corpus

REGION_5UTR = "5UTR"
REGION_CDS = "CDS"
REGION_3UTR = "3UTR"
ORTHOLOG_REGIONS = (REGION_5UTR, REGION_CDS, REGION_3UTR)


def _normalise(seq: str) -> str:
    s = "".join(str(seq).split()).upper().replace("T", "U")
    bad = set(s) - set("ACGU")
    if bad:
        raise ValueError(f"sequence contains non-ACGU characters: {sorted(bad)}")
    return s


def _region_slice(record: MRNARecord, region: str) -> Tuple[str, int, int]:
    if region == REGION_5UTR:
        return record.five_utr, 0, len(record.five_utr)
    if region == REGION_CDS:
        return record.cds, record.cds_start, record.cds_end
    if region == REGION_3UTR:
        return record.three_utr, record.cds_end, len(record.seq)
    raise ValueError(f"unknown region {region!r}")


def validate_ortholog_pair(pair: Mapping[str, object]) -> bool:
    """Check region consistency and CDS frame/protein safety."""
    region = str(pair.get("region", ""))
    if region not in ORTHOLOG_REGIONS:
        return False
    if str(pair.get("source_region", region)) != region:
        return False
    if str(pair.get("target_region", region)) != region:
        return False

    source_seq = _normalise(str(pair.get("source_seq", "")))
    target_seq = _normalise(str(pair.get("target_seq", "")))
    if not source_seq or not target_seq:
        return False

    if region == REGION_CDS:
        if len(source_seq) % 3 != 0 or len(target_seq) % 3 != 0:
            return False
        if translate(source_seq) != translate(target_seq):
            return False
        for side in ("source", "target"):
            start = pair.get(f"{side}_start")
            end = pair.get(f"{side}_end")
            cds_start = pair.get(f"{side}_cds_start")
            if start is None or end is None or cds_start is None:
                continue
            start_i = int(start)
            end_i = int(end)
            cds_start_i = int(cds_start)
            if (start_i - cds_start_i) % 3 != 0:
                return False
            if (end_i - cds_start_i) % 3 != 0:
                return False
    return True


def make_synthetic_ortholog_pair(record: MRNARecord, region: str, seed: int = 0) -> dict:
    """Create one offline ortholog-like coupling pair from a record."""
    source_seq, start, end = _region_slice(record, region)
    if not source_seq:
        raise ValueError(f"record {record.transcript_id!r} has empty {region}")
    if region == REGION_CDS:
        target_seq = synonymously_perturb_cds(source_seq, edit_fraction=0.12, seed=seed)
    else:
        target_seq = motif_preserving_utr_perturb(source_seq, edit_fraction=0.08, seed=seed)
    target_species = "mouse" if record.species == "human" else "human"
    pair = {
        "pair_id": f"{record.transcript_id}:{region}:synthetic",
        "source_id": record.transcript_id,
        "target_id": f"{record.transcript_id}_{target_species}_ortholog",
        "source_species": record.species,
        "target_species": target_species,
        "region": region,
        "source_region": region,
        "target_region": region,
        "source_start": start,
        "source_end": end,
        "target_start": start,
        "target_end": start + len(target_seq),
        "source_cds_start": record.cds_start,
        "target_cds_start": record.cds_start,
        "source_seq": source_seq,
        "target_seq": target_seq,
    }
    assert validate_ortholog_pair(pair), "synthetic ortholog pair failed validation"
    return pair


def synthesize_ortholog_pairs(
    records: Optional[Iterable[MRNARecord]] = None,
    n_pairs: int = 12,
    seed: int = 0,
) -> List[dict]:
    """Generate offline homolog pairs cycling through 5UTR/CDS/3UTR regions."""
    if records is None:
        records = synthesize_corpus(max(n_pairs, 3), seed=seed)
    pairs: List[dict] = []
    recs = list(records)
    if not recs:
        return []
    i = 0
    while len(pairs) < n_pairs and i < len(recs) * 2:
        record = recs[i % len(recs)]
        region = ORTHOLOG_REGIONS[i % len(ORTHOLOG_REGIONS)]
        try:
            pairs.append(make_synthetic_ortholog_pair(record, region, seed=seed + i))
        except ValueError:
            pass
        i += 1
    return pairs


def _load_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_ortholog_pairs(path: str, validate: bool = True) -> List[dict]:
    """Read local ortholog pairs from JSONL or simple CSV/TSV."""
    fp = Path(path)
    if fp.suffix.lower() == ".jsonl":
        rows = _load_jsonl(fp)
    else:
        delim = "\t" if fp.suffix.lower() in {".tsv", ".tab"} else ","
        with open(fp, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter=delim))

    pairs: List[dict] = []
    for i, row in enumerate(rows):
        pair = dict(row)
        pair.setdefault("pair_id", f"ortholog_{i:05d}")
        pair["region"] = str(pair.get("region", pair.get("source_region", ""))).upper()
        if pair["region"] == "5'UTR":
            pair["region"] = REGION_5UTR
        if pair["region"] == "3'UTR":
            pair["region"] = REGION_3UTR
        pair.setdefault("source_region", pair["region"])
        pair.setdefault("target_region", pair["region"])
        pair["source_seq"] = _normalise(str(pair["source_seq"]))
        pair["target_seq"] = _normalise(str(pair["target_seq"]))
        for key in ("source_start", "source_end", "target_start", "target_end",
                    "source_cds_start", "target_cds_start"):
            if key in pair and pair[key] not in ("", None):
                pair[key] = int(pair[key])
        if validate and not validate_ortholog_pair(pair):
            raise ValueError(f"invalid ortholog pair at row {i}: {pair.get('pair_id')}")
        pairs.append(pair)
    return pairs


def prepare_ortholog_pairs(
    path: Optional[str] = None,
    records: Optional[Iterable[MRNARecord]] = None,
    n_pairs: int = 12,
    seed: int = 0,
    output_jsonl: Optional[str] = None,
) -> List[dict]:
    """Prepare ortholog coupling pairs from local input or offline fallback."""
    if path is not None:
        pairs = read_ortholog_pairs(path, validate=True)
    else:
        pairs = synthesize_ortholog_pairs(records=records, n_pairs=n_pairs, seed=seed)
    if output_jsonl is not None:
        write_jsonl(pairs, output_jsonl)
    return pairs


def write_jsonl(pairs: Sequence[Mapping[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(dict(pair), sort_keys=True) + "\n")


__all__ = [
    "REGION_5UTR",
    "REGION_CDS",
    "REGION_3UTR",
    "ORTHOLOG_REGIONS",
    "validate_ortholog_pair",
    "make_synthetic_ortholog_pair",
    "synthesize_ortholog_pairs",
    "read_ortholog_pairs",
    "prepare_ortholog_pairs",
    "write_jsonl",
]
