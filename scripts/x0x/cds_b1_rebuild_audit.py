"""X0-X CDS-B1 rebuild audit for GSE207584 (iCodon zebrafish reporter library).

Phase X0-X (3'UTR & CDS transfer) — PURE DEVELOPMENT PREPARATION ONLY.  This
module does NOT touch the frozen 5' primary model, does NOT access sealed
labels, and does NOT trigger the formal X0-X gate.

WHAT THIS DOES
==============
GSE207584 is a synonymous-codon massive reporter library (Diez et al. 2022,
Sci Rep 12:12126).  Per the migration contract §16 and the benchmark registry,
CDS-B1 (EditBench-CDS-B1-Synonymous) stays DORMANT until the legacy liability
GSE207584 completes a *sequence / family / label* rebuild.

This audit performs that rebuild on the parts that ARE recoverable from the
provided files, and *proves with evidence* which part is NOT recoverable:

RECOVERABLE (rebuilt here):
  * family structure: protein -> set of synonymous codon-scheme groups
    (iCodon_1..5 / IDT / Genewiz).  All variants of a protein translate to the
    SAME protein (verified by construction via scripts/x0x/codon.py).
  * labels: measured reporter expression per (protein, group), 3 timepoints
    (2h/5h/8h) x 3 replicates (zf_library_*), aggregated + replicate-level.
  * protein-family split (S7): family-disjoint train/val/test.

NOT RECOVERABLE (blocker, proved here):
  * distinct per-variant synonymous nucleotide sequences.  The provided
    reference FASTA is keyed by construct *Name*, and every group of a protein
    shares the SAME Name set -> all groups collapse to the same sequence set.
    The "imperfect" library (actual measured sequences) has NO family structure
    (1691/1692 singleton).  Therefore we CANNOT emit a distinct synonymous
    sequence per (protein, group) from the current files.  This is the
    sequence-recovery blocker that keeps CDS-B1 DORMANT.

This audit NEVER fabricates a per-variant synonymous sequence.  It emits the
family-anchor CDS (one per protein) and flags per_variant_sequences = blocked.

OUTPUT (into --out-dir)
  group_registry.jsonl            protein families + n_variants + translated protein
  functional_observations.jsonl   expression per (protein,group) x endpoint (aggregate + replicate)
  sequence_entities.jsonl         family-anchor CDS (one per protein; per-variant flagged blocked)
  protein_family_split.json       S7 family-disjoint split (family_id -> split)
  cds_b1_rebuild_audit.json       evidence: counts + sequence-blocker check result
  D1_SHA256SUMS                   sha256 per emitted artifact
  D1_CANONICAL_MANIFEST.json      manifest (mirrors d1_3u_rebuild_finalize format)

Usage:
  python -m scripts.x0x.cds_b1_rebuild_audit \
      --perfect /mnt/.../GSE207584_Zebrafish-library-perfect.csv.gz \
      --fasta  /mnt/.../GSE207584_reference.fasta.gz \
      --out-dir artifacts/x0x/cds_b1_rebuild_20260807 \
      --seed 42 --split-train 0.70 --split-val 0.15
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from scripts.x0x import codon

TIMEPOINTS = ("2h", "5h", "8h")
REPLICATES = (1, 2, 3)
MEAS_COLS = [f"zf_library_{t}_{r}" for t in TIMEPOINTS for r in REPLICATES]


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_str(s: str) -> str:
    return sha256_hex(s.encode("utf-8"))


def jl(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv_gz(path: Path) -> List[dict]:
    with gzip.open(str(path), "rt") as f:
        return list(csv.DictReader(f))


def load_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    with gzip.open(str(path), "rt") as f:
        name: Optional[str] = None
        cur: List[str] = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(cur)
                name = line[1:]
                cur = []
            elif name is not None:
                cur.append(line)
        if name is not None:
            seqs[name] = "".join(cur)
    return seqs


def _dna_to_rna(dna: str) -> str:
    return dna.upper().replace("T", "U")


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


class B1Variant:
    """One (protein, group) synonymous-codon variant of a protein."""

    __slots__ = ("protein", "group", "names", "seqs", "meas", "protein_str")

    def __init__(self, protein: str, group: str):
        self.protein = protein
        self.group = group
        self.names: List[str] = []
        self.seqs: List[str] = []          # distinct design sequences (from FASTA)
        self.meas: List[Dict[str, List[Optional[float]]]] = []  # per Name measurement rows
        self.protein_str: Optional[str] = None

    @property
    def variant_id(self) -> str:
        return f"cds_{self.protein}__{self.group}"


def build_variants(perf: List[dict], fasta: Dict[str, str]) -> "OrderedDict[Tuple[str, str], B1Variant]":
    """Group perfect-library rows by (protein, group)."""
    variants: "OrderedDict[Tuple[str, str], B1Variant]" = OrderedDict()
    for row in perf:
        protein = row["Protein_id"]
        group = row["Group"]
        key = (protein, group)
        v = variants.get(key)
        if v is None:
            v = B1Variant(protein, group)
            variants[key] = v
        name = row["Name"]
        if name not in v.names:
            v.names.append(name)
        seq = fasta.get(name)
        if seq is not None and seq not in v.seqs:
            v.seqs.append(seq)
        meas_row: Dict[str, List[Optional[float]]] = {}
        for t in TIMEPOINTS:
            reps = []
            for r in REPLICATES:
                raw = row[f"zf_library_{t}_{r}"]
                try:
                    reps.append(float(raw))
                except (TypeError, ValueError):
                    reps.append(None)
            meas_row[t] = reps
        v.meas.append(meas_row)
        if v.protein_str is None and seq is not None:
            v.protein_str = codon.translate(_dna_to_rna(seq))
    return variants


def aggregate_variant(v: B1Variant) -> Dict[str, Optional[float]]:
    """Per-timepoint mean across replicate columns and replicate Names."""
    agg: Dict[str, Optional[float]] = {}
    for t in TIMEPOINTS:
        vals = []
        for m in v.meas:
            vals.extend(x for x in m[t] if x is not None)
        agg[t] = statistics.mean(vals) if vals else None
    return agg


# ---------------------------------------------------------------------------
# sequence blocker
# ---------------------------------------------------------------------------


def check_sequence_blocker(variants: "OrderedDict[Tuple[str, str], B1Variant]"
                           ) -> Dict:
    """Prove whether distinct per-variant synonymous sequences are recoverable.

    A protein is sequence-blocked if every group of that protein shares the
    same underlying FASTA sequence set (i.e. the FASTA cannot distinguish the
    codon schemes).  Returns evidence dict.
    """
    by_protein: Dict[str, Dict[str, B1Variant]] = {}
    for key, v in variants.items():
        by_protein.setdefault(key[0], {})[key[1]] = v

    blocked_proteins = 0
    distinct_seq_sets = 0
    examples: List[dict] = []
    for protein, gmap in by_protein.items():
        seq_sets = {frozenset(v.seqs) for v in gmap.values()}
        has_seq = any(len(v.seqs) > 0 for v in gmap.values())
        if has_seq and len(seq_sets) == 1:
            blocked_proteins += 1
            if len(examples) < 3:
                groups = sorted(gmap.keys())
                examples.append({
                    "protein": protein,
                    "n_groups": len(groups),
                    "groups": groups,
                    "n_distinct_sequence_sets": len(seq_sets),
                    "shared_sequence_set_size": len(next(iter(seq_sets))),
                })
        elif has_seq:
            distinct_seq_sets += 1

    n_proteins = len(by_protein)
    return {
        "n_proteins": n_proteins,
        "sequence_blocked_proteins": blocked_proteins,
        "proteins_with_distinct_group_sequences": distinct_seq_sets,
        "sequence_recovery": (
            "BLOCKED" if blocked_proteins == n_proteins else "PARTIAL"),
        "blocker": (
            "NO_DISTINCT_PER_VARIANT_SYNONYMOUS_SEQUENCES_IN_PROVIDED_FILES"),
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# canonical emitters
# ---------------------------------------------------------------------------


def build_group_registry(variants: "OrderedDict[Tuple[str, str], B1Variant]"
                         ) -> List[dict]:
    reg: "OrderedDict[str, dict]" = OrderedDict()
    for key, v in variants.items():
        protein = key[0]
        if protein not in reg:
            reg[protein] = {
                "group_id": f"fam_{protein}",
                "family_id": f"fam_{protein}",
                "protein_id": protein,
                "protein": v.protein_str,
                "n_variants": 0,
                "variants": [],
                "rankable": False,
            }
        reg[protein]["n_variants"] += 1
        reg[protein]["variants"].append(v.variant_id)
        reg[protein]["rankable"] = reg[protein]["n_variants"] >= 2
    return list(reg.values())


def build_functional_observations(variants: "OrderedDict[Tuple[str, str], B1Variant]"
                                  ) -> List[dict]:
    obs: List[dict] = []
    idx = 0
    for key, v in variants.items():
        seq_id = v.variant_id
        ctx = "ctx_gse207584"
        agg = aggregate_variant(v)
        # aggregate mean endpoints
        for t in TIMEPOINTS:
            if agg[t] is None:
                continue
            obs.append({
                "observation_id": f"cds_b1_obs_{idx}", "sequence_id": seq_id,
                "endpoint_id": f"ep_zf_library_{t}_mean", "context_id": ctx,
                "value": round(float(agg[t]), 6), "unit": f"zf_library_{t}",
                "replicate": None,
            })
            idx += 1
        # replicate-level endpoints
        for t in TIMEPOINTS:
            for r in REPLICATES:
                vals = [m[t][r - 1] for m in v.meas if m[t][r - 1] is not None]
                if not vals:
                    continue
                obs.append({
                    "observation_id": f"cds_b1_obs_{idx}", "sequence_id": seq_id,
                    "endpoint_id": f"ep_zf_library_{t}_{r}", "context_id": ctx,
                    "value": round(float(statistics.mean(vals)), 6),
                    "unit": f"zf_library_{t}", "replicate": r,
                })
                idx += 1
    return obs


def build_sequence_entities(variants: "OrderedDict[Tuple[str, str], B1Variant]"
                            ) -> List[dict]:
    """Emit one family-anchor CDS per protein (per-variant distinct seqs not
    recoverable -> flagged)."""
    entities: List[dict] = []
    by_protein: Dict[str, List[B1Variant]] = {}
    for key, v in variants.items():
        by_protein.setdefault(key[0], []).append(v)
    for protein, vs in by_protein.items():
        # canonical anchor: first available design sequence (DNA), frame-locked
        anchor = None
        for v in vs:
            if v.seqs:
                anchor = v.seqs[0]
                break
        if anchor is None:
            continue
        dna = anchor.upper()
        rna = _dna_to_rna(dna)
        state = codon.build_cds_state(rna)  # validates frame/start/stop
        entities.append({
            "sequence_id": f"fam_{protein}",
            "region_scope": "CDS",
            "protein_id": protein,
            "protein": state.protein,
            "n_codons": state.n_codons,
            "anchor_dna_sha256": sha256_str(dna),
            "anchor_rna_sha256": sha256_str(rna),
            "per_variant_distinct_sequences_recoverable": False,
        })
    return entities


def build_s7_split(families: Sequence[dict], seed: int,
                   train_frac: float, val_frac: float) -> Dict[str, str]:
    """Protein-family-disjoint split (S7).  Deterministic by sorted family id +
    seed; families are atomic (never split across partitions)."""
    import random
    fam_ids = sorted(f["family_id"] for f in families)
    rng = random.Random(seed)
    rng.shuffle(fam_ids)
    n = len(fam_ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_val = max(0, min(n_val, n - n_train))
    train = set(fam_ids[:n_train])
    val = set(fam_ids[n_train:n_train + n_val])
    test = set(fam_ids[n_train + n_val:])
    # sanity: disjoint & exhaustive
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(fam_ids)
    out: Dict[str, str] = {}
    for fid in fam_ids:
        if fid in train:
            out[fid] = "train"
        elif fid in val:
            out[fid] = "val"
        else:
            out[fid] = "test"
    return out


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(jl(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfect", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-train", type=float, default=0.70)
    ap.add_argument("--split-val", type=float, default=0.15)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    perf = load_csv_gz(Path(args.perfect))
    fasta = load_fasta(Path(args.fasta))
    variants = build_variants(perf, fasta)

    group_registry = build_group_registry(variants)
    observations = build_functional_observations(variants)
    sequence_entities = build_sequence_entities(variants)
    s7 = build_s7_split(group_registry, args.seed,
                        args.split_train, args.split_val)

    blocker = check_sequence_blocker(variants)
    n_rankable = sum(1 for g in group_registry if g["rankable"])
    n_var = len(variants)
    n_obs = len(observations)

    write_jsonl(out / "group_registry.jsonl", group_registry)
    write_jsonl(out / "functional_observations.jsonl", observations)
    write_jsonl(out / "sequence_entities.jsonl", sequence_entities)
    (out / "protein_family_split.json").write_text(
        json.dumps(s7, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    audit = {
        "phase": "X0X-CDS-B1-REBUILD-AUDIT",
        "accession": "GSE207584",
        "source_files": {
            "perfect": args.perfect,
            "fasta": args.fasta,
        },
        "counts": {
            "n_perfect_rows": len(perf),
            "n_fasta_sequences": len(fasta),
            "n_proteins": len(group_registry),
            "n_variants": n_var,
            "n_rankable_families": n_rankable,
            "n_observations": n_obs,
        },
        "sequence_blocker": blocker,
        "split": {
            "seed": args.seed,
            "n_families": len(group_registry),
            "n_train": sum(1 for v in s7.values() if v == "train"),
            "n_val": sum(1 for v in s7.values() if v == "val"),
            "n_test": sum(1 for v in s7.values() if v == "test"),
        },
        "verdict": (
            "REBUILD_PARTIAL_LABEL_FAMILY_ONLY" if n_rankable > 0 else
            "REBUILD_NO_RANKABLE_FAMILY"),
        "cds_b1_status": "DORMANT_BLOCKED_ON_SEQUENCE",
        "note": (
            "Labels (per protein x group) and family structure are rebuilt. "
            "Distinct per-variant synonymous sequences are NOT recoverable from "
            "the provided files (reference FASTA does not distinguish codon "
            "scheme groups); CDS-B1 stays DORMANT. No fabricated PASS."),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "cds_b1_rebuild_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # SHA256SUMS + manifest
    artifact_keys = [
        "group_registry.jsonl",
        "functional_observations.jsonl",
        "sequence_entities.jsonl",
        "protein_family_split.json",
        "cds_b1_rebuild_audit.json",
    ]
    sums = {k: sha256_file(out / k) for k in artifact_keys}
    sizes = {k: (out / k).stat().st_size for k in artifact_keys}
    (out / "D1_SHA256SUMS").write_text(
        "".join(f"{v}  {k}\n" for k, v in sums.items()), encoding="utf-8")
    manifest = {
        "phase": "X0X-CDS-B1-REBUILD-AUDIT",
        "artifact_files": sizes,
        "config_hash": "xeditflow-v1.1-CDS-B1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out),
        "sha256sums_file": "D1_SHA256SUMS",
        "status": "GENERATED",
    }
    (out / "D1_CANONICAL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"proteins={len(group_registry)} variants={n_var} "
          f"rankable_families={n_rankable} observations={n_obs}")
    print(f"sequence_recovery={blocker['sequence_recovery']} "
          f"(blocked_proteins={blocker['sequence_blocked_proteins']}/{blocker['n_proteins']})")
    print(f"split: train={audit['split']['n_train']} "
          f"val={audit['split']['n_val']} test={audit['split']['n_test']}")
    print(f"verdict={audit['verdict']} cds_b1_status={audit['cds_b1_status']}")
    print(f"wrote {len(artifact_keys)} artifacts to {out}")


if __name__ == "__main__":
    main()
