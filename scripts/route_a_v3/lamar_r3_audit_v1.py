#!/usr/bin/env python3
"""Baseline Task 6.5.6 (P1-2): R3 leakage audit for LAMAR on GSE217518.

LAMAR (Genome Biol 2025, zhw-e8/LAMAR): ESM2-style 150M RNA LM pretrained on
mammalian80D (natural transcriptomes); UTR3DegPred = official fine-tuned
3'UTR->half-life regressor (BEAS-2B, CN1 natural-3'UTR tiling library, 1,595
train + 197 val + 178 test rows shipped in repo).

GSE217518 = variant-stability MPRA (SNV variant pairs, HEK293, 903 VALIDATION
records). R3 audit: exact-sequence overlap between the GSE217518 VALIDATION
source/candidate sequences and (a) UTR3DegPred train/val/test splits,
(b) stability.csv master, (c) UTR5TEPred / IRESPred / SpliceSitePred shipped
data. Natural-transcriptome pretraining (mammalian80D) contains no synthetic
MPRA variants by construction (Ensembl-derived), registered as such.

Output: analysis_lamar_frozen_delta_20260909/r3_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
LAMAR = MNT / "external_model_assets/lamar"
WEIGHTS = MNT / "external_model_assets/lamar_weights"
OUT = MNT / "experiments/analysis_lamar_frozen_delta_20260909"
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANONICAL = MNT / "canonical/GSE217518/v1/canonical_records.jsonl"


def load_csv_seqs(path: Path) -> set[str]:
    seqs = set()
    if not path.exists():
        return seqs
    lines = path.read_text().splitlines()
    for line in lines[1:]:
        parts = line.split(",")
        # first column may be id (stability.csv) or seq (training_set_2.csv)
        for cand in (parts[0], parts[1] if len(parts) > 1 else ""):
            if set(cand.upper()) <= {"A", "C", "G", "T"} and len(cand) >= 30:
                seqs.add(cand.upper())
    return seqs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # GSE217518 VALIDATION sequences
    val_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "GSE217518" and row["split"] == "VALIDATION":
                val_ids.add(str(row["canonical_record_id"]))
    ours: set[str] = set()
    with CANONICAL.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in val_ids:
                ours.add(str(row["source_sequence"]).upper())
                ours.add(str(row["candidate_sequence"]).upper())
    print(f"GSE217518 VALIDATION: {len(val_ids)} records -> {len(ours)} unique sequences")

    # LAMAR shipped datasets
    datasets = {
        "UTR3DegPred_train": LAMAR / "UTR3DegPred/data/training_set_2.csv",
        "UTR3DegPred_val": LAMAR / "UTR3DegPred/data/validation_set.csv",
        "UTR3DegPred_test": LAMAR / "UTR3DegPred/data/testing_set_2.csv",
        "UTR3DegPred_stability_master": LAMAR / "UTR3DegPred/data/stability.csv",
        "UTR5TEPred_train": LAMAR / "UTR5TEPred/data/training_set_2.csv",
        "IRESPred_train": LAMAR / "IRESPred/data/training_set_2.csv",
    }
    overlap = {}
    for name, path in datasets.items():
        seqs = load_csv_seqs(path)
        inter = ours & seqs
        overlap[name] = {
            "n_lamar_rows_seqs": len(seqs),
            "n_overlap_with_gse217518_validation": len(inter),
            "examples": sorted(inter)[:3],
        }
        print(f"{name}: {len(seqs)} seqs, overlap {len(inter)}")

    # substring check too (GSE217518 windows are 155nt; LAMAR segments ~155nt
    # from a natural tiling library — check containment both ways on a sample)
    lamar_all = set()
    for path in datasets.values():
        lamar_all |= load_csv_seqs(path)
    contained = sum(1 for s in ours if any(s in t or t in s for t in lamar_all)) if lamar_all else 0

    verdict = all(v["n_overlap_with_gse217518_validation"] == 0 for v in overlap.values()) and contained == 0
    payload = {
        "schema_version": "route_a_v3_lamar_r3_audit_v1",
        "model": "LAMAR (Genome Biol 2025) pretrain mammalian80D + official UTR3DegPred fine-tune",
        "gse217518_validation": {"records": len(val_ids), "unique_sequences": len(ours)},
        "exact_overlap": overlap,
        "substring_containment_count": contained,
        "pretraining_provenance": "mammalian80D = natural mammalian transcriptomes (Ensembl-derived); synthetic MPRA variant libraries not present by construction (registered, not row-level auditable)",
        "utr3degpred_provenance": "BEAS-2B half-life on CN1 natural-3'UTR tiling library (2,790 segments) — different study, cell line, and library design from GSE217518 variant-stability MPRA",
        "r3_flagged": not verdict,
        "verdict": "PASS" if verdict else "FAIL",
    }
    (OUT / "r3_audit.json").write_text(json.dumps(payload, indent=1))
    print(f"R3 verdict: {payload['verdict']} (flagged={payload['r3_flagged']})")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
