#!/usr/bin/env python3
"""Critic spec Task 15.4 (tail): complete Dao cryptic-splicing screen for the
S1/M6 variant-window cohorts (v0 GT-AG proxy -> v1 rule set).

Dao, Jungers, Djuranovic & Mustoe 2025 (Nat Commun 16:6844, "U-rich elements
drive pervasive cryptic splicing in 3' UTR MPRAs"): cryptic splicing in
reporter 3' UTRs uses GT at +1 of the 5' donor, AG at -2 of the 3' acceptor,
with U-rich (polypyrimidine) tracts upstream of acceptors; U-rich elements
activate the pathway; 10-50% of published 3'UTR MPRA measurements are impacted.

Rule set v1 (DNA alphabet, sequence-level screen on each 155nt S1 / 100nt M6
window):
  donor_arch   : "GT" dinucleotide present
  acceptor_arch: "AG" dinucleotide with an upstream polypyrimidine tract
                 (>=5 consecutive T, or T-fraction >= 0.60 in the 20nt
                 immediately upstream of AG)
  intron_like  : a donor GT upstream of an acceptor AG at 20-500nt spacing
  u_rich      : window T-fraction >= 0.35 or a >=5T homopolymer (ARE-like
                 activator)
Variant-level risk: the ref->alt SNP creates or destroys any GT / AG /
polypyrimidine element at the variant position (near-splice SNV), or shifts
the window's intron_like count.

Output: analysis_track_b_20260908/cryptic_splicing_dao_v1.{json,md}
Report-grade (QC registration + paper limitation material), not a gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

BASE = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_track_b_20260908")
COHORTS = {
    "S1_stability": (BASE / "s1_stability_pairs.jsonl", ["y_sh", "y_hek"]),
    "M6_ndd_translation": (BASE / "m6_ndd_translation_pairs.jsonl", ["y_totalcount_log2fc"]),
}
OUT = BASE / "cryptic_splicing_dao_v1.json"
OUT_MD = BASE / "cryptic_splicing_dao_v1.md"


def has_polyT(seq: str, start: int, end: int) -> bool:
    seg = seq[max(0, start):max(0, end)]
    if len(seg) >= 5 and "TTTTT" in seg:
        return True
    return len(seg) >= 10 and seg.count("T") / len(seg) >= 0.60


def u_rich(seq: str) -> bool:
    return seq.count("T") / len(seq) >= 0.35 or "TTTTT" in seq


def arch_summary(seq: str) -> dict:
    donors = [i for i in range(len(seq) - 1) if seq[i:i + 2] == "GT"]
    acceptors = [
        j for j in range(len(seq) - 1)
        if seq[j:j + 2] == "AG" and has_polyT(seq, j - 20, j)
    ]
    intron_like = sum(
        1 for g in donors for a in acceptors if 20 <= a - g <= 500
    )
    return {
        "donor_count": len(donors),
        "acceptor_count": len(acceptors),
        "intron_like_pairs": intron_like,
        "u_rich": u_rich(seq),
    }


def variant_effects(ref: str, alt: str) -> dict:
    """Locate the SNP and test element create/destroy at the variant site."""
    diffs = [i for i, (a, b) in enumerate(zip(ref, alt)) if a != b]
    if len(diffs) != 1:
        return {"snv": False}
    i = diffs[0]
    out = {"snv": True, "pos": i, "ref_base": ref[i], "alt_base": alt[i]}
    # dinucleotides touching the variant position (i-1..i and i..i+1)
    def dins(seq):
        return {seq[i - 1:i + 1] if i > 0 else "", seq[i:i + 2]}
    ref_d, alt_d = dins(ref), dins(alt)
    out["creates_GT"] = "GT" in (alt_d - ref_d)
    out["destroys_GT"] = "GT" in (ref_d - alt_d)
    out["creates_AG"] = "AG" in (alt_d - ref_d)
    out["destroys_AG"] = "AG" in (ref_d - alt_d)
    # polypyrimidine tract change in the 20nt upstream of any AG at/after i
    def polyT_state(seq):
        for j in range(max(2, i), min(len(seq) - 1, i + 22)):
            if seq[j:j + 2] == "AG" and has_polyT(seq, j - 20, j):
                return True
        # AG overlapping the variant itself
        for j in (i - 1, i):
            if 0 <= j <= len(seq) - 2 and seq[j:j + 2] == "AG" and has_polyT(seq, j - 20, j):
                return True
        return False
    out["activates_acceptor"] = polyT_state(alt) and not polyT_state(ref)
    out["intron_like_delta"] = arch_summary(alt)["intron_like_pairs"] - arch_summary(ref)["intron_like_pairs"]
    return out


def cohort_stats(rows: list[dict], y_fields: list[str]) -> dict:
    n = len(rows)
    arch = {
        "intron_like_any": sum(1 for r in rows if r["_arch"]["intron_like_pairs"] > 0),
        "u_rich": sum(1 for r in rows if r["_arch"]["u_rich"]),
    }
    risk = [r for r in rows if r["_risk"]]
    safe = [r for r in rows if not r["_risk"]]
    out = {
        "n_pairs": n,
        "intron_like_any_frac": arch["intron_like_any"] / n,
        "u_rich_frac": arch["u_rich"] / n,
        "n_variant_risk": len(risk),
        "variant_risk_frac": len(risk) / n,
        "risk_breakdown": {
            k: sum(1 for r in rows if r.get("_eff", {}).get(k))
            for k in ("creates_GT", "destroys_GT", "creates_AG", "destroys_AG", "activates_acceptor")
        },
    }
    # effect-size comparison: |y| mean/median risk vs safe (artifact-driver check)
    for f in y_fields:
        vals_r = [abs(r[f]) for r in risk if r.get(f) is not None]
        vals_s = [abs(r[f]) for r in safe if r.get(f) is not None]
        if vals_r and vals_s:
            out[f"abs_{f}"] = {
                "risk_mean": mean(vals_r),
                "safe_mean": mean(vals_s),
                "ratio_risk_over_safe": mean(vals_r) / mean(vals_s) if mean(vals_s) > 0 else None,
                "n_risk": len(vals_r),
                "n_safe": len(vals_s),
            }
    return out


def main() -> int:
    report: dict = {"schema_version": "route_a_v3_cryptic_splicing_dao_v1",
                    "source": "Dao, Jungers, Djuranovic & Mustoe 2025, Nat Commun 16:6844",
                    "rule_set": "GT donor / AG acceptor + polypyrimidine tract (>=5T or T>=0.60 in 20nt upstream) / intron-like 20-500nt / U-rich activator (T>=0.35 or >=5T)",
                    "calibre": "report-grade QC registration (Task 15.4 tail), not a gate",
                    "cohorts": {}}
    for name, (path, y_fields) in COHORTS.items():
        rows = []
        for line in path.open():
            r = json.loads(line)
            if r.get("r3_flagged"):
                continue
            ref, alt = r["source_sequence"].upper(), r["candidate_sequence"].upper()
            r["_arch"] = arch_summary(alt)
            eff = variant_effects(ref, alt)
            r["_eff"] = eff
            r["_risk"] = bool(
                eff.get("creates_GT") or eff.get("destroys_GT")
                or eff.get("creates_AG") or eff.get("destroys_AG")
                or eff.get("activates_acceptor")
            )
            rows.append(r)
        report["cohorts"][name] = cohort_stats(rows, y_fields)

    OUT.write_text(json.dumps(report, indent=1))

    md = ["# Cryptic splicing Dao v1 screen — S1/M6 variant cohorts (Task 15.4 tail)", "",
          f"Rule set: {report['rule_set']}", "",
          "| cohort | n | intron-like any | U-rich | variant risk | |Δy| risk/safe (sh) | (hek) | (log2fc) |",
          "|---|---|---|---|---|---|---|---|"]
    for name, s in report["cohorts"].items():
        ratios = []
        for f, label in (("y_sh", "sh"), ("y_hek", "hek"), ("y_totalcount_log2fc", "log2fc")):
            v = s.get(f"abs_{f}")
            ratios.append(f"{v['ratio_risk_over_safe']:.2f}×" if v and v.get("ratio_risk_over_safe") else "—")
        md.append(
            f"| {name} | {s['n_pairs']} | {s['intron_like_any_frac']:.1%} | "
            f"{s['u_rich_frac']:.1%} | {s['variant_risk_frac']:.1%} (n={s['n_variant_risk']}) | "
            f"{' | '.join(ratios)} |")
    md += ["",
           "Reading: risk = the SNP itself creates/destroys a GT donor, AG acceptor,",
           "or polypyrimidine-tract-supported acceptor (near-splice SNV). A risk/safe",
           "effect ratio near 1 means measured effects are not concentrated in",
           "cryptic-splice-competent variants; a large ratio would flag artifact",
           "contamination (Dao 2025: 10-50% of published 3'UTR MPRA measurements",
           "impacted). Report-grade — limitation-note material for D5 far-band data.",
           ]
    OUT_MD.write_text("\n".join(md))
    print("\n".join(md[3:9]))
    print(f"wrote {OUT} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
