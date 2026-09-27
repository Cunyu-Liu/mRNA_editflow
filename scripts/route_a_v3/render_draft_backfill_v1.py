#!/usr/bin/env python3
"""Render the DeltaBench draft backfill blocks (S5.6 M1 four gates / S8.1 polyA 3-seed)
from the frozen harvest JSONs. Read-only producer: it never edits the draft; it writes a
reviewable markdown file that is then inserted verbatim by a human.

Output: experiments/deltabench_draft_backfill_v1/draft_backfill_sections_v1.md
Usage:  python render_draft_backfill_v1.py
Exit:   0 = at least one block rendered; 3 = neither harvest present (nothing to do)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RT2 = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
M1_JSON = RT2 / "experiments/xeditcritic_m1_intervention/adjudication_v1/m1_adjudication_v1.json"
P3_JSON = RT2 / "experiments/xeditcritic_v5_polya_3seed/harvest_v1/polya_3seed_harvest_v1.json"
OUT = RT2 / "experiments/deltabench_draft_backfill_v1"


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_m1(path: Path) -> str | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    g1, g2, g3 = d.get("G1") or {}, d.get("G2") or {}, d.get("G3") or {}
    g4 = d.get("G4") or {}
    verdict = d.get("verdict") or {}
    eff = g3.get("efficiency") or {}
    lines = [
        "## §5.6 backfill — M1 intervention arm (four gates, frozen amendment v1)",
        "",
        f"- **G1 direction gate** — MRL VALIDATION frozen-Δ task-macro Spearman = {fmt(g1.get('value'), 6)} "
        f"(recomputed from archived predictions, match = {g1.get('recompute_match')}); "
        f"baseline band: V2 3-seed ensemble {fmt(g1.get('baseline_v2_3seed_ensemble'), 4)} + 0.01; "
        f"gain vs ensemble = {fmt(g1.get('gain_vs_ensemble'), 6)}; "
        f"**direction_positive = {g1.get('direction_positive')}** (single-seed direction read only, per prereg).",
        f"- **G2 non-destruction** — polyA VALIDATION recompute = {fmt(g2.get('value'), 6)}; "
        f"drop vs frozen main row {fmt(g2.get('frozen_polya_main_row'), 4)} = {fmt(g2.get('drop_vs_main'), 6)}; "
        f"tolerance {g2.get('drop_tolerance')}; **pass = {g2.get('pass')}**.",
        f"- **G3 efficiency (report only)** — wallclock {fmt(eff.get('wallclock_h'), 2)} h; "
        f"peak CUDA memory {fmt(eff.get('peak_cuda_memory_gb'), 2)} GB; steps {eff.get('steps_total')}; "
        f"train pool {g3.get('train_pool_total')} rows (of which M1 {g3.get('m1_rows')}).",
    ]
    if g4.get("triggered"):
        arm = g4.get("arm_m1_row") or {}
        base = g4.get("v2_baseline_m1_row") or {}
        spectrum = g4.get("external_family_spectrum") or {}
        lines.append(
            f"- **G4 mechanism (triggered)** — M1 eval-row frozen-Δ overall Spearman: "
            f"arm = {fmt(arm.get('overall_spearman'), 6)} (n = {arm.get('n')}); "
            f"V2 baseline = {fmt(base.get('overall_spearman'), 6)}; external-family spectrum: "
            + "; ".join(f"{k} {fmt(v.get('overall_spearman'), 4)}" for k, v in sorted(spectrum.items()))
            + "."
        )
        per_context = arm.get("per_context") or {}
        if per_context:
            lines.append(
                "- G4 per-context readings: "
                + "; ".join(f"{k} {fmt(v.get('spearman'), 4)} (n={v.get('n')})" for k, v in sorted(per_context.items()))
                + "."
            )
    else:
        lines.append(
            "- **G4 mechanism — not triggered** (per prereg, G4 runs only if the G1 direction is positive): "
            f"note = {g4.get('note')}."
        )
    lines += [
        "",
        f"- **Verdict sentence for the draft**: G1 direction positive = **{verdict.get('G1_direction_positive')}**, "
        f"G2 non-destruction pass = **{verdict.get('G2_pass')}**, G4 triggered = **{verdict.get('G4_triggered')}**. "
        "State the result as-is (a negative direction is a valid mechanism deliverable); no peak picking "
        "(FINAL-EPOCH-6-FIXED); quote the single-seed limitation.",
        "",
    ]
    return "\n".join(lines)


def render_p3(path: Path) -> str | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    three = d.get("three_seed") or {}
    per_seed = d.get("per_seed") or {}
    ap = d.get("aparent_reference") or {}
    delta = d.get("delta_vs_aparent") or {}
    boot = d.get("bootstrap") or {}
    holm = d.get("holm_direction") or {}
    lines = [
        "## §8.1 backfill — polyA V5 3-seed robustness supplement (P0-1)",
        "",
        "| Seed | polyA VALIDATION Spearman (n) | top-1 | NDCG@10 |",
        "|---|---|---|---|",
    ]
    for name in ("seed20260907", "seed20260921", "seed20260922"):
        entry = per_seed.get(name) or {}
        lines.append(
            f"| {name} | {fmt(entry.get('spearman'), 6)} (n={entry.get('n')}) | "
            f"{fmt(entry.get('top_1'), 4)} | {fmt(entry.get('ndcg_at_10'), 4)} |"
        )
    lines += [
        "",
        f"- **3-seed mean = {fmt(three.get('mean'), 6)}**, range [{fmt((three.get('range') or [None, None])[0], 6)}, "
        f"{fmt((three.get('range') or [None, None])[1], 6)}]; main row {fmt(three.get('main_row_unchanged'), 6)} "
        "unchanged by design; new leaderboard row label = `polyA-V5-3seed-mean`.",
        f"- **Δ vs APARENT** (recomputed {fmt(ap.get('recomputed_spearman'), 6)}, frozen point {fmt(ap.get('frozen_point'), 4)}): "
        f"point {fmt(delta.get('point'), 6)}, CI95 [{fmt((delta.get('ci95') or [None, None])[0], 6)}, "
        f"{fmt((delta.get('ci95') or [None, None])[1], 6)}], excludes zero = {delta.get('ci_excludes_zero')}.",
        f"- Bootstrap: {boot.get('iterations')} iterations, seed {boot.get('seed')}, unit {boot.get('unit')}; "
        f"raw p = {fmt(boot.get('raw_p'), 6)} (cross-check seed 20260816 in the JSON).",
    ]
    direction = (holm.get("recomputed") or {}).get("polyA: V5-3seed-mean vs APARENT") if holm.get("available") else None
    if direction:
        lines.append(
            f"- **Holm family direction** (other three rows frozen): polyA recomputed raw p = {fmt(direction.get('raw_p'), 6)}, "
            f"holm p = {fmt(direction.get('holm_p'), 6)}, significant = {direction.get('significant_holm_0.05')} "
            f"(frozen table row kept unchanged at {fmt(holm.get('frozen_polyA_row'), 4)})."
        )
    lines += [
        "",
        f"- **Verdict sentence for the draft**: {d.get('verdict')} — write the 3-seed row as a robustness supplement "
        "(main row never replaced); if the CI crosses zero, switch to the interval-claim wording fixed by the prereg.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    blocks = []
    m1 = render_m1(M1_JSON)
    p3 = render_p3(P3_JSON)
    if m1:
        blocks.append(m1)
    if p3:
        blocks.append(p3)
    if not blocks:
        print("NOT_READY: neither harvest JSON exists yet (M1 adjudication / polyA 3-seed harvest).")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)
    header = (
        "# DeltaBench draft backfill blocks v1\n\n"
        f"- Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"- Sources: `{M1_JSON}` (present={M1_JSON.exists()}); `{P3_JSON}` (present={P3_JSON.exists()})\n"
        "- Usage: insert each block verbatim into the matching PENDING slot of `docs/paper/deltabench_main_paper_draft_v1.md`; "
        "numbers are read-only copies of the frozen harvest JSONs.\n\n"
    )
    out_path = OUT / "draft_backfill_sections_v1.md"
    out_path.write_text(header + "\n".join(blocks))
    print(f"rendered {len(blocks)} block(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())