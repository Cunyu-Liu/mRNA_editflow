#!/usr/bin/env python3
"""Baseline P0-3b: closed NDCG@K for the generation-line Table 4 (gap closure).

Closed NDCG (BENCHMARK_GUIDE §7.3): within each source's FROZEN measured
neighborhood, rank the measured candidates that the method's generated pool
actually contains by the method's own score (generation_score / critic_score),
and compute NDCG@K against the true measured outcome ordering (higher
direction-normalized delta = more relevant, linear gain, log discount).

Calibre note (honest): this measures ranking quality ON THE HIT SET -- sources
whose generated pool contains no measured candidate contribute 0 (the
generator failed to surface any measurable candidate); that is the same
open-support conditioning as recovery, so closed NDCG complements (not
replaces) the coverage-side metrics. Pool-empty sources are counted as 0 and
their fraction is reported alongside (support@B).

Rows: unguided / v5-guided / v8-specialist / v8-joint (TreeG has no
generation_score pool of its own -> skipped, noted).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MEASURED = MNT / "generation_eligibility/development_validation_v1/measured_neighborhood.private.jsonl"
OUT = MNT / "experiments/analysis_baseline_table4_20260908"
K = 10

RUNS = {
    "unguided_base": MNT / "experiments/xeditsetflow_v5/guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl",
    "v5_critic_guided": MNT / "experiments/xeditsetflow_v5/guided_b2_20260903/b2_full_891/guided/generated_candidates.private.jsonl",
    "v8_specialist_guided": MNT / "experiments/xeditsetflow_v5/guided_b2_v8_20260908/b2_full_891/guided/generated_candidates.private.jsonl",
    "v8_joint_guided": MNT / "experiments/xeditsetflow_v5/guided_b2_v8joint_20260908/b2_full_891/guided/generated_candidates.private.jsonl",
}


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(scored: list[tuple[float, float]], k: int) -> float:
    """scored = [(model_score, true_outcome)]; NDCG@k with linear gain on outcome."""
    if not scored:
        return 0.0
    ranked = sorted(scored, key=lambda x: -x[0])[:k]
    gains = [outcome for _s, outcome in ranked]
    ideal = sorted((o for _s, o in scored), reverse=True)[:k]
    d, i = dcg(gains), dcg(ideal)
    return d / i if i > 0 else 0.0


def main() -> int:
    # measured neighborhood: source -> {sequence: outcome}
    measured: dict[str, dict[str, float]] = defaultdict(dict)
    for line in MEASURED.open():
        row = json.loads(line)
        seq = str(row["candidate_sequence"]).upper().replace("T", "U")
        measured[str(row["source_key"])][seq] = float(row["measured_direction_normalized_delta"])

    report = {"schema_version": "route_a_v3_baseline_table4_closed_ndcg.v1",
              "k": K, "calibre": "hit-set closed NDCG@K (linear gain on measured outcome, "
              "log discount); empty-hit sources count 0 and support@B is reported alongside",
              "rows": {}}
    for name, path in RUNS.items():
        if not path.exists():
            report["rows"][name] = {"skipped": f"candidates file missing: {path}"}
            continue
        by_source: dict[str, dict[str, float]] = defaultdict(dict)
        for line in path.open():
            row = json.loads(line)
            seq = str(row["candidate_sequence"]).upper().replace("T", "U")
            score = float(row["critic_score"] if row.get("critic_score") is not None
                          else row["generation_score"])
            # keep max score per unique sequence
            if seq not in by_source[row["source_key"]] or score > by_source[row["source_key"]][seq]:
                by_source[row["source_key"]][seq] = score
        per_source_ndcg = []
        n_hit_sources = 0
        for source_key, pool in measured.items():
            gen = by_source.get(source_key, {})
            scored = [(gen[seq], outcome) for seq, outcome in pool.items() if seq in gen]
            if scored:
                n_hit_sources += 1
            per_source_ndcg.append(ndcg_at_k(scored, K))
        report["rows"][name] = {
            "closed_ndcg_at_10_source_macro": float(np.mean(per_source_ndcg)) if per_source_ndcg else None,
            "n_sources_total": len(measured),
            "n_sources_with_hit": n_hit_sources,
            "support_at_32": n_hit_sources / len(measured) if measured else None,
        }
        print(f"[{name}] closed-NDCG@10 = {report['rows'][name]['closed_ndcg_at_10_source_macro']:.4f} "
              f"(hit sources {n_hit_sources}/{len(measured)})", flush=True)

    report["rows"]["treeg_sample_then_select"] = {
        "skipped": "TreeG ranks a critic-scored 256-pool, not a generator output; "
                   "no generation_score pool of its own -- excluded from closed NDCG"}
    (OUT / "table4_closed_ndcg_v1.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT / 'table4_closed_ndcg_v1.json'}")

    # markdown supplement
    md = ["", "## Table 4 补充：closed NDCG@10（P0-3b，hit-set 口径）", "",
          "| method | closed NDCG@10 | hit sources | support@32 |",
          "|---|---|---|---|"]
    for name, r in report["rows"].items():
        if "skipped" in r:
            continue
        md.append("| %s | %.4f | %d/%d | %.4f |" % (
            name, r["closed_ndcg_at_10_source_macro"], r["n_sources_with_hit"],
            r["n_sources_total"], r["support_at_32"]))
    md += ["", "口径注：hit-set NDCG（生成池中命中 measured 的候选按模型分排序 vs 真值排序）；"
           "空命中源计 0——覆盖侧指标（recovery）与排序侧指标（NDCG）互补报告。"]
    with (OUT / "table4_generation_closed_form_v1.md").open("a") as fh:
        fh.write("\n".join(md) + "\n")
    print("appended to table4 markdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
