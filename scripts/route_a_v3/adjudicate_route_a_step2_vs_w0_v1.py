#!/usr/bin/env python3
"""Route A Step-2 preregistered gate adjudication (record-level paired bootstrap).

Gate (docs/paper/route2_route_a_necessity_certainty_analysis.md §4 Stage 2):
main criterion = CI not crossing zero ABOVE W0 0.1987 on the aligned GSE114002
VALIDATION frozen-delta caliber (K=10, frozen Task-1 evaluator).

Arms adjudicated on the same 730 records:
- route_a_step1_280k (zero-shot 280K pre-finetuned scorer)
- route_a_step2_task_adapt (Step-1 init + per-task LoRA/head, final pass 8)
vs W0 (critic from-scratch, 0.1987 reference) and pairwise step2-vs-step1
(task-adaptation harm quantification).

Paired bootstrap over validation records, 2,000 iterations, seed 20260816
(same as the W-ladder adjudication). Pure prediction-file statistics: no
model forward is involved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MNT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
STEP1_PREDS = Path(f"{MNT}/experiments/xeditcritic_route_a/280k_prefinetune_20260903/predictions.jsonl")
STEP2_PREDS = Path(f"{MNT}/experiments/xeditcritic_route_a/step2_task_adapt_20260903/predictions.jsonl")
W0_PREDS = Path(f"{MNT}/experiments/xeditcritic_w0/w0_screen_seed_20260907_runner_7303417c/w0_mrl_gse114002/final_validation_predictions.jsonl")
CANONICAL = Path(f"{MNT}/canonical/GSE114002/v1/canonical_records.private.jsonl")
MANIFEST = Path(f"{MNT}/manifests/route2_development_frozen_v1/development_manifest.jsonl")
OUT = Path(f"{MNT}/experiments/analysis_route_a_step2_adjudication_20260903/results.json")
BOOT_ITERS = 2000
BOOT_SEED = 20260816


def load_predictions(path: Path) -> dict[str, float]:
    preds = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id") or row.get("record_id"))
            value = row.get("predicted_direction_normalized_delta")
            if value is None:
                value = row.get("prediction")
            preds[rid] = float(value)
    return preds


def main() -> int:
    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "GSE114002" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))
    targets = {}
    with CANONICAL.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                targets[rid] = float(row["direction_normalized_delta"])

    step1 = load_predictions(STEP1_PREDS)
    step2 = load_predictions(STEP2_PREDS)
    w0 = load_predictions(W0_PREDS)
    shared = sorted(validation_ids & set(step1) & set(step2) & set(w0) & set(targets))
    if len(shared) != len(validation_ids):
        raise SystemExit(f"coverage mismatch: {len(shared)} shared vs {len(validation_ids)} validation ids")

    t = np.asarray([targets[r] for r in shared])
    p1 = np.asarray([step1[r] for r in shared])
    p2 = np.asarray([step2[r] for r in shared])
    pw = np.asarray([w0[r] for r in shared])

    rho = lambda p: float(spearmanr(t, p).statistic)
    result = {
        "schema_version": "route_a_v3_route_a_step2_adjudication.v1",
        "record_count": len(shared),
        "bootstrap_iterations": BOOT_ITERS,
        "bootstrap_seed": BOOT_SEED,
        "point_estimates": {
            "route_a_step1_280k": rho(p1),
            "route_a_step2_task_adapt": rho(p2),
            "w0_critic_from_scratch": rho(pw),
        },
        "comparisons": {},
    }
    rng = np.random.default_rng(BOOT_SEED)
    n = len(shared)
    for name, pa, pb in (
        ("step2_minus_w0_gate", p2, pw),
        ("step1_minus_w0", p1, pw),
        ("step2_minus_step1_harm", p2, p1),
    ):
        deltas = []
        for _ in range(BOOT_ITERS):
            idx = rng.integers(0, n, n)
            try:
                ra = spearmanr(t[idx], pa[idx]).statistic
                rb = spearmanr(t[idx], pb[idx]).statistic
                if np.isfinite(ra) and np.isfinite(rb):
                    deltas.append(ra - rb)
            except Exception:
                continue
        deltas = np.asarray(deltas)
        ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
        result["comparisons"][name] = {
            "delta_spearman": float(np.mean(deltas)),
            "bootstrap_ci_95": ci,
            "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "gate_pass_ci_above_reference": bool(ci[0] > 0) if name == "step2_minus_w0_gate" else None,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
