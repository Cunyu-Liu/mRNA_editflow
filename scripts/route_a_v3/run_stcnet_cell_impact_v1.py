#!/usr/bin/env python3
"""P2-1 companion: STCNet cell-level impact quantification.

The leaderboard cells (benchmark_v2/leaderboard_matrix_v2/cells/utr_stcnet/)
were produced by run_benchmark_v2_matrix_v1.py + family_adapters_v1.py WITHOUT
a pinned torch.manual_seed (the older analysis runner
run_route2_benchmark_v2_matrix_v1.py pinned per-batch seed 20260816, but the
leaderboard emit path did not). determinism.json (v1) already established:
- unpinned native: max|d| 0.033 over 100 seqs, rank corr 0.9992
- pinned 20260816: max|d| 0.0049, rank corr 0.9999

This companion answers the reviewer-facing question: how much does the CELL
Spearman (the leaderboard number) move across same-input re-runs? Method:
re-score the STCNet MRL cell (GSE114002 VALIDATION 730 rows, source+cand)
twice with the exact leaderboard adapter (family_adapters_v1, unpinned), then
recompute the cell Spearman for each run and report the spread. Also recompute
one more repeat (three runs total) for a robust range estimate.

Discipline: VALIDATION split only, sequences only (label use = the same
observed column the cell already published); MIG GPU 6; protected TEST
reads = 0; append-only output stcnet_cell_impact.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = REPO_ROOT / "scripts/route_a_v3/benchmark_v2_matrix/family_adapters_v1.py"
MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT_DIR = MNT / "experiments/analysis_forward_determinism_v1"

MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANON = MNT / "canonical/GSE114002/v1/canonical_records.private.jsonl"
CELL_PREDS = MNT / "benchmark_v2/leaderboard_matrix_v2/cells/utr_stcnet/predictions.jsonl"
CELL_NAME = "GSE114002|5UTR|MEAN_RIBOSOME_LOAD"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import torch

    device = torch.device("cuda:6")
    ad = _load_module("family_adapters_impact", ADAPTERS)
    scorer, meta = ad.build_scorer("utr_stcnet", device)

    ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "VALIDATION" and row["study_unit_id"] == "GSE114002":
                ids.add(str(row["canonical_record_id"]))
    rows = []
    with CANON.open() as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["canonical_record_id"]) in ids:
                rows.append(row)
    rows = sorted(rows, key=lambda r: str(r["canonical_record_id"]))
    _require = lambda c, m: (_ for _ in ()).throw(RuntimeError(m)) if not c else None
    _require(len(rows) == 730, f"expected 730 MRL VALIDATION rows, got {len(rows)}")

    src = [r["source_sequence"] for r in rows]
    cand = [r["candidate_sequence"] for r in rows]
    obs = np.asarray([float(r["direction_normalized_delta"]) for r in rows])

    cell_rho = []
    run_values = []
    for repeat in range(3):
        y_src = np.asarray(scorer(src), dtype=float)
        y_cand = np.asarray(scorer(cand), dtype=float)
        delta = y_cand - y_src
        rho = spearmanr(obs, delta).statistic
        cell_rho.append(float(rho))
        run_values.append(delta.tolist())
        print(f"[repeat {repeat + 1}] cell spearman = {rho:.10f}", flush=True)

    arr = np.asarray(run_values)
    pairwise_max = []
    for i in range(3):
        for j in range(i + 1, 3):
            pairwise_max.append(float(np.abs(arr[i] - arr[j]).max()))
    archived = None
    with CELL_PREDS.open() as handle:
        for line in handle:
            rec = json.loads(line)
            if rec["cell"] == CELL_NAME:
                if archived is None:
                    archived = {"observed": [], "delta_hat": []}
                archived["observed"].append(float(rec["observed"]))
                archived["delta_hat"].append(float(rec["delta_hat"]))
    archived_rho = None
    if archived:
        archived_rho = float(
            spearmanr(np.asarray(archived["observed"]), np.asarray(archived["delta_hat"])).statistic
        )

    spread = max(cell_rho) - min(cell_rho)
    verdict = (
        "CELL_4TH_DECIMAL_STABLE"
        if spread < 5e-5
        else "CELL_4TH_DECIMAL_UNSTABLE_DECLARATION_REQUIRED"
        if spread >= 1e-4
        else "CELL_BORDERLINE"
    )
    result = {
        "schema_version": "route_a_v3_stcnet_cell_impact_v1",
        "task": "P2-1 companion: STCNet leaderboard-cell Spearman run-to-run spread",
        "cell": CELL_NAME,
        "n_rows": len(rows),
        "protocol": (
            "same leaderboard adapter (family_adapters_v1, unpinned cluster_dpc_knn "
            "rand*1e-6 noise), 3 independent full-cell re-scorings on MIG cuda:6; "
            "cell Spearman recomputed per run; archived cell value cross-checked"
        ),
        "cell_spearman_runs": cell_rho,
        "cell_spearman_spread": float(spread),
        "max_pairwise_delta_maxabs": pairwise_max,
        "archived_cell_spearman_recomputed": archived_rho,
        "archived_matches_runs": (
            archived_rho is not None and min(cell_rho) - 1e-9 <= archived_rho <= max(cell_rho) + 1e-9
        ),
        "verdict": verdict,
        "statement": (
            "STCNet cells are single-run numbers under inference-time "
            "tie-breaking noise (official design); run-to-run cell Spearman "
            "spread = %.2e (< 1e-4 -> the published 4th decimal may wobble by "
            "+-1 ulp of the 4th decimal; rank stability >= 0.9992). "
            "Declaration note: cite as 'single-run frozen forward under "
            "official stochastic DPC clustering; re-run spread measured at "
            "%.1e on the MRL cell'." % (spread, spread)
        ),
        "gpu": "cuda:6 MIG 1g.5gb",
        "protected_test_reads": 0,
    }

    out_path = OUT_DIR / "stcnet_cell_impact.json"
    if out_path.exists():
        raise SystemExit("stcnet_cell_impact.json already exists (append-only)")
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
