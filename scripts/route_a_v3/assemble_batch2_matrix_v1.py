#!/usr/bin/env python3
"""Assemble batch-2 generalist results into the 7-model x 9-task matrix.

Sources (append-only, read-only aggregation):
  batch2a_orthrus_v1  : orthrus_4track, orthrus_6track (summary JSON)
  batch2b_codonmrna_v1: codonfm_80m (per-run run_detail.json, job crashed post-completion)
  batch2b2_mrnalm_v1  : mrnalm_5utr, mrnalm_3utr (summary JSON)
  batch2c_calm_v1     : calm (summary JSON)
  batch2d_lucaone_v1  : lucaone (summary JSON)

Output: batch2_matrix_v1.md + batch2_matrix_v1.json (this is a reporting artifact,
NOT a new experiment row; all numbers are copied verbatim from the source runs).
"""
import json
import glob
import os

BASE = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_generalist_rows_v2"
TASKS = [
    "gse114002_mrl", "gse269595_polya", "encsr854ruf_mprau", "gse200304_te",
    "gse149487_te", "gse149487_rna", "gse186455", "half_life_5utr", "half_life_3utr",
]
MODELS = [
    "orthrus_4track", "orthrus_6track", "codonfm_80m",
    "mrnalm_5utr", "mrnalm_3utr", "calm", "lucaone",
]


def load_summary(path):
    out = {}
    d = json.load(open(path))
    for task, models in d["results"].items():
        for mk, r in models.items():
            out[(mk, task)] = r
    return out


def load_rundetails(dirpath):
    out = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "*__*/run_detail.json"))):
        r = json.load(open(p))
        run_dir = os.path.basename(os.path.dirname(p))
        if "__" not in run_dir:
            continue
        task, mk = run_dir.rsplit("__", 1)
        out[(mk, task)] = r
    return out


def main():
    rows = {}
    provenance = {}
    for tag, src in [
        ("batch2a_orthrus_v1", os.path.join(BASE, "batch2a_orthrus_v1/frozen_delta_generalist_results.json")),
        ("batch2b2_mrnalm_v1", os.path.join(BASE, "batch2b2_mrnalm_v1/frozen_delta_generalist_results.json")),
        ("batch2c_calm_v1", os.path.join(BASE, "batch2c_calm_v1/frozen_delta_generalist_results.json")),
        ("batch2d_lucaone_v1", os.path.join(BASE, "batch2d_lucaone_v1/frozen_delta_generalist_results.json")),
    ]:
        if os.path.exists(src):
            got = load_summary(src)
            rows.update(got)
            provenance[tag] = f"summary JSON ({len(got)} rows)"
        else:
            provenance[tag] = "PENDING"
    b2 = os.path.join(BASE, "batch2b_codonmrna_v1")
    if os.path.isdir(b2):
        got = load_rundetails(b2)
        rows.update(got)
        provenance["batch2b_codonmrna_v1"] = f"per-run run_detail.json ({len(got)} rows; job crashed after task completion, before summary write)"

    matrix = {}
    missing = []
    for m in MODELS:
        matrix[m] = {}
        for t in TASKS:
            r = rows.get((m, t))
            if r is None:
                missing.append((m, t))
                matrix[m][t] = None
            else:
                v = r.get("task_macro_spearman")
                matrix[m][t] = None if v is None else round(float(v), 4)

    out_json = os.path.join(BASE, "batch2_matrix_v1.json")
    payload = {
        "artifact": "reporting-only matrix over batch-2 generalist rows",
        "sources": provenance,
        "models": MODELS,
        "tasks": TASKS,
        "spearman_matrix": matrix,
        "missing_cells": [f"{m}::{t}" for m, t in missing],
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Batch-2 generalist frozen-delta matrix (reporting artifact v1)\n")
    lines.append("All numbers = task_macro_spearman copied verbatim from source runs. ")
    lines.append("Rows: frozen backbone + linear probe on embedding differences; VAL eval; append-only.\n")
    lines.append("| model | " + " | ".join(TASKS) + " |")
    lines.append("|---" * (len(TASKS) + 1) + "|")
    for m in MODELS:
        cells = []
        for t in TASKS:
            v = matrix[m][t]
            cells.append("—" if v is None else f"{v:+.3f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    if missing:
        lines.append(f"\nMissing cells ({len(missing)}): " + ", ".join(f"{m}::{t}" for m, t in missing))
    lines.append("\n## Per-model mean over available cells")
    for m in MODELS:
        vals = [v for v in matrix[m].values() if v is not None]
        if vals:
            lines.append(f"- {m}: mean {sum(vals)/len(vals):+.4f} over {len(vals)} cells; max {max(vals):+.4f}; min {min(vals):+.4f}")
        else:
            lines.append(f"- {m}: no cells yet")
    lines.append("\n## Sources")
    for k, v in provenance.items():
        lines.append(f"- {k}: {v}")

    out_md = os.path.join(BASE, "batch2_matrix_v1.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")
    print(f"missing: {len(missing)}")


if __name__ == "__main__":
    main()
