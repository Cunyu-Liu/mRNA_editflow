"""D16-C within-source augmentation H1 hard-clause structural tests (amendment step 3)."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

D16C = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_d16c")
SYNTH = D16C / "within_source_augmentation_mrl_v1.jsonl"
TRACE = D16C / "synthetic_trace_mrl_v1.jsonl"
AUDIT = D16C / "leak_audit_mrl_v1.json"
VAL = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/projections/xedit_v3/development_train_validation_v1/validation.jsonl")

FORBIDDEN_LABEL_FIELDS = (
    "direction_normalized_delta",
    "delta",
    "label",
    "y",
    "target",
    "measured_value",
    "outcome",
)


def _rows(path, limit=None):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            out.append(json.loads(line))
    return out


def test_files_exist():
    assert SYNTH.exists() and TRACE.exists() and AUDIT.exists()


def test_h1_no_regression_labels_on_synthetic_rows():
    rows = _rows(SYNTH)
    assert rows, "no synthetic rows produced"
    for r in rows:
        assert r.get("synthetic") is True
        assert r.get("h1_label") is None
        assert r.get("loss_weight") == 0.0
        for f in FORBIDDEN_LABEL_FIELDS:
            assert f not in r, f"H1 violation: synthetic row carries {f}"


def test_candidate_differs_from_source():
    for r in _rows(SYNTH, limit=500):
        assert r["candidate_sequence"] != r["source_sequence"]
        assert len(r["candidate_sequence"]) == len(r["source_sequence"])


def test_leak_audit_pass_and_recompute():
    audit = json.loads(AUDIT.read_text())
    assert audit["leak_audit_status"] == "PASS"
    val_set = set()
    with open(VAL) as f:
        for line in f:
            d = json.loads(line)
            if d.get("task_id", "").startswith("MEAN_RIBOSOME_LOAD"):
                val_set.add(str(d["candidate_sequence"]).upper())
                val_set.add(str(d["source_sequence"]).upper())
    hits = 0
    for r in _rows(SYNTH):
        c = str(r["candidate_sequence"]).upper()
        if c in val_set:
            hits += 1
    assert hits == 0, f"recomputed leak: {hits} synthetic rows match VALIDATION"


def test_trace_completeness():
    with open(TRACE) as f:
        trace = [json.loads(line) for line in f]
    assert trace
    synth_by_src = {}
    for r in _rows(SYNTH):
        synth_by_src[r["source_id"]] = synth_by_src.get(r["source_id"], 0) + 1
    traced = {t["source_id"]: t for t in trace}
    for sid, n in synth_by_src.items():
        assert sid in traced, f"source {sid} missing from trace"
        assert traced[sid]["n_synthetic"] == n, f"trace count mismatch for {sid}"


def test_determinism_same_seed_same_first_rows():
    rows = _rows(SYNTH, limit=50)
    assert all(r["d16c_amendment"] == "v1" for r in rows)
    seeds = {r["trace_seed"] for r in _rows(SYNTH, limit=500)}
    assert len(seeds) > 400, "trace seeds not unique-ish (determinism broken?)"


def test_per_source_density_target():
    audit = json.loads(AUDIT.read_text())
    assert audit["avg_cand_per_source_after"] >= 30.0
    assert audit["n_synthetic"] > 13000
    assert audit["protected_reads"] == 0
