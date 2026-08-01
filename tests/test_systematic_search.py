"""Unit tests for scripts/data/systematic_search.py (D0-02)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.systematic_search import (
    CANDIDATES,
    EVIDENCE_GRADES,
    REQUIRED_FIELDS,
    acceptance_checks,
    render_results_md,
    run_discovery,
    title_matches,
    validate_record,
)


def test_frozen_candidates_cover_all_sub_benchmarks():
    benches = {c["sub_benchmark"] for c in CANDIDATES}
    assert benches == {
        "EditBench-5U-Natural",
        "EditBench-5U-Dense",
        "EditBench-3U-Variant",
        "EditBench-CDS-Synonymous",
    }


def test_frozen_candidates_have_p0_accessions():
    accs = {c["accession"] for c in CANDIDATES}
    for expected in (
        "GSE145046", "GSE114002", "GSE149487", "GSE246381",
        "GSE217518", "GSE200304", "ENCSR854RUF", "GSE207584", "GSE173083",
    ):
        assert expected in accs


def test_every_frozen_candidate_passes_schema_validation():
    for cand in CANDIDATES:
        assert validate_record(cand) == [], cand["candidate_id"]


def test_validate_record_catches_missing_and_bad_values():
    rec = {f: "x" for f in REQUIRED_FIELDS}
    rec["evidence_grade"] = "Z9"
    rec["wt_availability"] = "maybe"
    errors = validate_record(rec)
    assert any("evidence_grade" in e for e in errors)
    assert any("wt_availability" in e for e in errors)
    del rec["endpoint"]
    errors = validate_record(rec)
    assert any("endpoint" in e for e in errors)


def test_title_matches_keyword_rules():
    assert title_matches("Decoding mRNA translatability from 5' UTR", ["translatability"])
    assert not title_matches("Some other study", ["translatability"])
    assert title_matches("Anything", [])  # empty keyword list == no constraint


def test_offline_run_produces_valid_registry_and_acceptance():
    registry = run_discovery(offline=True)
    assert registry["contract_id"] == "utr_editflow_contract_v2"
    assert len(registry["candidates"]) == len(CANDIDATES)
    checks = acceptance_checks(registry)
    # offline mode: schema check present, live-verification check skipped
    assert len(checks) == 1
    assert checks[0][1] is True
    for cand in registry["candidates"]:
        assert cand["verification"]["offline"] is True


def test_results_md_renders_all_candidates():
    registry = run_discovery(offline=True)
    md = render_results_md(registry, acceptance_checks(registry))
    for cand in CANDIDATES:
        assert cand["accession"] in md
        assert cand["candidate_id"] in md


def test_cli_offline_writes_outputs(tmp_path):
    from scripts.data.systematic_search import main

    yaml_out = tmp_path / "candidates.yaml"
    md_out = tmp_path / "results.md"
    rc = main(["--offline", "--yaml-out", str(yaml_out), "--results-md", str(md_out),
               "--artifact-dir", str(tmp_path / "artifacts")])
    assert rc == 0
    import yaml

    loaded = yaml.safe_load(yaml_out.read_text(encoding="utf-8"))
    assert len(loaded["candidates"]) == len(CANDIDATES)
    assert "Acceptance" in md_out.read_text(encoding="utf-8")
